from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from positional_encodings.torch_encodings import PositionalEncoding1D
import numpy as np

from PIL import Image
import argparse
import copy
import torch

import warnings

import os
import os.path as osp
import json
from collections import defaultdict
from tqdm import tqdm

import random

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--ckpt_name", default='llava-onevision-qwen2-7b-ov', help="name to checkpoint.")
    parser.add_argument("--used_frame", type=int, default=12, help="Number of frames to use.")
    parser.add_argument("--image_aspect_ratio", type=str, default=None, help="Aspect ratio setting.")
    parser.add_argument("--sample_style", type=int, default=1, help="Sampling style.")
    parser.add_argument("--r", type=float, default=1.0, help="Token merging ratio.")
    args = parser.parse_args()
    return args

def sample_frames(total_frames, num_samples=20, sample_style=0):
    if sample_style == 0:
        if total_frames < num_samples:
            return list(range(total_frames))
        sampled_frames = random.sample(range(total_frames), num_samples)
        return sampled_frames
    elif sample_style == 1:
        if total_frames < num_samples:
            return list(range(total_frames))
        interval = total_frames / num_samples
        sampled_frames = [int(i * interval) for i in range(num_samples)]
        sampled_frames = [min(f, total_frames - 1) for f in sampled_frames]
        return sampled_frames
    elif sample_style == 2:
        if total_frames < num_samples:
            return list(range(total_frames))
        interval = total_frames // (num_samples+1)
        sampled_frames = [int(i * interval) for i in range(1,num_samples+1)]
        sampled_frames = [min(f, total_frames - 1) for f in sampled_frames]
        return sampled_frames
    else:
        raise ValueError("Invalid sample style")    

class llava_chatbot:
    def __init__(self, model, tokenizer, image_processor, max_length, used_frame):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.img_tensors = None
        self.used_frame = used_frame
        pe = PositionalEncoding1D(channels=24)
        dummy_input = torch.zeros((1, 27, 24))
        output = pe(dummy_input)
        self.idx_to_pe = {idx: pe for idx, pe in enumerate(output[0])}
        self.gap = 256/27
        self.patch_size = 14
        self.num_patches = 384//self.patch_size
    
    def get_pe(self, x, y, z):
        pe_x = self.idx_to_pe[x]
        pe_y = self.idx_to_pe[y]
        pe_z = self.idx_to_pe[z]
        return torch.cat((pe_x, pe_y, pe_z))
    
    def update_images(self, images_list, image_masks=None):
        images = [Image.open(img_path) for img_path in images_list]
        image_tensors = process_images(images, self.image_processor, self.model.config)
        image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in image_tensors]
        self.img_tensors = image_tensors
        self.image_sizes = [image.size for image in images]
        self.image_masks = image_masks
    
    def generate_response(self, question, conv_template="qwen_1_5"):
        question = f"{DEFAULT_IMAGE_TOKEN}"*self.used_frame + f"{question} \n\n"

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        
        cont = self.model.generate(
            input_ids,
            images=self.img_tensors,
            image_sizes=self.image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=1000,
            image_masks = self.image_masks
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        return text_outputs[0]

if __name__ == "__main__":
    args = parse_args()
    used_frame = args.used_frame
    ckpt_name = args.ckpt_name
    sample_style = args.sample_style
    image_aspect_ratio = args.image_aspect_ratio
    token_merging_ratio = args.r
    token_merging_ratio_str = str(token_merging_ratio).replace('.', '_')
    output_path = f'../openeqa/2025_tome/2d_tome_{ckpt_name}_{token_merging_ratio_str}.json'
    pretrained = f"lmms-lab/{ckpt_name}"
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    llava_model_args = {"multimodal": True}
    overwrite_config = {}
    overwrite_config["image_aspect_ratio"] = image_aspect_ratio
    overwrite_config['tome'] = '2D'
    overwrite_config['visual_token_merge_ratio'] = token_merging_ratio
    
    llava_model_args["overwrite_config"] = overwrite_config
    
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, **llava_model_args, attn_implementation="eager")
    model.eval()
    
    chatbot = llava_chatbot(model, tokenizer, image_processor, max_length, used_frame)
    
    print("output_path:", output_path)
    
    with open('../openeqa/open-eqa-v0.json') as f:
        questions = json.load(f)

    scenes = [scene for scene in os.listdir('../openeqa/scannet-v0')] + [scene for scene in os.listdir('../openeqa/hm3d-v0')]
    
    scene_to_qid = defaultdict(list)
    for q_id, question in enumerate(questions):
        if question['episode_history'].split('/')[-1] not in scenes:
            continue
        scene_to_qid[question['episode_history'].split('/')[-1]].append(q_id)
    
    predictions = []
    
    # if osp.exists(output_path):
    #     with open(output_path) as f:
    #         predictions = json.load(f)
    
    seen_qids = set([pred['question_id'] for pred in predictions])

    for scene_name in tqdm(scene_to_qid):

        video_path = os.path.join('../openeqa/scannet-v0', scene_name)
        if not os.path.exists(video_path):
            video_path = os.path.join('../openeqa/hm3d-v0', scene_name)
        frame_files = sorted([file for file in os.listdir(video_path) if file.endswith('rgb.png')])
        video_frames_idx = sample_frames(len(frame_files), used_frame, sample_style=sample_style)
        image_files = sorted([frame_files[i] for i in video_frames_idx])
        chatbot.update_images([os.path.join(video_path, img) for img in image_files])
        for question_id in scene_to_qid[scene_name]:
            if questions[question_id]['question_id'] in seen_qids:
                continue
            question = questions[question_id]['question']
            output = chatbot.generate_response(question)
            predictions.append({
                'question_id': questions[question_id]['question_id'],
                'gt': questions[question_id]['answer'],
                'answer': output
            })
            
            with open(output_path, 'w') as f:
                json.dump(predictions, f, indent=4)