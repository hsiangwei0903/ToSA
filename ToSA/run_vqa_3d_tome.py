from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
from utils import string_to_number
import argparse
import copy
import torch

import warnings

import os
import os.path as osp
import json
from collections import defaultdict
from tqdm import tqdm
from positional_encodings.torch_encodings import PositionalEncoding1D

import numpy as np

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--ckpt_name", default='llava-onevision-qwen2-7b-ov', help="name to checkpoint.")
    parser.add_argument("--image_aspect_ratio", type=str, default=None, help="Aspect ratio setting.")
    parser.add_argument("--r", type=float, default=0.05, help="Token merging ratio.")
    parser.add_argument("--alpha_style", type=str, default='uniform', help="Alpha style.")
    args = parser.parse_args()
    return args  

class llava_chatbot:
    def __init__(self, model, tokenizer, image_processor, max_length, used_frame=1):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.img_tensors = None
        self.depth_tensors = None
        self.used_frame = used_frame
        pe = PositionalEncoding1D(channels=24)
        dummy_input = torch.zeros((1, 27, 24))
        output = pe(dummy_input).half()
        self.idx_to_pe = {idx: pe for idx, pe in enumerate(output[0])}
        self.gap = np.float32(256 / 27)  # Use float32 for calculations to improve consistency
        self.patch_size = 14
        self.num_patches = 384 // self.patch_size
        
        torch.manual_seed(42)
        np.random.seed(42)
    
    def get_pe(self, x, y, z):
        pe_x = self.idx_to_pe[x]
        pe_y = self.idx_to_pe[y]
        pe_z = self.idx_to_pe[z]
        return torch.cat((pe_x, pe_y, pe_z))
    
    def process_depth(self, depth):
        if isinstance(depth, str):
            img = Image.open(depth).convert("L")  # Ensure grayscale mode
        else:
            img = depth
        
        img = img.resize((384, 384), Image.BILINEAR)  # Use fixed interpolation mode
        img = np.array(img, dtype=np.float32)  # Use float32 before final conversion
        
        avg_depth = np.zeros((27, 27), dtype=np.int16)  # Store indices as int16
        pes = []

        for i in range(self.num_patches):
            for j in range(self.num_patches):
                patch = img[i * self.patch_size:(i + 1) * self.patch_size, j * self.patch_size:(j + 1) * self.patch_size]
                avg_depth[i, j] = np.round(np.mean(patch) / self.gap).astype(np.int16)  # Ensure consistent rounding
                if avg_depth[i, j] >= 27: # Ensure that the index is within the range
                    avg_depth[i, j] = 26
                idx = (i, j, avg_depth[i, j])
                pes.append(self.get_pe(*idx))
        
        return torch.stack(pes)
    
    def update_images(self, images_list, image_masks=None):
        images = [Image.open(img_path).convert("RGB") for img_path in images_list]
        image_tensors = process_images(images, self.image_processor, self.model.config)
        self.depth_tensors = torch.stack([self.process_depth(img_path.replace('/val2014/', '/val2014_depth/')) for img_path in images_list]).to(dtype=torch.float16, device=device)
        image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in image_tensors]
        self.img_tensors = image_tensors
        self.image_sizes = [image.size for image in images]
        self.image_masks = image_masks
    
    def generate_response(self, question, conv_template="qwen_1_5"):
        question = f"{DEFAULT_IMAGE_TOKEN}" + f"'{question}' Answer the question using a single word or phrase. \n\n"

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
            image_masks = self.image_masks,
            depth_states = self.depth_tensors
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        return text_outputs[0]

if __name__ == "__main__":
    args = parse_args()
    ckpt_name = args.ckpt_name
    image_aspect_ratio = args.image_aspect_ratio
    token_merging_ratio = args.r
    token_merging_ratio_str = str(token_merging_ratio).replace('.', '_')
    alpha_style = args.alpha_style
    pretrained = f"lmms-lab/{ckpt_name}"
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    llava_model_args = {"multimodal": True}
    overwrite_config = {}
    overwrite_config["image_aspect_ratio"] = image_aspect_ratio
    overwrite_config['tome'] = '3D'
    overwrite_config['visual_token_merge_ratio'] = token_merging_ratio
    overwrite_config['alpha'] = []
    
    for i in range(26): # number of ViT layers
        if alpha_style == 'uniform':
            # overwrite_config['alpha'].append(0.5)
            overwrite_config['alpha'].append(0.0)
        elif alpha_style == 'decrease':
            overwrite_config['alpha'].append(1-i/25)
        elif alpha_style == 'increase':
            overwrite_config['alpha'].append(i/25)
        else:
            raise ValueError("Invalid alpha style")
    
    llava_model_args["overwrite_config"] = overwrite_config
    
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, **llava_model_args, attn_implementation="eager")
    model.eval()
    
    chatbot = llava_chatbot(model, tokenizer, image_processor, max_length)

    with open('vqav2/count_qa.json', 'r') as f:
        data = json.load(f)

    correct = 0
    incorrect = 0
    pbar = tqdm(range(len(data)), desc="Processing", dynamic_ncols=True)
    
    results = []
    cur_count = 0

    for i, ins in zip(pbar, data):
        img_path = osp.join('vqav2/val2014/COCO_val2014_{:012d}.jpg'.format(ins['image_id']))
        question = ins['question']
        ans = ins['answer']
        chatbot.update_images([img_path])
        response = chatbot.generate_response(question)
        response = response.split(".")[0]
        response = string_to_number(response)
        if response.lower() == ans.lower():
            correct += 1
        else:
            incorrect += 1
        results.append({
            'img_path': img_path,
            'question': question,
            'response': response,
            'answer': ans,
        })
    
        acc = correct / (correct + incorrect)
        pbar.set_postfix(acc=f"{acc:.4f}")
    
    print("Accuacy: ", 100 * correct / (correct + incorrect))
    
    results.append({
                'accuracy': 100 * correct / (correct + incorrect),
            })
    
    token_merging_ratio = str(token_merging_ratio).replace(".", "")
    
    # with open(f'vqav2/results/vqa_3d_tome_r_{token_merging_ratio}_{alpha_style}.json', 'w') as f:
    #         json.dump(results, f, indent=4)
            
    if alpha_style == 'uniform':
        with open(f'vqav2/results/vqa_3d_tome_r_{token_merging_ratio}_{alpha_style}_full_spatial.json', 'w') as f:
            json.dump(results, f, indent=4)
    # else:
    #     with open(f'vqa/results/gqa_3d_tome_r_{token_merging_ratio}_{alpha_style}.json', 'w') as f:
    #         json.dump(results, f, indent=4)