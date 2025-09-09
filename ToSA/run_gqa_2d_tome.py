from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

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
    parser.add_argument("--image_aspect_ratio", type=str, default=None, help="Aspect ratio setting.")
    parser.add_argument("--r", type=float, default=0.1, help="Token merging ratio.")
    args = parser.parse_args()
    return args  

class llava_chatbot:
    def __init__(self, model, tokenizer, image_processor, max_length):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.img_tensors = None
    
    def update_images(self, images_list, image_masks=None):
        images = [Image.open(img_path).convert('RGB') for img_path in images_list]
        image_tensors = process_images(images, self.image_processor, self.model.config)
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
            image_masks = self.image_masks
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        return text_outputs[0]

if __name__ == "__main__":
    args = parse_args()
    ckpt_name = args.ckpt_name
    image_aspect_ratio = args.image_aspect_ratio
    token_merging_ratio = args.r
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
    
    chatbot = llava_chatbot(model, tokenizer, image_processor, max_length)

    with open('gqa/testdev_balanced_questions.json', 'r') as f:
        data = json.load(f)
        
    img_path_to_qa = defaultdict(list)
    
    for d in data.values():
        img_path = os.path.join('gqa/images', d['imageId'] + '.jpg')
        question = d['question']
        answer = d['answer']
        img_path_to_qa[img_path].append((question, answer))

    correct = 0
    incorrect = 0
    pbar = tqdm(range(len(img_path_to_qa)), desc="Processing", dynamic_ncols=True)
    
    results = []

    for i, img_path in zip(pbar, img_path_to_qa):
        for question, ans in img_path_to_qa[img_path]:
            chatbot.update_images([img_path])
            response = chatbot.generate_response(question)
            response = response.split(".")[0]
            if response.lower() in ans.lower():
                correct += 1
            else:
                incorrect += 1
            results.append({
                'img_path': img_path,
                'question': question,
                'response': response,
                'answer': ans,
            })
            import pdb; pdb.set_trace()
        
        acc = correct / (correct + incorrect)
        pbar.set_postfix(acc=f"{acc:.4f}")
    
    print("Accuacy: ", 100 * correct / (correct + incorrect))
    
    results.append({
                'accuracy': 100 * correct / (correct + incorrect),
            })
    
    token_merging_ratio = str(token_merging_ratio).replace(".", "")
    with open(f'gqa/results/gqa_2d_tome_r_{token_merging_ratio}.json', 'w') as f:
        json.dump(results, f, indent=4)