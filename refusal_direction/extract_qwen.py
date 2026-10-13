import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import inspect
import argparse
import numpy as np
import configparser
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
import random
import json
import torch
import logging
import numpy as np  
import pickle
from PIL import Image
from io import BytesIO
from tqdm import tqdm

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
refusal_dir = os.path.dirname(os.path.abspath(__file__))
vanilla_dir = os.path.join(root_dir, 'Qwen_vanilla')
data_dir = os.path.join(refusal_dir, 'test.jsonl')

processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                vanilla_dir,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True)

processor.tokenizer.padding_side = "left"
tokenizer.padding_side = "left"
anchor_list = ['en', 'zh']

def load_images_from_folder(folder):
    images = []
    filenames = sorted(os.listdir(folder), key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else float('inf'))
    for filename in filenames:
        if filename.endswith((".png", ".jpg", ".JPEG", ".bmp", ".gif")):
            img = Image.open(os.path.join(folder, filename))
            if img is not None and int(filename.split('.')[0])<=16:
                images.append(img)
    return images

def load_images_from_folder_attack(folder):
    images = []
    filenames = sorted(os.listdir(folder), key=lambda x: int(x.split('_')[2]) if x.split('_')[2].isdigit() else float('inf'))
    for filename in filenames:
        if filename.endswith((".png", ".jpg", ".JPEG", ".bmp", ".gif")):
            img = Image.open(os.path.join(folder, filename))
            if img is not None and int(filename.split('_')[2]) <= 16:
                images.append(img)
    return images
    
attack_images = load_images_from_folder_attack(os.path.join(refusal_dir, 'constrain_16'))
before_images = load_images_from_folder(os.path.join(refusal_dir, 'add_images'))

lang_data = {lang: [] for lang in anchor_list}
with open(data_dir, 'r') as f:
    for line in f:
        for lang in anchor_list:
            raw_data = json.loads(line)[lang].strip()
            conversation = [    
                {
                    "role": "user",
                    "content": [
                        {"type": "image",},
                        {"type": "text", "text": f"{raw_data}"},
                    ],
                }
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            lang_data[lang].append(prompt.strip())

for k, v in lang_data.items():
    print(f'Loaded {len(v)} {k} samples. \n')

source_lan_emb = {}
for lang, data in tqdm(lang_data.items(), desc='Computing sentence embeddings'):
    batch_size = 1
    num_batches = 17
    sent_embs = []
    for i in range(num_batches):
        with torch.no_grad():
            if lang == 'en':
                input = processor(images=attack_images[i], text=lang_data['en'][i], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            else:
                input = processor(images=before_images[i], text=lang_data['zh'][i] , return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            outputs = model(**input, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of len L tensors: (N, seq_len, D), N = batch_size
        del outputs
        hidden_states = hidden_states[1:]  # Remove the input layer embeddings
        hidden_states = torch.stack(hidden_states)  # (L, N, seq_len, D)
        last_layer_emb = hidden_states[-1]
        hidden_states[-1] = last_layer_emb
        hidden_sent_embs = hidden_states[:, :, -1, :]
        sent_embs.append(hidden_sent_embs.detach().to('cpu'))
        del hidden_sent_embs, hidden_states
        torch.cuda.empty_cache()
    hidden_sent_embs = torch.cat(sent_embs, dim=1)  # (L, N, D)
    del sent_embs
    logging.info(f'Hidden sent: {hidden_sent_embs.shape}')
    torch.cuda.empty_cache()
    source_lan_emb[lang] = hidden_sent_embs
temp_harmful = source_lan_emb[anchor_list[0]]
temp_harmless = source_lan_emb[anchor_list[1]]

file_path = os.path.join(refusal_dir, 'refusal_qwen.pkl')
if not os.path.exists(os.path.dirname(file_path)):
    os.makedirs(os.path.dirname(file_path))
with open(file_path, 'wb') as f:
    pickle.dump(
    np.mean(temp_harmful.to(torch.float32).cpu().numpy(), axis=1) -
    np.mean(temp_harmless.to(torch.float32).cpu().numpy(), axis=1),
    f
)