import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import inspect
import argparse
import numpy as np
import configparser
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoTokenizer
import random
import os
import json
from PIL import Image
import torch
import logging
import numpy as np
from tqdm import tqdm   
import pickle

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
refusal_dir = os.path.dirname(os.path.abspath(__file__))
vanilla_dir = os.path.join(root_dir, 'LLaVA_Vanilla')
data_dir = os.path.join(refusal_dir, 'test.jsonl')
anchor_list = ['en', 'zh']

processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
tokenizer = AutoTokenizer.from_pretrained('llava-hf/llava-1.5-7b-hf')
model = LlavaForConditionalGeneration.from_pretrained(
                vanilla_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                local_files_only=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("pad_token is None, set to eos_token")
model.max_length = tokenizer.model_max_length
model.eval()

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
            prompt = f"USER: <image>\n{raw_data}\nASSISTANT:"
            lang_data[lang].append(prompt)
for k, v in lang_data.items():
    print(f'Loaded {len(v)} {k} samples. \n')

lang_data['en'] = processor(images=attack_images, text=lang_data['en'] , return_tensors="pt", padding=True, add_special_tokens=False)
lang_data['zh'] = processor(images=before_images, text=lang_data['zh'] , return_tensors="pt", padding=True, add_special_tokens=False)

source_lan_emb = {}
for lang, data in tqdm(lang_data.items(), desc='Computing sentence embeddings'):
    input_ids = data.input_ids
    attention_mask = data.attention_mask
    batch_size = min(1, input_ids.size(0))
    pixel_values = data.pixel_values
    num_batches = input_ids.size(0) // batch_size
    sent_embs = []
    
    for i in range(num_batches):
        batch_input_ids = input_ids[i * batch_size: (i + 1) * batch_size]
        batch_pixel_values = pixel_values[i * batch_size: (i + 1) * batch_size]
        batch_attention_mask = attention_mask[i * batch_size: (i + 1) * batch_size]
        with torch.no_grad():
            outputs = model(pixel_values = batch_pixel_values.to(model.device), input_ids=batch_input_ids.to(model.device), attention_mask=batch_attention_mask.to(model.device), output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of len L tensors: (N, seq_len, D), N = batch_size
        del outputs
        hidden_states = hidden_states[1:]  # Remove the input layer embeddings
        hidden_states = torch.stack(hidden_states)  # (L, N, seq_len, D)
        last_layer_emb = hidden_states[-1]
        hidden_states[-1] = last_layer_emb
        # hidden_sent_embs = torch.mean(hidden_states, dim=2)  # (L, N, D)
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

file_path = os.path.join(refusal_dir, 'refusal_llava.pkl')
if not os.path.exists(os.path.dirname(file_path)):
    os.makedirs(os.path.dirname(file_path))
with open(file_path, 'wb') as f:
    pickle.dump(
    np.mean(temp_harmful.to(torch.float32).cpu().numpy(), axis=1) -
    np.mean(temp_harmless.to(torch.float32).cpu().numpy(), axis=1),
    f
)