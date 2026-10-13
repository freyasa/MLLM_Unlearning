from dataclasses import field, dataclass
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from io import BytesIO
import json
import random
from tqdm import tqdm
import torch
import numpy as np
from sklearn.decomposition import PCA
from transformers import set_seed
from lib._json import load_json, save_to_json
from lib._pickle import load_from_pickle, save_to_pickle
from lib._batch_generate import batch_generating
from dataclasses import field, dataclass
from transformers import AutoProcessor, AutoTokenizer
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
import pandas as pd
from Extractor_qwen import EmbeddingExtractor,formulate_prompt_with_options
import logging 
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="forget", choices=["forget","retain"])
parser.add_argument("--forget_ratio", type=int, default=5)
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

## Load the dataset
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'MLLMU-Bench')
if args.forget_ratio == 5:
    forget_df = pd.read_parquet(os.path.join(data_dir, 'forget_5', 'train-00000-of-00001.parquet'))
    retain_df = pd.read_parquet(os.path.join(data_dir, 'retain_95', 'train-00000-of-00001.parquet'))
    cele_df = pd.read_parquet(os.path.join(data_dir, 'Retain_Set', 'train-00000-of-00001.parquet'))
elif args.forget_ratio == 10:
    forget_df = pd.read_parquet(os.path.join(data_dir, 'forget_10', 'train-00000-of-00001.parquet'))
    retain_df = pd.read_parquet(os.path.join(data_dir, 'retain_90', 'train-00000-of-00001.parquet'))
    cele_df = pd.read_parquet(os.path.join(data_dir, 'Retain_Set', 'train-00000-of-00001.parquet'))

## Load the dataset5
if args.dataset == "forget":
    df = forget_df
    id_list = df['ID'].unique().tolist()
    eval_samples = df[df['ID'].isin(id_list)]
elif args.dataset == "retain":
    retain_df_id_list = retain_df['ID'].unique().tolist()
    retain_eval_samples = retain_df[retain_df['ID'].isin(retain_df_id_list)]
    cele_df_id_list = cele_df['ID'].unique().tolist()
    cele_eval_samples = cele_df[cele_df['ID'].isin(cele_df_id_list)]

## Define the list
generation_questions_list = []
generation_image_list = []
classification_questions_list = []
classification_image_list = []
cloze_questions_list = []
cloze_image_list = []

if args.dataset == "forget":
    logger.info(f"len(eval_samples): {len(eval_samples)}")
else:
    logger.info(f"len(cele_eval_samples): {len(cele_eval_samples)}")
    logger.info(f"len(retain_eval_samples): {len(retain_eval_samples)}")

## Extract the embeddings
### For classification task, without few-shot AND with Note: If you do not know the answer
processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')
processor.tokenizer.padding_side = "left"
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')
tokenizer.padding_side = "left"

if args.dataset == "retain":
    for _, row in tqdm(retain_eval_samples.iterrows(), total=len(retain_eval_samples)):
        image_data = row["image"]["bytes"]
        image = Image.open(BytesIO(image_data)).convert("RGB")
        for idx, question_data in enumerate(row['Classification_Task'].get("Image_Textual_Questions", [])):
            question_with_options = formulate_prompt_with_options(question_data["Question"], question_data["Options"])
            conversation = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                },
                                {"type": "text", "text": f"{question_with_options}\nJust give ONE letter representing the answer directly."},
                            ],
                        }
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            classification_questions_list.append(prompt.strip())
            classification_image_list.append(image)
        for task in row["Generation_Task"]:
            if task["Type"] == 'Image_Textual':
                question = task["Question"]
                conversation = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                },
                                {"type": "text", "text": f"{question}\nAnswer the question based on your trained knowledge in one sentence in ENGLISH."},
                            ],
                        }
                    ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                generation_questions_list.append(prompt.strip())
                generation_image_list.append(image)
        for task in row["Mask_Task"]:
            if task["Type"] == 'Image_Textual':
                question = task["Question"]
                question = question.replace("__", "[Blank]") + "\nPlease **ONLY** provide the correct answer that should replace the [Blank]."
                conversation = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                },
                                {"type": "text", "text": f"{question}"},
                            ],
                        }
                    ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                cloze_questions_list.append(prompt.strip())
                cloze_image_list.append(image)
    for _, row in tqdm(cele_eval_samples.iterrows(), total=len(cele_eval_samples)):
        image_data = row["image"]["bytes"]
        image = Image.open(BytesIO(image_data)).convert("RGB")
        for idx, question_data in enumerate(row['Classification_Task'].get("Image_Textual_Questions", [])):
            question_with_options = formulate_prompt_with_options(question_data["Question"], question_data["Options"])
            conversation = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                },
                                {"type": "text", "text": f"{question_with_options}\nJust give ONE letter representing the answer directly."},
                            ],
                        }
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            classification_questions_list.append(prompt.strip())
            classification_image_list.append(image)
        for task in row["Generation_Task"]:
            if task["Type"] == 'Image_Textual':
                question = task["Question"]
                conversation = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                },
                                {"type": "text", "text": f"{question}\nAnswer the question based on your trained knowledge in one sentence in ENGLISH."},
                            ],
                        }
                    ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                generation_questions_list.append(prompt.strip())
                generation_image_list.append(image)
        for task in row["Mask_Task"]:
            if task["Type"] == 'Image_Textual':
                question = task["Question"]
                question = question.replace("__", "[Blank]") + "\nPlease **ONLY** provide the correct answer that should replace the [Blank]."
                conversation = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                },
                                {"type": "text", "text": f"{question}"},
                            ],
                        }
                    ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                cloze_questions_list.append(prompt.strip())
                cloze_image_list.append(image)
    all_prompt = classification_questions_list + generation_questions_list + cloze_questions_list
    all_image = classification_image_list + generation_image_list + cloze_image_list
else:
    for _, row in tqdm(eval_samples.iterrows(), total=len(eval_samples)):
        image_data = row["image"]["bytes"]
        image = Image.open(BytesIO(image_data)).convert("RGB")
        for idx, question_data in enumerate(row['Classification_Task'].get("Image_Textual_Questions", [])):
            question_with_options = formulate_prompt_with_options(question_data["Question"], question_data["Options"])
            conversation = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                },
                                {"type": "text", "text": f"{question_with_options}\nJust give ONE letter representing the answer directly."},
                            ],
                        }
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            classification_questions_list.append(prompt.strip())
            classification_image_list.append(image)
        for task in row["Generation_Task"]:
            if task["Type"] == 'Image_Textual':
                question = task["Question"]
                conversation = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                },
                                {"type": "text", "text": f"{question}\nAnswer the question based on your trained knowledge in one sentence in ENGLISH."},
                            ],
                        }
                    ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                generation_questions_list.append(prompt.strip())
                generation_image_list.append(image)
        for task in row["Mask_Task"]:
            if task["Type"] == 'Image_Textual':
                question = task["Question"]
                question = question.replace("__", "[Blank]") + "\nPlease **ONLY** provide the correct answer that should replace the [Blank]."
                conversation = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                },
                                {"type": "text", "text": f"{question}"},
                            ],
                        }
                ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                cloze_questions_list.append(prompt.strip())
                cloze_image_list.append(image)
    all_prompt = classification_questions_list + generation_questions_list + cloze_questions_list
    all_image = classification_image_list + generation_image_list + cloze_image_list
        
# shuffle the data
combined = list(zip(all_prompt, all_image))
random.shuffle(combined)
all_prompt, all_image = zip(*combined)
all_prompt = list(all_prompt)
all_image = list(all_image)

logger.info(f"len(all_prompt): {len(all_prompt)}")
logger.info(f"len(all_image): {len(all_image)}")
# define the extractor
extractor = EmbeddingExtractor('Qwen/Qwen2.5-VL-7B-Instruct', device='cuda')
layers = list(range(extractor.num_layers))
embeddings = extractor.extract_embeddings(
        images=all_image,
        prompts=all_prompt,
        batch_size=4,
        layers=layers
    )

output_dir = f"{data_dir}/embeddings/qwen/{args.dataset}_{args.forget_ratio}.pt"
os.makedirs(os.path.dirname(output_dir), exist_ok=True)
torch.save(embeddings, output_dir)
logger.info(f"Embeddings saved to {output_dir}")




