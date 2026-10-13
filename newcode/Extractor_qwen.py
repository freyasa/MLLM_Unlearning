from dataclasses import field, dataclass
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoProcessor,AutoTokenizer
from transformers import set_seed
from lib._json import load_json, save_to_json
from lib._pickle import load_from_pickle, save_to_pickle
from lib._batch_generate import batch_generating
from dataclasses import field, dataclass
from transformers import AutoTokenizer, AutoProcessor, AutoConfig
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
import logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def formulate_prompt_with_options(question, options):
    options_str = "\n".join([f"{key}: {value}" for key, value in options.items()])
    prompt = f"{question}\n{options_str}"
    return prompt

class EmbeddingExtractor:
    def __init__(self, model_name_or_path, device=None):
        self.device = device if device is not None \
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model from {model_name_or_path}")
        self.config = AutoConfig.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')
        self.num_layers = self.config.num_hidden_layers
        self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')
        self.processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')
        self.tokenizer.padding_side = "left"
        self.processor.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        ### edit here for the path of the LLaVA Vanilla model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                '/home/chenlding/MMUnlearner/Qwen_Vanilla',
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                low_cpu_mem_usage=True)
        logger.info(f"Model loaded successfully to {self.device}")
        logger.info(f"Number of layers: {self.num_layers}")
        
    def extract_embeddings(self, prompts, images,batch_size, layers):
        resid_pre_cache = {i: [] for i in layers}
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i+batch_size]
            batch_images = images[i:i+batch_size]
            batch_inputs = self.processor(
                images=batch_images,
                text=batch_prompts,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**batch_inputs, output_hidden_states=True)
            for layer_idx in layers:
                resid_pre_cache[layer_idx].append(
                    outputs.hidden_states[layer_idx][:, -1, :].detach().to('cpu')) 
            outputs = None
            torch.cuda.empty_cache()
            
        resid_pre_benign_embs = {
            layer: torch.cat(resid_pre_cache[layer], dim=0)
            for layer in layers}
        logger.info(f"resid_pre_benign_embs[{layers[0]}].shape: {resid_pre_benign_embs[layers[0]].shape}")
        
        H = torch.stack(list(resid_pre_benign_embs.values()), dim=1)
        logger.info(f"H's shape: {H.shape}")
        return H
    

    def extract_embeddings_text(self, prompts,batch_size, layers):
        resid_pre_cache = {i: [] for i in layers}
        
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i+batch_size]
            batch_inputs = self.tokenizer(
                text=batch_prompts,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**batch_inputs, output_hidden_states=True)
            for layer_idx in layers:
                resid_pre_cache[layer_idx].append(
                    outputs.hidden_states[layer_idx][:, -1, :].detach().to('cpu')) 
            outputs = None
            torch.cuda.empty_cache()
            
        resid_pre_benign_embs = {
            layer: torch.cat(resid_pre_cache[layer], dim=0)
            for layer in layers}
        logger.info(f"resid_pre_benign_embs[{layers[0]}].shape: {resid_pre_benign_embs[layers[0]].shape}")
        
        H = torch.stack(list(resid_pre_benign_embs.values()), dim=1)
        logger.info(f"H's shape: {H.shape}")
        return H
    
