from dataclasses import field, dataclass
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers.models.llama.modeling_llama import FlashAttentionKwargs
import random
import json
from tqdm import tqdm
import torch
import numpy as np
from typing import Optional, Tuple, Union, List, Dict
from sklearn.decomposition import PCA
from transformers import set_seed
from lib._json import load_json, save_to_json
from lib._pickle import load_from_pickle, save_to_pickle
from lib._batch_generate import batch_generating
from typing import List, Optional, Union, Tuple
from transformers import LlamaModel
from transformers import LlavaForConditionalGeneration
from transformers.cache_utils import Cache
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.llama.configuration_llama import LlamaConfig
import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class AlphaLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, 
                layer_idx: int, 
                steering_matrix: Optional[torch.Tensor] = None, 
                strength: float = 0.0
                ):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        
        device = next(self.parameters()).device
        if steering_matrix is not None:
            self.steering_matrix = steering_matrix.to(device)
        else:
            self.steering_matrix = None
        self.strength = strength
        
    def set_steering_parameters(
        self, 
        steering_matrix: Optional[torch.Tensor]=None, 
        strength: float = 0.0,
        device: Optional[torch.device]=None):
        
        device = next(self.parameters()).device if device is None else device
        
        if steering_matrix is not None and torch.any(steering_matrix):
            self.steering_matrix = steering_matrix.to(device)
        self.strength = strength
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        # Ensure steering_matrix is on the same device as hidden_states
        if hidden_states.shape[1] > 1: # Only apply steering on initial input
            if self.steering_matrix is not None and torch.any(self.steering_matrix):
                # Only apply steering once during input processing
                if self.steering_matrix.device != hidden_states.device:
                    self.steering_matrix = self.steering_matrix.to(hidden_states.device)
                # Calculate steering vector by multiplying the last token's hidden state with the steering matrix
                steering_vector = self.strength * hidden_states[:, -1, :] @ self.steering_matrix 
                # Reshape to match hidden_states dimensions and move to the same device
                steering_vector = steering_vector.unsqueeze(1).to(hidden_states.device)
                # Apply steering by adding the steering vector to hidden states
                hidden_states = hidden_states + steering_vector
                
        residual = hidden_states # resid_pre - save for residual connection

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        
        hidden_states = residual + hidden_states
        residual = hidden_states # resid_mid - save after attention residual
        
        # Normalize hidden states after attention, then pass through MLP
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # resid_post - final residual connection
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs
    
class LlamaModel_for_Steering(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [AlphaLlamaDecoderLayer(
                config=config, 
                layer_idx=layer_idx,
            )
            for layer_idx in range(config.num_hidden_layers)]
        )

    def set_steering_parameters(
        self, 
        steering_matrix: Optional[torch.Tensor]=None, 
        strength: Optional[list[float]] = None,
        device: Optional[torch.device] = None):
        device = next(self.parameters()).device if device is None else device
        
        if steering_matrix is not None:
            steering_matrix = steering_matrix.to(device)
        
        for layer_idx, layer in enumerate(self.layers):
            layer_steering_matrix = None
            if steering_matrix is not None:
                layer_steering_matrix = steering_matrix[layer_idx]
                
            layer.set_steering_parameters(
                steering_matrix=layer_steering_matrix, 
                strength=strength[layer_idx] if strength is not None else 0.0
            )
            torch.cuda.empty_cache()
            
class LlavaForSteering(LlavaForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.language_model.model = LlamaModel_for_Steering(config.text_config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        self.post_init()
        
    def set_steering_parameters(
            self, 
            steering_matrix: Optional[torch.Tensor]=None, 
            strength: Optional[list[float]] = None):
        device = next(self.parameters()).device
        if steering_matrix is not None:
            steering_matrix = steering_matrix.to(device)
        self.language_model.model.set_steering_parameters(
            steering_matrix=steering_matrix, 
            strength=strength,
            device=device
        )