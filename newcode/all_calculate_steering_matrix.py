import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
import torch
import numpy as np
from sklearn.decomposition import PCA
from transformers import set_seed
from lib._json import load_json, save_to_json
from lib._pickle import load_from_pickle, save_to_pickle
from lib._batch_generate import batch_generating
from dataclasses import field, dataclass
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
import pandas as pd
import pickle
import logging 
from argparse import ArgumentParser
from null_space_util import cal_tilde_delta_with_regularization_l, cal_steering_matrix_l, null_space_projection_l

parser = ArgumentParser()
parser.add_argument("--this_ratio", type=float, default=0.3)
parser.add_argument("--lambda_reg_para", type=float, default=0.1)
parser.add_argument("--model", type=str, default="llava")
parser.add_argument("--direction", type=str)
parser.add_argument("--forget_ratio", type=int, default=10)
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

refusal_direction_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'refusal_direction')
if args.model == "qwen":
    args.direction = os.path.join(refusal_direction_dir, f'refusal_qwen.pkl')
elif args.model == "llava":
    args.direction = os.path.join(refusal_direction_dir, f'refusal_llava.pkl')

### Load the dataset
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'MLLMU-Bench')
### define the layers and the ratio
this_ratio = args.this_ratio
lambda_reg_para = args.lambda_reg_para


if args.model == "qwen":    
    layers_ratio_list =  [(0, this_ratio),(1, this_ratio),(2, this_ratio),(3, this_ratio),(4, this_ratio),(5, this_ratio), (6, this_ratio), (7, this_ratio), (8, this_ratio), (9, this_ratio), (10, this_ratio), \
        (11, this_ratio), (12, this_ratio), (13, this_ratio), (14, this_ratio), (15, this_ratio), (16, this_ratio), (18, this_ratio), (19, this_ratio), (20, this_ratio), (21, this_ratio), (22, this_ratio), \
            (23, this_ratio), (24, this_ratio), (25, this_ratio), (26, this_ratio), (27, this_ratio)]
elif args.model == "llava":
    layers_ratio_list =  [(0, this_ratio),(1, this_ratio),(2, this_ratio),(3, this_ratio),(4, this_ratio),(5, this_ratio), (6, this_ratio), (7, this_ratio), (8, this_ratio), (9, this_ratio), (10, this_ratio), \
        (11, this_ratio), (12, this_ratio), (13, this_ratio), (14, this_ratio), (15, this_ratio), (16, this_ratio), (18, this_ratio), (19, this_ratio), (20, this_ratio), (21, this_ratio), (22, this_ratio), \
            (23, this_ratio), (24, this_ratio), (25, this_ratio), (26, this_ratio), (27, this_ratio),(28, this_ratio),(29, this_ratio),(30, this_ratio),(31, this_ratio)]
else:
    raise ValueError(f"Invalid model: {args.model}")

# load the embeddings
device = torch.device("cuda")
if args.model == "qwen":    
    embeds_dir = os.path.join(data_dir, 'embeddings', 'qwen')
elif args.model == "llava":
    embeds_dir = os.path.join(data_dir, 'embeddings', 'llava')
else:
    raise ValueError(f"Invalid model: {args.model}")

forget_embeds = torch.load(f"{embeds_dir}/forget_{args.forget_ratio}.pt", map_location=device)
retain_embeds = torch.load(f"{embeds_dir}/retain_{args.forget_ratio}.pt", map_location=device)

refusal_vectors_path = args.direction
print(f"refusal_vectors_path: {refusal_vectors_path}")
refusal_vectors = pickle.load(open(refusal_vectors_path, "rb"))

####### to be edit
refusal_vectors = torch.tensor(refusal_vectors, dtype=torch.float64).to(device)
logger.info("refusal vectors' shape: %s", refusal_vectors.shape)
num_layer = refusal_vectors.shape[0]
d_model = refusal_vectors.shape[1]

# initialize the steering matrix
P = torch.zeros(num_layer, d_model, d_model, device=device)
tilde_delta = torch.zeros(num_layer, d_model, d_model, device=device)
steering_matrix = torch.zeros(num_layer, d_model, d_model, device=device)

## calculate the steering matrix
for layer, ratio in tqdm(layers_ratio_list, desc="Calculating steering matrix"):
    logger.info(f"layer: {layer}, ratio: {ratio}")
    P_layer = null_space_projection_l(retain_embeds[:, layer, :], abs_nullspace_ratio=ratio)
    P[layer] = P_layer
    P_norm = torch.norm(P_layer)
    logger.info(f"P_norm: {P_norm}")
    tilde_delta_layer = cal_tilde_delta_with_regularization_l(forget_embeds[:, layer, :], \
        P_layer, refusal_vectors[layer], lambda_reg=lambda_reg_para, device=device)
    tilde_delta[layer] = tilde_delta_layer
    tilde_delta_norm = torch.norm(tilde_delta_layer)
    logger.info(f"tilde_delta_norm: {tilde_delta_norm}")
    
    steering_matrix_layer = cal_steering_matrix_l(P_layer, tilde_delta_layer, device=device)
    steering_matrix[layer] = steering_matrix_layer
    steering_matrix_norm = torch.norm(steering_matrix_layer)
    logger.info(f"steering matrix layer {layer} norm: {steering_matrix_norm}")


steering_matrix_path = os.path.join(data_dir, 'steering_matrix', f'{args.model}_{this_ratio}_{lambda_reg_para}_{args.forget_ratio}.pt')
os.makedirs(os.path.dirname(steering_matrix_path), exist_ok=True)
torch.save(steering_matrix, steering_matrix_path)
logger.info(f"steering matrix saved to {steering_matrix_path}")

# Clean up memory
forget_embeds = None; retain_embeds = None; refusal_vectors = None
P = None; tilde_delta = None; steering_matrix = None
torch.cuda.empty_cache()