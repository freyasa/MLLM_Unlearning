from dataclasses import field, dataclass
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os
import torch
import numpy as np
import pandas as pd
import sys
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
import logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def null_space_l(A, min_null_space_ratio=0.1, abs_nullspace_ratio=0.0):
    A = A.double()
    _, S, Vh = torch.linalg.svd(A.T @ A)
    M, N = A.shape[0], A.shape[1]
    if abs_nullspace_ratio > 0:
        num = int(N * abs_nullspace_ratio)
    else:    
        S_ = torch.sqrt(S)
        rcond = torch.finfo(S.dtype).eps * max(M, N)
        tol = torch.amax(S_) * rcond
        num = torch.sum(S_ < tol)
        if num / N < min_null_space_ratio:
            num = int(N * min_null_space_ratio)
    print(f"final null space ratio: {num / N}")
    Q = Vh[-num:,:].T.conj()
    return Q

def null_space_projection_l(A, min_null_space_ratio=0.1, abs_nullspace_ratio=0.0):
    Q = null_space_l(A, min_null_space_ratio, abs_nullspace_ratio)
    P = Q @ Q.T
    print(f"error. null. space: {torch.norm(P@A.T.double())}")
    return P


def cal_tilde_delta_with_regularization_l(
    H_h_layer, P_layer, refusal_vector, lambda_reg, device="cuda"):
    # Convert all inputs to float32
    H_h_layer = H_h_layer.double()
    P_layer = P_layer.double()
    refusal_vector = refusal_vector.double()
    X = H_h_layer @ P_layer
    A = X.T @ X + lambda_reg * (P_layer.T @ P_layer)
    b = X.T @ refusal_vector.repeat(X.shape[0], 1)
    tilde_delta_layer = torch.linalg.pinv(A) @ b
    result = X @ tilde_delta_layer
    avg_reconstruction_error = torch.norm(result - refusal_vector) / X.shape[0]
    print(f"avg_reconstruction_error: {avg_reconstruction_error}", end="\t")
    # print(f"refusal_vector norm: {torch.norm(refusal_vector)}")
    return tilde_delta_layer

def cal_steering_matrix_l(P_layer, tilde_delta_layer, device="cuda"):
    P_layer = P_layer.to(device)
    tilde_delta_layer = tilde_delta_layer.to(device)
    steering_matrix_layer = P_layer @ tilde_delta_layer
    return steering_matrix_layer


def compute_bleu(ground_truth, predicted_answer):
    reference = [ground_truth.split()]  # Reference needs to be a list of tokenized words
    hypothesis = predicted_answer.split()  # Hypothesis (predicted answer) is also tokenized
    smoothing_function = SmoothingFunction().method1
    bleu_score = sentence_bleu(reference, hypothesis, smoothing_function=smoothing_function)
    return bleu_score

def formulate_prompt_with_options(question, options):
    options_str = "\n".join([f"{key}: {value}" for key, value in options.items()])
    prompt = f"{question}\n{options_str}"
    return prompt

def load_and_combine_parquet_files(directory):
    parquet_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.parquet')]
    combined_df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
    return combined_df