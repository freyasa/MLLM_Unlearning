import dataclasses
import os, json
import typing
import warnings
from typing import List, Literal, Tuple, Optional

import numpy as np
from sklearn.decomposition import PCA
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import matplotlib.pyplot as plt
from tqdm import tqdm

from lixiaolong.malleable_model import MalleableModel
from lixiaolong.sdataset import ContrastivePair
from lixiaolong.sdataset import SteeringDataset

def get_model_layer_list(model) -> torch.nn.ModuleList:
    if isinstance(model, MalleableModel):
        model = model.model  
    if hasattr(model.language_model, "model"):  # mistral-like
        return model.language_model.model.layers

@dataclasses.dataclass
class SteeringVector:
    model_type: str
    directions: dict[int, np.ndarray]
    explained_variances: dict

    @classmethod
    def train(cls, model, tokenizer, steering_dataset: SteeringDataset, **kwargs) -> "SteeringVector":
        tokenizer.pad_token_id = 0
        dirs, variances = read_representations(
            model,
            tokenizer,
            steering_dataset.formatted_dataset,
            suffixes=steering_dataset.suffixes,
            **kwargs,
        )
        return cls(model_type=model.config.model_type, 
                   directions=dirs, 
                   explained_variances=variances)
    
    def save(self, file_path: str):
        if not file_path.endswith('.svec'):
            file_path += '.svec'
        
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        data = {
            "model_type": self.model_type,
            "directions": {k: v.tolist() for k, v in self.directions.items()},
            "explained_variances": self.explained_variances
        }
        with open(file_path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, file_path: str) -> "SteeringVector":
        if not file_path.endswith('.svec'):
            file_path += '.svec'
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        directions = {int(k): np.array(v) for k, v in data["directions"].items()}
        explained_variances = {int(k): v for k, v in data["explained_variances"].items()}
        return cls(model_type=data["model_type"], 
               directions=directions, 
               explained_variances=explained_variances)


def read_representations(model, tokenizer, inputs: list[ContrastivePair], hidden_layer_ids: typing.Iterable[int] | None = None, batch_size: int = 32, method: typing.Literal["pca_diff", "pca_center"] = "pca_center", save_analysis: bool = False, output_dir: str = "activation_steering_figures", accumulate_last_x_tokens: typing.Union[int, str] = 1, suffixes: typing.List[typing.Tuple[str, str]] = None) -> dict[int, np.ndarray]:
    if hidden_layer_ids is None:
        hidden_layer_ids = range(model.language_model.model.config.num_hidden_layers)
    n_layers = len(get_model_layer_list(model))
    print(f"n_layers: {n_layers}")
    hidden_layer_ids = [i if i >= 0 else n_layers + i for i in hidden_layer_ids]
    train_strs = [s for ex in inputs for s in (ex.positive, ex.negative)]
    
    # Call the batched_get_hiddens function to get the hidden states for each specified layer
    layer_hiddens = batched_get_hiddens(
        model, tokenizer, train_strs, hidden_layer_ids, batch_size, accumulate_last_x_tokens, suffixes
    )
    # Initialize an empty dictionary to store the directions for each layer
    directions: dict[int, np.ndarray] = {}
    explained_variances: dict[int, float] = {}
    
    # Iterate over each specified layer
    for layer in tqdm(hidden_layer_ids):
        # Retrieve the hidden states for the current layer
        h = layer_hiddens[layer]
        if method == "pca_diff":
            # Calculate the difference between positive and negative examples
            train = h[::2] - h[1::2]
        elif method == "pca_center":
            # Calculate the center of positive and negative examples
            center = (h[::2] + h[1::2]) / 2
            train = h
            
            # Subtract the center from the examples
            train[::2] -= center
            train[1::2] -= center
        else:
            raise ValueError("unknown method " + method)
        
        # Perform PCA with 1 component on the training data to extract the direction vector
        pca_model = PCA(n_components=1, whiten=False).fit(train)
        directions[layer] = pca_model.components_.astype(np.float32).squeeze(axis=0)
        explained_variances[layer] = pca_model.explained_variance_ratio_[0]
        
        # Project the hidden states onto the direction vector
        projected_hiddens = project_onto_direction(h, directions[layer])
        
        # Calculate the mean of positive examples being smaller than negative examples
        positive_smaller_mean = np.mean(
            [
                projected_hiddens[i] < projected_hiddens[i + 1]
                for i in range(0, len(inputs) * 2, 2)
            ]
        )
        # Calculate the mean of positive examples being larger than negative examples
        positive_larger_mean = np.mean(
            [
                projected_hiddens[i] > projected_hiddens[i + 1]
                for i in range(0, len(inputs) * 2, 2)
            ]
        )
        
        # If positive examples are smaller on average, flip the direction vector
        if positive_smaller_mean > positive_larger_mean:
            directions[layer] *= -1
    
    # Return the dictionary mapping layer IDs to their corresponding direction vectors
    return directions, explained_variances


def batched_get_hiddens(model, tokenizer, inputs: list[str], hidden_layer_ids: list[int],batch_size: int, accumulate_last_x_tokens: typing.Union[int, str] = 1, suffixes: typing.List[typing.Tuple[str, str]] = None) -> dict[int, np.ndarray]:
    # Split the input strings into batches based on the specified batch size
    batched_inputs = [
        inputs[p : p + batch_size] for p in range(0, len(inputs), batch_size)
    ]
    
    hidden_states = {layer: [] for layer in hidden_layer_ids}
    
    with torch.no_grad():
        for batch in tqdm(batched_inputs):
            out = model(
                **tokenizer(batch, padding=True, return_tensors="pt").to(model.device),
                output_hidden_states=True,
            )
            
            # Iterate over each specified layer ID
            for layer_id in hidden_layer_ids:
                # Adjust the layer index if it is negative
                hidden_idx = layer_id + 1 if layer_id >= 0 else layer_id
                
                # Iterate over each batch of hidden states
                for i, batch_hidden in enumerate(out.hidden_states[hidden_idx]):
                    if accumulate_last_x_tokens == "all":
                        accumulated_hidden_state = torch.mean(batch_hidden, dim=0)
                    elif accumulate_last_x_tokens == "suffix-only":
                        if suffixes:
                            # Tokenize the suffix
                            suffix_tokens = tokenizer.encode(suffixes[0][0], add_special_tokens=False)
                            # Get the hidden states for the suffix tokens
                            suffix_hidden = batch_hidden[-len(suffix_tokens):, :]
                            accumulated_hidden_state = torch.mean(suffix_hidden, dim=0)
                        else:
                            warnings.warn("'suffix-only' option used but no suffixes provided. Using last token instead.")
                            accumulated_hidden_state = batch_hidden[-1, :]
                    else:
                        accumulated_hidden_state = torch.mean(batch_hidden[-accumulate_last_x_tokens:, :], dim=0)
                    
                    hidden_states[layer_id].append(accumulated_hidden_state.squeeze().cpu().numpy())
            
            del out
    
    return {k: np.vstack(v) for k, v in hidden_states.items()}


def project_onto_direction(H, direction):

    mag = np.linalg.norm(direction)
    
    # Assert that the magnitude is not infinite to ensure validity
    assert not np.isinf(mag)
    
    # Perform the projection by multiplying the matrix H with the direction vector
    # Divide the result by the magnitude of the direction vector to normalize the projection
    return (H @ direction) / mag


@dataclasses.dataclass
class ContrastivePair:
    """
    A dataclass representing a pair of contrasting strings.

    Attributes:
        positive: The positive string in the pair.
        negative: The negative string in the pair.
    """
    positive: str
    negative: str

class SteeringDataset:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        examples: List,
        suffixes: List[Tuple[str, str]] = None,
    ):
        self.tokenizer = tokenizer
        self.suffixes = suffixes
        self.formatted_dataset = []
        self.formatted_dataset_pre_populated = []

        for example in examples:
            message_a = f"USER: {example[0]}"
            message_b = f"USER: {example[1]}"
            positive = message_a
            negative = message_b

            self.formatted_dataset_pre_populated.append(
                ContrastivePair(positive=positive, negative=negative)
            )

        if suffixes is not None:
            for positive_suffix, negative_suffix in suffixes:
                for pair in self.formatted_dataset_pre_populated:
                    self.formatted_dataset.append(
                        ContrastivePair(
                            positive=pair.positive + "\nASSISTANT: " + positive_suffix,
                            negative=pair.negative + "\nASSISTANT: " + negative_suffix
                        )
                    )


