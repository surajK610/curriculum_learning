'''
Modified from Neel Nanda's https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Head_Detector_Demo.ipynb#scrollTo=5ikyL8-S7u2Z
'''

from collections import defaultdict
import logging
from typing import cast, Dict, List, Tuple, Union
from typing_extensions import get_args, Literal

import os
import numpy as np
import torch
import yaml
import argparse
import pandas as pd
from tqdm import tqdm

from transformer_lens import HookedTransformer
from transformer_lens.utils import is_square
from transformer_lens.head_detector import (compute_head_attention_similarity_score, 
											get_previous_token_head_detection_pattern, 
											get_duplicate_token_head_detection_pattern,
											get_induction_head_detection_pattern)

PYTHIA_VOCAB_SIZE = 50277 #50304
N_LAYERS=12
MODEL = "EleutherAI/pythia-160m"
PYTHIA_CHECKPOINTS_OLD = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(range(1000, 143000 + 1, 10000)) + [143000]
PYTHIA_CHECKPOINTS = [512] + list(range(1000, 10000 + 1, 1000))

HeadName = Literal["previous_token_head", "duplicate_token_head", "induction_head"]
HEAD_NAMES = cast(List[HeadName], get_args(HeadName))
ErrorMeasure = Literal["abs", "mul"]

LayerHeadTuple = Tuple[int, int]
LayerToHead = Dict[int, List[int]]

INVALID_HEAD_NAME_ERR = (
	f"detection_pattern must be a Tensor or one of head names: {HEAD_NAMES}; got %s"
)

SEQ_LEN_ERR = (
	"The sequence must be non-empty and must fit within the model's context window."
)

DET_PAT_NOT_SQUARE_ERR = "The detection pattern must be a lower triangular matrix of shape (sequence_length, sequence_length); sequence_length=%d; got detection patern of shape %s"

def detect_head_batch_res(model, tokens_list, detection_pattern):
	batch_scores = [detect_head_res(model, tokens, detection_pattern) for tokens in tqdm(tokens_list)]
	return torch.stack(batch_scores).mean(0)

def detect_head_res(
	model: HookedTransformer,
	tokens: torch.Tensor,
	detection_pattern: Union[torch.Tensor, HeadName],
	*,
	exclude_bos: bool = False,
	exclude_current_token: bool = False,
	error_measure: ErrorMeasure = "mul",
) -> torch.Tensor:
	"""
	1. `"mul"` (default) multiplies both tensors element-wise and divides the sum of the result by the sum of the attention pattern (should be 1s and 0s only)
	2. `"abs"` calculates the mean element-wise absolute difference between the detection pattern and the actual attention pattern.
	Returns a (n_layers, n_heads) Tensor representing the score for each attention head.
	"""
	cfg = model.cfg
	_, cache = model.run_with_cache(tokens, remove_batch_dim=True)
	layer2heads = {
		layer_i: list(range(cfg.n_heads)) for layer_i in range(cfg.n_layers)
	}
	detection_pattern = cast(torch.Tensor,
		eval(f"get_{detection_pattern}_detection_pattern(tokens.cpu())"),
		).to(cfg.device)
	matches = -torch.ones(cfg.n_layers, cfg.n_heads)
	for layer, layer_heads in layer2heads.items():
		# [n_heads q_pos k_pos]
		layer_attention_patterns = cache["pattern", layer, "attn"]
		for head in layer_heads:
			head_attention_pattern = layer_attention_patterns[head, :, :]
			head_score = compute_head_attention_similarity_score(
				head_attention_pattern,
				detection_pattern=detection_pattern,
				exclude_bos=exclude_bos,
				exclude_current_token=exclude_current_token,
				error_measure=error_measure,
			)
			matches[layer, head] = head_score
	return matches

def detect_head_batch(model, tokens_list, detection_pattern):
	batch_scores = [detect_head(model, tokens, detection_pattern) for tokens in tqdm(tokens_list)]
	return torch.stack(batch_scores).mean(0)
		
def detect_head(
	model: HookedTransformer,
	tokens: torch.Tensor,
	detection_pattern: Union[torch.Tensor, HeadName],
	*,
	exclude_bos: bool = False,
	exclude_current_token: bool = False,
	error_measure: ErrorMeasure = "mul",
) -> torch.Tensor:
	"""
	1. `"mul"` (default) multiplies both tensors element-wise and divides the sum of the result by the sum of the attention pattern (should be 1s and 0s only)
	2. `"abs"` calculates the mean element-wise absolute difference between the detection pattern and the actual attention pattern.
	Returns a (n_layers, n_heads) Tensor representing the score for each attention head.
	"""
	cfg = model.cfg
	_, cache = model.run_with_cache(tokens, remove_batch_dim=True)
	layer2heads = {
		layer_i: list(range(cfg.n_heads)) for layer_i in range(cfg.n_layers)
	}
	detection_pattern = cast(torch.Tensor,
		eval(f"get_{detection_pattern}_detection_pattern(tokens.cpu())"),
		).to(cfg.device)
	matches = -torch.ones(cfg.n_layers, cfg.n_heads)
	for layer, layer_heads in layer2heads.items():
		# [n_heads q_pos k_pos]
		layer_attention_patterns = cache["pattern", layer, "attn"]
		for head in layer_heads:
			head_attention_pattern = layer_attention_patterns[head, :, :]
			head_score = compute_head_attention_similarity_score(
				head_attention_pattern,
				detection_pattern=detection_pattern,
				exclude_bos=exclude_bos,
				exclude_current_token=exclude_current_token,
				error_measure=error_measure,
			)
			matches[layer, head] = head_score
	return matches

def create_repeats_dataset(num_samples=50, min_vector_size=5, max_vector_size=50, min_num_repeats=5, max_num_repeats=20):
	"""Creates a dataset for the experiment."""
	dataset = []
	for _ in range(num_samples):
		vector_size = torch.randint(min_vector_size, max_vector_size, (1,)).item()
		num_repeats = torch.randint(min_num_repeats, max_num_repeats, (1,)).item()
		tokens = torch.randint(0, PYTHIA_VOCAB_SIZE, (1, vector_size))
		tokens = tokens.repeat((1, num_repeats))
		dataset.append(tokens)
	return dataset

def main(FLAGS):
	if FLAGS.dataset_path is None:
		dataset = create_repeats_dataset()
	else:
		if FLAGS.recompute == "True":
			dataset = create_repeats_dataset()
			assert FLAGS.dataset_path is not None, "dataset path must be specified"
			torch.save(dataset, FLAGS.dataset_path)
		dataset = torch.load(FLAGS.dataset_path)
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	dict_df_heads_max = {detection_pattern: pd.DataFrame(columns=PYTHIA_CHECKPOINTS, index=range(N_LAYERS)) for detection_pattern in HEAD_NAMES}
	dict_df_heads_mean = {detection_pattern: pd.DataFrame(columns=PYTHIA_CHECKPOINTS, index=range(N_LAYERS)) for detection_pattern in HEAD_NAMES}
	
	for checkpoint in PYTHIA_CHECKPOINTS:
		print("Number of Steps: ", checkpoint)
		model = HookedTransformer.from_pretrained(MODEL, checkpoint_value=checkpoint, device=device)
		for detection_pattern in HEAD_NAMES:
			batch_head_scores = detect_head_batch(model, dataset, detection_pattern)
			max_per_layer = batch_head_scores.max(1).values
			mean_per_layer = batch_head_scores.mean(1)
			for i in range(model.cfg.n_layers):
				dict_df_heads_max[detection_pattern][checkpoint][i] = max_per_layer[i].item()
				dict_df_heads_mean[detection_pattern][checkpoint][i] = mean_per_layer[i].item()
				
	for detection_pattern in HEAD_NAMES:
		dir_out = os.path.join("outputs/aheads", detection_pattern)
		os.makedirs(dir_out, exist_ok=True)
		dict_df_heads_max[detection_pattern].to_csv(os.path.join(dir_out, f"max_{detection_pattern}_deeper.csv"), sep="\t")
		dict_df_heads_mean[detection_pattern].to_csv(os.path.join(dir_out, f"mean_{detection_pattern}_deeper.csv"), sep="\t")
		
def intervene_head(model, tokens, detection_pattern, layer, head, intervention_pattern, intervention_type="zero"):
	"""
	1. `"zero"` (default) sets the attention pattern to all zeros
	2. `"random"` sets the attention pattern to random values
	3. `"intervention_pattern"` sets the attention pattern to the intervention pattern
	"""
	cfg = model.cfg
	_, cache = model.run_with_cache(tokens, remove_batch_dim=True)
	layer2heads = {
		layer_i: list(range(cfg.n_heads)) for layer_i in range(cfg.n_layers)
	}
	detection_pattern = cast(torch.Tensor,
		eval(f"get_{detection_pattern}_detection_pattern(tokens.cpu())"),
		).to(cfg.device)
 
	matches = -torch.ones(cfg.n_layers, cfg.n_heads)
	for layer, layer_heads in layer2heads.items():
		# [n_heads q_pos k_pos]
		layer_attention_patterns = cache["pattern", layer, "attn"]
		for head in layer_heads:
			head_attention_pattern = layer_attention_patterns[head, :, :]
			head_score = compute_head_attention_similarity_score(
				head_attention_pattern,
				detection_pattern=detection_pattern,
				exclude_bos=exclude_bos,
				exclude_current_token=exclude_current_token,
				error_measure=error_measure,
			)
			matches[layer, head] = head_score
	return matches
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset-path", default=None, type=str, help="path where dataset is")
	parser.add_argument("--recompute", default="False", type=str, help="recompute dataset and save")
	FLAGS = parser.parse_args()
	main(FLAGS)