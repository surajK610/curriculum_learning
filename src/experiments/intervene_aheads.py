'''
Modified from Neel Nanda's https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Head_Detector_Demo.ipynb#scrollTo=5ikyL8-S7u2Z
'''

from collections import defaultdict
import logging
from typing import cast, Dict, List, Tuple, Union
from typing_extensions import get_args, Literal
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import argparse
import pandas as pd
from tqdm import tqdm
from functools import partial

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from aheads import create_repeats_dataset
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

def copy_attention_head(model1, model2, layer_idx, head_idx, dataset):
  model1.encoder.layers[layer_idx].self_attn.in_proj_weight.data[head_idx,:,:] = model2.encoder.layers[layer_idx].self_attn.in_proj_weight.data[head_idx,:,:]
  return perplexity(model1, dataset)
  
def perplexity(model, dataset):
  data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
  with torch.no_grad():
    for batch in data_loader:
      inputs, targets = batch
      outputs = model(inputs)
      loss = F.cross_entropy(outputs, targets, reduction='sum')
      total_loss += loss.item()
      average_loss = total_loss / len(data_loader.dataset)
    return torch.exp(torch.tensor(average_loss))

def hook_all_attention_patterns(
	attn_pattern,
	hook,
	alt_cache,
	layer,
	head_idx
):
  attn_pattern[:, head_idx, :, :]  = alt_cache[hook.name][:, head_idx, :, :]
  return attn_pattern

def attention_intervention_single_input(model, token_context, token_candidate, alt_cache, layer, head_idx):
  temp_hook_fn = partial(hook_all_attention_patterns, alt_cache = alt_cache, layer = layer, head_idx=head_idx)
  logits = model.run_with_hooks(torch.concat((token_context, token_candidate), dim = 1),
                 fwd_hooks=[(f'blocks.{layer}.attn.hook_pattern', temp_hook_fn)],
                 return_type="logits")
  return logits


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

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset-path", default=None, type=str, help="path where dataset is")
	parser.add_argument("--recompute", default="False", type=str, help="recompute dataset and save")
	FLAGS = parser.parse_args()
	main(FLAGS)