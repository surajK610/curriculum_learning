'''
Modified from Neel Nanda's https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Head_Detector_Demo.ipynb#scrollTo=5ikyL8-S7u2Z
'''
#sfeature
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
import sys
import random

sys.path.append('/users/sanand14/data/sanand14/learning_dynamics/src/experiments/utils')
sys.path.append('/users/sanand14/data/sanand14/learning_dynamics/src/experiments')

from utils.probing_utils import AccuracyProbe
from utils.data_utils import makeHooks, decomposeHeads, decomposeSingleHead
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer

from transformers import BertModel, BertTokenizer
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

LAYER_COUNT = 13

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

def create_repeats_dataset(num_samples=50, min_vector_size=5, max_vector_size=50, min_num_repeats=5, max_num_repeats=20, max_vocab=PYTHIA_VOCAB_SIZE):
  """Creates a dataset for the experiment."""
  dataset = []
  for _ in range(num_samples):
    vector_size = torch.randint(min_vector_size, max_vector_size, (1,)).item()
    num_repeats = torch.randint(min_num_repeats, max_num_repeats, (1,)).item()
    tokens = torch.randint(0, max_vocab, (1, vector_size))
    tokens = tokens.repeat((1, num_repeats))
    dataset.append(tokens)
  return dataset

def generate_algo_task_dataset(model, num_samples, min_vector_size=3, max_vector_size=10, max_vocab=30, 
                                type : HeadName ="duplicate_token_head", device="cpu", resid=True):
    task_labels = []
    model.eval()
    model.config.output_hidden_states = True
    n_layers = model.config.num_hidden_layers
    if ~resid:
        cache = defaultdict(list)
        cache = makeHooks(model, cache, mlp=True, attn=True, embeddings=True, remove_batch_dim=False, device=device)
       
    relevant_activations = defaultdict(list)    
    for _ in tqdm(range(num_samples), desc=f'Generating activations for {type} dataset'):
        vector_size = torch.randint(min_vector_size, max_vector_size, (1,)).item()
        tokens = torch.randint(0, max_vocab, (1, vector_size))
        ## adds the CLS + SEP tokens
        if type == "duplicate_token_head":
          tokens = torch.randperm(max_vocab)[:vector_size].view(1, -1) # sample without replacement
          if random.random() < 0.5:
            idxs = torch.randperm(vector_size)[:2] # sample 2 indices
            tokens[:, idxs[0]] = tokens[:, idxs[1]] # duplicate
            idx = torch.tensor(idxs[0])
            label = torch.tensor([[1]])
          else:
            idx = torch.randint(0, vector_size, (1,)) 
            label = torch.tensor([[0]])
        elif type == "induction_head":
          mask_idx = torch.randint(1, vector_size, (1,)).item() # mask token
          tokens = tokens.repeat((1, 2)) # repeat twice
          tokens[:, vector_size + mask_idx] = 103 # replace second instance with MASK
          label = tokens[:, mask_idx] # label is the token that is masked
          # idx = torch.tensor([vector_size + mask_idx - 1]) # index of the token before the mask
          idx = torch.tensor([vector_size + mask_idx]) # index of the mask
        elif type == "previous_token_head":
          idx = torch.randint(1, vector_size, (1,)) # idx
          label = tokens[:, idx.item() - 1] # first token has no previous token
        else:
          raise ValueError(f"type must be one of {HEAD_NAMES}")
        tokens = torch.concat([torch.tensor([[101]]), tokens, torch.tensor([[102]])], axis=1)
        ## appends CLS + SEP tokens
        idx = idx + 1
        task_labels.append(label.to(device))
        with torch.no_grad():
          output = model(tokens.to(device))
          if ~resid:
            attn_outputs = cache['attention'].copy()
            mlp_outputs = cache['mlp'].copy()
            embedding_outputs = cache['embedding'].copy()
            
            cache['attention'].clear()
            cache['mlp'].clear()
            cache['embedding'].clear()
            relevant_activations[0].append(embedding_outputs[0].squeeze(0)[idx, :])
            ### embedding outputs
            for i in range(n_layers):
              relevant_activations[i+1].append(mlp_outputs[i].squeeze(0)[idx, :])
              ### mlp outputs
            for i in range(n_layers):
              relevant_activations[i+1+n_layers].append(attn_outputs[i].squeeze(0)[idx, :])
              ### attention outputs 
          else:
            cache = output.hidden_states
            for i, val in enumerate(cache):
              relevant_activations[i].append(val.squeeze(0)[idx, :])
    task_labels = torch.concat(task_labels, dim=0).unsqueeze(0).to(device)
    for i in relevant_activations:
        relevant_activations[i] = torch.vstack(relevant_activations[i])
    return relevant_activations, task_labels
  
def generate_ahead_token_dataset(model, num_samples, min_vector_size=3, max_vector_size=10, max_vocab=30, 
                                type : HeadName ="duplicate_token_head", device="cpu"):
    task_labels = []
    model.eval()
    relevant_activations = defaultdict(list)    
    for _ in tqdm(range(num_samples), desc=f'Generating activations for {type} dataset'):
        vector_size = torch.randint(min_vector_size, max_vector_size, (1,)).item()
        tokens = torch.randint(0, max_vocab, (1, vector_size))
        if type == "duplicate_token_head":
          labels = torch.tensor([[0] * vector_size + [1] * vector_size]) # no duplicate until repeat
          tokens = tokens.repeat((1, 2)) # repeat twice (maybe try repeat more in future)
          idxs = torch.arange(0, vector_size*2)
        elif type == "induction_head":
          labels = torch.roll(tokens, -1) # circular shift [0, 1, 2] -> [1, 2, 0]
          tokens = tokens.repeat((1, 2)) # repeat twice (maybe try repeat more in future)
          idxs = torch.arange(vector_size, vector_size*2)
        elif type == "previous_token_head":
          labels = tokens[:, :-1] # first token has no previous token
          idxs = torch.arange(1, vector_size) # no repeat for previous token
        else:
          raise ValueError(f"type must be one of {HEAD_NAMES}")
        
        task_labels.append(labels.to(device))
        with torch.no_grad():
          _, cache = model.run_with_cache(tokens.to(device))
          cache = cache.accumulated_resid()
          for i, val in enumerate(cache):
            relevant_activations[i].append(val.squeeze(0)[idxs, :])
    task_labels = torch.concat(task_labels, dim=1).to(device)
    for i in relevant_activations:
        relevant_activations[i] = torch.vstack(relevant_activations[i])
    return relevant_activations, task_labels

def main(FLAGS):
  if FLAGS.probe_residuals == "True":
    print("Probing Residuals", FLAGS.detection_pattern, FLAGS.checkpoint)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resid = FLAGS.resid == "True"
    # dict_df_heads = {detection_pattern: pd.DataFrame(columns=PYTHIA_CHECKPOINTS_OLD, index=range(N_LAYERS+1)) for detection_pattern in HEAD_NAMES}
    
    output_dir = "outputs/aheads"
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = FLAGS.checkpoint
    print("Number of Steps: ", checkpoint)
    # model = HookedTransformer.from_pretrained(MODEL, checkpoint_value=checkpoint, device=device)
    # save_model_name = MODEL.split("/")[-1] + "-step" + str(checkpoint)
    model = BertModel.from_pretrained(f"google/multiberts-seed_0-step_{checkpoint}k").to(device)
    save_model_name = f"google/multiberts-seed_0-step_{checkpoint}k".split("/")[-1]
    # for detection_pattern in HEAD_NAMES:
    detection_pattern = FLAGS.detection_pattern
    if FLAGS.make_dataset == "True":
      relevant_activations, task_labels = generate_algo_task_dataset(model, num_samples=40000, min_vector_size=8, max_vector_size=10, max_vocab=FLAGS.max_vocab, type=detection_pattern, device=device, resid=resid)
      torch.save(relevant_activations, os.path.join(output_dir, detection_pattern, f"relevant_activations{'_resid' if resid else ''}.pt"))
      torch.save(task_labels, os.path.join(output_dir, detection_pattern, f"task_labels{'_resid' if resid else ''}.pt"))
    else:
      relevant_activations = torch.load(os.path.join(output_dir, detection_pattern, f"relevant_activations{'_resid' if resid else ''}.pt"))
      task_labels = torch.load(os.path.join(output_dir, detection_pattern, f"task_labels{'_resid' if resid else ''}.pt"))
      
    num_labels = task_labels.max() + 1
    n_layers = model.config.num_hidden_layers
    # for i in range(n_layers+1):
    #   print("Training probe for layer", i)
      
    #   probe = AccuracyProbe(relevant_activations[0].shape[-1], num_labels, FLAGS.finetune_model).to(device)
    #   n_examples = relevant_activations[i].shape[0]
    #   train_len = int(0.8 * n_examples)
    #   perm = torch.randperm(n_examples)
    #   shuffled_data = relevant_activations[i].detach()[perm]
    #   shuffled_labels = task_labels.view(-1, 1)[perm]
      
    #   train_dataset = TensorDataset(shuffled_data[:train_len], shuffled_labels[:train_len])
    #   val_dataset = TensorDataset(shuffled_data[train_len:], shuffled_labels[train_len:])
    #   train_dataloader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True)
    #   val_dataloader = DataLoader(val_dataset, batch_size=FLAGS.batch_size, shuffle=False)
      
    #   trainer = Trainer(max_epochs=FLAGS.epochs) ## start w/ 1 epoch
    #   trainer.fit(probe, train_dataloader, val_dataloader)
    #   val_logs = trainer.validate(probe, val_dataloader)
      
    #   layer_str = "layer-" + str(i)
    #   os.makedirs(os.path.join(output_dir, detection_pattern, save_model_name, layer_str), exist_ok=True)
    #   with open(os.path.join(output_dir, detection_pattern, save_model_name, layer_str, f"val_acc{'' if resid else '_out'}.txt"), "w") as f:
    #     f.write(str(val_logs))
    #   print(val_logs)
    
    if ~resid:
      for i in range(n_layers):
        for attention_head in range(model.config.num_attention_heads):
          print(f"Training probe for layer {i}, head {attention_head}")
          head_activation_vectors = decomposeSingleHead(model, relevant_activations[i+n_layers+1].cpu().numpy(), i, attention_head)
          probe = AccuracyProbe(relevant_activations[0].shape[-1], num_labels, FLAGS.finetune_model).to(device)
          n_examples = head_activation_vectors.shape[0]
          train_len = int(0.8 * n_examples)
          perm = torch.randperm(n_examples)
          shuffled_data = torch.tensor(head_activation_vectors).to(device)[perm]
          shuffled_labels = task_labels.view(-1, 1)[perm]
          train_dataset = TensorDataset(shuffled_data[:train_len], shuffled_labels[:train_len])
          val_dataset = TensorDataset(shuffled_data[train_len:], shuffled_labels[train_len:])
          train_dataloader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True)
          val_dataloader = DataLoader(val_dataset, batch_size=FLAGS.batch_size, shuffle=False)
          trainer = Trainer(max_epochs=FLAGS.epochs) ## start w/ 1 epoch
          trainer.fit(probe, train_dataloader, val_dataloader)
          val_logs = trainer.validate(probe, val_dataloader)
          layer_str = "layer-" + str(i+1)
          os.makedirs(os.path.join(output_dir, detection_pattern, save_model_name, layer_str), exist_ok=True)
          with open(os.path.join(output_dir, detection_pattern, save_model_name, layer_str, f"val_acc_out{'_head_' + str(attention_head)}.txt"), "w") as f:
            f.write(str(val_logs))
          print(val_logs)
    

      # dict_df_heads[detection_pattern][checkpoint][i] = val_logs
                  
      # for detection_pattern in HEAD_NAMES:
      #   dir_out = os.path.join("outputs/aheads", detection_pattern)
      #   os.makedirs(dir_out, exist_ok=True)
      #   dict_df_heads[detection_pattern].to_csv(os.path.join(dir_out, f"probe_{detection_pattern}.csv"), sep="\t")
  else:
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
  parser.add_argument("--probe-residuals", default="False", type=str, help="probe residuals")
  parser.add_argument("--max-vocab", default=30, type=int, help="max vocab size")
  parser.add_argument("--finetune-model", default="linear", type=str, help="finetune model")
  parser.add_argument("--batch-size", default=32, type=int, help="batch size")
  parser.add_argument("--epochs", default=1, type=int, help="epochs")
  parser.add_argument("--resid", default="True", type=str, help="residuals") 
  parser.add_argument("--make-dataset", default="False", type=str, help="make dataset")
  
  parser.add_argument("--checkpoint", default=143000, type=int, help="checkpoint")
  parser.add_argument("--detection-pattern", default="previous_token_head", type=str, help="detection pattern")
  FLAGS = parser.parse_args()
  main(FLAGS)