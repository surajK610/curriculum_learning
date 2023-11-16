#!/usr/bin/env python3

'''
Takes raw text and saves BERT-cased features for that text to disk

Adapted from the BERT readme (and using the corresponding package) at

https://github.com/huggingface/pytorch-pretrained-BERT

###
John Hewitt, johnhew@stanford.edu
Feb 2019

'''
import torch
from transformers import AutoTokenizer, BertModel
from argparse import ArgumentParser
import h5py
import numpy as np
import os
from tqdm import tqdm


argp = ArgumentParser()
argp.add_argument('--input-path')
argp.add_argument('--output-path')
argp.add_argument("--seed", default=0, type=int, required=False)
argp.add_argument("--step", default=0, type=int, required=False)
argp.add_argument("--base", default="False", type=str, required=False)
argp.add_argument("--device", default=None, type=str, required=False)

args = argp.parse_args()
if args.device is None:
  device = "cuda" if torch.cuda.is_available() else "cpu"
else:
  device = args.device
    
# Load pre-trained model tokenizer (vocabulary)
# Crucially, do not do basic tokenization; PTB is tokenized. Just do wordpiece tokenization.
if args.base == "True":
  tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False)
  model = BertModel.from_pretrained("bert-base-cased").to(device)
else:
  tokenizer = AutoTokenizer.from_pretrained(f'google/multiberts-seed_{args.seed}-step_{args.step}k', use_fast=False)
  model = BertModel.from_pretrained(f"google/multiberts-seed_{args.seed}-step_{args.step}k").to(device)

model.config.output_hidden_states = True
model.eval()

LAYER_COUNT = 12 + 1 # +1 for embeddings
FEATURE_COUNT = 768

home = os.environ['LEARNING_DYNAMICS_HOME']

if args.base == "True":
  os.makedirs(os.path.join(home, f'data/structural-probes/embeddings/base/'), exist_ok=True)
  input_path = os.path.join(home, f'data/structural-probes/raw/{args.input_path}_raw.txt')
  output_path = os.path.join(home, f'data/structural-probes/embeddings/base/{args.input_path}.hdf5')
else:
  os.makedirs(os.path.join(home, f'data/structural-probes/embeddings/seed={args.seed}-step={args.step}k'), exist_ok=True)
  input_path = os.path.join(home, f'data/structural-probes/raw/{args.input_path}_raw.txt')
  output_path = os.path.join(home, f'data/structural-probes/embeddings/seed={args.seed}-step={args.step}k/{args.input_path}.hdf5')


with open(input_path, 'r', encoding='utf-8') as file:
  num_lines = sum(1 for _ in file)
    
with h5py.File(output_path, 'w') as fout:
  for index, line in enumerate(tqdm(open(input_path), total=num_lines)):
    line = line.strip() # Remove trailing characters
    line = '[CLS] ' + line + ' [SEP]'
    tokenized_text = tokenizer.wordpiece_tokenizer.tokenize(line)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    
    with torch.no_grad():
      encoded_layers = model(input_ids=torch.tensor([indexed_tokens]).to(device))
    
    dset = fout.create_dataset(str(index), (LAYER_COUNT, len(tokenized_text), FEATURE_COUNT))
    dset[:,:,:] = torch.vstack(encoded_layers.hidden_states).detach().cpu().numpy()