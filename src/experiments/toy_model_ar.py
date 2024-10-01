from collections import defaultdict
import logging
from typing import cast, Dict, List, Tuple, Union, Callable
from typing_extensions import get_args, Literal
import sys
import os
import numpy as np
import random 
import seaborn as sns
from PIL import Image
import re
from scipy.stats import spearmanr
from matplotlib.legend_handler import HandlerLine2D
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

import yaml
import argparse
import pandas as pd
from tqdm.notebook import tqdm
from functools import partial
from torch.utils.data import DataLoader, Dataset

from torch.utils.data import DataLoader, TensorDataset

sys.path.append('/users/sanand14/data/sanand14/learning_dynamics/src/experiments/utils')
sys.path.append('/users/sanand14/data/sanand14/learning_dynamics/src/experiments')

from aheads import create_repeats_dataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
from transformers import BertConfig, BertForMaskedLM, AdamW

import matplotlib.style as style
style.use('ggplot')

from toy_model import TrainingPipeline
from utils.toy_utils import bin_train_loop, create_dataloaders_bin, Probe, POSVocabGenerator, plot_pca_embeddings
from dataclasses import dataclass, field

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Callable
from tqdm import tqdm
from utils.toy_utils_ar import ICLVocabGenerator, CustomSequenceDataset
from utils.forgetting_utils import AdamEF

def evaluate_in_context(model, dataloader, device, eval_label_tokens):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    label_token_to_label = {v: k for k, v in eval_label_tokens.items()}  # Map label tokens to labels 0 or 1
    label_token_ids = torch.tensor([eval_label_tokens[0], eval_label_tokens[1]]).to(device)
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # Contains the query labels at the appropriate positions
            # print(inputs, labels)

            # Forward pass
            outputs = model(inputs, attention_mask=attention_mask)
            logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

            # Get the positions where labels are not -100 (i.e., the query label positions)
            label_mask = labels != -100  # Shape: (batch_size, seq_len)

            if label_mask.any():
                batch_size, seq_len, vocab_size = logits.size()
                label_positions = label_mask.nonzero(as_tuple=False)
                for idx in range(label_positions.size(0)):
                    batch_idx, pos = label_positions[idx]
                    logits_at_pos = logits[batch_idx, pos, :] 
                    logits_for_labels = logits_at_pos[label_token_ids] 
                    predicted_label_idx = torch.argmax(logits_for_labels).item()
                    predicted_label = predicted_label_idx  # Since labels are 0 or 1
                    # Actual label token
                    actual_label_token = labels[batch_idx, pos].item()
                    actual_label = label_token_to_label[actual_label_token]
                    # Check if prediction is correct
                    if predicted_label == actual_label:
                        correct_predictions += 1
                    total_predictions += 1
                    # print(predicted_label)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    # print(correct_predictions, total_predictions)
    
    model.train()
    return accuracy

def evaluate(model, dataloader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(inputs, attention_mask=attention_mask)
            logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

            # Get the positions where labels are not -100 (i.e., the query label positions)
            label_mask = labels != -100  # Shape: (batch_size, seq_len)

            if label_mask.any():
                predicted_tokens = torch.argmax(logits, dim=-1)  # Shape: (batch_size, seq_len)
                correct = (predicted_tokens == labels) & label_mask
                correct_predictions += correct.sum().item()
                total_predictions += label_mask.sum().item()

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    model.train()
    return accuracy

def _forget_embeddings(optimizer, model, step):
    if hasattr(optimizer, 'clear_embed_every_K_updates') and optimizer.clear_embed_every_K_updates > 0:
        if (isinstance(optimizer.clear_embed_every_K_updates, int) and (step+1) % optimizer.clear_embed_every_K_updates == 0):
            if optimizer.stop_after is None or (isinstance(optimizer.stop_after, int) and (step+1) < optimizer.stop_after):
                print(f"Clearing embeddings at step {(step+1)}", flush=True)
                model.transformer.wte.reset_parameters()
                    
def train(model,
         vocab_gen,
         train_loader,
         loss_fct, 
         optimizer, 
         scheduler, 
         device, 
         num_epochs, 
         val_loader,
         eval_loader_in_context_holdout,
         eval_loader_in_context, 
         eval_loader_in_weights,
         steps_eval=100
         ):
    model.train()
    metrics = {'context': [], 'weights': [], 'holdout': [], 'val': []}
    total_steps = 0
    for epoch in range(num_epochs):
        total_loss = 0
        for step, batch in enumerate(train_loader):
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            logits = outputs.logits 
            
            logits = logits.view(-1, logits.size(-1))  # Shape: (batch_size * seq_len, vocab_size)
            labels = labels.view(-1)
            # print(logits.shape, labels.shape)
            loss = loss_fct(logits, labels)
            # Backward pass and optimization
            # print(loss)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            _forget_embeddings(optimizer, model, total_steps)

            # Print loss every 100 steps or at the end of epoch
            if (step + 1) % 1000 == 0 or (step + 1) == len(train_loader):
                avg_loss = total_loss / (step + 1)
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], Loss: {avg_loss:.4f}")
                
            if (step + 1) % steps_eval == 0:
                metrics['holdout'].append(evaluate_in_context(
                            model, eval_loader_in_context_holdout, device, vocab_gen.eval_label_tokens))
                metrics['context'].append(evaluate_in_context(
                            model, eval_loader_in_context, device, vocab_gen.eval_label_tokens))
                metrics['weights'].append(evaluate(model, eval_loader_in_weights, device))
                metrics['val'].append(evaluate(model, val_loader, device))
                
            total_steps +=1
            # print(total_steps)
                            
                # Evaluate at the end of each epoch
        in_context_accuracy = evaluate_in_context(
            model, eval_loader_in_context, device, vocab_gen.eval_label_tokens
        )
        in_context_holdout_accuracy = evaluate_in_context(
            model, eval_loader_in_context_holdout, device, vocab_gen.eval_label_tokens
        )
        in_weights_accuracy = evaluate(model, eval_loader_in_weights, device)
        val_accuracy = evaluate(model, val_loader, device)
        
        print(f"After Epoch {epoch+1}:")
        print(f"  Evaluation Accuracy (In-Context Learning): {in_context_accuracy * 100:.2f}%")
        print(f"  Evaluation Accuracy (In-Context H Learning): {in_context_holdout_accuracy * 100:.2f}%")
        print(f"  Evaluation Accuracy (In-Weights Learning): {in_weights_accuracy * 100:.2f}%")
        print(f"  Evaluation Accuracy Val: {val_accuracy * 100:.2f}%")
    return metrics

def main(args):
        # Initialize the vocabulary generator
    vocab_gen = ICLVocabGenerator(a=args.a)
    num_icl_tokens = args.num_icl_tokens
    num_random_tokens = args.num_random_tokens
    bursty_ratio = args.bursty_ratio
    amb_ratio = args.amb_ratio
    sample_func=vocab_gen.zipfian if args.a > 0 else vocab_gen.uniform
    vocab_gen.parameterize_icl_vocab(num_icl_tokens=num_icl_tokens, num_random_tokens=num_random_tokens)

    # Generate training data
    num_train_examples = args.num_train_examples
    num_test_examples = args.num_test_examples
    train_sequences = vocab_gen.create_dataset_task_icl(num_examples=num_train_examples, 
                                                        sample_func=sample_func, 
                                                        bursty_ratio=bursty_ratio, 
                                                        amb_ratio=amb_ratio)
    val_sequences = vocab_gen.create_dataset_task_icl(num_examples=num_test_examples, 
                                                        sample_func=sample_func, 
                                                        bursty_ratio=bursty_ratio,
                                                        amb_ratio=amb_ratio)
    eval_sequences_in_context = vocab_gen.create_eval_dataset_in_context(num_examples=num_test_examples)
    eval_sequences_in_context_holdout = vocab_gen.create_eval_dataset_in_context(num_examples=num_test_examples, holdout=True)
    eval_sequences_in_weights = vocab_gen.create_eval_dataset_in_weights(num_examples=num_test_examples, sample_func=sample_func)

    # Create datasets
    block_size = args.block_size
    train_dataset = CustomSequenceDataset(train_sequences, 
                                        eos_token_id=vocab_gen.eos_token_id, 
                                        pad_token_id=vocab_gen.pad_token_id, 
                                        block_size=block_size)
    val_dataset = CustomSequenceDataset(val_sequences, 
                                        eos_token_id=vocab_gen.eos_token_id, 
                                        pad_token_id=vocab_gen.pad_token_id, 
                                        block_size=block_size)
    eval_dataset_in_context = CustomSequenceDataset(eval_sequences_in_context, 
                                                    eos_token_id=vocab_gen.eos_token_id, 
                                                    pad_token_id=vocab_gen.pad_token_id, 
                                                    block_size=block_size)
    eval_dataset_in_context_holdout = CustomSequenceDataset(eval_sequences_in_context_holdout, 
                                                    eos_token_id=vocab_gen.eos_token_id, 
                                                    pad_token_id=vocab_gen.pad_token_id, 
                                                    block_size=block_size)
    eval_dataset_in_weights = CustomSequenceDataset(eval_sequences_in_weights, 
                                                    eos_token_id=vocab_gen.eos_token_id, 
                                                    pad_token_id=vocab_gen.pad_token_id, 
                                                    block_size=block_size)

    # Create DataLoaders
    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    eval_loader_in_context = DataLoader(eval_dataset_in_context, batch_size=batch_size)
    eval_loader_in_context_holdout = DataLoader(eval_dataset_in_context_holdout, batch_size=batch_size)
    eval_loader_in_weights = DataLoader(eval_dataset_in_weights, batch_size=batch_size)
    
    config = GPT2Config(
        vocab_size=vocab_gen.vocab_size,
        n_positions=block_size,
        n_ctx=block_size,
        n_embd=args.n_embd,   # Reduced for faster training; adjust as needed
        n_layer=args.n_layer,    # Reduced number of layers
        n_head=args.n_head      # Adjusted number of heads
        )

    model = GPT2LMHeadModel(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)  # Higher learning rate for training from scratch
    if args.forget_steps > 0:
        print('Using AdamEF optimizer clearing every', args.forget_steps, 'stopping after', args.stop_forgetting_after, flush=True)
        optimizer = AdamEF(model.parameters(), lr=5e-5, lr_emb=5e-5, weight_decay=args.weight_decay, clear_embed_every_K_updates=args.forget_steps)
        optimizer.stop_after = args.stop_forgetting_after
        optimizer.embed_offset = list(filter(lambda p: p.requires_grad, model.parameters()))[0].numel()
        
    num_epochs = args.num_epochs
    total_steps = len(train_loader) * num_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps)

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    
    hist = train(model,
         vocab_gen,
         train_loader,
         loss_fct, 
         optimizer, 
         scheduler, 
         device, 
         num_epochs, 
         val_loader,
         eval_loader_in_context_holdout,
         eval_loader_in_context, 
         eval_loader_in_weights
         )
    print('saving results...', flush=True)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # for key, hist_val in hist.items():
    hist_df = pd.DataFrame(hist)
    hist_df.to_csv(os.path.join(output_dir, f'hist.csv'))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GPT-2 model with specified parameters.')

    # Add arguments for parameters that were hardcoded
    parser.add_argument('--a', type=float, default=1.0, help='Zipfian parameter')
    parser.add_argument('--num_icl_tokens', type=int, default=1600, help='Number of ICL tokens')
    parser.add_argument('--num_random_tokens', type=int, default=200, help='Number of random tokens')
    parser.add_argument('--bursty_ratio', type=float, default=1.0, help='Proportion of bursty sequences')
    parser.add_argument('--amb_ratio', type=float, default=0.05, help='Proportion of ambiguous sequences')
    parser.add_argument('--num_train_examples', type=int, default=100000, help='Number of training examples')
    parser.add_argument('--num_test_examples', type=int, default=2000, help='Number of test examples')
    parser.add_argument('--block_size', type=int, default=18, help='Block size for sequences')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for DataLoader')
    parser.add_argument('--n_embd', type=int, default=64, help='Embedding size in GPT-2 config')
    parser.add_argument('--n_layer', type=int, default=4, help='Number of layers in GPT-2 config')
    parser.add_argument('--n_head', type=int, default=4, help='Number of heads in GPT-2 config')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=8, help='Number of training epochs')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save results')
    parser.add_argument('--forget_steps', type=int, default=0)
    parser.add_argument('--stop_forgetting_after', type=int, default=1000000)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    args = parser.parse_args()
    main(args)
