from collections import defaultdict
import logging
from typing import cast, Dict, List, Tuple, Union
from typing_extensions import get_args, Literal
import sys
import os
import numpy as np
import random 
import seaborn as sns
from PIL import Image
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

import yaml
import argparse
import pandas as pd
from tqdm.notebook import tqdm
from functools import partial

from torch.utils.data import DataLoader, TensorDataset

sys.path.append('/users/sanand14/data/sanand14/learning_dynamics/src/experiments/utils')
sys.path.append('/users/sanand14/data/sanand14/learning_dynamics/src/experiments')

from aheads import create_repeats_dataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
from transformers import BertConfig, BertForMaskedLM, AdamW

special_token_dict_pos = {}
special_token_dict_dep = {}
noun_tokens = []
adj_tokens = []
seq_tokens = []
example_len = 0

class Probe(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.body = nn.Linear(num_features, 1, bias=False)
    def forward(self, x):
        if isinstance(x, list):
            x, _ = x
        return self.body(x)
        
def bin_step(model, batch):
    x, y = batch
    logits = model.forward(x)
    loss = F.binary_cross_entropy(torch.sigmoid(logits), y)
    acc = ((logits.squeeze() > 0.5).float() == y.squeeze()).float().mean()
    return loss, {"loss": loss.item(), "acc": acc.item()}

def bin_train_loop(model, train_dataloader, test_dataloader, optimizer, epochs):
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        pbar.set_description(f"Training Epoch {epoch}")
        for batch in train_dataloader:
            model.train()
            optimizer.zero_grad()
            loss, stats = bin_step(model, batch)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(**stats)
        model.eval()
        with torch.no_grad():
            pbar.set_description("Validation")
            results = bin_val_loop(model, test_dataloader)
    return results

def bin_val_loop(model, test_dataloader):
    model.eval()
    acc, losses = [], []
    with torch.no_grad():
        pbar = tqdm(test_dataloader)
        for val_batch in pbar:
            loss, stats = bin_step(model, val_batch)
            acc.append(stats["acc"])
            losses.append(stats["loss"])
            results = {"acc": np.mean(acc), "loss": np.mean(losses)}
            pbar.set_postfix(**results)
    return results
  
def parameterize_pos_vocab_old(num_pos_tokens):
  assert num_pos_tokens % 2 == 0, "Has to be even"
  global special_token_dict_pos
  global noun_tokens
  global adj_tokens
  special_token_dict_pos = {
      'cop': num_pos_tokens,
      'null': num_pos_tokens+1,
      'mask': num_pos_tokens+2
  }
  noun_tokens = range(num_pos_tokens//2)
  adj_tokens = range(num_pos_tokens//2, num_pos_tokens)
  
def parameterize_pos_vocab(num_pos_tokens):
  assert num_pos_tokens % 2 == 0, "Has to be even"
  global special_token_dict_pos
  global noun_tokens
  global adj_tokens
  special_token_dict_pos = {
      'cop': num_pos_tokens,
      'mask': num_pos_tokens+1
  }
  noun_tokens = range(num_pos_tokens//2)
  adj_tokens = range(num_pos_tokens//2, num_pos_tokens)

def tail_end_z(type='noun'):
    assert type in ['noun', 'adj'], "type not found"
    n_nouns, n_adjs = len(noun_tokens),len(adj_tokens)
    bot10_nouns, bot10_adjs = noun_tokens[-n_nouns//10:], adj_tokens[-n_adjs//10:] 
    # only works because probability monotonically decreasing in zipfian as token identity increases
    return random.choice(bot10_nouns) if type == 'noun' else random.choice(bot10_adjs)

def uniform(type='noun'):
    assert type in ['noun', 'adj'], "type not found"
    return random.choice(noun_tokens) if type == 'noun' else random.choice(adj_tokens)

def zipfian(type='noun', a=1.5):
    # print("A", a)
    assert type in ['noun', 'adj'], "type not found"
    if type == 'noun':
        map = {k:v for k,v in zip(range(len(noun_tokens)), noun_tokens)}
    else:
        map = {k:v for k,v in zip(range(len(adj_tokens)), adj_tokens)}    
    value = np.random.zipf(a)
    while value not in map.keys():
        value = np.random.zipf(a)
    return map[value]

def create_dataset_task_pos_old(num_examples, mask_probability=0.15, masking='train', sample_func=zipfian, tail_end=False, switch=False):
    dataset = []
    labels = []
    alt_labels = []
    if switch:
        sample_func = lambda type: sample_func('noun') if type == 'adj' else sample_func('adj')
        tail_end_z = lambda type: tail_end_z('noun') if type == 'adj' else tail_end_z('adj')
    for _ in range(num_examples):
        rand_val = random.random()
        if rand_val < 0.10: #0.40
            noun = sample_func('noun') if not tail_end else tail_end_z('noun')
            seq = [special_token_dict_pos['cop'], special_token_dict_pos['null'], noun]
            if rand_val < 0.05: #0.20
                adj = sample_func('adj') if not tail_end else tail_end_z('adj')
                seq.extend([adj, special_token_dict_pos['null'], special_token_dict_pos['null'], special_token_dict_pos['null']])
            else:
                seq.extend([noun, special_token_dict_pos['null'], noun, noun])
            seq_alt = seq.copy()
        elif rand_val < 0.20: #0.80
            noun = sample_func('noun') if not tail_end else tail_end_z('noun', )
            seq = [noun, special_token_dict_pos['cop'], special_token_dict_pos['null']]
            if rand_val < 0.15: #0.60
                adj = sample_func('adj') if not tail_end else tail_end_z('adj')
                seq.extend([adj, special_token_dict_pos['null'], special_token_dict_pos['null'], special_token_dict_pos['null']])
            else:
                seq.extend([noun, special_token_dict_pos['null'], noun, noun])
            seq_alt = seq.copy()
        
        if rand_val < 0.60: # 20 - 60  #0.80 
            adj, noun = sample_func('adj') if not tail_end else tail_end_z('adj'), sample_func('noun') if not tail_end else tail_end_z('noun')
            seq = [special_token_dict_pos['cop'], adj, noun]
            seq_alt = seq.copy()
            if rand_val < 0.40: #0.75
                seq.extend([adj, adj, adj, adj])
                seq_alt.extend([adj, special_token_dict_pos['null'], special_token_dict_pos['null'], special_token_dict_pos['null']])
            else:
                seq.extend([noun, adj, noun, noun])
                seq_alt.extend([noun, special_token_dict_pos['null'], noun, noun])
        else: # 60-100
            adj, noun = sample_func('adj') if not tail_end else tail_end_z('adj'), sample_func('noun')  if not tail_end else tail_end_z('noun')
            seq = [noun, special_token_dict_pos['cop'], adj]
            seq_alt = seq.copy()
            if rand_val < 0.80: #0.95
                seq.extend([adj, adj, adj, adj])
                seq_alt.extend([adj, special_token_dict_pos['null'], special_token_dict_pos['null'], special_token_dict_pos['null']])
            else:
                seq.extend([noun, adj, noun, noun])
                seq_alt.extend([noun, special_token_dict_pos['null'], noun, noun])
        label_seq = seq.copy()
        alt_labels_seq = seq_alt.copy()
        if masking=='train':
            for i in range(len(seq)):
                if random.random() < mask_probability:
                    seq[i] = special_token_dict_pos['mask']
                else:
                    label_seq[i] = -100 # ignore in loss fxn
                    alt_labels_seq[i] = -100
        else:
            for i in range(len(seq)):
                if i >= len(seq) - 3:
                    seq[i] = special_token_dict_pos['mask']
                else:
                    label_seq[i] = -100
                    alt_labels_seq[i] = -100
        dataset.append(seq)
        labels.append(label_seq)
        alt_labels.append(alt_labels_seq)
    return dataset, labels, alt_labels

def create_dataset_task_pos(num_examples, sample_func=zipfian, tail_end_z = tail_end_z, tail_end=False, switch=False):
    dataset = []
    labels = []
    if switch:
        sample_func = lambda type: sample_func('noun') if type == 'adj' else sample_func('adj')
        tail_end_z = lambda type: tail_end_z('noun') if type == 'adj' else tail_end_z('adj')
        
    for _ in range(num_examples):
        rand_val = random.random()
        if rand_val < 0.50: # 20 - 60  #0.80 
            adj, noun = sample_func('adj') if not tail_end else tail_end_z('adj'), sample_func('noun') if not tail_end else tail_end_z('noun')
            seq = [special_token_dict_pos['cop'], adj, noun]
            if rand_val < 0.25: #0.75
                seq.extend([adj, adj, adj, adj])
            else:
                seq.extend([noun, adj, noun, noun])
        else: # 60-100
            adj, noun = sample_func('adj') if not tail_end else tail_end_z('adj'), sample_func('noun')  if not tail_end else tail_end_z('noun')
            seq = [noun, special_token_dict_pos['cop'], adj]
            if rand_val < 0.75: #0.95
                seq.extend([adj, adj, adj, adj])
            else:
                seq.extend([noun, adj, noun, noun])
        label_seq = seq.copy()

        for i in range(len(seq)):
            if i >= len(seq) - 3:
                seq[i] = special_token_dict_pos['mask']
            else:
                label_seq[i] = -100
                
        dataset.append(seq)
        labels.append(label_seq)
    return dataset, labels

def parameterize_dep_vocab(num_dep_tokens=400, len_ex=20):
  global special_token_dict_dep
  global seq_tokens
  global example_len
  special_token_dict_dep = {
      'mask': num_dep_tokens
  }
  seq_tokens = range(num_dep_tokens)
  example_len = len_ex

## try + make harder 
## even smaller w the model

def generate_sequence(length, start_value, step_probability):
    sequence = [start_value]
    current_value = start_value

    for _ in range(length - 1):
        if random.random() < step_probability:
            # Change the value with the defined step change
            current_value += 1
        sequence.append(current_value)
    return sequence
    
def create_dataset_task_dep(num_examples, mask_probability=0.15, masking='train', elastic=True, step_prob = 0.90):
    assert example_len % 2 == 0, "example len must be even"
    seq_len = example_len // 2
    
    dataset = []
    labels = []
    alt_labels = []
    for _ in range(num_examples):
        rand_val = random.random()
        start_index = random.randint(0, len(seq_tokens) - seq_len)
        if elastic:
            seq = generate_sequence(seq_len, seq_tokens[start_index], step_prob) ## can have repeats
        else:
            seq = list(seq_tokens[start_index:start_index + seq_len])
        
        if rand_val < 0.80:
            seq *= 2
            seq_alt = seq.copy()
        else:
            if elastic: 
                change_ind = random.choice(range(2, seq_len+1))
                seq[-change_ind+1], seq[-change_ind] = seq[-change_ind], seq[-change_ind+1]
                seq *= 2
                seq_alt = seq.copy()
                seq[-change_ind+1] = seq[-change_ind]
            else:
                seq[-1], seq[-2] = seq[-2], seq[-1]
                seq *= 2
                seq_alt = seq.copy()
                seq[-1] = seq[-2]

        label_seq = seq.copy()
        alt_labels_seq = seq_alt.copy()
        
        if masking=='train':
            for i in range(len(seq)):
                if random.random() < mask_probability:
                    seq[i] = special_token_dict_dep['mask']
                else:
                    label_seq[i] = -100 # ignore in loss fxn
                    alt_labels_seq[i] = -100
        else:
            for i in range(len(seq)):
                if i >= len(seq) - seq_len:
                    seq[i] = special_token_dict_dep['mask']
                else:
                    label_seq[i] = -100
                    alt_labels_seq[i] = -100
        dataset.append(seq)
        labels.append(label_seq)
        alt_labels.append(alt_labels_seq)
        
    return dataset, labels, alt_labels
  
def create_dataloaders(num_train, num_val, device="cpu", task=create_dataset_task_pos, batch_size=128):
    inputs_t, labels_t = task(num_train)
    inputs_v, labels_v = task(num_val)
    inputs_e, labels_e = task(num_val, tail_end=True)
    inputs_s, labels_s = task(num_val, switch=True)
    
    # print(inputs_t[:5], labels_t[:5], alt_labels_t[:5])
    inputs_t = torch.tensor(inputs_t).to(device)
    labels_t = torch.tensor(labels_t).to(device)
    
    inputs_v = torch.tensor(inputs_v).to(device)
    labels_v = torch.tensor(labels_v).to(device)

    inputs_e = torch.tensor(inputs_e).to(device)
    labels_e = torch.tensor(labels_e).to(device)
    
    inputs_s = torch.tensor(inputs_s).to(device)
    labels_s = torch.tensor(labels_s).to(device)
    
    train_dataset = TensorDataset(inputs_t.detach(), labels_t)
    val_dataset = TensorDataset(inputs_v.detach(), labels_v)
    tail_end_val_dataset = TensorDataset(inputs_e.detach(), labels_e)
    switch_val_dataset = TensorDataset(inputs_s.detach(), labels_s)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    tail_end_val_dataloader = DataLoader(tail_end_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    switch_dataloader = DataLoader(switch_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_dataloader, val_dataloader, tail_end_val_dataloader, switch_dataloader

def create_dataloaders_bin(data, labels, device="cpu"):
    train_len = int(0.80 * len(data))
    inputs_t, labels_t = data[:train_len], labels[:train_len]
    inputs_v, labels_v = data[train_len:], labels[train_len:]
    train_dataset = TensorDataset(inputs_t.detach(), labels_t.view(-1, 1))
    val_dataset = TensorDataset(inputs_v.detach(), labels_v.view(-1, 1))
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    return train_dataloader, val_dataloader

def pca_pos(model, val_dataloader, title, c_step, probe_results=defaultdict(list), device="cpu", output_dir=None, plot=True):
    labels, hidden_layers = [], defaultdict(list)
    model.config.output_hidden_states=True
    num_hidden_states = model.config.num_hidden_layers + 1
    adj_min = min(adj_tokens)
    if plot:
      fig, axs = plt.subplots(1, num_hidden_states, figsize=(5*num_hidden_states, 5))
    for batch in val_dataloader:
        examples, _, _ = batch
        labels.append((examples[:, -4] < adj_min).float())
        with torch.no_grad():
            outputs = model(examples.to(model.device))
        for j in range(num_hidden_states):
            hidden_layers[j].append(outputs.hidden_states[j][:, -4, :])
    labels = torch.concat(labels, axis=0).unsqueeze(1)
    for i in range(num_hidden_states):
        torch_embed = torch.concat(hidden_layers[i], axis=0).squeeze()
        probe = Probe(torch_embed.shape[1]).to(model.device)
        train_dataloader_bin, val_dataloader_bin = create_dataloaders_bin(torch_embed, labels, device=model.device)
        optim_bin = torch.optim.AdamW(probe.parameters(), lr=1e-3) 
        results = bin_train_loop(probe, train_dataloader_bin, val_dataloader_bin, optim_bin, 3)
        probe_results[i].append(results['acc'])
        if plot:
          labels_numpy = labels.cpu().numpy().squeeze()
          np_embed = torch_embed.cpu().numpy()
          pca = PCA(n_components=2)
          data_pca = pca.fit_transform(np_embed)
          unique_labels = np.unique(labels_numpy)
          colors = matplotlib.colormaps.get_cmap('viridis').colors[:len(unique_labels)]
          for j, label in enumerate(unique_labels):
              axs[i].scatter(data_pca[labels_numpy == label, 0], data_pca[labels_numpy == label, 1], 
                          alpha=0.5, color=colors[j], label=f"Label {label}")
          axs[i].set_title(f"state {i} acc={results['acc']:0.2f}")
          axs[i].set_xlabel("Principal Component 1")
          axs[i].set_ylabel("Principal Component 2")
          axs[i].legend()
          axs[i].grid(True)
    if plot:           
      plt.suptitle(title)
      if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir,f'/pca_step_{c_step}.png'))
      plt.show()
    plt.close()
    return probe_results

def step(model, batch, hard_acc=False, criterion=nn.CrossEntropyLoss(), alt=False):
    if alt:
        x, y, alt_y = batch
    else:
        x, y = batch
    output = model.forward(x)
    logits = output.logits.transpose(1, 2)
    loss = criterion(logits, y)
    batch_len = logits.shape[0]
    where = (y != -100)
    y = y[where].view(batch_len, -1)
    if alt:
        alt_y = alt_y[where].view(batch_len, -1)
    preds = logits.argmax(axis=1)[where].view(batch_len, -1)
    ## full examples where alt != alt_label
    batch_same = (y == alt_y).all(axis=-1)
    if hard_acc:
        acc = (preds == y).all(axis=-1).float().mean() 
        if alt:
            alt_acc = (preds == alt_y).all(axis=-1).float().mean()
    else:
        acc = (preds == y)[~batch_same].float().mean() 
        if alt:
            alt_acc = (preds == alt_y)[~batch_same].float().mean()
    if alt:
        return loss, {"loss": loss.item(), "acc": acc.item(), "alt_acc": alt_acc.item()}
    return loss, {"loss": loss.item(), "acc": acc.item()}

def train_loop(model, train_dataloader, test_dataloader, optimizer, epochs, step_eval=1000, name=None, pca=True):
    tail_end_val_dataloader = None
    if isinstance(test_dataloader, tuple):
        if len(test_dataloader) == 3:
            test_dataloader, tail_end_val_dataloader, switch_val_dataloader = test_dataloader
        elif len(test_dataloader) == 2:
            test_dataloader, tail_end_val_dataloader = test_dataloader
        else:
            raise ValueError('Not recognized format for test_dataloader order/length')
        
    pbar = tqdm(range(epochs))
    val_stats = {}
    hist = {}
    hist_tail = {}
    hist_switch = {}
    c_step = 0
    probe_results = defaultdict(list)
    probe_results_tail = defaultdict(list)
    probe_results_switch = defaultdict(list)
    for epoch in pbar:
        pbar.set_description(f"Training Epoch {epoch}")
        for batch in train_dataloader:
            sys.stdout.flush()
            sys.stderr.flush()
            c_step += 1
            model.train()
            optimizer.zero_grad()
            loss, stats = step(model, batch, hard_acc=True)
            loss.backward()
            optimizer.step()
            stats.update(val_stats)
            pbar.set_postfix(**stats)
            if isinstance(step_eval, int):
                if c_step % step_eval == 0:
                    hist[c_step] = val_loop(model, test_dataloader)
                    if tail_end_val_dataloader is not None:
                        hist_tail[c_step] = val_loop(model, tail_end_val_dataloader)
                    if switch_val_dataloader is not None:
                        hist_switch[c_step] = val_loop(model, switch_val_dataloader)
                    if pca:
                        probe_results = pca_pos(model, test_dataloader, f'Step {c_step}', c_step, probe_results)
                        if tail_end_val_dataloader is not None:
                            probe_results_tail = pca_pos(model, tail_end_val_dataloader, f'Step {c_step}', c_step, probe_results_tail)
                        if switch_val_dataloader is not None:
                            probe_results_switch = pca_pos(model, switch_val_dataloader, f'Step {c_step}', c_step, probe_results_switch)
                    if name is not None:
                        torch.save(model.state_dict(), f'models/{name}_step_{c_step}.pth')
            elif isinstance(step_eval, list):
                if c_step in step_eval:
                    hist[c_step] = val_loop(model, test_dataloader)
                    if tail_end_val_dataloader is not None:
                        hist_tail[c_step] = val_loop(model, tail_end_val_dataloader)
                    if switch_val_dataloader is not None:
                        hist_switch[c_step] = val_loop(model, switch_val_dataloader)
                    if pca:
                        probe_results = pca_pos(model, test_dataloader, f'Step {c_step}', c_step, probe_results)
                        if tail_end_val_dataloader is not None:
                            probe_results_tail = pca_pos(model, tail_end_val_dataloader, f'Step {c_step}', c_step, probe_results_tail)
                        if switch_val_dataloader is not None:
                            probe_results_switch = pca_pos(model, switch_val_dataloader, f'Step {c_step}', c_step, probe_results_switch)
                    if name is not None:
                        torch.save(model.state_dict(), f'models/{name}_step_{c_step}.pth')
            else:
                raise ValueError('Not recognized format for step')
        model.eval()
        with torch.no_grad():
            pbar.set_description("Validation")
            val_stats = val_loop(model, test_dataloader)
            val_stats = {"val_" + key:val for key,val in val_stats.items()}
            pbar.set_postfix(**val_stats)
    return hist, probe_results, hist_tail, probe_results_tail, hist_switch, probe_results_switch
                
def val_loop(model, test_dataloader):
    model.eval()
    acc, acc_alt, losses = [], [], []
    with torch.no_grad():
        pbar = tqdm(test_dataloader)
        for val_batch in pbar:
            loss, stats = step(model, val_batch, hard_acc=True)
            acc.append(stats["acc"])
            acc_alt.append(stats["alt_acc"])
            losses.append(stats["loss"])
            results = {"acc": np.mean(acc), "alt_acc": np.mean(acc_alt), "loss": np.mean(losses)}
            pbar.set_postfix(**results)
    return results

def main(args):
    num_epochs = args.epochs
    batch_size = args.batch_size
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    parameterize_pos_vocab(args.vocab_size) if args.task == 'pos' else parameterize_dep_vocab(400, 20)
    
    task=create_dataset_task_pos if args.task == 'pos' else create_dataset_task_dep
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cudnn.benchmark = False
    
    config = BertConfig(
        vocab_size=args.vocab_size+2 if task==create_dataset_task_pos else 401, ## args.vocab_size+3 if have null token
        hidden_size=args.hidden_size, # 128  
        num_hidden_layers=args.hidden_num_layers, # 8
        num_attention_heads=args.num_attention_heads, # 8
        intermediate_size=args.intermediate_size # 512
    )
    sample_func = lambda type: zipfian(type, a=args.a) if args.sample_func == 'zipfian' else lambda type: uniform(type)
    partial_task = partial(task, sample_func=sample_func) if task==create_dataset_task_pos else task
    toy_bert_model = BertForMaskedLM(config).to(device)
    optimizer = torch.optim.AdamW(toy_bert_model.parameters(), lr=5e-5) 
    max_num_steps = args.dataset_size *  num_epochs/batch_size
    print('Max number of steps is ', max_num_steps, flush=True)
    print('creating dataset...', flush=True)
    train_dataloader, val_dataloader, tail_end_val_dataloader, switch_val_dataloader = create_dataloaders(args.dataset_size, 10_000, device=device, task=partial_task)
    # torch.save(train_dataloader.dataset.tensors[0], 'train_dataloader0.pth')
    # step_eval = list(range(0, 1010, 10)) + [1200, 1400, 1600, 1800, 2000, 2400, 3200, 4000, 6000, 8000, 16000, 28000]
    step_eval = list(range(0, 1000, 10)) + list(range(1000, 30000, 100))
    print('training...', flush=True)
    hist, probing_results, hist_tail, probing_results_tail, hist_switch, probing_results_switch = train_loop(toy_bert_model, train_dataloader, (val_dataloader, tail_end_val_dataloader, switch_val_dataloader), \
                                        optimizer, num_epochs, \
                                        step_eval=step_eval, name=None)#'pos_model')
    print('saving results...', flush=True)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    val_stats = val_loop(toy_bert_model, val_dataloader)
    print("VAL", val_stats) # 10 - 80 identical, 10 - 20 1 token diff, 20 - 80 2 token diff
    tail_stats = val_loop(toy_bert_model, tail_end_val_dataloader)
    print("TAIL VAL", tail_stats)
    switch_stats = val_loop(toy_bert_model, switch_val_dataloader)
    print("SWITCH VAL", switch_stats)
    
    hist_df = pd.DataFrame(hist)
    hist_df.to_csv(os.path.join(output_dir, 'hist.csv'))
    
    hist_tail_df = pd.DataFrame(hist_tail)
    hist_tail_df.to_csv(os.path.join(output_dir, 'hist_tail.csv'))
    
    hist_switch_df = pd.DataFrame(hist_switch)
    hist_switch_df.to_csv(os.path.join(output_dir, 'hist_switch.csv'))
    
    df = pd.DataFrame(probing_results)
    df = df.transpose()
    df.columns = step_eval[:len(df.columns)]
    df = df[::-1]
    df.to_csv(os.path.join(output_dir, 'pos_probing_results.csv'))

    ax = sns.heatmap(df, annot=False)
    ax.set_xlabel("Step")
    ax.set_ylabel("Layer")
    ax.set_title("POS Probing")
    plt.savefig(os.path.join(output_dir, 'pos_probing_steps.png'))
    
    
    df_tail = pd.DataFrame(probing_results_tail)
    df_tail = df_tail.transpose()
    df_tail.columns = step_eval[:len(df_tail.columns)]
    df_tail = df_tail[::-1]
    df_tail.to_csv(os.path.join(output_dir, 'pos_probing_tail_results.csv'))
    
    plt.figure()
    ax = sns.heatmap(df_tail, annot=False)
    ax.set_xlabel("Step")
    ax.set_ylabel("Layer")
    ax.set_title("Tail POS Probing")
    plt.savefig(os.path.join(output_dir, 'pos_probing_tail_steps.png'))
    plt.close()
    
    df_switch = pd.DataFrame(probing_results_switch)
    df_switch = df_switch.transpose()
    df_switch.columns = step_eval[:len(df_switch.columns)]
    df_switch = df_switch[::-1]
    df_switch.to_csv(os.path.join(output_dir, 'pos_probing_switch_results.csv'))
    
    plt.figure()
    ax = sns.heatmap(df_switch, annot=False)
    ax.set_xlabel("Step")
    ax.set_ylabel("Layer")
    ax.set_title("Switch POS Probing")
    plt.savefig(os.path.join(output_dir, 'pos_probing_switch_results.png'))
    plt.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on a toy task')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--task', type=str, default="pos", help='Task to train on')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--step_eval', type=int, default=1000, help='How often to evaluate')
    parser.add_argument('--dataset_size', type=int, default=5_000_000, help='Size of the dataset')
    parser.add_argument('--hidden_num_layers', type=int, default=8, help='Hidden size of the model')
    parser.add_argument('--num_attention_heads', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--hidden_size', type=int, default=16, help='Hidden size of the model')
    parser.add_argument('--intermediate_size', type=int, default=32, help='Intermediate size of the model')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='models', help='Output directory')
    parser.add_argument('--vocab_size', type=int, default=100, help='Vocab size of the model')
    parser.add_argument('--a', type=float, default=1.5, help='Zipfian parameter')
    parser.add_argument('--sample_func', type=str, default='zipfian', help='Sampling function')
    args = parser.parse_args()
    main(args)