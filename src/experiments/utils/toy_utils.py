from collections import defaultdict
from typing import List, Tuple, Callable
from typing_extensions import get_args, Literal
import sys
import numpy as np
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm

from torch.utils.data import DataLoader, TensorDataset

sys.path.append('/users/sanand14/data/sanand14/learning_dynamics/src/experiments/utils')
sys.path.append('/users/sanand14/data/sanand14/learning_dynamics/src/experiments')

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import numpy as np
from dataclasses import dataclass, field

## ------------------------------------------ PROBING ----------------------------------  
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
  
def create_dataloaders_bin(data, labels, device="cpu"):
    train_len = int(0.80 * len(data))
    inputs_t, labels_t = data[:train_len], labels[:train_len]
    inputs_v, labels_v = data[train_len:], labels[train_len:]
    train_dataset = TensorDataset(inputs_t.detach(), labels_t.view(-1, 1))
    val_dataset = TensorDataset(inputs_v.detach(), labels_v.view(-1, 1))
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    return train_dataloader, val_dataloader
  
## ------------------------------------------ DATASET CREATION FUNCS ----------------------------------

@dataclass
class POSVocabGenerator:
    special_token_dict_pos: dict = field(default_factory=dict)
    noun_tokens: List[int] = field(default_factory=list)
    adj_tokens: List[int] = field(default_factory=list)
    random_tokens: List[int] = field(default_factory=list)
    amb_tokens: List[int] = field(default_factory=list)
    
    def parameterize_pos_vocab(self, num_pos_tokens: int, num_random_tokens: int, prop_amb=0.0, bins=10, tail_only=False):
        assert num_pos_tokens % 2 == 0, "Number of POS tokens must be even"
        self.special_token_dict_pos = {'cop': num_pos_tokens, 'mask': num_pos_tokens + 1}
        self.noun_tokens = list(range(num_pos_tokens // 2))
        self.adj_tokens = list(range(num_pos_tokens // 2, num_pos_tokens))
        
        def choose_amb_tokens(lst):
            bin_size = len(lst) // bins
            binned = [lst[i*bin_size:(i+1)*bin_size] for i in range(bins)]
            binned[-1].extend(lst[bins*bin_size:])
            selected = [np.random.choice(bin_, size=int(np.ceil(len(bin_) * prop_amb)), replace=False).tolist() for bin_ in binned]
            selected = list(itertools.chain.from_iterable(selected))
            return selected
    
        if tail_only:
            self.amb_tokens = self.noun_tokens[int(-len(self.noun_tokens) * prop_amb):] + self.adj_tokens[int(-len(self.adj_tokens) * prop_amb):]
        else:
            self.amb_tokens = choose_amb_tokens(self.noun_tokens) + choose_amb_tokens(self.adj_tokens)
        print(self.amb_tokens)
        self.random_tokens = list(range(num_pos_tokens + 2, num_pos_tokens + 2 + num_random_tokens))

    def tail_end_z(self, type='noun'):
        assert type in ['noun', 'adj'], "type not found"
        tokens = self.noun_tokens if type == 'noun' else self.adj_tokens
        return random.choice(tokens[-len(tokens) // 10:])

    def uniform(self, type='noun'):
        assert type in ['noun', 'adj'], "type not found"
        tokens = self.noun_tokens if type == 'noun' else self.adj_tokens
        return random.choice(tokens)

    def zipfian(self, type='noun', a=1.5):
        assert type in ['noun', 'adj'], "type not found"
        tokens = self.noun_tokens if type == 'noun' else self.adj_tokens
        map = {k: v for k, v in enumerate(tokens)}
        value = np.random.zipf(a)
        while value not in map:
            value = np.random.zipf(a)
        return map[value]
    
    def get_vocab_tokens(self):
        return len(self.noun_tokens + self.adj_tokens + self.random_tokens) + len(self.special_token_dict_pos)

    def create_dataset_task_pos(self, num_examples: int, sample_func: Callable = zipfian, prop_amb_all=0.0, tail_end=False, switch=False, random_v=False, amb_only=False, non_amb_only=False, device=None) -> Tuple[List[List[int]], List[List[int]]]:
        dataset = []
        labels = []
        
        def get_sample_func_upd(sample_func):
            ## random embeddings
            if random_v:
                if len(self.random_tokens) == 0:
                    raise ValueError('No random tokens found')
                return lambda type: random.choice(self.random_tokens)
            ## switch and tail
            if switch and tail_end:
                return lambda type: self.tail_end_z('noun') if type == 'adj' else self.tail_end_z('adj')
            ## just switch 
            if switch:
                return lambda type: sample_func('noun') if type == 'adj' else sample_func('adj')
            ## just tail end of distribution (10% tokens by number)
            if tail_end:
                return self.tail_end_z
            ## ambigous tokens
            if amb_only:
                return lambda type: random.choice(self.amb_tokens)
            ## non-ambigous tokens
            if non_amb_only:
                def tmp_func_na(type):
                    while True:
                        if type == 'noun':
                            token = random.choice(self.noun_tokens)
                        else:
                            token = random.choice(self.adj_tokens)
                        if token not in self.amb_tokens:
                            return token
                return tmp_func_na
            ## proportion ambiguous (used during training)
            if prop_amb_all > 0.0:
                def tmp_func(type):
                    if random.random() > prop_amb_all:
                        return sample_func(type)
                    else:
                        return sample_func('noun') if type == 'adj' else sample_func('adj')
                return tmp_func            
            return sample_func

        sample_func_upd = get_sample_func_upd(sample_func)

        for _ in range(num_examples):
            rand_val = random.random()
            adj, noun = sample_func_upd('adj'), sample_func_upd('noun')
            seq = [self.special_token_dict_pos['cop'], adj, noun] if rand_val < 0.50 else [noun, self.special_token_dict_pos['cop'], adj]
            seq.extend([adj, adj, adj, adj] if rand_val < 0.25 or rand_val >= 0.75 else [noun, adj, noun, noun])
            label_seq = seq.copy()

            for i in range(len(seq)):
                if i >= len(seq) - 3:
                    seq[i] = self.special_token_dict_pos['mask']
                else:
                    label_seq[i] = -100

            dataset.append(seq)
            labels.append(label_seq)
        
        if device is not None:
            dataset = torch.tensor(dataset, device=device)
            labels = torch.tensor(labels, device=device)

        return dataset, labels
  
@dataclass
class POSVocabGeneratorOld:
    special_token_dict_pos: dict = field(default_factory=dict)
    noun_tokens: List[int] = field(default_factory=list)
    adj_tokens: List[int] = field(default_factory=list)

    def parameterize_pos_vocab_old(self, num_pos_tokens: int):
        assert num_pos_tokens % 2 == 0, "Number of POS tokens must be even"
        self.special_token_dict_pos = {'cop': num_pos_tokens, 'null': num_pos_tokens + 1, 'mask': num_pos_tokens + 2}
        self.noun_tokens = list(range(num_pos_tokens // 2))
        self.adj_tokens = list(range(num_pos_tokens // 2, num_pos_tokens))
    
    def tail_end_z(self, type='noun'):
        assert type in ['noun', 'adj'], "type not found"
        tokens = self.noun_tokens if type == 'noun' else self.adj_tokens
        return random.choice(tokens[-len(tokens) // 10:])
    
    def uniform(self, type='noun'):
        assert type in ['noun', 'adj'], "type not found"
        tokens = self.noun_tokens if type == 'noun' else self.adj_tokens
        return random.choice(tokens)

    def zipfian(self, type='noun', a=1.5):
        assert type in ['noun', 'adj'], "type not found"
        tokens = self.noun_tokens if type == 'noun' else self.adj_tokens
        map = {k: v for k, v in enumerate(tokens)}
        value = np.random.zipf(a)
        while value not in map:
            value = np.random.zipf(a)
        return map[value]
      
    def create_dataset_task_pos_old(self, num_examples: int,  sample_func : Callable, mask_probability=0.15, masking='train', tail_end=False, switch=False, num_random=0) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        dataset = []
        labels = []
        alt_labels = []

        for _ in range(num_examples):
            rand_val = random.random()
            seq, seq_alt = [], []
            label_seq, alt_labels_seq = [], []

            if switch:
                temp_sample_func = lambda type: sample_func('noun') if type == 'adj' else sample_func('adj')
                temp_tail_end_z = lambda type: self.tail_end_z('noun') if type == 'adj' else self.tail_end_z('adj')
            else:
                temp_sample_func = sample_func
                temp_tail_end_z = self.tail_end_z

            # Logic for sequence and label generation
            if rand_val < 0.10:
                noun = temp_sample_func('noun') if not tail_end else temp_tail_end_z('noun')
                seq = [self.special_token_dict_pos['cop'], self.special_token_dict_pos['null'], noun]
                if rand_val < 0.05:
                    adj = temp_sample_func('adj') if not tail_end else temp_tail_end_z('adj')
                    seq.extend([adj, self.special_token_dict_pos['null'], self.special_token_dict_pos['null'], self.special_token_dict_pos['null']])
                else:
                    seq.extend([noun, self.special_token_dict_pos['null'], noun, noun])
                seq_alt = seq.copy()
            elif rand_val < 0.20:
                noun = temp_sample_func('noun') if not tail_end else temp_tail_end_z('noun')
                seq = [noun, self.special_token_dict_pos['cop'], self.special_token_dict_pos['null']]
                if rand_val < 0.15:
                    adj = temp_sample_func('adj') if not tail_end else temp_tail_end_z('adj')
                    seq.extend([adj, self.special_token_dict_pos['null'], self.special_token_dict_pos['null'], self.special_token_dict_pos['null']])
                else:
                    seq.extend([noun, self.special_token_dict_pos['null'], noun, noun])
                seq_alt = seq.copy()
            elif rand_val < 0.60:
                adj, noun = temp_sample_func('adj'), temp_sample_func('noun')
                seq = [self.special_token_dict_pos['cop'], adj, noun]
                seq_alt = seq.copy()
                if rand_val < 0.40:
                    seq.extend([adj, adj, adj, adj])
                    seq_alt.extend([adj, self.special_token_dict_pos['null'], self.special_token_dict_pos['null'], self.special_token_dict_pos['null']])
                else:
                    seq.extend([noun, adj, noun, noun])
                    seq_alt.extend([noun, self.special_token_dict_pos['null'], noun, noun])
            else:
                adj, noun = temp_sample_func('adj'), temp_sample_func('noun')
                seq = [noun, self.special_token_dict_pos['cop'], adj]
                seq_alt = seq.copy()
                if rand_val < 0.80:
                    seq.extend([adj, adj, adj, adj])
                    seq_alt.extend([adj, self.special_token_dict_pos['null'], self.special_token_dict_pos['null'], self.special_token_dict_pos['null']])
                else:
                    seq.extend([noun, adj, noun, noun])
                    seq_alt.extend([noun, self.special_token_dict_pos['null'], noun, noun])

            # Masking for training or prediction
            if masking == 'train':
                for i in range(len(seq)):
                    if random.random() < mask_probability:
                        seq[i] = self.special_token_dict_pos['mask']
                    else:
                        label_seq[i] = -100
                        alt_labels_seq[i] = -100
            else:
                for i in range(len(seq)):
                    if i >= len(seq) - 3:
                        seq[i] = self.special_token_dict_pos['mask']
                    else:
                        label_seq[i] = -100
                        alt_labels_seq[i] = -100

            dataset.append(seq)
            labels.append(label_seq)
            alt_labels.append(alt_labels_seq)

        return dataset, labels, alt_labels

@dataclass
class DepVocabGenerator:
    special_token_dict_dep: dict = field(default_factory=dict)
    seq_tokens: List[int] = field(default_factory=list)
    example_len: int = 20  # Default value set to 20

    def parameterize_dep_vocab(self, num_dep_tokens=400, len_ex=20):
        self.special_token_dict_dep = {'mask': num_dep_tokens}
        self.seq_tokens = list(range(num_dep_tokens))
        self.example_len = len_ex

    def generate_sequence(self, length, start_value, step_probability):
        sequence = [start_value]
        current_value = start_value

        for _ in range(length - 1):
            if random.random() < step_probability:
                current_value += 1
            sequence.append(current_value)
        return sequence

    def create_dataset_task_dep(self, num_examples, mask_probability=0.15, masking='train', elastic=True, step_prob=0.90) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        assert self.example_len % 2 == 0, "example len must be even"
        seq_len = self.example_len // 2
        
        dataset = []
        labels = []
        alt_labels = []
        for _ in range(num_examples):
            rand_val = random.random()
            start_index = random.randint(0, len(self.seq_tokens) - seq_len)
            if elastic:
                seq = self.generate_sequence(seq_len, self.seq_tokens[start_index], step_prob)  # Can have repeats
            else:
                seq = self.seq_tokens[start_index:start_index + seq_len]

            seq, seq_alt = self._modify_sequences(rand_val, seq, seq_len, elastic)

            label_seq, alt_labels_seq = self._apply_masking(seq, seq_alt, masking, mask_probability, seq_len)

            dataset.append(seq)
            labels.append(label_seq)
            alt_labels.append(alt_labels_seq)
            
        return dataset, labels, alt_labels

    def _modify_sequences(self, rand_val, seq, seq_len, elastic):
        seq_alt = seq.copy()
        if rand_val < 0.80:
            seq *= 2
            seq_alt *= 2
        else:
            change_ind = random.choice(range(2, seq_len + 1)) if elastic else -1
            seq, seq_alt = self._swap_and_repeat(seq, seq_alt, change_ind)
        return seq, seq_alt

    def _swap_and_repeat(self, seq, seq_alt, change_ind):
        if change_ind != -1:
            seq[-change_ind + 1], seq[-change_ind] = seq[-change_ind], seq[-change_ind + 1]
            seq *= 2
            seq_alt *= 2
            seq_alt[-change_ind + 1] = seq_alt[-change_ind]
        else:
            seq[-1], seq[-2] = seq[-2], seq[-1]
            seq *= 2
            seq_alt *= 2
            seq_alt[-1] = seq_alt[-2]
        return seq, seq_alt

    def _apply_masking(self, seq, seq_alt, masking, mask_probability, seq_len):
        label_seq = seq.copy()
        alt_labels_seq = seq_alt.copy()
        if masking == 'train':
            for i in range(len(seq)):
                if random.random() < mask_probability:
                    seq[i] = self.special_token_dict_dep['mask']
                else:
                    label_seq[i] = -100  # Ignore in loss function
                    alt_labels_seq[i] = -100
        else:
            for i in range(len(seq)):
                if i >= len(seq) - seq_len:
                    seq[i] = self.special_token_dict_dep['mask']
                else:
                    label_seq[i] = -100
                    alt_labels_seq[i] = -100
        return label_seq, alt_labels_seq
