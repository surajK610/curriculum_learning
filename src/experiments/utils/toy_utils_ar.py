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
from torch.utils.data import DataLoader, Dataset

sys.path.append('/users/sanand14/data/sanand14/learning_dynamics/src/experiments/utils')
sys.path.append('/users/sanand14/data/sanand14/learning_dynamics/src/experiments')

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import numpy as np
from dataclasses import dataclass, field


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

@dataclass
class ICLVocabGenerator:
    tokens: List[int] = field(default_factory=list)
    random_tokens: List[int] = field(default_factory=list)
    a: float = 1.5
    bins: int = 10
    sample_func: str = 'zipfian'
    token_label_map: dict = field(default_factory=dict)  # Mapping from tokens to labels
    label_tokens: dict = field(default_factory=dict)     # Tokens representing labels
    vocab_size: int = 0  # Total vocabulary size including special tokens

    def parameterize_icl_vocab(self, num_icl_tokens: int, num_random_tokens: int, bins=10, a=1.0001, sample_func='zipfian'):
        """
        Initialize the vocabulary and token-label mappings.
        """
        # Use integer tokens
        self.tokens = list(range(num_icl_tokens))
        self.random_tokens = list(range(num_icl_tokens, num_icl_tokens + num_random_tokens))
        self.a = a
        self.bins = bins
        self.sample_func = sample_func

        # Create token to label mapping (1:1 mapping)
        self.token_label_map = {}
        self.label_tokens = {}
        label_token_start = num_icl_tokens + num_random_tokens  # Starting index for label tokens
        for idx, token in enumerate(self.tokens):
            label = idx  # Unique label for each token
            label_token = label_token_start + idx
            self.token_label_map[token] = label
            self.label_tokens[label] = label_token
                ## multiplicity of possible labels
        
        # Reserve IDs for evaluation label tokens (0 and 1)
        self.eval_label_tokens = {
            0: label_token_start,
            1: label_token_start + 1
        }

        # Define special tokens
        self.eos_token_id = label_token_start + len(self.tokens)
        self.pad_token_id = label_token_start + len(self.tokens) + 1

        # Update total vocabulary size
        self.vocab_size = self.pad_token_id + 1  # Include all tokens up to PAD token

    def _uniform(self, tokens):
        return random.choice(tokens)

    def uniform(self):
        return self._uniform(self.tokens)

    def _zipfian(self, tokens):
        token_map = {k + 1: v for k, v in enumerate(tokens)}
        while True:
            value = np.random.zipf(self.a)
            if value in token_map:
                return token_map[value]

    def zipfian(self):
        return self._zipfian(self.tokens)
    
    def create_dataset_task_icl(self, num_examples: int, sample_func: Callable = None, bursty_ratio: float = 0.0, amb_ratio : float = 0.05) -> List[List[int]]:
        """
        Generate training dataset with sequences sampled from sample_func distribution.
        Each sequence ends with a query token and its label.
        In 'bursty' sequences, the query class appears 3 times in the context,
        and another class also appears 3 times.
        Parameters:
            num_examples (int): Total number of sequences to generate.
            sample_func (Callable): Function to sample tokens from the token distribution.
            bursty_ratio (float): Proportion of sequences that should be bursty (between 0 and 1).
        Returns:
            List[List[int]]: A list of sequences, each sequence is a list of token IDs.
        """
        dataset = []
        if sample_func is None:
            sample_func = self.zipfian  
    
        num_bursty = int(num_examples * bursty_ratio)
        num_non_bursty = num_examples - num_bursty
        # print(num_bursty)
        for _ in tqdm(range(num_bursty)):
            seq = []
            query_token = sample_func()
            other_tokens = [token for token in self.tokens if token != query_token]
            if not other_tokens:
                raise ValueError("Not enough tokens to create bursty sequences with two classes.")
            non_query_tokens = random.choices(other_tokens, k=3)
            token_counts = {query_token: 3, non_query_tokens[0]: 3, non_query_tokens[1]: 1, non_query_tokens[2]: 1}
            labels = {}
            for token in token_counts.keys():
                if random.random() > 1 - amb_ratio:
                    labels[token] = random.choice([0, 1])
                else:
                    labels[token] = self.token_label_map[token]
            # labels = {query_token: random.choice(self.token_label_map[query_token]), 
            #           non_query_tokens[0]: random.choice(self.token_label_map[non_query_tokens[0]]), 
            #           non_query_tokens[1]: random.choice(self.token_label_map[non_query_tokens[1]]), 
            #           non_query_tokens[2]: random.choice(self.token_label_map[non_query_tokens[2]])}

            context_examples = []
            for token, count in token_counts.items():
                label = labels[token]
                # print(label)
                label_token = self.label_tokens[label]
                for _ in range(count):
                    context_examples.append([token, label_token])
            random.shuffle(context_examples)
            for example in context_examples:
                seq.extend(example)
                
            query_label = labels[query_token]
            query_label_token = self.label_tokens[query_label]
            seq.extend([query_token, query_label_token])
            dataset.append(seq)
    
        for _ in range(num_non_bursty):
            seq = []
            for _ in range(8):
                token = sample_func()
                label = random.choice(self.token_label_map[token])
                label_token = self.label_tokens[label]
                seq.extend([token, label_token])
    
            query_token = sample_func()
            query_label = random.choice(self.token_label_map[query_token])
            query_label_token = self.label_tokens[query_label]
            seq.extend([query_token, query_label_token])
            dataset.append(seq)
    
        random.shuffle(dataset)
    
        return dataset

    def create_eval_dataset_in_context(self, num_examples: int, sample_func: Callable = None, holdout=False) -> List[List[int]]:
        """
        Generate evaluation dataset for in-context learning on holdout tokens.
        Each sequence consists of context examples from 2 holdout classes (tokens not seen during training),
        with 4 examples each (total 8 examples). Labels for the two classes are randomly assigned to 0 and 1
        for each sequence. The query is randomly selected from one of the two classes.
        """
        dataset = []
        if sample_func is None:
            sample_func = self.zipfian  
    
        if holdout:
            holdout_tokens = self.random_tokens.copy()  # Tokens not used during training
        else:
            holdout_tokens = self.tokens.copy()
    
        for _ in range(num_examples):
            seq = []
            if holdout:
                query_token = random.choice(holdout_tokens)
                other_tokens = [token for token in holdout_tokens if token != query_token]
            else:
                query_token = sample_func()
                other_tokens = [token for token in self.tokens if token != query_token]
            if not other_tokens:
                raise ValueError("Not enough tokens to create bursty sequences with two classes.")
            non_query_tokens = random.choices(other_tokens, k=3)
            token_counts = {query_token: 4, non_query_tokens[0]: 4, non_query_tokens[1]: 0, non_query_tokens[2]: 0}
            labels = {query_token: random.choice([0, 1]), 
                      non_query_tokens[0]: random.choice([0, 1]), 
                      non_query_tokens[1]: random.choice([0, 1]), 
                      non_query_tokens[2]: random.choice([0, 1])}

            context_examples = []
            for token, count in token_counts.items():
                label = labels[token]

                label_token = self.label_tokens[label]
                for _ in range(count):
                    context_examples.append([token, label_token])
            random.shuffle(context_examples)
            for example in context_examples:
                seq.extend(example)
                
            query_label = labels[query_token]
            query_label_token = self.label_tokens[query_label]
            seq.extend([query_token, query_label_token])
            dataset.append(seq)
        return dataset

    def create_eval_dataset_in_weights(self, num_examples: int) -> List[List[int]]:
        """
        Generate evaluation dataset for in-weights learning.
        Sequences have labels same as in training, but the query token does not appear in the context.
        The model must use information stored in weights to predict the query label.
        """
        dataset = []
        training_tokens = self.tokens
        for _ in range(num_examples):
            seq = []

            context_tokens = random.sample(training_tokens, k=8)
            query_token = random.choice([t for t in training_tokens if t not in context_tokens])
            for token in context_tokens:
                label = self.token_label_map[token]
                label_token = self.label_tokens[label]
                seq.extend([token, label_token])

            query_label = self.token_label_map[query_token]
            query_label_token = self.label_tokens[query_label]
            seq.extend([query_token, query_label_token])
            dataset.append(seq)
        return dataset
    
class CustomSequenceDataset(Dataset):
    def __init__(self, sequences, eos_token_id, pad_token_id, block_size=17):
        self.examples = []
        self.block_size = block_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        for seq in sequences:
            # Truncate or pad to block_size
            seq = seq[:self.block_size]  # Leave space for <EOS>
            # seq.append(self.eos_token_id)  # Append <EOS> token
            padding_length = self.block_size - len(seq)
            input_ids = seq + ([self.pad_token_id] * padding_length)  # Pad with <PAD> token

            # Prepare attention_mask
            attention_mask = [1] * len(seq) + [0] * padding_length

            # Prepare labels
            labels = [-100] * self.block_size
            # labels = seq[1:] + [self.eos_token_id] + (padding_length)*[-100]

            # Identify the position of the query label
            seq_length = len(seq)
            # print(seq_len)
            # if seq_length >= 3:
                # The position of the query label is seq_length - 2 (before EOS token)
            query_label_pos = seq_length - 2
            labels[query_label_pos] = input_ids[query_label_pos+1]

            self.examples.append({
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        item = self.examples[i]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'labels': torch.tensor(item['labels'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long)
        }
