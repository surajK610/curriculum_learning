from collections import defaultdict, Counter
from dataclasses import dataclass, field
import logging
from typing import cast, Dict, List, Tuple, Union, Callable, Set
from typing_extensions import get_args, Literal
import sys
import os
import numpy as np
import random 
import seaborn as sns
from PIL import Image
import re
import itertools

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

from data_utils import (loadText, get_observation_class, 
                        load_conll_dataset, embedBertObservation, 
                        ObservationIterator, match_tokenized_to_untokenized)
from task import ParseDistanceTask, ParseDepthTask, CPosTask, FPosTask, DepTask, NerTask, PhrStartTask, PhrEndTask
from utils.forgetting_utils import AdamEF
from utils.toy_utils import bin_train_loop, bin_val_loop, create_dataloaders_bin, Probe, POSVocabGenerator
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
from transformers import BertConfig, BertModel, AdamW, BertTokenizer


@dataclass
class PTBVocab:
    train_ptb : List  = field(default_factory=list)
    test_ptb : List  = field(default_factory=list)
    vocab_size: int = 0
    vocab: Dict[str, Tuple[float, Dict[str, float]]] = field(default_factory=dict)
    nouns: Set[Tuple[str, int]] = field(default_factory=set)
    adjs: Set[Tuple[str, int]] = field(default_factory=set)
    nonce_words : Set[str] = field(default_factory=set)
    
    def __len__(self):
        return self.vocab_size
    
    def __getitem__(self, word):
        return self.vocab.get(word, (0, {}))
    
    def __contains__(self, word):
        return word in self.vocab
    
    def add_word(self, word, pos):
        self.vocab_size += 1
        tuple_word_info = self.vocab.get(word, (0, {}))
        count = tuple_word_info[0] + 1
        pos_dict = tuple_word_info[1]
        pos_dict[pos] = pos_dict.get(pos, 0) + 1
        self.vocab[word] = (count, pos_dict)
    
    def populate_pos_categories(self, proportion_pos: float = 0.8):
        for word, (count, pos_dict) in self.vocab.items():
            if pos_dict.get('NOUN', 0)/count > proportion_pos:
                self.nouns.add((word, count))
            if pos_dict.get('ADJ', 0)/count > proportion_pos:
                self.adjs.add((word, count))
                
    def _get_n(self, n: Union[int, float], sorted_set : List[Tuple[str, int]]):
        assert n > 0, "n must be a positive proportion or integer"
        if n < 1:
            return tuple(zip(*sorted_set[:n*len(sorted_set)]))[0]
        else:
            assert n <= len(sorted_set), "n must be less than length set"
            n = int(n)
            return tuple(zip(*sorted_set[:n]))[0]
        
    def get_nouns_tail(self, n : Union[int, float] = 1000):
        assert n > 0, "n must be a positive proportion or integer"
        sorted_nouns = sorted(self.nouns, key=lambda x: x[1])
        return self._get_n(n, sorted_nouns)
    
    def get_nouns_head(self, n : Union[int, float] = 1000):
        assert n > 0, "n must be a positive proportion or integer"
        sorted_nouns = sorted(self.nouns, key=lambda x: x[1], reverse=True)
        return self._get_n(n, sorted_nouns)
    
    def get_adjs_tail(self, n : Union[int, float] = 1000):
        assert n > 0, "n must be a positive proportion or integer"
        sorted_nouns = sorted(self.adjs, key=lambda x: x[1])
        return self._get_n(n, sorted_nouns)
    
    def get_adjs_head(self, n : Union[int, float] = 1000):
        assert n > 0, "n must be a positive proportion or integer"
        sorted_adjs = sorted(self.adjs, key=lambda x: x[1], reverse=True)
        return self._get_n(n, sorted_adjs)

@dataclass
class ICLBertProbing:
    model: BertModel
    tokenizer: BertTokenizer
    ptb_vocab : PTBVocab
    train_text : List[str]
    train_observations : List
    val_text : List
    val_observations : List[str]
    num_examples_test : int = 10000
    epochs: int = 3
    train_dataloader: DataLoader = None
    test_dataloader: Union[DataLoader, Dict[str, DataLoader], Tuple[DataLoader, ...]] = None
    device: str = "cpu"
    batch_size: int = 128
    nouns: List[int] = None
    adjs: List[int] = None
    
    def create_datasets(self, n=1500):
        tail_nouns = self.ptb_vocab.get_nouns_tail(n)
        head_nouns = self.ptb_vocab.get_nouns_head(n)
        tail_adjs = self.ptb_vocab.get_adjs_tail(n)
        head_adjs = self.ptb_vocab.get_adjs_head(n)
        nonce_tokens = self.ptb_vocab.nonce_words
        
        train_text, train_observations = self.train_text, self.train_observations
        val_text, val_observations = self.val_text[:self.num_examples_test], self.val_observations[:self.num_examples_test]
        tail_text, tail_observations = self.create_dataset(lambda: random.choice(tail_nouns), 
                                                      lambda: random.choice(tail_adjs))
        head_text, head_observations = self.create_dataset(lambda: random.choice(head_nouns), 
                                                      lambda: random.choice(head_adjs))
        switch_text, switch_observations = self.create_dataset(lambda: random.choice(head_adjs), 
                                                          lambda: random.choice(head_nouns))
        tail_switch_text, tail_switch_observations = self.create_dataset(lambda: random.choice(tail_adjs), 
                                                                    lambda: random.choice(tail_nouns))
        nonce_text, nonce_observations = self.create_dataset(lambda: random.choice(nonce_tokens), 
                                                        lambda: random.choice(nonce_tokens))
        
        self.train_dataloader = self.make_pos_dataloader(train_text, train_observations, shuffle=True)
        self.test_dataloader = {
            'val': self.make_pos_dataloader(val_text, val_observations, shuffle=False), 
            'tail': self.make_pos_dataloader(tail_text, tail_observations, shuffle=False), 
            'head': self.make_pos_dataloader(head_text, head_observations, shuffle=False), 
            'switch': self.make_pos_dataloader(switch_text, switch_observations, shuffle=False), 
            'tail_switch': self.make_pos_dataloader(tail_switch_text, tail_switch_observations, shuffle=False), 
            'heldout': self.make_pos_dataloader(nonce_text, nonce_observations, shuffle=False), 
        }
        
    def create_dataset(self, sample_noun, sample_adj):
        text_dataset = []
        observation_dataset = []
        
        observation_fieldnames_test = [
            "sentence",
            "upos_sentence",
            "embeddings",
        ]
        observation_class_test = get_observation_class(observation_fieldnames_test)
        
        
        template = 'The {} is {}.'
        upos_format = ['DET', 'NOUN', 'VERB', 'ADJ', 'PUNCT']
        for _ in range(self.num_examples_test):
            curr_noun, curr_adj = sample_noun(), sample_adj()
            curr_sentence = template.format(curr_noun, curr_adj)
            curr_sentence_words = ('The', curr_noun, 'is', curr_adj, '.')
            
            curr_observation = observation_class_test(sentence=curr_sentence_words, upos_sentence=upos_format, embeddings=None)
            text_dataset.append(curr_sentence)
            observation_dataset.append(curr_observation)
        return text_dataset, observation_dataset
    
    # random.choice(tail_nouns), random.choice(tail_adjs)

    def make_pos_dataloader(self, text_dataset, observation_dataset, shuffle=True):
        outer_inputs, outer_labels = defaultdict(list), defaultdict(list)
        dataloaders = {}
        pos_to_label = {
            'NOUN': 0,
            'ADJ': 1,
        }
        self.model.eval()
        for idx in tqdm(range(len(text_dataset)), desc='[computing pos encodings]'):
            text = "[CLS] " + text_dataset[idx] + " [SEP]"
            observation = observation_dataset[idx]
            
            if 'NOUN' in observation.upos_sentence or 'ADJ' in observation.upos_sentence:
                indices, labels = find_pos_info(observation.upos_sentence)
                labels = [pos_to_label[val] for val in labels]
                
                untokenized_sent = observation.sentence
                tokenized_sent = self.tokenizer.wordpiece_tokenizer.tokenize(text)
                indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_sent)
                segment_ids = [1 for x in tokenized_sent]

                tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
                segments_tensors = torch.tensor([segment_ids]).to(self.device)
                with torch.no_grad():
                    encoded_layers = self.model(tokens_tensor, segments_tensors, output_hidden_states=True)
                encoded_layers = encoded_layers[-1]
                
                untokenized_sent = observation.sentence
                untok_tok_mapping = match_tokenized_to_untokenized(
                    tokenized_sent, untokenized_sent
                )
                for layer in range(len(encoded_layers)):
                    single_layer_features = encoded_layers[layer].squeeze()
                    
                    single_layer_features = torch.stack([
                                torch.mean(
                                    single_layer_features[
                                        untok_tok_mapping[i][0] : untok_tok_mapping[i][-1] + 1, :
                                    ],
                                    axis=0,
                                )
                                for i in indices])
                    outer_inputs[layer].append(single_layer_features)
                    outer_labels[layer].append(labels)
        for layer in outer_inputs.keys():
            outer_inputs[layer] = torch.vstack(outer_inputs[layer]).to(self.device)
            outer_labels[layer] = torch.tensor(list(itertools.chain(*outer_labels[layer]))).to(self.device)
            dataset = TensorDataset(outer_inputs[layer].detach(), outer_labels[layer].view(-1, 1).float())
            dataloaders[layer] = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=0)
        return dataloaders
        
    def conduct_probing(self):
        layer_results = {}
        for layer in self.train_dataloader.keys():
            input_shape = self.train_dataloader[layer].dataset.tensors[0].shape[1]
            probe = Probe(input_shape).to(self.device)
            optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3)
            bin_train_loop(probe, self.train_dataloader[layer], self.test_dataloader['val'][layer], optimizer, self.epochs)
            results = {}
            for key, dataloader in self.test_dataloader.items():
                results[key] = bin_val_loop(probe, dataloader[layer])['acc']
            layer_results[layer] = results
        return layer_results
    

def find_pos_info(input_tuple):
    indices = []
    elements = []

    for index, element in enumerate(input_tuple):
        if element in ('ADJ', 'NOUN'):
            indices.append(index)
            elements.append(element)
            
    return indices, elements

def plot_hist_pos(ptb_vocab, type='noun', n = 2000):
    if type == 'noun':
        values, counts = list(zip(*ptb_vocab.nouns))
    else:
        values, counts = list(zip(*ptb_vocab.adjs))
    idx_sort = np.argsort(counts)[::-1][:n]
    values, counts = np.array(values)[idx_sort], np.array(counts)[idx_sort]
    plt.bar(values, counts)
    plt.xticks([]) 
    plt.title(f'Distribution of {type}')

def main(args):
    home = os.environ["LEARNING_DYNAMICS_HOME"]
    output_dir = os.path.join(home, args.output_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = args.model_name #"bert-base-cased"
    save_model_name = model_name.split('/')[-1]
    os.makedirs(os.path.join(output_dir, save_model_name), exist_ok=True)
    
    model = BertModel.from_pretrained(model_name).to(device)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    num_nonce_words = args.num_nonce
    nonce_tokens = [f'thenewtoken{i}' for i in range(num_nonce_words)]
    tokenizer.add_tokens(nonce_tokens)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=16)
    
    data_path = os.path.join(home, "data/ptb_3")
    train_data_path = os.path.join(data_path, "ptb3-wsj-train.conllx")
    dev_data_path = os.path.join(data_path, "ptb3-wsj-dev.conllx")
    
    train_text = loadText(train_data_path)
    dev_text = loadText(dev_data_path)
    
    observation_fieldnames = [
    "index",
    "sentence",
    "lemma_sentence",
    "upos_sentence",
    "xpos_sentence",
    "morph",
    "head_indices",
    "governance_relations",
    "secondary_relations",
    "extra_info",
    "embeddings",
    ]

    observation_class = get_observation_class(observation_fieldnames)
    train_observations = load_conll_dataset(train_data_path, observation_class)
    dev_observations = load_conll_dataset(dev_data_path, observation_class)
    
    ptb_vocab = PTBVocab(train_observations, dev_observations)
    ptb_vocab.nonce_words = nonce_tokens
    for observation in tqdm(ptb_vocab.train_ptb, desc="[adding nouns & adjs to vocab]"):
        indices, elements = find_pos_info(observation.upos_sentence)
        for idx, pos in zip(indices, elements):
            ptb_vocab.add_word(observation.sentence[idx], pos)

    ptb_vocab.populate_pos_categories()
    
    probing = ICLBertProbing(model=model, 
                             tokenizer=tokenizer, 
                             ptb_vocab=ptb_vocab, 
                             train_text=train_text, 
                             train_observations=train_observations,
                             val_text=dev_text, 
                             val_observations=dev_observations, 
                             epochs=args.epochs,
                             device=device)
    probing.create_datasets(n=1500)
    layer_results = probing.conduct_probing()
    for layer in layer_results.keys():
        layer_str = "layer-" + str(layer)
        os.makedirs(os.path.join(output_dir, save_model_name, layer_str), exist_ok=True)
        with open(os.path.join(output_dir, save_model_name, layer_str, "val_acc.txt"), "w") as f:
            f.write(str(layer_results[layer]))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on a toy task')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_examples_test', type=int, default=10000, help='Number of examples in test set')
    parser.add_argument('--num_nonce', type=int, default=1500, help='Number of random examples')
    parser.add_argument('--num_samples', type=int, default=1500, help='Number of samples for head/tail sampling')
    parser.add_argument('--model_name', type=str, default='bert-base-cased', help='Model name')
    parser.add_argument('--output_dir', type=str, default='outputs/icl_bert', help='Output directory')
    args = parser.parse_args()
    main(args)