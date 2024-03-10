from collections import defaultdict
from dataclasses import dataclass, field
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
from utils.toy_utils import bin_train_loop, create_dataloaders_bin, Probe, POSVocabGenerator
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
from transformers import BertConfig, BertForMaskedLM, AdamW

## ------------------------------------------ TRAINING ----------------------------------

@dataclass
class TrainingPipeline:
    model: torch.nn.Module
    vocab_gen: POSVocabGenerator
    criterion: Callable = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = None
    train_dataloader: DataLoader = None
    test_dataloader: Union[DataLoader, Dict[str, DataLoader], Tuple[DataLoader, ...]] = None
    device: str = "cpu"
    batch_size: int = 128
    epochs: int = 10
    num_train: int = 1_000_000
    num_val: int = 10_000
    step_eval: Union[int, List[int]] = 1000
    name: str = None
    pca: List[str] = field(default_factory=lambda: ['val', 'tail', 'random'])
    hist : Dict = {}
    probe_results : Dict = {}
    a : int = 1.5
    sample_func : str = 'zipfian'

    def step(self, batch, hard_acc=False):
        x, y = batch
        output = self.model.forward(x)
        logits = output.logits.transpose(1, 2)
        loss = self.criterion(logits, y)
        batch_len = logits.shape[0]
        
        where = (y != -100) # where y is masked
        y = y[where].view(batch_len, -1)
        preds = logits.argmax(axis=1)[where].view(batch_len, -1)
        
        if hard_acc:
            acc = (preds == y).all(axis=-1).float().mean() 
        else:
            acc = (preds == y).float().mean() 
        return loss, {"loss": loss.item(), "acc": acc.item()}
    
    def train_loop(self):
        if self.train_dataloader is None or self.test_dataloader is None:
            logging.debug("training/testing dataloader(s) not provided, creating new ones...")
            self.train_dataloader, self.test_dataloader = self._prepare_dataloaders()
            
        self._prepare_logging()
        
        pbar = tqdm(range(self.epochs))
        val_stats = {}
        c_step = 0
        for epoch in pbar:
            pbar.set_description(f"Training Epoch {epoch}")
            for batch in self.train_dataloader:
                sys.stdout.flush()
                sys.stderr.flush()
                c_step += 1
                self.model.train()
                self.optimizer.zero_grad()
                loss, stats = self.step(batch, hard_acc=True)
                loss.backward()
                self.optimizer.step()
                stats.update(val_stats)
                pbar.set_postfix(**stats)
                self._evaluate_during_training(c_step)
                
            self.model.eval()
            with torch.no_grad():
                pbar.set_description("Validation")
                val_stats = self.val_loop(self.test_dataloader)
                val_stats = {"val_" + key:val for key,val in val_stats.items()}
                pbar.set_postfix(**val_stats)
        return self.hist, self.probe_results
                
    def val_loop(self, test_dataloader):
        self.model.eval()
        acc, losses = [], []
        with torch.no_grad():
            pbar = tqdm(test_dataloader)
            for val_batch in pbar:
                _, stats = self.step(val_batch, hard_acc=True)
                acc.append(stats["acc"])
                losses.append(stats["loss"])
                results = {"acc": np.mean(acc), "loss": np.mean(losses)} #, "alt_acc": np.mean(acc_alt)}
                pbar.set_postfix(**results)
        return results
    
    def pca_pos(self, val_dataloader, title, c_step, output_dir=None, plot=False, probe_results=None):
        logging.debug(f"running probing/PCA for step {c_step}...")
        probe_results = probe_results if probe_results is not None else defaultdict(list)
        labels, hidden_layers = [], defaultdict(list)
        
        # Enabling model to output hidden states
        self.model.config.output_hidden_states = True
        num_hidden_states = self.model.config.num_hidden_layers + 1
            
        for batch in val_dataloader:
            examples, _ = batch
            labels.append((examples[:, -4] < self.adj_min).float())
            with torch.no_grad():
                outputs = self.model(examples.to(self.device))
            for j in range(num_hidden_states):
                hidden_layers[j].append(outputs.hidden_states[j][:, -4, :])

        labels = torch.concat(labels, axis=0).unsqueeze(1)

        for i in range(num_hidden_states):
            torch_embed = torch.concat(hidden_layers[i], axis=0).squeeze()
            probe = Probe(torch_embed.shape[1]).to(self.device)
            train_dataloader_bin, val_dataloader_bin = self.create_binary_dataloaders(torch_embed, labels)
            optim_bin = torch.optim.AdamW(probe.parameters(), lr=1e-3)
            results = self.binary_train_loop(probe, train_dataloader_bin, val_dataloader_bin, optim_bin, 3)
            probe_results[i].append(results['acc'])

            if plot:
                _, axs = plt.subplots(1, num_hidden_states, figsize=(5 * num_hidden_states, 5))
                self._plot_pca_results(axs[i], torch_embed, labels, results['acc'], i)
                
        if plot:
            plt.suptitle(title)
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, f'pca_step_{c_step}.png'))
            plt.show()
            plt.close()

        return probe_results

    def _plot_pca_results(self, ax, torch_embed, labels, acc, state_index):
        """
        Helper method to plot PCA results for a given hidden state.
        """
        logging.debug(f"plotting PCA results for hidden state {state_index}...")
        labels_numpy = labels.cpu().numpy().squeeze()
        np_embed = torch_embed.cpu().numpy()
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(np_embed)
        unique_labels = np.unique(labels_numpy)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

        for j, label in enumerate(unique_labels):
            ax.scatter(data_pca[labels_numpy == label, 0], data_pca[labels_numpy == label, 1],
                       alpha=0.5, color=colors[j], label=f"Label {label}")
        ax.set_title(f"State {state_index} acc={acc:0.2f}")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.legend()
        ax.grid(True)

    def _prepare_dataloaders(self):
        sample_func = lambda type: self.vocab_gen.zipfian(type, a=self.a) if self.sample_func == 'zipfian' else lambda type: self.vocab_gen.uniform(type)
        inputs_t, labels_t = self.vocab_gen.create_dataset_task_pos(self.num_train, sample_func=sample_func)
        inputs_v, labels_v = self.vocab_gen.create_dataset_task_pos(self.num_val, sample_func=sample_func)
        inputs_e, labels_e = self.vocab_gen.create_dataset_task_pos(self.num_val, sample_func=sample_func, tail_end=True)
        inputs_s, labels_s = self.vocab_gen.create_dataset_task_pos(self.num_val, sample_func=sample_func, switch=True)
        inputs_st, labels_st = self.vocab_gen.create_dataset_task_pos(self.num_val, sample_func=sample_func, switch=True, tail_end=True)
        inputs_r, labels_r = self.vocab_gen.create_dataset_task_pos(self.num_val, sample_func=sample_func, random=True)
        
        train_dataset = TensorDataset(inputs_t.detach(), labels_t)
        val_dataset = TensorDataset(inputs_v.detach(), labels_v)
        tail_end_val_dataset = TensorDataset(inputs_e.detach(), labels_e)
        switch_val_dataset = TensorDataset(inputs_s.detach(), labels_s)
        tail_switch_val_dataset = TensorDataset(inputs_st.detach(), labels_st)
        random_val_dataset = TensorDataset(inputs_r.detach(), labels_r)
        
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        tail_end_val_dataloader = DataLoader(tail_end_val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        switch_dataloader = DataLoader(switch_val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        tail_switch_dataloader = DataLoader(tail_switch_val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        random_dataloader = DataLoader(random_val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        test_dataloader = {
            'val': val_dataloader,
            'tail': tail_end_val_dataloader,
            'switch': switch_dataloader, 
            'tail_switch': tail_switch_dataloader,
            'random': random_dataloader
        }
        logging.debug("finished creating new dataloaders...")
        
        return train_dataloader, test_dataloader
    
    def _prepare_logging(self):
        logging.debug("preparing logging...")
        if isinstance(test_dataloader, dict):
            self.hist = {key: {} for key in test_dataloader.keys()}
            self.probe_results = {key: defaultdict(list) for key in test_dataloader.keys()}
        elif isinstance(test_dataloader, tuple):
            if len(test_dataloader) == 3:
                test_dataloader, tail_end_val_dataloader, switch_val_dataloader = test_dataloader
                test_dataloader = {'val': test_dataloader, 'tail': tail_end_val_dataloader, 'switch': switch_val_dataloader}
                self.hist = {key: {} for key in ['val', 'tail', 'switch']}
                self.probe_results = {key: defaultdict(list) for key in ['val', 'tail', 'switch']}
            elif len(test_dataloader) == 2:
                test_dataloader, tail_end_val_dataloader = test_dataloader
                test_dataloader = {'val': test_dataloader, 'tail': tail_end_val_dataloader}
                self.hist = {key: {} for key in ['val', 'tail']}
                self.probe_results = {key: defaultdict(list) for key in ['val', 'tail']}
            else:
                raise ValueError('Not recognized format for test_dataloader order/length')
            
    
    def _evaluate_during_training(self, c_step):
        logging.debug(f"evaluating during training step {c_step}...")
        if (isinstance(self.step_eval, int) and c_step % self.step_eval == 0) or (c_step in self.step_eval):
            for key, dataloader in self.test_dataloader.items():
                self.hist[key][c_step] = self.val_loop(dataloader)
                if key in self.pca:
                    self.probe_results[key] = self.pca_pos(dataloader, f'Step {c_step}', c_step, self.probe_results[key])
                if self.name:
                    logging.debug(f"saving model at step {c_step}...")
                    torch.save(self.model.state_dict(), f'models/{self.name}_step_{c_step}.pth')


def main(args):
    ## SETTING SEED
    logging.basicConfig(filename=args.log, level=logging.DEBUG)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    ## SETTING UP PARAMETERS
    num_random = args.num_random if args.num_random is not None else args.dataset_size // 10
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    ## SETTING UP TASK
    dset_gen = POSVocabGenerator()
    dset_gen.parameterize_pos_vocab(args.vocab_size, num_random)
    
    ## SETTING UP MODEL
    config = BertConfig(
        vocab_size=dset_gen.get_vocab_tokens(), ## args.vocab_size+3 if have null token
        hidden_size=args.hidden_size, # 128  
        num_hidden_layers=args.hidden_num_layers, # 8
        num_attention_heads=args.num_attention_heads, # 8
        intermediate_size=args.intermediate_size # 512
    )
    toy_bert_model = BertForMaskedLM(config).to(device)
    optimizer = torch.optim.AdamW(toy_bert_model.parameters(), lr=5e-5) 
    step_eval = list(range(0, 1000, 10)) + list(range(1000, 30000, 100))
    max_num_steps = args.dataset_size *  args.num_epochs/args.batch_size
    print('Max number of steps is ', max_num_steps, flush=True)
    pipeline = TrainingPipeline(
        model=toy_bert_model,
        vocab_gen=dset_gen,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        train_dataloader=None, ## set to none so will automatically set up
        test_dataloader=None, ## set to none so will automatically set up
        device=device,
        batch_size=args.batch_size,
        epochs=args.num_epochs,
        num_train=args.dataset_size,
        num_val=10_000,
        step_eval=step_eval,
        name=None, ## does not save model during training
        pca=['val', 'tail', 'random'],
        hist={},
        probe_results={},
        a=args.a,
        sample_func=args.sample_func)
        
    hist, probing_results = pipeline.train_loop()
    
    for key, val_dataloader in pipeline.test_dataloader.items():
        val_stats = pipeline.val_loop(val_dataloader)
        print(key, val_stats) # 10 - 80 identical, 10 - 20 1 token diff, 20 - 80 2 token diff
  
    print('saving results...', flush=True)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    for key, hist_val in hist.items():
        hist_df = pd.DataFrame(hist_val)
        hist_df.to_csv(os.path.join(output_dir, f'hist_{key}.csv'))
    
    for key, probe_val in probing_results.items():
        df = pd.DataFrame(probe_val)
        df = df.transpose()
        df.columns = step_eval[:len(df.columns)]
        df = df[::-1]
        df.to_csv(os.path.join(output_dir, f'pos_probing_results_{key}.csv'))
        
        ax = sns.heatmap(df, annot=False)
        ax.set_xlabel("Step")
        ax.set_ylabel("Layer")
        ax.set_title(f"POS Probing {key}")
        plt.savefig(os.path.join(output_dir, f'pos_probing_steps_{key}.png'))
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
    parser.add_argument('--num_random', type=int, default=None, help='Number of random examples')
    parser.add_argument('--log', type=str, default='toy_model.log', help='Log file')
    args = parser.parse_args()
    main(args)