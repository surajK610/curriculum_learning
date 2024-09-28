from collections import defaultdict, Counter
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
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
    
import yaml
import argparse
import pandas as pd
from tqdm.notebook import tqdm
from functools import partial
from torch.utils.data import DataLoader, TensorDataset

sys.path.append('/users/sanand14/data/sanand14/learning_dynamics/src/experiments/utils')
sys.path.append('/users/sanand14/data/sanand14/learning_dynamics/src/experiments')
from utils.forgetting_utils import AdamEF
from utils.toy_utils import bin_train_loop, create_dataloaders_bin, Probe, POSVocabGenerator, plot_pca_embeddings
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
from transformers import BertConfig, BertForMaskedLM, AdamW


REINIT_RANDOM = True

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
    hist : Dict = field(default_factory=dict)
    probe_results : Dict = field(default_factory=dict)
    a : int = 1.5
    prop_amb : float = 0.0
    bins : int = 10
    sample_func : str = 'zipfian'
    extra_eval : bool = False
    top_twenty : bool = False
    amb_test : bool = False
    output_dir : str = None
    plot_pca : bool = False

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
                print(f"Training step {c_step}", flush=True)
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
                self._forget_embeddings(c_step)
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
    
    def pca_pos(self, val_dataloader, title, c_step, probe_results=None):
        if not hasattr(self, 'adj_min'):
            self.adj_min = min(self.vocab_gen.adj_tokens)
        if not hasattr(self, 'adj_max'):
            self.adj_max = max(self.vocab_gen.adj_tokens)
        logging.debug(f"running probing/PCA for step {c_step}...")
        probe_results = probe_results if probe_results is not None else defaultdict(list)
        labels, hidden_layers = [], defaultdict(list)
        
        def get_labels(examples, blabels):
            if torch.all(examples[:, -4] <= self.adj_max) and self.prop_amb == 0:
                assert ((examples[:, -4] < self.adj_min).float() == (examples[:, -4] != blabels[:, -3]).float()).all(), "Noun/Adj labels are not consistent"
        
            return (examples[:, -4] != blabels[:, -3]).float()
            # random tokens if noun if not equal, adj if equal (this is what the use is)
            
        # Enabling model to output hidden states
        self.model.config.output_hidden_states = True
        num_hidden_states = self.model.config.num_hidden_layers + 1
        
        for batch in val_dataloader:
            examples, blabels = batch
            
            labels.append(get_labels(examples, blabels))
            with torch.no_grad():
                outputs = self.model(examples.to(self.device))
            for j in range(num_hidden_states):
                hidden_layers[j].append(outputs.hidden_states[j][:, -4, :])
        labels = torch.concat(labels, axis=0).unsqueeze(1)

        # for i in range(num_hidden_states):
        i = 0
        torch_embed = torch.concat(hidden_layers[i], axis=0).squeeze()
        probe = Probe(torch_embed.shape[1]).to(self.device)
        train_dataloader_bin, val_dataloader_bin = create_dataloaders_bin(torch_embed, labels)
        optim_bin = torch.optim.AdamW(probe.parameters(), lr=1e-3)
        results = bin_train_loop(probe, train_dataloader_bin, val_dataloader_bin, optim_bin, 3)
        probe_results[i].append(results['acc'])
        
        if self.plot_pca:
            _, axs = plt.subplots(1, 1, figsize=(5 * 1, 5))
            self._plot_pca_results(axs, torch_embed, labels, results['acc'], i)
    
            plt.suptitle(title)
            if self.output_dir is not None:
                os.makedirs(self.output_dir, exist_ok=True)
                plt.savefig(os.path.join(self.output_dir, f'pca_step_{c_step}.png'))
            plt.show()
            plt.close()

        return probe_results
    
    def rank_freq_acc(self, test_dataloader):
        self.model.eval()
        train_query_counts = Counter(self.train_dataloader.dataset.tensors[0][:, -4].tolist())
        sample_func = lambda type: self.vocab_gen.uniform('noun') if type == 'adj' else self.vocab_gen.uniform('adj')
        inputs_t, labels_t = self.vocab_gen.create_dataset_task_pos(100000, sample_func=sample_func, device=self.device)
        test_dataset = TensorDataset(inputs_t.detach(), labels_t)
        new_test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        # -4 is the position of the query token
        # test_dataloader.batch_size = 1
        rank_freq_acc_entries= {}
        with torch.no_grad():
            pbar = tqdm(new_test_dataloader)
            for val_batch in pbar:
                _, stats = self.step(val_batch, hard_acc=True)
                curr_x, _ = val_batch
                query = curr_x.squeeze()[-4].item()
                # print(train_query_counts[query])
                if train_query_counts[query] not in rank_freq_acc_entries:
                    rank_freq_acc_entries[train_query_counts[query]] = []
                rank_freq_acc_entries[train_query_counts[query]].append(stats["acc"]) 
        return rank_freq_acc_entries
                
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
        # plt.savefig(f"pca_state_{state_index}.png")
        
    def _prepare_dataloaders(self):
        test_dataloader = {}
        
        sample_func = lambda type: self.vocab_gen.zipfian(type) 
        if self.sample_func == 'uniform':
            sample_func = lambda type: self.vocab_gen.uniform(type)
        inputs_t, labels_t = self.vocab_gen.create_dataset_task_pos(self.num_train, sample_func=sample_func, device=self.device)
        inputs_v, labels_v = self.vocab_gen.create_dataset_task_pos(self.num_val, sample_func=sample_func, device=self.device)
        inputs_s, labels_s = self.vocab_gen.create_dataset_task_pos(self.num_val, sample_func=sample_func, switch=True, device=self.device)
        # inputs_ho, labels_ho = self.vocab_gen.create_dataset_task_pos(self.num_val, sample_func=sample_func, holdout_once=2, device=self.device)
        inputs_h, labels_h = self.vocab_gen.create_dataset_task_pos(self.num_val, sample_func=sample_func, holdout=True, device=self.device)
        # inputs_sh, labels_sh = self.vocab_gen.create_dataset_task_pos(self.num_val, sample_func=sample_func, switch=True, holdout=True, device=self.device)
        
        
        train_dataset = TensorDataset(inputs_t.detach(), labels_t)
        val_dataset = TensorDataset(inputs_v.detach(), labels_v)
        switch_val_dataset = TensorDataset(inputs_s.detach(), labels_s)
        # holdout_once_val_dataset = TensorDataset(inputs_ho.detach(), labels_ho)
        holdout_val_dataset = TensorDataset(inputs_h.detach(), labels_h)
       
        # switch_holdout_val_dataset = TensorDataset(inputs_sh.detach(), labels_sh)
        # random_val_dataset = TensorDataset(inputs_r.detach(), labels_r)
        
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        switch_dataloader = DataLoader(switch_val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        # holdout_once_dataloader = DataLoader(holdout_once_val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        holdout_dataloader = DataLoader(holdout_val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
       
        # switch_holdout_dataloader = DataLoader(switch_holdout_val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        # random_dataloader = DataLoader(random_val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        # self.holdout_once_dataloader = holdout_once_dataloader
        
        test_dataloader.update({
            'val': val_dataloader,
            'switch': switch_dataloader, 
            'holdout': holdout_dataloader,
            # 'holdout_switch': switch_holdout_dataloader,
        })
        
        if self.sample_func == 'zipfian':
            inputs_e, labels_e = self.vocab_gen.create_dataset_task_pos(self.num_val, sample_func=sample_func, tail_end=True, device=self.device)
            inputs_st, labels_st = self.vocab_gen.create_dataset_task_pos(self.num_val, sample_func=sample_func, switch=True, tail_end=True, device=self.device)
            tail_switch_val_dataset = TensorDataset(inputs_st.detach(), labels_st)
            tail_end_val_dataset = TensorDataset(inputs_e.detach(), labels_e)
            tail_switch_dataloader = DataLoader(tail_switch_val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            tail_end_val_dataloader = DataLoader(tail_end_val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            test_dataloader['tail'] = tail_end_val_dataloader
            test_dataloader['tail_switch'] = tail_switch_dataloader
            
        if self.prop_amb > 0 and self.amb_test:
            inputs_na, labels_na = self.vocab_gen.create_dataset_task_pos(self.num_val, sample_func=sample_func, non_amb_only=True, device=self.device)
            inputs_am, labels_am = self.vocab_gen.create_dataset_task_pos(self.num_val, sample_func=sample_func, amb_only=True, device=self.device)
            nonamb_val_dataset = TensorDataset(inputs_na.detach(), labels_na)
            amb_val_dataset = TensorDataset(inputs_am.detach(), labels_am)
            nonamb_val_dataloader = DataLoader(nonamb_val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            amb_val_dataloader = DataLoader(amb_val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            test_dataloader['unif_nonamb'] = nonamb_val_dataloader
            test_dataloader['unif_amb'] = amb_val_dataloader
            
            if self.sample_func == 'zipfian':
                for cbin in range(self.vocab_gen.bins):
                    bin_amb, labels_am = self.vocab_gen.create_dataset_task_pos(self.num_val//self.vocab_gen.bins, sample_func=sample_func, cbin=cbin, amb_only=True, device=self.device)
                    bin_nonamb, labels_nam = self.vocab_gen.create_dataset_task_pos(self.num_val//self.vocab_gen.bins, sample_func=sample_func, cbin=cbin, non_amb_only=True, device=self.device)
                    bin_nonamb_s, labels_nam_s = self.vocab_gen.create_dataset_task_pos(self.num_val//self.vocab_gen.bins, sample_func=sample_func, cbin=cbin, non_amb_only=True, switch=True, device=self.device)
                    
                    bin_amb_ds = TensorDataset(bin_amb.detach(), labels_am)
                    bin_nonamb_ds = TensorDataset(bin_nonamb.detach(), labels_nam)
                    bin_nonamb_s_ds = TensorDataset(bin_nonamb_s.detach(), labels_nam_s)
                    
                    test_dataloader[f'bin_{cbin}_amb'] = DataLoader(bin_amb_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
                    test_dataloader[f'bin_{cbin}_nonamb'] = DataLoader(bin_nonamb_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
                    test_dataloader[f'bin_{cbin}_nonamb_switch'] = DataLoader(bin_nonamb_s_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        if self.top_twenty:
            inputs_tt, labels_tt = self.vocab_gen.create_dataset_task_pos(self.num_val, sample_func=sample_func, top_20=True, device=self.device)
            holdout_tt_dataset = TensorDataset(inputs_tt.detach(), labels_tt)
            holdout_tt_dataloader = DataLoader(holdout_tt_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            test_dataloader['top_twenty'] = holdout_tt_dataloader
            
            inputs_tts, labels_tts = self.vocab_gen.create_dataset_task_pos(self.num_val, sample_func=sample_func, top_20=True, switch=True, device=self.device)
            holdout_tts_dataset = TensorDataset(inputs_tts.detach(), labels_tts)
            holdout_tts_dataloader = DataLoader(holdout_tts_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            test_dataloader['top_twenty_switch'] = holdout_tts_dataloader
            
        if self.extra_eval:
            inputs_hu, labels_hu = self.vocab_gen.create_dataset_task_pos(self.num_val, sample_func=sample_func, holdout_unif=True, device=self.device)
            inputs_hl, labels_hl = self.vocab_gen.create_dataset_task_pos(self.num_val, sample_func=sample_func, holdout_larg=True, device=self.device)
            holdout_unif_dataset = TensorDataset(inputs_hu.detach(), labels_hu)
            holdout_larg_dataset = TensorDataset(inputs_hl.detach(), labels_hl)
            holdout_unif_dataloader = DataLoader(holdout_unif_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            holdout_larg_dataloader = DataLoader(holdout_larg_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            test_dataloader['holdout_unif'] = holdout_unif_dataloader
            test_dataloader['holdout_larg'] = holdout_larg_dataloader
            
            
        logging.debug("finished creating new dataloaders...")
        
        return train_dataloader, test_dataloader
    
    def _prepare_logging(self):
        logging.debug("preparing logging...")
        if isinstance(self.test_dataloader, DataLoader):
            self.test_dataloader = {'val': test_dataloader}
            self.hist = {'val': {}, 'val_loss':{}}
            self.probe_results = {'val': defaultdict(list)}
        elif isinstance(self.test_dataloader, dict):
            self.hist = {key: {} for key in self.test_dataloader.keys()} 
            self.hist['val_loss'] = {}
            self.probe_results = {key: defaultdict(list) for key in self.pca}
        elif isinstance(self.test_dataloader, tuple):
            if len(self.test_dataloader) == 3:
                test_dataloader, tail_end_val_dataloader, switch_val_dataloader = test_dataloader
                self.test_dataloader = {'val': test_dataloader, 'tail': tail_end_val_dataloader, 'switch': switch_val_dataloader}
                self.hist = {key: {} for key in ['val', 'tail', 'switch']}
                self.hist['val_loss'] = {}
                self.probe_results = {key: defaultdict(list) for key in ['val', 'tail', 'switch']}
            elif len(test_dataloader) == 2:
                test_dataloader, tail_end_val_dataloader = test_dataloader
                self.test_dataloader = {'val': test_dataloader, 'tail': tail_end_val_dataloader}
                self.hist = {key: {} for key in ['val', 'tail']}
                self.hist['val_loss'] = {}
                self.probe_results = {key: defaultdict(list) for key in ['val', 'tail']}
            else:
                raise ValueError('Not recognized format for test_dataloader order/length')
    
    def _forget_embeddings(self, c_step):
        if hasattr(self.optimizer, 'clear_embed_every_K_updates') and self.optimizer.clear_embed_every_K_updates > 0:
            if (isinstance(self.optimizer.clear_embed_every_K_updates, int) and c_step % self.optimizer.clear_embed_every_K_updates == 0):
                if self.optimizer.stop_after is None or (isinstance(self.optimizer.stop_after, int) and c_step < self.optimizer.stop_after):
                    print(f"Clearing embeddings at step {c_step}", flush=True)
                    self.model.bert.embeddings.word_embeddings.reset_parameters()
                    
    def _evaluate_during_training(self, c_step):
        logging.debug(f"evaluating during training step {c_step}...")
        if (isinstance(self.step_eval, int) and c_step % self.step_eval == 0) or (isinstance(self.step_eval, (list, tuple)) and c_step in self.step_eval):
            for key, dataloader in self.test_dataloader.items():
                print(f"Running validation for {key} at step {c_step}", flush=True)
                if key not in ["holdout", "holdout_switch", "random"]: ## these have their own eval
                    metrics = self.val_loop(dataloader)
                    self.hist[key][c_step] = metrics['acc']    
                    if key == 'val':
                        self.hist['val_loss'][c_step] = metrics['loss']
                if key in self.pca:
                    # print(f"Running PCA for {key} at step {c_step}", flush=True)
                    self.probe_results[key] = self.pca_pos(dataloader, f'Step {c_step}', c_step, probe_results=self.probe_results[key])
                if self.name:
                    logging.debug(f"saving model at step {c_step}...")
                    torch.save(self.model.state_dict(), f'models/{self.name}_step_{c_step}.pth')
            if "holdout" in self.test_dataloader:
                self._holdout_eval(c_step)
            if "random" in self.test_dataloader:
                self._random_eval(c_step)
                    
    def _holdout_eval(self, c_step):
        logging.debug(f"running random evaluation at step {c_step}...")
        if not hasattr(self, 'initial_holdout_embs'):
            self.initial_random_embs = self.model.bert.embeddings.word_embeddings.weight.data[self.vocab_gen.random_tokens].clone()
        if not hasattr(self, 'initial_holdout_unif_embs'):
            self.model.bert.embeddings.word_embeddings.weight.data[self.vocab_gen.random_tokens_unif] = self.model.bert.embeddings.word_embeddings.weight.data[self.vocab_gen.random_tokens_unif].uniform_()
            self.initial_random_unif_embs = self.model.bert.embeddings.word_embeddings.weight.data[self.vocab_gen.random_tokens_unif].clone()
        if not hasattr(self, 'initial_holdout_larg_embs'):
            self.model.bert.embeddings.word_embeddings.weight.data[self.vocab_gen.random_tokens_larg] = self.model.bert.embeddings.word_embeddings.weight.data[self.vocab_gen.random_tokens_larg].normal_(mean=5.0, std=5.0)
            self.initial_random_larg_embs = self.model.bert.embeddings.word_embeddings.weight.data[self.vocab_gen.random_tokens_larg].clone()
        ## random initialization of embeddings from init distribution
        self.model.bert.embeddings.word_embeddings.weight.data[self.vocab_gen.random_tokens] = self.initial_random_embs
        self.model.bert.embeddings.word_embeddings.weight.data[self.vocab_gen.random_tokens_unif] = self.initial_random_unif_embs
        self.model.bert.embeddings.word_embeddings.weight.data[self.vocab_gen.random_tokens_larg] = self.initial_random_larg_embs
        
        print('Mean large random', torch.mean(self.model.bert.embeddings.word_embeddings.weight.data[self.vocab_gen.random_tokens_larg]).item())
        # self.model.train()
        ## see everything once at least
        # for epoch in range(5):
        # for batch in self.holdout_once_dataloader:
        #     self.optimizer.zero_grad()
        #     loss, stats = self.step(batch, hard_acc=True)
        #     loss.backward()
        #     self.optimizer.step()
        self.hist['holdout'][c_step] = self.val_loop(self.test_dataloader['holdout'])['acc']
        if self.extra_eval:
            self.hist['holdout_unif'][c_step] = self.val_loop(self.test_dataloader['holdout_unif'])['acc']
            self.hist['holdout_larg'][c_step] = self.val_loop(self.test_dataloader['holdout_larg'])['acc']
        
        # self.hist['holdout_switch'][c_step] = self.val_loop(self.test_dataloader['holdout_switch'])['acc']

    def _random_eval(self, c_step):
        logging.debug(f"running random evaluation at step {c_step}...")
        for random_v in self.vocab_gen.random_tokens:
            ## weighted combination of current embedings
            values = np.random.choice(self.vocab_gen.noun_tokens + self.vocab_gen.adj_tokens, 10, replace=False).tolist()   
            self.model.bert.embeddings.word_embeddings.weight.data[random_v] = torch.mean(self.model.bert.embeddings.word_embeddings.weight.data[values], axis=0)
        
        self.hist['random'][c_step] = self.val_loop(self.test_dataloader['random'])

def main(args):
    ## SETTING SEED
    logging.basicConfig(filename=args.log, level=logging.DEBUG)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    ## SETTING UP PARAMETERS
    num_random = args.num_random if args.num_random is not None else args.vocab_size // 10
    
    # never forget if negative
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    ## SETTING UP TASK
    sample_func = args.sample_func
    if args.a == 0: ## just easier for me to write for loop
        sample_func = 'uniform'
    dset_gen = POSVocabGenerator()
    dset_gen.parameterize_pos_vocab(args.vocab_size, num_random, extra_eval=args.extra_eval, prop_amb=args.prop_amb, bins=args.bins, tail_only=False, a=args.a, sample_func=sample_func)
    
    ## SETTING UP MODEL
    config = BertConfig(
        vocab_size=dset_gen.get_vocab_tokens(), ## args.vocab_size+3 if have null token
        hidden_size=args.hidden_size, # 128  
        num_hidden_layers=args.hidden_num_layers, # 8
        num_attention_heads=args.num_attention_heads, # 8
        intermediate_size=args.intermediate_size, # 512
    )
    toy_bert_model = BertForMaskedLM(config).to(device)
    if args.forget_steps <= 0:
        print('Using AdamW optimizer', flush=True)
        optimizer = torch.optim.AdamW(toy_bert_model.parameters(), lr=5e-5, weight_decay=args.weight_decay) 
    else:
        print('Using AdamEF optimizer clearing every', args.forget_steps, 'stopping after', args.stop_forgetting_after, flush=True)
        optimizer = AdamEF(toy_bert_model.parameters(), lr=5e-5, lr_emb=5e-5, weight_decay=args.weight_decay, clear_embed_every_K_updates=args.forget_steps)
        optimizer.stop_after = args.stop_forgetting_after
        optimizer.embed_offset = list(filter(lambda p: p.requires_grad, toy_bert_model.parameters()))[0].numel()
    
    print(optimizer.state_dict()['param_groups'][0]['weight_decay'], flush=True)
    step_eval = 100
    # step_eval = list(range(0, 1000, 20)) + list(range(1000, 30000, 100))
    max_num_steps = args.dataset_size *  args.epochs/args.batch_size
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
        epochs=args.epochs,
        num_train=args.dataset_size,
        num_val=10_000,
        step_eval=step_eval,
        name=None, ## does not save model during training
        pca=['val', 'tail'],#['val', 'tail', 'random'],
        hist={},
        probe_results={},
        a=args.a,
        prop_amb=args.prop_amb,
        bins=args.bins,
        sample_func=sample_func,
        extra_eval=args.extra_eval, 
        top_twenty=args.top_twenty,
        amb_test=args.amb_test, 
        output_dir=args.output_dir,
        plot_pca=args.plot_pca
        )
        
    hist, probing_results = pipeline.train_loop()
    
    for key, val_dataloader in pipeline.test_dataloader.items():
        val_stats = pipeline.val_loop(val_dataloader)
        print(key, val_stats) # 10 - 80 identical, 10 - 20 1 token diff, 20 - 80 2 token diff
        
    pipeline._holdout_eval(int(max_num_steps))
        
    rank_freq_acc_entries = pipeline.rank_freq_acc(pipeline.test_dataloader['val'])
    print('saving rank freq acc entries...', flush=True)
    
    print(pipeline.hist, pipeline.probe_results)
    print('saving results...', flush=True)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    json.dump(rank_freq_acc_entries, open(os.path.join(args.output_dir, 'rank_freq_acc.json'), 'w'))
    
    # for key, hist_val in hist.items():
    hist_df = pd.DataFrame(hist)
    hist_df.to_csv(os.path.join(output_dir, f'hist.csv'))

    for key, probe_val in probing_results.items():
        df = pd.DataFrame(probe_val)
        df = df.transpose()
        if isinstance(step_eval, int):
            df.columns = [i*step_eval for i in range(len(df.columns))]
        else:
            df.columns = step_eval[:len(df.columns)]
        df = df[::-1]
        df.to_csv(os.path.join(output_dir, f'pos_probing_results_{key}.csv'))
        
        ax = sns.heatmap(df, annot=False)
        ax.set_xlabel("Step")
        ax.set_ylabel("Layer")
        ax.set_title(f"POS Probing {key}")
        plt.savefig(os.path.join(output_dir, f'pos_probing_steps_{key}.png'))
        plt.close()
        
    noun_embeddings = pipeline.model.bert.embeddings.word_embeddings.weight[pipeline.vocab_gen.noun_tokens]
    adj_embeddings = pipeline.model.bert.embeddings.word_embeddings.weight[pipeline.vocab_gen.adj_tokens]
    random_embeddings = pipeline.model.bert.embeddings.word_embeddings.weight[pipeline.vocab_gen.random_tokens]
    random_unif_embeddings = pipeline.model.bert.embeddings.word_embeddings.weight[pipeline.vocab_gen.random_tokens_unif]
    random_larg_embeddings = pipeline.model.bert.embeddings.word_embeddings.weight[pipeline.vocab_gen.random_tokens_larg]

    combined_tensors = torch.cat((noun_embeddings, adj_embeddings, random_embeddings, random_unif_embeddings, random_larg_embeddings), dim=0).detach().cpu()
    labels = torch.tensor([1]*noun_embeddings.shape[0] + [2]*adj_embeddings.shape[0] + [3]*random_embeddings.shape[0] + [4]*random_unif_embeddings.shape[0] + [5]*random_larg_embeddings.shape[0])
    # Convert tensors to numpy arrays
    embedding_split = TensorDataset(combined_tensors, labels)
    torch.save(embedding_split, os.path.join(output_dir, 'embedding_split.pth'))
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined_tensors.numpy())
    labels = labels.numpy()
    plot_pca_embeddings(pca_result, labels, os.path.join(output_dir, 'pca_embeddings.png'))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on a toy task')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--task', type=str, default="pos", help='Task to train on')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--step_eval', type=int, default=1000, help='How often to evaluate')
    parser.add_argument('--dataset_size', type=int, default=8_000_000, help='Size of the dataset')
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
    parser.add_argument('--prop_amb', type=float, default=0.0, help='Proportion of time noun or adj is switched in training')
    parser.add_argument('--bins', type=int, default=10, help='Number of bins for probing')
    parser.add_argument('--weight_decay', type=float, default=0.00, help='Weight decay for optimizer')
    parser.add_argument('--forget_steps', type=int, default=-1, help='Number of steps to forget')
    parser.add_argument('--stop_forgetting_after', type=int, default=None, help='Number of steps to stop forget')
    parser.add_argument('--extra_eval', action='store_true', help='Extra evaluation')
    parser.add_argument('--top_twenty', action='store_true', help='Top twenty evaluation')
    parser.add_argument('--amb_test', action='store_true', help='Ambiguity test')
    parser.add_argument('--plot_pca', action='store_true', help='Plot PCA')
    args = parser.parse_args()
    main(args)
    