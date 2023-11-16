from transformers import AutoTokenizer, BertModel

import torch
from datasets import load_dataset

import os
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import sys
import argparse
import json
import yaml

sys.path.append('/users/sanand14/data/sanand14/learning_dynamics/src/experiments/utils')
sys.path.append('/users/sanand14/data/sanand14/learning_dynamics/src/experiments')
from utils.probing_utils import Probe
from utils.data_utils import generate_activations

def main(config):
  home = os.environ['LEARNING_DYNAMICS_HOME']
  if "device" not in config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
  else:
    device = config["device"]
    
  model_config = config["model"].split('/')[-1]
  tokenizer = AutoTokenizer.from_pretrained(model_config["name"], use_fast=True)
  model = BertModel.from_pretrained(model_config["name"]).to(device)
  model.eval()

  layers_to_save = ['embeddings'] + [f'encoder.layer.{i}' for i in range(model_config["num_hidden_layers"])]  
  dataset = load_dataset("conll2003")  
  
  relevant_activations, task_labels = generate_activations(model, tokenizer, 
                                                          dataset, device, 
                                                          split='train', task=config["experiment"]) #pos, chunk, ner
  relevant_activations_val, task_labels_val = generate_activations(model, tokenizer, 
                                                                  dataset, device, 
                                                                  split='validation', task=config["experiment"])
  
  probe_config = config["probe"]
  val_logs = {}
  for i, layer in enumerate(layers_to_save):
    print('Training probe for layer', layer)
    num_labels = task_labels.max() + 1
    input_dim = relevant_activations[i].shape[-1]
    
    probe = Probe(input_dim, num_labels, probe_config["finetune_model"]).to(device)
    
    train_dataset = TensorDataset(relevant_activations[i].detach(), task_labels.view(-1, 1))
    val_dataset = TensorDataset(relevant_activations_val[i].detach(), task_labels_val.view(-1, 1))

    train_dataloader = DataLoader(train_dataset, batch_size=int(probe_config["batch_size"]), shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=int(probe_config["batch_size"]), shuffle=False)
    
    trainer = Trainer(max_epochs=probe_config["epochs"]) ## start w/ 1 epoch
    trainer.fit(probe, train_dataloader, val_dataloader)

    val_logs[layer] = trainer.validate(probe, val_dataloader)
    
  output_dir = os.path.join(home, probe_config["output_dir"])
  os.makedirs(output_dir, exist_ok=True)
  json.dump(val_logs, open(os.path.join(output_dir, "layer_val_acc.json"), "w"))
  ## f"outputs/task/results/probe-{config['finetune_model']}-step={config.step}-seed={config.seed}.json"
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", required=True, type=str, help="path to config file")
  args = parser.parse_args()
  
  config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
  main(config)
