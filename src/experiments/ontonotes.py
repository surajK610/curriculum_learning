'''
Exactly the same script run as en_ewt-ud.py, so reusing that code w/
just file changed to ontonotes.py so that naming convention holds
'''
import torch

import os
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import sys
import argparse
import numpy as np
import yaml
import dill

sys.path.append('/users/sanand14/data/sanand14/learning_dynamics/src/experiments/utils')
sys.path.append('/users/sanand14/data/sanand14/learning_dynamics/src/experiments')
from utils.probing_utils import AccuracyProbe
from utils.data_utils import custom_pad_pos
from en_ewt_ud import main

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", required=True, type=str, help="path to config file")
  args = parser.parse_args()
  
  config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
  main(config)