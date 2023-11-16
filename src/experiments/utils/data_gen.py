import sys
import os
from collections import namedtuple, defaultdict

import torch as th
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

import numpy as np

import dill
import h5py
from tqdm import tqdm

sys.path.append('/users/sanand14/data/sanand14/learning_dynamics/src/experiments/utils')
from data_utils import loadText, saveBertHDF5, get_observation_class, load_conll_dataset, embedBertObservation, ObservationIterator
from task import ParseDistanceTask, ParseDepthTask, CPosTask, FPosTask, DepTask

import argparse

def main(args):
    home = os.environ["LEARNING_DYNAMICS_HOME"]
    # home = '.'
    if args.dataset == "ptb":
        data_path = os.path.join(home, "data/ptb_3")
        train_data_path = os.path.join(data_path, "ptb3-wsj-train.conllx")
        dev_data_path = os.path.join(data_path, "ptb3-wsj-dev.conllx")
        test_data_path = os.path.join(data_path, "ptb3-wsj-test.conllx")
    elif args.dataset == "ewt":
        data_path = os.path.join(home, "data/en_ewt-ud/")
        train_data_path = os.path.join(data_path, "en_ewt-ud-train.conllu")
        dev_data_path = os.path.join(data_path, "en_ewt-ud-dev.conllu")
        test_data_path = os.path.join(data_path, "en_ewt-ud-test.conllu")
    else:
        raise ValueError("Unknown dataset: " + args.dataset)
    
    model_name = args.model_name #"bert-base-cased"
    save_model_name = model_name.split('/')[-1]
    os.makedirs(os.path.join(data_path, "embeddings", save_model_name), exist_ok=True)
    train_hdf5_path = os.path.join(data_path, "embeddings", save_model_name,  "raw.train.layers.hdf5")
    dev_hdf5_path = os.path.join(data_path, "embeddings", save_model_name,  "raw.dev.layers.hdf5")
    test_hdf5_path = os.path.join(data_path, "embeddings", save_model_name, "raw.test.layers.hdf5")
    layer_index = args.layer_index #7
    task_name = args.task_name #"distance"
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    os.makedirs(os.path.join(data_path, "dataset", task_name, save_model_name), exist_ok=True)
    
    train_dataset_path = os.path.join(data_path,
        "dataset/",
        task_name,
        save_model_name,
        "train-layer-" + str(layer_index) + ".pt",
    )
    dev_dataset_path = os.path.join(data_path,
        "dataset/",
        task_name,
        save_model_name,
        "dev-layer-" + str(layer_index) + ".pt",
    )
    test_dataset_path = os.path.join(data_path,
        "dataset/",
        task_name,
        save_model_name,
        "test-layer-" + str(layer_index) + ".pt",
    )
    
    train_text = loadText(train_data_path)
    dev_text = loadText(dev_data_path)
    test_text = loadText(test_data_path)

    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert = BertModel.from_pretrained(model_name)
    if "cuda" in device.type:
        bert.cuda()
    bert.eval()

    LAYER_COUNT = 13
    FEATURE_COUNT = 768

    # NOTE: only call these functions once 
    if args.compute_embeddings == "True":
        print(train_hdf5_path)
        saveBertHDF5(train_hdf5_path, train_text, tokenizer, bert, LAYER_COUNT, FEATURE_COUNT, device=device)
        saveBertHDF5(dev_hdf5_path, dev_text, tokenizer, bert, LAYER_COUNT, FEATURE_COUNT, device=device)
        saveBertHDF5(test_hdf5_path, test_text, tokenizer, bert, LAYER_COUNT, FEATURE_COUNT, device=device)

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
    test_observations = load_conll_dataset(test_data_path, observation_class)

    train_observations = embedBertObservation(
        train_hdf5_path, train_observations, tokenizer, observation_class, layer_index
    )
    dev_observations = embedBertObservation(
        dev_hdf5_path, dev_observations, tokenizer, observation_class, layer_index
    )
    test_observations = embedBertObservation(
        test_hdf5_path, test_observations, tokenizer, observation_class, layer_index
    )

    if task_name == "distance":
        task = ParseDistanceTask()
    elif task_name == "depth":
        task = ParseDepthTask()
    elif task_name == "cpos":
        task = CPosTask()
    elif task_name == "fpos":
        task = FPosTask()
    elif task_name == "dep":
        task = DepTask()
    else:
        raise ValueError("Unknown task name: " + task_name)
    
    train_dataset = ObservationIterator(train_observations, task)
    dev_dataset = ObservationIterator(dev_observations, task)
    test_dataset = ObservationIterator(test_observations, task)
    
    th.save(train_dataset, train_dataset_path, pickle_module=dill)
    th.save(dev_dataset, dev_dataset_path, pickle_module=dill)
    th.save(test_dataset, test_dataset_path, pickle_module=dill)
    
    
if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--model-name", default="bert-base-cased", type=str)
    argp.add_argument("--layer-index", default=7, type=int)
    argp.add_argument("--task-name", default="distance", type=str)
    argp.add_argument("--dataset", default="ptb", type=str)
    argp.add_argument("--compute-embeddings", default="False", type=str)
    args = argp.parse_args()
    main(args)