import sys
import os
from collections import namedtuple, defaultdict

import torch as th
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, GPTNeoXForCausalLM, AutoTokenizer

import numpy as np

import dill
import h5py
from tqdm import tqdm
from datasets import load_dataset

sys.path.append('/users/sanand14/data/sanand14/learning_dynamics/src/experiments/utils')
from data_utils import loadText, loadTextOntonotes, savePythiaHDF5, saveBertHDF5, get_observation_class, load_conll_dataset, embedPythiaObservation, embedBertObservation, ObservationIterator, read_onto_notes_format
from task import ParseDistanceTask, ParseDepthTask, CPosTask, FPosTask, DepTask, NerTask, PhrStartTask, PhrEndTask

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
    elif args.dataset == "ontonotes":
        data_path = os.path.join(home, "data/ontonotes/")
        train_data_path = os.path.join(data_path, "conll-2012/v4/data/train/data/english/annotations/")
        dev_data_path = os.path.join(data_path, "conll-2012/v4/data/development/data/english/annotations/")
        test_data_path = os.path.join(data_path, "conll-2012/v4/data/development/data/english/annotations/")
    else:
        raise ValueError("Unknown dataset: " + args.dataset)
    
    model_name = args.model_name #"bert-base-cased"
    save_model_name = model_name.split('/')[-1]
    resid = args.resid == "True"
    if "pythia" in model_name:
        save_model_name += "-step" + str(args.model_step)
    os.makedirs(os.path.join(data_path, "embeddings", save_model_name), exist_ok=True)
    if resid:
        train_hdf5_path = os.path.join(data_path, "embeddings", save_model_name,  "raw.train.layers.hdf5")
        dev_hdf5_path = os.path.join(data_path, "embeddings", save_model_name,  "raw.dev.layers.hdf5")
        test_hdf5_path = os.path.join(data_path, "embeddings", save_model_name, "raw.test.layers.hdf5")
    else:
        train_hdf5_path = os.path.join(data_path, "embeddings", save_model_name,  "raw.out.train.layers.hdf5")
        dev_hdf5_path = os.path.join(data_path, "embeddings", save_model_name,  "raw.out.dev.layers.hdf5")
        test_hdf5_path = os.path.join(data_path, "embeddings", save_model_name, "raw.out.test.layers.hdf5")
        
    layer_index = args.layer_index #7
    attention_head = None
    if args.attention_head is not None:
        attention_head = int(args.attention_head)
    task_name = args.task_name #"distance"
    
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    os.makedirs(os.path.join(data_path, "dataset", task_name, save_model_name), exist_ok=True)
    
    train_dataset_path = os.path.join(data_path,
        "dataset/",
        task_name,
        save_model_name,
       f"train-layer-{layer_index}{'-'+str(attention_head) if attention_head is not None else ''}.pt",
    )
    dev_dataset_path = os.path.join(data_path,
        "dataset/",
        task_name,
        save_model_name,
        f"dev-layer-{layer_index}{'-'+str(attention_head) if attention_head is not None else ''}.pt",
    )
    test_dataset_path = os.path.join(data_path,
        "dataset/",
        task_name,
        save_model_name,
        f"test-layer-{layer_index}{'-'+str(attention_head) if attention_head is not None else ''}.pt",
    )

    if args.dataset == "ontonotes":
        train_text = loadTextOntonotes(train_data_path)
        dev_text = loadTextOntonotes(dev_data_path)
        test_text = loadTextOntonotes(test_data_path)
    else:    
        train_text = loadText(train_data_path)
        dev_text = loadText(dev_data_path)
        test_text = loadText(test_data_path)
        
    if "bert" in model_name:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
    elif "pythia" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision="step"+str(args.model_step), add_prefix_space=True) 
        ## maybe change first token later, but ok for now
        model = GPTNeoXForCausalLM.from_pretrained(model_name,revision="step"+str(args.model_step)) 
    else:
        raise ValueError("")
    
    if "cuda" in device.type:
        model.cuda()
    model.eval()

    LAYER_COUNT = 13
    FEATURE_COUNT = 768

    # NOTE: only call these functions once 
    if args.compute_embeddings == "True":
        print(train_hdf5_path)
        if "bert" in model_name:
            saveBertHDF5(train_hdf5_path, train_text, tokenizer, model, LAYER_COUNT, FEATURE_COUNT, device=device, resid=resid)
            saveBertHDF5(dev_hdf5_path, dev_text, tokenizer, model, LAYER_COUNT, FEATURE_COUNT, device=device, resid=resid)
            saveBertHDF5(test_hdf5_path, test_text, tokenizer, model, LAYER_COUNT, FEATURE_COUNT, device=device, resid=resid)
        elif "pythia" in model_name:
            savePythiaHDF5(train_hdf5_path, train_text, tokenizer, model, LAYER_COUNT, FEATURE_COUNT, device=device)
            savePythiaHDF5(dev_hdf5_path, dev_text, tokenizer, model, LAYER_COUNT, FEATURE_COUNT, device=device)
            savePythiaHDF5(test_hdf5_path, test_text, tokenizer, model, LAYER_COUNT, FEATURE_COUNT, device=device)
        else:
            raise ValueError("")
        
    if args.dataset == "ontonotes":
        observation_fieldnames = [
            "index",
            "sentence",
            "ner",
            "phrase_start",
            "phrase_end",
            "np_start",
            "np_end",
            "embeddings",
        ]
        observation_class = get_observation_class(observation_fieldnames)
        train_observations = read_onto_notes_format(train_data_path, observation_class)
        dev_observations = read_onto_notes_format(dev_data_path, observation_class)
        test_observations = read_onto_notes_format(test_data_path, observation_class)
    else:
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

    if "bert" in model_name:
        train_observations = embedBertObservation(
            model, train_hdf5_path, train_observations, tokenizer, observation_class, layer_index, attention_head
        )
        dev_observations = embedBertObservation(
            model, dev_hdf5_path, dev_observations, tokenizer, observation_class, layer_index, attention_head
        )
        test_observations = embedBertObservation(
            model, test_hdf5_path, test_observations, tokenizer, observation_class, layer_index, attention_head
        )
    elif "pythia" in model_name:
        train_observations = embedPythiaObservation(
            train_hdf5_path, train_observations, tokenizer, observation_class, layer_index
        )
        dev_observations = embedPythiaObservation(
            dev_hdf5_path, dev_observations, tokenizer, observation_class, layer_index
        )
        test_observations = embedPythiaObservation(
            test_hdf5_path, test_observations, tokenizer, observation_class, layer_index
        )
    else:
        raise ValueError("")     

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
    elif task_name == "ner":
        task = NerTask()
    elif task_name == "phrase_start":
        task = PhrStartTask()
    elif task_name == "phrase_end":
        task = PhrEndTask()
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
    argp.add_argument("--model-step", default=143000, type=int)
    argp.add_argument("--task-name", default="distance", type=str)
    argp.add_argument("--dataset", default="ptb", type=str)
    argp.add_argument("--compute-embeddings", default="False", type=str)
    argp.add_argument("--resid", default="True", type=str)
    argp.add_argument("--attention-head", default=None, type=str)
    args = argp.parse_args()
    main(args)