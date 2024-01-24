import dill
from tqdm import tqdm
import os
import time
import pickle
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import spearmanr
import argparse
import yaml

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import Trainer

import sys
sys.path.append('/users/sanand14/data/sanand14/learning_dynamics/src/experiments/utils')
sys.path.append('/users/sanand14/data/sanand14/learning_dynamics/src/experiments')

from utils.data_utils import custom_pad
from utils.probing_utils import L1DistanceProbe, L1DepthProbe

def main(config):
    home = os.environ['LEARNING_DYNAMICS_HOME']
    if "device" not in config:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config["device"]
        
    dataset_config = config["dataset"]
    dataset_dir = dataset_config["dir"]
    
    layer_idx = config["layer_idx"]
    attention_head = config["attention_head"]
    # if layer_idx > 12:
    #   layer_idx = int(np.ceil((layer_idx - 12)/12))
    model_name = config["model_name"].split('/')[-1]
    resid = config["resid"]
    if "pythia" in model_name:
        model_name += "-step" + str(config["model_step"])
    
    probe_config = config["probe"]
    finetune_model = probe_config["finetune-model"]
    num_epochs = probe_config["epochs"]
    batch_size = probe_config["batch_size"]
    rep_dim = probe_config["rep_dim"]
    input_size = probe_config["input_size"]
    output_dir = os.path.join(home, probe_config["output_dir"])
    
    lr = float(probe_config["lr"])

    train_dataset_path = os.path.join(
        home,
        dataset_dir,
        model_name,
        f"train-layer-{layer_idx}{'-'+str(attention_head) if attention_head is not None else ''}.pt",
    )
    dev_dataset_path = os.path.join(
        home,
        dataset_dir,
        model_name,
        f"dev-layer-{layer_idx}{'-'+str(attention_head) if attention_head is not None else ''}.pt",
    )

    train_dataset = torch.load(train_dataset_path, pickle_module=dill)
    dev_dataset = torch.load(dev_dataset_path, pickle_module=dill)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: custom_pad(x, return_all=True),
    )
    dev_data_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn= lambda x: custom_pad(x, return_all=True),
    )
    layer_name = "layer-" + str(layer_idx)
    os.makedirs(os.path.join(output_dir, model_name, layer_name), exist_ok=True)
    if config["experiment"] == "distance":
        probe = L1DistanceProbe(input_size, rep_dim, finetune_model).to(device)
        trainer = Trainer(max_epochs=num_epochs)
        trainer.fit(probe, train_data_loader, dev_data_loader)
        
        dev_prediction_batches = prediction_func(dev_data_loader, probe)
        test_uuas = reportUUAS(
            dev_prediction_batches, dev_data_loader
        )[0]
        test_distance_dspr, test_distance_dspr_list = reportDistanceSpearmanr(
            dev_prediction_batches, dev_data_loader
        )
        if resid:
            with open(os.path.join(output_dir, model_name, layer_name, "val_metrics_distance.txt"), "w") as f:
                f.write(f"Avg UUAS: {test_uuas:.4f}\n")
                f.write(f"Avg Distance DSpr.: {test_distance_dspr:.4f}\n")
        else:
            with open(os.path.join(output_dir, model_name, {layer_name}, f"val_metrics_distance_out{'_head_' + str(attention_head) if attention_head is not None else ''}.txt"), "w") as f:
                f.write(f"Avg UUAS: {test_uuas:.4f}\n")
                f.write(f"Avg Distance DSpr.: {test_distance_dspr:.4f}\n")
                
    elif config["experiment"] == "depth":
        probe = L1DepthProbe(input_size, rep_dim, finetune_model).to(device)
        trainer = Trainer(max_epochs=num_epochs)
        trainer.fit(probe, train_data_loader, dev_data_loader)
        dev_prediction_batches = prediction_func(dev_data_loader, probe)
        test_acc = reportRootAcc(
            dev_prediction_batches, dev_data_loader
        )
        test_depth_dspr, test_depth_dspr_list = reportDepthSpearmanr(
            dev_prediction_batches, dev_data_loader
        )
        if resid:
            with open(os.path.join(output_dir, model_name, layer_name, "val_metrics_depth.txt"), "w") as f:
                f.write(f"Avg Acc: {test_acc:.4f}\n")
                f.write(f"Avg Depth DSpr.: {test_depth_dspr:.4f}\n")
        else:
            with open(os.path.join(output_dir, model_name, layer_name, "val_metrics_depth_out.txt"), "w") as f:
                f.write(f"Avg Acc: {test_acc:.4f}\n")
                f.write(f"Avg Depth DSpr.: {test_depth_dspr:.4f}\n")
        
    else:
        raise ValueError("Invalid experiment type", config["experiment"])
  
def prediction_func(data_loader, probe):
    prediction_batches = []
    probe.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="[Pred Batch]"):
            observation_batch = batch[0]
            observation_batch = observation_batch.to(probe.device)
            predictions = probe(observation_batch)
            prediction_batches.append(predictions.cpu())

    return prediction_batches

def get_nopunct_argmin(prediction, words, poses):
    """
    Gets the argmin of predictions, but filters out all punctuation-POS-tagged words
    -----
    author: @john-hewitt
    https://github.com/john-hewitt/structural-probes
    """
    puncts = ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]
    original_argmin = np.argmin(prediction)
    for i in range(len(words)):
        argmin = np.argmin(prediction)
        if poses[argmin] not in puncts:
            return argmin
        else:
            prediction[argmin] = 9000
    return original_argmin

def prims_matrix_to_edges(matrix, words, poses):
    """
    Constructs a minimum spanning tree from the pairwise weights in matrix;
    returns the edges.

    Never lets punctuation-tagged words be part of the tree.
    -----
    author: @john-hewitt
    https://github.com/john-hewitt/structural-probes
    """
    pairs_to_distances = {}
    uf = UnionFind(len(matrix))
    for i_index, line in enumerate(matrix):
        for j_index, dist in enumerate(line):
            if poses[i_index] in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]:
                continue
            if poses[j_index] in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]:
                continue
            pairs_to_distances[(i_index, j_index)] = dist
    edges = []
    for (i_index, j_index), distance in sorted(
        pairs_to_distances.items(), key=lambda x: x[1]
    ):
        if uf.find(i_index) != uf.find(j_index):
            uf.union(i_index, j_index)
            edges.append((i_index, j_index))
    return edges

def reportDepthSpearmanr(prediction_batches, dataset):
    """
    Writes the Spearman correlations between predicted and true depths.

    For each sentence, computes the spearman correlation between predicted
    and true depths.

    Computes the average such metric between all sentences of the same length.
    Then computes the average Spearman across sentence lengths 5 to 50.

    Args:
        prediction_batches: A sequence of batches of predictions for a data split
        dataset: A sequence of batches of Observations
    """
    lengths_to_spearmanrs = defaultdict(list)
    for (
        prediction_batch,
        (data_batch, label_batch, length_batch, observation_batch),
    ) in zip(prediction_batches, dataset):
        for prediction, label, length, (observation, _) in zip(
            prediction_batch, label_batch, length_batch, observation_batch
        ):

            words = observation.sentence
            length = int(length)
            prediction = prediction[:length]
            label = label[:length].cpu()
            sent_spearmanr = spearmanr(prediction, label)
            lengths_to_spearmanrs[length].append(sent_spearmanr.correlation)

    mean_spearman_for_each_length = {
        length: np.mean(lengths_to_spearmanrs[length])
        for length in lengths_to_spearmanrs
    }

    mean = np.mean(
        [
            mean_spearman_for_each_length[x]
            for x in range(5, 51)
            if x in mean_spearman_for_each_length
        ]
    )

    return mean, mean_spearman_for_each_length

def reportUUAS(prediction_batches, dataset):
    """
    Computes the UUAS score for a dataset.
    From the true and predicted distances, computes a minimum spanning tree
    of each, and computes the percentage overlap between edges in all
    predicted and gold trees.
    All tokens with punctuation part-of-speech are excluded from the minimum
    spanning trees.

    Args:
        prediction_batches: A sequence of batches of predictions for a data split
        dataset: A sequence of batches of Observations
    """

    uspan_total = 0
    uspan_correct = 0
    total_sents = 0

    gold_edge_lens = []
    pred_edge_lens = []
    gold_edge_recall = {}
    gold_edge_cnt = {}
    for (
        prediction_batch,
        (data_batch, label_batch, length_batch, observation_batch),
    ) in tqdm(zip(prediction_batches, dataset), desc="[UUAS]"):
        for prediction, label, length, (observation, _) in zip(
            prediction_batch, label_batch, length_batch, observation_batch
        ):

            words = observation.sentence
            poses = observation.xpos_sentence
            length = int(length)
            assert length == len(observation.sentence)
            prediction = prediction[:length, :length]
            label = label[:length, :length].cpu()

            gold_edges = prims_matrix_to_edges(label, words, poses)
            gold_edges = [tuple(sorted(x)) for x in gold_edges]
            pred_edges = prims_matrix_to_edges(prediction, words, poses)
            pred_edges = [tuple(sorted(x)) for x in pred_edges]

            uspan_correct += len(
                set(gold_edges).intersection(
                    set(pred_edges)
                )
            )
            uspan_total += len(gold_edges)
            total_sents += 1

            # prediction length distribution
            for e in gold_edges:
                gold_edge_lens.append(e[1] - e[0])
            for e in pred_edges:
                pred_edge_lens.append(e[1] - e[0])

            # recall per edge type 
            gold_edges_set = set(str(e[0]) + '-' + str(e[1]) for e in gold_edges)
            for e in pred_edges:
                e_str = str(e[0]) + '-' + str(e[1])
                if(e_str in gold_edges_set):
                    if(observation.head_indices[e[0]] == str(e[1] + 1)):
                        edge_type = observation.governance_relations[e[0]]
                    else: 
                        assert(observation.head_indices[e[1]] == str(e[0] + 1))
                        edge_type = observation.governance_relations[e[1]]
                    if(edge_type in gold_edge_recall): gold_edge_recall[edge_type] += 1
                    else: gold_edge_recall[edge_type] = 1
            for edge_type in observation.governance_relations:
                if(edge_type in gold_edge_cnt): gold_edge_cnt[edge_type] += 1
                else: gold_edge_cnt[edge_type] = 1
    uuas = uspan_correct / float(uspan_total)

    gold_edge_lens = Counter(gold_edge_lens)
    pred_edge_lens = Counter(pred_edge_lens)
    return uuas, gold_edge_lens, pred_edge_lens, gold_edge_recall, gold_edge_cnt

def reportDistanceSpearmanr(prediction_batches, dataset):
    """
    Writes the Spearman correlations between predicted and true distances.

    For each word in each sentence, computes the Spearman correlation between
    all true distances between that word and all other words, and all
    predicted distances between that word and all other words.

    Computes the average such metric between all sentences of the same length.
    Then computes the average Spearman across sentence lengths 5 to 50.

    Args:
        prediction_batches: A sequence of batches of predictions for a data split
        dataset: A sequence of batches of Observations
    """

    lengths_to_spearmanrs = defaultdict(list)
    for (
        prediction_batch,
        (data_batch, label_batch, length_batch, observation_batch),
    ) in tqdm(zip(prediction_batches, dataset), desc="[DSpr.]"):
        for prediction, label, length, (observation, _) in zip(
            prediction_batch, label_batch, length_batch, observation_batch
        ):

            words = observation.sentence
            length = int(length)
            prediction = prediction[:length, :length]
            label = label[:length, :length].cpu()
            spearmanrs = [
                spearmanr(pred, gold) for pred, gold in zip(prediction, label)
            ]
            lengths_to_spearmanrs[length].extend([x.correlation for x in spearmanrs])

    mean_spearman_for_each_length = {
        length: np.mean(lengths_to_spearmanrs[length])
        for length in lengths_to_spearmanrs
    }
    mean = np.mean(
        [
            mean_spearman_for_each_length[x]
            for x in range(5, 51)
            if x in mean_spearman_for_each_length
        ]
    )

    return mean, mean_spearman_for_each_length

def reportRootAcc(prediction_batches, dataset):
    """
    Computes the root prediction accuracy.

    For each sentence in the corpus, the root token in the sentence
    should be the least deep. This is a simple evaluation.

    Computes the percentage of sentences for which the root token
    is the least deep according to the predicted depths.

    Args:
        prediction_batches: A sequence of batches of predictions for a data split
        dataset: A sequence of batches of Observations
    """
    total_sents = 0
    correct_root_predictions = 0
    for (
        prediction_batch,
        (data_batch, label_batch, length_batch, observation_batch),
    ) in zip(prediction_batches, dataset):
        for prediction, label, length, (observation, _) in zip(
            prediction_batch, label_batch, length_batch, observation_batch
        ):

            length = int(length)
            label = list(label[:length].cpu())
            prediction = prediction.data[:length]
            words = observation.sentence
            poses = observation.xpos_sentence
            # print(prediction.shape, len(words), len(poses))

            correct_root_predictions += label.index(0) == get_nopunct_argmin(
                prediction, words, poses
            )
            total_sents += 1

    root_acc = correct_root_predictions / float(total_sents)
    print('Root Acc:', root_acc)
    return root_acc

class UnionFind:
    """
    Naive UnionFind implementation for (slow) Prim's MST algorithm
    Used to compute minimum spanning trees for distance matrices
    -----
    author: @john-hewitt
    https://github.com/john-hewitt/structural-probes
    """

    def __init__(self, n):
        self.parents = list(range(n))

    def union(self, i, j):
        if self.find(i) != self.find(j):
            i_parent = self.find(i)
            self.parents[i_parent] = j

    def find(self, i):
        i_parent = i
        while True:
            if i_parent != self.parents[i_parent]:
                i_parent = self.parents[i_parent]
            else:
                break
        return i_parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="path to config file")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    main(config)