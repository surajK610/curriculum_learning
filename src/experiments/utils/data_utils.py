import sys

sys.path.append("..")

import os
from collections import namedtuple, defaultdict

from typing import Optional, Dict
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import numpy as np

import h5py
from tqdm import tqdm
from itertools import chain

import glob

AttnHead = namedtuple("AttnHead", "layer head")

def dataStat(data):
    """
    Statistics of text.
    """

    len_min, len_max, len_avg = float("Inf"), 0, 0
    text_len = []

    for text in data:
        text_len.append(len(text.split(" ")))

        if len_min > len(text):
            len_min = len(text)
        if len_max < len(text):
            len_max = len(text)
        len_avg += len(text)

    len_avg /= len(data)
    text_len = np.array(text_len)

    return len_min, len_max, len_avg, text_len


def loadText(data_path):
    """
    Yields batches of lines describing a sentence in conllx.

    Args:
        data_path: Path of a conllx file.
    Yields:
        a list of lines describing a single sentence in conllx.
    """
    text, t = [], []
    total_len = []
    for line in tqdm(open(data_path), total=total_len):
        if line.startswith("#"):
            continue

        if not line.strip():
            text += [" ".join(t)]
            t = []
        else:
            t.append(line.split("\t")[1])

    return text


def loadTextOntonotes(input_file):
    sentences = []
    for cur_file in tqdm(glob.glob(input_file + '**/*.*gold_conll', recursive=True)):
        with open(cur_file, 'r') as in_f:
            sen = []
            for line in in_f:
                if line.startswith('#'):
                    continue
                if not line.strip():
                    sentences.append(" ".join(sen))
                    sen = []
                    continue
                vals = line.split()
                sen.append(vals[3])
    return sentences

def read_onto_notes_format(input_file, observation_class):
    data = []
    idx = 0
    for cur_file in tqdm(glob.glob(input_file + '**/*.*gold_conll', recursive=True)):
        with open(cur_file, 'r') as in_f:
            sen = []
            ner = []
            np_start = []
            np_end = []
            phrase_start = []
            phrase_end = []
            prev_ner = ''
            for line in in_f:
                if line.startswith('#'):
                    continue
                if line.strip() == '':
                    datum = {
                            'index': idx,
                            'sentence': sen,
                            'ner': ner,
                            'phrase_start': phrase_start,
                            'phrase_end': phrase_end,
                            'np_start': np_start,
                            'np_end': np_end,
                            'embeddings': None,
                            }
                    data.append(observation_class(**datum))
                    sen = []
                    ner = []
                    np_start = []
                    np_end = []
                    phrase_start = []
                    phrase_end = []
                    idx += 1
                    continue
                vals = line.split()
                sen.append(vals[3])

                cur_ner = vals[10]
                if cur_ner.startswith('('):
                    cur_ner = cur_ner[1:]
                    prev_ner = cur_ner
                if cur_ner.endswith(')'):
                    cur_ner = prev_ner[:-1]
                    prev_ner = ''
                if prev_ner != '':
                    cur_ner = prev_ner
                if cur_ner != '*' and cur_ner.endswith('*'):
                    cur_ner = cur_ner[:-1]
                ner.append(cur_ner)

                constituency = vals[5]

                if '(NP' in constituency:
                    np_start.append('S')
                else:
                    np_start.append('NS')

                if 'NP)' in constituency:
                    np_end.append('E')
                else:
                    np_end.append('NE')

                if constituency.startswith('('):
                    phrase_start.append('S')
                else:
                    phrase_start.append('NS')

                if constituency.endswith(')'):
                    phrase_end.append('E')
                else:
                    phrase_end.append('NE')

    return data

def words_to_input_ids_and_last_token_index(tokenizer, words):
    inputs = tokenizer(words, return_tensors='pt', is_split_into_words=True) 
    
    # Calculate word boundaries in terms of token indexes
    word_ids = inputs.word_ids()
    word_boundaries = {}
    last_word_id = None
    for i, word_id in enumerate(word_ids):
        if word_id is not None:
            if word_id != last_word_id:
                word_boundaries[word_id] = [i]
            else:
                word_boundaries[word_id].append(i)
        last_word_id = word_id
  
  # Find last token index of each word
    last_token_index_per_word = [max(indexes) for word_id, indexes in word_boundaries.items()]
  
    return inputs, last_token_index_per_word, word_boundaries

def generate_gaussian_inputs(m1, s1, m2, s2, m3, s3, m4, s4, N, N_p, N_pp):
    f1 = torch.normal(mean=m1, std=s1)
    f2 = torch.normal(mean=m2, std=s2)
    f3 = torch.normal(mean=m3, std=s3)
    f4 = torch.normal(mean=m4, std=s4)
    full = torch.concat([f1, f2, f3, f4], dim=0)
    label = (torch.mean(f3) > N_p) if (torch.mean(f1) > N) else (torch.mean(f4) > N_pp)
    alt_label = (torch.mean(f3) > N_p) if (torch.mean(f2) > N) else (torch.mean(f4) > N_pp)
    return full, label, alt_label

def generate_activations(model, tokenizer, dataset, device, split='train', task='pos', ontonotes=False):
    ## NOTE: This function is outdated in that it uses the last token of each word as the word representation rather than a mean
    task_labels = []
    model.config.output_hidden_states = True
    model.eval()
    relevant_activations = defaultdict(list)
    for example in tqdm(dataset[split], desc=f'Generating activations for {split} set', total=len(dataset[split])):
        inputs, last_token_index_per_word, _ = words_to_input_ids_and_last_token_index(tokenizer, example['tokens'])
        inputs = {k:v.to(device) for k, v in inputs.items()}
        if ontonotes:
            task_labels.append(example['tags'])
        else:
            task_labels.append(example[f'{task}_tags'])
        with torch.no_grad():
            output = model(**inputs)
            for i, val in enumerate(output.hidden_states):
                relevant_activations[i].append(val.squeeze(0)[last_token_index_per_word, :])
    task_labels = torch.tensor(list(chain(*task_labels))).to(device)
    for i in relevant_activations:
        relevant_activations[i] = torch.vstack(relevant_activations[i])
    return relevant_activations, task_labels

def savePythiaHDF5(path, text, tokenizer, model, LAYER_COUNT, FEATURE_COUNT, device):
    model.eval()
    with h5py.File(path, "w") as fout:
        for index, line in enumerate(tqdm(text, desc="[saving embeddings]")):
            line = line.strip()
            tokenized_text = tokenizer.tokenize(line) ## NOTE: No start of sentence token, maybe change
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            segment_ids = [1 for x in tokenized_text]
            if "cuda" in device.type:
              tokens_tensor = torch.tensor([indexed_tokens]).cuda()
              segments_tensors = torch.tensor([segment_ids]).cuda()
            else:
              tokens_tensor = torch.tensor([indexed_tokens])
              segments_tensors = torch.tensor([segment_ids])
              
            with torch.no_grad():
                encoded_layers = model(
                    tokens_tensor, segments_tensors, output_hidden_states=True
                )
                # embeddings + 12 layers
                encoded_layers = encoded_layers[-1]
            dset = fout.create_dataset(
                str(index), (LAYER_COUNT, len(tokenized_text), FEATURE_COUNT)
            )
            dset[:, :, :] = np.vstack([x.cpu().numpy() for x in encoded_layers])

def embedPythiaObservation(hdf5_path, observations, tokenizer, observation_class, layer_index):
    hf = h5py.File(hdf5_path, "r")
    indices = list(hf.keys())

    single_layer_features_list = []

    for index in tqdm(sorted([int(x) for x in indices]), desc="[aligning embeddings]"):
        observation = observations[index]
        feature_stack = hf[str(index)]
        single_layer_features = feature_stack[layer_index]
        tokenized_sent = tokenizer.tokenize(" ".join(observation.sentence))
        untokenized_sent = observation.sentence
        _, _, untok_tok_mapping = words_to_input_ids_and_last_token_index(tokenizer, observation.sentence)

        assert single_layer_features.shape[0] == len(tokenized_sent)
        single_layer_features = torch.tensor(
            np.array([
                np.mean(
                    single_layer_features[
                        untok_tok_mapping[i][0] : untok_tok_mapping[i][-1] + 1, :
                    ], ## mean across words
                    axis=0,
                )
                for i in range(len(untokenized_sent))
            ])
        )
        assert single_layer_features.shape[0] == len(observation.sentence)
        single_layer_features_list.append(single_layer_features)

    embeddings = single_layer_features_list
    embedded_observations = []
    for observation, embedding in zip(observations, embeddings):
        embedded_observation = observation_class(*(observation[:-1]), embedding)
        embedded_observations.append(embedded_observation)

    return embedded_observations


def makeHooks(model, cache : Optional[defaultdict], mlp=True, attn=True, embeddings=True, remove_batch_dim=False, device='cpu'):
    
    def hook_embedding(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        if remove_batch_dim:
            cache['embedding'].append(output[0].detach().to(device))
        else:
            cache['embedding'].append(output.detach().to(device))
        
    def hook_self_attention(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        if remove_batch_dim:
            cache['attention'].append(output[0].detach().to(device))
        else:
            cache['attention'].append(output.detach().to(device))
        
    def hook_mlp(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        if remove_batch_dim:
            cache['mlp'].append(output[0].detach().to(device))
        else:
            cache['mlp'].append(output.detach().to(device))
    
    if mlp:
        for layer in model.encoder.layer:
            layer.output.dense.register_forward_hook(hook_mlp)
    if attn:
        for layer in model.encoder.layer:
            layer.attention.self.register_forward_hook(hook_self_attention)
    if embeddings:
        model.embeddings.register_forward_hook(hook_embedding)
    return cache

def decomposeHeads(model, attention_vectors):
    """
    Decompose attention heads into subspaces.
    `(cache['attention'][0] @ model.encoder.layer[0].attention.output.dense.weight.data.T) + model.encoder.layer[0].attention.output.dense.bias`
    """
    attention_head_dict = {}
    assert len(attention_vectors) == 12
    for i, attn_layer in enumerate(tqdm(attention_vectors, desc='Decomposing attention heads')):
        output_matrix = model.encoder.layer[i].attention.output.dense.weight.data.T
        for j in range(model.config.num_attention_heads):
            output_slice = output_matrix[j*64:(j+1)*64, :]
            if len(attn_layer.shape) == 2:
                attn_slice = attn_layer[:, j*64:(j+1)*64]
            elif len(attn_layer.shape) == 3:
                attn_slice = attn_layer[:, :, j*64:(j+1)*64] 
                ## if batch dim intact
            else:
                raise ValueError('Attention layer has unexpected shape')
            
            attention_head_dict[AttnHead(i, j)] =  (attn_slice @ output_slice).cpu().numpy()
    return attention_head_dict

def decomposeSingleHead(model, attention_vector, layer, head):
    """
    Decompose attention heads into subspaces.
    layer is 1-indexed so use layer-1
    """
    assert layer in range(1, model.config.num_hidden_layers+1)
    output_matrix = model.encoder.layer[layer-1].attention.output.dense.weight.data.T
    output_slice = output_matrix[head*64:(head+1)*64, :]
    if len(attention_vector.shape) == 2:
        attn_slice = attention_vector[:, head*64:(head+1)*64]
    elif len(attention_vector.shape) == 3:
        attn_slice = attention_vector[:, :, head*64:(head+1)*64] 
        ## if batch dim intact
    else:
        raise ValueError('Attention layer has unexpected shape')
    return attn_slice @ output_slice.cpu().numpy()
        
    
def saveBertHDF5(path, text, tokenizer, model, LAYER_COUNT, FEATURE_COUNT, device, resid=True):
    """
    Takes raw text and saves BERT-cased features for that text to disk
    Adapted from the BERT readme (and using the corresponding package) at
    https://github.com/huggingface/pytorch-pretrained-BERT
    """

    model.eval()
    if not resid:
        cache = defaultdict(list)
        cache = makeHooks(model, cache, mlp=True, attn=True, embeddings=True, remove_batch_dim=False, device=device)
        
    with h5py.File(path, "w") as fout:
        for index, line in enumerate(tqdm(text, desc="[saving embeddings]")):
            line = line.strip()  # Remove trailing characters
            line = "[CLS] " + line + " [SEP]"
            tokenized_text = tokenizer.wordpiece_tokenizer.tokenize(line)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            segment_ids = [1 for x in tokenized_text]

            # Convert inputs to PyTorch tensors
            if "cuda" in device.type:
              tokens_tensor = torch.tensor([indexed_tokens]).cuda()
              segments_tensors = torch.tensor([segment_ids]).cuda()
            else:
              tokens_tensor = torch.tensor([indexed_tokens])
              segments_tensors = torch.tensor([segment_ids])
            
            with torch.no_grad():
                encoded_layers = model(
                    tokens_tensor, segments_tensors, output_hidden_states=True
                )
                # embeddings + 12 layers
                encoded_layers = encoded_layers[-1]
                if not resid:
                    attn_outputs = cache['attention'].copy()
                    mlp_outputs = cache['mlp'].copy()
                    embedding_outputs = cache['embedding'].copy()
                    
                    cache['attention'].clear()
                    cache['mlp'].clear()
                    cache['embedding'].clear()
                    # only use output of the layer (i.e. attention + mlp)
            if not resid:
                dset = fout.create_dataset(
                    str(index), (1 + 2 * model.config.n_layers, len(tokenized_text), FEATURE_COUNT)
                ) ## 1 for embedding, 12 for attention, 12 for mlp
                dset[:, :, :] = np.vstack([embedding_outputs[0].cpu().numpy()] + [mlp_outputs[i].cpu().numpy() for i in range(model.config.n_layers)] + [attn_outputs[i].cpu().numpy() for i in range(model.config.n_layers)])
            else:
                dset = fout.create_dataset(
                    str(index), (LAYER_COUNT, len(tokenized_text), FEATURE_COUNT)
                )
                dset[:, :, :] = np.vstack([x.cpu().numpy() for x in encoded_layers])

def embedBertObservation(
    model, hdf5_path, observations, tokenizer, observation_class, layer_index, attention_head=None
):
    """
    Adds pre-computed BERT embeddings from disk to Observations.
    
    Reads pre-computed subword embeddings from hdf5-formatted file.
    Sentences should be given integer keys corresponding to their order
    in the original file.
    Embeddings should be of the form (layer_count, subword_sent_length, feature_count)
    subword_sent_length is the length of the sequence of subword tokens
    when the subword tokenizer was given each canonical token (as given
    by the conllx file) independently and tokenized each. Thus, there
    is a single alignment between the subword-tokenized sentence
    and the conllx tokens.

    Args:
        hdf5_path: The filepath of a hdf5 file containing embeddings.
        observations: A list of Observation objects composing a dataset.
        tokenizer: (optional) a tokenizer used to map from
            conllx tokens to subword tokens.
        layer_index: The index corresponding to the layer of representation
            to be used. (e.g., 0 for BERT embeddings, 1, ..., 12 for BERT 
            layer 1, ..., 12)
        attention_head: (optional) The index corresponding to the attention head 1, ..., 12
    Returns:
        A list of Observations with pre-computed embedding fields.
        
    Raises:
        AssertionError: sent_length of embedding was not the length of the
        corresponding sentence in the dataset.
    """

    hf = h5py.File(hdf5_path, "r")
    indices = list(hf.keys())

    single_layer_features_list = []

    for index in tqdm(sorted([int(x) for x in indices]), desc="[aligning embeddings]"):
        observation = observations[index]
        feature_stack = hf[str(index)]
        if attention_head is None:
            single_layer_features = feature_stack[layer_index]
        else:
            single_layer_features = feature_stack[12 + layer_index] ## layer is 1-indexed
            single_layer_features = decomposeSingleHead(model, single_layer_features, layer_index, attention_head)
            
        single_layer_features = feature_stack[layer_index]
        tokenized_sent = tokenizer.wordpiece_tokenizer.tokenize(
            "[CLS] " + " ".join(observation.sentence) + " [SEP]"
        )
        untokenized_sent = observation.sentence
        untok_tok_mapping = match_tokenized_to_untokenized(
            tokenized_sent, untokenized_sent
        )
        assert single_layer_features.shape[0] == len(tokenized_sent)
        single_layer_features = torch.tensor(
            np.array([
                np.mean(
                    single_layer_features[
                        untok_tok_mapping[i][0] : untok_tok_mapping[i][-1] + 1, :
                    ],
                    axis=0,
                )
                for i in range(len(untokenized_sent))
            ])
        )
        assert single_layer_features.shape[0] == len(observation.sentence)
        single_layer_features_list.append(single_layer_features)

    embeddings = single_layer_features_list
    embedded_observations = []
    for observation, embedding in zip(observations, embeddings):
        embedded_observation = observation_class(*(observation[:-1]), embedding)
        embedded_observations.append(embedded_observation)

    return embedded_observations

def get_observation_class(fieldnames):
    """
    Returns a namedtuple class for a single observation.

    The namedtuple class is constructed to hold all language and annotation
    information for a single sentence or document.

    Args:
        fieldnames: a list of strings corresponding to the information in each
            row of the conllx file being read in. (The file should not have
            explicit column headers though.)
    Returns:
        A namedtuple class; each observation in the dataset will be an instance
        of this class.
    """
    return namedtuple("Observation", fieldnames)


def generate_lines_for_sent(lines):
    """
    Yields batches of lines describing a sentence in conllx.

    Args:
        lines: Each line of a conllx file.
    Yields:
        a list of lines describing a single sentence in conllx.
    """

    buf = []
    for line in lines:
        if line.startswith("#"):
            continue
        if not line.strip():
            if buf:
                yield buf
                buf = []
            else:
                continue
        else:
            buf.append(line.strip())
    if buf:
        yield buf


def load_conll_dataset(filepath, observation_class):
    """
    Reads in a conllx file; generates Observation objects
    
    For each sentence in a conllx file, generates a single Observation
    object.

    Args:
        filepath: the filesystem path to the conll dataset

    Returns:
        A list of Observations 
    """
    observations = []

    lines = (x for x in open(filepath))
    for buf in generate_lines_for_sent(lines):
        conllx_lines = []
        for line in buf:
            conllx_lines.append(line.strip().split("\t"))
        embeddings = [None for x in range(len(conllx_lines))]
        observation = observation_class(*zip(*conllx_lines), embeddings)
        observations.append(observation)

    return observations

def match_tokenized_to_untokenized(tokenized_sent, untokenized_sent):
    """
    Aligns tokenized and untokenized sentence given subwords "##" prefixed

    Assuming that each subword token that does not start a new word is prefixed
    by two hashes, "##", computes an alignment between the un-subword-tokenized
    and subword-tokenized sentences.

    Args:
        tokenized_sent: a list of strings describing a subword-tokenized sentence
        untokenized_sent: a list of strings describing a sentence, no subword tok.
    Returns:
        A dictionary of type {int: list(int)} mapping each untokenized sentence
        index to a list of subword-tokenized sentence indices
    """
    mapping = defaultdict(list)
    untokenized_sent_index = 0
    tokenized_sent_index = 1
    while untokenized_sent_index < len(untokenized_sent) and tokenized_sent_index < len(
        tokenized_sent
    ):

        while tokenized_sent_index + 1 < len(tokenized_sent) and tokenized_sent[
            tokenized_sent_index + 1
        ].startswith("##"):

            mapping[untokenized_sent_index].append(tokenized_sent_index)
            tokenized_sent_index += 1

        mapping[untokenized_sent_index].append(tokenized_sent_index)
        untokenized_sent_index += 1
        tokenized_sent_index += 1

    return mapping

def custom_pad_pos(batch_observations, use_disk_embeddings=True, return_all=False):
    """
    Collates pos embeddings + labels; used as collate_fn of DataLoader.
    Does not pad, but instead stacks
    Args:
        batch_observations: A list of observations composing a batch
    
    Return:
        A tuple of:
            input batch
            label batch
            lengths-of-inputs batch
            Observation batch
    """
    if use_disk_embeddings:
        if hasattr(batch_observations[0][0].embeddings, "device"):
            seqs = [x[0].embeddings.clone().detach() for x in batch_observations]
        else:
            seqs = [torch.Tensor(x[0].embeddings) for x in batch_observations]
    else:
        seqs = [x[0].sentence for x in batch_observations]
    lengths = torch.tensor([len(x) for x in seqs])
    labels = [x[1] for x in batch_observations]
    
    seqs = torch.vstack(seqs)
    labels = torch.hstack(labels)
    seqs = seqs.view(-1, seqs.shape[-1])
    labels = labels.view(-1).long()
    if return_all:
        return seqs, labels, lengths, batch_observations
    return seqs, labels

def custom_pad(batch_observations, use_disk_embeddings=True, return_all=False):
    """
    Pads sequences with 0 and labels with -1; used as collate_fn of DataLoader.
    
    Loss functions will ignore -1 labels.
    If labels are 1D, pads to the maximum sequence lengtorch.
    If labels are 2D, pads all to (maxlen,maxlen).
    Args:
        batch_observations: A list of observations composing a batch
    
    Return:
        A tuple of:
            input batch, padded
            label batch, padded
            lengths-of-inputs batch, padded
            Observation batch (not padded)
    """
    if use_disk_embeddings:
        if hasattr(batch_observations[0][0].embeddings, "device"):
            seqs = [x[0].embeddings.clone().detach() for x in batch_observations]
        else:
            seqs = [torch.Tensor(x[0].embeddings) for x in batch_observations]
    else:
        seqs = [x[0].sentence for x in batch_observations]
    lengths = torch.tensor([len(x) for x in seqs])
    seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    label_shape = batch_observations[0][1].shape
    maxlen = int(max(lengths))
    label_maxshape = [maxlen for x in label_shape]
    labels = [-torch.ones(*label_maxshape) for x in seqs]
    for index, x in enumerate(batch_observations):
        length = x[1].shape[0]
        if len(label_shape) == 1:
            labels[index][:length] = x[1]
        elif len(label_shape) == 2:
            labels[index][:length, :length] = x[1]
        else:
            raise ValueError(
                "Labels must be either 1D or 2D right now; got either 0D or >3D"
            )
    labels = torch.stack(labels)
    if return_all:
        return seqs, labels, lengths, batch_observations
    return seqs, labels


class ObservationIterator(Dataset):
    """
    List Container for lists of Observations and labels for them.
    Used as the iterator for a PyTorch dataloader.
    -----
    author: @john-hewitt
    https://github.com/john-hewitt/structural-probes
    """

    def __init__(self, observations, task):
        self.observations = observations
        self.set_labels(observations, task)

    def set_labels(self, observations, task):
        """
        Constructs aand stores label for each observation.

        Args:
            observations: A list of observations describing a dataset
            task: a Task object which takes Observations and constructs labels.
        """
        self.labels = []
        for observation in tqdm(observations, desc="[computing labels]"):
            self.labels.append(task.labels(observation))

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.labels[idx]
