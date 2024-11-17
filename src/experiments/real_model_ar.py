"""
This file reproduces some of the behavioral results in 
"A Mechanistic Interpretation of Syllogistic Reasoning in
Auto-Regressive Language Models" (i.e. Figure 3).
"""

# %%
import argparse
import sys

import numpy as np
import torch
from transformer_lens import HookedTransformer

import random
import os
import json

PYTHIA_VOCAB_SIZE = 50277 #50304
N_LAYERS=12
PYTHIA_CHECKPOINTS_OLD = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(range(1000, 143000 + 1, 10000)) + [143000]
PYTHIA_CHECKPOINTS = [512] + list(range(1000, 10000 + 1, 1000))

vocabulary = [
        " A",
        " B",
        " C",
        " D",
        " E",
        " F",
        " G",
        " H",
        " I",
        " J",
        " K",
        " L",
        " M",
        " N",
        " O",
        " P",
        " Q",
        " R",
        " S",
        " T",
        " U",
        " V",
        " W",
        " X",
        " Y",
        " Z",
    ]

def logits_to_logit_diff(model, logits, correct_answer, incorrect_answer):
    """Compute logit difference"""
    correct_idx = model.to_single_token(correct_answer)
    incorrect_idx = model.to_single_token(incorrect_answer)
    return logits[0, -1, correct_idx] - logits[0, -1, incorrect_idx]

def generate_abstract_example(vocab_set, idx_1, idx_2, idx_3, nonce=False):
    """Helper function to generate abstract example"""
    item_1 = random.choice([f' [NEW_TOKEN{i}]' for i in range(20)]) if nonce else vocab_set[idx_1]
    item_2 = vocab_set[idx_2]
    item_3 = vocab_set[idx_3]
    prompt = (
        f"All{item_1} are{item_2}. "
        + f"All{item_2} are{item_3}. "
        + f"Therefore, all{item_1} are"
    )
    label = item_3
    wrong_label = item_2
    return prompt, label, wrong_label

def behavioral_analysis(
    data_size, model, vocabulary, q_type, correct_label, incorrect_label, device, nonce=False
):
    """Main function"""
    torch.set_grad_enabled(False)

    data = {
        "clean_x": [],
        "clean_y": [],
        "wrong_y": [],
    }

    for _ in range(data_size):
        vocab_set = np.random.choice(vocabulary, 8, replace=False)
        vocab_set_1 = vocab_set[:4]
        clean_x, clean_y, wrong_y = generate_abstract_example(vocab_set_1, 0, 1, 2, nonce=nonce)
        data["clean_x"].append(clean_x)
        data["clean_y"].append(clean_y)
        data["wrong_y"].append(wrong_y)
    accuracy = []
    accuracy_w = []
    
    total_samples = len(data["clean_x"])
    for i in range(total_samples):

        # if i % 25 == 0:
            # print(f"Progress: {round(i/total_samples, 3)}")
            
        clean_prompt = data[q_type][i]
        clean_label = data[correct_label][i]
        cf_label = data[incorrect_label][i]

        clean_tokens = model.to_tokens(clean_prompt)
        logits = model(clean_tokens)

        # accuracy.append(logits_to_logit_diff(model, logits, clean_label, cf_label) > 0)
        accuracy.append(logits_to_probability(model, logits, clean_label))
        accuracy_w.append(logits_to_probability(model, logits, cf_label))

    # print(f"Model Accuracy: {torch.mean(torch.Tensor(accuracy))}")
    return torch.mean(torch.Tensor(accuracy)).item(), torch.mean(torch.Tensor(accuracy_w)).item()

def logits_to_logit_diff(model, logits, correct_answer, incorrect_answer):
    """Compute logit difference"""
    correct_idx = model.to_single_token(correct_answer)
    incorrect_idx = model.to_single_token(incorrect_answer)
    return logits[0, -1, correct_idx] - logits[0, -1, incorrect_idx]

def logits_to_probability(model, logits, correct_answer):
    """Compute the probability of the correct answer"""
    correct_idx = model.to_single_token(correct_answer)
    logits_last = logits[0, -1, :]  # Shape: [vocab_size]
    probs = torch.softmax(logits_last, dim=-1)  # Shape: [vocab_size]
    prob_correct = probs[correct_idx]
    return prob_correct.item()

def expand_tokenizer(model, nonce_tokens):
    model.tokenizer.add_tokens(nonce_tokens)
    original_num_embeddings, embedding_dim = model.embed.W_E.shape
    new_num_embeddings = original_num_embeddings + len(nonce_tokens)
    
    new_embeddings = torch.nn.Parameter(torch.empty(new_num_embeddings, embedding_dim).to(model.embed.W_E.device))
    with torch.no_grad():
        new_embeddings[:original_num_embeddings] = model.embed.W_E
    
        initializer_range = model.cfg.initializer_range
    
        torch.nn.init.normal_(
            new_embeddings[original_num_embeddings:],  # Only initialize new embeddings
            mean=0.0,
            std=initializer_range
        )
    
    model.embed.W_E = new_embeddings
    
def main(args):
    home = os.environ["LEARNING_DYNAMICS_HOME"]
    output_dir = os.path.join(home, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    nonce_tokens = [f' [NEW_TOKEN{i}]' for i in range(20)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # steps = range(1000, 143000 + 1000, 5000)
    # clean_acc_dict, nonce_acc_dict = {}, {}
    # for step in steps:
    
    model_step = HookedTransformer.from_pretrained(args.model, checkpoint_value=args.steps, device=device)
    expand_tokenizer(model_step, nonce_tokens)
    assert len(model_step.tokenizer.encode(random.choice(nonce_tokens))) == 1, "Failed to add nonce tokens"
    
    clean_acc, wrong_acc = behavioral_analysis(
        args.data_size, model_step, vocabulary, "clean_x", "clean_y", "wrong_y", device
    )
    
    nonce_acc, nonce_wrong_acc = behavioral_analysis(
        args.data_size, model_step, vocabulary, "clean_x", "clean_y", "wrong_y", device, nonce=True
    )
    print("Step", args.steps, "Clean", clean_acc, "Nonce", nonce_acc)
    
    log_scores = {"step": args.steps,
                  "clean": clean_acc, 
                  "nonce": nonce_acc, 
                  "clean_wrong": wrong_acc, 
                  "nonce_wrong": nonce_wrong_acc}
    json.dump(log_scores, open(os.path.join(args.output_dir, "model_step_acc.json"), "w"))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--model",
            type=str,
            default="EleutherAI/pythia-1.4B",
            help="pythia step to use",
        )
    parser.add_argument(
            "--steps",
            type=int,
            default=143000,
            help="pythia step to use",
        )
    parser.add_argument(
            "--data_size",
            type=int,
            default=50,
            help="dataset size",
        )
    parser.add_argument(
            "--output_dir",
            type=str,
            default="outputs/toy_model_ar/pythia-1.4B",
            help="output dir",
        )
    args = parser.parse_args()

    main(args)