from audioop import bias
import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

# ------------------------------- PROBES ------------------------------- #
class Probe(pl.LightningModule):
  def __init__(self, num_features: int, num_output: int, finetune_model: str):
    super().__init__()
    self.save_hyperparameters()
    self.flatten = Flatten()

    if finetune_model == "linear":
      self.body = nn.Linear(num_features, num_output, bias=False)
    elif finetune_model == "1-layer":
      self.body = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Linear(512, num_output, bias=False),
      )
    elif finetune_model == "2-layer":
      self.body = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, num_output, bias=False),
      )
    elif finetune_model == "3-layer":
      self.body = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, num_output, bias=False),
      )
    else:
      assert False, "Not a valid finetune model"
    self.finetune_model = finetune_model

  def configure_optimizers(self):
    if self.finetune_model == "linear":
      optimizer = torch.optim.AdamW(self.parameters(), lr=1e-2)
    else:
      optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
    return optimizer

  def training_step(self, batch, batch_idx):
    loss, logs = self.step(batch)
    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return loss

  def validation_step(self, batch, batch_idx):
    loss, logs = self.step(batch)
    self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return loss

  def test_step(self, batch, batch_idx):
    loss, logs = self.step(batch)
    self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return loss
  
  def step(self, batch):
    x, y = batch
    y = y.squeeze()
    logits = self.forward(x)
    loss = nn.functional.cross_entropy(logits, y, reduction="sum")
    acc = (logits.squeeze().argmax(-1) == y).float().mean()
    return loss, {"loss": loss.item(), "acc": acc.item()}

  def forward(self, x):
    if isinstance(x, list):
        x, _ = x
    flat_x = self.flatten(x.float())
    return self.body(flat_x)
    
class AccuracyProbe(Probe):
  def __init__(self, num_features: int, num_output: int, finetune_model: str):
    super().__init__(num_features, num_output, finetune_model)

  def validation_step(self, batch, batch_idx):
    loss, logs = self.step(batch)
    self.log("val_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
    self.log("val_acc", logs["acc"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return {"loss": loss, "val_acc": logs["acc"]}
  
class L1DistanceProbe(Probe):
  def __init__(self, num_features: int, num_output: int, finetune_model: str):
    super().__init__(num_features, num_output, finetune_model)
    self.loss = L1DistanceLoss()
    
  def step(self, batch):
    x, y, lengths, _  = batch
    norms = self.forward(x)
    batch_loss, total_sents = self.loss(norms, y, lengths)
    return batch_loss, {"loss": batch_loss.item(), "total_sents": total_sents}
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, betas=[0.9, 0.999], eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,patience=0)
    return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'val_loss'
           }
    
  def forward(self, x):
    if isinstance(x, list):
        x, _ = x
    transformed = self.body(x)
    seqlen = transformed.shape[1]
    transformed = transformed.unsqueeze(2)
    transformed = transformed.expand(-1, -1, seqlen, -1)
    transposed = transformed.transpose(1, 2)
    diffs = transformed - transposed
    squared_diffs = diffs.pow(2)
    squared_distances = torch.sum(squared_diffs, -1)
    return squared_distances
    
    
class L1DepthProbe(Probe):
  def __init__(self, num_features: int, num_output: int, finetune_model: str):
    super().__init__(num_features, num_output, finetune_model)
    self.loss = L1DepthLoss()
    
  def step(self, batch):
    x, y, lengths, _  = batch
    norms = self.forward(x)
    batch_loss, total_sents = self.loss(norms, y, lengths)
    return batch_loss, {"loss": batch_loss.item(), "total_sents": total_sents}
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, betas=[0.9, 0.999], eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,patience=0)
    return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'val_loss'
           }
    
  def forward(self, x):
      if isinstance(x, list):
          x, _ = x
      transformed = self.body(x)
      batch_size, seq_len, rank = transformed.shape
      norms = torch.bmm(
          transformed.view(batch_size * seq_len, 1, rank),
          transformed.view(batch_size * seq_len, rank, 1),
      )
      norms = norms.view(batch_size, seq_len)
      return norms

# --------------------------- MODULES --------------------------- #

class Flatten(nn.Module):
  def forward(self, input):
    return input.view(input.size(0), -1)

class L1DistanceLoss(nn.Module):
  """Custom L1 loss for distance matrices."""

  def __init__(self):
    super().__init__()
    self.word_pair_dims = (1, 2)

  def forward(self, predictions, label_batch, length_batch):
    """ 
    Computes L1 loss on distance matrices.

    Ignores all entries where label_batch=-1
    Normalizes first within sentences (by dividing by the square of 
    the sentence length) and then across the batch.

    Args:
        predictions: A pytorch batch of predicted distances
        label_batch: A pytorch batch of true distances
        length_batch: A pytorch batch of sentence lengths

    Returns:
        A tuple of:
            batch_loss: average loss in the batch
            total_sents: number of sentences in the batch
    """
    labels_1s = (label_batch != -1).float()
    predictions_masked = predictions * labels_1s
    labels_masked = label_batch * labels_1s
    total_sents = torch.sum((length_batch != 0)).float()
    squared_lengths = length_batch.pow(2).float()
    if total_sents > 0:
        loss_per_sent = torch.sum(
            torch.abs(predictions_masked - labels_masked), dim=self.word_pair_dims
        )
        normalized_loss_per_sent = loss_per_sent / squared_lengths
        batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
    else:
        batch_loss = torch.tensor(0.0, device=predictions.device)
    return batch_loss, total_sents


class L1DepthLoss(nn.Module):
  """Custom L1 loss for depth sequences."""
  def __init__(self):
    super().__init__()
    self.word_dim = 1

  def forward(self, predictions, label_batch, length_batch):
    """ 
    Computes L1 loss on depth sequences.

    Ignores all entries where label_batch=-1
    Normalizes first within sentences (by dividing by the sentence length)
    and then across the batch.

    Args:
        predictions: A pytorch batch of predicted depths
        label_batch: A pytorch batch of true depths
        length_batch: A pytorch batch of sentence lengths

    Returns:
        A tuple of:
            batch_loss: average loss in the batch
            total_sents: number of sentences in the batch
    """
    total_sents = torch.sum(length_batch != 0).float()
    labels_1s = (label_batch != -1).float()
    predictions_masked = predictions * labels_1s
    labels_masked = label_batch * labels_1s
    if total_sents > 0:
        loss_per_sent = torch.sum(
            torch.abs(predictions_masked - labels_masked), dim=self.word_dim
        )
        normalized_loss_per_sent = loss_per_sent / length_batch.float()
        batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
    else:
        batch_loss = torch.tensor(0.0, device=predictions.device)
    return batch_loss, total_sents