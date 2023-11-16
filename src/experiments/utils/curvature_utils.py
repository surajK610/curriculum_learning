import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import torch
from fancy_einsum import einsum
import plotly.express as px

def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
  '''Plots a tensor as an image.'''
  px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
  '''Plots a tensor as a line plot.'''
  px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
  '''Plots a tensor as a scatter plot.'''
  x = utils.to_numpy(x)
  y = utils.to_numpy(y)
  px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)


def compute_global_curvature(cache, total_layers, stream_idx, sequence_length, include_mlps=True, include_attn=True):
  '''Computes the global curvature of the residual stream of a model with repeating sequences.'''
  curvatures = []
  for str_idx in range(stream_idx, len(cache["resid_pre", 0]), sequence_length):
    if include_mlps and include_attn:
      displacement = torch.sqrt(torch.sum(torch.square(cache["resid_post", total_layers - 1][str_idx] - cache["resid_pre", 0][str_idx])))
    elif include_mlps:
      displacement = torch.sqrt(torch.sum(torch.square(cache["resid_post", total_layers - 1][str_idx] - cache["resid_mid", 0][str_idx])))
    elif include_attn:
      displacement = torch.sqrt(torch.sum(torch.square(cache["resid_mid", total_layers - 1][str_idx] - cache["resid_pre", 0][str_idx])))
    else:
      raise ValueError("Must include either MLPs or attention in the curvature calculation.")
    
    distance = 0.0
    for layer in range(total_layers):
      if include_mlps:
        distance+=torch.linalg.norm(cache['mlp_out', layer][str_idx])
      if include_attn:
        distance+=torch.linalg.norm(cache['attn_out', layer][str_idx])

    curvatures.append((distance/displacement).cpu().numpy())
  return curvatures

def compute_global_curvature_nonrepeating(cache, total_layers, stream_idx, include_mlps=True, include_attn=True):
  '''Computes the global curvature of the residual stream of a model.'''
  displacement = torch.sqrt(torch.sum(torch.square(cache["resid_post", total_layers - 1][stream_idx] - cache["resid_pre", 0][stream_idx])))
  distance = 0.0
  for layer in range(total_layers):
    if include_mlps:
      distance+=torch.linalg.norm(cache['mlp_out', layer][stream_idx])
    if include_attn:
      distance+=torch.linalg.norm(cache['attn_out', layer][stream_idx])

  return (distance/displacement).cpu().numpy()
    
def early_decode(model, cache, layer, mid=False):
  '''Decodes the residual stream of a model at a given layer, returning the decoded tokens.'''
  return model.unembed(cache[f'blocks.{layer}.hook_resid_{"mid" if mid else "post"}'].unsqueeze(0))

def compute_prob_kls(prob_dists):
  '''Computes the kl distribution between the probability distributions of the ith and i+1th token predictions'''
  kls = []
  for i in range(len(prob_dists) - 1):
    kls.append(torch.nn.functional.kl_div(prob_dists[i+1], prob_dists[i], log_target=True).cpu().numpy())
  return kls

def compute_long_range_kls(prob_dists, gap=40):
  '''Computes the kl distribution between the probability distributions of the ith and i+gapth token predictions'''
  kls = []
  for i in range(len(prob_dists) - gap):
    kls.append(torch.nn.functional.kl_div(prob_dists[i+gap], prob_dists[i], log_target=True).cpu().numpy())
  return kls


def layer_wise_curvature(cache, total_layers, stream_idx, sequence_length, include_mlps=True, include_attn=True, cumulative=False):
  curvatures = []
  for str_idx in range(stream_idx, len(cache["resid_pre", 0]), sequence_length):
    if include_mlps and include_attn:
      displacement = torch.sqrt(torch.sum(torch.square(cache["resid_post", total_layers - 1][str_idx] - cache["resid_pre", 0][str_idx])))
    elif include_mlps:
      displacement = torch.sqrt(torch.sum(torch.square(cache["resid_post", total_layers - 1][str_idx] - cache["resid_mid", 0][str_idx])))
    elif include_attn:
      displacement = torch.sqrt(torch.sum(torch.square(cache["resid_mid", total_layers - 1][str_idx] - cache["resid_pre", 0][str_idx])))
    else:
      raise ValueError("Must include either MLPs or attention in the curvature calculation.")
    
    dict_distance = {}
    if cumulative:
        distance = 0.0
    for layer in range(total_layers):
      if not cumulative:
          distance = 0.0
    
      if include_mlps:
        distance+=torch.linalg.norm(cache['mlp_out', layer][str_idx])
      if include_attn:
        distance+=torch.linalg.norm(cache['attn_out', layer][str_idx])
      dict_distance[layer] = (distance/displacement).clone().detach().item()
    dict_distance['total'] = sum(dict_distance.values())
    curvatures.append(dict_distance)
  return curvatures

def layer_wise_norm_running_average(cache, total_layers, stream_idx, sequence_length, include_mlps=True, include_attn=True):
  norm_running_avgs = []
  rol_avg_dict_norm = {layer: 0 for layer in range(total_layers)}
  count = 0
  for str_idx in range(stream_idx, len(cache["resid_pre", 0]), sequence_length):
    count += 1
    for layer in range(total_layers):
      distance = 0.0
      if include_mlps:
        distance+=torch.linalg.norm(cache['mlp_out', layer][str_idx])
      if include_attn:
        distance+=torch.linalg.norm(cache['attn_out', layer][str_idx])
      rol_avg_dict_norm[layer] = (rol_avg_dict_norm[layer]*(count - 1) +(distance).clone().detach().item())/count 
    norm_running_avgs.append(rol_avg_dict_norm.copy())
    
  return norm_running_avgs


def layer_wise_consec_kl_divs(model, cache, total_layers, stream_idx, sequence_length, resid_post=False, include_mlps=True, include_attn=True):
  consec_kl_divs = []
  
  for str_idx in range(stream_idx, len(cache["resid_pre", 0]) - sequence_length, sequence_length):
    ## for loop goes all the way to one from the end because we are comparing the ith and i+1th token predictions
    kl_div_curr = {}
    for layer in range(total_layers):
      layer_distr, layer_distr_n = 0.0, 0.0
      if not resid_post:
        if include_mlps:
          layer_distr += cache['mlp_out', layer][str_idx]
          layer_distr_n += cache['mlp_out', layer][str_idx + sequence_length]
        if include_attn:
          layer_distr += cache['attn_out', layer][str_idx]
          layer_distr_n += cache['mlp_out', layer][str_idx + sequence_length]
      
      if resid_post:
        layer_distr += cache['resid_post', layer][str_idx]
        layer_distr_n += cache['resid_post', layer][str_idx + sequence_length]
      # print(layer_distr.unsqueeze([0, 1]).shape)
      scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer = -1, pos_slice=-1)
      logits_distr = model.unembed(layer_distr.view(1, 1, -1))
      logits_distr_n = model.unembed(layer_distr_n.view(1, 1, -1))
      # umembed expects batch_size x sequence_length x embedding_dim
      
      log_probs = torch.nn.functional.log_softmax(logits_distr, dim=-1).view(-1)
      log_probs_n = torch.nn.functional.log_softmax(logits_distr_n, dim=-1).view(-1)
      # convert back to 1 dimensional vector
      
      kl_div_curr[layer] = torch.nn.functional.kl_div(log_probs, log_probs_n, log_target=True).detach().item()
          
    consec_kl_divs.append(kl_div_curr.copy())
  
  ## one extra just so it is the same length as the other lists
  consec_kl_divs.append(kl_div_curr.copy())
  return consec_kl_divs

def compute_prob_kls(prob_dists):
  '''Computes the kl distribution between the probability distributions of the ith and i+1th token predictions'''
  kls = []
  for i in range(len(prob_dists) - 1):
    kls.append(torch.nn.functional.kl_div(prob_dists[i+1], prob_dists[i], log_target=True).cpu().numpy())
  return kls

def logits_to_ave_logit_diff(logits, answer_tokens, per_prompt=False, cumulative=False):
  # Only the final logits are relevant for the answer
  final_logits = logits[:, -1, :]
  answer_logits = final_logits.gather(dim=-1, index=answer_tokens)
  answer_logit_diff = answer_logits[:, 0] - answer_logits[:, 1]
  if per_prompt:
      return answer_logit_diff
  else:
      return answer_logit_diff.mean()

def residual_stack_to_logit_diff(residual_stack, cache, logit_diff_directions, prompts) -> float:
  scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer = -1, pos_slice=-1)
  return einsum("... batch d_model, batch d_model -> ...", scaled_residual_stack, logit_diff_directions)/len(prompts)

def layer_wise_losses(model, cache, total_layers, example, stream_idx, sequence_length, resid_post=False):
  losses_lw = []
  layer_loss = {}
  layer_val = {}
  if not resid_post:
    per_layer_residual, labels = cache.decompose_resid(layer=-1, return_labels=True)
    dict_lrs = {label: layer_residual for label, layer_residual in zip(labels, per_layer_residual)}
    for i in range(total_layers):
      layer_val[i] = dict_lrs[f'{i}_attn_out'] + dict_lrs[f'{i}_mlp_out']
  else:
    per_layer_residual, labels = cache.accumulated_resid(layer=-1, incl_mid=True, return_labels=True)
    dict_lrs = {label: layer_residual for label, layer_residual in zip(labels, per_layer_residual)}
    for i in range(total_layers - 1):
      layer_val[i] =  dict_lrs[f'{i+1}_pre'] ## after pre is before post
    layer_val[total_layers - 1] = dict_lrs[f'final_post']
                                           
  layer_val_stack = torch.stack([layer_val[i] for i in range(total_layers)], dim=0)
  layer_val_stack = layer_val_stack.squeeze(1)
  scaled_residual_stack = cache.apply_ln_to_stack(layer_val_stack, layer = -1) ## apply last ln layer
  
  for layer in range(total_layers):
    layer_distr = scaled_residual_stack[layer]
    layer_logits = model.unembed(layer_distr.unsqueeze(0))
    layer_loss[layer] = model.loss_fn(layer_logits, example.unsqueeze(0), per_token=True)[0].cpu()[stream_idx::sequence_length]
  
  for idx in range(len(layer_loss[0])):
    ## put into form cuz other stuff written loke that
    loss_curr = {}
    for layer in range(total_layers):
      loss_curr[layer] = layer_loss[layer][idx].item()
          
    losses_lw.append(loss_curr.copy())
  
  return losses_lw

def logit_attribution():
  pass