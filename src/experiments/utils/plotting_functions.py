import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.io as pio
import pandas as pd
import numpy as np
from .curvature_utils import residual_stack_to_logit_diff
import transformer_lens.utils as tutils


def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(tutils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(tutils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = tutils.to_numpy(x)
    y = tutils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)
    
def plot_curvature_vs_repetitions(results, sequence_position, offset=0, 
                                  show=True, save_fig=False, save_path=None):
  x = list(range(len(results[sequence_position]["curvatures"])))
  y = results[sequence_position]["curvatures"]
  plt.scatter(x[offset:], y[offset:], c="green", marker="^")
  plt.legend()
  plt.title("Global Curvature of the Residual Stream")
  plt.ylabel("Global Curvature")
  plt.xlabel("Repetitions")
  if save_fig and save_path:
    plt.savefig(save_path)
  if show:
    plt.show()
  plt.close()

def plot_earliest_1_rr_vs_repetitions(results, sequence_position, offset=0,
                                      show=True, save_fig=False, save_path=None):
  rr_dict = results[sequence_position]["ed_reciprocal_ranks"]
  x = list(range(len(rr_dict[0])))
  earliest_rr_of_1 = []
  for i in range(len(x)):
    early_rr = 12 # default value if no rr of 1
    for layer in range(len(rr_dict)):
      if rr_dict[layer][i] == 1:
        early_rr = layer
        break
    earliest_rr_of_1.append(early_rr)
  y = earliest_rr_of_1
  plt.scatter(x[offset:], y[offset:], c="green", marker="^")
  plt.legend()
  plt.title("Earliest RR of 1 Layer vs Repetitions")
  plt.ylabel("Layer")
  plt.xlabel("Repetitions")
  if save_fig and save_path:
    plt.savefig(save_path)
  if show:
    plt.show()
  plt.close()

def plot_rr_vs_repetitions(results, sequence_position, offset=0, layer=11, 
                           show=True, save_fig=False, save_path=None):
  rr_dict = results[sequence_position]["ed_reciprocal_ranks"]
  x = list(range(len(rr_dict[layer])))
  y = rr_dict[layer]
  plt.scatter(x[offset:], y[offset:], c="green", marker="^")
  plt.legend()
  plt.title(f"RR of Layer {layer} vs Repetitions")
  plt.ylabel("RR")
  plt.xlabel("Repetitions")
  if save_fig and save_path:
    print('saving figure')
    plt.savefig(save_path)
  if show:
    plt.show()
  plt.close()

def plot_loss_vs_repetitions(results, sequence_position, offset=0, 
                             show=True, save_fig=False, save_path=None):
  x = list(range(len(results[sequence_position]["losses"])))
  y = results[sequence_position]["losses"]
  plt.scatter(x[offset:], y[offset:], c="green", marker="^")
  plt.legend()
  plt.title("Loss vs Repetitions")
  plt.ylabel("Loss")
  plt.xlabel("Repetitions")
  if save_fig and save_path:
    plt.savefig(save_path)
  if show:
    plt.show()
  plt.close()
    
def plot_curvature_loss_vs_repetitions(results, sequence_position, offset=0,
                                        show=True, save_fig=False, save_path=None):
  x = list(range(len(results[sequence_position]["curvatures"])))
  curvatures = results[sequence_position]["curvatures"] 
  losses = results[sequence_position]["losses"]
  
  fig, ax1 = plt.subplots()
  ax1.scatter(x[offset:], curvatures[offset:], label='curvature', color='b')
  ax1.set_xlabel('Repetitions')
  ax1.set_ylabel('curvature', color='b')
  ax1.tick_params(axis='y', labelcolor='b')

  ax2 = ax1.twinx()

  ax2.scatter(x[offset:], losses[offset:], label='loss', color='r')
  ax2.set_ylabel('loss', color='r')
  ax2.tick_params(axis='y', labelcolor='r')

  lines1, labels1 = ax1.get_legend_handles_labels()
  lines2, labels2 = ax2.get_legend_handles_labels()
  lines = lines1 + lines2
  labels = labels1 + labels2
  ax1.legend(lines, labels, loc='upper right')

  plt.title("Curvature and Loss vs Repetitions")
  if save_fig and save_path:
    plt.savefig(save_path)
  if show:
    plt.show()
  plt.close()
  return fig
    
def plot_layer_curvature_loss_vs_repetitions(results, sequence_position, layer, offset=0,
                                        show=True, save_fig=False, save_path=None):
  x = list(range(len(results[sequence_position]["curvatures"])))
  curvatures = results[sequence_position]["curvatures"] 
  losses = results[sequence_position]["losses"]
  
  fig, ax1 = plt.subplots()
  ax1.scatter(x[offset:], [curv[layer] for curv in curvatures[offset:]], label=f'curvature_layer_{layer}', color='b')
  ax1.set_xlabel('Repetitions')
  ax1.set_ylabel(f'layer {layer} curvature', color='b')
  ax1.tick_params(axis='y', labelcolor='b')

  ax2 = ax1.twinx()

  ax2.scatter(x[offset:], losses[offset:], label='loss', color='r')
  ax2.set_ylabel('loss', color='r')
  ax2.tick_params(axis='y', labelcolor='r')
    
  lines1, labels1 = ax1.get_legend_handles_labels()
  lines2, labels2 = ax2.get_legend_handles_labels()
    
  lines =  lines1 + lines2
  labels =  labels1 + labels2
  ax1.legend(lines, labels, loc='upper right')

  plt.title("Curvature and Loss vs Repetitions")
  if save_fig and save_path:
    plt.savefig(save_path)
  if show:
    plt.show()
  plt.close()
  return fig
  
def plot_curvature_layers(results, sequence_position, repetition, show=True, save_fig=False, save_path=None):
  NUM_LAYERS = 12
  curvatures = results[sequence_position]["curvatures"] 
  losses = results[sequence_position]["losses"]
  
  fig, ax1 = plt.subplots()
  x = list(range(NUM_LAYERS))
  ax1.scatter(x, [curvatures[repetition][layer_val] for layer_val in x], label=f'layerwise_curvature_reptition={repetition}', color='b')
  ax1.set_xlabel('Layer')
  ax1.set_ylabel(f'layer curvature for repetition {repetition}', color='b')
  ax1.tick_params(axis='y', labelcolor='b')

  lines, labels = ax1.get_legend_handles_labels()
    
  plt.title(f"Layer Curvature for Repetition {repetition}")
  if save_fig and save_path:
    plt.savefig(save_path)
  if show:
    plt.show()
  plt.close()
  return fig
  
  
def plot_logit_lens(cache, prompts, answer_tokens, model, cumulative=False, show=True, save_fig=False, save_path=None):
  answer_residual_directions = model.tokens_to_residual_directions(answer_tokens)
  logit_diff_directions = answer_residual_directions[:, 0] - answer_residual_directions[:, 1]

  if cumulative:
    accumulated_residual, labels = cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)
    logit_lens_logit_diffs = residual_stack_to_logit_diff(accumulated_residual, cache, logit_diff_directions, prompts)
    plt.plot(np.arange(model.cfg.n_layers*2+1)/2, tutils.to_numpy(logit_lens_logit_diffs))
    plt.xticks(np.arange(model.cfg.n_layers*2+1)/2, labels, rotation=90)
    plt.title("Logit Difference From Accumulate Residual Stream")
  else:
    per_layer_residual, labels = cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
    per_layer_logit_diffs = residual_stack_to_logit_diff(per_layer_residual, cache, logit_diff_directions, prompts)
    plt.plot(np.arange(model.cfg.n_layers*2+2)/2, tutils.to_numpy(per_layer_logit_diffs))
    plt.title("Logit Difference From Each Layer")
    plt.xticks(np.arange(model.cfg.n_layers*2+2)/2, labels, rotation=90)
  if save_fig and save_path:
    print('saving figure')
    plt.savefig(save_path)
  if show:
    plt.show()
  plt.close()

##### USEFUL FXNS FOR FIRST PASS CAPITALS #####

def plot_curvature_vs_icl_examples(dataset_curvatures, show=True, save_fig=False, save_path=None):
  '''Plots the global curvature of the residual stream as a function of the number of ICL examples.'''
  curve_lang0 = dataset_curvatures[0]
  curve_lang1 = dataset_curvatures[1]
  curve_lang2 = dataset_curvatures[2]
  curve_lang3 = dataset_curvatures[3]

  curve_c_lang1 = dataset_curvatures[4]
  curve_c_lang2 = dataset_curvatures[5]
  curve_c_lang3 = dataset_curvatures[6]

  curve_an0 = dataset_curvatures[7]
  curve_an1 = dataset_curvatures[8]
  curve_an2 = dataset_curvatures[9]
  curve_an3 = dataset_curvatures[10]

  curve_c_an1 = dataset_curvatures[11]
  curve_c_an2 = dataset_curvatures[12]
  curve_c_an3 = dataset_curvatures[13]

  x = [0] * len(curve_lang0) + [1] * len(curve_lang1) + [2] * len(curve_lang2) + [3] * len(curve_lang3)
  x_counter = x[len(curve_lang0):]

  plt.scatter(x, curve_lang0 + curve_lang1 + curve_lang2 + curve_lang3, c="green", marker="^", label="Language")
  plt.scatter(x_counter, curve_c_lang1 + curve_c_lang2 + curve_c_lang3, c="blue", marker="^", label="Counterfactual Language")

  plt.scatter(x, curve_an0 + curve_an1 + curve_an2 + curve_an3, c="red", marker="o", label="Analogy")
  plt.scatter(x_counter, curve_c_an1 + curve_c_an2 + curve_c_an3, c="orange", marker="o", label="Counterfactual Analogy")

  plt.legend()
  plt.title("Global Curvature of the Residual Stream")
  plt.ylabel("Global Curvature")
  plt.xlabel("Number of ICL Examples")
  plt.xticks([0, 1, 2, 3])
  if save_fig and save_path:
    plt.savefig(save_path)
  if show:
    plt.show()
  plt.close()
    
def plot_global_curvature_dissected(dataset_curvatures, dataset_mlp_curvatures, dataset_attn_curvatures,
                                    show=True, save_fig=False, save_path=None):
  curve_lang0 = dataset_curvatures[0]
  curve_lang1 = dataset_curvatures[1]
  curve_lang2 = dataset_curvatures[2]
  curve_lang3 = dataset_curvatures[3]
  
  x = [0] * len(curve_lang0) + [1] * len(curve_lang1) + [2] * len(curve_lang2) + [3] * len(curve_lang3)

  plt.scatter(x, curve_lang0 + curve_lang1 + curve_lang2 + curve_lang3, c="green", marker="^", label="Full Model, Language")

  curve_lang0 = dataset_mlp_curvatures[0]
  curve_lang1 = dataset_mlp_curvatures[1]
  curve_lang2 = dataset_mlp_curvatures[2]
  curve_lang3 = dataset_mlp_curvatures[3]

  plt.scatter(x, curve_lang0 + curve_lang1 + curve_lang2 + curve_lang3, c="blue", marker="^", label="MLPs, Language")

  curve_lang0 = dataset_attn_curvatures[0]
  curve_lang1 = dataset_attn_curvatures[1]
  curve_lang2 = dataset_attn_curvatures[2]
  curve_lang3 = dataset_attn_curvatures[3]

  plt.scatter(x, curve_lang0 + curve_lang1 + curve_lang2 + curve_lang3, c="red", marker="^", label="ATTNs, Language")

  plt.legend()
  plt.title("Global Curvature of the Residual Stream")
  plt.ylabel("Global Curvature")
  plt.xlabel("Number of ICL Examples")
  plt.xticks([0, 1, 2, 3])
  if save_fig and save_path:
    plt.savefig(save_path)
  if show:
    plt.show()
  plt.close()
  
def plot_clusters(dataset_curvatures, sample_sequences, cluster_labels, plotly=True, hover=False, show=True, save_fig=False, save_path=None):
  
  pca = PCA(n_components=3)
  pca_curvatures = pca.fit_transform(dataset_curvatures)  
  df_pca = pd.DataFrame(pca_curvatures, columns=['PC1', 'PC2', 'PC3'])
  df_pca['Cluster'] = cluster_labels 
  
  if plotly: 
    if hover:
      fig = px.scatter_3d(
      df_pca, x='PC1', y='PC2', z='PC3',
      title='Clustered Curvature Graph',
      labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2', 'PC3': 'Principal Component 3'},
      hover_name=sample_sequences, 
      color='Cluster'
      )
    else:
      fig = px.scatter_3d(
        df_pca, x='PC1', y='PC2', z='PC3',
        title='Clustered Curvature Graph',
        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2', 'PC3': 'Principal Component 3'},
        text=sample_sequences, 
        color='Cluster'
        )
    fig.update_traces(textfont=dict(size=8))  # Set the font size for point labels

    if save_fig and save_path:
      pio.write_html(fig, save_path)
  else:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for cluster_label in set(cluster_labels):
      cluster_indices = (cluster_labels == cluster_label)
      ax.scatter(pca_curvatures[cluster_indices, 0], 
                pca_curvatures[cluster_indices, 1], 
                pca_curvatures[cluster_indices, 2], 
                label=f'Cluster {cluster_label}')
      
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title("Clustered Curvature Graph")
    ax.legend()
    if save_fig and save_path:
      plt.savefig(save_path)
    if show:
      plt.show()
    plt.close()