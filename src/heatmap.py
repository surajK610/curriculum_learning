import glob
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import re
import argparse

def extract_layer_number(layer_name, layer_pattern):
  match = layer_pattern.search(layer_name)
  if match:
    return int(match.group(0))
  else:
    return -1
        
def main(FLAGS):
  home = os.environ['LEARNING_DYNAMICS_HOME']
  file_pattern = os.path.join(home, f"outputs/{FLAGS.type}/results/probe-linear-step=*-seed=0.json")
  file_paths = glob.glob(file_pattern)

  heatmap_data = []
  pattern = re.compile(r"step=(\d+)-seed=0.json")
  layer_pattern = re.compile(r"\d+")

  for file_path in file_paths:
    match = pattern.search(os.path.basename(file_path))
    if match:
      step = int(match.group(1))
    else:
      print(f"Filename pattern does not match for file: {file_path}")
      continue

      # Read the JSON file
    with open(file_path, 'r') as f:
      data = json.load(f)

    for layer, scores in data.items():
      layer_number = extract_layer_number(layer, layer_pattern)
      heatmap_data.append({
        'step': step,
        'layer': layer,
        'layer_number': layer_number,
        'val_acc': scores[0]['val_acc'] 
      })

  heatmap_data = pd.DataFrame(heatmap_data)
  heatmap_data_layers_to_nums = dict(zip(heatmap_data['layer_number'], heatmap_data['layer']))
  heatmap_data_pivot = heatmap_data.pivot(index='layer_number', columns='step', values='val_acc').drop('layer_number', axis=1, errors='ignore')
  heatmap_data_pivot.sort_index(axis=0, inplace=True, ascending=False)
  heatmap_data_pivot.index = heatmap_data_pivot.index.map(heatmap_data_layers_to_nums)

  heatmap_data_pivot.columns = heatmap_data_pivot.columns.astype(int)  
  heatmap_data_pivot.sort_index(axis=1, inplace=True, ascending=True)

  if FLAGS.plot == 'plotly': 
    heatmap_data_pivot.columns = heatmap_data_pivot.columns.astype(str)
    fig = go.Figure(data=go.Heatmap(
    z=heatmap_data_pivot.values,
    x=heatmap_data_pivot.columns,
    y=heatmap_data_pivot.index,
    colorscale='Viridis'
    ))

    fig.update_layout(
        xaxis_title='Step (In Thousands)',
        yaxis_title='Layer'
    )

    output_filename = os.path.join(home, f"outputs/{FLAGS.type}/results/val_acc_heatmap.json")
    fig.write_json(output_filename)
    print(f"Heatmap saved as {output_filename}")
    
  else:
    plt.figure(figsize=(10, 8))
    print(heatmap_data_pivot)
    ax = sns.heatmap(heatmap_data_pivot, annot=True, fmt=".2f", cmap='viridis', annot_kws={"size": 10})
    plt.xlabel('Step (In Thousands)')
    plt.ylabel('Layer')
    plt.tight_layout()

    output_filename = os.path.join(home, f"outputs/{FLAGS.type}/results/val_acc_heatmap.png")
    plt.savefig(output_filename, dpi=300)
    print(f"Heatmap saved as {output_filename}")

    plt.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--type", default="pos", type=str, required=False)
  parser.add_argument("--plot", default="plt", type=str, required=False)
  
  FLAGS = parser.parse_args()
  main(FLAGS)
  