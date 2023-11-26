import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns

import os
import json
import pandas as pd

layer_list = ['layer-0', 'layer-1', 'layer-2', 
              'layer-3', 'layer-4', 'layer-5', 
              'layer-6', 'layer-7', 'layer-8', 
              'layer-9', 'layer-10', 'layer-11', 
              'layer-12']

layer_name_dict = {k:f'encoder.layer.{int(k.split("-")[1]) - 1}' if k != 'layer-0' else 'embeddings' for k in layer_list}

def parse_val_acc_epoch(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    content = content.replace("'", '"')
    try:
        data = json.loads(content)
        val_acc_epoch = data[0]['val_acc_epoch']
        return val_acc_epoch
    except json.JSONDecodeError as e:
        print(f"An error occurred while parsing JSON: {e}")
        return None
      
def parse_structural_metrics(file_path, parsestr):
    accuracy = None
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith(parsestr):
                accuracy = float(line.split(parsestr[-1])[1].strip())
                break 
    return accuracy
  
parse_depth_acc = lambda file_path: parse_structural_metrics(file_path, 'Avg Acc:')
parse_depth_spr = lambda file_path: parse_structural_metrics(file_path, 'Avg Depth DSpr.: ')
parse_dist_uuas = lambda file_path: parse_structural_metrics(file_path, 'Avg UUAS:')
parse_dist_spr = lambda file_path: parse_structural_metrics(file_path, 'Avg Distance DSpr.:')

def find_directories(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def find_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
  
def collate_validation_accuracy(root_dir):
  data = []
  for subdir in find_directories(root_dir):
    step = subdir.split("_")[-1][:-1] 
    ## number of steps removing k at end
    if set(find_directories(os.path.join(root_dir, subdir))) == set(layer_list):
      for layer in layer_list:
        filename = find_files(os.path.join(root_dir, subdir, layer))[0]
        file_path = os.path.join(root_dir, subdir, layer, filename)
        if filename == 'val_acc.txt':
          val_acc = parse_val_acc_epoch(file_path)
          data.append({'Step': step, 'Layer': layer, 'Val Acc': val_acc})
        elif filename == 'val_metrics_depth.txt':
          val_acc = parse_depth_acc(file_path)
          val_spr = parse_depth_spr(file_path)
          data.append({'Step': step, 'Layer': layer, 'Root Acc': val_acc, 'NSpr': val_spr})
        elif filename == 'val_metrics_distance.txt':
          val_uuas = parse_dist_uuas(file_path)
          val_spr = parse_dist_spr(file_path)
          data.append({'Step': step, 'Layer': layer, 'UUAS': val_uuas, 'DSpr': val_spr})
  return data
  
def main(FLAGS):  
  home = os.environ['LEARNING_DYNAMICS_HOME']
  root_dir = os.path.join(home, "outputs", FLAGS.dataset, FLAGS.exp)
  
  df = pd.DataFrame(collate_validation_accuracy(root_dir))
  df['Step'] = pd.to_numeric(df['Step'])
  df['Layer'] =pd.to_numeric( df['Layer'].apply(lambda x: x.split("-")[1]))
  df['Layer'] = df['Layer'].apply(lambda x: layer_name_dict[f'layer-{x}'])
  df.sort_values(by=['Step', 'Layer'], inplace=True)
  df = df.pivot(columns='Step', index='Layer', values=FLAGS.metric)
  
  if FLAGS.save == 'True':
    output_filename = os.path.join(root_dir, f"{FLAGS.metric.replace(' ', '_')}.csv")
    df.to_csv(output_filename, index=True, header=True, sep='\t')
    
  if FLAGS.plot == 'plotly' or FLAGS.plot == 'both': 
    df.columns = df.columns.astype(str)
    fig = go.Figure(data=go.Heatmap(
    z=df.values,
    x=df.columns,
    y=df.index,
    colorscale='Viridis'
    ))
    fig.update_layout(
        xaxis_title='Step (In Thousands)',
        yaxis_title='Layer'
    )
    output_filename = os.path.join(root_dir, f"{FLAGS.metric.replace(' ', '_')}__heatmap.json")
    fig.write_json(output_filename)
    
  elif FLAGS.plot == 'plt' or FLAGS.plot == 'both':
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt=".2f", cmap='viridis', annot_kws={"size": 10})
    plt.xlabel('Step (In Thousands)')
    plt.ylabel('Layer')
    plt.tight_layout()
    output_filename = os.path.join(root_dir, f"{FLAGS.metric.replace(' ', '_')}__heatmap.png")
    plt.savefig(output_filename, dpi=300)
    plt.close()
    

if __name__ == "__main__":
  argparser = argparse.ArgumentParser()
  argparser.add_argument("--save", type=str, default="True", help="save to csv")
  argparser.add_argument("--dataset", type=str, default="en_ewt-ud", help="en_ewt-ud, ptb_3, ontonotes")
  argparser.add_argument("--exp", type=str, default="cpos", help="experiment name")
  argparser.add_argument("--metric", type=str, default="Val Acc", help="Val Acc, Root Acc, UUAS, NSpr, DSpr")
  argparser.add_argument("--plot", type=str, default="both")
  FLAGS = argparser.parse_args()
  main(FLAGS)