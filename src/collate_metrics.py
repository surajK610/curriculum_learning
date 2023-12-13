import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.patches as patches

import os
import json
import pandas as pd

layer_list = ['layer-0', 'layer-1', 'layer-2', 
              'layer-3', 'layer-4', 'layer-5', 
              'layer-6', 'layer-7', 'layer-8', 
              'layer-9', 'layer-10', 'layer-11', 
              'layer-12']
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
paths_list = [
  'outputs/en_ewt-ud/cpos/Val_Acc.csv',
  'outputs/en_ewt-ud/fpos/Val_Acc.csv',
  'outputs/en_ewt-ud/dep/Val_Acc.csv',
  'outputs/ontonotes/ner/Val_Acc.csv',
  'outputs/ontonotes/phrase_start/Val_Acc.csv',
  'outputs/ontonotes/phrase_end/Val_Acc.csv',
  'outputs/ptb_3/depth/NSpr.csv',
  'outputs/ptb_3/distance/DSpr.csv',
]

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
    metric = None
    with open(file_path, 'r') as file:
      for line in file:
        if line.startswith(parsestr):
          metric = float(line.split(":")[1].strip())
          break 
    return metric
  
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
  
  if FLAGS.line_graph == 'True':
    fig = go.Figure()
    layer_order = []
    for i, path in enumerate(paths_list):
      df = pd.read_csv(os.path.join(home, path), sep='\t', index_col=0)
      color = colors[i]
      max_changes = []
      
      for idx, col in enumerate(df.columns):
        if not layer_order:
          layer_order = df.index.tolist()
      
        changes = df[col].diff().abs() 
        max_change_idx = changes.idxmax()  
        if pd.notnull(max_change_idx) and (df.index.get_loc(max_change_idx)) > 0:
          following_idx = df.index[df.index.get_loc(max_change_idx) - 1]
        else:
          following_idx = max_change_idx
       
        fig.add_trace(go.Scatter(
            x=[idx, idx+1],
            y=[following_idx, following_idx],
            mode='lines',
            line=dict(color=color, width=20-2*i),
            opacity=1,
            showlegend=True if idx == 0 else False,  
            name=path.split("/")[2]  
        ))
        max_changes.append(max_change_idx)
    fig.update_layout(
        xaxis_title='Step (In Thousands)',
        yaxis_title='Layer',
        legend_title="Tasks",
        yaxis=dict(
          type='category',
          categoryorder='array',
          categoryarray=layer_order
      )
    )

    root_dir = os.path.join(home, "outputs")
    output_filename = os.path.join(root_dir, "line-graph.json")
    fig.write_json(output_filename)
    return 
  
  if FLAGS.path_to_df is None:
    root_dir = os.path.join(home, "outputs", FLAGS.dataset, FLAGS.exp)
    
    df = pd.DataFrame(collate_validation_accuracy(root_dir))
    df['Step'] = pd.to_numeric(df['Step'])
    df['Layer'] =pd.to_numeric( df['Layer'].apply(lambda x: x.split("-")[1]))
    df = df.pivot(columns='Step', index='Layer', values=FLAGS.metric)
    df.sort_index(axis=1, inplace=True)
    df.sort_index(axis=0, inplace=True, ascending=False)
    df.index= df.index.map(lambda x: layer_name_dict[f'layer-{x}'])
    
    if FLAGS.save == 'True':
      output_filename = os.path.join(root_dir, f"{FLAGS.metric.replace(' ', '_')}.csv")
      df.to_csv(output_filename, index=True, header=True, sep='\t')
      
  else:
    path_to_df = os.path.join(home, "outputs", FLAGS.path_to_df)
    root_dir = os.path.dirname(path_to_df)
    df = pd.read_csv(path_to_df, sep='\t', index_col=0)
    df.sort_index(axis=0, inplace=True, ascending=False)
    df.index = df.index.map(lambda x: layer_name_dict[f'layer-{x+1}'])
    
  if (FLAGS.plot == 'plotly') or (FLAGS.plot == 'both'):
    df.columns = df.columns.astype(str)
    fig = go.Figure(data=go.Heatmap(
      z=df.values,
      x=df.columns,
      y=df.index,
      colorscale='Viridis'
    ))

    for idx, col in enumerate(df.columns):
      max_val = df[col].max()
      max_val_row = df.index.get_loc(df[col].idxmax())
      fig.add_shape(type='rect',
                    x0=idx-0.5, y0=max_val_row-0.5,
                    x1=idx+0.5, y1=max_val_row+0.5,
                    line=dict(color='White'))
  
      changes = df[col].diff().abs() 
      max_change_idx = changes.idxmax()  

      fig.add_shape(type='line',
                    x0=idx-0.5, y0=df.index.get_loc(max_change_idx)-0.5,
                    x1=idx+0.5, y1=df.index.get_loc(max_change_idx)-0.5,
                    line=dict(color='Red', width=3))
        
    fig.update_layout(
      xaxis_title='Step (In Thousands)',
      yaxis_title='Layer'
    ) 
    output_filename = os.path.join(root_dir, f"{FLAGS.metric.replace(' ', '_')}_heatmap.json")
    fig.write_json(output_filename)
    
  if (FLAGS.plot == 'plt') or (FLAGS.plot == 'both'):
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(df, annot=False, fmt=".2f", cmap='viridis', annot_kws={"size": 10})
    plt.xlabel('Step (In Thousands)')
    plt.ylabel('Layer')
    for col_idx, col in enumerate(df.columns):
        changes = df[col].diff().abs()  
        max_change_row_idx = np.where(df.index == changes.idxmax())[0][0]  
        next_row_idx = max_change_row_idx + 1 
        ax.plot([col_idx, col_idx+1], [max_change_row_idx, max_change_row_idx], color='red', lw=3)

        max_val_row = df.index.get_loc(df[col].idxmax())
        ax.add_patch(patches.Rectangle((col_idx, max_val_row), 1, 1, fill=False, edgecolor='white', lw=2))
    plt.tight_layout()
    output_filename = os.path.join(root_dir, f"{FLAGS.metric.replace(' ', '_')}_heatmap.png")
    plt.savefig(output_filename, dpi=300)
    plt.close()
    

if __name__ == "__main__":
  argparser = argparse.ArgumentParser()
  argparser.add_argument("--save", type=str, default="True", help="save to csv")
  argparser.add_argument("--dataset", type=str, default="en_ewt-ud", help="en_ewt-ud, ptb_3, ontonotes")
  argparser.add_argument("--exp", type=str, default="cpos", help="experiment name")
  argparser.add_argument("--metric", type=str, default="Val Acc", help="Val Acc, Root Acc, UUAS, NSpr, DSpr")
  argparser.add_argument("--plot", type=str, default="both")
  argparser.add_argument("--path-to-df", type=str, default=None)
  argparser.add_argument("--line-graph", type=str, default="False")
  FLAGS = argparser.parse_args()
  main(FLAGS)