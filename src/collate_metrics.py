import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import seaborn as sns
import matplotlib.patches as patches

import os
import json
import pandas as pd

from kneed.knee_locator import KneeLocator

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


def get_pushdown(df):
  layer_knees = {}
  min_knee, max_knee, sum_diffs = 100000, 0, 0
  x = df['Step']
  df['Layer'] = df['Layer'].apply(lambda x: int(x.split("-")[1]))
  df.index = df['Layer']
  df.drop('Layer', axis=1, inplace=True)
  
  for idx in df.index:
    y = df.iloc[idx]
    layer_knees[idx] = KneeLocator(x, y).knee
    if layer_knees[idx] < min_knee:
      min_knee = layer_knees[idx]
    if layer_knees[idx] > max_knee:
      max_knee = layer_knees[idx]
  
  for idx in range(1, len(layer_knees)):
    sum_diffs += layer_knees[idx] - layer_knees[idx-1]
  
  return min_knee, max_knee, sum_diffs


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

def collate_validation_accuracy(root_dir, seed, dataset, model, exp, resid, attention_head=None):
  data = []
  dirs = find_directories(root_dir)
  if model == 'pythia':
    dirs = [d for d in dirs if d.startswith('pythia')]
  elif model == 'bert':
    dirs = [d for d in dirs if 'multibert' in d and seed in d]
  for subdir in dirs:
    step = subdir.split("_")[-1][:-1] 
    if dataset == 'aheads' and model == 'pythia':
      step = subdir.split("_")[-1] # no k in back
    elif model == 'pythia':
      step = subdir.split("step")[-1]
    if set(find_directories(os.path.join(root_dir, subdir))) == set(layer_list):
      for layer in layer_list:
        filenames = find_files(os.path.join(root_dir, subdir, layer))
        if exp == 'depth':
          filename = 'val_metrics_depth.txt' if resid else f'val_metrics_depth_out{"_head_"+str(attention_head) if attention_head is not None else ""}.txt'
          if filename not in filenames:
            print(f"File {filename} not found in {os.path.join(root_dir, subdir, layer)}")
            continue
          assert filename in filenames
          file_path = os.path.join(root_dir, subdir, layer, filename)
          val_acc = parse_depth_acc(file_path)
          val_spr = parse_depth_spr(file_path)
          data.append({'Step': step, 'Layer': layer, 'Root Acc': val_acc, 'NSpr': val_spr})
        elif exp == 'distance':
          filename = 'val_metrics_distance.txt' if resid else f'val_metrics_distance_out{"_head_"+str(attention_head) if attention_head is not None else ""}.txt'
          if filename not in filenames:
            print(f"File {filename} not found in {os.path.join(root_dir, subdir, layer)}")
            continue
          assert filename in filenames
          file_path = os.path.join(root_dir, subdir, layer, filename)
          val_uuas = parse_dist_uuas(file_path)
          val_spr = parse_dist_spr(file_path)
          data.append({'Step': step, 'Layer': layer, 'UUAS': val_uuas, 'DSpr': val_spr})
        else:
          filename = 'val_acc.txt' if resid else f'val_acc_out{"_head_"+str(attention_head) if attention_head is not None else ""}.txt'
          if filename not in filenames:
            print(f"File {filename} not found in {os.path.join(root_dir, subdir, layer)}")
            continue
          assert filename in filenames
          file_path = os.path.join(root_dir, subdir, layer, filename)
          val_acc = parse_val_acc_epoch(file_path)
          data.append({'Step': step, 'Layer': layer, 'Val Acc': val_acc})
  return data
  
def main(FLAGS):  
  home = os.environ['LEARNING_DYNAMICS_HOME']
  resid = FLAGS.resid == 'True'
  print(resid)
  if FLAGS.gen_figure == 'True':
    file_paths = {
      'pythia_syntax': [
        'outputs/en_ewt-ud/seed_0/cpos/Val_Acc_pythia.csv',
        'outputs/en_ewt-ud/seed_0/fpos/Val_Acc_pythia.csv',
        'outputs/en_ewt-ud/seed_0/dep/Val_Acc_pythia.csv'
      ],
      'semantic':
        ['outputs/en_ewt-ud/seed_0/cpos/Val_Acc.csv',
        'outputs/en_ewt-ud/seed_0/fpos/Val_Acc.csv',
        'outputs/ontonotes/seed_0/ner/Val_Acc.csv'],
      'syntax':
        ['outputs/ontonotes/seed_0/phrase_start/Val_Acc.csv',
        'outputs/ontonotes/seed_0/phrase_end/Val_Acc.csv',
        'outputs/en_ewt-ud/seed_0/dep/Val_Acc.csv',
        'outputs/ptb_3/seed_0/depth/Root_Acc.csv',
        'outputs/ptb_3/seed_0/distance/UUAS.csv'
        ],
      'algorithmic':
        ['outputs/aheads/duplicate_token_head/Val_Acc.csv', 
         'outputs/aheads/induction_head/Val_Acc.csv',
         'outputs/aheads/previous_token_head/Val_Acc.csv']
    }
    if not resid:
      file_paths = {k: [s.replace('.', '_out.') for s in v] for k, v in file_paths.items()}
      print(file_paths.keys())
      
    print(file_paths.items())
    for key, fps in file_paths.items():
      print(key)
      n = len(fps)
      titles = [s.split("/")[3] for s in fps]
      if key == 'algorithmic':
        titles = ['dup', 'ind', 'prev']
      elif key == 'syntax':
        titles = ['phrase start', 'phrase end', 'dep', 'depth', 'dist']
      dataframes = [pd.read_csv(file_path, delimiter='\t').drop('Layer', axis=1) for file_path in fps]
      fig, axs = plt.subplots(1, n, figsize=(5*n, 5), sharey=True)

      for i, (ax, df, title) in enumerate(zip(axs, dataframes, titles)):
          df.index = df.index[::-1]
          if 'pythia' in fps[i]:
            cols = [col for col in df.columns if int(col) >= 64]
            df = df[cols]
            
          if not resid:
            df = df[['0', '20', '40', '60', '80', '100', '200', '1000', '1400', '1600', '1800', '2000']]
            # just for now to ensure consistency of step samples
          sns.heatmap(df, ax=ax, annot_kws={"size":16})
          cbar = ax.collections[0].colorbar
          cbar.ax.tick_params(labelsize=13)
          ax.set_title(title, fontsize=24)
          if i > 0:
            ax.set_ylabel('') 
          else:
            ax.set_ylabel('Layer', fontsize=22)
          ax.tick_params(axis='both', which='major', labelsize=16)
      
      plt.tight_layout()
      output_filename = os.path.join(home, "figures", f"{key}{'' if resid else '_out'}.png")
      plt.savefig(output_filename)
      plt.close()
    return 
    
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
    resid = FLAGS.resid == 'True'
    root_dir = os.path.join(home, "outputs", FLAGS.dataset, FLAGS.seed, FLAGS.exp)
    df = pd.DataFrame(collate_validation_accuracy(root_dir, FLAGS.seed, FLAGS.dataset, FLAGS.model, FLAGS.exp, resid, FLAGS.attention_head))
    df['Step'] = pd.to_numeric(df['Step'])
    df['Layer'] = pd.to_numeric( df['Layer'].apply(lambda x: x.split("-")[1]))
    df = df.pivot(columns='Step', index='Layer', values=FLAGS.metric)
    df.sort_index(axis=1, inplace=True)
    df.sort_index(axis=0, inplace=True, ascending=False)
    df.index= df.index.map(lambda x: layer_name_dict[f'layer-{x}'])
    print(df.columns)
    if not resid:
        df = df[[0, 20, 40, 60, 80, 100, 200, 1000, 1400, 1600, 1800, 2000]] # just for now to ensure consistency of step samples
    if FLAGS.save == 'True':
      if resid:
        output_filename = os.path.join(root_dir, f"{FLAGS.metric.replace(' ', '_')}{'' if resid else '_out'}.csv")
      else:
        os.makedirs(os.path.join(root_dir, "components"), exist_ok=True)
        output_filename = os.path.join(root_dir, "components", f"{FLAGS.metric.replace(' ', '_')}_heatmap{'' if resid else '_out'}{'_head_'+str(FLAGS.attention_head) if FLAGS.attention_head is not None else ''}.csv")
      df.to_csv(output_filename, index=True, header=True, sep='\t')
  else:
    path_to_df = os.path.join(home, "outputs", FLAGS.seed, FLAGS.path_to_df)
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
    if resid:
      output_filename = os.path.join(root_dir, f"{FLAGS.metric.replace(' ', '_')}_heatmap{'_pythia' if FLAGS.model == 'pythia' else ''}{'' if resid else '_out'}.json")
    else:
      output_filename = os.path.join(root_dir, "components", f"{FLAGS.metric.replace(' ', '_')}_heatmap{'' if resid else '_out'}{'_head_'+str(FLAGS.attention_head) if FLAGS.attention_head is not None else ''}.json")
    # output_filename = os.path.join(root_dir, f"{FLAGS.metric.replace(' ', '_')}_heatmap.json")
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
    if resid:
      output_filename = os.path.join(root_dir, f"{FLAGS.metric.replace(' ', '_')}_heatmap{'_pythia' if FLAGS.model == 'pythia' else ''}{'' if resid else '_out'}.png")
    else:
      output_filename = os.path.join(root_dir, "components", f"{FLAGS.metric.replace(' ', '_')}_heatmap{'' if resid else '_out'}{'_head_'+str(FLAGS.attention_head) if FLAGS.attention_head is not None else ''}.png")
    # output_filename = os.path.join(root_dir, f"{FLAGS.metric.replace(' ', '_')}_heatmap.png")
    plt.savefig(output_filename, dpi=300)
    plt.close()
    

if __name__ == "__main__":
  argparser = argparse.ArgumentParser()
  argparser.add_argument("--save", type=str, default="True", help="save to csv")
  argparser.add_argument("--dataset", type=str, default="en_ewt-ud", help="en_ewt-ud, ptb_3, ontonotes")
  argparser.add_argument("--model", type=str, default="bert", help="bert, pythia")
  argparser.add_argument("--exp", type=str, default="cpos", help="experiment name")
  argparser.add_argument("--metric", type=str, default="Val Acc", help="Val Acc, Root Acc, UUAS, NSpr, DSpr")
  argparser.add_argument("--plot", type=str, default="both")
  argparser.add_argument("--path-to-df", type=str, default=None)
  argparser.add_argument("--line-graph", type=str, default="False")
  argparser.add_argument("--gen-figure", type=str, default="False")
  argparser.add_argument("--resid", type=str, default="True")
  argparser.add_argument("--attention-head", type=int, default=None)
  argparser.add_argument("--seed", type=str, default="seed_0")
  FLAGS = argparser.parse_args()
  main(FLAGS)