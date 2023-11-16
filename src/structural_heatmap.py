import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
def find_yaml_files(directory):
    yaml_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.yaml'):
                yaml_files.append(file)
    return yaml_files
  
def main(FLAGS):
  heatmap_data = {}
  home = os.environ['LEARNING_DYNAMICS_HOME']
  directory = os.path.join(home, FLAGS.directory)
  
  for foldername in os.listdir(directory):
    for innerfolder in os.listdir(os.path.join(directory, foldername)):
      
      metric_file_path = os.path.join(directory, foldername, innerfolder, FLAGS.metricfile)
      if os.path.isfile(metric_file_path) and (FLAGS.loss in innerfolder):
        config_yaml = find_yaml_files(os.path.join(directory, foldername, innerfolder))[0]
        # print(config_yaml)
        parts = config_yaml.split('_')
        
        layer = int(parts[2]) - 1
        step = int(parts[4])
        
        with open(metric_file_path, 'r') as file:
          if FLAGS.metricfile == 'dev.root_acc':
            value = float(file.read().strip().split('\t')[0])
          else:
            value = float(file.read().strip())
        
        if layer not in heatmap_data:
            heatmap_data[layer] = {}
        heatmap_data[layer][step] = value
        
  metric = FLAGS.metricfile.split('.')[1]
 
  df = pd.DataFrame.from_dict(heatmap_data, orient='index')
  df = df.sort_index(ascending=False).sort_index(axis=1)
  df.index = df.index.astype(str).map(lambda x: f"encoder.layer.{x}")
  if FLAGS.plot == 'plotly': 
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
    output_filename = os.path.join(home, f"outputs/structural/results/{FLAGS.loss}/val_{metric}_heatmap.json")
    fig.write_json(output_filename)
    print(f"Heatmap saved as {output_filename}")
    
  else:
    
    plt.figure(figsize=(10, 8))
    
    ax = sns.heatmap(df, annot=True, fmt=".2f", cmap='viridis', annot_kws={"size": 10})
    plt.xlabel('Step (In Thousands)')
    plt.ylabel('Layer')
    plt.tight_layout()

    output_filename = os.path.join(home, f"outputs/structural/results/{FLAGS.loss}/val_{metric}_heatmap.png")
    plt.savefig(output_filename, dpi=300)
    print(f"Heatmap saved as {output_filename}")

    plt.close()
    

if __name__ == "__main__":
  argparser = argparse.ArgumentParser()
  argparser.add_argument("--directory", type=str, default="outputs/structural/results/")
  argparser.add_argument("--metricfile", type=str, default="dev.uuas")
  argparser.add_argument("--loss", type=str, default="dist")
  argparser.add_argument("--plot", type=str, default="plotly")
  FLAGS = argparser.parse_args()
  main(FLAGS)