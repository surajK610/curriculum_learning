import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def generate_heatmaps(folder_path, output_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    nrows = len(csv_files) // 3 + (len(csv_files) % 3 > 0)
    ncols = 3

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
    fig.tight_layout(pad=3.0)
    vmax = vmin = None

    for file in csv_files:
        df = pd.read_csv(os.path.join(folder_path, file), delimiter='\t').drop('Layer', axis=1)
        vmax = max(df.max().max(), vmax) if vmax is not None else df.max().max()
        vmin = min(df.min().min(), vmin) if vmin is not None else df.min().min()

    for i, file in enumerate(csv_files):
        df = pd.read_csv(os.path.join(folder_path, file), delimiter='\t').drop('Layer', axis=1) 
        df.index = df.index[::-1]
        ax = axes[i // ncols, i % ncols]
        sns.heatmap(df, ax=ax) #vmin=vmin, vmax=vmax, cbar=i == len(csv_files) - 1)
        ax.set_title(file.split('.')[0].split('heatmap_')[-1])  # Set subtitle as file name (without extension)

    for j in range(i + 1, nrows * ncols):
        axes[j // ncols, j % ncols].axis('off')
    plt.savefig(output_path)
    
def generate_heatmaps_unif(folder_path, output_path):
    # step_eval = list(range(0, 1010, 10)) + [1200, 1400, 1600, 1800, 2000, 2400, 3200, 4000, 6000, 8000, 16000, 28000]
    root_directory = 'outputs/toy_model/'
    csv_files = []
    for root, dirs, files in os.walk(root_directory):
        if os.path.basename(root).startswith('zipfw-sf'):
            for file in files:
                if file.endswith('results_val.csv'):
                    csv_files.append(os.path.join(root, file))
    
    nrows = len(csv_files) // 3 + (len(csv_files) % 3 > 0)
    ncols = 3

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
    fig.tight_layout(pad=3.0)
    vmax = vmin = None

    for file in csv_files:
        df = pd.read_csv(file)
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        vmax = max(df.max().max(), vmax) if vmax is not None else df.max().max()
        vmin = min(df.min().min(), vmin) if vmin is not None else df.min().min()

    for i, file in enumerate(csv_files):
        df = pd.read_csv(file)
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        df = df[::-1]
        df.columns = step_eval[:len(df.columns)]
        ax = axes[i // ncols, i % ncols]
        sns.heatmap(df, ax=ax) #vmin=vmin, vmax=vmax, cbar=i == len(csv_files) - 1)
        ax.set_title(file.split('/')[2].replace('unif-', ''))  # Set subtitle as file name (without extension)

    for j in range(i + 1, nrows * ncols):
        axes[j // ncols, j % ncols].axis('off')
    plt.savefig(output_path)
    
    
def generate_heatmaps_zipf(folder_path, output_path):
    # step_eval = list(range(0, 1010, 10)) + [1200, 1400, 1600, 1800, 2000, 2400, 3200, 4000, 6000, 8000, 16000, 28000]
    # step_eval = list(range(0, 1000, 10)) + list(range(1000, 16000, 100))
    
    root_directory = 'outputs/toy_model/'
    csv_files = []
    for root, dirs, files in os.walk(root_directory):
        if os.path.basename(root).startswith('zipfw-a'):
            for file in files:
                if file.endswith('results_tail.csv') and 'old' not in root and 'old_params' not in root and 'a_0' not in root and 'vs_10000-' in root:
                    csv_files.append(os.path.join(root, file))
    
    # csv_files = [f for f in csv_files if '100-' not in f] ## remove 100- files
    print(csv_files)
    nrows = len(csv_files) // 3 + (len(csv_files) % 3 > 0)
    ncols = 3

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 3.5 * nrows))
    fig.tight_layout(pad=5.0)
    vmax = vmin = None

    for file in csv_files:
        df = pd.read_csv(file)
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        vmax = max(df.max().max(), vmax) if vmax is not None else df.max().max()
        vmin = min(df.min().min(), vmin) if vmin is not None else df.min().min()

    for i, file in enumerate(csv_files):
        df = pd.read_csv(file)
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        df.index = df.index[::-1]
        # df = df[[str(x) for x in list(range(8000, 28000, 1000))]]
        ax = axes[i // ncols, i % ncols]
        sns.heatmap(df, ax=ax) #vmin=vmin, vmax=vmax, cbar=i == len(csv_files) - 1)
        ax.set_title(file.split('/')[2].replace('unif-', ''))  # Set subtitle as file name (without extension)

    for j in range(i + 1, nrows * ncols):
        axes[j // ncols, j % ncols].axis('off')
    plt.savefig(output_path)
    
if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('--folder_path', type=str, required=True)
    argp.add_argument('--output_path', type=str, required=True)
    args = argp.parse_args()
    # generate_heatmaps_unif(args.folder_path, args.output_path)
    generate_heatmaps_zipf(args.folder_path, args.output_path)
