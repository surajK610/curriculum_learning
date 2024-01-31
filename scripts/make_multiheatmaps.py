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

if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('--folder_path', type=str, required=True)
    argp.add_argument('--output_path', type=str, required=True)
    args = argp.parse_args()
    generate_heatmaps(args.folder_path, args.output_path)
