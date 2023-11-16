#!/bin/sh

# Request half an hour of runtime:
#SBATCH --time=24:00:00

# Ask for the GPU partition and 1 GPU
# skipping this for now.
# # SBATCH -p 3090-gcondo --gres=gpu:1

# Use more memory (8GB) and correct partition.
#SBATCH --mem=32G

# Specify a job name:
#SBATCH -J download_multiberts

# Specify an output file
#SBATCH -o ./outputs/download_berts/%x-%a.out
#SBATCH -e ./outputs/download_berts/%x-%a.out


wget "https://storage.googleapis.com/multiberts/public/intermediates/seed_0.zip"
unzip "seed_0.zip"
rm "seed_0.zip"