#!/bin/bash
#SBATCH --job-name=icl_bert
#SBATCH --output=outputs/icl_bert/seed_3/slurm_out/log_%a.out
#SBATCH --error=outputs/icl_bert/seed_3/slurm_out/log_%a.err
#SBATCH --array=0-23%24
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH --cpus-per-task=1

DATE=$(date +%m-%d)
SEED=seed_3

export LEARNING_DYNAMICS_HOME=/users/sanand14/data/sanand14/learning_dynamics
export EXPERIMENT_SRC_DIR=$LEARNING_DYNAMICS_HOME/src/experiments

source $LEARNING_DYNAMICS_HOME/venv/bin/activate 
steps=(0 20 40 60 80 100 120 140 160 180 200 300 400 500 600 700 800 900 1000 1200 1400 1600 1800 2000)
step_index=$((SLURM_ARRAY_TASK_ID % 24))
step=${steps[$step_index]}

echo "Running ICL Experiment with step: $step"

python3 $EXPERIMENT_SRC_DIR/icl_bert.py --epochs 4 --batch_size 32 --num_examples_test 10000 --model_name "google/multiberts-${SEED}-step_${step}k" --output_dir "outputs/icl_bert/${SEED}/"
