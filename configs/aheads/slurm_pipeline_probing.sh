#!/bin/bash
#SBATCH --job-name=aheads
#SBATCH --output=outputs/aheads/slurm_out/log_%a.out
#SBATCH --error=outputs/aheads/slurm_out/log_%a.err
#SBATCH --array=0-35%36
#SBATCH --time=12:00:00
#SBATCH --mem=64G

#SBATCH -p gpu --gres=gpu:1
#SBATCH --cpus-per-task=1

DATE=$(date +%m-%d)

export LEARNING_DYNAMICS_HOME=/users/sanand14/data/sanand14/learning_dynamics
export EXPERIMENT_SRC_DIR=$LEARNING_DYNAMICS_HOME/src/experiments
export DATASET=aheads

# module load python cuda
source $LEARNING_DYNAMICS_HOME/venv/bin/activate

steps=(0 20 40 60 80 100 200 1000 1400 1600 1800 2000)
# steps=(120 140 160 180 300 400 500 600 700 800 900 1200)

# steps=(0 1 2 4 8 16 32 64 128 256 512 1000 2000 4000 8000 16000 32000 64000 128000 143000)
types=(induction_head previous_token_head duplicate_token_head)

step_index=$((SLURM_ARRAY_TASK_ID % 12))
type_index=$((SLURM_ARRAY_TASK_ID / 12))

step=${steps[$step_index]}
type=${types[$type_index]}

python3 -m src.experiments.aheads --probe-residuals True --checkpoint $step --detection-pattern $type

if [ $SLURM_ARRAY_TASK_ID -eq 35 ]; then
  python3 src/collate_metrics.py --exp duplicate_token_head --dataset  aheads --metric "Val Acc"
  python3 src/collate_metrics.py --exp induction_head --dataset  aheads --metric "Val Acc"
  python3 src/collate_metrics.py --exp previous_token_head --dataset  aheads --metric "Val Acc"
fi