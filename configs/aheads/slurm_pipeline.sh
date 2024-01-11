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
export EXPERIMENT_CONFIG_DIR=$LEARNING_DYNAMICS_HOME/configs/ontonotes
export DATASET=ontonotes

# module load python cuda
source $LEARNING_DYNAMICS_HOME/venv/bin/activate 

python src/experiments/aheads.py --dataset-path outputs/aheads/dataset.pt --recompute False

if [ $(SLURM_ARRAY_TASK_ID) -eq 0 ]; then
  python3 src/collate_metrics.py --path-to-df aheads/induction_head/max_induction_head_deeper.csv --metric "max induction head deeper"
  python3 src/collate_metrics.py --path-to-df aheads/induction_head/mean_induction_head_deeper.csv --metric "mean induction head deeper"
  python3 src/collate_metrics.py --path-to-df aheads/induction_head/max_induction_head.csv --metric "max induction head"
  python3 src/collate_metrics.py --path-to-df aheads/induction_head/mean_induction_head.csv --metric "mean induction head"

  python3 src/collate_metrics.py --path-to-df aheads/previous_token_head/max_previous_token_head_deeper.csv --metric "max previous head deeper"
  python3 src/collate_metrics.py --path-to-df aheads/previous_token_head/mean_previous_token_head_deeper.csv --metric "mean previous head deeper"
  python3 src/collate_metrics.py --path-to-df aheads/previous_token_head/max_previous_token_head.csv --metric "max previous head"
  python3 src/collate_metrics.py --path-to-df aheads/previous_token_head/mean_previous_token_head.csv --metric "mean previous head"

  python3 src/collate_metrics.py --path-to-df aheads/duplicate_token_head/max_duplicate_token_head_deeper.csv --metric "max duplicate head deeper"
  python3 src/collate_metrics.py --path-to-df aheads/duplicate_token_head/mean_duplicate_token_head_deeper.csv --metric "mean duplicate head deeper"
  python3 src/collate_metrics.py --path-to-df aheads/duplicate_token_head/max_duplicate_token_head.csv --metric "max duplicate head"
  python3 src/collate_metrics.py --path-to-df aheads/duplicate_token_head/mean_duplicate_token_head.csv --metric "mean duplicate head"
fi