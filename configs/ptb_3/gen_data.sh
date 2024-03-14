#!/bin/bash
#SBATCH --job-name=ptb_gen_data
#SBATCH --output=outputs/ptb_3/seed_1/slurm_out/log_%a.out
#SBATCH --error=outputs/ptb_3/seed_1/slurm_out/log_%a.err
#SBATCH --array=0-23%24
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH --cpus-per-task=1

DATE=$(date +%m-%d)
RESID=True
SEED=seed_1
export LEARNING_DYNAMICS_HOME=/users/sanand14/data/sanand14/learning_dynamics
export EXPERIMENT_SRC_DIR=$LEARNING_DYNAMICS_HOME/src/experiments
export EXPERIMENT_CONFIG_DIR=$LEARNING_DYNAMICS_HOME/configs/ptb_3
export DATASET=ptb_3

# module load python cuda
source $LEARNING_DYNAMICS_HOME/venv/bin/activate

steps=(0 20 40 60 80 100 120 140 160 180 200 300 400 500 600 700 800 900 1000 1200 1400 1600 1800 2000)

step_index=$((SLURM_ARRAY_TASK_ID % 24))

step=${steps[$step_index]}

echo "Running Experiment with step: $step and seed $SEED with residual $RESID"

dirhere=$EXPERIMENT_CONFIG_DIR/${SEED}_step_${step}
mkdir -p $dirhere
python3 $EXPERIMENT_SRC_DIR/utils/data_gen.py --task-name depth --dataset ptb --model-name google/multiberts-${SEED}-step_${step}k --layer-index 0 --compute-embeddings True  --resid $RESID
