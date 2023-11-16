#!/bin/bash
#SBATCH --job-name=ontonotes
#SBATCH --output=outputs/ontonotes/slurm_out/log_%a.out
#SBATCH --error=outputs/ontonotes/slurm_out/log_%a.err
#SBATCH --array=0-11%12
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:1
#SBATCH --cpus-per-task=1

DATE=`date +%m-%d`

export LEARNING_DYNAMICS_HOME=/users/sanand14/data/sanand14/learning_dynamics
export EXPERIMENT_SRC_DIR=$LEARNING_DYNAMICS_HOME/src/experiments
export EXPERIMENT_CONFIG_DIR=$LEARNING_DYNAMICS_HOME/configs/ontonotes
export DATASET=ontonotes

steps=(0 20 40 60 80 100 200 1000 1400 1600 1800 2000)
types=(ner)

step_index=$((SLURM_ARRAY_TASK_ID % 12))
type_index=$((SLURM_ARRAY_TASK_ID / 12))

step=${steps[$step_index]}
type=${types[$type_index]}

dirhere=$EXPERIMENT_CONFIG_DIR/seed_0_step_$step
  mkdir -p $dirhere
  echo "model:
  name: "google/multiberts-seed_0-step_${step}k"
  num_hidden_layers: 12
experiment: "ner"
probe:
  finetune_model: "linear"
  epochs: 1
  batch_size: 32
  output_dir: "outputs/ontonotes/ner"
  lr: "0.001"
" > $dirhere/$type.yaml
  python3 src/experiments/ontonotes.py --config $dirhere/$type.yaml
done
