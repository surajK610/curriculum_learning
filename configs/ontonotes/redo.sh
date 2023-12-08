#!/bin/bash
#SBATCH --job-name=ontonotes
#SBATCH --output=outputs/ontonotes/slurm_out/log_%a.out
#SBATCH --error=outputs/ontonotes/slurm_out/log_%a.err
#SBATCH --array=1-3%4
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:1
#SBATCH --cpus-per-task=1

DATE=$(date +%m-%d)

export LEARNING_DYNAMICS_HOME=/users/sanand14/data/sanand14/learning_dynamics
export EXPERIMENT_SRC_DIR=$LEARNING_DYNAMICS_HOME/src/experiments
export EXPERIMENT_CONFIG_DIR=$LEARNING_DYNAMICS_HOME/configs/ontonotes
export DATASET=ontonotes

module load python/3.9.0 cuda/11.1.1 gcc/10.2
source $LEARNING_DYNAMICS_HOME/venv/bin/activate 


if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then 
  type=ner
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then 
  type=phrase_end
elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then 
  type=phrase_start
fi 

echo "Running Experiment with step: 1000 and type: $type"

type=phrase_start
for layer in {0..12}; do
    dirhere=$EXPERIMENT_CONFIG_DIR/seed_0_step_1000
    # srun -p gpu --gres=gpu:1 --mem=32G --time=2:00:00 python3 $EXPERIMENT_SRC_DIR/utils/data_gen.py --task-name phrase_start --dataset ontonotes --model-name google/multiberts-seed_0-step_1000k --layer-index $layer --compute-embeddings False
    srun -p gpu --gres=gpu:1 --mem=32G --time=2:00:00 python3 $EXPERIMENT_SRC_DIR/ontonotes.py --config $dirhere/${type}_${layer}.yaml
done