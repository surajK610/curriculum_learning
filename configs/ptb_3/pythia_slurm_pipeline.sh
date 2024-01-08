#!/bin/bash
#SBATCH --job-name=ptb_experiment_pythia
#SBATCH --output=outputs/ptb_3/slurm_out/log_%a.out
#SBATCH --error=outputs/ptb_3/slurm_out/log_%a.err
#SBATCH --array=0-39%40
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH --cpus-per-task=1

DATE=$(date +%m-%d)

export LEARNING_DYNAMICS_HOME=/users/sanand14/data/sanand14/learning_dynamics
export EXPERIMENT_SRC_DIR=$LEARNING_DYNAMICS_HOME/src/experiments
export EXPERIMENT_CONFIG_DIR=$LEARNING_DYNAMICS_HOME/configs/ptb_3
export DATASET=ptb_3

module load python/3.9.0 cuda/11.1.1 gcc/10.2
source $LEARNING_DYNAMICS_HOME/venv-3090/bin/activate

steps=(0 1 2 4 8 16 32 64 128 256 512 1000 2000 4000 8000 16000 32000 64000 128000 143000)
types=(depth distance)

step_index=$((SLURM_ARRAY_TASK_ID % 20))
type_index=$((SLURM_ARRAY_TASK_ID / 20))

step=${steps[$step_index]}
type=${types[$type_index]}

for layer in {0..12}; do
    dirhere=$EXPERIMENT_CONFIG_DIR/pythia_160m_step_${step}
    mkdir -p $dirhere
    if [[ "$layer" -eq 0 && "$type" == "depth" ]]; then
        python3 $EXPERIMENT_SRC_DIR/utils/data_gen.py --task-name $type --dataset ptb --model-name EleutherAI/pythia-160m --model-step ${step} --layer-index $layer --compute-embeddings True
    else
        python3 $EXPERIMENT_SRC_DIR/utils/data_gen.py --task-name $type --dataset ptb --model-name EleutherAI/pythia-160m --model-step ${step} --layer-index $layer --compute-embeddings False
    fi
    cat << EOF > $dirhere/${type}_${layer}.yaml
dataset:
  dir: "data/ptb_3/dataset/${type}"
layer_idx: $layer
experiment: "${type}"
model_name: "EleutherAI/pythia-160m"
model_step: "${step}"
model_type: "pythia"
probe:
  finetune-model: "linear"
  epochs: 10
  batch_size: 20
  rep_dim: 64
  input_size: 768
  output_dir: "outputs/ptb_3/${type}"
  lr: "1e-2"
EOF
    python3 $EXPERIMENT_SRC_DIR/ptb_3.py --config $dirhere/${type}_${layer}.yaml
done

if [ $(SLURM_ARRAY_TASK_ID) -eq 39 ]; then
  python3 src/collate_metrics.py --exp depth --dataset ptb_3 --metric "Root Acc"
  python3 src/collate_metrics.py --exp depth --dataset ptb_3 --metric "NSpr"
  python3 src/collate_metrics.py --exp distance --dataset ptb_3 --metric "UUAS"
  python3 src/collate_metrics.py --exp distance --dataset ptb_3 --metric "DSpr"
fi