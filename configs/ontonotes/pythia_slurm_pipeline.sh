#!/bin/bash
#SBATCH --job-name=ontonotes_pythia
#SBATCH --output=outputs/ontonotes/slurm_out/log_%a.out
#SBATCH --error=outputs/ontonotes/slurm_out/log_%a.err
#SBATCH --array=0-59%60
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH --cpus-per-task=1

DATE=$(date +%m-%d)

export LEARNING_DYNAMICS_HOME=/users/sanand14/data/sanand14/learning_dynamics
export EXPERIMENT_SRC_DIR=$LEARNING_DYNAMICS_HOME/src/experiments
export EXPERIMENT_CONFIG_DIR=$LEARNING_DYNAMICS_HOME/configs/ontonotes
export DATASET=ontonotes

module load python/3.9.0 cuda/11.1.1 gcc/10.2
source $LEARNING_DYNAMICS_HOME/venv-3090/bin/activate 

steps=(0 1 2 4 8 16 32 64 128 256 512 1000 2000 4000 8000 16000 32000 64000 128000 143000)
types=(ner phrase_start phrase_end)
num_labels=(19 2 2)

step_index=$((SLURM_ARRAY_TASK_ID % 20))
type_index=$((SLURM_ARRAY_TASK_ID / 20))

step=${steps[$step_index]}
type=${types[$type_index]}
num_labels_type=${num_labels[$type_index]}

echo "Running Pythia Experiment with step: $step and type: $type"

for layer in {0..12}; do
    dirhere=$EXPERIMENT_CONFIG_DIR/pythia_160m_step_${step}
    mkdir -p $dirhere
    if [[ "$layer" -eq 0 && "$type" == "ner" ]]; then
        python3 $EXPERIMENT_SRC_DIR/utils/data_gen.py --task-name $type --dataset ontonotes --model-name EleutherAI/pythia-160m --model-step ${step} --layer-index $layer --compute-embeddings True
    else
        python3 $EXPERIMENT_SRC_DIR/utils/data_gen.py --task-name $type --dataset ontonotes --model-name EleutherAI/pythia-160m --model-step ${step} --layer-index $layer --compute-embeddings False
    fi
    cat << EOF > $dirhere/${type}_${layer}.yaml
dataset:
  dir: "data/ontonotes/dataset/${type}"
  task_name: "${type}"
layer_idx: $layer
model_name: "EleutherAI/pythia-160m"
model_step: "${step}"
model_type: "pythia"
probe:
  finetune-model: "linear"
  epochs: 4
  batch_size: 32
  num_labels: $num_labels_type
  input_size: 768
  output_dir: "outputs/ontonotes/${type}"
  lr: "0.001"
EOF
    python3 $EXPERIMENT_SRC_DIR/ontonotes.py --config $dirhere/${type}_${layer}.yaml
done

if [ $(SLURM_ARRAY_TASK_ID) -eq 59 ]; then
  python3 src/collate_metrics.py --exp ner --dataset ontonotes --metric "Val Acc"
  python3 src/collate_metrics.py --exp phrase_start --dataset ontonotes --metric "Val Acc"
  python3 src/collate_metrics.py --exp phrase_end --dataset ontonotes --metric "Val Acc"
fi