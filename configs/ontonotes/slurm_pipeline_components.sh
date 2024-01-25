#!/bin/bash
#SBATCH --job-name=ontonotes
#SBATCH --output=outputs/ontonotes/slurm_out/log_%a.out
#SBATCH --error=outputs/ontonotes/slurm_out/log_%a.err
#SBATCH --array=0-431%432
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH --cpus-per-task=1

DATE=$(date +%m-%d)
RESID=False

export LEARNING_DYNAMICS_HOME=/users/sanand14/data/sanand14/learning_dynamics
export EXPERIMENT_SRC_DIR=$LEARNING_DYNAMICS_HOME/src/experiments
export EXPERIMENT_CONFIG_DIR=$LEARNING_DYNAMICS_HOME/configs/ontonotes
export DATASET=ontonotes

# module load python cuda
source $LEARNING_DYNAMICS_HOME/venv/bin/activate 

steps=(0 20 40 60 80 100 200 1000 1400 1600 1800 2000)
# steps=(120 140 160 180 300 400 500 600 700 800 900 1200)
types=(ner phrase_start phrase_end)
num_labels=(19 2 2)
layer=(1 2 3 4 5 6 7 8 9 10 11 12)

step_index=$((SLURM_ARRAY_TASK_ID % 144 / 12))
type_index=$((SLURM_ARRAY_TASK_ID / 144))
layer_index=$((SLURM_ARRAY_TASK_ID % 12))

step=${steps[$step_index]}
type=${types[$type_index]}
num_labels_type=${num_labels[$type_index]}
layer=${layer[$layer_index]}

echo "Running Experiment with step: $step and type: $type and layer $layer"
# dirhere=$EXPERIMENT_CONFIG_DIR/seed_0_step_${step}
# mkdir -p $dirhere
# python3 $EXPERIMENT_SRC_DIR/utils/data_gen.py --task-name $type --dataset ontonotes --model-name google/multiberts-seed_0-step_${step}k --layer-index 0 --compute-embeddings True  --resid $RESID
    
for head in ""{0..11}; do
  if [ -z "$head" ]; then
    dirhere=$EXPERIMENT_CONFIG_DIR/seed_0_step_${step}
    mkdir -p $dirhere
    python3 $EXPERIMENT_SRC_DIR/utils/data_gen.py --task-name $type --dataset ontonotes --model-name google/multiberts-seed_0-step_${step}k --layer-index $layer --compute-embeddings False --resid $RESID
    cat << EOF > $dirhere/${type}_${layer}.yaml
dataset:
  dir: "data/ontonotes/dataset/${type}"
  task_name: "${type}"
layer_idx: $layer
model_name: "google/multiberts-seed_0-step_${step}k"
model_step: "${step}k"
model_type: "multibert"
resid: $RESID
attention_head: null
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
  rm $LEARNING_DYNAMICS_HOME/data/ontonotes/dataset/${type}/multiberts-seed_0-step_${step}k/*-layer-${layer}.pt
  else
    dirhere=$EXPERIMENT_CONFIG_DIR/seed_0_step_${step}
    mkdir -p $dirhere
    python3 $EXPERIMENT_SRC_DIR/utils/data_gen.py --task-name $type --dataset ontonotes --model-name google/multiberts-seed_0-step_${step}k --layer-index $layer --attention-head $head --compute-embeddings False --resid $RESID
    cat << EOF > $dirhere/${type}_${layer}_${head}.yaml
dataset:
  dir: "data/ontonotes/dataset/${type}"
  task_name: "${type}"
layer_idx: $layer
model_name: "google/multiberts-seed_0-step_${step}k"
model_step: "${step}k"
model_type: "multibert"
resid: $RESID
attention_head: $head
probe:
  finetune-model: "linear"
  epochs: 4
  batch_size: 32
  num_labels: $num_labels_type
  input_size: 768
  output_dir: "outputs/ontonotes/${type}"
  lr: "0.001"
EOF
  python3 $EXPERIMENT_SRC_DIR/ontonotes.py --config $dirhere/${type}_${layer}_${head}.yaml
  rm $LEARNING_DYNAMICS_HOME/data/ontonotes/dataset/${type}/multiberts-seed_0-step_${step}k/*-layer-${layer}-${head}.pt
  fi
done


# for layer in {0..12}; do
#     dirhere=$EXPERIMENT_CONFIG_DIR/seed_0_step_${step}
#     mkdir -p $dirhere
#     if [[ "$layer" -eq 0 && "$type" == "ner" ]]; then
#         python3 $EXPERIMENT_SRC_DIR/utils/data_gen.py --task-name $type --dataset ontonotes --model-name google/multiberts-seed_0-step_${step}k --layer-index $layer --compute-embeddings True  --resid $RESID
#     else
#         python3 $EXPERIMENT_SRC_DIR/utils/data_gen.py --task-name $type --dataset ontonotes --model-name google/multiberts-seed_0-step_${step}k --layer-index $layer --compute-embeddings False  --resid $RESID
#     fi
#     cat << EOF > $dirhere/${type}_${layer}.yaml
# dataset:
#   dir: "data/ontonotes/dataset/${type}"
#   task_name: "${type}"
# layer_idx: $layer
# model_name: "google/multiberts-seed_0-step_${step}k"
# model_step: "${step}k"
# model_type: "multibert"
# resid: $RESID
# probe:
#   finetune-model: "linear"
#   epochs: 4
#   batch_size: 32
#   num_labels: $num_labels_type
#   input_size: 768
#   output_dir: "outputs/ontonotes/${type}"
#   lr: "0.001"
# EOF
#     python3 $EXPERIMENT_SRC_DIR/ontonotes.py --config $dirhere/${type}_${layer}.yaml
# done

# if [ $(SLURM_ARRAY_TASK_ID) -eq 35 ]; then
#   python3 src/collate_metrics.py --exp ner --dataset ontonotes --metric "Val Acc"
#   python3 src/collate_metrics.py --exp phrase_start --dataset ontonotes --metric "Val Acc"
#   python3 src/collate_metrics.py --exp phrase_end --dataset ontonotes --metric "Val Acc"
# fi