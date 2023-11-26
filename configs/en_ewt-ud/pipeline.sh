#!/bin/bash
DATE=`date +%m-%d`

export LEARNING_DYNAMICS_HOME=/users/sanand14/data/sanand14/learning_dynamics
export EXPERIMENT_SRC_DIR=$LEARNING_DYNAMICS_HOME/src/experiments
export EXPERIMENT_CONFIG_DIR=$LEARNING_DYNAMICS_HOME/configs/ud_en_ewt
export DATASET=en_ewt-ud

for i in 0 20 40 60 80 100 200 1000 1400 1600 1800 2000; do
  dirhere = $EXPERIMENT_CONFIG_DIR/seed_0_step_$i
  mkdir -p $dirhere
  for type in fpos cpos dep; do
    if type == fpos; then
    type_num_labels = 38
    elif type == cpos; then
    type_num_labels = 17
    else
    type_num_labels = 54
    fi
    for layer in {0..12}; do
      if layer == 0; then
        python3 src/experiments/utils/data_gen.py --task-name $type --dataset ewt --model-name google/multiberts-seed_0-step_{$i}k --layer-index $layer --compute-embeddings True
      else
        python3 src/experiments/utils/data_gen.py --task-name $type --dataset ewt --model-name google/multiberts-seed_0-step_{$i}k --layer-index $layer --compute-embeddings False
      fi
  echo "dataset:
  dir: "data/en_ewt-ud/dataset/$type"
  task_name: "$type"
layer_idx: $layer
model_name: "google/multiberts-seed_0-step_{$i}k"
probe:
  finetune-model: "linear"
  epochs: 1
  batch_size: 32
  num_labels: 38 ## 37 indicates unknown
  input_size: 768
  output_dir: "outputs/en_ewt-ud/$type"
  lr: "0.001"
" > $dirhere/$type_$layer.yaml
    python3 src/experiments/en_ewt-ud.py --config $dirhere/$type_$layer.yaml
    done
  done
done