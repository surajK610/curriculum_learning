#!/bin/bash
DATE=`date +%m-%d`

export LEARNING_DYNAMICS_HOME=/users/sanand14/data/sanand14/learning_dynamics
export EXPERIMENT_SRC_DIR=$LEARNING_DYNAMICS_HOME/src/experiments
export EXPERIMENT_CONFIG_DIR=$LEARNING_DYNAMICS_HOME/configs/ontonotes
export DATASET=ontonotes

for i in 0 20 40 60 80 100 200 1000 1400 1600 1800 2000; do
  dirhere = $EXPERIMENT_CONFIG_DIR/seed_0_step_$i
  mkdir -p $dirhere
  for type in ner phrase_start phrase_end; do
    if type == ner; then
    type_num_labels = 18
    else
    type_num_labels = 2
    fi
    for layer in {0..12}; do
      if layer == 0; then
        python3 src/experiments/utils/data_gen.py --task-name $type --dataset ewt --model-name google/multiberts-seed_0-step_{$i}k --layer-index $layer --compute-embeddings True
      else
        python3 src/experiments/utils/data_gen.py --task-name $type --dataset ewt --model-name google/multiberts-seed_0-step_{$i}k --layer-index $layer --compute-embeddings False
      fi
  echo "dataset:
  dir: "data/ontonotes/dataset/$type"
  task_name: "$type"
layer_idx: $layer
model_name: "google/multiberts-seed_0-step_{$i}k"
probe:
  finetune-model: "linear"
  epochs: 1
  batch_size: 32
  num_labels: ${type_num_labels} ## 37 indicates unknown
  input_size: 768
  output_dir: "outputs/ontonotes/$type"
  lr: "0.001"
" > $dirhere/$type_$layer.yaml
    python3 src/experiments/ontonotes.py --config $dirhere/$type_$layer.yaml
    done
  done
done