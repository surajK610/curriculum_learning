#!/bin/bash
DATE=`date +%m-%d`

export LEARNING_DYNAMICS_HOME=/users/sanand14/data/sanand14/learning_dynamics
export EXPERIMENT_SRC_DIR=$LEARNING_DYNAMICS_HOME/src/experiments
export EXPERIMENT_CONFIG_DIR=$LEARNING_DYNAMICS_HOME/configs/ptb_3
export DATASET=ptb_3

for i in 0 20 40 60 80 100 200 1000 1400 1600 1800 2000; do
  dirhere = $EXPERIMENT_CONFIG_DIR/seed_0_step_$i
  mkdir -p $dirhere
  for type in depth distance; do
    for layer in {0..12}; do
      if layer == 0; then
        python3 src/experiments/utils/data_gen.py --task-name $type --dataset ptb --model-name google/multiberts-seed_0-step_{$i}k --layer-index $layer --compute-embeddings True
      else
        python3 src/experiments/utils/data_gen.py --task-name $type --dataset ptb --model-name google/multiberts-seed_0-step_{$i}k --layer-index $layer --compute-embeddings False
      fi
  echo "dataset:
  dir: "data/ptb_3/dataset/$type"
layer_idx: $layer
experiment: "$type"
model_name: "google/multiberts-seed_0-step_{$i}k"
probe:
  finetune-model: "linear"
  epochs: 10
  batch_size: 20
  rep_dim: 64
  input_size: 768
  output_dir: "outputs/ptb_3/$type"
  lr: "1e-2"
" > $dirhere/$type_$layer.yaml
    python3 src/experiments/en_ewt-ud.py --config $dirhere/$type_$layer.yaml
    done
  done
done