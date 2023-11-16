DATE=`date +%m-%d`

export LEARNING_DYNAMICS_HOME=/users/sanand14/data/sanand14/learning_dynamics
export EXPERIMENT_SRC_DIR=$LEARNING_DYNAMICS_HOME/src/experiments
export EXPERIMENT_CONFIG_DIR=$LEARNING_DYNAMICS_HOME/configs/ontonotes
export DATASET=ontonotes

for i in 20 40 60 80 100 200 1000 1400 1600 1800 2000; do
  dirhere=$EXPERIMENT_CONFIG_DIR/seed_0_step_$i
  mkdir -p $dirhere
  echo "model:
  name: "google/multiberts-seed_0-step_${i}k"
  num_hidden_layers: 12
experiment: "ner"
probe:
  finetune_model: "linear"
  epochs: 1
  batch_size: 32
  output_dir: "outputs/ontonotes/ner"
  lr: "0.001"
" > $dirhere/$type_$layer.yaml
  python3 src/experiments/ontonotes.py --config configs/ontonotes/base/config_base_ner.yaml 
done
