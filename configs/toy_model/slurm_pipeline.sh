#!/bin/bash
#SBATCH --job-name=toy_model_zipf
#SBATCH --output=outputs/toy_model/slurm_out/logz_%a.out
#SBATCH --error=outputs/toy_model/slurm_out/logz_%a.err
#SBATCH --array=0-5%6
#SBATCH --time=12:00:00
#SBATCH --mem=64G

#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH --cpus-per-task=1

# module load python cuda
export LEARNING_DYNAMICS_HOME=/users/sanand14/data/sanand14/learning_dynamics
export EXPERIMENT_SRC_DIR=$LEARNING_DYNAMICS_HOME/src/experiments
source $LEARNING_DYNAMICS_HOME/venv/bin/activate

# num_layers=(1 2 6)
# vocab_sizes=(100 1000 10000)
# a_s=(1.0001 1.2 1.5)

num_layers=(6)
vocab_sizes=(20000 10000)
a_s=(1.0001 1.2 1.5)

# layer_index=$((SLURM_ARRAY_TASK_ID % 9 / 3))
# vocab_index=$((SLURM_ARRAY_TASK_ID % 3))
# a_index=$((SLURM_ARRAY_TASK_ID / 9))

layer_index=$((SLURM_ARRAY_TASK_ID / 9))
vocab_index=$((SLURM_ARRAY_TASK_ID % 3))
a_index=$((SLURM_ARRAY_TASK_ID / 3))

curr_layer=${num_layers[$layer_index]}
curr_vocab=${vocab_sizes[$vocab_index]}
curr_a=${a_s[$a_index]}

echo "Running with layer: $curr_layer, vocab: $curr_vocab, a: $curr_a"
python3 $EXPERIMENT_SRC_DIR/toy_model.py --hidden_num_layers $curr_layer --vocab_size $curr_vocab --a $curr_a --sample_func "zipfian" --output_dir "outputs/toy_model/zipf-layer_$curr_layer-vs_$curr_vocab-a_$curr_a"





