#!/bin/bash
#SBATCH --job-name=toy_model_zipf_ambr
#SBATCH --output=outputs/toy_model/slurm_out/logz_%a.out
#SBATCH --error=outputs/toy_model/slurm_out/logz_%a.err
#SBATCH --array=0-47%48
#SBATCH --time=24:00:00
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

prop_ambs=(0.0 0.01 0.10 0.50)
vocab_sizes=(100 1000 10000 20000)
a_s=(1.0001 1.2 1.5)

prop_amb_ind=$((SLURM_ARRAY_TASK_ID % 16 / 4))
vocab_index=$((SLURM_ARRAY_TASK_ID % 4))
a_index=$((SLURM_ARRAY_TASK_ID / 16))

# layer_index=$((SLURM_ARRAY_TASK_ID / 12))
# vocab_index=$((SLURM_ARRAY_TASK_ID % 4))
# a_index=$((SLURM_ARRAY_TASK_ID / 4))

curr_amb=${prop_ambs[$prop_amb_ind]}
curr_vocab=${vocab_sizes[$vocab_index]}
curr_a=${a_s[$a_index]}

echo "Running with layer: 6, vocab: $curr_vocab, a: $curr_a, amb: $curr_amb"
python3 $EXPERIMENT_SRC_DIR/toy_model.py --hidden_num_layers 6 --vocab_size $curr_vocab --a $curr_a --prop_amb $curr_amb --sample_func "zipfian" --hidden_size 64 --intermediate_size 128 --output_dir "outputs/toy_model/zipfr-amb_$curr_amb-vs_$curr_vocab-a_$curr_a"

