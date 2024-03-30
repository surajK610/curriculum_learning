#!/bin/bash
#SBATCH --job-name=toy_model_zipf_forg
#SBATCH --output=outputs/toy_model/slurm_out/log_%a.out
#SBATCH --error=outputs/toy_model/slurm_out/log_%a.err
#SBATCH --array=0-3%30
#SBATCH --time=24:00:00
#SBATCH --mem=64G

#SBATCH -p 3090-gcondo --gres=gpu:1 --constraint=a5000
#SBATCH --cpus-per-task=1

# module load python cuda
export LEARNING_DYNAMICS_HOME=/users/sanand14/data/sanand14/learning_dynamics
export EXPERIMENT_SRC_DIR=$LEARNING_DYNAMICS_HOME/src/experiments
source $LEARNING_DYNAMICS_HOME/venv/bin/activate

# num_layers=(1 2 6)
# vocab_sizes=(100 1000 10000)
# a_s=(1.0001 1.2 1.5)

forgetting=(-1 100 1000 5000)

f_index=$SLURM_ARRAY_TASK_ID

# layer_index=$((SLURM_ARRAY_TASK_ID / 12))
# vocab_index=$((SLURM_ARRAY_TASK_ID % 4))
# a_index=$((SLURM_ARRAY_TASK_ID / 4))

curr_for=${forgetting[$f_index]}

echo "Running with layer: 6, vocab: 1000, a: 1.0001, amb: 0.10, forgetting num steps: $curr_for"
python3 $EXPERIMENT_SRC_DIR/toy_model.py --forget_steps $curr_for --hidden_num_layers 6 --vocab_size 1000 --a 1.0001 --prop_amb 0.10 --sample_func "zipfian" --hidden_size 64 --intermediate_size 128 --output_dir "outputs/toy_model/zipfr-f_$curr_for-amb_0.10-vs_1000-a_1.0001" --weight_decay 0.01

