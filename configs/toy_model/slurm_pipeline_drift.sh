#!/bin/bash
#SBATCH --job-name=toy_model_zipf_ambr_drift
#SBATCH --output=outputs/toy_model/slurm_out/logz_%a.out
#SBATCH --error=outputs/toy_model/slurm_out/logz_%a.err
#SBATCH --array=0-3%30
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

embed_drifts=(0.0)
a_s=(0 1.0001 1.2 1.5)

# stop_forgettings=(500)
# a_s=(1.5)

a_index=$((SLURM_ARRAY_TASK_ID % 4))
drift_index=$((SLURM_ARRAY_TASK_ID / 4))

# layer_index=$((SLURM_ARRAY_TASK_ID / 12))
# vocab_index=$((SLURM_ARRAY_TASK_ID % 4))
# a_index=$((SLURM_ARRAY_TASK_ID / 4))

curr_a=${a_s[$a_index]}
curr_drift=${embed_drifts[$drift_index]}

echo "Drift Running with layer: 6, vocab: 10000, a: $curr_a, amb: 0.10, drift: $curr_drift"p
python3 $EXPERIMENT_SRC_DIR/toy_model_drift.py --hidden_num_layers 6 --vocab_size 10000 --a $curr_a --prop_amb 0.10 --sample_func "zipfian" --hidden_size 64 --intermediate_size 128 --output_dir "outputs/toy_model/drift/zipfw-drift_$curr_drift-amb_0.10-vs_10000-a_$curr_a" --weight_decay 0.01 --extra_eval --mod_prob $curr_drift

# python3 $EXPERIMENT_SRC_DIR/toy_model_drift.py --hidden_num_layers 6 --vocab_size 10000 --a 1.5 --prop_amb 0.10 --sample_func "zipfian" --hidden_size 64 --intermediate_size 128 --output_dir "outputs/toy_model/test/zipfw-drift_0.05-amb_0.10-vs_10000-a_1.5" --weight_decay 0.01 --extra_eval --mod_prob 0.05