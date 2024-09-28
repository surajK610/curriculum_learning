#!/bin/bash
#SBATCH --job-name=toy_model_zipf_ambr_stop_forg_sz
#SBATCH --output=outputs/toy_model/slurm_out/logz_%a.out
#SBATCH --error=outputs/toy_model/slurm_out/logz_%a.err
#SBATCH --array=0-1%30
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

# stop_forgettings=(10 20 30 40)
# stop_forgettings=(500 1000 1500 2000)
# a_s=(0 1.0001 1.2 1.5)
a_s=(2 3)

# stop_forgettings=(500)
# a_s=(1.5)
a_index=$(SLURM_ARRAY_TASK_ID)

# a_index=$((SLURM_ARRAY_TASK_ID % 4))
# forg_index=$((SLURM_ARRAY_TASK_ID % 3))

# layer_index=$((SLURM_ARRAY_TASK_ID / 12))
# vocab_index=$((SLURM_ARRAY_TASK_ID % 4))
# a_index=$((SLURM_ARRAY_TASK_ID / 4))

curr_a=${a_s[$a_index]}
curr_forg=5000 #${stop_forgettings[$forg_index]}
# curr_a=1.5

echo "SFRunning with layer: 6, vocab: 10000, a: $curr_a, amb: 0.10, stop_forget_steps: $curr_forg"
python3 $EXPERIMENT_SRC_DIR/toy_model.py --hidden_num_layers 6 --vocab_size 10000 --a $curr_a --prop_amb 0.10 --forget_steps 1000 --stop_forgetting_after $curr_forg --sample_func "zipfian" --hidden_size 64 --intermediate_size 128 --output_dir "outputs/toy_model/pca_embeddings/zipfw-sfs_$curr_forg-amb_0.10-vs_10000-a_$curr_a" --weight_decay 0.01 


#  --plot_pca --top_twenty 