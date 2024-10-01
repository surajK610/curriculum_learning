#!/bin/bash
#SBATCH --job-name=toy_model_ar_sf3
#SBATCH --output=outputs/toy_model_ar/slurm_out/log_sf1_%a.out
#SBATCH --error=outputs/toy_model_ar/slurm_out/log_sf1_%a.err
#SBATCH --array=0-45%45
#SBATCH --time=24:00:00
#SBATCH --mem=64G

#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH --cpus-per-task=1

# module load python cuda
export LEARNING_DYNAMICS_HOME=/users/sanand14/data/sanand14/learning_dynamics
export EXPERIMENT_SRC_DIR=$LEARNING_DYNAMICS_HOME/src/experiments
source $LEARNING_DYNAMICS_HOME/venv/bin/activate


k_s=(500 1000 2000)
num_examples=(150000 200000 400000)
a_s=(0 1.0001 1.5 2 3) # 45 

k_index=$(( SLURM_ARRAY_TASK_ID / 15 ))
num_example_index=$(( (SLURM_ARRAY_TASK_ID % 15 ) % 3 ))
a_index=$(( (SLURM_ARRAY_TASK_ID % 15) / 3 ))

curr_k=${k_s[$k_index]}
curr_num_examples=${num_examples[$num_example_index]}
curr_a=${a_s[$a_index]}


# a_index=$((SLURM_ARRAY_TASK_ID % 16 / 4))
# vocab_index=$((SLURM_ARRAY_TASK_ID % 4))
# prop_amb_ind=$((SLURM_ARRAY_TASK_ID / 16))

echo "Running with k: $curr_k, a: $curr_a, num examples: $curr_num_examples"
python3 $EXPERIMENT_SRC_DIR/toy_model_ar.py --num_epochs 10 --amb_ratio 0.05 --forget_steps $curr_k --stop_forgetting_after 10000 --num_train_examples $curr_num_examples --a $curr_a  --output_dir "outputs/toy_model_ar/sforg2_k_$curr_k-numtr_$curr_num_examples-a_$curr_a"