DATE=`date +%m-%d`

export LEARNING_DYNAMICS_HOME=/users/sanand14/data/sanand14/learning_dynamics
export EXPERIMENT_SRC_DIR=$LEARNING_DYNAMICS_HOME/src/experiments

python -m src.job_generation --experiment experiments/probe.json --date $DATE
sbatch ./job/probe-$DATE.sh