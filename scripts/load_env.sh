#!/bin/bash

# interact -n 20 -t 01:00:00 -m 10g -p 3090-gcondo
export LEARNING_DYNAMICS_HOME=/users/sanand14/data/sanand14/learning_dynamics

module load python/3.9.0 cuda/11.1.1 gcc/10.2
source $LEARNING_DYNAMICS_HOME/venv/bin/activate 
alias activate="source $LEARNING_DYNAMICS_HOME/venv/bin/activate"