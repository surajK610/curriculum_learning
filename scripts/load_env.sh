#!/bin/bash

# interact -n 20 -t 01:00:00 -m 10g -p 3090-gcondo
export LEARNING_DYNAMICS_HOME=/users/sanand14/data/sanand14/learning_dynamics

# module load python cuda
source $LEARNING_DYNAMICS_HOME/venv/bin/activate 
alias activate="source $LEARNING_DYNAMICS_HOME/venv/bin/activate"