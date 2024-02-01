#!/bin/sh

# interact -n 20 -t 01:00:00 -m 10g -p 3090-gcondo -g 1
# interact -q 3090-gcondo -g 1 -m 20g -f ampere

export LEARNING_DYNAMICS_HOME=/users/sanand14/data/sanand14/learning_dynamics

# module load python cuda
source $LEARNING_DYNAMICS_HOME/venv/bin/activate

ipnip=$(hostname -i)
ipnport=8889

echo "Paste the following command onto your local computer:"
echo "ssh -N -L ${ipnport}:${ipnip}:${ipnport} ${USER}@sshcampus.ccv.brown.edu"
output = $(python -m notebook --no-browser --port=$ipnport --ip=$ipnip)

