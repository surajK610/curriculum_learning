# interact -n 20 -t 01:00:00 -m 10g -p 3090-gcondo

export LEARNING_DYNAMICS_HOME=/users/sanand14/data/sanand14/learning_dynamics

# module load python cuda
source $LEARNING_DYNAMICS_HOME/venv/bin/activate

streamlist_host=$(hostname -i)
streamlist_port=8501

echo "ssh -N -L ${streamlist_port}:${streamlist_host}:${streamlist_port} ${USER}@sshcampus.ccv.brown.edu"
echo "Paste the following URL into your browser: http://localhost:${streamlist_port}"
output = $(python3 -m streamlit run $LEARNING_DYNAMICS_HOME/src/streamlit_app.py --server.port $streamlist_port --server.fileWatcherType none)