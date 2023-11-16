import streamlit as st
import pandas as pd
import json
import numpy as np
import matplotlib.image as mpimg

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
# Set page config
st.set_page_config(page_title="Streamlit App", page_icon="🧊", layout="wide")

def intro(offset=0):  

  st.write("# Training Step Analysis")

  st.sidebar.success("Select a visualization:")

  st.markdown(
    """
    The experiments in this app show layer-wise probing accuracy modeled as a function of the number of training steps. 
    
    We test sequences for informated related to 
    1. **F_POS** (i.e. fine grained part of speech information)
    2. **NER** (i.e. named entity recognition)
    3. **CHUNK** (i.e. courser grained pos information)
    
    All of our experiments are run with the mulitberts on the [CONLL2003](https://huggingface.co/datasets/conll2003/viewer/conll2003) dataset. 
    We train on the training split and evaluate on the validation split. Additionally, if a word is broken into multiple subwords, we only consider the last subword for evaluation.
    
    """)

    
def conll_heatmaps():
  st.write("## CONLL 2003 Heatmaps")
  
  option = st.selectbox(
    'Which task would you like to visualize?',
    ('F_POS', 'NER', 'CHUNK', 'ALL'))
  
  task_file = {'F_POS': 'outputs/pos/results/val_acc_heatmap.json', 
               'NER': 'outputs/ner/results/val_acc_heatmap.json',
               'CHUNK': 'outputs/chunk/results/val_acc_heatmap.json',
               'ALL': ['outputs/pos/results/val_acc_heatmap.json', 
                       'outputs/ner/results/val_acc_heatmap.json',
                       'outputs/chunk/results/val_acc_heatmap.json']}
  st.markdown(f"<h2 style='text-align: center; color: black;'>Layer-wise Validation Accuracy </h2>", unsafe_allow_html=True)
  fig = retrieve_plot(task_file[option])
  if isinstance(fig, list):
    for option, f in fig:
      st.markdown(f"<h2 style='text-align: center; color: red;'>{option} </h2>", unsafe_allow_html=True)
      st.plotly_chart(f, use_container_width=True)
  else:
    option, f = fig
    st.markdown(f"<h2 style='text-align: center; color: red;'>{option} </h2>", unsafe_allow_html=True)
    st.plotly_chart(f, use_container_width=True)
    
def structural_heatmaps():
  st.write("## Structural Probing Heatmaps")
  
  option = st.selectbox(
    'Which task would you like to visualize?',
    ('Parse Depth', 'Parse Distance', 'Both'))
  
  task_file = {'Parse Depth': ['outputs/structural/results/dep/val_spearmanr-5_50-mean_heatmap.json',
                               'outputs/structural/results/dep/val_root_acc_heatmap.json'], 
               'Parse Distance': ['outputs/structural/results/dist/val_spearmanr-5_50-mean_heatmap.json', 
                                  'outputs/structural/results/dist/val_uuas_heatmap.json'],
               'Both': ['outputs/structural/results/dep/val_spearmanr-5_50-mean_heatmap.json',
                       'outputs/structural/results/dep/val_root_acc_heatmap.json',
                       'outputs/structural/results/dist/val_spearmanr-5_50-mean_heatmap.json', 
                        'outputs/structural/results/dist/val_uuas_heatmap.json']
               }
  st.markdown(f"<h2 style='text-align: center; color: black;'>Layer-wise Validation Accuracy </h2>", unsafe_allow_html=True)
  fig = retrieve_plot(task_file[option], struct=True)
  if isinstance(fig, list):
    for option, f in fig:
      st.markdown(f"<h2 style='text-align: center; color: red;'>{option} </h2>", unsafe_allow_html=True)
      st.plotly_chart(f, use_container_width=True)
  else:
    option, f = fig
    st.markdown(f"<h2 style='text-align: center; color: red;'>{option} </h2>", unsafe_allow_html=True)
    st.plotly_chart(f, use_container_width=True)

# ------------------------- DATA READING AND PROCESSING -----------------------------

@st.cache_data
def retrieve_plot(file_path, struct=False):
  if isinstance(file_path, list):
    figs = []
    for fp in file_path:
      figs.append(retrieve_plot(fp, struct=struct))
    return figs
  
  with open(file_path, 'r') as f:
    figure_data = json.load(f)
    fig = go.Figure(figure_data)
    fig.update_yaxes(autorange='reversed')
    fig.update_yaxes(tickangle=-25) 
    fig.update_layout(
      height=600, 
      width=800
      )
  if struct:
    return (file_path.split('_')[1], fig)
  return (file_path.split('/')[1], fig)

    
page_names_to_funcs = {
  "Introduction": intro,
  "Conll2003": conll_heatmaps,
  "Structural Probing": structural_heatmaps,

}
st.cache_resource.clear()
demo_name = st.sidebar.selectbox("Select a visualization", list(page_names_to_funcs.keys()))
page_names_to_funcs[demo_name]()