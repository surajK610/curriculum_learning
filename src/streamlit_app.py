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
    1. **Ontonotes-5** (i.e. NER, Phrase Start, Phrase End)
    2. **Universal Dependencies** (i.e. CPOS, FPOS, Dep)
    3. **Penn Treebank 3** (i.e. structural probing)
    4. **Attention Heads** 
    
    All of our dataet experiments are run with the mulitberts models while the attention head experiments are run with the pythia models.
    We train on the training split and evaluate on the validation split. Additionally, if a word is broken into multiple subwords, we only consider the last subword for evaluation.
    
    """)

def line_graph():
  st.write("## Line Graph")
  st.markdown("Among Syntax and Semantic Tasks, this shows trends where the model is learning the fastest.")
  file_path = 'outputs/line-graph.json'
  with open(file_path, 'r') as f:
    figure_data = json.load(f)
    fig = go.Figure(figure_data)
    fig.update_yaxes(autorange='reversed')
    fig.update_yaxes(tickangle=-25) 
    fig.update_layout(
      height=600, 
      width=800
      )
  st.plotly_chart(fig, use_container_width=True)
  
def ud_heatmaps():
  st.write("## Universal Dependencies 2003 Heatmaps")
  
  option = st.selectbox(
    'Which task would you like to visualize?',
    ('CPOS', 'DEP', 'FPOS', 'ALL'))
  
  task_file = {'CPOS': 'outputs/en_ewt-ud/cpos/Val_Acc_heatmap.json', 
               'DEP': 'outputs/en_ewt-ud/dep/Val_Acc_heatmap.json',
               'FPOS': 'outputs/en_ewt-ud/fpos/Val_Acc_heatmap.json',
               'ALL': ['outputs/en_ewt-ud/cpos/Val_Acc_heatmap.json', 
                       'outputs/en_ewt-ud/dep/Val_Acc_heatmap.json',
                       'outputs/en_ewt-ud/fpos/Val_Acc_heatmap.json']}
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
    
def ptb_heatmaps():
  st.write("## Penn Treebank 3 Probing Heatmaps")
  
  option = st.selectbox(
    'Which structural probing task would you like to visualize?',
    ('Parse Depth', 'Parse Distance', 'Both'))
  
  task_file = {'Parse Depth': ['outputs/ptb_3/depth/NSpr_heatmap.json',
                               'outputs/ptb_3/depth/Root_Acc_heatmap.json'], 
               'Parse Distance': ['outputs/ptb_3/distance/DSpr_heatmap.json', 
                                  'outputs/ptb_3/distance/UUAS_heatmap.json'],
               'Both': ['outputs/ptb_3/depth/NSpr_heatmap.json', 
                        'outputs/ptb_3/depth/Root_Acc_heatmap.json', 
                        'outputs/ptb_3/distance/DSpr_heatmap.json',
                        'outputs/ptb_3/distance/UUAS_heatmap.json']
               }
  st.markdown(f"<h2 style='text-align: center; color: black;'>Layer-wise Validation Metrics</h2>", unsafe_allow_html=True)
  fig = retrieve_plot(task_file[option], name_idx=2)
  if isinstance(fig, list):
    for option, f in fig:
      st.markdown(f"<h2 style='text-align: center; color: red;'>{option} </h2>", unsafe_allow_html=True)
      st.plotly_chart(f, use_container_width=True)
  else:
    option, f = fig
    st.markdown(f"<h2 style='text-align: center; color: red;'>{option} </h2>", unsafe_allow_html=True)
    st.plotly_chart(f, use_container_width=True)
  
def ontonotes_heatmaps():
  st.write("## Ontonotes-5 Probing Heatmaps")
  
  option = st.selectbox(
    'Which structural probing task would you like to visualize?',
    ('NER', 'Phrase Start', 'Phrase End', 'All'))
  
  task_file = {'NER': 'outputs/ontonotes/ner/Val_Acc_heatmap.json',
               'Phrase Start': 'outputs/ontonotes/phrase_start/Val_Acc_heatmap.json',
               'Phrase End': 'outputs/ontonotes/phrase_end/Val_Acc_heatmap.json', 
               'All': ['outputs/ontonotes/ner/Val_Acc_heatmap.json', 
                       'outputs/ontonotes/phrase_start/Val_Acc_heatmap.json',
                       'outputs/ontonotes/phrase_end/Val_Acc_heatmap.json']
               }
  st.markdown(f"<h2 style='text-align: center; color: black;'>Layer-wise Validation Accuracy </h2>", unsafe_allow_html=True)
  fig = retrieve_plot(task_file[option], name_idx=2)
  if isinstance(fig, list):
    for option, f in fig:
      st.markdown(f"<h2 style='text-align: center; color: red;'>{option} </h2>", unsafe_allow_html=True)
      st.plotly_chart(f, use_container_width=True)
  else:
    option, f = fig
    st.markdown(f"<h2 style='text-align: center; color: red;'>{option} </h2>", unsafe_allow_html=True)
    st.plotly_chart(f, use_container_width=True)

def aheads_heatmaps():
  st.write("## Attention Heads Heatmaps")
  
  option = st.selectbox(
    'Which head type would you like to visualize?',
    ('Duplicate Token Heads', 'Previous Token Heads', 'Induction Heads', 'All', 'Probing'))
  
  task_file = {'Duplicate Token Heads': ['outputs/aheads/duplicate_token_head/max_duplicate_head_deeper_heatmap.json', 
                                         'outputs/aheads/duplicate_token_head/max_duplicate_head_heatmap.json',
                                         'outputs/aheads/duplicate_token_head/mean_duplicate_head_deeper_heatmap.json', 
                                         'outputs/aheads/duplicate_token_head/mean_duplicate_head_heatmap.json', 
                                         'outputs/aheads/duplicate_token_head/Val_Acc_heatmap.json'],
               'Previous Token Heads': ['outputs/aheads/previous_token_head/max_previous_head_deeper_heatmap.json', 
                                        'outputs/aheads/previous_token_head/max_previous_head_heatmap.json', 
                                        'outputs/aheads/previous_token_head/mean_previous_head_deeper_heatmap.json', 
                                        'outputs/aheads/previous_token_head/mean_previous_head_heatmap.json', 
                                        'outputs/aheads/previous_token_head/Val_Acc_heatmap.json'],
               'Induction Heads': ['outputs/aheads/induction_head/max_induction_head_deeper_heatmap.json', 
                                   'outputs/aheads/induction_head/max_induction_head_heatmap.json', 
                                   'outputs/aheads/induction_head/mean_induction_head_deeper_heatmap.json', 
                                   'outputs/aheads/induction_head/mean_induction_head_heatmap.json', 
                                   'outputs/aheads/induction_head/Val_Acc_heatmap.json'],
               'All': ['outputs/aheads/duplicate_token_head/max_duplicate_head_deeper_heatmap.json', 
                      'outputs/aheads/duplicate_token_head/max_duplicate_head_heatmap.json', 
                      'outputs/aheads/previous_token_head/max_previous_head_deeper_heatmap.json', 
                      'outputs/aheads/previous_token_head/max_previous_head_heatmap.json',
                      'outputs/aheads/induction_head/max_induction_head_deeper_heatmap.json', 
                      'outputs/aheads/induction_head/max_induction_head_heatmap.json'],
               'Probing': ['outputs/aheads/duplicate_token_head/Val_Acc_heatmap.json', 
                           'outputs/aheads/previous_token_head/Val_Acc_heatmap.json', 
                           'outputs/aheads/induction_head/Val_Acc_heatmap.json'],
               }
  st.markdown(f"<h2 style='text-align: center; color: black;'>Layer-wise Max Attention Head Detection </h2>", unsafe_allow_html=True)
  fig = retrieve_plot(task_file[option], name_idx=2)
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
def retrieve_plot(file_path, name_idx=2):
  if isinstance(file_path, list):
    figs = []
    for fp in file_path:
      figs.append(retrieve_plot(fp, name_idx=name_idx))
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
  task_name = file_path.split('/')[name_idx].capitalize() + ": "
  task_name += file_path.split('.')[-2].split('/')[-1]
  task_name = task_name.replace('_', ' ')
  return (task_name, fig)

    
page_names_to_funcs = {
  "Introduction": intro,
  "Universal Dependencies": ud_heatmaps,
  "Penn Treebank 3": ptb_heatmaps,
  "Ontonotes-5": ontonotes_heatmaps,
  "Attention Heads": aheads_heatmaps,
  "Line Graph": line_graph
}

st.cache_resource.clear()
demo_name = st.sidebar.selectbox("Select a visualization", list(page_names_to_funcs.keys()))
page_names_to_funcs[demo_name]()