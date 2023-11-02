import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

if 'x' not in st.session_state: st.session_state.x = [0,1]
if 'y' not in st.session_state: st.session_state.y = [0,2]
if 'fig' not in st.session_state: 
    fig, ax = plt.subplots()
    ax.set_ylim(0,10)
    ax.set_xlim(0,10)
    st.session_state.fig = (fig, ax)

x = st.session_state.x
y = st.session_state.y
fig, ax = st.session_state.fig

columns = st.columns([1,1.5])

def addPoint():
    x.append(x[-1]+1)
    y.append(y[-1]*1.2)
    modifyAx()

def modifyAx():
    line = mlines.Line2D(x,y,2,"dashed","blue","X",20)
    ax.add_line(line)

with columns[0]:
    st.button("Add point", on_click=addPoint)

with columns[1]:
    st.markdown("#### Phase diagram")
    st.pyplot(fig)
