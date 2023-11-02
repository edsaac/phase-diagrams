import streamlit as st
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import root
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go

## Define the derivative system
st.title("Attachment kinetics")

tab_list = st.tabs(["🔣 Governing eqs.",
                    "Python function",
                   "➗ Numerical Solution"])

with tab_list[0]:
    st.latex(r'''
        \begin{equation*}
            \left\{
            \begin{array}{rcl}
                \dfrac{dc}{dt} &=& 
                    - k_{\rm att} \; c \\[1em]
                c(t=0) &=& c_0
            \end{array}
            \right.
        \end{equation*}''')

with tab_list[1]:
    with st.echo():
        def kinetic_attachment(c, t, katt):
            """
            dc/dt = -katt * c 
            """
            dcdt =  - katt*c

            return dcdt

    odeintcall = st.container()

with tab_list[2]:
    ## Math globals
    time = np.arange(0,50,0.5)
    katt = st.slider(r"$k_{\rm att}$ [1/s]", 0.0, 1.0, 0.01, 0.001)
    log_scale = st.checkbox("log-scale", False)

    ## Calculate trayectories given initial condition
    
    with odeintcall:
        with st.echo():
            tray = odeint(kinetic_attachment, 1.0, time, args=(katt,))

    fig,ax = plt.subplots()

    ax.plot(time, tray, c="xkcd:red", lw=3)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Rel. Concentration [-]")
    ax.set_xlim(min(time), max(time))
    ax.set_ylim(0,1.05)
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(1e-8,1.05)
    st.pyplot(fig)
