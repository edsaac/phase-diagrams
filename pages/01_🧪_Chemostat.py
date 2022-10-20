import streamlit as st
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import root
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go

## Define the derivative system
def stream(Y,t,p):
    '''
    dY/dt = f(Y,t,p) where Y can have n-dimensions
    but only 2 here to get a phase diagram
    '''
    x,s = Y
    dxdt =  x*s/(p[0]+s) - p[1]*x - p[2]*x
    dsdt =  p[2]*(1-s) - p[3]*x*s/(p[0]+s)
    return [dxdt,dsdt]

def plot_stream_matplotlib():

    ## Calculate vector field
    dx,ds = stream([x,s],None,ndpr)
    
    ## Calculate trayectories given initial condition
    tray = odeint(stream,list(init.values()),time,args=(ndpr,))

    ## Calculate steady-state
    sol = root(stream,[tray[-1,0],tray[-1,1]],args=(None,ndpr))
    ssx,sss = sol.x

    kw_streams = {'density':2,'linewidth':1.5,'arrowsize':0.8,
              'arrowstyle':'->','color':'k','minlength':0.1}

    fig,ax = plt.subplots(figsize=[5,5])
    ax.set_aspect("equal")
    ax.streamplot(x,s,dx,ds,**kw_streams,zorder=2)
    ax.plot(tray.T[0],tray.T[1],lw=5,c='purple')
    ax.scatter([0,ssx],[1,sss],s=250,clip_on=False,ec='k',label="Steady-states",zorder=3)
    ax.scatter(tray[0,0],tray[0,1],s=250,marker='X',clip_on=False,ec='k',
               label=r"$(x_0,s_0)$",zorder=4,c='wheat')

    ax.grid(True,ls='dashed',lw=1,c=[0.5,0.5,0.5,0.5])
    ax.set_xlabel("$x$ [-]")
    ax.set_ylabel("$s$ [-]")
    ax.set(xlim=[0,1],ylim=[0,1])
    ax.legend(loc='upper right',fontsize=10)
    
    return fig

def plot_stream_plotly():

    ## Calculate vector field
    dx,ds = stream([x,s],None,ndpr)
    
    ## Calculate trayectories given initial condition
    tray = odeint(stream,list(init.values()),time,args=(ndpr,))

    ## Calculate steady-state
    sol = root(stream,[tray[-1,0],tray[-1,1]],args=(None,ndpr))
    ssx,sss = sol.x

    # kw_streams = {'density':2,'linewidth':1.5,'arrowsize':0.8,
            #   'arrowstyle':'->','color':'k','minlength':0.1}

    # fig,ax = plt.subplots(figsize=[5,5])
    # ax.set_aspect("equal")
    # ax.streamplot(x,s,dx,ds,**kw_streams,zorder=2)
    # ax.plot(tray.T[0],tray.T[1],lw=5,c='purple')
    # ax.scatter([0,ssx],[1,sss],s=250,clip_on=False,ec='k',label="Steady-states",zorder=3)
    # ax.scatter(tray[0,0],tray[0,1],s=250,marker='X',clip_on=False,ec='k',
    #            label=r"$(x_0,s_0)$",zorder=4,c='wheat')

    # ax.grid(True,ls='dashed',lw=1,c=[0.5,0.5,0.5,0.5])
    # ax.set_xlabel("$x$ [-]",)
    # ax.set_ylabel("$s$ [-]")
    # ax.set(xlim=[0,1],ylim=[0,1])
    # ax.legend(loc='upper right',fontsize=10)
    
    fig = ff.create_streamline(xlin,slin,dx,ds, density=1, name="Streamlines")
    
    print(sol.x)

    fig.add_trace(go.Scatter(x=[0, ssx], y=[1, sss],
                          mode='markers',
                          marker_size=14,
                          name="Source point"))

    fig.add_trace(go.Scatter(x=[tray[0,0]], y=[tray[0,1]],
                          mode='markers',
                          marker_size=14,
                          name="$(x_0, s_0)$"))

    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])

    return fig

def resetValues():
    global parValues
    parValues = [0.42, 10.0 ,20.0 ,0.15, 1000.0/2000.0 ,50.0]

########################

st.title("Chemostat")
#st.header("Equations")
tabList = st.tabs(["🔣 Governing eqs.", 
                   "➗ Nondimensionalization",
                   "🔢 Nondimensional eqs.",
                   "🏛️ Steady-state solution"])

with tabList[0]:
    st.latex(r'''
        \begin{equation*}
            \left\{
            \begin{array}{rcl}
                \dfrac{dX}{dt} &=& 
                    \underbrace{Y\hat{q}\dfrac{S}{K_S + S}X}_{\textsf{Growth}}
                    - 
                    \underbrace{bX}_{\textsf{Decay}}
                    -
                    \underbrace{\tfrac{Q}{V}X}_{\textsf{Outflow}}
                    \\[3em]
                \dfrac{dS}{dt} &=& 
                    \underbrace{\tfrac{Q}{V}(S_\infty-S)}_{\textsf{In/outflow}}
                    -
                    \underbrace{\hat{q}\dfrac{S}{K_S + S}X}_{\textsf{Consumption}}
            \end{array}
            \right.
        \end{equation*}''')

with tabList[1]:
    st.latex(r'''\begin{equation*}
        \begin{array}{rcl}
            s &=& S/S_\infty \\
            x &=& X/X_{\rm max} \\
            \tau &=& \hat{q}Yt \\
        \end{array}
    \end{equation*}''')

    st.latex(r'''
    \begin{equation*}
        \begin{array}{rcl}
            \pi_0 &=& K_S/S_\infty \\
            \pi_1 &=& b/\hat{q}Y \\
            \pi_2 &=& Q/V\hat{q}Y \\
            \pi_3 &=& X_{\rm max}/S_\infty Y \\
        \end{array}
    \end{equation*}''')

with tabList[2]: 
    st.latex(r'''
    \begin{equation*}
        \left\{
        \begin{array}{rcl}
            \dfrac{dx}{d\tau} &=& x\dfrac{s}{\pi_0 + s} - \pi_1 x - \pi_2 x \\[2em]
            \dfrac{ds}{d\tau} &=& \pi_2 \left(1-s\right) - \pi_3 x\dfrac{s}{\pi_0 + s}\\
        \end{array}
        \right.
    \end{equation*}''')

with tabList[3]:
    st.latex(r'''
    \begin{equation*}
        \begin{array}{cc}
            \textsf{Trivial:} & 
                x=0 & s=1 \\[2em]
            \textsf{Non-trivial:} &
                x = \dfrac{\pi_{2} \left(\pi_{0} \pi_{1} + \pi_{0} \pi_{2} + \pi_{1} + \pi_{2} - 1\right)}{\pi_{3} \left(\pi_{1} + \pi_{2}\right) \left(\pi_{1} + \pi_{2} - 1 \right)} & s = \dfrac{\pi_{0} \left(\pi_{1} + \pi_{2}\right)}{1 - \pi_{1} - \pi_{2}}
        \end{array}
    \end{equation*}''')

## Math globals
time = np.arange(0,50,0.5)
xlin = np.linspace(0.001,1.1,61)
slin = np.linspace(0.001,1.1,61)
x,s  = np.meshgrid(xlin,slin)

## This should be a dataclass
parKeys   = ['Y','q','Ks','b','Q/V','S0']
parLabels = [r'Y', r'\hat{q}', r'K_s', r'b', r'Q/V',    r'S^0']
parValues = [0.42, 10.0 ,20.0 ,0.15, 1000.0/2000.0 ,50.0 ]
parUnits  = ["mg(X)/mg(S)","mg(S)/mg(X)·d","mg(S)/L","1/d","1/d","mg(S)/L"]
parDescr  = ["Microbial yield ",
            "Max growth rate",
            "Half-saturation rate",
            "Die-off rate",
            "Dilution rate",
            "Substrate concentration"]

## Interactives
with st.sidebar:
    st.title('System control')
    st.header("🎚️ State variables")
    st.markdown(
         """
         - $$X$$: Biomass concentration [mg<sub>X</sub>/L]
         - $$S$$: Substrate concentration [mg<sub>S</sub>/L]
         """,True)

    st.header("🎛️ Modify parameters")
    with st.expander("Parameters:",expanded=True):    
        
        tabs = st.tabs(parKeys)
        for i,tab in enumerate(tabs):
            with tab:
                st.latex(rf"{{{parLabels[i]}}}: \textsf{{{parDescr[i]}}}")
                parValues[i] = st.number_input(
                        f"[{parUnits[i]}]", 
                        None , None, parValues[i], 0.05,
                        key=f"p_{i}", format="%.2f")

init = {'X':0.01,'S':0.01}
pars = {k:v for k,v in zip(parKeys,parValues)}
ndpr = [None] * 4

ndpr[0] = pars['Ks']/pars['S0']
ndpr[1] = pars['b']/(pars['q'] * pars['Y'])
ndpr[2] = pars['Q/V']/(pars['q'] * pars['Y'])
ndpr[3] = 1.0

st.header("")
columns = st.columns([1,1.5])
with columns[0]:
    st.info('👈 You can tweak the system parameters in the sidebar')

    with st.expander("Initial condition:",expanded=True):
        init['X'] = st.slider("x₀",0.0,1.0,0.01,0.05,"%.2f")
        init['S'] = st.slider("s₀",0.0,1.0,0.01,0.05,"%.2f")
    
    with st.expander("Nondimensional values:",expanded=False):
        for i,p in enumerate(ndpr):
            st.latex(rf"p_{{{i}}} = {{{p:.3f}}}")    

with columns[1]:
    st.markdown("#### Phase diagram")
    st.pyplot(plot_stream_matplotlib())
    st.plotly_chart(plot_stream_plotly(), use_container_width=True)
