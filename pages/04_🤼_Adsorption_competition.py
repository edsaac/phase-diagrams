import streamlit as st
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from utils import Parameter
from equations import adsorption_competition


def plot_stream_matplotlib(initial_condition, parameters):
    ## Math globals
    q_max = parameters[-1]

    ## Calculate trayectories given initial condition
    time = np.arange(0, 100, 0.1)
    tray = odeint(adsorption_competition, initial_condition, time, args=(parameters,))
    c0_t, c1_t, q0_t, q1_t = tray.T  # <-- Timeseries

    fig, axs = plt.subplots(1, 2, figsize=[9, 5])
    ax = axs[0]
    ax.set_title("$c_0, q_0$")
    ax.plot(c0_t, q0_t, lw=5, c="purple")
    ax.grid(True, ls="dashed", lw=1, c=[0.5, 0.5, 0.5, 0.5])
    ax.set_xlabel("Dissolved $c_0$ [-]")
    ax.set_ylabel("Adsorbed $q_0$ [-]")
    ax.set(xlim=[0, 1.1], ylim=[0, parameters[-1]])

    ax = axs[1]
    ax.set_title("$c_1, q_1$")
    ax.plot(c1_t, q1_t, lw=5, c="purple")
    ax.grid(True, ls="dashed", lw=1, c=[0.5, 0.5, 0.5, 0.5])
    ax.set_xlabel("Dissolved $c_1$ [-]")
    ax.set_ylabel("Adsorbed $q_1$ [-]")
    ax.set(xlim=[0, 1.1], ylim=[0, parameters[-1]])

    ## Breakthrough curves
    fig2, axs = plt.subplots(1, 2, figsize=[9, 5], sharex=True)
    ax = axs[0]
    ax.plot(time, c0_t, lw=2, label="$c_0$")
    ax.plot(time, c1_t, lw=2, label="$q_0$")
    ax.axhline(1.0, ls="dashed", c="blue", label="$c_\infty$")
    ax.set_ylabel(R"$c_i$ [-]")
    ax.set_xlabel(R"Time $t$ [-]")
    ax.set_ylim(bottom=-0.02)
    ax.legend()

    ax = axs[1]
    ax.plot(time, q0_t, lw=2, label="$q_0$")
    ax.plot(time, q1_t, lw=2, label="$q_1$")
    ax.axhline(q_max, ls="dashed", c="orange", label=R"$q_{max}$")
    ax.set_ylabel(R"$q_i$ [-]")
    ax.set_xlabel(R"Time $t$ [-]")
    ax.set_ylim(bottom=-0.02)
    ax.legend()

    return fig, fig2


def main():
    st.title("Competitive Adsoprtion-desorption")
    with st.expander("🔣 Governing equations", expanded=True):
        st.latex(R"""
            \begin{equation*}
                \left\{
                \begin{array}{rcl}
                    \dfrac{dc_0}{dt} 
                    &=& 
                    \underbrace{\tfrac{Q}{V}\left(c_{\infty,0} - c_0\right)}_{\textsf{Inflow \& outflow}}
                    -
                    \underbrace{k_{att, 0} \left(1 - \dfrac{\sum_i{q_i}}{q_{max}}\right) c_0}_{\textsf{Adsorption}}
                    +
                    \underbrace{k_{det, 0} \phantom{\dfrac{}{}} q_0}_{\textsf{Desorption}}   
                    
                    \\[3em]
                    \dfrac{dc_1}{dt} 
                    &=& 
                    \underbrace{\tfrac{Q}{V}\left(c_{\infty,1} - c_1\right)}_{\textsf{Inflow \& outflow}}
                    -
                    \underbrace{k_{att, 1} \left(1 - \dfrac{\sum_i{q_i}}{q_{max}}\right) c_1}_{\textsf{Adsorption}}
                    +
                    \underbrace{k_{det, 1} \phantom{\dfrac{}{}} q_1}_{\textsf{Desorption}}   
                    
                    \\[3em]
                
                    \dfrac{dq_0}{dt} &=& 
                    \underbrace{k_{att, 0} \left(1 - \dfrac{\sum_i{q_i}}{q_{max}}\right) c_0}_{\textsf{Adsorption}}
                    -
                    \underbrace{k_{det, 0} \phantom{\dfrac{}{}} q_0}_{\textsf{Desorption}}   
                    
                    \\[3em]
                    
                    \dfrac{dq_1}{dt} &=& 
                    \underbrace{k_{att, 1} \left(1 - \dfrac{\sum_i{q_i}}{q_{max}}\right) c_1}_{\textsf{Adsorption}}
                    -
                    \underbrace{k_{det, 1} \phantom{\dfrac{}{}} q_1}_{\textsf{Desorption}}
                \end{array}
                \right.
            \end{equation*}""")

    ## Tweakable parameters
    Q_v = Parameter("Q_v", 0.2, "1/d", "Volumetric flow rate ", R"Q/V")
    k_att0 = Parameter("k_{att, 0}", 0.8, "1/d", "Adsorption rate compound 0", R"k_{att, 0}")
    k_det0 = Parameter("k_{det, 0}", 0.5, "1/d", "Desorption rate compound 0", R"k_{det, 0}")
    k_att1 = Parameter("k_{att, 1}", 0.05, "1/d", "Adsorption rate compound 1", R"k_{att, 1}")
    k_det1 = Parameter("k_{det, 1}", 0.01, "1/d", "Desorption rate compound 1", R"k_{det, 1}")
    q_max = Parameter("q_max", 1.0, "mg(c)/mg(q)", "Adsorption capacity ", R"q_{max}")

    ## Interactives
    with st.sidebar:
        st.markdown(
            """
            - $c_0, c_1$: Dissolved concentration [mg/L]
            - $q_0, q_1$: Adsorbed concentration [mg/L]
            """
        )

        st.header("🎛️ Modify parameters")
        Q_v.render()
        k_att0.render()
        k_det0.render()
        k_att1.render()
        k_det1.render()
        q_max.render()

    initial_condition = {"c0": 0.0, "c1": 0.0, "q0": 0.0, "q1": 0.0}
    ndpr = [Q_v.value, k_att0.value, k_det0.value, k_att1.value, k_det1.value, q_max.value]

    st.divider()

    st.info("👈 You can tweak the system parameters in the sidebar")

    with st.expander("Initial condition:", expanded=True):
        initial_condition["c0"] = st.slider("c₀", 0.0, 1.0, 0.0, 0.05, "%.2f")
        initial_condition["c1"] = st.slider("c1", 0.0, 1.0, 0.0, 0.05, "%.2f")
        initial_condition["q0"] = st.slider("q₀", 0.0, q_max.value, 0.0, 0.05, "%.2f")
        initial_condition["q1"] = st.slider("q1", 0.0, q_max.value, 0.0, 0.05, "%.2f")

    st.markdown("#### Phase diagram")
    figs = plot_stream_matplotlib(list(initial_condition.values()), ndpr)

    for fig in figs:
        st.pyplot(fig)


if __name__ == "__main__":
    main()
