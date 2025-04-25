import streamlit as st
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from utils import Parameter
from equations import adsorption_triple_competition


def plot_stream_matplotlib(initial_condition, parameters):
    ## Math globals
    Q_v, cinit_0, k_att0, k_det0, cinit_1, k_att1, k_det1, cinit_2, k_att2, k_det2, q_max = parameters

    ## Calculate trayectories given initial condition
    time = np.arange(0, 100, 0.1)
    tray = odeint(adsorption_triple_competition, initial_condition, time, args=(parameters,))
    c0_t, c1_t, c2_t, q0_t, q1_t, q2_t = tray.T  # <-- Timeseries

    fig, axs = plt.subplots(1, 3, figsize=[14, 5])
    ax = axs[0]
    ax.set_title("$c_0, q_0$")
    ax.plot(c0_t, q0_t, lw=5, c="purple")
    ax.grid(True, ls="dashed", lw=1, c=[0.5, 0.5, 0.5, 0.5])
    ax.set_xlabel("Dissolved $c_0$ [-]")
    ax.set_ylabel("Adsorbed $q_0$ [-]")
    ax.set(xlim=[0, 1.2 * cinit_0], ylim=[0, q_max])

    ax = axs[1]
    ax.set_title("$c_1, q_1$")
    ax.plot(c1_t, q1_t, lw=5, c="purple")
    ax.grid(True, ls="dashed", lw=1, c=[0.5, 0.5, 0.5, 0.5])
    ax.set_xlabel("Dissolved $c_1$ [-]")
    ax.set_ylabel("Adsorbed $q_1$ [-]")
    ax.set(xlim=[0, 1.2 * cinit_1], ylim=[0, q_max])

    ax = axs[2]
    ax.set_title("$c_2, q_2$")
    ax.plot(c2_t, q2_t, lw=5, c="purple")
    ax.grid(True, ls="dashed", lw=1, c=[0.5, 0.5, 0.5, 0.5])
    ax.set_xlabel("Dissolved $c_1$ [-]")
    ax.set_ylabel("Adsorbed $q_1$ [-]")
    ax.set(xlim=[0, 1.2 * cinit_2], ylim=[0, q_max])

    ## Breakthrough curves
    fig2, axs = plt.subplots(1, 2, figsize=[11, 5], sharex=True)
    ax = axs[0]
    ax.plot(time, c0_t / cinit_0, lw=2, label="$c_0$")
    ax.plot(time, c1_t / cinit_1, lw=2, label="$c_1$")
    ax.plot(time, c2_t / cinit_2, lw=2, label="$c_2$")

    ax.axhline(1.0, ls="dashed", c="blue", label=R"$c_\infty$")
    ax.set_ylabel(R"$c_i$ [-]")
    ax.set_xlabel(R"Time $t$ [-]")
    ax.set_ylim(bottom=-0.02)
    ax.legend()

    ax = axs[1]
    ax.plot(time, q0_t, lw=2, label="$q_0$")
    ax.plot(time, q1_t, lw=2, label="$q_1$")
    ax.plot(time, q2_t, lw=2, label="$q_1$")

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
                    \dfrac{dc_i}{dt} 
                    &=& 
                    \underbrace{\tfrac{Q}{V}\left(c_{\infty,0} - c_i\right)}_{\textsf{Inflow \& outflow}}
                    -
                    \underbrace{k_{att, i} \left(1 - \dfrac{\sum_j{q_j}}{q_{max}}\right) c_i}_{\textsf{Adsorption}}
                    +
                    \underbrace{k_{det, i} \phantom{\dfrac{}{}} q_i}_{\textsf{Desorption}}   
                    
                    \\[3em]
                
                    \dfrac{dq_i}{dt} &=& 
                    \underbrace{k_{att, i} \left(1 - \dfrac{\sum_j{q_j}}{q_{max}}\right) c_i}_{\textsf{Adsorption}}
                    -
                    \underbrace{k_{det, i} \phantom{\dfrac{}{}} q_i}_{\textsf{Desorption}}   
                 
                \end{array}
                \right.
            \end{equation*}""")

    ## Tweakable parameters
    Q_v = Parameter("Q_v", 0.2, "1/d", "Volumetric flow rate ", R"Q/V")
    q_max = Parameter("q_max", 1.0, "mg(c)/mg(q)", "Adsorption capacity ", R"q_{max}")

    cinit_0 = Parameter("c_0", 1.0, "mg/L", "Influent concentration compound 0", R"c_{\infty, 0}")
    k_att0 = Parameter("k_{att, 0}", 0.1, "1/d", "Adsorption rate compound 0", R"k_{att, 0}")
    k_det0 = Parameter("k_{det, 0}", 0.05, "1/d", "Desorption rate compound 0", R"k_{det, 0}")

    cinit_1 = Parameter("c_1", 1.0, "mg/L", "Influent concentration compound 1", R"c_{\infty, 1}")
    k_att1 = Parameter("k_{att, 1}", 0.1, "1/d", "Adsorption rate compound 1", R"k_{att, 1}")
    k_det1 = Parameter("k_{det, 1}", 0.0, "1/d", "Desorption rate compound 1", R"k_{det, 1}")

    cinit_2 = Parameter("c_2", 10.0, "mg/L", "Influent concentration compound 2", R"c_{\infty, 2}")
    k_att2 = Parameter("k_{att, 2}", 0.5, "1/d", "Adsorption rate compound 2", R"k_{att, 2}")
    k_det2 = Parameter("k_{det, 2}", 0.4, "1/d", "Desorption rate compound 2", R"k_{det, 2}")

    ## Interactives
    with st.sidebar:
        st.markdown(
            """
            - $c_0, c_1, c_2$: Dissolved concentration [mg/L]
            - $q_0, q_1, c_2$: Adsorbed concentration [mg/L]
            """
        )

        st.header("🎛️ Modify parameters")
        Q_v.render()
        q_max.render()

        st.subheader("**Contaminant 0**")
        cinit_0.render()
        k_att0.render()
        k_det0.render()

        st.subheader("**Contaminant 1**")
        cinit_1.render()
        k_att1.render()
        k_det1.render()

        st.subheader("**Contaminant 2**")
        cinit_2.render()
        k_att2.render()
        k_det2.render()

    initial_condition = {"c0": 0.0, "c1": 0.0, "c2": 0.0, "q0": 0.0, "q1": 0.0, "q2": 0.0}
    ndpr = [
        Q_v.value,
        cinit_0.value,
        k_att0.value,
        k_det0.value,
        cinit_1.value,
        k_att1.value,
        k_det1.value,
        cinit_2.value,
        k_att2.value,
        k_det2.value,
        q_max.value,
    ]

    st.divider()

    st.info("👈 You can tweak the system parameters in the sidebar")

    with st.expander("Initial condition:", expanded=True):
        lcol, rcol = st.columns(2)
        with lcol:
            initial_condition["c0"] = st.slider("$c_0(t=0)$", 0.0, 1.0, 0.0, 0.05, "%.2f")
            initial_condition["c1"] = st.slider("$c_1(t=0)$", 0.0, 1.0, 0.0, 0.05, "%.2f")
            initial_condition["c2"] = st.slider("$c_2(t=0)$", 0.0, 1.0, 0.0, 0.05, "%.2f")

        with rcol:
            initial_condition["q0"] = st.slider("$q_0(t=0)$", 0.0, q_max.value, 0.0, 0.05, "%.2f")
            initial_condition["q1"] = st.slider("$q_1(t=0)$", 0.0, q_max.value, 0.0, 0.05, "%.2f")
            initial_condition["q2"] = st.slider("$q_2(t=0)$", 0.0, q_max.value, 0.0, 0.05, "%.2f")

    st.markdown("#### Phase diagram")
    figs = plot_stream_matplotlib(list(initial_condition.values()), ndpr)

    for fig in figs:
        st.pyplot(fig)


if __name__ == "__main__":
    main()
