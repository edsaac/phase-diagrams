import streamlit as st
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import root
import matplotlib.pyplot as plt

from utils import Parameter
from equations import adsorption_second_order


def plot_stream_matplotlib(initial_condition, parameters):
    ## Math globals
    q_max = parameters[-1]
    time = np.arange(0, 100, 0.2)
    clin = np.linspace(0.001, 1.5, 61)
    qlin = np.linspace(0.001, 1.1 * q_max, 61)
    c, q = np.meshgrid(clin, qlin)

    ## Calculate vector field
    dc, dq = adsorption_second_order([c, q], None, parameters)

    ## Calculate trayectories given initial condition
    tray = odeint(adsorption_second_order, initial_condition, time, args=(parameters,))
    c_t, q_t = tray.T  # <-- Timeseries

    ## Calculate steady-state
    sol = root(adsorption_second_order, [c_t[-1], q_t[-1]], args=(None, parameters))
    ssx, sss = sol.x

    kw_streams = {"density": 2, "linewidth": 1.5, "arrowsize": 0.8, "arrowstyle": "->", "color": "k", "minlength": 0.1}

    fig, ax = plt.subplots(figsize=[5, 5])
    # ax.set_aspect("equal")
    ax.streamplot(c, q, dc, dq, **kw_streams, zorder=2)
    ax.plot(c_t, q_t, lw=5, c="purple")
    ax.scatter([ssx], [sss], s=250, clip_on=False, ec="k", label="Steady-states", zorder=3)
    ax.scatter(c_t[0], q_t[0], s=250, marker="X", clip_on=False, ec="k", label=r"$(x_0,s_0)$", zorder=4, c="wheat")

    ax.grid(True, ls="dashed", lw=1, c=[0.5, 0.5, 0.5, 0.5])
    ax.set_xlabel("Dissolved $c$ [-]")
    ax.set_ylabel("Adsorbed $q$ [-]")
    ax.set(xlim=[0, clin.max()], ylim=[0, parameters[-1]])
    ax.legend(loc="lower right", fontsize=10)

    ## Breakthrough curves
    fig2, ax = plt.subplots(figsize=[5, 5])
    ax.plot(time, tray.T[0], lw=5, label="$c$")
    ax.plot(time, tray.T[1], lw=5, label="$q$")
    ax.axhline(1.0, ls="dashed", c="blue", label=R"$c_\infty$")
    ax.axhline(q_max, ls="dashed", c="orange", label=R"$q_{max}$")
    ax.set_ylabel(R"Concentration $c, q$ [-]")
    ax.set_xlabel(R"Time $t$ [-]")
    ax.set_ylim(bottom=-0.02)
    ax.legend()

    return fig, fig2


########################


def main():
    st.title("Adsorption-desorption")
    with st.expander("🔣 Governing equations", expanded=True):
        st.latex(R"""
            \begin{equation*}
                \left\{
                \begin{array}{rcl}
                    \dfrac{dc}{dt} 
                    &=& 
                    \underbrace{\tfrac{Q}{V}\left(c_\infty-c\right)}_{\textsf{Inflow \& outflow}}
                    -
                    \underbrace{k_{att} \left(1 - \dfrac{q}{q_{max}}\right) c}_{\textsf{Adsorption}}
                    +
                    \underbrace{k_{det} \left( \dfrac{q}{q_{max}} \right) q}_{\textsf{Desorption}}   
                    
                    \\[3em]
                
                    \dfrac{dq}{dt} &=& 
                    \underbrace{k_{att} \left(1 - \dfrac{q}{q_{max}}\right) c}_{\textsf{Adsorption}}
                    -
                    \underbrace{k_{det} \left( \dfrac{q}{q_{max}} \right) q}_{\textsf{Desorption}}   
                \end{array}
                \right.
            \end{equation*}""")

    ## Tweakable parameters
    Q_v = Parameter("Q_v", 0.1, "1/d", "Volumetric flow rate ", R"Q/V")
    k_att = Parameter("k_att", 1e-1, "mg(c)/L", "Adsorption rate ", R"k_{att}")
    k_det = Parameter("k_det", 1e-2, "mg(q)/L", "Desorption rate ", R"k_{det}")
    q_max = Parameter("q_max", 1.0, "mg(c)/mg(q)", "Adsorption capacity ", R"q_{max}")

    ## Interactives
    with st.sidebar:
        st.markdown(
            """
            - $c$: Dissolved concentration [mg/L]
            - $q$: Adsorbed concentration [mg/L]
            """
        )

        st.header("🎛️ Modify parameters")
        Q_v.render()
        k_att.render()
        k_det.render()
        q_max.render()

    initial_condition = {"c": 0.0, "q": 0.0}
    ndpr = [Q_v.value, k_att.value, k_det.value, q_max.value]

    st.divider()

    st.info("👈 You can tweak the system parameters in the sidebar")
    columns = st.columns([1, 1.5])
    with columns[0]:
        with st.expander("Initial condition:", expanded=True):
            initial_condition["c"] = st.slider("c₀", 0.0, 1.0, 0.0, 0.05, "%.2f")
            initial_condition["q"] = st.slider("q₀", 0.0, q_max.value, 0.0, 0.05, "%.2f")

    with columns[1]:
        with st.expander("Nondimensional values:", expanded=True):
            for i, p in enumerate(ndpr):
                st.latex(rf"p_{{{i}}} = {{{p:.3f}}}")

    st.markdown("#### Phase diagram")
    figs = plot_stream_matplotlib(list(initial_condition.values()), ndpr)

    for fig in figs:
        st.pyplot(fig)


if __name__ == "__main__":
    main()
