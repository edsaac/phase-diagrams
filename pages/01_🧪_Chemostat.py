import streamlit as st
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import root
import matplotlib.pyplot as plt

from utils import Parameter
from equations import chemostat


def plot_stream_matplotlib(initial_condition, parameters):
    ## Calculate vector field
    time = np.arange(0, 50, 0.5)
    xlin = np.linspace(0.001, 1.1, 61)
    slin = np.linspace(0.001, 1.1, 61)
    x, s = np.meshgrid(xlin, slin)
    dx, ds = chemostat([x, s], None, parameters)

    ## Calculate trayectories given initial condition
    tray = odeint(chemostat, initial_condition, time, args=(parameters,))
    x_t, s_t = tray.T  # <-- Timeseries

    ## Calculate steady-state (dxdt = dsdt = 0) using last time step of trajectory
    sol = root(chemostat, [x_t[-1], s_t[-1]], args=(None, parameters))
    ssx, sss = sol.x

    ## Plot
    kw_streams = {"density": 2, "linewidth": 1.5, "arrowsize": 0.8, "arrowstyle": "->", "color": "k", "minlength": 0.1}

    fig, ax = plt.subplots(figsize=[5, 5])
    ax.streamplot(x, s, dx, ds, **kw_streams, zorder=2)
    ax.plot(x_t, s_t, lw=5, c="purple")
    ax.scatter([0, ssx], [1, sss], s=250, clip_on=False, ec="k", label="Steady-states", zorder=3)
    ax.scatter(
        tray[0, 0], tray[0, 1], s=250, marker="X", clip_on=False, ec="k", label=R"$(x_0, s_0)$", zorder=4, c="wheat"
    )

    ax.set_aspect("equal")
    ax.grid(True, ls="dashed", lw=1, c=[0.5, 0.5, 0.5, 0.5])
    ax.set_xlabel("Biomass $x = X/X_{max}$ [-]")
    ax.set_ylabel("Substrate $s = S/S_0$ [-]")
    ax.set(xlim=[0, 1], ylim=[0, 1])
    ax.legend(loc="upper right", fontsize=10)

    return fig


def main():
    st.title("Chemostat")

    with st.expander("🔣 Governing equations", expanded=True):
        st.latex(R"""
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
            \end{equation*}""")

    with st.expander("➗ Nondimensionalization"):
        st.latex(R"""
            \begin{equation*}
            \begin{array}{rcl}
                s &=& S/S_\infty \\
                x &=& X/X_{\rm max} = X/(S_\infty Y) \\
                \tau &=& \hat{q}Yt \\
            \end{array}
        \end{equation*}""")

        st.latex(R"""
        \begin{equation*}
            \begin{array}{rcl}
                \pi_0 &=& K_S/S_\infty \\
                \pi_1 &=& b/\hat{q}Y \\
                \pi_2 &=& Q/V\hat{q}Y \\
                \pi_3 &=& X_{\rm max}/(S_\infty Y) = 1\\
            \end{array}
        \end{equation*}""")

    with st.expander("🔢 Nondimensional equations"):
        st.latex(R"""
        \begin{equation*}
            \left\{
            \begin{array}{rcl}
                \dfrac{dx}{d\tau} &=& x\dfrac{s}{\pi_0 + s} - \pi_1 x - \pi_2 x \\[2em]
                \dfrac{ds}{d\tau} &=& \pi_2 \left(1-s\right) - \pi_3 x\dfrac{s}{\pi_0 + s}\\
            \end{array}
            \right.
        \end{equation*}""")

    with st.expander("🏛️ Steady-state solution"):
        st.latex(R"""
        \begin{equation*}
            \begin{array}{cc}
                \textsf{Trivial:} & 
                    x=0 & s=1 \\[2em]
                \textsf{Non-trivial:} &
                    x = \dfrac{\pi_{2} \left(\pi_{0} \pi_{1} + \pi_{0} \pi_{2} + \pi_{1} + \pi_{2} - 1\right)}{\pi_{3} \left(\pi_{1} + \pi_{2}\right) \left(\pi_{1} + \pi_{2} - 1 \right)} & s = \dfrac{\pi_{0} \left(\pi_{1} + \pi_{2}\right)}{1 - \pi_{1} - \pi_{2}}
            \end{array}
        \end{equation*}""")

    ## Tweakable parameters
    microbial_yield = Parameter("Y", 0.42, "mg(X)/mg(S)", "Microbial yield ", R"Y")
    max_growth_rate = Parameter("q", 10.0, "mg(S)/mg(X)·d", "Max growth rate ", R"\hat{q}")
    half_saturation_rate = Parameter("Ks", 20.0, "mg(S)/L", "Half-saturation rate ", R"K_s")
    die_off_rate = Parameter("b", 0.15, "1/d", "Die-off rate ", R"b")
    dilution_rate = Parameter("Q/V", 1000.0 / 2000.0, "1/d", "Dilution rate ", R"Q/V")
    substrate_concentration = Parameter("S0", 50.0, "mg(S)/L", "Substrate concentration ", R"S_\infty")

    with st.sidebar:
        st.header("🎛️ Modify parameters")
        microbial_yield.render()
        max_growth_rate.render()
        half_saturation_rate.render()
        die_off_rate.render()
        dilution_rate.render()
        substrate_concentration.render()

    initial_condition = {"X": 0.01, "S": 0.01}

    ndpr = [None] * 4
    ndpr[0] = half_saturation_rate.value / substrate_concentration.value
    ndpr[1] = die_off_rate.value / (max_growth_rate.value * microbial_yield.value)
    ndpr[2] = dilution_rate.value / (max_growth_rate.value * microbial_yield.value)
    ndpr[3] = 1.0

    st.divider()
    columns = st.columns([1, 1.5])
    with columns[0]:
        st.info("👈 You can tweak the system parameters in the sidebar")

        with st.expander("Initial condition:", expanded=True):
            initial_condition["X"] = st.slider("x₀", 0.0, 1.0, 0.01, 0.05, "%.2f")
            initial_condition["S"] = st.slider("s₀", 0.0, 1.0, 0.01, 0.05, "%.2f")

        with st.expander("Nondimensional values:", expanded=False):
            for i, p in enumerate(ndpr):
                st.latex(rf"p_{{{i}}} = {{{p:.3f}}}")

    with columns[1]:
        st.markdown("#### Phase diagram")
        st.pyplot(plot_stream_matplotlib(list(initial_condition.values()), ndpr))


if __name__ == "__main__":
    main()
