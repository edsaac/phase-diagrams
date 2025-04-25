import streamlit as st

st.set_page_config(
    page_title="Dynamical systems",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Get Help": "https://duckduckgo.com/",
        "Report a bug": "https://github.com/edsaac/dashboards/tree/Streamlit",
        "About": "# An *extremely* cool app!",
    },
)

st.title("Dynamical systems")

st.sidebar.success("Select a demo above.")

st.markdown("""
    Let's check some phase diagrams from simple cases!

    **👈 Select a demo from the sidebar** to see some examples of dynamical systems!
    """)

st.latex(r"""
    \begin{equation}
    \left\{
    \begin{array}{c}
    \frac{dx}{dt} = f(x,y) \\[1em]
    \frac{dy}{dt} = g(x,y)
    \end{array}
    \right.
    \end{equation}
""")
