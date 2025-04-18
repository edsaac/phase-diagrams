import streamlit as st
from dataclasses import dataclass


@dataclass
class Parameter:
    name: str
    init_value: float
    units: str
    description: str
    latex: str

    def render(self):
        st.number_input(
            f"${self.latex}$",
            min_value=0.0,
            max_value=None,
            value=self.init_value,
            step=0.1,
            key=f"p_{self.name}",
            format="%.3f",
            help=f"{self.description} [{self.units}]",
        )

    @property
    def value(self):
        return st.session_state[f"p_{self.name}"]
