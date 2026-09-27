"""FabSentinel app entry point (Streamlit Cloud runs this file).

It sets page config, applies the shared style (views/style.py) and routes between
the three pages in src/views/:
  Home          - landing page (views/landing.py)
  Dashboard     - ML yield-risk + SHAP dashboard (views/overview.py)
  Yield Planner - physics-based yield/feasibility engine (views/yield_planner.py)
"""
import streamlit as st

from views.style import base_css

st.set_page_config(page_title="FabSentinel", page_icon=":material/memory:", layout="wide")

HOME = st.Page("views/landing.py", title="Home", icon=":material/home:", default=True)
DASHBOARD = st.Page("views/overview.py", title="Dashboard", icon=":material/monitoring:", url_path="dashboard")
PLANNER = st.Page("views/yield_planner.py", title="Yield Planner", icon=":material/calculate:", url_path="planner")

page = st.navigation([HOME, DASHBOARD, PLANNER], position="top")
# Dark mode is toggled on Home; the data pages stay light because Streamlit's
# widgets (tables, inputs) only follow the light theme in .streamlit/config.toml.
st.html(base_css(dark=st.session_state.get("dark", False) and page.title == "Home"))
page.run()
