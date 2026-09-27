"""FabSentinel app entry point (Streamlit Cloud runs this file).

It only sets page config and routes between the three pages in src/views/:
  Home          - landing page (views/landing.py)
  Dashboard     - ML yield-risk + SHAP dashboard (views/overview.py)
  Yield Planner - physics-based yield/feasibility engine (views/yield_planner.py)
"""
import streamlit as st

st.set_page_config(page_title="FabSentinel", page_icon=":material/memory:", layout="wide")

HOME = st.Page("views/landing.py", title="Home", icon=":material/home:", default=True)
DASHBOARD = st.Page("views/overview.py", title="Dashboard", icon=":material/monitoring:", url_path="dashboard")
PLANNER = st.Page("views/yield_planner.py", title="Yield Planner", icon=":material/calculate:", url_path="planner")

st.navigation([HOME, DASHBOARD, PLANNER], position="top").run()
