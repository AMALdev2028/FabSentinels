"""Yield Planner page: runs the FabYield engine from the browser.

Streamlit shows every file in src/pages/ as an extra page next to dashboard.py.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "fabyield"))
from fabyield import ConfigError, YieldEngine, parse_run_config  # noqa: E402

st.set_page_config(page_title="FabSentinel — Yield Planner", layout="wide")
st.title("Yield Planner")
st.caption("Deterministic yield, cycle-time and feasibility model (FabYield). "
           "Estimates from public data + stated assumptions — see fabyield/docs/TECHNICAL.md.")

with st.sidebar:
    node = st.selectbox("Process node", ["5nm", "7nm", "3nm"])
    tech = st.selectbox("Technology", ["logic", "memory", "analog", "power"])
    model = st.selectbox("Yield model", ["negative_binomial", "murphy", "poisson", "seeds"])
    w = st.number_input("Die width (mm)", 0.5, 40.0, 25.0, 0.5)
    h = st.number_input("Die height (mm)", 0.5, 40.0, 32.0, 0.5)
    target = st.number_input("Target good dies / month", 0, 10_000_000, 50_000, 1000)
    cap = st.number_input("Wafer starts capacity / month", 1, 1_000_000, 2500, 100)
    wafer = st.selectbox("Wafer diameter (mm)", [300, 450])
    edge = st.slider("Edge exclusion (mm)", 0.0, 6.0, 3.0, 0.5)
    bspdn = st.checkbox("Backside power delivery (3nm only)", disabled=node != "3nm")


@st.cache_data
def run(cfg: dict) -> dict:
    return YieldEngine(parse_run_config(cfg)).run()


cfg = {"target_good_dies_per_month": target, "die_width_mm": w, "die_height_mm": h, "wafer_diameter_mm": wafer,
       "process_node": node, "technology": tech, "yield_model": model, "wafer_starts_capacity_per_month": cap,
       "edge_exclusion_mm": edge, "backside_power_delivery": bspdn and node == "3nm",
       "monte_carlo": {"samples": 2000, "seed": 42}}
try:
    r = run(cfg)
except ConfigError as e:
    st.error(f"Invalid input: {e}")
    st.stop()

f, y, t = r["feasibility"], r["yield"], r["fabrication_time"]
{"ACHIEVABLE": st.success, "MARGINAL": st.warning, "INFEASIBLE": st.error}[f["verdict"]](
    f"**{f['verdict']}** — " + "; ".join(f["limiting_factors"]))

c = st.columns(4)
c[0].metric("Overall yield", f"{y['overall_yield']:.1%}")
c[1].metric("Good dies / month", f"{f['achievable_good_dies_per_month']:,.0f}", f"{f['gap_good_dies_per_month']:+,.0f} vs target")
c[2].metric("Fab cycle time", f"{t['calendar_days']:.0f} days", f"{t['mask_layers']} masks", delta_color="off")
c[3].metric("Wafer starts needed", f"{f['required_wafer_starts'] or 0:,}", f"capacity {cap:,}", delta_color="off")

a, b = st.columns(2)
with a:
    st.markdown("**Yield loss by mechanism (points)**")
    st.bar_chart(pd.Series({k.replace("_", " ").capitalize(): v["loss_points"]
                            for k, v in y["loss_breakdown"].items()}), horizontal=True)
    st.markdown("**Cycle time by module (days)**")
    st.bar_chart(pd.Series(t["critical_path"]["modules_days"]), horizontal=True)
with b:
    st.markdown("**Sensitivity (tornado)**")
    tor = pd.DataFrame(r["sensitivity"]["tornado"])
    st.dataframe(pd.DataFrame({
        "Parameter": tor["parameter"].str.replace("_", " ").str.capitalize(),
        "Varied by": tor["variation"].str.replace("+-", "±"),
        "Yield (low)": tor["yield_at_low"].map("{:.1%}".format),
        "Yield (high)": tor["yield_at_high"].map("{:.1%}".format),
        "Swing (pts)": tor["yield_swing_points"].round(1)}), hide_index=True)
    mc = r["sensitivity"]["monte_carlo"]
    st.caption(f"Monte Carlo 90% CI: {mc['yield']['p5']:.1%} – {mc['yield']['p95']:.1%}; "
               f"P(meet target) {mc['probability_meet_target']:.0%}")
    st.markdown("**Recommended improvements**")
    rec = pd.DataFrame(r["recommendations"])
    st.dataframe(pd.DataFrame({
        "Improvement": rec["name"], "Cost": rec["cost"],
        "Extra good dies / month": rec["good_dies_gain_per_month"].map("{:+,.0f}".format),
        "What it means": rec["action"]}), hide_index=True)

st.download_button("Download full result + audit trail (JSON)", json.dumps(r, indent=2, default=str),
                   "fabyield_result.json", "application/json")
