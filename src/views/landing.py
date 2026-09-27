"""Home / landing page for FabSentinel (Streamlit port of the "FabSentinel — Landing" design).

Layout, top to bottom (same as the design):
  glass nav bar -> hero + live wafer map -> dataset stats -> "How it works" (4 steps)
  -> stack -> call to action + team -> footer

Code map:
  CONFIG            colours, copy, defaults - edit here to customise
  load_stats()      data loading (real metrics.json, with a mock fallback)
  make_wafer()      data processing (deterministic wafer map from a seed)
  css() / *_html()  visualisation (HTML/CSS strings)
  render_*()        UI rendering, one function per section
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import streamlit as st

from views.style import ACCENT, TOKENS   # shared colours - edit views/style.py to re-theme every page

# =============================================================================
# CONFIG - change copy and defaults here (colours live in views/style.py)
# =============================================================================
ROOT = Path(__file__).resolve().parents[2]
CONFIG = {
    "accent": ACCENT,
    "light": TOKENS["light"],
    "dark": TOKENS["dark"],
    "wafer_grid": 16,               # dies per row/column in the illustrative wafer
    "flag_rate": 0.07,              # share of dies shown as flagged
    "scan_interval_s": 1.0,         # live wafer refresh period
    "github_url": "https://github.com/AMALdev2028/FabSentinels",
    "team": "Built by Amal Dev G &amp; Abhijay · Mentored by Ms. M. Harshanya, ECE Department<br>"
            "Chennai Institute of Technology — Project Based Learning",
}
# Shown when artifacts/metrics.json cannot be read (public SECOM dataset figures).
MOCK_STATS = {"runs": 1567, "sensors": 590, "failures": 104}

STEPS = [
    {"name": "Ingest", "tool": "Pandas", "title": "Clean the sensor data",
     "body": "Load the SECOM runs, drop sensors that never change or are mostly missing, and fill the "
             "remaining gaps so every run has a complete, comparable signal profile."},
    {"name": "Balance", "tool": "SMOTE", "title": "Fix the class imbalance",
     "body": "Only about 1 in 15 runs fails. SMOTE creates synthetic failing examples between real ones, "
             "so the model learns what failure looks like instead of always guessing “pass”."},
    {"name": "Predict", "tool": "XGBoost", "title": "Classify pass or fail",
     "body": "A gradient-boosted tree ensemble learns non-linear interactions between sensors and outputs "
             "a pass/fail prediction with a probability for every production run."},
    {"name": "Explain", "tool": "SHAP", "title": "Point to the root cause",
     "body": "SHAP breaks each prediction into per-sensor contributions. The sensors pushing hardest toward "
             "“fail” become a ranked shortlist for engineers to investigate."},
]
STACK = [("XGBoost", "Gradient-boosted classifier"), ("SMOTE", "Minority-class oversampling"),
         ("SHAP", "Per-prediction explanations"), ("Streamlit", "Four-tab live dashboard")]


# =============================================================================
# Data loading
# =============================================================================
@st.cache_data
def load_stats() -> tuple[dict[str, int], bool]:
    """Dataset figures for the stats strip. Returns (stats, is_live).

    Reads artifacts/metrics.json (written by train.py) and the raw sensor count from
    data/secom.data. Any failure falls back to MOCK_STATS so the page still renders;
    swap this function for an API call if the numbers ever live elsewhere.
    """
    try:
        m = json.loads((ROOT / "artifacts" / "metrics.json").read_text())
        with open(ROOT / "data" / "secom.data") as f:
            sensors = len(f.readline().split())
        runs = int(m["n_samples"])
        return {"runs": runs, "sensors": sensors, "failures": round(runs * float(m["failure_rate"]))}, True
    except (OSError, KeyError, ValueError, TypeError):
        return MOCK_STATS, False


# =============================================================================
# Processing
# =============================================================================
def make_wafer(seed: int, n: int = CONFIG["wafer_grid"], rate: float = CONFIG["flag_rate"]) -> list[bool]:
    """n*n dies, True = flagged. Same seed -> same map (illustrative, not model output)."""
    rng = random.Random(seed)
    return [rng.random() < rate for _ in range(n * n)]


# =============================================================================
# Visualisation (HTML / CSS) - page-specific rules; shared ones are in views/style.py
# =============================================================================
def css(t: dict, accent: str) -> str:
    return f"""<style>
header[data-testid="stHeader"] {{ display:none; }}           /* the glass nav replaces it on this page */
.block-container {{ padding-top: 1rem; }}
.stApp p, .stApp label, .stApp span {{ color: inherit; }}
.st-key-nav {{ position: sticky; top: 12px; z-index: 999; border-radius: 36px; padding: 6px 10px 6px 24px; }}
.fs-brand {{ display:flex; align-items:center; gap:10px; font-weight:600; font-size:17px; color:var(--ink); }}
.fs-links {{ display:flex; gap:28px; font-size:14px; }}
.fs-links a {{ color: var(--muted) !important; text-decoration:none; }}
.fs-links a:hover {{ color: var(--ink) !important; }}
.st-key-nav [data-testid="stMarkdownContainer"] p {{ font-size:17px; margin:0; }}
.st-key-cta_hero a {{ width: fit-content; }}
.st-key-cta_end {{ align-items: center; margin-top: 16px; }}
/* page links styled as pill buttons */
.st-key-cta_nav a, .st-key-cta_hero a, .st-key-cta_end a {{
  border-radius: 26px; padding: 0.55rem 1.3rem; justify-content:center; text-decoration:none;
}}
.st-key-cta_nav a, .st-key-cta_end a {{ background: var(--ink); }}
.st-key-cta_nav a p, .st-key-cta_end a p {{ color: var(--bg) !important; font-weight:500; }}
.st-key-cta_hero a {{ background: var(--accent); padding: 0.8rem 1.6rem; }}
.st-key-cta_hero a p {{ color: #FFFFFF !important; font-weight:500; font-size:15px; }}
.fs-eyebrow {{ font-family:'IBM Plex Mono',monospace; font-size:12px; letter-spacing:.12em; text-transform:uppercase; color:var(--accent); }}
.fs-h1 {{ font-family:'Instrument Serif',Georgia,serif; font-weight:400; font-size:76px; line-height:1.02;
         letter-spacing:-.02em; margin:18px 0; color:var(--ink); }}
.fs-h1 em, .fs-h2 em {{ color: var(--accent); }}
.fs-h2 {{ font-family:'Instrument Serif',Georgia,serif; font-weight:400; font-size:52px; line-height:1.05;
         letter-spacing:-.015em; margin:10px 0 0; color:var(--ink); }}
.fs-lead {{ font-size:18px; line-height:1.6; color:var(--muted); max-width:480px; }}
.fs-card {{ border-radius:28px; padding:36px 40px; }}
.fs-stats {{ display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); padding:0 40px; }}
.fs-stats > div {{ padding:32px 0; }}
.fs-stats > div + div {{ border-left:1px solid var(--line); padding-left:32px; }}
.fs-num {{ font-family:'Instrument Serif',Georgia,serif; font-size:48px; line-height:1; }}
.fs-small {{ font-size:14px; color:var(--muted); margin-top:6px; }}
.fs-mono {{ font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--accent); }}
.fs-row {{ display:flex; justify-content:space-between; align-items:baseline; padding:18px 0; border-bottom:1px solid var(--line); }}
.fs-row b {{ font-size:20px; font-weight:500; }}
.fs-wafer {{ width:min(400px, 86vw); aspect-ratio:1; border-radius:50%; padding:18px; margin:0 auto; box-sizing:border-box; }}
.fs-wafer-grid {{ width:100%; height:100%; border-radius:50%; overflow:hidden; display:grid;
                 grid-template-columns:repeat({CONFIG['wafer_grid']}, minmax(0,1fr)); gap:3px; }}
.fs-die {{ border-radius:2px; }}
.fs-die.on {{ animation: fs-pop .45s ease-out; }}
@keyframes fs-pop {{ from {{ opacity:.2; transform:scale(.6); }} to {{ opacity:1; transform:scale(1); }} }}
@media (prefers-reduced-motion: reduce) {{ .fs-die.on {{ animation:none; }} }}
.fs-legend {{ display:flex; justify-content:center; gap:20px; font-size:13px; color:var(--muted); margin-top:16px; }}
.fs-sw {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:middle; }}
.fs-center {{ text-align:center; }}
.fs-footer {{ display:flex; justify-content:space-between; border-top:1px solid var(--line); padding:24px 0 8px;
             font-size:13px; color:var(--muted); }}
.fs-footer a {{ color: var(--muted) !important; }}
/* phones: smaller type, stacked stats, hide section links */
@media (max-width: 640px) {{
  .fs-h1 {{ font-size:46px; }} .fs-h2 {{ font-size:36px; }}
  .fs-links {{ display:none; }}
  /* keep the nav on one row: brand left, buttons right */
  .st-key-nav [data-testid="stHorizontalBlock"] {{ flex-wrap:nowrap !important; gap:8px; }}
  .st-key-nav [data-testid="stColumn"] {{ min-width:0 !important; width:auto !important; flex:0 1 auto !important; }}
  .st-key-nav [data-testid="stColumn"]:first-child {{ flex:1 1 auto !important; }}
  .st-key-nav [data-testid="stColumn"]:nth-child(2) {{ display:none; }}
  .st-key-nav {{ padding:6px 8px 6px 16px; }}
  .st-key-cta_nav a {{ padding:0.45rem 0.9rem; }}
  .fs-card {{ padding:24px 20px; }}
  .fs-stats {{ grid-template-columns:1fr; padding:0 20px; }}
  .fs-stats > div + div {{ border-left:none; border-top:1px solid var(--line); padding-left:0; }}
}}
</style>"""


def wafer_html(flags: list[bool], die: str, accent: str) -> str:
    cells = "".join(f'<div class="fs-die{" on" if f else ""}" style="background:{accent if f else die}"></div>'
                    for f in flags)
    return (f'<div class="fs-wafer glass" role="img" aria-label="Illustrative wafer map, '
            f'{sum(flags)} of {len(flags)} dies flagged"><div class="fs-wafer-grid">{cells}</div></div>'
            f'<div class="fs-legend"><span><span class="fs-sw" style="background:{die}"></span>Predicted pass</span>'
            f'<span><span class="fs-sw" style="background:{accent}"></span>Flagged</span>'
            f'<span style="font-style:italic">Illustrative wafer map</span></div>')


# =============================================================================
# UI rendering - one function per section
# =============================================================================
def render_nav() -> None:
    with st.container(key="nav"):
        c1, c2, c3, c4 = st.columns([2.2, 4, 0.6, 1.6], vertical_alignment="center")
        c1.markdown(":material/blur_circular: **FabSentinel**")
        c2.html('<nav class="fs-links" aria-label="Sections"><a href="#how">How it works</a>'
                '<a href="#stack">Stack</a><a href="#team">Team</a></nav>')
        dark = st.session_state.dark
        c3.button("", icon=":material/light_mode:" if dark else ":material/dark_mode:", key="darkbtn",
                  help="Switch to light mode" if dark else "Switch to dark mode", on_click=_toggle_dark)
        with c4.container(key="cta_nav"):
            st.page_link("views/overview.py", label="Open live app")


def _toggle_dark() -> None:
    st.session_state.dark = not st.session_state.dark


def render_hero(t: dict, accent: str) -> None:
    left, right = st.columns([1.1, 1], gap="large", vertical_alignment="center")
    with left:
        st.html('<div style="height:48px"></div><div class="fs-eyebrow">Semiconductor yield · Explainable ML</div>'
                '<div class="fs-h1">Catch the failing wafer <em>before</em> it leaves the fab.</div>'
                '<p class="fs-lead">FabSentinel predicts pass/fail yield from 590 process sensors and tells you which '
                'sensors drove each prediction — so engineers can go straight to the root cause.</p>')
        b1, b2 = st.columns([1, 1.2])
        with b1.container(key="cta_hero"):
            st.page_link("views/overview.py", label="Try the dashboard", icon=":material/arrow_forward:")
        b2.link_button("View on GitHub", CONFIG["github_url"])
    with right:
        wafer_panel(t["die"], accent)


@st.fragment(run_every=CONFIG["scan_interval_s"])
def wafer_panel(die: str, accent: str) -> None:
    """Re-runs on its own every scan_interval_s (only this block, not the whole page)."""
    if st.session_state.live:
        st.session_state.seed += 1
    st.html(wafer_html(make_wafer(st.session_state.seed), die, accent))
    c1, c2 = st.columns(2, vertical_alignment="center")
    if c1.button("Scan another lot", icon=":material/refresh:", width="stretch"):
        st.session_state.seed += 1000
    c2.toggle("Live scan", key="live", help="Turn off to freeze the map")


def render_stats() -> None:
    stats, live = load_stats()
    if not live:
        st.warning("Live dataset metrics are unavailable, so reference SECOM figures are shown. "
                   "Run `python src/train.py` to regenerate artifacts/metrics.json.", icon=":material/info:")
    st.html(f'<div class="glass fs-card fs-stats" style="padding-top:0;padding-bottom:0">'
            f'<div><div class="fs-num">{stats["runs"]:,}</div><div class="fs-small">production runs in SECOM</div></div>'
            f'<div><div class="fs-num">{stats["sensors"]:,}</div><div class="fs-small">sensor signals per run</div></div>'
            f'<div><div class="fs-num" style="color:var(--accent)">{stats["failures"]:,}</div>'
            f'<div class="fs-small">failures — a heavily imbalanced problem</div></div></div>')


def render_how() -> None:
    st.html('<div id="how" style="height:72px"></div><div class="fs-eyebrow">How it works</div>'
            '<div class="fs-h2">From raw sensors to a root cause, in four steps.</div>')
    names = [s["name"] for s in STEPS]
    pick = st.segmented_control("Pipeline step", names, default=names[0], key="step",
                                label_visibility="collapsed", width="stretch")
    step = next((s for s in STEPS if s["name"] == pick), STEPS[0])   # deselected -> first step
    n = STEPS.index(step) + 1
    st.html(f'<div class="glass fs-card" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:32px">'
            f'<div><div class="fs-mono">0{n} · {step["tool"]}</div>'
            f'<div class="fs-h2" style="font-size:36px">{step["title"]}</div></div>'
            f'<p class="fs-lead" style="max-width:none;margin:0">{step["body"]}</p></div>')


def render_stack() -> None:
    rows = "".join(f'<div class="fs-row"><b>{k}</b><span class="fs-small" style="margin:0">{v}</span></div>'
                   for k, v in STACK)
    st.html(f'<div id="stack" style="height:72px"></div>'
            f'<div class="glass fs-card" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:48px;padding:64px 48px">'
            f'<div><div class="fs-eyebrow">Stack</div><div class="fs-h2">Small, honest tools. No black boxes.</div>'
            f'<p class="fs-lead" style="margin-top:16px">Every prediction ships with its reasons. The four-tab Streamlit '
            f'dashboard lets you explore the data, run predictions and read the explanations in one place.</p></div>'
            f'<div>{rows}</div></div>')


def render_cta() -> None:
    st.html('<div id="team" style="height:96px"></div><div class="fs-center">'
            '<div class="fs-h2" style="font-size:60px">See which sensor is <em>really</em> costing you yield.</div></div>')
    _, mid, _ = st.columns([2, 1, 2])
    with mid.container(key="cta_end"):
        st.page_link("views/overview.py", label="Open FabSentinel")
    st.html(f'<p class="fs-small fs-center" style="margin-top:24px">{CONFIG["team"]}</p>'
            f'<div class="fs-footer"><span>FabSentinel</span><a href="{CONFIG["github_url"]}">GitHub ↗</a></div>')


# =============================================================================
# Page
# =============================================================================
def main() -> None:
    st.session_state.setdefault("dark", False)       # session state: survives reruns, per browser tab
    st.session_state.setdefault("seed", 7)
    st.session_state.setdefault("live", True)
    theme = CONFIG["dark"] if st.session_state.dark else CONFIG["light"]
    st.html(css(theme, CONFIG["accent"]))
    render_nav()
    render_hero(theme, CONFIG["accent"])
    st.html('<div style="height:40px"></div>')
    render_stats()
    render_how()
    render_stack()
    render_cta()


if __name__ == "__main__":   # Streamlit runs pages as __main__; tests can import the helpers
    main()