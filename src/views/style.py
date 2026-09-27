"""Shared look for every page: fonts, colours, glass cards, page width, chart colours.

src/dashboard.py applies base_css() once for whichever page is showing, so Home,
Dashboard and Yield Planner always share one style. Change colours here (and the
matching primaryColor in .streamlit/config.toml) to re-theme the whole app.
"""
from __future__ import annotations

ACCENT = "#6D28D9"                     # purple: primary colour, "fail"/flagged
TOKENS = {
    "light": {"bg": "#F6F5F1", "ink": "#16181A", "muted": "#55585C", "line": "rgba(22,24,26,0.10)",
              "glass": "rgba(255,255,255,0.55)", "glass_border": "rgba(255,255,255,0.75)",
              "glass_hi": "rgba(255,255,255,0.95)", "die": "#D9D6CE", "blob_alpha": 0.30},
    "dark":  {"bg": "#121314", "ink": "#EDEBE6", "muted": "#A3A19B", "line": "rgba(255,255,255,0.10)",
              "glass": "rgba(32,33,36,0.55)", "glass_border": "rgba(255,255,255,0.10)",
              "glass_hi": "rgba(255,255,255,0.08)", "die": "#3A3C3F", "blob_alpha": 0.24},
}
BLOB_2, BLOB_3 = "#3B82F6", "#8B5CF6"  # background glow colours

# Chart colours (Plotly): pass = warm grey, fail = accent purple, gridlines = faint ink.
PASS, FAIL, GRID = "#A8A29A", ACCENT, "rgba(22,24,26,0.08)"
FONT = "IBM Plex Sans, sans-serif"


def hex_rgb(h: str) -> str:
    h = h.lstrip("#")
    return ", ".join(str(int(h[i:i + 2], 16)) for i in (0, 2, 4))


def base_css(dark: bool = False) -> str:
    t = TOKENS["dark" if dark else "light"]
    a = t["blob_alpha"]
    return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root {{ --bg:{t['bg']}; --ink:{t['ink']}; --muted:{t['muted']}; --line:{t['line']}; --accent:{ACCENT};
         --glass:{t['glass']}; --glass-border:{t['glass_border']}; --glass-hi:{t['glass_hi']}; }}
.stApp {{
  background:
    radial-gradient(520px circle at 80% 22%, rgba({hex_rgb(ACCENT)}, {a}), transparent 70%),
    radial-gradient(380px circle at 62% 46%, rgba({hex_rgb(BLOB_2)}, {a * 0.8}), transparent 70%),
    radial-gradient(480px circle at 6% 78%, rgba({hex_rgb(BLOB_3)}, {a * 0.7}), transparent 70%),
    var(--bg);
  background-attachment: fixed; color: var(--ink);
}}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ max-width: 1200px; padding-top: 3.5rem; }}
/* glass: landing cards + metric tiles, charts and tables on every page */
.glass, .st-key-nav, [data-testid="stMetric"], [data-testid="stPlotlyChart"],
[data-testid="stVegaLiteChart"], [data-testid="stDataFrame"] {{
  background: var(--glass); -webkit-backdrop-filter: blur(24px) saturate(180%); backdrop-filter: blur(24px) saturate(180%);
  border: 1px solid var(--glass-border); box-shadow: inset 0 1px 0 var(--glass-hi), 0 8px 32px rgba(0,0,0,0.06);
}}
[data-testid="stMetric"] {{ border-radius: 20px; padding: 18px 22px; min-height: 9.6rem; }}
[data-testid="stPlotlyChart"], [data-testid="stVegaLiteChart"] {{ border-radius: 20px; padding: 12px; }}
[data-testid="stDataFrame"] {{ border-radius: 16px; overflow: hidden; }}
[data-testid="stMetricValue"] {{ font-family: 'Instrument Serif', Georgia, serif; font-size: 2.8rem; line-height: 1.1; }}
.stButton button, .stLinkButton a, .stDownloadButton button {{
  border-radius: 26px; background: var(--glass) !important; color: var(--ink) !important;
  border: 1px solid var(--glass-border) !important; -webkit-backdrop-filter: blur(24px); backdrop-filter: blur(24px);
}}
.stButton button p, .stLinkButton a p, .stDownloadButton button p {{ color: var(--ink) !important; }}
@media (max-width: 640px) {{ .block-container {{ padding-top: 3rem; }} [data-testid="stMetricValue"] {{ font-size: 2.2rem; }} }}
</style>"""


def style_fig(fig, height: int | None = None):
    """Apply the app's chart look to a Plotly figure (fonts, transparent ground, soft grid)."""
    fig.update_layout(font=dict(family=FONT, size=13, color=TOKENS["light"]["ink"]),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      colorway=[FAIL, PASS, "#0F766E", "#C2410C"], margin=dict(l=8, r=8, t=24, b=8),
                      legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", x=0, y=1.02, yanchor="bottom"), hoverlabel=dict(font_family=FONT), bargap=0.08)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    if height:
        fig.update_layout(height=height)
    return fig
