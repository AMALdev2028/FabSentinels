# FabSentinel — AI-Driven Semiconductor Yield Prediction & Root-Cause Diagnostics

[[Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fabsentinels.streamlit.app/)

**Live app: [fabsentinels.streamlit.app](https://fabsentinels.streamlit.app/)**

A working implementation of the system described in the patent specification
*"AI-Driven System for Semiconductor Yield Prediction and Root-Cause Diagnostics
for Semiconductor Fabrication."* Built on the SECOM dataset (the preferred
embodiment named in the spec).

## How the code maps to the patent's modules

| Patent module | File |
|---|---|
| Data Acquisition Module | `src/data_acquisition.py` |
| Data Preprocessing Module | `src/preprocessing.py` |
| Feature Engineering (feature selection, part of preprocessing) | `src/preprocessing.py` |
| Machine Learning Prediction Module + Model Validation | `src/train.py` |
| Explainable AI Module (SHAP) | `src/train.py` (SHAP computation) + `src/views/overview.py` (display) |
| Visualization and Decision Support Module | `src/views/overview.py` (+ `src/views/landing.py` home page) |

## Try it now

No setup needed — the trained model and dashboard are already deployed:
**[https://fabsentinels.streamlit.app/](https://fabsentinels.streamlit.app/)**

The sections below cover running it locally for development or retraining.

## Setup

```bash
pip install -r requirements.txt
```

Dependency versions are pinned (not just named) in `requirements.txt`, and the Python
version is pinned in `.python-version` (3.11). This is so a redeploy six months from
now installs the exact same library versions the model was trained and pickled with,
instead of silently picking up a newer XGBoost/SHAP release that can't load an older
pickle.

Real SECOM data is already included in `data/secom.data` and `data/secom_labels.data`
(1567 wafer lots, 590 raw sensors, from the public UCI SECOM dataset).

## Run

1. **Train the model** (runs the full pipeline: load → preprocess → cross-validate →
   train → SHAP → save artifacts):
```bash
   cd src
   python3 train.py
```
   This writes everything the dashboard needs into `artifacts/`.

2. **Launch the app** (from the `src/` folder you're already in):
```bash
   streamlit run dashboard.py
```
   Opens at `http://localhost:8501` on the **Home** page. Use the top menu to reach
   **Dashboard** (four tabs: Yield Overview, High-Risk Lots, Root-Cause (SHAP) and
   Model Performance, matching the patent's Figure 5 engineering dashboard) and
   **Yield Planner**.

## App structure

`src/dashboard.py` is the entry point. Streamlit Cloud runs it, and all it does is
route between three pages:

| Page | File | What it is |
|---|---|---|
| Home | `src/views/landing.py` | Landing page: glass navigation bar, hero with a live (illustrative) wafer map, dataset stats, an interactive "How it works", the stack, and team. Has a light/dark toggle and works on phones. |
| Dashboard | `src/views/overview.py` | The ML dashboard (XGBoost + SHAP) |
| Yield Planner | `src/views/yield_planner.py` | Physics-based yield / feasibility engine (see below) |

**Customising the landing page:** colours, text, the wafer animation speed and links
are all in the `CONFIG` block at the top of `src/views/landing.py`. The app-wide
theme (background, accent colour, fonts) is in `.streamlit/config.toml`. The landing
stats are read from `artifacts/metrics.json`. If that file is missing, the page shows
the public SECOM figures and a notice instead of crashing.

## What the pipeline does

1. **Data Acquisition** — loads the 590 raw sensor channels and pass/fail labels.
2. **Preprocessing** — drops sensors with >55% missing values, median-imputes the
   rest, drops near-zero-variance sensors, drops one of each highly-correlated
   (>0.95) sensor pair, and standardizes the remaining ~270 features.
3. **Class imbalance correction** — SMOTE oversampling on the training folds only
   (never on validation/test data, to avoid leakage).
4. **Model** — XGBoost classifier, 80:20 train/test split, validated with
   Stratified 5-Fold Cross-Validation.
5. **Explainability** — SHAP `TreeExplainer` computes per-sensor contribution to
   each wafer lot's failure prediction.
6. **Dashboard** — Streamlit app surfacing yield risk, high-risk lot rankings,
   SHAP root-cause breakdowns (both global and per-lot), and model performance
   stats, with an adjustable decision threshold.

## Yield Planner (FabYield)

A second page in the dashboard, **Yield Planner**, is a different kind of tool.
FabSentinel *learns* failure risk from sensor data. FabYield *computes* yield from
physics: defect density, die area, the process flow (7nm / 5nm / 3nm GAA) and
fabrication time. It returns an ACHIEVABLE / MARGINAL / INFEASIBLE verdict for a
monthly production target, with a yield-loss breakdown, cycle time, sensitivity
analysis and ranked improvements.

Code, tests and full documentation are in [`fabyield/`](fabyield/README.md). It also
runs from the command line:

```bash
cd fabyield
python -m fabyield examples/n5_800mm2_50k.json
pytest                                   # 75 tests
```

Its numbers are estimates from public data plus stated assumptions (the 3nm values
and the wafer-start capacity are assumptions). See `fabyield/docs/TECHNICAL.md`.

## Development

A small smoke-test suite covers the preprocessing and training pipeline:

```bash
pip install pytest
pytest tests/ -v
```

Run these after touching anything in `src/` and before retraining/redeploying —
they catch the common failure mode of a preprocessing change silently breaking
what the dashboard expects to load.

## Honest performance note

SECOM is a small (1567 rows), highly imbalanced (~6.6% failure rate) real-world
dataset — this is a known-hard benchmark, not a toy problem. With the F1-optimal
threshold: **ROC-AUC ≈ 0.70, PR-AUC ≈ 0.16, recall ≈ 43%, precision ≈ 20%**. These
numbers are in line with published SECOM results elsewhere. The dashboard's
threshold slider lets you trade precision for recall live rather than being
locked into one operating point.

(`train.py` fixes `random_state=42` everywhere, so re-running it inside the exact
pinned environment in `requirements.txt` reproduces these numbers exactly. Re-run
it with different library versions, though, and expect small drift — same seed,
different internal algorithm implementation. The checked-in `artifacts/metrics.json`
is the source of truth; the dashboard's "Model Performance" tab always shows it live.)
