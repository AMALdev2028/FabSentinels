# FabSentinel — AI-Driven Semiconductor Yield Prediction & Root-Cause Diagnostics

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
| Explainable AI Module (SHAP) | `src/train.py` (SHAP computation) + `src/dashboard.py` (display) |
| Visualization and Decision Support Module | `src/dashboard.py` |

## Setup

```bash
pip install -r requirements.txt
```

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

2. **Launch the dashboard**:
   ```bash
   streamlit run dashboard.py
   ```
   Opens at `http://localhost:8501` with four tabs: Yield Overview, High-Risk Lots,
   Root-Cause (SHAP), and Model Performance — matching the patent's Figure 5
   engineering dashboard.

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

## Honest performance note

SECOM is a small (1567 rows), highly imbalanced (~6.6% failure rate) real-world
dataset — this is a known-hard benchmark, not a toy problem. With the F1-optimal
threshold: **ROC-AUC ≈ 0.72, PR-AUC ≈ 0.16, recall ≈ 38%, precision ≈ 24%**. These
numbers are in line with published SECOM results elsewhere. The dashboard's
threshold slider lets you trade precision for recall live rather than being
locked into one operating point.
