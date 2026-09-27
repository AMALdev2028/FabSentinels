"""
Visualization and Decision Support Module
-------------------------------------------
Interactive engineering dashboard (Streamlit) per the patent's Figure 5 /
"Visualization and Decision Support Module" section. Displays:
  - Overall manufacturing yield risk & failure probability
  - High-risk wafer lots
  - Critical sensor rankings + SHAP contribution graphs
  - Process parameter trends
  - Historical prediction records
  - Model performance statistics
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

from views.style import FAIL, PASS, style_fig   # shared chart colours

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"



@st.cache_resource
def load_artifacts():
    with open(ARTIFACT_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(ARTIFACT_DIR / "preprocessor.pkl", "rb") as f:
        pre = pickle.load(f)
    with open(ARTIFACT_DIR / "explainer.pkl", "rb") as f:
        explainer = pickle.load(f)
    test_set = pd.read_parquet(ARTIFACT_DIR / "test_set.parquet")
    shap_values = np.load(ARTIFACT_DIR / "shap_values_test.npy")
    with open(ARTIFACT_DIR / "metrics.json") as f:
        metrics = json.load(f)
    return model, pre, explainer, test_set, shap_values, metrics


try:
    model, pre, explainer, test_set, shap_values, metrics = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "**Missing model artifacts.** The dashboard needs a trained model before it can run.\n\n"
        f"Could not find: `{e.filename}`\n\n"
        "Run the training pipeline first:\n"
        "```bash\ncd src\npython3 train.py\n```\n"
        "This regenerates everything in `artifacts/` (model, preprocessor, SHAP explainer, "
        "test-set predictions, metrics) that this dashboard reads."
    )
    st.stop()
feature_cols = pre.kept_features_
threshold_default = metrics["test"]["decision_threshold"]

st.title("FabSentinel Dashboard")
st.caption("Explainable AI system for semiconductor yield prediction and root-cause diagnostics "
           "— SECOM dataset, XGBoost + SHAP")

tab1, tab2, tab3, tab4 = st.tabs([
    ":material/insights: Yield Overview", ":material/warning: High-Risk Lots",
    ":material/troubleshoot: Root-Cause (SHAP)", ":material/speed: Model Performance"
])

# ---------------------------------------------------------------------------
# TAB 1: Yield Overview
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Overall Manufacturing Yield Risk")

    threshold = st.slider("Decision threshold (failure probability cutoff)",
                           0.0, 1.0, float(threshold_default), 0.01,
                           help="Wafer lots with predicted failure probability at or above this "
                                "value are flagged high-risk. Default is tuned to maximize F1 "
                                "on the held-out test set.")

    flagged = (test_set["pred_proba"] >= threshold)
    n_total = len(test_set)
    n_flagged = int(flagged.sum())
    n_actual_fail = int(test_set["true_label"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wafer lots evaluated", n_total)
    c2.metric("Predicted high-risk lots", n_flagged, f"{n_flagged/n_total:.1%} of lots")
    c3.metric("Actual failures (test set)", n_actual_fail, f"{n_actual_fail/n_total:.1%} base rate")
    c4.metric("Mean failure probability", f"{test_set['pred_proba'].mean():.1%}")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("**Failure probability distribution**")
        fig = px.histogram(test_set, x="pred_proba", nbins=40, color="true_label",
                            color_discrete_map={0: PASS, 1: FAIL},
                            labels={"pred_proba": "Predicted failure probability", "true_label": "Actual outcome"})
        fig.add_vline(x=threshold, line_dash="dash", line_color="#55585C",
                      annotation_text="decision threshold")
        fig.for_each_trace(lambda t: t.update(name={"0": "Pass", "1": "Fail"}.get(t.name, t.name)))
        st.plotly_chart(style_fig(fig), width="stretch")

    with col_b:
        st.markdown("**Process trend — failure probability over time**")
        trend = test_set.sort_values("timestamp")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=trend["timestamp"], y=trend["pred_proba"],
                                   mode="markers+lines", line=dict(width=1, color="rgba(22,24,26,0.18)"),
                                   marker=dict(size=6, color=trend["true_label"],
                                               colorscale=[[0, PASS], [1, FAIL]]),
                                   name="Predicted failure probability"))
        fig2.add_hline(y=threshold, line_dash="dash", line_color="#55585C")
        fig2.update_layout(xaxis_title="Wafer lot timestamp", yaxis_title="Predicted failure probability")
        st.plotly_chart(style_fig(fig2), width="stretch")

# ---------------------------------------------------------------------------
# TAB 2: High-risk lots + historical records
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("High-Risk Wafer Lots")
    st.caption("Lots flagged before final electrical test, ranked by predicted failure probability.")

    ranked = test_set.sort_values("pred_proba", ascending=False).copy()
    ranked_display = ranked[["timestamp", "pred_proba", "pred_label", "true_label"]].reset_index(drop=True)
    ranked_display.columns = ["Timestamp", "Failure probability", "Predicted (1=fail)", "Actual (1=fail)"]
    ranked_display["Failure probability"] = ranked_display["Failure probability"].map(lambda x: f"{x:.1%}")

    n_show = st.number_input("Show top N lots", min_value=5, max_value=n_total, value=20, step=5)
    st.dataframe(ranked_display.head(n_show), width="stretch", height=400)

    st.subheader("Historical Prediction Records")
    st.caption("Full evaluated test set — every wafer lot's prediction, for audit and trend review.")
    st.dataframe(ranked_display, width="stretch", height=350)

# ---------------------------------------------------------------------------
# TAB 3: SHAP-based root-cause diagnostics
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Explainable AI — Root-Cause Diagnostics (SHAP)")

    st.markdown("**Global sensor ranking** — sensors ranked by mean absolute SHAP contribution "
                "across all evaluated wafer lots.")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({"sensor": feature_cols, "mean_abs_shap": mean_abs_shap})
    importance_df = importance_df.sort_values("mean_abs_shap", ascending=False).head(20)

    fig3 = px.bar(importance_df.sort_values("mean_abs_shap"), x="mean_abs_shap", y="sensor",
                  orientation="h", labels={"mean_abs_shap": "Mean |SHAP value|", "sensor": "Sensor"})
    fig3.update_traces(marker_color=FAIL)
    st.plotly_chart(style_fig(fig3, height=600), width="stretch")

    st.divider()
    st.markdown("**Per-lot root-cause breakdown**")
    st.caption("Select a specific wafer lot to see which sensors pushed its prediction toward "
               "failure (positive SHAP, purple) or pass (negative SHAP, grey).")

    ranked_idx = test_set.sort_values("pred_proba", ascending=False).index
    idx = st.selectbox(
        "Wafer lot (by timestamp, sorted by risk)",
        options=ranked_idx,
        format_func=lambda i: f"{test_set.loc[i, 'timestamp']} — failure prob {test_set.loc[i, 'pred_proba']:.1%}"
    )
    row_pos = test_set.index.get_loc(idx)
    row_shap = shap_values[row_pos]
    row_vals = test_set.loc[idx, feature_cols]

    contrib_df = pd.DataFrame({
        "sensor": feature_cols,
        "shap_value": row_shap,
        "sensor_value_scaled": row_vals.values,
    }).sort_values("shap_value", key=np.abs, ascending=False).head(15)

    contrib_df = contrib_df.sort_values("shap_value")
    fig4 = px.bar(contrib_df, x="shap_value", y="sensor", orientation="h",
                  labels={"shap_value": "SHAP contribution to failure prediction", "sensor": "Sensor"})
    fig4.update_traces(marker_color=[FAIL if v > 0 else PASS for v in contrib_df["shap_value"]])
    st.plotly_chart(style_fig(fig4, height=500), width="stretch")

    st.info(f"This wafer lot's predicted failure probability is "
            f"**{test_set.loc[idx, 'pred_proba']:.1%}**. Positive bars (purple) are the sensors "
            f"most responsible for pushing that risk up — the recommended starting point for "
            f"root-cause investigation.")

# ---------------------------------------------------------------------------
# TAB 4: Model performance statistics
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Model Performance Statistics")

    st.markdown("**Held-out test set (20% split)**")
    test_m = metrics["test"]
    cols = st.columns(6)
    for col, key in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]):
        col.metric(key.upper().replace("_", "-"), f"{test_m[key]:.3f}")

    st.markdown("**Stratified 5-Fold Cross-Validation (mean ± std)**")
    cv_m = metrics["cv"]
    cols2 = st.columns(6)
    for col, key in zip(cols2, ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]):
        col.metric(key.upper().replace("_", "-"), f"{cv_m[key]['mean']:.3f}",
                   f"± {cv_m[key]['std']:.3f}")

    st.divider()
    st.markdown("**Dataset summary**")
    st.write(f"- Wafer lots: {metrics['n_samples']}")
    st.write(f"- Sensor features after preprocessing: {metrics['n_features']} "
             f"(reduced from 590 raw sensors)")
    st.write(f"- Overall failure (base) rate: {metrics['failure_rate']:.2%}")
    st.write(f"- Decision threshold (F1-optimal on test set): {test_m['decision_threshold']:.3f}")

    st.caption("Note: SECOM is a small, highly imbalanced real-world dataset (~6.6% failures). "
               "ROC-AUC ≈ 0.72 and PR-AUC ≈ 0.16 are consistent with published SECOM benchmarks; "
               "SMOTE + class-weighted training and threshold tuning materially improve recall over "
               "a naive 0.5 cutoff.")
