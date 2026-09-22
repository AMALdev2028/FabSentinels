"""
Smoke tests for the FabSentinel pipeline.

These aren't exhaustive — they exist to catch the failure mode that matters most
for a small student project: "I changed something in preprocessing.py or train.py
and didn't notice it silently broke the dashboard." Run them after any change to
src/, before retraining or redeploying.

Run from the repo root (with the requirements.txt venv active):
    pip install pytest
    pytest tests/ -v
"""
import pickle
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_acquisition import load_raw_data          # noqa: E402
from preprocessing import Preprocessor, balance_classes  # noqa: E402


def test_data_loads_with_expected_shape():
    df = load_raw_data()
    assert df.shape[0] == 1567, "SECOM should have 1567 wafer lots"
    assert set(df["label"].unique()) <= {0, 1}, "labels must be mapped to 0 (pass) / 1 (fail)"


def test_preprocessor_drops_bad_features_and_leaves_no_nans():
    df = load_raw_data()
    X_raw = df.drop(columns=["label", "timestamp"])
    pre = Preprocessor()
    X = pre.fit_transform(X_raw)

    assert X.shape[1] < X_raw.shape[1], "preprocessing should drop some raw sensors"
    assert not X.isna().any().any(), "no NaNs should remain after imputation"
    assert list(X.columns) == pre.kept_features_


def test_preprocessor_transform_reuses_fitted_state():
    """transform() must apply the columns/imputer/scaler learned during fit(),
    not refit itself on whatever it's given — otherwise train/serve would drift."""
    df = load_raw_data()
    X_raw = df.drop(columns=["label", "timestamp"])
    pre = Preprocessor().fit(X_raw)

    X2 = pre.transform(X_raw.head(10))
    assert list(X2.columns) == pre.kept_features_
    assert X2.shape[0] == 10


def test_smote_balances_classes():
    df = load_raw_data()
    X_raw = df.drop(columns=["label", "timestamp"])
    y = df["label"]
    X = Preprocessor().fit_transform(X_raw)

    X_res, y_res = balance_classes(X, y)
    counts = y_res.value_counts()
    assert counts[0] == counts[1], "SMOTE should equalize pass/fail counts in the training data"


def test_saved_artifacts_load_and_predict():
    """If artifacts/ exists (train.py has been run), the model should load and
    produce a valid probability for real held-out rows — this is the exact
    load path the dashboard depends on."""
    artifact_dir = ROOT / "artifacts"
    model_path = artifact_dir / "model.pkl"
    if not model_path.exists():
        pytest.skip("artifacts/model.pkl not found — run `python3 src/train.py` first")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(artifact_dir / "preprocessor.pkl", "rb") as f:
        pre = pickle.load(f)
    test_set = pd.read_parquet(artifact_dir / "test_set.parquet")

    proba = model.predict_proba(test_set[pre.kept_features_].head(5))[:, 1]
    assert proba.shape == (5,)
    assert ((proba >= 0) & (proba <= 1)).all(), "predicted probabilities must be in [0, 1]"
