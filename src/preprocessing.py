"""
Data Preprocessing Module
--------------------------
Implements the five preprocessing sub-steps named in the patent:
  1. Missing value handling (median imputation)
  2. Removal of redundant features (low variance, excessive missing, high correlation)
  3. Feature selection (variance/correlation based, informative subset)
  4. Data normalization (common numerical range)
  5. Class imbalance correction (SMOTE + class weighting)
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


class Preprocessor:
    def __init__(self, missing_thresh: float = 0.55, corr_thresh: float = 0.95,
                 variance_thresh: float = 1e-6):
        self.missing_thresh = missing_thresh
        self.corr_thresh = corr_thresh
        self.variance_thresh = variance_thresh
        self.kept_features_: list[str] = []
        self.imputer_: SimpleImputer | None = None
        self.scaler_: StandardScaler | None = None

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        # 1. Drop features with too many missing values
        missing_frac = X.isna().mean()
        cols = missing_frac[missing_frac <= self.missing_thresh].index.tolist()
        X = X[cols]

        # 2. Median-impute the rest so downstream variance/correlation stats are valid
        imputer = SimpleImputer(strategy="median")
        X_imp = pd.DataFrame(imputer.fit_transform(X), columns=cols, index=X.index)

        # 3. Drop near-zero-variance (uninformative) sensors
        variances = X_imp.var()
        cols = variances[variances > self.variance_thresh].index.tolist()
        X_imp = X_imp[cols]

        # 4. Drop one of each highly-correlated pair (redundant sensors)
        corr = X_imp.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > self.corr_thresh)]
        cols = [c for c in cols if c not in to_drop]

        self.kept_features_ = cols
        self.imputer_ = SimpleImputer(strategy="median").fit(X[cols])
        self.scaler_ = StandardScaler().fit(self.imputer_.transform(X[cols]))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.kept_features_]
        X_imp = self.imputer_.transform(X)
        X_scaled = self.scaler_.transform(X_imp)
        return pd.DataFrame(X_scaled, columns=self.kept_features_, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


def balance_classes(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    """Class imbalance correction via SMOTE, per patent claim 4."""
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res
