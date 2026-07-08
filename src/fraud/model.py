"""Supervised scorer: gradient boosting with calibrated probabilities.

Uses scikit-learn's HistGradientBoostingClassifier so the repo installs and runs
fast anywhere. For production, LightGBM or XGBoost drop in behind the same
interface. Probabilities are calibrated (isotonic) because a fraud *score* is only
useful if it means what it says: a 0.9 should be fraud ~90% of the time so that
expected-loss ranking and cost-based thresholds are valid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

from fraud.features import FEATURE_COLS


def _matrix(df: pd.DataFrame, feature_cols: list[str] | None = None) -> np.ndarray:
    cols = feature_cols if feature_cols is not None else FEATURE_COLS
    return df[cols].to_numpy(dtype=float)


def train_scorer(train_df: pd.DataFrame, feature_cols: list[str] | None = None,
                 random_state: int = 0) -> CalibratedClassifierCV:
    """Train a calibrated fraud-probability model on the training slice."""
    base = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.08,
        max_iter=250,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=random_state,
    )
    # Internal stratified CV keeps at least some positives in every fold and
    # avoids the version-churn around prefit calibration.
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(_matrix(train_df, feature_cols), train_df["is_fraud"].to_numpy(int))
    # Remember which columns were used so score() stays consistent.
    model.feature_cols_ = feature_cols if feature_cols is not None else FEATURE_COLS
    return model


def score(model: CalibratedClassifierCV, df: pd.DataFrame) -> np.ndarray:
    """Return calibrated fraud probabilities for each transaction."""
    return model.predict_proba(_matrix(df, getattr(model, "feature_cols_", None)))[:, 1]
