"""Blend signals into one queue and rank it the way a fraud team should.

Three signals -> one number -> one ranked queue:
  * model probability (supervised, calibrated)
  * rule score        (transparent tripwires)
  * anomaly score     (unsupervised, catches novel patterns rules/model miss)

The queue is ranked by EXPECTED LOSS = P(fraud) x amount, not by probability
alone. Working the highest expected-loss alerts first maximises prevented dollars
under a fixed review budget, which is the core operational point of the project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from fraud.features import FEATURE_COLS


def anomaly_scores(train_df: pd.DataFrame, score_df: pd.DataFrame,
                   feature_cols: list[str] | None = None,
                   random_state: int = 0) -> np.ndarray:
    """Unsupervised novelty score in [0, 1] (higher = more anomalous)."""
    cols = feature_cols if feature_cols is not None else FEATURE_COLS
    iso = IsolationForest(n_estimators=200, contamination="auto",
                          random_state=random_state)
    iso.fit(train_df[cols].to_numpy(np.float32))
    raw = -iso.score_samples(score_df[cols].to_numpy(np.float32))  # higher = odder
    lo, hi = raw.min(), raw.max()
    return (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)


def blend(model_prob, rule_score, anomaly_score=None,
          w_model: float = 0.65, w_rule: float = 0.25, w_anomaly: float = 0.10):
    """Weighted blend of the available signals, renormalised over what's present."""
    model_prob = np.asarray(model_prob, dtype=float)
    rule_score = np.asarray(rule_score, dtype=float)
    parts = [(w_model, model_prob), (w_rule, rule_score)]
    if anomaly_score is not None:
        parts.append((w_anomaly, np.asarray(anomaly_score, dtype=float)))
    total_w = sum(w for w, _ in parts)
    return sum(w * s for w, s in parts) / total_w


def expected_loss(scores, amount) -> np.ndarray:
    """Expected fraud loss per transaction = P(fraud) x amount."""
    return np.asarray(scores, dtype=float) * np.asarray(amount, dtype=float)


def rank_queue(df: pd.DataFrame, scores, reason_codes=None) -> pd.DataFrame:
    """Return an investigator queue ordered by expected loss (highest first)."""
    q = df.copy()
    q["risk_score"] = np.asarray(scores, dtype=float)
    q["expected_loss"] = expected_loss(scores, q["amount"])
    if reason_codes is not None:
        q["reason_codes"] = list(reason_codes)
    return q.sort_values("expected_loss", ascending=False).reset_index(drop=True)


def rule_reason_codes(hits: pd.DataFrame, rule_names) -> list[str]:
    """Human-readable 'why flagged' string per row from the fired rules."""
    fired = hits[list(rule_names)].astype(bool)
    return [", ".join(n for n in rule_names if row[n]) or "model/anomaly only"
            for _, row in fired.iterrows()]
