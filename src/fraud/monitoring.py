"""Drift monitoring, because fraud is adversarial and non-stationary.

Fraudsters adapt and models decay. These functions let a scheduled job compare a
recent window against a training/reference window and raise a flag before precision
quietly collapses. Pure-numpy so there's no heavy dependency.

Rules of thumb for PSI:  < 0.10 stable | 0.10-0.25 moderate shift | > 0.25 large.
"""

from __future__ import annotations

import numpy as np


def population_stability_index(reference, actual, bins: int = 10) -> float:
    """PSI between a reference and an actual distribution (quantile-binned)."""
    reference = np.asarray(reference, dtype=float)
    actual = np.asarray(actual, dtype=float)

    edges = np.quantile(reference, np.linspace(0.0, 1.0, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if len(edges) < 3:                       # degenerate/constant reference
        return 0.0

    ref_pct = np.histogram(reference, edges)[0] / len(reference)
    act_pct = np.histogram(actual, edges)[0] / len(actual)

    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    act_pct = np.clip(act_pct, eps, None)
    return float(np.sum((act_pct - ref_pct) * np.log(act_pct / ref_pct)))


def ks_statistic(reference, actual) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (max ECDF gap), numpy-only."""
    reference = np.sort(np.asarray(reference, dtype=float))
    actual = np.sort(np.asarray(actual, dtype=float))
    grid = np.concatenate([reference, actual])
    cdf_ref = np.searchsorted(reference, grid, side="right") / len(reference)
    cdf_act = np.searchsorted(actual, grid, side="right") / len(actual)
    return float(np.max(np.abs(cdf_ref - cdf_act)))


def feature_drift_report(reference_df, actual_df, feature_cols, bins: int = 10):
    """PSI + KS per feature; sorted worst-first so triage is obvious."""
    import pandas as pd

    rows = []
    for col in feature_cols:
        rows.append({
            "feature": col,
            "psi": population_stability_index(reference_df[col], actual_df[col], bins),
            "ks": ks_statistic(reference_df[col], actual_df[col]),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
