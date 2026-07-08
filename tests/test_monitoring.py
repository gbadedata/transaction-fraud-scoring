"""Drift monitoring tests: no drift -> ~0, real shift -> clearly positive."""

import numpy as np

from fraud import monitoring


def test_psi_is_near_zero_for_same_distribution():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 10_000)
    y = rng.normal(0, 1, 10_000)
    assert monitoring.population_stability_index(x, y) < 0.05


def test_psi_flags_a_shifted_distribution():
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, 10_000)
    shifted = rng.normal(1.5, 1, 10_000)
    assert monitoring.population_stability_index(ref, shifted) > 0.25


def test_ks_zero_for_identical_and_high_for_disjoint():
    a = np.linspace(0, 1, 1000)
    assert monitoring.ks_statistic(a, a) == 0.0
    b = np.linspace(5, 6, 1000)
    assert monitoring.ks_statistic(a, b) > 0.99


def test_feature_drift_report_sorted_worst_first():
    import pandas as pd
    rng = np.random.default_rng(3)
    ref = pd.DataFrame({"stable": rng.normal(0, 1, 5000),
                        "drifting": rng.normal(0, 1, 5000)})
    act = pd.DataFrame({"stable": rng.normal(0, 1, 5000),
                        "drifting": rng.normal(2, 1, 5000)})
    rep = monitoring.feature_drift_report(ref, act, ["stable", "drifting"])
    assert rep.iloc[0]["feature"] == "drifting"      # worst drift first
    assert rep.iloc[0]["psi"] > rep.iloc[1]["psi"]
