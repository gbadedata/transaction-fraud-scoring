"""Metric tests on tiny, hand-checkable inputs."""

import numpy as np

from fraud import metrics


def test_precision_and_recall_at_k():
    y = np.array([1, 0, 1, 0, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])   # top-3 -> {1,0,1}
    assert metrics.precision_at_k(y, scores, 3) == 2 / 3
    assert metrics.recall_at_k(y, scores, 3) == 2 / 3   # 2 of 3 frauds caught


def test_value_weighted_recall_prioritises_big_fraud():
    y = np.array([1, 1, 0])
    amount = np.array([1000.0, 10.0, 50.0])
    # score ranks the $1000 fraud first -> top-1 captures ~99% of fraud dollars
    scores = np.array([0.9, 0.1, 0.5])
    vwr = metrics.value_weighted_recall_at_k(y, scores, amount, 1)
    assert abs(vwr - (1000.0 / 1010.0)) < 1e-9


def test_pr_auc_perfect_ranking_is_one():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert metrics.pr_auc(y, scores) == 1.0


def test_operating_cost_accounts_for_missed_fraud_and_friction():
    y = np.array([1, 0, 1])
    amount = np.array([100.0, 100.0, 200.0])
    scores = np.array([0.9, 0.9, 0.1])   # flags txn0 (TP) and txn1 (FP), misses txn2 (FN)
    op = metrics.operating_cost(y, scores, amount, threshold=0.5,
                                review_cost=5.0, friction_cost=20.0)
    # cost = missed 200 + 2 alerts*5 + 1 FP*20 = 230
    assert op.cost == 230.0
    assert op.alerts == 2
    assert op.caught_value == 100.0
    assert op.missed_value == 200.0


def test_threshold_for_capacity_returns_kth_highest():
    scores = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
    assert metrics.threshold_for_capacity(scores, 2) == 0.7   # 2nd highest


def test_best_threshold_minimises_cost():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.05).astype(int)
    amount = rng.uniform(10, 500, 500)
    scores = np.clip(y * 0.6 + rng.random(500) * 0.4, 0, 1)
    op = metrics.best_threshold_by_cost(y, scores, amount)
    # the chosen point should beat both extremes (flag-all and flag-none)
    flag_all = metrics.operating_cost(y, scores, amount, 0.0)
    flag_none = metrics.operating_cost(y, scores, amount, 1.01)
    assert op.cost <= flag_all.cost
    assert op.cost <= flag_none.cost
