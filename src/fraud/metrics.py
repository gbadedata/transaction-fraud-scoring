"""Evaluation metrics that match how a fraud team actually operates.

Accuracy is meaningless at <1% base rate. The metrics here are the ones a fraud
analyst reasons with:
  * precision@k        -- k = investigator capacity (alerts they can clear/day)
  * recall@k           -- share of fraud caught inside that capacity
  * value_weighted_recall -- share of fraud *dollars* caught (one big fraud
                             matters more than many tiny ones)
  * pr_auc             -- threshold-free ranking quality under imbalance
  * operating cost     -- $ of missed fraud + review labour + customer friction,
                          which is what actually sets the threshold
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score


def _top_k_mask(scores, k):
    scores = np.asarray(scores, dtype=float)
    k = int(min(k, len(scores)))
    if k <= 0:
        return np.zeros(len(scores), dtype=bool)
    cutoff_idx = np.argsort(-scores, kind="stable")[:k]
    mask = np.zeros(len(scores), dtype=bool)
    mask[cutoff_idx] = True
    return mask


def precision_at_k(y_true, scores, k) -> float:
    """Precision within the top-k highest-scoring alerts (k = review capacity)."""
    y = np.asarray(y_true).astype(int)
    mask = _top_k_mask(scores, k)
    n = int(mask.sum())
    return float((y[mask] == 1).sum() / n) if n else float("nan")


def recall_at_k(y_true, scores, k) -> float:
    """Share of all fraud cases captured within the top-k alerts."""
    y = np.asarray(y_true).astype(int)
    total = int(y.sum())
    if total == 0:
        return float("nan")
    mask = _top_k_mask(scores, k)
    return float((y[mask] == 1).sum() / total)


def value_weighted_recall_at_k(y_true, scores, amount, k) -> float:
    """Share of fraudulent *dollars* captured within the top-k alerts."""
    y = np.asarray(y_true).astype(int)
    amt = np.asarray(amount, dtype=float)
    total = float(amt[y == 1].sum())
    if total <= 0:
        return float("nan")
    mask = _top_k_mask(scores, k)
    return float(amt[mask & (y == 1)].sum() / total)


def pr_auc(y_true, scores) -> float:
    """Area under the precision-recall curve (average precision)."""
    return float(average_precision_score(np.asarray(y_true).astype(int), scores))


@dataclass
class Operating:
    threshold: float
    alerts: int
    precision: float
    recall: float
    caught_value: float
    missed_value: float
    cost: float


def operating_cost(y_true, scores, amount, threshold,
                   review_cost: float = 4.0, friction_cost: float = 12.0) -> Operating:
    """Total operating cost at a score threshold.

    cost = missed fraud $ (assumed lost)                     [FN]
         + review_cost per alert (analyst labour)            [all alerts]
         + friction_cost per false alert (blocked good cust) [FP]

    Assumes a flagged fraud is prevented. Tune the two unit costs to your economics.
    """
    y = np.asarray(y_true).astype(int)
    amt = np.asarray(amount, dtype=float)
    flagged = np.asarray(scores, dtype=float) >= threshold

    tp = flagged & (y == 1)
    fp = flagged & (y == 0)
    fn = (~flagged) & (y == 1)

    n_alerts = int(flagged.sum())
    caught_value = float(amt[tp].sum())
    missed_value = float(amt[fn].sum())
    cost = missed_value + review_cost * n_alerts + friction_cost * int(fp.sum())

    precision = float(tp.sum() / n_alerts) if n_alerts else float("nan")
    recall = float(tp.sum() / max(int(y.sum()), 1))
    return Operating(float(threshold), n_alerts, precision, recall,
                     caught_value, missed_value, cost)


def best_threshold_by_cost(y_true, scores, amount, n_grid: int = 200,
                           review_cost: float = 4.0, friction_cost: float = 12.0) -> Operating:
    """Scan thresholds and return the minimum-cost operating point."""
    scores = np.asarray(scores, dtype=float)
    grid = np.quantile(scores, np.linspace(0.0, 1.0, n_grid))
    grid = np.unique(np.round(grid, 6))
    best = None
    for t in grid:
        op = operating_cost(y_true, scores, amount, t, review_cost, friction_cost)
        if best is None or op.cost < best.cost:
            best = op
    return best


def threshold_for_capacity(scores, k: int) -> float:
    """The score threshold that yields ~k alerts (fills investigator capacity)."""
    scores = np.asarray(scores, dtype=float)
    k = int(min(max(k, 1), len(scores)))
    return float(np.sort(scores)[-k])
