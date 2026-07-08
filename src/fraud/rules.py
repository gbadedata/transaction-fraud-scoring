"""Rule catalogue.

Transparent, auditable tripwires that a fraud analyst can read, defend, and tune.
Rules are the regulated/interpretable baseline that runs alongside the model; the
final queue blends both (see scoring.py). Each rule carries metadata so the repo
can report *per-rule precision*, the number analysts care about when
deciding whether a rule earns its alert volume.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Rule:
    name: str
    description: str
    severity: int                       # 1 (soft) .. 3 (hard)
    func: Callable[[pd.DataFrame], pd.Series]


def _card_testing(df):        # rapid burst of tiny CNP auths
    return (df["txn_count_1h"] >= 5) & (df["amount"] <= 5.0) & (df["is_cnp"] == 1)


def _high_amount_new_card(df):  # large CNP spend on a young card
    return (df["amount"] >= 200.0) & (df["is_cnp"] == 1) & (df["card_age_days"] <= 60)


def _impossible_travel(df):   # implied speed faster than a commercial flight
    return df["geo_speed_kmh"] >= 900.0


def _velocity_spike(df):      # abnormal daily transaction count
    return df["txn_count_24h"] >= 20


RULES = [
    Rule("card_testing", "5+ sub-$5 CNP auths within 1h", 3, _card_testing),
    Rule("high_amount_new_card", "CNP >= $200 on card <= 60 days old", 2, _high_amount_new_card),
    Rule("impossible_travel", "Geo speed from prior txn >= 900 km/h", 3, _impossible_travel),
    Rule("velocity_spike", "20+ transactions in trailing 24h", 2, _velocity_spike),
]


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame of per-rule boolean hits plus aggregate columns.

    `rule_score` is a soft 0..1 signal (severity-weighted) suitable for blending;
    `rule_any` is the hard 'did anything fire' flag.
    """
    hits = pd.DataFrame(index=df.index)
    max_sev = sum(r.severity for r in RULES)
    weighted = pd.Series(0.0, index=df.index)
    for r in RULES:
        fired = r.func(df).astype(int)
        hits[r.name] = fired
        weighted += fired * r.severity
    hits["rule_any"] = hits[[r.name for r in RULES]].max(axis=1)
    hits["rule_score"] = weighted / max_sev
    return hits


def rule_precision(hits: pd.DataFrame, y_true: pd.Series) -> pd.DataFrame:
    """Precision, alert volume and recall contribution for each rule."""
    y = np.asarray(y_true).astype(int)
    rows = []
    total_fraud = int(y.sum())
    for r in RULES:
        fired = hits[r.name].to_numpy().astype(bool)
        n = int(fired.sum())
        tp = int((fired & (y == 1)).sum())
        rows.append({
            "rule": r.name,
            "alerts": n,
            "true_positives": tp,
            "precision": (tp / n) if n else float("nan"),
            "recall_contrib": (tp / total_fraud) if total_fraud else float("nan"),
        })
    return pd.DataFrame(rows)
