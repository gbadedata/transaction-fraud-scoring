"""End-to-end demo: data, features, rules, model, anomaly, ranked queue.

Runs on generated data, so it needs no downloads. It prints the numbers a
fraud-ops review actually looks at: per-rule precision, ranking quality, a
cost-chosen operating point, and what a fixed alert budget buys.

Method note: blend weights and the operating threshold are both selected on the
validation slice and only then measured on the test slice. Tuning either on the
test set would overstate results.

    python run_demo.py
"""

from __future__ import annotations

import numpy as np

from fraud import data, features, metrics, model, monitoring, rules, scoring
from fraud.features import FEATURE_COLS

ALERT_BUDGET = 100       # alerts the team can clear in the test window
REVIEW_COST = 4.0        # analyst cost per alert
FRICTION_COST = 12.0     # cost of blocking a good customer (a false positive)

# Blend weight candidates: (model, rules, anomaly). Selected on validation.
BLEND_CANDIDATES = [
    (1.00, 0.00, 0.00),
    (0.80, 0.20, 0.00),
    (0.70, 0.30, 0.00),
    (0.70, 0.20, 0.10),
    (0.60, 0.25, 0.15),
    (0.50, 0.30, 0.20),
]


def _signals(scorer, train, frame):
    prob = model.score(scorer, frame)
    hits = rules.apply_rules(frame)
    anomaly = scoring.anomaly_scores(train, frame)
    return prob, hits, anomaly


def _select_blend(prob, rule_score, anomaly, y, amount, budget):
    """Pick the weight set that captures the most fraud value within budget."""
    best_w, best_vwr = BLEND_CANDIDATES[0], -1.0
    for w in BLEND_CANDIDATES:
        s = scoring.blend(prob, rule_score, anomaly, *w)
        vwr = metrics.value_weighted_recall_at_k(y, s, amount, budget)
        if vwr > best_vwr:
            best_vwr, best_w = vwr, w
    return best_w, best_vwr


def main() -> None:
    print("Generating synthetic transactions...")
    df = features.add_features(data.generate_transactions(seed=42))
    fraud_value = df.loc[df.is_fraud == 1, "amount"].sum()
    print(f"  {len(df):,} transactions | fraud rate {df['is_fraud'].mean():.3%} "
          f"| fraud value ${fraud_value:,.0f}\n")

    train, valid, test, cuts = data.time_split(df)
    print(f"Time-based split (no leakage): train {len(train):,} | "
          f"valid {len(valid):,} | test {len(test):,}")
    print(f"  cutoffs {cuts[0]:%Y-%m-%d %H:%M} / {cuts[1]:%Y-%m-%d %H:%M}\n")

    print("Training calibrated gradient-boosted scorer...")
    scorer = model.train_scorer(train)
    v_prob, v_hits, v_anom = _signals(scorer, train, valid)
    t_prob, t_hits, t_anom = _signals(scorer, train, test)

    y_valid, amt_valid = valid["is_fraud"].to_numpy(), valid["amount"].to_numpy()
    y_test, amt_test = test["is_fraud"].to_numpy(), test["amount"].to_numpy()

    print("\nPer-rule precision on the test slice:")
    print(rules.rule_precision(t_hits, test["is_fraud"]).to_string(index=False))

    # Select blend weights on validation, then freeze them.
    weights, vwr_valid = _select_blend(v_prob, v_hits["rule_score"].to_numpy(),
                                       v_anom, y_valid, amt_valid, ALERT_BUDGET)
    t_blend = scoring.blend(t_prob, t_hits["rule_score"].to_numpy(), t_anom, *weights)
    print(f"\nBlend weights selected on validation (model, rules, anomaly): {weights}")
    print(f"  validation fraud-value recall @ {ALERT_BUDGET}: {vwr_valid:.3f}")

    print("\nRanking quality on the test slice (higher is better):")
    for name, s in [("model only", t_prob), ("blended", t_blend)]:
        print(f"  {name:12s} PR-AUC {metrics.pr_auc(y_test, s):.3f} "
              f"| precision@{ALERT_BUDGET} {metrics.precision_at_k(y_test, s, ALERT_BUDGET):.3f} "
              f"| recall@{ALERT_BUDGET} {metrics.recall_at_k(y_test, s, ALERT_BUDGET):.3f} "
              f"| value-recall@{ALERT_BUDGET} "
              f"{metrics.value_weighted_recall_at_k(y_test, s, amt_test, ALERT_BUDGET):.3f}")

    # Choose the cost-optimal threshold on validation, then apply it to test.
    v_blend = scoring.blend(v_prob, v_hits["rule_score"].to_numpy(), v_anom, *weights)
    chosen = metrics.best_threshold_by_cost(y_valid, v_blend, amt_valid,
                                            review_cost=REVIEW_COST,
                                            friction_cost=FRICTION_COST)
    op = metrics.operating_cost(y_test, t_blend, amt_test, chosen.threshold,
                                REVIEW_COST, FRICTION_COST)
    do_nothing = amt_test[y_test == 1].sum()
    print("\nOperating point (threshold chosen on validation, measured on test):")
    print(f"  threshold {chosen.threshold:.3f} -> {op.alerts} alerts "
          f"| precision {op.precision:.3f} | recall {op.recall:.3f}")
    print(f"  fraud value caught ${op.caught_value:,.0f} | missed ${op.missed_value:,.0f}")
    print(f"  operating cost ${op.cost:,.0f} vs ${do_nothing:,.0f} if nothing is reviewed")

    thr = metrics.threshold_for_capacity(t_blend, ALERT_BUDGET)
    cap = metrics.operating_cost(y_test, t_blend, amt_test, thr, REVIEW_COST, FRICTION_COST)
    vwr = metrics.value_weighted_recall_at_k(y_test, t_blend, amt_test, ALERT_BUDGET)
    print(f"\nAt a fixed budget of {ALERT_BUDGET} alerts: precision {cap.precision:.3f}, "
          f"catching ${cap.caught_value:,.0f} of fraud ({vwr:.0%} of fraud value).")

    drift = monitoring.feature_drift_report(train, test, FEATURE_COLS)
    print("\nTop feature drift, train vs test window (PSI):")
    print(drift.head(5).to_string(index=False))

    reasons = scoring.rule_reason_codes(t_hits, [r.name for r in rules.RULES])
    queue = scoring.rank_queue(test, t_blend, reasons)
    print("\nTop of the investigator queue (ranked by expected loss):")
    cols = ["transaction_id", "card_id", "amount", "risk_score",
            "expected_loss", "reason_codes", "is_fraud"]
    print(queue[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()
