"""End-to-end IEEE-CIS pipeline with entity resolution and ring detection.

Runs on the real dataset if it is present at data/ieee/, otherwise on a
schema-faithful mock so the whole thing works with no download.

    python run_ieee_demo.py

To use the real data, download the competition CSVs (see data/README.md) into
data/ieee/ and re-run; nothing else changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fraud import data, ieee_data, ieee_features, metrics, model, scoring

BUDGET = 200
REVIEW_COST, FRICTION_COST = 4.0, 20.0
BLEND_CANDIDATES = [
    (1.00, 0.00, 0.00), (0.80, 0.20, 0.00), (0.70, 0.20, 0.10),
    (0.60, 0.25, 0.15), (0.50, 0.30, 0.20),
]
IEEE_DIR = Path("data/ieee")


def load():
    tx = IEEE_DIR / "train_transaction.csv"
    idn = IEEE_DIR / "train_identity.csv"
    if tx.exists():
        print(f"Loading real IEEE-CIS from {IEEE_DIR}/ ...")
        return ieee_data.load_ieee(tx, idn if idn.exists() else None)
    print("Real IEEE-CIS not found; using the schema-faithful mock.")
    return ieee_data.load_ieee_frames(*ieee_data.mock_ieee_frames(seed=7))


def select_blend(prob, ring, anom, y, amount, budget):
    best_w, best = BLEND_CANDIDATES[0], -1.0
    for w in BLEND_CANDIDATES:
        s = scoring.blend(prob, ring, anom, *w)
        vwr = metrics.value_weighted_recall_at_k(y, s, amount, budget)
        if vwr > best:
            best, best_w = vwr, w
    return best_w


def main() -> None:
    df = load()
    df, cols = ieee_features.add_ieee_features(df)
    print(f"  {len(df):,} transactions | fraud rate {df['is_fraud'].mean():.3%} "
          f"| {len(cols)} features | fraud value ${df.loc[df.is_fraud==1,'amount'].sum():,.0f}\n")

    train, valid, test, cuts = data.time_split(df)
    print(f"Time-based split: train {len(train):,} | valid {len(valid):,} | test {len(test):,}")
    print(f"  cutoffs {cuts[0]:%Y-%m-%d} / {cuts[1]:%Y-%m-%d}\n")

    print("Training calibrated scorer on IEEE features...")
    scorer = model.train_scorer(train, feature_cols=cols)

    def sig(frame):
        return (model.score(scorer, frame),
                ieee_features.ring_rule_score(frame),
                scoring.anomaly_scores(train, frame, feature_cols=cols))

    v_prob, v_ring, v_anom = sig(valid)
    t_prob, t_ring, t_anom = sig(test)
    yv, av = valid["is_fraud"].to_numpy(), valid["amount"].to_numpy()
    yt, at = test["is_fraud"].to_numpy(), test["amount"].to_numpy()

    weights = select_blend(v_prob, v_ring, v_anom, yv, av, BUDGET)
    t_blend = scoring.blend(t_prob, t_ring, t_anom, *weights)
    v_blend = scoring.blend(v_prob, v_ring, v_anom, *weights)
    print(f"Blend weights selected on validation (model, ring, anomaly): {weights}\n")

    print("Ranking quality on test (higher is better):")
    for name, s in [("model only", t_prob), ("blended", t_blend)]:
        vwr = metrics.value_weighted_recall_at_k(yt, s, at, BUDGET)
        print(f"  {name:12s} PR-AUC {metrics.pr_auc(yt, s):.3f} "
              f"| precision@{BUDGET} {metrics.precision_at_k(yt, s, BUDGET):.3f} "
              f"| recall@{BUDGET} {metrics.recall_at_k(yt, s, BUDGET):.3f} "
              f"| value-recall@{BUDGET} {vwr:.3f}")

    chosen = metrics.best_threshold_by_cost(yv, v_blend, av, review_cost=REVIEW_COST,
                                            friction_cost=FRICTION_COST).threshold
    op = metrics.operating_cost(yt, t_blend, at, chosen, REVIEW_COST, FRICTION_COST)
    do_nothing = at[yt == 1].sum()
    print("\nOperating point (threshold chosen on validation, measured on test):")
    print(f"  threshold {chosen:.3f} -> {op.alerts} alerts | precision {op.precision:.3f} "
          f"| recall {op.recall:.3f}")
    print(f"  fraud value caught ${op.caught_value:,.0f} | missed ${op.missed_value:,.0f} "
          f"| cost ${op.cost:,.0f} vs ${do_nothing:,.0f}\n")

    # --- entity / ring detection ---
    print("Fraud rate by device-sharing degree (cards seen on the device so far):")
    import pandas as pd
    buck = pd.cut(test["device_prior_cards"], [-1, 0, 1, 3, 6, 10_000],
                 labels=["0", "1", "2-3", "4-6", "7+"])
    tab = test.assign(bucket=buck).groupby("bucket", observed=True)["is_fraud"].agg(
        fraud_rate="mean", transactions="size")
    print(tab.round(3).to_string())

    ring = test[test["device_prior_cards"] >= 4]
    ring_val = ring.loc[ring.is_fraud == 1, "amount"].sum()
    print(f"\nRing-linked transactions (device shared by 4+ cards): {len(ring):,} "
          f"| {ring['is_fraud'].mean():.0%} fraud | ${ring_val:,.0f} "
          f"fraud value, caught structurally without the model.\n")

    reasons = ieee_features.ring_reason_codes(test)
    queue = scoring.rank_queue(test, t_blend, reasons)
    print("Top of the investigator queue (ranked by expected loss):")
    show = queue[["TransactionID", "card1", "amount", "risk_score", "expected_loss",
                  "reason_codes", "is_fraud"]].head(10)
    print(show.to_string(index=False))


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()
