"""Generate the IEEE-CIS figures used in the README, from the real pipeline.

    python scripts/generate_ieee_figures.py

Runs on data/ieee/ if present, else the schema-faithful mock. Writes PNGs to
docs/img/. Everything derives from one seeded run so the charts stay in sync.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraud import data, ieee_data, ieee_features, metrics, model

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED = "#1f2933", "#9aa5b1"
MODEL_C, RAND_C, RING_C = "#7b8794", "#cbd2d9", "#c0392b"
BUDGET = 200


def style():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.labelsize": 11, "axes.edgecolor": "#cbd2d9", "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.grid": True, "grid.color": "#e4e7eb", "grid.linewidth": 0.8,
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    })


def load():
    tx = Path("data/ieee/train_transaction.csv")
    if tx.exists():
        return ieee_data.load_ieee(tx, Path("data/ieee/train_identity.csv"))
    return ieee_data.load_ieee_frames(*ieee_data.mock_ieee_frames(seed=7))


def value_vs_budget(test, score, y, amount):
    budgets = np.arange(10, 601, 10)
    model_curve = [metrics.value_weighted_recall_at_k(y, score, amount, b) for b in budgets]
    rng = np.random.default_rng(0)
    rand = np.zeros(len(budgets))
    for _ in range(20):
        r = rng.permutation(len(score))
        rand += [metrics.value_weighted_recall_at_k(y, r.astype(float), amount, b) for b in budgets]
    rand /= 20

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(budgets, model_curve, color=MODEL_C, lw=2.4, label="model + ring features")
    ax.plot(budgets, rand, color=RAND_C, lw=2.0, ls="--", label="random ordering")
    at_b = metrics.value_weighted_recall_at_k(y, score, amount, BUDGET)
    ax.scatter([BUDGET], [at_b], color=RING_C, zorder=5, s=45)
    ax.annotate(f"{at_b:.0%} of fraud value\nin the top {BUDGET} alerts",
                (BUDGET, at_b), xytext=(BUDGET + 40, at_b - 0.16),
                color=INK, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlabel("alert budget (transactions reviewed)")
    ax.set_ylabel("share of fraud value caught")
    ax.set_title("IEEE-CIS: fraud value recovered per alert budget")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    fig.savefig(OUT / "ieee_value_vs_budget.png")
    plt.close(fig)


def ring_signal(test):
    buckets = pd.cut(test["device_prior_cards"], [-1, 0, 1, 3, 6, 10_000],
                     labels=["0", "1", "2-3", "4-6", "7+"])
    grp = test.assign(b=buckets).groupby("b", observed=True)["is_fraud"].agg(["mean", "size"])

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    colors = [MUTED if i < 3 else RING_C for i in range(len(grp))]
    bars = ax.bar(grp.index.astype(str), grp["mean"], color=colors, width=0.68)
    for bar, n in zip(bars, grp["size"], strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={n:,}", ha="center", va="bottom", color=MUTED, fontsize=9)
    ax.set_xlabel("distinct cards seen on the device before this transaction")
    ax.set_ylabel("fraud rate")
    ax.set_title("IEEE-CIS: device sharing is a ring signal")
    ax.set_ylim(0, 1.12)
    fig.savefig(OUT / "ieee_ring_signal.png")
    plt.close(fig)


def main() -> None:
    style()
    df = load()
    df, cols = ieee_features.add_ieee_features(df)
    train, valid, test, _ = data.time_split(df)
    scorer = model.train_scorer(train, feature_cols=cols)
    score = model.score(scorer, test)
    y, amount = test["is_fraud"].to_numpy(), test["amount"].to_numpy()

    value_vs_budget(test, score, y, amount)
    ring_signal(test)
    print("wrote", OUT / "ieee_value_vs_budget.png")
    print("wrote", OUT / "ieee_ring_signal.png")


if __name__ == "__main__":
    main()
