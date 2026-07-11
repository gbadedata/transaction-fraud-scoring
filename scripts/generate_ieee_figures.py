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
    n = len(test)
    top = max(50, int(n * 0.12))
    budgets = np.unique(np.clip(np.linspace(max(1, top // 60), top, 60).astype(int), 1, n))
    model_curve = np.array(
        [metrics.value_weighted_recall_at_k(y, score, amount, b) for b in budgets])
    rng = np.random.default_rng(0)
    rand = np.zeros(len(budgets))
    for _ in range(10):
        r = rng.permutation(len(score)).astype(float)
        rand += [metrics.value_weighted_recall_at_k(y, r, amount, b) for b in budgets]
    rand /= 10

    # Annotate where the model recovers half of all fraud value.
    hit = model_curve >= 0.5
    half_idx = int(np.argmax(hit)) if hit.any() else len(budgets) - 1
    b_half, v_half = int(budgets[half_idx]), float(model_curve[half_idx])

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(budgets, model_curve, color=MODEL_C, lw=2.4, label="model ranking")
    ax.plot(budgets, rand, color=RAND_C, lw=2.0, ls="--", label="random ordering")
    ax.scatter([b_half], [v_half], color=RING_C, zorder=5, s=45)
    ax.annotate(f"{v_half:.0%} of fraud value\nin the top {b_half:,} reviewed",
                (b_half, v_half), xytext=(b_half * 1.05, max(0.08, v_half - 0.22)),
                color=INK, fontsize=10, arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlabel("alert budget (transactions reviewed)")
    ax.set_ylabel("share of fraud value caught")
    ax.set_title("IEEE-CIS: fraud value recovered per alert budget")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    fig.savefig(OUT / "ieee_value_vs_budget.png")
    plt.close(fig)


def ring_signal(test):
    from fraud.ieee_features import GENERIC_DEVICE_FREQ, _specific_device
    sub = test[_specific_device(test, GENERIC_DEVICE_FREQ)]
    buckets = pd.cut(sub["device_prior_cards"], [-1, 0, 1, 3, 6, 10_000],
                     labels=["0", "1", "2-3", "4-6", "7+"])
    grp = sub.assign(b=buckets).groupby("b", observed=True)["is_fraud"].agg(["mean", "size"])

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    colors = [MUTED if i < 3 else RING_C for i in range(len(grp))]
    bars = ax.bar(grp.index.astype(str), grp["mean"], color=colors, width=0.68)
    for bar, n in zip(bars, grp["size"], strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={n:,}", ha="center", va="bottom", color=MUTED, fontsize=9)
    ax.set_xlabel("distinct cards on the fingerprint before this transaction")
    ax.set_ylabel("fraud rate")
    ax.set_title("IEEE-CIS: sharing of a specific device fingerprint")
    ax.set_ylim(0, 1.12)
    fig.savefig(OUT / "ieee_ring_signal.png")
    plt.close(fig)


def main() -> None:
    style()
    df = load()
    df, cols = ieee_features.add_ieee_features(df)
    train, valid, test, _ = data.time_split(df)
    scorer = model.train_scorer(
        train, feature_cols=cols,
        max_depth=None, max_leaf_nodes=63, learning_rate=0.05,
        max_iter=300, min_samples_leaf=50,
    )
    score = model.score(scorer, test)
    y, amount = test["is_fraud"].to_numpy(), test["amount"].to_numpy()

    value_vs_budget(test, score, y, amount)
    ring_signal(test)
    print("wrote", OUT / "ieee_value_vs_budget.png")
    print("wrote", OUT / "ieee_ring_signal.png")


if __name__ == "__main__":
    main()
