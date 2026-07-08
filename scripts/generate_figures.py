"""Generate the figures used in the README from the real pipeline.

    python scripts/generate_figures.py

Writes PNGs to docs/img/. Everything is derived from a single seeded run, so the
charts in the README stay in sync with the code and are reproducible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve

from fraud import data, features, metrics, model, rules, scoring
from fraud.features import FEATURE_COLS

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

BUDGET = 100
REVIEW_COST, FRICTION_COST = 4.0, 12.0
INK = "#1f2933"
MUTED = "#9aa5b1"
MODEL_C = "#7b8794"
BLEND_C = "#c0392b"
GOOD_C = "#2e7d5b"
BAD_C = "#b23a3a"

BLEND_CANDIDATES = [
    (1.00, 0.00, 0.00), (0.80, 0.20, 0.00), (0.70, 0.30, 0.00),
    (0.70, 0.20, 0.10), (0.60, 0.25, 0.15), (0.50, 0.30, 0.20),
]


def style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": "#cbd2d9",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": True,
        "grid.color": "#e4e7eb",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "figure.titlesize": 13,
    })


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build():
    df = features.add_features(data.generate_transactions(seed=42))
    train, valid, test, _ = data.time_split(df)
    scorer = model.train_scorer(train)

    def sig(frame):
        return (model.score(scorer, frame),
                rules.apply_rules(frame),
                scoring.anomaly_scores(train, frame))

    v_prob, v_hits, v_anom = sig(valid)
    t_prob, t_hits, t_anom = sig(test)
    yv, av = valid["is_fraud"].to_numpy(), valid["amount"].to_numpy()
    yt, at = test["is_fraud"].to_numpy(), test["amount"].to_numpy()

    best_w, best = BLEND_CANDIDATES[0], -1.0
    for w in BLEND_CANDIDATES:
        s = scoring.blend(v_prob, v_hits["rule_score"].to_numpy(), v_anom, *w)
        vwr = metrics.value_weighted_recall_at_k(yv, s, av, BUDGET)
        if vwr > best:
            best, best_w = vwr, w

    t_blend = scoring.blend(t_prob, t_hits["rule_score"].to_numpy(), t_anom, *best_w)
    v_blend = scoring.blend(v_prob, v_hits["rule_score"].to_numpy(), v_anom, *best_w)
    return dict(scorer=scorer, test=test, yt=yt, at=at, t_prob=t_prob,
                t_blend=t_blend, t_hits=t_hits, yv=yv, av=av, v_blend=v_blend,
                weights=best_w)


def fig_value_vs_budget(ctx):
    yt, at = ctx["yt"], ctx["at"]
    ks = np.arange(1, 601)
    model_v = [metrics.value_weighted_recall_at_k(yt, ctx["t_prob"], at, k) for k in ks]
    blend_v = [metrics.value_weighted_recall_at_k(yt, ctx["t_blend"], at, k) for k in ks]
    total = at[yt == 1].sum()
    rand = np.minimum(ks * (at.sum() / len(at)), total) / total  # value from a random queue

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(ks, np.array(blend_v) * 100, color=BLEND_C, lw=2.4, label="Blended score")
    ax.plot(ks, np.array(model_v) * 100, color=MODEL_C, lw=2.0, label="Model only")
    ax.plot(ks, rand * 100, color=MUTED, lw=1.4, ls=":", label="Random ordering")

    b = int(BUDGET)
    ax.axvline(b, color=INK, lw=1.0, ls="--", alpha=0.6)
    ax.scatter([b], [blend_v[b - 1] * 100], color=BLEND_C, zorder=5, s=36)
    ax.annotate(f"{blend_v[b - 1] * 100:.0f}% of fraud value\nin {b} alerts",
                (b, blend_v[b - 1] * 100), xytext=(b + 30, blend_v[b - 1] * 100 - 20),
                fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1))
    ax.set_xlabel("Alert budget (transactions reviewed, ranked by score)")
    ax.set_ylabel("Fraud value recovered (%)")
    ax.set_title("Fraud value recovered vs. review budget")
    ax.set_ylim(0, 103)
    ax.set_xlim(0, 600)
    ax.legend(loc="lower right")
    despine(ax)
    fig.savefig(OUT / "value_vs_budget.png")
    plt.close(fig)


def fig_pr_curve(ctx):
    yt = ctx["yt"]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for name, s, c in [("Blended", ctx["t_blend"], BLEND_C),
                       ("Model only", ctx["t_prob"], MODEL_C)]:
        p, r, _ = precision_recall_curve(yt, s)
        ap = metrics.pr_auc(yt, s)
        ax.plot(r, p, color=c, lw=2.2, label=f"{name} (PR-AUC {ap:.2f})")
    base = yt.mean()
    ax.axhline(base, color=MUTED, ls=":", lw=1.4, label=f"Prevalence ({base:.1%})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall (test slice)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.legend(loc="lower left")
    despine(ax)
    fig.savefig(OUT / "pr_curve.png")
    plt.close(fig)


def fig_cost_curve(ctx):
    yt, at = ctx["yt"], ctx["at"]
    s = ctx["t_blend"]
    grid = np.unique(np.quantile(s, np.linspace(0, 1, 200)))
    costs = [metrics.operating_cost(yt, s, at, t, REVIEW_COST, FRICTION_COST).cost
             for t in grid]
    do_nothing = at[yt == 1].sum()
    chosen = metrics.best_threshold_by_cost(ctx["yv"], ctx["v_blend"], ctx["av"],
                                            review_cost=REVIEW_COST,
                                            friction_cost=FRICTION_COST).threshold
    at_test = metrics.operating_cost(yt, s, at, chosen, REVIEW_COST, FRICTION_COST)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(grid, np.array(costs) / 1000, color=BLEND_C, lw=2.2, label="Operating cost")
    ax.axhline(do_nothing / 1000, color=MUTED, ls=":", lw=1.6,
               label=f"Do nothing (${do_nothing/1000:.0f}k lost)")
    ax.axvline(chosen, color=INK, ls="--", lw=1.0, alpha=0.7)
    ax.scatter([chosen], [at_test.cost / 1000], color=INK, zorder=5, s=36)
    label = (f"chosen on validation\n"
             f"${at_test.cost/1000:.1f}k, precision {at_test.precision:.2f}")
    ax.annotate(label, (chosen, at_test.cost / 1000),
                xytext=(chosen + 0.12, at_test.cost / 1000 + 4),
                fontsize=10, arrowprops=dict(arrowstyle="->", color=INK, lw=1))
    ax.set_xlabel("Score threshold")
    ax.set_ylabel("Total operating cost ($ thousands)")
    ax.set_title("Cost sets the threshold, not a default cut-off")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper center")
    despine(ax)
    fig.savefig(OUT / "cost_curve.png")
    plt.close(fig)


def fig_calibration(ctx):
    yt = ctx["yt"]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1.6, label="Perfect calibration")
    frac_pos, mean_pred = calibration_curve(yt, ctx["t_prob"], n_bins=8, strategy="quantile")
    ax.plot(mean_pred, frac_pos, color=BLEND_C, lw=2.2, marker="o", ms=5,
            label="Calibrated model")
    ax.set_xlabel("Predicted fraud probability")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title("Scores mean what they say")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    despine(ax)
    fig.savefig(OUT / "calibration.png")
    plt.close(fig)


def fig_rule_precision(ctx):
    rep = rules.rule_precision(ctx["t_hits"], ctx["test"]["is_fraud"])
    rep = rep.fillna({"precision": 0.0})
    names = rep["rule"].tolist()
    prec = rep["precision"].to_numpy()
    alerts = rep["alerts"].to_numpy()
    colors = [GOOD_C if p >= 0.5 else BAD_C for p in prec]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bars = ax.bar(names, prec, color=colors, width=0.62)
    for bar, n, p in zip(bars, alerts, prec, strict=False):
        label = f"{p:.0%}\n({n} alerts)" if n else "no alerts"
        ax.text(bar.get_x() + bar.get_width() / 2, max(p, 0) + 0.03, label,
                ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_ylabel("Precision")
    ax.set_title("Per-rule precision: which rules earn their alerts")
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color=MUTED, ls=":", lw=1.2)
    ax.tick_params(axis="x", labelrotation=12)
    despine(ax)
    fig.savefig(OUT / "rule_precision.png")
    plt.close(fig)


def fig_feature_importance(ctx):
    X = ctx["test"][FEATURE_COLS].to_numpy(float)
    y = ctx["yt"]
    r = permutation_importance(ctx["scorer"], X, y, scoring="average_precision",
                               n_repeats=6, random_state=0)
    order = np.argsort(r.importances_mean)
    names = np.array(FEATURE_COLS)[order]
    vals = r.importances_mean[order]
    errs = r.importances_std[order]

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.barh(names, vals, xerr=errs, color=BLEND_C, alpha=0.85,
            error_kw=dict(ecolor=MUTED, lw=1))
    ax.set_xlabel("Drop in PR-AUC when feature is shuffled")
    ax.set_title("What the model relies on (permutation importance)")
    despine(ax)
    fig.savefig(OUT / "feature_importance.png")
    plt.close(fig)


def main():
    style()
    ctx = build()
    print(f"blend weights (model, rules, anomaly): {ctx['weights']}")
    for fn in (fig_value_vs_budget, fig_pr_curve, fig_cost_curve,
               fig_calibration, fig_rule_precision, fig_feature_importance):
        fn(ctx)
        print("wrote", fn.__name__)
    print("figures ->", OUT)


if __name__ == "__main__":
    main()
