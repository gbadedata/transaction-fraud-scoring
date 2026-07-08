# Model Card: Transaction Fraud Scorer

> Template model card. The numbers below are from the synthetic demo; replace them
> with results from your real dataset when you build this out.

## Intended use
Rank card transactions by fraud risk so a capacity-limited investigation team
reviews the highest **expected-loss** cases first. It is a *decision-support*
tool: outputs feed an analyst queue and rule engine, not an automatic block.

**Out of scope:** automated declines without human review; use on populations or
channels not represented in training; any use where a false decline's harm has
not been weighed against fraud loss.

## Data
- **Unit:** one authorisation attempt.
- **Source (demo):** `fraud.data.generate_transactions`, synthetic, with injected
  typologies (card testing, bust-out, takeover spend) and legitimate confounders
  (big-ticket purchases, travellers).
- **Label:** `is_fraud` (0/1). In production, define the label window explicitly
  (e.g. confirmed-fraud chargebacks within 90 days) and record label latency.
- **Split:** strictly time-based (train, then valid, then test). Never random; see
  `docs/decisions/0001-time-based-split.md`.

## Features
Leakage-safe, computed as-of each transaction (prior rows only): velocity
(1h/24h counts & sums), amount z-score vs the card's own history, geo distance and
implied speed from the previous transaction, MCC novelty, recency, night/CNP flags,
card age. Full list in `docs/data_dictionary.md`.

## Model
`HistGradientBoostingClassifier` with isotonic probability calibration
(`CalibratedClassifierCV`). Calibration matters: expected-loss ranking and
cost-based thresholds are only valid if a 0.9 score really means ~90% fraud.
LightGBM/XGBoost are drop-in replacements behind the same interface.

## Evaluation (demo data)
Metrics are chosen for a base rate under 1.5%. Accuracy is not reported because it
is meaningless here.

| Metric | Value (demo) |
|---|---|
| PR-AUC (average precision) | ~0.92 model, ~0.86 blended |
| Precision at 100-alert budget | ~1.00 |
| Fraud-value recall at 100-alert budget | ~0.35 blended vs ~0.23 model |
| Cost-chosen operating point (precision / recall) | ~0.78 / ~0.81 |

The operating point is chosen by cost, not by a default 0.5 threshold:
`cost = missed-fraud value + review labour + customer-friction`. The blend weights
and the threshold are selected on validation and then measured on test. The
fixed-budget view exposes the key tension: a 100-alert budget can be near-100%
precise yet still catch only a minority of fraud value, because fraud volume
exceeds review capacity.

## Monitoring
Scheduled PSI/KS drift checks on feature and score distributions (reference window
vs recent window); see `fraud.monitoring`. Fraud is adversarial and non-stationary,
so drift + periodic per-rule precision review are first-class, not afterthoughts.

## Limitations & ethics
- Trained on labelled fraud only; unlabelled/unknown fraud is invisible to the
  supervised signal (the anomaly layer partially mitigates this).
- Feature or label shift will degrade precision silently without monitoring.
- False declines harm real customers; the cost model must reflect that, and
  thresholds should be reviewed with the business, not set by the modeller alone.
- Guard against features that proxy protected attributes; review for disparate
  impact before any production use.
