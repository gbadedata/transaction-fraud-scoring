# Transaction Fraud Scoring and Alert Prioritisation

[![CI](https://github.com/gbadedata/transaction-fraud-scoring/actions/workflows/ci.yml/badge.svg)](https://github.com/gbadedata/transaction-fraud-scoring/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Lint: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A transaction fraud scoring pipeline built around the constraint that defines real fraud operations: an investigation team can only review a fixed number of alerts per day. The goal is to prevent the most fraud value within that budget, keep false declines low, and detect drift before precision degrades. This is a runnable reference implementation with tests, a warehouse layer, and a triage dashboard, not a single notebook.

## How it works

```
data ─▶ features ─▶ ┌ rules   ┐
                    │ model   ├─▶ blended score ─▶ expected-loss queue ─▶ cost/budget threshold ─▶ monitoring
                    └ anomaly ┘
```

A supervised model, a transparent rule catalogue, and an unsupervised anomaly detector each produce a signal. Those are blended into one risk score, the queue is ordered by expected loss (probability of fraud multiplied by amount), and the operating threshold is set from cost rather than a default cut-off. Drift on features and scores is monitored continuously.

## Quickstart

```bash
make setup     # pip install -e ".[dev]"
make test      # 22 tests
make demo      # full pipeline on synthetic data, no downloads
```

The repository ships a data generator that produces realistic transactions with embedded fraud typologies (card testing, stolen-card bust-out, account-takeover spend) and legitimate confounders (big-ticket purchases, travellers), so the problem is hard by construction.

## Example run

Output from `make demo` on the bundled seed:

```
74,017 transactions | fraud rate 1.18%
Time-based split: train 44,410 | valid 14,803 | test 14,804

Per-rule precision (test):
  card_testing          74 alerts   precision 1.00
  high_amount_new_card   6 alerts   precision 0.00   (trips on legitimate big-ticket spend)
  impossible_travel     77 alerts   precision 0.13   (travellers cause false positives)
  velocity_spike         0 alerts   precision  n/a   (fires on nothing on this data)

Blend weights selected on validation (model, rules, anomaly): (0.5, 0.3, 0.2)

Ranking quality (test):
  model only   PR-AUC 0.92 | fraud-value recall @100  0.23
  blended      PR-AUC 0.86 | fraud-value recall @100  0.35

Operating point (threshold chosen on validation, measured on test):
  precision 0.78 | recall 0.81 | fraud value caught $19.8k of $24.0k | cost $5.6k vs $24.0k
```

Two things in that output are the reason the project exists. First, PR-AUC is around 0.92 rather than 1.0: on honest, overlapping data nothing separates fraud perfectly, and a model that scores a perfect PR-AUC is usually signalling a leak. Second, the blend is not chosen to win PR-AUC. It is chosen on validation to maximise fraud value captured within the alert budget, which lifts value recall at 100 alerts from 0.23 to 0.35 (a large relative gain in dollars caught at the same cost) while trading away some PR-AUC. That is the correct trade when the objective is prevented loss under capacity.

The per-rule table is also doing real work: `high_amount_new_card` and `velocity_spike` earn no true positives on this data, which is exactly the evidence a team uses to retire or retune a rule.

## Design

| Component | Module | Rationale |
|---|---|---|
| Time-based split | `data.py`, [ADR-0001](docs/decisions/0001-time-based-split.md) | Train on the past and score the future. A random split leaks future information and inflates every metric. |
| Leakage-safe features | `features.py` | Every feature uses prior rows only, and velocity excludes the current row. Enforced by tests. |
| Rule catalogue | `rules.py` | Transparent tripwires with metadata and per-rule precision, so rules that only generate false positives can be retired. |
| Calibrated model | `model.py` | Isotonic-calibrated gradient boosting, so a 0.9 score corresponds to roughly 90 percent fraud and expected-loss ranking stays valid. |
| Anomaly layer | `scoring.py` | An unsupervised signal for patterns that are absent from the labels and the rules. |
| Expected-loss ranking | `scoring.py` | The queue is ordered by probability multiplied by amount, to maximise prevented value under a fixed budget. |
| Cost and budget thresholds | `metrics.py` | The threshold is set by cost (missed value plus review labour plus friction), with a separate fixed-budget view. |
| Drift monitoring | `monitoring.py` | PSI and KS on features and scores, because fraud is adversarial and non-stationary. |
| Selection discipline | `run_demo.py` | Blend weights and the operating threshold are selected on validation, then measured on test. |

## Metrics

The metrics are the operational ones for a base rate near one percent: precision at k and recall at k where k is the review budget, value-weighted recall (share of fraud value captured, since one large fraud outweighs many tiny ones), PR-AUC for threshold-free ranking quality, and an operating cost that combines missed fraud value, review labour, and customer friction. Accuracy is not reported, because at this base rate a model that flags nothing is already about 99 percent accurate.

## Repository layout

```
src/fraud/         core library (data, features, rules, model, scoring, metrics, monitoring)
tests/             22 unit tests; the feature tests pin down the no-leakage guarantee
dbt/               warehouse path: staging and velocity/fact marts with data-quality tests
dashboards/        Streamlit triage app (budget slider, ranked queue, reason codes)
docs/              model card, data dictionary, architecture decision records
run_demo.py        the pipeline end to end in one command
```

## Warehouse layer

The dbt models mirror the Python. `fct_card_velocity.sql` computes the same velocity features with SQL window functions and an `exclude current row` frame, which is the warehouse equivalent of the leakage guard. Staging carries data-quality tests (not-null, unique, accepted values, source freshness).

```bash
cp .env.example .env && docker compose up -d
cp dbt/profiles.example.yml ~/.dbt/profiles.yml
make dbt-build
```

## Extending to real data

The engine is real and tested; the data is synthetic. To run it in production:

1. Replace `generate_transactions` with a loader over a real feed such as IEEE-CIS or `ealtman2019/credit-card-transactions`. It only needs to return the schema in [`data/README.md`](data/README.md); everything downstream is unchanged.
2. Add entity-level features: per-merchant and per-device risk, and cross-card linkage for ring detection.
3. Tune the cost model to real unit economics and revisit the operating point with the business.
4. Enable the model-regression gate in `.github/workflows/ci.yml` so a drop in holdout PR-AUC fails the build, the same way unit tests catch code regressions.
5. Expand the rule catalogue and run the drift checks on a schedule.

## Tech stack

Python (numpy, pandas, scikit-learn), dbt with PostgreSQL, Streamlit, pytest, ruff, and GitHub Actions. The stack is intentionally small and reusable.

## Development

`make test` runs the suite; the feature tests are the important ones, since they prove the no-leakage property on hand-built fixtures. `make lint` runs ruff. CI runs both on every push. Design decisions live in `docs/` as a model card, a data dictionary, and architecture decision records.

## Data and limitations

All figures here come from synthetic data on a fixed seed and illustrate behaviour rather than real-world performance. The supervised signal only sees labelled fraud, so unknown fraud is invisible to it (the anomaly layer partially mitigates this). False declines harm real customers, so the cost assumptions and the chosen threshold should be reviewed with the business rather than set by the model alone. Raw data files are git-ignored; do not commit real cardholder data.

## License

MIT. See [LICENSE](LICENSE).
