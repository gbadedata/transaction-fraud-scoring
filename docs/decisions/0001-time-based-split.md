# ADR 0001: Split data by time, never randomly

**Status:** accepted

## Context
Fraud detection is a forecasting problem: we train on the past to score the
future. It is also adversarial and non-stationary. Typologies, merchants, and
customer behaviour all drift week to week.

A random train/test split shuffles future transactions into the training set. The
model then benefits from information it could never have at scoring time
(later transactions on the same card, contemporaneous fraud episodes), which
inflates every offline metric and produces a model that looks excellent in a
notebook and disappoints in production. This is one of the most common ways offline fraud
metrics overstate real-world performance.

## Decision
Split strictly chronologically: earliest transactions → train, then validation,
then the most recent slice → test (`fraud.data.time_split`, default 60/20/20 by
timestamp quantile). Velocity/history features are additionally computed as-of each
transaction, excluding the current row (`fraud.features`), so there is no leakage
*within* a split either.

## Consequences
- Offline metrics are lower but trustworthy, because they approximate deployment.
- Enables honest drift monitoring: the test window is genuinely "later" data, so
  train-vs-test PSI is a real preview of production drift.
- Any future time-series CV must preserve order (expanding or rolling origin), and
  the label-definition window must sit entirely before each split's cutoff to avoid
  label leakage.
