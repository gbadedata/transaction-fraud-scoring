# Data

The repo runs on **synthetic data** from `fraud.data.generate_transactions()`, with
no download required. To dump a copy for the warehouse / dbt path:

```python
from fraud import data, features
df = features.add_features(data.generate_transactions())
df.to_parquet("data/transactions.parquet")   # load into Postgres public.transactions
```

## Swapping in a real dataset
Replace the generator with a loader that returns the same schema
(`fraud.data.RAW_COLUMNS`). Good public options:

- **IEEE-CIS Fraud Detection** (Kaggle). Realistic e-commerce card fraud with rich
  anonymised features. A widely used public benchmark.
- **ealtman2019/credit-card-transactions**. Large labelled synthetic card
  transactions with geography and merchant detail.

Avoid using the ULB `creditcard.csv` "99.9% accuracy" example as a template. A
near-perfect score on that dataset reflects leakage and class imbalance, not real
detection quality.

Raw data files are git-ignored; never commit cardholder-like data.
