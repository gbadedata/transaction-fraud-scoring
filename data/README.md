# Data

The repo runs on **synthetic data** from `fraud.data.generate_transactions()`, with
no download required. To dump a copy for the warehouse / dbt path:

```python
from fraud import data, features
df = features.add_features(data.generate_transactions())
df.to_parquet("data/transactions.parquet")   # load into Postgres public.transactions
```

## Running on IEEE-CIS (real data, wired in)
The repo ships a real loader for the **IEEE-CIS Fraud Detection** dataset
(`fraud.ieee_data.load_ieee`) plus a schema-faithful mock, so the IEEE pipeline
runs with no download. To use the real data:

```bash
pip install kaggle
# accept the competition rules once at kaggle.com/c/ieee-fraud-detection, then:
kaggle competitions download -c ieee-fraud-detection -p data/ieee
cd data/ieee && unzip ieee-fraud-detection.zip && cd -
```

Expected paths after unzip:

```
data/ieee/train_transaction.csv
data/ieee/train_identity.csv
```

Then run the IEEE pipeline and the investigation queries:

```bash
python run_ieee_demo.py                 # entity resolution + ring detection
python scripts/run_investigation.py     # analyst SQL over the raw CSVs (DuckDB)
python scripts/generate_ieee_figures.py # regenerate the IEEE figures
```

Both scripts auto-detect `data/ieee/`; if it is absent they fall back to the mock.
See `docs/ieee_cis.md` for how the anonymised IEEE schema maps onto the pipeline
and how accounts and rings are reconstructed.

## Other public options
- **ealtman2019/credit-card-transactions**. Large labelled synthetic card
  transactions with geography and merchant detail. Fits the synthetic schema
  (`fraud.data.RAW_COLUMNS`) with a thin loader.

Avoid using the ULB `creditcard.csv` "99.9% accuracy" example as a template. A
near-perfect score on that dataset reflects leakage and class imbalance, not real
detection quality.

Raw data files are git-ignored; never commit cardholder-like data.
