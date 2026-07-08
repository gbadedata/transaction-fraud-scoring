# Data Dictionary

## Raw transaction (`fraud.data.RAW_COLUMNS`, dbt source `raw.transactions`)

| Column | Type | Description |
|---|---|---|
| `transaction_id` | int | Primary key, one row per authorisation attempt. |
| `card_id` | int | Card identifier. |
| `merchant_id` | int | Merchant identifier. |
| `mcc` | int | Merchant category code. |
| `amount` | float | Transaction amount. |
| `ts` / `event_ts` | datetime | Transaction timestamp (used for time split & velocity). |
| `lat`, `lon` | float | Transaction geolocation. |
| `channel` | str | `CP` (card present) or `CNP` (card not present). |
| `card_issue_date` | datetime | When the card was issued (for card age). |
| `is_fraud` | int | Label, 0/1. |

## Engineered features (`fraud.features.FEATURE_COLS`)

All computed **as-of** the transaction, using prior rows only (leakage-safe).

| Feature | Description | Signal |
|---|---|---|
| `amount` | Raw amount. | Large or unusual spend. |
| `txn_count_1h` / `txn_count_24h` | Prior transactions on the card in trailing 1h / 24h. | Velocity / card testing / bust-out. |
| `amt_sum_1h` / `amt_sum_24h` | Prior spend on the card in trailing 1h / 24h. | Rapid drain. |
| `amount_z` | Amount z-score vs the card's own prior history. | Out-of-character spend. |
| `time_since_last_s` | Seconds since the card's previous transaction. | Bursts; dormant-then-active. |
| `geo_dist_km` | Distance from the previous transaction. | Location jumps. |
| `geo_speed_kmh` | Implied travel speed since the previous transaction. | Impossible travel. |
| `mcc_novel` | 1 if the card has never used this MCC before. | New spending pattern. |
| `hour` | Hour of day (0 to 23). | Odd-hour activity. |
| `is_night` | 1 if 00:00 to 05:59. | Odd-hour activity. |
| `is_cnp` | 1 if card-not-present. | Higher-risk channel. |
| `card_age_days` | Days since card issue. | New-card abuse. |

## Warehouse models (dbt)

| Model | Grain | Purpose |
|---|---|---|
| `stg_transactions` | transaction | Cleaned, typed, validated raw feed. |
| `fct_card_velocity` | transaction | Velocity features via SQL window functions (`exclude current row`). |
| `fct_transactions` | transaction | Analyst-facing fact: fields + velocity + derived flags. |
