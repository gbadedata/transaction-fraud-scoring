-- Per-card velocity features computed in the warehouse.
--
-- These mirror src/fraud/features.py. The `exclude current row` clause is the
-- SQL equivalent of the leakage guard: each transaction's velocity counts prior
-- transactions only, never itself. Requires PostgreSQL 11+ (frame exclusion).
with t as (
    select * from {{ ref('stg_transactions') }}
)

select
    transaction_id,
    card_id,
    event_ts,
    amount,

    count(*) over w1                      as txn_count_1h,
    coalesce(sum(amount) over w1, 0)      as amt_sum_1h,
    count(*) over w24                     as txn_count_24h,
    coalesce(sum(amount) over w24, 0)     as amt_sum_24h,

    extract(epoch from (event_ts - lag(event_ts) over p)) as time_since_last_s
from t
window
    p as (partition by card_id order by event_ts),
    w1 as (
        partition by card_id order by event_ts
        range between interval '1 hour' preceding and current row
        exclude current row
    ),
    w24 as (
        partition by card_id order by event_ts
        range between interval '24 hours' preceding and current row
        exclude current row
    )
