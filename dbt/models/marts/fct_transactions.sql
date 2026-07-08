-- Analyst-facing transaction fact: cleaned fields + velocity + derived flags.
-- This is the table an investigation SQL notebook or dashboard queries.
select
    s.transaction_id,
    s.card_id,
    s.merchant_id,
    s.mcc,
    s.amount,
    s.event_ts,
    s.channel,
    s.is_fraud,

    v.txn_count_1h,
    v.amt_sum_1h,
    v.txn_count_24h,
    v.amt_sum_24h,
    v.time_since_last_s,

    extract(hour from s.event_ts)                              as event_hour,
    (extract(hour from s.event_ts) between 0 and 5)::int       as is_night,
    (s.channel = 'CNP')::int                                   as is_cnp,
    (s.event_ts::date - s.card_issue_date::date)               as card_age_days
from {{ ref('stg_transactions') }} s
left join {{ ref('fct_card_velocity') }} v using (transaction_id)
