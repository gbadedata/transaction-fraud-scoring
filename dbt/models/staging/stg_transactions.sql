-- Cleaned, typed transactions. The raw table is loaded from the generator
-- (or a real dataset) into public.transactions; see README for loading.
with raw as (
    select * from {{ source('raw', 'transactions') }}
)

select
    transaction_id,
    card_id,
    merchant_id,
    mcc,
    amount::numeric(12, 2)          as amount,
    ts::timestamp                   as event_ts,
    lat,
    lon,
    upper(channel)                  as channel,
    card_issue_date::timestamp      as card_issue_date,
    is_fraud
from raw
where amount >= 0
