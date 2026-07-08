-- Investigation SQL for IEEE-CIS (DuckDB dialect).
--
-- These are the queries a fraud analyst runs to *find* fraud before any model is
-- trained: where it concentrates, which devices link many cards (rings), which
-- domains and amounts carry risk. The runner (scripts/run_investigation.py) creates
-- three views from the CSVs, so these queries assume they exist:
--   tx      = train_transaction.csv
--   idn     = train_identity.csv
--   joined  = tx LEFT JOIN idn USING (TransactionID)
--
-- Run:  python scripts/run_investigation.py            (mock or data/ieee/)
--       python scripts/run_investigation.py --dir data/ieee

-- name: Fraud rate and volume by product
SELECT
    ProductCD,
    COUNT(*)              AS txns,
    SUM(isFraud)          AS frauds,
    ROUND(AVG(isFraud), 4) AS fraud_rate
FROM tx
GROUP BY ProductCD
ORDER BY fraud_rate DESC;

-- name: Devices linking many cards (ring candidates)
SELECT
    DeviceInfo,
    COUNT(*)                 AS txns,
    COUNT(DISTINCT card1)    AS distinct_cards,
    SUM(isFraud)             AS frauds,
    ROUND(AVG(isFraud), 3)   AS fraud_rate
FROM joined
WHERE DeviceInfo IS NOT NULL
GROUP BY DeviceInfo
HAVING COUNT(DISTINCT card1) >= 4
ORDER BY distinct_cards DESC
LIMIT 15;

-- name: Email domains by fraud rate (minimum volume)
SELECT
    P_emaildomain,
    COUNT(*)                 AS txns,
    COUNT(DISTINCT card1)    AS distinct_cards,
    ROUND(AVG(isFraud), 3)   AS fraud_rate
FROM tx
WHERE P_emaildomain IS NOT NULL
GROUP BY P_emaildomain
HAVING COUNT(*) >= 20
ORDER BY fraud_rate DESC
LIMIT 15;

-- name: Highest-velocity cards
SELECT
    card1,
    COUNT(*)                     AS txns,
    ROUND(SUM(TransactionAmt), 2) AS total_amount,
    SUM(isFraud)                 AS frauds
FROM tx
GROUP BY card1
ORDER BY txns DESC
LIMIT 15;

-- name: Fraud rate for round vs precise amounts
SELECT
    CASE WHEN TransactionAmt = FLOOR(TransactionAmt) THEN 'round' ELSE 'has_cents' END AS amount_type,
    COUNT(*)               AS txns,
    ROUND(AVG(isFraud), 4) AS fraud_rate
FROM tx
GROUP BY amount_type
ORDER BY fraud_rate DESC;

-- name: Anatomy of the largest ring
WITH ranked AS (
    SELECT DeviceInfo, COUNT(DISTINCT card1) AS distinct_cards
    FROM joined
    WHERE DeviceInfo IS NOT NULL
    GROUP BY DeviceInfo
    ORDER BY distinct_cards DESC
    LIMIT 1
)
SELECT
    j.DeviceInfo,
    j.card1,
    j.TransactionDT,
    j.TransactionAmt,
    j.isFraud
FROM joined j
JOIN ranked r ON j.DeviceInfo = r.DeviceInfo
ORDER BY j.TransactionDT
LIMIT 20;
