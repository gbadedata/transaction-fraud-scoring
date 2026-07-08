# IEEE-CIS: entity resolution and ring detection

The IEEE-CIS Fraud Detection dataset is card-not-present e-commerce fraud. It is a
useful stress test because it deliberately withholds the things a toy pipeline leans
on: there is **no account id** and **no geolocation**. What it gives instead is
anonymised card attributes, a billing region, email domains, a device fingerprint,
and hundreds of engineered Vesta columns. So the work that matters here is
reconstructing identity and finding structure, not tuning a classifier.

## What the dataset provides

`train_transaction.csv` (one row per transaction):

| Field | Meaning | Used for |
| --- | --- | --- |
| `TransactionID` | unique id | join key |
| `isFraud` | label | target |
| `TransactionDT` | seconds from a fixed reference | timestamp, time-based split |
| `TransactionAmt` | amount | value weighting, amount features |
| `ProductCD` | product code | risk varies sharply by product |
| `card1..card6` | anonymised card attributes | account reconstruction |
| `addr1, addr2` | billing region | account key, region signal |
| `P_emaildomain` | purchaser email domain | domain-rarity signal |
| `C1..C14` | Vesta counting features | model inputs |
| `D1..D15` | Vesta timedelta features | model inputs |
| `M1..M9` | match flags | model inputs |
| `V1..V339` | Vesta engineered features | optional model inputs |

`train_identity.csv` (present for a minority of transactions):

| Field | Meaning | Used for |
| --- | --- | --- |
| `DeviceInfo` | device fingerprint string | **ring detection** |
| `DeviceType` | desktop / mobile | model input |
| `id_01..id_38` | anonymised identity signals | model inputs |

The loader (`fraud.ieee_data.load_ieee`) joins the two tables on `TransactionID`,
derives a timestamp from `TransactionDT`, and exposes `amount` / `is_fraud` so the
existing model, scoring, and metrics code runs unchanged.

## Entity resolution: reconstructing an account

There is no `card_id`. The stable card attributes (`card1`, `card2`, `card3`,
`card5`) plus billing region (`addr1`) approximate a card/account, so we build:

```
uid = card1-card2-card3-card5-addr1
```

Grouping by `uid` lets us compute per-account behaviour: velocity in a trailing
window, time since the account's last transaction, and how far the current amount
sits from the account's own history. Every one of these is computed from **prior
rows only** (strictly-before, current row excluded), so nothing leaks from the
present into its own features.

## Ring detection: structure over shared entities

Organised fraud reuses infrastructure. One device drives many "different" cards; a
throwaway email domain shows up across a cluster of accounts. That shows up as
**one entity linked to many cards in a short window**. The primary signal is the
device, because `DeviceInfo` is high-cardinality: a normal device touches one or two
cards, so a device touching a dozen is not a coincidence.

The ring features are deliberately **structural counts, not fraud rates**:

- `device_prior_cards` — distinct cards seen on this device before this transaction
- `device_prior_txn` — transactions on this device before this transaction
- `email_domain_freq` — how common the purchaser domain is (rare domains are riskier)

Because they never read the label, they cannot leak, and they double as an
explanation: the queue can say *"device shared by 14+ cards"* instead of *"score
0.97"*. Two guards matter in practice:

- **Missing is not shared.** Most transactions have no device. Those rows are masked
  to zero so the large "missing" bucket cannot masquerade as one giant ring.
- **Low-cardinality fields are not rings.** Everyone uses a handful of email domains
  and lives in a handful of regions, so "cards per domain" or "cards per region" is
  not a ring signal. Only device sharing is, plus domain *rarity* as a weak add-on.

## What the model does with this, and what it does not

On IEEE-CIS the gradient-boosted model already learns the device-sharing feature, so
adding a separate ring rule to the ranking blend does not move precision or recall.
That is the honest result, and the ring work still earns its place three ways: the
counts are among the model's strongest inputs; they provide the human-readable reason
codes on the queue; and they back a deterministic structural rule that catches a ring
even in the window before the model has evidence, or if a model were unavailable.

## Investigation SQL

Before any model, an analyst should be able to find fraud by hand.
`sql/investigation.sql` (run via `scripts/run_investigation.py`, DuckDB over the raw
CSVs) surfaces fraud rate by product, devices linking many cards, email domains by
fraud rate, high-velocity cards, round-amount effects, and a time-ordered walk
through the largest ring. These are the queries that motivate the features above.

## Honesty about the mock

With no download, everything runs on a schema-faithful mock. The mock gives each card
stable attributes (so account reconstruction is real work), makes normal devices
high-cardinality, and injects rings as many distinct cards sharing one device and
often a rare email. It is a development and test aid, not a benchmark: reported
numbers from the mock are illustrative. Point the loader at the real files and every
script behaves identically.
