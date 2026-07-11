# IEEE-CIS: entity resolution and ring detection

The IEEE-CIS Fraud Detection dataset is card-not-present e-commerce fraud. It is a
useful stress test because it deliberately withholds the things a toy pipeline leans
on: there is **no account id** and **no geolocation**. What it gives instead is
anonymised card attributes, a billing region, email domains, a raw device string,
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
| `V1..V339` | Vesta engineered features | primary model inputs |

`train_identity.csv` (present for a minority of transactions):

| Field | Meaning | Used for |
| --- | --- | --- |
| `DeviceInfo` | raw device string (OS/browser family) | fingerprint + ring detection |
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

## Ring detection: structure over shared entities, and a real-data lesson

Organised fraud reuses infrastructure: one device drives many "different" cards, a
throwaway email domain shows up across a cluster of accounts. The hypothesis is that
this appears as one entity linked to many cards in a short window. We tested it
against the real data, and the honest result is worth more than a clean chart would
have been.

`DeviceInfo` on its own is **not** a fingerprint. Its most common values are operating
system and browser families ("Windows", "iOS Device", "MacOS", "Trident/7.0") shared
by enormous numbers of legitimate users, so raw "cards per device" is dominated by
popularity, not fraud. On the real data the fraud rate is flat across device-sharing
buckets, and an ungated rule will happily flag "device shared by thousands of cards"
about a Windows machine. This is the same low-cardinality trap that rules out "cards
per email domain" and "cards per region", and it applies to `DeviceInfo` too.

The fix is three-fold:

- **Build a more specific fingerprint.** `build_device_fp` combines `DeviceInfo` with
  browser (`id_31`) and screen resolution (`id_33`), which splits a coarse "Windows"
  into many specific fingerprints.
- **Give the model the context, not just the count.** `device_prior_cards` (distinct
  cards on the fingerprint before now) is kept as a model input alongside
  `device_fp_freq` (how common the fingerprint is overall), so the model can separate
  a rare shared fingerprint from a popular OS family. Both are structural, label-free.
- **Gate the human-facing parts.** The reason code and the deterministic ring rule
  only fire when the fingerprint is specific (its frequency is below
  `GENERIC_DEVICE_FREQ`). The queue can no longer say "device shared by 4000+ cards"
  about a common device.

The ring features remain **structural counts, never fraud rates**, so they cannot leak
the label:

- `device_prior_cards`: distinct cards seen on the fingerprint before this transaction
- `device_prior_txn`: transactions on the fingerprint before this transaction
- `device_fp_freq`: how common the fingerprint is overall (popular means generic)
- `email_domain_freq`: how common the purchaser domain is (rare domains are riskier)

Two guards carry over: missing identity is masked to zero so the large "no device"
bucket cannot look like one giant ring, and low-cardinality fields (domain, region)
contribute rarity rather than a cards-per-entity count.

## Model firepower: the Vesta columns are the signal

IEEE-CIS rewards heavy feature engineering, and the bulk of the signal lives in the
Vesta columns, not the entity features. The pipeline feeds the model all of `V1..V339`,
the full `C1..C14` and `D1..D15` (plus de-trended `Dn - day` versions), the numeric
and categorical identity fields, and label-free frequency encodings for `card1`,
`addr1`, and the email domain. A single calibrated gradient-boosted model on this set
is a fair, defensible baseline. It will not match competition winners, who stack heavy
ensembles and adversarial validation, and it does not pretend to.

The entity and ring features earn their place three ways even though they are not the
model's main lift: they are strong inputs in their own right, they provide the
human-readable reason codes on the queue, and they back a deterministic structural rule
that can catch a specific-fingerprint ring in the window before the model has evidence,
or if a model were unavailable. On real data, adding the ring score to the ranking
blend does not reliably beat the model alone, which is the honest result.

## Results on the real data

Numbers below are from a single seeded run on the full competition data (590,540
transactions, 3.499% fraud, $3,083,845 in fraud value), time-split 60/20/20.

Ranking and operating point on the test slice:

| Metric | Value |
| --- | --- |
| PR-AUC (average precision) | 0.47 |
| Cost-chosen threshold | ~11,900 alerts, precision 0.22, recall 0.65 |
| Fraud value recovered at that point | $380,070 of $609,934 (62%) |

PR-AUC of 0.47 is a deliberate, honest floor, not a disappointing one. Adding the full
Vesta feature set moved the top of the queue but not the whole curve, and a
higher-capacity model added little on top of that, which localises the ceiling in the
features rather than the learner. The remaining gap to published leaderboard scores is
bridged almost entirely by two techniques this repo refuses to use: aggregating
features over the entire dataset (which leaks future information into each row) and
stacking large ensembles. Both inflate an offline score and neither survives
deployment. The client key is anchored on `D1 - day` and every aggregate is
strictly-before, so this is a number you could actually run.

For a review-budget product the operating result is the headline: at the chosen point
the model recovers 62% of fraud value, and the value-versus-budget curve shows the full
precision/recall/value trade-off a team would navigate against its own headcount.

The device signal is honest too. After gating out generic OS/browser families, fraud
rate does rise with fingerprint sharing (about 9% at two cards, 15% at four to six,
against a 3.5% base), but it is a modest, non-monotonic signal, not a clean ring
detector, and the largest "specific" bucket falls back toward the base rate. The
structural rule flags roughly 9,700 transactions at 9% fraud, useful as an
investigator lens and a cold-start catch, not as a standalone precise rule.

## Investigation SQL

Before any model, an analyst should be able to find fraud by hand.
`sql/investigation.sql` (run via `scripts/run_investigation.py`, DuckDB over the raw
CSVs) surfaces fraud rate by product, devices linking many cards, email domains by
fraud rate, high-velocity cards, round-amount effects, and a time-ordered walk
through the largest ring. These are the queries that motivate the features above.

## Honesty about the mock

With no download, everything runs on a schema-faithful mock. The mock gives each card
stable attributes (so account reconstruction is real work) and, importantly, now
reproduces the device confound instead of hiding it: normal traffic uses generic
low-cardinality device strings shared by many cards, while rings share a specific, rare
fingerprint. Its Vesta columns carry only weak, mostly synthetic signal, so ranking
numbers from the mock are illustrative and do not predict real-data quality. Point the
loader at the real files and every script behaves identically.

