"""Synthetic transaction generator and time-based splitting.

The generator exists so the repo runs end-to-end with zero external downloads.
It embeds realistic *typologies* AND realistic *confounders*, so the problem is hard:
fraud overlaps with legitimate behaviour, rules produce false positives, and no
model catches everything. That difficulty is deliberate. A fraud demo that scores
a perfect PR-AUC is a warning sign, not a success.

Typologies injected:
  * card testing        -- rapid burst of tiny CNP auths (easy)
  * stolen-card bust-out -- a few larger CNP hits, often from a distant location
  * takeover spend       -- CNP spend at 2-4x the card's norm, near home (HARD:
                            overlaps with legitimate big-ticket purchases)
Confounders (all legitimate, is_fraud=0):
  * big-ticket purchases -- occasional large legit spend (breaks 'big = fraud')
  * travellers           -- legit activity from a far location (breaks 'far = fraud')

For the real portfolio build, replace `generate_transactions` with a loader over
IEEE-CIS or `ealtman2019/credit-card-transactions`; downstream code is unchanged.

RAW SCHEMA (one row per transaction)
    transaction_id  int | card_id int | merchant_id int | mcc int | amount float
    ts datetime | lat float | lon float | channel str ('CP'|'CNP')
    card_issue_date datetime | is_fraud int (0/1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# grocery, restaurants, misc-retail, wire, direct-marketing, gambling, ATM
MCC_CHOICES = np.array([5411, 5812, 5999, 4829, 5967, 7995, 6011])
MCC_PROBS = np.array([0.28, 0.22, 0.20, 0.05, 0.10, 0.05, 0.05])
MCC_PROBS = MCC_PROBS / MCC_PROBS.sum()

RAW_COLUMNS = [
    "transaction_id", "card_id", "merchant_id", "mcc", "amount",
    "ts", "lat", "lon", "channel", "card_issue_date", "is_fraud",
]


def generate_transactions(
    n_cards: int = 1500,
    days: int = 30,
    start: str = "2024-01-01",
    n_merchants: int = 800,
    n_card_testing: int = 35,
    n_bustout: int = 45,
    n_takeover: int = 80,
    n_travellers: int = 130,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a labelled transaction table with typologies and confounders."""
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)
    window_s = days * 86_400

    home_lat = rng.uniform(25.0, 49.0, n_cards)
    home_lon = rng.uniform(-124.0, -67.0, n_cards)
    typical_amt = rng.lognormal(mean=3.2, sigma=0.5, size=n_cards)   # ~$25 median
    daily_rate = rng.uniform(0.3, 3.0, n_cards)
    age_days = rng.integers(5, 900, n_cards)
    card_issue = start_ts - pd.to_timedelta(age_days, unit="D")

    # --- legitimate traffic (vectorised) ---
    n_txn = rng.poisson(daily_rate * days)
    total = int(n_txn.sum())
    card_idx = np.repeat(np.arange(n_cards), n_txn)
    secs = rng.uniform(0, window_s, total)
    amounts = np.maximum(1.0, rng.lognormal(np.log(typical_amt[card_idx]), 0.4))
    # ~4% legit big-ticket purchases so 'large amount' is NOT a clean fraud signal
    big = rng.random(total) < 0.04
    amounts[big] = rng.uniform(150.0, 1000.0, int(big.sum()))
    channel = np.where(rng.random(total) < 0.35, "CNP", "CP")
    lat = home_lat[card_idx] + rng.normal(0, 0.15, total)
    lon = home_lon[card_idx] + rng.normal(0, 0.15, total)
    mcc = rng.choice(MCC_CHOICES, size=total, p=MCC_PROBS)

    legit = pd.DataFrame({
        "card_id": card_idx + 1,
        "merchant_id": rng.integers(1, n_merchants + 1, total),
        "mcc": mcc,
        "amount": np.round(amounts, 2),
        "ts": start_ts + pd.to_timedelta(secs, unit="s"),
        "lat": lat, "lon": lon, "channel": channel,
        "card_issue_date": card_issue[card_idx],
        "is_fraud": 0,
    })

    frames = [legit]
    frames += _travellers(rng, n_travellers, n_cards, typical_amt, card_issue,
                          start_ts, window_s, n_merchants)
    frames += _card_testing(rng, n_card_testing, n_cards, home_lat, home_lon,
                            card_issue, start_ts, window_s, n_merchants)
    frames += _bustout(rng, n_bustout, n_cards, home_lat, home_lon, card_issue,
                       start_ts, window_s, n_merchants)
    frames += _takeover_spend(rng, n_takeover, n_cards, home_lat, home_lon,
                              typical_amt, card_issue, start_ts, window_s, n_merchants)

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("ts").reset_index(drop=True)
    df.insert(0, "transaction_id", np.arange(1, len(df) + 1))
    return df[RAW_COLUMNS]


def _travellers(rng, n, n_cards, typical_amt, card_issue, start_ts, window_s, n_merchants):
    """Legit trips: normal spend from a distant location (a benign confounder)."""
    out = []
    for cid in rng.integers(1, n_cards + 1, n):
        k = int(rng.integers(3, 9))
        t0 = rng.uniform(0, window_s - 3 * 86_400)
        offsets = np.sort(rng.uniform(0, 3 * 86_400, k))       # over a few days
        trip_lat, trip_lon = rng.uniform(25.0, 49.0), rng.uniform(-124.0, -67.0)
        amt = np.maximum(1.0, rng.lognormal(np.log(typical_amt[cid - 1]), 0.4))
        out.append(pd.DataFrame({
            "card_id": cid,
            "merchant_id": rng.integers(1, n_merchants + 1, k),
            "mcc": rng.choice(MCC_CHOICES, size=k, p=MCC_PROBS),
            "amount": np.round(np.maximum(1.0, rng.lognormal(np.log(amt), 0.4, k)), 2),
            "ts": start_ts + pd.to_timedelta(t0 + offsets, unit="s"),
            "lat": trip_lat + rng.normal(0, 0.08, k),
            "lon": trip_lon + rng.normal(0, 0.08, k),
            "channel": np.where(rng.random(k) < 0.4, "CNP", "CP"),
            "card_issue_date": card_issue[cid - 1],
            "is_fraud": 0,
        }))
    return out


def _card_testing(rng, n, n_cards, home_lat, home_lon, card_issue, start_ts,
                  window_s, n_merchants):
    """Card testing: a rapid burst of tiny CNP authorisations on one card."""
    out = []
    for cid in rng.integers(1, n_cards + 1, n):
        k = int(rng.integers(8, 20))
        t0 = rng.uniform(0, window_s - 600)
        offsets = np.sort(rng.uniform(0, 600, k))              # within ~10 minutes
        out.append(pd.DataFrame({
            "card_id": cid,
            "merchant_id": rng.integers(1, n_merchants + 1, k),
            "mcc": rng.choice([5967, 5999], size=k),
            "amount": np.round(rng.uniform(0.5, 3.0, k), 2),
            "ts": start_ts + pd.to_timedelta(t0 + offsets, unit="s"),
            "lat": home_lat[cid - 1] + rng.normal(0, 0.05, k),
            "lon": home_lon[cid - 1] + rng.normal(0, 0.05, k),
            "channel": "CNP",
            "card_issue_date": card_issue[cid - 1],
            "is_fraud": 1,
        }))
    return out


def _bustout(rng, n, n_cards, home_lat, home_lon, card_issue, start_ts,
             window_s, n_merchants):
    """Stolen-card bust-out: a few larger CNP hits, ~half from a distant location.

    Amounts ($120 to $900) deliberately overlap legit big-ticket spend, so 'amount'
    alone cannot separate fraud.
    """
    out = []
    for cid in rng.integers(1, n_cards + 1, n):
        k = int(rng.integers(2, 6))
        t0 = rng.uniform(0, window_s - 7200)
        offsets = np.sort(rng.uniform(0, 7200, k))            # within a couple of hours
        if rng.random() < 0.5:                                # distant
            base_lat, base_lon = rng.uniform(25.0, 49.0), rng.uniform(-124.0, -67.0)
        else:                                                 # near home (CNP anywhere)
            base_lat, base_lon = home_lat[cid - 1], home_lon[cid - 1]
        out.append(pd.DataFrame({
            "card_id": cid,
            "merchant_id": rng.integers(1, n_merchants + 1, k),
            "mcc": rng.choice([5999, 5812, 4829], size=k),
            "amount": np.round(rng.uniform(120.0, 900.0, k), 2),
            "ts": start_ts + pd.to_timedelta(t0 + offsets, unit="s"),
            "lat": base_lat + rng.normal(0, 0.05, k),
            "lon": base_lon + rng.normal(0, 0.05, k),
            "channel": "CNP",
            "card_issue_date": card_issue[cid - 1],
            "is_fraud": 1,
        }))
    return out


def _takeover_spend(rng, n, n_cards, home_lat, home_lon, typical_amt, card_issue,
                    start_ts, window_s, n_merchants):
    """Account-takeover spend: CNP at 2-4x the card's norm, near home, unhurried.

    The HARD class. It looks like a slightly-big online purchase, so it overlaps
    legitimate behaviour and drives genuine false negatives, which is what makes the
    precision and recall trade-off real.
    """
    out = []
    for cid in rng.integers(1, n_cards + 1, n):
        k = int(rng.integers(2, 5))
        t0 = rng.uniform(0, window_s - 6 * 3600)
        offsets = np.sort(rng.uniform(0, 6 * 3600, k))        # spread over hours
        mult = rng.uniform(2.0, 4.0, k)
        amt = np.minimum(typical_amt[cid - 1] * mult, 400.0)
        out.append(pd.DataFrame({
            "card_id": cid,
            "merchant_id": rng.integers(1, n_merchants + 1, k),
            "mcc": rng.choice(MCC_CHOICES, size=k, p=MCC_PROBS),
            "amount": np.round(amt, 2),
            "ts": start_ts + pd.to_timedelta(t0 + offsets, unit="s"),
            "lat": home_lat[cid - 1] + rng.normal(0, 0.05, k),
            "lon": home_lon[cid - 1] + rng.normal(0, 0.05, k),
            "channel": "CNP",
            "card_issue_date": card_issue[cid - 1],
            "is_fraud": 1,
        }))
    return out


def time_split(df: pd.DataFrame, train_frac: float = 0.6, valid_frac: float = 0.2):
    """Chronological train/valid/test split.

    Fraud is temporal and adversarial: a random split leaks the future into the
    past and inflates every metric. We split strictly by time. Returns
    (train, valid, test) as copies plus the two cut timestamps.
    """
    df = df.sort_values("ts")
    t_train = df["ts"].quantile(train_frac)
    t_valid = df["ts"].quantile(train_frac + valid_frac)
    train = df[df["ts"] <= t_train].copy()
    valid = df[(df["ts"] > t_train) & (df["ts"] <= t_valid)].copy()
    test = df[df["ts"] > t_valid].copy()
    return train, valid, test, (t_train, t_valid)
