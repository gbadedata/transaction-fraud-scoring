"""Leakage-safe feature engineering.

THE core discipline in fraud modelling: every feature for a transaction may use
information available *strictly before* that transaction only. Using anything
dated at-or-after the event (later transactions, the outcome, post-auth fields)
is target leakage and is the number-one way fraud models cheat. All velocity and
history features below are computed as-of the transaction time, excluding the
current row. `tests/test_features.py` locks this behaviour down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "amount",
    "txn_count_1h", "amt_sum_1h",
    "txn_count_24h", "amt_sum_24h",
    "amount_z",
    "time_since_last_s",
    "geo_dist_km", "geo_speed_kmh",
    "mcc_novel",
    "hour", "is_night", "is_cnp",
    "card_age_days",
]

_HOUR = 3_600 * 1_000_000_000          # nanoseconds
_DAY = 86_400 * 1_000_000_000


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return `df` with the engineered feature columns added.

    Features are computed per card, in time order, using only prior transactions.
    """
    df = df.sort_values(["card_id", "ts"]).reset_index(drop=True)
    parts = [_features_for_card(g) for _, g in df.groupby("card_id", sort=False)]
    out = pd.concat(parts).sort_index()
    return out


def _features_for_card(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("ts")
    ts = g["ts"].to_numpy("datetime64[ns]").astype("int64")
    amt = g["amount"].to_numpy(float)
    n = len(g)
    idx = np.arange(n)

    csum = np.concatenate([[0.0], np.cumsum(amt)])   # prefix sums for O(1) windows
    csq = np.concatenate([[0.0], np.cumsum(amt ** 2)])

    def window(width_ns):
        left = np.searchsorted(ts, ts - width_ns, side="left")
        count = idx - left                            # prior txns in window (excl. current)
        amt_sum = csum[idx] - csum[left]
        return count.astype(float), amt_sum

    c1h, s1h = window(_HOUR)
    c24h, s24h = window(_DAY)

    # amount z-score vs the card's own prior history
    n_prior = idx
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_prior = np.where(n_prior > 0, csum[idx] / np.maximum(n_prior, 1), np.nan)
        var_prior = np.where(
            n_prior > 1,
            csq[idx] / np.maximum(n_prior, 1) - mean_prior ** 2,
            np.nan,
        )
        std_prior = np.sqrt(np.clip(var_prior, 0.0, None))
        amount_z = np.where(std_prior > 1e-9, (amt - mean_prior) / std_prior, 0.0)
    amount_z = np.nan_to_num(amount_z, nan=0.0)

    # time since previous transaction (first ever -> treated as a long gap)
    tsl = np.diff(ts, prepend=ts[0]).astype(float) / 1e9
    if n:
        tsl[0] = 30 * 86_400.0

    # geo distance & implied speed from the previous transaction
    lat, lon = g["lat"].to_numpy(float), g["lon"].to_numpy(float)
    prev_lat, prev_lon = np.roll(lat, 1), np.roll(lon, 1)
    dist = haversine_km(prev_lat, prev_lon, lat, lon)
    if n:
        dist[0] = 0.0
    hours = np.maximum(tsl / 3600.0, 1.0 / 60.0)
    speed = dist / hours
    if n:
        speed[0] = 0.0

    novel = (~g["mcc"].duplicated().to_numpy()).astype(int)   # first time card sees this MCC
    hour = g["ts"].dt.hour.to_numpy()
    is_night = ((hour >= 0) & (hour <= 5)).astype(int)
    is_cnp = (g["channel"].to_numpy() == "CNP").astype(int)
    card_age = (g["ts"] - g["card_issue_date"]).dt.days.to_numpy(float)

    return g.assign(
        txn_count_1h=c1h, amt_sum_1h=s1h,
        txn_count_24h=c24h, amt_sum_24h=s24h,
        amount_z=amount_z,
        time_since_last_s=tsl,
        geo_dist_km=dist, geo_speed_kmh=speed,
        mcc_novel=novel,
        hour=hour, is_night=is_night, is_cnp=is_cnp,
        card_age_days=card_age,
    )
