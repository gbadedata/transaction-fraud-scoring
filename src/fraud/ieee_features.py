"""IEEE-CIS features: entity resolution, leakage-safe velocity, ring detection.

There is no account id in IEEE-CIS, so we build one. `uid` groups transactions that
share the stable card attributes (card1, card2, card3, card5) and billing region
(addr1); this approximates a card/account and is the standard technique for this
dataset. On top of that we compute:

  * account velocity and amount deviation, per uid, using prior rows only
  * ring signals: how many distinct cards have shared this device / email / address
    *before* the current transaction

The ring features are structural counts, not fraud rates, so they never touch the
label and cannot leak. `tests/test_ieee_features.py` verifies the leakage guard and
the ring counts on a hand-built fixture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_DAY_NS = 86_400 * 1_000_000_000
_M_COLS = [f"M{i}" for i in range(1, 10)]
_C_COLS = [f"C{i}" for i in range(1, 15)]
_D_COLS = [f"D{i}" for i in range(1, 6)]


def build_uid(df: pd.DataFrame) -> pd.Series:
    """Reconstruct an account key from stable card attributes + billing region."""
    parts = ["card1", "card2", "card3", "card5", "addr1"]
    present = [c for c in parts if c in df.columns]
    key = df[present].astype("string").fillna("na")
    return key.agg("-".join, axis=1)


def _prior_count(df: pd.DataFrame, key: pd.Series) -> np.ndarray:
    """Rows sharing `key` strictly before the current row (df in time order)."""
    return key.groupby(key).cumcount().to_numpy()


def _prior_distinct(df: pd.DataFrame, key: pd.Series, value: pd.Series) -> np.ndarray:
    """Distinct `value` seen within `key` strictly before the current row."""
    frame = pd.DataFrame({"k": key.to_numpy(), "v": value.to_numpy()})
    first = ~frame.duplicated(["k", "v"])                 # first time this pair appears
    incl = first.groupby(frame["k"]).cumsum().to_numpy()  # distinct-so-far incl. current
    return incl - first.to_numpy().astype(int)            # exclude current


def _uid_dynamics(df: pd.DataFrame, uid: pd.Series):
    """Per-uid leakage-safe velocity, time-since-last, and amount z-score."""
    n = len(df)
    cnt24 = np.zeros(n)
    sum24 = np.zeros(n)
    tsl = np.full(n, 30 * 86_400.0)
    amt_z = np.zeros(n)

    ts_all = df["ts"].to_numpy("datetime64[ns]").astype("int64")
    amt_all = df["amount"].to_numpy(float)
    pos_index = np.arange(n)

    for _, g in pd.Series(pos_index).groupby(uid.to_numpy(), sort=False):
        pos = g.to_numpy()
        if len(pos) == 1:
            continue
        ts = ts_all[pos]
        amt = amt_all[pos]
        idx = np.arange(len(pos))

        left = np.searchsorted(ts, ts - _DAY_NS, side="left")
        pref = np.concatenate([[0.0], np.cumsum(amt)])
        cnt24[pos] = idx - left
        sum24[pos] = pref[idx] - pref[left]

        gap = np.diff(ts, prepend=ts[0]).astype(float) / 1e9
        gap[0] = 30 * 86_400.0
        tsl[pos] = gap

        n_prior = idx
        csum = pref[idx]
        csq = np.concatenate([[0.0], np.cumsum(amt ** 2)])[idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_prior = np.where(n_prior > 0, csum / np.maximum(n_prior, 1), np.nan)
            var_prior = np.where(n_prior > 1,
                                 csq / np.maximum(n_prior, 1) - mean_prior ** 2, np.nan)
            std_prior = np.sqrt(np.clip(var_prior, 0.0, None))
            z = np.where(std_prior > 1e-9, (amt - mean_prior) / std_prior, 0.0)
        amt_z[pos] = np.nan_to_num(z, nan=0.0)

    return cnt24, sum24, tsl, amt_z


def add_ieee_features(df: pd.DataFrame):
    """Return (df_with_features, feature_cols) for the IEEE-CIS pipeline."""
    df = df.sort_values("ts").reset_index(drop=True)

    uid = build_uid(df)
    device = df.get("DeviceInfo", pd.Series(index=df.index, dtype="object")).fillna("missing")
    email = df.get("P_emaildomain", pd.Series(index=df.index, dtype="object")).fillna("missing")
    card1 = df["card1"]

    cnt24, sum24, tsl, amt_z = _uid_dynamics(df, uid)

    # An absent device is not a shared entity: mask those rows to 0 so the large
    # "missing" bucket cannot masquerade as a giant ring.
    device_valid = (df["DeviceInfo"].notna().to_numpy()
                    if "DeviceInfo" in df.columns else np.zeros(len(df), bool))
    email_valid = (df["P_emaildomain"].notna().to_numpy()
                   if "P_emaildomain" in df.columns else np.zeros(len(df), bool))

    # Email domain and billing region are low-cardinality (everyone uses a handful of
    # domains), so "cards per domain" is not a ring signal. Domain *rarity* is: a
    # little-used domain shared by fraud is meaningful. Device is high-cardinality, so
    # cards-per-device is the primary ring signal.
    email_freq = email.map(email.value_counts()).to_numpy(float) * email_valid

    feats = {
        # amount and time
        "amount_log": np.log1p(df["amount"].to_numpy(float)),
        "amount_cents": (df["amount"].to_numpy(float) % 1.0),
        "hour": df["ts"].dt.hour.to_numpy(),
        "dayofweek": df["ts"].dt.dayofweek.to_numpy(),
        "is_night": ((df["ts"].dt.hour >= 0) & (df["ts"].dt.hour <= 5)).astype(int).to_numpy(),
        # account (uid) dynamics
        "uid_txn_24h": cnt24,
        "uid_amt_24h": sum24,
        "uid_tsl_log": np.log1p(tsl),
        "uid_amount_z": amt_z,
        "uid_prior_txn": _prior_count(df, uid),
        # ring / entity structure (leakage-safe counts)
        "device_prior_txn": _prior_count(df, device) * device_valid,
        "device_prior_cards": _prior_distinct(df, device, card1) * device_valid,
        "email_domain_freq": email_freq,
        "card1_prior_txn": _prior_count(df, card1),
        # has an identity record at all (rare in normal traffic)
        "has_identity": device_valid.astype(int),
        "dist1": df.get("dist1", pd.Series(np.nan, index=df.index)).to_numpy(float),
    }

    # low-cardinality categoricals -> one-hot / codes
    if "ProductCD" in df.columns:
        for val in ["W", "C", "R", "H", "S"]:
            feats[f"pcd_{val}"] = (df["ProductCD"] == val).astype(int).to_numpy()
    for col in ["card4", "card6", "DeviceType"]:
        if col in df.columns:
            feats[f"{col}_code"] = pd.factorize(df[col])[0]
    for col in _M_COLS:
        if col in df.columns:
            feats[f"{col}_code"] = pd.factorize(df[col])[0]
    for col in _C_COLS + _D_COLS:
        if col in df.columns:
            feats[col] = df[col].to_numpy(float)

    out = df.copy()
    feature_cols = []
    for name, arr in feats.items():
        out[name] = arr
        feature_cols.append(name)

    out[feature_cols] = out[feature_cols].fillna(-1.0)
    return out, feature_cols


# --- explainability: why is this in the queue? -----------------------------------

def ring_reason_codes(df: pd.DataFrame, device_thr: int = 4,
                      velocity_thr: int = 5) -> list[str]:
    """Human-readable reasons from the ring/velocity structure."""
    reasons = []
    for _, r in df.iterrows():
        tags = []
        if r.get("device_prior_cards", 0) >= device_thr:
            tags.append(f"device shared by {int(r['device_prior_cards'])}+ cards")
        if r.get("uid_txn_24h", 0) >= velocity_thr:
            tags.append("account velocity")
        reasons.append("; ".join(tags) or "model/anomaly only")
    return reasons


def ring_rule_score(df: pd.DataFrame) -> np.ndarray:
    """Soft 0..1 ring signal for blending, from the structural counts."""
    dev = np.clip(df["device_prior_cards"].to_numpy(float) / 10.0, 0, 1)
    vel = np.clip(df["uid_txn_24h"].to_numpy(float) / 10.0, 0, 1)
    return np.maximum(dev, vel)
