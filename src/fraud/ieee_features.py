"""IEEE-CIS features: entity resolution, leakage-safe velocity, ring detection.

There is no account id in IEEE-CIS, so we build one. `uid` groups transactions that
share the stable card attributes (card1, card2, card3, card5) and billing region
(addr1); this approximates a card/account and is the standard technique for this
dataset. On top of that we compute per-account velocity and amount deviation, all
from prior rows only, plus the model's bulk signal: the Vesta V/C/D columns, the
identity fields, and frequency encodings.

On device rings, one honest lesson from real data. `DeviceInfo` on its own is not a
fingerprint: its common values are OS and browser families ("Windows", "iOS Device")
shared by huge numbers of legitimate users, so raw "cards per device" is dominated by
popularity, not fraud. We therefore (1) build a more specific fingerprint from
`DeviceInfo` plus browser and screen resolution, (2) keep the raw shared-card count as
a model input alongside the fingerprint's overall frequency, so the model can tell a
rare shared fingerprint from a common one, and (3) gate the human-facing reason code
and the ring rule so they only fire on *specific* fingerprints. This stops the queue
ever saying "device shared by 4000+ cards" about a Windows machine.

Ring features are structural counts, never fraud rates, so they cannot leak the label.
`tests/test_ieee_features.py` pins down the leakage guards, the ring counts, and the
device gating.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_DAY_NS = 86_400 * 1_000_000_000
_M_COLS = [f"M{i}" for i in range(1, 10)]
_C_COLS = [f"C{i}" for i in range(1, 15)]
_D_COLS = [f"D{i}" for i in range(1, 16)]
_D_NORM = ["D1", "D2", "D3", "D4", "D5", "D10", "D11", "D15"]
# Fraction of all transactions above which a device fingerprint is treated as a
# common/generic device (an OS or browser family) rather than a ring signal.
GENERIC_DEVICE_FREQ = 0.002


def build_uid(df: pd.DataFrame) -> pd.Series:
    """Reconstruct an account key from stable card attributes + billing region."""
    parts = ["card1", "card2", "card3", "card5", "addr1"]
    present = [c for c in parts if c in df.columns]
    key = df[present].astype("string").fillna("na")
    return key.agg("-".join, axis=1)


def build_device_fp(df: pd.DataFrame) -> pd.Series:
    """A more specific device fingerprint than DeviceInfo alone.

    Adds browser (id_31) and screen resolution (id_33) where present, which splits a
    coarse family like "Windows" into far more specific fingerprints.
    """
    base = df.get("DeviceInfo", pd.Series(index=df.index, dtype="object"))
    fp = base.astype("string").fillna("na")
    for col in ("id_31", "id_33"):
        if col in df.columns:
            fp = fp.str.cat(df[col].astype("string").fillna("na"), sep="|")
    return fp


def _prior_count(key: pd.Series) -> np.ndarray:
    """Rows sharing `key` strictly before the current row (df in time order)."""
    return key.groupby(key).cumcount().to_numpy()


def _prior_distinct(key: pd.Series, value: pd.Series) -> np.ndarray:
    """Distinct `value` seen within `key` strictly before the current row."""
    frame = pd.DataFrame({"k": key.to_numpy(), "v": value.to_numpy()})
    first = ~frame.duplicated(["k", "v"])                 # first time this pair appears
    incl = first.groupby(frame["k"]).cumsum().to_numpy()  # distinct-so-far incl. current
    return incl - first.to_numpy().astype(int)            # exclude current


def _uid_dynamics(df: pd.DataFrame, uid: pd.Series):
    """Per-uid leakage-safe velocity, time-since-last, and amount z-score.

    Only multi-transaction accounts enter the loop; singletons keep the defaults
    (no prior activity), which keeps this fast on hundreds of thousands of accounts.
    """
    n = len(df)
    cnt24 = np.zeros(n)
    sum24 = np.zeros(n)
    tsl = np.full(n, 30 * 86_400.0)
    amt_z = np.zeros(n)

    ts_all = df["ts"].to_numpy("datetime64[ns]").astype("int64")
    amt_all = df["amount"].to_numpy(float)
    pos_index = np.arange(n)

    dup = uid.duplicated(keep=False).to_numpy()
    for _, g in pd.Series(pos_index[dup]).groupby(uid.to_numpy()[dup], sort=False):
        pos = g.to_numpy()
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

        csum = pref[idx]
        csq = np.concatenate([[0.0], np.cumsum(amt ** 2)])[idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_prior = np.where(idx > 0, csum / np.maximum(idx, 1), np.nan)
            var_prior = np.where(idx > 1,
                                 csq / np.maximum(idx, 1) - mean_prior ** 2, np.nan)
            std_prior = np.sqrt(np.clip(var_prior, 0.0, None))
            z = np.where(std_prior > 1e-9, (amt - mean_prior) / std_prior, 0.0)
        amt_z[pos] = np.nan_to_num(z, nan=0.0)

    return cnt24, sum24, tsl, amt_z


def _present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def add_ieee_features(df: pd.DataFrame):
    """Return (df_with_features, feature_cols) for the IEEE-CIS pipeline."""
    df = df.sort_values("ts").reset_index(drop=True)
    n = len(df)

    uid = build_uid(df)
    fp = build_device_fp(df)
    email = df.get("P_emaildomain", pd.Series(index=df.index, dtype="object")).fillna("missing")
    card1 = df["card1"]

    device_valid = (df["DeviceInfo"].notna().to_numpy()
                    if "DeviceInfo" in df.columns else np.zeros(n, bool))
    email_valid = (df["P_emaildomain"].notna().to_numpy()
                   if "P_emaildomain" in df.columns else np.zeros(n, bool))

    cnt24, sum24, tsl, amt_z = _uid_dynamics(df, uid)

    feats = {
        # amount and time
        "amount_log": np.log1p(df["amount"].to_numpy(float)),
        "amount_cents": (df["amount"].to_numpy(float) % 1.0),
        "hour": df["ts"].dt.hour.to_numpy(),
        "dayofweek": df["ts"].dt.dayofweek.to_numpy(),
        "is_night": ((df["ts"].dt.hour >= 0) & (df["ts"].dt.hour <= 5)).astype(int).to_numpy(),
        # account (uid) dynamics, leakage-safe
        "uid_txn_24h": cnt24,
        "uid_amt_24h": sum24,
        "uid_tsl_log": np.log1p(tsl),
        "uid_amount_z": amt_z,
        "uid_prior_txn": _prior_count(uid),
        # device fingerprint: raw shared-card count plus how common the fingerprint is,
        # so the model can separate a rare shared fingerprint from a popular OS family.
        "device_prior_txn": _prior_count(fp) * device_valid,
        "device_prior_cards": _prior_distinct(fp, card1) * device_valid,
        "device_fp_freq": (fp.map(fp.value_counts()).to_numpy(float) / n) * device_valid,
        # frequency / rarity signals (label-free)
        "email_domain_freq": email.map(email.value_counts()).to_numpy(float) * email_valid,
        "card1_freq": card1.map(card1.value_counts()).to_numpy(float),
        "card1_prior_txn": _prior_count(card1),
        "has_identity": device_valid.astype(int),
    }
    if "addr1" in df.columns:
        a1 = df["addr1"]
        feats["addr1_freq"] = a1.map(a1.value_counts()).to_numpy(float)

    # --- stronger entity features: a D1-anchored client key -----------------------
    # D1 behaves like a registration date, so `D1 - day` is roughly constant per
    # client. Anchoring the key on it groups a client's transactions far more tightly
    # than card attributes alone, which is the strong signal on this dataset.
    amt = df["amount"].to_numpy(float)
    a1_str = (df["addr1"] if "addr1" in df.columns
              else pd.Series(index=df.index, dtype="object")).astype("string").fillna("na")
    if "D1" in df.columns and "TransactionDT" in df.columns:
        day = df["TransactionDT"].to_numpy(float) // 86_400
        d1n = pd.Series(df["D1"].to_numpy(float) - day, index=df.index).round(0)
    else:
        d1n = pd.Series(np.zeros(n), index=df.index)
    client = card1.astype("string").fillna("na").str.cat(
        [a1_str, d1n.astype("string")], sep="_")
    feats["client_prior_txn"] = client.groupby(client).cumcount().to_numpy()
    feats["client_freq"] = client.map(client.value_counts()).to_numpy(float)

    # Amount relative to the card's own prior history (strictly-before, vectorised).
    gp = df.groupby("card1")["amount"]
    prior_sum = gp.cumsum().to_numpy() - amt
    prior_cnt = gp.cumcount().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        card1_prior_mean = np.where(prior_cnt > 0, prior_sum / np.maximum(prior_cnt, 1), np.nan)
        feats["card1_prior_amt_mean"] = card1_prior_mean
        feats["amt_to_card1_mean"] = np.where(card1_prior_mean > 0, amt / card1_prior_mean, 1.0)

    # More label-free frequency encodings across card fields and a key combination.
    for c in ["card2", "card3", "card5"]:
        if c in df.columns:
            s = df[c]
            feats[f"{c}_freq"] = s.map(s.value_counts()).to_numpy(float)
    if "addr1" in df.columns:
        combo = card1.astype("string").fillna("na").str.cat(a1_str, sep="_")
        feats["card1_addr1_freq"] = combo.map(combo.value_counts()).to_numpy(float)

    # Vesta and identity numeric columns: the bulk of the model's signal on IEEE.
    v_cols = sorted((c for c in df.columns if re.fullmatch(r"V\d+", c)),
                    key=lambda c: int(c[1:]))
    id_num = [c for c in df.columns
              if c.startswith("id_") and pd.api.types.is_numeric_dtype(df[c])]
    id_cat = [c for c in df.columns
              if c.startswith("id_") and not pd.api.types.is_numeric_dtype(df[c])]
    passthrough = _present(df, _C_COLS + _D_COLS + ["dist1", "dist2"]) + v_cols + id_num
    for c in passthrough:
        feats[c] = df[c].to_numpy(float)

    # De-trended timedeltas: Dn minus the transaction day is more stable over time.
    if "TransactionDT" in df.columns:
        day = df["TransactionDT"].to_numpy(float) // 86_400
        for c in _present(df, _D_NORM):
            feats[f"{c}_norm"] = df[c].to_numpy(float) - day

    # low-cardinality categoricals -> one-hot / integer codes
    if "ProductCD" in df.columns:
        for val in ["W", "C", "R", "H", "S"]:
            feats[f"pcd_{val}"] = (df["ProductCD"] == val).astype(int).to_numpy()
    for col in ["card4", "card6", "DeviceType", *_M_COLS, *id_cat]:
        if col in df.columns:
            feats[f"{col}_code"] = pd.factorize(df[col])[0]

    feat_df = pd.DataFrame(feats, index=df.index).fillna(-1.0)
    out = pd.concat([df, feat_df], axis=1)
    # Passthrough columns (C/D/V/dist) live in both `df` and the filled feature block;
    # keep the filled copies so the model is not handed the same column twice.
    out = out.loc[:, ~out.columns.duplicated(keep="last")]
    return out, list(feat_df.columns)


# --- explainability and a structural rule ----------------------------------------

def _specific_device(df: pd.DataFrame, generic_freq: float) -> np.ndarray:
    """True where the device fingerprint is specific enough to mean something."""
    zeros = pd.Series(np.zeros(len(df)), index=df.index)
    freq = df.get("device_fp_freq", zeros).to_numpy(float)
    cards = df.get("device_prior_cards", zeros).to_numpy(float)
    return (cards > 0) & (freq > 0) & (freq < generic_freq)


def ring_reason_codes(df: pd.DataFrame, device_thr: int = 4, velocity_thr: int = 5,
                      generic_freq: float = GENERIC_DEVICE_FREQ) -> list[str]:
    """Human-readable reasons, with the device reason gated to specific fingerprints."""
    specific = _specific_device(df, generic_freq)
    reasons = []
    for i, (_, r) in enumerate(df.iterrows()):
        tags = []
        if specific[i] and r.get("device_prior_cards", 0) >= device_thr:
            tags.append(f"device shared by {int(r['device_prior_cards'])}+ cards")
        if r.get("uid_txn_24h", 0) >= velocity_thr:
            tags.append("account velocity")
        reasons.append("; ".join(tags) or "model/anomaly only")
    return reasons


def ring_rule_score(df: pd.DataFrame,
                    generic_freq: float = GENERIC_DEVICE_FREQ) -> np.ndarray:
    """Soft 0..1 ring signal for blending; the device part ignores common devices."""
    specific = _specific_device(df, generic_freq)
    dev = np.where(specific, np.clip(df["device_prior_cards"].to_numpy(float) / 10.0, 0, 1), 0.0)
    vel = np.clip(df["uid_txn_24h"].to_numpy(float) / 10.0, 0, 1)
    return np.maximum(dev, vel)
