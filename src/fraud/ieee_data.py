"""IEEE-CIS Fraud Detection: real loader and a schema-faithful mock.

The real dataset (https://www.kaggle.com/c/ieee-fraud-detection) has no clean
account id and no geolocation. It gives anonymised card attributes (card1..card6),
billing region (addr1/addr2), email domains, device fingerprints, and hundreds of
Vesta features. The interesting work is therefore *entity resolution* (reconstruct
an account from card1..card5 + addr1) and *ring detection* (accounts that share a
device, email, or address). See `ieee_features.py`.

`load_ieee` reads the real CSVs. `mock_ieee_frames` / `write_mock_ieee` produce a
small dataset with the same columns and injected rings, so the whole pipeline runs
and is tested without the multi-gigabyte download. The loader path is identical for
both: point it at the real files and nothing downstream changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# The competition encodes time as seconds from an undisclosed reference; the widely
# used convention is 2017-12-01. Only relative order matters for the pipeline.
REF_DATE = pd.Timestamp("2017-12-01")

EMAIL_DOMAINS = np.array([
    "gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "aol.com",
    "outlook.com", "icloud.com", "comcast.net",
])
PRODUCTCD = np.array(["W", "C", "R", "H", "S"])
CARD4 = np.array(["visa", "mastercard", "american express", "discover"])
CARD6 = np.array(["debit", "credit"])


def _finalise(tx: pd.DataFrame, idf: pd.DataFrame | None) -> pd.DataFrame:
    """Join identity, derive a timestamp, and add pipeline-standard columns."""
    if idf is not None:
        idf = idf.copy()
        idf.columns = [c.replace("-", "_") if c.startswith("id") else c
                       for c in idf.columns]
        df = tx.merge(idf, on="TransactionID", how="left")
    else:
        df = tx.copy()

    # Build derived columns in one shot; assigning them one at a time to a wide
    # (400+ column) frame triggers pandas fragmentation warnings.
    derived = {
        "ts": REF_DATE + pd.to_timedelta(df["TransactionDT"], unit="s"),
        "amount": df["TransactionAmt"].astype(float),
    }
    if "isFraud" in df.columns:
        derived["is_fraud"] = df["isFraud"].astype(int)
    df = pd.concat([df.reset_index(drop=True),
                    pd.DataFrame(derived).reset_index(drop=True)], axis=1)
    return df.sort_values("ts").reset_index(drop=True)


def load_ieee(transaction_path: str | Path, identity_path: str | Path | None = None,
              nrows: int | None = None) -> pd.DataFrame:
    """Load real IEEE-CIS CSVs (train_transaction[, train_identity])."""
    tx = pd.read_csv(transaction_path, nrows=nrows)
    idf = pd.read_csv(identity_path) if identity_path else None
    if idf is not None and nrows is not None:
        idf = idf[idf["TransactionID"].isin(tx["TransactionID"])]
    return _finalise(tx, idf)


def load_ieee_frames(tx: pd.DataFrame, idf: pd.DataFrame | None = None) -> pd.DataFrame:
    """Same as `load_ieee` but from in-memory frames (used by the mock and tests)."""
    return _finalise(tx, idf)


# --------------------------------------------------------------------------- mock

def mock_ieee_frames(n_cards: int = 4000, n_normal: int = 40_000, n_rings: int = 25,
                     ring_size: int = 16, n_bursts: int = 30, seed: int = 7):
    """Return (transaction_df, identity_df) mimicking IEEE-CIS, with fraud rings.

    Rings share one device (and often an email) across many distinct cards, which
    is the structural signal `ieee_features` is built to surface.
    """
    rng = np.random.default_rng(seed)
    window_s = 45 * 86_400
    max_card = 1000 + n_cards + n_rings * ring_size + 2

    # Stable per-card attributes (as in real IEEE, card2..card6 and addr1 belong to
    # the card, not the transaction). This is what makes account resolution possible.
    m = np.random.default_rng(seed + 999)
    A_card2 = m.integers(100, 600, max_card).astype(float)
    A_card3 = m.choice([150.0, 185.0], max_card)
    A_card4 = m.choice(CARD4, max_card, p=[0.6, 0.3, 0.06, 0.04])
    A_card5 = m.choice([102.0, 117.0, 126.0, 224.0], max_card)
    A_card6 = m.choice(CARD6, max_card, p=[0.7, 0.3])
    A_addr1 = m.integers(100, 500, max_card).astype(float)

    def base_block(card1, dt, amt, fraud, product=None, email=None, n=None):
        n = len(card1) if n is None else n
        c1 = card1.astype(int)
        return pd.DataFrame({
            "TransactionID": np.zeros(n, dtype=int),   # filled after concat
            "isFraud": fraud,
            "TransactionDT": dt.astype(int),
            "TransactionAmt": np.round(amt, 2),
            "ProductCD": product if product is not None
            else rng.choice(PRODUCTCD, n, p=[0.72, 0.12, 0.06, 0.06, 0.04]),
            "card1": c1,
            "card2": A_card2[c1],
            "card3": A_card3[c1],
            "card4": A_card4[c1],
            "card5": A_card5[c1],
            "card6": A_card6[c1],
            "addr1": A_addr1[c1],
            "addr2": np.full(n, 87.0),
            "dist1": rng.choice([np.nan, *range(0, 50)], n).astype(float),
            "P_emaildomain": email if email is not None
            else rng.choice(EMAIL_DOMAINS, n),
            "R_emaildomain": rng.choice([np.nan, *EMAIL_DOMAINS], n),
            **{f"C{i}": rng.integers(0, 8, n).astype(float) for i in range(1, 15)},
            **{f"D{i}": rng.choice([np.nan, *range(0, 200)], n).astype(float)
               for i in range(1, 16)},
            **{f"M{i}": rng.choice(["T", "F", np.nan], n) for i in range(1, 10)},
            # A handful of Vesta-style columns; the first few carry weak fraud signal
            # so the mock model has something to learn beyond the entity features.
            **{f"V{i}": rng.normal(0.0, 1.0, n)
               + (np.asarray(fraud, dtype=float) * 0.6 if i <= 6 else 0.0)
               for i in range(1, 21)},
        })

    blocks = []

    # --- normal traffic ---
    c1_normal = rng.integers(1000, 1000 + n_cards, n_normal)
    dt_normal = rng.uniform(0, window_s, n_normal)
    amt_normal = np.maximum(1.0, rng.lognormal(3.6, 0.7, n_normal))
    fr_normal = (rng.random(n_normal) < 0.015).astype(int)     # sparse background fraud
    blocks.append(base_block(c1_normal, dt_normal, amt_normal, fr_normal, n=n_normal))

    # --- fraud rings: distinct cards sharing one specific device (and often an email) ---
    next_card = 1000 + n_cards + 1
    for r in range(n_rings):
        cards = np.arange(next_card, next_card + ring_size)
        next_card += ring_size
        reps = rng.integers(1, 4, ring_size)
        c1 = np.repeat(cards, reps)
        n = len(c1)
        t0 = rng.uniform(0, window_s - 3 * 86_400)
        dt = t0 + np.sort(rng.uniform(0, 3 * 86_400, n))
        amt = rng.uniform(150.0, 900.0, n)
        share_email = rng.random() < 0.6
        email = np.full(n, f"ring{r}@securemail.cc") if share_email else None
        blocks.append(base_block(c1, dt, amt, np.ones(n, int),
                                 product=np.full(n, "C"), email=email, n=n))

    # --- single-card velocity bursts ---
    for _ in range(n_bursts):
        k = int(rng.integers(6, 15))
        card = int(rng.integers(1000, 1000 + n_cards))
        t0 = rng.uniform(0, window_s - 3600)
        dt = t0 + np.sort(rng.uniform(0, 3600, k))
        amt = rng.uniform(5.0, 60.0, k)
        blocks.append(base_block(np.full(k, card), dt, amt, np.ones(k, int),
                                 product=np.full(k, "C"), n=k))

    tx = pd.concat(blocks, ignore_index=True)
    tx = tx.sort_values("TransactionDT").reset_index(drop=True)
    tx["TransactionID"] = np.arange(1, len(tx) + 1)

    # --- identity table: present for ~30% of normal rows and for all ring/burst rows ---
    is_fraud = tx["isFraud"].to_numpy() == 1
    keep = is_fraud | (rng.random(len(tx)) < 0.30)
    idx = np.where(keep)[0]

    # Normal devices are generic OS/browser families, so many cards share the same
    # low-cardinality fingerprint (as in the real data). This is the confound that
    # makes raw "cards per device" meaningless on its own.
    generic = np.array(["Windows", "iOS Device", "MacOS", "Trident/7.0", "rv:11.0"])
    device = pd.Series(rng.choice(generic, len(idx)), index=idx)
    id31 = pd.Series(rng.choice(["chrome", "safari", "ie"], len(idx)), index=idx)
    id33 = pd.Series(rng.choice(["1920x1080", "1366x768"], len(idx)), index=idx)

    ring_lookup = {}
    start = 1000 + n_cards + 1
    for r in range(n_rings):
        cset = set(range(start, start + ring_size))
        start += ring_size
        ring_lookup[r] = cset
    tx_card = tx["card1"].to_numpy()
    # Ring rows share one *specific*, rare fingerprint (device + browser + screen),
    # which is the genuine signal the gated ring features are meant to catch.
    for r, cset in ring_lookup.items():
        rows = idx[np.isin(tx_card[idx], list(cset))]
        device.loc[rows] = f"SM-G9{r:02d}0V Build/RING{r:02d}"
        id31.loc[rows] = "mobile safari"
        id33.loc[rows] = f"07{r:02d}x1334"

    idf = pd.DataFrame({
        "TransactionID": tx["TransactionID"].to_numpy()[idx],
        "DeviceType": rng.choice(["desktop", "mobile"], len(idx)),
        "DeviceInfo": device.to_numpy(),
        "id_01": rng.integers(-100, 0, len(idx)).astype(float),
        "id_02": rng.integers(1, 500000, len(idx)).astype(float),
        "id_31": id31.to_numpy(),
        "id_33": id33.to_numpy(),
    })

    tx = tx.drop(columns=[c for c in ["isFraud_x"] if c in tx.columns])
    return tx, idf


def write_mock_ieee(out_dir: str | Path, **kwargs) -> tuple[Path, Path]:
    """Write mock CSVs shaped like train_transaction.csv / train_identity.csv."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tx, idf = mock_ieee_frames(**kwargs)
    tx_path, id_path = out / "train_transaction.csv", out / "train_identity.csv"
    tx.to_csv(tx_path, index=False)
    idf.to_csv(id_path, index=False)
    return tx_path, id_path
