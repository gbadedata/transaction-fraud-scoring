"""Tests for IEEE-CIS entity resolution and leakage-safe ring features.

The fixture hand-builds a device shared across three cards and one account with
repeat activity, so the expected counts are unambiguous.
"""

import pandas as pd

from fraud.ieee_features import (
    add_ieee_features,
    build_device_fp,
    build_uid,
    ring_reason_codes,
    ring_rule_score,
)


def _frame():
    base = pd.Timestamp("2018-01-01 00:00:00")
    # (offset_hours, card1, DeviceInfo, P_emaildomain)
    rows = [
        (0, 1, "dev-share", "a.com"),   # account 1, first ever
        (1, 1, "dev-share", "a.com"),   # account 1, repeat within 24h
        (2, 2, "dev-share", "a.com"),   # different card, same device
        (3, 3, "dev-share", "b.com"),   # third card on the device
        (4, 9, None, "a.com"),          # no device -> not a shared entity
    ]
    df = pd.DataFrame({
        "TransactionID": range(1, len(rows) + 1),
        "is_fraud": [0, 0, 1, 1, 0],
        "amount": [100.0, 50.0, 200.0, 300.0, 25.0],
        "ts": [base + pd.Timedelta(hours=h) for h, *_ in rows],
        "card1": [r[1] for r in rows],
        "card2": [10, 10, 20, 30, 90],      # stable per card1
        "card3": 150.0,
        "card5": 102.0,
        "addr1": [100, 100, 200, 300, 900],
        "DeviceInfo": [r[2] for r in rows],
        "P_emaildomain": [r[3] for r in rows],
    })
    return df


def test_build_uid_groups_same_card_attributes():
    df = _frame()
    uid = build_uid(df)
    assert uid.iloc[0] == uid.iloc[1]      # same card1 + attrs -> same account
    assert uid.iloc[0] != uid.iloc[2]      # different card -> different account


def test_device_prior_cards_counts_distinct_cards_before():
    f, _ = add_ieee_features(_frame())
    f = f.sort_values("TransactionID")
    # device sees: {1}, then {1}, then {1}, then {1,2} distinct cards before each row
    assert f["device_prior_cards"].tolist() == [0, 1, 1, 2, 0]


def test_missing_device_is_not_a_ring():
    f, _ = add_ieee_features(_frame())
    f = f.sort_values("TransactionID")
    assert f["device_prior_cards"].iloc[4] == 0
    assert f["device_prior_txn"].iloc[4] == 0
    assert f["has_identity"].iloc[4] == 0
    assert f["has_identity"].iloc[0] == 1


def test_account_velocity_excludes_current_row():
    f, _ = add_ieee_features(_frame())
    f = f.sort_values("TransactionID")
    # account 1 (rows 0,1) within 24h: prior counts are 0 then 1
    assert f["uid_txn_24h"].iloc[0] == 0
    assert f["uid_txn_24h"].iloc[1] == 1
    assert f["uid_prior_txn"].iloc[0] == 0
    assert f["uid_prior_txn"].iloc[1] == 1


def test_email_domain_freq_counts_domain_volume():
    f, _ = add_ieee_features(_frame())
    f = f.sort_values("TransactionID")
    # 'a.com' appears on 4 rows (cards 1,1,2,9), 'b.com' on 1
    assert f.loc[f["card1"] == 3, "email_domain_freq"].iloc[0] == 1
    assert f["email_domain_freq"].iloc[0] == 4


def test_features_have_no_nans():
    f, cols = add_ieee_features(_frame())
    assert not f[cols].isna().any().any()
    assert "device_prior_cards" in cols and "uid_amount_z" in cols


def test_device_fp_uses_browser_and_screen():
    # Same DeviceInfo family but different browser must yield different fingerprints,
    # which is how a coarse "Windows" gets split into specific devices.
    df = pd.DataFrame({
        "DeviceInfo": ["Windows", "Windows"],
        "id_31": ["chrome", "safari"],
        "id_33": ["1920x1080", "1920x1080"],
    })
    fp = build_device_fp(df)
    assert fp.iloc[0] != fp.iloc[1]


def test_device_reason_gated_to_specific_fingerprints():
    # A device shared by many cards only earns a reason code when its fingerprint is
    # specific (rare). A common/generic fingerprint is excluded even at high counts.
    df = pd.DataFrame({
        "device_prior_cards": [8, 8, 0],
        "device_fp_freq": [0.0001, 0.20, 0.0001],
        "uid_txn_24h": [0, 0, 0],
    })
    reasons = ring_reason_codes(df, device_thr=4, generic_freq=0.002)
    assert "device shared by 8+ cards" in reasons[0]   # specific + shared -> fires
    assert reasons[1] == "model/anomaly only"          # common device -> gated out
    assert reasons[2] == "model/anomaly only"          # no sharing -> nothing


def test_ring_rule_score_ignores_common_devices():
    df = pd.DataFrame({
        "device_prior_cards": [10, 10],
        "device_fp_freq": [0.0001, 0.20],
        "uid_txn_24h": [0, 0],
    })
    s = ring_rule_score(df, generic_freq=0.002)
    assert s[0] > 0     # specific device contributes to the ring score
    assert s[1] == 0    # common device contributes nothing (no velocity either)
