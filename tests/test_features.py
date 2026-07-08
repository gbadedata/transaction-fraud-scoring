"""Feature tests. The most important tests in the repo.

They pin down that velocity and history features look only at the past (no
leakage) and that geo, time, and novelty features compute correctly on a fixture.
"""

import pandas as pd

from fraud.features import add_features, haversine_km


def _one_card_frame():
    base = pd.Timestamp("2024-01-01 00:00:00")
    rows = [
        # t=0    first txn, no history
        (1, 1, 10, 5411, 20.0, base, 40.0, -74.0, "CP"),
        # +10 min, same place
        (2, 1, 11, 5411, 25.0, base + pd.Timedelta(minutes=10), 40.0, -74.0, "CP"),
        # +30 min, same place, new MCC
        (3, 1, 12, 5812, 30.0, base + pd.Timedelta(minutes=30), 40.0, -74.0, "CNP"),
        # +2h, far away (impossible travel)
        (4, 1, 13, 5999, 500.0, base + pd.Timedelta(hours=2), 34.0, -118.0, "CNP"),
    ]
    df = pd.DataFrame(rows, columns=[
        "transaction_id", "card_id", "merchant_id", "mcc", "amount",
        "ts", "lat", "lon", "channel",
    ])
    df["card_issue_date"] = base - pd.Timedelta(days=100)
    df["is_fraud"] = 0
    return df


def test_velocity_excludes_current_and_counts_only_past():
    f = add_features(_one_card_frame()).sort_values("transaction_id")
    counts_1h = f["txn_count_1h"].tolist()
    # txn1: 0 prior; txn2: 1 prior; txn3: 2 prior (all within 1h); txn4: 0 prior in last hour
    assert counts_1h == [0.0, 1.0, 2.0, 0.0]
    # amount summed over prior only, never includes the current row
    assert f["amt_sum_1h"].iloc[1] == 20.0
    assert f["amt_sum_1h"].iloc[2] == 45.0


def test_time_since_last_and_first_row_is_large_gap():
    f = add_features(_one_card_frame()).sort_values("transaction_id")
    tsl = f["time_since_last_s"].tolist()
    assert tsl[0] > 1_000_000            # first ever txn -> long-gap sentinel
    assert tsl[1] == 600                 # 10 minutes
    assert tsl[2] == 1200                # 20 minutes after txn2


def test_mcc_novelty_flags_first_use_only():
    f = add_features(_one_card_frame()).sort_values("transaction_id")
    # 5411 (new), 5411 (seen), 5812 (new), 5999 (new)
    assert f["mcc_novel"].tolist() == [1, 0, 1, 1]


def test_impossible_travel_speed_is_high():
    f = add_features(_one_card_frame()).sort_values("transaction_id")
    # NY -> LA in ~1.5h implies an impossible ground speed
    assert f["geo_speed_kmh"].iloc[3] > 900


def test_channel_and_card_age():
    f = add_features(_one_card_frame()).sort_values("transaction_id")
    assert f["is_cnp"].tolist() == [0, 0, 1, 1]
    assert (f["card_age_days"] >= 100).all()


def test_haversine_known_distance():
    # NYC -> LAX is ~3,940 km
    d = haversine_km(40.6413, -73.7781, 33.9416, -118.4085)
    assert 3800 < d < 4050
