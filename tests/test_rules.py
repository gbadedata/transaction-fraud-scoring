"""Rule tests: each tripwire must fire on the pattern it targets and stay silent
on clean traffic. Crafted rows make the expected behaviour unambiguous."""

import pandas as pd

from fraud.rules import apply_rules, rule_precision


def _row(**overrides):
    base = dict(
        amount=25.0, txn_count_1h=0, amt_sum_1h=0.0, txn_count_24h=1,
        amt_sum_24h=25.0, amount_z=0.0, time_since_last_s=3600.0,
        geo_dist_km=0.0, geo_speed_kmh=0.0, mcc_novel=0, hour=12,
        is_night=0, is_cnp=0, card_age_days=400,
    )
    base.update(overrides)
    return base


def test_card_testing_fires_on_burst_of_tiny_cnp():
    df = pd.DataFrame([_row(txn_count_1h=6, amount=1.5, is_cnp=1)])
    hits = apply_rules(df)
    assert hits["card_testing"].iloc[0] == 1
    assert hits["rule_any"].iloc[0] == 1


def test_card_testing_silent_on_normal_purchase():
    df = pd.DataFrame([_row(txn_count_1h=1, amount=42.0, is_cnp=1)])
    assert apply_rules(df)["card_testing"].iloc[0] == 0


def test_high_amount_new_card_requires_all_conditions():
    fires = pd.DataFrame([_row(amount=300.0, is_cnp=1, card_age_days=10)])
    assert apply_rules(fires)["high_amount_new_card"].iloc[0] == 1
    old_card = pd.DataFrame([_row(amount=300.0, is_cnp=1, card_age_days=400)])
    assert apply_rules(old_card)["high_amount_new_card"].iloc[0] == 0


def test_impossible_travel_fires_above_threshold():
    df = pd.DataFrame([_row(geo_speed_kmh=1500.0)])
    assert apply_rules(df)["impossible_travel"].iloc[0] == 1


def test_rule_score_is_between_zero_and_one():
    df = pd.DataFrame([
        _row(),
        _row(txn_count_1h=6, amount=1.0, is_cnp=1, geo_speed_kmh=2000.0),
    ])
    scores = apply_rules(df)["rule_score"]
    assert (scores >= 0).all() and (scores <= 1).all()
    assert scores.iloc[1] > scores.iloc[0]


def test_rule_precision_shape_and_values():
    df = pd.DataFrame([
        _row(txn_count_1h=6, amount=1.0, is_cnp=1),   # fraud, fires card_testing
        _row(txn_count_1h=6, amount=1.0, is_cnp=1),   # legit, fires card_testing (a FP)
    ])
    y = pd.Series([1, 0])
    rep = apply_rules(df)
    prec = rule_precision(rep, y).set_index("rule")
    assert prec.loc["card_testing", "alerts"] == 2
    assert prec.loc["card_testing", "precision"] == 0.5
