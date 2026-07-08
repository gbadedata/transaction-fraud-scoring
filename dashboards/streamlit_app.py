"""Investigator triage dashboard (stub).

Run: streamlit run dashboards/streamlit_app.py

Shows what a fraud analyst uses: a queue ranked by expected loss, each alert with
a reason, plus how precision and fraud-value recall move as the review budget
changes. It runs on generated data. Point `build()` at a real fct_transactions
table to run it on live data.
"""

from __future__ import annotations

import streamlit as st

from fraud import data, features, metrics, model, rules, scoring

st.set_page_config(page_title="Fraud Triage", layout="wide")


@st.cache_data(show_spinner="Generating & featurising transactions…")
def build():
    df = features.add_features(data.generate_transactions())
    train, _valid, test, _ = data.time_split(df)
    return train, test


@st.cache_resource(show_spinner="Training scorer…")
def get_model(_train):
    return model.train_scorer(_train)


train, test = build()
scorer = get_model(train)

prob = model.score(scorer, test)
hits = rules.apply_rules(test)
anomaly = scoring.anomaly_scores(train, test)
blended = scoring.blend(prob, hits["rule_score"].to_numpy(), anomaly)
reasons = scoring.rule_reason_codes(hits, [r.name for r in rules.RULES])
queue = scoring.rank_queue(test, blended, reasons)

y = test["is_fraud"].to_numpy()
amt = test["amount"].to_numpy()

st.title("Transaction Fraud: Investigator Triage")
st.caption(
    "Queue ranked by expected loss (P(fraud) × amount). `is_fraud` is shown only "
    "because this is labelled demo data. At real triage time you would not have it."
)

cap = st.slider("Daily review capacity (alerts the team can clear)", 20, 500, 100, 10)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Alerts worked", f"{cap}")
c2.metric("Precision @ capacity", f"{metrics.precision_at_k(y, blended, cap):.2f}")
c3.metric("Fraud-$ recall @ capacity",
          f"{metrics.value_weighted_recall_at_k(y, blended, amt, cap):.0%}")
c4.metric("PR-AUC (ranking quality)", f"{metrics.pr_auc(y, blended):.3f}")

st.subheader("Investigator queue")
st.dataframe(
    queue.loc[: cap - 1, ["transaction_id", "card_id", "amount", "risk_score",
                          "expected_loss", "reason_codes", "is_fraud"]],
    use_container_width=True, hide_index=True,
)

st.subheader("Per-rule precision (does each rule earn its alert volume?)")
st.dataframe(rules.rule_precision(hits, test["is_fraud"]),
             use_container_width=True, hide_index=True)
