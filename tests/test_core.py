import pandas as pd
import pytest

from src import core, data


@pytest.fixture(scope="module")
def snap():
    return data.generate_mrr_snapshots(n_customers=600, months=12, seed=4)


def test_waterfall_balances(snap):
    wf = core.mrr_waterfall(snap)
    recon = wf["opening_mrr"] + wf["new"] + wf["expansion"] + wf["contraction"] + wf["churned"]
    assert (recon - wf["closing_mrr"]).abs().max() < 1e-6


def test_hand_built_example():
    snap = pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 2, 3],
            "month_idx": [0, 1, 0, 1, 1],
            "plan": ["basic", "plus", "basic", "basic", "basic"],
            "mrr": [10.0, 25.0, 10.0, 0.0, 10.0],
        }
    )
    snap = snap[snap["mrr"] > 0]
    wf = core.mrr_waterfall(snap)
    row = wf.loc[1]
    assert row["opening_mrr"] == 20.0
    assert row["new"] == 10.0
    assert row["expansion"] == 15.0
    assert row["churned"] == -10.0
    assert row["closing_mrr"] == 35.0


def test_customer_counts_consistent(snap):
    c = core.customer_counts(snap)
    assert (c["active"].diff().dropna() == (c["new_customers"] - c["lost_customers"])[1:]).all()


def test_retention_bounds(snap):
    r = core.revenue_retention(core.mrr_waterfall(snap))
    assert (r["grr"] <= 1.0 + 1e-9).all()
    assert (r["nrr"] >= r["grr"] - 1e-9).all()


def test_ltv_is_positive(snap):
    counts = core.customer_counts(snap)
    out = core.ltv(snap, counts, window=4)
    assert out["ltv"] > 0 and 0 < out["monthly_churn"] < 1
