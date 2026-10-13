import numpy as np
import pandas as pd

PLANS = {"basic": 10.0, "plus": 25.0, "team": 80.0}


def generate_mrr_snapshots(n_customers=4000, months=24, seed=21):
    """Monthly recurring revenue per customer: signups, upgrades, downgrades and churn."""
    rng = np.random.default_rng(seed)
    plan_names = list(PLANS)
    start_month = np.sort(rng.integers(0, months, n_customers))
    # more customers join late (growing business)
    start_month = np.sort((months * rng.beta(1.6, 1.0, n_customers)).astype(int))
    start_month = np.clip(start_month, 0, months - 1)

    rows = []
    for cid, m0 in enumerate(start_month):
        plan = rng.choice(plan_names, p=[0.6, 0.3, 0.1])
        churn_p = {"basic": 0.055, "plus": 0.035, "team": 0.02}
        for m in range(m0, months):
            rows.append((cid, m, plan, PLANS[plan]))
            r = rng.random()
            if r < churn_p[plan]:
                break
            if r < churn_p[plan] + 0.03 and plan != "team":
                plan = plan_names[plan_names.index(plan) + 1]
            elif r < churn_p[plan] + 0.03 + 0.012 and plan != "basic":
                plan = plan_names[plan_names.index(plan) - 1]
    df = pd.DataFrame(rows, columns=["customer_id", "month_idx", "plan", "mrr"])
    df["month"] = pd.Timestamp("2023-01-01") + pd.to_timedelta(df["month_idx"] * 30.4375, unit="D")
    df["month"] = df["month"].dt.to_period("M")
    return df
