import pandas as pd


def mrr_waterfall(snapshots):
    """Split the change in MRR between consecutive months into its components."""
    wide = snapshots.pivot(index="customer_id", columns="month_idx", values="mrr").fillna(0.0)
    months = list(wide.columns)
    rows = []
    for prev, cur in zip(months[:-1], months[1:]):
        a, b = wide[prev], wide[cur]
        new = b[(a == 0) & (b > 0)].sum()
        churned = -a[(a > 0) & (b == 0)].sum()
        both = (a > 0) & (b > 0)
        delta = (b - a)[both]
        rows.append(
            {
                "month_idx": cur,
                "opening_mrr": a.sum(),
                "new": new,
                "expansion": delta[delta > 0].sum(),
                "contraction": delta[delta < 0].sum(),
                "churned": churned,
                "closing_mrr": b.sum(),
            }
        )
    return pd.DataFrame(rows).set_index("month_idx")


def customer_counts(snapshots):
    wide = snapshots.pivot(index="customer_id", columns="month_idx", values="mrr").fillna(0.0)
    active = (wide > 0).sum()
    started = ((wide > 0) & (wide.shift(1, axis=1).fillna(0) == 0)).sum()
    lost = ((wide.shift(1, axis=1).fillna(0) > 0) & (wide == 0)).sum()
    out = pd.DataFrame({"active": active, "new_customers": started, "lost_customers": lost})
    out["logo_churn"] = out["lost_customers"] / out["active"].shift(1)
    return out


def revenue_retention(waterfall):
    """Net and gross revenue retention for each month."""
    w = waterfall
    nrr = (w["opening_mrr"] + w["expansion"] + w["contraction"] + w["churned"]) / w["opening_mrr"]
    grr = (w["opening_mrr"] + w["contraction"] + w["churned"]) / w["opening_mrr"]
    return pd.DataFrame({"nrr": nrr, "grr": grr})


def ltv(snapshots, counts, window=6):
    """Simple LTV: ARPU divided by average monthly logo churn over the last months."""
    last = snapshots["month_idx"].max()
    recent = snapshots[snapshots["month_idx"] > last - window]
    arpu = recent["mrr"].sum() / len(recent)
    churn = counts["logo_churn"].dropna().tail(window).mean()
    return {"arpu": float(arpu), "monthly_churn": float(churn), "ltv": float(arpu / churn)}


def plan_mix(snapshots, month_idx):
    cur = snapshots[snapshots["month_idx"] == month_idx]
    g = cur.groupby("plan")
    return pd.DataFrame({"customers": g.size(), "mrr": g["mrr"].sum()})


def quick_ratio(waterfall):
    """(new + expansion) / (churn + contraction): above 4 is usually considered healthy."""
    w = waterfall
    return (w["new"] + w["expansion"]) / -(w["churned"] + w["contraction"])
