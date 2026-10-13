import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import core, data

OUT = Path("reports")


def main():
    OUT.mkdir(exist_ok=True)
    snap = data.generate_mrr_snapshots()
    wf = core.mrr_waterfall(snap)
    counts = core.customer_counts(snap)
    ret = core.revenue_retention(wf)
    ltv = core.ltv(snap, counts)
    qr = core.quick_ratio(wf)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(wf.index, wf["closing_mrr"], color="black")
    axes[0].set_title("Closing MRR")
    axes[0].set_xlabel("month")
    comp = wf[["new", "expansion", "contraction", "churned"]]
    comp.plot.bar(stacked=True, ax=axes[1], color=["black", "dimgrey", "darkgrey", "lightgrey"], width=0.9)
    axes[1].set_title("MRR movements")
    axes[1].set_xlabel("month")
    axes[1].tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "mrr.png", dpi=130)

    wf.round(2).to_csv(OUT / "mrr_waterfall.csv")
    last = wf.index.max()
    summary = {
        "customers_ever": int(snap["customer_id"].nunique()),
        "closing_mrr": round(float(wf.loc[last, "closing_mrr"]), 0),
        "active_customers": int(counts["active"].iloc[-1]),
        "avg_monthly_logo_churn_last_6m": round(ltv["monthly_churn"], 4),
        "net_revenue_retention_last_month": round(float(ret.loc[last, "nrr"]), 3),
        "gross_revenue_retention_last_month": round(float(ret.loc[last, "grr"]), 3),
        "arpu": round(ltv["arpu"], 2),
        "ltv": round(ltv["ltv"], 0),
        "quick_ratio_last_6m_avg": round(float(qr.tail(6).mean()), 2),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
