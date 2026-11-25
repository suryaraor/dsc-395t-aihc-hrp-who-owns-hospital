"""Stage 05: are health-system network features associated with quality and cost?

Sample: AHRQ 2022 general acute-care hospitals (acutehosp_flag == 1).
Exposures (AHRQ 2022 structure):
  in_system        hospital belongs to a multi-hospital system
  log_sys_size     log(# hospitals in the system), 0 if independent
  log_local_sys    log(1 + # same-system hospitals in the same state), a local-footprint proxy
Outcomes:
  star             CMS overall star rating 1-5 (ordered logit)
  readmit_hwr      hybrid hospital-wide 30-day readmission rate, % (OLS)
  mspb             Medicare spending per beneficiary ratio (OLS)
  log_markup       log(total charges / total costs), FY2022 cost report (OLS)
  log_rev_dsch     log(net patient revenue / discharge), FY2022 (OLS, price proxy)
Controls: log beds, major teaching, any teaching, urban, ownership type, critical access (CAH),
children's hospital, state FE. CAHs are cost-reimbursed by Medicare and mostly independent, so
leaving them uncontrolled would turn the in-system coefficient into a partial CAH-vs-PPS contrast.
SEs clustered by state. Associations only, not causal effects.
Sensitivity: "stable" sample of hospitals whose system status did not change 2018->2022, and
"pps" sample of short-term PPS hospitals only (POS subtype 01).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel

from common import CLEAN, TABLES, log, save_metrics

OUTCOMES = {"star": "Star rating (ordered logit, log-odds)", "readmit_hwr": "Hospital-wide readmission (pp)",
            "mspb": "Medicare spending per beneficiary (ratio)", "log_markup": "log(charges / costs)",
            "log_rev_dsch": "log(net revenue per discharge)"}
EXPOSURES = ["in_system", "log_sys_size", "log_local_sys"]
CONTROLS = "log_beds + teach_major + teach_any + urban + cah + childrens + C(own) + C(state)"


def build():
    h = pd.read_parquet(CLEAN / "hospitals.parquet")
    lab = pd.read_parquet(CLEAN / "labels.parquet")
    sy = pd.read_parquet(CLEAN / "systems.parquet")
    l22 = lab[lab.year == 2022][["ccn", "health_sys_id"]]
    l18 = lab[lab.year == 2018][["ccn", "health_sys_id"]].rename(columns={"health_sys_id": "sys18"})
    s22 = sy[sy.year == 2022][["health_sys_id", "hosp_cnt"]]
    d = h[(h["acutehosp_flag"] == 1) & (h["last_ahrq_year"] == 2022)].merge(l22, on="ccn", how="left")
    d = d.merge(s22, on="health_sys_id", how="left").merge(l18, on="ccn", how="left")
    d["in_system"] = d["health_sys_id"].notna().astype(int)
    d["log_sys_size"] = np.log(d["hosp_cnt"].fillna(1))
    local = d.groupby(["health_sys_id", "state"])["ccn"].transform("count")
    d["log_local_sys"] = np.where(d["in_system"] == 1, np.log1p(local - 1), 0.0)
    d["log_beds"] = np.log1p(d["hos_beds"].fillna(d["pos_beds"]))
    d["teach_major"] = d["hos_majteach"].fillna(0)
    d["teach_any"] = (d["hos_teachint"].fillna(0) > 0).astype(int)
    d["urban"] = (d["cr_rural"] == "U").astype(int)
    d["own"] = d["hos_ownership"].fillna(0).astype(int)
    d["cah"] = (d["pos_subtype"] == "11").astype(int)
    d["childrens"] = ((d["pos_subtype"] == "06") | (d["hos_children"].fillna(0) == 1)).astype(int)
    d["pps"] = (d["pos_subtype"] == "01").astype(int)
    d["log_markup"] = np.log(d["cr_markup"].where(d["cr_markup"].between(0.5, 30)))
    d["log_rev_dsch"] = np.log(d["cr_rev_per_dsch"].where(d["cr_rev_per_dsch"] > 1000))
    d["stable"] = (d["health_sys_id"].fillna("none") == d["sys18"].fillna("none")).astype(int)
    return d


def fit(d: pd.DataFrame, y: str, x: str):
    cols = [y, x, "log_beds", "teach_major", "teach_any", "urban", "cah", "childrens", "own", "state"]
    m = d[cols].dropna()
    m = m[m.groupby("state")["state"].transform("count") >= 5]
    if y == "star":
        X = pd.get_dummies(m[[x, "log_beds", "teach_major", "teach_any", "urban", "cah", "childrens"]].assign(
            own=m["own"].astype(str), state=m["state"]), columns=["own", "state"], drop_first=True, dtype=float)
        X = X.loc[:, X.nunique() > 1]  # e.g. no CAHs in the PPS sample; constant columns make the Hessian singular
        res = OrderedModel(m[y].astype(int), X, distr="logit").fit(method="bfgs", disp=False, maxiter=2000)
        b, se = res.params[x], res.bse[x]
    else:
        res = smf.ols(f"{y} ~ {x} + {CONTROLS}", m).fit(cov_type="cluster", cov_kwds={"groups": m["state"]})
        b, se = res.params[x], res.bse[x]
    return {"beta": float(b), "se": float(se), "lo": float(b - 1.96 * se), "hi": float(b + 1.96 * se),
            "p": float(2 * (1 - __import__("scipy").stats.norm.cdf(abs(b / se)))), "n": int(len(m))}


def main():
    d = build()
    rows, out = [], {"sample": len(d), "in_system_share": float(d["in_system"].mean()),
                     "stable_share": float(d["stable"].mean())}
    desc = d.groupby("in_system")[list(OUTCOMES)].mean().round(3)
    out["means_by_system_status"] = desc.to_dict(orient="index")
    out["cah_share"] = float(d["cah"].mean())
    for sample, dd in (("all", d), ("stable", d[d["stable"] == 1]), ("pps", d[d["pps"] == 1])):
        for y in OUTCOMES:
            for x in EXPOSURES:
                r = fit(dd, y, x)
                r.update(sample=sample, outcome=y, exposure=x)
                rows.append(r)
                log(f"{sample:6s} {y:12s} ~ {x:13s} b={r['beta']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}] "
                    f"p={r['p']:.3f} n={r['n']}")
    t = pd.DataFrame(rows)
    t.round(4).to_csv(TABLES / "quality_cost_results.csv", index=False)
    out["results"] = rows
    out["outcome_labels"] = OUTCOMES
    save_metrics("quality_cost", out)
    d[["ccn", "in_system", "log_sys_size", "log_local_sys"] + list(OUTCOMES)].to_parquet(
        CLEAN / "quality_cost_frame.parquet", index=False)


if __name__ == "__main__":
    main()
