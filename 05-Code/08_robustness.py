"""Stage 08: robustness checks (read-only over earlier stages' outputs).

  er_strict         matcher scores on pairs whose street number differs (where the silver
                    label rule, "same street + ZIP", cannot be rediscovered) and recall on
                    the strict Enrollments pairs alone
  audit             objective-audit baselines: always-match, Wilson 95% CIs, exact McNemar
                    test of C2 vs C3, and how often the LLM said "match"
  linkpred_ci       bootstrap 95% CIs (2,000 resamples of test hospitals) for Hits@1
  temporal_moves    Hits@1 on temporal changers after removing hospitals whose 2022
                    system is a renamed / re-numbered 2018 system (see stage 06)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from common import CLEAN, SEED, log, save_metrics


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def prf(y, p):
    yhat = p >= 0.5
    tp = int((yhat & (y == 1)).sum()); fp = int((yhat & (y == 0)).sum()); fn = int((~yhat & (y == 1)).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return {"n": int(len(y)), "pos": int(y.sum()), "precision": pr, "recall": rc, "f1": 2 * pr * rc / max(pr + rc, 1e-9)}


def er_checks(out):
    f = pd.read_parquet(CLEAN / "er_pairs_scored.parquet")
    te = f[f["split"] == "test"]
    lab = pd.read_parquet(CLEAN / "enroll_npi.parquet")
    strict = set(zip(lab["ccn"], lab["npi"]))
    diff = te[te["street_no_eq"] == 0]
    is_strict = np.array([(c, n) in strict for c, n in zip(te["ccn"], te["npi"])])
    out["er_strict"] = {
        "diff_street": {k: prf(diff["y"].values, diff[c].values) for k, c in (("C1", "p_c1"), ("C2", "p_c2"))},
        "strict_recall": {k: float((te.loc[is_strict, c] >= .5).mean()) for k, c in (("C1", "p_c1"), ("C2", "p_c2"))},
        "strict_test_pairs": int(is_strict.sum())}

    # objective audit, rebuilt exactly as stage 02 does
    band = te[te["p_c2"].between(0.3, 0.7)]
    own = lab.groupby("npi")["ccn"].apply(set)
    ob = band.assign(enrolled=band["npi"].map(own)).dropna(subset=["enrolled"])
    t = np.array([int(c in s) for c, s in zip(ob["ccn"], ob["enrolled"])])
    c2 = (ob["p_c2"].values >= .5).astype(int) == t
    c3 = (ob["p_c3"].values >= .5).astype(int) == t
    b, c = int((c3 & ~c2).sum()), int((c2 & ~c3).sum())
    n = len(t)
    out["audit"] = {"n": n, "hospitals": int(ob["ccn"].nunique()), "true_matches": int(t.sum()),
                    "acc_c2": float(c2.mean()), "acc_c3": float(c3.mean()),
                    "ci_c2": wilson(int(c2.sum()), n), "ci_c3": wilson(int(c3.sum()), n),
                    "acc_always_match": float(t.mean()), "llm_match_rate": float((ob["p_c3"] >= .5).mean()),
                    "only_llm_right": b, "only_c2_right": c,
                    "mcnemar_p": float(binomtest(b, b + c, 0.5).pvalue) if b + c else None}


def linkpred_checks(out):
    pr = pd.read_parquet(CLEAN / "linkpred_predictions.parquet")
    rng = np.random.default_rng(SEED)
    ci = {}
    for (split, model), g in pr.groupby(["split", "model"]):
        hit = (g["top1"] == g["true"]).values.astype(float)
        boots = [hit[rng.integers(0, len(hit), len(hit))].mean() for _ in range(2000)]
        ci[f"{split}|{model}"] = {"hits1": float(hit.mean()), "ci": [float(np.percentile(boots, 2.5)),
                                                                      float(np.percentile(boots, 97.5))]}
    out["linkpred_ci"] = ci

    lab = pd.read_parquet(CLEAN / "labels.parquet")
    l18 = dict(zip(lab[lab.year == 2018]["ccn"], lab[lab.year == 2018]["health_sys_id"]))
    l22 = pd.Series(dict(zip(lab[lab.year == 2022]["ccn"], lab[lab.year == 2022]["health_sys_id"])))
    sys18 = set(l18.values())
    reid = set()
    for s22, g in l22.groupby(l22):
        if s22 in sys18:
            continue
        prev = pd.Series([l18[c] for c in g.index if c in l18])
        if len(prev) and prev.value_counts().iloc[0] >= max(2, 0.5 * len(g)):
            reid.add(s22)
    # "Successor systems": new 2022 IDs whose members mostly came from one 2018 system
    # (mergers such as CommonSpirit = CHI + Dignity, rebrands, re-numbering).
    pred = {}
    for s22, g in l22.groupby(l22):
        if s22 in reid:
            pred[s22] = pd.Series([l18[c] for c in g.index if c in l18]).value_counts().index[0]
    names = pd.read_parquet(CLEAN / "systems.parquet").sort_values("year").groupby("health_sys_id")["health_sys_name"].last()
    tm = {}
    for model in ("gbm_votes|full", "name_rule|full", "sage|full", "rgcn|full", "spectral|full"):
        g = pr[(pr["split"] == "temporal") & (pr["model"] == model)]
        hit = g["top1"] == g["true"]
        succ = g["true"].isin(reid)
        new = ~g["ccn"].isin(l18.keys())
        row = {"n_all": int(len(g)), "hits1_all": float(hit.mean())}
        for lab, m in (("non_successor", ~succ), ("newly_affiliated", ~succ & new), ("switched", ~succ & ~new),
                       ("successor", succ)):
            row[lab] = {"n": int(m.sum()), "hits1": float(hit[m].mean())}
        # in successor systems, how often does the model name the predecessor 2018 system?
        row["successor"]["top1_is_predecessor"] = float((g.loc[succ, "top1"] == g.loc[succ, "true"].map(pred)).mean())
        tm[model] = row
    out["temporal_moves"] = {"successor_systems": len(reid),
                             "successor_examples": sorted(f"{names.get(s, s)} <- {names.get(p, p)}" for s, p in pred.items()),
                             "by_model": tm}

    # Holm correction over the 15 tests in each quality/cost sample
    qc = pd.read_csv(CLEAN.parents[1] / "06-Results" / "tables" / "quality_cost_results.csv")
    holm = {}
    for smp, g in qc.groupby("sample"):
        g = g.sort_values("p").reset_index(drop=True)
        k, keep = len(g), []
        for i, r in g.iterrows():
            if r["p"] > 0.05 / (k - i):
                break
            keep.append(f"{r['outcome']}~{r['exposure']}")
        holm[smp] = keep
    out["holm_survivors"] = holm


def main():
    out = {}
    er_checks(out)
    linkpred_checks(out)
    save_metrics("robustness", out)
    log(out["audit"]); log(out["temporal_moves"]); log(out["holm_survivors"])


if __name__ == "__main__":
    main()
