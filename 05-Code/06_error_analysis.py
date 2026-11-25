"""Stage 06: error analysis for entity resolution and link prediction.

Link prediction: every top-1 miss of the best model (gbm_votes|full) on both
splits gets one rule-based cause, checked in this order:
  isolated          hospital has no graph edges at all (only name/geography to go on)
  successor_sys       (temporal) the true 2022 system ID is new, but most of its members sat
                    in one 2018 system: a successor system (merger, rebrand or re-numbering)
  cold_start        true system has at most one member in the training labels
  no_signal         hospital name shares no distinctive token with the true system's name
                    AND no graph neighbor in the true system (often a rebrand)
  rival_neighbors   the hospital's neighbors vote more for the predicted system
                    (shared officers/owners/officials with another system: JV,
                    management contract, or a pending merger)
  merger_move       (temporal) hospital moved from one system to another 2018->2022
  other
Votes and member counts use training labels only (never the test hospitals' labels).
Entity resolution: the 20 most confident false positives and false negatives on
the test split, plus every LLM-vs-silver disagreement in the uncertain band.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from common import CLEAN, METRICS, SEED, TABLES, key_tokens, log, save_metrics


def mask_person(name: str) -> str:
    """Hide organization names that are an individual clinician's name (e.g. '... MD, LLC')."""
    return "[clinician-named practice]" if re.search(r"\b(MD|M\.D\.|DO|D\.O\.|DPM|PHD)\b", str(name)) else name


def main():
    pr = pd.read_parquet(CLEAN / "linkpred_predictions.parquet")
    h = pd.read_parquet(CLEAN / "hospitals.parquet").set_index("ccn")
    sy = pd.read_parquet(CLEAN / "systems.parquet").sort_values("year").groupby("health_sys_id").last()
    lab = pd.read_parquet(CLEAN / "labels.parquet")
    l18 = dict(zip(lab[lab.year == 2018]["ccn"], lab[lab.year == 2018]["health_sys_id"]))
    edges = pd.read_parquet(CLEAN / "graph_edges.parquet")
    has_edge = set(edges["ccn"])
    l22 = dict(zip(lab[lab.year == 2022]["ccn"], lab[lab.year == 2022]["health_sys_id"]))
    # Rebuild stage 04's random-split training set so votes never see test labels.
    known = set(h.index)
    l22_in = {c: s for c, s in l22.items() if c in known and s in sy.index}
    ccns = np.array(sorted(l22_in))
    np.random.default_rng(SEED).shuffle(ccns)
    l22_train = {c: l22_in[c] for c in ccns[: int(.7 * len(ccns))]}
    # 2022 system IDs absent in 2018 whose members mostly came from one 2018 system
    sys18 = set(l18.values())
    reid = set()
    for s22, g in pd.Series(l22).groupby(pd.Series(l22)):
        if s22 in sys18:
            continue
        prev = pd.Series([l18[c] for c in g.index if c in l18])
        if len(prev) and prev.value_counts().iloc[0] >= max(2, 0.5 * len(g)):
            reid.add(s22)
    ent_members = edges.groupby("entity")["ccn"].apply(list)
    nb = edges.groupby("ccn")["entity"].apply(list)

    def votes(ccn, ref):
        v = {}
        for e in nb.get(ccn, []):
            for o in ent_members[e]:
                if o != ccn and o in ref:
                    v[ref[o]] = v.get(ref[o], 0) + 1
        return v

    out, gallery = {}, []
    for split in ("random", "temporal"):
        p = pr[(pr["split"] == split) & (pr["model"] == "gbm_votes|full")].copy()
        ref = l22_train if split == "random" else l18
        train_members = pd.Series(list(ref.values())).value_counts()
        miss = p[p["top1"] != p["true"]].copy()
        causes = []
        for r in miss.itertuples():
            v = votes(r.ccn, ref)
            tname = sy.loc[r.true, "sys_name_norm"] if r.true in sy.index else ""
            if r.ccn not in has_edge:
                c = "isolated"
            elif split == "temporal" and r.true in reid:
                c = "successor_sys"
            elif train_members.get(r.true, 0) <= 1:
                c = "cold_start"
            elif not (key_tokens(h.loc[r.ccn, "name_norm"]) & key_tokens(tname)) and v.get(r.true, 0) == 0:
                c = "no_signal"
            elif v.get(r.top1, 0) > v.get(r.true, 0):
                c = "rival_neighbors"
            elif split == "temporal" and r.ccn in l18 and l18[r.ccn] != r.true:
                c = "merger_move"
            else:
                c = "other"
            causes.append(c)
        miss["cause"] = causes
        out[split] = {"test": len(p), "misses": len(miss), "causes": miss["cause"].value_counts().to_dict()}
        if split == "temporal":
            out[split]["reid_systems"] = len(reid)
            out[split]["test_in_reid_systems"] = int(p["true"].isin(reid).sum())
        for c, g in miss.groupby("cause"):
            for r in g.head(4).itertuples():
                gallery.append({"split": split, "cause": c, "ccn": r.ccn, "hospital": h.loc[r.ccn, "hospital_name"],
                                "state": h.loc[r.ccn, "state"],
                                "true_system": sy.loc[r.true, "health_sys_name"] if r.true in sy.index else r.true,
                                "predicted": sy.loc[r.top1, "health_sys_name"] if r.top1 in sy.index else r.top1,
                                "true_in_top5": bool(r.true in (r.top2, r.top3, r.top4, r.top5))})
        miss.to_csv(TABLES / f"errors_linkpred_{split}.csv", index=False)
    pd.DataFrame(gallery).to_csv(TABLES / "error_gallery.csv", index=False)

    # Entity resolution
    f = pd.read_parquet(CLEAN / "er_pairs_scored.parquet")
    te = f[f["split"] == "test"]
    n = pd.read_parquet(CLEAN / "nppes_orgs.parquet").set_index("npi")
    fp = te[(te["y"] == 0) & (te["p_c3"] >= .5)].nlargest(20, "p_c3")
    fn = te[(te["y"] == 1) & (te["p_c3"] < .5)].nsmallest(20, "p_c3")
    rows = []
    for kind, g in (("false_positive", fp), ("false_negative", fn)):
        for r in g.itertuples():
            rows.append({"kind": kind, "ccn": r.ccn, "hospital": h.loc[r.ccn, "hospital_name"],
                         "h_street": h.loc[r.ccn, "hospital_street"], "npi": r.npi, "nppes_name": mask_person(n.loc[r.npi, "lbn"]),
                         "nppes_street": n.loc[r.npi, "pl_street"], "taxonomy": n.loc[r.npi, "primary_taxonomy"],
                         "p": round(r.p_c3, 3)})
    pd.DataFrame(rows).to_csv(TABLES / "errors_er.csv", index=False)
    # use the same verdict file stage 02 scored with (blind API run if it was used)
    src = json.loads((METRICS / "er.json").read_text()).get("llm_source")
    vfile = "llm_verdicts_api.json" if src == "anthropic-api" else "llm_verdicts.json"
    verdicts = json.loads((CLEAN / vfile).read_text())
    reasons = pd.Series([v["reason"] for v in verdicts.values() if not v["match"]]).value_counts()
    out["er"] = {"top_fp": len(fp), "top_fn": len(fn),
                 "llm_rejection_reasons": reasons.head(12).to_dict(),
                 "llm_rejections": int(sum(not v["match"] for v in verdicts.values())),
                 "llm_accepts": int(sum(v["match"] for v in verdicts.values()))}
    save_metrics("errors", out)
    log(json.dumps(out, indent=1)[:2500])


if __name__ == "__main__":
    main()
