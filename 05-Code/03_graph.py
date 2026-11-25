"""Stage 03: build the heterogeneous hospital graph.

Node types: hospital, system, and "entity" nodes that hospitals can share.
Hospital-entity relations, grouped into leakage tiers (DECISIONS.md D-08):

  Tier A  owner_org    CMS All Owners, organizational owner (PECOS associate ID)
  Tier B  person_cms   CMS All Owners, individual owner/officer/manager (PECOS ID)
          official     NPPES authorized official (name + phone), via the ER crosswalk
  Tier C  parent_lbn   NPPES parent-organization legal name, via the ER crosswalk
          city         same city + state
          name_token   distinctive token of the hospital name (generic words removed)

Hubs are pruned: an entity shared by more than MAX_DEG hospitals is dropped
(e.g., a city with 60 hospitals or the token "ADVENTHEALTH" in 50 names is
kept, but tokens like "VALLEY" in 300 names are not).

Outputs: 04-Data/clean/graph_edges.parquet (hospital, entity, rel, tier),
         graph_nodes_hospital.parquet (features), metrics/graph.json
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import CLEAN, key_tokens, log, norm_name, phone10, save_metrics

MAX_DEG = {"owner_org": 400, "person_cms": 400, "official": 200, "parent_lbn": 200, "city": 60, "name_token": 60}
TIER = {"owner_org": "A", "person_cms": "B", "official": "B", "parent_lbn": "C", "city": "C", "name_token": "C"}


def main():
    h = pd.read_parquet(CLEAN / "hospitals.parquet")
    ccns = set(h["ccn"])
    owners = pd.read_parquet(CLEAN / "owners.parquet")
    owners = owners[owners["ccn"].isin(ccns)]
    xw = pd.read_parquet(CLEAN / "crosswalk_ccn_npi.parquet")
    nppes = pd.read_parquet(CLEAN / "nppes_orgs.parquet")
    nx = xw.merge(nppes, on="npi")

    edges = []
    o = owners[owners["owner_type"] == "O"]
    edges.append(pd.DataFrame({"ccn": o["ccn"], "entity": "own:" + o["owner_id"], "rel": "owner_org"}))
    p = owners[owners["owner_type"] == "I"]
    edges.append(pd.DataFrame({"ccn": p["ccn"], "entity": "per:" + p["owner_id"], "rel": "person_cms"}))

    ao_key = (norm_name(nx["ao_first"]) + "_" + norm_name(nx["ao_last"]) + "_" + phone10(nx["ao_phone"]).fillna(""))
    ok = nx["ao_last"].notna() & phone10(nx["ao_phone"]).notna()
    edges.append(pd.DataFrame({"ccn": nx.loc[ok, "ccn"], "entity": "ao:" + ao_key[ok], "rel": "official"}))

    par = nx[nx["parent_lbn"].notna()]
    edges.append(pd.DataFrame({"ccn": par["ccn"], "entity": "par:" + norm_name(par["parent_lbn"]), "rel": "parent_lbn"}))

    edges.append(pd.DataFrame({"ccn": h["ccn"], "entity": "city:" + h["state"] + "_" + h["city"], "rel": "city"}))

    tok_rows = [(c, t) for c, nm in zip(h["ccn"], h["name_norm"]) for t in key_tokens(nm)]
    tk = pd.DataFrame(tok_rows, columns=["ccn", "tok"])
    edges.append(pd.DataFrame({"ccn": tk["ccn"], "entity": "tok:" + tk["tok"], "rel": "name_token"}))

    e = pd.concat(edges).dropna().drop_duplicates()
    deg = e.groupby("entity")["ccn"].transform("nunique")
    e["deg"] = deg
    before = e.groupby("rel").size()
    keep = (e["deg"] >= 2) & (e["deg"] <= e["rel"].map(MAX_DEG))  # singletons carry no signal
    e = e[keep].copy()
    e["tier"] = e["rel"].map(TIER)
    e.to_parquet(CLEAN / "graph_edges.parquet", index=False)

    # Hospital node features (numeric, imputed) for the GNN.
    feat = pd.DataFrame({"ccn": h["ccn"]})
    feat["log_beds"] = np.log1p(h["hos_beds"].fillna(h["pos_beds"]).fillna(h["cr_beds"]).fillna(0))
    feat["teach_major"] = h["hos_majteach"].fillna(0)
    feat["teach_any"] = (h["hos_teachint"].fillna(0) > 0).astype(float)
    feat["acute"] = h["acutehosp_flag"].fillna(0)
    feat["children"] = h["hos_children"].fillna(0)
    own = h["hos_ownership"].fillna(0).astype(int)
    for k in range(1, 6):
        feat[f"own_{k}"] = (own == k).astype(float)
    feat["urban"] = (h["pos_urban"] == "1").astype(float)
    feat["log_costs"] = np.log1p(h["cr_total_costs"].fillna(0))
    feat.to_parquet(CLEAN / "graph_nodes_hospital.parquet", index=False)

    after = e.groupby("rel").agg(edges=("ccn", "size"), entities=("entity", "nunique"),
                                 hospitals=("ccn", "nunique"))
    stats = {
        "hospitals": len(h), "edges_total": len(e), "entities_total": int(e["entity"].nunique()),
        "by_relation": after.to_dict(orient="index"), "edges_before_pruning": before.to_dict(),
        "by_tier": e.groupby("tier").size().to_dict(),
        "hospitals_with_any_tierA": int(e.loc[e.tier == "A", "ccn"].nunique()),
        "hospitals_with_any_tierB": int(e.loc[e.tier == "B", "ccn"].nunique()),
        "hospitals_isolated": int(len(ccns - set(e["ccn"]))),
        "max_degree": MAX_DEG,
    }
    save_metrics("graph", stats)
    log(after.to_string())
    log(f"tiers {stats['by_tier']}, isolated hospitals {stats['hospitals_isolated']}")


if __name__ == "__main__":
    main()
