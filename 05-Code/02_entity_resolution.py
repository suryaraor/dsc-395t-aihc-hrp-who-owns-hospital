"""Stage 02: entity resolution, CMS hospitals (CCN) <-> NPPES organizations (NPI).

Pipeline
  1. Blocking: same state AND (same ZIP3 OR same city OR same phone)
  2. Features: rapidfuzz name/address scores, exact zip/city/phone/street-number,
     sentence-embedding cosine (all-MiniLM-L6-v2 on "name | street | city")
  3. Classifiers, trained on CMS Enrollment CCN<->NPI pairs, split by hospital:
       C1 fuzzy only (logistic regression on string features)
       C2 fuzzy + embedding (LightGBM)
       C3 C2 + LLM adjudication of the uncertain band (0.3 <= p <= 0.7)
  4. Crosswalk: per CCN, every NPI with final p >= 0.5 (top-1 marked primary)

LLM adjudication: if ANTHROPIC_API_KEY is set, Claude Haiku is called; otherwise
verdicts are read from 04-Data/clean/llm_verdicts.json (earlier non-blind
verdicts). Pairs without a verdict keep p from C2.
"""
from __future__ import annotations

import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from common import CLEAN, SEED, TABLES, log, norm_addr, norm_name, phone10, save_metrics

BAND = (0.3, 0.7)
VERDICTS = CLEAN / "llm_verdicts.json"          # earlier non-blind verdicts; used only when no API key is set
VERDICTS_API = CLEAN / "llm_verdicts_api.json"  # fresh, blind verdicts from the scripted API path
LLM_MODEL = "claude-haiku-5-5"
VERDICT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["match", "reason"],
                  "properties": {"match": {"type": "boolean"}, "reason": {"type": "string"}}}
BAND_EXPORT = CLEAN / "llm_band_pairs.json"


def load():
    h = pd.read_parquet(CLEAN / "hospitals.parquet")
    h = h[["ccn", "hospital_name", "name_norm", "hospital_street", "street_norm", "city", "state", "zip", "phone",
           "cc_name"]].copy()
    h["cc_name_norm"] = norm_name(h["cc_name"])
    n = pd.read_parquet(CLEAN / "nppes_orgs.parquet")
    n = n[n["deactivated"].isna()].copy()
    n["lbn_norm"] = norm_name(n["lbn"])
    n["other_norm"] = norm_name(n["other_name"])
    n["pl_street_norm"] = norm_addr(n["pl_street"])
    n["pl_city"] = n["pl_city"].str.upper().str.strip()
    n["pl_phone"] = phone10(n["pl_phone"])
    lab = pd.read_parquet(CLEAN / "enroll_npi.parquet")
    return h, n, lab


def block(h, n):
    h = h.assign(zip3=h["zip"].str[:3])
    n = n.assign(zip3=n["pl_zip"].str[:3])
    keys = []
    for col_h, col_n in (("zip3", "zip3"), ("city", "pl_city"), ("phone", "pl_phone")):
        a = h[["ccn", "state", col_h]].dropna()
        b = n[["npi", "pl_state", col_n]].dropna()
        keys.append(a.merge(b, left_on=["state", col_h], right_on=["pl_state", col_n])[["ccn", "npi"]])
    return pd.concat(keys).drop_duplicates()


def street_no(s: pd.Series) -> pd.Series:
    return s.str.extract(r"^(\d+)", expand=False)


def features(pairs, h, n, emb_h, emb_n):
    d = pairs.merge(h, on="ccn").merge(n, on="npi")
    f = pd.DataFrame({"ccn": d["ccn"].values, "npi": d["npi"].values})
    f["name_tset"] = [max(fuzz.token_set_ratio(a, b), fuzz.token_set_ratio(c, b) if c else 0,
                          fuzz.token_set_ratio(a, o) if o else 0)
                      for a, b, c, o in zip(d["name_norm"], d["lbn_norm"], d["cc_name_norm"], d["other_norm"])]
    f["name_ratio"] = [fuzz.ratio(a, b) for a, b in zip(d["name_norm"], d["lbn_norm"])]
    f["name_partial"] = [fuzz.partial_ratio(a, b) for a, b in zip(d["name_norm"], d["lbn_norm"])]
    f["addr_tset"] = [fuzz.token_set_ratio(a, b) for a, b in zip(d["street_norm"], d["pl_street_norm"])]
    f["street_no_eq"] = (street_no(d["street_norm"]) == street_no(d["pl_street_norm"])).astype(int).values
    f["zip_eq"] = (d["zip"] == d["pl_zip"]).astype(int).values
    f["city_eq"] = (d["city"] == d["pl_city"]).astype(int).values
    f["phone_eq"] = ((d["phone"] == d["pl_phone"]) & d["phone"].notna()).astype(int).values
    f["is_subpart"] = (d["is_subpart"] == "Y").astype(int).values
    f["tax_general_acute"] = d["primary_taxonomy"].str.startswith("282N").fillna(False).astype(int).values
    ih = h.reset_index(drop=True).reset_index().set_index("ccn")["index"]
    inn = n.reset_index(drop=True).reset_index().set_index("npi")["index"]
    a = emb_h[ih.loc[d["ccn"]].values]
    b = emb_n[inn.loc[d["npi"]].values]
    f["emb_cos"] = (a * b).sum(1)
    return f, d


def embed(h, n):
    from sentence_transformers import SentenceTransformer
    cache = CLEAN / "emb_cache.npz"
    if cache.exists():
        z = np.load(cache)
        if z["eh"].shape[0] == len(h) and z["en"].shape[0] == len(n):
            return z["eh"], z["en"]
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    th = (h["name_norm"] + " | " + h["street_norm"].fillna("") + " | " + h["city"].fillna("")).tolist()
    tn = (n["lbn_norm"] + " | " + n["pl_street_norm"].fillna("") + " | " + n["pl_city"].fillna("")).tolist()
    eh = m.encode(th, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    en = m.encode(tn, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    np.savez_compressed(cache, eh=eh, en=en)
    return eh, en


FUZZY = ["name_tset", "name_ratio", "name_partial", "addr_tset", "street_no_eq", "zip_eq", "city_eq", "phone_eq"]
FULL = FUZZY + ["emb_cos", "is_subpart", "tax_general_acute"]


def prf(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {"precision": precision_score(y, yhat, zero_division=0), "recall": recall_score(y, yhat, zero_division=0),
            "f1": f1_score(y, yhat, zero_division=0), "auprc": average_precision_score(y, p)}


def llm_adjudicate(band: pd.DataFrame) -> dict[str, dict]:
    """Return {"ccn|npi": {"match": bool, "reason": str}}."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic()
        # Separate cache so a keyed run never reuses the earlier non-blind verdicts.
        out = json.loads(VERDICTS_API.read_text()) if VERDICTS_API.exists() else {}
        for r in band.itertuples():
            key = f"{r.ccn}|{r.npi}"
            if key in out:
                continue
            prompt = ("Do these two records describe the same hospital facility (same Medicare provider)? "
                      "Rule: a match is this hospital under its legal, former or operator name, or its own "
                      "units; a co-located long-term acute care hospital, physician practice or other "
                      "hospital at the same address is NOT a match. "
                      "Answer only JSON {\"match\": true|false, \"reason\": \"<one line>\"}.\n"
                      f"A (CMS): {r.hospital_name} | {r.hospital_street} | {r.city} {r.state} {r.zip}\n"
                      f"B (NPPES): {r.lbn} / {r.other_name} | {r.pl_street} | {r.pl_city} {r.pl_state} "
                      f"{r.pl_zip} | taxonomy {r.primary_taxonomy} | subpart {r.is_subpart}")
            # Haiku 5.5 rejects sampling params; structured outputs guarantee the JSON shape.
            msg = client.messages.create(
                model=LLM_MODEL, max_tokens=1024,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
                messages=[{"role": "user", "content": prompt}])
            if msg.stop_reason == "refusal":
                raise SystemExit(f"LLM refused pair {key}")
            v = json.loads(next(b.text for b in msg.content if b.type == "text"))
            out[key] = {"match": bool(v["match"]), "reason": v["reason"]}
            VERDICTS_API.write_text(json.dumps(out, indent=1))
        return out
    if VERDICTS.exists():
        return json.loads(VERDICTS.read_text())
    raise SystemExit(f"No LLM verdicts: set ANTHROPIC_API_KEY or restore {VERDICTS.name} from the repo "
                     "(it cannot be regenerated without the API).")


def main():
    h, n, lab = load()
    log(f"hospitals {len(h)}, active NPPES orgs {len(n)}")
    pairs = block(h, n)
    lab_in = lab[lab["ccn"].isin(h["ccn"]) & lab["npi"].isin(n["npi"])]
    pos_strict = set(zip(lab_in["ccn"], lab_in["npi"]))
    # Facility-level label expansion (issue I-14): CMS Enrollments lists ONE NPI
    # per CCN, but a hospital registers several NPIs at the same site (legal
    # entity, subparts, psych/rehab units). Any active hospital-taxonomy NPI at
    # the enrolled NPI's exact normalized street + ZIP is the same facility.
    site = n.set_index("npi")[["pl_street_norm", "pl_zip"]]
    enrolled_site = lab_in.join(site, on="npi")[["ccn", "pl_street_norm", "pl_zip"]].dropna()
    sib = enrolled_site.merge(n[["npi", "pl_street_norm", "pl_zip"]], on=["pl_street_norm", "pl_zip"])
    pos = pos_strict | set(zip(sib["ccn"], sib["npi"]))
    log(f"labels: strict {len(pos_strict)}, facility-level {len(pos)}")
    pairs["y"] = [int((c, p) in pos) for c, p in zip(pairs["ccn"], pairs["npi"])]
    blocking_recall = pairs["y"].sum() / len(pos)
    log(f"candidate pairs {len(pairs)}; blocking recall {blocking_recall:.3f} ({pairs['y'].sum()}/{len(pos)})")

    eh, en = embed(h, n)
    f, d = features(pairs[["ccn", "npi"]], h, n, eh, en)
    f["y"] = pairs.set_index(["ccn", "npi"]).loc[list(zip(f["ccn"], f["npi"])), "y"].values

    # Split by hospital so a CCN's pairs never straddle train/test.
    labeled_ccn = np.array(sorted(lab_in["ccn"].unique()))
    r = np.random.default_rng(SEED)
    r.shuffle(labeled_ccn)
    k = len(labeled_ccn)
    split = {c: "train" for c in labeled_ccn[: int(.6 * k)]}
    split.update({c: "val" for c in labeled_ccn[int(.6 * k): int(.8 * k)]})
    split.update({c: "test" for c in labeled_ccn[int(.8 * k):]})
    f["split"] = f["ccn"].map(split).fillna("unlabeled")
    tr, va, te = (f[f.split == s] for s in ("train", "val", "test"))

    c1 = LogisticRegression(max_iter=2000, class_weight="balanced").fit(tr[FUZZY], tr["y"])
    c2 = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8,
                            colsample_bytree=0.9, random_state=SEED, verbose=-1)
    c2.fit(tr[FULL], tr["y"], eval_set=[(va[FULL], va["y"])])
    f["p_c1"] = c1.predict_proba(f[FUZZY])[:, 1]
    f["p_c2"] = c2.predict_proba(f[FULL])[:, 1]

    # Simple rule baseline: name token-set >= 90 and same zip.
    f["p_rule"] = ((f["name_tset"] >= 90) & (f["zip_eq"] == 1)).astype(float)

    # LLM band on the test split only (it is the evaluation that matters).
    te_mask = f["split"] == "test"
    band_mask = te_mask & f["p_c2"].between(*BAND)
    band = f[band_mask].merge(d[["ccn", "npi", "hospital_name", "hospital_street", "city", "state", "zip", "lbn",
                                 "other_name", "pl_street", "pl_city", "pl_state", "pl_zip", "primary_taxonomy"]],
                                 on=["ccn", "npi"])  # is_subpart already in f; merging it again made _x/_y columns
    # Record text only: no labels, scores, split or match features, so the adjudicator stays blind.
    record_cols = ["ccn", "npi", "hospital_name", "hospital_street", "city", "state", "zip", "lbn", "other_name",
                   "pl_street", "pl_city", "pl_state", "pl_zip", "primary_taxonomy", "is_subpart"]
    BAND_EXPORT.write_text(band[[c for c in record_cols if c in band.columns]].to_json(orient="records", indent=1))
    verdicts = llm_adjudicate(band)
    f["p_c3"] = f["p_c2"]
    n_adj = 0
    for i in f.index[band_mask]:
        v = verdicts.get(f"{f.at[i, 'ccn']}|{f.at[i, 'npi']}")
        if v is not None:
            f.at[i, "p_c3"] = 0.95 if v["match"] else 0.05
            n_adj += 1
    log(f"uncertain band (test): {int(band_mask.sum())} pairs, adjudicated {n_adj}")
    if "--export-only" in sys.argv:
        return

    te = f[te_mask]
    res = {name: prf(te["y"].values, te[col].values) for name, col in
           (("rule_name90_zip", "p_rule"), ("C1_fuzzy_LR", "p_c1"), ("C2_fuzzy_emb_LGBM", "p_c2"),
            ("C3_C2_plus_LLM", "p_c3"))}
    # Band-only view: how much did the LLM change inside the uncertain band?
    bt = f[band_mask]
    band_res = {"n": int(len(bt)), "positives": int(bt["y"].sum()),
                "acc_c2": float(((bt["p_c2"] >= .5) == bt["y"]).mean()) if len(bt) else None,
                "acc_c3": float(((bt["p_c3"] >= .5) == bt["y"]).mean()) if len(bt) else None}

    # Objective audit (issue I-14): silver labels mislabel co-located LTACHs and
    # campus-address variants. For band NPIs that CMS Enrollments assigns to a
    # specific CCN, the truth is unambiguous: match iff that CCN is this CCN.
    own = lab.groupby("npi")["ccn"].apply(set)
    ob = bt.assign(enrolled=bt["npi"].map(own)).dropna(subset=["enrolled"])
    ob_truth = np.array([int(c in s) for c, s in zip(ob["ccn"], ob["enrolled"])])
    band_res["objective_audit"] = {
        "n": int(len(ob)), "true_matches": int(ob_truth.sum()),
        "acc_c2": float(((ob["p_c2"] >= .5).astype(int) == ob_truth).mean()) if len(ob) else None,
        "acc_c3": float(((ob["p_c3"] >= .5).astype(int) == ob_truth).mean()) if len(ob) else None,
    }
    band_res["silver_vs_llm"] = {
        "silver1_llm1": int(((bt["y"] == 1) & (bt["p_c3"] >= .5)).sum()),
        "silver1_llm0": int(((bt["y"] == 1) & (bt["p_c3"] < .5)).sum()),
        "silver0_llm1": int(((bt["y"] == 0) & (bt["p_c3"] >= .5)).sum()),
        "silver0_llm0": int(((bt["y"] == 0) & (bt["p_c3"] < .5)).sum()),
    }

    # Per-hospital top-1 accuracy on test: is the best-scoring NPI a labeled one?
    def top1(col):
        g = te.sort_values(col, ascending=False).groupby("ccn").head(1)
        return float(g["y"].mean())

    importance = dict(zip(FULL, c2.booster_.feature_importance("gain").round(1).tolist()))
    # Crosswalk on all hospitals with C3-where-available else C2.
    f["p_final"] = f["p_c3"]
    xw = f[f["p_final"] >= 0.5].sort_values("p_final", ascending=False)
    xw["primary"] = ~xw.duplicated("ccn")
    xw[["ccn", "npi", "p_final", "primary"]].to_parquet(CLEAN / "crosswalk_ccn_npi.parquet", index=False)
    f.to_parquet(CLEAN / "er_pairs_scored.parquet", index=False)

    tab = pd.DataFrame(res).T.round(3)
    tab.to_csv(TABLES / "er_results.csv")
    save_metrics("er", {
        "hospitals": len(h), "nppes_active_orgs": len(n), "labeled_pairs": len(pos),
        "labeled_pairs_strict": len(pos_strict),
        "labeled_hospitals": int(lab_in["ccn"].nunique()), "candidate_pairs": len(pairs),
        "blocking_recall": blocking_recall, "pairs_per_hospital": len(pairs) / len(h),
        "test_pairs": int(te_mask.sum()), "test_positives": int(te["y"].sum()),
        "results_test": res, "llm_band": band_res, "top1_acc": {c: top1(c) for c in ("p_c1", "p_c2", "p_c3")},
        "feature_gain": importance, "crosswalk_hospitals": int(xw["ccn"].nunique()),
        "crosswalk_pairs": len(xw), "coverage": xw["ccn"].nunique() / len(h),
        "llm_source": "anthropic-api" if os.environ.get("ANTHROPIC_API_KEY") else "cached-verdicts",
    })
    log(tab.to_string())
    log(json.dumps(band_res))


if __name__ == "__main__":
    main()
