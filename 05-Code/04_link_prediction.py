"""Stage 04: predict hospital -> health-system links (AHRQ Compendium labels).

Task: for each held-out hospital, rank every AHRQ system; metrics are MRR,
Hits@1/5/10 (rank of the true system among all systems) and pairwise AUROC.

Splits
  random    2022 labels, hospitals split 70/15/15 (train/val/test)
  temporal  train on 2018 labels, test on 2022 hospitals whose system CHANGED
            (new to a system, or moved between systems) since 2018

Edge configurations (leakage ablation, DECISIONS.md D-08)
  full = tiers A+B+C,  noA = B+C,  noB = A+C,  C_only = C

Methods
  name_rule     distinctive-token Jaccard(hospital name, system name) + same-state bonus
  spectral      TruncatedSVD of the hospital-entity matrix -> cosine to system centroid
  gbm_votes     LightGBM on pair features: name similarity, geography, system size (training labels only),
                and neighbor "votes" per relation (share of a hospital's graph
                neighbors that belong to the system in the training labels)
  sage / rgcn   GraphSAGE and R-GCN encoders on the full node graph (hospitals,
                systems, entities), MLP decoder, trained with negative sampling;
                supervision edges are hidden from message passing each epoch
  persistence   (temporal only) predict the hospital's 2018 system
"""
from __future__ import annotations

import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from rapidfuzz import fuzz
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import RGCNConv, SAGEConv

from common import CLEAN, SEED, TABLES, key_tokens, log, save_metrics

torch.manual_seed(SEED)
np.random.seed(SEED)
RELS = ["owner_org", "person_cms", "official", "parent_lbn", "city", "name_token"]
CONFIGS = {"full": {"A", "B", "C"}, "noA": {"B", "C"}, "noB": {"A", "C"}, "C_only": {"C"}}
N_LAYERS = int(__import__("os").environ.get("GNN_LAYERS", "3"))


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self):
        self.h = pd.read_parquet(CLEAN / "hospitals.parquet").reset_index(drop=True)
        self.hidx = {c: i for i, c in enumerate(self.h["ccn"])}
        sy = pd.read_parquet(CLEAN / "systems.parquet").sort_values("year")
        self.sys = sy.groupby("health_sys_id", as_index=False).last().reset_index(drop=True)
        self.sidx = {s: i for i, s in enumerate(self.sys["health_sys_id"])}
        lab = pd.read_parquet(CLEAN / "labels.parquet")
        lab = lab[lab["ccn"].isin(self.hidx) & lab["health_sys_id"].isin(self.sidx)]
        self.lab = {y: dict(zip(g["ccn"], g["health_sys_id"])) for y, g in lab.groupby("year")}
        self.edges = pd.read_parquet(CLEAN / "graph_edges.parquet")
        self.feat = pd.read_parquet(CLEAN / "graph_nodes_hospital.parquet").set_index("ccn").loc[self.h["ccn"]]
        self.H, self.S = len(self.h), len(self.sys)
        self._name_emb()
        self._name_feats()

    def _name_emb(self):
        cache = CLEAN / "name_emb_cache.npz"
        if cache.exists():
            z = np.load(cache)
            if z["eh"].shape[0] == self.H and z["es"].shape[0] == self.S:
                self.eh, self.es = z["eh"], z["es"]
                return
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        self.eh = m.encode(self.h["name_norm"].tolist(), batch_size=256, normalize_embeddings=True)
        self.es = m.encode(self.sys["sys_name_norm"].tolist(), batch_size=256, normalize_embeddings=True)
        np.savez_compressed(cache, eh=self.eh, es=self.es)

    def _name_feats(self):
        ht = [key_tokens(n) for n in self.h["name_norm"]]
        st = [key_tokens(n) for n in self.sys["sys_name_norm"]]
        # Sparse token Jaccard via incidence matrices.
        vocab = {t: i for i, t in enumerate(sorted(set().union(*ht, *st)))}

        def inc(rows):
            r, c = zip(*[(i, vocab[t]) for i, ts in enumerate(rows) for t in ts]) if any(rows) else ([], [])
            return sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(rows), len(vocab)))
        Ih, Is = inc(ht), inc(st)
        inter = (Ih @ Is.T).toarray().astype(np.float32)
        nh = np.asarray(Ih.sum(1)).ravel()[:, None]
        ns = np.asarray(Is.sum(1)).ravel()[None, :]
        self.tok_jac = inter / np.maximum(nh + ns - inter, 1)
        # pandas 3 string columns are Arrow-backed and refuse [:, None] (issue I-18)
        obj = lambda s: s.fillna("").to_numpy(dtype=object)
        self.same_state = (obj(self.h["state"])[:, None] == obj(self.sys["health_sys_state"])[None, :]).astype(np.float32)
        self.same_city = ((obj(self.h["city"])[:, None] == obj(self.sys["health_sys_city"].str.upper())[None, :])
                          & (self.same_state > 0)).astype(np.float32)
        self.h_name = obj(self.h["name_norm"])
        self.s_name = obj(self.sys["sys_name_norm"])
        self.h_ccn = obj(self.h["ccn"])
        self.s_id = obj(self.sys["health_sys_id"])
        self.emb_sim = (self.eh @ self.es.T).astype(np.float32)
        # No label-year system size here: AHRQ hosp_cnt counts held-out members too, and
        # minus the training-label count it reveals the test answer.
        # System size is taken only from training labels (sys_known_members).

    def incidence(self, tiers):
        """Hospital x entity matrices per relation for the chosen tiers."""
        e = self.edges[self.edges["tier"].isin(tiers)]
        out = {}
        for rel, g in e.groupby("rel"):
            ents = {k: i for i, k in enumerate(g["entity"].unique())}
            r = g["ccn"].map(self.hidx).values
            c = g["entity"].map(ents).values
            out[rel] = sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(self.H, len(ents)))
        return out

    def label_matrix(self, labels: dict, ccns) -> sp.csr_matrix:
        r = [self.hidx[c] for c in ccns if c in labels]
        c = [self.sidx[labels[x]] for x in ccns if x in labels]
        return sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(self.H, self.S))


# ----------------------------------------------------------------------------- metrics
def rank_metrics(score: np.ndarray, test_h: np.ndarray, true_s: np.ndarray) -> dict:
    """score: H x S. Pessimistic ties (a tie counts against the true system)."""
    sc = score[test_h]
    t = sc[np.arange(len(test_h)), true_s][:, None]
    rank = (sc >= t).sum(1)  # includes itself
    y = np.zeros_like(sc, dtype=np.int8)
    y[np.arange(len(test_h)), true_s] = 1
    try:
        auc = roc_auc_score(y.ravel(), sc.ravel())
    except ValueError:
        auc = float("nan")
    return {"n": int(len(test_h)), "mrr": float((1 / rank).mean()), "hits1": float((rank <= 1).mean()),
            "hits5": float((rank <= 5).mean()), "hits10": float((rank <= 10).mean()),
            "median_rank": float(np.median(rank)), "auroc": float(auc)}


# ----------------------------------------------------------------------------- methods
def name_rule(D: Data, **_):
    return D.tok_jac + 0.05 * D.same_state + 0.01 * D.emb_sim


def neighbor_votes(D: Data, inc: dict, L: sp.csr_matrix) -> dict:
    """Per relation: share of a hospital's neighbors (via shared entities) in each system.

    Self-contributions are removed so a training hospital never votes for itself.
    """
    out = {}
    Ld = L.toarray()
    for rel in RELS:
        if rel not in inc:
            out[rel] = np.zeros((D.H, D.S), np.float32)
            continue
        B = inc[rel]
        deg = np.asarray(B.sum(0)).ravel()
        Bn = B.multiply(1.0 / np.maximum(deg - 1, 1)).tocsr()  # neighbors excluding self
        V = (Bn @ (B.T @ L)).toarray()
        self_w = np.asarray(Bn.multiply(B).sum(1)).ravel()
        V -= self_w[:, None] * Ld
        nnb = np.asarray((Bn @ np.asarray(B.sum(0)).ravel()[:, None] * 0 + Bn.sum(1))).ravel()
        out[rel] = (V / np.maximum(np.asarray(Bn.sum(1)).ravel()[:, None], 1e-9)).astype(np.float32)
        out[rel][nnb == 0] = 0
    return out


def spectral(D: Data, inc: dict, L: sp.csr_matrix, **_):
    if not inc:
        return np.zeros((D.H, D.S), np.float32)
    B = sp.hstack([inc[r] for r in inc]).tocsr()
    k = min(64, B.shape[1] - 1)
    Z = TruncatedSVD(k, random_state=SEED).fit_transform(B)
    Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    cnt = np.asarray(L.sum(0)).ravel()
    C = (L.T @ Z) / np.maximum(cnt[:, None], 1)
    C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
    s = Z @ C.T
    s[:, cnt == 0] = -1
    return s.astype(np.float32)


PAIR_FEATS = ["tok_jac", "emb_sim", "name_tset", "same_state", "same_city", "sys_known_members"] + \
             [f"vote_{r}" for r in RELS] + ["vote_any"]


def pair_frame(D: Data, votes, hs: np.ndarray, ss: np.ndarray, known_members: np.ndarray) -> pd.DataFrame:
    f = {"tok_jac": D.tok_jac[hs, ss], "emb_sim": D.emb_sim[hs, ss], "same_state": D.same_state[hs, ss],
         "same_city": D.same_city[hs, ss],
         "sys_known_members": np.log1p(known_members[ss])}
    f["name_tset"] = np.array([fuzz.token_set_ratio(a, b) / 100 for a, b in
                               zip(D.h_name[hs], D.s_name[ss])], np.float32)
    va = np.zeros(len(hs), np.float32)
    for r in RELS:
        f[f"vote_{r}"] = votes[r][hs, ss]
        va = np.maximum(va, f[f"vote_{r}"])
    f["vote_any"] = va
    return pd.DataFrame(f)


def candidates(D: Data, votes, hosp: np.ndarray, k_name=25) -> list[np.ndarray]:
    """Candidate systems per hospital: any vote > 0, same-state systems, top-k by name."""
    anyv = sum(votes[r] for r in RELS)
    out = []
    for h in hosp:
        c = set(np.nonzero(anyv[h] > 0)[0]) | set(np.nonzero(D.same_state[h] > 0)[0])
        c |= set(np.argsort(-(D.tok_jac[h] + 0.1 * D.emb_sim[h]))[:k_name])
        out.append(np.fromiter(c, int))
    return out


def gbm_votes(D: Data, inc: dict, L: sp.csr_matrix, train_h, train_s, val_h, val_s, **_):
    votes = neighbor_votes(D, inc, L)
    known = np.asarray(L.sum(0)).ravel()
    rng = np.random.default_rng(SEED)

    def build(hosp, true_s):
        cand = candidates(D, votes, hosp)
        hs = np.concatenate([np.full(len(c), h) for h, c in zip(hosp, cand)])
        ss = np.concatenate(cand)
        ys = np.concatenate([(c == t).astype(int) for c, t in zip(cand, true_s)])
        return hs, ss, ys

    # Training rows: each train hospital's own label is hidden from its features
    # by the self-removal in neighbor_votes; known-member counts exclude it too.
    hs, ss, ys = build(train_h, train_s)
    keep = (ys == 1) | (rng.random(len(ys)) < 0.35)  # subsample negatives
    hs, ss, ys = hs[keep], ss[keep], ys[keep]
    X = pair_frame(D, votes, hs, ss, known)
    own = np.zeros(len(hs))
    own[ys == 1] = 1
    X["sys_known_members"] = np.log1p(np.maximum(known[ss] - own, 0))
    vh, vs, vy = build(val_h, val_s)
    Xv = pair_frame(D, votes, vh, vs, known)
    model = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.04, num_leaves=63, min_child_samples=20,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.9, random_state=SEED, verbose=-1)
    model.fit(X[PAIR_FEATS], ys, eval_set=[(Xv[PAIR_FEATS], vy)],
              callbacks=[lgb.early_stopping(50, verbose=False)])

    def scorer(hosp):
        S = np.full((D.H, D.S), -1.0, np.float32)
        cand = candidates(D, votes, hosp)
        hs = np.concatenate([np.full(len(c), h) for h, c in zip(hosp, cand)])
        ss = np.concatenate(cand)
        S[hs, ss] = model.predict_proba(pair_frame(D, votes, hs, ss, known)[PAIR_FEATS])[:, 1]
        cand_recall = None
        return S, cand
    gain = dict(zip(PAIR_FEATS, model.booster_.feature_importance("gain").round(1).tolist()))
    return scorer, gain


# ----------------------------------------------------------------------------- GNN
class Encoder(nn.Module):
    def __init__(self, d_in, d_h, n_rel, kind):
        super().__init__()
        self.kind = kind
        self.lin = nn.Linear(d_in, d_h)
        # 3 layers: hospital -> shared entity -> sibling hospital -> system is a
        # 3-hop path, which a 2-layer encoder cannot see (issue I-19).
        self.convs = nn.ModuleList(
            [RGCNConv(d_h, d_h, n_rel, num_bases=4) if kind == "rgcn" else SAGEConv(d_h, d_h)
             for _ in range(N_LAYERS)])

    def forward(self, x, ei, et):
        x = F.relu(self.lin(x))
        hs = [x]
        for i, conv in enumerate(self.convs):
            x = conv(x, ei, et) if self.kind == "rgcn" else conv(x, ei)
            if i < len(self.convs) - 1:
                x = F.dropout(F.relu(x), 0.2, self.training)
            hs.append(x)
        return torch.stack(hs).sum(0)  # jumping-knowledge sum keeps 0..L-hop views


class Decoder(nn.Module):
    def __init__(self, d_h, d_pair):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * d_h + d_pair, d_h), nn.ReLU(), nn.Linear(d_h, 1))

    def forward(self, zh, zs, pf):
        return self.mlp(torch.cat([zh * zs, (zh - zs).abs(), pf], 1)).squeeze(1)


def gnn(D: Data, inc: dict, L: sp.csr_matrix, train_h, train_s, val_h, val_s, kind="sage", epochs=400, **_):
    # Node layout: [hospitals | systems | entities]
    ents, ent_rel, eh_rows, eh_cols, eh_types = [], [], [], [], []
    off = D.H + D.S
    for ri, rel in enumerate(RELS):
        if rel not in inc:
            continue
        B = inc[rel].tocoo()
        eh_rows.append(B.row)
        eh_cols.append(B.col + off)
        eh_types.append(np.full(B.nnz, ri))
        ent_rel.append(np.full(B.shape[1], ri))
        off += B.shape[1]
    N = off
    pca_h = PCA(24, random_state=SEED).fit_transform(D.eh)
    pca_s = PCA(24, random_state=SEED).fit(D.eh).transform(D.es)
    fh = D.feat.values.astype(np.float32)
    fh = (fh - fh.mean(0)) / (fh.std(0) + 1e-6)
    d_in = fh.shape[1] + 24 + 1 + len(RELS) + 2
    X = np.zeros((N, d_in), np.float32)
    X[:D.H, :fh.shape[1]] = fh
    X[:D.H, fh.shape[1]:fh.shape[1] + 24] = pca_h
    X[:D.H, -2] = 1
    X[D.H:D.H + D.S, fh.shape[1]:fh.shape[1] + 24] = pca_s
    X[D.H:D.H + D.S, fh.shape[1] + 24] = np.log1p(np.asarray(L.sum(0)).ravel())  # training-label size only
    X[D.H:D.H + D.S, -1] = 1
    if ent_rel:
        er = np.concatenate(ent_rel)
        X[D.H + D.S + np.arange(len(er)), fh.shape[1] + 25 + er] = 1
    X = torch.tensor(X)
    if eh_rows:
        r = np.concatenate(eh_rows); c = np.concatenate(eh_cols); t = np.concatenate(eh_types)
    else:
        r = c = t = np.zeros(0, int)
    MEMBER = len(RELS)  # relation id for hospital->system; reverse ids are offset
    n_rel = 2 * (len(RELS) + 1)

    def edge_index(mem_h, mem_s):
        src = np.concatenate([r, mem_h]); dst = np.concatenate([c, D.H + mem_s])
        typ = np.concatenate([t, np.full(len(mem_h), MEMBER)])
        ei = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])])
        et = np.concatenate([typ, typ + len(RELS) + 1])
        return torch.tensor(ei, dtype=torch.long), torch.tensor(et, dtype=torch.long)

    pair = lambda hs, ss: torch.tensor(np.stack([D.tok_jac[hs, ss], D.emb_sim[hs, ss], D.same_state[hs, ss],
                                                 D.same_city[hs, ss]], 1).astype(np.float32))
    enc = Encoder(d_in, 64, n_rel, kind)
    dec = Decoder(64, 4)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=5e-3, weight_decay=1e-5)
    rng = np.random.default_rng(SEED)
    state_sys = [np.nonzero(D.same_state[h])[0] for h in range(D.H)]
    best, best_state, patience = -1, None, 0
    ei_full, et_full = edge_index(train_h, train_s)
    for ep in range(epochs):
        enc.train(); dec.train()
        m = rng.random(len(train_h)) < 0.7  # 70% message edges, 30% supervision
        ei, et = edge_index(train_h[m], train_s[m])
        z = enc(X, ei, et)
        sh, ss = train_h[~m], train_s[~m]
        k = 8
        nh = np.repeat(sh, k)
        ns = np.where(rng.random(len(nh)) < 0.5, rng.integers(0, D.S, len(nh)),
                      [rng.choice(state_sys[h]) if len(state_sys[h]) else rng.integers(0, D.S) for h in nh])
        hs = np.concatenate([sh, nh]); sy = np.concatenate([ss, ns])
        y = torch.tensor(np.concatenate([np.ones(len(sh)), (ns == np.repeat(ss, k)).astype(float)]), dtype=torch.float)
        logit = dec(z[torch.tensor(hs)], z[D.H + torch.tensor(sy)], pair(hs, sy))
        loss = F.binary_cross_entropy_with_logits(logit, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 4:
            S = score_all(enc, dec, X, ei_full, et_full, D, val_h, pair)
            mrr = rank_metrics(S, val_h, val_s)["mrr"]
            if mrr > best:
                best, patience = mrr, 0
                best_state = ({k: v.clone() for k, v in enc.state_dict().items()},
                              {k: v.clone() for k, v in dec.state_dict().items()})
            else:
                patience += 1
                if patience >= 5:
                    break
    enc.load_state_dict(best_state[0]); dec.load_state_dict(best_state[1])

    def scorer(hosp):
        return score_all(enc, dec, X, ei_full, et_full, D, hosp, pair), None
    return scorer, {"val_mrr": best, "epochs": ep + 1}


@torch.no_grad()
def score_all(enc, dec, X, ei, et, D, hosp, pair):
    enc.eval(); dec.eval()
    z = enc(X, ei, et)
    S = np.full((D.H, D.S), -1e9, np.float32)
    zs = z[D.H:D.H + D.S]
    for chunk in np.array_split(hosp, max(1, len(hosp) // 200)):
        hs = np.repeat(chunk, D.S); ss = np.tile(np.arange(D.S), len(chunk))
        S[chunk] = dec(z[torch.tensor(hs)], zs[torch.tensor(ss)], pair(hs, ss)).view(len(chunk), D.S).numpy()
    return S


# ----------------------------------------------------------------------------- experiments
def split_random(D: Data):
    lab = D.lab[2022]
    ccns = np.array(sorted(lab))
    rng = np.random.default_rng(SEED)
    rng.shuffle(ccns)
    n = len(ccns)
    tr, va, te = ccns[: int(.7 * n)], ccns[int(.7 * n): int(.85 * n)], ccns[int(.85 * n):]
    idx = lambda cs: (np.array([D.hidx[c] for c in cs]), np.array([D.sidx[lab[c]] for c in cs]))
    return lab, idx(tr), idx(va), idx(te), tr


def split_temporal(D: Data):
    old, new = D.lab[2018], D.lab[2022]
    changed = np.array(sorted(c for c, s in new.items() if old.get(c) != s))
    ccns = np.array(sorted(old))
    rng = np.random.default_rng(SEED)
    rng.shuffle(ccns)
    va_c, tr_c = ccns[: int(.1 * len(ccns))], ccns[int(.1 * len(ccns)):]
    idx = lambda cs, lab: (np.array([D.hidx[c] for c in cs]), np.array([D.sidx[lab[c]] for c in cs]))
    stay = np.array(sorted(c for c, s in new.items() if old.get(c) == s))
    return old, idx(tr_c, old), idx(va_c, old), idx(changed, new), tr_c, {
        "test_changed": len(changed), "new_to_system": int(sum(c not in old for c in changed)),
        "moved_between_systems": int(sum(c in old for c in changed)), "stayers": len(stay),
        "persistence_acc_all_2022": float(len(stay) / len(new))}


METHODS = {"name_rule": name_rule, "spectral": spectral, "gbm_votes": gbm_votes,
           "sage": lambda *a, **k: gnn(*a, kind="sage", **k), "rgcn": lambda *a, **k: gnn(*a, kind="rgcn", **k)}


def run(D: Data, split: str, configs, methods):
    if split == "random":
        lab, (trh, trs), (vah, vas), (teh, tes), tr_ccn = split_random(D)
        info = {"train": len(trh), "val": len(vah), "test": len(teh)}
    else:
        lab, (trh, trs), (vah, vas), (teh, tes), tr_ccn, info = split_temporal(D)
    L = D.label_matrix(lab, tr_ccn)
    results, preds = {}, {}
    for cfg in configs:
        inc = D.incidence(CONFIGS[cfg])
        for m in methods:
            if m == "name_rule" and cfg != "full":
                continue  # graph-free; identical across configs
            t0 = time.time()
            fn = METHODS[m]
            if m in ("gbm_votes", "sage", "rgcn"):
                scorer, extra = fn(D, inc, L, train_h=trh, train_s=trs, val_h=vah, val_s=vas)
                S, _ = scorer(teh)
            else:
                S, extra = fn(D, inc=inc, L=L), None
            res = rank_metrics(S, teh, tes)
            res["seconds"] = round(time.time() - t0, 1)
            if extra:
                res["extra"] = extra
            key = f"{m}|{cfg}"
            results[key] = res
            top = np.argsort(-S[teh], axis=1)[:, :5]
            preds[key] = pd.DataFrame({"ccn": D.h_ccn[teh], "true": D.s_id[tes],
                                       **{f"top{i + 1}": D.s_id[top[:, i]] for i in range(5)}})
            log(f"[{split}] {key:22s} MRR {res['mrr']:.3f}  H@1 {res['hits1']:.3f}  H@10 {res['hits10']:.3f}  "
                f"AUROC {res['auroc']:.3f}  ({res['seconds']}s)")
    if split == "temporal":
        old = D.lab[2018]
        S = np.full((D.H, D.S), -1.0, np.float32)
        for h in teh:
            c = D.h_ccn[h]
            if c in old:
                S[h, D.sidx[old[c]]] = 1.0
        results["persistence|-"] = rank_metrics(S, teh, tes)
    return results, preds, info


def main():
    D = Data()
    log(f"hospitals {D.H}, systems {D.S}, labels {[(y, len(v)) for y, v in D.lab.items()]}")
    out = {"systems": D.S, "hospitals": D.H}
    methods = list(METHODS)
    if "--fast" in sys.argv:
        methods = ["name_rule", "spectral", "gbm_votes"]
    allpreds = {}
    for split in ("random", "temporal"):
        res, preds, info = run(D, split, list(CONFIGS), methods)
        out[split] = {"info": info, "results": res}
        for k, v in preds.items():
            v["split"], v["model"] = split, k
            allpreds[f"{split}|{k}"] = v
    save_metrics("linkpred", out)
    pd.concat(allpreds.values()).to_parquet(CLEAN / "linkpred_predictions.parquet", index=False)
    rows = []
    for split in ("random", "temporal"):
        for k, r in out[split]["results"].items():
            m, cfg = k.split("|")
            rows.append({"split": split, "method": m, "edges": cfg, **{x: r[x] for x in
                         ("n", "mrr", "hits1", "hits5", "hits10", "median_rank", "auroc")}})
    pd.DataFrame(rows).round(3).to_csv(TABLES / "linkpred_results.csv", index=False)
    log(json.dumps(out["temporal"]["info"]))


if __name__ == "__main__":
    main()
