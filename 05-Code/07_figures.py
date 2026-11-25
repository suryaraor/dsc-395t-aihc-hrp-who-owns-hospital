"""Stage 07: report figures from 06-Results/metrics (nothing is hard-coded)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGS, load_metrics

INK, CORAL, TEAL, VIOLET, SUN, MUTE = "#0b1020", "#ff5a36", "#13b3a3", "#7b5cff", "#e0b000", "#6b6f80"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "savefig.dpi": 300, "savefig.bbox": "tight"})


def er():
    m = load_metrics("er")
    r = m["results_test"]
    names = {"rule_name90_zip": "Rule\n(name≥90 + ZIP)", "C1_fuzzy_LR": "C1 fuzzy\n(LogReg)",
             "C2_fuzzy_emb_LGBM": "C2 + embedding\n(LightGBM)", "C3_C2_plus_LLM": "C3 + LLM\nadjudication"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [2.3, 1]})
    x = np.arange(len(names))
    for i, (k, c) in enumerate((("precision", INK), ("recall", TEAL), ("f1", CORAL))):
        v = [r[n][k] for n in names]
        a.bar(x + (i - 1) * 0.26, v, 0.25, color=c, label=k.upper() if k == "f1" else k.capitalize())
        for xi, vi in zip(x, v):
            a.text(xi + (i - 1) * 0.26, vi + 0.01, f"{vi:.2f}", ha="center", fontsize=6.2)
    a.set_xticks(x, names.values(), fontsize=7)
    a.set_ylim(0, 1.25); a.set_yticks([0, .2, .4, .6, .8, 1.0]); a.set_ylabel("Test score (silver labels)")
    a.legend(frameon=False, ncol=3, loc="upper left", fontsize=7, bbox_to_anchor=(0, 1.02))
    a.set_title("Entity resolution, CCN ↔ NPI", loc="left", fontsize=9, fontweight="bold")
    ob = m["llm_band"]["objective_audit"]
    b.bar([0, 1], [ob["acc_c2"], ob["acc_c3"]], color=[MUTE, CORAL], width=0.6)
    for xi, v in enumerate([ob["acc_c2"], ob["acc_c3"]]):
        b.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
    b.set_xticks([0, 1], ["C2 alone", "C2 + LLM"])
    b.set_ylim(0, 1); b.set_ylabel("Accuracy")
    b.set_title(f"Uncertain band, objective\naudit (n={ob['n']})", loc="left", fontsize=9, fontweight="bold")
    fig.savefig(FIGS / "fig_er.png"); fig.savefig(FIGS / "fig_er.pdf"); plt.close(fig)


def linkpred():
    m = load_metrics("linkpred")
    methods = [("name_rule", "Name rule"), ("spectral", "Spectral (SVD)"), ("sage", "GraphSAGE"),
               ("rgcn", "R-GCN"), ("gbm_votes", "GBM + neighbor votes")]
    cfgs = [("full", "All edges (A+B+C)", INK), ("noA", "No ownership (B+C)", TEAL),
            ("noB", "No people (A+C)", VIOLET), ("C_only", "Name/geo only (C)", CORAL)]
    w = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, split, title in ((axes[0], "random", "Random split (2022 labels)"),
                             (axes[1], "temporal", "Temporal: hospitals that changed system 2018→2022")):
        res = m[split]["results"]
        x = np.arange(len(methods))
        for i, (cfg, lab, col) in enumerate(cfgs):
            v = []
            for k, _ in methods:
                # the name rule uses no graph edges, so it is drawn once (grey)
                v.append(res.get(f"{k}|{cfg}", {}).get("mrr", np.nan) if k != "name_rule" else np.nan)
            ax.bar(x + (i - 1.5) * w, v, w * 0.95, color=col, label=lab)
        nr = res["name_rule|full"]["mrr"]
        ax.bar(0, nr, w * 1.6, color=MUTE)
        ax.text(0, nr + 0.02, f"{nr:.2f}", ha="center", fontsize=6)
        ax.set_xticks(x, [n for _, n in methods], rotation=20, ha="right", fontsize=7.5)
        ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold")
        ax.set_ylim(0, 1); ax.set_yticks([0, .2, .4, .6, .8, 1.0])
    axes[0].set_ylabel("MRR (mean of 1 / rank of the true\nsystem among %d; 1 = always first)" % m["systems"], fontsize=7)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=6.5, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.07))
    ti = m["temporal"]["info"]
    axes[1].text(0.98, 0.97, f"test = {ti['test_changed']} changers only.\nKeeping the 2018 system is right for\n"
                 f"{ti['persistence_acc_all_2022']:.0%} of all 2022 hospitals, 0% here.",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=6, color=MUTE)
    fig.savefig(FIGS / "fig_linkpred.png"); fig.savefig(FIGS / "fig_linkpred.pdf"); plt.close(fig)


SHORT = {"star": "Star rating", "readmit_hwr": "Readmission", "mspb": "Medicare spend\nper beneficiary",
         "log_markup": "Charges / cost", "log_rev_dsch": "Net revenue\nper discharge"}
UNITS = {"star": "log-odds (+ = better)", "readmit_hwr": "pct. points (- = better)",
         "mspb": "ratio, 1 = national (- = less)", "log_markup": "log points (+ = higher)",
         "log_rev_dsch": "log points"}


def forest():
    m = load_metrics("quality_cost")
    t = pd.DataFrame(m["results"])
    t = t[t["sample"] == "all"]
    labels = m["outcome_labels"]
    ex = {"in_system": ("In a system", INK), "log_sys_size": ("log system size", TEAL),
          "log_local_sys": ("log same-state siblings", CORAL)}
    fig, axes = plt.subplots(1, len(labels), figsize=(7.2, 2.3), gridspec_kw={"wspace": 0.25})
    for ax, (y, lab) in zip(axes, labels.items()):
        g = t[t["outcome"] == y]
        for j, (x, (xl, c)) in enumerate(ex.items()):
            r = g[g["exposure"] == x].iloc[0]
            ax.errorbar(r["beta"], -j, xerr=[[r["beta"] - r["lo"]], [r["hi"] - r["beta"]]], fmt="o", color=c,
                        ms=4, capsize=2, lw=1.2)
        ax.axvline(0, color=MUTE, lw=0.8, ls="--")
        ax.set_yticks([0, -1, -2], [v[0] for v in ex.values()] if ax is axes[0] else [""] * 3, fontsize=7)
        ax.set_title(f"{SHORT.get(y, lab)}\nn = {int(g['n'].max()):,}", fontsize=7, fontweight="bold")
        ax.set_xlabel(UNITS[y], fontsize=5.8)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.set_ylim(-2.6, 0.6)
    fig.suptitle("Association with system structure (β, 95% CI; controls incl. CAH + state FE; OLS SEs clustered by state).\n"
                 "Rows are on different scales: in a system = yes/no; size and siblings = per log unit (0 if independent).",
                 fontsize=7, x=0.01, ha="left", y=1.25)
    fig.savefig(FIGS / "fig_forest.png"); fig.savefig(FIGS / "fig_forest.pdf"); plt.close(fig)


def graph_and_errors():
    g = load_metrics("graph")
    e = load_metrics("errors")
    rel = g["by_relation"]
    tier = {"owner_org": "A", "person_cms": "B", "official": "B", "parent_lbn": "C", "city": "C", "name_token": "C"}
    col = {"A": CORAL, "B": VIOLET, "C": TEAL}
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.45})
    order = list(tier)
    a.barh(range(len(order)), [rel[r]["edges"] for r in order], color=[col[tier[r]] for r in order])
    for i, r in enumerate(order):
        a.text(rel[r]["edges"] + 300, i, f"{rel[r]['edges']:,} · {rel[r]['hospitals']:,} hosp.", va="center",
               fontsize=6.5)
    a.set_yticks(range(len(order)), [f"[{tier[r]}] {r}" for r in order], fontsize=7)
    a.invert_yaxis(); a.set_xlim(0, max(v["edges"] for v in rel.values()) * 1.4); a.set_xlabel("number of edges (bar labels: edges · hospitals with that edge)", fontsize=7)
    a.set_title("Edges by relation (tier A/B/C)", loc="left", fontsize=8.5, fontweight="bold")
    causes = ["isolated", "successor_sys", "cold_start", "no_signal", "rival_neighbors", "merger_move", "other"]
    x = np.arange(len(causes))
    for i, (split, c) in enumerate((("random", INK), ("temporal", CORAL))):
        tot = e[split]["misses"]
        b.bar(x + (i - .5) * 0.38, [e[split]["causes"].get(k, 0) / tot for k in causes], 0.36, color=c,
              label=f"{split} ({tot} misses / {e[split]['test']})")
    b.set_xticks(x, ["no edges", "successor", "cold start", "no link", "rival nbrs", "merger move", "other"],
                 fontsize=8, rotation=40, ha="right", rotation_mode="anchor")
    b.set_ylabel("Share of top-1 misses")
    b.legend(frameon=False, fontsize=6.5, loc="upper right"); b.set_ylim(0, 1.0)
    b.set_title("Why the best model missed", loc="left", fontsize=8.5, fontweight="bold")
    fig.savefig(FIGS / "fig_graph_errors.png"); fig.savefig(FIGS / "fig_graph_errors.pdf"); plt.close(fig)


def workflow():
    """Figure 1: the pipeline as it actually ran, with counts pulled from the metrics files."""
    from matplotlib.patches import FancyBboxPatch
    er_m, g, lp, qc = load_metrics("er"), load_metrics("graph"), load_metrics("linkpred"), load_metrics("quality_cost")
    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")
    cols = [(0.5, "1  Public data"), (20.5, "2  Ingest + clean"), (40.5, "3  Entity resolution"),
            (60.5, "4  Graph + learning"), (80.5, "5  Analysis")]
    for x, t in cols:
        ax.add_patch(FancyBboxPatch((x, 0.5), 18.6, 51, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc="#f6f4ee", ec="#d8d3c6", lw=0.6, ls="--"))
        ax.text(x + 1, 50.3, t, fontsize=6.0, fontweight="bold", color=MUTE, va="top")

    def box(x, y, title, sub, c, fc="white", h=6.6, w=17):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc=fc, ec=c, lw=1.0, zorder=3))
        ax.text(x + w / 2, y + h - 1.1, title, ha="center", va="top", fontsize=5.0, fontweight="bold", color=INK, zorder=4)
        ax.text(x + w / 2, y + 0.9, sub, ha="center", va="bottom", fontsize=4.1, color="#3b3f52", linespacing=1.2, zorder=4)
        return dict(l=x, r=x + w, b=y, t=y + h, cx=x + w / 2, cy=y + h / 2)

    def path(pts, c=INK, ls="-", lab=None, at=0, dx=0.6, dy=0.5, ha="left"):
        xs, ys = zip(*pts)
        ax.plot(xs[:-1] + (xs[-1],), ys, color=c, lw=0.8, ls=ls, zorder=2, solid_capstyle="butt")
        ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=0.8, mutation_scale=6, shrinkA=0, shrinkB=0))
        if lab:
            ax.text(pts[at][0] + dx, pts[at][1] + dy, lab, fontsize=4.3, color=c, ha=ha, va="bottom", style="italic")

    P = "#efeaff"
    ahrq = box(1.3, 38, "AHRQ Compendium", "2018 + 2022 hospital-system\nlinkage files (labels)", VIOLET, P)
    cms = box(1.3, 29, "POS + Care Compare", "CMS facility records; stars,\nreadmission, MSPB", VIOLET, P)
    own = box(1.3, 20, "Enrollments + Owners", "CMS CCN-NPI pairs; owner\norganizations + people", VIOLET, P)
    npp = box(1.3, 11, "NPPES registry", "11.8 GB CSV, every NPI", VIOLET, P)
    hc = box(1.3, 2, "HCRIS cost reports", "FY2022 charges, costs,\nnet revenue", VIOLET, P)
    cl = box(21.3, 29, "01 clean", f"{er_m['hospitals']:,} hospitals; Latin-1 fix;\nCCN / ZIP / phone keys", TEAL)
    nf = box(21.3, 11, "01b DuckDB filter", f"type-2 orgs, hospital taxonomy\n-> {er_m['nppes_active_orgs']:,} organizations", TEAL)
    bl = box(41.3, 38, "02 blocking", f"state AND (ZIP3 | city | phone)\n{er_m['candidate_pairs']:,} pairs, recall {er_m['blocking_recall']:.3f}", SUN)
    ft = box(41.3, 29, "02 features + matcher", "rapidfuzz + MiniLM cosine;\nLightGBM on Enrollments labels", SUN)
    llm = box(41.3, 20, "LLM adjudication", f"Claude labels the {er_m['llm_band']['n']} test pairs\nwith p in [0.3, 0.7]", CORAL, "#fff0eb")
    cw = box(41.3, 11, "CCN-NPI crosswalk", f"{er_m['crosswalk_hospitals']:,} hospitals\n({er_m['coverage']:.1%} coverage)", SUN)
    lpb = box(61.3, 29, "04 link prediction", f"rank all {lp['systems']} systems:\nname rule, SVD, GBM votes,\nSAGE, R-GCN; random +\ntemporal splits; tier ablation", CORAL, h=9.2)
    gr = box(61.3, 11, "03 hospital graph", f"{g['edges_total']:,} edges in tiers\nA owner / B people / C weak", CORAL)
    qcb = box(81.3, 33, "05 quality + cost", f"AHRQ 2022 membership;\nordered logit + OLS, state FE;\n{qc['sample']:,} acute hospitals", TEAL, h=8.6)
    erb = box(81.3, 19, "06 error analysis", "miss-cause taxonomy;\nmatching false pos / neg", TEAL)
    fg = box(81.3, 5, "07 figures + report", "every number read from\n06-Results/metrics/*.json", TEAL, "#e9f8f6")

    # column 1 -> 2
    path([(cms["r"], cms["cy"]), (cl["l"], cms["cy"])])
    path([(own["r"], own["cy"]), (20.0, own["cy"]), (20.0, cl["b"] + 1.5), (cl["l"], cl["b"] + 1.5)])
    path([(npp["r"], npp["cy"]), (nf["l"], npp["cy"])])
    # column 2 -> blocking (shared bus at x=39.8)
    path([(nf["r"], nf["cy"]), (39.8, nf["cy"]), (39.8, bl["cy"]), (bl["l"], bl["cy"])], lab="orgs", at=1, dx=-0.4, dy=1, ha="right")
    ax.plot([cl["r"], 39.8], [cl["cy"] + 1.5, cl["cy"] + 1.5], color=INK, lw=0.8, zorder=2)
    ax.text(39.3, cl["cy"] + 4.5, "hospitals", fontsize=4.3, style="italic", color=INK, rotation=90, ha="center", va="bottom")
    # column 3 chain
    path([(bl["cx"], bl["b"]), (ft["cx"], ft["t"])])
    path([(ft["cx"], ft["b"]), (llm["cx"], llm["t"])], lab="uncertain", at=0, dx=0.6, dy=-1.9)
    path([(llm["cx"], llm["b"]), (cw["cx"], cw["t"])], c=CORAL, lab="verdicts", at=0, dx=0.6, dy=-2.6)
    path([(ft["r"], ft["cy"] - 2), (59.6, ft["cy"] - 2), (59.6, cw["cy"] + 1.5), (cw["r"], cw["cy"] + 1.5)], c=MUTE)
    ax.text(60.25, 21, "confident", fontsize=4.3, style="italic", color=MUTE, rotation=90, ha="center", va="bottom")
    # into the graph
    path([(cw["r"], cw["cy"] - 1.5), (gr["l"], cw["cy"] - 1.5)], lab="officials", at=0, dx=0.4, dy=-2.6)
    path([(cl["cx"], cl["b"]), (cl["cx"], 18.8), (gr["cx"] - 3, 18.8), (gr["cx"] - 3, gr["t"])], lab="owners, people, names", at=1, dx=0.6, dy=0.2)
    # graph -> link prediction; AHRQ labels over the top
    path([(gr["cx"] + 3, gr["t"]), (gr["cx"] + 3, lpb["b"])], lab="typed edges", at=0, dx=0.6, dy=6)
    path([(ahrq["cx"], ahrq["t"]), (ahrq["cx"], 46.6), (lpb["cx"], 46.6), (lpb["cx"], lpb["t"])], c=VIOLET, ls="--",
         lab="system labels (random split: 2022; temporal: 2018 -> 2022)", at=1, dx=4, dy=0.3)
    # stage 05 reads AHRQ 2022 membership directly, not the predicted links
    path([(lpb["cx"], 46.6), (qcb["cx"], 46.6), (qcb["cx"], qcb["t"])], c=VIOLET, ls="--",
         lab="2022 membership", at=1, dx=-9.5, dy=0.3)
    path([(lpb["r"], lpb["b"] + 1.5), (79.9, lpb["b"] + 1.5), (79.9, erb["cy"]), (erb["l"], erb["cy"])], lab="ranks", at=0, dx=0.2, dy=0.3)
    path([(hc["r"], hc["cy"]), (80.6, hc["cy"]), (80.6, qcb["b"] + 1.5), (qcb["l"], qcb["b"] + 1.5)], c=MUTE,
         lab="cost outcomes", at=0, dx=22, dy=0.3)
    path([(qcb["cx"], qcb["b"]), (erb["cx"], erb["t"])])
    path([(erb["cx"], erb["b"]), (fg["cx"], fg["t"])])
    for d in (FIGS, FIGS.parents[1] / "07-Report" / "figures"):
        fig.savefig(d / "fig_workflow.png"); fig.savefig(d / "fig_workflow.pdf")
    plt.close(fig)


if __name__ == "__main__":
    er(); linkpred(); forest(); graph_and_errors(); workflow()
    print("figures:", sorted(p.name for p in FIGS.glob("*.png")))
