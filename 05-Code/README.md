# Who Owns the Hospital? Code

Pipeline for the DSC 395T High-risk Project. It covers four parts:

- LLM-assisted entity resolution (record matching) across CMS and NPPES
- a hospital–entity graph
- hospital-to-health-system link prediction, with AHRQ Compendium labels as ground truth
- a companion quality/cost association analysis

Every input is a public, organization-level file. No patient data or PHI is used anywhere.

Abbreviations: **CMS** Centers for Medicare & Medicaid Services · **CCN** CMS Certification Number · **NPPES / NPI** national provider registry / National Provider Identifier · **AHRQ** Agency for Healthcare Research and Quality · **POS** Provider of Services file · **HCRIS** cost reports · **ER** entity resolution (not emergency room).

## Prerequisites

- Python 3.11, [`uv`](https://docs.astral.sh/uv/) (`pip install uv`), and `curl` on PATH (the downloader uses it)
- About 15 GB of free disk: the NPPES zip is about 1 GB and its temporary extract is 11.8 GB. 16 GB RAM is recommended.
- Internet access on the first run: the data downloads, plus the `all-MiniLM-L6-v2` model from Hugging Face

## Setup

Run these from the project root, the folder that contains `05-Code/`:

```bash
uv venv venv-graph --python 3.11
uv pip install --python venv-graph torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python venv-graph -r 05-Code/requirements.txt
# activate: Windows  venv-graph\Scripts\activate     macOS/Linux  source venv-graph/bin/activate
cd 05-Code
python run_all.py            # or: python run_all.py --from 02
```

The pipeline runs on CPU only.

## Run order

| Stage | Script | What it does | Main outputs | Approx. time |
|---|---|---|---|---|
| 00 | `00_download.py` | Downloads every source and writes a SHA-256 manifest | `04-Data/raw/*`, `_manifest.json` | 5–15 min (network) |
| 01 | `01_clean.py` | Cleans and joins AHRQ, POS, cost report, Care Compare, owners and enrollments | `04-Data/clean/*.parquet`, `metrics/clean.json` | 1 min |
| 01b | `01b_nppes.py` | Streams the 11.8 GB NPPES file through DuckDB down to 34k hospital organizations | `nppes_orgs.parquet` | 5 min |
| 02 | `02_entity_resolution.py` | Blocking, fuzzy + embedding features, LightGBM, LLM adjudication of the uncertain band | `crosswalk_ccn_npi.parquet`, `metrics/er.json` | 5 min (first run embeds) |
| 03 | `03_graph.py` | Hospital–entity graph in three leakage tiers | `graph_edges.parquet`, `metrics/graph.json` | <1 min |
| 04 | `04_link_prediction.py` | Name rule, spectral, GBM + neighbor votes, GraphSAGE, R-GCN; random and temporal splits; 4-way tier ablation | `metrics/linkpred.json`, `tables/linkpred_results.csv` | 10 min |
| 05 | `05_quality_cost.py` | Ordered logit / OLS with state fixed effects; stable and PPS-only sensitivity samples | `metrics/quality_cost.json` | 1 min |
| 06 | `06_error_analysis.py` | Error taxonomy for matching and link prediction (training labels only) | `tables/errors_*.csv`, `metrics/errors.json` | <1 min |
| 07 | `07_figures.py` | Report figures from the metrics files | `06-Results/figures/*.png/pdf` | <1 min |
| 08 | `08_robustness.py` | Audit baselines and McNemar test, different-street matching, bootstrap CIs, real-move temporal score | `metrics/robustness.json` | <1 min |

**Fast path.** If `04-Data/clean/` already holds `nppes_orgs.parquet`, `emb_cache.npz` and `name_emb_cache.npz`, run `python run_all.py --from 02`; it takes about 20 minutes.

**Check your run.** You should get these headline numbers: matcher C2 F1 0.922; crosswalk covers 6,705 hospitals; GBM-votes random-split MRR 0.881 (Hits@1 0.851); temporal MRR 0.412 (Hits@1 0.336); in-system star β 0.92. From stage 08 (`robustness.json`): audit accuracy C2 0.58 vs C3 0.76 (McNemar p 0.04); different-street C2 F1 0.26; temporal Hits@1 0.52 for first-time joiners and 0.26 for system switchers. GNN numbers can move by a few hundredths across machines, because PyTorch CPU kernels are not bit-identical.

`run_all.py` stops before stage 01 if any AHRQ file is missing; see "Manual step" below.

## Manual step: AHRQ files

ahrq.gov blocks scripted downloads (HTTP 403), and the downloader marks those files `manual-needed`; it never spoofs a browser. Download these four files in a normal browser from `https://www.ahrq.gov/sites/default/files/wysiwyg/chsp/compendium/` and save them in `04-Data/raw/` under the names shown:

| Download | Save as |
|---|---|
| `chsp-hospital-linkage-2022-rev.csv` | `ahrq_hospital_linkage_2022.csv` |
| `chsp-compendium-2022.csv` | `ahrq_systems_2022.csv` |
| `chsp-hospital-linkage-2018.csv` | `ahrq_hospital_linkage_2018.csv` |
| `chsp-compendium-2018.csv` | `ahrq_systems_2018.csv` |

## Data versions

Results used the **November 2026** NPPES monthly file. CMS removes old monthly files, so if the link returns 404, use the oldest monthly file still listed at download.cms.gov/nppes. For alignment with the 2022 labels, a file close to December 2022 would be better. Care Compare is fetched as the *current* release, so outcome values drift with each refresh. CMS ownership, enrollment and POS files are pinned to December 2022.

## LLM adjudication: what was actually run

- **Reported run: blind, through the API.** The report's numbers use `04-Data/clean/llm_verdicts_api.json`: 216 calls to `claude-haiku-5-5` (2026-11-22), structured JSON output, effort `low`. Stage 02 uses this file whenever `ANTHROPIC_API_KEY` is set, and makes new calls only for pairs not already cached, so a rerun with the key reproduces the report exactly at no cost.
- **What the API prompt contains.** The match rule "match = this NPPES record is this hospital facility (its legal or former name, or its own units); co-located LTACHs, physician groups and other hospitals are not." plus each record's organizational NPPES/CMS fields: names, practice address, taxonomy and the subpart flag. No owners, officials, model scores, labels or patient data.
- **Earlier, non-blind verdicts.** `llm_verdicts.json` holds the first verdicts, made without blinding: the adjudicator could see the C1/C2 scores. They scored 0.81 on the audit against 0.76 for the blind run. Stage 02 falls back to them only when no API key is set, which will not reproduce the report's numbers. They are kept for transparency.
- **Commit both caches.** Without the key and without either file, stage 02 stops with an error rather than silently scoring the band without the LLM.

## Sharing the code

Push `05-Code/`, `03-Workflow/`, `06-Results/`, and these three files from `04-Data/clean/`: `llm_verdicts_api.json`, `llm_verdicts.json` and `llm_band_pairs.json`. All hold organizational records only; committing the API cache lets others reproduce the report without an API key or cost. Do **not** push the rest of `04-Data/`. Its raw files are large, and `owners.parquet` and `graph_edges.parquet` contain the names of individual owners, officers and authorized officials. These are public CMS/NPPES records, but linking them across files should not be republished. Anyone can rebuild the rest of `04-Data/` with stage 00. The **fast path** below needs locally built caches, so a fresh clone runs the full pipeline once.

## Notes

- **Reproducibility.** Seed 395 everywhere. Every number in the report comes from a file in `06-Results/metrics/`. Link-prediction results are from a single seed.
- Issues hit while building, and how each was fixed, are in `03-Workflow/ISSUES-LOG.md`. Choices between alternatives are in `03-Workflow/DECISIONS.md`.
