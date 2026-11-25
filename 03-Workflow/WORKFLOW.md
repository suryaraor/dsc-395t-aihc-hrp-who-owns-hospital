# Project Workflow: Mapping the Hidden Hospital Network

How the project was run. Each phase has inputs, a script, outputs and an exit check. The issues log (`ISSUES-LOG.md`) and decisions log (`DECISIONS.md`) were updated during every phase.

## Folder map

| Folder | What lives there |
|---|---|
| `03-Workflow/` | This file, `ISSUES-LOG.md`, `DECISIONS.md` |
| `04-Data/` | `raw/` (downloads + `_manifest.json`), `clean/` (parquet). Only the manifest and LLM caches are in the repo; stage 00 rebuilds the rest |
| `05-Code/` | Numbered pipeline scripts `00_` to `08_`, `common.py`, `run_all.py`, `requirements.txt`, `README.md` |
| `06-Results/` | `metrics/*.json`, `tables/*.csv`, `figures/*.png` |
| `07-Report/` | Final report (ACM `sigconf`) |
| `08-Slides/` | Slide deck |

## Phase map

```
P0 Setup ─► P1 Ingest ─► P2 Clean ─► P3 Entity resolution ─► P4 Graph ─► P5 Link prediction ─┐
                                                                                              ├─► P7 Error analysis ─► P8 Report/Slides
                                                                 P6 Quality & cost analysis ──┘
```

| Phase | Script | Inputs | Outputs | Exit check |
|---|---|---|---|---|
| P0 Setup | (venv, folders) | plan | `venv-graph`, numbered folders | `import torch_geometric` works |
| P1 Ingest | `00_download.py` | public URLs | `04-Data/raw/*`, `_manifest.json` | every file non-empty and hashed |
| P2 Clean | `01_clean.py`, `01b_nppes.py` | raw CSVs, NPPES zip | `04-Data/clean/*.parquet` | row counts logged, CCNs 6-char |
| P3 Entity resolution | `02_entity_resolution.py` | clean hospitals + NPPES orgs | `crosswalk_ccn_npi.parquet`, `metrics/er.json` | P/R/F1 for 3 configs |
| P4 Graph | `03_graph.py` | clean tables, crosswalk | `graph_edges.parquet`, `metrics/graph.json` | node/edge counts per type |
| P5 Link prediction | `04_link_prediction.py` | graph, AHRQ labels | `metrics/linkpred.json`, tables | baselines + GNN, random + temporal, leakage ablation |
| P6 Quality & cost | `05_quality_cost.py` | Care Compare, cost report, AHRQ | `metrics/quality_cost.json`, forest plot | coefficients with 95% CIs |
| P7 Error analysis | `06_error_analysis.py`, `08_robustness.py` | predictions | `tables/errors_*.csv`, `metrics/robustness.json` | top-20 FP/FN categorized |
| P8 Deliverables | `07_figures.py`, LaTeX, slides | everything | report PDF, slides | rubric checklist ticked |

## Rules for every phase

1. Scripts are numbered and re-runnable; each reads only from earlier stages.
2. Every number in the report comes from a file in `06-Results/metrics/`.
3. Any failure, surprise, or workaround goes in `ISSUES-LOG.md` when it happens (symptom, cause, fix, where).
4. Any choice between alternatives goes in `DECISIONS.md` (options, choice, why).
