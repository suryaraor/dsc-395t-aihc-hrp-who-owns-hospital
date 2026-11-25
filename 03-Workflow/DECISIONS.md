# Design Decisions

Each entry: the question, the options considered, what was chosen, and why.

## D-01 Environment: a dedicated `venv-graph` (Python 3.11, CPU PyTorch)
- **Options:** reuse `venv-nlp`; system Python 3.14; new venv.
- **Chosen:** new `venv-graph`, built with `uv`, Python 3.11, `torch==2.5.1` CPU wheel, PyTorch Geometric, DuckDB, rapidfuzz, sentence-transformers, statsmodels, LightGBM.
- **Why:** PyTorch Geometric wheels lag new Python releases, so 3.14 is risky. Reusing `venv-nlp` would tie this project to that one's pins (`numpy<2`). A CPU build is enough for a graph of about 10k nodes.

## D-02 Pipeline as numbered scripts, not notebooks
- **Options:** five notebooks as in the plan; numbered `.py` scripts plus a runner.
- **Chosen:** scripts `00_` to `07_` with `run_all.py`.
- **Why:** `nbconvert --execute` is blocked by App Control on this machine; scripts are easier to diff, rerun, and cite in the report. A notebook can still be generated for the submission link.

## D-03 Ground truth: AHRQ Compendium 2022 (train/test) and 2018 (temporal split)
- **Options:** 2022 only with a random split; 2018 to 2022 temporal split; 2020 to 2022.
- **Chosen:** 2018 to 2022 (see issue I-04 for why 2020 was out).
- **Why:** The temporal split is the "high-risk" test the plan promised: can a model trained on 2018 links predict the systems hospitals belong to in 2022, including ones that changed?

## D-04 Outcomes from the current Care Compare release
- **Options:** archived 2023 Care Compare (not scriptable, I-05); current release; drop D2.
- **Chosen:** current release, with FY2022 cost reports for the price and cost proxies.
- **Why:** Star ratings and readmissions move slowly.

## D-05 Matching labels from CMS Hospital Enrollments, not NPPES other-IDs
- **Options:** NPPES "Other Provider Identifier" (plan); about 150 hand-labeled pairs; CMS Enrollments CCN-NPI pairs.
- **Chosen:** Enrollments (7,269 pairs, 4,462 inside the matched universe).
- **Why:** The NPPES fields are empty (I-13). Enrollments is an official CMS crosswalk and covers almost every hospital.

## D-07 Facility-level match target with label expansion
- **Options:** strict (exactly the enrolled NPI); facility-level (any NPI of that facility).
- **Chosen:** facility-level, since the downstream graph needs *every* NPPES record of a hospital (authorized officials, parent LBNs).
- **Why:** The strict target punished correct sibling-NPI matches (I-14). The expansion rule is deterministic and independent of the classifier.

## D-08 Edge tiers for the leakage ablation
- **Tier A, ownership (near-label):** CMS All Owners organizational owners (shared `ASSOCIATE ID - OWNER`).
- **Tier B, shared people:** CMS individual owners, officers and managing employees (shared PECOS associate ID), plus NPPES authorized officials (same normalized name + phone).
- **Tier C, weak public signals:** NPPES parent-organization LBN, same city, distinctive-name-token overlap.
- **Ablation:** full (A+B+C), without A (B+C), C only. This replaces the single "with vs without leaky edges" comparison in the plan with a graded one. Using later outcomes against 2022 structure is a defensible "structure, then outcome" ordering, and the gap is stated as a limitation.

## D-12 Review cycle: fix the system-size leak, add an A+C ablation, and disclose the LLM provenance (2026-11-22)
- **Options:** keep the headline numbers and add caveats; re-run the affected stages and report the new numbers.
- **Chosen:** re-run. `sys_log_size` (label-year AHRQ `hosp_cnt`) was dropped from the GBM features and the GNN system features; size now comes from training labels only. An A+C (no people) configuration was added. Stage 05 gained CAH and children's-hospital controls and a PPS-only sample. Stage 06 now uses training labels only and adds a `system_reid` category. A new stage 08 (`08_robustness.py`) writes the audit baselines, the McNemar test, the different-street matching score, bootstrap CIs and the real-move temporal score.
- **Why:** a structured review of the draft found the leak, the circular silver labels and the non-blind LLM step. Every new number has to come from a metrics file, so the robustness checks became a stage, not hand calculations.
- **Deferred:** a blinded re-run of the LLM adjudication, done once an API key was available (D-13).

## D-13 Blind LLM adjudication through the API replaces the earlier non-blind verdicts (2026-11-22)

- **Chosen:** re-ran stage 02 with `ANTHROPIC_API_KEY` set: 216 calls to `claude-haiku-5-5`, record text only, structured JSON output. Cached to `llm_verdicts_api.json`; stages 03–08 re-run; report, slides, script and README updated.
- **Result:** audit accuracy 0.58 (C2) → 0.76 (C3), McNemar p = 0.04 (non-blind was 0.81, p = 0.006). Blind verdicts agree with the old ones on 166/216 pairs and say "match" less often (53% vs 75%). Crosswalk 6,714 → 6,705 hospitals; GBM random MRR 0.886 → 0.881; GNN scores moved several points (single-seed noise). Quality/cost unchanged. Pre-blind metrics kept in `06-Results/metrics/pre_blind/`.
- **Code:** `temperature` removed (rejected by Haiku 5.5); structured output via `output_config.format`; stage 06 now reads whichever verdict file stage 02 used.
