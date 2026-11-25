# Who Owns the Hospital?

**LLM entity resolution and graph learning for U.S. hospital–system affiliations, with a companion analysis of quality and cost.**

DSC 395T AI in Healthcare (Fall 2026), High-risk Project. Surya Rayarao and Naga Donikena.

By 2018 about 72% of U.S. hospitals belonged to a health system, but no public file records these links cleanly. Using seven public, organization-level sources (CMS, NPPES, AHRQ), we:

1. **Match** hospitals (CCN) to registry organizations (NPI) with string and embedding features and a gradient-boosted classifier. The 216 most uncertain pairs go to an LLM; blinded LLM verdicts raise audit accuracy from 0.58 to 0.76 (McNemar p = 0.04).
2. **Predict** each hospital's health system from a hospital–entity graph. A gradient-boosted ranker over "neighbor votes" puts the true system first for 85% of held-out hospitals, ahead of GraphSAGE (52%), R-GCN (38%) and a name rule (34%).
3. **Relate** system membership to quality and cost: higher star ratings, slightly lower readmission and Medicare spending, and about 31% higher charge-to-cost markups. These are observational associations, not causal effects.

No patient data or PHI is used anywhere.

## Where to look

| Folder | Contents |
|---|---|
| [`07-Report/`](07-Report/) | Final report: [`main.pdf`](07-Report/main.pdf) (ACM `sigconf` format) |
| [`08-Slides/`](08-Slides/) | Slide deck: [`slides.pdf`](08-Slides/slides.pdf) and the same deck as `slides.pptx` |
| [`05-Code/`](05-Code/) | Pipeline scripts `00_`–`08_`, `run_all.py`, `requirements.txt`. **Setup and run instructions: [`05-Code/README.md`](05-Code/README.md)** |
| [`06-Results/`](06-Results/) | Every reported number (`metrics/*.json`), result tables (`tables/*.csv`) and figures. `metrics/pre_blind/` holds results from the earlier, non-blind LLM run |
| [`03-Workflow/`](03-Workflow/) | Workflow, design decisions (`DECISIONS.md`) and issues log (`ISSUES-LOG.md`) |
| [`04-Data/`](04-Data/) | Only the LLM adjudication caches and the download manifest (see below) |

## Reproducing

See [`05-Code/README.md`](05-Code/README.md). In short: Python 3.11, install `05-Code/requirements.txt`, download the four AHRQ files by hand, then run `python run_all.py` from `05-Code/`. Stage 00 downloads every other source and checks it against `04-Data/raw/_manifest.json`.

## What is not in this repository

- **Raw and cleaned data** (`04-Data/raw`, most of `04-Data/clean`). The files are large (the NPPES zip alone is about 1 GB), and anyone can rebuild them with stage 00. Some cleaned files also link named individual owners and officials across CMS/NPPES files. Those records are public, but we chose not to republish the linked version.
- **LLM caches that are included**: `04-Data/clean/llm_verdicts_api.json` (the blind API verdicts used in the report), `llm_verdicts.json` (earlier non-blind verdicts, kept for transparency) and `llm_band_pairs.json`. They hold organizational records only. With them, stage 02 reproduces the report without an API key.
