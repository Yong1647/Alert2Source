# Alert2Source
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://alert2source-gkrpy2ympcvodd5tznxkfe.streamlit.app/)

Alert2Source is a staged pipeline for explaining air-quality alerts with SHAP
evidence, source-knowledge RAG, optional satellite visual context, and ranked
source attribution.

This repository is a cleaned, GitHub-oriented snapshot of the working
experiment files. Raw satellite inputs and generated satellite crops are
excluded (see `.gitignore`); slimmed demo data and pre-computed intermediate
outputs are included so the report generation and evaluation stages can run
without retraining from scratch.

## Repository layout

```text
Alert2Source/
├── config/                       # sanitized config templates (europe_23feat.yaml)
├── src/                          # TabSatFusion / AQFusionNet model utilities
├── pipeline/
│   ├── 00_build_source_registry.py   # source_registry.csv + RAG JSONL builder
│   ├── 01_train_lgbm_aqsplit.py      # LightGBM aligned baseline/surrogate training
│   ├── 02_diagnostic_eligibility.py  # report eligibility diagnostics
│   ├── 03_shap_reliability.py        # SHAP feature/group evidence extraction
│   ├── 04_generate_reports.py        # report generator with ranked_sources
│   ├── 05_eval_ranked_sources.py     # ranked_sources evaluation vs CAMS gold
│   └── 06_visual_conditional_geval.py# report quality/safety G-Eval
├── dashboard/                    # Streamlit dashboard
├── scripts/                      # helper scripts (dashboard-slim report builder)
├── data/                         # slim demo data (raw inputs are gitignored)
└── outputs/                      # pre-computed intermediate outputs + placeholders
```

Local packages (`src/`) live at the repository root and are imported by the
pipeline scripts (e.g. stage `01` imports `src.utils`). Run the scripts from the
repository root.

## Main runnable path

The intended run order matches the numeric prefixes in `pipeline/`:

1. `00_build_source_registry.py` — build the feature→source registry and RAG cards.
2. `01_train_lgbm_aqsplit.py` — *(optional)* retrain the aligned LightGBM if starting from raw data.
3. `02_diagnostic_eligibility.py` — compute report eligibility from backbone predictions + aligned LGBM.
4. `03_shap_reliability.py` — compute SHAP feature and source-group evidence.
5. `04_generate_reports.py` — generate reports with `ranked_sources` (SHAP / +RAG / +RAG+Image conditions).
6. `05_eval_ranked_sources.py` — evaluate `ranked_sources` against the CAMS-REG gold table.
7. `06_visual_conditional_geval.py` — *(optional)* report quality/safety G-Eval table.

Then inspect conditions with `dashboard/dashboard.py`.

See `pipeline/README.md` for the per-stage input/output table.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENAI_API_KEY (stages 04 and 06 need it)
```

Stages 04 and 06 call the OpenAI API; set `OPENAI_API_KEY` in your environment
or `.env`. Path overrides (`ALERT2SOURCE_*`) are documented in `.env.example`.

## Notes on excluded / external data

- Raw satellite inputs (`data/raw/`) and generated z13 satellite crops
  (`outputs/visual/images/`) are gitignored — they are large and/or licensed.
- The z13 satellite-crop builder that produces `outputs/visual/visual_cases.jsonl`
  and the crop PNGs is not part of this repository; the slim `visual_cases.jsonl`
  is included so the image condition can be inspected.
- Full backbone (AQFusionNet) retraining requires the raw data directory and the
  backbone training entrypoint, which are not included here.
