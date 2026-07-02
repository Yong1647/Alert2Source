# Alert2Source
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://alert2source-gkrpy2ympcvodd5tznxkfe.streamlit.app/)

Alert2Source is a staged pipeline for explaining air-quality alerts with SHAP evidence, source-knowledge RAG, optional visual context, and ranked source attribution.

This repository snapshot is a cleaned GitHub-oriented structure created from the uploaded working files. Large datasets, model checkpoints, generated crops, and full report outputs are intentionally excluded.

## Repository layout

```text
Alert2Source/
├── config/                    # sanitized config templates
├── src/                       # TabSatFusion / AQFusionNet model utilities
├── providers/                 # optional multi-model provider adapters
├── prompts/                   # prompt builders for unified multi-model experiments
├── pipeline/
│   ├── 00_source_registry/    # source_registry.csv + RAG JSONL builder
│   ├── 01_lgbm_baseline/      # LightGBM aligned baseline/surrogate training
│   ├── 02_diagnostic/         # report eligibility diagnostics
│   ├── 03_shap/               # SHAP feature/group evidence extraction
│   ├── 04_report_generation/  # latest report generator with ranked_sources
│   ├── 05_evaluation/         # ranked_sources and legacy report evaluation
│   └── 06_visual/             # older visual-context / VLM scripts
├── dashboard/                 # Streamlit dashboard
├── legacy_multimodel/         # optional/incomplete multi-model extension
├── data/                      # external data placeholders only
├── outputs/                   # generated outputs placeholders only
└── docs/                      # file map, missing-file checklist, original READMEs
```

## Main runnable path

1. Build source registry / RAG cards.
2. Compute diagnostic eligibility from backbone predictions and aligned LGBM models.
3. Compute SHAP feature and source-group evidence.
4. Generate reports with `pipeline/04_report_generation/generate_reports_v3.py`.
5. Evaluate `ranked_sources` with `pipeline/05_evaluation/eval_ranked_sources.py`.
6. Inspect conditions with `dashboard/dashboard.py`.

See `docs/CODE_STRUCTURE.md` and `docs/MISSING_FILES.md` before running the full paper pipeline.
