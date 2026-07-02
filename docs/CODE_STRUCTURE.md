# Code structure and mapping

This cleaned layout separates the current Alert2Source pipeline from older visual and multi-model experiments while keeping local packages (`src/`, `providers/`, `prompts/`) at repository root.

## Core stages

| Stage | Folder | Main files | Notes |
|---|---|---|---|
| Stage 0 | `src/`, `pipeline/01_lgbm_baseline/` | `model.py`, `dataset.py`, `train_lgbm_aqsplit.py` | Full TabSatFusion retraining still needs the missing main entrypoint and raw data. |
| Stage 1 | `pipeline/02_diagnostic/`, `pipeline/03_shap/` | `diagnostic_eligibility.py`, `shap_reliability.py` | Recomputes eligibility and SHAP evidence if result CSVs/LGBM models are provided. |
| Stage 2 | `pipeline/00_source_registry/` | `build_source_registry.py` | Generates `source_registry.csv` and `air_quality_rag_database.jsonl`. |
| Stage 3 | `pipeline/06_visual/` | legacy Google/VLM visual scripts | The latest `build_visual_cases_full.py` z13 crop script is missing. |
| Stage 4 | `pipeline/04_report_generation/`, `pipeline/05_evaluation/` | `generate_reports_v3.py`, `eval_ranked_sources.py` | Current report generator/evaluator path. |
| Dashboard | `dashboard/` | `dashboard.py` | Defaults were patched to repository-relative paths/env vars. |

## Why some scripts remain in `legacy_multimodel/`

`paper_12_generate_multimodel_reports.py` and `paper_13_evaluate_unified_geval.py` depend on optional provider configs and a missing helper module, so they are kept separate from the main reproducibility path.

## Path cleanup already applied

- `config/*.yaml`: absolute server paths replaced with `data/` and `outputs/` placeholders.
- `dashboard/dashboard.py`: absolute defaults replaced with `ALERT2SOURCE_*` environment variables and relative fallback paths.
- `pipeline/05_evaluation/eval_full_465_by_condition.py`: absolute defaults replaced with `ALERT2SOURCE_*` environment variables and relative fallback paths.
- `pipeline/04_report_generation/generate_reports_v3.py`: docstring example project root changed to `.`.

## Original-to-new file map

See `docs/file_map.csv`.
