# Pipeline

Alert2Source staged pipeline. Each script is a single stage; the numeric prefix
(`00`–`06`) is the intended run order. Scripts read inputs and write outputs via
CLI arguments and repository-relative paths, so run them from the repository root.

Local packages (`src/`, `prompts/`, `providers/`) live at the repository root and
are imported by these scripts (e.g. `01` imports `src.utils`).

## Stages

| # | Script | Task | Key inputs | Key outputs |
|---|--------|------|-----------|-------------|
| 00 | `00_build_source_registry.py` | Build the feature→source mapping registry and the RAG knowledge cards (self-contained rules). | none | `outputs/kb/source_registry.csv`, `outputs/kb/air_quality_rag_database.jsonl` |
| 01 | `01_train_lgbm_aqsplit.py` | Train the AQ-split-aligned LightGBM baseline/surrogate (per pollutant, bagged, grid search). | `config/europe_23feat.yaml`, `data/processed/final_master_data_FINAL.csv`, `data/raw/` satellite files | `outputs/models/lgbm/.../` (joblib models, `metadata_run{N}.json`, `test_predictions_run{N}.csv`) |
| 02 | `02_diagnostic_eligibility.py` | Decide which cases are eligible for report generation by comparing backbone predictions with the aligned LGBM (relative error / agreement). | config, backbone `test_detail_run{X}.csv` (`--result_dir`), `--lgbm_dir` | `outputs/diagnostic/diagnostic_eligibility_long.csv` (+ by-run / summary) |
| 03 | `03_shap_reliability.py` | Compute Tree-SHAP feature-level and source-group evidence plus bagging stability metrics. | config, `--result_dir`, `--lgbm_dir`, `source_registry.csv`, `diagnostic_eligibility_long.csv` | `outputs/shap/shap_feature_long.csv`, `shap_source_group_long.csv`, `shap_reliability_by_sample.csv`, `shap_reliability_summary.csv` |
| 04 | `04_generate_reports.py` | Current report generator. The LLM receives SHAP evidence + RAG cards + (image condition) the station satellite crop directly, and emits a structured report with `ranked_sources`. Handles the three conditions (SHAP / +RAG / +RAG+Image). | SHAP long CSVs, `source_registry.csv`, `air_quality_rag_database.jsonl`, `visual_cases.jsonl` (image condition), OpenAI key | `outputs/reports/full_shap_*.json` |
| 05 | `05_eval_ranked_sources.py` | Current evaluator. Reads the LLM `ranked_sources` verbatim and scores against CAMS-REG gold: AC@1/3, MRR, recall@3, precision, divergence. | `outputs/reports/full_shap_*.json`, `data/cams_reg_source_gold.csv` | `ranking_metrics_by_condition.csv` |
| 06 | `06_visual_conditional_geval.py` | Report-quality/safety G-Eval across methods (Direct / SHAP / +RAG / +RAG+Visual) on correctness, completeness, relevance, safety (+ fidelity). Produces the paper's LaTeX table. | `--reports method=path...` report JSONs, OpenAI key | per-method comparison JSON with `latex_table` |

## Main runnable path (headline v3 flow)

```
00  → build registry / RAG cards
01  → (optional) retrain LightGBM if starting from raw data
02  → diagnostic eligibility (needs backbone test-detail + LGBM models)
03  → SHAP evidence
04  → generate reports (ranked_sources; +image if visual_cases.jsonl provided)
05  → evaluate ranked_sources vs CAMS gold
06  → (optional) G-Eval report quality/safety table
```

## Notes / missing external pieces

- The z13 satellite-crop builder (`build_visual_cases_full.py`) is **not** included
  here; it produces `outputs/visual/visual_cases.jsonl` and the crop PNGs consumed by
  stage 04's image condition.
- Stage 02/03 require the backbone `test_detail_run{X}.csv` files; the backbone
  training entrypoint is not part of this pipeline folder.
- Stages 04 and 06 require an OpenAI API key (set `OPENAI_API_KEY`).
