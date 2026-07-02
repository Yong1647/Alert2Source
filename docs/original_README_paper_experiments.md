# SHAP--RAG paper experiment suite

This folder contains a clean experiment pipeline for the revised paper direction:

> AQFusionNet = multimodal prediction backbone  
> LightGBM = explanation-friendly diagnostic explainer  
> Tree SHAP = source-proxy diagnostic evidence  
> RAG/LLM = evidence-grounded root-cause report generator

Grad-CAM and Spatial Convergence are intentionally excluded from these scripts because the current paper direction no longer uses them as the central reliability mechanism.

---

## Uploaded code roles and relationships

### Original training and model files

- `train_lgbm.py`: trains pollutant-specific LightGBM ensembles from tabular features. In the uploaded version it uses a random row split and split-specific normalization. This is useful for prototyping, but it is not aligned with the station-level split used by `train_aqnet.py`.
- `train_aqnet.py`: trains AQFusionNet. It loads the LightGBM ensembles through `model.py`, uses station-level splits, and writes `test_detail_runX.csv` and `test_results.csv`.
- `model.py`: defines AQFusionNet / MultiOutputFusion. It combines Sentinel-2, Sentinel-5P, a LightGBM tabular encoder, a tabular MLP, cross-attention/concat fusion, and a LightGBM correction term.
- `dataset.py`, `transforms.py`, `utils.py`, `train_utils.py`: data loading, preprocessing, tabular feature construction, and evaluation utilities.
- `europe_23feat.yaml`: final 23-feature Europe configuration, including ship-density features.

### Original analysis/report files

- `anomaly_detection_multi.py`: prototype end-to-end anomaly pipeline. It performs AE anomaly scoring, WHO/relative-error filtering, LGBM+SHAP, optional Grad-CAM, FAISS indexing, and LLM report generation. It still contains the old Grad-CAM/Spatial Convergence framing.
- `shap_real_lgbm.py`: computes SHAP for saved LGBM models. It should not be used as-is for the final paper because the saved LGBM preprocessing statistics are not loaded.
- `evaluation_metrics.py`: evaluates Grad-CAM and report quality. For the final paper, only report metrics are relevant; Grad-CAM metrics are no longer central.
- `app.py`: Streamlit dashboard prototype. It still contains old Grad-CAM wording and is useful for demos, not for the final paper experiments.

---

## Critical change for paper experiments

The final paper uses LightGBM as a **diagnostic explainer**. Therefore, the LightGBM model must be trained and evaluated on the same station-level splits as AQFusionNet and with the same normalized tabular representation that AQFusionNet feeds into its LightGBM encoder.

Use `paper_01_train_lgbm_aqsplit.py` instead of the old `train_lgbm.py` for paper experiments.

---

## Experiment order

### 0. Build source registry and RAG database

```bash
python paper_00_build_source_registry.py --output_dir paper_outputs/kb
```

Outputs:

- `paper_outputs/kb/source_registry.csv`
- `paper_outputs/kb/air_quality_rag_database.jsonl`

### 1. Train aligned LightGBM diagnostic explainers

```bash
python paper_01_train_lgbm_aqsplit.py \
  --config config/europe_23feat.yaml \
  --save_dir results/lgbm/europe_23feat_aqsplit
```

Important: update the config used by AQFusionNet so that:

```yaml
paths:
  lgbm_dir: results/lgbm/europe_23feat_aqsplit
```

Then retrain AQFusionNet using the updated config:

```bash
python train_aqnet.py --config config/europe_23feat_aqsplit.yaml
```

### 2. Compute diagnostic eligibility and adequacy

```bash
python paper_02_diagnostic_eligibility.py \
  --config config/europe_23feat_aqsplit.yaml \
  --result_dir results/<AQFUSION_RESULT_DIR> \
  --lgbm_dir results/lgbm/europe_23feat_aqsplit \
  --runs 1 2 3 4 5 \
  --output_dir paper_outputs/diagnostic
```

Main paper table:

- test samples
- alert samples
- AQFusionNet error-pass samples
- LightGBM adequacy-pass samples
- final report samples

### 3. Compute SHAP source-proxy evidence and SHAP reliability

```bash
python paper_03_shap_reliability.py \
  --config config/europe_23feat_aqsplit.yaml \
  --result_dir results/<AQFUSION_RESULT_DIR> \
  --lgbm_dir results/lgbm/europe_23feat_aqsplit \
  --eligibility_csv paper_outputs/diagnostic/diagnostic_eligibility_long.csv \
  --source_registry paper_outputs/kb/source_registry.csv \
  --runs 1 2 3 4 5 \
  --output_dir paper_outputs/shap
```

Main paper tables:

- SHAP top-k stability across LightGBM bags
- top source agreement across LightGBM bags
- perturbation plausibility rate

### 4. Generate report ablation outputs

Recommended ablation modes:

1. `direct_llm`
2. `shap_only`
3. `shap_registry`
4. `shap_rag`
5. `shap_rag_reliability`

Example:

```bash
python paper_04_generate_reports.py \
  --eligibility_csv paper_outputs/diagnostic/diagnostic_eligibility_long.csv \
  --shap_feature_csv paper_outputs/shap/shap_feature_long.csv \
  --shap_group_csv paper_outputs/shap/shap_source_group_long.csv \
  --shap_reliability_csv paper_outputs/shap/shap_reliability_by_sample.csv \
  --source_registry paper_outputs/kb/source_registry.csv \
  --rag_jsonl paper_outputs/kb/air_quality_rag_database.jsonl \
  --mode shap_rag_reliability \
  --max_cases 30 \
  --n_repeats 3 \
  --output_json paper_outputs/reports/reports_shap_rag_reliability.json
```

If `OPENAI_API_KEY` is not set, prompts are saved instead of real reports.

### 5. Evaluate generated reports

```bash
python paper_05_evaluate_reports.py \
  --reports_json paper_outputs/reports/reports_shap_rag_reliability.json \
  --output_json paper_outputs/reports/eval_shap_rag_reliability.json
```

Optional G-Eval:

```bash
python paper_05_evaluate_reports.py \
  --reports_json paper_outputs/reports/reports_shap_rag_reliability.json \
  --output_json paper_outputs/reports/eval_shap_rag_reliability_geval.json \
  --run_geval \
  --geval_model gpt-4o
```

---

## What changed compared with the old pipeline

1. Removed Grad-CAM and Spatial Convergence from the final paper pipeline.
2. Defined LightGBM as the diagnostic explainer rather than claiming SHAP explains the entire AQFusionNet.
3. Added aligned LightGBM training with AQNet station-level splits.
4. Added diagnostic explainer adequacy gates.
5. Added deterministic source registry and RAG knowledge base.
6. Added SHAP reliability metrics:
   - bag-level top-k stability
   - top source agreement
   - perturbation plausibility
7. Added report ablations aligned with the revised paper:
   - direct LLM
   - SHAP only
   - SHAP + registry
   - SHAP + RAG
   - SHAP + RAG + reliability gate
8. Added Unsupported Source Claim Rate for report safety evaluation.

---

## Paper wording reminder

Use:

- operational source-proxy root-cause diagnosis
- source hypothesis
- source-proxy evidence
- diagnostic explainer
- reliability-gated report generation

Avoid:

- causal proof
- true root cause
- Spatial Convergence
- independent causal evidence
- eliminates hallucination
