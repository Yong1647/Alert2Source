#!/usr/bin/env python3
"""
paper_02_diagnostic_eligibility.py
==================================
Build the first paper-critical result: report-eligible samples and diagnostic explainer adequacy.

This script reads AQFusionNet test_detail_runX.csv files, loads the aligned LightGBM
models trained by paper_01_train_lgbm_aqsplit.py, and creates long-form sample/pollutant
records with:
  - AQFusionNet prediction reliability
  - LightGBM diagnostic-explainer adequacy
  - AQFusionNet--LightGBM agreement
  - WHO alert flag
  - final report-eligible flag

Why this file is needed
-----------------------
The final paper uses LightGBM as a diagnostic explainer rather than claiming that
Tree SHAP explains the full AQFusionNet. Therefore, we must show that the diagnostic
explainer is adequate on the same alert samples for which reports are generated.

Usage
-----
python paper_02_diagnostic_eligibility.py \
  --config config/europe_23feat.yaml \
  --result_dir results/2026..._europe_23feat \
  --lgbm_dir results/lgbm/europe_23feat_aqsplit \
  --runs 1 2 3 4 5 \
  --output_dir paper_outputs/diagnostic
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = os.path.abspath(os.getcwd())
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


WHO_IT3_THRESHOLDS = {
    "no2": 20.0,   # annual mean, μg/m3
    "o3": 70.0,    # peak season 8-hour mean proxy used in current pipeline
    "pm10": 30.0,  # annual mean, μg/m3
}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_metadata(lgbm_dir: str, pol: str, run: int) -> dict:
    path = os.path.join(lgbm_dir, pol, f"metadata_run{run}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing metadata {path}. Train aligned LGBM with paper_01_train_lgbm_aqsplit.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_from_metadata(df: pd.DataFrame, features: Sequence[str], meta: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for feat in features:
        st = meta["feature_stats"][feat]
        mean = float(st["mean"])
        std = max(float(st["std"]), 1e-8)
        vals = pd.to_numeric(df[feat], errors="coerce").fillna(mean)
        out[feat] = (vals - mean) / std
    return out


def denorm_pred(y_norm: np.ndarray, meta: dict) -> np.ndarray:
    st = meta["target_stats"]
    return np.asarray(y_norm) * float(st["std"]) + float(st["mean"])


def lgbm_predict(df: pd.DataFrame, lgbm_dir: str, pol: str, run: int) -> np.ndarray:
    meta = load_metadata(lgbm_dir, pol, run)
    feature_list_path = os.path.join(lgbm_dir, pol, "feature_list.pkl")
    model_path = os.path.join(lgbm_dir, pol, f"lgbm_models_run{run}.pkl")
    features = joblib.load(feature_list_path)
    models = joblib.load(model_path)
    features = [f for f in features if f in df.columns]
    X = normalize_from_metadata(df, features, meta)
    pred_norm = np.mean([m.predict(X) for m in models], axis=0)
    return denorm_pred(pred_norm, meta)


def safe_relerr(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.abs(pred - true) / (np.abs(true) + 1e-8)


def summarize_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"r2": np.nan, "mae": np.nan, "rmse": np.nan}
    return {
        "r2": float(r2_score(y_true[mask], y_pred[mask])) if mask.sum() > 1 else np.nan,
        "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
        "rmse": float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
    }


def process_run(cfg: dict, result_dir: str, lgbm_dir: str, run: int,
                relerr_thresh: float, agreement_thresh: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pollutants = cfg["pollutants"]
    path = os.path.join(result_dir, f"test_detail_run{run}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "idx" not in df.columns:
        df["idx"] = np.arange(len(df))

    records: List[dict] = []
    summary_rows: List[dict] = []

    for pol in pollutants:
        true_col = f"y_true_{pol}"
        pred_col = f"y_pred_{pol}"
        if true_col not in df.columns or pred_col not in df.columns:
            continue

        y_true = pd.to_numeric(df[true_col], errors="coerce").to_numpy(dtype=float)
        y_fusion = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=float)
        y_lgbm = lgbm_predict(df, lgbm_dir, pol, run)

        rel_fusion = safe_relerr(y_fusion, y_true)
        rel_lgbm = safe_relerr(y_lgbm, y_true)
        agreement_error = np.abs(y_fusion - y_lgbm) / (np.abs(y_fusion) + 1e-8)

        alert = y_true > WHO_IT3_THRESHOLDS.get(pol, np.inf)
        fusion_pass = rel_fusion < relerr_thresh
        lgbm_diag_pass = rel_lgbm < relerr_thresh
        agreement_pass = agreement_error < agreement_thresh
        final_oracle = alert & fusion_pass & lgbm_diag_pass
        final_deployable = alert & fusion_pass & agreement_pass

        for i, row in df.iterrows():
            records.append({
                "run": run,
                "sample_id": row.get("idx", i),
                "station_id": row.get("AirQualityStation", ""),
                "pollutant": pol,
                "Latitude": row.get("Latitude", np.nan),
                "Longitude": row.get("Longitude", np.nan),
                "y_true": y_true[i],
                "y_pred_fusion": y_fusion[i],
                "y_pred_lgbm": y_lgbm[i],
                "relerr_fusion": rel_fusion[i],
                "relerr_lgbm": rel_lgbm[i],
                "agreement_error": agreement_error[i],
                "alert_flag": bool(alert[i]),
                "fusion_pass": bool(fusion_pass[i]),
                "lgbm_diag_pass": bool(lgbm_diag_pass[i]),
                "agreement_pass": bool(agreement_pass[i]),
                "final_report_eligible_oracle": bool(final_oracle[i]),
                "final_report_eligible_deployable": bool(final_deployable[i]),
            })

        fusion_metrics = summarize_metrics(y_true, y_fusion)
        lgbm_metrics = summarize_metrics(y_true, y_lgbm)
        summary_rows.append({
            "run": run,
            "pollutant": pol,
            "test_samples": int(np.isfinite(y_true).sum()),
            "alert_samples": int(alert.sum()),
            "fusion_error_pass": int(fusion_pass.sum()),
            "lgbm_adequacy_pass": int(lgbm_diag_pass.sum()),
            "agreement_pass": int(agreement_pass.sum()),
            "final_report_samples_oracle": int(final_oracle.sum()),
            "final_report_samples_deployable": int(final_deployable.sum()),
            "fusion_r2": fusion_metrics["r2"],
            "fusion_mae": fusion_metrics["mae"],
            "fusion_rmse": fusion_metrics["rmse"],
            "lgbm_r2": lgbm_metrics["r2"],
            "lgbm_mae": lgbm_metrics["mae"],
            "lgbm_rmse": lgbm_metrics["rmse"],
            "mean_agreement_error": float(np.nanmean(agreement_error)),
        })

    return pd.DataFrame(records), pd.DataFrame(summary_rows)


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    count_cols = [
        "test_samples", "alert_samples", "fusion_error_pass", "lgbm_adequacy_pass",
        "agreement_pass", "final_report_samples_oracle", "final_report_samples_deployable",
    ]
    metric_cols = [
        "fusion_r2", "fusion_mae", "fusion_rmse", "lgbm_r2", "lgbm_mae", "lgbm_rmse", "mean_agreement_error",
    ]
    rows = []
    for pol, g in summary.groupby("pollutant"):
        row = {"pollutant": pol, "n_runs": int(g["run"].nunique())}
        for c in count_cols:
            row[f"{c}_mean"] = float(g[c].mean())
            row[f"{c}_std"] = float(g[c].std(ddof=0))
        for c in metric_cols:
            row[f"{c}_mean"] = float(g[c].mean())
            row[f"{c}_std"] = float(g[c].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--lgbm_dir", required=True)
    parser.add_argument("--runs", nargs="+", type=int, default=[1])
    parser.add_argument("--output_dir", default="paper_outputs/diagnostic")
    parser.add_argument("--relerr_thresh", type=float, default=0.20)
    parser.add_argument("--agreement_thresh", type=float, default=0.20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = load_config(args.config)

    all_long, all_summary = [], []
    for run in args.runs:
        print(f"[Run {run}] diagnostic eligibility")
        long_df, summary_df = process_run(cfg, args.result_dir, args.lgbm_dir, run,
                                          args.relerr_thresh, args.agreement_thresh)
        all_long.append(long_df)
        all_summary.append(summary_df)

    long = pd.concat(all_long, ignore_index=True) if all_long else pd.DataFrame()
    summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    agg = aggregate_summary(summary) if not summary.empty else pd.DataFrame()

    long_path = os.path.join(args.output_dir, "diagnostic_eligibility_long.csv")
    summary_path = os.path.join(args.output_dir, "diagnostic_eligibility_by_run.csv")
    agg_path = os.path.join(args.output_dir, "diagnostic_eligibility_summary.csv")
    long.to_csv(long_path, index=False)
    summary.to_csv(summary_path, index=False)
    agg.to_csv(agg_path, index=False)

    print(f"[OK] Long table: {long_path} ({len(long)} rows)")
    print(f"[OK] By-run summary: {summary_path}")
    print(f"[OK] Aggregated summary: {agg_path}")
    if not agg.empty:
        print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
