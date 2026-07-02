#!/usr/bin/env python3
"""
paper_03_shap_reliability.py
============================
Compute Tree SHAP evidence and explanation-reliability metrics for the LightGBM
diagnostic explainer.

This script is aligned with the final paper direction:
  AQFusionNet = prediction backbone
  LightGBM = explanation-friendly diagnostic explainer
  Tree SHAP = source-proxy diagnostic evidence
  RAG/LLM = grounded report generator

Inputs
------
- AQFusionNet result_dir/test_detail_runX.csv
- aligned LightGBM directory from paper_01_train_lgbm_aqsplit.py
- source_registry.csv from paper_00_build_source_registry.py
- diagnostic_eligibility_long.csv from paper_02_diagnostic_eligibility.py

Outputs
-------
- shap_feature_long.csv: sample/pollutant/feature SHAP values
- shap_source_group_long.csv: sample/pollutant/source-category aggregated SHAP
- shap_reliability_by_sample.csv: bag stability and perturbation plausibility per case
- shap_reliability_summary.csv: paper-ready metrics by pollutant

Usage
-----
python paper_03_shap_reliability.py \
  --config config/europe_23feat.yaml \
  --result_dir results/2026..._europe_23feat \
  --lgbm_dir results/lgbm/europe_23feat_aqsplit \
  --eligibility_csv paper_outputs/diagnostic/diagnostic_eligibility_long.csv \
  --source_registry paper_outputs/kb/source_registry.csv \
  --runs 1 2 3 4 5 \
  --output_dir paper_outputs/shap
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_metadata(lgbm_dir: str, pol: str, run: int) -> dict:
    path = os.path.join(lgbm_dir, pol, f"metadata_run{run}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
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


def load_registry(path: str) -> pd.DataFrame:
    reg = pd.read_csv(path)
    required = {"feature_name", "source_category", "source_type", "allowed_interpretation", "report_phrase"}
    missing = required - set(reg.columns)
    if missing:
        raise ValueError(f"source_registry missing columns: {missing}")
    return reg


def feature_to_category(registry: pd.DataFrame) -> Dict[str, str]:
    return dict(zip(registry["feature_name"], registry["source_category"]))


def pairwise_jaccard(sets: List[set]) -> float:
    if len(sets) < 2:
        return 1.0
    vals = []
    for a, b in itertools.combinations(sets, 2):
        denom = len(a | b)
        vals.append(len(a & b) / denom if denom else 1.0)
    return float(np.mean(vals)) if vals else 1.0


def compute_for_run(cfg: dict, result_dir: str, lgbm_dir: str, eligibility: pd.DataFrame,
                    registry: pd.DataFrame, run: int, top_k: int, eligible_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_path = os.path.join(result_dir, f"test_detail_run{run}.csv")
    df = pd.read_csv(test_path)
    if "idx" not in df.columns:
        df["idx"] = np.arange(len(df))
    df["sample_id_str"] = df["idx"].astype(str)

    cat_map = feature_to_category(registry)
    feature_rows: List[dict] = []
    group_rows: List[dict] = []
    rel_rows: List[dict] = []

    for pol in cfg["pollutants"]:
        pol_elig = eligibility[(eligibility["run"] == run) & (eligibility["pollutant"] == pol)].copy()
        if eligible_col not in pol_elig.columns:
            raise ValueError(f"{eligible_col} not found in eligibility csv")
        target_ids = set(pol_elig.loc[pol_elig[eligible_col].astype(bool), "sample_id"].astype(str))
        if not target_ids:
            print(f"[Run {run} {pol}] no eligible samples under {eligible_col}")
            continue

        sub_df = df[df["sample_id_str"].isin(target_ids)].copy()
        if sub_df.empty:
            print(f"[Run {run} {pol}] eligible ids not found in test_detail")
            continue

        feature_path = os.path.join(lgbm_dir, pol, "feature_list.pkl")
        model_path = os.path.join(lgbm_dir, pol, f"lgbm_models_run{run}.pkl")
        features = [f for f in joblib.load(feature_path) if f in sub_df.columns]
        models = joblib.load(model_path)
        meta = load_metadata(lgbm_dir, pol, run)
        X = normalize_from_metadata(sub_df, features, meta)

        # Ensemble SHAP: mean over bag models.
        shap_by_bag = []
        pred_by_bag = []
        for m in models:
            explainer = shap.TreeExplainer(m)
            sv = explainer.shap_values(X)
            shap_by_bag.append(np.asarray(sv, dtype=float))
            pred_by_bag.append(np.asarray(m.predict(X), dtype=float))
        shap_bags = np.stack(shap_by_bag, axis=0)  # (B, N, F)
        shap_mean = shap_bags.mean(axis=0)
        pred_norm = np.stack(pred_by_bag, axis=0).mean(axis=0)
        pred_raw = denorm_pred(pred_norm, meta)

        # Train median in normalized space for perturbation.
        train_ids = set(meta.get("split_ids", {}).get("train", []))
        # We do not have the full train dataframe here. Use metadata mean -> normalized 0 as stable baseline.
        # This is consistent with z-scored inputs: replacing a feature with 0 means replacing it by training/global mean.
        baseline_norm = {feat: 0.0 for feat in features}

        for row_pos, (_, row) in enumerate(sub_df.iterrows()):
            sample_id = row["idx"]
            sv = shap_mean[row_pos]
            abs_order = np.argsort(np.abs(sv))[::-1]

            # Bag-level top-k stability.
            top_sets = []
            top_source_groups = []
            for b in range(shap_bags.shape[0]):
                b_order = np.argsort(np.abs(shap_bags[b, row_pos]))[::-1]
                top_feats_b = [features[j] for j in b_order[:top_k]]
                top_sets.append(set(top_feats_b))
                # top positive source by signed sum among the bag's top evidence
                tmp = pd.DataFrame({"feature": features, "shap": shap_bags[b, row_pos]})
                tmp["source_category"] = tmp["feature"].map(cat_map).fillna("other")
                source_signed = tmp.groupby("source_category")["shap"].sum().sort_values(ascending=False)
                top_source_groups.append(source_signed.index[0] if not source_signed.empty else "other")
            topk_jaccard = pairwise_jaccard(top_sets)
            source_top_agreement = max(top_source_groups.count(g) for g in set(top_source_groups)) / len(top_source_groups)

            # Feature-level rows.
            for rank, j in enumerate(abs_order, 1):
                feat = features[j]
                feature_rows.append({
                    "run": run,
                    "sample_id": sample_id,
                    "pollutant": pol,
                    "feature_name": feat,
                    "feature_value_raw": row.get(feat, np.nan),
                    "feature_value_norm": X.iloc[row_pos][feat],
                    "shap_value": float(sv[j]),
                    "abs_shap": float(abs(sv[j])),
                    "shap_rank_abs": rank,
                    "shap_sign": "positive" if sv[j] > 0 else "negative" if sv[j] < 0 else "zero",
                    "source_category": cat_map.get(feat, "other"),
                })

            tmp_all = pd.DataFrame({"feature_name": features, "shap_value": sv})
            tmp_all["source_category"] = tmp_all["feature_name"].map(cat_map).fillna("other")
            g = tmp_all.groupby("source_category").agg(
                signed_shap_sum=("shap_value", "sum"),
                positive_shap_sum=("shap_value", lambda x: float(np.maximum(x, 0).sum())),
                abs_shap_sum=("shap_value", lambda x: float(np.abs(x).sum())),
            ).reset_index()
            g["source_rank_abs"] = g["abs_shap_sum"].rank(method="first", ascending=False).astype(int)
            g["source_rank_positive"] = g["positive_shap_sum"].rank(method="first", ascending=False).astype(int)
            for _, gr in g.iterrows():
                group_rows.append({
                    "run": run,
                    "sample_id": sample_id,
                    "pollutant": pol,
                    "source_category": gr["source_category"],
                    "signed_shap_sum": float(gr["signed_shap_sum"]),
                    "positive_shap_sum": float(gr["positive_shap_sum"]),
                    "abs_shap_sum": float(gr["abs_shap_sum"]),
                    "source_rank_abs": int(gr["source_rank_abs"]),
                    "source_rank_positive": int(gr["source_rank_positive"]),
                })

            # Perturbation plausibility: replace top positive source group features by normalized baseline (0.0).
            positive_groups = g[g["positive_shap_sum"] > 0].sort_values("positive_shap_sum", ascending=False)
            if positive_groups.empty:
                top_source = "none_positive"
                plausible = np.nan
                pred_drop_norm = np.nan
                pred_drop_raw = np.nan
            else:
                top_source = str(positive_groups.iloc[0]["source_category"])
                group_feats = [f for f in features if cat_map.get(f, "other") == top_source]
                X_pert = X.iloc[[row_pos]].copy()
                for feat in group_feats:
                    X_pert.loc[X_pert.index[0], feat] = baseline_norm[feat]
                pred_pert_norm = np.mean([m.predict(X_pert)[0] for m in models])
                pred_pert_raw = denorm_pred(np.array([pred_pert_norm]), meta)[0]
                pred_drop_norm = float(pred_norm[row_pos] - pred_pert_norm)
                pred_drop_raw = float(pred_raw[row_pos] - pred_pert_raw)
                plausible = bool(pred_drop_norm > 0)

            rel_rows.append({
                "run": run,
                "sample_id": sample_id,
                "pollutant": pol,
                "topk_feature_jaccard_across_bags": topk_jaccard,
                "top_source_agreement_across_bags": source_top_agreement,
                "top_positive_source_group": top_source,
                "perturbation_plausible": plausible,
                "prediction_drop_norm_when_top_source_masked": pred_drop_norm,
                "prediction_drop_raw_when_top_source_masked": pred_drop_raw,
                "n_bag_models": int(shap_bags.shape[0]),
            })

        print(f"[Run {run} {pol}] SHAP rows for {len(sub_df)} eligible samples")

    return pd.DataFrame(feature_rows), pd.DataFrame(group_rows), pd.DataFrame(rel_rows)


def summarize_reliability(rel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if rel.empty:
        return pd.DataFrame()
    for pol, g in rel.groupby("pollutant"):
        valid_plaus = g["perturbation_plausible"].dropna().astype(bool)
        rows.append({
            "pollutant": pol,
            "n_samples": int(len(g)),
            "topk_feature_jaccard_mean": float(g["topk_feature_jaccard_across_bags"].mean()),
            "topk_feature_jaccard_std": float(g["topk_feature_jaccard_across_bags"].std(ddof=0)),
            "top_source_agreement_mean": float(g["top_source_agreement_across_bags"].mean()),
            "top_source_agreement_std": float(g["top_source_agreement_across_bags"].std(ddof=0)),
            "perturbation_plausibility_rate": float(valid_plaus.mean()) if len(valid_plaus) else np.nan,
            "mean_prediction_drop_raw": float(g["prediction_drop_raw_when_top_source_masked"].mean()),
            "std_prediction_drop_raw": float(g["prediction_drop_raw_when_top_source_masked"].std(ddof=0)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--lgbm_dir", required=True)
    parser.add_argument("--eligibility_csv", required=True)
    parser.add_argument("--source_registry", required=True)
    parser.add_argument("--runs", nargs="+", type=int, default=[1])
    parser.add_argument("--output_dir", default="paper_outputs/shap")
    parser.add_argument("--eligible_col", default="final_report_eligible_oracle",
                        choices=["final_report_eligible_oracle", "final_report_eligible_deployable"])
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = load_config(args.config)
    eligibility = pd.read_csv(args.eligibility_csv)
    registry = load_registry(args.source_registry)

    all_feat, all_group, all_rel = [], [], []
    for run in args.runs:
        feat, group, rel = compute_for_run(cfg, args.result_dir, args.lgbm_dir, eligibility, registry,
                                           run, args.top_k, args.eligible_col)
        all_feat.append(feat)
        all_group.append(group)
        all_rel.append(rel)

    feature_long = pd.concat(all_feat, ignore_index=True) if all_feat else pd.DataFrame()
    group_long = pd.concat(all_group, ignore_index=True) if all_group else pd.DataFrame()
    rel_df = pd.concat(all_rel, ignore_index=True) if all_rel else pd.DataFrame()
    summary = summarize_reliability(rel_df)

    feature_long.to_csv(os.path.join(args.output_dir, "shap_feature_long.csv"), index=False)
    group_long.to_csv(os.path.join(args.output_dir, "shap_source_group_long.csv"), index=False)
    rel_df.to_csv(os.path.join(args.output_dir, "shap_reliability_by_sample.csv"), index=False)
    summary.to_csv(os.path.join(args.output_dir, "shap_reliability_summary.csv"), index=False)

    print(f"[OK] Feature SHAP: {len(feature_long)} rows")
    print(f"[OK] Source-group SHAP: {len(group_long)} rows")
    print(f"[OK] Reliability rows: {len(rel_df)}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
