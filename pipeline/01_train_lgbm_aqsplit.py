#!/usr/bin/env python3
"""
paper_01_train_lgbm_aqsplit.py
===============================
Train pollutant-specific LightGBM diagnostic explainers with the SAME station-level
run split used by train_aqnet.py and with the SAME normalized tabular representation
that AQFusionNet sends into TabularDTEncoder.

Why this file replaces / complements train_lgbm.py
--------------------------------------------------
Current train_lgbm.py is useful for prototype training, but it uses a random row split
and normalizes features with its own split-specific statistics. train_aqnet.py uses a
station-level split and passes DatasetStatistics-normalized tabular tensors into the LGBM
encoders. For paper experiments, the diagnostic explainer should be aligned with the
prediction backbone. This script fixes that by:

1. Using split_samples(..., seed=seed_base+run), same as train_aqnet.py.
2. Training LGBM on tabular features normalized by DatasetStatistics.from_samples(samples),
   matching the tensors passed into AQFusionNet.
3. Training targets in the same standardized units used by AQFusionNet loss.
4. Saving preprocessing metadata, station/sample split ids, and diagnostic test predictions.

Usage
-----
python paper_01_train_lgbm_aqsplit.py --config config/europe_23feat.yaml \
    --save_dir results/lgbm/europe_23feat_aqsplit

Then update config paths.lgbm_dir to that save_dir before retraining AQFusionNet.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from itertools import product
from typing import Dict, List, Sequence, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Allow running either from project root or from an analysis folder.
PROJECT_ROOT = os.path.abspath(os.getcwd())
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.utils import load_data, set_seed
    from src.transforms import DatasetStatistics
    from src.train_utils import split_samples
except Exception:  # pragma: no cover - fallback for local testing with uploaded files
    from utils import load_data, set_seed
    from transforms import DatasetStatistics
    from train_utils import split_samples


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def stat_key(feature: str) -> Tuple[str, str]:
    """Return DatasetStatistics mean/std attribute names for a feature."""
    special = {
        "Altitude": ("alt_mean", "alt_std"),
        "PopulationDensity": ("popdense_mean", "popdense_std"),
        "Latitude": ("lat_mean", "lat_std"),
        "Longitude": ("lon_mean", "lon_std"),
        "Temp_3yr": ("temp3yr_mean", "temp3yr_std"),
        "Wind_3yr": ("wind3yr_mean", "wind3yr_std"),
        "Precip_3yr": ("precip3yr_mean", "precip3yr_std"),
        "RH_3yr": ("rh3yr_mean", "rh3yr_std"),
    }
    if feature in special:
        return special[feature]
    return f"{feature}_mean", f"{feature}_std"


def target_stat_key(pollutant: str) -> Tuple[str, str]:
    return f"{pollutant}_mean", f"{pollutant}_std"


def normalize_df(df: pd.DataFrame, features: Sequence[str], stats: DatasetStatistics) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for feat in features:
        mean_attr, std_attr = stat_key(feat)
        mean = float(getattr(stats, mean_attr, 0.0))
        std = float(max(getattr(stats, std_attr, 1.0), 1e-8))
        vals = pd.to_numeric(df[feat], errors="coerce").fillna(mean)
        out[feat] = (vals - mean) / std
    return out


def normalize_target(y: pd.Series, pollutant: str, stats: DatasetStatistics) -> pd.Series:
    mean_attr, std_attr = target_stat_key(pollutant)
    mean = float(getattr(stats, mean_attr))
    std = float(max(getattr(stats, std_attr), 1e-8))
    return (y - mean) / std


def denormalize_target(y_norm: np.ndarray, pollutant: str, stats: DatasetStatistics) -> np.ndarray:
    mean_attr, std_attr = target_stat_key(pollutant)
    return np.asarray(y_norm) * float(getattr(stats, std_attr)) + float(getattr(stats, mean_attr))


def samples_to_df(samples: Sequence[dict]) -> pd.DataFrame:
    rows = []
    for s in samples:
        row = {k: v for k, v in s.items() if k not in {"img", "s5p"}}
        rows.append(row)
    return pd.DataFrame(rows)


def dump_metadata(path: str, cfg: dict, stats: DatasetStatistics, features: Sequence[str], pollutant: str,
                  run: int, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    feature_stats = {}
    for feat in features:
        mean_attr, std_attr = stat_key(feat)
        feature_stats[feat] = {
            "mean": float(getattr(stats, mean_attr, 0.0)),
            "std": float(max(getattr(stats, std_attr, 1.0), 1e-8)),
            "mean_attr": mean_attr,
            "std_attr": std_attr,
        }
    target_mean_attr, target_std_attr = target_stat_key(pollutant)
    meta = {
        "script": "paper_01_train_lgbm_aqsplit.py",
        "note": "LGBM trained on station-level AQNet split and DatasetStatistics-normalized tabular values.",
        "region": cfg.get("name"),
        "pollutant": pollutant,
        "run": run,
        "features": list(features),
        "feature_stats": feature_stats,
        "target_stats": {
            "mean": float(getattr(stats, target_mean_attr)),
            "std": float(max(getattr(stats, target_std_attr), 1e-8)),
            "mean_attr": target_mean_attr,
            "std_attr": target_std_attr,
        },
        "split_ids": {
            "train": train_df.get("idx", pd.Series(train_df.index)).astype(str).tolist(),
            "val": val_df.get("idx", pd.Series(val_df.index)).astype(str).tolist(),
            "test": test_df.get("idx", pd.Series(test_df.index)).astype(str).tolist(),
        },
        "split_stations": {
            "train": sorted(train_df.get("AirQualityStation", pd.Series(dtype=str)).astype(str).unique().tolist()),
            "val": sorted(val_df.get("AirQualityStation", pd.Series(dtype=str)).astype(str).unique().tolist()),
            "test": sorted(test_df.get("AirQualityStation", pd.Series(dtype=str)).astype(str).unique().tolist()),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--save_dir", default=None, help="Default: cfg.paths.lgbm_dir + '_aqsplit'")
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--n_bags", type=int, default=None)
    parser.add_argument("--max_grid", type=int, default=None, help="Optional cap for fast debugging")
    args = parser.parse_args()

    cfg = load_config(args.config)
    save_dir = args.save_dir or f"{cfg['paths']['lgbm_dir']}_aqsplit"
    os.makedirs(save_dir, exist_ok=True)

    pollutants = cfg["pollutants"]
    features = [f for f in cfg["tabular_features"]]
    runs = args.runs or cfg["training"]["runs"]
    n_bags = args.n_bags or cfg["lgbm"]["n_bags"]
    seed_base = cfg["training"]["seed_base"]

    samples, stations = load_data(cfg["paths"]["datadir"], cfg["paths"]["samples_file"])
    stats = DatasetStatistics.from_samples(samples)

    param_grid = cfg["lgbm"]["param_grid"]
    param_names = sorted(param_grid.keys())
    param_combos = list(product(*[param_grid[k] for k in param_names]))
    if args.max_grid:
        param_combos = param_combos[: args.max_grid]

    print("=" * 70)
    print("Aligned LightGBM diagnostic explainer training")
    print(f"Config: {args.config}")
    print(f"Save dir: {save_dir}")
    print(f"Pollutants: {pollutants}")
    print(f"Features ({len(features)}): {features}")
    print(f"Runs: {runs}, Bags: {n_bags}, Grid: {len(param_combos)}")
    print("=" * 70)

    all_summary = []

    for run in range(1, runs + 1):
        seed = seed_base + run
        set_seed(seed)
        val_ratio = cfg["training"]["split_ratio"]["val"]
        test_ratio = cfg["training"]["split_ratio"]["test"]
        samples_train, samples_val, samples_test, *_ = split_samples(
            samples, list(stations.keys()), test_size=test_ratio, val_size=val_ratio, seed=seed
        )
        df_train, df_val, df_test = map(samples_to_df, [samples_train, samples_val, samples_test])

        # Keep only features present in all dataframes.
        run_features = [f for f in features if f in df_train.columns and f in df_val.columns and f in df_test.columns]
        X_train = normalize_df(df_train, run_features, stats)
        X_val = normalize_df(df_val, run_features, stats)
        X_test = normalize_df(df_test, run_features, stats)

        print(f"\n--- Run {run}/{runs} seed={seed} | train={len(df_train)}, val={len(df_val)}, test={len(df_test)} ---")

        for pol in pollutants:
            if pol not in df_train.columns:
                print(f"[SKIP] {pol}: target missing")
                continue

            y_train_raw = pd.to_numeric(df_train[pol], errors="coerce")
            y_val_raw = pd.to_numeric(df_val[pol], errors="coerce")
            y_test_raw = pd.to_numeric(df_test[pol], errors="coerce")
            train_mask = y_train_raw.notna()
            val_mask = y_val_raw.notna()
            test_mask = y_test_raw.notna()

            y_train = normalize_target(y_train_raw[train_mask], pol, stats)
            y_val = normalize_target(y_val_raw[val_mask], pol, stats)
            y_test = y_test_raw[test_mask]
            Xtr, Xva, Xte = X_train.loc[train_mask], X_val.loc[val_mask], X_test.loc[test_mask]

            best_r2, best_rmse = -np.inf, np.inf
            best_models, best_params = None, None

            print(f"  [{pol.upper()}] training: {len(Xtr)} train / {len(Xva)} val / {len(Xte)} test")
            for combo_idx, combo in enumerate(param_combos, 1):
                params = dict(zip(param_names, combo))
                bag_preds = []
                models = []
                for bag_seed in range(n_bags):
                    model = lgb.LGBMRegressor(
                        **params,
                        n_estimators=cfg["lgbm"]["n_estimators"],
                        objective="regression",
                        random_state=seed + bag_seed,
                        n_jobs=6,
                        verbose=-1,
                    )
                    model.fit(
                        Xtr,
                        y_train,
                        eval_set=[(Xva, y_val)],
                        callbacks=[lgb.early_stopping(cfg["lgbm"]["early_stopping"], verbose=False)],
                    )
                    bag_preds.append(model.predict(Xva))
                    models.append(model)

                val_pred_raw = denormalize_target(np.mean(bag_preds, axis=0), pol, stats)
                val_true_raw = y_val_raw[val_mask].values
                rmse = float(np.sqrt(mean_squared_error(val_true_raw, val_pred_raw)))
                r2 = float(r2_score(val_true_raw, val_pred_raw))
                if r2 > best_r2:
                    best_r2, best_rmse = r2, rmse
                    best_models, best_params = models, params

            assert best_models is not None
            test_pred_norm = np.mean([m.predict(Xte) for m in best_models], axis=0)
            test_pred_raw = denormalize_target(test_pred_norm, pol, stats)
            test_r2 = float(r2_score(y_test.values, test_pred_raw))
            test_rmse = float(np.sqrt(mean_squared_error(y_test.values, test_pred_raw)))
            test_mae = float(mean_absolute_error(y_test.values, test_pred_raw))
            print(f"    best val R2={best_r2:.4f} | test R2={test_r2:.4f}, RMSE={test_rmse:.3f}, MAE={test_mae:.3f}")

            pol_dir = os.path.join(save_dir, pol)
            os.makedirs(pol_dir, exist_ok=True)
            joblib.dump(best_models, os.path.join(pol_dir, f"lgbm_models_run{run}.pkl"))
            joblib.dump(run_features, os.path.join(pol_dir, "feature_list.pkl"))
            with open(os.path.join(pol_dir, f"best_params_run{run}.json"), "w", encoding="utf-8") as f:
                json.dump(best_params, f, indent=2)
            dump_metadata(
                os.path.join(pol_dir, f"metadata_run{run}.json"),
                cfg, stats, run_features, pol, run, df_train, df_val, df_test,
            )

            pred_df = df_test.loc[test_mask, [c for c in ["idx", "AirQualityStation", "Latitude", "Longitude"] if c in df_test.columns]].copy()
            pred_df["pollutant"] = pol
            pred_df["y_true"] = y_test.values
            pred_df["y_pred_lgbm"] = test_pred_raw
            pred_df["run"] = run
            pred_df.to_csv(os.path.join(pol_dir, f"test_predictions_run{run}.csv"), index=False)

            all_summary.append({
                "run": run,
                "pollutant": pol,
                "val_r2": best_r2,
                "val_rmse": best_rmse,
                "test_r2": test_r2,
                "test_rmse": test_rmse,
                "test_mae": test_mae,
                "n_train": int(len(Xtr)),
                "n_val": int(len(Xva)),
                "n_test": int(len(Xte)),
            })

    pd.DataFrame(all_summary).to_csv(os.path.join(save_dir, "lgbm_aqsplit_summary.csv"), index=False)
    shutil.copy2(args.config, os.path.join(save_dir, "config_used.yaml"))
    print(f"\n[OK] Saved aligned LGBM models and summary to {save_dir}")


if __name__ == "__main__":
    main()
