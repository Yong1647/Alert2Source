"""
src/train_utils.py — 학습 유틸리티 (config 기반)
================================================
변경점 (vs train_utils_busan.py):
  - POLLUTANTS_3/6 하드코딩 제거 → config에서 전달
  - build_tabular: mode 숫자 → config features 리스트 기반
  - import 경로: src.transforms
"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import torch

os.environ["OMP_NUM_THREADS"] = "6"


def eval_metrics(y, y_hat):
    """R2, MAE, MSE 계산"""
    if isinstance(y, torch.Tensor):
        y = y.numpy()
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.numpy()
    y = np.array(y).flatten()
    y_hat = np.array(y_hat).flatten()
    return [r2_score(y, y_hat), mean_absolute_error(y, y_hat), mean_squared_error(y, y_hat)]


def split_samples(samples, stations, test_size=0.25, val_size=0.25,
                  seed=None, return_idx=False):
    """측정소 단위 split (같은 측정소 샘플은 같은 set에)"""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    stations_list = list(set(stations))
    stations_train, stations_test = train_test_split(
        stations_list, test_size=test_size, random_state=seed)
    real_val_size = val_size / (1 - test_size)
    stations_train, stations_val = train_test_split(
        stations_train, test_size=real_val_size, random_state=seed)
    stations_train = set(stations_train)
    stations_val = set(stations_val)
    stations_test = set(stations_test)

    samples_train = [s for s in samples if s["AirQualityStation"] in stations_train]
    samples_val = [s for s in samples if s["AirQualityStation"] in stations_val]
    samples_test = [s for s in samples if s["AirQualityStation"] in stations_test]

    if return_idx:
        idx_train = [i for i, s in enumerate(samples) if s["AirQualityStation"] in stations_train]
        idx_val = [i for i, s in enumerate(samples) if s["AirQualityStation"] in stations_val]
        idx_test = [i for i, s in enumerate(samples) if s["AirQualityStation"] in stations_test]
        return samples_train, samples_val, samples_test, idx_train, idx_val, idx_test

    return samples_train, samples_val, samples_test, stations_train, stations_val, stations_test


# ================================================================
# build_tabular: config features 기반 (mode 숫자 방식도 하위호환)
# ================================================================

# 하위호환용 feature 목록
_MODE_FEATURES = {
    "8": [
        "Altitude", "PopulationDensity",
        "rural", "suburban", "urban",
        "traffic", "industrial", "background",
    ],
    "20": [
        "Altitude", "Latitude", "Longitude", "PopulationDensity",
        "Precip_3yr", "RH_3yr", "Temp_3yr", "Wind_3yr",
        "num_buildings_1km", "num_buildings_500m",
        "num_factory_1km", "num_factory_500m",
        "num_industrial_landuse_1km", "num_industrial_landuse_500m",
        "num_roads_1km", "num_roads_500m",
        "rural", "traffic", "urban",
        "Stability_3yr",
    ],
}


def build_tabular(sample, mode="20", feature_list=None):
    """
    tabular tensor 생성

    Args:
        sample: batch dict
        mode: "8", "20" (하위호환) — feature_list가 있으면 무시됨
        feature_list: config에서 가져온 피처 리스트 (우선 사용)
    """
    if feature_list is not None:
        features = [sample[f] for f in feature_list]
    elif mode in _MODE_FEATURES:
        features = [sample[f] for f in _MODE_FEATURES[mode]]
    else:
        raise ValueError(f"Unknown tabular mode: {mode}. Use feature_list or mode in {list(_MODE_FEATURES.keys())}")

    return torch.stack(features, dim=1).float()


# ================================================================
# test_multi: 오염물질 리스트를 외부에서 전달받음
# ================================================================

def test_multi(model, dataloader, device, datastats, pollutants, tabular_mode="20",
               feature_list=None):
    """
    Multi-pollutant evaluation

    Args:
        pollutants: ["o3", "pm10", "so2"] 등 — config에서 전달
        feature_list: config tabular_features (optional)
    """
    from src.transforms import Normalize
    model.eval()

    measurements = {p: [] for p in pollutants}
    predictions = {p: [] for p in pollutants}

    undo_fn = {
        "no2": Normalize.undo_no2_standardization,
        "o3": Normalize.undo_o3_standardization,
        "pm10": Normalize.undo_pm10_standardization,
        "so2": Normalize.undo_so2_standardization,
        "co": Normalize.undo_co_standardization,
        "pm25": Normalize.undo_pm25_standardization,
    }

    with torch.no_grad():
        for idx, sample in enumerate(dataloader):
            img = sample["img"].float().to(device)
            s5p = sample["s5p"].float().unsqueeze(dim=1).to(device)
            tabular = build_tabular(sample, mode=tabular_mode, feature_list=feature_list).to(device)
            model_input = {"img": img, "s5p": s5p, "tabular": tabular}

            outputs = model(model_input)

            for i, pol in enumerate(pollutants):
                y = sample[pol].float().to(device).squeeze()
                y_hat = outputs[i].squeeze()
                measurements[pol].append(y.cpu().numpy().item())
                predictions[pol].append(y_hat.cpu().numpy().item())

    # undo normalization
    for pol in pollutants:
        fn = undo_fn.get(pol)
        if fn:
            measurements[pol] = fn(datastats, np.array(measurements[pol]))
            predictions[pol] = fn(datastats, np.array(predictions[pol]))

    return measurements, predictions


def test_plotter_multi(output_dir, test_y, test_y_hat, train_y, train_y_hat, pollutants):
    """Multi-pollutant scatter plots"""
    n = len(pollutants)
    fig, axs = plt.subplots(n, 2, figsize=(12, 4 * n))
    fig.subplots_adjust(hspace=0.5, wspace=0.25)

    if n == 1:
        axs = axs.reshape(1, 2)

    for i, pol in enumerate(pollutants):
        for j, (y, y_hat, label) in enumerate([
            (test_y[pol], test_y_hat[pol], "test"),
            (train_y[pol], train_y_hat[pol], "train")
        ]):
            axs[i, j].scatter(y, y_hat, s=2)
            axs[i, j].set_title(f"{pol.upper()} {label}")
            axs[i, j].set_xlabel("Measurements")
            axs[i, j].set_ylabel("Predictions")
            axs[i, j].set_aspect('equal')
            axs[i, j].axline((0, 0), slope=1, c="red")

    plt.savefig(os.path.join(output_dir, "predictions.png"), dpi=150, bbox_inches='tight')
    plt.close()
