"""
utils_busan.py — 부울경 데이터 로딩 유틸리티
=============================================
원본 utils_3poll.py 대비 변경점:
  1. S5P/S2 경로 자동 감지
  2. 6종 오염물질 지원
  3. DatasetStatistics 자동 계산
"""

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import xarray as xr

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def step_single(x, y, model, loss_fn, optimizer):
    """Single output training step"""
    y_hat = model(x).squeeze()
    loss_val = loss_fn(y.squeeze().to("cuda:0"), y_hat)
    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()
    from src.train_utils import eval_metrics
    metrics = eval_metrics(y.detach().cpu(), y_hat.detach().cpu())
    return loss_val.detach().cpu(), metrics


def step_multi(x, y_list, model, loss_fn, optimizer):
    """Multi output training step (3poll or 6poll)"""
    outputs = model(x)  # tuple of tensors
    y_stack = torch.stack(y_list)
    y_hat_stack = torch.stack(list(outputs))
    loss_val = loss_fn(y_hat_stack, y_stack.to("cuda:0"))
    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()
    from src.train_utils import eval_metrics
    metrics = eval_metrics(y_stack.detach().cpu(), y_hat_stack.detach().cpu())
    return loss_val.detach().cpu(), metrics


def load_data(datadir, samples_file):
    """
    samples.csv + S5P NetCDF + S2 .npy 로딩
    
    디렉토리 구조:
        datadir/
        ├── sentinel-5p/stn_000_s5p_no2.nc
        ├── sentinel-2/stn_000_2019_spring.npy
        └── samples_aqnet.csv
    """
    if not isinstance(samples_file, pd.DataFrame):
        samples_df = pd.read_csv(samples_file, index_col="idx")
    else:
        samples_df = samples_file

    # NaN 행 제거 (no2 기준)
    if "no2" in samples_df.columns:
        samples_df = samples_df[samples_df["no2"].notna()]

    print(f"Available columns: {list(samples_df.columns)}")
    print(f"Total samples: {len(samples_df)}")

    samples = []
    stations = {}

    # S5P 경로 탐색
    s5p_base = os.path.join(datadir, "sentinel-5p")
    s2_base = os.path.join(datadir, "sentinel-2")

    for station in tqdm(samples_df.AirQualityStation.unique(), desc="Loading data"):
        station_obs = samples_df[samples_df.AirQualityStation == station]

        # S5P 로딩 (측정소당 1개)
        s5p_path = station_obs.s5p_path.unique()
        if len(s5p_path) == 0:
            continue
        s5p_path = s5p_path[0]

        s5p_full = os.path.join(s5p_base, s5p_path)
        if not os.path.exists(s5p_full):
            print(f"  S5P 없음: {s5p_full}")
            continue

        try:
            s5p_data = xr.open_dataset(s5p_full)
            s5p_array = s5p_data.tropospheric_NO2_column_number_density.values.squeeze()
            s5p_data.close()
        except Exception as e:
            print(f"  S5P 읽기 실패 ({station}): {e}")
            continue

        for idx in station_obs.index.values:
            row = samples_df.loc[idx]
            sample = row.to_dict()
            sample["idx"] = idx
            sample["s5p"] = s5p_array

            # S2 로딩 (처음 한 번만)
            img_path = sample.get("img_path", "")
            if station not in stations:
                s2_full = os.path.join(s2_base, img_path)
                if os.path.exists(s2_full):
                    stations[station] = np.load(s2_full)
                else:
                    # 다른 계절 파일이라도 찾기
                    found = False
                    for f in os.listdir(s2_base):
                        if f.startswith(station) and f.endswith(".npy"):
                            stations[station] = np.load(os.path.join(s2_base, f))
                            found = True
                            break
                    if not found:
                        continue

            samples.append(sample)

    print(f"Loaded: {len(samples)} samples, {len(stations)} stations")
    return samples, stations


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def none_or_true(value):
    if value == 'None':
        return None
    elif value == "True":
        return True
    return value
