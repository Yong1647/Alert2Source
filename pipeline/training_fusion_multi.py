"""
training_fusion_multi.py — TabSatFusion (multi-output) backbone training
========================================================================
Trains the multimodal TabSatFusion backbone (LGBM leaf-embedding +
cross-attention hybrid) that predicts NO2, O3, and PM10 concentrations.
This is the "Stage 0" backbone whose per-run test detail is consumed by
the diagnostic / SHAP stages (02, 03).

Prerequisite: the LGBM leaf-embedding models must exist first (Stage 01,
``01_train_lgbm_aqsplit.py``), since ``get_model`` loads them from
``--lgbm_dir`` to build the tabular leaf-index embedding.

Run from the repository root so that ``src`` is importable:

    # Stage 01 first (produces the LGBM models)
    python pipeline/01_train_lgbm_aqsplit.py --config config/europe_23feat.yaml

    # Then the fusion backbone
    python pipeline/training_fusion_multi.py \
        --samples_file data/processed/final_master_data_FINAL.csv \
        --datadir /path/to/sentinel_root \
        --lgbm_dir outputs/models/lgbm/europe_23feat_aqsplit \
        --result_dir outputs/backbone

Outputs a timestamped folder under ``--result_dir`` containing
``test_detail_run{run}.csv`` (consumed by Stage 02 via its ``--result_dir``),
``test_results.csv``, and ``model.pt``.

Note on data: ``--datadir`` must point at the raw Sentinel root holding
``sentinel-2/`` (.npy) and ``sentinel-5p/`` (.nc) per-station files. These are
NOT shipped in the repo (see ``.gitignore`` / ``data/README.md``); point this at
your local Sentinel download or the released data archive.
"""

import warnings
import os
import sys
warnings.filterwarnings("ignore", message="X does not have valid feature names")

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"

import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader

# Local packages live at the repository root; run from there.
PROJECT_ROOT = os.path.abspath(os.getcwd())
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.dataset import NO2PredictionDataset
    from src.transforms import ChangeBandOrder, ToTensor, DatasetStatistics, Normalize, Randomize
    from src.utils import load_data, set_seed
    from src.train_utils import (eval_metrics, split_samples, build_tabular,
                                 test_multi, test_plotter_multi)
    from src.model import get_model
except Exception:  # fallback when the src/*.py modules sit next to this file
    from dataset import NO2PredictionDataset
    from transforms import ChangeBandOrder, ToTensor, DatasetStatistics, Normalize, Randomize
    from utils import load_data, set_seed
    from train_utils import (eval_metrics, split_samples, build_tabular,
                             test_multi, test_plotter_multi)
    from model import get_model

# Pollutant sets (previously in train_utils; kept local so src stays config-driven).
POLLUTANTS_3 = ["no2", "o3", "pm10"]
POLLUTANTS_6 = ["no2", "o3", "pm10", "so2", "co", "pm25"]

# ================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--samples_file', default="data/processed/final_master_data_FINAL.csv", type=str)
parser.add_argument('--datadir', default="data/raw/", type=str,
                    help="Raw Sentinel root holding sentinel-2/ (.npy) and sentinel-5p/ (.nc)")
parser.add_argument('--lgbm_dir', default="outputs/models/lgbm/europe_23feat_aqsplit",
                    help="LGBM leaf-embedding models (per-pollutant subfolders), from Stage 01")
parser.add_argument('--result_dir', default="outputs/backbone")
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--runs', default=5, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--weight_decay', default=0.005, type=float)
parser.add_argument('--pollutants', default=3, type=int, choices=[3, 6])
parser.add_argument('--tabular', default=20, type=int)
args = parser.parse_args()

POLLUTANTS = POLLUTANTS_3 if args.pollutants == 3 else POLLUTANTS_6
prediction_count = len(POLLUTANTS)
tabular_mode = str(args.tabular)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

experiment_folder = "_".join([
    datetime.today().strftime('%Y%m%d_%H%M'),
    str(args.epochs), "epochs",
    f"{prediction_count}poll", f"tab{args.tabular}", "fusion"
])
output_dir = os.path.join(args.result_dir, experiment_folder)
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print(f"TabSatFusion Multi-Output Training")
print(f"  Pollutants: {POLLUTANTS}")
print(f"  Tabular: {args.tabular}")
print(f"  LGBM dir: {args.lgbm_dir}")
print(f"  Device: {device}")
print(f"  Output: {output_dir}")
print("=" * 60)

# ================================================================
# 데이터 로딩
# ================================================================
print("Loading data...")
samples, stations = load_data(args.datadir, args.samples_file)
datastats = DatasetStatistics.from_samples(samples)
tf = transforms.Compose([ChangeBandOrder(), Normalize(datastats), Randomize(), ToTensor()])

# ================================================================
# 학습
# ================================================================
loss_fn = nn.MSELoss()
performances_test = []
performances_train = []


def save_detail_csv(samples_list, y_true_dict, y_pred_dict, path):
    df = pd.DataFrame(samples_list).reset_index(drop=True)
    for pol in POLLUTANTS:
        df[f'y_true_{pol}'] = y_true_dict[pol]
        df[f'y_pred_{pol}'] = y_pred_dict[pol]
    df.to_csv(path, index=False)


for run in tqdm(range(1, args.runs + 1), unit="run"):
    seed = run + 2023
    set_seed(seed)
    print(f"\n--- Run {run}/{args.runs} (seed={seed}) ---")

    splits = split_samples(samples, list(stations.keys()), 0.25, 0.25, seed=seed)
    samples_train, samples_val, samples_test = splits[0], splits[1], splits[2]

    ds_train = NO2PredictionDataset(args.datadir, samples_train, transforms=tf, station_imgs=stations)
    ds_val = NO2PredictionDataset(args.datadir, samples_val, transforms=tf, station_imgs=stations)
    ds_test = NO2PredictionDataset(args.datadir, samples_test, transforms=tf, station_imgs=stations)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=0)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0)
    dl_train_eval = DataLoader(ds_train, batch_size=1, shuffle=False, num_workers=0)

    # Model
    model = get_model(
        device, "mobilenet_v3_small",
        tabular_switch=True, S5p_switch=True,
        checkpoint=True,
        prediction_count=prediction_count,
        tabular_input_count=args.tabular,
        lgbm_dir=args.lgbm_dir,
        pollutant_names=POLLUTANTS,
        run=run
    )
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.5, min_lr=1e-7)

    print(f"  Train: {len(ds_train)}, Val: {len(ds_val)}, Test: {len(ds_test)}")

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        loss_epoch = []

        for batch in dl_train:
            img = batch["img"].float().to(device)
            s5p = batch["s5p"].float().unsqueeze(dim=1).to(device)
            tabular = build_tabular(batch, mode=tabular_mode).to(device)
            model_input = {"img": img, "s5p": s5p, "tabular": tabular}

            y_list = [batch[pol].float() for pol in POLLUTANTS]
            outputs = model(model_input)

            y_stack = torch.stack(y_list)
            y_hat_stack = torch.stack(list(outputs))
            loss = loss_fn(y_hat_stack, y_stack.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.item())

        avg_loss = np.mean(loss_epoch)
        scheduler.step(avg_loss)
        torch.cuda.empty_cache()

        # Validation
        val_y, val_y_hat = test_multi(model, dl_val, device, datastats, POLLUTANTS, tabular_mode)
        eval_val = {p: eval_metrics(val_y[p], val_y_hat[p]) for p in POLLUTANTS}

        val_str = " | ".join([f"{p}: R2={eval_val[p][0]:.3f}" for p in POLLUTANTS])
        print(f"  Epoch {epoch}: loss={avg_loss:.4f} | {val_str}")

    # Test
    test_y, test_y_hat = test_multi(model, dl_test, device, datastats, POLLUTANTS, tabular_mode)
    train_y, train_y_hat = test_multi(model, dl_train_eval, device, datastats, POLLUTANTS, tabular_mode)

    eval_test = {p: eval_metrics(test_y[p], test_y_hat[p]) for p in POLLUTANTS}
    eval_train = {p: eval_metrics(train_y[p], train_y_hat[p]) for p in POLLUTANTS}

    print(f"  Test: {' | '.join([f'{p}: R2={eval_test[p][0]:.3f}' for p in POLLUTANTS])}")

    # Save details
    save_detail_csv(samples_test, test_y, test_y_hat,
                    os.path.join(output_dir, f"test_detail_run{run}.csv"))

    test_plotter_multi(output_dir, test_y, test_y_hat, train_y, train_y_hat, POLLUTANTS)

    test_row = []
    train_row = []
    for pol in POLLUTANTS:
        test_row.extend(eval_test[pol])
        train_row.extend(eval_train[pol])
    performances_test.append(test_row)
    performances_train.append(train_row)

# Save model
torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

# Results
cols = []
for pol in POLLUTANTS:
    cols.extend([f"r2_{pol}", f"mae_{pol}", f"mse_{pol}"])
df_test = pd.DataFrame(performances_test, columns=cols)
df_test.to_csv(os.path.join(output_dir, "test_results.csv"), index=False)

print("\n" + "=" * 60)
print("Results Summary")
for pol in POLLUTANTS:
    r2 = df_test[f"r2_{pol}"].mean()
    mae = df_test[f"mae_{pol}"].mean()
    print(f"  {pol.upper():5s}: R2={r2:.4f}, MAE={mae:.4f}")
print(f"Saved to: {output_dir}")
print("done.")
