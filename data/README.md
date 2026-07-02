# Data directory

Large and license-restricted inputs are intentionally **not** committed to this
GitHub code package. This folder ships only slim demo/derived tables; the raw
satellite inputs must be obtained separately (see below).

## What is included here

- `data/processed/final_master_data_FINAL.csv` — per-station master table
  (23 tabular source-proxy features + NO2/O3/PM10 targets + image/S5P paths).
- `data/cams_reg_source_gold.csv` — CAMS-REG gold source shares for evaluation.

## What is NOT included (obtain separately)

The TabSatFusion backbone training (`pipeline/training_fusion_multi.py`) reads
raw Sentinel files, expected under `--datadir`:

```
<datadir>/
├── sentinel-2/   <img_path>.npy   # per-station Sentinel-2 crops
└── sentinel-5p/  <s5p_path>.nc    # per-station Sentinel-5P columns
```

These are gitignored (`data/raw/`) because they are large (tens of GB) and are
redistributed via an external archive, not Git.

- **Archive DOI / download:** `<ZENODO_OR_HF_DOI — to be added>`
- After downloading, point `--datadir` at the extracted Sentinel root (or unpack
  into `data/raw/`).

## Data sources and attribution

If you use or redistribute these data, retain the following attributions:

- **Sentinel-2 & Sentinel-5P** — Copernicus programme, free/full/open data
  policy (Regulation (EU) No 1159/2013). Credit:
  *"Contains modified Copernicus Sentinel data [year]."*
- **CAMS-REG emission data** (`cams_reg_source_gold.csv`) — Copernicus Atmosphere
  Monitoring Service. Credit:
  *"Generated using Copernicus Atmosphere Monitoring Service information [year]."*
- **Ground-truth air quality & station metadata** — European Environment Agency
  (EEA) benchmark curated by Rowley & Karakuş (2023),
  doi:10.1016/j.rse.2023.113609; EEA standard reuse policy (attribution).
- **Local infrastructure features** (road/building/factory counts, land use) —
  derived from **OpenStreetMap**, © OpenStreetMap contributors, licensed under
  the **Open Database License (ODbL) 1.0**. Redistribution of OSM-derived data
  must keep this attribution and remain share-alike under ODbL.
