#!/usr/bin/env python3
"""Make dashboard-slim report JSONs.

The Streamlit dashboard only reads a handful of fields per case. This script
strips each full report JSON down to just those fields so the app fits inside
free-tier hosting (Streamlit Community Cloud / HF Spaces) without the ~70 MB
originals.

Kept per case_key:
  - pollutant
  - eligibility: station_id, Latitude, Longitude, relerr_fusion
  - top_groups: [{source_category, positive_shap_sum}]
  - structured_reports: [{ranked_sources}]
  - reports: [report_text, ...]

Usage:
  python scripts/make_dashboard_slim.py \
    --in outputs/reports/full_shap_only.json \
         outputs/reports/full_shap_rag.json \
         outputs/reports/full_shap_rag_image.json \
    --out_dir outputs/reports_slim
"""
import argparse
import json
import os


def slim_entry(e: dict) -> dict:
    el = e.get("eligibility") or {}
    slim_elig = {
        k: el.get(k)
        for k in ("station_id", "Latitude", "Longitude", "relerr_fusion")
        if k in el
    }
    tg = e.get("top_groups") or []
    slim_tg = [
        {"source_category": g.get("source_category"),
         "positive_shap_sum": g.get("positive_shap_sum", 0)}
        for g in tg if isinstance(g, dict)
    ]
    sr = e.get("structured_reports")
    sr0 = (sr[0] if isinstance(sr, list) and sr else sr) or {}
    ranked = sr0.get("ranked_sources") or []
    reports = e.get("reports")
    if isinstance(reports, str):
        reports = [reports]
    reports = [r for r in (reports or []) if isinstance(r, str)]
    return {
        "pollutant": e.get("pollutant"),
        "eligibility": slim_elig,
        "top_groups": slim_tg,
        "structured_reports": [{"ranked_sources": ranked}],
        "reports": reports,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for path in args.inputs:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        slim = {k: slim_entry(v) for k, v in data.items()}
        out = os.path.join(args.out_dir, os.path.basename(path))
        with open(out, "w", encoding="utf-8") as f:
            json.dump(slim, f, ensure_ascii=False)
        print(f"{os.path.getsize(path)/1e6:6.1f} MB -> {os.path.getsize(out)/1e6:5.2f} MB  {out}")


if __name__ == "__main__":
    main()
