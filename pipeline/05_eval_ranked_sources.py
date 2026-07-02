#!/usr/bin/env python3
"""Evaluate LLM-produced ranked_sources against CAMS-REG gold.

This replaces the old b + beta*r + gamma*v formula-based evaluator. The v3
pipeline now has the LLM emit `ranked_sources` (a full ordering of the four
emission sources) directly in each structured report, synthesizing SHAP + RAG
+ image. We read that ordering verbatim and score it against the CAMS-REG
sector gold. There is no beta/gamma to tune anymore: the ranking is the model's.

Metrics (NO2 + PM10 only; O3 has no emission gold and is excluded):
  AC@1 / AC@2 / AC@3   dominant emitter is within the model's top-1/2/3
  MRR                  1 / rank of the dominant emitter (0 if absent)
  recall@1 / recall@3  fraction of CAMS present-set captured in top-1/3
                       (recall@1 is bounded by 1/|present|; read with recall@3)
  precision            model's #1 is in the CAMS present-set
  SHAP!=CAMS           divergence: model's #1 != CAMS dominant = 1 - AC@1

The three conditions (SHAP, SHAP+RAG, SHAP+RAG+Image) can now differ, because
the LLM ranks differently as evidence is added -- unlike the old formula where
RAG/image could not move the SHAP ranking.

Usage:
  python eval_ranked_sources.py \
    --reports_dir /path/to/reports \
    --gold /path/to/cams_reg_source_gold.csv \
    --out_csv /path/to/ranking_metrics_by_condition.csv
"""
import argparse
import json
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

EMISSION = ["traffic", "urban_anthropogenic", "industrial", "port_shipping"]
CAT2GOLD = {"traffic": "traffic", "industrial": "industrial",
            "urban_anthropogenic": "urban", "port_shipping": "port_shipping"}
GOLD2CAT = {v: k for k, v in CAT2GOLD.items()}
POL2PFX = {"no2": "nox", "pm10": "pm10"}   # O3 excluded (no emission gold)

CONDITION_FILES = {
    "SHAP": "full_shap_only.json",
    "SHAP+RAG": "full_shap_rag.json",
    "SHAP+RAG+Image": "full_shap_rag_image.json",
}


# ---------------------------------------------------------------------------
# reading ranked_sources out of an entry
# ---------------------------------------------------------------------------
def structured_of(entry):
    """Return the structured report dict for an entry (handles list/dict)."""
    sr = entry.get("structured_reports")
    if isinstance(sr, list):
        return sr[0] if sr else {}
    if isinstance(sr, dict):
        return sr
    return {}


def parse_report_json(entry):
    """Fallback: parse ranked_sources out of the raw report text if the
    structured field is missing."""
    rpt = entry.get("reports")
    txt = ""
    if isinstance(rpt, list):
        for r in reversed(rpt):
            if isinstance(r, str) and r.strip():
                txt = r
                break
    elif isinstance(rpt, str):
        txt = rpt
    s = txt.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        a, b = s.find("{"), s.rfind("}")
        if 0 <= a < b:
            try:
                return json.loads(s[a:b + 1])
            except Exception:
                return {}
        return {}


def clean_ranked(raw):
    """Coerce whatever the model gave into a permutation of the four sources.

    Keep the first valid occurrence of each source in order, then append any
    missing sources in canonical order so every case yields all four exactly
    once. Mirrors normalize_ranked_sources in the generator."""
    ranked = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, str):
                continue
            c = item.strip().lower().replace(" ", "_")
            if c in EMISSION and c not in ranked:
                ranked.append(c)
    for c in EMISSION:
        if c not in ranked:
            ranked.append(c)
    return ranked


def ranked_sources_of(entry):
    """Model's emission-source ranking for this entry, cleaned to 4 items.

    Returns (ranked_list, source_flag) where source_flag tells us where the
    ranking came from: 'structured', 'report_text', or 'fallback_canonical'."""
    sr = structured_of(entry)
    raw = sr.get("ranked_sources")
    if isinstance(raw, list) and raw:
        return clean_ranked(raw), "structured"
    # fallback: try parsing the raw report text
    obj = parse_report_json(entry)
    raw2 = obj.get("ranked_sources")
    if isinstance(raw2, list) and raw2:
        return clean_ranked(raw2), "report_text"
    # last resort: canonical order (flags a generation problem)
    return clean_ranked(None), "fallback_canonical"


def station_of(entry):
    return (entry.get("eligibility") or {}).get("station_id")


def pollutant_of(entry):
    return entry.get("pollutant")


# ---------------------------------------------------------------------------
# gold
# ---------------------------------------------------------------------------
def cams_target(station, pol, gold):
    if pol not in POL2PFX or station not in gold.index:
        return None
    r = gold.loc[station]
    if isinstance(r, pd.DataFrame):
        r = r.iloc[0]
    pfx = POL2PFX[pol]
    present = [c for c in EMISSION if bool(r[f"{pfx}_present_{CAT2GOLD[c]}"])]
    dom = GOLD2CAT.get(r[f"{pfx}_dom_modeled"])
    share = {c: float(r[f"{pfx}_share_{CAT2GOLD[c]}"]) for c in EMISSION}
    return {"present": present, "dom": dom, "share": share}


# ---------------------------------------------------------------------------
# per-condition evaluation
# ---------------------------------------------------------------------------
def load_condition(reports_dir, filename):
    path = f"{reports_dir}/{filename}"
    with open(path) as f:
        return json.load(f)


def eval_condition(data, gold, condition, dedup=True):
    src_counts = {"structured": 0, "report_text": 0, "fallback_canonical": 0}

    # Pass 1: gather eligible entries (NO2/PM10 in gold) with their ranking.
    eligible = []   # (ck, stn, pol, rank, cm)
    for ck, entry in data.items():
        pol = pollutant_of(entry)
        stn = station_of(entry)
        cm = cams_target(stn, pol, gold)
        rank, src = ranked_sources_of(entry)
        src_counts[src] += 1
        if cm is None:
            continue
        rel = (entry.get("eligibility") or {}).get("relerr_fusion")
        rel = rel if isinstance(rel, (int, float)) else float("inf")
        eligible.append((ck, stn, pol, rank, cm, rel))

    # Pass 2: dedup to one entry per (station, pollutant), keeping min relerr.
    if dedup:
        best = {}
        for tup in eligible:
            ck, stn, pol, rank, cm, rel = tup
            key = (stn, pol)
            if key not in best or rel < best[key][5]:
                best[key] = tup
        eligible = list(best.values())

    rows = []
    for ck, stn, pol, rank, cm, rel in eligible:
        dom = cm["dom"]
        present = set(cm["present"])
        row = {"condition": condition, "case": ck, "station": stn, "pollutant": pol,
               "top1": rank[0], "dom": dom}
        # AC@k: dominant emitter appears within the top-k of the ranking
        row["ac1"] = float(dom in rank[:1]) if dom else np.nan
        row["ac2"] = float(dom in rank[:2]) if dom else np.nan
        row["ac3"] = float(dom in rank[:3]) if dom else np.nan
        rpos = (rank.index(dom) + 1) if dom in rank else None
        row["mrr"] = (1.0 / rpos) if rpos else (0.0 if dom else np.nan)
        # recall@k: fraction of the CAMS present-set captured in the top-k.
        # recall@1 is bounded by 1/|present|, so it mostly reflects present-set
        # size; read it alongside recall@3, not on its own.
        row["recall1"] = (len(set(rank[:1]) & present) / len(present)
                          if present else np.nan)
        row["recall3"] = (len(set(rank[:3]) & present) / len(present)
                          if present else np.nan)
        row["prec_top1_present"] = float(rank[0] in present) if present else np.nan
        row["diverges"] = float(rank[0] != dom) if dom else np.nan
        rows.append(row)
    return pd.DataFrame(rows), src_counts


def summarize(df):
    return {
        "n": int(df.shape[0]),
        "ac1": round(df["ac1"].mean(), 3),
        "ac2": round(df["ac2"].mean(), 3),
        "ac3": round(df["ac3"].mean(), 3),
        "mrr": round(df["mrr"].mean(), 3),
        "recall1": round(df["recall1"].mean(), 3),
        "recall3": round(df["recall3"].mean(), 3),
        "precision": round(df["prec_top1_present"].mean(), 3),
        "divergence": round(df["diverges"].mean(), 3),
    }


def random_baselines():
    # 4 candidates, ranking is a uniform random permutation.
    #   AC@k = P(dominant in top-k) = k/4  -> AC@1 .25, AC@2 .50, AC@3 .75
    #   MRR  = mean(1/rank) over uniform position = (1+1/2+1/3+1/4)/4
    #   recall@k: with a present-set of size m, top-k captures on average
    #     k*m/4 of them out of m -> recall@k = k/4 in expectation, independent
    #     of m. So recall@1 ~ .25, recall@3 ~ .75 (references only).
    mrr = sum(1.0 / i for i in range(1, 5)) / 4.0
    return {"ac1": 0.25, "ac2": 0.50, "ac3": 0.75, "mrr": round(mrr, 3),
            "recall1": 0.25, "recall3": 0.75, "precision": 1.0,
            "divergence": 0.75}  # rough references


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_dir", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--per_case_csv", default=None,
                    help="optional: write per-case rows for all conditions")
    ap.add_argument("--common_only", action="store_true",
                    help="evaluate only case_keys present in ALL conditions "
                         "(fair cross-condition comparison on an identical set)")
    args = ap.parse_args()

    gold = pd.read_csv(args.gold).set_index("AirQualityStation")

    # Load all conditions first.
    loaded = {}
    for cond, fname in CONDITION_FILES.items():
        try:
            loaded[cond] = load_condition(args.reports_dir, fname)
        except FileNotFoundError:
            print(f"[skip] {cond}: {fname} not found")
    if not loaded:
        print("No conditions found.")
        return

    # Optionally restrict to the intersection of case_keys across conditions.
    common_keys = None
    if args.common_only:
        sets = [set(d.keys()) for d in loaded.values()]
        common_keys = set.intersection(*sets)
        print(f"[common_only] intersection of case_keys across "
              f"{len(loaded)} conditions = {len(common_keys)} cases\n")

    all_summ = []
    all_rows = []
    print("=== per-condition ranked_sources evaluation (NO2+PM10, O3 excluded) ===\n")
    for cond, data in loaded.items():
        if common_keys is not None:
            data = {k: v for k, v in data.items() if k in common_keys}
        df, srcc = eval_condition(data, gold, cond)
        summ = summarize(df)
        summ["condition"] = cond
        all_summ.append(summ)
        all_rows.append(df)
        n_total = sum(srcc.values())
        print(f"[{cond}] cases_total={n_total}  "
              f"ranked_from_structured={srcc['structured']}  "
              f"from_report_text={srcc['report_text']}  "
              f"fallback_canonical={srcc['fallback_canonical']}")
        if srcc["fallback_canonical"] > 0:
            print(f"    WARNING: {srcc['fallback_canonical']} cases had no "
                  f"ranked_sources -> canonical order used (check generation).")

    if not all_summ:
        print("No conditions evaluated.")
        return

    summ_df = pd.DataFrame(all_summ)[
        ["condition", "n", "ac1", "ac2", "ac3", "mrr",
         "precision", "divergence"]]

    rb = random_baselines()
    print("\n=== random baselines (4 candidates) ===")
    print(f"  ac1~{rb['ac1']}  ac2~{rb['ac2']}  ac3~{rb['ac3']}  "
          f"mrr~{rb['mrr']}  recall1~{rb['recall1']}  recall3~{rb['recall3']}  "
          f"divergence~{rb['divergence']}")

    print("\n=== metrics by condition ===")
    print(summ_df.to_string(index=False))

    print("\n(AC@k = dominant emitter within top-k (k=1,2,3);\n"
          " MRR = reciprocal rank of the dominant emitter;\n"
          " recall@k = fraction of CAMS present-set captured in top-k;\n"
          "   note recall@1 is bounded by 1/|present|, so read it with recall@3;\n"
          " divergence = model top-1 != CAMS dominant = 1 - AC@1, the structural finding;\n"
          " conditions can now differ because the LLM re-ranks as evidence is added.)")

    if args.out_csv:
        summ_df.to_csv(args.out_csv, index=False)
        print(f"\n[written] {args.out_csv}")
    if args.per_case_csv and all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(args.per_case_csv, index=False)
        print(f"[written] {args.per_case_csv}")


if __name__ == "__main__":
    main()
