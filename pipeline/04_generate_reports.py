#!/usr/bin/env python3
"""
paper_08_generate_1step_image_reports_v3.py
============================================
One-step SHAP+RAG+raw-image Alert2Source reports with structured visual-label predictions.

Built DIRECTLY from raw evidence sources -- the SAME inputs as
paper_04_generate_reports_v2_stratified.py. This script does NOT consume
reports_shap_rag_stratified.json. SHAP / source-registry / RAG / reliability are read from
their source CSV/JSONL files, RAG is retrieved in-process (lexical_retrieve over --rag_jsonl),
and the SHAP+RAG evidence prompt is constructed by paper_04's build_prompt(). The ONLY additions
for the image setting are:
  (1) the station-centered satellite image (from --visual_cases_jsonl, sent as image_url), and
  (2) the v3 structured JSON output: per-category visual prediction (4 emission sources) +
      primary_source_hypothesis (incl. none_secondary) + context (geo/met role) + report.

build_image_prompt() reuses build_prompt() for the evidence, then replaces paper_04's
"5-section Required output" with the v3 image+JSON output spec.

Usage
-----
python paper_08_generate_1step_image_reports_v3.py \
  --eligibility_csv   paper_outputs/diagnostic/diagnostic_eligibility_long.csv \
  --shap_feature_csv  paper_outputs/shap/shap_feature_long.csv \
  --shap_group_csv    paper_outputs/shap/shap_source_group_long.csv \
  --shap_reliability_csv paper_outputs/shap/shap_reliability_by_sample.csv \
  --source_registry   paper_outputs/kb/source_registry.csv \
  --rag_jsonl         paper_outputs/kb/air_quality_rag_database.jsonl \
  --visual_cases_jsonl paper_outputs/visual/visual_cases.jsonl \
  --project_root      . \
  --mode shap_rag --cases_per_pollutant 20 --n_repeats 1 \
  --output_json       paper_outputs/reports/reports_shap_rag_image_1step_stratified.json
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ==============================
# Raw-evidence + prompt building (reused verbatim from paper_04)
# ==============================

def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def lexical_retrieve(query: str, cards: List[dict], top_k: int = 5) -> List[dict]:
    """Dependency-light retrieval. Good enough for a controlled knowledge base."""
    q_tokens = set(re.findall(r"[a-zA-Z0-9_]+", query.lower()))
    scored = []
    for card in cards:
        text = " ".join(str(v) for v in card.values()).lower()
        tokens = set(re.findall(r"[a-zA-Z0-9_]+", text))
        score = len(q_tokens & tokens) / (len(q_tokens) + 1e-8)
        # Small boosts for exact category/pollutant/feature hits.
        for t in q_tokens:
            if t in text:
                score += 0.02
        scored.append((score, card))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for s, c in scored[:top_k] if s > 0]


def retrieve_cards(cards: List[dict], pollutant: str, top_features: List[dict],
                   top_groups: Optional[List[dict]] = None,
                   lexical_query: Optional[str] = None, lexical_extra: int = 0) -> List[dict]:
    """Deterministic source-knowledge assembly for one case.

    Pure lexical overlap (lexical_retrieve) silently drops the universal governance cards
    (rule.*, template.*) because the query never contains their tokens, and it can let an
    off-pollutant card crowd out a SHAP feature card. This assembles the grounding set
    deterministically instead:
      1. feature.{name} for EVERY SHAP top feature (so every attributed feature is grounded),
      2. pollutant.{pollutant} mechanism card for THIS pollutant only (off-pollutant excluded),
      3. provenance.* for any source category present in the SHAP features,
      4. all rule.* attribution / guardrail / reliability rules (always injected),
      5. template.* report template (always injected).
    `lexical_extra` (default 0) optionally appends a few extra lexical hits not already chosen.
    """
    by_id = {c.get("id"): c for c in cards if isinstance(c, dict) and c.get("id")}
    chosen: List[dict] = []
    seen = set()

    def add(card):
        if not card:
            return
        cid = card.get("id")
        if cid in seen:
            return
        seen.add(cid)
        chosen.append(card)

    # 1) feature card for every SHAP top feature (case-specific, SHAP order)
    feat_cats = set()
    for f in (top_features or []):
        name = f.get("feature_name")
        add(by_id.get(f"feature.{name}"))
        if f.get("source_category"):
            feat_cats.add(f.get("source_category"))

    # 2) pollutant mechanism card for THIS pollutant only
    add(by_id.get(f"pollutant.{str(pollutant).lower()}"))

    # 3) provenance cards for any source category present in the SHAP features
    for c in cards:
        if str(c.get("id", "")).startswith("provenance.") and c.get("source_category") in feat_cats:
            add(c)

    # 4) universal attribution / guardrail / reliability rules (always)
    for c in cards:
        if str(c.get("id", "")).startswith("rule."):
            add(c)

    # 5) report template (always)
    for c in cards:
        if str(c.get("id", "")).startswith("template."):
            add(c)

    # 6) optional lexical supplement (off by default)
    if lexical_extra and lexical_query:
        added = 0
        for c in lexical_retrieve(lexical_query, cards, top_k=len(cards)):
            if added >= lexical_extra:
                break
            if c.get("id") not in seen:
                add(c)
                added += 1

    return chosen


def build_case_key(row: pd.Series) -> str:
    return f"run{row['run']}__sample{row['sample_id']}__{row['pollutant']}"

def format_top_features(shap_feat: pd.DataFrame, run: int, sample_id: str, pol: str, k: int = 5) -> List[dict]:
    g = shap_feat[
        (shap_feat["run"] == run)
        & (shap_feat["sample_id"].astype(str) == str(sample_id))
        & (shap_feat["pollutant"] == pol)
    ].sort_values("shap_rank_abs")
    out = []
    for _, r in g.head(k).iterrows():
        out.append({
            "feature_name": r["feature_name"],
            "feature_value_raw": r.get("feature_value_raw", np.nan),
            "shap_value": float(r["shap_value"]),
            "direction": "INCREASES prediction" if r["shap_value"] > 0 else "DECREASES prediction" if r["shap_value"] < 0 else "ZERO contribution",
            "source_category": r.get("source_category", "other"),
        })
    return out

def format_top_groups(shap_group: pd.DataFrame, run: int, sample_id: str, pol: str, k: int = 3) -> List[dict]:
    g = shap_group[
        (shap_group["run"] == run)
        & (shap_group["sample_id"].astype(str) == str(sample_id))
        & (shap_group["pollutant"] == pol)
    ].sort_values("source_rank_positive")
    out = []
    for _, r in g.head(k).iterrows():
        out.append({
            "source_category": r["source_category"],
            "positive_shap_sum": float(r["positive_shap_sum"]),
            "signed_shap_sum": float(r["signed_shap_sum"]),
            "abs_shap_sum": float(r["abs_shap_sum"]),
        })
    return out

def registry_context(registry: pd.DataFrame, top_features: List[dict]) -> str:
    rows = []
    for f in top_features:
        reg = registry[registry["feature_name"] == f["feature_name"]]
        if reg.empty:
            continue
        r = reg.iloc[0]
        rows.append(
            f"- {r.feature_name}: category={r.source_category}; allowed={r.allowed_interpretation}; "
            f"forbidden={r.forbidden_claim}; report_phrase={r.report_phrase}"
        )
    return "\n".join(rows)

def build_prompt(mode: str, elig_row: pd.Series, top_features: List[dict], top_groups: List[dict],
                 registry_txt: str, rag_cards: List[dict], rel_txt: str) -> Tuple[str, str]:
    pol = str(elig_row["pollutant"]).upper()
    header = (
        "You are an environmental data analyst. Your task is to translate structured model evidence "
        "into an operational air-quality root-cause report. You must not infer causes directly. "
        "Use only the evidence provided. Do not claim interventional causality or chemical source apportionment."
    )

    if mode == "direct_llm":
        header += "\nThis ablation intentionally provides no SHAP evidence. Use uncertainty language."
    else:
        header += (
            "\nPositive SHAP means INCREASES prediction; negative SHAP means DECREASES prediction. "
            "Use exact feature names and exact direction phrases. A negative SHAP contribution means the "
            "feature pushed the model's prediction DOWN for this case; weigh sign and magnitude when ranking sources."
        )

    if mode in {"shap_registry", "shap_rag", "shap_rag_reliability"}:
        header += "\nUse the source registry to map features to source categories and to avoid forbidden claims."
    if mode in {"shap_rag", "shap_rag_reliability"}:
        header += "\nUse retrieved domain cards as grounding context, but do not introduce sources absent from SHAP evidence."
    if mode == "shap_rag_reliability":
        header += "\nApply reliability gates. If prediction or diagnostic evidence is weak, use low-confidence wording."

    feature_block = "\n".join(
        f"#{i+1}. {f['feature_name']}: value={f['feature_value_raw']}, SHAP={f['shap_value']:+.4f}, "
        f"direction={f['direction']}, source_category={f['source_category']}"
        for i, f in enumerate(top_features)
    ) or "No SHAP feature evidence provided."
    group_block = "\n".join(
        f"- {g['source_category']}: positive_sum={g['positive_shap_sum']:+.4f}, signed_sum={g['signed_shap_sum']:+.4f}"
        for g in top_groups
    ) or "No source-group evidence provided."
    rag_block = "\n".join(f"[{c.get('id')}] {c.get('text')}" for c in rag_cards) or "No RAG cards provided."

    user = f"""
Sample evidence
---------------
Run: {elig_row['run']}
Sample ID: {elig_row['sample_id']}
Pollutant: {pol}
Observed concentration: {elig_row.get('y_true', 'NA')}
AQFusionNet prediction: {elig_row.get('y_pred_fusion', 'NA')}
LightGBM diagnostic prediction: {elig_row.get('y_pred_lgbm', 'NA')}
Alert flag: {elig_row.get('alert_flag', 'NA')}

Top SHAP features
-----------------
{feature_block if mode != 'direct_llm' else 'Ablation mode: SHAP evidence intentionally hidden.'}

Source-group SHAP evidence
--------------------------
{group_block if mode != 'direct_llm' else 'Ablation mode: source-group evidence intentionally hidden.'}

Source registry context
-----------------------
{registry_txt if mode in {'shap_registry', 'shap_rag', 'shap_rag_reliability'} else 'Not provided in this ablation.'}

Retrieved RAG domain cards
--------------------------
{rag_block if mode in {'shap_rag', 'shap_rag_reliability'} else 'Not provided in this ablation.'}

Reliability evidence
--------------------
{rel_txt if mode == 'shap_rag_reliability' else 'Not provided in this ablation.'}

Required output
---------------
1. Alert summary.
2. Top source-proxy evidence. Mention exact feature names and exact SHAP direction phrases.
3. Root-cause source hypothesis. Use "source hypothesis" or "source-proxy evidence" language.
4. Uncertainty and limitations. Mention if evidence is insufficient.
5. Recommended interpretation for decision-makers.
""".strip()
    return header, user

def select_cases_for_reporting(cases: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """
    Select report cases for paper experiments.

    Changes vs original paper_04_generate_reports.py:
    - Original: cases = eligible.head(max_cases), which often selects only run1/NO2.
    - New: optional stratified sampling by pollutant and run.
    - New: optional shuffle to avoid order bias.
    """
    out = cases.copy()
    if args.pollutants:
        out = out[out["pollutant"].astype(str).isin(args.pollutants)].copy()
    if args.shuffle:
        out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    if args.cases_per_pollutant is not None:
        pieces = []
        for pol, g in out.groupby("pollutant", sort=True):
            if args.cases_per_run is not None:
                subpieces = []
                for run, rg in g.groupby("run", sort=True):
                    subpieces.append(rg.head(args.cases_per_run))
                gsel = pd.concat(subpieces, ignore_index=True) if subpieces else g.head(0)
                # Cap per pollutant if needed
                gsel = gsel.head(args.cases_per_pollutant)
            else:
                gsel = g.head(args.cases_per_pollutant)
            pieces.append(gsel)
        out = pd.concat(pieces, ignore_index=True) if pieces else out.head(0)
    elif args.max_cases is not None:
        out = out.head(args.max_cases)

    return out.reset_index(drop=True)



# ==============================
# Image handling + JSONL utils (from paper_08 v3)
# ==============================

def load_jsonl_by_key(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None or not str(path):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = obj.get("case_key") or obj.get("key")
            if key:
                out[str(key)] = obj
    return out

def candidate_roots(project_root: Optional[Path], manifest_path: Optional[Path]) -> List[Path]:
    roots: List[Path] = []
    for p in [
        Path.cwd(),
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
        project_root,
        manifest_path.parent if manifest_path else None,
        manifest_path.parent.parent if manifest_path and manifest_path.parent else None,
    ]:
        if p is not None:
            try:
                pp = p.resolve()
            except Exception:
                pp = p
            if pp not in roots:
                roots.append(pp)
    return roots

def resolve_existing_path(path_like: Any, roots: Iterable[Path]) -> Optional[Path]:
    if not path_like:
        return None
    raw = Path(str(path_like)).expanduser()
    if raw.is_absolute() and raw.exists():
        return raw
    if raw.exists():
        return raw.resolve()
    for root in roots:
        cand = (root / raw).resolve()
        if cand.exists():
            return cand
    return None

def image_label(path: Path, rec: Optional[Dict[str, Any]] = None) -> str:
    rec = rec or {}
    if rec.get("label"):
        return str(rec.get("label"))
    name = path.name
    zm = re.search(r"_z(\d+)_", name)
    if zm:
        z = int(zm.group(1))
        if z >= 16:
            scale_hint = "local scale; useful for roads/buildings near the station"
        elif z >= 14:
            scale_hint = "neighborhood scale; useful for 3--5 km context"
        else:
            scale_hint = "wide scale; useful for 5--10 km context such as ports, waterways, or regional land cover"
        return f"Satellite image, zoom={z} ({scale_hint})"
    if "streetview" in name.lower():
        return "Street View image"
    return "Geospatial image"

def image_to_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"

def extract_image_records(
    case_key: str,
    visual_case: Optional[Dict[str, Any]],
    visual_audit: Optional[Dict[str, Any]],
    roots: Iterable[Path],
    include_streetview: bool = False,
) -> List[Dict[str, Any]]:
    """Collect image records from visual_cases and/or visual_inspection metadata.

    The generation prompt uses only the resolved image files and labels. It does
    not use visual_inspection textual outputs.
    """
    records: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def add_record(path_like: Any, meta: Optional[Dict[str, Any]] = None, source: str = "unknown") -> None:
        meta = meta or {}
        p = resolve_existing_path(path_like, roots)
        if p is None:
            return
        key = str(p.resolve())
        if key in seen:
            return
        seen.add(key)
        rec: Dict[str, Any] = {
            "path": str(p),
            "label": image_label(p, meta),
            "source": meta.get("source", source),
        }
        for fld in ["zoom", "heading", "maptype", "scale", "provider"]:
            if fld in meta:
                rec[fld] = meta.get(fld)
        records.append(rec)

    # 1) Prefer explicit image_records in visual inspection because those are
    # already selected for the visual experiment. We do NOT use inspection text.
    if isinstance(visual_audit, dict):
        for rec in visual_audit.get("image_records", []) or []:
            if isinstance(rec, dict):
                add_record(rec.get("path"), rec, source=str(rec.get("source", "visual_inspection")))
        for p in visual_audit.get("image_paths", []) or []:
            add_record(p, {}, source="visual_inspection")

    # 2) Fallback to the visual case manifest.
    if isinstance(visual_case, dict):
        sm = visual_case.get("static_map", {}) or {}
        if sm:
            add_record(sm.get("image_path"), {**sm, "source": "static_map"}, source="static_map")
        if include_streetview:
            for item in visual_case.get("streetview_images", []) or []:
                if isinstance(item, dict):
                    add_record(item.get("image_path"), {**item, "source": "streetview"}, source="streetview")

    # Prefer wider satellite images first, then local images, then streetview.
    def sort_key(rec: Dict[str, Any]) -> Tuple[int, int, str]:
        src = str(rec.get("source", ""))
        p = Path(str(rec.get("path", "")))
        zm = re.search(r"_z(\d+)_", p.name)
        z = int(zm.group(1)) if zm else int(rec.get("zoom") or 99)
        if "street" in src.lower() or "street" in p.name.lower():
            return (1, 999, p.name)
        return (0, z, p.name)

    return sorted(records, key=sort_key)

def radius_hint(top_features: List[Dict[str, Any]]) -> str:
    hints: List[str] = []
    for f in top_features or []:
        name = str(f.get("feature_name", ""))
        if re.search(r"_10km", name):
            hints.append(f"- {name}: 10 km proxy; if only local imagery is provided, non-visibility should be treated as crop-limited.")
        elif re.search(r"_5km", name):
            hints.append(f"- {name}: 5 km proxy; if only local imagery is provided, non-visibility should be treated as crop-limited.")
        elif re.search(r"_1km", name):
            hints.append(f"- {name}: 1 km proxy; local-scale imagery can be informative.")
        elif re.search(r"_500m", name):
            hints.append(f"- {name}: 500 m proxy; local-scale imagery can be informative.")
    return "\n".join(hints) if hints else "- No explicit radius-coded feature among the top SHAP features."

def image_context_lines(image_records: List[Dict[str, Any]]) -> str:
    if not image_records:
        return "- no image available"
    lines: List[str] = []
    for i, rec in enumerate(image_records, start=1):
        path = Path(str(rec.get("path", "")))
        lines.append(f"- Image {i}: {rec.get('label')}; file={path.name}")
    return "\n".join(lines)



# ==============================
# v3 taxonomy + output spec
# ==============================

EMISSION_SOURCES = [
    "traffic",
    "urban_anthropogenic",
    "industrial",
    "port_shipping",
]

CONTEXT_CATEGORIES = [
    "geographic_context",
    "meteorological_condition",
]

IMAGE_GROUNDING_RULES = (
    "Additional direct-image grounding rules for the final visual setting:\n"
    "1. You receive the station-centered geospatial image crop(s) directly, together with SHAP evidence and source-knowledge RAG entries.\n"
    "2. SHAP, RAG, and the image are THREE complementary lines of evidence. SHAP quantifies which proxy features drive the model's concentration prediction; the image provides direct spatial verification of what is physically present near the station; RAG provides domain knowledge. Synthesize all three to rank the sources.\n"
    "3. Do NOT assume SHAP is always correct. SHAP reflects the model's learned associations, which can be biased or wrong, especially when the model fits the pollutant poorly. A clear, confident visual cue is independent ground-truth evidence that can correct SHAP.\n"
    "4. Override rule: a STRONG visual cue (status=present with confidence 4 or 5) MAY raise a source above its SHAP rank, INCLUDING a source whose SHAP is negative or zero, when the image clearly shows that source's physical signature (e.g. a large highway interchange for traffic, dense built-up blocks for urban, factory sheds/stacks for industrial, harbor/docks/cranes/vessels for port_shipping). A WEAK or ambiguous cue (confidence 1-3) must NOT override SHAP; in that case defer to SHAP.\n"
    "5. When SHAP and the image conflict, follow the stronger evidence and state the reason explicitly in ranking_rationale (e.g. 'SHAP negative but harbor clearly visible at conf 5 -> elevated').\n"
    "6. If a SHAP-supported source is not visible in the retrieved crop, describe it as not visually corroborated in the retrieved crop; do not claim that the source is absent from the real environment. Crop non-visibility is not evidence against a source.\n"
    "7. Meteorological variables cannot be visually assessed from static geospatial imagery.\n"
    "8. Use source-proxy/source-hypothesis wording and preserve uncertainty.\n"
    "9. The alert/report-eligible status is defined by the study threshold and eligibility filters, not by the observed value exceeding model predictions. In the alert summary, do not write that the alert is triggered because the observation exceeds TabSatFusion or LightGBM predictions; report predictions only as supporting model estimates.\n"
    "10. Avoid generic policy prescriptions such as urban planning or traffic management actions. Keep the recommendation to evidence interpretation, targeted follow-up inspection, or contextual review.\n"
    "11. Taxonomy: only the four emission sources (traffic, urban_anthropogenic, industrial, port_shipping) are root-cause candidates and must be ranked. geographic_context and meteorological_condition are context modulators, reported but never placed in ranked_sources. Use no other category."
)

V3_OUTPUT_SPEC = """First produce structured per-category visual-label predictions for the FOUR emission sources, then synthesize SHAP + RAG + image into a ranked list of sources, then produce the report.
Emission-source categories to predict and rank: traffic, urban_anthropogenic, industrial, port_shipping.
Allowed status values: present, visually_absent, not_visible_in_crop, uncertain, not_visually_assessable.
Status definitions:
- present: a visible cue for the category is in the crop(s).
- visually_absent: the crop scale/quality is usable for the category, but no cue is visible.
- not_visible_in_crop: the category may matter at a broader radius, but the crop(s) cannot verify it.
- uncertain: visual evidence is ambiguous even after inspecting the image(s).
- not_visually_assessable: cannot be judged from static geospatial imagery.
not_visible_in_crop and visually_absent describe the CROP, not real-world absence.

ranked_sources: rank ALL FOUR emission sources from most to least likely to be the dominant local source, by holistically weighing SHAP, RAG, and the image. Guidance:
- A source with strong positive SHAP support ranks high.
- A source shown by a STRONG visual cue (present, confidence 4-5) may rank high EVEN IF its SHAP is negative, zero, or weak; explain this in ranking_rationale.
- A WEAK or ambiguous visual cue (confidence 1-3) does not outweigh SHAP; defer to SHAP in that case.
- Crop non-visibility (not_visible_in_crop) is neutral: it neither raises nor lowers a source.
- Every one of the four sources must appear exactly once in ranked_sources.

Required one-step report content (the report field, plain text). Write ALL SIX sections below as a multi-sentence narrative; do not compress them into one or two sentences:
1. Alert summary. Report-eligible elevated case under the study criterion; state the observed concentration and the supporting model estimates, but do not define the alert as the observation exceeding the model predictions.
2. SHAP-supported source-proxy evidence. Cite the leading contributing features by their EXACT feature name, EXACT SHAP direction phrase (increases/decreases prediction), AND the numeric SHAP value, e.g. "PopulationDensity (+0.3705, increases prediction)". Cover the top positive-SHAP features, and note any strongly negative-SHAP features too.
3. Image-grounded spatial verification. Describe the cues actually visible in the crop(s) and connect each to its source category. Where a visible cue agrees with positive SHAP, note the corroboration. Where a STRONG visible cue (conf 4-5) contradicts SHAP (SHAP negative/zero but the feature is clearly present, or vice versa), state the conflict and explain how you weighed the two lines of evidence.
4. Crop-limited or not-assessable cues. Distinguish crop non-visibility from real-world absence.
5. Root-cause source hypothesis. State the top entry of ranked_sources as the leading hypothesis, justified by the synthesis of SHAP, RAG, and image. If SHAP and image disagreed, explain why the chosen source won. If the pollutant is driven by context rather than any local emission source (e.g. a secondary/photochemical pollutant), set primary_source_hypothesis to none_secondary and say so.
6. Limitations and recommended interpretation. Note absent or partial visual corroboration where relevant; recommend targeted follow-up; avoid broad policy prescriptions.
Context modulators: any positive-SHAP meteorological_condition or geographic_context feature must also be reported in the body using its exact feature name and SHAP direction, framed as an accumulation, dispersion, or formation context (its role is additionally captured in the context field). These context features must never appear in ranked_sources or be named as the root-cause source.

primary_source_hypothesis: the top entry of ranked_sources. If the pollutant drivers are purely contextual (e.g. a secondary/photochemical pollutant) and no emission source is meaningfully implicated, set it to none_secondary.

Return EXACTLY ONE valid JSON object and no surrounding markdown/code fence. Use this exact schema:
{
  "ranked_sources": ["<1st of traffic|urban_anthropogenic|industrial|port_shipping>", "<2nd>", "<3rd>", "<4th>"],
  "ranking_rationale": {
    "traffic": "one sentence: how SHAP and image were weighed for this source",
    "urban_anthropogenic": "one sentence",
    "industrial": "one sentence",
    "port_shipping": "one sentence"
  },
  "primary_source_hypothesis": "traffic|urban_anthropogenic|industrial|port_shipping|none_secondary",
  "visual_evidence_prediction": {
    "traffic": {"status": "present|visually_absent|not_visible_in_crop|uncertain|not_visually_assessable", "confidence": 1, "rationale": "short visual rationale"},
    "urban_anthropogenic": {"status": "present|visually_absent|not_visible_in_crop|uncertain|not_visually_assessable", "confidence": 1, "rationale": "short visual rationale"},
    "industrial": {"status": "present|visually_absent|not_visible_in_crop|uncertain|not_visually_assessable", "confidence": 1, "rationale": "short visual rationale"},
    "port_shipping": {"status": "present|visually_absent|not_visible_in_crop|uncertain|not_visually_assessable", "confidence": 1, "rationale": "short visual rationale"}
  },
  "context": {
    "geographic_context": {"role": "increases|decreases|neutral", "rationale": "short"},
    "meteorological_condition": {"role": "increases|decreases|neutral", "rationale": "short"}
  },
  "report": "concise operational report text"
}
The confidence field must be an integer from 1 to 5, where 5 means highly confident.
ranked_sources must contain all four emission sources exactly once. The report field must be plain text, not markdown tables."""


# ---- condition -> (build_prompt mode, attach image?) -------------------------
# The four experimental conditions share ONE output schema (the v3 JSON). The text
# conditions reuse the same SHAP/RAG evidence prompt and emit not_visually_assessable
# for every emission source (no image was inspected); only shap_rag_image attaches the
# satellite crop and produces real per-source visual labels.
CONDITION_MAP = {
    "direct_llm":     ("direct_llm", False),
    "shap_only":      ("shap_only",  False),
    "shap_rag":       ("shap_rag",   False),
    "shap_rag_image": ("shap_rag",   True),
}

TEXT_GROUNDING_RULES = (
    "Additional grounding rules for the no-image (statistical-only) setting:\n"
    "1. No station image is provided in this setting. You must NOT claim to observe any visual cue, "
    "built-up area, road, factory, or port from imagery.\n"
    "2. For EVERY emission source in visual_evidence_prediction, set status to not_visually_assessable "
    "and confidence to 1, because no image was inspected.\n"
    "3. Ground the report strictly in the SHAP evidence (and retrieved domain cards, if provided). "
    "Do not invent visual corroboration.\n"
    "4. Rank all four emission sources in ranked_sources using the available evidence. Use the SHAP "
    "magnitude and direction to order them; a more strongly contributing source ranks higher. SHAP is "
    "the only case-specific signal here, so the ranking follows SHAP, but treat SHAP as the model's "
    "learned association rather than ground truth and preserve uncertainty.\n"
    "5. Do not write an image-grounded or crop-limited section; state that no image was available and omit visual claims.\n"
    "6. Use source-proxy / source-hypothesis wording and preserve uncertainty.\n"
    "7. The alert/report-eligible status is defined by the study threshold and eligibility filters, not by the observation exceeding model predictions. Report predictions only as supporting model estimates.\n"
    "8. Avoid generic policy prescriptions; keep the recommendation to evidence interpretation or targeted follow-up.\n"
    "9. Taxonomy: only the four emission sources (traffic, urban_anthropogenic, industrial, port_shipping) are root-cause candidates and must be ranked. "
    "geographic_context and meteorological_condition are context modulators, reported but never placed in ranked_sources. Use no other category."
)

V3_OUTPUT_SPEC_TEXT = """First produce structured per-category visual-label predictions for the FOUR emission sources, then rank the sources from the SHAP evidence, then produce the report.
Emission-source categories to predict and rank: traffic, urban_anthropogenic, industrial, port_shipping.
No image is provided in this setting, so the visual labels cannot be assessed: set EVERY emission source's status to "not_visually_assessable" and confidence to 1.

ranked_sources: rank ALL FOUR emission sources from most to least likely to be the dominant local source, using the SHAP evidence (and RAG if provided). Order them by SHAP contribution; a more strongly contributing source ranks higher. Every one of the four sources must appear exactly once. Treat SHAP as the model's learned association, not ground truth, and keep the ranking's confidence appropriately hedged.

Required report content (the report field, plain text):
1. Alert summary. Report-eligible elevated case under the study criterion; do not define the alert as exceeding model predictions.
2. SHAP-supported source-proxy evidence. Use exact feature names and exact SHAP direction phrases and numeric values. If SHAP evidence is hidden in this setting, state that evidence is insufficient for case-specific attribution and keep the ranking generic and low-confidence.
3. Source ranking. State ranked_sources and the SHAP-based reasoning that orders them; the top entry is the leading hypothesis. If the pollutant is driven by context rather than any local emission source (e.g. a secondary/photochemical pollutant), say so and set primary_source_hypothesis to none_secondary.
4. Limitations and recommended interpretation. Note that no image was available for visual corroboration; targeted follow-up; avoid broad policy prescriptions.
Context modulators: any positive-SHAP meteorological_condition or geographic_context feature must be reported in the body using its exact feature name and SHAP direction, framed as an accumulation, dispersion, or formation context (its role is additionally captured in the context field). These context features must never appear in ranked_sources or be named as the root-cause source.

primary_source_hypothesis: the top entry of ranked_sources. If the pollutant drivers are purely contextual (e.g. a secondary/photochemical pollutant), set it to none_secondary.

Return EXACTLY ONE valid JSON object and no surrounding markdown/code fence. Use this exact schema:
{
  "ranked_sources": ["<1st of traffic|urban_anthropogenic|industrial|port_shipping>", "<2nd>", "<3rd>", "<4th>"],
  "ranking_rationale": {
    "traffic": "one sentence: SHAP-based reason for this source's rank",
    "urban_anthropogenic": "one sentence",
    "industrial": "one sentence",
    "port_shipping": "one sentence"
  },
  "primary_source_hypothesis": "traffic|urban_anthropogenic|industrial|port_shipping|none_secondary",
  "visual_evidence_prediction": {
    "traffic": {"status": "not_visually_assessable", "confidence": 1, "rationale": "no image inspected"},
    "urban_anthropogenic": {"status": "not_visually_assessable", "confidence": 1, "rationale": "no image inspected"},
    "industrial": {"status": "not_visually_assessable", "confidence": 1, "rationale": "no image inspected"},
    "port_shipping": {"status": "not_visually_assessable", "confidence": 1, "rationale": "no image inspected"}
  },
  "context": {
    "geographic_context": {"role": "increases|decreases|neutral", "rationale": "short"},
    "meteorological_condition": {"role": "increases|decreases|neutral", "rationale": "short"}
  },
  "report": "concise operational report text"
}
ranked_sources must contain all four emission sources exactly once. For this no-image setting, all four emission-source statuses are not_visually_assessable with confidence 1.
The report field must be plain text, not markdown tables."""



# ==============================
# v3 structured-output parsing (from paper_08 v3)
# ==============================

ALLOWED_VISUAL_STATUSES = {
    "present",
    "visually_absent",
    "not_visible_in_crop",
    "uncertain",
    "not_visually_assessable",
}

def extract_json_object(text: str) -> Dict[str, Any]:
    """Parse a model response that should contain one JSON object.

    The prompt asks for JSON-only output, but this function tolerates accidental
    code fences or surrounding text to make batch generation more robust.
    """
    if not text or not str(text).strip():
        raise ValueError("empty model output")
    s = str(text).strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I).strip()
        s = re.sub(r"\s*```$", "", s).strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        start = s.find("{")
        end = s.rfind("}")
        if start < 0 or end <= start:
            raise
        obj = json.loads(s[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("model output JSON is not an object")
    return obj

def normalize_visual_prediction(obj: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    pred = obj.get("visual_evidence_prediction") or obj.get("visual_evidence") or {}
    if not isinstance(pred, dict):
        pred = {}
    out: Dict[str, Dict[str, Any]] = {}
    for cat in EMISSION_SOURCES:
        raw = pred.get(cat, {})
        if isinstance(raw, str):
            status = raw.strip().lower(); rationale = ""; confidence = None
        elif isinstance(raw, dict):
            status = str(raw.get("status", "uncertain")).strip().lower()
            rationale = str(raw.get("rationale", "")).strip()
            confidence = raw.get("confidence")
        else:
            status = "uncertain"; rationale = "missing or invalid category prediction"; confidence = None
        status = status.replace(" ", "_")
        if status not in ALLOWED_VISUAL_STATUSES:
            status = "uncertain"
        try:
            confidence_i = max(1, min(5, int(confidence)))
        except Exception:
            confidence_i = None
        out[cat] = {"status": status, "confidence": confidence_i, "rationale": rationale}
    return out

ALLOWED_CONTEXT_ROLES = {"increases", "decreases", "neutral"}

def normalize_context(obj: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    ctx = obj.get("context") or {}
    if not isinstance(ctx, dict):
        ctx = {}
    out: Dict[str, Dict[str, Any]] = {}
    for cat in CONTEXT_CATEGORIES:
        raw = ctx.get(cat, {})
        if isinstance(raw, dict):
            role = str(raw.get("role", "neutral")).strip().lower()
            rationale = str(raw.get("rationale", "")).strip()
        else:
            role = "neutral"; rationale = ""
        if role not in ALLOWED_CONTEXT_ROLES:
            role = "neutral"
        out[cat] = {"role": role, "rationale": rationale}
    return out

def normalize_primary_source(obj: Dict[str, Any]) -> Optional[str]:
    val = obj.get("primary_source_hypothesis")
    if not isinstance(val, str):
        return None
    v = val.strip().lower().replace(" ", "_")
    allowed = set(EMISSION_SOURCES) | {"none_secondary"}
    return v if v in allowed else None

def normalize_ranked_sources(obj: Dict[str, Any]) -> List[str]:
    """Return a clean permutation of the four emission sources.

    The model is asked for a 4-element ranking, but may emit duplicates, omit
    entries, or misspell. Keep the first valid occurrence of each source in the
    given order, then append any missing sources (so downstream eval always sees
    all four exactly once). If the field is absent/garbage, fall back to the
    canonical EMISSION_SOURCES order.
    """
    raw = obj.get("ranked_sources")
    ranked: List[str] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, str):
                continue
            c = item.strip().lower().replace(" ", "_")
            if c in EMISSION_SOURCES and c not in ranked:
                ranked.append(c)
    for c in EMISSION_SOURCES:  # append any source the model dropped
        if c not in ranked:
            ranked.append(c)
    return ranked

def normalize_ranking_rationale(obj: Dict[str, Any]) -> Dict[str, str]:
    raw = obj.get("ranking_rationale") or {}
    out: Dict[str, str] = {}
    if isinstance(raw, dict):
        for cat in EMISSION_SOURCES:
            v = raw.get(cat)
            out[cat] = str(v).strip() if isinstance(v, (str, int, float)) else ""
    else:
        for cat in EMISSION_SOURCES:
            out[cat] = ""
    return out

def normalize_structured_report(raw_text: str) -> Dict[str, Any]:
    obj = extract_json_object(raw_text)
    pred = normalize_visual_prediction(obj)
    ctx = normalize_context(obj)
    primary = normalize_primary_source(obj)
    ranked = normalize_ranked_sources(obj)
    rationale = normalize_ranking_rationale(obj)
    # If the model gave a ranking but no/invalid primary, default primary to the top rank.
    if primary is None and ranked:
        primary = ranked[0]
    report = obj.get("report", "")
    if isinstance(report, (dict, list)):
        report = json.dumps(report, ensure_ascii=False)
    report = str(report).strip()
    return {
        "ranked_sources": ranked,
        "ranking_rationale": rationale,
        "primary_source_hypothesis": primary,
        "visual_evidence_prediction": pred,
        "context": ctx,
        "report": report,
        "raw_output_parse_ok": True,
    }



# ==============================
# OpenAI vision call (from paper_08 v3)
# ==============================

def call_openai_vision_report(
    system: str,
    user: str,
    image_records: List[Dict[str, Any]],
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    image_detail: str,
) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    content: List[Dict[str, Any]] = [{"type": "text", "text": user}]
    for i, rec in enumerate(image_records, start=1):
        path = Path(str(rec["path"]))
        content.append({"type": "text", "text": f"Image {i}: {rec.get('label', 'Geospatial image')}"})
        image_payload: Dict[str, Any] = {"url": image_to_data_url(path)}
        if image_detail and image_detail.lower() != "none":
            image_payload["detail"] = image_detail
        content.append({"type": "image_url", "image_url": image_payload})

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    return {"text": text, "usage": usage.model_dump() if hasattr(usage, "model_dump") else None}



# ==============================
# Image prompt = paper_04 evidence + v3 output spec
# ==============================

def build_unified_prompt(mode: str, elig_row, top_features: List[dict], top_groups: List[dict],
                         registry_txt: str, rag_cards: List[dict], rel_txt: str,
                         image_records: List[Dict[str, Any]], with_image: bool) -> Tuple[str, str]:
    """One prompt builder for all four conditions.

    The EVIDENCE block (Sample evidence, SHAP features, source-group, registry, RAG) is
    produced by paper_04's build_prompt() and is byte-identical across conditions; only the
    OUTPUT INSTRUCTION (and, for the image condition, the attached crop) differs:
      - text conditions (direct_llm / shap_only / shap_rag): no image; the model fills every
        emission source's visual label with not_visually_assessable and grounds the report in
        SHAP/RAG only.
      - image condition (shap_rag_image): the satellite crop is attached and the model produces
        real per-source visual labels.
    Both emit the SAME v3 JSON schema, so every condition has an identical output format.
    """
    system, user = build_prompt(mode, elig_row, top_features, top_groups, registry_txt, rag_cards, rel_txt)

    # Drop paper_04's "Required output" (5-section text report) and everything after it; keep evidence.
    cut = user.find("Required output")
    evidence = user[:cut].rstrip() if cut >= 0 else user.rstrip()

    if with_image:
        system = system + "\n\n" + IMAGE_GROUNDING_RULES
        image_block = (
            "\n\nDirect visual input context\n"
            "---------------------------\n"
            "The image(s) attached to this prompt are station-centered geospatial crop(s) for this case. "
            "Inspect the image(s) directly; do not rely on any precomputed visual evidence card.\n"
            "Provided image(s):\n"
            f"{image_context_lines(image_records)}\n"
            "Radius and crop-interpretation hints from SHAP feature names:\n"
            f"{radius_hint(top_features)}\n"
        )
        user = evidence + image_block + "\n\n" + V3_OUTPUT_SPEC
    else:
        system = system + "\n\n" + TEXT_GROUNDING_RULES
        user = evidence + "\n\n" + V3_OUTPUT_SPEC_TEXT
    return system, user



# ==============================
# Main
# ==============================

def reports_are_complete(case: Dict[str, Any], n_repeats: int) -> bool:
    reps = case.get("reports", []) or []
    valid = [r for r in reps if isinstance(r, str) and r.strip() and not r.startswith("[API_ERROR]") and not r.startswith("[PROMPT_ONLY")]
    return len(valid) >= n_repeats

def _empty_structured(report_text: str, parse_error: str) -> Dict[str, Any]:
    return {
        "primary_source_hypothesis": None,
        "visual_evidence_prediction": normalize_visual_prediction({}),
        "context": normalize_context({}),
        "report": report_text,
        "raw_output_parse_ok": False,
        "parse_error": parse_error,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="One-step SHAP+RAG+image reports (v3) built directly from raw evidence (paper_04 inputs)."
    )
    # ---- raw evidence inputs (same as paper_04) ----
    ap.add_argument("--eligibility_csv", required=True)
    ap.add_argument("--shap_feature_csv", required=True)
    ap.add_argument("--shap_group_csv", required=True)
    ap.add_argument("--source_registry", required=True)
    ap.add_argument("--rag_jsonl", required=True)
    ap.add_argument("--condition", default="shap_rag_image",
                    choices=["direct_llm", "shap_only", "shap_rag", "shap_rag_image"],
                    help="direct_llm / shap_only / shap_rag are text-only; shap_rag_image attaches "
                         "the station satellite crop. All four emit the same v3 JSON output schema.")
    ap.add_argument("--eligible_col", default="final_report_eligible_oracle")
    # ---- case selection (same as paper_04) ----
    ap.add_argument("--max_cases", type=int, default=0, help="0 = all eligible (after filters).")
    ap.add_argument("--cases_per_pollutant", type=int, default=None)
    ap.add_argument("--cases_per_run", type=int, default=None)
    ap.add_argument("--pollutants", nargs="*", default=None)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dedup_stations", action="store_true",
                    help="Keep ONE case per (station_id, pollutant) for dashboard/full runs "
                         "(picks the representative run by --dedup_metric). Do NOT use for the stratified eval set.")
    ap.add_argument("--dedup_metric", default="relerr_fusion",
                    help="Column minimized when choosing the representative run per (station_id, pollutant).")
    # ---- image inputs ----
    ap.add_argument("--visual_cases_jsonl", default="", help="visual_cases.jsonl (static_map.image_path per case).")
    ap.add_argument("--visual_inspection_jsonl", default="", help="Optional; used for image paths only, NOT prompt text.")
    ap.add_argument("--project_root", default="", help="Root to resolve relative image paths.")
    ap.add_argument("--include_streetview", action="store_true")
    ap.add_argument("--require_images", action="store_true")
    # ---- OpenAI / generation ----
    ap.add_argument("--openai_key", default=os.getenv("OPENAI_API_KEY", ""))
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=2500)
    ap.add_argument("--n_repeats", type=int, default=3)
    ap.add_argument("--image_detail", default="auto", choices=["auto", "low", "high", "none"])
    ap.add_argument("--sleep_s", type=float, default=0.5)
    ap.add_argument("--require_api", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    if args.require_api and not args.openai_key:
        print("[ERROR] --require_api set but OPENAI_API_KEY/--openai_key missing.", file=sys.stderr)
        sys.exit(1)
    if args.max_cases == 0:
        args.max_cases = None  # select_cases_for_reporting -> no head() cap = all

    # ---- resolve condition -> (build_prompt mode, attach image?) ----
    mode, with_image = CONDITION_MAP[args.condition]
    print(f"[INFO] Condition: {args.condition}  (build_prompt mode={mode}, image={'YES' if with_image else 'no'})")
    if with_image and not args.visual_cases_jsonl:
        print("[ERROR] condition shap_rag_image requires --visual_cases_jsonl (image manifest).", file=sys.stderr)
        sys.exit(1)
    if with_image and not args.project_root:
        print("[WARN] shap_rag_image without --project_root: relative image paths may not resolve.", file=sys.stderr)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)

    # ---- load raw evidence (paper_04 style) ----
    elig = pd.read_csv(args.eligibility_csv)
    shap_feat = pd.read_csv(args.shap_feature_csv)
    shap_group = pd.read_csv(args.shap_group_csv)
    registry = pd.read_csv(args.source_registry)
    rag_cards_all = load_jsonl(args.rag_jsonl)

    # ---- load image manifests + resolution roots (only meaningful for the image condition) ----
    vis_cases_path = Path(args.visual_cases_jsonl) if (with_image and args.visual_cases_jsonl) else None
    vis_audit_path = Path(args.visual_inspection_jsonl) if (with_image and args.visual_inspection_jsonl) else None
    project_root = Path(args.project_root).expanduser().resolve() if args.project_root else None
    visual_cases = load_jsonl_by_key(vis_cases_path) if vis_cases_path else {}
    visual_audit = load_jsonl_by_key(vis_audit_path) if vis_audit_path else {}
    roots = candidate_roots(project_root, vis_cases_path or vis_audit_path or Path(args.eligibility_csv))
    if with_image:
        print("[INFO] Path resolution roots:")
        for r in roots:
            print(f"  - {r}")

    # ---- select cases (paper_04 style) ----
    cases = elig[elig[args.eligible_col].astype(bool)].copy()
    # Optional: collapse to ONE representative case per (station_id, pollutant). The eligibility
    # pool is run-pooled, so the same station can appear under several runs; for a station map
    # (dashboard / full coverage) we keep the run with the smallest --dedup_metric (most accurate
    # prediction). NOT for the stratified eval set, which stays as-is for the paper's ablation.
    if args.dedup_stations:
        before = len(cases)
        if args.dedup_metric in cases.columns:
            cases = (cases.sort_values(["station_id", "pollutant", args.dedup_metric],
                                       ascending=[True, True, True])
                          .drop_duplicates(subset=["station_id", "pollutant"], keep="first"))
            note = f"min {args.dedup_metric}"
        else:
            cases = cases.drop_duplicates(subset=["station_id", "pollutant"], keep="first")
            note = f"first (column '{args.dedup_metric}' not found)"
        print(f"[INFO] Dedup to one case per (station_id, pollutant) by {note}: {before} -> {len(cases)}")
    # The IMAGE condition can only run on cases that have an image entry in the manifest, so intersect
    # with the manifest first. Text conditions need no image, so they skip this intersection entirely.
    if with_image and visual_cases:
        manifest_keys = set(visual_cases.keys())
        before = len(cases)
        cases = cases[cases.apply(build_case_key, axis=1).isin(manifest_keys)].copy()
        print(f"[INFO] Intersect eligible cases with visual_cases manifest: {before} -> {len(cases)} have an image.")
        if len(cases) == 0:
            print("[WARN] No eligible case appears in visual_cases. Check that manifest case_keys "
                  "(runN__sampleID__pollutant) overlap with the eligibility selection / eligible_col.")
    cases = select_cases_for_reporting(cases, args)
    print("[INFO] Selected cases by pollutant/run:")
    print(cases.groupby(["pollutant", "run"]).size().to_string() if len(cases) else "[WARN] none selected")

    # ---- resume ----
    out_path = Path(args.output_json)
    out: Dict[str, Any] = {}
    if out_path.exists() and not args.overwrite:
        try:
            out = json.load(open(out_path, encoding="utf-8"))
        except Exception:
            out = {}

    rows = list(cases.iterrows())
    for idx, (_, erow) in enumerate(rows, start=1):
        run = int(erow["run"]); sid = str(erow["sample_id"]); pol = str(erow["pollutant"])
        key = build_case_key(erow)
        if (not args.overwrite) and key in out and reports_are_complete(out[key], args.n_repeats):
            print(f"[{idx}/{len(rows)}] {key}: skip (complete)")
            continue

        # ---- evidence (identical across all conditions) ----
        top_features = format_top_features(shap_feat, run, sid, pol, k=5)
        top_groups = format_top_groups(shap_group, run, sid, pol, k=3)
        reg_txt = registry_context(registry, top_features)
        query = (f"{pol} " + " ".join(f["feature_name"] for f in top_features)
                 + " " + " ".join(g["source_category"] for g in top_groups))
        rag_cards = retrieve_cards(rag_cards_all, pol, top_features, top_groups, lexical_query=query)
        rel_txt = ""  # reliability gating removed; build_prompt renders the slot as "Not provided".

        # ---- image (only for the image condition; text conditions carry no image) ----
        if with_image:
            image_records = extract_image_records(
                key, visual_cases.get(key), visual_audit.get(key),
                roots=roots, include_streetview=args.include_streetview,
            )
            if not image_records and args.require_images:
                raise FileNotFoundError(
                    f"No resolvable image for {key}. Check --project_root and visual_cases image paths."
                )
        else:
            image_records = []

        system, user = build_unified_prompt(
            mode, erow, top_features, top_groups, reg_txt, rag_cards, rel_txt, image_records, with_image
        )

        new_reports: List[str] = []
        structured_reports: List[Dict[str, Any]] = []
        visual_preds: List[Dict[str, Any]] = []
        raw_outputs: List[str] = []
        usage_list: List[Any] = []
        for _rep in range(args.n_repeats):
            if not args.openai_key:
                preview = ("[PROMPT_ONLY_NO_API_KEY]\n"
                           "SYSTEM:\n" + system + "\n\nUSER_TEXT:\n" + user + "\n\n"
                           "IMAGE_PATHS:\n" + "\n".join(r["path"] for r in image_records))
                new_reports.append(preview)
                structured_reports.append(_empty_structured(preview, "no_api_key_prompt_only"))
                visual_preds.append(normalize_visual_prediction({}))
            elif with_image and not image_records:
                # Only the image condition treats a missing crop as an error; text conditions never have one.
                err = "[API_ERROR] No resolvable image records for the image condition."
                new_reports.append(err)
                structured_reports.append(_empty_structured(err, "no_resolvable_image_records"))
                visual_preds.append(normalize_visual_prediction({}))
            else:
                try:
                    # image_records is [] for text conditions -> the call sends text only.
                    res = call_openai_vision_report(
                        system, user, image_records, args.model, args.openai_key,
                        args.temperature, args.max_tokens, args.image_detail,
                    )
                    raw = res["text"]; raw_outputs.append(raw); usage_list.append(res.get("usage"))
                    try:
                        structured = normalize_structured_report(raw)
                    except Exception as pe:
                        structured = _empty_structured(raw, repr(pe))
                    structured_reports.append(structured)
                    visual_preds.append(structured["visual_evidence_prediction"])
                    new_reports.append(structured.get("report", raw))
                except Exception as e:
                    err = f"[API_ERROR] {repr(e)}"
                    new_reports.append(err)
                    structured_reports.append(_empty_structured(err, repr(e)))
                    visual_preds.append(normalize_visual_prediction({}))
            if args.sleep_s:
                time.sleep(args.sleep_s)

        evidence_categories = sorted({g["source_category"] for g in top_groups if g["positive_shap_sum"] > 0})
        out[key] = {
            "condition": args.condition,                  # direct_llm | shap_only | shap_rag | shap_rag_image
            "with_image": with_image,
            "mode": "shap_rag_image_1step" if with_image else f"{mode}_text_v3",
            "base_mode": mode,
            "run": run, "sample_id": sid, "pollutant": pol,
            "eligibility": erow.to_dict(),
            "top_features": top_features,
            "top_groups": top_groups,
            "evidence_categories": evidence_categories,
            "retrieved_cards": rag_cards,
            "visual_image_records": image_records,        # [] for text conditions
            "visual_image_paths": [r["path"] for r in image_records],
            "system_prompt": system,
            "user_prompt": user,
            "reports": new_reports,                       # narrative only (back-compat)
            "structured_reports": structured_reports,     # full v3 objects (same schema for all conditions)
            "visual_evidence_predictions": visual_preds,  # per-repeat 4-source status/confidence
            "raw_model_outputs": raw_outputs,
            "report_usage": usage_list,
            "primary_source_hypothesis": (structured_reports[0].get("primary_source_hypothesis")
                                          if structured_reports else None),
            "context_modulators": (structured_reports[0].get("context") if structured_reports else None),
        }
        json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
        nerr = sum(1 for r in new_reports if isinstance(r, str) and r.startswith("[API_ERROR]"))
        print(f"[{idx}/{len(rows)}] {key}: images={len(image_records)}, reports={len(new_reports)}, api_errors={nerr}")

    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    print(f"[OK] Saved: {out_path} ({len(out)} cases)")


if __name__ == "__main__":
    main()
