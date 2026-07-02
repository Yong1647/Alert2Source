#!/usr/bin/env python3
"""
paper_11_visual_conditional_geval.py

Visual-conditional G-Eval for Alert2Source report-generation ablations.

This evaluator is intended for the main quantitative comparison among:
  1) Direct LLM
  2) SHAP only
  3) SHAP + Source-Knowledge RAG
  4) SHAP + Source-Knowledge RAG + Visual

Key design choice:
- The same four report-quality dimensions are used for all methods:
  correctness, completeness, relevance, and safety.
- Visual evidence is conditional. If no visual evidence card is supplied for a
  method/case, the report is NOT penalized for omitting visual discussion; it is
  penalized only if it invents visual claims. If a visual evidence card is
  supplied, the judge evaluates whether the report correctly uses visual status
  labels and separates SHAP-supported evidence from visually corroborated or
  crop-limited cues.

The script also computes lightweight structural metrics:
- Fidelity: mention rate of positive top-k SHAP features.
- Polarity Accuracy: direction preservation for mentioned SHAP features.
- USCR: unsupported source-claim rate, evidence-aware for visual reports.
- Entropy: generation diversity over detected source-category sets.

Typical usage:
python paper_11_visual_conditional_geval.py \
  --reports \
    direct=/path/to/reports_direct_llm_stratified.json \
    shap=/path/to/reports_shap_only_stratified.json \
    shap_rag=/path/to/reports_shap_rag_stratified.json \
    shap_rag_visual=/path/to/reports_shap_rag_visual_v2_stratified.json \
  --output_json /path/to/eval_visual_conditional_all_methods.json \
  --run_geval \
  --geval_model gpt-4o \
  --judge_scope all_reports

For a cheaper pilot:
python paper_11_visual_conditional_geval.py \
  --reports shap_rag_visual=reports_shap_rag_visual_v2_stratified.json \
  --output_json eval_pilot.json \
  --run_geval --max_cases 3 --judge_scope first_report
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCORE_LABELS = ["1", "2", "3", "4", "5"]

DIMENSION_WEIGHTS = {
    "correctness": 0.35,
    "completeness": 0.20,
    "relevance": 0.20,
    "safety": 0.25,
}

G_EVAL_DIMENSIONS: Dict[str, Dict[str, Any]] = {
    "correctness": {
        "title": "Evidence Correctness and Visual-Conditional Status Preservation",
        "description": (
            "Assess whether the report accurately reflects the provided evidence package. "
            "It should preserve the pollutant, alert context, important SHAP feature names, "
            "SHAP directions, and source-proxy categories. If a visual evidence card is provided, "
            "it should also preserve visual status labels such as present, visually_absent, "
            "not_visible_in_crop, uncertain, and not_visually_assessable. If no visual card is "
            "provided, do not penalize the report for omitting visual evidence; penalize only "
            "invented visual claims."
        ),
        "guidelines": {
            5: "Accurate and specific; preserves key SHAP/source-proxy evidence and, when available, visual statuses without contradiction.",
            4: "Mostly accurate; minor omissions or wording imprecision, but no important evidence is contradicted.",
            3: "Partially accurate; captures some evidence but omits or blurs an important SHAP/source/visual distinction.",
            2: "Weak; misstates important features, directions, source categories, or visual statuses.",
            1: "Incorrect; contradicts the evidence package or invents unsupported source/visual claims.",
        },
    },
    "completeness": {
        "title": "Report Completeness",
        "description": (
            "Assess whether the report covers the required operational elements: alert summary, "
            "SHAP-supported source-proxy evidence when available, source-proxy hypothesis, uncertainty, "
            "and recommended interpretation. If a visual evidence card is provided, the report should "
            "separate visually corroborated cues from crop-limited, uncertain, visually_absent, or "
            "not_visually_assessable cues. If no visual card is provided, do not penalize omission of "
            "visual sections."
        ),
        "guidelines": {
            5: "Complete; includes all relevant evidence blocks and limitations, with visual/non-visual sections handled appropriately.",
            4: "Mostly complete; one minor required element is abbreviated or less explicit.",
            3: "Moderately complete; several useful elements are present but at least one major component is missing.",
            2: "Incomplete; most evidence blocks or limitations are missing.",
            1: "Bare or uninformative; does not provide an operational root-cause report.",
        },
    },
    "relevance": {
        "title": "Case-Specific Relevance",
        "description": (
            "Assess whether the report is specific to this case rather than generic pollution advice. "
            "It should focus on source-proxy categories actually supported by the evidence. If visual "
            "evidence is provided, it should use visible spatial cues as corroborative context for "
            "SHAP-supported hypotheses, not as unrelated observations."
        ),
        "guidelines": {
            5: "Highly case-specific; source-proxy interpretation is tightly tied to the provided evidence.",
            4: "Mostly case-specific; small amount of generic wording but the main interpretation is evidence-focused.",
            3: "Somewhat relevant; mixes evidence-based interpretation with generic or weakly connected advice.",
            2: "Mostly generic; only shallow connection to the provided evidence.",
            1: "Irrelevant or generic; could apply to almost any pollution alert.",
        },
    },
    "safety": {
        "title": "Safety, Causal Calibration, and Source-Proxy Boundaries",
        "description": (
            "Assess whether the report avoids unsafe or overconfident claims. It must not claim "
            "interventional causality, chemical source apportionment, direct emissions measurement, "
            "or a specific emitter/vessel/source unless supplied. It should distinguish emission-source "
            "proxies from contextual conditions such as meteorology or geography. If visual evidence is "
            "provided, it should use crop-limitation language and should not treat non-visibility in a "
            "retrieved crop as absence in the broader environment. If no visual card is provided, it should "
            "not invent visual confirmation."
        ),
        "guidelines": {
            5: "Very safe; consistently uses source-proxy language, handles uncertainty and crop limits carefully.",
            4: "Mostly safe; minor overconfident phrase but no serious causal/source-apportionment overclaim.",
            3: "Acceptable but uneven; some calibration or limitation language is missing.",
            2: "Unsafe in important places; proxy variables or visual cues are treated as stronger evidence than warranted.",
            1: "Clearly unsafe; claims causal proof, chemical apportionment, specific emitters, or absence from non-visibility.",
        },
    },
}

SOURCE_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "traffic": [
        r"\btraffic\b", r"\broad\b", r"\broads\b", r"vehicle", r"transport", r"num_roads_1km", r"traffic-related",
    ],
    "urban_anthropogenic": [
        r"urban", r"anthropogenic", r"population", r"building", r"built[- ]environment", r"PopulationDensity", r"num_buildings_1km",
    ],
    "industrial": [
        r"industrial", r"industry", r"factory", r"factories", r"plant", r"plants",
    ],
    "port_shipping": [
        r"port", r"shipping", r"ship", r"vessel", r"maritime", r"AIS", r"ship_density",
    ],
    "meteorological_condition": [
        r"meteorolog", r"weather", r"temperature", r"Temp_3yr", r"stability", r"Stability_3yr", r"wind", r"accumulation", r"dispersion", r"formation",
    ],
    "geographic_context": [
        r"geographic", r"regional", r"rural", r"suburban", r"longitude", r"latitude", r"altitude", r"elevation", r"coast", r"waterway", r"river",
    ],
    "vegetation_open_area": [
        r"vegetation", r"green space", r"greenspace", r"open area", r"park", r"parks", r"forest",
    ],
}

VISUAL_WORDS = [
    r"visual", r"visually", r"image", r"imagery", r"satellite", r"crop", r"visible", r"not_visible_in_crop",
    r"visually_absent", r"not_visually_assessable", r"corroborated",
]

FORBIDDEN_PATTERNS = [
    r"\bprove[sd]?\b.*\bcaus", r"\bcaus(?:e|ed|al|ality)\b.*\bprove",
    r"directly caused", r"direct cause", r"true cause", r"chemical source apportionment",
    r"specific vessel", r"specific emitter", r"definitive source", r"definitively identify",
    r"not visible.*therefore.*absent", r"absence of visual.*implies.*absence",
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_by_key(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    out: Dict[str, Any] = {}
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


def real_reports(case: Dict[str, Any]) -> List[str]:
    vals = []
    for r in case.get("reports", []) or []:
        s = str(r)
        if s.startswith("[PROMPT_ONLY") or s.startswith("[API_ERROR]"):
            continue
        if not s.strip():
            continue
        vals.append(s)
    return vals


def compact_num(x: Any, ndigits: int = 4) -> Any:
    try:
        return round(float(x), ndigits)
    except Exception:
        return x


def normalize_status(st: Any) -> str:
    s = str(st or "").strip().lower()
    if s == "absent":
        return "visually_absent"
    if s in {"not visible", "not_visible", "not visible in crop"}:
        return "not_visible_in_crop"
    return s


def extract_visual_inspection(case_key: str, case: Dict[str, Any], external_visual: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    vis = case.get("visual_inspection")
    if isinstance(vis, dict) and vis:
        return vis
    ext = external_visual.get(case_key)
    if isinstance(ext, dict):
        vis2 = ext.get("visual_inspection") or ext.get("parsed")
        if isinstance(vis2, dict) and vis2:
            return vis2
    return None


def visual_present_categories(vis: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(vis, dict):
        return []
    cats = vis.get("visual_categories", {}) or {}
    out = []
    for cat, obj in cats.items():
        if isinstance(obj, dict) and normalize_status(obj.get("status")) == "present":
            out.append(str(cat))
    return sorted(set(out))


def compact_evidence_for_judge(method_name: str, case_key: str, case: Dict[str, Any], external_visual: Dict[str, Any]) -> str:
    elig = case.get("eligibility", {}) or {}
    top_features = []
    for f in case.get("top_features", [])[:8]:
        top_features.append({
            "feature": f.get("feature_name"),
            "value": compact_num(f.get("feature_value_raw")),
            "shap": compact_num(f.get("shap_value")),
            "direction": f.get("direction"),
            "source_category": f.get("source_category"),
        })
    top_groups = []
    for g in case.get("top_groups", [])[:6]:
        top_groups.append({
            "source_category": g.get("source_category"),
            "positive_shap_sum": compact_num(g.get("positive_shap_sum")),
            "signed_shap_sum": compact_num(g.get("signed_shap_sum")),
        })
    cards = []
    for c in case.get("retrieved_cards", [])[:8]:
        cards.append({
            "id": c.get("id"),
            "module": c.get("module"),
            "feature_name": c.get("feature_name"),
            "source_category": c.get("source_category"),
            "text": c.get("text"),
        })
    vis = extract_visual_inspection(case_key, case, external_visual)
    visual_payload: Optional[Dict[str, Any]] = None
    if isinstance(vis, dict):
        visual_payload = {
            "visual_categories": vis.get("visual_categories", {}),
            "shap_visual_alignment": vis.get("shap_visual_alignment", {}),
            "visual_summary": vis.get("visual_summary"),
        }
    payload = {
        "method_name": method_name,
        "case_key": case_key,
        "pollutant": case.get("pollutant"),
        "alert_context": {
            "observed_concentration": compact_num(elig.get("y_true")),
            "TabSatFusion_prediction": compact_num(elig.get("y_pred_fusion")),
            "LightGBM_diagnostic_prediction": compact_num(elig.get("y_pred_lgbm")),
            "alert_flag": elig.get("alert_flag"),
        },
        "reference_SHAP_source_proxy_evidence": top_features,
        "reference_source_group_evidence": top_groups,
        "evidence_categories": case.get("evidence_categories", []),
        "source_knowledge_cards_available_to_this_method": cards if cards else None,
        "visual_evidence_card_available_to_this_method": visual_payload,
        "conditional_visual_evaluation_rule": (
            "If visual_evidence_card_available_to_this_method is null, do not penalize the report for not mentioning visual evidence; "
            "only penalize invented visual claims. If it is not null, evaluate whether visual statuses and crop limitations are preserved."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_judge_prompt(dim: str, evidence: str, report: str) -> str:
    rubric = G_EVAL_DIMENSIONS[dim]
    guidelines = "\n".join(f"{score}: {txt}" for score, txt in sorted(rubric["guidelines"].items(), reverse=True))
    return f"""
Evaluate an air-quality source-proxy root-cause report.

Dimension: {rubric['title']}
Criterion: {rubric['description']}

Scoring rubric:
{guidelines}

Evidence package:
{evidence}

Report to evaluate:
{report}

Return exactly one digit from 1 to 5. Do not output any explanation.
""".strip()


def normalize_token(tok: str) -> Optional[str]:
    s = str(tok).strip()
    # OpenAI token may include a leading space. Keep only a bare score digit.
    m = re.search(r"[1-5]", s)
    if not m:
        return None
    # Avoid parsing multi-character labels like "10"; G-Eval prompt requests one digit.
    return m.group(0)


def parse_digit(text: str) -> Optional[str]:
    m = re.search(r"[1-5]", str(text))
    return m.group(0) if m else None


def softmax_from_logprobs(lp: Dict[str, float]) -> Dict[str, float]:
    mx = max(lp.values())
    exps = {k: math.exp(v - mx) for k, v in lp.items()}
    denom = sum(exps.values())
    return {k: v / denom for k, v in exps.items()}


def call_weighted_judge(prompt: str, model: str, api_key: str, top_logprobs: int, retries: int, sleep_s: float) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a strict evaluator. Return exactly one digit from 1,2,3,4,5. No explanation."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=top_logprobs,
            )
            choice = resp.choices[0]
            text = choice.message.content or ""
            hard = parse_digit(text)
            token_logprobs: Dict[str, float] = {}
            try:
                lp_obj = choice.logprobs.content[0]
                for item in lp_obj.top_logprobs:
                    lab = normalize_token(item.token)
                    if lab in SCORE_LABELS:
                        # If both "5" and " 5" appear, keep the better logprob.
                        token_logprobs[lab] = max(float(item.logprob), token_logprobs.get(lab, -1e9))
                chosen_lab = normalize_token(lp_obj.token)
                if chosen_lab in SCORE_LABELS:
                    token_logprobs[chosen_lab] = max(float(lp_obj.logprob), token_logprobs.get(chosen_lab, -1e9))
            except Exception:
                pass

            if token_logprobs:
                min_lp = min(token_logprobs.values())
                missing = []
                for lab in SCORE_LABELS:
                    if lab not in token_logprobs:
                        token_logprobs[lab] = min_lp - 20.0
                        missing.append(lab)
                probs = softmax_from_logprobs(token_logprobs)
                weighted = sum(int(s) * probs[s] for s in SCORE_LABELS)
                return {
                    "weighted_score": weighted,
                    "hard_score": int(hard) if hard else None,
                    "probabilities": probs,
                    "score_logprobs": token_logprobs,
                    "missing_score_tokens": missing,
                    "method": "weighted_logprobs" if not missing else "weighted_logprobs_with_missing_label_floor",
                    "judge_text": text,
                }

            if hard:
                return {
                    "weighted_score": float(hard),
                    "hard_score": int(hard),
                    "probabilities": None,
                    "score_logprobs": None,
                    "missing_score_tokens": SCORE_LABELS,
                    "method": "discrete_text_fallback_no_logprobs",
                    "judge_text": text,
                }
            return {"weighted_score": None, "method": "no_parseable_score", "judge_text": text}
        except Exception as e:
            last_err = repr(e)
            if attempt < retries:
                time.sleep(sleep_s * (2 ** attempt) + random.random() * 0.2)
    return {"weighted_score": None, "method": "api_error", "api_error": last_err}


def text_contains_feature(text: str, feature: str) -> bool:
    if not feature:
        return False
    # Exact feature names are usually included with underscores; use a case-insensitive literal search.
    return re.search(re.escape(str(feature)), text, flags=re.I) is not None


def classify_direction_in_window(text: str, feature: str, window_chars: int = 240) -> str:
    """Return increase/decrease/unknown for local wording around a feature mention."""
    if not feature:
        return "unknown"
    m = re.search(re.escape(str(feature)), text, flags=re.I)
    if not m:
        return "unknown"
    lo = max(0, m.start() - window_chars)
    hi = min(len(text), m.end() + window_chars)
    w = text[lo:hi].lower()
    inc_patterns = ["increases prediction", "increase prediction", "increased prediction", "positive shap", "shap=+", "shap value of +", "contributed positively", "positive contribution"]
    dec_patterns = ["decreases prediction", "decrease prediction", "decreased prediction", "negative shap", "shap=-", "shap value of -", "contributed negatively", "negative contribution"]
    if any(p in w for p in inc_patterns):
        return "increase"
    if any(p in w for p in dec_patterns):
        return "decrease"
    return "unknown"


def expected_direction(direction_text: Any, shap_value: Any = None) -> str:
    s = str(direction_text or "").lower()
    if "increase" in s:
        return "increase"
    if "decrease" in s:
        return "decrease"
    try:
        return "increase" if float(shap_value) >= 0 else "decrease"
    except Exception:
        return "unknown"


def positive_top_features(case: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
    feats = []
    for f in case.get("top_features", []) or []:
        try:
            if float(f.get("shap_value", 0)) > 0:
                feats.append(f)
        except Exception:
            continue
    return feats[:top_k]


def compute_fidelity(case: Dict[str, Any], reports: List[str], top_k: int) -> Dict[str, Any]:
    feats = positive_top_features(case, top_k=top_k)
    if not feats:
        return {"fidelity_mean": None, "fidelity_values": [], "expected_features": []}
    values = []
    for r in reports:
        n = sum(1 for f in feats if text_contains_feature(r, str(f.get("feature_name"))))
        values.append(n / len(feats))
    return {
        "fidelity_mean": mean(values),
        "fidelity_values": values,
        "expected_features": [f.get("feature_name") for f in feats],
    }


def compute_polarity(case: Dict[str, Any], reports: List[str], top_k: int) -> Dict[str, Any]:
    feats = positive_top_features(case, top_k=top_k)
    checks = []
    for ridx, r in enumerate(reports):
        for f in feats:
            name = str(f.get("feature_name"))
            if not text_contains_feature(r, name):
                continue
            exp = expected_direction(f.get("direction"), f.get("shap_value"))
            found = classify_direction_in_window(r, name)
            # If the report mentions the feature but does not explicitly repeat polarity, do not count as wrong;
            # the correctness dimension will judge missing detail. This matches a conservative text metric.
            correct = (found == exp) or (found == "unknown")
            checks.append({
                "report_idx": ridx,
                "feature": name,
                "expected": exp,
                "found": found,
                "correct": bool(correct),
            })
    if not checks:
        return {"polarity_accuracy": None, "n_checks": 0, "checks": []}
    return {
        "polarity_accuracy": sum(1 for c in checks if c["correct"]) / len(checks),
        "n_checks": len(checks),
        "checks": checks,
    }


def detect_categories(text: str) -> List[str]:
    found = []
    for cat, pats in SOURCE_CATEGORY_KEYWORDS.items():
        for pat in pats:
            if re.search(pat, text, flags=re.I):
                found.append(cat)
                break
    return sorted(set(found))


def detect_visual_claim(text: str) -> bool:
    return any(re.search(p, text, flags=re.I) for p in VISUAL_WORDS)


def detect_forbidden_patterns(text: str) -> List[str]:
    out = []
    for p in FORBIDDEN_PATTERNS:
        if re.search(p, text, flags=re.I | re.S):
            out.append(p)
    return out


def compute_uscr(case_key: str, case: Dict[str, Any], reports: List[str], external_visual: Dict[str, Any]) -> Dict[str, Any]:
    shap_allowed = set(str(c) for c in (case.get("evidence_categories", []) or []))
    vis = extract_visual_inspection(case_key, case, external_visual)
    visual_allowed = set(visual_present_categories(vis))
    all_visual_categories = set()
    if isinstance(vis, dict):
        all_visual_categories = set(str(c) for c in ((vis.get("visual_categories", {}) or {}).keys()))

    evidence_allowed = set(shap_allowed)
    if vis is not None:
        # In the visual condition, reports are expected to mention not-visible, uncertain,
        # and not-assessable categories as limitations. Therefore the simple USCR metric
        # treats every category in the supplied visual card as mention-supported.
        # The G-Eval safety dimension judges whether such categories are improperly
        # promoted to causal/source claims.
        evidence_allowed.update(all_visual_categories)

    unsupported_count = 0
    total_claims = 0
    details = []
    for ridx, r in enumerate(reports):
        claimed = set(detect_categories(r))
        # Do not penalize a report for generic visual/non-visual vocabulary alone.
        unsupported = sorted(c for c in claimed if c not in evidence_allowed)
        forb = detect_forbidden_patterns(r)
        total_claims += len(claimed)
        unsupported_count += len(unsupported)
        details.append({
            "report_idx": ridx,
            "claimed_categories": sorted(claimed),
            "shap_evidence_categories": sorted(shap_allowed),
            "visual_present_categories": sorted(visual_allowed),
            "visual_card_categories": sorted(all_visual_categories),
            "evidence_allowed_categories": sorted(evidence_allowed),
            "unsupported_categories": unsupported,
            "forbidden_patterns": forb,
            "invented_visual_claim_without_visual_card": bool(vis is None and detect_visual_claim(r)),
        })
    rate = unsupported_count / total_claims if total_claims else 0.0
    return {
        "unsupported_source_claim_rate": rate,
        "total_claims": total_claims,
        "unsupported_count": unsupported_count,
        "details": details,
    }


def compute_entropy(reports: List[str]) -> Dict[str, Any]:
    if not reports:
        return {"entropy_raw": None, "entropy_normalized": None, "n_unique_sets": 0}
    keys = [tuple(detect_categories(r)) for r in reports]
    counts: Dict[Tuple[str, ...], int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    n = len(keys)
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * math.log(p)
    max_ent = math.log(n) if n > 1 else 0.0
    return {
        "entropy_raw": ent,
        "entropy_normalized": ent / max_ent if max_ent > 0 else 0.0,
        "n_unique_sets": len(counts),
        "sets": {"|".join(k) if k else "<none>": v for k, v in counts.items()},
    }


def compute_visual_status_sanity(case_key: str, case: Dict[str, Any], reports: List[str], external_visual: Dict[str, Any]) -> Dict[str, Any]:
    vis = extract_visual_inspection(case_key, case, external_visual)
    if vis is None:
        return {"visual_card_available": False}
    cats = vis.get("visual_categories", {}) or {}
    present = []
    non_present = []
    not_assess = []
    for cat, obj in cats.items():
        if not isinstance(obj, dict):
            continue
        st = normalize_status(obj.get("status"))
        if st == "present":
            present.append(str(cat))
        elif st == "not_visually_assessable":
            not_assess.append(str(cat))
        else:
            non_present.append(str(cat))

    rows = []
    for ridx, r in enumerate(reports):
        rlow = r.lower()
        mentioned_present = [c for c in present if re.search(re.escape(c), r, flags=re.I)]
        # crude but useful: non-present category named near "visually corroborated/confirmed/present" is an overclaim
        overclaims = []
        for c in non_present + not_assess:
            for m in re.finditer(re.escape(c), r, flags=re.I):
                window = rlow[max(0, m.start() - 140): min(len(rlow), m.end() + 140)]
                if any(w in window for w in ["visually corroborated", "visually confirmed", "status=present", "status = present", "visible source"]):
                    overclaims.append(c)
                    break
        rows.append({
            "report_idx": ridx,
            "present_categories": sorted(present),
            "mentioned_present_categories": sorted(mentioned_present),
            "non_present_or_not_assessable_overclaims": sorted(set(overclaims)),
        })
    return {
        "visual_card_available": True,
        "present_categories": sorted(present),
        "non_present_categories": sorted(non_present),
        "not_visually_assessable_categories": sorted(not_assess),
        "per_report": rows,
    }


def mean(vals: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None]
    return sum(xs) / len(xs) if xs else None


def std(vals: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None]
    if not xs:
        return None
    if len(xs) == 1:
        return 0.0
    return float(statistics.pstdev(xs))


def parse_reports_args(items: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--reports item must be NAME=PATH, got: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty method name in --reports item: {item}")
        out[name] = Path(path)
    return out


def existing_result_lookup(out_obj: Dict[str, Any], method_name: str, case_key: str, report_idx: int, dim: str) -> Optional[Dict[str, Any]]:
    try:
        return out_obj["methods"][method_name]["per_case"][case_key]["report_evals"][str(report_idx)]["geval"][dim]
    except Exception:
        return None


def summarize_method(method_obj: Dict[str, Any]) -> Dict[str, Any]:
    # Structural metrics are case-level means.
    fidelity_vals = []
    polarity_vals = []
    uscr_vals = []
    entropy_vals = []
    for row in method_obj.get("per_case", {}).values():
        fid = row.get("fidelity", {}).get("fidelity_mean")
        pol = row.get("polarity", {}).get("polarity_accuracy")
        uscr = row.get("uscr", {}).get("unsupported_source_claim_rate")
        ent = row.get("entropy", {}).get("entropy_normalized")
        if fid is not None:
            fidelity_vals.append(fid)
        if pol is not None:
            polarity_vals.append(pol)
        if uscr is not None:
            uscr_vals.append(uscr)
        if ent is not None:
            entropy_vals.append(ent)

    dim_scores = {d: [] for d in G_EVAL_DIMENSIONS}
    final_scores = []
    judged_reports = 0
    for row in method_obj.get("per_case", {}).values():
        for rep_eval in row.get("report_evals", {}).values():
            geval = rep_eval.get("geval", {})
            got_any = False
            for d in G_EVAL_DIMENSIONS:
                sc = geval.get(d, {}).get("weighted_score")
                if sc is not None:
                    dim_scores[d].append(float(sc))
                    got_any = True
            fsc = geval.get("final", {}).get("weighted_score")
            if fsc is not None:
                final_scores.append(float(fsc))
            if got_any:
                judged_reports += 1

    summary: Dict[str, Any] = {
        "structural_metrics": {
            "fidelity": {"mean": mean(fidelity_vals), "std": std(fidelity_vals), "n": len(fidelity_vals)},
            "polarity_accuracy": {"mean": mean(polarity_vals), "std": std(polarity_vals), "n": len(polarity_vals)},
            "uscr": {"mean": mean(uscr_vals), "std": std(uscr_vals), "n": len(uscr_vals)},
            "entropy_normalized": {"mean": mean(entropy_vals), "std": std(entropy_vals), "n": len(entropy_vals)},
        },
        "geval": {
            "judged_reports": judged_reports,
            "dimension_weights_for_final": DIMENSION_WEIGHTS,
        },
    }
    for d in G_EVAL_DIMENSIONS:
        summary["geval"][d] = {"mean": mean(dim_scores[d]), "std": std(dim_scores[d]), "n": len(dim_scores[d])}
    summary["geval"]["final"] = {"mean": mean(final_scores), "std": std(final_scores), "n": len(final_scores)}
    return summary


def make_latex_table(summary: Dict[str, Any]) -> str:
    rows = []
    method_order = list(summary.get("methods", {}).keys())
    header = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\caption{Visual-conditional report-generation evaluation. Visual evidence is conditional: non-visual methods are not penalized for omitting visual discussion, while the visual method is evaluated for preserving visual status and crop-limit language.}\n"
        "\\label{tab:report_visual_conditional}\n"
        "\\begin{tabular}{lcccccccc}\n"
        "\\toprule\n"
        "Method & Fidelity$\\uparrow$ & Polarity$\\uparrow$ & USCR$\\downarrow$ & Correct.$\\uparrow$ & Complete$\\uparrow$ & Relevant$\\uparrow$ & Safety$\\uparrow$ & Final$\\uparrow$ \\\\\n"
        "\\midrule\n"
    )
    for m in method_order:
        s = summary["methods"][m]["summary"]
        st = s["structural_metrics"]
        gv = s["geval"]
        def fmt(x: Any) -> str:
            return "--" if x is None else f"{float(x):.3f}"
        rows.append(
            f"{m} & {fmt(st['fidelity']['mean'])} & {fmt(st['polarity_accuracy']['mean'])} & {fmt(st['uscr']['mean'])} & "
            f"{fmt(gv['correctness']['mean'])} & {fmt(gv['completeness']['mean'])} & {fmt(gv['relevance']['mean'])} & "
            f"{fmt(gv['safety']['mean'])} & {fmt(gv['final']['mean'])} \\\\"
        )
    footer = "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    return header + "\n".join(rows) + footer


def main() -> None:
    ap = argparse.ArgumentParser(description="Visual-conditional G-Eval for Alert2Source report ablations.")
    ap.add_argument("--reports", nargs="+", required=True, help="Named report JSONs: method_name=/path/to/reports.json")
    ap.add_argument("--visual_inspection_jsonl", default="", help="Optional external visual_inspection_v2.jsonl for methods without embedded visual_inspection.")
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--openai_key", default=os.getenv("OPENAI_API_KEY", ""))
    ap.add_argument("--run_geval", action="store_true")
    ap.add_argument("--geval_model", default="gpt-4o")
    ap.add_argument("--top_logprobs", type=int, default=20)
    ap.add_argument("--judge_scope", choices=["first_report", "all_reports"], default="all_reports")
    ap.add_argument("--max_cases", type=int, default=0, help="0 means all cases")
    ap.add_argument("--top_k_fidelity", type=int, default=3)
    ap.add_argument("--resume", action="store_true", help="Reuse existing scores in output_json when present.")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--sleep_s", type=float, default=0.5)
    args = ap.parse_args()

    if args.run_geval and not args.openai_key:
        raise RuntimeError("OPENAI_API_KEY is required for --run_geval")

    report_paths = parse_reports_args(args.reports)
    external_visual = load_jsonl_by_key(Path(args.visual_inspection_jsonl)) if args.visual_inspection_jsonl else {}
    out_path = Path(args.output_json)
    previous: Dict[str, Any] = {}
    if args.resume and out_path.exists():
        try:
            previous = load_json(out_path)
            print(f"[INFO] Loaded existing output for resume: {out_path}")
        except Exception as e:
            print(f"[WARN] Could not load existing output for resume: {e}", file=sys.stderr)

    out: Dict[str, Any] = {
        "metadata": {
            "rubric_version": "visual_conditional_rootcause_v1",
            "judge_model": args.geval_model,
            "judge_scope": args.judge_scope,
            "dimension_weights_for_final": DIMENSION_WEIGHTS,
            "visual_condition_rule": (
                "No visual card: do not penalize visual omission, penalize invented visual claims. "
                "Visual card present: evaluate visual-status preservation, crop-limit language, and SHAP/visual separation."
            ),
        },
        "methods": {},
    }

    for method_name, path in report_paths.items():
        data = load_json(path)
        items = list(data.items())
        if args.max_cases and args.max_cases > 0:
            items = items[: args.max_cases]
        method_obj: Dict[str, Any] = {
            "input_path": str(path),
            "n_cases": len(items),
            "per_case": {},
        }
        print(f"[INFO] Method={method_name}: {len(items)} cases from {path}")

        for case_idx, (case_key, case) in enumerate(items, start=1):
            reports = real_reports(case)
            reports_to_judge = reports[:1] if args.judge_scope == "first_report" else reports
            case_row: Dict[str, Any] = {
                "n_real_reports": len(reports),
                "n_judged_reports_requested": len(reports_to_judge),
                "visual_card_available": extract_visual_inspection(case_key, case, external_visual) is not None,
                "fidelity": compute_fidelity(case, reports, args.top_k_fidelity),
                "polarity": compute_polarity(case, reports, args.top_k_fidelity),
                "uscr": compute_uscr(case_key, case, reports, external_visual),
                "entropy": compute_entropy(reports),
                "visual_status_sanity": compute_visual_status_sanity(case_key, case, reports, external_visual),
                "report_evals": {},
            }

            evidence = compact_evidence_for_judge(method_name, case_key, case, external_visual)
            for ridx, report in enumerate(reports_to_judge):
                rep_eval: Dict[str, Any] = {"geval": {}}
                dim_scores: Dict[str, float] = {}
                for dim in G_EVAL_DIMENSIONS:
                    reused = existing_result_lookup(previous, method_name, case_key, ridx, dim) if args.resume else None
                    if reused is not None and reused.get("weighted_score") is not None:
                        res = reused
                    elif args.run_geval:
                        prompt = build_judge_prompt(dim, evidence, report)
                        res = call_weighted_judge(prompt, args.geval_model, args.openai_key, args.top_logprobs, args.retries, args.sleep_s)
                        time.sleep(args.sleep_s)
                    else:
                        res = {"weighted_score": None, "method": "not_run"}
                    rep_eval["geval"][dim] = res
                    if res.get("weighted_score") is not None:
                        dim_scores[dim] = float(res["weighted_score"])
                if all(d in dim_scores for d in DIMENSION_WEIGHTS):
                    final = sum(DIMENSION_WEIGHTS[d] * dim_scores[d] for d in DIMENSION_WEIGHTS)
                    rep_eval["geval"]["final"] = {
                        "weighted_score": final,
                        "weights": DIMENSION_WEIGHTS,
                        "method": "weighted_sum_of_dimension_scores",
                    }
                else:
                    rep_eval["geval"]["final"] = {"weighted_score": None, "method": "not_available"}
                case_row["report_evals"][str(ridx)] = rep_eval
            method_obj["per_case"][case_key] = case_row
            print(f"[{method_name} {case_idx}/{len(items)}] {case_key}: structural + {len(reports_to_judge)} report G-Eval records")

            # Periodic checkpoint for long API runs.
            out["methods"][method_name] = method_obj
            method_obj["summary"] = summarize_method(method_obj)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)

        method_obj["summary"] = summarize_method(method_obj)
        out["methods"][method_name] = method_obj

    out["latex_table"] = make_latex_table(out)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n[SUMMARY]")
    for m, obj in out["methods"].items():
        s = obj["summary"]
        final = s["geval"]["final"]["mean"]
        print(f"- {m}: final={final}, judged_reports={s['geval']['judged_reports']}")
    print(f"[OK] Saved: {out_path}")
    print("\n[LaTeX table]\n" + out["latex_table"])


if __name__ == "__main__":
    main()
