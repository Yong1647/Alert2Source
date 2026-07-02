#!/usr/bin/env python3
"""
paper_00_build_source_registry.py
==================================
Builds the deterministic Source Registry and the lightweight RAG knowledge-base
used by the SHAP--RAG root-cause reporting experiments.

Why this file exists
--------------------
The final paper direction uses LightGBM as an explanation-friendly diagnostic
explainer and Tree SHAP as source-proxy evidence. The LLM must not freely infer
root causes. It should verbalize source candidates selected by SHAP and grounded
by a deterministic feature-to-source registry plus pollutant/domain rules.

Outputs
-------
- source_registry.csv: exact feature -> source category / allowed language rules
- air_quality_rag_database.jsonl: domain cards for simple RAG retrieval

Usage
-----
python paper_00_build_source_registry.py --output_dir paper_outputs/kb
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

import pandas as pd


@dataclass
class SourceRegistryEntry:
    feature_name: str
    source_category: str
    source_type: str
    pollutant_relevance: str
    allowed_interpretation: str
    forbidden_claim: str
    report_phrase: str


def build_source_registry() -> List[SourceRegistryEntry]:
    """Feature-to-source mapping aligned with europe_23feat.yaml."""
    rows = [
        SourceRegistryEntry(
            "num_roads_500m", "traffic", "emission_proxy", "NO2, PM10",
            "Road density within 500 m supports a local traffic-related source-proxy hypothesis.",
            "Do not claim that road density proves traffic caused the alert.",
            "traffic-related source proxies",
        ),
        SourceRegistryEntry(
            "num_roads_1km", "traffic", "emission_proxy", "NO2, PM10",
            "Road density within 1 km supports a traffic-related source-proxy hypothesis.",
            "Do not claim that road density directly measures vehicle emissions or proves causality.",
            "traffic-related source proxies",
        ),
        SourceRegistryEntry(
            "traffic", "traffic", "station_type_proxy", "NO2, PM10",
            "A traffic station indicator supports a traffic-exposure interpretation.",
            "Do not claim that the station type alone identifies the true emission source.",
            "traffic-exposure context",
        ),
        SourceRegistryEntry(
            "PopulationDensity", "urban_anthropogenic", "activity_proxy", "NO2, O3, PM10",
            "Population density supports an urban anthropogenic activity hypothesis.",
            "Do not claim that population directly caused pollution.",
            "urban anthropogenic activity proxies",
        ),
        SourceRegistryEntry(
            "urban", "urban_anthropogenic", "station_type_proxy", "NO2, O3, PM10",
            "An urban station indicator supports a dense built-environment interpretation.",
            "Do not claim that the urban label alone proves a source.",
            "urban context",
        ),
        SourceRegistryEntry(
            "num_buildings_500m", "urban_anthropogenic", "activity_proxy", "NO2, PM10",
            "Building count within 500 m supports local urban-density evidence.",
            "Do not claim that buildings directly emit pollutants.",
            "local built-environment proxies",
        ),
        SourceRegistryEntry(
            "num_buildings_1km", "urban_anthropogenic", "activity_proxy", "NO2, PM10",
            "Building count within 1 km supports urban-density evidence.",
            "Do not claim that buildings directly emit pollutants.",
            "built-environment proxies",
        ),
        SourceRegistryEntry(
            "num_factory_500m", "industrial", "emission_proxy", "NO2, PM10",
            "Factory count within 500 m supports a local industrial activity hypothesis.",
            "Do not identify a specific facility as the cause without external evidence.",
            "local industrial activity proxies",
        ),
        SourceRegistryEntry(
            "num_factory_1km", "industrial", "emission_proxy", "NO2, PM10",
            "Factory count within 1 km supports an industrial activity hypothesis.",
            "Do not identify a specific facility as the cause without external evidence.",
            "industrial activity proxies",
        ),
        SourceRegistryEntry(
            "num_industrial_landuse_500m", "industrial", "land_use_proxy", "NO2, PM10",
            "Industrial land use within 500 m supports a local industrial source-proxy hypothesis.",
            "Do not claim that industrial land use proves actual emissions from a facility.",
            "local industrial land-use proxies",
        ),
        SourceRegistryEntry(
            "num_industrial_landuse_1km", "industrial", "land_use_proxy", "NO2, PM10",
            "Industrial land use within 1 km supports an industrial source-proxy hypothesis.",
            "Do not claim that industrial land use proves actual emissions from a facility.",
            "industrial land-use proxies",
        ),
        SourceRegistryEntry(
            "ship_density_1km", "port_shipping", "maritime_activity_proxy", "NO2, PM10",
            "AIS-derived ship density within 1 km supports a local port/shipping-related source-proxy hypothesis.",
            "Do not claim that a specific vessel caused the alert.",
            "local port/shipping activity proxies",
        ),
        SourceRegistryEntry(
            "ship_density_5km", "port_shipping", "maritime_activity_proxy", "NO2, PM10",
            "AIS-derived ship density within 5 km supports a port/shipping-related source-proxy hypothesis.",
            "Do not claim that a specific vessel caused the alert.",
            "port/shipping activity proxies",
        ),
        SourceRegistryEntry(
            "ship_density_10km", "port_shipping", "maritime_activity_proxy", "NO2, PM10",
            "AIS-derived ship density within 10 km supports a regional maritime activity hypothesis.",
            "Do not claim that shipping is the only source without corroborating evidence.",
            "regional port/shipping activity proxies",
        ),
        SourceRegistryEntry(
            "Temp_3yr", "meteorological_condition", "formation_condition", "O3, PM10",
            "Temperature may support photochemical formation or accumulation conditions, especially for O3.",
            "Do not describe temperature as an emission source.",
            "temperature-related formation condition",
        ),
        SourceRegistryEntry(
            "Wind_3yr", "meteorological_condition", "dispersion_condition", "NO2, O3, PM10",
            "Wind speed may affect dispersion, dilution, or transport conditions.",
            "Do not describe wind as an emission source.",
            "wind-related dispersion condition",
        ),
        SourceRegistryEntry(
            "Precip_3yr", "meteorological_condition", "removal_condition", "PM10",
            "Precipitation may affect wet deposition and pollutant washout.",
            "Do not describe precipitation as an emission source.",
            "precipitation-related washout condition",
        ),
        SourceRegistryEntry(
            "RH_3yr", "meteorological_condition", "formation_condition", "PM10, O3",
            "Relative humidity may influence secondary aerosol formation and atmospheric chemistry conditions.",
            "Do not describe humidity as an emission source.",
            "humidity-related atmospheric condition",
        ),
        SourceRegistryEntry(
            "Stability_3yr", "meteorological_condition", "accumulation_condition", "NO2, PM10",
            "Atmospheric stability may support pollutant trapping and near-surface accumulation.",
            "Do not describe stability as an emission source.",
            "stability-related accumulation condition",
        ),
        SourceRegistryEntry(
            "Altitude", "geographic_context", "context_variable", "O3, PM10",
            "Altitude provides elevation context that can affect dispersion, mixing, and regional pollutant patterns.",
            "Do not describe altitude as an emission source.",
            "elevation context",
        ),
        SourceRegistryEntry(
            "Latitude", "geographic_context", "context_variable", "NO2, O3, PM10",
            "Latitude provides broad regional or climatic context.",
            "Do not describe latitude as a pollution source.",
            "regional geographic context",
        ),
        SourceRegistryEntry(
            "Longitude", "geographic_context", "context_variable", "NO2, O3, PM10",
            "Longitude provides broad regional context.",
            "Do not describe longitude as a pollution source.",
            "regional geographic context",
        ),
        SourceRegistryEntry(
            "rural", "geographic_context", "station_type_proxy", "NO2, O3, PM10",
            "A rural station indicator provides station-context information and may indicate lower local anthropogenic exposure.",
            "Do not describe the rural label as a direct cause.",
            "rural station context",
        ),
    ]
    return rows


def build_rag_cards(registry: List[SourceRegistryEntry]) -> List[Dict[str, object]]:
    """Build JSONL-ready cards. The registry itself is deterministic lookup; these cards are retrieval context."""
    cards: List[Dict[str, object]] = []

    # Source registry cards per feature
    for entry in registry:
        cards.append({
            "id": f"feature.{entry.feature_name}",
            "module": "source_proxy_registry",
            "feature_name": entry.feature_name,
            "source_category": entry.source_category,
            "source_type": entry.source_type,
            "pollutant_relevance": entry.pollutant_relevance,
            "text": (
                f"Feature {entry.feature_name} maps to source category {entry.source_category}. "
                f"Allowed interpretation: {entry.allowed_interpretation} "
                f"Report phrase: {entry.report_phrase}. Forbidden claim: {entry.forbidden_claim}"
            ),
        })

    # Pollutant mechanism cards
    cards.extend([
        {
            "id": "pollutant.no2",
            "module": "pollutant_mechanism",
            "pollutant": "no2",
            "source_category": "traffic industrial port_shipping urban_anthropogenic",
            "text": (
                "NO2 is commonly associated with combustion-related source proxies such as road traffic, "
                "industrial combustion, dense urban activity, and shipping or port activity. Reports must describe "
                "these as source-proxy hypotheses, not as causal proof."
            ),
        },
        {
            "id": "pollutant.o3",
            "module": "pollutant_mechanism",
            "pollutant": "o3",
            "source_category": "meteorological_condition regional_context precursor_related_context",
            "text": (
                "O3 is a secondary pollutant formed through photochemical reactions. Temperature, radiation, wind, "
                "humidity, and stability may support formation, dispersion, or accumulation conditions, but they are "
                "not direct emission sources. Avoid saying that temperature or wind emitted ozone."
            ),
        },
        {
            "id": "pollutant.pm10",
            "module": "pollutant_mechanism",
            "pollutant": "pm10",
            "source_category": "traffic industrial port_shipping urban_anthropogenic meteorological_condition",
            "text": (
                "PM10 can be associated with traffic-related resuspension, industrial activity, construction or dense "
                "urban activity proxies, maritime activity near ports, and meteorological removal or accumulation conditions. "
                "Reports should distinguish emission proxies from dispersion or washout conditions."
            ),
        },
    ])

    # Attribution and reliability rules
    cards.extend([
        {
            "id": "rule.shap_polarity",
            "module": "attribution_rule",
            "text": (
                "Positive SHAP means the feature increases the diagnostic explainer prediction relative to the background. "
                "Negative SHAP means the feature decreases the prediction. Negative SHAP features must not be described as "
                "causes of an elevated alert. Use exact language: INCREASES prediction or DECREASES prediction."
            ),
        },
        {
            "id": "rule.root_cause_scope",
            "module": "report_guardrail",
            "text": (
                "The framework performs operational source-proxy root-cause diagnosis. It does not prove interventional "
                "causality and does not replace chemical source apportionment. Use source hypothesis, source-proxy evidence, "
                "and operational diagnosis language."
            ),
        },
        {
            "id": "rule.reliability_gate",
            "module": "reliability_rule",
            "text": (
                "Use high-confidence source-hypothesis wording only when the AQFusionNet prediction reliability and the "
                "LightGBM diagnostic explainer adequacy gates pass. If either gate fails, use uncertainty language and state "
                "that the source-proxy evidence is insufficient for confident diagnosis."
            ),
        },
        {
            "id": "template.root_cause_report",
            "module": "report_template",
            "text": (
                "Required report sections: Alert summary; prediction and diagnostic-explainer reliability; top SHAP source-proxy "
                "evidence; root-cause source hypothesis; uncertainty and limitations; recommended interpretation. Do not mention "
                "source categories absent from SHAP/source-registry evidence."
            ),
        },
        {
            "id": "provenance.ship_density",
            "module": "data_provenance",
            "source_category": "port_shipping",
            "text": (
                "ship_density_1km, ship_density_5km, and ship_density_10km are AIS-derived maritime traffic density proxies. "
                "For each monitoring station, grid cells within radius r in {1, 5, 10} km are selected, and ship density is "
                "computed as the average ship residence time over those cells. Positive attribution supports a port/shipping "
                "source-proxy hypothesis but does not identify a specific vessel."
            ),
        },
    ])
    return cards


def write_jsonl(path: str, records: Iterable[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="paper_outputs/kb")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    registry = build_source_registry()
    registry_df = pd.DataFrame([asdict(r) for r in registry])
    registry_path = os.path.join(args.output_dir, "source_registry.csv")
    registry_df.to_csv(registry_path, index=False)

    cards = build_rag_cards(registry)
    jsonl_path = os.path.join(args.output_dir, "air_quality_rag_database.jsonl")
    write_jsonl(jsonl_path, cards)

    print(f"[OK] Source registry: {registry_path} ({len(registry_df)} rows)")
    print(f"[OK] RAG database: {jsonl_path} ({len(cards)} cards)")


if __name__ == "__main__":
    main()
