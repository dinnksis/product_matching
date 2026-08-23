"""Run the versioned Qwen semantic-extraction pilot without label leakage.

Raw responses are append-only and resumable. Human labels are loaded only after
all API calls have completed, then joined locally by pair_id.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import re
import time
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import jsonschema
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "qwen_semantic_pilot_v1" / "pilot_inputs.parquet"
DEFAULT_LABELS = ROOT / "data" / "qwen_semantic_pilot_v1" / "pilot_labels.parquet"
DEFAULT_METADATA = (
    ROOT / "data" / "qwen_semantic_pilot_v1" / "pilot_sampling_metadata.parquet"
)
DEFAULT_PROMPT = ROOT / "prompts" / "qwen_semantic_extraction_v1_3.md"
DEFAULT_SCHEMA = ROOT / "schemas" / "qwen_semantic_extraction_v1_3.schema.json"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "qwen_semantic_extraction_v1_3_smoke50"
DEFAULT_MODEL = "Qwen3.5-397B-A17B-FP8"
DEFAULT_API_BASE = "http://localhost:8194/v1"
FORBIDDEN_INPUT_FIELDS = {"target", "label", "human_label", "is_match", "match_label"}
FORBIDDEN_OUTPUT_KEYS = {"target", "label", "human_label", "is_match", "match_score", "verdict", "decision"}
MATCH_JUDGEMENT_RE = re.compile(
    r"\b(?:non[- ]?match|is a match|same product|different product)\b|"
    r"(?:являются|не являются)\s+(?:одним|одинаковыми|разными)\s+товар",
    re.I,
)
GENERIC_CONCEPTS = {
    "attribute", "attributes", "feature", "features", "other", "specification",
    "specifications", "product", "unknown", "value",
}
ANCHOR_STRENGTH = {
    "exact_sku": "strong",
    "manufacturer_part_number": "strong",
    "exact_model": "strong",
    "model_family": "medium",
    "product_line": "medium",
    "brand": "weak",
    "other_identity": "medium",
}
ANCHOR_CONCEPTS = {
    "exact_sku": {"sku", "exact_sku"},
    "manufacturer_part_number": {
        "manufacturer_part_number", "part_number", "mpn", "base_model_number",
    },
    "exact_model": {"model_number", "model_name", "exact_model"},
    "model_family": {"model_family"},
    "product_line": {"product_line", "series", "collection"},
    "brand": {"brand"},
}
ABSENCE_MARKERS = {
    "unknown", "unspecified", "not specified", "not provided", "no data",
    "no brand", "unbranded", "none", "n a", "не указано", "не указан",
    "не указана", "неизвестно", "нет данных", "нет бренда", "без бренда",
    "не определен", "не определён", "уточнить у продавца",
}
HARD_COUNTRY_AS_BRAND_VALUES = {
    "кнр", "prc", "cn", "китайская народная республика",
}
QUANTITY_CONCEPTS = {
    "package_quantity", "quantity", "item_count", "piece_count", "unit_count",
    "number_of_items",
}
MEASUREMENT_ATTRIBUTE_TOKENS = {
    "вес", "масса", "объем", "объём", "длина", "ширина", "высота",
    "weight", "mass", "volume", "length", "width", "height",
}
COUNT_ATTRIBUTE_TOKENS = {
    "количество", "комплект", "набор", "штук", "шт", "quantity", "count",
    "pieces", "pcs", "pack",
}
UNIT_TO_BASE = {
    "ml": ("volume_ml", 1.0),
    "мл": ("volume_ml", 1.0),
    "milliliter": ("volume_ml", 1.0),
    "milliliters": ("volume_ml", 1.0),
    "l": ("volume_ml", 1000.0),
    "л": ("volume_ml", 1000.0),
    "liter": ("volume_ml", 1000.0),
    "liters": ("volume_ml", 1000.0),
    "g": ("mass_g", 1.0),
    "г": ("mass_g", 1.0),
    "gram": ("mass_g", 1.0),
    "grams": ("mass_g", 1.0),
    "kg": ("mass_g", 1000.0),
    "кг": ("mass_g", 1000.0),
    "kilogram": ("mass_g", 1000.0),
    "kilograms": ("mass_g", 1000.0),
    "mm": ("length_mm", 1.0),
    "мм": ("length_mm", 1.0),
    "cm": ("length_mm", 10.0),
    "см": ("length_mm", 10.0),
    "m": ("length_mm", 1000.0),
    "м": ("length_mm", 1000.0),
}
REVIEW_SCENARIOS = (
    "schema_or_request_error",
    "empty_extraction",
    "possible_match_judgement_language",
    "generic_concept",
    "duplicate_semantic_fact",
    "title_attribute_merged_fact",
    "title_only_fact",
    "attribute_only_fact",
    "missing_information",
    "conflicting_sources",
    "multiple_differences",
    "sparse_description",
    "potential_hard",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen semantic extraction on a label-isolated pilot."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--sampling-metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--prompt-version",
        default=None,
        help="Version stored in outputs; defaults to the prompt filename stem.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--review-size", type=int, default=80)
    parser.add_argument(
        "--validation-profile",
        choices=("legacy", "v1_4"),
        default=None,
        help="Semantic checks; auto-selects v1_4 for a v1_4 prompt version.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and write request previews without calling Qwen.",
    )
    parser.add_argument(
        "--skip-request-previews",
        action="store_true",
        help="Do not materialize the large label-free payload preview JSONL.",
    )
    parser.add_argument(
        "--skip-post-analysis",
        action="store_true",
        help="Keep a scalable raw checkpoint and postpone label loading/analysis.",
    )
    return parser.parse_args()


def now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_int(*parts: Any) -> int:
    return int.from_bytes(
        hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()[:8],
        "little",
    )


def parse_attributes(value: Any) -> dict[str, Any]:
    parsed = value if isinstance(value, dict) else json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("attributes must be a JSON object")
    return parsed


def qwen_input(row: Any) -> dict[str, Any]:
    payload = {
        "category": str(row.category),
        "item_a": {
            "title": str(row.title_a),
            "attributes": parse_attributes(row.attributes_a_json),
        },
        "item_b": {
            "title": str(row.title_b),
            "attributes": parse_attributes(row.attributes_b_json),
        },
    }
    if set(payload) != {"category", "item_a", "item_b"} or any(
        set(payload[item]) != {"title", "attributes"} for item in ("item_a", "item_b")
    ):
        raise RuntimeError("Unexpected Qwen payload structure")
    return payload


def extract_json(raw: str) -> dict[str, Any]:
    text = str(raw).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response does not contain a JSON object")
    result = json.loads(text[start : end + 1])
    if not isinstance(result, dict):
        raise ValueError("response JSON root is not an object")
    return result


def recursive_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from recursive_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from recursive_keys(nested)


def normalize_surface(value: Any) -> str:
    return " ".join(re.sub(r"[^0-9a-zа-яё]+", " ", str(value).casefold()).split())


def absence_marker(value: Any) -> bool:
    normalized = normalize_surface(value)
    return normalized in ABSENCE_MARKERS or (
        normalized.startswith("уточн") and "продав" in normalized
    )


def semantic_value_candidates(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    result: list[str] = []
    for key in ("raw_value", "normalized_value"):
        candidate = value.get(key)
        normalized = normalize_surface(candidate)
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def semantic_value_is_absent(value: Any) -> bool:
    return any(absence_marker(candidate) for candidate in semantic_value_candidates(value))


def item_text(item: dict[str, Any]) -> str:
    attributes = item.get("attributes", {})
    parts = [str(item.get("title", ""))]
    if isinstance(attributes, dict):
        parts.extend(f"{key} {value}" for key, value in attributes.items())
    return normalize_surface(" ".join(parts))


def value_present_in_item(value: Any, item: dict[str, Any]) -> bool:
    source = item_text(item)
    for candidate in semantic_value_candidates(value):
        if absence_marker(candidate):
            continue
        compact = candidate.replace(" ", "")
        if len(compact) < 3 or compact.isdigit():
            continue
        if candidate in source:
            return True
    return False


def value_tokens_present_in_title(value: Any, title: str) -> bool:
    title_tokens = set(normalize_surface(title).split())
    for candidate in semantic_value_candidates(value):
        tokens = [token for token in candidate.split() if len(token) >= 3]
        if tokens and all(token in title_tokens for token in tokens):
            return True
    return False


def quantity_concept(concept: Any) -> bool:
    concept = str(concept)
    return (
        concept in QUANTITY_CONCEPTS
        or concept.endswith("_quantity")
        or concept.endswith("_count")
    )


def quantity_value_errors(value: Any, concept: str, side: str) -> list[str]:
    if not isinstance(value, dict):
        return []
    errors: list[str] = []
    raw_value = normalize_surface(value.get("value", {}).get("raw_value"))
    evidence = value.get("evidence", [])
    if not re.search(r"\d", raw_value):
        errors.append(f"{concept}.{side} quantity has no explicit numeric count")
    for source in evidence:
        if source.get("source") != "attribute":
            continue
        raw_name = normalize_surface(source.get("raw_attribute_name"))
        measurement = any(token in raw_name for token in MEASUREMENT_ATTRIBUTE_TOKENS)
        count = any(token in raw_name for token in COUNT_ATTRIBUTE_TOKENS)
        if measurement and not count:
            errors.append(
                f"{concept}.{side} quantity is sourced from a measurement attribute"
            )
    return errors


def normalized_identity_values_equal(value_a: Any, value_b: Any) -> bool:
    if not isinstance(value_a, dict) or not isinstance(value_b, dict):
        return False
    normalized_a = value_a.get("normalized_value")
    normalized_b = value_b.get("normalized_value")
    if normalized_a is None or normalized_b is None:
        return False
    key_a = normalize_surface(normalized_a)
    key_b = normalize_surface(normalized_b)
    raw_tokens_a = normalize_surface(value_a.get("raw_value")).split()
    raw_tokens_b = normalize_surface(value_b.get("raw_value")).split()
    same_token_count = bool(raw_tokens_a and len(raw_tokens_a) == len(raw_tokens_b))
    return bool(key_a and key_a == key_b and same_token_count)


def evidence_supported(evidence: dict[str, Any], title: str, attributes: dict[str, Any]) -> bool:
    fragment = normalize_surface(evidence["raw_fragment"])
    if not fragment:
        return False
    if evidence["source"] == "title":
        return evidence["raw_attribute_name"] is None and fragment in normalize_surface(title)
    raw_name = evidence["raw_attribute_name"]
    if not isinstance(raw_name, str) or raw_name not in attributes:
        return False
    source_text = f"{raw_name} {attributes[raw_name]}"
    return fragment in normalize_surface(source_text)


def semantic_value_text(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    normalized = value.get("normalized_value")
    if normalized is not None and str(normalized).strip():
        return str(normalized)
    raw_value = value.get("raw_value")
    if raw_value is not None and str(raw_value).strip():
        return str(raw_value)
    raw_values = value.get("raw_values")
    if isinstance(raw_values, list) and raw_values:
        return " ".join(map(str, raw_values))
    return None


def conservative_identity_key(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    raw_value = value.get("raw_value")
    if raw_value is not None and str(raw_value).strip():
        text = str(raw_value)
    elif isinstance(value.get("raw_values"), list) and value["raw_values"]:
        text = " ".join(map(str, value["raw_values"]))
    else:
        text = semantic_value_text(value)
    if text is None:
        return None
    key = re.sub(r"[^0-9a-zа-яё]+", "", text.casefold())
    return key or None


def anchor_concept_supported(anchor_type: Any, concept: Any) -> bool:
    anchor_type, concept = str(anchor_type), str(concept)
    if anchor_type == "other_identity":
        return any(
            token in concept
            for token in ("identifier", "identity", "product_code")
        )
    return concept in ANCHOR_CONCEPTS.get(anchor_type, set())


def numeric_normalized_value(value: Any) -> float | None:
    if not isinstance(value, dict):
        return None
    candidate = value.get("normalized_value")
    if candidate is None:
        return None
    text = str(candidate).strip().replace(",", ".")
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return None
    return float(text)


def semantic_values_equivalent(value_a: Any, value_b: Any) -> bool:
    if not isinstance(value_a, dict) or not isinstance(value_b, dict):
        return False
    raw_a = conservative_identity_key(value_a)
    raw_b = conservative_identity_key(value_b)
    if raw_a is not None and raw_a == raw_b:
        return True
    normalized_a = value_a.get("normalized_value")
    normalized_b = value_b.get("normalized_value")
    unit_a = str(value_a.get("unit") or "").strip().casefold()
    unit_b = str(value_b.get("unit") or "").strip().casefold()
    if normalized_a is not None and normalized_b is not None:
        norm_a = normalize_surface(normalized_a)
        norm_b = normalize_surface(normalized_b)
        if norm_a and norm_a == norm_b and unit_a == unit_b:
            return True
    number_a = numeric_normalized_value(value_a)
    number_b = numeric_normalized_value(value_b)
    converted_a = UNIT_TO_BASE.get(unit_a)
    converted_b = UNIT_TO_BASE.get(unit_b)
    if (
        number_a is not None
        and number_b is not None
        and converted_a is not None
        and converted_b is not None
        and converted_a[0] == converted_b[0]
    ):
        return abs(number_a * converted_a[1] - number_b * converted_b[1]) < 1e-9
    return False


def normalize_extraction(extraction: dict[str, Any]) -> dict[str, Any]:
    """Convert version-specific Qwen JSON to the stable pair representation."""
    if "semantic_facts" in extraction:
        anchors: list[dict[str, Any]] = []
        differences: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []
        for source_fact in extraction.get("semantic_facts", []):
            fact = copy.deepcopy(source_fact)
            relation = fact["relation"]
            side_a, side_b = fact.get("a"), fact.get("b")
            value_a = side_a["value"] if side_a is not None else None
            value_b = side_b["value"] if side_b is not None else None
            evidence_a = side_a["evidence"] if side_a is not None else []
            evidence_b = side_b["evidence"] if side_b is not None else []
            if relation == "identity_same":
                anchor_type = str(fact["anchor_type"])
                anchors.append(
                    {
                        "anchor_type": anchor_type,
                        "concept": fact["concept"],
                        "value_a": value_a,
                        "value_b": value_b,
                        "relation": "same",
                        "strength": ANCHOR_STRENGTH.get(anchor_type, "medium"),
                        "evidence_a": evidence_a,
                        "evidence_b": evidence_b,
                        "confidence": fact["confidence"],
                    }
                )
                continue
            if relation in {"missing_a", "missing_b"}:
                missing_a = relation == "missing_a"
                missing.append(
                    {
                        "concept": fact["concept"],
                        "value_a": value_a,
                        "value_b": value_b,
                        "relation": relation,
                        "relation_direction": "item_a" if missing_a else "item_b",
                        "evidence_a": evidence_a,
                        "evidence_b": evidence_b,
                        "confidence": fact["confidence"],
                    }
                )
                continue
            direction = fact.get("direction")
            differences.append(
                {
                    "concept": fact["concept"],
                    "value_a": value_a,
                    "value_b": value_b,
                    "relation": relation,
                    "relation_direction": direction or "symmetric",
                    "evidence_a": evidence_a,
                    "evidence_b": evidence_b,
                    "confidence": fact["confidence"],
                }
            )
        normalized = {
            "identity_anchors": anchors,
            "differences": differences,
            "missing_information": missing,
        }
        facts = anchors + differences + missing
        normalized["pair_summary"] = {
            "salient_concepts": list(
                dict.fromkeys(
                    str(fact["concept"]) for fact in facts if fact.get("concept")
                )
            ),
            "uncertainties": [],
        }
        normalized["extraction_warnings"] = []
        return normalized

    normalized = copy.deepcopy(extraction)
    for anchor in normalized.get("identity_anchors", []):
        anchor.setdefault(
            "strength",
            ANCHOR_STRENGTH.get(str(anchor.get("anchor_type")), "medium"),
        )
    expanded_missing: list[dict[str, Any]] = []
    for fact in normalized.get("missing_information", []):
        if "known_value" not in fact:
            expanded_missing.append(fact)
            continue
        relation = fact["relation"]
        missing_a = relation == "missing_a"
        expanded_missing.append(
            {
                "concept": fact["concept"],
                "value_a": None if missing_a else fact["known_value"],
                "value_b": fact["known_value"] if missing_a else None,
                "relation": relation,
                "relation_direction": "item_a" if missing_a else "item_b",
                "evidence_a": [] if missing_a else fact["known_evidence"],
                "evidence_b": fact["known_evidence"] if missing_a else [],
                "confidence": fact["confidence"],
            }
        )
    normalized["missing_information"] = expanded_missing
    facts = (
        normalized.get("identity_anchors", [])
        + normalized.get("differences", [])
        + normalized.get("missing_information", [])
    )
    salient_concepts = list(
        dict.fromkeys(str(fact["concept"]) for fact in facts if fact.get("concept"))
    )
    normalized["pair_summary"] = {
        "salient_concepts": salient_concepts,
        "uncertainties": [],
    }
    normalized["extraction_warnings"] = []
    return normalized


def v14_fact_validation_errors(
    fact: dict[str, Any], payload: dict[str, Any], index: int
) -> list[str]:
    errors: list[str] = []
    concept = str(fact.get("concept"))
    relation = str(fact.get("relation"))
    sides = {side: fact.get(side) for side in ("a", "b")}
    for side, side_value in sides.items():
        if not isinstance(side_value, dict):
            continue
        semantic_value = side_value.get("value")
        if semantic_value_is_absent(semantic_value):
            errors.append(
                f"semantic_facts[{index}].{side} contains an absence marker as a value"
            )
        if concept == "brand" and any(
            candidate in HARD_COUNTRY_AS_BRAND_VALUES
            for candidate in semantic_value_candidates(semantic_value)
        ):
            errors.append(
                f"semantic_facts[{index}].{side} country-like value used as brand"
            )
        if quantity_concept(concept):
            errors.extend(
                f"semantic_facts[{index}] {error}"
                for error in quantity_value_errors(side_value, concept, side)
            )

    if relation in {"missing_a", "missing_b"}:
        missing_side = "a" if relation == "missing_a" else "b"
        known_side = "b" if missing_side == "a" else "a"
        known = sides.get(known_side)
        known_value = known.get("value") if isinstance(known, dict) else None
        if value_present_in_item(known_value, payload[f"item_{missing_side}"]):
            errors.append(
                f"semantic_facts[{index}] missing value is present on item_{missing_side}"
            )

    if relation == "different_value" and concept in {"product_type", "calculator_type"}:
        for side, other_side in (("a", "b"), ("b", "a")):
            other = sides.get(other_side)
            if not isinstance(other, dict):
                continue
            evidence_sources = {
                evidence.get("source") for evidence in other.get("evidence", [])
            }
            current = sides.get(side)
            current_value = current.get("value") if isinstance(current, dict) else None
            if evidence_sources == {"attribute"} and value_tokens_present_in_title(
                current_value, str(payload[f"item_{other_side}"]["title"])
            ):
                errors.append(
                    f"semantic_facts[{index}] possible title/attribute source conflict "
                    f"inside item_{other_side}"
                )
    return errors


def unified_semantic_validation_errors(
    extraction: dict[str, Any],
    payload: dict[str, Any],
    validation_profile: str = "legacy",
) -> list[str]:
    errors: list[str] = []
    facts = extraction.get("semantic_facts", [])
    concepts = [str(fact.get("concept")) for fact in facts]
    duplicates = sorted(
        concept for concept, count in Counter(concepts).items() if count > 1
    )
    if duplicates:
        errors.append(f"semantic_facts contains duplicate concepts: {duplicates}")
    generic = sorted(set(concepts) & GENERIC_CONCEPTS)
    if generic:
        errors.append(f"semantic_facts contains generic concepts: {generic}")
    missing_count = sum(
        fact.get("relation") in {"missing_a", "missing_b"} for fact in facts
    )
    if missing_count > 2:
        errors.append(f"semantic_facts contains {missing_count} missing facts; max=2")
    for index, fact in enumerate(facts):
        for side in ("a", "b"):
            side_value = fact.get(side)
            if side_value is None:
                continue
            title = payload[f"item_{side}"]["title"]
            attributes = payload[f"item_{side}"]["attributes"]
            for evidence in side_value.get("evidence", []):
                if not evidence_supported(evidence, title, attributes):
                    errors.append(
                        f"semantic_facts[{index}].{side}.evidence is not source-supported"
                    )
        relation = fact.get("relation")
        side_a, side_b = fact.get("a"), fact.get("b")
        value_a = side_a.get("value") if isinstance(side_a, dict) else None
        value_b = side_b.get("value") if isinstance(side_b, dict) else None
        if relation == "identity_same":
            key_a = conservative_identity_key(value_a)
            key_b = conservative_identity_key(value_b)
            normalized_equivalent = (
                validation_profile == "v1_4"
                and fact.get("anchor_type") in {"brand", "product_line", "model_family"}
                and normalized_identity_values_equal(value_a, value_b)
            )
            if (key_a is None or key_b is None or key_a != key_b) and not normalized_equivalent:
                errors.append(
                    f"semantic_facts[{index}] identity_same values differ conservatively"
                )
            if not anchor_concept_supported(fact.get("anchor_type"), fact.get("concept")):
                errors.append(
                    f"semantic_facts[{index}] unsupported anchor_type/concept: "
                    f"{fact.get('anchor_type')}/{fact.get('concept')}"
                )
        if relation in {"different_value", "subset", "more_specific"} and (
            semantic_values_equivalent(value_a, value_b)
        ):
            errors.append(
                f"semantic_facts[{index}] relation={relation} but values are equivalent"
            )
        if validation_profile == "v1_4":
            errors.extend(v14_fact_validation_errors(fact, payload, index))
    return errors


def semantic_validation_errors(
    extraction: dict[str, Any],
    payload: dict[str, Any],
    validation_profile: str = "legacy",
) -> list[str]:
    errors: list[str] = []
    forbidden = sorted({key.casefold() for key in recursive_keys(extraction)} & FORBIDDEN_OUTPUT_KEYS)
    if forbidden:
        errors.append(f"forbidden output keys: {forbidden}")
    if "semantic_facts" in extraction:
        errors.extend(
            unified_semantic_validation_errors(extraction, payload, validation_profile)
        )
        return errors
    for section in ("identity_anchors", "differences"):
        for index, fact in enumerate(extraction.get(section, [])):
            for side in ("a", "b"):
                title = payload[f"item_{side}"]["title"]
                attributes = payload[f"item_{side}"]["attributes"]
                for evidence in fact.get(f"evidence_{side}", []):
                    if not evidence_supported(evidence, title, attributes):
                        errors.append(
                            f"{section}[{index}].evidence_{side} is not source-supported"
                        )
    for index, anchor in enumerate(extraction.get("identity_anchors", [])):
        key_a = conservative_identity_key(anchor.get("value_a"))
        key_b = conservative_identity_key(anchor.get("value_b"))
        if key_a is None or key_b is None or key_a != key_b:
            errors.append(
                f"identity_anchors[{index}] relation=same but conservative values differ"
            )
    for index, fact in enumerate(extraction.get("differences", [])):
        relation = fact.get("relation")
        value_a, value_b = fact.get("value_a"), fact.get("value_b")
        evidence_a = fact.get("evidence_a", [])
        evidence_b = fact.get("evidence_b", [])
        if relation in {"missing_a", "missing_b"}:
            errors.append(f"differences[{index}] contains a missing relation")
        if relation == "same":
            errors.append(f"differences[{index}] contains forbidden relation=same")
        if relation == "conflicting_sources":
            for side, value, evidence in (
                ("a", value_a, evidence_a),
                ("b", value_b, evidence_b),
            ):
                if (value is None) != (not evidence):
                    errors.append(
                        f"differences[{index}] conflicting_sources has inconsistent value/evidence_{side}"
                    )
            if not (
                (value_a is not None and len(evidence_a) >= 2)
                or (value_b is not None and len(evidence_b) >= 2)
            ):
                errors.append(
                    f"differences[{index}] conflicting_sources requires >=2 evidence on a conflicted side"
                )
        elif value_a is None or value_b is None or not evidence_a or not evidence_b:
            errors.append(
                f"differences[{index}] non-conflict relation requires values and evidence on both sides"
            )
    for index, fact in enumerate(extraction.get("missing_information", [])):
        relation = fact.get("relation")
        if "known_value" in fact:
            known_side = "b" if relation == "missing_a" else "a"
            title = payload[f"item_{known_side}"]["title"]
            attributes = payload[f"item_{known_side}"]["attributes"]
            for evidence in fact.get("known_evidence", []):
                if not evidence_supported(evidence, title, attributes):
                    errors.append(
                        f"missing_information[{index}].known_evidence is not supported by item_{known_side}"
                    )
            continue
        for side in ("a", "b"):
            title = payload[f"item_{side}"]["title"]
            attributes = payload[f"item_{side}"]["attributes"]
            for evidence in fact.get(f"evidence_{side}", []):
                if not evidence_supported(evidence, title, attributes):
                    errors.append(
                        f"missing_information[{index}].evidence_{side} is not source-supported"
                    )
        if relation == "missing_a" and not (
            fact.get("value_a") is None
            and fact.get("value_b") is not None
            and not fact.get("evidence_a")
            and fact.get("evidence_b")
        ):
            errors.append(f"missing_information[{index}] violates missing_a semantics")
        if relation == "missing_b" and not (
            fact.get("value_b") is None
            and fact.get("value_a") is not None
            and not fact.get("evidence_b")
            and fact.get("evidence_a")
        ):
            errors.append(f"missing_information[{index}] violates missing_b semantics")
    section_concepts: dict[str, list[str]] = {
        section: [str(fact.get("concept")) for fact in extraction.get(section, [])]
        for section in ("identity_anchors", "differences", "missing_information")
    }
    for section, concepts in section_concepts.items():
        duplicates = sorted(concept for concept, count in Counter(concepts).items() if count > 1)
        if duplicates:
            errors.append(f"{section} contains duplicate concepts: {duplicates}")
    for left, right in (
        ("identity_anchors", "differences"),
        ("identity_anchors", "missing_information"),
        ("differences", "missing_information"),
    ):
        overlap = sorted(set(section_concepts[left]) & set(section_concepts[right]))
        if overlap:
            errors.append(f"concepts occur in {left} and {right}: {overlap}")
    return errors


class QwenClient:
    def __init__(
        self,
        api_base: str,
        model: str,
        timeout: float,
        retries: int,
        max_tokens: int,
        system_prompt: str,
        validator: jsonschema.protocols.Validator,
        validation_profile: str = "legacy",
    ) -> None:
        self.base = api_base.rstrip("/")
        self.url = self.base + "/chat/completions"
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.validator = validator
        self.validation_profile = validation_profile

    def preflight(self) -> None:
        request = urllib.request.Request(self.base + "/models", method="GET")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=min(self.timeout, 15.0)) as response:
            payload = json.load(response)
        served = [str(row.get("id")) for row in payload.get("data", [])]
        if self.model not in served:
            raise RuntimeError(
                f"Requested model {self.model!r} is not served; available={served}"
            )

    def ask(self, pair_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        user_content = json.dumps({"pair": payload}, ensure_ascii=False)
        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        encoded = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
        request_hash = sha256_bytes(encoded)
        last_error: Exception | None = None
        last_raw: str | None = None
        body: dict[str, Any] | None = None
        choice: dict[str, Any] | None = None
        completed_attempt = 0
        started = time.perf_counter()
        for attempt in range(1, self.retries + 1):
            try:
                request = urllib.request.Request(
                    self.url,
                    data=encoded,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(request, timeout=self.timeout) as response:
                    body = json.load(response)
                choice = body["choices"][0]
                last_raw = str(choice["message"]["content"])
                completed_attempt = attempt
                break
            except Exception as error:
                last_error = error
                if attempt < self.retries:
                    time.sleep(min(20.0, 2.0 ** (attempt - 1)))
        if body is None or choice is None or last_raw is None:
            return {
                "pair_id": pair_id,
                "status": "error",
                "attempt": self.retries,
                "request_hash": request_hash,
                "error_type": type(last_error).__name__ if last_error else "UnknownError",
                "error": str(last_error),
                "latency_seconds": time.perf_counter() - started,
                "raw_response": last_raw,
                "completed_at": now(),
            }

        response_metadata = {
            "pair_id": pair_id,
            "attempt": completed_attempt,
            "request_hash": request_hash,
            "response_id": body.get("id"),
            "response_model": body.get("model", self.model),
            "finish_reason": choice.get("finish_reason"),
            "usage": body.get("usage", {}),
            "latency_seconds": time.perf_counter() - started,
            "raw_response": last_raw,
            "completed_at": now(),
        }
        try:
            extraction = extract_json(last_raw)
        except Exception as error:
            return {
                **response_metadata,
                "status": "invalid",
                "validation_stage": "json",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        schema_errors = sorted(
            error.message for error in self.validator.iter_errors(extraction)
        )
        if schema_errors:
            return {
                **response_metadata,
                "status": "invalid",
                "validation_stage": "schema",
                "error_type": "SchemaValidationError",
                "error": "schema: " + " | ".join(schema_errors[:12]),
                "schema_errors": schema_errors,
                "parsed_response": extraction,
            }
        semantic_errors = semantic_validation_errors(
            extraction, payload, self.validation_profile
        )
        if semantic_errors:
            return {
                **response_metadata,
                "status": "invalid",
                "validation_stage": "semantic",
                "error_type": "SemanticValidationError",
                "error": "semantic validation: " + " | ".join(semantic_errors[:12]),
                "semantic_errors": semantic_errors,
                "parsed_response": extraction,
            }
        normalized_extraction = normalize_extraction(extraction)
        return {
            **response_metadata,
            "status": "ok",
            "schema_response": extraction,
            "parsed_response": normalized_extraction,
        }


def read_latest_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("pair_id"):
                latest[str(row["pair_id"])] = row
    return latest


def read_completed_checkpoint(
    path: Path,
    prompt_sha: str,
    schema_sha: str,
    model: str,
    validation_profile: str,
) -> dict[str, str]:
    """Read only resumable IDs/statuses, never retain multi-GB raw responses."""
    completed: dict[str, str] = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pair_id = row.get("pair_id")
            status = row.get("status")
            requested_model = row.get("requested_model", row.get("response_model", ""))
            if (
                pair_id
                and status in {"ok", "invalid"}
                and row.get("prompt_sha256") == prompt_sha
                and row.get("schema_sha256") == schema_sha
                and str(requested_model).casefold() == model.casefold()
                and row.get("validation_profile", "legacy") == validation_profile
            ):
                completed[str(pair_id)] = str(status)
    return completed


def fact_key(fact: dict[str, Any]) -> tuple[Any, ...]:
    def normalized(side: str) -> Any:
        value = fact.get(f"value_{side}")
        if value is None:
            return None
        return (
            value.get("normalized_value")
            or value.get("raw_value")
            or tuple(value.get("raw_values", []))
        )
    return (fact.get("concept"), fact.get("relation"), normalized("a"), normalized("b"))


def source_types(fact: dict[str, Any]) -> set[str]:
    return {
        evidence["source"]
        for side in ("a", "b")
        for evidence in fact.get(f"evidence_{side}", [])
    }


def diagnostic_flags(extraction: dict[str, Any], raw_response: str) -> list[str]:
    anchors = extraction["identity_anchors"]
    differences = extraction["differences"]
    missing = extraction["missing_information"]
    facts = anchors + differences + missing
    flags: list[str] = []
    if not facts:
        flags.append("empty_extraction")
    concepts = [str(fact.get("concept", "")) for fact in facts]
    if any(concept in GENERIC_CONCEPTS for concept in concepts):
        flags.append("generic_concept")
    keys = [fact_key(fact) for fact in differences + missing]
    if len(keys) != len(set(keys)):
        flags.append("duplicate_semantic_fact")
    if any(source_types(fact) == {"title", "attribute"} for fact in facts):
        flags.append("title_attribute_merged_fact")
    if any(source_types(fact) == {"title"} for fact in facts):
        flags.append("title_only_fact")
    if any(source_types(fact) == {"attribute"} for fact in facts):
        flags.append("attribute_only_fact")
    if missing:
        flags.append("missing_information")
    if any(fact.get("relation") == "conflicting_sources" for fact in differences):
        flags.append("conflicting_sources")
    if len(differences) >= 3:
        flags.append("multiple_differences")
    if MATCH_JUDGEMENT_RE.search(raw_response):
        flags.append("possible_match_judgement_language")
    return flags


def full_pair_object(
    input_row: Any,
    response: dict[str, Any],
    human_label: int,
    model: str,
    prompt_version: str,
    prompt_sha: str,
    schema_sha: str,
) -> dict[str, Any]:
    extraction = response["parsed_response"]
    return {
        "pair_id": str(input_row.pair_id),
        "item_id_a": int(input_row.item_id_a),
        "item_id_b": int(input_row.item_id_b),
        "category": str(input_row.category),
        "identity_anchors": extraction["identity_anchors"],
        "differences": extraction["differences"],
        "missing_information": extraction["missing_information"],
        "pair_summary": extraction["pair_summary"],
        "extraction_warnings": extraction["extraction_warnings"],
        "extraction_metadata": {
            "model": response.get("response_model") or model,
            "prompt_version": prompt_version,
            "prompt_sha256": prompt_sha,
            "schema_sha256": schema_sha,
            "response_id": response.get("response_id"),
            "finish_reason": response.get("finish_reason"),
            "usage": response.get("usage", {}),
            "latency_seconds": response.get("latency_seconds"),
            "raw_response": response.get("raw_response"),
        },
        # This field is attached here, after inference, and was absent from payload.
        "human_label": int(human_label),
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def flattened_row(pair: dict[str, Any]) -> dict[str, Any]:
    metadata = pair["extraction_metadata"]
    flags = diagnostic_flags(pair, str(metadata.get("raw_response") or ""))
    return {
        "pair_id": pair["pair_id"],
        "item_id_a": pair["item_id_a"],
        "item_id_b": pair["item_id_b"],
        "category": pair["category"],
        "human_label": pair["human_label"],
        "identity_anchor_count": len(pair["identity_anchors"]),
        "difference_count": len(pair["differences"]),
        "missing_information_count": len(pair["missing_information"]),
        "identity_anchors_json": json.dumps(pair["identity_anchors"], ensure_ascii=False),
        "differences_json": json.dumps(pair["differences"], ensure_ascii=False),
        "missing_information_json": json.dumps(pair["missing_information"], ensure_ascii=False),
        "pair_summary_json": json.dumps(pair["pair_summary"], ensure_ascii=False),
        "extraction_warnings_json": json.dumps(pair["extraction_warnings"], ensure_ascii=False),
        "diagnostic_flags_json": json.dumps(flags, ensure_ascii=False),
        "model": metadata["model"],
        "prompt_version": metadata["prompt_version"],
        "prompt_sha256": metadata["prompt_sha256"],
        "schema_sha256": metadata["schema_sha256"],
        "latency_seconds": metadata["latency_seconds"],
        "raw_response": metadata["raw_response"],
    }


def build_manual_review_queue(
    flat: pd.DataFrame,
    inputs: pd.DataFrame,
    metadata: pd.DataFrame | None,
    review_size: int,
) -> pd.DataFrame:
    review = flat.merge(inputs, on=["pair_id", "item_id_a", "item_id_b", "category"], how="left")
    if metadata is not None:
        sampling = metadata.drop(columns=["category", "human_label"], errors="ignore")
        review = review.merge(sampling, on="pair_id", how="left")
    review["diagnostic_flags"] = review["diagnostic_flags_json"].map(json.loads)
    for scenario in REVIEW_SCENARIOS:
        if scenario in review.columns:
            continue
        review[scenario] = review["diagnostic_flags"].map(lambda flags: scenario in flags)
    if "sparse_description" not in review.columns:
        review["sparse_description"] = False
    if "potential_hard" not in review.columns:
        review["potential_hard"] = False
    review["review_scenarios"] = review.apply(
        lambda row: json.dumps(
            [scenario for scenario in REVIEW_SCENARIOS if bool(row.get(scenario, False))],
            ensure_ascii=False,
        ),
        axis=1,
    )
    review["stable_review_rank"] = review["pair_id"].map(
        lambda pair_id: stable_int("manual_review", pair_id)
    )
    selected: list[int] = []
    selected_set: set[int] = set()
    ordered = review.sort_values("stable_review_rank")
    while len(selected) < min(review_size, len(review)):
        progress = False
        for scenario in REVIEW_SCENARIOS:
            candidates = ordered[ordered[scenario].fillna(False)]
            candidate = next(
                (index for index in candidates.index if index not in selected_set), None
            )
            if candidate is not None and len(selected) < review_size:
                selected.append(candidate)
                selected_set.add(candidate)
                progress = True
        if not progress:
            break
    if len(selected) < min(review_size, len(review)):
        for index in ordered.index:
            if index not in selected_set:
                selected.append(index)
                selected_set.add(index)
                if len(selected) == min(review_size, len(review)):
                    break
    queue = review.loc[selected].copy()
    queue["inference_error"] = ""
    queue["manual_review_status"] = ""
    queue["manual_review_notes"] = ""
    columns = [
        "pair_id", "category", "human_label", "review_scenarios",
        "title_a", "attributes_a_json", "title_b", "attributes_b_json",
        "identity_anchors_json", "differences_json", "missing_information_json",
        "pair_summary_json", "raw_response", "inference_error",
        "manual_review_status", "manual_review_notes",
    ]
    return queue[columns]


def analyze_results(
    requested: pd.DataFrame,
    latest: dict[str, dict[str, Any]],
    labels_path: Path,
    metadata_path: Path,
    output_dir: Path,
    model: str,
    prompt_version: str,
    prompt_sha: str,
    schema_sha: str,
    review_size: int,
) -> dict[str, Any]:
    # Critical ordering: labels are first opened here, after every Qwen future has completed.
    labels = pd.read_parquet(labels_path)
    if set(labels.columns) != {"pair_id", "human_label"}:
        raise ValueError("labels parquet must contain exactly pair_id,human_label")
    label_by_pair = labels.set_index("pair_id", verify_integrity=True)["human_label"]
    requested_ids = set(requested["pair_id"])
    if not requested_ids.issubset(label_by_pair.index):
        raise RuntimeError("Labels are missing requested pair IDs")
    input_by_pair = requested.set_index("pair_id", drop=False, verify_integrity=True)
    successful = [
        latest[pair_id]
        for pair_id in requested["pair_id"]
        if pair_id in latest and latest[pair_id].get("status") == "ok"
    ]
    pair_objects = [
        full_pair_object(
            input_by_pair.loc[response["pair_id"]],
            response,
            int(label_by_pair.loc[response["pair_id"]]),
            model,
            prompt_version,
            prompt_sha,
            schema_sha,
        )
        for response in successful
    ]
    write_jsonl(output_dir / "parsed_extractions.jsonl", pair_objects)
    flat = pd.DataFrame([flattened_row(pair) for pair in pair_objects])
    flat.to_parquet(output_dir / "parsed_extractions.parquet", index=False)

    relation_counts: Counter[str] = Counter()
    concept_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    concept_raw_name_counts: Counter[tuple[str, str, str]] = Counter()
    anchor_type_counts: Counter[tuple[str, str]] = Counter()
    exploded_rows: list[dict[str, Any]] = []
    for pair in pair_objects:
        facts = pair["identity_anchors"] + pair["differences"] + pair["missing_information"]
        relation_counts.update(str(fact["relation"]) for fact in facts)
        concept_counts.update(str(fact["concept"]) for fact in facts)
        anchor_type_counts.update(
            (str(anchor["anchor_type"]), str(anchor["strength"]))
            for anchor in pair["identity_anchors"]
        )
        semantic_signature = json.dumps(
            {
                "identity_anchors": pair["identity_anchors"],
                "differences": pair["differences"],
                "missing_information": pair["missing_information"],
            },
            ensure_ascii=False,
        )
        for section in ("identity_anchors", "differences", "missing_information"):
            for fact_index, fact in enumerate(pair[section]):
                exploded_rows.append(
                    {
                        "pair_id": pair["pair_id"],
                        "category": pair["category"],
                        "human_label": pair["human_label"],
                        "section": section,
                        "fact_index": fact_index,
                        "concept": fact["concept"],
                        "relation": fact["relation"],
                        "fact_json": json.dumps(fact, ensure_ascii=False),
                        "full_semantic_signature_json": semantic_signature,
                    }
                )
                for side in ("a", "b"):
                    for evidence in fact.get(f"evidence_{side}", []):
                        if evidence["source"] == "attribute":
                            concept_raw_name_counts[
                                (
                                    str(pair["category"]),
                                    str(fact["concept"]),
                                    str(evidence["raw_attribute_name"]),
                                )
                            ] += 1
        flag_counts.update(
            diagnostic_flags(pair, str(pair["extraction_metadata"].get("raw_response") or ""))
        )
    pd.DataFrame(relation_counts.most_common(), columns=["relation", "count"]).to_csv(
        output_dir / "relation_frequency.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(concept_counts.most_common(), columns=["concept", "count"]).to_csv(
        output_dir / "canonical_concept_frequency.csv", index=False, encoding="utf-8-sig"
    )
    mapping_columns = [
        "category", "canonical_concept", "raw_attribute_name", "evidence_occurrences"
    ]
    mapping = pd.DataFrame(
        [
            {
                "category": category,
                "canonical_concept": concept,
                "raw_attribute_name": raw_name,
                "evidence_occurrences": count,
            }
            for (category, concept, raw_name), count in concept_raw_name_counts.items()
        ],
        columns=mapping_columns,
    )
    if len(mapping):
        mapping = mapping.sort_values(
            ["category", "raw_attribute_name", "evidence_occurrences"],
            ascending=[True, True, False],
        )
        ambiguity = (
            mapping.groupby(["category", "raw_attribute_name"], observed=True)
            .agg(
                canonical_concept_count=("canonical_concept", "nunique"),
                canonical_concepts=(
                    "canonical_concept", lambda values: "; ".join(sorted(set(values)))
                ),
                evidence_occurrences=("evidence_occurrences", "sum"),
            )
            .reset_index()
            .sort_values(
                ["canonical_concept_count", "evidence_occurrences"],
                ascending=[False, False],
            )
        )
    else:
        ambiguity = pd.DataFrame(
            columns=[
                "category", "raw_attribute_name", "canonical_concept_count",
                "canonical_concepts", "evidence_occurrences",
            ]
        )
    mapping.to_csv(
        output_dir / "canonical_concept_raw_attribute_mapping.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ambiguity.to_csv(
        output_dir / "canonical_mapping_consistency.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {"anchor_type": anchor_type, "strength": strength, "count": count}
            for (anchor_type, strength), count in anchor_type_counts.items()
        ],
        columns=["anchor_type", "strength", "count"],
    ).to_csv(output_dir / "identity_anchor_frequency.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        exploded_rows,
        columns=[
            "pair_id", "category", "human_label", "section", "fact_index",
            "concept", "relation", "fact_json", "full_semantic_signature_json",
        ],
    ).to_parquet(
        output_dir / "exploded_semantic_facts.parquet", index=False
    )
    errors = [
        latest[pair_id]
        for pair_id in requested["pair_id"]
        if pair_id in latest and latest[pair_id].get("status") in {"invalid", "error"}
    ]
    missing_response_ids = [pair_id for pair_id in requested["pair_id"] if pair_id not in latest]
    error_types = Counter(str(row.get("error_type", "unknown")) for row in errors)
    validation_stages = Counter(
        str(row.get("validation_stage", "request")) for row in errors
    )
    invalid_count = sum(row.get("status") == "invalid" for row in errors)
    request_error_count = sum(row.get("status") == "error" for row in errors)
    statistics = {
        "requested_pairs": len(requested),
        "successful_extractions": len(pair_objects),
        "failed_extractions": len(errors),
        "invalid_extractions": invalid_count,
        "request_errors": request_error_count,
        "missing_responses": len(missing_response_ids),
        "json_and_schema_validity_rate": len(pair_objects) / max(1, len(requested)),
        "error_types": dict(error_types),
        "validation_stages": dict(validation_stages),
        "mean_identity_anchors": float(flat["identity_anchor_count"].mean()) if len(flat) else None,
        "mean_differences": float(flat["difference_count"].mean()) if len(flat) else None,
        "mean_missing_information": float(flat["missing_information_count"].mean()) if len(flat) else None,
        "automatic_diagnostic_flags": dict(flag_counts),
        "relation_counts": dict(relation_counts),
        "unique_canonical_concepts": len(concept_counts),
        "raw_attribute_names_mapped_to_multiple_concepts": int(
            ambiguity["canonical_concept_count"].gt(1).sum()
        ) if len(ambiguity) else 0,
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha,
        "schema_sha256": schema_sha,
        "human_labels_loaded_after_inference": True,
    }
    (output_dir / "error_statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = pd.read_parquet(metadata_path) if metadata_path.exists() else None
    error_rows = []
    for error in errors:
        pair_id = str(error["pair_id"])
        source = input_by_pair.loc[pair_id]
        error_rows.append(
            {
                "pair_id": pair_id,
                "category": source["category"],
                "human_label": int(label_by_pair.loc[pair_id]),
                "review_scenarios": json.dumps(["schema_or_request_error"]),
                "title_a": source["title_a"],
                "attributes_a_json": source["attributes_a_json"],
                "title_b": source["title_b"],
                "attributes_b_json": source["attributes_b_json"],
                "identity_anchors_json": "[]",
                "differences_json": "[]",
                "missing_information_json": "[]",
                "pair_summary_json": "{}",
                "raw_response": error.get("raw_response"),
                "inference_error": error.get("error"),
                "manual_review_status": "",
                "manual_review_notes": "",
            }
        )
    failed_frame = pd.DataFrame(error_rows)
    failed_frame.to_csv(output_dir / "failed_extractions.csv", index=False, encoding="utf-8-sig")
    success_review_size = max(0, review_size - min(len(failed_frame), 10))
    review_parts = []
    if len(failed_frame):
        review_parts.append(failed_frame.head(10))
    if len(flat) and success_review_size:
        review_parts.append(
            build_manual_review_queue(flat, requested, metadata, success_review_size)
        )
    if review_parts:
        pd.concat(review_parts, ignore_index=True).to_csv(
            output_dir / "manual_review_queue.csv", index=False, encoding="utf-8-sig"
        )
    report = f"""# Qwen semantic extraction pilot: результаты

Prompt: `{prompt_version}`. Запрошено пар: **{len(requested)}**. Успешных
schema-valid extraction: **{len(pair_objects)}**; невалидных ответов: **{invalid_count}**;
ошибок запроса после retries: **{request_error_count}**;
ответ отсутствует: **{len(missing_response_ids)}**. JSON/schema validity:
**{statistics["json_and_schema_validity_rate"]:.2%}**.

Human label не входил ни в один Qwen payload и был загружен только после
завершения всех API futures. Raw ответы в `raw_responses.jsonl` label не содержат.

Среднее число identity anchors: **{statistics["mean_identity_anchors"] or 0:.2f}**;
differences: **{statistics["mean_differences"] or 0:.2f}**; missing facts:
**{statistics["mean_missing_information"] or 0:.2f}**.

Перед изменением prompt/schema необходимо вручную разметить
`manual_review_queue.csv` значениями `GOOD`, `BAD` или `AMBIGUOUS` и пояснить
ошибки. В первую очередь проверяются duplicate facts, missing-vs-mismatch,
hallucinated evidence, generic/fragmented concepts, identity-anchor strength,
title-only/attribute-only facts и conflicting sources. Human label служит только
контекстом последующего анализа и не является целью extraction prompt.

Pilot завершает этот этап. Скрипт намеренно не умеет автоматически переходить
на весь RULE_DISCOVERY без явно переданного другого dataset.
"""
    (output_dir / "run_report.md").write_text(report, encoding="utf-8")
    return statistics


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.max_pairs < 1 or args.retries < 1:
        raise ValueError("workers, max-pairs and retries must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path, labels_path = args.dataset.resolve(), args.labels.resolve()
    prompt_path, schema_path = args.prompt.resolve(), args.schema.resolve()
    prompt_version = args.prompt_version or prompt_path.stem
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", prompt_version):
        raise ValueError("prompt-version must contain only letters, digits, '.', '_' or '-'")
    validation_profile = args.validation_profile or (
        "v1_4" if "v1_4" in prompt_version.casefold() else "legacy"
    )
    inputs = pd.read_parquet(dataset_path)
    required = {
        "pair_id", "item_id_a", "item_id_b", "category", "title_a",
        "attributes_a_json", "title_b", "attributes_b_json",
    }
    if set(inputs.columns) != required:
        raise ValueError(
            f"Dataset must contain exactly label-free columns {sorted(required)}; "
            f"got={sorted(inputs.columns)}"
        )
    if inputs["pair_id"].duplicated().any():
        raise ValueError("Dataset pair IDs must be unique")
    requested = inputs.head(args.max_pairs).copy()
    prompt_text = prompt_path.read_text(encoding="utf-8")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    prompt_sha, schema_sha = sha256_file(prompt_path), sha256_file(schema_path)
    system_prompt = (
        prompt_text
        + "\n\n## JSON Schema (обязательна)\n"
        + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    )

    request_preview_path = output_dir / "request_payloads.jsonl"

    def preview_rows() -> Iterable[dict[str, Any]]:
        for row in requested.itertuples(index=False):
            yield {"pair_id": str(row.pair_id), "qwen_input": qwen_input(row)}

    if not args.skip_request_previews:
        write_jsonl(request_preview_path, preview_rows())
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "pairs": len(requested),
                    "qwen_input_fields": ["category", "item_a.title", "item_a.attributes", "item_b.title", "item_b.attributes"],
                    "human_label_in_qwen_input": False,
                    "prompt_version": prompt_version,
                    "validation_profile": validation_profile,
                    "request_preview": (
                        str(request_preview_path)
                        if not args.skip_request_previews
                        else "SKIPPED"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    client = QwenClient(
        args.api_base,
        args.model,
        args.timeout,
        args.retries,
        args.max_tokens,
        system_prompt,
        validator,
        validation_profile,
    )
    client.preflight()
    raw_path = output_dir / "raw_responses.jsonl"
    completed_status = read_completed_checkpoint(
        raw_path, prompt_sha, schema_sha, args.model, validation_profile
    )
    requested_ids = set(requested["pair_id"].astype(str))
    completed = requested_ids & set(completed_status)
    total_jobs = len(requested) - len(completed)

    def jobs() -> Iterable[dict[str, Any]]:
        for row in requested.itertuples(index=False):
            pair_id = str(row.pair_id)
            if pair_id not in completed:
                yield {"pair_id": pair_id, "qwen_input": qwen_input(row)}

    mode = "a" if raw_path.exists() else "w"
    added_ok = 0
    added_invalid = 0
    added_errors = 0
    with raw_path.open(mode, encoding="utf-8", buffering=1) as stream:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            job_iterator = iter(jobs())
            futures: dict[concurrent.futures.Future[Any], str] = {}

            def submit_next() -> bool:
                try:
                    job = next(job_iterator)
                except StopIteration:
                    return False
                future = pool.submit(client.ask, job["pair_id"], job["qwen_input"])
                futures[future] = job["pair_id"]
                return True

            for _ in range(min(total_jobs, args.workers * 2)):
                submit_next()

            finished = 0
            while futures:
                done, _ = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    futures.pop(future, None)
                    result = future.result()
                    finished += 1
                # Raw checkpoint is deliberately label-free.
                    if any(key in result for key in FORBIDDEN_INPUT_FIELDS):
                        raise RuntimeError("Internal error: label leaked into raw checkpoint")
                    result.update(
                        {
                            "prompt_version": prompt_version,
                            "prompt_sha256": prompt_sha,
                            "schema_sha256": schema_sha,
                            "requested_model": args.model,
                            "validation_profile": validation_profile,
                        }
                    )
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    if result["status"] == "ok":
                        added_ok += 1
                    elif result["status"] == "invalid":
                        added_invalid += 1
                    else:
                        added_errors += 1
                    if finished % 10 == 0 or finished == total_jobs:
                        print(
                            f"Qwen: {finished}/{total_jobs} new; ok={added_ok}, "
                            f"invalid={added_invalid}, request_errors={added_errors}, "
                            f"reused={len(completed)}",
                            flush=True,
                        )
                    submit_next()

    if args.skip_post_analysis:
        checkpoint = read_completed_checkpoint(
            raw_path, prompt_sha, schema_sha, args.model, validation_profile
        )
        checkpoint = {
            pair_id: status
            for pair_id, status in checkpoint.items()
            if pair_id in requested_ids
        }
        statistics = {
            "requested_pairs": len(requested),
            "checkpointed_pairs": len(checkpoint),
            "checkpoint_status_counts": dict(Counter(checkpoint.values())),
            "new_ok": added_ok,
            "new_invalid": added_invalid,
            "new_request_errors": added_errors,
            "reused": len(completed),
            "post_analysis": "SKIPPED_FOR_SCALABILITY",
            "human_labels_loaded": False,
        }
        manifest = {
            "schema_version": 1,
            "created_at": now(),
            "dataset": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "labels": str(labels_path),
            "labels_loaded_only_after_inference": True,
            "labels_loaded_in_this_run": False,
            "api_base": args.api_base,
            "model": args.model,
            "workers": args.workers,
            "max_pairs": args.max_pairs,
            "prompt_version": prompt_version,
            "validation_profile": validation_profile,
            "prompt": str(prompt_path),
            "prompt_sha256": prompt_sha,
            "schema": str(schema_path),
            "schema_sha256": schema_sha,
            "statistics": statistics,
        }
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(statistics, ensure_ascii=False, indent=2))
        return

    latest = read_latest_jsonl(raw_path)
    statistics = analyze_results(
        requested,
        latest,
        labels_path,
        args.sampling_metadata.resolve(),
        output_dir,
        args.model,
        prompt_version,
        prompt_sha,
        schema_sha,
        args.review_size,
    )
    manifest = {
        "schema_version": 1,
        "created_at": now(),
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "labels": str(labels_path),
        "labels_sha256": sha256_file(labels_path),
        "labels_loaded_only_after_inference": True,
        "api_base": args.api_base,
        "model": args.model,
        "workers": args.workers,
        "max_pairs": args.max_pairs,
        "prompt_version": prompt_version,
        "validation_profile": validation_profile,
        "prompt": str(prompt_path),
        "prompt_sha256": prompt_sha,
        "schema": str(schema_path),
        "schema_sha256": schema_sha,
        "statistics": statistics,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(statistics, ensure_ascii=False, indent=2), flush=True)
    print(f"Результаты сохранены в {output_dir}", flush=True)


if __name__ == "__main__":
    main()
