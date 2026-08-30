#!/usr/bin/env python3
"""Rebuild sparse soft-positive pairs as rich asymmetric seller listings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from rapidfuzz.fuzz import ratio, token_set_ratio


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from item_pipeline.normalization import (
    canonical_json_dumps,
    normalize_text,
    parse_attributes,
    stable_hash64,
)
from scripts.freeze_generated_pair_dataset import canonical_card


VERSION = "soft_positive_pair_rehydration_v2_alias_target_qwen_judge"
REFERENCE_POLICY = "label1_source_pair_exact_product_type_rich_card_v1"
PROMPT = ROOT / "item_pipeline/prompts/rehydrate_soft_positive_pair.md"
JUDGE_PROMPT = ROOT / "item_pipeline/prompts/judge_rehydrated_soft_positive_pair.md"
DEFAULT_SOURCE = ROOT / "item_pipeline/artifacts/soft_positive_tier_ab_16351_qwen_v1_composed"
DEFAULT_OUTPUT = ROOT / "item_pipeline/artifacts/soft_positive_tier_ab_16351_qwen_v2_rehydrated_raw"
DEFAULT_PILOT_INPUTS = ROOT / "data/qwen_atomic_differences_v2_full_train/pilot_inputs.parquet"
DEFAULT_PILOT_LABELS = ROOT / "data/qwen_atomic_differences_v2_full_train/pilot_labels.parquet"
DEFAULT_MODEL = "qwen3.5-397b-a17b-fp8"
DEFAULT_BASE_URL = "http://0.0.0.0:8994/v1"
TYPE_KEYS = {"тип товара", "вид товара", "категория товара", "тип"}
BRAND_KEYS = {
    "бренд", "brand", "производитель", "марка", "марка тс", "изготовитель",
    "завод изготовитель", "название бренда",
}
IDENTIFIER_RE = re.compile(
    r"(?i)(?:\bsku\b|\bартикул\b|\bарт\.?\s*[-:#]?\s*[a-zа-я0-9-]*\d|"
    r"код\s+(?:товара|изделия|продукта|модели|производителя)|партномер|part\s*number)"
)
TITLE_IDENTIFIER_RE = re.compile(
    r"(?i)\b(?:sku|артикул|арт\.|код\s+(?:товара|изделия|продукта|модели|производителя)|партномер|part\s*number)"
    r"\s*[:#-]?\s*[a-zа-я0-9._/-]*"
)
WORD_RE = re.compile(r"[0-9a-zа-я]+")
MEASUREMENT_RE = re.compile(
    r"(?i)^\s*[+-]?(?:\d+(?:[.,]\d+)?)\s*(?:кг|г|мг|л|мл|мм|см|м|шт|gb|гб|tb|тб|%)\s*$"
)
NUMERIC_CONCEPT_PARTS = {
    "weight",
    "quantity",
    "count",
    "capacity",
    "size_mm",
    "age_min",
    "min_age",
}
COLOR_WORDS = {
    "белый", "белая", "белое", "черный", "черная", "черное", "чёрный",
    "красный", "красная", "синий", "синяя", "зеленый", "зелёный",
    "желтый", "жёлтый", "фиолетовый", "розовый", "серый", "серебристый",
    "золотой", "оранжевый", "бежевый", "коричневый", "голубой",
}
COLOR_STEM_RE = re.compile(
    r"(?i)^(?:бел|черн|красн|син|зелен|зелён|желт|жёлт|фиолет|розов|сер|"
    r"серебрист|золот|оранжев|бежев|коричнев|голуб|бирюз|кораллов|мятн)[а-я-]*$"
)
COUNTRY_VALUES = {
    "австралия", "австрия", "беларусь", "бельгия", "болгария", "бразилия",
    "великобритания", "венгрия", "вьетнам", "германия", "греция", "дания",
    "индия", "индонезия", "испания", "италия", "казахстан", "канада",
    "китай", "кнр", "корея", "латвия", "литва", "малайзия", "мексика",
    "нидерланды", "норвегия", "пакистан", "польша", "португалия", "россия",
    "рф", "румыния", "сербия", "сингапур", "сша", "таиланд", "тайвань",
    "турция", "узбекистан", "украина", "финляндия", "франция", "чехия",
    "швейцария", "швеция", "южная корея", "япония",
}
EXPECTED_QUALITY_CHECKS = {
    "same_concrete_product",
    "only_atomic_semantic_difference",
    "brands_are_observed_or_targeted",
    "no_sku_or_article_copied",
    "seller_views_are_structurally_different",
}
GENERIC_BAD_VALUES = {
    "нет", "не указан", "не указано", "неизвестно", "none", "null", "-", "—",
    "нет бренда", "без бренда", "no brand", "nobrand", "no-name", "noname",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pilot-inputs", type=Path, default=DEFAULT_PILOT_INPUTS)
    parser.add_argument("--pilot-labels", type=Path, default=DEFAULT_PILOT_LABELS)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=60)
    parser.add_argument("--count", type=int)
    parser.add_argument(
        "--task-index",
        dest="task_indices",
        action="append",
        type=int,
        help="Generate only an exact composition_index; may be repeated (smoke/audit use).",
    )
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--task-seed-offset", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--max-tokens", type=int, default=3_600)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--generation-attempts", type=int, default=3)
    parser.add_argument("--task-retries", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = canonical_json_dumps(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("response must be a JSON object")
    return value


def words(value: Any) -> list[str]:
    return WORD_RE.findall(normalize_text(value))


def compact_card(name: Any, attributes: Any, *, max_attributes: int = 24) -> dict[str, Any]:
    parsed = parse_attributes(attributes)
    clean_name = re.sub(r"\s+", " ", TITLE_IDENTIFIER_RE.sub(" ", str(name))).strip()
    return {
        "name": clean_name[:500],
        "attributes": {
            str(key)[:160]: str(value)[:700]
            for key, value in list(parsed.items())[:max_attributes]
            if not IDENTIFIER_RE.search(str(key))
        },
    }


def explicit_product_type(attributes: dict[str, str]) -> set[str]:
    return {
        normalize_text(value)
        for key, value in attributes.items()
        if normalize_text(key) in TYPE_KEYS and normalize_text(value)
    }


def brand_values(attributes: dict[str, str]) -> set[str]:
    return {
        normalize_text(value)
        for key, value in attributes.items()
        if normalize_text(key) in BRAND_KEYS and normalize_text(value)
    }


def is_brand_semantic(concept: Any, key: Any) -> bool:
    """Classify by the executable attribute key, not noisy concept substrings."""

    key_text = normalize_text(key)
    if key_text in BRAND_KEYS:
        return True
    return bool(re.search(r"(?:^|\s)(?:бренд|производитель|изготовитель)(?:$|\s)", key_text))


def valid_reference_brand(value: str) -> bool:
    normalized = normalize_text(value)
    return bool(
        normalized
        and normalized not in GENERIC_BAD_VALUES
        and normalized not in COLOR_WORDS
        and not COLOR_STEM_RE.fullmatch(normalized)
        and normalized not in COUNTRY_VALUES
        and not MEASUREMENT_RE.fullmatch(normalized)
        and not normalized.isdigit()
        and not IDENTIFIER_RE.search(normalized)
    )


class ReferenceIndex:
    def __init__(self, inputs_path: Path, labels_path: Path) -> None:
        inputs = pd.read_parquet(inputs_path).set_index("pair_id", drop=False)
        labels = pd.read_parquet(labels_path)
        self.inputs = inputs
        self.labels = {
            str(row.pair_id): int(row.human_label)
            for row in labels.itertuples(index=False)
        }

    def pair(self, pair_id: str) -> dict[str, Any] | None:
        if pair_id not in self.inputs.index:
            return None
        row = self.inputs.loc[pair_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return {
            "pair_id": pair_id,
            "human_label": self.labels.get(pair_id),
            "category": str(row["category"]),
            "a": compact_card(row["title_a"], row["attributes_a_json"]),
            "b": compact_card(row["title_b"], row["attributes_b_json"]),
        }

    def references(self, rule: dict[str, Any], product_type: str) -> dict[str, Any]:
        candidates: list[tuple[int, str, str, dict[str, Any]]] = []
        positive_pairs: list[dict[str, Any]] = []
        product_key = normalize_text(product_type)
        for source in rule.get("source_examples") or []:
            pair_id = str(source.get("source_pair_id") or "")
            pair = self.pair(pair_id)
            if pair is None:
                continue
            if pair["human_label"] == 1:
                positive_pairs.append(pair)
            for side in ("a", "b"):
                card = pair[side]
                attrs = card["attributes"]
                exact_type = product_key in explicit_product_type(attrs)
                title_match = product_key in normalize_text(card["name"])
                score = (
                    int(exact_type) * 10_000
                    + int(title_match) * 3_000
                    + min(len(attrs), 30) * 30
                    + min(len(words(card["name"])), 20)
                )
                candidates.append((score, pair_id, side, card))
        if not candidates:
            raise ValueError("rule has no available source reference cards")
        candidates.sort(key=lambda value: (-value[0], value[1], value[2]))
        content = [entry[3] for entry in candidates[:2]]
        exact = [
            entry[3]
            for entry in candidates
            if product_key in explicit_product_type(entry[3]["attributes"])
            or product_key in normalize_text(entry[3]["name"])
        ]
        if exact:
            content[0] = exact[0]
        if not positive_pairs:
            raise ValueError("rule has no label=1 source pair template")
        # Match the typical human-positive card density (roughly 9 attrs/side)
        # rather than systematically selecting the richest available pair.
        positive_pairs.sort(
            key=lambda pair: (
                int(len(pair["a"]["attributes"]) < 5)
                + int(len(pair["b"]["attributes"]) < 5),
                abs(len(pair["a"]["attributes"]) - 9)
                + abs(len(pair["b"]["attributes"]) - 9),
                str(pair["pair_id"]),
            )
        )
        template = positive_pairs[0]
        brands = {
            value
            for card in [*content, template["a"], template["b"]]
            for value in brand_values(card["attributes"])
            if valid_reference_brand(value)
        }
        concept_key = normalize_text(rule.get("concept"))
        attribute_key = normalize_text(rule.get("required_attribute_key"))
        if is_brand_semantic(concept_key, attribute_key):
            grounded_brand_transitions: list[list[str]] = []
            seen_transitions: set[tuple[str, str]] = set()
            for source in (rule.get("source_examples") or []):
                value_a = normalize_text(source.get("target_value_a"))
                value_b = normalize_text(source.get("target_value_b"))
                if (
                    not valid_reference_brand(value_a)
                    or not valid_reference_brand(value_b)
                    or value_a == value_b
                ):
                    continue
                signature = tuple(sorted((value_a, value_b)))
                if signature in seen_transitions:
                    continue
                seen_transitions.add(signature)
                grounded_brand_transitions.append([value_a, value_b])
            brands.update(
                normalize_text(source.get(field))
                for source in (rule.get("source_examples") or [])
                for field in ("target_value_a", "target_value_b")
                if valid_reference_brand(str(source.get(field) or ""))
            )
        else:
            grounded_brand_transitions = []
        return {
            "content_references": content,
            "seller_pair_template": {
                "pair_id": template["pair_id"],
                "item_a": template["a"],
                "item_b": template["b"],
            },
            "allowed_observed_brands": sorted(brands),
            "grounded_brand_transitions": grounded_brand_transitions,
        }


def effective_application(task: dict[str, Any], references: dict[str, Any]) -> dict[str, Any]:
    application = dict(task["application"])
    application["source_original_value"] = application["original_value"]
    application["source_new_value"] = application["new_value"]
    application["pre_repair_required"] = False
    concept = normalize_text(task["rule"].get("concept"))
    key = normalize_text(task["rule"].get("required_attribute_key"))
    is_brand = is_brand_semantic(concept, key)
    if not is_brand:
        return application
    allowed = set(references["allowed_observed_brands"])
    values_valid = (
        normalize_text(application["original_value"]) in allowed
        and normalize_text(application["new_value"]) in allowed
        and normalize_text(application["original_value"])
        != normalize_text(application["new_value"])
        and not value_semantic_issues(concept, key, application["original_value"])
        and not value_semantic_issues(concept, key, application["new_value"])
    )
    if values_valid:
        return application
    transitions = references["grounded_brand_transitions"]
    if not transitions:
        raise ValueError("malformed brand application has no grounded source transition")
    selected = transitions[
        stable_hash64(20260831, task["composition_index"]) % len(transitions)
    ]
    application["original_value"], application["new_value"] = selected
    application["pre_repair_required"] = True
    return application


def build_prompt(task: dict[str, Any], references: dict[str, Any], feedback: list[str]) -> str:
    rule = task["rule"]
    application = task.get("effective_application") or effective_application(task, references)
    source_examples = [
        {
            "title_a": source.get("title_a"),
            "title_b": source.get("title_b"),
            "target_value_a": source.get("target_value_a"),
            "target_value_b": source.get("target_value_b"),
            "source_is_singleton": source.get("source_is_singleton"),
        }
        for source in (rule.get("source_examples") or [])[:4]
    ]
    template = references["seller_pair_template"]
    target_counts = {
        "item_a": max(5, min(20, len(template["item_a"]["attributes"]))),
        "item_b": max(5, min(20, len(template["item_b"]["attributes"]))),
    }
    payload = {
        "task": "rehydrate_trusted_soft_positive_atomic_pair",
        "target_label": 1,
        "category": task["category"],
        "product_type": task["product_type"],
        "trusted_rule": {
            "generation_rule_id": rule["generation_rule_id"],
            "concept": rule["concept"],
            "required_attribute_key": rule["required_attribute_key"],
            "generation_action": rule.get("generation_action"),
            "required_postcondition": rule.get("required_postcondition"),
        },
        "current_application_preferred_unless_semantically_invalid": application,
        "original_malformed_application_for_provenance": task["application"],
        "mandatory_pre_repair_instruction": (
            "Use the pre-repaired original_value/new_value exactly and set values_repaired=true."
            if application.get("pre_repair_required")
            else "Keep the preferred values unless a different semantic-type repair is essential."
        ),
        "source_evidence_for_value_repair": source_examples,
        "content_references_same_product_type": references["content_references"],
        "allowed_observed_brands": references["allowed_observed_brands"],
        "human_positive_pair_structure_template": template,
        "desired_attribute_counts": target_counts,
        "diversity_nonce_not_for_output": task["diversity_nonce"],
        "required_output": {
            "item_a": {"name": "string", "attributes": "object", "category": task["category"]},
            "item_b": {"name": "string", "attributes": "object", "category": task["category"]},
            "application": {
                "generation_rule_id": rule["generation_rule_id"],
                "concept": rule["concept"],
                "canonical_attribute_key": rule["required_attribute_key"],
                "attribute_key_a": "natural seller key used in item_a",
                "attribute_key_b": "natural seller key used in item_b",
                "original_value": "exact value used in item_a target attribute",
                "new_value": "exact value used in item_b target attribute",
                "values_repaired": "boolean",
                "repair_reason": "empty string when unchanged, otherwise short reason",
            },
            "identity_facts": ["at least one exact non-target fact shared by both items"],
            "quality_checks": {
                "same_concrete_product": True,
                "only_atomic_semantic_difference": True,
                "brands_are_observed_or_targeted": True,
                "no_sku_or_article_copied": True,
                "seller_views_are_structurally_different": True,
            },
        },
        "output_contract": {
            "top_level_keys_exactly": ["item_a", "item_b", "application", "identity_facts", "quality_checks"],
            "each_item_keys_exactly": ["name", "attributes", "category"],
            "canonical_target_key_for_meaning": rule["required_attribute_key"],
            "target_keys_may_be_natural_aliases": True,
            "attribute_count_each_between": [5, 20],
            "name_word_count_each_between": [5, 18],
            "different_attribute_key_sets": True,
            "all_other_same_named_non_target_keys_have_identical_values": True,
            "forbidden_anywhere": [
                "SKU",
                "артикул",
                "арт. + code",
                "код товара",
                "партномер",
                "случайный внутренний ID",
            ],
            "identity_facts_minimum": 2,
            "non_target_brand_policy": (
                "use exactly one allowed_observed_brand identically on both sides"
                if references["allowed_observed_brands"]
                else "omit every brand/manufacturer/marca key"
            ),
            "no_text_outside_json": True,
        },
    }
    if feedback:
        payload["previous_attempt_rejection_reasons"] = feedback[-8:]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def value_semantic_issues(concept: str, key: str, value: str) -> list[str]:
    issues: list[str] = []
    normalized = normalize_text(value)
    concept_key = normalize_text(concept)
    key_text = normalize_text(key)
    if not normalized or normalized in GENERIC_BAD_VALUES:
        issues.append("target_value_generic_or_empty")
    is_brand = is_brand_semantic(concept_key, key_text)
    if is_brand:
        if normalized in COLOR_WORDS or COLOR_STEM_RE.fullmatch(normalized):
            issues.append("brand_is_color")
        if normalized in COUNTRY_VALUES:
            issues.append("brand_is_country")
        if MEASUREMENT_RE.fullmatch(normalized):
            issues.append("brand_is_measurement")
        if normalized.isdigit():
            issues.append("brand_is_numeric")
    if any(part in concept_key for part in NUMERIC_CONCEPT_PARTS):
        if not re.search(r"\d", normalized):
            issues.append("numeric_concept_without_number")
    if "country" in concept_key or "страна" in key_text:
        if re.search(r"\d", normalized) or len(words(normalized)) > 4:
            issues.append("country_value_malformed")
    is_type = (
        concept_key in {"type", "product_type", "item_type"}
        or key_text in {"тип", "тип товара", "вид товара", "вид изделия"}
    )
    if is_type:
        if normalized in COLOR_WORDS or COLOR_STEM_RE.fullmatch(normalized):
            issues.append("type_value_is_color")
        if MEASUREMENT_RE.fullmatch(normalized) or normalized.isdigit():
            issues.append("type_value_is_measurement_or_number")
    return issues


def validate_response(
    response: dict[str, Any],
    task: dict[str, Any],
    references: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str], dict[str, Any]]:
    errors: list[str] = []
    required_top = {"item_a", "item_b", "application", "identity_facts", "quality_checks"}
    if set(response) != required_top:
        return None, ["top_level_shape"], {}
    items: list[dict[str, Any]] = []
    for side in ("item_a", "item_b"):
        item = response.get(side)
        if not isinstance(item, dict) or set(item) != {"name", "attributes", "category"}:
            errors.append(f"{side}_shape")
            continue
        if item.get("category") != task["category"]:
            errors.append(f"{side}_category")
        if not isinstance(item.get("name"), str) or not item["name"].strip():
            errors.append(f"{side}_name")
        attrs = item.get("attributes")
        if not isinstance(attrs, dict) or any(
            not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value.strip()
            for key, value in (attrs.items() if isinstance(attrs, dict) else [])
        ):
            errors.append(f"{side}_attributes")
            continue
        if not 5 <= len(attrs) <= 20:
            errors.append(f"{side}_attribute_count")
        if not 5 <= len(words(item["name"])) <= 18:
            errors.append(f"{side}_name_word_count")
        if IDENTIFIER_RE.search(item["name"]) or any(IDENTIFIER_RE.search(str(key)) for key in attrs):
            errors.append(f"{side}_identifier")
        items.append(item)
    if len(items) != 2:
        return None, errors, {}
    left, right = items
    left_attrs, right_attrs = left["attributes"], right["attributes"]
    if {normalize_text(key) for key in left_attrs} == {normalize_text(key) for key in right_attrs}:
        errors.append("attribute_key_sets_identical")
    application = response.get("application")
    rule = task["rule"]
    required_application = {
        "generation_rule_id", "concept", "canonical_attribute_key",
        "attribute_key_a", "attribute_key_b", "original_value",
        "new_value", "values_repaired", "repair_reason",
    }
    if not isinstance(application, dict) or set(application) != required_application:
        errors.append("application_shape")
        return None, errors, {}
    target_key = rule["required_attribute_key"]
    if (
        application.get("generation_rule_id") != rule["generation_rule_id"]
        or application.get("concept") != rule["concept"]
        or application.get("canonical_attribute_key") != target_key
        or not isinstance(application.get("values_repaired"), bool)
    ):
        errors.append("application_identity")
    key_a = application.get("attribute_key_a")
    key_b = application.get("attribute_key_b")
    if not isinstance(key_a, str) or not key_a.strip() or not isinstance(key_b, str) or not key_b.strip():
        errors.append("target_key_shape")
    elif key_a not in left_attrs or key_b not in right_attrs:
        errors.append("target_key_missing")
    else:
        if left_attrs[key_a] != application.get("original_value"):
            errors.append("original_value_mismatch")
        if right_attrs[key_b] != application.get("new_value"):
            errors.append("new_value_mismatch")
    original = str(application.get("original_value") or "")
    new = str(application.get("new_value") or "")
    if normalize_text(original) == normalize_text(new):
        errors.append("target_values_equal")
    errors.extend(value_semantic_issues(rule["concept"], target_key, original))
    errors.extend(value_semantic_issues(rule["concept"], target_key, new))

    allowed_brands = set(references["allowed_observed_brands"])
    is_brand_target = is_brand_semantic(rule["concept"], target_key)
    if is_brand_target:
        grounded_transitions = {
            tuple(sorted((normalize_text(a), normalize_text(b))))
            for a, b in references["grounded_brand_transitions"]
        }
        if (
            normalize_text(original) not in allowed_brands
            or normalize_text(new) not in allowed_brands
        ):
            errors.append("brand_target_not_source_grounded")
        if tuple(sorted((normalize_text(original), normalize_text(new)))) not in grounded_transitions:
            errors.append("brand_transition_not_source_grounded")
    else:
        observed_output_brands: list[str] = []
        for attrs in (left_attrs, right_attrs):
            observed_output_brands.extend(brand_values(attrs))
        if observed_output_brands and (
            not allowed_brands or any(value not in allowed_brands for value in observed_output_brands)
        ):
            errors.append("invented_or_changed_brand")

    left_normalized = {normalize_text(k): normalize_text(v) for k, v in left_attrs.items()}
    right_normalized = {normalize_text(k): normalize_text(v) for k, v in right_attrs.items()}
    target_normalized = {
        normalize_text(value)
        for value in (target_key, key_a, key_b)
        if isinstance(value, str)
    }
    common = set(left_normalized) & set(right_normalized)
    non_target_common = common - target_normalized
    conflicting_common = {
        key for key in non_target_common if left_normalized[key] != right_normalized[key]
    }
    exact_identity = {
        key for key in non_target_common if left_normalized[key] == right_normalized[key]
    }
    identity_facts = response.get("identity_facts")
    if not isinstance(identity_facts, list) or len(identity_facts) < 2 or any(
        not isinstance(value, str) or not value.strip() for value in identity_facts
    ):
        errors.append("identity_facts_missing")
    checks = response.get("quality_checks")
    if (
        not isinstance(checks, dict)
        or set(checks) != EXPECTED_QUALITY_CHECKS
        or any(value is not True for value in checks.values())
    ):
        errors.append("quality_checks_not_all_true")

    generated_cards = {
        canonical_card(
            pd.Series(
                {
                    "name": item["name"],
                    "attributes": canonical_json_dumps(item["attributes"]),
                }
            )
        )
        for item in (left, right)
    }
    reference_cards = {
        canonical_card(
            pd.Series(
                {
                    "name": card["name"],
                    "attributes": canonical_json_dumps(card["attributes"]),
                }
            )
        )
        for card in [
            *references["content_references"],
            references["seller_pair_template"]["item_a"],
            references["seller_pair_template"]["item_b"],
        ]
    }
    if generated_cards & reference_cards:
        errors.append("exact_reference_card_copy")
    title_token_set = token_set_ratio(left["name"], right["name"]) / 100.0
    title_ratio = ratio(left["name"], right["name"]) / 100.0
    if title_ratio >= 0.985:
        errors.append("titles_too_similar")
    if title_token_set < 0.35:
        errors.append("titles_too_dissimilar")
    ptype = normalize_text(task["product_type"])
    ptype_tokens = set(words(ptype))
    for side, item in (("a", left), ("b", right)):
        candidate_values = [item["name"], *item["attributes"].values()]
        type_evidence = any(
            ptype in normalize_text(value)
            or token_set_ratio(ptype, normalize_text(value)) >= 85
            or (
                ptype_tokens
                and len(ptype_tokens & set(words(value))) / len(ptype_tokens) >= 0.5
            )
            for value in candidate_values
        )
        if not type_evidence:
            errors.append(f"product_type_not_explicit_{side}")
    metrics = {
        "attribute_count_a": len(left_attrs),
        "attribute_count_b": len(right_attrs),
        "name_word_count_a": len(words(left["name"])),
        "name_word_count_b": len(words(right["name"])),
        "shared_key_count": len(common),
        "exact_shared_non_target_count": len(exact_identity),
        "lexically_conflicting_shared_non_target_count": len(conflicting_common),
        "title_token_set_ratio": title_token_set,
        "title_ratio": title_ratio,
        "values_repaired": bool(application.get("values_repaired")),
    }
    expected_application = task.get("effective_application") or effective_application(task, references)
    if expected_application.get("pre_repair_required"):
        if not application.get("values_repaired"):
            errors.append("required_pre_repair_not_acknowledged")
        if (
            normalize_text(original) != normalize_text(expected_application["original_value"])
            or normalize_text(new) != normalize_text(expected_application["new_value"])
        ):
            errors.append("required_pre_repair_values_changed")
    if errors:
        return None, sorted(set(errors)), metrics
    return response, [], metrics


def sanitize_generated_response(response: dict[str, Any]) -> dict[str, Any]:
    """Drop explicit identifier fields before validation; atoms never target them."""

    sanitized = json.loads(json.dumps(response, ensure_ascii=False))
    for side in ("item_a", "item_b"):
        item = sanitized.get(side)
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("name"), str):
            item["name"] = re.sub(
                r"\s+",
                " ",
                TITLE_IDENTIFIER_RE.sub(" ", item["name"]),
            ).strip()
        attributes = item.get("attributes")
        if isinstance(attributes, dict):
            item["attributes"] = {
                key: value
                for key, value in attributes.items()
                if not IDENTIFIER_RE.search(str(key))
            }
    return sanitized


class QwenClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        system_prompt: str,
        timeout: float,
        retries: int,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.retries = retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.local = threading.local()

    def session(self) -> requests.Session:
        session = getattr(self.local, "session", None)
        if session is None:
            session = requests.Session()
            session.trust_env = False
            self.local.session = session
        return session

    def request(self, prompt: str, seed: int) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "top_p": 0.9,
            "max_tokens": self.max_tokens,
            "seed": int(seed),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self.session().post(self.url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                body = response.json()
                choices = body.get("choices") or []
                if len(choices) != 1:
                    raise ValueError("response must contain one choice")
                content = choices[0].get("message", {}).get("content")
                if not isinstance(content, str):
                    raise ValueError("response content is not text")
                usage = body.get("usage") or {}
                details = usage.get("completion_tokens_details") or {}
                return {
                    "value": parse_json_content(content),
                    "request_attempts": attempt,
                    "latency_seconds": time.perf_counter() - started,
                    "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(usage.get("completion_tokens") or 0),
                    "reasoning_tokens": int(details.get("reasoning_tokens") or 0),
                    "response_id": str(body.get("id") or ""),
                }
            except Exception as error:
                last_error = error
                if attempt < self.retries:
                    time.sleep(min(4.0, 0.5 * 2 ** (attempt - 1)) + random.random() * 0.2)
        raise RuntimeError(f"Qwen request failed: {type(last_error).__name__}: {last_error}") from last_error


def judge_candidate(
    client: QwenClient,
    task: dict[str, Any],
    candidate: dict[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "category": task["category"],
        "product_type": task["product_type"],
        "trusted_target_label": 1,
        "trusted_atomic_rule": {
            "generation_rule_id": task["rule"]["generation_rule_id"],
            "concept": task["rule"]["concept"],
            "canonical_attribute_key": task["rule"]["required_attribute_key"],
        },
        "candidate": {
            "item_a": candidate["item_a"],
            "item_b": candidate["item_b"],
            "target_application": {
                key: candidate["application"][key]
                for key in (
                    "concept",
                    "canonical_attribute_key",
                    "attribute_key_a",
                    "attribute_key_b",
                    "original_value",
                    "new_value",
                )
            },
        },
    }
    raw = client.request(
        json.dumps(payload, ensure_ascii=False, indent=2),
        seed,
    )
    verdict = raw["value"]
    if set(verdict) != {"accept", "fatal_issues"}:
        raise ValueError("judge_shape")
    if not isinstance(verdict["accept"], bool) or not isinstance(verdict["fatal_issues"], list):
        raise ValueError("judge_types")
    if any(not isinstance(issue, str) or not issue.strip() for issue in verdict["fatal_issues"]):
        raise ValueError("judge_issue_shape")
    if verdict["accept"] and verdict["fatal_issues"]:
        raise ValueError("judge_accept_with_issues")
    if not verdict["accept"] and not verdict["fatal_issues"]:
        raise ValueError("judge_reject_without_issue")
    return verdict, raw


def build_tasks(
    source_dir: Path,
    count: int | None,
    seed: int,
    task_indices: list[int] | None = None,
) -> list[dict[str, Any]]:
    metadata = pd.read_parquet(source_dir / "pair_generation_metadata.parquet")
    pairs = pd.read_parquet(source_dir / "pairs.parquet")
    items = pd.read_parquet(source_dir / "items.parquet")
    pair_map = {(int(row.id1), int(row.id2)): row for row in pairs.itertuples(index=False)}
    item_map = {int(row.id): row for row in items.itertuples(index=False)}
    metadata = metadata.sort_values("composition_index", kind="stable")
    if task_indices:
        requested = set(task_indices)
        if len(requested) != len(task_indices):
            raise ValueError("--task-index values must be unique")
        available = set(metadata["composition_index"].astype(int))
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"unknown --task-index values: {missing}")
        metadata = metadata[metadata["composition_index"].astype(int).isin(requested)]
    elif count is not None and count < len(metadata):
        if count < 1:
            raise ValueError("--count must be positive")
        positions = sorted({min(len(metadata) - 1, math.floor(i * len(metadata) / count)) for i in range(count)})
        if len(positions) != count:
            raise RuntimeError("failed to construct deterministic task sample")
        metadata = metadata.iloc[positions]
    tasks: list[dict[str, Any]] = []
    for row in metadata.itertuples(index=False):
        source_key = (int(row.id1), int(row.id2))
        if source_key not in pair_map or source_key[0] not in item_map or source_key[1] not in item_map:
            raise RuntimeError("source composition is inconsistent")
        rules = json.loads(row.rules_json)
        applications = json.loads(row.applications_json)
        if len(rules) != 1 or len(applications) != 1 or int(row.target) != 1:
            raise RuntimeError("rehydration requires atomic target=1 source pairs")
        composition_index = int(row.composition_index)
        nonce = hashlib.blake2s(
            f"{VERSION}:{seed}:{composition_index}".encode("utf-8"), digest_size=16
        ).hexdigest()
        tasks.append(
            {
                "composition_index": composition_index,
                "source_id1": source_key[0],
                "source_id2": source_key[1],
                "category": str(row.category),
                "product_type": str(row.product_type),
                "component": str(row.component),
                "rule": rules[0],
                "application": applications[0],
                "source_semantic_signature": str(row.semantic_signature),
                "source_run_signature": str(row.source_run_signature),
                "diversity_nonce": nonce,
            }
        )
    return tasks


def generate_task(
    task: dict[str, Any],
    references: dict[str, Any],
    client: QwenClient,
    judge_client: QwenClient,
    *,
    seed: int,
    seed_offset: int,
    task_retry_round: int,
    generation_attempts: int,
) -> dict[str, Any]:
    feedback: list[str] = []
    total_usage = Counter()
    started = time.perf_counter()
    for generation_attempt in range(1, generation_attempts + 1):
        prompt = build_prompt(task, references, feedback)
        request_seed = stable_hash64(
            seed,
            canonical_json_dumps(
                {
                    "composition_index": task["composition_index"],
                    "seed_offset": seed_offset,
                    "task_retry_round": task_retry_round,
                    "generation_attempt": generation_attempt,
                }
            ),
        ) % (2**31 - 1)
        try:
            raw = client.request(prompt, request_seed)
        except Exception as error:
            feedback.append(f"request_error:{type(error).__name__}:{error}")
            continue
        for key in ("request_attempts", "prompt_tokens", "completion_tokens", "reasoning_tokens"):
            total_usage[key] += int(raw[key])
        candidate = sanitize_generated_response(raw["value"])
        value, errors, metrics = validate_response(candidate, task, references)
        if value is None:
            feedback.extend(errors)
            continue
        judge_seed = stable_hash64(
            seed,
            canonical_json_dumps(
                {
                    "composition_index": task["composition_index"],
                    "seed_offset": seed_offset,
                    "task_retry_round": task_retry_round,
                    "generation_attempt": generation_attempt,
                    "stage": "semantic_judge",
                }
            ),
        ) % (2**31 - 1)
        try:
            verdict, judge_raw = judge_candidate(
                judge_client,
                task,
                value,
                seed=judge_seed,
            )
        except Exception as error:
            feedback.append(f"judge_error:{type(error).__name__}:{error}")
            continue
        for key in ("request_attempts", "prompt_tokens", "completion_tokens", "reasoning_tokens"):
            total_usage[key] += int(judge_raw[key])
        if not verdict["accept"]:
            feedback.extend(f"semantic_judge:{issue}" for issue in verdict["fatal_issues"][:8])
            continue
        return {
            "status": "accepted",
            "task": task,
            "value": value,
            "metrics": metrics,
            "generation_attempt": generation_attempt,
            "task_retry_round": task_retry_round,
            "task_seed_offset": seed_offset,
            "request_attempts": int(total_usage["request_attempts"]),
            "prompt_tokens": int(total_usage["prompt_tokens"]),
            "completion_tokens": int(total_usage["completion_tokens"]),
            "reasoning_tokens": int(total_usage["reasoning_tokens"]),
            "response_id": raw["response_id"],
            "latency_seconds": time.perf_counter() - started,
            "rejection_history": feedback,
            "semantic_judge": verdict,
        }
    return {
        "status": "error",
        "task": task,
        "task_retry_round": task_retry_round,
        "error": feedback[-1] if feedback else "generation_failed",
        "rejection_history": feedback,
        "latency_seconds": time.perf_counter() - started,
    }


def frames_from_results(results: dict[int, dict[str, Any]], run_signature: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    item_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    for composition_index, result in sorted(results.items()):
        task = result["task"]
        value = result["value"]
        id1 = -4_000_000_000 - composition_index * 2
        id2 = id1 - 1
        for item_id, item in ((id1, value["item_a"]), (id2, value["item_b"])):
            item_rows.append(
                {
                    "id": item_id,
                    "name": item["name"].strip(),
                    "attributes": canonical_json_dumps(item["attributes"]),
                    "category": item["category"],
                }
            )
        pair_rows.append({"id1": id1, "id2": id2, "target": 1})
        application = value["application"]
        metrics = result["metrics"]
        metadata_rows.append(
            {
                "task_index": composition_index,
                "composition_index": composition_index,
                "id1": id1,
                "id2": id2,
                "target": 1,
                "category": task["category"],
                "product_type": task["product_type"],
                "component": task["component"],
                "generation_rule_id": task["rule"]["generation_rule_id"],
                "source_rule_id": task["rule"].get("source_rule_id", ""),
                "concept": task["rule"]["concept"],
                "attribute_key": application["canonical_attribute_key"],
                "attribute_key_a": application["attribute_key_a"],
                "attribute_key_b": application["attribute_key_b"],
                "original_value": application["original_value"],
                "new_value": application["new_value"],
                "source_original_value": task["application"]["original_value"],
                "source_new_value": task["application"]["new_value"],
                "values_repaired": application["values_repaired"],
                "repair_reason": application["repair_reason"],
                "identity_facts_json": canonical_json_dumps(value["identity_facts"]),
                "applications_json": canonical_json_dumps([application]),
                "rules_json": canonical_json_dumps([task["rule"]]),
                "source_id1": task["source_id1"],
                "source_id2": task["source_id2"],
                "source_semantic_signature": task["source_semantic_signature"],
                "source_run_signature": task["source_run_signature"],
                "reference_pair_id": result["references"]["seller_pair_template"]["pair_id"],
                "generation_attempt": result["generation_attempt"],
                "task_retry_round": result["task_retry_round"],
                "task_seed_offset": result["task_seed_offset"],
                "request_attempts": result["request_attempts"],
                "latency_seconds": result["latency_seconds"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "reasoning_tokens": result["reasoning_tokens"],
                "response_id": result["response_id"],
                "rejection_history": canonical_json_dumps(result["rejection_history"]),
                "semantic_judge_json": canonical_json_dumps(result["semantic_judge"]),
                **metrics,
                "rehydration_version": VERSION,
                "reference_policy": REFERENCE_POLICY,
                "run_signature": run_signature,
                "completed_at": datetime.now(UTC).isoformat(),
            }
        )
    return pd.DataFrame(item_rows), pd.DataFrame(pair_rows), pd.DataFrame(metadata_rows)


def validate_dataset(items: pd.DataFrame, pairs: pd.DataFrame, metadata: pd.DataFrame) -> dict[str, Any]:
    issues: list[str] = []
    if len(items) != len(pairs) * 2 or len(metadata) != len(pairs):
        issues.append("dimensions")
    if items["id"].duplicated().any() or metadata["task_index"].duplicated().any():
        issues.append("duplicate_ids_or_tasks")
    item_ids = set(items["id"].astype(int))
    if item_ids != set(pairs["id1"].astype(int)) | set(pairs["id2"].astype(int)):
        issues.append("pair_item_id_mismatch")
    if not pairs["target"].eq(1).all() or not metadata["target"].eq(1).all():
        issues.append("target_not_one")
    item_map = items.set_index("id")
    left_categories = pairs["id1"].map(item_map["category"])
    right_categories = pairs["id2"].map(item_map["category"])
    if not left_categories.equals(right_categories):
        issues.append("cross_category")
    cards = [canonical_card(row) for _, row in items.iterrows()]
    duplicate_cards = len(cards) - len(set(cards))
    if duplicate_cards:
        issues.append("duplicate_global_cards")
    attr_counts = items["attributes"].map(lambda value: len(parse_attributes(value)))
    word_counts = items["name"].map(lambda value: len(words(value)))
    if not attr_counts.between(5, 20).all() or not word_counts.between(5, 18).all():
        issues.append("distribution_bounds")
    return {
        "version": VERSION,
        "valid": not issues,
        "issues": issues,
        "pairs": len(pairs),
        "items": len(items),
        "globally_unique_cards": len(set(cards)),
        "duplicate_global_card_excess": duplicate_cards,
        "attribute_count_mean": float(attr_counts.mean()) if len(attr_counts) else 0.0,
        "attribute_count_median": float(attr_counts.median()) if len(attr_counts) else 0.0,
        "name_word_count_mean": float(word_counts.mean()) if len(word_counts) else 0.0,
        "name_word_count_median": float(word_counts.median()) if len(word_counts) else 0.0,
        "values_repaired": int(metadata["values_repaired"].sum()) if len(metadata) else 0,
    }


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.generation_attempts < 1 or args.task_retries < 0:
        raise ValueError("workers/attempts must be positive and task-retries non-negative")
    source_dir = absolute(args.source_dir)
    output_dir = absolute(args.output_dir)
    pilot_inputs = absolute(args.pilot_inputs)
    pilot_labels = absolute(args.pilot_labels)
    system_prompt = PROMPT.read_text(encoding="utf-8").strip()
    judge_system_prompt = JUDGE_PROMPT.read_text(encoding="utf-8").strip()
    if args.count is not None and args.task_indices:
        raise ValueError("--count and --task-index cannot be combined")
    tasks = build_tasks(source_dir, args.count, args.seed, args.task_indices)
    run_config = {
        "version": VERSION,
        "source_dir": str(source_dir),
        "source_summary_sha256": sha256_file(source_dir / "summary.json"),
        "source_metadata_sha256": sha256_file(source_dir / "pair_generation_metadata.parquet"),
        "pilot_inputs_sha256": sha256_file(pilot_inputs),
        "pilot_labels_sha256": sha256_file(pilot_labels),
        "prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "judge_prompt_sha256": hashlib.sha256(judge_system_prompt.encode("utf-8")).hexdigest(),
        "semantic_judge": True,
        "model": args.model,
        "api_base_url": args.base_url.rstrip("/"),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "generation_attempts": args.generation_attempts,
        "seed": args.seed,
        "requested_tasks": [task["composition_index"] for task in tasks],
        "reference_policy": REFERENCE_POLICY,
    }
    run_signature = sha256_json(run_config)
    reference_index = ReferenceIndex(pilot_inputs, pilot_labels)
    references = {
        task["composition_index"]: reference_index.references(task["rule"], task["product_type"])
        for task in tasks
    }
    client = QwenClient(
        base_url=args.base_url,
        model=args.model,
        system_prompt=system_prompt,
        timeout=args.timeout_seconds,
        retries=args.request_retries,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    judge_client = QwenClient(
        base_url=args.base_url,
        model=args.model,
        system_prompt=judge_system_prompt,
        timeout=args.timeout_seconds,
        retries=args.request_retries,
        temperature=0.0,
        max_tokens=500,
    )

    results: dict[int, dict[str, Any]] = {}
    if (output_dir / "pair_generation_metadata.parquet").is_file():
        previous_metadata = pd.read_parquet(output_dir / "pair_generation_metadata.parquet")
        if set(previous_metadata["run_signature"].astype(str)) != {run_signature}:
            raise RuntimeError("existing checkpoint has a different run signature")
        previous_items = pd.read_parquet(output_dir / "items.parquet").set_index("id")
        previous_pairs = pd.read_parquet(output_dir / "pairs.parquet")
        previous_pair_map = {int(row.id1): row for row in previous_pairs.itertuples(index=False)}
        for row in previous_metadata.itertuples(index=False):
            index = int(row.composition_index)
            if index not in references or int(row.id1) not in previous_pair_map:
                continue
            pair = previous_pair_map[int(row.id1)]
            left = previous_items.loc[int(pair.id1)]
            right = previous_items.loc[int(pair.id2)]
            application = json.loads(row.applications_json)[0]
            results[index] = {
                "status": "accepted",
                "task": next(task for task in tasks if task["composition_index"] == index),
                "value": {
                    "item_a": {"name": left["name"], "attributes": parse_attributes(left["attributes"]), "category": left["category"]},
                    "item_b": {"name": right["name"], "attributes": parse_attributes(right["attributes"]), "category": right["category"]},
                    "application": application,
                    "identity_facts": json.loads(row.identity_facts_json),
                    "quality_checks": {"checkpoint": True},
                },
                "metrics": {key: getattr(row, key) for key in (
                    "attribute_count_a", "attribute_count_b", "name_word_count_a", "name_word_count_b",
                    "shared_key_count", "exact_shared_non_target_count", "title_token_set_ratio",
                    "lexically_conflicting_shared_non_target_count", "title_ratio", "values_repaired",
                )},
                "generation_attempt": int(row.generation_attempt),
                "task_retry_round": int(row.task_retry_round),
                "task_seed_offset": int(row.task_seed_offset),
                "request_attempts": int(row.request_attempts),
                "prompt_tokens": int(row.prompt_tokens),
                "completion_tokens": int(row.completion_tokens),
                "reasoning_tokens": int(row.reasoning_tokens),
                "response_id": str(row.response_id),
                "latency_seconds": float(row.latency_seconds),
                "rejection_history": json.loads(row.rejection_history),
                "semantic_judge": json.loads(row.semantic_judge_json),
                "references": references[index],
            }

    seen_cards = set()
    for result in results.values():
        for side in ("item_a", "item_b"):
            item = result["value"][side]
            seen_cards.add(canonical_card(pd.Series({"name": item["name"], "attributes": canonical_json_dumps(item["attributes"])})))

    task_by_index = {task["composition_index"]: task for task in tasks}
    pending = [index for index in task_by_index if index not in results]
    errors: dict[int, dict[str, Any]] = {}
    started = time.perf_counter()
    last_checkpoint_count = len(results)

    def checkpoint() -> None:
        items, pairs, metadata = frames_from_results(results, run_signature)
        validation = validate_dataset(items, pairs, metadata) if len(results) else {"valid": False, "pairs": 0}
        atomic_parquet(items, output_dir / "items.parquet")
        atomic_parquet(pairs, output_dir / "pairs.parquet")
        atomic_parquet(metadata, output_dir / "pair_generation_metadata.parquet")
        atomic_json(validation, output_dir / "validation_report.json")
        atomic_json([errors[index] for index in sorted(errors)], output_dir / "errors.json")
        summary = {
            **run_config,
            "run_signature": run_signature,
            "count": len(tasks),
            "generated_pairs": len(results),
            "completed": len(results),
            "pending": len(tasks) - len(results),
            "errors": len(errors),
            "workers": args.workers,
            "generation_attempts": args.generation_attempts,
            "task_retries": args.task_retries,
            "task_seed_offsets": (
                sorted({int(value) for value in metadata["task_seed_offset"]})
                if len(metadata)
                else []
            ),
            "request_attempts": int(metadata["request_attempts"].sum()) if len(metadata) else 0,
            "prompt_tokens": int(metadata["prompt_tokens"].sum()) if len(metadata) else 0,
            "completion_tokens": int(metadata["completion_tokens"].sum()) if len(metadata) else 0,
            "reasoning_tokens": int(metadata["reasoning_tokens"].sum()) if len(metadata) else 0,
            "values_repaired": int(metadata["values_repaired"].sum()) if len(metadata) else 0,
            "validation_valid": bool(validation.get("valid")),
            "elapsed_seconds": time.perf_counter() - started,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        atomic_json(summary, output_dir / "summary.json")

    print(
        f"rehydrate requested={len(tasks)} resumed={len(results)} pending={len(pending)} "
        f"workers={args.workers} model={args.model}",
        flush=True,
    )
    for task_retry_round in range(args.task_retries + 1):
        if not pending:
            break
        round_pending = list(pending)
        pending = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    generate_task,
                    task_by_index[index],
                    references[index],
                    client,
                    judge_client,
                    seed=args.seed,
                    seed_offset=args.task_seed_offset,
                    task_retry_round=task_retry_round,
                    generation_attempts=args.generation_attempts,
                ): index
                for index in round_pending
            }
            for future in as_completed(future_map):
                index = future_map[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = {"status": "error", "task": task_by_index[index], "error": f"worker:{type(error).__name__}:{error}", "rejection_history": []}
                if result["status"] == "accepted":
                    duplicate = False
                    local_keys: list[str] = []
                    for side in ("item_a", "item_b"):
                        item = result["value"][side]
                        key = canonical_card(pd.Series({"name": item["name"], "attributes": canonical_json_dumps(item["attributes"])}))
                        if key in seen_cards or key in local_keys:
                            duplicate = True
                        local_keys.append(key)
                    if duplicate:
                        result = {"status": "error", "task": task_by_index[index], "error": "duplicate_global_card", "rejection_history": ["duplicate_global_card"]}
                    else:
                        result["references"] = references[index]
                        results[index] = result
                        seen_cards.update(local_keys)
                        errors.pop(index, None)
                if result["status"] != "accepted":
                    pending.append(index)
                    errors[index] = {
                        "task_index": index,
                        "source_id1": task_by_index[index]["source_id1"],
                        "source_id2": task_by_index[index]["source_id2"],
                        "category": task_by_index[index]["category"],
                        "product_type": task_by_index[index]["product_type"],
                        "error": result.get("error", "unknown"),
                        "rejection_history": result.get("rejection_history", []),
                        "task_retry_round": task_retry_round,
                    }
                if len(results) >= last_checkpoint_count + args.checkpoint_every:
                    elapsed = max(time.perf_counter() - started, 1e-6)
                    print(
                        f"rehydrate saved={len(results)}/{len(tasks)} errors={len(errors)} "
                        f"rate={(len(results) - (len(tasks)-len(round_pending))) / elapsed:.2f}/s",
                        flush=True,
                    )
                    checkpoint()
                    last_checkpoint_count = len(results)
        checkpoint()

    checkpoint()
    print(f"rehydrate complete={len(results)} pending={len(tasks)-len(results)}", flush=True)
    return 0 if len(results) == len(tasks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
