"""Leakage-safe, cheap features and risk/coverage utilities for CatBoost-1.

The module deliberately contains no embedding or neural-model code.  Label-free
Qwen rule definitions may be used as templates, but every label-derived rule
statistic is recomputed inside an outer training fold.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from scipy import sparse


SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)
NUMBER_RE = re.compile(r"(?<![a-zа-яё])\d+(?:[.,]\d+)?", re.IGNORECASE)
CODE_RE = re.compile(
    r"(?<![0-9a-zа-яё])(?=[0-9a-zа-яё._/-]{4,}(?![0-9a-zа-яё]))"
    r"(?=[0-9a-zа-яё._/-]*\d)(?=[0-9a-zа-яё._/-]*[a-zа-яё])"
    r"[0-9a-zа-яё]+(?:[._/-][0-9a-zа-яё]+)*",
    re.IGNORECASE,
)
MEASUREMENT_RE = re.compile(
    r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*"
    r"(tb|тб|gb|гб|mb|мб|kb|кб|kg|кг|mg|мг|g|гр|г|ml|мл|l|л|"
    r"mm|мм|cm|см|m|м|kw|квт|w|вт|v|в|hz|гц|mah|мач|dpi|dpt|дптр|шт)\b",
    re.IGNORECASE,
)
DIMENSION_CHAIN_RE = re.compile(
    r"(?<![\w.])((?:\d+(?:[.,]\d+)?\s*[xх×*]\s*){1,3}\d+(?:[.,]\d+)?)\s*"
    r"(mm|мм|cm|см|m|м)\b",
    re.IGNORECASE,
)

UNIT_SCALE: dict[str, tuple[str, float]] = {
    "tb": ("bytes", 1024.0**4), "тб": ("bytes", 1024.0**4),
    "gb": ("bytes", 1024.0**3), "гб": ("bytes", 1024.0**3),
    "mb": ("bytes", 1024.0**2), "мб": ("bytes", 1024.0**2),
    "kb": ("bytes", 1024.0), "кб": ("bytes", 1024.0),
    "kg": ("weight_g", 1000.0), "кг": ("weight_g", 1000.0),
    "mg": ("weight_g", 0.001), "мг": ("weight_g", 0.001),
    "g": ("weight_g", 1.0), "гр": ("weight_g", 1.0), "г": ("weight_g", 1.0),
    "ml": ("volume_ml", 1.0), "мл": ("volume_ml", 1.0),
    "l": ("volume_ml", 1000.0), "л": ("volume_ml", 1000.0),
    "mm": ("length_mm", 1.0), "мм": ("length_mm", 1.0),
    "cm": ("length_mm", 10.0), "см": ("length_mm", 10.0),
    "m": ("length_mm", 1000.0), "м": ("length_mm", 1000.0),
    "kw": ("power_w", 1000.0), "квт": ("power_w", 1000.0),
    "w": ("power_w", 1.0), "вт": ("power_w", 1.0),
    "v": ("voltage_v", 1.0), "в": ("voltage_v", 1.0),
    "hz": ("frequency_hz", 1.0), "гц": ("frequency_hz", 1.0),
    "mah": ("capacity_mah", 1.0), "мач": ("capacity_mah", 1.0),
    "dpi": ("optical", 1.0), "dpt": ("optical", 1.0), "дптр": ("optical", 1.0),
    "шт": ("pack_count", 1.0),
}

CONCEPT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("brand", re.compile(r"бренд|марка|производител|brand|vendor", re.I)),
    ("model_number", re.compile(r"модел|артикул|парт.?номер|part.?number|sku|mpn|oem|код товар|model", re.I)),
    ("ram_storage", re.compile(r"оперативн|встроенн.*памят|объем памяти|накопител|storage|ram|rom|ssd|hdd", re.I)),
    ("optical", re.compile(r"диоптр|оптическ|фокус|линз|рефракц|dpi", re.I)),
    ("dimensions", re.compile(r"габарит|длина|ширина|высота|диаметр|толщина|dimensions?", re.I)),
    ("volume", re.compile(r"объ[её]м|литраж|volume", re.I)),
    ("weight", re.compile(r"вес|масса|weight", re.I)),
    ("pack_count", re.compile(r"количеств|комплект|упаков|штук|шт\.?$|pack|count|quantity", re.I)),
    ("power", re.compile(r"мощност|потребляемая мощность|power|ватт", re.I)),
    ("size", re.compile(r"размер|ростов|обхват|size", re.I)),
    ("color", re.compile(r"цвет|оттенок|color", re.I)),
    ("material", re.compile(r"материал|состав|сырь[её]|material", re.I)),
]

NUMERIC_TYPES = (
    "size", "ram_storage", "volume", "weight", "pack_count", "power",
    "dimensions", "optical", "model_number", "voltage", "frequency", "capacity",
)

REGIME_BY_CATEGORY = {
    "Одежда": "variant_tolerant",
    "Обувь": "variant_tolerant",
    "Галантерея и аксессуары": "variant_tolerant",
    "Ювелирные изделия": "variant_tolerant",
    "Электроника": "configuration_sensitive",
    "Бытовая техника": "configuration_sensitive",
    "Автотовары": "configuration_sensitive",
    "Строительство и ремонт": "configuration_sensitive",
    "Музыкальные инструменты": "configuration_sensitive",
    "Мебель": "numeric_spec_sensitive",
    "Спорт и отдых": "numeric_spec_sensitive",
    "Канцелярские товары": "numeric_spec_sensitive",
    "Детские товары": "numeric_spec_sensitive",
    "Хобби и творчество": "numeric_spec_sensitive",
    "Дом и сад": "numeric_spec_sensitive",
    "Продукты питания": "quantity_volume_sensitive",
    "Бытовая химия": "quantity_volume_sensitive",
    "Красота и гигиена": "quantity_volume_sensitive",
    "Товары для животных": "quantity_volume_sensitive",
    "Аптека": "quantity_volume_sensitive",
}


def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().replace("ё", "е")
    return SPACE_RE.sub(" ", text).strip()


def _flatten(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [normalize_text(json.dumps(value, ensure_ascii=False, sort_keys=True))]
    if isinstance(value, (list, tuple, set)):
        return [item for nested in value for item in _flatten(nested)]
    text = normalize_text(value)
    return [text] if text else []


def parse_attributes(raw: Any) -> dict[str, tuple[str, ...]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, tuple[str, ...]] = {}
    for key, value in raw.items():
        normalized_key = normalize_text(key)
        values = tuple(sorted(set(_flatten(value))))
        if normalized_key and values:
            result[normalized_key] = values
    return result


def concept_for_key(key: str, learned_map: Mapping[str, str] | None = None) -> str | None:
    key = normalize_text(key)
    if learned_map and key in learned_map:
        return learned_map[key]
    # Guard English concept names before the broad multilingual regexes below.
    # Substring matching used to classify ``frame_*`` as RAM, ``country_*`` as
    # a count and ``optical_power`` as generic power.
    ascii_tokens = set(re.findall(r"[a-z0-9]+", key))
    if "optical" in ascii_tokens or "lens" in ascii_tokens or "pupillary" in ascii_tokens or "frame" in ascii_tokens:
        return "optical"
    if "country" in ascii_tokens and "count" not in ascii_tokens:
        return None
    for concept, pattern in CONCEPT_PATTERNS:
        if pattern.search(key):
            return concept
    return None


def _legacy_concept_for_key(key: str, learned_map: Mapping[str, str] | None = None) -> str | None:
    """Reproduce the frozen CatBoost-1 feature contract, including old regex order."""

    key = normalize_text(key)
    if learned_map and key in learned_map:
        return learned_map[key]
    for concept, pattern in CONCEPT_PATTERNS:
        if pattern.search(key):
            return concept
    return None


def semantic_family(value: str | None) -> str | None:
    if not value:
        return None
    normalized = normalize_text(value).replace("_", " ")
    ascii_tokens = set(re.findall(r"[a-z0-9]+", normalized))
    if "optical" in ascii_tokens or "lens" in ascii_tokens or "pupillary" in ascii_tokens or "frame" in ascii_tokens:
        return "optical"
    if "country" in ascii_tokens and "count" not in ascii_tokens:
        return None
    for family, pattern in CONCEPT_PATTERNS:
        if pattern.search(normalized):
            return family
    return value if value in {"brand", "model_number", "color", "material", *NUMERIC_TYPES} else None


def _legacy_semantic_family(value: str | None) -> str | None:
    if not value:
        return None
    normalized = normalize_text(value).replace("_", " ")
    for family, pattern in CONCEPT_PATTERNS:
        if pattern.search(normalized):
            return family
    return value if value in {"brand", "model_number", "color", "material", *NUMERIC_TYPES} else None


def build_label_free_attribute_concept_map(
    accepted_facts_path: Path,
    *,
    min_support: int = 3,
    min_purity: float = 0.70,
) -> tuple[dict[str, str], pd.DataFrame]:
    """Learn raw attribute-name -> concept aliases without reading any labels."""

    facts = pd.read_parquet(
        accepted_facts_path,
        columns=["concept", "sanitized_fact_json"],
    )
    counts: Counter[tuple[str, str]] = Counter()
    for concept, payload in facts.itertuples(index=False, name=None):
        try:
            fact = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            continue
        for side in ("a", "b"):
            evidence = (fact.get(side) or {}).get("evidence") or []
            for entry in evidence:
                if entry.get("source") != "attribute" or not entry.get("raw_attribute_name"):
                    continue
                key = normalize_text(entry["raw_attribute_name"])
                canonical = normalize_text(concept)
                if key and canonical:
                    counts[(key, canonical)] += 1
    by_key: dict[str, Counter[str]] = defaultdict(Counter)
    for (key, concept), count in counts.items():
        by_key[key][concept] = count
    rows: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}
    for key, concept_counts in sorted(by_key.items()):
        concept, support = concept_counts.most_common(1)[0]
        total = sum(concept_counts.values())
        purity = support / total
        accepted = support >= min_support and purity >= min_purity
        rows.append({
            "attribute_key": key,
            "concept": concept,
            "support": support,
            "total": total,
            "purity": purity,
            "accepted": accepted,
        })
        if accepted:
            mapping[key] = concept
    return mapping, pd.DataFrame(rows)


def load_label_free_rule_registry(
    path: Path,
    *,
    allowed_relations: set[str] | None = None,
    allowed_roles: set[str] | None = None,
) -> tuple[dict[tuple[str, str], int], pd.DataFrame]:
    """Read only definition columns; stored effects/support are intentionally ignored."""

    columns = ["rule_id", "canonical_rule", "concept", "relation"]
    if allowed_roles is not None:
        columns.append("rule_role")
    definitions = pd.read_parquet(path, columns=columns).drop_duplicates("rule_id")
    definitions["concept"] = definitions["concept"].map(normalize_text)
    definitions["relation"] = definitions["relation"].map(normalize_text)
    if allowed_relations is not None:
        definitions = definitions.loc[definitions["relation"].isin(allowed_relations)]
    if allowed_roles is not None:
        definitions = definitions.loc[definitions["rule_role"].isin(allowed_roles)]
    definitions = definitions.sort_values("rule_id", kind="stable").reset_index(drop=True)
    registry: dict[tuple[str, str], int] = {}
    for index, row in definitions.iterrows():
        registry.setdefault((row["concept"], row["relation"]), int(index))
    return registry, definitions


@dataclass(frozen=True)
class ItemRecord:
    title: str
    tokens: frozenset[str]
    codes: frozenset[str]
    attributes: dict[str, tuple[str, ...]]
    concepts: dict[str, frozenset[str]]
    typed_values: dict[str, frozenset[str]]


def _number_token(number: str) -> str:
    value = float(number.replace(",", "."))
    return f"{value:.9g}"


def _typed_measurements(key: str, values: Iterable[str], title: str, concept: str | None) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    payload = " ".join(values)
    if concept in NUMERIC_TYPES or concept in {"model_number"}:
        for number in NUMBER_RE.findall(payload):
            result[str(concept)].add(_number_token(number))
    for match in DIMENSION_CHAIN_RE.finditer(payload):
        dimension, scale = UNIT_SCALE[match.group(2).casefold().replace("ё", "е")]
        if dimension != "length_mm":
            continue
        normalized = [
            f"{float(number.replace(',', '.')) * scale:.9g}"
            for number in NUMBER_RE.findall(match.group(1))
        ]
        result["dimensions"].add("x".join(normalized))
    for match in MEASUREMENT_RE.finditer(payload):
        number = float(match.group(1).replace(",", "."))
        dimension, scale = UNIT_SCALE[match.group(2).casefold().replace("ё", "е")]
        typed = {
            "bytes": "ram_storage", "weight_g": "weight", "volume_ml": "volume",
            "length_mm": "dimensions", "power_w": "power", "voltage_v": "voltage",
            "frequency_hz": "frequency", "capacity_mah": "capacity",
            "optical": "optical", "pack_count": "pack_count",
        }[dimension]
        result[typed].add(f"{number * scale:.9g}")
    # Titles contribute only explicit unit-bearing quantities and model-like codes.
    if key == "__title__":
        for code in CODE_RE.findall(title):
            digits = "|".join(NUMBER_RE.findall(code))
            if digits:
                result["model_number"].add(normalize_text(code))
    return result


def build_item_record(
    title: Any,
    raw_attributes: Any,
    learned_concepts: Mapping[str, str],
    *,
    legacy_semantics: bool = False,
) -> ItemRecord:
    normalized_title = normalize_text(title)
    attributes = parse_attributes(raw_attributes)
    concepts: dict[str, set[str]] = defaultdict(set)
    typed: dict[str, set[str]] = defaultdict(set)
    for key, values in attributes.items():
        concept = (
            _legacy_concept_for_key(key, learned_concepts)
            if legacy_semantics else concept_for_key(key, learned_concepts)
        )
        if concept:
            concepts[concept].update(values)
        family = (
            _legacy_semantic_family(concept) or _legacy_concept_for_key(key)
            if legacy_semantics else semantic_family(concept) or concept_for_key(key)
        )
        if family and family != concept:
            concepts[family].update(values)
        for typed_family, family_values in _typed_measurements(key, values, normalized_title, family).items():
            typed[typed_family].update(family_values)
    for family, family_values in _typed_measurements("__title__", [normalized_title], normalized_title, None).items():
        typed[family].update(family_values)
    codes = set(CODE_RE.findall(normalized_title))
    codes.update(concepts.get("model_number", set()))
    return ItemRecord(
        title=normalized_title,
        tokens=frozenset(TOKEN_RE.findall(normalized_title)),
        codes=frozenset(codes),
        attributes=attributes,
        concepts={key: frozenset(value) for key, value in concepts.items()},
        typed_values={key: frozenset(value) for key, value in typed.items()},
    )


def _set_state(first: frozenset[str] | set[str], second: frozenset[str] | set[str]) -> tuple[float, float, float, float]:
    both = bool(first) and bool(second)
    overlap = bool(first & second)
    return float(both), float(overlap), float(both and first != second), float(bool(first) ^ bool(second))


def _jaccard(first: set[str] | frozenset[str], second: set[str] | frozenset[str]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def _relation_for_values(first: frozenset[str], second: frozenset[str]) -> str | None:
    if not first or not second:
        return "missing_one_side" if bool(first) ^ bool(second) else None
    if first == second:
        return None
    if first & second:
        return "specificity_difference"
    first_text, second_text = " ".join(sorted(first)), " ".join(sorted(second))
    if first_text in second_text or second_text in first_text:
        return "specificity_difference"
    return "different_value"


def extract_pair_features(
    pairs: pd.DataFrame,
    items: pd.DataFrame,
    learned_concepts: Mapping[str, str],
    rule_registry: Mapping[tuple[str, str], int],
    *,
    legacy_semantics: bool = False,
) -> tuple[pd.DataFrame, list[list[int]]]:
    """Build label-free pair features and fired rule indices."""

    required_items = pd.unique(pairs[["id1", "id2"]].to_numpy().reshape(-1))
    selected = items.loc[items["id"].isin(required_items), ["id", "name", "attributes", "category"]].copy()
    if selected["id"].duplicated().any() or len(selected) != len(required_items):
        missing = len(required_items) - selected["id"].nunique()
        raise ValueError(f"Item lookup is not one-to-one; missing={missing}")
    records = [
        build_item_record(
            name, attributes, learned_concepts, legacy_semantics=legacy_semantics
        )
        for name, attributes in selected[["name", "attributes"]].itertuples(index=False, name=None)
    ]
    position = pd.Series(np.arange(len(selected), dtype=np.int32), index=selected["id"].to_numpy())
    left_pos = position.loc[pairs["id1"]].to_numpy(dtype=np.int32)
    right_pos = position.loc[pairs["id2"]].to_numpy(dtype=np.int32)
    categories = selected["category"].astype(str).to_numpy()
    pair_categories = categories[left_pos]
    if not np.array_equal(pair_categories, categories[right_pos]):
        raise ValueError("Cross-category pairs are not supported")

    rows: list[dict[str, Any]] = []
    fired_rules: list[list[int]] = []
    for lp, rp, category in zip(left_pos, right_pos, pair_categories):
        first, second = records[int(lp)], records[int(rp)]
        common_keys = set(first.attributes) & set(second.attributes)
        union_keys = set(first.attributes) | set(second.attributes)
        exact_attrs = sum(bool(set(first.attributes[k]) & set(second.attributes[k])) for k in common_keys)
        conflicting_attrs = len(common_keys) - exact_attrs
        attribute_similarities = [
            max(
                fuzz.ratio(value_a, value_b) / 100.0
                for value_a in first.attributes[key]
                for value_b in second.attributes[key]
            )
            for key in common_keys
        ]
        title_numbers_a = set(NUMBER_RE.findall(first.title))
        title_numbers_b = set(NUMBER_RE.findall(second.title))
        row: dict[str, Any] = {
            "title_exact": float(first.title == second.title),
            "title_ratio": fuzz.ratio(first.title, second.title) / 100.0,
            "title_token_set": fuzz.token_set_ratio(first.title, second.title) / 100.0,
            "title_token_sort": fuzz.token_sort_ratio(first.title, second.title) / 100.0,
            "title_wratio": fuzz.WRatio(first.title, second.title) / 100.0,
            "title_token_jaccard": _jaccard(first.tokens, second.tokens),
            "title_length_ratio": min(len(first.title), len(second.title)) / max(1, len(first.title), len(second.title)),
            "title_length_delta": abs(len(first.title) - len(second.title)),
            "title_number_overlap": len(title_numbers_a & title_numbers_b),
            "title_number_jaccard": _jaccard(title_numbers_a, title_numbers_b),
            "attribute_count_a": len(first.attributes),
            "attribute_count_b": len(second.attributes),
            "attribute_count_delta": abs(len(first.attributes) - len(second.attributes)),
            "attribute_common_keys": len(common_keys),
            "attribute_comparable": len(common_keys),
            "attribute_key_jaccard": len(common_keys) / max(1, len(union_keys)),
            "attribute_exact": exact_attrs,
            "attribute_conflict": conflicting_attrs,
            "attribute_exact_ratio": exact_attrs / max(1, len(common_keys)),
            "attribute_conflict_ratio": conflicting_attrs / max(1, len(common_keys)),
            "attribute_value_similarity_mean": float(np.mean(attribute_similarities)) if attribute_similarities else 0.0,
            "attribute_value_similarity_min": float(np.min(attribute_similarities)) if attribute_similarities else 0.0,
            "attribute_missing_a": len(set(second.attributes) - set(first.attributes)),
            "attribute_missing_b": len(set(first.attributes) - set(second.attributes)),
            "category": str(category),
        }
        for concept in ("brand", "model_number", "color", "material"):
            both, match, conflict, one_missing = _set_state(
                first.concepts.get(concept, frozenset()), second.concepts.get(concept, frozenset())
            )
            prefix = "model_code" if concept == "model_number" else concept
            row[f"{prefix}_both"] = both
            row[f"{prefix}_match"] = match
            row[f"{prefix}_conflict"] = conflict
            row[f"{prefix}_one_missing"] = one_missing
        code_both, code_match, code_conflict, code_missing = _set_state(first.codes, second.codes)
        row.update({
            "title_code_both": code_both,
            "title_code_match": code_match,
            "title_code_conflict": code_conflict,
            "title_code_one_missing": code_missing,
            "title_code_jaccard": _jaccard(first.codes, second.codes),
        })
        conflict_types: list[str] = []
        for family in NUMERIC_TYPES:
            values_a = first.typed_values.get(family, frozenset())
            values_b = second.typed_values.get(family, frozenset())
            both, match, conflict, one_missing = _set_state(values_a, values_b)
            row[f"num_{family}_both"] = both
            row[f"num_{family}_match"] = match
            row[f"num_{family}_conflict"] = conflict
            row[f"num_{family}_one_missing"] = one_missing
            row[f"num_{family}_overlap_count"] = len(values_a & values_b)
            row[f"num_{family}_unmatched_count"] = len(values_a ^ values_b)
            if conflict:
                conflict_types.append(family)
        signature = "+".join(conflict_types) if conflict_types else "none"
        primary = conflict_types[0] if conflict_types else "none"
        regime = REGIME_BY_CATEGORY.get(str(category), "unknown_mixed")
        row.update({
            "primary_conflict_type": primary,
            "conflict_signature": signature,
            "category_primary_conflict": f"{category}||{primary}",
            "category_conflict_signature": f"{category}||{signature}",
            "matching_regime": regime,
            "regime_conflict_signature": f"{regime}||{signature}",
        })
        pair_rule_indices: set[int] = set()
        all_concepts = set(first.concepts) | set(second.concepts)
        for concept in all_concepts:
            relation = _relation_for_values(
                first.concepts.get(concept, frozenset()), second.concepts.get(concept, frozenset())
            )
            if relation is not None and (concept, relation) in rule_registry:
                pair_rule_indices.add(rule_registry[(concept, relation)])
        row["rule_fired_count"] = len(pair_rule_indices)
        row["rule_any_fired"] = float(bool(pair_rule_indices))
        rows.append(row)
        fired_rules.append(sorted(pair_rule_indices))
    frame = pd.DataFrame(rows)
    numeric = frame.select_dtypes(exclude=["object"]).columns
    frame[numeric] = frame[numeric].astype(np.float32)
    return frame, fired_rules


def rule_lists_to_csr(rule_lists: Sequence[Sequence[int]], n_rules: int) -> sparse.csr_matrix:
    indptr = np.zeros(len(rule_lists) + 1, dtype=np.int64)
    indices: list[int] = []
    for row, values in enumerate(rule_lists):
        indices.extend(sorted(set(values)))
        indptr[row + 1] = len(indices)
    return sparse.csr_matrix(
        (np.ones(len(indices), dtype=np.float32), np.asarray(indices, dtype=np.int32), indptr),
        shape=(len(rule_lists), n_rules),
    )


def category_rule_matrix(
    rule_lists: Sequence[Sequence[int]], categories: Sequence[str]
) -> tuple[sparse.csr_matrix, list[str]]:
    tokens = [[f"{category}||{rule}" for rule in rules] for category, rules in zip(categories, rule_lists)]
    vocabulary = sorted({token for row in tokens for token in row})
    lookup = {token: index for index, token in enumerate(vocabulary)}
    encoded = [[lookup[token] for token in row] for row in tokens]
    return rule_lists_to_csr(encoded, len(vocabulary)), vocabulary


def _logit(values: np.ndarray | float) -> np.ndarray:
    values = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values))


def _aggregate_rule_evidence(
    matrix: sparse.csr_matrix,
    support: np.ndarray,
    positive: np.ndarray,
    baseline: np.ndarray | float,
    prior_strength: float,
    min_support: int = 1,
    effect_clip: float | None = None,
) -> pd.DataFrame:
    eligible = support >= min_support
    probability = (positive + prior_strength * baseline) / np.maximum(support + prior_strength, 1e-12)
    effect = (_logit(probability) - _logit(baseline)) * (support / (support + prior_strength))
    effect = np.where(eligible, effect, 0.0)
    if effect_clip is not None:
        effect = np.clip(effect, -effect_clip, effect_clip)
    positive_effect = np.maximum(effect, 0.0)
    negative_effect = np.maximum(-effect, 0.0)
    fired = np.asarray(matrix @ eligible.astype(np.float64)).ravel()
    frame = pd.DataFrame({
        "rule_evidence_fired": fired,
        "rule_evidence_support_sum_log": np.asarray(matrix @ (np.log1p(support) * eligible)).ravel(),
        "rule_positive_evidence_sum": np.asarray(matrix @ positive_effect).ravel(),
        "rule_negative_evidence_sum": np.asarray(matrix @ negative_effect).ravel(),
        "rule_positive_count": np.asarray(matrix @ (effect > 0).astype(np.float64)).ravel(),
        "rule_negative_count": np.asarray(matrix @ (effect < 0).astype(np.float64)).ravel(),
    }, dtype=np.float32).assign(
        rule_evidence_agreement=lambda x: ((x.rule_positive_count > 0) ^ (x.rule_negative_count > 0)).astype(np.float32),
        rule_evidence_disagreement=lambda x: ((x.rule_positive_count > 0) & (x.rule_negative_count > 0)).astype(np.float32),
    )
    frame["rule_positive_evidence_mean"] = (
        frame["rule_positive_evidence_sum"] / frame["rule_positive_count"].clip(lower=1)
    ).astype(np.float32)
    frame["rule_negative_evidence_mean"] = (
        frame["rule_negative_evidence_sum"] / frame["rule_negative_count"].clip(lower=1)
    ).astype(np.float32)
    return frame


def fold_rule_evidence(
    matrix: sparse.csr_matrix,
    target: np.ndarray,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    *,
    prior_strength: float = 20.0,
    min_support: int = 1,
    effect_clip: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OOF-safe target evidence: LOO for train, full outer-train for validation."""

    train_matrix = matrix[train_indices].tocsr()
    valid_matrix = matrix[valid_indices].tocsr()
    train_target = target[train_indices].astype(np.float64)
    support = np.asarray(train_matrix.sum(axis=0)).ravel().astype(np.float64)
    positive = np.asarray(train_matrix.T @ train_target).ravel().astype(np.float64)
    baseline = float(train_target.mean())
    valid = _aggregate_rule_evidence(
        valid_matrix, support, positive, baseline, prior_strength, min_support, effect_clip
    )

    # For each training row, subtract its own contribution from every fired rule.
    # There are only two possible vectors (row target 0 or 1).
    n_train = len(train_indices)
    train_parts: list[pd.DataFrame] = []
    for label in (0, 1):
        local = np.flatnonzero(train_target == label)
        if not len(local):
            continue
        loo_support = np.maximum(support - 1.0, 0.0)
        loo_positive = positive - float(label)
        loo_baseline = (train_target.sum() - label) / max(1, n_train - 1)
        part = _aggregate_rule_evidence(
            train_matrix[local], loo_support, loo_positive, loo_baseline,
            prior_strength, min_support, effect_clip,
        )
        part.index = local
        train_parts.append(part)
    train = pd.concat(train_parts).sort_index().reset_index(drop=True)
    return train, valid


def crossfit_fold_rule_evidence(
    matrix: sparse.csr_matrix,
    target: np.ndarray,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    inner_folds: np.ndarray,
    *,
    prior_strength: float = 50.0,
    min_support: int = 50,
    effect_clip: float | None = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nested target encoding for an outer fold, with no row using its own label."""

    if len(inner_folds) != len(train_indices):
        raise ValueError("inner_folds must align with train_indices")
    outer_train_matrix = matrix[train_indices].tocsr()
    outer_valid_matrix = matrix[valid_indices].tocsr()
    outer_target = target[train_indices].astype(np.float64)

    support = np.asarray(outer_train_matrix.sum(axis=0)).ravel().astype(np.float64)
    positive = np.asarray(outer_train_matrix.T @ outer_target).ravel().astype(np.float64)
    valid = _aggregate_rule_evidence(
        outer_valid_matrix, support, positive, float(outer_target.mean()),
        prior_strength, min_support, effect_clip,
    )

    encoded_parts: list[pd.DataFrame] = []
    for inner_fold in sorted(np.unique(inner_folds)):
        inner_valid = np.flatnonzero(inner_folds == inner_fold)
        inner_train = np.flatnonzero(inner_folds != inner_fold)
        fit_matrix = outer_train_matrix[inner_train]
        query_matrix = outer_train_matrix[inner_valid]
        fit_target = outer_target[inner_train]
        inner_support = np.asarray(fit_matrix.sum(axis=0)).ravel().astype(np.float64)
        inner_positive = np.asarray(fit_matrix.T @ fit_target).ravel().astype(np.float64)
        part = _aggregate_rule_evidence(
            query_matrix, inner_support, inner_positive, float(fit_target.mean()),
            prior_strength, min_support, effect_clip,
        )
        part.index = inner_valid
        encoded_parts.append(part)
    train = pd.concat(encoded_parts).sort_index().reset_index(drop=True)
    if len(train) != len(train_indices):
        raise AssertionError("Nested rule encoding did not cover every outer-train row")
    return train, valid


def rule_family_masks(definitions: pd.DataFrame) -> dict[str, np.ndarray]:
    """Map clean rule columns to stable semantic families, retaining an other bucket."""

    families: dict[str, list[int]] = defaultdict(list)
    for index, concept in enumerate(definitions["concept"].astype(str)):
        family = semantic_family(concept) or "other"
        families[family].append(index)
    return {
        family: np.asarray(indices, dtype=np.int32)
        for family, indices in sorted(families.items())
    }


def category_balanced_weights(categories: Sequence[str]) -> np.ndarray:
    values = pd.Series(categories, dtype="string")
    counts = values.value_counts()
    weights = values.map({key: len(values) / (len(counts) * count) for key, count in counts.items()})
    return weights.to_numpy(dtype=np.float32)


def wilson_upper(errors: int | np.ndarray, accepted: int | np.ndarray, confidence: float = 0.95) -> np.ndarray:
    """Two-sided Wilson interval's upper endpoint (z=1.95996 at 95%)."""

    if confidence != 0.95:
        from scipy.stats import norm

        z = float(norm.ppf(0.5 + confidence / 2.0))
    else:
        z = 1.959963984540054
    errors_array = np.asarray(errors, dtype=np.float64)
    accepted_array = np.asarray(accepted, dtype=np.float64)
    result = np.ones(np.broadcast(errors_array, accepted_array).shape, dtype=np.float64)
    valid = accepted_array > 0
    n = accepted_array[valid]
    p = np.broadcast_to(errors_array, result.shape)[valid] / n
    denominator = 1.0 + z * z / n
    center = p + z * z / (2.0 * n)
    radius = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    result[valid] = (center + radius) / denominator
    return result


def threshold_states(scores: np.ndarray, target: np.ndarray, side: str) -> pd.DataFrame:
    """Return tie-safe states for strict p<t (negative) or p>t (positive)."""

    scores = np.asarray(scores, dtype=np.float64)
    target = np.asarray(target, dtype=np.int8)
    order = np.argsort(scores, kind="stable")
    if side == "positive":
        order = order[::-1]
        mistakes = 1 - target[order]
    elif side == "negative":
        mistakes = target[order]
    else:
        raise ValueError("side must be 'negative' or 'positive'")
    ordered_scores = scores[order]
    ends = np.r_[np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]) + 1, len(scores)]
    accepted = ends.astype(np.int64)
    errors = np.cumsum(mistakes, dtype=np.int64)[ends - 1]
    boundary = ordered_scores[ends - 1]
    threshold = np.nextafter(boundary, np.inf if side == "negative" else -np.inf)
    frame = pd.DataFrame({
        "side": side,
        "threshold": threshold,
        "boundary_score": boundary,
        "accepted": accepted,
        "errors": errors,
        "coverage": accepted / len(scores),
        "empirical_error": errors / accepted,
        "error_ucb_95": wilson_upper(errors, accepted),
    })
    empty = pd.DataFrame([{
        "side": side,
        "threshold": 0.0 if side == "negative" else 1.0,
        "boundary_score": np.nan,
        "accepted": 0,
        "errors": 0,
        "coverage": 0.0,
        "empirical_error": 0.0,
        "error_ucb_95": 1.0,
    }])
    return pd.concat([empty, frame], ignore_index=True)


def best_side_state(states: pd.DataFrame, risk_limit: float) -> dict[str, Any]:
    eligible = states.loc[states["error_ucb_95"] < risk_limit]
    if eligible.empty:
        return states.iloc[0].to_dict()
    return eligible.sort_values(["accepted", "errors"], ascending=[False, True]).iloc[0].to_dict()


def best_total_state(
    negative_states: pd.DataFrame,
    positive_states: pd.DataFrame,
    risk_limit: float,
    total_examples: int,
) -> dict[str, Any]:
    """Maximize union of the two non-overlapping score tails at a Wilson bound."""

    # Wilson UCB is always above the empirical rate, so no feasible state can
    # have errors >= risk_limit * total_examples.
    max_errors = max(0, int(math.ceil(risk_limit * total_examples)))
    neg = negative_states.loc[negative_states.errors <= max_errors].copy()
    pos = positive_states.loc[positive_states.errors <= max_errors].copy()
    # For an exact error count only the longest tail can dominate at strict risk.
    neg = neg.sort_values("accepted").groupby("errors", as_index=False).tail(1).set_index("errors")
    pos = pos.sort_values("accepted").groupby("errors", as_index=False).tail(1).set_index("errors")
    best: dict[str, Any] | None = None
    pos_errors_all = pos.index.to_numpy(dtype=np.int64)
    pos_accepted_all = pos["accepted"].to_numpy(dtype=np.int64)
    pos_threshold_all = pos["threshold"].to_numpy(dtype=np.float64)
    for neg_errors, neg_row in neg.iterrows():
        eligible = pos_errors_all <= max_errors - int(neg_errors)
        accepted = int(neg_row.accepted) + pos_accepted_all[eligible]
        errors = int(neg_errors) + pos_errors_all[eligible]
        valid = (accepted > 0) & (accepted <= total_examples)
        ucb = wilson_upper(errors, accepted)
        valid &= ucb < risk_limit
        if not valid.any():
            continue
        local = np.flatnonzero(eligible)
        candidates = np.flatnonzero(valid)
        # Largest accepted tail wins; errors break exact-coverage ties.
        ordering = np.lexsort((errors[candidates], -accepted[candidates]))
        chosen = int(candidates[ordering[0]])
        pos_index = int(local[chosen])
        candidate = {
            "accepted": int(accepted[chosen]),
            "errors": int(errors[chosen]),
            "coverage": float(accepted[chosen] / total_examples),
            "empirical_error": float(errors[chosen] / accepted[chosen]),
            "error_ucb_95": float(ucb[chosen]),
            "t_neg": float(neg_row.threshold),
            "t_pos": float(pos_threshold_all[pos_index]),
            "neg_accepted": int(neg_row.accepted),
            "neg_errors": int(neg_errors),
            "neg_coverage": float(neg_row.accepted / total_examples),
            "pos_accepted": int(pos_accepted_all[pos_index]),
            "pos_errors": int(pos_errors_all[pos_index]),
            "pos_coverage": float(pos_accepted_all[pos_index] / total_examples),
        }
        if best is None or (candidate["accepted"], -candidate["errors"]) > (best["accepted"], -best["errors"]):
            best = candidate
    return best or {
        "accepted": 0, "errors": 0, "coverage": 0.0, "empirical_error": 0.0,
        "error_ucb_95": 1.0, "t_neg": 0.0, "t_pos": 1.0,
        "neg_accepted": 0, "neg_errors": 0, "neg_coverage": 0.0,
        "pos_accepted": 0, "pos_errors": 0, "pos_coverage": 0.0,
    }


VARIANT_COLUMNS = {
    "V1_global": {"drop_categorical": True, "category_rule": False, "regime": False},
    "V2_global_category": {"drop_categorical": False, "category_rule": False, "regime": False},
    "V3_category_aware": {"drop_categorical": False, "category_rule": True, "regime": False},
    "V4_matching_regimes": {"drop_categorical": False, "category_rule": True, "regime": True},
}


def variant_frame(
    base: pd.DataFrame,
    global_evidence: pd.DataFrame,
    category_evidence: pd.DataFrame | None,
    variant: str,
) -> tuple[pd.DataFrame, list[str]]:
    spec = VARIANT_COLUMNS[variant]
    numeric_base = base.select_dtypes(exclude=["object"]).copy()
    frame = pd.concat([numeric_base.reset_index(drop=True), global_evidence.add_prefix("global_").reset_index(drop=True)], axis=1)
    categorical: list[str] = []
    if not spec["drop_categorical"]:
        frame["category"] = base["category"].astype(str).to_numpy()
        categorical.append("category")
    if spec["category_rule"]:
        if category_evidence is None:
            raise ValueError("Category evidence is required for category-aware variants")
        frame = pd.concat([frame, category_evidence.add_prefix("category_").reset_index(drop=True)], axis=1)
        for column in ("primary_conflict_type", "conflict_signature", "category_primary_conflict", "category_conflict_signature"):
            frame[column] = base[column].astype(str).to_numpy()
            categorical.append(column)
    if spec["regime"]:
        for column in ("matching_regime", "regime_conflict_signature"):
            frame[column] = base[column].astype(str).to_numpy()
            categorical.append(column)
    return frame, categorical
