from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .normalization import normalize_text


# Synthetic anchors use one stable Russian surface key per canonical concept.
# This makes applicability locally verifiable instead of asking the model to
# guess which arbitrary donor field corresponds to the rule.
CONCEPT_ATTRIBUTE_KEYS: dict[str, str] = {
    "compatible_models": "Совместимые модели",
    "main_ingredient": "Основной ингредиент",
    "target_species": "Вид животного",
    "string_gauge_range": "Диапазон толщины струн",
    "model_series": "Серия модели",
    "width_mm": "Ширина",
    "total_volume": "Объем",
    "metal_color": "Цвет металла",
    "model_line": "Линейка модели",
    "gold_color": "Цвет золота",
    "lens_power": "Оптическая сила линз",
    "sphere_power": "Сферическая сила",
    "paper_format": "Формат бумаги",
    "line_diameter": "Диаметр лески",
    "bed_width": "Ширина спального места",
    "instrument_size": "Размер инструмента",
    "breaking_load": "Разрывная нагрузка",
    "tea_type": "Вид чая",
    "curtain_width": "Ширина шторы",
    "page_yield": "Ресурс печати",
    "base_curve": "Радиус кривизны",
    "thickness": "Толщина",
    "curtain_height": "Высота шторы",
    "battery_capacity": "Емкость аккумулятора",
    "wheel_diameter": "Диаметр колес",
    "ssd_capacity": "Объем SSD",
    "string_gauge": "Толщина струн",
    "lens_type": "Тип линз",
    "storage_capacity": "Объем памяти",
    "flavor_profile": "Вкус",
    "case_diameter": "Диаметр корпуса",
    "ram_capacity": "Объем оперативной памяти",
    "vehicle_compatibility": "Совместимость с автомобилем",
    "gemstone_type": "Вид камня",
    "heel_height": "Высота каблука",
    "package_length": "Длина упаковки",
    "color_pattern": "Цветовой рисунок",
    "fabric_type": "Тип ткани",
    "height_cm": "Высота",
    "filler_material": "Материал наполнителя",
    "finish_type": "Финиш",
    "design_theme": "Тема дизайна",
    "orientation": "Ориентация",
    "fragrance_family": "Группа аромата",
    "material_type": "Материал",
    "total_weight": "Общий вес",
    "material_composition": "Состав материала",
    "max_load": "Максимальная нагрузка",
    "product_form": "Форма продукта",
    "diameter": "Диаметр",
    "key_tuning": "Тональность",
    "pasta_shape": "Форма макарон",
    "compatible_phone_model": "Совместимая модель телефона",
    "custom_name": "Персональное имя",
    "engraving_text": "Текст гравировки",
    "lash_thickness": "Толщина ресниц",
    "blade_length": "Длина лезвия",
    "lash_curl": "Изгиб ресниц",
    "lash_length": "Длина ресниц",
    "hook_size": "Размер крючка",
    "needle_diameter": "Диаметр спицы",
    "curl_type": "Тип завитка",
    "sheet_size": "Размер листа",
    "target_surface": "Назначение поверхности",
    "active_ingredient": "Действующее вещество",
    "design_pattern": "Рисунок",
    "frame_size": "Размер рамы",
    "cable_length": "Длина кабеля",
    "load_index": "Индекс нагрузки",
    "target_animal": "Животное",
    "instrument_type": "Тип инструмента",
    "cylinder_power": "Цилиндр",
    "oxygen_permeability": "Кислородопроницаемость",
    "compatible_model": "Совместимая модель",
    "axis": "Ось",
}


# A shared category is not enough for a two-rule anchor.  Every set below is a
# conservative product-level compatibility group: any two concepts in one set
# can naturally be explicit properties of the same concrete product card.
COMPATIBLE_CONCEPT_GROUPS: tuple[frozenset[str], ...] = (
    frozenset(
        {
            "lens_power",
            "sphere_power",
            "base_curve",
            "lens_type",
            "cylinder_power",
            "oxygen_permeability",
            "axis",
        }
    ),
    frozenset({"wheel_diameter", "frame_size"}),
    frozenset({"curtain_width", "curtain_height"}),
    frozenset({"paper_format", "thickness"}),
    frozenset({"lash_thickness", "lash_curl", "lash_length", "curl_type"}),
    frozenset({"instrument_size", "string_gauge", "string_gauge_range", "key_tuning"}),
    frozenset({"battery_capacity", "ssd_capacity", "storage_capacity", "ram_capacity"}),
    frozenset({"tea_type", "flavor_profile"}),
    frozenset({"line_diameter", "breaking_load"}),
    frozenset({"gemstone_type", "metal_color", "gold_color"}),
    frozenset({"total_volume", "target_surface"}),
    frozenset({"flavor_profile", "product_form", "total_weight"}),
    frozenset({"curl", "length", "thickness"}),
    frozenset({"diameter", "length"}),
    frozenset({"color", "size"}),
    frozenset({"gold_color", "insert", "insert_stone", "size", "brand", "length"}),
    frozenset({"diameter", "model"}),
    frozenset({"width", "color"}),
    frozenset({"color", "material"}),
    frozenset({"length", "color"}),
    frozenset({"package_quantity", "variety"}),
)

ALIAS_CONCEPT_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"color", "color_code"}),
    frozenset({"gold_color", "metal_color"}),
    frozenset({"length", "length_mm"}),
    frozenset({"cpu_model", "processor_model"}),
)


ANCHOR_HINTS: dict[str, str] = {
    "oxygen_permeability": "Создай именно контактные линзы, не очки, раствор, контейнер или аксессуар.",
    "base_curve": "Создай именно контактные линзы с явным радиусом кривизны.",
    "axis": "Создай торические контактные линзы с явной осью коррекции.",
    "cylinder_power": "Создай торические контактные линзы с явным цилиндром.",
    "lens_power": "Создай контактные линзы с явной оптической силой.",
    "sphere_power": "Создай контактные линзы или рецептурную оптику с явной сферической силой.",
    "load_index": "Создай автомобильную шину; индекс нагрузки не является массой или размером товара.",
    "case_diameter": "Создай наручные часы; диаметр относится к корпусу часов, не к упаковке.",
    "paper_format": "Создай бумажное изделие с явным стандартным форматом бумаги, не размером упаковки.",
    "storage_capacity": "Создай устройство или накопитель, где объем памяти является собственной характеристикой товара.",
    "page_yield": "Создай картридж или другой печатный расходник с явным ресурсом печати.",
    "vehicle_compatibility": "Создай неуниверсальный автомобильный аксессуар для одной явной модели автомобиля.",
    "compatible_phone_model": "Создай неуниверсальный аксессуар для одной явной модели телефона.",
    "compatible_model": "Создай неуниверсальный аксессуар или расходник для одной явной модели устройства.",
}


@dataclass(frozen=True)
class MutationRule:
    generation_rule_id: str
    source_rule_id: str
    generation_tier: str
    label: int
    concept: str
    relation: str
    semantic_family: str
    attribute_key: str
    anchor_hint: str
    allowed_categories: tuple[str, ...]
    generation_action: str
    required_postcondition: str
    source_path: str
    allowed_product_types: tuple[str, ...] = ()
    allowed_anchor_context_keys: tuple[str, ...] = ()
    required_anchor_context_keys: tuple[str, ...] = ()
    forbidden_anchor_attribute_patterns: tuple[str, ...] = ()
    target_value_pattern: str = ""
    forbidden_target_value_pattern: str = ""
    target_value_domain: tuple[str, ...] = ()
    allowed_value_transitions: tuple[tuple[str, str], ...] = ()
    primary_task_safety_cap: int | None = None
    profile_capacity_policy_version: str = ""
    profile_capacity_policy_sha256: str = ""
    source_examples: tuple[dict[str, Any], ...] = ()

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "generation_rule_id": self.generation_rule_id,
            "source_rule_id": self.source_rule_id,
            "generation_tier": self.generation_tier,
            "label": self.label,
            "concept": self.concept,
            "relation": self.relation,
            "semantic_family": self.semantic_family,
            "required_attribute_key": self.attribute_key,
            "anchor_hint": self.anchor_hint,
            "allowed_product_types": list(self.allowed_product_types),
            "allowed_anchor_context_keys": list(self.allowed_anchor_context_keys),
            "required_anchor_context_keys": list(self.required_anchor_context_keys),
            "forbidden_anchor_attribute_patterns": list(
                self.forbidden_anchor_attribute_patterns
            ),
            "target_value_pattern": self.target_value_pattern,
            "forbidden_target_value_pattern": self.forbidden_target_value_pattern,
            "target_value_domain": list(self.target_value_domain),
            "allowed_value_transitions": [
                list(transition) for transition in self.allowed_value_transitions
            ],
            "primary_task_safety_cap": self.primary_task_safety_cap,
            "profile_capacity_policy_version": self.profile_capacity_policy_version,
            "profile_capacity_policy_sha256": self.profile_capacity_policy_sha256,
            "source_examples": list(self.source_examples),
            "generation_action": self.generation_action,
            "required_postcondition": self.required_postcondition,
        }


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _categories(value: Any) -> tuple[str, ...]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        parsed = json.loads(stripped)
    else:
        parsed = value
    if not isinstance(parsed, (list, tuple)):
        raise ValueError("allowed_categories must be a JSON array or list")
    categories = tuple(sorted({str(item).strip() for item in parsed if str(item).strip()}))
    return categories


def _string_array(value: Any, field: str) -> tuple[str, ...]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ()
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"{field} must be a JSON array or list")
    return tuple(dict.fromkeys(str(item).strip() for item in parsed if str(item).strip()))


def _object_array(value: Any, field: str) -> tuple[dict[str, Any], ...]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ()
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, (list, tuple)) or any(
        not isinstance(item, dict) for item in parsed
    ):
        raise ValueError(f"{field} must be a JSON array of objects")
    return tuple(dict(item) for item in parsed)


def _value_transitions(value: Any) -> tuple[tuple[str, str], ...]:
    """Parse canonical, unordered pairs while retaining display values."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ()
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, (list, tuple)):
        raise ValueError("allowed_value_transitions must be a JSON array or list")
    transitions: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for position, raw_pair in enumerate(parsed):
        if not isinstance(raw_pair, (list, tuple)) or len(raw_pair) != 2:
            raise ValueError(
                "allowed_value_transitions entries must be two-member arrays: "
                f"position={position}"
            )
        members = tuple(str(member).strip() for member in raw_pair)
        normalized = tuple(normalize_text(member) for member in members)
        if not all(normalized):
            raise ValueError(
                "allowed_value_transitions members must be non-empty: "
                f"position={position}"
            )
        if normalized[0] == normalized[1]:
            raise ValueError(
                "allowed_value_transitions members must be distinct: "
                f"position={position}"
            )
        order = sorted(range(2), key=lambda index: (normalized[index], members[index]))
        pair = (members[order[0]], members[order[1]])
        signature = tuple(sorted(normalized))
        if signature in seen:
            raise ValueError(
                "allowed_value_transitions contains a duplicate unordered pair: "
                f"position={position}"
            )
        seen.add(signature)
        transitions.append(pair)
    transitions.sort(key=lambda pair: tuple(normalize_text(member) for member in pair))
    return tuple(transitions)


def _label(row: dict[str, Any], source: Path) -> int:
    for key in ("label", "generated_label", "target"):
        value = row.get(key)
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            result = int(value)
            if result not in {0, 1}:
                raise ValueError(f"{source}: rule label must be 0 or 1")
            return result
    tier = _text(row.get("generation_tier") or row.get("selection_tier"))
    relation = _text(row.get("relation"))
    if relation == "different_value" and (
        "NEGATIVE" in tier or "rare" in source.name.casefold()
    ):
        return 0
    raise ValueError(f"{source}: rule is missing an unambiguous binary label")


def _rule_from_row(row: dict[str, Any], source: Path) -> MutationRule:
    source_rule_id = _text(row.get("source_rule_id") or row.get("rule_id"))
    if not source_rule_id:
        raise ValueError(f"{source}: rule is missing source_rule_id/rule_id")
    generation_rule_id = _text(row.get("generation_rule_id"))
    if not generation_rule_id:
        generation_rule_id = f"gen_rare_neg_{source_rule_id}"
    tier = _text(
        row.get("generation_tier")
        or row.get("selection_tier")
        or row.get("generation_status")
    )
    concept = _text(row.get("concept"))
    relation = _text(row.get("relation"))
    if not tier or not concept or not relation:
        raise ValueError(f"{source}: rule {source_rule_id} has an incomplete definition")
    attribute_key = _text(
        row.get("attribute_key") or row.get("required_attribute_key")
    ) or CONCEPT_ATTRIBUTE_KEYS.get(concept)
    if not attribute_key:
        raise ValueError(
            f"{source}: rule {source_rule_id} has no canonical attribute key for {concept!r}"
        )
    action = _text(row.get("generation_action")) or (
        "Replace exactly one explicit value with a different category-valid value "
        "for the same semantic concept."
    )
    postcondition = _text(
        row.get("required_postcondition") or row.get("generation_guard")
    ) or (
        "Change only this explicit concept, preserve all unrelated facts, and update "
        "every title and attribute mention consistently."
    )
    forbidden_anchor_patterns = _string_array(
        row.get("forbidden_anchor_attribute_patterns"),
        "forbidden_anchor_attribute_patterns",
    )
    target_value_pattern = _text(row.get("target_value_pattern"))
    forbidden_target_value_pattern = _text(
        row.get("forbidden_target_value_pattern")
    )
    allowed_anchor_context_keys = _string_array(
        row.get("allowed_anchor_context_keys"), "allowed_anchor_context_keys"
    )
    required_anchor_context_keys = _string_array(
        row.get("required_anchor_context_keys"), "required_anchor_context_keys"
    )
    normalized_required_context_keys = {
        normalize_text(key) for key in required_anchor_context_keys
    }
    if (
        normalize_text("Тип товара") in normalized_required_context_keys
        or normalize_text(attribute_key) in normalized_required_context_keys
    ):
        raise ValueError(
            f"{source}: rule {source_rule_id} uses a target/type key as required context"
        )
    normalized_allowed_context_keys = {
        normalize_text(key) for key in allowed_anchor_context_keys
    }
    if allowed_anchor_context_keys and not normalized_required_context_keys.issubset(
        normalized_allowed_context_keys
    ):
        raise ValueError(
            f"{source}: rule {source_rule_id} has required context outside "
            "allowed_anchor_context_keys"
        )
    target_value_domain = _string_array(
        row.get("target_value_domain"), "target_value_domain"
    )
    allowed_value_transitions = _value_transitions(
        row.get("allowed_value_transitions")
    )
    if target_value_domain and allowed_value_transitions:
        normalized_domain = {normalize_text(value) for value in target_value_domain}
        outside_domain = sorted(
            {
                member
                for pair in allowed_value_transitions
                for member in pair
                if normalize_text(member) not in normalized_domain
            },
            key=normalize_text,
        )
        if outside_domain:
            raise ValueError(
                f"{source}: rule {source_rule_id} has transition members outside "
                f"target_value_domain: {outside_domain}"
            )
    raw_primary_cap = row.get("primary_task_safety_cap")
    primary_task_safety_cap = (
        None
        if raw_primary_cap is None
        or (isinstance(raw_primary_cap, float) and math.isnan(raw_primary_cap))
        or str(raw_primary_cap).strip() == ""
        else int(raw_primary_cap)
    )
    if primary_task_safety_cap is not None and primary_task_safety_cap < 1:
        raise ValueError(
            f"{source}: rule {source_rule_id} has non-positive primary_task_safety_cap"
        )
    for field, patterns in {
        "forbidden_anchor_attribute_patterns": forbidden_anchor_patterns,
        "target_value_pattern": (target_value_pattern,) if target_value_pattern else (),
        "forbidden_target_value_pattern": (
            (forbidden_target_value_pattern,)
            if forbidden_target_value_pattern
            else ()
        ),
    }.items():
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(
                    f"{source}: rule {source_rule_id} has invalid {field}: {pattern!r}"
                ) from error
    return MutationRule(
        generation_rule_id=generation_rule_id,
        source_rule_id=source_rule_id,
        generation_tier=tier,
        label=_label(row, source),
        concept=concept,
        relation=relation,
        semantic_family=_text(row.get("semantic_family")) or "unspecified",
        attribute_key=attribute_key,
        anchor_hint=(
            _text(row.get("anchor_hint") or row.get("applicability_instruction"))
            or ANCHOR_HINTS.get(concept, "")
        ),
        allowed_categories=_categories(row.get("allowed_categories")),
        generation_action=action,
        required_postcondition=postcondition,
        source_path=str(source.resolve()),
        allowed_product_types=_string_array(
            row.get("allowed_product_types"), "allowed_product_types"
        ),
        allowed_anchor_context_keys=allowed_anchor_context_keys,
        required_anchor_context_keys=required_anchor_context_keys,
        forbidden_anchor_attribute_patterns=forbidden_anchor_patterns,
        target_value_pattern=target_value_pattern,
        forbidden_target_value_pattern=forbidden_target_value_pattern,
        target_value_domain=target_value_domain,
        allowed_value_transitions=allowed_value_transitions,
        primary_task_safety_cap=primary_task_safety_cap,
        profile_capacity_policy_version=_text(
            row.get("profile_capacity_policy_version")
        ),
        profile_capacity_policy_sha256=_text(
            row.get("profile_capacity_policy_sha256")
        ),
        source_examples=_object_array(row.get("source_examples"), "source_examples"),
    )


def rules_are_product_compatible(left: MutationRule, right: MutationRule) -> bool:
    """Return whether two rules can safely target one concrete product card."""
    if left.label != right.label or left.concept == right.concept:
        return False
    if not set(left.allowed_categories) & set(right.allowed_categories):
        return False
    if any({left.concept, right.concept}.issubset(group) for group in ALIAS_CONCEPT_GROUPS):
        return False
    required_context_keys = {
        normalize_text(key)
        for rule in (left, right)
        for key in rule.required_anchor_context_keys
    }
    if required_context_keys & {
        normalize_text(left.attribute_key),
        normalize_text(right.attribute_key),
        normalize_text("Тип товара"),
    }:
        return False
    for rule in (left, right):
        allowed_context_keys = {
            normalize_text(key) for key in rule.allowed_anchor_context_keys
        }
        if allowed_context_keys and not required_context_keys.issubset(
            allowed_context_keys
        ):
            return False
    left_types = {normalize_text(value) for value in left.allowed_product_types}
    right_types = {normalize_text(value) for value in right.allowed_product_types}
    if left_types and right_types:
        if not left_types & right_types or left.attribute_key == right.attribute_key:
            return False
        return any(
            {left.concept, right.concept}.issubset(group)
            for group in COMPATIBLE_CONCEPT_GROUPS
        )
    return any(
        {left.concept, right.concept}.issubset(group)
        for group in COMPATIBLE_CONCEPT_GROUPS
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        return pd.read_csv(path).to_dict("records")
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("rules")
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected a JSON array or an object with rules")
        return value
    raise ValueError(f"Unsupported rule catalog format: {path}")


def load_mutation_rules(
    paths: Iterable[Path],
    *,
    tiers: set[str] | None = None,
    labels: set[int] | None = None,
) -> list[MutationRule]:
    """Load and deduplicate machine rules from JSON, JSONL or CSV catalogs.

    Later files replace duplicate definitions.  This lets a rich executable JSON
    overlay a broader candidate CSV while retaining one stable rule identifier.
    Rules without an allowed category remain visible to callers, but cannot be
    selected for generation.
    """

    by_id: dict[str, MutationRule] = {}
    for raw_path in paths:
        path = raw_path.resolve()
        for row in _read_rows(path):
            rule = _rule_from_row(row, path)
            previous = by_id.get(rule.generation_rule_id)
            if previous is not None and previous.source_rule_id != rule.source_rule_id:
                raise ValueError(
                    f"Conflicting source IDs for {rule.generation_rule_id}: "
                    f"{previous.source_rule_id} != {rule.source_rule_id}"
                )
            by_id[rule.generation_rule_id] = rule
    rules = sorted(by_id.values(), key=lambda rule: rule.generation_rule_id)
    if tiers is not None:
        rules = [rule for rule in rules if rule.generation_tier in tiers]
    if labels is not None:
        rules = [rule for rule in rules if rule.label in labels]
    if not rules:
        raise ValueError("No mutation rules remain after loading and filtering")
    return rules


def rules_by_category(rules: Iterable[MutationRule]) -> dict[str, list[MutationRule]]:
    result: dict[str, list[MutationRule]] = {}
    for rule in rules:
        for category in rule.allowed_categories:
            result.setdefault(category, []).append(rule)
    for category in result:
        result[category].sort(key=lambda rule: rule.generation_rule_id)
    return dict(sorted(result.items()))


def catalog_summary(rules: Iterable[MutationRule]) -> dict[str, Any]:
    values = list(rules)
    selectable = [rule for rule in values if rule.allowed_categories]
    tier_counts: dict[str, int] = {}
    for rule in values:
        tier_counts[rule.generation_tier] = tier_counts.get(rule.generation_tier, 0) + 1
    policy_versions = sorted(
        {rule.profile_capacity_policy_version for rule in values if rule.profile_capacity_policy_version}
    )
    policy_hashes = sorted(
        {rule.profile_capacity_policy_sha256 for rule in values if rule.profile_capacity_policy_sha256}
    )
    return {
        "loaded_rules": len(values),
        "selectable_rules": len(selectable),
        "source_scoped_rules": sum(bool(rule.allowed_product_types) for rule in values),
        "product_type_profiles": sum(len(rule.allowed_product_types) for rule in values),
        "rules_without_allowed_categories": len(values) - len(selectable),
        "labels": sorted({rule.label for rule in values}),
        "categories": sorted({category for rule in selectable for category in rule.allowed_categories}),
        "tier_counts": dict(sorted(tier_counts.items())),
        "finite_domain_rules": sum(bool(rule.target_value_domain) for rule in values),
        "transition_constrained_rules": sum(
            bool(rule.allowed_value_transitions) for rule in values
        ),
        "allowed_value_transition_pairs": sum(
            len(rule.allowed_value_transitions) for rule in values
        ),
        "required_anchor_context_rules": sum(
            bool(rule.required_anchor_context_keys) for rule in values
        ),
        "capacity_limited_rules": sum(
            rule.primary_task_safety_cap is not None for rule in values
        ),
        "profile_capacity_policy_versions": policy_versions,
        "profile_capacity_policy_sha256s": policy_hashes,
    }
