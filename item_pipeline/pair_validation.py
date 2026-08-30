from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from .normalization import canonical_json_dumps, normalize_text, parse_attributes
from .pair_rules import MutationRule
from .rule_values import canonical_target_value
from .validation import clean_generated_item


FORBIDDEN_ANCHOR_IDENTIFIER_KEY_RE = re.compile(
    r"(?<!\w)(?:sku|артикул(?:а|у|ом|е|ы|ов)?)(?!\w)", re.IGNORECASE
)
FORBIDDEN_ANCHOR_IDENTIFIER_TITLE_RE = re.compile(
    r"(?<!\w)(?:sku|артикул(?:а|у|ом|е|ы|ов)?)\s*[:#№-]?\s*[a-zа-яё0-9]",
    re.IGNORECASE,
)

NUMERIC_SEMANTIC_CONCEPTS = {
    "axis",
    "case_diameter",
    "diameter",
    "length",
    "length_mm",
    "optical_power",
    "package_quantity",
    "size",
    "storage_capacity",
    "wheel_diameter",
    "width",
}
CANONICAL_UNIT_CONCEPTS = {
    "case_diameter",
    "diameter",
    "length",
    "length_mm",
    "package_quantity",
    "wheel_diameter",
    "width",
}
SEMANTIC_EQUIVALENT_VALUE_GROUPS: dict[str, tuple[frozenset[str], ...]] = {
    "material": (
        frozenset(
            {
                "искусственная кожа",
                "экокожа",
                "эко кожа",
                "кожзам",
                "кожзаменитель",
                "pu кожа",
                "полиуретановая кожа",
                "винилискожа",
            }
        ),
    ),
    "insert": (
        frozenset({"фианит", "фианиты", "кубический цирконий"}),
    ),
    "insert_stone": (
        frozenset({"фианит", "фианиты", "кубический цирконий"}),
    ),
    "insert_type": (
        frozenset({"фианит", "фианиты", "кубический цирконий"}),
    ),
}


@dataclass(frozen=True)
class MutationValidation:
    item: dict[str, Any]
    valid: bool
    reasons: list[str]
    metrics: dict[str, float]


@dataclass(frozen=True)
class RuleAnchorValidation:
    item: dict[str, Any]
    evidence: list[dict[str, str]]
    valid: bool
    reasons: list[str]
    metrics: dict[str, float]


def _metadata_mutation_rule(raw: dict[str, Any]) -> MutationRule:
    return MutationRule(
        generation_rule_id=str(raw["generation_rule_id"]),
        source_rule_id=str(raw["source_rule_id"]),
        generation_tier=str(raw["generation_tier"]),
        label=int(raw["label"]),
        concept=str(raw["concept"]),
        relation=str(raw["relation"]),
        semantic_family=str(raw.get("semantic_family") or "unspecified"),
        attribute_key=str(
            raw.get("required_attribute_key") or raw.get("attribute_key") or ""
        ),
        anchor_hint=str(raw.get("anchor_hint") or ""),
        allowed_categories=tuple(str(value) for value in raw["allowed_categories"]),
        generation_action=str(raw.get("generation_action") or ""),
        required_postcondition=str(raw.get("required_postcondition") or ""),
        source_path=str(raw.get("source_path") or "metadata"),
        allowed_product_types=tuple(
            str(value) for value in raw.get("allowed_product_types") or []
        ),
        allowed_anchor_context_keys=tuple(
            str(value) for value in raw.get("allowed_anchor_context_keys") or []
        ),
        required_anchor_context_keys=tuple(
            str(value) for value in raw.get("required_anchor_context_keys") or []
        ),
        forbidden_anchor_attribute_patterns=tuple(
            str(value)
            for value in raw.get("forbidden_anchor_attribute_patterns") or []
        ),
        target_value_pattern=str(raw.get("target_value_pattern") or ""),
        forbidden_target_value_pattern=str(
            raw.get("forbidden_target_value_pattern") or ""
        ),
        target_value_domain=tuple(
            str(value) for value in raw.get("target_value_domain") or []
        ),
        allowed_value_transitions=tuple(
            (str(value[0]), str(value[1]))
            for value in raw.get("allowed_value_transitions") or []
        ),
        primary_task_safety_cap=(
            int(raw["primary_task_safety_cap"])
            if raw.get("primary_task_safety_cap") is not None
            else None
        ),
        profile_capacity_policy_version=str(
            raw.get("profile_capacity_policy_version") or ""
        ),
        profile_capacity_policy_sha256=str(
            raw.get("profile_capacity_policy_sha256") or ""
        ),
        source_examples=tuple(
            dict(value) for value in raw.get("source_examples") or []
        ),
    )


def _contains_value(text: Any, value: Any) -> bool:
    haystack = normalize_text(text)
    needle = normalize_text(value)
    if not haystack or not needle:
        return False
    if haystack == needle:
        return True
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None


def _starts_with_value(text: Any, value: Any) -> bool:
    haystack = normalize_text(text)
    needle = normalize_text(value)
    if not haystack or not needle:
        return False
    return re.match(rf"^{re.escape(needle)}(?!\w)", haystack) is not None


def _value_occurrences(text: Any, value: Any) -> int:
    haystack = normalize_text(text)
    needle = normalize_text(value)
    if not haystack or not needle:
        return 0
    return len(re.findall(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack))


def _target_value_reasons(
    rule: MutationRule,
    value: Any,
    *,
    prefix: str,
    product_type: Any = "",
) -> list[str]:
    normalized = normalize_text(value)
    reasons: list[str] = []
    if rule.target_value_pattern and re.fullmatch(
        rule.target_value_pattern, normalized, flags=re.IGNORECASE
    ) is None:
        reasons.append(f"{prefix}:target_value_pattern_mismatch")
    if rule.forbidden_target_value_pattern and re.search(
        rule.forbidden_target_value_pattern, normalized, flags=re.IGNORECASE
    ):
        reasons.append(f"{prefix}:forbidden_target_value_pattern")
    if (rule.target_value_domain or rule.concept in CANONICAL_UNIT_CONCEPTS) and canonical_target_value(
        rule.concept,
        product_type or (rule.allowed_product_types[0] if rule.allowed_product_types else ""),
        value,
        rule.target_value_domain,
    ) is None:
        reasons.append(
            f"{prefix}:outside_target_value_domain"
            if rule.target_value_domain
            else f"{prefix}:missing_or_invalid_canonical_unit"
        )
    return reasons


def _numeric_components(value: Any) -> tuple[str, ...]:
    normalized = normalize_text(value).replace(",", ".")
    return tuple(
        str(float(component))
        for component in re.findall(r"[+-]?\d+(?:\.\d+)?", normalized)
    )


def _target_values_semantically_equivalent(
    rule: MutationRule, original: Any, new: Any, *, product_type: Any = ""
) -> bool:
    left, right = normalize_text(original), normalize_text(new)
    if not left or not right:
        return False
    if rule.target_value_domain or rule.concept in CANONICAL_UNIT_CONCEPTS:
        scoped_type = product_type or (
            rule.allowed_product_types[0] if rule.allowed_product_types else ""
        )
        left_canonical = canonical_target_value(
            rule.concept, scoped_type, original, rule.target_value_domain
        )
        right_canonical = canonical_target_value(
            rule.concept, scoped_type, new, rule.target_value_domain
        )
        if left_canonical is not None and left_canonical == right_canonical:
            return True
    if fuzz.ratio(left, right) >= 94:
        return True
    if rule.concept in NUMERIC_SEMANTIC_CONCEPTS:
        left_numbers, right_numbers = _numeric_components(left), _numeric_components(right)
        if left_numbers and left_numbers == right_numbers:
            return True
    return any(
        left in group and right in group
        for group in SEMANTIC_EQUIVALENT_VALUE_GROUPS.get(rule.concept, ())
    )


def _transition_signature(left: Any, right: Any) -> tuple[str, str]:
    return tuple(sorted((normalize_text(left), normalize_text(right))))


def _allowed_transition_signatures(
    rule: MutationRule,
) -> frozenset[tuple[str, str]]:
    return frozenset(
        _transition_signature(left, right)
        for left, right in rule.allowed_value_transitions
    )


def _is_allowed_transition_endpoint(rule: MutationRule, value: Any) -> bool:
    normalized = normalize_text(value)
    return any(
        normalized in {normalize_text(left), normalize_text(right)}
        for left, right in rule.allowed_value_transitions
    )


def _exact_substitution_title(
    anchor_name: Any,
    replacements: list[tuple[str, str]],
) -> tuple[str, bool]:
    expected = normalize_text(anchor_name)
    by_old: dict[str, str] = {}
    ambiguous = False
    for raw_old, raw_new in replacements:
        old, new = normalize_text(raw_old), normalize_text(raw_new)
        previous = by_old.get(old)
        if previous is not None and previous != new:
            ambiguous = True
        by_old[old] = new
    for old, new in sorted(by_old.items(), key=lambda item: len(item[0]), reverse=True):
        expected = re.sub(rf"(?<!\w){re.escape(old)}(?!\w)", new, expected)
    return expected, ambiguous


def validate_rule_anchor(
    item: dict[str, Any],
    product_type: str,
    evidence: list[dict[str, Any]],
    *,
    category: str,
    rules: list[MutationRule],
) -> RuleAnchorValidation:
    raw_attributes = item.get("attributes")
    expected_keys = list(raw_attributes) if isinstance(raw_attributes, dict) else []
    cleaned = clean_generated_item(item, expected_keys)
    attributes = cleaned["attributes"]
    reasons: list[str] = []

    if cleaned["category"] != category:
        reasons.append("anchor_category_mismatch")
    if not isinstance(raw_attributes, dict) or not 3 <= len(raw_attributes) <= 12:
        reasons.append("anchor_attribute_count_out_of_bounds")
    if any(not key.strip() or not value for key, value in attributes.items()):
        reasons.append("anchor_empty_attribute")
    if any(
        FORBIDDEN_ANCHOR_IDENTIFIER_KEY_RE.search(normalize_text(key))
        for key in attributes
    ):
        reasons.append("anchor_forbidden_sku_or_article_attribute")
    if FORBIDDEN_ANCHOR_IDENTIFIER_TITLE_RE.search(cleaned["name"]):
        reasons.append("anchor_forbidden_sku_or_article_in_name")
    if "Тип товара" not in attributes:
        reasons.append("anchor_missing_product_type_attribute")
    elif normalize_text(attributes["Тип товара"]) != normalize_text(product_type):
        reasons.append("anchor_product_type_mismatch")
    elif not _starts_with_value(cleaned["name"], product_type):
        reasons.append("anchor_name_does_not_start_with_product_type")

    scoped_type_sets = [
        {normalize_text(value) for value in rule.allowed_product_types}
        for rule in rules
        if rule.allowed_product_types
    ]
    if scoped_type_sets:
        allowed_types = set.intersection(*scoped_type_sets)
        if not allowed_types:
            reasons.append("anchor_rules_have_no_shared_product_type")
        elif normalize_text(product_type) not in allowed_types:
            reasons.append("anchor_product_type_outside_rule_scope")
    target_attribute_keys = {rule.attribute_key for rule in rules}
    for rule in rules:
        for required_key in rule.required_anchor_context_keys:
            if required_key not in attributes:
                reasons.append(
                    "anchor_missing_required_context_key:"
                    f"{rule.generation_rule_id}:{required_key}"
                )
    context_key_sets = [
        set(rule.allowed_anchor_context_keys)
        for rule in rules
        if rule.allowed_anchor_context_keys
    ]
    if context_key_sets:
        allowed_context_keys = set.intersection(*context_key_sets)
        unexpected_context_keys = (
            set(attributes) - {"Тип товара"} - target_attribute_keys - allowed_context_keys
        )
        if unexpected_context_keys:
            reasons.append("anchor_context_key_outside_rule_allowlist")
        actual_context_keys = sorted(
            set(attributes) - {"Тип товара"} - target_attribute_keys
        )
        title_value_order = [
            "Тип товара",
            *actual_context_keys,
            *[rule.attribute_key for rule in rules],
        ]
        expected_name = normalize_text(
            " ".join(attributes[key] for key in title_value_order if key in attributes)
        )
        if normalize_text(cleaned["name"]) != expected_name:
            reasons.append("anchor_name_not_exact_attribute_value_concatenation")
    for rule in rules:
        for pattern in rule.forbidden_anchor_attribute_patterns:
            if any(
                re.search(pattern, normalize_text(key), flags=re.IGNORECASE)
                for key in attributes
                if key not in target_attribute_keys
            ):
                reasons.append(
                    f"anchor_forbidden_dependent_attribute:{rule.generation_rule_id}"
                )
                break

    expected_by_id = {rule.generation_rule_id: rule for rule in rules}
    if len(evidence) != len(rules):
        reasons.append("anchor_evidence_count_mismatch")
    seen_rule_ids: set[str] = set()
    seen_attribute_keys: set[str] = set()
    seen_attribute_values: set[str] = set()
    cleaned_evidence: list[dict[str, str]] = []
    for position, raw in enumerate(evidence):
        if not isinstance(raw, dict):
            reasons.append(f"anchor_evidence_{position}:not_an_object")
            continue
        rule_id = str(raw.get("generation_rule_id") or "")
        concept = str(raw.get("concept") or "")
        attribute_key = str(raw.get("attribute_key") or "")
        attribute_value = str(raw.get("attribute_value") or "").strip()
        rule = expected_by_id.get(rule_id)
        if rule is None:
            reasons.append(f"anchor_evidence_{position}:unexpected_rule")
            continue
        if rule_id in seen_rule_ids:
            reasons.append(f"anchor_evidence_{position}:duplicate_rule")
        seen_rule_ids.add(rule_id)
        if concept != rule.concept:
            reasons.append(f"anchor_evidence_{position}:concept_mismatch")
        if attribute_key != rule.attribute_key:
            reasons.append(f"anchor_evidence_{position}:wrong_attribute_key")
        if attribute_key in seen_attribute_keys:
            reasons.append(f"anchor_evidence_{position}:shared_attribute_key")
        seen_attribute_keys.add(attribute_key)
        if attribute_key not in attributes:
            reasons.append(f"anchor_evidence_{position}:attribute_missing")
        elif normalize_text(attributes[attribute_key]) != normalize_text(attribute_value):
            reasons.append(f"anchor_evidence_{position}:value_mismatch")
        if not attribute_value:
            reasons.append(f"anchor_evidence_{position}:empty_value")
        elif not _contains_value(cleaned["name"], attribute_value):
            reasons.append(f"anchor_evidence_{position}:value_missing_from_name")
        elif _value_occurrences(cleaned["name"], attribute_value) != 1:
            reasons.append(f"anchor_evidence_{position}:value_not_unique_in_name")
        normalized_attribute_value = normalize_text(attribute_value)
        if normalized_attribute_value in seen_attribute_values:
            reasons.append(f"anchor_evidence_{position}:shared_target_value")
        seen_attribute_values.add(normalized_attribute_value)
        reasons.extend(
            _target_value_reasons(
                rule,
                attribute_value,
                prefix=f"anchor_evidence_{position}",
                product_type=product_type,
            )
        )
        if rule.allowed_value_transitions and not _is_allowed_transition_endpoint(
            rule, attribute_value
        ):
            reasons.append(
                f"anchor_evidence_{position}:outside_allowed_value_transitions"
            )
        cleaned_evidence.append(
            {
                "generation_rule_id": rule_id,
                "concept": concept,
                "attribute_key": attribute_key,
                "attribute_value": attribute_value,
            }
        )
    if seen_rule_ids != set(expected_by_id):
        reasons.append("anchor_not_all_rules_covered")
    return RuleAnchorValidation(
        item=cleaned,
        evidence=cleaned_evidence,
        valid=not reasons,
        reasons=reasons,
        metrics={
            "anchor_attribute_count": float(len(attributes)),
            "anchor_target_count": float(len(cleaned_evidence)),
        },
    )


def validate_mutation(
    item: dict[str, Any],
    applications: list[dict[str, Any]],
    *,
    anchor: dict[str, Any],
    rules: list[MutationRule],
    evidence: list[dict[str, str]] | None = None,
    minimum_name_similarity: float = 0.35,
) -> MutationValidation:
    anchor_attributes = parse_attributes(anchor["attributes"])
    product_type = anchor_attributes.get("Тип товара", "")
    expected_keys = list(anchor_attributes)
    cleaned = clean_generated_item(item, expected_keys)
    mutated_attributes = cleaned["attributes"]
    reasons: list[str] = []

    if cleaned["category"] != str(anchor["category"]):
        reasons.append("category_mismatch")
    raw_attributes = item.get("attributes")
    if not isinstance(raw_attributes, dict) or set(raw_attributes) != set(expected_keys):
        reasons.append("attribute_schema_mismatch")
    if any(not value for value in mutated_attributes.values()):
        reasons.append("empty_attribute_value")

    actual_changed_keys = {
        key
        for key in expected_keys
        if normalize_text(anchor_attributes[key])
        != normalize_text(mutated_attributes[key])
    }
    expected_by_id = {rule.generation_rule_id: rule for rule in rules}
    evidence_by_id = {
        str(entry.get("generation_rule_id") or ""): entry
        for entry in (evidence or [])
    }
    if evidence is not None and set(evidence_by_id) != set(expected_by_id):
        reasons.append("verified_evidence_rule_mismatch")
    if len(expected_by_id) != len(rules):
        reasons.append("duplicate_requested_rule")
    if len({rule.concept for rule in rules}) != len(rules):
        reasons.append("duplicate_requested_concept")
    if len({rule.label for rule in rules}) != 1:
        reasons.append("mixed_rule_labels")
    if any(str(anchor["category"]) not in rule.allowed_categories for rule in rules):
        reasons.append("rule_not_allowed_for_category")
    if len(applications) != len(rules):
        reasons.append("application_count_mismatch")

    seen_rule_ids: set[str] = set()
    seen_changed_keys: set[str] = set()
    declared_changed_keys: set[str] = set()
    title_replacements: list[tuple[str, str]] = []
    for position, application in enumerate(applications):
        if not isinstance(application, dict):
            reasons.append(f"application_{position}:not_an_object")
            continue
        rule_id = str(application.get("generation_rule_id") or "")
        rule = expected_by_id.get(rule_id)
        if rule is None:
            reasons.append(f"application_{position}:unexpected_rule")
            continue
        if rule_id in seen_rule_ids:
            reasons.append(f"application_{position}:duplicate_rule")
        seen_rule_ids.add(rule_id)
        if str(application.get("concept") or "") != rule.concept:
            reasons.append(f"application_{position}:concept_mismatch")

        attribute_key = str(application.get("attribute_key") or "")
        changed_keys = [attribute_key] if attribute_key else []
        if attribute_key not in expected_keys:
            reasons.append(f"application_{position}:unknown_changed_key")
        if attribute_key in seen_changed_keys:
            reasons.append(f"application_{position}:shared_changed_key")
        seen_changed_keys.add(attribute_key)
        declared_changed_keys.add(attribute_key)
        if attribute_key != rule.attribute_key:
            reasons.append(f"application_{position}:wrong_rule_attribute_key")
        verified = evidence_by_id.get(rule_id)
        if verified is not None:
            if attribute_key != verified.get("attribute_key"):
                reasons.append(f"application_{position}:key_differs_from_anchor_evidence")
            if normalize_text(application.get("original_value")) != normalize_text(
                verified.get("attribute_value")
            ):
                reasons.append(f"application_{position}:original_differs_from_anchor_evidence")

        original = str(application.get("original_value") or "").strip()
        new = str(application.get("new_value") or "").strip()
        if not original or not new:
            reasons.append(f"application_{position}:empty_value")
            continue
        if normalize_text(original) == normalize_text(new):
            reasons.append(f"application_{position}:value_not_changed")
        elif rule.label == 0 and _target_values_semantically_equivalent(
            rule, original, new, product_type=product_type
        ):
            reasons.append(
                f"application_{position}:semantically_equivalent_target_values"
            )
        if (
            rule.allowed_value_transitions
            and _transition_signature(original, new)
            not in _allowed_transition_signatures(rule)
        ):
            reasons.append(f"application_{position}:disallowed_value_transition")
        reasons.extend(
            _target_value_reasons(
                rule,
                original,
                prefix=f"application_{position}:old",
                product_type=product_type,
            )
        )
        reasons.extend(
            _target_value_reasons(
                rule,
                new,
                prefix=f"application_{position}:new",
                product_type=product_type,
            )
        )
        title_replacements.append((original, new))
        usable_keys = [key for key in changed_keys if key in anchor_attributes]
        if not any(_contains_value(anchor_attributes[key], original) for key in usable_keys):
            reasons.append(f"application_{position}:original_not_in_anchor_attributes")
        if not any(_contains_value(mutated_attributes[key], new) for key in usable_keys):
            reasons.append(f"application_{position}:new_not_in_mutated_attributes")
        if any(_contains_value(mutated_attributes[key], original) for key in usable_keys):
            reasons.append(f"application_{position}:old_value_remains_in_changed_keys")

        if not _contains_value(anchor["name"], original):
            reasons.append(f"application_{position}:original_missing_from_anchor_name")
        if _contains_value(cleaned["name"], original):
            reasons.append(f"application_{position}:old_value_remains_in_name")
        if not _contains_value(cleaned["name"], new):
            reasons.append(f"application_{position}:new_value_missing_from_name")

    if seen_rule_ids != set(expected_by_id):
        reasons.append("not_all_rules_applied")
    if actual_changed_keys != declared_changed_keys:
        reasons.append("declared_changed_keys_do_not_match_actual_changes")
    if not actual_changed_keys:
        reasons.append("no_attribute_value_changed")
    if all(
        normalize_text(anchor_attributes[key]) == normalize_text(mutated_attributes[key])
        for key in expected_keys
    ) and normalize_text(anchor["name"]) == normalize_text(cleaned["name"]):
        reasons.append("card_unchanged")

    expected_title, ambiguous_title_replacement = _exact_substitution_title(
        anchor["name"], title_replacements
    )
    if ambiguous_title_replacement:
        reasons.append("ambiguous_target_value_replacement")
    elif normalize_text(cleaned["name"]) != expected_title:
        reasons.append("title_not_exact_target_substitution")

    for key in set(expected_keys) - actual_changed_keys:
        stable_value = anchor_attributes[key]
        if _contains_value(anchor["name"], stable_value) and not _contains_value(
            cleaned["name"], stable_value
        ):
            reasons.append(f"unrelated_title_value_missing:{key}")

    name_similarity = fuzz.token_set_ratio(str(anchor["name"]), cleaned["name"]) / 100.0
    if name_similarity < minimum_name_similarity:
        reasons.append("name_changed_too_much")
    changed_fraction = len(actual_changed_keys) / max(1, len(expected_keys))
    return MutationValidation(
        item=cleaned,
        valid=not reasons,
        reasons=reasons,
        metrics={
            "rule_count": float(len(rules)),
            "changed_attribute_count": float(len(actual_changed_keys)),
            "changed_attribute_fraction": float(changed_fraction),
            "name_similarity": float(name_similarity),
        },
    )


def validate_pair_dataset(
    items_path: Path,
    pairs_path: Path,
    *,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    items = pd.read_parquet(items_path)
    pairs = pd.read_parquet(pairs_path)
    item_columns = ["id", "name", "attributes", "category"]
    pair_columns = ["id1", "id2", "target"]
    missing_item_columns = set(item_columns) - set(items.columns)
    missing_pair_columns = set(pair_columns) - set(pairs.columns)
    if missing_item_columns or missing_pair_columns:
        raise ValueError(
            f"Missing columns: items={sorted(missing_item_columns)}, "
            f"pairs={sorted(missing_pair_columns)}"
        )

    invalid_items = 0
    canonical_cards: list[str] = []
    for row in items[item_columns].itertuples(index=False):
        try:
            attributes = parse_attributes(row.attributes)
        except Exception:
            attributes = {}
        if not str(row.name).strip() or not str(row.category).strip() or not attributes:
            invalid_items += 1
        canonical_cards.append(
            canonical_json_dumps(
                {
                    "name": normalize_text(row.name),
                    "attributes": attributes,
                    "category": str(row.category),
                }
            )
        )

    item_ids = set(items["id"])
    missing_ids = int(
        (~pairs["id1"].isin(item_ids)).sum() + (~pairs["id2"].isin(item_ids)).sum()
    )
    category_by_id = items.drop_duplicates("id", keep=False).set_index("id")["category"]
    left_category = pairs["id1"].map(category_by_id)
    right_category = pairs["id2"].map(category_by_id)
    cross_category = int(left_category.ne(right_category).sum())
    self_pairs = int(pairs["id1"].eq(pairs["id2"]).sum())
    invalid_targets = int((~pairs["target"].isin([0, 1])).sum())
    unordered = pd.DataFrame(
        {
            "low": pairs[["id1", "id2"]].min(axis=1),
            "high": pairs[["id1", "id2"]].max(axis=1),
        }
    )
    duplicate_pairs = int(unordered.duplicated(keep=False).sum())

    metadata_errors = 0
    semantic_metadata_errors = 0
    metadata_rows: int | None = None
    if metadata_path is not None and metadata_path.exists():
        metadata = pd.read_parquet(metadata_path)
        metadata_rows = len(metadata)
        pair_keys = {
            (int(id1), int(id2)) for id1, id2 in zip(pairs["id1"], pairs["id2"])
        }
        metadata_keys = {
            (int(id1), int(id2))
            for id1, id2 in zip(metadata["id1"], metadata["id2"])
        }
        metadata_errors += len(pair_keys ^ metadata_keys)
        metadata_errors += int(metadata[["id1", "id2"]].duplicated(keep=False).sum())
        pair_contract = {
            (int(row.id1), int(row.id2)): {
                "target": int(row.target),
                "left_category": left_category.iloc[position],
                "right_category": right_category.iloc[position],
            }
            for position, row in enumerate(pairs.itertuples(index=False))
        }
        item_by_id = items.drop_duplicates("id", keep=False).set_index("id")
        for row in metadata.itertuples(index=False):
            row_semantic_error = False
            pair_key = (int(row.id1), int(row.id2))
            contract = pair_contract.get(pair_key)
            if contract is None:
                metadata_errors += 1
            elif (
                int(row.target) != contract["target"]
                or str(row.category) != str(contract["left_category"])
                or str(row.category) != str(contract["right_category"])
            ):
                metadata_errors += 1
            try:
                rules = json.loads(row.rules_json)
                evidence = json.loads(row.anchor_evidence_json)
                applications = json.loads(row.applications_json)
            except Exception:
                metadata_errors += 1
                continue
            if not rules:
                metadata_errors += 1
                continue
            category = str(row.category)
            labels = {int(rule["label"]) for rule in rules}
            if labels != {int(row.target)}:
                metadata_errors += 1
            if any(category not in rule["allowed_categories"] for rule in rules):
                metadata_errors += 1
            if row.id1 not in item_by_id.index or row.id2 not in item_by_id.index:
                continue
            left = item_by_id.loc[row.id1]
            right = item_by_id.loc[row.id2]
            try:
                left_attributes = parse_attributes(left.attributes)
                right_attributes = parse_attributes(right.attributes)
                metadata_rules = [_metadata_mutation_rule(rule) for rule in rules]
                rule_by_id = {
                    str(rule["generation_rule_id"]): rule for rule in rules
                }
                evidence_by_id = {
                    str(entry["generation_rule_id"]): entry for entry in evidence
                }
                application_by_id = {
                    str(entry["generation_rule_id"]): entry for entry in applications
                }
            except Exception:
                semantic_metadata_errors += 1
                continue
            product_type = str(
                getattr(row, "product_type", "")
                or left_attributes.get("Тип товара")
                or ""
            )
            anchor_check = validate_rule_anchor(
                {
                    "name": left["name"],
                    "attributes": left_attributes,
                    "category": left["category"],
                },
                product_type,
                evidence,
                category=category,
                rules=metadata_rules,
            )
            mutation_check = validate_mutation(
                {
                    "name": right["name"],
                    "attributes": right_attributes,
                    "category": right["category"],
                },
                applications,
                anchor={
                    "id": int(row.id1),
                    "name": left["name"],
                    "attributes": left_attributes,
                    "category": left["category"],
                },
                rules=metadata_rules,
                evidence=anchor_check.evidence,
            )
            if not anchor_check.valid or not mutation_check.valid:
                row_semantic_error = True
            if not (
                set(rule_by_id) == set(evidence_by_id) == set(application_by_id)
                and len(rule_by_id) == len(rules)
            ):
                row_semantic_error = True
            target_keys: set[str] = set()
            for rule_id, rule in rule_by_id.items():
                evidence_entry = evidence_by_id.get(rule_id, {})
                application = application_by_id.get(rule_id, {})
                required_key = str(rule.get("required_attribute_key") or "")
                key = str(application.get("attribute_key") or "")
                old = str(application.get("original_value") or "")
                new = str(application.get("new_value") or "")
                target_keys.add(key)
                if (
                    key != required_key
                    or evidence_entry.get("attribute_key") != key
                    or normalize_text(evidence_entry.get("attribute_value"))
                    != normalize_text(old)
                    or key not in left_attributes
                    or key not in right_attributes
                    or normalize_text(left_attributes.get(key)) != normalize_text(old)
                    or normalize_text(right_attributes.get(key)) != normalize_text(new)
                    or normalize_text(old) == normalize_text(new)
                    or not _contains_value(left["name"], old)
                    or _contains_value(right["name"], old)
                    or not _contains_value(right["name"], new)
                ):
                    row_semantic_error = True
            actual_changed = {
                key
                for key in set(left_attributes) | set(right_attributes)
                if normalize_text(left_attributes.get(key))
                != normalize_text(right_attributes.get(key))
            }
            if target_keys != actual_changed:
                row_semantic_error = True
            for key in set(left_attributes) - actual_changed:
                if _contains_value(left["name"], left_attributes[key]) and not _contains_value(
                    right["name"], left_attributes[key]
                ):
                    row_semantic_error = True
            if row_semantic_error:
                semantic_metadata_errors += 1

    duplicate_ids = int(items["id"].duplicated(keep=False).sum())
    duplicate_cards = int(pd.Series(canonical_cards).duplicated(keep=False).sum())
    structural_errors = (
        invalid_items
        + duplicate_ids
        + duplicate_cards
        + missing_ids
        + cross_category
        + self_pairs
        + invalid_targets
        + duplicate_pairs
        + metadata_errors
        + semantic_metadata_errors
    )
    return {
        "version": "rule_first_generated_pairs_validation_v3",
        "items_path": str(items_path.resolve()),
        "pairs_path": str(pairs_path.resolve()),
        "items": int(len(items)),
        "pairs": int(len(pairs)),
        "categories": left_category.value_counts().sort_index().to_dict(),
        "invalid_items": invalid_items,
        "duplicate_id_rows": duplicate_ids,
        "duplicate_full_card_rows": duplicate_cards,
        "missing_pair_item_references": missing_ids,
        "cross_category_pairs": cross_category,
        "self_pairs": self_pairs,
        "invalid_targets": invalid_targets,
        "duplicate_unordered_pair_rows": duplicate_pairs,
        "metadata_rows": metadata_rows,
        "metadata_errors": metadata_errors,
        "semantic_metadata_error_rows": semantic_metadata_errors,
        "valid": structural_errors == 0 and len(items) == 2 * len(pairs),
    }
