from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
from pathlib import Path
from typing import Any

import requests

from .normalization import normalize_text, parse_attributes
from .pair_rules import MutationRule
from .rule_values import canonical_target_value


REASONING_EFFORTS = {"minimal", "low", "medium", "high", "xhigh", "max"}


def _apply_generation_controls(
    payload: dict[str, Any], reasoning_effort: str | None
) -> None:
    if reasoning_effort is None:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        return
    if reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(f"Unsupported reasoning effort: {reasoning_effort}")
    payload["reasoning"] = {"effort": reasoning_effort, "exclude": True}


def _request_retry_delay(error: Exception, attempt: int) -> float | None:
    """Return a safe retry delay, or None for a permanent HTTP failure."""

    if isinstance(error, requests.HTTPError) and error.response is not None:
        status = int(error.response.status_code)
        if 400 <= status < 500 and status not in {408, 409, 425, 429}:
            return None
        if status == 429:
            retry_after = error.response.headers.get("Retry-After", "").strip()
            try:
                return max(0.0, min(60.0, float(retry_after)))
            except ValueError:
                pass
    return min(15.0, 0.5 * (2 ** (attempt - 1))) + random.random() * 0.25


def _request_error_detail(error: Exception) -> str:
    """Return useful hosted-API diagnostics without exposing request secrets."""

    if not isinstance(error, requests.HTTPError) or error.response is None:
        return f"{type(error).__name__}: {error}"

    response = error.response
    details: dict[str, Any] = {"http_status": int(response.status_code)}
    retry_after = response.headers.get("Retry-After", "").strip()
    if retry_after:
        details["retry_after"] = retry_after[:64]
    request_id = (
        response.headers.get("x-request-id", "").strip()
        or response.headers.get("cf-ray", "").strip()
    )
    if request_id:
        details["request_id"] = request_id[:128]

    try:
        payload = response.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        api_error = payload.get("error")
        if isinstance(api_error, dict):
            for key in ("code", "message"):
                value = api_error.get(key)
                if isinstance(value, (str, int, float)):
                    details[f"error_{key}"] = value
            metadata = api_error.get("metadata")
            if isinstance(metadata, dict):
                for key in ("provider_name", "raw"):
                    value = metadata.get(key)
                    if isinstance(value, (str, int, float)):
                        details["provider_name" if key == "provider_name" else "provider_raw"] = value
        elif isinstance(api_error, str):
            details["error_message"] = api_error
    elif getattr(response, "text", ""):
        details["response_text"] = str(response.text)

    rendered = json.dumps(details, ensure_ascii=False, sort_keys=True)
    rendered = re.sub(
        r"(?i)bearer\s+[a-z0-9._-]+|(?:sk-or-v1-|sk-)[a-z0-9._-]{8,}",
        "[redacted]",
        rendered,
    )
    return rendered[:1200]


def _shared_allowed_product_types(rules: list[MutationRule]) -> list[str]:
    scoped = [rule.allowed_product_types for rule in rules if rule.allowed_product_types]
    if not scoped:
        return []
    display_by_normalized = {
        value.casefold(): value for values in scoped for value in values
    }
    shared = set(display_by_normalized)
    for values in scoped:
        shared &= {value.casefold() for value in values}
    return [display_by_normalized[value] for value in sorted(shared)]


def _shared_allowed_anchor_context_keys(rules: list[MutationRule]) -> list[str]:
    scoped = [
        rule.allowed_anchor_context_keys
        for rule in rules
        if rule.allowed_anchor_context_keys
    ]
    if not scoped:
        return []
    shared = set(scoped[0])
    for values in scoped[1:]:
        shared &= set(values)
    return sorted(shared)


def _required_anchor_context_keys(rules: list[MutationRule]) -> list[str]:
    display_by_normalized = {
        normalize_text(key): key
        for rule in rules
        for key in rule.required_anchor_context_keys
    }
    return [
        display_by_normalized[key]
        for key in sorted(display_by_normalized)
    ]


def _transition_endpoint_values(rule: MutationRule) -> list[str]:
    display_by_normalized = {
        normalize_text(value): value
        for transition in rule.allowed_value_transitions
        for value in transition
    }
    return [display_by_normalized[key] for key in sorted(display_by_normalized)]


def _prompt_source_examples(rule: MutationRule) -> list[dict[str, Any]]:
    """Keep catalog evidence raw, but make unitless count examples unambiguous."""

    result: list[dict[str, Any]] = []
    dimension_concepts = {
        "case_diameter",
        "diameter",
        "length",
        "length_mm",
        "wheel_diameter",
        "width",
    }
    for source in rule.source_examples:
        example = dict(source)
        if (
            rule.concept == "package_quantity"
            or rule.concept in dimension_concepts
            or rule.target_value_domain
        ):
            valid = True
            for field in ("target_value_a", "target_value_b"):
                raw = str(example.get(field) or "").strip()
                canonical = canonical_target_value(
                    rule.concept,
                    rule.allowed_product_types[0]
                    if rule.allowed_product_types
                    else "",
                    raw,
                    rule.target_value_domain,
                )
                if (
                    canonical is None
                    and rule.concept == "package_quantity"
                    and normalize_text(raw).isdigit()
                ):
                    canonical = f"{int(normalize_text(raw))} шт"
                if canonical is None:
                    valid = False
                    break
                example[field] = canonical
            if not valid:
                continue
        result.append(example)
    return result


def _prompt_rule_payload(rule: MutationRule) -> dict[str, Any]:
    payload = rule.prompt_payload()
    # Raw examples remain in generation metadata for exact provenance.  The
    # prompt receives only the effective, contract-valid representation below.
    payload.pop("source_examples", None)
    return payload


def discover_model(base_url: str, requested_model: str | None, timeout: float) -> str:
    if requested_model:
        return requested_model
    session = requests.Session()
    session.trust_env = False
    response = session.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
    response.raise_for_status()
    models = [str(entry["id"]) for entry in response.json().get("data", [])]
    if len(models) != 1:
        raise ValueError(f"Expected one served model, found {models}; pass --model")
    return models[0]


def _parse_json_content(content: str) -> Any:
    text = content.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    return json.loads(text)


def generated_item_schema(category: str, attribute_keys: list[str]) -> dict[str, Any]:
    properties = {
        key: {"type": "string", "minLength": 1, "maxLength": 1000}
        for key in attribute_keys
    }
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1, "maxLength": 240},
            "attributes": {
                "type": "object",
                "properties": properties,
                "required": attribute_keys,
                "additionalProperties": False,
            },
            "category": {"type": "string", "const": category},
        },
        "required": ["name", "attributes", "category"],
        "additionalProperties": False,
    }


def mutated_item_schema(
    category: str,
    attribute_keys: list[str],
    rules: list[MutationRule],
) -> dict[str, Any]:
    item_schema = generated_item_schema(category, attribute_keys)
    for rule in rules:
        transition_values = _transition_endpoint_values(rule)
        if transition_values and rule.attribute_key in attribute_keys:
            item_schema["properties"]["attributes"]["properties"][
                rule.attribute_key
            ]["enum"] = transition_values
    rule_ids = [rule.generation_rule_id for rule in rules]
    concepts = [rule.concept for rule in rules]
    all_rules_have_transitions = all(rule.allowed_value_transitions for rule in rules)
    transition_values = sorted(
        {
            value
            for rule in rules
            for value in _transition_endpoint_values(rule)
        },
        key=normalize_text,
    )
    application_value_schema: dict[str, Any] = {
        "type": "string",
        "minLength": 1,
        "maxLength": 500,
    }
    if all_rules_have_transitions and transition_values:
        application_value_schema["enum"] = transition_values
    return {
        "type": "object",
        "properties": {
            "item": item_schema,
            "applications": {
                "type": "array",
                "minItems": len(rules),
                "maxItems": len(rules),
                "items": {
                    "type": "object",
                    "properties": {
                        "generation_rule_id": {
                            "type": "string",
                            "enum": rule_ids,
                        },
                        "concept": {"type": "string", "enum": concepts},
                        "original_value": dict(application_value_schema),
                        "new_value": dict(application_value_schema),
                        "attribute_key": {
                            "type": "string",
                            "enum": attribute_keys,
                        },
                    },
                    "required": [
                        "generation_rule_id",
                        "concept",
                        "original_value",
                        "new_value",
                        "attribute_key",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["item", "applications"],
        "additionalProperties": False,
    }


def rule_anchor_schema(category: str, rules: list[MutationRule]) -> dict[str, Any]:
    rule_ids = [rule.generation_rule_id for rule in rules]
    concepts = [rule.concept for rule in rules]
    target_keys = [rule.attribute_key for rule in rules]
    allowed_product_types = _shared_allowed_product_types(rules)
    target_key_set = set(target_keys)
    allowed_context_keys = [
        key
        for key in _shared_allowed_anchor_context_keys(rules)
        if key not in target_key_set and key != "Тип товара"
    ]
    required_context_keys = [
        key
        for key in _required_anchor_context_keys(rules)
        if key not in target_key_set and key != "Тип товара"
    ]
    context_keys = list(dict.fromkeys([*allowed_context_keys, *required_context_keys]))
    strict_context_allowlist = any(
        rule.allowed_anchor_context_keys for rule in rules
    )
    product_type_schema: dict[str, Any] = {
        "type": "string",
        "minLength": 1,
        "maxLength": 160,
    }
    if allowed_product_types:
        product_type_schema["enum"] = allowed_product_types
    return {
        "type": "object",
        "properties": {
            "item": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1, "maxLength": 240},
                    "attributes": {
                        "type": "object",
                        "properties": {
                            "Тип товара": product_type_schema,
                            **{
                                rule.attribute_key: {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 500,
                                    **(
                                        {"enum": _transition_endpoint_values(rule)}
                                        if rule.allowed_value_transitions
                                        else {}
                                    ),
                                }
                                for rule in rules
                            },
                            **{
                                key: {"type": "string", "minLength": 1, "maxLength": 500}
                                for key in context_keys
                            },
                        },
                        "required": list(
                            dict.fromkeys(
                                ["Тип товара", *target_keys, *required_context_keys]
                            )
                        ),
                        "additionalProperties": (
                            False
                            if strict_context_allowlist
                            else {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 1000,
                            }
                        ),
                        "minProperties": max(3, len(target_keys) + 1),
                        "maxProperties": (
                            1 + len(set(target_keys) | set(context_keys))
                            if strict_context_allowlist
                            else 12
                        ),
                    },
                    "category": {"type": "string", "const": category},
                },
                "required": ["name", "attributes", "category"],
                "additionalProperties": False,
            },
            "product_type": product_type_schema,
            "evidence": {
                "type": "array",
                "minItems": len(rules),
                "maxItems": len(rules),
                "items": {
                    "type": "object",
                    "properties": {
                        "generation_rule_id": {"type": "string", "enum": rule_ids},
                        "concept": {"type": "string", "enum": concepts},
                        "attribute_key": {"type": "string", "enum": target_keys},
                        "attribute_value": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 500,
                        },
                    },
                    "required": [
                        "generation_rule_id",
                        "concept",
                        "attribute_key",
                        "attribute_value",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["item", "product_type", "evidence"],
        "additionalProperties": False,
    }


def _compact_card(row: dict[str, Any], *, max_attributes: int = 40) -> dict[str, Any]:
    attributes = parse_attributes(row["attributes"])
    return {
        "id": int(row["id"]),
        "name": str(row["name"])[:500],
        "attributes": {
            key: value[:600]
            for key, value in list(attributes.items())[:max_attributes]
        },
    }


def build_generation_prompt(
    anchor: dict[str, Any],
    examples: list[dict[str, Any]],
    *,
    feedback: list[str] | None = None,
) -> str:
    attributes = parse_attributes(anchor["attributes"])
    payload = {
        "task": "generate_new_standalone_item",
        "category": str(anchor["category"]),
        "subtype": str(anchor["subtype"]),
        "required_attribute_keys": list(attributes),
        "required_output_json": {
            "name": "непустая строка с названием нового товара",
            "attributes": {
                key: f"непустая строка со значением свойства {key}"
                for key in attributes
            },
            "category": str(anchor["category"]),
        },
        "output_contract": {
            "top_level_keys_exactly": ["name", "attributes", "category"],
            "attribute_keys_exactly": list(attributes),
            "all_attribute_values_are_strings": True,
            "no_markdown_or_text_outside_json": True,
        },
        "schema_donor": _compact_card(anchor),
        "style_examples": [_compact_card(row) for row in examples],
    }
    if feedback:
        payload["previous_attempt_rejection_reasons"] = feedback
    return (
        "Создай новую карточку по следующему заданию. Значения schema donor и "
        "примеров нужны только для понимания формата; не копируй сам товар.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def build_mutation_prompt(
    item: dict[str, Any],
    rules: list[MutationRule],
    evidence: list[dict[str, Any]],
    *,
    feedback: list[str] | None = None,
    diversity_nonce: str | None = None,
    global_rejections: list[dict[str, Any]] | None = None,
) -> str:
    attributes = parse_attributes(item["attributes"])
    if diversity_nonce is None:
        diversity_nonce = hashlib.blake2s(
            json.dumps(
                {
                    "item_id": item.get("id"),
                    "rule_ids": [rule.generation_rule_id for rule in rules],
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8"),
            digest_size=16,
        ).hexdigest()
    payload: dict[str, Any] = {
        "task": "create_counterfactual_pair_item",
        "target_label": rules[0].label,
        "category": str(item["category"]),
        "base_item": {
            "id": int(item["id"]),
            "name": str(item["name"]),
            "attributes": attributes,
            "category": str(item["category"]),
        },
        "rules": [_prompt_rule_payload(rule) for rule in rules],
        "verified_rule_targets": evidence,
        "diversity_nonce_not_for_output": diversity_nonce,
        "diversity_instruction": (
            "Используй nonce только как источник случайности при выборе новых "
            "целевых значений. Не повторяй ни один полный набор переходов из "
            "previous_global_rejections ни в прямом, ни в обратном направлении. "
            "Никогда не выводи nonce, его части или числа из него."
        ),
        "required_attribute_keys": list(attributes),
        "required_output_json": {
            "item": {
                "name": "непустая строка с названием изменённого товара",
                "attributes": {
                    key: f"непустая строка со значением свойства {key}"
                    for key in attributes
                },
                "category": str(item["category"]),
            },
            "applications": [
                {
                    "generation_rule_id": rule.generation_rule_id,
                    "concept": rule.concept,
                    "original_value": (
                        "точное старое значение, равное одному концу разрешённой пары"
                        if rule.allowed_value_transitions
                        else "точное старое значение"
                    ),
                    "new_value": (
                        "точное значение другого конца той же разрешённой пары"
                        if rule.allowed_value_transitions
                        else "точное новое значение"
                    ),
                    "attribute_key": rule.attribute_key,
                }
                for rule in rules
            ],
        },
        "output_contract": {
            "top_level_keys_exactly": ["item", "applications"],
            "item_keys_exactly": ["name", "attributes", "category"],
            "attribute_keys_exactly": list(attributes),
            "applications_count_exactly": len(rules),
            "no_markdown_or_text_outside_json": True,
        },
        "local_validation_contract": {
            "apply_every_requested_rule_exactly_once": True,
            "preserve_category": True,
            "preserve_exact_attribute_key_set": True,
            "preserve_every_unrelated_attribute_value": True,
            "change_exact_verified_attribute_key_for_each_rule": True,
            "changed_attribute_keys_must_be_disjoint_between_rules": True,
            "old_value_must_disappear_from_title": True,
            "new_value_must_appear_verbatim_in_title": True,
            "all_unrelated_attribute_values_already_in_title_must_remain": True,
            "finite_domain_values_must_use_distinct_canonical_members": any(
                bool(rule.target_value_domain) for rule in rules
            ),
            "allowed_value_transitions_are_exact_and_unordered": {
                rule.generation_rule_id: [
                    list(transition) for transition in rule.allowed_value_transitions
                ]
                for rule in rules
                if rule.allowed_value_transitions
            },
            "package_quantity_requires_explicit_piece_unit": any(
                rule.concept == "package_quantity" for rule in rules
            ),
            "physical_dimensions_require_explicit_unit": any(
                rule.concept
                in {
                    "case_diameter",
                    "diameter",
                    "length",
                    "length_mm",
                    "wheel_diameter",
                    "width",
                }
                for rule in rules
            ),
            "do_not_repeat_previous_global_rejections": True,
            "diversity_nonce_must_not_appear_in_output": True,
        },
    }
    if global_rejections:
        payload["previous_global_rejections"] = global_rejections
    if feedback:
        payload["previous_attempt_rejection_reasons"] = feedback
    return (
        "Создай вторую карточку контролируемой пары из исходной карточки. "
        "Примени все переданные правила, выполни diversity_instruction и верни "
        "только JSON по схеме.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def build_rule_anchor_prompt(
    style_donor: dict[str, Any],
    rules: list[MutationRule],
    *,
    feedback: list[str] | None = None,
    diversity_nonce: str | None = None,
    global_rejections: list[dict[str, Any]] | None = None,
) -> str:
    category = str(style_donor["category"])
    allowed_product_types = _shared_allowed_product_types(rules)
    target_keys = {rule.attribute_key for rule in rules}
    allowed_context_keys = [
        key
        for key in _shared_allowed_anchor_context_keys(rules)
        if key not in target_keys and key != "Тип товара"
    ]
    required_context_keys = [
        key
        for key in _required_anchor_context_keys(rules)
        if key not in target_keys and key != "Тип товара"
    ]
    context_keys = list(dict.fromkeys([*allowed_context_keys, *required_context_keys]))
    strict_context_allowlist = any(
        rule.allowed_anchor_context_keys for rule in rules
    )
    name_value_order = [
        "Тип товара",
        *context_keys,
        *[rule.attribute_key for rule in rules],
    ]
    if diversity_nonce is None:
        diversity_nonce = hashlib.blake2s(
            str(style_donor.get("id", "")).encode("utf-8"), digest_size=16
        ).hexdigest()
    targets = [
        {
            "generation_rule_id": rule.generation_rule_id,
            "concept": rule.concept,
            "required_attribute_key": rule.attribute_key,
            "applicability_instruction": rule.anchor_hint,
            "allowed_product_types": list(rule.allowed_product_types),
            "allowed_anchor_context_keys": list(
                rule.allowed_anchor_context_keys
            ),
            "required_anchor_context_keys": list(
                rule.required_anchor_context_keys
            ),
            "forbidden_anchor_attribute_patterns": list(
                rule.forbidden_anchor_attribute_patterns
            ),
            "target_value_domain": list(rule.target_value_domain),
            "allowed_value_transitions": [
                list(transition) for transition in rule.allowed_value_transitions
            ],
            "mutation_guard": rule.required_postcondition,
        }
        for rule in rules
    ]
    payload: dict[str, Any] = {
        "task": "generate_rule_applicable_anchor_item",
        "category": category,
        "rules": [_prompt_rule_payload(rule) for rule in rules],
        "required_targets": targets,
        "allowed_product_types": allowed_product_types,
        "allowed_anchor_context_keys": allowed_context_keys,
        "required_anchor_context_keys": required_context_keys,
        "required_name_value_order": name_value_order,
        "diversity_nonce_not_for_output": diversity_nonce,
        "diversity_instruction": (
            "Используй nonce только как источник случайности: выбери естественно "
            "звучащий, подходящий типу бренд и не самые шаблонные целевые значения. "
            "Не повторяй ни один полный набор исходных и новых целевых значений из "
            "previous_global_rejections ни в прямом, ни в обратном направлении. "
            "Никогда не выводи сам nonce, его части или числа из него."
        ),
        "source_scope_examples": [
            {
                "generation_rule_id": rule.generation_rule_id,
                "examples": examples,
            }
            for rule in rules
            if (examples := _prompt_source_examples(rule))
        ],
        "style_donor_only": _compact_card(style_donor),
        "required_output_json": {
            "item": {
                "name": (
                    "ровно значения атрибутов из required_name_value_order в указанном "
                    "порядке, соединённые одиночными пробелами; без других слов и чисел"
                ),
                "attributes": {
                    "Тип товара": (
                        "одно точное значение из allowed_product_types"
                        if allowed_product_types
                        else "конкретный тип товара, подходящий всем правилам"
                    ),
                    **{
                        rule.attribute_key: (
                            "одно точное исходное значение из любого конца "
                            f"allowed_value_transitions правила {rule.generation_rule_id}"
                            if rule.allowed_value_transitions
                            else f"явное конкретное значение для {rule.concept}"
                        )
                        for rule in rules
                    },
                    **(
                        {
                            key: (
                                "нейтральное реалистичное контекстное значение"
                            )
                            for key in (
                                required_context_keys
                                or allowed_context_keys[:1]
                            )
                        }
                        if required_context_keys or allowed_context_keys
                        else {}
                        if strict_context_allowlist
                        else {
                            "другие реалистичные свойства": (
                                "непустые согласованные строки"
                            )
                        }
                    ),
                },
                "category": category,
            },
            "product_type": (
                "точное значение из allowed_product_types и attributes['Тип товара']"
                if allowed_product_types
                else "точное значение attributes['Тип товара']"
            ),
            "evidence": [
                {
                    "generation_rule_id": rule.generation_rule_id,
                    "concept": rule.concept,
                    "attribute_key": rule.attribute_key,
                    "attribute_value": f"точное полное значение attributes[{rule.attribute_key!r}]",
                }
                for rule in rules
            ],
        },
        "local_validation_contract": {
            "product_type_must_be_explicit_and_applicable_to_every_rule": True,
            "product_type_must_equal_one_allowed_value": bool(allowed_product_types),
            "name_must_start_with_exact_product_type": True,
            "sold_item_subtype_and_target_meaning_must_match_source_examples": True,
            "required_attribute_keys_must_exist_exactly": [
                rule.attribute_key for rule in rules
            ],
            "required_anchor_context_keys_must_exist": required_context_keys,
            "attribute_keys_must_equal_exactly": (
                name_value_order if strict_context_allowlist else None
            ),
            "additional_attribute_keys_forbidden": bool(
                strict_context_allowlist
            ),
            "one_distinct_attribute_key_per_rule": True,
            "evidence_value_must_equal_full_attribute_value": True,
            "every_target_value_must_appear_verbatim_in_name": True,
            "finite_domain_value_must_be_exact_canonical_member": any(
                bool(rule.target_value_domain) for rule in rules
            ),
            "transition_constrained_anchor_value_must_be_exact_endpoint": {
                rule.generation_rule_id: [
                    list(transition) for transition in rule.allowed_value_transitions
                ]
                for rule in rules
                if rule.allowed_value_transitions
            },
            "package_quantity_requires_explicit_piece_unit": any(
                rule.concept == "package_quantity" for rule in rules
            ),
            "physical_dimensions_require_explicit_unit": any(
                rule.concept
                in {
                    "case_diameter",
                    "diameter",
                    "length",
                    "length_mm",
                    "wheel_diameter",
                    "width",
                }
                for rule in rules
            ),
            "reject_package_or_service_field_when_rule_targets_product_spec": True,
            "forbid_sku_and_article_in_name_and_attributes": True,
            "forbid_every_declared_dependent_attribute_pattern": True,
            "non_target_keys_must_come_from_allowed_anchor_context_keys": bool(
                strict_context_allowlist
            ),
            "name_is_exact_concatenation_of_attribute_values": True,
            "do_not_repeat_previous_global_rejections": True,
            "diversity_nonce_must_not_appear_in_output": True,
            "attribute_count_between_3_and_12": True,
            "no_markdown_or_text_outside_json": True,
        },
    }
    if global_rejections:
        payload["previous_global_rejections"] = global_rejections
    if feedback:
        payload["previous_attempt_rejection_reasons"] = feedback
    return (
        "Сначала создай новую исходную карточку специально под переданные правила. "
        "Не применяй правила: значения пока должны быть исходными. Выбери один "
        "конкретный тип товара точно из allowed_product_types, если список задан. "
        "Выполни diversity_instruction, но не выводи diversity_nonce_not_for_output. "
        "Непустые source_scope_examples задают допустимый смысл свойства, но не копируй их "
        "бренды, модели или числа. Карточка-пример задаёт только стиль категории. Верни только JSON.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def parse_generated_item(
    response: dict[str, Any],
    *,
    expected_category: str,
    expected_keys: list[str],
) -> dict[str, Any]:
    choices = response.get("choices") or []
    if len(choices) != 1:
        raise ValueError("Qwen response must contain exactly one choice")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("Qwen response content is not a string")
    value = _parse_json_content(content)
    if not isinstance(value, dict) or set(value) != {"name", "attributes", "category"}:
        raise ValueError("Generated item must contain exactly name, attributes, category")
    if value["category"] != expected_category:
        raise ValueError("Generated category differs from requested category")
    attributes = value["attributes"]
    if not isinstance(attributes, dict) or set(attributes) != set(expected_keys):
        raise ValueError("Generated attributes do not exactly match the donor key set")
    if not isinstance(value["name"], str) or not value["name"].strip():
        raise ValueError("Generated name is empty")
    if any(not isinstance(item, str) or not item.strip() for item in attributes.values()):
        raise ValueError("Every generated attribute value must be a non-empty string")
    value["attributes"] = {key: attributes[key] for key in expected_keys}
    usage = response.get("usage") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return {
        "item": value,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "reasoning_tokens": int(completion_details.get("reasoning_tokens") or 0),
        "request_cost": float(usage.get("cost") or 0.0),
        "response_id": str(response.get("id") or ""),
    }


def parse_mutated_item(
    response: dict[str, Any],
    *,
    expected_category: str,
    expected_keys: list[str],
    rules: list[MutationRule],
) -> dict[str, Any]:
    choices = response.get("choices") or []
    if len(choices) != 1:
        raise ValueError("Qwen response must contain exactly one choice")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("Qwen response content is not a string")
    value = _parse_json_content(content)
    if not isinstance(value, dict) or set(value) != {"item", "applications"}:
        raise ValueError("Mutation response must contain exactly item and applications")
    item = value["item"]
    if not isinstance(item, dict) or set(item) != {"name", "attributes", "category"}:
        raise ValueError("Mutated item must contain exactly name, attributes, category")
    if item["category"] != expected_category:
        raise ValueError("Mutated category differs from requested category")
    attributes = item["attributes"]
    if not isinstance(attributes, dict) or set(attributes) != set(expected_keys):
        raise ValueError("Mutated attributes do not exactly match the base key set")
    if not isinstance(item["name"], str) or not item["name"].strip():
        raise ValueError("Mutated item name is empty")
    if any(not isinstance(raw, str) or not raw.strip() for raw in attributes.values()):
        raise ValueError("Every mutated attribute value must be a non-empty string")
    applications = value["applications"]
    if not isinstance(applications, list) or len(applications) != len(rules):
        raise ValueError("Mutation response must describe every requested rule")
    required_application_keys = {
        "generation_rule_id",
        "concept",
        "original_value",
        "new_value",
        "attribute_key",
    }
    for application in applications:
        if not isinstance(application, dict) or set(application) != required_application_keys:
            raise ValueError("A mutation application has an invalid shape")
    item["attributes"] = {key: attributes[key] for key in expected_keys}
    usage = response.get("usage") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return {
        "item": item,
        "applications": applications,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "reasoning_tokens": int(completion_details.get("reasoning_tokens") or 0),
        "request_cost": float(usage.get("cost") or 0.0),
        "response_id": str(response.get("id") or ""),
    }


def parse_rule_anchor(
    response: dict[str, Any],
    *,
    expected_category: str,
    rules: list[MutationRule],
) -> dict[str, Any]:
    choices = response.get("choices") or []
    if len(choices) != 1:
        raise ValueError("Qwen response must contain exactly one choice")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("Qwen response content is not a string")
    value = _parse_json_content(content)
    if not isinstance(value, dict) or set(value) != {"item", "product_type", "evidence"}:
        raise ValueError("Rule anchor must contain exactly item, product_type, evidence")
    item = value["item"]
    if not isinstance(item, dict) or set(item) != {"name", "attributes", "category"}:
        raise ValueError("Rule anchor item must contain exactly name, attributes, category")
    if item["category"] != expected_category:
        raise ValueError("Rule anchor category differs from requested category")
    if not isinstance(item["name"], str) or not item["name"].strip():
        raise ValueError("Rule anchor name is empty")
    attributes = item["attributes"]
    if not isinstance(attributes, dict) or not 3 <= len(attributes) <= 12:
        raise ValueError("Rule anchor attributes must contain 3 to 12 fields")
    if any(
        not isinstance(key, str)
        or not key.strip()
        or not isinstance(raw, str)
        or not raw.strip()
        for key, raw in attributes.items()
    ):
        raise ValueError("Every rule anchor attribute key/value must be a non-empty string")
    evidence = value["evidence"]
    if not isinstance(evidence, list) or len(evidence) != len(rules):
        raise ValueError("Rule anchor must describe every requested rule")
    required_keys = {
        "generation_rule_id", "concept", "attribute_key", "attribute_value"
    }
    if any(not isinstance(entry, dict) or set(entry) != required_keys for entry in evidence):
        raise ValueError("Rule anchor evidence has an invalid shape")
    if not isinstance(value["product_type"], str) or not value["product_type"].strip():
        raise ValueError("Rule anchor product_type is empty")
    usage = response.get("usage") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return {
        "item": item,
        "product_type": value["product_type"],
        "evidence": evidence,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "reasoning_tokens": int(completion_details.get("reasoning_tokens") or 0),
        "request_cost": float(usage.get("cost") or 0.0),
        "response_id": str(response.get("id") or ""),
    }
class QwenItemClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        system_prompt: str,
        timeout: float = 120.0,
        retries: int = 8,
        temperature: float = 0.7,
        max_tokens: int = 1400,
        structured_output: bool = True,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.url = f"{self.base_url}/chat/completions"
        self.model = model
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.retries = retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.structured_output = structured_output
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort
        self.local = threading.local()

    def session(self) -> requests.Session:
        session = getattr(self.local, "session", None)
        if session is None:
            session = requests.Session()
            # Local vLLM must ignore workstation proxy settings, while hosted
            # APIs such as OpenRouter are reachable through the managed proxy.
            session.trust_env = bool(self.api_key)
            if self.api_key:
                session.headers.update(
                    {
                        "Authorization": f"Bearer {self.api_key}",
                        "User-Agent": "product-matching-item-pipeline/1.0",
                        "HTTP-Referer": "https://github.com/dinnksis/product_matching",
                        "X-Title": "product_matching",
                    }
                )
            self.local.session = session
        return session

    def generate(
        self,
        prompt: str,
        *,
        category: str,
        attribute_keys: list[str],
        seed: int,
    ) -> dict[str, Any]:
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
        }
        _apply_generation_controls(payload, self.reasoning_effort)
        if self.structured_output:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "generated_product_item",
                    "strict": True,
                    "schema": generated_item_schema(category, attribute_keys),
                },
            }
        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self.session().post(
                    self.url, json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                parsed = parse_generated_item(
                    response.json(),
                    expected_category=category,
                    expected_keys=attribute_keys,
                )
                return {
                    **parsed,
                    "request_attempts": attempt,
                    "latency_seconds": time.perf_counter() - started,
                }
            except Exception as error:
                last_error = error
                if attempt < self.retries:
                    delay = _request_retry_delay(error, attempt)
                    if delay is None:
                        break
                    time.sleep(delay)
        raise RuntimeError(
            f"Qwen item request failed after {attempt} attempts; "
            f"last_error={_request_error_detail(last_error)}"
        ) from last_error


class QwenPairClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        system_prompt: str,
        timeout: float = 120.0,
        retries: int = 8,
        temperature: float = 0.25,
        max_tokens: int = 1800,
        structured_output: bool = True,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.url = f"{self.base_url}/chat/completions"
        self.model = model
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.retries = retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.structured_output = structured_output
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort
        self.local = threading.local()

    def session(self) -> requests.Session:
        session = getattr(self.local, "session", None)
        if session is None:
            session = requests.Session()
            session.trust_env = bool(self.api_key)
            if self.api_key:
                session.headers.update(
                    {
                        "Authorization": f"Bearer {self.api_key}",
                        "User-Agent": "product-matching-item-pipeline/1.0",
                        "HTTP-Referer": "https://github.com/dinnksis/product_matching",
                        "X-Title": "product_matching",
                    }
                )
            self.local.session = session
        return session

    def generate_anchor(
        self,
        prompt: str,
        *,
        category: str,
        rules: list[MutationRule],
        seed: int,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": max(self.temperature, 0.45),
            "top_p": 0.9,
            "max_tokens": self.max_tokens,
            "seed": int(seed),
        }
        _apply_generation_controls(payload, self.reasoning_effort)
        if self.structured_output:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "rule_applicable_anchor_item",
                    "strict": True,
                    "schema": rule_anchor_schema(category, rules),
                },
            }
        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                attempt_payload = {
                    **payload,
                    "seed": int(seed) + attempt - 1,
                }
                response = self.session().post(
                    self.url, json=attempt_payload, timeout=self.timeout
                )
                response.raise_for_status()
                parsed = parse_rule_anchor(
                    response.json(), expected_category=category, rules=rules
                )
                return {
                    **parsed,
                    "request_attempts": attempt,
                    "latency_seconds": time.perf_counter() - started,
                }
            except Exception as error:
                last_error = error
                if attempt < self.retries:
                    delay = _request_retry_delay(error, attempt)
                    if delay is None:
                        break
                    time.sleep(delay)
        raise RuntimeError(
            f"Qwen rule-anchor request failed after {attempt} attempts; "
            f"last_error={_request_error_detail(last_error)}"
        ) from last_error

    def mutate(
        self,
        prompt: str,
        *,
        category: str,
        attribute_keys: list[str],
        rules: list[MutationRule],
        seed: int,
    ) -> dict[str, Any]:
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
        }
        _apply_generation_controls(payload, self.reasoning_effort)
        if self.structured_output:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "mutated_product_pair_item",
                    "strict": True,
                    "schema": mutated_item_schema(category, attribute_keys, rules),
                },
            }
        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                attempt_payload = {
                    **payload,
                    "seed": int(seed) + attempt - 1,
                }
                response = self.session().post(
                    self.url, json=attempt_payload, timeout=self.timeout
                )
                response.raise_for_status()
                parsed = parse_mutated_item(
                    response.json(),
                    expected_category=category,
                    expected_keys=attribute_keys,
                    rules=rules,
                )
                return {
                    **parsed,
                    "request_attempts": attempt,
                    "latency_seconds": time.perf_counter() - started,
                }
            except Exception as error:
                last_error = error
                if attempt < self.retries:
                    delay = _request_retry_delay(error, attempt)
                    if delay is None:
                        break
                    time.sleep(delay)
        raise RuntimeError(
            f"Qwen pair mutation request failed after {attempt} attempts; "
            f"last_error={_request_error_detail(last_error)}"
        ) from last_error


def load_system_prompt(path: Path) -> str:
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt is empty: {path}")
    return prompt
