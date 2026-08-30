from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .generate import _atomic_json, _atomic_parquet, sha256_text, utc_now
from .normalization import (
    canonical_json_dumps,
    json_dumps,
    normalize_text,
    parse_attributes,
    stable_hash64,
)
from .pair_rules import (
    MutationRule,
    catalog_summary,
    load_mutation_rules,
    rules_are_product_compatible,
)
from .rule_schedule import (
    LABEL_QUOTA_POLICY_VERSION,
    SCHEDULE_VERSION,
    ScheduledRuleBundle,
    build_balanced_rule_schedule,
)
from .pair_validation import (
    MutationValidation,
    RuleAnchorValidation,
    validate_mutation,
    validate_pair_dataset,
    validate_rule_anchor,
)
from .qwen import QwenPairClient, build_mutation_prompt, build_rule_anchor_prompt
from .rule_values import canonical_target_value


SEMANTIC_SIGNATURE_VERSION = "category_product_type_canonical_applications_v2"
ATTEMPT_DIVERSITY_VERSION = (
    "per_outer_attempt_nonce_bundle_rejections_global_card_v3"
)
MAX_GLOBAL_REJECTION_PROMPT_ITEMS = 64


SemanticScopeKey = tuple[str, str, tuple[tuple[str, str], ...]]


@dataclass(frozen=True)
class AttemptDiversityContext:
    forbidden_semantic_signatures: frozenset[str]
    forbidden_card_keys: frozenset[str]
    global_rejections: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PairGenerationTask:
    task_index: int
    mutated_id: int
    seed: int
    anchor: dict[str, Any]


def _canonical_card_key(item: dict[str, Any]) -> str:
    """Return a global product-card key independent of source category.

    Category is deliberately excluded: generated cards from two category-scoped
    rules must still collide when their visible name and attributes are the same.
    This matches the stricter freeze-time uniqueness contract.
    """

    attributes = {
        normalize_text(key): normalize_text(value)
        for key, value in parse_attributes(item["attributes"]).items()
    }
    return canonical_json_dumps(
        {
            "name": normalize_text(item["name"]),
            "attributes": attributes,
        }
    )


def _semantic_pair_signature(
    category: Any,
    product_type: Any,
    applications: list[dict[str, Any]],
) -> str:
    """Return a direction- and rule-order-invariant mutation signature."""

    normalized_applications = []
    for application in applications:
        concept = normalize_text(application.get("concept"))
        values = []
        for field in ("original_value", "new_value"):
            raw_value = application.get(field)
            canonical = canonical_target_value(concept, product_type, raw_value)
            values.append(canonical or normalize_text(raw_value))
        normalized_applications.append(
            (
                concept,
                normalize_text(application.get("attribute_key")),
                tuple(sorted(values)),
            )
        )
    normalized_applications.sort()
    payload = {
        "category": normalize_text(category),
        "product_type": normalize_text(product_type),
        "applications": normalized_applications,
    }
    return sha256_text(canonical_json_dumps(payload))


def _semantic_scope_key(
    category: Any,
    product_type: Any,
    applications: list[dict[str, Any]],
) -> SemanticScopeKey:
    """Group candidates that can produce the same semantic signature."""

    targets = sorted(
        (
            normalize_text(application.get("concept")),
            normalize_text(application.get("attribute_key")),
        )
        for application in applications
    )
    return (
        normalize_text(category),
        normalize_text(product_type),
        tuple(targets),
    )


def _bundle_semantic_scope(scheduled_bundle: ScheduledRuleBundle) -> SemanticScopeKey:
    return _semantic_scope_key(
        scheduled_bundle.category,
        scheduled_bundle.primary.product_type,
        [
            {
                "concept": rule.concept,
                "attribute_key": rule.attribute_key,
            }
            for rule in scheduled_bundle.rules
        ],
    )


def _attempt_diversity_nonce(
    task: PairGenerationTask,
    *,
    task_seed_offset: int,
    task_retry_round: int,
    selection_attempt: int,
    stage: str,
    stage_attempt: int,
) -> str:
    payload = {
        "version": ATTEMPT_DIVERSITY_VERSION,
        "task_index": task.task_index,
        "task_seed": task.seed,
        "task_seed_offset": task_seed_offset,
        "task_retry_round": task_retry_round,
        "selection_attempt": selection_attempt,
        "stage": stage,
        "stage_attempt": stage_attempt,
    }
    return hashlib.blake2s(
        canonical_json_dumps(payload).encode("utf-8"), digest_size=16
    ).hexdigest()


def _nonce_appears_in_output(nonce: str, value: Any) -> bool:
    return normalize_text(nonce) in normalize_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True)
    )


def _global_rejection_prompt_entry(
    reason: str,
    *,
    base: dict[str, Any],
    mutated: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    applications = json.loads(str(metadata["applications_json"]))
    evidence = json.loads(str(metadata["anchor_evidence_json"]))
    return {
        "reason": reason,
        "product_type": str(metadata["product_type"]),
        "applications": [
            {
                "concept": str(application.get("concept", "")),
                "attribute_key": str(application.get("attribute_key", "")),
                "original_value": str(application.get("original_value", "")),
                "new_value": str(application.get("new_value", "")),
            }
            for application in applications
        ],
        "anchor_target_values": [
            {
                "concept": str(item.get("concept", "")),
                "attribute_key": str(item.get("attribute_key", "")),
                "attribute_value": str(item.get("attribute_value", "")),
            }
            for item in evidence
        ],
        "base_name": str(base["name"])[:240],
        "mutated_name": str(mutated["name"])[:240],
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rule_metadata(rule: MutationRule) -> dict[str, Any]:
    return {
        **rule.prompt_payload(),
        "allowed_categories": list(rule.allowed_categories),
        "source_path": rule.source_path,
    }


def build_pair_tasks(
    items: pd.DataFrame,
    *,
    count: int,
    seed: int,
    mutated_id_start: int | None,
    categories: list[str] | None,
    available_rule_categories: set[str],
    scheduled_donor_ids: list[int] | None = None,
) -> list[PairGenerationTask]:
    required = ["id", "name", "attributes", "category"]
    missing = set(required) - set(items.columns)
    if missing:
        raise ValueError(f"Base items are missing columns: {sorted(missing)}")
    if items["id"].duplicated().any():
        raise ValueError("Base items contain duplicate IDs")
    item_categories = set(items["category"].astype(str))
    requested_categories = (
        set(categories)
        if categories
        else item_categories & available_rule_categories
    )
    unknown_items = requested_categories - item_categories
    if unknown_items:
        raise ValueError(f"Requested categories are absent from base items: {sorted(unknown_items)}")
    without_rules = requested_categories - available_rule_categories
    if without_rules:
        raise ValueError(
            "Requested categories have no selectable rules: " + ", ".join(sorted(without_rules))
        )
    eligible = items[items["category"].astype(str).isin(requested_categories)].copy()
    if count < 1 or count > len(eligible):
        raise ValueError(f"count must be in [1, {len(eligible)}]")
    if scheduled_donor_ids is None:
        eligible["_rank"] = eligible["id"].map(
            lambda value: stable_hash64(seed, int(value))
        )
        eligible = eligible.sort_values(["_rank", "id"], kind="stable").head(count)
    else:
        scheduled_donor_ids = [int(value) for value in scheduled_donor_ids]
        if len(scheduled_donor_ids) != count:
            raise ValueError("Scheduled donor count does not match count")
        if len(set(scheduled_donor_ids)) != count:
            raise ValueError("Scheduled donor IDs are not unique")
        eligible["_scheduled_id"] = eligible["id"].map(int)
        available_ids = set(eligible["_scheduled_id"])
        absent_ids = set(scheduled_donor_ids) - available_ids
        if absent_ids:
            raise ValueError(
                "Scheduled donors are absent or category-incompatible: "
                + ", ".join(str(value) for value in sorted(absent_ids))
            )
        scheduled_rank = {
            donor_id: index for index, donor_id in enumerate(scheduled_donor_ids)
        }
        eligible = eligible[eligible["_scheduled_id"].isin(scheduled_rank)].copy()
        eligible["_rank"] = eligible["_scheduled_id"].map(scheduled_rank)
        eligible = eligible.sort_values("_rank", kind="stable")
    if mutated_id_start is None:
        mutated_id_start = min(0, int(items["id"].min())) - 1
    mutated_ids = [mutated_id_start - index for index in range(count)]
    if set(mutated_ids) & set(items["id"].astype(int)):
        raise ValueError("Mutated ID range overlaps base item IDs")

    tasks: list[PairGenerationTask] = []
    for task_index, (mutated_id, row) in enumerate(
        zip(mutated_ids, eligible[required].to_dict("records"), strict=True)
    ):
        if not parse_attributes(row["attributes"]):
            raise ValueError(f"Base item {row['id']} has no usable attributes")
        task_seed = stable_hash64(seed, int(row["id"])) % (2**31 - 1)
        tasks.append(
            PairGenerationTask(
                task_index=task_index,
                mutated_id=int(mutated_id),
                seed=int(task_seed),
                anchor=row,
            )
        )
    return tasks


def select_rule_bundle(
    category_rules: list[MutationRule],
    *,
    task_seed: int,
    selection_attempt: int,
    two_rule_fraction: float,
) -> list[MutationRule]:
    if not category_rules:
        raise ValueError("A pair task has no category-compatible rules")
    rng = np.random.default_rng(
        stable_hash64(task_seed, f"rule-selection:{selection_attempt}")
    )
    order = rng.permutation(len(category_rules)).tolist()
    first = category_rules[int(order[0])]
    compatible_pairs = [
        (category_rules[int(left)], category_rules[int(right)])
        for left_index, left in enumerate(order)
        for right in order[left_index + 1 :]
        if rules_are_product_compatible(
            category_rules[int(left)], category_rules[int(right)]
        )
    ]
    use_two = bool(compatible_pairs) and float(rng.random()) < two_rule_fraction
    return list(compatible_pairs[0]) if use_two else [first]


def generate_pair_task(
    task: PairGenerationTask,
    *,
    client: QwenPairClient,
    scheduled_bundle: ScheduledRuleBundle,
    diversity_context: AttemptDiversityContext,
    rule_schedule_sha256: str,
    pair_attempts: int,
    anchor_attempts: int,
    mutation_attempts: int,
    task_retry_round: int,
    task_retries_config: int,
    task_seed_offset: int,
    run_signature: str,
    prompt_sha256: str,
    label_one_fraction: float | None,
    planned_target_counts: dict[str, int],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    category = str(task.anchor["category"])
    if (
        scheduled_bundle.task_index != task.task_index
        or scheduled_bundle.donor_id != int(task.anchor["id"])
        or scheduled_bundle.category != category
    ):
        raise RuntimeError("Pair task does not align with its balanced rule schedule")
    scheduled_rules = list(scheduled_bundle.rules)
    rejection_history: list[dict[str, Any]] = []
    total_latency = 0.0
    total_request_attempts = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_reasoning_tokens = 0
    total_request_cost = 0.0
    last_error: Exception | None = None
    forbidden_semantic_signatures = set(
        diversity_context.forbidden_semantic_signatures
    )
    forbidden_card_keys = set(diversity_context.forbidden_card_keys)
    global_rejections = [dict(item) for item in diversity_context.global_rejections]
    effective_task_seed = task.seed + task_seed_offset
    for selection_attempt in range(1, pair_attempts + 1):
        global_selection_attempt = task_retry_round * pair_attempts + selection_attempt
        # Keep the rule/profile assignment fixed across attempts and task retries.
        # The model seed, prompt nonce and rejection feedback provide diversity.
        rules = scheduled_rules
        if any(category not in rule.allowed_categories for rule in rules):
            raise RuntimeError("Internal error: selected rule is incompatible with item category")
        if len({rule.label for rule in rules}) != 1:
            raise RuntimeError("Cannot combine mutation rules with different labels")
        anchor_validation: RuleAnchorValidation | None = None
        anchor_response: dict[str, Any] | None = None
        anchor_diversity_nonce: str | None = None
        successful_anchor_attempt: int | None = None
        anchor_feedback: list[str] = []
        for anchor_attempt in range(1, anchor_attempts + 1):
            candidate_nonce = _attempt_diversity_nonce(
                task,
                task_seed_offset=task_seed_offset,
                task_retry_round=task_retry_round,
                selection_attempt=selection_attempt,
                stage="anchor",
                stage_attempt=anchor_attempt,
            )
            prompt = build_rule_anchor_prompt(
                task.anchor,
                rules,
                feedback=anchor_feedback,
                diversity_nonce=candidate_nonce,
                global_rejections=global_rejections,
            )
            try:
                response = client.generate_anchor(
                    prompt,
                    category=category,
                    rules=rules,
                    seed=(
                        effective_task_seed
                        + global_selection_attempt * 100
                        + anchor_attempt
                    ),
                )
            except Exception as error:
                last_error = error
                anchor_feedback = [f"anchor_request_error:{type(error).__name__}"]
                continue
            total_latency += float(response["latency_seconds"])
            total_request_attempts += int(response["request_attempts"])
            total_prompt_tokens += int(response["prompt_tokens"])
            total_completion_tokens += int(response["completion_tokens"])
            total_reasoning_tokens += int(response.get("reasoning_tokens") or 0)
            total_request_cost += float(response.get("request_cost") or 0.0)
            candidate = validate_rule_anchor(
                response["item"],
                response["product_type"],
                response["evidence"],
                category=category,
                rules=rules,
            )
            if candidate.valid and _nonce_appears_in_output(
                candidate_nonce,
                {
                    "item": response["item"],
                    "product_type": response["product_type"],
                    "evidence": response["evidence"],
                },
            ):
                anchor_feedback = ["anchor_diversity_nonce_leaked"]
                continue
            if candidate.valid:
                anchor_validation = candidate
                anchor_response = response
                anchor_diversity_nonce = candidate_nonce
                successful_anchor_attempt = anchor_attempt
                break
            anchor_feedback = candidate.reasons
        if (
            anchor_validation is None
            or anchor_response is None
            or anchor_diversity_nonce is None
            or successful_anchor_attempt is None
        ):
            rejection_history.append(
                {
                    "selection_attempt": selection_attempt,
                    "stage": "anchor",
                    "rule_ids": [rule.generation_rule_id for rule in rules],
                    "reasons": anchor_feedback,
                }
            )
            continue

        base_item = {
            "id": int(task.anchor["id"]),
            "name": anchor_validation.item["name"],
            "attributes": json_dumps(anchor_validation.item["attributes"]),
            "category": anchor_validation.item["category"],
        }
        mutation_feedback: list[str] = []
        mutation_response: dict[str, Any] | None = None
        validation: MutationValidation | None = None
        mutation_diversity_nonce: str | None = None
        successful_mutation_attempt: int | None = None
        for mutation_attempt in range(1, mutation_attempts + 1):
            candidate_nonce = _attempt_diversity_nonce(
                task,
                task_seed_offset=task_seed_offset,
                task_retry_round=task_retry_round,
                selection_attempt=selection_attempt,
                stage="mutation",
                stage_attempt=mutation_attempt,
            )
            prompt = build_mutation_prompt(
                base_item,
                rules,
                anchor_validation.evidence,
                feedback=mutation_feedback,
                diversity_nonce=candidate_nonce,
                global_rejections=global_rejections,
            )
            try:
                response = client.mutate(
                    prompt,
                    category=category,
                    attribute_keys=list(anchor_validation.item["attributes"]),
                    rules=rules,
                    seed=(
                        effective_task_seed
                        + global_selection_attempt * 100
                        + 50
                        + mutation_attempt
                    ),
                )
            except Exception as error:
                last_error = error
                mutation_feedback = [f"mutation_request_error:{type(error).__name__}"]
                continue
            total_latency += float(response["latency_seconds"])
            total_request_attempts += int(response["request_attempts"])
            total_prompt_tokens += int(response["prompt_tokens"])
            total_completion_tokens += int(response["completion_tokens"])
            total_reasoning_tokens += int(response.get("reasoning_tokens") or 0)
            total_request_cost += float(response.get("request_cost") or 0.0)
            candidate = validate_mutation(
                response["item"],
                response["applications"],
                anchor=base_item,
                rules=rules,
                evidence=anchor_validation.evidence,
            )
            if candidate.valid and _nonce_appears_in_output(
                candidate_nonce,
                {
                    "item": response["item"],
                    "applications": response["applications"],
                },
            ):
                mutation_feedback = ["mutation_diversity_nonce_leaked"]
                continue
            if candidate.valid:
                validation = candidate
                mutation_response = response
                mutation_diversity_nonce = candidate_nonce
                successful_mutation_attempt = mutation_attempt
                break
            mutation_feedback = candidate.reasons
        if (
            validation is None
            or mutation_response is None
            or mutation_diversity_nonce is None
            or successful_mutation_attempt is None
        ):
            rejection_history.append(
                {
                    "selection_attempt": selection_attempt,
                    "stage": "mutation",
                    "rule_ids": [rule.generation_rule_id for rule in rules],
                    "reasons": mutation_feedback,
                }
            )
            continue

        mutated_item = {
            "id": int(task.mutated_id),
            "name": validation.item["name"],
            "attributes": json_dumps(validation.item["attributes"]),
            "category": validation.item["category"],
        }
        target = int(rules[0].label)
        if target != scheduled_bundle.target:
            raise RuntimeError("Generated target does not match the scheduled target")
        pair = {
            "id1": int(base_item["id"]),
            "id2": int(task.mutated_id),
            "target": target,
        }
        rules_json = json_dumps([_rule_metadata(rule) for rule in rules])
        semantic_signature = _semantic_pair_signature(
            category,
            anchor_response["product_type"],
            mutation_response["applications"],
        )
        result_card_keys = {
            _canonical_card_key(base_item),
            _canonical_card_key(mutated_item),
        }
        global_rejection_reasons: list[str] = []
        if len(result_card_keys) != 2 or result_card_keys & forbidden_card_keys:
            global_rejection_reasons.append("generated_duplicate_full_card")
        if semantic_signature in forbidden_semantic_signatures:
            global_rejection_reasons.append("generated_semantic_signature_limit")
        if global_rejection_reasons:
            candidate_metadata = {
                "applications_json": json_dumps(mutation_response["applications"]),
                "anchor_evidence_json": json_dumps(anchor_validation.evidence),
                "product_type": str(anchor_response["product_type"]),
            }
            global_rejections.append(
                _global_rejection_prompt_entry(
                    "+".join(global_rejection_reasons),
                    base=base_item,
                    mutated=mutated_item,
                    metadata=candidate_metadata,
                )
            )
            global_rejections = global_rejections[
                -MAX_GLOBAL_REJECTION_PROMPT_ITEMS:
            ]
            forbidden_card_keys.update(result_card_keys)
            if "generated_semantic_signature_limit" in global_rejection_reasons:
                forbidden_semantic_signatures.add(semantic_signature)
            rejection_history.append(
                {
                    "selection_attempt": selection_attempt,
                    "stage": "global_diversity",
                    "rule_ids": [rule.generation_rule_id for rule in rules],
                    "reasons": global_rejection_reasons,
                }
            )
            last_error = RuntimeError("+".join(global_rejection_reasons))
            continue
        metadata = {
            "task_index": int(task.task_index),
            "id1": pair["id1"],
            "id2": pair["id2"],
            "target": target,
            "category": category,
            "rule_count": len(rules),
            "rule_ids": json_dumps([rule.generation_rule_id for rule in rules]),
            "source_rule_ids": json_dumps([rule.source_rule_id for rule in rules]),
            "rule_tiers": json_dumps([rule.generation_tier for rule in rules]),
            "concepts": json_dumps([rule.concept for rule in rules]),
            "rules_json": rules_json,
            "anchor_evidence_json": json_dumps(anchor_validation.evidence),
            "applications_json": json_dumps(mutation_response["applications"]),
            "semantic_signature": semantic_signature,
            "semantic_signature_version": SEMANTIC_SIGNATURE_VERSION,
            "attempt_diversity_version": ATTEMPT_DIVERSITY_VERSION,
            "global_rejection_prompt_item_limit": (
                MAX_GLOBAL_REJECTION_PROMPT_ITEMS
            ),
            "source_style_id": int(task.anchor["id"]),
            "balanced_rule_schedule": True,
            "rule_schedule_version": SCHEDULE_VERSION,
            "rule_schedule_sha256": rule_schedule_sha256,
            "scheduled_primary_rule_id": (
                scheduled_bundle.primary.rule.generation_rule_id
            ),
            "scheduled_primary_profile_id": scheduled_bundle.primary.profile_id,
            "scheduled_primary_product_type": scheduled_bundle.primary.product_type,
            "scheduled_primary_task_cap": scheduled_bundle.primary.primary_task_cap,
            "scheduled_secondary_rule_id": (
                scheduled_bundle.secondary.rule.generation_rule_id
                if scheduled_bundle.secondary is not None
                else None
            ),
            "scheduled_secondary_profile_id": (
                scheduled_bundle.secondary.profile_id
                if scheduled_bundle.secondary is not None
                else None
            ),
            "scheduled_secondary_product_type": (
                scheduled_bundle.secondary.product_type
                if scheduled_bundle.secondary is not None
                else None
            ),
            "scheduled_rule_ids": json_dumps(
                [profile.rule.generation_rule_id for profile in scheduled_bundle.profiles]
            ),
            "scheduled_rule_profile_ids": json_dumps(
                [profile.profile_id for profile in scheduled_bundle.profiles]
            ),
            "scheduled_target": scheduled_bundle.target,
            "label_quota_enabled": label_one_fraction is not None,
            "label_quota_policy_version": LABEL_QUOTA_POLICY_VERSION,
            "requested_label_one_fraction": label_one_fraction,
            "planned_target_counts_json": json_dumps(planned_target_counts),
            "planned_label_zero_tasks": int(planned_target_counts["0"]),
            "planned_label_one_tasks": int(planned_target_counts["1"]),
            "planned_label_one_fraction": (
                planned_target_counts["1"] / sum(planned_target_counts.values())
            ),
            "profile_capacity_policy_version": (
                scheduled_bundle.primary.rule.profile_capacity_policy_version
            ),
            "profile_capacity_policy_sha256": (
                scheduled_bundle.primary.rule.profile_capacity_policy_sha256
            ),
            "product_type": str(anchor_response["product_type"]),
            "selection_attempt": selection_attempt,
            "anchor_attempt": int(successful_anchor_attempt),
            "mutation_attempt": int(successful_mutation_attempt),
            "task_retry_round": task_retry_round,
            "task_seed_offset": task_seed_offset,
            "pair_attempts_config": pair_attempts,
            "anchor_attempts_config": anchor_attempts,
            "mutation_attempts_config": mutation_attempts,
            "task_retries_config": task_retries_config,
            "request_attempts": total_request_attempts,
            "latency_seconds": total_latency,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "reasoning_tokens": total_reasoning_tokens,
            "request_cost": total_request_cost,
            "anchor_response_id": str(anchor_response["response_id"]),
            "mutation_response_id": str(mutation_response["response_id"]),
            "anchor_diversity_nonce_sha256": sha256_text(
                str(anchor_diversity_nonce)
            ),
            "mutation_diversity_nonce_sha256": sha256_text(
                str(mutation_diversity_nonce)
            ),
            "global_rejection_feedback_count": len(global_rejections),
            "forbidden_semantic_signature_count": len(
                forbidden_semantic_signatures
            ),
            "forbidden_card_key_count": len(forbidden_card_keys),
            "run_signature": run_signature,
            "prompt_sha256": prompt_sha256,
            "model": client.model,
            "rejection_history": json_dumps(rejection_history),
            **anchor_validation.metrics,
            **validation.metrics,
            "completed_at": utc_now(),
        }
        return base_item, mutated_item, pair, metadata
    raise RuntimeError(
        f"pair task {task.task_index} failed after {pair_attempts} rule selections; "
        f"last_rejections={rejection_history[-2:]}; last_error={last_error!r}"
    )


def _load_pair_checkpoint(
    output_dir: Path,
    run_signature: str,
) -> tuple[
    dict[int, dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[int, dict[str, Any]],
]:
    paths = {
        "base": output_dir / "base_items.parquet",
        "mutated": output_dir / "mutated_items.parquet",
        "pairs": output_dir / "pairs.parquet",
        "metadata": output_dir / "pair_generation_metadata.parquet",
    }
    exists = {name: path.exists() for name, path in paths.items()}
    if not any(exists.values()):
        return {}, {}, {}, {}
    if not all(exists.values()):
        raise ValueError(f"Incomplete pair checkpoint: {exists}")
    base = pd.read_parquet(paths["base"])
    mutated = pd.read_parquet(paths["mutated"])
    pairs = pd.read_parquet(paths["pairs"])
    metadata = pd.read_parquet(paths["metadata"])
    if (
        metadata["task_index"].duplicated().any()
        or base["id"].duplicated().any()
        or mutated["id"].duplicated().any()
    ):
        raise ValueError("Pair checkpoint contains duplicate task indices or item IDs")
    if set(metadata["run_signature"].astype(str)) != {run_signature}:
        raise ValueError(
            "Pair checkpoint belongs to another configuration; use another output directory"
        )
    base_by_id = {int(row["id"]): row for row in base.to_dict("records")}
    mutated_by_id = {int(row["id"]): row for row in mutated.to_dict("records")}
    pair_by_id2 = {int(row["id2"]): row for row in pairs.to_dict("records")}
    base_by_task: dict[int, dict[str, Any]] = {}
    mutated_by_task: dict[int, dict[str, Any]] = {}
    pairs_by_task: dict[int, dict[str, Any]] = {}
    metadata_by_task: dict[int, dict[str, Any]] = {}
    for row in metadata.to_dict("records"):
        task_index = int(row["task_index"])
        id1 = int(row["id1"])
        id2 = int(row["id2"])
        if id1 not in base_by_id or id2 not in mutated_by_id or id2 not in pair_by_id2:
            raise ValueError("Pair checkpoint files do not align on id2")
        base_by_task[task_index] = base_by_id[id1]
        mutated_by_task[task_index] = mutated_by_id[id2]
        pairs_by_task[task_index] = pair_by_id2[id2]
        metadata_by_task[task_index] = row
    return base_by_task, mutated_by_task, pairs_by_task, metadata_by_task


def _write_pair_checkpoint(
    output_dir: Path,
    base_by_task: dict[int, dict[str, Any]],
    mutated_by_task: dict[int, dict[str, Any]],
    pairs_by_task: dict[int, dict[str, Any]],
    metadata_by_task: dict[int, dict[str, Any]],
) -> None:
    if not mutated_by_task:
        return
    indices = sorted(mutated_by_task)
    base = pd.DataFrame([base_by_task[index] for index in indices])[
        ["id", "name", "attributes", "category"]
    ]
    mutated = pd.DataFrame([mutated_by_task[index] for index in indices])[
        ["id", "name", "attributes", "category"]
    ]
    pairs = pd.DataFrame([pairs_by_task[index] for index in indices])[
        ["id1", "id2", "target"]
    ]
    metadata = pd.DataFrame([metadata_by_task[index] for index in indices])
    combined = pd.concat([base, mutated], ignore_index=True)[
        ["id", "name", "attributes", "category"]
    ]
    _atomic_parquet(base, output_dir / "base_items.parquet")
    _atomic_parquet(mutated, output_dir / "mutated_items.parquet")
    _atomic_parquet(pairs, output_dir / "pairs.parquet")
    _atomic_parquet(metadata, output_dir / "pair_generation_metadata.parquet")
    _atomic_parquet(combined, output_dir / "items.parquet")


def run_pair_generation(
    *,
    items_path: Path,
    rule_paths: list[Path],
    output_dir: Path,
    client: QwenPairClient,
    system_prompt: str,
    count: int,
    seed: int,
    mutated_id_start: int | None,
    categories: list[str] | None,
    tiers: set[str] | None,
    workers: int,
    pair_attempts: int,
    anchor_attempts: int = 3,
    mutation_attempts: int = 3,
    checkpoint_every: int,
    two_rule_fraction: float,
    task_retries: int = 3,
    task_seed_offset: int = 0,
    semantic_signature_limit: int = 2,
    label_one_fraction: float | None = None,
) -> dict[str, Any]:
    if (
        workers < 1
        or pair_attempts < 1
        or anchor_attempts < 1
        or mutation_attempts < 1
        or checkpoint_every < 1
    ):
        raise ValueError(
            "workers, pair_attempts, anchor_attempts, mutation_attempts and "
            "checkpoint_every must be positive"
        )
    if not 0.0 <= two_rule_fraction <= 1.0:
        raise ValueError("two_rule_fraction must be in [0, 1]")
    if label_one_fraction is not None and not 0.0 <= label_one_fraction <= 1.0:
        raise ValueError("label_one_fraction must be in [0, 1]")
    if task_retries < 0:
        raise ValueError("task_retries must be non-negative")
    if semantic_signature_limit < 1:
        raise ValueError("semantic_signature_limit must be positive")
    items_path = items_path.resolve()
    resolved_rule_paths = [path.resolve() for path in rule_paths]
    items = pd.read_parquet(items_path)
    rules = load_mutation_rules(resolved_rule_paths, tiers=tiers)
    rule_schedule = build_balanced_rule_schedule(
        items,
        rules,
        count=count,
        seed=seed,
        two_rule_fraction=two_rule_fraction,
        categories=categories,
        semantic_signature_limit=semantic_signature_limit,
        label_one_fraction=label_one_fraction,
    )
    planned_rule_schedule = rule_schedule.summary()
    tasks = build_pair_tasks(
        items,
        count=count,
        seed=seed,
        mutated_id_start=mutated_id_start,
        categories=categories,
        available_rule_categories={
            category for rule in rules for category in rule.allowed_categories
        },
        scheduled_donor_ids=[entry.donor_id for entry in rule_schedule.entries],
    )
    for task, scheduled_bundle in zip(
        tasks, rule_schedule.entries, strict=True
    ):
        if (
            task.task_index != scheduled_bundle.task_index
            or int(task.anchor["id"]) != scheduled_bundle.donor_id
            or str(task.anchor["category"]) != scheduled_bundle.category
        ):
            raise RuntimeError("Balanced rule schedule does not align with pair tasks")
    effective_mutated_id_start = tasks[0].mutated_id
    prompt_sha256 = sha256_text(system_prompt)
    signature_payload = {
        "version": "rule_first_pair_generation_v3",
        "attempt_diversity_version": ATTEMPT_DIVERSITY_VERSION,
        "global_rejection_prompt_item_limit": MAX_GLOBAL_REJECTION_PROMPT_ITEMS,
        "global_duplicate_card_retry": True,
        "semantic_signature_retry": True,
        "semantic_signature_version": SEMANTIC_SIGNATURE_VERSION,
        "semantic_signature_limit": semantic_signature_limit,
        "balanced_rule_schedule": True,
        "rule_schedule_version": SCHEDULE_VERSION,
        "rule_schedule_sha256": rule_schedule.schedule_sha256,
        "rule_schedule_config": {
            "category_allocation": "global_total_exposure_balance_with_donor_caps",
            "primary_profile_allocation": "capacity_aware_least_total_exposed",
            "secondary_profile_allocation": "compatible_joint_endpoint_balance",
            "two_rule_allocation": "clip_before_pairable_total_exposure_confound",
            "fixed_bundle_across_retries": True,
            "requested_two_rule_fraction": two_rule_fraction,
            "label_quota_policy_version": LABEL_QUOTA_POLICY_VERSION,
            "requested_label_one_fraction": label_one_fraction,
            "planned_target_counts": planned_rule_schedule[
                "planned_target_counts"
            ],
        },
        "planned_rule_schedule": planned_rule_schedule,
        "base_items_path": str(items_path),
        "base_items_sha256": _sha256_file(items_path),
        "rule_catalogs": [
            {"path": str(path), "sha256": _sha256_file(path)}
            for path in resolved_rule_paths
        ],
        "rule_catalog_summary": catalog_summary(rules),
        "rule_tiers": sorted(tiers) if tiers is not None else None,
        "model": client.model,
        "api_base_url": str(getattr(client, "base_url", "")),
        "reasoning_effort": getattr(client, "reasoning_effort", None),
        "temperature": float(client.temperature),
        "max_tokens": int(client.max_tokens),
        "structured_output": bool(getattr(client, "structured_output", True)),
        "prompt_sha256": prompt_sha256,
        "count": count,
        "seed": seed,
        "mutated_id_start": effective_mutated_id_start,
        "categories": categories,
        "two_rule_fraction": two_rule_fraction,
        "label_one_fraction": label_one_fraction,
        "pair_attempts": pair_attempts,
        "anchor_attempts": anchor_attempts,
        "mutation_attempts": mutation_attempts,
    }
    run_signature = sha256_text(
        json.dumps(signature_payload, ensure_ascii=False, sort_keys=True)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    base_by_task, mutated_by_task, pairs_by_task, metadata_by_task = _load_pair_checkpoint(
        output_dir, run_signature
    )
    for task_index, row in metadata_by_task.items():
        checkpoint_task = tasks[task_index]
        expected = rule_schedule.bundle_for_task(task_index).provenance(
            rule_schedule.schedule_sha256
        )
        actual_scheduled_rule_ids = json.loads(str(row.get("scheduled_rule_ids", "")))
        actual_scheduled_profile_ids = json.loads(
            str(row.get("scheduled_rule_profile_ids", ""))
        )
        secondary_rule_id = row.get("scheduled_secondary_rule_id")
        secondary_profile_id = row.get("scheduled_secondary_profile_id")
        if pd.isna(secondary_rule_id):
            secondary_rule_id = None
        if pd.isna(secondary_profile_id):
            secondary_profile_id = None
        primary_task_cap = row.get("scheduled_primary_task_cap")
        if pd.isna(primary_task_cap):
            primary_task_cap = None
        else:
            primary_task_cap = int(primary_task_cap)
        expected_anchor_nonce_sha256 = sha256_text(
            _attempt_diversity_nonce(
                checkpoint_task,
                task_seed_offset=int(row.get("task_seed_offset", 0)),
                task_retry_round=int(row.get("task_retry_round", 0)),
                selection_attempt=int(row.get("selection_attempt", 0)),
                stage="anchor",
                stage_attempt=int(row.get("anchor_attempt", 0)),
            )
        )
        expected_mutation_nonce_sha256 = sha256_text(
            _attempt_diversity_nonce(
                checkpoint_task,
                task_seed_offset=int(row.get("task_seed_offset", 0)),
                task_retry_round=int(row.get("task_retry_round", 0)),
                selection_attempt=int(row.get("selection_attempt", 0)),
                stage="mutation",
                stage_attempt=int(row.get("mutation_attempt", 0)),
            )
        )
        schedule_mismatch = {
            "attempt_diversity_version": str(
                row.get("attempt_diversity_version") or ""
            )
            != ATTEMPT_DIVERSITY_VERSION,
            "global_rejection_prompt_item_limit": int(
                row.get("global_rejection_prompt_item_limit", -1)
            )
            != MAX_GLOBAL_REJECTION_PROMPT_ITEMS,
            "anchor_diversity_nonce_sha256": str(
                row.get("anchor_diversity_nonce_sha256") or ""
            )
            != expected_anchor_nonce_sha256,
            "mutation_diversity_nonce_sha256": str(
                row.get("mutation_diversity_nonce_sha256") or ""
            )
            != expected_mutation_nonce_sha256,
            "source_style_id": int(row.get("source_style_id", -1))
            != expected["source_style_id"],
            "category": str(row.get("category")) != expected["category"],
            "balanced_rule_schedule": bool(row.get("balanced_rule_schedule")) is not True,
            "rule_schedule_version": str(row.get("rule_schedule_version"))
            != SCHEDULE_VERSION,
            "rule_schedule_sha256": str(row.get("rule_schedule_sha256"))
            != rule_schedule.schedule_sha256,
            "scheduled_primary_rule_id": str(row.get("scheduled_primary_rule_id"))
            != expected["scheduled_primary_rule_id"],
            "scheduled_primary_profile_id": str(row.get("scheduled_primary_profile_id"))
            != expected["scheduled_primary_profile_id"],
            "scheduled_primary_task_cap": primary_task_cap
            != expected["scheduled_primary_task_cap"],
            "profile_capacity_policy_version": str(
                row.get("profile_capacity_policy_version") or ""
            )
            != rule_schedule.profile_capacity_policy_version,
            "profile_capacity_policy_sha256": str(
                row.get("profile_capacity_policy_sha256") or ""
            )
            != rule_schedule.profile_capacity_policy_sha256,
            "scheduled_secondary_rule_id": secondary_rule_id
            != expected["scheduled_secondary_rule_id"],
            "scheduled_secondary_profile_id": secondary_profile_id
            != expected["scheduled_secondary_profile_id"],
            "scheduled_rule_ids": actual_scheduled_rule_ids
            != expected["scheduled_rule_ids"],
            "scheduled_rule_profile_ids": actual_scheduled_profile_ids
            != expected["scheduled_rule_profile_ids"],
            "scheduled_target": int(row.get("scheduled_target", -1))
            != expected["scheduled_target"],
            "target": int(row.get("target", -1)) != expected["scheduled_target"],
        }
        if label_one_fraction is not None:
            requested_fraction = row.get("requested_label_one_fraction")
            planned_counts_raw = str(row.get("planned_target_counts_json", ""))
            try:
                checkpoint_planned_counts = json.loads(planned_counts_raw)
            except json.JSONDecodeError:
                checkpoint_planned_counts = None
            schedule_mismatch.update(
                {
                    "label_quota_enabled": bool(row.get("label_quota_enabled"))
                    is not True,
                    "label_quota_policy_version": str(
                        row.get("label_quota_policy_version") or ""
                    )
                    != LABEL_QUOTA_POLICY_VERSION,
                    "requested_label_one_fraction": pd.isna(requested_fraction)
                    or not np.isclose(
                        float(requested_fraction), label_one_fraction
                    ),
                    "planned_target_counts_json": checkpoint_planned_counts
                    != planned_rule_schedule["planned_target_counts"],
                }
            )
        failed_fields = sorted(
            field for field, mismatch in schedule_mismatch.items() if mismatch
        )
        if failed_fields:
            raise ValueError(
                "Existing checkpoint does not match the balanced rule schedule: "
                f"task_index={task_index}, fields={failed_fields}"
            )
    accepted_card_keys = {
        _canonical_card_key(item)
        for collection in (base_by_task, mutated_by_task)
        for item in collection.values()
    }
    if len(accepted_card_keys) != len(base_by_task) + len(mutated_by_task):
        raise ValueError("Existing checkpoint contains duplicate full cards")
    semantic_signature_counts: Counter[str] = Counter()
    for row in metadata_by_task.values():
        if row.get("semantic_signature_version") != SEMANTIC_SIGNATURE_VERSION:
            raise ValueError(
                "Existing checkpoint metadata has an unexpected "
                "semantic_signature_version"
            )
        semantic_signature = row.get("semantic_signature")
        if not isinstance(semantic_signature, str) or not semantic_signature:
            raise ValueError(
                "Existing checkpoint metadata has no semantic_signature"
            )
        semantic_signature_counts[semantic_signature] += 1
    over_limit = {
        signature: count
        for signature, count in semantic_signature_counts.items()
        if count > semantic_signature_limit
    }
    if over_limit:
        raise ValueError(
            "Existing checkpoint exceeds semantic_signature_limit: "
            f"{over_limit}"
        )
    saturated_signatures_by_scope: dict[SemanticScopeKey, set[str]] = defaultdict(set)
    forbidden_card_keys_by_scope: dict[SemanticScopeKey, set[str]] = defaultdict(set)
    semantic_feedback_by_scope: dict[
        SemanticScopeKey, dict[str, dict[str, Any]]
    ] = defaultdict(dict)
    duplicate_feedback_by_scope: dict[
        SemanticScopeKey, dict[str, dict[str, Any]]
    ] = defaultdict(dict)
    for task_index, row in metadata_by_task.items():
        applications = json.loads(str(row["applications_json"]))
        scope = _semantic_scope_key(
            row["category"], row["scheduled_primary_product_type"], applications
        )
        base = base_by_task[task_index]
        mutated = mutated_by_task[task_index]
        forbidden_card_keys_by_scope[scope].update(
            {_canonical_card_key(base), _canonical_card_key(mutated)}
        )
        semantic_signature = str(row["semantic_signature"])
        if semantic_signature_counts[semantic_signature] >= semantic_signature_limit:
            saturated_signatures_by_scope[scope].add(semantic_signature)
            semantic_feedback_by_scope[scope][semantic_signature] = (
                _global_rejection_prompt_entry(
                    "generated_semantic_signature_limit",
                    base=base,
                    mutated=mutated,
                    metadata=row,
                )
            )

    def diversity_context_for_bundle(
        scheduled_bundle: ScheduledRuleBundle,
    ) -> AttemptDiversityContext:
        scope = _bundle_semantic_scope(scheduled_bundle)
        semantic_feedback = semantic_feedback_by_scope.get(scope, {})
        duplicate_feedback = duplicate_feedback_by_scope.get(scope, {})
        prompt_rejections = [
            semantic_feedback[key] for key in sorted(semantic_feedback)
        ]
        remaining = MAX_GLOBAL_REJECTION_PROMPT_ITEMS - len(prompt_rejections)
        if remaining > 0:
            prompt_rejections.extend(
                duplicate_feedback[key]
                for key in sorted(duplicate_feedback)[:remaining]
            )
        return AttemptDiversityContext(
            forbidden_semantic_signatures=frozenset(
                saturated_signatures_by_scope.get(scope, set())
            ),
            forbidden_card_keys=frozenset(
                forbidden_card_keys_by_scope.get(scope, set())
            ),
            global_rejections=tuple(
                prompt_rejections[:MAX_GLOBAL_REJECTION_PROMPT_ITEMS]
            ),
        )

    def register_global_rejection(
        reason: str,
        *,
        base: dict[str, Any],
        mutated: dict[str, Any],
        metadata: dict[str, Any],
        result_card_keys: set[str],
        semantic_signature: str,
    ) -> None:
        applications = json.loads(str(metadata["applications_json"]))
        scope = _semantic_scope_key(
            metadata["category"],
            metadata["scheduled_primary_product_type"],
            applications,
        )
        entry = _global_rejection_prompt_entry(
            reason,
            base=base,
            mutated=mutated,
            metadata=metadata,
        )
        forbidden_card_keys_by_scope[scope].update(result_card_keys)
        if reason == "generated_semantic_signature_limit":
            saturated_signatures_by_scope[scope].add(semantic_signature)
            semantic_feedback_by_scope[scope][semantic_signature] = entry
        else:
            key = sha256_text(canonical_json_dumps(entry))
            duplicate_feedback_by_scope[scope][key] = entry
            if len(duplicate_feedback_by_scope[scope]) > MAX_GLOBAL_REJECTION_PROMPT_ITEMS:
                del duplicate_feedback_by_scope[scope][
                    sorted(duplicate_feedback_by_scope[scope])[0]
                ]

    prior_semantic_signature_retry_events = 0
    prior_summary_path = output_dir / "summary.json"
    if metadata_by_task and prior_summary_path.is_file():
        prior_summary = json.loads(prior_summary_path.read_text(encoding="utf-8"))
        if prior_summary.get("run_signature") != run_signature:
            raise ValueError("Existing summary belongs to another configuration")
        prior_semantic_signature_retry_events = int(
            prior_summary.get("semantic_signature_retry_events", 0)
        )
    pending = [task for task in tasks if task.task_index not in mutated_by_task]
    resumed_count = len(mutated_by_task)
    print(
        f"generate-pairs requested={count} resumed={len(mutated_by_task)} "
        f"pending={len(pending)} workers={workers} model={client.model} "
        f"rules={len(rules)} profiles={len(rule_schedule.eligible_profiles)} "
        f"scheduled_two={planned_rule_schedule['scheduled_two_rule_tasks']} "
        f"planned_targets={planned_rule_schedule['planned_target_counts']} "
        f"task_retries={task_retries}",
        flush=True,
    )
    errors: list[dict[str, Any]] = []
    task_retry_events = 0
    duplicate_retry_events = 0
    semantic_signature_retry_events_this_run = 0
    started = time.perf_counter()
    completed_since_checkpoint = 0
    iterator = iter(pending)
    retry_queue: deque[tuple[PairGenerationTask, int]] = deque()
    inflight: dict[
        Future[
            tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]
        ],
        tuple[PairGenerationTask, int],
    ] = {}

    def submit_task(
        executor: ThreadPoolExecutor,
        task: PairGenerationTask,
        task_retry_round: int,
    ) -> None:
        scheduled_bundle = rule_schedule.bundle_for_task(task.task_index)
        future = executor.submit(
            generate_pair_task,
            task,
            client=client,
            scheduled_bundle=scheduled_bundle,
            diversity_context=diversity_context_for_bundle(scheduled_bundle),
            rule_schedule_sha256=rule_schedule.schedule_sha256,
            pair_attempts=pair_attempts,
            anchor_attempts=anchor_attempts,
            mutation_attempts=mutation_attempts,
            task_retry_round=task_retry_round,
            task_retries_config=task_retries,
            task_seed_offset=task_seed_offset,
            run_signature=run_signature,
            prompt_sha256=prompt_sha256,
            label_one_fraction=label_one_fraction,
            planned_target_counts=planned_rule_schedule["planned_target_counts"],
        )
        inflight[future] = (task, task_retry_round)

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        try:
            task = next(iterator)
        except StopIteration:
            if not retry_queue:
                return False
            task, task_retry_round = retry_queue.popleft()
        else:
            task_retry_round = 0
        submit_task(executor, task, task_retry_round)
        return True

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in range(min(len(pending), workers * 2)):
            submit_next(executor)
        while inflight:
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for future in done:
                task, task_retry_round = inflight.pop(future)
                try:
                    base, mutated, pair, metadata = future.result()
                    result_card_keys = {
                        _canonical_card_key(base),
                        _canonical_card_key(mutated),
                    }
                    semantic_signature = metadata.get("semantic_signature")
                    if (
                        not isinstance(semantic_signature, str)
                        or not semantic_signature
                    ):
                        raise RuntimeError("generated_missing_semantic_signature")
                    if (
                        len(result_card_keys) != 2
                        or result_card_keys & accepted_card_keys
                    ):
                        register_global_rejection(
                            "generated_duplicate_full_card",
                            base=base,
                            mutated=mutated,
                            metadata=metadata,
                            result_card_keys=result_card_keys,
                            semantic_signature=semantic_signature,
                        )
                        raise RuntimeError("generated_duplicate_full_card")
                    if (
                        semantic_signature_counts[semantic_signature]
                        >= semantic_signature_limit
                    ):
                        register_global_rejection(
                            "generated_semantic_signature_limit",
                            base=base,
                            mutated=mutated,
                            metadata=metadata,
                            result_card_keys=result_card_keys,
                            semantic_signature=semantic_signature,
                        )
                        raise RuntimeError("generated_semantic_signature_limit")
                    base_by_task[task.task_index] = base
                    mutated_by_task[task.task_index] = mutated
                    pairs_by_task[task.task_index] = pair
                    metadata_by_task[task.task_index] = metadata
                    accepted_card_keys.update(result_card_keys)
                    semantic_signature_counts[semantic_signature] += 1
                    applications = json.loads(str(metadata["applications_json"]))
                    accepted_scope = _semantic_scope_key(
                        metadata["category"],
                        metadata["scheduled_primary_product_type"],
                        applications,
                    )
                    forbidden_card_keys_by_scope[accepted_scope].update(
                        result_card_keys
                    )
                    if (
                        semantic_signature_counts[semantic_signature]
                        == semantic_signature_limit
                    ):
                        register_global_rejection(
                            "generated_semantic_signature_limit",
                            base=base,
                            mutated=mutated,
                            metadata=metadata,
                            result_card_keys=result_card_keys,
                            semantic_signature=semantic_signature,
                        )
                    completed_since_checkpoint += 1
                except Exception as error:
                    if "generated_duplicate_full_card" in str(error):
                        duplicate_retry_events += 1
                    if "generated_semantic_signature_limit" in str(error):
                        semantic_signature_retry_events_this_run += 1
                    if task_retry_round < task_retries:
                        task_retry_events += 1
                        retry_queue.append((task, task_retry_round + 1))
                        submit_next(executor)
                    else:
                        errors.append(
                            {
                                "task_index": task.task_index,
                                "source_id": int(task.anchor["id"]),
                                "category": str(task.anchor["category"]),
                                "task_retry_rounds": task_retry_round,
                                "error": repr(error),
                            }
                        )
                        if len(errors) <= 5 or len(errors) % 100 == 0:
                            print(
                                "generate-pairs terminal-error "
                                f"task={task.task_index} errors={len(errors)} "
                                f"kind={type(error).__name__} detail={str(error)[:500]}",
                                flush=True,
                            )
                        submit_next(executor)
                else:
                    submit_next(executor)
                if completed_since_checkpoint >= checkpoint_every:
                    _write_pair_checkpoint(
                        output_dir,
                        base_by_task,
                        mutated_by_task,
                        pairs_by_task,
                        metadata_by_task,
                    )
                    completed_since_checkpoint = 0
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    generated_this_run = len(mutated_by_task) - resumed_count
                    print(
                        f"generate-pairs saved={len(mutated_by_task)}/{count} "
                        f"errors={len(errors)} retried={task_retry_events} "
                        f"retry_queue={len(retry_queue)} "
                        f"rate={generated_this_run / elapsed:.2f}/s",
                        flush=True,
                    )

    _write_pair_checkpoint(
        output_dir,
        base_by_task,
        mutated_by_task,
        pairs_by_task,
        metadata_by_task,
    )
    elapsed = time.perf_counter() - started
    rule_count_distribution: dict[str, int] = {}
    for row in metadata_by_task.values():
        key = str(int(row["rule_count"]))
        rule_count_distribution[key] = rule_count_distribution.get(key, 0) + 1
    two_rule_pairs = int(rule_count_distribution.get("2", 0))
    def metadata_int_distribution(field: str) -> dict[str, int]:
        counts = Counter(
            int(row[field]) for row in metadata_by_task.values() if field in row
        )
        return {str(key): int(value) for key, value in sorted(counts.items())}

    realized_task_seed_offsets = sorted(
        {int(row["task_seed_offset"]) for row in metadata_by_task.values()}
    )
    realized_rule_schedule = rule_schedule.realized_summary(
        metadata_by_task.keys()
    )
    validation = (
        validate_pair_dataset(
            output_dir / "items.parquet",
            output_dir / "pairs.parquet",
            metadata_path=output_dir / "pair_generation_metadata.parquet",
        )
        if mutated_by_task
        else {"valid": False, "pairs": 0}
    )
    summary = {
        **signature_payload,
        **planned_rule_schedule,
        **realized_rule_schedule,
        "realized_rule_schedule": realized_rule_schedule,
        "run_signature": run_signature,
        "generated_pairs": len(mutated_by_task),
        "accepted_prompt_tokens": int(
            sum(int(row.get("prompt_tokens") or 0) for row in metadata_by_task.values())
        ),
        "accepted_completion_tokens": int(
            sum(int(row.get("completion_tokens") or 0) for row in metadata_by_task.values())
        ),
        "accepted_reasoning_tokens": int(
            sum(int(row.get("reasoning_tokens") or 0) for row in metadata_by_task.values())
        ),
        "accepted_request_cost": float(
            sum(float(row.get("request_cost") or 0.0) for row in metadata_by_task.values())
        ),
        "pending": count - len(mutated_by_task),
        "errors": len(errors),
        "task_retries_this_run": task_retries,
        "task_seed_offset_this_run": task_seed_offset,
        "realized_task_seed_offsets": realized_task_seed_offsets,
        "realized_task_seed_offset_distribution": metadata_int_distribution(
            "task_seed_offset"
        ),
        "realized_task_retry_round_distribution": metadata_int_distribution(
            "task_retry_round"
        ),
        "realized_selection_attempt_distribution": metadata_int_distribution(
            "selection_attempt"
        ),
        "realized_anchor_attempt_distribution": metadata_int_distribution(
            "anchor_attempt"
        ),
        "realized_mutation_attempt_distribution": metadata_int_distribution(
            "mutation_attempt"
        ),
        "realized_pair_attempts_config_distribution": metadata_int_distribution(
            "pair_attempts_config"
        ),
        "realized_anchor_attempts_config_distribution": metadata_int_distribution(
            "anchor_attempts_config"
        ),
        "realized_mutation_attempts_config_distribution": metadata_int_distribution(
            "mutation_attempts_config"
        ),
        "realized_task_retries_config_distribution": metadata_int_distribution(
            "task_retries_config"
        ),
        "task_retry_events": task_retry_events,
        "duplicate_retry_events": duplicate_retry_events,
        "semantic_signature_retry_events": (
            prior_semantic_signature_retry_events
            + semantic_signature_retry_events_this_run
        ),
        "semantic_signature_retry_events_this_run": (
            semantic_signature_retry_events_this_run
        ),
        "semantic_signature_unique_count": len(semantic_signature_counts),
        "semantic_signature_max_count": max(
            semantic_signature_counts.values(), default=0
        ),
        "elapsed_seconds_this_run": elapsed,
        "items_path": str((output_dir / "items.parquet").resolve()),
        "pairs_path": str((output_dir / "pairs.parquet").resolve()),
        "metadata_path": str(
            (output_dir / "pair_generation_metadata.parquet").resolve()
        ),
        "validation_valid": bool(validation["valid"]),
        "rule_count_distribution": dict(sorted(rule_count_distribution.items())),
        "realized_two_rule_fraction": (
            two_rule_pairs / len(metadata_by_task) if metadata_by_task else 0.0
        ),
    }
    _atomic_json(errors, output_dir / "errors.json")
    _atomic_json(validation, output_dir / "validation_report.json")
    _atomic_json(summary, output_dir / "summary.json")
    return summary
