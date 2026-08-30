"""Freeze N pairs whose individual cards are globally unique across categories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from item_pipeline.normalization import (
    canonical_json_dumps,
    normalize_text,
    parse_attributes,
    stable_hash64,
)
from item_pipeline.pair_generate import (
    ATTEMPT_DIVERSITY_VERSION,
    MAX_GLOBAL_REJECTION_PROMPT_ITEMS,
    PairGenerationTask,
    SEMANTIC_SIGNATURE_VERSION,
    _attempt_diversity_nonce,
    _semantic_pair_signature,
)
from item_pipeline.pair_validation import validate_pair_dataset
from item_pipeline.rule_schedule import SCHEDULE_VERSION


SOURCE_SEMANTIC_SIGNATURE_KEYS = (
    "semantic_signature_retry",
    "semantic_signature_version",
    "semantic_signature_limit",
    "semantic_signature_unique_count",
    "semantic_signature_max_count",
    "semantic_signature_retry_events",
    "semantic_signature_retry_events_this_run",
)
SOURCE_ATTEMPT_DIVERSITY_PREFIX = "attempt_diversity_"
SCHEDULE_PROVENANCE_PREFIXES = (
    "rule_schedule_",
    "scheduled_",
    "completed_scheduled_",
    "pending_scheduled_",
    "eligible_",
    "primary_rule_",
    "secondary_rule_",
    "total_rule_",
    "category_",
    "requested_two_rule_",
    "realized_",
    "profile_capacity_",
    "profile_primary_",
    "capacity_",
    "balanced_total_",
)
SCHEDULE_PROVENANCE_KEYS = {
    "balanced_rule_schedule",
    "planned_rule_schedule",
    "realized_rule_schedule",
    "two_rule_target_clipped",
    "max_identical_scheduled_bundle_count",
}
FROZEN_SCHEDULE_VERSION = "frozen_rule_schedule_subset_v1"
FROZEN_ATTEMPT_DIVERSITY_VERSION = "frozen_attempt_diversity_subset_v1"
FROZEN_GLOBAL_CARD_UNIQUENESS_VERSION = (
    "frozen_category_agnostic_global_card_uniqueness_v1"
)
GLOBAL_CARD_KEY_VERSION = (
    "normalized_name_attributes_nfkc_casefold_yo_orderless_no_category_v1"
)
GLOBAL_CARD_DROP_REASONS = {
    "identical_pair_global_card",
    "duplicate_global_card",
}

ATTEMPT_DIVERSITY_METADATA_COLUMNS = {
    "task_index",
    "id2",
    "source_style_id",
    "attempt_diversity_version",
    "task_seed_offset",
    "task_retry_round",
    "selection_attempt",
    "anchor_attempt",
    "mutation_attempt",
    "pair_attempts_config",
    "anchor_attempts_config",
    "mutation_attempts_config",
    "anchor_diversity_nonce_sha256",
    "mutation_diversity_nonce_sha256",
    "global_rejection_feedback_count",
    "forbidden_semantic_signature_count",
    "forbidden_card_key_count",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def canonical_card(row: pd.Series) -> str:
    """Return a category-agnostic, order-insensitive normalized card key."""

    normalized_attributes = sorted(
        [normalize_text(key), normalize_text(value)]
        for key, value in parse_attributes(row["attributes"]).items()
    )
    return canonical_json_dumps(
        {
            "name": normalize_text(row["name"]),
            "attributes": normalized_attributes,
        }
    )


def _card_key_multiset_sha256(keys: list[str]) -> str:
    return _sha256_json(sorted(keys))


def global_card_uniqueness_provenance(
    source_items: pd.DataFrame,
    frozen_items: pd.DataFrame,
    dropped: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize the category-agnostic card-uniqueness freeze contract."""

    required = {"id", "name", "attributes", "category"}
    for label, frame in (("source", source_items), ("frozen", frozen_items)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{label} items lack global-card columns: {sorted(missing)}"
            )

    source_keys = [canonical_card(row) for _, row in source_items.iterrows()]
    frozen_keys = [canonical_card(row) for _, row in frozen_items.iterrows()]
    source_counts = Counter(source_keys)
    frozen_counts = Counter(frozen_keys)
    source_duplicate_keys = {
        key for key, value in source_counts.items() if value > 1
    }
    frozen_duplicate_keys = {
        key for key, value in frozen_counts.items() if value > 1
    }
    source_categories: dict[str, set[str]] = {}
    for key, category in zip(source_keys, source_items["category"], strict=True):
        source_categories.setdefault(key, set()).add(normalize_text(category))
    cross_category_duplicate_keys = {
        key
        for key in source_duplicate_keys
        if len(source_categories.get(key, set())) > 1
    }
    drop_reason_counts = Counter(str(row.get("reason") or "") for row in dropped)
    return {
        "version": FROZEN_GLOBAL_CARD_UNIQUENESS_VERSION,
        "card_key_version": GLOBAL_CARD_KEY_VERSION,
        "category_agnostic": True,
        "text_normalization": "NFKC+casefold+yo_to_e+collapsed_whitespace",
        "attributes_order_insensitive": True,
        "source_card_count": len(source_keys),
        "source_unique_card_count": len(source_counts),
        "source_duplicate_card_group_count": len(source_duplicate_keys),
        "source_duplicate_card_row_count": sum(
            source_counts[key] for key in source_duplicate_keys
        ),
        "source_duplicate_card_excess_count": sum(
            source_counts[key] - 1 for key in source_duplicate_keys
        ),
        "source_cross_category_duplicate_card_group_count": len(
            cross_category_duplicate_keys
        ),
        "source_cross_category_duplicate_card_row_count": sum(
            source_counts[key] for key in cross_category_duplicate_keys
        ),
        "source_card_keys_sha256": _card_key_multiset_sha256(source_keys),
        "frozen_card_count": len(frozen_keys),
        "frozen_unique_card_count": len(frozen_counts),
        "frozen_duplicate_card_group_count": len(frozen_duplicate_keys),
        "frozen_duplicate_card_row_count": sum(
            frozen_counts[key] for key in frozen_duplicate_keys
        ),
        "frozen_card_keys_sha256": _card_key_multiset_sha256(frozen_keys),
        "dropped_pair_count": len(dropped),
        "global_card_collision_drop_pair_count": sum(
            drop_reason_counts[reason] for reason in GLOBAL_CARD_DROP_REASONS
        ),
        "drop_reason_counts": {
            reason: int(value)
            for reason, value in sorted(drop_reason_counts.items())
        },
    }


def source_generation_provenance(source_summary: dict[str, Any]) -> dict[str, Any]:
    semantic_signature = {
        key: source_summary[key]
        for key in SOURCE_SEMANTIC_SIGNATURE_KEYS
        if key in source_summary
    }
    attempt_diversity = {
        key: value
        for key, value in source_summary.items()
        if key.startswith(SOURCE_ATTEMPT_DIVERSITY_PREFIX)
    }
    rule_schedule = {
        key: value
        for key, value in source_summary.items()
        if key in SCHEDULE_PROVENANCE_KEYS
        or key.startswith(SCHEDULE_PROVENANCE_PREFIXES)
    }
    for nested_key in (
        "rule_schedule",
        "rule_schedule_summary",
        "balanced_rule_schedule_summary",
    ):
        nested_schedule = source_summary.get(nested_key)
        if isinstance(nested_schedule, dict):
            rule_schedule.update(nested_schedule)
    return {
        "semantic_signature": semantic_signature,
        "attempt_diversity": attempt_diversity,
        "rule_schedule": rule_schedule,
    }


def _strict_metadata_int(value: Any, *, field: str, task_index: int) -> int:
    if isinstance(value, bool) or pd.isna(value):
        raise ValueError(
            f"invalid integer metadata field {field} at task {task_index}"
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"invalid integer metadata field {field} at task {task_index}"
        ) from error
    try:
        if float(value) != float(parsed):
            raise ValueError
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"non-integral metadata field {field} at task {task_index}"
        ) from error
    return parsed


def _integer_distribution(values: list[int]) -> dict[str, int]:
    counts = Counter(values)
    return {str(key): int(counts[key]) for key in sorted(counts)}


def attempt_diversity_provenance(
    metadata: pd.DataFrame,
    source_summary: dict[str, Any],
) -> dict[str, Any]:
    """Validate and summarize the accepted-row attempt-diversity protocol."""

    if metadata.empty:
        raise ValueError("attempt-diversity metadata must not be empty")
    missing = ATTEMPT_DIVERSITY_METADATA_COLUMNS - set(metadata.columns)
    if missing:
        raise ValueError(
            "metadata lacks attempt-diversity columns: " f"{sorted(missing)}"
        )
    reported_version = str(
        source_summary.get("attempt_diversity_version") or ""
    )
    if reported_version != ATTEMPT_DIVERSITY_VERSION:
        raise ValueError(
            "source summary has an unexpected attempt-diversity version: "
            f"{reported_version!r}"
        )
    observed_versions = set(metadata["attempt_diversity_version"].astype(str))
    if observed_versions != {ATTEMPT_DIVERSITY_VERSION}:
        raise ValueError(
            "metadata mixes or uses unexpected attempt-diversity versions: "
            f"{sorted(observed_versions)}"
        )
    try:
        run_seed = int(source_summary["seed"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("source summary has no valid generation seed") from error

    anchor_hashes: list[str] = []
    mutation_hashes: list[str] = []
    feedback_counts: list[int] = []
    forbidden_signature_counts: list[int] = []
    forbidden_card_counts: list[int] = []
    for row in metadata.to_dict("records"):
        task_index = _strict_metadata_int(
            row["task_index"], field="task_index", task_index=-1
        )
        source_style_id = _strict_metadata_int(
            row["source_style_id"],
            field="source_style_id",
            task_index=task_index,
        )
        task_seed_offset = _strict_metadata_int(
            row["task_seed_offset"],
            field="task_seed_offset",
            task_index=task_index,
        )
        task_retry_round = _strict_metadata_int(
            row["task_retry_round"],
            field="task_retry_round",
            task_index=task_index,
        )
        selection_attempt = _strict_metadata_int(
            row["selection_attempt"],
            field="selection_attempt",
            task_index=task_index,
        )
        pair_attempts = _strict_metadata_int(
            row["pair_attempts_config"],
            field="pair_attempts_config",
            task_index=task_index,
        )
        anchor_attempt = _strict_metadata_int(
            row["anchor_attempt"], field="anchor_attempt", task_index=task_index
        )
        mutation_attempt = _strict_metadata_int(
            row["mutation_attempt"],
            field="mutation_attempt",
            task_index=task_index,
        )
        anchor_attempts = _strict_metadata_int(
            row["anchor_attempts_config"],
            field="anchor_attempts_config",
            task_index=task_index,
        )
        mutation_attempts = _strict_metadata_int(
            row["mutation_attempts_config"],
            field="mutation_attempts_config",
            task_index=task_index,
        )
        if (
            task_retry_round < 0
            or not 1 <= selection_attempt <= pair_attempts
            or not 1 <= anchor_attempt <= anchor_attempts
            or not 1 <= mutation_attempt <= mutation_attempts
        ):
            raise ValueError(
                f"invalid attempt coordinates at task {task_index}"
            )
        task = PairGenerationTask(
            task_index=task_index,
            mutated_id=_strict_metadata_int(
                row["id2"], field="id2", task_index=task_index
            ),
            seed=int(stable_hash64(run_seed, source_style_id) % (2**31 - 1)),
            anchor={"id": source_style_id},
        )
        common = {
            "task_seed_offset": task_seed_offset,
            "task_retry_round": task_retry_round,
            "selection_attempt": selection_attempt,
        }
        expected_anchor_hash = hashlib.sha256(
            _attempt_diversity_nonce(
                task,
                **common,
                stage="anchor",
                stage_attempt=anchor_attempt,
            ).encode("utf-8")
        ).hexdigest()
        expected_mutation_hash = hashlib.sha256(
            _attempt_diversity_nonce(
                task,
                **common,
                stage="mutation",
                stage_attempt=mutation_attempt,
            ).encode("utf-8")
        ).hexdigest()
        anchor_hash = str(row["anchor_diversity_nonce_sha256"])
        mutation_hash = str(row["mutation_diversity_nonce_sha256"])
        if anchor_hash != expected_anchor_hash:
            raise ValueError(
                f"anchor diversity nonce hash mismatch at task {task_index}"
            )
        if mutation_hash != expected_mutation_hash:
            raise ValueError(
                f"mutation diversity nonce hash mismatch at task {task_index}"
            )
        anchor_hashes.append(anchor_hash)
        mutation_hashes.append(mutation_hash)

        feedback_count = _strict_metadata_int(
            row["global_rejection_feedback_count"],
            field="global_rejection_feedback_count",
            task_index=task_index,
        )
        forbidden_signature_count = _strict_metadata_int(
            row["forbidden_semantic_signature_count"],
            field="forbidden_semantic_signature_count",
            task_index=task_index,
        )
        forbidden_card_count = _strict_metadata_int(
            row["forbidden_card_key_count"],
            field="forbidden_card_key_count",
            task_index=task_index,
        )
        if not 0 <= feedback_count <= MAX_GLOBAL_REJECTION_PROMPT_ITEMS:
            raise ValueError(
                f"invalid global rejection feedback count at task {task_index}"
            )
        if forbidden_signature_count < 0 or forbidden_card_count < 0:
            raise ValueError(
                f"negative forbidden diversity count at task {task_index}"
            )
        feedback_counts.append(feedback_count)
        forbidden_signature_counts.append(forbidden_signature_count)
        forbidden_card_counts.append(forbidden_card_count)

    row_count = len(metadata)
    if len(set(anchor_hashes)) != row_count:
        raise ValueError("accepted rows reuse an anchor diversity nonce")
    if len(set(mutation_hashes)) != row_count:
        raise ValueError("accepted rows reuse a mutation diversity nonce")
    return {
        "version": FROZEN_ATTEMPT_DIVERSITY_VERSION,
        "attempt_diversity_version": ATTEMPT_DIVERSITY_VERSION,
        "selected_task_count": row_count,
        "anchor_nonce_hash_valid_count": row_count,
        "anchor_nonce_hash_unique_count": len(set(anchor_hashes)),
        "mutation_nonce_hash_valid_count": row_count,
        "mutation_nonce_hash_unique_count": len(set(mutation_hashes)),
        "global_rejection_feedback_count_distribution": (
            _integer_distribution(feedback_counts)
        ),
        "forbidden_semantic_signature_count_distribution": (
            _integer_distribution(forbidden_signature_counts)
        ),
        "forbidden_card_key_count_distribution": (
            _integer_distribution(forbidden_card_counts)
        ),
    }


def _json_list(value: Any, *, field: str) -> list[Any]:
    if isinstance(value, list):
        parsed = value
    else:
        try:
            parsed = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid JSON list in metadata field {field}") from error
    if not isinstance(parsed, list):
        raise ValueError(f"metadata field {field} is not a JSON list")
    return parsed


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def frozen_subset_provenance(
    metadata: pd.DataFrame,
    source_summary: dict[str, Any],
) -> dict[str, Any]:
    """Recompute schedule and semantic-cap facts for the published subset."""

    if metadata.empty:
        raise ValueError("frozen metadata must not be empty")
    required = {
        "task_index",
        "source_style_id",
        "category",
        "product_type",
        "rule_count",
        "rule_ids",
        "applications_json",
        "semantic_signature",
        "semantic_signature_version",
        "balanced_rule_schedule",
        "rule_schedule_version",
        "rule_schedule_sha256",
        "scheduled_primary_rule_id",
        "scheduled_primary_profile_id",
        "scheduled_primary_task_cap",
        "scheduled_secondary_rule_id",
        "scheduled_secondary_profile_id",
        "scheduled_rule_ids",
        "scheduled_rule_profile_ids",
        "profile_capacity_policy_version",
        "profile_capacity_policy_sha256",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(
            "frozen metadata lacks schedule/signature columns: "
            f"{sorted(missing)}"
        )
    if metadata["task_index"].duplicated().any():
        raise ValueError("frozen metadata contains duplicate task indices")
    if metadata["source_style_id"].duplicated().any():
        raise ValueError("frozen metadata reuses a style donor")
    if not metadata["balanced_rule_schedule"].map(bool).all():
        raise ValueError("frozen metadata disables balanced scheduling")

    schedule_versions = set(metadata["rule_schedule_version"].astype(str))
    if schedule_versions != {SCHEDULE_VERSION}:
        raise ValueError(
            f"unexpected frozen schedule versions: {sorted(schedule_versions)}"
        )
    source_schedule_sha = str(source_summary.get("rule_schedule_sha256") or "")
    schedule_hashes = set(metadata["rule_schedule_sha256"].astype(str))
    if len(schedule_hashes) != 1:
        raise ValueError("frozen metadata mixes multiple source schedules")
    metadata_schedule_sha = next(iter(schedule_hashes))
    if source_schedule_sha and metadata_schedule_sha != source_schedule_sha:
        raise ValueError("frozen metadata schedule differs from source summary")

    signature_versions = set(metadata["semantic_signature_version"].astype(str))
    if signature_versions != {SEMANTIC_SIGNATURE_VERSION}:
        raise ValueError(
            "unexpected frozen semantic-signature versions: "
            f"{sorted(signature_versions)}"
        )
    signature_limit = int(source_summary.get("semantic_signature_limit", 0))
    if signature_limit < 1:
        raise ValueError("source summary has no positive semantic-signature limit")
    policy_versions = set(metadata["profile_capacity_policy_version"].astype(str))
    policy_hashes = set(metadata["profile_capacity_policy_sha256"].astype(str))
    expected_policy_version = str(
        source_summary.get("profile_capacity_policy_version") or ""
    )
    expected_policy_sha = str(
        source_summary.get("profile_capacity_policy_sha256") or ""
    )
    if bool(expected_policy_version) != bool(expected_policy_sha):
        raise ValueError("source capacity-policy version/SHA presence differs")
    if not expected_policy_version:
        # Catalogs without finite-domain profiles intentionally have no policy.
        # Preserve that signed absence instead of attributing another catalog's
        # checked-in policy to the frozen subset.
        if policy_versions != {""} or policy_hashes != {""}:
            raise ValueError(
                "frozen metadata invents a capacity policy absent from source"
            )
        if metadata["scheduled_primary_task_cap"].notna().any():
            raise ValueError("policy-free frozen metadata contains primary caps")
        policy_version = ""
        policy_sha256 = ""
    else:
        if policy_versions != {expected_policy_version}:
            raise ValueError("frozen metadata capacity-policy version differs from source")
        if policy_hashes != {expected_policy_sha} or len(expected_policy_sha) != 64:
            raise ValueError("frozen metadata capacity-policy SHA differs from source")
        policy_version = expected_policy_version
        policy_sha256 = expected_policy_sha

    primary_rule_usage: Counter[str] = Counter()
    secondary_rule_usage: Counter[str] = Counter()
    primary_profile_usage: Counter[str] = Counter()
    secondary_profile_usage: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    signature_counts: Counter[str] = Counter()
    primary_caps: dict[str, int] = {}
    schedule_payload: list[dict[str, Any]] = []

    ordered = metadata.sort_values("task_index", kind="stable")
    task_indices = ordered["task_index"].astype(int).tolist()
    for row in ordered.to_dict("records"):
        task_index = int(row["task_index"])
        actual_rule_ids = [
            str(value) for value in _json_list(row["rule_ids"], field="rule_ids")
        ]
        scheduled_rule_ids = [
            str(value)
            for value in _json_list(
                row["scheduled_rule_ids"], field="scheduled_rule_ids"
            )
        ]
        scheduled_profile_ids = [
            str(value)
            for value in _json_list(
                row["scheduled_rule_profile_ids"],
                field="scheduled_rule_profile_ids",
            )
        ]
        if (
            actual_rule_ids != scheduled_rule_ids
            or len(scheduled_rule_ids) not in {1, 2}
            or len(scheduled_profile_ids) != len(scheduled_rule_ids)
            or int(row["rule_count"]) != len(scheduled_rule_ids)
        ):
            raise ValueError(
                f"actual rules differ from frozen schedule at task {task_index}"
            )

        primary_rule_id = str(row["scheduled_primary_rule_id"])
        primary_profile_id = str(row["scheduled_primary_profile_id"])
        if (
            primary_rule_id != scheduled_rule_ids[0]
            or primary_profile_id != scheduled_profile_ids[0]
        ):
            raise ValueError(f"invalid frozen primary rule at task {task_index}")
        secondary_rule = row["scheduled_secondary_rule_id"]
        secondary_profile = row["scheduled_secondary_profile_id"]
        secondary_rule = None if pd.isna(secondary_rule) else str(secondary_rule)
        secondary_profile = (
            None if pd.isna(secondary_profile) else str(secondary_profile)
        )
        expected_secondary_rule = (
            scheduled_rule_ids[1] if len(scheduled_rule_ids) == 2 else None
        )
        expected_secondary_profile = (
            scheduled_profile_ids[1] if len(scheduled_profile_ids) == 2 else None
        )
        if (
            secondary_rule != expected_secondary_rule
            or secondary_profile != expected_secondary_profile
        ):
            raise ValueError(f"invalid frozen secondary rule at task {task_index}")

        applications = _json_list(
            row["applications_json"], field="applications_json"
        )
        if not all(isinstance(value, dict) for value in applications):
            raise ValueError(f"invalid frozen applications at task {task_index}")
        derived_signature = _semantic_pair_signature(
            row["category"], row["product_type"], applications
        )
        if str(row["semantic_signature"]) != derived_signature:
            raise ValueError(
                f"semantic signature differs in frozen task {task_index}"
            )
        signature_counts[derived_signature] += 1

        primary_rule_usage[primary_rule_id] += 1
        primary_profile_usage[primary_profile_id] += 1
        if secondary_rule is not None and secondary_profile is not None:
            secondary_rule_usage[secondary_rule] += 1
            secondary_profile_usage[secondary_profile] += 1
        category_counts[str(row["category"])] += 1

        raw_cap = row["scheduled_primary_task_cap"]
        if not pd.isna(raw_cap):
            cap = int(raw_cap)
            prior = primary_caps.setdefault(primary_profile_id, cap)
            if prior != cap or cap < 1:
                raise ValueError(
                    f"inconsistent primary cap for profile {primary_profile_id}"
                )
        schedule_payload.append(
            {
                "task_index": task_index,
                "donor_id": int(row["source_style_id"]),
                "category": str(row["category"]),
                "profiles": scheduled_profile_ids,
            }
        )

    signature_max = max(signature_counts.values(), default=0)
    if signature_max > signature_limit:
        raise ValueError(
            "frozen subset exceeds semantic-signature limit: "
            f"{signature_max} > {signature_limit}"
        )
    cap_violations = {
        profile_id: {"usage": primary_profile_usage[profile_id], "cap": cap}
        for profile_id, cap in primary_caps.items()
        if primary_profile_usage[profile_id] > cap
    }
    if cap_violations:
        raise ValueError(f"frozen subset exceeds profile caps: {cap_violations}")

    all_rule_ids = set(primary_rule_usage) | set(secondary_rule_usage)
    all_profile_ids = set(primary_profile_usage) | set(secondary_profile_usage)
    total_rule_usage = primary_rule_usage + secondary_rule_usage
    total_profile_usage = primary_profile_usage + secondary_profile_usage
    source_eligible_rules = int(source_summary.get("eligible_rules", 0))
    source_eligible_profiles = int(source_summary.get("eligible_rule_profiles", 0))
    two_rule_tasks = int(sum(secondary_rule_usage.values()))
    selected_count = len(ordered)
    return {
        "version": FROZEN_SCHEDULE_VERSION,
        "source_rule_schedule_version": SCHEDULE_VERSION,
        "source_rule_schedule_sha256": metadata_schedule_sha,
        "profile_capacity_policy_version": policy_version,
        "profile_capacity_policy_sha256": policy_sha256,
        "selected_task_count": selected_count,
        "selected_task_index_sha256": _sha256_json(task_indices),
        "selected_schedule_sha256": _sha256_json(schedule_payload),
        "selected_task_index_min": min(task_indices),
        "selected_task_index_max": max(task_indices),
        "source_eligible_rules": source_eligible_rules,
        "source_eligible_rule_profiles": source_eligible_profiles,
        "primary_rule_coverage": len(primary_rule_usage),
        "primary_rule_profile_coverage": len(primary_profile_usage),
        "total_rule_coverage": len(all_rule_ids),
        "total_rule_profile_coverage": len(all_profile_ids),
        "full_primary_rule_coverage": (
            source_eligible_rules > 0
            and len(primary_rule_usage) == source_eligible_rules
        ),
        "full_primary_rule_profile_coverage": (
            source_eligible_profiles > 0
            and len(primary_profile_usage) == source_eligible_profiles
        ),
        "category_task_counts": dict(sorted(category_counts.items())),
        "primary_rule_usage": dict(sorted(primary_rule_usage.items())),
        "secondary_rule_usage": dict(sorted(secondary_rule_usage.items())),
        "total_rule_usage": dict(sorted(total_rule_usage.items())),
        "primary_rule_profile_usage": dict(
            sorted(primary_profile_usage.items())
        ),
        "secondary_rule_profile_usage": dict(
            sorted(secondary_profile_usage.items())
        ),
        "total_rule_profile_usage": dict(sorted(total_profile_usage.items())),
        "primary_rule_profile_caps": dict(sorted(primary_caps.items())),
        "primary_rule_profile_cap_violations": {},
        "two_rule_tasks": two_rule_tasks,
        "two_rule_fraction": two_rule_tasks / selected_count,
        "semantic_signature_version": SEMANTIC_SIGNATURE_VERSION,
        "semantic_signature_limit": signature_limit,
        "semantic_signature_unique_count": len(signature_counts),
        "semantic_signature_max_count": signature_max,
    }


def freeze(source_dir: Path, output_dir: Path, count: int) -> dict[str, Any]:
    if count < 1:
        raise ValueError("count must be positive")
    source_dir, output_dir = source_dir.resolve(), output_dir.resolve()
    paths = {
        "items": source_dir / "items.parquet",
        "pairs": source_dir / "pairs.parquet",
        "metadata": source_dir / "pair_generation_metadata.parquet",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source artifacts: {missing}")
    if output_dir == source_dir:
        raise ValueError("output-dir must differ from source-dir")

    items = pd.read_parquet(paths["items"])
    pairs = pd.read_parquet(paths["pairs"])
    metadata = pd.read_parquet(paths["metadata"])
    if items["id"].duplicated().any():
        raise ValueError("source items contain duplicate IDs")
    if pairs[["id1", "id2"]].duplicated().any():
        raise ValueError("source pairs contain duplicate ordered IDs")
    if metadata[["id1", "id2"]].duplicated().any():
        raise ValueError("source metadata contain duplicate ordered IDs")
    metadata_by_pair = metadata.set_index(["id1", "id2"], drop=False)
    item_by_id = items.set_index("id", drop=False)
    source_card_keys = {
        int(row["id"]): canonical_card(row) for _, row in items.iterrows()
    }

    selected_pairs: list[dict[str, Any]] = []
    selected_metadata: list[pd.Series] = []
    selected_item_ids: list[int] = []
    seen_cards: set[str] = set()
    seen_item_ids: set[int] = set()
    dropped: list[dict[str, Any]] = []
    for pair in pairs.itertuples(index=False):
        id1, id2 = int(pair.id1), int(pair.id2)
        reason = ""
        if id1 not in item_by_id.index or id2 not in item_by_id.index:
            reason = "missing_item"
        elif (id1, id2) not in metadata_by_pair.index:
            reason = "missing_metadata"
        elif id1 in seen_item_ids or id2 in seen_item_ids:
            reason = "reused_item_id"
        else:
            left_key = source_card_keys[id1]
            right_key = source_card_keys[id2]
            if left_key == right_key:
                reason = "identical_pair_global_card"
            elif left_key in seen_cards or right_key in seen_cards:
                reason = "duplicate_global_card"
        if reason:
            dropped.append({"id1": id1, "id2": id2, "reason": reason})
            continue

        selected_pairs.append({"id1": id1, "id2": id2, "target": int(pair.target)})
        selected_metadata.append(metadata_by_pair.loc[(id1, id2)])
        selected_item_ids.extend([id1, id2])
        seen_item_ids.update([id1, id2])
        seen_cards.update([left_key, right_key])
        if len(selected_pairs) == count:
            break

    if len(selected_pairs) < count:
        raise RuntimeError(
            f"only {len(selected_pairs)} globally unique pairs remain; requested={count}"
        )

    frozen_pairs = pd.DataFrame(selected_pairs, columns=["id1", "id2", "target"])
    frozen_items = item_by_id.loc[selected_item_ids].reset_index(drop=True)
    frozen_metadata = pd.DataFrame(selected_metadata).reset_index(drop=True)
    base_items = item_by_id.loc[frozen_pairs["id1"].tolist()].reset_index(drop=True)
    mutated_items = item_by_id.loc[frozen_pairs["id2"].tolist()].reset_index(drop=True)
    global_card_uniqueness = global_card_uniqueness_provenance(
        items,
        frozen_items,
        dropped,
    )
    if (
        global_card_uniqueness["frozen_card_count"] != count * 2
        or global_card_uniqueness["frozen_unique_card_count"] != count * 2
        or global_card_uniqueness["frozen_duplicate_card_group_count"] != 0
        or global_card_uniqueness["frozen_duplicate_card_row_count"] != 0
    ):
        raise RuntimeError(
            "frozen individual cards are not globally unique across categories"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_parquet(base_items, output_dir / "base_items.parquet")
    atomic_parquet(mutated_items, output_dir / "mutated_items.parquet")
    atomic_parquet(frozen_items, output_dir / "items.parquet")
    atomic_parquet(frozen_pairs, output_dir / "pairs.parquet")
    atomic_parquet(
        frozen_metadata, output_dir / "pair_generation_metadata.parquet"
    )
    report = validate_pair_dataset(
        output_dir / "items.parquet",
        output_dir / "pairs.parquet",
        metadata_path=output_dir / "pair_generation_metadata.parquet",
    )
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if report.get("valid") is not True:
        raise RuntimeError(f"frozen dataset did not validate: {report}")

    source_summary_path = source_dir / "summary.json"
    source_summary = (
        json.loads(source_summary_path.read_text(encoding="utf-8"))
        if source_summary_path.is_file()
        else {}
    )
    generation_provenance = source_generation_provenance(source_summary)
    frozen_provenance = frozen_subset_provenance(frozen_metadata, source_summary)
    frozen_attempt_diversity = attempt_diversity_provenance(
        frozen_metadata, source_summary
    )
    summary = {
        "version": "frozen_generated_pairs_v3",
        "generated_pairs": count,
        "pending": 0,
        "validation_valid": True,
        "source_dir": str(source_dir),
        "source_generated_pairs": len(pairs),
        "source_run_signature": source_summary.get("run_signature"),
        "source_model": source_summary.get("model"),
        "source_api_base_url": source_summary.get("api_base_url"),
        "source_structured_output": source_summary.get("structured_output"),
        "source_reasoning_effort": source_summary.get("reasoning_effort"),
        "source_prompt_sha256": source_summary.get("prompt_sha256"),
        "source_rule_catalogs": source_summary.get("rule_catalogs"),
        "source_rule_catalog_summary": source_summary.get("rule_catalog_summary"),
        "source_rule_tiers": source_summary.get("rule_tiers"),
        "source_seed": source_summary.get("seed"),
        "source_base_items_path": source_summary.get("base_items_path"),
        "source_base_items_sha256": source_summary.get("base_items_sha256"),
        "source_label_one_fraction": source_summary.get("label_one_fraction"),
        "source_planned_target_counts": source_summary.get(
            "planned_target_counts"
        ),
        "source_realized_target_counts": source_summary.get(
            "realized_target_counts"
        ),
        "source_two_rule_fraction": source_summary.get("two_rule_fraction"),
        "source_realized_two_rule_fraction": source_summary.get(
            "realized_two_rule_fraction"
        ),
        "source_rule_count_distribution": source_summary.get(
            "rule_count_distribution"
        ),
        **{
            f"source_{key}": source_summary.get(key)
            for key in SOURCE_SEMANTIC_SIGNATURE_KEYS
        },
        "source_attempt_diversity_version": source_summary.get(
            "attempt_diversity_version"
        ),
        "source_rule_schedule": generation_provenance["rule_schedule"],
        "source_generation_provenance": generation_provenance,
        "frozen_rule_schedule": frozen_provenance,
        "frozen_attempt_diversity": frozen_attempt_diversity,
        "frozen_global_card_uniqueness": global_card_uniqueness,
        "frozen_semantic_signature": {
            key: frozen_provenance[key]
            for key in (
                "semantic_signature_version",
                "semantic_signature_limit",
                "semantic_signature_unique_count",
                "semantic_signature_max_count",
            )
        },
        "dropped_before_target": len(dropped),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "dropped_pairs.json").write_text(
        json.dumps(dropped, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "source_dir": str(source_dir),
        "source_files": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "source_summary": (
            {
                "path": str(source_summary_path),
                "sha256": sha256(source_summary_path),
                "run_signature": source_summary.get("run_signature"),
            }
            if source_summary_path.is_file()
            else None
        ),
        "source_generation_provenance": generation_provenance,
        "source_semantic_signature": generation_provenance["semantic_signature"],
        "source_rule_schedule": generation_provenance["rule_schedule"],
        "frozen_rule_schedule": frozen_provenance,
        "frozen_attempt_diversity": frozen_attempt_diversity,
        "frozen_global_card_uniqueness": global_card_uniqueness,
        "frozen_semantic_signature": summary["frozen_semantic_signature"],
        "count": count,
        "dropped_before_target": len(dropped),
        "drop_reason_counts": {
            str(reason): int(value)
            for reason, value in pd.Series(
                [row["reason"] for row in dropped], dtype="object"
            ).value_counts().items()
        },
        "output_files": {
            name: {
                "path": str(output_dir / name),
                "sha256": sha256(output_dir / name),
            }
            for name in (
                "base_items.parquet",
                "mutated_items.parquet",
                "items.parquet",
                "pairs.parquet",
                "pair_generation_metadata.parquet",
                "validation_report.json",
                "summary.json",
                "dropped_pairs.json",
            )
        },
    }
    (output_dir / "freeze_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"summary": summary, "validation": report, "manifest": manifest}


def main() -> None:
    args = parse_args()
    result = freeze(args.source_dir, args.output_dir, args.count)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
