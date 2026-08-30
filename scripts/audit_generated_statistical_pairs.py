#!/usr/bin/env python3
"""Audit a generated statistical-rule pair checkpoint without modifying it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from item_pipeline.normalization import (  # noqa: E402
    canonical_json_dumps,
    normalize_text,
    parse_attributes,
    stable_hash64,
    tokenize,
)
from item_pipeline.pair_generate import (  # noqa: E402
    ATTEMPT_DIVERSITY_VERSION,
    MAX_GLOBAL_REJECTION_PROMPT_ITEMS,
    PairGenerationTask,
    SEMANTIC_SIGNATURE_VERSION,
    _attempt_diversity_nonce,
    _semantic_pair_signature,
)


AUDIT_VERSION = "generated_statistical_pairs_audit_v2"
DEFAULT_REFERENCE_ITEMS = ROOT / "data" / "items_human.parquet"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "generated_statistical_pair_audits"
DEFAULT_SEMANTIC_SIGNATURE_CAP = 2
DEFAULT_SAMPLE_LIMIT = 20

REQUIRED_SOURCE_FILES = {
    "metadata": "pair_generation_metadata.parquet",
    "base_items": "base_items.parquet",
    "mutated_items": "mutated_items.parquet",
    "pairs": "pairs.parquet",
}

IDENTIFIER_KEY_RE = re.compile(
    r"(?<!\w)(?:sku|артикул(?:а|у|ом|е|ы|ов)?)(?!\w)", re.IGNORECASE
)
IDENTIFIER_TITLE_RE = re.compile(
    r"(?<!\w)(?:sku|артикул(?:а|у|ом|е|ы|ов)?)\s*[:#№-]?\s*[a-zа-яё0-9]",
    re.IGNORECASE,
)
MODEL_KEY_RE = re.compile(
    r"(?<!\w)(?:модель|model)(?!\w)", re.IGNORECASE
)
NUMERIC_ONLY_RE = re.compile(r"^[+-]?\d+(?:[.,]\d+)?$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MODEL_CONCEPTS = {
    "compatible_model",
    "compatible_models",
    "compatible_phone_model",
    "cpu_model",
    "model",
    "model_line",
    "model_series",
    "processor_model",
    "vehicle_compatibility",
}
MATERIAL_CONCEPTS = {
    "fabric_type",
    "filler_material",
    "material",
    "material_composition",
    "material_type",
}
MATERIAL_EQUIVALENCE_GROUPS = tuple(
    frozenset(normalize_text(value) for value in group)
    for group in (
        {
            "искусственная кожа",
            "экокожа",
            "эко кожа",
            "кожзам",
            "кожзаменитель",
            "pu кожа",
            "полиуретановая кожа",
            "винилискожа",
        },
        {"натуральная кожа", "кожа натуральная", "genuine leather"},
        {"хлопок", "100% хлопок", "100 % хлопок", "чистый хлопок"},
        {"шерсть", "100% шерсть", "100 % шерсть", "натуральная шерсть"},
        {"полиэстер", "полиэфир", "polyester"},
        {"искусственный мех", "экомех", "эко мех"},
        {"нержавеющая сталь", "сталь нержавеющая", "stainless steel"},
    )
)


class IssueCollector:
    """Collect row-level issues with stable, bounded task-index examples."""

    def __init__(self, sample_limit: int) -> None:
        self.sample_limit = sample_limit
        self._tasks: dict[str, set[int]] = defaultdict(set)

    def add(self, issue: str, task_index: Any) -> None:
        try:
            task = int(task_index)
        except (TypeError, ValueError, OverflowError):
            return
        self._tasks[issue].add(task)

    def add_many(self, issue: str, task_indices: Iterable[Any]) -> None:
        for task_index in task_indices:
            self.add(issue, task_index)

    def task_indices(self, issue: str) -> list[int]:
        return sorted(self._tasks.get(issue, ()))

    def report(self) -> dict[str, dict[str, Any]]:
        return {
            issue: {
                "affected_task_count": len(tasks),
                "sample_task_indices": sorted(tasks)[: self.sample_limit],
            }
            for issue, tasks in sorted(self._tasks.items())
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-items",
        type=Path,
        default=DEFAULT_REFERENCE_ITEMS if DEFAULT_REFERENCE_ITEMS.exists() else None,
        help=(
            "Human item parquet used for complexity comparison. Defaults to "
            "data/items_human.parquet when it exists."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "JSON report path. Defaults to reports/generated_statistical_pair_audits/"
            "<source directory>.json. The output may not be inside --source-dir."
        ),
    )
    parser.add_argument(
        "--semantic-signature-cap",
        type=int,
        help="Override the cap from summary.json (fallback: 2).",
    )
    parser.add_argument("--sample-limit", type=int, default=DEFAULT_SAMPLE_LIMIT)
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _json_list(value: Any) -> list[Any] | None:
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, list) else None


def _task_index(row: Mapping[str, Any], fallback: int) -> int:
    try:
        return int(row.get("task_index", fallback))
    except (TypeError, ValueError, OverflowError):
        return fallback


def _safe_attributes(value: Any) -> dict[str, str] | None:
    try:
        return parse_attributes(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _canonical_card(row: Mapping[str, Any]) -> str | None:
    attributes = _safe_attributes(row.get("attributes"))
    if attributes is None:
        return None
    normalized_attributes = {
        normalize_text(key): normalize_text(value)
        for key, value in attributes.items()
    }
    return canonical_json_dumps(
        {
            "name": normalize_text(row.get("name")),
            "attributes": normalized_attributes,
            "category": normalize_text(row.get("category")),
        }
    )


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _distribution(values: Iterable[int | float]) -> dict[str, Any]:
    series = pd.Series(list(values), dtype="float64")
    if series.empty:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "max": None,
            "mean": None,
        }
    quantiles = series.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "count": int(len(series)),
        "min": float(series.min()),
        "p10": float(quantiles.loc[0.10]),
        "p25": float(quantiles.loc[0.25]),
        "p50": float(quantiles.loc[0.50]),
        "p75": float(quantiles.loc[0.75]),
        "p90": float(quantiles.loc[0.90]),
        "max": float(series.max()),
        "mean": float(series.mean()),
    }


def _usage_statistics(counter: Counter[str]) -> dict[str, Any]:
    counts = pd.Series(list(counter.values()), dtype="float64")
    if counts.empty:
        return {
            "unique_ids": 0,
            "total_assignments": 0,
            "counts": {},
            "min_count": 0,
            "max_count": 0,
            "mean_count": 0.0,
            "max_to_min_ratio": None,
            "coefficient_of_variation": None,
        }
    mean = float(counts.mean())
    minimum = int(counts.min())
    return {
        "unique_ids": int(len(counter)),
        "total_assignments": int(counts.sum()),
        "counts": _counter_dict(counter),
        "min_count": minimum,
        "max_count": int(counts.max()),
        "mean_count": mean,
        "max_to_min_ratio": float(counts.max() / minimum) if minimum else None,
        "coefficient_of_variation": (
            float(counts.std(ddof=0) / mean) if mean else None
        ),
    }


def _material_values_equivalent(left: Any, right: Any) -> bool:
    normalized_left = normalize_text(left)
    normalized_right = normalize_text(right)
    if not normalized_left or not normalized_right:
        return False
    if normalized_left == normalized_right:
        return True
    return any(
        normalized_left in group and normalized_right in group
        for group in MATERIAL_EQUIVALENCE_GROUPS
    )


def _is_numeric_only(value: Any) -> bool:
    return NUMERIC_ONLY_RE.fullmatch(normalize_text(value).replace(" ", "")) is not None


def _complexity(frame: pd.DataFrame) -> dict[str, Any]:
    _require_columns(frame, {"name", "attributes"}, "items")
    name_word_counts = [len(tokenize(value)) for value in frame["name"]]
    attribute_counts: list[int] = []
    parse_errors = 0
    for value in frame["attributes"]:
        attributes = _safe_attributes(value)
        if attributes is None:
            parse_errors += 1
            continue
        attribute_counts.append(len(attributes))
    return {
        "rows": int(len(frame)),
        "name_word_count": _distribution(name_word_counts),
        "attribute_count": _distribution(attribute_counts),
        "attribute_parse_error_rows": parse_errors,
    }


def _complexity_comparison(
    synthetic: dict[str, Any], reference: dict[str, Any] | None
) -> dict[str, float | None]:
    if reference is None:
        return {
            "name_word_mean_ratio": None,
            "name_word_p50_delta": None,
            "attribute_mean_ratio": None,
            "attribute_p50_delta": None,
        }

    def ratio(numerator: Any, denominator: Any) -> float | None:
        if numerator is None or denominator in (None, 0):
            return None
        return float(numerator / denominator)

    synthetic_words = synthetic["name_word_count"]
    reference_words = reference["name_word_count"]
    synthetic_attributes = synthetic["attribute_count"]
    reference_attributes = reference["attribute_count"]
    return {
        "name_word_mean_ratio": ratio(
            synthetic_words["mean"], reference_words["mean"]
        ),
        "name_word_p50_delta": (
            float(synthetic_words["p50"] - reference_words["p50"])
            if synthetic_words["p50"] is not None
            and reference_words["p50"] is not None
            else None
        ),
        "attribute_mean_ratio": ratio(
            synthetic_attributes["mean"], reference_attributes["mean"]
        ),
        "attribute_p50_delta": (
            float(synthetic_attributes["p50"] - reference_attributes["p50"])
            if synthetic_attributes["p50"] is not None
            and reference_attributes["p50"] is not None
            else None
        ),
    }


def _effective_signature_cap(
    summary: dict[str, Any] | None, override: int | None
) -> tuple[int, str]:
    if override is not None:
        if override < 1:
            raise ValueError("semantic signature cap must be positive")
        return override, "cli"
    if summary is not None:
        try:
            reported = int(summary.get("semantic_signature_limit", 0))
        except (TypeError, ValueError):
            reported = 0
        if reported > 0:
            return reported, "summary"
    return DEFAULT_SEMANTIC_SIGNATURE_CAP, "default"


def _audit_semantic_signatures(
    metadata_records: list[dict[str, Any]],
    summary: dict[str, Any] | None,
    cap: int,
    cap_source: str,
    issues: IssueCollector,
) -> dict[str, Any]:
    derived_by_task: dict[int, str] = {}
    stored_by_task: dict[int, str] = {}
    version_by_task: dict[int, str] = {}

    for position, row in enumerate(metadata_records):
        task = _task_index(row, position)
        applications = _json_list(row.get("applications_json"))
        if applications is None or not all(
            isinstance(application, dict) for application in applications
        ):
            issues.add("invalid_applications_json", task)
            continue
        derived = _semantic_pair_signature(
            row.get("category"), row.get("product_type"), applications
        )
        derived_by_task[task] = derived
        stored = row.get("semantic_signature")
        if not isinstance(stored, str) or not stored:
            issues.add("missing_semantic_signature", task)
        else:
            stored_by_task[task] = stored
            if stored != derived:
                issues.add("semantic_signature_mismatch", task)
        version = row.get("semantic_signature_version")
        if not isinstance(version, str) or not version:
            issues.add("missing_semantic_signature_version", task)
        else:
            version_by_task[task] = version
            if version != SEMANTIC_SIGNATURE_VERSION:
                issues.add("semantic_signature_version_mismatch", task)

    tasks_by_signature: dict[str, list[int]] = defaultdict(list)
    for task, signature in derived_by_task.items():
        tasks_by_signature[signature].append(task)
    signature_counts = Counter(
        {signature: len(tasks) for signature, tasks in tasks_by_signature.items()}
    )
    over_cap = {
        signature: tasks
        for signature, tasks in tasks_by_signature.items()
        if len(tasks) > cap
    }
    for tasks in over_cap.values():
        issues.add_many("semantic_signature_cap_exceeded", tasks)

    count_histogram = Counter(str(count) for count in signature_counts.values())
    top_repeated = [
        {
            "signature": signature,
            "count": int(count),
            "sample_task_indices": sorted(tasks_by_signature[signature])[
                : issues.sample_limit
            ],
        }
        for signature, count in sorted(
            signature_counts.items(), key=lambda pair: (-pair[1], pair[0])
        )
        if count > 1
    ][: issues.sample_limit]

    unique_count = len(signature_counts)
    max_count = max(signature_counts.values(), default=0)
    summary_unique = summary.get("semantic_signature_unique_count") if summary else None
    summary_max = summary.get("semantic_signature_max_count") if summary else None
    summary_consistent: bool | None = None
    if summary_unique is not None or summary_max is not None:
        try:
            summary_consistent = (
                int(summary_unique) == unique_count and int(summary_max) == max_count
            )
        except (TypeError, ValueError):
            summary_consistent = False
        if summary_consistent is False:
            issues.add_many(
                "semantic_signature_summary_mismatch", derived_by_task.keys()
            )

    return {
        "expected_version": SEMANTIC_SIGNATURE_VERSION,
        "observed_versions": sorted(set(version_by_task.values())),
        "cap": cap,
        "cap_source": cap_source,
        "derived_rows": len(derived_by_task),
        "stored_rows": len(stored_by_task),
        "stored_match_rows": int(
            sum(stored_by_task.get(task) == value for task, value in derived_by_task.items())
        ),
        "unique_count": unique_count,
        "max_count": int(max_count),
        "within_cap": max_count <= cap,
        "value_counts_summary": {
            "unique_count": unique_count,
            "max_count": int(max_count),
            "cap": cap,
            "within_cap": max_count <= cap,
        },
        "over_cap_signature_count": len(over_cap),
        "over_cap_task_count": len(
            {task for tasks in over_cap.values() for task in tasks}
        ),
        "value_count_histogram": _counter_dict(count_histogram),
        "top_repeated_signatures": top_repeated,
        "reported_unique_count": summary_unique,
        "reported_max_count": summary_max,
        "summary_consistent": summary_consistent,
    }


def _audit_attempt_diversity(
    metadata_records: list[dict[str, Any]],
    summary: dict[str, Any] | None,
    issues: IssueCollector,
) -> dict[str, Any]:
    reported_version = None
    run_seed: int | None = None
    if summary is not None:
        reported_version = summary.get(
            "attempt_diversity_version",
            summary.get("source_attempt_diversity_version"),
        )
        raw_seed = summary.get("seed", summary.get("source_seed"))
        try:
            run_seed = int(raw_seed)
        except (TypeError, ValueError, OverflowError):
            run_seed = None

    observed_versions: set[str] = set()
    version_match_rows = 0
    anchor_hashes_by_value: dict[str, list[int]] = defaultdict(list)
    mutation_hashes_by_value: dict[str, list[int]] = defaultdict(list)
    anchor_hash_valid_rows = 0
    mutation_hash_valid_rows = 0
    anchor_hash_recomputed_rows = 0
    mutation_hash_recomputed_rows = 0
    feedback_counts: list[int] = []
    forbidden_signature_counts: list[int] = []
    forbidden_card_counts: list[int] = []

    def parse_int(row: Mapping[str, Any], field: str) -> int | None:
        value = row.get(field)
        if isinstance(value, bool) or value is None or pd.isna(value):
            return None
        try:
            parsed = int(value)
            if float(value) != float(parsed):
                return None
        except (TypeError, ValueError, OverflowError):
            return None
        return parsed

    for position, row in enumerate(metadata_records):
        task = _task_index(row, position)
        version = row.get("attempt_diversity_version")
        if not isinstance(version, str) or not version:
            issues.add("missing_attempt_diversity_version", task)
        else:
            observed_versions.add(version)
            if version == ATTEMPT_DIVERSITY_VERSION:
                version_match_rows += 1
            else:
                issues.add("attempt_diversity_version_mismatch", task)

        anchor_hash = row.get("anchor_diversity_nonce_sha256")
        mutation_hash = row.get("mutation_diversity_nonce_sha256")
        if not isinstance(anchor_hash, str) or SHA256_RE.fullmatch(anchor_hash) is None:
            issues.add("invalid_anchor_diversity_nonce_sha256", task)
        else:
            anchor_hash_valid_rows += 1
            anchor_hashes_by_value[anchor_hash].append(task)
        if (
            not isinstance(mutation_hash, str)
            or SHA256_RE.fullmatch(mutation_hash) is None
        ):
            issues.add("invalid_mutation_diversity_nonce_sha256", task)
        else:
            mutation_hash_valid_rows += 1
            mutation_hashes_by_value[mutation_hash].append(task)

        coordinates = {
            field: parse_int(row, field)
            for field in (
                "source_style_id",
                "id2",
                "task_seed_offset",
                "task_retry_round",
                "selection_attempt",
                "anchor_attempt",
                "mutation_attempt",
                "pair_attempts_config",
                "anchor_attempts_config",
                "mutation_attempts_config",
            )
        }
        if run_seed is None:
            issues.add("attempt_diversity_seed_missing", task)
        elif any(value is None for value in coordinates.values()):
            issues.add("invalid_attempt_diversity_coordinates", task)
        else:
            source_style_id = int(coordinates["source_style_id"])
            selection_attempt = int(coordinates["selection_attempt"])
            anchor_attempt = int(coordinates["anchor_attempt"])
            mutation_attempt = int(coordinates["mutation_attempt"])
            pair_attempts = int(coordinates["pair_attempts_config"])
            anchor_attempts = int(coordinates["anchor_attempts_config"])
            mutation_attempts = int(coordinates["mutation_attempts_config"])
            task_retry_round = int(coordinates["task_retry_round"])
            if (
                task_retry_round < 0
                or not 1 <= selection_attempt <= pair_attempts
                or not 1 <= anchor_attempt <= anchor_attempts
                or not 1 <= mutation_attempt <= mutation_attempts
            ):
                issues.add("invalid_attempt_diversity_coordinates", task)
            else:
                pair_task = PairGenerationTask(
                    task_index=task,
                    mutated_id=int(coordinates["id2"]),
                    seed=int(
                        stable_hash64(run_seed, source_style_id) % (2**31 - 1)
                    ),
                    anchor={"id": source_style_id},
                )
                common = {
                    "task_seed_offset": int(coordinates["task_seed_offset"]),
                    "task_retry_round": task_retry_round,
                    "selection_attempt": selection_attempt,
                }
                expected_anchor = hashlib.sha256(
                    _attempt_diversity_nonce(
                        pair_task,
                        **common,
                        stage="anchor",
                        stage_attempt=anchor_attempt,
                    ).encode("utf-8")
                ).hexdigest()
                expected_mutation = hashlib.sha256(
                    _attempt_diversity_nonce(
                        pair_task,
                        **common,
                        stage="mutation",
                        stage_attempt=mutation_attempt,
                    ).encode("utf-8")
                ).hexdigest()
                if anchor_hash == expected_anchor:
                    anchor_hash_recomputed_rows += 1
                else:
                    issues.add("anchor_diversity_nonce_mismatch", task)
                if mutation_hash == expected_mutation:
                    mutation_hash_recomputed_rows += 1
                else:
                    issues.add("mutation_diversity_nonce_mismatch", task)

        for field, destination, upper_bound in (
            (
                "global_rejection_feedback_count",
                feedback_counts,
                MAX_GLOBAL_REJECTION_PROMPT_ITEMS,
            ),
            (
                "forbidden_semantic_signature_count",
                forbidden_signature_counts,
                None,
            ),
            ("forbidden_card_key_count", forbidden_card_counts, None),
        ):
            parsed = parse_int(row, field)
            if (
                parsed is None
                or parsed < 0
                or (upper_bound is not None and parsed > upper_bound)
            ):
                issues.add(f"invalid_{field}", task)
            else:
                destination.append(parsed)

    for hashes, issue in (
        (anchor_hashes_by_value, "duplicate_anchor_diversity_nonce"),
        (mutation_hashes_by_value, "duplicate_mutation_diversity_nonce"),
    ):
        for tasks in hashes.values():
            if len(tasks) > 1:
                issues.add_many(issue, tasks)

    row_count = len(metadata_records)
    summary_consistent = reported_version == ATTEMPT_DIVERSITY_VERSION
    if not summary_consistent:
        issues.add_many(
            "attempt_diversity_summary_mismatch",
            (_task_index(row, index) for index, row in enumerate(metadata_records)),
        )

    def distribution(values: list[int]) -> dict[str, int]:
        return _counter_dict(Counter(str(value) for value in values))

    return {
        "expected_version": ATTEMPT_DIVERSITY_VERSION,
        "reported_version": reported_version,
        "observed_versions": sorted(observed_versions),
        "summary_consistent": summary_consistent,
        "metadata_rows": row_count,
        "version_match_rows": version_match_rows,
        "anchor_nonce_hash_valid_rows": anchor_hash_valid_rows,
        "anchor_nonce_hash_recomputed_rows": anchor_hash_recomputed_rows,
        "anchor_nonce_hash_unique_count": len(anchor_hashes_by_value),
        "mutation_nonce_hash_valid_rows": mutation_hash_valid_rows,
        "mutation_nonce_hash_recomputed_rows": mutation_hash_recomputed_rows,
        "mutation_nonce_hash_unique_count": len(mutation_hashes_by_value),
        "global_rejection_feedback_count_distribution": distribution(
            feedback_counts
        ),
        "forbidden_semantic_signature_count_distribution": distribution(
            forbidden_signature_counts
        ),
        "forbidden_card_key_count_distribution": distribution(
            forbidden_card_counts
        ),
        "protocol_consistent": (
            summary_consistent
            and observed_versions == {ATTEMPT_DIVERSITY_VERSION}
            and version_match_rows == row_count
            and anchor_hash_recomputed_rows == row_count
            and mutation_hash_recomputed_rows == row_count
            and len(anchor_hashes_by_value) == row_count
            and len(mutation_hashes_by_value) == row_count
            and len(feedback_counts) == row_count
            and len(forbidden_signature_counts) == row_count
            and len(forbidden_card_counts) == row_count
        ),
    }


def _audit_rules(
    metadata_records: list[dict[str, Any]],
    summary: dict[str, Any] | None,
    issues: IssueCollector,
) -> dict[str, Any]:
    actual_usage: Counter[str] = Counter()
    scheduled_usage: Counter[str] = Counter()
    actual_primary_usage: Counter[str] = Counter()
    scheduled_primary_usage: Counter[str] = Counter()
    scheduled_profile_usage: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    product_type_counts: Counter[str] = Counter()
    category_product_type_counts: Counter[str] = Counter()
    actual_two_rule_tasks: list[int] = []
    scheduled_two_rule_tasks: list[int] = []
    aligned_rows = 0
    application_aligned_rows = 0

    for position, row in enumerate(metadata_records):
        task = _task_index(row, position)
        category = str(row.get("category") or "")
        product_type = str(row.get("product_type") or "")
        category_counts[category] += 1
        product_type_counts[product_type] += 1
        category_product_type_counts[f"{category}\u0000{product_type}"] += 1

        actual = _json_list(row.get("rule_ids"))
        if actual is None or not actual:
            issues.add("invalid_actual_rule_ids", task)
            actual_ids: list[str] = []
        else:
            actual_ids = [str(value) for value in actual]
            actual_usage.update(actual_ids)
            actual_primary_usage[actual_ids[0]] += 1
            if len(actual_ids) == 2:
                actual_two_rule_tasks.append(task)
            elif len(actual_ids) != 1:
                issues.add("invalid_actual_rule_count", task)

        scheduled = _json_list(row.get("scheduled_rule_ids"))
        profiles = _json_list(row.get("scheduled_rule_profile_ids"))
        if scheduled is None or not scheduled:
            issues.add("invalid_scheduled_rule_ids", task)
            scheduled_ids: list[str] = []
        else:
            scheduled_ids = [str(value) for value in scheduled]
            scheduled_usage.update(scheduled_ids)
            scheduled_primary_usage[scheduled_ids[0]] += 1
            if len(scheduled_ids) == 2:
                scheduled_two_rule_tasks.append(task)
            elif len(scheduled_ids) != 1:
                issues.add("invalid_scheduled_rule_count", task)
        if profiles is None or len(profiles) != len(scheduled_ids):
            issues.add("invalid_scheduled_profile_ids", task)
        else:
            scheduled_profile_usage.update(str(value) for value in profiles)

        if actual_ids == scheduled_ids and actual_ids:
            aligned_rows += 1
        else:
            issues.add("scheduled_actual_rule_mismatch", task)
        applications = _json_list(row.get("applications_json"))
        application_rule_ids = (
            [
                str(application.get("generation_rule_id"))
                for application in applications
                if isinstance(application, dict)
                and application.get("generation_rule_id") is not None
            ]
            if applications is not None
            else []
        )
        if application_rule_ids == actual_ids and actual_ids:
            application_aligned_rows += 1
        else:
            issues.add("application_actual_rule_mismatch", task)
        try:
            recorded_rule_count = int(row.get("rule_count"))
        except (TypeError, ValueError, OverflowError):
            recorded_rule_count = -1
        if recorded_rule_count != len(actual_ids):
            issues.add("recorded_rule_count_mismatch", task)

    expected_rule_count = None
    if summary:
        catalog_summary = summary.get("rule_catalog_summary")
        if isinstance(catalog_summary, dict):
            raw_expected = catalog_summary.get("selectable_rules")
            try:
                expected_rule_count = int(raw_expected)
            except (TypeError, ValueError):
                expected_rule_count = None
    actual_ids = set(actual_usage)
    scheduled_ids = set(scheduled_usage)
    union_ids = actual_ids | scheduled_ids
    intersection_ids = actual_ids & scheduled_ids
    row_count = len(metadata_records)

    return {
        "actual": _usage_statistics(actual_usage),
        "scheduled": _usage_statistics(scheduled_usage),
        "actual_primary": _usage_statistics(actual_primary_usage),
        "scheduled_primary": _usage_statistics(scheduled_primary_usage),
        "scheduled_profiles": _usage_statistics(scheduled_profile_usage),
        "alignment": {
            "aligned_rows": aligned_rows,
            "mismatch_rows": row_count - aligned_rows,
            "application_aligned_rows": application_aligned_rows,
            "application_mismatch_rows": row_count - application_aligned_rows,
        },
        "coverage": {
            "expected_selectable_rule_count": expected_rule_count,
            "actual_unique_rule_count": len(actual_ids),
            "scheduled_unique_rule_count": len(scheduled_ids),
            "actual_fraction_of_expected": (
                len(actual_ids) / expected_rule_count
                if expected_rule_count and expected_rule_count > 0
                else None
            ),
            "scheduled_fraction_of_expected": (
                len(scheduled_ids) / expected_rule_count
                if expected_rule_count and expected_rule_count > 0
                else None
            ),
            "actual_scheduled_jaccard": (
                len(intersection_ids) / len(union_ids) if union_ids else None
            ),
            "actual_only_ids": sorted(actual_ids - scheduled_ids),
            "scheduled_only_ids": sorted(scheduled_ids - actual_ids),
        },
        "categories": {
            "unique_count": len(category_counts),
            "counts": _counter_dict(category_counts),
        },
        "product_types": {
            "unique_count": len(product_type_counts),
            "counts": _counter_dict(product_type_counts),
            "category_product_type_unique_count": len(category_product_type_counts),
            "category_product_type_counts": [
                {
                    "category": key.split("\u0000", 1)[0],
                    "product_type": key.split("\u0000", 1)[1],
                    "count": int(count),
                }
                for key, count in sorted(category_product_type_counts.items())
            ],
        },
        "two_rule": {
            "actual_tasks": len(actual_two_rule_tasks),
            "actual_fraction": (
                len(actual_two_rule_tasks) / row_count if row_count else 0.0
            ),
            "scheduled_tasks": len(scheduled_two_rule_tasks),
            "scheduled_fraction": (
                len(scheduled_two_rule_tasks) / row_count if row_count else 0.0
            ),
        },
    }


def _audit_pair_integrity(
    metadata: pd.DataFrame,
    base_items: pd.DataFrame,
    mutated_items: pd.DataFrame,
    pairs: pd.DataFrame,
    issues: IssueCollector,
) -> tuple[dict[str, Any], dict[int, int]]:
    _require_columns(metadata, {"task_index", "id1", "id2", "target"}, "metadata")
    _require_columns(base_items, {"id", "name", "attributes", "category"}, "base items")
    _require_columns(
        mutated_items, {"id", "name", "attributes", "category"}, "mutated items"
    )
    _require_columns(pairs, {"id1", "id2", "target"}, "pairs")

    records = metadata.to_dict("records")
    id_to_task: dict[int, int] = {}
    metadata_contract: Counter[tuple[int, int, int]] = Counter()
    for position, row in enumerate(records):
        task = _task_index(row, position)
        try:
            id1, id2, target = int(row["id1"]), int(row["id2"]), int(row["target"])
        except (TypeError, ValueError, OverflowError):
            issues.add("invalid_metadata_pair_contract", task)
            continue
        id_to_task[id1] = task
        id_to_task[id2] = task
        metadata_contract[(id1, id2, target)] += 1

    pair_contract: Counter[tuple[int, int, int]] = Counter()
    for row in pairs.to_dict("records"):
        try:
            pair_contract[(int(row["id1"]), int(row["id2"]), int(row["target"]))] += 1
        except (TypeError, ValueError, OverflowError):
            continue
    for contract, count in (metadata_contract - pair_contract).items():
        task = id_to_task.get(contract[0], id_to_task.get(contract[1], -1))
        for _ in range(count):
            issues.add("metadata_pair_alignment_mismatch", task)
    for contract, count in (pair_contract - metadata_contract).items():
        task = id_to_task.get(contract[0], id_to_task.get(contract[1], -1))
        for _ in range(count):
            issues.add("metadata_pair_alignment_mismatch", task)

    duplicate_task_rows = metadata[metadata["task_index"].duplicated(keep=False)]
    issues.add_many("duplicate_task_index", duplicate_task_rows["task_index"].tolist())
    target_counts: Counter[str] = Counter()
    for row in pairs.to_dict("records"):
        try:
            target = int(row["target"])
            id1, id2 = int(row["id1"]), int(row["id2"])
        except (TypeError, ValueError, OverflowError):
            target_counts["invalid"] += 1
            continue
        target_counts[str(target)] += 1
        if target not in {0, 1}:
            issues.add(
                "invalid_pair_target",
                id_to_task.get(id1, id_to_task.get(id2, -1)),
            )

    return (
        {
            "metadata_rows": int(len(metadata)),
            "base_item_rows": int(len(base_items)),
            "mutated_item_rows": int(len(mutated_items)),
            "pair_rows": int(len(pairs)),
            "unique_task_indices": int(metadata["task_index"].nunique()),
            "target_counts": _counter_dict(target_counts),
            "metadata_pair_contract_equal": metadata_contract == pair_contract,
        },
        id_to_task,
    )


def _audit_duplicates_and_heuristics(
    base_items: pd.DataFrame,
    mutated_items: pd.DataFrame,
    pairs: pd.DataFrame,
    metadata_records: list[dict[str, Any]],
    id_to_task: dict[int, int],
    issues: IssueCollector,
) -> tuple[dict[str, Any], dict[str, Any]]:
    combined = pd.concat(
        [
            base_items.assign(_side="base"),
            mutated_items.assign(_side="mutated"),
        ],
        ignore_index=True,
    )
    card_key_by_id: dict[int, str] = {}
    tasks_by_card: dict[str, list[int]] = defaultdict(list)
    identifiers: set[int] = set()
    numeric_models: set[int] = set()

    for row in combined.to_dict("records"):
        try:
            item_id = int(row["id"])
        except (TypeError, ValueError, OverflowError):
            continue
        task = id_to_task.get(item_id, -1)
        attributes = _safe_attributes(row.get("attributes"))
        if attributes is None:
            issues.add("invalid_item_attributes", task)
            continue
        card_key = _canonical_card(row)
        if card_key is not None:
            card_key_by_id[item_id] = card_key
            tasks_by_card[card_key].append(task)
        if IDENTIFIER_TITLE_RE.search(str(row.get("name") or "")) or any(
            IDENTIFIER_KEY_RE.search(str(key)) for key in attributes
        ):
            identifiers.add(task)
        if any(
            MODEL_KEY_RE.search(str(key)) and _is_numeric_only(value)
            for key, value in attributes.items()
        ):
            numeric_models.add(task)

    duplicate_card_groups = {
        key: tasks for key, tasks in tasks_by_card.items() if len(tasks) > 1
    }
    for tasks in duplicate_card_groups.values():
        issues.add_many("duplicate_full_card", tasks)

    tasks_by_pair_card: dict[str, list[int]] = defaultdict(list)
    tasks_by_unordered_id_pair: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row in pairs.to_dict("records"):
        try:
            id1, id2 = int(row["id1"]), int(row["id2"])
        except (TypeError, ValueError, OverflowError):
            continue
        task = id_to_task.get(id1, id_to_task.get(id2, -1))
        tasks_by_unordered_id_pair[tuple(sorted((id1, id2)))].append(task)
        left, right = card_key_by_id.get(id1), card_key_by_id.get(id2)
        if left is not None and right is not None:
            tasks_by_pair_card[canonical_json_dumps(sorted((left, right)))].append(task)
    duplicate_pair_card_groups = {
        key: tasks for key, tasks in tasks_by_pair_card.items() if len(tasks) > 1
    }
    duplicate_id_pair_groups = {
        key: tasks
        for key, tasks in tasks_by_unordered_id_pair.items()
        if len(tasks) > 1
    }
    for tasks in duplicate_pair_card_groups.values():
        issues.add_many("duplicate_order_insensitive_card_pair", tasks)
    for tasks in duplicate_id_pair_groups.values():
        issues.add_many("duplicate_unordered_id_pair", tasks)

    material_synonyms: set[int] = set()
    application_numeric_models: set[int] = set()
    for position, row in enumerate(metadata_records):
        task = _task_index(row, position)
        applications = _json_list(row.get("applications_json"))
        if applications is None:
            continue
        for application in applications:
            if not isinstance(application, dict):
                continue
            concept = normalize_text(application.get("concept"))
            key = str(application.get("attribute_key") or "")
            old = application.get("original_value")
            new = application.get("new_value")
            if concept in MATERIAL_CONCEPTS and _material_values_equivalent(old, new):
                material_synonyms.add(task)
            if (
                concept in MODEL_CONCEPTS or MODEL_KEY_RE.search(key)
            ) and (_is_numeric_only(old) or _is_numeric_only(new)):
                application_numeric_models.add(task)
    numeric_models |= application_numeric_models

    issues.add_many("sku_or_article", identifiers)
    issues.add_many("numeric_only_model", numeric_models)
    issues.add_many("material_synonym_mutation", material_synonyms)

    duplicate_report = {
        "card_rows": int(len(combined)),
        "unique_full_cards": len(tasks_by_card),
        "duplicate_full_card_groups": len(duplicate_card_groups),
        "duplicate_full_card_affected_tasks": len(
            {task for tasks in duplicate_card_groups.values() for task in tasks}
        ),
        "unique_order_insensitive_card_pairs": len(tasks_by_pair_card),
        "duplicate_order_insensitive_card_pair_groups": len(
            duplicate_pair_card_groups
        ),
        "duplicate_order_insensitive_card_pair_affected_tasks": len(
            {task for tasks in duplicate_pair_card_groups.values() for task in tasks}
        ),
        "duplicate_unordered_id_pair_groups": len(duplicate_id_pair_groups),
    }
    heuristic_report = {
        "sku_or_article_affected_tasks": len(identifiers),
        "numeric_only_model_affected_tasks": len(numeric_models),
        "material_synonym_mutation_affected_tasks": len(material_synonyms),
        "material_synonym_groups": [sorted(group) for group in MATERIAL_EQUIVALENCE_GROUPS],
    }
    return duplicate_report, heuristic_report


def audit_source(
    source_dir: Path,
    *,
    reference_items: Path | None = None,
    semantic_signature_cap: int | None = None,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> dict[str, Any]:
    if sample_limit < 1:
        raise ValueError("sample limit must be positive")
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source_dir}")
    paths = {name: source_dir / filename for name, filename in REQUIRED_SOURCE_FILES.items()}
    missing_files = [str(path) for path in paths.values() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"source directory is incomplete: {missing_files}")

    summary = _read_json(source_dir / "summary.json")
    cap, cap_source = _effective_signature_cap(summary, semantic_signature_cap)
    metadata = pd.read_parquet(paths["metadata"])
    base_items = pd.read_parquet(paths["base_items"])
    mutated_items = pd.read_parquet(paths["mutated_items"])
    pairs = pd.read_parquet(paths["pairs"])
    _require_columns(metadata, {"task_index"}, "metadata")
    metadata_records = metadata.sort_values("task_index", kind="stable").to_dict("records")
    issues = IssueCollector(sample_limit)

    integrity, id_to_task = _audit_pair_integrity(
        metadata, base_items, mutated_items, pairs, issues
    )
    semantic = _audit_semantic_signatures(
        metadata_records, summary, cap, cap_source, issues
    )
    attempt_diversity = _audit_attempt_diversity(
        metadata_records, summary, issues
    )
    rules = _audit_rules(metadata_records, summary, issues)
    duplicates, heuristics = _audit_duplicates_and_heuristics(
        base_items,
        mutated_items,
        pairs,
        metadata_records,
        id_to_task,
        issues,
    )

    synthetic_combined = pd.concat([base_items, mutated_items], ignore_index=True)
    synthetic_complexity = {
        "base": _complexity(base_items),
        "mutated": _complexity(mutated_items),
        "combined": _complexity(synthetic_combined),
    }
    reference_report: dict[str, Any] | None = None
    reference_matching_categories: dict[str, Any] | None = None
    reference_path_text: str | None = None
    if reference_items is not None:
        reference_path = reference_items.resolve()
        if not reference_path.exists():
            raise FileNotFoundError(f"reference items do not exist: {reference_path}")
        reference = pd.read_parquet(
            reference_path, columns=["name", "attributes", "category"]
        )
        reference_report = _complexity(reference)
        synthetic_categories = set(metadata["category"].astype(str))
        matching = reference[reference["category"].astype(str).isin(synthetic_categories)]
        reference_matching_categories = _complexity(matching)
        reference_path_text = str(reference_path)

    issue_report = issues.report()
    affected_tasks = sorted(
        {
            task
            for issue in issue_report.values()
            for task in issue["sample_task_indices"]
        }
    )
    return {
        "version": AUDIT_VERSION,
        "source_dir": str(source_dir),
        "reference_items": reference_path_text,
        "summary_present": summary is not None,
        "integrity": integrity,
        "semantic_signatures": semantic,
        "attempt_diversity": attempt_diversity,
        "rules": rules,
        "duplicates": duplicates,
        "heuristics": heuristics,
        "complexity": {
            "synthetic": synthetic_complexity,
            "human_reference_all": reference_report,
            "human_reference_matching_synthetic_categories": (
                reference_matching_categories
            ),
            "synthetic_vs_human_matching_categories": _complexity_comparison(
                synthetic_complexity["combined"], reference_matching_categories
            ),
        },
        "issue_type_count": len(issue_report),
        "sample_affected_task_indices": affected_tasks[:sample_limit],
        "issues": issue_report,
        "passed": not issue_report,
    }


def _default_output(source_dir: Path) -> Path:
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", source_dir.name).strip("._")
    if not safe_name:
        safe_name = "generated_pairs"
    return DEFAULT_OUTPUT_DIR / f"{safe_name}.json"


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _atomic_json(report: dict[str, Any], output: Path) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source_dir = args.source_dir.resolve()
    output = (args.output or _default_output(source_dir)).resolve()
    if _is_within(output, source_dir):
        raise ValueError("audit output must be outside --source-dir")
    report = audit_source(
        source_dir,
        reference_items=args.reference_items,
        semantic_signature_cap=args.semantic_signature_cap,
        sample_limit=args.sample_limit,
    )
    _atomic_json(report, output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
