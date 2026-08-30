#!/usr/bin/env python3
"""Validate, freeze, publish and train on mixed semantic-rule pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from freeze_generated_pair_dataset import freeze
from item_pipeline.normalization import normalize_text, parse_attributes
from item_pipeline.pair_generate import (
    ATTEMPT_DIVERSITY_VERSION,
    SEMANTIC_SIGNATURE_VERSION,
)
from item_pipeline.pair_rules import MutationRule, load_mutation_rules
from item_pipeline.pair_validation import validate_pair_dataset
from item_pipeline.rule_schedule import SCHEDULE_VERSION, build_balanced_rule_schedule

import launch_statistical_rule_kaggle_experiment as strict_checks
import push_mixed_generation_rule_pairs_dataset as mixed_upload
import run_kaggle_notebook as kaggle


DEFAULT_RAW_DIR = ROOT / (
    "item_pipeline/artifacts/semantic_rule_pairs_transition_positive_10k_raw"
)
DEFAULT_FROZEN_DIR = ROOT / (
    "item_pipeline/artifacts/semantic_rule_pairs_transition_positive_10k"
)
DEFAULT_RULE_CATALOG = ROOT / (
    "configs/generation_rule_catalog_statistical_v1/"
    "semantic_all_pairs_cross_split_p80_support5_scoped_compact_transition_positive_v4.json"
)
DEFAULT_RULE_CATALOG_VERSION = (
    "semantic_all_pairs_cross_split_p80_support5_scoped_compact_transition_positive_v4"
)
DEFAULT_RULE_CATALOG_SHA256 = (
    "a0aff8fa850c6b568c867c704b04be28a4f7c47a20dbb79723925c9dc2fd8245"
)
DEFAULT_RULE_CATALOG_MANIFEST_SHA256 = (
    "bcfe10919171ecbeadd99a0283e1bec52b472f091f61ef5095bfcf86975b8801"
)
DEFAULT_POSITIVE_RULE_CAPACITIES = {
    "gen_sem_all_2bd42ae67368b6da139a": 4,
    "gen_sem_all_fc1bf7245474d5979bbc": 12,
    "gen_sem_all_a29a2640f81133199b22": 14,
    "gen_sem_all_1fd8ee0362b4a69694eb": 16,
}
DEFAULT_POSITIVE_RULE_CONCEPTS = {
    "gen_sem_all_2bd42ae67368b6da139a": "age_rating",
    "gen_sem_all_fc1bf7245474d5979bbc": "game_type",
    "gen_sem_all_a29a2640f81133199b22": "target_audience",
    "gen_sem_all_1fd8ee0362b4a69694eb": "target_audience_age",
}
DEFAULT_REQUIRED_POSITIVE_CONTEXT_KEYS = ("Бренд", "Название игры")
DEFAULT_POSITIVE_TRANSITION_COUNT = 23
DEFAULT_POSITIVE_TRANSITION_REPETITIONS = 2
DEFAULT_API_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_STYLE_DONOR_POOL_VERSION = "generated_style_donors_virtual_id_copies_v1"
DEFAULT_SEMANTIC_SIGNATURE_LIMIT = 2
DEFAULT_PROMPT = ROOT / "item_pipeline/prompts/mutate_item_by_rules.md"
DEFAULT_NOTEBOOK = ROOT / (
    "notebooks/minilm_5ep_team_ablation/"
    "minilm_5ep_semantic_transition_positive_10k_v1_2xt4.ipynb"
)
DEFAULT_DATASET_SLUG = mixed_upload.DEFAULT_DATASET_SLUG
DEFAULT_ARTIFACT_TAG = "semantic-transition-positive-10k-v1"
DEFAULT_EXPERIMENT = "minilm_5ep_semantic_transition_positive_10k_v1"
DEFAULT_KERNEL_SLUG = (
    "product-matching-minilm-5ep-semantic-transition-positive-10k-v1"
)
DEFAULT_TITLE = "MiniLM 5ep: semantic transition-positive 10k v1"
DEFAULT_LABEL_SOURCE = "openrouter_semantic_transition_rule_generation_v1"
FROZEN_OOD_CATEGORIES = {"Одежда", "Бытовая техника"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--frozen-dir", type=Path, default=DEFAULT_FROZEN_DIR)
    parser.add_argument("--pair-count", type=int, default=mixed_upload.DEFAULT_PAIR_COUNT)
    parser.add_argument(
        "--expected-target0", type=int, default=mixed_upload.DEFAULT_TARGET0
    )
    parser.add_argument(
        "--expected-target1", type=int, default=mixed_upload.DEFAULT_TARGET1
    )
    parser.add_argument(
        "--expected-rule-catalog", type=Path, default=DEFAULT_RULE_CATALOG
    )
    parser.add_argument(
        "--expected-source-items",
        type=Path,
        help=(
            "optional extra pin for the style-donor pool; when omitted, resolve "
            "and verify the path/SHA already signed by the raw run summary"
        ),
    )
    parser.add_argument("--expected-prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--expected-model")
    parser.add_argument("--expected-api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--expected-reasoning-effort", default="low")
    parser.add_argument("--expected-tier", action="append")
    parser.add_argument("--dataset-slug", default=DEFAULT_DATASET_SLUG)
    parser.add_argument("--artifact-tag", default=DEFAULT_ARTIFACT_TAG)
    parser.add_argument("--experiment-label", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--kernel-slug", default=DEFAULT_KERNEL_SLUG)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--label-source", default=DEFAULT_LABEL_SOURCE)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument(
        "--dry-run-only",
        action="store_true",
        help="validate and stage locally without mutating Kaggle",
    )
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_prompt(path: Path) -> str:
    return hashlib.sha256(
        path.read_text(encoding="utf-8").strip().encode("utf-8")
    ).hexdigest()


def read_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {description}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must be a JSON object: {path}")
    return value


def verify_style_donor_pool(
    source_items_path: Path,
    source_sha: str,
    source_items: pd.DataFrame,
) -> dict[str, Any]:
    required_columns = ["id", "name", "attributes", "category"]
    if missing := set(required_columns) - set(source_items):
        raise RuntimeError(f"style-donor pool lacks columns: {sorted(missing)}")
    if source_items.empty or source_items["id"].duplicated().any():
        raise RuntimeError("style-donor pool must be non-empty with unique IDs")
    forbidden = set(source_items["category"].astype(str)) & FROZEN_OOD_CATEGORIES
    if forbidden:
        raise RuntimeError(
            f"style-donor pool contains frozen OOD categories: {sorted(forbidden)}"
        )
    result: dict[str, Any] = {
        "path": str(source_items_path),
        "sha256": source_sha,
        "rows": len(source_items),
        "unique_ids": int(source_items["id"].nunique()),
    }
    manifest_path = source_items_path.parent / "manifest.json"
    if not manifest_path.is_file():
        result["manifest"] = None
        return result
    manifest = read_json(manifest_path, "style-donor pool manifest")
    output_value = str(manifest.get("output_path") or "").strip()
    output_path = Path(output_value) if output_value else None
    if output_path is not None and not output_path.is_absolute():
        output_path = ROOT / output_path
    if (
        output_path is None
        or output_path.resolve() != source_items_path
        or manifest.get("output_sha256") != source_sha
        or int(manifest.get("rows", -1)) != len(source_items)
        or int(manifest.get("unique_ids", -1)) != len(source_items)
    ):
        raise RuntimeError("style-donor pool manifest differs from the pinned input")
    observed_categories = {
        str(key): int(value)
        for key, value in source_items["category"].value_counts().sort_index().items()
    }
    manifest_categories = {
        str(key): int(value)
        for key, value in (manifest.get("category_counts") or {}).items()
    }
    if manifest_categories != observed_categories:
        raise RuntimeError("style-donor pool manifest category counts differ")
    source_value = str(manifest.get("source_path") or "").strip()
    source_path = Path(source_value) if source_value else None
    if source_path is not None and not source_path.is_absolute():
        source_path = ROOT / source_path
    if source_path is None or not source_path.resolve().is_file():
        raise RuntimeError("style-donor pool manifest has no available source")
    source_path = source_path.resolve()
    if sha256_file(source_path) != manifest.get("source_sha256"):
        raise RuntimeError("style-donor pool source SHA differs from its manifest")
    source_rows = len(pd.read_parquet(source_path, columns=["id"]))
    if int(manifest.get("source_rows", -1)) != source_rows:
        raise RuntimeError("style-donor pool source row count differs")
    copies = int(manifest.get("copies", -1))
    source_frame = pd.read_parquet(source_path)
    if source_frame.columns.tolist() != required_columns:
        raise RuntimeError("style-donor source columns differ from the builder contract")
    if source_frame.empty or source_frame["id"].duplicated().any():
        raise RuntimeError("style-donor source must have unique IDs")
    minimum_id = int(source_frame["id"].min())
    maximum_id = int(source_frame["id"].max())
    id_span = maximum_id - minimum_id + 1
    if (
        copies != 2
        or source_rows * copies != len(source_items)
        or int(manifest.get("id_offset_span", -1)) != id_span
    ):
        raise RuntimeError("style-donor virtual-copy geometry differs")
    expected_copies: list[pd.DataFrame] = []
    for copy_index in range(copies):
        frame = source_frame.copy()
        frame["id"] = frame["id"].astype("int64") - copy_index * id_span
        expected_copies.append(frame)
    expected_pool = pd.concat(expected_copies, ignore_index=True)
    observed_pool = source_items[required_columns].reset_index(drop=True)
    if not observed_pool.equals(expected_pool):
        raise RuntimeError("style-donor pool is not an exact x2 virtual-ID copy")
    result["manifest"] = {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "version": manifest.get("version"),
        "copies": manifest.get("copies"),
        "id_offset_span": id_span,
        "source_path": str(source_path),
        "source_sha256": manifest.get("source_sha256"),
        "source_rows": source_rows,
        "virtual_copy_content_verified": True,
    }
    return result


def run(command: list[str]) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def target_counts(frame: pd.DataFrame) -> dict[str, int]:
    return mixed_upload.observed_target_counts(frame)


def schedule_sha256(metadata: pd.DataFrame) -> str:
    payload: list[dict[str, Any]] = []
    ordered = metadata.sort_values("task_index", kind="stable")
    for row in ordered.to_dict("records"):
        try:
            profiles = json.loads(str(row["scheduled_rule_profile_ids"]))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError("metadata has invalid scheduled profile JSON") from error
        if not isinstance(profiles, list) or not all(
            isinstance(value, str) and value for value in profiles
        ):
            raise RuntimeError("metadata scheduled profiles must be non-empty strings")
        payload.append(
            {
                "task_index": int(row["task_index"]),
                "donor_id": int(row["source_style_id"]),
                "category": str(row["category"]),
                "profiles": profiles,
            }
        )
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def rule_index(rules: list[MutationRule]) -> dict[str, MutationRule]:
    result = {rule.generation_rule_id: rule for rule in rules}
    if len(result) != len(rules):
        raise RuntimeError("mixed rule catalog contains duplicate generation IDs")
    return result


def transition_signature(left: Any, right: Any) -> tuple[str, str]:
    """Canonicalize one exact transition without retaining its direction."""

    return tuple(sorted((normalize_text(left), normalize_text(right))))


def verify_transition_catalog(
    rule_catalog: Path,
    manifest: dict[str, Any],
    catalog_sha: str,
) -> dict[str, Any]:
    """Pin the reviewed v4 catalog, its 23 transitions and context contract."""

    if manifest.get("catalog_version") != DEFAULT_RULE_CATALOG_VERSION:
        raise RuntimeError(
            "mixed rule catalog is not the reviewed transition-positive v4"
        )
    if manifest.get("output_sha256") != catalog_sha:
        raise RuntimeError("mixed rule catalog manifest does not pin its output")
    if catalog_sha != DEFAULT_RULE_CATALOG_SHA256:
        raise RuntimeError("pinned transition-positive v4 catalog SHA differs")
    manifest_counts = {
        str(key): int(value)
        for key, value in (manifest.get("label_counts") or {}).items()
    }
    if manifest_counts != {"0": 3_674, "1": 4}:
        raise RuntimeError(
            f"transition-positive v4 catalog label counts differ: {manifest_counts}"
        )
    selection = manifest.get("selection") or {}
    if selection.get("positive_selection") != (
        "manual_exact_unordered_transition_allowlist"
    ):
        raise RuntimeError("positive transition catalog is not manually allowlisted")
    positive_ids = {str(value) for value in selection.get("positive_allowlist") or []}
    if positive_ids != set(DEFAULT_POSITIVE_RULE_CAPACITIES):
        raise RuntimeError(
            "transition-positive v4 allowlist differs from the four pinned rules"
        )
    if (
        int(manifest.get("transition_rule_count", -1)) != 4
        or int(manifest.get("transition_count", -1))
        != DEFAULT_POSITIVE_TRANSITION_COUNT
        or int(manifest.get("transition_capacity", -1))
        != sum(DEFAULT_POSITIVE_RULE_CAPACITIES.values())
        or int(selection.get("recommended_target1_count", -1))
        != sum(DEFAULT_POSITIVE_RULE_CAPACITIES.values())
        or float(selection.get("recommended_label_one_fraction", -1)) != 0.0046
        or float(selection.get("recommended_two_rule_fraction", -1)) != 0.0
    ):
        raise RuntimeError("transition-positive v4 manifest capacity contract differs")
    provenance = manifest.get("transition_provenance") or {}
    if (
        provenance.get("semantics") != "each listed pair is exact and unordered"
        or provenance.get("cross_split_validation_scope") != "rule_profile_only"
        or provenance.get("exact_transitions_cross_split_validated") is not False
    ):
        raise RuntimeError("transition-positive v4 evidence scope differs")

    try:
        raw_rules = json.loads(rule_catalog.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid mixed rule catalog: {rule_catalog}") from error
    if not isinstance(raw_rules, list):
        raise RuntimeError("mixed rule catalog must be a JSON array")
    observed_counts = Counter(int(rule.get("label", -1)) for rule in raw_rules)
    if observed_counts != Counter({0: 3_674, 1: 4}):
        raise RuntimeError(
            f"transition-positive v4 rule counts differ: {observed_counts}"
        )
    catalog_positive_ids = {
        str(rule.get("generation_rule_id"))
        for rule in raw_rules
        if int(rule.get("label", -1)) == 1
    }
    if catalog_positive_ids != positive_ids:
        raise RuntimeError("catalog positive rules differ from the manual allowlist")

    transitions_by_rule: dict[str, frozenset[tuple[str, str]]] = {}
    context_by_rule: dict[str, tuple[str, ...]] = {}
    for raw_rule in raw_rules:
        if int(raw_rule.get("label", -1)) != 1:
            continue
        rule_id = str(raw_rule.get("generation_rule_id") or "")
        raw_transitions = raw_rule.get("allowed_value_transitions")
        if not isinstance(raw_transitions, list) or not raw_transitions:
            raise RuntimeError(f"positive rule has no transitions: {rule_id}")
        signatures: set[tuple[str, str]] = set()
        for transition in raw_transitions:
            if not isinstance(transition, list) or len(transition) != 2:
                raise RuntimeError(f"positive rule has malformed transition: {rule_id}")
            signature = transition_signature(transition[0], transition[1])
            if not all(signature) or signature[0] == signature[1]:
                raise RuntimeError(f"positive rule has empty/equal transition: {rule_id}")
            if signature in signatures:
                raise RuntimeError(
                    f"positive rule repeats an unordered transition: {rule_id}"
                )
            signatures.add(signature)
        expected_capacity = DEFAULT_POSITIVE_RULE_CAPACITIES[rule_id]
        required_context = tuple(raw_rule.get("required_anchor_context_keys") or [])
        allowed_context = tuple(raw_rule.get("allowed_anchor_context_keys") or [])
        if (
            raw_rule.get("concept") != DEFAULT_POSITIVE_RULE_CONCEPTS[rule_id]
            or raw_rule.get("manual_positive_allowlist") is not True
            or raw_rule.get("manual_transition_allowlist") is not True
            or raw_rule.get("manual_positive_review_version")
            != DEFAULT_RULE_CATALOG_VERSION
            or raw_rule.get("transition_positive_review_version")
            != DEFAULT_RULE_CATALOG_VERSION
            or raw_rule.get("allowed_value_transitions_unordered") is not True
            or raw_rule.get("value_transition_semantics")
            != "exact_unordered_pairs"
            or int(raw_rule.get("primary_task_safety_cap", -1))
            != expected_capacity
            or expected_capacity
            != len(signatures) * DEFAULT_POSITIVE_TRANSITION_REPETITIONS
            or required_context != DEFAULT_REQUIRED_POSITIVE_CONTEXT_KEYS
            or allowed_context != DEFAULT_REQUIRED_POSITIVE_CONTEXT_KEYS
        ):
            raise RuntimeError(
                f"positive rule transition/context contract differs: {rule_id}"
            )
        transitions_by_rule[rule_id] = frozenset(signatures)
        context_by_rule[rule_id] = required_context

    if sum(map(len, transitions_by_rule.values())) != (
        DEFAULT_POSITIVE_TRANSITION_COUNT
    ):
        raise RuntimeError("transition-positive catalog does not contain 23 slots")
    return {
        "positive_rule_ids": positive_ids,
        "positive_rule_counts": dict(DEFAULT_POSITIVE_RULE_CAPACITIES),
        "transitions_by_rule": transitions_by_rule,
        "required_context_keys_by_rule": context_by_rule,
        "transition_count": DEFAULT_POSITIVE_TRANSITION_COUNT,
        "transition_repetitions": DEFAULT_POSITIVE_TRANSITION_REPETITIONS,
    }


def _json_array(value: Any, *, field: str, task_index: int) -> list[Any]:
    try:
        parsed = value if isinstance(value, list) else json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"metadata has invalid {field} JSON at task {task_index}"
        ) from error
    if not isinstance(parsed, list):
        raise RuntimeError(f"metadata {field} is not an array at task {task_index}")
    return parsed


def verify_positive_transition_rows(
    metadata: pd.DataFrame,
    generated_items: pd.DataFrame,
    rules: list[MutationRule],
    transition_contract: dict[str, Any],
) -> dict[str, Any]:
    """Recompute the exact positive transition and preserved-context census."""

    required_metadata = {
        "id1",
        "id2",
        "target",
        "task_index",
        "scheduled_rule_ids",
        "applications_json",
    }
    if missing := required_metadata - set(metadata):
        raise RuntimeError(
            f"metadata lacks transition columns: {sorted(missing)}"
        )
    required_items = {"id", "attributes"}
    if missing := required_items - set(generated_items):
        raise RuntimeError(f"generated items lack transition columns: {sorted(missing)}")
    if generated_items["id"].duplicated().any():
        raise RuntimeError("generated transition items contain duplicate IDs")

    by_rule = rule_index(rules)
    item_by_id = generated_items.set_index("id")
    allowed_by_rule: dict[str, frozenset[tuple[str, str]]] = (
        transition_contract["transitions_by_rule"]
    )
    context_by_rule: dict[str, tuple[str, ...]] = transition_contract[
        "required_context_keys_by_rule"
    ]
    transition_counts: Counter[tuple[str, tuple[str, str]]] = Counter()
    positive_rows = metadata.loc[metadata["target"].astype(int).eq(1)]
    for row in positive_rows.to_dict("records"):
        task_index = int(row["task_index"])
        scheduled_ids = [
            str(value)
            for value in _json_array(
                row["scheduled_rule_ids"],
                field="scheduled_rule_ids",
                task_index=task_index,
            )
        ]
        applications = _json_array(
            row["applications_json"],
            field="applications_json",
            task_index=task_index,
        )
        if len(scheduled_ids) != 1 or len(applications) != 1:
            raise RuntimeError(
                f"positive transition task is not atomic at task {task_index}"
            )
        rule_id = scheduled_ids[0]
        rule = by_rule.get(rule_id)
        if rule is None or rule.label != 1 or rule_id not in allowed_by_rule:
            raise RuntimeError(
                f"positive task uses an unreviewed transition rule at task {task_index}"
            )
        application = applications[0]
        if not isinstance(application, dict):
            raise RuntimeError(
                f"positive transition application is not an object at task {task_index}"
            )
        if (
            str(application.get("generation_rule_id") or "") != rule_id
            or str(application.get("concept") or "") != rule.concept
            or str(application.get("attribute_key") or "") != rule.attribute_key
        ):
            raise RuntimeError(
                f"positive transition application differs from its rule at task {task_index}"
            )
        original = str(application.get("original_value") or "").strip()
        new = str(application.get("new_value") or "").strip()
        signature = transition_signature(original, new)
        if signature not in allowed_by_rule[rule_id]:
            raise RuntimeError(
                f"positive task uses a disallowed value transition at task {task_index}"
            )
        scoped_signature = (rule_id, signature)
        transition_counts[scoped_signature] += 1
        if transition_counts[scoped_signature] > int(
            transition_contract["transition_repetitions"]
        ):
            raise RuntimeError(
                f"positive transition exceeds its repetition cap at task {task_index}"
            )

        id1, id2 = int(row["id1"]), int(row["id2"])
        if id1 not in item_by_id.index or id2 not in item_by_id.index:
            raise RuntimeError(
                f"positive transition cards are missing at task {task_index}"
            )
        anchor_attributes = parse_attributes(item_by_id.loc[id1, "attributes"])
        mutated_attributes = parse_attributes(item_by_id.loc[id2, "attributes"])
        if (
            rule.attribute_key not in anchor_attributes
            or rule.attribute_key not in mutated_attributes
            or normalize_text(anchor_attributes[rule.attribute_key])
            != normalize_text(original)
            or normalize_text(mutated_attributes[rule.attribute_key])
            != normalize_text(new)
        ):
            raise RuntimeError(
                f"positive transition values differ from the cards at task {task_index}"
            )
        for context_key in context_by_rule[rule_id]:
            if (
                context_key not in anchor_attributes
                or context_key not in mutated_attributes
                or not normalize_text(anchor_attributes[context_key])
                or normalize_text(anchor_attributes[context_key])
                != normalize_text(mutated_attributes[context_key])
            ):
                raise RuntimeError(
                    "positive transition required context is missing or changed at "
                    f"task {task_index}: {context_key}"
                )

    expected_scoped_transitions = {
        (rule_id, signature)
        for rule_id, signatures in allowed_by_rule.items()
        for signature in signatures
    }
    expected_repetitions = int(transition_contract["transition_repetitions"])
    expected_counts = Counter(
        {key: expected_repetitions for key in expected_scoped_transitions}
    )
    if transition_counts != expected_counts:
        missing_or_wrong = {
            f"{rule_id}:{left}<->{right}": {
                "observed": int(transition_counts[(rule_id, (left, right))]),
                "expected": expected_repetitions,
            }
            for rule_id, (left, right) in sorted(expected_scoped_transitions)
            if transition_counts[(rule_id, (left, right))] != expected_repetitions
        }
        raise RuntimeError(
            "positive transition coverage/counts differ from the exact v4 quota: "
            f"{missing_or_wrong}"
        )
    usage = [
        {
            "rule_id": rule_id,
            "values": [left, right],
            "count": int(transition_counts[(rule_id, (left, right))]),
        }
        for rule_id, (left, right) in sorted(expected_scoped_transitions)
    ]
    return {
        "positive_rows": len(positive_rows),
        "transition_coverage": len(transition_counts),
        "expected_transition_coverage": len(expected_scoped_transitions),
        "maximum_transition_count": max(transition_counts.values(), default=0),
        "required_context_keys": list(DEFAULT_REQUIRED_POSITIVE_CONTEXT_KEYS),
        "usage": usage,
    }


def verify_schedule_rows(
    metadata: pd.DataFrame,
    pairs: pd.DataFrame,
    rules: list[MutationRule],
    source_items: pd.DataFrame,
    summary: dict[str, Any],
    expected_positive_rule_counts: dict[str, int] | None = None,
    generated_items: pd.DataFrame | None = None,
    transition_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    required = {
        "id1",
        "id2",
        "target",
        "task_index",
        "source_style_id",
        "category",
        "product_type",
        "rule_count",
        "rule_ids",
        "scheduled_rule_ids",
        "scheduled_rule_profile_ids",
        "rule_schedule_sha256",
        "rules_json",
        "applications_json",
        "model",
        "run_signature",
    }
    if missing := required - set(metadata):
        raise RuntimeError(f"metadata lacks mixed schedule columns: {sorted(missing)}")
    if len(metadata) != len(pairs):
        raise RuntimeError("pair and metadata row counts differ")
    task_indices = metadata["task_index"].astype(int).sort_values().tolist()
    if task_indices != list(range(len(metadata))):
        raise RuntimeError("mixed schedule task indices are not exactly 0..N-1")
    if metadata["source_style_id"].duplicated().any():
        raise RuntimeError("mixed schedule reuses a style donor")

    by_rule = rule_index(rules)
    source_category = source_items.set_index("id")["category"]
    pair_targets = {
        (int(row.id1), int(row.id2)): int(row.target)
        for row in pairs.itertuples(index=False)
    }
    label_rule_counts: Counter[int] = Counter()
    primary_rule_counts_by_label: dict[int, Counter[str]] = {
        0: Counter(),
        1: Counter(),
    }
    for row in metadata.to_dict("records"):
        pair_key = (int(row["id1"]), int(row["id2"]))
        if pair_key not in pair_targets or pair_targets[pair_key] != int(row["target"]):
            raise RuntimeError("metadata target does not match its pair")
        donor_id = int(row["source_style_id"])
        if donor_id not in source_category.index:
            raise RuntimeError(f"scheduled donor is absent: {donor_id}")
        if str(source_category.loc[donor_id]) != str(row["category"]):
            raise RuntimeError("scheduled donor category differs from metadata")
        try:
            actual_ids = json.loads(str(row["rule_ids"]))
            scheduled_ids = json.loads(str(row["scheduled_rule_ids"]))
            metadata_rules = json.loads(str(row["rules_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError("metadata has invalid rule JSON") from error
        if actual_ids != scheduled_ids or not isinstance(scheduled_ids, list):
            raise RuntimeError("actual rule IDs differ from the mixed schedule")
        if len(scheduled_ids) not in {1, 2}:
            raise RuntimeError("each generated pair must apply one or two rules")
        if not isinstance(metadata_rules, list):
            raise RuntimeError("rules_json must be an array")
        embedded_ids = [str(rule["generation_rule_id"]) for rule in metadata_rules]
        if embedded_ids != scheduled_ids:
            raise RuntimeError("embedded rules differ from scheduled rule IDs")
        selected: list[MutationRule] = []
        for rule_id in scheduled_ids:
            rule = by_rule.get(str(rule_id))
            if rule is None:
                raise RuntimeError(f"scheduled rule is absent from catalog: {rule_id}")
            selected.append(rule)
        labels = {rule.label for rule in selected}
        if labels != {int(row["target"])}:
            raise RuntimeError("scheduled rule labels differ from pair target")
        category = str(row["category"])
        if category in FROZEN_OOD_CATEGORIES:
            raise RuntimeError(
                f"mixed train schedule uses frozen OOD category: {category}"
            )
        if any(category not in rule.allowed_categories for rule in selected):
            raise RuntimeError("scheduled rule is outside its category scope")
        product_type = normalize_text(row["product_type"])
        for rule in selected:
            allowed = {normalize_text(value) for value in rule.allowed_product_types}
            if allowed and product_type not in allowed:
                raise RuntimeError("scheduled rule is outside its product-type scope")
        label_rule_counts[int(row["target"])] += len(selected)
        primary_rule_counts_by_label[int(row["target"])][str(scheduled_ids[0])] += 1

    observed_schedule_sha = schedule_sha256(metadata)
    reported_schedule_sha = str(summary.get("rule_schedule_sha256") or "")
    if observed_schedule_sha != reported_schedule_sha:
        raise RuntimeError(
            f"mixed schedule SHA differs: {observed_schedule_sha} != "
            f"{reported_schedule_sha}"
        )
    if set(metadata["rule_schedule_sha256"].astype(str)) != {
        reported_schedule_sha
    }:
        raise RuntimeError("metadata mixes rule schedule SHAs")
    observed_positive_counts = dict(sorted(primary_rule_counts_by_label[1].items()))
    if (
        expected_positive_rule_counts is not None
        and observed_positive_counts != expected_positive_rule_counts
    ):
        raise RuntimeError(
            "positive primary-rule usage differs from the exact balanced quota: "
            f"{observed_positive_counts} != {expected_positive_rule_counts}"
        )
    transition_checks = None
    if transition_contract is not None:
        if generated_items is None:
            raise RuntimeError("transition verification requires generated items")
        transition_checks = verify_positive_transition_rows(
            metadata,
            generated_items,
            rules,
            transition_contract,
        )
    return {
        "rule_schedule_sha256": observed_schedule_sha,
        "primary_rule_coverage": int(
            metadata["scheduled_rule_ids"].map(
                lambda raw: json.loads(str(raw))[0]
            ).nunique()
        ),
        "label_rule_applications": {
            str(label): int(label_rule_counts[label]) for label in (0, 1)
        },
        "primary_rule_usage_by_label": {
            str(label): dict(sorted(counts.items()))
            for label, counts in primary_rule_counts_by_label.items()
        },
        "positive_transitions": transition_checks,
    }


def require_complete_raw(
    *,
    raw_dir: Path,
    pair_count: int,
    expected_counts: dict[str, int],
    rule_catalog: Path,
    source_items_path: Path | None,
    prompt_path: Path,
    expected_model: str,
    expected_api_base_url: str,
    expected_reasoning_effort: str,
    expected_tiers: set[str] | None,
) -> dict[str, Any]:
    paths = {
        "items": raw_dir / "items.parquet",
        "pairs": raw_dir / "pairs.parquet",
        "metadata": raw_dir / "pair_generation_metadata.parquet",
        "summary": raw_dir / "summary.json",
        "validation": raw_dir / "validation_report.json",
        "errors": raw_dir / "errors.json",
    }
    if missing := [str(path) for path in paths.values() if not path.is_file()]:
        raise RuntimeError(f"mixed generation outputs are incomplete: {missing}")
    summary = read_json(paths["summary"], "mixed generation summary")
    validation = read_json(paths["validation"], "mixed validation report")
    generated = int(summary.get("generated_pairs", -1))
    if generated != pair_count or int(summary.get("pending", -1)) != 0:
        raise RuntimeError(f"mixed generation is not complete: {summary}")
    if validation.get("valid") is not True or int(validation.get("pairs", -1)) != pair_count:
        raise RuntimeError(f"mixed validation report is not complete: {validation}")
    errors = json.loads(paths["errors"].read_text(encoding="utf-8"))
    if errors != [] or int(summary.get("errors", -1)) != 0:
        raise RuntimeError("complete mixed generation must have no terminal task errors")
    if summary.get("model") != expected_model:
        raise RuntimeError(
            f"generation model differs: {summary.get('model')!r} != {expected_model!r}"
        )
    if str(summary.get("api_base_url") or "").rstrip("/") != (
        expected_api_base_url.rstrip("/")
    ):
        raise RuntimeError("mixed generation did not use the pinned OpenRouter API")
    if summary.get("structured_output") is not False:
        raise RuntimeError("OpenRouter mixed generation must use prompt-only JSON")
    if summary.get("reasoning_effort") != expected_reasoning_effort:
        raise RuntimeError(
            "reasoning effort differs: "
            f"{summary.get('reasoning_effort')!r} != {expected_reasoning_effort!r}"
        )
    if summary.get("prompt_sha256") != sha256_prompt(prompt_path):
        raise RuntimeError("mixed generation prompt SHA differs")
    run_signature = str(summary.get("run_signature") or "")
    if len(run_signature) != 64 or any(
        character not in "0123456789abcdef" for character in run_signature
    ):
        raise RuntimeError("mixed generation has no valid run signature")

    catalog_sha = sha256_file(rule_catalog)
    expected_catalog_contract = [{"path": str(rule_catalog), "sha256": catalog_sha}]
    if summary.get("rule_catalogs") != expected_catalog_contract:
        raise RuntimeError("mixed generation rule catalog path/SHA differs")
    manifest_path = rule_catalog.with_suffix(".manifest.json")
    manifest = read_json(manifest_path, "mixed rule catalog manifest")
    if sha256_file(manifest_path) != DEFAULT_RULE_CATALOG_MANIFEST_SHA256:
        raise RuntimeError("pinned transition-positive v4 manifest SHA differs")
    transition_contract = verify_transition_catalog(
        rule_catalog, manifest, catalog_sha
    )
    positive_rule_ids = transition_contract["positive_rule_ids"]
    recommended_two_rule_fraction = (manifest.get("selection") or {}).get(
        "recommended_two_rule_fraction"
    )
    if recommended_two_rule_fraction != 0.0:
        raise RuntimeError(
            "mixed rule catalog does not explicitly recommend an atomic first run"
        )
    if float(summary.get("two_rule_fraction", -1)) != 0.0:
        raise RuntimeError(
            "the first mixed semantic-rule experiment must use one atomic rule per pair"
        )
    if expected_tiers is not None and set(summary.get("rule_tiers") or []) != expected_tiers:
        raise RuntimeError("mixed generation rule tiers differ")
    rules = load_mutation_rules(
        [rule_catalog], tiers=expected_tiers, labels={0, 1}
    )
    if {rule.label for rule in rules} != {0, 1}:
        raise RuntimeError("mixed executable catalog must contain both labels")
    if any(not rule.allowed_categories for rule in rules):
        raise RuntimeError("every mixed executable rule must have a category scope")

    reported_source_value = str(summary.get("base_items_path") or "").strip()
    if not reported_source_value:
        raise RuntimeError("mixed generation has no style-donor source path")
    reported_source = Path(reported_source_value)
    if not reported_source.is_absolute():
        reported_source = ROOT / reported_source
    reported_source = reported_source.resolve()
    if source_items_path is not None and reported_source != source_items_path:
        raise RuntimeError("mixed generation source-item path differs from explicit pin")
    source_items_path = reported_source
    if not source_items_path.is_file():
        raise RuntimeError(f"mixed generation style-donor pool is missing: {source_items_path}")
    source_sha = sha256_file(source_items_path)
    if summary.get("base_items_sha256") != source_sha:
        raise RuntimeError("mixed generation source-item SHA differs")

    expected_label_one_fraction = expected_counts["1"] / pair_count
    if float(summary.get("label_one_fraction", -1)) != expected_label_one_fraction:
        raise RuntimeError("mixed raw label-one fraction differs from the pinned quota")
    if summary.get("label_quota_enabled") is not True:
        raise RuntimeError("mixed raw run did not enable the exact label quota")
    for scope in (
        summary,
        summary.get("planned_rule_schedule") or {},
        summary.get("realized_rule_schedule") or {},
    ):
        count_key = (
            "realized_target_counts"
            if "realized_target_counts" in scope
            else "planned_target_counts"
        )
        observed = {
            str(key): int(value)
            for key, value in (scope.get(count_key) or {}).items()
        }
        if observed != expected_counts:
            raise RuntimeError(
                f"mixed {count_key} differs from exact target quota: {observed}"
            )

    expected_positive_rule_counts = dict(
        sorted(transition_contract["positive_rule_counts"].items())
    )
    if expected_counts["1"] != sum(expected_positive_rule_counts.values()):
        raise RuntimeError(
            "positive target quota differs from the full reviewed transition capacity"
        )
    planned_positive_usage = {
        rule_id: int((summary.get("primary_rule_usage") or {}).get(rule_id, 0))
        for rule_id in sorted(positive_rule_ids)
    }
    realized_positive_usage = {
        rule_id: int(
            (summary.get("realized_primary_rule_usage") or {}).get(rule_id, 0)
        )
        for rule_id in sorted(positive_rule_ids)
    }
    if planned_positive_usage != expected_positive_rule_counts:
        raise RuntimeError(
            "planned positive primary-rule usage is not exactly balanced: "
            f"{planned_positive_usage}"
        )
    if realized_positive_usage != planned_positive_usage:
        raise RuntimeError("realized positive primary-rule usage differs from plan")

    generated_items = pd.read_parquet(paths["items"])
    pairs = pd.read_parquet(paths["pairs"])
    metadata = pd.read_parquet(paths["metadata"])
    if len(pairs) != pair_count or len(metadata) != pair_count:
        raise RuntimeError("mixed pair/metadata dimensions differ")
    if not metadata["rule_count"].eq(1).all():
        raise RuntimeError("the first mixed experiment contains 1+1 rule bundles")
    if target_counts(pairs) != expected_counts or target_counts(metadata) != expected_counts:
        raise RuntimeError("mixed raw target counts differ from the pinned quota")
    if set(metadata["model"].astype(str)) != {expected_model}:
        raise RuntimeError("mixed metadata model differs")
    if set(metadata["run_signature"].astype(str)) != {run_signature}:
        raise RuntimeError("mixed metadata run signature differs")
    if set(metadata["semantic_signature_version"].astype(str)) != {
        SEMANTIC_SIGNATURE_VERSION
    }:
        raise RuntimeError("mixed semantic-signature version differs")
    if set(metadata["rule_schedule_version"].astype(str)) != {SCHEDULE_VERSION}:
        raise RuntimeError("mixed rule-schedule version differs")
    if set(metadata["attempt_diversity_version"].astype(str)) != {
        ATTEMPT_DIVERSITY_VERSION
    }:
        raise RuntimeError("mixed attempt-diversity version differs")
    source_items = pd.read_parquet(source_items_path)
    style_donor_pool = verify_style_donor_pool(
        source_items_path,
        source_sha,
        source_items,
    )
    donor_manifest = style_donor_pool.get("manifest") or {}
    if (
        style_donor_pool["rows"] != 20_000
        or donor_manifest.get("version") != DEFAULT_STYLE_DONOR_POOL_VERSION
        or int(donor_manifest.get("copies", -1)) != 2
    ):
        raise RuntimeError("final mixed run did not use the verified x2 style-donor pool")
    try:
        requested_count = int(summary["count"])
        schedule_seed = int(summary["seed"])
        signature_limit = int(summary["semantic_signature_limit"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise RuntimeError("mixed raw summary has invalid schedule inputs") from error
    if requested_count != pair_count:
        raise RuntimeError("mixed raw schedule count differs from exact 10k")
    if signature_limit != DEFAULT_SEMANTIC_SIGNATURE_LIMIT:
        raise RuntimeError("mixed raw semantic-signature limit differs from 2")
    categories = summary.get("categories")
    if categories is not None and not isinstance(categories, list):
        raise RuntimeError("mixed raw schedule categories must be null or an array")
    rebuilt_schedule = build_balanced_rule_schedule(
        source_items,
        rules,
        count=requested_count,
        seed=schedule_seed,
        two_rule_fraction=0.0,
        categories=categories,
        semantic_signature_limit=signature_limit,
        label_one_fraction=expected_label_one_fraction,
    )
    rebuilt_schedule_checks = (
        strict_checks.verify_rebuilt_balanced_rule_schedule_metadata(
            metadata,
            summary,
            len(rules),
            rebuilt_schedule,
        )
    )
    schedule = verify_schedule_rows(
        metadata,
        pairs,
        rules,
        source_items,
        summary,
        expected_positive_rule_counts=expected_positive_rule_counts,
        generated_items=generated_items,
        transition_contract=transition_contract,
    )
    fresh_validation = validate_pair_dataset(
        paths["items"], paths["pairs"], metadata_path=paths["metadata"]
    )
    if fresh_validation.get("valid") is not True:
        raise RuntimeError(f"fresh mixed validation failed: {fresh_validation}")
    strict_checks.verify_semantic_signature_metadata(
        metadata, summary, int(summary["semantic_signature_limit"])
    )
    strict_checks.verify_attempt_provenance_metadata(metadata, summary)
    strict_checks.verify_attempt_diversity_metadata(metadata, summary)
    return {
        "summary": summary,
        "catalog_sha256": catalog_sha,
        "catalog_manifest": str(manifest_path),
        "catalog_manifest_sha256": sha256_file(manifest_path),
        "catalog_rules": len(rules),
        "style_donor_pool": style_donor_pool,
        "quota": mixed_upload.target_quota_provenance(expected_counts, pair_count),
        "target_counts": expected_counts,
        "schedule": schedule,
        "rebuilt_schedule": rebuilt_schedule_checks,
        "fresh_validation": fresh_validation,
        "_rules": rules,
        "_transition_contract": transition_contract,
    }


def verify_frozen(
    frozen: dict[str, Any],
    frozen_dir: Path,
    expected_counts: dict[str, int],
    pair_count: int,
    raw_schedule_sha: str,
    expected_positive_rule_counts: dict[str, int],
    rules: list[MutationRule],
    transition_contract: dict[str, Any],
    expected_primary_rule_coverage: int,
) -> dict[str, Any]:
    pairs = pd.read_parquet(frozen_dir / "pairs.parquet")
    if len(pairs) != pair_count or target_counts(pairs) != expected_counts:
        raise RuntimeError("frozen mixed target counts differ from the raw quota")
    summary = frozen.get("summary") or {}
    schedule = summary.get("frozen_rule_schedule") or {}
    if int(schedule.get("selected_task_count", -1)) != pair_count:
        raise RuntimeError("frozen mixed schedule has the wrong task count")
    if schedule.get("source_rule_schedule_sha256") != raw_schedule_sha:
        raise RuntimeError("frozen mixed schedule SHA differs from raw")
    if int(summary.get("dropped_before_target", -1)) != 0:
        raise RuntimeError("10k mixed source unexpectedly required freeze-time drops")
    if int(summary.get("source_generated_pairs", -1)) != pair_count:
        raise RuntimeError("mixed freeze did not consume an exact 10k raw run")
    source_counts = {
        str(key): int(value)
        for key, value in (summary.get("source_realized_target_counts") or {}).items()
    }
    if source_counts != expected_counts:
        raise RuntimeError("mixed frozen source target counts differ from raw quota")
    primary_usage = {
        str(key): int(value)
        for key, value in (schedule.get("primary_rule_usage") or {}).items()
    }
    frozen_positive_usage = {
        rule_id: int(primary_usage.get(rule_id, 0))
        for rule_id in sorted(expected_positive_rule_counts)
    }
    if frozen_positive_usage != expected_positive_rule_counts:
        raise RuntimeError("frozen positive primary-rule usage differs from exact quota")
    if (
        int(schedule.get("primary_rule_coverage", -1))
        != expected_primary_rule_coverage
        or schedule.get("full_primary_rule_coverage") is not True
    ):
        raise RuntimeError(
            "frozen mixed data lost transition-v4 primary-rule coverage"
        )
    dropped_path = frozen_dir / "dropped_pairs.json"
    if not dropped_path.is_file() or json.loads(
        dropped_path.read_text(encoding="utf-8")
    ) != []:
        raise RuntimeError("exact mixed freeze must not contain dropped pairs")
    strict_checks.verify_frozen_attempt_diversity(frozen, pair_count=pair_count)
    uniqueness = strict_checks.verify_frozen_global_card_uniqueness(
        frozen,
        raw_dir=Path(str(summary["source_dir"])),
        frozen_dir=frozen_dir,
        pair_count=pair_count,
    )
    frozen_metadata = pd.read_parquet(
        frozen_dir / "pair_generation_metadata.parquet"
    )
    frozen_items = pd.read_parquet(frozen_dir / "items.parquet")
    transition_checks = verify_positive_transition_rows(
        frozen_metadata,
        frozen_items,
        rules,
        transition_contract,
    )
    return {
        "target_counts": expected_counts,
        "positive_rule_counts": frozen_positive_usage,
        "primary_rule_coverage": int(schedule["primary_rule_coverage"]),
        "global_card_uniqueness": uniqueness,
        "positive_transitions": transition_checks,
    }


def write_report(experiment: str, result: dict[str, Any]) -> Path:
    path = ROOT / "reports" / f"{experiment}_launcher.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"created_at": datetime.now(UTC).isoformat(), **result},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def main() -> None:
    args = parse_args()
    expected_counts = mixed_upload.expected_target_counts(
        args.expected_target0, args.expected_target1, args.pair_count
    )
    env_file = absolute(args.env_file)
    kaggle.load_dotenv(env_file)
    expected_model = (args.expected_model or os.getenv("MODEL", "")).strip()
    if not expected_model:
        raise ValueError("set MODEL in .env or pass --expected-model")
    raw_dir = absolute(args.raw_dir)
    frozen_dir = absolute(args.frozen_dir)
    rule_catalog = absolute(args.expected_rule_catalog)
    source_items = (
        absolute(args.expected_source_items)
        if args.expected_source_items is not None
        else None
    )
    prompt = absolute(args.expected_prompt)
    notebook = absolute(args.notebook)
    expected_tiers = set(args.expected_tier) if args.expected_tier else None

    raw = require_complete_raw(
        raw_dir=raw_dir,
        pair_count=args.pair_count,
        expected_counts=expected_counts,
        rule_catalog=rule_catalog,
        source_items_path=source_items,
        prompt_path=prompt,
        expected_model=expected_model,
        expected_api_base_url=args.expected_api_base_url,
        expected_reasoning_effort=args.expected_reasoning_effort,
        expected_tiers=expected_tiers,
    )
    frozen = freeze(raw_dir, frozen_dir, args.pair_count)
    frozen_checks = verify_frozen(
        frozen,
        frozen_dir,
        expected_counts,
        args.pair_count,
        str(raw["schedule"]["rule_schedule_sha256"]),
        raw["schedule"]["primary_rule_usage_by_label"]["1"],
        raw["_rules"],
        raw["_transition_contract"],
        int(raw["catalog_rules"]),
    )

    owner = kaggle.validate_slug(
        os.getenv("KAGGLE_USERNAME", "").strip(), "KAGGLE_USERNAME"
    )
    dataset_ref = f"{owner}/{args.dataset_slug}"
    python = sys.executable
    upload_command = [
        python,
        "scripts/push_mixed_generation_rule_pairs_dataset.py",
        "--env-file",
        str(env_file),
        "--source-dir",
        str(frozen_dir),
        "--expected-pairs",
        str(args.pair_count),
        "--expected-target0",
        str(expected_counts["0"]),
        "--expected-target1",
        str(expected_counts["1"]),
        "--dataset-slug",
        args.dataset_slug,
        "--artifact-tag",
        args.artifact_tag,
        "--label-source",
        args.label_source,
    ]
    run(upload_command + ["--dry-run"])
    if not args.dry_run_only:
        run(upload_command)
    upload_manifest_path = (
        ROOT / ".kaggle" / "datasets" / args.dataset_slug / "upload_manifest.json"
    )
    upload_manifest = read_json(upload_manifest_path, "mixed upload manifest")
    manifest_counts = {
        str(key): int(value)
        for key, value in (upload_manifest.get("targets") or {}).items()
    }
    if (
        upload_manifest.get("dataset") != dataset_ref
        or upload_manifest.get("pairs") != args.pair_count
        or manifest_counts != expected_counts
    ):
        raise RuntimeError("mixed upload manifest differs from the pinned run")
    upload_manifest_sha = sha256_file(upload_manifest_path)

    notes = (
        f"Frozen MiniLM 5ep human train plus {args.pair_count:,} OpenRouter-generated "
        f"semantic-rule pairs (target0={expected_counts['0']}, "
        f"target1={expected_counts['1']}). Model {expected_model}; prompt-only JSON; "
        f"reasoning effort {args.expected_reasoning_effort}. Executable rule catalog "
        f"SHA-256 {raw['catalog_sha256']}. "
        f"{raw['quota']['rationale']} Unit sample weight; frozen checkpoint, "
        "recipe and IID/hard/OOD validation unchanged. This is a data+compute "
        f"ablation. Source dataset {dataset_ref}. Upload manifest SHA-256 "
        f"{upload_manifest_sha}."
    )
    run(
        [
            python,
            "scripts/create_mixed_generation_rule_10k_notebook.py",
            "--pair-count",
            str(args.pair_count),
            "--expected-target0",
            str(expected_counts["0"]),
            "--expected-target1",
            str(expected_counts["1"]),
            "--artifact-tag",
            args.artifact_tag,
            "--output",
            str(notebook),
            "--experiment-label",
            args.experiment_label,
            "--dataset-ref",
            dataset_ref,
            "--upload-manifest-sha256",
            upload_manifest_sha,
            "--label-source",
            args.label_source,
            "--notes",
            notes,
        ]
    )
    notebook_command = [
        python,
        "scripts/run_kaggle_notebook.py",
        str(notebook),
        "--env-file",
        str(env_file),
        "--slug",
        args.kernel_slug,
        "--title",
        args.title,
        "--dataset",
        "alexproger23/product-matching-validation-splits-v1",
        "--dataset",
        "alexproger23/product-matching-minilm-llm-pretrain-5ep",
        "--dataset",
        "alexproger23/product-matching-minilm-5ep-significance-v1",
        "--dataset",
        dataset_ref,
        "--no-env-sources",
    ]
    run(notebook_command + ["--dry-run"])
    if args.dry_run_only:
        result = {
            "status": "dry_run_complete",
            "dataset_ref": dataset_ref,
            "target_counts": expected_counts,
            "catalog_sha256": raw["catalog_sha256"],
            "catalog_manifest_sha256": raw["catalog_manifest_sha256"],
            "quota": raw["quota"],
            "style_donor_pool": raw["style_donor_pool"],
            "schedule": raw["schedule"],
            "rebuilt_schedule": raw["rebuilt_schedule"],
            "upload_manifest_sha256": upload_manifest_sha,
            "notebook": str(notebook),
            "kernel_slug": args.kernel_slug,
            "frozen_checks": frozen_checks,
        }
    else:
        run(notebook_command)
        output_root = Path(
            os.getenv("KAGGLE_OUTPUT_DIR", "") or ROOT / "artifacts" / "kaggle"
        ).expanduser()
        if not output_root.is_absolute():
            output_root = ROOT / output_root
        output_dir = output_root.resolve() / args.kernel_slug
        completion = read_json(
            output_dir / "notebook_completed.json", "Kaggle completion marker"
        )
        sync = read_json(
            output_dir / "google_sheets_sync.json", "Google Sheets sync marker"
        )
        comparison = read_json(
            output_dir / "baseline_comparison.json", "baseline comparison"
        )
        if completion.get("status") != "complete":
            raise RuntimeError(f"Kaggle completion is unsuccessful: {completion}")
        if completion.get("experiment") != args.experiment_label:
            raise RuntimeError("downloaded Kaggle artifacts belong to another run")
        run_id = str(completion.get("run_id") or "")
        if not run_id or sync.get("run_id") != run_id:
            raise RuntimeError("Kaggle completion and Sheets sync run IDs differ")
        if sync.get("status") != "synced" or sync.get("comparison_sheet") != "data_exps":
            raise RuntimeError(f"metrics were not synced to data_exps: {sync}")
        if comparison.get("status") != "ready":
            raise RuntimeError(f"baseline comparison is incomplete: {comparison}")
        sources = (completion.get("train_data") or {}).get("label_source_counts") or {}
        if int(sources.get(args.label_source, -1)) != args.pair_count:
            raise RuntimeError("Kaggle train data has the wrong generated count")
        notes_value = str(completion.get("notes") or "")
        if dataset_ref not in notes_value or upload_manifest_sha not in notes_value:
            raise RuntimeError("Kaggle completion does not pin generated data")
        result = {
            "status": "complete",
            "run_id": run_id,
            "dataset_ref": dataset_ref,
            "target_counts": expected_counts,
            "catalog_sha256": raw["catalog_sha256"],
            "catalog_manifest_sha256": raw["catalog_manifest_sha256"],
            "quota": raw["quota"],
            "style_donor_pool": raw["style_donor_pool"],
            "rebuilt_schedule": raw["rebuilt_schedule"],
            "upload_manifest_sha256": upload_manifest_sha,
            "kernel_slug": args.kernel_slug,
            "baseline_comparison": comparison,
            "frozen_checks": frozen_checks,
        }
    report_path = write_report(args.experiment_label, result)
    print(json.dumps({**result, "launcher_report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
