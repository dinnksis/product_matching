"""Freeze a statistical-rule pair run, publish it, and run frozen MiniLM 5ep."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from freeze_generated_pair_dataset import (
    FROZEN_ATTEMPT_DIVERSITY_VERSION,
    FROZEN_GLOBAL_CARD_UNIQUENESS_VERSION,
    GLOBAL_CARD_KEY_VERSION,
    attempt_diversity_provenance,
    freeze,
    global_card_uniqueness_provenance,
)
from item_pipeline.pair_generate import (
    ATTEMPT_DIVERSITY_VERSION,
    SEMANTIC_SIGNATURE_VERSION,
    _semantic_pair_signature,
)
from item_pipeline.pair_rules import load_mutation_rules
from item_pipeline.rule_schedule import (
    SCHEDULE_VERSION,
    BalancedRuleSchedule,
    build_balanced_rule_schedule,
)
import run_kaggle_notebook as kaggle


DEFAULT_RAW = (
    ROOT
    / "item_pipeline"
    / "artifacts"
    / "rule_first_pairs_stat_p80_scoped_v3_diversity_normalized_v2_raw5200"
)
DEFAULT_FROZEN = (
    ROOT / "item_pipeline" / "artifacts" / "rule_first_pairs_stat_p80_scoped_v3_5000"
)
DEFAULT_NOTEBOOK = ROOT / (
    "notebooks/minilm_5ep_team_ablation/"
    "minilm_5ep_stat_p80_scoped_5k_v3_2xt4.ipynb"
)
DEFAULT_DATASET_SLUG = "product-matching-generation-rules-stat-p80-scoped-5k-v3"
DEFAULT_ARTIFACT_TAG = "stat-p80-scoped-5k-v3"
DEFAULT_EXPERIMENT = "minilm_5ep_stat_p80_scoped_5k_v3"
DEFAULT_KERNEL_SLUG = "product-matching-minilm-5ep-stat-p80-scoped-5k-v3"
DEFAULT_TITLE = "MiniLM 5ep: source-scoped statistical p80 rules 5k v3"
DEFAULT_RULE_CATALOG = ROOT / (
    "configs/generation_rule_catalog_statistical_v1/"
    "statistical_negative_rules_min2_p80_scoped_v3.json"
)
DEFAULT_SOURCE_ITEMS = ROOT / "item_pipeline/artifacts/generated/items.parquet"
EXPECTED_SOURCE_ITEMS_SHA256 = (
    "54672a0241b9586563812246be77b24f976a253a9f4e732d65d2484496a13883"
)
DEFAULT_PROFILE_CAPACITY_POLICY = ROOT / (
    "configs/generation_rule_catalog_statistical_v1/profile_capacity_policy_v1.json"
)
EXPECTED_PROFILE_CAPACITY_POLICY_VERSION = "profile_capacity_policy_v1"
EXPECTED_PROFILE_CAPACITY_POLICY_SHA256 = (
    "829f9c458f1a2a72f16b766dfaf3abbdbad9a1874d71feb4457eab7da52b3fee"
)
EXPECTED_RULE_CATALOG_SHA256 = (
    "017e8ced6035695474007d5cc91e72870d77e5bef6b2348cdd13bde6cbdfdc6c"
)
DEFAULT_PROMPT = ROOT / "item_pipeline/prompts/mutate_item_by_rules.md"
DEFAULT_MODEL = "qwen3.5-397b-a17b-fp8"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_SEMANTIC_SIGNATURE_LIMIT = 2
DEFAULT_LABEL_SOURCE = "qwen_rule_first_generation_v3_scoped"
DEFAULT_TIERS = (
    "STAT_LABEL0_CROSS_SPLIT_MIN2_P80_SCOPED",
    "STAT_LABEL0_MIN2_P80_SCOPED",
)
EXPECTED_CATALOG_SELECTION = {
    "level": "category_concept_product_type",
    "relation": "different_value",
    "target_label": 0,
    "minimum_singletons": 2,
    "minimum_singleton_probability": 0.8,
    "minimum_all_occurrence_probability_or_unanimous_override": 0.8,
    "minimum_observed_attribute_key_support": 2,
    "minimum_product_type_pair_support": 2,
    "minimum_product_type_target_probability": 0.8,
    "maximum_product_types_per_rule": 12,
    "maximum_source_examples_per_product_type_rule": 4,
    "forbid_sku_article_part_number_oem": True,
    "require_non_identifier_singleton_evidence": True,
    "source_scoped_product_types": True,
    "split_rule_per_product_type": True,
    "profile_specific_support_and_split_tiers": True,
    "numeric_target_value_patterns": True,
    "source_examples_use_raw_side_aligned_values": True,
    "restrict_anchor_context_keys": True,
    "canonical_anchor_title_from_attribute_values": True,
    "semantic_value_equivalence_validation": True,
    "forbid_numeric_only_model_values": True,
    "canonical_product_type_aliases_before_profile_grouping": True,
    "deduplicate_canonical_profile_source_pairs": True,
    "finite_target_value_domain_validation": True,
    "canonical_target_values_in_semantic_signature": True,
    "canonical_quantity_units_required": True,
    "canonical_dimension_units_required": True,
    "prompt_source_examples_satisfy_effective_target_contract": True,
    "insert_stone_alias_equivalence_validation": True,
    "profile_capacity_policy_version": EXPECTED_PROFILE_CAPACITY_POLICY_VERSION,
    "profile_capacity_policy_sha256": EXPECTED_PROFILE_CAPACITY_POLICY_SHA256,
    "primary_task_capacity_formula": (
        "min(combinations(domain_size,2)*semantic_signature_limit,safety_cap)"
    ),
    "excluded_source_pair_ids": [
        "rp_6a0e6c961b827e858cab46f5",
        "rp_eb1e31b61dfa4f86f81697ef",
    ],
    "exclude_semantic_review": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--frozen-dir", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--pair-count", type=int, default=5_000)
    parser.add_argument("--dataset-slug", default=DEFAULT_DATASET_SLUG)
    parser.add_argument("--artifact-tag", default=DEFAULT_ARTIFACT_TAG)
    parser.add_argument("--experiment-label", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--kernel-slug", default=DEFAULT_KERNEL_SLUG)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--expected-rule-catalog", type=Path, default=DEFAULT_RULE_CATALOG)
    parser.add_argument("--expected-source-items", type=Path, default=DEFAULT_SOURCE_ITEMS)
    parser.add_argument("--expected-prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--expected-model", default=DEFAULT_MODEL)
    parser.add_argument("--expected-temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--expected-semantic-signature-limit",
        type=int,
        default=DEFAULT_SEMANTIC_SIGNATURE_LIMIT,
    )
    parser.add_argument("--expected-tier", action="append")
    parser.add_argument("--label-source", default=DEFAULT_LABEL_SOURCE)
    parser.add_argument("--minimum-two-rule-fraction", type=float, default=0.075)
    parser.add_argument(
        "--dry-run-only",
        action="store_true",
        help="freeze data and validate both Kaggle payloads without external mutation",
    )
    parser.add_argument(
        "--finalize-local-output",
        action="store_true",
        help=(
            "validate already-downloaded Kaggle outputs and update only the local "
            "launcher report; do not freeze, upload or run Kaggle"
        ),
    )
    parser.add_argument(
        "--local-output-dir",
        type=Path,
        help=(
            "downloaded kernel output directory for --finalize-local-output; "
            "defaults to artifacts/kaggle/<kernel-slug>"
        ),
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {description}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must be a JSON object: {path}")
    return value


def verify_local_upload_payload(
    upload_manifest_path: Path,
    *,
    dataset_ref: str,
    pair_count: int,
    label_source: str,
) -> tuple[dict[str, Any], str, int]:
    manifest = read_json_object(upload_manifest_path, "upload manifest")
    expected_fields = {
        "dataset": dataset_ref,
        "pairs": pair_count,
        "items": pair_count * 2,
        "label_source": label_source,
        "checkpoint": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in expected_fields.items()
        if manifest.get(key) != expected
    }
    targets = manifest.get("targets") or {}
    if not isinstance(targets, dict) or int(targets.get("0", -1)) != pair_count:
        mismatches["targets.0"] = {
            "expected": pair_count,
            "actual": targets.get("0") if isinstance(targets, dict) else targets,
        }
    if mismatches:
        raise RuntimeError(f"local upload manifest contract differs: {mismatches}")

    provenance = manifest.get("source_provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("local upload manifest has no source provenance")
    run_signature = provenance.get("run_signature")
    if not isinstance(run_signature, str) or len(run_signature) != 64:
        raise RuntimeError("local upload manifest has no generation run signature")
    required_provenance = {
        name: provenance.get(name)
        for name in (
            "semantic_signature",
            "rule_schedule",
            "attempt_diversity",
            "frozen_rule_schedule",
            "frozen_semantic_signature",
            "frozen_attempt_diversity",
        )
    }
    invalid = [
        name for name, value in required_provenance.items() if not isinstance(value, dict)
    ]
    if invalid:
        raise RuntimeError(f"local upload manifest dropped source provenance: {invalid}")
    semantic = required_provenance["semantic_signature"]
    rule_schedule = required_provenance["rule_schedule"]
    attempt_diversity = required_provenance["attempt_diversity"]
    frozen_rule_schedule = required_provenance["frozen_rule_schedule"]
    frozen_semantic = required_provenance["frozen_semantic_signature"]
    frozen_attempt_diversity = required_provenance["frozen_attempt_diversity"]
    assert isinstance(semantic, dict)
    assert isinstance(rule_schedule, dict)
    assert isinstance(attempt_diversity, dict)
    assert isinstance(frozen_rule_schedule, dict)
    assert isinstance(frozen_semantic, dict)
    assert isinstance(frozen_attempt_diversity, dict)
    try:
        semantic_limit = int(semantic.get("semantic_signature_limit", -1))
        semantic_max = int(semantic.get("semantic_signature_max_count", -1))
        frozen_semantic_limit = int(
            frozen_semantic.get("semantic_signature_limit", -1)
        )
        frozen_semantic_max = int(
            frozen_semantic.get("semantic_signature_max_count", -1)
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("invalid semantic-signature source provenance") from error
    schedule_sha256 = rule_schedule.get("rule_schedule_sha256")
    provenance_checks = {
        "semantic retry": semantic.get("semantic_signature_retry") is True,
        "semantic version": semantic.get("semantic_signature_version")
        == SEMANTIC_SIGNATURE_VERSION,
        "semantic cap": semantic_limit > 0 and 0 <= semantic_max <= semantic_limit,
        "balanced schedule": rule_schedule.get("balanced_rule_schedule") is True,
        "schedule version": rule_schedule.get("rule_schedule_version")
        == SCHEDULE_VERSION,
        "schedule hash": isinstance(schedule_sha256, str)
        and len(schedule_sha256) == 64,
        "attempt diversity": attempt_diversity.get("attempt_diversity_version")
        == ATTEMPT_DIVERSITY_VERSION,
        "frozen task count": frozen_rule_schedule.get("selected_task_count")
        == pair_count,
        "frozen schedule version": frozen_rule_schedule.get(
            "source_rule_schedule_version"
        )
        == SCHEDULE_VERSION,
        "frozen schedule hash": frozen_rule_schedule.get(
            "source_rule_schedule_sha256"
        )
        == schedule_sha256,
        "frozen semantic version": frozen_semantic.get(
            "semantic_signature_version"
        )
        == SEMANTIC_SIGNATURE_VERSION,
        "frozen semantic cap": frozen_semantic_limit == semantic_limit
        and 0 <= frozen_semantic_max <= frozen_semantic_limit,
        "frozen attempt version": frozen_attempt_diversity.get("version")
        == FROZEN_ATTEMPT_DIVERSITY_VERSION,
        "frozen attempt source version": frozen_attempt_diversity.get(
            "attempt_diversity_version"
        )
        == ATTEMPT_DIVERSITY_VERSION,
        "frozen attempt count": frozen_attempt_diversity.get("selected_task_count")
        == pair_count,
    }
    failed_provenance = [
        name for name, valid in provenance_checks.items() if not valid
    ]
    if failed_provenance:
        raise RuntimeError(
            "local upload manifest source provenance differs: "
            f"{failed_provenance}"
        )

    files = manifest.get("files")
    if not isinstance(files, dict) or "freeze_manifest.json" not in files:
        raise RuntimeError("upload manifest has no frozen payload file inventory")
    for name, expected in files.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise RuntimeError(f"unsafe upload payload filename: {name!r}")
        if not isinstance(expected, dict):
            raise RuntimeError(f"invalid upload payload entry for {name!r}")
        path = upload_manifest_path.parent / name
        if not path.is_file():
            raise RuntimeError(f"missing staged upload payload file: {path}")
        actual = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        pinned = {"bytes": expected.get("bytes"), "sha256": expected.get("sha256")}
        if actual != pinned:
            raise RuntimeError(
                f"staged upload payload file differs for {name!r}: "
                f"actual={actual}, pinned={pinned}"
            )
    return manifest, sha256_file(upload_manifest_path), len(files)


def verify_local_frozen_manifest(
    frozen_dir: Path,
    upload_manifest: dict[str, Any],
    *,
    pair_count: int,
) -> dict[str, Any]:
    path = frozen_dir / "freeze_manifest.json"
    manifest = read_json_object(path, "frozen dataset manifest")
    uploaded_entry = (upload_manifest.get("files") or {}).get(
        "freeze_manifest.json"
    ) or {}
    actual_sha256 = sha256_file(path)
    if uploaded_entry.get("sha256") != actual_sha256:
        raise RuntimeError(
            "local frozen manifest differs from the manifest in the upload payload"
        )
    if int(manifest.get("count", -1)) != pair_count:
        raise RuntimeError("local frozen manifest has the wrong pair count")
    uniqueness = manifest.get("frozen_global_card_uniqueness")
    if not isinstance(uniqueness, dict):
        raise RuntimeError("local frozen manifest has no global-card provenance")
    expected_uniqueness = {
        "version": FROZEN_GLOBAL_CARD_UNIQUENESS_VERSION,
        "card_key_version": GLOBAL_CARD_KEY_VERSION,
        "category_agnostic": True,
        "frozen_card_count": pair_count * 2,
        "frozen_unique_card_count": pair_count * 2,
        "frozen_duplicate_card_group_count": 0,
        "frozen_duplicate_card_row_count": 0,
    }
    mismatches = {
        key: {"expected": expected, "actual": uniqueness.get(key)}
        for key, expected in expected_uniqueness.items()
        if uniqueness.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"local frozen global-card contract differs: {mismatches}")
    return uniqueness


def finalize_local_output(
    *,
    output_dir: Path,
    upload_manifest_path: Path,
    frozen_dir: Path,
    dataset_ref: str,
    pair_count: int,
    label_source: str,
    experiment_label: str,
    kernel_slug: str,
) -> dict[str, Any]:
    upload_manifest, upload_manifest_sha256, payload_file_count = (
        verify_local_upload_payload(
            upload_manifest_path,
            dataset_ref=dataset_ref,
            pair_count=pair_count,
            label_source=label_source,
        )
    )
    frozen_global_card_uniqueness = verify_local_frozen_manifest(
        frozen_dir,
        upload_manifest,
        pair_count=pair_count,
    )

    completion_path = output_dir / "notebook_completed.json"
    sync_path = output_dir / "google_sheets_sync.json"
    comparison_path = output_dir / "baseline_comparison.json"
    run_id_path = output_dir / "experiment_run_id.txt"
    completion = read_json_object(completion_path, "Kaggle completion marker")
    sync = read_json_object(sync_path, "Google Sheets sync marker")
    comparison = read_json_object(comparison_path, "baseline comparison")
    if completion.get("status") != "complete":
        raise RuntimeError(f"Kaggle completion is not successful: {completion}")
    if completion.get("experiment") != experiment_label:
        raise RuntimeError(
            f"stale or wrong experiment artifact: {completion.get('experiment')!r}"
        )
    if completion.get("experiment_group") != "data":
        raise RuntimeError("completion artifact is not a data experiment")
    run_id = str(completion.get("run_id") or "")
    if not run_id or not run_id_path.is_file():
        raise RuntimeError("local output has no stable experiment run ID")
    if run_id_path.read_text(encoding="utf-8").strip() != run_id:
        raise RuntimeError("completion and experiment_run_id.txt differ")
    if str(sync.get("run_id") or "") != run_id:
        raise RuntimeError("completion and Google Sheets sync run IDs differ")
    if sync.get("status") != "synced" or sync.get("comparison_sheet") != "data_exps":
        raise RuntimeError(f"metrics were not synced to data_exps: {sync}")
    if sync.get("experiment_group") != "data":
        raise RuntimeError("Google Sheets sync has the wrong experiment group")
    if comparison.get("status") != "ready":
        raise RuntimeError(f"baseline comparison is not ready: {comparison}")
    if str(comparison.get("candidate_run_id") or "") != run_id:
        raise RuntimeError("baseline comparison belongs to a different run")
    if completion.get("baseline_comparison") != comparison:
        raise RuntimeError("embedded and standalone baseline comparisons differ")

    notes = str(completion.get("notes") or "")
    if dataset_ref not in notes or upload_manifest_sha256 not in notes:
        raise RuntimeError(
            "completion notes do not pin the generated Dataset and upload manifest"
        )
    train_data = completion.get("train_data") or {}
    label_source_counts = train_data.get("label_source_counts") or {}
    if int(label_source_counts.get(label_source, -1)) != pair_count:
        raise RuntimeError(
            f"Kaggle train data has wrong generated source/count: {train_data}"
        )
    training_report = completion.get("training_report")
    if not isinstance(training_report, dict):
        raise RuntimeError("completion has no embedded training report")
    training_source_counts = training_report.get("training_source_counts") or {}
    if int(training_source_counts.get(label_source, -1)) != pair_count:
        raise RuntimeError("training report has the wrong generated source/count")

    training_report_paths = sorted(output_dir.glob("*/training_report.json"))
    if len(training_report_paths) != 1:
        raise RuntimeError(
            "expected exactly one standalone training report under local output; "
            f"found={training_report_paths}"
        )
    training_report_path = training_report_paths[0]
    if read_json_object(training_report_path, "standalone training report") != (
        training_report
    ):
        raise RuntimeError("embedded and standalone training reports differ")
    model_path = training_report_path.parent / "model.safetensors"
    if not model_path.is_file() or model_path.stat().st_size < 1:
        raise RuntimeError(f"local output has no trained model: {model_path}")

    comparison_splits = comparison.get("splits") or {}
    validation_splits = training_report.get("validation_splits") or {}
    prediction_paths: dict[str, str] = {}
    metrics: dict[str, float] = {}
    for split in ("iid", "hard", "ood"):
        validation = validation_splits.get(split) or {}
        compared = comparison_splits.get(split) or {}
        try:
            metric = float(validation["macro_average_precision"])
            compared_metric = float(compared["candidate_macro_average_precision"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"missing local {split} validation metrics") from error
        if not np.isfinite(metric) or abs(metric - compared_metric) > 1e-12:
            raise RuntimeError(
                f"training report and comparison differ for {split}: "
                f"{metric} != {compared_metric}"
            )
        prediction_name = validation.get("predictions_file")
        if not isinstance(prediction_name, str) or Path(prediction_name).name != (
            prediction_name
        ):
            raise RuntimeError(f"invalid {split} prediction filename")
        prediction_path = training_report_path.parent / prediction_name
        if not prediction_path.is_file() or prediction_path.stat().st_size < 1:
            raise RuntimeError(f"missing {split} predictions: {prediction_path}")
        prediction_paths[split] = str(prediction_path)
        metrics[split] = metric

    return {
        "status": "complete",
        "finalized_from_local_output": True,
        "run_id": run_id,
        "dataset_ref": dataset_ref,
        "upload_manifest_sha256": upload_manifest_sha256,
        "kernel_slug": kernel_slug,
        "completion": str(completion_path),
        "google_sheets_sync": str(sync_path),
        "baseline_comparison": comparison,
        "frozen_global_card_uniqueness": frozen_global_card_uniqueness,
        "local_artifact_validation": {
            "upload_payload_files_verified": payload_file_count,
            "source_run_signature": upload_manifest["source_provenance"][
                "run_signature"
            ],
            "semantic_signature_version": upload_manifest["source_provenance"][
                "semantic_signature"
            ]["semantic_signature_version"],
            "rule_schedule_sha256": upload_manifest["source_provenance"][
                "rule_schedule"
            ]["rule_schedule_sha256"],
            "training_report": str(training_report_path),
            "model": str(model_path),
            "prediction_files": prediction_paths,
            "macro_average_precision": metrics,
        },
    }


def write_launcher_report(experiment_label: str, result: dict[str, Any]) -> Path:
    report_path = ROOT / "reports" / f"{experiment_label}_launcher.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {"created_at": datetime.now(UTC).isoformat(), **result},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return report_path


def require_statistical_catalog_contract(
    catalog_manifest: dict[str, Any], expected_tiers: set[str]
) -> dict[str, Any]:
    selection = catalog_manifest.get("selection") or {}
    mismatches = {
        key: {"expected": expected, "actual": selection.get(key)}
        for key, expected in EXPECTED_CATALOG_SELECTION.items()
        if selection.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"unexpected statistical catalog policy: {mismatches}")

    raw_tier_counts = catalog_manifest.get("tier_counts") or {}
    try:
        tier_counts = {
            tier: int(raw_tier_counts[tier]) for tier in sorted(expected_tiers)
        }
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"catalog manifest is missing expected tiers: {sorted(expected_tiers)}"
        ) from error
    if any(count < 1 for count in tier_counts.values()):
        raise RuntimeError(f"catalog main tiers must be non-empty: {tier_counts}")
    return {
        "tier_counts": tier_counts,
        "main_rule_count": sum(tier_counts.values()),
    }


def rebuild_expected_rule_schedule(
    source_items_path: Path,
    rule_catalog_path: Path,
    summary: dict[str, Any],
    expected_tiers: set[str],
    expected_semantic_signature_limit: int,
) -> BalancedRuleSchedule:
    """Rebuild the complete schedule from pinned inputs, never from raw rows."""

    source_items_path = source_items_path.resolve()
    if not source_items_path.is_file():
        raise RuntimeError(f"expected source items do not exist: {source_items_path}")
    source_sha256 = sha256_file(source_items_path)
    if source_sha256 != EXPECTED_SOURCE_ITEMS_SHA256:
        raise RuntimeError(
            "expected source-items SHA is not pinned: "
            f"{source_sha256} != {EXPECTED_SOURCE_ITEMS_SHA256}"
        )
    if summary.get("base_items_sha256") != source_sha256:
        raise RuntimeError("raw summary source-items SHA differs from the pinned input")
    reported_source_path = Path(str(summary.get("base_items_path") or ""))
    if not reported_source_path.is_absolute():
        reported_source_path = ROOT / reported_source_path
    if reported_source_path.resolve() != source_items_path:
        raise RuntimeError("raw summary source-items path differs from the pinned input")

    try:
        requested = int(summary["count"])
        seed = int(summary["seed"])
        two_rule_fraction = float(summary["two_rule_fraction"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise RuntimeError("raw summary has invalid schedule inputs") from error
    categories = summary.get("categories")
    if categories is not None and not isinstance(categories, list):
        raise RuntimeError("raw summary categories must be null or an array")
    rules = load_mutation_rules(
        [rule_catalog_path], tiers=expected_tiers, labels={0}
    )
    donors = pd.read_parquet(source_items_path)
    schedule = build_balanced_rule_schedule(
        donors,
        rules,
        count=requested,
        seed=seed,
        two_rule_fraction=two_rule_fraction,
        categories=categories,
        semantic_signature_limit=expected_semantic_signature_limit,
    )
    if schedule.schedule_sha256 != str(summary.get("rule_schedule_sha256") or ""):
        raise RuntimeError("rebuilt rule-schedule SHA differs from the raw summary")
    return schedule


def verify_pending_task_errors(
    errors_path: Path,
    metadata: pd.DataFrame,
    summary: dict[str, Any],
    expected_schedule: BalancedRuleSchedule,
) -> dict[str, Any]:
    """Prove that every and only absent scheduled task has one error record."""

    if not errors_path.is_file():
        raise RuntimeError(f"raw generation has no errors file: {errors_path}")
    try:
        errors = json.loads(errors_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("raw errors file is not valid JSON") from error
    if not isinstance(errors, list) or not all(isinstance(row, dict) for row in errors):
        raise RuntimeError("raw errors file must contain an array of objects")

    accepted = metadata["task_index"].astype(int).tolist()
    if len(accepted) != len(set(accepted)):
        raise RuntimeError("raw metadata contains duplicate task indices")
    scheduled = set(range(len(expected_schedule.entries)))
    accepted_set = set(accepted)
    if not accepted_set <= scheduled:
        raise RuntimeError("raw metadata contains task indices outside the schedule")
    expected_pending = scheduled - accepted_set
    try:
        reported_pending = int(summary.get("pending", -1))
        reported_errors = int(summary.get("errors", -1))
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError("raw summary has invalid pending/error counts") from error
    if reported_pending != len(expected_pending) or reported_errors != len(errors):
        raise RuntimeError("raw pending/error counts differ from scheduled gaps")

    error_tasks: list[int] = []
    for row in errors:
        try:
            task_index = int(row["task_index"])
            source_id = int(row["source_id"])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise RuntimeError("raw error record has invalid task/source IDs") from error
        if not isinstance(row.get("error"), str) or not row["error"]:
            raise RuntimeError("raw error record has no failure text")
        if task_index < 0 or task_index >= len(expected_schedule.entries):
            raise RuntimeError("raw error task is outside the rebuilt schedule")
        bundle = expected_schedule.bundle_for_task(task_index)
        if source_id != bundle.donor_id or str(row.get("category")) != bundle.category:
            raise RuntimeError("raw error record differs from the rebuilt schedule")
        error_tasks.append(task_index)
    if len(error_tasks) != len(set(error_tasks)):
        raise RuntimeError("raw errors contain duplicate task indices")
    if set(error_tasks) != expected_pending:
        raise RuntimeError("raw errors are not the exact complement of accepted tasks")
    return {
        "scheduled_tasks": len(expected_schedule.entries),
        "accepted_tasks": len(accepted_set),
        "pending_tasks": len(expected_pending),
        "pending_task_indices": sorted(expected_pending),
    }


def verify_rebuilt_balanced_rule_schedule_metadata(
    metadata: pd.DataFrame,
    summary: dict[str, Any],
    expected_rule_count: int,
    expected_schedule: BalancedRuleSchedule,
) -> dict[str, Any]:
    """Validate accepted rows as an arbitrary subset of one rebuilt full plan."""

    if summary.get("balanced_rule_schedule") is not True:
        raise RuntimeError("raw generation lacks balanced rule scheduling")
    if summary.get("rule_schedule_version") != SCHEDULE_VERSION:
        raise RuntimeError("raw generation has an unexpected rule-schedule version")
    expected_planned = expected_schedule.summary()
    if summary.get("planned_rule_schedule") != expected_planned:
        raise RuntimeError("reported planned schedule differs from the rebuilt schedule")
    for key, value in expected_planned.items():
        if summary.get(key) != value:
            raise RuntimeError(f"top-level planned schedule field differs: {key}")
    if int(expected_planned.get("eligible_rules", -1)) != expected_rule_count:
        raise RuntimeError("rebuilt schedule does not cover the full main catalog")
    if int(expected_planned.get("eligible_rule_profiles", -1)) != expected_rule_count:
        raise RuntimeError("rebuilt schedule has an unexpected profile count")

    required_columns = {
        "task_index",
        "source_style_id",
        "category",
        "product_type",
        "rule_count",
        "rule_ids",
        "balanced_rule_schedule",
        "rule_schedule_version",
        "rule_schedule_sha256",
        "scheduled_primary_rule_id",
        "scheduled_primary_profile_id",
        "scheduled_primary_product_type",
        "scheduled_primary_task_cap",
        "scheduled_secondary_rule_id",
        "scheduled_secondary_profile_id",
        "scheduled_secondary_product_type",
        "scheduled_rule_ids",
        "scheduled_rule_profile_ids",
        "profile_capacity_policy_version",
        "profile_capacity_policy_sha256",
    }
    missing = required_columns - set(metadata.columns)
    if missing:
        raise RuntimeError(
            f"raw metadata lacks balanced-schedule columns: {sorted(missing)}"
        )
    if metadata["task_index"].duplicated().any():
        raise RuntimeError("raw metadata contains duplicate task indices")
    if metadata["source_style_id"].duplicated().any():
        raise RuntimeError("balanced rule schedule reuses a style donor")

    def nullable(value: Any) -> Any:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return value

    task_indices: list[int] = []
    for row in metadata.to_dict("records"):
        try:
            task_index = int(row["task_index"])
        except (TypeError, ValueError, OverflowError) as error:
            raise RuntimeError("raw metadata has an invalid task index") from error
        if task_index < 0 or task_index >= len(expected_schedule.entries):
            raise RuntimeError("raw metadata task is outside the rebuilt schedule")
        task_indices.append(task_index)
        expected = expected_schedule.bundle_for_task(task_index).provenance(
            expected_schedule.schedule_sha256
        )
        try:
            actual_rule_ids = json.loads(str(row["rule_ids"]))
            scheduled_rule_ids = json.loads(str(row["scheduled_rule_ids"]))
            scheduled_profile_ids = json.loads(
                str(row["scheduled_rule_profile_ids"])
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError("raw metadata has invalid scheduled rule JSON") from error
        if (
            actual_rule_ids != expected["scheduled_rule_ids"]
            or scheduled_rule_ids != expected["scheduled_rule_ids"]
            or scheduled_profile_ids != expected["scheduled_rule_profile_ids"]
            or int(row["rule_count"]) != len(expected["scheduled_rule_ids"])
        ):
            raise RuntimeError(
                f"accepted task {task_index} differs from its rebuilt rule bundle"
            )
        scalar_fields = (
            "source_style_id",
            "category",
            "scheduled_primary_rule_id",
            "scheduled_primary_profile_id",
            "scheduled_primary_product_type",
            "scheduled_secondary_rule_id",
            "scheduled_secondary_profile_id",
            "scheduled_secondary_product_type",
            "rule_schedule_sha256",
        )
        for field in scalar_fields:
            actual = nullable(row.get(field))
            wanted = expected[field]
            if field == "source_style_id":
                try:
                    matches = int(actual) == int(wanted)
                except (TypeError, ValueError, OverflowError):
                    matches = False
            else:
                matches = actual == wanted
            if not matches:
                raise RuntimeError(
                    f"accepted task {task_index} differs from rebuilt field {field}"
                )
        raw_cap = nullable(row.get("scheduled_primary_task_cap"))
        actual_cap = None if raw_cap is None else int(raw_cap)
        if actual_cap != expected["scheduled_primary_task_cap"]:
            raise RuntimeError(
                f"accepted task {task_index} differs from rebuilt primary cap"
            )
        if str(row["product_type"]) != expected["scheduled_primary_product_type"]:
            raise RuntimeError(
                f"accepted task {task_index} differs from rebuilt product type"
            )
        schedule_enabled = row["balanced_rule_schedule"]
        if not isinstance(schedule_enabled, (bool, np.bool_)) or not bool(
            schedule_enabled
        ):
            raise RuntimeError("raw metadata disables balanced rule scheduling")
        if str(row["rule_schedule_version"]) != SCHEDULE_VERSION:
            raise RuntimeError("raw metadata has an unexpected schedule version")
        if str(row["profile_capacity_policy_version"]) != (
            expected_schedule.profile_capacity_policy_version
        ):
            raise RuntimeError("raw metadata has an unexpected capacity-policy version")
        if str(row["profile_capacity_policy_sha256"]) != (
            expected_schedule.profile_capacity_policy_sha256
        ):
            raise RuntimeError("raw metadata has an unexpected capacity-policy SHA")

    expected_realized = expected_schedule.realized_summary(task_indices)
    if summary.get("realized_rule_schedule") != expected_realized:
        raise RuntimeError("reported realized schedule differs from accepted tasks")
    for key, value in expected_realized.items():
        if summary.get(key) != value:
            raise RuntimeError(f"top-level realized schedule field differs: {key}")
    return {
        "schedule_sha256": expected_schedule.schedule_sha256,
        "eligible_rules": expected_rule_count,
        "primary_rule_coverage": expected_realized["realized_primary_rule_coverage"],
        "primary_rule_profile_coverage": expected_realized[
            "realized_primary_rule_profile_coverage"
        ],
        "two_rule_tasks": expected_realized["realized_scheduled_two_rule_tasks"],
        "two_rule_fraction": expected_realized[
            "realized_scheduled_two_rule_fraction"
        ],
        "capacity_limited_profiles": expected_planned[
            "capacity_limited_rule_profiles"
        ],
        "completed_scheduled_tasks": len(task_indices),
        "pending_scheduled_tasks": len(expected_schedule.entries) - len(task_indices),
        "partial_ready": len(task_indices) < len(expected_schedule.entries),
    }


def verify_semantic_signature_metadata(
    metadata: pd.DataFrame,
    summary: dict[str, Any],
    expected_limit: int,
) -> dict[str, int]:
    required_columns = {
        "category",
        "product_type",
        "applications_json",
        "semantic_signature",
        "semantic_signature_version",
    }
    missing = required_columns - set(metadata.columns)
    if missing:
        raise RuntimeError(
            f"raw metadata lacks semantic-signature columns: {sorted(missing)}"
        )
    if set(metadata["semantic_signature_version"].astype(str)) != {
        SEMANTIC_SIGNATURE_VERSION
    }:
        raise RuntimeError("raw metadata has an unexpected semantic-signature version")

    signatures: list[str] = []
    for row in metadata[list(required_columns)].to_dict("records"):
        try:
            applications = json.loads(str(row["applications_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError("raw metadata has invalid applications_json") from error
        if not isinstance(applications, list):
            raise RuntimeError("raw metadata applications_json is not an array")
        expected_signature = _semantic_pair_signature(
            row["category"], row["product_type"], applications
        )
        actual_signature = str(row["semantic_signature"])
        if actual_signature != expected_signature:
            raise RuntimeError("raw metadata semantic signature does not match applications")
        signatures.append(actual_signature)

    counts = pd.Series(signatures, dtype="string").value_counts()
    actual = {
        "unique_count": int(len(counts)),
        "max_count": int(counts.max()) if len(counts) else 0,
    }
    if actual["max_count"] > expected_limit:
        raise RuntimeError(
            "raw metadata exceeds semantic-signature limit: "
            f"{actual['max_count']} > {expected_limit}"
        )
    reported = {
        "unique_count": int(summary.get("semantic_signature_unique_count", -1)),
        "max_count": int(summary.get("semantic_signature_max_count", -1)),
    }
    if reported != actual:
        raise RuntimeError(
            f"reported semantic-signature statistics differ: {reported} != {actual}"
        )
    return actual


def verify_attempt_provenance_metadata(
    metadata: pd.DataFrame, summary: dict[str, Any]
) -> dict[str, Any]:
    fields = (
        "task_seed_offset",
        "task_retry_round",
        "selection_attempt",
        "pair_attempts_config",
        "anchor_attempts_config",
        "mutation_attempts_config",
        "task_retries_config",
    )
    missing = set(fields) - set(metadata.columns)
    if missing:
        raise RuntimeError(
            f"raw metadata lacks attempt provenance: {sorted(missing)}"
        )

    def distribution(field: str) -> dict[str, int]:
        counts = metadata[field].astype(int).value_counts().sort_index()
        return {str(int(key)): int(value) for key, value in counts.items()}

    expected = {
        "realized_task_seed_offset_distribution": distribution(
            "task_seed_offset"
        ),
        "realized_task_retry_round_distribution": distribution(
            "task_retry_round"
        ),
        "realized_selection_attempt_distribution": distribution(
            "selection_attempt"
        ),
        "realized_pair_attempts_config_distribution": distribution(
            "pair_attempts_config"
        ),
        "realized_anchor_attempts_config_distribution": distribution(
            "anchor_attempts_config"
        ),
        "realized_mutation_attempts_config_distribution": distribution(
            "mutation_attempts_config"
        ),
        "realized_task_retries_config_distribution": distribution(
            "task_retries_config"
        ),
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"reported attempt provenance differs: {key}")
    offsets = sorted(metadata["task_seed_offset"].astype(int).unique().tolist())
    if summary.get("realized_task_seed_offsets") != offsets:
        raise RuntimeError("reported realized task seed offsets differ")
    for metadata_field, summary_field in (
        ("pair_attempts_config", "pair_attempts"),
        ("anchor_attempts_config", "anchor_attempts"),
        ("mutation_attempts_config", "mutation_attempts"),
    ):
        if set(metadata[metadata_field].astype(int)) != {
            int(summary.get(summary_field, -1))
        }:
            raise RuntimeError(
                f"metadata {metadata_field} differs from run signature"
            )
    return {"task_seed_offsets": offsets, **expected}


def verify_attempt_diversity_metadata(
    metadata: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, Any]:
    try:
        return attempt_diversity_provenance(metadata, summary)
    except ValueError as error:
        raise RuntimeError(
            f"raw attempt-diversity verification failed: {error}"
        ) from error


def verify_balanced_rule_schedule_metadata(
    metadata: pd.DataFrame,
    summary: dict[str, Any],
    expected_rule_count: int,
) -> dict[str, Any]:
    if summary.get("balanced_rule_schedule") is not True:
        raise RuntimeError("raw generation lacks balanced rule scheduling")
    if summary.get("rule_schedule_version") != SCHEDULE_VERSION:
        raise RuntimeError("raw generation has an unexpected rule-schedule version")
    schedule_sha256 = str(summary.get("rule_schedule_sha256") or "")
    if len(schedule_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in schedule_sha256
    ):
        raise RuntimeError("raw generation has an invalid rule-schedule SHA-256")
    required_columns = {
        "task_index",
        "source_style_id",
        "category",
        "rule_count",
        "rule_ids",
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
    missing = required_columns - set(metadata.columns)
    if missing:
        raise RuntimeError(
            f"raw metadata lacks balanced-schedule columns: {sorted(missing)}"
        )
    if not metadata["balanced_rule_schedule"].map(bool).all():
        raise RuntimeError("raw metadata disables balanced rule scheduling")
    if set(metadata["rule_schedule_version"].astype(str)) != {SCHEDULE_VERSION}:
        raise RuntimeError("raw metadata has an unexpected rule-schedule version")
    if set(metadata["rule_schedule_sha256"].astype(str)) != {schedule_sha256}:
        raise RuntimeError("raw metadata rule-schedule SHA differs from summary")
    if metadata["source_style_id"].duplicated().any():
        raise RuntimeError("balanced rule schedule reuses a style donor")
    ordered = metadata.sort_values("task_index", kind="stable")
    task_indices = ordered["task_index"].astype(int).tolist()
    if task_indices != list(range(len(ordered))):
        raise RuntimeError("balanced rule schedule task indices are not contiguous")

    schedule_payload: list[dict[str, Any]] = []
    primary_rule_usage: dict[str, int] = {}
    secondary_rule_usage: dict[str, int] = {}
    primary_profile_usage: dict[str, int] = {}
    secondary_profile_usage: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    two_rule_tasks = 0
    observed_primary_caps: dict[str, int | None] = {}

    def increment(counts: dict[str, int], key: str) -> None:
        counts[key] = counts.get(key, 0) + 1

    for row in ordered.to_dict("records"):
        try:
            actual_rule_ids = json.loads(str(row["rule_ids"]))
            scheduled_rule_ids = json.loads(str(row["scheduled_rule_ids"]))
            scheduled_profile_ids = json.loads(
                str(row["scheduled_rule_profile_ids"])
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError("raw metadata has invalid scheduled rule JSON") from error
        if (
            not isinstance(actual_rule_ids, list)
            or scheduled_rule_ids != actual_rule_ids
            or not isinstance(scheduled_profile_ids, list)
            or len(scheduled_profile_ids) != len(scheduled_rule_ids)
            or len(scheduled_rule_ids) not in {1, 2}
            or int(row["rule_count"]) != len(scheduled_rule_ids)
        ):
            raise RuntimeError("actual rules differ from the balanced schedule")
        primary_rule_id = str(row["scheduled_primary_rule_id"])
        primary_profile_id = str(row["scheduled_primary_profile_id"])
        raw_primary_cap = row["scheduled_primary_task_cap"]
        primary_cap = None if pd.isna(raw_primary_cap) else int(raw_primary_cap)
        previous_cap = observed_primary_caps.setdefault(primary_profile_id, primary_cap)
        if previous_cap != primary_cap:
            raise RuntimeError("scheduled primary cap changes within one profile")
        if (
            primary_rule_id != str(scheduled_rule_ids[0])
            or primary_profile_id != str(scheduled_profile_ids[0])
        ):
            raise RuntimeError("scheduled primary rule/profile is inconsistent")
        secondary_rule_id = row["scheduled_secondary_rule_id"]
        secondary_profile_id = row["scheduled_secondary_profile_id"]
        if pd.isna(secondary_rule_id):
            secondary_rule_id = None
        if pd.isna(secondary_profile_id):
            secondary_profile_id = None
        expected_secondary_rule = (
            str(scheduled_rule_ids[1]) if len(scheduled_rule_ids) == 2 else None
        )
        expected_secondary_profile = (
            str(scheduled_profile_ids[1]) if len(scheduled_profile_ids) == 2 else None
        )
        if (
            secondary_rule_id != expected_secondary_rule
            or secondary_profile_id != expected_secondary_profile
        ):
            raise RuntimeError("scheduled secondary rule/profile is inconsistent")
        increment(primary_rule_usage, primary_rule_id)
        increment(primary_profile_usage, primary_profile_id)
        if expected_secondary_rule is not None:
            two_rule_tasks += 1
            increment(secondary_rule_usage, expected_secondary_rule)
            increment(secondary_profile_usage, expected_secondary_profile)
        category = str(row["category"])
        increment(category_counts, category)
        schedule_payload.append(
            {
                "task_index": int(row["task_index"]),
                "donor_id": int(row["source_style_id"]),
                "category": category,
                "profiles": [str(value) for value in scheduled_profile_ids],
            }
        )

    actual_schedule_sha256 = hashlib.sha256(
        json.dumps(
            schedule_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if actual_schedule_sha256 != schedule_sha256:
        raise RuntimeError("raw metadata does not reproduce the rule-schedule SHA-256")

    reported_planned = summary.get("planned_rule_schedule")
    if not isinstance(reported_planned, dict):
        raise RuntimeError("raw summary has no planned_rule_schedule")
    if reported_planned.get("rule_schedule_sha256") != schedule_sha256:
        raise RuntimeError("planned rule schedule SHA differs from summary")
    expected_top_level = {
        key: reported_planned.get(key)
        for key in (
            "scheduled_tasks",
            "eligible_rules",
            "eligible_rule_profiles",
            "primary_rule_coverage",
            "primary_rule_profile_coverage",
            "category_task_quotas",
            "primary_rule_usage",
            "primary_rule_profile_usage",
            "secondary_rule_usage",
            "secondary_rule_profile_usage",
            "total_rule_usage",
            "total_rule_profile_usage",
            "scheduled_two_rule_tasks",
            "scheduled_two_rule_fraction",
        )
    }
    actual_top_level = {key: summary.get(key) for key in expected_top_level}
    if actual_top_level != expected_top_level:
        raise RuntimeError("top-level and planned rule-schedule summaries differ")
    if int(summary.get("scheduled_tasks", -1)) != len(metadata):
        raise RuntimeError("scheduled task count differs from raw metadata")
    if int(summary.get("eligible_rules", -1)) != expected_rule_count:
        raise RuntimeError("balanced schedule does not cover the full main catalog")
    if int(summary.get("eligible_rule_profiles", -1)) != expected_rule_count:
        raise RuntimeError("balanced schedule has an unexpected profile count")
    if int(summary.get("primary_rule_coverage", -1)) != expected_rule_count:
        raise RuntimeError("planned primary rule coverage is incomplete")
    if int(summary.get("primary_rule_profile_coverage", -1)) != expected_rule_count:
        raise RuntimeError("planned primary profile coverage is incomplete")
    if primary_rule_usage != summary.get("primary_rule_usage"):
        raise RuntimeError("planned primary rule usage differs from raw metadata")
    if primary_profile_usage != summary.get("primary_rule_profile_usage"):
        raise RuntimeError("planned primary profile usage differs from raw metadata")
    reported_secondary_rules = {
        key: int(value)
        for key, value in (summary.get("secondary_rule_usage") or {}).items()
        if int(value) > 0
    }
    reported_secondary_profiles = {
        key: int(value)
        for key, value in (summary.get("secondary_rule_profile_usage") or {}).items()
        if int(value) > 0
    }
    if secondary_rule_usage != reported_secondary_rules:
        raise RuntimeError("planned secondary rule usage differs from raw metadata")
    if secondary_profile_usage != reported_secondary_profiles:
        raise RuntimeError("planned secondary profile usage differs from raw metadata")
    total_profile_usage = {
        profile_id: int(primary_profile_usage.get(profile_id, 0))
        + int(secondary_profile_usage.get(profile_id, 0))
        for profile_id in primary_profile_usage
    }
    total_rule_usage = {
        rule_id: int(primary_rule_usage.get(rule_id, 0))
        + int(secondary_rule_usage.get(rule_id, 0))
        for rule_id in primary_rule_usage
    }
    if total_profile_usage != summary.get("total_rule_profile_usage"):
        raise RuntimeError("planned total profile usage differs from raw metadata")
    if total_rule_usage != summary.get("total_rule_usage"):
        raise RuntimeError("planned total rule usage differs from raw metadata")
    raw_caps = summary.get("profile_primary_task_caps") or {}
    try:
        planned_caps = {str(key): int(value) for key, value in raw_caps.items()}
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("raw summary has invalid profile primary caps") from error
    for profile_id, usage in primary_profile_usage.items():
        expected_cap = planned_caps.get(profile_id)
        if observed_primary_caps.get(profile_id) != expected_cap:
            raise RuntimeError("metadata primary cap differs from planned cap")
        if expected_cap is not None and usage > expected_cap:
            raise RuntimeError(
                f"profile exceeds primary cap: {profile_id}: {usage}>{expected_cap}"
            )
    if summary.get("profile_capacity_policy_version") != (
        EXPECTED_PROFILE_CAPACITY_POLICY_VERSION
    ):
        raise RuntimeError("raw schedule has an unexpected capacity policy version")
    if summary.get("profile_capacity_policy_sha256") != (
        EXPECTED_PROFILE_CAPACITY_POLICY_SHA256
    ):
        raise RuntimeError("raw schedule has an unexpected capacity policy SHA")
    if set(metadata["profile_capacity_policy_version"].astype(str)) != {
        EXPECTED_PROFILE_CAPACITY_POLICY_VERSION
    }:
        raise RuntimeError("metadata has an unexpected capacity policy version")
    if set(metadata["profile_capacity_policy_sha256"].astype(str)) != {
        EXPECTED_PROFILE_CAPACITY_POLICY_SHA256
    }:
        raise RuntimeError("metadata has an unexpected capacity policy SHA")
    primary_values = list(primary_profile_usage.values())
    total_values = list(total_profile_usage.values())
    expected_ranges = {
        "primary_rule_profile_usage_min": min(primary_values, default=0),
        "primary_rule_profile_usage_max": max(primary_values, default=0),
        "primary_rule_profile_usage_skew": (
            max(primary_values, default=0) - min(primary_values, default=0)
        ),
        "total_rule_profile_usage_min": min(total_values, default=0),
        "total_rule_profile_usage_max": max(total_values, default=0),
        "total_rule_profile_usage_skew": (
            max(total_values, default=0) - min(total_values, default=0)
        ),
    }
    for key, value in expected_ranges.items():
        if int(summary.get(key, -1)) != value:
            raise RuntimeError(f"reported rule exposure range differs: {key}")
    saturated = set(summary.get("capacity_saturated_single_rule_profiles") or [])
    if not saturated <= set(planned_caps):
        raise RuntimeError("capacity saturation list contains an uncapped profile")
    balanced_values = [
        count for profile_id, count in total_profile_usage.items()
        if profile_id not in saturated
    ]
    balanced_skew = max(balanced_values, default=0) - min(
        balanced_values, default=0
    )
    if balanced_skew > 1:
        raise RuntimeError(f"balanced total rule exposure skew is too large: {balanced_skew}")
    for key, value in {
        "balanced_total_rule_profile_usage_min": min(balanced_values, default=0),
        "balanced_total_rule_profile_usage_max": max(balanced_values, default=0),
        "balanced_total_rule_profile_usage_skew": balanced_skew,
    }.items():
        if int(summary.get(key, -1)) != value:
            raise RuntimeError(f"reported balanced exposure range differs: {key}")
    if category_counts != summary.get("category_task_quotas"):
        raise RuntimeError("category quotas differ from raw metadata")
    if int(summary.get("scheduled_two_rule_tasks", -1)) != two_rule_tasks:
        raise RuntimeError("scheduled 1+1 count differs from raw metadata")
    scheduled_two_rule_fraction = two_rule_tasks / len(metadata) if len(metadata) else 0.0
    if abs(
        float(summary.get("scheduled_two_rule_fraction", -1))
        - scheduled_two_rule_fraction
    ) > 1e-12:
        raise RuntimeError("scheduled 1+1 fraction differs from raw metadata")

    realized = summary.get("realized_rule_schedule")
    if not isinstance(realized, dict):
        raise RuntimeError("raw summary has no realized_rule_schedule")
    expected_realized = {
        "completed_scheduled_tasks": len(metadata),
        "pending_scheduled_tasks": 0,
        "realized_primary_rule_coverage": len(primary_rule_usage),
        "realized_primary_rule_profile_coverage": len(primary_profile_usage),
        "realized_category_task_counts": category_counts,
        "realized_scheduled_two_rule_tasks": two_rule_tasks,
    }
    for key, value in expected_realized.items():
        if realized.get(key) != value or summary.get(key) != value:
            raise RuntimeError(f"reported realized schedule field differs: {key}")
    return {
        "schedule_sha256": schedule_sha256,
        "eligible_rules": expected_rule_count,
        "primary_rule_coverage": len(primary_rule_usage),
        "primary_rule_profile_coverage": len(primary_profile_usage),
        "two_rule_tasks": two_rule_tasks,
        "two_rule_fraction": scheduled_two_rule_fraction,
        "capacity_limited_profiles": len(planned_caps),
        "balanced_total_exposure_skew": balanced_skew,
        "raw_total_exposure_skew": expected_ranges[
            "total_rule_profile_usage_skew"
        ],
    }


def require_complete_raw(
    raw_dir: Path,
    pair_count: int,
    *,
    expected_source_items: Path,
    expected_rule_catalog: Path,
    expected_prompt: Path,
    expected_model: str,
    expected_temperature: float,
    expected_semantic_signature_limit: int,
    expected_tiers: set[str],
    minimum_two_rule_fraction: float,
) -> dict[str, Any]:
    summary_path = raw_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"raw generation has no completion summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    generated = int(summary.get("generated_pairs", -1))
    pending = int(summary.get("pending", -1))
    requested = int(summary.get("count", -1))
    scheduled = int(summary.get("scheduled_tasks", -1))
    if (
        generated < pair_count
        or pending < 0
        or requested != scheduled
        or generated + pending != requested
    ):
        raise RuntimeError(
            "raw generation is not freeze-ready or internally consistent: "
            f"requested={requested}, scheduled={scheduled}, "
            f"generated={generated}, pending={pending}, minimum={pair_count}"
        )
    if summary.get("validation_valid") is not True:
        raise RuntimeError("raw generation summary is not validation-valid")
    if summary.get("model") != expected_model:
        raise RuntimeError(
            f"unexpected Qwen model: {summary.get('model')!r} != {expected_model!r}"
        )
    if summary.get("structured_output") is not False:
        raise RuntimeError("expected plain JSON generation without structured output")
    if summary.get("version") != "rule_first_pair_generation_v3":
        raise RuntimeError(f"unexpected pair-generation version: {summary.get('version')}")
    if summary.get("attempt_diversity_version") != ATTEMPT_DIVERSITY_VERSION:
        raise RuntimeError(
            "raw generation has an unexpected attempt-diversity version"
        )
    if summary.get("global_duplicate_card_retry") is not True:
        raise RuntimeError("raw generation lacks global duplicate-card retry")
    if summary.get("semantic_signature_retry") is not True:
        raise RuntimeError("raw generation lacks semantic-signature retry")
    if summary.get("semantic_signature_version") != SEMANTIC_SIGNATURE_VERSION:
        raise RuntimeError("raw generation has an unexpected semantic-signature version")
    if (
        int(summary.get("semantic_signature_limit", -1))
        != expected_semantic_signature_limit
    ):
        raise RuntimeError(
            "unexpected semantic-signature limit: "
            f"{summary.get('semantic_signature_limit')} != "
            f"{expected_semantic_signature_limit}"
        )
    if abs(float(summary.get("temperature", -1)) - expected_temperature) > 1e-12:
        raise RuntimeError(
            f"unexpected Qwen temperature: {summary.get('temperature')} != {expected_temperature}"
        )
    if set(summary.get("rule_tiers") or []) != expected_tiers:
        raise RuntimeError(
            f"unexpected rule tiers: {summary.get('rule_tiers')} != {sorted(expected_tiers)}"
        )
    expected_catalog_sha = sha256_file(expected_rule_catalog)
    if expected_catalog_sha != EXPECTED_RULE_CATALOG_SHA256:
        raise RuntimeError(
            "expected rule catalog SHA is not pinned: "
            f"{expected_catalog_sha} != {EXPECTED_RULE_CATALOG_SHA256}"
        )
    catalog_contract = summary.get("rule_catalogs") or []
    expected_catalog_contract = [{
        "path": str(expected_rule_catalog.resolve()),
        "sha256": expected_catalog_sha,
    }]
    if catalog_contract != expected_catalog_contract:
        raise RuntimeError(
            f"raw rule catalog mismatch: {catalog_contract} != {expected_catalog_contract}"
        )
    expected_prompt_sha = hashlib.sha256(
        expected_prompt.read_text(encoding="utf-8").strip().encode("utf-8")
    ).hexdigest()
    if summary.get("prompt_sha256") != expected_prompt_sha:
        raise RuntimeError(
            f"raw prompt SHA mismatch: {summary.get('prompt_sha256')} != {expected_prompt_sha}"
        )
    manifest_path = expected_rule_catalog.with_suffix(".manifest.json")
    catalog_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if catalog_manifest.get("output_sha256") != expected_catalog_sha:
        raise RuntimeError("rule catalog manifest does not match the executable catalog")
    catalog_policy = require_statistical_catalog_contract(
        catalog_manifest, expected_tiers
    )
    if sha256_file(DEFAULT_PROFILE_CAPACITY_POLICY) != (
        EXPECTED_PROFILE_CAPACITY_POLICY_SHA256
    ):
        raise RuntimeError("checked-in profile capacity policy SHA is not pinned")
    if catalog_manifest.get("profile_capacity_policy_sha256") != (
        EXPECTED_PROFILE_CAPACITY_POLICY_SHA256
    ):
        raise RuntimeError("catalog manifest capacity policy differs from launcher pin")
    expected_rule_count = int(catalog_policy["main_rule_count"])
    expected_tier_counts = catalog_policy["tier_counts"]
    summary_catalog = summary.get("rule_catalog_summary") or {}
    raw_summary_tier_counts = summary_catalog.get("tier_counts") or {}
    try:
        summary_tier_counts = {
            str(tier): int(count)
            for tier, count in raw_summary_tier_counts.items()
        }
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"raw generation has invalid catalog tier counts: {raw_summary_tier_counts}"
        ) from error
    expected_summary = {
        "loaded_rules": expected_rule_count,
        "selectable_rules": expected_rule_count,
        "source_scoped_rules": expected_rule_count,
        "product_type_profiles": expected_rule_count,
        "rules_without_allowed_categories": 0,
        "labels": [0],
        "tier_counts": expected_tier_counts,
    }
    actual_summary = {
        key: summary_tier_counts if key == "tier_counts" else summary_catalog.get(key)
        for key in expected_summary
    }
    if actual_summary != expected_summary:
        raise RuntimeError(
            "raw generation catalog summary does not match the v3 main tiers: "
            f"{actual_summary} != {expected_summary}"
        )

    expected_schedule = rebuild_expected_rule_schedule(
        expected_source_items,
        expected_rule_catalog,
        summary,
        expected_tiers,
        expected_semantic_signature_limit,
    )

    metadata_path = raw_dir / "pair_generation_metadata.parquet"
    pairs_path = raw_dir / "pairs.parquet"
    validation_path = raw_dir / "validation_report.json"
    errors_path = raw_dir / "errors.json"
    if not all(
        path.is_file()
        for path in (metadata_path, pairs_path, validation_path, errors_path)
    ):
        raise RuntimeError(
            "raw generation is missing metadata, pairs, errors or validation report"
        )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("valid") is not True or int(validation.get("pairs", -1)) != generated:
        raise RuntimeError(f"raw validation report is not complete: {validation}")
    metadata = pd.read_parquet(metadata_path)
    pairs = pd.read_parquet(pairs_path)
    if len(metadata) != generated or len(pairs) != generated:
        raise RuntimeError("raw pair/metadata row counts differ from the summary")
    if not pairs["target"].eq(0).all() or not metadata["target"].eq(0).all():
        raise RuntimeError("this experiment requires only target=0 pairs")
    observed_tiers = {
        tier
        for raw in metadata["rule_tiers"].astype(str)
        for tier in json.loads(raw)
    }
    if observed_tiers != expected_tiers:
        raise RuntimeError(
            f"actual metadata tiers differ: {sorted(observed_tiers)} != {sorted(expected_tiers)}"
        )
    if set(metadata["model"].astype(str)) != {expected_model}:
        raise RuntimeError("actual metadata model differs from the expected Qwen model")
    if set(metadata["run_signature"].astype(str)) != {str(summary["run_signature"])}:
        raise RuntimeError("metadata run signature differs from summary")
    rule_schedule_statistics = verify_rebuilt_balanced_rule_schedule_metadata(
        metadata, summary, expected_rule_count, expected_schedule
    )
    pending_task_statistics = verify_pending_task_errors(
        errors_path, metadata, summary, expected_schedule
    )
    semantic_signature_statistics = verify_semantic_signature_metadata(
        metadata, summary, expected_semantic_signature_limit
    )
    attempt_provenance_statistics = verify_attempt_provenance_metadata(
        metadata, summary
    )
    attempt_diversity_statistics = verify_attempt_diversity_metadata(
        metadata, summary
    )
    realized_two_rule_fraction = float(metadata["rule_count"].eq(2).mean())
    if realized_two_rule_fraction < minimum_two_rule_fraction:
        raise RuntimeError(
            "too few 1+1 combinations: "
            f"{realized_two_rule_fraction:.3f} < {minimum_two_rule_fraction:.3f}"
        )
    reported_fraction = float(summary.get("realized_two_rule_fraction", -1))
    if abs(realized_two_rule_fraction - reported_fraction) > 1e-12:
        raise RuntimeError("reported and actual two-rule fractions differ")
    summary["verified_catalog_manifest"] = str(manifest_path)
    summary["verified_catalog_sha256"] = expected_catalog_sha
    summary["verified_catalog_main_rules"] = expected_rule_count
    summary["verified_realized_two_rule_fraction"] = realized_two_rule_fraction
    summary["verified_semantic_signature_statistics"] = (
        semantic_signature_statistics
    )
    summary["verified_rule_schedule_statistics"] = rule_schedule_statistics
    summary["verified_pending_task_statistics"] = pending_task_statistics
    summary["verified_attempt_provenance_statistics"] = (
        attempt_provenance_statistics
    )
    summary["verified_attempt_diversity_statistics"] = (
        attempt_diversity_statistics
    )
    return summary


def verify_frozen_rule_schedule(
    frozen: dict[str, Any],
    raw_summary: dict[str, Any],
    *,
    pair_count: int,
    expected_semantic_signature_limit: int,
    minimum_two_rule_fraction: float,
) -> dict[str, Any]:
    summary = frozen.get("summary") or {}
    schedule = summary.get("frozen_rule_schedule") or {}
    expected_main_rules = int(raw_summary.get("verified_catalog_main_rules", -1))
    raw_schedule = raw_summary.get("verified_rule_schedule_statistics") or {}
    checks = {
        "selected_task_count": int(schedule.get("selected_task_count", -1))
        == pair_count,
        "source_schedule_sha": schedule.get("source_rule_schedule_sha256")
        == raw_schedule.get("schedule_sha256"),
        "primary_rule_coverage": int(schedule.get("primary_rule_coverage", -1))
        == expected_main_rules,
        "primary_profile_coverage": int(
            schedule.get("primary_rule_profile_coverage", -1)
        )
        == expected_main_rules,
        "full_primary_rule_coverage": schedule.get("full_primary_rule_coverage")
        is True,
        "full_primary_profile_coverage": schedule.get(
            "full_primary_rule_profile_coverage"
        )
        is True,
        "capacity_violations": schedule.get(
            "primary_rule_profile_cap_violations"
        )
        == {},
        "semantic_signature_version": schedule.get("semantic_signature_version")
        == SEMANTIC_SIGNATURE_VERSION,
        "semantic_signature_limit": int(
            schedule.get("semantic_signature_limit", -1)
        )
        == expected_semantic_signature_limit,
        "semantic_signature_cap": int(
            schedule.get("semantic_signature_max_count", -1)
        )
        <= expected_semantic_signature_limit,
        "capacity_policy_version": schedule.get(
            "profile_capacity_policy_version"
        )
        == EXPECTED_PROFILE_CAPACITY_POLICY_VERSION,
        "capacity_policy_sha": schedule.get("profile_capacity_policy_sha256")
        == EXPECTED_PROFILE_CAPACITY_POLICY_SHA256,
        "two_rule_fraction": float(schedule.get("two_rule_fraction", -1))
        >= minimum_two_rule_fraction,
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"frozen rule schedule verification failed: {failed}")
    return schedule


def verify_frozen_attempt_diversity(
    frozen: dict[str, Any],
    *,
    pair_count: int,
) -> dict[str, Any]:
    summary = frozen.get("summary") or {}
    provenance = summary.get("frozen_attempt_diversity") or {}
    checks = {
        "version": provenance.get("version")
        == FROZEN_ATTEMPT_DIVERSITY_VERSION,
        "attempt_diversity_version": provenance.get(
            "attempt_diversity_version"
        )
        == ATTEMPT_DIVERSITY_VERSION,
        "selected_task_count": int(
            provenance.get("selected_task_count", -1)
        )
        == pair_count,
        "anchor_nonce_hash_valid_count": int(
            provenance.get("anchor_nonce_hash_valid_count", -1)
        )
        == pair_count,
        "anchor_nonce_hash_unique_count": int(
            provenance.get("anchor_nonce_hash_unique_count", -1)
        )
        == pair_count,
        "mutation_nonce_hash_valid_count": int(
            provenance.get("mutation_nonce_hash_valid_count", -1)
        )
        == pair_count,
        "mutation_nonce_hash_unique_count": int(
            provenance.get("mutation_nonce_hash_unique_count", -1)
        )
        == pair_count,
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(
            f"frozen attempt-diversity verification failed: {failed}"
        )
    return provenance


def verify_frozen_global_card_uniqueness(
    frozen: dict[str, Any],
    *,
    raw_dir: Path,
    frozen_dir: Path,
    pair_count: int,
) -> dict[str, Any]:
    """Recompute and enforce category-agnostic individual-card uniqueness."""

    summary = frozen.get("summary") or {}
    manifest = frozen.get("manifest") or {}
    provenance = summary.get("frozen_global_card_uniqueness") or {}
    source_items_path = raw_dir / "items.parquet"
    frozen_items_path = frozen_dir / "items.parquet"
    dropped_path = frozen_dir / "dropped_pairs.json"
    persisted_summary_path = frozen_dir / "summary.json"
    persisted_manifest_path = frozen_dir / "freeze_manifest.json"
    if not all(
        path.is_file()
        for path in (
            source_items_path,
            frozen_items_path,
            dropped_path,
            persisted_summary_path,
            persisted_manifest_path,
        )
    ):
        raise RuntimeError(
            "frozen global-card verification artifacts are incomplete"
        )
    dropped = json.loads(dropped_path.read_text(encoding="utf-8"))
    if not isinstance(dropped, list) or not all(
        isinstance(row, dict) for row in dropped
    ):
        raise RuntimeError("frozen dropped-pair provenance is invalid")
    recomputed = global_card_uniqueness_provenance(
        pd.read_parquet(source_items_path),
        pd.read_parquet(frozen_items_path),
        dropped,
    )
    persisted_summary = json.loads(
        persisted_summary_path.read_text(encoding="utf-8")
    )
    persisted_manifest = json.loads(
        persisted_manifest_path.read_text(encoding="utf-8")
    )
    checks = {
        "version": provenance.get("version")
        == FROZEN_GLOBAL_CARD_UNIQUENESS_VERSION,
        "card_key_version": provenance.get("card_key_version")
        == GLOBAL_CARD_KEY_VERSION,
        "category_agnostic": provenance.get("category_agnostic") is True,
        "attributes_order_insensitive": provenance.get(
            "attributes_order_insensitive"
        )
        is True,
        "frozen_card_count": int(provenance.get("frozen_card_count", -1))
        == pair_count * 2,
        "frozen_unique_card_count": int(
            provenance.get("frozen_unique_card_count", -1)
        )
        == pair_count * 2,
        "frozen_duplicate_card_groups": int(
            provenance.get("frozen_duplicate_card_group_count", -1)
        )
        == 0,
        "frozen_duplicate_card_rows": int(
            provenance.get("frozen_duplicate_card_row_count", -1)
        )
        == 0,
        "dropped_pair_count": int(provenance.get("dropped_pair_count", -1))
        == int(summary.get("dropped_before_target", -2)),
        "recomputed_provenance": provenance == recomputed,
        "manifest_provenance": manifest.get(
            "frozen_global_card_uniqueness"
        )
        == provenance,
        "persisted_summary_provenance": persisted_summary.get(
            "frozen_global_card_uniqueness"
        )
        == provenance,
        "persisted_manifest_provenance": persisted_manifest.get(
            "frozen_global_card_uniqueness"
        )
        == provenance,
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(
            "frozen global card uniqueness verification failed: "
            f"{failed}"
        )
    return provenance


def main() -> None:
    args = parse_args()
    if args.pair_count < 1:
        raise ValueError("pair-count must be positive")
    if not 0.0 <= args.minimum_two_rule_fraction <= 1.0:
        raise ValueError("minimum-two-rule-fraction must be in [0, 1]")
    if args.expected_semantic_signature_limit < 1:
        raise ValueError("expected-semantic-signature-limit must be positive")
    if args.finalize_local_output:
        frozen_dir = (
            args.frozen_dir if args.frozen_dir.is_absolute() else ROOT / args.frozen_dir
        ).resolve()
        upload_manifest_path = (
            ROOT
            / ".kaggle"
            / "datasets"
            / args.dataset_slug
            / "upload_manifest.json"
        )
        local_manifest = read_json_object(upload_manifest_path, "upload manifest")
        dataset_ref = str(local_manifest.get("dataset") or "")
        dataset_parts = dataset_ref.split("/")
        if len(dataset_parts) != 2 or dataset_parts[1] != args.dataset_slug:
            raise RuntimeError(
                f"local upload manifest has the wrong Dataset ref: {dataset_ref!r}"
            )
        if args.local_output_dir is not None:
            output_dir = (
                args.local_output_dir
                if args.local_output_dir.is_absolute()
                else ROOT / args.local_output_dir
            ).resolve()
        else:
            output_dir = (ROOT / "artifacts" / "kaggle" / args.kernel_slug).resolve()
        result = finalize_local_output(
            output_dir=output_dir,
            upload_manifest_path=upload_manifest_path,
            frozen_dir=frozen_dir,
            dataset_ref=dataset_ref,
            pair_count=args.pair_count,
            label_source=args.label_source,
            experiment_label=args.experiment_label,
            kernel_slug=args.kernel_slug,
        )
        write_launcher_report(args.experiment_label, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return
    raw_dir = (args.raw_dir if args.raw_dir.is_absolute() else ROOT / args.raw_dir).resolve()
    frozen_dir = (
        args.frozen_dir if args.frozen_dir.is_absolute() else ROOT / args.frozen_dir
    ).resolve()
    notebook = (
        args.notebook if args.notebook.is_absolute() else ROOT / args.notebook
    ).resolve()
    expected_rule_catalog = (
        args.expected_rule_catalog
        if args.expected_rule_catalog.is_absolute()
        else ROOT / args.expected_rule_catalog
    ).resolve()
    expected_source_items = (
        args.expected_source_items
        if args.expected_source_items.is_absolute()
        else ROOT / args.expected_source_items
    ).resolve()
    expected_prompt = (
        args.expected_prompt
        if args.expected_prompt.is_absolute()
        else ROOT / args.expected_prompt
    ).resolve()
    expected_tiers = set(args.expected_tier or DEFAULT_TIERS)
    raw_summary = require_complete_raw(
        raw_dir,
        args.pair_count,
        expected_source_items=expected_source_items,
        expected_rule_catalog=expected_rule_catalog,
        expected_prompt=expected_prompt,
        expected_model=args.expected_model,
        expected_temperature=args.expected_temperature,
        expected_semantic_signature_limit=args.expected_semantic_signature_limit,
        expected_tiers=expected_tiers,
        minimum_two_rule_fraction=args.minimum_two_rule_fraction,
    )
    frozen = freeze(raw_dir, frozen_dir, args.pair_count)
    frozen_rule_schedule = verify_frozen_rule_schedule(
        frozen,
        raw_summary,
        pair_count=args.pair_count,
        expected_semantic_signature_limit=args.expected_semantic_signature_limit,
        minimum_two_rule_fraction=args.minimum_two_rule_fraction,
    )
    frozen_attempt_diversity = verify_frozen_attempt_diversity(
        frozen,
        pair_count=args.pair_count,
    )
    frozen_global_card_uniqueness = verify_frozen_global_card_uniqueness(
        frozen,
        raw_dir=raw_dir,
        frozen_dir=frozen_dir,
        pair_count=args.pair_count,
    )

    env_file = args.env_file if args.env_file.is_absolute() else ROOT / args.env_file
    kaggle.load_dotenv(env_file)
    owner = kaggle.validate_slug(
        os.getenv("KAGGLE_USERNAME", "").strip(), "KAGGLE_USERNAME"
    )
    dataset_ref = f"{owner}/{args.dataset_slug}"
    python = sys.executable
    upload = [
        python,
        "scripts/push_generation_rule_pairs_dataset.py",
        "--env-file",
        str(env_file),
        "--source-dir",
        str(frozen_dir),
        "--expected-pairs",
        str(args.pair_count),
        "--dataset-slug",
        args.dataset_slug,
        "--artifact-tag",
        args.artifact_tag,
        "--message",
        f"Add {args.pair_count} source-scoped statistical p80 negative pairs v3",
        "--label-source",
        args.label_source,
    ]
    run(upload + ["--dry-run"])
    if not args.dry_run_only:
        run(upload)

    upload_manifest_path = ROOT / ".kaggle" / "datasets" / args.dataset_slug / "upload_manifest.json"
    if not upload_manifest_path.is_file():
        raise RuntimeError(f"uploader did not create {upload_manifest_path}")
    upload_manifest_sha256 = sha256_file(upload_manifest_path)
    upload_manifest = json.loads(upload_manifest_path.read_text(encoding="utf-8"))
    if (
        upload_manifest.get("dataset") != dataset_ref
        or upload_manifest.get("pairs") != args.pair_count
        or upload_manifest.get("label_source") != args.label_source
    ):
        raise RuntimeError(f"unexpected upload manifest: {upload_manifest}")
    upload_provenance = upload_manifest.get("source_provenance") or {}
    upload_attempt_diversity = upload_provenance.get("attempt_diversity") or {}
    upload_frozen_attempt_diversity = upload_provenance.get(
        "frozen_attempt_diversity"
    ) or {}
    if upload_attempt_diversity.get("attempt_diversity_version") != (
        ATTEMPT_DIVERSITY_VERSION
    ):
        raise RuntimeError(
            "upload manifest dropped source attempt-diversity provenance"
        )
    if upload_frozen_attempt_diversity != frozen_attempt_diversity:
        raise RuntimeError(
            "upload manifest changed frozen attempt-diversity provenance"
        )
    uploaded_freeze_manifest = (upload_manifest.get("files") or {}).get(
        "freeze_manifest.json"
    ) or {}
    if uploaded_freeze_manifest.get("sha256") != sha256_file(
        frozen_dir / "freeze_manifest.json"
    ):
        raise RuntimeError(
            "upload manifest dropped the frozen global-card provenance"
        )

    notes = (
        f"Frozen MiniLM 5ep baseline human train plus {args.pair_count:,} "
        "Qwen-generated negative pairs from category- and product-type-scoped atomic rules. Rules "
        "require at least two singleton observations, singleton P(label=0)>=0.8, "
        "P(label=0)>=0.8 across all occurrences, and at least two source pairs with "
        "P(label=0)>=0.8 inside an exact product-type profile. Contradictory override "
        "and semantic-review rules are excluded. SKU/article concepts are vetoed. "
        "Every pair passes source-type, dependent-field, exact target-attribute and "
        "literal title-substitution checks; anchor titles contain no hidden facts "
        "outside their validated attributes. "
        f"At most {args.expected_semantic_signature_limit} pairs share one "
        "direction-invariant category/product-type mutation signature. "
        f"Realized compatible two-rule fraction: "
        f"{float(frozen_rule_schedule['two_rule_fraction']):.3f} in the frozen subset. "
        "Published cards are globally unique under category-agnostic normalized "
        "name+attribute identity. "
        "Unit sample weight; frozen validation/checkpoint/recipe unchanged. This "
        f"is a data+compute ablation. Source dataset {dataset_ref}."
        f" Upload manifest SHA-256 {upload_manifest_sha256}."
    )
    run(
        [
            python,
            "scripts/create_generation_rule_10k_notebook.py",
            "--pair-count",
            str(args.pair_count),
            "--artifact-tag",
            args.artifact_tag,
            "--output",
            str(notebook),
            "--experiment-label",
            args.experiment_label,
            "--dataset-ref",
            dataset_ref,
            "--upload-manifest-sha256",
            upload_manifest_sha256,
            "--expected-label-source",
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
            "raw_summary": raw_summary,
            "frozen_summary": frozen["summary"],
            "dataset_ref": dataset_ref,
            "upload_manifest_sha256": upload_manifest_sha256,
            "notebook": str(notebook),
            "kernel_slug": args.kernel_slug,
            "frozen_global_card_uniqueness": frozen_global_card_uniqueness,
        }
    else:
        run(notebook_command)
        configured_output_root = os.getenv("KAGGLE_OUTPUT_DIR", "").strip()
        output_root = (
            Path(configured_output_root).expanduser()
            if configured_output_root
            else ROOT / "artifacts" / "kaggle"
        )
        if not output_root.is_absolute():
            output_root = ROOT / output_root
        output_dir = output_root.resolve() / args.kernel_slug
        completion_path = output_dir / "notebook_completed.json"
        sync_path = output_dir / "google_sheets_sync.json"
        comparison_path = output_dir / "baseline_comparison.json"
        if not completion_path.is_file() or not sync_path.is_file():
            raise RuntimeError(
                "Kaggle run finished without completion or Google Sheets artifact"
            )
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        sync = json.loads(sync_path.read_text(encoding="utf-8"))
        comparison = (
            json.loads(comparison_path.read_text(encoding="utf-8"))
            if comparison_path.is_file()
            else None
        )
        if completion.get("status") != "complete":
            raise RuntimeError(f"Kaggle completion is not successful: {completion}")
        if completion.get("experiment") != args.experiment_label:
            raise RuntimeError(
                f"stale or wrong experiment artifact: {completion.get('experiment')!r}"
            )
        run_id = str(completion.get("run_id") or "")
        if not run_id or str(sync.get("run_id") or "") != run_id:
            raise RuntimeError("completion and Google Sheets sync run IDs differ")
        train_data = completion.get("train_data") or {}
        label_source_counts = train_data.get("label_source_counts") or {}
        if int(label_source_counts.get(args.label_source, -1)) != args.pair_count:
            raise RuntimeError(
                f"Kaggle train data has wrong generated source/count: {train_data}"
            )
        if upload_manifest_sha256 not in str(completion.get("notes") or ""):
            raise RuntimeError("completion notes do not pin the generated Dataset manifest")
        if sync.get("status") != "synced" or sync.get("comparison_sheet") != "data_exps":
            raise RuntimeError(f"metrics were not synced to data_exps: {sync}")
        result = {
            "status": "complete",
            "run_id": run_id,
            "dataset_ref": dataset_ref,
            "upload_manifest_sha256": upload_manifest_sha256,
            "kernel_slug": args.kernel_slug,
            "completion": str(completion_path),
            "google_sheets_sync": str(sync_path),
            "baseline_comparison": comparison,
            "frozen_global_card_uniqueness": frozen_global_card_uniqueness,
        }

    write_launcher_report(args.experiment_label, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
