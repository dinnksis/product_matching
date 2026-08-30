#!/usr/bin/env python3
"""Aggregate the MiniLM-5ep SFT campaign and apply stage-wise IID Holm correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import create_minilm_5ep_sft_hparam_notebooks as campaign
import create_qwen_training_notebook as qwen_builder
from src.experiment_significance import (
    align_predictions,
    compare_prediction_frames,
    macro_average_precision,
    read_prediction_artifact,
)

DEFAULT_ARTIFACTS_DIR = ROOT / "artifacts" / "kaggle"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "minilm_5ep_sft_hparam_search_v1"


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = [float(value) for value in p_values]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("Holm correction requires finite p-values in [0, 1]")
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [1.0] * len(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def _metric(report: Mapping[str, Any], split: str, name: str) -> Any:
    return report.get("validation_splits", {}).get(split, {}).get(name)


def _comparison_metric(
    completion: Mapping[str, Any], split: str, name: str
) -> Any:
    return (
        completion.get("baseline_comparison", {})
        .get("splits", {})
        .get(split, {})
        .get(name)
    )


def completion_path(artifacts_dir: Path, kernel_slug: str) -> Path:
    return artifacts_dir / kernel_slug / "notebook_completed.json"


def _stage_family_size(stage: Mapping[str, Any]) -> int:
    family = stage.get("family", {})
    candidates = sum(
        variant.get("role", "candidate") != "current_protocol_control"
        for variant in stage.get("variants", [])
    )
    value = family.get("maximum_hypotheses", candidates)
    if isinstance(value, bool) or not isinstance(value, int) or value < candidates:
        raise campaign.CampaignConfigError("Invalid planned hypothesis family size")
    return value


def _direct_axis_metadata(
    stage: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[str | None, Any, list[Any]]:
    """Attach machine-readable boundary metadata to an initial direct stage."""
    axis = stage.get("axis")
    if not isinstance(axis, Mapping) or len(axis) != 1:
        return None, None, []
    axis_name = str(next(iter(axis)))
    return (
        axis_name,
        campaign.stage_materializer._effective_axis_value(config, axis_name),
        campaign.stage_materializer._declared_extension_levels(
            stage,
            axis_name=axis_name,
        ),
    )


def adaptive_execution_entries(
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return unique run expectations; statistical families remain projections."""
    base_config = campaign.cross_builder.load_training_config(
        campaign.BASE_CONFIG_PATH
    )
    contract = campaign.normalized_campaign_execution_contract(
        plan,
        lock,
        base_config=base_config,
    )
    entries: list[dict[str, Any]] = []
    for origin in contract["origins"]:
        config = deepcopy(dict(origin["resolved_config"]))
        entries.append(
            {
                "stage": str(origin["origin_effective_stage"]),
                "experiment": str(origin["experiment"]),
                "kernel_slug": str(origin["kernel_slug"]),
                "role": str(origin["source_role"]),
                "planned_overrides": campaign.resolved_config_overrides(
                    base_config, config
                ),
                "expected_config": config,
                "expected_recipe_sha256": str(origin["recipe_sha256"]),
                "expected_source_sha256": str(
                    origin["expected_source_sha256"]
                ),
                "loss_variant": str(origin["loss_variant"]),
                "expected_loss_hook_sha256": str(origin["loss_hook_sha256"]),
                "expected_run_id": str(origin["run_id"]),
                "expected_iid_predictions_sha256": str(
                    origin["iid_predictions_sha256"]
                ),
                "expected_completion_sha256": str(origin["completion_sha256"]),
                "expected_notes": str(origin["completion_notes"]),
                "expected_notes_sha256": str(
                    origin["completion_notes_sha256"]
                ),
                "provenance_alias": None,
                "is_hypothesis": bool(origin["source_is_hypothesis"]),
                "family_id": None,
                "hypothesis_family_size": None,
                "origin_id": str(origin["origin_id"]),
                "origin_ids": [str(origin["origin_id"])],
                "parent_provenance": None,
                "axis": None,
                "level": None,
                "conditional_extension_levels": [],
                "stage_lock_payload_sha256": contract[
                    "lock_payload_sha256"
                ],
                "stage_lock_transition_kind": "adaptive_reuse",
                "completion_path": str(origin["completion_artifact_path"]),
                "reused_origin": True,
            }
        )
    for normalized in contract["variants"]:
        entries.append(
            {
                "stage": contract["effective_stage"],
                "experiment": normalized["experiment"],
                "kernel_slug": normalized["kernel_slug"],
                "role": normalized["role"],
                "planned_overrides": deepcopy(normalized["planned_overrides"]),
                "expected_config": deepcopy(normalized["expected_config"]),
                "expected_recipe_sha256": normalized["recipe_sha256"],
                "expected_source_sha256": normalized["source_sha256"],
                "loss_variant": normalized["loss_variant"],
                "expected_loss_hook_sha256": normalized["loss_hook_sha256"],
                "expected_run_id": None,
                "expected_iid_predictions_sha256": None,
                "expected_completion_sha256": None,
                "expected_notes": normalized["expected_notes"],
                "expected_notes_sha256": hashlib.sha256(
                    normalized["expected_notes"].encode("utf-8")
                ).hexdigest(),
                "provenance_alias": None,
                "is_hypothesis": normalized["is_hypothesis"],
                "family_id": normalized["family_id"],
                "hypothesis_family_size": normalized[
                    "hypothesis_family_size"
                ],
                "origin_id": None,
                "origin_ids": deepcopy(normalized["origin_ids"]),
                "parent_provenance": deepcopy(
                    normalized["parent_provenance"]
                ),
                "axis": None,
                "level": None,
                "conditional_extension_levels": [],
                "stage_lock_payload_sha256": contract[
                    "lock_payload_sha256"
                ],
                "stage_lock_transition_kind": "adaptive_stage",
                "completion_path": str(
                    completion_path(
                        DEFAULT_ARTIFACTS_DIR,
                        str(normalized["kernel_slug"]),
                    )
                ),
                "reused_origin": False,
                "variant": normalized["variant"],
            }
        )
    by_experiment: dict[str, dict[str, Any]] = {}
    for entry in entries:
        experiment = str(entry["experiment"])
        previous = by_experiment.get(experiment)
        if previous is not None and previous != entry:
            raise RuntimeError(
                f"Adaptive run {experiment!r} has conflicting reused/new entries"
            )
        by_experiment[experiment] = entry
    return [by_experiment[key] for key in sorted(by_experiment)]


def expected_entries(
    plan: Mapping[str, Any],
    stage_name: str | None,
    *,
    stage_lock: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    base_config = campaign.cross_builder.load_training_config(
        campaign.BASE_CONFIG_PATH
    )
    _, source_sha256 = campaign.baseline_builder.embedded_sources()
    entries: list[dict[str, Any]] = []
    if stage_lock is not None:
        if stage_lock.get("schema_version") == 2:
            contract = campaign.normalized_campaign_execution_contract(
                plan,
                stage_lock,
                base_config=base_config,
            )
            if stage_name is not None and stage_name not in set(
                contract["accepted_stage_filters"]
            ):
                raise campaign.CampaignConfigError(
                    f"--stage {stage_name!r} differs from locked stage "
                    f"{contract['effective_stage']!r}"
                )
            return adaptive_execution_entries(plan, stage_lock)
        effective_stage = campaign.stage_lock_effective_stage(stage_lock)
        if stage_name is not None and stage_name not in {
            str(stage_lock["target_stage"]),
            effective_stage,
        }:
            raise campaign.CampaignConfigError(
                f"--stage {stage_name!r} differs from locked stage "
                f"{effective_stage!r}"
            )
        family_size = int(
            stage_lock["resolved_stage"]["family"]["maximum_hypotheses"]
        )
        transition_kind = str(
            stage_lock.get("transition_kind", "stage_transition")
        )
        parent = stage_lock["parent"]
        resolved_stage = stage_lock["resolved_stage"]
        axis_name = str(resolved_stage["axis"])
        conditional_extension_levels = deepcopy(
            resolved_stage.get("conditional_extension_levels", [])
        )
        parent_provenance = campaign.stage_lock_parent_provenance(stage_lock)
        common_lock_fields = {
            "stage_lock_payload_sha256": stage_lock["lock_payload_sha256"],
            "stage_lock_transition_kind": transition_kind,
            "parent_provenance": parent_provenance,
        }
        if transition_kind == "conditional_boundary_extension":
            for prior in stage_lock["prior_entries"]:
                entry = deepcopy(dict(prior))
                entry.update(common_lock_fields)
                entry["extension_consumed_axis"] = axis_name
                entries.append(entry)
        else:
            parent_config = dict(parent["resolved_config"])
            parent_overrides = campaign.resolved_config_overrides(
                base_config, parent_config
            )
            parent_loss = str(
                parent.get("loss_variant", campaign.DEFAULT_LOSS_VARIANT)
            )
            _, _, calculated_parent_loss_sha = campaign.variant_loss(
                {"loss_variant": parent_loss}
            )
            entries.append(
                {
                    "stage": effective_stage,
                    "experiment": str(parent["experiment"]),
                    "kernel_slug": str(parent["kernel_slug"]),
                    "role": "stage_anchor",
                    "planned_overrides": parent_overrides,
                    "expected_config": parent_config,
                    "expected_recipe_sha256": str(parent["recipe_sha256"]),
                    "expected_source_sha256": str(
                        parent.get("code_bundle_sha256", source_sha256)
                    ),
                    "loss_variant": parent_loss,
                    "expected_loss_hook_sha256": str(
                        parent.get(
                            "loss_hook_sha256", calculated_parent_loss_sha
                        )
                    ),
                    "expected_run_id": str(parent["run_id"]),
                    "expected_iid_predictions_sha256": str(
                        parent["iid_predictions_sha256"]
                    ),
                    "expected_completion_sha256": str(
                        parent["completion_sha256"]
                    ),
                    "expected_notes": parent.get("notes"),
                    "provenance_alias": None,
                    "is_hypothesis": False,
                    "hypothesis_family_size": family_size,
                    "axis": axis_name,
                    "level": resolved_stage["parent_level"],
                    "conditional_extension_levels": conditional_extension_levels,
                    **common_lock_fields,
                }
            )
        variants = (
            (effective_stage, variant)
            for variant in resolved_stage["variants"]
        )
    else:
        variants = campaign.ready_variants(plan, stage_name=stage_name)
    for stage, variant in variants:
        expected_config = campaign.variant_config(base_config, plan, variant)
        loss_variant, _, loss_hook_sha256 = campaign.variant_loss(variant)
        if stage_lock is None:
            stage_definition = campaign.stage_materializer.stage_by_name(
                plan, stage
            )
            family_size = _stage_family_size(stage_definition)
            axis_name, axis_level, conditional_extension_levels = (
                _direct_axis_metadata(stage_definition, expected_config)
            )
        else:
            axis_name = str(variant.get("axis", resolved_stage["axis"]))
            axis_level = variant.get(
                "level",
                campaign.stage_materializer._effective_axis_value(
                    expected_config, axis_name
                ),
            )
        entries.append(
            {
                "stage": stage,
                "experiment": str(variant["experiment"]),
                "kernel_slug": str(variant["kernel_slug"]),
                "role": str(variant.get("role", "candidate")),
                "planned_overrides": dict(variant["overrides"]),
                "expected_config": expected_config,
                "expected_recipe_sha256": campaign.team_builder.canonical_sha256(
                    expected_config
                ),
                "expected_source_sha256": source_sha256,
                "loss_variant": loss_variant,
                "expected_loss_hook_sha256": loss_hook_sha256,
                "expected_run_id": None,
                "expected_iid_predictions_sha256": None,
                "expected_completion_sha256": None,
                "expected_notes": campaign._variant_notes(
                    str(plan["campaign"]),
                    stage,
                    variant,
                    expected_config,
                    stage_lock=stage_lock,
                ),
                "provenance_alias": (
                    None
                    if stage_lock is not None
                    else campaign.variant_provenance_alias(variant, stage=stage)
                ),
                "is_hypothesis": True
                if stage_lock is not None
                else str(variant.get("role", "candidate"))
                != "current_protocol_control",
                "hypothesis_family_size": family_size,
                "axis": axis_name,
                "level": axis_level,
                "conditional_extension_levels": deepcopy(
                    conditional_extension_levels
                ),
            }
        )
        if stage_lock is not None:
            entries[-1].update(common_lock_fields)
            if transition_kind == "conditional_boundary_extension":
                entries[-1]["extension_consumed_axis"] = axis_name
    return entries


def validate_frozen_training_contract(
    report: Mapping[str, Any],
    *,
    experiment: str,
) -> None:
    """Repeat the launcher's frozen sampling/external-weight contract."""
    if (
        report.get("training_sampling") != "none"
        or report.get("training_loss_weighting") != "none"
        or report.get("training_subset") != "all"
        or report.get("original_training_examples") != 306_669
        or report.get("training_unique_coverage_per_epoch") != 1.0
        or any(
            report.get(name) != 1.0
            for name in (
                "training_loss_weight_min",
                "training_loss_weight_median",
                "training_loss_weight_max",
            )
        )
    ):
        raise RuntimeError(
            f"Run {experiment} changed frozen sampling or external "
            "sample-weight semantics"
        )


def completion_row(
    entry: Mapping[str, Any],
    path: Path,
    *,
    baseline_run_id: str,
) -> dict[str, Any]:
    row = dict(entry)
    row.update(
        {
            "completed": False,
            "completion_path": str(path),
            "run_id": None,
            "status": "missing",
            "completed_at_utc": None,
            "epochs": entry["planned_overrides"].get("epochs"),
            "learning_rate": entry["planned_overrides"].get("learning_rate"),
            "weight_decay": None,
            "warmup_ratio": None,
            "label_smoothing": None,
            "max_grad_norm": None,
            "batch_size_per_gpu": None,
            "gradient_accumulation": None,
            "effective_batch": None,
            "seed": None,
            "iid_macro_ap": None,
            "hard_macro_ap": None,
            "ood_macro_ap": None,
            "hard_recall_at_p99": None,
            "hard_roc_auc": None,
            "ood_log_loss": None,
            "iid_delta": None,
            "iid_p_value": None,
            "iid_p_holm_3_splits": None,
            "iid_ci95_low": None,
            "iid_ci95_high": None,
            "iid_delta_vs_anchor": None,
            "iid_p_value_vs_anchor": None,
            "iid_ci95_low_vs_anchor": None,
            "iid_ci95_high_vs_anchor": None,
            "hard_delta": None,
            "ood_delta": None,
            "training_seconds": None,
            "total_pipeline_seconds": None,
            "peak_vram_gib": None,
            "recipe_sha256": None,
            "code_bundle_sha256": None,
            "iid_predictions_sha256": None,
            "sheets_sync_status": None,
        }
    )
    if not path.is_file():
        return row
    expected_completion_sha256 = entry.get("expected_completion_sha256")
    if (
        expected_completion_sha256 is not None
        and campaign.stage_materializer.file_sha256(path)
        != expected_completion_sha256
    ):
        raise RuntimeError(
            f"Reused stage anchor {entry['experiment']} completion SHA-256 differs"
        )
    completion = json.loads(path.read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        raise RuntimeError(f"Run {entry['experiment']} has a non-complete artifact")
    run_id = str(completion.get("run_id", "")).strip()
    if not run_id:
        raise RuntimeError(f"Run {entry['experiment']} has an empty run_id")
    if entry.get("expected_run_id") not in (None, run_id):
        raise RuntimeError(f"Run {entry['experiment']} differs from the locked run_id")
    if completion.get("experiment") != entry["experiment"]:
        raise RuntimeError(
            f"Experiment mismatch in {path}: {completion.get('experiment')!r}"
        )
    if completion.get("experiment_group") != "sft":
        raise RuntimeError(f"Run {entry['experiment']} was not routed to sft_exps")
    comparison = completion.get("baseline_comparison", {})
    if comparison.get("baseline_run_id") != baseline_run_id:
        raise RuntimeError(f"Run {entry['experiment']} used a different baseline")
    if comparison.get("candidate_run_id") != run_id:
        raise RuntimeError(f"Run {entry['experiment']} comparison run_id differs")
    if comparison.get("status") != "ready":
        raise RuntimeError(f"Run {entry['experiment']} comparison is not ready")
    if comparison.get("method") != "paired_component_permutation":
        raise RuntimeError(f"Run {entry['experiment']} used an unexpected test")
    comparison_path = path.parent / "baseline_comparison.json"
    if not comparison_path.is_file():
        raise RuntimeError(f"Run {entry['experiment']} has no comparison artifact")
    comparison_artifact = json.loads(comparison_path.read_text(encoding="utf-8"))
    if comparison_artifact != comparison:
        raise RuntimeError(
            f"Run {entry['experiment']} completion and comparison artifacts differ"
        )
    report = completion.get("training_report", {})
    if not isinstance(report, Mapping):
        raise RuntimeError(f"Run {entry['experiment']} has no training report object")
    validate_frozen_training_contract(
        report,
        experiment=str(entry["experiment"]),
    )
    args = report.get("args", {})
    splits = report.get("validation_splits", {})
    if set(splits) != {"iid", "hard", "ood"}:
        raise RuntimeError(f"Run {entry['experiment']} has incomplete validation")
    if completion.get("frozen_recipe_sha256") != entry["expected_recipe_sha256"]:
        raise RuntimeError(
            f"Run {entry['experiment']} has an unexpected frozen recipe hash"
        )
    if completion.get("code_bundle_sha256") != entry["expected_source_sha256"]:
        raise RuntimeError(
            f"Run {entry['experiment']} has an unexpected embedded source hash"
        )
    if completion.get("initial_checkpoint_ref") != campaign.team_builder.CHECKPOINT_DATASET:
        raise RuntimeError(f"Run {entry['experiment']} used a different checkpoint")
    if (
        completion.get("initial_checkpoint_manifest_sha256")
        != campaign.team_builder.CHECKPOINT_MANIFEST_SHA256
    ):
        raise RuntimeError(
            f"Run {entry['experiment']} used a different checkpoint manifest"
        )
    if completion.get("loss_hook_sha256") != entry["expected_loss_hook_sha256"]:
        raise RuntimeError(
            f"Run {entry['experiment']} did not use the planned frozen loss"
        )
    expected_notes = entry.get("expected_notes")
    completion_notes = completion.get("notes")
    if expected_notes is not None and completion_notes != expected_notes:
        alias = entry.get("provenance_alias")
        if not isinstance(alias, Mapping):
            raise RuntimeError(
                f"Run {entry['experiment']} notes differ from the frozen campaign"
            )
        if entry.get("role") != "current_protocol_control" or not isinstance(
            completion_notes, str
        ):
            raise RuntimeError("Only the protocol control may use a notes alias")
        notes_sha256 = hashlib.sha256(completion_notes.encode("utf-8")).hexdigest()
        if notes_sha256 != alias["accepted_completion_notes_sha256"]:
            raise RuntimeError("Control notes differ from the exact provenance alias")
        try:
            legacy_notes = json.loads(completion_notes)
        except json.JSONDecodeError as error:
            raise RuntimeError("Aliased control notes are not JSON") from error
        if (
            not isinstance(legacy_notes, Mapping)
            or legacy_notes.get("stage") != alias["recorded_stage"]
            or entry.get("stage") != alias["canonical_stage"]
        ):
            raise RuntimeError("Aliased control notes have unexpected stage provenance")
    train_data = completion.get("train_data", {})
    if (
        train_data.get("train_pairs") != 306_669
        or train_data.get("items") != 711_304
        or train_data.get("same_size_as_human_baseline") is not True
    ):
        raise RuntimeError(f"Run {entry['experiment']} changed frozen human data")
    training_configs = list(path.parent.rglob("training_config.json"))
    if len(training_configs) != 1:
        raise RuntimeError(
            f"Run {entry['experiment']} must contain exactly one training_config.json, "
            f"found {training_configs}"
        )
    actual_training_config = json.loads(
        training_configs[0].read_text(encoding="utf-8")
    )
    expected_training_config = entry["expected_config"]
    if set(actual_training_config) != set(expected_training_config):
        raise RuntimeError(
            f"Run {entry['experiment']} training config keys differ from the plan"
        )
    for key, expected in expected_training_config.items():
        if key == "model":
            continue
        if actual_training_config.get(key) != expected:
            raise RuntimeError(
                f"Run {entry['experiment']} training_config has {key}="
                f"{actual_training_config.get(key)!r}, expected {expected!r}"
            )
    planned = entry["planned_overrides"]
    for key, expected in planned.items():
        if key == "model_load_kwargs":
            continue
        if args.get(key) != expected:
            raise RuntimeError(
                f"Run {entry['experiment']} has {key}={args.get(key)!r}, "
                f"expected {expected!r}"
            )
    baseline_metrics = {
        "iid": 0.789388132774931,
        "hard": 0.3655009201486312,
        "ood": 0.6426602624552971,
    }
    for split in ("iid", "hard", "ood"):
        score = _metric(report, split, "macro_average_precision")
        delta = _comparison_metric(
            completion, split, "delta_macro_average_precision"
        )
        raw_p = _comparison_metric(completion, split, "p_value")
        ci_low = _comparison_metric(completion, split, "ci95_low")
        ci_high = _comparison_metric(completion, split, "ci95_high")
        numeric = (score, delta, raw_p, ci_low, ci_high)
        if any(
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric
        ):
            raise RuntimeError(
                f"Run {entry['experiment']} has non-finite {split} statistics"
            )
        if not 0 <= float(score) <= 1 or not 0 <= float(raw_p) <= 1:
            raise RuntimeError(
                f"Run {entry['experiment']} has invalid {split} score/p-value"
            )
        if float(ci_low) > float(ci_high):
            raise RuntimeError(f"Run {entry['experiment']} has reversed {split} CI")
        expected_delta = float(score) - baseline_metrics[split]
        if not math.isclose(float(delta), expected_delta, abs_tol=1e-12):
            raise RuntimeError(
                f"Run {entry['experiment']} has inconsistent {split} delta"
            )
    effective_batch = (
        int(args["batch_size"])
        * 2
        * int(args["gradient_accumulation"])
    )
    row.update(
        {
            "completed": completion.get("status") == "complete",
            "status": completion.get("status"),
            "run_id": run_id,
            "completed_at_utc": completion.get("completed_at_utc"),
            "epochs": args.get("epochs"),
            "learning_rate": args.get("learning_rate"),
            "weight_decay": args.get("weight_decay"),
            "warmup_ratio": args.get("warmup_ratio"),
            "label_smoothing": args.get("label_smoothing"),
            "max_grad_norm": args.get("max_grad_norm"),
            "batch_size_per_gpu": args.get("batch_size"),
            "gradient_accumulation": args.get("gradient_accumulation"),
            "effective_batch": effective_batch,
            "seed": args.get("seed"),
            "iid_macro_ap": _metric(report, "iid", "macro_average_precision"),
            "hard_macro_ap": _metric(report, "hard", "macro_average_precision"),
            "ood_macro_ap": _metric(report, "ood", "macro_average_precision"),
            "hard_recall_at_p99": _metric(
                report, "hard", "recall_at_precision_0_99"
            ),
            "hard_roc_auc": _metric(report, "hard", "roc_auc"),
            "ood_log_loss": _metric(report, "ood", "log_loss"),
            "iid_delta": _comparison_metric(
                completion, "iid", "delta_macro_average_precision"
            ),
            "iid_p_value": _comparison_metric(completion, "iid", "p_value"),
            "iid_p_holm_3_splits": _comparison_metric(
                completion, "iid", "p_value_holm"
            ),
            "iid_ci95_low": _comparison_metric(completion, "iid", "ci95_low"),
            "iid_ci95_high": _comparison_metric(completion, "iid", "ci95_high"),
            "hard_delta": _comparison_metric(
                completion, "hard", "delta_macro_average_precision"
            ),
            "ood_delta": _comparison_metric(
                completion, "ood", "delta_macro_average_precision"
            ),
            "training_seconds": report.get("training_seconds"),
            "total_pipeline_seconds": report.get("total_pipeline_seconds"),
            "peak_vram_gib": max(report.get("peak_vram_gib_by_rank", [float("nan")])),
            "recipe_sha256": completion.get("frozen_recipe_sha256"),
            "code_bundle_sha256": completion.get("code_bundle_sha256"),
            "sheets_sync_status": None,
        }
    )
    sync_path = path.parent / "google_sheets_sync.json"
    if not sync_path.is_file():
        raise RuntimeError(f"Run {entry['experiment']} has no Sheets sync artifact")
    if (path.parent / "sheets_sync_pending.json").exists():
        raise RuntimeError(f"Run {entry['experiment']} Sheets sync is still pending")
    sync = json.loads(sync_path.read_text(encoding="utf-8"))
    if (
        sync.get("status") != "synced"
        or sync.get("run_id") != run_id
        or sync.get("experiment_group") != "sft"
        or sync.get("comparison_sheet") != "sft_exps"
        or sync.get("spreadsheet_id") != qwen_builder.EXPERIMENT_SPREADSHEET_ID
    ):
        raise RuntimeError(
            f"Run {entry['experiment']} was not verifiably synchronized to sft_exps"
        )
    row["sheets_sync_status"] = "synced"
    iid_predictions = list(
        path.parent.rglob("iid_validation_predictions.parquet")
    )
    if len(iid_predictions) != 1:
        raise RuntimeError(
            f"Run {entry['experiment']} must have exactly one IID parquet"
        )
    iid_predictions_sha256 = campaign.stage_materializer.file_sha256(
        iid_predictions[0]
    )
    row["iid_predictions_sha256"] = iid_predictions_sha256
    expected_iid_sha256 = entry.get("expected_iid_predictions_sha256")
    if expected_iid_sha256 is not None:
        if iid_predictions_sha256 != expected_iid_sha256:
            raise RuntimeError(
                f"Stage anchor {entry['experiment']} IID predictions SHA-256 differs"
            )
    return row


def _iid_prediction_path(artifacts_dir: Path, kernel_slug: str) -> Path:
    matches = list(
        (artifacts_dir / kernel_slug).rglob("iid_validation_predictions.parquet")
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one IID prediction parquet for {kernel_slug!r}, found {matches}"
        )
    return matches[0]


def add_anchor_relative_iid(
    frame: pd.DataFrame,
    *,
    artifacts_dir: Path,
    permutations: int = campaign.team_builder.SIGNIFICANCE_PERMUTATIONS,
    bootstrap_resamples: int = campaign.team_builder.SIGNIFICANCE_BOOTSTRAP_RESAMPLES,
    seed: int = campaign.team_builder.SIGNIFICANCE_SEED,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Recompute IID statistics against each stage's frozen selection anchor."""
    frame = frame.copy()
    for column in (
        "iid_delta_vs_anchor",
        "iid_p_value_vs_anchor",
        "iid_ci95_low_vs_anchor",
        "iid_ci95_high_vs_anchor",
    ):
        if column not in frame:
            frame[column] = math.nan
    for stage, part in frame.groupby("stage", sort=False):
        declared_anchors = part[part["role"] == "stage_anchor"]
        if declared_anchors.empty:
            declared_anchors = part[
                part["role"] == "current_protocol_control"
            ]
        if declared_anchors.empty:
            continue
        if len(declared_anchors) != 1:
            raise RuntimeError(f"Stage {stage} must declare exactly one stage anchor")
        anchor_index = declared_anchors.index[0]
        if not bool(frame.at[anchor_index, "completed"]):
            continue
        anchor_path = _iid_prediction_path(
            artifacts_dir,
            str(frame.at[anchor_index, "kernel_slug"]),
        )
        anchor_predictions = read_prediction_artifact(anchor_path)
        frame.at[anchor_index, "iid_delta_vs_anchor"] = 0.0
        frame.at[anchor_index, "iid_p_value_vs_anchor"] = 1.0
        frame.at[anchor_index, "iid_ci95_low_vs_anchor"] = 0.0
        frame.at[anchor_index, "iid_ci95_high_vs_anchor"] = 0.0
        hypotheses = part[
            part.get(
                "is_hypothesis",
                ~part["role"].isin(
                    {"stage_anchor", "current_protocol_control"}
                ),
            ).astype(bool)
        ]
        for candidate_index in hypotheses.index:
            if not bool(frame.at[candidate_index, "completed"]):
                continue
            candidate_path = _iid_prediction_path(
                artifacts_dir,
                str(frame.at[candidate_index, "kernel_slug"]),
            )
            comparison = cached_anchor_comparison(
                anchor_path=anchor_path,
                candidate_path=candidate_path,
                anchor_predictions=anchor_predictions,
                permutations=permutations,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed,
                cache_dir=cache_dir,
            )
            anchor_ap = float(frame.at[anchor_index, "iid_macro_ap"])
            candidate_ap = float(frame.at[candidate_index, "iid_macro_ap"])
            if not math.isclose(
                float(comparison["baseline_macro_average_precision"]),
                anchor_ap,
                abs_tol=1e-12,
            ) or not math.isclose(
                float(comparison["candidate_macro_average_precision"]),
                candidate_ap,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"Anchor-relative IID metrics disagree with reports in stage {stage}"
                )
            frame.at[candidate_index, "iid_delta_vs_anchor"] = comparison[
                "delta_macro_average_precision"
            ]
            frame.at[candidate_index, "iid_p_value_vs_anchor"] = comparison[
                "p_value"
            ]
            frame.at[candidate_index, "iid_ci95_low_vs_anchor"] = comparison[
                "ci95_low"
            ]
            frame.at[candidate_index, "iid_ci95_high_vs_anchor"] = comparison[
                "ci95_high"
            ]
    return frame


def _read_anchor_comparison_cache(
    path: Path,
    *,
    expected_key: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid anchor-comparison cache: {path}") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Anchor-comparison cache is not an object: {path}")
    expected_text = campaign.stage_materializer.canonical_json_dumps(payload) + "\n"
    if path.read_text(encoding="utf-8") != expected_text:
        raise RuntimeError(f"Anchor-comparison cache is not canonical JSON: {path}")
    unhashed = dict(payload)
    stored_sha = unhashed.pop("cache_payload_sha256", None)
    if stored_sha != campaign.stage_materializer.canonical_sha256(unhashed):
        raise RuntimeError(f"Anchor-comparison cache SHA-256 is invalid: {path}")
    if payload.get("key") != dict(expected_key):
        raise RuntimeError(f"Anchor-comparison cache key differs: {path}")
    comparison = payload.get("comparison")
    if not isinstance(comparison, Mapping):
        raise RuntimeError(f"Anchor-comparison cache has no comparison: {path}")
    if (
        comparison.get("permutations") != expected_key["permutations"]
        or comparison.get("bootstrap_resamples")
        != expected_key["bootstrap_resamples"]
        or comparison.get("seed") != expected_key["seed"]
    ):
        raise RuntimeError(f"Anchor-comparison cache protocol differs: {path}")
    return dict(comparison)


def cached_anchor_comparison(
    *,
    anchor_path: Path,
    candidate_path: Path,
    anchor_predictions: pd.DataFrame,
    permutations: int,
    bootstrap_resamples: int,
    seed: int,
    cache_dir: Path | None,
) -> dict[str, Any]:
    key = {
        "schema_version": 1,
        "anchor_iid_predictions_sha256": campaign.stage_materializer.file_sha256(
            anchor_path
        ),
        "candidate_iid_predictions_sha256": campaign.stage_materializer.file_sha256(
            candidate_path
        ),
        "comparison_code_sha256": campaign.stage_materializer.file_sha256(
            ROOT / "src" / "experiment_significance.py"
        ),
        "permutations": permutations,
        "bootstrap_resamples": bootstrap_resamples,
        "seed": seed,
    }
    if cache_dir is None:
        return compare_prediction_frames(
            anchor_predictions,
            read_prediction_artifact(candidate_path),
            permutations=permutations,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    key_sha256 = campaign.stage_materializer.canonical_sha256(key)
    cache_path = cache_dir / f"{key_sha256}.json"
    if cache_path.exists():
        return _read_anchor_comparison_cache(cache_path, expected_key=key)
    comparison = compare_prediction_frames(
        anchor_predictions,
        read_prediction_artifact(candidate_path),
        permutations=permutations,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "key": key,
        "comparison": comparison,
    }
    payload["cache_payload_sha256"] = campaign.stage_materializer.canonical_sha256(
        payload
    )
    serialized = campaign.stage_materializer.canonical_json_dumps(payload) + "\n"
    try:
        descriptor = os.open(
            cache_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError:
        return _read_anchor_comparison_cache(cache_path, expected_key=key)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    return comparison


def add_stage_holm(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["iid_p_holm_stage"] = math.nan
    for stage, part in frame.groupby("stage", sort=False):
        # Keep the family size fixed at the number of planned hypotheses. Missing
        # runs receive p=1 only for the interim calculation; their displayed value
        # stays blank. This avoids anti-conservative p-values early in a campaign.
        if "is_hypothesis" in part:
            hypotheses = part[part["is_hypothesis"].astype(bool)]
        else:
            hypotheses = part[part["role"] != "current_protocol_control"]
        family_size = len(hypotheses)
        if "hypothesis_family_size" in part:
            declared_sizes = {
                int(value)
                for value in part["hypothesis_family_size"].dropna().tolist()
            }
            if len(declared_sizes) > 1:
                raise RuntimeError(
                    f"Stage {stage} has inconsistent hypothesis families"
                )
            if declared_sizes:
                family_size = declared_sizes.pop()
        if family_size < len(hypotheses):
            raise RuntimeError(
                f"Stage {stage} family is smaller than its materialized hypotheses"
            )
        use_anchor_p = (
            "iid_p_value_vs_anchor" in part
            and bool(
                part["role"].isin(
                    {"stage_anchor", "current_protocol_control"}
                ).any()
            )
            and bool(part["iid_p_value_vs_anchor"].notna().any())
        )
        p_column = "iid_p_value_vs_anchor" if use_anchor_p else "iid_p_value"
        planned_p_values = [
            float(getattr(row, p_column))
            if bool(row.completed) and pd.notna(getattr(row, p_column))
            else 1.0
            for row in hypotheses.itertuples()
        ]
        planned_p_values.extend([1.0] * (family_size - len(hypotheses)))
        adjusted = holm_adjust(planned_p_values)
        for index, value in zip(
            hypotheses.index,
            adjusted[: len(hypotheses)],
            strict=True,
        ):
            if bool(frame.at[index, "completed"]) and pd.notna(
                frame.at[index, p_column]
            ):
                frame.at[index, "iid_p_holm_stage"] = value
    return frame


def _machine_boundary_extension_evidence(
    part: pd.DataFrame,
    *,
    selection_pool: pd.DataFrame,
    best: pd.Series,
    tie_margin: float,
) -> list[dict[str, Any]] | None:
    """Use frozen axis/level declarations; return ``None`` for legacy rows."""
    if "axis" not in part or "level" not in part:
        return None
    axes = {
        str(value)
        for value in part["axis"].tolist()
        if value is not None and not (isinstance(value, float) and math.isnan(value))
    }
    if not axes:
        return None
    if len(axes) != 1:
        raise RuntimeError(f"Stage has inconsistent boundary axes: {sorted(axes)}")
    axis_name = axes.pop()
    if "extension_consumed_axis" in part and any(
        value == axis_name for value in part["extension_consumed_axis"].tolist()
    ):
        return []

    declared: list[Any] = []
    if "conditional_extension_levels" in part:
        for raw in part["conditional_extension_levels"].tolist():
            if raw is None or (isinstance(raw, float) and math.isnan(raw)):
                continue
            if not isinstance(raw, list):
                raise RuntimeError(
                    "conditional_extension_levels must contain frozen lists"
                )
            for level in raw:
                if not any(
                    campaign.stage_materializer.canonical_json_dumps(level)
                    == campaign.stage_materializer.canonical_json_dumps(existing)
                    for existing in declared
                ):
                    declared.append(level)
    if not declared:
        return []

    raw_levels = part["level"].tolist()
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in [*raw_levels, *declared]
    ):
        raise RuntimeError("Boundary-extension axis levels must be finite numbers")
    levels = sorted({float(value) for value in raw_levels})
    if len(levels) < 2:
        return []
    best_level = float(best["level"])
    direction: str | None = None
    neighbor_level: float | None = None
    extension_level: Any = None
    lower_extensions = [level for level in declared if float(level) < levels[0]]
    higher_extensions = [level for level in declared if float(level) > levels[-1]]
    if best_level == levels[0] and lower_extensions:
        direction = "lower"
        neighbor_level = levels[1]
        extension_level = max(lower_extensions, key=float)
    elif best_level == levels[-1] and higher_extensions:
        direction = "higher"
        neighbor_level = levels[-2]
        extension_level = min(higher_extensions, key=float)
    if direction is None or neighbor_level is None:
        return []
    neighbor = selection_pool[
        selection_pool["level"].map(float) == neighbor_level
    ]
    if len(neighbor) != 1:
        raise RuntimeError(
            f"Stage axis {axis_name!r} has {len(neighbor)} nearest-interior rows"
        )
    gain = float(best["iid_macro_ap"] - neighbor.iloc[0]["iid_macro_ap"])
    if gain <= tie_margin:
        return []
    return [
        {
            "axis": axis_name,
            "direction": direction,
            "level": extension_level,
            "gain_over_nearest_interior": gain,
        }
    ]


def stage_summary(
    frame: pd.DataFrame,
    *,
    tie_margin: float,
    control_gate: Mapping[str, Any],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for stage, part in frame.groupby("stage", sort=False):
        completed = part[part["completed"]].sort_values(
            "iid_macro_ap", ascending=False
        )
        extension_lock = bool(
            "stage_lock_transition_kind" in part
            and part["stage_lock_transition_kind"]
            .eq("conditional_boundary_extension")
            .any()
        )
        locked_stage = bool(part["role"].eq("stage_anchor").any()) or extension_lock
        if locked_stage and "is_hypothesis" in part:
            expected = int(part["is_hypothesis"].astype(bool).sum())
            completed_count = int(
                (part["completed"] & part["is_hypothesis"].astype(bool)).sum()
            )
        else:
            expected = len(part)
            completed_count = len(completed)
        summary: dict[str, Any] = {
            "expected_runs": expected,
            "completed_runs": completed_count,
            "complete": bool(part["completed"].all()),
            "decision_status": "pending",
        }
        if locked_stage:
            summary["expected_entries_including_reused_anchor"] = len(part)
            summary["completed_entries_including_reused_anchor"] = len(completed)
        if completed.empty:
            summaries[str(stage)] = summary
            continue
        declared_anchors = part[part["role"] == "stage_anchor"]
        if not declared_anchors.empty:
            anchors = completed[completed["role"] == "stage_anchor"]
            if anchors.empty:
                summary["control_gate"] = "pending"
                summary["anchor_reuse"] = "pending"
                summaries[str(stage)] = summary
                continue
            if len(anchors) != 1 or len(declared_anchors) != 1:
                raise RuntimeError(f"Stage {stage} must have exactly one stage anchor")
            selection_anchor = anchors.iloc[0]
            summary.update(
                {
                    "control_gate": "passed",
                    "anchor_reuse": "validated",
                    "selection_anchor_experiment": selection_anchor["experiment"],
                    "selection_anchor_run_id": selection_anchor["run_id"],
                    "selection_anchor_iid_macro_ap": float(
                        selection_anchor["iid_macro_ap"]
                    ),
                }
            )
        else:
            anchors = completed[completed["role"] == control_gate["role"]]
            if anchors.empty:
                summary["control_gate"] = "pending"
                summaries[str(stage)] = summary
                continue
            if len(anchors) != 1:
                raise RuntimeError(f"Stage {stage} has multiple completed controls")
            selection_anchor = anchors.iloc[0]
            runtime_drift = abs(
                float(selection_anchor["total_pipeline_seconds"])
                / float(control_gate["baseline_total_pipeline_seconds"])
                - 1.0
            )
            control_checks = {
                "iid_delta": abs(float(selection_anchor["iid_delta"]))
                <= float(control_gate["max_abs_iid_delta"]),
                "hard_delta": abs(float(selection_anchor["hard_delta"]))
                <= float(control_gate["max_abs_hard_delta"]),
                "ood_delta": abs(float(selection_anchor["ood_delta"]))
                <= float(control_gate["max_abs_ood_delta"]),
                "runtime_drift": runtime_drift
                <= float(control_gate["max_runtime_relative_drift"]),
            }
            control_passed = all(control_checks.values())
            summary.update(
                {
                    "control_gate": "passed" if control_passed else "failed",
                    "control_experiment": selection_anchor["experiment"],
                    "control_checks": control_checks,
                    "control_runtime_relative_drift": runtime_drift,
                }
            )
            if not control_passed:
                summary["decision_status"] = "invalid_control"
                summaries[str(stage)] = summary
                continue
        if "is_hypothesis" in completed:
            candidates = completed[completed["is_hypothesis"].astype(bool)]
        else:
            candidates = completed[
                ~completed["role"].isin(
                    {control_gate["role"], "stage_anchor"}
                )
            ]
        if candidates.empty:
            summaries[str(stage)] = summary
            continue
        best = candidates.iloc[0]
        if not summary["complete"]:
            summaries[str(stage)] = summary
            continue

        gain_over_anchor = float(
            best["iid_macro_ap"] - selection_anchor["iid_macro_ap"]
        )
        challenger_selected = gain_over_anchor > tie_margin
        recommended = best if challenger_selected else selection_anchor
        selection_pool = pd.concat([anchors, candidates], ignore_index=False)
        practical_best = float(selection_pool["iid_macro_ap"].max())
        shortlist = selection_pool[
            selection_pool["iid_macro_ap"] >= practical_best - tie_margin
        ]
        extension_evidence = _machine_boundary_extension_evidence(
            part,
            selection_pool=selection_pool,
            best=best,
            tie_margin=tie_margin,
        )
        if extension_evidence is None:
            # Backward-compatible fallback for old summary rows that predate the
            # frozen axis/level fields. New coordinate stages never enter here.
            lrs = sorted(
                part["planned_overrides"]
                .map(lambda item: item.get("learning_rate"))
                .dropna()
                .unique()
            )
            epochs = sorted(
                part["planned_overrides"]
                .map(lambda item: item.get("epochs"))
                .dropna()
                .unique()
            )
            extension_evidence = []
            if lrs and best["learning_rate"] in {lrs[0], lrs[-1]} and len(lrs) > 1:
                neighbor_lr = lrs[1] if best["learning_rate"] == lrs[0] else lrs[-2]
                neighbor = selection_pool[
                    (selection_pool["epochs"] == best["epochs"])
                    & (selection_pool["learning_rate"] == neighbor_lr)
                ]
                if len(neighbor) == 1:
                    gain = float(
                        best["iid_macro_ap"] - neighbor.iloc[0]["iid_macro_ap"]
                    )
                    if gain > tie_margin:
                        extension_evidence.append(
                            {
                                "axis": "learning_rate",
                                "direction": (
                                    "lower"
                                    if best["learning_rate"] == lrs[0]
                                    else "higher"
                                ),
                                "gain_over_nearest_interior": gain,
                            }
                        )
            if epochs and best["epochs"] == epochs[-1] and len(epochs) > 1:
                neighbor = selection_pool[
                    (selection_pool["learning_rate"] == best["learning_rate"])
                    & (selection_pool["epochs"] == epochs[-2])
                ]
                if len(neighbor) == 1:
                    gain = float(
                        best["iid_macro_ap"] - neighbor.iloc[0]["iid_macro_ap"]
                    )
                    if gain > tie_margin:
                        extension_evidence.append(
                            {
                                "axis": "epochs",
                                "direction": "higher",
                                "gain_over_nearest_interior": gain,
                            }
                        )
        if not challenger_selected:
            extension_evidence = []
        recommended_extension = (
            max(
                extension_evidence,
                key=lambda item: item["gain_over_nearest_interior"],
            )
            if extension_evidence
            else None
        )
        summary.update(
            {
                "decision_status": "ready",
                "recommendation": (
                    "select_challenger"
                    if challenger_selected
                    else "retain_current_recipe"
                ),
                "recommended_experiment": recommended["experiment"],
                "recommended_run_id": recommended["run_id"],
                "recommended_iid_macro_ap": float(recommended["iid_macro_ap"]),
                "best_candidate_experiment": best["experiment"],
                "best_candidate_run_id": best["run_id"],
                "best_candidate_iid_macro_ap": float(best["iid_macro_ap"]),
                "best_candidate_iid_delta": float(
                    best["iid_delta_vs_anchor"]
                    if "iid_delta_vs_anchor" in best
                    and pd.notna(best["iid_delta_vs_anchor"])
                    else best["iid_delta"]
                ),
                "best_candidate_iid_p_holm_stage": float(
                    best["iid_p_holm_stage"]
                ),
                "best_candidate_gain_over_anchor": gain_over_anchor,
                "challenger_selected": challenger_selected,
                "practical_shortlist": shortlist["experiment"].tolist(),
                "boundary_extension_evidence": extension_evidence,
                "needs_boundary_extension": bool(extension_evidence),
                "recommended_extension": recommended_extension,
            }
        )
        if not locked_stage:
            summary["best_candidate_gain_over_control"] = gain_over_anchor
        summaries[str(stage)] = summary
    return summaries


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [
        "experiment",
        "role",
        "epochs",
        "learning_rate",
        "iid_macro_ap",
        "iid_delta",
        "iid_delta_vs_anchor",
        "iid_p_value_vs_anchor",
        "iid_p_holm_stage",
        "hard_macro_ap",
        "ood_macro_ap",
        "training_seconds",
        "status",
    ]
    view = frame.reindex(columns=columns).copy()
    for column in (
        "learning_rate",
        "iid_macro_ap",
        "iid_delta",
        "iid_delta_vs_anchor",
        "iid_p_value_vs_anchor",
        "iid_p_holm_stage",
        "hard_macro_ap",
        "ood_macro_ap",
        "training_seconds",
    ):
        view[column] = view[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.6g}"
        )
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in view.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def report_text(
    *,
    plan: Mapping[str, Any],
    frame: pd.DataFrame,
    summary: Mapping[str, Any],
) -> str:
    table = markdown_table(frame)
    completed = int(frame["completed"].sum())
    return "\n".join(
        [
            f"# {plan['campaign']}",
            "",
            f"Completed entries: **{completed}/{len(frame)}**.",
            (
                "Primary selection metric: IID macro AP. Hard and OOD are "
                "reported only as diagnostics."
            ),
            (
                "`iid_p_holm_stage` uses the full planned hypothesis family, "
                "including reserved conditional extensions."
            ),
            (
                "Selection statistics are recomputed from local IID parquets "
                "against the frozen stage anchor (the exact control for the "
                "initial LR stage)."
            ),
            "",
            table,
            "",
            "## Stage decisions",
            "",
            "```json",
            json.dumps(summary, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _json_cell(value: Any) -> Any:
    """Preserve frozen JSON containers while normalizing scalar pandas NA."""
    if isinstance(value, (Mapping, list)):
        return value
    missing = pd.isna(value)
    if not hasattr(missing, "__len__") and bool(missing):
        return None
    return value


def write_stage_snapshot(
    output_dir: Path,
    *,
    effective_stage: str,
    files: Mapping[str, str],
) -> Path:
    """Update an in-progress stage, then freeze its terminal ready decision."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9_]*(?:__[a-z0-9_]+)?", effective_stage):
        raise RuntimeError(f"Unsafe effective stage name: {effective_stage!r}")
    stage_dir = output_dir / "stages" / effective_stage
    summary_path = stage_dir / "summary.json"
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Existing stage snapshot is invalid: {summary_path}"
            ) from error
        decision = existing.get("stages", {}).get(effective_stage, {})
        finalized = (
            decision.get("complete") is True
            and decision.get("decision_status") == "ready"
            and decision.get("needs_boundary_extension") is not True
        )
        if finalized:
            for name, content in files.items():
                path = stage_dir / name
                if not path.is_file() or path.read_text(encoding="utf-8") != content:
                    raise RuntimeError(
                        f"Completed stage snapshot is immutable and differs: {path}"
                    )
            return stage_dir
    # Commit summary.json last: complete=true is the immutable snapshot marker.
    for name, content in files.items():
        if name != "summary.json":
            _atomic_write_text(stage_dir / name, content)
    _atomic_write_text(summary_path, files["summary.json"])
    return stage_dir


def load_adaptive_prerequisite_closure(
    *,
    plan: Mapping[str, Any],
    root_lock: Mapping[str, Any],
    root_lock_path: Path,
    base_config: Mapping[str, Any],
) -> list[tuple[dict[str, Any], Path]]:
    """Validate every referenced lock from its own fixed local authority."""
    reloaded_root = campaign.load_campaign_lock(
        root_lock_path,
        plan=plan,
        base_config=base_config,
    )
    if reloaded_root != dict(root_lock):
        raise RuntimeError(
            "Adaptive root lock object differs from its fixed local lock path"
        )
    ordered: list[tuple[dict[str, Any], Path]] = []
    by_sha: dict[str, Path] = {}
    visiting: set[str] = set()

    def visit(lock: Mapping[str, Any], path: Path) -> None:
        lock_sha = str(lock["lock_payload_sha256"])
        resolved_path = path.resolve(strict=True)
        previous_path = by_sha.get(lock_sha)
        if previous_path is not None:
            if previous_path != resolved_path:
                raise RuntimeError(
                    "One prerequisite payload SHA is referenced from two paths"
                )
            return
        if lock_sha in visiting:
            raise RuntimeError("Adaptive prerequisite closure contains a cycle")
        visiting.add(lock_sha)
        prerequisites = lock.get("prerequisites", [])
        if lock.get("schema_version") == 2:
            if not isinstance(prerequisites, list):
                raise RuntimeError("Adaptive prerequisites are malformed")
            for reference in prerequisites:
                if not isinstance(reference, Mapping):
                    raise RuntimeError("Adaptive prerequisite reference is malformed")
                prerequisite_path = Path(str(reference["lock_path"]))
                prerequisite = campaign.load_campaign_lock(
                    prerequisite_path,
                    plan=plan,
                    base_config=base_config,
                )
                if (
                    prerequisite.get("lock_payload_sha256")
                    != reference.get("lock_payload_sha256")
                    or prerequisite.get("schema_version")
                    != reference.get("schema_version")
                    or prerequisite.get("kind") != reference.get("kind")
                ):
                    raise RuntimeError(
                        "Adaptive prerequisite differs from its frozen reference"
                    )
                visit(prerequisite, prerequisite_path)
        visiting.remove(lock_sha)
        by_sha[lock_sha] = resolved_path
        ordered.append((dict(lock), resolved_path))

    visit(reloaded_root, root_lock_path)
    return ordered


def _adaptive_origin_row(origin: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(origin["resolved_config"])
    completion = json.loads(
        Path(str(origin["completion_artifact_path"])).read_text(encoding="utf-8")
    )
    report = completion.get("training_report", {})
    splits = report.get("validation_splits", {}) if isinstance(report, Mapping) else {}
    return {
        "stage": str(origin["origin_effective_stage"]),
        "effective_stage": str(origin["origin_effective_stage"]),
        "experiment": str(origin["experiment"]),
        "kernel_slug": str(origin["kernel_slug"]),
        "role": str(origin["source_role"]),
        "is_hypothesis": bool(origin["source_is_hypothesis"]),
        "completed": True,
        "status": "complete",
        "run_id": str(origin["run_id"]),
        "seed": int(config["seed"]),
        "loss_variant": str(origin["loss_variant"]),
        "loss_hook_sha256": str(origin["loss_hook_sha256"]),
        "iid_macro_ap": float(origin["iid_macro_ap"]),
        "hard_macro_ap": _metric(report, "hard", "macro_average_precision"),
        "ood_macro_ap": _metric(report, "ood", "macro_average_precision"),
        "total_pipeline_seconds": (
            report.get("total_pipeline_seconds")
            if isinstance(report, Mapping)
            else None
        ),
        "resolved_config": config,
        "recipe_sha256": str(origin["recipe_sha256"]),
        "recipe_family_sha256": str(origin["recipe_family_sha256"]),
        "code_bundle_sha256": str(origin["code_bundle_sha256"]),
        "expected_source_sha256": str(origin["expected_source_sha256"]),
        "iid_predictions_sha256": str(origin["iid_predictions_sha256"]),
        "completion_sha256": str(origin["completion_sha256"]),
        "completion_path": str(origin["completion_artifact_path"]),
        "expected_notes_sha256": str(origin["completion_notes_sha256"]),
        "origin_id": str(origin["origin_id"]),
        "reused_origin": True,
        "iid_delta": None,
        "iid_p_value": None,
        "iid_p_holm_3_splits": None,
        "iid_ci95_low": None,
        "iid_ci95_high": None,
        "iid_delta_vs_anchor": None,
        "iid_p_value_vs_anchor": None,
        "iid_ci95_low_vs_anchor": None,
        "iid_ci95_high_vs_anchor": None,
        "iid_p_holm_stage": None,
        "sheets_sync_status": None,
        "validation_splits_present": sorted(splits),
    }


def adaptive_run_rows(
    *,
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    artifacts_dir: Path,
) -> list[dict[str, Any]]:
    entries = adaptive_execution_entries(plan, lock)
    origin_by_experiment = {
        str(origin["experiment"]): origin for origin in lock.get("origins", [])
    }
    rows: list[dict[str, Any]] = []
    for entry in entries:
        experiment = str(entry["experiment"])
        if entry["reused_origin"]:
            origin = origin_by_experiment[experiment]
            expected_completion = (
                artifacts_dir.resolve()
                / str(origin["kernel_slug"])
                / "notebook_completed.json"
            ).resolve(strict=True)
            if Path(str(origin["completion_artifact_path"])) != expected_completion:
                raise RuntimeError(
                    f"Reused origin {experiment!r} is outside --artifacts-dir"
                )
            rows.append(_adaptive_origin_row(origin))
            continue
        path = completion_path(artifacts_dir, str(entry["kernel_slug"]))
        row = completion_row(
            entry,
            path,
            baseline_run_id=str(plan["baseline_run_id"]),
        )
        row.update(
            {
                "effective_stage": str(lock["effective_stage"]),
                # Missing completion artifacts still need their frozen seed so
                # matched-seed confirmation can remain pending instead of
                # crashing while it builds the group ledger.
                "seed": int(entry["expected_config"]["seed"]),
                "resolved_config": deepcopy(entry["expected_config"]),
                "recipe_family_sha256": entry["variant"].get(
                    "expected_recipe_family_sha256"
                ),
                "loss_hook_sha256": entry["expected_loss_hook_sha256"],
                "expected_source_sha256": entry["expected_source_sha256"],
                "expected_notes_sha256": entry["expected_notes_sha256"],
                "origin_id": None,
                "origin_ids": deepcopy(entry["origin_ids"]),
                "family_id": entry["family_id"],
                "hypothesis_family_size": entry["hypothesis_family_size"],
                "reused_origin": False,
            }
        )
        if row["completed"]:
            iid_path = _iid_prediction_path(
                artifacts_dir, str(entry["kernel_slug"])
            )
            row["iid_predictions_sha256"] = (
                campaign.stage_materializer.file_sha256(iid_path)
            )
        rows.append(row)
    completed_by_run_id: dict[str, dict[str, Any]] = {}
    missing_experiments: set[str] = set()
    unique_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["completed"]:
            run_id = str(row["run_id"])
            previous = completed_by_run_id.get(run_id)
            if previous is not None:
                if previous != row:
                    raise RuntimeError(
                        f"Completed run_id {run_id!r} has conflicting rows"
                    )
                continue
            completed_by_run_id[run_id] = row
        else:
            experiment = str(row["experiment"])
            if experiment in missing_experiments:
                raise RuntimeError(f"Duplicate missing run {experiment!r}")
            missing_experiments.add(experiment)
        unique_rows.append(row)
    return unique_rows


def _adaptive_comparison(
    *,
    anchor: Mapping[str, Any],
    candidate: Mapping[str, Any],
    artifacts_dir: Path,
    cache_dir: Path,
) -> dict[str, Any] | None:
    if not anchor.get("completed") or not candidate.get("completed"):
        return None
    anchor_path = _iid_prediction_path(artifacts_dir, str(anchor["kernel_slug"]))
    candidate_path = _iid_prediction_path(
        artifacts_dir, str(candidate["kernel_slug"])
    )
    comparison = cached_anchor_comparison(
        anchor_path=anchor_path,
        candidate_path=candidate_path,
        anchor_predictions=read_prediction_artifact(anchor_path),
        permutations=campaign.team_builder.SIGNIFICANCE_PERMUTATIONS,
        bootstrap_resamples=campaign.team_builder.SIGNIFICANCE_BOOTSTRAP_RESAMPLES,
        seed=campaign.team_builder.SIGNIFICANCE_SEED,
        cache_dir=cache_dir,
    )
    if not math.isclose(
        float(comparison["baseline_macro_average_precision"]),
        float(anchor["iid_macro_ap"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        float(comparison["candidate_macro_average_precision"]),
        float(candidate["iid_macro_ap"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Adaptive anchor-relative IID metrics disagree")
    return dict(comparison)


def adaptive_hypothesis_projections(
    *,
    closure: Sequence[tuple[Mapping[str, Any], Path]],
    rows: Sequence[Mapping[str, Any]],
    artifacts_dir: Path,
    cache_dir: Path,
) -> list[dict[str, Any]]:
    by_experiment = {str(row["experiment"]): row for row in rows}
    schema_v2 = {
        str(lock["mode"]): lock
        for lock, _ in closure
        if lock.get("schema_version") == 2
    }
    origin_by_id: dict[str, Mapping[str, Any]] = {}
    for lock in schema_v2.values():
        for origin in lock.get("origins", []):
            origin_by_id[str(origin["origin_id"])] = origin
    projections: list[dict[str, Any]] = []

    primary = schema_v2.get("loss_primary")
    if primary is not None:
        anchor_origin = origin_by_id[str(primary["family"]["anchor_origin_id"])]
        candidates = list(primary["resolved_stage"]["variants"])
        overlay = schema_v2.get("loss_overlay")
        if overlay is not None and overlay.get("execution_status") == "runnable":
            candidates.extend(overlay["resolved_stage"]["variants"])
        family_id = str(primary["family"]["family_id"])
        maximum = int(primary["family"]["maximum_hypotheses"])
        if maximum != 5:
            raise RuntimeError("Primary loss family must reserve exactly five hypotheses")
        if overlay is not None and (
            overlay["family"]["family_id"] != family_id
            or int(overlay["family"]["maximum_hypotheses"]) != maximum
        ):
            raise RuntimeError("Loss overlay differs from the frozen primary family")
        anchor_row = by_experiment[str(anchor_origin["experiment"])]
        candidate_rows = []
        raw_p_values = []
        comparisons = []
        for variant in candidates:
            row = by_experiment[str(variant["experiment"])]
            comparison = _adaptive_comparison(
                anchor=anchor_row,
                candidate=row,
                artifacts_dir=artifacts_dir,
                cache_dir=cache_dir,
            )
            comparisons.append(comparison)
            raw_p_values.append(
                float(comparison["p_value"]) if comparison is not None else 1.0
            )
            candidate_rows.append((variant, row))
        adjusted = holm_adjust(
            [*raw_p_values, *([1.0] * (maximum - len(raw_p_values)))]
        )
        projections.append(
            {
                "family_id": family_id,
                "correction": "holm",
                "maximum_hypotheses": maximum,
                "anchor": {
                    "experiment": anchor_row["experiment"],
                    "run_id": anchor_row["run_id"],
                },
                "candidates": [
                    {
                        "experiment": row["experiment"],
                        "run_id": row["run_id"],
                        "loss_variant": variant["loss_variant"],
                        "iid_delta_vs_anchor": (
                            comparison["delta_macro_average_precision"]
                            if comparison is not None
                            else None
                        ),
                        "iid_p_value_vs_anchor": (
                            comparison["p_value"] if comparison is not None else None
                        ),
                        "iid_ci95_low_vs_anchor": (
                            comparison["ci95_low"] if comparison is not None else None
                        ),
                        "iid_ci95_high_vs_anchor": (
                            comparison["ci95_high"] if comparison is not None else None
                        ),
                        "iid_p_holm_family": (
                            adjusted[index] if comparison is not None else None
                        ),
                    }
                    for index, ((variant, row), comparison) in enumerate(
                        zip(candidate_rows, comparisons, strict=True)
                    )
                ],
                "reserved_p_equals_1": maximum - len(candidates),
            }
        )

    refine = schema_v2.get("loss_lr_refine")
    if refine is not None:
        anchor_origin = origin_by_id[str(refine["family"]["anchor_origin_id"])]
        anchor_row = by_experiment[str(anchor_origin["experiment"])]
        candidates = list(refine["resolved_stage"]["variants"])
        raw_p_values = []
        comparisons = []
        for variant in candidates:
            comparison = _adaptive_comparison(
                anchor=anchor_row,
                candidate=by_experiment[str(variant["experiment"])],
                artifacts_dir=artifacts_dir,
                cache_dir=cache_dir,
            )
            comparisons.append(comparison)
            raw_p_values.append(
                float(comparison["p_value"]) if comparison is not None else 1.0
            )
        maximum = int(refine["family"]["maximum_hypotheses"])
        if maximum != 2:
            raise RuntimeError("Loss LR-refine family must reserve two hypotheses")
        adjusted = holm_adjust(
            [*raw_p_values, *([1.0] * (maximum - len(raw_p_values)))]
        )
        projections.append(
            {
                "family_id": str(refine["family"]["family_id"]),
                "correction": "holm",
                "maximum_hypotheses": maximum,
                "anchor": {
                    "experiment": anchor_row["experiment"],
                    "run_id": anchor_row["run_id"],
                },
                "candidates": [
                    {
                        "experiment": variant["experiment"],
                        "run_id": by_experiment[str(variant["experiment"])]["run_id"],
                        "loss_variant": variant["loss_variant"],
                        "iid_delta_vs_anchor": (
                            comparison["delta_macro_average_precision"]
                            if comparison is not None
                            else None
                        ),
                        "iid_p_value_vs_anchor": (
                            comparison["p_value"] if comparison is not None else None
                        ),
                        "iid_ci95_low_vs_anchor": (
                            comparison["ci95_low"] if comparison is not None else None
                        ),
                        "iid_ci95_high_vs_anchor": (
                            comparison["ci95_high"] if comparison is not None else None
                        ),
                        "iid_p_holm_family": (
                            adjusted[index] if comparison is not None else None
                        ),
                    }
                    for index, (variant, comparison) in enumerate(
                        zip(candidates, comparisons, strict=True)
                    )
                ],
                "reserved_p_equals_1": maximum - len(candidates),
            }
        )
    return projections


def _confirmation_runtime_gate(
    *,
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    runtime_check_path: Path | None,
    selected_recipe_group_id: str | None,
) -> dict[str, Any]:
    group_by_id = {
        str(group["recipe_group_id"]): group
        for group in lock["resolved_stage"]["recipe_groups"]
    }
    if selected_recipe_group_id is None:
        if runtime_check_path is not None:
            raise RuntimeError(
                "Inference runtime cannot be attested before confirmation selection"
            )
        return {
            "required": True,
            "status": "pending_selection",
            "selected_recipe_group_id": None,
            "checked_recipe_family_sha256": None,
        }
    selected_group = group_by_id.get(selected_recipe_group_id)
    if selected_group is None:
        raise RuntimeError("Confirmation selected an unknown recipe group")
    expected_family = str(selected_group["recipe_family_sha256"])
    if runtime_check_path is None:
        return {
            "required": True,
            "status": "pending",
            "selected_recipe_group_id": selected_recipe_group_id,
            "checked_recipe_family_sha256": None,
        }
    serialized = runtime_check_path.read_text(encoding="utf-8")
    payload = json.loads(serialized)
    if not isinstance(payload, Mapping):
        raise RuntimeError("Inference runtime check must be a JSON object")
    required = {
        "schema_version",
        "campaign",
        "confirmation_lock_payload_sha256",
        "status",
        "selected_recipe_group_id",
        "checked_recipe_family_sha256",
        "check_seconds",
        "public_seconds",
        "private_seconds",
        "runtime_check_payload_sha256",
    }
    unhashed = dict(payload)
    stored_sha = unhashed.pop("runtime_check_payload_sha256", None)
    numeric = (
        payload.get("check_seconds"),
        payload.get("public_seconds"),
        payload.get("private_seconds"),
    )
    if (
        set(payload) != required
        or serialized
        != campaign.stage_materializer.canonical_json_dumps(payload) + "\n"
        or stored_sha != campaign.stage_materializer.canonical_sha256(unhashed)
        or payload.get("schema_version") != 1
        or payload.get("campaign") != plan.get("campaign")
        or payload.get("confirmation_lock_payload_sha256")
        != lock.get("lock_payload_sha256")
        or payload.get("status") != "passed"
        or payload.get("selected_recipe_group_id") != selected_recipe_group_id
        or payload.get("checked_recipe_family_sha256") != expected_family
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
            for value in numeric
        )
        or float(numeric[0]) >= 48
        or float(numeric[1]) >= 288
        or float(numeric[2]) >= 624
    ):
        raise RuntimeError("Inference runtime check differs from the frozen gate")
    return dict(payload)


def verify_confirmation_iid_predictions(
    *,
    groups: Sequence[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]],
    baseline_group_id: str,
    artifacts_dir: Path,
) -> None:
    """Bind matched-seed selection metrics to the downloaded IID parquets."""
    frames: dict[str, pd.DataFrame] = {}
    for _, group_rows in groups:
        for row in group_rows:
            if not row.get("completed"):
                continue
            experiment = str(row["experiment"])
            if experiment in frames:
                continue
            path = _iid_prediction_path(artifacts_dir, str(row["kernel_slug"]))
            declared_sha = row.get("iid_predictions_sha256")
            actual_sha = campaign.stage_materializer.file_sha256(path)
            if declared_sha != actual_sha:
                raise RuntimeError(
                    f"Confirmation IID SHA differs for {experiment!r}"
                )
            frame = read_prediction_artifact(path)
            score_column = "score" if "score" in frame else "predict"
            category_column = (
                "category" if "category" in frame else "category_1"
            )
            calculated = macro_average_precision(
                frame["target"].to_numpy(),
                frame[score_column].to_numpy(),
                frame[category_column].to_numpy(),
            )
            if not math.isclose(
                calculated,
                float(row["iid_macro_ap"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"Confirmation IID report/parquet metric differs for "
                    f"{experiment!r}"
                )
            frames[experiment] = frame

    baseline_matches = [
        group_rows
        for group, group_rows in groups
        if str(group["recipe_group_id"]) == baseline_group_id
    ]
    if len(baseline_matches) != 1:
        raise RuntimeError("Confirmation has no unique matched-seed baseline group")
    baseline_by_seed = {
        int(row["seed"]): row
        for row in baseline_matches[0]
        if row.get("completed")
    }
    for _, group_rows in groups:
        for row in group_rows:
            if not row.get("completed"):
                continue
            baseline = baseline_by_seed.get(int(row["seed"]))
            if baseline is None:
                continue
            align_predictions(
                frames[str(baseline["experiment"])],
                frames[str(row["experiment"])],
            )


def adaptive_confirmation_projection(
    *,
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    artifacts_dir: Path,
    runtime_check_path: Path | None,
) -> dict[str, Any] | None:
    if lock.get("mode") != "confirmation":
        return None
    by_experiment = {str(row["experiment"]): row for row in rows}
    origin_by_id = {
        str(origin["origin_id"]): origin for origin in lock["origins"]
    }
    variants_by_group: dict[str, list[Mapping[str, Any]]] = {}
    for variant in lock["resolved_stage"]["variants"]:
        variants_by_group.setdefault(str(variant["recipe_group_id"]), []).append(
            variant
        )
    groups = []
    for group in lock["resolved_stage"]["recipe_groups"]:
        group_id = str(group["recipe_group_id"])
        origin = origin_by_id[str(group["origin_seed42_id"])]
        group_rows = [by_experiment[str(origin["experiment"])] ]
        group_rows.extend(
            by_experiment[str(variant["experiment"])]
            for variant in variants_by_group.get(group_id, [])
        )
        group_rows.sort(key=lambda row: int(row["seed"]))
        groups.append((group, group_rows))
    baseline_group_id = str(lock["family"]["baseline_recipe_group_id"])
    verify_confirmation_iid_predictions(
        groups=groups,
        baseline_group_id=baseline_group_id,
        artifacts_dir=artifacts_dir,
    )
    baseline_rows = next(rows_ for group, rows_ in groups if group["recipe_group_id"] == baseline_group_id)
    expected_seeds = {int(seed) for seed in lock["decision"]["seeds"]}
    for group, group_rows in groups:
        actual_seeds = [int(row["seed"]) for row in group_rows]
        if len(actual_seeds) != len(set(actual_seeds)) or set(actual_seeds) != expected_seeds:
            raise RuntimeError(
                f"Confirmation group {group['recipe_group_id']!r} does not cover "
                "the exact matched-seed set"
            )
    baseline_by_seed = {int(row["seed"]): row for row in baseline_rows}
    baseline_complete = all(bool(row["completed"]) for row in baseline_rows)
    acceptance = lock["decision"]["acceptance"]
    evaluations = []
    mean_iid_by_group: dict[str, Decimal] = {}
    for group, group_rows in groups:
        complete = all(bool(row["completed"]) for row in group_rows)
        decimal_deltas = (
            [
                Decimal(str(row["iid_macro_ap"]))
                - Decimal(
                    str(baseline_by_seed[int(row["seed"])]["iid_macro_ap"])
                )
                for row in group_rows
            ]
            if complete
            and baseline_complete
            and all(int(row["seed"]) in baseline_by_seed for row in group_rows)
            else []
        )
        mean_delta = (
            sum(decimal_deltas, Decimal("0")) / Decimal(len(decimal_deltas))
            if decimal_deltas
            else None
        )
        mean_iid = (
            sum(
                (Decimal(str(row["iid_macro_ap"])) for row in group_rows),
                Decimal("0"),
            )
            / Decimal(len(group_rows))
            if complete
            else None
        )
        if mean_iid is not None:
            mean_iid_by_group[str(group["recipe_group_id"])] = mean_iid
        accepted = (
            group["recipe_group_id"] == baseline_group_id
            or (
                len(decimal_deltas) == 3
                and mean_delta
                >= Decimal(str(acceptance["min_mean_iid_delta"]))
                and sum(delta > 0 for delta in decimal_deltas)
                >= int(acceptance["min_positive_seed_count"])
                and min(decimal_deltas)
                >= Decimal(str(acceptance["min_worst_seed_iid_delta"]))
            )
        )
        evaluations.append(
            {
                "recipe_group_id": group["recipe_group_id"],
                "recipe_family_sha256": group["recipe_family_sha256"],
                "roles": deepcopy(group["roles"]),
                "complete": complete,
                "seed_run_ids": {
                    str(row["seed"]): row["run_id"] for row in group_rows
                },
                "iid_deltas_vs_matched_baseline": [
                    float(delta) for delta in decimal_deltas
                ],
                "mean_iid_delta": float(mean_delta) if mean_delta is not None else None,
                "positive_seed_count": sum(
                    delta > 0 for delta in decimal_deltas
                ),
                "worst_seed_iid_delta": (
                    float(min(decimal_deltas)) if decimal_deltas else None
                ),
                "accepted": accepted,
                "mean_iid_macro_ap": float(mean_iid) if mean_iid is not None else None,
            }
        )
    complete = all(item["complete"] for item in evaluations)
    eligible = [item for item in evaluations if item["accepted"] and item["complete"]]
    selected = None
    shortlist = []
    if complete and eligible:
        best_mean = max(
            mean_iid_by_group[str(item["recipe_group_id"])] for item in eligible
        )
        tie_margin = Decimal(
            str(plan["selection_protocol"]["practical_tie_margin"])
        )
        shortlist = [
            item
            for item in eligible
            if mean_iid_by_group[str(item["recipe_group_id"])]
            >= best_mean - tie_margin
        ]
        tie_order = list(lock["decision"]["final_tie_break_order"])

        loss_stage = campaign.stage_materializer.stage_by_name(
            plan, "special_loss_screen"
        )
        declared_losses = list(loss_stage["loss_variants"])
        declared_losses.extend(
            value
            for value in loss_stage["conditional_combination"][
                "variants_by_balance"
            ].values()
            if value not in declared_losses
        )

        def tie_rank(item: Mapping[str, Any]) -> tuple[int, int, str]:
            roles = list(item["roles"])
            ranks = [tie_order.index(role) for role in roles if role in tie_order]
            if ranks:
                return (
                    min(ranks),
                    0,
                    str(item["recipe_group_id"]),
                )
            loss_token_rank = (
                tie_order.index("loss_variants_declaration_order")
                if "loss_variants_declaration_order" in tie_order
                else len(tie_order)
            )
            loss_variant = next(
                str(group["loss_variant"])
                for group in lock["resolved_stage"]["recipe_groups"]
                if group["recipe_group_id"] == item["recipe_group_id"]
            )
            loss_rank = (
                declared_losses.index(loss_variant)
                if loss_variant in declared_losses
                else len(declared_losses)
            )
            return (
                loss_token_rank,
                loss_rank,
                str(item["recipe_group_id"]),
            )

        selected = min(shortlist, key=tie_rank)["recipe_group_id"]
    runtime_gate = _confirmation_runtime_gate(
        plan=plan,
        lock=lock,
        runtime_check_path=runtime_check_path,
        selected_recipe_group_id=selected,
    )
    return {
        "comparison": lock["decision"]["matched_seed_comparison"],
        "acceptance": deepcopy(acceptance),
        "groups": evaluations,
        "practical_shortlist_recipe_group_ids": [
            item["recipe_group_id"] for item in shortlist
        ],
        "selected_recipe_group_id": (
            selected if runtime_gate["status"] == "passed" else None
        ),
        "selection_before_runtime_gate": selected,
        "runtime_gate": runtime_gate,
        "decision_status": (
            "ready"
            if complete and runtime_gate["status"] == "passed"
            else "runtime_gate_pending"
            if complete
            else "pending_runs"
        ),
    }


def summarize_adaptive_campaign_lock(
    *,
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    lock_path: Path,
    artifacts_dir: Path,
    output_dir: Path,
    runtime_check_path: Path | None = None,
) -> dict[str, Any]:
    if lock.get("schema_version") != 2:
        raise RuntimeError("Adaptive summarizer requires a schema-v2 lock")
    base_config = campaign.cross_builder.load_training_config(campaign.BASE_CONFIG_PATH)
    closure = load_adaptive_prerequisite_closure(
        plan=plan,
        root_lock=lock,
        root_lock_path=lock_path,
        base_config=base_config,
    )
    rows = adaptive_run_rows(plan=plan, lock=lock, artifacts_dir=artifacts_dir)
    completed_run_ids = [str(row["run_id"]) for row in rows if row["completed"]]
    if len(completed_run_ids) != len(set(completed_run_ids)):
        raise RuntimeError("Adaptive run ledger contains duplicate run IDs")
    current_variants = {
        str(variant["experiment"])
        for variant in lock["resolved_stage"]["variants"]
    }
    rows_by_experiment = {str(row["experiment"]): row for row in rows}
    runs_complete = all(
        rows_by_experiment[experiment]["completed"] for experiment in current_variants
    )
    if lock["execution_status"] == "skipped":
        runs_complete = True
    projections = adaptive_hypothesis_projections(
        closure=closure,
        rows=rows,
        artifacts_dir=artifacts_dir,
        cache_dir=output_dir / "anchor_comparisons",
    )
    confirmation = adaptive_confirmation_projection(
        plan=plan,
        lock=lock,
        rows=rows,
        artifacts_dir=artifacts_dir,
        runtime_check_path=runtime_check_path,
    )
    decision_status = (
        confirmation["decision_status"]
        if confirmation is not None
        else "ready"
        if runs_complete
        else "pending"
    )
    decision_ready = runs_complete and decision_status == "ready"
    stage_decision = {
        "complete": decision_ready,
        "runs_complete": runs_complete,
        "decision_status": decision_status,
        "branch_status": lock["execution_status"],
        "expected_new_runs": len(current_variants),
        "completed_new_runs": sum(
            bool(rows_by_experiment[experiment]["completed"])
            for experiment in current_variants
        ),
        "needs_boundary_extension": False,
    }
    adaptive = campaign._adaptive_lock_module()
    runnable_hashes = sorted(
        str(item["lock_payload_sha256"])
        for item, _ in closure
        if item.get("schema_version") == 2
        and item.get("execution_status") == "runnable"
    )
    receipt_hashes = sorted(
        str(item["lock_payload_sha256"])
        for item, _ in closure
        if item.get("schema_version") == 2
        and item.get("execution_status") == "skipped"
    )
    budget = lock["budget"]
    result: dict[str, Any] = {
        "schema_version": 2,
        "campaign": plan["campaign"],
        "execution_status": "complete" if decision_ready else "pending",
        "execution_lock_sha256s": runnable_hashes,
        "execution_receipt_sha256s": receipt_hashes,
        "execution_campaign_lock_sha256s": sorted(
            [*runnable_hashes, *receipt_hashes]
        ),
        "effective_stage": lock["effective_stage"],
        "mode": lock["mode"],
        "budget": {
            "history_complete_through": lock["effective_stage"],
            "counting_identity": budget["counting_identity"],
            "unique_kernel_slugs": deepcopy(
                budget["all_unique_kernel_slugs_after"]
            ),
            "unique_kernels": int(budget["resulting_unique_kernels"]),
            "hard_limit": int(budget["hard_limit"]),
            "source_lock_budget_sha256": adaptive.canonical_sha256(budget),
        },
        "adaptive_closure": [
            {
                "schema_version": item["schema_version"],
                "kind": item["kind"],
                "mode": item.get("mode"),
                "effective_stage": item["effective_stage"],
                "execution_status": item.get("execution_status", "runnable"),
                "lock_payload_sha256": item["lock_payload_sha256"],
                "lock_path": str(path),
            }
            for item, path in closure
        ],
        "stages": {str(lock["effective_stage"]): stage_decision},
        "hypothesis_families": projections,
        "confirmation": confirmation,
        "runs": rows,
    }
    result["summary_payload_sha256"] = adaptive.canonical_sha256(result)
    summary_text = json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        default=str,
        allow_nan=False,
    )
    frame = pd.DataFrame(rows)
    csv_text = frame.to_csv(index=False)
    report = "\n".join(
        [
            f"# {plan['campaign']} — {lock['effective_stage']}",
            "",
            f"Execution status: **{result['execution_status']}**.",
            f"Decision status: **{decision_status}**.",
            "",
            "```json",
            json.dumps(
                {
                    "hypothesis_families": projections,
                    "confirmation": confirmation,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            "```",
            "",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_stage_snapshot(
        output_dir,
        effective_stage=str(lock["effective_stage"]),
        files={"runs.csv": csv_text, "summary.json": summary_text, "report.md": report},
    )
    _atomic_write_text(output_dir / "runs.csv", csv_text)
    _atomic_write_text(output_dir / "summary.json", summary_text)
    _atomic_write_text(output_dir / "report.md", report)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=campaign.DEFAULT_PLAN)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage")
    parser.add_argument("--stage-lock", type=Path)
    parser.add_argument(
        "--inference-runtime-check",
        type=Path,
        help="canonical confirmation runtime attestation; ignored for other stages",
    )
    args = parser.parse_args()
    plan_path = args.plan if args.plan.is_absolute() else ROOT / args.plan
    artifacts_dir = (
        args.artifacts_dir
        if args.artifacts_dir.is_absolute()
        else ROOT / args.artifacts_dir
    )
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    )
    plan = campaign.load_plan(plan_path)
    stage_lock_path = args.stage_lock
    if stage_lock_path is not None and not stage_lock_path.is_absolute():
        stage_lock_path = ROOT / stage_lock_path
    base_config = campaign.cross_builder.load_training_config(
        campaign.BASE_CONFIG_PATH
    )
    stage_lock = (
        campaign.load_campaign_lock(
            stage_lock_path,
            plan=plan,
            base_config=base_config,
        )
        if stage_lock_path is not None
        else None
    )
    if stage_lock is not None and stage_lock.get("schema_version") == 2:
        contract = campaign.normalized_campaign_execution_contract(
            plan,
            stage_lock,
            base_config=base_config,
        )
        if args.stage is not None and args.stage not in set(
            contract["accepted_stage_filters"]
        ):
            raise campaign.CampaignConfigError(
                f"--stage {args.stage!r} differs from locked stage "
                f"{contract['effective_stage']!r}"
            )
        adaptive = campaign._adaptive_lock_module()
        trusted = adaptive.load_trusted_provenance(
            adaptive.trusted_provenance_manifest_path(
                stage_lock_path.resolve(strict=True)
            ),
            plan=plan,
        )
        if Path(str(trusted["artifacts_dir"])) != artifacts_dir.resolve(strict=True):
            raise RuntimeError(
                "--artifacts-dir differs from the schema-v2 trusted authority"
            )
        runtime_check_path = args.inference_runtime_check
        if runtime_check_path is not None and not runtime_check_path.is_absolute():
            runtime_check_path = ROOT / runtime_check_path
        result = summarize_adaptive_campaign_lock(
            plan=plan,
            lock=stage_lock,
            lock_path=stage_lock_path,
            artifacts_dir=artifacts_dir,
            output_dir=output_dir,
            runtime_check_path=runtime_check_path,
        )
        print(json.dumps(result["stages"], ensure_ascii=False, indent=2))
        return
    rows = []
    for entry in expected_entries(
        plan,
        args.stage,
        stage_lock=stage_lock,
    ):
        path = completion_path(artifacts_dir, entry["kernel_slug"])
        rows.append(
            completion_row(
                entry,
                path,
                baseline_run_id=str(plan["baseline_run_id"]),
            )
        )
    frame = pd.DataFrame(rows)
    frame = add_anchor_relative_iid(
        frame,
        artifacts_dir=artifacts_dir,
        cache_dir=output_dir / "anchor_comparisons",
    )
    frame = add_stage_holm(frame)
    completed_run_ids = frame.loc[frame["completed"], "run_id"]
    if completed_run_ids.isna().any() or completed_run_ids.duplicated().any():
        raise RuntimeError("Completed campaign runs must have unique non-empty run_id values")
    frame = frame.sort_values(
        ["stage", "completed", "iid_macro_ap"],
        ascending=[True, False, False],
        na_position="last",
    ).reset_index(drop=True)
    summary = stage_summary(
        frame,
        tie_margin=float(plan["selection_protocol"]["practical_tie_margin"]),
        control_gate=plan["control_gate"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_frame = frame.drop(
        columns=[
            "planned_overrides",
            "expected_config",
            "expected_recipe_sha256",
            "expected_source_sha256",
            "expected_loss_hook_sha256",
            "expected_run_id",
            "expected_iid_predictions_sha256",
            "expected_completion_sha256",
            "expected_notes",
            "provenance_alias",
            "parent_provenance",
        ],
        errors="ignore",
    )
    result = {
        "schema_version": 1,
        "campaign": plan["campaign"],
        "baseline_run_id": plan["baseline_run_id"],
        "selection_protocol": plan["selection_protocol"],
        "stages": summary,
        "runs": [
            {key: _json_cell(value) for key, value in row.items()}
            for row in csv_frame.to_dict(orient="records")
        ],
    }
    if stage_lock is not None:
        result["stage_lock"] = {
            "path": str(stage_lock_path),
            "lock_payload_sha256": stage_lock["lock_payload_sha256"],
            "source_stage": stage_lock["source_stage"],
            "target_stage": stage_lock["target_stage"],
            "effective_stage": campaign.stage_lock_effective_stage(stage_lock),
            "transition_kind": stage_lock.get(
                "transition_kind", "stage_transition"
            ),
            "parent": campaign.stage_lock_parent_provenance(stage_lock),
        }
    summary_text = json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        default=str,
        allow_nan=False,
    )
    csv_text = csv_frame.to_csv(index=False)
    report = report_text(plan=plan, frame=frame, summary=summary)

    for effective_stage in summary:
        stage_frame = frame[frame["stage"] == effective_stage].copy()
        stage_csv_frame = csv_frame[csv_frame["stage"] == effective_stage].copy()
        stage_result = dict(result)
        stage_result["stages"] = {effective_stage: summary[effective_stage]}
        stage_result["runs"] = [
            {key: _json_cell(value) for key, value in row.items()}
            for row in stage_csv_frame.to_dict(orient="records")
        ]
        if stage_lock is not None and effective_stage != campaign.stage_lock_effective_stage(
            stage_lock
        ):
            stage_result.pop("stage_lock", None)
        write_stage_snapshot(
            output_dir,
            effective_stage=effective_stage,
            files={
                "runs.csv": stage_csv_frame.to_csv(index=False),
                "summary.json": json.dumps(
                    stage_result,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                    allow_nan=False,
                ),
                "report.md": report_text(
                    plan=plan,
                    frame=stage_frame,
                    summary={effective_stage: summary[effective_stage]},
                ),
            },
        )

    _atomic_write_text(output_dir / "runs.csv", csv_text)
    _atomic_write_text(output_dir / "summary.json", summary_text)
    _atomic_write_text(output_dir / "report.md", report)
    table = markdown_table(frame)
    print(table)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
