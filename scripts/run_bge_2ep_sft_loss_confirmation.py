#!/usr/bin/env python3
"""Guarded sequential BGE loss screen and directional seed confirmation.

The default invocation is a network-free plan.  Live mode is intentionally
post-LR/e2, requires the exact frozen private baseline Dataset and exact
selection/Sheets receipts, and never resubmits a terminally failed slug.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import create_bge_2ep_sft_candidate_notebooks as candidate_builder
import create_bge_2ep_sft_loss_confirmation_notebooks as loss_builder
import create_bge_2ep_sft_notebooks as baseline_builder
import create_qwen_training_notebook as shared
import push_bge_2ep_sft_baseline_dataset as baseline_uploader
import run_bge_2ep_sft_candidates as candidate_runner
import run_bge_2ep_sft_kaggle as baseline_launcher
import run_kaggle_notebook as kaggle
import summarize_bge_2ep_sft_comparisons as comparator


OWNER = candidate_runner.OWNER
DEFAULT_REPORT_ROOT = ROOT / "reports" / loss_builder.WORKFLOW
DEFAULT_EPOCH_RECEIPT = (
    candidate_runner.DEFAULT_REPORT_ROOT / "e2" / candidate_runner.EPOCH_RECEIPT_FILENAME
)
SCREEN_RECEIPT_FILENAME = "loss_screen_receipt.json"
LOCAL_SCREEN_RECEIPT_FILENAME = "loss_screen_receipt.local.json"
FINAL_RECEIPT_FILENAME = "loss_confirmation_receipt.json"
LOCAL_FINAL_RECEIPT_FILENAME = "loss_confirmation_receipt.local.json"
SCREEN_FAMILY = "bge2_sft_selected_recipe_loss_transfer_v1"
SEED17_FAMILY = "bge2_sft_loss_direction_seed17_v1"
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
SLIM_OUTPUT_PATTERN = candidate_runner.SLIM_OUTPUT_PATTERN
MAX_RUN_TIMEOUT_SECONDS = 32_400
# IID and hard deliberately exclude the two former OOD categories.  The loss
# hook still trains over all 20 categories after those two are promoted.
EXPECTED_COMPARISON_CATEGORIES = 18
LOSS_KERNEL_SLUG_PREFIXES = ("pm-b2-lbce-", "pm-b2-lsqrt-")
LOSS_KERNEL_SLUG_PATTERN = re.compile(
    r"pm-b2-(?:lbce|lsqrt)-[0-9a-f]{12}-s(?:17|42)-l1"
)
ATTEMPT_LEDGER_PATH = (
    ROOT / ".kaggle" / "audit" / loss_builder.WORKFLOW / "attempted_kernel_slugs.json"
)


class LossConfirmationWorkflowError(RuntimeError):
    """Raised when the loss-confirmation workflow cannot continue exactly."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LossConfirmationWorkflowError(f"Could not read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise LossConfirmationWorkflowError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == serialized:
            return
        raise LossConfirmationWorkflowError(f"Refusing to overwrite a differing receipt: {path}")
    path.write_text(serialized, encoding="utf-8")


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or HASH_PATTERN.fullmatch(value) is None:
        raise LossConfirmationWorkflowError(f"{label} is not an exact lowercase SHA-256")
    return value


def plan() -> dict[str, Any]:
    policy = loss_builder.load_policy()
    return {
        "schema_version": 1,
        "mode": "plan_only",
        "workflow": loss_builder.WORKFLOW,
        "campaign": baseline_builder.CAMPAIGN,
        "blocked_until": [
            "final frozen-v4 baseline output and private slim Dataset are exact",
            "LR and e2 selection receipts are complete and comparison-Sheets-synced",
            "selected e1/e2 plain-BCE output is strictly revalidated",
        ],
        "screen": {
            "execution": "sequential",
            "seed": 42,
            "anchor": "reuse exact selected e1/e2 plain-BCE run",
            "only_new_candidate": loss_builder.SQRT_BALANCED_BCE,
            "accept": "paired IID delta strictly > 0.002",
            "hard": "diagnostic only",
            "ood": "-1; no parquet/comparison",
        },
        "seed17": {
            "if_bce_wins_seed42": [loss_builder.PLAIN_BCE],
            "if_challenger_wins_seed42": [
                loss_builder.PLAIN_BCE,
                loss_builder.SQRT_BALANCED_BCE,
            ],
            "challenger_final_gate": (
                "positive IID delta on both seeds and mean delta >= 0.002"
            ),
            "execution": "one kernel at a time; matched BCE before challenger",
        },
        "kernel_budget": {
            "historical_before_lr_e2": policy["execution"][
                "historical_bge_kernel_slugs_before_lr_e2"
            ],
            "expected_after_lr_e2": 7,
            "maximum_new": 3,
            "hard_total_cap": 10,
            "counting_identity": "unique kernel_slug union",
        },
        "forbidden": [
            "fanout",
            "resubmit terminal failure",
            "ODS",
            "runtime ablation",
            "checkpoint export/resume",
            "extra loss/LR/regularization points",
        ],
        "mutation": False,
    }


def screen_accepts_challenger(iid_delta: float) -> bool:
    """The one-sided practical gate; equality to 0.002 is a rejection."""
    if isinstance(iid_delta, bool) or not isinstance(iid_delta, (int, float)):
        raise LossConfirmationWorkflowError("Loss-screen IID delta must be numeric")
    if not math.isfinite(float(iid_delta)):
        raise LossConfirmationWorkflowError("Loss-screen IID delta must be finite")
    return float(iid_delta) > 0.002


def exact_bound_delta(stored: object, bound: object, *, label: str) -> float:
    """Return only an exactly equal finite bound delta; no tolerance at gates."""
    values: list[float] = []
    for value in (stored, bound):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise LossConfirmationWorkflowError(f"{label} delta must be finite numeric")
        values.append(float(value))
    if values[0] != values[1]:
        raise LossConfirmationWorkflowError(f"{label} delta differs from bound comparison")
    return values[1]


def confirmation_variant_keys(seed42_challenger_accepted: bool) -> list[str]:
    if not isinstance(seed42_challenger_accepted, bool):
        raise LossConfirmationWorkflowError("Loss-screen branch must be boolean")
    keys = ["confirm_bce_s17"]
    if seed42_challenger_accepted:
        keys.append("confirm_sqrt_s17")
    return keys


def final_loss_decision(seed42_delta: float, seed17_delta: float | None) -> dict[str, Any]:
    """Apply the predeclared two-seed directional rule without hard metrics."""
    accepted42 = screen_accepts_challenger(seed42_delta)
    if not accepted42:
        if seed17_delta is not None:
            raise LossConfirmationWorkflowError(
                "Rejected seed42 challenger must not have a seed17 challenger delta"
            )
        return {
            "selected_loss_variant": loss_builder.PLAIN_BCE,
            "challenger_accepted": False,
            "mean_iid_delta": None,
        }
    if (
        isinstance(seed17_delta, bool)
        or not isinstance(seed17_delta, (int, float))
        or not math.isfinite(float(seed17_delta))
    ):
        raise LossConfirmationWorkflowError(
            "Accepted seed42 challenger requires a finite matched seed17 delta"
        )
    mean_delta = (float(seed42_delta) + float(seed17_delta)) / 2
    accepted = float(seed42_delta) > 0 and float(seed17_delta) > 0 and mean_delta >= 0.002
    return {
        "selected_loss_variant": (
            loss_builder.SQRT_BALANCED_BCE if accepted else loss_builder.PLAIN_BCE
        ),
        "challenger_accepted": accepted,
        "mean_iid_delta": mean_delta,
    }


def _parent_from_output(
    *,
    entry: Mapping[str, Any],
    completion: Mapping[str, Any],
) -> dict[str, Any]:
    return loss_builder.validate_parent_receipt(
        {
            "run_id": completion["run_id"],
            "experiment": completion["experiment"],
            "campaign_identity_sha256": completion["campaign_identity_sha256"],
            "source_sha256": completion["code_bundle_sha256"],
            "recipe_sha256": completion["frozen_recipe_sha256"],
            "checkpoint_manifest_sha256": completion[
                "initial_checkpoint_manifest_sha256"
            ],
            "checkpoint_model_sha256": completion["initial_checkpoint_model_sha256"],
            "validation_manifest_sha256": completion["validation_manifest_sha256"],
            "loss_hook_sha256": completion["loss_hook_sha256"],
            "config": dict(entry["expected_config"]),
        }
    )


def _validate_epoch_comparison(
    comparison: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
    anchor: Mapping[str, Any],
    challenger: Mapping[str, Any],
) -> None:
    """Validate the exact Sheets-bound e2 paired-comparison contract."""
    exact_keys = {
        "schema_version",
        "status",
        "baseline_run_id",
        "candidate_run_id",
        "baseline_experiment",
        "candidate_experiment",
        "baseline_manifest_sha256",
        "method",
        "confidence_interval_method",
        "multiple_testing_correction",
        "holm_family",
        "holm_family_members",
        "primary_split",
        "diagnostic_splits",
        "practical_tie_margin",
        "iid_practical_relation",
        "ood_policy",
        "splits",
    }
    if set(comparison) != exact_keys:
        raise LossConfirmationWorkflowError("Epoch comparison fields differ")
    expected = {
        "schema_version": 1,
        "status": "ready_ood_disabled",
        "baseline_run_id": anchor["run_id"],
        "candidate_run_id": challenger["run_id"],
        "baseline_experiment": anchor["experiment"],
        "candidate_experiment": challenger["experiment"],
        "baseline_manifest_sha256": authority["context"]["manifest_sha256"],
        "method": "paired_component_permutation",
        "confidence_interval_method": "paired_component_bootstrap_percentile",
        "multiple_testing_correction": (
            "holm_within_planned_candidate_family_per_split"
        ),
        "holm_family": candidate_runner.EPOCH_FAMILY_NAME,
        "holm_family_members": [challenger["experiment"]],
        "primary_split": "iid",
        "diagnostic_splits": ["hard"],
        "practical_tie_margin": comparator.PRACTICAL_TIE_MARGIN,
        "ood_policy": "disabled_train_contaminated_no_paired_comparison",
    }
    for key, value in expected.items():
        if comparison.get(key) != value:
            raise LossConfirmationWorkflowError(f"Epoch comparison differs at {key}")

    splits = comparison.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"iid", "hard", "ood"}:
        raise LossConfirmationWorkflowError("Epoch comparison split contract differs")
    if splits.get("ood") != comparator._ood_result():
        raise LossConfirmationWorkflowError("Epoch comparison fabricated OOD evidence")
    result_keys = {
        "examples",
        "categories",
        "components",
        "baseline_macro_average_precision",
        "candidate_macro_average_precision",
        "delta_macro_average_precision",
        "p_value",
        "ci95_low",
        "ci95_high",
        "permutations",
        "bootstrap_resamples",
        "seed",
        "p_value_holm",
        "holm_family",
        "holm_family_size",
    }
    resample_contract: tuple[int, int] | None = None
    for split_index, split in enumerate(comparator.SPLITS):
        result = splits.get(split)
        if not isinstance(result, Mapping) or set(result) != result_keys:
            raise LossConfirmationWorkflowError(
                f"Epoch comparison {split} result fields differ"
            )
        integer_expected = {
            "examples": comparator.EXPECTED_ROWS[split],
            "categories": EXPECTED_COMPARISON_CATEGORIES,
            "seed": 42 + split_index,
            "holm_family_size": 1,
        }
        for key, value in integer_expected.items():
            if isinstance(result.get(key), bool) or result.get(key) != value:
                raise LossConfirmationWorkflowError(
                    f"Epoch comparison {split} differs at {key}"
                )
        for key in ("components", "permutations", "bootstrap_resamples"):
            value = result.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise LossConfirmationWorkflowError(
                    f"Epoch comparison {split} has invalid {key}"
                )
        observed_resamples = (
            int(result["permutations"]),
            int(result["bootstrap_resamples"]),
        )
        if resample_contract is None:
            resample_contract = observed_resamples
        elif observed_resamples != resample_contract:
            raise LossConfirmationWorkflowError(
                "Epoch comparison resample counts differ by split"
            )
        numeric: dict[str, float] = {}
        for key in (
            "baseline_macro_average_precision",
            "candidate_macro_average_precision",
            "delta_macro_average_precision",
            "p_value",
            "p_value_holm",
            "ci95_low",
            "ci95_high",
        ):
            value = result.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise LossConfirmationWorkflowError(
                    f"Epoch comparison {split} has invalid {key}"
                )
            numeric[key] = float(value)
        if not 0 <= numeric["baseline_macro_average_precision"] <= 1 or not 0 <= numeric[
            "candidate_macro_average_precision"
        ] <= 1:
            raise LossConfirmationWorkflowError(
                f"Epoch comparison {split} macro AP is outside [0, 1]"
            )
        if not 0 <= numeric["p_value"] <= 1 or numeric["p_value_holm"] != numeric[
            "p_value"
        ]:
            raise LossConfirmationWorkflowError(
                f"Epoch comparison {split} one-member Holm value differs"
            )
        if numeric["ci95_low"] > numeric["ci95_high"]:
            raise LossConfirmationWorkflowError(
                f"Epoch comparison {split} confidence interval is reversed"
            )
        expected_delta = (
            numeric["candidate_macro_average_precision"]
            - numeric["baseline_macro_average_precision"]
        )
        if not math.isclose(
            numeric["delta_macro_average_precision"],
            expected_delta,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise LossConfirmationWorkflowError(
                f"Epoch comparison {split} delta is inconsistent"
            )
        expected_anchor_ap = comparator._metric_from_report(
            anchor["report"], split=split, label="selected e1 epoch anchor"
        )
        expected_challenger_ap = comparator._metric_from_report(
            challenger["report"], split=split, label="e2 epoch challenger"
        )
        if not math.isclose(
            numeric["baseline_macro_average_precision"],
            expected_anchor_ap,
            rel_tol=0,
            abs_tol=1e-12,
        ) or not math.isclose(
            numeric["candidate_macro_average_precision"],
            expected_challenger_ap,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise LossConfirmationWorkflowError(
                f"Epoch comparison {split} report binding differs"
            )
        if (
            result.get("holm_family") != candidate_runner.EPOCH_FAMILY_NAME
            or result.get("holm_family_size") != 1
        ):
            raise LossConfirmationWorkflowError(
                f"Epoch comparison {split} Holm family differs"
            )

    iid_delta = float(splits["iid"]["delta_macro_average_precision"])
    if comparison.get("iid_practical_relation") != comparator._practical_relation(
        iid_delta, comparator.PRACTICAL_TIE_MARGIN
    ):
        raise LossConfirmationWorkflowError("Epoch comparison practical relation differs")


def _epoch_selection_from_bound_comparison(
    *,
    summary: Mapping[str, Any],
    comparison: Mapping[str, Any],
    e1_parent: Mapping[str, Any],
    e2_parent: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Bind the summary to the Sheets comparison and recompute the e1/e2 gate."""
    if summary.get("splits") != comparison.get("splits"):
        raise LossConfirmationWorkflowError(
            "Epoch summary splits differ from the Sheets-bound comparison"
        )
    splits = comparison.get("splits")
    if not isinstance(splits, Mapping) or not isinstance(splits.get("iid"), Mapping):
        raise LossConfirmationWorkflowError("Epoch comparison has no paired IID result")
    delta = splits["iid"].get("delta_macro_average_precision")
    if (
        isinstance(delta, bool)
        or not isinstance(delta, (int, float))
        or not math.isfinite(float(delta))
    ):
        raise LossConfirmationWorkflowError("Epoch paired IID delta is invalid")
    select_e2 = float(delta) > comparator.PRACTICAL_TIE_MARGIN
    expected_selection = {
        "selected_experiment": (
            e2_parent["experiment"] if select_e2 else e1_parent["experiment"]
        ),
        "selected_run_id": e2_parent["run_id"] if select_e2 else e1_parent["run_id"],
        "selected_epoch": 2 if select_e2 else 1,
        "rule": "select e2 iff paired IID delta > 0.002",
    }
    if summary.get("selection") != expected_selection:
        raise LossConfirmationWorkflowError(
            "Epoch summary selection was not recomputed from bound IID"
        )
    return select_e2, expected_selection


def _validate_epoch_receipt(
    path: Path,
    *,
    authority: Mapping[str, Any],
    lr_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_path = path.expanduser().resolve(strict=True)
    receipt = _read_json(receipt_path)
    exact_keys = {
        "schema_version",
        "status",
        "campaign",
        "stage",
        "family_name",
        "frozen_baseline_dataset",
        "lr_selection_receipt_sha256",
        "parent",
        "candidate_run_id",
        "candidate_experiment",
        "comparison_sheets_synced",
        "comparison_sync_marker",
        "epoch_summary_path",
        "epoch_summary_sha256",
        "selection",
        "epoch_3",
    }
    if set(receipt) != exact_keys:
        raise LossConfirmationWorkflowError("Epoch selection receipt fields differ")
    expected = {
        "schema_version": 1,
        "status": "complete",
        "campaign": baseline_builder.CAMPAIGN,
        "stage": "epoch_line",
        "family_name": candidate_runner.EPOCH_FAMILY_NAME,
        "frozen_baseline_dataset": candidate_runner._baseline_dataset_receipt(authority),
        "lr_selection_receipt_sha256": comparator.sha256_file(
            Path(lr_receipt["_receipt_path"])
        ),
        "parent": lr_receipt["selected_parent"],
        "comparison_sheets_synced": True,
        "epoch_3": "deferred",
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise LossConfirmationWorkflowError(f"Epoch selection receipt differs at {key}")
    summary_path = Path(str(receipt["epoch_summary_path"])).expanduser().resolve(strict=True)
    if summary_path != receipt_path.parent / "epoch_family_summary.json":
        raise LossConfirmationWorkflowError("Epoch summary is not beside its receipt")
    if comparator.sha256_file(summary_path) != _require_hash(
        receipt.get("epoch_summary_sha256"), "epoch summary SHA-256"
    ):
        raise LossConfirmationWorkflowError("Epoch summary changed after selection")
    summary = _read_json(summary_path)
    e2_entry = candidate_runner.build_e2_entry(authority, lr_receipt, write=False)
    e2_dir = baseline_launcher.output_root() / str(e2_entry["kernel_slug"])
    candidate_runner.validate_candidate_output(e2_dir, entry=e2_entry)
    e2_completion = _read_json(e2_dir / baseline_uploader.COMPLETION_FILENAME)
    e2_parent = _parent_from_output(entry=e2_entry, completion=e2_completion)
    if (
        receipt.get("candidate_run_id") != e2_parent["run_id"]
        or receipt.get("candidate_experiment") != e2_parent["experiment"]
    ):
        raise LossConfirmationWorkflowError("Epoch receipt candidate binding differs")
    exact_summary = {
        "schema_version": 1,
        "status": "complete",
        "campaign": baseline_builder.CAMPAIGN,
        "family_name": candidate_runner.EPOCH_FAMILY_NAME,
        "primary_split": "iid",
        "diagnostic_splits": ["hard"],
        "hard_used_for_selection": False,
        "ood_used_for_selection": False,
        "practical_tie_margin": comparator.PRACTICAL_TIE_MARGIN,
        "anchor": lr_receipt["selected_parent"],
    }
    if set(summary) != {
        *exact_summary,
        "candidate",
        "splits",
        "selection",
    }:
        raise LossConfirmationWorkflowError("Epoch summary fields differ")
    for key, value in exact_summary.items():
        if summary.get(key) != value:
            raise LossConfirmationWorkflowError(f"Epoch summary differs at {key}")
    candidate_summary = summary.get("candidate")
    if not isinstance(candidate_summary, Mapping) or candidate_summary != {
        "run_id": e2_parent["run_id"],
        "experiment": e2_parent["experiment"],
        "recipe_sha256": e2_parent["recipe_sha256"],
    }:
        raise LossConfirmationWorkflowError("Epoch summary candidate differs")
    augmented_path = receipt_path.parent / "completion_with_comparison.json"
    augmented = _read_json(augmented_path)
    raw = dict(augmented)
    comparison = raw.pop("baseline_comparison", None)
    if raw != e2_completion or not isinstance(comparison, Mapping):
        raise LossConfirmationWorkflowError("Epoch augmented completion differs from raw e2")
    candidate_runner.validate_augmented_completion(
        augmented,
        expected_baseline_run_id=lr_receipt["selected_parent"]["run_id"],
    )
    comparison_path = receipt_path.parent / "baseline_comparison.json"
    if _read_json(comparison_path) != comparison:
        raise LossConfirmationWorkflowError(
            "Epoch standalone and embedded comparisons differ"
        )
    epoch_anchor = _load_comparison_side(
        Path(lr_receipt["selected_directory"]), label="selected e1 epoch anchor"
    )
    epoch_challenger = _load_comparison_side(e2_dir, label="e2 epoch challenger")
    _validate_epoch_comparison(
        comparison,
        authority=authority,
        anchor=epoch_anchor,
        challenger=epoch_challenger,
    )
    select_e2, expected_selection = _epoch_selection_from_bound_comparison(
        summary=summary,
        comparison=comparison,
        e1_parent=lr_receipt["selected_parent"],
        e2_parent=e2_parent,
    )
    if receipt.get("selection") != expected_selection:
        raise LossConfirmationWorkflowError(
            "Epoch receipt selection differs from bound paired comparison"
        )
    marker_declaration = receipt.get("comparison_sync_marker")
    marker_path = receipt_path.parent / candidate_runner.COMPARISON_SYNC_FILENAME
    if not isinstance(marker_declaration, Mapping) or marker_declaration.get("path") != str(marker_path):
        raise LossConfirmationWorkflowError("Epoch comparison Sheets marker path differs")
    if comparator.sha256_file(marker_path) != _require_hash(
        marker_declaration.get("sha256"), "epoch comparison marker SHA-256"
    ):
        raise LossConfirmationWorkflowError("Epoch comparison Sheets marker changed")
    marker = _read_json(marker_path)
    candidate_runner.validate_comparison_sync_marker(marker, completion=augmented)
    if marker_declaration.get("completion_canonical_sha256") != marker.get(
        "completion_canonical_sha256"
    ):
        raise LossConfirmationWorkflowError("Epoch marker/completion binding differs")
    return {
        **receipt,
        "summary": summary,
        "e2_entry": e2_entry,
        "e2_directory": e2_dir,
        "e2_parent": e2_parent,
        "selected_e2": select_e2,
        "_receipt_path": receipt_path,
    }


def _selected_e1_entry(
    authority: Mapping[str, Any],
    lr_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    parent = lr_receipt["selected_parent"]
    if lr_receipt["selected_source"] == "baseline":
        return dict(authority["entry"]), Path(authority["source_dir"])
    entries = candidate_runner.build_lr_entries(authority, write=False)
    by_experiment = {entry["experiment"]: entry for entry in entries}
    entry = by_experiment.get(parent["experiment"])
    if entry is None:
        raise LossConfirmationWorkflowError("Selected e1 is outside the exact LR family")
    return entry, Path(lr_receipt["selected_directory"])


def _prediction_binding(directory: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for split in comparator.SPLITS:
        matches = list(directory.rglob(baseline_uploader.PREDICTION_FILENAMES[split]))
        if len(matches) != 1 or not matches[0].is_file():
            raise LossConfirmationWorkflowError(f"Selected output has no exact {split} predictions")
        result[split] = {
            "path": str(matches[0]),
            "bytes": matches[0].stat().st_size,
            "sha256": comparator.sha256_file(matches[0]),
        }
    return result


def load_selected_authority(
    *,
    authority: Mapping[str, Any],
    lr_receipt_path: Path,
    epoch_receipt_path: Path,
) -> dict[str, Any]:
    lr_receipt = candidate_runner.load_lr_selection_receipt(
        lr_receipt_path,
        authority=authority,
    )
    if lr_receipt.get("comparison_sheets_synced") is not True:
        raise LossConfirmationWorkflowError("Loss workflow requires synced LR comparisons")
    epoch_receipt = _validate_epoch_receipt(
        epoch_receipt_path,
        authority=authority,
        lr_receipt=lr_receipt,
    )
    if epoch_receipt["selected_e2"]:
        entry = epoch_receipt["e2_entry"]
        directory = epoch_receipt["e2_directory"]
        parent = epoch_receipt["e2_parent"]
    else:
        entry, directory = _selected_e1_entry(authority, lr_receipt)
        if lr_receipt["selected_source"] == "baseline":
            completion = _read_json(directory / baseline_uploader.COMPLETION_FILENAME)
            parent = loss_builder.validate_parent_receipt(
                {
                    **authority["baseline_parent"],
                    "config": dict(entry["expected_config"]),
                },
                require_seed=42,
                require_plain_bce=True,
            )
            if completion.get("run_id") != parent["run_id"]:
                raise LossConfirmationWorkflowError("Selected baseline raw output differs")
        else:
            candidate_runner.validate_candidate_output(directory, entry=entry)
            completion = _read_json(directory / baseline_uploader.COMPLETION_FILENAME)
            parent = _parent_from_output(entry=entry, completion=completion)
    completion = _read_json(directory / baseline_uploader.COMPLETION_FILENAME)
    parent = loss_builder.validate_parent_receipt(
        parent,
        require_seed=42,
        require_plain_bce=True,
    )
    exact = {
        "run_id": parent["run_id"],
        "experiment": parent["experiment"],
        "campaign_identity_sha256": parent["campaign_identity_sha256"],
        "code_bundle_sha256": parent["source_sha256"],
        "frozen_recipe_sha256": parent["recipe_sha256"],
        "loss_hook_sha256": loss_builder.LOSS_HOOK_SHA256[loss_builder.PLAIN_BCE],
        "loss_variant": loss_builder.PLAIN_BCE,
        "initial_checkpoint_manifest_sha256": parent["checkpoint_manifest_sha256"],
        "initial_checkpoint_model_sha256": parent["checkpoint_model_sha256"],
        "validation_manifest_sha256": parent["validation_manifest_sha256"],
    }
    for key, value in exact.items():
        if completion.get(key) != value:
            raise LossConfirmationWorkflowError(f"Selected plain-BCE anchor differs at {key}")
    if completion.get("training_report", {}).get("validation_splits", {}).get("ood") != baseline_builder.OOD_SENTINEL:
        raise LossConfirmationWorkflowError("Selected anchor changed OOD=-1")
    if list(directory.rglob("ood_validation_predictions.parquet")):
        raise LossConfirmationWorkflowError("Selected anchor contains OOD predictions")
    return {
        "entry": entry,
        "directory": directory,
        "completion": completion,
        "parent": parent,
        "prediction_binding": _prediction_binding(directory),
        "lr_receipt": lr_receipt,
        "epoch_receipt": epoch_receipt,
    }


def _validate_loss_slug_sequence(slugs: Sequence[str], *, label: str) -> list[str]:
    values = list(slugs)
    if len(set(values)) != len(values):
        raise LossConfirmationWorkflowError(f"{label} contains duplicate kernel slugs")
    for slug in values:
        if not isinstance(slug, str) or LOSS_KERNEL_SLUG_PATTERN.fullmatch(slug) is None:
            raise LossConfirmationWorkflowError(
                f"{label} contains a non-loss-workflow slug: {slug!r}"
            )
    return values


def _attempt_ledger_payload(slugs: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "workflow": loss_builder.WORKFLOW,
        "campaign": baseline_builder.CAMPAIGN,
        "owner": OWNER,
        "attempted_loss_kernel_slugs": _validate_loss_slug_sequence(
            slugs, label="Attempt ledger"
        ),
    }


def load_attempt_ledger(path: Path = ATTEMPT_LEDGER_PATH) -> list[str]:
    """Read the append-only local push-intent ledger; missing means no attempts."""
    path = path.expanduser()
    if path.is_symlink():
        raise LossConfirmationWorkflowError("Loss attempt ledger must not be a symlink")
    path = path.resolve()
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file():
        raise LossConfirmationWorkflowError("Loss attempt ledger is not a regular file")
    payload = _read_json(path)
    if set(payload) != {
        "schema_version",
        "workflow",
        "campaign",
        "owner",
        "attempted_loss_kernel_slugs",
    }:
        raise LossConfirmationWorkflowError("Loss attempt ledger fields differ")
    slugs = payload.get("attempted_loss_kernel_slugs")
    if not isinstance(slugs, list):
        raise LossConfirmationWorkflowError("Loss attempt ledger slug list is invalid")
    expected = _attempt_ledger_payload(slugs)
    if payload != expected:
        raise LossConfirmationWorkflowError("Loss attempt ledger binding differs")
    return list(slugs)


def record_kernel_push_intent(
    *,
    path: Path,
    expected_prior_slugs: Sequence[str],
    kernel_slug: str,
) -> dict[str, Any]:
    """Append before push, so a failed API call still permanently spends the slug."""
    path = path.expanduser()
    if path.is_symlink():
        raise LossConfirmationWorkflowError("Loss attempt ledger must not be a symlink")
    path = path.resolve()
    expected_prior = _validate_loss_slug_sequence(
        expected_prior_slugs, label="Expected prior attempt ledger"
    )
    observed = load_attempt_ledger(path)
    if observed != expected_prior:
        raise LossConfirmationWorkflowError(
            "Loss attempt ledger differs; replacement/resubmission is forbidden"
        )
    _validate_loss_slug_sequence([kernel_slug], label="Current loss kernel")
    if kernel_slug in observed:
        raise LossConfirmationWorkflowError(
            "Loss kernel already has a push intent; resubmission is forbidden"
        )
    payload = _attempt_ledger_payload([*observed, kernel_slug])
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return payload


def remote_loss_kernel_slugs(cli: list[str]) -> list[str]:
    """List every loss-workflow attempt under the authenticated owner scope."""
    refs: set[str] = set()
    for prefix in LOSS_KERNEL_SLUG_PREFIXES:
        result = kaggle.run_command(
            cli
            + [
                "kernels",
                "list",
                "--mine",
                "--search",
                prefix.rstrip("-"),
                "--page-size",
                "100",
                "--format",
                "json",
            ],
            check=False,
        )
        if result.returncode:
            raise LossConfirmationWorkflowError(
                "Could not audit prior remote BGE loss kernels"
            )
        listed = baseline_launcher._listed_kernel_refs(result.stdout)
        if len(listed) >= 100:
            raise LossConfirmationWorkflowError(
                "Remote loss-kernel audit may be paginated; refusing to push"
            )
        refs.update(listed)
    owner_prefix = f"{OWNER}/"
    slugs = sorted(
        reference[len(owner_prefix) :]
        for reference in refs
        if reference.startswith(owner_prefix)
        and any(
            reference[len(owner_prefix) :].startswith(prefix)
            for prefix in LOSS_KERNEL_SLUG_PREFIXES
        )
    )
    return _validate_loss_slug_sequence(slugs, label="Remote loss-kernel audit")


def audit_prior_attempts_before_push(
    cli: list[str],
    *,
    expected_prior_slugs: Sequence[str],
    ledger_path: Path = ATTEMPT_LEDGER_PATH,
) -> dict[str, Any]:
    """Require remote and append-only local attempt unions to match exactly."""
    expected = _validate_loss_slug_sequence(
        expected_prior_slugs, label="Expected prior loss attempts"
    )
    local = load_attempt_ledger(ledger_path)
    if local != expected:
        raise LossConfirmationWorkflowError(
            "Local prior loss attempts differ; changed-identity replacement is forbidden"
        )
    remote = remote_loss_kernel_slugs(cli)
    if set(remote) != set(expected):
        raise LossConfirmationWorkflowError(
            "Remote prior loss attempts differ; changed-identity replacement is forbidden"
        )
    return {
        "schema_version": 1,
        "attempt_ledger_path": str(ledger_path.expanduser().resolve()),
        "expected_prior_loss_slugs": expected,
        "local_prior_loss_slugs": local,
        "remote_prior_loss_slugs": remote,
    }


def kernel_budget(
    *,
    authority: Mapping[str, Any],
    selected: Mapping[str, Any],
    planned_new_slugs: Sequence[str],
    attempted_loss_slugs_before_stage: Sequence[str],
    reserve_worst_case: bool,
) -> dict[str, Any]:
    policy = loss_builder.load_policy()
    historical = list(
        policy["execution"]["historical_bge_kernel_slugs_before_lr_e2"]
    )
    if authority["entry"]["kernel_slug"] != historical[-1]:
        raise LossConfirmationWorkflowError("Frozen-v4 baseline slug differs from budget ledger")
    lr_entries = candidate_runner.build_lr_entries(authority, write=False)
    e2_entry = selected["epoch_receipt"]["e2_entry"]
    consumed = [*historical, *(entry["kernel_slug"] for entry in lr_entries), e2_entry["kernel_slug"]]
    if len(consumed) != 7 or len(set(consumed)) != 7:
        raise LossConfirmationWorkflowError("Pre-loss BGE kernel union is not exactly seven")
    planned = _validate_loss_slug_sequence(
        planned_new_slugs, label="Planned loss kernels"
    )
    if len(set(planned)) != len(planned) or set(planned) & set(consumed):
        raise LossConfirmationWorkflowError("New loss kernel slugs collide with prior history")
    attempted = _validate_loss_slug_sequence(
        attempted_loss_slugs_before_stage, label="Attempted loss kernels"
    )
    if set(attempted) & set(consumed):
        raise LossConfirmationWorkflowError("Attempted loss slugs collide with prior history")
    unplanned_attempts = sorted(set(attempted) - set(planned))
    reserved_slots = 3 if reserve_worst_case else len(planned)
    cap = int(policy["execution"]["max_total_bge_kernel_slugs"])
    projected_total = (
        len(consumed) + len(unplanned_attempts) + max(len(planned), reserved_slots)
    )
    if projected_total > cap:
        raise LossConfirmationWorkflowError("Loss workflow could exceed the total BGE kernel cap")
    return {
        "schema_version": 1,
        "counting_identity": "unique kernel_slug union",
        "historical_before_lr_e2": historical,
        "lr_e2_slugs": [*(entry["kernel_slug"] for entry in lr_entries), e2_entry["kernel_slug"]],
        "prior_unique_slugs": consumed,
        "attempt_ledger_path": str(ATTEMPT_LEDGER_PATH),
        "attempted_loss_slugs_before_stage": attempted,
        "unplanned_attempted_loss_slugs": unplanned_attempts,
        "planned_new_slugs": planned,
        "reserved_worst_case_new": reserved_slots,
        "maximum_total": cap,
        "projected_total": projected_total,
    }


def build_entries(
    authority: Mapping[str, Any],
    selected: Mapping[str, Any],
    *,
    keys: Sequence[str],
    parents: Mapping[str, Mapping[str, Any]],
    write: bool,
) -> list[dict[str, Any]]:
    specs = [loss_builder.variant_spec(key) for key in keys]
    entries = loss_builder.build_campaign(
        owner=OWNER,
        baseline_context=authority["context"],
        baseline_entry=authority["entry"],
        specs=specs,
        parents=parents,
        write=write,
    )
    if [entry["key"] for entry in entries] != list(keys):
        raise LossConfirmationWorkflowError("Loss execution order differs")
    return entries


def expected_dataset_sources(entry: Mapping[str, Any]) -> list[str]:
    context = candidate_builder.validate_baseline_context(entry["baseline_context"])
    return [
        str(entry["validation_dataset"]),
        str(entry["checkpoint_dataset"]),
        context["dataset_ref"],
        baseline_launcher.CREDENTIALS_DATASET,
    ]


def runner_command(entry: Mapping[str, Any], *, env_file: Path) -> list[str]:
    loss_builder.load_notebook(Path(entry["notebook"]), entry=entry)
    context = candidate_builder.validate_baseline_context(entry["baseline_context"])
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_kaggle_notebook.py"),
        str(entry["notebook"]),
        "--env-file",
        str(env_file),
        "--slug",
        str(entry["kernel_slug"]),
        "--title",
        str(entry["title"]),
        "--dataset",
        str(entry["validation_dataset"]),
        "--dataset",
        str(entry["checkpoint_dataset"]),
        "--dataset",
        context["dataset_ref"],
        "--no-env-sources",
        "--no-gpu-check",
        "--no-download",
        "--dry-run",
    ]


def run_inner_runner(command: list[str]) -> None:
    candidate_runner.run_inner_runner(command)


def validate_staged_kernel_metadata(entry: Mapping[str, Any]) -> dict[str, Any]:
    stage_dir = kaggle.STAGE_ROOT / str(entry["kernel_slug"])
    metadata_path = stage_dir / "kernel-metadata.json"
    if not metadata_path.is_file():
        raise LossConfirmationWorkflowError("Loss dry-run produced no kernel metadata")
    metadata = _read_json(metadata_path)
    expected = {
        "id": f"{OWNER}/{entry['kernel_slug']}",
        "title": entry["title"],
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
        "dataset_sources": expected_dataset_sources(entry),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise LossConfirmationWorkflowError(f"Staged loss metadata differs at {key}")
    loss_builder.load_notebook(stage_dir / "notebook.ipynb", entry=entry)
    if list(stage_dir.rglob("ood_validation_predictions.parquet")):
        raise LossConfirmationWorkflowError("Staged loss kernel contains OOD predictions")
    return metadata


def push_after_final_gates(
    cli: list[str],
    *,
    kernel_ref: str,
    entry: Mapping[str, Any],
    authority: Mapping[str, Any],
    run_timeout: int,
    expected_prior_loss_slugs: Sequence[str],
    attempt_ledger_path: Path = ATTEMPT_LEDGER_PATH,
) -> None:
    validate_staged_kernel_metadata(entry)
    audit_prior_attempts_before_push(
        cli,
        expected_prior_slugs=expected_prior_loss_slugs,
        ledger_path=attempt_ledger_path,
    )
    baseline_launcher.confirm_remote_absence(cli, kernel_ref)
    # Catch an old-identity attempt appearing during the two-pass exact-slug
    # absence check.  The private Dataset gate remains the final remote read.
    audit_prior_attempts_before_push(
        cli,
        expected_prior_slugs=expected_prior_loss_slugs,
        ledger_path=attempt_ledger_path,
    )
    candidate_runner.verify_remote_baseline_dataset(cli, authority)
    record_kernel_push_intent(
        path=attempt_ledger_path,
        expected_prior_slugs=expected_prior_loss_slugs,
        kernel_slug=str(entry["kernel_slug"]),
    )
    stage_dir = kaggle.STAGE_ROOT / str(entry["kernel_slug"])
    result = kaggle.run_command(
        cli
        + [
            "kernels",
            "push",
            "-p",
            str(stage_dir),
            "--timeout",
            str(run_timeout),
            "--accelerator",
            "NvidiaTeslaT4",
        ]
    )
    if "not valid dataset sources" in result.stdout.casefold():
        raise LossConfirmationWorkflowError("Kaggle rejected a loss Dataset attachment")
    candidate_runner.verify_remote_candidate_sources(
        cli,
        kernel_ref=kernel_ref,
        entry=entry,
    )


def _expected_loss_receipt(entry: Mapping[str, Any]) -> dict[str, Any]:
    parent = loss_builder.validate_parent_receipt(entry["parent"])
    return {
        "schema_version": 1,
        "workflow": loss_builder.WORKFLOW,
        "stage": entry["stage"],
        "role_in_stage": entry["role_in_stage"],
        "seed": entry["seed"],
        "loss_variant": entry["loss_variant"],
        "parent": {
            "run_id": parent["run_id"],
            "experiment": parent["experiment"],
            "campaign_identity_sha256": parent["campaign_identity_sha256"],
            "recipe_sha256": parent["recipe_sha256"],
            "loss_hook_sha256": parent["loss_hook_sha256"],
        },
        "fresh_start": True,
        "checkpoint_resume": False,
        "workflow_source_ledger": entry["workflow_source_ledger"],
        "workflow_ledger_sha256": entry["workflow_ledger_sha256"],
        "primary_split": "iid",
        "diagnostic_splits": ["hard"],
        "ood_macro_average_precision": -1.0,
        "ood_comparison": None,
    }


def validate_loss_output(
    directory: Path,
    *,
    entry: Mapping[str, Any],
    require_sheets: bool = True,
) -> dict[str, Any]:
    directory = directory.expanduser().resolve(strict=True)
    if list(directory.rglob("ood_validation_predictions.parquet")):
        raise LossConfirmationWorkflowError("Loss output contains forbidden OOD predictions")
    forbidden_weight_patterns = (
        "model.safetensors*",
        "pytorch_model*.bin",
        "checkpoint-*",
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
        "rng_state*.pth",
        "training_args.bin",
        "trainer_state.json",
    )
    if any(list(directory.rglob(pattern)) for pattern in forbidden_weight_patterns):
        raise LossConfirmationWorkflowError("Loss output contains forbidden checkpoint state")
    completion_path = directory / baseline_uploader.COMPLETION_FILENAME
    completion = _read_json(completion_path)
    run_id = str(completion.get("run_id", ""))
    if re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
        raise LossConfirmationWorkflowError("Loss completion run_id is not 32-hex")
    exact = {
        "loss_variant": entry["loss_variant"],
        "loss_hook_sha256": entry["loss_hook_sha256"],
        "candidate_generator_sha256": entry["candidate_generator_sha256"],
        "loss_confirmation": _expected_loss_receipt(entry),
    }
    for key, value in exact.items():
        if completion.get(key) != value:
            raise LossConfirmationWorkflowError(f"Loss completion differs at {key}")
    parent = loss_builder.validate_parent_receipt(entry["parent"])
    expected_parent = {
        "run_id": parent["run_id"],
        "experiment": parent["experiment"],
        "campaign_identity_sha256": parent["campaign_identity_sha256"],
        "source_sha256": parent["source_sha256"],
        "recipe_sha256": parent["recipe_sha256"],
        "config": parent["config"],
    }
    if completion.get("stage_parent") != expected_parent:
        raise LossConfirmationWorkflowError("Loss completion stage parent differs")
    context = candidate_builder.validate_baseline_context(entry["baseline_context"])
    expected_gate = {
        "status": "passed",
        "dataset_ref": context["dataset_ref"],
        "dataset_version": context["dataset_version"],
        "manifest_sha256": context["manifest_sha256"],
        "baseline_run_id": context["binding"]["baseline_run_id"],
        "baseline_identity_sha256": context["binding"]["campaign_identity_sha256"],
        "baseline_source_sha256": context["binding"]["source_sha256"],
        "ood_predictions": False,
    }
    gate = _read_json(directory / candidate_builder.BASELINE_GATE_FILENAME)
    if gate != expected_gate or completion.get("frozen_baseline_dataset") != expected_gate:
        raise LossConfirmationWorkflowError("Loss output baseline Dataset gate differs")
    # Reuse the exhaustive frozen BGE validator after changing only its hardcoded
    # top-level loss-name expectation in an isolated temporary copy.  The custom
    # hook SHA remains untouched and is still checked against trainer output.
    with tempfile.TemporaryDirectory(prefix="bge-loss-validate-") as temp_dir:
        normalized = Path(temp_dir) / "output"
        shutil.copytree(directory, normalized)
        normalized_completion_path = normalized / baseline_uploader.COMPLETION_FILENAME
        normalized_completion = _read_json(normalized_completion_path)
        normalized_completion["loss_variant"] = loss_builder.PLAIN_BCE
        normalized_completion_path.write_text(
            json.dumps(normalized_completion, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        validator = (
            baseline_launcher.validate_run_output
            if require_sheets
            else baseline_launcher.validate_run_payload
        )
        validated = validator(normalized, entry=entry)
    return {
        **validated,
        "loss_variant": entry["loss_variant"],
        "seed": entry["seed"],
        "baseline_dataset_gate": gate,
        "prediction_binding": _prediction_binding(directory),
    }


def _local_output(entry: Mapping[str, Any]) -> Path | None:
    directory = baseline_launcher.output_root() / str(entry["kernel_slug"])
    if not directory.exists():
        return None
    if not directory.is_dir():
        raise LossConfirmationWorkflowError(f"Loss output path is not a directory: {directory}")
    validate_loss_output(directory, entry=entry)
    return directory


def download_output(
    cli: list[str],
    *,
    entry: Mapping[str, Any],
    full_download: bool,
) -> Path:
    root = baseline_launcher.output_root()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / str(entry["kernel_slug"])
    if destination.exists():
        raise LossConfirmationWorkflowError(f"Refusing to replace loss output: {destination}")
    kernel_ref = f"{OWNER}/{entry['kernel_slug']}"
    staging: Path | None = None
    for attempt in range(1, 4):
        candidate = Path(tempfile.mkdtemp(prefix=f".{entry['kernel_slug']}.download-", dir=root))
        command = cli + [
            "kernels",
            "output",
            kernel_ref,
            "-p",
            str(candidate),
            "--force",
            "--page-size",
            "200",
        ]
        if not full_download:
            command.extend(["--file-pattern", SLIM_OUTPUT_PATTERN])
        result = kaggle.run_command(command, check=False)
        if result.returncode == 0:
            staging = candidate
            break
        shutil.rmtree(candidate)
        if attempt < 3:
            time.sleep(3 if attempt == 1 else 8)
    if staging is None:
        raise LossConfirmationWorkflowError(f"Could not download loss output: {kernel_ref}")
    try:
        validate_loss_output(staging, entry=entry, require_sheets=False)
        baseline_launcher.retry_pending_sheets_sync(staging, kernel_ref=kernel_ref)
        validate_loss_output(staging, entry=entry, require_sheets=True)
        staging.rename(destination)
    except Exception:
        print(f"Invalid loss download preserved at: {staging}", file=sys.stderr)
        raise
    return destination


def execute_entry(
    *,
    cli: list[str],
    env_file: Path,
    authority: Mapping[str, Any],
    entry: Mapping[str, Any],
    poll_interval: int,
    wait_timeout: int,
    run_timeout: int,
    full_download: bool,
    expected_prior_loss_slugs: Sequence[str],
    attempt_ledger_path: Path = ATTEMPT_LEDGER_PATH,
) -> Path:
    existing = _local_output(entry)
    if existing is not None:
        return existing
    kernel_ref = f"{OWNER}/{entry['kernel_slug']}"
    status = baseline_launcher.remote_kernel_status(cli, kernel_ref)
    if status == "absence_unconfirmed":
        run_inner_runner(runner_command(entry, env_file=env_file))
        validate_staged_kernel_metadata(entry)
        push_after_final_gates(
            cli,
            kernel_ref=kernel_ref,
            entry=entry,
            authority=authority,
            run_timeout=run_timeout,
            expected_prior_loss_slugs=expected_prior_loss_slugs,
            attempt_ledger_path=attempt_ledger_path,
        )
        kaggle.wait_for_kernel(
            cli,
            kernel_ref,
            poll_interval=poll_interval,
            wait_timeout=wait_timeout,
        )
    elif status in {"queued", "running"}:
        candidate_runner.verify_remote_candidate_sources(cli, kernel_ref=kernel_ref, entry=entry)
        kaggle.wait_for_kernel(
            cli,
            kernel_ref,
            poll_interval=poll_interval,
            wait_timeout=wait_timeout,
        )
    elif status in kaggle.TERMINAL_FAILURE:
        kaggle.run_command(cli + ["kernels", "logs", kernel_ref], check=False)
        raise LossConfirmationWorkflowError(
            f"Loss kernel {kernel_ref} is terminally failed; resubmission is forbidden"
        )
    elif status in kaggle.TERMINAL_SUCCESS:
        candidate_runner.verify_remote_candidate_sources(cli, kernel_ref=kernel_ref, entry=entry)
    else:
        raise LossConfirmationWorkflowError(f"Unexpected loss kernel status: {status!r}")
    return download_output(cli, entry=entry, full_download=full_download)


def _load_comparison_side(
    directory: Path,
    *,
    label: str,
) -> dict[str, Any]:
    if list(directory.rglob("ood_validation_predictions.parquet")):
        raise LossConfirmationWorkflowError(f"{label} contains forbidden OOD predictions")
    completion = _read_json(directory / baseline_uploader.COMPLETION_FILENAME)
    report = completion.get("training_report")
    if not isinstance(report, Mapping) or report.get("validation_splits", {}).get("ood") != baseline_builder.OOD_SENTINEL:
        raise LossConfirmationWorkflowError(f"{label} changed the exact OOD sentinel")
    predictions: dict[str, Path] = {}
    for split in comparator.SPLITS:
        paths = list(directory.rglob(baseline_uploader.PREDICTION_FILENAMES[split]))
        if len(paths) != 1 or not paths[0].is_file() or paths[0].stat().st_size <= 0:
            raise LossConfirmationWorkflowError(f"{label} has no exact {split} predictions")
        predictions[split] = paths[0]
    return {
        "directory": directory,
        "completion": completion,
        "report": report,
        "run_id": completion["run_id"],
        "experiment": completion["experiment"],
        "predictions": predictions,
    }


def _paired_split(
    anchor: Mapping[str, Any],
    challenger: Mapping[str, Any],
    *,
    split: str,
    family: str,
    permutations: int,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    result = comparator.compare_prediction_frames(
        comparator.read_prediction_artifact(anchor["predictions"][split]),
        comparator.read_prediction_artifact(challenger["predictions"][split]),
        permutations=permutations,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
    )
    if result["examples"] != comparator.EXPECTED_ROWS[split]:
        raise LossConfirmationWorkflowError(f"Loss comparison has unexpected {split} rows")
    anchor_ap = comparator._metric_from_report(anchor["report"], split=split, label="loss anchor")
    challenger_ap = comparator._metric_from_report(
        challenger["report"], split=split, label="loss challenger"
    )
    if not math.isclose(float(result["baseline_macro_average_precision"]), anchor_ap, abs_tol=1e-12):
        raise LossConfirmationWorkflowError(f"Loss anchor {split} report/predictions differ")
    if not math.isclose(float(result["candidate_macro_average_precision"]), challenger_ap, abs_tol=1e-12):
        raise LossConfirmationWorkflowError(f"Loss challenger {split} report/predictions differ")
    result["p_value_holm"] = result["p_value"]
    result["holm_family"] = family
    result["holm_family_size"] = 1
    return result


def validate_loss_comparison(
    comparison: Mapping[str, Any],
    *,
    anchor: Mapping[str, Any],
    challenger: Mapping[str, Any],
    family: str,
    baseline_manifest_sha256: str,
) -> None:
    exact_keys = {
        "schema_version",
        "status",
        "baseline_run_id",
        "candidate_run_id",
        "baseline_experiment",
        "candidate_experiment",
        "baseline_manifest_sha256",
        "method",
        "confidence_interval_method",
        "multiple_testing_correction",
        "holm_family",
        "holm_family_members",
        "primary_split",
        "diagnostic_splits",
        "practical_tie_margin",
        "iid_practical_relation",
        "ood_policy",
        "splits",
    }
    if set(comparison) != exact_keys:
        raise LossConfirmationWorkflowError("Loss comparison fields differ")
    expected = {
        "schema_version": 1,
        "status": "ready_ood_disabled",
        "baseline_run_id": anchor["run_id"],
        "candidate_run_id": challenger["run_id"],
        "baseline_experiment": anchor["experiment"],
        "candidate_experiment": challenger["experiment"],
        "baseline_manifest_sha256": baseline_manifest_sha256,
        "method": "paired_component_permutation",
        "confidence_interval_method": "paired_component_bootstrap_percentile",
        "multiple_testing_correction": "holm_within_planned_candidate_family_per_split",
        "holm_family": family,
        "holm_family_members": [challenger["experiment"]],
        "primary_split": "iid",
        "diagnostic_splits": ["hard"],
        "practical_tie_margin": 0.002,
        "ood_policy": "disabled_train_contaminated_no_paired_comparison",
    }
    for key, value in expected.items():
        if comparison.get(key) != value:
            raise LossConfirmationWorkflowError(f"Loss comparison differs at {key}")
    splits = comparison.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"iid", "hard", "ood"}:
        raise LossConfirmationWorkflowError("Loss comparison split contract differs")
    if splits.get("ood") != comparator._ood_result():
        raise LossConfirmationWorkflowError("Loss comparison fabricated OOD evidence")
    result_keys = {
        "examples",
        "categories",
        "components",
        "baseline_macro_average_precision",
        "candidate_macro_average_precision",
        "delta_macro_average_precision",
        "p_value",
        "ci95_low",
        "ci95_high",
        "permutations",
        "bootstrap_resamples",
        "seed",
        "p_value_holm",
        "holm_family",
        "holm_family_size",
    }
    resample_contract: tuple[int, int] | None = None
    for split_index, split in enumerate(comparator.SPLITS):
        result = splits.get(split)
        if not isinstance(result, Mapping) or set(result) != result_keys:
            raise LossConfirmationWorkflowError(
                f"Loss comparison {split} result fields differ"
            )
        integer_expected = {
            "examples": comparator.EXPECTED_ROWS[split],
            "categories": EXPECTED_COMPARISON_CATEGORIES,
            "seed": 42 + split_index,
            "holm_family_size": 1,
        }
        for key, expected_value in integer_expected.items():
            if isinstance(result.get(key), bool) or result.get(key) != expected_value:
                raise LossConfirmationWorkflowError(
                    f"Loss comparison {split} differs at {key}"
                )
        for key in ("components", "permutations", "bootstrap_resamples"):
            value = result.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise LossConfirmationWorkflowError(
                    f"Loss comparison {split} has invalid {key}"
                )
        observed_resamples = (
            int(result["permutations"]),
            int(result["bootstrap_resamples"]),
        )
        if resample_contract is None:
            resample_contract = observed_resamples
        elif observed_resamples != resample_contract:
            raise LossConfirmationWorkflowError(
                "Loss comparison resample counts differ by split"
            )
        numeric: dict[str, float] = {}
        for key in (
            "baseline_macro_average_precision",
            "candidate_macro_average_precision",
            "delta_macro_average_precision",
            "p_value",
            "p_value_holm",
            "ci95_low",
            "ci95_high",
        ):
            value = result.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise LossConfirmationWorkflowError(
                    f"Loss comparison {split} has invalid {key}"
                )
            numeric[key] = float(value)
        if not 0 <= numeric["baseline_macro_average_precision"] <= 1 or not 0 <= numeric[
            "candidate_macro_average_precision"
        ] <= 1:
            raise LossConfirmationWorkflowError(
                f"Loss comparison {split} macro AP is outside [0, 1]"
            )
        if not 0 <= numeric["p_value"] <= 1 or numeric["p_value_holm"] != numeric[
            "p_value"
        ]:
            raise LossConfirmationWorkflowError(
                f"Loss comparison {split} one-member Holm value differs"
            )
        if numeric["ci95_low"] > numeric["ci95_high"]:
            raise LossConfirmationWorkflowError(
                f"Loss comparison {split} confidence interval is reversed"
            )
        expected_delta = (
            numeric["candidate_macro_average_precision"]
            - numeric["baseline_macro_average_precision"]
        )
        if not math.isclose(
            numeric["delta_macro_average_precision"],
            expected_delta,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise LossConfirmationWorkflowError(
                f"Loss comparison {split} delta is inconsistent"
            )
        anchor_ap = comparator._metric_from_report(
            anchor["report"], split=split, label="loss comparison anchor"
        )
        challenger_ap = comparator._metric_from_report(
            challenger["report"], split=split, label="loss comparison challenger"
        )
        if not math.isclose(
            numeric["baseline_macro_average_precision"],
            anchor_ap,
            rel_tol=0,
            abs_tol=1e-12,
        ) or not math.isclose(
            numeric["candidate_macro_average_precision"],
            challenger_ap,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise LossConfirmationWorkflowError(
                f"Loss comparison {split} saved metrics differ from raw reports"
            )
        if result.get("holm_family") != family or result.get("holm_family_size") != 1:
            raise LossConfirmationWorkflowError(f"Loss comparison {split} Holm family differs")
    iid_delta = splits["iid"].get("delta_macro_average_precision")
    if (
        isinstance(iid_delta, bool)
        or not isinstance(iid_delta, (int, float))
        or not math.isfinite(float(iid_delta))
    ):
        raise LossConfirmationWorkflowError("Loss comparison IID delta is invalid")
    if comparison.get("iid_practical_relation") != comparator._practical_relation(
        float(iid_delta), 0.002
    ):
        raise LossConfirmationWorkflowError("Loss comparison practical relation differs")


def materialize_comparison(
    *,
    anchor_dir: Path,
    challenger_dir: Path,
    output_dir: Path,
    family: str,
    baseline_manifest_sha256: str,
    sync_sheets: bool,
    permutations: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    output_dir = comparator.validate_output_isolation(output_dir, [anchor_dir, challenger_dir])
    anchor = _load_comparison_side(anchor_dir, label="loss anchor")
    challenger = _load_comparison_side(challenger_dir, label="loss challenger")
    if anchor["run_id"] == challenger["run_id"]:
        raise LossConfirmationWorkflowError("Loss comparison reused one run_id")
    splits = {
        split: _paired_split(
            anchor,
            challenger,
            split=split,
            family=family,
            permutations=permutations,
            bootstrap_resamples=bootstrap_resamples,
            seed=42 + index,
        )
        for index, split in enumerate(comparator.SPLITS)
    }
    comparison = {
        "schema_version": 1,
        "status": "ready_ood_disabled",
        "baseline_run_id": anchor["run_id"],
        "candidate_run_id": challenger["run_id"],
        "baseline_experiment": anchor["experiment"],
        "candidate_experiment": challenger["experiment"],
        "baseline_manifest_sha256": baseline_manifest_sha256,
        "method": "paired_component_permutation",
        "confidence_interval_method": "paired_component_bootstrap_percentile",
        "multiple_testing_correction": "holm_within_planned_candidate_family_per_split",
        "holm_family": family,
        "holm_family_members": [challenger["experiment"]],
        "primary_split": "iid",
        "diagnostic_splits": ["hard"],
        "practical_tie_margin": 0.002,
        "iid_practical_relation": comparator._practical_relation(
            float(splits["iid"]["delta_macro_average_precision"]), 0.002
        ),
        "ood_policy": "disabled_train_contaminated_no_paired_comparison",
        "splits": {
            "iid": splits["iid"],
            "hard": splits["hard"],
            "ood": comparator._ood_result(),
        },
    }
    validate_loss_comparison(
        comparison,
        anchor=anchor,
        challenger=challenger,
        family=family,
        baseline_manifest_sha256=baseline_manifest_sha256,
    )
    augmented = {
        **challenger["completion"],
        "experiment_group": "sft",
        "baseline_comparison": comparison,
    }
    candidate_runner.validate_augmented_completion(
        augmented,
        expected_baseline_run_id=anchor["run_id"],
    )
    comparison_path = output_dir / "baseline_comparison.json"
    completion_path = output_dir / "completion_with_comparison.json"
    _write_json(comparison_path, comparison)
    _write_json(completion_path, augmented)
    marker_declaration: dict[str, Any] | None = None
    if sync_sheets:
        marker = candidate_runner.sync_augmented_completion(
            augmented,
            output_dir=output_dir,
            expected_baseline_run_id=anchor["run_id"],
        )
        marker_path = output_dir / candidate_runner.COMPARISON_SYNC_FILENAME
        marker_declaration = {
            "path": str(marker_path),
            "sha256": comparator.sha256_file(marker_path),
            "completion_canonical_sha256": marker["completion_canonical_sha256"],
        }
    return {
        "anchor": anchor,
        "challenger": challenger,
        "comparison": comparison,
        "augmented_completion": augmented,
        "comparison_path": comparison_path,
        "completion_path": completion_path,
        "comparison_sync_marker": marker_declaration,
        "comparison_sheets_synced": sync_sheets,
    }


def summarize_screen(
    *,
    authority: Mapping[str, Any],
    selected: Mapping[str, Any],
    entry: Mapping[str, Any],
    directory: Path,
    output_dir: Path,
    sync_sheets: bool,
    permutations: int,
    bootstrap_resamples: int,
    budget: Mapping[str, Any],
) -> dict[str, Any]:
    validate_loss_output(directory, entry=entry)
    result = materialize_comparison(
        anchor_dir=Path(selected["directory"]),
        challenger_dir=directory,
        output_dir=output_dir,
        family=SCREEN_FAMILY,
        baseline_manifest_sha256=authority["context"]["manifest_sha256"],
        sync_sheets=sync_sheets,
        permutations=permutations,
        bootstrap_resamples=bootstrap_resamples,
    )
    iid_delta = float(result["comparison"]["splits"]["iid"]["delta_macro_average_precision"])
    accepted = screen_accepts_challenger(iid_delta)
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "workflow": loss_builder.WORKFLOW,
        "campaign": baseline_builder.CAMPAIGN,
        "stage": "loss_screen",
        "family_name": SCREEN_FAMILY,
        "frozen_baseline_dataset": candidate_runner._baseline_dataset_receipt(authority),
        "lr_selection_receipt_sha256": comparator.sha256_file(
            Path(selected["lr_receipt"]["_receipt_path"])
        ),
        "epoch_selection_receipt_sha256": comparator.sha256_file(
            Path(selected["epoch_receipt"]["_receipt_path"])
        ),
        "anchor": {
            "parent": selected["parent"],
            "directory": str(selected["directory"]),
            "prediction_binding": selected["prediction_binding"],
            "loss_variant": loss_builder.PLAIN_BCE,
            "reused_existing_run": True,
        },
        "challenger": {
            "run_id": result["challenger"]["run_id"],
            "experiment": result["challenger"]["experiment"],
            "kernel_slug": entry["kernel_slug"],
            "directory": str(directory),
            "identity_sha256": entry["identity_sha256"],
            "recipe_sha256": entry["recipe_sha256"],
            "loss_variant": loss_builder.SQRT_BALANCED_BCE,
            "loss_hook_sha256": entry["loss_hook_sha256"],
            "prediction_binding": _prediction_binding(directory),
        },
        "primary_split": "iid",
        "diagnostic_splits": ["hard"],
        "hard_used_for_selection": False,
        "ood": {"macro_average_precision": -1.0, "comparison": None},
        "iid_delta": iid_delta,
        "acceptance_threshold": 0.002,
        "threshold_relation": "strictly_greater_than",
        "seed42_winner": (
            loss_builder.SQRT_BALANCED_BCE if accepted else loss_builder.PLAIN_BCE
        ),
        "challenger_accepted_for_seed17": accepted,
        "comparison_path": str(result["comparison_path"]),
        "comparison_sha256": comparator.sha256_file(result["comparison_path"]),
        "completion_with_comparison_path": str(result["completion_path"]),
        "completion_with_comparison_sha256": comparator.sha256_file(result["completion_path"]),
        "comparison_sheets_synced": result["comparison_sheets_synced"],
        "comparison_sync_marker": result["comparison_sync_marker"],
        "kernel_budget": dict(budget),
    }
    receipt_path = output_dir / (
        SCREEN_RECEIPT_FILENAME if sync_sheets else LOCAL_SCREEN_RECEIPT_FILENAME
    )
    _write_json(receipt_path, receipt)
    return {"receipt": receipt, "receipt_path": receipt_path, "comparison": result}


def load_screen_receipt(
    path: Path,
    *,
    authority: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_path = path.expanduser().resolve(strict=True)
    receipt = _read_json(receipt_path)
    required = {
        "schema_version",
        "status",
        "workflow",
        "campaign",
        "stage",
        "family_name",
        "frozen_baseline_dataset",
        "lr_selection_receipt_sha256",
        "epoch_selection_receipt_sha256",
        "anchor",
        "challenger",
        "primary_split",
        "diagnostic_splits",
        "hard_used_for_selection",
        "ood",
        "iid_delta",
        "acceptance_threshold",
        "threshold_relation",
        "seed42_winner",
        "challenger_accepted_for_seed17",
        "comparison_path",
        "comparison_sha256",
        "completion_with_comparison_path",
        "completion_with_comparison_sha256",
        "comparison_sheets_synced",
        "comparison_sync_marker",
        "kernel_budget",
    }
    if set(receipt) != required:
        raise LossConfirmationWorkflowError("Loss-screen receipt fields differ")
    expected = {
        "schema_version": 1,
        "status": "complete",
        "workflow": loss_builder.WORKFLOW,
        "campaign": baseline_builder.CAMPAIGN,
        "stage": "loss_screen",
        "family_name": SCREEN_FAMILY,
        "frozen_baseline_dataset": candidate_runner._baseline_dataset_receipt(authority),
        "lr_selection_receipt_sha256": comparator.sha256_file(
            Path(selected["lr_receipt"]["_receipt_path"])
        ),
        "epoch_selection_receipt_sha256": comparator.sha256_file(
            Path(selected["epoch_receipt"]["_receipt_path"])
        ),
        "primary_split": "iid",
        "diagnostic_splits": ["hard"],
        "hard_used_for_selection": False,
        "ood": {"macro_average_precision": -1.0, "comparison": None},
        "acceptance_threshold": 0.002,
        "threshold_relation": "strictly_greater_than",
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise LossConfirmationWorkflowError(f"Loss-screen receipt differs at {key}")
    anchor = receipt.get("anchor")
    if not isinstance(anchor, Mapping) or anchor != {
        "parent": selected["parent"],
        "directory": str(selected["directory"]),
        "prediction_binding": selected["prediction_binding"],
        "loss_variant": loss_builder.PLAIN_BCE,
        "reused_existing_run": True,
    }:
        raise LossConfirmationWorkflowError("Loss-screen anchor binding differs")
    challenger = receipt.get("challenger")
    if not isinstance(challenger, Mapping):
        raise LossConfirmationWorkflowError("Loss-screen challenger binding is missing")
    entry = build_entries(
        authority,
        selected,
        keys=["screen_sqrt_s42"],
        parents={"screen_sqrt_s42": selected["parent"]},
        write=False,
    )[0]
    directory = baseline_launcher.output_root() / str(entry["kernel_slug"])
    validate_loss_output(directory, entry=entry)
    completion = _read_json(directory / baseline_uploader.COMPLETION_FILENAME)
    expected_challenger = {
        "run_id": completion["run_id"],
        "experiment": entry["experiment"],
        "kernel_slug": entry["kernel_slug"],
        "directory": str(directory),
        "identity_sha256": entry["identity_sha256"],
        "recipe_sha256": entry["recipe_sha256"],
        "loss_variant": loss_builder.SQRT_BALANCED_BCE,
        "loss_hook_sha256": entry["loss_hook_sha256"],
        "prediction_binding": _prediction_binding(directory),
    }
    if dict(challenger) != expected_challenger:
        raise LossConfirmationWorkflowError("Loss-screen challenger output differs")
    expected_budget = kernel_budget(
        authority=authority,
        selected=selected,
        planned_new_slugs=[entry["kernel_slug"]],
        attempted_loss_slugs_before_stage=[],
        reserve_worst_case=True,
    )
    if receipt.get("kernel_budget") != expected_budget:
        raise LossConfirmationWorkflowError("Loss-screen kernel budget receipt differs")
    comparison_path = Path(str(receipt["comparison_path"])).expanduser().resolve(strict=True)
    completion_path = Path(str(receipt["completion_with_comparison_path"])).expanduser().resolve(strict=True)
    if comparison_path != receipt_path.parent / "baseline_comparison.json" or completion_path != receipt_path.parent / "completion_with_comparison.json":
        raise LossConfirmationWorkflowError("Loss-screen comparison paths differ")
    if comparator.sha256_file(comparison_path) != _require_hash(receipt["comparison_sha256"], "screen comparison SHA"):
        raise LossConfirmationWorkflowError("Loss-screen comparison changed")
    if comparator.sha256_file(completion_path) != _require_hash(
        receipt["completion_with_comparison_sha256"], "screen completion SHA"
    ):
        raise LossConfirmationWorkflowError("Loss-screen augmented completion changed")
    comparison = _read_json(comparison_path)
    augmented = _read_json(completion_path)
    raw = dict(augmented)
    embedded_comparison = raw.pop("baseline_comparison", None)
    if raw != completion or embedded_comparison != comparison:
        raise LossConfirmationWorkflowError("Loss-screen augmented completion differs")
    validate_loss_comparison(
        comparison,
        anchor=_load_comparison_side(Path(selected["directory"]), label="loss anchor"),
        challenger=_load_comparison_side(directory, label="loss challenger"),
        family=SCREEN_FAMILY,
        baseline_manifest_sha256=authority["context"]["manifest_sha256"],
    )
    candidate_runner.validate_augmented_completion(
        augmented,
        expected_baseline_run_id=selected["parent"]["run_id"],
    )
    iid_delta = float(comparison["splits"]["iid"]["delta_macro_average_precision"])
    bound_iid_delta = exact_bound_delta(
        receipt.get("iid_delta"), iid_delta, label="Loss-screen receipt"
    )
    accepted = screen_accepts_challenger(iid_delta)
    expected_winner = loss_builder.SQRT_BALANCED_BCE if accepted else loss_builder.PLAIN_BCE
    if (
        receipt["challenger_accepted_for_seed17"] is not accepted
        or receipt["seed42_winner"] != expected_winner
    ):
        raise LossConfirmationWorkflowError("Loss-screen branch was not recomputed from IID")
    if receipt.get("comparison_sheets_synced") is not True:
        raise LossConfirmationWorkflowError("Seed17 requires synced seed42 comparison")
    marker_declaration = receipt.get("comparison_sync_marker")
    marker_path = receipt_path.parent / candidate_runner.COMPARISON_SYNC_FILENAME
    if not isinstance(marker_declaration, Mapping) or marker_declaration.get("path") != str(marker_path):
        raise LossConfirmationWorkflowError("Loss-screen Sheets marker path differs")
    if comparator.sha256_file(marker_path) != _require_hash(
        marker_declaration.get("sha256"), "loss-screen marker SHA"
    ):
        raise LossConfirmationWorkflowError("Loss-screen Sheets marker changed")
    marker = _read_json(marker_path)
    candidate_runner.validate_comparison_sync_marker(marker, completion=augmented)
    if marker_declaration.get("completion_canonical_sha256") != marker.get(
        "completion_canonical_sha256"
    ):
        raise LossConfirmationWorkflowError("Loss-screen marker/completion binding differs")
    return {
        **receipt,
        # Never propagate a receipt-sourced float into the two-seed mean.
        "iid_delta": bound_iid_delta,
        "entry": entry,
        "directory": directory,
        "completion": completion,
        "comparison": comparison,
        "accepted": accepted,
        "_receipt_path": receipt_path,
    }


def summarize_confirmation(
    *,
    authority: Mapping[str, Any],
    selected: Mapping[str, Any],
    screen: Mapping[str, Any],
    bce_entry: Mapping[str, Any],
    bce_dir: Path,
    sqrt_entry: Mapping[str, Any] | None,
    sqrt_dir: Path | None,
    output_dir: Path,
    sync_sheets: bool,
    permutations: int,
    bootstrap_resamples: int,
    budget: Mapping[str, Any],
) -> dict[str, Any]:
    bce_validation = validate_loss_output(bce_dir, entry=bce_entry)
    seed42_delta = float(screen["iid_delta"])
    seed17_delta: float | None = None
    comparison_result: dict[str, Any] | None = None
    if screen["accepted"]:
        if sqrt_entry is None or sqrt_dir is None:
            raise LossConfirmationWorkflowError("Accepted loss screen requires matched seed17 challenger")
        validate_loss_output(sqrt_dir, entry=sqrt_entry)
        comparison_result = materialize_comparison(
            anchor_dir=bce_dir,
            challenger_dir=sqrt_dir,
            output_dir=output_dir / "paired_seed17",
            family=SEED17_FAMILY,
            baseline_manifest_sha256=authority["context"]["manifest_sha256"],
            sync_sheets=sync_sheets,
            permutations=permutations,
            bootstrap_resamples=bootstrap_resamples,
        )
        seed17_delta = float(
            comparison_result["comparison"]["splits"]["iid"][
                "delta_macro_average_precision"
            ]
        )
    else:
        if sqrt_entry is not None or sqrt_dir is not None:
            raise LossConfirmationWorkflowError("Rejected loss screen must not run seed17 challenger")
    decision = final_loss_decision(seed42_delta, seed17_delta)
    challenger_final = bool(decision["challenger_accepted"])
    final_loss = str(decision["selected_loss_variant"])
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "workflow": loss_builder.WORKFLOW,
        "campaign": baseline_builder.CAMPAIGN,
        "stage": "seed_confirmation",
        "loss_screen_receipt_sha256": comparator.sha256_file(
            Path(screen["_receipt_path"])
        ),
        "branch": (
            "matched_bce_and_challenger_seed17"
            if screen["accepted"]
            else "matched_bce_seed17_only"
        ),
        "execution_order": (
            [bce_entry["kernel_slug"], sqrt_entry["kernel_slug"]]
            if sqrt_entry is not None
            else [bce_entry["kernel_slug"]]
        ),
        "seed42": {
            "anchor_run_id": selected["parent"]["run_id"],
            "challenger_run_id": screen["challenger"]["run_id"],
            "iid_delta": seed42_delta,
            "screen_threshold": 0.002,
            "challenger_passed": screen["accepted"],
        },
        "seed17": {
            "bce_run_id": bce_validation["run_id"],
            "bce_experiment": bce_entry["experiment"],
            "bce_kernel_slug": bce_entry["kernel_slug"],
            "bce_identity_sha256": bce_entry["identity_sha256"],
            "bce_recipe_sha256": bce_entry["recipe_sha256"],
            "bce_loss_hook_sha256": bce_entry["loss_hook_sha256"],
            "bce_parent_run_id": bce_entry["parent"]["run_id"],
            "bce_prediction_binding": _prediction_binding(bce_dir),
            "challenger_run_id": (
                comparison_result["challenger"]["run_id"]
                if comparison_result is not None
                else None
            ),
            "challenger_kernel_slug": (
                sqrt_entry["kernel_slug"] if sqrt_entry is not None else None
            ),
            "challenger_identity_sha256": (
                sqrt_entry["identity_sha256"] if sqrt_entry is not None else None
            ),
            "challenger_recipe_sha256": (
                sqrt_entry["recipe_sha256"] if sqrt_entry is not None else None
            ),
            "challenger_loss_hook_sha256": (
                sqrt_entry["loss_hook_sha256"] if sqrt_entry is not None else None
            ),
            "challenger_parent_run_id": (
                sqrt_entry["parent"]["run_id"] if sqrt_entry is not None else None
            ),
            "challenger_prediction_binding": (
                _prediction_binding(sqrt_dir) if sqrt_dir is not None else None
            ),
            "iid_delta": seed17_delta,
            "comparison_artifacts": (
                {
                    "comparison_path": str(comparison_result["comparison_path"]),
                    "comparison_sha256": comparator.sha256_file(
                        comparison_result["comparison_path"]
                    ),
                    "completion_with_comparison_path": str(
                        comparison_result["completion_path"]
                    ),
                    "completion_with_comparison_sha256": comparator.sha256_file(
                        comparison_result["completion_path"]
                    ),
                }
                if comparison_result is not None
                else None
            ),
        },
        "final_gate": {
            "seed42_delta_strictly_positive": seed42_delta > 0,
            "seed17_delta_strictly_positive": (
                seed17_delta > 0 if seed17_delta is not None else None
            ),
            "mean_iid_delta": (
                (seed42_delta + seed17_delta) / 2
                if seed17_delta is not None
                else None
            ),
            "required_mean_iid_delta": 0.002,
            "challenger_accepted": challenger_final,
        },
        "selected_loss_variant": final_loss,
        "selected_recipe": dict(selected["parent"]["config"]),
        "selected_recipe_sha256": selected["parent"]["recipe_sha256"],
        "selected_loss_hook_sha256": loss_builder.LOSS_HOOK_SHA256[final_loss],
        "primary_split": "iid",
        "diagnostic_splits": ["hard"],
        "hard_used_for_selection": False,
        "ood": {"macro_average_precision": -1.0, "comparison": None},
        "seed17_comparison_sheets_synced": (
            comparison_result["comparison_sheets_synced"]
            if comparison_result is not None
            else None
        ),
        "seed17_comparison_sync_marker": (
            comparison_result["comparison_sync_marker"]
            if comparison_result is not None
            else None
        ),
        "kernel_budget": dict(budget),
    }
    receipt_path = output_dir / (
        FINAL_RECEIPT_FILENAME if sync_sheets else LOCAL_FINAL_RECEIPT_FILENAME
    )
    _write_json(receipt_path, receipt)
    return {"receipt": receipt, "receipt_path": receipt_path}


def _stage_entry(entry: Mapping[str, Any], env_file: Path) -> None:
    run_inner_runner(runner_command(entry, env_file=env_file))
    run_inner_runner(runner_command(entry, env_file=env_file))
    validate_staged_kernel_metadata(entry)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guarded post-LR/e2 BGE loss and seed confirmation"
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--stage", choices=("screen", "confirm"), default="screen")
    parser.add_argument("--baseline-dataset-version", type=int)
    parser.add_argument("--baseline-manifest-sha256")
    parser.add_argument("--baseline-source-dir", type=Path)
    parser.add_argument("--baseline-stage-dir", type=Path, default=baseline_uploader.STAGE_DIR)
    parser.add_argument("--lr-receipt", type=Path, default=candidate_runner.DEFAULT_LR_RECEIPT)
    parser.add_argument("--epoch-receipt", type=Path, default=DEFAULT_EPOCH_RECEIPT)
    parser.add_argument(
        "--screen-receipt",
        type=Path,
        default=DEFAULT_REPORT_ROOT / "screen" / SCREEN_RECEIPT_FILENAME,
    )
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--permutations", type=int, default=2_000)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--full-download", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="stage the next kernel only")
    mode.add_argument(
        "--summarize-local",
        action="store_true",
        help="use already validated outputs without Kaggle/Sheets mutation",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="explicitly allow sequential Kaggle and sft_exps mutations",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not (args.dry_run or args.summarize_local or args.execute):
        print(json.dumps(plan(), ensure_ascii=False, indent=2))
        return 0
    if args.baseline_dataset_version is None or args.baseline_manifest_sha256 is None:
        raise SystemExit(
            "--baseline-dataset-version and --baseline-manifest-sha256 are required"
        )
    if args.permutations < 1 or args.bootstrap_resamples < 1:
        raise SystemExit("comparison resample counts must be positive")
    env_file = args.env_file if args.env_file.is_absolute() else ROOT / args.env_file
    kaggle.load_dotenv(env_file)
    owner = kaggle.validate_slug(os.getenv("KAGGLE_USERNAME", OWNER).strip(), "KAGGLE_USERNAME")
    if owner != OWNER:
        raise LossConfirmationWorkflowError(f"Frozen BGE owner must remain {OWNER}")
    authority = candidate_runner.load_local_baseline_authority(
        owner=owner,
        dataset_version=args.baseline_dataset_version,
        source_dir=args.baseline_source_dir,
        stage_dir=args.baseline_stage_dir,
        expected_manifest_sha256=args.baseline_manifest_sha256,
    )
    selected = load_selected_authority(
        authority=authority,
        lr_receipt_path=args.lr_receipt,
        epoch_receipt_path=args.epoch_receipt,
    )
    report_root = args.report_root.expanduser().resolve()
    reports_root = (ROOT / "reports").resolve()
    if report_root != reports_root and reports_root not in report_root.parents:
        raise LossConfirmationWorkflowError("Loss reports must remain below reports/")

    if args.execute:
        baseline_launcher._enforce_live_environment()
        if not os.getenv("KAGGLE_API_TOKEN", "").strip():
            raise SystemExit("Set KAGGLE_API_TOKEN in .env")
        cli = kaggle.kaggle_command()
        poll_interval = kaggle.env_int("KAGGLE_POLL_INTERVAL_SECONDS", 30, minimum=5)
        wait_timeout = kaggle.env_int("KAGGLE_WAIT_TIMEOUT_SECONDS", 45_000, minimum=60)
        run_timeout = kaggle.env_int(
            "KAGGLE_RUN_TIMEOUT_SECONDS", MAX_RUN_TIMEOUT_SECONDS, minimum=60
        )
        if run_timeout > MAX_RUN_TIMEOUT_SECONDS:
            raise LossConfirmationWorkflowError("BGE loss kernel timeout exceeds 9 hours")
    else:
        cli = []
        poll_interval = wait_timeout = run_timeout = 0

    with baseline_launcher.exclusive_campaign_lock():
        if args.stage == "screen":
            entry = build_entries(
                authority,
                selected,
                keys=["screen_sqrt_s42"],
                parents={"screen_sqrt_s42": selected["parent"]},
                write=True,
            )[0]
            budget = kernel_budget(
                authority=authority,
                selected=selected,
                planned_new_slugs=[entry["kernel_slug"]],
                attempted_loss_slugs_before_stage=[],
                reserve_worst_case=True,
            )
            if args.dry_run:
                _stage_entry(entry, env_file)
                print(json.dumps({
                    "mode": "dry_run",
                    "stage": "screen",
                    "staged": [entry["kernel_slug"]],
                    "kernel_budget": budget,
                    "kaggle_contacted": False,
                }, ensure_ascii=False, indent=2))
                return 0
            directory = (
                execute_entry(
                    cli=cli,
                    env_file=env_file,
                    authority=authority,
                    entry=entry,
                    poll_interval=poll_interval,
                    wait_timeout=wait_timeout,
                    run_timeout=run_timeout,
                    full_download=args.full_download,
                    expected_prior_loss_slugs=[],
                )
                if args.execute
                else _local_output(entry)
            )
            if directory is None:
                raise LossConfirmationWorkflowError("Local seed42 loss output is missing")
            result = summarize_screen(
                authority=authority,
                selected=selected,
                entry=entry,
                directory=directory,
                output_dir=report_root / "screen",
                sync_sheets=args.execute,
                permutations=args.permutations,
                bootstrap_resamples=args.bootstrap_resamples,
                budget=budget,
            )
        else:
            screen = load_screen_receipt(
                args.screen_receipt,
                authority=authority,
                selected=selected,
            )
            bce_entry = build_entries(
                authority,
                selected,
                keys=["confirm_bce_s17"],
                parents={"confirm_bce_s17": selected["parent"]},
                write=True,
            )[0]
            planned = [bce_entry["kernel_slug"]]
            prospective_budget = kernel_budget(
                authority=authority,
                selected=selected,
                planned_new_slugs=[screen["entry"]["kernel_slug"], *planned],
                attempted_loss_slugs_before_stage=[screen["entry"]["kernel_slug"]],
                reserve_worst_case=screen["accepted"],
            )
            if args.dry_run:
                _stage_entry(bce_entry, env_file)
                print(json.dumps({
                    "mode": "dry_run",
                    "stage": "confirm",
                    "staged": [bce_entry["kernel_slug"]],
                    "next_after_valid_output": (
                        "conditional sqrt seed17" if screen["accepted"] else "final receipt"
                    ),
                    "kernel_budget": prospective_budget,
                    "kaggle_contacted": False,
                }, ensure_ascii=False, indent=2))
                return 0
            bce_dir = (
                execute_entry(
                    cli=cli,
                    env_file=env_file,
                    authority=authority,
                    entry=bce_entry,
                    poll_interval=poll_interval,
                    wait_timeout=wait_timeout,
                    run_timeout=run_timeout,
                    full_download=args.full_download,
                    expected_prior_loss_slugs=[screen["entry"]["kernel_slug"]],
                )
                if args.execute
                else _local_output(bce_entry)
            )
            if bce_dir is None:
                raise LossConfirmationWorkflowError("Local seed17 BCE output is missing")
            sqrt_entry: dict[str, Any] | None = None
            sqrt_dir: Path | None = None
            if screen["accepted"]:
                bce_completion = _read_json(bce_dir / baseline_uploader.COMPLETION_FILENAME)
                bce_parent = _parent_from_output(entry=bce_entry, completion=bce_completion)
                sqrt_entry = build_entries(
                    authority,
                    selected,
                    keys=["confirm_sqrt_s17"],
                    parents={"confirm_sqrt_s17": bce_parent},
                    write=True,
                )[0]
                planned.append(sqrt_entry["kernel_slug"])
                # Recheck the now-concrete three-slug union before the
                # conditional challenger can be staged or pushed.
                kernel_budget(
                    authority=authority,
                    selected=selected,
                    planned_new_slugs=[screen["entry"]["kernel_slug"], *planned],
                    attempted_loss_slugs_before_stage=[
                        screen["entry"]["kernel_slug"],
                        bce_entry["kernel_slug"],
                    ],
                    reserve_worst_case=False,
                )
                sqrt_dir = (
                    execute_entry(
                        cli=cli,
                        env_file=env_file,
                        authority=authority,
                        entry=sqrt_entry,
                        poll_interval=poll_interval,
                        wait_timeout=wait_timeout,
                        run_timeout=run_timeout,
                        full_download=args.full_download,
                        expected_prior_loss_slugs=[
                            screen["entry"]["kernel_slug"],
                            bce_entry["kernel_slug"],
                        ],
                    )
                    if args.execute
                    else _local_output(sqrt_entry)
                )
                if sqrt_dir is None:
                    raise LossConfirmationWorkflowError("Local seed17 challenger output is missing")
            budget = kernel_budget(
                authority=authority,
                selected=selected,
                planned_new_slugs=[screen["entry"]["kernel_slug"], *planned],
                attempted_loss_slugs_before_stage=[
                    screen["entry"]["kernel_slug"],
                    *planned,
                ],
                reserve_worst_case=False,
            )
            result = summarize_confirmation(
                authority=authority,
                selected=selected,
                screen=screen,
                bce_entry=bce_entry,
                bce_dir=bce_dir,
                sqrt_entry=sqrt_entry,
                sqrt_dir=sqrt_dir,
                output_dir=report_root / "confirm",
                sync_sheets=args.execute,
                permutations=args.permutations,
                bootstrap_resamples=args.bootstrap_resamples,
                budget=budget,
            )
    print(json.dumps({
        "mode": "execute" if args.execute else "summarize_local",
        "stage": args.stage,
        "receipt": str(result["receipt_path"]),
        "selection": (
            result["receipt"].get("selected_loss_variant")
            or result["receipt"].get("seed42_winner")
        ),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
