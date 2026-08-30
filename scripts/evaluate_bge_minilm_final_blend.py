#!/usr/bin/env python3
"""Evaluate the two frozen BGE/MiniLM final logit blends locally.

The tool has no Kaggle or Google Sheets integration.  It reads a completed BGE
slim output and the byte-exact selected MiniLM e3 predictions, verifies their
IID/hard row binding, and evaluates only the two weights frozen in
``configs/bge_minilm_final_blend_v1.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "bge_minilm_final_blend_v1.json"
DEFAULT_MINILM_DIR = (
    ROOT
    / "artifacts"
    / "kaggle"
    / "pm-minilm5-sft-e3-lr8e5-v1"
    / "minilm_5ep_team_data_loss_ablation"
)
SPLITS = ("iid", "hard")
REQUIRED_COLUMNS = ("id1", "id2", "target", "category_1", "score")
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
RUN_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
LOSS_KERNEL_SLUG_PATTERN = re.compile(
    r"pm-b2-(?:lbce|lsqrt)-[0-9a-f]{12}-s17-l1"
)
WORKFLOW = "bge_2ep_sft_loss_confirmation_v1"
CAMPAIGN = "bge_2ep_sft_oodtrain_v1"
PLAIN_BCE = "bce_finite_guard_v1"
SQRT_BALANCED_BCE = "balanced_category_class_sqrt_bce"
LOSS_HOOK_SHA256 = {
    PLAIN_BCE: "37a7173f708420128b8ca24aa0be073698dcad3e0e2bbc3a39c6d958be9c7e30",
    SQRT_BALANCED_BCE: (
        "0ee68461176acaccf1473de2e5119765b1de9020a1327b675d9106d00df1ca5c"
    ),
}
SEED17_EXPERIMENT = {
    PLAIN_BCE: "bge2_sft_loss_bce_s17_v1",
    SQRT_BALANCED_BCE: "bge2_sft_loss_sqrt_s17_v1",
}
FINAL_RECEIPT_KEYS = {
    "schema_version",
    "status",
    "workflow",
    "campaign",
    "stage",
    "loss_screen_receipt_sha256",
    "branch",
    "execution_order",
    "seed42",
    "seed17",
    "final_gate",
    "selected_loss_variant",
    "selected_recipe",
    "selected_recipe_sha256",
    "selected_loss_hook_sha256",
    "primary_split",
    "diagnostic_splits",
    "hard_used_for_selection",
    "ood",
    "seed17_comparison_sheets_synced",
    "seed17_comparison_sync_marker",
    "kernel_budget",
}
SCREEN_RECEIPT_KEYS = {
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

EXPECTED_CONTRACT: dict[str, Any] = {
    "schema_version": 1,
    "name": "bge_minilm_final_blend_v1",
    "bge_authority": {
        "final_receipt_filename": "loss_confirmation_receipt.json",
        "screen_receipt_filename": "loss_screen_receipt.json",
        "lr_receipt_filename": "lr_selection_receipt.json",
        "epoch_receipt_filename": "epoch_selection_receipt.json",
        "selected_seed": 17,
        "prediction_hash_source": (
            "selected_loss_variant_seed17_prediction_binding"
        ),
    },
    "frozen_minilm": {
        "experiment": "minilm5_sft_e3_lr8e5_v1",
        "run_id": "2facdf6ad2b842a8a2f3e4c66410675e",
        "directory_hint": (
            "artifacts/kaggle/pm-minilm5-sft-e3-lr8e5-v1/"
            "minilm_5ep_team_data_loss_ablation"
        ),
        "predictions": {
            "iid": {
                "filename": "iid_validation_predictions.parquet",
                "rows": 12_000,
                "bytes": 5_203_122,
                "sha256": (
                    "2fd79620059ad4425afe3998912b08f8cd6f9db06d54ca2fd042a9b5f7215e00"
                ),
            },
            "hard": {
                "filename": "hard_validation_predictions.parquet",
                "rows": 5_814,
                "bytes": 2_394_519,
                "sha256": (
                    "da696070b57c310ba5378861780d624b62517c23fafd391deb84153e52700a28"
                ),
            },
        },
    },
    "splits": {
        "primary": "iid",
        "diagnostic": ["hard"],
        "expected_category_count": 18,
    },
    "blend": {
        "space": "logit",
        "bge_weights": [0.6, 0.7],
        "score_domain": "strict_open_unit_interval_no_clipping",
    },
    "selection": {
        "metric": "macro_average_precision",
        "primary_split": "iid",
        "hard_used_for_selection": False,
        "tie_abs_tolerance": 1e-12,
        "blend_tie_break_order": ["logit_bge_0p7", "logit_bge_0p6"],
        "final_tie_break_order": [
            "minilm_only",
            "bge_only",
            "logit_bge_0p7",
            "logit_bge_0p6",
        ],
    },
    "ood": {
        "evaluated": False,
        "macro_average_precision": -1.0,
        "comparison": None,
        "prediction_file": None,
        "used_for_selection": False,
        "reason": "former OOD categories were included in BGE supervised training",
    },
}


class FinalBlendError(ValueError):
    """Raised when the frozen final-blend contract cannot be satisfied."""


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FinalBlendError(f"Could not read {label}: {error}") from error
    if not isinstance(value, dict):
        raise FinalBlendError(f"{label} must be a JSON object")
    return value


def load_frozen_contract(path: Path = CONTRACT_PATH) -> tuple[dict[str, Any], str]:
    resolved = path.expanduser().resolve(strict=True)
    contract = _read_json_object(resolved, label="blend contract")
    if contract != EXPECTED_CONTRACT:
        raise FinalBlendError("Final blend contract differs from the frozen v1 contract")
    return contract, sha256_file(resolved)


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or HASH_PATTERN.fullmatch(value) is None:
        raise FinalBlendError(f"{label} must be an exact lowercase SHA-256")
    return value


def _require_run_id(value: object, label: str) -> str:
    if not isinstance(value, str) or RUN_ID_PATTERN.fullmatch(value) is None:
        raise FinalBlendError(f"{label} must be a 32-hex run_id")
    return value


def _finite_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise FinalBlendError(f"{label} must be finite numeric")
    return float(value)


def _load_named_receipt(path: Path, *, filename: str, label: str) -> tuple[dict[str, Any], Path]:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise FinalBlendError(f"Could not resolve {label}: {error}") from error
    if resolved.name != filename:
        raise FinalBlendError(f"{label} must be the authoritative {filename}")
    _regular_file(resolved, label)
    return _read_json_object(resolved, label=label), resolved


def _validate_public_parent(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalBlendError(f"{label} parent is missing")
    parent = dict(value)
    required = {
        "run_id",
        "experiment",
        "campaign_identity_sha256",
        "source_sha256",
        "recipe_sha256",
        "checkpoint_manifest_sha256",
        "checkpoint_model_sha256",
        "validation_manifest_sha256",
        "loss_hook_sha256",
        "config",
    }
    if set(parent) != required:
        raise FinalBlendError(f"{label} parent fields differ")
    _require_run_id(parent["run_id"], f"{label} parent run_id")
    if not isinstance(parent["experiment"], str) or not parent["experiment"]:
        raise FinalBlendError(f"{label} parent experiment is missing")
    for key in required - {"run_id", "experiment", "config"}:
        _require_hash(parent[key], f"{label} parent {key}")
    if not isinstance(parent["config"], Mapping) or not parent["config"]:
        raise FinalBlendError(f"{label} parent config is missing")
    return parent


def _validate_prediction_binding(value: object, *, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != set(SPLITS):
        raise FinalBlendError(f"{label} prediction binding must contain exact IID/hard")
    result: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        item = value[split]
        if not isinstance(item, Mapping) or set(item) != {"path", "bytes", "sha256"}:
            raise FinalBlendError(f"{label} {split} prediction binding fields differ")
        raw_path = item["path"]
        if not isinstance(raw_path, str) or not raw_path or not Path(raw_path).is_absolute():
            raise FinalBlendError(f"{label} {split} prediction path must be absolute")
        size = item["bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise FinalBlendError(f"{label} {split} prediction byte size is invalid")
        result[split] = {
            "path": raw_path,
            "bytes": size,
            "sha256": _require_hash(item["sha256"], f"{label} {split} prediction SHA"),
        }
    return result


def _validate_upstream_lr_epoch(
    *,
    lr: Mapping[str, Any],
    lr_path: Path,
    epoch: Mapping[str, Any],
    epoch_path: Path,
) -> dict[str, Any]:
    lr_fixed = {
        "schema_version": 1,
        "status": "complete",
        "campaign": CAMPAIGN,
        "stage": "lr_log_line",
        "primary_split": "iid",
        "diagnostic_splits": ["hard"],
        "comparison_sheets_synced": True,
        "ood": {"comparison": None, "macro_average_precision": -1.0},
    }
    for key, expected in lr_fixed.items():
        if lr.get(key) != expected:
            raise FinalBlendError(f"LR selection receipt differs at {key}")
    lr_parent = _validate_public_parent(lr.get("selected_parent"), label="LR selected")
    if lr.get("selected_source") not in {"baseline", "candidate"}:
        raise FinalBlendError("LR selected source is invalid")
    baseline_dataset = lr.get("frozen_baseline_dataset")
    if not isinstance(baseline_dataset, Mapping):
        raise FinalBlendError("LR receipt has no frozen baseline binding")

    epoch_fixed = {
        "schema_version": 1,
        "status": "complete",
        "campaign": CAMPAIGN,
        "stage": "epoch_line",
        "comparison_sheets_synced": True,
        "epoch_3": "deferred",
        "frozen_baseline_dataset": baseline_dataset,
        "lr_selection_receipt_sha256": sha256_file(lr_path),
        "parent": lr_parent,
    }
    for key, expected in epoch_fixed.items():
        if epoch.get(key) != expected:
            raise FinalBlendError(f"Epoch selection receipt differs at {key}")
    selection = epoch.get("selection")
    if not isinstance(selection, Mapping) or set(selection) != {
        "selected_experiment",
        "selected_run_id",
        "selected_epoch",
        "rule",
    }:
        raise FinalBlendError("Epoch selection declaration differs")
    if selection.get("rule") != "select e2 iff paired IID delta > 0.002":
        raise FinalBlendError("Epoch selection rule differs")
    selected_epoch = selection.get("selected_epoch")
    if selected_epoch == 1:
        expected_run = lr_parent["run_id"]
        expected_experiment = lr_parent["experiment"]
    elif selected_epoch == 2:
        expected_run = epoch.get("candidate_run_id")
        expected_experiment = epoch.get("candidate_experiment")
    else:
        raise FinalBlendError("Epoch selection must choose epoch 1 or 2")
    if (
        selection.get("selected_run_id") != expected_run
        or selection.get("selected_experiment") != expected_experiment
    ):
        raise FinalBlendError("Epoch selected identity differs from its branch")
    _require_run_id(selection["selected_run_id"], "epoch selected run_id")
    return {
        "selected_run_id": selection["selected_run_id"],
        "selected_experiment": selection["selected_experiment"],
        "baseline_dataset": dict(baseline_dataset),
        "lr_receipt_sha256": sha256_file(lr_path),
        "epoch_receipt_sha256": sha256_file(epoch_path),
    }


def load_final_bge_authority(
    *,
    final_receipt_path: Path,
    screen_receipt_path: Path,
    lr_receipt_path: Path,
    epoch_receipt_path: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the local receipt chain and derive the selected seed-17 output."""
    authority = contract["bge_authority"]
    final, final_path = _load_named_receipt(
        final_receipt_path,
        filename=authority["final_receipt_filename"],
        label="final loss-confirmation receipt",
    )
    screen, screen_path = _load_named_receipt(
        screen_receipt_path,
        filename=authority["screen_receipt_filename"],
        label="loss-screen receipt",
    )
    lr, lr_path = _load_named_receipt(
        lr_receipt_path,
        filename=authority["lr_receipt_filename"],
        label="LR selection receipt",
    )
    epoch, epoch_path = _load_named_receipt(
        epoch_receipt_path,
        filename=authority["epoch_receipt_filename"],
        label="epoch selection receipt",
    )
    if set(final) != FINAL_RECEIPT_KEYS:
        raise FinalBlendError("Final loss-confirmation receipt fields differ")
    if set(screen) != SCREEN_RECEIPT_KEYS:
        raise FinalBlendError("Loss-screen receipt fields differ")
    final_fixed = {
        "schema_version": 1,
        "status": "complete",
        "workflow": WORKFLOW,
        "campaign": CAMPAIGN,
        "stage": "seed_confirmation",
        "primary_split": "iid",
        "diagnostic_splits": ["hard"],
        "hard_used_for_selection": False,
        "ood": {"macro_average_precision": -1.0, "comparison": None},
        "loss_screen_receipt_sha256": sha256_file(screen_path),
    }
    for key, expected in final_fixed.items():
        if final.get(key) != expected:
            raise FinalBlendError(f"Final loss-confirmation receipt differs at {key}")
    screen_fixed = {
        "schema_version": 1,
        "status": "complete",
        "workflow": WORKFLOW,
        "campaign": CAMPAIGN,
        "stage": "loss_screen",
        "primary_split": "iid",
        "diagnostic_splits": ["hard"],
        "hard_used_for_selection": False,
        "ood": {"macro_average_precision": -1.0, "comparison": None},
        "acceptance_threshold": 0.002,
        "threshold_relation": "strictly_greater_than",
        "comparison_sheets_synced": True,
        "lr_selection_receipt_sha256": sha256_file(lr_path),
        "epoch_selection_receipt_sha256": sha256_file(epoch_path),
    }
    for key, expected in screen_fixed.items():
        if screen.get(key) != expected:
            raise FinalBlendError(f"Loss-screen receipt differs at {key}")
    upstream = _validate_upstream_lr_epoch(
        lr=lr,
        lr_path=lr_path,
        epoch=epoch,
        epoch_path=epoch_path,
    )
    if upstream["epoch_receipt_sha256"] != screen["epoch_selection_receipt_sha256"]:
        raise FinalBlendError("Loss-screen epoch receipt SHA differs")
    if screen.get("frozen_baseline_dataset") != upstream["baseline_dataset"]:
        raise FinalBlendError("Loss-screen frozen baseline binding differs")

    anchor = screen.get("anchor")
    if not isinstance(anchor, Mapping) or set(anchor) != {
        "parent",
        "directory",
        "prediction_binding",
        "loss_variant",
        "reused_existing_run",
    }:
        raise FinalBlendError("Loss-screen anchor declaration differs")
    anchor_parent = _validate_public_parent(anchor["parent"], label="loss-screen anchor")
    if (
        anchor_parent["run_id"] != upstream["selected_run_id"]
        or anchor_parent["experiment"] != upstream["selected_experiment"]
    ):
        raise FinalBlendError("Loss-screen anchor differs from epoch selection")
    if anchor.get("loss_variant") != PLAIN_BCE or anchor.get("reused_existing_run") is not True:
        raise FinalBlendError("Loss-screen anchor is not the reused plain BCE parent")
    _validate_prediction_binding(anchor.get("prediction_binding"), label="loss-screen anchor")
    challenger = screen.get("challenger")
    if not isinstance(challenger, Mapping):
        raise FinalBlendError("Loss-screen challenger declaration is missing")
    challenger_run_id = _require_run_id(
        challenger.get("run_id"), "loss-screen challenger run_id"
    )
    if challenger.get("loss_variant") != SQRT_BALANCED_BCE:
        raise FinalBlendError("Loss-screen challenger loss differs")
    _validate_prediction_binding(
        challenger.get("prediction_binding"), label="loss-screen challenger"
    )
    seed42_delta = _finite_number(screen.get("iid_delta"), "loss-screen IID delta")
    passed42 = seed42_delta > 0.002
    if (
        screen.get("challenger_accepted_for_seed17") is not passed42
        or screen.get("seed42_winner")
        != (SQRT_BALANCED_BCE if passed42 else PLAIN_BCE)
    ):
        raise FinalBlendError("Loss-screen branch differs from its IID delta")

    seed42 = final.get("seed42")
    if not isinstance(seed42, Mapping) or dict(seed42) != {
        "anchor_run_id": anchor_parent["run_id"],
        "challenger_run_id": challenger_run_id,
        "iid_delta": seed42_delta,
        "screen_threshold": 0.002,
        "challenger_passed": passed42,
    }:
        raise FinalBlendError("Final receipt seed42 binding differs from loss screen")
    seed17 = final.get("seed17")
    expected_seed17_keys = {
        "bce_run_id",
        "bce_experiment",
        "bce_kernel_slug",
        "bce_identity_sha256",
        "bce_recipe_sha256",
        "bce_loss_hook_sha256",
        "bce_parent_run_id",
        "bce_prediction_binding",
        "challenger_run_id",
        "challenger_kernel_slug",
        "challenger_identity_sha256",
        "challenger_recipe_sha256",
        "challenger_loss_hook_sha256",
        "challenger_parent_run_id",
        "challenger_prediction_binding",
        "iid_delta",
        "comparison_artifacts",
    }
    if not isinstance(seed17, Mapping) or set(seed17) != expected_seed17_keys:
        raise FinalBlendError("Final receipt seed17 fields differ")
    bce_run_id = _require_run_id(seed17["bce_run_id"], "seed17 BCE run_id")
    if seed17.get("bce_experiment") != SEED17_EXPERIMENT[PLAIN_BCE]:
        raise FinalBlendError("Seed17 BCE experiment differs")
    bce_slug = seed17.get("bce_kernel_slug")
    if (
        not isinstance(bce_slug, str)
        or LOSS_KERNEL_SLUG_PATTERN.fullmatch(bce_slug) is None
        or not bce_slug.startswith("pm-b2-lbce-")
    ):
        raise FinalBlendError("Seed17 BCE kernel slug differs")
    for key in ("bce_identity_sha256", "bce_recipe_sha256", "bce_loss_hook_sha256"):
        _require_hash(seed17.get(key), f"seed17 {key}")
    if seed17["bce_loss_hook_sha256"] != LOSS_HOOK_SHA256[PLAIN_BCE]:
        raise FinalBlendError("Seed17 BCE loss hook differs")
    if seed17.get("bce_parent_run_id") != anchor_parent["run_id"]:
        raise FinalBlendError("Seed17 BCE parent differs from selected recipe")
    bce_predictions = _validate_prediction_binding(
        seed17.get("bce_prediction_binding"), label="seed17 BCE"
    )

    seed17_delta: float | None
    challenger_predictions: dict[str, dict[str, Any]] | None
    if passed42:
        challenger17_run_id = _require_run_id(
            seed17.get("challenger_run_id"), "seed17 challenger run_id"
        )
        challenger_slug = seed17.get("challenger_kernel_slug")
        if (
            not isinstance(challenger_slug, str)
            or LOSS_KERNEL_SLUG_PATTERN.fullmatch(challenger_slug) is None
            or not challenger_slug.startswith("pm-b2-lsqrt-")
        ):
            raise FinalBlendError("Seed17 challenger kernel slug differs")
        for key in (
            "challenger_identity_sha256",
            "challenger_recipe_sha256",
            "challenger_loss_hook_sha256",
        ):
            _require_hash(seed17.get(key), f"seed17 {key}")
        if seed17["challenger_loss_hook_sha256"] != LOSS_HOOK_SHA256[SQRT_BALANCED_BCE]:
            raise FinalBlendError("Seed17 challenger loss hook differs")
        if seed17.get("challenger_parent_run_id") != bce_run_id:
            raise FinalBlendError("Seed17 challenger parent differs from matched BCE")
        if challenger17_run_id == bce_run_id or challenger_slug == bce_slug:
            raise FinalBlendError("Seed17 BCE and challenger identities must be distinct")
        challenger_predictions = _validate_prediction_binding(
            seed17.get("challenger_prediction_binding"), label="seed17 challenger"
        )
        seed17_delta = _finite_number(seed17.get("iid_delta"), "seed17 IID delta")
        if final.get("branch") != "matched_bce_and_challenger_seed17":
            raise FinalBlendError("Final receipt seed17 branch differs")
        if final.get("execution_order") != [bce_slug, challenger_slug]:
            raise FinalBlendError("Final receipt seed17 execution order differs")
        if not isinstance(seed17.get("comparison_artifacts"), Mapping):
            raise FinalBlendError("Seed17 comparison artifacts are missing")
        if final.get("seed17_comparison_sheets_synced") is not True or not isinstance(
            final.get("seed17_comparison_sync_marker"), Mapping
        ):
            raise FinalBlendError("Seed17 comparison is not Sheets-synced")
    else:
        challenger17_run_id = None
        challenger_slug = None
        challenger_predictions = None
        null_keys = (
            "challenger_run_id",
            "challenger_kernel_slug",
            "challenger_identity_sha256",
            "challenger_recipe_sha256",
            "challenger_loss_hook_sha256",
            "challenger_parent_run_id",
            "challenger_prediction_binding",
            "iid_delta",
            "comparison_artifacts",
        )
        if any(seed17.get(key) is not None for key in null_keys):
            raise FinalBlendError("Rejected seed42 branch contains seed17 challenger data")
        seed17_delta = None
        if final.get("branch") != "matched_bce_seed17_only":
            raise FinalBlendError("Final receipt BCE-only branch differs")
        if final.get("execution_order") != [bce_slug]:
            raise FinalBlendError("Final receipt BCE-only execution order differs")
        if (
            final.get("seed17_comparison_sheets_synced") is not None
            or final.get("seed17_comparison_sync_marker") is not None
        ):
            raise FinalBlendError("BCE-only branch claims a seed17 comparison sync")

    mean_delta = (
        (seed42_delta + seed17_delta) / 2 if seed17_delta is not None else None
    )
    challenger_final = bool(
        passed42
        and seed17_delta is not None
        and seed42_delta > 0
        and seed17_delta > 0
        and mean_delta is not None
        and mean_delta >= 0.002
    )
    expected_gate = {
        "seed42_delta_strictly_positive": seed42_delta > 0,
        "seed17_delta_strictly_positive": (
            seed17_delta > 0 if seed17_delta is not None else None
        ),
        "mean_iid_delta": mean_delta,
        "required_mean_iid_delta": 0.002,
        "challenger_accepted": challenger_final,
    }
    if final.get("final_gate") != expected_gate:
        raise FinalBlendError("Final loss gate was not recomputed from both seeds")
    selected_loss = SQRT_BALANCED_BCE if challenger_final else PLAIN_BCE
    if final.get("selected_loss_variant") != selected_loss:
        raise FinalBlendError("Final selected loss differs from the two-seed gate")
    if final.get("selected_loss_hook_sha256") != LOSS_HOOK_SHA256[selected_loss]:
        raise FinalBlendError("Final selected loss hook differs")
    if (
        final.get("selected_recipe") != anchor_parent["config"]
        or final.get("selected_recipe_sha256") != anchor_parent["recipe_sha256"]
    ):
        raise FinalBlendError("Final selected recipe differs from the LR/e2 parent")
    if not isinstance(final.get("kernel_budget"), Mapping):
        raise FinalBlendError("Final receipt has no kernel-budget evidence")

    if selected_loss == PLAIN_BCE:
        selected_run_id = bce_run_id
        selected_experiment = SEED17_EXPERIMENT[PLAIN_BCE]
        selected_slug = bce_slug
        selected_identity = seed17["bce_identity_sha256"]
        selected_recipe = seed17["bce_recipe_sha256"]
        selected_predictions = bce_predictions
        selected_parent_run_id = anchor_parent["run_id"]
    else:
        selected_run_id = challenger17_run_id
        selected_experiment = SEED17_EXPERIMENT[SQRT_BALANCED_BCE]
        selected_slug = challenger_slug
        selected_identity = seed17["challenger_identity_sha256"]
        selected_recipe = seed17["challenger_recipe_sha256"]
        selected_predictions = challenger_predictions
        selected_parent_run_id = bce_run_id
    if selected_run_id is None or selected_slug is None or selected_predictions is None:
        raise FinalBlendError("Final receipt did not resolve one selected seed17 output")
    return {
        "selected_loss_variant": selected_loss,
        "selected_loss_hook_sha256": LOSS_HOOK_SHA256[selected_loss],
        "selected_run_id": selected_run_id,
        "selected_experiment": selected_experiment,
        "selected_kernel_slug": selected_slug,
        "selected_identity_sha256": selected_identity,
        "selected_recipe_sha256": selected_recipe,
        "selected_parent_run_id": selected_parent_run_id,
        "prediction_binding": selected_predictions,
        "receipt_chain": {
            "final": {"filename": final_path.name, "sha256": sha256_file(final_path)},
            "screen": {"filename": screen_path.name, "sha256": sha256_file(screen_path)},
            "lr": {"filename": lr_path.name, "sha256": sha256_file(lr_path)},
            "epoch": {"filename": epoch_path.name, "sha256": sha256_file(epoch_path)},
        },
    }


def _regular_file(path: Path, label: str) -> None:
    try:
        mode = os.lstat(path).st_mode
    except OSError as error:
        raise FinalBlendError(f"Could not inspect {label}: {error}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise FinalBlendError(f"{label} must be a non-symlink regular file")


def _resolve_input_dir(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise FinalBlendError(f"{label} directory must not be a symlink")
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise FinalBlendError(f"Could not resolve {label} directory: {error}") from error
    if not resolved.is_dir():
        raise FinalBlendError(f"{label} input is not a directory: {resolved}")
    return resolved


def _find_exactly_one(root: Path, filename: str, *, label: str) -> Path:
    matches = sorted(root.rglob(filename))
    if len(matches) != 1:
        raise FinalBlendError(
            f"Expected exactly one {filename} below {label}, found {len(matches)}"
        )
    _regular_file(matches[0], f"{label} {filename}")
    return matches[0]


def _prediction_paths(root: Path, *, label: str) -> dict[str, Path]:
    return {
        split: _find_exactly_one(
            root,
            f"{split}_validation_predictions.parquet",
            label=label,
        )
        for split in SPLITS
    }


def _validate_prediction_frame(
    path: Path,
    *,
    split: str,
    label: str,
    expected_rows: int,
    expected_category_count: int,
) -> pd.DataFrame:
    try:
        import pyarrow.parquet as parquet

        available = set(parquet.ParquetFile(path).schema.names)
    except Exception as error:
        raise FinalBlendError(f"Could not inspect {label} {split} parquet: {error}") from error
    missing = set(REQUIRED_COLUMNS) - available
    if missing:
        raise FinalBlendError(
            f"{label} {split} predictions miss columns {sorted(missing)}"
        )
    try:
        frame = pd.read_parquet(path, columns=list(REQUIRED_COLUMNS))
    except Exception as error:
        raise FinalBlendError(f"Could not read {label} {split} predictions: {error}") from error
    if len(frame) != expected_rows:
        raise FinalBlendError(
            f"{label} {split} row count differs: {len(frame)} != {expected_rows}"
        )
    if frame[list(REQUIRED_COLUMNS)].isnull().any().any():
        raise FinalBlendError(f"{label} {split} predictions contain null required values")

    try:
        target = pd.to_numeric(frame["target"], errors="raise").to_numpy(dtype=np.float64)
        score = pd.to_numeric(frame["score"], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise FinalBlendError(f"{label} {split} target/score is not numeric") from error
    if not np.isfinite(target).all() or not set(np.unique(target)) <= {0.0, 1.0}:
        raise FinalBlendError(f"{label} {split} targets must be finite binary 0/1")
    if not np.isfinite(score).all() or np.any(score <= 0.0) or np.any(score >= 1.0):
        raise FinalBlendError(
            f"{label} {split} scores must be finite and strictly inside (0, 1); "
            "the frozen evaluator does not clip"
        )
    categories = frame["category_1"].to_numpy()
    if any(not isinstance(value, str) or not value for value in categories):
        raise FinalBlendError(f"{label} {split} categories must be non-empty strings")
    unique_categories = np.unique(categories)
    if len(unique_categories) != expected_category_count:
        raise FinalBlendError(
            f"{label} {split} category count differs: "
            f"{len(unique_categories)} != {expected_category_count}"
        )
    for category in unique_categories:
        category_target = target[categories == category]
        if set(np.unique(category_target)) != {0.0, 1.0}:
            raise FinalBlendError(
                f"{label} {split} category {category!r} must contain both classes"
            )

    left = frame["id1"].astype(str).to_numpy()
    right = frame["id2"].astype(str).to_numpy()
    if np.any(left == right):
        raise FinalBlendError(f"{label} {split} predictions contain a self-pair")
    unordered = pd.Series(
        np.where(left <= right, left + "\x1f" + right, right + "\x1f" + left)
    )
    if unordered.duplicated().any():
        raise FinalBlendError(f"{label} {split} predictions contain duplicate pairs")

    result = frame.copy()
    result["target"] = target
    result["score"] = score
    return result


def _strict_bind(bge: pd.DataFrame, minilm: pd.DataFrame, *, split: str) -> None:
    for column in ("id1", "id2", "target", "category_1"):
        if not np.array_equal(bge[column].to_numpy(), minilm[column].to_numpy()):
            left = bge[column].to_numpy()
            right = minilm[column].to_numpy()
            mismatch = np.flatnonzero(left != right)
            index = int(mismatch[0]) if len(mismatch) else -1
            raise FinalBlendError(
                f"BGE/MiniLM {split} strict row binding differs at {column}, row {index}"
            )


def _row_binding_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for id1, id2, target, category in frame[
        ["id1", "id2", "target", "category_1"]
    ].itertuples(index=False, name=None):
        row = [str(id1), str(id2), int(target), category]
        digest.update(canonical_json(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def logit_blend(
    bge_score: np.ndarray,
    minilm_score: np.ndarray,
    *,
    bge_weight: float,
) -> np.ndarray:
    """Return a stable sigmoid of the weighted logits, without clipping."""
    bge = np.asarray(bge_score, dtype=np.float64)
    minilm = np.asarray(minilm_score, dtype=np.float64)
    if bge.shape != minilm.shape:
        raise FinalBlendError("BGE and MiniLM score shapes differ")
    if (
        not np.isfinite(bge).all()
        or not np.isfinite(minilm).all()
        or np.any(bge <= 0)
        or np.any(bge >= 1)
        or np.any(minilm <= 0)
        or np.any(minilm >= 1)
    ):
        raise FinalBlendError("Logit blend requires finite scores strictly inside (0, 1)")
    if bge_weight not in (0.6, 0.7):
        raise FinalBlendError("Only frozen BGE weights 0.6 and 0.7 are allowed")
    bge_logit = np.log(bge) - np.log1p(-bge)
    minilm_logit = np.log(minilm) - np.log1p(-minilm)
    value = bge_weight * bge_logit + (1.0 - bge_weight) * minilm_logit
    result = np.empty_like(value)
    positive = value >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    if not np.isfinite(result).all() or np.any(result <= 0) or np.any(result >= 1):
        raise FinalBlendError("Logit blend produced a non-finite or boundary score")
    return result


def _score_metrics(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    target = frame["target"].to_numpy(dtype=np.float64)
    categories = frame["category_1"].to_numpy(dtype=str)
    per_category: dict[str, Any] = {}
    aps: list[float] = []
    for category in sorted(np.unique(categories)):
        selected = categories == category
        category_target = target[selected]
        ap = float(average_precision_score(category_target, scores[selected]))
        aps.append(ap)
        per_category[category] = {
            "examples": int(selected.sum()),
            "positive_examples": int(category_target.sum()),
            "average_precision": ap,
        }
    return {
        "macro_average_precision": float(np.mean(aps)),
        "overall_average_precision": float(average_precision_score(target, scores)),
        "per_category": per_category,
    }


def _metric_delta(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "macro_average_precision": float(
            candidate["macro_average_precision"] - baseline["macro_average_precision"]
        ),
        "overall_average_precision": float(
            candidate["overall_average_precision"] - baseline["overall_average_precision"]
        ),
        "per_category_average_precision": {},
    }
    for category, metrics in candidate["per_category"].items():
        result["per_category_average_precision"][category] = float(
            metrics["average_precision"]
            - baseline["per_category"][category]["average_precision"]
        )
    return result


def _select_candidate(
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    candidates: Sequence[str],
    tie_order: Sequence[str],
    tolerance: float,
) -> dict[str, Any]:
    if set(candidates) != set(tie_order) or len(candidates) != len(tie_order):
        raise FinalBlendError("Selection candidates and tie order differ")
    values = {
        name: float(metrics[name]["macro_average_precision"]) for name in candidates
    }
    maximum = max(values.values())
    tied = [name for name in tie_order if maximum - values[name] <= tolerance]
    selected = tied[0]
    return {
        "selected": selected,
        "selected_iid_macro_average_precision": values[selected],
        "maximum_iid_macro_average_precision": maximum,
        "tie_candidates": tied,
        "tie_applied": len(tied) > 1,
    }


def _validate_bge_completion(
    root: Path,
    frames: Mapping[str, pd.DataFrame],
    metrics: Mapping[str, Mapping[str, Any]],
    authority: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    completion_path = _find_exactly_one(root, "notebook_completed.json", label="BGE")
    completion = _read_json_object(completion_path, label="BGE completion")
    if completion.get("status") != "complete":
        raise FinalBlendError("BGE completion status is not complete")
    run_id = completion.get("run_id")
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise FinalBlendError("BGE completion run_id is not a 32-hex identifier")
    if not isinstance(completion.get("experiment"), str) or not completion["experiment"]:
        raise FinalBlendError("BGE completion experiment is missing")
    exact_authority = {
        "run_id": authority["selected_run_id"],
        "experiment": authority["selected_experiment"],
        "campaign_identity_sha256": authority["selected_identity_sha256"],
        "frozen_recipe_sha256": authority["selected_recipe_sha256"],
        "loss_hook_sha256": authority["selected_loss_hook_sha256"],
        "loss_variant": authority["selected_loss_variant"],
    }
    for key, expected in exact_authority.items():
        if completion.get(key) != expected:
            raise FinalBlendError(f"BGE completion differs from final receipt at {key}")
    loss_confirmation = completion.get("loss_confirmation")
    if not isinstance(loss_confirmation, Mapping):
        raise FinalBlendError("BGE completion has no loss-confirmation binding")
    loss_fixed = {
        "workflow": WORKFLOW,
        "stage": "seed_confirmation",
        "seed": 17,
        "loss_variant": authority["selected_loss_variant"],
        "primary_split": "iid",
        "diagnostic_splits": ["hard"],
        "ood_macro_average_precision": -1.0,
        "ood_comparison": None,
        "fresh_start": True,
        "checkpoint_resume": False,
    }
    for key, expected in loss_fixed.items():
        if loss_confirmation.get(key) != expected:
            raise FinalBlendError(f"BGE completion loss confirmation differs at {key}")
    loss_parent = loss_confirmation.get("parent")
    if (
        not isinstance(loss_parent, Mapping)
        or loss_parent.get("run_id") != authority["selected_parent_run_id"]
    ):
        raise FinalBlendError("BGE completion loss-confirmation parent differs")
    report = completion.get("training_report")
    if not isinstance(report, Mapping):
        raise FinalBlendError("BGE completion has no training_report")
    if report.get("evaluated_validation_splits") != ["iid", "hard"]:
        raise FinalBlendError("BGE completion evaluated split declaration differs")
    validation = report.get("validation_splits")
    if not isinstance(validation, Mapping) or set(validation) != {"iid", "hard", "ood"}:
        raise FinalBlendError("BGE completion validation split keys differ")
    ood = validation.get("ood")
    if not isinstance(ood, Mapping) or ood.get("evaluated") is not False:
        raise FinalBlendError("BGE completion does not disable OOD evaluation")
    for key in ("macro_average_precision", "overall_average_precision"):
        if ood.get(key) != -1.0:
            raise FinalBlendError(f"BGE completion OOD {key} is not -1")
    if ood.get("predictions_file") is not None:
        raise FinalBlendError("BGE completion claims an OOD prediction file")
    for split in SPLITS:
        saved = validation.get(split)
        if not isinstance(saved, Mapping):
            raise FinalBlendError(f"BGE completion lacks {split} metrics")
        if saved.get("examples") != len(frames[split]):
            raise FinalBlendError(f"BGE completion {split} row count differs")
        for key in ("macro_average_precision", "overall_average_precision"):
            value = saved.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isclose(
                    float(value),
                    float(metrics[split][key]),
                    rel_tol=0,
                    abs_tol=1e-12,
                )
            ):
                raise FinalBlendError(f"BGE completion {split} {key} differs from parquet")
    return completion, completion_path


def _validate_optional_minilm_completion(root: Path, contract: Mapping[str, Any]) -> None:
    matches = sorted(root.rglob("notebook_completed.json"))
    if not matches:
        return
    if len(matches) != 1:
        raise FinalBlendError("MiniLM input contains multiple completion receipts")
    _regular_file(matches[0], "MiniLM completion")
    completion = _read_json_object(matches[0], label="MiniLM completion")
    frozen = contract["frozen_minilm"]
    if completion.get("status") != "complete":
        raise FinalBlendError("MiniLM completion is not complete")
    if completion.get("run_id") != frozen["run_id"]:
        raise FinalBlendError("MiniLM completion run_id differs from frozen e3")
    if completion.get("experiment") != frozen["experiment"]:
        raise FinalBlendError("MiniLM completion experiment differs from frozen e3")


def evaluate_final_blend(
    *,
    bge_dir: Path,
    minilm_dir: Path,
    bge_authority: Mapping[str, Any],
    contract: Mapping[str, Any],
    contract_sha256: str,
    enforce_frozen_contract: bool = True,
) -> dict[str, Any]:
    """Validate both inputs and return a deterministic, OOD-disabled report."""
    if enforce_frozen_contract and contract != EXPECTED_CONTRACT:
        raise FinalBlendError("Evaluation contract is not the frozen v1 contract")
    if contract["bge_authority"] != EXPECTED_CONTRACT["bge_authority"]:
        raise FinalBlendError("BGE final-receipt authority differs from the frozen contract")
    if contract["blend"] != EXPECTED_CONTRACT["blend"]:
        raise FinalBlendError("Blend method or weights differ from the frozen contract")
    if contract["selection"] != EXPECTED_CONTRACT["selection"]:
        raise FinalBlendError("Blend selection policy differs from the frozen contract")
    if contract["ood"] != EXPECTED_CONTRACT["ood"]:
        raise FinalBlendError("OOD policy differs from the frozen contract")
    required_authority = {
        "selected_loss_variant",
        "selected_loss_hook_sha256",
        "selected_run_id",
        "selected_experiment",
        "selected_kernel_slug",
        "selected_identity_sha256",
        "selected_recipe_sha256",
        "selected_parent_run_id",
        "prediction_binding",
        "receipt_chain",
    }
    if set(bge_authority) != required_authority:
        raise FinalBlendError("Final BGE authority fields differ")
    selected_bindings = _validate_prediction_binding(
        bge_authority["prediction_binding"], label="final selected BGE"
    )

    bge_root = _resolve_input_dir(bge_dir, "BGE")
    minilm_root = _resolve_input_dir(minilm_dir, "MiniLM")
    if bge_root == minilm_root or bge_root in minilm_root.parents or minilm_root in bge_root.parents:
        raise FinalBlendError("BGE and MiniLM input directories must be disjoint")
    if bge_root.name != bge_authority["selected_kernel_slug"]:
        raise FinalBlendError("BGE directory name differs from final selected kernel slug")
    if list(bge_root.rglob("ood*validation_predictions.parquet")):
        raise FinalBlendError("BGE final slim output contains forbidden OOD predictions")

    bge_paths = _prediction_paths(bge_root, label="BGE")
    minilm_paths = _prediction_paths(minilm_root, label="MiniLM")
    frozen_minilm = contract["frozen_minilm"]
    expected_categories = int(contract["splits"]["expected_category_count"])
    frames: dict[str, dict[str, pd.DataFrame]] = {}
    file_bindings: dict[str, dict[str, Any]] = {"bge": {}, "minilm": {}}
    for split in SPLITS:
        receipt_path = Path(selected_bindings[split]["path"]).expanduser().resolve(
            strict=True
        )
        if receipt_path != bge_paths[split]:
            raise FinalBlendError(
                f"BGE {split} path differs from final receipt prediction binding"
            )
        bge_hash = sha256_file(bge_paths[split])
        if bge_hash != selected_bindings[split]["sha256"]:
            raise FinalBlendError(f"BGE {split} prediction SHA-256 differs from final receipt")
        if bge_paths[split].stat().st_size != selected_bindings[split]["bytes"]:
            raise FinalBlendError(f"BGE {split} prediction bytes differ from final receipt")
        mini_binding = frozen_minilm["predictions"][split]
        mini_hash = sha256_file(minilm_paths[split])
        if mini_hash != mini_binding["sha256"]:
            raise FinalBlendError(f"MiniLM {split} prediction SHA-256 differs from frozen e3")
        if minilm_paths[split].stat().st_size != mini_binding["bytes"]:
            raise FinalBlendError(f"MiniLM {split} prediction byte size differs from frozen e3")
        rows = int(mini_binding["rows"])
        bge_frame = _validate_prediction_frame(
            bge_paths[split],
            split=split,
            label="BGE",
            expected_rows=rows,
            expected_category_count=expected_categories,
        )
        minilm_frame = _validate_prediction_frame(
            minilm_paths[split],
            split=split,
            label="MiniLM",
            expected_rows=rows,
            expected_category_count=expected_categories,
        )
        _strict_bind(bge_frame, minilm_frame, split=split)
        frames[split] = {"bge": bge_frame, "minilm": minilm_frame}
        file_bindings["bge"][split] = {
            "filename": bge_paths[split].name,
            "rows": rows,
            "bytes": bge_paths[split].stat().st_size,
            "sha256": bge_hash,
        }
        file_bindings["minilm"][split] = dict(mini_binding)

    split_reports: dict[str, Any] = {}
    bge_metrics_for_completion: dict[str, Any] = {}
    candidate_names = ["bge_only", "minilm_only", "logit_bge_0p6", "logit_bge_0p7"]
    for split in SPLITS:
        bge_frame = frames[split]["bge"]
        mini_frame = frames[split]["minilm"]
        bge_score = bge_frame["score"].to_numpy(dtype=np.float64)
        mini_score = mini_frame["score"].to_numpy(dtype=np.float64)
        model_metrics: dict[str, Any] = {
            "bge_only": _score_metrics(bge_frame, bge_score),
            "minilm_only": _score_metrics(bge_frame, mini_score),
        }
        for weight in contract["blend"]["bge_weights"]:
            name = f"logit_bge_{str(weight).replace('.', 'p')}"
            model_metrics[name] = _score_metrics(
                bge_frame,
                logit_blend(bge_score, mini_score, bge_weight=float(weight)),
            )
        if set(model_metrics) != set(candidate_names):
            raise FinalBlendError("Evaluator produced a candidate outside the frozen set")
        deltas = {
            name: {
                "vs_bge": _metric_delta(model_metrics[name], model_metrics["bge_only"]),
                "vs_minilm": _metric_delta(
                    model_metrics[name], model_metrics["minilm_only"]
                ),
            }
            for name in ("logit_bge_0p6", "logit_bge_0p7")
        }
        split_reports[split] = {
            "role": "primary" if split == "iid" else "diagnostic_only",
            "examples": len(bge_frame),
            "positive_examples": int(bge_frame["target"].sum()),
            "category_count": expected_categories,
            "row_binding_sha256": _row_binding_sha256(bge_frame),
            "models": model_metrics,
            "blend_deltas": deltas,
        }
        bge_metrics_for_completion[split] = model_metrics["bge_only"]

    completion, completion_path = _validate_bge_completion(
        bge_root,
        {split: frames[split]["bge"] for split in SPLITS},
        bge_metrics_for_completion,
        bge_authority,
    )
    _validate_optional_minilm_completion(minilm_root, contract)

    policy = contract["selection"]
    iid_metrics = split_reports["iid"]["models"]
    blend_selection = _select_candidate(
        iid_metrics,
        candidates=("logit_bge_0p6", "logit_bge_0p7"),
        tie_order=policy["blend_tie_break_order"],
        tolerance=float(policy["tie_abs_tolerance"]),
    )
    final_selection = _select_candidate(
        iid_metrics,
        candidates=tuple(candidate_names),
        tie_order=policy["final_tie_break_order"],
        tolerance=float(policy["tie_abs_tolerance"]),
    )
    selected = final_selection["selected"]
    report = {
        "schema_version": 1,
        "status": "complete",
        "name": contract["name"],
        "contract_sha256": _require_hash(contract_sha256, "contract SHA-256"),
        "inputs": {
            "bge": {
                "run_id": completion["run_id"],
                "experiment": completion["experiment"],
                "selected_loss_variant": bge_authority["selected_loss_variant"],
                "selected_kernel_slug": bge_authority["selected_kernel_slug"],
                "completion_sha256": sha256_file(completion_path),
                "predictions": file_bindings["bge"],
                "authority_receipts": bge_authority["receipt_chain"],
            },
            "minilm": {
                "run_id": frozen_minilm["run_id"],
                "experiment": frozen_minilm["experiment"],
                "predictions": file_bindings["minilm"],
            },
        },
        "method": {
            "blend_space": "logit",
            "evaluated_bge_weights": [0.6, 0.7],
            "score_clipping": False,
        },
        "splits": split_reports,
        "selection": {
            "primary_split": "iid",
            "metric": "macro_average_precision",
            "hard_used_for_selection": False,
            "tie_abs_tolerance": policy["tie_abs_tolerance"],
            "blend": blend_selection,
            "final": {
                **final_selection,
                "recommend_blend": selected.startswith("logit_bge_"),
            },
        },
        "ood": dict(contract["ood"]),
        "claims": {
            "ood_claim": False,
            "hidden_test_gain_claim": False,
            "runtime_viability_claim": False,
        },
    }
    # Fail before returning if any NaN/Infinity slipped into a serialized metric.
    canonical_json(report)
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    selected = report["selection"]["final"]["selected"]
    lines = [
        "# Final BGE + MiniLM blend evaluation",
        "",
        f"Final IID recommendation: `{selected}`.",
        "",
        "| split | candidate | macro AP | overall AP | delta macro vs BGE | delta macro vs MiniLM |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for split in SPLITS:
        section = report["splits"][split]
        for name, metrics in section["models"].items():
            if name.startswith("logit_bge_"):
                delta_bge = section["blend_deltas"][name]["vs_bge"][
                    "macro_average_precision"
                ]
                delta_minilm = section["blend_deltas"][name]["vs_minilm"][
                    "macro_average_precision"
                ]
                delta_bge_text = f"{delta_bge:.12f}"
                delta_minilm_text = f"{delta_minilm:.12f}"
            else:
                delta_bge_text = "—"
                delta_minilm_text = "—"
            lines.append(
                f"| {split} | {name} | {metrics['macro_average_precision']:.12f} | "
                f"{metrics['overall_average_precision']:.12f} | {delta_bge_text} | "
                f"{delta_minilm_text} |"
            )
    lines.extend(
        [
            "",
            "Selection uses IID macro AP only; hard is diagnostic and cannot change the recommendation.",
            "OOD is not evaluated: its metric sentinel is exactly `-1`, and no OOD claim is made.",
            "No hidden-test gain or runtime viability is inferred by this report.",
            "",
        ]
    )
    return "\n".join(lines)


def _prepare_output(path: Path, *, inputs: Sequence[Path]) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    for source in inputs:
        source_resolved = source.expanduser().resolve(strict=True)
        if resolved == source_resolved or source_resolved in resolved.parents:
            raise FinalBlendError("Report output must not be inside an input directory")
    if resolved.exists() or resolved.is_symlink():
        raise FinalBlendError(f"Refusing to overwrite existing report: {resolved}")
    return resolved


def _write_new_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(value)
    except FileExistsError as error:
        raise FinalBlendError(f"Refusing to overwrite existing report: {path}") from error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen 0.6/0.7 BGE+MiniLM logit blends locally"
    )
    parser.add_argument("--bge-dir", type=Path, required=True)
    parser.add_argument("--final-receipt", type=Path, required=True)
    parser.add_argument("--screen-receipt", type=Path, required=True)
    parser.add_argument("--lr-receipt", type=Path, required=True)
    parser.add_argument("--epoch-receipt", type=Path, required=True)
    parser.add_argument("--minilm-dir", type=Path, default=DEFAULT_MINILM_DIR)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--report-markdown", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    contract, contract_sha = load_frozen_contract()
    bge_authority = load_final_bge_authority(
        final_receipt_path=args.final_receipt,
        screen_receipt_path=args.screen_receipt,
        lr_receipt_path=args.lr_receipt,
        epoch_receipt_path=args.epoch_receipt,
        contract=contract,
    )
    report = evaluate_final_blend(
        bge_dir=args.bge_dir,
        minilm_dir=args.minilm_dir,
        bge_authority=bge_authority,
        contract=contract,
        contract_sha256=contract_sha,
    )
    input_dirs = (
        _resolve_input_dir(args.bge_dir, "BGE"),
        _resolve_input_dir(args.minilm_dir, "MiniLM"),
    )
    output_paths: dict[str, Path] = {}
    if args.report_json is not None:
        output_paths["json"] = _prepare_output(args.report_json, inputs=input_dirs)
    if args.report_markdown is not None:
        output_paths["markdown"] = _prepare_output(
            args.report_markdown, inputs=input_dirs
        )
    if len(set(output_paths.values())) != len(output_paths):
        raise FinalBlendError("JSON and Markdown report paths must be different")
    paths = list(output_paths.values())
    if any(
        left in right.parents or right in left.parents
        for index, left in enumerate(paths)
        for right in paths[index + 1 :]
    ):
        raise FinalBlendError("JSON and Markdown report paths must not contain each other")

    written: dict[str, str] = {}
    if "json" in output_paths:
        path = output_paths["json"]
        _write_new_text(
            path,
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
        )
        written["json"] = str(path)
    if "markdown" in output_paths:
        path = output_paths["markdown"]
        _write_new_text(path, render_markdown(report))
        written["markdown"] = str(path)
    if written:
        print(
            json.dumps(
                {
                    "status": "complete",
                    "selection": report["selection"]["final"],
                    "report_sha256": canonical_sha256(report),
                    "written": written,
                    "kaggle_contacted": False,
                    "sheets_contacted": False,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
