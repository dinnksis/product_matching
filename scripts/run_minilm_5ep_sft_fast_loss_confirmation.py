#!/usr/bin/env python3
"""Run the user-shortened two-seed MiniLM loss confirmation safely.

The frozen v1 search remains unchanged.  Once its three loss-stage locks are
complete, this wrapper asks the existing schema-v2 materializer for the full
confirmation authority in an isolated directory, then exposes only the two
prespecified seed-17 variants needed for a matched tuned-BCE/loss comparison.
The seed-42 screening runs are reused.

This is directional replication across two training seeds, not a formal
training-seed significance analysis.  The direct candidate-vs-BCE comparison
within each seed still uses the frozen component-paired permutation/bootstrap
implementation on the exact same IID examples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import create_minilm_5ep_sft_hparam_notebooks as builder
import materialize_minilm_5ep_sft_loss_confirmation as adaptive
import minilm_5ep_sft_loss_fast_track_support as fast_track
import run_minilm_5ep_sft_hparam_kaggle as launcher
import summarize_minilm_5ep_sft_hparams as summarizer


DEFAULT_PROTOCOL = ROOT / "configs" / "minilm_5ep_sft_fast_loss_confirm_v1.json"
DEFAULT_PLAN = ROOT / "configs" / "minilm_5ep_sft_hparam_search_v1.json"
DEFAULT_CAMPAIGN_DIR = ROOT / "reports" / "minilm_5ep_sft_hparam_search_v1"
DEFAULT_SUMMARY = DEFAULT_CAMPAIGN_DIR / "summary.json"
DEFAULT_BASELINE_SUMMARY = DEFAULT_CAMPAIGN_DIR / "stages" / "lr_log_line" / "summary.json"
DEFAULT_LOCKS_DIR = DEFAULT_CAMPAIGN_DIR / "stage_locks"
DEFAULT_ARTIFACTS_DIR = ROOT / "artifacts" / "kaggle"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "minilm_5ep_sft_fast_loss_confirm_v1"
DEFAULT_SOURCE_LOCK_NAME = "source_confirmation.lock.json"
DEFAULT_MANIFEST_NAME = "fast_confirmation_manifest.json"
DEFAULT_CONFIRMATION_LAUNCHER = (
    ROOT / "scripts" / "run_minilm_5ep_sft_fast_loss_confirmation_kaggle.py"
)

PROTOCOL_KIND = "minilm_5ep_sft_fast_loss_confirmation_manifest"
SUMMARY_KIND = "minilm_5ep_sft_fast_loss_confirmation_summary"
REQUIRED_PREREQUISITE_LOCKS = (
    "special_loss_screen__primary.lock.json",
    "special_loss_screen__overlay.lock.json",
    "special_loss_screen__lr_refine.lock.json",
)


class FastConfirmationError(RuntimeError):
    """Raised when the shortened confirmation cannot proceed without guessing."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FastConfirmationError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def canonical_json_dumps(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise FastConfirmationError(f"Value is not finite canonical JSON: {error}") from error


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_dumps(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                FastConfirmationError(f"Non-finite JSON constant {token!r} in {label}")
            ),
        )
    except OSError as error:
        raise FastConfirmationError(f"Cannot read {label} {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise FastConfirmationError(f"Invalid JSON in {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise FastConfirmationError(f"{label} must be a JSON object")
    return value


def _resolved(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_protocol(path: Path, *, plan: Mapping[str, Any]) -> dict[str, Any]:
    protocol = load_json_object(path, label="fast-confirmation protocol")
    expected_top = {
        "schema_version",
        "protocol_id",
        "source_campaign",
        "source_plan_canonical_sha256",
        "purpose",
        "fast_track_authority",
        "screen_seed",
        "replication_seed",
        "replication_seed_rationale",
        "recipes",
        "execution",
        "selection",
        "statistics",
        "checkpoint_handling",
        "ods_runtime_check",
    }
    if set(protocol) != expected_top:
        raise FastConfirmationError("Fast protocol top-level schema differs")
    if (
        protocol.get("schema_version") != 1
        or protocol.get("protocol_id") != "minilm_5ep_sft_fast_loss_confirm_v1"
        or protocol.get("source_campaign") != "minilm_5ep_sft_hparam_search_v1"
        or protocol.get("purpose")
        != "Directional two-seed replication of the selected special-loss recipe against the tuned BCE recipe after the user shortened confirmation."
        or protocol.get("screen_seed") != 42
        or protocol.get("replication_seed") != 17
        or protocol.get("replication_seed_rationale")
        != "Seed 17 is the first non-screen seed prespecified by the frozen v1 confirmation plan; it is not selected after observing loss results."
        or protocol.get("ods_runtime_check") is not False
    ):
        raise FastConfirmationError("Fast protocol identity/seeds/no-ODS contract differs")
    if canonical_sha256(plan) != protocol.get("source_plan_canonical_sha256"):
        raise FastConfirmationError("Frozen v1 plan differs from the fast protocol pin")
    authority = protocol.get("fast_track_authority")
    recipes = protocol.get("recipes")
    execution = protocol.get("execution")
    selection = protocol.get("selection")
    statistics = protocol.get("statistics")
    checkpoint = protocol.get("checkpoint_handling")
    if not all(
        isinstance(value, Mapping)
        for value in (authority, recipes, execution, selection, statistics, checkpoint)
    ):
        raise FastConfirmationError("Fast protocol sections must be objects")
    if dict(authority) != {
        "policy_path": "configs/minilm_5ep_sft_loss_fast_track_v1.json",
        "policy_canonical_sha256": "e32559035aae79f85a99a035f0f6a23e7206fda0f5d785839e2f0396000ffef4",
        "receipt_path": "reports/minilm_5ep_sft_hparam_search_v1/fast_track/max_grad_norm_skip.receipt.json",
        "receipt_file_sha256": "d39c931c130c7addd5635c89cab96c4fdd3824fa40fc7e24926b1dc58738580f",
        "receipt_payload_sha256": "4458e389a111c4aa09dd7fa211f289f7c7dab38910b43cac7fd98c84b0f61986",
        "freeze_manifest_path": "configs/minilm_5ep_sft_loss_fast_track_v1.freeze.json",
        "freeze_manifest_file_sha256": "eaf8d7479fd3bbf96e194e8e9f669ad70a3227fafd61031cda22ce10c75fe35b",
        "freeze_manifest_payload_sha256": "28bd0ef53e24438ee37f3d447fbf8a9ebc18a84407dbe78e2275ad1f081851ef",
        "legacy_receipt_freeze": {
            "archive_path": "configs/minilm_5ep_sft_loss_fast_track_receipt_freeze_db7165.json",
            "file_sha256": "db7165cb27abe41c2846b2bcc5b928b81750e87960d31b5915c72574a0181fbe",
            "manifest_payload_sha256": "2e4ad36f81c2d5f294454955f1e5cad6b2ae86b22b4b3ae6595d05f82f1e3717",
        },
        "require_validated_policy_receipt_and_freeze_manifest": True,
        "launcher_wrapper": "scripts/run_minilm_5ep_sft_fast_loss_confirmation_kaggle.py",
    }:
        raise FastConfirmationError("Fast-track authority contract differs")
    if dict(recipes) != {
        "comparator_role": "selected_regularized_bce_recipe",
        "candidate_role": "loss_finalist_1",
        "candidate_rule": "best_non_bce_after_optional_lr_refinement",
    }:
        raise FastConfirmationError("Fast protocol recipe roles differ")
    if dict(execution) != {
        "new_kernels": 2,
        "sequential": True,
        "background_fanout": False,
        "force_resubmit": False,
        "retry_failed": False,
        "google_sheets_group": "sft",
        "google_sheets_tab": "sft_exps",
    }:
        raise FastConfirmationError("Fast protocol execution contract differs")
    if dict(selection) != {
        "primary_metric": "iid_macro_ap",
        "comparison": "candidate_loss_recipe_minus_tuned_bce_recipe_at_matched_seed",
        "rationale": "Adopt the selected loss only when its direction replicates on the prespecified second seed and its two-seed mean clears the frozen practical margin.",
        "require_each_seed_delta_strictly_positive": True,
        "minimum_mean_iid_delta": 0.002,
        "failure_or_tie_selects": "selected_regularized_bce_recipe",
        "accepted_selects": "loss_finalist_1",
        "selected_recipe_reference_seed": 42,
        "hard_is_diagnostic_only": True,
        "ood_is_diagnostic_only": True,
    }:
        raise FastConfirmationError("Fast protocol selection contract differs")
    if dict(statistics) != {
        "within_seed_iid_method": "paired_component_permutation",
        "within_seed_ci_method": "paired_component_bootstrap_percentile",
        "training_seed_interpretation": "directional_replication_only",
        "formal_training_seed_significance_claim": False,
    }:
        raise FastConfirmationError("Fast protocol statistical contract differs")
    if dict(checkpoint) != {
        "download_supported": False,
        "download_performed": False,
        "reason": "The frozen selected-checkpoint downloader targets the original three-seed/runtime-gated confirmation and is intentionally not reused for this no-ODS fast confirmation.",
    }:
        raise FastConfirmationError("Fast protocol checkpoint/no-download contract differs")
    return protocol


def _role_group(lock: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    matches = [
        group
        for group in lock["resolved_stage"]["recipe_groups"]
        if role in group.get("roles", [])
    ]
    if len(matches) != 1:
        raise FastConfirmationError(f"Source lock must contain exactly one group for {role!r}")
    return matches[0]


def select_fast_pair(
    *,
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Select the exact reused seed-42 and new seed-17 BCE/loss recipe pairs."""
    if lock.get("schema_version") != 2 or lock.get("mode") != "confirmation":
        raise FastConfirmationError("Source lock is not a schema-v2 confirmation lock")
    decision = lock.get("decision", {})
    if decision.get("seed42_reuse") is not True or decision.get("seeds") != [17, 42, 2026]:
        raise FastConfirmationError("Source lock does not preserve the frozen seed set")
    origin_by_id = {
        str(origin["origin_id"]): origin for origin in lock.get("origins", [])
    }
    contract = builder.normalized_campaign_execution_contract(plan, lock)
    normalized_by_group_seed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for entry in contract["variants"]:
        group_id = str(entry["variant"].get("recipe_group_id", ""))
        seed = int(entry["expected_config"]["seed"])
        key = (group_id, seed)
        if key in normalized_by_group_seed:
            raise FastConfirmationError(f"Duplicate confirmation variant for {key}")
        normalized_by_group_seed[key] = entry

    result: dict[str, Any] = {}
    for side, role in (
        ("comparator", str(protocol["recipes"]["comparator_role"])),
        ("candidate", str(protocol["recipes"]["candidate_role"])),
    ):
        group = _role_group(lock, role)
        origin = origin_by_id.get(str(group["origin_seed42_id"]))
        if origin is None:
            raise FastConfirmationError(f"Group {group['recipe_group_id']!r} has no seed-42 origin")
        if int(origin["resolved_config"]["seed"]) != int(protocol["screen_seed"]):
            raise FastConfirmationError(f"{role} origin is not seed 42")
        entry = normalized_by_group_seed.get(
            (str(group["recipe_group_id"]), int(protocol["replication_seed"]))
        )
        if entry is None:
            raise FastConfirmationError(f"{role} has no prespecified seed-17 variant")
        entry = deepcopy(dict(entry))
        # The normalized builder contract owns recipe/provenance identity.  The
        # launcher validator additionally needs the frozen historical baseline
        # metrics to recheck each standalone completion artifact.
        entry["baseline_metrics"] = deepcopy(dict(plan["baseline_metrics"]))
        entry["provenance_alias"] = None
        expected_from_origin = deepcopy(dict(origin["resolved_config"]))
        expected_from_origin["seed"] = int(protocol["replication_seed"])
        if entry["expected_config"] != expected_from_origin:
            raise FastConfirmationError(f"{role} seed-17 recipe differs beyond seed")
        if entry["loss_variant"] != origin["loss_variant"] or entry[
            "loss_hook_sha256"
        ] != origin["loss_hook_sha256"]:
            raise FastConfirmationError(f"{role} loss identity differs between seeds")
        result[side] = {"role": role, "group": group, "origin": origin, "entry": entry}

    if result["comparator"]["origin"]["loss_variant"] != "bce":
        raise FastConfirmationError("Fast comparator must use BCE")
    if result["candidate"]["origin"]["loss_variant"] == "bce":
        raise FastConfirmationError("Fast candidate must be a non-BCE loss")
    if result["comparator"]["group"]["recipe_group_id"] == result["candidate"]["group"][
        "recipe_group_id"
    ]:
        raise FastConfirmationError("BCE and loss candidate collapsed into one recipe group")
    experiments = {
        str(result[side]["entry"]["experiment"]) for side in ("comparator", "candidate")
    }
    if len(experiments) != 2:
        raise FastConfirmationError("Fast confirmation must contain two distinct experiments")
    return result


def decide_loss(*, screen_delta: float, replication_delta: float, threshold: float) -> dict[str, Any]:
    values = [float(screen_delta), float(replication_delta), float(threshold)]
    if any(not math.isfinite(value) for value in values) or threshold < 0:
        raise FastConfirmationError("Confirmation deltas/threshold must be finite")
    mean_delta = (Decimal(str(screen_delta)) + Decimal(str(replication_delta))) / Decimal(2)
    accepted = (
        Decimal(str(screen_delta)) > 0
        and Decimal(str(replication_delta)) > 0
        and mean_delta >= Decimal(str(threshold))
    )
    return {
        "accepted": accepted,
        "screen_seed_iid_delta": float(screen_delta),
        "replication_seed_iid_delta": float(replication_delta),
        "mean_iid_delta": float(mean_delta),
        "minimum_mean_iid_delta": float(threshold),
        "both_seed_deltas_strictly_positive": screen_delta > 0 and replication_delta > 0,
        "selected_role": "loss_finalist_1" if accepted else "selected_regularized_bce_recipe",
    }


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _load_legacy_receipt_freeze(
    *, protocol: Mapping[str, Any], freeze: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    """Validate the archived detached authority embedded in the real receipt.

    The completed skip receipt predates the recovery-safe current freeze.  The
    current freeze explicitly preserves that exact receipt authority as a
    read-only archive.  Confirmation pins and checks both authorities instead
    of pretending that the historical receipt was signed by the current file.
    """
    authority = protocol["fast_track_authority"]
    legacy_pin = authority.get("legacy_receipt_freeze")
    semantic_contract = freeze.get("semantic_contract")
    if (
        not isinstance(legacy_pin, Mapping)
        or not isinstance(semantic_contract, Mapping)
        or semantic_contract.get("legacy_receipt_freeze") != legacy_pin
    ):
        raise FastConfirmationError(
            "Current fast-track freeze does not preserve the pinned legacy receipt freeze"
        )
    try:
        raw_archive = Path(str(legacy_pin["archive_path"]))
        archive_path = _resolved(raw_archive)
        default_archive = fast_track.LEGACY_RECEIPT_FREEZE_ARCHIVE.resolve(strict=True)
        resolved_archive = archive_path.resolve(strict=True)
    except (KeyError, OSError) as error:
        raise FastConfirmationError(
            "Pinned legacy receipt freeze archive is unavailable"
        ) from error
    if (
        archive_path.is_symlink()
        or not resolved_archive.is_file()
        or resolved_archive != default_archive
        or file_sha256(resolved_archive) != legacy_pin.get("file_sha256")
    ):
        raise FastConfirmationError("Pinned legacy receipt freeze archive differs")
    archived = load_json_object(resolved_archive, label="legacy receipt freeze archive")
    if (
        resolved_archive.read_text(encoding="utf-8")
        != canonical_json_dumps(archived) + "\n"
        or archived.get("manifest_payload_sha256")
        != legacy_pin.get("manifest_payload_sha256")
    ):
        raise FastConfirmationError("Pinned legacy receipt freeze archive differs")
    return resolved_archive, archived


def _validate_fast_track_binding(
    *,
    protocol: Mapping[str, Any],
    policy_path: Path,
    receipt_path: Path,
    freeze_manifest_path: Path,
    policy: Mapping[str, Any],
    receipt: Mapping[str, Any],
    freeze: Mapping[str, Any],
) -> None:
    authority = protocol["fast_track_authority"]
    if fast_track.policy_sha256(policy) != authority["policy_canonical_sha256"]:
        raise FastConfirmationError("Validated fast-track policy differs from protocol pin")
    policy_authority = receipt.get("policy_authority")
    source_plan = receipt.get("source_plan")
    freeze_authority = receipt.get("freeze_authority")
    if not all(
        isinstance(value, Mapping)
        for value in (policy_authority, source_plan, freeze_authority)
    ):
        raise FastConfirmationError("Validated fast-track receipt has no nested authority")
    if policy_authority.get("canonical_sha256") != authority["policy_canonical_sha256"]:
        raise FastConfirmationError("Validated fast-track receipt is bound to another policy")
    if source_plan.get("canonical_sha256") != protocol["source_plan_canonical_sha256"]:
        raise FastConfirmationError("Fast-track receipt is bound to another source plan")
    if receipt.get("external_actions") != {
        "kaggle_submit_requires_explicit_flag": True,
        "google_sheets_tab": "sft_exps",
        "require_exact_sheets_sync": True,
        "ods_submission_allowed": False,
    }:
        raise FastConfirmationError("Fast-track receipt does not preserve no-ODS/sft_exps scope")
    try:
        configured_policy = (ROOT / authority["policy_path"]).resolve(strict=True)
        configured_receipt = (ROOT / authority["receipt_path"]).resolve(strict=True)
        configured_freeze = (ROOT / authority["freeze_manifest_path"]).resolve(strict=True)
        provided_policy = policy_path.resolve(strict=True)
        provided_receipt = receipt_path.resolve(strict=True)
        provided_freeze = freeze_manifest_path.resolve(strict=True)
        default_policy = fast_track.DEFAULT_POLICY.resolve(strict=True)
        default_receipt = fast_track.DEFAULT_RECEIPT.resolve(strict=True)
        default_freeze = fast_track.DEFAULT_FREEZE_MANIFEST.resolve(strict=True)
        recorded_policy = Path(str(policy_authority["path"])).resolve(strict=True)
        recorded_freeze = Path(str(freeze_authority["path"])).resolve(strict=True)
    except OSError as error:
        raise FastConfirmationError(
            "Fast confirmation requires the exact reviewed policy/receipt/freeze authority"
        ) from error
    except KeyError as error:
        raise FastConfirmationError(
            "Fast confirmation receipt authority is incomplete"
        ) from error
    if (
        policy_path.is_symlink()
        or freeze_manifest_path.is_symlink()
        or receipt_path.is_symlink()
        or not provided_policy.is_file()
        or not provided_receipt.is_file()
        or not provided_freeze.is_file()
        or provided_policy != configured_policy
        or configured_policy != default_policy
        or provided_receipt != configured_receipt
        or configured_receipt != default_receipt
        or provided_freeze != configured_freeze
        or configured_freeze != default_freeze
        or recorded_policy != provided_policy
        # The historical receipt records the canonical freeze *path* but its
        # authority hashes are verified against the archived legacy bytes.
        or recorded_freeze != provided_freeze
    ):
        raise FastConfirmationError(
            "Fast confirmation requires the exact reviewed policy/receipt/freeze authority"
        )
    if (
        file_sha256(provided_receipt) != authority["receipt_file_sha256"]
        or receipt.get("summary_payload_sha256")
        != authority["receipt_payload_sha256"]
    ):
        raise FastConfirmationError("Completed fast-track receipt differs from protocol pin")
    if (
        freeze.get("policy_canonical_sha256") != authority["policy_canonical_sha256"]
        or freeze.get("manifest_payload_sha256")
        != authority["freeze_manifest_payload_sha256"]
        or file_sha256(provided_freeze) != authority["freeze_manifest_file_sha256"]
    ):
        raise FastConfirmationError("Current fast-track detached freeze binding differs")
    _, archived_freeze = _load_legacy_receipt_freeze(
        protocol=protocol,
        freeze=freeze,
    )
    if (
        set(freeze_authority)
        != {
            "path",
            "file_sha256",
            "manifest_payload_sha256",
            "reviewed_execution_file_sha256",
            "reviewed_review_file_sha256",
        }
        or freeze_authority.get("file_sha256")
        != authority["legacy_receipt_freeze"]["file_sha256"]
        or freeze_authority.get("manifest_payload_sha256")
        != authority["legacy_receipt_freeze"]["manifest_payload_sha256"]
        or freeze_authority.get("reviewed_execution_file_sha256")
        != archived_freeze.get("reviewed_execution_file_sha256")
        or freeze_authority.get("reviewed_review_file_sha256")
        != archived_freeze.get("reviewed_review_file_sha256")
    ):
        raise FastConfirmationError("Legacy receipt detached freeze binding differs")


def build_manifest(
    *,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    plan_path: Path,
    lock_path: Path,
    lock: Mapping[str, Any],
    selected: Mapping[str, Any],
    policy_path: Path,
    policy: Mapping[str, Any],
    receipt_path: Path,
    receipt: Mapping[str, Any],
    freeze_manifest_path: Path,
    freeze: Mapping[str, Any],
) -> dict[str, Any]:
    legacy_freeze_path, legacy_freeze = _load_legacy_receipt_freeze(
        protocol=protocol,
        freeze=freeze,
    )
    recipes = {}
    launch_order = []
    for side in ("comparator", "candidate"):
        item = selected[side]
        origin = item["origin"]
        entry = item["entry"]
        recipes[side] = {
            "role": item["role"],
            "recipe_group_id": item["group"]["recipe_group_id"],
            "recipe_family_sha256": item["group"]["recipe_family_sha256"],
            "loss_variant": origin["loss_variant"],
            "loss_hook_sha256": origin["loss_hook_sha256"],
            "screen_seed_42": {
                "experiment": origin["experiment"],
                "kernel_slug": origin["kernel_slug"],
                "run_id": origin["run_id"],
                "recipe_sha256": origin["recipe_sha256"],
                "iid_predictions_sha256": origin["iid_predictions_sha256"],
                "completion_sha256": origin["completion_sha256"],
            },
            "replication_seed_17": {
                "experiment": entry["experiment"],
                "kernel_slug": entry["kernel_slug"],
                "recipe_sha256": entry["recipe_sha256"],
                "source_sha256": entry["source_sha256"],
                "expected_config": deepcopy(entry["expected_config"]),
                "expected_notes": entry["expected_notes"],
            },
        }
        launch_order.append(str(entry["experiment"]))
    exposed_experiments = set(launch_order)
    unexposed = sorted(
        {
            str(variant["experiment"])
            for variant in lock["resolved_stage"]["variants"]
            if str(variant["experiment"]) not in exposed_experiments
        }
    )
    projected_entries = [
        {
            "role": recipes[side]["role"],
            "experiment": recipes[side]["replication_seed_17"]["experiment"],
            "kernel_slug": recipes[side]["replication_seed_17"]["kernel_slug"],
            "seed": 17,
            "recipe_sha256": recipes[side]["replication_seed_17"]["recipe_sha256"],
            "source_sha256": recipes[side]["replication_seed_17"]["source_sha256"],
            "loss_variant": recipes[side]["loss_variant"],
            "loss_hook_sha256": recipes[side]["loss_hook_sha256"],
        }
        for side in ("comparator", "candidate")
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": PROTOCOL_KIND,
        "protocol_id": protocol["protocol_id"],
        "protocol_path": _relative(protocol_path),
        "protocol_canonical_sha256": canonical_sha256(protocol),
        "wrapper_code_sha256": file_sha256(Path(__file__)),
        "confirmation_launcher_path": _relative(DEFAULT_CONFIRMATION_LAUNCHER),
        "confirmation_launcher_code_sha256": file_sha256(DEFAULT_CONFIRMATION_LAUNCHER),
        "fast_track_policy_path": _relative(policy_path),
        "fast_track_policy_sha256": fast_track.policy_sha256(policy),
        "fast_track_receipt_path": _relative(receipt_path),
        "fast_track_receipt_file_sha256": file_sha256(receipt_path),
        "fast_track_receipt_payload_sha256": receipt["summary_payload_sha256"],
        "fast_track_receipt_legacy_freeze_archive_path": _relative(
            legacy_freeze_path
        ),
        "fast_track_receipt_legacy_freeze_archive_file_sha256": file_sha256(
            legacy_freeze_path
        ),
        "fast_track_receipt_legacy_freeze_archive_payload_sha256": legacy_freeze[
            "manifest_payload_sha256"
        ],
        "fast_track_freeze_manifest_path": _relative(freeze_manifest_path),
        "fast_track_freeze_manifest_file_sha256": file_sha256(freeze_manifest_path),
        "fast_track_freeze_manifest_payload_sha256": freeze[
            "manifest_payload_sha256"
        ],
        "source_plan_path": _relative(plan_path),
        "source_plan_canonical_sha256": canonical_sha256(builder.load_plan(plan_path)),
        "source_confirmation_lock_path": _relative(lock_path),
        "source_confirmation_lock_file_sha256": file_sha256(lock_path),
        "source_confirmation_lock_payload_sha256": lock["lock_payload_sha256"],
        "screen_seed": protocol["screen_seed"],
        "replication_seed": protocol["replication_seed"],
        "recipes": recipes,
        "launch_order": launch_order,
        "new_kernel_count": 2,
        "execution_projection": {
            "kind": "exact_two_entry_seed17_projection",
            "runnable_entries": projected_entries,
            "runnable_entry_count": 2,
            "unexposed_source_lock_experiments": unexposed,
            "unexposed_source_lock_experiments_sha256": canonical_sha256(unexposed),
            "forbidden_training_seeds": [2026],
            "allows_arbitrary_experiment_passthrough": False,
            "runtime_attestation_path_exposed": False,
            "ods_submission_path_exposed": False,
            "checkpoint_download_path_exposed": False,
        },
        "execution_policy": deepcopy(protocol["execution"]),
        "selection_rule": deepcopy(protocol["selection"]),
        "statistics_contract": deepcopy(protocol["statistics"]),
        "checkpoint_handling": deepcopy(protocol["checkpoint_handling"]),
        "ods_runtime_check": False,
    }
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    return payload


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = canonical_json_dumps(payload) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        existing = path.read_text(encoding="utf-8")
        if existing != serialized:
            raise FastConfirmationError(f"Existing immutable manifest conflicts: {path}")
        return
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def prepare(
    *,
    protocol_path: Path,
    plan_path: Path,
    summary_path: Path,
    baseline_summary_path: Path,
    locks_dir: Path,
    artifacts_dir: Path,
    output_dir: Path,
    policy_path: Path,
    receipt_path: Path,
    freeze_manifest_path: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any]]:
    if (
        plan_path.is_symlink()
        or plan_path.resolve(strict=True) != DEFAULT_PLAN.resolve(strict=True)
    ):
        raise FastConfirmationError("Fast confirmation requires the default frozen plan")
    if (
        protocol_path.is_symlink()
        or protocol_path.resolve(strict=True) != DEFAULT_PROTOCOL.resolve(strict=True)
    ):
        raise FastConfirmationError("Fast confirmation requires the default frozen protocol")
    plan = builder.load_plan(plan_path)
    protocol = load_protocol(protocol_path, plan=plan)
    prerequisite_paths = [locks_dir / name for name in REQUIRED_PREREQUISITE_LOCKS]
    source_lock_path = output_dir / DEFAULT_SOURCE_LOCK_NAME
    # All adaptive materialization/replay, read_lock dispatch, campaign-lock
    # loading and contract projection happen inside the validated process-local
    # predecessor context.  The unchanged core rejects this fast-track chain
    # once the context exits.
    with fast_track.patched_loss_predecessor(
        policy_path=policy_path,
        receipt_path=receipt_path,
        freeze_manifest_path=freeze_manifest_path,
    ) as (policy, receipt, freeze):
        _validate_fast_track_binding(
            protocol=protocol,
            policy_path=policy_path,
            receipt_path=receipt_path,
            freeze_manifest_path=freeze_manifest_path,
            policy=policy,
            receipt=receipt,
            freeze=freeze,
        )
        missing = [str(path) for path in prerequisite_paths if not path.is_file()]
        if missing:
            raise FastConfirmationError(
                "Fast confirmation waits for all loss stages; missing: "
                + ", ".join(missing)
            )
        adaptive.materialize(
            mode="confirmation",
            plan_path=plan_path,
            summary_path=summary_path,
            artifacts_dir=artifacts_dir,
            prerequisite_lock_paths=prerequisite_paths,
            baseline_summary_path=baseline_summary_path,
            output_path=source_lock_path,
        )
        lock = builder.load_campaign_lock(source_lock_path, plan=plan)
        selected = select_fast_pair(plan=plan, lock=lock, protocol=protocol)
        manifest = build_manifest(
            protocol_path=protocol_path,
            protocol=protocol,
            plan_path=plan_path,
            lock_path=source_lock_path,
            lock=lock,
            selected=selected,
            policy_path=policy_path,
            policy=policy,
            receipt_path=receipt_path,
            receipt=receipt,
            freeze_manifest_path=freeze_manifest_path,
            freeze=freeze,
        )
    _write_once(output_dir / DEFAULT_MANIFEST_NAME, manifest)
    return manifest, source_lock_path, lock, selected


def _launcher_command(
    *,
    manifest_path: Path,
    plan_path: Path,
    lock_path: Path,
    env_file: Path,
    action: str,
    policy_path: Path,
    receipt_path: Path,
    freeze_manifest_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(DEFAULT_CONFIRMATION_LAUNCHER),
        "--fast-confirmation-manifest",
        str(manifest_path),
        "--fast-track-policy",
        str(policy_path),
        "--fast-track-receipt",
        str(receipt_path),
        "--fast-track-freeze-manifest",
        str(freeze_manifest_path),
        "--plan",
        str(plan_path),
        "--stage-lock",
        str(lock_path),
        "--env-file",
        str(env_file),
    ]
    if action == "dry-run":
        return [*command, "--dry-run"]
    if action == "submit":
        return [*command, "--submit", "--wait"]
    raise FastConfirmationError(f"Unknown launcher action {action!r}")


def _validate_execution_projection(
    manifest: Mapping[str, Any],
    *,
    plan_path: Path,
    lock_path: Path,
) -> list[str]:
    expected_top = {
        "schema_version",
        "kind",
        "protocol_id",
        "protocol_path",
        "protocol_canonical_sha256",
        "wrapper_code_sha256",
        "confirmation_launcher_path",
        "confirmation_launcher_code_sha256",
        "fast_track_policy_path",
        "fast_track_policy_sha256",
        "fast_track_receipt_path",
        "fast_track_receipt_file_sha256",
        "fast_track_receipt_payload_sha256",
        "fast_track_receipt_legacy_freeze_archive_path",
        "fast_track_receipt_legacy_freeze_archive_file_sha256",
        "fast_track_receipt_legacy_freeze_archive_payload_sha256",
        "fast_track_freeze_manifest_path",
        "fast_track_freeze_manifest_file_sha256",
        "fast_track_freeze_manifest_payload_sha256",
        "source_plan_path",
        "source_plan_canonical_sha256",
        "source_confirmation_lock_path",
        "source_confirmation_lock_file_sha256",
        "source_confirmation_lock_payload_sha256",
        "screen_seed",
        "replication_seed",
        "recipes",
        "launch_order",
        "new_kernel_count",
        "execution_projection",
        "execution_policy",
        "selection_rule",
        "statistics_contract",
        "checkpoint_handling",
        "ods_runtime_check",
        "manifest_payload_sha256",
    }
    plan = builder.load_plan(plan_path)
    protocol = load_protocol(DEFAULT_PROTOCOL, plan=plan)
    policy = fast_track.load_policy(fast_track.DEFAULT_POLICY)
    freeze = fast_track.load_freeze_manifest(
        fast_track.DEFAULT_FREEZE_MANIFEST, policy=policy
    )
    authority = protocol["fast_track_authority"]
    legacy_freeze_path, legacy_freeze = _load_legacy_receipt_freeze(
        protocol=protocol,
        freeze=freeze,
    )
    source_lock_document = load_json_object(lock_path, label="source confirmation lock")
    unhashed = dict(manifest)
    stored = unhashed.pop("manifest_payload_sha256", None)
    if (
        set(manifest) != expected_top
        or manifest.get("schema_version") != 1
        or manifest.get("kind") != PROTOCOL_KIND
        or stored != canonical_sha256(unhashed)
        or manifest.get("protocol_path") != _relative(DEFAULT_PROTOCOL)
        or manifest.get("protocol_canonical_sha256") != canonical_sha256(protocol)
        or manifest.get("wrapper_code_sha256") != file_sha256(Path(__file__))
        or manifest.get("confirmation_launcher_path")
        != _relative(DEFAULT_CONFIRMATION_LAUNCHER)
        or manifest.get("confirmation_launcher_code_sha256")
        != file_sha256(DEFAULT_CONFIRMATION_LAUNCHER)
        or manifest.get("fast_track_policy_path")
        != _relative(fast_track.DEFAULT_POLICY)
        or manifest.get("fast_track_policy_sha256")
        != fast_track.policy_sha256(policy)
        or manifest.get("fast_track_receipt_path")
        != authority["receipt_path"]
        or manifest.get("fast_track_receipt_file_sha256")
        != authority["receipt_file_sha256"]
        or manifest.get("fast_track_receipt_payload_sha256")
        != authority["receipt_payload_sha256"]
        or manifest.get("fast_track_receipt_legacy_freeze_archive_path")
        != _relative(legacy_freeze_path)
        or manifest.get("fast_track_receipt_legacy_freeze_archive_file_sha256")
        != file_sha256(legacy_freeze_path)
        or manifest.get("fast_track_receipt_legacy_freeze_archive_payload_sha256")
        != legacy_freeze.get("manifest_payload_sha256")
        or manifest.get("fast_track_freeze_manifest_path")
        != _relative(fast_track.DEFAULT_FREEZE_MANIFEST)
        or manifest.get("fast_track_freeze_manifest_file_sha256")
        != file_sha256(fast_track.DEFAULT_FREEZE_MANIFEST)
        or manifest.get("fast_track_freeze_manifest_payload_sha256")
        != freeze.get("manifest_payload_sha256")
        or manifest.get("source_plan_path") != _relative(plan_path)
        or manifest.get("source_plan_canonical_sha256")
        != canonical_sha256(plan)
        or manifest.get("source_confirmation_lock_path") != _relative(lock_path)
        or manifest.get("source_confirmation_lock_file_sha256") != file_sha256(lock_path)
        or manifest.get("source_confirmation_lock_payload_sha256")
        != source_lock_document.get("lock_payload_sha256")
        or manifest.get("screen_seed") != 42
        or manifest.get("replication_seed") != 17
        or manifest.get("new_kernel_count") != 2
        or manifest.get("ods_runtime_check") is not False
    ):
        raise FastConfirmationError("Fast execution manifest identity/hash/binding differs")
    if (
        manifest.get("execution_policy") != protocol["execution"]
        or manifest.get("selection_rule") != protocol["selection"]
        or manifest.get("statistics_contract") != protocol["statistics"]
    ):
        raise FastConfirmationError("Manifest protocol rules differ from the frozen protocol")
    launch_order = list(manifest.get("launch_order", []))
    if len(launch_order) != 2 or len(set(launch_order)) != 2:
        raise FastConfirmationError("Manifest launch order is not the exact matched pair")
    projection = manifest.get("execution_projection")
    expected_projection_keys = {
        "kind",
        "runnable_entries",
        "runnable_entry_count",
        "unexposed_source_lock_experiments",
        "unexposed_source_lock_experiments_sha256",
        "forbidden_training_seeds",
        "allows_arbitrary_experiment_passthrough",
        "runtime_attestation_path_exposed",
        "ods_submission_path_exposed",
        "checkpoint_download_path_exposed",
    }
    if not isinstance(projection, Mapping) or set(projection) != expected_projection_keys:
        raise FastConfirmationError("Manifest has no exact execution projection")
    projected = projection.get("runnable_entries")
    projected_entry_keys = {
        "role",
        "experiment",
        "kernel_slug",
        "seed",
        "recipe_sha256",
        "source_sha256",
        "loss_variant",
        "loss_hook_sha256",
    }
    if not isinstance(projected, list) or any(
        not isinstance(entry, Mapping) or set(entry) != projected_entry_keys
        for entry in projected
    ):
        raise FastConfirmationError("Projected runnable entries are malformed")
    projected_experiments = [str(entry["experiment"]) for entry in projected]
    unexposed = projection.get("unexposed_source_lock_experiments")
    if (
        projection.get("kind") != "exact_two_entry_seed17_projection"
        or projection.get("runnable_entry_count") != 2
        or projected_experiments != launch_order
        or [entry["role"] for entry in projected]
        != ["selected_regularized_bce_recipe", "loss_finalist_1"]
        or any(entry["seed"] != 17 for entry in projected)
        or not isinstance(unexposed, list)
        or unexposed != sorted(set(unexposed))
        or set(unexposed) & set(launch_order)
        or projection.get("unexposed_source_lock_experiments_sha256")
        != canonical_sha256(unexposed)
        or projection.get("forbidden_training_seeds") != [2026]
        or projection.get("allows_arbitrary_experiment_passthrough") is not False
        or projection.get("runtime_attestation_path_exposed") is not False
        or projection.get("ods_submission_path_exposed") is not False
        or projection.get("checkpoint_download_path_exposed") is not False
    ):
        raise FastConfirmationError("Manifest execution projection is not the exact safe pair")
    if manifest.get("checkpoint_handling") != {
        "download_supported": False,
        "download_performed": False,
        "reason": "The frozen selected-checkpoint downloader targets the original three-seed/runtime-gated confirmation and is intentionally not reused for this no-ODS fast confirmation.",
    }:
        raise FastConfirmationError("Manifest unexpectedly exposes checkpoint download")
    return launch_order


def run_selected(
    *,
    manifest: Mapping[str, Any],
    plan_path: Path,
    lock_path: Path,
    env_file: Path,
    action: str,
    policy_path: Path,
    receipt_path: Path,
    freeze_manifest_path: Path,
) -> None:
    launch_order = _validate_execution_projection(
        manifest,
        plan_path=plan_path,
        lock_path=lock_path,
    )
    policy = fast_track.load_policy(policy_path)
    freeze = fast_track.load_freeze_manifest(freeze_manifest_path, policy=policy)
    receipt = fast_track.validate_receipt(
        receipt_path,
        policy=policy,
        policy_path=policy_path,
        freeze_manifest_path=freeze_manifest_path,
    )
    protocol = load_protocol(DEFAULT_PROTOCOL, plan=builder.load_plan(plan_path))
    _validate_fast_track_binding(
        protocol=protocol,
        policy_path=policy_path,
        receipt_path=receipt_path,
        freeze_manifest_path=freeze_manifest_path,
        policy=policy,
        receipt=receipt,
        freeze=freeze,
    )
    manifest_path = lock_path.parent / DEFAULT_MANIFEST_NAME
    persisted_manifest = load_json_object(
        manifest_path, label="persisted fast-confirmation manifest"
    )
    if (
        persisted_manifest != dict(manifest)
        or manifest_path.read_text(encoding="utf-8")
        != canonical_json_dumps(persisted_manifest) + "\n"
        or manifest.get("fast_track_policy_path") != _relative(policy_path)
        or manifest.get("fast_track_policy_sha256") != fast_track.policy_sha256(policy)
        or manifest.get("fast_track_receipt_path") != _relative(receipt_path)
        or manifest.get("fast_track_receipt_file_sha256") != file_sha256(receipt_path)
        or manifest.get("fast_track_receipt_payload_sha256")
        != receipt.get("summary_payload_sha256")
        or manifest.get("fast_track_freeze_manifest_path")
        != _relative(freeze_manifest_path)
        or manifest.get("fast_track_freeze_manifest_file_sha256")
        != file_sha256(freeze_manifest_path)
        or manifest.get("fast_track_freeze_manifest_payload_sha256")
        != freeze.get("manifest_payload_sha256")
        or manifest.get("ods_runtime_check") is not False
        or receipt_path.resolve(strict=True) != fast_track.DEFAULT_RECEIPT.resolve(strict=True)
        or freeze_manifest_path.resolve(strict=True)
        != fast_track.DEFAULT_FREEZE_MANIFEST.resolve(strict=True)
    ):
        raise FastConfirmationError("Manifest fast-track/no-ODS authority binding differs")
    if launch_order != list(manifest["launch_order"]):
        raise FastConfirmationError("Validated launch order changed")
    subprocess.run(
        _launcher_command(
            manifest_path=manifest_path,
            plan_path=plan_path,
            lock_path=lock_path,
            env_file=env_file,
            action=action,
            policy_path=policy_path,
            receipt_path=receipt_path,
            freeze_manifest_path=freeze_manifest_path,
        ),
        cwd=ROOT,
        check=True,
    )


def _prediction_path(directory: Path, split: str) -> Path:
    matches = list(directory.rglob(f"{split}_validation_predictions.parquet"))
    if len(matches) != 1:
        raise FastConfirmationError(
            f"Expected one {split} prediction parquet under {directory}, found {matches}"
        )
    return matches[0]


def _completion_metrics(path: Path) -> dict[str, float]:
    completion = load_json_object(path, label="completion")
    splits = completion.get("training_report", {}).get("validation_splits", {})
    result = {}
    for split in ("iid", "hard", "ood"):
        value = splits.get(split, {}).get("macro_average_precision")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise FastConfirmationError(f"Completion has invalid {split} macro AP")
        result[split] = float(value)
    return result


def _validate_sft_exps_sync(directory: Path, *, run_id: str) -> dict[str, Any]:
    if (directory / "sheets_sync_pending.json").exists():
        raise FastConfirmationError(f"Sheets synchronization is pending under {directory}")
    sync = load_json_object(directory / "google_sheets_sync.json", label="Sheets sync")
    expected = {
        "status": "synced",
        "run_id": run_id,
        "experiment_group": "sft",
        "comparison_sheet": "sft_exps",
        "spreadsheet_id": launcher.qwen_builder.EXPERIMENT_SPREADSHEET_ID,
    }
    mismatches = {
        key: {"expected": value, "actual": sync.get(key)}
        for key, value in expected.items()
        if sync.get(key) != value
    }
    if mismatches:
        raise FastConfirmationError(
            "Run was not verifiably synchronized to sft_exps: "
            + canonical_json_dumps(mismatches)
        )
    return sync


def _validate_paired_iid_absolute_metrics(
    comparison: Mapping[str, Any],
    *,
    seed: int,
    comparator_ap: float,
    candidate_ap: float,
) -> float:
    reported_delta = float(candidate_ap) - float(comparator_ap)
    checks = (
        ("baseline", comparison.get("baseline_macro_average_precision"), comparator_ap),
        ("candidate", comparison.get("candidate_macro_average_precision"), candidate_ap),
        ("delta", comparison.get("delta_macro_average_precision"), reported_delta),
    )
    for label, compared, reported in checks:
        if (
            not isinstance(compared, (int, float))
            or not math.isfinite(float(compared))
            or not math.isclose(float(compared), float(reported), abs_tol=1e-12)
        ):
            raise FastConfirmationError(
                f"Seed {seed} paired IID {label} AP differs from report"
            )
    return reported_delta


def _require_four_unique_run_ids(
    run_identity: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> list[str]:
    run_ids = [
        str(run_identity[seed][side].get("run_id", "")).strip()
        for seed in (42, 17)
        for side in ("comparator", "candidate")
    ]
    if any(not run_id for run_id in run_ids) or len(set(run_ids)) != 4:
        raise FastConfirmationError("The two-seed matched pair must contain four unique run_ids")
    return run_ids


def _require_validation_run_id_match(
    validation: Mapping[str, Any],
    completion: Mapping[str, Any],
    *,
    side: str,
) -> str:
    validated = str(validation.get("run_id", "")).strip()
    completed = str(completion.get("run_id", "")).strip()
    if not validated or validated != completed:
        raise FastConfirmationError(
            f"{side} strict launcher validation returned another run_id"
        )
    return completed


def _require_bound_completion_run_id_match(
    *, bound_run_id: Any, completion: Mapping[str, Any], label: str
) -> str:
    bound = str(bound_run_id).strip()
    completed = str(completion.get("run_id", "")).strip()
    if not bound or bound != completed:
        raise FastConfirmationError(f"{label} completion has another run_id")
    return completed


def summarize(
    *,
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    selected: Mapping[str, Any],
    artifacts_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    validated_seed17 = {}
    metrics: dict[int, dict[str, dict[str, float]]] = {42: {}, 17: {}}
    prediction_paths: dict[int, dict[str, Path]] = {42: {}, 17: {}}
    run_identity: dict[int, dict[str, dict[str, Any]]] = {42: {}, 17: {}}
    for side in ("comparator", "candidate"):
        origin = selected[side]["origin"]
        entry = selected[side]["entry"]
        origin_completion = Path(str(origin["completion_artifact_path"]))
        origin_dir = origin_completion.parent
        origin_completion_document = load_json_object(
            origin_completion,
            label="seed-42 origin completion",
        )
        _require_bound_completion_run_id_match(
            bound_run_id=origin["run_id"],
            completion=origin_completion_document,
            label=f"seed-42 {side}",
        )
        _validate_sft_exps_sync(origin_dir, run_id=str(origin["run_id"]))
        metrics[42][side] = _completion_metrics(origin_completion)
        prediction_paths[42][side] = _prediction_path(origin_dir, "iid")
        run_identity[42][side] = {
            "experiment": origin["experiment"],
            "kernel_slug": origin["kernel_slug"],
            "run_id": origin["run_id"],
        }

        replication_dir = artifacts_dir / str(entry["kernel_slug"])
        validated_seed17[side] = launcher.validate_run_output(replication_dir, entry=entry)
        completion_path = replication_dir / "notebook_completed.json"
        completion = load_json_object(completion_path, label="replication completion")
        _require_validation_run_id_match(
            validated_seed17[side], completion, side=side
        )
        _validate_sft_exps_sync(replication_dir, run_id=str(completion["run_id"]))
        metrics[17][side] = _completion_metrics(completion_path)
        prediction_paths[17][side] = _prediction_path(replication_dir, "iid")
        run_identity[17][side] = {
            "experiment": entry["experiment"],
            "kernel_slug": entry["kernel_slug"],
            "run_id": completion["run_id"],
        }

    comparisons = {}
    raw_p_values = []
    for seed in (42, 17):
        comparison = summarizer.cached_anchor_comparison(
            anchor_path=prediction_paths[seed]["comparator"],
            candidate_path=prediction_paths[seed]["candidate"],
            anchor_predictions=summarizer.read_prediction_artifact(
                prediction_paths[seed]["comparator"]
            ),
            permutations=builder.team_builder.SIGNIFICANCE_PERMUTATIONS,
            bootstrap_resamples=builder.team_builder.SIGNIFICANCE_BOOTSTRAP_RESAMPLES,
            seed=builder.team_builder.SIGNIFICANCE_SEED,
            cache_dir=output_dir / "paired_comparisons",
        )
        _validate_paired_iid_absolute_metrics(
            comparison,
            seed=seed,
            comparator_ap=metrics[seed]["comparator"]["iid"],
            candidate_ap=metrics[seed]["candidate"]["iid"],
        )
        comparisons[str(seed)] = {
            "seed": seed,
            "comparator": run_identity[seed]["comparator"],
            "candidate": run_identity[seed]["candidate"],
            "iid": comparison,
            "hard_macro_ap_delta_diagnostic": (
                metrics[seed]["candidate"]["hard"] - metrics[seed]["comparator"]["hard"]
            ),
            "ood_macro_ap_delta_diagnostic": (
                metrics[seed]["candidate"]["ood"] - metrics[seed]["comparator"]["ood"]
            ),
        }
        raw_p_values.append(float(comparison["p_value"]))
    _require_four_unique_run_ids(run_identity)
    adjusted = summarizer.holm_adjust(raw_p_values)
    for seed, p_holm in zip((42, 17), adjusted, strict=True):
        comparisons[str(seed)]["iid"]["p_value_holm_2_seeds_diagnostic"] = p_holm

    outcome = decide_loss(
        screen_delta=float(comparisons["42"]["iid"]["delta_macro_average_precision"]),
        replication_delta=float(comparisons["17"]["iid"]["delta_macro_average_precision"]),
        threshold=float(protocol["selection"]["minimum_mean_iid_delta"]),
    )
    selected_side = "candidate" if outcome["accepted"] else "comparator"
    checkpoint = selected[selected_side]["origin"]
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": SUMMARY_KIND,
        "protocol_id": protocol["protocol_id"],
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "decision_status": "ready",
        "interpretation": "directional_two_seed_replication_not_formal_training_seed_inference",
        "new_kernels_completed": 2,
        "sft_exps_sync_verified": True,
        "comparisons": comparisons,
        "selection": outcome,
        "selected_seed42_recipe_reference": {
            "seed": 42,
            "role": outcome["selected_role"],
            "experiment": checkpoint["experiment"],
            "kernel_slug": checkpoint["kernel_slug"],
            "run_id": checkpoint["run_id"],
            "loss_variant": checkpoint["loss_variant"],
            "loss_hook_sha256": checkpoint["loss_hook_sha256"],
            "recipe_sha256": checkpoint["recipe_sha256"],
            "recipe_family_sha256": checkpoint["recipe_family_sha256"],
        },
        "hard_ood_selection_used": False,
        "formal_training_seed_significance_claim": False,
        "checkpoint_download_supported": False,
        "checkpoint_download_performed": False,
        "ods_submission_or_runtime_check_performed": False,
    }
    result["summary_payload_sha256"] = canonical_sha256(result)
    _atomic_write(output_dir / "summary.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    rows = []
    for seed in (42, 17):
        for side in ("comparator", "candidate"):
            identity = run_identity[seed][side]
            rows.append(
                {
                    "seed": seed,
                    "side": side,
                    **identity,
                    "loss_variant": selected[side]["origin"]["loss_variant"],
                    "iid_macro_ap": metrics[seed][side]["iid"],
                    "hard_macro_ap": metrics[seed][side]["hard"],
                    "ood_macro_ap": metrics[seed][side]["ood"],
                    "sft_exps_sync_verified": seed == 42 or bool(validated_seed17[side]),
                }
            )
    csv_path = output_dir / "runs.csv"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=output_dir, delete=False
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
        temporary_csv = Path(stream.name)
    os.replace(temporary_csv, csv_path)

    report = "\n".join(
        [
            "# MiniLM-5ep SFT — fast loss confirmation",
            "",
            "This is directional replication on the prespecified seeds 42 and 17; it is not formal inference over training-seed variability.",
            "",
            f"Decision: **{outcome['selected_role']}**; loss accepted: **{outcome['accepted']}**.",
            f"Seed 42 IID delta (loss − BCE): **{outcome['screen_seed_iid_delta']:.9f}**.",
            f"Seed 17 IID delta (loss − BCE): **{outcome['replication_seed_iid_delta']:.9f}**.",
            f"Mean IID delta: **{outcome['mean_iid_delta']:.9f}** (required ≥ 0.002 and both deltas > 0).",
            "",
            "Both new seed-17 runs were strictly validated as synchronized to `sft_exps`. Hard/OOD are diagnostic only.",
            "No ODS submission/runtime check and no checkpoint download are part of this shortened protocol; the selected seed-42 run is recorded only as a recipe reference.",
            "",
        ]
    )
    _atomic_write(output_dir / "report.md", report)
    _atomic_write(
        output_dir / "fast_confirmation_completed.json",
        canonical_json_dumps(
            {
                "status": "complete",
                "summary_payload_sha256": result["summary_payload_sha256"],
                "selected_seed42_recipe_run_id": result[
                    "selected_seed42_recipe_reference"
                ]["run_id"],
                "checkpoint_download_performed": False,
                "ods_submission_or_runtime_check_performed": False,
            }
        )
        + "\n",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "dry-run", "submit", "summarize"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--locks-dir", type=Path, default=DEFAULT_LOCKS_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument(
        "--fast-track-policy", type=Path, default=fast_track.DEFAULT_POLICY
    )
    parser.add_argument(
        "--fast-track-receipt", type=Path, default=fast_track.DEFAULT_RECEIPT
    )
    parser.add_argument(
        "--fast-track-freeze-manifest",
        type=Path,
        default=fast_track.DEFAULT_FREEZE_MANIFEST,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    protocol_path = _resolved(args.protocol)
    plan_path = _resolved(args.plan)
    summary_path = _resolved(args.summary)
    baseline_summary_path = _resolved(args.baseline_summary)
    locks_dir = _resolved(args.locks_dir)
    artifacts_dir = _resolved(args.artifacts_dir)
    output_dir = _resolved(args.output_dir)
    env_file = _resolved(args.env_file)
    policy_path = _resolved(args.fast_track_policy)
    receipt_path = _resolved(args.fast_track_receipt)
    freeze_manifest_path = _resolved(args.fast_track_freeze_manifest)
    manifest, lock_path, lock, selected = prepare(
        protocol_path=protocol_path,
        plan_path=plan_path,
        summary_path=summary_path,
        baseline_summary_path=baseline_summary_path,
        locks_dir=locks_dir,
        artifacts_dir=artifacts_dir,
        output_dir=output_dir,
        policy_path=policy_path,
        receipt_path=receipt_path,
        freeze_manifest_path=freeze_manifest_path,
    )
    if args.action in {"dry-run", "submit"}:
        run_selected(
            manifest=manifest,
            plan_path=plan_path,
            lock_path=lock_path,
            env_file=env_file,
            action=args.action,
            policy_path=policy_path,
            receipt_path=receipt_path,
            freeze_manifest_path=freeze_manifest_path,
        )
    elif args.action == "summarize":
        protocol = load_protocol(protocol_path, plan=builder.load_plan(plan_path))
        result = summarize(
            protocol=protocol,
            manifest=manifest,
            selected=selected,
            artifacts_dir=artifacts_dir,
            output_dir=output_dir,
        )
        print(json.dumps(result["selection"], ensure_ascii=False, indent=2))
    else:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
