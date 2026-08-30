#!/usr/bin/env python3
"""Plan or read-only-download the selected MiniLM-5ep SFT seed-42 checkpoint.

The final confirmation summary selects a recipe family, while the confirmation
lock binds that family to the already executed seed-42 origin.  This utility
resolves that origin and its one execution authority without selecting a model
again.  The default mode is local plan-only.  ``--download`` is deliberately
limited to the read-only Kaggle ``kernels status`` and ``kernels output`` API
operations; it never submits, versions, repairs Sheets state, or resubmits a
kernel.

The campaign locks do not pre-bind a Kaggle kernel-version metadata document,
so this tool does not pretend that a post-hoc ``kernels pull -m`` response is an
independent content authority.  It explicitly trusts authenticated Kaggle
status/output for the fixed campaign owner, then binds the downloaded bytes by
lock provenance, full IID behavioral replay, a content address, and a local
atomic no-replace manifest.  Modes 0444/0555 are tamper-evidence, not an OS
immutable flag: local same-UID isolation is an explicit assumption, and every
reuse replays and rehashes before acceptance.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import stat
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import create_minilm_5ep_sft_hparam_notebooks as builder
import materialize_minilm_5ep_sft_loss_confirmation as adaptive
import run_kaggle_notebook as kaggle
import run_minilm_5ep_sft_hparam_kaggle as launcher
import summarize_minilm_5ep_sft_hparams as summarizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "configs" / "minilm_5ep_sft_hparam_search_v1.json"
DEFAULT_CONFIRMATION_LOCK = (
    ROOT
    / "reports"
    / "minilm_5ep_sft_hparam_search_v1"
    / "stage_locks"
    / "confirmation__matched_seeds.lock.json"
)
DEFAULT_CONFIRMATION_SUMMARY = (
    ROOT / "reports" / "minilm_5ep_sft_hparam_search_v1" / "summary.json"
)
SELECTED_CHECKPOINTS_ROOT = ROOT / "artifacts" / "kaggle" / "selected_checkpoints"

CONFIRMATION_STAGE = "confirmation__matched_seeds"
CAMPAIGN_KAGGLE_OWNER = "alexproger23"
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_KIND = "minilm_5ep_sft_selected_checkpoint_manifest"
MANIFEST_SUFFIX = ".selected-checkpoint.json"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
REQUIRED_MODEL_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "training_config.json",
    "training_report.json",
    "iid_validation_predictions.parquet",
    "hard_validation_predictions.parquet",
    "ood_validation_predictions.parquet",
)
REQUIRED_ROOT_FILES = {
    "notebook_completed.json",
    "baseline_comparison.json",
    "google_sheets_sync.json",
    "experiment_run_id.txt",
    "experiment_started_at_utc.txt",
    "cross_encoder_config.json",
}
IID_REPLAY_COLUMNS = (
    "pair_index",
    "id1",
    "id2",
    "target",
    "product_text_1",
    "product_text_2",
    "score",
    "score_ab",
    "score_ba",
)
IID_EXPECTED_ROWS = 12_000
INFERENCE_MAX_LENGTH = 384
# Saved validation scores were produced by FP16 autocast on T4, while the
# verifier intentionally loads the same checkpoint in deterministic FP32 CPU
# mode.  A complete 24,000-orientation replay of the existing MiniLM checkpoint
# measured a maximum absolute backend difference of 0.00224358.  The fixed
# 0.003 ceiling is conservative relative to that full calibration while still
# rejecting behaviorally different weights or tokenization.
SCORE_ABSOLUTE_TOLERANCE = 3e-3


class SelectedCheckpointError(RuntimeError):
    """Raised when selection, provenance, or a downloaded checkpoint differs."""


@dataclass(frozen=True)
class SelectedCheckpoint:
    campaign: str
    confirmation_lock_path: str
    confirmation_lock_payload_sha256: str
    confirmation_summary_path: str
    confirmation_summary_payload_sha256: str
    selected_recipe_group_id: str
    recipe_family_sha256: str
    origin_id: str
    experiment: str
    run_id: str
    kernel_slug: str
    kaggle_kernel_ref: str
    seed: int
    recipe_sha256: str
    loss_variant: str
    loss_hook_sha256: str
    code_bundle_sha256: str
    source_authority_kind: str
    source_lock_path: str | None
    source_lock_payload_sha256: str | None
    bound_output_directory: str
    materialization_root: str


def _reject_constant(value: str) -> None:
    raise SelectedCheckpointError(f"Non-finite JSON constant is forbidden: {value}")


def _object_without_duplicate_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SelectedCheckpointError(f"Duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _parse_json_object(serialized: str, *, label: str, path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            serialized,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise SelectedCheckpointError(f"Invalid {label}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SelectedCheckpointError(f"{label} must be a JSON object: {path}")
    return payload


def load_secure_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        serialized = secure_read_bytes(path, label=label).decode("utf-8")
    except UnicodeDecodeError as error:
        raise SelectedCheckpointError(f"Invalid UTF-8 {label}: {path}") from error
    return _parse_json_object(serialized, label=label, path=path)


def assert_no_symlink_components(path: Path, *, label: str) -> Path:
    """Resolve an existing path while rejecting a symlink in every component."""
    if not path.is_absolute():
        raise SelectedCheckpointError(f"{label} must be absolute: {path}")
    normalized = Path(os.path.normpath(str(path)))
    if normalized != path:
        raise SelectedCheckpointError(f"{label} is not a normalized path: {path}")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise SelectedCheckpointError(f"Missing {label} component: {current}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise SelectedCheckpointError(
                f"{label} contains a symlink component: {current}"
            )
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:  # pragma: no cover - lstat above gives the usual path.
        raise SelectedCheckpointError(f"Missing {label}: {path}") from error
    if resolved != path:
        raise SelectedCheckpointError(f"{label} does not resolve to itself: {path}")
    return resolved


def secure_read_bytes_and_sha256(path: Path, *, label: str) -> tuple[bytes, str]:
    """Read and hash once through an O_NOFOLLOW fd bound to one linked inode."""
    path = assert_no_symlink_components(path, label=label)
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise SelectedCheckpointError(
            f"{label} is not a singly-linked regular file: {path}"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise SelectedCheckpointError(f"{label} inode changed while opening")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
            digest.update(block)
        after = os.fstat(descriptor)
        path_after = os.lstat(path)
        if (
            (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_nlink,
            )
            != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
                opened.st_nlink,
            )
            or _regular_file_snapshot(path_after) != _regular_file_snapshot(after)
        ):
            raise SelectedCheckpointError(f"{label} changed while being read")
        return b"".join(chunks), digest.hexdigest()
    finally:
        os.close(descriptor)


def secure_read_bytes(path: Path, *, label: str) -> bytes:
    return secure_read_bytes_and_sha256(path, label=label)[0]


def secure_file_sha256(path: Path, *, label: str) -> str:
    """Hash one regular file without following or reopening its pathname."""
    path = assert_no_symlink_components(path, label=label)
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise SelectedCheckpointError(
            f"{label} is not a singly-linked regular file: {path}"
        )
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise SelectedCheckpointError(f"{label} inode changed while opening")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        path_after = os.lstat(path)
        if (
            (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_nlink,
            )
            != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
                opened.st_nlink,
            )
            or _regular_file_snapshot(path_after) != _regular_file_snapshot(after)
        ):
            raise SelectedCheckpointError(f"{label} changed while being hashed")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _require_sha(value: Any, *, label: str) -> str:
    result = value.strip() if isinstance(value, str) else ""
    if SHA256_RE.fullmatch(result) is None:
        raise SelectedCheckpointError(f"{label} is not a SHA-256")
    return result


def _require_text(value: Any, *, label: str) -> str:
    result = value.strip() if isinstance(value, str) else ""
    if not result:
        raise SelectedCheckpointError(f"{label} must be a non-empty string")
    return result


def _resolve_file(path: Path, *, label: str) -> Path:
    resolved = assert_no_symlink_components(path, label=label)
    if not resolved.is_file():
        raise SelectedCheckpointError(f"{label} is not a file: {resolved}")
    return resolved


def validate_confirmation_summary(
    summary: Mapping[str, Any],
    *,
    summary_path: Path,
    lock: Mapping[str, Any],
    lock_path: Path,
    plan: Mapping[str, Any],
) -> str:
    """Validate the final confirmation projection and return its pre-gate choice."""
    stored_summary_sha = _require_sha(
        summary.get("summary_payload_sha256"), label="confirmation summary payload SHA"
    )
    unhashed = dict(summary)
    unhashed.pop("summary_payload_sha256", None)
    if adaptive.canonical_sha256(unhashed) != stored_summary_sha:
        raise SelectedCheckpointError("Confirmation summary payload SHA differs")
    if (
        summary.get("schema_version") != 2
        or summary.get("campaign") != plan.get("campaign")
        or summary.get("mode") != "confirmation"
        or summary.get("effective_stage") != CONFIRMATION_STAGE
        or lock.get("mode") != "confirmation"
        or lock.get("effective_stage") != CONFIRMATION_STAGE
        or lock.get("execution_status") != "runnable"
    ):
        raise SelectedCheckpointError("Lock/summary is not the final confirmation stage")

    lock_sha = _require_sha(
        lock.get("lock_payload_sha256"), label="confirmation lock payload SHA"
    )
    campaign_hashes = summary.get("execution_campaign_lock_sha256s")
    runnable_hashes = summary.get("execution_lock_sha256s")
    if (
        not isinstance(campaign_hashes, list)
        or not isinstance(runnable_hashes, list)
        or lock_sha not in campaign_hashes
        or lock_sha not in runnable_hashes
    ):
        raise SelectedCheckpointError("Confirmation summary omits its root lock SHA")

    resolved_lock_path = _resolve_file(
        lock_path, label="confirmation root lock"
    )
    closure = summary.get("adaptive_closure")
    root_refs = [
        row
        for row in closure
        if isinstance(row, Mapping)
        and row.get("lock_payload_sha256") == lock_sha
    ] if isinstance(closure, list) else []
    if len(root_refs) != 1:
        raise SelectedCheckpointError("Summary has no unique confirmation root-lock ref")
    root_ref = root_refs[0]
    root_ref_path = _resolve_file(
        Path(_require_text(root_ref.get("lock_path"), label="summary root-lock path")),
        label="summary root-lock path",
    )
    if (
        root_ref.get("schema_version") != 2
        or root_ref.get("kind") != adaptive.LOCK_KIND
        or root_ref.get("mode") != "confirmation"
        or root_ref.get("effective_stage") != CONFIRMATION_STAGE
        or root_ref.get("execution_status") != "runnable"
        or root_ref_path != resolved_lock_path
    ):
        raise SelectedCheckpointError("Summary root-lock reference differs")

    confirmation = summary.get("confirmation")
    stages = summary.get("stages")
    stage = stages.get(CONFIRMATION_STAGE) if isinstance(stages, Mapping) else None
    if not isinstance(confirmation, Mapping) or not isinstance(stage, Mapping):
        raise SelectedCheckpointError("Summary has no confirmation decision projection")
    decision_status = confirmation.get("decision_status")
    if (
        decision_status not in {"runtime_gate_pending", "ready"}
        or stage.get("runs_complete") is not True
        or stage.get("decision_status") != decision_status
        or stage.get("completed_new_runs") != stage.get("expected_new_runs")
    ):
        raise SelectedCheckpointError(
            "Confirmation must have all matched-seed runs and a fixed pre-gate choice"
        )
    selected = _require_text(
        confirmation.get("selection_before_runtime_gate"),
        label="selection_before_runtime_gate",
    )
    if selected not in confirmation.get("practical_shortlist_recipe_group_ids", []):
        raise SelectedCheckpointError("Selected recipe is absent from practical shortlist")
    runtime_gate = confirmation.get("runtime_gate")
    if not isinstance(runtime_gate, Mapping) or runtime_gate.get("required") is not True:
        raise SelectedCheckpointError("Confirmation runtime gate contract is missing")
    if decision_status == "runtime_gate_pending":
        if (
            summary.get("execution_status") != "pending"
            or confirmation.get("selected_recipe_group_id") is not None
            or runtime_gate.get("status") != "pending"
            or runtime_gate.get("selected_recipe_group_id") != selected
        ):
            raise SelectedCheckpointError("Pending runtime-gate projection is inconsistent")
    else:
        if (
            summary.get("execution_status") != "complete"
            or confirmation.get("selected_recipe_group_id") != selected
            or runtime_gate.get("status") != "passed"
            or runtime_gate.get("selected_recipe_group_id") != selected
        ):
            raise SelectedCheckpointError("Passed runtime-gate projection is inconsistent")

    groups = confirmation.get("groups")
    selected_evaluations = [
        row
        for row in groups
        if isinstance(row, Mapping) and row.get("recipe_group_id") == selected
    ] if isinstance(groups, list) else []
    if (
        len(selected_evaluations) != 1
        or selected_evaluations[0].get("complete") is not True
        or selected_evaluations[0].get("accepted") is not True
    ):
        raise SelectedCheckpointError("Selected confirmation group is not complete/accepted")
    return selected


def _load_lock_closure(
    *,
    plan: Mapping[str, Any],
    base_config: Mapping[str, Any],
    root_lock: Mapping[str, Any],
    root_lock_path: Path,
) -> list[tuple[dict[str, Any], Path]]:
    """Load the exact prerequisite graph referenced by the confirmation lock."""
    ordered: list[tuple[dict[str, Any], Path]] = []
    by_sha: dict[str, Path] = {}
    visiting: set[str] = set()

    def visit(lock: Mapping[str, Any], path: Path) -> None:
        lock_sha = _require_sha(lock.get("lock_payload_sha256"), label="closure lock SHA")
        resolved_path = _resolve_file(path, label="campaign closure lock")
        if lock_sha in by_sha:
            if by_sha[lock_sha] != resolved_path:
                raise SelectedCheckpointError(
                    "One lock payload SHA is referenced from multiple paths"
                )
            return
        if lock_sha in visiting:
            raise SelectedCheckpointError("Confirmation lock closure contains a cycle")
        visiting.add(lock_sha)
        if lock.get("schema_version") == 2:
            references = lock.get("prerequisites")
            if not isinstance(references, list):
                raise SelectedCheckpointError("Adaptive prerequisites are malformed")
            for reference in references:
                if not isinstance(reference, Mapping):
                    raise SelectedCheckpointError("Adaptive prerequisite is malformed")
                prerequisite_path = Path(
                    _require_text(reference.get("lock_path"), label="prerequisite path")
                )
                prerequisite_path = _resolve_file(
                    prerequisite_path, label="campaign prerequisite lock"
                )
                prerequisite = builder.load_campaign_lock(
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
                    raise SelectedCheckpointError(
                        "Prerequisite differs from its confirmation-lock reference"
                    )
                visit(prerequisite, prerequisite_path)
        visiting.remove(lock_sha)
        by_sha[lock_sha] = resolved_path
        ordered.append((dict(lock), resolved_path))

    visit(root_lock, root_lock_path)
    return ordered


def _entry_matches_origin(entry: Mapping[str, Any], origin: Mapping[str, Any]) -> bool:
    expected_notes = entry.get("expected_notes")
    notes_match = expected_notes == origin.get("completion_notes")
    alias = entry.get("provenance_alias")
    if not notes_match and isinstance(alias, Mapping):
        notes_match = (
            alias.get("accepted_completion_notes_sha256")
            == origin.get("completion_notes_sha256")
            and entry.get("role") == "current_protocol_control"
        )
    return bool(
        entry.get("experiment") == origin.get("experiment")
        and entry.get("kernel_slug") == origin.get("kernel_slug")
        and entry.get("recipe_sha256") == origin.get("recipe_sha256")
        and entry.get("source_sha256") == origin.get("code_bundle_sha256")
        and entry.get("loss_variant") == origin.get("loss_variant")
        and entry.get("loss_hook_sha256") == origin.get("loss_hook_sha256")
        and entry.get("expected_config") == origin.get("resolved_config")
        and notes_match
    )


def _schema_v1_parent_matches_origin(
    parent: Mapping[str, Any], origin: Mapping[str, Any]
) -> bool:
    mapping = {
        "experiment": "experiment",
        "run_id": "run_id",
        "kernel_slug": "kernel_slug",
        "recipe_sha256": "recipe_sha256",
        "code_bundle_sha256": "code_bundle_sha256",
        "loss_variant": "loss_variant",
        "loss_hook_sha256": "loss_hook_sha256",
        "resolved_config": "resolved_config",
        "notes": "completion_notes",
        "completion_sha256": "completion_sha256",
        "iid_predictions_sha256": "iid_predictions_sha256",
        "iid_predictions_relative_path": "iid_predictions_relative_path",
        "training_config_artifact_sha256": "training_config_artifact_sha256",
    }
    return all(
        parent.get(parent_key) == origin.get(origin_key)
        for parent_key, origin_key in mapping.items()
    )


def _entry_from_frozen_parent(
    *, plan: Mapping[str, Any], origin: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the pure run-validator view from an already validated lock parent."""
    return {
        "stage": origin.get("origin_effective_stage"),
        "experiment": origin["experiment"],
        "role": origin.get("source_role", "stage_anchor"),
        "kernel_slug": origin["kernel_slug"],
        "recipe_sha256": origin["recipe_sha256"],
        "source_sha256": origin["code_bundle_sha256"],
        "loss_variant": origin["loss_variant"],
        "loss_hook_sha256": origin["loss_hook_sha256"],
        "expected_config": origin["resolved_config"],
        "baseline_metrics": dict(plan["baseline_metrics"]),
        "expected_notes": origin["completion_notes"],
        "provenance_alias": None,
    }


def _source_dispatch(
    *,
    plan: Mapping[str, Any],
    base_config: Mapping[str, Any],
    confirmation_lock: Mapping[str, Any],
    confirmation_lock_path: Path,
    origin: Mapping[str, Any],
) -> tuple[dict[str, Any], str, Path | None, str | None]:
    matches: list[tuple[dict[str, Any], str, Path | None, str | None]] = []

    # The original protocol control and initial LR line were dispatched from
    # the ready plan before any stage lock existed.
    root_entries = launcher.campaign_variants(
        plan,
        stage=None,
        only=None,
        stage_lock=None,
    )
    for entry in root_entries:
        if entry.get("experiment") == origin.get("experiment"):
            if not _entry_matches_origin(entry, origin):
                raise SelectedCheckpointError(
                    "Selected origin differs from its ready-plan execution authority"
                )
            matches.append((entry, "ready_plan_variant", None, None))

    for lock, path in _load_lock_closure(
        plan=plan,
        base_config=base_config,
        root_lock=confirmation_lock,
        root_lock_path=confirmation_lock_path,
    ):
        if lock.get("execution_status", "runnable") != "runnable":
            continue
        if lock.get("schema_version") == 1:
            parent = lock.get("parent")
            if (
                isinstance(parent, Mapping)
                and parent.get("experiment") == origin.get("experiment")
            ):
                if not _schema_v1_parent_matches_origin(parent, origin):
                    raise SelectedCheckpointError(
                        "Selected origin differs from its immutable source-lock parent"
                    )
                matches.append(
                    (
                        _entry_from_frozen_parent(plan=plan, origin=origin),
                        "campaign_lock_parent",
                        path,
                        str(lock["lock_payload_sha256"]),
                    )
                )
        variants = lock.get("resolved_stage", {}).get("variants", [])
        if not any(
            isinstance(variant, Mapping)
            and variant.get("experiment") == origin.get("experiment")
            for variant in variants
        ):
            continue
        entries = launcher.campaign_variants(
            plan,
            stage=None,
            only=None,
            stage_lock=lock,
        )
        candidates = [
            entry
            for entry in entries
            if entry.get("experiment") == origin.get("experiment")
        ]
        if len(candidates) != 1 or not _entry_matches_origin(candidates[0], origin):
            raise SelectedCheckpointError(
                "Selected origin differs from its immutable source-lock authority"
            )
        matches.append(
            (
                candidates[0],
                "campaign_lock_variant",
                path,
                str(lock["lock_payload_sha256"]),
            )
        )

    if len(matches) != 1:
        identities = [
            {
                "kind": kind,
                "path": str(path) if path is not None else None,
                "sha256": lock_sha,
            }
            for _, kind, path, lock_sha in matches
        ]
        raise SelectedCheckpointError(
            "Selected seed-42 origin has no unique execution authority: "
            + json.dumps(identities, sort_keys=True)
        )
    return matches[0]


def _verify_summary_selection_from_artifacts(
    *,
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    lock_path: Path,
    summary: Mapping[str, Any],
) -> None:
    """Recompute matched-seed selection from the exact local run artifacts."""
    trusted_manifest_path = _resolve_file(
        adaptive.trusted_provenance_manifest_path(lock_path),
        label="confirmation trusted-provenance manifest",
    )
    trusted = adaptive.load_trusted_provenance(
        trusted_manifest_path,
        plan=plan,
    )
    artifacts_dir = assert_no_symlink_components(
        Path(str(trusted["artifacts_dir"])),
        label="confirmation trusted artifacts directory",
    )
    if not artifacts_dir.is_dir():
        raise SelectedCheckpointError(
            "Confirmation trusted artifacts path is not a directory"
        )
    rows = summarizer.adaptive_run_rows(
        plan=plan,
        lock=lock,
        artifacts_dir=artifacts_dir,
    )
    recomputed = summarizer.adaptive_confirmation_projection(
        plan=plan,
        lock=lock,
        rows=rows,
        artifacts_dir=artifacts_dir,
        runtime_check_path=None,
    )
    observed = summary.get("confirmation")
    if not isinstance(recomputed, Mapping) or not isinstance(observed, Mapping):
        raise SelectedCheckpointError("Could not recompute confirmation selection")
    stable_fields = {
        "comparison",
        "acceptance",
        "groups",
        "practical_shortlist_recipe_group_ids",
        "selection_before_runtime_gate",
    }
    mismatches = {
        key: {"expected": recomputed.get(key), "actual": observed.get(key)}
        for key in stable_fields
        if observed.get(key) != recomputed.get(key)
    }
    if mismatches:
        raise SelectedCheckpointError(
            "Confirmation summary selection differs from bound run artifacts: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    if summary.get("runs") != rows:
        raise SelectedCheckpointError(
            "Confirmation summary run ledger differs from bound run artifacts"
        )


def _lock_bound_kernel_ref(
    *, origin: Mapping[str, Any], completion_path: Path
) -> str:
    serialized, observed_sha = secure_read_bytes_and_sha256(
        completion_path, label="selected origin completion"
    )
    if observed_sha != origin.get("completion_sha256"):
        raise SelectedCheckpointError("Selected origin completion SHA differs")
    try:
        completion = _parse_json_object(
            serialized.decode("utf-8"),
            label="selected origin completion",
            path=completion_path,
        )
    except UnicodeDecodeError as error:
        raise SelectedCheckpointError(
            f"Invalid UTF-8 selected origin completion: {completion_path}"
        ) from error
    kernel_slug = _require_text(
        origin.get("kernel_slug"), label="selected origin kernel slug"
    )
    if re.fullmatch(r"[a-z0-9][a-z0-9-]*", kernel_slug) is None:
        raise SelectedCheckpointError(
            "Selected origin kernel slug is unsafe"
        )
    expected = f"{CAMPAIGN_KAGGLE_OWNER}/{kernel_slug}"
    recorded = completion.get("kaggle_kernel_ref")
    if recorded not in {"", expected}:
        raise SelectedCheckpointError(
            "Completion Kaggle ref contradicts the fixed campaign owner/kernel slug"
        )
    return expected


def resolve_selected_checkpoint(
    *,
    plan_path: Path = DEFAULT_PLAN,
    confirmation_lock_path: Path = DEFAULT_CONFIRMATION_LOCK,
    confirmation_summary_path: Path = DEFAULT_CONFIRMATION_SUMMARY,
) -> tuple[SelectedCheckpoint, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve and validate the selected origin; perform no network or writes."""
    plan_path = _resolve_file(plan_path, label="campaign plan")
    confirmation_lock_path = _resolve_file(
        confirmation_lock_path, label="confirmation lock"
    )
    confirmation_summary_path = _resolve_file(
        confirmation_summary_path, label="confirmation summary"
    )
    plan = builder.load_plan(plan_path)
    base_config = builder.cross_builder.load_training_config(builder.BASE_CONFIG_PATH)
    lock = builder.load_campaign_lock(
        confirmation_lock_path,
        plan=plan,
        base_config=base_config,
    )
    summary = load_secure_json_object(
        confirmation_summary_path, label="confirmation summary"
    )
    selected_group_id = validate_confirmation_summary(
        summary,
        summary_path=confirmation_summary_path,
        lock=lock,
        lock_path=confirmation_lock_path,
        plan=plan,
    )
    _verify_summary_selection_from_artifacts(
        plan=plan,
        lock=lock,
        lock_path=confirmation_lock_path,
        summary=summary,
    )

    groups = lock.get("resolved_stage", {}).get("recipe_groups", [])
    selected_groups = [
        group
        for group in groups
        if isinstance(group, Mapping)
        and group.get("recipe_group_id") == selected_group_id
    ]
    if len(selected_groups) != 1:
        raise SelectedCheckpointError("Lock has no unique selected recipe group")
    group = selected_groups[0]
    origin_id = _require_text(
        group.get("origin_seed42_id"), label="selected seed-42 origin ID"
    )
    origins = [
        origin
        for origin in lock.get("origins", [])
        if isinstance(origin, Mapping) and origin.get("origin_id") == origin_id
    ]
    if len(origins) != 1:
        raise SelectedCheckpointError("Lock has no unique selected seed-42 origin")
    origin = origins[0]
    config = origin.get("resolved_config")
    if (
        not isinstance(config, Mapping)
        or config.get("seed") != 42
        or lock.get("decision", {}).get("selected_checkpoint_seed") != 42
        or origin.get("recipe_family_sha256") != group.get("recipe_family_sha256")
        or origin.get("loss_variant") != group.get("loss_variant")
        or origin.get("loss_hook_sha256") != group.get("loss_hook_sha256")
    ):
        raise SelectedCheckpointError("Selected group/origin seed-42 identity differs")

    evaluation = next(
        row
        for row in summary["confirmation"]["groups"]
        if row["recipe_group_id"] == selected_group_id
    )
    if evaluation.get("recipe_family_sha256") != group.get("recipe_family_sha256"):
        raise SelectedCheckpointError("Summary selected recipe-family SHA differs")
    seed_run_ids = evaluation.get("seed_run_ids")
    if not isinstance(seed_run_ids, Mapping) or seed_run_ids.get("42") != origin.get(
        "run_id"
    ):
        raise SelectedCheckpointError("Summary selected seed-42 run ID differs")

    entry, authority_kind, source_lock_path, source_lock_sha = _source_dispatch(
        plan=plan,
        base_config=base_config,
        confirmation_lock=lock,
        confirmation_lock_path=confirmation_lock_path,
        origin=origin,
    )
    completion_path = Path(
        _require_text(
            origin.get("completion_artifact_path"),
            label="selected origin completion path",
        )
    )
    bound_output_directory = completion_path.parent
    kaggle_kernel_ref = _lock_bound_kernel_ref(
        origin=origin,
        completion_path=completion_path,
    )
    if (
        bound_output_directory.name != origin.get("kernel_slug")
        or assert_no_symlink_components(
            bound_output_directory, label="bound selected output"
        )
        != bound_output_directory
    ):
        raise SelectedCheckpointError(
            "Selected origin destination is not its bound kernel artifact root"
        )

    selected = SelectedCheckpoint(
        campaign=str(plan["campaign"]),
        confirmation_lock_path=str(confirmation_lock_path),
        confirmation_lock_payload_sha256=str(lock["lock_payload_sha256"]),
        confirmation_summary_path=str(confirmation_summary_path),
        confirmation_summary_payload_sha256=str(summary["summary_payload_sha256"]),
        selected_recipe_group_id=selected_group_id,
        recipe_family_sha256=str(group["recipe_family_sha256"]),
        origin_id=origin_id,
        experiment=str(origin["experiment"]),
        run_id=str(origin["run_id"]),
        kernel_slug=str(origin["kernel_slug"]),
        kaggle_kernel_ref=kaggle_kernel_ref,
        seed=42,
        recipe_sha256=str(origin["recipe_sha256"]),
        loss_variant=str(origin["loss_variant"]),
        loss_hook_sha256=str(origin["loss_hook_sha256"]),
        code_bundle_sha256=str(origin["code_bundle_sha256"]),
        source_authority_kind=authority_kind,
        source_lock_path=str(source_lock_path) if source_lock_path else None,
        source_lock_payload_sha256=source_lock_sha,
        bound_output_directory=str(bound_output_directory),
        materialization_root=str(
            SELECTED_CHECKPOINTS_ROOT / str(origin["kernel_slug"])
        ),
    )
    return selected, dict(origin), dict(entry), summary


def validate_safetensors_file(path: Path) -> None:
    path = _resolve_file(path, label="model.safetensors")
    before = os.lstat(path)
    if before.st_size <= 8 or before.st_nlink != 1:
        raise SelectedCheckpointError(
            "model.safetensors is empty/truncated/hard-linked"
        )
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as tensors:
            keys = list(tensors.keys())
            if not keys:
                raise SelectedCheckpointError("model.safetensors contains no tensors")
            # get_slice validates every declared dtype/shape/data offset without
            # duplicating the full 450 MiB state dict in memory.
            for key in keys:
                tensor_slice = tensors.get_slice(key)
                tensor_slice.get_shape()
    except SelectedCheckpointError:
        raise
    except Exception as error:
        raise SelectedCheckpointError(
            f"model.safetensors failed strict safetensors validation: {error}"
        ) from error
    after = os.lstat(path)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_nlink,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_nlink,
    ):
        raise SelectedCheckpointError("model.safetensors changed during validation")


def snapshot_regular_tree(root: Path, *, label: str) -> dict[str, tuple[int, ...]]:
    """Return a non-following inode snapshot and reject every special/symlink node."""
    root = assert_no_symlink_components(root, label=label)
    if not root.is_dir():
        raise SelectedCheckpointError(f"{label} is not a directory: {root}")
    result: dict[str, tuple[int, ...]] = {}
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                metadata = entry.stat(follow_symlinks=False)
                relative = str(path.relative_to(root))
                if stat.S_ISLNK(metadata.st_mode):
                    raise SelectedCheckpointError(
                        f"{label} contains a symlink: {relative}"
                    )
                if stat.S_ISDIR(metadata.st_mode):
                    result[relative + "/"] = (
                        metadata.st_dev,
                        metadata.st_ino,
                        metadata.st_mode,
                        metadata.st_mtime_ns,
                        metadata.st_ctime_ns,
                        metadata.st_nlink,
                    )
                    pending.append(path)
                elif stat.S_ISREG(metadata.st_mode):
                    if metadata.st_nlink != 1:
                        raise SelectedCheckpointError(
                            f"{label} contains a hard-linked file: {relative}"
                        )
                    result[relative] = (
                        metadata.st_dev,
                        metadata.st_ino,
                        metadata.st_mode,
                        metadata.st_size,
                        metadata.st_mtime_ns,
                        metadata.st_ctime_ns,
                        metadata.st_nlink,
                    )
                else:
                    raise SelectedCheckpointError(
                        f"{label} contains a non-regular node: {relative}"
                    )
    return dict(sorted(result.items()))


def _regular_file_snapshot(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_nlink,
    )


def _directory_snapshot(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_nlink,
    )


def _safe_directory_entry_name(name: str, *, label: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or Path(name).name != name
        or str(Path(name)) != name
    ):
        raise SelectedCheckpointError(f"Unsafe {label} directory entry name")
    return name


def _open_directory_fd(path: Path, *, label: str) -> int:
    path = assert_no_symlink_components(path, label=label)
    before = os.lstat(path)
    if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise SelectedCheckpointError(f"{label} is not a real directory")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or _inode_identity(opened) != _inode_identity(before)
    ):
        os.close(descriptor)
        raise SelectedCheckpointError(f"{label} inode changed while opening")
    return descriptor


def _open_directory_entry_at(
    parent_descriptor: int, name: str, *, label: str
) -> int:
    name = _safe_directory_entry_name(name, label=label)
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_descriptor,
    )
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise SelectedCheckpointError(f"{label} is not a real directory")
    return descriptor


def _read_regular_fd(
    descriptor: int, *, label: str
) -> tuple[bytes, tuple[int, ...]]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise SelectedCheckpointError(f"{label} is not a singly-linked regular file")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        chunks.append(block)
    after = os.fstat(descriptor)
    snapshot = _regular_file_snapshot(after)
    if snapshot != _regular_file_snapshot(before):
        raise SelectedCheckpointError(f"{label} changed while being read")
    return b"".join(chunks), snapshot


def _atomic_rename_no_replace_at(
    *,
    source_parent_descriptor: int,
    source_name: str,
    destination_parent_descriptor: int,
    destination_name: str,
) -> bool:
    """Atomically rename one entry, returning False if the target exists."""
    source_name = _safe_directory_entry_name(source_name, label="publish source")
    destination_name = _safe_directory_entry_name(
        destination_name, label="publish destination"
    )
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source_name)
    destination_bytes = os.fsencode(destination_name)
    if sys.platform == "darwin":
        function = getattr(library, "renameatx_np", None)
        if function is None:  # pragma: no cover - supported macOS is 10.12+.
            raise SelectedCheckpointError("renameatx_np is unavailable")
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            source_parent_descriptor,
            source_bytes,
            destination_parent_descriptor,
            destination_bytes,
            0x00000004,  # RENAME_EXCL
        )
    elif sys.platform.startswith("linux"):
        function = getattr(library, "renameat2", None)
        if function is None:  # pragma: no cover - fail closed on unusual libc.
            raise SelectedCheckpointError("renameat2 is unavailable")
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            source_parent_descriptor,
            source_bytes,
            destination_parent_descriptor,
            destination_bytes,
            0x00000001,  # RENAME_NOREPLACE
        )
    else:  # pragma: no cover - campaign tooling runs on macOS/Linux.
        raise SelectedCheckpointError(
            "No supported atomic no-replace rename primitive on this platform"
        )
    if result == 0:
        return True
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        return False
    raise SelectedCheckpointError(
        "Atomic no-replace publish failed: " + os.strerror(error_number)
    )


def _assert_directory_entry_matches_fd(
    *,
    parent_descriptor: int,
    name: str,
    descriptor: int,
    label: str,
) -> os.stat_result:
    name = _safe_directory_entry_name(name, label=label)
    opened = os.fstat(descriptor)
    observed = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
        or _inode_identity(opened) != _inode_identity(observed)
    ):
        raise SelectedCheckpointError(f"{label} pathname/inode changed")
    return opened


def _remove_owned_regular_at(
    *,
    parent_descriptor: int,
    name: str,
    expected_identity: tuple[int, int],
) -> None:
    name = _safe_directory_entry_name(name, label="owned temporary file")
    try:
        metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    if (
        not stat.S_ISREG(metadata.st_mode)
        or _inode_identity(metadata) != expected_identity
    ):
        raise SelectedCheckpointError(
            "Refusing to clean a temporary file whose inode/type changed"
        )
    os.unlink(name, dir_fd=parent_descriptor)


def hash_captured_regular_files(
    root: Path,
    snapshot: Mapping[str, tuple[int, ...]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    """Rehash every file from an inode snapshot and reject metadata drift."""
    root = assert_no_symlink_components(root, label=label)
    result: dict[str, dict[str, Any]] = {}
    for relative, captured in sorted(snapshot.items()):
        if relative.endswith("/"):
            continue
        path = _contained_file(root, Path(relative), label=f"{label} file {relative}")
        before = os.lstat(path)
        if _regular_file_snapshot(before) != captured:
            raise SelectedCheckpointError(
                f"{label} file metadata changed before hashing: {relative}"
            )
        digest = secure_file_sha256(path, label=f"{label} file {relative}")
        after = os.lstat(path)
        if _regular_file_snapshot(after) != captured:
            raise SelectedCheckpointError(
                f"{label} file metadata changed while hashing: {relative}"
            )
        result[relative] = {"bytes": before.st_size, "sha256": digest}
    return result


def output_tree_sha256(files: Mapping[str, Mapping[str, Any]]) -> str:
    projection = {
        "schema_version": 1,
        "kind": "minilm_5ep_sft_downloaded_output_tree",
        "files": [
            {
                "path": path,
                "bytes": int(metadata["bytes"]),
                "sha256": str(metadata["sha256"]),
            }
            for path, metadata in sorted(files.items())
        ],
    }
    return adaptive.canonical_sha256(projection)


def _contained_file(root: Path, relative: Path, *, label: str) -> Path:
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise SelectedCheckpointError(f"Unsafe {label} relative path: {relative}")
    candidate = _resolve_file(root / relative, label=label)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise SelectedCheckpointError(f"{label} escapes downloaded output root") from error
    return candidate


def _read_parquet_secure(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> Any:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        payload, observed_sha256 = secure_read_bytes_and_sha256(path, label=label)
        if (
            expected_sha256 is not None
            and observed_sha256
            != _require_sha(expected_sha256, label=f"{label} expected SHA")
        ):
            raise SelectedCheckpointError(f"{label} SHA differs from its authority")
        return pq.read_table(pa.BufferReader(payload)).to_pandas()
    except SelectedCheckpointError:
        raise
    except Exception as error:
        raise SelectedCheckpointError(f"Could not read {label}: {error}") from error


def _load_offline_transformer(model_dir: Path) -> tuple[Any, Any, Any]:
    """Load config, tokenizer, and model locally in deterministic FP32 CPU mode."""
    # ``local_files_only`` is the operative network boundary.  The environment
    # flags additionally prevent a library regression from attempting a hub
    # lookup while resolving classes or metadata.
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        config = AutoConfig.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as error:
        raise SelectedCheckpointError(
            f"Downloaded offline Transformers load failed: {error}"
        ) from error
    architectures = getattr(config, "architectures", None)
    if (
        getattr(config, "model_type", None) != "xlm-roberta"
        or not isinstance(architectures, list)
        or "XLMRobertaForSequenceClassification" not in architectures
        or getattr(config, "num_labels", None) != 1
        or not getattr(tokenizer, "is_fast", False)
        or not isinstance(getattr(tokenizer, "model_max_length", None), int)
        or getattr(tokenizer, "model_max_length", 0) < INFERENCE_MAX_LENGTH
    ):
        raise SelectedCheckpointError(
            "Downloaded config/tokenizer has another inference graph"
        )
    model.to(torch.device("cpu"))
    model.float()
    model.eval()
    return torch, tokenizer, model


def verify_full_iid_replay(
    *,
    model_dir: Path,
    downloaded_iid_path: Path,
    bound_iid_path: Path,
    bound_iid_sha256: str,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Replay both orientations of every frozen IID pair with downloaded assets."""
    if batch_size <= 0:
        raise SelectedCheckpointError("IID replay batch_size must be positive")
    downloaded = _read_parquet_secure(
        downloaded_iid_path, label="downloaded IID predictions"
    )
    bound = _read_parquet_secure(
        bound_iid_path,
        label="bound IID predictions",
        expected_sha256=bound_iid_sha256,
    )
    missing_downloaded = set(IID_REPLAY_COLUMNS) - set(downloaded.columns)
    missing_bound = set(IID_REPLAY_COLUMNS) - set(bound.columns)
    if missing_downloaded or missing_bound:
        raise SelectedCheckpointError(
            "IID replay parquet lacks required columns: "
            f"downloaded={sorted(missing_downloaded)}, bound={sorted(missing_bound)}"
        )
    if len(downloaded) != IID_EXPECTED_ROWS or len(bound) != IID_EXPECTED_ROWS:
        raise SelectedCheckpointError(
            f"IID replay requires exactly {IID_EXPECTED_ROWS} bound pairs"
        )
    try:
        import numpy as np
        import pandas as pd

        pd.testing.assert_frame_equal(
            downloaded.loc[:, IID_REPLAY_COLUMNS].reset_index(drop=True),
            bound.loc[:, IID_REPLAY_COLUMNS].reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
        )
    except (AssertionError, KeyError, ValueError) as error:
        raise SelectedCheckpointError(
            "Downloaded IID pair IDs/texts/scores differ from the lock-bound parquet"
        ) from error
    pair_indices = downloaded["pair_index"].to_numpy()
    if not np.array_equal(pair_indices, np.arange(IID_EXPECTED_ROWS)):
        raise SelectedCheckpointError("IID pair_index is not the exact complete row order")
    if downloaded[["id1", "id2", "product_text_1", "product_text_2"]].isnull().any().any():
        raise SelectedCheckpointError("IID replay inputs contain null IDs/texts")
    if not all(
        isinstance(value, str)
        for column in ("product_text_1", "product_text_2")
        for value in downloaded[column].array
    ):
        raise SelectedCheckpointError("IID replay product texts are not exact strings")
    if not set(downloaded["target"].unique()).issubset({0, 1, 0.0, 1.0}):
        raise SelectedCheckpointError("IID replay targets are not binary labels")
    if any("fallback" in str(column).lower() for column in downloaded.columns):
        raise SelectedCheckpointError("IID prediction artifact contains fallback fields")
    expected_ab = downloaded["score_ab"].to_numpy(dtype=np.float64)
    expected_ba = downloaded["score_ba"].to_numpy(dtype=np.float64)
    expected_mean = downloaded["score"].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(expected_ab).all()
        or not np.isfinite(expected_ba).all()
        or not np.isfinite(expected_mean).all()
        or np.any(expected_ab < 0)
        or np.any(expected_ab > 1)
        or np.any(expected_ba < 0)
        or np.any(expected_ba > 1)
        or not np.allclose(
            expected_mean,
            (expected_ab + expected_ba) / 2.0,
            rtol=0.0,
            atol=1e-7,
        )
    ):
        raise SelectedCheckpointError("Stored IID AB/BA probabilities are invalid")

    torch, tokenizer, model = _load_offline_transformer(model_dir)
    max_difference = 0.0
    checked = 0
    try:
        with torch.inference_mode():
            for start in range(0, IID_EXPECTED_ROWS, batch_size):
                part = downloaded.iloc[start : start + batch_size]
                first = part["product_text_1"].astype(str).tolist()
                second = part["product_text_2"].astype(str).tolist()
                encoded = tokenizer(
                    first + second,
                    second + first,
                    add_special_tokens=True,
                    padding=True,
                    truncation="longest_first",
                    max_length=INFERENCE_MAX_LENGTH,
                    return_tensors="pt",
                )
                packed = {
                    key: value.to(torch.device("cpu"))
                    for key, value in encoded.items()
                }
                outputs = model(**packed)
                logits = getattr(outputs, "logits", None)
                if logits is None or tuple(logits.shape) != (2 * len(part), 1):
                    raise SelectedCheckpointError(
                        "Downloaded model returned an invalid IID replay logit shape"
                    )
                probabilities = (
                    logits.reshape(-1).float().sigmoid().cpu().numpy()
                )
                actual_ab = probabilities[: len(part)].astype(np.float64)
                actual_ba = probabilities[len(part) :].astype(np.float64)
                differences = np.concatenate(
                    [
                        np.abs(actual_ab - expected_ab[start : start + len(part)]),
                        np.abs(actual_ba - expected_ba[start : start + len(part)]),
                    ]
                )
                if not np.isfinite(differences).all():
                    raise SelectedCheckpointError(
                        "Downloaded model produced non-finite IID replay scores"
                    )
                batch_max = float(differences.max(initial=0.0))
                max_difference = max(max_difference, batch_max)
                if batch_max > SCORE_ABSOLUTE_TOLERANCE:
                    raise SelectedCheckpointError(
                        "Downloaded model/tokenizer IID replay differs from frozen "
                        f"AB/BA scores: max_abs_diff={batch_max:.9g} > "
                        f"{SCORE_ABSOLUTE_TOLERANCE}"
                    )
                checked += len(part)
    finally:
        del model
    if checked != IID_EXPECTED_ROWS:
        raise SelectedCheckpointError("IID replay did not score every bound pair")
    return {
        "pairs": checked,
        "orientations": checked * 2,
        "max_length": INFERENCE_MAX_LENGTH,
        "device": "cpu_fp32",
        "saved_score_backend": "t4_fp16_autocast",
        "absolute_tolerance": SCORE_ABSOLUTE_TOLERANCE,
        "max_absolute_difference": max_difference,
        "fallback_pairs": 0,
    }


def validate_downloaded_checkpoint(
    directory: Path,
    *,
    selected: SelectedCheckpoint,
    origin: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a full Kaggle output without mutating any external service."""
    directory = assert_no_symlink_components(directory, label="downloaded output")
    if not directory.is_dir():
        raise SelectedCheckpointError(f"Downloaded output is not a directory: {directory}")
    tree_before = snapshot_regular_tree(directory, label="downloaded output")
    hashes_before = hash_captured_regular_files(
        directory, tree_before, label="downloaded output"
    )
    root_names = {path.name for path in directory.iterdir()}
    if missing_root := REQUIRED_ROOT_FILES - root_names:
        raise SelectedCheckpointError(
            f"Downloaded output lacks required root files: {sorted(missing_root)}"
        )
    if any(name.startswith("sheets_sync_pending") for name in root_names):
        raise SelectedCheckpointError("Downloaded output contains pending Sheets state")
    for name in REQUIRED_ROOT_FILES:
        _resolve_file(directory / name, label=f"required root file {name}")
    try:
        launcher_validation = launcher.validate_run_output(directory, entry=entry)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        raise SelectedCheckpointError(f"Downloaded run validation failed: {error}") from error
    if launcher_validation.get("run_id") != selected.run_id:
        raise SelectedCheckpointError("Downloaded run ID differs from selected origin")

    try:
        old_root = assert_no_symlink_components(
            Path(str(origin["completion_artifact_path"])).parent,
            label="bound selected output",
        )
        relative_bindings = {
            "completion": (
                Path(str(origin["completion_artifact_path"])).relative_to(old_root),
                str(origin["completion_sha256"]),
            ),
            "training_config": (
                Path(str(origin["training_config_artifact_path"])).relative_to(old_root),
                str(origin["training_config_artifact_sha256"]),
            ),
            "iid_predictions": (
                Path(str(origin["iid_predictions_relative_path"])),
                str(origin["iid_predictions_sha256"]),
            ),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise SelectedCheckpointError(
            "Selected-origin artifact bindings are missing or escape their output root"
        ) from error
    for label, (relative_path, expected_sha) in relative_bindings.items():
        candidate = _contained_file(directory, relative_path, label=label)
        if secure_file_sha256(candidate, label=label) != expected_sha:
            raise SelectedCheckpointError(
                f"Downloaded {label} differs from selected-origin SHA"
            )

    model_relative = relative_bindings["training_config"][0].parent
    if len(model_relative.parts) != 1:
        raise SelectedCheckpointError("Selected model directory is not one contained level")
    if (
        relative_bindings["training_config"][0]
        != model_relative / "training_config.json"
        or relative_bindings["iid_predictions"][0]
        != model_relative / "iid_validation_predictions.parquet"
    ):
        raise SelectedCheckpointError(
            "Selected config/IID artifact paths differ from the exact model tree"
        )
    model_dir = assert_no_symlink_components(
        directory / model_relative, label="downloaded model directory"
    )
    if not model_dir.is_dir():
        raise SelectedCheckpointError("Downloaded model path is not a real directory")
    model_prefix = str(model_relative) + "/"
    observed_model_tree = {
        relative[len(model_prefix) :]
        for relative in tree_before
        if relative.startswith(model_prefix) and relative != model_prefix
    }
    if observed_model_tree != set(REQUIRED_MODEL_FILES):
        raise SelectedCheckpointError(
            "Downloaded model output tree differs: "
            + json.dumps(
                {
                    "expected": sorted(REQUIRED_MODEL_FILES),
                    "actual": sorted(observed_model_tree),
                },
                sort_keys=True,
            )
        )
    files: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_MODEL_FILES:
        path = _resolve_file(model_dir / name, label=f"required model file {name}")
        if path.stat().st_size <= 0:
            raise SelectedCheckpointError(f"Required model file is empty: {name}")
        files[name] = dict(hashes_before[f"{model_relative}/{name}"])
    validate_safetensors_file(model_dir / "model.safetensors")
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "training_config.json",
        "training_report.json",
    ):
        load_secure_json_object(model_dir / name, label=f"required model JSON {name}")
    model_config = load_secure_json_object(model_dir / "config.json", label="model config")
    architectures = model_config.get("architectures")
    if (
        model_config.get("model_type") != "xlm-roberta"
        or not isinstance(architectures, list)
        or "XLMRobertaForSequenceClassification" not in architectures
    ):
        raise SelectedCheckpointError("Downloaded model config has another inference graph")
    training_config = load_secure_json_object(
        model_dir / "training_config.json", label="downloaded training config"
    )
    if (
        training_config.get("max_length") != INFERENCE_MAX_LENGTH
        or training_config.get("symmetric_validation") is not True
    ):
        raise SelectedCheckpointError(
            "Downloaded training config changed max_length/symmetric validation"
        )
    if _lock_bound_kernel_ref(
        origin=origin,
        completion_path=directory / "notebook_completed.json",
    ) != selected.kaggle_kernel_ref:
        raise SelectedCheckpointError("Downloaded completion Kaggle owner/ref differs")
    bound_iid_path = _resolve_file(
        Path(str(origin["iid_predictions_artifact_path"])),
        label="bound IID predictions",
    )
    if secure_file_sha256(
        bound_iid_path, label="bound IID predictions"
    ) != str(origin["iid_predictions_sha256"]):
        raise SelectedCheckpointError("Bound IID predictions SHA differs from origin")
    replay = verify_full_iid_replay(
        model_dir=model_dir,
        downloaded_iid_path=model_dir / "iid_validation_predictions.parquet",
        bound_iid_path=bound_iid_path,
        bound_iid_sha256=str(origin["iid_predictions_sha256"]),
    )
    tree_after = snapshot_regular_tree(directory, label="downloaded output")
    if tree_after != tree_before:
        raise SelectedCheckpointError("Downloaded output changed during validation")
    hashes_after = hash_captured_regular_files(
        directory, tree_after, label="downloaded output after IID replay"
    )
    if hashes_after != hashes_before:
        raise SelectedCheckpointError(
            "Downloaded output bytes changed during validation/IID replay"
        )
    tree_sha256 = output_tree_sha256(hashes_after)
    return {
        "status": "validated",
        "run_id": selected.run_id,
        "experiment": selected.experiment,
        "model_dir": str(model_dir),
        "required_files": files,
        "iid_replay": replay,
        "output_tree": {
            "sha256": tree_sha256,
            "files": hashes_after,
        },
    }


CommandRunner = Callable[[list[str]], Any]


def _run_kaggle_command(command: list[str]) -> Any:
    return kaggle.run_command(command, check=False)


def _inode_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def ensure_directory_path(path: Path, *, label: str) -> Path:
    """Create missing directory components while rejecting every symlink alias."""
    if not path.is_absolute() or Path(os.path.normpath(str(path))) != path:
        raise SelectedCheckpointError(f"{label} must be an absolute normalized path")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            try:
                os.mkdir(current, 0o755)
            except FileExistsError:
                pass
            metadata = os.lstat(current)
        except OSError as error:
            raise SelectedCheckpointError(
                f"Could not inspect/create {label}: {current}: {error}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise SelectedCheckpointError(
                f"{label} has a symlink/non-directory component: {current}"
            )
    return assert_no_symlink_components(path, label=label)


def _assert_materialization_root(selected: SelectedCheckpoint) -> Path:
    expected = SELECTED_CHECKPOINTS_ROOT / selected.kernel_slug
    observed = Path(selected.materialization_root)
    if observed != expected or not observed.is_absolute():
        raise SelectedCheckpointError(
            "Selected materialization root differs from the fixed content store"
        )
    return observed


def _one_level_relative_directory(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or value != value.strip():
        raise SelectedCheckpointError(
            f"{label} must be one normalized relative directory name"
        )
    text = _require_text(value, label=label)
    relative = Path(text)
    if (
        relative.is_absolute()
        or len(relative.parts) != 1
        or relative.parts[0] in {".", ".."}
        or str(relative) != text
    ):
        raise SelectedCheckpointError(
            f"{label} must be one normalized relative directory name"
        )
    return relative


def _manifest_integrity(payload: Mapping[str, Any]) -> str:
    stored = _require_sha(
        payload.get("manifest_payload_sha256"),
        label="selected-checkpoint manifest payload SHA",
    )
    unhashed = dict(payload)
    unhashed.pop("manifest_payload_sha256", None)
    if adaptive.canonical_sha256(unhashed) != stored:
        raise SelectedCheckpointError("Selected-checkpoint manifest payload SHA differs")
    return stored


def selected_checkpoint_manifest(
    *,
    selected: SelectedCheckpoint,
    validation: Mapping[str, Any],
    destination: Path,
) -> dict[str, Any]:
    output_tree = validation.get("output_tree")
    required_files = validation.get("required_files")
    replay = validation.get("iid_replay")
    tree_files = output_tree.get("files") if isinstance(output_tree, Mapping) else None
    if (
        not isinstance(output_tree, Mapping)
        or not isinstance(tree_files, Mapping)
        or not isinstance(required_files, Mapping)
        or set(required_files) != set(REQUIRED_MODEL_FILES)
        or not isinstance(replay, Mapping)
    ):
        raise SelectedCheckpointError(
            "Validated checkpoint lacks exact tree/model/IID evidence"
        )
    tree_sha = _require_sha(
        output_tree.get("sha256"), label="validated output tree SHA"
    )
    if output_tree_sha256(tree_files) != tree_sha:
        raise SelectedCheckpointError("Validated output tree SHA differs from its files")
    if destination != Path(selected.materialization_root) / tree_sha:
        raise SelectedCheckpointError(
            "Content-addressed destination differs from validated tree SHA"
        )
    model_dir = Path(str(validation.get("model_dir", "")))
    try:
        model_relative = model_dir.relative_to(destination)
    except ValueError as error:
        raise SelectedCheckpointError(
            "Validated model directory is outside the content-addressed tree"
        ) from error
    model_relative = _one_level_relative_directory(
        str(model_relative), label="validated model relative path"
    )
    for name in REQUIRED_MODEL_FILES:
        metadata = required_files[name]
        tree_metadata = tree_files.get(str(model_relative / name))
        if (
            not isinstance(metadata, Mapping)
            or metadata != tree_metadata
            or not isinstance(metadata.get("bytes"), int)
            or metadata["bytes"] <= 0
        ):
            raise SelectedCheckpointError(
                f"Required model file manifest binding differs: {name}"
            )
        _require_sha(metadata.get("sha256"), label=f"required model file {name} SHA")
    replay_max_difference = replay.get("max_absolute_difference")
    if (
        replay.get("pairs") != IID_EXPECTED_ROWS
        or replay.get("orientations") != IID_EXPECTED_ROWS * 2
        or replay.get("max_length") != INFERENCE_MAX_LENGTH
        or replay.get("absolute_tolerance") != SCORE_ABSOLUTE_TOLERANCE
        or replay.get("fallback_pairs") != 0
        or not isinstance(replay_max_difference, (int, float))
        or not math.isfinite(float(replay_max_difference))
        or not 0 <= float(replay_max_difference) <= SCORE_ABSOLUTE_TOLERANCE
    ):
        raise SelectedCheckpointError("Validated IID replay evidence is inconsistent")
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "campaign": selected.campaign,
        "confirmation_lock": {
            "path": selected.confirmation_lock_path,
            "payload_sha256": selected.confirmation_lock_payload_sha256,
        },
        "confirmation_summary": {
            "path": selected.confirmation_summary_path,
            "payload_sha256": selected.confirmation_summary_payload_sha256,
        },
        "selected_recipe_group_id": selected.selected_recipe_group_id,
        "recipe_family_sha256": selected.recipe_family_sha256,
        "origin": {
            "origin_id": selected.origin_id,
            "experiment": selected.experiment,
            "run_id": selected.run_id,
            "seed": selected.seed,
            "recipe_sha256": selected.recipe_sha256,
            "loss_variant": selected.loss_variant,
            "loss_hook_sha256": selected.loss_hook_sha256,
            "code_bundle_sha256": selected.code_bundle_sha256,
        },
        "source_authority": {
            "kind": selected.source_authority_kind,
            "lock_path": selected.source_lock_path,
            "lock_payload_sha256": selected.source_lock_payload_sha256,
        },
        "kaggle_authority": {
            "owner": CAMPAIGN_KAGGLE_OWNER,
            "kernel_ref": selected.kaggle_kernel_ref,
            "authenticated_platform_trusted": True,
            "remote_kernel_metadata_prepinned": False,
            "model_sha256_prepinned_before_download": False,
            "identity_basis": (
                "fixed-owner Kaggle status/output plus lock-bound completion/config/IID "
                "provenance and complete behavioral IID replay"
            ),
        },
        "local_storage_assumption": {
            "publish": "atomic_no_replace",
            "file_mode": "0444",
            "directory_mode": "0555",
            "os_immutable_flag_asserted": False,
            "same_uid_concurrent_mutation_after_verification_out_of_scope": True,
        },
        "output_tree": {
            "directory": str(destination),
            "sha256": tree_sha,
            "file_count": len(tree_files),
            "model_relative_path": str(model_relative),
        },
        "required_model_files": {
            name: dict(required_files[name]) for name in REQUIRED_MODEL_FILES
        },
        "iid_replay": dict(replay),
    }
    manifest["manifest_payload_sha256"] = adaptive.canonical_sha256(manifest)
    return manifest


def _canonical_manifest_bytes(payload: Mapping[str, Any]) -> bytes:
    _manifest_integrity(payload)
    return (adaptive.canonical_json_dumps(payload) + "\n").encode("utf-8")


def _write_or_validate_manifest(
    path: Path,
    payload: Mapping[str, Any],
    *,
    parent_descriptor: int | None = None,
) -> str:
    """No-replace publish, freeze the parent, then verify one held manifest fd."""
    expected = _canonical_manifest_bytes(payload)
    parent = assert_no_symlink_components(
        path.parent, label="selected-checkpoint manifest parent"
    )
    output_tree = payload.get("output_tree")
    tree_sha = (
        _require_sha(
            output_tree.get("sha256"), label="selected-checkpoint output-tree SHA"
        )
        if isinstance(output_tree, Mapping)
        else ""
    )
    if path.parent != parent or path.name != tree_sha + MANIFEST_SUFFIX:
        raise SelectedCheckpointError("Selected-checkpoint manifest path differs")

    owns_parent_descriptor = parent_descriptor is None
    directory_descriptor = (
        _open_directory_fd(parent, label="selected-checkpoint manifest parent")
        if parent_descriptor is None
        else parent_descriptor
    )
    parent_before = os.fstat(directory_descriptor)
    if (
        not stat.S_ISDIR(parent_before.st_mode)
        or _inode_identity(parent_before) != _inode_identity(os.lstat(parent))
    ):
        if owns_parent_descriptor:
            os.close(directory_descriptor)
        raise SelectedCheckpointError("Manifest parent descriptor/path differs")

    descriptor: int | None = None
    temporary_name: str | None = None
    temporary_identity: tuple[int, int] | None = None
    try:
        try:
            descriptor = os.open(
                path.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptor,
            )
        except FileNotFoundError:
            for _ in range(128):
                candidate = f".manifest-write-{secrets.token_hex(16)}.tmp"
                try:
                    descriptor = os.open(
                        candidate,
                        os.O_RDWR
                        | os.O_CREAT
                        | os.O_EXCL
                        | getattr(os, "O_NOFOLLOW", 0),
                        0o600,
                        dir_fd=directory_descriptor,
                    )
                except FileExistsError:  # pragma: no cover - random collision.
                    continue
                temporary_name = candidate
                temporary_identity = _inode_identity(os.fstat(descriptor))
                break
            if descriptor is None or temporary_name is None:
                raise SelectedCheckpointError(
                    "Could not allocate a private manifest temporary file"
                )
            offset = 0
            while offset < len(expected):
                written = os.write(descriptor, expected[offset:])
                if written <= 0:
                    raise SelectedCheckpointError(
                        "Could not finish selected-checkpoint manifest write"
                    )
                offset += written
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
            private_bytes, _ = _read_regular_fd(
                descriptor, label="private selected-checkpoint manifest"
            )
            if private_bytes != expected:
                raise SelectedCheckpointError(
                    "Private selected-checkpoint manifest bytes differ"
                )
            published = _atomic_rename_no_replace_at(
                source_parent_descriptor=directory_descriptor,
                source_name=temporary_name,
                destination_parent_descriptor=directory_descriptor,
                destination_name=path.name,
            )
            if published:
                temporary_name = None
                temporary_identity = None
            else:
                _remove_owned_regular_at(
                    parent_descriptor=directory_descriptor,
                    name=temporary_name,
                    expected_identity=temporary_identity,
                )
                temporary_name = None
                temporary_identity = None
                os.close(descriptor)
                descriptor = os.open(
                    path.name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_descriptor,
                )

        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o444
        ):
            raise SelectedCheckpointError(
                "Existing selected-checkpoint manifest is mutable/hard-linked"
            )

        # Parent immutability closes the path-swap window before any bytes are
        # accepted.  The final dirfd lookup must still identify this held fd.
        os.fchmod(directory_descriptor, 0o555)
        os.fsync(directory_descriptor)
        frozen_parent = os.fstat(directory_descriptor)
        parent_by_path = os.lstat(parent)
        if (
            not stat.S_ISDIR(frozen_parent.st_mode)
            or stat.S_IMODE(frozen_parent.st_mode) != 0o555
            or _inode_identity(frozen_parent) != _inode_identity(parent_by_path)
        ):
            raise SelectedCheckpointError(
                "Committed manifest parent did not freeze on the held inode"
            )

        observed, first_read_snapshot = _read_regular_fd(
            descriptor, label="selected-checkpoint manifest"
        )
        first_descriptor_metadata = os.fstat(descriptor)
        first_entry_metadata = os.stat(
            path.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        first_parent_metadata = os.fstat(directory_descriptor)
        first_parent_by_path = os.lstat(parent)
        if (
            _regular_file_snapshot(first_descriptor_metadata)
            != first_read_snapshot
            or _regular_file_snapshot(first_entry_metadata) != first_read_snapshot
            or stat.S_IMODE(first_descriptor_metadata.st_mode) != 0o444
            or stat.S_IMODE(first_parent_metadata.st_mode) != 0o555
            or _inode_identity(first_parent_metadata)
            != _inode_identity(first_parent_by_path)
        ):
            raise SelectedCheckpointError(
                "Selected-checkpoint manifest changed after its first fd read"
            )
        final_observed, final_read_snapshot = _read_regular_fd(
            descriptor, label="selected-checkpoint manifest final verification"
        )
        final_descriptor_metadata = os.fstat(descriptor)
        final_entry_metadata = os.stat(
            path.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        final_parent_metadata = os.fstat(directory_descriptor)
        final_parent_by_path = os.lstat(parent)
        if (
            _regular_file_snapshot(final_descriptor_metadata)
            != final_read_snapshot
            or _regular_file_snapshot(final_entry_metadata) != final_read_snapshot
            or first_read_snapshot != final_read_snapshot
            or stat.S_IMODE(final_descriptor_metadata.st_mode) != 0o444
            or stat.S_IMODE(final_parent_metadata.st_mode) != 0o555
            or _inode_identity(final_parent_metadata)
            != _inode_identity(final_parent_by_path)
        ):
            raise SelectedCheckpointError(
                "Selected-checkpoint manifest changed during final fd verification"
            )
        if observed != expected or final_observed != expected:
            raise SelectedCheckpointError(
                "Existing selected-checkpoint manifest differs; refusing overwrite"
            )
        return hashlib.sha256(final_observed).hexdigest()
    finally:
        if temporary_name is not None and temporary_identity is not None:
            _remove_owned_regular_at(
                parent_descriptor=directory_descriptor,
                name=temporary_name,
                expected_identity=temporary_identity,
            )
        if descriptor is not None:
            os.close(descriptor)
        if owns_parent_descriptor:
            os.close(directory_descriptor)


def _load_canonical_manifest(path: Path) -> dict[str, Any]:
    parent = assert_no_symlink_components(
        path.parent, label="selected-checkpoint manifest parent"
    )
    directory_descriptor = _open_directory_fd(
        parent, label="selected-checkpoint manifest parent"
    )
    descriptor: int | None = None
    try:
        parent_metadata = os.fstat(directory_descriptor)
        if stat.S_IMODE(parent_metadata.st_mode) != 0o555:
            raise SelectedCheckpointError(
                "Committed selected-checkpoint manifest parent is not mode 0555"
            )
        descriptor = os.open(
            _safe_directory_entry_name(path.name, label="selected-checkpoint manifest"),
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o444
        ):
            raise SelectedCheckpointError(
                "Selected-checkpoint manifest is mutable/hard-linked"
            )
        serialized, first_read_snapshot = _read_regular_fd(
            descriptor, label="selected-checkpoint manifest"
        )
        first_metadata = os.fstat(descriptor)
        first_entry_metadata = os.stat(
            path.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            _regular_file_snapshot(first_metadata) != first_read_snapshot
            or _regular_file_snapshot(first_entry_metadata) != first_read_snapshot
            or stat.S_IMODE(os.fstat(directory_descriptor).st_mode) != 0o555
            or _inode_identity(os.fstat(directory_descriptor))
            != _inode_identity(os.lstat(parent))
        ):
            raise SelectedCheckpointError(
                "Selected-checkpoint manifest changed after its first fd read"
            )
        final_serialized, final_read_snapshot = _read_regular_fd(
            descriptor, label="selected-checkpoint manifest final verification"
        )
        final_metadata = os.fstat(descriptor)
        final_entry_metadata = os.stat(
            path.name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            _regular_file_snapshot(final_metadata) != final_read_snapshot
            or _regular_file_snapshot(final_entry_metadata) != final_read_snapshot
            or first_read_snapshot != final_read_snapshot
            or stat.S_IMODE(os.fstat(directory_descriptor).st_mode) != 0o555
            or _inode_identity(os.fstat(directory_descriptor))
            != _inode_identity(os.lstat(parent))
        ):
            raise SelectedCheckpointError(
                "Selected-checkpoint manifest changed during final fd verification"
            )
        if final_serialized != serialized:
            raise SelectedCheckpointError(
                "Selected-checkpoint manifest bytes changed between fd reads"
            )
        serialized = final_serialized
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_descriptor)
    try:
        payload = _parse_json_object(
            serialized.decode("utf-8"),
            label="selected-checkpoint manifest",
            path=path,
        )
    except UnicodeDecodeError as error:
        raise SelectedCheckpointError(
            f"Invalid UTF-8 selected-checkpoint manifest: {path}"
        ) from error
    if serialized != _canonical_manifest_bytes(payload):
        raise SelectedCheckpointError(
            "Selected-checkpoint manifest is not exact canonical JSON"
        )
    return payload


def _manifest_matches_selected(
    payload: Mapping[str, Any], selected: SelectedCheckpoint
) -> bool:
    origin = payload.get("origin")
    kaggle_authority = payload.get("kaggle_authority")
    confirmation_lock = payload.get("confirmation_lock")
    confirmation_summary = payload.get("confirmation_summary")
    return bool(
        payload.get("schema_version") == MANIFEST_SCHEMA_VERSION
        and payload.get("kind") == MANIFEST_KIND
        and payload.get("campaign") == selected.campaign
        and isinstance(confirmation_lock, Mapping)
        and confirmation_lock.get("payload_sha256")
        == selected.confirmation_lock_payload_sha256
        and isinstance(confirmation_summary, Mapping)
        and confirmation_summary.get("payload_sha256")
        == selected.confirmation_summary_payload_sha256
        and payload.get("selected_recipe_group_id")
        == selected.selected_recipe_group_id
        and payload.get("recipe_family_sha256") == selected.recipe_family_sha256
        and isinstance(origin, Mapping)
        and origin.get("origin_id") == selected.origin_id
        and origin.get("run_id") == selected.run_id
        and isinstance(kaggle_authority, Mapping)
        and kaggle_authority.get("owner") == CAMPAIGN_KAGGLE_OWNER
        and kaggle_authority.get("kernel_ref") == selected.kaggle_kernel_ref
    )


def _assert_read_only_tree(
    root: Path, snapshot: Mapping[str, tuple[int, ...]]
) -> None:
    root_mode = stat.S_IMODE(os.lstat(root).st_mode)
    if root_mode != 0o555:
        raise SelectedCheckpointError("Content-addressed output root is not mode 0555")
    for relative, captured in snapshot.items():
        mode = stat.S_IMODE(captured[2])
        expected = 0o555 if relative.endswith("/") else 0o444
        if mode != expected:
            raise SelectedCheckpointError(
                f"Content-addressed output has mutable mode: {relative}={oct(mode)}"
            )


def _set_tree_read_only(root: Path) -> dict[str, tuple[int, ...]]:
    snapshot = snapshot_regular_tree(root, label="materialized output before chmod")
    for relative in sorted(
        (path for path in snapshot if not path.endswith("/")),
        key=lambda value: len(Path(value).parts),
        reverse=True,
    ):
        os.chmod(root / relative, 0o444, follow_symlinks=False)
    for relative in sorted(
        (path for path in snapshot if path.endswith("/")),
        key=lambda value: len(Path(value.rstrip("/")).parts),
        reverse=True,
    ):
        os.chmod(root / relative.rstrip("/"), 0o555, follow_symlinks=False)
    os.chmod(root, 0o555, follow_symlinks=False)
    result = snapshot_regular_tree(root, label="read-only materialized output")
    _assert_read_only_tree(root, result)
    return result


def _copy_validated_tree_once(
    *,
    source: Path,
    destination: Path,
    source_snapshot: Mapping[str, tuple[int, ...]],
    expected_hashes: Mapping[str, Mapping[str, Any]],
) -> None:
    """Build one private tree and remove only its exact inode on copy failure."""
    source = assert_no_symlink_components(source, label="validated staging")
    parent = assert_no_symlink_components(
        destination.parent, label="content-addressed parent"
    )
    if destination.parent != parent or destination.name not in {
        output_tree_sha256(expected_hashes)
    }:
        raise SelectedCheckpointError("Content-addressed copy destination differs")
    try:
        os.mkdir(destination, 0o700)
    except FileExistsError as error:
        raise SelectedCheckpointError(
            "Content-addressed destination already exists; refusing replacement"
        ) from error
    destination_identity = _inode_identity(os.lstat(destination))
    try:
        for relative in sorted(
            (path for path in source_snapshot if path.endswith("/")),
            key=lambda value: len(Path(value.rstrip("/")).parts),
        ):
            os.mkdir(destination / relative.rstrip("/"), 0o700)
        for relative, expected in sorted(expected_hashes.items()):
            source_path = _contained_file(
                source, Path(relative), label=f"validated staging file {relative}"
            )
            if (
                _regular_file_snapshot(os.lstat(source_path))
                != source_snapshot[relative]
            ):
                raise SelectedCheckpointError(
                    f"Validated staging changed before content copy: {relative}"
                )
            source_descriptor = os.open(
                source_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                destination_descriptor = os.open(
                    destination / relative,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                )
                digest = hashlib.sha256()
                copied = 0
                try:
                    opened_source = os.fstat(source_descriptor)
                    if (
                        _regular_file_snapshot(opened_source)
                        != source_snapshot[relative]
                    ):
                        raise SelectedCheckpointError(
                            "Validated staging inode changed while copying: "
                            + relative
                        )
                    while True:
                        block = os.read(source_descriptor, 1024 * 1024)
                        if not block:
                            break
                        digest.update(block)
                        copied += len(block)
                        offset = 0
                        while offset < len(block):
                            written = os.write(
                                destination_descriptor, block[offset:]
                            )
                            if written <= 0:
                                raise SelectedCheckpointError(
                                    f"Short content-addressed write: {relative}"
                                )
                            offset += written
                    os.fsync(destination_descriptor)
                    if (
                        _regular_file_snapshot(os.fstat(source_descriptor))
                        != source_snapshot[relative]
                    ):
                        raise SelectedCheckpointError(
                            f"Validated staging changed during content copy: {relative}"
                        )
                finally:
                    os.close(destination_descriptor)
            finally:
                os.close(source_descriptor)
            if (
                copied != int(expected["bytes"])
                or digest.hexdigest() != expected["sha256"]
            ):
                raise SelectedCheckpointError(
                    f"Content-addressed copy differs from validated bytes: {relative}"
                )
    except BaseException:
        _remove_private_staging(
            destination, expected_identity=destination_identity
        )
        raise


def _remove_private_staging(
    staging: Path, *, expected_identity: tuple[int, int]
) -> None:
    """Remove only the exact temporary directory inode created by this process."""
    try:
        metadata = os.lstat(staging)
    except FileNotFoundError:
        return
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or _inode_identity(metadata) != expected_identity
    ):
        raise SelectedCheckpointError(
            "Refusing to clean a staging path whose inode/type changed"
        )
    snapshot = snapshot_regular_tree(staging, label="private staging cleanup")
    for relative in sorted(
        (path for path in snapshot if path.endswith("/")),
        key=lambda value: len(Path(value.rstrip("/")).parts),
    ):
        os.chmod(staging / relative.rstrip("/"), 0o700, follow_symlinks=False)
    os.chmod(staging, 0o700, follow_symlinks=False)
    shutil.rmtree(staging)


def _validate_materialized_tree(
    *,
    destination: Path,
    selected: SelectedCheckpoint,
    origin: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_downloaded_checkpoint(
        destination, selected=selected, origin=origin, entry=entry
    )
    output_tree = validation["output_tree"]
    if destination.name != output_tree["sha256"]:
        raise SelectedCheckpointError(
            "Existing content-addressed directory name differs from its bytes"
        )
    snapshot = snapshot_regular_tree(destination, label="materialized output")
    _assert_read_only_tree(destination, snapshot)
    return validation


def _existing_materialization(
    *,
    selected: SelectedCheckpoint,
    origin: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> dict[str, Any] | None:
    root = _assert_materialization_root(selected)
    try:
        metadata = os.lstat(root)
    except FileNotFoundError:
        existing_parent = root.parent
        while True:
            try:
                os.lstat(existing_parent)
                break
            except FileNotFoundError:
                if existing_parent == Path(existing_parent.anchor):
                    raise SelectedCheckpointError(
                        "Could not find an existing materialization ancestor"
                    )
                existing_parent = existing_parent.parent
        assert_no_symlink_components(
            existing_parent, label="materialization ancestor"
        )
        return None
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise SelectedCheckpointError(
            "Materialization root is a symlink/non-directory"
        )
    root = assert_no_symlink_components(root, label="materialization root")
    if stat.S_IMODE(os.lstat(root).st_mode) != 0o555:
        # A previous read/download may have created this uncommitted root.  Do
        # not trust anything in it; the remote output will be re-downloaded and
        # every no-replace collision validated before a later commit.
        return None
    manifests: list[tuple[Path, dict[str, Any]]] = []
    with os.scandir(root) as entries:
        for directory_entry in entries:
            if not directory_entry.name.endswith(MANIFEST_SUFFIX):
                continue
            path = root / directory_entry.name
            payload = _load_canonical_manifest(path)
            if _manifest_matches_selected(payload, selected):
                manifests.append((path, payload))
    if not manifests:
        raise SelectedCheckpointError(
            "Frozen materialization root has no matching commit manifest"
        )
    if len(manifests) != 1:
        raise SelectedCheckpointError(
            "Multiple manifests claim the same selected checkpoint"
        )
    manifest_path, recorded = manifests[0]
    recorded_output_tree = recorded.get("output_tree")
    if not isinstance(recorded_output_tree, Mapping):
        raise SelectedCheckpointError("Existing manifest output_tree is malformed")
    tree_sha = _require_sha(
        recorded_output_tree.get("sha256"),
        label="existing materialized tree SHA",
    )
    model_relative = _one_level_relative_directory(
        recorded_output_tree.get("model_relative_path"),
        label="existing manifest model relative path",
    )
    destination = root / tree_sha
    validation = _validate_materialized_tree(
        destination=destination,
        selected=selected,
        origin=origin,
        entry=entry,
    )
    validation["model_dir"] = str(destination / model_relative)
    expected = selected_checkpoint_manifest(
        selected=selected, validation=validation, destination=destination
    )
    if recorded != expected:
        raise SelectedCheckpointError(
            "Existing manifest differs from fully revalidated checkpoint"
        )
    return {
        **validation,
        "status": "already_materialized_and_revalidated",
        "kernel_ref": selected.kaggle_kernel_ref,
        "destination": str(destination),
        "manifest": str(manifest_path),
        "manifest_file_sha256": hashlib.sha256(
            _canonical_manifest_bytes(recorded)
        ).hexdigest(),
        "remote_operations": [],
    }


def download_selected_checkpoint(
    *,
    selected: SelectedCheckpoint,
    origin: Mapping[str, Any],
    entry: Mapping[str, Any],
    cli: list[str],
    command_runner: CommandRunner = _run_kaggle_command,
) -> dict[str, Any]:
    """Materialize one existing COMPLETE output without replacing any old tree."""
    kernel_ref = selected.kaggle_kernel_ref
    expected_ref = f"{CAMPAIGN_KAGGLE_OWNER}/{selected.kernel_slug}"
    if (
        re.fullmatch(r"[a-z0-9][a-z0-9-]*", selected.kernel_slug) is None
        or kernel_ref != expected_ref
    ):
        raise SelectedCheckpointError(
            "Selected Kaggle owner/ref differs from the fixed campaign authority"
        )
    bound_output = assert_no_symlink_components(
        Path(selected.bound_output_directory), label="bound selected output"
    )
    if bound_output.name != selected.kernel_slug or not bound_output.is_dir():
        raise SelectedCheckpointError("Bound selected output/kernel slug differs")
    if _lock_bound_kernel_ref(
        origin=origin,
        completion_path=bound_output / "notebook_completed.json",
    ) != kernel_ref:
        raise SelectedCheckpointError("Bound completion owner/ref changed")

    existing = _existing_materialization(
        selected=selected, origin=origin, entry=entry
    )
    if existing is not None:
        return existing
    if command_runner is _run_kaggle_command and not os.getenv(
        "KAGGLE_API_TOKEN", ""
    ).strip():
        raise SelectedCheckpointError(
            "KAGGLE_API_TOKEN is required when remote read-only download is needed"
        )

    materialization_root = ensure_directory_path(
        _assert_materialization_root(selected), label="materialization root"
    )
    parent_identity = _inode_identity(os.lstat(materialization_root))

    status_result = command_runner(cli + ["kernels", "status", kernel_ref])
    status = (
        kaggle.extract_status(status_result.stdout)
        if getattr(status_result, "returncode", 1) == 0
        else None
    )
    if status not in kaggle.TERMINAL_SUCCESS:
        raise SelectedCheckpointError(
            f"Selected remote kernel is not terminal COMPLETE: {kernel_ref} ({status!r})"
        )

    if _inode_identity(
        os.lstat(
            assert_no_symlink_components(
                materialization_root, label="materialization root"
            )
        )
    ) != parent_identity:
        raise SelectedCheckpointError("Materialization root changed during status")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{selected.kernel_slug}.full-download-",
            dir=materialization_root.parent,
        )
    )
    os.chmod(staging, 0o700)
    staging = assert_no_symlink_components(staging, label="private download staging")
    staging_identity = _inode_identity(os.lstat(staging))
    try:
        materialization_descriptor = _open_directory_fd(
            materialization_root, label="materialization root"
        )
    except BaseException:
        _remove_private_staging(staging, expected_identity=staging_identity)
        raise
    if _inode_identity(os.fstat(materialization_descriptor)) != parent_identity:
        os.close(materialization_descriptor)
        _remove_private_staging(staging, expected_identity=staging_identity)
        raise SelectedCheckpointError(
            "Materialization root changed before full-output download"
        )
    try:
        output_result = command_runner(
            cli
            + [
                "kernels",
                "output",
                kernel_ref,
                "-p",
                str(staging),
                "--force",
                "--page-size",
                "200",
            ]
        )
        if getattr(output_result, "returncode", 1) != 0:
            raise SelectedCheckpointError(
                f"Kaggle full-output download failed for {kernel_ref}"
            )
        staging = assert_no_symlink_components(
            staging, label="private download staging"
        )
        if _inode_identity(os.lstat(staging)) != staging_identity:
            raise SelectedCheckpointError(
                "Download staging inode changed while Kaggle wrote output"
            )
        validation = validate_downloaded_checkpoint(
            staging,
            selected=selected,
            origin=origin,
            entry=entry,
        )
        if _inode_identity(
            os.lstat(
                assert_no_symlink_components(
                    materialization_root, label="materialization root"
                )
            )
        ) != parent_identity:
            raise SelectedCheckpointError("Materialization root changed during download")
        if _lock_bound_kernel_ref(
            origin=origin,
            completion_path=bound_output / "notebook_completed.json",
        ) != kernel_ref:
            raise SelectedCheckpointError("Bound completion changed during download")
        source_snapshot = snapshot_regular_tree(staging, label="validated staging")
        source_hashes = hash_captured_regular_files(
            staging, source_snapshot, label="validated staging after replay"
        )
        output_tree = validation.get("output_tree")
        if (
            not isinstance(output_tree, Mapping)
            or source_hashes != output_tree.get("files")
            or output_tree_sha256(source_hashes) != output_tree.get("sha256")
        ):
            raise SelectedCheckpointError(
                "Validated staging bytes changed after IID replay"
            )
        tree_sha = _require_sha(
            output_tree["sha256"], label="validated output tree SHA"
        )
        destination = materialization_root / tree_sha
        try:
            validated_model_relative = Path(
                str(validation["model_dir"])
            ).relative_to(staging)
        except (KeyError, TypeError, ValueError) as error:
            raise SelectedCheckpointError(
                "Validated model directory is not contained in staging"
            ) from error
        validated_model_relative = _one_level_relative_directory(
            str(validated_model_relative),
            label="validated staging model relative path",
        )

        install_holder = Path(
            tempfile.mkdtemp(
                prefix=f".{selected.kernel_slug}.{tree_sha}.install-",
                dir=materialization_root.parent,
            )
        )
        os.chmod(install_holder, 0o700)
        install_holder = assert_no_symlink_components(
            install_holder, label="private content install holder"
        )
        install_holder_identity = _inode_identity(os.lstat(install_holder))
        install_holder_descriptor: int | None = None
        temporary_content_descriptor: int | None = None
        content_descriptor: int | None = None
        try:
            private_tree = install_holder / tree_sha
            _copy_validated_tree_once(
                source=staging,
                destination=private_tree,
                source_snapshot=source_snapshot,
                expected_hashes=source_hashes,
            )
            private_snapshot = snapshot_regular_tree(
                private_tree, label="private content tree after copy"
            )
            private_hashes = hash_captured_regular_files(
                private_tree,
                private_snapshot,
                label="private content tree after copy",
            )
            if private_hashes != source_hashes:
                raise SelectedCheckpointError(
                    "Private content tree differs after installation"
                )
            private_snapshot = _set_tree_read_only(private_tree)
            frozen_private_hashes = hash_captured_regular_files(
                private_tree,
                private_snapshot,
                label="frozen private content tree",
            )
            if frozen_private_hashes != source_hashes:
                raise SelectedCheckpointError(
                    "Private content tree bytes changed while freezing"
                )
            # macOS RENAME_EXCL refuses to move a mode-0555 directory.  Keep
            # only this held private root writable for the rename itself; all
            # descendants are already frozen, and the published root is
            # fchmod'ed back through its held fd before any manifest commit.
            os.chmod(private_tree, 0o700, follow_symlinks=False)

            install_holder_descriptor = _open_directory_fd(
                install_holder, label="private content install holder"
            )
            temporary_content_descriptor = _open_directory_entry_at(
                install_holder_descriptor,
                tree_sha,
                label="private content tree",
            )
            published = _atomic_rename_no_replace_at(
                source_parent_descriptor=install_holder_descriptor,
                source_name=tree_sha,
                destination_parent_descriptor=materialization_descriptor,
                destination_name=tree_sha,
            )
            if published:
                content_descriptor = temporary_content_descriptor
                temporary_content_descriptor = None
                os.fchmod(content_descriptor, 0o555)
                os.fsync(content_descriptor)
                _assert_directory_entry_matches_fd(
                    parent_descriptor=materialization_descriptor,
                    name=tree_sha,
                    descriptor=content_descriptor,
                    label="published content-addressed tree",
                )
                materialization_status = "downloaded_validated_and_materialized"
            else:
                os.close(temporary_content_descriptor)
                temporary_content_descriptor = None
                existing_validation = _validate_materialized_tree(
                    destination=destination,
                    selected=selected,
                    origin=origin,
                    entry=entry,
                )
                if existing_validation["output_tree"]["files"] != source_hashes:
                    raise SelectedCheckpointError(
                        "Existing content address differs; refusing replacement"
                    )
                validation = existing_validation
                content_descriptor = _open_directory_entry_at(
                    materialization_descriptor,
                    tree_sha,
                    label="existing content-addressed tree",
                )
                materialization_status = "existing_content_address_revalidated"

            validation["model_dir"] = str(destination / validated_model_relative)
            manifest_payload = selected_checkpoint_manifest(
                selected=selected,
                validation=validation,
                destination=destination,
            )
            manifest_path = materialization_root / f"{tree_sha}{MANIFEST_SUFFIX}"
            manifest_file_sha = _write_or_validate_manifest(
                manifest_path,
                manifest_payload,
                parent_descriptor=materialization_descriptor,
            )
            _assert_directory_entry_matches_fd(
                parent_descriptor=materialization_descriptor,
                name=tree_sha,
                descriptor=content_descriptor,
                label="committed content-addressed tree",
            )
            committed_snapshot = snapshot_regular_tree(
                destination, label="committed content-addressed tree"
            )
            _assert_read_only_tree(destination, committed_snapshot)
            committed_hashes = hash_captured_regular_files(
                destination,
                committed_snapshot,
                label="committed content-addressed tree",
            )
            if committed_hashes != source_hashes:
                raise SelectedCheckpointError(
                    "Committed content-addressed output bytes changed"
                )
        finally:
            if temporary_content_descriptor is not None:
                os.close(temporary_content_descriptor)
            if content_descriptor is not None:
                os.close(content_descriptor)
            if install_holder_descriptor is not None:
                os.close(install_holder_descriptor)
            _remove_private_staging(
                install_holder, expected_identity=install_holder_identity
            )
        validation.update(
            {
                "status": materialization_status,
                "kernel_ref": kernel_ref,
                "destination": str(destination),
                "manifest": str(manifest_path),
                "manifest_file_sha256": manifest_file_sha,
                "remote_operations": ["kernels status", "kernels output"],
            }
        )
        return validation
    finally:
        try:
            _remove_private_staging(staging, expected_identity=staging_identity)
        finally:
            os.close(materialization_descriptor)


def _absolute_from_root(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def plan_payload(selected: SelectedCheckpoint) -> dict[str, Any]:
    return {
        "status": "plan_only",
        "network_actions": 0,
        "selected_checkpoint": asdict(selected),
        "required_model_files": list(REQUIRED_MODEL_FILES),
        "download_contract": {
            "remote_status_must_be": sorted(kaggle.TERMINAL_SUCCESS),
            "remote_operations": ["kernels status", "kernels output"],
            "full_output_only": True,
            "submit_or_resubmit": False,
            "google_sheets_repair": False,
            "fixed_kaggle_owner": CAMPAIGN_KAGGLE_OWNER,
            "remote_kernel_metadata_prepinned": False,
            "trust_boundary": (
                "authenticated fixed-owner Kaggle output plus lock-bound output "
                "provenance and complete IID behavioral replay"
            ),
            "destination_policy": (
                "private O_EXCL build plus atomic no-replace content-address publish; "
                "never replace slim output"
            ),
            "local_storage_assumption": (
                "0444/0555 are tamper-evident modes, not OS immutable flags; "
                "same-UID mutation after final verification is outside the trust boundary"
            ),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--confirmation-lock", type=Path, default=DEFAULT_CONFIRMATION_LOCK
    )
    parser.add_argument(
        "--confirmation-summary", type=Path, default=DEFAULT_CONFIRMATION_SUMMARY
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument(
        "--download",
        action="store_true",
        help="read remote status and full-download the one existing COMPLETE output",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        selected, origin, entry, _ = resolve_selected_checkpoint(
            plan_path=_absolute_from_root(args.plan),
            confirmation_lock_path=_absolute_from_root(args.confirmation_lock),
            confirmation_summary_path=_absolute_from_root(args.confirmation_summary),
        )
        if not args.download:
            print(json.dumps(plan_payload(selected), ensure_ascii=False, indent=2))
            return

        env_file = _absolute_from_root(args.env_file)
        kaggle.load_dotenv(env_file)
        result = download_selected_checkpoint(
            selected=selected,
            origin=origin,
            entry=entry,
            cli=kaggle.kaggle_command(),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (
        SelectedCheckpointError,
        builder.CampaignConfigError,
        adaptive.AdaptiveMaterializationError,
        OSError,
    ) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
