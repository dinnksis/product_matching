#!/usr/bin/env python3
"""Materialize immutable loss-screen and confirmation locks for MiniLM-5ep SFT.

This module is deliberately separate from the axis-stage materializer.  It
freezes the adaptive part of the campaign into schema-v2 locks without making
the existing notebook generator, launcher, or summarizer understand those
locks yet.

The four modes are:

* ``loss_primary``: four declared non-BCE losses at the tuned BCE recipe;
* ``loss_overlay``: the optional best-balance x focal combination;
* ``loss_lr_refine``: the optional 0.5x/2x LR line for the best non-BCE loss;
* ``confirmation``: seeds 17/2026 for baseline, tuned BCE, and one/two losses.

Every output is canonical JSON with an internal SHA-256 and is created with
O_EXCL.  A valid existing output is returned without reselecting a parent.
Validation requires a trusted provenance manifest supplied independently from
the lock payload.  The manifest pins immutable summary snapshots, exact
prerequisite locks and one artifact root.  Thus a rehashed lock cannot appoint
an alternate summary or artifact tree as its own authority, while a mutable
``reports/.../summary.json`` may safely advance after a stage was frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import create_minilm_5ep_sft_hparam_notebooks as generator
import materialize_minilm_5ep_sft_hparam_stage as axis_materializer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "configs" / "minilm_5ep_sft_hparam_search_v1.json"
DEFAULT_SUMMARY = (
    ROOT / "reports" / "minilm_5ep_sft_hparam_search_v1" / "summary.json"
)
DEFAULT_ARTIFACTS_DIR = ROOT / "artifacts" / "kaggle"
DEFAULT_LOCKS_DIR = (
    ROOT / "reports" / "minilm_5ep_sft_hparam_search_v1" / "stage_locks"
)

SCHEMA_VERSION = 2
LOCK_KIND = "minilm_5ep_sft_adaptive_stage_lock"
RECEIPT_KIND = "minilm_5ep_sft_adaptive_branch_receipt"
TRUSTED_PROVENANCE_KIND = "minilm_5ep_sft_trusted_provenance_manifest"
TRUSTED_PROVENANCE_SCHEMA_VERSION = 1
MODES = ("loss_primary", "loss_overlay", "loss_lr_refine", "confirmation")
EFFECTIVE_STAGES = {
    "loss_primary": "special_loss_screen__primary",
    "loss_overlay": "special_loss_screen__overlay",
    "loss_lr_refine": "special_loss_screen__lr_refine",
    "confirmation": "confirmation__matched_seeds",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class AdaptiveMaterializationError(RuntimeError):
    """Raised when an adaptive decision cannot be frozen without guessing."""


class ExistingAdaptiveLockConflictError(AdaptiveMaterializationError):
    """Raised when an immutable adaptive lock is corrupt or has another identity."""


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
        raise AdaptiveMaterializationError(
            f"Value is not canonical finite JSON: {error}"
        ) from error


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_dumps(value).encode("utf-8")).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return axis_materializer.file_sha256(path)


def _bound_file_path(path: Path, *, label: str) -> str:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise AdaptiveMaterializationError(
            f"Bound {label} does not exist: {path}"
        ) from error
    if not resolved.is_file():
        raise AdaptiveMaterializationError(f"Bound {label} is not a file: {resolved}")
    return str(resolved)


def _bound_directory_path(path: Path, *, label: str) -> str:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise AdaptiveMaterializationError(
            f"Bound {label} does not exist: {path}"
        ) from error
    if not resolved.is_dir():
        raise AdaptiveMaterializationError(
            f"Bound {label} is not a directory: {resolved}"
        )
    return str(resolved)


def trusted_provenance_manifest_path(lock_path: Path) -> Path:
    """Return the fixed sidecar path; it is never selected by lock payload."""
    return lock_path.with_name(f"{lock_path.name}.trusted-provenance.json")


def trusted_provenance_archive_dir(lock_path: Path) -> Path:
    """Return the fixed content-addressed source archive for one lock."""
    return lock_path.with_name(f"{lock_path.name}.trusted-provenance")


def _shallow_lock_payload(path: Path) -> dict[str, Any]:
    payload = _load_json(path, label="trusted prerequisite lock")
    stored = str(payload.get("lock_payload_sha256", ""))
    unhashed = dict(payload)
    unhashed.pop("lock_payload_sha256", None)
    if SHA256_RE.fullmatch(stored) is None or stored != canonical_sha256(unhashed):
        raise AdaptiveMaterializationError(
            f"Trusted prerequisite lock has an invalid payload SHA: {path}"
        )
    if path.read_text(encoding="utf-8") != canonical_json_dumps(payload) + "\n":
        raise AdaptiveMaterializationError(
            f"Trusted prerequisite lock is not canonical JSON: {path}"
        )
    return payload


def _canonical_path_without_read(path: Path, *, label: str) -> str:
    try:
        resolved = path.resolve(strict=False)
    except OSError as error:
        raise AdaptiveMaterializationError(
            f"Could not canonicalize {label}: {path}"
        ) from error
    if not resolved.is_absolute():  # pragma: no cover - Path.resolve is absolute.
        raise AdaptiveMaterializationError(f"{label} must be absolute")
    return str(resolved)


def _write_source_snapshot_once(path: Path, document: Mapping[str, Any]) -> None:
    serialized = canonical_json_dumps(document) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != serialized:
            raise ExistingAdaptiveLockConflictError(
                f"Immutable trusted source snapshot differs: {path}"
            )
        return
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())


def build_trusted_provenance(
    *,
    plan: Mapping[str, Any],
    source_document_paths: Sequence[Path],
    prerequisite_lock_paths: Sequence[Path],
    artifacts_dir: Path,
    output_lock_path: Path,
) -> dict[str, Any]:
    """Archive caller-selected authorities independently from a schema-v2 lock."""
    archive_dir = trusted_provenance_archive_dir(output_lock_path).resolve(
        strict=False
    )
    sources: list[dict[str, Any]] = []
    seen_source_paths: set[str] = set()
    for raw_path in source_document_paths:
        path = Path(_bound_file_path(raw_path, label="trusted source summary"))
        if str(path) in seen_source_paths:
            raise AdaptiveMaterializationError(
                f"Duplicate trusted source-summary path: {path}"
            )
        seen_source_paths.add(str(path))
        document = _load_json(path, label="trusted source summary")
        document_sha = _summary_document_sha(document)
        snapshot_path = archive_dir / "sources" / f"{document_sha}.json"
        _write_source_snapshot_once(snapshot_path, document)
        sources.append(
            {
                "expected_source_path": str(path),
                "snapshot_path": str(snapshot_path),
                "document_sha256": document_sha,
                "snapshot_file_sha256": file_sha256(snapshot_path),
            }
        )
    lock_authorities = []
    seen_lock_paths: set[str] = set()
    for raw_path in prerequisite_lock_paths:
        path = Path(_bound_file_path(raw_path, label="trusted prerequisite lock"))
        if str(path) in seen_lock_paths:
            raise AdaptiveMaterializationError(
                f"Duplicate trusted prerequisite-lock path: {path}"
            )
        seen_lock_paths.add(str(path))
        payload = _shallow_lock_payload(path)
        lock_authorities.append(
            {
                "lock_path": str(path),
                "lock_payload_sha256": payload["lock_payload_sha256"],
                "lock_file_sha256": file_sha256(path),
            }
        )
    result = {
        "schema_version": TRUSTED_PROVENANCE_SCHEMA_VERSION,
        "kind": TRUSTED_PROVENANCE_KIND,
        "campaign": plan["campaign"],
        "source_plan_sha256": canonical_sha256(plan),
        "archive_dir": str(archive_dir),
        "artifacts_dir": _bound_directory_path(
            artifacts_dir, label="trusted artifacts root"
        ),
        "source_documents": sorted(
            sources,
            key=lambda row: (row["expected_source_path"], row["document_sha256"]),
        ),
        "prerequisite_locks": sorted(
            lock_authorities,
            key=lambda row: (row["lock_payload_sha256"], row["lock_path"]),
        ),
    }
    result["trusted_provenance_payload_sha256"] = canonical_sha256(result)
    return result


def validate_trusted_provenance(
    trusted: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> None:
    if not isinstance(trusted, Mapping) or set(trusted) != {
        "schema_version",
        "kind",
        "campaign",
        "source_plan_sha256",
        "archive_dir",
        "artifacts_dir",
        "source_documents",
        "prerequisite_locks",
        "trusted_provenance_payload_sha256",
    }:
        raise ExistingAdaptiveLockConflictError(
            "Trusted provenance manifest schema differs"
        )
    stored = trusted.get("trusted_provenance_payload_sha256")
    unhashed = dict(trusted)
    unhashed.pop("trusted_provenance_payload_sha256", None)
    if (
        trusted.get("schema_version") != TRUSTED_PROVENANCE_SCHEMA_VERSION
        or trusted.get("kind") != TRUSTED_PROVENANCE_KIND
        or trusted.get("campaign") != plan.get("campaign")
        or trusted.get("source_plan_sha256") != canonical_sha256(plan)
        or stored != canonical_sha256(unhashed)
    ):
        raise ExistingAdaptiveLockConflictError(
            "Trusted provenance manifest identity/hash differs"
        )
    artifacts_dir = _require_text(
        trusted.get("artifacts_dir"), label="trusted artifacts root"
    )
    if (
        _bound_directory_path(Path(artifacts_dir), label="trusted artifacts root")
        != artifacts_dir
    ):
        raise ExistingAdaptiveLockConflictError(
            "Trusted artifacts root is not canonical"
        )
    archive_dir_text = _require_text(
        trusted.get("archive_dir"), label="trusted archive root"
    )
    archive_dir = Path(archive_dir_text)
    if _canonical_path_without_read(
        archive_dir, label="trusted archive root"
    ) != archive_dir_text:
        raise ExistingAdaptiveLockConflictError(
            "Trusted archive root is not canonical"
        )
    documents = trusted.get("source_documents")
    locks = trusted.get("prerequisite_locks")
    if (
        not isinstance(documents, list)
        or not isinstance(locks, list)
        or any(not isinstance(row, Mapping) for row in documents)
        or any(not isinstance(row, Mapping) for row in locks)
    ):
        raise ExistingAdaptiveLockConflictError(
            "Trusted provenance authorities are malformed"
        )
    if documents != sorted(
        documents,
        key=lambda row: (
            row.get("expected_source_path"),
            row.get("document_sha256"),
        ),
    ) or locks != sorted(
        locks, key=lambda row: (row.get("lock_payload_sha256"), row.get("lock_path"))
    ):
        raise ExistingAdaptiveLockConflictError(
            "Trusted provenance authorities are not canonical"
        )
    seen_source_paths: set[str] = set()
    for authority in documents:
        if not isinstance(authority, Mapping) or set(authority) != {
            "expected_source_path",
            "snapshot_path",
            "document_sha256",
            "snapshot_file_sha256",
        }:
            raise ExistingAdaptiveLockConflictError(
                "Trusted source authority is malformed"
            )
        expected_path_text = _require_text(
            authority.get("expected_source_path"), label="trusted source path"
        )
        if (
            _canonical_path_without_read(
                Path(expected_path_text), label="trusted source path"
            )
            != expected_path_text
            or expected_path_text in seen_source_paths
        ):
            raise ExistingAdaptiveLockConflictError(
                "Trusted source path is not canonical/unique"
            )
        seen_source_paths.add(expected_path_text)
        document_sha = _require_sha(
            authority.get("document_sha256"), label="trusted source document SHA"
        )
        snapshot_sha = _require_sha(
            authority.get("snapshot_file_sha256"),
            label="trusted source snapshot file SHA",
        )
        snapshot_path_text = _require_text(
            authority.get("snapshot_path"), label="trusted source snapshot path"
        )
        snapshot_path = Path(snapshot_path_text)
        expected_snapshot_path = archive_dir / "sources" / f"{document_sha}.json"
        try:
            resolved_snapshot_path = snapshot_path.resolve(strict=True)
            resolved_snapshot_path.relative_to(archive_dir)
        except (OSError, ValueError) as error:
            raise ExistingAdaptiveLockConflictError(
                "Trusted source snapshot escapes/is missing from its archive"
            ) from error
        if (
            str(resolved_snapshot_path) != snapshot_path_text
            or resolved_snapshot_path != expected_snapshot_path
            or file_sha256(resolved_snapshot_path) != snapshot_sha
        ):
            raise ExistingAdaptiveLockConflictError(
                "Trusted source snapshot path/hash differs"
            )
        document = _load_json(
            resolved_snapshot_path, label="trusted source snapshot"
        )
        if (
            _summary_document_sha(document) != document_sha
            or resolved_snapshot_path.read_text(encoding="utf-8")
            != canonical_json_dumps(document) + "\n"
        ):
            raise ExistingAdaptiveLockConflictError(
                "Trusted source snapshot content/hash differs"
            )
    for authority in locks:
        if not isinstance(authority, Mapping) or set(authority) != {
            "lock_path",
            "lock_payload_sha256",
            "lock_file_sha256",
        }:
            raise ExistingAdaptiveLockConflictError(
                "Trusted prerequisite authority is malformed"
            )
        path_text = _require_text(
            authority.get("lock_path"), label="trusted prerequisite path"
        )
        path = Path(path_text)
        if _bound_file_path(path, label="trusted prerequisite lock") != path_text:
            raise ExistingAdaptiveLockConflictError(
                "Trusted prerequisite path is not canonical"
            )
        payload = _shallow_lock_payload(path)
        if (
            payload["lock_payload_sha256"]
            != authority.get("lock_payload_sha256")
            or file_sha256(path) != authority.get("lock_file_sha256")
        ):
            raise ExistingAdaptiveLockConflictError(
                "Trusted prerequisite lock hash differs"
            )


def _write_trusted_provenance_once(
    lock_path: Path,
    trusted: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> Path:
    validate_trusted_provenance(trusted, plan=plan)
    path = trusted_provenance_manifest_path(lock_path)
    serialized = canonical_json_dumps(trusted) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        existing = _load_json(path, label="trusted provenance manifest")
        if existing != dict(trusted) or path.read_text(encoding="utf-8") != serialized:
            raise ExistingAdaptiveLockConflictError(
                "Immutable trusted provenance manifest differs"
            )
        return path
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def load_trusted_provenance(
    path: Path, *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    trusted = _load_json(path, label="trusted provenance manifest")
    validate_trusted_provenance(trusted, plan=plan)
    if path.read_text(encoding="utf-8") != canonical_json_dumps(trusted) + "\n":
        raise ExistingAdaptiveLockConflictError(
            "Trusted provenance manifest is not canonical JSON"
        )
    return trusted


def _requested_authority_paths(
    paths: Sequence[Path], *, label: str
) -> list[str]:
    result = [
        _canonical_path_without_read(Path(path), label=label) for path in paths
    ]
    if len(result) != len(set(result)):
        raise AdaptiveMaterializationError(f"Duplicate requested {label} path")
    return sorted(result)


def _trusted_for_materialization(
    *,
    plan: Mapping[str, Any],
    output_path: Path,
    source_document_paths: Sequence[Path],
    prerequisite_lock_paths: Sequence[Path],
    artifacts_dir: Path,
) -> dict[str, Any]:
    """Resolve fixed authority without consulting a schema-v2 lock payload."""
    manifest_path = trusted_provenance_manifest_path(output_path)
    if manifest_path.exists():
        trusted = load_trusted_provenance(manifest_path, plan=plan)
        requested_sources = _requested_authority_paths(
            source_document_paths, label="source-summary"
        )
        frozen_sources = sorted(
            str(row["expected_source_path"])
            for row in trusted["source_documents"]
        )
        requested_locks = _requested_authority_paths(
            prerequisite_lock_paths, label="prerequisite-lock"
        )
        frozen_locks = sorted(
            str(row["lock_path"]) for row in trusted["prerequisite_locks"]
        )
        requested_artifacts = _canonical_path_without_read(
            artifacts_dir, label="artifacts root"
        )
        if (
            requested_sources != frozen_sources
            or requested_locks != frozen_locks
            or requested_artifacts != trusted["artifacts_dir"]
        ):
            raise ExistingAdaptiveLockConflictError(
                "Requested provenance authorities differ from the immutable manifest"
            )
        return trusted
    if output_path.exists():
        raise ExistingAdaptiveLockConflictError(
            "Existing adaptive lock has no fixed trusted-provenance manifest"
        )
    return build_trusted_provenance(
        plan=plan,
        source_document_paths=source_document_paths,
        prerequisite_lock_paths=prerequisite_lock_paths,
        artifacts_dir=artifacts_dir,
        output_lock_path=output_path,
    )


def _trusted_source_documents(
    trusted: Mapping[str, Any]
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    result = []
    for authority in trusted["source_documents"]:
        document = _load_json(
            Path(authority["snapshot_path"]), label="trusted source snapshot"
        )
        result.append((authority, document))
    return result


def _require_sha(value: Any, *, label: str) -> str:
    result = str(value or "").strip()
    if SHA256_RE.fullmatch(result) is None:
        raise AdaptiveMaterializationError(f"{label} is not a SHA-256")
    return result


def _require_text(value: Any, *, label: str) -> str:
    result = value.strip() if isinstance(value, str) else ""
    if not result:
        raise AdaptiveMaterializationError(f"{label} must be a non-empty string")
    return result


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdaptiveMaterializationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AdaptiveMaterializationError(f"{label} must be finite")
    return result


def _strict_delta_greater(candidate: float, anchor: float, threshold: float) -> bool:
    """Apply plan decimal thresholds without binary-float boundary drift."""
    return Decimal(str(candidate)) - Decimal(str(anchor)) > Decimal(str(threshold))


def _delta_at_most(candidate: float, anchor: float, threshold: float) -> bool:
    return Decimal(str(candidate)) - Decimal(str(anchor)) <= Decimal(str(threshold))


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        return axis_materializer.load_json_object(path, label=label)
    except axis_materializer.StageMaterializationError as error:
        raise AdaptiveMaterializationError(str(error)) from error


def load_plan(path: Path = DEFAULT_PLAN) -> dict[str, Any]:
    try:
        plan = generator.load_plan(path)
    except (OSError, ValueError, generator.CampaignConfigError) as error:
        raise AdaptiveMaterializationError(f"Campaign plan is invalid: {error}") from error
    _validate_adaptive_plan(plan)
    return plan


def _stage(plan: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    try:
        return axis_materializer.stage_by_name(plan, name)
    except axis_materializer.StageMaterializationError as error:
        raise AdaptiveMaterializationError(str(error)) from error


def _validate_adaptive_plan(plan: Mapping[str, Any]) -> None:
    protocol = plan.get("selection_protocol")
    budget = plan.get("budget")
    loss = _stage(plan, "special_loss_screen")
    confirmation = _stage(plan, "confirmation")
    if not isinstance(protocol, Mapping) or protocol.get("primary_metric") != "iid_macro_ap":
        raise AdaptiveMaterializationError("Adaptive selection must use IID macro AP")
    if protocol.get("selection_uses_hard") is not False or protocol.get(
        "selection_uses_ood"
    ) is not False:
        raise AdaptiveMaterializationError("Hard/OOD must remain diagnostic-only")
    if _finite_number(
        protocol.get("practical_tie_margin"), label="practical_tie_margin"
    ) != 0.002:
        raise AdaptiveMaterializationError("Unexpected practical tie margin")
    if not isinstance(budget, Mapping) or budget.get("maximum_total_kernels") != 37:
        raise AdaptiveMaterializationError("Campaign hard kernel cap must be 37")
    if dict(budget) != {
        "typical_total_kernels": 28,
        "maximum_total_kernels": 37,
        "typical_remaining_after_completed_control": 27,
        "counting_rule": (
            "Reused parent and seed-42 runs count once; conditional boundary, "
            "interaction, loss-LR and second-finalist runs count only when triggered."
        ),
    }:
        raise AdaptiveMaterializationError("Campaign kernel-budget contract changed")

    losses = loss.get("loss_variants")
    expected_losses = [
        "bce",
        "balanced_binary_bce",
        "balanced_category_class_sqrt_bce",
        "balanced_category_class_bce",
        "focal_bce_gamma2_scale4",
    ]
    if losses != expected_losses:
        raise AdaptiveMaterializationError(
            "special_loss_screen.loss_variants differs from the reviewed order"
        )
    if loss.get("typical_new_runs") != 4 or loss.get("maximum_new_runs") != 7:
        raise AdaptiveMaterializationError("Special-loss 4/7 run budget changed")
    combo = loss.get("conditional_combination")
    if not isinstance(combo, Mapping) or any(
        (
            combo.get("run_if")
            != "best_balance_delta_positive_and_focal_delta_positive",
            combo.get("selection_metric") != "iid_macro_ap",
            combo.get("minimum_raw_delta_vs_tuned_bce") != 0.0,
            combo.get("delta_operator") != "strictly_greater_than",
            combo.get("balance_tie_break") != "loss_variants_declaration_order",
        )
    ):
        raise AdaptiveMaterializationError("Conditional loss trigger changed")
    combinations = combo.get("variants_by_balance")
    expected_combinations = {
        "balanced_binary_bce": "balanced_binary_focal_gamma2_scale4",
        "balanced_category_class_sqrt_bce": (
            "balanced_category_class_sqrt_focal_gamma2_scale4"
        ),
        "balanced_category_class_bce": (
            "balanced_category_class_focal_gamma2_scale4"
        ),
    }
    if combinations != expected_combinations:
        raise AdaptiveMaterializationError("Conditional loss registry changed")
    if any(name not in generator.LOSS_VARIANT_SHA256 for name in [*losses, *combinations.values()]):
        raise AdaptiveMaterializationError("Plan references an unregistered loss hook")

    families = loss.get("families")
    if not isinstance(families, Mapping):
        raise AdaptiveMaterializationError("Loss hypothesis families are missing")
    primary = families.get("primary_loss_screen")
    refine = families.get("winner_lr_refinement")
    if primary != {
        "planned_candidate_hypotheses": 4,
        "reserved_conditional_combinations": 1,
        "maximum_hypotheses": 5,
    }:
        raise AdaptiveMaterializationError("Primary loss family must reserve Holm m=5")
    if refine != {
        "planned_candidate_hypotheses": 2,
        "maximum_hypotheses": 2,
    }:
        raise AdaptiveMaterializationError("Loss LR family must use Holm m=2")

    if loss.get("winner_lr_refinement_multipliers") != [0.5, 1.0, 2.0]:
        raise AdaptiveMaterializationError("Loss LR multipliers changed")
    trigger = loss.get("winner_lr_refinement_trigger")
    if not isinstance(trigger, Mapping) or trigger != {
        "loss_must_be_non_bce": True,
        "minimum_iid_delta_vs_tuned_bce": 0.002,
        "delta_operator": "strictly_greater_than",
        "boundary_extension": False,
    }:
        raise AdaptiveMaterializationError("Loss LR-refinement trigger changed")

    finalists = loss.get("confirmation_finalists")
    if not isinstance(finalists, Mapping) or finalists.get("first") != (
        "best_non_bce_after_optional_lr_refinement"
    ) or finalists.get("include_first_even_if_seed42_is_below_tuned_bce") is not True:
        raise AdaptiveMaterializationError("First loss-finalist rule changed")
    second = finalists.get("second")
    if not isinstance(second, Mapping) or second != {
        "must_have_distinct_loss_variant": True,
        "minimum_raw_iid_delta_vs_tuned_bce": 0.0,
        "delta_operator": "strictly_greater_than",
        "maximum_iid_gap_from_first": 0.002,
        "tie_break": "loss_variants_declaration_order",
    }:
        raise AdaptiveMaterializationError("Second loss-finalist rule changed")

    if confirmation.get("seeds") != [17, 42, 2026]:
        raise AdaptiveMaterializationError("Confirmation seeds changed")
    if (
        confirmation.get("typical_new_runs") != 6
        or confirmation.get("maximum_new_runs") != 8
    ):
        raise AdaptiveMaterializationError("Confirmation 6/8 run budget changed")
    if confirmation.get("seed_42_runs_are_reused") is not True:
        raise AdaptiveMaterializationError("Confirmation must reuse seed 42")
    recipes = confirmation.get("recipes")
    if not isinstance(recipes, Mapping) or recipes != {
        "baseline_recipe": "current_protocol_baseline_recipe",
        "tuned_bce_recipe": "selected_regularized_bce_recipe",
        "top_k_loss_finalists": 2,
        "second_loss_finalist_is_conditional": True,
    }:
        raise AdaptiveMaterializationError("Confirmation recipe contract changed")
    if confirmation.get("recipe_family_identity") != (
        "config_without_seed_plus_loss_variant_and_loss_hook_sha256"
    ):
        raise AdaptiveMaterializationError("Recipe-family identity changed")
    if confirmation.get("comparison") != (
        "seed_matched_baseline_recipe_vs_each_finalist"
    ):
        raise AdaptiveMaterializationError("Confirmation comparison changed")
    if confirmation.get("final_tie_break_order") != [
        "current_protocol_baseline_recipe",
        "selected_regularized_bce_recipe",
        "loss_variants_declaration_order",
    ]:
        raise AdaptiveMaterializationError("Confirmation final tie-break changed")
    if confirmation.get("require_inference_runtime_check") is not True:
        raise AdaptiveMaterializationError("Confirmation runtime check must remain required")
    acceptance = confirmation.get("acceptance")
    if not isinstance(acceptance, Mapping) or acceptance != {
        "min_mean_iid_delta": 0.002,
        "min_positive_seed_count": 2,
        "min_worst_seed_iid_delta": -0.001,
        "hard_is_diagnostic_only": True,
        "ood_is_diagnostic_only": True,
    }:
        raise AdaptiveMaterializationError("Confirmation acceptance contract changed")


def _base_config() -> dict[str, Any]:
    return _load_json(generator.BASE_CONFIG_PATH, label="base SFT config")


def _current_source_sha256() -> str:
    try:
        _, source_sha256 = generator.baseline_builder.embedded_sources()
    except Exception as error:  # pragma: no cover - delegated source bundler detail.
        raise AdaptiveMaterializationError(
            f"Could not freeze the current embedded source bundle: {error}"
        ) from error
    return _require_sha(source_sha256, label="current embedded source SHA")


def _frozen_training_contract_template() -> dict[str, Any]:
    return {
        "train_data": {
            "train_pairs": 306_669,
            "items": 711_304,
            "same_size_as_human_baseline": True,
        },
        "training_report": {
            "training_sampling": "none",
            "training_loss_weighting": "none",
            "training_subset": "all",
            "original_training_examples": 306_669,
            "training_unique_coverage_per_epoch": 1.0,
            "training_loss_weight_min": 1.0,
            "training_loss_weight_median": 1.0,
            "training_loss_weight_max": 1.0,
        },
    }


def _validate_frozen_training_contract(
    completion: Mapping[str, Any], *, experiment: str
) -> dict[str, Any]:
    expected = _frozen_training_contract_template()
    expected_train_data = expected["train_data"]
    train_data = completion.get("train_data")
    if not isinstance(train_data, Mapping) or dict(train_data) != expected_train_data:
        raise AdaptiveMaterializationError(
            f"{experiment} changed the frozen human training data"
        )
    report = completion.get("training_report")
    if not isinstance(report, Mapping):
        raise AdaptiveMaterializationError(f"{experiment} has no training report")
    expected_report = expected["training_report"]
    frozen_report = {key: report.get(key) for key in expected_report}
    if frozen_report != expected_report:
        raise AdaptiveMaterializationError(
            f"{experiment} changed frozen sampling or external sample weights"
        )
    # These fields, rather than an opaque completion hash alone, make a lock
    # independently revalidatable after the downloaded artifact is archived.
    return {
        "train_data": deepcopy(expected_train_data),
        "training_report": frozen_report,
    }


def recipe_family_sha256(
    config: Mapping[str, Any], *, loss_variant: str, loss_hook_sha256: str
) -> str:
    without_seed = deepcopy(dict(config))
    without_seed.pop("seed", None)
    return canonical_sha256(
        {
            "resolved_config_without_seed": without_seed,
            "loss_variant": loss_variant,
            "loss_hook_sha256": loss_hook_sha256,
        }
    )


def _safe_overrides(plan: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    base = _base_config()
    try:
        overrides = generator.resolved_config_overrides(base, config)
        validated = generator.variant_config(
            base,
            plan,
            {"experiment": "adaptive_materialized_recipe", "overrides": overrides},
        )
    except generator.CampaignConfigError as error:
        raise AdaptiveMaterializationError(f"Unsafe resolved config: {error}") from error
    if validated != dict(config):
        raise AdaptiveMaterializationError("Resolved config does not round-trip safely")
    return overrides


def _loss_order(plan: Mapping[str, Any]) -> list[str]:
    stage = _stage(plan, "special_loss_screen")
    losses = list(stage["loss_variants"])
    combinations = stage["conditional_combination"]["variants_by_balance"]
    losses.extend(combinations[name] for name in losses if name in combinations)
    return losses


def _loss_rank(plan: Mapping[str, Any], loss_variant: str) -> int:
    order = _loss_order(plan)
    try:
        return order.index(loss_variant)
    except ValueError as error:
        raise AdaptiveMaterializationError(
            f"Loss {loss_variant!r} has no deterministic campaign rank"
        ) from error


def _number_token(value: Any) -> str:
    number = _finite_number(value, label="identity number")
    if number == 0:
        return "0"
    text = f"{number:.12g}".lower()
    if "e" in text:
        mantissa, exponent = text.split("e", 1)
        sign = "m" if int(exponent) < 0 else "p"
        return f"{mantissa.replace('.', 'p')}{sign}{abs(int(exponent))}"
    if number.is_integer():
        return str(int(number))
    return text.replace(".", "p")


def _loss_token(loss_variant: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "-", loss_variant.lower()).strip("-")
    if not token:
        raise AdaptiveMaterializationError("Could not form loss identity token")
    return token


def _variant_identity(
    *, mode: str, loss_variant: str, config: Mapping[str, Any]
) -> tuple[str, str, str, str]:
    hook_sha = generator.LOSS_VARIANT_SHA256[loss_variant]
    recipe_sha = canonical_sha256(config)
    family_sha = recipe_family_sha256(
        config, loss_variant=loss_variant, loss_hook_sha256=hook_sha
    )
    seed = int(config["seed"])
    if mode in {"loss_primary", "loss_overlay"}:
        stem = f"minilm5_sft_loss_{_loss_token(loss_variant).replace('-', '_')}_{family_sha[:8]}_s{seed}_v1"
    elif mode == "loss_lr_refine":
        stem = (
            f"minilm5_sft_losslr_{_loss_token(loss_variant).replace('-', '_')}_"
            f"lr{_number_token(config['learning_rate'])}_{family_sha[:8]}_s{seed}_v1"
        )
    elif mode == "confirmation":
        stem = f"minilm5_sft_confirm_{family_sha[:12]}_s{seed}_v1"
    else:  # pragma: no cover - public callers are checked earlier.
        raise AdaptiveMaterializationError(f"Unknown adaptive mode {mode!r}")
    if re.fullmatch(r"[a-z0-9][a-z0-9_]*", stem) is None:
        raise AdaptiveMaterializationError(f"Unsafe experiment identity {stem!r}")
    return stem, "pm-" + stem.replace("_", "-"), recipe_sha, family_sha


def _variant(
    *,
    plan: Mapping[str, Any],
    mode: str,
    config: Mapping[str, Any],
    loss_variant: str,
    expected_source_sha256: str,
    origin_ids: Sequence[str],
    role: str = "candidate",
    is_hypothesis: bool = True,
    family_size: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if loss_variant not in generator.LOSS_VARIANT_SHA256:
        raise AdaptiveMaterializationError(f"Unregistered loss {loss_variant!r}")
    resolved = deepcopy(dict(config))
    overrides = _safe_overrides(plan, resolved)
    experiment, slug, recipe_sha, family_sha = _variant_identity(
        mode=mode, loss_variant=loss_variant, config=resolved
    )
    result: dict[str, Any] = {
        "variant_id": f"variant_{family_sha[:16]}_s{int(resolved['seed'])}",
        "experiment": experiment,
        "kernel_slug": slug,
        "title": slug,
        "role": role,
        "is_hypothesis": is_hypothesis,
        "seed": int(resolved["seed"]),
        "loss_variant": loss_variant,
        "expected_loss_hook_sha256": generator.LOSS_VARIANT_SHA256[loss_variant],
        "expected_source_sha256": _require_sha(
            expected_source_sha256, label="variant expected source SHA"
        ),
        "origin_ids": list(origin_ids),
        "overrides": overrides,
        "resolved_config": resolved,
        "expected_recipe_sha256": recipe_sha,
        "expected_recipe_family_sha256": family_sha,
    }
    if family_size is not None:
        result["hypothesis_family_size"] = family_size
    if extra:
        result.update(deepcopy(dict(extra)))
    return result


def _artifact_root(artifacts_dir: Path, row: Mapping[str, Any]) -> Path:
    try:
        return axis_materializer._artifact_root(artifacts_dir, row)
    except axis_materializer.StageMaterializationError as error:
        raise AdaptiveMaterializationError(str(error)) from error


def _exactly_one(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise AdaptiveMaterializationError(
            f"Expected exactly one {name!r} below {root}, found {len(matches)}"
        )
    return matches[0]


def _summary_document_sha(summary: Mapping[str, Any]) -> str:
    stored = summary.get("summary_payload_sha256")
    if stored is None:
        return canonical_sha256(summary)
    unhashed = dict(summary)
    unhashed.pop("summary_payload_sha256", None)
    expected = canonical_sha256(unhashed)
    if stored != expected:
        raise AdaptiveMaterializationError("Summary payload SHA-256 is invalid")
    return str(stored)


def _completion_notes_from_locks(
    row: Mapping[str, Any],
    locks: Sequence[Mapping[str, Any]],
) -> str | None:
    experiment = str(row.get("experiment", ""))
    run_id = str(row.get("run_id", ""))
    candidates: set[str] = set()
    expected_row_notes = row.get("expected_notes")
    if isinstance(expected_row_notes, str):
        candidates.add(expected_row_notes)
    for lock in locks:
        if lock.get("schema_version") == 1:
            for key in ("parent", "extension_source"):
                origin = lock.get(key)
                if (
                    isinstance(origin, Mapping)
                    and origin.get("experiment") == experiment
                    and origin.get("run_id") == run_id
                    and isinstance(origin.get("notes"), str)
                ):
                    candidates.add(str(origin["notes"]))
            prior = lock.get("prior_entries", [])
            if isinstance(prior, list):
                for entry in prior:
                    if (
                        isinstance(entry, Mapping)
                        and entry.get("experiment") == experiment
                        and entry.get("expected_run_id") == run_id
                        and isinstance(entry.get("expected_notes"), str)
                    ):
                        candidates.add(str(entry["expected_notes"]))
            resolved = lock.get("resolved_stage", {})
            variants = resolved.get("variants", []) if isinstance(resolved, Mapping) else []
            for variant in variants if isinstance(variants, list) else []:
                if isinstance(variant, Mapping) and variant.get("experiment") == experiment:
                    try:
                        notes = generator._variant_notes(
                            str(lock["campaign"]),
                            str(lock["effective_stage"]),
                            variant,
                            variant["resolved_config"],
                            stage_lock=lock,
                        )
                    except (KeyError, TypeError, generator.CampaignConfigError) as error:
                        raise AdaptiveMaterializationError(
                            "Could not reconstruct schema-v1 candidate notes"
                        ) from error
                    candidates.add(notes)
        elif lock.get("schema_version") == SCHEMA_VERSION:
            origins = lock.get("origins", [])
            for origin in origins if isinstance(origins, list) else []:
                if (
                    isinstance(origin, Mapping)
                    and origin.get("experiment") == experiment
                    and origin.get("run_id") == run_id
                    and isinstance(origin.get("completion_notes"), str)
                ):
                    candidates.add(str(origin["completion_notes"]))
            resolved = lock.get("resolved_stage", {})
            variants = resolved.get("variants", []) if isinstance(resolved, Mapping) else []
            for variant in variants if isinstance(variants, list) else []:
                if isinstance(variant, Mapping) and variant.get("experiment") == experiment:
                    candidates.add(expected_variant_notes(lock, variant))
    if len(candidates) > 1:
        raise AdaptiveMaterializationError(
            f"Conflicting exact-notes provenance for {experiment!r}/{run_id!r}"
        )
    return next(iter(candidates), None)


def resolve_origin(
    *,
    plan: Mapping[str, Any],
    row: Mapping[str, Any],
    artifacts_dir: Path,
    provenance_locks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    experiment = _require_text(row.get("experiment"), label="origin experiment")
    run_id = _require_text(row.get("run_id"), label=f"{experiment} run_id")
    if row.get("completed") is not True or row.get("status") != "complete":
        raise AdaptiveMaterializationError(f"Origin {experiment!r} is not complete")
    iid_macro_ap = _finite_number(
        row.get("iid_macro_ap"), label=f"{experiment} IID macro AP"
    )
    if not 0 <= iid_macro_ap <= 1:
        raise AdaptiveMaterializationError(f"{experiment} IID macro AP is outside [0,1]")
    loss_variant = _require_text(
        row.get("loss_variant"), label=f"{experiment} loss_variant"
    )
    allowed_losses = generator.stage_loss_allowlist(
        plan, stage_name="special_loss_screen"
    ) | generator.stage_loss_allowlist(plan, stage_name="confirmation")
    if loss_variant not in allowed_losses or loss_variant not in generator.LOSS_VARIANT_SHA256:
        raise AdaptiveMaterializationError(
            f"Origin {experiment!r} uses undeclared loss {loss_variant!r}"
        )

    root = _artifact_root(artifacts_dir, row)
    completion_path = root / "notebook_completed.json"
    completion = _load_json(completion_path, label=f"{experiment} completion")
    if (
        completion.get("status") != "complete"
        or completion.get("experiment") != experiment
        or completion.get("run_id") != run_id
    ):
        raise AdaptiveMaterializationError(
            f"Completion artifact does not identify origin {experiment!r}/{run_id!r}"
        )
    frozen_training_contract = _validate_frozen_training_contract(
        completion, experiment=experiment
    )
    recipe_sha = _require_sha(
        completion.get("frozen_recipe_sha256"), label=f"{experiment} recipe SHA"
    )
    row_recipe_sha = row.get("recipe_sha256")
    if row_recipe_sha is not None and row_recipe_sha != recipe_sha:
        raise AdaptiveMaterializationError(f"{experiment} summary recipe SHA differs")
    expected_hook = generator.LOSS_VARIANT_SHA256[loss_variant]
    if completion.get("loss_hook_sha256") != expected_hook:
        raise AdaptiveMaterializationError(
            f"{experiment} completion loss hook differs from {loss_variant!r}"
        )
    if row.get("loss_hook_sha256") not in (None, expected_hook):
        raise AdaptiveMaterializationError(f"{experiment} summary loss hook differs")

    config_path = _exactly_one(root, "training_config.json")
    artifact_config = _load_json(config_path, label=f"{experiment} training config")
    resolved_config = deepcopy(artifact_config)
    base = _base_config()
    if "model" in base:
        resolved_config["model"] = base["model"]
    declared_config = row.get("resolved_config")
    if declared_config is not None and declared_config != resolved_config:
        raise AdaptiveMaterializationError(
            f"{experiment} summary resolved config differs from its artifact"
        )
    _safe_overrides(plan, resolved_config)
    if canonical_sha256(resolved_config) != recipe_sha:
        raise AdaptiveMaterializationError(
            f"{experiment} resolved config does not reproduce recipe SHA"
        )
    if row.get("seed") is not None and row.get("seed") != resolved_config.get("seed"):
        raise AdaptiveMaterializationError(f"{experiment} summary seed differs")

    iid_path = _exactly_one(root, "iid_validation_predictions.parquet")
    iid_sha = file_sha256(iid_path)
    if row.get("iid_predictions_sha256") not in (None, iid_sha):
        raise AdaptiveMaterializationError(
            f"{experiment} summary IID predictions SHA differs"
        )
    notes = completion.get("notes")
    if not isinstance(notes, str):
        raise AdaptiveMaterializationError(f"{experiment} completion has no notes string")
    try:
        parsed_notes = json.loads(notes)
    except json.JSONDecodeError as error:
        raise AdaptiveMaterializationError(
            f"{experiment} completion notes are not JSON"
        ) from error
    if not isinstance(parsed_notes, Mapping):
        raise AdaptiveMaterializationError(f"{experiment} completion notes are not an object")
    expected_notes = _completion_notes_from_locks(row, provenance_locks)
    row_notes_sha = row.get("expected_notes_sha256")
    if expected_notes is not None and notes != expected_notes:
        raise AdaptiveMaterializationError(
            f"{experiment} notes differ from exact origin provenance"
        )
    if expected_notes is None:
        if row_notes_sha is None:
            raise AdaptiveMaterializationError(
                f"{experiment} has no independently frozen notes provenance"
            )
        _require_sha(row_notes_sha, label=f"{experiment} expected notes SHA")
    if row_notes_sha is not None and row_notes_sha != text_sha256(notes):
        raise AdaptiveMaterializationError(f"{experiment} notes SHA differs")

    family_sha = recipe_family_sha256(
        resolved_config,
        loss_variant=loss_variant,
        loss_hook_sha256=expected_hook,
    )
    kernel_slug = _require_text(
        row.get("kernel_slug") or root.name, label=f"{experiment} kernel_slug"
    )
    code_bundle_sha = _require_sha(
        completion.get("code_bundle_sha256"), label=f"{experiment} code-bundle SHA"
    )
    row_source_sha = row.get("code_bundle_sha256") or row.get(
        "expected_source_sha256"
    )
    if row_source_sha is not None and row_source_sha != code_bundle_sha:
        raise AdaptiveMaterializationError(f"{experiment} summary source SHA differs")
    completion_sha = file_sha256(completion_path)
    origin_id = f"origin_{family_sha[:12]}_s{int(resolved_config['seed'])}_{run_id[:8]}"
    return {
        "origin_id": origin_id,
        "experiment": experiment,
        "run_id": run_id,
        "kernel_slug": kernel_slug,
        "source_role": str(row.get("role", "candidate")),
        "source_is_hypothesis": bool(
            row.get(
                "is_hypothesis",
                row.get("role") not in {"current_protocol_control", "stage_anchor"},
            )
        ),
        "origin_effective_stage": str(
            row.get("effective_stage") or row.get("stage") or "unknown"
        ),
        "iid_macro_ap": iid_macro_ap,
        "resolved_config": resolved_config,
        "recipe_sha256": recipe_sha,
        "recipe_family_sha256": family_sha,
        "loss_variant": loss_variant,
        "loss_hook_sha256": expected_hook,
        "code_bundle_sha256": code_bundle_sha,
        "expected_source_sha256": code_bundle_sha,
        "iid_predictions_sha256": iid_sha,
        "iid_predictions_artifact_path": _bound_file_path(
            iid_path, label=f"{experiment} IID predictions"
        ),
        "iid_predictions_relative_path": str(iid_path.relative_to(root)),
        "completion_sha256": completion_sha,
        "completion_artifact_path": _bound_file_path(
            completion_path, label=f"{experiment} completion"
        ),
        "training_config_artifact_sha256": file_sha256(config_path),
        "training_config_artifact_path": _bound_file_path(
            config_path, label=f"{experiment} training config"
        ),
        "frozen_training_contract": frozen_training_contract,
        "frozen_training_contract_sha256": canonical_sha256(
            frozen_training_contract
        ),
        "completion_notes": notes,
        "completion_notes_sha256": text_sha256(notes),
    }


def expected_variant_notes(
    lock: Mapping[str, Any], variant: Mapping[str, Any]
) -> str:
    """Return the exact notes string a future schema-v2 generator must embed."""
    config = variant.get("resolved_config")
    if not isinstance(config, Mapping):
        raise AdaptiveMaterializationError("Locked variant has no resolved config")
    origin_by_id = {
        origin.get("origin_id"): origin
        for origin in lock.get("origins", [])
        if isinstance(origin, Mapping)
    }
    lineage = []
    for origin_id in variant.get("origin_ids", []):
        origin = origin_by_id.get(origin_id)
        if not isinstance(origin, Mapping):
            raise AdaptiveMaterializationError(
                f"Locked variant references unknown origin {origin_id!r}"
            )
        lineage.append(
            {
                "origin_id": origin_id,
                "experiment": origin["experiment"],
                "run_id": origin["run_id"],
                "recipe_sha256": origin["recipe_sha256"],
                "recipe_family_sha256": origin["recipe_family_sha256"],
                "loss_variant": origin["loss_variant"],
                "loss_hook_sha256": origin["loss_hook_sha256"],
                "expected_source_sha256": origin["expected_source_sha256"],
                "code_bundle_sha256": origin["code_bundle_sha256"],
                "iid_predictions_sha256": origin["iid_predictions_sha256"],
                "completion_notes_sha256": origin["completion_notes_sha256"],
            }
        )
    details = {
        "campaign": lock["campaign"],
        "stage": lock["effective_stage"],
        "role": variant.get("role", "candidate"),
        "loss_variant": variant["loss_variant"],
        "epochs": config["epochs"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "warmup_ratio": config["warmup_ratio"],
        "label_smoothing": config["label_smoothing"],
        "max_grad_norm": config["max_grad_norm"],
        "batch_size_per_gpu": config["batch_size"],
        "gradient_accumulation": config["gradient_accumulation"],
        "effective_batch": int(config["batch_size"])
        * 2
        * int(config["gradient_accumulation"]),
        "model_load_kwargs": config.get("model_load_kwargs", {}),
        "seed": config["seed"],
        "stage_lock_payload_sha256": lock["lock_payload_sha256"],
        "expected_source_sha256": variant["expected_source_sha256"],
        "origin_lineage": lineage,
    }
    return canonical_json_dumps(details)


def _lock_ref(lock: Mapping[str, Any], *, lock_path: Path) -> dict[str, Any]:
    result = {
        "schema_version": int(lock["schema_version"]),
        "kind": str(lock["kind"]),
        "mode": lock.get("mode"),
        "effective_stage": str(lock["effective_stage"]),
        "lock_payload_sha256": str(lock["lock_payload_sha256"]),
        "lock_path": _bound_file_path(lock_path, label="prerequisite lock"),
        "execution_status": str(lock.get("execution_status", "runnable")),
    }
    if lock.get("schema_version") == SCHEMA_VERSION:
        frozen_budget = deepcopy(dict(lock["budget"]))
        result["budget_payload_sha256"] = canonical_sha256(frozen_budget)
        result["frozen_budget"] = frozen_budget
        result["budget_all_unique_kernel_slugs_after"] = deepcopy(
            frozen_budget["all_unique_kernel_slugs_after"]
        )
        result["budget_resulting_unique_kernels"] = int(
            frozen_budget["resulting_unique_kernels"]
        )
    elif lock.get("schema_version") == 1:
        parent = lock.get("parent")
        if not isinstance(parent, Mapping):
            raise AdaptiveMaterializationError(
                "Schema-v1 prerequisite has no frozen parent"
            )
        result["frozen_parent"] = deepcopy(dict(parent))
        result["frozen_parent_sha256"] = canonical_sha256(parent)
    return result


def _with_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    result["lock_payload_sha256"] = canonical_sha256(result)
    return result


def validate_lock_payload(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any] | None = None,
    trusted_provenance: Mapping[str, Any] | None = None,
) -> None:
    if set(payload) != {
        "schema_version",
        "kind",
        "campaign",
        "mode",
        "source_stage",
        "target_stage",
        "effective_stage",
        "execution_status",
        "expected_source_sha256",
        "source_plan_sha256",
        "decision_inputs_summary_sha256",
        "selection_metric",
        "prerequisites",
        "family",
        "origins",
        "decision_evidence_sha256",
        "decision",
        "resolved_stage",
        "budget",
        "lock_payload_sha256",
    }:
        raise ExistingAdaptiveLockConflictError(
            "Adaptive lock top-level schema differs"
        )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ExistingAdaptiveLockConflictError("Unsupported adaptive lock schema")
    if payload.get("kind") not in {LOCK_KIND, RECEIPT_KIND}:
        raise ExistingAdaptiveLockConflictError("Unsupported adaptive lock kind")
    mode = payload.get("mode")
    if mode not in MODES or payload.get("effective_stage") != EFFECTIVE_STAGES[mode]:
        raise ExistingAdaptiveLockConflictError("Adaptive lock mode/stage differs")
    status = payload.get("execution_status")
    expected_kind = RECEIPT_KIND if status == "skipped" else LOCK_KIND
    if status not in {"runnable", "skipped"} or payload.get("kind") != expected_kind:
        raise ExistingAdaptiveLockConflictError("Adaptive lock status/kind differs")
    stored = str(payload.get("lock_payload_sha256", ""))
    unhashed = dict(payload)
    unhashed.pop("lock_payload_sha256", None)
    if stored != canonical_sha256(unhashed):
        raise ExistingAdaptiveLockConflictError("Adaptive lock payload SHA is invalid")
    if plan is None:
        raise ExistingAdaptiveLockConflictError(
            "Full adaptive-lock validation requires the exact campaign plan"
        )
    if trusted_provenance is None:
        raise ExistingAdaptiveLockConflictError(
            "Schema-v2 execution requires caller-supplied trusted provenance"
        )
    validate_trusted_provenance(trusted_provenance, plan=plan)
    trusted_sources = _trusted_source_documents(trusted_provenance)
    trusted_lock_authorities = {
        str(row["lock_path"]): row
        for row in trusted_provenance["prerequisite_locks"]
    }
    trusted_artifacts_dir = Path(str(trusted_provenance["artifacts_dir"]))
    if plan is not None:
        if payload.get("campaign") != plan.get("campaign") or payload.get(
            "source_plan_sha256"
        ) != canonical_sha256(plan):
            raise ExistingAdaptiveLockConflictError("Adaptive lock belongs to another plan")
        expected_sources = {
            "loss_primary": axis_materializer.expected_source_stage(
                plan, target_stage="special_loss_screen", coordinate=None
            ),
            "loss_overlay": EFFECTIVE_STAGES["loss_primary"],
            "loss_lr_refine": "special_loss_screen__primary_final",
            "confirmation": "special_loss_screen__final",
        }
        if (
            payload.get("selection_metric")
            != plan.get("selection_protocol", {}).get("primary_metric")
            or payload.get("source_stage") != expected_sources[mode]
            or payload.get("target_stage")
            != ("confirmation" if mode == "confirmation" else "special_loss_screen")
        ):
            raise ExistingAdaptiveLockConflictError(
                "Adaptive selection/source/target contract differs"
            )
    execution_source_sha256 = _require_sha(
        payload.get("expected_source_sha256"), label="execution source SHA"
    )

    origins = payload.get("origins")
    resolved = payload.get("resolved_stage")
    budget = payload.get("budget")
    if not isinstance(origins, list) or not isinstance(resolved, Mapping):
        raise ExistingAdaptiveLockConflictError("Adaptive lock has malformed origins/stage")
    if payload.get("decision_evidence_sha256") != canonical_sha256(
        {"origins": origins}
    ):
        raise ExistingAdaptiveLockConflictError("Frozen decision evidence SHA differs")
    variants = resolved.get("variants")
    if not isinstance(variants, list) or (status == "skipped" and variants):
        raise ExistingAdaptiveLockConflictError("Adaptive lock variants/status differ")
    if status == "runnable" and not variants:
        raise ExistingAdaptiveLockConflictError("Runnable adaptive lock has no variants")
    if not isinstance(budget, Mapping):
        raise ExistingAdaptiveLockConflictError("Adaptive lock has no budget ledger")
    if set(budget) != {
        "counting_identity",
        "plan_counting_rule",
        "history_snapshot",
        "history_snapshot_sha256",
        "prior_unique_kernel_slugs",
        "new_unique_kernel_slugs",
        "all_unique_kernel_slugs_after",
        "prior_unique_kernels",
        "new_unique_kernels",
        "resulting_unique_kernels",
        "hard_limit",
    }:
        raise ExistingAdaptiveLockConflictError("Adaptive budget schema differs")
    prior = budget.get("prior_unique_kernel_slugs")
    new = budget.get("new_unique_kernel_slugs")
    all_after = budget.get("all_unique_kernel_slugs_after")
    if not all(isinstance(value, list) for value in (prior, new, all_after)):
        raise ExistingAdaptiveLockConflictError("Budget kernel unions are malformed")
    if any(
        not isinstance(slug, str) or not slug
        for values in (prior, new, all_after)
        for slug in values
    ):
        raise ExistingAdaptiveLockConflictError("Budget kernel identities are malformed")
    if budget.get("counting_identity") != "kernel_slug_union":
        raise ExistingAdaptiveLockConflictError("Budget counting identity differs")
    if plan is not None and budget.get("plan_counting_rule") != plan.get(
        "budget", {}
    ).get("counting_rule"):
        raise ExistingAdaptiveLockConflictError("Budget plan counting rule differs")
    if prior != sorted(set(prior)) or new != sorted(set(new)):
        raise ExistingAdaptiveLockConflictError("Budget kernel lists are not canonical sets")
    if all_after != sorted(set(prior) | set(new)):
        raise ExistingAdaptiveLockConflictError("Budget all-kernel union differs")
    if (
        budget.get("prior_unique_kernels") != len(prior)
        or budget.get("new_unique_kernels") != len(new)
        or budget.get("resulting_unique_kernels") != len(all_after)
        or budget.get("hard_limit") != 37
        or len(all_after) > 37
    ):
        raise ExistingAdaptiveLockConflictError("Budget counts or hard cap differ")
    if sorted(variant.get("kernel_slug") for variant in variants) != new:
        raise ExistingAdaptiveLockConflictError("Budget new kernels differ from variants")
    history_snapshot = budget.get("history_snapshot")
    if (
        not isinstance(history_snapshot, Mapping)
        or budget.get("history_snapshot_sha256")
        != canonical_sha256(history_snapshot)
        or history_snapshot.get("prior_unique_kernel_slugs") != prior
    ):
        raise ExistingAdaptiveLockConflictError("Budget history snapshot differs")
    source_documents = history_snapshot.get("source_documents")
    prerequisite_budgets = history_snapshot.get("prerequisite_budgets")
    frozen_origin_slugs = history_snapshot.get("frozen_origin_kernel_slugs")
    if not all(
        isinstance(value, list)
        for value in (source_documents, prerequisite_budgets, frozen_origin_slugs)
    ):
        raise ExistingAdaptiveLockConflictError("Budget history components are malformed")
    if set(history_snapshot) != {
        "source_documents",
        "prerequisite_budgets",
        "frozen_origin_kernel_slugs",
        "prior_unique_kernel_slugs",
    } or any(not isinstance(row, Mapping) for row in source_documents) or source_documents != sorted(
        source_documents,
        key=lambda row: (row.get("document_sha256"), row.get("kernel_slugs")),
    ):
        raise ExistingAdaptiveLockConflictError(
            "Budget history snapshot shape/order differs"
        )
    canonical_source_authorities: dict[str, Mapping[str, Any]] = {}
    trusted_documents_by_authority: dict[tuple[str, str], Mapping[str, Any]] = {}
    for authority, document in trusted_sources:
        document_sha = str(authority["document_sha256"])
        expected_source_path = str(authority["expected_source_path"])
        trusted_documents_by_authority[(expected_source_path, document_sha)] = document
        previous = canonical_source_authorities.get(document_sha)
        if previous is None or expected_source_path < str(
            previous["expected_source_path"]
        ):
            canonical_source_authorities[document_sha] = authority
    expected_payload_source_authorities = {
        (str(authority["expected_source_path"]), document_sha)
        for document_sha, authority in canonical_source_authorities.items()
    }
    observed_payload_source_authorities = {
        (
            str(document.get("document_path", "")),
            str(document.get("document_sha256", "")),
        )
        for document in source_documents
        if isinstance(document, Mapping)
    }
    if observed_payload_source_authorities != expected_payload_source_authorities:
        raise ExistingAdaptiveLockConflictError(
            "Budget source documents differ from caller-supplied trusted authorities"
        )

    reconstructed_prior: set[str] = set()
    seen_document_shas: set[str] = set()
    for document in source_documents:
        document_slugs = (
            document.get("kernel_slugs")
            if isinstance(document, Mapping)
            else None
        )
        document_sha = (
            str(document.get("document_sha256", ""))
            if isinstance(document, Mapping)
            else ""
        )
        document_path_text = (
            str(document.get("document_path", ""))
            if isinstance(document, Mapping)
            else ""
        )
        if (
            not isinstance(document, Mapping)
            or set(document)
            != {"document_sha256", "document_path", "kernel_slugs"}
            or SHA256_RE.fullmatch(document_sha) is None
            or document_sha in seen_document_shas
            or not isinstance(document_slugs, list)
            or any(not isinstance(slug, str) or not slug for slug in document_slugs)
            or document_slugs != sorted(set(document_slugs))
        ):
            raise ExistingAdaptiveLockConflictError("Budget source-document snapshot differs")
        bound_document = trusted_documents_by_authority.get(
            (document_path_text, document_sha)
        )
        if (
            not isinstance(bound_document, Mapping)
            or _summary_document_sha(bound_document) != document_sha
            or sorted(_budget_documents([bound_document], [], []))
            != document_slugs
        ):
            raise ExistingAdaptiveLockConflictError(
                "Budget source ledger differs from its trusted archived document"
            )
        seen_document_shas.add(document_sha)
        reconstructed_prior.update(document_slugs)
    prerequisites = payload.get("prerequisites")
    if not isinstance(prerequisites, list):
        raise ExistingAdaptiveLockConflictError(
            "Prerequisite lock references are malformed"
        )
    refs_by_sha = {
        str(ref.get("lock_payload_sha256")): ref
        for ref in prerequisites
        if isinstance(ref, Mapping)
    }
    if len(refs_by_sha) != len(prerequisites):
        raise ExistingAdaptiveLockConflictError("Prerequisite lock references are malformed")
    observed_prerequisite_authorities = {
        (str(ref.get("lock_path", "")), str(ref.get("lock_payload_sha256", "")))
        for ref in prerequisites
        if isinstance(ref, Mapping)
    }
    expected_prerequisite_authorities = {
        (str(path), str(authority["lock_payload_sha256"]))
        for path, authority in trusted_lock_authorities.items()
    }
    if observed_prerequisite_authorities != expected_prerequisite_authorities:
        raise ExistingAdaptiveLockConflictError(
            "Prerequisites differ from caller-supplied trusted authorities"
        )
    validated_prerequisite_locks: list[Mapping[str, Any]] = []
    for reference in prerequisites:
        lock_path_text = _require_text(
            reference.get("lock_path"), label="prerequisite lock path"
        )
        lock_path = Path(lock_path_text)
        if (
            _bound_file_path(lock_path, label="immutable prerequisite lock")
            != lock_path_text
        ):
            raise ExistingAdaptiveLockConflictError(
                "Prerequisite lock path is not canonical"
            )
        authority = trusted_lock_authorities.get(lock_path_text)
        if (
            not isinstance(authority, Mapping)
            or authority.get("lock_payload_sha256")
            != reference.get("lock_payload_sha256")
        ):
            raise ExistingAdaptiveLockConflictError(
                "Prerequisite lock is not a trusted authority"
            )
        shallow = _shallow_lock_payload(lock_path)
        if shallow.get("schema_version") == 1:
            try:
                actual_lock = generator.load_stage_lock(
                    lock_path,
                    plan=plan,
                    base_config=_base_config(),
                )
            except (generator.CampaignConfigError, OSError) as error:
                raise ExistingAdaptiveLockConflictError(
                    f"Strict schema-v1 provenance validation failed: {error}"
                ) from error
        elif shallow.get("schema_version") == SCHEMA_VERSION:
            actual_lock = shallow
        else:
            raise ExistingAdaptiveLockConflictError(
                "Unsupported trusted prerequisite-lock schema"
            )
        if _lock_ref(actual_lock, lock_path=lock_path) != reference:
            raise ExistingAdaptiveLockConflictError(
                "Prerequisite reference differs from its immutable lock"
            )
        validated_prerequisite_locks.append(actual_lock)
    if any(not isinstance(row, Mapping) for row in prerequisite_budgets) or [
        row.get("lock_payload_sha256") for row in prerequisite_budgets
    ] != sorted(refs_by_sha):
        raise ExistingAdaptiveLockConflictError("Budget prerequisite snapshots differ")
    for snapshot in prerequisite_budgets:
        if not isinstance(snapshot, Mapping):
            raise ExistingAdaptiveLockConflictError("Prerequisite budget is not an object")
        if set(snapshot) != {
            "lock_payload_sha256",
            "lock_path",
            "budget_payload_sha256",
            "kernel_slugs",
        }:
            raise ExistingAdaptiveLockConflictError(
                "Prerequisite budget snapshot shape differs"
            )
        ref = refs_by_sha[str(snapshot["lock_payload_sha256"])]
        if snapshot.get("lock_path") != ref.get("lock_path"):
            raise ExistingAdaptiveLockConflictError(
                "Prerequisite budget lock path differs"
            )
        snapshot_slugs = snapshot.get("kernel_slugs")
        if (
            not isinstance(snapshot_slugs, list)
            or any(not isinstance(slug, str) or not slug for slug in snapshot_slugs)
            or snapshot_slugs != sorted(set(snapshot_slugs))
            or snapshot.get("budget_payload_sha256")
            != ref.get("budget_payload_sha256")
        ):
            raise ExistingAdaptiveLockConflictError("Chained prerequisite budget differs")
        if ref.get("schema_version") == SCHEMA_VERSION and (
            snapshot_slugs
            != ref.get("budget_all_unique_kernel_slugs_after")
            or snapshot_slugs
            != (
                ref.get("frozen_budget", {}).get(
                    "all_unique_kernel_slugs_after"
                )
                if isinstance(ref.get("frozen_budget"), Mapping)
                else None
            )
            or ref.get("budget_payload_sha256")
            != (
                canonical_sha256(ref["frozen_budget"])
                if isinstance(ref.get("frozen_budget"), Mapping)
                else None
            )
            or ref.get("budget_resulting_unique_kernels")
            != len(snapshot_slugs)
        ):
            raise ExistingAdaptiveLockConflictError(
                "Chained adaptive budget union differs"
            )
        reconstructed_prior.update(snapshot_slugs)
    if (
        any(not isinstance(slug, str) or not slug for slug in frozen_origin_slugs)
        or frozen_origin_slugs != sorted(set(frozen_origin_slugs))
    ):
        raise ExistingAdaptiveLockConflictError("Frozen origin budget slugs differ")
    if frozen_origin_slugs != sorted(
        {
            str(origin.get("kernel_slug"))
            for origin in origins
            if isinstance(origin, Mapping)
        }
    ):
        raise ExistingAdaptiveLockConflictError("Budget omits a frozen origin kernel")
    reconstructed_prior.update(frozen_origin_slugs)
    if sorted(reconstructed_prior) != prior:
        raise ExistingAdaptiveLockConflictError("Budget prior is not the exact chained union")
    decision_summary_sha = _require_sha(
        payload.get("decision_inputs_summary_sha256"),
        label="decision-input summary SHA",
    )
    if sum(
        document.get("document_sha256") == decision_summary_sha
        for document in source_documents
    ) != 1:
        raise ExistingAdaptiveLockConflictError(
            "Budget history does not freeze the decision-input summary"
        )
    prior_ranges = {
        "loss_primary": (18, 22),
        "loss_overlay": (22, 26),
        "loss_lr_refine": (22, 27),
        "confirmation": (22, 29),
    }
    low, high = prior_ranges[mode]
    if not low <= len(prior) <= high:
        raise ExistingAdaptiveLockConflictError(
            f"{mode} prior budget must be in [{low}, {high}]"
        )
    predecessor_modes = {
        "loss_overlay": "loss_primary",
        "loss_lr_refine": "loss_overlay",
        "confirmation": "loss_lr_refine",
    }
    predecessor_mode = predecessor_modes.get(str(mode))
    if predecessor_mode is not None:
        predecessor_refs = [
            ref
            for ref in prerequisites
            if isinstance(ref, Mapping) and ref.get("mode") == predecessor_mode
        ]
        if len(predecessor_refs) != 1 or prior != predecessor_refs[0].get(
            "budget_all_unique_kernel_slugs_after"
        ):
            raise ExistingAdaptiveLockConflictError(
                "Budget prior is not the exact immediate-predecessor union"
            )

    origin_ids: set[str] = set()
    for origin in origins:
        if not isinstance(origin, Mapping):
            raise ExistingAdaptiveLockConflictError("Origin is not an object")
        if set(origin) != {
            "origin_id",
            "experiment",
            "run_id",
            "kernel_slug",
            "source_role",
            "source_is_hypothesis",
            "origin_effective_stage",
            "iid_macro_ap",
            "resolved_config",
            "recipe_sha256",
            "recipe_family_sha256",
            "loss_variant",
            "loss_hook_sha256",
            "code_bundle_sha256",
            "expected_source_sha256",
            "iid_predictions_sha256",
            "iid_predictions_artifact_path",
            "iid_predictions_relative_path",
            "completion_sha256",
            "completion_artifact_path",
            "training_config_artifact_sha256",
            "training_config_artifact_path",
            "frozen_training_contract",
            "frozen_training_contract_sha256",
            "completion_notes",
            "completion_notes_sha256",
        }:
            raise ExistingAdaptiveLockConflictError("Origin schema differs")
        origin_id = _require_text(origin.get("origin_id"), label="origin_id")
        if origin_id in origin_ids:
            raise ExistingAdaptiveLockConflictError("Duplicate origin_id")
        origin_ids.add(origin_id)
        experiment = _require_text(
            origin.get("experiment"), label="origin experiment"
        )
        run_id = _require_text(origin.get("run_id"), label="origin run_id")
        origin_kernel_slug = _require_text(
            origin.get("kernel_slug"), label="origin kernel_slug"
        )
        if (
            Path(origin_kernel_slug).name != origin_kernel_slug
            or origin_kernel_slug in {".", ".."}
        ):
            raise ExistingAdaptiveLockConflictError(
                "Origin kernel_slug is not a single safe path component"
            )
        _require_text(
            origin.get("origin_effective_stage"), label="origin effective stage"
        )
        iid_macro_ap = _finite_number(
            origin.get("iid_macro_ap"), label="origin IID macro AP"
        )
        if not 0 <= iid_macro_ap <= 1:
            raise ExistingAdaptiveLockConflictError(
                "Origin IID macro AP is outside [0,1]"
            )
        for key in (
            "recipe_sha256",
            "recipe_family_sha256",
            "loss_hook_sha256",
            "code_bundle_sha256",
            "expected_source_sha256",
            "iid_predictions_sha256",
            "completion_sha256",
            "training_config_artifact_sha256",
            "frozen_training_contract_sha256",
            "completion_notes_sha256",
        ):
            _require_sha(origin.get(key), label=f"origin {key}")
        if origin.get("expected_source_sha256") != origin.get("code_bundle_sha256"):
            raise ExistingAdaptiveLockConflictError("Origin expected source SHA differs")
        frozen_training_contract = origin.get("frozen_training_contract")
        if (
            frozen_training_contract != _frozen_training_contract_template()
            or origin.get("frozen_training_contract_sha256")
            != canonical_sha256(frozen_training_contract)
        ):
            raise ExistingAdaptiveLockConflictError(
                "Origin frozen training/sampling contract differs"
            )
        if not isinstance(origin.get("source_role"), str) or not isinstance(
            origin.get("source_is_hypothesis"), bool
        ):
            raise ExistingAdaptiveLockConflictError(
                "Origin role/hypothesis provenance is malformed"
            )
        notes = origin.get("completion_notes")
        config = origin.get("resolved_config")
        if not isinstance(notes, str) or text_sha256(notes) != origin.get(
            "completion_notes_sha256"
        ):
            raise ExistingAdaptiveLockConflictError("Origin exact notes differ")
        try:
            parsed_notes = json.loads(notes)
        except (TypeError, json.JSONDecodeError) as error:
            raise ExistingAdaptiveLockConflictError(
                "Origin exact notes are not JSON"
            ) from error
        if (
            not isinstance(parsed_notes, Mapping)
            or canonical_json_dumps(parsed_notes) != notes
        ):
            raise ExistingAdaptiveLockConflictError(
                "Origin exact notes are not a canonical object"
            )
        if not isinstance(config, Mapping) or canonical_sha256(config) != origin.get(
            "recipe_sha256"
        ):
            raise ExistingAdaptiveLockConflictError("Origin recipe SHA differs")
        loss_variant = str(origin.get("loss_variant", ""))
        if loss_variant not in generator.LOSS_VARIANT_SHA256 or generator.LOSS_VARIANT_SHA256[
            loss_variant
        ] != origin.get("loss_hook_sha256"):
            raise ExistingAdaptiveLockConflictError("Origin loss hook differs")
        if recipe_family_sha256(
            config,
            loss_variant=loss_variant,
            loss_hook_sha256=str(origin["loss_hook_sha256"]),
        ) != origin.get("recipe_family_sha256"):
            raise ExistingAdaptiveLockConflictError("Origin recipe-family SHA differs")
        completion_path = Path(
            _require_text(
                origin.get("completion_artifact_path"),
                label="origin completion artifact path",
            )
        )
        training_config_path = Path(
            _require_text(
                origin.get("training_config_artifact_path"),
                label="origin training-config artifact path",
            )
        )
        iid_predictions_path = Path(
            _require_text(
                origin.get("iid_predictions_artifact_path"),
                label="origin IID artifact path",
            )
        )
        bound_paths = (
            (completion_path, "completion", origin["completion_sha256"]),
            (
                training_config_path,
                "training config",
                origin["training_config_artifact_sha256"],
            ),
            (
                iid_predictions_path,
                "IID predictions",
                origin["iid_predictions_sha256"],
            ),
        )
        try:
            trusted_origin_root = (
                trusted_artifacts_dir / origin_kernel_slug
            ).resolve(strict=True)
            trusted_origin_root.relative_to(trusted_artifacts_dir)
        except (OSError, ValueError) as error:
            raise ExistingAdaptiveLockConflictError(
                "Origin kernel artifact directory escapes/is missing from trusted root"
            ) from error
        if completion_path != trusted_origin_root / "notebook_completed.json":
            raise ExistingAdaptiveLockConflictError(
                "Origin completion path is not the trusted kernel completion"
            )
        for artifact_path, label, expected_sha in bound_paths:
            try:
                artifact_path.relative_to(trusted_origin_root)
            except ValueError as error:
                raise ExistingAdaptiveLockConflictError(
                    f"Origin {label} escapes its trusted kernel artifact directory"
                ) from error
            if (
                _bound_file_path(artifact_path, label=f"origin {label}")
                != str(artifact_path)
                or file_sha256(artifact_path) != expected_sha
            ):
                raise ExistingAdaptiveLockConflictError(
                    f"Origin immutable {label} binding differs"
                )
        relative_iid_path = _require_text(
            origin.get("iid_predictions_relative_path"),
            label="origin IID relative path",
        )
        relative_iid = Path(relative_iid_path)
        if relative_iid.is_absolute() or ".." in relative_iid.parts:
            raise ExistingAdaptiveLockConflictError(
                "Origin IID relative path must be a contained relative path"
            )
        try:
            expected_iid_path = (
                completion_path.parent / relative_iid
            ).resolve(strict=True)
        except OSError as error:
            raise ExistingAdaptiveLockConflictError(
                "Origin IID relative artifact is missing"
            ) from error
        if expected_iid_path != iid_predictions_path:
            raise ExistingAdaptiveLockConflictError(
                "Origin IID artifact path escapes its completion root"
            )

        matching_rows: list[Mapping[str, Any]] = []
        for _, trusted_document in trusted_sources:
            rows = trusted_document.get("runs", [])
            if not isinstance(rows, list):
                continue
            matching_rows.extend(
                row
                for row in rows
                if isinstance(row, Mapping)
                and row.get("experiment") == experiment
                and row.get("run_id") == run_id
            )
        if not matching_rows:
            raise ExistingAdaptiveLockConflictError(
                "Origin is absent from all trusted archived summaries"
            )
        expected_origins: dict[str, Mapping[str, Any]] = {}
        for row in matching_rows:
            row_slug = _require_text(
                row.get("kernel_slug"), label=f"{experiment} trusted kernel_slug"
            )
            if Path(row_slug).name != row_slug or row_slug in {".", ".."}:
                raise ExistingAdaptiveLockConflictError(
                    "Trusted origin kernel_slug is not a single safe path component"
                )
            expected_root = (trusted_artifacts_dir / row_slug).resolve(
                strict=True
            )
            try:
                expected_root.relative_to(trusted_artifacts_dir)
            except ValueError as error:
                raise ExistingAdaptiveLockConflictError(
                    "Trusted origin artifact directory escapes its root"
                ) from error
            if expected_root != completion_path.parent:
                raise ExistingAdaptiveLockConflictError(
                    "Origin completion is outside its trusted kernel artifact directory"
                )
            try:
                reconstructed = resolve_origin(
                    plan=plan,
                    row=row,
                    artifacts_dir=trusted_artifacts_dir,
                    provenance_locks=validated_prerequisite_locks,
                )
            except AdaptiveMaterializationError as error:
                raise ExistingAdaptiveLockConflictError(
                    f"Trusted origin reconstruction failed: {error}"
                ) from error
            expected_origins[canonical_json_dumps(reconstructed)] = reconstructed
        if len(expected_origins) != 1 or origin != next(
            iter(expected_origins.values())
        ):
            raise ExistingAdaptiveLockConflictError(
                "Origin differs from its trusted archived summary/artifact reconstruction"
            )
        bound_completion = _load_json(
            completion_path, label=f"bound completion for {experiment}"
        )
        if (
            bound_completion.get("status") != "complete"
            or bound_completion.get("experiment") != experiment
            or bound_completion.get("run_id") != run_id
            or bound_completion.get("frozen_recipe_sha256")
            != origin["recipe_sha256"]
            or bound_completion.get("loss_hook_sha256")
            != origin["loss_hook_sha256"]
            or bound_completion.get("code_bundle_sha256")
            != origin["code_bundle_sha256"]
            or bound_completion.get("notes") != origin["completion_notes"]
            or _validate_frozen_training_contract(
                bound_completion, experiment=experiment
            )
            != frozen_training_contract
        ):
            raise ExistingAdaptiveLockConflictError(
                "Origin differs from its immutable completion artifact"
            )
        expected_origin_id = (
            f"origin_{origin['recipe_family_sha256'][:12]}_"
            f"s{int(config['seed'])}_{run_id[:8]}"
        )
        if origin_id != expected_origin_id or not experiment:
            raise ExistingAdaptiveLockConflictError(
                "Origin deterministic identity differs"
            )
        if plan is not None:
            _safe_overrides(plan, config)

    seen_experiments: set[str] = set()
    seen_slugs: set[str] = set()
    for variant in variants:
        if not isinstance(variant, Mapping):
            raise ExistingAdaptiveLockConflictError("Variant is not an object")
        loss_variant = str(variant.get("loss_variant", ""))
        config = variant.get("resolved_config")
        if not isinstance(config, Mapping) or loss_variant not in generator.LOSS_VARIANT_SHA256:
            raise ExistingAdaptiveLockConflictError("Variant recipe is malformed")
        if variant.get("expected_loss_hook_sha256") != generator.LOSS_VARIANT_SHA256[
            loss_variant
        ]:
            raise ExistingAdaptiveLockConflictError("Variant loss hook differs")
        if variant.get("expected_source_sha256") != execution_source_sha256:
            raise ExistingAdaptiveLockConflictError("Variant execution source SHA differs")
        if canonical_sha256(config) != variant.get("expected_recipe_sha256"):
            raise ExistingAdaptiveLockConflictError("Variant recipe SHA differs")
        if recipe_family_sha256(
            config,
            loss_variant=loss_variant,
            loss_hook_sha256=str(variant["expected_loss_hook_sha256"]),
        ) != variant.get("expected_recipe_family_sha256"):
            raise ExistingAdaptiveLockConflictError("Variant recipe-family SHA differs")
        if any(origin_id not in origin_ids for origin_id in variant.get("origin_ids", [])):
            raise ExistingAdaptiveLockConflictError("Variant references an unknown origin")
        experiment, slug, recipe_sha, family_sha = _variant_identity(
            mode=str(mode), loss_variant=loss_variant, config=config
        )
        if (
            variant.get("experiment") != experiment
            or variant.get("kernel_slug") != slug
            or variant.get("expected_recipe_sha256") != recipe_sha
            or variant.get("expected_recipe_family_sha256") != family_sha
        ):
            raise ExistingAdaptiveLockConflictError("Variant deterministic identity differs")
        if experiment in seen_experiments or slug in seen_slugs:
            raise ExistingAdaptiveLockConflictError("Duplicate adaptive variant identity")
        seen_experiments.add(experiment)
        seen_slugs.add(slug)
        if plan is not None:
            if generator.resolved_config_overrides(_base_config(), config) != variant.get(
                "overrides"
            ):
                raise ExistingAdaptiveLockConflictError("Variant overrides differ")
            _safe_overrides(plan, config)
    if plan is not None:
        _validate_mode_semantics(payload, plan=plan)


def _validate_mode_semantics(
    payload: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> None:
    mode = str(payload["mode"])
    status = str(payload["execution_status"])
    family = payload["family"]
    decision = payload["decision"]
    resolved = payload["resolved_stage"]
    origins = payload["origins"]
    variants = resolved["variants"]
    if not isinstance(family, Mapping) or not isinstance(decision, Mapping):
        raise ExistingAdaptiveLockConflictError("Adaptive family/decision is malformed")
    by_id = {str(origin["origin_id"]): origin for origin in origins}
    loss_stage = _stage(plan, "special_loss_screen")
    confirmation_stage = _stage(plan, "confirmation")
    execution_source = str(payload["expected_source_sha256"])
    prerequisites = payload["prerequisites"]

    expected_prerequisite_modes: dict[str, set[str]] = {
        "loss_primary": set(),
        "loss_overlay": {"loss_primary"},
        "loss_lr_refine": {"loss_primary", "loss_overlay"},
        "confirmation": {"loss_primary", "loss_overlay", "loss_lr_refine"},
    }
    if mode == "loss_primary":
        expected_source_stage = axis_materializer.expected_source_stage(
            plan, target_stage="special_loss_screen", coordinate=None
        )
        if len(prerequisites) != 1:
            raise ExistingAdaptiveLockConflictError(
                "Primary lock must cite exactly one schema-v1 source lock"
            )
        reference = prerequisites[0]
        expected_keys = {
            "schema_version",
            "kind",
            "mode",
            "effective_stage",
            "lock_payload_sha256",
            "lock_path",
            "execution_status",
            "frozen_parent",
            "frozen_parent_sha256",
        }
        frozen_parent = (
            reference.get("frozen_parent")
            if isinstance(reference, Mapping)
            else None
        )
        if (
            not isinstance(reference, Mapping)
            or set(reference) != expected_keys
            or reference.get("schema_version") != 1
            or reference.get("kind") != "minilm_5ep_sft_stage_lock"
            or reference.get("mode") is not None
            or reference.get("effective_stage") != expected_source_stage
            or reference.get("execution_status") != "runnable"
            or SHA256_RE.fullmatch(
                str(reference.get("lock_payload_sha256", ""))
            )
            is None
            or not isinstance(frozen_parent, Mapping)
            or reference.get("frozen_parent_sha256")
            != canonical_sha256(frozen_parent)
        ):
            raise ExistingAdaptiveLockConflictError(
                "Primary schema-v1 prerequisite reference differs"
            )
    else:
        expected_modes = expected_prerequisite_modes[mode]
        if len(prerequisites) != len(expected_modes):
            raise ExistingAdaptiveLockConflictError(
                "Adaptive prerequisite chain length differs"
            )
        observed_modes: set[str] = set()
        for reference in prerequisites:
            expected_keys = {
                "schema_version",
                "kind",
                "mode",
                "effective_stage",
                "lock_payload_sha256",
                "lock_path",
                "execution_status",
                "budget_payload_sha256",
                "frozen_budget",
                "budget_all_unique_kernel_slugs_after",
                "budget_resulting_unique_kernels",
            }
            if not isinstance(reference, Mapping) or set(reference) != expected_keys:
                raise ExistingAdaptiveLockConflictError(
                    "Adaptive prerequisite reference shape differs"
                )
            ref_mode = str(reference.get("mode", ""))
            ref_status = reference.get("execution_status")
            ref_slugs = reference.get("budget_all_unique_kernel_slugs_after")
            frozen_budget = reference.get("frozen_budget")
            if (
                reference.get("schema_version") != SCHEMA_VERSION
                or ref_mode not in expected_modes
                or ref_mode in observed_modes
                or reference.get("effective_stage") != EFFECTIVE_STAGES[ref_mode]
                or ref_status not in {"runnable", "skipped"}
                or reference.get("kind")
                != (RECEIPT_KIND if ref_status == "skipped" else LOCK_KIND)
                or SHA256_RE.fullmatch(
                    str(reference.get("lock_payload_sha256", ""))
                )
                is None
                or SHA256_RE.fullmatch(
                    str(reference.get("budget_payload_sha256", ""))
                )
                is None
                or not isinstance(frozen_budget, Mapping)
                or reference.get("budget_payload_sha256")
                != canonical_sha256(frozen_budget)
                or not isinstance(ref_slugs, list)
                or any(not isinstance(slug, str) or not slug for slug in ref_slugs)
                or ref_slugs != sorted(set(ref_slugs))
                or frozen_budget.get("all_unique_kernel_slugs_after") != ref_slugs
                or frozen_budget.get("resulting_unique_kernels") != len(ref_slugs)
                or frozen_budget.get("history_snapshot_sha256")
                != canonical_sha256(frozen_budget.get("history_snapshot"))
                or frozen_budget.get("counting_identity") != "kernel_slug_union"
                or frozen_budget.get("plan_counting_rule")
                != plan["budget"]["counting_rule"]
                or frozen_budget.get("hard_limit") != 37
                or reference.get("budget_resulting_unique_kernels")
                != len(ref_slugs)
            ):
                raise ExistingAdaptiveLockConflictError(
                    "Adaptive prerequisite reference differs"
                )
            observed_modes.add(ref_mode)
        if observed_modes != expected_modes or prerequisites != sorted(
            prerequisites,
            key=lambda row: (
                row["schema_version"],
                row["effective_stage"],
                row["lock_payload_sha256"],
            ),
        ):
            raise ExistingAdaptiveLockConflictError(
                "Adaptive prerequisite order/modes differ"
            )
    reference_by_mode = {
        str(reference["mode"]): reference
        for reference in prerequisites
        if isinstance(reference, Mapping) and reference.get("mode") is not None
    }

    def require_completed_variant_notes(
        origin: Mapping[str, Any],
        *,
        producer_mode: str,
        lineage: Sequence[Mapping[str, Any]],
    ) -> None:
        reference = reference_by_mode.get(producer_mode)
        if reference is None:
            raise ExistingAdaptiveLockConflictError(
                f"Missing {producer_mode!r} provenance for completed origin"
            )
        config = origin["resolved_config"]
        frozen_lineage = [
            {
                "origin_id": parent["origin_id"],
                "experiment": parent["experiment"],
                "run_id": parent["run_id"],
                "recipe_sha256": parent["recipe_sha256"],
                "recipe_family_sha256": parent["recipe_family_sha256"],
                "loss_variant": parent["loss_variant"],
                "loss_hook_sha256": parent["loss_hook_sha256"],
                "expected_source_sha256": parent["expected_source_sha256"],
                "code_bundle_sha256": parent["code_bundle_sha256"],
                "iid_predictions_sha256": parent["iid_predictions_sha256"],
                "completion_notes_sha256": parent["completion_notes_sha256"],
            }
            for parent in lineage
        ]
        expected_notes = canonical_json_dumps(
            {
                "campaign": payload["campaign"],
                "stage": EFFECTIVE_STAGES[producer_mode],
                "role": origin["source_role"],
                "loss_variant": origin["loss_variant"],
                "epochs": config["epochs"],
                "learning_rate": config["learning_rate"],
                "weight_decay": config["weight_decay"],
                "warmup_ratio": config["warmup_ratio"],
                "label_smoothing": config["label_smoothing"],
                "max_grad_norm": config["max_grad_norm"],
                "batch_size_per_gpu": config["batch_size"],
                "gradient_accumulation": config["gradient_accumulation"],
                "effective_batch": int(config["batch_size"])
                * 2
                * int(config["gradient_accumulation"]),
                "model_load_kwargs": config.get("model_load_kwargs", {}),
                "seed": config["seed"],
                "stage_lock_payload_sha256": reference["lock_payload_sha256"],
                "expected_source_sha256": origin["expected_source_sha256"],
                "origin_lineage": frozen_lineage,
            }
        )
        if origin.get("completion_notes") != expected_notes:
            raise ExistingAdaptiveLockConflictError(
                f"Completed {producer_mode} origin exact notes/lineage differ"
            )

    def exact_loss_map(
        evidence: Sequence[Mapping[str, Any]], expected_losses: Sequence[str]
    ) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for origin in evidence:
            loss_name = str(origin.get("loss_variant"))
            if loss_name in result:
                raise ExistingAdaptiveLockConflictError(
                    f"Frozen evidence contains duplicate loss {loss_name!r}"
                )
            result[loss_name] = origin
        if set(result) != set(expected_losses):
            raise ExistingAdaptiveLockConflictError(
                "Frozen loss evidence differs from the declared family"
            )
        return result

    def require_primary_evidence(
        evidence: Sequence[Mapping[str, Any]], *, allow_combo: bool
    ) -> tuple[Mapping[str, Any], dict[str, Mapping[str, Any]]]:
        base_losses = list(loss_stage["loss_variants"])
        observed_losses = [str(origin.get("loss_variant")) for origin in evidence]
        extras = [loss_name for loss_name in observed_losses if loss_name not in base_losses]
        declared_combos = set(
            loss_stage["conditional_combination"]["variants_by_balance"].values()
        )
        if (
            len(extras) > (1 if allow_combo else 0)
            or any(loss_name not in declared_combos for loss_name in extras)
        ):
            raise ExistingAdaptiveLockConflictError("Primary loss evidence has extra losses")
        base_evidence = [
            origin
            for origin in evidence
            if str(origin.get("loss_variant")) in base_losses
        ]
        base_map = exact_loss_map(base_evidence, base_losses)
        combo_plan = loss_stage["conditional_combination"]
        balance_names = [
            name for name in base_losses if name in combo_plan["variants_by_balance"]
        ]
        best_balance = min(
            (base_map[name] for name in balance_names),
            key=lambda origin: (
                -float(origin["iid_macro_ap"]),
                balance_names.index(str(origin["loss_variant"])),
            ),
        )
        focal = base_map["focal_bce_gamma2_scale4"]
        base_anchor = base_map["bce"]
        combo_threshold = float(combo_plan["minimum_raw_delta_vs_tuned_bce"])
        combo_triggered = _strict_delta_greater(
            float(best_balance["iid_macro_ap"]),
            float(base_anchor["iid_macro_ap"]),
            combo_threshold,
        ) and _strict_delta_greater(
            float(focal["iid_macro_ap"]),
            float(base_anchor["iid_macro_ap"]),
            combo_threshold,
        )
        expected_combo = str(
            combo_plan["variants_by_balance"][best_balance["loss_variant"]]
        )
        expected_extras = [expected_combo] if allow_combo and combo_triggered else []
        if allow_combo and reference_by_mode.get("loss_overlay", {}).get(
            "execution_status"
        ) != ("runnable" if combo_triggered else "skipped"):
            raise ExistingAdaptiveLockConflictError(
                "Overlay prerequisite status differs from its frozen trigger evidence"
            )
        if sorted(extras) != sorted(expected_extras):
            raise ExistingAdaptiveLockConflictError(
                "Overlay evidence does not match its exact plan trigger"
            )
        loss_map = exact_loss_map(evidence, [*base_losses, *expected_extras])
        anchor = loss_map["bce"]
        expected_anchor_stage = axis_materializer.expected_source_stage(
            plan, target_stage="special_loss_screen", coordinate=None
        )
        if (
            anchor.get("resolved_config", {}).get("seed") != 42
            or anchor.get("source_role") != "stage_anchor"
            or anchor.get("source_is_hypothesis") is not False
            or anchor.get("origin_effective_stage") != expected_anchor_stage
        ):
            raise ExistingAdaptiveLockConflictError(
                "Tuned BCE evidence role/stage is not the exact seed-42 anchor"
            )
        for loss_name, origin in loss_map.items():
            if (
                origin.get("resolved_config") != anchor.get("resolved_config")
                or origin.get("resolved_config", {}).get("seed") != 42
            ):
                raise ExistingAdaptiveLockConflictError(
                    f"Loss evidence {loss_name!r} changed the tuned optimizer config"
                )
        primary_candidates = [loss_map[name] for name in base_losses if name != "bce"]
        for origin in primary_candidates:
            if (
                origin.get("source_role") != "candidate"
                or origin.get("source_is_hypothesis") is not True
                or origin.get("origin_effective_stage")
                != EFFECTIVE_STAGES["loss_primary"]
                or origin.get("code_bundle_sha256") != execution_source
                or origin.get("expected_source_sha256") != execution_source
            ):
                raise ExistingAdaptiveLockConflictError(
                    "Primary loss candidate role/stage/source differs"
                )
            require_completed_variant_notes(
                origin,
                producer_mode="loss_primary",
                lineage=[anchor],
            )
        if expected_extras:
            combo = loss_map[expected_combo]
            if (
                combo.get("source_role") != "candidate"
                or combo.get("source_is_hypothesis") is not True
                or combo.get("origin_effective_stage")
                != EFFECTIVE_STAGES["loss_overlay"]
                or combo.get("code_bundle_sha256") != execution_source
                or combo.get("expected_source_sha256") != execution_source
            ):
                raise ExistingAdaptiveLockConflictError(
                    "Overlay origin role/stage/source differs"
                )
            require_completed_variant_notes(
                combo,
                producer_mode="loss_overlay",
                lineage=[anchor, best_balance, focal],
            )
        return anchor, loss_map

    if mode == "loss_primary":
        expected_losses = list(loss_stage["loss_variants"])[1:]
        if status != "runnable" or len(origins) != 1:
            raise ExistingAdaptiveLockConflictError("Primary loss lock contract differs")
        anchor = origins[0]
        expected_anchor_stage = axis_materializer.expected_source_stage(
            plan, target_stage="special_loss_screen", coordinate=None
        )
        if (
            anchor.get("loss_variant") != "bce"
            or anchor.get("resolved_config", {}).get("seed") != 42
            or anchor.get("source_role") != "stage_anchor"
            or anchor.get("source_is_hypothesis") is not False
            or anchor.get("origin_effective_stage") != expected_anchor_stage
        ):
            raise ExistingAdaptiveLockConflictError(
                "Primary BCE anchor role/stage differs"
            )
        frozen_parent = prerequisites[0]["frozen_parent"]
        parent_origin_fields = {
            "experiment": "experiment",
            "run_id": "run_id",
            "kernel_slug": "kernel_slug",
            "recipe_sha256": "recipe_sha256",
            "iid_predictions_sha256": "iid_predictions_sha256",
            "iid_predictions_relative_path": "iid_predictions_relative_path",
            "completion_sha256": "completion_sha256",
            "training_config_artifact_sha256": (
                "training_config_artifact_sha256"
            ),
            "resolved_config": "resolved_config",
            "code_bundle_sha256": "code_bundle_sha256",
            "loss_hook_sha256": "loss_hook_sha256",
            "loss_variant": "loss_variant",
            "notes": "completion_notes",
        }
        if any(
            frozen_parent.get(parent_key) != anchor.get(origin_key)
            for parent_key, origin_key in parent_origin_fields.items()
        ):
            raise ExistingAdaptiveLockConflictError(
                "Primary anchor differs from the strict schema-v1 parent provenance"
            )
        expected_family = {
            "family_id": "special_loss_primary_seed42",
            "correction": "holm",
            "anchor_origin_id": anchor["origin_id"],
            "planned_candidate_hypotheses": 4,
            "reserved_conditional_hypotheses": 1,
            "maximum_hypotheses": 5,
            "reserved_slot_state": "p_equals_1_until_overlay_decision",
        }
        expected_variants = [
            _variant(
                plan=plan,
                mode="loss_primary",
                config=anchor["resolved_config"],
                loss_variant=loss_name,
                expected_source_sha256=execution_source,
                origin_ids=[anchor["origin_id"]],
                family_size=5,
                extra={"loss_declaration_rank": _loss_rank(plan, loss_name)},
            )
            for loss_name in expected_losses
        ]
        expected_decision = {
            "anchor_origin_id": anchor["origin_id"],
            "anchor_loss_variant": "bce",
            "seed": 42,
            "overlay_slot_reserved": True,
            "source_stage_snapshot_sha256": decision.get(
                "source_stage_snapshot_sha256"
            ),
        }
        _require_sha(
            expected_decision["source_stage_snapshot_sha256"],
            label="source-stage snapshot SHA",
        )
        expected_resolved = {
            "strategy": loss_stage["strategy"],
            "reused_origins": [anchor["origin_id"]],
            "variants": expected_variants,
        }
        if (
            dict(family) != expected_family
            or dict(decision) != expected_decision
            or dict(resolved) != expected_resolved
        ):
            raise ExistingAdaptiveLockConflictError(
                "Primary semantic projection differs from frozen anchor and plan"
            )
        return

    if mode == "loss_overlay":
        anchor, loss_map = require_primary_evidence(origins, allow_combo=False)
        combo_plan = loss_stage["conditional_combination"]
        balance_names = [
            name
            for name in loss_stage["loss_variants"]
            if name in combo_plan["variants_by_balance"]
        ]
        best_balance = min(
            (loss_map[name] for name in balance_names),
            key=lambda origin: (
                -float(origin["iid_macro_ap"]),
                balance_names.index(str(origin["loss_variant"])),
            ),
        )
        focal = loss_map["focal_bce_gamma2_scale4"]
        threshold = float(combo_plan["minimum_raw_delta_vs_tuned_bce"])
        triggered = _strict_delta_greater(
            float(best_balance["iid_macro_ap"]),
            float(anchor["iid_macro_ap"]),
            threshold,
        ) and _strict_delta_greater(
            float(focal["iid_macro_ap"]),
            float(anchor["iid_macro_ap"]),
            threshold,
        )
        combo_name = combo_plan["variants_by_balance"][best_balance["loss_variant"]]
        expected_variants = (
            [
                _variant(
                    plan=plan,
                    mode="loss_overlay",
                    config=anchor["resolved_config"],
                    loss_variant=combo_name,
                    expected_source_sha256=execution_source,
                    origin_ids=[
                        anchor["origin_id"],
                        best_balance["origin_id"],
                        focal["origin_id"],
                    ],
                    family_size=5,
                    extra={
                        "primary_family_slot": 5,
                        "loss_declaration_rank": _loss_rank(plan, combo_name),
                    },
                )
            ]
            if triggered
            else []
        )
        expected_family = {
            "family_id": "special_loss_primary_seed42",
            "correction": "holm",
            "anchor_origin_id": anchor["origin_id"],
            "maximum_hypotheses": 5,
            "reserved_slot": 5,
            "reserved_slot_state": "materialized" if triggered else "unused_p_equals_1",
        }
        expected_decision = {
            "decision_id": "best_balance_x_focal_overlay",
            "rule": combo_plan["run_if"],
            "operator": combo_plan["delta_operator"],
            "threshold": threshold,
            "best_balance_origin_id": best_balance["origin_id"],
            "best_balance_loss_variant": best_balance["loss_variant"],
            "best_balance_iid_delta_vs_tuned_bce": (
                best_balance["iid_macro_ap"] - anchor["iid_macro_ap"]
            ),
            "focal_origin_id": focal["origin_id"],
            "focal_iid_delta_vs_tuned_bce": (
                focal["iid_macro_ap"] - anchor["iid_macro_ap"]
            ),
            "triggered": triggered,
            "resolved_combination_loss_variant": combo_name if triggered else None,
        }
        expected_resolved = {
            "strategy": "conditional_best_balance_x_focal",
            "reused_origins": sorted(by_id),
            "variants": expected_variants,
        }
        if (
            status != ("runnable" if triggered else "skipped")
            or dict(family) != expected_family
            or dict(decision) != expected_decision
            or dict(resolved) != expected_resolved
        ):
            raise ExistingAdaptiveLockConflictError(
                "Overlay semantic projection differs from full primary evidence"
            )
        return

    if mode == "loss_lr_refine":
        anchor, _ = require_primary_evidence(origins, allow_combo=True)
        winner = _best_non_bce(plan, origins)
        trigger_plan = loss_stage["winner_lr_refinement_trigger"]
        threshold = float(trigger_plan["minimum_iid_delta_vs_tuned_bce"])
        triggered = _strict_delta_greater(
            float(winner["iid_macro_ap"]),
            float(anchor["iid_macro_ap"]),
            threshold,
        )
        expected_variants = []
        center_lr = float(winner["resolved_config"]["learning_rate"])
        if triggered:
            for multiplier in loss_stage["winner_lr_refinement_multipliers"]:
                if float(multiplier) == 1.0:
                    continue
                config = deepcopy(dict(winner["resolved_config"]))
                config["learning_rate"] = center_lr * float(multiplier)
                expected_variants.append(
                    _variant(
                        plan=plan,
                        mode="loss_lr_refine",
                        config=config,
                        loss_variant=str(winner["loss_variant"]),
                        expected_source_sha256=execution_source,
                        origin_ids=[winner["origin_id"], anchor["origin_id"]],
                        family_size=2,
                        extra={
                            "learning_rate_multiplier": float(multiplier),
                            "center_learning_rate": center_lr,
                        },
                    )
                )
        expected_family = {
            "family_id": f"special_loss_lr_refine_{winner['recipe_family_sha256'][:12]}",
            "correction": "holm",
            "anchor_origin_id": winner["origin_id"],
            "planned_candidate_hypotheses": 2,
            "materialized_candidate_hypotheses": len(expected_variants),
            "maximum_hypotheses": 2,
        }
        expected_decision = {
            "decision_id": "best_non_bce_lr_refinement",
            "winner_origin_id": winner["origin_id"],
            "winner_loss_variant": winner["loss_variant"],
            "winner_iid_delta_vs_tuned_bce": (
                winner["iid_macro_ap"] - anchor["iid_macro_ap"]
            ),
            "operator": trigger_plan["delta_operator"],
            "threshold": threshold,
            "triggered": triggered,
            "center_reused": True,
            "boundary_extension": False,
        }
        expected_resolved = {
            "strategy": "winner_non_bce_log2_lr_line",
            "reused_origins": [winner["origin_id"]],
            "variants": expected_variants,
        }
        if (
            status != ("runnable" if triggered else "skipped")
            or dict(family) != expected_family
            or dict(decision) != expected_decision
            or dict(resolved) != expected_resolved
        ):
            raise ExistingAdaptiveLockConflictError(
                "LR-refine semantic projection differs from full loss evidence"
            )
        return

    if mode == "confirmation":
        evidence_contract = decision.get("evidence_contract")
        if not isinstance(evidence_contract, Mapping) or set(evidence_contract) != {
            "baseline_origin_id",
            "tuned_bce_origin_id",
            "primary_family_origin_ids",
            "lr_refinement_executed",
            "lr_center_origin_id",
            "lr_candidate_origin_ids",
        }:
            raise ExistingAdaptiveLockConflictError(
                "Confirmation frozen evidence contract is malformed"
            )
        baseline = by_id.get(str(evidence_contract["baseline_origin_id"]))
        tuned = by_id.get(str(evidence_contract["tuned_bce_origin_id"]))
        if not isinstance(baseline, Mapping) or not isinstance(tuned, Mapping):
            raise ExistingAdaptiveLockConflictError("Confirmation anchors are missing")
        control = next(
            variant
            for variant in _stage(plan, "lr_log_line")["variants"]
            if variant.get("role") == "current_protocol_control"
        )
        alias = control["provenance_alias"]
        if (
            baseline.get("experiment") != control["experiment"]
            or baseline.get("resolved_config") != _base_config()
            or baseline.get("loss_variant") != "bce"
            or baseline.get("resolved_config", {}).get("seed") != 42
            or baseline.get("source_role") != "current_protocol_control"
            or baseline.get("source_is_hypothesis") is not False
            or baseline.get("origin_effective_stage") != "lr_log_line"
            or baseline.get("completion_notes_sha256")
            != alias["accepted_completion_notes_sha256"]
            or tuned.get("loss_variant") != "bce"
            or tuned.get("resolved_config", {}).get("seed") != 42
            or tuned.get("source_role") != "stage_anchor"
            or tuned.get("source_is_hypothesis") is not False
        ):
            raise ExistingAdaptiveLockConflictError(
                "Confirmation baseline/tuned-BCE provenance differs"
            )
        primary_ids = evidence_contract["primary_family_origin_ids"]
        lr_ids = evidence_contract["lr_candidate_origin_ids"]
        if (
            not isinstance(primary_ids, list)
            or not isinstance(lr_ids, list)
            or len(primary_ids) != len(set(primary_ids))
            or len(lr_ids) != len(set(lr_ids))
            or any(origin_id not in by_id for origin_id in [*primary_ids, *lr_ids])
            or tuned["origin_id"] not in primary_ids
        ):
            raise ExistingAdaptiveLockConflictError(
                "Confirmation evidence-origin membership differs"
            )
        expected_evidence_ids = {
            str(evidence_contract["baseline_origin_id"]),
            *map(str, primary_ids),
            *map(str, lr_ids),
        }
        if set(by_id) != expected_evidence_ids:
            raise ExistingAdaptiveLockConflictError(
                "Confirmation contains evidence outside its frozen contract"
            )
        primary_evidence = [by_id[origin_id] for origin_id in primary_ids]
        primary_anchor, primary_loss_map = require_primary_evidence(
            primary_evidence, allow_combo=True
        )
        if primary_anchor["origin_id"] != tuned["origin_id"]:
            raise ExistingAdaptiveLockConflictError(
                "Confirmation tuned BCE is not the primary family anchor"
            )
        ordered_primary_losses = list(loss_stage["loss_variants"])
        ordered_primary_losses.extend(
            loss_name
            for loss_name in _loss_order(plan)
            if loss_name in primary_loss_map
            and loss_name not in ordered_primary_losses
        )
        if primary_ids != [
            primary_loss_map[loss_name]["origin_id"]
            for loss_name in ordered_primary_losses
        ]:
            raise ExistingAdaptiveLockConflictError(
                "Confirmation primary evidence order differs"
            )
        initial_first = _best_non_bce(plan, primary_evidence)
        center = by_id.get(str(evidence_contract["lr_center_origin_id"]))
        if center != initial_first:
            raise ExistingAdaptiveLockConflictError(
                "Confirmation LR center is not the primary non-BCE winner"
            )
        lr_executed = evidence_contract["lr_refinement_executed"]
        lr_trigger_plan = loss_stage["winner_lr_refinement_trigger"]
        expected_lr_executed = _strict_delta_greater(
            float(initial_first["iid_macro_ap"]),
            float(tuned["iid_macro_ap"]),
            float(lr_trigger_plan["minimum_iid_delta_vs_tuned_bce"]),
        )
        if (
            not isinstance(lr_executed, bool)
            or lr_executed is not expected_lr_executed
            or reference_by_mode["loss_lr_refine"]["execution_status"]
            != ("runnable" if expected_lr_executed else "skipped")
            or len(lr_ids) != (2 if lr_executed else 0)
        ):
            raise ExistingAdaptiveLockConflictError(
                "Confirmation LR execution evidence differs"
            )
        first = initial_first
        if lr_executed:
            lr_evidence = [by_id[origin_id] for origin_id in lr_ids]
            multipliers = []
            for origin in lr_evidence:
                if (
                    origin.get("loss_variant") != center.get("loss_variant")
                    or origin.get("loss_hook_sha256") != center.get("loss_hook_sha256")
                    or origin.get("resolved_config", {}).get("seed") != 42
                    or origin.get("source_role") != "candidate"
                    or origin.get("source_is_hypothesis") is not True
                    or origin.get("origin_effective_stage")
                    != EFFECTIVE_STAGES["loss_lr_refine"]
                    or origin.get("code_bundle_sha256") != execution_source
                    or origin.get("expected_source_sha256") != execution_source
                ):
                    raise ExistingAdaptiveLockConflictError(
                        "Confirmation LR candidate lineage differs"
                    )
                expected = deepcopy(dict(center["resolved_config"]))
                multiplier = float(origin["resolved_config"]["learning_rate"]) / float(
                    center["resolved_config"]["learning_rate"]
                )
                expected["learning_rate"] = float(
                    center["resolved_config"]["learning_rate"]
                ) * multiplier
                if origin["resolved_config"] != expected:
                    raise ExistingAdaptiveLockConflictError(
                        "Confirmation LR candidate changed a non-LR config"
                    )
                multipliers.append(multiplier)
                require_completed_variant_notes(
                    origin,
                    producer_mode="loss_lr_refine",
                    lineage=[center, primary_anchor],
                )
            if multipliers != [0.5, 2.0] or len(
                {origin["code_bundle_sha256"] for origin in lr_evidence}
            ) != 1:
                raise ExistingAdaptiveLockConflictError(
                    "Confirmation LR candidate multiplier/source differs"
                )
            best_alt = min(
                lr_evidence,
                key=lambda origin: (
                    -float(origin["iid_macro_ap"]),
                    float(origin["resolved_config"]["learning_rate"]),
                ),
            )
            tie = float(plan["selection_protocol"]["practical_tie_margin"])
            if _strict_delta_greater(
                float(best_alt["iid_macro_ap"]),
                float(center["iid_macro_ap"]),
                tie,
            ):
                first = best_alt
        second_plan = loss_stage["confirmation_finalists"]["second"]
        remaining = [
            origin
            for origin in primary_evidence
            if origin["loss_variant"] not in {"bce", first["loss_variant"]}
            and _strict_delta_greater(
                float(origin["iid_macro_ap"]),
                float(tuned["iid_macro_ap"]),
                float(second_plan["minimum_raw_iid_delta_vs_tuned_bce"]),
            )
            and _delta_at_most(
                float(first["iid_macro_ap"]),
                float(origin["iid_macro_ap"]),
                float(second_plan["maximum_iid_gap_from_first"]),
            )
        ]
        second = (
            min(
                remaining,
                key=lambda origin: (
                    -float(origin["iid_macro_ap"]),
                    _loss_rank(plan, str(origin["loss_variant"])),
                    str(origin["experiment"]),
                ),
            )
            if remaining
            else None
        )
        role_origins: list[tuple[str, Mapping[str, Any]]] = [
            ("current_protocol_baseline_recipe", baseline),
            ("selected_regularized_bce_recipe", tuned),
            ("loss_finalist_1", first),
        ]
        if second is not None:
            role_origins.append(("loss_finalist_2", second))
        groups_by_family: dict[str, dict[str, Any]] = {}
        for role, origin in role_origins:
            group = groups_by_family.setdefault(
                str(origin["recipe_family_sha256"]),
                {
                    "recipe_group_id": f"recipe_{origin['recipe_family_sha256'][:16]}",
                    "recipe_family_sha256": origin["recipe_family_sha256"],
                    "roles": [],
                    "origin_seed42_id": origin["origin_id"],
                    "loss_variant": origin["loss_variant"],
                    "loss_hook_sha256": origin["loss_hook_sha256"],
                },
            )
            group["roles"].append(role)
        expected_groups = list(groups_by_family.values())
        group_origin_by_id = {origin["origin_id"]: origin for _, origin in role_origins}
        expected_variants = []
        for group in expected_groups:
            group_origin = group_origin_by_id[group["origin_seed42_id"]]
            for seed in confirmation_stage["seeds"]:
                if seed == 42:
                    continue
                config = deepcopy(dict(group_origin["resolved_config"]))
                config["seed"] = seed
                expected_variants.append(
                    _variant(
                        plan=plan,
                        mode="confirmation",
                        config=config,
                        loss_variant=str(group_origin["loss_variant"]),
                        expected_source_sha256=execution_source,
                        origin_ids=[group_origin["origin_id"]],
                        role="confirmation_candidate",
                        is_hypothesis=False,
                        extra={
                            "recipe_group_id": group["recipe_group_id"],
                            "confirmation_roles": list(group["roles"]),
                            "matched_baseline_role": (
                                "current_protocol_baseline_recipe"
                            ),
                        },
                    )
                )
        expected_decision = {
            "evidence_contract": dict(evidence_contract),
            "first_loss_finalist_origin_id": first["origin_id"],
            "first_loss_finalist_included_even_if_seed42_below_tuned_bce": True,
            "second_loss_finalist_origin_id": second["origin_id"] if second else None,
            "second_loss_finalist_triggered": second is not None,
            "second_loss_rule": deepcopy(dict(second_plan)),
            "seed42_reuse": True,
            "seeds": [17, 42, 2026],
            "recipe_groups": expected_groups,
            "matched_seed_comparison": confirmation_stage["comparison"],
            "acceptance": deepcopy(dict(confirmation_stage["acceptance"])),
            "final_tie_break_order": deepcopy(
                confirmation_stage["final_tie_break_order"]
            ),
            "selected_checkpoint_seed": 42,
            "require_inference_runtime_check": True,
        }
        baseline_group = next(
            group
            for group in expected_groups
            if "current_protocol_baseline_recipe" in group["roles"]
        )
        expected_family = {
            "family_id": "confirmation_matched_seed_recipe_comparisons",
            "correction": "fixed_matched_seed_acceptance",
            "baseline_recipe_group_id": baseline_group["recipe_group_id"],
            "maximum_non_baseline_recipe_comparisons": 3,
            "hard_ood_selection": False,
        }
        expected_resolved = {
            "strategy": "matched_seed_confirmation",
            "reused_origins": sorted(by_id),
            "recipe_groups": expected_groups,
            "variants": expected_variants,
        }
        if (
            status != "runnable"
            or dict(family) != expected_family
            or dict(decision) != expected_decision
            or dict(resolved) != expected_resolved
        ):
            raise ExistingAdaptiveLockConflictError(
                "Confirmation semantic projection differs from frozen evidence"
            )
        return

    raise ExistingAdaptiveLockConflictError(f"Unsupported adaptive mode {mode!r}")


def read_lock(
    path: Path,
    *,
    plan: Mapping[str, Any] | None = None,
    trusted_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        resolved_path = path.resolve(strict=True)
    except OSError as error:
        raise ExistingAdaptiveLockConflictError(
            f"Adaptive lock path does not exist: {path}"
        ) from error
    payload = _load_json(path, label="adaptive lock")
    validate_lock_payload(
        payload,
        plan=plan,
        trusted_provenance=trusted_provenance,
    )
    if path.read_text(encoding="utf-8") != canonical_json_dumps(payload) + "\n":
        raise ExistingAdaptiveLockConflictError("Adaptive lock is not canonical JSON")
    return payload


def load_provenance_lock(
    path: Path,
    *,
    plan: Mapping[str, Any],
    trusted_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _load_json(path, label="provenance lock")
    if payload.get("schema_version") == SCHEMA_VERSION:
        return read_lock(
            path,
            plan=plan,
            trusted_provenance=trusted_provenance,
        )
    if payload.get("schema_version") == 1:
        try:
            return generator.load_stage_lock(
                path,
                plan=plan,
                base_config=_base_config(),
            )
        except (generator.CampaignConfigError, OSError) as error:
            raise AdaptiveMaterializationError(
                f"Strict schema-v1 provenance validation failed: {error}"
            ) from error
    raise AdaptiveMaterializationError("Unsupported provenance lock schema")


def _write_once(
    path: Path,
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    mode: str,
    trusted_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_trusted_provenance_once(path, trusted_provenance, plan=plan)
    serialized = canonical_json_dumps(payload) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        existing = read_lock(
            path,
            plan=plan,
            trusted_provenance=trusted_provenance,
        )
        if existing.get("mode") != mode or existing != payload:
            raise ExistingAdaptiveLockConflictError(
                "Another process created a different immutable adaptive lock"
            )
        return existing
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    return dict(payload)


def _existing_replay(
    output_path: Path,
    *,
    plan: Mapping[str, Any],
    mode: str,
    prerequisite_locks: Sequence[Mapping[str, Any]] | None = None,
    prerequisite_lock_paths: Sequence[Path] | None = None,
    trusted_provenance: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not output_path.exists():
        return None
    existing = read_lock(
        output_path,
        plan=plan,
        trusted_provenance=trusted_provenance,
    )
    if existing.get("mode") != mode:
        raise ExistingAdaptiveLockConflictError(
            f"Existing output is mode {existing.get('mode')!r}, expected {mode!r}"
        )
    if prerequisite_locks is not None and (
        prerequisite_lock_paths is None
        or existing.get("prerequisites")
        != _prerequisite_refs(prerequisite_locks, prerequisite_lock_paths)
    ):
        raise ExistingAdaptiveLockConflictError(
            "Existing output cites different immutable prerequisites"
        )
    return existing


def _budget_documents(
    documents: Iterable[Mapping[str, Any]],
    locks: Iterable[Mapping[str, Any]],
    origins: Iterable[Mapping[str, Any]],
) -> set[str]:
    slugs: set[str] = set()
    owners: dict[str, str] = {}

    def add(value: Any, *, owner: Any = None) -> None:
        slug = _require_text(value, label="budget kernel slug")
        owner_text = str(owner).strip() if owner is not None else ""
        previous = owners.get(slug)
        if previous and owner_text and previous != owner_text:
            raise AdaptiveMaterializationError(
                f"Kernel slug {slug!r} belongs to both {previous!r} and {owner_text!r}"
            )
        if owner_text:
            owners[slug] = owner_text
        slugs.add(slug)

    for document in documents:
        budget = document.get("budget")
        if isinstance(budget, Mapping):
            for key in ("unique_kernel_slugs", "all_unique_kernel_slugs_after"):
                values = budget.get(key, [])
                if isinstance(values, list):
                    for value in values:
                        add(value)
        rows = document.get("runs", [])
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, Mapping) and row.get("completed") is True:
                    add(row.get("kernel_slug"), owner=row.get("experiment"))
    for lock in locks:
        budget = lock.get("budget")
        if isinstance(budget, Mapping) and isinstance(
            budget.get("all_unique_kernel_slugs_after"), list
        ):
            for value in budget["all_unique_kernel_slugs_after"]:
                add(value)
        for key in ("parent", "extension_source"):
            value = lock.get(key)
            if isinstance(value, Mapping) and value.get("kernel_slug"):
                add(value["kernel_slug"], owner=value.get("experiment"))
        for value in lock.get("origins", []) if isinstance(lock.get("origins"), list) else []:
            if isinstance(value, Mapping) and value.get("kernel_slug"):
                add(value["kernel_slug"], owner=value.get("experiment"))
        resolved = lock.get("resolved_stage", {})
        variants = resolved.get("variants", []) if isinstance(resolved, Mapping) else []
        for value in variants if isinstance(variants, list) else []:
            if isinstance(value, Mapping) and value.get("kernel_slug"):
                add(value["kernel_slug"], owner=value.get("experiment"))
        prior = lock.get("prior_entries", [])
        for value in prior if isinstance(prior, list) else []:
            if isinstance(value, Mapping) and value.get("kernel_slug"):
                add(value["kernel_slug"], owner=value.get("experiment"))
    for origin in origins:
        add(origin.get("kernel_slug"), owner=origin.get("experiment"))
    return slugs


def _budget(
    *,
    plan: Mapping[str, Any],
    documents: Sequence[Mapping[str, Any]],
    document_paths: Sequence[Path],
    prerequisite_locks: Sequence[Mapping[str, Any]],
    prerequisite_lock_paths: Sequence[Path],
    origins: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(documents) != len(document_paths):
        raise AdaptiveMaterializationError(
            "Every budget document requires one immutable source path"
        )
    if len(prerequisite_locks) != len(prerequisite_lock_paths):
        raise AdaptiveMaterializationError(
            "Every budget prerequisite requires one immutable lock path"
        )
    prior = _budget_documents(documents, prerequisite_locks, origins)
    new = {_require_text(row.get("kernel_slug"), label="new kernel_slug") for row in variants}
    overlap = prior & new
    if overlap:
        raise AdaptiveMaterializationError(
            f"New adaptive variants reuse prior kernel slugs: {sorted(overlap)}"
        )
    all_after = prior | new
    hard_limit = int(plan["budget"]["maximum_total_kernels"])
    if len(all_after) > hard_limit:
        raise AdaptiveMaterializationError(
            f"Kernel budget would be {len(all_after)}, above hard cap {hard_limit}"
        )
    document_snapshots_by_sha: dict[str, dict[str, Any]] = {}
    for document, document_path in zip(documents, document_paths):
        bound_document_path = Path(
            _bound_file_path(document_path, label="budget source document")
        )
        bound_document = _load_json(
            bound_document_path, label="bound budget source document"
        )
        if bound_document != dict(document):
            raise AdaptiveMaterializationError(
                "Budget source mapping differs from its immutable document"
            )
        snapshot = {
            "document_sha256": _summary_document_sha(document),
            "document_path": str(bound_document_path),
            "kernel_slugs": sorted(_budget_documents([document], [], [])),
        }
        previous = document_snapshots_by_sha.get(snapshot["document_sha256"])
        if previous is not None and previous["kernel_slugs"] != snapshot["kernel_slugs"]:
            raise AdaptiveMaterializationError(
                "The same history-document SHA produced different kernel ledgers"
            )
        if previous is None or snapshot["document_path"] < previous["document_path"]:
            document_snapshots_by_sha[snapshot["document_sha256"]] = snapshot
    document_snapshots = list(document_snapshots_by_sha.values())
    prerequisite_snapshots = []
    for lock, lock_path in zip(prerequisite_locks, prerequisite_lock_paths):
        prerequisite_budget = lock.get("budget")
        if lock.get("schema_version") == SCHEMA_VERSION:
            if not isinstance(prerequisite_budget, Mapping) or not isinstance(
                prerequisite_budget.get("all_unique_kernel_slugs_after"), list
            ):
                raise AdaptiveMaterializationError(
                    "Adaptive prerequisite has no frozen budget union"
                )
            prerequisite_kernel_slugs = sorted(
                set(prerequisite_budget["all_unique_kernel_slugs_after"])
            )
        else:
            prerequisite_kernel_slugs = sorted(
                _budget_documents([], [lock], [])
            )
        prerequisite_snapshots.append(
            {
                "lock_payload_sha256": str(lock["lock_payload_sha256"]),
                "lock_path": _bound_file_path(
                    lock_path, label="budget prerequisite lock"
                ),
                "budget_payload_sha256": (
                    canonical_sha256(lock["budget"])
                    if lock.get("schema_version") == SCHEMA_VERSION
                    else None
                ),
                "kernel_slugs": prerequisite_kernel_slugs,
            }
        )
    history_snapshot = {
        "source_documents": sorted(
            document_snapshots,
            key=lambda row: (row["document_sha256"], row["kernel_slugs"]),
        ),
        "prerequisite_budgets": sorted(
            prerequisite_snapshots,
            key=lambda row: row["lock_payload_sha256"],
        ),
        "frozen_origin_kernel_slugs": sorted(
            {_require_text(origin.get("kernel_slug"), label="origin kernel slug") for origin in origins}
        ),
        "prior_unique_kernel_slugs": sorted(prior),
    }
    return {
        "counting_identity": "kernel_slug_union",
        "plan_counting_rule": str(plan["budget"]["counting_rule"]),
        "history_snapshot": history_snapshot,
        "history_snapshot_sha256": canonical_sha256(history_snapshot),
        "prior_unique_kernel_slugs": sorted(prior),
        "new_unique_kernel_slugs": sorted(new),
        "all_unique_kernel_slugs_after": sorted(all_after),
        "prior_unique_kernels": len(prior),
        "new_unique_kernels": len(new),
        "resulting_unique_kernels": len(all_after),
        "hard_limit": hard_limit,
    }


def _require_complete_budget_history(
    documents: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    source_stage: str,
) -> None:
    ledger_attested = False
    for document in documents:
        budget = document.get("budget")
        if (
            isinstance(budget, Mapping)
            and budget.get("history_complete_through") == source_stage
            and isinstance(budget.get("unique_kernel_slugs"), list)
        ):
            ledger_attested = True
    if ledger_attested:
        return

    regularization = _stage(plan, "regularization_coordinate_search")
    order = regularization.get("execution_order")
    if not isinstance(order, list) or not order:
        raise AdaptiveMaterializationError("Regularization execution order is malformed")
    expected_stages = {
        "lr_log_line",
        "epoch_line",
        *(f"regularization_coordinate_search__{coordinate}" for coordinate in order),
    }
    completed_stages: set[str] = set()
    for document in documents:
        stages = document.get("stages")
        if not isinstance(stages, Mapping):
            continue
        for stage_name, decision in stages.items():
            if (
                stage_name in expected_stages
                and isinstance(decision, Mapping)
                and decision.get("complete") is True
                and decision.get("decision_status") == "ready"
                and decision.get("needs_boundary_extension") is not True
            ):
                completed_stages.add(str(stage_name))
    missing = expected_stages - completed_stages
    if missing:
        raise AdaptiveMaterializationError(
            "loss_primary requires either an attested kernel ledger or all completed "
            f"pre-loss stage summaries; missing {sorted(missing)}"
        )


def _adaptive_summary_lock_hashes(summary: Mapping[str, Any]) -> set[str]:
    values = summary.get("execution_lock_sha256s")
    if isinstance(values, list):
        result = {str(value) for value in values}
    else:
        one = summary.get("execution_lock_sha256")
        result = {str(one)} if one is not None else set()
    for value in result:
        _require_sha(value, label="summary execution lock SHA")
    return result


def validate_execution_summary(
    summary: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    required_locks: Sequence[Mapping[str, Any]],
) -> None:
    _summary_document_sha(summary)
    if summary.get("campaign") != plan.get("campaign"):
        raise AdaptiveMaterializationError("Execution summary belongs to another campaign")
    if summary.get("execution_status") != "complete":
        raise AdaptiveMaterializationError("Execution summary is not complete")
    expected = {
        str(lock["lock_payload_sha256"])
        for lock in required_locks
        if lock.get("schema_version") == SCHEMA_VERSION
        and lock.get("execution_status") == "runnable"
    }
    if _adaptive_summary_lock_hashes(summary) != expected:
        raise AdaptiveMaterializationError(
            "Execution summary does not cite exactly the required adaptive locks"
        )
    rows = summary.get("runs")
    if not isinstance(rows, list):
        raise AdaptiveMaterializationError("Execution summary has no runs list")
    completed_ids = [
        str(row.get("run_id"))
        for row in rows
        if isinstance(row, Mapping) and row.get("completed") is True
    ]
    if len(completed_ids) != len(set(completed_ids)):
        raise AdaptiveMaterializationError("Execution summary has duplicate run IDs")


def _row_for(
    summary: Mapping[str, Any], *, experiment: str, run_id: str | None = None
) -> Mapping[str, Any]:
    rows = summary.get("runs")
    if not isinstance(rows, list):
        raise AdaptiveMaterializationError("Summary has no runs list")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("experiment") == experiment
        and (run_id is None or row.get("run_id") == run_id)
    ]
    if len(matches) != 1:
        raise AdaptiveMaterializationError(
            f"Expected one summary row for {experiment!r}/{run_id or '*'}, found {len(matches)}"
        )
    return matches[0]


def _deduplicate_origins(origins: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for origin in origins:
        origin_id = str(origin["origin_id"])
        value = deepcopy(dict(origin))
        if origin_id in by_id and by_id[origin_id] != value:
            raise AdaptiveMaterializationError(f"Origin {origin_id!r} changed across inputs")
        by_id[origin_id] = value
    return [by_id[key] for key in sorted(by_id)]


def _resolve_lock_runs(
    *,
    plan: Mapping[str, Any],
    summary: Mapping[str, Any],
    lock: Mapping[str, Any],
    artifacts_dir: Path,
    provenance_locks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for frozen in lock.get("origins", []):
        row = _row_for(
            summary,
            experiment=str(frozen["experiment"]),
            run_id=str(frozen["run_id"]),
        )
        origin = resolve_origin(
            plan=plan,
            row=row,
            artifacts_dir=artifacts_dir,
            provenance_locks=provenance_locks,
        )
        if origin != frozen:
            raise AdaptiveMaterializationError(
                f"Reused origin {origin['experiment']!r} differs from its immutable lock"
            )
        result.append(origin)
    if lock.get("execution_status") == "runnable":
        for variant in lock["resolved_stage"]["variants"]:
            row = _row_for(summary, experiment=str(variant["experiment"]))
            origin = resolve_origin(
                plan=plan,
                row=row,
                artifacts_dir=artifacts_dir,
                provenance_locks=provenance_locks,
            )
            if (
                origin["recipe_sha256"] != variant["expected_recipe_sha256"]
                or origin["recipe_family_sha256"]
                != variant["expected_recipe_family_sha256"]
                or origin["loss_variant"] != variant["loss_variant"]
                or origin["loss_hook_sha256"]
                != variant["expected_loss_hook_sha256"]
                or origin["code_bundle_sha256"]
                != variant["expected_source_sha256"]
                or origin["resolved_config"] != variant["resolved_config"]
                or origin["source_role"] != variant["role"]
                or origin["source_is_hypothesis"]
                is not variant["is_hypothesis"]
            ):
                raise AdaptiveMaterializationError(
                    f"Completed run {origin['experiment']!r} differs from its lock recipe"
                )
            result.append(origin)
    return _deduplicate_origins(result)


def _prerequisite_refs(
    locks: Sequence[Mapping[str, Any]], lock_paths: Sequence[Path]
) -> list[dict[str, Any]]:
    if len(locks) != len(lock_paths):
        raise AdaptiveMaterializationError(
            "Every prerequisite lock requires one immutable lock path"
        )
    return sorted(
        (
            _lock_ref(lock, lock_path=lock_path)
            for lock, lock_path in zip(locks, lock_paths)
        ),
        key=lambda row: (row["schema_version"], row["effective_stage"], row["lock_payload_sha256"]),
    )


def _base_payload(
    *,
    plan: Mapping[str, Any],
    mode: str,
    status: str,
    expected_source_sha256: str,
    source_stage: str,
    summary: Mapping[str, Any],
    locks: Sequence[Mapping[str, Any]],
    lock_paths: Sequence[Path],
    origins: Sequence[Mapping[str, Any]],
    family: Mapping[str, Any],
    resolved_stage: Mapping[str, Any],
    budget: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    frozen_origins = _deduplicate_origins(origins)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": RECEIPT_KIND if status == "skipped" else LOCK_KIND,
        "campaign": plan["campaign"],
        "mode": mode,
        "source_stage": source_stage,
        "target_stage": (
            "confirmation" if mode == "confirmation" else "special_loss_screen"
        ),
        "effective_stage": EFFECTIVE_STAGES[mode],
        "execution_status": status,
        "expected_source_sha256": _require_sha(
            expected_source_sha256, label="execution source SHA"
        ),
        "source_plan_sha256": canonical_sha256(plan),
        "decision_inputs_summary_sha256": _summary_document_sha(summary),
        "selection_metric": "iid_macro_ap",
        "prerequisites": _prerequisite_refs(locks, lock_paths),
        "family": deepcopy(dict(family)),
        "origins": frozen_origins,
        "decision_evidence_sha256": canonical_sha256(
            {"origins": frozen_origins}
        ),
        "decision": deepcopy(dict(decision)),
        "resolved_stage": deepcopy(dict(resolved_stage)),
        "budget": deepcopy(dict(budget)),
    }


def _require_modes(
    locks: Sequence[Mapping[str, Any]], required: Sequence[str]
) -> dict[str, Mapping[str, Any]]:
    found: dict[str, Mapping[str, Any]] = {}
    for lock in locks:
        mode = lock.get("mode")
        if mode in required:
            if mode in found:
                raise AdaptiveMaterializationError(f"Duplicate prerequisite mode {mode!r}")
            found[str(mode)] = lock
    missing = set(required) - set(found)
    if missing:
        raise AdaptiveMaterializationError(
            f"Missing prerequisite adaptive locks: {sorted(missing)}"
        )
    return found


def materialize_loss_primary_lock(
    *,
    plan: Mapping[str, Any],
    summary: Mapping[str, Any],
    summary_path: Path,
    artifacts_dir: Path,
    prerequisite_locks: Sequence[Mapping[str, Any]],
    prerequisite_lock_paths: Sequence[Path],
    history_documents: Sequence[Mapping[str, Any]],
    history_document_paths: Sequence[Path],
    output_path: Path,
    source_stage: str | None = None,
) -> dict[str, Any]:
    _validate_adaptive_plan(plan)
    trusted_provenance = _trusted_for_materialization(
        plan=plan,
        output_path=output_path,
        source_document_paths=[summary_path, *history_document_paths],
        prerequisite_lock_paths=prerequisite_lock_paths,
        artifacts_dir=artifacts_dir,
    )
    replay = _existing_replay(
        output_path,
        plan=plan,
        mode="loss_primary",
        prerequisite_locks=prerequisite_locks,
        prerequisite_lock_paths=prerequisite_lock_paths,
        trusted_provenance=trusted_provenance,
    )
    if replay is not None:
        return replay
    expected_source = axis_materializer.expected_source_stage(
        plan, target_stage="special_loss_screen", coordinate=None
    )
    source_stage = source_stage or expected_source
    if source_stage != expected_source:
        raise AdaptiveMaterializationError(
            f"loss_primary source must be {expected_source!r}, got {source_stage!r}"
        )
    matching_v1 = [
        lock
        for lock in prerequisite_locks
        if lock.get("schema_version") == 1 and lock.get("effective_stage") == source_stage
    ]
    if len(matching_v1) != 1:
        raise AdaptiveMaterializationError(
            "loss_primary requires the exact final-coordinate schema-v1 lock"
        )
    try:
        snapshot, row = axis_materializer.source_snapshot(
            summary,
            campaign_name=str(plan["campaign"]),
            source_stage=source_stage,
        )
    except axis_materializer.StageMaterializationError as error:
        raise AdaptiveMaterializationError(str(error)) from error
    parent = resolve_origin(
        plan=plan,
        row=row,
        artifacts_dir=artifacts_dir,
        provenance_locks=prerequisite_locks,
    )
    if parent["loss_variant"] != "bce" or parent["loss_hook_sha256"] != generator.LOSS_VARIANT_SHA256["bce"]:
        raise AdaptiveMaterializationError("Loss primary anchor must be exact BCE")
    if parent["resolved_config"].get("seed") != 42:
        raise AdaptiveMaterializationError("Loss primary anchor must reuse seed 42")
    _require_complete_budget_history(
        [summary, *history_documents], plan=plan, source_stage=source_stage
    )

    loss_stage = _stage(plan, "special_loss_screen")
    execution_source_sha256 = _current_source_sha256()
    family_plan = loss_stage["families"]["primary_loss_screen"]
    maximum = int(family_plan["maximum_hypotheses"])
    variants = [
        _variant(
            plan=plan,
            mode="loss_primary",
            config=parent["resolved_config"],
            loss_variant=name,
            expected_source_sha256=execution_source_sha256,
            origin_ids=[parent["origin_id"]],
            family_size=maximum,
            extra={"loss_declaration_rank": _loss_rank(plan, name)},
        )
        for name in loss_stage["loss_variants"]
        if name != "bce"
    ]
    family = {
        "family_id": "special_loss_primary_seed42",
        "correction": "holm",
        "anchor_origin_id": parent["origin_id"],
        "planned_candidate_hypotheses": len(variants),
        "reserved_conditional_hypotheses": int(
            family_plan["reserved_conditional_combinations"]
        ),
        "maximum_hypotheses": maximum,
        "reserved_slot_state": "p_equals_1_until_overlay_decision",
    }
    budget = _budget(
        plan=plan,
        documents=[summary, *history_documents],
        document_paths=[summary_path, *history_document_paths],
        prerequisite_locks=prerequisite_locks,
        prerequisite_lock_paths=prerequisite_lock_paths,
        origins=[parent],
        variants=variants,
    )
    if not 18 <= budget["prior_unique_kernels"] <= 22:
        raise AdaptiveMaterializationError(
            "Pre-loss campaign history must contain 18 to 22 unique kernels"
        )
    payload = _with_hash(
        _base_payload(
            plan=plan,
            mode="loss_primary",
            status="runnable",
            expected_source_sha256=execution_source_sha256,
            source_stage=source_stage,
            summary=summary,
            locks=prerequisite_locks,
            lock_paths=prerequisite_lock_paths,
            origins=[parent],
            family=family,
            resolved_stage={
                "strategy": loss_stage["strategy"],
                "reused_origins": [parent["origin_id"]],
                "variants": variants,
            },
            budget=budget,
            decision={
                "anchor_origin_id": parent["origin_id"],
                "anchor_loss_variant": "bce",
                "seed": 42,
                "overlay_slot_reserved": True,
                "source_stage_snapshot_sha256": canonical_sha256(snapshot),
            },
        )
    )
    validate_lock_payload(
        payload,
        plan=plan,
        trusted_provenance=trusted_provenance,
    )
    return _write_once(
        output_path,
        payload,
        plan=plan,
        mode="loss_primary",
        trusted_provenance=trusted_provenance,
    )


def materialize_loss_overlay_lock(
    *,
    plan: Mapping[str, Any],
    summary: Mapping[str, Any],
    summary_path: Path,
    artifacts_dir: Path,
    prerequisite_locks: Sequence[Mapping[str, Any]],
    prerequisite_lock_paths: Sequence[Path],
    history_documents: Sequence[Mapping[str, Any]],
    history_document_paths: Sequence[Path],
    output_path: Path,
) -> dict[str, Any]:
    _validate_adaptive_plan(plan)
    replay_pairs = [
        (lock, path)
        for lock, path in zip(prerequisite_locks, prerequisite_lock_paths)
        if lock.get("mode") == "loss_primary"
    ]
    trusted_provenance = _trusted_for_materialization(
        plan=plan,
        output_path=output_path,
        source_document_paths=[summary_path, *history_document_paths],
        prerequisite_lock_paths=[path for _, path in replay_pairs],
        artifacts_dir=artifacts_dir,
    )
    replay = _existing_replay(
        output_path,
        plan=plan,
        mode="loss_overlay",
        prerequisite_locks=[lock for lock, _ in replay_pairs],
        prerequisite_lock_paths=[path for _, path in replay_pairs],
        trusted_provenance=trusted_provenance,
    )
    if replay is not None:
        return replay
    required = _require_modes(prerequisite_locks, ["loss_primary"])
    primary = required["loss_primary"]
    validate_execution_summary(summary, plan=plan, required_locks=[primary])
    evidence = _resolve_lock_runs(
        plan=plan,
        summary=summary,
        lock=primary,
        artifacts_dir=artifacts_dir,
        provenance_locks=prerequisite_locks,
    )
    anchor_id = str(primary["family"]["anchor_origin_id"])
    by_id = {origin["origin_id"]: origin for origin in evidence}
    anchor = by_id[anchor_id]
    by_loss = {origin["loss_variant"]: origin for origin in evidence}
    combo_plan = _stage(plan, "special_loss_screen")["conditional_combination"]
    balance_names = [
        name
        for name in _stage(plan, "special_loss_screen")["loss_variants"]
        if name in combo_plan["variants_by_balance"]
    ]
    if any(name not in by_loss for name in [*balance_names, "focal_bce_gamma2_scale4"]):
        raise AdaptiveMaterializationError("Primary summary lacks complete loss evidence")
    best_balance = min(
        (by_loss[name] for name in balance_names),
        key=lambda origin: (-origin["iid_macro_ap"], balance_names.index(origin["loss_variant"])),
    )
    focal = by_loss["focal_bce_gamma2_scale4"]
    balance_delta = best_balance["iid_macro_ap"] - anchor["iid_macro_ap"]
    focal_delta = focal["iid_macro_ap"] - anchor["iid_macro_ap"]
    threshold = float(combo_plan["minimum_raw_delta_vs_tuned_bce"])
    triggered = _strict_delta_greater(
        best_balance["iid_macro_ap"], anchor["iid_macro_ap"], threshold
    ) and _strict_delta_greater(
        focal["iid_macro_ap"], anchor["iid_macro_ap"], threshold
    )
    combo_name = combo_plan["variants_by_balance"][best_balance["loss_variant"]]
    execution_source_sha256 = _current_source_sha256()
    variants = (
        [
            _variant(
                plan=plan,
                mode="loss_overlay",
                config=anchor["resolved_config"],
                loss_variant=combo_name,
                expected_source_sha256=execution_source_sha256,
                origin_ids=[anchor["origin_id"], best_balance["origin_id"], focal["origin_id"]],
                family_size=5,
                extra={
                    "primary_family_slot": 5,
                    "loss_declaration_rank": _loss_rank(plan, combo_name),
                },
            )
        ]
        if triggered
        else []
    )
    origins = evidence
    budget = _budget(
        plan=plan,
        documents=[summary, *history_documents],
        document_paths=[summary_path, *history_document_paths],
        prerequisite_locks=[primary],
        prerequisite_lock_paths=[
            prerequisite_lock_paths[prerequisite_locks.index(primary)]
        ],
        origins=origins,
        variants=variants,
    )
    decision = {
        "decision_id": "best_balance_x_focal_overlay",
        "rule": combo_plan["run_if"],
        "operator": "strictly_greater_than",
        "threshold": threshold,
        "best_balance_origin_id": best_balance["origin_id"],
        "best_balance_loss_variant": best_balance["loss_variant"],
        "best_balance_iid_delta_vs_tuned_bce": balance_delta,
        "focal_origin_id": focal["origin_id"],
        "focal_iid_delta_vs_tuned_bce": focal_delta,
        "triggered": triggered,
        "resolved_combination_loss_variant": combo_name if triggered else None,
    }
    family = {
        "family_id": primary["family"]["family_id"],
        "correction": "holm",
        "anchor_origin_id": anchor["origin_id"],
        "maximum_hypotheses": 5,
        "reserved_slot": 5,
        "reserved_slot_state": "materialized" if triggered else "unused_p_equals_1",
    }
    payload = _with_hash(
        _base_payload(
            plan=plan,
            mode="loss_overlay",
            status="runnable" if triggered else "skipped",
            expected_source_sha256=execution_source_sha256,
            source_stage=EFFECTIVE_STAGES["loss_primary"],
            summary=summary,
            locks=[primary],
            lock_paths=[
                prerequisite_lock_paths[prerequisite_locks.index(primary)]
            ],
            origins=origins,
            family=family,
            resolved_stage={
                "strategy": "conditional_best_balance_x_focal",
                "reused_origins": sorted(origin["origin_id"] for origin in origins),
                "variants": variants,
            },
            budget=budget,
            decision=decision,
        )
    )
    validate_lock_payload(
        payload,
        plan=plan,
        trusted_provenance=trusted_provenance,
    )
    return _write_once(
        output_path,
        payload,
        plan=plan,
        mode="loss_overlay",
        trusted_provenance=trusted_provenance,
    )


def _primary_family_evidence(
    *,
    plan: Mapping[str, Any],
    summary: Mapping[str, Any],
    artifacts_dir: Path,
    primary: Mapping[str, Any],
    overlay: Mapping[str, Any],
    provenance_locks: Sequence[Mapping[str, Any]],
    validate_summary_refs: bool = True,
) -> list[dict[str, Any]]:
    if validate_summary_refs:
        validate_execution_summary(
            summary, plan=plan, required_locks=[primary, overlay]
        )
    evidence = _resolve_lock_runs(
        plan=plan,
        summary=summary,
        lock=primary,
        artifacts_dir=artifacts_dir,
        provenance_locks=provenance_locks,
    )
    if overlay.get("execution_status") == "runnable":
        evidence.extend(
            _resolve_lock_runs(
                plan=plan,
                summary=summary,
                lock=overlay,
                artifacts_dir=artifacts_dir,
                provenance_locks=provenance_locks,
            )
        )
    else:
        evidence_by_id = {origin["origin_id"]: origin for origin in evidence}
        for frozen in overlay.get("origins", []):
            if evidence_by_id.get(frozen.get("origin_id")) != frozen:
                raise AdaptiveMaterializationError(
                    "Skipped overlay receipt has different frozen decision origins"
                )
    return _deduplicate_origins(evidence)


def _best_non_bce(
    plan: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    candidates = [origin for origin in evidence if origin["loss_variant"] != "bce"]
    if not candidates:
        raise AdaptiveMaterializationError("No completed non-BCE loss candidate")
    return min(
        candidates,
        key=lambda origin: (
            -float(origin["iid_macro_ap"]),
            _loss_rank(plan, str(origin["loss_variant"])),
            str(origin["experiment"]),
        ),
    )


def materialize_loss_lr_refine_lock(
    *,
    plan: Mapping[str, Any],
    summary: Mapping[str, Any],
    summary_path: Path,
    artifacts_dir: Path,
    prerequisite_locks: Sequence[Mapping[str, Any]],
    prerequisite_lock_paths: Sequence[Path],
    history_documents: Sequence[Mapping[str, Any]],
    history_document_paths: Sequence[Path],
    output_path: Path,
) -> dict[str, Any]:
    _validate_adaptive_plan(plan)
    replay_pairs = [
        (lock, path)
        for lock, path in zip(prerequisite_locks, prerequisite_lock_paths)
        if lock.get("mode") in {"loss_primary", "loss_overlay"}
    ]
    trusted_provenance = _trusted_for_materialization(
        plan=plan,
        output_path=output_path,
        source_document_paths=[summary_path, *history_document_paths],
        prerequisite_lock_paths=[path for _, path in replay_pairs],
        artifacts_dir=artifacts_dir,
    )
    replay = _existing_replay(
        output_path,
        plan=plan,
        mode="loss_lr_refine",
        prerequisite_locks=[lock for lock, _ in replay_pairs],
        prerequisite_lock_paths=[path for _, path in replay_pairs],
        trusted_provenance=trusted_provenance,
    )
    if replay is not None:
        return replay
    required = _require_modes(prerequisite_locks, ["loss_primary", "loss_overlay"])
    primary, overlay = required["loss_primary"], required["loss_overlay"]
    required_paths = [
        prerequisite_lock_paths[prerequisite_locks.index(lock)]
        for lock in (primary, overlay)
    ]
    evidence = _primary_family_evidence(
        plan=plan,
        summary=summary,
        artifacts_dir=artifacts_dir,
        primary=primary,
        overlay=overlay,
        provenance_locks=prerequisite_locks,
    )
    anchor = next(
        origin
        for origin in evidence
        if origin["origin_id"] == primary["family"]["anchor_origin_id"]
    )
    winner = _best_non_bce(plan, evidence)
    trigger_plan = _stage(plan, "special_loss_screen")[
        "winner_lr_refinement_trigger"
    ]
    delta = winner["iid_macro_ap"] - anchor["iid_macro_ap"]
    threshold = float(trigger_plan["minimum_iid_delta_vs_tuned_bce"])
    triggered = winner["loss_variant"] != "bce" and _strict_delta_greater(
        winner["iid_macro_ap"], anchor["iid_macro_ap"], threshold
    )
    multipliers = _stage(plan, "special_loss_screen")[
        "winner_lr_refinement_multipliers"
    ]
    variants: list[dict[str, Any]] = []
    execution_source_sha256 = _current_source_sha256()
    if triggered:
        center_lr = float(winner["resolved_config"]["learning_rate"])
        for multiplier in multipliers:
            if float(multiplier) == 1.0:
                continue
            config = deepcopy(winner["resolved_config"])
            config["learning_rate"] = center_lr * float(multiplier)
            variants.append(
                _variant(
                    plan=plan,
                    mode="loss_lr_refine",
                    config=config,
                    loss_variant=str(winner["loss_variant"]),
                    expected_source_sha256=execution_source_sha256,
                    origin_ids=[winner["origin_id"], anchor["origin_id"]],
                    family_size=2,
                    extra={
                        "learning_rate_multiplier": float(multiplier),
                        "center_learning_rate": center_lr,
                    },
                )
            )
    origins = evidence
    budget = _budget(
        plan=plan,
        documents=[summary, *history_documents],
        document_paths=[summary_path, *history_document_paths],
        prerequisite_locks=[primary, overlay],
        prerequisite_lock_paths=required_paths,
        origins=origins,
        variants=variants,
    )
    decision = {
        "decision_id": "best_non_bce_lr_refinement",
        "winner_origin_id": winner["origin_id"],
        "winner_loss_variant": winner["loss_variant"],
        "winner_iid_delta_vs_tuned_bce": delta,
        "operator": "strictly_greater_than",
        "threshold": threshold,
        "triggered": triggered,
        "center_reused": True,
        "boundary_extension": False,
    }
    family = {
        "family_id": f"special_loss_lr_refine_{winner['recipe_family_sha256'][:12]}",
        "correction": "holm",
        "anchor_origin_id": winner["origin_id"],
        "planned_candidate_hypotheses": 2,
        "materialized_candidate_hypotheses": len(variants),
        "maximum_hypotheses": 2,
    }
    payload = _with_hash(
        _base_payload(
            plan=plan,
            mode="loss_lr_refine",
            status="runnable" if triggered else "skipped",
            expected_source_sha256=execution_source_sha256,
            source_stage="special_loss_screen__primary_final",
            summary=summary,
            locks=[primary, overlay],
            lock_paths=required_paths,
            origins=origins,
            family=family,
            resolved_stage={
                "strategy": "winner_non_bce_log2_lr_line",
                "reused_origins": [winner["origin_id"]],
                "variants": variants,
            },
            budget=budget,
            decision=decision,
        )
    )
    validate_lock_payload(
        payload,
        plan=plan,
        trusted_provenance=trusted_provenance,
    )
    return _write_once(
        output_path,
        payload,
        plan=plan,
        mode="loss_lr_refine",
        trusted_provenance=trusted_provenance,
    )


def _loss_evidence_with_lr(
    *,
    plan: Mapping[str, Any],
    summary: Mapping[str, Any],
    artifacts_dir: Path,
    primary: Mapping[str, Any],
    overlay: Mapping[str, Any],
    refine: Mapping[str, Any],
    provenance_locks: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    validate_execution_summary(
        summary, plan=plan, required_locks=[primary, overlay, refine]
    )
    primary_evidence = _primary_family_evidence(
        plan=plan,
        summary=summary,
        artifacts_dir=artifacts_dir,
        primary=primary,
        overlay=overlay,
        provenance_locks=provenance_locks,
        validate_summary_refs=False,
    )
    tuned = next(
        origin
        for origin in primary_evidence
        if origin["origin_id"] == primary["family"]["anchor_origin_id"]
    )
    initial_first = _best_non_bce(plan, primary_evidence)
    first = dict(initial_first)
    all_evidence = list(primary_evidence)
    if refine.get("execution_status") == "runnable":
        if refine["decision"]["winner_origin_id"] != initial_first["origin_id"]:
            raise AdaptiveMaterializationError("LR-refine center differs from recomputed L1")
        refine_evidence = _resolve_lock_runs(
            plan=plan,
            summary=summary,
            lock=refine,
            artifacts_dir=artifacts_dir,
            provenance_locks=provenance_locks,
        )
        all_evidence.extend(refine_evidence)
        center = next(
            origin
            for origin in refine_evidence
            if origin["origin_id"] == refine["family"]["anchor_origin_id"]
        )
        variant_experiments = {
            str(variant["experiment"])
            for variant in refine["resolved_stage"]["variants"]
        }
        alternatives = [
            origin
            for origin in refine_evidence
            if origin["experiment"] in variant_experiments
        ]
        best_alt = min(
            alternatives,
            key=lambda origin: (
                -origin["iid_macro_ap"],
                next(
                    variant["learning_rate_multiplier"]
                    for variant in refine["resolved_stage"]["variants"]
                    if variant["experiment"] == origin["experiment"]
                ),
            ),
        )
        tie = float(plan["selection_protocol"]["practical_tie_margin"])
        if _strict_delta_greater(
            best_alt["iid_macro_ap"], center["iid_macro_ap"], tie
        ):
            first = dict(best_alt)
    elif refine["decision"]["winner_origin_id"] != initial_first["origin_id"]:
        raise AdaptiveMaterializationError("Skipped LR receipt froze another L1")
    else:
        evidence_by_id = {origin["origin_id"]: origin for origin in primary_evidence}
        for frozen in refine.get("origins", []):
            if evidence_by_id.get(frozen.get("origin_id")) != frozen:
                raise AdaptiveMaterializationError(
                    "Skipped LR-refine receipt has different frozen decision origins"
                )
    return _deduplicate_origins(all_evidence), tuned, first


def _baseline_origin(
    *,
    plan: Mapping[str, Any],
    summary: Mapping[str, Any],
    artifacts_dir: Path,
    provenance_locks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = summary.get("runs")
    if not isinstance(rows, list):
        raise AdaptiveMaterializationError("Baseline summary has no runs")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("role") == "current_protocol_control"
        and row.get("completed") is True
    ]
    if len(matches) != 1:
        raise AdaptiveMaterializationError(
            "Confirmation requires exactly one current-protocol control origin"
        )
    origin = resolve_origin(
        plan=plan,
        row=matches[0],
        artifacts_dir=artifacts_dir,
        provenance_locks=provenance_locks,
    )
    if origin["loss_variant"] != "bce" or origin["resolved_config"].get("seed") != 42:
        raise AdaptiveMaterializationError("Current-protocol baseline must be BCE seed 42")
    if origin["resolved_config"] != _base_config():
        raise AdaptiveMaterializationError(
            "Current-protocol baseline config differs from the frozen base recipe"
        )
    lr_stage = _stage(plan, "lr_log_line")
    controls = [
        variant
        for variant in lr_stage.get("variants", [])
        if isinstance(variant, Mapping)
        and variant.get("role") == "current_protocol_control"
    ]
    if len(controls) != 1 or origin["experiment"] != controls[0].get("experiment"):
        raise AdaptiveMaterializationError(
            "Confirmation baseline is not the declared current-protocol control"
        )
    alias = controls[0].get("provenance_alias")
    if (
        not isinstance(alias, Mapping)
        or origin["completion_notes_sha256"]
        != alias.get("accepted_completion_notes_sha256")
    ):
        raise AdaptiveMaterializationError(
            "Current-protocol control does not match its exact legacy-notes alias"
        )
    return origin


def materialize_confirmation_lock(
    *,
    plan: Mapping[str, Any],
    summary: Mapping[str, Any],
    summary_path: Path,
    baseline_summary: Mapping[str, Any],
    baseline_summary_path: Path,
    artifacts_dir: Path,
    prerequisite_locks: Sequence[Mapping[str, Any]],
    prerequisite_lock_paths: Sequence[Path],
    history_documents: Sequence[Mapping[str, Any]],
    history_document_paths: Sequence[Path],
    output_path: Path,
) -> dict[str, Any]:
    _validate_adaptive_plan(plan)
    replay_pairs = [
        (lock, path)
        for lock, path in zip(prerequisite_locks, prerequisite_lock_paths)
        if lock.get("mode")
        in {"loss_primary", "loss_overlay", "loss_lr_refine"}
    ]
    trusted_provenance = _trusted_for_materialization(
        plan=plan,
        output_path=output_path,
        source_document_paths=[
            summary_path,
            baseline_summary_path,
            *history_document_paths,
        ],
        prerequisite_lock_paths=[path for _, path in replay_pairs],
        artifacts_dir=artifacts_dir,
    )
    replay = _existing_replay(
        output_path,
        plan=plan,
        mode="confirmation",
        prerequisite_locks=[lock for lock, _ in replay_pairs],
        prerequisite_lock_paths=[path for _, path in replay_pairs],
        trusted_provenance=trusted_provenance,
    )
    if replay is not None:
        return replay
    required = _require_modes(
        prerequisite_locks, ["loss_primary", "loss_overlay", "loss_lr_refine"]
    )
    primary = required["loss_primary"]
    overlay = required["loss_overlay"]
    refine = required["loss_lr_refine"]
    required_paths = [
        prerequisite_lock_paths[prerequisite_locks.index(lock)]
        for lock in (primary, overlay, refine)
    ]
    evidence, tuned, first = _loss_evidence_with_lr(
        plan=plan,
        summary=summary,
        artifacts_dir=artifacts_dir,
        primary=primary,
        overlay=overlay,
        refine=refine,
        provenance_locks=prerequisite_locks,
    )
    baseline = _baseline_origin(
        plan=plan,
        summary=baseline_summary,
        artifacts_dir=artifacts_dir,
        provenance_locks=prerequisite_locks,
    )
    finalist_plan = _stage(plan, "special_loss_screen")["confirmation_finalists"]
    second_plan = finalist_plan["second"]
    remaining = [
        origin
        for origin in evidence
        if origin["loss_variant"] not in {"bce", first["loss_variant"]}
        and _strict_delta_greater(
            origin["iid_macro_ap"],
            tuned["iid_macro_ap"],
            float(second_plan["minimum_raw_iid_delta_vs_tuned_bce"]),
        )
        and _delta_at_most(
            first["iid_macro_ap"],
            origin["iid_macro_ap"],
            float(second_plan["maximum_iid_gap_from_first"]),
        )
    ]
    second = (
        min(
            remaining,
            key=lambda origin: (
                -origin["iid_macro_ap"],
                _loss_rank(plan, str(origin["loss_variant"])),
                origin["experiment"],
            ),
        )
        if remaining
        else None
    )

    role_origins: list[tuple[str, Mapping[str, Any]]] = [
        ("current_protocol_baseline_recipe", baseline),
        ("selected_regularized_bce_recipe", tuned),
        ("loss_finalist_1", first),
    ]
    if second is not None:
        role_origins.append(("loss_finalist_2", second))
    groups_by_family: dict[str, dict[str, Any]] = {}
    for role, origin in role_origins:
        family_sha = str(origin["recipe_family_sha256"])
        group = groups_by_family.setdefault(
            family_sha,
            {
                "recipe_group_id": f"recipe_{family_sha[:16]}",
                "recipe_family_sha256": family_sha,
                "roles": [],
                "origin_seed42_id": origin["origin_id"],
                "loss_variant": origin["loss_variant"],
                "loss_hook_sha256": origin["loss_hook_sha256"],
            },
        )
        group["roles"].append(role)
    groups = sorted(groups_by_family.values(), key=lambda group: role_origins.index(next(
        pair for pair in role_origins if pair[0] in group["roles"]
    )))
    origin_by_id = {origin["origin_id"]: origin for _, origin in role_origins}
    variants: list[dict[str, Any]] = []
    execution_source_sha256 = _current_source_sha256()
    seeds = list(_stage(plan, "confirmation")["seeds"])
    for group in groups:
        origin = origin_by_id[group["origin_seed42_id"]]
        for seed in seeds:
            if seed == 42:
                continue
            config = deepcopy(origin["resolved_config"])
            config["seed"] = seed
            variants.append(
                _variant(
                    plan=plan,
                    mode="confirmation",
                    config=config,
                    loss_variant=str(origin["loss_variant"]),
                    expected_source_sha256=execution_source_sha256,
                    origin_ids=[origin["origin_id"]],
                    role="confirmation_candidate",
                    is_hypothesis=False,
                    extra={
                        "recipe_group_id": group["recipe_group_id"],
                        "confirmation_roles": list(group["roles"]),
                        "matched_baseline_role": "current_protocol_baseline_recipe",
                    },
                )
            )
    origins = _deduplicate_origins([baseline, *evidence])
    budget = _budget(
        plan=plan,
        documents=[summary, baseline_summary, *history_documents],
        document_paths=[
            summary_path,
            baseline_summary_path,
            *history_document_paths,
        ],
        prerequisite_locks=[primary, overlay, refine],
        prerequisite_lock_paths=required_paths,
        origins=origins,
        variants=variants,
    )
    confirmation_plan = _stage(plan, "confirmation")
    evidence_by_experiment = {
        origin["experiment"]: origin for origin in evidence
    }
    primary_experiments = [
        primary["origins"][0]["experiment"],
        *(
            variant["experiment"]
            for variant in primary["resolved_stage"]["variants"]
        ),
        *(
            [overlay["resolved_stage"]["variants"][0]["experiment"]]
            if overlay["execution_status"] == "runnable"
            else []
        ),
    ]
    lr_experiments = [
        variant["experiment"] for variant in refine["resolved_stage"]["variants"]
    ]
    evidence_contract = {
        "baseline_origin_id": baseline["origin_id"],
        "tuned_bce_origin_id": tuned["origin_id"],
        "primary_family_origin_ids": [
            evidence_by_experiment[experiment]["origin_id"]
            for experiment in primary_experiments
        ],
        "lr_refinement_executed": refine["execution_status"] == "runnable",
        "lr_center_origin_id": refine["decision"]["winner_origin_id"],
        "lr_candidate_origin_ids": [
            evidence_by_experiment[experiment]["origin_id"]
            for experiment in lr_experiments
        ],
    }
    decision = {
        "evidence_contract": evidence_contract,
        "first_loss_finalist_origin_id": first["origin_id"],
        "first_loss_finalist_included_even_if_seed42_below_tuned_bce": True,
        "second_loss_finalist_origin_id": second["origin_id"] if second else None,
        "second_loss_finalist_triggered": second is not None,
        "second_loss_rule": deepcopy(dict(second_plan)),
        "seed42_reuse": True,
        "seeds": seeds,
        "recipe_groups": groups,
        "matched_seed_comparison": confirmation_plan["comparison"],
        "acceptance": deepcopy(dict(confirmation_plan["acceptance"])),
        "final_tie_break_order": deepcopy(
            confirmation_plan["final_tie_break_order"]
        ),
        "selected_checkpoint_seed": 42,
        "require_inference_runtime_check": bool(
            confirmation_plan["require_inference_runtime_check"]
        ),
    }
    family = {
        "family_id": "confirmation_matched_seed_recipe_comparisons",
        "correction": "fixed_matched_seed_acceptance",
        "baseline_recipe_group_id": next(
            group["recipe_group_id"]
            for group in groups
            if "current_protocol_baseline_recipe" in group["roles"]
        ),
        "maximum_non_baseline_recipe_comparisons": 3,
        "hard_ood_selection": False,
    }
    payload = _with_hash(
        _base_payload(
            plan=plan,
            mode="confirmation",
            status="runnable",
            expected_source_sha256=execution_source_sha256,
            source_stage="special_loss_screen__final",
            summary=summary,
            locks=[primary, overlay, refine],
            lock_paths=required_paths,
            origins=origins,
            family=family,
            resolved_stage={
                "strategy": "matched_seed_confirmation",
                "reused_origins": sorted(origin["origin_id"] for origin in origins),
                "recipe_groups": groups,
                "variants": variants,
            },
            budget=budget,
            decision=decision,
        )
    )
    validate_lock_payload(
        payload,
        plan=plan,
        trusted_provenance=trusted_provenance,
    )
    return _write_once(
        output_path,
        payload,
        plan=plan,
        mode="confirmation",
        trusted_provenance=trusted_provenance,
    )


def _discover_history(summary_path: Path) -> list[Path]:
    stage_dir = summary_path.parent / "stages"
    if not stage_dir.is_dir():
        return []
    return sorted(path for path in stage_dir.glob("*/summary.json") if path != summary_path)


def materialize(
    *,
    mode: str,
    plan_path: Path = DEFAULT_PLAN,
    summary_path: Path = DEFAULT_SUMMARY,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    prerequisite_lock_paths: Sequence[Path] = (),
    history_summary_paths: Sequence[Path] = (),
    baseline_summary_path: Path | None = None,
    output_path: Path | None = None,
    source_stage: str | None = None,
) -> dict[str, Any]:
    if mode not in MODES:
        raise AdaptiveMaterializationError(f"Unknown mode {mode!r}")
    plan = load_plan(plan_path)
    if output_path is None:
        output_path = DEFAULT_LOCKS_DIR / f"{EFFECTIVE_STAGES[mode]}.lock.json"
    lock_paths = list(prerequisite_lock_paths)
    locks = []
    for path in lock_paths:
        shallow = _shallow_lock_payload(path)
        trusted = (
            load_trusted_provenance(
                trusted_provenance_manifest_path(path), plan=plan
            )
            if shallow.get("schema_version") == SCHEMA_VERSION
            else None
        )
        locks.append(
            load_provenance_lock(
                path,
                plan=plan,
                trusted_provenance=trusted,
            )
        )
    relevant_modes = {
        "loss_primary": None,
        "loss_overlay": {"loss_primary"},
        "loss_lr_refine": {"loss_primary", "loss_overlay"},
        "confirmation": {"loss_primary", "loss_overlay", "loss_lr_refine"},
    }[mode]
    replay_pairs = [
        (lock, path)
        for lock, path in zip(locks, lock_paths)
        if relevant_modes is None or lock.get("mode") in relevant_modes
    ]
    if output_path.exists():
        trusted = load_trusted_provenance(
            trusted_provenance_manifest_path(output_path), plan=plan
        )
        frozen_source_paths = {
            str(row["expected_source_path"])
            for row in trusted["source_documents"]
        }
        mandatory_source_paths = {
            _canonical_path_without_read(
                summary_path, label="selection-summary replay path"
            )
        }
        if mode == "confirmation":
            if baseline_summary_path is None:
                raise AdaptiveMaterializationError(
                    "confirmation requires --baseline-summary"
                )
            mandatory_source_paths.add(
                _canonical_path_without_read(
                    baseline_summary_path, label="baseline-summary replay path"
                )
            )
        if not mandatory_source_paths.issubset(frozen_source_paths):
            raise ExistingAdaptiveLockConflictError(
                "Replay summary/baseline authority differs from the frozen manifest"
            )
        if history_summary_paths:
            requested_source_paths = mandatory_source_paths | set(
                _requested_authority_paths(
                    history_summary_paths, label="history-summary replay"
                )
            )
            if requested_source_paths != frozen_source_paths:
                raise ExistingAdaptiveLockConflictError(
                    "Replay history authorities differ from the frozen manifest"
                )
        requested_prerequisites = set(
            _requested_authority_paths(
                [path for _, path in replay_pairs],
                label="prerequisite-lock replay",
            )
        )
        frozen_prerequisites = {
            str(row["lock_path"]) for row in trusted["prerequisite_locks"]
        }
        if (
            requested_prerequisites != frozen_prerequisites
            or _canonical_path_without_read(
                artifacts_dir, label="artifacts replay root"
            )
            != trusted["artifacts_dir"]
        ):
            raise ExistingAdaptiveLockConflictError(
                "Replay prerequisite/artifact authority differs from the frozen manifest"
            )
        replay = _existing_replay(
            output_path,
            plan=plan,
            mode=mode,
            prerequisite_locks=[lock for lock, _ in replay_pairs],
            prerequisite_lock_paths=[path for _, path in replay_pairs],
            trusted_provenance=trusted,
        )
        if replay is None:  # pragma: no cover - guarded by output_path.exists().
            raise ExistingAdaptiveLockConflictError("Existing replay lock disappeared")
        return replay
    summary = _load_json(summary_path, label="selection summary")
    history_paths = list(history_summary_paths) or _discover_history(summary_path)
    history = [_load_json(path, label="history summary") for path in history_paths]
    common = {
        "plan": plan,
        "summary": summary,
        "summary_path": summary_path,
        "artifacts_dir": artifacts_dir,
        "prerequisite_locks": locks,
        "prerequisite_lock_paths": lock_paths,
        "history_documents": history,
        "history_document_paths": history_paths,
        "output_path": output_path,
    }
    if mode == "loss_primary":
        return materialize_loss_primary_lock(**common, source_stage=source_stage)
    if mode == "loss_overlay":
        return materialize_loss_overlay_lock(**common)
    if mode == "loss_lr_refine":
        return materialize_loss_lr_refine_lock(**common)
    if baseline_summary_path is None:
        raise AdaptiveMaterializationError("confirmation requires --baseline-summary")
    baseline = _load_json(baseline_summary_path, label="baseline summary")
    return materialize_confirmation_lock(
        **common,
        baseline_summary=baseline,
        baseline_summary_path=baseline_summary_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--prerequisite-lock", type=Path, action="append", default=[])
    parser.add_argument("--history-summary", type=Path, action="append", default=[])
    parser.add_argument("--source-stage")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or DEFAULT_LOCKS_DIR / f"{EFFECTIVE_STAGES[args.mode]}.lock.json"
    existed = output.exists()
    payload = materialize(
        mode=args.mode,
        plan_path=args.plan,
        summary_path=args.summary,
        artifacts_dir=args.artifacts_dir,
        prerequisite_lock_paths=args.prerequisite_lock,
        history_summary_paths=args.history_summary,
        baseline_summary_path=args.baseline_summary,
        output_path=output,
        source_stage=args.source_stage,
    )
    print(
        canonical_json_dumps(
            {
                "status": "reused" if existed else (
                    "skipped" if payload["execution_status"] == "skipped" else "created"
                ),
                "mode": payload["mode"],
                "effective_stage": payload["effective_stage"],
                "output": str(output),
                "lock_payload_sha256": payload["lock_payload_sha256"],
                "new_unique_kernels": payload["budget"]["new_unique_kernels"],
            }
        )
    )


if __name__ == "__main__":
    main()
