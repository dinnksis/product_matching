#!/usr/bin/env python3
"""Fail-closed authority for the MiniLM loss fast-track.

The reviewed v1 campaign is immutable. The only behavioral override supplied
here is process-local: ``special_loss_screen`` may consume the completed
``classifier_dropout`` winner instead of an unrun ``max_grad_norm`` axis. A
detached, reviewed freeze manifest pins every fast-track executable before a
zero-kernel skip receipt can be created or consumed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import continue_minilm_5ep_sft_campaign as core_controller
import create_minilm_5ep_sft_hparam_notebooks as generator
import materialize_minilm_5ep_sft_hparam_stage as axis_materializer
import materialize_minilm_5ep_sft_loss_confirmation as adaptive


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "minilm_5ep_sft_loss_fast_track_v1.json"
DEFAULT_FREEZE_MANIFEST = ROOT / "configs" / "minilm_5ep_sft_loss_fast_track_v1.freeze.json"
LEGACY_RECEIPT_FREEZE_ARCHIVE = (
    ROOT
    / "configs"
    / "minilm_5ep_sft_loss_fast_track_receipt_freeze_db7165.json"
)
DEFAULT_PLAN = ROOT / "configs" / "minilm_5ep_sft_hparam_search_v1.json"
DEFAULT_REPORT_DIR = ROOT / "reports" / "minilm_5ep_sft_hparam_search_v1"
DEFAULT_SUMMARY = DEFAULT_REPORT_DIR / "summary.json"
DEFAULT_ARTIFACTS_DIR = ROOT / "artifacts" / "kaggle"
DEFAULT_RECEIPT = DEFAULT_REPORT_DIR / "fast_track" / "max_grad_norm_skip.receipt.json"
DEFAULT_LONG_SLUG_AUDIT_RECEIPT = (
    DEFAULT_REPORT_DIR
    / "fast_track"
    / "recovery"
    / "long_slug_savekernel_absence.receipt.json"
)

POLICY_KIND = "minilm_5ep_sft_loss_fast_track_policy"
FREEZE_KIND = "minilm_5ep_sft_loss_fast_track_freeze_manifest"
RECEIPT_KIND = "minilm_5ep_sft_loss_fast_track_skip_receipt"
SOURCE_STAGE = "regularization_coordinate_search__classifier_dropout"
SKIPPED_STAGE = "regularization_coordinate_search__max_grad_norm"
LOSS_PRIMARY_STAGE = "special_loss_screen__primary"
LOSS_STAGES = (
    LOSS_PRIMARY_STAGE,
    "special_loss_screen__overlay",
    "special_loss_screen__lr_refine",
)
ALLOWED_LOSS_MODES = ("loss_primary", "loss_overlay", "loss_lr_refine")
LOSS_STAGE_BY_MODE = dict(zip(ALLOWED_LOSS_MODES, LOSS_STAGES))
SHORT_REMOTE_MODE_TOKEN = {
    "loss_primary": "lp",
    "loss_overlay": "lo",
    "loss_lr_refine": "lr",
}
KAGGLE_REMOTE_IDENTITY_MAX_LENGTH = 50
SHORT_REMOTE_FAMILY_HEX_LENGTH = 24
PRIMARY_LOSSES = (
    "balanced_binary_bce",
    "balanced_category_class_sqrt_bce",
    "balanced_category_class_bce",
    "focal_bce_gamma2_scale4",
)
FROZEN_REAL_TRAIN_DATA = {
    "train_pairs": 306_669,
    "items": 711_304,
    "positive_rate": 0.26131105524197096,
    "same_size_as_human_baseline": True,
    "train_pairs_sha256": "001bb234bb631291a17fe3822989f9b47475a47185224a5f25b88a28fc6169a3",
    "items_sha256": "5491ebbfd891a396af8c0b1a4b16b61b3ed89a19c2b84d407b059096794511e0",
    "label_source_counts": {"unspecified": 306_669},
}
FROZEN_REAL_TRAINING_REPORT = {
    "training_sampling": "none",
    "training_loss_weighting": "none",
    "training_subset": "all",
    "original_training_examples": 306_669,
    "training_unique_coverage_per_epoch": 1.0,
    "training_loss_weight_min": 1.0,
    "training_loss_weight_median": 1.0,
    "training_loss_weight_max": 1.0,
}
FROZEN_CORE_TRAINING_CONTRACT = {
    "train_data": {
        "train_pairs": 306_669,
        "items": 711_304,
        "same_size_as_human_baseline": True,
    },
    "training_report": deepcopy(FROZEN_REAL_TRAINING_REPORT),
}
POLICY_CANONICAL_SHA256 = "e32559035aae79f85a99a035f0f6a23e7206fda0f5d785839e2f0396000ffef4"
PLAN_CANONICAL_SHA256 = "611396d4fcfc2150ce62f38d08efdea1668904d9bf530f6bc5d55e2ffef31857"
LEGACY_RECEIPT_FREEZE_FILE_SHA256 = "db7165cb27abe41c2846b2bcc5b928b81750e87960d31b5915c72574a0181fbe"
LEGACY_RECEIPT_FREEZE_PAYLOAD_SHA256 = "2e4ad36f81c2d5f294454955f1e5cad6b2ae86b22b4b3ae6595d05f82f1e3717"
LONG_SLUG_AUDIT_FILE_SHA256 = "672a9d8a1bd0b449278ca709c4f3ff75ddb9aac93c464b3a1ee792fd26a5b244"
LONG_SLUG_AUDIT_PAYLOAD_SHA256 = "49423ea9fc8e14c6ac26d21fb1606cbac4fc99ef11770f362ed944e1af695abd"
REJECTED_PRIMARY_LOCK_FILE_SHA256 = "b608ce860469f18afb7d68e5cce2204f1e9690d753817f6ec86fb748d9ffe694"
REJECTED_PRIMARY_LOCK_PAYLOAD_SHA256 = "dedd2f34ef0b841c0ced0650d8493e3b52b0b34c204ae1c79f83b46c01c15f9f"
AUTHORIZATION_SCOPE = (
    "User requested on 2026-08-28 to finish classifier_dropout, skip the "
    "max_grad_norm coordinate, move directly to the declared loss screen, avoid "
    "ODS, and use only one additional seed later."
)
FROZEN_CORE_FILE_SHA256 = {
    "scripts/create_minilm_5ep_sft_hparam_notebooks.py": "1a4bb3801a046bba58ab5d317ff93da62bab94aab5cd693a4dbd11b4494ac542",
    "scripts/run_minilm_5ep_sft_hparam_kaggle.py": "be400d06a7367d42974e1c6dd4be0b901b7005a5fa0c9165f2dd65987b7fbe6b",
    "scripts/summarize_minilm_5ep_sft_hparams.py": "c5e05ab18eed92f16c4aeb5594348c103c30caecad5d429a5a17d2176a5bb1f7",
    "scripts/materialize_minilm_5ep_sft_loss_confirmation.py": "70d6f58c6850c798cc954db49c4191fcab92da613fc6b0e57c970bd80a695505",
    "scripts/continue_minilm_5ep_sft_campaign.py": "3b0dc673da6b8444c692fa64f6c8a186cfbdc4ff67fa74c44a7d9ee2d702be24",
}
FAST_TRACK_EXECUTION_PATHS = (
    "configs/minilm_5ep_sft_loss_fast_track_v1.json",
    "configs/minilm_5ep_sft_loss_fast_track_receipt_freeze_db7165.json",
    "scripts/minilm_5ep_sft_loss_fast_track_support.py",
    "scripts/materialize_minilm_5ep_sft_loss_fast_track_receipt.py",
    "scripts/materialize_minilm_5ep_sft_loss_fast_track.py",
    "scripts/create_minilm_5ep_sft_loss_fast_track_notebooks.py",
    "scripts/run_minilm_5ep_sft_loss_fast_track_kaggle.py",
    "scripts/summarize_minilm_5ep_sft_loss_fast_track.py",
    "scripts/continue_minilm_5ep_sft_loss_fast_track.py",
    "scripts/recover_minilm_5ep_sft_loss_fast_track_short_slugs.py",
)
FAST_TRACK_REVIEW_PATHS = (
    "tests/test_minilm_5ep_sft_loss_fast_track.py",
    "docs/minilm-5ep-sft-loss-fast-track.md",
)
FREEZE_SEMANTIC_CONTRACT = {
    "source_stage": SOURCE_STAGE,
    "skipped_stage": SKIPPED_STAGE,
    "skipped_coordinate_evaluated": False,
    "allowed_schema2_modes": list(ALLOWED_LOSS_MODES),
    "allowed_schema2_stages": list(LOSS_STAGES),
    "primary_new_loss_variants": list(PRIMARY_LOSSES),
    "primary_new_kernels": 4,
    "conditional_max_new_kernels": 3,
    "maximum_new_loss_kernels": 7,
    "hard_kernel_cap": 37,
    "generator_and_summarizer_lock_only": True,
    "launcher_modes": ["dry_run", "submit_wait"],
    "ods_submission_allowed": False,
    "confirmation_protocol": "external_one_additional_seed_only",
    "exact_prepared_human_train_data": deepcopy(FROZEN_REAL_TRAIN_DATA),
    "exact_unweighted_full_human_training_report": deepcopy(
        FROZEN_REAL_TRAINING_REPORT
    ),
    "short_remote_identity": {
        "affected_modes": list(ALLOWED_LOSS_MODES),
        "confirmation_unchanged": True,
        "family_sha_hex_length": SHORT_REMOTE_FAMILY_HEX_LENGTH,
        "format": "pm-m5-{lp|lo|lr}-{family_sha24}-s{seed}-v1",
        "maximum_slug_and_title_length": KAGGLE_REMOTE_IDENTITY_MAX_LENGTH,
        "slug_equals_title": True,
    },
    "legacy_receipt_freeze": {
        "archive_path": str(LEGACY_RECEIPT_FREEZE_ARCHIVE.relative_to(ROOT)),
        "file_sha256": LEGACY_RECEIPT_FREEZE_FILE_SHA256,
        "manifest_payload_sha256": LEGACY_RECEIPT_FREEZE_PAYLOAD_SHA256,
    },
    "long_slug_recovery": {
        "audit_receipt_path": str(
            DEFAULT_LONG_SLUG_AUDIT_RECEIPT.relative_to(ROOT)
        ),
        "audit_receipt_file_sha256": LONG_SLUG_AUDIT_FILE_SHA256,
        "audit_receipt_payload_sha256": LONG_SLUG_AUDIT_PAYLOAD_SHA256,
        "rejected_primary_lock_file_sha256": REJECTED_PRIMARY_LOCK_FILE_SHA256,
        "rejected_primary_lock_payload_sha256": REJECTED_PRIMARY_LOCK_PAYLOAD_SHA256,
        "kaggle_mutations": 0,
    },
}
HISTORY_STAGES = (
    "lr_log_line",
    "epoch_line",
    "regularization_coordinate_search__effective_batch",
    "regularization_coordinate_search__warmup_ratio",
    "regularization_coordinate_search__weight_decay",
    "regularization_coordinate_search__label_smoothing",
    SOURCE_STAGE,
)
REQUIRED_NORMAL_LOCK_STAGES = HISTORY_STAGES[1:]
ALLOWED_BOUNDARY_STAGES = {
    "lr_log_line",
    "epoch_line",
    "regularization_coordinate_search__weight_decay",
    "regularization_coordinate_search__label_smoothing",
}
SHA_RE = re.compile(r"[0-9a-f]{64}")
_PATCH_ACTIVE = False
_TRAINING_CONTRACT_PATCH_ACTIVE = False
_IDENTITY_PATCH_ACTIVE = False
_ORIGINAL_FROZEN_TRAINING_VALIDATOR = adaptive._validate_frozen_training_contract
_ORIGINAL_FROZEN_TRAINING_TEMPLATE = adaptive._frozen_training_contract_template
_ORIGINAL_VARIANT_IDENTITY = adaptive._variant_identity


class FastTrackError(core_controller.ControllerError):
    """The fast-track authority is missing, inconsistent, or unsafe."""


def canonical_json_dumps(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise FastTrackError(f"Non-canonical fast-track value: {error}") from error


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_dumps(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FastTrackError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FastTrackError(f"{label} must be an existing regular file: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                FastTrackError(f"Non-finite JSON value {token!r} in {label}")
            ),
        )
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as error:
        raise FastTrackError(f"Invalid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise FastTrackError(f"{label} must contain a JSON object")
    canonical_json_dumps(payload)
    return payload


def _require_exact_keys(value: Any, keys: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise FastTrackError(f"{label} schema differs from the reviewed contract")
    return value


def _require_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise FastTrackError(f"{label} must be a lowercase SHA-256")
    return value


def _require_metric(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FastTrackError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise FastTrackError(f"{label} must be finite and in [0,1]")
    return result


def _canonical_file_path(path: Path, *, label: str) -> str:
    if path.is_symlink():
        raise FastTrackError(f"{label} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise FastTrackError(f"Missing {label}: {path}") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise FastTrackError(f"{label} must be a regular non-symlink file")
    return str(resolved)


def _canonical_dir_path(path: Path, *, label: str) -> str:
    if path.is_symlink():
        raise FastTrackError(f"{label} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise FastTrackError(f"Missing {label}: {path}") from error
    if not resolved.is_dir() or resolved.is_symlink():
        raise FastTrackError(f"{label} must be a directory, not a symlink")
    return str(resolved)


def _resolved_under_root(raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise FastTrackError(f"{label} must be a non-empty repository-relative path")
    unresolved = ROOT / raw
    if unresolved.is_symlink():
        raise FastTrackError(f"{label} must not be a symlink")
    path = unresolved.resolve(strict=True)
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise FastTrackError(f"{label} escapes the repository: {raw!r}") from error
    if path.is_symlink() or not path.is_file():
        raise FastTrackError(f"{label} must resolve to a regular non-symlink file")
    return path


def load_policy(path: Path = DEFAULT_POLICY) -> dict[str, Any]:
    unresolved = path if path.is_absolute() else ROOT / path
    if unresolved.is_symlink():
        raise FastTrackError("Fast-track policy must not be a symlink")
    resolved = unresolved.resolve(strict=True)
    if resolved != DEFAULT_POLICY.resolve(strict=True):
        raise FastTrackError("Fast-track policy path is not the reviewed default path")
    policy = load_json(resolved, label="fast-track policy")
    expected_top = {
        "schema_version", "kind", "campaign", "source_plan",
        "source_plan_canonical_sha256", "authorization_scope", "source_stage",
        "skipped_coordinates", "skip_semantics", "loss_execution",
        "external_actions", "budget", "frozen_core_file_sha256",
    }
    if set(policy) != expected_top:
        raise FastTrackError("Fast-track policy schema differs from the reviewed v1 policy")
    if (
        policy.get("schema_version") != 1
        or policy.get("kind") != POLICY_KIND
        or policy.get("campaign") != "minilm_5ep_sft_hparam_search_v1"
        or policy.get("source_plan") != "configs/minilm_5ep_sft_hparam_search_v1.json"
        or policy.get("source_plan_canonical_sha256") != PLAN_CANONICAL_SHA256
        or policy.get("authorization_scope") != AUTHORIZATION_SCOPE
        or policy.get("source_stage") != SOURCE_STAGE
        or policy.get("skipped_coordinates") != ["max_grad_norm"]
        or canonical_sha256(policy) != POLICY_CANONICAL_SHA256
    ):
        raise FastTrackError("Fast-track policy identity/source/skip differs")
    if policy.get("skip_semantics") != {
        "claim_coordinate_was_evaluated": False,
        "reuse_selected_source_parent": True,
        "required_inherited_values": {"max_grad_norm": 1.0},
        "reason": "Explicit scope reduction; no max_grad_norm challenger is trained and no metric claim is made for the skipped axis.",
    }:
        raise FastTrackError("Fast-track skip semantics changed")
    if policy.get("loss_execution") != {
        "anchor_loss_variant": "bce", "reuse_anchor_without_new_kernel": True,
        "primary_new_loss_variants": list(PRIMARY_LOSSES), "primary_new_kernels": 4,
        "allow_declared_conditional_overlay": True,
        "allow_declared_loss_lr_refinement": True,
        "default_stop_after": "special_loss_screen__lr_refine",
    }:
        raise FastTrackError("Fast-track loss execution contract changed")
    if policy.get("external_actions") != {
        "kaggle_submit_requires_explicit_flag": True, "google_sheets_tab": "sft_exps",
        "require_exact_sheets_sync": True, "ods_submission_allowed": False,
    }:
        raise FastTrackError("Fast-track external-action contract changed")
    if policy.get("budget") != {
        "counting_identity": "kernel_slug_union", "hard_limit": 37,
        "expected_prior_unique_kernel_range": [18, 22], "maximum_new_loss_kernels": 7,
    }:
        raise FastTrackError("Fast-track budget contract changed")
    plan_path = _resolved_under_root(policy.get("source_plan"), label="source_plan")
    plan = load_json(plan_path, label="frozen v1 plan")
    if plan.get("campaign") != policy["campaign"] or canonical_sha256(plan) != PLAN_CANONICAL_SHA256:
        raise FastTrackError("Frozen v1 plan identity/hash differs from policy")
    regularization = next((stage for stage in plan.get("stages", []) if isinstance(stage, Mapping) and stage.get("name") == "regularization_coordinate_search"), None)
    if not isinstance(regularization, Mapping) or regularization.get("execution_order") != [
        "effective_batch", "warmup_ratio", "weight_decay", "label_smoothing", "classifier_dropout", "max_grad_norm"
    ]:
        raise FastTrackError("Frozen coordinate order is not the reviewed order")
    loss_stage = next((stage for stage in plan["stages"] if isinstance(stage, Mapping) and stage.get("name") == "special_loss_screen"), None)
    if (
        not isinstance(loss_stage, Mapping)
        or loss_stage.get("loss_variants") != ["bce", *PRIMARY_LOSSES]
        or loss_stage.get("maximum_new_runs") != 7
        or loss_stage.get("families", {}).get("primary_loss_screen", {}).get("reserved_conditional_combinations") != 1
        or loss_stage.get("families", {}).get("winner_lr_refinement", {}).get("maximum_hypotheses") != 2
    ):
        raise FastTrackError("Frozen plan loss family/budget differs")
    adaptive._validate_adaptive_plan(plan)
    if policy.get("frozen_core_file_sha256") != FROZEN_CORE_FILE_SHA256:
        raise FastTrackError("Policy frozen core source registry changed")
    for raw, expected in FROZEN_CORE_FILE_SHA256.items():
        if file_sha256(_resolved_under_root(raw, label="frozen core source")) != expected:
            raise FastTrackError(f"Frozen core source changed: {raw}")
    return policy


def policy_sha256(policy: Mapping[str, Any]) -> str:
    return canonical_sha256(policy)


def load_freeze_manifest(path: Path = DEFAULT_FREEZE_MANIFEST, *, policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    selected_policy = dict(policy) if policy is not None else load_policy()
    unresolved = path if path.is_absolute() else ROOT / path
    if unresolved.is_symlink():
        raise FastTrackError("Fast-track freeze manifest must not be a symlink")
    resolved = unresolved.resolve(strict=True)
    if resolved != DEFAULT_FREEZE_MANIFEST.resolve(strict=True):
        raise FastTrackError("Freeze manifest path is not the reviewed detached authority")
    manifest = load_json(resolved, label="fast-track freeze manifest")
    if resolved.read_text(encoding="utf-8") != canonical_json_dumps(manifest) + "\n":
        raise FastTrackError("Fast-track freeze manifest is not canonical JSON")
    _require_exact_keys(manifest, {
        "schema_version", "kind", "campaign", "authority_role",
        "policy_canonical_sha256", "semantic_contract",
        "reviewed_execution_file_sha256", "reviewed_review_file_sha256",
        "manifest_payload_sha256",
    }, label="freeze manifest")
    unhashed = dict(manifest)
    stored = unhashed.pop("manifest_payload_sha256", None)
    if (
        manifest.get("schema_version") != 1 or manifest.get("kind") != FREEZE_KIND
        or manifest.get("campaign") != selected_policy["campaign"]
        or manifest.get("authority_role") != "reviewed_trusted_local_detached_execution_authority"
        or manifest.get("policy_canonical_sha256") != policy_sha256(selected_policy)
        or manifest.get("semantic_contract") != FREEZE_SEMANTIC_CONTRACT
        or stored != canonical_sha256(unhashed)
    ):
        raise FastTrackError("Fast-track freeze manifest identity/hash differs")
    for registry, expected_paths in (
        (manifest.get("reviewed_execution_file_sha256"), FAST_TRACK_EXECUTION_PATHS),
        (manifest.get("reviewed_review_file_sha256"), FAST_TRACK_REVIEW_PATHS),
    ):
        if not isinstance(registry, Mapping) or set(registry) != set(expected_paths):
            raise FastTrackError("Freeze manifest reviewed-file registry differs")
        for raw in expected_paths:
            expected = _require_sha(registry.get(raw), label=f"freeze SHA for {raw}")
            if file_sha256(_resolved_under_root(raw, label="freeze-reviewed file")) != expected:
                raise FastTrackError(f"Reviewed fast-track file drifted: {raw}")
    return manifest


def _load_legacy_receipt_freeze_archive() -> dict[str, Any]:
    """Load the exact authority that was embedded in the existing skip receipt."""
    manifest = load_json(
        LEGACY_RECEIPT_FREEZE_ARCHIVE,
        label="legacy receipt freeze archive",
    )
    if (
        file_sha256(LEGACY_RECEIPT_FREEZE_ARCHIVE)
        != LEGACY_RECEIPT_FREEZE_FILE_SHA256
        or LEGACY_RECEIPT_FREEZE_ARCHIVE.read_text(encoding="utf-8")
        != canonical_json_dumps(manifest) + "\n"
    ):
        raise FastTrackError("Legacy receipt freeze archive bytes drifted")
    _require_exact_keys(
        manifest,
        {
            "schema_version", "kind", "campaign", "authority_role",
            "policy_canonical_sha256", "semantic_contract",
            "reviewed_execution_file_sha256", "reviewed_review_file_sha256",
            "manifest_payload_sha256",
        },
        label="legacy receipt freeze archive",
    )
    unhashed = dict(manifest)
    stored = unhashed.pop("manifest_payload_sha256", None)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != FREEZE_KIND
        or manifest.get("campaign") != "minilm_5ep_sft_hparam_search_v1"
        or manifest.get("authority_role")
        != "reviewed_trusted_local_detached_execution_authority"
        or manifest.get("policy_canonical_sha256") != POLICY_CANONICAL_SHA256
        or stored != LEGACY_RECEIPT_FREEZE_PAYLOAD_SHA256
        or stored != canonical_sha256(unhashed)
    ):
        raise FastTrackError("Legacy receipt freeze archive identity/hash differs")
    return manifest


def _lock_is_boundary(payload: Mapping[str, Any]) -> bool:
    return payload.get("transition_kind") == "conditional_boundary_extension"


def _strict_schema1_lock(reference_path: Path, *, plan: Mapping[str, Any], base_config: Mapping[str, Any]) -> dict[str, Any]:
    try:
        lock = generator.load_stage_lock(reference_path, plan=plan, base_config=base_config)
    except (generator.CampaignConfigError, OSError) as error:
        raise FastTrackError(f"Strict schema-v1 history validation failed: {error}") from error
    if lock.get("schema_version") != 1:
        raise FastTrackError("Pre-receipt history contains a non-schema-v1 lock")
    return lock


def _history_lock_reference(lock: core_controller.LockInfo) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "effective_stage": lock.effective_stage,
        "is_boundary": lock.is_boundary,
        "path": _canonical_file_path(lock.path, label="prior stage lock"),
        "lock_payload_sha256": lock.payload_sha256,
        "file_sha256": file_sha256(lock.path),
    }


def _validate_allowed_lock_identities(locks: Sequence[core_controller.LockInfo], *, require_completed_dropout: bool) -> None:
    seen: set[tuple[str, bool]] = set()
    normal: set[str] = set()
    for lock in locks:
        identity = (lock.effective_stage, lock.is_boundary)
        if identity in seen:
            raise FastTrackError("Duplicate pre-receipt stage/boundary lock identity")
        seen.add(identity)
        if lock.schema_version != 1:
            raise FastTrackError("Any schema-v2/loss/confirmation lock before receipt is forbidden")
        if lock.effective_stage == SKIPPED_STAGE:
            raise FastTrackError("A max_grad_norm lock exists before the explicit skip receipt")
        if lock.is_boundary:
            if lock.effective_stage not in ALLOWED_BOUNDARY_STAGES:
                raise FastTrackError(f"Undeclared pre-receipt boundary lock for {lock.effective_stage!r}")
        elif lock.effective_stage not in REQUIRED_NORMAL_LOCK_STAGES:
            raise FastTrackError(f"Future/unknown pre-receipt lock {lock.effective_stage!r}")
        else:
            normal.add(lock.effective_stage)
    if require_completed_dropout and normal != set(REQUIRED_NORMAL_LOCK_STAGES):
        raise FastTrackError("Receipt history is not the exact normal schema-v1 chain through dropout")


def _summary_bound_hash(document: Mapping[str, Any]) -> str | None:
    binding = document.get("stage_lock")
    if binding is None:
        return None
    if not isinstance(binding, Mapping):
        raise FastTrackError("Stage summary lock binding is malformed")
    return _require_sha(binding.get("lock_payload_sha256"), label="stage summary lock SHA")


def _validate_history_summaries(
    *,
    report_dir: Path,
    locks: Sequence[core_controller.LockInfo],
    campaign: str,
    through_stage: str,
    require_exact_through_dropout: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stage_index = HISTORY_STAGES.index(through_stage)
    expected = HISTORY_STAGES if require_exact_through_dropout else HISTORY_STAGES[: stage_index + 1]
    stages_dir = report_dir / "stages"
    observed_paths = sorted(stages_dir.glob("*/summary.json")) if stages_dir.exists() else []
    observed_names = {path.parent.name for path in observed_paths}
    bad = observed_names - set(expected)
    if bad:
        raise FastTrackError(f"Future/unknown preserved summaries exist before receipt: {sorted(bad)}")
    if require_exact_through_dropout and observed_names != set(HISTORY_STAGES):
        raise FastTrackError("Receipt requires one preserved summary for every stage through dropout")

    references: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    for stage in expected:
        summary_path = stages_dir / stage / "summary.json"
        if not summary_path.is_file():
            if require_exact_through_dropout:
                raise FastTrackError(f"Missing preserved stage summary {stage!r}")
            continue
        document = load_json(summary_path, label=f"preserved {stage} summary")
        stages = document.get("stages")
        decision = stages.get(stage) if isinstance(stages, Mapping) and len(stages) == 1 else None
        if (
            document.get("schema_version") != 1 or document.get("campaign") != campaign
            or not isinstance(decision, Mapping) or decision.get("complete") is not True
            or decision.get("decision_status") != "ready"
        ):
            raise FastTrackError(f"Preserved stage summary {stage!r} is not complete/ready schema-v1")
        bound_sha = _summary_bound_hash(document)
        stage_locks = [lock for lock in locks if lock.effective_stage == stage]
        matches = [lock for lock in stage_locks if lock.payload_sha256 == bound_sha]
        if stage == "lr_log_line" and not stage_locks and bound_sha is None:
            bound_lock: core_controller.LockInfo | None = None
        elif len(matches) == 1:
            bound_lock = matches[0]
        else:
            raise FastTrackError(f"Preserved summary {stage!r} is not bound to one allowed lock")
        boundary = next((lock for lock in stage_locks if lock.is_boundary), None)
        normal = next((lock for lock in stage_locks if not lock.is_boundary), None)
        if (boundary or normal) is not bound_lock:
            raise FastTrackError(f"Preserved summary {stage!r} does not bind its final transition")
        references.append({
            "effective_stage": stage,
            "path": _canonical_file_path(summary_path, label="preserved stage summary"),
            "document_sha256": adaptive._summary_document_sha(document),
            "file_sha256": file_sha256(summary_path),
            "bound_lock_payload_sha256": bound_sha,
            "bound_lock_is_boundary": bound_lock.is_boundary if bound_lock else None,
        })
        documents.append(document)
    return references, documents


def validate_pre_receipt_authority(
    authority: core_controller.Authority,
    *,
    report_dir: Path,
    artifacts_dir: Path,
    require_completed_dropout: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate the exact schema-v1 prefix before the skip receipt exists."""
    if authority.summary is None or authority.current_stage not in HISTORY_STAGES:
        raise FastTrackError("Pre-receipt root summary is outside the allowed schema-v1 history")
    if authority.summary.get("schema_version") != 1:
        raise FastTrackError("A schema-v2 root summary exists before the skip receipt")
    _validate_allowed_lock_identities(authority.locks, require_completed_dropout=require_completed_dropout)
    base_config = generator.cross_builder.load_training_config(generator.BASE_CONFIG_PATH)
    strict_payloads: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for lock in sorted(authority.locks, key=lambda item: str(item.path.resolve())):
        payload = _strict_schema1_lock(lock.path.resolve(strict=True), plan=authority.plan, base_config=base_config)
        if (
            payload.get("effective_stage") != lock.effective_stage
            or payload.get("lock_payload_sha256") != lock.payload_sha256
            or _lock_is_boundary(payload) is not lock.is_boundary
        ):
            raise FastTrackError("Core lock index differs from strict schema-v1 lock")
        strict_payloads.append(payload)
        references.append(_history_lock_reference(lock))

    current_index = HISTORY_STAGES.index(authority.current_stage)
    decision = authority.current_decision
    if not isinstance(decision, Mapping):
        raise FastTrackError("Pre-receipt root summary has no decision")
    complete = decision.get("complete") is True
    for lock in authority.locks:
        lock_index = HISTORY_STAGES.index(lock.effective_stage)
        if lock.is_boundary:
            if lock_index > current_index:
                raise FastTrackError("Future boundary lock exists before receipt")
            if lock_index == current_index and decision.get("needs_boundary_extension") is not True:
                raise FastTrackError("Premature boundary lock exists before receipt")
        else:
            maximum_index = current_index + (1 if complete else 0)
            if lock_index > maximum_index:
                raise FastTrackError("Future/out-of-order normal lock exists before receipt")
    if require_completed_dropout and (
        authority.current_stage != SOURCE_STAGE or not complete
        or decision.get("decision_status") != "ready"
        or decision.get("control_gate") != "passed"
        or decision.get("needs_boundary_extension") is not False
    ):
        raise FastTrackError("Dropout root decision is not complete, ready, passed, and closed")

    _, summary_documents = _validate_history_summaries(
        report_dir=report_dir, locks=authority.locks, campaign=str(authority.plan["campaign"]),
        through_stage=authority.current_stage,
        require_exact_through_dropout=require_completed_dropout,
    )
    if require_completed_dropout:
        source_lock = next((lock for lock in authority.locks if lock.effective_stage == SOURCE_STAGE and not lock.is_boundary), None)
        if source_lock is None or not core_controller.lock_has_all_local_markers(source_lock, artifacts_dir):
            raise FastTrackError("Dropout variants are not complete and exactly Sheets-synced")
    return references, strict_payloads, summary_documents


def _kernel_slugs_from_lock(lock: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    resolved = lock.get("resolved_stage")
    variants = resolved.get("variants", []) if isinstance(resolved, Mapping) else []
    for row in variants if isinstance(variants, list) else []:
        if isinstance(row, Mapping) and isinstance(row.get("kernel_slug"), str):
            result.add(str(row["kernel_slug"]))
    for key in ("parent", "extension_source"):
        row = lock.get(key)
        if isinstance(row, Mapping) and isinstance(row.get("kernel_slug"), str):
            result.add(str(row["kernel_slug"]))
    for key in ("prior_entries", "origins"):
        rows = lock.get(key, [])
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, Mapping) and isinstance(row.get("kernel_slug"), str):
                result.add(str(row["kernel_slug"]))
    budget = lock.get("budget")
    if isinstance(budget, Mapping):
        for key in ("unique_kernel_slugs", "all_unique_kernel_slugs_after"):
            values = budget.get(key, [])
            if isinstance(values, list):
                result.update(str(value) for value in values)
    return result


def _reconstruct_history_kernel_slugs(plan: Mapping[str, Any], locks: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]]) -> list[str]:
    result: set[str] = set()
    lr_stage = next(stage for stage in plan["stages"] if stage.get("name") == "lr_log_line")
    for variant in lr_stage.get("variants", []):
        if not isinstance(variant, Mapping) or not isinstance(variant.get("kernel_slug"), str):
            raise FastTrackError("Frozen LR plan has a malformed kernel identity")
        result.add(str(variant["kernel_slug"]))
    for lock in locks:
        result.update(_kernel_slugs_from_lock(lock))
    for summary in summaries:
        try:
            result.update(core_controller._summary_kernel_slugs(summary))
        except core_controller.ControllerError as error:
            raise FastTrackError(str(error)) from error
    return sorted(result)


def _write_snapshot_once(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = canonical_json_dumps(document) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_text(encoding="utf-8") != serialized:
            raise FastTrackError("Existing immutable root-summary snapshot differs")
        return
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())


def _selected_row(summary: Mapping[str, Any]) -> tuple[dict[str, Any], Mapping[str, Any]]:
    stages = summary.get("stages")
    decision = stages.get(SOURCE_STAGE) if isinstance(stages, Mapping) else None
    if not isinstance(decision, Mapping):
        raise FastTrackError("Dropout summary has no exact terminal decision")
    rows = summary.get("runs", [])
    matches = [
        row for row in rows if isinstance(row, dict)
        and row.get("experiment") == decision.get("recommended_experiment")
        and row.get("run_id") == decision.get("recommended_run_id")
    ] if isinstance(rows, list) else []
    if len(matches) != 1:
        raise FastTrackError("Dropout recommendation does not identify exactly one summary row")
    return matches[0], decision


def _short_loss_remote_slug(*, mode: str, family_sha: str, seed: Any) -> str:
    token = SHORT_REMOTE_MODE_TOKEN.get(mode)
    if token is None:
        raise FastTrackError(f"Short remote identity does not allow mode {mode!r}")
    _require_sha(family_sha, label="short remote recipe-family SHA")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise FastTrackError("Short remote identity seed must be a non-negative integer")
    slug = (
        f"pm-m5-{token}-{family_sha[:SHORT_REMOTE_FAMILY_HEX_LENGTH]}-"
        f"s{seed}-v1"
    )
    if (
        len(slug) > KAGGLE_REMOTE_IDENTITY_MAX_LENGTH
        or re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug) is None
    ):
        raise FastTrackError(f"Unsafe short Kaggle identity {slug!r}")
    return slug


def _validate_short_remote_variant(
    variant: Mapping[str, Any], *, label: str
) -> str:
    slug = variant.get("kernel_slug")
    title = variant.get("title")
    if (
        not isinstance(slug, str)
        or not isinstance(title, str)
        or title != slug
        or not 1 <= len(slug) <= KAGGLE_REMOTE_IDENTITY_MAX_LENGTH
        or len(title) > KAGGLE_REMOTE_IDENTITY_MAX_LENGTH
        or re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug) is None
    ):
        raise FastTrackError(
            f"{label} must bind one safe <=50-character slug/title identity"
        )
    return slug


def _validate_short_remote_lock(lock: Mapping[str, Any]) -> None:
    mode = str(lock.get("mode"))
    if mode not in ALLOWED_LOSS_MODES:
        raise FastTrackError("Short remote lock mode is outside loss fast-track")
    resolved = lock.get("resolved_stage")
    variants = resolved.get("variants") if isinstance(resolved, Mapping) else None
    if not isinstance(variants, list):
        raise FastTrackError("Short remote lock has no exact variant list")
    status = lock.get("execution_status")
    expected_count = {
        "loss_primary": 4,
        "loss_overlay": 1,
        "loss_lr_refine": 2,
    }[mode]
    if (
        status == "runnable" and len(variants) != expected_count
    ) or (
        status == "skipped" and variants
    ):
        raise FastTrackError("Short remote lock variant count/status differs")
    slugs: list[str] = []
    for index, variant in enumerate(variants):
        if not isinstance(variant, Mapping):
            raise FastTrackError("Short remote lock variant is not an object")
        slug = _validate_short_remote_variant(
            variant, label=f"short remote variant {index}"
        )
        expected = _short_loss_remote_slug(
            mode=mode,
            family_sha=str(variant.get("expected_recipe_family_sha256")),
            seed=variant.get("seed"),
        )
        if slug != expected:
            raise FastTrackError("Short remote variant identity differs from recipe")
        slugs.append(slug)
    if len(slugs) != len(set(slugs)):
        raise FastTrackError("Short remote loss identities collide")


@contextmanager
def _patched_short_remote_identity() -> Iterator[None]:
    """Shorten only loss-screen Kaggle identities; keep recipe identities."""
    global _IDENTITY_PATCH_ACTIVE
    if _IDENTITY_PATCH_ACTIVE:
        raise FastTrackError("Short remote identity adapter is not re-entrant")
    if adaptive._variant_identity is not _ORIGINAL_VARIANT_IDENTITY:
        raise FastTrackError("Frozen adaptive variant identity authority drifted")
    original = _ORIGINAL_VARIANT_IDENTITY

    def short_variant_identity(
        *, mode: str, loss_variant: str, config: Mapping[str, Any]
    ) -> tuple[str, str, str, str]:
        experiment, original_slug, recipe_sha, family_sha = original(
            mode=mode, loss_variant=loss_variant, config=config
        )
        if mode not in ALLOWED_LOSS_MODES:
            return experiment, original_slug, recipe_sha, family_sha
        short_slug = _short_loss_remote_slug(
            mode=mode,
            family_sha=family_sha,
            seed=config.get("seed"),
        )
        return experiment, short_slug, recipe_sha, family_sha

    _IDENTITY_PATCH_ACTIVE = True
    adaptive._variant_identity = short_variant_identity
    try:
        yield
    finally:
        changed = adaptive._variant_identity is not short_variant_identity
        adaptive._variant_identity = original
        _IDENTITY_PATCH_ACTIVE = False
        if changed:
            raise FastTrackError(
                "Frozen variant identity authority changed inside fast-track"
            )


@contextmanager
def _patched_real_training_contract_validator() -> Iterator[None]:
    """Bridge the exact real 7-field provenance into the frozen 3-field API.

    Every downloaded campaign completion records two additional prepared-frame
    hashes plus prevalence and label-source provenance.  The frozen adaptive
    validator predates those fields and rejects them because it compares the
    whole mapping.  This adapter first requires the one exact real mapping, then
    invokes that unchanged validator with only its legacy projection.  It is
    process-local, non-reentrant and restored even when validation raises.
    """
    global _TRAINING_CONTRACT_PATCH_ACTIVE
    if _TRAINING_CONTRACT_PATCH_ACTIVE:
        raise FastTrackError("Frozen training-contract adapter is not re-entrant")
    if (
        adaptive._validate_frozen_training_contract
        is not _ORIGINAL_FROZEN_TRAINING_VALIDATOR
        or adaptive._frozen_training_contract_template
        is not _ORIGINAL_FROZEN_TRAINING_TEMPLATE
        or _ORIGINAL_FROZEN_TRAINING_TEMPLATE()
        != FROZEN_CORE_TRAINING_CONTRACT
    ):
        raise FastTrackError("Frozen adaptive training-contract authority drifted")

    original_validator = _ORIGINAL_FROZEN_TRAINING_VALIDATOR
    original_template = _ORIGINAL_FROZEN_TRAINING_TEMPLATE

    def strict_real_validator(
        completion: Mapping[str, Any], *, experiment: str
    ) -> dict[str, Any]:
        train_data = completion.get("train_data")
        try:
            exact_train_data = (
                isinstance(train_data, Mapping)
                and canonical_json_dumps(dict(train_data))
                == canonical_json_dumps(FROZEN_REAL_TRAIN_DATA)
            )
        except FastTrackError:
            exact_train_data = False
        if not exact_train_data:
            raise adaptive.AdaptiveMaterializationError(
                f"{experiment} changed the exact prepared human training data"
            )

        report = completion.get("training_report")
        if not isinstance(report, Mapping):
            raise adaptive.AdaptiveMaterializationError(
                f"{experiment} has no training report"
            )
        real_report = {
            key: report.get(key) for key in FROZEN_REAL_TRAINING_REPORT
        }
        try:
            exact_report = canonical_json_dumps(real_report) == canonical_json_dumps(
                FROZEN_REAL_TRAINING_REPORT
            )
        except FastTrackError:
            exact_report = False
        if not exact_report:
            raise adaptive.AdaptiveMaterializationError(
                f"{experiment} changed frozen sampling or external sample weights"
            )

        projected = deepcopy(dict(completion))
        projected["train_data"] = deepcopy(
            FROZEN_CORE_TRAINING_CONTRACT["train_data"]
        )
        validated = original_validator(projected, experiment=experiment)
        if validated != FROZEN_CORE_TRAINING_CONTRACT:
            raise adaptive.AdaptiveMaterializationError(
                "Frozen adaptive validator returned another training contract"
            )
        return deepcopy(validated)

    _TRAINING_CONTRACT_PATCH_ACTIVE = True
    adaptive._validate_frozen_training_contract = strict_real_validator
    try:
        yield
    finally:
        validator_changed = (
            adaptive._validate_frozen_training_contract is not strict_real_validator
        )
        template_changed = (
            adaptive._frozen_training_contract_template is not original_template
        )
        adaptive._validate_frozen_training_contract = original_validator
        adaptive._frozen_training_contract_template = original_template
        _TRAINING_CONTRACT_PATCH_ACTIVE = False
        if validator_changed or template_changed:
            raise FastTrackError(
                "Frozen training-contract authority changed inside fast-track"
            )


def _resolve_origin_with_real_training_contract(
    *, plan: Mapping[str, Any], row: Mapping[str, Any], artifacts_dir: Path,
    provenance_locks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    kwargs = {
        "plan": plan,
        "row": row,
        "artifacts_dir": artifacts_dir,
        "provenance_locks": provenance_locks,
    }
    if _TRAINING_CONTRACT_PATCH_ACTIVE:
        return adaptive.resolve_origin(**kwargs)
    with _patched_real_training_contract_validator():
        return adaptive.resolve_origin(**kwargs)


def _selected_parent_payload(
    *, plan: Mapping[str, Any], summary: Mapping[str, Any], artifacts_dir: Path,
    strict_locks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    row, _ = _selected_row(summary)
    try:
        origin = _resolve_origin_with_real_training_contract(
            plan=plan, row=row, artifacts_dir=artifacts_dir, provenance_locks=strict_locks
        )
    except adaptive.AdaptiveMaterializationError as error:
        raise FastTrackError(f"Selected dropout artifact provenance failed: {error}") from error
    config = origin.get("resolved_config")
    if (
        origin.get("origin_effective_stage") != SOURCE_STAGE
        or origin.get("loss_variant") != "bce"
        or origin.get("loss_hook_sha256") != generator.LOSS_VARIANT_SHA256["bce"]
        or not isinstance(config, Mapping) or config.get("seed") != 42
        or config.get("max_grad_norm") != 1.0
    ):
        raise FastTrackError("Selected dropout origin is not the tuned seed-42 BCE parent")
    completion_path = Path(str(origin["completion_artifact_path"]))
    config_path = Path(str(origin["training_config_artifact_path"]))
    predictions_path = Path(str(origin["iid_predictions_artifact_path"]))
    sync_path = completion_path.parent / "google_sheets_sync.json"
    if not core_controller._output_is_fully_sheets_synced(completion_path.parent, experiment=str(origin["experiment"])):
        raise FastTrackError("Selected dropout origin is not exactly synced to sft_exps")
    sync = load_json(sync_path, label="selected parent Sheets sync marker")
    if (
        sync.get("status") != "synced" or sync.get("run_id") != origin["run_id"]
        or sync.get("experiment_group") != "sft" or sync.get("comparison_sheet") != "sft_exps"
    ):
        raise FastTrackError("Selected parent Sheets sync marker differs")
    artifact_paths = {
        "completion": _canonical_file_path(completion_path, label="selected completion"),
        "training_config": _canonical_file_path(config_path, label="selected training config"),
        "iid_predictions": _canonical_file_path(predictions_path, label="selected IID predictions"),
        "sheets_sync": _canonical_file_path(sync_path, label="selected Sheets sync"),
    }
    return {
        "experiment": origin["experiment"],
        "run_id": origin["run_id"],
        "kernel_slug": origin["kernel_slug"],
        "source_stage": SOURCE_STAGE,
        "source_role": origin["source_role"],
        "source_is_hypothesis": origin["source_is_hypothesis"],
        "loss_variant": origin["loss_variant"],
        "loss_hook_sha256": origin["loss_hook_sha256"],
        "seed": 42,
        "resolved_config": deepcopy(dict(config)),
        "recipe_sha256": origin["recipe_sha256"],
        "recipe_family_sha256": origin["recipe_family_sha256"],
        "code_bundle_sha256": origin["code_bundle_sha256"],
        "expected_source_sha256": origin["expected_source_sha256"],
        "iid_predictions_sha256": origin["iid_predictions_sha256"],
        "completion_sha256": origin["completion_sha256"],
        "training_config_artifact_sha256": origin["training_config_artifact_sha256"],
        "completion_notes_sha256": origin["completion_notes_sha256"],
        "iid_macro_ap": _require_metric(origin.get("iid_macro_ap"), label="selected IID macro AP"),
        "hard_macro_ap": _require_metric(row.get("hard_macro_ap"), label="selected hard macro AP"),
        "ood_macro_ap": _require_metric(row.get("ood_macro_ap"), label="selected OOD macro AP"),
        "sheets_sync_status": "synced",
        "summary_row_sha256": canonical_sha256(row),
        "artifact_paths": artifact_paths,
        "artifact_file_sha256s": {key: file_sha256(Path(value)) for key, value in artifact_paths.items()},
    }


SELECTED_PARENT_KEYS = {
    "experiment", "run_id", "kernel_slug", "source_stage", "source_role",
    "source_is_hypothesis", "loss_variant", "loss_hook_sha256", "seed",
    "resolved_config", "recipe_sha256", "recipe_family_sha256",
    "code_bundle_sha256", "expected_source_sha256", "iid_predictions_sha256",
    "completion_sha256", "training_config_artifact_sha256", "completion_notes_sha256",
    "iid_macro_ap", "hard_macro_ap", "ood_macro_ap", "sheets_sync_status",
    "summary_row_sha256", "artifact_paths", "artifact_file_sha256s",
}


def _primary_kernel_slugs(plan: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    slugs = [
        str(adaptive._variant(
            plan=plan, mode="loss_primary", config=config, loss_variant=loss_variant,
            expected_source_sha256="0" * 64, origin_ids=["receipt_anchor"],
        )["kernel_slug"])
        for loss_variant in PRIMARY_LOSSES
    ]
    if len(slugs) != 4 or len(set(slugs)) != 4:
        raise FastTrackError("Primary loss screen does not materialize four unique kernels")
    return sorted(slugs)


def _budget_payload(*, plan: Mapping[str, Any], prior_slugs: Sequence[str], selected_parent: Mapping[str, Any]) -> dict[str, Any]:
    prior = sorted(set(prior_slugs))
    if prior != list(prior_slugs):
        raise FastTrackError("Pre-loss kernel ledger is not a canonical exact union")
    primary = _primary_kernel_slugs(plan, selected_parent["resolved_config"])
    if set(primary) & set(prior):
        raise FastTrackError("Primary loss kernels collide with pre-loss history")
    primary_result = sorted(set(prior) | set(primary))
    result = {
        "counting_identity": "kernel_slug_union",
        "history_complete_through": SOURCE_STAGE,
        "unique_kernel_slugs": prior,
        "unique_kernels": len(prior),
        "primary_new_kernel_slugs": primary,
        "primary_new_kernels": 4,
        "primary_resulting_kernel_slugs": primary_result,
        "primary_resulting_kernels": len(primary_result),
        "conditional_max_new_kernels": 3,
        "maximum_new_loss_kernels": 7,
        "maximum_resulting_kernels": len(prior) + 7,
        "hard_limit": 37,
    }
    if not 18 <= len(prior) <= 22 or result["maximum_resulting_kernels"] > 37:
        raise FastTrackError("Fast-track budget is outside the reviewed range/cap")
    return result


def build_receipt(
    *, policy_path: Path = DEFAULT_POLICY,
    freeze_manifest_path: Path = DEFAULT_FREEZE_MANIFEST,
    plan_path: Path = DEFAULT_PLAN, summary_path: Path = DEFAULT_SUMMARY,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR, receipt_path: Path = DEFAULT_RECEIPT,
) -> dict[str, Any]:
    policy = load_policy(policy_path)
    freeze = load_freeze_manifest(freeze_manifest_path, policy=policy)
    if plan_path.resolve(strict=True) != DEFAULT_PLAN.resolve(strict=True):
        raise FastTrackError("Fast-track receipt must use the frozen default v1 plan")
    paths = core_controller.ControllerPaths(
        plan=plan_path, report_dir=summary_path.parent, summary=summary_path,
        locks_dir=summary_path.parent / "stage_locks", artifacts_dir=artifacts_dir,
        state_path=summary_path.parent / "controller_state.json",
        baseline_summary=summary_path.parent / "stages" / "lr_log_line" / "summary.json",
    )
    authority = core_controller.inspect_authority(paths)
    lock_refs, strict_locks, history_summaries = validate_pre_receipt_authority(
        authority, report_dir=paths.report_dir, artifacts_dir=artifacts_dir,
        require_completed_dropout=True,
    )
    root_summary = load_json(summary_path, label="root dropout summary")
    if authority.summary != root_summary:
        raise FastTrackError("Root summary changed while receipt authority was inspected")
    source_summary_path = summary_path.parent / "stages" / SOURCE_STAGE / "summary.json"
    source_summary = load_json(source_summary_path, label="preserved dropout summary")
    if source_summary != root_summary:
        raise FastTrackError("Root and preserved dropout summaries are not the same document")
    root_file_sha = file_sha256(summary_path)
    source_summary_file_sha = file_sha256(source_summary_path)
    if root_file_sha != source_summary_file_sha:
        raise FastTrackError(
            "Root and preserved dropout summaries are not the same file bytes"
        )
    root_document_sha = adaptive._summary_document_sha(root_summary)
    snapshot_path = receipt_path.parent / "authority" / f"dropout_root_summary.{root_document_sha}.json"
    _write_snapshot_once(snapshot_path, root_summary)
    selected_parent = _selected_parent_payload(
        plan=authority.plan, summary=source_summary, artifacts_dir=artifacts_dir,
        strict_locks=strict_locks,
    )
    prior_slugs = _reconstruct_history_kernel_slugs(authority.plan, strict_locks, history_summaries)
    if prior_slugs != list(authority.unique_kernel_slugs):
        raise FastTrackError("Controller kernel union differs from exact lock/summary reconstruction")
    budget = _budget_payload(plan=authority.plan, prior_slugs=prior_slugs, selected_parent=selected_parent)
    source_lock_ref = next(ref for ref in lock_refs if ref["effective_stage"] == SOURCE_STAGE and ref["is_boundary"] is False)
    summary_refs, _ = _validate_history_summaries(
        report_dir=summary_path.parent, locks=authority.locks,
        campaign=str(authority.plan["campaign"]), through_stage=SOURCE_STAGE,
        require_exact_through_dropout=True,
    )
    source_summary_ref = next(ref for ref in summary_refs if ref["effective_stage"] == SOURCE_STAGE)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": RECEIPT_KIND,
        "campaign": policy["campaign"],
        "policy_authority": {
            "path": _canonical_file_path(policy_path, label="fast-track policy"),
            "canonical_sha256": policy_sha256(policy),
        },
        "freeze_authority": {
            "path": _canonical_file_path(freeze_manifest_path, label="freeze manifest"),
            "file_sha256": file_sha256(freeze_manifest_path),
            "manifest_payload_sha256": freeze["manifest_payload_sha256"],
            "reviewed_execution_file_sha256": deepcopy(freeze["reviewed_execution_file_sha256"]),
            "reviewed_review_file_sha256": deepcopy(freeze["reviewed_review_file_sha256"]),
        },
        "source_plan": {
            "path": _canonical_file_path(plan_path, label="frozen v1 plan"),
            "canonical_sha256": canonical_sha256(authority.plan),
            "file_sha256": file_sha256(plan_path),
        },
        "source_stage": SOURCE_STAGE,
        "source_stage_summary": deepcopy(source_summary_ref),
        "source_root_summary": {
            "original_path": _canonical_file_path(summary_path, label="root dropout summary"),
            "original_file_sha256_at_materialization": root_file_sha,
            "snapshot_path": _canonical_file_path(snapshot_path, label="root summary snapshot"),
            "document_sha256": root_document_sha,
            "snapshot_file_sha256": file_sha256(snapshot_path),
        },
        "source_lock": deepcopy(source_lock_ref),
        "selected_parent": selected_parent,
        "skipped_coordinate": {
            "name": "max_grad_norm", "stage": SKIPPED_STAGE, "evaluated": False,
            "new_kernels": 0, "metric_claim": None, "inherited_value": 1.0,
            "reason": policy["skip_semantics"]["reason"],
        },
        "loss_execution": deepcopy(policy["loss_execution"]),
        "external_actions": deepcopy(policy["external_actions"]),
        "artifacts_dir": _canonical_dir_path(artifacts_dir, label="Kaggle artifacts directory"),
        "prior_stage_locks": lock_refs,
        "prior_stage_summaries": summary_refs,
        "budget": budget,
    }
    payload["summary_payload_sha256"] = canonical_sha256(payload)
    return payload


def _validate_reference_file(reference: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(reference["path"]))
    if _canonical_file_path(path, label=label) != str(path) or file_sha256(path) != reference["file_sha256"]:
        raise FastTrackError(f"{label} path/file SHA differs")
    return path


def validate_receipt(
    path: Path = DEFAULT_RECEIPT,
    *, policy: Mapping[str, Any] | None = None,
    policy_path: Path = DEFAULT_POLICY,
    freeze_manifest_path: Path = DEFAULT_FREEZE_MANIFEST,
) -> dict[str, Any]:
    selected_policy = dict(policy) if policy is not None else load_policy(policy_path)
    freeze = load_freeze_manifest(freeze_manifest_path, policy=selected_policy)
    if path.is_symlink():
        raise FastTrackError("Fast-track skip receipt must not be a symlink")
    receipt = load_json(path.resolve(strict=True), label="fast-track skip receipt")
    _require_exact_keys(receipt, {
        "schema_version", "kind", "campaign", "policy_authority", "freeze_authority",
        "source_plan", "source_stage", "source_stage_summary", "source_root_summary",
        "source_lock", "selected_parent", "skipped_coordinate", "loss_execution",
        "external_actions", "artifacts_dir", "prior_stage_locks",
        "prior_stage_summaries", "budget", "summary_payload_sha256",
    }, label="skip receipt")
    unhashed = dict(receipt)
    stored = unhashed.pop("summary_payload_sha256", None)
    if (
        receipt.get("schema_version") != 1 or receipt.get("kind") != RECEIPT_KIND
        or receipt.get("campaign") != selected_policy["campaign"]
        or stored != canonical_sha256(unhashed) or receipt.get("source_stage") != SOURCE_STAGE
    ):
        raise FastTrackError("Fast-track skip receipt identity/hash/source differs")

    policy_authority = _require_exact_keys(receipt.get("policy_authority"), {"path", "canonical_sha256"}, label="receipt policy authority")
    if (
        Path(str(policy_authority["path"])).resolve(strict=True) != DEFAULT_POLICY.resolve(strict=True)
        or policy_authority["canonical_sha256"] != policy_sha256(selected_policy)
    ):
        raise FastTrackError("Receipt policy authority differs")
    freeze_authority = _require_exact_keys(receipt.get("freeze_authority"), {
        "path", "file_sha256", "manifest_payload_sha256",
        "reviewed_execution_file_sha256", "reviewed_review_file_sha256",
    }, label="receipt freeze authority")
    authority_path_matches = (
        Path(str(freeze_authority["path"])).resolve(strict=True)
        == DEFAULT_FREEZE_MANIFEST.resolve(strict=True)
    )
    current_freeze_matches = (
        freeze_authority["file_sha256"] == file_sha256(DEFAULT_FREEZE_MANIFEST)
        and freeze_authority["manifest_payload_sha256"]
        == freeze["manifest_payload_sha256"]
        and freeze_authority["reviewed_execution_file_sha256"]
        == freeze["reviewed_execution_file_sha256"]
        and freeze_authority["reviewed_review_file_sha256"]
        == freeze["reviewed_review_file_sha256"]
    )
    legacy_freeze = _load_legacy_receipt_freeze_archive()
    legacy_freeze_matches = (
        freeze_authority["file_sha256"] == LEGACY_RECEIPT_FREEZE_FILE_SHA256
        and freeze_authority["manifest_payload_sha256"]
        == LEGACY_RECEIPT_FREEZE_PAYLOAD_SHA256
        and freeze_authority["reviewed_execution_file_sha256"]
        == legacy_freeze["reviewed_execution_file_sha256"]
        and freeze_authority["reviewed_review_file_sha256"]
        == legacy_freeze["reviewed_review_file_sha256"]
    )
    if not authority_path_matches or not (
        current_freeze_matches or legacy_freeze_matches
    ):
        raise FastTrackError("Receipt detached freeze authority differs")
    source_plan = _require_exact_keys(receipt.get("source_plan"), {"path", "canonical_sha256", "file_sha256"}, label="receipt source plan")
    plan_path = Path(str(source_plan["path"]))
    plan = load_json(plan_path, label="receipt-bound frozen plan")
    if (
        plan_path.resolve(strict=True) != DEFAULT_PLAN.resolve(strict=True)
        or source_plan["canonical_sha256"] != PLAN_CANONICAL_SHA256
        or canonical_sha256(plan) != PLAN_CANONICAL_SHA256
        or source_plan["file_sha256"] != file_sha256(plan_path)
    ):
        raise FastTrackError("Receipt source-plan binding differs")

    source_summary_probe = _require_exact_keys(receipt.get("source_stage_summary"), {
        "effective_stage", "path", "document_sha256", "file_sha256",
        "bound_lock_payload_sha256", "bound_lock_is_boundary",
    }, label="receipt source-stage summary")
    source_summary_path_probe = Path(str(source_summary_probe["path"]))
    report_dir = source_summary_path_probe.parents[2]
    if source_summary_path_probe != report_dir / "stages" / SOURCE_STAGE / "summary.json":
        raise FastTrackError("Receipt dropout summary path is not the canonical stage path")

    lock_refs = receipt.get("prior_stage_locks")
    if not isinstance(lock_refs, list) or not lock_refs:
        raise FastTrackError("Receipt prior lock ledger is missing")
    if lock_refs != sorted(lock_refs, key=lambda row: str(row.get("path", ""))):
        raise FastTrackError("Receipt prior lock ledger is not canonical path order")
    base_config = generator.cross_builder.load_training_config(generator.BASE_CONFIG_PATH)
    strict_locks: list[dict[str, Any]] = []
    identities: set[tuple[str, bool]] = set()
    for item in lock_refs:
        ref = _require_exact_keys(item, {
            "schema_version", "effective_stage", "is_boundary", "path",
            "lock_payload_sha256", "file_sha256",
        }, label="receipt prior lock reference")
        if ref["schema_version"] != 1 or not isinstance(ref["is_boundary"], bool):
            raise FastTrackError("Receipt prior lock is not exact schema-v1")
        lock_path = _validate_reference_file(ref, label="receipt prior stage lock")
        if lock_path.parent != report_dir / "stage_locks":
            raise FastTrackError("Receipt prior lock is outside the bound campaign lock directory")
        lock = _strict_schema1_lock(lock_path, plan=plan, base_config=base_config)
        identity = (str(ref["effective_stage"]), bool(ref["is_boundary"]))
        if identity in identities:
            raise FastTrackError("Receipt repeats a stage/boundary lock identity")
        identities.add(identity)
        if (
            lock.get("effective_stage") != ref["effective_stage"]
            or _lock_is_boundary(lock) is not ref["is_boundary"]
            or lock.get("lock_payload_sha256") != ref["lock_payload_sha256"]
        ):
            raise FastTrackError("Receipt lock reference differs from strict payload")
        strict_locks.append(lock)
    pseudo_locks = tuple(
        core_controller.LockInfo(
            path=Path(str(ref["path"])), payload=lock, schema_version=1,
            effective_stage=str(ref["effective_stage"]),
            payload_sha256=str(ref["lock_payload_sha256"]), execution_status="runnable",
            is_boundary=bool(ref["is_boundary"]),
            kernel_slugs=tuple(sorted(_kernel_slugs_from_lock(lock))),
        )
        for ref, lock in zip(lock_refs, strict_locks)
    )
    _validate_allowed_lock_identities(pseudo_locks, require_completed_dropout=True)

    summary_refs = receipt.get("prior_stage_summaries")
    if not isinstance(summary_refs, list) or len(summary_refs) != len(HISTORY_STAGES):
        raise FastTrackError("Receipt preserved-summary ledger is incomplete")
    if [ref.get("effective_stage") for ref in summary_refs] != list(HISTORY_STAGES):
        raise FastTrackError("Receipt preserved-summary ledger order/stages differ")
    summary_documents: list[dict[str, Any]] = []
    for item in summary_refs:
        ref = _require_exact_keys(item, {
            "effective_stage", "path", "document_sha256", "file_sha256",
            "bound_lock_payload_sha256", "bound_lock_is_boundary",
        }, label="receipt stage-summary reference")
        summary_path = _validate_reference_file(ref, label="receipt preserved stage summary")
        document = load_json(summary_path, label="receipt preserved stage summary")
        stage = str(ref["effective_stage"])
        if summary_path != report_dir / "stages" / stage / "summary.json":
            raise FastTrackError("Receipt stage-summary path differs from its bound stage")
        decision = document.get("stages", {}).get(stage)
        bound_sha = _summary_bound_hash(document)
        matching = [lock for lock in pseudo_locks if lock.effective_stage == stage and lock.payload_sha256 == bound_sha]
        bound_boundary = matching[0].is_boundary if len(matching) == 1 else None
        valid_unlocked_lr = (
            stage == "lr_log_line" and bound_sha is None
            and ref["bound_lock_payload_sha256"] is None
            and ref["bound_lock_is_boundary"] is None
        )
        if (
            document.get("schema_version") != 1
            or document.get("campaign") != selected_policy["campaign"]
            or not isinstance(decision, Mapping) or decision.get("complete") is not True
            or decision.get("decision_status") != "ready"
            or adaptive._summary_document_sha(document) != ref["document_sha256"]
            or bound_sha != ref["bound_lock_payload_sha256"]
            or (bound_boundary != ref["bound_lock_is_boundary"] and not valid_unlocked_lr)
        ):
            raise FastTrackError("Receipt stage-summary binding differs")
        summary_documents.append(document)

    source_summary_ref = source_summary_probe
    expected_source_summary_ref = next(ref for ref in summary_refs if ref["effective_stage"] == SOURCE_STAGE)
    if dict(source_summary_ref) != dict(expected_source_summary_ref):
        raise FastTrackError("Receipt source-stage summary is not the dropout ledger entry")
    source_summary = summary_documents[-1]
    root_ref = _require_exact_keys(receipt.get("source_root_summary"), {
        "original_path", "original_file_sha256_at_materialization", "snapshot_path",
        "document_sha256", "snapshot_file_sha256",
    }, label="receipt root-summary snapshot")
    original_path = Path(str(root_ref["original_path"]))
    snapshot_path = Path(str(root_ref["snapshot_path"]))
    if original_path.resolve(strict=False) != (report_dir / "summary.json").resolve(strict=False):
        raise FastTrackError("Receipt root-summary original path differs")
    snapshot = load_json(snapshot_path, label="receipt root-summary snapshot")
    if (
        _canonical_file_path(snapshot_path, label="receipt root-summary snapshot") != str(snapshot_path)
        or file_sha256(snapshot_path) != root_ref["snapshot_file_sha256"]
        or adaptive._summary_document_sha(snapshot) != root_ref["document_sha256"]
        or snapshot != source_summary
        or root_ref["document_sha256"] != source_summary_ref["document_sha256"]
    ):
        raise FastTrackError("Receipt immutable root-summary snapshot differs")
    _require_sha(root_ref["original_file_sha256_at_materialization"], label="materialized root-summary file SHA")
    if (
        root_ref["original_file_sha256_at_materialization"]
        != source_summary_ref["file_sha256"]
    ):
        raise FastTrackError(
            "Materialized root-summary file SHA is not bound to the preserved dropout summary"
        )
    expected_snapshot = (
        path.resolve(strict=True).parent
        / "authority"
        / f"dropout_root_summary.{root_ref['document_sha256']}.json"
    )
    if snapshot_path != expected_snapshot:
        raise FastTrackError("Receipt root-summary snapshot path differs from its receipt authority directory")
    current_root = load_json(original_path, label="current root campaign summary")
    current_stages = current_root.get("stages")
    current_stage = next(iter(current_stages), None) if isinstance(current_stages, Mapping) and len(current_stages) == 1 else None
    if current_root.get("schema_version") == 1:
        if (
            current_root != snapshot
            or file_sha256(original_path) != root_ref["original_file_sha256_at_materialization"]
        ):
            raise FastTrackError("Schema-v1 root summary changed after receipt materialization")
    elif (
        current_root.get("schema_version") != 2
        or current_root.get("campaign") != selected_policy["campaign"]
        or current_stage not in LOSS_STAGES
    ):
        raise FastTrackError("Current root summary is not dropout or an authorized loss stage")

    source_lock_ref = _require_exact_keys(receipt.get("source_lock"), {
        "schema_version", "effective_stage", "is_boundary", "path",
        "lock_payload_sha256", "file_sha256",
    }, label="receipt source lock")
    expected_source_lock_ref = next(ref for ref in lock_refs if ref["effective_stage"] == SOURCE_STAGE and ref["is_boundary"] is False)
    if dict(source_lock_ref) != dict(expected_source_lock_ref):
        raise FastTrackError("Receipt source lock is not the normal dropout lock")

    selected = _require_exact_keys(receipt.get("selected_parent"), SELECTED_PARENT_KEYS, label="receipt selected parent")
    for key in (
        "loss_hook_sha256", "recipe_sha256", "recipe_family_sha256", "code_bundle_sha256",
        "expected_source_sha256", "iid_predictions_sha256", "completion_sha256",
        "training_config_artifact_sha256", "completion_notes_sha256", "summary_row_sha256",
    ):
        _require_sha(selected[key], label=f"selected parent {key}")
    for key in ("iid_macro_ap", "hard_macro_ap", "ood_macro_ap"):
        _require_metric(selected[key], label=f"selected parent {key}")
    artifacts_dir = Path(str(receipt.get("artifacts_dir")))
    if _canonical_dir_path(artifacts_dir, label="receipt artifacts directory") != str(artifacts_dir):
        raise FastTrackError("Receipt artifacts directory is not canonical")
    reconstructed_selected = _selected_parent_payload(
        plan=plan, summary=source_summary, artifacts_dir=artifacts_dir,
        strict_locks=strict_locks,
    )
    if dict(selected) != reconstructed_selected:
        raise FastTrackError("Receipt selected parent differs from summary/artifact provenance")

    if receipt.get("skipped_coordinate") != {
        "name": "max_grad_norm", "stage": SKIPPED_STAGE, "evaluated": False,
        "new_kernels": 0, "metric_claim": None, "inherited_value": 1.0,
        "reason": selected_policy["skip_semantics"]["reason"],
    }:
        raise FastTrackError("Receipt skip statement differs from policy")
    if receipt.get("loss_execution") != selected_policy["loss_execution"] or receipt.get("external_actions") != selected_policy["external_actions"]:
        raise FastTrackError("Receipt loss/action scope differs from policy")
    exact_prior = _reconstruct_history_kernel_slugs(plan, strict_locks, summary_documents)
    expected_budget = _budget_payload(plan=plan, prior_slugs=exact_prior, selected_parent=selected)
    budget = _require_exact_keys(receipt.get("budget"), set(expected_budget), label="receipt budget")
    if dict(budget) != expected_budget:
        raise FastTrackError("Receipt budget/union/arithmetic differs from exact history")
    return receipt


def write_receipt_once(
    path: Path, payload: Mapping[str, Any], *, policy_path: Path = DEFAULT_POLICY,
    freeze_manifest_path: Path = DEFAULT_FREEZE_MANIFEST,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = canonical_json_dumps(payload) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        existing = validate_receipt(path, policy_path=policy_path, freeze_manifest_path=freeze_manifest_path)
        if existing != dict(payload) or path.read_text(encoding="utf-8") != serialized:
            raise FastTrackError("Existing immutable skip receipt differs")
        return existing
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    return dict(payload)


@contextmanager
def patched_loss_predecessor(
    *, policy_path: Path = DEFAULT_POLICY, receipt_path: Path = DEFAULT_RECEIPT,
    freeze_manifest_path: Path = DEFAULT_FREEZE_MANIFEST,
) -> Iterator[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Validate authority and install both narrow process-local adapters."""
    global _PATCH_ACTIVE
    if _PATCH_ACTIVE:
        raise FastTrackError("Loss predecessor fast-track context is not re-entrant")
    policy = load_policy(policy_path)
    freeze = load_freeze_manifest(freeze_manifest_path, policy=policy)
    receipt = validate_receipt(
        receipt_path, policy=policy, policy_path=policy_path,
        freeze_manifest_path=freeze_manifest_path,
    )
    original = axis_materializer.expected_source_stage

    def expected_source_stage(plan: Mapping[str, Any], *, target_stage: str, coordinate: str | None) -> str:
        if target_stage == "special_loss_screen" and coordinate is None:
            if canonical_sha256(plan) != policy["source_plan_canonical_sha256"]:
                raise axis_materializer.StageMaterializationError("Fast-track predecessor override received another plan")
            return SOURCE_STAGE
        return original(plan, target_stage=target_stage, coordinate=coordinate)

    with _patched_real_training_contract_validator(), _patched_short_remote_identity():
        _PATCH_ACTIVE = True
        axis_materializer.expected_source_stage = expected_source_stage
        try:
            yield policy, receipt, freeze
        finally:
            try:
                if axis_materializer.expected_source_stage is not expected_source_stage:
                    raise FastTrackError("Loss predecessor resolver changed inside fast-track")
            finally:
                axis_materializer.expected_source_stage = original
                _PATCH_ACTIVE = False


def load_scoped_loss_lock(
    *, plan_path: Path, stage_lock_path: Path, policy_path: Path = DEFAULT_POLICY,
    receipt_path: Path = DEFAULT_RECEIPT,
    freeze_manifest_path: Path = DEFAULT_FREEZE_MANIFEST,
) -> dict[str, Any]:
    """Strictly load one of the three authorized schema-v2 loss locks."""
    if plan_path.resolve(strict=True) != DEFAULT_PLAN.resolve(strict=True):
        raise FastTrackError("Fast-track wrappers require the frozen default plan")
    if stage_lock_path.is_symlink():
        raise FastTrackError("Fast-track stage lock must not be a symlink")
    lock_path = stage_lock_path.resolve(strict=True)
    plan = generator.load_plan(plan_path)
    base = generator.cross_builder.load_training_config(generator.BASE_CONFIG_PATH)
    with patched_loss_predecessor(
        policy_path=policy_path, receipt_path=receipt_path,
        freeze_manifest_path=freeze_manifest_path,
    ):
        try:
            lock = generator.load_campaign_lock(lock_path, plan=plan, base_config=base)
        except generator.CampaignConfigError as error:
            raise FastTrackError(f"Fast-track lock validation failed: {error}") from error
    mode = lock.get("mode")
    if (
        lock.get("schema_version") != 2 or mode not in ALLOWED_LOSS_MODES
        or lock.get("effective_stage") != LOSS_STAGE_BY_MODE.get(str(mode))
        or lock.get("execution_status") not in {"runnable", "skipped"}
    ):
        raise FastTrackError("Fast-track wrappers permit only schema-v2 primary/overlay/LR-refine locks")
    _validate_short_remote_lock(lock)
    return lock


def run_forwarded_main(
    main: Any, argv: Sequence[str], *, policy_path: Path, receipt_path: Path,
    freeze_manifest_path: Path,
) -> None:
    """Run an unchanged core CLI under the reviewed process-local override."""
    import sys

    previous = list(sys.argv)
    sys.argv = [previous[0], *argv]
    try:
        with patched_loss_predecessor(
            policy_path=policy_path, receipt_path=receipt_path,
            freeze_manifest_path=freeze_manifest_path,
        ):
            main()
    finally:
        sys.argv = previous
