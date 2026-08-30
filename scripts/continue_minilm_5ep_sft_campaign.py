#!/usr/bin/env python3
"""Safely continue the locked MiniLM-5ep SFT search one stage at a time.

This file is deliberately an *untrusted orchestrator*.  Selection decisions,
recipes, provenance and budget ledgers continue to come from the existing
materializers, stage locks and summarizers.  The controller only determines
which already-declared transition is next and invokes the strict CLIs in the
order documented in ``docs/minilm-5ep-sft-hparam-search.md``.

The default is plan-only and never starts a subprocess.  ``--submit`` is the
only mode which may eventually call the Kaggle launcher.  Before every such
call the controller validates local authority, runs the notebook generator and
then runs the launcher in ``--dry-run`` mode.  Remote variants are always run
with ``--submit --wait``; force/retry/fan-out flags are never generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "configs" / "minilm_5ep_sft_hparam_search_v1.json"
DEFAULT_REPORT_DIR = ROOT / "reports" / "minilm_5ep_sft_hparam_search_v1"
DEFAULT_SUMMARY = DEFAULT_REPORT_DIR / "summary.json"
DEFAULT_LOCKS_DIR = DEFAULT_REPORT_DIR / "stage_locks"
DEFAULT_ARTIFACTS_DIR = ROOT / "artifacts" / "kaggle"
DEFAULT_STATE_PATH = DEFAULT_REPORT_DIR / "controller_state.json"
BASELINE_SUMMARY = DEFAULT_REPORT_DIR / "stages" / "lr_log_line" / "summary.json"

GENERATOR = ROOT / "scripts" / "create_minilm_5ep_sft_hparam_notebooks.py"
LAUNCHER = ROOT / "scripts" / "run_minilm_5ep_sft_hparam_kaggle.py"
SUMMARIZER = ROOT / "scripts" / "summarize_minilm_5ep_sft_hparams.py"
AXIS_MATERIALIZER = ROOT / "scripts" / "materialize_minilm_5ep_sft_hparam_stage.py"
ADAPTIVE_MATERIALIZER = (
    ROOT / "scripts" / "materialize_minilm_5ep_sft_loss_confirmation.py"
)

CAMPAIGN_KIND = "minilm_5ep_sft_campaign_controller_state"
HARD_KERNEL_CAP = 37
MAX_CONTROLLER_ACTIONS = 64
EXPERIMENT_SPREADSHEET_ID = "1CtqT52XOrFyHfFt6rCiOMlnq6snJMlsMOJ0ubH79ikA"
FORBIDDEN_LAUNCHER_FLAGS = {
    "--allow-background-fanout",
    "--force-resubmit",
    "--retry-failed",
    "--no-wait",
}

ADAPTIVE_STAGES = (
    ("loss_primary", "special_loss_screen__primary"),
    ("loss_overlay", "special_loss_screen__overlay"),
    ("loss_lr_refine", "special_loss_screen__lr_refine"),
    ("confirmation", "confirmation__matched_seeds"),
)


class ControllerError(RuntimeError):
    """Local authority is missing, inconsistent or unsafe to act on."""


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
        raise ControllerError(f"Non-canonical controller value: {error}") from error


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
            raise ControllerError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ControllerError(f"{label} is missing: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ControllerError(f"Non-finite JSON constant {token!r} in {label}")
            ),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ControllerError(f"{label} is invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ControllerError(f"{label} must contain a JSON object: {path}")
    canonical_json_dumps(value)
    return value


def _require_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ControllerError(f"{label} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class ControllerPaths:
    root: Path = ROOT
    plan: Path = DEFAULT_PLAN
    report_dir: Path = DEFAULT_REPORT_DIR
    summary: Path = DEFAULT_SUMMARY
    locks_dir: Path = DEFAULT_LOCKS_DIR
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR
    state_path: Path = DEFAULT_STATE_PATH
    baseline_summary: Path = BASELINE_SUMMARY


@dataclass(frozen=True)
class StageSpec:
    effective_stage: str
    schema_version: int
    predecessor: str | None
    mode: str | None = None
    target_stage: str | None = None
    coordinate: str | None = None


@dataclass(frozen=True)
class LockInfo:
    path: Path
    payload: dict[str, Any]
    schema_version: int
    effective_stage: str
    payload_sha256: str
    execution_status: str
    is_boundary: bool
    kernel_slugs: tuple[str, ...]


@dataclass(frozen=True)
class Authority:
    plan: dict[str, Any]
    plan_sha256: str
    stages: tuple[StageSpec, ...]
    summary: dict[str, Any] | None
    summary_sha256: str | None
    current_stage: str | None
    current_decision: dict[str, Any] | None
    locks: tuple[LockInfo, ...]
    unique_kernel_slugs: tuple[str, ...]

    @property
    def hard_cap(self) -> int:
        return int(self.plan["budget"]["maximum_total_kernels"])


@dataclass(frozen=True)
class CommandSpec:
    phase: str
    argv: tuple[str, ...]

    def as_json(self) -> dict[str, Any]:
        return {"phase": self.phase, "argv": list(self.argv)}


@dataclass(frozen=True)
class Action:
    kind: str
    stage: str | None
    reason: str
    commands: tuple[CommandSpec, ...] = ()
    lock_path: Path | None = None
    lock_sha256: str | None = None
    estimated_new_kernels: int = 0

    def as_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "stage": self.stage,
            "reason": self.reason,
            "lock_path": str(self.lock_path) if self.lock_path else None,
            "lock_sha256": self.lock_sha256,
            "estimated_new_kernels": self.estimated_new_kernels,
            "commands": [command.as_json() for command in self.commands],
        }


@dataclass
class ControllerResult:
    status: str
    stop_reason: str
    exit_code: int
    state: dict[str, Any]


def _stage_specs(plan: Mapping[str, Any]) -> tuple[StageSpec, ...]:
    stages = plan.get("stages")
    if not isinstance(stages, list):
        raise ControllerError("Campaign plan has no stages list")
    by_name = {
        str(stage.get("name")): stage
        for stage in stages
        if isinstance(stage, Mapping) and isinstance(stage.get("name"), str)
    }
    required = {
        "lr_log_line",
        "epoch_line",
        "regularization_coordinate_search",
        "special_loss_screen",
        "confirmation",
    }
    if set(by_name) != required:
        raise ControllerError(
            "Campaign stages differ from the frozen LR/epoch/coordinate/loss/confirmation protocol"
        )
    coordinate_stage = by_name["regularization_coordinate_search"]
    order = coordinate_stage.get("execution_order")
    axes = coordinate_stage.get("axes")
    expected_order = [
        "effective_batch",
        "warmup_ratio",
        "weight_decay",
        "label_smoothing",
        "classifier_dropout",
        "max_grad_norm",
    ]
    if order != expected_order or not isinstance(axes, Mapping) or set(order) != set(axes):
        raise ControllerError("Coordinate execution order differs from the frozen plan")

    specs: list[StageSpec] = [
        StageSpec("lr_log_line", 1, None, target_stage="lr_log_line"),
        StageSpec("epoch_line", 1, "lr_log_line", target_stage="epoch_line"),
    ]
    predecessor = "epoch_line"
    for coordinate in order:
        effective = f"regularization_coordinate_search__{coordinate}"
        specs.append(
            StageSpec(
                effective,
                1,
                predecessor,
                target_stage="regularization_coordinate_search",
                coordinate=coordinate,
            )
        )
        predecessor = effective
    for mode, effective in ADAPTIVE_STAGES:
        specs.append(StageSpec(effective, 2, predecessor, mode=mode))
        predecessor = effective
    return tuple(specs)


def validate_plan(plan: Mapping[str, Any]) -> tuple[StageSpec, ...]:
    _require_text(plan.get("campaign"), label="plan campaign")
    if plan.get("schema_version") != 1:
        raise ControllerError("Unsupported campaign-plan schema")
    budget = plan.get("budget")
    if not isinstance(budget, Mapping) or budget.get("maximum_total_kernels") != HARD_KERNEL_CAP:
        raise ControllerError(f"Campaign hard kernel cap must remain {HARD_KERNEL_CAP}")
    protocol = plan.get("selection_protocol")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("search_strategy") != "staged_logarithmic_coordinate_search"
        or protocol.get("maximum_boundary_extensions_per_axis") != 1
        or protocol.get("primary_metric") != "iid_macro_ap"
    ):
        raise ControllerError("Selection protocol differs from the frozen staged search")
    return _stage_specs(plan)


def _verify_payload_hash(payload: Mapping[str, Any], *, label: str) -> str:
    stored = _require_text(payload.get("lock_payload_sha256"), label=f"{label} hash")
    unhashed = dict(payload)
    unhashed.pop("lock_payload_sha256", None)
    if canonical_sha256(unhashed) != stored:
        raise ControllerError(f"{label} payload SHA is invalid")
    return stored


def _load_lock(
    path: Path,
    *,
    campaign: str,
    plan_sha256: str,
    valid_stages: set[str],
) -> LockInfo:
    payload = load_json_object(path, label="stage lock")
    schema = payload.get("schema_version")
    if schema not in {1, 2}:
        raise ControllerError(f"Unsupported stage-lock schema in {path}")
    if payload.get("campaign") != campaign or payload.get("source_plan_sha256") != plan_sha256:
        raise ControllerError(f"Stage lock belongs to another campaign/plan: {path}")
    payload_sha = _verify_payload_hash(payload, label=f"stage lock {path.name}")
    effective = _require_text(payload.get("effective_stage"), label="lock effective_stage")
    if effective not in valid_stages:
        raise ControllerError(f"Unknown effective stage {effective!r} in {path}")
    resolved = payload.get("resolved_stage")
    if not isinstance(resolved, Mapping) or not isinstance(resolved.get("variants"), list):
        raise ControllerError(f"Stage lock has no resolved variants: {path}")
    variants = resolved["variants"]
    slugs: list[str] = []
    for variant in variants:
        if not isinstance(variant, Mapping):
            raise ControllerError(f"Stage lock variant is malformed: {path}")
        slugs.append(_require_text(variant.get("kernel_slug"), label="variant kernel_slug"))
    if len(slugs) != len(set(slugs)):
        raise ControllerError(f"Stage lock repeats a kernel slug: {path}")

    if schema == 1:
        if payload.get("kind") != "minilm_5ep_sft_stage_lock":
            raise ControllerError(f"Unexpected schema-v1 lock kind: {path}")
        execution_status = "runnable"
        is_boundary = payload.get("transition_kind") == "conditional_boundary_extension"
    else:
        execution_status = payload.get("execution_status")
        expected_effective_by_mode = dict(ADAPTIVE_STAGES)
        if expected_effective_by_mode.get(payload.get("mode")) != effective:
            raise ControllerError(f"Adaptive lock mode/effective stage differs: {path}")
        expected_kind = (
            "minilm_5ep_sft_adaptive_branch_receipt"
            if execution_status == "skipped"
            else "minilm_5ep_sft_adaptive_stage_lock"
        )
        if execution_status not in {"runnable", "skipped"} or payload.get("kind") != expected_kind:
            raise ControllerError(f"Unexpected schema-v2 lock status/kind: {path}")
        if execution_status == "skipped" and variants:
            raise ControllerError(f"Skipped receipt unexpectedly contains variants: {path}")
        is_boundary = False
        budget = payload.get("budget")
        if not isinstance(budget, Mapping):
            raise ControllerError(f"Adaptive lock has no frozen budget: {path}")
        union = budget.get("all_unique_kernel_slugs_after")
        if (
            budget.get("hard_limit") != HARD_KERNEL_CAP
            or not isinstance(union, list)
            or len(union) != len(set(union))
            or budget.get("resulting_unique_kernels") != len(union)
            or len(union) > HARD_KERNEL_CAP
        ):
            raise ControllerError(f"Adaptive lock budget is invalid: {path}")
    return LockInfo(
        path=path,
        payload=payload,
        schema_version=int(schema),
        effective_stage=effective,
        payload_sha256=payload_sha,
        execution_status=str(execution_status),
        is_boundary=is_boundary,
        kernel_slugs=tuple(slugs),
    )


def _summary_stage(summary: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    stages = summary.get("stages")
    if not isinstance(stages, Mapping) or len(stages) != 1:
        raise ControllerError("Root summary must describe exactly one terminal stage")
    stage, decision = next(iter(stages.items()))
    if not isinstance(stage, str) or not isinstance(decision, dict):
        raise ControllerError("Root summary stage decision is malformed")
    return stage, decision


def _summary_kernel_slugs(summary: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    runs = summary.get("runs", [])
    if not isinstance(runs, list):
        raise ControllerError("Summary runs must be a list")
    for row in runs:
        if not isinstance(row, Mapping):
            raise ControllerError("Summary run row is malformed")
        slug = row.get("kernel_slug")
        if slug is not None:
            result.add(_require_text(slug, label="summary kernel_slug"))
    budget = summary.get("budget")
    if isinstance(budget, Mapping) and isinstance(budget.get("unique_kernel_slugs"), list):
        budget_slugs = [_require_text(value, label="budget kernel_slug") for value in budget["unique_kernel_slugs"]]
        if len(budget_slugs) != len(set(budget_slugs)):
            raise ControllerError("Summary budget repeats kernel slugs")
        result.update(budget_slugs)
    return result


def inspect_authority(paths: ControllerPaths) -> Authority:
    plan = load_json_object(paths.plan, label="campaign plan")
    stages = validate_plan(plan)
    campaign = str(plan["campaign"])
    plan_sha = canonical_sha256(plan)
    valid_stages = {stage.effective_stage for stage in stages}

    paths.locks_dir.mkdir(parents=True, exist_ok=True)
    locks = tuple(
        _load_lock(
            path,
            campaign=campaign,
            plan_sha256=plan_sha,
            valid_stages=valid_stages,
        )
        for path in sorted(paths.locks_dir.glob("*.lock.json"))
    )
    identities = [(lock.effective_stage, lock.is_boundary, lock.schema_version) for lock in locks]
    if len(identities) != len(set(identities)):
        raise ControllerError("Duplicate lock identity for one stage/transition")

    summary: dict[str, Any] | None = None
    summary_sha: str | None = None
    current_stage: str | None = None
    current_decision: dict[str, Any] | None = None
    if paths.summary.exists():
        summary = load_json_object(paths.summary, label="root campaign summary")
        if summary.get("campaign") != campaign:
            raise ControllerError("Root summary belongs to another campaign")
        current_stage, current_decision = _summary_stage(summary)
        if current_stage not in valid_stages:
            raise ControllerError(f"Root summary has unknown stage {current_stage!r}")
        if summary.get("schema_version") not in {1, 2}:
            raise ControllerError("Root summary has unsupported schema")
        if summary.get("schema_version") == 2:
            if summary.get("effective_stage") != current_stage:
                raise ControllerError("Adaptive summary effective stage differs")
            stored_summary_sha = summary.get("summary_payload_sha256")
            if stored_summary_sha is not None:
                unhashed = dict(summary)
                unhashed.pop("summary_payload_sha256", None)
                if stored_summary_sha != canonical_sha256(unhashed):
                    raise ControllerError("Adaptive summary payload SHA is invalid")
            budget = summary.get("budget")
            union = budget.get("unique_kernel_slugs") if isinstance(budget, Mapping) else None
            if (
                not isinstance(budget, Mapping)
                or budget.get("hard_limit") != HARD_KERNEL_CAP
                or budget.get("history_complete_through") != current_stage
                or not isinstance(union, list)
                or len(union) != len(set(union))
                or budget.get("unique_kernels") != len(union)
                or len(union) > HARD_KERNEL_CAP
            ):
                raise ControllerError("Adaptive summary budget ledger is invalid")
        summary_sha = file_sha256(paths.summary)
    elif locks:
        raise ControllerError("Stage locks exist but root summary is missing")

    kernel_slugs: set[str] = set()
    if summary is not None:
        lr_stage = next(
            stage for stage in plan["stages"] if stage.get("name") == "lr_log_line"
        )
        for variant in lr_stage.get("variants", []):
            if not isinstance(variant, Mapping):
                raise ControllerError("LR plan variant is malformed")
            kernel_slugs.add(
                _require_text(variant.get("kernel_slug"), label="LR kernel_slug")
            )
    summary_paths = []
    if summary is not None:
        summary_paths.append(paths.summary)
    stages_dir = paths.report_dir / "stages"
    if stages_dir.exists():
        summary_paths.extend(sorted(stages_dir.glob("*/summary.json")))
    seen_summary_paths: set[Path] = set()
    for summary_path in summary_paths:
        resolved = summary_path.resolve()
        if resolved in seen_summary_paths:
            continue
        seen_summary_paths.add(resolved)
        document = summary if summary_path == paths.summary and summary is not None else load_json_object(summary_path, label="stage summary")
        if document.get("campaign") != campaign:
            raise ControllerError(f"Stage summary belongs to another campaign: {summary_path}")
        kernel_slugs.update(_summary_kernel_slugs(document))
    for lock in locks:
        kernel_slugs.update(lock.kernel_slugs)
        for key in ("prior_entries", "origins"):
            values = lock.payload.get(key, [])
            if values is None:
                continue
            if not isinstance(values, list):
                raise ControllerError(f"Lock {key} ledger is malformed: {lock.path}")
            for value in values:
                if isinstance(value, Mapping) and value.get("kernel_slug") is not None:
                    kernel_slugs.add(_require_text(value.get("kernel_slug"), label=f"lock {key} kernel_slug"))
        budget = lock.payload.get("budget")
        if isinstance(budget, Mapping) and isinstance(budget.get("all_unique_kernel_slugs_after"), list):
            kernel_slugs.update(str(value) for value in budget["all_unique_kernel_slugs_after"])
    if len(kernel_slugs) > HARD_KERNEL_CAP:
        raise ControllerError(
            f"Trusted lock/summary kernel union is {len(kernel_slugs)}, above cap {HARD_KERNEL_CAP}"
        )
    if summary is not None and summary.get("schema_version") == 1 and current_stage is not None:
        current_index = [stage.effective_stage for stage in stages].index(current_stage)
        last_schema1_index = max(
            index for index, stage in enumerate(stages) if stage.schema_version == 1
        )
        for stage in stages[1 : min(current_index, last_schema1_index) + 1]:
            if not any(
                lock.effective_stage == stage.effective_stage
                and lock.schema_version == 1
                and not lock.is_boundary
                for lock in locks
            ):
                raise ControllerError(
                    f"Completed schema-v1 history is missing {stage.effective_stage!r} lock"
                )
    return Authority(
        plan=plan,
        plan_sha256=plan_sha,
        stages=stages,
        summary=summary,
        summary_sha256=summary_sha,
        current_stage=current_stage,
        current_decision=current_decision,
        locks=locks,
        unique_kernel_slugs=tuple(sorted(kernel_slugs)),
    )


def _lock_filename(stage: StageSpec) -> str:
    if stage.schema_version == 2:
        return f"{stage.effective_stage}.lock.json"
    if stage.effective_stage == "lr_log_line":
        raise ControllerError("Base LR stage has no normal transition lock")
    if stage.coordinate:
        return f"regularization_coordinate_search_{stage.coordinate}.lock.json"
    return f"{stage.effective_stage}.lock.json"


def _lock_for_stage(
    authority: Authority,
    stage: str,
    *,
    boundary: bool | None = None,
) -> LockInfo | None:
    matches = [
        lock
        for lock in authority.locks
        if lock.effective_stage == stage
        and (boundary is None or lock.is_boundary is boundary)
    ]
    if len(matches) > 1:
        raise ControllerError(f"Ambiguous locks for stage {stage!r}")
    return matches[0] if matches else None


def _summary_lock(authority: Authority) -> LockInfo | None:
    summary = authority.summary
    if summary is None or authority.current_stage is None:
        return None
    hashes: set[str] = set()
    stage_lock = summary.get("stage_lock")
    if isinstance(stage_lock, Mapping) and stage_lock.get("lock_payload_sha256"):
        hashes.add(str(stage_lock["lock_payload_sha256"]))
    for key in ("execution_campaign_lock_sha256s", "execution_lock_sha256s", "execution_receipt_sha256s"):
        values = summary.get(key, [])
        if isinstance(values, list):
            hashes.update(str(value) for value in values)
    matches = [
        lock
        for lock in authority.locks
        if lock.effective_stage == authority.current_stage
        and lock.payload_sha256 in hashes
    ]
    if len(matches) > 1:
        # A schema-v2 closure may include prior hashes, but effective_stage keeps
        # the current root unique.  Multiple current hashes are unsafe.
        raise ControllerError("Root summary is bound to multiple current-stage locks")
    if matches:
        return matches[0]
    if authority.current_stage == "lr_log_line" and summary.get("schema_version") == 1:
        return None
    raise ControllerError("Root summary is not bound to its current immutable lock")


def _output_is_fully_sheets_synced(
    directory: Path,
    *,
    experiment: str,
) -> bool:
    completion_path = directory / "notebook_completed.json"
    try:
        completion = load_json_object(completion_path, label="completion marker")
    except ControllerError:
        return False
    run_id = completion.get("run_id")
    if (
        completion.get("status") != "complete"
        or completion.get("experiment") != experiment
        or completion.get("experiment_group") != "sft"
        or not isinstance(run_id, str)
        or not run_id.strip()
        or run_id != run_id.strip()
    ):
        return False
    if (directory / "sheets_sync_pending.json").exists():
        return False
    try:
        sync = load_json_object(
            directory / "google_sheets_sync.json",
            label="Google Sheets sync marker",
        )
    except ControllerError:
        return False
    expected = {
        "status": "synced",
        "run_id": run_id,
        "experiment_group": "sft",
        "comparison_sheet": "sft_exps",
        "spreadsheet_id": EXPERIMENT_SPREADSHEET_ID,
    }
    return all(sync.get(key) == value for key, value in expected.items())


def lock_has_all_local_markers(lock: LockInfo, artifacts_dir: Path) -> bool:
    """Return true only when every local run is complete and exactly Sheets-synced."""
    variants = lock.payload["resolved_stage"]["variants"]
    if lock.execution_status == "skipped":
        return True
    if not variants:
        return False
    return all(
        _output_is_fully_sheets_synced(
            artifacts_dir / str(variant["kernel_slug"]),
            experiment=str(variant["experiment"]),
        )
        for variant in variants
    )


def _command(*values: object, phase: str) -> CommandSpec:
    return CommandSpec(phase=phase, argv=tuple(str(value) for value in values))


class CampaignController:
    def __init__(
        self,
        *,
        paths: ControllerPaths = ControllerPaths(),
        submit: bool = False,
        stop_after: str = "confirmation__matched_seeds",
        runtime_check: Path | None = None,
        python_executable: str = sys.executable,
        runner: Callable[..., subprocess.CompletedProcess[Any]] | None = None,
    ) -> None:
        self.paths = paths
        self.submit = submit
        self.stop_after = stop_after
        self.runtime_check = runtime_check
        self.python_executable = python_executable
        self.runner = runner or subprocess.run
        self.executed_commands: list[dict[str, Any]] = []
        self.attempted_execution_keys: set[str] = set()
        self.attempted_runtime_attestation_keys: set[str] = set()
        self.actions_executed = 0
        self.started_at = datetime.now(timezone.utc).isoformat()

    def _stage_index(self, authority: Authority, stage: str) -> int:
        names = [spec.effective_stage for spec in authority.stages]
        try:
            return names.index(stage)
        except ValueError as error:
            raise ControllerError(f"Unknown controller stage {stage!r}") from error

    def _estimated_new_kernels(self, authority: Authority, stage: StageSpec, *, boundary: bool = False) -> int:
        if boundary:
            return 1
        if stage.effective_stage == "lr_log_line":
            raw = next(item for item in authority.plan["stages"] if item["name"] == "lr_log_line")
            return len(raw["variants"])
        if stage.schema_version == 1:
            return 2
        # Adaptive cardinality is data-dependent and must be frozen by its
        # materializer.  Zero means "unknown until lock", not "free".
        return 0

    def _validate_existing_lock_transition(
        self,
        authority: Authority,
        lock: LockInfo,
        *,
        expected: StageSpec,
        current_stage: str,
        boundary: bool,
    ) -> None:
        if lock.schema_version != expected.schema_version:
            raise ControllerError("Existing next lock has the wrong schema")
        if lock.effective_stage != expected.effective_stage or lock.is_boundary is not boundary:
            raise ControllerError("Existing next lock has the wrong stage/transition kind")
        if lock.schema_version == 1:
            payload = lock.payload
            if boundary:
                if payload.get("source_stage") != current_stage or payload.get("target_stage") != expected.target_stage:
                    raise ControllerError("Boundary lock predecessor/target differs")
            elif payload.get("source_stage") != expected.predecessor or payload.get("target_stage") != expected.target_stage:
                raise ControllerError("Stage lock violates strict coordinate predecessor order")
            if payload.get("coordinate") != expected.coordinate:
                raise ControllerError("Stage lock coordinate differs from the next declared coordinate")
        else:
            if lock.payload.get("mode") != expected.mode:
                raise ControllerError("Adaptive lock mode differs from the next declared mode")

    def _find_existing_next_lock(
        self,
        authority: Authority,
        *,
        expected: StageSpec,
        current_stage: str,
        boundary: bool = False,
    ) -> LockInfo | None:
        lock = _lock_for_stage(authority, expected.effective_stage, boundary=boundary)
        if lock is not None:
            self._validate_existing_lock_transition(
                authority,
                lock,
                expected=expected,
                current_stage=current_stage,
                boundary=boundary,
            )
        current_index = self._stage_index(authority, current_stage)
        expected_index = self._stage_index(authority, expected.effective_stage)
        for candidate in authority.locks:
            candidate_index = self._stage_index(authority, candidate.effective_stage)
            same_stage_normal_before_boundary = (
                boundary
                and candidate_index == expected_index
                and candidate is not lock
                and not candidate.is_boundary
            )
            if candidate_index > expected_index or (
                candidate_index == expected_index
                and candidate is not lock
                and not same_stage_normal_before_boundary
            ):
                raise ControllerError(
                    "A future/out-of-order lock exists beyond the next permitted transition"
                )
            if candidate_index > current_index + 1 and not boundary:
                raise ControllerError("A stage lock skips a declared predecessor")
        return lock

    def _generator_command(self, *, lock: LockInfo | None, stage: str) -> CommandSpec:
        argv: list[object] = [
            self.python_executable,
            GENERATOR,
            "--plan",
            self.paths.plan,
        ]
        if lock is None:
            argv.extend(["--stage", stage])
        else:
            argv.extend(["--stage-lock", lock.path])
        return _command(*argv, phase="generator_precheck")

    def _launcher_command(
        self,
        *,
        lock: LockInfo | None,
        stage: str,
        submit: bool,
    ) -> CommandSpec:
        argv: list[object] = [
            self.python_executable,
            LAUNCHER,
            "--plan",
            self.paths.plan,
        ]
        if lock is None:
            argv.extend(["--stage", stage])
        else:
            argv.extend(["--stage-lock", lock.path])
        argv.extend(["--submit", "--wait"] if submit else ["--dry-run"])
        command = _command(*argv, phase="submit_wait" if submit else "launcher_dry_run")
        forbidden = FORBIDDEN_LAUNCHER_FLAGS & set(command.argv)
        if forbidden:
            raise ControllerError(f"Unsafe launcher flags generated: {sorted(forbidden)}")
        return command

    def _summarizer_command(
        self,
        *,
        lock: LockInfo | None,
        stage: str,
        runtime_check: Path | None = None,
    ) -> CommandSpec:
        argv: list[object] = [
            self.python_executable,
            SUMMARIZER,
            "--plan",
            self.paths.plan,
            "--artifacts-dir",
            self.paths.artifacts_dir,
            "--output-dir",
            self.paths.report_dir,
        ]
        if lock is None:
            argv.extend(["--stage", stage])
        else:
            argv.extend(["--stage-lock", lock.path])
        if runtime_check is not None:
            argv.extend(["--inference-runtime-check", runtime_check])
        return _command(*argv, phase="summarize")

    def _execution_action(self, authority: Authority, lock: LockInfo | None, *, stage: str) -> Action:
        key = lock.payload_sha256 if lock is not None else "base:lr_log_line"
        if key in self.attempted_execution_keys:
            return Action("stop", stage, "pending_stage_after_one_resume_attempt")
        if lock is not None:
            projected = set(authority.unique_kernel_slugs) | set(lock.kernel_slugs)
            if len(projected) > authority.hard_cap:
                return Action("stop", stage, "hard_kernel_cap_reached")
            if lock_has_all_local_markers(lock, self.paths.artifacts_dir):
                return Action(
                    "summarize_existing_outputs",
                    stage,
                    "all local completion markers exist; validate/summarize without Kaggle",
                    commands=(self._summarizer_command(lock=lock, stage=stage),),
                    lock_path=lock.path,
                    lock_sha256=lock.payload_sha256,
                )
        commands = (
            self._generator_command(lock=lock, stage=stage),
            self._launcher_command(lock=lock, stage=stage, submit=False),
            self._launcher_command(lock=lock, stage=stage, submit=True),
            self._summarizer_command(lock=lock, stage=stage),
        )
        return Action(
            "execute_stage",
            stage,
            "runnable stage/receipt requires strict local prechecks then sequential resume",
            commands=commands,
            lock_path=lock.path if lock else None,
            lock_sha256=lock.payload_sha256 if lock else key,
        )

    def _axis_materialize_action(
        self,
        authority: Authority,
        *,
        source_stage: str,
        target: StageSpec,
        boundary: bool,
    ) -> Action:
        estimated = self._estimated_new_kernels(authority, target, boundary=boundary)
        if len(authority.unique_kernel_slugs) + estimated > authority.hard_cap:
            return Action(
                "stop",
                target.effective_stage,
                "hard_kernel_cap_reached",
                estimated_new_kernels=estimated,
            )
        output = (
            self.paths.locks_dir / f"{source_stage}_boundary.lock.json"
            if boundary
            else self.paths.locks_dir / _lock_filename(target)
        )
        argv: list[object] = [
            self.python_executable,
            AXIS_MATERIALIZER,
            "--plan",
            self.paths.plan,
            "--summary",
            self.paths.summary,
            "--artifacts-dir",
            self.paths.artifacts_dir,
            "--from-stage",
            source_stage,
        ]
        if boundary:
            argv.append("--boundary-extension")
        else:
            argv.extend(["--to-stage", target.target_stage])
            if target.coordinate:
                argv.extend(["--coordinate", target.coordinate])
        argv.extend(["--output", output])
        return Action(
            "materialize_boundary" if boundary else "materialize_stage",
            target.effective_stage,
            "freeze the next schema-v1 transition from the terminal summary",
            commands=(_command(*argv, phase="materialize"),),
            lock_path=output,
            estimated_new_kernels=estimated,
        )

    def _adaptive_materialize_action(
        self,
        authority: Authority,
        *,
        target: StageSpec,
    ) -> Action:
        mode = target.mode
        if mode is None:
            raise ControllerError("Adaptive target has no mode")
        output = self.paths.locks_dir / _lock_filename(target)
        argv: list[object] = [
            self.python_executable,
            ADAPTIVE_MATERIALIZER,
            mode,
            "--plan",
            self.paths.plan,
            "--summary",
            self.paths.summary,
            "--artifacts-dir",
            self.paths.artifacts_dir,
        ]
        by_mode = {
            str(lock.payload.get("mode")): lock
            for lock in authority.locks
            if lock.schema_version == 2
        }
        if mode == "loss_primary":
            prerequisite = _summary_lock(authority)
            if prerequisite is None or prerequisite.effective_stage != target.predecessor:
                raise ControllerError("Loss primary requires the terminal max_grad_norm lock")
            argv.extend(["--prerequisite-lock", prerequisite.path])
        elif mode == "loss_overlay":
            prerequisite = by_mode.get("loss_primary")
            if prerequisite is None:
                raise ControllerError("Loss overlay requires the primary lock")
            argv.extend(["--prerequisite-lock", prerequisite.path])
        elif mode == "loss_lr_refine":
            for required in ("loss_primary", "loss_overlay"):
                prerequisite = by_mode.get(required)
                if prerequisite is None:
                    raise ControllerError(f"Loss LR refine requires {required}")
                argv.extend(["--prerequisite-lock", prerequisite.path])
        elif mode == "confirmation":
            if not self.paths.baseline_summary.is_file():
                raise ControllerError("Frozen LR baseline summary is missing for confirmation")
            argv.extend(["--baseline-summary", self.paths.baseline_summary])
            for required in ("loss_primary", "loss_overlay", "loss_lr_refine"):
                prerequisite = by_mode.get(required)
                if prerequisite is None:
                    raise ControllerError(f"Confirmation requires {required}")
                argv.extend(["--prerequisite-lock", prerequisite.path])
        argv.extend(["--output", output])
        return Action(
            "materialize_adaptive",
            target.effective_stage,
            "freeze the next schema-v2 adaptive decision without recomputing it",
            commands=(_command(*argv, phase="materialize"),),
            lock_path=output,
        )

    def decide(self, authority: Authority) -> Action:
        stage_names = [stage.effective_stage for stage in authority.stages]
        if self.stop_after not in stage_names:
            raise ControllerError(f"Unknown --stop-after stage {self.stop_after!r}")
        if len(authority.unique_kernel_slugs) > authority.hard_cap:
            return Action("stop", authority.current_stage, "hard_kernel_cap_exceeded")
        if authority.summary is None:
            first = authority.stages[0]
            estimated = self._estimated_new_kernels(authority, first)
            if len(authority.unique_kernel_slugs) + estimated > authority.hard_cap:
                return Action(
                    "stop",
                    first.effective_stage,
                    "hard_kernel_cap_reached",
                    estimated_new_kernels=estimated,
                )
            return self._execution_action(authority, None, stage=first.effective_stage)

        current = _require_text(authority.current_stage, label="current stage")
        decision = authority.current_decision
        if not isinstance(decision, Mapping):
            raise ControllerError("Current stage has no decision mapping")
        current_index = self._stage_index(authority, current)
        stop_index = self._stage_index(authority, self.stop_after)
        if current_index > stop_index:
            return Action(
                "stop",
                current,
                "requested_stop_stage_already_passed",
            )
        complete = decision.get("complete") is True
        current_lock = _summary_lock(authority)
        if not complete:
            future_locks = [
                lock
                for lock in authority.locks
                if self._stage_index(authority, lock.effective_stage) > current_index
            ]
            if future_locks:
                return Action(
                    "stop",
                    current,
                    "local_authority_future_lock",
                    lock_path=future_locks[0].path,
                    lock_sha256=future_locks[0].payload_sha256,
                )
            premature_boundary = next(
                (
                    lock
                    for lock in authority.locks
                    if lock.effective_stage == current
                    and lock.is_boundary
                    and (current_lock is None or not current_lock.is_boundary)
                ),
                None,
            )
            if premature_boundary is not None:
                return Action(
                    "stop",
                    current,
                    "local_authority_premature_boundary_lock",
                    lock_path=premature_boundary.path,
                    lock_sha256=premature_boundary.payload_sha256,
                )
        decision_status = decision.get("decision_status")
        if not complete:
            if (
                current == "confirmation__matched_seeds"
                and decision.get("runs_complete") is True
                and decision_status == "runtime_gate_pending"
            ):
                if self.runtime_check is None:
                    return Action("stop", current, "runtime_attestation_needed")
                if not self.runtime_check.is_file():
                    raise ControllerError(f"Runtime attestation is missing: {self.runtime_check}")
                runtime_attempt_key = (
                    current_lock.payload_sha256
                    if current_lock is not None
                    else f"base:{current}"
                )
                if runtime_attempt_key in self.attempted_runtime_attestation_keys:
                    return Action(
                        "stop",
                        current,
                        "runtime_gate_pending_after_one_attempt",
                        lock_path=current_lock.path if current_lock else None,
                        lock_sha256=(
                            current_lock.payload_sha256 if current_lock else None
                        ),
                    )
                return Action(
                    "apply_runtime_attestation",
                    current,
                    "re-summarize confirmation with the supplied local runtime attestation",
                    commands=(
                        self._summarizer_command(
                            lock=current_lock,
                            stage=current,
                            runtime_check=self.runtime_check,
                        ),
                    ),
                    lock_path=current_lock.path if current_lock else None,
                    lock_sha256=current_lock.payload_sha256 if current_lock else None,
                )
            if decision_status not in {"pending", "pending_runs", None}:
                return Action("stop", current, "selection_ambiguity")
            return self._execution_action(authority, current_lock, stage=current)

        if decision_status != "ready":
            return Action("stop", current, "selection_ambiguity")
        if authority.summary.get("schema_version") == 1:
            if decision.get("control_gate") != "passed":
                return Action("stop", current, "control_gate_not_passed")
            if not decision.get("recommended_experiment") or not decision.get("recommended_run_id"):
                return Action("stop", current, "selection_ambiguity")
            needs_boundary = decision.get("needs_boundary_extension")
            if not isinstance(needs_boundary, bool):
                raise ControllerError("Schema-v1 decision has no explicit boundary flag")
            if needs_boundary:
                if current_lock is not None and current_lock.is_boundary:
                    return Action("stop", current, "boundary_extension_already_used")
                current_spec = authority.stages[current_index]
                allowed_boundary = current in {
                    "lr_log_line",
                    "epoch_line",
                    "regularization_coordinate_search__weight_decay",
                    "regularization_coordinate_search__label_smoothing",
                }
                if not allowed_boundary:
                    return Action("stop", current, "undeclared_boundary_extension")
                existing = self._find_existing_next_lock(
                    authority,
                    expected=current_spec,
                    current_stage=current,
                    boundary=True,
                )
                if existing is not None:
                    return self._execution_action(authority, existing, stage=current)
                return self._axis_materialize_action(
                    authority,
                    source_stage=current,
                    target=current_spec,
                    boundary=True,
                )

        if current_index >= stop_index:
            return Action("stop", current, "requested_stop_stage_complete")
        if current_index + 1 >= len(authority.stages):
            return Action("stop", current, "campaign_complete")
        target = authority.stages[current_index + 1]
        existing = self._find_existing_next_lock(
            authority,
            expected=target,
            current_stage=current,
            boundary=False,
        )
        if existing is not None:
            return self._execution_action(authority, existing, stage=target.effective_stage)
        if target.schema_version == 1:
            return self._axis_materialize_action(
                authority,
                source_stage=current,
                target=target,
                boundary=False,
            )
        return self._adaptive_materialize_action(authority, target=target)

    def _authority_json(self, authority: Authority) -> dict[str, Any]:
        return {
            "plan_path": str(self.paths.plan),
            "plan_sha256": authority.plan_sha256,
            "summary_path": str(self.paths.summary),
            "summary_file_sha256": authority.summary_sha256,
            "current_stage": authority.current_stage,
            "lock_payload_sha256s": [lock.payload_sha256 for lock in authority.locks],
            "observed_unique_kernel_slugs": list(authority.unique_kernel_slugs),
            "observed_unique_kernels": len(authority.unique_kernel_slugs),
            "hard_kernel_cap": authority.hard_cap,
        }

    def _write_state(
        self,
        *,
        authority: Authority | None,
        planned_action: Action | None,
        status: str,
        stop_reason: str,
        error: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "kind": CAMPAIGN_KIND,
            "authority_role": "diagnostic_only_not_a_selection_authority",
            "campaign": authority.plan["campaign"] if authority else None,
            "invocation_mode": "submit" if self.submit else "plan_only",
            "started_at_utc": self.started_at,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "stop_after": self.stop_after,
            "runtime_check_path": str(self.runtime_check) if self.runtime_check else None,
            "status": status,
            "stop_reason": stop_reason,
            "error": error,
            "authority": self._authority_json(authority) if authority else None,
            "planned_action": planned_action.as_json() if planned_action else None,
            "executed_commands": self.executed_commands,
        }
        payload["controller_state_payload_sha256"] = canonical_sha256(payload)
        self.paths.state_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = canonical_json_dumps(payload) + "\n"
        temporary = self.paths.state_path.with_name(
            f".{self.paths.state_path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, self.paths.state_path)
        return payload

    def _run_command(self, command: CommandSpec) -> tuple[bool, str | None]:
        if FORBIDDEN_LAUNCHER_FLAGS & set(command.argv):
            return False, "unsafe_launcher_flag"
        try:
            completed = self.runner(
                list(command.argv),
                cwd=self.paths.root,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            self.executed_commands.append(
                {**command.as_json(), "returncode": None, "error": str(error)}
            )
            return False, "subprocess_start_failure"
        returncode = int(getattr(completed, "returncode", 1))
        self.executed_commands.append(
            {**command.as_json(), "returncode": returncode}
        )
        if returncode == 0:
            return True, None
        output = " ".join(
            str(value or "")
            for value in (getattr(completed, "stdout", ""), getattr(completed, "stderr", ""))
        ).lower()
        if command.phase == "submit_wait" and any(
            token in output for token in ("quota", "gpu limit", "too many requests", "429")
        ):
            return False, "kaggle_quota_failure"
        if command.phase == "submit_wait":
            return False, "kaggle_or_api_failure"
        if command.phase == "summarize":
            return False, "validator_or_summarizer_failure"
        return False, "local_precheck_failure"

    def _execute(self, action: Action) -> tuple[bool, str | None]:
        self.actions_executed += 1
        for command in action.commands:
            ok, reason = self._run_command(command)
            if not ok:
                return False, reason
        if action.kind.startswith("materialize"):
            if action.lock_path is None or not action.lock_path.is_file():
                return False, "materializer_did_not_create_lock"
        elif action.kind in {"execute_stage", "summarize_existing_outputs"}:
            key = action.lock_sha256 or f"base:{action.stage}"
            self.attempted_execution_keys.add(key)
        elif action.kind == "apply_runtime_attestation":
            key = action.lock_sha256 or f"base:{action.stage}"
            self.attempted_runtime_attestation_keys.add(key)
        return True, None

    def run(self) -> ControllerResult:
        authority: Authority | None = None
        planned_action: Action | None = None
        try:
            while self.actions_executed < MAX_CONTROLLER_ACTIONS:
                authority = inspect_authority(self.paths)
                planned_action = self.decide(authority)
                if not self.submit:
                    state = self._write_state(
                        authority=authority,
                        planned_action=planned_action,
                        status="plan_only",
                        stop_reason="plan_only_no_subprocesses",
                    )
                    return ControllerResult("plan_only", "plan_only_no_subprocesses", 0, state)
                if planned_action.kind == "stop":
                    success_reasons = {
                        "requested_stop_stage_complete",
                        "requested_stop_stage_already_passed",
                        "campaign_complete",
                        "runtime_attestation_needed",
                    }
                    status = "stopped" if planned_action.reason not in success_reasons else "complete_or_paused"
                    state = self._write_state(
                        authority=authority,
                        planned_action=planned_action,
                        status=status,
                        stop_reason=planned_action.reason,
                    )
                    exit_code = 0 if planned_action.reason in success_reasons else 2
                    return ControllerResult(status, planned_action.reason, exit_code, state)
                ok, failure_reason = self._execute(planned_action)
                if not ok:
                    state = self._write_state(
                        authority=authority,
                        planned_action=planned_action,
                        status="failed_closed",
                        stop_reason=str(failure_reason),
                    )
                    return ControllerResult("failed_closed", str(failure_reason), 2, state)
            raise ControllerError("Controller action safety limit reached")
        except ControllerError as error:
            state = self._write_state(
                authority=authority,
                planned_action=planned_action,
                status="failed_closed",
                stop_reason="local_authority_or_validator_failure",
                error=str(error),
            )
            return ControllerResult(
                "failed_closed",
                "local_authority_or_validator_failure",
                2,
                state,
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plan-only",
        action="store_true",
        help="print/write only the next safe action (default)",
    )
    mode.add_argument(
        "--submit",
        action="store_true",
        help="execute strict local prechecks and then sequential --submit --wait",
    )
    parser.add_argument(
        "--stop-after",
        default="confirmation__matched_seeds",
        choices=[
            "lr_log_line",
            "epoch_line",
            "regularization_coordinate_search__effective_batch",
            "regularization_coordinate_search__warmup_ratio",
            "regularization_coordinate_search__weight_decay",
            "regularization_coordinate_search__label_smoothing",
            "regularization_coordinate_search__classifier_dropout",
            "regularization_coordinate_search__max_grad_norm",
            "special_loss_screen__primary",
            "special_loss_screen__overlay",
            "special_loss_screen__lr_refine",
            "confirmation__matched_seeds",
        ],
    )
    parser.add_argument(
        "--runtime-check",
        type=Path,
        help="existing canonical runtime attestation used only to close confirmation",
    )
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    state_path = args.state_path
    if not state_path.is_absolute():
        state_path = ROOT / state_path
    runtime_check = args.runtime_check
    if runtime_check is not None and not runtime_check.is_absolute():
        runtime_check = ROOT / runtime_check
    paths = ControllerPaths(state_path=state_path)
    result = CampaignController(
        paths=paths,
        submit=bool(args.submit),
        stop_after=args.stop_after,
        runtime_check=runtime_check,
    ).run()
    print(canonical_json_dumps(result.state))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
