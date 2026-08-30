#!/usr/bin/env python3
"""Continue the MiniLM SFT campaign from dropout directly through loss search.

This controller is an isolated policy overlay on the frozen v1 controller.  It
keeps every completed stage and schema-v2 loss rule, but removes the unrun
``max_grad_norm`` coordinate and the original three-seed/runtime confirmation.
The default terminal stage is loss-LR refinement; a separate one-additional-seed
check can consume its winner later.

The default is plan-only.  Only ``--submit`` may call Kaggle, always through the
existing sequential ``--submit --wait`` launcher.  This script contains no ODS
operation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import continue_minilm_5ep_sft_campaign as core
import minilm_5ep_sft_loss_fast_track_support as support


ROOT = support.ROOT
GENERATOR = ROOT / "scripts" / "create_minilm_5ep_sft_loss_fast_track_notebooks.py"
LAUNCHER = ROOT / "scripts" / "run_minilm_5ep_sft_loss_fast_track_kaggle.py"
SUMMARIZER = ROOT / "scripts" / "summarize_minilm_5ep_sft_loss_fast_track.py"
MATERIALIZER = ROOT / "scripts" / "materialize_minilm_5ep_sft_loss_fast_track.py"
RECEIPT_MATERIALIZER = (
    ROOT / "scripts" / "materialize_minilm_5ep_sft_loss_fast_track_receipt.py"
)
DEFAULT_STATE = (
    support.DEFAULT_REPORT_DIR / "fast_track" / "loss_controller_state.json"
)
STOP_CHOICES = (
    support.SOURCE_STAGE,
    *support.LOSS_STAGES,
)


def fast_stage_specs(plan: Mapping[str, Any]) -> tuple[core.StageSpec, ...]:
    """Project the reviewed stages onto the explicitly authorized fast track."""
    original = core._ORIGINAL_FAST_TRACK_STAGE_SPECS(plan)  # type: ignore[attr-defined]
    result: list[core.StageSpec] = []
    for stage in original:
        if stage.effective_stage in {
            support.SKIPPED_STAGE,
            "confirmation__matched_seeds",
        }:
            continue
        if stage.effective_stage == support.LOSS_PRIMARY_STAGE:
            stage = replace(stage, predecessor=support.SOURCE_STAGE)
        result.append(stage)
    names = [stage.effective_stage for stage in result]
    if support.SKIPPED_STAGE in names or "confirmation__matched_seeds" in names:
        raise core.ControllerError("Fast-track stage projection retained a forbidden stage")
    if names[-3:] != list(support.LOSS_STAGES):
        raise core.ControllerError("Fast-track loss stage suffix differs")
    return tuple(result)


def install_stage_projection() -> None:
    if not hasattr(core, "_ORIGINAL_FAST_TRACK_STAGE_SPECS"):
        core._ORIGINAL_FAST_TRACK_STAGE_SPECS = core._stage_specs  # type: ignore[attr-defined]
    elif core._stage_specs is not fast_stage_specs:
        raise core.ControllerError("Core stage resolver was already replaced")
    core._stage_specs = fast_stage_specs


class LossFastTrackController(core.CampaignController):
    def __init__(
        self,
        *,
        policy_path: Path = support.DEFAULT_POLICY,
        receipt_path: Path = support.DEFAULT_RECEIPT,
        freeze_manifest_path: Path = support.DEFAULT_FREEZE_MANIFEST,
        **kwargs: Any,
    ) -> None:
        self.policy_path = policy_path.resolve(strict=True)
        self.policy = support.load_policy(self.policy_path)
        self.freeze_manifest_path = freeze_manifest_path.resolve(strict=True)
        self.freeze_manifest = support.load_freeze_manifest(
            self.freeze_manifest_path,
            policy=self.policy,
        )
        self.receipt_path = receipt_path.resolve(strict=False)
        kwargs.setdefault("stop_after", "special_loss_screen__lr_refine")
        super().__init__(**kwargs)

    def _fast_prefix(self, executable: Path) -> list[object]:
        return [
            self.python_executable,
            executable,
            "--fast-track-policy",
            self.policy_path,
            "--fast-track-receipt",
            self.receipt_path,
            "--fast-track-freeze-manifest",
            self.freeze_manifest_path,
        ]

    def _generator_command(
        self, *, lock: core.LockInfo | None, stage: str
    ) -> core.CommandSpec:
        argv: list[object]
        if lock is not None and lock.schema_version == 2:
            argv = [*self._fast_prefix(GENERATOR), "--plan", self.paths.plan]
        else:
            argv = [self.python_executable, core.GENERATOR, "--plan", self.paths.plan]
        if lock is None:
            argv.extend(["--stage", stage])
        else:
            argv.extend(["--stage-lock", lock.path])
        return core._command(*argv, phase="generator_precheck")

    def _launcher_command(
        self,
        *,
        lock: core.LockInfo | None,
        stage: str,
        submit: bool,
    ) -> core.CommandSpec:
        argv: list[object]
        if lock is not None and lock.schema_version == 2:
            argv = [*self._fast_prefix(LAUNCHER), "--plan", self.paths.plan]
        else:
            argv = [self.python_executable, core.LAUNCHER, "--plan", self.paths.plan]
        if lock is None:
            argv.extend(["--stage", stage])
        else:
            argv.extend(["--stage-lock", lock.path])
        argv.extend(["--submit", "--wait"] if submit else ["--dry-run"])
        command = core._command(
            *argv, phase="submit_wait" if submit else "launcher_dry_run"
        )
        forbidden = core.FORBIDDEN_LAUNCHER_FLAGS & set(command.argv)
        if forbidden:
            raise core.ControllerError(
                f"Unsafe fast-track launcher flags generated: {sorted(forbidden)}"
            )
        return command

    def _summarizer_command(
        self,
        *,
        lock: core.LockInfo | None,
        stage: str,
        runtime_check: Path | None = None,
    ) -> core.CommandSpec:
        if runtime_check is not None:
            raise core.ControllerError("Fast-track loss controller has no runtime/ODS gate")
        if lock is not None and lock.schema_version == 2:
            argv: list[object] = [
                *self._fast_prefix(SUMMARIZER),
                "--plan",
                self.paths.plan,
            ]
        else:
            argv = [self.python_executable, core.SUMMARIZER, "--plan", self.paths.plan]
        argv.extend(
            [
                "--artifacts-dir",
                self.paths.artifacts_dir,
                "--output-dir",
                self.paths.report_dir,
            ]
        )
        if lock is None:
            argv.extend(["--stage", stage])
        else:
            argv.extend(["--stage-lock", lock.path])
        return core._command(*argv, phase="summarize")

    def _receipt_action(self) -> core.Action:
        argv: list[object] = [
            self.python_executable,
            RECEIPT_MATERIALIZER,
            "--policy",
            self.policy_path,
            "--freeze-manifest",
            self.freeze_manifest_path,
            "--plan",
            self.paths.plan,
            "--summary",
            self.paths.summary,
            "--artifacts-dir",
            self.paths.artifacts_dir,
            "--output",
            self.receipt_path,
        ]
        return core.Action(
            "materialize_fast_track_receipt",
            support.LOSS_PRIMARY_STAGE,
            "freeze the explicit zero-kernel max_grad_norm skip and complete budget ledger",
            commands=(core._command(*argv, phase="materialize"),),
            lock_path=self.receipt_path,
            estimated_new_kernels=0,
        )

    def _adaptive_materialize_action(
        self,
        authority: core.Authority,
        *,
        target: core.StageSpec,
    ) -> core.Action:
        mode = target.mode
        if mode not in {"loss_primary", "loss_overlay", "loss_lr_refine"}:
            raise core.ControllerError(f"Unsupported fast-track adaptive mode {mode!r}")
        support.validate_receipt(
            self.receipt_path,
            policy=self.policy,
            policy_path=self.policy_path,
            freeze_manifest_path=self.freeze_manifest_path,
        )
        output = self.paths.locks_dir / core._lock_filename(target)
        argv: list[object] = [
            *self._fast_prefix(MATERIALIZER),
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
            prerequisite = core._summary_lock(authority)
            if (
                prerequisite is None
                or prerequisite.schema_version != 1
                or prerequisite.effective_stage != support.SOURCE_STAGE
            ):
                raise core.ControllerError(
                    "Fast-track loss primary requires the terminal dropout lock"
                )
            argv.extend(
                [
                    "--source-stage",
                    support.SOURCE_STAGE,
                    "--history-summary",
                    self.receipt_path,
                    "--prerequisite-lock",
                    prerequisite.path,
                ]
            )
        elif mode == "loss_overlay":
            prerequisite = by_mode.get("loss_primary")
            if prerequisite is None:
                raise core.ControllerError("Loss overlay requires the primary lock")
            argv.extend(["--prerequisite-lock", prerequisite.path])
        else:
            for required in ("loss_primary", "loss_overlay"):
                prerequisite = by_mode.get(required)
                if prerequisite is None:
                    raise core.ControllerError(f"Loss LR refine requires {required}")
                argv.extend(["--prerequisite-lock", prerequisite.path])
        argv.extend(["--output", output])
        return core.Action(
            "materialize_adaptive",
            target.effective_stage,
            "freeze the next declared loss decision under the immutable skip receipt",
            commands=(core._command(*argv, phase="materialize"),),
            lock_path=output,
        )

    def decide(self, authority: core.Authority) -> core.Action:
        if not self.receipt_path.exists():
            support.validate_pre_receipt_authority(
                authority,
                report_dir=self.paths.report_dir,
                artifacts_dir=self.paths.artifacts_dir,
                require_completed_dropout=(
                    authority.current_stage == support.SOURCE_STAGE
                    and isinstance(authority.current_decision, Mapping)
                    and authority.current_decision.get("complete") is True
                ),
            )
        if (
            authority.current_stage == support.SOURCE_STAGE
            and isinstance(authority.current_decision, Mapping)
            and authority.current_decision.get("complete") is True
            and self.stop_after != support.SOURCE_STAGE
        ):
            if not self.receipt_path.exists():
                return self._receipt_action()
            support.validate_receipt(
                self.receipt_path,
                policy=self.policy,
                policy_path=self.policy_path,
                freeze_manifest_path=self.freeze_manifest_path,
            )
        return super().decide(authority)

    def _authority_json(self, authority: core.Authority) -> dict[str, Any]:
        payload = super()._authority_json(authority)
        payload.update(
            {
                "fast_track_policy_path": str(self.policy_path),
                "fast_track_policy_sha256": support.policy_sha256(self.policy),
                "fast_track_freeze_manifest_path": str(self.freeze_manifest_path),
                "fast_track_freeze_manifest_sha256": self.freeze_manifest[
                    "manifest_payload_sha256"
                ],
                "max_grad_norm_execution": "skipped_without_metric_claim",
                "ods_submission_allowed": False,
            }
        )
        if self.receipt_path.exists():
            receipt = support.validate_receipt(
                self.receipt_path,
                policy=self.policy,
                policy_path=self.policy_path,
                freeze_manifest_path=self.freeze_manifest_path,
            )
            payload.update(
                {
                    "fast_track_receipt_path": str(self.receipt_path),
                    "fast_track_receipt_sha256": receipt["summary_payload_sha256"],
                }
            )
        return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--submit", action="store_true")
    parser.add_argument(
        "--stop-after",
        choices=STOP_CHOICES,
        default="special_loss_screen__lr_refine",
    )
    parser.add_argument("--policy", type=Path, default=support.DEFAULT_POLICY)
    parser.add_argument("--receipt", type=Path, default=support.DEFAULT_RECEIPT)
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=support.DEFAULT_FREEZE_MANIFEST,
    )
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    install_stage_projection()
    state_path = args.state_path
    if not state_path.is_absolute():
        state_path = ROOT / state_path
    policy_path = args.policy if args.policy.is_absolute() else ROOT / args.policy
    receipt_path = args.receipt if args.receipt.is_absolute() else ROOT / args.receipt
    freeze_manifest_path = (
        args.freeze_manifest
        if args.freeze_manifest.is_absolute()
        else ROOT / args.freeze_manifest
    )
    paths = core.ControllerPaths(state_path=state_path)
    result = LossFastTrackController(
        paths=paths,
        submit=bool(args.submit),
        stop_after=args.stop_after,
        runtime_check=None,
        policy_path=policy_path,
        receipt_path=receipt_path,
        freeze_manifest_path=freeze_manifest_path,
    ).run()
    print(core.canonical_json_dumps(result.state))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
