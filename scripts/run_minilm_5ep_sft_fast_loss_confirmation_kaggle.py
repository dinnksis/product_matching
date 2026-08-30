#!/usr/bin/env python3
"""Run only the two manifest-bound seed-17 fast-loss confirmation kernels.

This entrypoint owns the shortened confirmation boundary.  It never accepts an
experiment selector from the caller: the exact tuned-BCE/loss pair is derived
from the canonical confirmation manifest and source lock under the detached
fast-track policy/receipt/freeze authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import minilm_5ep_sft_loss_fast_track_support as fast_track
import run_minilm_5ep_sft_fast_loss_confirmation as confirmation
import run_minilm_5ep_sft_hparam_kaggle as core_launcher


ROOT = Path(__file__).resolve().parents[1]


def _resolved(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _require_canonical_manifest(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise confirmation.FastConfirmationError(
            "Fast-confirmation manifest must not be a symlink"
        )
    resolved = path.resolve(strict=True)
    manifest = confirmation.load_json_object(
        resolved, label="fast-confirmation execution manifest"
    )
    if resolved.read_text(encoding="utf-8") != confirmation.canonical_json_dumps(manifest) + "\n":
        raise confirmation.FastConfirmationError(
            "Fast-confirmation execution manifest is not canonical JSON"
        )
    return manifest


def load_bound_execution(
    *,
    manifest_path: Path,
    plan_path: Path,
    lock_path: Path,
    policy_path: Path,
    receipt_path: Path,
    freeze_manifest_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate and derive the exact two runnable entries without side effects."""
    if (
        plan_path.is_symlink()
        or plan_path.resolve(strict=True) != confirmation.DEFAULT_PLAN.resolve(strict=True)
    ):
        raise confirmation.FastConfirmationError(
            "Fast confirmation requires the default frozen plan"
        )
    if lock_path.is_symlink() or not lock_path.resolve(strict=True).is_file():
        raise confirmation.FastConfirmationError(
            "Fast confirmation requires a regular non-symlink source lock"
        )
    resolved_lock = lock_path.resolve(strict=True)
    try:
        resolved_manifest = manifest_path.resolve(strict=True)
    except OSError as error:
        raise confirmation.FastConfirmationError(
            "Fast confirmation requires its exact colocated execution manifest"
        ) from error
    if (
        resolved_lock.name != confirmation.DEFAULT_SOURCE_LOCK_NAME
        or resolved_manifest
        != resolved_lock.parent / confirmation.DEFAULT_MANIFEST_NAME
    ):
        raise confirmation.FastConfirmationError(
            "Fast confirmation requires its exact source lock and colocated manifest"
        )
    manifest = _require_canonical_manifest(manifest_path)
    plan = confirmation.builder.load_plan(plan_path)
    protocol = confirmation.load_protocol(confirmation.DEFAULT_PROTOCOL, plan=plan)
    confirmation._validate_execution_projection(
        manifest,
        plan_path=plan_path,
        lock_path=lock_path,
    )
    with fast_track.patched_loss_predecessor(
        policy_path=policy_path,
        receipt_path=receipt_path,
        freeze_manifest_path=freeze_manifest_path,
    ) as (policy, receipt, freeze):
        confirmation._validate_fast_track_binding(
            protocol=protocol,
            policy_path=policy_path,
            receipt_path=receipt_path,
            freeze_manifest_path=freeze_manifest_path,
            policy=policy,
            receipt=receipt,
            freeze=freeze,
        )
        lock = confirmation.builder.load_campaign_lock(lock_path, plan=plan)
        selected = confirmation.select_fast_pair(
            plan=plan,
            lock=lock,
            protocol=protocol,
        )
        expected_manifest = confirmation.build_manifest(
            protocol_path=confirmation.DEFAULT_PROTOCOL,
            protocol=protocol,
            plan_path=plan_path,
            lock_path=lock_path,
            lock=lock,
            selected=selected,
            policy_path=policy_path,
            policy=policy,
            receipt_path=receipt_path,
            receipt=receipt,
            freeze_manifest_path=freeze_manifest_path,
            freeze=freeze,
        )
    if manifest != expected_manifest:
        raise confirmation.FastConfirmationError(
            "Fast-confirmation manifest differs from the exact reconstructed authority"
        )
    entries = [dict(selected[side]["entry"]) for side in ("comparator", "candidate")]
    if (
        [entry["experiment"] for entry in entries] != manifest["launch_order"]
        or [entry["expected_config"].get("seed") for entry in entries] != [17, 17]
        or [entry["loss_variant"] == "bce" for entry in entries] != [True, False]
    ):
        raise confirmation.FastConfirmationError(
            "Reconstructed execution is not the exact matched seed-17 BCE/loss pair"
        )
    return manifest, entries


def _core_argv(
    *,
    plan_path: Path,
    lock_path: Path,
    env_file: Path,
    experiment: str,
    submit: bool,
) -> list[str]:
    argv = [
        "--plan",
        str(plan_path),
        "--stage-lock",
        str(lock_path),
        "--only",
        experiment,
        "--env-file",
        str(env_file),
    ]
    return [*argv, "--submit", "--wait"] if submit else [*argv, "--dry-run"]


def execute(
    *,
    manifest_path: Path,
    plan_path: Path,
    lock_path: Path,
    env_file: Path,
    policy_path: Path,
    receipt_path: Path,
    freeze_manifest_path: Path,
    action: str,
) -> list[Mapping[str, Any]]:
    if action not in {"dry-run", "submit"}:
        raise confirmation.FastConfirmationError("Unknown confirmation launcher action")
    default_env = (ROOT / ".env").resolve(strict=False)
    if env_file.is_symlink() or env_file.resolve(strict=False) != default_env:
        raise confirmation.FastConfirmationError(
            "Fast confirmation requires the canonical repository .env path"
        )
    _, entries = load_bound_execution(
        manifest_path=manifest_path,
        plan_path=plan_path,
        lock_path=lock_path,
        policy_path=policy_path,
        receipt_path=receipt_path,
        freeze_manifest_path=freeze_manifest_path,
    )
    validated: list[Mapping[str, Any]] = []
    for entry in entries:
        if action == "submit":
            fast_track.run_forwarded_main(
                core_launcher.main,
                _core_argv(
                    plan_path=plan_path,
                    lock_path=lock_path,
                    env_file=env_file,
                    experiment=str(entry["experiment"]),
                    submit=False,
                ),
                policy_path=policy_path,
                receipt_path=receipt_path,
                freeze_manifest_path=freeze_manifest_path,
            )
        fast_track.run_forwarded_main(
            core_launcher.main,
            _core_argv(
                plan_path=plan_path,
                lock_path=lock_path,
                env_file=env_file,
                experiment=str(entry["experiment"]),
                submit=action == "submit",
            ),
            policy_path=policy_path,
            receipt_path=receipt_path,
            freeze_manifest_path=freeze_manifest_path,
        )
        if action == "submit":
            validated.append(
                core_launcher.validate_run_output(
                    core_launcher.output_root() / str(entry["kernel_slug"]),
                    entry=entry,
                )
            )
    return validated


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--fast-confirmation-manifest", type=Path, required=True)
    parser.add_argument("--fast-track-policy", type=Path, default=fast_track.DEFAULT_POLICY)
    parser.add_argument("--fast-track-receipt", type=Path, default=fast_track.DEFAULT_RECEIPT)
    parser.add_argument(
        "--fast-track-freeze-manifest",
        type=Path,
        default=fast_track.DEFAULT_FREEZE_MANIFEST,
    )
    parser.add_argument("--plan", type=Path, default=confirmation.DEFAULT_PLAN)
    parser.add_argument("--stage-lock", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--submit", action="store_true")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args(argv)
    if args.submit is not args.wait:
        parser.error("confirmation launcher requires exactly --dry-run OR --submit --wait")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validated = execute(
        manifest_path=_resolved(args.fast_confirmation_manifest),
        plan_path=_resolved(args.plan),
        lock_path=_resolved(args.stage_lock),
        env_file=_resolved(args.env_file),
        policy_path=_resolved(args.fast_track_policy),
        receipt_path=_resolved(args.fast_track_receipt),
        freeze_manifest_path=_resolved(args.fast_track_freeze_manifest),
        action="submit" if args.submit else "dry-run",
    )
    if validated:
        print(json.dumps(validated, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
