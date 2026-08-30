#!/usr/bin/env python3
"""Run the frozen adaptive materializer under the validated loss fast-track."""

from __future__ import annotations

import argparse
from pathlib import Path

import materialize_minilm_5ep_sft_loss_confirmation as core
import minilm_5ep_sft_loss_fast_track_support as support


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("mode", choices=support.ALLOWED_LOSS_MODES)
    parser.add_argument("--fast-track-policy", type=Path, default=support.DEFAULT_POLICY)
    parser.add_argument("--fast-track-receipt", type=Path, default=support.DEFAULT_RECEIPT)
    parser.add_argument(
        "--fast-track-freeze-manifest",
        type=Path,
        default=support.DEFAULT_FREEZE_MANIFEST,
    )
    parser.add_argument("--plan", type=Path, default=support.DEFAULT_PLAN)
    parser.add_argument("--summary", type=Path, default=support.DEFAULT_SUMMARY)
    parser.add_argument("--artifacts-dir", type=Path, default=support.DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--prerequisite-lock", type=Path, action="append", default=[])
    parser.add_argument("--history-summary", type=Path, action="append", default=[])
    parser.add_argument("--source-stage")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "loss_primary":
        if (
            len(args.prerequisite_lock) != 1
            or args.source_stage != support.SOURCE_STAGE
            or args.history_summary != [args.fast_track_receipt]
        ):
            parser.error(
                "loss_primary requires exactly the dropout prerequisite, source-stage "
                "classifier_dropout, and the fast-track receipt as its sole history summary"
            )
    elif args.mode == "loss_overlay":
        if len(args.prerequisite_lock) != 1 or args.source_stage is not None or args.history_summary:
            parser.error("loss_overlay requires exactly one prerequisite and no source/history override")
    elif len(args.prerequisite_lock) != 2 or args.source_stage is not None or args.history_summary:
        parser.error("loss_lr_refine requires exactly primary+overlay prerequisites")
    remaining = [
        args.mode,
        "--plan", str(args.plan),
        "--summary", str(args.summary),
        "--artifacts-dir", str(args.artifacts_dir),
    ]
    for prerequisite in args.prerequisite_lock:
        remaining.extend(["--prerequisite-lock", str(prerequisite)])
    for history in args.history_summary:
        remaining.extend(["--history-summary", str(history)])
    if args.source_stage is not None:
        remaining.extend(["--source-stage", args.source_stage])
    remaining.extend(["--output", str(args.output)])
    support.run_forwarded_main(
        core.main,
        remaining,
        policy_path=args.fast_track_policy,
        receipt_path=args.fast_track_receipt,
        freeze_manifest_path=args.fast_track_freeze_manifest,
    )
    support.load_scoped_loss_lock(
        plan_path=args.plan,
        stage_lock_path=args.output,
        policy_path=args.fast_track_policy,
        receipt_path=args.fast_track_receipt,
        freeze_manifest_path=args.fast_track_freeze_manifest,
    )


if __name__ == "__main__":
    main()
