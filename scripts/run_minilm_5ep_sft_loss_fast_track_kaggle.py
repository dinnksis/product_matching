#!/usr/bin/env python3
"""Launch/monitor SFT loss runs under the validated fast-track authority."""

from __future__ import annotations

import argparse
from pathlib import Path

import minilm_5ep_sft_loss_fast_track_support as support
import run_minilm_5ep_sft_hparam_kaggle as core


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--fast-track-policy", type=Path, default=support.DEFAULT_POLICY)
    parser.add_argument("--fast-track-receipt", type=Path, default=support.DEFAULT_RECEIPT)
    parser.add_argument(
        "--fast-track-freeze-manifest",
        type=Path,
        default=support.DEFAULT_FREEZE_MANIFEST,
    )
    parser.add_argument("--plan", type=Path, default=support.DEFAULT_PLAN)
    parser.add_argument("--stage-lock", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--submit", action="store_true")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    if args.submit is not args.wait:
        parser.error("launcher requires exactly --dry-run OR --submit --wait")
    support.load_scoped_loss_lock(
        plan_path=args.plan,
        stage_lock_path=args.stage_lock,
        policy_path=args.fast_track_policy,
        receipt_path=args.fast_track_receipt,
        freeze_manifest_path=args.fast_track_freeze_manifest,
    )
    remaining = ["--plan", str(args.plan), "--stage-lock", str(args.stage_lock)]
    remaining.extend(["--submit", "--wait"] if args.submit else ["--dry-run"])
    support.run_forwarded_main(
        core.main,
        remaining,
        policy_path=args.fast_track_policy,
        receipt_path=args.fast_track_receipt,
        freeze_manifest_path=args.fast_track_freeze_manifest,
    )


if __name__ == "__main__":
    main()
