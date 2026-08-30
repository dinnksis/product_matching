#!/usr/bin/env python3
"""Build frozen SFT notebooks under the validated loss fast-track authority."""

from __future__ import annotations

import argparse
from pathlib import Path

import create_minilm_5ep_sft_hparam_notebooks as core
import minilm_5ep_sft_loss_fast_track_support as support


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
    args = parser.parse_args()
    support.load_scoped_loss_lock(
        plan_path=args.plan,
        stage_lock_path=args.stage_lock,
        policy_path=args.fast_track_policy,
        receipt_path=args.fast_track_receipt,
        freeze_manifest_path=args.fast_track_freeze_manifest,
    )
    remaining = ["--plan", str(args.plan), "--stage-lock", str(args.stage_lock)]
    support.run_forwarded_main(
        core.main,
        remaining,
        policy_path=args.fast_track_policy,
        receipt_path=args.fast_track_receipt,
        freeze_manifest_path=args.fast_track_freeze_manifest,
    )


if __name__ == "__main__":
    main()
