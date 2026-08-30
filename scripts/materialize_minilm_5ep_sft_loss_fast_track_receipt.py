#!/usr/bin/env python3
"""Create/replay the immutable receipt that skips max_grad_norm after dropout."""

from __future__ import annotations

import argparse
from pathlib import Path

import minilm_5ep_sft_loss_fast_track_support as support


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--policy", type=Path, default=support.DEFAULT_POLICY)
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=support.DEFAULT_FREEZE_MANIFEST,
    )
    parser.add_argument("--plan", type=Path, default=support.DEFAULT_PLAN)
    parser.add_argument("--summary", type=Path, default=support.DEFAULT_SUMMARY)
    parser.add_argument(
        "--artifacts-dir", type=Path, default=support.DEFAULT_ARTIFACTS_DIR
    )
    parser.add_argument("--output", type=Path, default=support.DEFAULT_RECEIPT)
    args = parser.parse_args()
    existed = args.output.exists()
    if existed:
        payload = support.validate_receipt(
            args.output,
            policy_path=args.policy,
            freeze_manifest_path=args.freeze_manifest,
        )
    else:
        payload = support.build_receipt(
            policy_path=args.policy,
            freeze_manifest_path=args.freeze_manifest,
            plan_path=args.plan,
            summary_path=args.summary,
            artifacts_dir=args.artifacts_dir,
            receipt_path=args.output,
        )
        payload = support.write_receipt_once(
            args.output,
            payload,
            policy_path=args.policy,
            freeze_manifest_path=args.freeze_manifest,
        )
        payload = support.validate_receipt(
            args.output,
            policy_path=args.policy,
            freeze_manifest_path=args.freeze_manifest,
        )
    print(
        support.canonical_json_dumps(
            {
                "status": "reused" if existed else "created",
                "output": str(args.output),
                "summary_payload_sha256": payload["summary_payload_sha256"],
                "source_stage": payload["source_stage"],
                "skipped_coordinate": payload["skipped_coordinate"]["name"],
                "prior_unique_kernels": payload["budget"]["unique_kernels"],
                "new_kernels_for_skip": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
