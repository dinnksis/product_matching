#!/usr/bin/env python3
"""Create or version the private prepared mxbai training Dataset on Kaggle."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import create_mxbai_training_notebook as builder
import push_kaggle_training_dataset as shared_push
import run_kaggle_notebook as kaggle


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--dataset-dir", type=Path, default=builder.DEFAULT_DATASET_DIR)
    parser.add_argument("--message", default="mxbai xsmall balanced human plus LLM")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kaggle.load_dotenv(args.env_file)
    owner = os.getenv("KAGGLE_USERNAME", "").strip()
    if not owner:
        kaggle.fail("set KAGGLE_USERNAME in .env")
    owner = kaggle.validate_slug(owner, "KAGGLE_USERNAME")
    if not args.dry_run and not os.getenv("KAGGLE_API_TOKEN", "").strip():
        kaggle.fail("set KAGGLE_API_TOKEN in .env")
    manifest = builder.build_dataset(args.dataset_dir, owner)
    dataset_ref = str(manifest["dataset"])
    print(f"Prepared private dataset payload: {args.dataset_dir}")
    print(f"Dataset reference: {dataset_ref}")
    if args.dry_run:
        print("Dry run complete; Kaggle was not contacted.")
        return
    cli = kaggle.kaggle_command()
    previous_status = shared_push.dataset_status(cli, dataset_ref)
    previous_version = (
        int(previous_status.get("current_version_number", 0))
        if previous_status
        else 0
    )
    if previous_status is not None:
        command = cli + [
            "datasets", "version", "--path", str(args.dataset_dir),
            "--message", args.message, "--keep-tabular",
        ]
    else:
        command = cli + [
            "datasets", "create", "--path", str(args.dataset_dir), "--keep-tabular",
        ]
    kaggle.run_command(command)
    shared_push.wait_until_ready(
        cli,
        dataset_ref,
        minimum_version=previous_version + 1,
    )
    print(f"Private dataset is ready: https://www.kaggle.com/datasets/{dataset_ref}")


if __name__ == "__main__":
    main()
