#!/usr/bin/env python3
"""Build and create/version the private Kaggle training dataset."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import nbformat as nbf

import create_qwen_training_notebook as builder
import run_kaggle_notebook as kaggle


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload raw training data and exact source bundle to a private Kaggle Dataset"
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--dataset-dir", type=Path, default=builder.DEFAULT_DATASET_DIR)
    parser.add_argument("--notebook", type=Path, default=builder.DEFAULT_NOTEBOOK)
    parser.add_argument("--message", help="Kaggle dataset version message")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build local files but do not contact Kaggle",
    )
    return parser.parse_args()


def dataset_status(cli: list[str], dataset_ref: str) -> dict[str, object] | None:
    result = kaggle.run_command(
        cli + ["datasets", "status", dataset_ref, "--format", "json"],
        check=False,
    )
    if result.returncode:
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def wait_until_ready(
    cli: list[str],
    dataset_ref: str,
    timeout: int = 900,
    minimum_version: int | None = None,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        payload = dataset_status(cli, dataset_ref)
        status = str(payload.get("status", "")).lower() if payload else ""
        version = int(payload.get("current_version_number", 0)) if payload else 0
        version_is_current = minimum_version is None or version >= minimum_version
        if version_is_current and any(
            marker in status for marker in ("ready", "complete", "successful")
        ):
            return payload
        if any(marker in status for marker in ("error", "failed", "failure")):
            kaggle.fail(f"Kaggle dataset processing failed with status {status!r}", 1)
        time.sleep(5)
    version_hint = (
        f" at version >= {minimum_version}" if minimum_version is not None else ""
    )
    kaggle.fail(
        f"Kaggle dataset was not ready{version_hint} after {timeout} seconds",
        124,
    )


def main() -> None:
    args = parse_args()
    env_file = args.env_file if args.env_file.is_absolute() else ROOT / args.env_file
    kaggle.load_dotenv(env_file)
    owner = os.getenv("KAGGLE_USERNAME", "").strip()
    if not owner:
        kaggle.fail("set KAGGLE_USERNAME in .env")
    owner = kaggle.validate_slug(owner, "KAGGLE_USERNAME")
    if not args.dry_run and not os.getenv("KAGGLE_API_TOKEN", "").strip():
        kaggle.fail("set KAGGLE_API_TOKEN in .env")

    dataset_dir = (
        args.dataset_dir if args.dataset_dir.is_absolute() else ROOT / args.dataset_dir
    )
    notebook_path = args.notebook if args.notebook.is_absolute() else ROOT / args.notebook
    manifest = builder.build_dataset(dataset_dir, owner)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(builder.build_notebook(manifest), notebook_path)
    dataset_ref = str(manifest["dataset"])
    bundle = manifest["code_bundle"]
    assert isinstance(bundle, dict)
    version_message = args.message or f"Training code {str(bundle['sha256'])[:12]}"

    print(f"Prepared private dataset payload: {dataset_dir}")
    print(f"Dataset reference: {dataset_ref}")
    print(f"Notebook: {notebook_path}")
    if args.dry_run:
        print("Dry run complete; Kaggle was not contacted.")
        return

    cli = kaggle.kaggle_command()
    print("Checking whether the Kaggle dataset already exists...")
    previous_status = dataset_status(cli, dataset_ref)
    previous_version = (
        int(previous_status.get("current_version_number", 0))
        if previous_status
        else 0
    )
    if previous_status is not None:
        command = cli + [
            "datasets",
            "version",
            "--path",
            str(dataset_dir),
            "--message",
            version_message,
            "--keep-tabular",
        ]
    else:
        command = cli + [
            "datasets",
            "create",
            "--path",
            str(dataset_dir),
            "--keep-tabular",
        ]
    kaggle.run_command(command)
    wait_until_ready(
        cli,
        dataset_ref,
        minimum_version=previous_version + 1,
    )
    print(f"Private dataset is ready: https://www.kaggle.com/datasets/{dataset_ref}")


if __name__ == "__main__":
    main()
