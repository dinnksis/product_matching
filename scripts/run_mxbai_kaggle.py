#!/usr/bin/env python3
"""Generate and submit the configured mxbai balanced-training notebook."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat as nbf

import create_cross_encoder_training_notebook as cross_builder
import create_mxbai_training_notebook as builder
import run_kaggle_notebook as kaggle


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DATASET_FILES = {
    "mxbai_balanced_items.parquet",
    "mxbai_balanced_train_pairs.parquet",
    "mxbai_balanced_val_pairs.parquet",
    "mxbai_balanced_report.json",
    builder.shared.MANIFEST_NAME,
}


def verify_dataset_is_mountable(dataset_ref: str, expected_bundle_hash: str) -> None:
    """Fail before GPU submission if the private prepared Dataset is unavailable."""
    cli = kaggle.kaggle_command()
    status = kaggle.run_command(
        cli + ["datasets", "status", dataset_ref], check=False
    )
    if status.returncode or "ready" not in status.stdout.lower():
        raise SystemExit(
            f"Kaggle Dataset {dataset_ref!r} is not ready; publish it with "
            "scripts/push_mxbai_training_dataset.py before launching training."
        )
    files = kaggle.run_command(
        cli + ["datasets", "files", dataset_ref], check=False
    )
    missing = sorted(name for name in REQUIRED_DATASET_FILES if name not in files.stdout)
    if files.returncode or missing:
        raise SystemExit(
            f"Kaggle Dataset {dataset_ref!r} is missing required files: {missing}"
        )
    with tempfile.TemporaryDirectory(prefix="mxbai-dataset-manifest-") as temp_dir:
        manifest_download = kaggle.run_command(
            cli
            + [
                "datasets",
                "download",
                dataset_ref,
                "-f",
                builder.shared.MANIFEST_NAME,
                "-p",
                temp_dir,
                "-o",
                "-q",
            ],
            check=False,
        )
        manifest_path = Path(temp_dir) / builder.shared.MANIFEST_NAME
        if manifest_download.returncode or not manifest_path.is_file():
            raise SystemExit(f"Could not verify manifest of Kaggle Dataset {dataset_ref!r}")
        remote_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    remote_hash = str(remote_manifest["code_bundle"]["sha256"])
    if remote_hash != expected_bundle_hash:
        raise SystemExit(
            f"Kaggle Dataset {dataset_ref!r} is still serving bundle {remote_hash}; "
            f"the notebook requires {expected_bundle_hash}. Wait for the new Dataset "
            "version or publish it before launching training."
        )
    print(f"Verified remote prepared Dataset: {dataset_ref}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--config", type=Path, default=builder.DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    owner = builder.shared.dotenv_username(args.env_file)
    if not owner:
        raise SystemExit("Set KAGGLE_USERNAME in .env")
    config = cross_builder.load_training_config(args.config)
    manifest = builder.build_dataset(builder.DEFAULT_DATASET_DIR, owner)
    notebook = builder.build_notebook(manifest, config)
    builder.DEFAULT_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, builder.DEFAULT_NOTEBOOK)
    if not args.dry_run:
        code_bundle = manifest["code_bundle"]
        assert isinstance(code_bundle, dict)
        verify_dataset_is_mountable(
            str(manifest["dataset"]),
            str(code_bundle["sha256"]),
        )
    command = [
        sys.executable,
        str(ROOT / "scripts/run_kaggle_notebook.py"),
        str(builder.DEFAULT_NOTEBOOK),
        "--env-file", str(args.env_file),
        "--slug", "product-matching-mxbai-xsmall-balanced-training",
        "--title", "Product Matching mxbai xsmall Balanced Training",
        "--dataset", str(manifest["dataset"]),
        "--no-env-sources",
    ]
    if args.dry_run:
        command.append("--dry-run")
    elif not args.wait:
        command.append("--no-wait")
    if args.no_download:
        command.append("--no-download")
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
