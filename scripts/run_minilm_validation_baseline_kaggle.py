#!/usr/bin/env python3
"""Build, verify and run the three-validation MiniLM baseline on Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat as nbf

import create_cross_encoder_training_notebook as cross_builder
import create_minilm_validation_baseline_notebook as builder
import create_qwen_training_notebook as shared
import run_kaggle_notebook as kaggle


ROOT = Path(__file__).resolve().parents[1]
KERNEL_SLUG = "product-matching-minilm-iid-hard-ood-baseline-v1"
REQUIRED_FILES = {
    "validation_splits_manifest.json",
    *builder.REMOTE_FILES.values(),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_remote_dataset(dataset_ref: str, expected_manifest_hash: str) -> None:
    cli = kaggle.kaggle_command()
    status = kaggle.run_command(
        cli + ["datasets", "status", dataset_ref, "--format", "json"],
        check=False,
    )
    if status.returncode or '"ready"' not in status.stdout.lower():
        raise SystemExit(f"Kaggle Dataset {dataset_ref!r} is not ready")
    files = kaggle.run_command(
        cli + ["datasets", "files", dataset_ref, "--format", "json", "--page-size", "200"],
        check=False,
    )
    missing = sorted(name for name in REQUIRED_FILES if name not in files.stdout)
    if files.returncode or missing:
        raise SystemExit(f"Kaggle Dataset {dataset_ref!r} is missing files: {missing}")
    with tempfile.TemporaryDirectory(prefix="validation-splits-manifest-") as temp_dir:
        download = kaggle.run_command(
            cli
            + [
                "datasets", "download", dataset_ref,
                "-f", "validation_splits_manifest.json",
                "-p", temp_dir,
                "-o", "-q",
            ],
            check=False,
        )
        manifest_path = Path(temp_dir) / "validation_splits_manifest.json"
        if download.returncode or not manifest_path.is_file():
            raise SystemExit("Could not download the remote validation manifest")
        remote_hash = file_sha256(manifest_path)
    if remote_hash != expected_manifest_hash:
        raise SystemExit(
            f"Remote validation manifest is {remote_hash}, expected {expected_manifest_hash}"
        )
    print(f"Verified frozen Dataset: {dataset_ref}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--config", type=Path, default=builder.DEFAULT_CONFIG)
    parser.add_argument("--source-dir", type=Path, default=builder.DEFAULT_SOURCE_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    owner = shared.dotenv_username(args.env_file)
    if not owner:
        raise SystemExit("Set KAGGLE_USERNAME in .env")
    dataset = builder.load_manifest(args.source_dir, owner)
    config = cross_builder.load_training_config(args.config)
    notebook = builder.build_notebook(dataset, config)
    builder.DEFAULT_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, builder.DEFAULT_NOTEBOOK)
    if not args.dry_run:
        verify_remote_dataset(
            str(dataset["dataset"]),
            str(dataset["manifest_sha256"]),
        )

    command = [
        sys.executable,
        str(ROOT / "scripts/run_kaggle_notebook.py"),
        str(builder.DEFAULT_NOTEBOOK),
        "--env-file", str(args.env_file),
        "--slug", KERNEL_SLUG,
        "--title", "Product Matching MiniLM IID Hard OOD Baseline v1",
        "--dataset", str(dataset["dataset"]),
        "--no-env-sources",
    ]
    if args.dry_run:
        command.append("--dry-run")
    if args.no_wait:
        command.append("--no-wait")
    if args.no_download:
        command.append("--no-download")
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
