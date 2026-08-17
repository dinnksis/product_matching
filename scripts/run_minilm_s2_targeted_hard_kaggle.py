#!/usr/bin/env python3
"""Publish, launch, and monitor the 2xT4 MiniLM S2 targeted-hard experiment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

import create_minilm_s2_targeted_hard_notebook as builder
import create_qwen_training_notebook as shared
import run_kaggle_notebook as kaggle


ROOT = Path(__file__).resolve().parents[1]
KERNEL_SLUG = "product-matching-minilm-s2-targeted-hard"
RAW_DATASET_REF = "dinakepecheva/e-cup-human-data"
VALIDATION_DATASET_REF = "alexproger23/product-matching-validation-splits-v1"


def publish_code_dataset(env_file: Path, owner: str, dry_run: bool) -> dict[str, object]:
    kaggle.load_dotenv(env_file)
    manifest = builder.build_code_dataset(builder.DEFAULT_CODE_DATASET_DIR, owner)
    dataset_ref = str(manifest["dataset"])
    print(f"Prepared private code Dataset: {dataset_ref}")
    if dry_run:
        return manifest
    cli = kaggle.kaggle_command()
    status = kaggle.run_command(cli + ["datasets", "status", dataset_ref], check=False)
    if status.returncode == 0:
        command = cli + [
            "datasets",
            "version",
            "--path",
            str(builder.DEFAULT_CODE_DATASET_DIR),
            "--message",
            f"S2 targeted hard {str(manifest['code_bundle']['sha256'])[:12]}",
        ]
    else:
        command = cli + [
            "datasets",
            "create",
            "--path",
            str(builder.DEFAULT_CODE_DATASET_DIR),
        ]
    kaggle.run_command(command)
    import push_kaggle_training_dataset as uploader

    uploader.wait_until_ready(cli, dataset_ref)
    return manifest


def monitor(env_file: Path) -> None:
    kaggle.load_dotenv(env_file)
    username = os.getenv("KAGGLE_USERNAME", "").strip()
    if not username:
        kaggle.fail("set KAGGLE_USERNAME in .env")
    kernel_ref = f"{username}/{KERNEL_SLUG}"
    cli = kaggle.kaggle_command()
    kaggle.wait_for_kernel(
        cli,
        kernel_ref,
        poll_interval=kaggle.env_int("KAGGLE_POLL_INTERVAL_SECONDS", 30, 5),
        wait_timeout=kaggle.env_int("KAGGLE_WAIT_TIMEOUT_SECONDS", 45000, 60),
    )
    output_root = Path(os.getenv("KAGGLE_OUTPUT_DIR", "artifacts/kaggle"))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    destination = output_root / KERNEL_SLUG
    destination.mkdir(parents=True, exist_ok=True)
    kaggle.run_command(
        cli
        + [
            "kernels",
            "output",
            kernel_ref,
            "-p",
            str(destination),
            "--force",
            "--page-size",
            "200",
        ]
    )
    print(f"Outputs downloaded to: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--monitor-existing", action="store_true")
    args = parser.parse_args()
    env_file = args.env_file if args.env_file.is_absolute() else ROOT / args.env_file
    if args.monitor_existing:
        monitor(env_file)
        return
    owner = shared.dotenv_username(env_file)
    if not owner:
        raise SystemExit("Set KAGGLE_USERNAME in .env")
    manifest = publish_code_dataset(env_file, owner, args.dry_run)
    builder.DEFAULT_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(builder.build_notebook(manifest), builder.DEFAULT_NOTEBOOK)
    command = [
        sys.executable,
        str(ROOT / "scripts/run_kaggle_notebook.py"),
        str(builder.DEFAULT_NOTEBOOK),
        "--env-file",
        str(env_file),
        "--slug",
        KERNEL_SLUG,
        "--title",
        "Product Matching MiniLM S2 Targeted Hard",
        "--dataset",
        RAW_DATASET_REF,
        "--dataset",
        VALIDATION_DATASET_REF,
        "--dataset",
        str(manifest["dataset"]),
        "--no-env-sources",
        "--no-google-sheets-credentials",
    ]
    if args.dry_run:
        command.append("--dry-run")
    elif not args.wait:
        command.append("--no-wait")
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
