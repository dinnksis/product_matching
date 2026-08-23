#!/usr/bin/env python3
"""Generate and submit the two-epoch balanced MiniLM Kaggle notebook."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

import create_cross_encoder_training_notebook as cross_builder
import create_minilm_balanced_training_notebook as builder
import create_mxbai_training_notebook as balanced_builder
from run_mxbai_kaggle import verify_dataset_is_mountable


ROOT = Path(__file__).resolve().parents[1]
KERNEL_SLUG = "product-matching-minilm-balanced-llm-2-epochs"


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
    owner = balanced_builder.shared.dotenv_username(args.env_file)
    if not owner:
        raise SystemExit("Set KAGGLE_USERNAME in .env")
    config = cross_builder.load_training_config(args.config)
    manifest = balanced_builder.build_dataset(
        balanced_builder.DEFAULT_DATASET_DIR,
        owner,
    )
    notebook = builder.build_notebook(manifest, config)
    builder.DEFAULT_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, builder.DEFAULT_NOTEBOOK)

    code_bundle = manifest["code_bundle"]
    assert isinstance(code_bundle, dict)
    if not args.dry_run:
        verify_dataset_is_mountable(
            str(manifest["dataset"]),
            str(code_bundle["sha256"]),
        )

    command = [
        sys.executable,
        str(ROOT / "scripts/run_kaggle_notebook.py"),
        str(builder.DEFAULT_NOTEBOOK),
        "--env-file", str(args.env_file),
        "--slug", KERNEL_SLUG,
        "--title", "Product Matching MiniLM Balanced LLM 2 Epochs",
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
