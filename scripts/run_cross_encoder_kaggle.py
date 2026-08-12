#!/usr/bin/env python3
"""Generate and submit the configured cross-encoder notebook to Kaggle."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

import create_cross_encoder_training_notebook as builder


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and run the configured MiniLM cross-encoder Kaggle notebook"
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--config", type=Path, default=builder.DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--wait",
        action="store_true",
        help="wait for completion and download outputs; default is submit-and-return",
    )
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_file = args.env_file if args.env_file.is_absolute() else ROOT / args.env_file
    owner = builder.shared.dotenv_username(env_file)
    if not owner:
        raise SystemExit("Set KAGGLE_USERNAME in .env")
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    training_config = builder.load_training_config(config_path)
    manifest = builder.shared.build_dataset(builder.shared.DEFAULT_DATASET_DIR, owner)
    notebook = builder.build_notebook(manifest, training_config)
    builder.DEFAULT_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, builder.DEFAULT_NOTEBOOK)

    command = [
        sys.executable,
        str(ROOT / "scripts/run_kaggle_notebook.py"),
        str(builder.DEFAULT_NOTEBOOK),
        "--env-file",
        str(env_file),
        "--slug",
        "product-matching-minilm-training",
        "--title",
        "Product Matching MiniLM Training",
        "--dataset",
        str(manifest["dataset"]),
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
