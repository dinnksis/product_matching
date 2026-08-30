#!/usr/bin/env python3
"""Generate and submit the final 5ep-checkpoint/all-human MiniLM run."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import create_minilm_5ep_full_human_notebook as builder
import run_kaggle_notebook as kaggle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLUG = "product-matching-minilm-5ep-full-human-final"
DEFAULT_TITLE = "Product Matching MiniLM 5ep Full Human Final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--notebook", type=Path, default=builder.DEFAULT_NOTEBOOK)
    parser.add_argument("--config", type=Path, default=builder.DEFAULT_CONFIG)
    parser.add_argument("--data-dataset")
    parser.add_argument(
        "--checkpoint-dataset",
        default=builder.DEFAULT_CHECKPOINT_DATASET_REF,
    )
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--skip-cli-checks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kaggle.load_dotenv(args.env_file)
    username = os.getenv("KAGGLE_USERNAME", "").strip()
    if not username:
        raise SystemExit("Set KAGGLE_USERNAME in .env")
    username = kaggle.validate_slug(username, "KAGGLE_USERNAME")
    data_dataset_ref = args.data_dataset or f"{username}/e-cup-human-data"
    checkpoint_dataset_ref = args.checkpoint_dataset.strip()
    if len(checkpoint_dataset_ref.split("/")) != 2:
        raise SystemExit("--checkpoint-dataset must be owner/dataset-slug")

    notebook_path = args.notebook if args.notebook.is_absolute() else ROOT / args.notebook
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    audit = builder.write_notebook(
        notebook_path,
        config_path=config_path,
        data_dataset_ref=data_dataset_ref,
        checkpoint_dataset_ref=checkpoint_dataset_ref,
    )
    print(
        f"Built {notebook_path.relative_to(ROOT)} with "
        f"{audit['pairs_rows']:,} full-human train pairs"
    )

    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_kaggle_notebook.py"),
        str(notebook_path),
        "--env-file",
        str(args.env_file),
        "--slug",
        args.slug,
        "--title",
        args.title,
        "--dataset",
        data_dataset_ref,
        "--dataset",
        checkpoint_dataset_ref,
        "--no-env-sources",
        "--no-google-sheets-credentials",
    ]
    if args.dry_run:
        command.append("--dry-run")
    if args.no_wait:
        command.append("--no-wait")
    if args.no_download:
        command.append("--no-download")
    if args.skip_cli_checks:
        command.append("--skip-cli-checks")
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
