#!/usr/bin/env python3
"""Generate and launch the BGE reranker runtime benchmark on Kaggle."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

import create_bge_reranker_runtime_notebook as builder
import run_kaggle_notebook as kaggle


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    owner = builder.dotenv_username(args.env_file)
    if not owner:
        raise SystemExit("Set KAGGLE_USERNAME in .env")
    notebook = builder.build_notebook(owner)
    builder.OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, builder.OUTPUT_PATH)

    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_kaggle_notebook.py"),
        str(builder.OUTPUT_PATH),
        "--env-file", str(args.env_file),
        "--slug", builder.KERNEL_SLUG,
        "--title", "Product Matching BGE Reranker v2 M3 Runtime",
        "--dataset", f"{owner}/{builder.DATASET_SLUG}",
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
