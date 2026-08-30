#!/usr/bin/env python3
"""Generate and submit the frozen three-model inference benchmark."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import create_inference_benchmark_notebook as builder
import run_kaggle_notebook as kaggle


KERNEL_SLUG = "product-matching-inference-backend-benchmark-v1"
KERNEL_TITLE = "Product Matching Inference Backend Benchmark v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--download-existing",
        action="store_true",
        help="Download an already completed run without creating a new kernel version",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kaggle.load_dotenv(args.env_file)
    owner = kaggle.validate_slug(os.getenv("KAGGLE_USERNAME", "").strip(), "KAGGLE_USERNAME")
    if args.download_existing:
        destination = ROOT / "artifacts" / "kaggle" / KERNEL_SLUG
        destination.mkdir(parents=True, exist_ok=True)
        kaggle.run_command(
            kaggle.kaggle_command()
            + [
                "kernels", "output", f"{owner}/{KERNEL_SLUG}",
                "-p", str(destination), "--force", "--page-size", "200",
            ]
        )
        required = [
            destination / "notebook_completed.json",
            destination / "inference_benchmark_results" / "inference_benchmark_results.csv",
            destination / "inference_benchmark_results" / "inference_benchmark_summary.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise SystemExit(f"Downloaded benchmark output is incomplete: {missing}")
        print(f"Downloaded and validated benchmark outputs: {destination}")
        return
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/create_inference_benchmark_notebook.py")],
        check=True, cwd=ROOT,
    )
    command = [
        sys.executable,
        str(ROOT / "scripts/run_kaggle_notebook.py"),
        str(builder.OUTPUT_PATH),
        "--env-file", str(args.env_file),
        "--slug", KERNEL_SLUG,
        "--title", KERNEL_TITLE,
        "--dataset", f"{owner}/{builder.CHECKPOINT_DATASET_SLUG}",
        "--dataset", builder.RAW_DATASET_REF,
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
