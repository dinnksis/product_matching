#!/usr/bin/env python3
"""Generate, submit, or download one neural benefit-router OOF notebook."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import create_benefit_router_oof_notebooks as builder
import run_kaggle_notebook as kaggle


ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "bge-v2-m3": ("product-matching-bge-benefit-oof-v2", "BGE Benefit Router OOF v2"),
    "minilm-5ep": ("product-matching-minilm-benefit-oof-v1", "MiniLM Benefit Router OOF v1"),
    "rumodernbert": ("product-matching-rumodernbert-benefit-oof-v2", "RuModernBERT Benefit Router OOF v2"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=sorted(PROFILES))
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--download-existing", action="store_true")
    parser.add_argument(
        "--skip-cli-checks",
        action="store_true",
        help="Skip flaky Kaggle CLI pre/post checks; push still uses authenticated Python API.",
    )
    return parser.parse_args()


def destination(slug: str) -> Path:
    root = Path(os.getenv("KAGGLE_OUTPUT_DIR", "artifacts/kaggle"))
    return (root if root.is_absolute() else ROOT / root) / slug


def validate_output(path: Path) -> None:
    required = [
        path / "notebook_completed.json",
        path / "google_sheets_sync.json",
    ]
    oof = list(path.rglob("oof_predictions.parquet"))
    reports = list(path.rglob("training_report.json"))
    if any(not item.is_file() for item in required) or len(oof) != 1 or len(reports) != 1:
        raise SystemExit(f"Incomplete downloaded OOF output: {path}")


def main() -> None:
    args = parse_args()
    kaggle.load_dotenv(args.env_file)
    owner = os.getenv("KAGGLE_USERNAME", "").strip()
    if not owner:
        raise SystemExit("Set KAGGLE_USERNAME in .env")
    slug, title = PROFILES[args.profile]
    output = destination(slug)
    if args.download_existing:
        output.mkdir(parents=True, exist_ok=True)
        kaggle.run_command(
            kaggle.kaggle_command()
            + ["kernels", "output", f"{owner}/{slug}", "-p", str(output), "--force"]
        )
        validate_output(output)
        print(f"Downloaded and validated: {output}")
        return

    subprocess.run(
        [sys.executable, str(ROOT / "scripts/create_benefit_router_oof_notebooks.py"), args.profile],
        cwd=ROOT,
        check=True,
    )
    command = [
        sys.executable,
        str(ROOT / "scripts/run_kaggle_notebook.py"),
        str(builder.OUTPUT_DIR / builder.NOTEBOOKS[args.profile]),
        "--env-file", str(args.env_file),
        "--slug", slug,
        "--title", title,
        "--dataset", builder.architecture.VALIDATION_DATASET_REF,
        "--dataset", builder.architecture.RAW_DATASET_REF,
        "--no-env-sources",
    ]
    if args.profile == "minilm-5ep":
        checkpoint = builder.architecture.load_configuration()["profiles"][args.profile][
            "initial_checkpoint_dataset"
        ]
        command.extend(["--dataset", str(checkpoint)])
    if args.dry_run:
        command.append("--dry-run")
    if args.no_wait:
        command.extend(["--no-wait", "--no-download"])
    if args.skip_cli_checks:
        command.append("--skip-cli-checks")
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    if not args.dry_run and not args.no_wait:
        validate_output(output)


if __name__ == "__main__":
    main()
