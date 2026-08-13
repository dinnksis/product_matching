#!/usr/bin/env python3
"""Build, submit, monitor, and validate the prevalence-shift Kaggle job."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

import create_prevalence_shift_notebook as builder
import push_kaggle_training_dataset as dataset_push
import run_kaggle_notebook as kaggle


ROOT = Path(__file__).resolve().parents[1]
TITLE = "Product Matching Prior Shift Diagnostic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--monitor-existing",
        action="store_true",
        help="wait for the already submitted kernel and download outputs",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="submit and wait instead of returning after the background launch",
    )
    return parser.parse_args()


def output_directory() -> Path:
    root = Path(os.getenv("KAGGLE_OUTPUT_DIR", "artifacts/kaggle"))
    if not root.is_absolute():
        root = ROOT / root
    return root / builder.KERNEL_SLUG


def download_outputs(cli: list[str], kernel_ref: str) -> Path:
    output_dir = output_directory()
    output_dir.mkdir(parents=True, exist_ok=True)
    kaggle.run_command(
        cli
        + [
            "kernels",
            "output",
            kernel_ref,
            "-p",
            str(output_dir),
            "--force",
            "--page-size",
            "200",
        ]
    )
    return output_dir


def validate_outputs(output_dir: Path) -> dict[str, object]:
    required = {
        "diagnostic_report.json",
        "prevalence_weighting_results.csv",
        "bootstrap_sanity_summary.csv",
        "ap_vs_prevalence.png",
        "training_report.json",
        "notebook_completed.json",
        "google_sheets_sync.json",
        "COMPLETED",
    }
    paths = {path.name: path for path in output_dir.rglob("*") if path.is_file()}
    missing = sorted(required - set(paths))
    if missing:
        raise RuntimeError(f"Downloaded Kaggle output is missing: {missing}")
    completion = json.loads(paths["notebook_completed.json"].read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        raise RuntimeError(f"Unexpected completion status: {completion.get('status')!r}")
    sheets = json.loads(paths["google_sheets_sync.json"].read_text(encoding="utf-8"))
    if sheets.get("status") != "synced":
        pending = paths.get("sheets_sync_pending.json")
        pending_text = pending.read_text(encoding="utf-8") if pending else ""
        raise RuntimeError(
            "Google Sheets synchronization did not complete: "
            + json.dumps(sheets, ensure_ascii=False)
            + (f"; pending={pending_text[:1000]}" if pending_text else "")
        )
    report = json.loads(paths["diagnostic_report.json"].read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "baseline": report["baseline"],
                "ap_021_solution": report["ap_021_solution"],
                "google_sheets_sync": sheets,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return report


def monitor(cli: list[str], kernel_ref: str) -> None:
    kaggle.wait_for_kernel(
        cli,
        kernel_ref,
        poll_interval=kaggle.env_int("KAGGLE_POLL_INTERVAL_SECONDS", 30, minimum=5),
        wait_timeout=kaggle.env_int("KAGGLE_WAIT_TIMEOUT_SECONDS", 45000, minimum=60),
    )
    validate_outputs(download_outputs(cli, kernel_ref))


def upload_dataset(cli: list[str], dataset_ref: str, dataset_dir: Path) -> None:
    status = kaggle.run_command(
        cli + ["datasets", "status", dataset_ref], check=False
    )
    if status.returncode == 0:
        command = cli + [
            "datasets",
            "version",
            "--path",
            str(dataset_dir),
            "--message",
            "Frozen lexical CatBoost prevalence diagnostic",
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
    dataset_push.wait_until_ready(cli, dataset_ref)


def main() -> None:
    args = parse_args()
    env_file = args.env_file if args.env_file.is_absolute() else ROOT / args.env_file
    kaggle.load_dotenv(env_file)
    owner = kaggle.validate_slug(
        os.getenv("KAGGLE_USERNAME", "").strip(), "KAGGLE_USERNAME"
    )
    kernel_ref = f"{owner}/{builder.KERNEL_SLUG}"
    cli = kaggle.kaggle_command()

    if args.monitor_existing:
        monitor(cli, kernel_ref)
        return

    manifest = builder.build_dataset(builder.DEFAULT_DATASET_DIR, owner)
    builder.DEFAULT_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(builder.build_notebook(manifest), builder.DEFAULT_NOTEBOOK)
    dataset_ref = str(manifest["dataset"])
    print(f"Prepared notebook: {builder.DEFAULT_NOTEBOOK}")
    print(f"Prepared Dataset payload: {builder.DEFAULT_DATASET_DIR}")
    if args.dry_run:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_kaggle_notebook.py"),
            str(builder.DEFAULT_NOTEBOOK),
            "--env-file",
            str(env_file),
            "--slug",
            builder.KERNEL_SLUG,
            "--title",
            TITLE,
            "--dataset",
            dataset_ref,
            "--no-env-sources",
            "--cpu",
            "--dry-run",
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        return

    if not os.getenv("KAGGLE_API_TOKEN", "").strip():
        kaggle.fail("set KAGGLE_API_TOKEN in .env")
    upload_dataset(cli, dataset_ref, builder.DEFAULT_DATASET_DIR)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_kaggle_notebook.py"),
        str(builder.DEFAULT_NOTEBOOK),
        "--env-file",
        str(env_file),
        "--slug",
        builder.KERNEL_SLUG,
        "--title",
        TITLE,
        "--dataset",
        dataset_ref,
        "--no-env-sources",
        "--cpu",
    ]
    if not args.wait:
        command.append("--no-wait")
    subprocess.run(command, cwd=ROOT, check=True)
    if args.wait:
        validate_outputs(output_directory())
    else:
        print(
            "Background Kaggle run submitted. Monitor with: "
            "python scripts/run_prevalence_shift_kaggle.py --monitor-existing"
        )


if __name__ == "__main__":
    main()
