#!/usr/bin/env python3
"""Recover the four serialization-ablation rows without rerunning training."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.google_sheets_logger import safe_error_message, sync_experiment

import run_kaggle_notebook as kaggle


DEFAULT_ARTIFACT_DIR = (
    ROOT / "artifacts/kaggle/product-matching-minilm-serialization-ablation"
)
DEFAULT_SPREADSHEET_ID = "1CtqT52XOrFyHfFt6rCiOMlnq6snJMlsMOJ0ubH79ikA"
DEFAULT_KERNEL_REF = "dinakepecheva/product-matching-minilm-serialization-ablation"
DEFAULT_CREDENTIAL_FILENAME = "google-service-account.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--spreadsheet-id", default=DEFAULT_SPREADSHEET_ID)
    parser.add_argument("--kernel-ref", default=DEFAULT_KERNEL_REF)
    return parser.parse_args()


@contextmanager
def service_account_json(args: argparse.Namespace) -> Iterator[str]:
    if args.key is not None:
        key_path = args.key.expanduser().resolve()
        if not key_path.is_file():
            raise FileNotFoundError(
                f"Google service-account key is unavailable: {key_path}"
            )
        yield key_path.read_text(encoding="utf-8")
        return

    env_file = args.env_file.resolve()
    kaggle.load_dotenv(env_file)
    dataset_ref = os.getenv("KAGGLE_GOOGLE_SHEETS_CREDENTIALS_DATASET", "").strip()
    if not dataset_ref:
        raise RuntimeError(
            "Pass --key or configure KAGGLE_GOOGLE_SHEETS_CREDENTIALS_DATASET"
        )
    with tempfile.TemporaryDirectory(prefix="serialization-sheets-recovery-") as temp:
        result = subprocess.run(
            kaggle.kaggle_command()
            + [
                "datasets",
                "download",
                dataset_ref,
                "--path",
                temp,
                "--unzip",
                "--force",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"Kaggle credential Dataset download failed with code {result.returncode}"
            )
        candidates = list(Path(temp).rglob(DEFAULT_CREDENTIAL_FILENAME))
        if len(candidates) != 1:
            raise RuntimeError(
                "Kaggle credential Dataset must contain exactly one "
                f"{DEFAULT_CREDENTIAL_FILENAME}"
            )
        yield candidates[0].read_text(encoding="utf-8")


def load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def completions(
    aggregate: dict[str, object],
    report: dict[str, object],
    kernel_ref: str,
) -> list[dict[str, object]]:
    parent_run_id = uuid.UUID(hex=str(aggregate["run_id"]))
    runs = report.get("runs")
    if not isinstance(runs, list) or len(runs) != 4:
        raise ValueError("The ablation report must contain exactly four runs")
    result: list[dict[str, object]] = []
    for training_report in runs:
        if not isinstance(training_report, dict):
            raise ValueError("Invalid training report")
        variant = str(training_report["serialization"])
        result.append(
            {
                "status": "complete",
                "run_id": uuid.uuid5(parent_run_id, variant).hex,
                "started_at_utc": aggregate["started_at_utc"],
                "completed_at_utc": aggregate["completed_at_utc"],
                "experiment": f"{aggregate['experiment']}_{variant.lower()}",
                "model": aggregate["model"],
                "dataset_ref": aggregate["dataset_ref"],
                "kaggle_kernel_ref": kernel_ref,
                "code_bundle_sha256": aggregate["code_bundle_sha256"],
                "training_wall_seconds": training_report["training_seconds"],
                "training_report": training_report,
            }
        )
    return result


def main() -> int:
    args = parse_args()
    artifact_dir = args.artifacts.resolve()
    aggregate = load_json(artifact_dir / "notebook_completed.json")
    report = load_json(artifact_dir / "serialization_ablation/ablation_report.json")
    output: list[dict[str, object]] = []
    try:
        with service_account_json(args) as credential:
            for completion in completions(aggregate, report, args.kernel_ref):
                sync_result = sync_experiment(
                    spreadsheet_id=args.spreadsheet_id,
                    service_account_json=credential,
                    completion=completion,
                )
                row = {"status": "synced", **sync_result}
                output.append(row)
                variant = str(completion["training_report"]["serialization"])
                destination = artifact_dir / f"google_sheets_sync_{variant.lower()}.json"
                destination.write_text(
                    json.dumps(row, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                pending = artifact_dir / f"google_sheets_sync_{variant.lower()}_pending.json"
                pending.unlink(missing_ok=True)
                print(
                    f"{variant}: {sync_result['experiment_action']}; "
                    f"category rows={sync_result['category_metrics_count']}"
                )
    except Exception as error:
        raise RuntimeError(safe_error_message(error)) from error
    summary_path = artifact_dir / "google_sheets_recovery.json"
    summary_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Synced {len(output)} experiment rows; summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
