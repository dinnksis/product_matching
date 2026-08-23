#!/usr/bin/env python3
"""Download a completed Kaggle report and idempotently sync it to Google Sheets."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_kaggle_notebook as kaggle
from scripts.create_qwen_training_notebook import EXPERIMENT_SPREADSHEET_ID
from scripts.push_google_sheets_credentials_dataset import DEFAULT_KEY_PATH
from src.google_sheets_logger import load_service_account_info, sync_experiment


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", help="Kaggle kernel ref owner/slug")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY_PATH)
    parser.add_argument("--spreadsheet-id", default=EXPERIMENT_SPREADSHEET_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kaggle.load_dotenv(args.env_file)
    key_path = args.key.expanduser().resolve()
    service_account_json = key_path.read_text(encoding="utf-8")
    load_service_account_info(service_account_json)

    with tempfile.TemporaryDirectory(prefix="kaggle-experiment-sync-") as temp_dir:
        result = kaggle.run_command(
            kaggle.kaggle_command()
            + [
                "kernels",
                "output",
                args.kernel,
                "-p",
                temp_dir,
                "--force",
                "--file-pattern",
                r"(^|/)notebook_completed\.json$",
                "--page-size",
                "200",
            ],
            check=False,
        )
        if result.returncode:
            kaggle.fail(f"could not download completion report from {args.kernel!r}")
        matches = list(Path(temp_dir).rglob("notebook_completed.json"))
        if len(matches) != 1:
            kaggle.fail(
                "expected exactly one notebook_completed.json in Kaggle outputs, "
                f"found {matches}"
            )
        completion = json.loads(matches[0].read_text(encoding="utf-8"))

    if not completion.get("kaggle_kernel_ref"):
        completion["kaggle_kernel_ref"] = args.kernel
    sync_result = sync_experiment(
        spreadsheet_id=args.spreadsheet_id,
        service_account_json=service_account_json,
        completion=completion,
    )
    print(json.dumps({"status": "synced", **sync_result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
