#!/usr/bin/env python3
"""Recover the two augmentation rows in Google Sheets without retraining."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.google_sheets_logger import safe_error_message, sync_experiment
from sync_serialization_ablation_sheets import service_account_json


DEFAULT_ARTIFACT_DIR = (
    ROOT / "artifacts/kaggle/product-matching-minilm-s2-combined-augmentation"
)
DEFAULT_MANIFEST = (
    ROOT
    / ".kaggle/datasets/product-matching-minilm-s2-augmentation-code/minilm_s2_augmentation_manifest.json"
)
DEFAULT_SPREADSHEET_ID = "1CtqT52XOrFyHfFt6rCiOMlnq6snJMlsMOJ0ubH79ikA"
DEFAULT_KERNEL_REF = "dinakepecheva/product-matching-minilm-s2-combined-augmentation"
RUNS = ("A_BASELINE", "B_SHUFFLE_SWAP")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--spreadsheet-id", default=DEFAULT_SPREADSHEET_ID)
    parser.add_argument("--kernel-ref", default=DEFAULT_KERNEL_REF)
    args = parser.parse_args()
    artifact_dir = args.artifacts.resolve()
    output_dir = artifact_dir / "minilm_s2_combined_augmentation"
    parent_run_id = uuid.UUID(
        hex=(artifact_dir / "experiment_run_id.txt").read_text(encoding="utf-8").strip()
    )
    started_at = (artifact_dir / "experiment_started_at_utc.txt").read_text(
        encoding="utf-8"
    ).strip()
    completed_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    completions = []
    for run_name in RUNS:
        report = json.loads(
            (output_dir / "runs" / run_name / "training_report.json").read_text(
                encoding="utf-8"
            )
        )
        completions.append(
            {
                "status": "complete",
                "run_id": uuid.uuid5(parent_run_id, run_name).hex,
                "started_at_utc": started_at,
                "completed_at_utc": completed_at,
                "experiment": report["experiment"],
                "model": report["model"],
                "dataset_ref": manifest["dataset"],
                "kaggle_kernel_ref": args.kernel_ref,
                "code_bundle_sha256": manifest["code_bundle"]["sha256"],
                "training_wall_seconds": report["training_seconds"],
                "training_report": report,
            }
        )
    (output_dir / "run_completions.json").write_text(
        json.dumps({item["training_report"]["run_name"]: item for item in completions}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    sync_rows = []
    try:
        with service_account_json(args) as credential:
            for completion in completions:
                result = sync_experiment(
                    spreadsheet_id=args.spreadsheet_id,
                    service_account_json=credential,
                    completion=completion,
                )
                row = {"status": "synced", **result}
                sync_rows.append(row)
                run_name = str(completion["training_report"]["run_name"])
                (artifact_dir / f"google_sheets_sync_{run_name.lower()}.json").write_text(
                    json.dumps(row, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(
                    f"{run_name}: {result['experiment_action']}; "
                    f"category rows={result['category_metrics_count']}"
                )
    except Exception as error:
        raise RuntimeError(safe_error_message(error)) from error
    (artifact_dir / "google_sheets_recovery.json").write_text(
        json.dumps(sync_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
