#!/usr/bin/env python3
"""Recover S0/S2 new-validation-split rows without rerunning MiniLM."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from src.google_sheets_logger import safe_error_message, sync_experiment
from sync_serialization_ablation_sheets import service_account_json


DEFAULT_ARTIFACT_DIR = (
    ROOT / "artifacts/kaggle/product-matching-minilm-s0-s2-new-splits"
)
DEFAULT_SPREADSHEET_ID = "1CtqT52XOrFyHfFt6rCiOMlnq6snJMlsMOJ0ubH79ikA"
DEFAULT_KERNEL_REF = "dinakepecheva/product-matching-minilm-s0-s2-new-splits"
DEFAULT_DATASET_REF = "alexproger23/product-matching-validation-splits-v1"
DEFAULT_CODE_SHA256 = "44ab1e9eed5bcb69a47521a4ddd50351437c55f5c6fb708d6738ff434c30e811"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--spreadsheet-id", default=DEFAULT_SPREADSHEET_ID)
    parser.add_argument("--kernel-ref", default=DEFAULT_KERNEL_REF)
    parser.add_argument("--dataset-ref", default=DEFAULT_DATASET_REF)
    parser.add_argument("--code-sha256", default=DEFAULT_CODE_SHA256)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def main() -> int:
    args = parse_args()
    artifact_dir = args.artifacts.resolve()
    output_dir = artifact_dir / "minilm_s0_s2_new_splits"
    aggregate = json.loads(
        (output_dir / "aggregate_report.json").read_text(encoding="utf-8")
    )
    parent_run_id = uuid.UUID(
        hex=(artifact_dir / "experiment_run_id.txt").read_text(encoding="utf-8").strip()
    )
    started_at = (
        artifact_dir / "experiment_started_at_utc.txt"
    ).read_text(encoding="utf-8").strip()
    completed_at = utc_now()
    synced: list[dict[str, object]] = []

    try:
        with service_account_json(args) as credential:
            for variant, split_reports in aggregate["reports"].items():
                for split, raw_report in split_reports.items():
                    key = f"{variant}_{split}"
                    report = dict(raw_report)
                    report.setdefault(
                        "examples_per_second", report.get("validation_pairs_per_second")
                    )
                    report.setdefault(
                        "total_pipeline_seconds",
                        report.get("total_evaluation_pipeline_seconds"),
                    )
                    completion = {
                        "status": "complete",
                        "run_id": uuid.uuid5(parent_run_id, key).hex,
                        "started_at_utc": started_at,
                        "completed_at_utc": completed_at,
                        "experiment": report["experiment"],
                        "model": report["model"],
                        "dataset_ref": args.dataset_ref,
                        "kaggle_kernel_ref": args.kernel_ref,
                        "code_bundle_sha256": args.code_sha256,
                        "training_wall_seconds": report["training_seconds"],
                        "training_report": report,
                    }
                    result = sync_experiment(
                        spreadsheet_id=args.spreadsheet_id,
                        service_account_json=credential,
                        completion=completion,
                    )
                    synced.append({"variant": variant, "split": split, **result})
                    print(
                        f"{variant}/{split}: {result['experiment_action']}; "
                        f"category rows={result['category_metrics_count']}"
                    )
    except Exception as error:
        raise RuntimeError(safe_error_message(error)) from error

    destination = artifact_dir / "google_sheets_recovery.json"
    destination.write_text(
        json.dumps(synced, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Synced {len(synced)} experiment rows; summary: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
