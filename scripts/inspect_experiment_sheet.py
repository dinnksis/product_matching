#!/usr/bin/env python3
"""Read one experiment row from the shared Google Sheet without modifying it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from src.google_sheets_logger import (
    EXPERIMENT_HEADERS,
    SheetsRestClient,
    service_account_token,
)
from sync_serialization_ablation_sheets import service_account_json


SPREADSHEET_ID = "1CtqT52XOrFyHfFt6rCiOMlnq6snJMlsMOJ0ubH79ikA"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--key", type=Path)
    args = parser.parse_args()
    with service_account_json(args) as credential:
        client = SheetsRestClient(
            spreadsheet_id=SPREADSHEET_ID,
            access_token=service_account_token(credential),
            request=requests.request,
        )
        rows = client.get_values("'experiments'!A2:AQ")
    matches = []
    for raw in rows:
        padded = list(raw) + [""] * max(0, len(EXPERIMENT_HEADERS) - len(raw))
        row = dict(zip(EXPERIMENT_HEADERS, padded))
        if args.query.casefold() not in str(row["experiment"]).casefold():
            continue
        report = {}
        try:
            report = json.loads(str(row["report_json"]) or "{}")
        except json.JSONDecodeError:
            pass
        matches.append(
            {
                key: row[key]
                for key in (
                    "run_id",
                    "started_at_utc",
                    "experiment",
                    "model",
                    "dataset_ref",
                    "kaggle_kernel_ref",
                    "training_examples",
                    "original_training_examples",
                    "validation_examples",
                    "validation_positive_rate",
                    "macro_average_precision",
                    "overall_average_precision",
                    "epochs",
                    "sampling",
                    "train_subset",
                    "symmetric_validation",
                )
            }
            | {
                "report_data_sources": report.get("data_sources"),
                "report_args": report.get("args"),
            }
        )
    print(json.dumps(matches, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
