#!/usr/bin/env python3
"""Create or validate the architecture-only experiment worksheet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.create_qwen_training_notebook import EXPERIMENT_SPREADSHEET_ID
from scripts.push_google_sheets_credentials_dataset import DEFAULT_KEY_PATH
from src.google_sheets_logger import (
    ARCHITECTURE_EXPERIMENT_HEADERS,
    ARCHITECTURE_EXPERIMENTS_SHEET,
    SheetsRestClient,
    ensure_architecture_table,
    service_account_token,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY_PATH)
    parser.add_argument("--spreadsheet-id", default=EXPERIMENT_SPREADSHEET_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import requests

    key_json = args.key.expanduser().resolve().read_text(encoding="utf-8")
    client = SheetsRestClient(
        spreadsheet_id=args.spreadsheet_id,
        access_token=service_account_token(key_json),
        request=requests.request,
    )
    sheet_id = ensure_architecture_table(client)
    print(
        json.dumps(
            {
                "spreadsheet_url": (
                    f"https://docs.google.com/spreadsheets/d/"
                    f"{args.spreadsheet_id}/edit"
                ),
                "worksheet": ARCHITECTURE_EXPERIMENTS_SHEET,
                "sheet_id": sheet_id,
                "columns": list(ARCHITECTURE_EXPERIMENT_HEADERS),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
