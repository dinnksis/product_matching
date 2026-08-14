#!/usr/bin/env python3
"""Create or validate the compact experiment leaderboard worksheet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.create_qwen_training_notebook import EXPERIMENT_SPREADSHEET_ID
from scripts.push_google_sheets_credentials_dataset import DEFAULT_KEY_PATH
from src.google_sheets_logger import (
    EXPERIMENTS_SHEET,
    EXPERIMENT_HEADERS,
    SheetsRestClient,
    ensure_tables,
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
    sheet_ids = ensure_tables(client)
    print(
        json.dumps(
            {
                "spreadsheet_url": (
                    f"https://docs.google.com/spreadsheets/d/{args.spreadsheet_id}/edit"
                ),
                "worksheet": EXPERIMENTS_SHEET,
                "sheet_id": sheet_ids[EXPERIMENTS_SHEET],
                "columns": list(EXPERIMENT_HEADERS),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
