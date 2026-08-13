#!/usr/bin/env python3
"""Create or rotate the private Kaggle Dataset holding Google credentials."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import push_kaggle_training_dataset as shared_push
import run_kaggle_notebook as kaggle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KEY_PATH = Path.home() / "Downloads" / "ecom-matching-b86a221abe49.json"
DATASET_SLUG = "ecom-matching-google-sheets-credentials"
CREDENTIAL_FILENAME = "google-service-account.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a Google service-account key to a private Kaggle Dataset"
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY_PATH)
    parser.add_argument(
        "--message",
        default="Rotate Google Sheets service-account credential",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_and_validate_key(path: Path) -> dict[str, object]:
    if not path.is_file():
        kaggle.fail(f"Google service-account key does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        kaggle.fail(f"Google service-account key is invalid JSON: {error}")
    required = {"type", "project_id", "private_key", "client_email", "token_uri"}
    if not isinstance(payload, dict) or payload.get("type") != "service_account":
        kaggle.fail("Google credential is not a service-account JSON object")
    missing = sorted(required - set(payload))
    if missing:
        kaggle.fail(f"Google service-account key is missing fields: {missing}")
    return payload


def stage_payload(directory: Path, key_path: Path, owner: str) -> None:
    key = load_and_validate_key(key_path)
    credential_path = directory / CREDENTIAL_FILENAME
    shutil.copyfile(key_path, credential_path)
    credential_path.chmod(0o600)
    metadata = {
        "title": "Ecom Matching Google Sheets Credentials",
        "id": f"{owner}/{DATASET_SLUG}",
        "licenses": [{"name": "unknown"}],
        "description": (
            "Private Google service-account credential used only to append "
            "product-matching experiment reports to Google Sheets."
        ),
    }
    (directory / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Validated service account: {key['client_email']}")


def main() -> None:
    args = parse_args()
    env_file = args.env_file if args.env_file.is_absolute() else ROOT / args.env_file
    kaggle.load_dotenv(env_file)
    owner = kaggle.validate_slug(
        os.getenv("KAGGLE_USERNAME", "").strip(),
        "KAGGLE_USERNAME",
    )
    if not args.dry_run and not os.getenv("KAGGLE_API_TOKEN", "").strip():
        kaggle.fail("set KAGGLE_API_TOKEN in .env")
    key_path = args.key.expanduser().resolve()
    dataset_ref = f"{owner}/{DATASET_SLUG}"

    with tempfile.TemporaryDirectory(prefix="kaggle-google-credentials-") as temp_dir:
        payload_dir = Path(temp_dir)
        stage_payload(payload_dir, key_path, owner)
        print(f"Prepared private credential Dataset: {dataset_ref}")
        if args.dry_run:
            print("Dry run complete; the credential was not uploaded.")
            return

        cli = kaggle.kaggle_command()
        previous_status = shared_push.dataset_status(cli, dataset_ref)
        previous_version = (
            int(previous_status.get("current_version_number", 0))
            if previous_status
            else 0
        )
        if previous_status is None:
            command = cli + [
                "datasets",
                "create",
                "--path",
                str(payload_dir),
                "--keep-tabular",
            ]
        else:
            command = cli + [
                "datasets",
                "version",
                "--path",
                str(payload_dir),
                "--message",
                args.message,
                "--keep-tabular",
            ]
        kaggle.run_command(command)
        shared_push.wait_until_ready(
            cli,
            dataset_ref,
            minimum_version=previous_version + 1,
        )
        files = kaggle.run_command(cli + ["datasets", "files", dataset_ref])
        if CREDENTIAL_FILENAME not in files.stdout:
            kaggle.fail(
                f"Uploaded Dataset does not contain {CREDENTIAL_FILENAME!r}",
                1,
            )

    print(f"Private credential Dataset is ready: https://www.kaggle.com/datasets/{dataset_ref}")


if __name__ == "__main__":
    main()
