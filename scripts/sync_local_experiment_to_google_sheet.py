#!/usr/bin/env python3
"""Idempotently upload a local three-split validation report to experiments_v2."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_kaggle_notebook import load_dotenv
from src.google_sheets_logger import (
    DEFAULT_EXPERIMENT_SPREADSHEET_ID,
    SheetsLoggerError,
    load_service_account_info,
    safe_error_message,
    sync_experiment,
    utc_now,
)


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_VALIDATION_SPLITS = ("iid", "hard", "ood")
DEFAULT_KEY_PATH = ROOT / "secrets" / "google-service-account.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload one completed local run to the compact experiments_v2 sheet. "
            "Repeated calls with the same run_id update the existing row."
        )
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--model")
    parser.add_argument("--dataset-ref", default="local:validation_splits_v1/human")
    parser.add_argument("--run-id")
    parser.add_argument("--started-at-utc")
    parser.add_argument("--completed-at-utc")
    parser.add_argument("--key", type=Path)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=ROOT / ".env",
        help=(
            "Optional dotenv file loaded without shell evaluation. Existing "
            "environment variables take precedence; a missing default file is fine."
        ),
    )
    parser.add_argument(
        "--spreadsheet-id",
        help=(
            "Defaults to EXPERIMENT_SPREADSHEET_ID after --env-file is loaded, "
            "then to the repository experiments_v2 spreadsheet."
        ),
    )
    parser.add_argument(
        "--completion-output",
        type=Path,
        help="Defaults to server_run_completed.json beside training_report.json",
    )
    parser.add_argument(
        "--sync-output",
        type=Path,
        help="Defaults to google_sheets_sync.json beside training_report.json",
    )
    return parser.parse_args()


def load_training_report(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        report = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SheetsLoggerError(f"Training report does not exist: {resolved}") from error
    except json.JSONDecodeError as error:
        raise SheetsLoggerError(f"Training report is not valid JSON: {resolved}") from error
    if not isinstance(report, dict):
        raise SheetsLoggerError("Training report must contain a JSON object")
    validation_splits = report.get("validation_splits")
    if not isinstance(validation_splits, Mapping):
        raise SheetsLoggerError(
            "Training report has no validation_splits mapping; refusing to upload "
            "an incomplete validation run"
        )
    missing = [name for name in REQUIRED_VALIDATION_SPLITS if name not in validation_splits]
    if missing:
        raise SheetsLoggerError(
            f"Training report is missing required validation splits: {missing}"
        )
    for split in REQUIRED_VALIDATION_SPLITS:
        metrics = validation_splits[split]
        if not isinstance(metrics, Mapping):
            raise SheetsLoggerError(f"Validation split {split!r} must be a JSON object")
        required_metrics = ("macro_average_precision", "overall_average_precision")
        absent = [name for name in required_metrics if name not in metrics]
        if absent:
            raise SheetsLoggerError(
                f"Validation split {split!r} is missing metrics: {absent}"
            )
        invalid = [
            name
            for name in required_metrics
            if isinstance(metrics[name], bool)
            or not isinstance(metrics[name], (int, float))
            or not math.isfinite(float(metrics[name]))
        ]
        if invalid:
            raise SheetsLoggerError(
                f"Validation split {split!r} has non-finite metrics: {invalid}"
            )
    return report


def _default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"server-{stamp}-{uuid.uuid4().hex[:8]}"


def build_local_completion(
    report: Mapping[str, Any],
    *,
    experiment: str,
    model: str | None,
    dataset_ref: str,
    run_id: str,
    started_at_utc: str | None,
    completed_at_utc: str | None,
) -> dict[str, Any]:
    report_args = report.get("args")
    report_args = report_args if isinstance(report_args, Mapping) else {}
    resolved_model = (model or str(report_args.get("model", ""))).strip()
    if not resolved_model:
        raise SheetsLoggerError("Model is absent from both --model and report args")
    experiment = experiment.strip()
    run_id = run_id.strip()
    if not experiment:
        raise SheetsLoggerError("Experiment name must not be empty")
    if not run_id:
        raise SheetsLoggerError("Local completion run_id must not be empty")
    return {
        "status": "complete",
        "run_id": run_id,
        "started_at_utc": started_at_utc or "",
        "completed_at_utc": completed_at_utc or utc_now(),
        "experiment": experiment,
        "model": resolved_model,
        "dataset_ref": dataset_ref.strip(),
        "kaggle_kernel_ref": "",
        "code_bundle_sha256": "",
        "training_report": dict(report),
    }


def _existing_completion(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    return payload


def load_optional_dotenv(path: Path) -> None:
    """Load an existing dotenv file with the repository's non-executing parser."""
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        load_dotenv(resolved)


def service_account_json(key_path: Path | None) -> str:
    if key_path is not None:
        resolved = key_path.expanduser().resolve()
        try:
            value = resolved.read_text(encoding="utf-8")
        except FileNotFoundError as error:
            raise SheetsLoggerError(
                f"Google service-account key does not exist: {resolved}"
            ) from error
        load_service_account_info(value)
        return value

    environment_value = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if environment_value:
        load_service_account_info(environment_value)
        return environment_value

    environment_path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON_PATH", "").strip()
    candidates = [Path(environment_path)] if environment_path else []
    candidates.append(DEFAULT_KEY_PATH)
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            value = resolved.read_text(encoding="utf-8")
            load_service_account_info(value)
            return value
    raise SheetsLoggerError(
        "Google service-account credentials are unavailable. Pass --key, set "
        "GOOGLE_SERVICE_ACCOUNT_JSON_PATH, or place the key at "
        f"{DEFAULT_KEY_PATH}"
    )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    load_optional_dotenv(args.env_file)
    spreadsheet_id = (
        args.spreadsheet_id
        if args.spreadsheet_id is not None
        else os.environ.get(
            "EXPERIMENT_SPREADSHEET_ID", DEFAULT_EXPERIMENT_SPREADSHEET_ID
        )
    )
    report_path = args.report.expanduser().resolve()
    output_dir = report_path.parent
    completion_output = (
        args.completion_output.expanduser().resolve()
        if args.completion_output
        else output_dir / "server_run_completed.json"
    )
    sync_output = (
        args.sync_output.expanduser().resolve()
        if args.sync_output
        else output_dir / "google_sheets_sync.json"
    )
    run_id = ""
    try:
        report = load_training_report(report_path)
        existing_completion = _existing_completion(completion_output)
        run_id = (
            (args.run_id or "").strip()
            or str(existing_completion.get("run_id", "")).strip()
            or _default_run_id()
        )
        completion = build_local_completion(
            report,
            experiment=args.experiment,
            model=args.model,
            dataset_ref=args.dataset_ref,
            run_id=run_id,
            started_at_utc=(
                args.started_at_utc
                or str(existing_completion.get("started_at_utc", "")).strip()
                or None
            ),
            completed_at_utc=(
                args.completed_at_utc
                or str(existing_completion.get("completed_at_utc", "")).strip()
                or None
            ),
        )
        write_json(completion_output, completion)
        result = sync_experiment(
            spreadsheet_id=spreadsheet_id,
            service_account_json=service_account_json(args.key),
            completion=completion,
        )
        payload = {"status": "synced", **result}
        write_json(sync_output, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as error:
        payload = {
            "status": "pending",
            "run_id": run_id,
            "failed_at_utc": utc_now(),
            "error": safe_error_message(error),
        }
        write_json(sync_output, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
