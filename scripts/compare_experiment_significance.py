#!/usr/bin/env python3
"""Compare a completed run with a baseline and optionally sync Google Sheets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.push_google_sheets_credentials_dataset import DEFAULT_KEY_PATH
from src.experiment_significance import compare_experiment_directories
from src.google_sheets_logger import (
    DEFAULT_EXPERIMENT_SPREADSHEET_ID,
    COMPARISON_SHEETS,
    sync_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired component-level significance tests on frozen "
            "IID/hard/OOD predictions."
        )
    )
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--baseline-run-id", required=True)
    parser.add_argument("--candidate-completion", type=Path, required=True)
    parser.add_argument(
        "--experiment-group",
        choices=sorted(COMPARISON_SHEETS),
        required=True,
    )
    parser.add_argument("--permutations", type=int, default=2_000)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--comparison-output",
        type=Path,
        help="Defaults to <candidate-dir>/baseline_comparison.json",
    )
    parser.add_argument(
        "--completion-output",
        type=Path,
        help="Defaults to <candidate-dir>/completion_with_comparison.json",
    )
    parser.add_argument(
        "--sync-google-sheets",
        action="store_true",
        help="Upsert the augmented completion into experiments_v2 and its group sheet",
    )
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY_PATH)
    parser.add_argument(
        "--spreadsheet-id",
        default=DEFAULT_EXPERIMENT_SPREADSHEET_ID,
    )
    return parser.parse_args()


def read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    completion = read_json_object(args.candidate_completion.expanduser().resolve())
    candidate_run_id = str(completion.get("run_id", "")).strip()
    if not candidate_run_id:
        raise ValueError("Candidate completion has no non-empty run_id")
    candidate_dir = args.candidate_dir.expanduser().resolve()
    comparison = compare_experiment_directories(
        args.baseline_dir.expanduser().resolve(),
        candidate_dir,
        baseline_run_id=args.baseline_run_id,
        candidate_run_id=candidate_run_id,
        permutations=args.permutations,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    augmented_completion = {
        **completion,
        "experiment_group": args.experiment_group,
        "baseline_comparison": comparison,
    }
    comparison_output = (
        args.comparison_output.expanduser().resolve()
        if args.comparison_output
        else candidate_dir / "baseline_comparison.json"
    )
    completion_output = (
        args.completion_output.expanduser().resolve()
        if args.completion_output
        else candidate_dir / "completion_with_comparison.json"
    )
    write_json(comparison_output, comparison)
    write_json(completion_output, augmented_completion)

    result: dict[str, object] = {
        "comparison_output": str(comparison_output),
        "completion_output": str(completion_output),
        "experiment_group": args.experiment_group,
        "baseline_run_id": args.baseline_run_id,
        "candidate_run_id": candidate_run_id,
    }
    if args.sync_google_sheets:
        service_account_json = args.key.expanduser().resolve().read_text(
            encoding="utf-8"
        )
        sync_result = sync_experiment(
            spreadsheet_id=args.spreadsheet_id,
            service_account_json=service_account_json,
            completion=augmented_completion,
        )
        sync_output = candidate_dir / "google_sheets_comparison_sync.json"
        write_json(sync_output, sync_result)
        result["google_sheets_sync"] = sync_result
        result["google_sheets_sync_output"] = str(sync_output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
