#!/usr/bin/env python3
"""Evaluate every unweighted subset of four saved architecture predictions."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.create_qwen_training_notebook import EXPERIMENT_SPREADSHEET_ID
from src.google_sheets_logger import (
    ARCHITECTURE_EXPERIMENT_HEADERS,
    ARCHITECTURE_EXPERIMENTS_SHEET,
    SheetsRestClient,
    SheetsLoggerError,
    _append_with_verification,
    build_architecture_experiment_row,
    column_letter,
    ensure_architecture_table,
    service_account_token,
)


MODELS = ("gte", "rumodernbert", "bge", "minilm")
SPLITS = {
    "ordinary": "iid_validation_predictions.parquet",
    "hard": "hard_validation_predictions.parquet",
    "OOD": "ood_validation_predictions.parquet",
}
METHODS = ("mean_probability", "mean_rank")
EXPERIMENT_VERSION = "architecture_score_ensemble_v1"
KEY_COLUMNS = ("id1", "id2")
CONTEXT_COLUMNS = ("target", "category_1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-root", type=Path, default=ROOT / "preds")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "architecture_ensemble_v1",
    )
    parser.add_argument(
        "--google-key",
        type=Path,
        default=ROOT / "google-service-account.json",
    )
    parser.add_argument("--spreadsheet-id", default=EXPERIMENT_SPREADSHEET_ID)
    parser.add_argument("--no-google-sheets", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def model_combinations(models: Sequence[str] = MODELS) -> list[tuple[str, ...]]:
    return [
        combination
        for size in range(1, len(models) + 1)
        for combination in itertools.combinations(models, size)
    ]


def _validate_prediction(frame: pd.DataFrame, *, model: str, split: str) -> None:
    required = {*KEY_COLUMNS, *CONTEXT_COLUMNS, "score", "probability"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{model}/{split} is missing columns: {missing}")
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"{model}/{split} contains duplicate pair identifiers")
    if not frame["target"].isin([0.0, 1.0]).all():
        raise ValueError(f"{model}/{split} contains non-binary labels")
    scores = frame["score"].to_numpy(dtype=np.float64)
    probabilities = frame["probability"].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all() or not np.isfinite(probabilities).all():
        raise ValueError(f"{model}/{split} contains non-finite scores")
    if np.max(np.abs(scores - probabilities), initial=0.0) > 1e-7:
        raise ValueError(f"{model}/{split} score and probability differ")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError(f"{model}/{split} probability is outside [0, 1]")


def align_split(
    predictions_root: Path,
    split: str,
    filename: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    aligned: pd.DataFrame | None = None
    fingerprints: dict[str, str] = {}
    for model in MODELS:
        path = predictions_root / f"preds_{model}" / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing prediction file: {path}")
        fingerprints[f"{model}/{split}"] = sha256_file(path)
        frame = pd.read_parquet(
            path,
            columns=[*KEY_COLUMNS, *CONTEXT_COLUMNS, "score", "probability"],
        )
        _validate_prediction(frame, model=model, split=split)
        current = frame.rename(
            columns={
                "target": f"target_{model}",
                "category_1": f"category_{model}",
                "score": f"{model}_probability",
                "probability": f"probability_check_{model}",
            }
        )
        if aligned is None:
            aligned = current
        else:
            before = len(aligned)
            aligned = aligned.merge(
                current,
                on=list(KEY_COLUMNS),
                how="inner",
                validate="one_to_one",
                sort=False,
            )
            if len(aligned) != before or len(aligned) != len(current):
                raise ValueError(
                    f"{model}/{split} pair identifiers do not match other models"
                )
    assert aligned is not None
    reference = MODELS[0]
    for model in MODELS[1:]:
        if not np.array_equal(
            aligned[f"target_{reference}"].to_numpy(),
            aligned[f"target_{model}"].to_numpy(),
        ):
            raise ValueError(f"{model}/{split} labels differ after pair-ID join")
        if not np.array_equal(
            aligned[f"category_{reference}"].astype(str).to_numpy(),
            aligned[f"category_{model}"].astype(str).to_numpy(),
        ):
            raise ValueError(f"{model}/{split} categories differ after pair-ID join")

    result = aligned[[*KEY_COLUMNS]].copy()
    result["target"] = aligned[f"target_{reference}"].to_numpy(dtype=np.float64)
    result["category"] = aligned[f"category_{reference}"].astype(str).to_numpy()
    for model in MODELS:
        probability = aligned[f"{model}_probability"].to_numpy(dtype=np.float64)
        result[f"{model}_probability"] = probability
        result[f"{model}_normalized_rank"] = (
            pd.Series(probability).rank(method="average", ascending=True).to_numpy()
            / len(result)
        )
    return result, fingerprints


def macro_average_precision(frame: pd.DataFrame, score: np.ndarray) -> float:
    values = []
    for category in sorted(frame["category"].unique()):
        mask = frame["category"].to_numpy() == category
        values.append(average_precision_score(frame.loc[mask, "target"], score[mask]))
    return float(np.mean(values))


def metrics(frame: pd.DataFrame, score: np.ndarray) -> dict[str, float]:
    target = frame["target"].to_numpy(dtype=np.float64)
    return {
        "macro_average_precision": macro_average_precision(frame, score),
        "overall_average_precision": float(average_precision_score(target, score)),
        "roc_auc": float(roc_auc_score(target, score)),
        "log_loss": float(log_loss(target, np.clip(score, 1e-7, 1 - 1e-7))),
    }


def deterministic_run_id(models: Sequence[str], method: str) -> str:
    name = f"{EXPERIMENT_VERSION}|{method}|{'+'.join(models)}"
    return "ensemble_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]


def source_fingerprint(fingerprints: Mapping[str, str]) -> str:
    payload = json.dumps(fingerprints, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def completion_for_row(
    row: Mapping[str, Any],
    split_metrics: Mapping[str, Mapping[str, float]],
    fingerprints: Mapping[str, str],
    aligned_paths: Mapping[str, Path],
    completed_at_utc: str,
) -> dict[str, Any]:
    models = str(row["models"])
    method = str(row["aggregation_method"])
    return {
        "run_id": deterministic_run_id(models.split("+"), method),
        "completed_at_utc": completed_at_utc,
        "status": "complete",
        "experiment": EXPERIMENT_VERSION,
        "architecture": "score_ensemble",
        "model": models,
        "initial_checkpoint_ref": "saved_predictions_only",
        "dataset_ref": "frozen_human_iid_hard_ood",
        "kaggle_kernel_ref": "",
        "code_bundle_sha256": source_fingerprint(fingerprints),
        "serialization": "S2_VALUES_ONLY",
        "serialization_sha256": "",
        "ensemble_models": models,
        "aggregation_method": method,
        "ensemble_size": int(row["ensemble_size"]),
        "mean_ap": float(row["mean_AP"]),
        "training_report": {
            "validation_splits": {
                "iid": split_metrics["ordinary"],
                "hard": split_metrics["hard"],
                "ood": split_metrics["OOD"],
            },
            "args": {},
        },
        "artifacts": {
            "predictions": {
                "iid": str(aligned_paths["ordinary"]),
                "hard": str(aligned_paths["hard"]),
                "ood": str(aligned_paths["OOD"]),
            }
        },
        "technical_notes": (
            "Unweighted saved-score ensemble; joined by id1/id2; "
            "no fitting, stacking, or weight optimization."
        ),
    }


def sheet_client(key_path: Path, spreadsheet_id: str) -> SheetsRestClient:
    import requests

    key_json = key_path.resolve().read_text(encoding="utf-8")
    access_token: str | None = None
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            access_token = service_account_token(key_json)
            break
        except Exception as error:
            last_error = error
            if attempt + 1 < 4:
                delay = 0.5 * 2**attempt
                print(
                    f"Google OAuth retry {attempt + 1}/4 after "
                    f"{type(error).__name__}",
                    flush=True,
                )
                time.sleep(delay)
    if access_token is None:
        assert last_error is not None
        raise last_error
    return SheetsRestClient(
        spreadsheet_id=spreadsheet_id,
        access_token=access_token,
        request=requests.request,
    )


def sync_sheet_rows(
    client: SheetsRestClient,
    rows: Sequence[tuple[Mapping[str, Any], Sequence[Any]]],
) -> list[dict[str, str]]:
    """Upsert many distinct experiment rows with a bounded number of requests."""
    last_column = column_letter(len(ARCHITECTURE_EXPERIMENT_HEADERS))
    sheet = "'" + ARCHITECTURE_EXPERIMENTS_SHEET.replace("'", "''") + "'"
    metadata = client.metadata()
    sheet_item = next(
        (
            item
            for item in metadata.get("sheets", [])
            if isinstance(item, Mapping)
            and isinstance(item.get("properties"), Mapping)
            and item["properties"].get("title") == ARCHITECTURE_EXPERIMENTS_SHEET
        ),
        None,
    )
    if sheet_item is None:
        ensure_architecture_table(client)
    else:
        properties = sheet_item["properties"]
        grid = properties.get("gridProperties", {})
        current_columns = int(grid.get("columnCount", 0) or 0)
        if current_columns < len(ARCHITECTURE_EXPERIMENT_HEADERS):
            client.batch_update_spreadsheet(
                [
                    {
                        "updateSheetProperties": {
                            "properties": {
                                "sheetId": int(properties["sheetId"]),
                                "gridProperties": {
                                    "columnCount": len(
                                        ARCHITECTURE_EXPERIMENT_HEADERS
                                    )
                                },
                            },
                            "fields": "gridProperties.columnCount",
                        }
                    }
                ]
            )
    header_rows = client.get_values(f"{sheet}!A1:{last_column}1")
    existing_header = list(header_rows[0]) if header_rows else []
    while existing_header and existing_header[-1] == "":
        existing_header.pop()
    if existing_header and list(
        ARCHITECTURE_EXPERIMENT_HEADERS[: len(existing_header)]
    ) != existing_header:
        raise SheetsLoggerError(
            f"Worksheet {ARCHITECTURE_EXPERIMENTS_SHEET!r} has an incompatible header"
        )
    if existing_header != list(ARCHITECTURE_EXPERIMENT_HEADERS):
        client.update_values(
            f"{sheet}!A1:{last_column}1",
            [ARCHITECTURE_EXPERIMENT_HEADERS],
        )
    existing = client.get_values(f"{sheet}!A2:{last_column}")
    positions: dict[str, int] = {}
    for row_number, current in enumerate(existing, start=2):
        if not current or not str(current[0]).strip():
            continue
        run_id = str(current[0])
        if run_id in positions:
            raise SheetsLoggerError(
                f"Worksheet contains duplicate architecture run_id {run_id!r}"
            )
        positions[run_id] = row_number

    updates: list[tuple[str, Sequence[Any]]] = []
    additions: list[Sequence[Any]] = []
    addition_ids: set[str] = set()
    sync_rows: list[dict[str, str]] = []
    for completion, row in rows:
        run_id = str(completion["run_id"])
        if run_id in positions:
            row_number = positions[run_id]
            updates.append(
                (f"{sheet}!A{row_number}:{last_column}{row_number}", row)
            )
            action = "updated"
        else:
            additions.append(row)
            addition_ids.add(run_id)
            action = "appended"
        sync_rows.append(
            {
                "run_id": run_id,
                "models": str(
                    completion.get("ensemble_models")
                    or completion.get("model")
                    or ""
                ),
                "aggregation_method": str(
                    completion.get("aggregation_method")
                    or completion.get("analysis_type")
                    or ""
                ),
                "action": action,
            }
        )

    if updates:
        client.batch_update_values(updates)
    if additions:
        def committed() -> bool:
            current = client.get_values(f"{sheet}!A2:A")
            observed = {
                str(values[0]) for values in current if values and str(values[0]).strip()
            }
            return addition_ids <= observed

        _append_with_verification(
            client,
            f"{sheet}!A:{last_column}",
            additions,
            committed,
        )

    observed_rows = client.get_values(f"{sheet}!A2:A")
    observed_ids = {
        str(values[0])
        for values in observed_rows
        if values and str(values[0]).strip()
    }
    expected_ids = {str(completion["run_id"]) for completion, _ in rows}
    missing = sorted(expected_ids - observed_ids)
    if missing:
        raise SheetsLoggerError(
            f"Google Sheets verification is missing ensemble rows: {missing}"
        )
    return sync_rows


def evaluate(
    predictions_root: Path,
    output_dir: Path,
) -> tuple[
    pd.DataFrame,
    dict[tuple[str, str], dict[str, dict[str, float]]],
    dict[str, str],
    dict[str, Path],
]:
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_frames: dict[str, pd.DataFrame] = {}
    aligned_paths: dict[str, Path] = {}
    all_fingerprints: dict[str, str] = {}
    for split, filename in SPLITS.items():
        frame, fingerprints = align_split(predictions_root, split, filename)
        aligned_frames[split] = frame
        all_fingerprints.update(fingerprints)
        path = output_dir / f"{split.lower()}_aligned_scores.parquet"
        frame.to_parquet(path, index=False, compression="zstd")
        aligned_paths[split] = path.resolve()

    rows: list[dict[str, Any]] = []
    detailed: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for combination in model_combinations():
        models = "+".join(combination)
        for method in METHODS:
            split_results: dict[str, dict[str, float]] = {}
            row: dict[str, Any] = {
                "models": models,
                "aggregation_method": method,
                "ensemble_size": len(combination),
            }
            suffix = "probability" if method == "mean_probability" else "normalized_rank"
            for split, frame in aligned_frames.items():
                columns = [f"{model}_{suffix}" for model in combination]
                score = frame[columns].mean(axis=1).to_numpy(dtype=np.float64)
                split_results[split] = metrics(frame, score)
                row[f"{split}_AP"] = split_results[split]["macro_average_precision"]
            row["mean_AP"] = float(
                np.mean([row["ordinary_AP"], row["hard_AP"], row["OOD_AP"]])
            )
            rows.append(row)
            detailed[(models, method)] = split_results
    result = pd.DataFrame(rows).sort_values(
        ["ensemble_size", "models", "aggregation_method"],
        kind="stable",
    )
    return result.reset_index(drop=True), detailed, all_fingerprints, aligned_paths


def best_summary(result: pd.DataFrame) -> dict[str, Any]:
    def record(frame: pd.DataFrame, column: str) -> dict[str, Any]:
        row = frame.sort_values(column, ascending=False, kind="stable").iloc[0]
        return {
            "models": row["models"],
            "aggregation_method": row["aggregation_method"],
            column: float(row[column]),
        }

    singles = result[result["ensemble_size"] == 1]
    pairs = result[result["ensemble_size"] == 2]
    triples = result[result["ensemble_size"] == 3]
    all_four = result[result["ensemble_size"] == 4]
    best_single = singles.sort_values("mean_AP", ascending=False, kind="stable").iloc[0]
    best_triple = triples.sort_values("mean_AP", ascending=False, kind="stable").iloc[0]
    best_all_four = all_four.sort_values("mean_AP", ascending=False, kind="stable").iloc[0]
    return {
        "best_overall": record(result, "mean_AP"),
        "best_hard": record(result, "hard_AP"),
        "best_OOD": record(result, "OOD_AP"),
        "best_2_model": record(pairs, "mean_AP"),
        "best_3_model": record(triples, "mean_AP"),
        "best_single": record(singles, "mean_AP"),
        "best_all_4": record(all_four, "mean_AP"),
        "all_4_delta_vs_best_single": float(
            best_all_four["mean_AP"] - best_single["mean_AP"]
        ),
        "all_4_delta_vs_best_triple": float(
            best_all_four["mean_AP"] - best_triple["mean_AP"]
        ),
    }


def main() -> None:
    args = parse_args()
    started = datetime.now(timezone.utc)
    result, detailed, fingerprints, aligned_paths = evaluate(
        args.predictions_root.resolve(), args.output_dir.resolve()
    )
    csv_columns = [
        "models",
        "aggregation_method",
        "ordinary_AP",
        "hard_AP",
        "OOD_AP",
        "mean_AP",
    ]
    csv_path = args.output_dir / "ensemble_results.csv"
    result[csv_columns].to_csv(csv_path, index=False)
    summary = best_summary(result)
    summary.update(
        {
            "experiment_version": EXPERIMENT_VERSION,
            "rows": len(result),
            "source_predictions_sha256": fingerprints,
            "aligned_score_files": {
                split: str(path) for split, path in aligned_paths.items()
            },
            "csv": str(csv_path.resolve()),
            "elapsed_seconds": (
                datetime.now(timezone.utc) - started
            ).total_seconds(),
        }
    )
    summary_path = args.output_dir / "ensemble_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if not args.no_google_sheets:
        client = sheet_client(args.google_key, args.spreadsheet_id)
        completed_at = utc_now()
        pending_rows = []
        for index, row in result.iterrows():
            models = str(row["models"])
            method = str(row["aggregation_method"])
            completion = completion_for_row(
                row,
                detailed[(models, method)],
                fingerprints,
                aligned_paths,
                completed_at,
            )
            sheet_row = build_architecture_experiment_row(completion)
            pending_rows.append((completion, sheet_row))
        sync_rows = sync_sheet_rows(client, pending_rows)
        for index, sync_row in enumerate(sync_rows):
            print(
                f"Google Sheets {index + 1:02d}/{len(result)}: "
                f"{sync_row['action']} {sync_row['models']} "
                f"{sync_row['aggregation_method']}",
                flush=True,
            )
        (args.output_dir / "google_sheets_sync.json").write_text(
            json.dumps(sync_rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(result[csv_columns].to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
