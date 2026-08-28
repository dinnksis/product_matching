#!/usr/bin/env python3
"""Measure score correlation and thresholded error overlap for four models."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.create_qwen_training_notebook import EXPERIMENT_SPREADSHEET_ID
from scripts.evaluate_architecture_ensembles import (
    MODELS,
    sheet_client,
    sync_sheet_rows,
)
from src.google_sheets_logger import build_architecture_experiment_row


SPLITS = ("ordinary", "hard", "OOD")
EXPERIMENT_VERSION = "architecture_prediction_diversity_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aligned-dir",
        type=Path,
        default=ROOT / "reports" / "architecture_ensemble_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "architecture_diversity_v1",
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


def best_f1_threshold(target: Sequence[float], score: Sequence[float]) -> dict[str, float]:
    target_array = np.asarray(target, dtype=np.int8)
    score_array = np.asarray(score, dtype=np.float64)
    precision, recall, thresholds = precision_recall_curve(target_array, score_array)
    if not len(thresholds):
        return {"threshold": 0.5, "best_f1": 0.0, "precision": 0.0, "recall": 0.0}
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    best = int(np.nanargmax(f1))
    return {
        "threshold": float(thresholds[best]),
        "best_f1": float(f1[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
    }


def _correlation(first: pd.Series, second: pd.Series, method: str) -> float:
    value = first.corr(second, method=method)
    return float(value) if pd.notna(value) else float("nan")


def _overlap(correct_a: np.ndarray, correct_b: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    a = correct_a[mask]
    b = correct_b[mask]
    return {
        "n": int(mask.sum()),
        "both_correct": int((a & b).sum()),
        "both_wrong": int((~a & ~b).sum()),
        "a_correct_b_wrong": int((a & ~b).sum()),
        "b_correct_a_wrong": int((~a & b).sum()),
    }


def _classification(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    positive = target == 1
    negative = ~positive
    tp = int((positive & prediction).sum())
    fn = int((positive & ~prediction).sum())
    tn = int((negative & ~prediction).sum())
    fp = int((negative & prediction).sum())
    return {
        "examples": int(len(target)),
        "correct": tp + tn,
        "errors": fp + fn,
        "accuracy": float((tp + tn) / len(target)),
        "true_positive": tp,
        "false_negative": fn,
        "true_negative": tn,
        "false_positive": fp,
        "precision": float(tp / max(1, tp + fp)),
        "recall": float(tp / max(1, tp + fn)),
        "specificity": float(tn / max(1, tn + fp)),
    }


def analyze(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, pd.DataFrame]]]:
    ordinary = frames["ordinary"]
    thresholds: dict[str, dict[str, float]] = {
        model: best_f1_threshold(
            ordinary["target"], ordinary[f"{model}_probability"]
        )
        for model in MODELS
    }
    threshold_table = pd.DataFrame(
        [{"model": model, **thresholds[model]} for model in MODELS]
    )

    pair_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    hard_rows: list[dict[str, Any]] = []
    matrices: dict[str, dict[str, pd.DataFrame]] = {}
    predictions_by_split: dict[str, dict[str, np.ndarray]] = {}
    correct_by_split: dict[str, dict[str, np.ndarray]] = {}

    for split, frame in frames.items():
        target = frame["target"].to_numpy(dtype=np.int8)
        score_columns = [f"{model}_probability" for model in MODELS]
        scores = frame[score_columns].rename(
            columns={f"{model}_probability": model for model in MODELS}
        )
        matrices[split] = {
            "pearson": scores.corr(method="pearson"),
            "spearman": scores.corr(method="spearman"),
        }
        predictions = {
            model: frame[f"{model}_probability"].to_numpy(dtype=np.float64)
            >= thresholds[model]["threshold"]
            for model in MODELS
        }
        correctness = {model: predictions[model] == target for model in MODELS}
        predictions_by_split[split] = predictions
        correct_by_split[split] = correctness

        positive = target == 1
        negative = ~positive
        all_mask = np.ones(len(frame), dtype=bool)
        for model in MODELS:
            other_correct_count = np.sum(
                np.column_stack(
                    [correctness[other] for other in MODELS if other != model]
                ),
                axis=1,
            )
            unique_correct = correctness[model] & (other_correct_count == 0)
            unique_errors = ~correctness[model] & (other_correct_count >= 2)
            model_rows.append(
                {
                    "model": model,
                    "split": split,
                    "threshold": thresholds[model]["threshold"],
                    **_classification(target, predictions[model]),
                    "unique_correct": int(unique_correct.sum()),
                    "unique_errors": int(unique_errors.sum()),
                    "unique_correct_positive": int((unique_correct & positive).sum()),
                    "unique_correct_negative": int((unique_correct & negative).sum()),
                    "unique_errors_positive": int((unique_errors & positive).sum()),
                    "unique_errors_negative": int((unique_errors & negative).sum()),
                }
            )

        for model_a, model_b in itertools.combinations(MODELS, 2):
            score_a = frame[f"{model_a}_probability"]
            score_b = frame[f"{model_b}_probability"]
            correct_a = correctness[model_a]
            correct_b = correctness[model_b]
            row: dict[str, Any] = {
                "split": split,
                "model_a": model_a,
                "model_b": model_b,
                "threshold_a": thresholds[model_a]["threshold"],
                "threshold_b": thresholds[model_b]["threshold"],
                "pearson": _correlation(score_a, score_b, "pearson"),
                "spearman": _correlation(score_a, score_b, "spearman"),
                "pearson_positive": _correlation(score_a[positive], score_b[positive], "pearson"),
                "spearman_positive": _correlation(score_a[positive], score_b[positive], "spearman"),
                "pearson_negative": _correlation(score_a[negative], score_b[negative], "pearson"),
                "spearman_negative": _correlation(score_a[negative], score_b[negative], "spearman"),
            }
            for prefix, mask in (
                ("overall", all_mask),
                ("positive", positive),
                ("negative", negative),
            ):
                row.update(
                    {
                        f"{prefix}_{name}": value
                        for name, value in _overlap(correct_a, correct_b, mask).items()
                    }
                )
            row["disagreement_rate"] = float(
                (row["overall_a_correct_b_wrong"] + row["overall_b_correct_a_wrong"])
                / len(frame)
            )
            row["oracle_accuracy"] = float(1.0 - row["overall_both_wrong"] / len(frame))
            pair_rows.append(row)

    hard = frames["hard"]
    hard_target = hard["target"].to_numpy(dtype=np.int8)
    hard_positive = hard_target == 1
    hard_negative = ~hard_positive
    minilm_correct = correct_by_split["hard"]["minilm"]
    for model in MODELS:
        if model == "minilm":
            continue
        model_correct = correct_by_split["hard"][model]
        hard_rows.append(
            {
                "model": model,
                "reference_model": "minilm",
                "corrected_hard_negatives": int(
                    (hard_negative & ~minilm_correct & model_correct).sum()
                ),
                "regressed_hard_negatives": int(
                    (hard_negative & minilm_correct & ~model_correct).sum()
                ),
                "net_hard_negatives": int(
                    (hard_negative & ~minilm_correct & model_correct).sum()
                    - (hard_negative & minilm_correct & ~model_correct).sum()
                ),
                "corrected_hard_positives": int(
                    (hard_positive & ~minilm_correct & model_correct).sum()
                ),
                "regressed_hard_positives": int(
                    (hard_positive & minilm_correct & ~model_correct).sum()
                ),
                "net_hard_positives": int(
                    (hard_positive & ~minilm_correct & model_correct).sum()
                    - (hard_positive & minilm_correct & ~model_correct).sum()
                ),
            }
        )

    return (
        threshold_table,
        pd.DataFrame(pair_rows),
        pd.DataFrame(model_rows),
        pd.DataFrame(hard_rows),
        matrices,
    )


def deterministic_run_id(*parts: str) -> str:
    payload = "|".join((EXPERIMENT_VERSION, *parts))
    return "diversity_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _base_completion(run_id: str, completed_at: str, source_hash: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "completed_at_utc": completed_at,
        "status": "complete",
        "experiment": EXPERIMENT_VERSION,
        "architecture": "prediction_diversity",
        "initial_checkpoint_ref": "saved_predictions_only",
        "dataset_ref": "frozen_human_iid_hard_ood",
        "code_bundle_sha256": source_hash,
        "serialization": "S2_VALUES_ONLY",
        "training_report": {
            "validation_splits": {"iid": {}, "hard": {}, "ood": {}},
            "args": {},
        },
    }


def sheet_completions(
    pairwise: pd.DataFrame,
    uniqueness: pd.DataFrame,
    hard_vs_minilm: pd.DataFrame,
    *,
    completed_at: str,
    source_hash: str,
) -> list[dict[str, Any]]:
    completions: list[dict[str, Any]] = []
    for record in pairwise.to_dict(orient="records"):
        split = str(record["split"])
        model_a = str(record["model_a"])
        model_b = str(record["model_b"])
        completion = _base_completion(
            deterministic_run_id("pairwise", split, model_a, model_b),
            completed_at,
            source_hash,
        )
        completion.update(
            {
                "model": f"{model_a} vs {model_b}",
                "analysis_type": "pairwise_error_overlap",
                "validation_split": split,
                "technical_notes": "Thresholds selected once per model on ordinary by best F1.",
                **record,
                "both_correct": record["overall_both_correct"],
                "both_wrong": record["overall_both_wrong"],
                "a_correct_b_wrong": record["overall_a_correct_b_wrong"],
                "b_correct_a_wrong": record["overall_b_correct_a_wrong"],
            }
        )
        completions.append(completion)
    for record in uniqueness.to_dict(orient="records"):
        split = str(record["split"])
        model = str(record["model"])
        completion = _base_completion(
            deterministic_run_id("uniqueness", split, model), completed_at, source_hash
        )
        completion.update(
            {
                "model": model,
                "analysis_type": "model_unique_errors",
                "validation_split": split,
                "threshold_a": record["threshold"],
                "technical_notes": "unique_correct: all other models wrong; unique_errors: model wrong and >=2/3 others correct.",
                **record,
            }
        )
        completions.append(completion)
    for record in hard_vs_minilm.to_dict(orient="records"):
        model = str(record["model"])
        completion = _base_completion(
            deterministic_run_id("hard_vs_minilm", model), completed_at, source_hash
        )
        completion.update(
            {
                "model": model,
                "analysis_type": "hard_vs_minilm",
                "validation_split": "hard",
                "technical_notes": "Corrections and regressions relative to frozen MiniLM decisions.",
                **record,
            }
        )
        completions.append(completion)
    return completions


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    source_files: dict[str, str] = {}
    for split in SPLITS:
        path = args.aligned_dir / f"{split.lower()}_aligned_scores.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing aligned score file: {path}")
        frames[split] = pd.read_parquet(path)
        source_files[split] = sha256_file(path)

    thresholds, pairwise, uniqueness, hard_vs_minilm, matrices = analyze(frames)
    thresholds.to_csv(args.output_dir / "ordinary_f1_thresholds.csv", index=False)
    pairwise.to_csv(args.output_dir / "pairwise_diversity.csv", index=False)
    uniqueness.to_csv(args.output_dir / "model_uniqueness.csv", index=False)
    hard_vs_minilm.to_csv(args.output_dir / "hard_vs_minilm.csv", index=False)
    for split, methods in matrices.items():
        for method, matrix in methods.items():
            matrix.to_csv(args.output_dir / f"{split.lower()}_{method}_correlation_matrix.csv")

    summary = {
        "experiment_version": EXPERIMENT_VERSION,
        "threshold_policy": "per-model best F1 on ordinary; frozen for hard and OOD",
        "thresholds": thresholds.set_index("model").to_dict(orient="index"),
        "source_aligned_scores_sha256": source_files,
        "pairwise_rows": len(pairwise),
        "model_uniqueness_rows": len(uniqueness),
        "hard_vs_minilm_rows": len(hard_vs_minilm),
    }
    (args.output_dir / "diversity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if not args.no_google_sheets:
        source_hash = hashlib.sha256(
            json.dumps(source_files, sort_keys=True).encode("utf-8")
        ).hexdigest()
        completions = sheet_completions(
            pairwise,
            uniqueness,
            hard_vs_minilm,
            completed_at=utc_now(),
            source_hash=source_hash,
        )
        client = sheet_client(args.google_key, args.spreadsheet_id)
        sync = sync_sheet_rows(
            client,
            [(completion, build_architecture_experiment_row(completion)) for completion in completions],
        )
        (args.output_dir / "google_sheets_sync.json").write_text(
            json.dumps(sync, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Google Sheets synchronized: {len(sync)} rows", flush=True)

    print(pairwise.to_string(index=False))
    print(hard_vs_minilm.to_string(index=False))


if __name__ == "__main__":
    main()
