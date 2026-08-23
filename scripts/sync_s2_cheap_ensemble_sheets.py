#!/usr/bin/env python3
"""Write the leakage-safe S2 LogisticRegression and CatBoost results to Sheets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.google_sheets_logger import safe_error_message, sync_experiment
from sync_serialization_ablation_sheets import service_account_json


DEFAULT_ARTIFACT_DIR = ROOT / "artifacts/cheap_ensemble_s2"
DEFAULT_SPREADSHEET_ID = "1CtqT52XOrFyHfFt6rCiOMlnq6snJMlsMOJ0ubH79ikA"
DEFAULT_KERNEL_REF = "dinakepecheva/product-matching-minilm-serialization-ablation"
MODEL_NAMES = {
    "logistic": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 + LogisticRegression",
    "catboost": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 + CatBoost",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--spreadsheet-id", default=DEFAULT_SPREADSHEET_ID)
    parser.add_argument("--kernel-ref", default=DEFAULT_KERNEL_REF)
    args = parser.parse_args()

    artifact_dir = args.artifacts.resolve()
    report_path = artifact_dir / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    import pandas as pd

    validation_target = pd.read_parquet(
        artifact_dir / "oof_predictions.parquet", columns=["target"]
    )["target"]
    completed_at = datetime.fromtimestamp(
        report_path.stat().st_mtime, tz=timezone.utc
    ).isoformat(timespec="seconds").replace("+00:00", "Z")
    started_at = datetime.fromtimestamp(
        report_path.stat().st_mtime - float(report["wall_seconds"]), tz=timezone.utc
    ).isoformat(timespec="seconds").replace("+00:00", "Z")
    parent_id = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"e-cup-2026/minilm-s2-cheap-ensemble/{sha256(report_path)}",
    )
    config = report["config"]
    completions = []
    for model_key in ("logistic", "catboost"):
        metric = report["metrics"][model_key]
        training_report = {
            "run_name": model_key,
            "experiment": f"minilm_s2_cheap_ensemble_{model_key}",
            "training_examples": report["pairs"],
            "validation_examples": report["pairs"],
            "validation_positive_examples": int(validation_target.sum()),
            "validation_positive_rate": float(validation_target.mean()),
            "macro_average_precision": metric["macro_average_precision"],
            "overall_average_precision": metric["overall_average_precision"],
            "macro_roc_auc": metric["macro_roc_auc"],
            "overall_roc_auc": metric["overall_roc_auc"],
            "per_category_average_precision": metric["per_category_average_precision"],
            "training_seconds": report["wall_seconds"],
            "meta_oof_protocol": report["meta_oof_protocol"],
            "transformer_retrained": False,
            "transformer_baseline_macro_average_precision": report["metrics"]["transformer"]["macro_average_precision"],
            "absolute_delta": metric["macro_average_precision"] - report["metrics"]["transformer"]["macro_average_precision"],
            "hard_slices": report["hard_slices"],
            "top_catboost_features": report["top_catboost_features"] if model_key == "catboost" else [],
            "args": {
                "model": MODEL_NAMES[model_key],
                "seed": config["seed"],
                "sampling": report["meta_oof_protocol"],
                "train_subset": report["pairs"],
                "loss_weighting": report["category_weighting"],
                "transformer_serialization": "S2_VALUES_ONLY",
                "meta_oof_folds": config["meta_oof_folds"],
                model_key: config["logistic_regression" if model_key == "logistic" else "catboost"],
            },
        }
        completions.append(
            {
                "status": "complete",
                "run_id": uuid.uuid5(parent_id, model_key).hex,
                "started_at_utc": started_at,
                "completed_at_utc": completed_at,
                "experiment": training_report["experiment"],
                "model": MODEL_NAMES[model_key],
                "dataset_ref": "local:human-S2-validation-predictions-51470",
                "kaggle_kernel_ref": args.kernel_ref,
                "code_bundle_sha256": sha256(ROOT / "src/cheap_ensemble.py"),
                "training_wall_seconds": report["wall_seconds"],
                "training_report": training_report,
            }
        )

    results = []
    try:
        with service_account_json(args) as credential:
            for completion in completions:
                result = sync_experiment(
                    spreadsheet_id=args.spreadsheet_id,
                    service_account_json=credential,
                    completion=completion,
                )
                results.append({"status": "synced", **result})
                print(
                    f"{completion['experiment']}: {result['experiment_action']}; "
                    f"category rows={result['category_metrics_count']}"
                )
    except Exception as error:
        raise RuntimeError(safe_error_message(error)) from error
    (artifact_dir / "google_sheets_sync.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
