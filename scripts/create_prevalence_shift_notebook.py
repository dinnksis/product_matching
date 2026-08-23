#!/usr/bin/env python3
"""Build the frozen-CatBoost prevalence-shift diagnostic notebook and input."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

import create_qwen_training_notebook as shared


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = ROOT / "notebooks" / "05_prevalence_shift_diagnostic.ipynb"
DEFAULT_DATASET_DIR = (
    ROOT / ".kaggle" / "datasets" / "product-matching-prevalence-diagnostic"
)
DATASET_SLUG = "product-matching-prevalence-diagnostic"
KERNEL_SLUG = "product-matching-prior-shift-diagnostic"
PREDICTIONS_SOURCE = (
    ROOT
    / "artifacts"
    / "kaggle"
    / "product-matching-qwen-embedding-boosting"
    / "embedding_boosting"
    / "01_names_lexical"
    / "validation_predictions.parquet"
)
MODEL_SOURCE = PREDICTIONS_SOURCE.with_name("model.cbm")
BASELINE_REPORT_SOURCE = PREDICTIONS_SOURCE.with_name("report.json")
DIAGNOSTIC_SOURCE = ROOT / "src" / "prevalence_shift_diagnostic.py"
INPUT_FILES = {
    "validation_predictions.parquet": PREDICTIONS_SOURCE,
    "model.cbm": MODEL_SOURCE,
    "baseline_report.json": BASELINE_REPORT_SOURCE,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(source).strip())


def markdown(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def build_dataset(directory: Path, owner: str) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, object]] = {}
    for destination_name, source in INPUT_FILES.items():
        if not source.is_file():
            raise FileNotFoundError(f"Required frozen artifact is missing: {source}")
        destination = directory / destination_name
        shutil.copy2(source, destination)
        files[destination_name] = {
            "source": str(source.relative_to(ROOT)),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }
    source_hash = sha256_file(DIAGNOSTIC_SOURCE)
    manifest: dict[str, object] = {
        "dataset": f"{owner}/{DATASET_SLUG}",
        "purpose": "Frozen names-only lexical CatBoost prevalence-shift diagnostic",
        "files": files,
        "diagnostic_source": {
            "path": str(DIAGNOSTIC_SOURCE.relative_to(ROOT)),
            "sha256": source_hash,
        },
    }
    (directory / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "title": "Product Matching Prevalence Diagnostic",
        "id": f"{owner}/{DATASET_SLUG}",
        "licenses": [{"name": "unknown"}],
        "description": (
            "Private frozen validation predictions and model artifact for a "
            "class-prior shift diagnostic. No training data or credentials."
        ),
    }
    (directory / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_notebook(manifest: dict[str, object]) -> nbf.NotebookNode:
    dataset_ref = str(manifest["dataset"])
    files = manifest["files"]
    assert isinstance(files, dict)
    diagnostic_source = DIAGNOSTIC_SOURCE.read_text(encoding="utf-8")
    diagnostic_sha = sha256_file(DIAGNOSTIC_SOURCE)
    cells = [
        markdown(
            """
            # Frozen lexical CatBoost: sensitivity of AP to positive prevalence

            This diagnostic performs no training, threshold selection, or
            calibration. It keeps all human-validation labels and frozen scores
            unchanged. A constant weight is assigned only to negatives to
            simulate a class-prior shift. Competition AP is always calculated
            independently in each of the 20 categories and macro-averaged.
            """
        ),
        code(
            f"""
            import hashlib
            import importlib.util
            import json
            import os
            import subprocess
            import sys
            import time
            from datetime import datetime, timezone
            from pathlib import Path

            import numpy as np
            import pandas as pd

            WORKING_ROOT = Path("/kaggle/working")
            OUTPUT_DIR = WORKING_ROOT / "prevalence_shift_diagnostic"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            KAGGLE_INPUT_ROOT = Path("/kaggle/input")
            INPUT_ROOT = None
            EXPECTED_DATASET_REF = {dataset_ref!r}
            EXPECTED_FILES = {json.dumps(files, ensure_ascii=False)!r}
            EXPECTED_DIAGNOSTIC_SHA256 = {diagnostic_sha!r}
            print(json.dumps({{
                "dataset_ref": EXPECTED_DATASET_REF,
                "input_search_root": str(KAGGLE_INPUT_ROOT),
                "output_dir": str(OUTPUT_DIR),
            }}, ensure_ascii=False, indent=2))
            """
        ),
        shared.experiment_run_initialization_cell(),
        markdown("## Verify frozen inputs and reproduce the baseline"),
        code(
            f"""
            expected_files = json.loads(EXPECTED_FILES)
            prediction_sha = expected_files["validation_predictions.parquet"]["sha256"]
            candidates = []
            for candidate in KAGGLE_INPUT_ROOT.rglob("validation_predictions.parquet"):
                if hashlib.sha256(candidate.read_bytes()).hexdigest() == prediction_sha:
                    candidates.append(candidate.parent)
            if len(candidates) != 1:
                raise RuntimeError(
                    "Expected exactly one attached directory containing the frozen "
                    f"predictions SHA, found: {{[str(path) for path in candidates]}}"
                )
            INPUT_ROOT = candidates[0]
            print(f"Resolved frozen input directory: {{INPUT_ROOT}}")
            for filename, metadata in expected_files.items():
                path = INPUT_ROOT / filename
                if not path.is_file():
                    raise FileNotFoundError(f"Missing frozen input: {{path}}")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != metadata["sha256"]:
                    raise RuntimeError(
                        f"SHA-256 mismatch for {{filename}}: {{digest}} != {{metadata['sha256']}}"
                    )
                print(f"Verified {{filename}}: {{digest}}")

            embedded_path = WORKING_ROOT / "prevalence_shift_diagnostic.py"
            embedded_path.write_text({diagnostic_source!r}, encoding="utf-8")
            if hashlib.sha256(embedded_path.read_bytes()).hexdigest() != EXPECTED_DIAGNOSTIC_SHA256:
                raise RuntimeError("Embedded diagnostic source hash mismatch")
            spec = importlib.util.spec_from_file_location(
                "product_matching_prevalence_shift_diagnostic", embedded_path
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("Could not load embedded diagnostic module")
            diagnostic = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = diagnostic
            spec.loader.exec_module(diagnostic)

            predictions = pd.read_parquet(INPUT_ROOT / "validation_predictions.parquet")
            predictions = diagnostic.validate_predictions(predictions)
            frozen_report = json.loads(
                (INPUT_ROOT / "baseline_report.json").read_text(encoding="utf-8")
            )
            print(f"Frozen validation rows: {{len(predictions):,}}")
            print(f"Frozen model SHA-256: {{expected_files['model.cbm']['sha256']}}")
            """
        ),
        markdown("## Weighted class-prior diagnostic and physical bootstrap sanity check"),
        code(
            """
            started = time.perf_counter()
            original_prevalence = float(predictions["target"].mean())
            target_prevalences = (
                original_prevalence,
                0.262,
                0.20,
                0.15,
                0.10,
                0.075,
                0.066,
                0.05,
                0.03,
            )
            result = diagnostic.run_diagnostic(
                predictions,
                target_prevalences=target_prevalences,
                bootstrap_prevalences=(0.10, 0.075, 0.05),
                bootstrap_repeats=30,
                bootstrap_seed=20260814,
                target_macro_ap=0.21,
            )
            diagnostic.save_outputs(result, OUTPUT_DIR)
            diagnostic_seconds = time.perf_counter() - started

            weighting = result["weighting_table"]
            bootstrap = result["bootstrap_summary"]
            display_columns = [
                "target_prevalence",
                "negative_weight",
                "effective_prevalence",
                "mean_category_weighted_prevalence",
                "macro_average_precision",
                "macro_roc_auc",
            ]
            print(weighting[display_columns].to_string(index=False))
            print("\\nBootstrap sanity check:")
            print(bootstrap.to_string(index=False))
            print("\\nMacro AP = 0.21 solution:")
            print(json.dumps(result["ap_021_solution"], ensure_ascii=False, indent=2))
            """
        ),
        markdown(
            """
            ## Interpretation

            AP depends on precision, and precision depends on how many negatives
            accompany a fixed number of positives. Increasing a constant weight
            on every negative therefore lowers precision at the same recall and
            lowers AP, even though every score and every positive/negative
            ordering is unchanged.

            ROC-AUC is the probability that a randomly selected positive is
            ranked above a randomly selected negative. Constant class weights do
            not change those pairwise comparisons, so weighted ROC-AUC remains
            invariant up to floating-point noise.

            If the frozen model reaches macro AP near 0.21 at some effective
            prevalence, the only valid conclusion is that a class-prior shift
            alone is mathematically sufficient to produce that scale of AP
            reduction. It does not estimate or identify the hidden public-test
            prevalence.
            """
        ),
        markdown("## Save completion report for artifacts and Google Sheets"),
        code(
            """
            baseline = result["baseline"]
            compact_weighting = json.loads(
                result["weighting_table"].to_json(
                    orient="records", force_ascii=False
                )
            )
            compact_bootstrap = json.loads(
                result["bootstrap_summary"].to_json(
                    orient="records", force_ascii=False
                )
            )
            baseline_categories = result["baseline_per_category"]
            per_category_ap = {
                str(row.category): float(row.average_precision)
                for row in baseline_categories.itertuples(index=False)
            }
            training_report = {
                "training_seconds": 0.0,
                "validation_seconds": diagnostic_seconds,
                "total_pipeline_seconds": diagnostic_seconds,
                "training_examples": 0,
                "original_training_examples": 0,
                "validation_examples": int(baseline["pairs"]),
                "validation_positive_examples": int(baseline["positive_examples"]),
                "validation_positive_rate": float(baseline["global_prevalence"]),
                "macro_average_precision": float(baseline["macro_average_precision"]),
                "overall_average_precision": float(baseline["global_average_precision"]),
                "macro_roc_auc": float(baseline["macro_roc_auc"]),
                "overall_roc_auc": float(baseline["global_roc_auc"]),
                "per_category_average_precision": per_category_ap,
                "prevalence_weighting_results": compact_weighting,
                "bootstrap_sanity_summary": compact_bootstrap,
                "ap_021_solution": result["ap_021_solution"],
                "maximum_macro_roc_auc_delta": result["maximum_macro_roc_auc_delta"],
                "frozen_model_sha256": expected_files["model.cbm"]["sha256"],
                "frozen_predictions_sha256": expected_files[
                    "validation_predictions.parquet"
                ]["sha256"],
                "args": {
                    "model": "names-only-lexical-catboost-frozen",
                    "epochs": 0,
                    "sampling": "constant-negative-weight",
                    "train_subset": "none-frozen-validation-predictions",
                    "loss_weighting": "none-no-training",
                    "symmetric_validation": False,
                    "seed": 20260814,
                    "bootstrap_repeats": 30,
                    "competition_group": "category",
                },
            }
            report_path = OUTPUT_DIR / "training_report.json"
            report_path.write_text(
                json.dumps(training_report, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            completed_at = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ).replace("+00:00", "Z")
            completion = {
                "status": "complete",
                "run_id": EXPERIMENT_RUN_ID,
                "started_at_utc": EXPERIMENT_STARTED_AT_UTC,
                "completed_at_utc": completed_at,
                "experiment": "lexical-catboost-prevalence-shift-diagnostic",
                "model": "names-only-lexical-catboost-frozen",
                "dataset_ref": EXPECTED_DATASET_REF,
                "kaggle_kernel_ref": (
                    os.getenv("KAGGLE_KERNEL_RUN_ID")
                    or os.getenv("KAGGLE_KERNEL_INFERENCE_RUN_ID")
                    or ""
                ),
                "code_bundle_sha256": EXPECTED_DIAGNOSTIC_SHA256,
                "training_wall_seconds": diagnostic_seconds,
                "training_report": training_report,
            }
            completion_path = WORKING_ROOT / "notebook_completed.json"
            completion_path.write_text(
                json.dumps(completion, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            (OUTPUT_DIR / "COMPLETED").write_text("ok\\n", encoding="utf-8")
            print(json.dumps(completion, ensure_ascii=False, indent=2))
            """
        ),
        *shared.google_sheets_tracking_cells(),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        }
    )
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"prevalence-diagnostic-cell-{index}", "exec")
    return notebook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    notebook_path = args.notebook.resolve()
    manifest = build_dataset(dataset_dir, args.owner)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(manifest), notebook_path)
    print(f"Dataset payload: {dataset_dir}")
    print(f"Notebook: {notebook_path}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
