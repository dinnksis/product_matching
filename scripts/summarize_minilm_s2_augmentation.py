#!/usr/bin/env python3
"""Summarize the two-run MiniLM S2 augmentation experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.minilm_s2_augmentation import AUGMENTED, BASELINE, RUNS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict[str, Any]] = {}
    for run_name in RUNS:
        report_path = args.runs_dir / run_name / "training_report.json"
        if not report_path.is_file():
            raise RuntimeError(f"Missing completed report: {report_path}")
        reports[run_name] = json.loads(report_path.read_text(encoding="utf-8"))

    rows = []
    for run_name, report in reports.items():
        hard = report["hard_group_metrics"]
        rows.append(
            {
                "run": run_name,
                "attribute_shuffle": report["args"]["attribute_shuffle"],
                "random_pair_swap": report["args"]["random_pair_swap"],
                "PR_AUC": report["macro_average_precision"],
                "overall_PR_AUC": report["overall_average_precision"],
                "train_time_seconds": report["training_seconds"],
                "inference_pairs_per_second": report["validation_pairs_per_second"],
                "inference_pair_orders": report["inference_pair_orders"],
                "avg_tokens": report["avg_tokens"],
                "p95_tokens": report["p95_tokens"],
                "training_pair_swap_rate": report["training_pair_swap_rate"],
                "positive_mean_score": hard["positive_only"]["mean_score"],
                "hard_looking_macro_PR_AUC": hard["hard_looking"]["macro_average_precision"],
                "hard_looking_support": hard["hard_looking"]["support"],
            }
        )
    comparison = pd.DataFrame(rows).sort_values("run").reset_index(drop=True)
    comparison.to_csv(args.output_dir / "augmentation_comparison.csv", index=False)

    baseline = reports[BASELINE]
    augmented = reports[AUGMENTED]
    absolute_delta = float(
        augmented["macro_average_precision"] - baseline["macro_average_precision"]
    )
    relative_delta = float(absolute_delta / baseline["macro_average_precision"])
    augmented_wins = absolute_delta > 0
    best_run = AUGMENTED if augmented_wins else BASELINE
    inference_contract_equal = (
        baseline["inference_pair_orders"] == augmented["inference_pair_orders"] == 1
        and baseline["validation_examples"] == augmented["validation_examples"]
        and baseline["avg_tokens"] == augmented["avg_tokens"]
        and baseline["p95_tokens"] == augmented["p95_tokens"]
    )
    preparation = json.loads(
        (args.prepared_dir / "preparation_report.json").read_text(encoding="utf-8")
    )
    default_pipeline = {
        "serialization": "S2_VALUES_ONLY",
        "attribute_shuffle": augmented_wins,
        "random_pair_swap": augmented_wins,
        "training_only_augmentations": True,
        "validation_and_inference_attribute_order": "deterministic",
        "inference_pair_order": "A_TO_B_ONLY",
        "selected_from_experiment": "minilm_s2_combined_augmentation",
        "selected_run": best_run,
    }
    summary = {
        "primary_metric": "mean category-wise sklearn average_precision_score",
        "baseline_PR_AUC": baseline["macro_average_precision"],
        "augmented_PR_AUC": augmented["macro_average_precision"],
        "absolute_delta": absolute_delta,
        "relative_delta": relative_delta,
        "baseline_training_seconds": baseline["training_seconds"],
        "augmented_training_seconds": augmented["training_seconds"],
        "best_run": best_run,
        "augmentation_improved": augmented_wins,
        "inference_contract_equal": inference_contract_equal,
        "split": preparation["split"],
        "default_training_pipeline": default_pipeline,
        "runs": reports,
    }
    (args.output_dir / "augmentation_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "default_training_pipeline.json").write_text(
        json.dumps(default_pipeline, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "best_run.txt").write_text(best_run + "\n", encoding="utf-8")

    try:
        table = comparison.to_markdown(index=False, floatfmt=".6f")
    except ImportError:
        headers = [str(column) for column in comparison.columns]
        rendered_rows = []
        for row in comparison.itertuples(index=False, name=None):
            rendered_rows.append(
                [f"{value:.6f}" if isinstance(value, float) else str(value) for value in row]
            )
        table = "\n".join(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join("---" for _ in headers) + " |",
                *("| " + " | ".join(row) + " |" for row in rendered_rows),
            ]
        )
    report_text = "\n".join(
        [
            "# MiniLM S2 combined augmentation",
            "",
            "Human labels only; identical grouped split, base model, optimizer, LR, batch size, max_length and one epoch.",
            "Validation and inference use deterministic S2 order and exactly one A-to-B forward.",
            "",
            "## Results",
            "",
            table,
            "",
            "## Delta",
            "",
            f"- baseline macro PR-AUC: `{baseline['macro_average_precision']:.6f}`",
            f"- augmented macro PR-AUC: `{augmented['macro_average_precision']:.6f}`",
            f"- absolute delta: `{absolute_delta:+.6f}`",
            f"- relative delta: `{relative_delta:+.3%}`",
            f"- baseline/augmented train time: `{baseline['training_seconds']:.1f}s` / `{augmented['training_seconds']:.1f}s`",
            f"- equal one-forward inference contract: `{inference_contract_equal}`",
            "",
            "Positive-only PR-AUC is not reported because AP is undefined when a subset has one class; mean positive score is shown instead.",
            "",
            "## Selection",
            "",
            f"Selected run: **{best_run}**.",
            (
                "Combined shuffle+swap becomes the default training pipeline."
                if augmented_wins
                else "The unaugmented S2 pipeline remains the default."
            ),
            "",
        ]
    )
    (args.output_dir / "report.md").write_text(report_text, encoding="utf-8")
    print(table)
    print(f"BEST_RUN={best_run}")
    print(f"ABSOLUTE_DELTA={absolute_delta:+.8f}")


if __name__ == "__main__":
    main()
