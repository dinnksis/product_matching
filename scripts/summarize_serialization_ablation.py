#!/usr/bin/env python3
"""Build the final comparison and human-readable ablation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for report_path in sorted(args.runs_dir.glob("*/training_report.json")):
        reports.append(json.loads(report_path.read_text(encoding="utf-8")))
    if len(reports) != 4:
        raise RuntimeError(f"Expected four completed reports, found {len(reports)}")
    rows = [
        {
            "serialization": report["serialization"],
            "PR_AUC": report["macro_average_precision"],
            "overall_PR_AUC": report["overall_average_precision"],
            "ROC_AUC": report["macro_roc_auc"],
            "train_time": report["training_seconds"],
            "inference_speed": report["validation_pairs_per_second"],
            "avg_tokens": report["avg_tokens"],
            "p95_tokens": report["p95_tokens"],
            "max_length_fraction": report["max_length_fraction"],
        }
        for report in reports
    ]
    comparison = pd.DataFrame(rows).sort_values("serialization").reset_index(drop=True)
    best_row = comparison.sort_values(["PR_AUC", "serialization"], ascending=[False, True]).iloc[0]
    best_variant = str(best_row["serialization"])
    comparison.to_csv(args.output_dir / "serialization_comparison.csv", index=False)
    frequency = pd.read_csv(args.prepared_dir / "attribute_name_frequency.csv")
    frequency_summary = json.loads(
        (args.prepared_dir / "preparation_report.json").read_text(encoding="utf-8")
    )["frequency_threshold"]
    figure, axis = plt.subplots(figsize=(9, 5))
    supports = frequency["item_support"].clip(lower=1)
    axis.hist(supports, bins=50, log=True, color="#4472C4", alpha=0.85)
    axis.axvline(
        frequency_summary["selected_item_support_threshold"],
        color="#C00000",
        linestyle="--",
        label=f"threshold={frequency_summary['selected_item_support_threshold']}",
    )
    axis.set_xscale("log")
    axis.set_xlabel("Train items containing attribute name (log scale)")
    axis.set_ylabel("Number of attribute names (log scale)")
    axis.set_title("Attribute-name frequency on the fixed train subset")
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "attribute_name_frequency.png", dpi=150)
    plt.close(figure)
    summary = {
        "primary_metric": "mean category-wise sklearn average_precision_score",
        "best_serialization": best_variant,
        "best_macro_average_precision": float(best_row["PR_AUC"]),
        "frequency_threshold": frequency_summary,
        "runs": reports,
    }
    (args.output_dir / "ablation_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown_table = comparison.to_markdown(index=False, floatfmt=".6f")
    report_text = "\n".join(
        [
            "# MiniLM serialization ablation",
            "",
            "Primary PR-AUC is macro average precision over the 20 product categories.",
            "All variants use the same human-only split, train subset and optimization settings.",
            "",
            "## Attribute-name threshold",
            "",
            f"- selected item-support threshold: `{frequency_summary['selected_item_support_threshold']}`;",
            f"- frequent keys: `{frequency_summary['selected_frequent_keys']}`;",
            f"- achieved train attribute-occurrence coverage: `{frequency_summary['achieved_occurrence_coverage']:.3%}`.",
            "",
            "## Results",
            "",
            markdown_table,
            "",
            "## Selection",
            "",
            f"Best serialization: **{best_variant}** with macro PR-AUC `{float(best_row['PR_AUC']):.6f}`.",
            "Only its checkpoint and the S0_TITLE baseline checkpoint are retained.",
            "",
        ]
    )
    (args.output_dir / "report.md").write_text(report_text, encoding="utf-8")
    (args.output_dir / "best_variant.txt").write_text(best_variant + "\n", encoding="utf-8")
    print(markdown_table)
    print(f"BEST_SERIALIZATION={best_variant}")


if __name__ == "__main__":
    main()
