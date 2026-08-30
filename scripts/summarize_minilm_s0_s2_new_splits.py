#!/usr/bin/env python3
"""Aggregate S0/S2 metrics across IID, hard, and OOD validation splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluations-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    rows = []
    for variant in config["variants"]:
        reports[variant] = {}
        for split in config["validation_splits"]:
            path = args.evaluations_dir / variant / split / "evaluation_report.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            reports[variant][split] = report
            rows.append(
                {
                    "serialization": variant,
                    "split": split,
                    "pairs": report["validation_examples"],
                    "positive_rate": report["validation_positive_rate"],
                    "macro_average_precision": report["macro_average_precision"],
                    "overall_average_precision": report["overall_average_precision"],
                    "macro_roc_auc": report["macro_roc_auc"],
                    "pairs_per_second": report["validation_pairs_per_second"],
                    "avg_tokens": report["avg_tokens"],
                    "p95_tokens": report["p95_tokens"],
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "s0_s2_split_comparison.csv", index=False)
    pivots = table.pivot(index="split", columns="serialization", values="macro_average_precision")
    deltas = {
        split: float(pivots.loc[split, "S2_VALUES_ONLY"] - pivots.loc[split, "S0_TITLE"])
        for split in config["validation_splits"]
    }
    gate_config = config["catboost_gate"]
    primary = str(gate_config["primary_split"])
    stress = [split for split in config["validation_splits"] if split != primary]
    gate_passed = deltas[primary] > float(gate_config["minimum_primary_delta"]) and all(
        deltas[split] >= -float(gate_config["maximum_hard_or_ood_drop"])
        for split in stress
    )
    aggregate = {
        "experiment": config["experiment"],
        "reports": reports,
        "s2_minus_s0_macro_ap": deltas,
        "catboost_gate": {
            "passed": gate_passed,
            "rule": gate_config,
        },
    }
    (args.output_dir / "aggregate_report.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# S0 vs S2 on new human validation splits",
        "",
        table.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## S2 − S0 macro AP",
        "",
    ]
    lines.extend(f"- `{split}`: `{delta:+.6f}`" for split, delta in deltas.items())
    lines.extend(["", f"CatBoost gate: **{'PASS' if gate_passed else 'FAIL'}**", ""])
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (args.output_dir / "COMPLETED").write_text("complete\n", encoding="utf-8")
    print(table.to_string(index=False), flush=True)
    print(json.dumps({"deltas": deltas, "catboost_gate_passed": gate_passed}, indent=2), flush=True)


if __name__ == "__main__":
    main()
