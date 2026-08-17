#!/usr/bin/env python3
"""Compare the frozen S2 baseline with the x2 targeted-hard S2 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


MODELS = {"baseline_s2": "baseline", "targeted_hard_s2": "hard_trained"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-evaluations", type=Path, required=True)
    parser.add_argument("--hard-evaluations", type=Path, required=True)
    parser.add_argument("--hard-clean-audit", type=Path, required=True)
    parser.add_argument("--mining-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_report(directory: Path, split: str) -> dict[str, Any]:
    return json.loads((directory / split / "evaluation_report.json").read_text(encoding="utf-8"))


def align_audit(predictions: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    keys = ["id1", "id2"]
    if predictions[keys].equals(audit[keys]):
        result = audit.copy()
        result["score"] = predictions["score"].to_numpy(dtype=np.float32)
        return result
    left = predictions.copy()
    left["canonical_id1"] = np.minimum(left["id1"], left["id2"])
    left["canonical_id2"] = np.maximum(left["id1"], left["id2"])
    right = audit.copy()
    right["canonical_id1"] = np.minimum(right["id1"], right["id2"])
    right["canonical_id2"] = np.maximum(right["id1"], right["id2"])
    result = right.merge(
        left[["canonical_id1", "canonical_id2", "target", "score"]].rename(
            columns={"target": "prediction_target"}
        ),
        on=["canonical_id1", "canonical_id2"],
        how="left",
        validate="one_to_one",
    )
    if result["score"].isna().any() or not np.array_equal(
        result["target"].to_numpy(dtype=np.int8),
        result["prediction_target"].to_numpy(dtype=np.int8),
    ):
        raise ValueError("hard-clean audit and predictions are not aligned")
    return result


def slice_metrics(frame: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    part = frame.loc[mask]
    positive = part["target"].eq(1)
    negative = part["target"].eq(0)
    both = part["target"].nunique() == 2
    return {
        "pairs": len(part),
        "positives": int(positive.sum()),
        "prevalence": float(positive.mean()) if len(part) else None,
        "average_precision": (
            float(average_precision_score(part["target"], part["score"])) if both else None
        ),
        "roc_auc": float(roc_auc_score(part["target"], part["score"])) if both else None,
        "positive_mean_score": float(part.loc[positive, "score"].mean()) if positive.any() else None,
        "negative_mean_score": float(part.loc[negative, "score"].mean()) if negative.any() else None,
        "fnr_at_0_5": float(part.loc[positive, "score"].lt(0.5).mean()) if positive.any() else None,
        "fpr_at_0_5": float(part.loc[negative, "score"].ge(0.5).mean()) if negative.any() else None,
    }


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.6f}"
        )
    rows = [list(map(str, display.columns)), ["---"] * len(display.columns)]
    rows.extend(
        [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        for row in display.itertuples(index=False, name=None)
    )
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    mining = json.loads(args.mining_report.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    directories = {
        "baseline_s2": args.baseline_evaluations,
        "targeted_hard_s2": args.hard_evaluations,
    }
    rows = []
    reports: dict[str, dict[str, Any]] = {}
    for model, directory in directories.items():
        reports[model] = {}
        for split in config["validation_splits"]:
            report = load_report(directory, split)
            reports[model][split] = report
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "pairs": report["validation_examples"],
                    "prevalence": report["validation_positive_rate"],
                    "macro_average_precision": report["macro_average_precision"],
                    "overall_average_precision": report["overall_average_precision"],
                    "macro_roc_auc": report["macro_roc_auc"],
                    "overall_roc_auc": report["overall_roc_auc"],
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "main_metrics.csv", index=False)
    pivot = metrics.pivot(index="split", columns="model", values="macro_average_precision")
    deltas = {
        split: float(pivot.loc[split, "targeted_hard_s2"] - pivot.loc[split, "baseline_s2"])
        for split in config["validation_splits"]
    }
    success = deltas["hard_clean"] > 0 and deltas["iid"] >= -float(
        config["maximum_iid_macro_ap_drop"]
    )

    audit = pd.read_csv(args.hard_clean_audit)
    slice_rows = []
    for model, directory in directories.items():
        predictions = pd.read_parquet(directory / "hard_clean/predictions.parquet")
        frame = align_audit(predictions, audit)
        masks = {
            "hard_positive": frame["target"].eq(1),
            "hard_negative": frame["target"].eq(0),
            "numeric_conflict": frame["numeric_conflict"].astype(bool),
            "code_conflict": frame["code_conflict"].astype(bool),
            "model_conflict": frame["model_code_conflict"].astype(bool),
            "sku_vs_human_title": frame["sku_vs_human_title"].astype(bool),
        }
        for name, mask in masks.items():
            slice_rows.append({"model": model, "slice": name, **slice_metrics(frame, mask)})
    slice_frame = pd.DataFrame(slice_rows)
    slice_frame.to_csv(args.output_dir / "hard_clean_slice_comparison.csv", index=False)

    report = {
        "experiment": config["experiment"],
        "causal_change": "deterministic x2 oversampling of p85 OOF-hard, audit-eligible human train pairs",
        "llm_labels_used": False,
        "baseline_reused": True,
        "oof_folds": mining["oof_folds"],
        "mining": mining,
        "reports": reports,
        "deltas_macro_average_precision": deltas,
        "success_gate": {
            "passed": success,
            "hard_clean_delta_must_be_positive": True,
            "maximum_iid_drop": float(config["maximum_iid_macro_ap_drop"]),
        },
        "hard_clean_slices": slice_frame.to_dict("records"),
    }
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    delta_frame = pd.DataFrame(
        [{"split": split, "hard_minus_baseline_macro_ap": delta} for split, delta in deltas.items()]
    )
    delta_frame.to_csv(args.output_dir / "metric_deltas.csv", index=False)
    lines = [
        "# S2 targeted hard training",
        "",
        markdown_table(metrics),
        "",
        "## Hard-trained minus baseline macro AP",
        "",
        *[f"- `{split}`: `{delta:+.6f}`" for split, delta in deltas.items()],
        "",
        f"Success gate: **{'PASS' if success else 'FAIL'}**",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (args.output_dir / "COMPLETED").write_text("complete\n", encoding="utf-8")
    print(json.dumps({"deltas": deltas, "success_gate": success}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
