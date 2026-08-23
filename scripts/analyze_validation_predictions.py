#!/usr/bin/env python3
"""Create reproducible error-analysis artifacts from saved pair predictions."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pair_features import extract_product_names, name_ngram_cosine


NUMBER = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=250)
    return parser.parse_args()


def best_f1_metrics(targets: pd.Series, scores: pd.Series) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(targets, scores)
    if not len(thresholds):
        return {"best_f1": 0.0, "threshold": 0.5, "precision": 0.0, "recall": 0.0}
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    best = int(np.nanargmax(f1))
    return {
        "best_f1": float(f1[best]),
        "threshold": float(thresholds[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
    }


def binary_metrics(targets: pd.Series, scores: pd.Series, threshold: float) -> dict[str, Any]:
    positive = targets.eq(1).to_numpy()
    predicted = scores.ge(threshold).to_numpy()
    true_positive = int((positive & predicted).sum())
    false_positive = int((~positive & predicted).sum())
    false_negative = int((positive & ~predicted).sum())
    true_negative = int((~positive & ~predicted).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "threshold": float(threshold),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
    }


def numeric_tokens(text: Any) -> frozenset[str]:
    return frozenset(
        token.replace(",", ".") for token in NUMBER.findall(str(text).casefold())
    )


def add_analysis_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["name_1"] = extract_product_names(result["product_text_1"])
    result["name_2"] = extract_product_names(result["product_text_2"])
    result["name_equal"] = result["name_1"].eq(result["name_2"])
    result["name_ngram_cosine"] = name_ngram_cosine(result)
    first_numbers = result["product_text_1"].map(numeric_tokens)
    second_numbers = result["product_text_2"].map(numeric_tokens)
    result["numbers_present_both"] = np.fromiter(
        (bool(first) and bool(second) for first, second in zip(first_numbers, second_numbers)),
        dtype=bool,
        count=len(result),
    )
    result["numbers_equal"] = np.fromiter(
        (first == second for first, second in zip(first_numbers, second_numbers)),
        dtype=bool,
        count=len(result),
    )
    result["numbers_disjoint"] = np.fromiter(
        (
            bool(first) and bool(second) and first.isdisjoint(second)
            for first, second in zip(first_numbers, second_numbers)
        ),
        dtype=bool,
        count=len(result),
    )
    result["reached_max_length"] = (
        result["reached_max_length_ab"] | result["reached_max_length_ba"]
    )
    return result


def category_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for category, group in frame.groupby("category_1", sort=True):
        row = {
            "category": str(category),
            "pairs": len(group),
            "positives": int(group["target"].sum()),
            "positive_rate": float(group["target"].mean()),
            "average_precision": float(
                average_precision_score(group["target"], group["score"])
            ),
            "truncation_rate": float(group["reached_max_length"].mean()),
            "mean_order_gap": float(group["score_order_gap"].mean()),
        }
        row.update(best_f1_metrics(group["target"], group["score"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("average_precision")


def review_queue(frame: pd.DataFrame, top_k: int) -> pd.DataFrame:
    selected: dict[int, set[str]] = {}

    def mark(indices: pd.Index, reason: str) -> None:
        for index in indices:
            selected.setdefault(int(index), set()).add(reason)

    negatives = frame[frame["target"].eq(0)]
    positives = frame[frame["target"].eq(1)]
    mark(negatives.nlargest(top_k, "score").index, "model_high_score_negative")
    mark(positives.nsmallest(top_k, "score").index, "model_low_score_positive")
    mark(
        negatives.nlargest(top_k, "name_ngram_cosine").index,
        "lexically_near_duplicate_negative",
    )
    mark(
        positives.nsmallest(top_k, "name_ngram_cosine").index,
        "lexically_dissimilar_positive",
    )
    result = frame.loc[sorted(selected)].copy()
    result.insert(
        0,
        "review_reason",
        [";".join(sorted(selected[int(index)])) for index in result.index],
    )
    result["review_priority"] = np.where(
        result["target"].eq(0), result["score"], 1.0 - result["score"]
    )
    return result.sort_values("review_priority", ascending=False)


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    output_dir = args.output_dir or args.predictions.with_suffix("").with_name(
        f"{args.predictions.stem}_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = add_analysis_features(pd.read_parquet(args.predictions))
    required = {"target", "score", "score_ab", "score_ba", "category_1"}
    if missing := required - set(frame):
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    if not frame["target"].isin([0.0, 1.0]).all():
        raise ValueError("Validation targets must be binary")

    best = best_f1_metrics(frame["target"], frame["score"])
    truncated = frame["reached_max_length"]
    summary = {
        "source": str(args.predictions),
        "pairs": len(frame),
        "positives": int(frame["target"].sum()),
        "positive_rate": float(frame["target"].mean()),
        "overall_average_precision": float(
            average_precision_score(frame["target"], frame["score"])
        ),
        "ab_average_precision": float(
            average_precision_score(frame["target"], frame["score_ab"])
        ),
        "ba_average_precision": float(
            average_precision_score(frame["target"], frame["score_ba"])
        ),
        **best,
        "threshold_0_5": binary_metrics(frame["target"], frame["score"], 0.5),
        "threshold_best_f1": binary_metrics(
            frame["target"], frame["score"], best["threshold"]
        ),
        "truncation_rate": float(truncated.mean()),
        "truncated_average_precision": float(
            average_precision_score(
                frame.loc[truncated, "target"], frame.loc[truncated, "score"]
            )
        ),
        "not_truncated_average_precision": float(
            average_precision_score(
                frame.loc[~truncated, "target"], frame.loc[~truncated, "score"]
            )
        ),
        "order_gap_quantiles": {
            str(quantile): float(value)
            for quantile, value in frame["score_order_gap"].quantile(
                [0.5, 0.9, 0.95, 0.99, 1.0]
            ).items()
        },
        "equal_name_pairs": int(frame["name_equal"].sum()),
        "equal_name_negative_pairs": int(
            (frame["name_equal"] & frame["target"].eq(0)).sum()
        ),
        "high_ngram_negative_pairs": int(
            (frame["name_ngram_cosine"].ge(0.95) & frame["target"].eq(0)).sum()
        ),
        "low_ngram_positive_pairs": int(
            (frame["name_ngram_cosine"].lt(0.2) & frame["target"].eq(1)).sum()
        ),
    }

    categories = category_metrics(frame)
    threshold_rows = [
        binary_metrics(frame["target"], frame["score"], threshold)
        for threshold in sorted(set(np.arange(0.05, 1.0, 0.05)) | {best["threshold"]})
    ]
    ngram_bins = pd.cut(
        frame["name_ngram_cosine"],
        [-0.001, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 1.001],
    )
    ngram_metrics = (
        frame.assign(name_ngram_bin=ngram_bins)
        .groupby("name_ngram_bin", observed=True)
        .agg(
            pairs=("target", "size"),
            positives=("target", "sum"),
            positive_rate=("target", "mean"),
            mean_model_score=("score", "mean"),
        )
        .reset_index()
    )
    ngram_metrics["name_ngram_bin"] = ngram_metrics["name_ngram_bin"].astype(str)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    categories.to_csv(output_dir / "category_metrics.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(
        output_dir / "threshold_metrics.csv", index=False
    )
    ngram_metrics.to_csv(output_dir / "ngram_metrics.csv", index=False)
    frame[frame["target"].eq(0)].nlargest(args.top_k, "score").to_parquet(
        output_dir / "top_false_positives.parquet", index=False
    )
    frame[frame["target"].eq(1)].nsmallest(args.top_k, "score").to_parquet(
        output_dir / "top_false_negatives.parquet", index=False
    )
    frame.nlargest(args.top_k, "score_order_gap").to_parquet(
        output_dir / "top_order_sensitive.parquet", index=False
    )
    review_queue(frame, args.top_k).to_parquet(
        output_dir / "label_review_queue.parquet", index=False
    )
    frame.to_parquet(output_dir / "validation_enriched.parquet", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved analysis to {output_dir}")


if __name__ == "__main__":
    main()
