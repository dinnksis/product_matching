"""Class-prior shift diagnostic for frozen product-matching predictions.

The competition metric is mean category-wise ``average_precision_score``.
This module keeps every validation row and score fixed and changes only the
sample weight assigned to negatives.  It deliberately contains no model fit,
threshold selection, or score calibration code.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


REQUIRED_COLUMNS = ("id1", "id2", "target", "category", "predict")


def validate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Validation predictions are missing columns: {missing}")
    result = frame.loc[:, REQUIRED_COLUMNS].reset_index(drop=True).copy()
    if result[["id1", "id2"]].duplicated().any():
        raise ValueError("Validation predictions contain duplicate ordered pairs")
    if not result["target"].isin([0, 1]).all():
        raise ValueError("target must be binary")
    if not np.isfinite(result["predict"].to_numpy(dtype=np.float64)).all():
        raise ValueError("predict contains NaN or infinity")
    if result["category"].isna().any():
        raise ValueError("category contains missing values")
    for category, part in result.groupby("category", sort=True):
        if part["target"].nunique() != 2:
            raise ValueError(f"Category {category!r} does not contain both classes")
    result["target"] = result["target"].astype(np.int8)
    result["category"] = result["category"].astype(str)
    result["predict"] = result["predict"].astype(np.float64)
    return result


def negative_weight(p_original: float, p_target: float) -> float:
    if not 0.0 < p_original < 1.0:
        raise ValueError("p_original must be strictly between zero and one")
    if not 0.0 < p_target < 1.0:
        raise ValueError("p_target must be strictly between zero and one")
    return p_original * (1.0 - p_target) / (
        p_target * (1.0 - p_original)
    )


def sample_weights(target: np.ndarray, w_neg: float) -> np.ndarray:
    if not math.isfinite(w_neg) or w_neg <= 0:
        raise ValueError("w_neg must be finite and positive")
    return np.where(np.asarray(target) == 1, 1.0, w_neg).astype(np.float64)


def effective_prevalence(target: np.ndarray, weights: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return float(np.sum(target * weights) / np.sum(weights))


def grouped_metrics(
    frame: pd.DataFrame,
    weights: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Calculate the exact competition macro AP plus weighted ROC-AUC.

    Average precision and ROC-AUC are calculated independently in every
    competition category.  The category metrics are then averaged with equal
    category weight, exactly matching the competition's AP aggregation.
    """
    if len(frame) != len(weights):
        raise ValueError("frame and weights have different lengths")
    target = frame["target"].to_numpy(dtype=np.int8)
    scores = frame["predict"].to_numpy(dtype=np.float64)
    categories = frame["category"].astype(str).to_numpy()
    rows: list[dict[str, float | int | str]] = []
    for category in sorted(pd.unique(categories)):
        mask = categories == category
        category_target = target[mask]
        category_scores = scores[mask]
        category_weights = weights[mask]
        rows.append(
            {
                "category": str(category),
                "pairs": int(mask.sum()),
                "positive_examples": int(category_target.sum()),
                "unweighted_prevalence": float(category_target.mean()),
                "weighted_prevalence": effective_prevalence(
                    category_target, category_weights
                ),
                "average_precision": float(
                    average_precision_score(
                        category_target,
                        category_scores,
                        sample_weight=category_weights,
                    )
                ),
                "roc_auc": float(
                    roc_auc_score(
                        category_target,
                        category_scores,
                        sample_weight=category_weights,
                    )
                ),
            }
        )
    per_category = pd.DataFrame(rows)
    summary = {
        "global_weighted_prevalence": effective_prevalence(target, weights),
        "mean_category_weighted_prevalence": float(
            per_category["weighted_prevalence"].mean()
        ),
        "macro_average_precision": float(per_category["average_precision"].mean()),
        "global_average_precision": float(
            average_precision_score(target, scores, sample_weight=weights)
        ),
        "macro_roc_auc": float(per_category["roc_auc"].mean()),
        "global_roc_auc": float(
            roc_auc_score(target, scores, sample_weight=weights)
        ),
    }
    return summary, per_category


def evaluate_target_prevalences(
    frame: pd.DataFrame,
    target_prevalences: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = validate_predictions(frame)
    target = frame["target"].to_numpy(dtype=np.int8)
    p_original = float(target.mean())
    summaries: list[dict[str, float]] = []
    category_frames: list[pd.DataFrame] = []
    for p_target in target_prevalences:
        w_neg = negative_weight(p_original, float(p_target))
        weights = sample_weights(target, w_neg)
        summary, per_category = grouped_metrics(frame, weights)
        summaries.append(
            {
                "target_prevalence": float(p_target),
                "negative_weight": w_neg,
                "effective_prevalence": summary["global_weighted_prevalence"],
                **summary,
            }
        )
        per_category.insert(0, "negative_weight", w_neg)
        per_category.insert(0, "target_prevalence", float(p_target))
        category_frames.append(per_category)
    return pd.DataFrame(summaries), pd.concat(category_frames, ignore_index=True)


def find_prevalence_for_macro_ap(
    frame: pd.DataFrame,
    target_macro_ap: float = 0.21,
    iterations: int = 80,
) -> dict[str, float]:
    frame = validate_predictions(frame)
    target = frame["target"].to_numpy(dtype=np.int8)
    p_original = float(target.mean())

    def evaluate(p_target: float) -> tuple[float, dict[str, float]]:
        w_neg = negative_weight(p_original, p_target)
        summary, _ = grouped_metrics(frame, sample_weights(target, w_neg))
        return summary["macro_average_precision"], summary

    lower = 1e-5
    upper = p_original
    lower_ap, _ = evaluate(lower)
    upper_ap, _ = evaluate(upper)
    if not lower_ap <= target_macro_ap <= upper_ap:
        raise ValueError(
            f"Target macro AP {target_macro_ap} is outside the reachable interval "
            f"[{lower_ap}, {upper_ap}]"
        )
    for _ in range(iterations):
        midpoint = (lower + upper) / 2.0
        midpoint_ap, _ = evaluate(midpoint)
        if midpoint_ap < target_macro_ap:
            lower = midpoint
        else:
            upper = midpoint
    p_target = (lower + upper) / 2.0
    macro_ap, summary = evaluate(p_target)
    return {
        "target_macro_average_precision": float(target_macro_ap),
        "target_prevalence": p_target,
        "negative_weight": negative_weight(p_original, p_target),
        "macro_average_precision": macro_ap,
        **summary,
    }


def bootstrap_sanity_check(
    frame: pd.DataFrame,
    target_prevalences: Iterable[float],
    *,
    repeats: int = 30,
    seed: int = 20260814,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Physically resample negatives to check constant-negative weighting.

    Every positive is retained exactly once.  Within each competition category,
    negatives are sampled with replacement to approximately ``w_neg * n_neg``.
    Scores of sampled copies are unchanged.
    """
    if repeats < 2:
        raise ValueError("bootstrap repeats must be at least two")
    frame = validate_predictions(frame)
    target = frame["target"].to_numpy(dtype=np.int8)
    p_original = float(target.mean())
    category_parts = [part.reset_index(drop=True) for _, part in frame.groupby("category", sort=True)]
    detailed: list[dict[str, float | int]] = []
    for target_index, p_target in enumerate(target_prevalences):
        p_target = float(p_target)
        w_neg = negative_weight(p_original, p_target)
        for repeat in range(repeats):
            rng = np.random.default_rng(seed + target_index * 100_000 + repeat)
            sampled_parts: list[pd.DataFrame] = []
            for part in category_parts:
                positives = part.loc[part["target"].eq(1)]
                negatives = part.loc[part["target"].eq(0)]
                negative_draws = max(1, int(round(w_neg * len(negatives))))
                positions = rng.integers(0, len(negatives), size=negative_draws)
                sampled_negatives = negatives.iloc[positions]
                sampled_parts.append(
                    pd.concat([positives, sampled_negatives], ignore_index=True)
                )
            sampled = pd.concat(sampled_parts, ignore_index=True)
            metrics, _ = grouped_metrics(sampled, np.ones(len(sampled), dtype=np.float64))
            detailed.append(
                {
                    "target_prevalence": p_target,
                    "negative_weight": w_neg,
                    "repeat": repeat,
                    "sampled_pairs": len(sampled),
                    **metrics,
                }
            )
    details = pd.DataFrame(detailed)
    summaries: list[dict[str, float | int]] = []
    for p_target, part in details.groupby("target_prevalence", sort=True):
        weighted_table, _ = evaluate_target_prevalences(frame, [float(p_target)])
        reference = weighted_table.iloc[0]
        summaries.append(
            {
                "target_prevalence": float(p_target),
                "negative_weight": float(reference["negative_weight"]),
                "weighted_macro_ap": float(reference["macro_average_precision"]),
                "bootstrap_macro_ap_mean": float(part["macro_average_precision"].mean()),
                "bootstrap_macro_ap_std": float(part["macro_average_precision"].std(ddof=1)),
                "weighted_macro_roc_auc": float(reference["macro_roc_auc"]),
                "bootstrap_macro_roc_auc_mean": float(part["macro_roc_auc"].mean()),
                "bootstrap_macro_roc_auc_std": float(part["macro_roc_auc"].std(ddof=1)),
                "bootstrap_repeats": int(len(part)),
            }
        )
    return pd.DataFrame(summaries), details


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def run_diagnostic(
    frame: pd.DataFrame,
    *,
    target_prevalences: Sequence[float],
    bootstrap_prevalences: Sequence[float] = (0.10, 0.075, 0.05),
    bootstrap_repeats: int = 30,
    bootstrap_seed: int = 20260814,
    target_macro_ap: float = 0.21,
) -> dict[str, object]:
    frame = validate_predictions(frame)
    baseline, baseline_categories = grouped_metrics(
        frame, np.ones(len(frame), dtype=np.float64)
    )
    weighting, weighting_categories = evaluate_target_prevalences(
        frame, target_prevalences
    )
    bootstrap, bootstrap_details = bootstrap_sanity_check(
        frame,
        bootstrap_prevalences,
        repeats=bootstrap_repeats,
        seed=bootstrap_seed,
    )
    ap_021 = find_prevalence_for_macro_ap(frame, target_macro_ap=target_macro_ap)
    roc_delta = float(
        np.max(np.abs(weighting["macro_roc_auc"] - baseline["macro_roc_auc"]))
    )
    return {
        "baseline": {
            "pairs": len(frame),
            "positive_examples": int(frame["target"].sum()),
            "global_prevalence": float(frame["target"].mean()),
            **baseline,
        },
        "weighting_table": weighting,
        "per_category_weighting": weighting_categories,
        "baseline_per_category": baseline_categories,
        "bootstrap_summary": bootstrap,
        "bootstrap_details": bootstrap_details,
        "ap_021_solution": ap_021,
        "maximum_macro_roc_auc_delta": roc_delta,
        "configuration": {
            "target_prevalences": [float(value) for value in target_prevalences],
            "bootstrap_prevalences": [float(value) for value in bootstrap_prevalences],
            "bootstrap_repeats": bootstrap_repeats,
            "bootstrap_seed": bootstrap_seed,
            "target_macro_ap": target_macro_ap,
        },
    }


def save_outputs(result: dict[str, object], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    weighting = result["weighting_table"]
    per_category = result["per_category_weighting"]
    bootstrap = result["bootstrap_summary"]
    bootstrap_details = result["bootstrap_details"]
    baseline_categories = result["baseline_per_category"]
    assert isinstance(weighting, pd.DataFrame)
    assert isinstance(per_category, pd.DataFrame)
    assert isinstance(bootstrap, pd.DataFrame)
    assert isinstance(bootstrap_details, pd.DataFrame)
    assert isinstance(baseline_categories, pd.DataFrame)
    weighting.to_csv(output_dir / "prevalence_weighting_results.csv", index=False)
    per_category.to_csv(output_dir / "per_category_weighted_metrics.csv", index=False)
    bootstrap.to_csv(output_dir / "bootstrap_sanity_summary.csv", index=False)
    bootstrap_details.to_csv(output_dir / "bootstrap_sanity_repeats.csv", index=False)
    baseline_categories.to_csv(output_dir / "baseline_per_category.csv", index=False)

    report = {
        key: value
        for key, value in result.items()
        if not isinstance(value, pd.DataFrame)
    }
    report["weighting_table"] = _records(weighting)
    report["bootstrap_summary"] = _records(bootstrap)
    report["baseline_per_category"] = _records(baseline_categories)
    (output_dir / "diagnostic_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    ordered = weighting.sort_values("effective_prevalence")
    solution = result["ap_021_solution"]
    assert isinstance(solution, dict)
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.plot(
        100.0 * ordered["effective_prevalence"],
        ordered["macro_average_precision"],
        marker="o",
        linewidth=2,
        label="Weighted competition macro AP",
    )
    axis.scatter(
        [100.0 * float(solution["global_weighted_prevalence"])],
        [float(solution["macro_average_precision"])],
        marker="X",
        s=110,
        color="crimson",
        label="Macro AP = 0.21 solution",
        zorder=3,
    )
    axis.axhline(0.21, color="crimson", linestyle="--", linewidth=1)
    axis.set_xlabel("Global effective positive prevalence, %")
    axis.set_ylabel("Competition macro average precision")
    axis.set_title("Frozen names-only CatBoost: AP under class-prior shift")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ap_vs_prevalence.png", dpi=170)
    plt.close(fig)
