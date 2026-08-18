"""Paired significance tests for frozen product-matching validation splits."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


SPLITS = ("iid", "hard", "ood")
PREDICTION_FILENAMES = {
    split: f"{split}_validation_predictions.parquet" for split in SPLITS
}


class SignificanceError(ValueError):
    """Raised when two prediction artifacts cannot be compared safely."""


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        parent = self.parent.setdefault(value, value)
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _score_column(frame: pd.DataFrame) -> str:
    for name in ("score", "predict"):
        if name in frame.columns:
            return name
    raise SignificanceError("Prediction artifact has neither 'score' nor 'predict'")


def _category_column(frame: pd.DataFrame) -> str:
    for name in ("category", "category_1"):
        if name in frame.columns:
            return name
    raise SignificanceError(
        "Prediction artifact has neither 'category' nor 'category_1'"
    )


def read_prediction_artifact(path: Path) -> pd.DataFrame:
    """Read only columns needed by the paired comparison."""
    if not path.is_file():
        raise SignificanceError(f"Prediction artifact is missing: {path}")
    schema = pd.read_parquet(path, columns=[])
    available = set(schema.columns)
    # Some parquet engines return no columns for columns=[]; fall back to the
    # lightweight pyarrow schema rather than loading product texts and diagnostics.
    if not available:
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise SignificanceError(
                "pyarrow is required to inspect prediction parquet columns"
            ) from error
        available = set(parquet.ParquetFile(path).schema.names)
    score = next((name for name in ("score", "predict") if name in available), None)
    category = next(
        (name for name in ("category", "category_1") if name in available),
        None,
    )
    required = {"id1", "id2", "target"}
    missing = required - available
    if missing:
        raise SignificanceError(
            f"Prediction artifact is missing columns: {sorted(missing)}"
        )
    if score is None:
        raise SignificanceError("Prediction artifact has neither 'score' nor 'predict'")
    if category is None:
        raise SignificanceError(
            "Prediction artifact has neither 'category' nor 'category_1'"
        )
    return pd.read_parquet(
        path,
        columns=["id1", "id2", "target", category, score],
    )


def _pair_key(id1: Any, id2: Any) -> tuple[str, str]:
    left, right = str(id1), str(id2)
    return (left, right) if left <= right else (right, left)


def align_predictions(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> pd.DataFrame:
    """Align two prediction frames and validate the frozen-pair contract."""

    def normalized(frame: pd.DataFrame, score_name: str) -> pd.DataFrame:
        required = {"id1", "id2", "target"}
        missing = required - set(frame.columns)
        if missing:
            raise SignificanceError(
                f"Prediction artifact is missing columns: {sorted(missing)}"
            )
        score_column = _score_column(frame)
        category_column = _category_column(frame)
        result = pd.DataFrame(
            {
                "id1": frame["id1"].astype(str),
                "id2": frame["id2"].astype(str),
                "target": pd.to_numeric(frame["target"], errors="raise"),
                "category": frame[category_column].astype(str),
                score_name: pd.to_numeric(frame[score_column], errors="raise"),
            }
        )
        result["pair_key"] = [
            _pair_key(left, right)
            for left, right in zip(result["id1"], result["id2"], strict=True)
        ]
        if result["pair_key"].duplicated().any():
            duplicate = result.loc[result["pair_key"].duplicated(), "pair_key"].iloc[0]
            raise SignificanceError(f"Prediction artifact contains duplicate pair {duplicate}")
        return result.set_index("pair_key", drop=True).sort_index()

    baseline_normalized = normalized(baseline, "baseline_score")
    candidate_normalized = normalized(candidate, "candidate_score")
    if not baseline_normalized.index.equals(candidate_normalized.index):
        missing_candidate = baseline_normalized.index.difference(
            candidate_normalized.index
        )
        missing_baseline = candidate_normalized.index.difference(
            baseline_normalized.index
        )
        raise SignificanceError(
            "Candidate and baseline pair sets differ: "
            f"missing_candidate={len(missing_candidate)}, "
            f"missing_baseline={len(missing_baseline)}"
        )

    if not np.array_equal(
        baseline_normalized["target"].to_numpy(),
        candidate_normalized["target"].to_numpy(),
    ):
        raise SignificanceError("Candidate and baseline targets differ")
    if not np.array_equal(
        baseline_normalized["category"].to_numpy(),
        candidate_normalized["category"].to_numpy(),
    ):
        raise SignificanceError("Candidate and baseline categories differ")

    aligned = baseline_normalized[
        ["id1", "id2", "target", "category", "baseline_score"]
    ].copy()
    aligned["candidate_score"] = candidate_normalized["candidate_score"]
    for column in ("target", "baseline_score", "candidate_score"):
        values = aligned[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise SignificanceError(f"Prediction column {column!r} is not finite")
    if not set(np.unique(aligned["target"])) <= {0.0, 1.0}:
        raise SignificanceError("Prediction targets must be binary 0/1")
    return aligned.reset_index(drop=True)


def component_codes(aligned: pd.DataFrame) -> np.ndarray:
    """Return a connected-component code for every validation pair."""
    union_find = _UnionFind()
    for left, right in zip(aligned["id1"], aligned["id2"], strict=True):
        union_find.union(str(left), str(right))
    roots = [union_find.find(str(left)) for left in aligned["id1"]]
    codes, _ = pd.factorize(np.asarray(roots, dtype=object), sort=True)
    components = pd.Series(codes, index=aligned.index)
    category_counts = aligned.assign(_component=codes).groupby("_component")[
        "category"
    ].nunique()
    if (category_counts > 1).any():
        raise SignificanceError("A validation component spans multiple categories")
    return components.to_numpy(dtype=np.int64)


def macro_average_precision(
    target: np.ndarray,
    scores: np.ndarray,
    categories: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> float:
    values: list[float] = []
    for category in np.unique(categories):
        selected = categories == category
        category_target = target[selected]
        weights = sample_weight[selected] if sample_weight is not None else None
        if weights is not None:
            positive_weight = float(weights[category_target == 1].sum())
            if positive_weight <= 0:
                return float("nan")
        elif not np.any(category_target == 1):
            raise SignificanceError(f"Category {category!r} has no positive examples")
        values.append(
            float(
                average_precision_score(
                    category_target,
                    scores[selected],
                    sample_weight=weights,
                )
            )
        )
    if not values:
        raise SignificanceError("Prediction artifact is empty")
    return float(np.mean(values))


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return Holm step-down adjusted p-values, preserving the input keys."""
    if not p_values:
        return {}
    for name, value in p_values.items():
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise SignificanceError(f"Invalid p-value for {name!r}: {value!r}")
    ordered = sorted(p_values, key=lambda name: (p_values[name], name))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, (count - rank) * float(p_values[name]))
        adjusted[name] = min(1.0, running)
    return {name: adjusted[name] for name in p_values}


def compare_prediction_frames(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    permutations: int = 2_000,
    bootstrap_resamples: int = 2_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare macro AP with a paired component permutation and bootstrap."""
    if permutations < 1 or bootstrap_resamples < 1:
        raise ValueError("permutations and bootstrap_resamples must be positive")
    aligned = align_predictions(baseline, candidate)
    target = aligned["target"].to_numpy(dtype=np.int8)
    categories = aligned["category"].to_numpy(dtype=str)
    baseline_scores = aligned["baseline_score"].to_numpy(dtype=np.float64)
    candidate_scores = aligned["candidate_score"].to_numpy(dtype=np.float64)
    components = component_codes(aligned)
    component_count = int(components.max()) + 1
    random = np.random.default_rng(seed)

    baseline_ap = macro_average_precision(target, baseline_scores, categories)
    candidate_ap = macro_average_precision(target, candidate_scores, categories)
    observed_delta = candidate_ap - baseline_ap

    extreme = 0
    tolerance = np.finfo(np.float64).eps * 16
    for _ in range(permutations):
        swap = random.integers(0, 2, size=component_count, dtype=np.int8).astype(bool)
        row_swap = swap[components]
        permuted_candidate = np.where(
            row_swap,
            baseline_scores,
            candidate_scores,
        )
        permuted_baseline = np.where(
            row_swap,
            candidate_scores,
            baseline_scores,
        )
        permuted_delta = macro_average_precision(
            target,
            permuted_candidate,
            categories,
        ) - macro_average_precision(target, permuted_baseline, categories)
        if abs(permuted_delta) + tolerance >= abs(observed_delta):
            extreme += 1
    p_value = (extreme + 1) / (permutations + 1)

    component_categories = np.empty(component_count, dtype=object)
    for component in range(component_count):
        component_categories[component] = categories[components == component][0]
    components_by_category = {
        category: np.flatnonzero(component_categories == category)
        for category in np.unique(categories)
    }
    bootstrap_deltas: list[float] = []
    max_attempts = bootstrap_resamples * 20
    attempts = 0
    while len(bootstrap_deltas) < bootstrap_resamples and attempts < max_attempts:
        attempts += 1
        component_weights = np.zeros(component_count, dtype=np.float64)
        for category_components in components_by_category.values():
            sampled = random.choice(
                category_components,
                size=len(category_components),
                replace=True,
            )
            component_weights += np.bincount(
                sampled,
                minlength=component_count,
            )
        row_weights = component_weights[components]
        candidate_sample = macro_average_precision(
            target,
            candidate_scores,
            categories,
            sample_weight=row_weights,
        )
        baseline_sample = macro_average_precision(
            target,
            baseline_scores,
            categories,
            sample_weight=row_weights,
        )
        delta = candidate_sample - baseline_sample
        if np.isfinite(delta):
            bootstrap_deltas.append(float(delta))
    if len(bootstrap_deltas) < bootstrap_resamples:
        raise SignificanceError(
            "Could not draw enough valid component-bootstrap samples; "
            "a category may have too few positive components"
        )
    ci_low, ci_high = np.quantile(bootstrap_deltas, [0.025, 0.975])
    return {
        "examples": int(len(aligned)),
        "categories": int(len(np.unique(categories))),
        "components": component_count,
        "baseline_macro_average_precision": baseline_ap,
        "candidate_macro_average_precision": candidate_ap,
        "delta_macro_average_precision": observed_delta,
        "p_value": float(p_value),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "permutations": permutations,
        "bootstrap_resamples": bootstrap_resamples,
        "seed": seed,
    }


def compare_experiment_directories(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    baseline_run_id: str,
    candidate_run_id: str,
    permutations: int = 2_000,
    bootstrap_resamples: int = 2_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare all three frozen validation splits and apply Holm correction."""
    if not baseline_run_id.strip() or not candidate_run_id.strip():
        raise SignificanceError("baseline_run_id and candidate_run_id are required")
    split_results: dict[str, dict[str, Any]] = {}
    for split_index, split in enumerate(SPLITS):
        filename = PREDICTION_FILENAMES[split]
        baseline_path = baseline_dir / filename
        candidate_path = candidate_dir / filename
        if not baseline_path.is_file() or not candidate_path.is_file():
            raise SignificanceError(
                f"Missing paired predictions for {split}: "
                f"baseline={baseline_path.is_file()}, "
                f"candidate={candidate_path.is_file()}"
            )
        split_results[split] = compare_prediction_frames(
            read_prediction_artifact(baseline_path),
            read_prediction_artifact(candidate_path),
            permutations=permutations,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed + split_index,
        )
    adjusted = holm_adjust(
        {split: float(result["p_value"]) for split, result in split_results.items()}
    )
    for split, p_value_holm in adjusted.items():
        split_results[split]["p_value_holm"] = p_value_holm
    return {
        "schema_version": 1,
        "status": "ready",
        "baseline_run_id": baseline_run_id.strip(),
        "candidate_run_id": candidate_run_id.strip(),
        "method": "paired_component_permutation",
        "confidence_interval_method": "paired_component_bootstrap_percentile",
        "multiple_testing_correction": "holm_3_splits",
        "splits": split_results,
    }
