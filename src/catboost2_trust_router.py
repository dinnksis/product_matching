"""Utilities for a CatBoost-2 model that predicts CatBoost-1 decision errors."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.catboost1_early_exit import best_side_state, threshold_states, wilson_upper


CONFIDENCE_COLUMNS = (
    "p_cb1",
    "logit_p_cb1",
    "abs_p_cb1_minus_half",
    "min_p_cb1_one_minus_p",
    "entropy_p_cb1",
    "cb1_predicted_class",
)


def cb1_feature_frame(base: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Exact frozen C1A feature view: numeric cheap features plus category."""

    numeric = base.select_dtypes(exclude="object").copy().reset_index(drop=True)
    numeric = numeric.loc[:, ~numeric.columns.str.startswith("rule_")]
    numeric["category"] = base["category"].astype(str).to_numpy()
    return numeric, ["category"]


def confidence_features(probability: np.ndarray, epsilon: float = 1e-7) -> pd.DataFrame:
    probability = np.asarray(probability, dtype=np.float64)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("CatBoost-1 probabilities must be finite and inside [0, 1]")
    clipped = np.clip(probability, epsilon, 1.0 - epsilon)
    return pd.DataFrame({
        "p_cb1": probability.astype(np.float32),
        "logit_p_cb1": np.log(clipped / (1.0 - clipped)).astype(np.float32),
        "abs_p_cb1_minus_half": np.abs(probability - 0.5).astype(np.float32),
        "min_p_cb1_one_minus_p": np.minimum(probability, 1.0 - probability).astype(np.float32),
        "entropy_p_cb1": (
            -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
        ).astype(np.float32),
        "cb1_predicted_class": (probability >= 0.5).astype(np.float32),
    })


def trust_feature_frame(
    base: pd.DataFrame,
    probability: np.ndarray,
) -> tuple[pd.DataFrame, list[str]]:
    frame, categorical = cb1_feature_frame(base)
    confidence = confidence_features(probability)
    for column in CONFIDENCE_COLUMNS:
        frame[column] = confidence[column].to_numpy()
    return frame, categorical


def decision_error(probability: np.ndarray, target: np.ndarray) -> np.ndarray:
    predicted = np.asarray(probability) >= 0.5
    labels = np.asarray(target, dtype=np.int8) == 1
    return (predicted != labels).astype(np.int8)


def detection_metrics(target_error: np.ndarray, q_error: np.ndarray) -> dict[str, float]:
    return {
        "error_prevalence": float(np.mean(target_error)),
        "roc_auc_error_detection": float(roc_auc_score(target_error, q_error)),
        "pr_auc_error_detection": float(average_precision_score(target_error, q_error)),
    }


def calibration_table(
    target_error: np.ndarray,
    q_error: np.ndarray,
    bins: int = 10,
) -> pd.DataFrame:
    frame = pd.DataFrame({"target_error": target_error, "q_error": q_error})
    ranked = frame["q_error"].rank(method="first")
    frame["bin"] = pd.qcut(ranked, q=min(bins, len(frame)), labels=False, duplicates="drop")
    return frame.groupby("bin", as_index=False).agg(
        examples=("target_error", "size"),
        q_min=("q_error", "min"),
        q_max=("q_error", "max"),
        mean_q_error=("q_error", "mean"),
        actual_error=("target_error", "mean"),
        errors=("target_error", "sum"),
    )


def full_oof_operating_point(
    error_score: np.ndarray,
    target_error: np.ndarray,
    risk_limit: float,
) -> dict[str, Any]:
    return best_side_state(threshold_states(error_score, target_error, "negative"), risk_limit)


def crossfit_threshold_routing(
    error_score: np.ndarray,
    target_error: np.ndarray,
    folds: np.ndarray,
    risk_limit: float,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    accepted = np.zeros(len(error_score), dtype=bool)
    details: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds)):
        calibration = folds != fold
        held = ~calibration
        state = full_oof_operating_point(
            error_score[calibration], target_error[calibration], risk_limit
        )
        threshold = float(state["threshold"])
        selected = held & (error_score < threshold)
        accepted |= selected
        details.append({
            "held_fold": int(fold),
            "threshold": threshold,
            "calibration_accepted": int(state["accepted"]),
            "calibration_errors": int(state["errors"]),
            "calibration_ucb_95": float(state["error_ucb_95"]),
            "held_accepted": int(selected.sum()),
            "held_errors": int(target_error[selected].sum()),
        })
    count = int(accepted.sum())
    errors = int(target_error[accepted].sum())
    ucb = float(wilson_upper(errors, count)) if count else 1.0
    return ({
        "accepted": count,
        "coverage": count / len(error_score),
        "errors": errors,
        "empirical_error": errors / count if count else 0.0,
        "error_ucb_95": ucb,
        "passes_risk": bool(count and ucb < risk_limit),
        "verified_coverage": count / len(error_score) if count and ucb < risk_limit else 0.0,
    }, pd.DataFrame(details), accepted)


def split_routing_metrics(
    split: str,
    method: str,
    risk_limit: float,
    threshold: float,
    error_score: np.ndarray,
    p_cb1: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    accepted = np.asarray(error_score) < threshold
    predicted = np.asarray(p_cb1) >= 0.5
    errors_mask = decision_error(p_cb1, target).astype(bool)
    count = int(accepted.sum())
    errors = int((accepted & errors_mask).sum())
    negative = accepted & ~predicted
    positive = accepted & predicted
    return {
        "split": split,
        "method": method,
        "risk_limit": risk_limit,
        "threshold": threshold,
        "pairs": len(target),
        "accepted": count,
        "coverage": count / len(target),
        "errors": errors,
        "empirical_error": errors / count if count else 0.0,
        "error_ucb_95": float(wilson_upper(errors, count)) if count else 1.0,
        "negative_accepted": int(negative.sum()),
        "negative_coverage": float(negative.mean()),
        "negative_errors": int((negative & errors_mask).sum()),
        "positive_accepted": int(positive.sum()),
        "positive_coverage": float(positive.mean()),
        "positive_errors": int((positive & errors_mask).sum()),
    }


def assert_component_disjoint(folds: np.ndarray, components: Sequence[int]) -> None:
    frame = pd.DataFrame({"fold": folds, "component": components})
    counts = frame.groupby("component")["fold"].nunique()
    if int(counts.max()) != 1:
        raise AssertionError("A product component occurs in multiple outer folds")

