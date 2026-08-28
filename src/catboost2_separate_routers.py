"""Joint risk/coverage selection for separate negative and positive trust routers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.catboost1_early_exit import threshold_states, wilson_upper


def best_two_low_tail_state(
    negative_states: pd.DataFrame,
    positive_states: pd.DataFrame,
    risk_limit: float,
    total_examples: int,
) -> dict[str, Any]:
    """Maximize two low-score tails under one combined Wilson bound."""

    max_errors = max(0, int(math.ceil(risk_limit * total_examples)))
    neg = negative_states.loc[negative_states["errors"] <= max_errors].copy()
    pos = positive_states.loc[positive_states["errors"] <= max_errors].copy()
    neg = neg.sort_values("accepted").groupby("errors", as_index=False).tail(1).set_index("errors")
    pos = pos.sort_values("accepted").groupby("errors", as_index=False).tail(1).set_index("errors")
    pos_errors = pos.index.to_numpy(dtype=np.int64)
    pos_accepted = pos["accepted"].to_numpy(dtype=np.int64)
    pos_thresholds = pos["threshold"].to_numpy(dtype=np.float64)
    best: dict[str, Any] | None = None
    for neg_errors, neg_row in neg.iterrows():
        eligible = pos_errors <= max_errors - int(neg_errors)
        if not eligible.any():
            continue
        local = np.flatnonzero(eligible)
        accepted = int(neg_row.accepted) + pos_accepted[eligible]
        errors = int(neg_errors) + pos_errors[eligible]
        ucb = wilson_upper(errors, accepted)
        valid = (accepted > 0) & (accepted <= total_examples) & (ucb < risk_limit)
        if not valid.any():
            continue
        candidates = np.flatnonzero(valid)
        order = np.lexsort((errors[candidates], -accepted[candidates]))
        chosen = int(candidates[int(order[0])])
        pos_index = int(local[chosen])
        candidate = {
            "accepted": int(accepted[chosen]),
            "errors": int(errors[chosen]),
            "coverage": float(accepted[chosen] / total_examples),
            "empirical_error": float(errors[chosen] / accepted[chosen]),
            "error_ucb_95": float(ucb[chosen]),
            "threshold_neg": float(neg_row.threshold),
            "threshold_pos": float(pos_thresholds[pos_index]),
            "negative_accepted": int(neg_row.accepted),
            "negative_errors": int(neg_errors),
            "positive_accepted": int(pos_accepted[pos_index]),
            "positive_errors": int(pos_errors[pos_index]),
        }
        if best is None or (candidate["accepted"], -candidate["errors"]) > (
            best["accepted"], -best["errors"]
        ):
            best = candidate
    return best or {
        "accepted": 0,
        "errors": 0,
        "coverage": 0.0,
        "empirical_error": 0.0,
        "error_ucb_95": 1.0,
        "threshold_neg": 0.0,
        "threshold_pos": 0.0,
        "negative_accepted": 0,
        "negative_errors": 0,
        "positive_accepted": 0,
        "positive_errors": 0,
    }


def select_separate_thresholds(
    q_error: np.ndarray,
    target_error: np.ndarray,
    predicted_class: np.ndarray,
    risk_limit: float,
) -> dict[str, Any]:
    predicted_class = np.asarray(predicted_class, dtype=np.int8)
    negative = predicted_class == 0
    positive = ~negative
    negative_states = threshold_states(q_error[negative], target_error[negative], "negative")
    positive_states = threshold_states(q_error[positive], target_error[positive], "negative")
    return best_two_low_tail_state(
        negative_states, positive_states, risk_limit, len(target_error)
    )


def apply_separate_thresholds(
    q_error: np.ndarray,
    predicted_class: np.ndarray,
    threshold_neg: float,
    threshold_pos: float,
) -> np.ndarray:
    predicted_class = np.asarray(predicted_class, dtype=np.int8)
    return np.where(
        predicted_class == 0,
        np.asarray(q_error) < threshold_neg,
        np.asarray(q_error) < threshold_pos,
    )


def routing_summary(
    accepted: np.ndarray,
    target_error: np.ndarray,
    predicted_class: np.ndarray,
    risk_limit: float,
) -> dict[str, Any]:
    accepted = np.asarray(accepted, dtype=bool)
    target_error = np.asarray(target_error, dtype=np.int8)
    predicted_class = np.asarray(predicted_class, dtype=np.int8)
    negative_zone = predicted_class == 0
    positive_zone = ~negative_zone
    negative = accepted & negative_zone
    positive = accepted & positive_zone
    count = int(accepted.sum())
    errors = int(target_error[accepted].sum())
    negative_count = int(negative.sum())
    positive_count = int(positive.sum())
    negative_errors = int(target_error[negative].sum())
    positive_errors = int(target_error[positive].sum())
    ucb = float(wilson_upper(errors, count)) if count else 1.0
    return {
        "accepted": count,
        "coverage": count / len(accepted),
        "errors": errors,
        "empirical_error": errors / count if count else 0.0,
        "error_ucb_95": ucb,
        "passes_risk": bool(count and ucb < risk_limit),
        "verified_coverage": count / len(accepted) if count and ucb < risk_limit else 0.0,
        "negative_accepted": negative_count,
        "negative_coverage": negative_count / len(accepted),
        "negative_within_zone_coverage": negative_count / max(1, int(negative_zone.sum())),
        "negative_errors": negative_errors,
        "negative_empirical_error": negative_errors / negative_count if negative_count else 0.0,
        "negative_error_ucb_95": float(wilson_upper(negative_errors, negative_count)) if negative_count else 1.0,
        "positive_accepted": positive_count,
        "positive_coverage": positive_count / len(accepted),
        "positive_within_zone_coverage": positive_count / max(1, int(positive_zone.sum())),
        "positive_errors": positive_errors,
        "positive_empirical_error": positive_errors / positive_count if positive_count else 0.0,
        "positive_error_ucb_95": float(wilson_upper(positive_errors, positive_count)) if positive_count else 1.0,
    }


def crossfit_separate_routing(
    q_error: np.ndarray,
    target_error: np.ndarray,
    predicted_class: np.ndarray,
    folds: np.ndarray,
    risk_limit: float,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    accepted = np.zeros(len(q_error), dtype=bool)
    details: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds)):
        calibration = folds != fold
        held = ~calibration
        state = select_separate_thresholds(
            q_error[calibration], target_error[calibration], predicted_class[calibration], risk_limit
        )
        selected = apply_separate_thresholds(
            q_error[held], predicted_class[held],
            float(state["threshold_neg"]), float(state["threshold_pos"]),
        )
        accepted[np.flatnonzero(held)] = selected
        held_summary = routing_summary(
            selected, target_error[held], predicted_class[held], risk_limit
        )
        details.append({
            "held_fold": int(fold),
            "threshold_neg": float(state["threshold_neg"]),
            "threshold_pos": float(state["threshold_pos"]),
            "calibration_accepted": int(state["accepted"]),
            "calibration_errors": int(state["errors"]),
            "calibration_ucb_95": float(state["error_ucb_95"]),
            **{f"held_{key}": value for key, value in held_summary.items()},
        })
    return routing_summary(
        accepted, target_error, predicted_class, risk_limit
    ), pd.DataFrame(details), accepted

