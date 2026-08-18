"""Shared binary metrics for the frozen IID/hard/OOD validation protocol."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import log_loss, precision_recall_curve, roc_auc_score


def _binary_arrays(
    target: np.ndarray,
    probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(target, dtype=np.int8).reshape(-1)
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    if len(target) == 0 or len(target) != len(probability):
        raise ValueError("target and probability must have the same non-zero length")
    if not set(np.unique(target)) <= {0, 1}:
        raise ValueError("target must contain only binary 0/1 values")
    if np.unique(target).size != 2:
        raise ValueError("binary validation metrics require both target classes")
    if not np.isfinite(probability).all():
        raise ValueError("probability must contain only finite values")
    if (probability < 0).any() or (probability > 1).any():
        raise ValueError("probability values must be in [0, 1]")
    return target, probability


def recall_at_precision(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    minimum_precision: float = 0.99,
) -> tuple[float, float | None]:
    """Return the maximum recall at a score threshold meeting precision floor."""
    if not 0 < minimum_precision <= 1:
        raise ValueError("minimum_precision must be in (0, 1]")
    target, probability = _binary_arrays(target, probability)
    precision, recall, thresholds = precision_recall_curve(target, probability)
    # precision/recall include a final no-positive operating point without a
    # corresponding threshold. Exclude it so an unavailable P99 returns R=0.
    eligible = np.flatnonzero(precision[:-1] >= minimum_precision)
    if eligible.size == 0:
        return 0.0, None
    eligible_recall = recall[eligible]
    best_recall = float(eligible_recall.max())
    # With equal recall prefer the lower threshold: it is the least restrictive
    # operating point and remains deterministic when scores contain ties.
    best_index = int(eligible[eligible_recall == best_recall][0])
    return best_recall, float(thresholds[best_index])


def binary_probability_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    minimum_precision: float = 0.99,
) -> dict[str, Any]:
    """Return protocol metrics derived from one split's probabilities."""
    target, probability = _binary_arrays(target, probability)
    recall, threshold = recall_at_precision(
        target,
        probability,
        minimum_precision=minimum_precision,
    )
    clipped = np.clip(probability, 1e-15, 1 - 1e-15)
    precision_label = f"{minimum_precision:.2f}".replace(".", "_")
    return {
        f"recall_at_precision_{precision_label}": recall,
        f"threshold_at_precision_{precision_label}": threshold,
        "roc_auc": float(roc_auc_score(target, probability)),
        "log_loss": float(log_loss(target, clipped, labels=[0, 1])),
    }
