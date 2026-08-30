"""Leakage-safe targets, features, and routing helpers for specialist routers."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


BGE_FEATURE_COLUMNS = (
    "bge_probability",
    "bge_logit",
    "bge_abs_from_half",
    "bge_uncertainty",
    "bge_entropy",
    "bge_score_order_gap",
    "bge_token_length_ab",
    "bge_token_length_ba",
    "bge_token_length_max",
)


def binary_logloss(target: Sequence[int], probability: Sequence[float]) -> np.ndarray:
    """Per-example binary log loss."""

    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(probability, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError("Target and probability must be aligned one-dimensional arrays")
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("Target must be binary")
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("Probability must be finite and lie in [0, 1]")
    clipped = np.clip(p, 1e-7, 1.0 - 1e-7)
    return -(y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped))


def benefit_targets(
    target: Sequence[int],
    bge_probability: Sequence[float],
    specialist_probability: Sequence[float],
    *,
    classification_margin: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Return regression benefit and a materially-helpful classification target."""

    if classification_margin < 0:
        raise ValueError("classification_margin must be non-negative")
    benefit = binary_logloss(target, bge_probability) - binary_logloss(
        target, specialist_probability
    )
    return benefit.astype(np.float32), (benefit > classification_margin).astype(np.int8)


def bge_features(predictions: pd.DataFrame) -> pd.DataFrame:
    """Features available after BGE inference, without specialist information."""

    required = {"score"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing BGE prediction columns: {sorted(missing)}")
    p = predictions["score"].to_numpy(dtype=np.float64)
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("BGE score must be finite and lie in [0, 1]")
    clipped = np.clip(p, 1e-7, 1.0 - 1e-7)
    result = pd.DataFrame(
        {
            "bge_probability": p.astype(np.float32),
            "bge_logit": np.log(clipped / (1.0 - clipped)).astype(np.float32),
            "bge_abs_from_half": np.abs(p - 0.5).astype(np.float32),
            "bge_uncertainty": (1.0 - 2.0 * np.abs(p - 0.5)).astype(np.float32),
            "bge_entropy": (
                -(clipped * np.log(clipped) + (1.0 - clipped) * np.log1p(-clipped))
            ).astype(np.float32),
        }
    )
    optional = {
        "logit": "bge_raw_logit",
        "score_order_gap": "bge_score_order_gap",
        "token_length_ab": "bge_token_length_ab",
        "token_length_ba": "bge_token_length_ba",
    }
    for source, destination in optional.items():
        if source in predictions:
            result[destination] = predictions[source].to_numpy(dtype=np.float32)
    if {"token_length_ab", "token_length_ba"}.issubset(predictions.columns):
        result["bge_token_length_max"] = predictions[
            ["token_length_ab", "token_length_ba"]
        ].max(axis=1).to_numpy(dtype=np.float32)
    return result


def router_feature_frame(
    cheap_features: pd.DataFrame,
    bge_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the frozen router feature view and explicitly reject leakage columns."""

    if len(cheap_features) != len(bge_predictions):
        raise ValueError("Cheap features and BGE predictions have different row counts")
    forbidden_tokens = ("minilm", "rumodern", "specialist", "embedding", "target", "label")
    forbidden = [
        str(column)
        for column in cheap_features.columns
        if any(token in str(column).lower() for token in forbidden_tokens)
    ]
    if forbidden:
        raise ValueError(f"Forbidden pre-routing feature columns: {forbidden}")

    # Raw Qwen rule evidence is intentionally omitted. The compact semantic and
    # lexical comparisons are cheap, deterministic, and available in production.
    keep = [column for column in cheap_features.columns if not str(column).startswith("rule_")]
    result = cheap_features.loc[:, keep].copy().reset_index(drop=True)
    for column, values in bge_features(bge_predictions).items():
        result[column] = values.to_numpy()
    categorical = [column for column in result.columns if result[column].dtype == object]
    for column in categorical:
        result[column] = result[column].fillna("__missing__").astype(str)
    return result, categorical


def assert_component_disjoint(folds: Sequence[int], components: Sequence[int]) -> None:
    frame = pd.DataFrame({"fold": folds, "component": components})
    if frame["fold"].isna().any() or frame["component"].isna().any():
        raise AssertionError("Fold/component values cannot be missing")
    if int(frame.groupby("component")["fold"].nunique().max()) != 1:
        raise AssertionError("A product component occurs in multiple router folds")


def assert_pair_ids_disjoint(pairs: pd.DataFrame) -> None:
    """Verify directly that no product id occurs in more than one fold."""

    required = {"id1", "id2", "fold"}
    missing = required - set(pairs.columns)
    if missing:
        raise AssertionError(f"Missing pair-fold columns: {sorted(missing)}")
    occurrences = pd.concat(
        [
            pairs[["id1", "fold"]].rename(columns={"id1": "id"}),
            pairs[["id2", "fold"]].rename(columns={"id2": "id"}),
        ],
        ignore_index=True,
    )
    if int(occurrences.groupby("id")["fold"].nunique().max()) != 1:
        raise AssertionError("A product id occurs in multiple router folds")


def deterministic_random_priority(id1: Sequence[int], id2: Sequence[int], seed: int) -> np.ndarray:
    """Stable pseudo-random baseline independent of row order and labels."""

    left = np.asarray(id1, dtype=np.uint64)
    right = np.asarray(id2, dtype=np.uint64)
    value = left * np.uint64(0x9E3779B185EBCA87)
    value ^= right * np.uint64(0xC2B2AE3D27D4EB4F)
    value ^= np.uint64(seed)
    value ^= value >> np.uint64(30)
    value *= np.uint64(0xBF58476D1CE4E5B9)
    value ^= value >> np.uint64(27)
    return (value >> np.uint64(11)).astype(np.float64) / float(1 << 53)
