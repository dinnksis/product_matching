"""Compact features for routing RuModern after one-way BGE and MiniLM."""

from __future__ import annotations

import numpy as np
import pandas as pd


SEQUENTIAL_SCORE_COLUMNS = (
    "minilm_probability",
    "minilm_logit",
    "bge_minilm_disagreement",
    "bge_minilm_signed_difference",
    "bge_minilm_mean",
    "bge_minilm_min",
    "bge_minilm_max",
)


def sequential_feature_frame(
    base_features: pd.DataFrame,
    bge_probability: np.ndarray,
    minilm_probability: np.ndarray,
) -> pd.DataFrame:
    """Append only cheap MiniLM/BGE score interactions to compact BGE features."""

    result = base_features.copy().reset_index(drop=True)
    bge = np.asarray(bge_probability, dtype=np.float64)
    mini = np.asarray(minilm_probability, dtype=np.float64)
    if len(result) != len(bge) or len(bge) != len(mini):
        raise ValueError("Sequential router inputs are misaligned")
    if not np.isfinite(bge).all() or not np.isfinite(mini).all():
        raise ValueError("Sequential router scores must be finite")
    clipped = np.clip(mini, 1e-7, 1.0 - 1e-7)
    result["minilm_probability"] = mini.astype(np.float32)
    result["minilm_logit"] = np.log(clipped / (1.0 - clipped)).astype(np.float32)
    result["bge_minilm_disagreement"] = np.abs(bge - mini).astype(np.float32)
    result["bge_minilm_signed_difference"] = (mini - bge).astype(np.float32)
    result["bge_minilm_mean"] = ((bge + mini) * 0.5).astype(np.float32)
    result["bge_minilm_min"] = np.minimum(bge, mini).astype(np.float32)
    result["bge_minilm_max"] = np.maximum(bge, mini).astype(np.float32)
    return result
