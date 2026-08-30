"""Minimal production features for the one-direction MiniLM benefit router."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Sequence

import numpy as np
import pandas as pd


VARIANT_COLUMNS = {
    "score_category": (
        "category",
        "bge_probability",
        "bge_logit",
        "bge_abs_from_half",
        "bge_uncertainty",
        "bge_entropy",
        "bge_raw_logit",
    ),
    "score_title": (
        "category",
        "title_exact",
        "title_ratio",
        "title_token_sort",
        "title_token_jaccard",
        "title_length_ratio",
        "title_length_delta",
        "title_number_overlap",
        "title_number_jaccard",
        "bge_probability",
        "bge_logit",
        "bge_abs_from_half",
        "bge_uncertainty",
        "bge_entropy",
        "bge_raw_logit",
    ),
}
SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)
NUMBER_RE = re.compile(r"(?<![a-zа-яё])\d+(?:[.,]\d+)?", re.IGNORECASE)


def normalize_title(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().replace("ё", "е")
    return SPACE_RE.sub(" ", text).strip()


def _jaccard(first: set[str], second: set[str]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def title_feature_frame(left: Sequence[Any], right: Sequence[Any]) -> pd.DataFrame:
    """Compute only the title columns used by the compact production router."""

    from rapidfuzz import fuzz

    if len(left) != len(right):
        raise ValueError("Left/right title arrays have different lengths")
    rows = []
    for raw_left, raw_right in zip(left, right):
        first = normalize_title(raw_left)
        second = normalize_title(raw_right)
        first_tokens = set(TOKEN_RE.findall(first))
        second_tokens = set(TOKEN_RE.findall(second))
        first_numbers = set(NUMBER_RE.findall(first))
        second_numbers = set(NUMBER_RE.findall(second))
        rows.append(
            {
                "title_exact": float(first == second),
                "title_ratio": fuzz.ratio(first, second) / 100.0,
                "title_token_sort": fuzz.token_sort_ratio(first, second) / 100.0,
                "title_token_jaccard": _jaccard(first_tokens, second_tokens),
                "title_length_ratio": min(len(first), len(second))
                / max(1, len(first), len(second)),
                "title_length_delta": abs(len(first) - len(second)),
                "title_number_overlap": len(first_numbers & second_numbers),
                "title_number_jaccard": _jaccard(first_numbers, second_numbers),
            }
        )
    result = pd.DataFrame(rows)
    result[result.columns] = result.astype(np.float32)
    return result


def bge_feature_frame(probability: Sequence[float], raw_logit: Sequence[float]) -> pd.DataFrame:
    probability = np.asarray(probability, dtype=np.float64)
    raw_logit = np.asarray(raw_logit, dtype=np.float64)
    if probability.ndim != 1 or len(probability) != len(raw_logit):
        raise ValueError("BGE probability/logit arrays are misaligned")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("BGE probabilities must be finite and in [0, 1]")
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    return pd.DataFrame(
        {
            "bge_probability": probability.astype(np.float32),
            "bge_logit": np.log(clipped / (1.0 - clipped)).astype(np.float32),
            "bge_abs_from_half": np.abs(probability - 0.5).astype(np.float32),
            "bge_uncertainty": (1.0 - 2.0 * np.abs(probability - 0.5)).astype(np.float32),
            "bge_entropy": (
                -(clipped * np.log(clipped) + (1.0 - clipped) * np.log1p(-clipped))
            ).astype(np.float32),
            "bge_raw_logit": raw_logit.astype(np.float32),
        }
    )


def cached_feature_frame(
    cheap: pd.DataFrame,
    probability: Sequence[float],
    raw_logit: Sequence[float],
    variant: str,
) -> pd.DataFrame:
    if variant not in VARIANT_COLUMNS:
        raise ValueError(f"Unknown fast router variant: {variant}")
    result = cheap.copy().reset_index(drop=True)
    for column, values in bge_feature_frame(probability, raw_logit).items():
        result[column] = values.to_numpy()
    result["category"] = result["category"].fillna("__missing__").astype(str)
    columns = list(VARIANT_COLUMNS[variant])
    missing = [column for column in columns if column not in result]
    if missing:
        raise ValueError(f"Missing compact router features: {missing}")
    return result.loc[:, columns]


def runtime_feature_frame(
    category: Sequence[Any],
    left_title: Sequence[Any],
    right_title: Sequence[Any],
    probability: Sequence[float],
    raw_logit: Sequence[float],
    variant: str,
) -> pd.DataFrame:
    if variant not in VARIANT_COLUMNS:
        raise ValueError(f"Unknown fast router variant: {variant}")
    result = pd.DataFrame(
        {"category": pd.Series(category).fillna("__missing__").astype(str)}
    )
    if variant == "score_title":
        result = pd.concat(
            [result.reset_index(drop=True), title_feature_frame(left_title, right_title)],
            axis=1,
        )
    result = pd.concat(
        [result.reset_index(drop=True), bge_feature_frame(probability, raw_logit)], axis=1
    )
    return result.loc[:, list(VARIANT_COLUMNS[variant])]
