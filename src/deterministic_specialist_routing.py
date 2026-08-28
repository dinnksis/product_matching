"""Frozen, label-free routing policies for saved neural specialist scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


ROUTE_BUDGETS = (0.05, 0.10, 0.15, 0.20, 0.30)
SCORE_MODES = ("replace", "blend_50_50", "blend_specialist_25")
NUMERIC_CONFLICT_COLUMNS = (
    "num_size_conflict",
    "num_ram_storage_conflict",
    "num_volume_conflict",
    "num_weight_conflict",
    "num_dimensions_conflict",
    "num_pack_count_conflict",
    "num_power_conflict",
    "num_optical_conflict",
)


@dataclass(frozen=True)
class RoutingPolicy:
    name: str
    priority: Literal["uncertainty_abs", "uncertainty_entropy", "domain_conflict"]
    expert: Literal["minilm", "rumodernbert", "dynamic_conflict"]
    description: str


POLICIES = (
    RoutingPolicy(
        "uncertainty_abs_minilm",
        "uncertainty_abs",
        "minilm",
        "Smallest abs(BGE probability - 0.5) routed to MiniLM.",
    ),
    RoutingPolicy(
        "uncertainty_entropy_minilm",
        "uncertainty_entropy",
        "minilm",
        "Largest binary entropy routed to MiniLM; mathematically rank-equivalent to abs uncertainty.",
    ),
    RoutingPolicy(
        "uncertainty_abs_rumodernbert",
        "uncertainty_abs",
        "rumodernbert",
        "Smallest abs(BGE probability - 0.5) routed to RuModernBERT.",
    ),
    RoutingPolicy(
        "domain_conflict_minilm",
        "domain_conflict",
        "minilm",
        "Frozen domain-prior score routed to MiniLM.",
    ),
    RoutingPolicy(
        "domain_conflict_rumodernbert",
        "domain_conflict",
        "rumodernbert",
        "Frozen domain-prior score routed to RuModernBERT.",
    ),
    RoutingPolicy(
        "domain_conflict_dynamic",
        "domain_conflict",
        "dynamic_conflict",
        "Complex structured conflicts use RuModernBERT; other routed pairs use MiniLM.",
    ),
)


def _probabilities(values: pd.Series | np.ndarray) -> np.ndarray:
    probability = np.asarray(values, dtype=np.float64)
    if probability.ndim != 1 or not np.isfinite(probability).all():
        raise ValueError("Probabilities must be a finite one-dimensional array")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("Probabilities must lie in [0, 1]")
    return probability


def uncertainty_abs(probability: pd.Series | np.ndarray) -> np.ndarray:
    """Return high-is-uncertain score equivalent to negative abs(p - 0.5)."""
    probability = _probabilities(probability)
    # Rounding makes mathematically symmetric p and 1-p ties deterministic.
    return np.round(1.0 - 2.0 * np.abs(probability - 0.5), decimals=14)


def uncertainty_entropy(probability: pd.Series | np.ndarray) -> np.ndarray:
    """Return binary entropy; its ordering is identical to abs uncertainty."""
    symmetric_probability = 0.5 * uncertainty_abs(probability)
    clipped = np.clip(symmetric_probability, 1e-12, 1.0 - 1e-12)
    return -(clipped * np.log(clipped) + (1.0 - clipped) * np.log1p(-clipped))


def structured_conflict_components(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in NUMERIC_CONFLICT_COLUMNS if column not in frame]
    required = {
        "model_code_conflict",
        "title_code_conflict",
        "brand_conflict",
        "title_token_set",
        "attribute_key_jaccard",
    }
    missing.extend(sorted(required - set(frame.columns)))
    if missing:
        raise ValueError(f"Missing routing features: {sorted(set(missing))}")
    numeric_count = (
        frame.loc[:, NUMERIC_CONFLICT_COLUMNS].to_numpy(dtype=np.float64) > 0.5
    ).sum(axis=1)
    identifier_conflict = (
        (frame["model_code_conflict"].to_numpy(dtype=np.float64) > 0.5)
        | (frame["title_code_conflict"].to_numpy(dtype=np.float64) > 0.5)
    )
    brand_conflict = frame["brand_conflict"].to_numpy(dtype=np.float64) > 0.5
    return pd.DataFrame(
        {
            "identifier_conflict": identifier_conflict,
            "brand_conflict": brand_conflict,
            "numeric_conflict_count": numeric_count,
            "any_numeric_conflict": numeric_count > 0,
            "multiple_numeric_conflicts": numeric_count >= 2,
            "title_dissimilarity": 1.0
            - np.clip(frame["title_token_set"].to_numpy(dtype=np.float64), 0.0, 1.0),
            "low_attribute_overlap": (
                frame["attribute_key_jaccard"].to_numpy(dtype=np.float64) < 0.10
            ),
        },
        index=frame.index,
    )


def domain_conflict_priority(frame: pd.DataFrame) -> np.ndarray:
    """Predeclared domain score. Coefficients are not fitted on validation labels."""
    components = structured_conflict_components(frame)
    base = uncertainty_abs(frame["bge_probability"])
    return (
        base
        + 0.25 * components["identifier_conflict"].to_numpy(dtype=np.float64)
        + 0.15 * components["any_numeric_conflict"].to_numpy(dtype=np.float64)
        + 0.10 * components["title_dissimilarity"].to_numpy(dtype=np.float64)
        + 0.05 * components["low_attribute_overlap"].to_numpy(dtype=np.float64)
    )


def routing_priority(frame: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "uncertainty_abs":
        return uncertainty_abs(frame["bge_probability"])
    if kind == "uncertainty_entropy":
        return uncertainty_entropy(frame["bge_probability"])
    if kind == "domain_conflict":
        return domain_conflict_priority(frame)
    raise ValueError(f"Unknown routing priority: {kind}")


def top_budget_mask(
    priority: np.ndarray,
    budget: float,
    id1: pd.Series | np.ndarray,
    id2: pd.Series | np.ndarray,
) -> np.ndarray:
    priority = np.asarray(priority, dtype=np.float64)
    left = np.asarray(id1, dtype=np.int64)
    right = np.asarray(id2, dtype=np.int64)
    if not 0.0 <= budget <= 1.0:
        raise ValueError("Route budget must lie in [0, 1]")
    if len(priority) != len(left) or len(priority) != len(right):
        raise ValueError("Priority and pair keys have different lengths")
    route_count = int(np.floor(len(priority) * budget + 1e-12))
    mask = np.zeros(len(priority), dtype=bool)
    if route_count:
        order = np.lexsort((right, left, -priority))
        mask[order[:route_count]] = True
    return mask


def expert_assignment(
    frame: pd.DataFrame,
    routed: np.ndarray,
    expert: str,
) -> np.ndarray:
    routed = np.asarray(routed, dtype=bool)
    if len(routed) != len(frame):
        raise ValueError("Route mask length differs from frame")
    assignment = np.full(len(frame), "bge", dtype=object)
    if expert in {"minilm", "rumodernbert"}:
        assignment[routed] = expert
        return assignment
    if expert != "dynamic_conflict":
        raise ValueError(f"Unknown expert assignment: {expert}")
    components = structured_conflict_components(frame)
    complex_conflict = components["multiple_numeric_conflicts"].to_numpy() | (
        components["identifier_conflict"].to_numpy()
        & components["any_numeric_conflict"].to_numpy()
    )
    assignment[routed & ~complex_conflict] = "minilm"
    assignment[routed & complex_conflict] = "rumodernbert"
    return assignment


def routed_scores(
    frame: pd.DataFrame,
    assignment: np.ndarray,
    score_mode: str,
) -> np.ndarray:
    bge = frame["bge_probability"].to_numpy(dtype=np.float64)
    result = bge.copy()
    if score_mode == "replace":
        specialist_weight = 1.0
    elif score_mode == "blend_50_50":
        specialist_weight = 0.5
    elif score_mode == "blend_specialist_25":
        specialist_weight = 0.25
    else:
        raise ValueError(f"Unknown score mode: {score_mode}")
    for specialist in ("minilm", "rumodernbert"):
        mask = np.asarray(assignment) == specialist
        specialist_score = frame[f"{specialist}_probability"].to_numpy(dtype=np.float64)
        result[mask] = (
            (1.0 - specialist_weight) * bge[mask]
            + specialist_weight * specialist_score[mask]
        )
    return result


def estimate_private_t4_runtime(
    pairs: int,
    minilm_coverage: float,
    rumodernbert_coverage: float,
    measurements: pd.DataFrame,
    *,
    common_io_seconds: float = 9.5,
) -> dict[str, float]:
    indexed = measurements.set_index("model")
    required = {"bge", "minilm", "rumodernbert"}
    if not required.issubset(indexed.index):
        raise ValueError("Runtime measurements do not contain all three models")
    coverages = {
        "bge": 1.0,
        "minilm": float(minilm_coverage),
        "rumodernbert": float(rumodernbert_coverage),
    }
    total = common_io_seconds
    bge_variable = pairs / float(indexed.loc["bge", "pairs_per_second"])
    specialist_variable = 0.0
    for model, coverage in coverages.items():
        if coverage <= 0:
            continue
        model_seconds = float(indexed.loc[model, "load_seconds"]) + (
            pairs * coverage / float(indexed.loc[model, "pairs_per_second"])
        )
        total += model_seconds
        if model != "bge":
            specialist_variable += model_seconds
    bge_only = (
        common_io_seconds
        + float(indexed.loc["bge", "load_seconds"])
        + bge_variable
    )
    return {
        "estimated_private_t4_seconds": total,
        "estimated_private_t4_minutes": total / 60.0,
        "runtime_multiplier_vs_bge": total / bge_only,
        "specialist_overhead_seconds": specialist_variable,
    }
