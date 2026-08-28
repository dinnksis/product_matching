"""Saved-prediction analysis for selectively routing BGE pairs to specialists.

The module is intentionally training-free.  It provides label-aware diagnostic
oracles, pairwise correction proxies, and label-free slice summaries.  Oracle
selection uses labels and is therefore an upper-bound diagnostic, never a
production router.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss


SPECIALISTS = ("minilm", "rumodernbert")
NUMERIC_FAMILIES = (
    "size",
    "ram_storage",
    "volume",
    "weight",
    "dimensions",
    "pack_count",
    "power",
    "optical",
)
ROUTE_BUDGETS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
SCORE_MODES: Mapping[str, float | None] = {
    "replace": 1.0,
    "blend_specialist_25": 0.25,
    "blend_50_50": 0.50,
    "blend_specialist_75": 0.75,
    "mean_normalized_rank": None,
}


def normalized_rank(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return pd.Series(array).rank(method="average", ascending=True).to_numpy() / len(array)


def safe_average_precision(target: Sequence[float], score: Sequence[float]) -> float:
    target_array = np.asarray(target, dtype=np.float64)
    if len(target_array) == 0 or np.unique(target_array).size < 2:
        return math.nan
    return float(average_precision_score(target_array, np.asarray(score, dtype=np.float64)))


def safe_macro_average_precision(
    target: Sequence[float],
    score: Sequence[float],
    category: Sequence[str],
) -> tuple[float, int]:
    frame = pd.DataFrame(
        {
            "target": np.asarray(target, dtype=np.float64),
            "score": np.asarray(score, dtype=np.float64),
            "category": np.asarray(category, dtype=str),
        }
    )
    values: list[float] = []
    for _, group in frame.groupby("category", sort=True):
        value = safe_average_precision(group["target"], group["score"])
        if math.isfinite(value):
            values.append(value)
    return (float(np.mean(values)), len(values)) if values else (math.nan, 0)


def binary_logloss(target: np.ndarray, score: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(score, dtype=np.float64), 1e-7, 1 - 1e-7)
    target = np.asarray(target, dtype=np.float64)
    return -(target * np.log(clipped) + (1 - target) * np.log(1 - clipped))


def comparison_status(
    bge_loss: Sequence[float],
    specialist_loss: Sequence[float],
    *,
    tolerance: float,
) -> np.ndarray:
    difference = np.asarray(bge_loss, dtype=np.float64) - np.asarray(
        specialist_loss, dtype=np.float64
    )
    return np.select(
        [difference > tolerance, difference < -tolerance],
        ["specialist_better", "bge_better"],
        default="approximately_equal",
    )


def add_pairwise_proxies(
    frame: pd.DataFrame,
    specialist: str,
    *,
    threshold: float = 0.5,
    logloss_tolerance: float = 0.005,
    absolute_error_tolerance: float = 0.01,
) -> pd.DataFrame:
    result = frame.copy()
    target = result["target"].to_numpy(dtype=np.float64)
    bge = result["bge_probability"].to_numpy(dtype=np.float64)
    score = result[f"{specialist}_probability"].to_numpy(dtype=np.float64)

    bge_correct = (bge >= threshold) == (target >= 0.5)
    specialist_correct = (score >= threshold) == (target >= 0.5)
    result["binary_status"] = np.select(
        [specialist_correct & ~bge_correct, bge_correct & ~specialist_correct],
        ["specialist_better", "bge_better"],
        default="approximately_equal",
    )
    result["bge_binary_error"] = (~bge_correct).astype(np.int8)
    result["specialist_binary_error"] = (~specialist_correct).astype(np.int8)

    result["bge_logloss"] = binary_logloss(target, bge)
    result["specialist_logloss"] = binary_logloss(target, score)
    result["logloss_gain"] = result["bge_logloss"] - result["specialist_logloss"]
    result["logloss_status"] = comparison_status(
        result["bge_logloss"],
        result["specialist_logloss"],
        tolerance=logloss_tolerance,
    )

    result["bge_absolute_error"] = np.abs(bge - target)
    result["specialist_absolute_error"] = np.abs(score - target)
    result["absolute_error_gain"] = (
        result["bge_absolute_error"] - result["specialist_absolute_error"]
    )
    result["absolute_error_status"] = comparison_status(
        result["bge_absolute_error"],
        result["specialist_absolute_error"],
        tolerance=absolute_error_tolerance,
    )
    return result


def _state_from_flags(frame: pd.DataFrame, prefix: str) -> pd.Series:
    match = frame[f"{prefix}_match"].to_numpy(dtype=np.float64) > 0.5
    conflict = frame[f"{prefix}_conflict"].to_numpy(dtype=np.float64) > 0.5
    one_missing = frame[f"{prefix}_one_missing"].to_numpy(dtype=np.float64) > 0.5
    both = frame[f"{prefix}_both"].to_numpy(dtype=np.float64) > 0.5
    values = np.select(
        [match, conflict, one_missing, both],
        ["match", "conflict", "one_missing", "both_ambiguous"],
        default="both_missing",
    )
    return pd.Series(values, index=frame.index, dtype="string")


def _fixed_cut(
    values: pd.Series,
    edges: Sequence[float],
    labels: Sequence[str],
) -> pd.Series:
    return pd.cut(
        values.astype(float),
        bins=list(edges),
        labels=list(labels),
        include_lowest=True,
        right=False,
    ).astype("string").fillna(labels[-1])


def build_slice_columns(frame: pd.DataFrame) -> dict[str, pd.Series]:
    slices: dict[str, pd.Series] = {
        "category": frame["category"].astype("string"),
        "label": frame["target"].astype(int).astype("string"),
    }
    slices["bge_score_bin"] = _fixed_cut(
        frame["bge_probability"],
        (0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95, 0.975, 0.99, 1.000001),
        ("00-.01", ".01-.025", ".025-.05", ".05-.10", ".10-.20", ".20-.40", ".40-.60", ".60-.80", ".80-.90", ".90-.95", ".95-.975", ".975-.99", ".99-1"),
    )
    uncertainty = (frame["bge_probability"].astype(float) - 0.5).abs()
    slices["bge_uncertainty"] = _fixed_cut(
        uncertainty,
        (0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.500001),
        ("very_uncertain", "uncertain", "mid", "confident", "very_confident", "extreme"),
    )
    slices["title_similarity"] = _fixed_cut(
        frame["title_token_set"],
        (0, 0.50, 0.70, 0.85, 0.95, 0.999999, 1.000001),
        ("very_low", "low", "medium", "high", "near_exact", "exact_token_set"),
    )
    slices["title_exact_near"] = pd.Series(
        np.select(
            [
                frame["title_exact"].to_numpy() > 0.5,
                frame["title_token_set"].to_numpy() >= 0.95,
                frame["title_token_set"].to_numpy() >= 0.85,
            ],
            ["exact", "near_exact", "high_nonexact"],
            default="not_near",
        ),
        index=frame.index,
        dtype="string",
    )
    for prefix, name in (
        ("brand", "brand_state"),
        ("model_code", "model_attribute_state"),
        ("title_code", "title_code_state"),
    ):
        slices[name] = _state_from_flags(frame, prefix)

    model_match = (frame["model_code_match"] > 0.5) | (frame["title_code_match"] > 0.5)
    model_conflict = (frame["model_code_conflict"] > 0.5) | (
        frame["title_code_conflict"] > 0.5
    )
    model_missing = (frame["model_code_one_missing"] > 0.5) | (
        frame["title_code_one_missing"] > 0.5
    )
    slices["model_sku_code_state"] = pd.Series(
        np.select(
            [model_match, model_conflict, model_missing],
            ["match", "conflict", "one_missing"],
            default="no_evidence",
        ),
        index=frame.index,
        dtype="string",
    )

    numeric_conflicts = np.zeros(len(frame), dtype=np.int16)
    numeric_matches = np.zeros(len(frame), dtype=np.int16)
    numeric_missing = np.zeros(len(frame), dtype=np.int16)
    for family in NUMERIC_FAMILIES:
        prefix = f"num_{family}"
        slices[f"numeric_{family}_state"] = _state_from_flags(frame, prefix)
        numeric_conflicts += (frame[f"{prefix}_conflict"].to_numpy() > 0.5).astype(np.int16)
        numeric_matches += (frame[f"{prefix}_match"].to_numpy() > 0.5).astype(np.int16)
        numeric_missing += (frame[f"{prefix}_one_missing"].to_numpy() > 0.5).astype(np.int16)
    slices["numeric_conflict_count"] = pd.Series(
        np.where(numeric_conflicts == 0, "none", np.where(numeric_conflicts == 1, "one", "multiple")),
        index=frame.index,
        dtype="string",
    )
    slices["numeric_match_count"] = pd.Series(
        np.where(numeric_matches == 0, "none", np.where(numeric_matches == 1, "one", "multiple")),
        index=frame.index,
        dtype="string",
    )
    slices["numeric_primary_conflict"] = frame["primary_conflict_type"].astype("string")

    slices["attribute_key_overlap"] = _fixed_cut(
        frame["attribute_key_jaccard"],
        (0, 1e-12, 0.05, 0.10, 0.20, 0.40, 1.000001),
        ("none", "tiny", "low", "medium", "high", "very_high"),
    )
    slices["attribute_agreement"] = pd.Series(
        np.select(
            [
                frame["attribute_common_keys"].to_numpy() <= 0,
                frame["attribute_exact_ratio"].to_numpy() >= 0.80,
                frame["attribute_conflict_ratio"].to_numpy() >= 0.80,
            ],
            ["no_common_keys", "mostly_exact", "mostly_conflict"],
            default="mixed",
        ),
        index=frame.index,
        dtype="string",
    )
    attribute_count = frame[["attribute_count_a", "attribute_count_b"]].max(axis=1)
    slices["attribute_count"] = _fixed_cut(
        attribute_count,
        (0, 1, 5, 10, 20, 40, float("inf")),
        ("zero", "1-4", "5-9", "10-19", "20-39", "40+"),
    )
    missing_total = (
        frame["attribute_missing_a"].to_numpy(dtype=np.float64)
        + frame["attribute_missing_b"].to_numpy(dtype=np.float64)
        + numeric_missing
        + frame["brand_one_missing"].to_numpy(dtype=np.float64)
        + frame["model_code_one_missing"].to_numpy(dtype=np.float64)
    )
    slices["missingness"] = _fixed_cut(
        pd.Series(missing_total, index=frame.index),
        (0, 1, 3, 6, 11, 21, float("inf")),
        ("none", "low", "medium", "high", "very_high", "extreme"),
    )
    slices["bge_sequence_length"] = _fixed_cut(
        frame["bge_token_length_max"],
        (0, 128, 256, 384, 385, float("inf")),
        ("<128", "128-255", "256-383", "at_384", ">384"),
    )
    return slices


def pairwise_summary(frame: pd.DataFrame, specialist: str, split: str) -> dict[str, Any]:
    target = frame["target"].to_numpy(dtype=np.float64)
    category = frame["category"].astype(str).to_numpy()
    bge = frame["bge_probability"].to_numpy(dtype=np.float64)
    score = frame[f"{specialist}_probability"].to_numpy(dtype=np.float64)
    bge_macro, bge_categories = safe_macro_average_precision(target, bge, category)
    specialist_macro, specialist_categories = safe_macro_average_precision(target, score, category)
    row: dict[str, Any] = {
        "split": split,
        "specialist": specialist,
        "n": len(frame),
        "positive_rate": float(target.mean()),
        "bge_macro_ap": bge_macro,
        "specialist_macro_ap": specialist_macro,
        "macro_ap_delta": specialist_macro - bge_macro,
        "bge_overall_ap": safe_average_precision(target, bge),
        "specialist_overall_ap": safe_average_precision(target, score),
        "bge_logloss": float(binary_logloss(target, bge).mean()),
        "specialist_logloss": float(binary_logloss(target, score).mean()),
        "bge_mae": float(np.abs(bge - target).mean()),
        "specialist_mae": float(np.abs(score - target).mean()),
        "ap_category_count": min(bge_categories, specialist_categories),
        "binary_threshold": 0.5,
        "binary_threshold_source": "fixed_not_fitted",
    }
    for proxy in ("binary", "logloss", "absolute_error"):
        counts = frame[f"{proxy}_status"].value_counts()
        row[f"{proxy}_specialist_better"] = int(counts.get("specialist_better", 0))
        row[f"{proxy}_bge_better"] = int(counts.get("bge_better", 0))
        row[f"{proxy}_approximately_equal"] = int(counts.get("approximately_equal", 0))
        row[f"{proxy}_net_correction"] = (
            row[f"{proxy}_specialist_better"] - row[f"{proxy}_bge_better"]
        )
    return row


def slice_metric_rows(
    frame: pd.DataFrame,
    specialist: str,
    split: str,
    *,
    min_size: int = 20,
) -> list[dict[str, Any]]:
    slices = build_slice_columns(frame)
    rows: list[dict[str, Any]] = []
    for slice_type, assignments in slices.items():
        for slice_value in sorted(assignments.dropna().astype(str).unique()):
            mask = assignments.astype(str).to_numpy() == slice_value
            if int(mask.sum()) < min_size:
                continue
            part = frame.loc[mask]
            target = part["target"].to_numpy(dtype=np.float64)
            category = part["category"].astype(str).to_numpy()
            bge = part["bge_probability"].to_numpy(dtype=np.float64)
            score = part[f"{specialist}_probability"].to_numpy(dtype=np.float64)
            bge_macro, bge_category_count = safe_macro_average_precision(target, bge, category)
            specialist_macro, specialist_category_count = safe_macro_average_precision(
                target, score, category
            )
            binary_counts = part["binary_status"].value_counts()
            logloss_counts = part["logloss_status"].value_counts()
            absolute_counts = part["absolute_error_status"].value_counts()
            corrected = int(binary_counts.get("specialist_better", 0))
            regressed = int(binary_counts.get("bge_better", 0))
            rows.append(
                {
                    "split": split,
                    "specialist": specialist,
                    "slice_type": slice_type,
                    "slice_value": slice_value,
                    "n": len(part),
                    "positive_rate": float(target.mean()),
                    "ap_category_count": min(bge_category_count, specialist_category_count),
                    "bge_macro_ap": bge_macro,
                    "specialist_macro_ap": specialist_macro,
                    "macro_ap_delta": specialist_macro - bge_macro,
                    "bge_overall_ap": safe_average_precision(target, bge),
                    "specialist_overall_ap": safe_average_precision(target, score),
                    "overall_ap_delta": safe_average_precision(target, score)
                    - safe_average_precision(target, bge),
                    "bge_binary_error_rate_at_0p5": float(part["bge_binary_error"].mean()),
                    "specialist_binary_error_rate_at_0p5": float(
                        part["specialist_binary_error"].mean()
                    ),
                    "binary_corrected": corrected,
                    "binary_regressed": regressed,
                    "binary_net_correction": corrected - regressed,
                    "binary_net_correction_rate": (corrected - regressed) / len(part),
                    "logloss_specialist_better": int(
                        logloss_counts.get("specialist_better", 0)
                    ),
                    "logloss_bge_better": int(logloss_counts.get("bge_better", 0)),
                    "bge_logloss": float(part["bge_logloss"].mean()),
                    "specialist_logloss": float(part["specialist_logloss"].mean()),
                    "logloss_gain": float(part["logloss_gain"].mean()),
                    "absolute_error_specialist_better": int(
                        absolute_counts.get("specialist_better", 0)
                    ),
                    "absolute_error_bge_better": int(
                        absolute_counts.get("bge_better", 0)
                    ),
                    "bge_mae": float(part["bge_absolute_error"].mean()),
                    "specialist_mae": float(part["specialist_absolute_error"].mean()),
                    "mae_gain": float(part["absolute_error_gain"].mean()),
                }
            )
    return rows


def stable_slice_table(slice_metrics: pd.DataFrame, *, min_size: int = 100) -> pd.DataFrame:
    key = ["specialist", "slice_type", "slice_value"]
    metric_columns = [
        "n",
        "macro_ap_delta",
        "logloss_gain",
        "mae_gain",
        "binary_net_correction_rate",
    ]
    parts = []
    for split in ("iid", "hard", "ood"):
        part = slice_metrics.loc[slice_metrics["split"] == split, key + metric_columns].copy()
        part = part.rename(columns={column: f"{split}_{column}" for column in metric_columns})
        parts.append(part)
    result = parts[0]
    for part in parts[1:]:
        result = result.merge(part, on=key, how="outer", validate="one_to_one")

    def proxy_better(split: str) -> pd.Series:
        support = result[f"{split}_n"].fillna(0) >= min_size
        logloss_better = result[f"{split}_logloss_gain"].fillna(-np.inf) > 0
        mae_better = result[f"{split}_mae_gain"].fillna(-np.inf) > 0
        net_not_worse = result[f"{split}_binary_net_correction_rate"].fillna(-np.inf) >= 0
        ap = result[f"{split}_macro_ap_delta"]
        ap_not_worse = ap.isna() | (ap > 0)
        return support & logloss_better & mae_better & net_not_worse & ap_not_worse

    for split in ("iid", "hard", "ood"):
        result[f"{split}_better"] = proxy_better(split)
    result["stable_iid_hard"] = result["iid_better"] & result["hard_better"]
    result["stable_all_three"] = result["stable_iid_hard"] & result["ood_better"]
    result["production_candidate"] = result["stable_iid_hard"] & (
        result["slice_type"] != "category"
    )
    result["min_iid_hard_logloss_gain"] = result[
        ["iid_logloss_gain", "hard_logloss_gain"]
    ].min(axis=1)
    return result.sort_values(
        ["production_candidate", "stable_all_three", "min_iid_hard_logloss_gain"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def alternate_scores(
    frame: pd.DataFrame,
    score_mode: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if score_mode not in SCORE_MODES:
        raise ValueError(f"Unknown score mode: {score_mode}")
    bge = frame["bge_probability"].to_numpy(dtype=np.float64)
    if score_mode == "mean_normalized_rank":
        base = normalized_rank(bge)
        alternatives = {
            specialist: 0.5
            * (base + normalized_rank(frame[f"{specialist}_probability"].to_numpy()))
            for specialist in SPECIALISTS
        }
        return base, alternatives
    specialist_weight = SCORE_MODES[score_mode]
    assert specialist_weight is not None
    alternatives = {
        specialist: (1.0 - specialist_weight) * bge
        + specialist_weight * frame[f"{specialist}_probability"].to_numpy(dtype=np.float64)
        for specialist in SPECIALISTS
    }
    return bge.copy(), alternatives


def choose_oracle_alternative(
    target: Sequence[float],
    base: Sequence[float],
    alternatives: Mapping[str, Sequence[float]],
    route_expert: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_array = np.asarray(target, dtype=np.float64)
    base_array = np.asarray(base, dtype=np.float64)
    if route_expert in SPECIALISTS:
        selected = np.asarray(alternatives[route_expert], dtype=np.float64)
        chosen = np.full(len(target_array), route_expert, dtype=object)
    elif route_expert == "best_expert":
        stacked = np.vstack(
            [np.asarray(alternatives[name], dtype=np.float64) for name in SPECIALISTS]
        )
        benefits = np.abs(base_array - target_array)[None, :] - np.abs(
            stacked - target_array[None, :]
        )
        selected_index = np.argmax(benefits, axis=0)
        selected = stacked[selected_index, np.arange(len(target_array))]
        chosen = np.asarray(SPECIALISTS, dtype=object)[selected_index]
    else:
        raise ValueError(f"Unknown route expert: {route_expert}")
    benefit = np.abs(base_array - target_array) - np.abs(selected - target_array)
    return selected, benefit, chosen


def oracle_route_mask(
    benefit: Sequence[float],
    category: Sequence[str],
    budget: float,
    policy: str,
) -> np.ndarray:
    benefit_array = np.asarray(benefit, dtype=np.float64)
    category_array = np.asarray(category, dtype=str)
    mask = np.zeros(len(benefit_array), dtype=bool)

    def select(indices: np.ndarray, capacity: int) -> None:
        if capacity <= 0:
            return
        eligible = indices[benefit_array[indices] > 0]
        if not len(eligible):
            return
        order = np.argsort(-benefit_array[eligible], kind="stable")
        mask[eligible[order[:capacity]]] = True

    if policy == "global_directional":
        select(np.arange(len(mask)), int(math.floor(budget * len(mask))))
    elif policy == "category_balanced_directional":
        for value in sorted(np.unique(category_array)):
            indices = np.flatnonzero(category_array == value)
            select(indices, int(math.floor(budget * len(indices))))
    else:
        raise ValueError(f"Unknown oracle policy: {policy}")
    return mask


def oracle_routing_rows(
    frame: pd.DataFrame,
    split: str,
    *,
    budgets: Iterable[float] = ROUTE_BUDGETS,
    score_modes: Iterable[str] = SCORE_MODES,
    policies: Iterable[str] = ("global_directional", "category_balanced_directional"),
) -> list[dict[str, Any]]:
    target = frame["target"].to_numpy(dtype=np.float64)
    category = frame["category"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for score_mode in score_modes:
        base, alternatives = alternate_scores(frame, score_mode)
        baseline_macro, _ = safe_macro_average_precision(target, base, category)
        baseline_overall = safe_average_precision(target, base)
        for route_expert in (*SPECIALISTS, "best_expert"):
            selected, benefit, chosen = choose_oracle_alternative(
                target, base, alternatives, route_expert
            )
            for policy in policies:
                for budget in budgets:
                    routed = oracle_route_mask(benefit, category, float(budget), policy)
                    final = base.copy()
                    final[routed] = selected[routed]
                    macro_ap, category_count = safe_macro_average_precision(
                        target, final, category
                    )
                    coverage = float(routed.mean())
                    gain = macro_ap - baseline_macro
                    row = {
                        "split": split,
                        "route_expert": route_expert,
                        "oracle_policy": policy,
                        "score_mode": score_mode,
                        "route_budget": float(budget),
                        "pairs": len(frame),
                        "routed_pairs": int(routed.sum()),
                        "route_coverage": coverage,
                        "positive_benefit_pairs": int((benefit > 0).sum()),
                        "baseline_macro_ap": baseline_macro,
                        "macro_ap": macro_ap,
                        "macro_ap_gain": gain,
                        "gain_per_route_coverage": gain / coverage if coverage else 0.0,
                        "baseline_overall_ap": baseline_overall,
                        "overall_ap": safe_average_precision(target, final),
                        "ap_category_count": category_count,
                        "routed_minilm": int(((chosen == "minilm") & routed).sum()),
                        "routed_rumodernbert": int(
                            ((chosen == "rumodernbert") & routed).sum()
                        ),
                        "selection_uses_label": True,
                    }
                    rows.append(row)
    return rows


def compact_oracle_summary(results: pd.DataFrame) -> pd.DataFrame:
    key = ["split", "route_budget", "route_expert"]
    best_indices = results.groupby(key, sort=True)["macro_ap"].idxmax()
    return results.loc[
        best_indices,
        [
            *key,
            "macro_ap_gain",
            "macro_ap",
            "route_coverage",
            "gain_per_route_coverage",
            "oracle_policy",
            "score_mode",
            "routed_minilm",
            "routed_rumodernbert",
        ],
    ].sort_values(key, kind="stable").reset_index(drop=True)

