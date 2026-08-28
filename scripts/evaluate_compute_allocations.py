#!/usr/bin/env python3
"""OOF-safe compute allocation for BGE, MiniLM and RuModernBERT."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_selective_specialists import SPLITS, load_split
from scripts.train_benefit_routers import (
    bge_prediction_view,
    catboost_parameters,
    load_train,
    macro_ap,
)
from src.benefit_router import benefit_targets, router_feature_frame
from src.deterministic_specialist_routing import top_budget_mask


EXPERIMENT = "compute_allocation_bge_minilm_rumodern_v1"
SEQUENTIAL_ROUTER_COLUMN = "benefit_rumodernbert_over_ensemble_classification"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/compute_allocation_bge_minilm_rumodern.json",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reuse-sequential-router", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sequential_features(base: pd.DataFrame, bge: np.ndarray, mini: np.ndarray) -> pd.DataFrame:
    result = base.copy()
    bge = np.asarray(bge, dtype=np.float64)
    mini = np.asarray(mini, dtype=np.float64)
    if len(result) != len(bge) or len(bge) != len(mini):
        raise ValueError("Sequential features and neural scores are misaligned")
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


def fit_sequential_router(
    config: dict[str, Any],
    frame: pd.DataFrame,
    features: pd.DataFrame,
    categorical: list[str],
    artifact_dir: Path,
    smoke: bool,
) -> tuple[np.ndarray, Any, dict[str, Any]]:
    from catboost import CatBoostClassifier, Pool

    bge = frame["bge_probability"].to_numpy(dtype=np.float64)
    mini = frame["minilm_probability"].to_numpy(dtype=np.float64)
    ru = frame["rumodernbert_probability"].to_numpy(dtype=np.float64)
    current = 0.5 * bge + 0.5 * mini
    _, target = benefit_targets(
        frame["target"],
        current,
        ru,
        classification_margin=float(config["classification_margin_logloss"]),
    )
    folds = frame["fold"].to_numpy(dtype=np.int8)
    oof = np.full(len(frame), np.nan, dtype=np.float64)
    parameters = catboost_parameters(config, "classification", smoke)
    for fold in sorted(np.unique(folds)):
        train_idx = np.flatnonzero(folds != fold)
        valid_idx = np.flatnonzero(folds == fold)
        model = CatBoostClassifier(**parameters)
        model.fit(Pool(features.iloc[train_idx], target[train_idx], cat_features=categorical))
        oof[valid_idx] = model.predict_proba(
            Pool(features.iloc[valid_idx], cat_features=categorical)
        )[:, 1]
    if not np.isfinite(oof).all():
        raise RuntimeError("Sequential RuModern router produced incomplete OOF")
    model = CatBoostClassifier(**parameters)
    model.fit(Pool(features, target, cat_features=categorical))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "router_rumodernbert_over_bge_minilm_classification.cbm"
    model.save_model(model_path)
    pd.DataFrame(
        {
            "id1": frame["id1"],
            "id2": frame["id2"],
            "fold": frame["fold"],
            SEQUENTIAL_ROUTER_COLUMN: oof.astype(np.float32),
        }
    ).to_parquet(artifact_dir / "router_oof_predictions.parquet", index=False)
    manifest = {
        "status": "complete",
        "target": "logloss(current_50_50_bge_minilm) - logloss(rumodernbert)",
        "target_kind": "classification",
        "classification_margin_logloss": float(config["classification_margin_logloss"]),
        "component_disjoint_router_oof": True,
        "ru_score_used_as_feature": False,
        "mini_score_used_as_feature": True,
        "feature_columns": features.columns.tolist(),
        "categorical_columns": categorical,
        "classification_positive_rate": float(target.mean()),
        "model_sha256": sha256_file(model_path),
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return oof.astype(np.float32), model, manifest


def load_sequential_router(
    artifact_dir: Path, frame: pd.DataFrame, expected_columns: list[str]
) -> tuple[np.ndarray, Any, dict[str, Any]]:
    from catboost import CatBoostClassifier

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["feature_columns"] != expected_columns:
        raise ValueError("Saved sequential router feature schema differs")
    saved = pd.read_parquet(artifact_dir / "router_oof_predictions.parquet")
    if not saved[["id1", "id2"]].equals(frame[["id1", "id2"]].reset_index(drop=True)):
        raise ValueError("Saved sequential router OOF is not aligned")
    model = CatBoostClassifier()
    model.load_model(
        artifact_dir / "router_rumodernbert_over_bge_minilm_classification.cbm"
    )
    return saved[SEQUENTIAL_ROUTER_COLUMN].to_numpy(dtype=np.float32), model, manifest


def restricted_top_mask(
    priority: np.ndarray,
    eligible: np.ndarray,
    coverage: float,
    id1: pd.Series,
    id2: pd.Series,
) -> np.ndarray:
    count = int(np.floor(len(priority) * coverage + 1e-12))
    candidates = np.flatnonzero(np.asarray(eligible, dtype=bool))
    if count > len(candidates):
        raise ValueError("Requested coverage exceeds eligible hierarchical subset")
    order = np.lexsort(
        (
            id2.to_numpy(dtype=np.int64)[candidates],
            id1.to_numpy(dtype=np.int64)[candidates],
            -np.asarray(priority, dtype=np.float64)[candidates],
        )
    )
    result = np.zeros(len(priority), dtype=bool)
    result[candidates[order[:count]]] = True
    return result


def exclusive_masks(
    mini_priority: np.ndarray,
    ru_priority: np.ndarray,
    mini_coverage: float,
    ru_coverage: float,
    id1: pd.Series,
    id2: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    size = len(mini_priority)
    quotas = [
        int(np.floor(size * mini_coverage + 1e-12)),
        int(np.floor(size * ru_coverage + 1e-12)),
    ]
    if sum(quotas) > size:
        raise ValueError("Exclusive specialist coverages exceed 100%")
    left = id1.to_numpy(dtype=np.int64)
    right = id2.to_numpy(dtype=np.int64)
    rows = np.tile(np.arange(size, dtype=np.int64), 2)
    experts = np.repeat(np.arange(2, dtype=np.int8), size)
    priorities = np.concatenate(
        [np.asarray(mini_priority, dtype=np.float64), np.asarray(ru_priority, dtype=np.float64)]
    )
    order = np.lexsort((experts, right[rows], left[rows], -priorities))
    assigned = np.zeros(size, dtype=bool)
    masks = [np.zeros(size, dtype=bool), np.zeros(size, dtype=bool)]
    remaining = quotas.copy()
    for candidate in order:
        expert = int(experts[candidate])
        row = int(rows[candidate])
        if remaining[expert] and not assigned[row]:
            masks[expert][row] = True
            assigned[row] = True
            remaining[expert] -= 1
            if not any(remaining):
                break
    if any(remaining):
        raise RuntimeError(f"Exclusive allocation did not fill quotas: {remaining}")
    return masks[0], masks[1]


def aggregate_scores(
    frame: pd.DataFrame,
    mini_mask: np.ndarray,
    ru_mask: np.ndarray,
    mini_weight: float,
    ru_weight: float,
    scheme: str,
) -> np.ndarray:
    bge = frame["bge_probability"].to_numpy(dtype=np.float64)
    mini = frame["minilm_probability"].to_numpy(dtype=np.float64)
    ru = frame["rumodernbert_probability"].to_numpy(dtype=np.float64)
    score = bge.copy()
    score[mini_mask] = (1.0 - mini_weight) * bge[mini_mask] + mini_weight * mini[mini_mask]
    if scheme == "exclusive":
        if np.any(mini_mask & ru_mask):
            raise AssertionError("Exclusive masks overlap")
        score[ru_mask] = (1.0 - ru_weight) * bge[ru_mask] + ru_weight * ru[ru_mask]
    elif scheme == "hierarchical":
        if np.any(ru_mask & ~mini_mask):
            raise AssertionError("Hierarchical RuModern mask must be inside MiniLM mask")
        score[ru_mask] = (1.0 - ru_weight) * score[ru_mask] + ru_weight * ru[ru_mask]
    elif scheme != "mini_only":
        raise ValueError(f"Unknown scheme: {scheme}")
    return score


def best_weights(
    frame: pd.DataFrame,
    mini_mask: np.ndarray,
    ru_mask: np.ndarray,
    scheme: str,
    mini_weights: list[float],
    ru_weights: list[float],
) -> tuple[float, float, float]:
    candidates = []
    actual_ru_weights = [0.0] if scheme == "mini_only" else ru_weights
    for mini_weight in mini_weights:
        for ru_weight in actual_ru_weights:
            score = aggregate_scores(
                frame, mini_mask, ru_mask, mini_weight, ru_weight, scheme
            )
            candidates.append(
                (
                    macro_ap(
                        frame["target"].to_numpy(dtype=np.int8),
                        score,
                        frame["category"].astype(str).to_numpy(),
                    ),
                    mini_weight,
                    ru_weight,
                )
            )
    return max(candidates, key=lambda item: (item[0], -item[1], -item[2]))


def select_oof_policies(config: dict[str, Any], frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    mini_priority = frame["benefit_minilm_classification"].to_numpy(dtype=np.float64)
    ru_bge_priority = frame["benefit_rumodernbert_classification"].to_numpy(dtype=np.float64)
    ru_ensemble_priority = frame[SEQUENTIAL_ROUTER_COLUMN].to_numpy(dtype=np.float64)
    mini_weights = [float(value) for value in config["mini_weights"]]
    ru_weights = [float(value) for value in config["ru_weights"]]
    rows: list[dict[str, Any]] = []

    full_mask = np.ones(len(frame), dtype=bool)
    full_ap, full_weight, _ = best_weights(
        frame, full_mask, np.zeros(len(frame), dtype=bool), "mini_only", mini_weights, [0.0]
    )
    bge_ap = macro_ap(
        frame["target"].to_numpy(dtype=np.int8),
        frame["bge_probability"].to_numpy(dtype=np.float64),
        frame["category"].astype(str).to_numpy(),
    )

    for mini_coverage in config["mini_coverages"]:
        mini_mask = top_budget_mask(
            mini_priority, float(mini_coverage), frame["id1"], frame["id2"]
        )
        ap, mini_weight, ru_weight = best_weights(
            frame,
            mini_mask,
            np.zeros(len(frame), dtype=bool),
            "mini_only",
            mini_weights,
            [0.0],
        )
        rows.append(
            {
                "architecture": f"mini_{float(mini_coverage):.2f}",
                "scheme": "mini_only",
                "mini_coverage": float(mini_coverage),
                "ru_coverage": 0.0,
                "ru_router_target": "none",
                "mini_weight": mini_weight,
                "ru_weight": ru_weight,
                "oof_macro_ap": ap,
            }
        )

    for mini_coverage in config["mini_coverages"]:
        for ru_coverage in config["ru_coverages"]:
            if float(mini_coverage) + float(ru_coverage) <= 1.0:
                mini_mask, ru_mask = exclusive_masks(
                    mini_priority,
                    ru_bge_priority,
                    float(mini_coverage),
                    float(ru_coverage),
                    frame["id1"],
                    frame["id2"],
                )
                ap, mini_weight, ru_weight = best_weights(
                    frame, mini_mask, ru_mask, "exclusive", mini_weights, ru_weights
                )
                rows.append(
                    {
                        "architecture": f"exclusive_m{float(mini_coverage):.2f}_r{float(ru_coverage):.2f}",
                        "scheme": "exclusive",
                        "mini_coverage": float(mini_coverage),
                        "ru_coverage": float(ru_coverage),
                        "ru_router_target": "vs_bge",
                        "mini_weight": mini_weight,
                        "ru_weight": ru_weight,
                        "oof_macro_ap": ap,
                    }
                )

            hierarchy_candidates = []
            hierarchical_mini = top_budget_mask(
                mini_priority, float(mini_coverage), frame["id1"], frame["id2"]
            )
            for target_name, priority in (
                ("vs_bge", ru_bge_priority),
                ("vs_bge_minilm", ru_ensemble_priority),
            ):
                hierarchical_ru = restricted_top_mask(
                    priority,
                    hierarchical_mini,
                    float(ru_coverage),
                    frame["id1"],
                    frame["id2"],
                )
                candidate = best_weights(
                    frame,
                    hierarchical_mini,
                    hierarchical_ru,
                    "hierarchical",
                    mini_weights,
                    ru_weights,
                )
                hierarchy_candidates.append((*candidate, target_name))
            ap, mini_weight, ru_weight, target_name = max(
                hierarchy_candidates, key=lambda item: (item[0], item[3] == "vs_bge_minilm")
            )
            rows.append(
                {
                    "architecture": f"hierarchical_m{float(mini_coverage):.2f}_r{float(ru_coverage):.2f}",
                    "scheme": "hierarchical",
                    "mini_coverage": float(mini_coverage),
                    "ru_coverage": float(ru_coverage),
                    "ru_router_target": target_name,
                    "mini_weight": mini_weight,
                    "ru_weight": ru_weight,
                    "oof_macro_ap": ap,
                }
            )
    policies = pd.DataFrame(rows)
    policies["oof_delta_vs_bge"] = policies["oof_macro_ap"] - bge_ap
    policies["oof_delta_vs_full_bge_minilm"] = policies["oof_macro_ap"] - full_ap
    anchor = {
        "bge_oof_macro_ap": bge_ap,
        "full_bge_minilm_oof_macro_ap": full_ap,
        "full_bge_minilm_weight": full_weight,
    }
    return policies, anchor


def validation_base_features(
    config: dict[str, Any],
    split: str,
    filename: str,
    frame: pd.DataFrame,
    feature_columns: list[str],
    categorical: list[str],
) -> pd.DataFrame:
    reserved = {
        "id1", "id2", "target", "category", "bge_probability",
        "minilm_probability", "rumodernbert_probability", "bge_token_length_max",
    }
    cheap_columns = [column for column in frame.columns if column not in reserved]
    cheap = frame[["category", *cheap_columns]].copy()
    bge_raw = pd.read_parquet(
        resolve(config["predictions_root"]) / "preds_bge" / filename,
        columns=[
            "id1", "id2", "score", "logit", "score_order_gap",
            "token_length_ab", "token_length_ba",
        ],
    )
    if not bge_raw[["id1", "id2"]].reset_index(drop=True).equals(
        frame[["id1", "id2"]].reset_index(drop=True)
    ):
        raise ValueError(f"Raw BGE predictions differ on {split}")
    features, actual_categorical = router_feature_frame(cheap, bge_prediction_view(bge_raw))
    if actual_categorical != categorical:
        raise ValueError(f"Categorical feature schema differs on {split}")
    if set(features) != set(feature_columns):
        raise ValueError(f"Base router feature schema differs on {split}")
    return features[feature_columns]


def apply_policy(frame: pd.DataFrame, policy: Any) -> np.ndarray:
    mini_priority = frame["benefit_minilm_classification"].to_numpy(dtype=np.float64)
    ru_bge_priority = frame["benefit_rumodernbert_classification"].to_numpy(dtype=np.float64)
    if policy.scheme == "mini_only":
        mini_mask = top_budget_mask(
            mini_priority, policy.mini_coverage, frame["id1"], frame["id2"]
        )
        ru_mask = np.zeros(len(frame), dtype=bool)
    elif policy.scheme == "exclusive":
        mini_mask, ru_mask = exclusive_masks(
            mini_priority,
            ru_bge_priority,
            policy.mini_coverage,
            policy.ru_coverage,
            frame["id1"],
            frame["id2"],
        )
    else:
        mini_mask = top_budget_mask(
            mini_priority, policy.mini_coverage, frame["id1"], frame["id2"]
        )
        ru_priority = (
            frame[SEQUENTIAL_ROUTER_COLUMN].to_numpy(dtype=np.float64)
            if policy.ru_router_target == "vs_bge_minilm"
            else ru_bge_priority
        )
        ru_mask = restricted_top_mask(
            ru_priority,
            mini_mask,
            policy.ru_coverage,
            frame["id1"],
            frame["id2"],
        )
    return aggregate_scores(
        frame,
        mini_mask,
        ru_mask,
        policy.mini_weight,
        policy.ru_weight,
        policy.scheme,
    )


def evaluate_validation(
    config: dict[str, Any],
    policies: pd.DataFrame,
    anchor: dict,
    base_manifest: dict,
    sequential_manifest: dict,
    sequential_model: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from catboost import CatBoostClassifier, Pool

    base_dir = resolve(config["base_router_artifact_dir"])
    mini_model = CatBoostClassifier()
    mini_model.load_model(base_dir / "models/router_minilm_classification.cbm")
    ru_bge_model = CatBoostClassifier()
    ru_bge_model.load_model(base_dir / "models/router_rumodernbert_classification.cbm")
    categorical = list(base_manifest["categorical_columns"])
    rows = []
    sources: dict[str, Any] = {}
    for split, filename in SPLITS.items():
        frame, split_sources = load_split(
            resolve(config["predictions_root"]),
            resolve(config["validation_feature_cache_dir"]),
            split,
            filename,
            specialists=("minilm", "rumodernbert"),
        )
        sources.update(split_sources)
        base_features = validation_base_features(
            config,
            split,
            filename,
            frame,
            list(base_manifest["feature_columns"]),
            categorical,
        )
        base_pool = Pool(base_features, cat_features=categorical)
        frame["benefit_minilm_classification"] = mini_model.predict_proba(base_pool)[:, 1]
        frame["benefit_rumodernbert_classification"] = ru_bge_model.predict_proba(base_pool)[:, 1]
        sequential = sequential_features(
            base_features,
            frame["bge_probability"].to_numpy(dtype=np.float64),
            frame["minilm_probability"].to_numpy(dtype=np.float64),
        )
        if sequential.columns.tolist() != sequential_manifest["feature_columns"]:
            raise ValueError(f"Sequential router feature schema differs on {split}")
        frame[SEQUENTIAL_ROUTER_COLUMN] = sequential_model.predict_proba(
            Pool(sequential, cat_features=categorical)
        )[:, 1]
        bge = frame["bge_probability"].to_numpy(dtype=np.float64)
        full = (1.0 - anchor["full_bge_minilm_weight"]) * bge + anchor[
            "full_bge_minilm_weight"
        ] * frame["minilm_probability"].to_numpy(dtype=np.float64)
        for architecture, score, mini_coverage, ru_coverage in (
            ("bge_100", bge, 0.0, 0.0),
            ("full_bge_minilm", full, 1.0, 0.0),
        ):
            rows.append(
                {
                    "architecture": architecture,
                    "split": split,
                    "mini_coverage": mini_coverage,
                    "ru_coverage": ru_coverage,
                    "macro_ap": macro_ap(
                        frame["target"].to_numpy(dtype=np.int8),
                        score,
                        frame["category"].astype(str).to_numpy(),
                    ),
                }
            )
        for policy in policies.itertuples(index=False):
            score = apply_policy(frame, policy)
            rows.append(
                {
                    "architecture": policy.architecture,
                    "split": split,
                    "mini_coverage": policy.mini_coverage,
                    "ru_coverage": policy.ru_coverage,
                    "macro_ap": macro_ap(
                        frame["target"].to_numpy(dtype=np.int8),
                        score,
                        frame["category"].astype(str).to_numpy(),
                    ),
                }
            )
    return pd.DataFrame(rows), sources


def estimated_runtime(config: dict[str, Any], mini_coverage: float, ru_coverage: float) -> float:
    runtime = config["runtime_seconds"]
    values = [
        runtime["fixed_and_routing"],
        runtime["bge_100"],
        runtime["minilm_100"],
        runtime["rumodernbert_100"],
    ]
    if any(value is None for value in values):
        return float("nan")
    return float(values[0]) + float(values[1]) + mini_coverage * float(values[2]) + ru_coverage * float(values[3])


def build_main_table(
    config: dict[str, Any], policies: pd.DataFrame, validation: pd.DataFrame
) -> pd.DataFrame:
    policy_lookup = policies.set_index("architecture")
    anchor_part = validation.loc[validation["architecture"].eq("full_bge_minilm")].set_index("split")
    anchor_mean = float(anchor_part["macro_ap"].mean())
    rows = []
    for architecture, part in validation.groupby("architecture", sort=False):
        by_split = part.set_index("split")
        values = {split: float(by_split.loc[split, "macro_ap"]) for split in SPLITS}
        mini_coverage = float(part["mini_coverage"].iloc[0])
        ru_coverage = float(part["ru_coverage"].iloc[0])
        if architecture in policy_lookup.index:
            policy = policy_lookup.loc[architecture]
            scheme = str(policy["scheme"])
            ru_target = str(policy["ru_router_target"])
            mini_weight = float(policy["mini_weight"])
            ru_weight = float(policy["ru_weight"])
            oof_ap = float(policy["oof_macro_ap"])
        else:
            scheme = "baseline"
            ru_target = "none"
            mini_weight = 0.0 if architecture == "bge_100" else float("nan")
            ru_weight = 0.0
            oof_ap = float("nan")
        mean_ap = float(np.mean(list(values.values())))
        runtime_seconds = estimated_runtime(config, mini_coverage, ru_coverage)
        if architecture == "full_bge_minilm" and config["runtime_seconds"]["full_bge_minilm_end_to_end"] is not None:
            runtime_seconds = float(config["runtime_seconds"]["full_bge_minilm_end_to_end"])
        rows.append(
            {
                "architecture": architecture,
                "scheme": scheme,
                "mini_coverage": mini_coverage,
                "ru_coverage": ru_coverage,
                "ru_router_target": ru_target,
                "mini_weight": mini_weight,
                "ru_weight": ru_weight,
                "oof_macro_ap": oof_ap,
                "iid_macro_ap": values["iid"],
                "hard_macro_ap": values["hard"],
                "ood_macro_ap": values["ood"],
                "mean_macro_ap": mean_ap,
                "delta_vs_full_bge_minilm": mean_ap - anchor_mean,
                "estimated_private_runtime_seconds": runtime_seconds,
                "passes_13_minutes": bool(runtime_seconds <= 780.0) if np.isfinite(runtime_seconds) else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def pareto_frontier(table: pd.DataFrame) -> pd.DataFrame:
    candidates = table.loc[~table["architecture"].eq("bge_100")].copy()
    runtime_available = candidates["estimated_private_runtime_seconds"].notna().all()
    keep = []
    for row in candidates.itertuples(index=False):
        if runtime_available:
            dominated = (
                (candidates["estimated_private_runtime_seconds"] <= row.estimated_private_runtime_seconds)
                & (candidates["mean_macro_ap"] >= row.mean_macro_ap)
                & (
                    (candidates["estimated_private_runtime_seconds"] < row.estimated_private_runtime_seconds)
                    | (candidates["mean_macro_ap"] > row.mean_macro_ap)
                )
            ).any()
        else:
            dominated = (
                (candidates["mini_coverage"] <= row.mini_coverage)
                & (candidates["ru_coverage"] <= row.ru_coverage)
                & (candidates["mean_macro_ap"] >= row.mean_macro_ap)
                & (
                    (candidates["mini_coverage"] < row.mini_coverage)
                    | (candidates["ru_coverage"] < row.ru_coverage)
                    | (candidates["mean_macro_ap"] > row.mean_macro_ap)
                )
            ).any()
        keep.append(not dominated)
    return candidates.loc[keep].sort_values(
        ["mini_coverage", "ru_coverage", "mean_macro_ap"],
        ascending=[True, True, False],
    )


def choose_candidates(table: pd.DataFrame) -> pd.DataFrame:
    specialists = table.loc[~table["scheme"].isin(["baseline", "mini_only"])].copy()
    runtime_known = specialists["estimated_private_runtime_seconds"].notna().all()
    if runtime_known:
        eligible = specialists.loc[specialists["estimated_private_runtime_seconds"] <= 780.0]
        quality_pool = eligible if not eligible.empty else specialists
    else:
        quality_pool = specialists
    quality = quality_pool.sort_values("mean_macro_ap", ascending=False).iloc[0]
    reserve_pool = specialists.loc[
        (specialists["mini_coverage"] <= 0.40) & (specialists["ru_coverage"] <= 0.10)
    ]
    reserve = reserve_pool.sort_values("mean_macro_ap", ascending=False).iloc[0]
    result = pd.DataFrame([quality, reserve]).drop_duplicates("architecture")
    result.insert(0, "candidate_role", ["production_quality", "reserved_compute"][: len(result)])
    return result


def report_markdown(table: pd.DataFrame, frontier: pd.DataFrame, candidates: pd.DataFrame) -> str:
    ordered = table.sort_values(["mean_macro_ap", "mini_coverage"], ascending=[False, True])
    lines = [
        "# BGE / MiniLM / RuModernBERT compute allocation",
        "",
        "Routing targets, coverage masks and aggregation weights were frozen on component-disjoint human-train OOF before IID/Hard/OOD evaluation.",
        "The sequential RuModern router never sees the RuModern score; it may use MiniLM only because it is evaluated inside the MiniLM-routed subset.",
        "",
        "Absolute runtime is pending new end-to-end H100 measurements. The frontier below is therefore cost-independent: a row is dominated only when another row uses no more MiniLM and no more RuModern coverage while giving at least the same mean AP.",
        "",
        "| architecture | Mini | Ru | scheme | Ru target | IID | Hard | OOD | mean | delta vs full BGE+Mini |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ordered.itertuples(index=False):
        lines.append(
            f"| {row.architecture} | {row.mini_coverage:.0%} | {row.ru_coverage:.0%} | "
            f"{row.scheme} | {row.ru_router_target} | {row.iid_macro_ap:.6f} | "
            f"{row.hard_macro_ap:.6f} | {row.ood_macro_ap:.6f} | {row.mean_macro_ap:.6f} | "
            f"{row.delta_vs_full_bge_minilm:+.6f} |"
        )
    lines.extend(["", "## Pareto frontier", ""])
    lines.append(", ".join(frontier["architecture"].tolist()))
    lines.extend(["", "## Provisional candidates", ""])
    for row in candidates.itertuples(index=False):
        lines.append(
            f"- {row.candidate_role}: `{row.architecture}` — mean {row.mean_macro_ap:.6f}, "
            f"MiniLM {row.mini_coverage:.0%}, RuModern {row.ru_coverage:.0%}."
        )
    lines.extend(
        [
            "",
            "Final 13-minute and reserved-runtime choices must be recomputed after filling `runtime_seconds` in the config with the new implementation timings.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or resolve(config["output_dir"])
    if args.smoke and args.output_dir is None:
        output_dir = output_dir.with_name(output_dir.name + "_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    sequential_dir = resolve(config["sequential_router_artifact_dir"])
    if args.smoke:
        sequential_dir = sequential_dir.with_name(sequential_dir.name + "_smoke")

    frame, cheap = load_train(config, args.smoke, ("minilm", "rumodernbert"))
    base_features, categorical = router_feature_frame(cheap, bge_prediction_view(frame))
    base_manifest = json.loads(
        (resolve(config["base_router_artifact_dir"]) / "manifest.json").read_text(encoding="utf-8")
    )
    if base_features.columns.tolist() != base_manifest["feature_columns"]:
        raise ValueError("Existing base router schema does not match rebuilt OOF features")
    sequential = sequential_features(
        base_features,
        frame["bge_probability"].to_numpy(dtype=np.float64),
        frame["minilm_probability"].to_numpy(dtype=np.float64),
    )
    if args.reuse_sequential_router:
        sequential_oof, sequential_model, sequential_manifest = load_sequential_router(
            sequential_dir, frame, sequential.columns.tolist()
        )
    else:
        sequential_oof, sequential_model, sequential_manifest = fit_sequential_router(
            config, frame, sequential, categorical, sequential_dir, args.smoke
        )

    saved_base_oof = pd.read_parquet(
        resolve(config["base_router_artifact_dir"]) / "router_oof_predictions.parquet"
    )
    base_oof = frame[["id1", "id2"]].merge(
        saved_base_oof[
            [
                "id1",
                "id2",
                "benefit_minilm_classification",
                "benefit_rumodernbert_classification",
            ]
        ],
        on=["id1", "id2"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if base_oof.isna().any().any():
        raise ValueError("Existing base-router OOF is not aligned")
    for column in (
        "benefit_minilm_classification",
        "benefit_rumodernbert_classification",
    ):
        frame[column] = base_oof[column].to_numpy(dtype=np.float32)
    frame[SEQUENTIAL_ROUTER_COLUMN] = sequential_oof
    policies, anchor = select_oof_policies(config, frame)
    policies.to_csv(output_dir / "oof_policy_selection.csv", index=False)
    (output_dir / "frozen_policy.json").write_text(
        json.dumps(
            {"anchor": anchor, "policies": policies.to_dict("records")},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Protocol boundary: validation labels are loaded only after policies exist.
    validation, validation_sources = evaluate_validation(
        config,
        policies,
        anchor,
        base_manifest,
        sequential_manifest,
        sequential_model,
    )
    validation.to_csv(output_dir / "split_metrics.csv", index=False)
    table = build_main_table(config, policies, validation)
    table.to_csv(output_dir / "main_table.csv", index=False)
    frontier = pareto_frontier(table)
    frontier.to_csv(output_dir / "pareto_frontier.csv", index=False)
    candidates = choose_candidates(table)
    candidates.to_csv(output_dir / "recommended_architectures.csv", index=False)
    (output_dir / "REPORT.md").write_text(
        report_markdown(table, frontier, candidates), encoding="utf-8"
    )
    manifest = {
        "status": "complete",
        "experiment": EXPERIMENT,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "neural_models_retrained": False,
        "new_router": "RuModern benefit over 50/50 BGE+MiniLM current score",
        "oof_parameters_frozen_before_validation": True,
        "absolute_runtime_available": bool(table["estimated_private_runtime_seconds"].notna().all()),
        "quality_protocol": "saved max_length=384 symmetric probability predictions",
        "runtime_warning": "old timing measurements intentionally not reused",
        "validation_sources": validation_sources,
        "outputs": {
            "main_table": "main_table.csv",
            "pareto_frontier": "pareto_frontier.csv",
            "recommended_architectures": "recommended_architectures.csv",
            "report": "REPORT.md",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(output_dir),
                "anchor": anchor,
                "recommended": candidates[
                    ["candidate_role", "architecture", "mean_macro_ap"]
                ].to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
