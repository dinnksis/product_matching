#!/usr/bin/env python3
"""Train and evaluate a compact one-way RuModern router after routed MiniLM."""

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

from scripts.analyze_selective_specialists import SPLITS
from scripts.train_benefit_routers import catboost_parameters, macro_ap
from src.benefit_router import benefit_targets
from src.deterministic_specialist_routing import top_budget_mask
from src.fast_benefit_router import cached_feature_frame
from src.fast_sequential_router import sequential_feature_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/fast_oneway_rumodern_router.json",
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_frame(path: Path, oof: bool) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"id1", "id2", "target", "score_ab"}
    required.update({"fold", "component_id", "oof_row_index"} if oof else {"category_1"})
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    order = "oof_row_index" if oof else "pair_index"
    if order in frame:
        frame = frame.sort_values(order, kind="stable")
    return frame.reset_index(drop=True)


def check_alignment(frames: list[pd.DataFrame], label: str) -> None:
    key = frames[0][["id1", "id2", "target"]]
    if any(not key.equals(frame[["id1", "id2", "target"]]) for frame in frames[1:]):
        raise ValueError(f"Prediction alignment differs on {label}")


def probability_logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


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
        raise ValueError("RuModern coverage exceeds the MiniLM subset")
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


def aggregate(
    bge: np.ndarray,
    mini: np.ndarray,
    ru: np.ndarray,
    mini_mask: np.ndarray,
    ru_mask: np.ndarray,
    mini_weight: float,
    ru_weight: float,
) -> np.ndarray:
    score = np.asarray(bge, dtype=np.float64).copy()
    score[mini_mask] = (
        (1.0 - mini_weight) * score[mini_mask] + mini_weight * mini[mini_mask]
    )
    score[ru_mask] = (1.0 - ru_weight) * score[ru_mask] + ru_weight * ru[ru_mask]
    return score


def load_oof(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    root = resolve(config["predictions_root"])
    bge = prediction_frame(root / "preds_bge/oof_predictions.parquet", True)
    mini = prediction_frame(root / "preds_minilm/oof_predictions.parquet", True)
    ru = prediction_frame(root / "preds_rumodernbert/oof_predictions.parquet", True)
    check_alignment([bge, mini, ru], "OOF")
    pairs = pd.read_parquet(resolve(config["human_train_pairs_path"]))
    cheap = pd.read_parquet(resolve(config["train_feature_cache_path"]))
    if len(bge) != len(pairs) or len(bge) != len(cheap):
        raise ValueError("OOF predictions and feature cache differ in length")
    if not pairs[["id1", "id2", "target"]].equals(bge[["id1", "id2", "target"]]):
        raise ValueError("OOF predictions are not aligned with human train")
    mini_router = pd.read_parquet(resolve(config["mini_router_oof_path"]))
    if not mini_router[["id1", "id2"]].equals(bge[["id1", "id2"]]):
        raise ValueError("MiniLM router OOF is not aligned")
    frame = bge[["id1", "id2", "target", "category", "fold", "score_ab"]].rename(
        columns={"score_ab": "bge_probability"}
    )
    frame["minilm_probability"] = mini["score_ab"].to_numpy(dtype=np.float32)
    frame["rumodern_probability"] = ru["score_ab"].to_numpy(dtype=np.float32)
    base = cached_feature_frame(
        cheap,
        frame["bge_probability"],
        probability_logit(frame["bge_probability"].to_numpy()),
        "score_category",
    )
    features = sequential_feature_frame(
        base, frame["bge_probability"].to_numpy(), frame["minilm_probability"].to_numpy()
    )
    return frame, features, mini_router["benefit_score_category"].to_numpy(dtype=np.float64)


def fit_router(
    config: dict[str, Any], frame: pd.DataFrame, features: pd.DataFrame, output_dir: Path
) -> tuple[np.ndarray, Any, dict[str, Any]]:
    from catboost import CatBoostClassifier, Pool

    current = (
        float(config["mini_weight"]) * frame["minilm_probability"].to_numpy(dtype=np.float64)
        + (1.0 - float(config["mini_weight"]))
        * frame["bge_probability"].to_numpy(dtype=np.float64)
    )
    _, target = benefit_targets(
        frame["target"],
        current,
        frame["rumodern_probability"],
        classification_margin=float(config["classification_margin_logloss"]),
    )
    folds = frame["fold"].to_numpy(dtype=np.int8)
    oof = np.full(len(frame), np.nan, dtype=np.float64)
    parameters = catboost_parameters(config, "classification", False)
    for fold in sorted(np.unique(folds)):
        train = np.flatnonzero(folds != fold)
        valid = np.flatnonzero(folds == fold)
        model = CatBoostClassifier(**parameters)
        model.fit(Pool(features.iloc[train], target[train], cat_features=["category"]))
        oof[valid] = model.predict_proba(
            Pool(features.iloc[valid], cat_features=["category"])
        )[:, 1]
    if not np.isfinite(oof).all():
        raise RuntimeError("Incomplete component-disjoint RuModern router OOF")
    model = CatBoostClassifier(**parameters)
    model.fit(Pool(features, target, cat_features=["category"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "router_rumodern_oneway_score_category.cbm"
    model.save_model(model_path)
    pd.DataFrame(
        {"id1": frame["id1"], "id2": frame["id2"], "fold": frame["fold"], "priority": oof}
    ).to_parquet(output_dir / "router_oof_predictions.parquet", index=False)
    manifest = {
        "status": "complete",
        "target": "logloss(oneway_50_50_bge_minilm) - logloss(rumodern_oneway)",
        "component_disjoint_router_oof": True,
        "feature_columns": features.columns.tolist(),
        "categorical_columns": ["category"],
        "classification_positive_rate": float(target.mean()),
        "model_sha256": sha256(model_path),
    }
    return oof, model, manifest


def evaluate_oof(
    config: dict[str, Any], frame: pd.DataFrame, mini_priority: np.ndarray, ru_priority: np.ndarray
) -> tuple[dict[str, float], dict[str, Any]]:
    target = frame["target"].to_numpy(dtype=np.int8)
    category = frame["category"].astype(str).to_numpy()
    bge = frame["bge_probability"].to_numpy(dtype=np.float64)
    mini = frame["minilm_probability"].to_numpy(dtype=np.float64)
    ru = frame["rumodern_probability"].to_numpy(dtype=np.float64)
    mini_mask = top_budget_mask(
        mini_priority, float(config["mini_coverage"]), frame["id1"], frame["id2"]
    )
    ru_mask = restricted_top_mask(
        ru_priority,
        mini_mask,
        float(config["rumodern_coverage"]),
        frame["id1"],
        frame["id2"],
    )
    mini_weight = float(config["mini_weight"])
    candidates = []
    for ru_weight in config["rumodern_weights"]:
        score = aggregate(bge, mini, ru, mini_mask, ru_mask, mini_weight, float(ru_weight))
        candidates.append((macro_ap(target, score, category), float(ru_weight)))
    best_ap, ru_weight = max(candidates, key=lambda value: (value[0], -value[1]))
    mini_only = aggregate(
        bge, mini, ru, mini_mask, np.zeros(len(frame), dtype=bool), mini_weight, 0.0
    )
    metrics = {
        "bge": macro_ap(target, bge, category),
        "mini_40": macro_ap(target, mini_only, category),
        "mini_40_ru_5": best_ap,
    }
    policy = {
        "variant": "score_category",
        "mini_coverage": float(config["mini_coverage"]),
        "rumodern_coverage": float(config["rumodern_coverage"]),
        "mini_weight": mini_weight,
        "rumodern_weight": ru_weight,
        "rumodern_weight_candidates": [
            {"weight": weight, "oof_macro_ap": ap} for ap, weight in candidates
        ],
    }
    return metrics, policy


def load_validation(
    config: dict[str, Any], split: str, filename: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = resolve(config["predictions_root"])
    bge = prediction_frame(root / "preds_bge" / filename, False)
    mini = prediction_frame(root / "preds_minilm" / filename, False)
    ru = prediction_frame(root / "preds_rumodernbert" / filename, False)
    check_alignment([bge, mini, ru], split)
    cheap = pd.read_parquet(resolve(config["validation_feature_cache_dir"]) / f"{split}_base.parquet")
    if len(cheap) != len(bge):
        raise ValueError(f"Feature cache length differs on {split}")
    frame = bge[["id1", "id2", "target", "category_1", "score_ab"]].rename(
        columns={"category_1": "category", "score_ab": "bge_probability"}
    )
    frame["minilm_probability"] = mini["score_ab"].to_numpy(dtype=np.float32)
    frame["rumodern_probability"] = ru["score_ab"].to_numpy(dtype=np.float32)
    base = cached_feature_frame(
        cheap,
        frame["bge_probability"],
        probability_logit(frame["bge_probability"].to_numpy()),
        "score_category",
    )
    return frame, base


def evaluate_validation(
    config: dict[str, Any], policy: dict[str, Any], ru_model: Any
) -> pd.DataFrame:
    from catboost import CatBoostClassifier, Pool

    mini_model = CatBoostClassifier()
    mini_model.load_model(resolve(config["mini_router_model_path"]))
    rows = []
    for split, filename in SPLITS.items():
        frame, base = load_validation(config, split, filename)
        mini_priority = mini_model.predict_proba(Pool(base, cat_features=["category"]))[:, 1]
        sequential = sequential_feature_frame(
            base,
            frame["bge_probability"].to_numpy(),
            frame["minilm_probability"].to_numpy(),
        )
        ru_priority = ru_model.predict_proba(
            Pool(sequential, cat_features=["category"])
        )[:, 1]
        mini_mask = top_budget_mask(
            mini_priority, policy["mini_coverage"], frame["id1"], frame["id2"]
        )
        ru_mask = restricted_top_mask(
            ru_priority,
            mini_mask,
            policy["rumodern_coverage"],
            frame["id1"],
            frame["id2"],
        )
        target = frame["target"].to_numpy(dtype=np.int8)
        category = frame["category"].astype(str).to_numpy()
        bge = frame["bge_probability"].to_numpy(dtype=np.float64)
        mini = frame["minilm_probability"].to_numpy(dtype=np.float64)
        ru = frame["rumodern_probability"].to_numpy(dtype=np.float64)
        mini_only = aggregate(
            bge, mini, ru, mini_mask, np.zeros(len(frame), dtype=bool),
            policy["mini_weight"], 0.0,
        )
        combined = aggregate(
            bge, mini, ru, mini_mask, ru_mask,
            policy["mini_weight"], policy["rumodern_weight"],
        )
        for method, score in (("bge", bge), ("mini_40", mini_only), ("mini_40_ru_5", combined)):
            rows.append(
                {"split": split, "method": method, "macro_ap": macro_ap(target, score, category)}
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = resolve(config["output_dir"])
    report_dir = resolve(config["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    frame, features, mini_priority = load_oof(config)
    ru_priority, ru_model, manifest = fit_router(config, frame, features, output_dir)
    oof_metrics, policy = evaluate_oof(config, frame, mini_priority, ru_priority)
    validation = evaluate_validation(config, policy, ru_model)
    validation.to_csv(report_dir / "split_metrics.csv", index=False)
    table = validation.pivot(index="method", columns="split", values="macro_ap").reset_index()
    table["mean_macro_ap"] = table[list(SPLITS)].mean(axis=1)
    table.to_csv(report_dir / "main_table.csv", index=False)
    manifest.update(
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "neural_models_retrained": False,
            "inference_direction": "AB only",
            "policy": policy,
            "oof_metrics": oof_metrics,
            "validation_metrics_are_evaluation_only": True,
        }
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (report_dir / "frozen_policy.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"oof": oof_metrics, "policy": policy, "validation": table.to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
