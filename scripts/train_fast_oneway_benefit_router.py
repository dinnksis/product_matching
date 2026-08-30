#!/usr/bin/env python3
"""Train compact loss-benefit routers for one-direction BGE/MiniLM inference."""

from __future__ import annotations

import argparse
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
from src.deterministic_specialist_routing import top_budget_mask, uncertainty_abs
from src.fast_benefit_router import VARIANT_COLUMNS, cached_feature_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/fast_oneway_benefit_router.json",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def checked_prediction(path: Path, model: str, oof: bool) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"id1", "id2", "target", "score_ab"}
    required.update(
        {"fold", "component_id", "oof_row_index"} if oof else {"category_1"}
    )
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{model} predictions are missing: {sorted(missing)}")
    order = "oof_row_index" if oof else "pair_index"
    if order in frame:
        frame = frame.sort_values(order, kind="stable")
    return frame.reset_index(drop=True)


def align_predictions(bge: pd.DataFrame, mini: pd.DataFrame, label: str) -> None:
    columns = ["id1", "id2", "target"]
    if not bge[columns].equals(mini[columns]):
        raise ValueError(f"BGE/MiniLM prediction alignment differs on {label}")


def probability_logit(values: pd.Series | np.ndarray) -> np.ndarray:
    probability = np.asarray(values, dtype=np.float64)
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


def load_oof(config: dict[str, Any], smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    bge = checked_prediction(resolve(config["oof_predictions"]["bge"]), "bge", True)
    mini = checked_prediction(resolve(config["oof_predictions"]["minilm"]), "minilm", True)
    align_predictions(bge, mini, "OOF")
    pairs = pd.read_parquet(resolve(config["human_train_pairs_path"]))
    cheap = pd.read_parquet(resolve(config["train_feature_cache_path"]))
    if len(pairs) != len(bge) or len(cheap) != len(bge):
        raise ValueError("OOF, human pairs, and cheap features differ in length")
    if not pairs[["id1", "id2", "target"]].equals(bge[["id1", "id2", "target"]]):
        raise ValueError("OOF predictions are not aligned with human train")
    frame = bge[
        ["id1", "id2", "target", "category", "component_id", "fold", "score_ab"]
    ].rename(columns={"score_ab": "bge_probability"})
    frame["bge_raw_logit"] = probability_logit(frame["bge_probability"])
    frame["minilm_probability"] = mini["score_ab"].to_numpy(dtype=np.float32)
    if smoke:
        selected_folds = sorted(frame["fold"].unique())[:2]
        chosen = np.flatnonzero(frame["fold"].isin(selected_folds).to_numpy())[:6000]
        frame = frame.iloc[chosen].reset_index(drop=True)
        cheap = cheap.iloc[chosen].reset_index(drop=True)
    return frame, cheap


def fit_variants(
    config: dict[str, Any], frame: pd.DataFrame, cheap: pd.DataFrame, output_dir: Path, smoke: bool
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]:
    from catboost import CatBoostClassifier, Pool

    _, target = benefit_targets(
        frame["target"],
        frame["bge_probability"],
        frame["minilm_probability"],
        classification_margin=float(config["classification_margin_logloss"]),
    )
    folds = frame["fold"].to_numpy(dtype=np.int8)
    predictions = frame[
        ["id1", "id2", "target", "category", "fold", "bge_probability", "minilm_probability"]
    ].copy()
    models: dict[str, Any] = {}
    feature_frames: dict[str, pd.DataFrame] = {}
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for variant in config["variants"]:
        features = cached_feature_frame(
            cheap,
            frame["bge_probability"],
            frame["bge_raw_logit"],
            variant,
        )
        feature_frames[variant] = features
        categorical = ["category"]
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
            raise RuntimeError(f"Incomplete OOF for {variant}")
        predictions[f"benefit_{variant}"] = oof.astype(np.float32)
        model = CatBoostClassifier(**parameters)
        model.fit(Pool(features, target, cat_features=categorical))
        model.save_model(model_dir / f"router_{variant}.cbm")
        models[variant] = model
    predictions.to_parquet(output_dir / "router_oof_predictions.parquet", index=False)
    return predictions, models, feature_frames


def blend(frame: pd.DataFrame, mask: np.ndarray, weight: float) -> np.ndarray:
    bge = frame["bge_probability"].to_numpy(dtype=np.float64)
    mini = frame["minilm_probability"].to_numpy(dtype=np.float64)
    result = bge.copy()
    result[mask] = (1.0 - weight) * bge[mask] + weight * mini[mask]
    return result


def select_policies(config: dict[str, Any], frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    target = frame["target"].to_numpy(dtype=np.int8)
    category = frame["category"].astype(str).to_numpy()
    bge = frame["bge_probability"].to_numpy(dtype=np.float64)
    bge_ap = macro_ap(target, bge, category)
    full_candidates = []
    all_mask = np.ones(len(frame), dtype=bool)
    for weight in config["blend_weights"]:
        score = blend(frame, all_mask, float(weight))
        full_candidates.append((macro_ap(target, score, category), float(weight)))
    full_ap, full_weight = max(full_candidates, key=lambda item: (item[0], -item[1]))

    rows = []
    priorities = {
        **{
            variant: frame[f"benefit_{variant}"].to_numpy(dtype=np.float64)
            for variant in config["variants"]
        },
        "uncertainty": uncertainty_abs(bge),
    }
    for method, priority in priorities.items():
        for coverage in config["coverages"]:
            mask = top_budget_mask(priority, float(coverage), frame["id1"], frame["id2"])
            candidates = []
            for weight in config["blend_weights"]:
                score = blend(frame, mask, float(weight))
                candidates.append((macro_ap(target, score, category), float(weight)))
            ap, weight = max(candidates, key=lambda item: (item[0], -item[1]))
            rows.append(
                {
                    "method": method,
                    "coverage": float(coverage),
                    "mini_weight": weight,
                    "oof_macro_ap": ap,
                    "oof_delta_vs_bge": ap - bge_ap,
                    "oof_delta_vs_full_oneway": ap - full_ap,
                }
            )
    policies = pd.DataFrame(rows)
    chosen = (
        policies.sort_values(["coverage", "oof_macro_ap", "method"], ascending=[True, False, True])
        .groupby("coverage", as_index=False)
        .first()
    )
    return chosen, {
        "bge_oneway_oof_ap": bge_ap,
        "full_oneway_oof_ap": full_ap,
        "full_oneway_weight": full_weight,
        "all_policy_candidates": policies.to_dict("records"),
    }


def load_validation(config: dict[str, Any], split: str, filename: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    bge = checked_prediction(
        resolve(config["predictions_root"]) / "preds_bge" / filename, "bge", False
    )
    mini = checked_prediction(
        resolve(config["predictions_root"]) / "preds_minilm" / filename, "minilm", False
    )
    align_predictions(bge, mini, split)
    pairs = pd.read_parquet(
        resolve(config["validation_feature_cache_dir"]) / f"{split}_pairs.parquet"
    )
    cheap = pd.read_parquet(
        resolve(config["validation_feature_cache_dir"]) / f"{split}_base.parquet"
    )
    if not pairs[["id1", "id2"]].reset_index(drop=True).equals(
        bge[["id1", "id2"]].reset_index(drop=True)
    ) or not np.array_equal(
        pairs["target"].to_numpy(dtype=np.float64),
        bge["target"].to_numpy(dtype=np.float64),
    ):
        raise ValueError(f"Validation feature cache is not aligned on {split}")
    frame = bge[["id1", "id2", "target", "category_1", "score_ab"]].rename(
        columns={
            "category_1": "category",
            "score_ab": "bge_probability",
        }
    )
    frame["bge_raw_logit"] = probability_logit(frame["bge_probability"])
    frame["minilm_probability"] = mini["score_ab"].to_numpy(dtype=np.float32)
    return frame, cheap


def evaluate_validation(
    config: dict[str, Any], policies: pd.DataFrame, anchor: dict, models: dict[str, Any]
) -> pd.DataFrame:
    from catboost import Pool

    rows = []
    for split, filename in SPLITS.items():
        frame, cheap = load_validation(config, split, filename)
        for variant, model in models.items():
            features = cached_feature_frame(
                cheap,
                frame["bge_probability"],
                frame["bge_raw_logit"],
                variant,
            )
            frame[f"benefit_{variant}"] = model.predict_proba(
                Pool(features, cat_features=["category"])
            )[:, 1]
        target = frame["target"].to_numpy(dtype=np.int8)
        category = frame["category"].astype(str).to_numpy()
        bge = frame["bge_probability"].to_numpy(dtype=np.float64)
        full = (1.0 - anchor["full_oneway_weight"]) * bge + anchor[
            "full_oneway_weight"
        ] * frame["minilm_probability"].to_numpy(dtype=np.float64)
        for method, coverage, score in (
            ("bge_oneway", 0.0, bge),
            ("full_oneway", 1.0, full),
        ):
            rows.append(
                {
                    "split": split,
                    "method": method,
                    "coverage": coverage,
                    "macro_ap": macro_ap(target, score, category),
                }
            )
        for policy in policies.itertuples(index=False):
            priority = (
                uncertainty_abs(bge)
                if policy.method == "uncertainty"
                else frame[f"benefit_{policy.method}"].to_numpy(dtype=np.float64)
            )
            mask = top_budget_mask(
                priority, policy.coverage, frame["id1"], frame["id2"]
            )
            score = blend(frame, mask, policy.mini_weight)
            rows.append(
                {
                    "split": split,
                    "method": policy.method,
                    "coverage": policy.coverage,
                    "macro_ap": macro_ap(target, score, category),
                }
            )
    return pd.DataFrame(rows)


def main_table(policies: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, coverage in validation[["method", "coverage"]].drop_duplicates().itertuples(index=False):
        part = validation.loc[
            validation["method"].eq(method) & np.isclose(validation["coverage"], coverage)
        ].set_index("split")
        values = {split: float(part.loc[split, "macro_ap"]) for split in SPLITS}
        policy = policies.loc[
            policies["method"].eq(method) & np.isclose(policies["coverage"], coverage)
        ]
        rows.append(
            {
                "method": method,
                "coverage": coverage,
                "mini_weight": float(policy["mini_weight"].iloc[0]) if not policy.empty else np.nan,
                "oof_macro_ap": float(policy["oof_macro_ap"].iloc[0]) if not policy.empty else np.nan,
                "iid_macro_ap": values["iid"],
                "hard_macro_ap": values["hard"],
                "ood_macro_ap": values["ood"],
                "mean_macro_ap": float(np.mean(list(values.values()))),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifact_dir = resolve(config["output_dir"])
    report_dir = resolve(config["report_dir"])
    if args.smoke:
        artifact_dir = artifact_dir.with_name(artifact_dir.name + "_smoke")
        report_dir = report_dir.with_name(report_dir.name + "_smoke")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    frame, cheap = load_oof(config, args.smoke)
    oof, models, _ = fit_variants(config, frame, cheap, artifact_dir, args.smoke)
    policies, anchor = select_policies(config, oof)
    policies.to_csv(report_dir / "oof_selected_policies.csv", index=False)
    (report_dir / "frozen_policy.json").write_text(
        json.dumps({"anchor": anchor, "policies": policies.to_dict("records")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    validation = evaluate_validation(config, policies, anchor, models)
    validation.to_csv(report_dir / "split_metrics.csv", index=False)
    table = main_table(policies, validation)
    table.to_csv(report_dir / "main_table.csv", index=False)
    manifest = {
        "status": "complete",
        "experiment": "fast_oneway_benefit_router_v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "neural_models_retrained": False,
        "inference_direction": "AB only",
        "benefit_target": "logloss(BGE_AB) - logloss(MiniLM_AB) > margin",
        "oof_parameters_frozen_before_validation": True,
        "variants": {
            variant: {
                "feature_columns": list(VARIANT_COLUMNS[variant]),
                "model": f"models/router_{variant}.cbm",
            }
            for variant in config["variants"]
        },
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "table": table.to_dict("records")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
