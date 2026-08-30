#!/usr/bin/env python
"""Compare global and separate NEG/POS CatBoost-2 trust routers."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_catboost2_trust_router import (
    build_nested_cb1_scores,
    catboost_config,
    predict_classifier,
    select_smoke_rows,
    train_classifier,
)
from src.catboost2_separate_routers import (
    apply_separate_thresholds,
    crossfit_separate_routing,
    routing_summary,
    select_separate_thresholds,
)
from src.catboost2_trust_router import (
    cb1_feature_frame,
    decision_error,
    detection_metrics,
    trust_feature_frame,
)
from src.catboost1_early_exit import category_balanced_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/catboost2_separate_routers.json"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="Integration check; metrics are invalid")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("catboost2_separate_routers")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "run.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def train_route_model(
    frame: pd.DataFrame,
    categorical: list[str],
    target_error: np.ndarray,
    route_mask: np.ndarray,
    config: dict[str, Any],
    route: str,
    fold: int | str,
):
    route_target = target_error[route_mask]
    if len(np.unique(route_target)) != 2:
        raise ValueError(f"{route} fold {fold} does not contain both error classes")
    return train_classifier(
        frame.loc[route_mask].reset_index(drop=True), route_target, categorical, config
    )


def method_summary_rows(
    previous_oof: pd.DataFrame,
    separate_masks: dict[float, np.ndarray],
    risk_limits: list[float],
) -> tuple[pd.DataFrame, dict[tuple[str, float], np.ndarray]]:
    target_error = previous_oof["target_error"].to_numpy(dtype=np.int8)
    predicted = previous_oof["cb1_prediction"].to_numpy(dtype=np.int8)
    masks: dict[tuple[str, float], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for risk in risk_limits:
        for method, prefix in (
            ("CatBoost1_confidence", "accepted_CatBoost1_confidence"),
            ("CatBoost2_global", "accepted_CatBoost2_trust"),
        ):
            mask = previous_oof[f"{prefix}_{risk:g}"].to_numpy(dtype=bool)
            masks[(method, risk)] = mask
            rows.append({
                "method": method, "risk_limit": risk,
                **routing_summary(mask, target_error, predicted, risk),
            })
        mask = separate_masks[risk]
        masks[("CatBoost2_separate_NEG_POS", risk)] = mask
        rows.append({
            "method": "CatBoost2_separate_NEG_POS", "risk_limit": risk,
            **routing_summary(mask, target_error, predicted, risk),
        })
    return pd.DataFrame(rows), masks


def write_results(
    comparison: pd.DataFrame,
    detection: pd.DataFrame,
    validation: pd.DataFrame,
    path: Path,
    smoke: bool,
) -> None:
    lines = [
        "# Separate CatBoost-2 NEG/POS router results",
        "",
        "SMOKE MODE: metrics are invalid." if smoke else
        "Primary numbers use strict nested C1 stacking and component-disjoint C2 OOF.",
        "",
        "## Train OOF comparison",
        "",
        "| method | risk | total cov | neg cov | pos cov | errors | Wilson UCB | pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.risk_limit:.4f} | {row.coverage:.6f} | "
            f"{row.negative_coverage:.6f} | {row.positive_coverage:.6f} | "
            f"{row.errors} | {row.error_ucb_95:.6f} | {row.passes_risk} |"
        )
    lines.extend([
        "",
        "## Error detection",
        "",
        "| router | examples | error prevalence | ROC-AUC | PR-AUC |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for row in detection.itertuples(index=False):
        lines.append(
            f"| {row.router} | {row.examples} | {row.error_prevalence:.6f} | "
            f"{row.roc_auc_error_detection:.6f} | {row.pr_auc_error_detection:.6f} |"
        )
    lines.extend([
        "",
        "## Frozen thresholds on IID / Hard / OOD",
        "",
        "| split | method | risk | total cov | neg cov | pos cov | total err | neg err | pos err | UCB |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in validation.itertuples(index=False):
        lines.append(
            f"| {row.split} | {row.method} | {row.risk_limit:.4f} | {row.coverage:.6f} | "
            f"{row.negative_coverage:.6f} | {row.positive_coverage:.6f} | {row.errors} | "
            f"{row.negative_errors} | {row.positive_errors} | {row.error_ucb_95:.6f} |"
        )
    lines.extend(["", "Thresholds were selected on train OOF only. Google Sheets was not modified.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or resolve(config["output_dir"])
    if args.smoke and args.output_dir is None:
        output_dir = output_dir / "_smoke"
    logger = configure_logging(output_dir)
    started = time.perf_counter()
    if args.smoke:
        logger.warning("SMOKE MODE: metrics are invalid")

    experiment1_dir = resolve(config["experiment1_dir"])
    c1_source_dir = resolve(config["catboost1_source_dir"])
    previous_oof = pd.read_parquet(experiment1_dir / "oof_predictions.parquet")
    source_oof = pd.read_parquet(c1_source_dir / "oof_predictions.parquet")
    base = pd.read_parquet(c1_source_dir / "feature_cache" / "base_features.parquet")
    if not (len(previous_oof) == len(source_oof) == len(base)):
        raise ValueError("Experiment 1, C1 OOF and base cache are not aligned")
    if not np.array_equal(previous_oof[["id1", "id2"]], source_oof[["id1", "id2"]]):
        raise ValueError("Experiment 1 pair order differs from C1 source")
    if args.smoke:
        rows = select_smoke_rows(previous_oof, int(config["smoke_pairs_per_fold"]))
        previous_oof = previous_oof.iloc[rows].reset_index(drop=True)
        source_oof = source_oof.iloc[rows].reset_index(drop=True)
        base = base.iloc[rows].reset_index(drop=True)

    target = previous_oof["target"].to_numpy(dtype=np.int8)
    p_cb1_oof = previous_oof["p_cb1_oof"].to_numpy(dtype=np.float64)
    target_error = decision_error(p_cb1_oof, target)
    predicted = (p_cb1_oof >= 0.5).astype(np.int8)
    folds = previous_oof["fold"].to_numpy(dtype=np.int8)
    categories = previous_oof["category"].astype(str).to_numpy()
    c1_frame, c1_categorical = cb1_feature_frame(base)

    experiment1_config = json.loads(resolve(config["experiment1_config"]).read_text(encoding="utf-8"))
    iterations = int(config["smoke_iterations"]) if args.smoke else None
    c1_config = catboost_config(
        experiment1_config["catboost1"], int(config["threads"]), iterations
    )
    router_config = catboost_config(
        experiment1_config["catboost2"], int(config["threads"]), iterations
    )
    logger.info("Building strict nested C1 scores for %s rows", f"{len(base):,}")
    nested_p = build_nested_cb1_scores(
        c1_frame, c1_categorical, target, categories, folds, c1_config, logger
    )
    np.save(output_dir / "nested_cb1_scores.npy", nested_p.astype(np.float32))

    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    q_separate_oof = np.full(len(base), np.nan, dtype=np.float64)
    importance_parts: list[pd.DataFrame] = []
    for held_fold in sorted(np.unique(folds)):
        train = folds != held_fold
        held = ~train
        p_train = nested_p[int(held_fold), train]
        error_train = decision_error(p_train, target[train])
        predicted_train = (p_train >= 0.5).astype(np.int8)
        train_frame, categorical = trust_feature_frame(
            base.loc[train].reset_index(drop=True), p_train
        )
        held_frame, held_categorical = trust_feature_frame(
            base.loc[held].reset_index(drop=True), p_cb1_oof[held]
        )
        held_predicted = predicted[held]
        if categorical != held_categorical:
            raise AssertionError("Categorical schema mismatch")
        for route, route_class in (("NEG", 0), ("POS", 1)):
            route_train = predicted_train == route_class
            route_held = held_predicted == route_class
            local_config = dict(router_config)
            local_config["random_seed"] = (
                int(router_config.get("random_seed", 3026))
                + int(held_fold) + (100 if route == "POS" else 0)
            )
            logger.info(
                "Training Router_%s fold %d: rows=%s errors=%s (%.2f%%)",
                route, held_fold, f"{int(route_train.sum()):,}",
                f"{int(error_train[route_train].sum()):,}",
                100 * error_train[route_train].mean(),
            )
            model, pool = train_route_model(
                train_frame, categorical, error_train, route_train,
                local_config, route, int(held_fold),
            )
            held_positions = np.flatnonzero(held)
            q_separate_oof[held_positions[route_held]] = predict_classifier(
                model, held_frame.loc[route_held].reset_index(drop=True), categorical
            )
            model.save_model(str(models_dir / f"router_{route.lower()}_fold{held_fold}.cbm"))
            importance_parts.append(pd.DataFrame({
                "router": route, "fold": int(held_fold), "feature": train_frame.columns,
                "importance": model.get_feature_importance(pool),
            }))
    if not np.isfinite(q_separate_oof).all():
        raise AssertionError("Missing separate-router OOF predictions")

    risk_limits = [float(value) for value in config["risk_limits"]]
    separate_masks: dict[float, np.ndarray] = {}
    threshold_details: list[pd.DataFrame] = []
    frozen_rows: list[dict[str, Any]] = []
    frozen_separate: dict[float, dict[str, Any]] = {}
    for risk in risk_limits:
        summary, details, accepted = crossfit_separate_routing(
            q_separate_oof, target_error, predicted, folds, risk
        )
        separate_masks[risk] = accepted
        details.insert(0, "risk_limit", risk)
        threshold_details.append(details)
        state = select_separate_thresholds(q_separate_oof, target_error, predicted, risk)
        frozen_separate[risk] = state
        frozen_rows.append({"risk_limit": risk, **state})

    comparison, method_masks = method_summary_rows(previous_oof, separate_masks, risk_limits)
    comparison.to_csv(output_dir / "crossfit_comparison.csv", index=False)
    pd.concat(threshold_details, ignore_index=True).to_csv(
        output_dir / "crossfit_separate_threshold_details.csv", index=False
    )
    pd.DataFrame(frozen_rows).to_csv(output_dir / "frozen_separate_thresholds.csv", index=False)
    pd.concat(importance_parts, ignore_index=True).to_csv(
        output_dir / "feature_importance_by_router_fold.csv", index=False
    )

    detection_rows: list[dict[str, Any]] = []
    for name, mask, scores in (
        ("GLOBAL", np.ones(len(predicted), dtype=bool), previous_oof["q_error_oof"].to_numpy()),
        ("NEG", predicted == 0, q_separate_oof),
        ("POS", predicted == 1, q_separate_oof),
    ):
        metrics = detection_metrics(target_error[mask], np.asarray(scores)[mask])
        detection_rows.append({"router": name, "examples": int(mask.sum()), **metrics})
    detection = pd.DataFrame(detection_rows)
    detection.to_csv(output_dir / "error_detection_metrics.csv", index=False)

    oof_output = previous_oof.copy()
    oof_output["q_error_separate_oof"] = q_separate_oof
    for (method, risk), mask in method_masks.items():
        oof_output[f"accepted_{method}_{risk:g}"] = mask
    oof_output.to_parquet(output_dir / "oof_predictions.parquet", index=False)

    # Deployment routers use genuine original C1 OOF scores, exactly as Experiment 1.
    full_frame, categorical = trust_feature_frame(base, p_cb1_oof)
    full_models: dict[int, Any] = {}
    full_importance: list[pd.DataFrame] = []
    for route, route_class in (("NEG", 0), ("POS", 1)):
        route_mask = predicted == route_class
        local_config = dict(router_config)
        local_config["random_seed"] = int(router_config.get("random_seed", 3026)) + (100 if route == "POS" else 0)
        logger.info("Training full Router_%s on %s rows", route, f"{int(route_mask.sum()):,}")
        model, pool = train_route_model(
            full_frame, categorical, target_error, route_mask, local_config, route, "full"
        )
        model.save_model(str(models_dir / f"router_{route.lower()}_full.cbm"))
        full_models[route_class] = model
        full_importance.append(pd.DataFrame({
            "router": route, "feature": full_frame.columns,
            "importance": model.get_feature_importance(pool),
        }))
    pd.concat(full_importance, ignore_index=True).to_csv(
        output_dir / "full_model_feature_importance.csv", index=False
    )

    # Frozen global thresholds from Experiment 1; no validation labels influence selection.
    previous_thresholds = pd.read_csv(experiment1_dir / "frozen_train_oof_thresholds.csv")
    single_thresholds = {
        (str(row.method).replace("CatBoost2_trust", "CatBoost2_global"), float(row.risk_limit)):
        float(row.threshold)
        for row in previous_thresholds.itertuples(index=False)
    }
    validation_rows: list[dict[str, Any]] = []
    for split in config["validation_splits"]:
        previous_split = pd.read_parquet(experiment1_dir / f"{split}_predictions.parquet")
        split_base = pd.read_parquet(
            experiment1_dir / "validation_feature_cache" / f"{split}_base.parquet"
        )
        if args.smoke:
            limit = int(config["smoke_validation_rows"])
            previous_split = previous_split.head(limit).reset_index(drop=True)
            split_base = split_base.head(limit).reset_index(drop=True)
        p_split = previous_split["p_cb1"].to_numpy(dtype=np.float64)
        split_target = previous_split["target"].to_numpy(dtype=np.int8)
        split_error = decision_error(p_split, split_target)
        split_predicted = (p_split >= 0.5).astype(np.int8)
        split_frame, split_categorical = trust_feature_frame(split_base, p_split)
        q_separate = np.full(len(previous_split), np.nan, dtype=np.float64)
        for route_class in (0, 1):
            mask = split_predicted == route_class
            if mask.any():
                q_separate[mask] = predict_classifier(
                    full_models[route_class], split_frame.loc[mask].reset_index(drop=True),
                    split_categorical,
                )
        split_output = previous_split.copy()
        split_output["q_error_separate"] = q_separate
        for risk in risk_limits:
            for method, score_column in (
                ("CatBoost1_confidence", "cb1_uncertainty"),
                ("CatBoost2_global", "q_error"),
            ):
                threshold = single_thresholds[(method, risk)]
                accepted = previous_split[score_column].to_numpy() < threshold
                validation_rows.append({
                    "split": split, "method": method, "risk_limit": risk,
                    **routing_summary(accepted, split_error, split_predicted, risk),
                })
                split_output[f"accepted_{method}_{risk:g}"] = accepted
            state = frozen_separate[risk]
            accepted = apply_separate_thresholds(
                q_separate, split_predicted,
                float(state["threshold_neg"]), float(state["threshold_pos"]),
            )
            validation_rows.append({
                "split": split, "method": "CatBoost2_separate_NEG_POS", "risk_limit": risk,
                **routing_summary(accepted, split_error, split_predicted, risk),
            })
            split_output[f"accepted_CatBoost2_separate_NEG_POS_{risk:g}"] = accepted
        split_output.to_parquet(output_dir / f"{split}_predictions.parquet", index=False)
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(output_dir / "iid_hard_ood_comparison.csv", index=False)
    write_results(comparison, detection, validation, output_dir / "RESULTS.md", args.smoke)

    from catboost import __version__ as catboost_version

    manifest = {
        "experiment": "catboost2_separate_routers_v1",
        "smoke": args.smoke,
        "experiment1_reused": str(experiment1_dir.resolve()),
        "strict_nested_stacking": True,
        "nested_c1_models": 10,
        "outer_router_models": 10,
        "separate_thresholds_jointly_optimized": True,
        "combined_wilson_constraint": True,
        "rules_used": False,
        "neural_predictions_used": False,
        "validation_used_for_threshold_selection": False,
        "catboost_version": catboost_version,
        "python": platform.python_version(),
        "elapsed_seconds": time.perf_counter() - started,
        "google_sheets_written": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "COMPLETED").write_text("ok\n", encoding="utf-8")
    logger.info("Completed in %.1f minutes", manifest["elapsed_seconds"] / 60)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
