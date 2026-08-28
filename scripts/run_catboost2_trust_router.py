#!/usr/bin/env python
"""Train and evaluate a strictly nested CatBoost-2 trust/error router."""

from __future__ import annotations

import argparse
import hashlib
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

from src.catboost1_early_exit import (
    build_label_free_attribute_concept_map,
    category_balanced_weights,
    extract_pair_features,
)
from src.catboost2_trust_router import (
    assert_component_disjoint,
    calibration_table,
    cb1_feature_frame,
    confidence_features,
    crossfit_threshold_routing,
    decision_error,
    detection_metrics,
    full_oof_operating_point,
    split_routing_metrics,
    trust_feature_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/catboost2_trust_router.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rebuild-validation-features", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Fast integration test; metrics are invalid")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("catboost2_trust_router")
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


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def ordered_pair_digest(pairs: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(pairs[["id1", "id2"]], index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def catboost_config(raw: dict[str, Any], threads: int, iterations: int | None = None) -> dict[str, Any]:
    result = dict(raw)
    if iterations is not None:
        result["iterations"] = int(iterations)
    result.update({
        "loss_function": "Logloss",
        "task_type": "CPU",
        "thread_count": int(threads),
        "allow_writing_files": False,
    })
    return result


def train_classifier(
    frame: pd.DataFrame,
    target: np.ndarray,
    categorical: list[str],
    config: dict[str, Any],
    *,
    weights: np.ndarray | None = None,
):
    from catboost import CatBoostClassifier, Pool

    model = CatBoostClassifier(**config)
    pool = Pool(frame, label=target, weight=weights, cat_features=categorical)
    model.fit(pool)
    return model, pool


def predict_classifier(model, frame: pd.DataFrame, categorical: list[str]) -> np.ndarray:
    from catboost import Pool

    return model.predict_proba(Pool(frame, cat_features=categorical))[:, 1]


def select_smoke_rows(oof: pd.DataFrame, per_fold: int) -> np.ndarray:
    indices = np.concatenate([
        np.flatnonzero(oof["fold"].to_numpy() == fold)[:per_fold]
        for fold in sorted(oof["fold"].unique())
    ])
    return np.sort(indices)


def build_nested_cb1_scores(
    base_frame: pd.DataFrame,
    categorical: list[str],
    target: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    config: dict[str, Any],
    logger: logging.Logger,
) -> np.ndarray:
    """For outer H and inner J, score J with C1 trained without H and J."""

    unique_folds = sorted(np.unique(folds))
    nested = np.full((len(unique_folds), len(target)), np.nan, dtype=np.float64)
    for left_pos, left_fold in enumerate(unique_folds):
        for right_fold in unique_folds[left_pos + 1:]:
            fit = (folds != left_fold) & (folds != right_fold)
            logger.info(
                "Nested C1 excludes folds %d,%d: train=%s",
                left_fold, right_fold, f"{int(fit.sum()):,}",
            )
            weights = category_balanced_weights(categories[fit])
            local_config = dict(config)
            local_config["random_seed"] = int(config.get("random_seed", 2026)) + 100 + 10 * int(left_fold) + int(right_fold)
            model, _ = train_classifier(
                base_frame.loc[fit].reset_index(drop=True), target[fit], categorical,
                local_config, weights=weights,
            )
            left_rows = folds == left_fold
            right_rows = folds == right_fold
            # Outer right -> inner left; outer left -> inner right.
            nested[int(right_fold), left_rows] = predict_classifier(
                model, base_frame.loc[left_rows].reset_index(drop=True), categorical
            )
            nested[int(left_fold), right_rows] = predict_classifier(
                model, base_frame.loc[right_rows].reset_index(drop=True), categorical
            )
    for outer_fold in unique_folds:
        required = folds != outer_fold
        if not np.isfinite(nested[int(outer_fold), required]).all():
            raise AssertionError(f"Missing nested C1 scores for outer fold {outer_fold}")
        if np.isfinite(nested[int(outer_fold), ~required]).any():
            raise AssertionError("Outer-held rows unexpectedly received inner C1 scores")
    return nested


def load_or_build_validation_features(
    config: dict[str, Any],
    output_dir: Path,
    rebuild: bool,
    smoke: bool,
    logger: logging.Logger,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    cache = output_dir / "validation_feature_cache"
    cache.mkdir(parents=True, exist_ok=True)
    split_paths = {name: resolve(path) for name, path in config["validation_splits"].items()}
    items_path = resolve(config["items_path"])
    facts_path = resolve(config["accepted_facts_path"])
    identity = {
        "version": "frozen_c1a_legacy_semantics_v1",
        "items": file_identity(items_path),
        "accepted_facts": file_identity(facts_path),
        "splits": {name: file_identity(path) for name, path in split_paths.items()},
        "smoke": smoke,
        "smoke_rows": int(config["smoke_validation_rows"]),
    }
    identity_path = cache / "identity.json"
    paths_exist = all(
        (cache / f"{name}_pairs.parquet").exists() and (cache / f"{name}_base.parquet").exists()
        for name in split_paths
    )
    if (
        not rebuild and paths_exist and identity_path.exists()
        and json.loads(identity_path.read_text(encoding="utf-8")) == identity
    ):
        logger.info("Reusing frozen C1A validation feature cache")
        return {
            name: (
                pd.read_parquet(cache / f"{name}_pairs.parquet"),
                pd.read_parquet(cache / f"{name}_base.parquet"),
            )
            for name in split_paths
        }

    pair_parts: list[pd.DataFrame] = []
    split_bounds: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, path in split_paths.items():
        pairs = pd.read_parquet(path, columns=["id1", "id2", "target"])
        pairs["target"] = pairs["target"].astype(np.int8)
        if smoke:
            pairs = pairs.head(int(config["smoke_validation_rows"])).copy()
        pair_parts.append(pairs)
        split_bounds[name] = (cursor, cursor + len(pairs))
        cursor += len(pairs)
    combined = pd.concat(pair_parts, ignore_index=True)
    required_ids = pd.unique(combined[["id1", "id2"]].to_numpy().reshape(-1))
    items = pd.read_parquet(items_path, columns=["id", "name", "attributes", "category"])
    items = items.loc[items["id"].isin(required_ids)].reset_index(drop=True)
    concept_map, _ = build_label_free_attribute_concept_map(
        facts_path,
        min_support=int(config["attribute_alias_min_support"]),
        min_purity=float(config["attribute_alias_min_purity"]),
    )
    logger.info("Extracting frozen C1A features for %s validation pairs", f"{len(combined):,}")
    base, _ = extract_pair_features(
        combined, items, concept_map, {}, legacy_semantics=True
    )
    result: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for name, (start, end) in split_bounds.items():
        pairs = combined.iloc[start:end].reset_index(drop=True)
        split_base = base.iloc[start:end].reset_index(drop=True)
        pairs.to_parquet(cache / f"{name}_pairs.parquet", index=False)
        split_base.to_parquet(cache / f"{name}_base.parquet", index=False)
        result[name] = (pairs, split_base)
    identity_path.write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def write_results(
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    detection: dict[str, float],
    split_metrics: pd.DataFrame,
    output_path: Path,
    smoke: bool,
) -> None:
    lines = [
        "# CatBoost-2 trust-router results",
        "",
        "SMOKE MODE: metrics are not experiment results." if smoke else
        "Primary train numbers use strict nested C1 stacking and component-disjoint C2 OOF.",
        "",
        f"Error detection ROC-AUC: `{detection['roc_auc_error_detection']:.6f}`.",
        f"Error detection PR-AUC: `{detection['pr_auc_error_detection']:.6f}`.",
        f"C1 binary error prevalence: `{detection['error_prevalence']:.6f}`.",
        "",
        "## Main comparison",
        "",
        "| method | coverage@0.1% | coverage@0.2% | coverage@0.5% |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.coverage_at_0p1:.6f} | "
            f"{row.coverage_at_0p2:.6f} | {row.coverage_at_0p5:.6f} |"
        )
    lines.extend([
        "",
        "## Train OOF risk/coverage",
        "",
        "| method | risk | accepted | coverage | errors | empirical error | Wilson UCB | pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.risk_limit:.4f} | {row.accepted} | {row.coverage:.6f} | "
            f"{row.errors} | {row.empirical_error:.6f} | {row.error_ucb_95:.6f} | {row.passes_risk} |"
        )
    lines.extend([
        "",
        "Frozen negative-only C1 reference: `3.18% / 4.11% / 6.43%` at 0.1% / 0.2% / 0.5%.",
        "",
        "## Frozen-threshold IID / Hard / OOD",
        "",
        "| split | method | risk | accepted | coverage | errors | empirical error | neg exits | pos exits |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in split_metrics.itertuples(index=False):
        lines.append(
            f"| {row.split} | {row.method} | {row.risk_limit:.4f} | {row.accepted} | "
            f"{row.coverage:.6f} | {row.errors} | {row.empirical_error:.6f} | "
            f"{row.negative_accepted} | {row.positive_accepted} |"
        )
    lines.extend(["", "Thresholds were selected on train OOF only. Google Sheets was not modified.", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


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

    source_dir = resolve(config["catboost1_source_dir"])
    oof = pd.read_parquet(source_dir / "oof_predictions.parquet")
    base = pd.read_parquet(source_dir / "feature_cache" / "base_features.parquet")
    if len(oof) != len(base):
        raise ValueError("C1 OOF predictions and cached base features are misaligned")
    if not np.array_equal(oof["category"].astype(str), base["category"].astype(str)):
        raise ValueError("C1 category alignment mismatch")
    score_column = str(config["catboost1_score_column"])
    required = {"id1", "id2", "target", "category", "component_id", "fold", score_column}
    if missing := required - set(oof.columns):
        raise ValueError(f"Missing C1 OOF columns: {sorted(missing)}")
    if args.smoke:
        rows = select_smoke_rows(oof, int(config["smoke_pairs_per_fold"]))
        oof = oof.iloc[rows].reset_index(drop=True)
        base = base.iloc[rows].reset_index(drop=True)

    target = oof["target"].to_numpy(dtype=np.int8)
    categories = oof["category"].astype(str).to_numpy()
    folds = oof["fold"].to_numpy(dtype=np.int8)
    components = oof["component_id"].to_numpy()
    p_cb1_oof = oof[score_column].to_numpy(dtype=np.float64)
    assert_component_disjoint(folds, components)
    c1_frame, c1_categorical = cb1_feature_frame(base)
    logger.info("Train rows=%s C1 features=%d", f"{len(oof):,}", len(c1_frame.columns))

    from catboost import __version__ as catboost_version

    c1_iterations = int(config["smoke_iterations"]) if args.smoke else None
    c2_iterations = int(config["smoke_iterations"]) if args.smoke else None
    c1_config = catboost_config(config["catboost1"], int(config["threads"]), c1_iterations)
    c2_config = catboost_config(config["catboost2"], int(config["threads"]), c2_iterations)
    nested_p = build_nested_cb1_scores(
        c1_frame, c1_categorical, target, categories, folds, c1_config, logger
    )

    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    q_oof = np.full(len(oof), np.nan, dtype=np.float64)
    importance_parts: list[pd.DataFrame] = []
    for held_fold in sorted(np.unique(folds)):
        train = folds != held_fold
        held = ~train
        p_train = nested_p[int(held_fold), train]
        error_train = decision_error(p_train, target[train])
        train_frame, categorical = trust_feature_frame(
            base.loc[train].reset_index(drop=True), p_train
        )
        held_frame, held_categorical = trust_feature_frame(
            base.loc[held].reset_index(drop=True), p_cb1_oof[held]
        )
        if categorical != held_categorical or list(train_frame) != list(held_frame):
            raise AssertionError("C2 feature schema mismatch")
        local_config = dict(c2_config)
        local_config["random_seed"] = int(c2_config.get("random_seed", 2026)) + int(held_fold)
        logger.info(
            "Training C2 outer fold %d: rows=%s errors=%s (%.2f%%)",
            held_fold, f"{int(train.sum()):,}", f"{int(error_train.sum()):,}", 100 * error_train.mean(),
        )
        model, pool = train_classifier(
            train_frame, error_train, categorical, local_config
        )
        q_oof[held] = predict_classifier(model, held_frame, categorical)
        model.save_model(str(models_dir / f"catboost2_trust_fold{held_fold}.cbm"))
        importance_parts.append(pd.DataFrame({
            "fold": int(held_fold), "feature": train_frame.columns,
            "importance": model.get_feature_importance(pool),
        }))
    if not np.isfinite(q_oof).all():
        raise AssertionError("Missing C2 OOF predictions")

    target_error = decision_error(p_cb1_oof, target)
    cb1_uncertainty = confidence_features(p_cb1_oof)["min_p_cb1_one_minus_p"].to_numpy()
    detection = detection_metrics(target_error, q_oof)
    calibration = calibration_table(target_error, q_oof, int(config["calibration_bins"]))
    methods = {
        "CatBoost1_confidence": cb1_uncertainty,
        "CatBoost2_trust": q_oof,
    }
    risk_limits = [float(value) for value in config["risk_limits"]]
    crossfit_rows: list[dict[str, Any]] = []
    threshold_detail_parts: list[pd.DataFrame] = []
    full_threshold_rows: list[dict[str, Any]] = []
    accepted_columns: dict[str, np.ndarray] = {}
    frozen_thresholds: dict[tuple[str, float], float] = {}
    for method, error_score in methods.items():
        for risk in risk_limits:
            summary, details, accepted = crossfit_threshold_routing(
                error_score, target_error, folds, risk
            )
            crossfit_rows.append({"method": method, "risk_limit": risk, **summary})
            details.insert(0, "method", method)
            details.insert(1, "risk_limit", risk)
            threshold_detail_parts.append(details)
            accepted_columns[f"accepted_{method}_{risk:g}"] = accepted
            state = full_oof_operating_point(error_score, target_error, risk)
            threshold = float(state["threshold"])
            frozen_thresholds[(method, risk)] = threshold
            full_threshold_rows.append({"method": method, "risk_limit": risk, **state})

    crossfit_summary = pd.DataFrame(crossfit_rows)
    comparison = crossfit_summary.pivot(
        index="method", columns="risk_limit", values="verified_coverage"
    ).reset_index().rename(columns={
        0.001: "coverage_at_0p1",
        0.002: "coverage_at_0p2",
        0.005: "coverage_at_0p5",
    })
    reference = pd.DataFrame([{
        "method": "CatBoost1_frozen_negative_reference",
        "coverage_at_0p1": 0.0317899755110559,
        "coverage_at_0p2": 0.04108990475072472,
        "coverage_at_0p5": 0.06425168504152685,
    }])
    comparison = pd.concat([comparison, reference], ignore_index=True)
    full_thresholds = pd.DataFrame(full_threshold_rows)
    # Thresholds are frozen here, before validation data is loaded.
    crossfit_summary.to_csv(output_dir / "crossfit_risk_coverage.csv", index=False)
    comparison.to_csv(output_dir / "main_comparison.csv", index=False)
    pd.concat(threshold_detail_parts, ignore_index=True).to_csv(
        output_dir / "crossfit_threshold_details.csv", index=False
    )
    full_thresholds.to_csv(output_dir / "frozen_train_oof_thresholds.csv", index=False)
    calibration.to_csv(output_dir / "q_error_calibration.csv", index=False)
    pd.concat(importance_parts, ignore_index=True).to_csv(
        output_dir / "feature_importance_by_fold.csv", index=False
    )
    oof_output = oof[["id1", "id2", "target", "category", "component_id", "fold"]].copy()
    oof_output["p_cb1_oof"] = p_cb1_oof
    oof_output["cb1_prediction"] = (p_cb1_oof >= 0.5).astype(np.int8)
    oof_output["target_error"] = target_error
    oof_output["cb1_uncertainty"] = cb1_uncertainty
    oof_output["q_error_oof"] = q_oof
    for column, values in accepted_columns.items():
        oof_output[column] = values
    oof_output.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    (output_dir / "detection_metrics.json").write_text(
        json.dumps(detection, indent=2), encoding="utf-8"
    )

    # Deployment models: full C1 on train, then C2 on genuine C1 OOF errors.
    logger.info("Training full C1 deployment model")
    full_c1_weights = category_balanced_weights(categories)
    full_c1, _ = train_classifier(
        c1_frame, target, c1_categorical, c1_config, weights=full_c1_weights
    )
    full_c1.save_model(str(models_dir / "catboost1_full.cbm"))
    logger.info("Training full C2 deployment model")
    full_c2_frame, c2_categorical = trust_feature_frame(base, p_cb1_oof)
    full_c2, full_c2_pool = train_classifier(
        full_c2_frame, target_error, c2_categorical, c2_config
    )
    full_c2.save_model(str(models_dir / "catboost2_trust_full.cbm"))
    pd.DataFrame({
        "feature": full_c2_frame.columns,
        "importance": full_c2.get_feature_importance(full_c2_pool),
    }).to_csv(output_dir / "full_model_feature_importance.csv", index=False)

    validation = load_or_build_validation_features(
        config, output_dir, args.rebuild_validation_features, args.smoke, logger
    )
    split_metric_rows: list[dict[str, Any]] = []
    for split, (pairs, split_base) in validation.items():
        split_target = pairs["target"].to_numpy(dtype=np.int8)
        split_c1_frame, split_c1_categorical = cb1_feature_frame(split_base)
        p_split = predict_classifier(full_c1, split_c1_frame, split_c1_categorical)
        split_c2_frame, split_c2_categorical = trust_feature_frame(split_base, p_split)
        q_split = predict_classifier(full_c2, split_c2_frame, split_c2_categorical)
        uncertainty_split = confidence_features(p_split)["min_p_cb1_one_minus_p"].to_numpy()
        split_output = pairs.copy()
        split_output["p_cb1"] = p_split
        split_output["cb1_prediction"] = (p_split >= 0.5).astype(np.int8)
        split_output["target_error"] = decision_error(p_split, split_target)
        split_output["cb1_uncertainty"] = uncertainty_split
        split_output["q_error"] = q_split
        split_scores = {
            "CatBoost1_confidence": uncertainty_split,
            "CatBoost2_trust": q_split,
        }
        for method, error_score in split_scores.items():
            for risk in risk_limits:
                threshold = frozen_thresholds[(method, risk)]
                split_metric_rows.append(split_routing_metrics(
                    split, method, risk, threshold, error_score, p_split, split_target
                ))
                split_output[f"accepted_{method}_{risk:g}"] = error_score < threshold
        split_output.to_parquet(output_dir / f"{split}_predictions.parquet", index=False)
    split_metrics = pd.DataFrame(split_metric_rows)
    split_metrics.to_csv(output_dir / "iid_hard_ood_routing.csv", index=False)
    write_results(
        crossfit_summary, comparison, detection, split_metrics,
        output_dir / "RESULTS.md", args.smoke,
    )

    manifest = {
        "experiment": "catboost2_trust_router_v1",
        "smoke": args.smoke,
        "human_train_only_for_training_and_thresholds": True,
        "strict_nested_stacking": True,
        "nested_c1_models": 10,
        "outer_c2_models": int(len(np.unique(folds))),
        "cb1_changed": False,
        "rules_used": False,
        "neural_predictions_used": False,
        "thresholds_selected_before_validation_load": True,
        "train_rows": len(oof),
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
