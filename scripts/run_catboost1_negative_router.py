#!/usr/bin/env python
"""CatBoost-1.1: conservative negative-only product-matching router."""

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
from scipy import sparse
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.catboost1_early_exit import (
    REGIME_BY_CATEGORY,
    best_side_state,
    build_label_free_attribute_concept_map,
    category_balanced_weights,
    category_rule_matrix,
    crossfit_fold_rule_evidence,
    extract_pair_features,
    load_label_free_rule_registry,
    rule_family_masks,
    rule_lists_to_csr,
    semantic_family,
    threshold_states,
    wilson_upper,
)
from src.validation_splits import stable_component_ids


VARIANTS: dict[str, dict[str, Any]] = {
    "C1A_category_no_rules": {
        "rules": False, "category_aware": False, "regime": False, "positive_weight": 1.0,
    },
    "C1B_category_clean_rules": {
        "rules": True, "category_aware": False, "regime": False, "positive_weight": 1.0,
    },
    "C1C_category_clean_rules_posw3": {
        "rules": True, "category_aware": False, "regime": False, "positive_weight": 3.0,
    },
    "C1D_regime_clean_rules_posw3": {
        "rules": True, "category_aware": True, "regime": True, "positive_weight": 3.0,
    },
}

GROUP_EVIDENCE_COLUMNS = (
    "rule_evidence_fired",
    "rule_positive_evidence_sum",
    "rule_negative_evidence_sum",
    "rule_positive_evidence_mean",
    "rule_negative_evidence_mean",
    "rule_evidence_disagreement",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/catboost1_negative_router.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--smoke", action="store_true", help="Fast integration check, not a valid experiment")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("catboost1_negative_router")
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


def source_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def ordered_pair_digest(pairs: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(pairs[["id1", "id2"]], index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def make_smoke_subset(pairs: pd.DataFrame, limit: int) -> pd.DataFrame:
    components = stable_component_ids(pairs)
    helper = pairs.assign(_component=components)
    sizes = helper.groupby("_component").size().sort_index()
    chosen: list[int] = []
    total = 0
    for component, size in sizes.items():
        chosen.append(component)
        total += int(size)
        if total >= limit:
            break
    return helper.loc[helper._component.isin(chosen)].drop(columns="_component").reset_index(drop=True)


def assign_group_folds(
    target: np.ndarray,
    categories: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    strata = pd.Series(categories).astype(str) + "||" + pd.Series(target).astype(int).astype(str)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result = np.full(len(target), -1, dtype=np.int8)
    dummy = np.zeros(len(target), dtype=np.int8)
    for fold, (_, valid) in enumerate(splitter.split(dummy, strata, groups=groups)):
        result[valid] = fold
    if np.any(result < 0):
        raise AssertionError("Not every row received a fold")
    return result


def add_raw_rule_features(
    base: pd.DataFrame,
    matrix: sparse.csr_matrix,
    definitions: pd.DataFrame,
) -> pd.DataFrame:
    result = base.copy()
    result["rule_raw_candidate_count"] = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
    for relation in sorted(definitions["relation"].unique()):
        columns = np.flatnonzero(definitions["relation"].to_numpy() == relation)
        result[f"rule_raw_relation_{relation}_count"] = np.asarray(matrix[:, columns].sum(axis=1)).ravel().astype(np.float32)
    for family, columns in rule_family_masks(definitions).items():
        result[f"rule_raw_family_{family}_count"] = np.asarray(matrix[:, columns].sum(axis=1)).ravel().astype(np.float32)
    return result


def load_or_build_features(
    config: dict[str, Any],
    output_dir: Path,
    pairs: pd.DataFrame,
    items: pd.DataFrame,
    rebuild: bool,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, sparse.csr_matrix, sparse.csr_matrix, pd.DataFrame]:
    cache = output_dir / "feature_cache"
    cache.mkdir(parents=True, exist_ok=True)
    paths = {
        "base": cache / "base_features.parquet",
        "global": cache / "clean_global_rules.npz",
        "category": cache / "clean_category_rules.npz",
        "definitions": cache / "clean_rule_definitions.parquet",
        "identity": cache / "cache_identity.json",
    }
    identity = {
        "version": 2,
        "rows": len(pairs),
        "ordered_pair_digest": ordered_pair_digest(pairs),
        "items": source_identity(resolve(config["items_path"])),
        "accepted_facts": source_identity(resolve(config["accepted_facts_path"])),
        "rule_definitions": source_identity(resolve(config["rule_definitions_path"])),
        "allowed_roles": config["rules"]["allowed_roles"],
        "allowed_relations": config["rules"]["allowed_relations"],
        "alias_min_support": config["rules"]["attribute_alias_min_support"],
        "alias_min_purity": config["rules"]["attribute_alias_min_purity"],
    }
    identity_matches = paths["identity"].exists() and json.loads(paths["identity"].read_text(encoding="utf-8")) == identity
    if not rebuild and identity_matches and all(path.exists() for path in paths.values()):
        logger.info("Reusing CatBoost-1.1 feature cache")
        return (
            pd.read_parquet(paths["base"]),
            sparse.load_npz(paths["global"]).tocsr(),
            sparse.load_npz(paths["category"]).tocsr(),
            pd.read_parquet(paths["definitions"]),
        )

    logger.info("Building label-free attribute aliases")
    concept_map, audit = build_label_free_attribute_concept_map(
        resolve(config["accepted_facts_path"]),
        min_support=int(config["rules"]["attribute_alias_min_support"]),
        min_purity=float(config["rules"]["attribute_alias_min_purity"]),
    )
    audit.to_parquet(cache / "attribute_concept_map_audit.parquet", index=False)
    registry, definitions = load_label_free_rule_registry(
        resolve(config["rule_definitions_path"]),
        allowed_relations=set(config["rules"]["allowed_relations"]),
        allowed_roles=set(config["rules"]["allowed_roles"]),
    )
    logger.info("Whitelisted %s label-free rule templates", f"{len(definitions):,}")
    started = time.perf_counter()
    base, fired = extract_pair_features(pairs, items, concept_map, registry)
    global_matrix = rule_lists_to_csr(fired, len(definitions))
    category_matrix, category_vocabulary = category_rule_matrix(
        fired, base["category"].astype(str).tolist()
    )
    base = add_raw_rule_features(base, global_matrix, definitions)
    definition_audit = definitions.copy()
    definition_audit["semantic_family"] = [semantic_family(value) or "other" for value in definitions["concept"]]
    definition_audit["extracted_support_label_free"] = np.asarray(global_matrix.sum(axis=0)).ravel().astype(np.int64)
    definition_audit.to_parquet(cache / "clean_rule_template_audit.parquet", index=False)
    definition_audit.groupby(["relation", "semantic_family"], as_index=False).agg(
        definitions=("rule_id", "size"),
        fired_definitions=("extracted_support_label_free", lambda values: int((values > 0).sum())),
        observations=("extracted_support_label_free", "sum"),
    ).to_csv(cache / "clean_rule_template_summary.csv", index=False)
    base.to_parquet(paths["base"], index=False)
    sparse.save_npz(paths["global"], global_matrix, compressed=True)
    sparse.save_npz(paths["category"], category_matrix, compressed=True)
    definitions.to_parquet(paths["definitions"], index=False)
    (cache / "category_rule_vocabulary.json").write_text(
        json.dumps(category_vocabulary, ensure_ascii=False), encoding="utf-8"
    )
    paths["identity"].write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "Built %s rows, %d columns, %s clean rule observations in %.1f minutes",
        f"{len(base):,}", len(base.columns), f"{global_matrix.nnz:,}", (time.perf_counter() - started) / 60,
    )
    return base, global_matrix, category_matrix, definitions


def grouped_rule_evidence(
    matrix: sparse.csr_matrix,
    definitions: pd.DataFrame,
    target: np.ndarray,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    inner_folds: np.ndarray,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    kwargs = {
        "prior_strength": float(config["rules"]["prior_strength"]),
        "min_support": int(config["rules"]["global_min_support"]),
        "effect_clip": float(config["rules"]["effect_clip"]),
    }
    train_all, valid_all = crossfit_fold_rule_evidence(
        matrix, target, train_indices, valid_indices, inner_folds, **kwargs
    )
    train_parts = [train_all.add_prefix("all_")]
    valid_parts = [valid_all.add_prefix("all_")]
    for family, columns in rule_family_masks(definitions).items():
        train_family, valid_family = crossfit_fold_rule_evidence(
            matrix[:, columns].tocsr(), target, train_indices, valid_indices, inner_folds, **kwargs
        )
        prefix = f"family_{family}_"
        train_parts.append(train_family[list(GROUP_EVIDENCE_COLUMNS)].add_prefix(prefix))
        valid_parts.append(valid_family[list(GROUP_EVIDENCE_COLUMNS)].add_prefix(prefix))
    return pd.concat(train_parts, axis=1), pd.concat(valid_parts, axis=1)


def make_variant_frame(
    base: pd.DataFrame,
    global_evidence: pd.DataFrame,
    category_evidence: pd.DataFrame,
    variant: str,
) -> tuple[pd.DataFrame, list[str]]:
    spec = VARIANTS[variant]
    numeric = base.select_dtypes(exclude="object").copy().reset_index(drop=True)
    if not spec["rules"]:
        numeric = numeric.loc[:, ~numeric.columns.str.startswith("rule_")]
    frame = numeric
    categorical = ["category"]
    frame["category"] = base["category"].astype(str).to_numpy()
    if spec["rules"]:
        frame = pd.concat([frame, global_evidence.add_prefix("global_").reset_index(drop=True)], axis=1)
    if spec["category_aware"]:
        frame = pd.concat([frame, category_evidence.add_prefix("category_").reset_index(drop=True)], axis=1)
        for column in (
            "primary_conflict_type", "conflict_signature", "category_primary_conflict",
            "category_conflict_signature",
        ):
            frame[column] = base[column].astype(str).to_numpy()
            categorical.append(column)
    if spec["regime"]:
        for column in ("matching_regime", "regime_conflict_signature"):
            frame[column] = base[column].astype(str).to_numpy()
            categorical.append(column)
    return frame, categorical


def macro_ap(target: np.ndarray, scores: np.ndarray, categories: np.ndarray) -> tuple[float, dict[str, float]]:
    values: dict[str, float] = {}
    for category in sorted(pd.unique(categories)):
        mask = categories == category
        values[str(category)] = float(average_precision_score(target[mask], scores[mask]))
    return float(np.mean(list(values.values()))), values


def crossfit_negative_routing(
    oof: pd.DataFrame,
    score_column: str,
    risk_limit: float,
    segment_mode: str,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    if segment_mode == "global":
        segments = pd.Series("__global__", index=oof.index)
    else:
        segments = oof[segment_mode].astype(str)
    accepted_mask = np.zeros(len(oof), dtype=bool)
    threshold_rows: list[dict[str, Any]] = []
    for fold in sorted(oof["fold"].unique()):
        calibration_mask = oof["fold"].to_numpy() != fold
        validation_mask = oof["fold"].to_numpy() == fold
        for segment in sorted(pd.unique(segments[validation_mask])):
            calibration = calibration_mask & segments.eq(segment).to_numpy()
            validation = validation_mask & segments.eq(segment).to_numpy()
            if not calibration.any():
                continue
            state = best_side_state(
                threshold_states(
                    oof.loc[calibration, score_column].to_numpy(),
                    oof.loc[calibration, "target"].to_numpy(dtype=np.int8),
                    "negative",
                ),
                risk_limit,
            )
            threshold = float(state["threshold"])
            selected = validation & (oof[score_column].to_numpy() < threshold)
            accepted_mask |= selected
            errors = int(oof.loc[selected, "target"].sum())
            threshold_rows.append({
                "variant": score_column.removeprefix("p_match_"),
                "risk_limit": risk_limit,
                "segment_mode": segment_mode,
                "held_fold": int(fold),
                "segment": str(segment),
                "threshold": threshold,
                "calibration_accepted": int(state["accepted"]),
                "calibration_errors": int(state["errors"]),
                "calibration_ucb": float(state["error_ucb_95"]),
                "validation_accepted": int(selected.sum()),
                "validation_errors": errors,
            })
    accepted = int(accepted_mask.sum())
    errors = int(oof.loc[accepted_mask, "target"].sum())
    category_counts = oof.loc[accepted_mask, "category"].value_counts()
    fold_counts = oof.loc[accepted_mask, "fold"].value_counts()
    error_ucb = float(wilson_upper(errors, accepted)) if accepted else 1.0
    summary = {
        "variant": score_column.removeprefix("p_match_"),
        "risk_limit": risk_limit,
        "segment_mode": segment_mode,
        "accepted": accepted,
        "coverage": accepted / len(oof),
        "errors": errors,
        "empirical_error": errors / accepted if accepted else 0.0,
        "error_ucb_95": error_ucb,
        "passes_risk": bool(accepted and error_ucb < risk_limit),
        "verified_coverage": accepted / len(oof) if accepted and error_ucb < risk_limit else 0.0,
        "accepted_categories": int(len(category_counts)),
        "max_category_share": float(category_counts.iloc[0] / accepted) if accepted else 0.0,
        "folds_with_acceptance": int(len(fold_counts)),
        "min_fold_accepted": int(fold_counts.reindex(range(int(oof.fold.max()) + 1), fill_value=0).min()),
    }
    return summary, pd.DataFrame(threshold_rows), accepted_mask


def full_calibration_thresholds(
    oof: pd.DataFrame,
    score_column: str,
    risk_limit: float,
    segment_mode: str,
) -> list[dict[str, Any]]:
    segments = pd.Series("__global__", index=oof.index) if segment_mode == "global" else oof[segment_mode].astype(str)
    rows: list[dict[str, Any]] = []
    for segment in sorted(pd.unique(segments)):
        mask = segments.eq(segment).to_numpy()
        state = best_side_state(
            threshold_states(
                oof.loc[mask, score_column].to_numpy(),
                oof.loc[mask, "target"].to_numpy(dtype=np.int8),
                "negative",
            ),
            risk_limit,
        )
        rows.append({
            "variant": score_column.removeprefix("p_match_"),
            "risk_limit": risk_limit,
            "segment_mode": segment_mode,
            "segment": str(segment),
            "threshold": float(state["threshold"]),
            "accepted": int(state["accepted"]),
            "errors": int(state["errors"]),
            "coverage_within_segment": float(state["coverage"]),
            "error_ucb_95": float(state["error_ucb_95"]),
        })
    return rows


def write_results(
    metrics: pd.DataFrame,
    routing: pd.DataFrame,
    selected: list[str],
    path: Path,
) -> None:
    category = routing[routing.segment_mode.eq("category")].copy()
    rows = []
    for _, metric in metrics.iterrows():
        row: dict[str, Any] = {"variant": metric.variant, "macro_ap": metric.oof_macro_ap}
        for risk in sorted(category.risk_limit.unique()):
            value = category[(category.variant == metric.variant) & (category.risk_limit == risk)].iloc[0]
            row[f"verified_cov_{risk * 100:g}%"] = value.verified_coverage
            row[f"ucb_{risk * 100:g}%"] = value.error_ucb_95
        rows.append(row)
    display = pd.DataFrame(rows)
    headers = display.columns.tolist()
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        values = [row[0], *(f"{float(value):.6f}" for value in row[1:])]
        table.append("| " + " | ".join(values) + " |")
    text = [
        "# CatBoost-1.1 negative-router results", "",
        "Primary numbers are fold-to-fold cross-fitted category thresholds and route-specific",
        "negative Wilson 95% bounds. Positive early exit is intentionally disabled.", "",
        *table, "",
        "Selected variants: " + ", ".join(f"`{variant}`" for variant in selected), "",
        "Use `crossfit_negative_routing.csv` for global/category/regime diagnostics and",
        "`crossfit_threshold_details.csv` for every held-fold threshold.", "",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or resolve(config["output_dir"])
    if args.smoke and args.output_dir is None:
        output_dir = output_dir / "_smoke"
    logger = configure_logging(output_dir)
    started = time.perf_counter()
    if args.smoke:
        logger.warning("SMOKE MODE: metrics are not experiment results")

    pairs_path = resolve(config["human_train_pairs_path"])
    items_path = resolve(config["items_path"])
    pairs = pd.read_parquet(pairs_path, columns=["id1", "id2", "target"])
    pairs["target"] = pairs["target"].astype(np.int8)
    if args.smoke:
        pairs = make_smoke_subset(pairs, int(config["smoke_pairs"]))
    required_ids = pd.unique(pairs[["id1", "id2"]].to_numpy().reshape(-1))
    items = pd.read_parquet(items_path, columns=["id", "name", "attributes", "category"])
    items = items.loc[items.id.isin(required_ids)].reset_index(drop=True)
    lookup = items.set_index("id")["category"]
    categories = lookup.loc[pairs.id1].astype(str).to_numpy()
    if not np.array_equal(categories, lookup.loc[pairs.id2].astype(str).to_numpy()):
        raise ValueError("Cross-category pairs found")
    target = pairs.target.to_numpy(dtype=np.int8)
    components = stable_component_ids(pairs)
    n_folds = 2 if args.smoke else int(config["outer_folds"])
    outer_folds = assign_group_folds(target, categories, components, n_folds, int(config["seed"]))
    logger.info("Prepared %d product/component-disjoint outer folds over %s pairs", n_folds, f"{len(pairs):,}")

    base, global_rules, category_rules, definitions = load_or_build_features(
        config, output_dir, pairs, items, args.rebuild_features or args.smoke, logger
    )
    regimes = np.asarray([REGIME_BY_CATEGORY.get(category, "unknown_mixed") for category in categories])
    predictions = {variant: np.full(len(pairs), np.nan) for variant in args.variants}
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    importance_parts: list[pd.DataFrame] = []

    from catboost import CatBoostClassifier, Pool, __version__ as catboost_version

    model_config = dict(config["catboost"])
    model_config["iterations"] = int(config["smoke_iterations"]) if args.smoke else int(model_config["iterations"])
    model_config.update({
        "loss_function": "Logloss", "task_type": "CPU", "thread_count": int(config["threads"]),
        "allow_writing_files": False, "verbose": int(config["logging_period"]),
    })
    for fold in range(n_folds):
        train_idx = np.flatnonzero(outer_folds != fold)
        valid_idx = np.flatnonzero(outer_folds == fold)
        inner_n = 2 if args.smoke else int(config["inner_folds"])
        inner_folds = assign_group_folds(
            target[train_idx], categories[train_idx], components[train_idx], inner_n,
            int(config["seed"]) + 100 + fold,
        )
        logger.info("Fold %d/%d: nested clean-rule encoding", fold + 1, n_folds)
        global_train, global_valid = grouped_rule_evidence(
            global_rules, definitions, target, train_idx, valid_idx, inner_folds, config
        )
        category_train, category_valid = crossfit_fold_rule_evidence(
            category_rules, target, train_idx, valid_idx, inner_folds,
            prior_strength=float(config["rules"]["prior_strength"]),
            min_support=int(config["rules"]["category_min_support"]),
            effect_clip=float(config["rules"]["effect_clip"]),
        )
        for variant in args.variants:
            train_frame, categorical = make_variant_frame(
                base.iloc[train_idx].reset_index(drop=True), global_train, category_train, variant
            )
            valid_frame, valid_categorical = make_variant_frame(
                base.iloc[valid_idx].reset_index(drop=True), global_valid, category_valid, variant
            )
            if categorical != valid_categorical or list(train_frame) != list(valid_frame):
                raise AssertionError("Feature schema mismatch")
            weights = category_balanced_weights(categories[train_idx])
            weights *= np.where(target[train_idx] == 1, VARIANTS[variant]["positive_weight"], 1.0)
            weights /= weights.mean()
            fit_config = dict(model_config)
            fit_config["random_seed"] = int(config["seed"]) + fold
            logger.info(
                "Training %s fold %d: features=%d positive_weight=%.1f",
                variant, fold, len(train_frame.columns), VARIANTS[variant]["positive_weight"],
            )
            model = CatBoostClassifier(**fit_config)
            train_pool = Pool(train_frame, label=target[train_idx], weight=weights, cat_features=categorical)
            valid_pool = Pool(valid_frame, cat_features=categorical)
            model.fit(train_pool)
            predictions[variant][valid_idx] = model.predict_proba(valid_pool)[:, 1]
            model.save_model(str(models_dir / f"{variant}_fold{fold}.cbm"))
            importance_parts.append(pd.DataFrame({
                "variant": variant, "fold": fold, "feature": train_frame.columns,
                "importance": model.get_feature_importance(train_pool),
            }))

    oof = pairs.copy()
    oof["category"] = categories
    oof["matching_regime"] = regimes
    oof["component_id"] = components
    oof["fold"] = outer_folds
    metrics_rows: list[dict[str, Any]] = []
    category_ap_rows: list[dict[str, Any]] = []
    global_operating_rows: list[dict[str, Any]] = []
    curve_parts: list[pd.DataFrame] = []
    routing_rows: list[dict[str, Any]] = []
    threshold_parts: list[pd.DataFrame] = []
    full_threshold_rows: list[dict[str, Any]] = []
    risk_limits = [float(value) for value in config["risk_limits"]]
    for variant, scores in predictions.items():
        if not np.isfinite(scores).all():
            raise AssertionError(f"Missing OOF predictions for {variant}")
        score_column = f"p_match_{variant}"
        oof[score_column] = scores
        value, by_category = macro_ap(target, scores, categories)
        metrics_rows.append({
            "variant": variant, "oof_macro_ap": value,
            "oof_overall_ap": float(average_precision_score(target, scores)),
            "oof_roc_auc": float(roc_auc_score(target, scores)),
            "oof_logloss": float(log_loss(target, scores, labels=[0, 1])),
        })
        category_ap_rows.extend(
            {"variant": variant, "category": category, "oof_ap": ap}
            for category, ap in by_category.items()
        )
        curve = threshold_states(scores, target, "negative")
        curve.insert(0, "variant", variant)
        curve_parts.append(curve)
        for risk in risk_limits:
            state = best_side_state(curve, risk)
            global_operating_rows.append({"variant": variant, "risk_limit": risk, **state})
            for mode in ("global", "category", "matching_regime"):
                routing, details, _ = crossfit_negative_routing(oof, score_column, risk, mode)
                routing_rows.append(routing)
                threshold_parts.append(details)
                full_threshold_rows.extend(full_calibration_thresholds(oof, score_column, risk, mode))

    metrics = pd.DataFrame(metrics_rows)
    routing = pd.DataFrame(routing_rows)
    category_routing = routing[routing.segment_mode.eq("category")]
    ranking_rows = []
    for variant in args.variants:
        row = {"variant": variant}
        for risk in risk_limits:
            record = category_routing[
                (category_routing.variant == variant) & (category_routing.risk_limit == risk)
            ].iloc[0]
            row[f"verified_{risk:g}"] = record.verified_coverage
        row["macro_ap"] = metrics.loc[metrics.variant == variant, "oof_macro_ap"].iloc[0]
        ranking_rows.append(row)
    ranking = pd.DataFrame(ranking_rows)
    rank_columns = [f"verified_{risk:g}" for risk in risk_limits] + ["macro_ap"]
    selected = ranking.sort_values(rank_columns, ascending=False, kind="stable").head(2).variant.tolist()

    oof.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    metrics.to_csv(output_dir / "standalone_metrics.csv", index=False)
    pd.DataFrame(category_ap_rows).to_csv(output_dir / "category_ap.csv", index=False)
    pd.concat(importance_parts, ignore_index=True).to_csv(output_dir / "feature_importance_by_fold.csv", index=False)
    pd.concat(curve_parts, ignore_index=True).to_parquet(output_dir / "global_negative_risk_curves.parquet", index=False)
    pd.DataFrame(global_operating_rows).to_csv(output_dir / "global_negative_operating_points.csv", index=False)
    routing.to_csv(output_dir / "crossfit_negative_routing.csv", index=False)
    pd.concat(threshold_parts, ignore_index=True).to_csv(output_dir / "crossfit_threshold_details.csv", index=False)
    pd.DataFrame(full_threshold_rows).to_csv(output_dir / "full_oof_calibration_thresholds.csv", index=False)
    ranking.to_csv(output_dir / "selection_ranking.csv", index=False)
    (output_dir / "selected_variants.json").write_text(
        json.dumps({"selected": selected, "ranking": rank_columns}, indent=2), encoding="utf-8"
    )
    write_results(metrics, routing, selected, output_dir / "RESULTS.md")
    manifest = {
        "experiment": "catboost1_negative_router_v2",
        "smoke": args.smoke,
        "human_train_only": True,
        "iid_hard_ood_loaded": False,
        "positive_early_exit": False,
        "outer_folds": n_folds,
        "inner_folds": 2 if args.smoke else int(config["inner_folds"]),
        "variants": args.variants,
        "variant_specs": {key: VARIANTS[key] for key in args.variants},
        "clean_rule_definitions": len(definitions),
        "clean_rule_observations": global_rules.nnz,
        "catboost": model_config,
        "catboost_version": catboost_version,
        "python": platform.python_version(),
        "elapsed_seconds": time.perf_counter() - started,
        "google_sheets_written": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "COMPLETED").write_text("ok\n", encoding="utf-8")
    logger.info("Completed in %.1f minutes. Selected: %s", manifest["elapsed_seconds"] / 60, ", ".join(selected))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
