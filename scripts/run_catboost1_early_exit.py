#!/usr/bin/env python
"""Run the four CPU CatBoost-1 early-exit experiments on human-train only."""

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
    VARIANT_COLUMNS,
    best_side_state,
    best_total_state,
    build_label_free_attribute_concept_map,
    category_balanced_weights,
    category_rule_matrix,
    extract_pair_features,
    fold_rule_evidence,
    load_label_free_rule_registry,
    rule_lists_to_csr,
    threshold_states,
    variant_frame,
)
from src.validation_splits import stable_component_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/catboost1_early_exit.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANT_COLUMNS), default=list(VARIANT_COLUMNS))
    parser.add_argument("--smoke", action="store_true", help="Fast code-path check, not a valid experiment")
    return parser.parse_args()


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("catboost1_early_exit")
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


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def source_identity(path: Path, *, sha256: bool = False) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if sha256:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    return result


def make_smoke_subset(pairs: pd.DataFrame, components: np.ndarray, limit: int) -> pd.DataFrame:
    helper = pairs.assign(_component=components)
    component_sizes = helper.groupby("_component").size().sort_index()
    keep: list[int] = []
    count = 0
    for component, size in component_sizes.items():
        keep.append(component)
        count += int(size)
        if count >= limit:
            break
    result = helper.loc[helper._component.isin(keep)].drop(columns="_component").reset_index(drop=True)
    return result


def make_folds(pairs: pd.DataFrame, categories: np.ndarray, n_splits: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    components = stable_component_ids(pairs)
    strata = pd.Series(categories).astype(str) + "||" + pairs["target"].astype(int).astype(str)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(pairs), -1, dtype=np.int8)
    dummy = np.zeros(len(pairs), dtype=np.int8)
    for fold, (train, valid) in enumerate(splitter.split(dummy, strata, groups=components)):
        folds[valid] = fold
        train_items = set(pd.unique(pairs.iloc[train][["id1", "id2"]].to_numpy().reshape(-1)))
        valid_items = set(pd.unique(pairs.iloc[valid][["id1", "id2"]].to_numpy().reshape(-1)))
        overlap = train_items & valid_items
        if overlap:
            raise AssertionError(f"Fold {fold} leaks {len(overlap)} item ids")
        fold_targets = pairs.iloc[valid]["target"].to_numpy(dtype=np.int8)
        if len(np.unique(fold_targets)) != 2:
            raise AssertionError(f"Fold {fold} does not contain both target classes")
    if np.any(folds < 0):
        raise AssertionError("Some rows were not assigned to an OOF fold")
    return folds, components


def load_or_build_features(
    config: dict[str, Any],
    output_dir: Path,
    pairs: pd.DataFrame,
    items: pd.DataFrame,
    rebuild: bool,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, sparse.csr_matrix, sparse.csr_matrix, pd.DataFrame]:
    cache_dir = output_dir / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_path = cache_dir / "base_features.parquet"
    global_rules_path = cache_dir / "global_rule_matrix.npz"
    category_rules_path = cache_dir / "category_rule_matrix.npz"
    definitions_path = cache_dir / "rule_definitions_label_free.parquet"
    identity_path = cache_dir / "cache_identity.json"
    facts_path = resolve(ROOT, config["accepted_facts_path"])
    rule_path = resolve(ROOT, config["rule_definitions_path"])
    pair_digest = hashlib.sha256(
        pd.util.hash_pandas_object(pairs[["id1", "id2"]], index=False).to_numpy().tobytes()
    ).hexdigest()
    cache_identity = {
        "rows": len(pairs),
        "ordered_pair_digest": pair_digest,
        "items": source_identity(resolve(ROOT, config["items_path"]), sha256=False),
        "accepted_facts": source_identity(facts_path, sha256=False),
        "rule_definitions": source_identity(rule_path, sha256=False),
        "alias_min_support": int(config["rule_templates"]["attribute_alias_min_support"]),
        "alias_min_purity": float(config["rule_templates"]["attribute_alias_min_purity"]),
    }
    expected = (base_path, global_rules_path, category_rules_path, definitions_path, identity_path)
    identity_matches = False
    if identity_path.exists():
        identity_matches = json.loads(identity_path.read_text(encoding="utf-8")) == cache_identity
    if not rebuild and all(path.exists() for path in expected) and identity_matches:
        logger.info("Reusing cached cheap features and rule matrices")
        base = pd.read_parquet(base_path)
        if len(base) != len(pairs):
            raise ValueError("Feature cache row count differs; rerun with --rebuild-features")
        return (
            base,
            sparse.load_npz(global_rules_path).tocsr(),
            sparse.load_npz(category_rules_path).tocsr(),
            pd.read_parquet(definitions_path),
        )

    if not rebuild and any(path.exists() for path in expected) and not identity_matches:
        logger.info("Feature cache identity changed; rebuilding it")
    logger.info("Building label-free attribute/concept aliases from %s", facts_path)
    concept_map, concept_audit = build_label_free_attribute_concept_map(
        facts_path,
        min_support=int(config["rule_templates"]["attribute_alias_min_support"]),
        min_purity=float(config["rule_templates"]["attribute_alias_min_purity"]),
    )
    concept_audit.to_parquet(cache_dir / "attribute_concept_map_audit.parquet", index=False)
    logger.info("Accepted %s label-free attribute aliases", f"{len(concept_map):,}")
    registry, definitions = load_label_free_rule_registry(rule_path)
    logger.info(
        "Loaded %s label-free rule definitions; stored label statistics were not read",
        f"{len(definitions):,}",
    )

    started = time.perf_counter()
    base, fired = extract_pair_features(pairs, items, concept_map, registry)
    global_matrix = rule_lists_to_csr(fired, len(definitions))
    category_matrix, category_vocabulary = category_rule_matrix(fired, base["category"].astype(str).tolist())
    base.to_parquet(base_path, index=False)
    sparse.save_npz(global_rules_path, global_matrix, compressed=True)
    sparse.save_npz(category_rules_path, category_matrix, compressed=True)
    definitions.to_parquet(definitions_path, index=False)
    identity_path.write_text(json.dumps(cache_identity, ensure_ascii=False, indent=2), encoding="utf-8")
    (cache_dir / "category_rule_vocabulary.json").write_text(
        json.dumps(category_vocabulary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "Built %s pair rows, %d columns and %s fired rule observations in %.1f minutes",
        f"{len(base):,}", len(base.columns), f"{global_matrix.nnz:,}", (time.perf_counter() - started) / 60,
    )
    return base, global_matrix, category_matrix, definitions


def macro_ap(target: np.ndarray, scores: np.ndarray, categories: np.ndarray) -> tuple[float, dict[str, float]]:
    by_category: dict[str, float] = {}
    for category in sorted(pd.unique(categories)):
        mask = categories == category
        if len(np.unique(target[mask])) != 2:
            raise ValueError(f"Category {category!r} lacks a target class")
        by_category[str(category)] = float(average_precision_score(target[mask], scores[mask]))
    return float(np.mean(list(by_category.values()))), by_category


def write_markdown_summary(summary: pd.DataFrame, selected: list[str], path: Path) -> None:
    display = summary[[
        "variant", "oof_macro_ap", "coverage_at_0_1pct", "coverage_at_0_2pct",
        "coverage_at_0_5pct", "neg_coverage_at_0_1pct", "pos_coverage_at_0_1pct",
    ]].copy()
    for column in display.columns[1:]:
        display[column] = display[column].map(lambda value: f"{value:.6f}")
    headers = display.columns.tolist()
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    table.extend("| " + " | ".join(map(str, row)) + " |" for row in display.itertuples(index=False, name=None))
    lines = [
        "# CatBoost-1 early-exit OOF results",
        "",
        "All rows come exclusively from `human_train_pairs.parquet`; folds are product/component-disjoint.",
        "Wilson 95% upper bounds are strict (`UCB < risk limit`).",
        "",
        *table,
        "",
        "Selected for the next experiment: " + ", ".join(f"`{item}`" for item in selected),
        "",
        "`neg_coverage` and `pos_coverage` in the compact table are the tails of the jointly optimal",
        "total operating point at 0.1%. Full side-specific and joint operating points are in",
        "`risk_coverage_operating_points.csv`; complete tie-safe sweeps are in `risk_coverage_curves.parquet`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or resolve(ROOT, config["output_dir"])
    if args.smoke and args.output_dir is None:
        output_dir = output_dir / "_smoke"
    logger = configure_logging(output_dir)
    started = time.perf_counter()
    if args.smoke:
        logger.warning("SMOKE MODE: outputs are not valid experiment results")

    items_path = resolve(ROOT, config["items_path"])
    pairs_path = resolve(ROOT, config["human_train_pairs_path"])
    logger.info("Reading human-train pairs only: %s", pairs_path)
    pairs = pd.read_parquet(pairs_path, columns=["id1", "id2", "target"])
    pairs["target"] = pairs["target"].astype(np.int8)
    if not set(pd.unique(pairs.target)).issubset({0, 1}):
        raise ValueError("target must be binary")
    components = stable_component_ids(pairs)
    if args.smoke:
        pairs = make_smoke_subset(pairs, components, int(config["smoke_pairs"]))
        components = stable_component_ids(pairs)
    required_ids = pd.unique(pairs[["id1", "id2"]].to_numpy().reshape(-1))
    logger.info("Reading item table and retaining %s referenced products", f"{len(required_ids):,}")
    items = pd.read_parquet(items_path, columns=["id", "name", "attributes", "category"])
    items = items.loc[items.id.isin(required_ids)].reset_index(drop=True)
    category_lookup = items.set_index("id")["category"]
    categories = category_lookup.loc[pairs.id1].astype(str).to_numpy()
    right_categories = category_lookup.loc[pairs.id2].astype(str).to_numpy()
    if not np.array_equal(categories, right_categories):
        raise ValueError("Found cross-category pairs")

    n_splits = 2 if args.smoke else int(config["folds"])
    folds, components = make_folds(pairs, categories, n_splits, int(config["seed"]))
    fold_frame = pairs.copy()
    fold_frame["category"] = categories
    fold_frame["component_id"] = components
    fold_frame["fold"] = folds
    fold_frame.to_parquet(output_dir / "oof_fold_assignments.parquet", index=False)
    logger.info("Created %d deterministic stratified component-disjoint folds", n_splits)

    base, global_rules, category_rules, definitions = load_or_build_features(
        config, output_dir, pairs, items, args.rebuild_features or args.smoke, logger
    )
    if not np.array_equal(base["category"].astype(str).to_numpy(), categories):
        raise AssertionError("Feature rows/categories are not aligned to pairs")

    from catboost import CatBoostClassifier, Pool, __version__ as catboost_version

    target = pairs.target.to_numpy(dtype=np.int8)
    predictions: dict[str, np.ndarray] = {variant: np.full(len(pairs), np.nan) for variant in args.variants}
    feature_importances: list[pd.DataFrame] = []
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    training = dict(config["catboost"])
    training["iterations"] = int(config["smoke_iterations"]) if args.smoke else int(training["iterations"])
    training.update({
        "loss_function": "Logloss",
        "task_type": "CPU",
        "thread_count": int(config["threads"]),
        "allow_writing_files": False,
        "verbose": int(config["logging_period"]),
    })

    for fold in range(n_splits):
        train_idx = np.flatnonzero(folds != fold)
        valid_idx = np.flatnonzero(folds == fold)
        logger.info(
            "Fold %d/%d: train=%s valid=%s",
            fold + 1, n_splits, f"{len(train_idx):,}", f"{len(valid_idx):,}",
        )
        global_train, global_valid = fold_rule_evidence(
            global_rules, target, train_idx, valid_idx,
            prior_strength=float(config["rule_templates"]["prior_strength"]),
        )
        category_train, category_valid = fold_rule_evidence(
            category_rules, target, train_idx, valid_idx,
            prior_strength=float(config["rule_templates"]["prior_strength"]),
        )
        weights = category_balanced_weights(categories[train_idx])

        for variant in args.variants:
            logger.info("Training %s fold %d on CPU", variant, fold)
            needs_category = bool(VARIANT_COLUMNS[variant]["category_rule"])
            train_frame, categorical = variant_frame(
                base.iloc[train_idx].reset_index(drop=True), global_train,
                category_train if needs_category else None, variant,
            )
            valid_frame, valid_categorical = variant_frame(
                base.iloc[valid_idx].reset_index(drop=True), global_valid,
                category_valid if needs_category else None, variant,
            )
            if categorical != valid_categorical or list(train_frame.columns) != list(valid_frame.columns):
                raise AssertionError("Train/validation feature schema mismatch")
            model_config = dict(training)
            model_config["random_seed"] = int(config["seed"]) + fold
            model = CatBoostClassifier(**model_config)
            train_pool = Pool(train_frame, label=target[train_idx], weight=weights, cat_features=categorical)
            valid_pool = Pool(valid_frame, cat_features=categorical)
            model.fit(train_pool)
            predictions[variant][valid_idx] = model.predict_proba(valid_pool)[:, 1]
            model.save_model(str(models_dir / f"{variant}_fold{fold}.cbm"))
            importance = pd.DataFrame({
                "variant": variant,
                "fold": fold,
                "feature": train_frame.columns,
                "importance": model.get_feature_importance(train_pool),
            })
            feature_importances.append(importance)
            del model, train_pool, valid_pool, train_frame, valid_frame

    oof = fold_frame.copy()
    metric_rows: list[dict[str, Any]] = []
    category_metric_rows: list[dict[str, Any]] = []
    curve_parts: list[pd.DataFrame] = []
    operating_rows: list[dict[str, Any]] = []
    risk_limits = [float(value) for value in config["risk_limits"]]
    for variant, scores in predictions.items():
        if not np.isfinite(scores).all():
            raise AssertionError(f"{variant} has missing/non-finite OOF predictions")
        oof[f"p_match_{variant}"] = scores
        score_macro_ap, category_scores = macro_ap(target, scores, categories)
        metric_rows.append({
            "variant": variant,
            "oof_macro_ap": score_macro_ap,
            "oof_overall_ap": float(average_precision_score(target, scores)),
            "oof_roc_auc": float(roc_auc_score(target, scores)),
            "oof_logloss": float(log_loss(target, scores, labels=[0, 1])),
        })
        category_metric_rows.extend(
            {"variant": variant, "category": category, "oof_ap": score}
            for category, score in category_scores.items()
        )
        neg_curve = threshold_states(scores, target, "negative")
        pos_curve = threshold_states(scores, target, "positive")
        for frame in (neg_curve, pos_curve):
            frame.insert(0, "variant", variant)
            curve_parts.append(frame)
        for risk in risk_limits:
            neg = best_side_state(neg_curve, risk)
            pos = best_side_state(pos_curve, risk)
            total = best_total_state(neg_curve, pos_curve, risk, len(target))
            operating_rows.extend([
                {"variant": variant, "risk_limit": risk, "scope": "negative", **neg},
                {"variant": variant, "risk_limit": risk, "scope": "positive", **pos},
                {"variant": variant, "risk_limit": risk, "scope": "total", **total},
            ])

    metrics = pd.DataFrame(metric_rows)
    operating = pd.DataFrame(operating_rows)
    summary_rows: list[dict[str, Any]] = []
    for metric in metric_rows:
        variant = metric["variant"]
        row = dict(metric)
        for risk in risk_limits:
            total = operating.loc[
                (operating.variant == variant) & (operating.risk_limit == risk) & (operating.scope == "total")
            ].iloc[0]
            suffix = f"{risk * 100:g}".replace(".", "_") + "pct"
            row[f"coverage_at_{suffix}"] = float(total.coverage)
            row[f"neg_coverage_at_{suffix}"] = float(total.neg_coverage)
            row[f"pos_coverage_at_{suffix}"] = float(total.pos_coverage)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    rank_columns = [f"coverage_at_{str(risk * 100).replace('.', '_')}pct" for risk in risk_limits]
    # Stable lexicographic priority: strictest risk, then looser risks, then macro AP.
    selected = summary.sort_values(rank_columns + ["oof_macro_ap"], ascending=False, kind="stable").head(2)["variant"].tolist()

    oof.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    metrics.to_csv(output_dir / "standalone_metrics.csv", index=False)
    pd.DataFrame(category_metric_rows).to_csv(output_dir / "category_ap.csv", index=False)
    pd.concat(feature_importances, ignore_index=True).to_csv(output_dir / "feature_importance_by_fold.csv", index=False)
    pd.concat(curve_parts, ignore_index=True).to_parquet(output_dir / "risk_coverage_curves.parquet", index=False)
    operating.to_csv(output_dir / "risk_coverage_operating_points.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "selected_variants.json").write_text(
        json.dumps({"selected": selected, "ranking": rank_columns + ["oof_macro_ap"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_summary(summary, selected, output_dir / "RESULTS.md")

    manifest = {
        "experiment": "catboost1_early_exit_v1",
        "smoke": args.smoke,
        "human_train_only": True,
        "iid_hard_ood_loaded": False,
        "neural_predictions_used": False,
        "catboost2_used": False,
        "folds": n_splits,
        "seed": int(config["seed"]),
        "variants": args.variants,
        "catboost": training,
        "catboost_version": catboost_version,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_files": {
            "pairs": source_identity(pairs_path, sha256=True),
            "items": source_identity(items_path, sha256=False),
            "accepted_facts_label_free": source_identity(resolve(ROOT, config["accepted_facts_path"]), sha256=False),
            "rule_definitions_label_free_columns_only": source_identity(resolve(ROOT, config["rule_definitions_path"]), sha256=False),
        },
        "rule_definition_count": len(definitions),
        "rows": len(pairs),
        "positive_rate": float(target.mean()),
        "elapsed_seconds": time.perf_counter() - started,
        "google_sheets_written": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "COMPLETED").write_text("ok\n", encoding="utf-8")
    logger.info("Completed in %.1f minutes. Selected: %s", manifest["elapsed_seconds"] / 60, ", ".join(selected))
    logger.info("Readable report: %s", output_dir / "RESULTS.md")


if __name__ == "__main__":
    # Avoid accidental oversubscription underneath CatBoost's own thread pool.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
