#!/usr/bin/env python3
"""Train leakage-safe OOF LogisticRegression and CatBoost over frozen S2 scores."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cheap_ensemble import (
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    build_pair_features,
    fit_hashed_char_idf,
    prepare_item_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, default=ROOT / "data/items_human.parquet")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "artifacts/manual/S2_VALUES_ONLY/validation_predictions.parquet",
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/cheap_ensemble_s2.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/cheap_ensemble_s2",
    )
    return parser.parse_args()


def macro_metrics(
    target: np.ndarray, scores: np.ndarray, categories: np.ndarray
) -> dict[str, Any]:
    per_category_ap = {}
    per_category_roc = {}
    for category in sorted(pd.unique(categories)):
        mask = categories == category
        per_category_ap[str(category)] = float(average_precision_score(target[mask], scores[mask]))
        per_category_roc[str(category)] = float(roc_auc_score(target[mask], scores[mask]))
    return {
        "macro_average_precision": float(np.mean(list(per_category_ap.values()))),
        "overall_average_precision": float(average_precision_score(target, scores)),
        "macro_roc_auc": float(np.mean(list(per_category_roc.values()))),
        "overall_roc_auc": float(roc_auc_score(target, scores)),
        "per_category_average_precision": per_category_ap,
        "per_category_roc_auc": per_category_roc,
    }


def component_family_groups(items: pd.DataFrame, pairs: pd.DataFrame) -> np.ndarray:
    parent = np.arange(len(items), dtype=np.int32)
    size = np.ones(len(items), dtype=np.int32)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def union(first: int, second: int) -> None:
        left, right = find(first), find(second)
        if left == right:
            return
        if size[left] < size[right]:
            left, right = right, left
        parent[right] = left
        size[left] += size[right]

    positions = pd.Series(np.arange(len(items), dtype=np.int32), index=items["id"].to_numpy())
    left = positions.loc[pairs["id1"]].to_numpy(dtype=np.int32)
    right = positions.loc[pairs["id2"]].to_numpy(dtype=np.int32)
    for first, second in zip(left, right):
        union(int(first), int(second))
    family_owner: dict[str, int] = {}
    for position, signature in enumerate(items["family_signature"].tolist()):
        if not signature:
            continue
        owner = family_owner.setdefault(signature, position)
        union(owner, position)
    return np.asarray([find(int(position)) for position in left], dtype=np.int32)


def category_weights(categories: np.ndarray) -> np.ndarray:
    counts = pd.Series(categories).value_counts()
    weights = pd.Series(categories).map(1.0 / counts).to_numpy(dtype=np.float64)
    return weights / weights.mean()


def logistic_model(config: dict[str, Any]) -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
            ("category", OneHotEncoder(handle_unknown="ignore"), ["category"]),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocessing),
            (
                "model",
                LogisticRegression(
                    C=float(config["C"]),
                    max_iter=int(config["max_iter"]),
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def catboost_model(config: dict[str, Any], seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=int(config["iterations"]),
        depth=int(config["depth"]),
        learning_rate=float(config["learning_rate"]),
        l2_leaf_reg=float(config["l2_leaf_reg"]),
        random_strength=float(config["random_strength"]),
        loss_function=str(config["loss_function"]),
        random_seed=seed,
        thread_count=int(config["thread_count"]),
        task_type="CPU",
        allow_writing_files=False,
        verbose=False,
    )


def slice_report(
    frame: pd.DataFrame, score_columns: list[str]
) -> dict[str, Any]:
    groups = {
        "hard_conflict_like": (
            frame["critical_conflict"].astype(bool)
            & (frame["title_token_jaccard"].ge(0.6) | frame["transformer_score"].ge(0.5))
        ),
        "sku_human_asymmetry": frame["sku_human_asymmetry"].astype(bool),
    }
    output: dict[str, Any] = {}
    for name, mask in groups.items():
        part = frame.loc[mask]
        metrics = {"support": len(part), "positive_rate": float(part["target"].mean())}
        for score_column in score_columns:
            if part["target"].nunique() == 2:
                category_values = []
                for _, category_part in part.groupby("category", sort=True):
                    if category_part["target"].nunique() == 2:
                        category_values.append(
                            average_precision_score(
                                category_part["target"], category_part[score_column]
                            )
                        )
                metrics[score_column] = (
                    float(np.mean(category_values)) if category_values else None
                )
        output[name] = metrics
    hard_negative = frame[
        frame["target"].eq(0)
        & frame["critical_conflict"].astype(bool)
        & frame["title_token_jaccard"].ge(0.6)
    ]
    hard_positive = frame[
        frame["target"].eq(1)
        & frame["sku_human_asymmetry"].astype(bool)
        & frame["transformer_score"].lt(0.5)
    ]
    output["label_specific"] = {
        "hard_negative_support": len(hard_negative),
        "hard_positive_support": len(hard_positive),
        **{
            f"hard_negative_mean_{column}": float(hard_negative[column].mean())
            for column in score_columns
        },
        **{
            f"hard_positive_mean_{column}": float(hard_positive[column].mean())
            for column in score_columns
        },
    }
    return output


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = args.output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    started = time.perf_counter()
    predictions = pd.read_parquet(args.predictions)
    required_prediction_columns = {"id1", "id2", "target", config["transformer_score_column"]}
    if not required_prediction_columns.issubset(predictions.columns):
        raise ValueError("Frozen S2 predictions are missing required columns")
    if predictions[["id1", "id2"]].duplicated().any():
        raise ValueError("Frozen S2 predictions contain duplicate pairs")

    raw_items = pd.read_parquet(args.items, columns=["id", "name", "attributes", "category"])
    char_config = config["char_tfidf"]
    idf_path = args.output_dir / "char_idf.npy"
    if idf_path.is_file():
        char_idf = np.load(idf_path)
        if len(char_idf) != int(char_config["n_features"]):
            raise ValueError("Cached char IDF dimension does not match config")
        idf_documents = len(raw_items)
    else:
        char_idf, idf_documents = fit_hashed_char_idf(
            raw_items["name"].fillna("").astype(str),
            n_features=int(char_config["n_features"]),
            ngram_min=int(char_config["ngram_min"]),
            ngram_max=int(char_config["ngram_max"]),
            batch_size=int(char_config["batch_size"]),
        )
        np.save(idf_path, char_idf)
    required_ids = set(predictions["id1"]) | set(predictions["id2"])
    selected_raw = raw_items[raw_items["id"].isin(required_ids)].reset_index(drop=True)
    if len(selected_raw) != len(required_ids):
        raise ValueError("items_human does not contain every validation product")
    items = prepare_item_records(selected_raw)
    transformer_scores = predictions[str(config["transformer_score_column"])].to_numpy(dtype=np.float32)
    features = build_pair_features(
        items,
        predictions,
        transformer_scores,
        char_idf,
        ngram_min=int(char_config["ngram_min"]),
        ngram_max=int(char_config["ngram_max"]),
    )
    target = predictions["target"].to_numpy(dtype=np.int8)
    categories = features["category"].to_numpy()
    groups = component_family_groups(items, predictions)
    folds = np.full(len(features), -1, dtype=np.int8)
    splitter = StratifiedGroupKFold(
        n_splits=int(config["meta_oof_folds"]), shuffle=True, random_state=int(config["seed"])
    )
    strata = pd.Series(categories).astype(str) + "|" + pd.Series(target).astype(str)
    logistic_oof = np.full(len(features), np.nan, dtype=np.float32)
    catboost_oof = np.full(len(features), np.nan, dtype=np.float32)
    weights = category_weights(categories)
    fold_reports = []
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(features, strata, groups)
    ):
        folds[validation_index] = fold
        if set(groups[train_index]) & set(groups[validation_index]):
            raise RuntimeError("Meta OOF fold leaks an item/family component")
        logistic = logistic_model(config["logistic_regression"])
        logistic.fit(features.iloc[train_index], target[train_index], model__sample_weight=weights[train_index])
        logistic_oof[validation_index] = logistic.predict_proba(features.iloc[validation_index])[:, 1]
        catboost = catboost_model(config["catboost"], int(config["seed"]) + fold)
        catboost.fit(
            features.iloc[train_index],
            target[train_index],
            sample_weight=weights[train_index],
            cat_features=["category"],
        )
        catboost_oof[validation_index] = catboost.predict_proba(features.iloc[validation_index])[:, 1]
        fold_reports.append(
            {
                "fold": fold,
                "train_pairs": len(train_index),
                "validation_pairs": len(validation_index),
                "validation_groups": len(set(groups[validation_index])),
                "positive_rate": float(target[validation_index].mean()),
            }
        )
        print(json.dumps(fold_reports[-1]), flush=True)
    if (folds < 0).any() or not np.isfinite(logistic_oof).all() or not np.isfinite(catboost_oof).all():
        raise RuntimeError("OOF predictions are incomplete")

    final_logistic = logistic_model(config["logistic_regression"])
    final_logistic.fit(features, target, model__sample_weight=weights)
    joblib.dump(final_logistic, models_dir / "logistic_pipeline.joblib", compress=3)
    final_catboost = catboost_model(config["catboost"], int(config["seed"]))
    final_catboost.fit(features, target, sample_weight=weights, cat_features=["category"])
    final_catboost.save_model(models_dir / "catboost_model.cbm")
    importance = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "importance": final_catboost.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(args.output_dir / "catboost_feature_importance.csv", index=False)

    output = predictions[["id1", "id2", "target"]].copy()
    output["category"] = categories
    output["fold"] = folds
    output["transformer_score"] = transformer_scores
    output["logistic_score"] = logistic_oof
    output["catboost_score"] = catboost_oof
    output.to_parquet(args.output_dir / "oof_predictions.parquet", index=False)
    feature_output = pd.concat([output[["id1", "id2", "target", "category", "fold"]], features.drop(columns="category")], axis=1)
    feature_output.to_parquet(args.output_dir / "features.parquet", index=False)
    metric_columns = {
        "transformer": transformer_scores,
        "logistic": logistic_oof,
        "catboost": catboost_oof,
    }
    metrics = {
        name: macro_metrics(target, scores, categories)
        for name, scores in metric_columns.items()
    }
    analysis = feature_output.copy()
    analysis["transformer_score"] = transformer_scores
    analysis["logistic_score"] = logistic_oof
    analysis["catboost_score"] = catboost_oof
    report = {
        "experiment": config["experiment"],
        "human_only": True,
        "transformer_retrained": False,
        "transformer_score_source": str(args.predictions),
        "transformer_score_column": config["transformer_score_column"],
        "meta_oof_protocol": "5-fold stratified item+family-component-disjoint on frozen transformer holdout",
        "pairs": len(features),
        "unique_products": len(items),
        "idf_documents": idf_documents,
        "feature_count": len(FEATURE_COLUMNS),
        "folds": fold_reports,
        "category_weighting": "equal total training weight per category; no hard-example upweighting",
        "metrics": metrics,
        "hard_slices": slice_report(
            analysis, ["transformer_score", "logistic_score", "catboost_score"]
        ),
        "top_catboost_features": importance.head(20).to_dict("records"),
        "wall_seconds": time.perf_counter() - started,
        "config": config,
    }
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "feature_config.json").write_text(
        json.dumps(
            {
                "feature_columns": list(FEATURE_COLUMNS),
                "numeric_features": list(NUMERIC_FEATURES),
                "char_tfidf": char_config,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({name: value["macro_average_precision"] for name, value in metrics.items()}, indent=2))
    print(importance.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
