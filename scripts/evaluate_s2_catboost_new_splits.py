#!/usr/bin/env python3
"""Fit a leakage-safe S2 CatBoost stacker and score three external splits."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cheap_ensemble import (
    FEATURE_COLUMNS,
    build_pair_features,
    fit_hashed_char_idf,
    prepare_item_records,
)
from scripts.train_s2_cheap_ensemble import category_weights


SPLIT_FILES = {
    "iid": "human_iid_validation_pairs.parquet",
    "hard": "human_hard_validation_pairs.parquet",
    "ood": "human_ood_validation_pairs.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--meta-predictions", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--s2-evaluations-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/cheap_ensemble_s2.json"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def macro_metrics(
    target: np.ndarray, scores: np.ndarray, categories: np.ndarray
) -> dict[str, Any]:
    per_category_ap: dict[str, float] = {}
    per_category_roc: dict[str, float] = {}
    for category in sorted(pd.unique(categories)):
        mask = categories == category
        per_category_ap[str(category)] = float(
            average_precision_score(target[mask], scores[mask])
        )
        per_category_roc[str(category)] = float(roc_auc_score(target[mask], scores[mask]))
    return {
        "macro_average_precision": float(np.mean(list(per_category_ap.values()))),
        "overall_average_precision": float(average_precision_score(target, scores)),
        "macro_roc_auc": float(np.mean(list(per_category_roc.values()))),
        "overall_roc_auc": float(roc_auc_score(target, scores)),
        "per_category_average_precision": per_category_ap,
        "per_category_roc_auc": per_category_roc,
    }


def unordered_pair_index(frame: pd.DataFrame) -> pd.MultiIndex:
    left = frame["id1"].astype(str).to_numpy()
    right = frame["id2"].astype(str).to_numpy()
    return pd.MultiIndex.from_arrays(
        [np.minimum(left, right), np.maximum(left, right)], names=["left", "right"]
    )


def load_test_predictions(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for split, filename in SPLIT_FILES.items():
        labelled = pd.read_parquet(args.split_dir / filename)
        prediction_path = args.s2_evaluations_dir / split / "predictions.parquet"
        predictions = pd.read_parquet(prediction_path)
        if len(labelled) != len(predictions):
            raise ValueError(f"{split}: labelled/prediction row count mismatch")
        if not unordered_pair_index(labelled).equals(unordered_pair_index(predictions)):
            raise ValueError(f"{split}: labelled/prediction pair order mismatch")
        if not np.array_equal(
            labelled["target"].to_numpy(dtype=np.int8),
            predictions["target"].to_numpy(dtype=np.int8),
        ):
            raise ValueError(f"{split}: labelled/prediction targets differ")
        output[split] = predictions
    return output


def catboost_model(config: dict[str, Any]) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=int(config["iterations"]),
        depth=int(config["depth"]),
        learning_rate=float(config["learning_rate"]),
        l2_leaf_reg=float(config["l2_leaf_reg"]),
        random_strength=float(config["random_strength"]),
        loss_function=str(config["loss_function"]),
        random_seed=42,
        thread_count=int(config["thread_count"]),
        task_type="CPU",
        allow_writing_files=False,
        verbose=False,
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    started = time.perf_counter()

    tests = load_test_predictions(args)
    test_ids: set[object] = set()
    for frame in tests.values():
        test_ids.update(frame["id1"].tolist())
        test_ids.update(frame["id2"].tolist())

    meta_all = pd.read_parquet(args.meta_predictions)
    required = {"id1", "id2", "target", "score"}
    if not required.issubset(meta_all.columns):
        raise ValueError(f"Meta predictions are missing {sorted(required - set(meta_all.columns))}")
    safe_mask = ~meta_all["id1"].isin(test_ids) & ~meta_all["id2"].isin(test_ids)
    meta = meta_all.loc[safe_mask].reset_index(drop=True)
    if meta.empty:
        raise ValueError("No leakage-safe meta-training pairs remain")
    meta_ids = set(meta["id1"].tolist()) | set(meta["id2"].tolist())
    if meta_ids & test_ids:
        raise RuntimeError("Meta-training and external validation share products")

    raw_items = pd.read_parquet(
        args.items, columns=["id", "name", "attributes", "category"]
    )
    char_config = config["char_tfidf"]
    char_idf, idf_documents = fit_hashed_char_idf(
        raw_items["name"].fillna("").astype(str),
        n_features=int(char_config["n_features"]),
        ngram_min=int(char_config["ngram_min"]),
        ngram_max=int(char_config["ngram_max"]),
        batch_size=int(char_config["batch_size"]),
    )
    np.save(args.output_dir / "char_idf.npy", char_idf)

    required_ids = meta_ids | test_ids
    selected_raw = raw_items[raw_items["id"].isin(required_ids)].reset_index(drop=True)
    if len(selected_raw) != len(required_ids):
        raise ValueError("items file does not contain every required product")
    items = prepare_item_records(selected_raw)

    train_features = build_pair_features(
        items,
        meta,
        meta["score"].to_numpy(dtype=np.float32),
        char_idf,
        ngram_min=int(char_config["ngram_min"]),
        ngram_max=int(char_config["ngram_max"]),
    )
    train_target = meta["target"].to_numpy(dtype=np.int8)
    train_categories = train_features["category"].to_numpy()
    model = catboost_model(config["catboost"])
    model.fit(
        train_features,
        train_target,
        sample_weight=category_weights(train_categories),
        cat_features=["category"],
    )
    model.save_model(args.output_dir / "catboost_model.cbm")
    importance = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(args.output_dir / "catboost_feature_importance.csv", index=False)

    split_reports: dict[str, dict[str, Any]] = {}
    for split, frame in tests.items():
        features = build_pair_features(
            items,
            frame,
            frame["score"].to_numpy(dtype=np.float32),
            char_idf,
            ngram_min=int(char_config["ngram_min"]),
            ngram_max=int(char_config["ngram_max"]),
        )
        target = frame["target"].to_numpy(dtype=np.int8)
        categories = features["category"].to_numpy()
        transformer_score = frame["score"].to_numpy(dtype=np.float32)
        catboost_score = model.predict_proba(features)[:, 1].astype(np.float32)
        transformer_metrics = macro_metrics(target, transformer_score, categories)
        catboost_metrics = macro_metrics(target, catboost_score, categories)
        prediction_output = frame[["id1", "id2", "target", "category", "score"]].copy()
        prediction_output = prediction_output.rename(columns={"score": "transformer_score"})
        prediction_output["catboost_score"] = catboost_score
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        prediction_output.to_parquet(split_dir / "predictions.parquet", index=False)
        split_reports[split] = {
            "pairs": len(frame),
            "positives": int(target.sum()),
            "categories": int(pd.Series(categories).nunique()),
            "transformer": transformer_metrics,
            "catboost": catboost_metrics,
            "absolute_delta_macro_ap": (
                catboost_metrics["macro_average_precision"]
                - transformer_metrics["macro_average_precision"]
            ),
        }
        print(
            json.dumps(
                {
                    "split": split,
                    "transformer_macro_ap": transformer_metrics["macro_average_precision"],
                    "catboost_macro_ap": catboost_metrics["macro_average_precision"],
                    "delta": split_reports[split]["absolute_delta_macro_ap"],
                }
            ),
            flush=True,
        )

    report = {
        "experiment": "minilm_s2_catboost_new_validation_splits",
        "protocol": (
            "CatBoost fit on frozen leakage-safe old S2 holdout scores after removing "
            "every product present in IID/hard/OOD; external tests use new S2 scores"
        ),
        "meta_pairs_before_filter": len(meta_all),
        "meta_pairs_after_filter": len(meta),
        "removed_meta_pairs": int((~safe_mask).sum()),
        "meta_test_item_overlap": 0,
        "idf_documents": idf_documents,
        "feature_count": len(FEATURE_COLUMNS),
        "category_weighting": "equal total training weight per category",
        "split_reports": split_reports,
        "top_catboost_features": importance.head(20).to_dict("records"),
        "wall_seconds": time.perf_counter() - started,
        "config": config,
    }
    (args.output_dir / "evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "COMPLETED").write_text("complete\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
