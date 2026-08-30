#!/usr/bin/env python3
"""Evaluate frozen C1A negative routing before the BGE+MiniLM rank ensemble."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.catboost1_early_exit import extract_pair_features


MODEL_DIR = ROOT / "artifacts/catboost1_negative_router_v2/models"
ALIAS_PATH = (
    ROOT / "artifacts/catboost1_negative_router_v2/feature_cache/attribute_concept_map_audit.parquet"
)
OOF_PATH = ROOT / "artifacts/catboost1_negative_router_v2/oof_predictions.parquet"
ITEMS_PATH = ROOT / "data/items_human.parquet"
PAIRS = {
    "ordinary": ROOT / "data/validation_splits_v1/human_iid_validation_pairs.parquet",
    "hard": ROOT / "data/validation_splits_v1/human_hard_validation_pairs.parquet",
    "OOD": ROOT / "data/validation_splits_v1/human_ood_validation_pairs.parquet",
}
PREDICTIONS = {
    "bge": ROOT / "preds/preds_bge",
    "minilm": ROOT / "preds/preds_minilm",
}
PREDICTION_FILENAMES = {
    "ordinary": "iid_validation_predictions.parquet",
    "hard": "hard_validation_predictions.parquet",
    "OOD": "ood_validation_predictions.parquet",
}
OUTPUT_DIR = ROOT / "reports/c1a_bge_minilm_cascade_v1"
ROUTING_FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def normalized_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", ascending=True).to_numpy(dtype=np.float64) / len(values)


def macro_ap(frame: pd.DataFrame, scores: np.ndarray) -> float:
    values = []
    categories = frame["category"].astype(str).to_numpy()
    target = frame["target"].to_numpy(dtype=np.int8)
    for category in sorted(pd.unique(categories)):
        mask = categories == category
        values.append(average_precision_score(target[mask], scores[mask]))
    return float(np.mean(values))


def c1a_frame(base: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    numeric = base.select_dtypes(exclude="object").copy().reset_index(drop=True)
    numeric = numeric.loc[:, ~numeric.columns.str.startswith("rule_")]
    numeric["category"] = base["category"].astype(str).to_numpy()
    if list(numeric.columns) != feature_names:
        missing = [name for name in feature_names if name not in numeric]
        extra = [name for name in numeric if name not in feature_names]
        raise RuntimeError(f"C1A schema mismatch: missing={missing}, extra={extra}")
    return numeric


def load_neural_scores(split: str, pairs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    result = []
    for model in ("bge", "minilm"):
        path = PREDICTIONS[model] / PREDICTION_FILENAMES[split]
        predictions = pd.read_parquet(path, columns=["id1", "id2", "score"])
        aligned = pairs[["id1", "id2"]].merge(
            predictions, on=["id1", "id2"], how="left", validate="one_to_one", sort=False
        )
        if aligned["score"].isna().any() or len(aligned) != len(pairs):
            raise RuntimeError(f"Could not align {model}/{split}")
        result.append(aligned["score"].to_numpy(dtype=np.float64))
    return result[0], result[1]


def cascade_scores(
    catboost_scores: np.ndarray,
    bge_scores: np.ndarray,
    minilm_scores: np.ndarray,
    rejected: np.ndarray,
) -> np.ndarray:
    routed = ~rejected
    if not rejected.any() or not routed.any():
        raise ValueError("Routing candidate must contain both exits")
    output = np.empty(len(rejected), dtype=np.float64)
    rejected_fraction = float(rejected.mean())
    output[rejected] = rejected_fraction * normalized_rank(catboost_scores[rejected])
    neural = (
        normalized_rank(bge_scores[routed]) + normalized_rank(minilm_scores[routed])
    ) * 0.5
    output[routed] = rejected_fraction + (1.0 - rejected_fraction) * neural
    return output


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    pair_parts = []
    for split, path in PAIRS.items():
        frame = pd.read_parquet(path, columns=["id1", "id2", "target"])
        frame["split"] = split
        frame["split_index"] = np.arange(len(frame), dtype=np.int64)
        pair_parts.append(frame)
    pairs = pd.concat(pair_parts, ignore_index=True)
    required_ids = pd.unique(pairs[["id1", "id2"]].to_numpy().reshape(-1))
    items = pd.read_parquet(ITEMS_PATH, columns=["id", "name", "attributes", "category"])
    items = items.loc[items["id"].isin(required_ids)].reset_index(drop=True)
    alias = pd.read_parquet(ALIAS_PATH)
    learned_concepts = dict(
        alias.loc[alias["accepted"], ["attribute_key", "concept"]].itertuples(index=False, name=None)
    )
    feature_started = time.perf_counter()
    base, _ = extract_pair_features(pairs, items, learned_concepts, {})
    feature_seconds = time.perf_counter() - feature_started

    models = []
    for fold in range(5):
        model = CatBoostClassifier()
        model.load_model(str(MODEL_DIR / f"C1A_category_no_rules_fold{fold}.cbm"))
        models.append(model)
    feature_frame = c1a_frame(base, models[0].feature_names_)
    pool = Pool(feature_frame, cat_features=["category"])
    prediction_started = time.perf_counter()
    c1a_scores = np.mean([model.predict_proba(pool)[:, 1] for model in models], axis=0)
    prediction_seconds = time.perf_counter() - prediction_started
    pairs["category"] = base["category"].astype(str).to_numpy()
    pairs["c1a_score"] = c1a_scores

    oof = pd.read_parquet(OOF_PATH, columns=["p_match_C1A_category_no_rules"])
    oof_scores = oof["p_match_C1A_category_no_rules"].to_numpy(dtype=np.float64)
    thresholds = {
        fraction: float(np.quantile(oof_scores, fraction, method="linear"))
        for fraction in ROUTING_FRACTIONS
    }
    rows = []
    score_parts = []
    for split in PAIRS:
        frame = pairs.loc[pairs["split"] == split].sort_values("split_index").reset_index(drop=True)
        bge, minilm = load_neural_scores(split, frame)
        original = (normalized_rank(bge) + normalized_rank(minilm)) * 0.5
        original_ap = macro_ap(frame, original)
        catboost = frame["c1a_score"].to_numpy(dtype=np.float64)
        target = frame["target"].to_numpy(dtype=np.int8)
        score_frame = frame[["id1", "id2", "target", "category", "c1a_score"]].copy()
        score_frame["bge_score"] = bge
        score_frame["minilm_score"] = minilm
        score_frame["original_ensemble_score"] = original
        score_parts.append(score_frame.assign(split=split))
        for requested_fraction, threshold in thresholds.items():
            rejected = catboost < threshold
            if not rejected.any() or rejected.all():
                continue
            cascade = cascade_scores(catboost, bge, minilm, rejected)
            rejected_count = int(rejected.sum())
            rejected_positives = int(target[rejected].sum())
            rows.append(
                {
                    "split": split,
                    "oof_requested_rejection_fraction": requested_fraction,
                    "threshold": threshold,
                    "pairs": len(frame),
                    "rejected_pairs": rejected_count,
                    "rejected_fraction": rejected_count / len(frame),
                    "neural_pairs": int((~rejected).sum()),
                    "neural_fraction": float((~rejected).mean()),
                    "rejected_positives": rejected_positives,
                    "rejected_positive_rate": rejected_positives / rejected_count,
                    "original_macro_ap": original_ap,
                    "cascade_macro_ap": macro_ap(frame, cascade),
                    "macro_ap_delta": macro_ap(frame, cascade) - original_ap,
                }
            )
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "routing_sweep.csv", index=False)
    pd.concat(score_parts, ignore_index=True).to_parquet(
        output_dir / "aligned_c1a_neural_scores.parquet", index=False, compression="zstd"
    )
    summary = {
        "status": "complete",
        "feature_seconds": feature_seconds,
        "catboost_prediction_seconds": prediction_seconds,
        "pairs": len(pairs),
        "unique_items": len(items),
        "threshold_source": "human-train C1A component-disjoint OOF quantiles",
        "thresholds": {str(key): value for key, value in thresholds.items()},
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
