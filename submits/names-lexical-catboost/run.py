#!/usr/bin/env python3
"""Fast names-only lexical CatBoost submission."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from rapidfuzz import fuzz


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "matching_model.cbm"
SPACE_RE = re.compile(r"\s+")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_path", "--items-path", "-i", required=True, type=Path)
    parser.add_argument("--matches_path", "--matches-path", "-m", required=True, type=Path)
    parser.add_argument("--output_path", "--output-path", "-o", required=True, type=Path)
    parser.add_argument("--limit", type=int, help="local smoke-test only")
    return parser.parse_args()


def clean(value: object) -> str:
    if value is None:
        return ""
    return SPACE_RE.sub(" ", str(value)).strip().casefold().replace("ё", "е")


def build_features(
    names: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    categories: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for left_position, right_position in zip(left, right):
        first = clean(names[left_position])
        second = clean(names[right_position])
        first_numbers = set(NUMBER_RE.findall(first))
        second_numbers = set(NUMBER_RE.findall(second))
        number_union = first_numbers | second_numbers
        longest = max(len(first), len(second))
        rows.append(
            (
                fuzz.ratio(first, second) / 100.0,
                fuzz.token_set_ratio(first, second) / 100.0,
                fuzz.token_sort_ratio(first, second) / 100.0,
                float(first == second),
                min(len(first), len(second)) / longest if longest else 1.0,
                len(first_numbers & second_numbers) / max(1, len(number_union)),
                float(bool(first_numbers) and bool(second_numbers)),
                abs(len(first) - len(second)),
            )
        )
    lexical = pd.DataFrame(
        rows,
        columns=[
            "name_ratio",
            "name_token_set_ratio",
            "name_token_sort_ratio",
            "name_exact",
            "name_length_ratio",
            "name_numeric_jaccard",
            "name_numbers_both",
            "name_length_delta",
        ],
        dtype=np.float32,
    )
    category = pd.get_dummies(
        pd.Series(categories, name="category"),
        prefix="category",
        dtype=np.float32,
    )
    return pd.concat([lexical, category], axis=1)


def align(features: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    unknown = set(features.columns) - set(feature_names)
    if unknown:
        raise RuntimeError(f"Unknown test categories/features: {sorted(unknown)}")
    for column in feature_names:
        if column not in features:
            features[column] = np.float32(0.0)
    return features.loc[:, feature_names]


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    matches = pd.read_parquet(args.matches_path, columns=["id1", "id2"])
    if args.limit:
        matches = matches.head(args.limit).copy()
    required_ids = pd.unique(matches[["id1", "id2"]].to_numpy().reshape(-1))
    items = pd.read_parquet(args.items_path, columns=["id", "name", "category"])
    items = items[items.id.isin(required_ids)].reset_index(drop=True)
    if len(items) != len(required_ids):
        raise ValueError("matches reference missing item IDs")

    lookup = pd.Series(np.arange(len(items), dtype=np.int32), index=items.id.to_numpy())
    left = lookup.loc[matches.id1].to_numpy()
    right = lookup.loc[matches.id2].to_numpy()
    names = items.name.fillna("").astype(str).to_numpy()
    categories = items.category.fillna("").astype(str).to_numpy()[left]
    print(
        f"[{time.perf_counter() - started:7.1f}s] Loaded {len(matches):,} pairs "
        f"and {len(items):,} required items",
        flush=True,
    )

    features = build_features(names, left, right, categories)
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    scores = model.predict_proba(align(features, model.feature_names_))[:, 1]
    if len(scores) != len(matches) or not np.isfinite(scores).all():
        raise RuntimeError("invalid predictions")

    output = matches[["id1", "id2"]].copy()
    output["predict"] = scores
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_path, index=False)
    print(
        f"[{time.perf_counter() - started:7.1f}s] Saved {len(output):,} predictions",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
