"""Reusable helpers for exploratory analysis of the human-labelled dataset.

The functions in this module deliberately avoid using ``target`` while building
features.  This keeps the lightweight baseline suitable for honest out-of-fold
evaluation and makes the same feature builder reusable at inference time.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold


NON_ALNUM_RE = re.compile(r"[^0-9a-zа-яё]+", re.IGNORECASE)
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
TOKEN_RE = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)
IDENTIFIER_KEY_RE = re.compile(
    r"(?:^|\b)(?:артикул|партномер|oem|oe-код|штрих.?код|код товара|"
    r"код производителя|sku)(?:\b|$)|^модель$",
    re.IGNORECASE,
)
BRAND_KEYS = ("бренд", "бренд в одежде и обуви", "производитель")


@dataclass(frozen=True)
class PairFeatureData:
    """Engineered features plus arrays needed for analysis and validation."""

    features: pd.DataFrame
    left_positions: np.ndarray
    right_positions: np.ndarray
    categories: np.ndarray
    targets: np.ndarray


def normalize_text(value: object) -> str:
    """Normalize marketplace text without destroying model/article tokens."""

    text = unicodedata.normalize("NFKC", str(value)).lower().replace("ё", "е")
    return " ".join(NON_ALNUM_RE.sub(" ", text).split())


def set_jaccard(left: set[str], right: set[str]) -> float:
    """Return Jaccard similarity, treating two empty sets as uninformative."""

    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


def load_human_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two parquet files currently used by the report."""

    data_dir = Path(data_dir)
    items = pd.read_parquet(data_dir / "items_human.parquet")
    matches = pd.read_parquet(data_dir / "matches.parquet")
    return items, matches


def validate_human_data(items: pd.DataFrame, matches: pd.DataFrame) -> pd.Series:
    """Calculate the main schema and referential-integrity checks."""

    expected_item_columns = {"id", "name", "attributes", "category"}
    expected_match_columns = {"id1", "id2", "target"}
    if set(items.columns) != expected_item_columns:
        raise ValueError(f"Unexpected item columns: {items.columns.tolist()}")
    if set(matches.columns) != expected_match_columns:
        raise ValueError(f"Unexpected match columns: {matches.columns.tolist()}")

    item_ids = pd.Index(items["id"])
    low = np.minimum(matches["id1"].to_numpy(), matches["id2"].to_numpy())
    high = np.maximum(matches["id1"].to_numpy(), matches["id2"].to_numpy())
    canonical_pairs = pd.MultiIndex.from_arrays([low, high])

    return pd.Series(
        {
            "items": len(items),
            "pairs": len(matches),
            "unique_item_ids": items["id"].nunique(),
            "item_null_cells": int(items.isna().sum().sum()),
            "pair_null_cells": int(matches.isna().sum().sum()),
            "duplicate_item_ids": int(items.duplicated("id").sum()),
            "duplicate_unordered_pairs": int(canonical_pairs.duplicated().sum()),
            "self_pairs": int((matches["id1"] == matches["id2"]).sum()),
            "missing_id1_references": int((~matches["id1"].isin(item_ids)).sum()),
            "missing_id2_references": int((~matches["id2"].isin(item_ids)).sum()),
            "target_values": tuple(sorted(matches["target"].unique().tolist())),
        },
        name="value",
    )


def pair_category_summary(
    items: pd.DataFrame, matches: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Return item/pair counts and label prevalence for each category."""

    category_by_id = items.set_index("id")["category"]
    left_category = matches["id1"].map(category_by_id)
    right_category = matches["id2"].map(category_by_id)
    cross_category_pairs = int((left_category != right_category).sum())

    pair_stats = (
        matches.assign(category=left_category)
        .groupby("category", observed=True)["target"]
        .agg(pairs="size", positives="sum", positive_rate="mean")
    )
    pair_stats["positives"] = pair_stats["positives"].astype(int)
    pair_stats["negatives"] = pair_stats["pairs"] - pair_stats["positives"]
    item_counts = items["category"].value_counts().rename("items")
    summary = item_counts.to_frame().join(pair_stats, how="outer")
    summary = summary.sort_values("positive_rate")
    return summary, left_category.to_numpy(), cross_category_pairs


def attribute_key_summary(
    items: pd.DataFrame, top_n: int = 20
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Parse JSON attributes and summarize key coverage overall/by category."""

    overall: Counter[str] = Counter()
    by_category: dict[str, Counter[str]] = {
        category: Counter() for category in items["category"].unique()
    }
    key_counts = np.empty(len(items), dtype=np.int16)
    invalid_json = np.zeros(len(items), dtype=bool)

    for index, (raw_attributes, category) in enumerate(
        zip(items["attributes"], items["category"])
    ):
        try:
            attributes = json.loads(raw_attributes)
        except (TypeError, json.JSONDecodeError):
            invalid_json[index] = True
            key_counts[index] = 0
            continue
        if not isinstance(attributes, dict):
            invalid_json[index] = True
            key_counts[index] = 0
            continue
        key_counts[index] = len(attributes)
        overall.update(attributes.keys())
        by_category[category].update(attributes.keys())

    top_keys = [key for key, _ in overall.most_common(top_n)]
    overall_table = pd.DataFrame(
        [
            {
                "key": key,
                "items": overall[key],
                "coverage": overall[key] / len(items),
            }
            for key in top_keys
        ]
    ).set_index("key")

    category_sizes = items["category"].value_counts()
    coverage_by_category = pd.DataFrame(
        {
            category: {
                key: counter[key] / category_sizes[category] for key in top_keys
            }
            for category, counter in by_category.items()
        }
    ).T
    coverage_by_category.index.name = "category"

    quality = pd.Series(
        {
            "invalid_json_rows": int(invalid_json.sum()),
            "empty_attribute_objects": int((key_counts == 0).sum()),
            "mean_keys_per_item": float(key_counts.mean()),
            "median_keys_per_item": float(np.median(key_counts)),
            "p95_keys_per_item": float(np.quantile(key_counts, 0.95)),
        },
        name="value",
    )
    return overall_table, coverage_by_category, quality


def _identifier_tokens(normalized_name: str, attributes: dict[str, object]) -> set[str]:
    values = [normalized_name]
    values.extend(
        str(value)
        for key, value in attributes.items()
        if IDENTIFIER_KEY_RE.search(key)
    )
    return {
        token
        for token in TOKEN_RE.findall(normalize_text(" ".join(values)))
        if len(token) >= 3 and any(character.isdigit() for character in token)
    }


def _brand(attributes: dict[str, object]) -> str:
    for key in BRAND_KEYS:
        value = normalize_text(attributes.get(key, ""))
        if value:
            return value
    return ""


def build_pair_features(
    items: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    progress_every: int | None = 100_000,
) -> PairFeatureData:
    """Build inexpensive pair features from names and structured attributes."""

    position_by_id = pd.Series(
        np.arange(len(items), dtype=np.int32), index=items["id"].to_numpy()
    )
    left_positions = position_by_id.loc[matches["id1"]].to_numpy()
    right_positions = position_by_id.loc[matches["id2"]].to_numpy()

    names = items["name"].to_numpy()
    raw_attributes = items["attributes"].to_numpy()
    item_categories = items["category"].astype("category")
    category_codes = item_categories.cat.codes.to_numpy()
    categories = item_categories.astype(str).to_numpy()[left_positions]
    targets = matches["target"].astype(np.int8).to_numpy()

    feature_names = [
        "name_ratio",
        "name_token_set_ratio",
        "name_token_sort_ratio",
        "name_exact",
        "name_length_ratio",
        "numeric_jaccard",
        "identifier_overlap",
        "brand_equal",
        "brand_conflict",
        "attr_key_jaccard",
        "attr_value_match_ratio",
        "attr_value_conflict_ratio",
        "attr_exact",
    ]
    feature_arrays = {
        name: np.zeros(len(matches), dtype=np.float32) for name in feature_names
    }
    shared_key_counts = np.zeros(len(matches), dtype=np.int16)
    matching_value_counts = np.zeros(len(matches), dtype=np.int16)

    for index, (left_position, right_position) in enumerate(
        zip(left_positions, right_positions)
    ):
        left_name = normalize_text(names[left_position])
        right_name = normalize_text(names[right_position])
        feature_arrays["name_ratio"][index] = fuzz.ratio(left_name, right_name) / 100
        feature_arrays["name_token_set_ratio"][index] = (
            fuzz.token_set_ratio(left_name, right_name) / 100
        )
        feature_arrays["name_token_sort_ratio"][index] = (
            fuzz.token_sort_ratio(left_name, right_name) / 100
        )
        feature_arrays["name_exact"][index] = left_name == right_name
        longest_name = max(len(left_name), len(right_name))
        feature_arrays["name_length_ratio"][index] = (
            min(len(left_name), len(right_name)) / longest_name
            if longest_name
            else 1.0
        )
        feature_arrays["numeric_jaccard"][index] = set_jaccard(
            set(NUMBER_RE.findall(left_name)), set(NUMBER_RE.findall(right_name))
        )

        left_attributes = json.loads(raw_attributes[left_position])
        right_attributes = json.loads(raw_attributes[right_position])
        left_keys = set(left_attributes)
        right_keys = set(right_attributes)
        shared_keys = left_keys & right_keys
        feature_arrays["attr_key_jaccard"][index] = set_jaccard(
            left_keys, right_keys
        )

        matching_values = 0
        conflicting_values = 0
        for key in shared_keys:
            left_value = normalize_text(left_attributes[key])
            right_value = normalize_text(right_attributes[key])
            if not left_value or not right_value:
                continue
            if left_value == right_value:
                matching_values += 1
            else:
                conflicting_values += 1
        compared_values = matching_values + conflicting_values
        feature_arrays["attr_value_match_ratio"][index] = (
            matching_values / compared_values if compared_values else 0.0
        )
        feature_arrays["attr_value_conflict_ratio"][index] = (
            conflicting_values / compared_values if compared_values else 0.0
        )
        shared_key_counts[index] = len(shared_keys)
        matching_value_counts[index] = matching_values
        feature_arrays["attr_exact"][index] = (
            raw_attributes[left_position] == raw_attributes[right_position]
        )

        left_brand = _brand(left_attributes)
        right_brand = _brand(right_attributes)
        feature_arrays["brand_equal"][index] = bool(
            left_brand and right_brand and left_brand == right_brand
        )
        feature_arrays["brand_conflict"][index] = bool(
            left_brand and right_brand and left_brand != right_brand
        )
        feature_arrays["identifier_overlap"][index] = bool(
            _identifier_tokens(left_name, left_attributes)
            & _identifier_tokens(right_name, right_attributes)
        )

        if progress_every and (index + 1) % progress_every == 0:
            print(f"Built features for {index + 1:,}/{len(matches):,} pairs")

    features = pd.DataFrame(feature_arrays)
    features["shared_attr_keys"] = shared_key_counts
    features["matching_attr_values"] = matching_value_counts
    features["category_code"] = category_codes[left_positions]
    return PairFeatureData(
        features=features,
        left_positions=left_positions,
        right_positions=right_positions,
        categories=categories,
        targets=targets,
    )


def macro_average_precision(
    targets: np.ndarray, scores: np.ndarray, categories: np.ndarray
) -> tuple[float, pd.Series]:
    """Calculate the competition metric and per-category AP values."""

    category_scores = {}
    for category in sorted(np.unique(categories)):
        mask = categories == category
        category_scores[category] = average_precision_score(
            targets[mask], scores[mask]
        )
    per_category = pd.Series(category_scores, name="average_precision")
    per_category.index.name = "category"
    return float(per_category.mean()), per_category


def univariate_feature_scores(pair_data: PairFeatureData) -> pd.DataFrame:
    """Measure how far each individual fast feature gets on its own."""

    inverse_features = {"brand_conflict", "attr_value_conflict_ratio"}
    records = []
    for column in pair_data.features.columns:
        if column == "category_code":
            continue
        scores = pair_data.features[column].to_numpy()
        if column in inverse_features:
            scores = -scores
        macro_ap, _ = macro_average_precision(
            pair_data.targets, scores, pair_data.categories
        )
        records.append(
            {
                "feature": column,
                "macro_ap": macro_ap,
                "overall_ap": average_precision_score(pair_data.targets, scores),
            }
        )
    return pd.DataFrame(records).set_index("feature").sort_values(
        "macro_ap", ascending=False
    )


def _component_groups(
    item_count: int,
    left_positions: np.ndarray,
    right_positions: np.ndarray,
    edge_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build disjoint-set components and return roots and root sizes."""

    parent = np.arange(item_count, dtype=np.int32)
    sizes = np.ones(item_count, dtype=np.int32)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if sizes[left_root] < sizes[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        sizes[left_root] += sizes[right_root]

    if edge_mask is None:
        edge_mask = np.ones(len(left_positions), dtype=bool)
    for left, right in zip(left_positions[edge_mask], right_positions[edge_mask]):
        union(int(left), int(right))

    roots = np.fromiter(
        (find(index) for index in range(item_count)),
        dtype=np.int32,
        count=item_count,
    )
    root_sizes = np.bincount(roots, minlength=item_count)
    return roots, root_sizes


def graph_summary(item_count: int, pair_data: PairFeatureData) -> tuple[pd.Series, np.ndarray]:
    """Summarize pair graph structure and produce leakage-safe CV groups."""

    roots, root_sizes = _component_groups(
        item_count, pair_data.left_positions, pair_data.right_positions
    )
    groups = roots[pair_data.left_positions]
    component_sizes = root_sizes[root_sizes > 0]

    positive_mask = pair_data.targets == 1
    positive_roots, positive_root_sizes = _component_groups(
        item_count,
        pair_data.left_positions,
        pair_data.right_positions,
        positive_mask,
    )
    positive_component_sizes = positive_root_sizes[positive_root_sizes > 1]
    negative_mask = ~positive_mask
    negative_inside_positive_component = int(
        (
            positive_roots[pair_data.left_positions[negative_mask]]
            == positive_roots[pair_data.right_positions[negative_mask]]
        ).sum()
    )

    item_degree = np.bincount(
        np.concatenate([pair_data.left_positions, pair_data.right_positions]),
        minlength=item_count,
    )
    summary = pd.Series(
        {
            "items_with_degree_one": int((item_degree == 1).sum()),
            "degree_one_share": float((item_degree == 1).mean()),
            "maximum_item_degree": int(item_degree.max()),
            "all_edge_components": int(len(component_sizes)),
            "median_all_edge_component_size": float(np.median(component_sizes)),
            "maximum_all_edge_component_size": int(component_sizes.max()),
            "positive_components": int(len(positive_component_sizes)),
            "maximum_positive_component_size": int(positive_component_sizes.max()),
            "negative_edges_inside_positive_components": negative_inside_positive_component,
        },
        name="value",
    )
    return summary, groups


def cross_validated_light_baseline(
    pair_data: PairFeatureData,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Evaluate a small CPU model with component-disjoint cross-validation."""

    feature_columns = pair_data.features.columns.tolist()
    category_column_index = feature_columns.index("category_code")
    matrix = pair_data.features.to_numpy(dtype=np.float32)
    targets = pair_data.targets
    out_of_fold = np.zeros(len(targets), dtype=np.float32)
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=42
    )
    fold_records = []

    for fold, (train_index, valid_index) in enumerate(
        splitter.split(matrix, targets, groups), start=1
    ):
        model = HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=2.0,
            categorical_features=[category_column_index],
            random_state=fold,
        )
        model.fit(matrix[train_index], targets[train_index])
        fold_scores = model.predict_proba(matrix[valid_index])[:, 1]
        out_of_fold[valid_index] = fold_scores
        fold_records.append(
            {
                "fold": fold,
                "train_pairs": len(train_index),
                "valid_pairs": len(valid_index),
                "overall_ap": average_precision_score(
                    targets[valid_index], fold_scores
                ),
            }
        )

    return out_of_fold, pd.DataFrame(fold_records).set_index("fold")


def hard_examples(
    items: pd.DataFrame,
    matches: pd.DataFrame,
    pair_data: PairFeatureData,
    scores: np.ndarray,
    *,
    count: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return confident OOF errors for qualitative review."""

    names = items["name"].to_numpy()
    negative_order = np.argsort(np.where(pair_data.targets == 0, scores, -1))[
        -count:
    ][::-1]
    positive_order = np.argsort(np.where(pair_data.targets == 1, scores, 2))[:count]

    def make_table(indices: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "id1": matches.iloc[indices]["id1"].to_numpy(),
                "id2": matches.iloc[indices]["id2"].to_numpy(),
                "category": pair_data.categories[indices],
                "target": pair_data.targets[indices],
                "oof_score": np.round(scores[indices], 4),
                "name1": names[pair_data.left_positions[indices]],
                "name2": names[pair_data.right_positions[indices]],
            }
        )

    return make_table(negative_order), make_table(positive_order)
