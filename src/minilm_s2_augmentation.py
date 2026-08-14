"""Human-only preparation and diagnostics for the combined S2 augmentation test."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.serialization_ablation import (
    deterministic_subset,
    grouped_split_masks,
    normalize_text,
    parse_attributes,
    select_frequent_keys,
    serialize_product,
    stable_hash,
)
from src.validation_audit import lexical_pair_table, prepare_items


BASELINE = "A_BASELINE"
AUGMENTED = "B_SHUFFLE_SWAP"
RUNS = (BASELINE, AUGMENTED)


def serialize_values_shuffled(
    title: Any,
    attributes: Sequence[tuple[str, str]],
    *,
    seed: int,
) -> str:
    """Keep title first and shuffle intact attribute entries before dropping keys."""

    entries = list(attributes)
    if len(entries) > 1:
        np.random.default_rng(seed).shuffle(entries)
    fields = [normalize_text(title)]
    fields.extend(value for _, value in entries)
    return ". ".join(field.rstrip(". ") for field in fields if field).strip()


def shuffled_pair_texts(
    pairs: pd.DataFrame,
    items: pd.DataFrame,
    parsed_attributes: Sequence[Sequence[tuple[str, str]]],
    *,
    seed: int,
    epoch: int,
) -> pd.DataFrame:
    positions = pd.Series(np.arange(len(items), dtype=np.int32), index=items["id"].to_numpy())
    left = positions.loc[pairs["id1"]].to_numpy(dtype=np.int32)
    right = positions.loc[pairs["id2"]].to_numpy(dtype=np.int32)
    names = items["name"].to_numpy()
    categories = items["category"].astype(str).to_numpy()

    def side_text(pair_index: int, item_position: int, side: int) -> str:
        item_seed = stable_hash(f"{epoch}:{pair_index}:{side}", seed)
        return serialize_values_shuffled(
            names[item_position],
            parsed_attributes[item_position],
            seed=item_seed,
        )

    result = pairs[["id1", "id2"]].reset_index(drop=True).copy()
    result["product_text_1"] = [
        side_text(index, int(position), 0) for index, position in enumerate(left)
    ]
    result["product_text_2"] = [
        side_text(index, int(position), 1) for index, position in enumerate(right)
    ]
    result["category_1"] = categories[left]
    result["category_2"] = categories[right]
    if (result["category_1"] != result["category_2"]).any():
        raise ValueError("Cross-category pairs are not supported")
    return result


def _category_summary(
    pairs: pd.DataFrame, item_categories: pd.Series
) -> dict[str, dict[str, float]]:
    categories = item_categories.loc[pairs["id1"]].to_numpy()
    frame = pd.DataFrame({"category": categories, "target": pairs["target"].to_numpy()})
    return {
        str(category): {
            "pairs": int(len(part)),
            "positive_rate": float(part["target"].mean()),
        }
        for category, part in frame.groupby("category", sort=True)
    }


def prepare_combined_experiment(
    items_path: Path,
    matches_path: Path,
    output_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    items = pd.read_parquet(items_path, columns=["id", "name", "attributes", "category"])
    matches = pd.read_parquet(matches_path, columns=["id1", "id2", "target"])
    if not matches["target"].isin([0.0, 1.0]).all():
        raise ValueError("Human targets must be binary")
    parsed = [parse_attributes(raw) for raw in items["attributes"].tolist()]
    train_mask, validation_mask, signatures = grouped_split_masks(
        items,
        matches,
        float(config["validation_fraction"]),
        int(config["split_seed"]),
        parsed,
    )
    train_pool_indices = np.flatnonzero(train_mask)
    train_indices = deterministic_subset(
        train_pool_indices,
        int(config["train_subset_size"]),
        int(config["train_subset_seed"]),
    )
    validation_indices = np.flatnonzero(validation_mask)
    train_pairs = matches.iloc[train_indices].reset_index(drop=True)
    validation_pairs = matches.iloc[validation_indices].reset_index(drop=True)
    train_ids = set(train_pairs["id1"]) | set(train_pairs["id2"])
    validation_ids = set(validation_pairs["id1"]) | set(validation_pairs["id2"])
    if train_ids & validation_ids:
        raise RuntimeError("Grouped split leaked item IDs")
    positions = pd.Series(np.arange(len(items), dtype=np.int32), index=items["id"].to_numpy())
    train_positions = positions.loc[np.asarray(sorted(train_ids), dtype=np.int64)].to_numpy()
    _, frequency, frequency_summary = select_frequent_keys(
        (parsed[int(position)] for position in train_positions),
        config["hybrid_frequency"],
    )
    key_rank = {
        key: rank for rank, key in enumerate(frequency["attribute_name"].tolist())
    }
    required_ids = train_ids | validation_ids
    required_mask = items["id"].isin(required_ids).to_numpy()
    required_positions = np.flatnonzero(required_mask)
    deterministic_items = items.loc[required_mask, ["id", "category"]].reset_index(drop=True)
    deterministic_items["text_s2_values_only"] = [
        serialize_product(
            items.iloc[int(position)]["name"],
            parsed[int(position)],
            "S2_VALUES_ONLY",
            set(),
            key_rank,
        )
        for position in required_positions
    ]
    augmented = shuffled_pair_texts(
        train_pairs,
        items,
        parsed,
        seed=int(config["attribute_order_seed"]),
        epoch=0,
    )

    audit_items = prepare_items(items.loc[required_mask].reset_index(drop=True))
    validation_features = lexical_pair_table(audit_items, validation_pairs)
    threshold = float(config["hard_groups"]["high_name_similarity_threshold"])
    validation_groups = validation_features[["id1", "id2", "target", "category"]].copy()
    validation_groups["high_name_similarity"] = validation_features["name_token_set_ratio"].ge(threshold)
    validation_groups["critical_variant_conflict"] = validation_features[
        ["number_conflict", "measure_conflict", "model_conflict"]
    ].astype(bool).any(axis=1)
    validation_groups["hard_looking"] = (
        validation_groups["high_name_similarity"]
        | validation_groups["critical_variant_conflict"]
    )

    item_categories = items.set_index("id", verify_integrity=True)["category"]
    train_family = {signatures[int(position)] for position in train_positions if signatures[int(position)]}
    validation_positions = positions.loc[
        np.asarray(sorted(validation_ids), dtype=np.int64)
    ].to_numpy()
    validation_family = {
        signatures[int(position)]
        for position in validation_positions
        if signatures[int(position)]
    }
    family_overlap = train_family & validation_family
    if family_overlap:
        raise RuntimeError("Grouped split leaked family signatures")

    deterministic_items.to_parquet(output_dir / "items.parquet", index=False)
    train_pairs.to_parquet(output_dir / "train_pairs.parquet", index=False)
    validation_pairs.to_parquet(output_dir / "validation_pairs.parquet", index=False)
    augmented.to_parquet(output_dir / "train_augmented_texts_epoch0.parquet", index=False)
    validation_groups.to_parquet(output_dir / "validation_groups.parquet", index=False)
    frequency.to_csv(output_dir / "attribute_name_frequency.csv", index=False)
    report = {
        "human_labels_only": True,
        "serialization": "S2_VALUES_ONLY",
        "split": {
            "strategy": str(config["split_strategy"]),
            "seed": int(config["split_seed"]),
            "train_pool_pairs": int(train_mask.sum()),
            "train_pairs": len(train_pairs),
            "validation_pairs": len(validation_pairs),
            "train_items": len(train_ids),
            "validation_items": len(validation_ids),
            "overlapping_item_ids": 0,
            "overlapping_family_signatures": 0,
        },
        "frequency_threshold": frequency_summary,
        "normalization": {
            "implementation": "src.serialization_ablation.normalize_text",
            "same_as_serialization_ablation": True,
        },
        "attribute_shuffle": {
            "title_always_first": True,
            "entry_key_value_integrity": True,
            "keys_emitted_by_s2": False,
            "seed": int(config["attribute_order_seed"]),
            "epoch": 0,
        },
        "hard_groups": {
            column: {
                "support": int(validation_groups[column].sum()),
                "positive_rate": float(
                    validation_groups.loc[validation_groups[column], "target"].mean()
                ),
            }
            for column in (
                "high_name_similarity",
                "critical_variant_conflict",
                "hard_looking",
            )
        },
        "train_categories": _category_summary(train_pairs, item_categories),
        "validation_categories": _category_summary(validation_pairs, item_categories),
    }
    (output_dir / "preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def group_metrics(
    predictions: pd.DataFrame,
    validation_groups: pd.DataFrame,
) -> dict[str, Any]:
    keys = ["id1", "id2", "target"]
    group_columns = [
        "high_name_similarity",
        "critical_variant_conflict",
        "hard_looking",
    ]
    merged = predictions.merge(
        validation_groups[keys + group_columns],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if merged[group_columns].isna().any().any():
        raise RuntimeError("Validation hard-group join is incomplete")
    result: dict[str, Any] = {}
    positives = merged[merged["target"].eq(1)]
    result["positive_only"] = {
        "pr_auc": None,
        "reason": "PR-AUC is undefined on a subset containing only positive labels",
        "support": len(positives),
        "mean_score": float(positives["score"].mean()),
        "p10_score": float(positives["score"].quantile(0.10)),
    }
    for group in ("high_name_similarity", "critical_variant_conflict", "hard_looking"):
        part = merged[merged[group].astype(bool)]
        categories = {}
        for category, category_part in part.groupby("category", sort=True):
            if category_part["target"].nunique() < 2:
                continue
            categories[str(category)] = float(
                average_precision_score(category_part["target"], category_part["score"])
            )
        result[group] = {
            "support": len(part),
            "positive_rate": float(part["target"].mean()),
            "overall_average_precision": (
                float(average_precision_score(part["target"], part["score"]))
                if part["target"].nunique() == 2
                else None
            ),
            "macro_average_precision": (
                float(np.mean(list(categories.values()))) if categories else None
            ),
            "categories_with_both_labels": len(categories),
        }
    return result
