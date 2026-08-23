#!/usr/bin/env python3
"""Build an equal category × label train from human data plus weak LLM labels."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blended_data import (
    canonical_pair,
    category_balance_plan,
    select_llm_supplement,
)
from src.data_pipeline import component_split, serialize_product


ITEM_COLUMNS = ["id", "name", "attributes", "category"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-items", type=Path, default=Path("data/items_human.parquet"))
    parser.add_argument("--human-matches", type=Path, default=Path("data/matches.parquet"))
    parser.add_argument("--llm-items", type=Path, default=Path("data/llm_data/items.parquet"))
    parser.add_argument("--llm-matches", type=Path, default=Path("data/llm_data/matches_llm.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("prepared/mxbai_balanced"))
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--target-per-class", type=int)
    parser.add_argument("--llm-weight", type=float, default=0.35)
    parser.add_argument(
        "--balance-loss-mass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="make total sample_weight equal for every category-label group",
    )
    parser.add_argument("--max-attribute-chars", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def map_pair_category_codes(
    pairs: pd.DataFrame,
    llm_items_path: Path,
    categories: list[str],
) -> tuple[np.ndarray, dict[str, int]]:
    table = pq.read_table(llm_items_path, columns=["id", "category"])
    item_ids = table["id"].combine_chunks().to_numpy(zero_copy_only=False)
    if pd.Index(item_ids).has_duplicates:
        raise ValueError("LLM item ids must be unique")
    item_categories = table["category"].combine_chunks().to_pandas()
    category_to_code = {category: code for code, category in enumerate(categories)}
    item_codes = item_categories.map(category_to_code).to_numpy()
    if pd.isna(item_codes).any():
        unknown = sorted(set(item_categories[pd.isna(item_codes)].astype(str)))
        raise ValueError(f"Unknown LLM item categories: {unknown}")
    item_codes = item_codes.astype(np.int8)

    item_code_lookup = pd.Series(item_codes, index=item_ids)

    def lookup(ids: np.ndarray) -> np.ndarray:
        values = item_code_lookup.reindex(ids).to_numpy()
        if pd.isna(values).any():
            raise ValueError("Some LLM pair ids are absent from LLM items")
        return values.astype(np.int8)

    first_codes = lookup(pairs["id1"].to_numpy(dtype=np.int64))
    second_codes = lookup(pairs["id2"].to_numpy(dtype=np.int64))
    if not np.array_equal(first_codes, second_codes):
        raise ValueError("LLM data contains cross-category pairs")
    return first_codes, category_to_code


def serialize_frame(frame: pd.DataFrame, max_attribute_chars: int) -> pd.DataFrame:
    result = frame.copy()
    result["product_text"] = result.apply(
        serialize_product,
        axis=1,
        max_attribute_chars=max_attribute_chars,
    )
    return result[["id", "name", "category", "product_text"]]


def write_required_items(
    human_items: pd.DataFrame,
    llm_items_path: Path,
    required_ids: np.ndarray,
    output_path: Path,
    max_attribute_chars: int,
) -> dict[str, Any]:
    required_ids = np.unique(required_ids.astype(np.int64))
    human_mask = human_items["id"].isin(required_ids)
    selected_human = human_items.loc[human_mask, ITEM_COLUMNS]
    human_ids = selected_human["id"].to_numpy(dtype=np.int64)
    llm_required = np.setdiff1d(required_ids, human_ids, assume_unique=False)
    lengths: list[np.ndarray] = []
    written_ids: list[np.ndarray] = []
    writer: pq.ParquetWriter | None = None

    def write(frame: pd.DataFrame) -> None:
        nonlocal writer
        if frame.empty:
            return
        serialized = serialize_frame(frame, max_attribute_chars)
        lengths.append(serialized["product_text"].str.len().to_numpy(dtype=np.int32))
        written_ids.append(serialized["id"].to_numpy(dtype=np.int64))
        table = pa.Table.from_pandas(serialized, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
        writer.write_table(table)

    write(selected_human)
    llm_file = pq.ParquetFile(llm_items_path)
    for row_group in range(llm_file.metadata.num_row_groups):
        table = llm_file.read_row_group(row_group, columns=ITEM_COLUMNS)
        ids = table["id"].to_numpy(zero_copy_only=False)
        selected = np.flatnonzero(np.isin(ids, llm_required, assume_unique=True))
        if len(selected):
            write(table.take(pa.array(selected)).to_pandas())
    if writer is not None:
        writer.close()
    else:
        raise RuntimeError("No items were written")

    all_written_ids = np.concatenate(written_ids)
    missing = np.setdiff1d(required_ids, all_written_ids, assume_unique=False)
    if len(missing):
        raise ValueError(f"Could not find {len(missing)} required items")
    if len(np.unique(all_written_ids)) != len(all_written_ids):
        raise ValueError("Prepared items contain duplicate ids")
    all_lengths = np.concatenate(lengths)
    return {
        "items": len(all_written_ids),
        "human_items": len(human_ids),
        "llm_only_items": len(llm_required),
        "product_text_chars": {
            str(quantile): float(value)
            for quantile, value in zip(
                (0.5, 0.9, 0.95, 0.99, 1.0),
                np.quantile(all_lengths, (0.5, 0.9, 0.95, 0.99, 1.0)),
            )
        },
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    human_items = pd.read_parquet(args.human_items, columns=ITEM_COLUMNS)
    human_matches = pd.read_parquet(
        args.human_matches, columns=["id1", "id2", "target"]
    )
    if human_items["id"].duplicated().any():
        raise ValueError("Human item ids must be unique")
    human_train, validation, diagnostics = component_split(
        human_matches,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    category_lookup = human_items.set_index("id", verify_integrity=True)["category"]
    human_train_categories = human_train["id1"].map(category_lookup)
    if human_train_categories.isna().any():
        raise ValueError("Some human train ids are absent from human items")
    plan, target_per_class = category_balance_plan(
        human_train,
        human_train_categories,
        target_per_class=args.target_per_class,
    )

    llm_pairs = pd.read_parquet(
        args.llm_matches, columns=["id1", "id2", "target"]
    )
    category_codes, category_to_code = map_pair_category_codes(
        llm_pairs,
        args.llm_items,
        plan.index.tolist(),
    )
    validation_ids = set(validation["id1"]) | set(validation["id2"])
    human_pair_keys = {
        canonical_pair(first, second)
        for first, second in human_matches[["id1", "id2"]].itertuples(index=False)
    }
    supplement = select_llm_supplement(
        llm_pairs,
        category_codes,
        category_to_code,
        plan,
        forbidden_item_ids=validation_ids,
        forbidden_pairs=human_pair_keys,
        llm_weight=args.llm_weight,
        seed=args.seed,
    )
    del llm_pairs, category_codes

    human_train = human_train.copy()
    human_train["llm_target_raw"] = np.nan
    human_train["label_source"] = "human"
    human_train["sample_weight"] = np.float32(1.0)
    human_train["_category"] = human_train_categories.astype(str).to_numpy()
    final_train = pd.concat([human_train, supplement], ignore_index=True)
    final_labels = (final_train["target"] >= 0.5).astype(np.int8)
    if args.balance_loss_mass:
        group_indices = final_train.assign(_label=final_labels).groupby(
            ["_category", "_label"]
        ).indices
        group_mass = np.array(
            [
                final_train.iloc[positions]["sample_weight"].sum()
                for positions in group_indices.values()
            ],
            dtype=np.float64,
        )
        desired_mass = float(group_mass.mean())
        for positions, mass in zip(group_indices.values(), group_mass):
            weight_column = final_train.columns.get_loc("sample_weight")
            final_train.iloc[positions, weight_column] = (
                final_train.iloc[positions, weight_column]
                * desired_mass
                / float(mass)
            )
    rng = np.random.default_rng(args.seed)
    final_train = final_train.iloc[rng.permutation(len(final_train))].reset_index(drop=True)

    final_counts = (
        final_train.assign(_label=(final_train["target"] >= 0.5).astype(np.int8))
        .groupby(["_category", "_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1], fill_value=0)
    )
    if not (final_counts.to_numpy() == target_per_class).all():
        raise RuntimeError(f"Final train is not exactly balanced:\n{final_counts}")

    validation = validation.copy()
    validation["llm_target_raw"] = np.nan
    validation["label_source"] = "human"
    validation["sample_weight"] = np.float32(1.0)
    train_output = final_train.drop(columns="_category")
    train_output.to_parquet(args.output_dir / "train_pairs.parquet", index=False)
    validation.to_parquet(args.output_dir / "val_pairs.parquet", index=False)

    required_ids = pd.unique(
        pd.concat(
            [
                train_output["id1"],
                train_output["id2"],
                validation["id1"],
                validation["id2"],
            ],
            ignore_index=True,
        )
    )
    item_report = write_required_items(
        human_items,
        args.llm_items,
        required_ids,
        args.output_dir / "items.parquet",
        args.max_attribute_chars,
    )

    source_counts = train_output["label_source"].value_counts().to_dict()
    weighted_class_mass = (
        train_output.assign(_label=(train_output["target"] >= 0.5).astype(np.int8))
        .groupby("_label")["sample_weight"]
        .sum()
        .to_dict()
    )
    weighted_category_label_mass = (
        train_output.assign(
            _category=final_train["_category"].to_numpy(),
            _label=(train_output["target"] >= 0.5).astype(np.int8),
        )
        .groupby(["_category", "_label"])["sample_weight"]
        .sum()
    )
    source_weight_stats = (
        train_output.groupby("label_source")["sample_weight"]
        .agg(["min", "mean", "max", "sum"])
        .to_dict("index")
    )
    selected_raw_targets = (
        supplement["llm_target_raw"].value_counts().sort_index().to_dict()
    )
    category_rows: dict[str, Any] = {}
    for category in final_counts.index:
        category_rows[str(category)] = {
            **{key: int(value) for key, value in plan.loc[category].items()},
            "final_negative": int(final_counts.loc[category, 0]),
            "final_positive": int(final_counts.loc[category, 1]),
            "final_total": int(final_counts.loc[category].sum()),
        }
    train_ids = set(train_output["id1"]) | set(train_output["id2"])
    report = {
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "target_per_category_class": target_per_class,
        "target_per_category_total": target_per_class * 2,
        "categories": len(final_counts),
        "train_pairs": len(train_output),
        "human_train_pairs": int(source_counts.get("human", 0)),
        "llm_train_pairs": int(source_counts.get("llm", 0)),
        "validation_pairs": len(validation),
        "validation_source": "human_only",
        "llm_base_weight": args.llm_weight,
        "loss_mass_balanced_by_category_label": args.balance_loss_mass,
        "weighted_class_mass": {
            str(key): float(value) for key, value in weighted_class_mass.items()
        },
        "weighted_category_label_mass_min": float(
            weighted_category_label_mass.min()
        ),
        "weighted_category_label_mass_max": float(
            weighted_category_label_mass.max()
        ),
        "source_weight_stats": {
            str(source): {
                str(metric): float(value) for metric, value in metrics.items()
            }
            for source, metrics in source_weight_stats.items()
        },
        "selected_llm_raw_target_counts": {
            str(key): int(value) for key, value in selected_raw_targets.items()
        },
        "train_validation_item_overlap": len(train_ids & validation_ids),
        "human_split": diagnostics.__dict__,
        "items": item_report,
        "per_category": category_rows,
        "max_attribute_chars": args.max_attribute_chars,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if report["train_validation_item_overlap"]:
        raise RuntimeError("Prepared blended train leaks item ids into validation")
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
