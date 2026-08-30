"""Prepare a label-isolated atomic-difference pilot from the current human train."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from scripts.create_qwen_extraction_pilot import (  # noqa: E402
    SAMPLING_FLAGS,
    attach_features,
    sample_pilot,
    stable_rank,
)
from src.product_matching.eda import load_human_data  # noqa: E402


DEFAULT_TRAIN = ROOT / "prepared" / "validation_splits_v1" / "human" / "train_pairs.parquet"
DEFAULT_CATEGORIES = (
    ROOT / "reports" / "rule_discovery_data_audit" / "category_distribution.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "data" / "qwen_atomic_difference_pilot_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--sampling-mode",
        choices=("balanced_scenarios", "all"),
        default="balanced_scenarios",
    )
    return parser.parse_args()


def stable_pair_ids(train: pd.DataFrame) -> pd.Series:
    values = [
        "rp_"
        + hashlib.sha256(
            f"human_train_v1|{int(row.id1)}|{int(row.id2)}|{int(row.target)}".encode()
        ).hexdigest()[:24]
        for row in train.itertuples(index=False)
    ]
    result = pd.Series(values, index=train.index, name="pair_id")
    if result.duplicated().any():
        raise RuntimeError("Stable pair IDs are not unique")
    return result


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(args.train.resolve(), columns=["id1", "id2", "target"])
    items, _ = load_human_data(args.data_dir.resolve())
    category_by_id = items.set_index("id", verify_integrity=True)["category"]
    train["category"] = train["id1"].map(category_by_id)
    category_b = train["id2"].map(category_by_id)
    if train["category"].isna().any() or category_b.isna().any():
        raise RuntimeError("Train contains product IDs absent from items_human")
    if train["category"].ne(category_b).any():
        raise RuntimeError("Train contains cross-category pairs")
    train.insert(0, "pair_id", stable_pair_ids(train))

    categories = pd.read_csv(args.categories.resolve())[
        ["category", "pairs", "positives", "empirical_size_band"]
    ]
    actual = (
        train.groupby("category", observed=True)["target"]
        .agg(pairs="size", positives="sum")
        .reset_index()
    )
    checked = categories.merge(actual, on="category", suffixes=("_expected", "_actual"))
    if len(checked) != train["category"].nunique() or not (
        checked["pairs_expected"].eq(checked["pairs_actual"])
        & checked["positives_expected"].eq(checked["positives_actual"])
    ).all():
        raise RuntimeError("Category audit does not match current human train")

    featured = attach_features(train, items)
    featured["stable_rank"] = featured["pair_id"].map(
        lambda pair_id: stable_rank(args.seed, str(pair_id))
    )
    pilot = sample_pilot(featured, categories, args.size, args.sampling_mode)

    input_columns = [
        "pair_id",
        "id1",
        "id2",
        "category",
        "title_a",
        "attributes_a_json",
        "title_b",
        "attributes_b_json",
    ]
    inputs = pilot[input_columns].rename(
        columns={"id1": "item_id_a", "id2": "item_id_b"}
    )
    labels = pilot[["pair_id", "target"]].rename(columns={"target": "human_label"})
    metadata_columns = [
        "pair_id",
        "category",
        "target",
        "attributes_count_a",
        "attributes_count_b",
        "detail_ratio",
        "title_similarity",
        "shared_raw_attribute_keys",
        "raw_shared_value_difference_count",
        *SAMPLING_FLAGS,
    ]
    metadata = pilot[metadata_columns].rename(columns={"target": "human_label"})
    metadata["sampling_tags"] = metadata.apply(
        lambda row: json.dumps(
            [flag for flag in SAMPLING_FLAGS if bool(row[flag])], ensure_ascii=False
        ),
        axis=1,
    )
    if {"target", "label", "human_label"} & set(inputs.columns):
        raise RuntimeError("Label leaked into Qwen input")

    inputs.to_parquet(output_dir / "pilot_inputs.parquet", index=False)
    labels.to_parquet(output_dir / "pilot_labels.parquet", index=False)
    metadata.to_parquet(output_dir / "pilot_sampling_metadata.parquet", index=False)
    summary = {
        "source_train": str(args.train.resolve()),
        "source_pairs": len(train),
        "pilot_pairs": len(inputs),
        "sampling_mode": args.sampling_mode,
        "categories": int(inputs["category"].nunique()),
        "labels_stored_separately": True,
        "label_counts": {
            str(int(key)): int(value) for key, value in labels["human_label"].value_counts().items()
        },
        "seed": args.seed,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
