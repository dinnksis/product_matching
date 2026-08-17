#!/usr/bin/env python3
"""Prepare fixed S0/S2 texts for the three external human validation splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.serialization_ablation import parse_attributes, select_frequent_keys, serialize_product


def pair_ids(frame: pd.DataFrame) -> set[int]:
    return set(frame["id1"].astype("int64")) | set(frame["id2"].astype("int64"))


def validate_pairs(frame: pd.DataFrame, name: str, expected: int) -> None:
    required = {"id1", "id2", "target"}
    if required - set(frame):
        raise ValueError(f"{name} missing columns: {sorted(required - set(frame))}")
    if len(frame) != expected:
        raise ValueError(f"{name}: expected {expected} pairs, got {len(frame)}")
    if not frame["target"].isin([0.0, 1.0]).all():
        raise ValueError(f"{name} contains non-binary labels")
    if (frame["id1"] == frame["id2"]).any():
        raise ValueError(f"{name} contains self-pairs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--iid", type=Path, required=True)
    parser.add_argument("--hard", type=Path, required=True)
    parser.add_argument("--ood", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    items = pd.read_parquet(args.items, columns=["id", "name", "attributes", "category"])
    if items["id"].duplicated().any():
        raise ValueError("items contain duplicate IDs")
    pairs = {
        "train": pd.read_parquet(args.train, columns=["id1", "id2", "target"]),
        "iid": pd.read_parquet(args.iid, columns=["id1", "id2", "target"]),
        "hard": pd.read_parquet(args.hard, columns=["id1", "id2", "target"]),
        "ood": pd.read_parquet(args.ood, columns=["id1", "id2", "target"]),
    }
    for name, frame in pairs.items():
        validate_pairs(frame, name, int(config["expected_pairs"][name]))

    ids = {name: pair_ids(frame) for name, frame in pairs.items()}
    overlap = {}
    names = list(ids)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            count = len(ids[left] & ids[right])
            overlap[f"{left}__{right}"] = count
            if count:
                raise RuntimeError(f"item leakage between {left} and {right}: {count}")
    required_ids = set().union(*ids.values())
    item_ids = set(items["id"].astype("int64"))
    missing = required_ids - item_ids
    if missing:
        raise ValueError(f"{len(missing)} pair item IDs are absent from items")

    parsed = [parse_attributes(raw) for raw in items["attributes"].tolist()]
    id_to_position = pd.Series(np.arange(len(items), dtype=np.int32), index=items["id"].to_numpy())
    train_positions = id_to_position.loc[np.asarray(sorted(ids["train"]), dtype=np.int64)].to_numpy()
    _, frequency_table, frequency_summary = select_frequent_keys(
        (parsed[int(position)] for position in train_positions),
        config["hybrid_frequency"],
    )
    key_rank = {
        key: rank for rank, key in enumerate(frequency_table["attribute_name"].tolist())
    }
    required_mask = items["id"].isin(required_ids).to_numpy()
    positions = np.flatnonzero(required_mask)
    prepared_items = items.loc[required_mask, ["id", "category"]].reset_index(drop=True)
    for variant in config["variants"]:
        prepared_items[f"text_{variant.lower()}"] = [
            serialize_product(
                items.iloc[int(position)]["name"],
                parsed[int(position)],
                variant,
                set(),
                key_rank,
            )
            for position in positions
        ]
    prepared_items.to_parquet(args.output_dir / "items.parquet", index=False)
    pairs["train"].to_parquet(args.output_dir / "train_pairs.parquet", index=False)
    # The shared trainer performs its initial validation on IID.
    pairs["iid"].to_parquet(args.output_dir / "validation_pairs.parquet", index=False)
    for split in ("iid", "hard", "ood"):
        pairs[split].to_parquet(args.output_dir / f"{split}_pairs.parquet", index=False)
    frequency_table.to_csv(args.output_dir / "attribute_name_frequency.csv", index=False)

    category_lookup = items.set_index("id", verify_integrity=True)["category"]
    report = {
        "experiment": config["experiment"],
        "source_items": len(items),
        "prepared_items": len(prepared_items),
        "pairs": {
            name: {
                "pairs": len(frame),
                "positives": int(frame["target"].sum()),
                "positive_rate": float(frame["target"].mean()),
                "categories": int(category_lookup.loc[frame["id1"]].nunique()),
            }
            for name, frame in pairs.items()
        },
        "item_overlap_counts": overlap,
        "frequency": frequency_summary,
        "variants": config["variants"],
    }
    (args.output_dir / "preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
