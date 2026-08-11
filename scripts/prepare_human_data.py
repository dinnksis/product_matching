from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline import component_split, serialize_product


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare human-labelled train/validation data")
    parser.add_argument("--items", type=Path, default=Path("data/items_human.parquet"))
    parser.add_argument("--matches", type=Path, default=Path("data/matches.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("prepared/human"))
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attribute-chars", type=int, default=6000)
    return parser.parse_args()


def category_report(frame: pd.DataFrame, items: pd.DataFrame) -> dict[str, dict[str, float]]:
    categories = items.set_index("id")["category"]
    report = frame.assign(category=frame["id1"].map(categories)).groupby("category")["target"].agg(["size", "mean"])
    return {
        str(category): {"pairs": int(row["size"]), "positive_rate": float(row["mean"])}
        for category, row in report.iterrows()
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    items = pd.read_parquet(args.items, columns=["id", "name", "attributes", "category"])
    matches = pd.read_parquet(args.matches, columns=["id1", "id2", "target"])
    if items["id"].duplicated().any():
        raise ValueError("Item ids must be unique")

    print(f"Serializing {len(items):,} products...")
    items = items.copy()
    items["product_text"] = items.apply(
        serialize_product, axis=1, max_attribute_chars=args.max_attribute_chars
    )
    train, validation, diagnostics = component_split(
        matches, validation_fraction=args.validation_fraction, seed=args.seed
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    items[["id", "name", "category", "product_text"]].to_parquet(
        args.output_dir / "items.parquet", index=False
    )
    train.to_parquet(args.output_dir / "train_pairs.parquet", index=False)
    validation.to_parquet(args.output_dir / "val_pairs.parquet", index=False)
    report = {
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "max_attribute_chars": args.max_attribute_chars,
        "elapsed_seconds": time.perf_counter() - started,
        "split": diagnostics.__dict__,
        "train_categories": category_report(train, items),
        "validation_categories": category_report(validation, items),
        "product_text_chars": {
            str(q): float(value)
            for q, value in items["product_text"].str.len().quantile([0.5, 0.9, 0.95, 0.99, 1.0]).items()
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
