"""Create label-free, category-aware attribute ontology jobs from RULE_DISCOVERY."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = ROOT / "data" / "rule_discovery_split_v1" / "rule_discovery_pairs.parquet"
DEFAULT_ITEMS = ROOT / "data" / "items_human.parquet"
DEFAULT_OUTPUT = ROOT / "data" / "qwen_attribute_ontology_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare batched attribute ontology jobs.")
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--example-values", type=int, default=3)
    return parser.parse_args()


def parse_attributes(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    parsed = json.loads(str(value))
    return parsed if isinstance(parsed, dict) else {}


def short_value(value: Any, limit: int = 200) -> str:
    text = " ".join(str(value).split())
    return text[:limit]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if not 0 < args.coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    if args.batch_size < 1 or args.example_values < 1:
        raise ValueError("batch-size and example-values must be positive")
    pairs_path = args.pairs.resolve()
    items_path = args.items.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_parquet(pairs_path, columns=["id1", "id2"])
    discovery_ids = pd.Index(
        pd.unique(pd.concat([pairs["id1"], pairs["id2"]], ignore_index=True))
    )
    items = pd.read_parquet(items_path, columns=["id", "attributes", "category"])
    items = items[items["id"].isin(discovery_ids)].copy()
    if items["id"].duplicated().any():
        raise ValueError("items.id must be unique")
    missing_ids = len(discovery_ids.difference(pd.Index(items["id"])))
    if missing_ids:
        raise ValueError(f"Missing {missing_ids} RULE_DISCOVERY products in items")

    name_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    parsed_attributes: list[dict[str, Any]] = []
    for row in items.itertuples(index=False):
        attributes = parse_attributes(row.attributes)
        parsed_attributes.append(attributes)
        name_counts[str(row.category)].update(map(str, attributes))

    selected: set[tuple[str, str]] = set()
    category_stats: list[dict[str, Any]] = []
    for category in sorted(name_counts):
        counts = name_counts[category]
        total = sum(counts.values())
        running = 0
        selected_names: list[str] = []
        for raw_name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            selected.add((category, raw_name))
            selected_names.append(raw_name)
            running += count
            if running / max(1, total) >= args.coverage:
                break
        category_stats.append(
            {
                "category": category,
                "raw_attribute_names": len(counts),
                "selected_names": len(selected_names),
                "attribute_occurrences": total,
                "selected_occurrences": running,
                "realized_coverage": running / max(1, total),
            }
        )

    value_counts: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row, attributes in zip(items.itertuples(index=False), parsed_attributes):
        category = str(row.category)
        for raw_name, value in attributes.items():
            key = (category, str(raw_name))
            if key in selected:
                normalized_value = short_value(value)
                if normalized_value:
                    value_counts[key][normalized_value] += 1

    entries: list[dict[str, Any]] = []
    for category in sorted(name_counts):
        counts = name_counts[category]
        category_keys = sorted(
            (key for key in selected if key[0] == category),
            key=lambda key: (-counts[key[1]], key[1]),
        )
        for _, raw_name in category_keys:
            entry_id = "an_" + hashlib.sha256(
                f"{category}|{raw_name}".encode("utf-8")
            ).hexdigest()[:20]
            examples = [
                value
                for value, _ in value_counts[(category, raw_name)].most_common(
                    args.example_values
                )
            ]
            entries.append(
                {
                    "entry_id": entry_id,
                    "category": category,
                    "raw_attribute_name": raw_name,
                    "product_occurrences": int(counts[raw_name]),
                    "example_values": examples,
                }
            )

    batches: list[dict[str, Any]] = []
    by_category: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_category[entry["category"]].append(entry)
    for category in sorted(by_category):
        category_entries = by_category[category]
        for offset in range(0, len(category_entries), args.batch_size):
            chunk = category_entries[offset : offset + args.batch_size]
            batch_id = "ab_" + hashlib.sha256(
                (category + "|" + "|".join(row["entry_id"] for row in chunk)).encode("utf-8")
            ).hexdigest()[:20]
            batches.append(
                {
                    "batch_id": batch_id,
                    "category": category,
                    "attributes": [
                        {
                            "entry_id": row["entry_id"],
                            "raw_attribute_name": row["raw_attribute_name"],
                            "example_values": row["example_values"],
                        }
                        for row in chunk
                    ],
                }
            )

    entries_frame = pd.DataFrame(entries)
    stats_frame = pd.DataFrame(category_stats)
    entries_frame.to_parquet(output_dir / "ontology_entries.parquet", index=False)
    entries_frame.to_csv(output_dir / "ontology_entries.csv", index=False, encoding="utf-8-sig")
    stats_frame.to_csv(output_dir / "coverage_by_category.csv", index=False, encoding="utf-8-sig")
    write_jsonl(output_dir / "ontology_batches.jsonl", batches)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "RULE_DISCOVERY products only; no labels",
        "pairs": str(pairs_path),
        "pairs_sha256": sha256_file(pairs_path),
        "items": str(items_path),
        "items_sha256": sha256_file(items_path),
        "discovery_pairs": len(pairs),
        "discovery_products": len(items),
        "categories": len(category_stats),
        "coverage_target_per_category": args.coverage,
        "selected_category_attribute_names": len(entries),
        "batches": len(batches),
        "batch_size": args.batch_size,
        "labels_used": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
