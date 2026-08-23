"""Discover category-specific attributes whose value conflicts imply non-matches.

The script uses only human labels. It does not assign new labels: it ranks
attribute keys for later review by an openly licensed model and by humans.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd


SPACE_RE = re.compile(r"\s+")


def normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().replace("ё", "е")
    return SPACE_RE.sub(" ", text).strip()


def parse_attributes(raw: str) -> dict[str, str]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        return {}
    result: dict[str, str] = {}
    for raw_key, raw_value in parsed.items():
        key, value = normalize(raw_key), normalize(raw_value)
        if key and value:
            result[key] = value
    return result


def parse_product_text(raw: str) -> dict[str, str]:
    """Recover flat attributes from the baseline's serialized product text."""
    result: dict[str, str] = {}
    for line in str(raw).splitlines():
        if ":" not in line:
            continue
        raw_key, raw_value = line.split(":", 1)
        key, value = normalize(raw_key), normalize(raw_value)
        if key in {"категория", "название", "характеристики обрезаны"}:
            continue
        if key and value:
            result[key] = value
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank variant-defining attribute candidates from human labels"
    )
    parser.add_argument("--items", type=Path, default=Path("data/items_human.parquet"))
    parser.add_argument("--matches", type=Path, default=Path("data/matches.parquet"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/variant_attributes")
    )
    parser.add_argument("--min-conflicts", type=int, default=20)
    parser.add_argument("--min-negative-rate", type=float, default=0.8)
    parser.add_argument("--examples-per-key", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=50_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    available_columns = pd.read_parquet(args.items).columns.tolist()
    attribute_column = "attributes" if "attributes" in available_columns else "product_text"
    if attribute_column not in available_columns:
        raise ValueError("Items must contain either attributes or product_text")
    items = pd.read_parquet(
        args.items, columns=["id", "name", attribute_column, "category"]
    ).set_index("id", verify_integrity=True)
    matches = pd.read_parquet(args.matches, columns=["id1", "id2", "target"])

    stats: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"conflict_0": 0, "conflict_1": 0, "equal_0": 0, "equal_1": 0}
    )
    examples: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    category_counts = matches.assign(
        category=matches["id1"].map(items["category"])
    ).groupby("category")["target"].agg(["size", "sum"])

    for position, row in enumerate(matches.itertuples(index=False), start=1):
        left = items.loc[row.id1]
        right = items.loc[row.id2]
        if left["category"] != right["category"]:
            raise ValueError(f"Cross-category pair: {row.id1}, {row.id2}")
        category = str(left["category"])
        target = int(row.target)
        parser = parse_attributes if attribute_column == "attributes" else parse_product_text
        left_attributes = parser(left[attribute_column])
        right_attributes = parser(right[attribute_column])
        for key in left_attributes.keys() & right_attributes.keys():
            left_value, right_value = left_attributes[key], right_attributes[key]
            relation = "equal" if left_value == right_value else "conflict"
            stats[(category, key)][f"{relation}_{target}"] += 1
            if relation == "conflict" and len(examples[(category, key)]) < args.examples_per_key:
                examples[(category, key)].append(
                    {
                        "category": category,
                        "attribute": key,
                        "id1": int(row.id1),
                        "id2": int(row.id2),
                        "target": target,
                        "value1": left_value,
                        "value2": right_value,
                        "name1": str(left["name"]),
                        "name2": str(right["name"]),
                    }
                )
        if args.progress_every and position % args.progress_every == 0:
            print(f"Processed {position:,}/{len(matches):,} human pairs", flush=True)

    records = []
    for (category, key), counts in stats.items():
        conflicts = counts["conflict_0"] + counts["conflict_1"]
        equals = counts["equal_0"] + counts["equal_1"]
        if not conflicts:
            continue
        category_negative_rate = 1.0 - float(
            category_counts.loc[category, "sum"] / category_counts.loc[category, "size"]
        )
        negative_rate = counts["conflict_0"] / conflicts
        # Five pseudo-observations shrink small groups toward their category base rate.
        smoothed_negative_rate = (
            counts["conflict_0"] + 5.0 * category_negative_rate
        ) / (conflicts + 5.0)
        records.append(
            {
                "category": category,
                "attribute": key,
                **counts,
                "conflicts": conflicts,
                "equals": equals,
                "negative_rate_given_conflict": negative_rate,
                "category_negative_rate": category_negative_rate,
                "smoothed_negative_rate": smoothed_negative_rate,
                "negative_lift": smoothed_negative_rate - category_negative_rate,
                "candidate": conflicts >= args.min_conflicts
                and negative_rate >= args.min_negative_rate,
            }
        )

    candidates = pd.DataFrame(records).sort_values(
        ["candidate", "negative_lift", "conflicts"], ascending=[False, False, False]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output_dir / "attribute_candidates.csv", index=False)

    candidate_keys = set(
        candidates.loc[candidates["candidate"], ["category", "attribute"]]
        .itertuples(index=False, name=None)
    )
    with (args.output_dir / "candidate_examples.jsonl").open("w", encoding="utf-8") as stream:
        for key in sorted(candidate_keys):
            for example in examples[key]:
                stream.write(json.dumps(example, ensure_ascii=False) + "\n")

    summary = {
        "human_pairs": len(matches),
        "categories": int(category_counts.shape[0]),
        "category_attribute_rows": len(candidates),
        "candidate_rows": int(candidates["candidate"].sum()),
        "min_conflicts": args.min_conflicts,
        "min_negative_rate": args.min_negative_rate,
        "attribute_source": attribute_column,
        "outputs": {
            "candidates": str(args.output_dir / "attribute_candidates.csv"),
            "examples": str(args.output_dir / "candidate_examples.jsonl"),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
