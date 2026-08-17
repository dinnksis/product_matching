"""Generate controlled hard negatives from positive human training pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd


def normalize(value: object) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", str(value)).casefold().replace("ё", "е").split()
    )


def fields(product_text: str) -> dict[str, tuple[str, str]]:
    result = {}
    for line in str(product_text).splitlines():
        if ":" not in line:
            continue
        raw_key, raw_value = line.split(":", 1)
        key = normalize(raw_key)
        if key not in {"категория", "название", "характеристики обрезаны"}:
            value = raw_value.strip()
            if key and value:
                result[key] = (raw_key.strip(), value)
    return result


def replace_value(
    product_text: str, name: str, key: str, old_value: str, new_value: str
) -> tuple[str, str]:
    lines = []
    replaced = False
    new_name = str(name)
    for line in str(product_text).splitlines():
        if ":" not in line:
            lines.append(line)
            continue
        raw_key, raw_value = line.split(":", 1)
        normalized_key = normalize(raw_key)
        if normalized_key == key:
            lines.append(f"{raw_key}: {new_value}")
            replaced = True
        elif normalized_key == "название":
            # Keep the serialized name consistent when the exact attribute value
            # is explicitly present there. Otherwise leave wording untouched.
            pattern = re.compile(re.escape(old_value), flags=re.IGNORECASE)
            changed = pattern.sub(lambda _match: new_value, raw_value.strip())
            lines.append(f"{raw_key}: {changed}")
            new_name = changed
        else:
            lines.append(line)
    if not replaced:
        raise ValueError(f"Attribute {key!r} was not found")
    return "\n".join(lines), new_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, default=Path("data/human_items.parquet"))
    parser.add_argument(
        "--pairs", type=Path, default=Path("data/human_train_pairs.parquet")
    )
    parser.add_argument(
        "--registry", type=Path,
        default=Path("reports/variant_attributes/qwen_validation.jsonl"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("prepared/hard_negatives_v1")
    )
    parser.add_argument("--max-pairs", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stable_index(parts: tuple[object, ...], size: int) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % size


def main() -> None:
    args = parse_args()
    approved: dict[str, set[str]] = defaultdict(set)
    with args.registry.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            verdict = row["qwen"]
            if verdict["decision"] == "variant_defining" and verdict["safe_to_mutate"]:
                approved[str(row["category"])].add(str(row["attribute"]))
    if not approved:
        raise ValueError("Qwen registry contains no approved attributes")

    items = pd.read_parquet(
        args.items, columns=["id", "name", "category", "product_text"]
    ).set_index("id", verify_integrity=True)
    pairs = pd.read_parquet(args.pairs, columns=["id1", "id2", "target"])
    positives = pairs[pairs["target"] == 1].copy()

    parsed: dict[int, dict[str, tuple[str, str]]] = {}
    value_catalog: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item_id, row in items.iterrows():
        category = str(row["category"])
        keys = approved.get(category)
        if not keys:
            continue
        item_fields = fields(row["product_text"])
        relevant = {key: item_fields[key] for key in keys & item_fields.keys()}
        if relevant:
            parsed[int(item_id)] = relevant
            for key, (_, value) in relevant.items():
                value_catalog[(category, key)].add(value)
    ordered_catalog = {
        key: sorted(values, key=lambda value: (normalize(value), value))
        for key, values in value_catalog.items()
        if len({normalize(value) for value in values}) >= 2
    }

    minimum_id = int(items.index.min())
    next_synthetic_id = min(-1, minimum_id - 1)
    generated_pairs = []
    generated_items = []
    transformations = []

    for pair in positives.itertuples(index=False):
        left = items.loc[pair.id1]
        right = items.loc[pair.id2]
        category = str(left["category"])
        if category != str(right["category"]):
            continue
        left_fields = parsed.get(int(pair.id1), {})
        right_fields = parsed.get(int(pair.id2), {})
        eligible = []
        for key in approved.get(category, set()) & left_fields.keys() & right_fields.keys():
            old_left = left_fields[key][1]
            old_right = right_fields[key][1]
            if normalize(old_left) != normalize(old_right):
                continue
            replacements = [
                value for value in ordered_catalog.get((category, key), [])
                if normalize(value) != normalize(old_right)
            ]
            if replacements:
                eligible.append((key, old_right, replacements))
        if not eligible:
            continue
        eligible.sort(key=lambda item: item[0])
        chosen = eligible[stable_index((args.seed, pair.id1, pair.id2), len(eligible))]
        key, old_value, replacements = chosen
        new_value = replacements[
            stable_index((args.seed, pair.id1, pair.id2, key), len(replacements))
        ]
        new_text, new_name = replace_value(
            str(right["product_text"]), str(right["name"]), key, old_value, new_value
        )
        synthetic_id = next_synthetic_id
        next_synthetic_id -= 1
        generated_items.append(
            {
                "id": synthetic_id,
                "name": new_name,
                "category": category,
                "product_text": new_text,
            }
        )
        generated_pairs.append(
            {
                "id1": int(pair.id1),
                "id2": synthetic_id,
                "target": 0.0,
                "sample_weight": 1.0,
                "label_source": "synthetic_hard_negative_qwen_registry_v1",
            }
        )
        transformations.append(
            {
                "synthetic_id": synthetic_id,
                "source_id": int(pair.id2),
                "parent_id1": int(pair.id1),
                "parent_id2": int(pair.id2),
                "category": category,
                "attribute": key,
                "old_value": old_value,
                "new_value": new_value,
            }
        )
        if len(generated_pairs) >= args.max_pairs:
            break

    output_pairs = pd.DataFrame(generated_pairs)
    output_items = pd.DataFrame(generated_items)
    output_transformations = pd.DataFrame(transformations)
    if output_pairs.empty:
        raise RuntimeError("No hard negatives could be generated")
    if output_pairs[["id1", "id2"]].duplicated().any():
        raise RuntimeError("Generated duplicate pairs")
    if output_items["id"].duplicated().any() or output_items["id"].isin(items.index).any():
        raise RuntimeError("Synthetic item ids are not unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_pairs.to_parquet(args.output_dir / "hard_negative_pairs.parquet", index=False)
    output_items.to_parquet(args.output_dir / "hard_negative_items.parquet", index=False)
    output_transformations.to_parquet(
        args.output_dir / "transformations.parquet", index=False
    )
    manifest = {
        "generator": "generate_hard_negatives.py",
        "seed": args.seed,
        "max_pairs": args.max_pairs,
        "approved_category_attributes": sum(map(len, approved.values())),
        "generated_pairs": len(output_pairs),
        "generated_items": len(output_items),
        "categories": output_pairs["id1"].map(items["category"]).value_counts().to_dict(),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
