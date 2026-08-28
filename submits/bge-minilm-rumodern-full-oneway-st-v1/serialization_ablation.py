"""Deterministic data preparation for the MiniLM serialization ablation."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


VARIANTS = ("S0_TITLE", "S1_KEY_VALUE", "S2_VALUES_ONLY", "S3_HYBRID")

_SPACE_RE = re.compile(r"\s+")
_NUMBER_UNIT_RE = re.compile(
    r"(?<![\w.-])(\d+(?:[.,]\d+)?)\s*"
    r"(терабайт(?:а|ов)?|тб|tb|гигабайт(?:а|ов)?|гб|gb|"
    r"мегабайт(?:а|ов)?|мб|mb|килобайт(?:а|ов)?|кб|kb|"
    r"килограмм(?:а|ов)?|кг|kg|миллиграмм(?:а|ов)?|мг|mg|грамм(?:а|ов)?|гр|г|g|"
    r"миллилитр(?:а|ов)?|мл|ml|литр(?:а|ов)?|л|l|"
    r"миллиметр(?:а|ов)?|мм|mm|сантиметр(?:а|ов)?|см|cm|"
    r"герц|гц|hz|киловатт(?:а|ов)?|квт|kw|ватт(?:а|ов)?|вт|w)\b",
    flags=re.IGNORECASE,
)
_UNIT_ALIASES = {
    "терабайт": "tb", "терабайта": "tb", "терабайтов": "tb", "тб": "tb", "tb": "tb",
    "гигабайт": "gb", "гигабайта": "gb", "гигабайтов": "gb", "гб": "gb", "gb": "gb",
    "мегабайт": "mb", "мегабайта": "mb", "мегабайтов": "mb", "мб": "mb", "mb": "mb",
    "килобайт": "kb", "килобайта": "kb", "килобайтов": "kb", "кб": "kb", "kb": "kb",
    "килограмм": "kg", "килограмма": "kg", "килограммов": "kg", "кг": "kg", "kg": "kg",
    "миллиграмм": "mg", "миллиграмма": "mg", "миллиграммов": "mg", "мг": "mg", "mg": "mg",
    "грамм": "g", "грамма": "g", "граммов": "g", "гр": "g", "г": "g", "g": "g",
    "миллилитр": "ml", "миллилитра": "ml", "миллилитров": "ml", "мл": "ml", "ml": "ml",
    "литр": "l", "литра": "l", "литров": "l", "л": "l", "l": "l",
    "миллиметр": "mm", "миллиметра": "mm", "миллиметров": "mm", "мм": "mm", "mm": "mm",
    "сантиметр": "cm", "сантиметра": "cm", "сантиметров": "cm", "см": "cm", "cm": "cm",
    "герц": "hz", "гц": "hz", "hz": "hz",
    "киловатт": "kw", "киловатта": "kw", "киловаттов": "kw", "квт": "kw", "kw": "kw",
    "ватт": "w", "ватта": "w", "ваттов": "w", "вт": "w", "w": "w",
}
_BRAND_KEY_RE = re.compile(r"(?:^|\b)(бренд|brand|марка|производитель)(?:\b|$)")
_MODEL_KEY_RE = re.compile(
    r"(?:модел|model|артикул|партномер|part\s*number|sku|код\s*(?:товара|модели)|mpn|oem)"
)
_ALNUM_ID_RE = re.compile(r"(?=[a-zа-я0-9-]*[a-zа-я])(?=[a-zа-я0-9-]*\d)[a-zа-я0-9]+(?:-[a-zа-я0-9]+)+|(?=[a-zа-я0-9]*[a-zа-я])(?=[a-zа-я0-9]*\d)[a-zа-я0-9]{5,}")


@dataclass(frozen=True)
class SplitSummary:
    strategy: str
    seed: int
    train_pool_pairs: int
    train_subset_pairs: int
    validation_pairs: int
    train_subset_items: int
    validation_items: int
    overlapping_item_ids: int
    overlapping_family_signatures: int


def normalize_text(value: Any) -> str:
    """Normalize safely while preserving digits, punctuation, and model codes."""

    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = unicodedata.normalize("NFKC", str(value)).casefold().replace("ё", "е")

    def unit_replacement(match: re.Match[str]) -> str:
        number = match.group(1).replace(",", ".")
        unit = _UNIT_ALIASES[match.group(2).casefold().replace("ё", "е")]
        return f"{number} {unit}"

    text = _NUMBER_UNIT_RE.sub(unit_replacement, text)
    return _SPACE_RE.sub(" ", text).strip()


def _flatten_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        result: list[str] = []
        for key in sorted(value, key=lambda item: normalize_text(item)):
            for nested in _flatten_value(value[key]):
                key_text = normalize_text(key)
                result.append(f"{key_text}: {nested}" if key_text else nested)
        return result
    if isinstance(value, (list, tuple, set)):
        result = []
        for nested in value:
            result.extend(_flatten_value(nested))
        return result
    text = normalize_text(value)
    return [text] if text else []


def parse_attributes(raw: Any) -> list[tuple[str, str]]:
    """Return normalized non-empty key/value pairs without inventing tokens."""

    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid attributes JSON: {raw[:120]}") from error
    else:
        data = raw
    if data is None:
        return []
    if isinstance(data, Mapping):
        pairs: list[tuple[str, str]] = []
        for key, value in data.items():
            key_text = normalize_text(key)
            if not key_text:
                continue
            pairs.extend((key_text, item) for item in _flatten_value(value) if item)
        return pairs
    if isinstance(data, (list, tuple)):
        pairs = []
        for entry in data:
            if isinstance(entry, Mapping):
                key = entry.get("key", entry.get("name"))
                value = entry.get("value")
                if key is not None and value is not None:
                    key_text = normalize_text(key)
                    pairs.extend((key_text, item) for item in _flatten_value(value) if key_text and item)
        return pairs
    raise ValueError("Attributes must be a JSON object or a list of key/value records")


def stable_hash(value: Any, seed: int) -> int:
    payload = f"{seed}\0{value}".encode("utf-8", errors="replace")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def family_signature(category: Any, title: Any, attributes: Sequence[tuple[str, str]]) -> str:
    brands = sorted({value for key, value in attributes if _BRAND_KEY_RE.search(key)})
    models = sorted({value for key, value in attributes if _MODEL_KEY_RE.search(key)})
    if not models:
        models = sorted(set(_ALNUM_ID_RE.findall(normalize_text(title))))
    if not models:
        return ""
    return "|".join((normalize_text(category), ",".join(brands), ",".join(models)))


def _union_find_groups(
    item_count: int,
    left: np.ndarray,
    right: np.ndarray,
    family_signatures: Sequence[str],
) -> np.ndarray:
    parent = np.arange(item_count, dtype=np.int32)
    size = np.ones(item_count, dtype=np.int32)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def union(first: int, second: int) -> None:
        root1, root2 = find(first), find(second)
        if root1 == root2:
            return
        if size[root1] < size[root2]:
            root1, root2 = root2, root1
        parent[root2] = root1
        size[root1] += size[root2]

    for first, second in zip(left, right):
        union(int(first), int(second))
    representatives: dict[str, int] = {}
    for position, signature in enumerate(family_signatures):
        if not signature:
            continue
        previous = representatives.setdefault(signature, position)
        if previous != position:
            union(position, previous)
    return np.fromiter((find(index) for index in range(item_count)), dtype=np.int32, count=item_count)


def grouped_split_masks(
    items: pd.DataFrame,
    matches: pd.DataFrame,
    validation_fraction: float,
    seed: int,
    parsed_attributes: Sequence[Sequence[tuple[str, str]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = pd.Series(np.arange(len(items), dtype=np.int32), index=items["id"].to_numpy())
    left = positions.loc[matches["id1"]].to_numpy(dtype=np.int32)
    right = positions.loc[matches["id2"]].to_numpy(dtype=np.int32)
    signatures = [
        family_signature(row.category, row.name, attributes)
        for row, attributes in zip(items.itertuples(index=False), parsed_attributes)
    ]
    roots = _union_find_groups(len(items), left, right, signatures)
    pair_roots = roots[left]
    unique_roots = np.unique(pair_roots)
    validation_roots = np.asarray(
        [root for root in unique_roots if stable_hash(int(root), seed) / 2**64 < validation_fraction],
        dtype=unique_roots.dtype,
    )
    validation_mask = np.isin(pair_roots, validation_roots)
    if not validation_mask.any() or validation_mask.all():
        raise RuntimeError("Grouped split is degenerate")
    return ~validation_mask, validation_mask, np.asarray(signatures, dtype=object)


def deterministic_subset(indices: np.ndarray, size: int, seed: int) -> np.ndarray:
    if size >= len(indices):
        return np.sort(indices)
    ordered = sorted((stable_hash(int(index), seed), int(index)) for index in indices)
    return np.sort(np.asarray([index for _, index in ordered[:size]], dtype=np.int64))


def select_frequent_keys(
    parsed_attributes: Iterable[Sequence[tuple[str, str]]],
    configuration: Mapping[str, Any],
) -> tuple[set[str], pd.DataFrame, dict[str, Any]]:
    item_support: Counter[str] = Counter()
    occurrence_count: Counter[str] = Counter()
    for attributes in parsed_attributes:
        item_support.update({key for key, _ in attributes})
        occurrence_count.update(key for key, _ in attributes)
    rows = [
        {"attribute_name": key, "item_support": item_support[key], "occurrences": count}
        for key, count in occurrence_count.items()
    ]
    table = pd.DataFrame(rows).sort_values(
        ["occurrences", "item_support", "attribute_name"], ascending=[False, False, True]
    ).reset_index(drop=True)
    if table.empty:
        raise ValueError("No non-empty attributes were found in train items")
    table["cumulative_occurrence_coverage"] = table["occurrences"].cumsum() / table["occurrences"].sum()
    target_coverage = float(configuration["target_coverage"])
    boundary = int(np.searchsorted(table["cumulative_occurrence_coverage"].to_numpy(), target_coverage, side="left"))
    boundary = min(boundary, len(table) - 1)
    automatic_threshold = int(table.iloc[boundary]["item_support"])
    threshold = max(int(configuration["minimum_item_support"]), automatic_threshold)
    candidates = table.loc[table["item_support"] >= threshold, "attribute_name"].tolist()
    maximum = int(configuration["maximum_frequent_keys"])
    frequent = set(candidates[:maximum])
    table["is_frequent"] = table["attribute_name"].isin(frequent)
    achieved = float(table.loc[table["is_frequent"], "occurrences"].sum() / table["occurrences"].sum())
    summary = {
        "strategy": str(configuration["strategy"]),
        "target_occurrence_coverage": target_coverage,
        "minimum_item_support": int(configuration["minimum_item_support"]),
        "automatic_item_support_threshold": automatic_threshold,
        "selected_item_support_threshold": threshold,
        "maximum_frequent_keys": maximum,
        "selected_frequent_keys": len(frequent),
        "achieved_occurrence_coverage": achieved,
        "unique_attribute_names": len(table),
        "total_attribute_occurrences": int(table["occurrences"].sum()),
    }
    return frequent, table, summary


def serialize_product(
    title: Any,
    attributes: Sequence[tuple[str, str]],
    variant: str,
    frequent_keys: set[str],
    key_rank: Mapping[str, int],
) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown serialization variant: {variant}")
    title_text = normalize_text(title)
    if variant == "S0_TITLE" or not attributes:
        return title_text
    ordered = sorted(attributes, key=lambda pair: (key_rank.get(pair[0], math.inf), pair[0], pair[1]))
    fields = [title_text] if title_text else []
    for key, value in ordered:
        if variant == "S1_KEY_VALUE" or (variant == "S3_HYBRID" and key in frequent_keys):
            fields.append(f"{key}: {value}")
        else:
            fields.append(value)
    return ". ".join(field.rstrip(". ") for field in fields if field).strip()


def _category_counts(pairs: pd.DataFrame, item_categories: pd.Series) -> dict[str, dict[str, float]]:
    categories = item_categories.loc[pairs["id1"]].to_numpy()
    frame = pd.DataFrame({"category": categories, "target": pairs["target"].to_numpy()})
    return {
        str(category): {"pairs": int(len(part)), "positive_rate": float(part["target"].mean())}
        for category, part in frame.groupby("category", sort=True)
    }


def prepare_ablation_data(
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
    subset_indices = deterministic_subset(
        train_pool_indices,
        int(config["train_subset_size"]),
        int(config["train_subset_seed"]),
    )
    validation_indices = np.flatnonzero(validation_mask)
    train_pairs = matches.iloc[subset_indices].reset_index(drop=True)
    validation_pairs = matches.iloc[validation_indices].reset_index(drop=True)
    train_ids = set(train_pairs["id1"]) | set(train_pairs["id2"])
    validation_ids = set(validation_pairs["id1"]) | set(validation_pairs["id2"])
    overlap = train_ids & validation_ids
    if overlap:
        raise RuntimeError(f"Grouped split leaked {len(overlap)} item ids")
    id_to_position = pd.Series(np.arange(len(items), dtype=np.int32), index=items["id"].to_numpy())
    train_positions = id_to_position.loc[np.asarray(sorted(train_ids), dtype=np.int64)].to_numpy()
    frequent_keys, frequency_table, frequency_summary = select_frequent_keys(
        (parsed[int(position)] for position in train_positions),
        config["hybrid_frequency"],
    )
    key_rank = {
        key: rank for rank, key in enumerate(frequency_table["attribute_name"].tolist())
    }
    required_ids = train_ids | validation_ids
    required_mask = items["id"].isin(required_ids).to_numpy()
    required_positions = np.flatnonzero(required_mask)
    prepared_items = items.loc[required_mask, ["id", "category"]].reset_index(drop=True)
    for variant in VARIANTS:
        prepared_items[f"text_{variant.lower()}"] = [
            serialize_product(
                items.iloc[int(position)]["name"],
                parsed[int(position)],
                variant,
                frequent_keys,
                key_rank,
            )
            for position in required_positions
        ]
    item_categories = items.set_index("id", verify_integrity=True)["category"]
    train_family = {signatures[int(position)] for position in train_positions if signatures[int(position)]}
    validation_positions = id_to_position.loc[np.asarray(sorted(validation_ids), dtype=np.int64)].to_numpy()
    validation_family = {signatures[int(position)] for position in validation_positions if signatures[int(position)]}
    split_summary = SplitSummary(
        strategy=str(config["split_strategy"]),
        seed=int(config["split_seed"]),
        train_pool_pairs=int(train_mask.sum()),
        train_subset_pairs=len(train_pairs),
        validation_pairs=len(validation_pairs),
        train_subset_items=len(train_ids),
        validation_items=len(validation_ids),
        overlapping_item_ids=0,
        overlapping_family_signatures=len(train_family & validation_family),
    )
    if split_summary.overlapping_family_signatures:
        raise RuntimeError("Grouped split leaked family signatures")
    prepared_items.to_parquet(output_dir / "items.parquet", index=False)
    train_pairs.to_parquet(output_dir / "train_pairs.parquet", index=False)
    validation_pairs.to_parquet(output_dir / "validation_pairs.parquet", index=False)
    frequency_table.to_csv(output_dir / "attribute_name_frequency.csv", index=False)
    (output_dir / "frequent_attribute_names.json").write_text(
        json.dumps(sorted(frequent_keys), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "split": asdict(split_summary),
        "frequency_threshold": frequency_summary,
        "train_categories": _category_counts(train_pairs, item_categories),
        "validation_categories": _category_counts(validation_pairs, item_categories),
        "normalization": {
            "unicode": "NFKC",
            "case": "casefold",
            "whitespace": "collapsed",
            "decimal_separator_in_measurements": ".",
            "unit_aliases": sorted(set(_UNIT_ALIASES.values())),
            "digits_removed": False,
            "model_code_punctuation_removed": False,
        },
    }
    (output_dir / "preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
