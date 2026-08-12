from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


# The order is intentionally semantic rather than frequency-based. A substring
# match covers variants such as "партномер (артикул производителя)".
PRIORITY_KEY_PARTS = (
    "бренд",
    "brand",
    "модель",
    "model",
    "артикул",
    "партномер",
    "part number",
    "sku",
    "код товара",
    "тип",
    "вид",
    "размер",
    "объем",
    "объём",
    "вес",
    "цвет",
    "материал",
    "комплектац",
)

_SPACE = re.compile(r"\s+")


def clean_field(value: Any) -> str:
    """Normalize whitespace without damaging model numbers or punctuation."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return _SPACE.sub(" ", str(value)).strip()


def _key_priority(key: str) -> tuple[int, str]:
    normalized = clean_field(key).casefold()
    for rank, part in enumerate(PRIORITY_KEY_PARTS):
        if part in normalized:
            return rank, normalized
    return len(PRIORITY_KEY_PARTS), normalized


def serialize_attributes(raw_attributes: str, max_chars: int | None = 6000) -> str:
    """Convert JSON attributes into deterministic, readable key/value lines.

    Character truncation is only a storage/safety guard. The final model input
    must additionally be truncated with its tokenizer.
    """
    try:
        attributes = json.loads(raw_attributes)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid attributes JSON: {str(raw_attributes)[:120]}") from error
    if not isinstance(attributes, dict):
        raise ValueError("Attributes JSON must contain an object")

    fields: list[str] = []
    for key, value in sorted(attributes.items(), key=lambda item: _key_priority(item[0])):
        key_text, value_text = clean_field(key), clean_field(value)
        if key_text and value_text:
            fields.append(f"{key_text}: {value_text}")
    text = "\n".join(fields)
    if max_chars is not None and len(text) > max_chars:
        text = (
            text[:max_chars].rsplit("\n", 1)[0].rstrip()
            + "\nХарактеристики обрезаны: да"
        )
    return text


def serialize_product(row: pd.Series, max_attribute_chars: int | None = 6000) -> str:
    """Serialize a product as one ``field: value`` record per line.

    Keeping the category, name and attributes in the same flat representation
    makes truncation predictable: identifiers and other high-priority
    attributes are emitted first by :func:`serialize_attributes`.
    """
    parts = [
        f"Категория: {clean_field(row['category'])}",
        f"Название: {clean_field(row['name'])}",
    ]
    attributes = serialize_attributes(row["attributes"], max_chars=max_attribute_chars)
    if attributes:
        parts.extend(attributes.splitlines())
    return "\n".join(parts)


def truncate_tokens(text: str, tokenizer: Any, max_tokens: int) -> str:
    """Tokenizer-aware truncation for the final per-product budget."""
    token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_tokens)
    return tokenizer.decode(token_ids, skip_special_tokens=True)


@dataclass(frozen=True)
class SplitDiagnostics:
    train_pairs: int
    validation_pairs: int
    train_items: int
    validation_items: int
    overlapping_items: int


def component_split(
    matches: pd.DataFrame, validation_fraction: float = 0.15, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, SplitDiagnostics]:
    """Split whole connected components so no product leaks across splits."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    all_ids = pd.unique(matches[["id1", "id2"]].to_numpy().reshape(-1))
    positions = pd.Series(np.arange(len(all_ids), dtype=np.int64), index=all_ids)
    left = positions.loc[matches["id1"]].to_numpy()
    right = positions.loc[matches["id2"]].to_numpy()
    parent = np.arange(len(all_ids), dtype=np.int64)
    size = np.ones(len(all_ids), dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for first, second in zip(left, right):
        root1, root2 = find(int(first)), find(int(second))
        if root1 == root2:
            continue
        if size[root1] < size[root2]:
            root1, root2 = root2, root1
        parent[root2] = root1
        size[root1] += size[root2]

    component = np.fromiter((find(int(node)) for node in left), dtype=np.int64, count=len(left))
    unique_components = np.unique(component)
    rng = np.random.default_rng(seed)
    validation_components = unique_components[
        rng.random(len(unique_components)) < validation_fraction
    ]
    is_validation = np.isin(component, validation_components)
    train = matches.loc[~is_validation].reset_index(drop=True)
    validation = matches.loc[is_validation].reset_index(drop=True)

    train_ids = set(train["id1"]) | set(train["id2"])
    validation_ids = set(validation["id1"]) | set(validation["id2"])
    diagnostics = SplitDiagnostics(
        train_pairs=len(train),
        validation_pairs=len(validation),
        train_items=len(train_ids),
        validation_items=len(validation_ids),
        overlapping_items=len(train_ids & validation_ids),
    )
    if diagnostics.overlapping_items:
        raise RuntimeError("Internal error: item leakage in component split")
    return train, validation, diagnostics


def attach_item_fields(
    pairs: pd.DataFrame, items: pd.DataFrame, fields: Iterable[str] = ("name", "category")
) -> pd.DataFrame:
    fields = list(fields)
    lookup = items.set_index("id", verify_integrity=True)[fields]
    left = lookup.reindex(pairs["id1"].to_numpy()).add_suffix("_1")
    right = lookup.reindex(pairs["id2"].to_numpy()).add_suffix("_2")
    left.index, right.index = pairs.index, pairs.index
    result = pd.concat([pairs, left, right], axis=1)
    if result[[f"{field}_{side}" for field in fields for side in (1, 2)]].isna().any().any():
        raise ValueError("Some pair ids are absent from items")
    return result
