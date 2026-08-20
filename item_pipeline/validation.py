from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from .normalization import (
    extract_subtype,
    json_dumps,
    normalize_text,
    output_text,
    parse_attributes,
    title_attribute_token_coverage,
)


IDENTITY_KEY_RE = re.compile(
    r"бренд|brand|модель|model|артикул|партномер|part.?number|sku|mpn|oem|код"
)


@dataclass(frozen=True)
class CandidateValidation:
    item: dict[str, Any]
    valid: bool
    reasons: list[str]
    metrics: dict[str, float]


def clean_generated_item(item: dict[str, Any], expected_keys: list[str]) -> dict[str, Any]:
    raw_attributes = item.get("attributes")
    if not isinstance(raw_attributes, dict):
        raw_attributes = {}
    attributes = {
        key: output_text(raw_attributes.get(key, "")) for key in expected_keys
    }
    return {
        "name": output_text(item.get("name", "")),
        "attributes": attributes,
        "category": str(item.get("category", "")),
    }


def validate_candidate(
    item: dict[str, Any],
    *,
    anchor: dict[str, Any],
    examples: list[dict[str, Any]],
    existing_normalized_names: set[str],
    min_changed_value_fraction: float = 0.35,
    max_example_name_similarity: float = 0.97,
    minimum_name_chars: int = 5,
    maximum_name_chars: int = 240,
) -> CandidateValidation:
    anchor_attributes = parse_attributes(anchor["attributes"])
    expected_keys = list(anchor_attributes)
    cleaned = clean_generated_item(item, expected_keys)
    attributes = cleaned["attributes"]
    reasons: list[str] = []

    if cleaned["category"] != str(anchor["category"]):
        reasons.append("category_mismatch")
    if set(item.get("attributes", {})) != set(expected_keys):
        reasons.append("attribute_schema_mismatch")
    if any(not value for value in attributes.values()):
        reasons.append("empty_attribute_value")
    anchor_subtype = str(anchor.get("subtype") or "")
    generated_subtype = extract_subtype(cleaned["name"], attributes)
    if anchor_subtype and not anchor_subtype.startswith("__title__:"):
        if generated_subtype != anchor_subtype:
            reasons.append("subtype_changed")
    if not minimum_name_chars <= len(cleaned["name"]) <= maximum_name_chars:
        reasons.append("name_length_out_of_bounds")

    changed = [
        key
        for key in expected_keys
        if normalize_text(attributes[key]) != normalize_text(anchor_attributes[key])
    ]
    changed_fraction = len(changed) / max(1, len(expected_keys))
    if changed_fraction < min_changed_value_fraction:
        reasons.append("too_few_values_changed")

    identity_keys = [key for key in expected_keys if IDENTITY_KEY_RE.search(normalize_text(key))]
    if identity_keys and not any(key in changed for key in identity_keys):
        reasons.append("identity_fields_unchanged")

    normalized_name = normalize_text(cleaned["name"])
    if normalized_name in existing_normalized_names:
        reasons.append("exact_name_copy")
    reference_names = [str(anchor["name"]), *[str(row["name"]) for row in examples]]
    maximum_similarity = max(
        (fuzz.token_set_ratio(cleaned["name"], name) / 100.0 for name in reference_names),
        default=0.0,
    )
    if maximum_similarity >= max_example_name_similarity:
        reasons.append("name_too_similar_to_example")

    metrics = {
        "changed_value_fraction": float(changed_fraction),
        "maximum_example_name_similarity": float(maximum_similarity),
        "title_attribute_token_coverage": float(
            title_attribute_token_coverage(cleaned["name"], attributes)
        ),
        "name_chars": float(len(cleaned["name"])),
        "attribute_count": float(len(attributes)),
    }
    return CandidateValidation(
        item=cleaned,
        valid=not reasons,
        reasons=reasons,
        metrics=metrics,
    )


def validate_generated_dataset(
    items_path: Path,
    *,
    reference_path: Path | None = None,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    items = pd.read_parquet(items_path)
    required = ["id", "name", "attributes", "category"]
    missing = set(required) - set(items.columns)
    if missing:
        raise ValueError(f"Generated items are missing columns: {sorted(missing)}")

    invalid_rows: list[dict[str, Any]] = []
    generated_keys: set[str] = set()
    attribute_counts: list[int] = []
    coverages: list[float] = []
    normalized_names: list[str] = []
    canonical_attributes: list[str] = []
    for position, row in enumerate(items.itertuples(index=False)):
        reasons: list[str] = []
        try:
            attributes = parse_attributes(row.attributes)
        except Exception as error:
            attributes = {}
            reasons.append(f"invalid_attributes:{type(error).__name__}")
        if not str(row.name).strip():
            reasons.append("empty_name")
        if not str(row.category).strip():
            reasons.append("empty_category")
        if not attributes:
            reasons.append("empty_attributes")
        if reasons:
            invalid_rows.append({"position": position, "id": int(row.id), "reasons": reasons})
        generated_keys.update(attributes)
        attribute_counts.append(len(attributes))
        coverages.append(title_attribute_token_coverage(str(row.name), attributes))
        normalized_names.append(normalize_text(row.name))
        canonical_attributes.append(json_dumps(attributes))

    reference_keys: set[str] = set()
    matching_reference_names: set[str] = set()
    if reference_path is not None:
        generated_name_set = set(normalized_names)
        parquet = pq.ParquetFile(reference_path)
        for batch in parquet.iter_batches(
            batch_size=65_536,
            columns=["name", "attributes"],
        ):
            columns = batch.to_pydict()
            for name, raw in zip(columns["name"], columns["attributes"]):
                normalized_reference_name = normalize_text(name)
                if normalized_reference_name in generated_name_set:
                    matching_reference_names.add(normalized_reference_name)
                reference_keys.update(parse_attributes(raw, keep_empty=True))

    id_duplicates = int(items["id"].duplicated(keep=False).sum())
    name_duplicates = int(pd.Series(normalized_names).duplicated(keep=False).sum())
    full_duplicates = int(
        pd.DataFrame(
            {
                "name": normalized_names,
                "attributes": canonical_attributes,
                "category": items["category"].astype(str).to_numpy(),
            }
        ).duplicated(keep=False).sum()
    )
    exact_reference_names = sum(name in matching_reference_names for name in normalized_names)

    metadata_alignment: dict[str, Any] | None = None
    if metadata_path is not None and metadata_path.exists():
        metadata = pd.read_parquet(metadata_path)
        metadata_alignment = {
            "rows": int(len(metadata)),
            "missing_item_ids": int(len(set(items["id"]) - set(metadata["id"]))),
            "extra_metadata_ids": int(len(set(metadata["id"]) - set(items["id"]))),
        }

    def quantiles(values: list[float | int]) -> dict[str, float]:
        if not values:
            return {}
        series = pd.Series(values, dtype=np.float64)
        return {
            key: float(series.quantile(q))
            for key, q in (("p10", 0.10), ("p50", 0.50), ("p90", 0.90))
        }

    structural_errors = len(invalid_rows) + id_duplicates + full_duplicates
    return {
        "version": "generated_items_validation_v1",
        "items_path": str(items_path.resolve()),
        "rows": int(len(items)),
        "categories": items["category"].value_counts().sort_index().to_dict(),
        "unique_attribute_keys": int(len(generated_keys)),
        "reference_attribute_keys": int(len(reference_keys)),
        "new_attribute_keys": sorted(generated_keys - reference_keys) if reference_path else [],
        "invalid_rows": invalid_rows[:100],
        "invalid_row_count": int(len(invalid_rows)),
        "duplicate_id_rows": id_duplicates,
        "duplicate_normalized_name_rows": name_duplicates,
        "duplicate_full_card_rows": full_duplicates,
        "exact_reference_name_rows": int(exact_reference_names),
        "attribute_count": quantiles(attribute_counts),
        "title_attribute_token_coverage": quantiles(coverages),
        "metadata_alignment": metadata_alignment,
        "valid": structural_errors == 0 and exact_reference_names == 0,
    }
