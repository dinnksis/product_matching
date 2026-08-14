"""Disk-backed preparation for full weak-label cross-encoder training.

The LLM corpus is deliberately too large for the ordinary pandas-based trainer:
materializing both product texts for every pair and then caching both pair
orientations multiplies memory and disk usage.  This module tokenizes every
referenced product once and stores pairs as two int32 item positions plus a
float32 soft target.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.minilm_serialization import (
    DEFAULT_VARIANT,
    VARIANTS,
    AttributeFrequency,
    normalize_text,
    parse_attributes,
    serialize_product,
)


CACHE_VERSION = 2
POSITION_DTYPE = np.int32
TOKEN_DTYPE = np.int32


def _json_line(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _path_signature(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _configuration_fingerprint(configuration: dict[str, Any]) -> str:
    payload = json.dumps(
        configuration,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def _iter_pair_batches(
    pair_paths: Sequence[Path],
    *,
    batch_size: int,
    max_pairs: int | None,
) -> Iterator[pa.RecordBatch]:
    emitted = 0
    for path in pair_paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            columns=["id1", "id2", "target"], batch_size=batch_size
        ):
            if max_pairs is not None:
                remaining = max_pairs - emitted
                if remaining <= 0:
                    return
                if len(batch) > remaining:
                    batch = batch.slice(0, remaining)
            emitted += len(batch)
            yield batch


def _pair_count(pair_paths: Sequence[Path], max_pairs: int | None) -> int:
    total = sum(pq.ParquetFile(path).metadata.num_rows for path in pair_paths)
    return total if max_pairs is None else min(total, max_pairs)


def _required_item_ids(
    pair_paths: Sequence[Path],
    *,
    batch_size: int,
    max_pairs: int | None,
) -> np.ndarray:
    endpoint_parts: list[np.ndarray] = []
    pairs = 0
    started = time.perf_counter()
    for batch in _iter_pair_batches(
        pair_paths, batch_size=batch_size, max_pairs=max_pairs
    ):
        endpoint_parts.append(
            batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=True)
        )
        endpoint_parts.append(
            batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=True)
        )
        pairs += len(batch)
        if pairs % (batch_size * 10) < len(batch):
            _json_line(
                {
                    "cache_stage": "collect_required_ids",
                    "pairs": pairs,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
    if not endpoint_parts:
        raise ValueError("No training pairs were found")
    endpoints = np.concatenate(endpoint_parts)
    del endpoint_parts
    required = np.unique(endpoints)
    _json_line(
        {
            "cache_stage": "required_ids_ready",
            "pairs": pairs,
            "unique_items": len(required),
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    return required


def _membership_positions(
    sorted_values: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(sorted_values, values)
    valid = positions < len(sorted_values)
    found = np.zeros(len(values), dtype=bool)
    if valid.any():
        found[valid] = sorted_values[positions[valid]] == values[valid]
    return positions, found


def balanced_prefix_lengths(first: int, second: int, budget: int) -> tuple[int, int]:
    """Return prefix lengths that retain both products under a pair budget."""
    if budget < 0:
        raise ValueError("Pair token budget must be non-negative")
    if first + second <= budget:
        return first, second
    first_keep = min(first, budget // 2)
    second_keep = min(second, budget // 2)
    remaining = budget - first_keep - second_keep
    if remaining:
        add_first = min(first - first_keep, remaining)
        first_keep += add_first
        remaining -= add_first
        second_keep += min(second - second_keep, remaining)
    return first_keep, second_keep


@dataclass(frozen=True)
class FullPairCache:
    directory: Path
    metadata: dict[str, Any]
    item_tokens: np.memmap
    item_offsets: np.ndarray
    left_positions: np.ndarray
    right_positions: np.ndarray
    targets: np.ndarray
    pair_lengths: np.ndarray

    @classmethod
    def load(cls, directory: Path) -> "FullPairCache":
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not metadata.get("complete"):
            raise ValueError(f"Incomplete full-pair cache: {directory}")
        return cls(
            directory=directory,
            metadata=metadata,
            item_tokens=np.memmap(
                directory / "item_tokens.i32", dtype=TOKEN_DTYPE, mode="r"
            ),
            item_offsets=np.load(directory / "item_offsets.npy", mmap_mode="r"),
            left_positions=np.load(
                directory / "left_positions.npy", mmap_mode="r"
            ),
            right_positions=np.load(
                directory / "right_positions.npy", mmap_mode="r"
            ),
            targets=np.load(directory / "targets.npy", mmap_mode="r"),
            pair_lengths=np.load(directory / "pair_lengths.npy", mmap_mode="r"),
        )

    @property
    def pair_count(self) -> int:
        return len(self.targets)

    @property
    def item_count(self) -> int:
        return len(self.item_offsets) - 1

    def tokens_for_item(self, position: int) -> np.ndarray:
        start = int(self.item_offsets[position])
        stop = int(self.item_offsets[position + 1])
        return self.item_tokens[start:stop]


def _cache_complete(directory: Path) -> bool:
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    required = (
        "item_tokens.i32",
        "item_offsets.npy",
        "sorted_item_ids.npy",
        "sorted_item_positions.npy",
        "left_positions.npy",
        "right_positions.npy",
        "targets.npy",
        "pair_lengths.npy",
        "attribute_name_frequency.csv",
        "frequent_attribute_names.json",
        "serialization.json",
    )
    return bool(metadata.get("complete")) and all(
        (directory / name).is_file() for name in required
    )


def _attribute_frequencies(
    *,
    item_paths: Sequence[Path],
    required_ids: np.ndarray,
    batch_size: int,
) -> list[AttributeFrequency]:
    """Count normalized keys on exactly the products referenced by train pairs."""
    item_support: Counter[str] = Counter()
    occurrence_count: Counter[str] = Counter()
    selected_items = 0
    input_rows = 0
    started = time.perf_counter()

    for item_path in item_paths:
        parquet = pq.ParquetFile(item_path)
        schema_names = set(parquet.schema_arrow.names)
        required_columns = {"id", "attributes"}
        if not required_columns.issubset(schema_names):
            raise ValueError(
                "Exact MiniLM serialization requires raw id/attributes columns; "
                f"unsupported schema in {item_path}: {sorted(schema_names)}"
            )
        for batch in parquet.iter_batches(
            columns=["id", "attributes"], batch_size=batch_size
        ):
            ids = batch.column(0).to_numpy(zero_copy_only=False).astype(
                np.int64, copy=False
            )
            _, selected = _membership_positions(required_ids, ids)
            input_rows += len(batch)
            if not selected.any():
                continue
            selected_indices = np.flatnonzero(selected)
            raw_attributes = batch.column(1).take(
                pa.array(selected_indices, type=pa.int64())
            ).to_pylist()
            for raw in raw_attributes:
                parsed = parse_attributes(raw)
                item_support.update({key for key, _ in parsed})
                occurrence_count.update(key for key, _ in parsed)
            selected_items += len(selected_indices)
            if selected_items % (batch_size * 20) < len(selected_indices):
                _json_line(
                    {
                        "cache_stage": "attribute_frequency",
                        "items": selected_items,
                        "required_items": len(required_ids),
                        "input_rows": input_rows,
                        "unique_keys": len(occurrence_count),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )

    if selected_items != len(required_ids):
        raise ValueError(
            "Cannot compute attribute order: item sources contain duplicates or "
            f"are missing endpoints (selected={selected_items}, required={len(required_ids)})"
        )
    return [
        AttributeFrequency(
            attribute_name=key,
            item_support=item_support[key],
            occurrences=occurrence_count[key],
        )
        for key in sorted(
            occurrence_count,
            key=lambda key: (-occurrence_count[key], -item_support[key], key),
        )
    ]


def _read_attribute_frequencies(path: Path) -> list[AttributeFrequency]:
    rows: list[AttributeFrequency] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = {"attribute_name", "item_support", "occurrences"}
        if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
            raise ValueError(
                f"Attribute-frequency CSV must contain {sorted(expected)}: {path}"
            )
        for raw in reader:
            key = str(raw["attribute_name"])
            if not key or key in seen:
                raise ValueError(f"Empty or duplicate attribute name in {path}: {key!r}")
            seen.add(key)
            rows.append(
                AttributeFrequency(
                    attribute_name=key,
                    item_support=int(raw["item_support"]),
                    occurrences=int(raw["occurrences"]),
                )
            )
    return rows


def _write_attribute_frequencies(
    path: Path,
    rows: Sequence[AttributeFrequency],
    frequent_keys: set[str],
) -> None:
    total = sum(row.occurrences for row in rows)
    cumulative = 0
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "attribute_name",
                "item_support",
                "occurrences",
                "cumulative_occurrence_coverage",
                "is_frequent",
            ),
        )
        writer.writeheader()
        for row in rows:
            cumulative += row.occurrences
            writer.writerow(
                {
                    "attribute_name": row.attribute_name,
                    "item_support": row.item_support,
                    "occurrences": row.occurrences,
                    "cumulative_occurrence_coverage": (
                        cumulative / total if total else 0.0
                    ),
                    "is_frequent": row.attribute_name in frequent_keys,
                }
            )


def _read_frequent_keys(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(
        isinstance(value, str) for value in payload
    ):
        raise ValueError(f"Frequent-key JSON must be a list of strings: {path}")
    normalized = {normalize_text(value) for value in payload}
    return {value for value in normalized if value}


def _tokenize_items(
    *,
    item_paths: Sequence[Path],
    required_ids: np.ndarray,
    tokenizer: Any,
    output: Path,
    product_token_limit: int,
    serialization_variant: str,
    frequent_keys: set[str],
    key_rank: dict[str, int],
    batch_size: int,
) -> dict[str, int | float]:
    item_ids = np.lib.format.open_memmap(
        output / "item_ids.npy",
        mode="w+",
        dtype=np.int64,
        shape=(len(required_ids),),
    )
    offsets = np.lib.format.open_memmap(
        output / "item_offsets.npy",
        mode="w+",
        dtype=np.int64,
        shape=(len(required_ids) + 1,),
    )
    offsets[0] = 0
    item_position = 0
    token_position = 0
    input_rows = 0
    started = time.perf_counter()
    token_path = output / "item_tokens.i32"

    with token_path.open("wb", buffering=16 * 1024 * 1024) as token_stream:
        for item_path in item_paths:
            parquet = pq.ParquetFile(item_path)
            schema_names = set(parquet.schema_arrow.names)
            raw_columns = {"id", "name", "attributes"}
            if raw_columns.issubset(schema_names):
                columns = ["id", "name", "attributes"]
            else:
                raise ValueError(
                    "Exact MiniLM serialization requires raw id/name/attributes "
                    f"columns; unsupported schema in {item_path}: {sorted(schema_names)}"
                )

            for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
                ids = batch.column(0).to_numpy(zero_copy_only=False).astype(
                    np.int64, copy=False
                )
                _, selected = _membership_positions(required_ids, ids)
                input_rows += len(batch)
                if not selected.any():
                    continue
                selected_indices = np.flatnonzero(selected)
                selected_batch = pa.Table.from_batches([batch]).take(
                    pa.array(selected_indices, type=pa.int64())
                )
                selected_ids = ids[selected_indices]
                names = selected_batch["name"].to_pylist()
                attributes = selected_batch["attributes"].to_pylist()
                texts = [
                    serialize_product(
                        name,
                        parse_attributes(raw_attributes),
                        serialization_variant,
                        frequent_keys,
                        key_rank,
                    )
                    for name, raw_attributes in zip(names, attributes)
                ]
                encoded = tokenizer(
                    texts,
                    add_special_tokens=False,
                    padding=False,
                    truncation=True,
                    max_length=product_token_limit,
                    return_attention_mask=False,
                )["input_ids"]
                lengths = np.fromiter(
                    (len(sequence) for sequence in encoded),
                    dtype=np.int64,
                    count=len(encoded),
                )
                flattened = np.fromiter(
                    chain.from_iterable(encoded),
                    dtype=TOKEN_DTYPE,
                    count=int(lengths.sum()),
                )
                stop = item_position + len(selected_ids)
                if stop > len(required_ids):
                    raise ValueError("Item sources contain duplicate required IDs")
                item_ids[item_position:stop] = selected_ids
                offsets[item_position + 1 : stop + 1] = token_position + np.cumsum(
                    lengths
                )
                flattened.tofile(token_stream)
                item_position = stop
                token_position += len(flattened)

                if item_position and item_position % (batch_size * 20) < len(selected_ids):
                    _json_line(
                        {
                            "cache_stage": "tokenize_items",
                            "items": item_position,
                            "required_items": len(required_ids),
                            "input_rows": input_rows,
                            "tokens": token_position,
                            "items_per_second": item_position
                            / (time.perf_counter() - started),
                        }
                    )

    item_ids.flush()
    offsets.flush()
    if item_position != len(required_ids):
        present = np.sort(np.asarray(item_ids[:item_position]))
        missing = np.setdiff1d(required_ids, present, assume_unique=False)
        raise ValueError(
            f"Item sources are missing {len(missing)} pair endpoints; "
            f"examples={missing[:10].tolist()}"
        )
    order = np.argsort(item_ids, kind="stable")
    sorted_ids = np.asarray(item_ids[order], dtype=np.int64)
    if len(sorted_ids) > 1 and (np.diff(sorted_ids) <= 0).any():
        raise ValueError("Item sources contain duplicate IDs")
    np.save(output / "sorted_item_ids.npy", sorted_ids)
    np.save(
        output / "sorted_item_positions.npy",
        order.astype(POSITION_DTYPE, copy=False),
    )
    del order, sorted_ids, item_ids, offsets
    elapsed = time.perf_counter() - started
    return {
        "items": item_position,
        "tokens": token_position,
        "input_rows": input_rows,
        "elapsed_seconds": elapsed,
    }


def _map_pairs(
    *,
    pair_paths: Sequence[Path],
    output: Path,
    max_pairs: int | None,
    batch_size: int,
    max_length: int,
    special_tokens: int,
) -> dict[str, Any]:
    pair_count = _pair_count(pair_paths, max_pairs)
    left_output = np.lib.format.open_memmap(
        output / "left_positions.npy",
        mode="w+",
        dtype=POSITION_DTYPE,
        shape=(pair_count,),
    )
    right_output = np.lib.format.open_memmap(
        output / "right_positions.npy",
        mode="w+",
        dtype=POSITION_DTYPE,
        shape=(pair_count,),
    )
    targets_output = np.lib.format.open_memmap(
        output / "targets.npy",
        mode="w+",
        dtype=np.float32,
        shape=(pair_count,),
    )
    lengths_output = np.lib.format.open_memmap(
        output / "pair_lengths.npy",
        mode="w+",
        dtype=np.int32,
        shape=(pair_count,),
    )
    sorted_ids = np.load(output / "sorted_item_ids.npy", mmap_mode="r")
    sorted_positions = np.load(
        output / "sorted_item_positions.npy", mmap_mode="r"
    )
    offsets = np.load(output / "item_offsets.npy", mmap_mode="r")
    written = 0
    target_counts: dict[str, int] = {}
    started = time.perf_counter()

    for batch in _iter_pair_batches(
        pair_paths, batch_size=batch_size, max_pairs=max_pairs
    ):
        left_ids = batch.column(0).to_numpy(zero_copy_only=False).astype(
            np.int64, copy=False
        )
        right_ids = batch.column(1).to_numpy(zero_copy_only=False).astype(
            np.int64, copy=False
        )
        targets = batch.column(2).to_numpy(zero_copy_only=False).astype(
            np.float32, copy=False
        )
        if not np.isfinite(targets).all() or not (
            (targets >= 0.0) & (targets <= 1.0)
        ).all():
            raise ValueError("Pair targets must be finite soft labels in [0, 1]")
        left_indices, left_found = _membership_positions(sorted_ids, left_ids)
        right_indices, right_found = _membership_positions(sorted_ids, right_ids)
        if not left_found.all() or not right_found.all():
            raise RuntimeError("Pair-to-item mapping is incomplete")
        left_positions = sorted_positions[left_indices].astype(
            POSITION_DTYPE, copy=False
        )
        right_positions = sorted_positions[right_indices].astype(
            POSITION_DTYPE, copy=False
        )
        left_lengths = offsets[left_positions + 1] - offsets[left_positions]
        right_lengths = offsets[right_positions + 1] - offsets[right_positions]
        pair_lengths = np.minimum(
            max_length, left_lengths + right_lengths + special_tokens
        ).astype(np.int32, copy=False)

        stop = written + len(batch)
        left_output[written:stop] = left_positions
        right_output[written:stop] = right_positions
        targets_output[written:stop] = targets
        lengths_output[written:stop] = pair_lengths
        values, counts = np.unique(targets.astype(np.float64), return_counts=True)
        for value, count in zip(values, counts):
            key = format(float(value), ".12g")
            target_counts[key] = target_counts.get(key, 0) + int(count)
        written = stop

        if written % (batch_size * 10) < len(batch):
            _json_line(
                {
                    "cache_stage": "map_pairs",
                    "pairs": written,
                    "total_pairs": pair_count,
                    "pairs_per_second": written
                    / (time.perf_counter() - started),
                }
            )

    for array in (left_output, right_output, targets_output, lengths_output):
        array.flush()
    if written != pair_count:
        raise RuntimeError(f"Expected {pair_count} pairs, wrote {written}")
    return {
        "pairs": written,
        "target_counts": dict(sorted(target_counts.items(), key=lambda item: float(item[0]))),
        "elapsed_seconds": time.perf_counter() - started,
    }


def build_full_pair_cache(
    *,
    item_paths: Sequence[Path],
    pair_paths: Sequence[Path],
    tokenizer: Any,
    model_name: str,
    cache_root: Path,
    max_length: int = 512,
    serialization_variant: str = DEFAULT_VARIANT,
    attribute_frequency_csv: Path | None = None,
    frequent_keys_json: Path | None = None,
    item_batch_size: int = 8192,
    pair_batch_size: int = 1_000_000,
    max_pairs: int | None = None,
    rebuild: bool = False,
) -> FullPairCache:
    """Build or reuse a compact item-token/pair-position mmap cache."""
    item_paths = tuple(Path(path) for path in item_paths)
    pair_paths = tuple(Path(path) for path in pair_paths)
    attribute_frequency_csv = (
        Path(attribute_frequency_csv) if attribute_frequency_csv is not None else None
    )
    frequent_keys_json = (
        Path(frequent_keys_json) if frequent_keys_json is not None else None
    )
    if not item_paths or not pair_paths:
        raise ValueError("At least one item file and one pair file are required")
    optional_paths = tuple(
        path
        for path in (attribute_frequency_csv, frequent_keys_json)
        if path is not None
    )
    for path in (*item_paths, *pair_paths, *optional_paths):
        if not path.is_file():
            raise FileNotFoundError(path)
    if serialization_variant not in VARIANTS:
        raise ValueError(
            f"Unknown serialization variant {serialization_variant!r}; choose {VARIANTS}"
        )
    if serialization_variant == "S3_HYBRID" and frequent_keys_json is None:
        raise ValueError("S3_HYBRID requires frequent_keys_json")
    if max_length <= 8 or item_batch_size <= 0 or pair_batch_size <= 0:
        raise ValueError("Cache sizes and max_length must be positive")
    if max_pairs is not None and max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=True))
    product_token_limit = max_length - special_tokens
    if product_token_limit <= 0:
        raise ValueError("max_length leaves no room for product tokens")

    configuration = {
        "version": CACHE_VERSION,
        "model": model_name,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_size": len(tokenizer) if hasattr(tokenizer, "__len__") else None,
        "max_length": max_length,
        "serialization_variant": serialization_variant,
        "attribute_frequency_csv": (
            _path_signature(attribute_frequency_csv)
            if attribute_frequency_csv is not None
            else None
        ),
        "frequent_keys_json": (
            _path_signature(frequent_keys_json)
            if frequent_keys_json is not None
            else None
        ),
        "item_batch_size": item_batch_size,
        "pair_batch_size": pair_batch_size,
        "max_pairs": max_pairs,
        "special_tokens": special_tokens,
        "items": [_path_signature(path) for path in item_paths],
        "pairs": [_path_signature(path) for path in pair_paths],
    }
    fingerprint = _configuration_fingerprint(configuration)
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / f"llm-full-{fingerprint}"
    if _cache_complete(destination) and not rebuild:
        _json_line({"cache_reused": str(destination), "fingerprint": fingerprint})
        return FullPairCache.load(destination)

    staging = cache_root / f".{destination.name}.building"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    started = time.perf_counter()
    try:
        required_ids = _required_item_ids(
            pair_paths, batch_size=pair_batch_size, max_pairs=max_pairs
        )
        np.save(staging / "required_item_ids.npy", required_ids)
        if attribute_frequency_csv is None:
            frequency_rows = _attribute_frequencies(
                item_paths=item_paths,
                required_ids=required_ids,
                batch_size=item_batch_size,
            )
            frequency_source = "computed_from_referenced_training_items"
        else:
            frequency_rows = _read_attribute_frequencies(attribute_frequency_csv)
            frequency_source = str(attribute_frequency_csv.resolve())
        frequent_keys = _read_frequent_keys(frequent_keys_json)
        key_rank = {
            row.attribute_name: rank for rank, row in enumerate(frequency_rows)
        }
        _write_attribute_frequencies(
            staging / "attribute_name_frequency.csv",
            frequency_rows,
            frequent_keys,
        )
        (staging / "frequent_attribute_names.json").write_text(
            json.dumps(sorted(frequent_keys), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        serialization_report = {
            "variant": serialization_variant,
            "format": "title then globally ranked attributes joined by '. '",
            "category_included": False,
            "attribute_frequency_source": frequency_source,
            "attribute_keys": len(key_rank),
            "frequent_attribute_keys": len(frequent_keys),
        }
        (staging / "serialization.json").write_text(
            json.dumps(serialization_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        item_report = _tokenize_items(
            item_paths=item_paths,
            required_ids=required_ids,
            tokenizer=tokenizer,
            output=staging,
            product_token_limit=product_token_limit,
            serialization_variant=serialization_variant,
            frequent_keys=frequent_keys,
            key_rank=key_rank,
            batch_size=item_batch_size,
        )
        del required_ids
        pair_report = _map_pairs(
            pair_paths=pair_paths,
            output=staging,
            max_pairs=max_pairs,
            batch_size=pair_batch_size,
            max_length=max_length,
            special_tokens=special_tokens,
        )
        metadata = {
            "complete": True,
            "fingerprint": fingerprint,
            "configuration": configuration,
            "serialization": serialization_report,
            "items": item_report,
            "pairs": pair_report,
            "total_elapsed_seconds": time.perf_counter() - started,
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # The required-ID list is useful while building but redundant once the
        # sorted lookup and mapped pair arrays exist.
        (staging / "required_item_ids.npy").unlink()
        (staging / "item_ids.npy").unlink()
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
        _json_line(
            {
                "cache_ready": str(destination),
                "fingerprint": fingerprint,
                "items": item_report["items"],
                "pairs": pair_report["pairs"],
                "elapsed_seconds": metadata["total_elapsed_seconds"],
            }
        )
        return FullPairCache.load(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
