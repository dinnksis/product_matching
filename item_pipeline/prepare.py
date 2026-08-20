from __future__ import annotations

import heapq
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .normalization import (
    extract_subtype,
    parse_attributes,
    retrieval_text,
    stable_hash64,
    title_attribute_token_coverage,
    tokenize,
)


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npy(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as destination:
        np.save(destination, array)
    os.replace(temporary, path)


def build_exemplar_bank(
    items_path: Path,
    *,
    max_items_per_category: int,
    seed: int,
    limit_rows: int | None = None,
    progress_every: int = 1_000_000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Take the lowest stable hashes per category in one streaming pass."""
    if max_items_per_category < 1:
        raise ValueError("max_items_per_category must be positive")
    parquet = pq.ParquetFile(items_path)
    required = {"id", "name", "attributes", "category"}
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"{items_path} is missing columns: {sorted(missing)}")

    heaps: dict[str, list[tuple[int, int, tuple[Any, ...]]]] = defaultdict(list)
    category_counts: Counter[str] = Counter()
    subtype_counts: dict[str, Counter[str]] = defaultdict(Counter)
    invalid_attributes = 0
    processed = 0
    next_progress = progress_every
    started = time.perf_counter()

    for batch in parquet.iter_batches(
        batch_size=65_536,
        columns=["id", "name", "attributes", "category"],
    ):
        columns = batch.to_pydict()
        for item_id, name, raw_attributes, category in zip(
            columns["id"],
            columns["name"],
            columns["attributes"],
            columns["category"],
        ):
            if limit_rows is not None and processed >= limit_rows:
                break
            processed += 1
            category_text = str(category)
            category_counts[category_text] += 1
            try:
                attributes = parse_attributes(raw_attributes)
            except (TypeError, ValueError, json.JSONDecodeError):
                invalid_attributes += 1
                continue
            subtype = extract_subtype(str(name), attributes)
            subtype_counts[category_text][subtype] += 1
            priority = stable_hash64(seed, int(item_id))
            record = (
                int(item_id),
                str(name),
                str(raw_attributes),
                category_text,
                subtype,
                retrieval_text(str(name), attributes),
                int(priority),
            )
            entry = (-priority, -int(item_id), record)
            heap = heaps[category_text]
            if len(heap) < max_items_per_category:
                heapq.heappush(heap, entry)
            elif priority < -heap[0][0]:
                heapq.heapreplace(heap, entry)

            if processed >= next_progress:
                print(
                    f"prepare rows={processed:,} categories={len(category_counts)} "
                    f"bank={sum(len(value) for value in heaps.values()):,} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
                next_progress += progress_every
        if limit_rows is not None and processed >= limit_rows:
            break

    records = [entry[2] for heap in heaps.values() for entry in heap]
    bank = pd.DataFrame(
        records,
        columns=[
            "id",
            "name",
            "attributes",
            "category",
            "subtype",
            "retrieval_text",
            "sample_hash",
        ],
    ).sort_values(["category", "sample_hash", "id"], kind="stable", ignore_index=True)
    if bank["id"].duplicated().any():
        raise RuntimeError("Deterministic sampler produced duplicate item IDs")

    category_profile: dict[str, Any] = {}
    for category, part in bank.groupby("category", sort=True):
        attribute_counts: list[int] = []
        coverages: list[float] = []
        name_lengths: list[int] = []
        name_tokens: list[int] = []
        key_counts: Counter[str] = Counter()
        for row in part.itertuples(index=False):
            attributes = parse_attributes(row.attributes)
            attribute_counts.append(len(attributes))
            coverages.append(title_attribute_token_coverage(row.name, attributes))
            name_lengths.append(len(str(row.name)))
            name_tokens.append(len(tokenize(row.name)))
            key_counts.update(attributes.keys())

        def quantiles(values: list[float | int]) -> dict[str, float]:
            series = pd.Series(values, dtype=np.float64)
            levels = (
                ("p01", 0.01),
                ("p10", 0.10),
                ("p50", 0.50),
                ("p90", 0.90),
                ("p99", 0.99),
            )
            return {label: float(series.quantile(q)) for label, q in levels}

        category_profile[str(category)] = {
            "source_rows": int(category_counts[str(category)]),
            "bank_rows": int(len(part)),
            "source_unique_subtypes": int(len(subtype_counts[str(category)])),
            "bank_unique_subtypes": int(part["subtype"].nunique()),
            "top_source_subtypes": subtype_counts[str(category)].most_common(50),
            "top_bank_keys": key_counts.most_common(100),
            "name_chars": quantiles(name_lengths),
            "name_tokens": quantiles(name_tokens),
            "attribute_count": quantiles(attribute_counts),
            "title_attribute_token_coverage": quantiles(coverages),
        }

    source_stat = items_path.stat()
    profile = {
        "version": "item_exemplar_bank_v1",
        "source": {
            "path": str(items_path.resolve()),
            "bytes": int(source_stat.st_size),
            "rows_available": int(parquet.metadata.num_rows),
            "rows_scanned": int(processed),
        },
        "seed": int(seed),
        "max_items_per_category": int(max_items_per_category),
        "bank_rows": int(len(bank)),
        "invalid_attribute_rows": int(invalid_attributes),
        "categories": category_profile,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return bank, profile


def encode_bank(
    bank: pd.DataFrame,
    output_dir: Path,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 256,
    device: str | None = None,
    max_sequence_length: int = 256,
    local_files_only: bool = False,
) -> dict[str, Any]:
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "sentence-transformers is required for dense indexing; install "
            "item_pipeline/requirements.txt or pass --skip-embeddings"
        ) from error

    started = time.perf_counter()
    model = SentenceTransformer(
        model_name,
        device=device,
        local_files_only=local_files_only,
    )
    model.max_seq_length = max_sequence_length
    embeddings = model.encode(
        bank["retrieval_text"].astype(str).tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_npy(embeddings.astype(np.float16), output_dir / "embeddings.f16.npy")
    _atomic_npy(bank["id"].to_numpy(dtype=np.int64), output_dir / "embedding_ids.npy")
    return {
        "model": model_name,
        "dimension": int(embeddings.shape[1]),
        "normalized": True,
        "max_sequence_length": int(max_sequence_length),
        "batch_size": int(batch_size),
        "device": str(device or getattr(model, "device", "auto")),
        "local_files_only": bool(local_files_only),
        "elapsed_seconds": time.perf_counter() - started,
    }


def prepare_index(
    items_path: Path,
    output_dir: Path,
    *,
    max_items_per_category: int,
    seed: int,
    limit_rows: int | None,
    skip_embeddings: bool,
    embedding_model: str,
    embedding_batch_size: int,
    embedding_device: str | None,
    embedding_local_files_only: bool,
) -> dict[str, Any]:
    bank, profile = build_exemplar_bank(
        items_path,
        max_items_per_category=max_items_per_category,
        seed=seed,
        limit_rows=limit_rows,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(bank, output_dir / "exemplar_bank.parquet")
    profile["embedding"] = {"status": "building"} if not skip_embeddings else None
    _atomic_json(profile, output_dir / "profile.json")
    if not skip_embeddings:
        profile["embedding"] = encode_bank(
            bank,
            output_dir,
            model_name=embedding_model,
            batch_size=embedding_batch_size,
            device=embedding_device,
            local_files_only=embedding_local_files_only,
        )
    else:
        for stale_path in (
            output_dir / "embeddings.f16.npy",
            output_dir / "embedding_ids.npy",
        ):
            if stale_path.exists():
                stale_path.unlink()
        profile["embedding"] = None
    _atomic_json(profile, output_dir / "profile.json")
    return profile
