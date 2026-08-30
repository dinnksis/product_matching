#!/usr/bin/env python3
"""Input serialization and one-way SentenceTransformers inference helpers."""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import ensemble_base as base
from data_pipeline import serialize_product

MAX_LENGTH = int(os.getenv("PM_MAX_LENGTH", "384"))
BATCH_SIZE = int(os.getenv("PM_BATCH_SIZE", "1024"))
ATTENTION_IMPLEMENTATION = os.getenv("PM_ATTENTION_IMPLEMENTATION", "sdpa")
MAX_ATTRIBUTE_CHARS = 6000


def load_inputs(items_path: Path, matches_path: Path, limit: int | None, started: float):
    pairs = pd.read_parquet(matches_path, columns=["id1", "id2"])
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        pairs = pairs.iloc[:limit].copy()
    pairs = pairs.reset_index(drop=True)
    if pairs.empty or pairs.isna().any().any():
        raise ValueError("matches parquet is empty or contains null IDs")

    items = pd.read_parquet(items_path, columns=["id", "name", "attributes", "category"])
    needed = pd.unique(pairs[["id1", "id2"]].to_numpy().reshape(-1))
    selected = items.loc[items["id"].isin(needed)].copy().reset_index(drop=True)
    if selected["id"].duplicated().any() or selected["id"].nunique() != len(needed):
        raise ValueError("items parquet does not contain every referenced ID exactly once")

    serialized = [
        serialize_product(
            {"category": category, "name": name, "attributes": attributes},
            max_attribute_chars=MAX_ATTRIBUTE_CHARS,
        )
        for name, attributes, category in selected[
            ["name", "attributes", "category"]
        ].itertuples(index=False, name=None)
    ]
    selected_ids = selected["id"].to_numpy()
    text_by_id = pd.Series(serialized, index=selected_ids)
    title_by_id = pd.Series(selected["name"].tolist(), index=selected_ids)
    category_by_id = pd.Series(selected["category"].astype(str).tolist(), index=selected_ids)
    left_ids, right_ids = pairs["id1"], pairs["id2"]
    left = text_by_id.loc[left_ids].tolist()
    right = text_by_id.loc[right_ids].tolist()
    left_titles = title_by_id.loc[left_ids].tolist()
    right_titles = title_by_id.loc[right_ids].tolist()
    left_categories = category_by_id.loc[left_ids].tolist()
    right_categories = category_by_id.loc[right_ids].tolist()
    if left_categories != right_categories:
        raise ValueError("cross-category pairs are not supported")
    base.log(
        f"Loaded {len(pairs):,} pairs and serialized {len(selected):,} unique items",
        started,
    )
    return pairs, left, right, left_titles, right_titles, left_categories


def load_cross_encoder(model_path: Path, label: str, torch, started: float):
    from sentence_transformers import CrossEncoder

    required = ("config.json", "tokenizer_config.json")
    missing = [
        name
        for name in required
        if not (model_path / name).is_file() or not (model_path / name).stat().st_size
    ]
    has_single = (model_path / "model.safetensors").is_file()
    has_shards = (model_path / "model.safetensors.index.json").is_file()
    if missing or not (has_single or has_shards):
        raise FileNotFoundError(
            f"offline {label} checkpoint is incomplete: missing={missing}, "
            f"has_safetensors={has_single}, has_sharded_safetensors={has_shards}"
        )
    load_started = time.perf_counter()
    model = CrossEncoder(
        str(model_path),
        device="cuda:0",
        local_files_only=True,
        max_length=MAX_LENGTH,
        model_kwargs={
            "torch_dtype": torch.float16,
            "attn_implementation": ATTENTION_IMPLEMENTATION,
        },
    )
    if label == "RuModernBERT":
        reference_compile = getattr(model.model.config, "reference_compile", None)
        if reference_compile is not False:
            raise RuntimeError(
                f"RuModernBERT reference_compile must be False, got {reference_compile!r}"
            )
        base.log("RuModernBERT torch.compile explicitly disabled", started)
    torch.cuda.synchronize()
    base.log(
        f"Loaded {label} via SentenceTransformers in "
        f"{time.perf_counter() - load_started:.1f}s",
        started,
    )
    return model


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def score_oneway(model, left: list[str], right: list[str], label: str, torch, started: float):
    pair_count = len(left)
    if pair_count != len(right):
        raise ValueError(f"{label} pair texts are misaligned")
    if not pair_count:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty.copy()
    order = np.argsort(
        np.fromiter((len(a) + len(b) for a, b in zip(left, right)), dtype=np.int64),
        kind="stable",
    )
    sorted_pairs = [(left[int(index)], right[int(index)]) for index in order]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    inference_started = time.perf_counter()
    with torch.inference_mode():
        raw = model.predict(
            sorted_pairs,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            activation_fn=torch.nn.Identity(),
            convert_to_numpy=True,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - inference_started
    sorted_logits = np.asarray(raw, dtype=np.float32).reshape(-1)
    if len(sorted_logits) != pair_count:
        raise RuntimeError(f"{label} returned an unexpected number of scores")
    logits = np.empty(pair_count, dtype=np.float32)
    logits[order] = sorted_logits
    base.log(
        f"{label}: {pair_count:,} one-way forwards in {elapsed:.1f}s "
        f"({pair_count / elapsed:.1f}/s), peak CUDA "
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB",
        started,
    )
    return sigmoid(logits), logits


def clear_cuda(torch) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
