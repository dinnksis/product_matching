#!/usr/bin/env python3
"""Offline BGE + MiniLM normalized-rank ensemble without fallbacks."""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("OMP_NUM_THREADS", "20")
os.environ.setdefault("MKL_NUM_THREADS", "20")
os.environ.setdefault("RAYON_NUM_THREADS", "20")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

ROOT = Path(__file__).resolve().parent
BGE_PATH = ROOT / "models" / "bge-reranker-v2-m3-human-ft-v1"
MINILM_PATH = ROOT / "models" / "minilm-5ep-human-ft-v1"
ATTRIBUTE_FREQUENCY_PATH = ROOT / "attribute_name_frequency.csv"
MAX_LENGTH = int(os.getenv("PM_MAX_LENGTH", "384"))
TOKENIZATION_BATCH_SIZE = int(os.getenv("PM_TOKENIZATION_BATCH_SIZE", "2048"))
BGE_PAIR_BATCH_SIZE = int(os.getenv("PM_BGE_PAIR_BATCH_SIZE", "2048"))
MINILM_PAIR_BATCH_SIZE = int(os.getenv("PM_MINILM_PAIR_BATCH_SIZE", "4096"))
BGE_PAIR_TOKEN_BUDGET = int(os.getenv("PM_BGE_PAIR_TOKEN_BUDGET", "786432"))
MINILM_PAIR_TOKEN_BUDGET = int(os.getenv("PM_MINILM_PAIR_TOKEN_BUDGET", "1572864"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items_path", "--items-path", "-i", required=True, type=Path)
    parser.add_argument("--matches_path", "--matches-path", "-m", required=True, type=Path)
    parser.add_argument("--output_path", "--output-path", "-o", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def log(message: str, started: float) -> None:
    print(f"[{time.perf_counter() - started:7.1f}s] {message}", flush=True)


def require_columns(names: list[str], required: set[str], label: str) -> None:
    missing = required.difference(names)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def attribute_rank() -> dict[str, int]:
    if not ATTRIBUTE_FREQUENCY_PATH.is_file():
        raise FileNotFoundError(ATTRIBUTE_FREQUENCY_PATH)
    with ATTRIBUTE_FREQUENCY_PATH.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if "attribute_name" not in (reader.fieldnames or []):
            raise ValueError("attribute frequency table has no attribute_name column")
        names = [row["attribute_name"] for row in reader]
    if len(names) != 24_916 or len(set(names)) != len(names):
        raise ValueError(f"unexpected attribute rank table: {len(names)} rows")
    return {name: rank for rank, name in enumerate(names)}


def load_pairs(items_path: Path, matches_path: Path, limit: int | None, started: float):
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from serialization_ablation import parse_attributes, serialize_product

    match_schema = pq.read_schema(matches_path)
    require_columns(match_schema.names, {"id1", "id2"}, "matches parquet")
    matches = pq.read_table(
        matches_path, columns=["id1", "id2"], memory_map=True, use_threads=True
    ).combine_chunks()
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        matches = matches.slice(0, limit)
    if matches.num_rows == 0:
        raise ValueError("matches parquet contains no rows")
    id1 = matches["id1"].chunk(0)
    id2 = matches["id2"].chunk(0)
    if id1.null_count or id2.null_count:
        raise ValueError("matches parquet contains null IDs")
    needed_ids = pc.unique(pa.concat_arrays([id1, id2]))

    item_schema = pq.read_schema(items_path)
    require_columns(item_schema.names, {"id", "name", "attributes"}, "items parquet")
    items = pq.read_table(
        items_path,
        columns=["id", "name", "attributes"],
        memory_map=True,
        use_threads=True,
    ).combine_chunks()
    selected = items.filter(pc.is_in(items["id"], value_set=needed_ids)).combine_chunks()
    if selected.num_rows != len(needed_ids):
        raise ValueError("items parquet does not contain every referenced ID")
    if pc.count_distinct(selected["id"]).as_py() != len(needed_ids):
        raise ValueError("items parquet contains duplicate referenced IDs")
    selected_ids = selected["id"].chunk(0)
    left_index = pc.index_in(id1, value_set=selected_ids)
    right_index = pc.index_in(id2, value_set=selected_ids)
    if left_index.null_count or right_index.null_count:
        raise ValueError("at least one match ID is absent from items parquet")

    ranks = attribute_rank()
    names = selected["name"].to_pylist()
    attributes = selected["attributes"].to_pylist()
    texts = [
        serialize_product(name, parse_attributes(raw), "S2_VALUES_ONLY", set(), ranks)
        for name, raw in zip(names, attributes)
    ]
    left_positions = left_index.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    right_positions = right_index.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    left = [texts[index] for index in left_positions]
    right = [texts[index] for index in right_positions]
    id1_values, id2_values = id1.to_pylist(), id2.to_pylist()
    log(
        f"Loaded {matches.num_rows:,} pairs and serialized {len(texts):,} unique items",
        started,
    )
    del items, selected, matches, needed_ids, selected_ids, texts, names, attributes
    gc.collect()
    return id1_values, id2_values, left, right


def configure_writable_cache(output_path: Path) -> None:
    cache_root = output_path.parent / ".bge-minilm-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(cache_root / "huggingface" / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "transformers")
    # ModernBERT may compile Triton kernels on its first inference.  Kaggle's
    # /root is read-only, so keep both compiler caches beside the writable CSV.
    os.environ["TRITON_CACHE_DIR"] = str(cache_root / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "torchinductor")
    os.environ["TORCH_COMPILE_DEBUG_DIR"] = str(cache_root / "torch_compile_debug")


def load_model(model_path: Path, torch, started: float, label: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    required = [
        model_path / "config.json",
        model_path / "model.safetensors",
        model_path / "tokenizer_config.json",
    ]
    missing = [path.name for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"offline {label} checkpoint is incomplete: {missing}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
        attn_implementation="sdpa",
    ).cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    log(f"Loaded {label} on {torch.cuda.get_device_name(0)}", started)
    return tokenizer, model


def model_scores(
    left,
    right,
    torch,
    tokenizer,
    model,
    batch_size: int,
    pair_token_budget: int,
    label: str,
    started: float,
):
    import numpy as np

    pair_count = len(left)
    scores = np.empty(pair_count, dtype=np.float32)
    combined_left = left + right
    combined_right = right + left
    encoded_lists: dict[str, list[list[int]]] = {}
    tokenization_started = time.perf_counter()
    for start in range(0, len(combined_left), TOKENIZATION_BATCH_SIZE):
        end = min(len(combined_left), start + TOKENIZATION_BATCH_SIZE)
        encoded = tokenizer(
            combined_left[start:end],
            combined_right[start:end],
            padding=False,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        for key, values in encoded.items():
            encoded_lists.setdefault(key, []).extend(values)
    tokenization_seconds = time.perf_counter() - tokenization_started
    del combined_left, combined_right
    lengths = np.fromiter(
        (
            max(
                len(encoded_lists["input_ids"][index]),
                len(encoded_lists["input_ids"][index + pair_count]),
            )
            for index in range(pair_count)
        ),
        dtype=np.int32,
        count=pair_count,
    )
    order = np.argsort(lengths, kind="stable")
    log(
        f"{label}: tokenized {pair_count * 2:,} directed pairs in "
        f"{tokenization_seconds:.1f}s",
        started,
    )

    if batch_size < 1 or pair_token_budget < MAX_LENGTH:
        raise ValueError(f"invalid {label} batching configuration")
    torch.cuda.synchronize()
    inference_started = time.perf_counter()
    completed = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        while completed < pair_count:
            candidate_end = min(pair_count, completed + batch_size)
            candidate_length = int(lengths[order[candidate_end - 1]])
            padded_length = min(MAX_LENGTH, ((candidate_length + 7) // 8) * 8)
            token_limited_batch = max(1, pair_token_budget // padded_length)
            end = min(pair_count, completed + min(batch_size, token_limited_batch))
            indices = order[completed:end]
            size = end - completed
            directed_indices = np.concatenate([indices, indices + pair_count])
            features = [
                {key: encoded_lists[key][int(index)] for key in encoded_lists}
                for index in directed_indices
            ]
            encoded = tokenizer.pad(
                features,
                padding=True,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )
            encoded = {
                key: value.pin_memory().cuda(non_blocking=True)
                for key, value in encoded.items()
            }
            probability = torch.sigmoid(model(**encoded).logits.reshape(-1).float())
            scores[indices] = ((probability[:size] + probability[size:]) * 0.5).cpu().numpy()
            completed = end
            if completed == pair_count or completed % max(batch_size * 10, 20_000) < size:
                log(f"{label}: scored {completed:,}/{pair_count:,} pairs", started)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - inference_started
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    log(
        f"{label}: inference {pair_count / elapsed:.1f} pairs/s, "
        f"tokenization {tokenization_seconds:.1f}s, peak CUDA memory {peak_gib:.2f} GiB",
        started,
    )
    del encoded_lists, lengths, order
    gc.collect()
    return scores


def normalized_rank(values):
    """Equivalent to pandas rank(method='average', ascending=True) / N."""
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = ((start + 1) + end) * 0.5
        start = end
    return ranks / len(values)


def write_output(path: Path, id1, id2, scores, started: float) -> None:
    if not (len(id1) == len(id2) == len(scores)):
        raise RuntimeError("output arrays have different lengths")
    if not all(math.isfinite(float(score)) for score in scores):
        raise RuntimeError("scores contain NaN or infinity")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["id1", "id2", "predict"])
        writer.writerows((a, b, float(score)) for a, b, score in zip(id1, id2, scores))
    os.replace(temporary, path)
    log(f"Wrote {len(scores):,} ensemble predictions to {path}", started)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    id1, id2, left, right = load_pairs(args.items_path, args.matches_path, args.limit, started)
    configure_writable_cache(args.output_path)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    log(
        f"CUDA device={torch.cuda.get_device_name(0)}, "
        f"memory={torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB",
        started,
    )

    bge_tokenizer, bge_model = load_model(BGE_PATH, torch, started, "BGE")
    bge_scores = model_scores(
        left,
        right,
        torch,
        bge_tokenizer,
        bge_model,
        BGE_PAIR_BATCH_SIZE,
        BGE_PAIR_TOKEN_BUDGET,
        "BGE",
        started,
    )
    del bge_model, bge_tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    minilm_tokenizer, minilm_model = load_model(MINILM_PATH, torch, started, "MiniLM")
    minilm_scores = model_scores(
        left,
        right,
        torch,
        minilm_tokenizer,
        minilm_model,
        MINILM_PAIR_BATCH_SIZE,
        MINILM_PAIR_TOKEN_BUDGET,
        "MiniLM",
        started,
    )
    ensemble = (normalized_rank(bge_scores) + normalized_rank(minilm_scores)) * 0.5
    write_output(args.output_path, id1, id2, ensemble, started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
