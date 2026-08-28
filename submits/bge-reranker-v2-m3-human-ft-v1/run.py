#!/usr/bin/env python3
"""Offline full-coverage BGE reranker submission runtime."""

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

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "bge-reranker-v2-m3-human-ft-v1"
ATTRIBUTE_FREQUENCY_PATH = ROOT / "attribute_name_frequency.csv"
MAX_LENGTH = int(os.getenv("PM_MAX_LENGTH", "384"))
PAIR_BATCH_SIZE = int(os.getenv("PM_PAIR_BATCH_SIZE", "256"))


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


def load_pairs(
    items_path: Path,
    matches_path: Path,
    limit: int | None,
    started: float,
):
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
    cache_root = output_path.parent / ".bge-reranker-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(cache_root / "huggingface" / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "transformers")


def load_model(output_path: Path, started: float):
    configure_writable_cache(output_path)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    required = [
        MODEL_PATH / "config.json",
        MODEL_PATH / "model.safetensors",
        MODEL_PATH / "tokenizer_config.json",
    ]
    missing = [path.name for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"offline BGE checkpoint is incomplete: {missing}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        attn_implementation="sdpa",
    ).cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    log(f"Loaded BGE checkpoint on {torch.cuda.get_device_name(0)}", started)
    return torch, tokenizer, model


def model_scores(left, right, torch, tokenizer, model, started: float):
    import numpy as np

    scores = np.empty(len(left), dtype=np.float32)
    order = np.argsort(
        np.fromiter((len(a) + len(b) for a, b in zip(left, right)), dtype=np.int64),
        kind="stable",
    )
    inference_started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for start in range(0, len(left), PAIR_BATCH_SIZE):
            end = min(len(left), start + PAIR_BATCH_SIZE)
            indices = order[start:end]
            size = end - start
            batch_left = [left[index] for index in indices]
            batch_right = [right[index] for index in indices]
            encoded = tokenizer(
                batch_left + batch_right,
                batch_right + batch_left,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )
            encoded = {
                key: value.cuda(non_blocking=True) for key, value in encoded.items()
            }
            probability = torch.sigmoid(model(**encoded).logits.reshape(-1).float())
            symmetric = (probability[:size] + probability[size:]) * 0.5
            scores[indices] = symmetric.cpu().numpy()
            completed = end
            if completed == len(left) or completed % (PAIR_BATCH_SIZE * 20) == 0:
                log(f"Scored {completed:,}/{len(left):,} pairs", started)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    log(
        f"Inference complete: {len(left) / inference_seconds:.1f} pairs/s, "
        f"peak CUDA memory {peak_gib:.2f} GiB",
        started,
    )
    return scores


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
    log(f"Wrote {len(scores):,} predictions to {path}", started)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    id1, id2, left, right = load_pairs(
        args.items_path, args.matches_path, args.limit, started
    )
    torch, tokenizer, model = load_model(args.output_path, started)
    scores = model_scores(left, right, torch, tokenizer, model, started)
    write_output(args.output_path, id1, id2, scores, started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
