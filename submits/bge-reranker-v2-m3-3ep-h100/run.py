#!/usr/bin/env python3
"""Fast offline H100 inference for the trained BGE product cross-encoder."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any


# Set every cache/thread option before importing Torch, Transformers or PyArrow.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("OMP_NUM_THREADS", "20")
os.environ.setdefault("MKL_NUM_THREADS", "20")
os.environ.setdefault("RAYON_NUM_THREADS", "20")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.getenv(
        "PM_MODEL_PATH",
        str(ROOT / "models" / "bge-reranker-v2-m3-3ep-h100"),
    )
)
MAX_LENGTH = int(os.getenv("PM_MAX_LENGTH", "384"))
PAIR_BATCH_SIZE = int(os.getenv("PM_PAIR_BATCH_SIZE", "512"))
MIN_PAIR_BATCH_SIZE = int(os.getenv("PM_MIN_PAIR_BATCH_SIZE", "64"))
PUBLIC_MAX_PAIRS = int(os.getenv("PM_PUBLIC_MAX_PAIRS", "200000"))
PUBLIC_SOFT_LIMIT = float(os.getenv("PM_PUBLIC_SOFT_LIMIT_SECONDS", "320"))
PRIVATE_SOFT_LIMIT = float(os.getenv("PM_PRIVATE_SOFT_LIMIT_SECONDS", "740"))
DEADLINE_RESERVE = float(os.getenv("PM_DEADLINE_RESERVE_SECONDS", "22"))

SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items_path", "--items-path", "-i", required=True, type=Path)
    parser.add_argument("--matches_path", "--matches-path", "-m", required=True, type=Path)
    parser.add_argument("--output_path", "--output-path", "-o", required=True, type=Path)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def log(message: str, started: float) -> None:
    print(f"[{time.perf_counter() - started:7.1f}s] {message}", flush=True)


def require_columns(names: list[str], required: set[str], label: str) -> None:
    missing = required.difference(names)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def clean_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return SPACE_RE.sub(" ", str(value)).strip()


def key_priority(key: str) -> tuple[int, str]:
    normalized = clean_field(key).casefold()
    for rank, part in enumerate(PRIORITY_KEY_PARTS):
        if part in normalized:
            return rank, normalized
    return len(PRIORITY_KEY_PARTS), normalized


def serialize_product(category: Any, name: Any, raw_attributes: Any) -> str:
    """Reproduce the exact serializer used to prepare this checkpoint's train."""
    try:
        attributes = json.loads(raw_attributes)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid attributes JSON: {str(raw_attributes)[:120]}") from error
    if not isinstance(attributes, dict):
        raise ValueError("attributes JSON must contain an object")

    fields: list[str] = []
    for key, value in sorted(attributes.items(), key=lambda item: key_priority(item[0])):
        key_text, value_text = clean_field(key), clean_field(value)
        if key_text and value_text:
            fields.append(f"{key_text}: {value_text}")
    attribute_text = "\n".join(fields)
    if len(attribute_text) > 6000:
        attribute_text = (
            attribute_text[:6000].rsplit("\n", 1)[0].rstrip()
            + "\nХарактеристики обрезаны: да"
        )

    parts = [
        f"Категория: {clean_field(category)}",
        f"Название: {clean_field(name)}",
    ]
    if attribute_text:
        parts.extend(attribute_text.splitlines())
    return "\n".join(parts)


def configure_writable_cache(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_root = output_path.parent / ".bge-hf-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(cache_root / "huggingface" / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "transformers")
    os.environ["HF_MODULES_CACHE"] = str(cache_root / "modules")
    return cache_root


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

    match_schema = pq.read_schema(matches_path)
    require_columns(match_schema.names, {"id1", "id2"}, "matches parquet")
    matches = pq.read_table(
        matches_path,
        columns=["id1", "id2"],
        memory_map=True,
        use_threads=True,
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
    require_columns(
        item_schema.names,
        {"id", "name", "attributes", "category"},
        "items parquet",
    )
    items = pq.read_table(
        items_path,
        columns=["id", "name", "attributes", "category"],
        memory_map=True,
        use_threads=True,
    ).combine_chunks()
    selected = items.filter(pc.is_in(items["id"], value_set=needed_ids)).combine_chunks()
    if (
        selected.num_rows != len(needed_ids)
        or pc.count_distinct(selected["id"]).as_py() != len(needed_ids)
    ):
        raise ValueError("items parquet must contain one unique row for every referenced ID")

    selected_ids = selected["id"].chunk(0)
    left_index = pc.index_in(id1, value_set=selected_ids)
    right_index = pc.index_in(id2, value_set=selected_ids)
    if left_index.null_count or right_index.null_count:
        raise ValueError("at least one match ID is absent from items parquet")
    left_positions = left_index.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    right_positions = right_index.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)

    categories = selected["category"].to_pylist()
    names = selected["name"].to_pylist()
    attributes = selected["attributes"].to_pylist()
    texts = [
        serialize_product(category, name, raw)
        for category, name, raw in zip(categories, names, attributes)
    ]

    # One orientation is needed to pass the public time limit. Putting the longer
    # record first was the strongest causal single-pass rule on both frozen splits.
    left: list[str] = []
    right: list[str] = []
    left_names: list[str] = []
    right_names: list[str] = []
    for left_position, right_position in zip(left_positions, right_positions):
        first_text, second_text = texts[left_position], texts[right_position]
        first_name, second_name = str(names[left_position]), str(names[right_position])
        if len(first_text) >= len(second_text):
            left.append(first_text)
            right.append(second_text)
            left_names.append(first_name)
            right_names.append(second_name)
        else:
            left.append(second_text)
            right.append(first_text)
            left_names.append(second_name)
            right_names.append(first_name)

    id1_values, id2_values = id1.to_pylist(), id2.to_pylist()
    log(
        f"Loaded {matches.num_rows:,} pairs; serialized {len(texts):,} referenced items",
        started,
    )
    del (
        items,
        selected,
        matches,
        needed_ids,
        selected_ids,
        texts,
        names,
        attributes,
        categories,
        left_index,
        right_index,
        left_positions,
        right_positions,
    )
    gc.collect()
    return id1_values, id2_values, left, right, left_names, right_names


def load_model(started: float):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    required = {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }
    missing = [name for name in sorted(required) if not (MODEL_PATH / name).is_file()]
    if missing:
        raise FileNotFoundError(f"offline BGE checkpoint is incomplete: {missing}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    major, _ = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if major >= 8 else torch.float16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        use_fast=True,
        trust_remote_code=False,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="sdpa",
        torch_dtype=dtype,
    ).cuda().eval()
    model.config.use_cache = False
    log(
        f"Loaded checkpoint on {torch.cuda.get_device_name(0)}; "
        f"dtype={dtype}, attention=sdpa",
        started,
    )
    return torch, tokenizer, model, dtype


def encode_batch(tokenizer, left: list[str], right: list[str], indices):
    batch_indices = indices.tolist()
    encoded = tokenizer(
        [left[index] for index in batch_indices],
        [right[index] for index in batch_indices],
        add_special_tokens=True,
        padding=True,
        truncation="longest_first",
        max_length=MAX_LENGTH,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    # Pinned host tensors make the H2D copy genuinely asynchronous.
    encoded = {key: value.pin_memory() for key, value in encoded.items()}
    return indices, encoded


def is_cuda_oom(error: RuntimeError) -> bool:
    return "out of memory" in str(error).casefold()


def model_scores(
    left: list[str],
    right: list[str],
    torch,
    tokenizer,
    model,
    dtype,
    deadline: float,
    started: float,
):
    import numpy as np

    scores = np.full(len(left), np.nan, dtype=np.float32)
    lengths = np.fromiter(
        (len(first) + len(second) for first, second in zip(left, right)),
        dtype=np.int64,
        count=len(left),
    )
    # Longest first: the initial batch validates the configured batch size at
    # the worst sequence length, then globally sorted batches minimize padding.
    order = np.argsort(lengths, kind="stable")[::-1]
    del lengths

    position = 0
    batch_size = max(MIN_PAIR_BATCH_SIZE, PAIR_BATCH_SIZE)
    last_batch_seconds: float | None = None
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        while position < len(order):
            retry_with_smaller_batch = False
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="tokenizer") as pool:
                end = min(len(order), position + batch_size)
                future: Future = pool.submit(
                    encode_batch, tokenizer, left, right, order[position:end].copy()
                )
                while position < len(order):
                    remaining_seconds = deadline - time.perf_counter()
                    required_seconds = DEADLINE_RESERVE + (
                        1.5 * last_batch_seconds if last_batch_seconds is not None else 5.0
                    )
                    if remaining_seconds < required_seconds:
                        future.cancel()
                        break

                    indices, encoded = future.result()
                    current_end = position + len(indices)
                    next_future: Future | None = None
                    if current_end < len(order):
                        next_end = min(len(order), current_end + batch_size)
                        next_future = pool.submit(
                            encode_batch,
                            tokenizer,
                            left,
                            right,
                            order[current_end:next_end].copy(),
                        )

                    batch_started = time.perf_counter()
                    try:
                        gpu_batch = {
                            key: value.cuda(non_blocking=True)
                            for key, value in encoded.items()
                        }
                        logits = model(**gpu_batch).logits.reshape(-1)
                        probabilities = torch.sigmoid(logits.float()).cpu().numpy()
                    except RuntimeError as error:
                        if not is_cuda_oom(error) or batch_size <= MIN_PAIR_BATCH_SIZE:
                            raise
                        if next_future is not None:
                            next_future.cancel()
                        del encoded
                        if "gpu_batch" in locals():
                            del gpu_batch
                        torch.cuda.empty_cache()
                        new_batch_size = max(MIN_PAIR_BATCH_SIZE, batch_size // 2)
                        log(
                            f"CUDA OOM at batch={batch_size}; retrying with batch={new_batch_size}",
                            started,
                        )
                        batch_size = new_batch_size
                        retry_with_smaller_batch = True
                        break

                    scores[indices] = probabilities
                    position = current_end
                    last_batch_seconds = time.perf_counter() - batch_started
                    del encoded, gpu_batch, logits, probabilities
                    if position == len(order) or position % (batch_size * 20) == 0:
                        log(
                            f"Scored {position:,}/{len(order):,} pairs; "
                            f"batch={batch_size}",
                            started,
                        )
                    if next_future is None:
                        break
                    future = next_future
            if not retry_with_smaller_batch:
                break

    remaining = np.flatnonzero(~np.isfinite(scores))
    return scores, remaining, batch_size


def lexical_scores(left_names: list[str], right_names: list[str], indices):
    import numpy as np

    result = np.empty(len(indices), dtype=np.float32)
    for output_index, pair_index in enumerate(indices):
        first, second = left_names[pair_index], right_names[pair_index]
        first_tokens = set(TOKEN_RE.findall(first.casefold()))
        second_tokens = set(TOKEN_RE.findall(second.casefold()))
        union = first_tokens | second_tokens
        score = len(first_tokens & second_tokens) / len(union) if union else 0.0
        first_numbers = set(NUMBER_RE.findall(first))
        second_numbers = set(NUMBER_RE.findall(second))
        if first_numbers and second_numbers and not (first_numbers & second_numbers):
            score *= 0.25
        result[output_index] = score
    return result


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
        writer.writerows(
            (first_id, second_id, float(score))
            for first_id, second_id, score in zip(id1, id2, scores)
        )
    os.replace(temporary, path)
    log(f"Wrote {len(scores):,} predictions to {path}", started)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    cache_root = configure_writable_cache(args.output_path)
    id1, id2, left, right, left_names, right_names = load_pairs(
        args.items_path,
        args.matches_path,
        args.limit,
        started,
    )
    if args.skip_model:
        import numpy as np

        all_indices = np.arange(len(left), dtype=np.int64)
        scores = lexical_scores(left_names, right_names, all_indices)
        log("--skip-model: model-independent contract smoke test", started)
    else:
        soft_limit = (
            PUBLIC_SOFT_LIMIT if len(left) <= PUBLIC_MAX_PAIRS else PRIVATE_SOFT_LIMIT
        )
        deadline = started + soft_limit
        log(f"Using writable Transformers cache at {cache_root}", started)
        torch, tokenizer, model, dtype = load_model(started)
        scores, remaining, final_batch_size = model_scores(
            left,
            right,
            torch,
            tokenizer,
            model,
            dtype,
            deadline,
            started,
        )
        log(f"Final inference batch size: {final_batch_size}", started)
        if len(remaining):
            log(f"Soft deadline near; lexical fallback for {len(remaining):,} pairs", started)
            scores[remaining] = lexical_scores(left_names, right_names, remaining)
    write_output(args.output_path, id1, id2, scores, started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
