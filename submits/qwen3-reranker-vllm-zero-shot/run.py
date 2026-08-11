#!/usr/bin/env python3
"""Fast, offline E-CUP product-matching inference with Qwen3 and vLLM."""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import sys
import time
from pathlib import Path


# Set runtime knobs before importing torch/vLLM/tokenizers.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("OMP_NUM_THREADS", "20")
os.environ.setdefault("MKL_NUM_THREADS", "20")
os.environ.setdefault("RAYON_NUM_THREADS", "20")
os.environ.setdefault("VLLM_CACHE_ROOT", "/tmp/vllm-cache")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton-cache")


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "Qwen3-Reranker-0.6B"

# Defaults are deliberately latency-oriented for one H100 80 GB. They can be
# overridden for profiling without rebuilding the archive.
MAX_ITEM_CHARS = int(os.getenv("PM_MAX_ITEM_CHARS", "180"))
MAX_MODEL_LEN = int(os.getenv("PM_MAX_MODEL_LEN", "256"))
MAX_NUM_SEQS = int(os.getenv("PM_MAX_NUM_SEQS", "1024"))
MAX_NUM_BATCHED_TOKENS = int(os.getenv("PM_MAX_NUM_BATCHED_TOKENS", "131072"))
SCORE_CHUNK_SIZE = int(os.getenv("PM_SCORE_CHUNK_SIZE", "50000"))

CHAT_TEMPLATE = r'''<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Determine whether the Query and Document describe the same marketplace product. Treat wording and schema differences as irrelevant, but distinguish genuinely different models or material product configurations.
<Query>: {{ messages | selectattr("role", "eq", "query") | map(attribute="content") | first }}
<Document>: {{ messages | selectattr("role", "eq", "document") | map(attribute="content") | first }}<|im_end|>
<|im_start|>assistant
<think>

</think>
'''


def log(message: str, started_at: float) -> None:
    print(f"[{time.perf_counter() - started_at:8.1f}s] {message}", flush=True)


def clean_scalar(value: object, limit: int | None = None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
    return text if limit is None else text[:limit]


def serialize_item(category: object, name: object, attributes: object) -> str:
    """Keep the highest-value fields first and avoid per-row JSON parsing."""
    category_text = clean_scalar(category, 80)
    name_text = clean_scalar(name, 180)
    attributes_text = clean_scalar(attributes)
    prefix = f"Категория: {category_text}\nНазвание: {name_text}\nАтрибуты: "
    remaining = max(0, MAX_ITEM_CHARS - len(prefix))
    # The final slice is intentional: unusually long names must not silently
    # defeat the latency-oriented item cap.
    return (prefix + attributes_text[:remaining])[:MAX_ITEM_CHARS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items_path", "--items-path", "-i", dest="items_path", required=True)
    parser.add_argument(
        "--matches_path", "--matches-path", "-m", dest="matches_path", required=True
    )
    parser.add_argument(
        "--output_path", "--output-path", "-o", dest="output_path", required=True
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="local format/preprocessing test only; writes constant scores",
    )
    return parser.parse_args()


def require_columns(schema_names: list[str], required: set[str], label: str) -> None:
    missing = required.difference(schema_names)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def load_pairs(items_path: Path, matches_path: Path, started_at: float):
    try:
        import numpy as np
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "PyArrow is unavailable. Run the submission through run.sh so bundled "
            "vendor packages are on PYTHONPATH."
        ) from error

    match_schema = pq.read_schema(matches_path)
    require_columns(match_schema.names, {"id1", "id2"}, "matches parquet")
    matches = pq.read_table(
        matches_path,
        columns=["id1", "id2"],
        memory_map=True,
        use_threads=True,
    ).combine_chunks()
    if matches.num_rows == 0:
        raise ValueError("matches parquet contains no rows")
    id1 = matches["id1"].chunk(0)
    id2 = matches["id2"].chunk(0)
    if id1.null_count or id2.null_count:
        raise ValueError("matches parquet contains null product IDs")
    needed_ids = pc.unique(pa.concat_arrays([id1, id2]))
    log(
        f"Loaded {matches.num_rows:,} pairs and {len(needed_ids):,} required item IDs",
        started_at,
    )

    item_columns = ["id", "name", "attributes", "category"]
    item_schema = pq.read_schema(items_path)
    require_columns(item_schema.names, set(item_columns), "items parquet")
    items = pq.read_table(
        items_path,
        columns=item_columns,
        memory_map=True,
        use_threads=True,
    ).combine_chunks()

    # Filtering and ID lookup stay in Arrow/C++; Python sees only items that
    # actually occur in candidate pairs.
    mask = pc.is_in(items["id"], value_set=needed_ids)
    selected = items.filter(mask).combine_chunks()
    if selected.num_rows != len(needed_ids):
        unique_found = pc.count_distinct(selected["id"]).as_py()
        raise ValueError(
            f"Expected {len(needed_ids):,} unique referenced items, found "
            f"{unique_found:,} unique rows ({selected.num_rows:,} total rows)"
        )
    selected_ids = selected["id"].chunk(0)
    left_index = pc.index_in(id1, value_set=selected_ids)
    right_index = pc.index_in(id2, value_set=selected_ids)
    if left_index.null_count or right_index.null_count:
        raise ValueError("at least one match ID is absent from items parquet")

    names = selected["name"].to_pylist()
    attributes = selected["attributes"].to_pylist()
    categories = selected["category"].to_pylist()
    item_texts = [
        serialize_item(category, name, attrs)
        for category, name, attrs in zip(categories, names, attributes)
    ]
    left_index_np = left_index.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    right_index_np = right_index.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    queries = [item_texts[index] for index in left_index_np]
    documents = [item_texts[index] for index in right_index_np]
    id1_values = id1.to_pylist()
    id2_values = id2.to_pylist()

    del items, selected, mask, names, attributes, categories, item_texts
    del matches, needed_ids, selected_ids, left_index, right_index
    gc.collect()
    log(
        f"Prepared {len(queries):,} text pairs; item text cap={MAX_ITEM_CHARS} chars",
        started_at,
    )
    return id1_values, id2_values, queries, documents


def load_model(started_at: float):
    if not MODEL_PATH.is_dir() or not (MODEL_PATH / "model.safetensors").is_file():
        raise FileNotFoundError(f"Offline model checkpoint is incomplete: {MODEL_PATH}")

    import torch
    from vllm import LLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this submission")
    device_name = torch.cuda.get_device_name(0)
    log(f"CUDA device: {device_name}; loading {MODEL_PATH.name}", started_at)
    model = LLM(
        model=str(MODEL_PATH),
        runner="pooling",
        hf_overrides={
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=True,
        enforce_eager=True,
        disable_log_stats=True,
        trust_remote_code=False,
        seed=42,
    )
    log(
        f"Model ready: max_len={MAX_MODEL_LEN}, max_num_seqs={MAX_NUM_SEQS}, "
        f"max_batched_tokens={MAX_NUM_BATCHED_TOKENS}",
        started_at,
    )
    return model


def predict(model, queries: list[str], documents: list[str], started_at: float):
    import numpy as np

    scores = np.empty(len(queries), dtype=np.float32)
    inference_start = time.perf_counter()
    for start in range(0, len(queries), SCORE_CHUNK_SIZE):
        end = min(start + SCORE_CHUNK_SIZE, len(queries))
        outputs = model.score(
            queries[start:end],
            documents[start:end],
            chat_template=CHAT_TEMPLATE,
            truncate_prompt_tokens=MAX_MODEL_LEN,
            use_tqdm=False,
        )
        scores[start:end] = [output.outputs.score for output in outputs]
        elapsed = time.perf_counter() - inference_start
        log(
            f"Scored {end:,}/{len(queries):,} pairs "
            f"({end / max(elapsed, 1e-9):,.1f} pairs/s)",
            started_at,
        )
    if not np.isfinite(scores).all():
        raise RuntimeError("vLLM returned NaN or infinite scores")
    return scores


def write_submission(
    output_path: Path,
    id1_values: list[object],
    id2_values: list[object],
    scores,
    started_at: float,
) -> None:
    if not (len(id1_values) == len(id2_values) == len(scores)):
        raise RuntimeError("output arrays have different lengths")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, lineterminator="\n")
        writer.writerow(["id1", "id2", "predict"])
        writer.writerows(
            (id1, id2, float(score))
            for id1, id2, score in zip(id1_values, id2_values, scores)
        )
    os.replace(temporary_path, output_path)
    log(f"Wrote {len(scores):,} predictions to {output_path}", started_at)


def main() -> int:
    args = parse_args()
    started_at = time.perf_counter()
    log(
        f"Starting Qwen3 vLLM inference with Python {sys.version.split()[0]}",
        started_at,
    )
    id1, id2, queries, documents = load_pairs(
        Path(args.items_path), Path(args.matches_path), started_at
    )
    if args.skip_model:
        import numpy as np

        scores = np.full(len(queries), 0.5, dtype=np.float32)
        log("--skip-model active: generated constant local-test scores", started_at)
    else:
        model = load_model(started_at)
        scores = predict(model, queries, documents, started_at)
    write_submission(Path(args.output_path), id1, id2, scores, started_at)
    log("Submission pipeline complete", started_at)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
