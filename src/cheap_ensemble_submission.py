#!/usr/bin/env python3
"""Offline MiniLM S2 plus a fitted lightweight meta-model."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import time
from pathlib import Path


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("OMP_NUM_THREADS", "20")
os.environ.setdefault("MKL_NUM_THREADS", "20")
os.environ.setdefault("RAYON_NUM_THREADS", "20")

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models/minilm-s2-values-only"
ATTRIBUTE_FREQUENCY_PATH = ROOT / "attribute_name_frequency.csv"
ENSEMBLE_CONFIG_PATH = ROOT / "ensemble_config.json"
FEATURE_CONFIG_PATH = ROOT / "feature_config.json"
CHAR_IDF_PATH = ROOT / "char_idf.npy"
MAX_LENGTH = int(os.getenv("PM_MAX_LENGTH", "256"))
PAIR_BATCH_SIZE = int(os.getenv("PM_PAIR_BATCH_SIZE", "512"))
PUBLIC_MAX_PAIRS = int(os.getenv("PM_PUBLIC_MAX_PAIRS", "200000"))
PUBLIC_SOFT_LIMIT = float(os.getenv("PM_PUBLIC_SOFT_LIMIT_SECONDS", "315"))
PRIVATE_SOFT_LIMIT = float(os.getenv("PM_PRIVATE_SOFT_LIMIT_SECONDS", "735"))
DEADLINE_RESERVE = float(os.getenv("PM_DEADLINE_RESERVE_SECONDS", "25"))
TOKEN_RE = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


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


def configure_writable_cache(output_path: Path) -> None:
    cache_root = output_path.parent / ".minilm-s2-ensemble-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(cache_root / "huggingface/hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "transformers")
    os.environ["JOBLIB_TEMP_FOLDER"] = str(cache_root / "joblib")
    os.environ["MPLCONFIGDIR"] = str(cache_root / "matplotlib")


def require_columns(names: list[str], required: set[str], label: str) -> None:
    missing = required.difference(names)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def attribute_rank() -> dict[str, int]:
    with ATTRIBUTE_FREQUENCY_PATH.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        names = [row["attribute_name"] for row in reader]
    if len(names) != 24_916 or len(set(names)) != len(names):
        raise ValueError(f"unexpected attribute rank table: {len(names)} rows")
    return {name: rank for rank, name in enumerate(names)}


def load_pairs(items_path: Path, matches_path: Path, limit: int | None, started: float):
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from cheap_ensemble import prepare_item_records
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
    id1, id2 = matches["id1"].chunk(0), matches["id2"].chunk(0)
    if id1.null_count or id2.null_count:
        raise ValueError("matches parquet contains null IDs")
    needed_ids = pc.unique(pa.concat_arrays([id1, id2]))

    item_schema = pq.read_schema(items_path)
    require_columns(
        item_schema.names, {"id", "name", "attributes", "category"}, "items parquet"
    )
    items = pq.read_table(
        items_path,
        columns=["id", "name", "attributes", "category"],
        memory_map=True,
        use_threads=True,
    ).combine_chunks()
    selected = items.filter(pc.is_in(items["id"], value_set=needed_ids)).combine_chunks()
    if selected.num_rows != len(needed_ids) or pc.count_distinct(selected["id"]).as_py() != len(needed_ids):
        raise ValueError("items parquet does not contain one unique row for every referenced ID")
    selected_ids = selected["id"].chunk(0)
    left_index = pc.index_in(id1, value_set=selected_ids)
    right_index = pc.index_in(id2, value_set=selected_ids)
    if left_index.null_count or right_index.null_count:
        raise ValueError("at least one match ID is absent from items parquet")

    raw_items = selected.to_pandas()
    item_records = prepare_item_records(raw_items)
    ranks = attribute_rank()
    texts = [
        serialize_product(row.name, parse_attributes(row.attributes), "S2_VALUES_ONLY", set(), ranks)
        for row in raw_items.itertuples(index=False)
    ]
    left_positions = left_index.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    right_positions = right_index.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    left = [texts[index] for index in left_positions]
    right = [texts[index] for index in right_positions]
    pair_frame = pd.DataFrame({"id1": id1.to_pylist(), "id2": id2.to_pylist()})
    log(f"Loaded {len(pair_frame):,} pairs and prepared {len(item_records):,} items", started)
    del items, selected, matches, needed_ids, selected_ids, raw_items, texts
    gc.collect()
    return pair_frame, item_records, left, right


def lexical_scores(left: list[str], right: list[str]):
    import numpy as np

    result = np.empty(len(left), dtype=np.float32)
    for index, (a, b) in enumerate(zip(left, right)):
        left_tokens, right_tokens = set(TOKEN_RE.findall(a)), set(TOKEN_RE.findall(b))
        union = left_tokens | right_tokens
        score = len(left_tokens & right_tokens) / len(union) if union else 0.0
        left_numbers, right_numbers = set(NUMBER_RE.findall(a)), set(NUMBER_RE.findall(b))
        if left_numbers and right_numbers and not left_numbers & right_numbers:
            score *= 0.25
        result[index] = score
    return result


def load_model(started: float):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, local_files_only=True, attn_implementation="sdpa"
    ).cuda().eval()
    log(f"Loaded S2 checkpoint on {torch.cuda.get_device_name(0)}", started)
    return torch, tokenizer, model


def model_scores(left, right, torch, tokenizer, model, deadline: float, started: float):
    import numpy as np

    scores = np.empty(len(left), dtype=np.float32)
    order = np.argsort(
        np.fromiter((len(a) + len(b) for a, b in zip(left, right)), dtype=np.int64),
        kind="stable",
    )
    completed = 0
    last_batch_seconds: float | None = None
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for start in range(0, len(left), PAIR_BATCH_SIZE):
            remaining = deadline - time.perf_counter()
            if last_batch_seconds is not None and remaining < last_batch_seconds * 1.4 + DEADLINE_RESERVE:
                break
            end = min(len(left), start + PAIR_BATCH_SIZE)
            indices = order[start:end]
            size = end - start
            encoded = tokenizer(
                [left[index] for index in indices] + [right[index] for index in indices],
                [right[index] for index in indices] + [left[index] for index in indices],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )
            encoded = {key: value.cuda(non_blocking=True) for key, value in encoded.items()}
            batch_started = time.perf_counter()
            probability = torch.sigmoid(model(**encoded).logits.reshape(-1).float())
            scores[indices] = ((probability[:size] + probability[size:]) * 0.5).cpu().numpy()
            torch.cuda.synchronize()
            last_batch_seconds = time.perf_counter() - batch_started
            completed = end
            if completed == len(left) or completed % (PAIR_BATCH_SIZE * 20) == 0:
                log(f"Scored {completed:,}/{len(left):,} pairs", started)
    return scores, order[completed:]


def meta_scores(features, started: float):
    config = json.loads(ENSEMBLE_CONFIG_PATH.read_text(encoding="utf-8"))
    model_type = config["model_type"]
    if model_type == "logistic":
        import joblib

        model = joblib.load(ROOT / "models/logistic_pipeline.joblib")
    elif model_type == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier()
        model.load_model(ROOT / "models/catboost_model.cbm")
    else:
        raise ValueError(f"unsupported ensemble model_type: {model_type!r}")
    scores = model.predict_proba(features)[:, 1]
    log(f"Applied {model_type} meta-model", started)
    return scores


def write_output(path: Path, pairs, scores, started: float) -> None:
    if len(pairs) != len(scores) or not all(math.isfinite(float(score)) for score in scores):
        raise RuntimeError("invalid output scores")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["id1", "id2", "predict"])
        writer.writerows(
            (row.id1, row.id2, float(score))
            for row, score in zip(pairs.itertuples(index=False), scores)
        )
    os.replace(temporary, path)
    log(f"Wrote {len(scores):,} predictions to {path}", started)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    configure_writable_cache(args.output_path)
    import numpy as np
    from cheap_ensemble import FEATURE_COLUMNS, build_pair_features

    pairs, items, left, right = load_pairs(args.items_path, args.matches_path, args.limit, started)
    feature_config = json.loads(FEATURE_CONFIG_PATH.read_text(encoding="utf-8"))
    if feature_config["feature_columns"] != list(FEATURE_COLUMNS):
        raise RuntimeError("bundled feature/model contract mismatch")
    char_idf = np.load(CHAR_IDF_PATH)
    char_config = feature_config["char_tfidf"]
    features = build_pair_features(
        items,
        pairs,
        np.zeros(len(pairs), dtype=np.float32),
        char_idf,
        ngram_min=int(char_config["ngram_min"]),
        ngram_max=int(char_config["ngram_max"]),
    )
    log("Built lexical/numeric/attribute features", started)
    if args.skip_model:
        transformer = lexical_scores(left, right)
        log("--skip-model: lexical placeholder used for transformer_score", started)
    else:
        soft_limit = PUBLIC_SOFT_LIMIT if len(pairs) <= PUBLIC_MAX_PAIRS else PRIVATE_SOFT_LIMIT
        torch, tokenizer, model = load_model(started)
        transformer, remaining = model_scores(
            left, right, torch, tokenizer, model, started + soft_limit, started
        )
        if len(remaining):
            log(f"Soft deadline near; lexical fallback for {len(remaining):,} pairs", started)
            transformer[remaining] = lexical_scores(
                [left[index] for index in remaining],
                [right[index] for index in remaining],
            )
    features["transformer_score"] = transformer
    write_output(args.output_path, pairs, meta_scores(features, started), started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
