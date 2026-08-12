#!/usr/bin/env python3
"""Offline, deadline-aware names-only Jina reranker submission."""
from __future__ import annotations

import argparse
import math
import os
import re
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("OMP_NUM_THREADS", "20")

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "jina-reranker-v2-base-multilingual"
BATCH_SIZE = int(os.getenv("PM_BATCH_SIZE", "512"))
MAX_LENGTH = int(os.getenv("PM_MAX_LENGTH", "128"))
SCORE_CHUNK_SIZE = int(os.getenv("PM_SCORE_CHUNK_SIZE", "20000"))
PUBLIC_MAX_PAIRS = int(os.getenv("PM_PUBLIC_MAX_PAIRS", "200000"))
PUBLIC_SOFT_LIMIT = float(os.getenv("PM_PUBLIC_SOFT_LIMIT_SECONDS", "320"))
PRIVATE_SOFT_LIMIT = float(os.getenv("PM_PRIVATE_SOFT_LIMIT_SECONDS", "740"))
TOKEN_RE = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_path", "--items-path", "-i", required=True, type=Path)
    parser.add_argument("--matches_path", "--matches-path", "-m", required=True, type=Path)
    parser.add_argument("--output_path", "--output-path", "-o", required=True, type=Path)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def log(message: str, started: float) -> None:
    print(f"[{time.perf_counter() - started:7.1f}s] {message}", flush=True)


def load_pairs(items_path: Path, matches_path: Path, limit: int | None):
    import pandas as pd

    matches = pd.read_parquet(matches_path, columns=["id1", "id2"])
    if limit is not None:
        matches = matches.head(limit).copy()
    items = pd.read_parquet(items_path, columns=["id", "name"])
    if items["id"].duplicated().any():
        raise ValueError("item IDs must be unique")
    names = items.set_index("id")["name"]
    left = matches["id1"].map(names)
    right = matches["id2"].map(names)
    if left.isna().any() or right.isna().any():
        raise ValueError("matches reference missing item IDs")
    return matches, left.astype(str).tolist(), right.astype(str).tolist()


def lexical_score(left: str, right: str) -> float:
    a, b = set(TOKEN_RE.findall(left.casefold())), set(TOKEN_RE.findall(right.casefold()))
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def fallback_scores(left: list[str], right: list[str]) -> list[float]:
    return [lexical_score(a, b) for a, b in zip(left, right)]


def configure_writable_cache(output_path: Path) -> Path:
    """Put dynamic Transformers modules beside the guaranteed-writable output."""
    cache_root = output_path.parent / ".product-matching-hf-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    # Assign rather than setdefault: the runtime image may provide defaults
    # pointing to read-only /root. Transformers is imported only afterwards.
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(cache_root / "huggingface" / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "transformers")
    os.environ["HF_MODULES_CACHE"] = str(cache_root / "modules")
    return cache_root


def load_model():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        use_flash_attn=False,
    ).cuda().eval()
    return torch, tokenizer, model


def model_scores(
    left: list[str], right: list[str], torch, tokenizer, model, deadline: float, started: float
) -> tuple[list[float], int]:
    scores: list[float] = []
    completed = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for chunk_start in range(0, len(left), SCORE_CHUNK_SIZE):
            if time.perf_counter() >= deadline:
                break
            chunk_end = min(len(left), chunk_start + SCORE_CHUNK_SIZE)
            for start in range(chunk_start, chunk_end, BATCH_SIZE):
                if time.perf_counter() >= deadline:
                    return scores, completed
                end = min(chunk_end, start + BATCH_SIZE)
                encoded = tokenizer(
                    left[start:end],
                    right[start:end],
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    pad_to_multiple_of=8,
                    return_tensors="pt",
                )
                encoded = {key: value.cuda(non_blocking=True) for key, value in encoded.items()}
                logits = model(**encoded).logits.reshape(-1)
                scores.extend(torch.sigmoid(logits.float()).cpu().tolist())
                completed = end
            log(f"Scored {completed:,}/{len(left):,} pairs", started)
    return scores, completed


def validate(scores: list[float], expected: int) -> None:
    if len(scores) != expected:
        raise ValueError(f"expected {expected} scores, got {len(scores)}")
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("scores contain NaN or infinity")


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    matches, left, right = load_pairs(args.items_path, args.matches_path, args.limit)
    log(f"Loaded {len(matches):,} pairs", started)
    if args.skip_model:
        scores = fallback_scores(left, right)
    else:
        cache_root = configure_writable_cache(args.output_path)
        log(f"Using writable Transformers cache at {cache_root}", started)
        soft_limit = PUBLIC_SOFT_LIMIT if len(matches) <= PUBLIC_MAX_PAIRS else PRIVATE_SOFT_LIMIT
        deadline = started + soft_limit
        torch, tokenizer, model = load_model()
        log(f"Loaded Jina model; batch={BATCH_SIZE}, max_length={MAX_LENGTH}", started)
        scores, completed = model_scores(left, right, torch, tokenizer, model, deadline, started)
        if completed < len(matches):
            log(f"Deadline near; lexical fallback for {len(matches) - completed:,} pairs", started)
            scores.extend(fallback_scores(left[completed:], right[completed:]))
    validate(scores, len(matches))
    output = matches[["id1", "id2"]].copy()
    output["predict"] = scores
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_path, index=False)
    log(f"Saved {len(output):,} predictions to {args.output_path}", started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
