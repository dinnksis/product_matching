#!/usr/bin/env python3
"""Shared CLI, offline cache setup and output code for the full ensemble."""

from __future__ import annotations

import argparse
import csv
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
BGE_PATH = ROOT / "models/bge_final"
MINILM_PATH = ROOT / "models/minilm_final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items_path", "--items-path", "-i", required=True, type=Path)
    parser.add_argument("--matches_path", "--matches-path", "-m", required=True, type=Path)
    parser.add_argument("--output_path", "--output-path", "-o", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def log(message: str, started: float) -> None:
    print(f"[{time.perf_counter() - started:7.1f}s] {message}", flush=True)


def configure_writable_cache(output_path: Path) -> None:
    cache_root = output_path.parent / ".full-triple-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(cache_root / "huggingface" / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "transformers")
    os.environ["TRITON_CACHE_DIR"] = str(cache_root / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "torchinductor")
    os.environ["TORCH_COMPILE_DEBUG_DIR"] = str(cache_root / "torch_compile_debug")


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
