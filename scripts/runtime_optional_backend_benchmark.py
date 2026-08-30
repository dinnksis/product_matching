#!/usr/bin/env python3
"""Benchmark SentenceTransformers or vLLM on frozen cross-encoder checkpoints."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import runtime_inference_benchmark as core


MODEL_BATCH = {"bge": 32, "minilm": 192}
MAX_LENGTH = 384


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sentence_transformers", "vllm"), required=True)
    parser.add_argument("--model", choices=tuple(MODEL_BATCH), required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sample_frame(args: argparse.Namespace) -> pd.DataFrame:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    _, base_sample = core.load_references(
        args.input_root, manifest, args.sample_size, args.seed
    )
    return core.reference_for_model(
        args.input_root, manifest, args.model, base_sample
    )


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result.astype(np.float32)


def sentence_transformers_scores(
    model_dir: Path, left: list[str], right: list[str], batch_size: int
) -> tuple[np.ndarray, dict[str, float | str]]:
    import torch
    from sentence_transformers import CrossEncoder

    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = CrossEncoder(
        str(model_dir),
        device="cuda",
        max_length=MAX_LENGTH,
        local_files_only=True,
        model_kwargs={
            "attn_implementation": "sdpa",
            "torch_dtype": torch.float16,
        },
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    pairs = list(zip(left + right, right + left))
    inference_started = time.perf_counter()
    raw = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        activation_fn=torch.nn.Identity(),
        convert_to_numpy=True,
    )
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    logits = np.asarray(raw, dtype=np.float32).reshape(-1)
    probability = sigmoid(logits)
    pair_count = len(left)
    scores = (probability[:pair_count] + probability[pair_count:]) * 0.5
    timing: dict[str, float | str] = {
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "backend_wall_seconds": load_seconds + inference_seconds,
        "pairs_per_second": pair_count / inference_seconds,
        "end_to_end_pairs_per_second": pair_count / (load_seconds + inference_seconds),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "precision": "float16",
        "package_version": importlib.metadata.version("sentence-transformers"),
    }
    del model, pairs
    gc.collect()
    torch.cuda.empty_cache()
    return scores, timing


def vllm_scores(
    model_dir: Path, left: list[str], right: list[str]
) -> tuple[np.ndarray, dict[str, float | str]]:
    import torch
    from vllm import LLM, PoolingParams

    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = LLM(
        model=str(model_dir),
        runner="pooling",
        dtype="float16",
        max_model_len=MAX_LENGTH,
        max_num_seqs=256,
        max_num_batched_tokens=65_536,
        gpu_memory_utilization=0.90,
        tensor_parallel_size=1,
        enforce_eager=True,
        trust_remote_code=False,
        disable_log_stats=True,
    )
    load_seconds = time.perf_counter() - load_started
    combined_left = left + right
    combined_right = right + left
    inference_started = time.perf_counter()
    outputs = model.score(
        combined_left,
        combined_right,
        truncate_prompt_tokens=MAX_LENGTH,
        use_tqdm=False,
        pooling_params=PoolingParams(use_activation=False),
    )
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    logits = np.asarray([output.outputs.score for output in outputs], dtype=np.float32)
    probability = sigmoid(logits)
    pair_count = len(left)
    scores = (probability[:pair_count] + probability[pair_count:]) * 0.5
    timing: dict[str, float | str] = {
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "backend_wall_seconds": load_seconds + inference_seconds,
        "pairs_per_second": pair_count / inference_seconds,
        "end_to_end_pairs_per_second": pair_count / (load_seconds + inference_seconds),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "precision": "float16",
        "package_version": importlib.metadata.version("vllm"),
    }
    del model, outputs
    gc.collect()
    torch.cuda.empty_cache()
    return scores, timing


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = sample_frame(args)
    left = frame["product_text_1"].astype(str).tolist()
    right = frame["product_text_2"].astype(str).tolist()
    model_dir = args.checkpoint_root / args.model
    started = time.perf_counter()
    try:
        if args.backend == "sentence_transformers":
            scores, timing = sentence_transformers_scores(
                model_dir, left, right, MODEL_BATCH[args.model]
            )
        else:
            scores, timing = vllm_scores(model_dir, left, right)
        diagnostics = core.score_diagnostics(frame, scores)
        finite = bool(np.isfinite(scores).all())
        equivalent = (
            finite
            and diagnostics["pearson"] >= 0.999
            and abs(diagnostics["ap_delta"]) <= 0.002
        )
        report = {
            "model": args.model,
            "backend": args.backend,
            "status": "ok" if equivalent else "non_equivalent",
            "equivalent": equivalent,
            "pairs": len(frame),
            "batch": MODEL_BATCH[args.model] if args.backend == "sentence_transformers" else 256,
            "max_length": MAX_LENGTH,
            **timing,
            **diagnostics,
            "process_wall_seconds": time.perf_counter() - started,
            "error": "",
        }
        prediction_path = args.output_dir / f"{args.model}_{args.backend}_scores.parquet"
        pd.DataFrame(
            {
                "id1": frame["id1"],
                "id2": frame["id2"],
                "target": frame["target"],
                "category": frame["category"],
                "reference_score": frame["reference_score"],
                "score": scores,
            }
        ).to_parquet(prediction_path, index=False, compression="zstd")
        report["predictions_file"] = prediction_path.name
    except Exception as error:
        report = {
            "model": args.model,
            "backend": args.backend,
            "status": "error",
            "equivalent": False,
            "pairs": len(frame),
            "batch": MODEL_BATCH[args.model] if args.backend == "sentence_transformers" else 256,
            "max_length": MAX_LENGTH,
            "process_wall_seconds": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
        }
    output_path = args.output_dir / f"{args.model}_{args.backend}.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0 if report["status"] != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
