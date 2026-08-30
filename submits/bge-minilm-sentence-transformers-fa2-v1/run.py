#!/usr/bin/env python3
"""BGE + MiniLM rank ensemble through SentenceTransformers CrossEncoder."""

from __future__ import annotations

import gc
import time

import numpy as np

import ensemble_base as base


# Fixed experiment settings. Do not inherit PM_MAX_LENGTH=384 from the older
# base image: this submission intentionally benchmarks the requested 192 tokens.
MAX_LENGTH = 192
BATCH_SIZE = 1024
ATTENTION_IMPLEMENTATION = "sdpa"


def load_cross_encoder(model_path, label: str, torch, started: float):
    from sentence_transformers import CrossEncoder

    required = [
        model_path / "config.json",
        model_path / "model.safetensors",
        model_path / "tokenizer_config.json",
    ]
    missing = [path.name for path in required if not path.is_file() or not path.stat().st_size]
    if missing:
        raise FileNotFoundError(f"offline {label} checkpoint is incomplete: {missing}")
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
    torch.cuda.synchronize()
    base.log(
        f"Loaded {label} via SentenceTransformers in "
        f"{time.perf_counter() - load_started:.1f}s "
        f"(attention={ATTENTION_IMPLEMENTATION})",
        started,
    )
    return model


def score_pairs(model, left, right, label: str, torch, started: float) -> np.ndarray:
    if BATCH_SIZE < 1:
        raise ValueError("PM_BATCH_SIZE must be positive")
    pair_count = len(left)
    # Cheap bucketing reduces padding while CrossEncoder keeps its own batched
    # Rust tokenization. TOKENIZERS_PARALLELISM/RAYON_NUM_THREADS provide the
    # requested CPU tokenizer parallelism without duplicating model processes.
    order = np.argsort(
        np.fromiter((len(a) + len(b) for a, b in zip(left, right)), dtype=np.int64),
        kind="stable",
    )
    pairs = [(left[int(index)], right[int(index)]) for index in order]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    inference_started = time.perf_counter()
    with torch.inference_mode():
        raw = model.predict(
            pairs,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            activation_fn=torch.nn.Identity(),
            convert_to_numpy=True,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - inference_started
    sorted_scores = np.asarray(raw, dtype=np.float32).reshape(-1)
    if len(sorted_scores) != pair_count:
        raise RuntimeError(
            f"{label} returned {len(sorted_scores)} scores for {pair_count} pairs"
        )
    scores = np.empty(pair_count, dtype=np.float32)
    scores[order] = sorted_scores
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    base.log(
        f"{label}: {pair_count:,} pairs in {elapsed:.1f}s "
        f"({pair_count / elapsed:.1f} pairs/s), peak CUDA {peak_gib:.2f} GiB",
        started,
    )
    return scores


def clear_cuda(torch) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def main() -> int:
    args = base.parse_args()
    started = time.perf_counter()
    id1, id2, left, right = base.load_pairs(
        args.items_path, args.matches_path, args.limit, started
    )
    base.configure_writable_cache(args.output_path)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.set_float32_matmul_precision("high")
    base.log(
        f"CUDA={torch.cuda.get_device_name(0)}, max_length={MAX_LENGTH}, "
        f"batch={BATCH_SIZE}, one_direction=True",
        started,
    )

    bge = load_cross_encoder(base.BGE_PATH, "BGE", torch, started)
    bge_logits = score_pairs(bge, left, right, "BGE", torch, started)
    del bge
    clear_cuda(torch)

    minilm = load_cross_encoder(base.MINILM_PATH, "MiniLM", torch, started)
    minilm_logits = score_pairs(minilm, left, right, "MiniLM", torch, started)
    del minilm
    clear_cuda(torch)

    ensemble = (
        base.normalized_rank(bge_logits) + base.normalized_rank(minilm_logits)
    ) * 0.5
    base.write_output(args.output_path, id1, id2, ensemble, started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
