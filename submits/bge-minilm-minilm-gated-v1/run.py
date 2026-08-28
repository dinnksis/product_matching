#!/usr/bin/env python3
"""Fast MiniLM-gated BGE ensemble using frozen checkpoints."""

from __future__ import annotations

import gc
import os
import time

import numpy as np

import ensemble_base as base


# Fixed once on ordinary validation: its median MiniLM probability.
# The same absolute threshold is used for every competition pair.
MINILM_BGE_GATE = float(os.getenv("PM_MINILM_BGE_GATE", "0.0602078252"))


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
    torch.set_float32_matmul_precision("high")
    base.log(
        f"CUDA device={torch.cuda.get_device_name(0)}, "
        f"memory={torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB",
        started,
    )

    # MiniLM is the cheap first stage and also supplies the final score for
    # confidently negative pairs. It uses the exact training serialization.
    tokenizer, model = base.load_model(base.MINILM_PATH, torch, started, "MiniLM")
    minilm_scores = base.model_scores(
        left,
        right,
        torch,
        tokenizer,
        model,
        base.MINILM_PAIR_BATCH_SIZE,
        base.MINILM_PAIR_TOKEN_BUDGET,
        "MiniLM",
        started,
    )
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    routed = np.flatnonzero(minilm_scores > MINILM_BGE_GATE)
    routed_left = [left[int(index)] for index in routed]
    routed_right = [right[int(index)] for index in routed]
    base.log(
        f"MiniLM gate={MINILM_BGE_GATE:.10f}: routing {len(routed):,}/"
        f"{len(minilm_scores):,} pairs ({len(routed) / len(minilm_scores):.1%}) to BGE",
        started,
    )

    if len(routed) == 0:
        raise RuntimeError("MiniLM gate selected zero BGE pairs; refusing invalid run")

    tokenizer, model = base.load_model(base.BGE_PATH, torch, started, "BGE")
    bge_routed_scores = base.model_scores(
        routed_left,
        routed_right,
        torch,
        tokenizer,
        model,
        base.BGE_PAIR_BATCH_SIZE,
        base.BGE_PAIR_TOKEN_BUDGET,
        "BGE-routed",
        started,
    )

    # Mean probability was validated for this partial-BGE cascade. Rejected
    # pairs retain their MiniLM probability; every output remains continuous.
    scores = minilm_scores.copy()
    scores[routed] = (minilm_scores[routed] + bge_routed_scores) * 0.5
    base.write_output(args.output_path, id1, id2, scores, started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
