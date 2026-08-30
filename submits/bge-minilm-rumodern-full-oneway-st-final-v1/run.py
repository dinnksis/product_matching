#!/usr/bin/env python3
"""Full one-way BGE + MiniLM + RuModernBERT, without CatBoost routing."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import ensemble_base as base
import runtime_base as runtime

ROOT = Path(__file__).resolve().parent
RUMODERN_PATH = ROOT / "models/rumodernbert_final"


def main() -> int:
    args = base.parse_args()
    started = time.perf_counter()
    base.configure_writable_cache(args.output_path)
    pairs, left, right = runtime.load_inputs(
        args.items_path, args.matches_path, args.limit, started
    )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.set_float32_matmul_precision("high")
    base.log(
        f"CUDA={torch.cuda.get_device_name(0)}, max_length={runtime.MAX_LENGTH}, "
        f"batch={runtime.BATCH_SIZE}, direction=AB, full triple, equal blend",
        started,
    )
    probabilities = []
    for path, label in (
        (base.BGE_PATH, "BGE"),
        (base.MINILM_PATH, "MiniLM"),
        (RUMODERN_PATH, "RuModernBERT"),
    ):
        model = runtime.load_cross_encoder(path, label, torch, started)
        probability = runtime.score_oneway(model, left, right, label, torch, started)
        probabilities.append(probability.astype(np.float32))
        del model
        runtime.clear_cuda(torch)
    prediction = np.mean(np.stack(probabilities, axis=0), axis=0).astype(np.float32)
    base.write_output(
        args.output_path,
        pairs["id1"].tolist(),
        pairs["id2"].tolist(),
        prediction,
        started,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
