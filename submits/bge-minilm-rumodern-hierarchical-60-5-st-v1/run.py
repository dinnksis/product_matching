#!/usr/bin/env python3
"""BGE 100% -> MiniLM 60% -> RuModernBERT 5% hierarchical ensemble."""

from __future__ import annotations

import gc
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import ensemble_base as base
import router_base as runtime


ROOT = Path(__file__).resolve().parent
RUMODERN_PATH = ROOT / "models/rumodernbert-base-human-ft-v1"
MINI_ROUTER_PATH = ROOT / "router_minilm_classification.cbm"
SEQUENTIAL_ROUTER_PATH = ROOT / "router_rumodernbert_over_bge_minilm_classification.cbm"
SEQUENTIAL_MANIFEST_PATH = ROOT / "sequential_router_manifest.json"
MINI_COVERAGE = float(os.getenv("PM_MINILM_COVERAGE", "0.60"))
RUMODERN_COVERAGE = float(os.getenv("PM_RUMODERN_COVERAGE", "0.05"))


def top_mask(
    priority: np.ndarray,
    coverage: float,
    pairs: pd.DataFrame,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    if not 0.0 <= coverage <= 1.0:
        raise ValueError("route coverage must lie in [0, 1]")
    count = int(math.floor(len(priority) * coverage + 1e-12))
    candidates = (
        np.arange(len(priority), dtype=np.int64)
        if eligible is None
        else np.flatnonzero(np.asarray(eligible, dtype=bool))
    )
    if count > len(candidates):
        raise ValueError("route coverage exceeds eligible hierarchical subset")
    order = np.lexsort(
        (
            pairs["id2"].to_numpy(dtype=np.int64)[candidates],
            pairs["id1"].to_numpy(dtype=np.int64)[candidates],
            -np.asarray(priority, dtype=np.float64)[candidates],
        )
    )
    result = np.zeros(len(priority), dtype=bool)
    result[candidates[order[:count]]] = True
    return result


def sequential_features(
    base_features: pd.DataFrame,
    bge_probability: np.ndarray,
    minilm_probability: np.ndarray,
) -> pd.DataFrame:
    result = base_features.copy().reset_index(drop=True)
    bge = np.asarray(bge_probability, dtype=np.float64)
    mini = np.asarray(minilm_probability, dtype=np.float64)
    clipped = np.clip(mini, 1e-7, 1.0 - 1e-7)
    result["minilm_probability"] = mini.astype(np.float32)
    result["minilm_logit"] = np.log(clipped / (1.0 - clipped)).astype(np.float32)
    result["bge_minilm_disagreement"] = np.abs(bge - mini).astype(np.float32)
    result["bge_minilm_signed_difference"] = (mini - bge).astype(np.float32)
    result["bge_minilm_mean"] = ((bge + mini) * 0.5).astype(np.float32)
    result["bge_minilm_min"] = np.minimum(bge, mini).astype(np.float32)
    result["bge_minilm_max"] = np.maximum(bge, mini).astype(np.float32)
    return result


def main() -> int:
    args = base.parse_args()
    started = time.perf_counter()
    base.configure_writable_cache(args.output_path)
    pairs, items, left, right = runtime.load_inputs(
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
        f"batch={runtime.BATCH_SIZE}, symmetric=True, "
        f"MiniLM={MINI_COVERAGE:.0%}, RuModern={RUMODERN_COVERAGE:.0%}",
        started,
    )

    bge_model = runtime.load_cross_encoder(base.BGE_PATH, "BGE", torch, started)
    bge = runtime.score_symmetric(
        bge_model, left, right, "BGE", torch, started, collect_lengths=True
    )
    del bge_model
    runtime.clear_cuda(torch)

    base_features, base_manifest = runtime.build_router_features(
        pairs, items, bge, started
    )
    from catboost import CatBoostClassifier, Pool

    mini_router = CatBoostClassifier()
    mini_router.load_model(MINI_ROUTER_PATH)
    base_pool = Pool(
        base_features,
        cat_features=base_manifest["categorical_columns"],
        feature_names=base_manifest["feature_columns"],
    )
    mini_priority = mini_router.predict_proba(base_pool)[:, 1]
    mini_mask = top_mask(mini_priority, MINI_COVERAGE, pairs)
    mini_indices = np.flatnonzero(mini_mask)
    base.log(
        f"MiniLM router selected {len(mini_indices):,}/{len(pairs):,} "
        f"({mini_mask.mean():.3%})",
        started,
    )
    del mini_router, base_pool

    minilm_model = runtime.load_cross_encoder(base.MINILM_PATH, "MiniLM", torch, started)
    if len(mini_indices):
        mini_probability = runtime.score_symmetric(
            minilm_model,
            [left[int(index)] for index in mini_indices],
            [right[int(index)] for index in mini_indices],
            "MiniLM routed",
            torch,
            started,
            collect_lengths=False,
        )["probability"]
    else:
        mini_probability = np.empty(0, dtype=np.float32)
    del minilm_model
    runtime.clear_cuda(torch)

    sequential_manifest = json.loads(
        SEQUENTIAL_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    sequential = sequential_features(
        base_features.iloc[mini_indices],
        bge["probability"][mini_indices],
        mini_probability,
    )
    if sequential.columns.tolist() != sequential_manifest["feature_columns"]:
        raise RuntimeError("sequential RuModern router feature contract mismatch")
    sequential_router = CatBoostClassifier()
    sequential_router.load_model(SEQUENTIAL_ROUTER_PATH)
    if len(sequential):
        sequential_priority = sequential_router.predict_proba(
            Pool(
                sequential,
                cat_features=sequential_manifest["categorical_columns"],
                feature_names=sequential_manifest["feature_columns"],
            )
        )[:, 1]
    else:
        sequential_priority = np.empty(0, dtype=np.float64)
    full_priority = np.full(len(pairs), -np.inf, dtype=np.float64)
    full_priority[mini_indices] = sequential_priority
    ru_mask = top_mask(full_priority, RUMODERN_COVERAGE, pairs, eligible=mini_mask)
    ru_indices = np.flatnonzero(ru_mask)
    base.log(
        f"Sequential router selected {len(ru_indices):,}/{len(pairs):,} "
        f"RuModern pairs ({ru_mask.mean():.3%})",
        started,
    )
    del sequential_router, sequential, base_features, items
    gc.collect()

    rumodern_model = runtime.load_cross_encoder(
        RUMODERN_PATH, "RuModernBERT", torch, started
    )
    if len(ru_indices):
        ru_probability = runtime.score_symmetric(
            rumodern_model,
            [left[int(index)] for index in ru_indices],
            [right[int(index)] for index in ru_indices],
            "RuModernBERT routed",
            torch,
            started,
            collect_lengths=False,
        )["probability"]
    else:
        ru_probability = np.empty(0, dtype=np.float32)
    del rumodern_model
    runtime.clear_cuda(torch)

    prediction = bge["probability"].copy()
    prediction[mini_indices] = (
        0.5 * prediction[mini_indices] + 0.5 * mini_probability
    ).astype(np.float32)
    prediction[ru_indices] = (
        0.5 * prediction[ru_indices] + 0.5 * ru_probability
    ).astype(np.float32)
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
