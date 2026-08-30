#!/usr/bin/env python3
"""Fast one-way BGE 100% -> MiniLM 40% -> RuModernBERT 5% ensemble."""

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
import runtime_base as runtime
from fast_benefit_router import runtime_feature_frame
from fast_sequential_router import sequential_feature_frame


ROOT = Path(__file__).resolve().parent
RUMODERN_PATH = ROOT / "models/rumodernbert_final"
MINI_ROUTER_PATH = ROOT / "router_minilm_oneway_score_category.cbm"
RU_ROUTER_PATH = ROOT / "router_rumodern_oneway_score_category.cbm"
MINI_MANIFEST_PATH = ROOT / "mini_router_manifest.json"
RU_MANIFEST_PATH = ROOT / "rumodern_router_manifest.json"
MINI_COVERAGE = float(os.getenv("PM_MINILM_COVERAGE", "0.40"))
RU_COVERAGE = float(os.getenv("PM_RUMODERN_COVERAGE", "0.05"))


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


def main() -> int:
    args = base.parse_args()
    started = time.perf_counter()
    base.configure_writable_cache(args.output_path)
    pairs, left, right, left_titles, right_titles, categories = runtime.load_inputs(
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
        f"batch={runtime.BATCH_SIZE}, direction=AB, "
        f"MiniLM={MINI_COVERAGE:.0%}, RuModern={RU_COVERAGE:.0%}",
        started,
    )

    bge_model = runtime.load_cross_encoder(base.BGE_PATH, "BGE", torch, started)
    bge_probability, bge_logits = runtime.score_oneway(
        bge_model, left, right, "BGE", torch, started
    )
    del bge_model
    runtime.clear_cuda(torch)

    mini_manifest = json.loads(MINI_MANIFEST_PATH.read_text(encoding="utf-8"))
    feature_started = time.perf_counter()
    base_features = runtime_feature_frame(
        categories,
        left_titles,
        right_titles,
        bge_probability,
        bge_logits,
        mini_manifest["variant"],
    )
    if base_features.columns.tolist() != mini_manifest["feature_columns"]:
        raise RuntimeError("compact MiniLM router feature contract mismatch")
    from catboost import CatBoostClassifier, Pool

    mini_router = CatBoostClassifier()
    mini_router.load_model(MINI_ROUTER_PATH)
    mini_priority = mini_router.predict_proba(
        Pool(base_features, cat_features=mini_manifest["categorical_columns"])
    )[:, 1]
    mini_mask = top_mask(mini_priority, MINI_COVERAGE, pairs)
    mini_indices = np.flatnonzero(mini_mask)
    base.log(
        f"Mini router features+predict in {time.perf_counter() - feature_started:.1f}s; "
        f"selected {len(mini_indices):,}/{len(pairs):,}",
        started,
    )
    del mini_router

    mini_model = runtime.load_cross_encoder(base.MINILM_PATH, "MiniLM", torch, started)
    mini_probability, _ = runtime.score_oneway(
        mini_model,
        [left[int(index)] for index in mini_indices],
        [right[int(index)] for index in mini_indices],
        "MiniLM routed",
        torch,
        started,
    )
    del mini_model
    runtime.clear_cuda(torch)

    ru_manifest = json.loads(RU_MANIFEST_PATH.read_text(encoding="utf-8"))
    sequential = sequential_feature_frame(
        base_features.iloc[mini_indices],
        bge_probability[mini_indices],
        mini_probability,
    )
    if sequential.columns.tolist() != ru_manifest["feature_columns"]:
        raise RuntimeError("compact RuModern router feature contract mismatch")
    ru_router = CatBoostClassifier()
    ru_router.load_model(RU_ROUTER_PATH)
    if len(sequential):
        sequential_priority = ru_router.predict_proba(
            Pool(sequential, cat_features=ru_manifest["categorical_columns"])
        )[:, 1]
    else:
        sequential_priority = np.empty(0, dtype=np.float64)
    full_priority = np.full(len(pairs), -np.inf, dtype=np.float64)
    full_priority[mini_indices] = sequential_priority
    ru_mask = top_mask(full_priority, RU_COVERAGE, pairs, eligible=mini_mask)
    ru_indices = np.flatnonzero(ru_mask)
    base.log(
        f"Ru router selected {len(ru_indices):,}/{len(pairs):,} inside MiniLM subset",
        started,
    )
    del ru_router
    del sequential, base_features
    gc.collect()

    ru_model = runtime.load_cross_encoder(RUMODERN_PATH, "RuModernBERT", torch, started)
    ru_probability, _ = runtime.score_oneway(
        ru_model,
        [left[int(index)] for index in ru_indices],
        [right[int(index)] for index in ru_indices],
        "RuModernBERT routed",
        torch,
        started,
    )
    del ru_model
    runtime.clear_cuda(torch)

    prediction = bge_probability.copy()
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
