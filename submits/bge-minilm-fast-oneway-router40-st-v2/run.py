#!/usr/bin/env python3
"""One-way BGE plus compact CatBoost-routed one-way MiniLM on 40% of pairs."""

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
from fast_benefit_router import runtime_feature_frame


ROOT = Path(__file__).resolve().parent
ROUTER_PATH = ROOT / "router_score_title.cbm"
ROUTER_MANIFEST_PATH = ROOT / "router_manifest.json"
MAX_LENGTH = int(os.getenv("PM_MAX_LENGTH", "384"))
BATCH_SIZE = int(os.getenv("PM_BATCH_SIZE", "1024"))
ROUTE_COVERAGE = float(os.getenv("PM_ROUTE_COVERAGE", "0.40"))
ATTENTION_IMPLEMENTATION = os.getenv("PM_ATTENTION_IMPLEMENTATION", "sdpa")


def load_inputs(items_path: Path, matches_path: Path, limit: int | None, started: float):
    from serialization_ablation import parse_attributes, serialize_product

    pairs = pd.read_parquet(matches_path, columns=["id1", "id2"])
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        pairs = pairs.iloc[:limit].copy()
    pairs = pairs.reset_index(drop=True)
    if pairs.empty or pairs.isna().any().any():
        raise ValueError("matches parquet is empty or contains null IDs")
    items = pd.read_parquet(
        items_path, columns=["id", "name", "attributes", "category"]
    )
    needed = pd.unique(pairs[["id1", "id2"]].to_numpy().reshape(-1))
    selected = items.loc[items["id"].isin(needed)].copy().reset_index(drop=True)
    if selected["id"].duplicated().any() or selected["id"].nunique() != len(needed):
        raise ValueError("items parquet does not contain every referenced ID exactly once")

    ranks = base.attribute_rank()
    serialized = [
        serialize_product(name, parse_attributes(attributes), "S2_VALUES_ONLY", set(), ranks)
        for name, attributes in selected[["name", "attributes"]].itertuples(
            index=False, name=None
        )
    ]
    selected_ids = selected["id"].to_numpy()
    text_by_id = pd.Series(serialized, index=selected_ids)
    title_by_id = pd.Series(selected["name"].tolist(), index=selected_ids)
    category_by_id = pd.Series(selected["category"].astype(str).tolist(), index=selected_ids)
    left_ids = pairs["id1"]
    right_ids = pairs["id2"]
    left = text_by_id.loc[left_ids].tolist()
    right = text_by_id.loc[right_ids].tolist()
    left_titles = title_by_id.loc[left_ids].tolist()
    right_titles = title_by_id.loc[right_ids].tolist()
    categories = category_by_id.loc[left_ids].tolist()
    right_categories = category_by_id.loc[right_ids].tolist()
    if categories != right_categories:
        raise ValueError("cross-category pairs are not supported")
    base.log(
        f"Loaded {len(pairs):,} pairs and serialized {len(selected):,} unique items",
        started,
    )
    return pairs, left, right, left_titles, right_titles, categories


def load_cross_encoder(model_path: Path, label: str, torch, started: float):
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
    if model_path.name == "rumodernbert-base-human-ft-v1":
        reference_compile = getattr(model.model.config, "reference_compile", None)
        if reference_compile is not False:
            raise RuntimeError(
                f"RuModernBERT reference_compile must be False, got {reference_compile!r}"
            )
        base.log("RuModernBERT torch.compile explicitly disabled", started)
    torch.cuda.synchronize()
    base.log(
        f"Loaded {label} via SentenceTransformers in "
        f"{time.perf_counter() - load_started:.1f}s",
        started,
    )
    return model


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def score_oneway(model, left: list[str], right: list[str], label: str, torch, started: float):
    pair_count = len(left)
    if pair_count != len(right):
        raise ValueError(f"{label} pair texts are misaligned")
    if not pair_count:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    order = np.argsort(
        np.fromiter((len(a) + len(b) for a, b in zip(left, right)), dtype=np.int64),
        kind="stable",
    )
    sorted_pairs = [(left[int(index)], right[int(index)]) for index in order]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    inference_started = time.perf_counter()
    with torch.inference_mode():
        raw = model.predict(
            sorted_pairs,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            activation_fn=torch.nn.Identity(),
            convert_to_numpy=True,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - inference_started
    sorted_logits = np.asarray(raw, dtype=np.float32).reshape(-1)
    if len(sorted_logits) != pair_count:
        raise RuntimeError(f"{label} returned an unexpected number of scores")
    logits = np.empty(pair_count, dtype=np.float32)
    logits[order] = sorted_logits
    base.log(
        f"{label}: {pair_count:,} one-way forwards in {elapsed:.1f}s "
        f"({pair_count / elapsed:.1f}/s), peak CUDA "
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB",
        started,
    )
    return sigmoid(logits), logits


def route_top_budget(priority: np.ndarray, pairs: pd.DataFrame) -> np.ndarray:
    if not 0.0 <= ROUTE_COVERAGE <= 1.0:
        raise ValueError("route coverage must lie in [0, 1]")
    route_count = int(math.floor(len(priority) * ROUTE_COVERAGE + 1e-12))
    order = np.lexsort(
        (
            pairs["id2"].to_numpy(dtype=np.int64),
            pairs["id1"].to_numpy(dtype=np.int64),
            -np.asarray(priority, dtype=np.float64),
        )
    )
    result = np.zeros(len(priority), dtype=bool)
    result[order[:route_count]] = True
    return result


def clear_cuda(torch) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def main() -> int:
    args = base.parse_args()
    started = time.perf_counter()
    base.configure_writable_cache(args.output_path)
    pairs, left, right, left_titles, right_titles, categories = load_inputs(
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
        f"CUDA={torch.cuda.get_device_name(0)}, max_length={MAX_LENGTH}, "
        f"batch={BATCH_SIZE}, direction=AB, MiniLM coverage={ROUTE_COVERAGE:.0%}",
        started,
    )

    bge_model = load_cross_encoder(base.BGE_PATH, "BGE", torch, started)
    bge_probability, bge_logits = score_oneway(
        bge_model, left, right, "BGE", torch, started
    )
    del bge_model
    clear_cuda(torch)

    router_manifest = json.loads(ROUTER_MANIFEST_PATH.read_text(encoding="utf-8"))
    feature_started = time.perf_counter()
    router_features = runtime_feature_frame(
        categories,
        left_titles,
        right_titles,
        bge_probability,
        bge_logits,
        router_manifest["variant"],
    )
    if router_features.columns.tolist() != router_manifest["feature_columns"]:
        raise RuntimeError("compact router feature contract mismatch")
    from catboost import CatBoostClassifier, Pool

    router = CatBoostClassifier()
    router.load_model(ROUTER_PATH)
    router_scores = router.predict_proba(
        Pool(router_features, cat_features=router_manifest["categorical_columns"])
    )[:, 1]
    routed = route_top_budget(router_scores, pairs)
    routed_indices = np.flatnonzero(routed)
    base.log(
        f"Compact CatBoost features+predict in {time.perf_counter() - feature_started:.1f}s; "
        f"routed {len(routed_indices):,}/{len(pairs):,} ({routed.mean():.3%})",
        started,
    )
    del router, router_features
    gc.collect()

    minilm_model = load_cross_encoder(base.MINILM_PATH, "MiniLM", torch, started)
    mini_probability, _ = score_oneway(
        minilm_model,
        [left[int(index)] for index in routed_indices],
        [right[int(index)] for index in routed_indices],
        "MiniLM routed",
        torch,
        started,
    )
    del minilm_model
    clear_cuda(torch)

    prediction = bge_probability.copy()
    prediction[routed_indices] = (
        0.5 * prediction[routed_indices] + 0.5 * mini_probability
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
