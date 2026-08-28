#!/usr/bin/env python3
"""BGE backbone plus CatBoost-routed MiniLM on exactly 40% of pairs."""

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


ROOT = Path(__file__).resolve().parent
ROUTER_PATH = ROOT / "router_minilm_classification.cbm"
ROUTER_MANIFEST_PATH = ROOT / "router_manifest.json"
CONCEPT_MAP_PATH = ROOT / "attribute_concept_map.json"
MAX_LENGTH = int(os.getenv("PM_MAX_LENGTH", "384"))
BATCH_SIZE = int(os.getenv("PM_BATCH_SIZE", "1024"))
ROUTE_COVERAGE = float(os.getenv("PM_ROUTE_COVERAGE", "0.40"))
ATTENTION_IMPLEMENTATION = os.getenv("PM_ATTENTION_IMPLEMENTATION", "sdpa")


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def load_inputs(items_path: Path, matches_path: Path, limit: int | None, started: float):
    from serialization_ablation import parse_attributes, serialize_product

    pairs = pd.read_parquet(matches_path, columns=["id1", "id2"])
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        pairs = pairs.iloc[:limit].copy()
    pairs = pairs.reset_index(drop=True)
    if pairs.empty or pairs[["id1", "id2"]].isna().any().any():
        raise ValueError("matches parquet is empty or contains null IDs")

    items = pd.read_parquet(
        items_path, columns=["id", "name", "attributes", "category"]
    )
    _require_columns(items, {"id", "name", "attributes", "category"}, "items parquet")
    needed = pd.unique(pairs[["id1", "id2"]].to_numpy().reshape(-1))
    selected = items.loc[items["id"].isin(needed)].copy().reset_index(drop=True)
    if selected["id"].duplicated().any() or selected["id"].nunique() != len(needed):
        raise ValueError("items parquet does not contain a unique row for every pair ID")

    ranks = base.attribute_rank()
    texts = [
        serialize_product(name, parse_attributes(attributes), "S2_VALUES_ONLY", set(), ranks)
        for name, attributes in selected[["name", "attributes"]].itertuples(
            index=False, name=None
        )
    ]
    text_by_id = pd.Series(texts, index=selected["id"].to_numpy())
    left = text_by_id.loc[pairs["id1"]].tolist()
    right = text_by_id.loc[pairs["id2"]].tolist()
    base.log(
        f"Loaded {len(pairs):,} pairs and serialized {len(selected):,} unique items",
        started,
    )
    return pairs, selected, left, right


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
    torch.cuda.synchronize()
    base.log(
        f"Loaded {label} via SentenceTransformers in "
        f"{time.perf_counter() - load_started:.1f}s",
        started,
    )
    return model


def token_lengths(tokenizer, left: list[str], right: list[str]) -> tuple[np.ndarray, np.ndarray]:
    forward = np.empty(len(left), dtype=np.int32)
    reverse = np.empty(len(left), dtype=np.int32)
    tokenization_batch_size = 2048
    for start in range(0, len(left), tokenization_batch_size):
        stop = min(start + tokenization_batch_size, len(left))
        first = left[start:stop]
        second = right[start:stop]
        size = stop - start
        encoded = tokenizer(
            first + second,
            second + first,
            add_special_tokens=True,
            padding=False,
            truncation="longest_first",
            max_length=MAX_LENGTH,
            return_attention_mask=False,
        )["input_ids"]
        forward[start:stop] = np.fromiter(
            (len(sequence) for sequence in encoded[:size]), dtype=np.int32, count=size
        )
        reverse[start:stop] = np.fromiter(
            (len(sequence) for sequence in encoded[size:]), dtype=np.int32, count=size
        )
    return forward, reverse


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def score_symmetric(
    model,
    left: list[str],
    right: list[str],
    label: str,
    torch,
    started: float,
    *,
    collect_lengths: bool,
) -> dict[str, np.ndarray]:
    if len(left) != len(right) or not left:
        raise ValueError(f"{label} received empty or misaligned pair texts")
    pair_count = len(left)
    lengths_ab = lengths_ba = None
    if collect_lengths:
        length_started = time.perf_counter()
        lengths_ab, lengths_ba = token_lengths(model.tokenizer, left, right)
        base.log(
            f"{label}: collected router token lengths in "
            f"{time.perf_counter() - length_started:.1f}s",
            started,
        )

    directed_pairs = list(zip(left, right)) + list(zip(right, left))
    order = np.argsort(
        np.fromiter((len(a) + len(b) for a, b in directed_pairs), dtype=np.int64),
        kind="stable",
    )
    sorted_pairs = [directed_pairs[int(index)] for index in order]
    del directed_pairs
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
    if len(sorted_logits) != 2 * pair_count:
        raise RuntimeError(f"{label} returned an unexpected number of scores")
    logits = np.empty(2 * pair_count, dtype=np.float32)
    logits[order] = sorted_logits
    logits_ab, logits_ba = logits[:pair_count], logits[pair_count:]
    probability_ab, probability_ba = _sigmoid(logits_ab), _sigmoid(logits_ba)
    base.log(
        f"{label}: {2 * pair_count:,} directed forwards in {elapsed:.1f}s "
        f"({2 * pair_count / elapsed:.1f}/s), peak CUDA "
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB",
        started,
    )
    result = {
        "probability": ((probability_ab + probability_ba) * 0.5).astype(np.float32),
        "logit": ((logits_ab + logits_ba) * 0.5).astype(np.float32),
        "score_order_gap": np.abs(probability_ab - probability_ba).astype(np.float32),
    }
    if collect_lengths:
        assert lengths_ab is not None and lengths_ba is not None
        result["token_length_ab"] = lengths_ab
        result["token_length_ba"] = lengths_ba
    return result


def build_router_features(
    pairs: pd.DataFrame,
    items: pd.DataFrame,
    bge: dict[str, np.ndarray],
    started: float,
) -> tuple[pd.DataFrame, dict]:
    from src.catboost1_early_exit import extract_pair_features

    manifest = json.loads(ROUTER_MANIFEST_PATH.read_text(encoding="utf-8"))
    concept_map = json.loads(CONCEPT_MAP_PATH.read_text(encoding="utf-8"))
    feature_started = time.perf_counter()
    cheap, _ = extract_pair_features(pairs, items, concept_map, {})
    cheap = cheap.loc[:, [column for column in cheap if not column.startswith("rule_")]]
    probability = np.asarray(bge["probability"], dtype=np.float64)
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    cheap["bge_probability"] = probability.astype(np.float32)
    cheap["bge_logit"] = np.log(clipped / (1.0 - clipped)).astype(np.float32)
    cheap["bge_abs_from_half"] = np.abs(probability - 0.5).astype(np.float32)
    cheap["bge_uncertainty"] = (1.0 - 2.0 * np.abs(probability - 0.5)).astype(np.float32)
    cheap["bge_entropy"] = (
        -(clipped * np.log(clipped) + (1.0 - clipped) * np.log1p(-clipped))
    ).astype(np.float32)
    cheap["bge_raw_logit"] = bge["logit"]
    cheap["bge_score_order_gap"] = bge["score_order_gap"]
    cheap["bge_token_length_ab"] = bge["token_length_ab"].astype(np.float32)
    cheap["bge_token_length_ba"] = bge["token_length_ba"].astype(np.float32)
    cheap["bge_token_length_max"] = np.maximum(
        bge["token_length_ab"], bge["token_length_ba"]
    ).astype(np.float32)
    categorical = manifest["categorical_columns"]
    for column in categorical:
        cheap[column] = cheap[column].fillna("__missing__").astype(str)
    expected = manifest["feature_columns"]
    missing = [column for column in expected if column not in cheap]
    extras = [column for column in cheap if column not in expected]
    if missing or extras:
        raise RuntimeError(f"router feature contract mismatch; missing={missing}, extras={extras}")
    base.log(
        f"Built {len(expected)} CatBoost router features in "
        f"{time.perf_counter() - feature_started:.1f}s",
        started,
    )
    return cheap.loc[:, expected], manifest


def route_top_budget(
    priority: np.ndarray, pairs: pd.DataFrame, coverage: float = ROUTE_COVERAGE
) -> np.ndarray:
    if not 0.0 <= coverage <= 1.0:
        raise ValueError("route coverage must lie in [0, 1]")
    priority = np.asarray(priority, dtype=np.float64)
    route_count = int(math.floor(len(priority) * coverage + 1e-12))
    order = np.lexsort(
        (
            pairs["id2"].to_numpy(dtype=np.int64),
            pairs["id1"].to_numpy(dtype=np.int64),
            -priority,
        )
    )
    mask = np.zeros(len(priority), dtype=bool)
    mask[order[:route_count]] = True
    return mask


def clear_cuda(torch) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def main() -> int:
    args = base.parse_args()
    started = time.perf_counter()
    base.configure_writable_cache(args.output_path)
    pairs, items, left, right = load_inputs(
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
        f"batch={BATCH_SIZE}, symmetric=True, MiniLM coverage={ROUTE_COVERAGE:.0%}",
        started,
    )

    bge_model = load_cross_encoder(base.BGE_PATH, "BGE", torch, started)
    bge = score_symmetric(
        bge_model, left, right, "BGE", torch, started, collect_lengths=True
    )
    del bge_model
    clear_cuda(torch)

    router_frame, router_manifest = build_router_features(pairs, items, bge, started)
    from catboost import CatBoostClassifier, Pool

    router = CatBoostClassifier()
    router.load_model(ROUTER_PATH)
    router_scores = router.predict_proba(
        Pool(
            router_frame,
            cat_features=router_manifest["categorical_columns"],
            feature_names=router_manifest["feature_columns"],
        )
    )[:, 1]
    routed = route_top_budget(router_scores, pairs)
    routed_indices = np.flatnonzero(routed)
    base.log(
        f"CatBoost routed {len(routed_indices):,}/{len(pairs):,} pairs "
        f"({routed.mean():.3%})",
        started,
    )
    del router, router_frame, items
    gc.collect()

    minilm_model = load_cross_encoder(base.MINILM_PATH, "MiniLM", torch, started)
    minilm = score_symmetric(
        minilm_model,
        [left[int(index)] for index in routed_indices],
        [right[int(index)] for index in routed_indices],
        "MiniLM routed",
        torch,
        started,
        collect_lengths=False,
    )["probability"]
    del minilm_model
    clear_cuda(torch)

    prediction = bge["probability"].copy()
    prediction[routed_indices] = (
        0.5 * prediction[routed_indices] + 0.5 * minilm
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

