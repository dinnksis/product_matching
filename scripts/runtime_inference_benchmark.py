#!/usr/bin/env python3
"""Benchmark frozen cross-encoder checkpoints without changing their scores."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score


MODEL_CONFIG = {
    "bge": {
        "label": "bge_reranker_v2_m3_human_ft_v1",
        "baseline_batch": 32,
        "batch_candidates": [64, 96, 128, 192, 256],
    },
    "minilm": {
        "label": "minilm_5ep_synthetic_pretrain_human_ft_s2_v1",
        "baseline_batch": 192,
        "batch_candidates": [256, 384, 512, 768, 1024],
    },
    "rumodernbert": {
        "label": "rumodernbert_base_random_head_human_ft_v1",
        "baseline_batch": 128,
        "batch_candidates": [192, 256, 384, 512],
    },
}
SPLITS = ("iid", "hard", "ood")
MAX_LENGTH = 384


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=30_000)
    parser.add_argument("--probe-size", type=int, default=2_048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip full validation and optional backends; shortlist native runners only",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def release_cuda(torch: Any, *objects: Any) -> None:
    del objects
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def macro_ap(frame: pd.DataFrame, score: np.ndarray) -> float:
    values = []
    for _, indices in frame.groupby("category", sort=True).groups.items():
        positions = np.asarray(list(indices), dtype=np.int64)
        target = frame.loc[positions, "target"].to_numpy(dtype=np.int8)
        if np.unique(target).size < 2:
            continue
        values.append(average_precision_score(target, score[positions]))
    if not values:
        raise RuntimeError("macro AP cannot be computed: no category has both classes")
    return float(np.mean(values))


def score_diagnostics(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    reference = frame["reference_score"].to_numpy(dtype=np.float64)
    candidate = np.asarray(scores, dtype=np.float64)
    difference = np.abs(candidate - reference)
    return {
        "pearson": float(pearsonr(reference, candidate).statistic),
        "spearman": float(spearmanr(reference, candidate).statistic),
        "mean_abs_difference": float(difference.mean()),
        "max_abs_difference": float(difference.max()),
        "macro_ap": macro_ap(frame, candidate),
        "reference_macro_ap": macro_ap(frame, reference),
        "ap_delta": macro_ap(frame, candidate) - macro_ap(frame, reference),
    }


def reference_path(input_root: Path, staged_name: str) -> Path:
    matches = list(input_root.glob(f"**/{staged_name}"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {staged_name}, found {matches}")
    return matches[0]


def load_references(
    input_root: Path,
    manifest: dict[str, Any],
    sample_size: int,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    by_model: dict[str, dict[str, pd.DataFrame]] = {}
    required = [
        "id1", "id2", "target", "category_1", "product_text_1", "product_text_2", "score"
    ]
    for model_name in MODEL_CONFIG:
        by_model[model_name] = {}
        for split in SPLITS:
            declaration = manifest["references"][model_name][split]
            path = reference_path(input_root, declaration["staged_name"])
            if path.stat().st_size != declaration["bytes"] or sha256_file(path) != declaration["sha256"]:
                raise RuntimeError(f"Reference file changed: {path}")
            by_model[model_name][split] = pd.read_parquet(path, columns=required)

    canonical: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        base = by_model["bge"][split].rename(columns={"category_1": "category"}).copy()
        keys = base[["id1", "id2"]]
        if keys.duplicated().any():
            raise RuntimeError(f"Duplicate pair identifiers in {split}")
        for model_name in ("minilm", "rumodernbert"):
            other = by_model[model_name][split]
            joined = keys.merge(
                other[["id1", "id2", "target", "category_1", "product_text_1", "product_text_2"]],
                on=["id1", "id2"], how="left", validate="one_to_one", sort=False,
            )
            if joined["target"].isna().any() or len(joined) != len(base):
                raise RuntimeError(f"Pair identifiers differ for {model_name}/{split}")
            if not np.array_equal(joined["target"].to_numpy(), base["target"].to_numpy()):
                raise RuntimeError(f"Targets differ for {model_name}/{split}")
            if not np.array_equal(joined["category_1"].astype(str), base["category"].astype(str)):
                raise RuntimeError(f"Categories differ for {model_name}/{split}")
            if not np.array_equal(joined["product_text_1"].astype(str), base["product_text_1"].astype(str)):
                raise RuntimeError(f"Left serialization differs for {model_name}/{split}")
            if not np.array_equal(joined["product_text_2"].astype(str), base["product_text_2"].astype(str)):
                raise RuntimeError(f"Right serialization differs for {model_name}/{split}")
        canonical[split] = base.reset_index(drop=True)

    total = sum(len(frame) for frame in canonical.values())
    allocations = {split: min(len(frame), max(1, round(sample_size * len(frame) / total))) for split, frame in canonical.items()}
    while sum(allocations.values()) > sample_size:
        split = max(allocations, key=lambda name: allocations[name])
        allocations[split] -= 1
    while sum(allocations.values()) < min(sample_size, total):
        available = [name for name in SPLITS if allocations[name] < len(canonical[name])]
        allocations[max(available, key=lambda name: len(canonical[name]) - allocations[name])] += 1

    sampled = []
    for offset, split in enumerate(SPLITS):
        frame = canonical[split]
        count = allocations[split]
        if count < len(frame):
            frame = frame.sample(n=count, random_state=seed + offset).sort_index()
        frame = frame.reset_index(drop=True)
        frame["split"] = split
        sampled.append(frame)
    sample = pd.concat(sampled, ignore_index=True)
    return canonical, sample


def reference_for_model(
    input_root: Path,
    manifest: dict[str, Any],
    model_name: str,
    frame: pd.DataFrame,
    split: str | None = None,
) -> pd.DataFrame:
    pieces = []
    requested_splits = (split,) if split else SPLITS
    for current in requested_splits:
        declaration = manifest["references"][model_name][current]
        ref = pd.read_parquet(
            reference_path(input_root, declaration["staged_name"]),
            columns=["id1", "id2", "score"],
        ).rename(columns={"score": "reference_score"})
        ref["split"] = current
        pieces.append(ref)
    reference = pd.concat(pieces, ignore_index=True)
    join_keys = ["id1", "id2", "split"] if "split" in frame else ["id1", "id2"]
    result = frame.drop(columns=["score", "reference_score"], errors="ignore").merge(
        reference, on=join_keys, how="left", validate="one_to_one", sort=False
    )
    if result["reference_score"].isna().any() or len(result) != len(frame):
        raise RuntimeError(f"Could not align references for {model_name}")
    return result.reset_index(drop=True)


def measure_raw_preprocessing(
    input_root: Path,
    sample: pd.DataFrame,
    frequency_path: Path,
) -> dict[str, float]:
    from src.serialization_ablation import parse_attributes, serialize_product

    candidates = list(input_root.glob("**/items_human.parquet"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one items_human.parquet, found {candidates}")
    started = time.perf_counter()
    items = pd.read_parquet(candidates[0], columns=["id", "name", "attributes"])
    read_seconds = time.perf_counter() - started
    needed = set(sample[["id1", "id2"]].to_numpy().reshape(-1).tolist())
    selected = items.loc[items["id"].isin(needed)].copy()
    if selected["id"].nunique() != len(needed):
        raise RuntimeError("Raw item catalogue does not cover benchmark sample")
    frequency = pd.read_csv(frequency_path, usecols=["attribute_name"])
    ranks = {str(key): index for index, key in enumerate(frequency["attribute_name"].tolist())}
    serialization_started = time.perf_counter()
    selected["text"] = [
        serialize_product(row.name, parse_attributes(row.attributes), "S2_VALUES_ONLY", set(), ranks)
        for row in selected.itertuples(index=False)
    ]
    serialization_seconds = time.perf_counter() - serialization_started
    text_by_id = selected.set_index("id")["text"]
    left = text_by_id.loc[sample["id1"]].to_numpy(dtype=str)
    right = text_by_id.loc[sample["id2"]].to_numpy(dtype=str)
    if not np.array_equal(left, sample["product_text_1"].astype(str).to_numpy()):
        raise RuntimeError("Raw S2 left text differs from frozen training serialization")
    if not np.array_equal(right, sample["product_text_2"].astype(str).to_numpy()):
        raise RuntimeError("Raw S2 right text differs from frozen training serialization")
    return {
        "data_read_seconds": read_seconds,
        "serialization_seconds": serialization_seconds,
        "preprocessing_seconds": read_seconds + serialization_seconds,
        "unique_items": float(len(selected)),
    }


def load_native_model(model_dir: Path, attention: str = "sdpa") -> tuple[Any, Any, Any, float, str]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        raise RuntimeError(f"Tokenizer has no pad token: {model_dir}")
    kwargs: dict[str, Any] = {"local_files_only": True, "attn_implementation": attention}
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, **kwargs).cuda().eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    # torch.cuda.is_bf16_supported() can return True on Kaggle T4 even though
    # compute capability 7.5 has no native BF16 tensor-core path. That makes the
    # reranker several times slower. Require Ampere (SM80) or newer explicitly;
    # H100 (SM90) still uses BF16 as intended.
    compute_capability = torch.cuda.get_device_capability()
    native_bf16 = torch.cuda.is_bf16_supported() and compute_capability[0] >= 8
    dtype_name = "bfloat16" if native_bf16 else "float16"
    return torch, tokenizer, model, load_seconds, dtype_name


def predict_onthefly(
    torch: Any,
    tokenizer: Any,
    model: Any,
    left: list[str],
    right: list[str],
    batch_size: int,
    dtype_name: str,
) -> tuple[np.ndarray, dict[str, float]]:
    scores = np.empty(len(left), dtype=np.float32)
    order = np.argsort(
        np.fromiter((len(a) + len(b) for a, b in zip(left, right)), dtype=np.int64),
        kind="stable",
    )
    tokenization_seconds = 0.0
    inference_seconds = 0.0
    started = time.perf_counter()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        for start in range(0, len(left), batch_size):
            indices = order[start : start + batch_size]
            batch_left = [left[index] for index in indices]
            batch_right = [right[index] for index in indices]
            token_started = time.perf_counter()
            encoded = tokenizer(
                batch_left + batch_right,
                batch_right + batch_left,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )
            tokenization_seconds += time.perf_counter() - token_started
            forward_started = time.perf_counter()
            encoded = {key: value.cuda(non_blocking=True) for key, value in encoded.items()}
            logits = model(**encoded).logits.reshape(-1).float()
            probability = torch.sigmoid(logits)
            size = len(indices)
            symmetric = (probability[:size] + probability[size:]) * 0.5
            scores[indices] = symmetric.cpu().numpy()
            torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - forward_started
    total_seconds = time.perf_counter() - started
    return scores, {
        "tokenize_seconds": tokenization_seconds,
        "inference_seconds": inference_seconds,
        "total_seconds": total_seconds,
        "pairs_per_second": len(left) / total_seconds,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }


def predict_pretokenized(
    torch: Any,
    tokenizer: Any,
    model: Any,
    left: list[str],
    right: list[str],
    batch_size: int,
    dtype_name: str,
) -> tuple[np.ndarray, dict[str, float]]:
    """Tokenize in large Rust batches, then bucket by exact token length."""

    token_started = time.perf_counter()
    encoded_lists: dict[str, list[list[int]]] = {}
    combined_left = left + right
    combined_right = right + left
    tokenizer_batch = 2_048
    for start in range(0, len(combined_left), tokenizer_batch):
        encoded = tokenizer(
            combined_left[start : start + tokenizer_batch],
            combined_right[start : start + tokenizer_batch],
            padding=False,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        for key, values in encoded.items():
            encoded_lists.setdefault(key, []).extend(values)
    tokenization_seconds = time.perf_counter() - token_started
    pair_count = len(left)
    lengths = np.asarray(
        [
            max(len(encoded_lists["input_ids"][index]), len(encoded_lists["input_ids"][index + pair_count]))
            for index in range(pair_count)
        ],
        dtype=np.int32,
    )
    order = np.argsort(lengths, kind="stable")
    scores = np.empty(pair_count, dtype=np.float32)
    inference_seconds = 0.0
    started = time.perf_counter()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        for start in range(0, pair_count, batch_size):
            indices = order[start : start + batch_size]
            flat_indices = np.concatenate([indices, indices + pair_count])
            features = [
                {key: encoded_lists[key][int(index)] for key in encoded_lists}
                for index in flat_indices
            ]
            batch = tokenizer.pad(
                features, padding=True, pad_to_multiple_of=8, return_tensors="pt"
            )
            forward_started = time.perf_counter()
            batch = {
                key: value.pin_memory().cuda(non_blocking=True)
                for key, value in batch.items()
            }
            logits = model(**batch).logits.reshape(-1).float()
            probability = torch.sigmoid(logits)
            size = len(indices)
            symmetric = (probability[:size] + probability[size:]) * 0.5
            scores[indices] = symmetric.cpu().numpy()
            torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - forward_started
    inference_wall = time.perf_counter() - started
    total_seconds = tokenization_seconds + inference_wall
    return scores, {
        "tokenize_seconds": tokenization_seconds,
        "inference_seconds": inference_seconds,
        "total_seconds": total_seconds,
        "pairs_per_second": pair_count / total_seconds,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }


def benchmark_call(
    *,
    model_name: str,
    backend: str,
    phase: str,
    batch_size: int | None,
    frame: pd.DataFrame,
    load_seconds: float,
    precision: str,
    call: Callable[[], tuple[np.ndarray, dict[str, float]]],
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    import torch

    try:
        scores, timing = call()
        diagnostics = score_diagnostics(frame, scores)
        row = {
            "model": model_name,
            "backend": backend,
            "phase": phase,
            "status": "ok",
            "pairs": len(frame),
            "batch": batch_size,
            "max_length": MAX_LENGTH,
            "precision": precision,
            "load_seconds": load_seconds,
            **timing,
            **diagnostics,
            "error": "",
        }
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        row = {
            "model": model_name, "backend": backend, "phase": phase,
            "status": "oom", "pairs": len(frame), "batch": batch_size,
            "max_length": MAX_LENGTH, "precision": precision,
            "load_seconds": load_seconds, "error": str(error)[:500],
        }
        scores = None
    except Exception as error:
        row = {
            "model": model_name, "backend": backend, "phase": phase,
            "status": "error", "pairs": len(frame), "batch": batch_size,
            "max_length": MAX_LENGTH, "precision": precision,
            "load_seconds": load_seconds,
            "error": f"{type(error).__name__}: {error}"[:500],
        }
        scores = None
        traceback.print_exc()
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False, default=str), flush=True)
    return scores, row


def sentence_transformers_probe(
    model_name: str,
    model_dir: Path,
    frame: pd.DataFrame,
    batch_size: int,
    rows: list[dict[str, Any]],
) -> None:
    if importlib.util.find_spec("sentence_transformers") is None:
        rows.append({
            "model": model_name, "backend": "sentence_transformers", "phase": "compatibility",
            "status": "unsupported", "pairs": 0, "batch": batch_size,
            "max_length": MAX_LENGTH, "error": "sentence-transformers is not installed in the Kaggle image",
        })
        return
    import torch
    from sentence_transformers import CrossEncoder

    started = time.perf_counter()
    try:
        cross_encoder = CrossEncoder(
            str(model_dir), max_length=MAX_LENGTH,
            model_kwargs={"local_files_only": True, "attn_implementation": "sdpa"},
        )
        load_seconds = time.perf_counter() - started
        left = frame["product_text_1"].astype(str).tolist()
        right = frame["product_text_2"].astype(str).tolist()

        def call() -> tuple[np.ndarray, dict[str, float]]:
            inference_started = time.perf_counter()
            raw = cross_encoder.predict(
                list(zip(left + right, right + left)),
                batch_size=batch_size,
                show_progress_bar=False,
                activation_fn=torch.nn.Identity(),
                convert_to_numpy=True,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - inference_started
            logits = np.asarray(raw, dtype=np.float32).reshape(-1)
            probability = 1.0 / (1.0 + np.exp(-logits))
            scores = (probability[: len(left)] + probability[len(left) :]) * 0.5
            return scores, {
                "tokenize_seconds": None,
                "inference_seconds": elapsed,
                "total_seconds": elapsed,
                "pairs_per_second": len(left) / elapsed,
                "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
            }

        benchmark_call(
            model_name=model_name, backend="sentence_transformers", phase="probe",
            batch_size=batch_size, frame=frame, load_seconds=load_seconds,
            precision="library_default", call=call, rows=rows,
        )
    except Exception as error:
        rows.append({
            "model": model_name, "backend": "sentence_transformers", "phase": "compatibility",
            "status": "error", "pairs": len(frame), "batch": batch_size,
            "max_length": MAX_LENGTH, "load_seconds": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}"[:500],
        })
        traceback.print_exc()
    finally:
        if "cross_encoder" in locals():
            del cross_encoder
        release_cuda(torch)


def flash_attention_probe(
    model_name: str,
    model_dir: Path,
    frame: pd.DataFrame,
    batch_size: int,
    rows: list[dict[str, Any]],
) -> None:
    if importlib.util.find_spec("flash_attn") is None:
        rows.append({
            "model": model_name, "backend": "transformers_flash_attention_2",
            "phase": "compatibility", "status": "unsupported", "pairs": 0,
            "batch": batch_size, "max_length": MAX_LENGTH,
            "error": "flash-attn is not installed in the Kaggle image",
        })
        return
    import torch

    try:
        torch_module, tokenizer, model, load_seconds, precision = load_native_model(
            model_dir, attention="flash_attention_2"
        )
        left = frame["product_text_1"].astype(str).tolist()
        right = frame["product_text_2"].astype(str).tolist()
        benchmark_call(
            model_name=model_name, backend="transformers_flash_attention_2", phase="probe",
            batch_size=batch_size, frame=frame, load_seconds=load_seconds,
            precision=precision,
            call=lambda: predict_onthefly(
                torch_module, tokenizer, model, left, right, batch_size, precision
            ), rows=rows,
        )
    except Exception as error:
        rows.append({
            "model": model_name, "backend": "transformers_flash_attention_2",
            "phase": "compatibility", "status": "error", "pairs": len(frame),
            "batch": batch_size, "max_length": MAX_LENGTH,
            "error": f"{type(error).__name__}: {error}"[:500],
        })
        traceback.print_exc()
    finally:
        if "model" in locals():
            del model
        if "tokenizer" in locals():
            del tokenizer
        release_cuda(torch)


def vllm_compatibility(model_name: str, model_dir: Path, rows: list[dict[str, Any]]) -> None:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    architecture = (config.get("architectures") or [""])[0]
    row: dict[str, Any] = {
        "model": model_name, "backend": "vllm", "phase": "compatibility",
        "status": "unsupported", "pairs": 0, "batch": None,
        "max_length": MAX_LENGTH, "architecture": architecture,
    }
    if importlib.util.find_spec("vllm") is None:
        row["error"] = "vLLM is not installed in the Kaggle image"
    else:
        try:
            from vllm.model_executor.models import ModelRegistry

            checker = getattr(ModelRegistry, "is_architecture_supported", None)
            if checker is None:
                checker = getattr(ModelRegistry, "is_model_supported", None)
            architecture_supported = bool(checker(architecture)) if checker else None
            row["architecture_supported"] = architecture_supported
        except Exception as error:
            row["architecture_supported"] = None
            row["registry_error"] = f"{type(error).__name__}: {error}"[:300]
        if row.get("architecture_supported") is False:
            row["error"] = f"vLLM ModelRegistry does not support {architecture}"
        elif int(config.get("num_labels", len(config.get("id2label", {"0": "LABEL_0"})))) == 1:
            row["error"] = (
                "Checkpoint uses one BCE logit; vLLM classify exposes class probabilities "
                "and cannot be treated as sigmoid(logit) without an equivalence adapter."
            )
        else:
            row["error"] = "Exact paired-token classification interface requires a verified adapter"
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    canonical, base_sample = load_references(
        args.input_root, manifest, args.sample_size, args.seed
    )
    frequency = reference_path(args.input_root, manifest["attribute_frequency"]["staged_name"])
    preprocessing = measure_raw_preprocessing(args.input_root, base_sample, frequency)
    print(json.dumps({"preprocessing": preprocessing}, ensure_ascii=False), flush=True)

    rows: list[dict[str, Any]] = []
    winners: dict[str, dict[str, Any]] = {}
    for model_name, profile in MODEL_CONFIG.items():
        print(f"===== {model_name} =====", flush=True)
        model_dir = args.checkpoint_root / model_name
        sample = reference_for_model(args.input_root, manifest, model_name, base_sample)
        probe = sample.iloc[: min(args.probe_size, len(sample))].reset_index(drop=True)
        left = sample["product_text_1"].astype(str).tolist()
        right = sample["product_text_2"].astype(str).tolist()
        probe_left = probe["product_text_1"].astype(str).tolist()
        probe_right = probe["product_text_2"].astype(str).tolist()

        torch, tokenizer, model, load_seconds, precision = load_native_model(model_dir)
        torch.backends.cuda.matmul.allow_tf32 = True
        baseline_scores, baseline_row = benchmark_call(
            model_name=model_name, backend="transformers_native_baseline", phase="sample",
            batch_size=profile["baseline_batch"], frame=sample,
            load_seconds=load_seconds, precision=precision,
            call=lambda: predict_onthefly(
                torch, tokenizer, model, left, right, profile["baseline_batch"], precision
            ), rows=rows,
        )
        if baseline_scores is None:
            raise RuntimeError(f"Frozen baseline failed for {model_name}")

        viable_batches = [(profile["baseline_batch"], baseline_row["pairs_per_second"])]
        for batch in profile["batch_candidates"]:
            _, row = benchmark_call(
                model_name=model_name, backend="transformers_native_batch_sweep", phase="probe",
                batch_size=batch, frame=probe, load_seconds=load_seconds, precision=precision,
                call=lambda current=batch: predict_onthefly(
                    torch, tokenizer, model, probe_left, probe_right, current, precision
                ), rows=rows,
            )
            if row["status"] == "ok" and row.get("pearson", 0.0) >= 0.999:
                viable_batches.append((batch, row["pairs_per_second"]))
            if row["status"] == "oom":
                break
        best_batch = max(viable_batches, key=lambda item: item[1])[0]
        optimized_scores, optimized_row = benchmark_call(
            model_name=model_name, backend="transformers_native_large_batch", phase="sample",
            batch_size=best_batch, frame=sample, load_seconds=load_seconds, precision=precision,
            call=lambda: predict_onthefly(
                torch, tokenizer, model, left, right, best_batch, precision
            ), rows=rows,
        )
        pretokenized_scores, pretokenized_row = benchmark_call(
            model_name=model_name, backend="transformers_native_pretokenized_bucketed",
            phase="sample", batch_size=best_batch, frame=sample,
            load_seconds=load_seconds, precision=precision,
            call=lambda: predict_pretokenized(
                torch, tokenizer, model, left, right, best_batch, precision
            ), rows=rows,
        )
        candidates = [
            ("transformers_native_baseline", profile["baseline_batch"], baseline_scores, baseline_row),
            ("transformers_native_large_batch", best_batch, optimized_scores, optimized_row),
            (
                "transformers_native_pretokenized_bucketed", best_batch,
                pretokenized_scores, pretokenized_row,
            ),
        ]
        eligible = [
            candidate for candidate in candidates
            if candidate[2] is not None
            and candidate[3].get("pearson", 0.0) >= 0.999
            and abs(candidate[3].get("ap_delta", 1.0)) <= 0.002
        ]
        selected_backend, selected_batch, _, selected_row = min(
            eligible, key=lambda candidate: candidate[3]["total_seconds"]
        )
        selected_predictor = (
            predict_pretokenized
            if selected_backend == "transformers_native_pretokenized_bucketed"
            else predict_onthefly
        )

        if not args.skip_compile and hasattr(torch, "compile"):
            try:
                compile_started = time.perf_counter()
                compiled = torch.compile(model, mode="reduce-overhead", dynamic=True)
                _, compile_row = benchmark_call(
                    model_name=model_name, backend="transformers_torch_compile", phase="probe",
                    batch_size=selected_batch, frame=probe, load_seconds=load_seconds,
                    precision=precision,
                    call=lambda: selected_predictor(
                        torch, tokenizer, compiled, probe_left, probe_right, selected_batch, precision
                    ), rows=rows,
                )
                compile_row["compile_plus_probe_seconds"] = time.perf_counter() - compile_started
                del compiled
            except Exception as error:
                rows.append({
                    "model": model_name, "backend": "transformers_torch_compile",
                    "phase": "compatibility", "status": "error", "pairs": len(probe),
                    "batch": selected_batch, "max_length": MAX_LENGTH,
                    "error": f"{type(error).__name__}: {error}"[:500],
                })

        full_metrics: dict[str, Any] = {}
        if not args.quick:
            for split in SPLITS:
                full = canonical[split].copy()
                full["split"] = split
                full = reference_for_model(args.input_root, manifest, model_name, full, split=split)
                full_left = full["product_text_1"].astype(str).tolist()
                full_right = full["product_text_2"].astype(str).tolist()
                full_scores, row = benchmark_call(
                    model_name=model_name, backend=selected_backend, phase=f"full_{split}",
                    batch_size=selected_batch, frame=full, load_seconds=load_seconds,
                    precision=precision,
                    call=lambda a=full_left, b=full_right: selected_predictor(
                        torch, tokenizer, model, a, b, selected_batch, precision
                    ), rows=rows,
                )
                if full_scores is None:
                    raise RuntimeError(f"Selected runner failed on {model_name}/{split}")
                prediction_path = args.output_dir / f"{model_name}_{selected_backend}_{split}.parquet"
                pd.DataFrame({
                    "id1": full["id1"].to_numpy(), "id2": full["id2"].to_numpy(),
                    "target": full["target"].to_numpy(), "score": full_scores,
                    "reference_score": full["reference_score"].to_numpy(),
                }).to_parquet(prediction_path, index=False, compression="zstd")
                full_metrics[split] = {
                    "macro_ap": row["macro_ap"], "reference_macro_ap": row["reference_macro_ap"],
                    "ap_delta": row["ap_delta"], "pearson": row["pearson"],
                    "mean_abs_difference": row["mean_abs_difference"],
                    "max_abs_difference": row["max_abs_difference"],
                    "total_seconds": row["total_seconds"], "prediction_path": prediction_path.name,
                }

        winners[model_name] = {
            "backend": selected_backend,
            "batch": selected_batch,
            "precision": precision,
            "load_seconds": load_seconds,
            "sample_total_seconds": next(
                row["total_seconds"] for row in reversed(rows)
                if row["model"] == model_name and row["backend"] == selected_backend and row["phase"] == "sample"
            ),
            "sample_pairs": len(sample),
            "full_validation": full_metrics,
        }
        del model, tokenizer
        release_cuda(torch)

        if not args.quick:
            flash_attention_probe(model_name, model_dir, probe, selected_batch, rows)
            sentence_transformers_probe(
                model_name, model_dir, probe, selected_batch, rows
            )
        vllm_compatibility(model_name, model_dir, rows)

    results = pd.DataFrame(rows)
    results_path = args.output_dir / "inference_benchmark_results.csv"
    results.to_csv(results_path, index=False)

    estimates: dict[str, Any] = {}
    for model_name, winner in winners.items():
        per_pair = winner["sample_total_seconds"] / winner["sample_pairs"]
        estimates[model_name] = {
            str(pairs): winner["load_seconds"] + preprocessing["data_read_seconds"] + (
                preprocessing["serialization_seconds"] / len(base_sample) + per_pair
            ) * pairs
            for pairs in (115_000, 275_000)
        }
    summary = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "hardware": {
            "note": "Kaggle timing is for one visible benchmark GPU; final H100 limits require Docker verification",
        },
        "sample_size": len(base_sample),
        "quick_mode": args.quick,
        "preprocessing": preprocessing,
        "winners": winners,
        "rough_t4_extrapolation_seconds": estimates,
        "competition_limit_verdict": "NOT_VALID_ON_T4; run selected runners in competition H100 Docker",
        "results_csv": results_path.name,
    }
    summary_path = args.output_dir / "inference_benchmark_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
