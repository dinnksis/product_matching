#!/usr/bin/env python3
"""Train and evaluate one MiniLM serialization variant on one GPU."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cross_encoder_training import (
    CrossEncoderBatchCollator,
    CrossEncoderPairDataset,
    PairTokenCache,
    build_pair_token_cache,
)
from src.qwen_reranker import preferred_cuda_dtype
from src.qwen_training import FixedLengthBatchSampler, LengthBucketBatchSampler
from src.serialization_ablation import VARIANTS


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--token-cache-dir", type=Path, required=True)
    return parser.parse_args()


def attach_text(pairs: pd.DataFrame, items: pd.DataFrame, text_column: str) -> pd.DataFrame:
    lookup = items.set_index("id", verify_integrity=True)[[text_column, "category"]]
    left = lookup.reindex(pairs["id1"].to_numpy()).add_suffix("_1")
    right = lookup.reindex(pairs["id2"].to_numpy()).add_suffix("_2")
    left.index = pairs.index
    right.index = pairs.index
    result = pd.concat([pairs.reset_index(drop=True), left.reset_index(drop=True), right.reset_index(drop=True)], axis=1)
    result = result.rename(
        columns={f"{text_column}_1": "product_text_1", f"{text_column}_2": "product_text_2"}
    )
    required = ["product_text_1", "product_text_2", "category_1", "category_2"]
    if result[required].isna().any().any():
        raise ValueError("Some pair ids are missing from prepared items")
    if (result["category_1"] != result["category_2"]).any():
        raise ValueError("Cross-category pairs are not supported")
    return result


def loader_options(config: dict[str, Any], *, persistent: bool) -> dict[str, Any]:
    workers = int(config["dataloader_workers"])
    options: dict[str, Any] = {"num_workers": workers, "pin_memory": True}
    if workers:
        options.update(
            persistent_workers=persistent,
            prefetch_factor=int(config["prefetch_factor"]),
        )
    return options


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    validation: pd.DataFrame,
    cache: PairTokenCache,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_length: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    score_ab = np.full(len(validation), np.nan, dtype=np.float32)
    score_ba = np.full(len(validation), np.nan, dtype=np.float32)
    iterator = iter(loader)
    try:
        warmup = next(iterator)
    except StopIteration as error:
        raise ValueError("Validation loader is empty") from error
    warmup_batch = {
        key: value.to(device, non_blocking=True)
        for key, value in warmup.items()
        if key not in {"pair_indices", "orientations", "targets", "sample_weights"}
    }
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        model(**warmup_batch)
    torch.cuda.synchronize(device)
    del iterator, warmup, warmup_batch
    started = time.perf_counter()
    with torch.inference_mode():
        for packed in loader:
            pair_indices = packed.pop("pair_indices").numpy()
            orientations = packed.pop("orientations").numpy()
            packed.pop("targets")
            packed.pop("sample_weights")
            batch = {key: value.to(device, non_blocking=True) for key, value in packed.items()}
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                probabilities = model(**batch).logits.squeeze(-1).float().sigmoid()
            scores = probabilities.cpu().numpy()
            forward_mask = ~orientations
            score_ab[pair_indices[forward_mask]] = scores[forward_mask]
            score_ba[pair_indices[orientations]] = scores[orientations]
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - started
    symmetric = np.isfinite(score_ba).any()
    if not np.isfinite(score_ab).all() or (symmetric and not np.isfinite(score_ba).all()):
        raise RuntimeError("Validation produced missing or non-finite predictions")
    score = (score_ab + score_ba) / 2 if symmetric else score_ab
    target = validation["target"].to_numpy(dtype=np.float32)
    categories = validation["category_1"].astype(str).to_numpy()
    metric_frame = pd.DataFrame({"target": target, "score": score, "category": categories})
    per_category_ap = {
        str(category): float(average_precision_score(part["target"], part["score"]))
        for category, part in metric_frame.groupby("category", sort=True)
    }
    per_category_roc = {
        str(category): float(roc_auc_score(part["target"], part["score"]))
        for category, part in metric_frame.groupby("category", sort=True)
    }
    lengths = np.concatenate([cache.forward_lengths, cache.reverse_lengths] if symmetric else [cache.forward_lengths])
    metrics = {
        "validation_seconds": inference_seconds,
        "validation_pairs_per_second": float(len(validation) / inference_seconds),
        "validation_forward_sequences_per_second": float(len(validation) * (2 if symmetric else 1) / inference_seconds),
        "macro_average_precision": float(np.mean(list(per_category_ap.values()))),
        "overall_average_precision": float(average_precision_score(target, score)),
        "macro_roc_auc": float(np.mean(list(per_category_roc.values()))),
        "overall_roc_auc": float(roc_auc_score(target, score)),
        "per_category_average_precision": per_category_ap,
        "per_category_roc_auc": per_category_roc,
        "avg_tokens": float(np.mean(lengths)),
        "p95_tokens": float(np.quantile(lengths, 0.95)),
        "max_tokens": int(np.max(lengths)),
        "max_length_fraction": float(np.mean(lengths >= max_length)),
        "mean_score_order_gap": float(np.mean(np.abs(score_ab - score_ba))) if symmetric else 0.0,
    }
    predictions = validation[["id1", "id2", "target", "category_1"]].rename(columns={"category_1": "category"}).copy()
    predictions["score_ab"] = score_ab
    predictions["score_ba"] = score_ba if symmetric else np.nan
    predictions["score"] = score
    predictions["score_order_gap"] = np.abs(score_ab - score_ba) if symmetric else 0.0
    predictions["tokens_ab"] = cache.forward_lengths
    predictions["tokens_ba"] = cache.reverse_lengths
    return metrics, predictions


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if int(config["epochs"]) != 1:
        raise ValueError("Serialization screening is intentionally restricted to one epoch")
    if int(config["batch_size"]) * int(config["gradient_accumulation"]) != 64:
        raise ValueError("Configured single-GPU effective batch must remain 64 for this ablation")
    pipeline_started = time.perf_counter()
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    text_column = f"text_{args.variant.lower()}"
    items = pd.read_parquet(args.prepared_dir / "items.parquet", columns=["id", "category", text_column])
    train_pairs = pd.read_parquet(args.prepared_dir / "train_pairs.parquet")
    validation_pairs = pd.read_parquet(args.prepared_dir / "validation_pairs.parquet")
    train = attach_text(train_pairs, items, text_column)
    validation = attach_text(validation_pairs, items, text_column)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id")
    train_cache = build_pair_token_cache(
        train,
        tokenizer,
        args.token_cache_dir,
        f"train-{args.variant.lower()}",
        str(args.model_path),
        int(config["max_length"]),
        int(config["tokenization_batch_size"]),
        int(config["tokenization_log_every"]),
    )
    validation_cache = build_pair_token_cache(
        validation,
        tokenizer,
        args.token_cache_dir,
        f"validation-{args.variant.lower()}",
        str(args.model_path),
        int(config["max_length"]),
        int(config["tokenization_batch_size"]),
        int(config["tokenization_log_every"]),
    )
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train_targets = train["target"].to_numpy(dtype=np.float32)
    validation_targets = validation["target"].to_numpy(dtype=np.float32)
    train_dataset = CrossEncoderPairDataset(train_cache, train_targets)
    validation_dataset = CrossEncoderPairDataset(validation_cache, validation_targets)
    train_sampler = LengthBucketBatchSampler(
        train_cache.forward_lengths,
        train_cache.reverse_lengths,
        batch_size=int(config["batch_size"]),
        weights=None,
        bucket_size_multiplier=int(config["bucket_size_multiplier"]),
        seed=seed,
    )
    validation_sampler = FixedLengthBatchSampler(
        validation_cache,
        np.arange(len(validation_dataset)),
        int(config["eval_batch_size"]),
        both_orientations=bool(config["symmetric_validation"]),
    )
    collator = CrossEncoderBatchCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collator,
        **loader_options(config, persistent=True),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=validation_sampler,
        collate_fn=collator,
        **loader_options(config, persistent=False),
    )
    amp_dtype = preferred_cuda_dtype()
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=1,
        attn_implementation=str(config["attention_implementation"]),
        local_files_only=True,
    ).to(device)
    if bool(config["gradient_checkpointing"]):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim >= 2 and "layer_norm" not in name.lower() else no_decay).append(parameter)
    optimizer = AdamW(
        [
            {"params": decay, "weight_decay": float(config["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(config["learning_rate"]),
    )
    accumulation = int(config["gradient_accumulation"])
    total_updates = math.ceil(len(train_loader) / accumulation)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        max(1, int(total_updates * float(config["warmup_ratio"]))),
        total_updates,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    training_started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    processed = useful_tokens = padded_tokens = 0
    interval_started = training_started
    interval_loss = 0.0
    interval_steps = 0
    train_sampler.set_epoch(0)
    model.train()
    for step, packed in enumerate(train_loader):
        packed.pop("pair_indices")
        packed.pop("orientations")
        targets = packed.pop("targets").to(device, non_blocking=True)
        packed.pop("sample_weights")
        if float(config["label_smoothing"]):
            smoothing = float(config["label_smoothing"])
            targets = targets * (1 - smoothing) + 0.5 * smoothing
        batch = {key: value.to(device, non_blocking=True) for key, value in packed.items()}
        useful_tokens += int(batch["attention_mask"].sum())
        padded_tokens += batch["attention_mask"].numel()
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(**batch).logits.squeeze(-1)
            raw_loss = F.binary_cross_entropy_with_logits(logits.float(), targets)
            loss = raw_loss / accumulation
        scaler.scale(loss).backward()
        should_update = (step + 1) % accumulation == 0 or step + 1 == len(train_loader)
        if should_update:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, float(config["max_grad_norm"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        processed += len(targets)
        interval_loss += float(raw_loss.detach())
        interval_steps += 1
        if (step + 1) % int(config["log_every"]) == 0 or step + 1 == len(train_loader):
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - interval_started
            print(
                json.dumps(
                    {
                        "variant": args.variant,
                        "step": step + 1,
                        "steps": len(train_loader),
                        "loss": interval_loss / interval_steps,
                        "examples_seen": processed,
                        "examples_per_second_interval": interval_steps * int(config["batch_size"]) / elapsed,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    }
                ),
                flush=True,
            )
            interval_started = time.perf_counter()
            interval_loss = 0.0
            interval_steps = 0
    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_started
    metrics, predictions = evaluate(
        model,
        validation_loader,
        validation,
        validation_cache,
        device,
        amp_dtype,
        int(config["max_length"]),
    )
    model.save_pretrained(args.checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.checkpoint_dir)
    predictions.to_parquet(args.output_dir / "validation_predictions.parquet", index=False)
    report = {
        "experiment": str(config["experiment"]),
        "serialization": args.variant,
        "model": str(config["model"]),
        "model_revision": args.model_revision,
        "training_seconds": training_seconds,
        "training_wall_seconds": training_seconds,
        "total_pipeline_seconds": time.perf_counter() - pipeline_started,
        "training_examples": len(train_dataset),
        "original_training_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "validation_positive_examples": int(validation_targets.sum()),
        "validation_positive_rate": float(validation_targets.mean()),
        "examples_per_second": float(len(train_dataset) / training_seconds),
        "padding_efficiency": float(useful_tokens / padded_tokens),
        "peak_vram_gib_by_rank": [float(torch.cuda.max_memory_allocated(device) / 2**30)],
        **metrics,
        "args": {
            **config,
            "variant": args.variant,
            "model_revision": args.model_revision,
            "effective_batch_size": int(config["batch_size"]) * accumulation,
        },
    }
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "COMPLETED").write_text("complete\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
