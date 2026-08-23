#!/usr/bin/env python3
"""Train one side of the controlled S2 augmentation comparison on one GPU."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cross_encoder_training import (
    CrossEncoderBatchCollator,
    CrossEncoderPairDataset,
    build_pair_token_cache,
)
from src.minilm_s2_augmentation import RUNS, group_metrics
from src.qwen_reranker import preferred_cuda_dtype
from src.qwen_training import FixedLengthBatchSampler, LengthBucketBatchSampler
from train_serialization_ablation import attach_text, evaluate, loader_options


class ForwardLengthBucketBatchSampler(Sampler[list[tuple[int, bool]]]):
    """Shuffle examples and bucket by length without ever reversing a pair."""

    def __init__(
        self,
        forward_lengths: Sequence[int],
        batch_size: int,
        *,
        bucket_size_multiplier: int = 50,
        seed: int = 42,
    ) -> None:
        self.forward_lengths = np.asarray(forward_lengths)
        self.batch_size = batch_size
        self.bucket_size = max(batch_size, batch_size * bucket_size_multiplier)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return math.ceil(len(self.forward_lengths) / self.batch_size)

    def __iter__(self) -> Iterator[list[tuple[int, bool]]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        indices = rng.permutation(len(self.forward_lengths)).tolist()
        ordered: list[int] = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=lambda index: self.forward_lengths[index])
            ordered.extend(bucket)
        batches = [
            [(index, False) for index in ordered[start : start + self.batch_size]]
            for start in range(0, len(ordered), self.batch_size)
        ]
        rng.shuffle(batches)
        yield from batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--run-name", choices=RUNS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--token-cache-dir", type=Path, required=True)
    return parser.parse_args()


def augmented_train_frame(prepared_dir: Path, pairs: pd.DataFrame) -> pd.DataFrame:
    train = pd.read_parquet(prepared_dir / "train_augmented_texts_epoch0.parquet")
    if not np.array_equal(train[["id1", "id2"]].to_numpy(), pairs[["id1", "id2"]].to_numpy()):
        raise RuntimeError("Augmented texts do not preserve train-pair order")
    train["target"] = pairs["target"].to_numpy(dtype=np.float32)
    return train


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_config = config["runs"][args.run_name]
    if int(config["epochs"]) != 1:
        raise ValueError("This controlled screening experiment is restricted to one epoch")
    if int(config["batch_size"]) * int(config["gradient_accumulation"]) != 64:
        raise ValueError("Single-GPU effective batch size must remain 64")
    if bool(config["symmetric_validation"]):
        raise ValueError("Validation must use one deterministic A-to-B forward")

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

    items = pd.read_parquet(
        args.prepared_dir / "items.parquet",
        columns=["id", "category", "text_s2_values_only"],
    )
    train_pairs = pd.read_parquet(args.prepared_dir / "train_pairs.parquet")
    validation_pairs = pd.read_parquet(args.prepared_dir / "validation_pairs.parquet")
    if bool(run_config["attribute_shuffle"]):
        train = augmented_train_frame(args.prepared_dir, train_pairs)
    else:
        train = attach_text(train_pairs, items, "text_s2_values_only")
    validation = attach_text(validation_pairs, items, "text_s2_values_only")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id")
    cache_stem = args.run_name.lower()
    train_cache = build_pair_token_cache(
        train,
        tokenizer,
        args.token_cache_dir,
        f"train-{cache_stem}",
        str(args.model_path),
        int(config["max_length"]),
        int(config["tokenization_batch_size"]),
        int(config["tokenization_log_every"]),
    )
    validation_cache = build_pair_token_cache(
        validation,
        tokenizer,
        args.token_cache_dir,
        f"validation-{cache_stem}",
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
    sampler_kwargs = {
        "batch_size": int(config["batch_size"]),
        "bucket_size_multiplier": int(config["bucket_size_multiplier"]),
        "seed": seed,
    }
    if bool(run_config["random_pair_swap"]):
        train_sampler: Any = LengthBucketBatchSampler(
            train_cache.forward_lengths,
            train_cache.reverse_lengths,
            weights=None,
            **sampler_kwargs,
        )
    else:
        train_sampler = ForwardLengthBucketBatchSampler(
            train_cache.forward_lengths,
            **sampler_kwargs,
        )
    validation_sampler = FixedLengthBatchSampler(
        validation_cache,
        np.arange(len(validation_dataset)),
        int(config["eval_batch_size"]),
        both_orientations=False,
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
        target = decay if parameter.ndim >= 2 and "layer_norm" not in name.lower() else no_decay
        target.append(parameter)
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
    processed = reversed_pairs = useful_tokens = padded_tokens = 0
    interval_started = training_started
    interval_loss = 0.0
    interval_steps = 0
    train_sampler.set_epoch(0)
    model.train()
    for step, packed in enumerate(train_loader):
        packed.pop("pair_indices")
        orientations = packed.pop("orientations")
        reversed_pairs += int(orientations.sum())
        targets = packed.pop("targets").to(device, non_blocking=True)
        packed.pop("sample_weights")
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
                        "run": args.run_name,
                        "step": step + 1,
                        "steps": len(train_loader),
                        "loss": interval_loss / interval_steps,
                        "examples_seen": processed,
                        "random_swap_rate": reversed_pairs / processed,
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
    hard_metrics = group_metrics(
        predictions,
        pd.read_parquet(args.prepared_dir / "validation_groups.parquet"),
    )
    model.save_pretrained(args.checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.checkpoint_dir)
    predictions.to_parquet(args.output_dir / "validation_predictions.parquet", index=False)
    report = {
        "experiment": f"{config['experiment']}_{args.run_name.lower()}",
        "run_name": args.run_name,
        "serialization": "S2_VALUES_ONLY",
        "model": str(config["model"]),
        "model_revision": args.model_revision,
        "human_labels_only": True,
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
        "training_reversed_pairs": reversed_pairs,
        "training_pair_swap_rate": float(reversed_pairs / processed),
        "inference_pair_orders": 1,
        "hard_group_metrics": hard_metrics,
        **metrics,
        "args": {
            **config,
            "run_name": args.run_name,
            "attribute_shuffle": bool(run_config["attribute_shuffle"]),
            "random_pair_swap": bool(run_config["random_pair_swap"]),
            "model_revision": args.model_revision,
            "effective_batch_size": int(config["batch_size"]) * accumulation,
            "validation_pair_order": "A_TO_B_ONLY",
        },
    }
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "COMPLETED").write_text("complete\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
