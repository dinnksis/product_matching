#!/usr/bin/env python3
"""Training-only BGE export path for the frozen 365,654-pair human corpus.

This deliberately does not evaluate a validation split.  It reuses the shared
cross-encoder data, batching, loss-hook and optimizer contracts, plus the
memory-safe BGE AdamW/gradient-clipping guards, but writes only a deployable
model/tokenizer and a metric-free training report.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_bge_2ep_sft as bge
import train_cross_encoder as shared


EXPECTED_TRAIN_ROWS = 365_654
EXPECTED_TRAIN_POSITIVES = 93_890
EXPECTED_SOURCE_COUNTS = {
    "human_train": 306_669,
    "human_iid": 12_000,
    "human_hard": 5_814,
    "human_former_ood": 41_171,
}
EXPECTED_EPOCHS = 2
EXPECTED_STEPS_PER_EPOCH_PER_RANK = 22_854
EXPECTED_UPDATES_PER_EPOCH = 1_905
EXPECTED_TOTAL_UPDATES = 3_810
EXPECTED_WARMUP_UPDATES = 190
MAX_FULLTRAIN_OVERFLOW_SKIPS = 16
EXPECTED_RECIPE_SHA256 = (
    "d46be9217a43396ecc8c594fc1864ee93761d288c30e5a40041adbb28bd7adfe"
)
EXPECTED_LOSS_HOOK_SHA256 = (
    "f7a89be9b96d3d492322b3a0c4a8180cfaa0987e336c775bc80e445ad63a9ade"
)
EXPECTED_SPECIAL_TOKENS_MAP_SHA256 = (
    "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835"
)
EXPECTED_SPECIAL_TOKENS_MAP_BYTES = 964


class FinalFulltrainContractError(ValueError):
    """Raised before export if the selected final recipe has drifted."""


def accumulation_group_denominator(
    batch_sizes: list[int] | tuple[int, ...],
    *,
    step: int,
    gradient_accumulation: int,
) -> int:
    """Return the exact local-example denominator for one optimizer group."""
    if gradient_accumulation <= 0 or not 0 <= step < len(batch_sizes):
        raise FinalFulltrainContractError("Invalid accumulation geometry")
    group_start = (step // gradient_accumulation) * gradient_accumulation
    group_end = min(group_start + gradient_accumulation, len(batch_sizes))
    denominator = sum(batch_sizes[group_start:group_end])
    if denominator <= 0:
        raise FinalFulltrainContractError("Empty accumulation group")
    return denominator


def validate_final_contract(args: Any, *, world_size: int) -> None:
    expected = {
        "model_backend": "sequence_classification",
        "trust_remote_code": False,
        "epochs": EXPECTED_EPOCHS,
        "batch_size": 8,
        "eval_batch_size": 32,
        "gradient_accumulation": 12,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "max_length": 384,
        "attention_implementation": "sdpa",
        "sampling": "none",
        "train_subset": "all",
        "loss_weighting": "none",
        "lexical_hard_negative_strength": 0.0,
        "gradient_checkpointing": True,
        "bucket_size_multiplier": 50,
        "dataloader_workers": 2,
        "prefetch_factor": 2,
        "tokenization_batch_size": 1024,
        "tokenization_log_every": 20,
        "symmetric_validation": True,
        "label_smoothing": 0.0,
        "max_grad_norm": 0.5,
        "log_every": 50,
        "seed": 42,
    }
    mismatches = {
        key: {"actual": getattr(args, key), "expected": value}
        for key, value in expected.items()
        if getattr(args, key) != value
    }
    if world_size != bge.EXPECTED_WORLD_SIZE:
        mismatches["world_size"] = {
            "actual": world_size,
            "expected": bge.EXPECTED_WORLD_SIZE,
        }
    if args.validation_split:
        mismatches["validation_split"] = {
            "actual": list(args.validation_split),
            "expected": [],
        }
    if args.model_load_kwarg:
        mismatches["model_load_kwarg"] = {
            "actual": list(args.model_load_kwarg),
            "expected": [],
        }
    runtime_config = json.loads(args.config.read_text(encoding="utf-8"))
    normalized_config = dict(runtime_config)
    normalized_config["model"] = "model/pretrain_bge_2ep"
    canonical_payload = json.dumps(
        normalized_config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    observed_recipe_sha256 = hashlib.sha256(canonical_payload).hexdigest()
    if observed_recipe_sha256 != EXPECTED_RECIPE_SHA256:
        mismatches["canonical_recipe_sha256"] = {
            "actual": observed_recipe_sha256,
            "expected": EXPECTED_RECIPE_SHA256,
        }
    if mismatches:
        raise FinalFulltrainContractError(
            "Final BGE full-train recipe differs: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def train_and_export() -> None:
    args = shared.parse_args()
    shared.validate_args(args)
    pipeline_started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not distributed:
        raise FinalFulltrainContractError("Final BGE export requires DDP")
    torch.cuda.set_device(local_rank)
    process_group_timeout = timedelta(hours=1)
    dist.init_process_group(backend="nccl", timeout=process_group_timeout)
    control_group = dist.new_group(backend="gloo", timeout=process_group_timeout)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    is_main = rank == 0
    validate_final_contract(args, world_size=world_size)
    bge.validate_memory_geometry(json.loads(args.config.read_text(encoding="utf-8")))

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    items = pd.read_parquet(
        args.prepared_dir / "items.parquet",
        columns=["id", "product_text", "category"],
    )
    train_pairs = pd.read_parquet(args.prepared_dir / "train_pairs.parquet")
    train = shared.attach_item_fields(
        train_pairs, items, fields=("product_text", "category")
    )
    if len(train) != EXPECTED_TRAIN_ROWS:
        raise FinalFulltrainContractError("Final BGE train row count changed")
    if int(train["target"].sum()) != EXPECTED_TRAIN_POSITIVES:
        raise FinalFulltrainContractError("Final BGE positive count changed")
    if (train["category_1"] != train["category_2"]).any():
        raise FinalFulltrainContractError("Final BGE train has cross-category pairs")
    if not train["target"].isin([0.0, 1.0]).all():
        raise FinalFulltrainContractError("Final BGE targets are not binary")
    source_counts = {
        str(key): int(value)
        for key, value in train["label_source"].value_counts().items()
    }
    if source_counts != EXPECTED_SOURCE_COUNTS:
        raise FinalFulltrainContractError("Final BGE label_source counts changed")
    train = train.reset_index(drop=True)

    tokenizer = shared.AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        raise FinalFulltrainContractError("Final BGE tokenizer has no pad token")
    train_cache, validation_caches = shared.create_caches(
        train,
        {},
        tokenizer,
        args,
        is_main,
        distributed,
        control_group,
    )
    if validation_caches:
        raise FinalFulltrainContractError("Final export created validation caches")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    train_targets = train["target"].to_numpy(dtype=np.float32)
    train_categories = train["category_1"].astype(str).tolist()
    training_loss_weights = shared.build_training_loss_weights(
        train_categories,
        train_targets,
        mode=args.loss_weighting,
        lexical_similarities=None,
        lexical_hard_negative_strength=args.lexical_hard_negative_strength,
    )
    if "sample_weight" in train:
        training_loss_weights *= train["sample_weight"].to_numpy(dtype=np.float32)
    if (
        not np.isfinite(training_loss_weights).all()
        or (training_loss_weights <= 0).any()
    ):
        raise FinalFulltrainContractError("Final BGE loss weights are invalid")
    training_loss_weights /= training_loss_weights.mean()
    if not np.array_equal(
        training_loss_weights, np.ones(EXPECTED_TRAIN_ROWS, dtype=np.float32)
    ):
        raise FinalFulltrainContractError("Plain BCE unexpectedly changed sample weights")
    training_source_weight_mass = {
        str(source): float(training_loss_weights[positions].sum())
        for source, positions in train.groupby("label_source").indices.items()
    }
    expected_mass = {key: float(value) for key, value in EXPECTED_SOURCE_COUNTS.items()}
    if training_source_weight_mass != expected_mass:
        raise FinalFulltrainContractError("Final BGE source weight mass changed")

    train_dataset = shared.CrossEncoderPairDataset(
        train_cache,
        train_targets,
        sample_weights=training_loss_weights,
    )
    train_sampler = shared.LengthBucketBatchSampler(
        train_cache.forward_lengths,
        train_cache.reverse_lengths,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
        weights=None,
        bucket_size_multiplier=args.bucket_size_multiplier,
        seed=args.seed,
    )
    if len(train_sampler) != EXPECTED_STEPS_PER_EPOCH_PER_RANK:
        raise FinalFulltrainContractError("Final BGE microstep geometry changed")
    collator = shared.CrossEncoderBatchCollator(tokenizer.pad_token_id)
    train_loader = shared.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collator,
        **shared.loader_options(args, persistent=True),
    )
    loss_hook = shared.load_loss_hook(args.loss_hook)
    if loss_hook.metadata.get("sha256") != EXPECTED_LOSS_HOOK_SHA256:
        raise FinalFulltrainContractError("Final BGE loss-hook identity changed")
    loss_hook.initialize(
        train_frame=train,
        device=device,
        rank=rank,
        world_size=world_size,
    )
    del train, items, train_pairs

    amp_dtype = shared.preferred_cuda_dtype()
    if amp_dtype != torch.float16:
        raise FinalFulltrainContractError("Final BGE export requires FP16 on T4")
    model = shared.AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=1,
        attn_implementation=args.attention_implementation,
        trust_remote_code=False,
        **shared.model_load_kwargs(args),
    )
    model.config.id2label = {0: "MATCH_SCORE"}
    model.config.label2id = {"MATCH_SCORE": 0}
    model = model.to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != bge.EXPECTED_PARAMETERS:
        raise FinalFulltrainContractError("Final BGE parameter count changed")
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    training_model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )
    named_parameters = [
        (name, parameter)
        for name, parameter in training_model.named_parameters()
        if parameter.requires_grad
    ]
    decay_parameters = [
        parameter
        for name, parameter in named_parameters
        if parameter.ndim >= 2 and "layer_norm" not in name.lower()
    ]
    no_decay_parameters = [
        parameter
        for name, parameter in named_parameters
        if parameter.ndim < 2 or "layer_norm" in name.lower()
    ]
    optimizer = shared.AdamW(
        [
            {"params": decay_parameters, "weight_decay": args.weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_updates = updates_per_epoch * args.epochs
    warmup_updates = max(1, int(total_updates * args.warmup_ratio))
    if (
        updates_per_epoch != EXPECTED_UPDATES_PER_EPOCH
        or total_updates != EXPECTED_TOTAL_UPDATES
        or warmup_updates != EXPECTED_WARMUP_UPDATES
    ):
        raise FinalFulltrainContractError("Final BGE optimizer schedule changed")
    scheduler = shared.get_cosine_schedule_with_warmup(
        optimizer, warmup_updates, total_updates
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    trainable_parameters = [parameter for _, parameter in named_parameters]

    if is_main:
        print(
            json.dumps(
                {
                    "purpose": "final_deployment_export",
                    "quality_evaluation": False,
                    "gpu": torch.cuda.get_device_name(device),
                    "world_size": world_size,
                    "model": args.model,
                    "amp_dtype": "float16",
                    "trainable_parameters": sum(
                        parameter.numel() for parameter in trainable_parameters
                    ),
                    "train_pairs": len(train_dataset),
                    "epochs": args.epochs,
                    "steps_per_epoch_per_rank": len(train_loader),
                    "updates_per_epoch": updates_per_epoch,
                    "total_updates": total_updates,
                    "warmup_updates": warmup_updates,
                    "effective_batch": (
                        args.batch_size * world_size * args.gradient_accumulation
                    ),
                    "validation_splits": [],
                    "source_counts": source_counts,
                    "loss_hook": loss_hook.metadata,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    training_started = time.perf_counter()
    local_examples = local_useful_tokens = local_padded_tokens = 0
    optimizer_step_attempts = optimizer_steps_succeeded = overflow_skips = 0
    epoch_batch_geometry: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        emitted_batch_sizes = tuple(len(batch) for batch in train_sampler)
        if (
            len(emitted_batch_sizes) != EXPECTED_STEPS_PER_EPOCH_PER_RANK
            or sum(emitted_batch_sizes) != EXPECTED_TRAIN_ROWS // world_size
            or emitted_batch_sizes.count(3) != 1
            or any(size not in {3, args.batch_size} for size in emitted_batch_sizes)
        ):
            raise FinalFulltrainContractError(
                "Final BGE emitted batch-size sequence changed"
            )
        batch_size_digest = hashlib.sha256(
            json.dumps(emitted_batch_sizes, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        partial_batch_step = emitted_batch_sizes.index(3)
        partial_group_denominator = accumulation_group_denominator(
            emitted_batch_sizes,
            step=partial_batch_step,
            gradient_accumulation=args.gradient_accumulation,
        )
        local_geometry = torch.tensor(
            [partial_batch_step, partial_group_denominator],
            dtype=torch.int64,
            device=device,
        )
        gathered_geometry = [
            torch.zeros_like(local_geometry) for _ in range(world_size)
        ]
        dist.all_gather(gathered_geometry, local_geometry)
        if any(
            not torch.equal(gathered_geometry[0], rank_geometry)
            for rank_geometry in gathered_geometry[1:]
        ):
            raise FinalFulltrainContractError(
                "Final BGE accumulation geometry differs across ranks"
            )
        epoch_batch_geometry.append(
            {
                "epoch": epoch + 1,
                "batch_sizes_sha256": batch_size_digest,
                "partial_batch_step_zero_based": partial_batch_step,
                "partial_batch_size": 3,
                "partial_group_examples_per_rank": partial_group_denominator,
                "ranks_geometry_equal": True,
            }
        )
        training_model.train()
        interval_started = time.perf_counter()
        interval_loss = torch.zeros((), dtype=torch.float32, device=device)
        interval_steps = interval_examples = 0
        interval_useful_tokens = interval_padded_tokens = 0
        previous_step_finished = interval_started
        interval_data_seconds = 0.0

        for step, packed in enumerate(train_loader):
            batch_ready = time.perf_counter()
            interval_data_seconds += batch_ready - previous_step_finished
            pair_indices = packed.pop("pair_indices").to(device, non_blocking=True)
            orientations = packed.pop("orientations").to(device, non_blocking=True)
            cpu_targets = packed.pop("targets")
            batch_examples = len(cpu_targets)
            useful_tokens = int(packed["attention_mask"].sum())
            padded_tokens = packed["attention_mask"].numel()
            targets = cpu_targets.to(device, non_blocking=True)
            weights = packed.pop("sample_weights").to(device, non_blocking=True)
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in packed.items()
            }
            expected_batch_examples = emitted_batch_sizes[step]
            if batch_examples != expected_batch_examples:
                raise FinalFulltrainContractError(
                    "Observed batch differs from deterministic sampler replay"
                )
            group_examples = accumulation_group_denominator(
                emitted_batch_sizes,
                step=step,
                gradient_accumulation=args.gradient_accumulation,
            )
            should_update = (
                (step + 1) % args.gradient_accumulation == 0
                or step + 1 == len(train_loader)
            )
            sync_context = training_model.no_sync() if not should_update else nullcontext()
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = shared.relevance_logits(
                        training_model(**batch), args.model_backend
                    )
                    raw_loss, _ = loss_hook.compute(
                        logits=logits.float(),
                        targets=targets,
                        sample_weights=weights,
                        pair_indices=pair_indices,
                        orientations=orientations,
                        epoch=epoch,
                        step=step,
                    )
                    # raw_loss is a mean over this microbatch.  Weight it by the
                    # exact example share of its (possibly partial/shuffled)
                    # optimizer group so every row in that group has 1/N mass.
                    loss = raw_loss * (batch_examples / group_examples)
                scaler.scale(loss).backward()

            if should_update:
                optimizer_step_attempts += 1
                scaler.unscale_(optimizer)
                shared.torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, args.max_grad_norm
                )
                scale_before_step = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                if float(scaler.get_scale()) >= scale_before_step:
                    scheduler.step()
                    optimizer_steps_succeeded += 1
                else:
                    overflow_skips += 1
                    if overflow_skips > MAX_FULLTRAIN_OVERFLOW_SKIPS:
                        raise FloatingPointError(
                            "Final BGE training exceeded bounded AMP overflow skips"
                        )
                optimizer.zero_grad(set_to_none=True)

            local_examples += batch_examples
            local_useful_tokens += useful_tokens
            local_padded_tokens += padded_tokens
            interval_examples += batch_examples
            interval_useful_tokens += useful_tokens
            interval_padded_tokens += padded_tokens
            interval_loss += raw_loss.detach()
            interval_steps += 1
            previous_step_finished = time.perf_counter()

            if is_main and (
                (step + 1) % args.log_every == 0 or step + 1 == len(train_loader)
            ):
                torch.cuda.synchronize(device)
                interval_seconds = time.perf_counter() - interval_started
                seconds_per_step = interval_seconds / interval_steps
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "step": step + 1,
                            "steps": len(train_loader),
                            "loss": float(interval_loss) / interval_steps,
                            "examples_per_second": (
                                interval_examples * world_size / interval_seconds
                            ),
                            "seconds_per_step": seconds_per_step,
                            "epoch_eta_minutes": (
                                (len(train_loader) - step - 1)
                                * seconds_per_step
                                / 60
                            ),
                            "data_wait_fraction_approx": (
                                interval_data_seconds / interval_seconds
                            ),
                            "padding_efficiency": (
                                interval_useful_tokens / interval_padded_tokens
                            ),
                            "peak_vram_gib": (
                                torch.cuda.max_memory_allocated(device) / 2**30
                            ),
                            "learning_rate": scheduler.get_last_lr()[0],
                            "optimizer_steps_succeeded": optimizer_steps_succeeded,
                            "amp_overflow_skips": overflow_skips,
                        }
                    ),
                    flush=True,
                )
                interval_started = time.perf_counter()
                interval_loss = torch.zeros((), dtype=torch.float32, device=device)
                interval_steps = interval_examples = 0
                interval_useful_tokens = interval_padded_tokens = 0
                interval_data_seconds = 0.0
                previous_step_finished = interval_started

    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_started
    totals = torch.tensor(
        [local_examples, local_useful_tokens, local_padded_tokens],
        dtype=torch.float64,
        device=device,
    )
    elapsed = torch.tensor(training_seconds, dtype=torch.float64, device=device)
    counters = torch.tensor(
        [optimizer_step_attempts, optimizer_steps_succeeded, overflow_skips],
        dtype=torch.int64,
        device=device,
    )
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    gathered_counters = [torch.zeros_like(counters) for _ in range(world_size)]
    dist.all_gather(gathered_counters, counters)
    if any(not torch.equal(gathered_counters[0], value) for value in gathered_counters[1:]):
        raise RuntimeError("Final BGE optimizer counters differ across ranks")
    if optimizer_step_attempts != EXPECTED_TOTAL_UPDATES:
        raise FinalFulltrainContractError("Final BGE attempted update count changed")
    if int(totals[0].item()) != EXPECTED_TRAIN_ROWS * EXPECTED_EPOCHS:
        raise FinalFulltrainContractError("Final BGE example coverage changed")

    peak_memory = torch.tensor(
        [torch.cuda.max_memory_allocated(device) / 2**30],
        dtype=torch.float64,
        device=device,
    )
    gathered_peak_memory = [torch.zeros_like(peak_memory) for _ in range(world_size)]
    dist.all_gather(gathered_peak_memory, peak_memory)
    peak_memory_by_rank = [float(value.item()) for value in gathered_peak_memory]

    if is_main:
        report = {
            "schema_version": 1,
            "status": "complete",
            "purpose": "final_deployment_export",
            "quality_evaluation": False,
            "validation_splits": [],
            "validation_predictions_written": False,
            "training_seconds": float(elapsed.item()),
            "total_pipeline_seconds_before_save": time.perf_counter() - pipeline_started,
            "training_examples": int(totals[0].item()),
            "original_training_examples": EXPECTED_TRAIN_ROWS,
            "epochs": EXPECTED_EPOCHS,
            "training_subset": args.train_subset,
            "training_sampling": args.sampling,
            "training_loss_weighting": args.loss_weighting,
            "loss_hook": loss_hook.metadata,
            "training_unique_coverage_per_epoch": 1.0,
            "training_source_counts": source_counts,
            "training_source_weight_mass": training_source_weight_mass,
            "steps_per_epoch_per_rank": len(train_loader),
            "updates_per_epoch": updates_per_epoch,
            "planned_optimizer_updates": total_updates,
            "optimizer_step_attempts": optimizer_step_attempts,
            "optimizer_steps_succeeded": optimizer_steps_succeeded,
            "amp_overflow_skips": overflow_skips,
            "warmup_updates": warmup_updates,
            "gradient_accumulation_normalization": "sample_exact_group_mean_v1",
            "epoch_batch_geometry": epoch_batch_geometry,
            "examples_per_second": float(totals[0].item() / elapsed.item()),
            "padding_efficiency": float(totals[1].item() / totals[2].item()),
            "peak_vram_gib_by_rank": peak_memory_by_rank,
            "args": vars(args),
        }
        args.output_dir.mkdir(parents=True, exist_ok=False)
        inference_model = training_model.module
        inference_model.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
        # Transformers releases differ on whether save_pretrained re-emits this
        # file.  Pin the exact verified initial tokenizer companion explicitly.
        initial_special_tokens = Path(args.model) / "special_tokens_map.json"
        if (
            not initial_special_tokens.is_file()
            or initial_special_tokens.stat().st_size
            != EXPECTED_SPECIAL_TOKENS_MAP_BYTES
            or hashlib.sha256(initial_special_tokens.read_bytes()).hexdigest()
            != EXPECTED_SPECIAL_TOKENS_MAP_SHA256
        ):
            raise FinalFulltrainContractError(
                "Initial special_tokens_map.json changed"
            )
        shutil.copy2(
            initial_special_tokens,
            args.output_dir / "special_tokens_map.json",
        )
        (args.output_dir / "training_report.json").write_text(
            json.dumps(report, default=str, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "training_config.json").write_text(
            args.config.read_text(encoding="utf-8"), encoding="utf-8"
        )
        print(json.dumps(report, default=str, ensure_ascii=False, indent=2), flush=True)
        print(f"Saved final BGE export payload to {args.output_dir}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main() -> int:
    previous_adamw, previous_clip_grad_norm = bge.install_full_training_guards()
    try:
        train_and_export()
    finally:
        bge.restore_full_training_guards(previous_adamw, previous_clip_grad_norm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
