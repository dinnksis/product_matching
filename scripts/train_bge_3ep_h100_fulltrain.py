#!/usr/bin/env python3
"""Single-H100 three-epoch BGE export on all 365,654 human pairs.

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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_bge_3ep_h100 as bge
import train_cross_encoder as shared


EXPECTED_TRAIN_ROWS = 365_654
EXPECTED_TRAIN_POSITIVES = 93_890
EXPECTED_SOURCE_COUNTS = {
    "human_train": 306_669,
    "human_iid": 12_000,
    "human_hard": 5_814,
    "human_former_ood": 41_171,
}
EXPECTED_EPOCHS = 3
EXPECTED_STEPS_PER_EPOCH = 5_714
EXPECTED_UPDATES_PER_EPOCH = 1_905
EXPECTED_TOTAL_UPDATES = 5_715
EXPECTED_WARMUP_UPDATES = 285
EXPECTED_RECIPE_SHA256 = (
    "1135b3e43462ccde5a4f085b256ae594b63f29bdc27c54daf3696ecfb1c864dc"
)
EXPECTED_LOSS_HOOK_SHA256 = (
    "5a2350ddf544e1e2d56cc8aa5255221f936854393a16d8ec4c11e950be2d6251"
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
        "batch_size": 64,
        "eval_batch_size": 192,
        "gradient_accumulation": 3,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "max_length": 384,
        "attention_implementation": "sdpa",
        "sampling": "none",
        "train_subset": "all",
        "loss_weighting": "none",
        "lexical_hard_negative_strength": 0.0,
        "gradient_checkpointing": False,
        "bucket_size_multiplier": 50,
        "dataloader_workers": 16,
        "prefetch_factor": 4,
        "tokenization_batch_size": 1024,
        "tokenization_log_every": 50,
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
    if world_size != 1:
        mismatches["world_size"] = {
            "actual": world_size,
            "expected": 1,
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

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise FinalFulltrainContractError("Final H100 export requires one process")
    bge.require_single_h100()
    torch.cuda.set_device(0)
    rank = 0
    world_size = 1
    device = torch.device("cuda", 0)
    is_main = True
    validate_final_contract(args, world_size=world_size)
    bge.load_training_config(args.config)

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
        False,
        None,
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
    if len(train_sampler) != EXPECTED_STEPS_PER_EPOCH:
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
    if amp_dtype != torch.bfloat16:
        raise FinalFulltrainContractError("Final BGE export requires BF16 on H100")
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
    training_model = model
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
                    "amp_dtype": "bfloat16",
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
            len(emitted_batch_sizes) != EXPECTED_STEPS_PER_EPOCH
            or sum(emitted_batch_sizes) != EXPECTED_TRAIN_ROWS
            or emitted_batch_sizes.count(22) != 1
            or any(size not in {22, args.batch_size} for size in emitted_batch_sizes)
        ):
            raise FinalFulltrainContractError(
                "Final BGE emitted batch-size sequence changed"
            )
        batch_size_digest = hashlib.sha256(
            json.dumps(emitted_batch_sizes, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        partial_batch_step = emitted_batch_sizes.index(22)
        partial_group_denominator = accumulation_group_denominator(
            emitted_batch_sizes,
            step=partial_batch_step,
            gradient_accumulation=args.gradient_accumulation,
        )
        epoch_batch_geometry.append(
            {
                "epoch": epoch + 1,
                "batch_sizes_sha256": batch_size_digest,
                "partial_batch_step_zero_based": partial_batch_step,
                "partial_batch_size": 22,
                "partial_group_examples": partial_group_denominator,
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
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
            loss.backward()

            if should_update:
                optimizer_step_attempts += 1
                shared.torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, args.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer_steps_succeeded += 1
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
    if optimizer_step_attempts != EXPECTED_TOTAL_UPDATES:
        raise FinalFulltrainContractError("Final BGE attempted update count changed")
    if local_examples != EXPECTED_TRAIN_ROWS * EXPECTED_EPOCHS:
        raise FinalFulltrainContractError("Final BGE example coverage changed")

    peak_memory_by_rank = [torch.cuda.max_memory_allocated(device) / 2**30]

    if is_main:
        report = {
            "schema_version": 1,
            "status": "complete",
            "purpose": "final_deployment_export",
            "quality_evaluation": False,
            "validation_splits": [],
            "validation_predictions_written": False,
            "training_seconds": training_seconds,
            "total_pipeline_seconds_before_save": time.perf_counter() - pipeline_started,
            "training_examples": local_examples,
            "original_training_examples": EXPECTED_TRAIN_ROWS,
            "epochs": EXPECTED_EPOCHS,
            "training_subset": args.train_subset,
            "training_sampling": args.sampling,
            "training_loss_weighting": args.loss_weighting,
            "loss_hook": loss_hook.metadata,
            "training_unique_coverage_per_epoch": 1.0,
            "training_source_counts": source_counts,
            "training_source_weight_mass": training_source_weight_mass,
            "steps_per_epoch": len(train_loader),
            "updates_per_epoch": updates_per_epoch,
            "planned_optimizer_updates": total_updates,
            "optimizer_step_attempts": optimizer_step_attempts,
            "optimizer_steps_succeeded": optimizer_steps_succeeded,
            "amp_overflow_skips": overflow_skips,
            "warmup_updates": warmup_updates,
            "gradient_accumulation_normalization": "sample_exact_group_mean_v1",
            "epoch_batch_geometry": epoch_batch_geometry,
            "examples_per_second": float(local_examples / training_seconds),
            "padding_efficiency": float(local_useful_tokens / local_padded_tokens),
            "peak_vram_gib_by_rank": peak_memory_by_rank,
            "args": vars(args),
        }
        args.output_dir.mkdir(parents=True, exist_ok=False)
        inference_model = training_model
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


def main() -> int:
    previous_adamw = shared.AdamW
    previous_clip_grad_norm = shared.torch.nn.utils.clip_grad_norm_
    previous_dtype = shared.preferred_cuda_dtype
    shared.AdamW = bge.adamw_foreach_disabled
    shared.torch.nn.utils.clip_grad_norm_ = bge.strict_clip_grad_norm
    shared.preferred_cuda_dtype = bge.h100_bfloat16_dtype
    try:
        train_and_export()
    finally:
        shared.AdamW = previous_adamw
        shared.torch.nn.utils.clip_grad_norm_ = previous_clip_grad_norm
        shared.preferred_cuda_dtype = previous_dtype
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
