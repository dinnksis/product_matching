"""Efficient LoRA/full fine-tuning of Qwen3-Reranker for product matching.

The prepared product text contains category, name and priority-ordered attributes.
Pair tokenization is cached once, batches are grouped by length, and validation is
run once after all training epochs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from sklearn.metrics import average_precision_score
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline import attach_item_fields
from src.experiment_protocol import validation_split_paths
from src.qwen_reranker import preferred_cuda_dtype
from src.qwen_training import (
    FixedLengthBatchSampler,
    LengthBucketBatchSampler,
    PackedBatchCollator,
    PackedPairDataset,
    TokenCache,
    balanced_sampling_weights,
    build_token_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--prepared-dir", type=Path, default=Path("prepared/human"))
    parser.add_argument("--output-dir", type=Path, default=Path("model/qwen_products_lora"))
    parser.add_argument("--token-cache-dir", type=Path)
    parser.add_argument(
        "--validation-split",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Named validation pair file; repeat for IID, hard and OOD. Relative "
            "paths are resolved below --prepared-dir. Defaults to iid=val_pairs.parquet."
        ),
    )
    parser.add_argument("--hard-negatives", type=Path)
    parser.add_argument("--hard-negative-weight", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16, help="Per GPU")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="Per GPU")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument(
        "--attention-implementation",
        default="sdpa",
        choices=["sdpa", "flash_attention_2"],
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument(
        "--lora-targets",
        choices=["attention", "attention_mlp"],
        default="attention_mlp",
    )
    parser.add_argument("--training-mode", choices=["lora", "full"], default="lora")
    parser.add_argument(
        "--sampling",
        choices=["none", "category", "category_label"],
        default="category_label",
    )
    parser.add_argument("--bucket-size-multiplier", type=int, default=50)
    parser.add_argument("--dataloader-workers", type=int, default=2, help="Per DDP process")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--tokenization-batch-size", type=int, default=512)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--symmetric-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "eval-batch-size": args.eval_batch_size,
        "gradient-accumulation": args.gradient_accumulation,
        "max-length": args.max_length,
        "tokenization-batch-size": args.tokenization_batch_size,
        "bucket-size-multiplier": args.bucket_size_multiplier,
        "prefetch-factor": args.prefetch_factor,
        "log-every": args.log_every,
        "max-grad-norm": args.max_grad_norm,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"These arguments must be positive: {', '.join(invalid)}")
    if args.dataloader_workers < 0:
        raise ValueError("dataloader-workers must be non-negative")
    if args.hard_negative_weight <= 0:
        raise ValueError("hard-negative-weight must be positive")


def add_hard_negatives(
    train_pairs: pd.DataFrame,
    validation_pairs: pd.DataFrame,
    path: Path | None,
    weight: float,
) -> pd.DataFrame:
    train_pairs = train_pairs.reset_index(drop=True).copy()
    train_pairs["sample_weight"] = 1.0
    train_pairs["is_hard_negative"] = False
    if path is None:
        return train_pairs

    hard = pd.read_parquet(path)
    required = {"id1", "id2"}
    if missing := required - set(hard.columns):
        raise ValueError(f"Hard-negative file is missing columns: {sorted(missing)}")
    hard = hard.copy()
    if "target" not in hard:
        hard["target"] = 0.0
    if (hard["target"] != 0).any():
        raise ValueError("The hard-negative file must contain only target=0 rows")
    if (hard["id1"] == hard["id2"]).any():
        raise ValueError("The hard-negative file contains self-pairs")

    validation_ids = set(validation_pairs["id1"]) | set(validation_pairs["id2"])
    leaking = hard["id1"].isin(validation_ids) | hard["id2"].isin(validation_ids)
    if leaking.any():
        print(f"Dropped {int(leaking.sum()):,} hard negatives touching validation items")
        hard = hard.loc[~leaking].copy()

    def canonical_keys(frame: pd.DataFrame) -> pd.MultiIndex:
        left = np.minimum(frame["id1"].to_numpy(), frame["id2"].to_numpy())
        right = np.maximum(frame["id1"].to_numpy(), frame["id2"].to_numpy())
        return pd.MultiIndex.from_arrays([left, right])

    existing = canonical_keys(train_pairs)
    if existing.has_duplicates:
        raise ValueError("Training pairs contain duplicate unordered pairs")
    hard_keys = canonical_keys(hard)
    existing_positions = existing.get_indexer(hard_keys)
    matched_positions = np.unique(existing_positions[existing_positions >= 0])
    if len(matched_positions):
        train_pairs.loc[matched_positions, "sample_weight"] = weight
        train_pairs.loc[matched_positions, "is_hard_negative"] = True
    new_mask = (existing_positions < 0) & ~hard_keys.duplicated()
    hard = hard.loc[new_mask, ["id1", "id2", "target"]]
    hard["sample_weight"] = weight
    hard["is_hard_negative"] = True
    print(
        f"Reweighted {len(matched_positions):,} known and added {len(hard):,} external "
        f"hard negatives with loss weight {weight:g}"
    )
    return pd.concat([train_pairs, hard], ignore_index=True)


def loader_options(args: argparse.Namespace, *, persistent: bool) -> dict[str, Any]:
    options: dict[str, Any] = {
        "num_workers": args.dataloader_workers,
        "pin_memory": True,
    }
    if args.dataloader_workers:
        options.update(
            persistent_workers=persistent,
            prefetch_factor=args.prefetch_factor,
        )
    return options


def create_caches(
    train: pd.DataFrame,
    validations: dict[str, pd.DataFrame],
    tokenizer: Any,
    args: argparse.Namespace,
    is_main: bool,
    distributed: bool,
) -> tuple[TokenCache, dict[str, TokenCache]]:
    cache_root = args.token_cache_dir or (
        Path("artifacts/token_cache") / args.output_dir.name
    )
    payload: dict[str, Any] | None = None
    if is_main:
        train_cache = build_token_cache(
            train,
            tokenizer,
            cache_root,
            "train",
            args.model,
            args.max_length,
            args.tokenization_batch_size,
        )
        validation_caches = {
            name: build_token_cache(
                validation,
                tokenizer,
                cache_root,
                f"validation_{name}",
                args.model,
                args.max_length,
                args.tokenization_batch_size,
            )
            for name, validation in validations.items()
        }
        payload = {
            "train_path": str(train_cache.directory),
            "validation_paths": {
                name: str(cache.directory)
                for name, cache in validation_caches.items()
            },
        }
    if distributed:
        message: list[Any] = [payload]
        dist.broadcast_object_list(message, src=0)
        payload = message[0]
    if payload is None:
        raise RuntimeError("Token cache paths were not initialized")
    train_path = payload.get("train_path")
    validation_paths = payload.get("validation_paths")
    if not isinstance(train_path, str) or not isinstance(validation_paths, dict):
        raise RuntimeError(f"Invalid token cache payload: {payload}")
    if set(validation_paths) != set(validations):
        raise RuntimeError(f"Validation token cache payload differs: {payload}")
    return TokenCache.load(Path(train_path)), {
        name: TokenCache.load(Path(validation_paths[name]))
        for name in validations
    }


class BinaryReranker(torch.nn.Module):
    """Run the Qwen decoder and project its last state to only no/yes rows.

    ``logits_to_keep=1`` avoids vocabulary logits at earlier positions, but the
    regular CausalLM head still projects the last position to all 151k tokens.
    The loss uses only no/yes, so selecting those two tied-embedding rows before
    the projection is mathematically equivalent and avoids the full-vocab GEMM.
    """

    def __init__(
        self, causal_model: torch.nn.Module, no_id: int, yes_id: int
    ) -> None:
        super().__init__()
        self.causal_model = causal_model
        self.register_buffer(
            "class_token_ids", torch.tensor([no_id, yes_id], dtype=torch.long)
        )

    def forward(self, **batch: torch.Tensor) -> torch.Tensor:
        base = (
            self.causal_model.get_base_model()
            if hasattr(self.causal_model, "get_base_model")
            else self.causal_model
        )
        hidden = base.model(
            **batch, use_cache=False, return_dict=True
        ).last_hidden_state[:, -1]
        class_weights = base.lm_head.weight.index_select(0, self.class_token_ids)
        return F.linear(hidden, class_weights)


@dataclass(frozen=True)
class ValidationResult:
    macro_average_precision: float
    overall_average_precision: float
    per_category_average_precision: dict[str, float]
    scores: np.ndarray
    scores_ab: np.ndarray
    scores_ba: np.ndarray


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    targets: np.ndarray,
    categories: list[str],
    device: torch.device,
    distributed: bool,
    world_size: int,
) -> ValidationResult | None:
    model.eval()
    local: list[tuple[int, bool, float]] = []
    with torch.inference_mode():
        for packed in loader:
            pair_indices = packed.pop("pair_indices").tolist()
            orientations = packed.pop("orientations").tolist()
            packed.pop("targets")
            packed.pop("sample_weights")
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in packed.items()
            }
            probabilities = model(**batch).float().softmax(1)[:, 1]
            local.extend(
                (int(index), bool(reverse), float(probability))
                for index, reverse, probability in zip(
                    pair_indices, orientations, probabilities.cpu().tolist()
                )
            )
    gathered: list[Any]
    if distributed:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    if distributed and dist.get_rank() != 0:
        return None

    scores_ab = np.full(len(targets), np.nan, dtype=np.float32)
    scores_ba = np.full(len(targets), np.nan, dtype=np.float32)
    for part in gathered:
        for index, reverse, score in part:
            destination = scores_ba if reverse else scores_ab
            if np.isfinite(destination[index]):
                raise RuntimeError(
                    f"Validation produced a duplicate score for pair {index}, "
                    f"reverse={reverse}"
                )
            destination[index] = score
    if not np.isfinite(scores_ab).all():
        missing = int((~np.isfinite(scores_ab)).sum())
        raise RuntimeError(f"Validation produced {missing} missing A/B scores")
    has_reverse_scores = bool(np.isfinite(scores_ba).any())
    if has_reverse_scores and not np.isfinite(scores_ba).all():
        missing = int((~np.isfinite(scores_ba)).sum())
        raise RuntimeError(f"Validation produced {missing} missing B/A scores")
    scores = (
        (scores_ab + scores_ba) / 2.0
        if has_reverse_scores
        else scores_ab.copy()
    )
    frame = pd.DataFrame({"target": targets, "predict": scores, "category": categories})
    per_category = frame.groupby("category").apply(
        lambda group: average_precision_score(group["target"], group["predict"]),
        include_groups=False,
    )
    return ValidationResult(
        macro_average_precision=float(per_category.mean()),
        overall_average_precision=float(average_precision_score(targets, scores)),
        per_category_average_precision={
            str(key): float(value) for key, value in per_category.items()
        },
        scores=scores,
        scores_ab=scores_ab,
        scores_ba=scores_ba,
    )


def build_validation_predictions(
    validation: pd.DataFrame,
    cache: TokenCache,
    result: ValidationResult,
    max_length: int,
) -> pd.DataFrame:
    columns = [
        "id1",
        "id2",
        "target",
        "category_1",
        "category_2",
        "product_text_1",
        "product_text_2",
    ]
    if missing := [column for column in columns if column not in validation]:
        raise ValueError(f"Validation metadata is missing columns: {missing}")
    predictions = validation[columns].reset_index(drop=True).copy()
    predictions.insert(0, "pair_index", np.arange(len(predictions), dtype=np.int64))
    predictions["score"] = result.scores
    predictions["score_ab"] = result.scores_ab
    predictions["score_ba"] = result.scores_ba
    predictions["score_order_gap"] = np.abs(result.scores_ab - result.scores_ba)
    predictions["token_length_ab"] = cache.forward_lengths.astype(np.int32)
    predictions["token_length_ba"] = cache.reverse_lengths.astype(np.int32)
    predictions["reached_max_length_ab"] = (
        predictions["token_length_ab"] >= max_length
    )
    predictions["reached_max_length_ba"] = (
        predictions["token_length_ba"] >= max_length
    )
    return predictions


def main() -> None:
    args = parse_args()
    validate_args(args)
    pipeline_started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    device = torch.device("cuda", local_rank)
    is_main = rank == 0
    if (
        args.attention_implementation == "flash_attention_2"
        and torch.cuda.get_device_capability(device)[0] < 8
    ):
        raise ValueError(
            "flash_attention_2 is not supported on Turing GPUs such as T4; use sdpa"
        )
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    items_path = args.prepared_dir / "items.parquet"
    train_path = args.prepared_dir / "train_pairs.parquet"
    validation_paths = validation_split_paths(
        args.prepared_dir,
        args.validation_split,
    )
    if missing_files := [str(path) for path in validation_paths.values() if not path.is_file()]:
        raise FileNotFoundError(
            f"Validation pair files do not exist: {missing_files}"
        )
    items = pd.read_parquet(items_path)
    required_item_columns = {"id", "product_text", "category"}
    if missing := required_item_columns - set(items.columns):
        raise ValueError(
            f"Prepared items are missing {sorted(missing)}; rerun scripts/prepare_human_data.py"
        )
    items = items[["id", "product_text", "category"]]
    train_pairs = pd.read_parquet(train_path)
    original_train_pairs = len(train_pairs)
    validation_pairs = {
        name: pd.read_parquet(path) for name, path in validation_paths.items()
    }
    all_validation_pairs = pd.concat(
        [frame[["id1", "id2"]] for frame in validation_pairs.values()],
        ignore_index=True,
    )
    train_pairs = add_hard_negatives(
        train_pairs,
        all_validation_pairs,
        args.hard_negatives,
        args.hard_negative_weight,
    )
    train = attach_item_fields(train_pairs, items, fields=("product_text", "category"))
    validations = {
        name: attach_item_fields(
            pairs, items, fields=("product_text", "category")
        )
        for name, pairs in validation_pairs.items()
    }
    if (train["category_1"] != train["category_2"]).any():
        raise ValueError("Training contains cross-category pairs")
    if not train["target"].between(0, 1).all():
        raise ValueError("Training targets must be probabilities in [0, 1]")
    for name, validation in validations.items():
        if (validation["category_1"] != validation["category_2"]).any():
            raise ValueError(f"Validation split {name!r} contains cross-category pairs")
        if not validation["target"].isin([0.0, 1.0]).all():
            raise ValueError(
                f"Validation split {name!r} targets must be binary for average precision"
            )

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_cache, validation_caches = create_caches(
        train, validations, tokenizer, args, is_main, distributed
    )
    # The fast tokenizer has already finished its internally parallel work.
    # Disable its fork warning before DataLoader worker processes are created.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    train_targets = train["target"].to_numpy(dtype=np.float32)
    train_categories = train["category_1"].astype(str).tolist()
    validation_targets = {
        name: validation["target"].to_numpy(dtype=np.float32)
        for name, validation in validations.items()
    }
    validation_categories = {
        name: validation["category_1"].astype(str).tolist()
        for name, validation in validations.items()
    }
    sample_weights = train["sample_weight"].to_numpy(dtype=np.float32)
    hard_negative_count = int(train["is_hard_negative"].sum())
    sampling_weights = balanced_sampling_weights(
        train_categories, train_targets, args.sampling
    )
    train_dataset = PackedPairDataset(train_cache, train_targets, sample_weights)
    validation_datasets = {
        name: PackedPairDataset(validation_caches[name], validation_targets[name])
        for name in validations
    }
    train_sampler = LengthBucketBatchSampler(
        train_cache.forward_lengths,
        train_cache.reverse_lengths,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
        weights=sampling_weights,
        bucket_size_multiplier=args.bucket_size_multiplier,
        seed=args.seed,
    )
    validation_samplers = {
        name: FixedLengthBatchSampler(
            validation_caches[name],
            np.arange(rank, len(validation_datasets[name]), world_size),
            args.eval_batch_size,
            both_orientations=args.symmetric_validation,
        )
        for name in validations
    }
    collator = PackedBatchCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collator,
        **loader_options(args, persistent=True),
    )
    validation_loaders = {
        name: DataLoader(
            validation_datasets[name],
            batch_sampler=validation_samplers[name],
            collate_fn=collator,
            **loader_options(args, persistent=False),
        )
        for name in validations
    }
    del train, items, all_validation_pairs, validation_pairs

    model_dtype = preferred_cuda_dtype()
    parameter_dtype = torch.float32 if args.training_mode == "full" else model_dtype
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=parameter_dtype,
        attn_implementation=args.attention_implementation,
    )
    if args.training_mode == "lora":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if args.lora_targets == "attention_mlp":
            target_modules.extend(["gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_rank * 2,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            ),
        )
    model = model.to(device)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if is_main:
        print(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(device),
                    "world_size": world_size,
                    "dtype": str(model_dtype),
                    "training_mode": args.training_mode,
                    "train_pairs": len(train_dataset),
                    "original_train_pairs": original_train_pairs,
                    "validation_pairs": {
                        name: len(dataset)
                        for name, dataset in validation_datasets.items()
                    },
                    "hard_negative_pairs": hard_negative_count,
                    "per_device_batch": args.batch_size,
                    "effective_batch": args.batch_size
                    * world_size
                    * args.gradient_accumulation,
                    "gradient_checkpointing": args.gradient_checkpointing,
                    "dataloader_workers_total": args.dataloader_workers * world_size,
                },
                ensure_ascii=False,
            )
        )
        if args.training_mode == "lora":
            model.print_trainable_parameters()
        else:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Full fine-tuning: {trainable:,} trainable parameters")

    no_id = tokenizer("no", add_special_tokens=False).input_ids[0]
    yes_id = tokenizer("yes", add_special_tokens=False).input_ids[0]
    training_model: torch.nn.Module = BinaryReranker(model, no_id, yes_id).to(device)
    if distributed:
        training_model = DistributedDataParallel(
            training_model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 2e-4 if args.training_mode == "lora" else 2e-5
    trainable_parameters = [p for p in training_model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_parameters, lr=learning_rate)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        max(1, int(total_updates * 0.05)),
        total_updates,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=model_dtype == torch.float16)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    training_started = time.perf_counter()
    local_examples = local_useful_tokens = local_padded_tokens = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        training_model.train()
        interval_started = time.perf_counter()
        interval_loss = torch.zeros((), dtype=torch.float32, device=device)
        interval_steps = 0
        interval_examples = 0
        interval_useful_tokens = 0
        interval_padded_tokens = 0
        previous_step_finished = interval_started
        interval_data_seconds = 0.0

        for step, packed in enumerate(train_loader):
            batch_ready = time.perf_counter()
            interval_data_seconds += batch_ready - previous_step_finished
            packed.pop("pair_indices")
            packed.pop("orientations")
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
            group_start = (step // args.gradient_accumulation) * args.gradient_accumulation
            group_size = min(
                args.gradient_accumulation, len(train_loader) - group_start
            )
            should_update = (
                (step + 1) % args.gradient_accumulation == 0
                or step + 1 == len(train_loader)
            )
            sync_context = (
                training_model.no_sync()
                if distributed and not should_update
                else nullcontext()
            )
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=model_dtype):
                    logits = training_model(**batch)
                    per_example_loss = F.binary_cross_entropy_with_logits(
                        (logits[:, 1] - logits[:, 0]).float(),
                        targets,
                        reduction="none",
                    )
                    raw_loss = (per_example_loss * weights).sum() / weights.sum()
                    loss = raw_loss / group_size
                scaler.scale(loss).backward()

            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, args.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
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

            if is_main and ((step + 1) % args.log_every == 0 or step + 1 == len(train_loader)):
                torch.cuda.synchronize(device)
                interval_seconds = time.perf_counter() - interval_started
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "step": step + 1,
                            "steps": len(train_loader),
                            "loss": float(interval_loss) / interval_steps,
                            "examples_per_second": interval_examples
                            * world_size
                            / interval_seconds,
                            "data_wait_fraction_approx": interval_data_seconds
                            / interval_seconds,
                            "padding_efficiency": interval_useful_tokens
                            / interval_padded_tokens,
                            "peak_vram_gib": torch.cuda.max_memory_allocated(device)
                            / 2**30,
                            "learning_rate": scheduler.get_last_lr()[0],
                        }
                    )
                )
                interval_started = time.perf_counter()
                interval_loss = torch.zeros((), dtype=torch.float32, device=device)
                interval_steps = 0
                interval_examples = 0
                interval_useful_tokens = 0
                interval_padded_tokens = 0
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
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)

    # Shut persistent training workers down before validation workers are
    # started, otherwise both pools remain alive and contend for the same CPUs.
    del train_loader
    inference_model = training_model.module if distributed else training_model
    torch.cuda.synchronize(device)
    validation_started = time.perf_counter()
    validation_results: dict[str, ValidationResult] = {}
    validation_seconds_by_split: dict[str, float] = {}
    for name, validation_loader in validation_loaders.items():
        split_started = time.perf_counter()
        validation_result = evaluate(
            inference_model,
            validation_loader,
            validation_targets[name],
            validation_categories[name],
            device,
            distributed,
            world_size,
        )
        torch.cuda.synchronize(device)
        validation_seconds_by_split[name] = time.perf_counter() - split_started
        if is_main:
            if validation_result is None:
                raise RuntimeError(
                    f"Main rank did not receive metrics for validation split {name!r}"
                )
            validation_results[name] = validation_result
    if distributed:
        dist.barrier()
    torch.cuda.synchronize(device)
    validation_seconds = time.perf_counter() - validation_started
    peak_memory = torch.tensor(
        [torch.cuda.max_memory_allocated(device) / 2**30],
        dtype=torch.float64,
        device=device,
    )
    if distributed:
        gathered_peak_memory = [
            torch.zeros_like(peak_memory) for _ in range(world_size)
        ]
        dist.all_gather(gathered_peak_memory, peak_memory)
        peak_memory_by_rank = [float(value.item()) for value in gathered_peak_memory]
    else:
        peak_memory_by_rank = [float(peak_memory.item())]

    if is_main:
        validation_reports: dict[str, dict[str, Any]] = {}
        validation_predictions: dict[str, pd.DataFrame] = {}
        for name, validation in validations.items():
            result = validation_results[name]
            predictions = build_validation_predictions(
                validation,
                validation_caches[name],
                result,
                args.max_length,
            )
            predictions_filename = f"{name}_validation_predictions.parquet"
            order_gap = predictions["score_order_gap"]
            validation_predictions[name] = predictions
            validation_reports[name] = {
                "examples": len(predictions),
                "positive_examples": int(validation_targets[name].sum()),
                "positive_rate": float(validation_targets[name].mean()),
                "macro_average_precision": result.macro_average_precision,
                "overall_average_precision": result.overall_average_precision,
                "per_category_average_precision": (
                    result.per_category_average_precision
                ),
                "predictions_file": predictions_filename,
                "mean_score_order_gap": (
                    float(order_gap.mean()) if order_gap.notna().any() else None
                ),
            }
        primary_name = (
            "iid" if "iid" in validation_reports else next(iter(validation_reports))
        )
        primary = validation_reports[primary_name]
        report = {
            "training_seconds": float(elapsed.item()),
            "validation_seconds": validation_seconds,
            "validation_seconds_by_split": validation_seconds_by_split,
            "total_pipeline_seconds": time.perf_counter() - pipeline_started,
            "training_examples": int(totals[0].item()),
            "original_training_examples": original_train_pairs,
            "hard_negative_pairs": hard_negative_count,
            "primary_validation_split": primary_name,
            "validation_splits": validation_reports,
            "validation_examples": primary["examples"],
            "validation_positive_examples": primary["positive_examples"],
            "validation_positive_rate": primary["positive_rate"],
            "examples_per_second": float(totals[0].item() / elapsed.item()),
            "padding_efficiency": float(totals[1].item() / totals[2].item()),
            "peak_vram_gib_by_rank": peak_memory_by_rank,
            "macro_average_precision": primary["macro_average_precision"],
            "overall_average_precision": primary["overall_average_precision"],
            "per_category_average_precision": primary[
                "per_category_average_precision"
            ],
            "validation_predictions_file": primary["predictions_file"],
            "mean_score_order_gap": primary["mean_score_order_gap"],
            "args": vars(args),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        inference_model.causal_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        for name, predictions in validation_predictions.items():
            predictions.to_parquet(
                args.output_dir / validation_reports[name]["predictions_file"],
                index=False,
                compression="zstd",
            )
        (args.output_dir / "training_report.json").write_text(
            json.dumps(report, default=str, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(report, default=str, ensure_ascii=False, indent=2))
        print(f"Saved {args.training_mode} model to {args.output_dir}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
