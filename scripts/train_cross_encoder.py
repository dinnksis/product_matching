"""Configurable DDP fine-tuning for compact product-pair cross-encoders."""

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
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cross_encoder_training import (
    CrossEncoderBatchCollator,
    CrossEncoderPairDataset,
    PairTokenCache,
    build_pair_token_cache,
)
from src.data_pipeline import attach_item_fields
from src.experiment_protocol import validation_split_paths
from src.pair_features import (
    build_training_loss_weights,
    category_label_downsample,
    name_ngram_cosine,
)
from src.qwen_reranker import preferred_cuda_dtype
from src.qwen_training import (
    FixedLengthBatchSampler,
    LengthBucketBatchSampler,
    balanced_sampling_weights,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "cross_encoder_minilm.json"
CONFIG_KEYS = {
    "model",
    "epochs",
    "batch_size",
    "eval_batch_size",
    "gradient_accumulation",
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "max_length",
    "attention_implementation",
    "sampling",
    "train_subset",
    "loss_weighting",
    "lexical_hard_negative_strength",
    "bucket_size_multiplier",
    "dataloader_workers",
    "prefetch_factor",
    "tokenization_batch_size",
    "tokenization_log_every",
    "gradient_checkpointing",
    "symmetric_validation",
    "label_smoothing",
    "max_grad_norm",
    "log_every",
    "seed",
}


@dataclass(frozen=True)
class ValidationResult:
    macro_average_precision: float
    overall_average_precision: float
    per_category_average_precision: dict[str, float]
    scores: np.ndarray
    scores_ab: np.ndarray
    scores_ba: np.ndarray


def load_config_from_cli() -> tuple[Path, dict[str, Any]]:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    known, _ = pre_parser.parse_known_args()
    config_path = known.config
    if not config_path.is_file():
        raise SystemExit(f"Config does not exist: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit("Training config must contain a JSON object")
    if unknown := set(config) - CONFIG_KEYS:
        raise SystemExit(f"Unknown training config keys: {sorted(unknown)}")
    return config_path, config


def parse_args() -> argparse.Namespace:
    config_path, config = load_config_from_cli()

    def configured(name: str, fallback: Any) -> Any:
        return config.get(name, fallback)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=config_path)
    parser.add_argument("--model", default=configured("model", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"))
    parser.add_argument("--prepared-dir", type=Path, default=Path("prepared/human"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/cross_encoder_minilm"))
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
    parser.add_argument("--epochs", type=int, default=configured("epochs", 1))
    parser.add_argument("--batch-size", type=int, default=configured("batch_size", 96), help="Per GPU")
    parser.add_argument("--eval-batch-size", type=int, default=configured("eval_batch_size", 192), help="Per GPU")
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=configured("gradient_accumulation", 1),
    )
    parser.add_argument("--learning-rate", type=float, default=configured("learning_rate", 2e-5))
    parser.add_argument("--weight-decay", type=float, default=configured("weight_decay", 0.01))
    parser.add_argument("--warmup-ratio", type=float, default=configured("warmup_ratio", 0.05))
    parser.add_argument("--max-length", type=int, default=configured("max_length", 256))
    parser.add_argument(
        "--attention-implementation",
        choices=["eager", "sdpa"],
        default=configured("attention_implementation", "sdpa"),
    )
    parser.add_argument(
        "--sampling",
        choices=["none", "category", "category_label"],
        default=configured("sampling", "category_label"),
    )
    parser.add_argument(
        "--train-subset",
        choices=["all", "category_label_downsample"],
        default=configured("train_subset", "all"),
        help="Optional deterministic filtering before tokenization",
    )
    parser.add_argument(
        "--loss-weighting",
        choices=["none", "category_label_sqrt"],
        default=configured("loss_weighting", "none"),
        help="Per-example BCE weighting; unlike sampling, retains every row",
    )
    parser.add_argument(
        "--lexical-hard-negative-strength",
        type=float,
        default=configured("lexical_hard_negative_strength", 0.0),
        help="Within-category emphasis for lexically similar negative pairs",
    )
    parser.add_argument(
        "--bucket-size-multiplier",
        type=int,
        default=configured("bucket_size_multiplier", 50),
    )
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=configured("dataloader_workers", 2),
        help="Per DDP process",
    )
    parser.add_argument("--prefetch-factor", type=int, default=configured("prefetch_factor", 2))
    parser.add_argument(
        "--tokenization-batch-size",
        type=int,
        default=configured("tokenization_batch_size", 512),
    )
    parser.add_argument(
        "--tokenization-log-every",
        type=int,
        default=configured("tokenization_log_every", 50),
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=configured("gradient_checkpointing", False),
    )
    parser.add_argument(
        "--symmetric-validation",
        action=argparse.BooleanOptionalAction,
        default=configured("symmetric_validation", True),
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=configured("label_smoothing", 0.0)
    )
    parser.add_argument("--max-grad-norm", type=float, default=configured("max_grad_norm", 1.0))
    parser.add_argument("--log-every", type=int, default=configured("log_every", 20))
    parser.add_argument("--seed", type=int, default=configured("seed", 42))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "eval-batch-size": args.eval_batch_size,
        "gradient-accumulation": args.gradient_accumulation,
        "learning-rate": args.learning_rate,
        "max-length": args.max_length,
        "bucket-size-multiplier": args.bucket_size_multiplier,
        "prefetch-factor": args.prefetch_factor,
        "tokenization-batch-size": args.tokenization_batch_size,
        "tokenization-log-every": args.tokenization_log_every,
        "max-grad-norm": args.max_grad_norm,
        "log-every": args.log_every,
    }
    if invalid := [name for name, value in positive.items() if value <= 0]:
        raise ValueError(f"These arguments must be positive: {', '.join(invalid)}")
    if args.dataloader_workers < 0:
        raise ValueError("dataloader-workers must be non-negative")
    if args.weight_decay < 0:
        raise ValueError("weight-decay must be non-negative")
    if args.lexical_hard_negative_strength < 0:
        raise ValueError("lexical-hard-negative-strength must be non-negative")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup-ratio must be in [0, 1)")
    if not 0 <= args.label_smoothing < 1:
        raise ValueError("label-smoothing must be in [0, 1)")


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
    control_group: dist.ProcessGroup | None,
) -> tuple[PairTokenCache, dict[str, PairTokenCache]]:
    cache_root = args.token_cache_dir or (
        Path("artifacts/token_cache") / args.output_dir.name
    )
    payload: dict[str, Any] | None = None
    local_error: Exception | None = None
    if is_main:
        try:
            train_cache = build_pair_token_cache(
                train,
                tokenizer,
                cache_root,
                "train",
                args.model,
                args.max_length,
                args.tokenization_batch_size,
                args.tokenization_log_every,
            )
            validation_caches = {
                name: build_pair_token_cache(
                    validation,
                    tokenizer,
                    cache_root,
                    f"validation_{name}",
                    args.model,
                    args.max_length,
                    args.tokenization_batch_size,
                    args.tokenization_log_every,
                )
                for name, validation in validations.items()
            }
            payload = {
                "train_path": str(train_cache.directory),
                "validation_paths": {
                    name: str(cache.directory)
                    for name, cache in validation_caches.items()
                },
                "error": None,
            }
        except Exception as error:
            local_error = error
            payload = {
                "paths": None,
                "error": f"{type(error).__name__}: {error}",
            }
    if distributed:
        if control_group is None:
            raise RuntimeError("Distributed cache coordination requires a control group")
        message: list[Any] = [payload]
        # Tokenization can take longer than NCCL's normal collective timeout.
        # Keep this CPU-only coordination off the GPU process group.
        dist.broadcast_object_list(message, src=0, group=control_group)
        payload = message[0]
    if payload is None:
        raise RuntimeError("Token cache paths were not initialized")
    if payload["error"] is not None:
        if local_error is not None:
            raise local_error
        raise RuntimeError(f"Rank 0 failed to create token caches: {payload['error']}")
    train_path = payload.get("train_path")
    validation_paths = payload.get("validation_paths")
    if not isinstance(train_path, str) or not isinstance(validation_paths, dict):
        raise RuntimeError(f"Invalid token cache payload: {payload}")
    if set(validation_paths) != set(validations):
        raise RuntimeError(f"Validation token cache payload differs: {payload}")
    return PairTokenCache.load(Path(train_path)), {
        name: PairTokenCache.load(Path(validation_paths[name]))
        for name in validations
    }


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    targets: np.ndarray,
    categories: list[str],
    device: torch.device,
    amp_dtype: torch.dtype,
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
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(**batch).logits.squeeze(-1)
            probabilities = logits.float().sigmoid().cpu().tolist()
            local.extend(
                (int(index), bool(reverse), float(probability))
                for index, reverse, probability in zip(
                    pair_indices, orientations, probabilities
                )
            )

    if distributed:
        gathered: list[Any] = [None] * world_size
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
    overall_ap = float(average_precision_score(targets, scores))
    return ValidationResult(
        macro_average_precision=float(per_category.mean()),
        overall_average_precision=overall_ap,
        per_category_average_precision={
            str(key): float(value) for key, value in per_category.items()
        },
        scores=scores,
        scores_ab=scores_ab,
        scores_ba=scores_ba,
    )


def build_validation_predictions(
    validation: pd.DataFrame,
    validation_cache: PairTokenCache,
    result: ValidationResult,
    max_length: int,
) -> pd.DataFrame:
    """Combine validation metadata and model scores into an analysis-ready table."""
    columns = [
        "id1",
        "id2",
        "target",
        "category_1",
        "category_2",
        "product_text_1",
        "product_text_2",
    ]
    missing = [column for column in columns if column not in validation]
    if missing:
        raise ValueError(f"Validation metadata is missing columns: {missing}")
    if len(validation) != len(result.scores):
        raise ValueError("Validation metadata and prediction lengths differ")

    predictions = validation[columns].reset_index(drop=True).copy()
    predictions.insert(0, "pair_index", np.arange(len(predictions), dtype=np.int64))
    predictions["score"] = result.scores
    predictions["score_ab"] = result.scores_ab
    predictions["score_ba"] = result.scores_ba
    predictions["score_order_gap"] = np.abs(result.scores_ab - result.scores_ba)
    predictions["token_length_ab"] = validation_cache.forward_lengths.astype(np.int32)
    predictions["token_length_ba"] = validation_cache.reverse_lengths.astype(np.int32)
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
    control_group: dist.ProcessGroup | None = None
    if distributed:
        torch.cuda.set_device(local_rank)
        process_group_timeout = timedelta(hours=1)
        dist.init_process_group(backend="nccl", timeout=process_group_timeout)
        control_group = dist.new_group(
            backend="gloo",
            timeout=process_group_timeout,
        )
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    device = torch.device("cuda", local_rank)
    is_main = rank == 0
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    items = pd.read_parquet(
        args.prepared_dir / "items.parquet",
        columns=["id", "product_text", "category"],
    )
    train_pairs = pd.read_parquet(args.prepared_dir / "train_pairs.parquet")
    validation_paths = validation_split_paths(
        args.prepared_dir,
        args.validation_split,
    )
    missing_validation_files = [
        str(path) for path in validation_paths.values() if not path.is_file()
    ]
    if missing_validation_files:
        raise FileNotFoundError(
            f"Validation pair files do not exist: {missing_validation_files}"
        )
    train = attach_item_fields(
        train_pairs, items, fields=("product_text", "category")
    )
    validations = {
        name: attach_item_fields(
            pd.read_parquet(path), items, fields=("product_text", "category")
        )
        for name, path in validation_paths.items()
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
    original_train_pairs = len(train)
    if args.train_subset == "category_label_downsample":
        train = category_label_downsample(train, seed=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        raise ValueError("Cross-encoder tokenizer must define pad_token_id")
    train_cache, validation_caches = create_caches(
        train,
        validations,
        tokenizer,
        args,
        is_main,
        distributed,
        control_group,
    )
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
    lexical_similarities = (
        name_ngram_cosine(train)
        if args.lexical_hard_negative_strength > 0
        else None
    )
    training_loss_weights = build_training_loss_weights(
        train_categories,
        train_targets,
        mode=args.loss_weighting,
        lexical_similarities=lexical_similarities,
        lexical_hard_negative_strength=args.lexical_hard_negative_strength,
    )
    data_sample_weights = (
        train["sample_weight"].to_numpy(dtype=np.float32)
        if "sample_weight" in train
        else np.ones(len(train), dtype=np.float32)
    )
    if not np.isfinite(data_sample_weights).all() or (data_sample_weights <= 0).any():
        raise ValueError("Training sample_weight values must be finite and positive")
    training_loss_weights *= data_sample_weights
    training_loss_weights /= training_loss_weights.mean()
    training_source_counts = (
        {str(key): int(value) for key, value in train["label_source"].value_counts().items()}
        if "label_source" in train
        else {"unspecified": len(train)}
    )
    training_source_weight_mass = (
        {
            str(source): float(training_loss_weights[positions].sum())
            for source, positions in train.groupby("label_source").indices.items()
        }
        if "label_source" in train
        else {"unspecified": float(training_loss_weights.sum())}
    )
    sampling_weights = balanced_sampling_weights(
        train_categories, train_targets, args.sampling
    )
    train_dataset = CrossEncoderPairDataset(
        train_cache,
        train_targets,
        sample_weights=training_loss_weights,
    )
    validation_datasets = {
        name: CrossEncoderPairDataset(
            validation_caches[name], validation_targets[name]
        )
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
    collator = CrossEncoderBatchCollator(tokenizer.pad_token_id)
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
    del train, items

    amp_dtype = preferred_cuda_dtype()
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=1,
        attn_implementation=args.attention_implementation,
    )
    model.config.id2label = {0: "MATCH_SCORE"}
    model.config.label2id = {"MATCH_SCORE": 0}
    model = model.to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    training_model: torch.nn.Module = model
    if distributed:
        training_model = DistributedDataParallel(
            training_model,
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
    optimizer = AdamW(
        [
            {"params": decay_parameters, "weight_decay": args.weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        max(1, int(total_updates * args.warmup_ratio)),
        total_updates,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    trainable_parameters = [parameter for _, parameter in named_parameters]

    if is_main:
        print(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(device),
                    "world_size": world_size,
                    "model": args.model,
                    "architecture": type(model).__name__,
                    "amp_dtype": str(amp_dtype),
                    "trainable_parameters": sum(p.numel() for p in trainable_parameters),
                    "train_pairs": len(train_dataset),
                    "original_train_pairs": original_train_pairs,
                    "train_subset": args.train_subset,
                    "validation_pairs": {
                        name: len(dataset)
                        for name, dataset in validation_datasets.items()
                    },
                    "steps_per_epoch": len(train_loader),
                    "per_device_batch": args.batch_size,
                    "effective_batch": args.batch_size
                    * world_size
                    * args.gradient_accumulation,
                    "dataloader_workers_total": args.dataloader_workers * world_size,
                    "validation_schedule": "after_training",
                    "sampling": args.sampling,
                    "loss_weighting": args.loss_weighting,
                    "sampler_unique_coverage_per_epoch": (
                        1.0 if args.sampling == "none" else None
                    ),
                    "loss_weight_min": float(training_loss_weights.min()),
                    "loss_weight_median": float(np.median(training_loss_weights)),
                    "loss_weight_max": float(training_loss_weights.max()),
                    "training_source_counts": training_source_counts,
                    "training_source_weight_mass": training_source_weight_mass,
                    "weighted_positive_fraction": float(
                        training_loss_weights[train_targets >= 0.5].sum()
                        / training_loss_weights.sum()
                    ),
                    "negative_name_ngram_cosine_quantiles": (
                        {
                            str(quantile): float(value)
                            for quantile, value in zip(
                                (0.5, 0.9, 0.99),
                                np.quantile(
                                    lexical_similarities[train_targets < 0.5],
                                    (0.5, 0.9, 0.99),
                                ),
                            )
                        }
                        if lexical_similarities is not None
                        else None
                    ),
                }
            ),
            flush=True,
        )

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
        interval_steps = interval_examples = 0
        interval_useful_tokens = interval_padded_tokens = 0
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
            if args.label_smoothing:
                targets = targets * (1 - args.label_smoothing) + 0.5 * args.label_smoothing
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
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = training_model(**batch).logits.squeeze(-1)
                    per_example_loss = F.binary_cross_entropy_with_logits(
                        logits.float(), targets, reduction="none"
                    )
                    raw_loss = (per_example_loss * weights).sum() / weights.sum()
                    loss = raw_loss / group_size
                scaler.scale(loss).backward()

            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, args.max_grad_norm
                )
                scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # FP16 GradScaler may skip its first overflowing optimizer step.
                # Advance the LR schedule only when optimizer.step actually ran.
                if scaler.get_scale() >= scale_before_step:
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
                            "examples_per_second": interval_examples
                            * world_size
                            / interval_seconds,
                            "seconds_per_step": seconds_per_step,
                            "epoch_eta_minutes": (len(train_loader) - step - 1)
                            * seconds_per_step
                            / 60,
                            "data_wait_fraction_approx": interval_data_seconds
                            / interval_seconds,
                            "padding_efficiency": interval_useful_tokens
                            / interval_padded_tokens,
                            "peak_vram_gib": torch.cuda.max_memory_allocated(device)
                            / 2**30,
                            "learning_rate": scheduler.get_last_lr()[0],
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
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)

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
            amp_dtype,
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
        primary_name = "iid" if "iid" in validation_reports else next(iter(validation_reports))
        primary = validation_reports[primary_name]
        report = {
            "training_seconds": float(elapsed.item()),
            "validation_seconds": validation_seconds,
            "validation_seconds_by_split": validation_seconds_by_split,
            "total_pipeline_seconds": time.perf_counter() - pipeline_started,
            "training_examples": int(totals[0].item()),
            "original_training_examples": original_train_pairs,
            "training_subset": args.train_subset,
            "training_sampling": args.sampling,
            "training_loss_weighting": args.loss_weighting,
            "training_unique_coverage_per_epoch": (
                1.0 if args.sampling == "none" else None
            ),
            "training_loss_weight_min": float(training_loss_weights.min()),
            "training_loss_weight_median": float(
                np.median(training_loss_weights)
            ),
            "training_loss_weight_max": float(training_loss_weights.max()),
            "training_source_counts": training_source_counts,
            "training_source_weight_mass": training_source_weight_mass,
            "primary_validation_split": primary_name,
            "validation_splits": validation_reports,
            # Compatibility aliases for older analysis tools. The canonical
            # multi-split results live under validation_splits.
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
        inference_model.save_pretrained(args.output_dir, safe_serialization=True)
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
        (args.output_dir / "training_config.json").write_text(
            args.config.read_text(encoding="utf-8"), encoding="utf-8"
        )
        print(json.dumps(report, default=str, ensure_ascii=False, indent=2), flush=True)
        print(f"Saved cross-encoder to {args.output_dir}", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
