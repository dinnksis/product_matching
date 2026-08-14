#!/usr/bin/env python3
"""Full single-GPU fine-tuning on every LLM-labelled product pair.

The default run uses all non-OOD pairs, including the ambiguous 5/9--7/9
targets, as soft labels.  OOD categories remain excluded unless
``--include-ood`` is explicitly passed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_full_data import (  # noqa: E402
    FullPairCache,
    balanced_prefix_lengths,
    build_full_pair_cache,
)
from src.early_learning_regularization import (  # noqa: E402
    binary_elr_loss,
    make_binary_elr_targets,
)
from src.minilm_serialization import DEFAULT_VARIANT, VARIANTS  # noqa: E402


DEFAULT_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
VALIDATION_SPLITS = ("iid", "hard", "ood")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream and full-fine-tune a cross-encoder on all LLM pairs"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("prepared/validation_splits_v1/llm"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/minilm_llm_full")
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("artifacts/llm_full_cache")
    )
    parser.add_argument(
        "--human-validation-dir",
        type=Path,
        default=Path("prepared/validation_splits_v1/human"),
    )
    parser.add_argument(
        "--human-items",
        type=Path,
        default=Path("data/items_human.parquet"),
        help="Raw human items with id/name/attributes/category columns",
    )
    parser.add_argument(
        "--include-ood",
        action="store_true",
        help=(
            "Also train on ood_items/ood_pairs. Do not use this while comparing "
            "models with the frozen OOD validation protocol."
        ),
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        help="Optional deterministic prefix for a smoke test; default uses every pair",
    )
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--elr-beta", type=float, default=0.7)
    parser.add_argument("--elr-lambda", type=float, default=3.0)
    parser.add_argument("--elr-epsilon", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--serialization-variant",
        choices=VARIANTS,
        default=DEFAULT_VARIANT,
        help="MiniLM ablation serialization; default is title plus key/value fields",
    )
    parser.add_argument(
        "--attribute-frequency-csv",
        type=Path,
        help=(
            "Reuse attribute_name_frequency.csv from the serialization ablation; "
            "otherwise rank keys on all referenced training items"
        ),
    )
    parser.add_argument(
        "--frequent-keys-json",
        type=Path,
        help="frequent_attribute_names.json; required only for S3_HYBRID",
    )
    parser.add_argument(
        "--attention-implementation", choices=["sdpa", "eager"], default="sdpa"
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")

    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--bucket-size-multiplier", type=int, default=100)
    parser.add_argument("--pad-to-multiple-of", type=int, default=8)
    parser.add_argument("--item-tokenization-batch-size", type=int, default=8192)
    parser.add_argument("--pair-cache-batch-size", type=int, default=1_000_000)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--best-validation-split",
        choices=VALIDATION_SPLITS,
        default="iid",
        help="Split whose macro AP updates checkpoint-best; default avoids OOD tuning",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--save-every-updates",
        type=int,
        default=5000,
        help="Overwrite checkpoint-last every N optimizer updates; zero disables it",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume exactly from OUTPUT_DIR/checkpoint-last",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "eval-batch-size": args.eval_batch_size,
        "gradient-accumulation": args.gradient_accumulation,
        "learning-rate": args.learning_rate,
        "max-grad-norm": args.max_grad_norm,
        "max-length": args.max_length,
        "prefetch-factor": args.prefetch_factor,
        "bucket-size-multiplier": args.bucket_size_multiplier,
        "pad-to-multiple-of": args.pad_to_multiple_of,
        "item-tokenization-batch-size": args.item_tokenization_batch_size,
        "pair-cache-batch-size": args.pair_cache_batch_size,
        "log-every": args.log_every,
    }
    if invalid := [name for name, value in positive.items() if value <= 0]:
        raise ValueError(f"These arguments must be positive: {', '.join(invalid)}")
    if args.num_workers < 0 or args.save_every_updates < 0:
        raise ValueError("num-workers and save-every-updates must be non-negative")
    if args.weight_decay < 0:
        raise ValueError("weight-decay must be non-negative")
    if not 0 <= args.elr_beta < 1:
        raise ValueError("elr-beta must be in [0, 1)")
    if args.elr_lambda < 0:
        raise ValueError("elr-lambda must be non-negative")
    if not 0 < args.elr_epsilon < 0.5:
        raise ValueError("elr-epsilon must be in (0, 0.5)")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup-ratio must be in [0, 1)")
    if args.max_pairs is not None and args.max_pairs <= 0:
        raise ValueError("max-pairs must be positive")
    if args.serialization_variant == "S3_HYBRID" and args.frequent_keys_json is None:
        raise ValueError("S3_HYBRID requires --frequent-keys-json")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def json_line(payload: dict[str, Any]) -> None:
    print(json.dumps(json_safe(payload), ensure_ascii=False), flush=True)


def data_paths(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    item_paths = [args.data_dir / "non_ood_items.parquet"]
    pair_paths = [args.data_dir / "non_ood_pairs.parquet"]
    if args.include_ood:
        item_paths.append(args.data_dir / "ood_items.parquet")
        pair_paths.append(args.data_dir / "ood_pairs.parquet")
    return item_paths, pair_paths


def validation_pair_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        name: args.human_validation_dir / f"{name}_validation_pairs.parquet"
        for name in VALIDATION_SPLITS
    }


@dataclass(frozen=True)
class PairTemplate:
    prefix: np.ndarray
    middle: np.ndarray
    suffix: np.ndarray
    uses_token_type_ids: bool

    @property
    def special_tokens(self) -> int:
        return len(self.prefix) + len(self.middle) + len(self.suffix)


def infer_pair_template(tokenizer: Any) -> PairTemplate:
    """Infer where a tokenizer inserts pair special tokens."""
    first_sentinel, second_sentinel = -1_000_000_001, -1_000_000_002
    built = list(
        tokenizer.build_inputs_with_special_tokens(
            [first_sentinel], [second_sentinel]
        )
    )
    if built.count(first_sentinel) != 1 or built.count(second_sentinel) != 1:
        raise ValueError("Could not infer tokenizer pair special-token template")
    first_position = built.index(first_sentinel)
    second_position = built.index(second_sentinel)
    if first_position >= second_position:
        raise ValueError("Tokenizer reordered pair sequences unexpectedly")
    template = PairTemplate(
        prefix=np.asarray(built[:first_position], dtype=np.int64),
        middle=np.asarray(built[first_position + 1 : second_position], dtype=np.int64),
        suffix=np.asarray(built[second_position + 1 :], dtype=np.int64),
        uses_token_type_ids="token_type_ids" in tokenizer.model_input_names,
    )
    expected = tokenizer.num_special_tokens_to_add(pair=True)
    if template.special_tokens != expected:
        raise ValueError(
            f"Inferred {template.special_tokens} pair special tokens, expected {expected}"
        )
    return template


class FullPairDataset:
    """Lazy mmap dataset that is safe to reopen in DataLoader workers."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self._cache: FullPairCache | None = None

    def _get_cache(self) -> FullPairCache:
        if self._cache is None:
            self._cache = FullPairCache.load(self.cache_dir)
        return self._cache

    def __len__(self) -> int:
        return self._get_cache().pair_count

    def __getitem__(self, key: int | tuple[int, bool]) -> dict[str, Any]:
        if isinstance(key, tuple):
            index, reverse = key
        else:
            index, reverse = key, False
        cache = self._get_cache()
        left = int(cache.left_positions[index])
        right = int(cache.right_positions[index])
        if reverse:
            left, right = right, left
        return {
            "first": cache.tokens_for_item(left),
            "second": cache.tokens_for_item(right),
            "target": float(cache.targets[index]),
            "pair_index": index,
            "reverse": reverse,
        }

    def __getstate__(self) -> dict[str, Any]:
        return {"cache_dir": self.cache_dir, "_cache": None}


class BucketBatchSampler:
    """One-pass shuffled batches with local length bucketing and resumability."""

    def __init__(
        self,
        lengths: Sequence[int],
        *,
        batch_size: int,
        bucket_size_multiplier: int,
        seed: int,
    ) -> None:
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.bucket_size = batch_size * bucket_size_multiplier
        self.seed = seed
        self.epoch = 0
        self.start_batch = 0

    @property
    def total_batches(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)

    def set_epoch(self, epoch: int, *, start_batch: int = 0) -> None:
        if not 0 <= start_batch <= self.total_batches:
            raise ValueError("start_batch is outside the epoch")
        self.epoch = epoch
        self.start_batch = start_batch

    def __len__(self) -> int:
        return self.total_batches - self.start_batch

    @staticmethod
    def _reverse_orientation(indices: np.ndarray, seed: int) -> np.ndarray:
        # SplitMix64 gives a stable, well-mixed bit without storing 10M booleans.
        values = indices.astype(np.uint64, copy=False) + np.uint64(seed)
        values += np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        values = (values ^ (values >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
        values ^= values >> np.uint64(31)
        return (values & np.uint64(1)).astype(bool)

    def __iter__(self) -> Iterator[list[tuple[int, bool]]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        indices = rng.permutation(len(self.lengths))
        batch_number = 0
        orientation_seed = self.seed + self.epoch * 1_000_003
        for bucket_start in range(0, len(indices), self.bucket_size):
            bucket = indices[bucket_start : bucket_start + self.bucket_size]
            order = np.argsort(self.lengths[bucket], kind="stable")
            bucket = bucket[order]
            for start in range(0, len(bucket), self.batch_size):
                batch = bucket[start : start + self.batch_size]
                if batch_number >= self.start_batch:
                    orientations = self._reverse_orientation(batch, orientation_seed)
                    yield [
                        (int(index), bool(reverse))
                        for index, reverse in zip(batch, orientations)
                    ]
                batch_number += 1


class SymmetricEvaluationBatchSampler:
    """Length-bucket every validation pair in both A/B orientations."""

    def __init__(self, lengths: Sequence[int], *, batch_size: int) -> None:
        if batch_size < 2:
            raise ValueError("eval-batch-size must be at least 2")
        self.lengths = np.asarray(lengths)
        self.pairs_per_batch = max(1, batch_size // 2)

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.pairs_per_batch)

    def __iter__(self) -> Iterator[list[tuple[int, bool]]]:
        indices = np.argsort(self.lengths, kind="stable")
        for start in range(0, len(indices), self.pairs_per_batch):
            batch = indices[start : start + self.pairs_per_batch]
            yield [
                (int(index), reverse)
                for index in batch
                for reverse in (False, True)
            ]


@dataclass(frozen=True)
class PairCollator:
    template: PairTemplate
    pad_token_id: int
    max_length: int
    pad_to_multiple_of: int
    padding_side: str = "right"
    tokenizer: Any | None = None

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        pair_budget = self.max_length - self.template.special_tokens
        sequences: list[np.ndarray] = []
        token_types: list[np.ndarray] = []
        for row in rows:
            first = np.asarray(row["first"], dtype=np.int64)
            second = np.asarray(row["second"], dtype=np.int64)
            first_keep, second_keep = balanced_prefix_lengths(
                len(first), len(second), pair_budget
            )
            first = first[:first_keep]
            second = second[:second_keep]
            sequence = np.concatenate(
                (
                    self.template.prefix,
                    first,
                    self.template.middle,
                    second,
                    self.template.suffix,
                )
            )
            sequences.append(sequence)
            if self.template.uses_token_type_ids:
                if self.tokenizer is None:
                    raise RuntimeError("Tokenizer is required for token_type_ids")
                token_types.append(
                    np.asarray(
                        self.tokenizer.create_token_type_ids_from_sequences(
                            first.tolist(), second.tolist()
                        ),
                        dtype=np.int64,
                    )
                )

        maximum = max(map(len, sequences))
        padded_length = min(
            self.max_length,
            math.ceil(maximum / self.pad_to_multiple_of) * self.pad_to_multiple_of,
        )
        input_ids = torch.full(
            (len(rows), padded_length), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros(
            (len(rows), padded_length), dtype=torch.long
        )
        token_type_ids = (
            torch.zeros((len(rows), padded_length), dtype=torch.long)
            if token_types
            else None
        )
        for row_index, sequence in enumerate(sequences):
            start = 0 if self.padding_side == "right" else padded_length - len(sequence)
            stop = start + len(sequence)
            input_ids[row_index, start:stop] = torch.from_numpy(sequence)
            attention_mask[row_index, start:stop] = 1
            if token_type_ids is not None:
                token_type_ids[row_index, start:stop] = torch.from_numpy(
                    token_types[row_index]
                )

        batch: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": torch.tensor(
                [row["target"] for row in rows], dtype=torch.float32
            ),
            "pair_indices": torch.tensor(
                [row["pair_index"] for row in rows], dtype=torch.long
            ),
            "orientations": torch.tensor(
                [row["reverse"] for row in rows], dtype=torch.bool
            ),
        }
        if token_type_ids is not None:
            batch["token_type_ids"] = token_type_ids
        return batch


def loader_options(args: argparse.Namespace) -> dict[str, Any]:
    options: dict[str, Any] = {
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers:
        options.update(
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
        )
    return options


def load_human_validation_frames(
    human_items: Path,
    pair_paths: dict[str, Path],
) -> dict[str, Any]:
    """Load the small human pair tables and attach trusted categories."""
    import pandas as pd

    item_frame = pd.read_parquet(human_items, columns=["id", "category"])
    category_by_id = item_frame.set_index("id", verify_integrity=True)["category"]
    validations: dict[str, Any] = {}
    for name, path in pair_paths.items():
        frame = pd.read_parquet(path, columns=["id1", "id2", "target"])
        if not frame["target"].isin([0.0, 1.0]).all():
            raise ValueError(f"Human validation {name!r} contains non-binary labels")
        first_categories = frame["id1"].map(category_by_id)
        second_categories = frame["id2"].map(category_by_id)
        if first_categories.isna().any() or second_categories.isna().any():
            raise ValueError(f"Human validation {name!r} references missing items")
        if not first_categories.eq(second_categories).all():
            raise ValueError(f"Human validation {name!r} has cross-category pairs")
        frame = frame.copy()
        frame["category"] = first_categories.astype(str)
        validations[name] = frame
    return validations


def evaluate_human_validation(
    *,
    model: Any,
    loader: Any,
    cache: FullPairCache,
    validations: dict[str, Any],
    device: Any,
    amp_dtype: Any,
    epoch: int,
    output_dir: Path,
    max_length: int,
) -> dict[str, Any]:
    """Evaluate IID/hard/OOD in both orientations and persist predictions."""
    import pandas as pd
    import torch
    from sklearn.metrics import average_precision_score

    started = time.perf_counter()
    scores_ab = np.full(cache.pair_count, np.nan, dtype=np.float32)
    scores_ba = np.full(cache.pair_count, np.nan, dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for packed in loader:
            pair_indices = packed.pop("pair_indices").numpy()
            orientations = packed.pop("orientations").numpy()
            packed.pop("targets")
            model_inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in packed.items()
            }
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(**model_inputs).logits.squeeze(-1)
            probabilities = logits.float().sigmoid().cpu().numpy()
            for index, reverse, score in zip(
                pair_indices, orientations, probabilities
            ):
                destination = scores_ba if bool(reverse) else scores_ab
                pair_index = int(index)
                if np.isfinite(destination[pair_index]):
                    raise RuntimeError(
                        f"Duplicate validation prediction for pair {pair_index}"
                    )
                destination[pair_index] = float(score)

    if not np.isfinite(scores_ab).all() or not np.isfinite(scores_ba).all():
        raise RuntimeError("Validation did not produce both orientations for every pair")

    epoch_dir = output_dir / "validation" / f"epoch-{epoch:02d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    split_reports: dict[str, Any] = {}
    offset = 0
    for name, frame in validations.items():
        stop = offset + len(frame)
        targets = frame["target"].to_numpy(dtype=np.float32)
        split_ab = scores_ab[offset:stop]
        split_ba = scores_ba[offset:stop]
        scores = (split_ab + split_ba) / 2.0
        per_category: dict[str, float] = {}
        categories = frame["category"].to_numpy(dtype=str)
        for category in sorted(set(categories)):
            selected = categories == category
            per_category[category] = float(
                average_precision_score(targets[selected], scores[selected])
            )
        overall_ap = float(average_precision_score(targets, scores))
        macro_ap = float(np.mean(list(per_category.values())))
        predictions = pd.DataFrame(
            {
                "pair_index": np.arange(len(frame), dtype=np.int64),
                "id1": frame["id1"].to_numpy(dtype=np.int64),
                "id2": frame["id2"].to_numpy(dtype=np.int64),
                "target": targets,
                "category": categories,
                "score": scores,
                "score_ab": split_ab,
                "score_ba": split_ba,
                "score_order_gap": np.abs(split_ab - split_ba),
                "token_length": np.asarray(cache.pair_lengths[offset:stop]),
            }
        )
        predictions["reached_max_length"] = (
            predictions["token_length"] >= max_length
        )
        predictions_file = epoch_dir / f"{name}_predictions.parquet"
        predictions.to_parquet(predictions_file, index=False, compression="zstd")
        split_reports[name] = {
            "examples": len(frame),
            "positive_examples": int(targets.sum()),
            "positive_rate": float(targets.mean()),
            "macro_average_precision": macro_ap,
            "overall_average_precision": overall_ap,
            "per_category_average_precision": per_category,
            "mean_score_order_gap": float(predictions["score_order_gap"].mean()),
            "reached_max_length_rate": float(
                predictions["reached_max_length"].mean()
            ),
            "predictions_file": str(predictions_file),
        }
        offset = stop
    if offset != cache.pair_count:
        raise RuntimeError(
            f"Validation metadata has {offset} pairs, cache has {cache.pair_count}"
        )
    report = {
        "epoch": epoch,
        "elapsed_seconds": time.perf_counter() - started,
        "splits": split_reports,
    }
    write_json(epoch_dir / "metrics.json", report)
    json_line({"validation": report})
    return report


def model_for_saving(model: Any) -> Any:
    return getattr(model, "_orig_mod", model)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_checkpoint(
    *,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    elr_targets: Any,
    output_dir: Path,
    state: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    import torch

    checkpoint = output_dir / "checkpoint-last"
    staging = output_dir / ".checkpoint-last.tmp"
    previous = output_dir / ".checkpoint-last.previous"
    for path in (staging, previous):
        if path.exists():
            shutil.rmtree(path)
    staging.mkdir(parents=True)
    model_for_saving(model).save_pretrained(staging, safe_serialization=True)
    tokenizer.save_pretrained(staging)
    torch.save(optimizer.state_dict(), staging / "optimizer.pt")
    torch.save(scheduler.state_dict(), staging / "scheduler.pt")
    torch.save(elr_targets.detach().cpu(), staging / "elr_targets.pt")
    torch.save(
        {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
        staging / "rng_state.pt",
    )
    write_json(staging / "training_state.json", state)
    write_json(staging / "training_args.json", vars(args))
    if checkpoint.exists():
        os.replace(checkpoint, previous)
    os.replace(staging, checkpoint)
    if previous.exists():
        shutil.rmtree(previous)
    json_line({"checkpoint_saved": str(checkpoint), **state})


def snapshot_checkpoint(
    source: Path,
    destination: Path,
    *,
    replace: bool,
) -> None:
    """Atomically retain a checkpoint using hard links instead of another copy."""
    staging = destination.parent / f".{destination.name}.tmp"
    previous = destination.parent / f".{destination.name}.previous"
    for path in (staging, previous):
        if path.exists():
            shutil.rmtree(path)
    if destination.exists() and not replace:
        raise FileExistsError(f"Checkpoint snapshot already exists: {destination}")
    copied_files = 0

    def link_or_copy(source_file: str, destination_file: str) -> str:
        nonlocal copied_files
        try:
            os.link(source_file, destination_file)
            return destination_file
        except OSError:
            copied_files += 1
            return shutil.copy2(source_file, destination_file)

    shutil.copytree(source, staging, copy_function=link_or_copy)
    if destination.exists():
        os.replace(destination, previous)
    os.replace(staging, destination)
    if previous.exists():
        shutil.rmtree(previous)
    json_line(
        {
            "checkpoint_snapshot": str(destination),
            "source": str(source),
            "files_copied_without_hardlinks": copied_files,
        }
    )


def resume_state(output_dir: Path) -> tuple[Path, dict[str, Any]]:
    checkpoint = output_dir / "checkpoint-last"
    state_path = checkpoint / "training_state.json"
    if not checkpoint.is_dir() or not state_path.is_file():
        raise FileNotFoundError(
            f"--resume requested but checkpoint is missing: {checkpoint}"
        )
    return checkpoint, json.loads(state_path.read_text(encoding="utf-8"))


def validate_resume_args(args: argparse.Namespace, checkpoint: Path) -> None:
    saved_path = checkpoint / "training_args.json"
    if not saved_path.is_file():
        raise FileNotFoundError(f"Checkpoint training args are missing: {saved_path}")
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    current = json_safe(vars(args))
    immutable = (
        "model",
        "include_ood",
        "max_pairs",
        "human_validation_dir",
        "human_items",
        "epochs",
        "batch_size",
        "eval_batch_size",
        "gradient_accumulation",
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "max_grad_norm",
        "elr_beta",
        "elr_lambda",
        "elr_epsilon",
        "max_length",
        "serialization_variant",
        "attribute_frequency_csv",
        "frequent_keys_json",
        "attention_implementation",
        "gradient_checkpointing",
        "torch_compile",
        "amp_dtype",
        "bucket_size_multiplier",
        "pad_to_multiple_of",
        "best_validation_split",
        "seed",
    )
    changed = {
        name: {"checkpoint": saved.get(name), "current": current.get(name)}
        for name in immutable
        if saved.get(name) != current.get(name)
    }
    if changed:
        raise ValueError(
            "Resume requires the same data/model/optimizer schedule; changed args: "
            + json.dumps(changed, ensure_ascii=False, sort_keys=True)
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    item_paths, pair_paths = data_paths(args)
    human_pair_paths = validation_pair_paths(args)
    missing_inputs = [
        str(path)
        for path in (*item_paths, *pair_paths, args.human_items, *human_pair_paths.values())
        if not path.is_file()
    ]
    if missing_inputs:
        raise FileNotFoundError(
            f"Required training/validation files are missing: {missing_inputs}"
        )
    checkpoint_path: Path | None = None
    saved_state: dict[str, Any] | None = None
    if args.resume:
        checkpoint_path, saved_state = resume_state(args.output_dir)
        validate_resume_args(args, checkpoint_path)
    elif not args.cache_only and (
        any(
            (args.output_dir / name).exists()
            for name in ("checkpoint-last", "checkpoint-best", "model")
        )
        or any(args.output_dir.glob("checkpoint-epoch-*"))
    ):
        raise FileExistsError(
            f"Training output already exists in {args.output_dir}; "
            "pass --resume or choose another --output-dir"
        )

    # Imports stay below --help/argument validation so the script can report a
    # useful error before GPU-only dependencies are installed.
    try:
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_cosine_schedule_with_warmup,
        )
    except ImportError as error:
        raise SystemExit(
            "Install the server dependencies first: "
            "pip install -r requirements-cross-encoder.txt"
        ) from error

    tokenizer_source = checkpoint_path or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id")
    tokenizer_max_length = int(tokenizer.model_max_length)
    if tokenizer_max_length < 1_000_000 and args.max_length > tokenizer_max_length:
        raise ValueError(
            f"--max-length {args.max_length} exceeds tokenizer/model limit "
            f"{tokenizer_max_length}"
        )
    template = infer_pair_template(tokenizer)
    if template.special_tokens != tokenizer.num_special_tokens_to_add(pair=True):
        raise RuntimeError("Tokenizer pair template is inconsistent")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    cache = build_full_pair_cache(
        item_paths=item_paths,
        pair_paths=pair_paths,
        tokenizer=tokenizer,
        model_name=args.model,
        cache_root=args.cache_dir,
        max_length=args.max_length,
        serialization_variant=args.serialization_variant,
        attribute_frequency_csv=args.attribute_frequency_csv,
        frequent_keys_json=args.frequent_keys_json,
        item_batch_size=args.item_tokenization_batch_size,
        pair_batch_size=args.pair_cache_batch_size,
        max_pairs=args.max_pairs,
        rebuild=args.rebuild_cache,
    )
    validation_cache = build_full_pair_cache(
        item_paths=[args.human_items],
        pair_paths=list(human_pair_paths.values()),
        tokenizer=tokenizer,
        model_name=args.model,
        cache_root=args.cache_dir / "human_validation",
        max_length=args.max_length,
        serialization_variant=args.serialization_variant,
        attribute_frequency_csv=cache.directory / "attribute_name_frequency.csv",
        frequent_keys_json=cache.directory / "frequent_attribute_names.json",
        item_batch_size=args.item_tokenization_batch_size,
        pair_batch_size=args.pair_cache_batch_size,
        max_pairs=None,
        rebuild=args.rebuild_cache,
    )
    if args.cache_only:
        json_line(
            {
                "status": "cache_only_complete",
                "training_cache": cache.directory,
                "training_items": cache.item_count,
                "training_pairs": cache.pair_count,
                "validation_cache": validation_cache.directory,
                "validation_items": validation_cache.item_count,
                "validation_pairs": validation_cache.pair_count,
            }
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full LLM fine-tuning")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"This entry point expects exactly one visible GPU, found {torch.cuda.device_count()}"
        )
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU does not support bf16; pass --amp-dtype fp16")

    dataset = FullPairDataset(cache.directory)
    validation_dataset = FullPairDataset(validation_cache.directory)
    validations = load_human_validation_frames(args.human_items, human_pair_paths)
    expected_validation_pairs = sum(len(frame) for frame in validations.values())
    if expected_validation_pairs != validation_cache.pair_count:
        raise RuntimeError(
            f"Human validation has {expected_validation_pairs} metadata rows but "
            f"the cache has {validation_cache.pair_count} pairs"
        )
    validation_offset = 0
    for name, frame in validations.items():
        validation_stop = validation_offset + len(frame)
        if not np.array_equal(
            np.asarray(validation_cache.targets[validation_offset:validation_stop]),
            frame["target"].to_numpy(dtype=np.float32),
        ):
            raise RuntimeError(f"Validation cache targets are misaligned for {name!r}")
        validation_offset = validation_stop
    sampler = BucketBatchSampler(
        cache.pair_lengths,
        batch_size=args.batch_size,
        bucket_size_multiplier=args.bucket_size_multiplier,
        seed=args.seed,
    )
    collator = PairCollator(
        template=template,
        pad_token_id=tokenizer.pad_token_id,
        max_length=args.max_length,
        pad_to_multiple_of=args.pad_to_multiple_of,
        padding_side=tokenizer.padding_side,
        tokenizer=tokenizer if template.uses_token_type_ids else None,
    )
    validation_sampler = SymmetricEvaluationBatchSampler(
        validation_cache.pair_lengths,
        batch_size=args.eval_batch_size,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=validation_sampler,
        collate_fn=collator,
        **loader_options(args),
    )

    model_source = checkpoint_path or args.model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        num_labels=1,
        attn_implementation=args.attention_implementation,
    )
    model.config.id2label = {0: "MATCH_SCORE"}
    model.config.label2id = {"MATCH_SCORE": 0}
    model.to(device)
    elr_targets = make_binary_elr_targets(cache.pair_count, device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    decay_parameters = []
    no_decay_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2 and "layer_norm" not in name.lower():
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)
    optimizer_groups = [
        {"params": decay_parameters, "weight_decay": args.weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]
    try:
        optimizer = AdamW(optimizer_groups, lr=args.learning_rate, fused=True)
    except TypeError:
        optimizer = AdamW(optimizer_groups, lr=args.learning_rate)

    total_batches = sampler.total_batches
    updates_per_epoch = math.ceil(total_batches / args.gradient_accumulation)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_updates * args.warmup_ratio)),
        num_training_steps=total_updates,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    start_epoch = 0
    start_batch = 0
    global_update = 0
    examples_seen = 0
    restored_rng_state: dict[str, Any] | None = None
    if saved_state is not None:
        assert checkpoint_path is not None
        if saved_state.get("cache_fingerprint") != cache.metadata["fingerprint"]:
            raise ValueError("Checkpoint was created from a different data cache")
        if (
            saved_state.get("validation_cache_fingerprint")
            != validation_cache.metadata["fingerprint"]
        ):
            raise ValueError("Checkpoint was created from a different validation cache")
        start_epoch = int(saved_state["epoch"])
        start_batch = int(saved_state["next_batch"])
        global_update = int(saved_state["global_update"])
        examples_seen = int(saved_state["examples_seen"])
        optimizer.load_state_dict(
            torch.load(
                checkpoint_path / "optimizer.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        scheduler.load_state_dict(
            torch.load(
                checkpoint_path / "scheduler.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        restored_rng_state = torch.load(
            checkpoint_path / "rng_state.pt",
            map_location="cpu",
            weights_only=True,
        )
        elr_path = checkpoint_path / "elr_targets.pt"
        if not elr_path.is_file():
            raise FileNotFoundError(f"Checkpoint ELR history is missing: {elr_path}")
        restored_elr_targets = torch.load(
            elr_path,
            map_location=device,
            weights_only=True,
        )
        if restored_elr_targets.shape != elr_targets.shape:
            raise ValueError(
                "Checkpoint ELR history has shape "
                f"{tuple(restored_elr_targets.shape)}, expected {tuple(elr_targets.shape)}"
            )
        elr_targets.copy_(restored_elr_targets)
        json_line({"resumed_from": checkpoint_path, **saved_state})

    if args.torch_compile:
        model = torch.compile(model, dynamic=True)
    if restored_rng_state is not None:
        torch.set_rng_state(restored_rng_state["cpu"])
        torch.cuda.set_rng_state_all(restored_rng_state["cuda"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_history_path = args.output_dir / "validation_history.json"
    if saved_state is not None:
        validation_history = saved_state.get("validation_history", [])
        if not isinstance(validation_history, list):
            raise ValueError("Checkpoint validation_history must contain a list")
    else:
        validation_history = []
    if len(validation_history) != start_epoch:
        raise ValueError(
            "Validation history and checkpoint disagree: "
            f"history has {len(validation_history)} completed epochs, "
            f"checkpoint starts at epoch {start_epoch}"
        )
    write_json(validation_history_path, validation_history)
    best_score = max(
        (
            float(entry["splits"][args.best_validation_split]["macro_average_precision"])
            for entry in validation_history
        ),
        default=-math.inf,
    )
    best_epoch = next(
        (
            int(entry["epoch"])
            for entry in validation_history
            if float(
                entry["splits"][args.best_validation_split][
                    "macro_average_precision"
                ]
            )
            == best_score
        ),
        None,
    )
    best_checkpoint_path = args.output_dir / "checkpoint-best"
    if args.resume and start_epoch > 0 and start_batch == 0:
        completed_epoch_path = (
            args.output_dir / f"checkpoint-epoch-{start_epoch:02d}"
        )
        if not completed_epoch_path.exists():
            snapshot_checkpoint(
                args.output_dir / "checkpoint-last",
                completed_epoch_path,
                replace=False,
            )
    if args.resume and best_epoch is not None and not best_checkpoint_path.exists():
        epoch_source = args.output_dir / f"checkpoint-epoch-{best_epoch:02d}"
        if not epoch_source.exists() and best_epoch == start_epoch:
            epoch_source = args.output_dir / "checkpoint-last"
        if not epoch_source.exists():
            raise FileNotFoundError(
                f"Cannot reconstruct missing best checkpoint for epoch {best_epoch}"
            )
        snapshot_checkpoint(epoch_source, best_checkpoint_path, replace=False)
    write_json(args.output_dir / "training_args.json", vars(args))
    json_line(
        {
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": torch.cuda.get_device_capability(device),
            "model": args.model,
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "pairs": cache.pair_count,
            "items": cache.item_count,
            "all_soft_targets_retained": True,
            "include_ood": args.include_ood,
            "serialization": cache.metadata["serialization"],
            "max_length": args.max_length,
            "target_counts": cache.metadata["pairs"]["target_counts"],
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "steps_per_epoch": total_batches,
            "updates_per_epoch": updates_per_epoch,
            "elr": {
                "beta": args.elr_beta,
                "lambda": args.elr_lambda,
                "epsilon": args.elr_epsilon,
                "target_history_mib": elr_targets.numel()
                * elr_targets.element_size()
                / 2**20,
            },
            "validation_pairs": {
                name: len(frame) for name, frame in validations.items()
            },
            "validation_cache": validation_cache.directory,
            "best_validation_split": args.best_validation_split,
            "amp_dtype": str(amp_dtype),
            "cache": cache.directory,
        }
    )

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer.zero_grad(set_to_none=True)
    training_started = time.perf_counter()
    interval_started = training_started
    interval_loss = 0.0
    interval_supervised_loss = 0.0
    interval_elr_regularizer = 0.0
    interval_elr_agreement = 0.0
    interval_batches = 0
    interval_examples = 0
    interval_useful_tokens = 0
    interval_padded_tokens = 0
    completed_loss_sum = 0.0
    completed_supervised_loss_sum = 0.0
    completed_elr_regularizer_sum = 0.0
    completed_batches = 0
    run_examples = 0
    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(start_epoch, args.epochs):
        epoch_start_batch = start_batch if epoch == start_epoch else 0
        sampler.set_epoch(epoch, start_batch=epoch_start_batch)
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collator,
            **loader_options(args),
        )
        model.train()
        for batch_index, packed in enumerate(loader, start=epoch_start_batch):
            pair_indices = packed.pop("pair_indices").to(
                device, non_blocking=True
            )
            packed.pop("orientations")
            targets = packed.pop("targets").to(device, non_blocking=True)
            batch_examples = len(targets)
            useful_tokens = int(packed["attention_mask"].sum())
            padded_tokens = packed["attention_mask"].numel()
            model_inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in packed.items()
            }
            accumulation_start = (
                batch_index // args.gradient_accumulation
            ) * args.gradient_accumulation
            accumulation_size = min(
                args.gradient_accumulation, total_batches - accumulation_start
            )
            should_update = (
                (batch_index + 1) % args.gradient_accumulation == 0
                or batch_index + 1 == total_batches
            )
            sync_context = nullcontext()
            with sync_context:
                with torch.autocast("cuda", dtype=amp_dtype):
                    logits = model(**model_inputs).logits.squeeze(-1)
                    elr = binary_elr_loss(
                        logits=logits,
                        labels=targets,
                        example_indices=pair_indices,
                        target_history=elr_targets,
                        beta=args.elr_beta,
                        regularization_strength=args.elr_lambda,
                        epsilon=args.elr_epsilon,
                    )
                    loss = elr.total / accumulation_size
                scaler.scale(loss).backward()

            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, args.max_grad_norm
                )
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                    global_update += 1
                optimizer.zero_grad(set_to_none=True)

            examples_seen += batch_examples
            run_examples += batch_examples
            interval_examples += batch_examples
            interval_useful_tokens += useful_tokens
            interval_padded_tokens += padded_tokens
            loss_value = float(elr.total.detach())
            supervised_loss_value = float(elr.supervised.detach())
            elr_regularizer_value = float(elr.regularizer.detach())
            elr_agreement_value = float(elr.mean_agreement.detach())
            interval_loss += loss_value
            interval_supervised_loss += supervised_loss_value
            interval_elr_regularizer += elr_regularizer_value
            interval_elr_agreement += elr_agreement_value
            interval_batches += 1
            completed_loss_sum += loss_value
            completed_supervised_loss_sum += supervised_loss_value
            completed_elr_regularizer_sum += elr_regularizer_value
            completed_batches += 1

            if (batch_index + 1) % args.log_every == 0 or batch_index + 1 == total_batches:
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - interval_started
                json_line(
                    {
                        "epoch": epoch + 1,
                        "batch": batch_index + 1,
                        "batches": total_batches,
                        "global_update": global_update,
                        "loss": interval_loss / interval_batches,
                        "supervised_loss": interval_supervised_loss
                        / interval_batches,
                        "elr_regularizer": interval_elr_regularizer
                        / interval_batches,
                        "elr_mean_agreement": interval_elr_agreement
                        / interval_batches,
                        "examples_per_second": interval_examples / elapsed,
                        "padding_efficiency": (
                            interval_useful_tokens / interval_padded_tokens
                        ),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "peak_vram_gib": torch.cuda.max_memory_allocated(device)
                        / 1024**3,
                        "epoch_eta_minutes": (total_batches - batch_index - 1)
                        * elapsed
                        / interval_batches
                        / 60,
                    }
                )
                interval_started = time.perf_counter()
                interval_loss = 0.0
                interval_supervised_loss = 0.0
                interval_elr_regularizer = 0.0
                interval_elr_agreement = 0.0
                interval_batches = 0
                interval_examples = 0
                interval_useful_tokens = 0
                interval_padded_tokens = 0

            if (
                should_update
                and args.save_every_updates
                and global_update % args.save_every_updates == 0
            ):
                save_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    elr_targets=elr_targets,
                    output_dir=args.output_dir,
                    state={
                        "cache_fingerprint": cache.metadata["fingerprint"],
                        "validation_cache_fingerprint": validation_cache.metadata[
                            "fingerprint"
                        ],
                        "epoch": epoch,
                        "next_batch": batch_index + 1,
                        "global_update": global_update,
                        "examples_seen": examples_seen,
                        "validation_history": validation_history,
                        "best_validation_score": (
                            best_score if math.isfinite(best_score) else None
                        ),
                        "best_epoch": best_epoch,
                    },
                    args=args,
                )

        start_batch = 0
        validation_report = evaluate_human_validation(
            model=model,
            loader=validation_loader,
            cache=validation_cache,
            validations=validations,
            device=device,
            amp_dtype=amp_dtype,
            epoch=epoch + 1,
            output_dir=args.output_dir,
            max_length=args.max_length,
        )
        validation_report["learning_rate"] = scheduler.get_last_lr()[0]
        validation_report["global_update"] = global_update
        write_json(
            args.output_dir
            / "validation"
            / f"epoch-{epoch + 1:02d}"
            / "metrics.json",
            validation_report,
        )
        validation_history.append(validation_report)
        write_json(validation_history_path, validation_history)
        current_score = float(
            validation_report["splits"][args.best_validation_split][
                "macro_average_precision"
            ]
        )
        improved = current_score > best_score
        if improved:
            best_score = current_score
            best_epoch = epoch + 1
        save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            elr_targets=elr_targets,
            output_dir=args.output_dir,
            state={
                "cache_fingerprint": cache.metadata["fingerprint"],
                "validation_cache_fingerprint": validation_cache.metadata[
                    "fingerprint"
                ],
                "epoch": epoch + 1,
                "next_batch": 0,
                "global_update": global_update,
                "examples_seen": examples_seen,
                "validation_history": validation_history,
                "best_validation_score": best_score,
                "best_epoch": best_epoch,
            },
            args=args,
        )
        epoch_checkpoint = args.output_dir / f"checkpoint-epoch-{epoch + 1:02d}"
        snapshot_checkpoint(
            args.output_dir / "checkpoint-last",
            epoch_checkpoint,
            replace=False,
        )
        if improved:
            snapshot_checkpoint(
                epoch_checkpoint,
                args.output_dir / "checkpoint-best",
                replace=True,
            )
        interval_started = time.perf_counter()

    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_started
    final_dir = args.output_dir / "model"
    staging_final = args.output_dir / ".model.tmp"
    if staging_final.exists():
        shutil.rmtree(staging_final)
    staging_final.mkdir(parents=True)
    model_for_saving(model).save_pretrained(staging_final, safe_serialization=True)
    tokenizer.save_pretrained(staging_final)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    os.replace(staging_final, final_dir)
    report = {
        "status": "complete",
        "model": args.model,
        "output_model": str(final_dir),
        "cache": str(cache.directory),
        "cache_fingerprint": cache.metadata["fingerprint"],
        "validation_cache": str(validation_cache.directory),
        "validation_cache_fingerprint": validation_cache.metadata["fingerprint"],
        "training_pairs": cache.pair_count,
        "training_items": cache.item_count,
        "target_counts": cache.metadata["pairs"]["target_counts"],
        "include_ood": args.include_ood,
        "all_soft_targets_retained": True,
        "elr": {
            "beta": args.elr_beta,
            "lambda": args.elr_lambda,
            "epsilon": args.elr_epsilon,
        },
        "serialization": cache.metadata["serialization"],
        "max_length": args.max_length,
        "epochs": args.epochs,
        "examples_seen": examples_seen,
        "examples_processed_this_run": run_examples,
        "optimizer_updates": global_update,
        "training_seconds": training_seconds,
        "examples_per_second": run_examples / training_seconds,
        "mean_batch_loss": completed_loss_sum / max(1, completed_batches),
        "mean_supervised_loss": completed_supervised_loss_sum
        / max(1, completed_batches),
        "mean_elr_regularizer": completed_elr_regularizer_sum
        / max(1, completed_batches),
        "validation_history": validation_history,
        "best_validation_split": args.best_validation_split,
        "best_validation_score": best_score,
        "best_epoch": best_epoch,
        "best_checkpoint": str(args.output_dir / "checkpoint-best"),
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "args": vars(args),
    }
    write_json(args.output_dir / "training_report.json", report)
    json_line(report)


if __name__ == "__main__":
    main()
