from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from src.qwen_reranker import INSTRUCTION, PREFIX, SUFFIX


def _frame_fingerprint(frame: pd.DataFrame, configuration: dict[str, Any]) -> str:
    """Hash the texts and IDs so stale token caches cannot be reused silently."""
    digest = hashlib.sha256(
        json.dumps(configuration, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )
    columns = ["id1", "id2", "target", "product_text_1", "product_text_2"]
    row_hashes = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy()
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()[:20]


def _balanced_prefixes(
    first: Sequence[int], second: Sequence[int], budget: int
) -> tuple[Sequence[int], Sequence[int]]:
    """Keep the beginnings of both products and give unused space to the longer one."""
    if len(first) + len(second) <= budget:
        return first, second
    first_keep = min(len(first), budget // 2)
    second_keep = min(len(second), budget // 2)
    remaining = budget - first_keep - second_keep
    if remaining:
        add_first = min(len(first) - first_keep, remaining)
        first_keep += add_first
        remaining -= add_first
        second_keep += min(len(second) - second_keep, remaining)
    return first[:first_keep], second[:second_keep]


@dataclass(frozen=True)
class TokenCache:
    directory: Path
    forward_tokens: np.ndarray
    forward_offsets: np.ndarray
    reverse_tokens: np.ndarray
    reverse_offsets: np.ndarray

    @classmethod
    def load(cls, directory: Path) -> "TokenCache":
        return cls(
            directory=directory,
            forward_tokens=np.load(directory / "forward_tokens.npy", mmap_mode="r"),
            forward_offsets=np.load(directory / "forward_offsets.npy", mmap_mode="r"),
            reverse_tokens=np.load(directory / "reverse_tokens.npy", mmap_mode="r"),
            reverse_offsets=np.load(directory / "reverse_offsets.npy", mmap_mode="r"),
        )

    @property
    def size(self) -> int:
        return len(self.forward_offsets) - 1

    @property
    def forward_lengths(self) -> np.ndarray:
        return np.diff(self.forward_offsets)

    @property
    def reverse_lengths(self) -> np.ndarray:
        return np.diff(self.reverse_offsets)

    def sequence(self, index: int, reverse: bool = False) -> np.ndarray:
        tokens = self.reverse_tokens if reverse else self.forward_tokens
        offsets = self.reverse_offsets if reverse else self.forward_offsets
        start, end = int(offsets[index]), int(offsets[index + 1])
        return tokens[start:end]


def build_token_cache(
    frame: pd.DataFrame,
    tokenizer: Any,
    cache_root: Path,
    split_name: str,
    model_name: str,
    max_length: int,
    tokenization_batch_size: int = 512,
) -> TokenCache:
    """Batched-tokenize both pair orientations and store compact mmap arrays."""
    configuration = {
        "version": 2,
        "model": model_name,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_size": len(tokenizer) if hasattr(tokenizer, "__len__") else None,
        "max_length": max_length,
        "instruction": INSTRUCTION,
        "prefix": PREFIX,
        "suffix": SUFFIX,
    }
    fingerprint = _frame_fingerprint(frame, configuration)
    directory = cache_root / f"{split_name}-{fingerprint}"
    metadata_path = directory / "metadata.json"
    required = (
        metadata_path,
        directory / "forward_tokens.npy",
        directory / "forward_offsets.npy",
        directory / "reverse_tokens.npy",
        directory / "reverse_offsets.npy",
    )
    if all(path.exists() for path in required):
        cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        if cached.get("configuration") == configuration and cached.get("rows") == len(frame):
            return TokenCache.load(directory)

    directory.mkdir(parents=True, exist_ok=True)
    prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)
    query_ids = tokenizer.encode(
        f"<Instruct>: {INSTRUCTION}\n<Query>:\n", add_special_tokens=False
    )
    document_ids = tokenizer.encode("\n<Document>:\n", add_special_tokens=False)
    suffix_ids = tokenizer.encode(SUFFIX, add_special_tokens=False)
    product_budget = max_length - sum(
        map(len, (prefix_ids, query_ids, document_ids, suffix_ids))
    )
    if product_budget < 16:
        raise ValueError(
            f"max_length={max_length} leaves only {product_budget} product tokens"
        )

    forward_offsets = np.zeros(len(frame) + 1, dtype=np.int64)
    reverse_offsets = np.zeros(len(frame) + 1, dtype=np.int64)
    forward_chunks: list[np.ndarray] = []
    reverse_chunks: list[np.ndarray] = []
    forward_position = reverse_position = 0
    started = time.perf_counter()

    for start in range(0, len(frame), tokenization_batch_size):
        part = frame.iloc[start : start + tokenization_batch_size]
        first_texts = part["product_text_1"].astype(str).tolist()
        second_texts = part["product_text_2"].astype(str).tolist()
        encoded = tokenizer(
            first_texts + second_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        first_ids = encoded[: len(part)]
        second_ids = encoded[len(part) :]
        forward_sequences: list[list[int]] = []
        reverse_sequences: list[list[int]] = []

        for offset, (first, second) in enumerate(zip(first_ids, second_ids), start=1):
            kept_first, kept_second = _balanced_prefixes(first, second, product_budget)
            forward = list(
                chain(prefix_ids, query_ids, kept_first, document_ids, kept_second, suffix_ids)
            )
            kept_second, kept_first = _balanced_prefixes(second, first, product_budget)
            reverse = list(
                chain(prefix_ids, query_ids, kept_second, document_ids, kept_first, suffix_ids)
            )
            forward_sequences.append(forward)
            reverse_sequences.append(reverse)
            forward_position += len(forward)
            reverse_position += len(reverse)
            forward_offsets[start + offset] = forward_position
            reverse_offsets[start + offset] = reverse_position

        forward_chunks.append(
            np.fromiter(chain.from_iterable(forward_sequences), dtype=np.int32)
        )
        reverse_chunks.append(
            np.fromiter(chain.from_iterable(reverse_sequences), dtype=np.int32)
        )

    np.save(
        directory / "forward_tokens.npy",
        np.concatenate(forward_chunks) if forward_chunks else np.empty(0, dtype=np.int32),
    )
    np.save(directory / "forward_offsets.npy", forward_offsets)
    np.save(
        directory / "reverse_tokens.npy",
        np.concatenate(reverse_chunks) if reverse_chunks else np.empty(0, dtype=np.int32),
    )
    np.save(directory / "reverse_offsets.npy", reverse_offsets)
    metadata = {
        "configuration": configuration,
        "rows": len(frame),
        "product_token_budget": product_budget,
        "elapsed_seconds": time.perf_counter() - started,
        "forward_tokens": int(forward_position),
        "reverse_tokens": int(reverse_position),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"token_cache": str(directory), **metadata}, ensure_ascii=False))
    return TokenCache.load(directory)


class PackedPairDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        cache: TokenCache,
        targets: Sequence[float],
        sample_weights: Sequence[float] | None = None,
    ) -> None:
        if cache.size != len(targets):
            raise ValueError("Token cache and target lengths differ")
        self.cache = cache
        self.targets = np.asarray(targets, dtype=np.float32)
        self.sample_weights = (
            np.ones(len(targets), dtype=np.float32)
            if sample_weights is None
            else np.asarray(sample_weights, dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, key: int | tuple[int, bool]) -> dict[str, Any]:
        if isinstance(key, tuple):
            index, reverse = key
        else:
            index, reverse = key, False
        return {
            "input_ids": self.cache.sequence(index, reverse=reverse),
            "target": self.targets[index],
            "sample_weight": self.sample_weights[index],
            "pair_index": index,
            "reverse": reverse,
        }


@dataclass(frozen=True)
class PackedBatchCollator:
    pad_token_id: int

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_length = max(len(row["input_ids"]) for row in rows)
        input_ids = torch.full(
            (len(rows), max_length), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((len(rows), max_length), dtype=torch.long)
        for row_index, row in enumerate(rows):
            sequence = torch.tensor(row["input_ids"], dtype=torch.long)
            input_ids[row_index, -len(sequence) :] = sequence
            attention_mask[row_index, -len(sequence) :] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": torch.tensor([row["target"] for row in rows], dtype=torch.float32),
            "sample_weights": torch.tensor(
                [row["sample_weight"] for row in rows], dtype=torch.float32
            ),
            "pair_indices": torch.tensor(
                [row["pair_index"] for row in rows], dtype=torch.long
            ),
            "orientations": torch.tensor(
                [row["reverse"] for row in rows], dtype=torch.bool
            ),
        }


def balanced_sampling_weights(
    categories: Sequence[Any], targets: Sequence[float], mode: str
) -> np.ndarray | None:
    if mode == "none":
        return None
    data = pd.DataFrame(
        {"category": pd.Series(categories, dtype="string"), "target": np.asarray(targets)}
    )
    group_columns = ["category"]
    if mode == "category_label":
        data["label"] = (data["target"] >= 0.5).astype(np.int8)
        group_columns.append("label")
    elif mode != "category":
        raise ValueError(f"Unknown sampling mode: {mode}")
    counts = data.groupby(group_columns, dropna=False)["target"].transform("size")
    weights = 1.0 / counts.to_numpy(dtype=np.float64)
    return weights / weights.sum()


class LengthBucketBatchSampler(Sampler[list[tuple[int, bool]]]):
    """Balanced DDP sampling followed by local length bucketing."""

    def __init__(
        self,
        forward_lengths: Sequence[int],
        reverse_lengths: Sequence[int],
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        weights: np.ndarray | None = None,
        bucket_size_multiplier: int = 50,
        seed: int = 42,
    ) -> None:
        self.forward_lengths = np.asarray(forward_lengths)
        self.reverse_lengths = np.asarray(reverse_lengths)
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.weights = weights
        self.bucket_size = max(batch_size, batch_size * bucket_size_multiplier)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        local_size = math.ceil(len(self.forward_lengths) / self.world_size)
        return math.ceil(local_size / self.batch_size)

    def __iter__(self) -> Iterator[list[tuple[int, bool]]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        size = len(self.forward_lengths)
        local_size = math.ceil(size / self.world_size)
        total_size = local_size * self.world_size
        if self.weights is None:
            indices = rng.permutation(size)
            if total_size > size:
                indices = np.concatenate([indices, indices[: total_size - size]])
        else:
            indices = rng.choice(size, size=total_size, replace=True, p=self.weights)
        local_indices = indices[self.rank : total_size : self.world_size]
        orientations = rng.integers(0, 2, size=len(local_indices), dtype=np.int8).astype(bool)
        keys = list(zip(local_indices.tolist(), orientations.tolist()))

        ordered: list[tuple[int, bool]] = []
        for start in range(0, len(keys), self.bucket_size):
            bucket = keys[start : start + self.bucket_size]
            bucket.sort(
                key=lambda key: (
                    self.reverse_lengths[key[0]]
                    if key[1]
                    else self.forward_lengths[key[0]]
                )
            )
            ordered.extend(bucket)
        batches = [
            ordered[start : start + self.batch_size]
            for start in range(0, len(ordered), self.batch_size)
        ]
        rng.shuffle(batches)
        yield from batches


class FixedLengthBatchSampler(Sampler[list[tuple[int, bool]]]):
    """Deterministic, padding-efficient batches for final validation."""

    def __init__(
        self,
        cache: TokenCache,
        pair_indices: Sequence[int],
        batch_size: int,
        both_orientations: bool = True,
    ) -> None:
        keys = [(int(index), False) for index in pair_indices]
        if both_orientations:
            keys.extend((int(index), True) for index in pair_indices)
        forward_lengths = cache.forward_lengths
        reverse_lengths = cache.reverse_lengths
        keys.sort(
            key=lambda key: reverse_lengths[key[0]] if key[1] else forward_lengths[key[0]]
        )
        self.batches = [
            keys[start : start + batch_size]
            for start in range(0, len(keys), batch_size)
        ]

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self) -> Iterator[list[tuple[int, bool]]]:
        yield from self.batches
