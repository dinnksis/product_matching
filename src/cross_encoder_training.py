from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _frame_fingerprint(frame: pd.DataFrame, configuration: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(configuration, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )
    columns = ["id1", "id2", "target", "product_text_1", "product_text_2"]
    row_hashes = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy()
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()[:20]


@dataclass(frozen=True)
class PairTokenCache:
    directory: Path
    forward_tokens: np.ndarray
    forward_offsets: np.ndarray
    reverse_tokens: np.ndarray
    reverse_offsets: np.ndarray
    forward_token_types: np.ndarray | None = None
    reverse_token_types: np.ndarray | None = None

    @classmethod
    def load(cls, directory: Path) -> "PairTokenCache":
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        has_token_types = bool(metadata.get("has_token_type_ids"))
        return cls(
            directory=directory,
            forward_tokens=np.load(directory / "forward_tokens.npy", mmap_mode="r"),
            forward_offsets=np.load(directory / "forward_offsets.npy", mmap_mode="r"),
            reverse_tokens=np.load(directory / "reverse_tokens.npy", mmap_mode="r"),
            reverse_offsets=np.load(directory / "reverse_offsets.npy", mmap_mode="r"),
            forward_token_types=(
                np.load(directory / "forward_token_types.npy", mmap_mode="r")
                if has_token_types
                else None
            ),
            reverse_token_types=(
                np.load(directory / "reverse_token_types.npy", mmap_mode="r")
                if has_token_types
                else None
            ),
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

    @property
    def has_token_type_ids(self) -> bool:
        return self.forward_token_types is not None

    def sequence(self, index: int, reverse: bool = False) -> dict[str, np.ndarray]:
        tokens = self.reverse_tokens if reverse else self.forward_tokens
        offsets = self.reverse_offsets if reverse else self.forward_offsets
        start, end = int(offsets[index]), int(offsets[index + 1])
        result = {"input_ids": tokens[start:end]}
        token_types = self.reverse_token_types if reverse else self.forward_token_types
        if token_types is not None:
            result["token_type_ids"] = token_types[start:end]
        return result

def _cache_is_complete(directory: Path, configuration: dict[str, Any], rows: int) -> bool:
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = [
        directory / "forward_tokens.npy",
        directory / "forward_offsets.npy",
        directory / "reverse_tokens.npy",
        directory / "reverse_offsets.npy",
    ]
    if metadata.get("has_token_type_ids"):
        required.extend(
            [
                directory / "forward_token_types.npy",
                directory / "reverse_token_types.npy",
            ]
        )
    return (
        metadata.get("configuration") == configuration
        and metadata.get("rows") == rows
        and all(path.is_file() for path in required)
    )


def build_pair_token_cache(
    frame: pd.DataFrame,
    tokenizer: Any,
    cache_root: Path,
    split_name: str,
    model_name: str,
    max_length: int,
    tokenization_batch_size: int = 512,
    log_every: int = 50,
) -> PairTokenCache:
    """Tokenize both pair orientations once and store compact mmap arrays."""
    configuration = {
        "version": 1,
        "model": model_name,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_size": len(tokenizer) if hasattr(tokenizer, "__len__") else None,
        "max_length": max_length,
    }
    fingerprint = _frame_fingerprint(frame, configuration)
    directory = cache_root / f"{split_name}-{fingerprint}"
    if _cache_is_complete(directory, configuration, len(frame)):
        print(json.dumps({"token_cache_reused": str(directory), "rows": len(frame)}), flush=True)
        return PairTokenCache.load(directory)
    directory.mkdir(parents=True, exist_ok=True)

    forward_offsets = np.zeros(len(frame) + 1, dtype=np.int64)
    reverse_offsets = np.zeros(len(frame) + 1, dtype=np.int64)
    forward_chunks: list[np.ndarray] = []
    reverse_chunks: list[np.ndarray] = []
    forward_type_chunks: list[np.ndarray] = []
    reverse_type_chunks: list[np.ndarray] = []
    forward_position = reverse_position = 0
    has_token_types: bool | None = None
    started = time.perf_counter()

    for batch_number, start in enumerate(
        range(0, len(frame), tokenization_batch_size), start=1
    ):
        part = frame.iloc[start : start + tokenization_batch_size]
        first_texts = part["product_text_1"].astype(str).tolist()
        second_texts = part["product_text_2"].astype(str).tolist()
        part_size = len(part)
        encoded = tokenizer(
            first_texts + second_texts,
            second_texts + first_texts,
            add_special_tokens=True,
            padding=False,
            truncation="longest_first",
            max_length=max_length,
            return_attention_mask=False,
        )
        current_has_token_types = "token_type_ids" in encoded
        if has_token_types is None:
            has_token_types = current_has_token_types
        elif has_token_types != current_has_token_types:
            raise RuntimeError("Tokenizer changed token_type_ids behavior between batches")

        forward_sequences = encoded["input_ids"][:part_size]
        reverse_sequences = encoded["input_ids"][part_size:]
        for offset, (forward, reverse) in enumerate(
            zip(forward_sequences, reverse_sequences), start=1
        ):
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
        if current_has_token_types:
            token_types = encoded["token_type_ids"]
            forward_type_chunks.append(
                np.fromiter(chain.from_iterable(token_types[:part_size]), dtype=np.int32)
            )
            reverse_type_chunks.append(
                np.fromiter(chain.from_iterable(token_types[part_size:]), dtype=np.int32)
            )

        processed = start + part_size
        if batch_number % log_every == 0 or processed == len(frame):
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "tokenizing_split": split_name,
                        "rows": processed,
                        "total_rows": len(frame),
                        "pair_orientations_per_second": 2 * processed / elapsed,
                        "elapsed_seconds": elapsed,
                    }
                ),
                flush=True,
            )

    empty = np.empty(0, dtype=np.int32)
    np.save(
        directory / "forward_tokens.npy",
        np.concatenate(forward_chunks) if forward_chunks else empty,
    )
    np.save(directory / "forward_offsets.npy", forward_offsets)
    np.save(
        directory / "reverse_tokens.npy",
        np.concatenate(reverse_chunks) if reverse_chunks else empty,
    )
    np.save(directory / "reverse_offsets.npy", reverse_offsets)
    if has_token_types:
        np.save(directory / "forward_token_types.npy", np.concatenate(forward_type_chunks))
        np.save(directory / "reverse_token_types.npy", np.concatenate(reverse_type_chunks))

    metadata = {
        "configuration": configuration,
        "rows": len(frame),
        "has_token_type_ids": bool(has_token_types),
        "elapsed_seconds": time.perf_counter() - started,
        "forward_tokens": int(forward_position),
        "reverse_tokens": int(reverse_position),
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"token_cache": str(directory), **metadata}), flush=True)
    return PairTokenCache.load(directory)


class CrossEncoderPairDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        cache: PairTokenCache,
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
            **self.cache.sequence(index, reverse=reverse),
            "target": self.targets[index],
            "sample_weight": self.sample_weights[index],
            "pair_index": index,
            "reverse": reverse,
        }


@dataclass(frozen=True)
class CrossEncoderBatchCollator:
    pad_token_id: int

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_length = max(len(row["input_ids"]) for row in rows)
        input_ids = torch.full(
            (len(rows), max_length), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((len(rows), max_length), dtype=torch.long)
        has_token_types = "token_type_ids" in rows[0]
        token_type_ids = (
            torch.zeros((len(rows), max_length), dtype=torch.long)
            if has_token_types
            else None
        )
        for row_index, row in enumerate(rows):
            length = len(row["input_ids"])
            input_ids[row_index, :length] = torch.from_numpy(
                np.asarray(row["input_ids"]).copy()
            ).to(dtype=torch.long)
            attention_mask[row_index, :length] = 1
            if token_type_ids is not None:
                token_type_ids[row_index, :length] = torch.from_numpy(
                    np.asarray(row["token_type_ids"]).copy()
                ).to(dtype=torch.long)
        batch = {
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
        if token_type_ids is not None:
            batch["token_type_ids"] = token_type_ids
        return batch
