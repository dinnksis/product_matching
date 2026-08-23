from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .normalization import normalize_text, tokenize


def bm25_scores(
    query: str,
    documents: list[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> np.ndarray:
    """Small-pool BM25 used after category/subtype filtering."""
    if not documents:
        return np.empty(0, dtype=np.float32)
    query_tokens = list(dict.fromkeys(tokenize(query)))
    counters = [Counter(tokenize(document)) for document in documents]
    lengths = np.asarray([sum(counter.values()) for counter in counters], dtype=np.float32)
    average_length = max(float(lengths.mean()), 1.0)
    scores = np.zeros(len(documents), dtype=np.float32)
    total = len(documents)
    for token in query_tokens:
        document_frequency = sum(token in counter for counter in counters)
        if not document_frequency:
            continue
        inverse_document_frequency = math.log(
            1.0 + (total - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        for index, counter in enumerate(counters):
            frequency = counter.get(token, 0)
            if not frequency:
                continue
            denominator = frequency + k1 * (
                1.0 - b + b * float(lengths[index]) / average_length
            )
            scores[index] += inverse_document_frequency * frequency * (k1 + 1.0) / denominator
    return scores


def reciprocal_rank_fusion(
    dense_scores: np.ndarray | None,
    lexical_scores: np.ndarray,
    *,
    dense_weight: float = 0.55,
    lexical_weight: float = 0.45,
    rank_constant: float = 20.0,
) -> np.ndarray:
    size = len(lexical_scores)
    fused = np.zeros(size, dtype=np.float32)

    def add(scores: np.ndarray, weight: float) -> None:
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(size, dtype=np.int32)
        ranks[order] = np.arange(1, size + 1)
        fused[:] += weight / (rank_constant + ranks)

    if dense_scores is not None:
        add(np.asarray(dense_scores, dtype=np.float32), dense_weight)
    effective_lexical_weight = lexical_weight if dense_scores is not None else 1.0
    add(np.asarray(lexical_scores, dtype=np.float32), effective_lexical_weight)
    return fused


class HybridRetriever:
    """Category -> subtype -> dense+BM25 retrieval over an exemplar bank."""

    def __init__(
        self,
        bank: pd.DataFrame,
        embeddings: np.ndarray | None = None,
        *,
        min_subtype_pool: int = 8,
    ) -> None:
        required = {"id", "category", "subtype", "retrieval_text", "name"}
        missing = required - set(bank.columns)
        if missing:
            raise ValueError(f"Exemplar bank is missing columns: {sorted(missing)}")
        self.bank = bank.reset_index(drop=True)
        self.embeddings = embeddings
        if embeddings is not None and len(embeddings) != len(bank):
            raise ValueError("Embedding count does not match exemplar bank")
        self.min_subtype_pool = min_subtype_pool
        self.by_category: dict[str, np.ndarray] = {}
        self.by_subtype: dict[tuple[str, str], np.ndarray] = {}
        category_positions: dict[str, list[int]] = defaultdict(list)
        subtype_positions: dict[tuple[str, str], list[int]] = defaultdict(list)
        for position, row in enumerate(self.bank.itertuples(index=False)):
            category = str(row.category)
            subtype = str(row.subtype)
            category_positions[category].append(position)
            subtype_positions[(category, subtype)].append(position)
        self.by_category = {
            key: np.asarray(value, dtype=np.int32) for key, value in category_positions.items()
        }
        self.by_subtype = {
            key: np.asarray(value, dtype=np.int32) for key, value in subtype_positions.items()
        }

    @classmethod
    def from_index_dir(cls, index_dir: Path) -> "HybridRetriever":
        bank = pd.read_parquet(index_dir / "exemplar_bank.parquet")
        embeddings_path = index_dir / "embeddings.f16.npy"
        embedding_ids_path = index_dir / "embedding_ids.npy"
        embeddings: np.ndarray | None = None
        if embeddings_path.exists():
            if not embedding_ids_path.exists():
                raise ValueError("Dense index exists without embedding_ids.npy")
            embedding_ids = np.load(embedding_ids_path)
            if not np.array_equal(embedding_ids, bank["id"].to_numpy(dtype=np.int64)):
                raise ValueError("Dense index item order differs from exemplar bank")
            embeddings = np.load(embeddings_path, mmap_mode="r")
        elif embedding_ids_path.exists():
            raise ValueError("embedding_ids.npy exists without dense embeddings")
        return cls(bank, embeddings)

    def candidate_positions(self, anchor_position: int, needed: int) -> np.ndarray:
        row = self.bank.iloc[anchor_position]
        subtype_key = (str(row["category"]), str(row["subtype"]))
        positions = self.by_subtype.get(subtype_key, np.empty(0, dtype=np.int32))
        if len(positions) < max(self.min_subtype_pool, needed + 1):
            positions = self.by_category[str(row["category"])]
        return positions[positions != anchor_position]

    def retrieve(self, anchor_position: int, *, k: int = 5) -> list[int]:
        candidates = self.candidate_positions(anchor_position, k)
        if not len(candidates):
            return []
        anchor = self.bank.iloc[anchor_position]
        exact_name = normalize_text(anchor["name"])
        candidates = np.asarray(
            [
                position
                for position in candidates
                if normalize_text(self.bank.iloc[int(position)]["name"]) != exact_name
            ],
            dtype=np.int32,
        )
        if not len(candidates):
            return []
        documents = self.bank.iloc[candidates]["retrieval_text"].astype(str).tolist()
        lexical = bm25_scores(str(anchor["retrieval_text"]), documents)
        dense: np.ndarray | None = None
        if self.embeddings is not None:
            anchor_vector = np.asarray(self.embeddings[anchor_position], dtype=np.float32)
            candidate_vectors = np.asarray(self.embeddings[candidates], dtype=np.float32)
            dense = candidate_vectors @ anchor_vector
        fused = reciprocal_rank_fusion(dense, lexical)
        order = np.argsort(-fused, kind="stable")[:k]
        return [int(candidates[index]) for index in order]
