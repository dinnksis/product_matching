from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer


def extract_product_names(product_texts: Sequence[Any]) -> pd.Series:
    """Extract the serialized ``Название`` line without depending on raw items."""
    texts = pd.Series(product_texts, dtype="string")
    return (
        texts.str.extract(r"(?mi)^Название:\s*(.*)$", expand=False)
        .fillna("")
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def name_ngram_cosine(
    frame: pd.DataFrame,
    *,
    batch_size: int = 8192,
    n_features: int = 2**18,
) -> np.ndarray:
    """Compute stateless char 3-5-gram cosine similarity for each product pair."""
    first = extract_product_names(frame["product_text_1"])
    second = extract_product_names(frame["product_text_2"])
    vectorizer = HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
    )
    similarities = np.empty(len(frame), dtype=np.float32)
    for start in range(0, len(frame), batch_size):
        stop = min(len(frame), start + batch_size)
        first_vectors = vectorizer.transform(first.iloc[start:stop])
        second_vectors = vectorizer.transform(second.iloc[start:stop])
        similarities[start:stop] = np.asarray(
            first_vectors.multiply(second_vectors).sum(axis=1)
        ).ravel()
    return np.clip(similarities, 0.0, 1.0)


def category_label_downsample(
    frame: pd.DataFrame,
    *,
    category_column: str = "category_1",
    target_column: str = "target",
    seed: int = 42,
) -> pd.DataFrame:
    """Downsample each category's majority label without balancing categories.

    Every minority row is retained. Categories therefore keep different sizes,
    determined by their available minority examples, and no row is repeated.
    """
    if category_column not in frame or target_column not in frame:
        raise ValueError("Training frame is missing category or target columns")
    if not frame[target_column].isin([0.0, 1.0]).all():
        raise ValueError("Category-label downsampling requires binary targets")

    selected: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    labels = frame[target_column].astype(np.int8)
    for _, category_positions in frame.groupby(
        category_column, sort=True, dropna=False
    ).indices.items():
        category_positions = np.asarray(category_positions, dtype=np.int64)
        category_labels = labels.iloc[category_positions].to_numpy()
        negative_positions = category_positions[category_labels == 0]
        positive_positions = category_positions[category_labels == 1]
        if not len(negative_positions) or not len(positive_positions):
            raise ValueError("Every category must contain both target labels")
        size = min(len(negative_positions), len(positive_positions))
        selected.extend(
            [
                rng.choice(negative_positions, size=size, replace=False),
                rng.choice(positive_positions, size=size, replace=False),
            ]
        )
    positions = np.concatenate(selected)
    rng.shuffle(positions)
    return frame.iloc[positions].reset_index(drop=True)


def build_training_loss_weights(
    categories: Sequence[Any],
    targets: Sequence[float],
    *,
    mode: str = "none",
    lexical_similarities: Sequence[float] | None = None,
    lexical_hard_negative_strength: float = 0.0,
) -> np.ndarray:
    """Build moderate loss weights while retaining every training example.

    ``category_label_sqrt`` is deliberately softer than inverse-frequency
    resampling: a minority example gets a square-root frequency correction,
    while the epoch still contains every row exactly once. Optional lexical
    weighting only redistributes weight *within* each category's negatives, so
    highly similar hard negatives receive more attention without changing the
    total category/label mass.
    """
    targets_array = np.asarray(targets, dtype=np.float32)
    data = pd.DataFrame(
        {
            "category": pd.Series(categories, dtype="string"),
            "label": (targets_array >= 0.5).astype(np.int8),
        }
    )
    if mode == "none":
        weights = np.ones(len(data), dtype=np.float64)
    elif mode == "category_label_sqrt":
        counts = data.groupby(["category", "label"], dropna=False)[
            "label"
        ].transform("size")
        weights = 1.0 / np.sqrt(counts.to_numpy(dtype=np.float64))
    else:
        raise ValueError(f"Unknown loss-weighting mode: {mode}")

    if lexical_hard_negative_strength:
        if lexical_similarities is None:
            raise ValueError("Lexical similarities are required for hard-negative weights")
        similarities = np.asarray(lexical_similarities, dtype=np.float64)
        if len(similarities) != len(data):
            raise ValueError("Lexical similarities and targets have different lengths")
        if not np.isfinite(similarities).all():
            raise ValueError("Lexical similarities must be finite")
        modifier = np.ones(len(data), dtype=np.float64)
        negatives = data["label"].eq(0).to_numpy()
        modifier[negatives] += (
            lexical_hard_negative_strength
            * np.clip(similarities[negatives], 0.0, 1.0)
        )
        # Preserve each negative category's total mass and only redistribute it
        # from easy to lexically confusing pairs.
        negative_groups = data.loc[negatives].groupby("category", dropna=False).indices
        negative_positions = np.flatnonzero(negatives)
        for positions in negative_groups.values():
            absolute_positions = negative_positions[np.asarray(positions)]
            modifier[absolute_positions] /= modifier[absolute_positions].mean()
        weights *= modifier

    weights /= weights.mean()
    return weights.astype(np.float32)
