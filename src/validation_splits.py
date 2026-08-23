from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import HashingVectorizer

from src.pair_features import extract_product_names, name_ngram_cosine


TOKEN_RE = re.compile(r"[0-9a-zа-яё]+", flags=re.IGNORECASE)
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


@dataclass(frozen=True)
class HardSelectionTargets:
    total: int
    model_errors: int
    lexical_surprises: int
    diagnostic: int


def stable_component_ids(pairs: pd.DataFrame) -> np.ndarray:
    """Return a stable component id (the minimum item id) for every pair."""
    required = {"id1", "id2"}
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"Pairs are missing columns: {sorted(missing)}")

    all_ids = pd.unique(pairs[["id1", "id2"]].to_numpy().reshape(-1))
    positions = pd.Series(np.arange(len(all_ids), dtype=np.int64), index=all_ids)
    left = positions.loc[pairs["id1"]].to_numpy(dtype=np.int64)
    right = positions.loc[pairs["id2"]].to_numpy(dtype=np.int64)
    parent = np.arange(len(all_ids), dtype=np.int64)
    size = np.ones(len(all_ids), dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for first, second in zip(left, right):
        root1, root2 = find(int(first)), find(int(second))
        if root1 == root2:
            continue
        if size[root1] < size[root2]:
            root1, root2 = root2, root1
        parent[root2] = root1
        size[root1] += size[root2]

    roots = np.fromiter(
        (find(index) for index in range(len(all_ids))),
        dtype=np.int64,
        count=len(all_ids),
    )
    component_minimum = np.full(len(all_ids), np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(component_minimum, roots, np.asarray(all_ids, dtype=np.int64))
    return component_minimum[roots[left]]


def proportional_group_quotas(
    frame: pd.DataFrame,
    total: int,
    *,
    group_columns: Sequence[str] = ("category", "target"),
) -> dict[tuple[object, ...], int]:
    """Allocate an exact total by largest remainder while preserving strata."""
    if total < 0 or total > len(frame):
        raise ValueError(f"Requested {total} rows from a population of {len(frame)}")
    counts = frame.groupby(list(group_columns), dropna=False, sort=True).size()
    if counts.empty:
        if total:
            raise ValueError("Cannot allocate rows from an empty population")
        return {}

    expected = counts.astype(np.float64) * (total / len(frame))
    quotas = np.floor(expected).astype(np.int64)
    remainder = total - int(quotas.sum())
    if remainder:
        fractional = (expected - quotas).sort_values(ascending=False, kind="stable")
        for key in fractional.index:
            if remainder == 0:
                break
            if quotas.loc[key] < counts.loc[key]:
                quotas.loc[key] += 1
                remainder -= 1
    if remainder:
        raise RuntimeError(f"Unable to allocate {remainder} requested rows")

    result: dict[tuple[object, ...], int] = {}
    for key, value in quotas.items():
        normalized_key = key if isinstance(key, tuple) else (key,)
        result[normalized_key] = int(value)
    return result


def sample_component_anchors(
    frame: pd.DataFrame,
    total: int,
    *,
    seed: int,
    group_columns: Sequence[str] = ("category", "target"),
) -> pd.Index:
    """Sample one anchor per component with approximately natural strata."""
    if "component_id" not in frame:
        raise ValueError("Frame is missing component_id")
    quotas = proportional_group_quotas(frame, total, group_columns=group_columns)
    rng = np.random.default_rng(seed)
    selected: list[object] = []
    selected_components: set[int] = set()

    grouped = frame.groupby(list(group_columns), dropna=False, sort=True)
    for raw_key, group in grouped:
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        needed = quotas[key]
        positions = group.index.to_numpy(copy=True)
        rng.shuffle(positions)
        for index in positions:
            component = int(frame.at[index, "component_id"])
            if component in selected_components:
                continue
            selected.append(index)
            selected_components.add(component)
            needed -= 1
            if needed == 0:
                break
        if needed:
            raise ValueError(f"Not enough independent components for stratum {key}: {needed}")
    if len(selected) != total:
        raise RuntimeError(f"Expected {total} anchors, selected {len(selected)}")
    return pd.Index(selected)


def sample_components_to_pair_count(
    frame: pd.DataFrame,
    total_pairs: int,
    *,
    seed: int,
) -> set[int]:
    """Randomly select whole components totaling an exact number of pairs.

    Components are shuffled uniformly. Because all rows of a selected component
    are retained, their expected contribution follows the source pair
    distribution without allowing item leakage.
    """
    if "component_id" not in frame:
        raise ValueError("Frame is missing component_id")
    if total_pairs < 0 or total_pairs > len(frame):
        raise ValueError(
            f"Requested {total_pairs} pairs from a population of {len(frame)}"
        )
    component_sizes = frame.groupby("component_id", sort=False).size()
    components = component_sizes.index.to_numpy(dtype=np.int64, copy=True)
    rng = np.random.default_rng(seed)
    rng.shuffle(components)

    selected: set[int] = set()
    remaining = total_pairs
    for component in components:
        size = int(component_sizes.loc[component])
        if size > remaining:
            continue
        selected.add(int(component))
        remaining -= size
        if remaining == 0:
            break
    if remaining:
        raise ValueError(
            f"Could not form an exact {total_pairs}-pair component sample; "
            f"{remaining} pairs remain"
        )
    return selected


def _paired_hashing_cosine(
    first: Sequence[object],
    second: Sequence[object],
    *,
    analyzer: str,
    ngram_range: tuple[int, int],
    batch_size: int = 4096,
    n_features: int = 2**18,
) -> np.ndarray:
    vectorizer = HashingVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
    )
    first_series = pd.Series(first, dtype="string").fillna("")
    second_series = pd.Series(second, dtype="string").fillna("")
    similarities = np.empty(len(first_series), dtype=np.float32)
    for start in range(0, len(first_series), batch_size):
        stop = min(len(first_series), start + batch_size)
        first_vectors = vectorizer.transform(first_series.iloc[start:stop])
        second_vectors = vectorizer.transform(second_series.iloc[start:stop])
        similarities[start:stop] = np.asarray(
            first_vectors.multiply(second_vectors).sum(axis=1)
        ).ravel()
    return np.clip(similarities, 0.0, 1.0)


def _set_similarity(first: str, second: str, pattern: re.Pattern[str]) -> float:
    left = set(pattern.findall(first.casefold()))
    right = set(pattern.findall(second.casefold()))
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


def build_hard_features(
    predictions: pd.DataFrame,
    light_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Build frozen model-error, lexical-surprise, and disagreement signals."""
    required = {
        "id1",
        "id2",
        "target",
        "category_1",
        "product_text_1",
        "product_text_2",
        "score",
        "score_order_gap",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    light_required = {
        "id1",
        "id2",
        "target",
        "category_light_score",
        "global_light_score",
    }
    missing = light_required.difference(light_predictions.columns)
    if missing:
        raise ValueError(f"Light predictions are missing columns: {sorted(missing)}")

    frame = predictions.copy()
    light = light_predictions[list(light_required)].copy()
    light["target"] = light["target"].astype(np.float64)
    frame = frame.merge(
        light,
        on=["id1", "id2", "target"],
        how="left",
        validate="one_to_one",
    )
    if frame[["category_light_score", "global_light_score"]].isna().any().any():
        raise ValueError("Some MiniLM predictions have no matching light prediction")

    first_names = extract_product_names(frame["product_text_1"])
    second_names = extract_product_names(frame["product_text_2"])
    frame["name_char_ngram_cosine"] = name_ngram_cosine(frame)
    frame["full_text_word_ngram_cosine"] = _paired_hashing_cosine(
        frame["product_text_1"],
        frame["product_text_2"],
        analyzer="word",
        ngram_range=(1, 2),
    )

    first_values = first_names.tolist()
    second_values = second_names.tolist()
    frame["name_levenshtein_ratio"] = np.fromiter(
        (fuzz.ratio(first, second) / 100.0 for first, second in zip(first_values, second_values)),
        dtype=np.float32,
        count=len(frame),
    )
    frame["name_token_set_ratio"] = np.fromiter(
        (
            fuzz.token_set_ratio(first, second) / 100.0
            for first, second in zip(first_values, second_values)
        ),
        dtype=np.float32,
        count=len(frame),
    )
    frame["name_token_sort_ratio"] = np.fromiter(
        (
            fuzz.token_sort_ratio(first, second) / 100.0
            for first, second in zip(first_values, second_values)
        ),
        dtype=np.float32,
        count=len(frame),
    )
    frame["name_token_jaccard"] = np.fromiter(
        (
            _set_similarity(first, second, TOKEN_RE)
            for first, second in zip(first_values, second_values)
        ),
        dtype=np.float32,
        count=len(frame),
    )
    frame["name_number_jaccard"] = np.fromiter(
        (
            _set_similarity(first, second, NUMBER_RE)
            for first, second in zip(first_values, second_values)
        ),
        dtype=np.float32,
        count=len(frame),
    )
    first_lengths = first_names.str.len().to_numpy(dtype=np.float32)
    second_lengths = second_names.str.len().to_numpy(dtype=np.float32)
    frame["name_length_ratio"] = np.minimum(first_lengths, second_lengths) / np.maximum(
        np.maximum(first_lengths, second_lengths), 1.0
    )

    frame["lexical_similarity"] = (
        0.30 * frame["name_char_ngram_cosine"]
        + 0.20 * frame["name_token_set_ratio"]
        + 0.15 * frame["name_levenshtein_ratio"]
        + 0.10 * frame["name_token_sort_ratio"]
        + 0.10 * frame["name_token_jaccard"]
        + 0.10 * frame["full_text_word_ngram_cosine"]
        + 0.05 * frame["name_length_ratio"]
    ).clip(0.0, 1.0)

    targets = frame["target"].to_numpy(dtype=np.float32)
    scores = frame["score"].to_numpy(dtype=np.float32)
    frame["model_is_error"] = (scores >= 0.5) != (targets >= 0.5)
    frame["model_wrongness"] = np.where(targets >= 0.5, 1.0 - scores, scores)
    frame["lexical_hardness"] = np.where(
        targets >= 0.5,
        1.0 - frame["lexical_similarity"].to_numpy(dtype=np.float32),
        frame["lexical_similarity"].to_numpy(dtype=np.float32),
    )
    frame["model_light_gap"] = np.abs(
        frame["score"].to_numpy(dtype=np.float32)
        - frame["category_light_score"].to_numpy(dtype=np.float32)
    )
    group_columns = ["category_1", "target"]
    gap_rank = frame.groupby(group_columns, dropna=False)["model_light_gap"].rank(pct=True)
    order_rank = frame.groupby(group_columns, dropna=False)["score_order_gap"].rank(pct=True)
    frame["diagnostic_hardness"] = 0.65 * gap_rank + 0.35 * order_rank
    frame["category"] = frame["category_1"].astype(str)
    return frame


def hard_selection_targets(
    total: int,
    *,
    model_error_fraction: float = 0.40,
    lexical_fraction: float = 0.50,
) -> HardSelectionTargets:
    if not 0 <= model_error_fraction <= 1 or not 0 <= lexical_fraction <= 1:
        raise ValueError("Hard selection fractions must be in [0, 1]")
    if model_error_fraction + lexical_fraction > 1:
        raise ValueError("Hard selection fractions cannot sum to more than 1")
    model_errors = int(round(total * model_error_fraction))
    lexical = int(round(total * lexical_fraction))
    return HardSelectionTargets(
        total=total,
        model_errors=model_errors,
        lexical_surprises=lexical,
        diagnostic=total - model_errors - lexical,
    )


def select_hard_anchors(
    features: pd.DataFrame,
    total: int,
    *,
    model_error_fraction: float = 0.40,
    lexical_fraction: float = 0.50,
) -> pd.DataFrame:
    """Select hard anchors with fixed source shares and natural group quotas."""
    required = {
        "category",
        "target",
        "component_id",
        "model_is_error",
        "model_wrongness",
        "lexical_hardness",
        "diagnostic_hardness",
    }
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Hard features are missing columns: {sorted(missing)}")

    targets = hard_selection_targets(
        total,
        model_error_fraction=model_error_fraction,
        lexical_fraction=lexical_fraction,
    )
    quotas = proportional_group_quotas(features, total)
    remaining = dict(quotas)
    selected_indices: list[object] = []
    selected_set: set[object] = set()
    selected_components: set[int] = set()
    reasons: dict[object, str] = {}

    def select_from(
        candidates: pd.DataFrame,
        score_column: str,
        count: int,
        reason: str,
    ) -> None:
        if count == 0:
            return
        ordered = candidates.sort_values(score_column, ascending=False, kind="stable")
        needed = count
        for index, row in ordered.iterrows():
            if index in selected_set:
                continue
            component = int(row["component_id"])
            if component in selected_components:
                continue
            key = (row["category"], row["target"])
            if remaining.get(key, 0) <= 0:
                continue
            selected_indices.append(index)
            selected_set.add(index)
            selected_components.add(component)
            reasons[index] = reason
            remaining[key] -= 1
            needed -= 1
            if needed == 0:
                return
        raise ValueError(f"Could not select {needed} additional {reason} anchors")

    select_from(
        features.loc[features["model_is_error"]],
        "model_wrongness",
        targets.model_errors,
        "confident_minilm_v1_error",
    )
    select_from(
        features,
        "lexical_hardness",
        targets.lexical_surprises,
        "lexical_surprise",
    )
    select_from(
        features,
        "diagnostic_hardness",
        targets.diagnostic,
        "model_disagreement_or_order_gap",
    )

    if any(remaining.values()):
        raise RuntimeError(f"Hard group quotas were not filled: {remaining}")
    selected = features.loc[selected_indices].copy()
    selected["selection_reason"] = [reasons[index] for index in selected.index]
    return selected


def item_ids(pairs: pd.DataFrame) -> set[int]:
    return set(pairs["id1"].astype(np.int64)) | set(pairs["id2"].astype(np.int64))


def assert_item_disjoint(named_pairs: Iterable[tuple[str, pd.DataFrame]]) -> None:
    seen: dict[int, str] = {}
    for name, pairs in named_pairs:
        for item_id in item_ids(pairs):
            previous = seen.setdefault(item_id, name)
            if previous != name:
                raise ValueError(
                    f"Item {item_id} occurs in both {previous!r} and {name!r}"
                )
