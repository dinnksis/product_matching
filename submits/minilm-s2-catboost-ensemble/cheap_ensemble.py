"""Compact lexical, numeric, code, and attribute features for S2 stacking."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as sparse_normalize

try:
    from src.serialization_ablation import family_signature, normalize_text, parse_attributes
except ModuleNotFoundError:  # Standalone competition ZIP keeps modules at its root.
    from serialization_ablation import family_signature, normalize_text, parse_attributes


TOKEN_RE = re.compile(r"[0-9a-zа-я]+", re.IGNORECASE)
PLAIN_NUMBER_RE = re.compile(
    r"(?<![0-9a-zа-я])\d+(?:[.,]\d+)?(?![0-9a-zа-я])", re.IGNORECASE
)
CODE_RE = re.compile(
    r"(?=[0-9a-zа-я-]*[a-zа-я])(?=[0-9a-zа-я-]*\d)"
    r"(?:[0-9a-zа-я]+(?:-[0-9a-zа-я]+)+|[0-9a-zа-я]{5,})",
    re.IGNORECASE,
)
SLASH_SPEC_RE = re.compile(r"(?<!\w)(\d{1,5})\s*/\s*(\d{1,5})(?!\w)")
MEASURE_RE = re.compile(
    r"(?<!\w)(\d+(?:[.,]\d+)?)\s*"
    r"(tb|gb|mb|kb|kg|mg|g|ml|l|mm|cm|hz|kw|w|mah|v|шт|дюйм)(?!\w)",
    re.IGNORECASE,
)

FAMILY_FACTORS: dict[str, tuple[str, float]] = {
    "tb": ("memory_mb", 1024.0 * 1024.0),
    "gb": ("memory_mb", 1024.0),
    "mb": ("memory_mb", 1.0),
    "kb": ("memory_mb", 1.0 / 1024.0),
    "kg": ("mass_mg", 1_000_000.0),
    "g": ("mass_mg", 1000.0),
    "mg": ("mass_mg", 1.0),
    "l": ("volume_ml", 1000.0),
    "ml": ("volume_ml", 1.0),
    "cm": ("length_mm", 10.0),
    "mm": ("length_mm", 1.0),
    "hz": ("frequency_hz", 1.0),
    "kw": ("power_w", 1000.0),
    "w": ("power_w", 1.0),
    "mah": ("battery_mah", 1.0),
    "v": ("voltage_v", 1.0),
    "шт": ("quantity", 1.0),
    "дюйм": ("inch", 1.0),
}

BRAND_KEY_RE = re.compile(r"(?:^|\b)(бренд|brand|марка|производитель)(?:\b|$)")
MODEL_KEY_RE = re.compile(
    r"модел|model|артикул|партномер|part\s*number|sku|код\s*(?:товара|модели)|mpn|oem"
)
MEMORY_KEY_RE = re.compile(
    r"памят|memory|накопител|storage|(?:^|\b)ram(?:\b|$)|(?:^|\b)rom(?:\b|$)|ssd|hdd"
)
COLOR_KEY_RE = re.compile(r"цвет|оттенок|color")
FAMILY_PATTERNS = {
    "brand": BRAND_KEY_RE,
    "model": MODEL_KEY_RE,
    "memory": MEMORY_KEY_RE,
    "color": COLOR_KEY_RE,
}


NUMERIC_FEATURES = (
    "transformer_score",
    "title_exact",
    "title_token_jaccard",
    "title_char_tfidf_cosine",
    "title_fuzzy_ratio",
    "title_common_tokens",
    "title_length_min",
    "title_length_max",
    "title_length_ratio",
    "title_length_delta",
    "number_count_min",
    "number_count_max",
    "number_overlap_count",
    "number_unmatched_count",
    "number_set_exact",
    "numeric_context_conflict_count",
    "unit_overlap_count",
    "unit_conflict_count",
    "slash_spec_match",
    "slash_spec_conflict",
    "code_count_min",
    "code_count_max",
    "code_overlap_count",
    "code_exact",
    "code_conflict",
    "attribute_value_count_min",
    "attribute_value_count_max",
    "attribute_value_overlap_count",
    "attribute_value_overlap_ratio",
    "attribute_values_similarity",
    "attribute_values_token_jaccard",
    "brand_match",
    "brand_conflict",
    "model_match",
    "model_conflict",
    "memory_match",
    "memory_conflict",
    "color_match",
    "color_conflict",
    "critical_conflict",
    "sku_human_asymmetry",
)
FEATURE_COLUMNS = (*NUMERIC_FEATURES, "category")


def normalized_tokens(text: str) -> frozenset[str]:
    return frozenset(TOKEN_RE.findall(text))


def extract_numbers(text: str) -> frozenset[str]:
    return frozenset(match.replace(",", ".") for match in PLAIN_NUMBER_RE.findall(text))


def extract_codes(text: str) -> frozenset[str]:
    return frozenset(match.replace("-", "") for match in CODE_RE.findall(text))


def extract_slash_specs(text: str) -> frozenset[tuple[int, int]]:
    return frozenset((int(first), int(second)) for first, second in SLASH_SPEC_RE.findall(text))


def extract_measures(text: str) -> Mapping[str, frozenset[float]]:
    result: dict[str, set[float]] = {}
    for raw_number, raw_unit in MEASURE_RE.findall(text):
        family, factor = FAMILY_FACTORS[raw_unit.casefold()]
        value = round(float(raw_number.replace(",", ".")) * factor, 6)
        result.setdefault(family, set()).add(value)
    return {family: frozenset(values) for family, values in result.items()}


def semantic_values(
    attributes: Sequence[tuple[str, str]], pattern: re.Pattern[str]
) -> frozenset[str]:
    return frozenset(value for key, value in attributes if pattern.search(key))


def item_record(row: Any) -> dict[str, Any]:
    title = normalize_text(row.name)
    attributes = parse_attributes(row.attributes)
    values = tuple(value for _, value in attributes)
    return {
        "id": row.id,
        "category": str(row.category),
        "title": title,
        "tokens": normalized_tokens(title),
        "numbers": extract_numbers(title),
        "measures": extract_measures(title + " " + " ".join(values)),
        "slash_specs": extract_slash_specs(title),
        "codes": extract_codes(title),
        "attribute_values": frozenset(values),
        "attribute_values_text": " ".join(values),
        "family_signature": family_signature(row.category, row.name, attributes),
        **{
            f"{family}_values": semantic_values(attributes, pattern)
            for family, pattern in FAMILY_PATTERNS.items()
        },
    }


def prepare_item_records(items: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame.from_records(item_record(row) for row in items.itertuples(index=False))


def hashing_vectorizer(
    *, n_features: int, ngram_min: int = 3, ngram_max: int = 5
) -> HashingVectorizer:
    return HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(ngram_min, ngram_max),
        n_features=n_features,
        alternate_sign=False,
        binary=True,
        norm=None,
        lowercase=False,
        dtype=np.float32,
    )


def fit_hashed_char_idf(
    titles: Iterable[str],
    *,
    n_features: int,
    ngram_min: int = 3,
    ngram_max: int = 5,
    batch_size: int = 10_000,
) -> tuple[np.ndarray, int]:
    vectorizer = hashing_vectorizer(
        n_features=n_features, ngram_min=ngram_min, ngram_max=ngram_max
    )
    document_frequency = np.zeros(n_features, dtype=np.int64)
    batch: list[str] = []
    document_count = 0
    for title in titles:
        batch.append(normalize_text(title))
        if len(batch) < batch_size:
            continue
        matrix = vectorizer.transform(batch)
        document_frequency += np.bincount(matrix.indices, minlength=n_features)
        document_count += len(batch)
        batch.clear()
    if batch:
        matrix = vectorizer.transform(batch)
        document_frequency += np.bincount(matrix.indices, minlength=n_features)
        document_count += len(batch)
    idf = np.log((1.0 + document_count) / (1.0 + document_frequency)) + 1.0
    return idf.astype(np.float32), document_count


def hashed_char_tfidf(
    titles: Sequence[str],
    idf: np.ndarray,
    *,
    ngram_min: int = 3,
    ngram_max: int = 5,
) -> sparse.csr_matrix:
    vectorizer = hashing_vectorizer(
        n_features=len(idf), ngram_min=ngram_min, ngram_max=ngram_max
    )
    matrix = vectorizer.transform(titles).tocsr()
    matrix = matrix.multiply(idf).tocsr()
    return sparse_normalize(matrix, norm="l2", copy=False)


def _set_pair(left: frozenset[Any], right: frozenset[Any]) -> tuple[int, int, float]:
    overlap = len(left & right)
    union = len(left | right)
    return overlap, union, overlap / union if union else 0.0


def _match_conflict(left: frozenset[str], right: frozenset[str]) -> tuple[float, float]:
    both = bool(left and right)
    return float(bool(left & right)), float(both and not left & right)


def build_pair_features(
    items: pd.DataFrame,
    pairs: pd.DataFrame,
    transformer_scores: Sequence[float],
    char_idf: np.ndarray,
    *,
    ngram_min: int = 3,
    ngram_max: int = 5,
) -> pd.DataFrame:
    positions = pd.Series(np.arange(len(items), dtype=np.int32), index=items["id"].to_numpy())
    left_positions = positions.loc[pairs["id1"]].to_numpy(dtype=np.int32)
    right_positions = positions.loc[pairs["id2"]].to_numpy(dtype=np.int32)
    title_vectors = hashed_char_tfidf(
        items["title"].tolist(), char_idf, ngram_min=ngram_min, ngram_max=ngram_max
    )
    char_cosine = np.asarray(
        title_vectors[left_positions].multiply(title_vectors[right_positions]).sum(axis=1)
    ).reshape(-1)
    records: list[dict[str, Any]] = []
    for row_index, (left_position, right_position) in enumerate(
        zip(left_positions, right_positions)
    ):
        left = items.iloc[int(left_position)]
        right = items.iloc[int(right_position)]
        if left.category != right.category:
            raise ValueError("Cross-category pair encountered")
        token_overlap, token_union, token_jaccard = _set_pair(left.tokens, right.tokens)
        number_overlap, number_union, _ = _set_pair(left.numbers, right.numbers)
        code_overlap, _, _ = _set_pair(left.codes, right.codes)
        value_overlap, _, value_ratio = _set_pair(
            left.attribute_values, right.attribute_values
        )
        value_tokens_left = normalized_tokens(left.attribute_values_text)
        value_tokens_right = normalized_tokens(right.attribute_values_text)
        _, _, value_token_jaccard = _set_pair(value_tokens_left, value_tokens_right)
        longest = max(len(left.title), len(right.title))
        unit_overlap = unit_conflict = 0
        for family in set(left.measures) & set(right.measures):
            if left.measures[family] & right.measures[family]:
                unit_overlap += 1
            else:
                unit_conflict += 1
        slash_match = bool(left.slash_specs & right.slash_specs)
        slash_conflict = bool(
            left.slash_specs and right.slash_specs and not left.slash_specs & right.slash_specs
        )
        unmatched_left = left.numbers - right.numbers
        unmatched_right = right.numbers - left.numbers
        numeric_context_conflicts = (
            min(len(unmatched_left), len(unmatched_right)) if number_overlap else 0
        ) + unit_conflict + int(slash_conflict)
        code_conflict = bool(left.codes and right.codes and not left.codes & right.codes)
        family_features: dict[str, float] = {}
        for family in FAMILY_PATTERNS:
            match, conflict = _match_conflict(
                left[f"{family}_values"], right[f"{family}_values"]
            )
            family_features[f"{family}_match"] = match
            family_features[f"{family}_conflict"] = conflict
        critical_conflict = bool(
            numeric_context_conflicts
            or code_conflict
            or family_features["model_conflict"]
            or family_features["memory_conflict"]
        )
        records.append(
            {
                "transformer_score": float(transformer_scores[row_index]),
                "title_exact": float(left.title == right.title),
                "title_token_jaccard": token_jaccard,
                "title_char_tfidf_cosine": float(char_cosine[row_index]),
                "title_fuzzy_ratio": fuzz.ratio(left.title, right.title) / 100.0,
                "title_common_tokens": token_overlap,
                "title_length_min": min(len(left.title), len(right.title)),
                "title_length_max": max(len(left.title), len(right.title)),
                "title_length_ratio": min(len(left.title), len(right.title)) / longest if longest else 1.0,
                "title_length_delta": abs(len(left.title) - len(right.title)),
                "number_count_min": min(len(left.numbers), len(right.numbers)),
                "number_count_max": max(len(left.numbers), len(right.numbers)),
                "number_overlap_count": number_overlap,
                "number_unmatched_count": number_union - number_overlap,
                "number_set_exact": float(bool(left.numbers) and left.numbers == right.numbers),
                "numeric_context_conflict_count": numeric_context_conflicts,
                "unit_overlap_count": unit_overlap,
                "unit_conflict_count": unit_conflict,
                "slash_spec_match": float(slash_match),
                "slash_spec_conflict": float(slash_conflict),
                "code_count_min": min(len(left.codes), len(right.codes)),
                "code_count_max": max(len(left.codes), len(right.codes)),
                "code_overlap_count": code_overlap,
                "code_exact": float(bool(left.codes) and left.codes == right.codes),
                "code_conflict": float(code_conflict),
                "attribute_value_count_min": min(
                    len(left.attribute_values), len(right.attribute_values)
                ),
                "attribute_value_count_max": max(
                    len(left.attribute_values), len(right.attribute_values)
                ),
                "attribute_value_overlap_count": value_overlap,
                "attribute_value_overlap_ratio": value_ratio,
                "attribute_values_similarity": fuzz.token_set_ratio(
                    left.attribute_values_text, right.attribute_values_text
                ) / 100.0,
                "attribute_values_token_jaccard": value_token_jaccard,
                **family_features,
                "critical_conflict": float(critical_conflict),
                "sku_human_asymmetry": float(bool(left.codes) != bool(right.codes)),
                "category": str(left.category),
            }
        )
    result = pd.DataFrame.from_records(records)
    return result.loc[:, FEATURE_COLUMNS]
