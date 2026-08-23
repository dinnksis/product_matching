"""Validation-split and error-pattern audit for human-labelled product pairs.

The module deliberately contains no embedding model.  It builds parallel,
interpretable representations of names and attributes, creates several
leakage-aware split scenarios, and evaluates the same lexical CatBoost on each
scenario.  Error-analysis helpers operate on saved out-of-fold predictions so
the expensive split experiment and the exploratory n-gram work stay separate.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.metrics import average_precision_score


SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^0-9a-zа-яё]+", re.IGNORECASE)
LETTER_DIGIT_1 = re.compile(r"(?<=[^\W\d_])(?=\d)", re.UNICODE)
LETTER_DIGIT_2 = re.compile(r"(?<=\d)(?=[^\W\d_])", re.UNICODE)
TOKEN_RE = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
ALNUM_ID_RE = re.compile(r"(?=[0-9a-zа-яё-]*[a-zа-яё])(?=[0-9a-zа-яё-]*\d)[0-9a-zа-яё-]{3,}", re.IGNORECASE)
MEASURE_RE = re.compile(
    r"(?<!\w)(\d+(?:[.,]\d+)?)\s*(мл|ml|л|l|г|gr|гр|kg|кг|мг|mg|gb|гб|tb|тб|см|cm|мм|mm|м|m|шт)(?!\w)",
    re.IGNORECASE,
)

BRAND_KEY_RE = re.compile(r"бренд|brand|производитель", re.IGNORECASE)
MODEL_KEY_RE = re.compile(r"модель|model|артикул|партномер|part.?number|sku|mpn|oem|код товара", re.IGNORECASE)
SELLER_KEY_RE = re.compile(r"продавец|seller|магазин|store|поставщик|vendor", re.IGNORECASE)
COLOR_KEY_RE = re.compile(r"цвет|оттенок|color", re.IGNORECASE)

MEASURE_FACTORS = {
    "мл": ("volume_ml", 1.0), "ml": ("volume_ml", 1.0),
    "л": ("volume_ml", 1000.0), "l": ("volume_ml", 1000.0),
    "мг": ("mass_mg", 1.0), "mg": ("mass_mg", 1.0),
    "г": ("mass_mg", 1000.0), "гр": ("mass_mg", 1000.0), "gr": ("mass_mg", 1000.0),
    "кг": ("mass_mg", 1_000_000.0), "kg": ("mass_mg", 1_000_000.0),
    "gb": ("memory_gb", 1.0), "гб": ("memory_gb", 1.0),
    "tb": ("memory_gb", 1024.0), "тб": ("memory_gb", 1024.0),
    "мм": ("length_mm", 1.0), "mm": ("length_mm", 1.0),
    "см": ("length_mm", 10.0), "cm": ("length_mm", 10.0),
    "м": ("length_mm", 1000.0), "m": ("length_mm", 1000.0),
    "шт": ("quantity", 1.0),
}


@dataclass(frozen=True)
class SplitResult:
    name: str
    train_mask: np.ndarray
    valid_mask: np.ndarray
    notes: str


def stable_hash(value: object, seed: int = 0) -> int:
    payload = f"{seed}\0{value}".encode("utf-8", errors="replace")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def normalize_name(value: object) -> str:
    """Moderate normalization that preserves numbers and identifiers."""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold().replace("ё", "е")
    text = LETTER_DIGIT_1.sub(" ", text)
    text = LETTER_DIGIT_2.sub(" ", text)
    text = PUNCT_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def lexical_name(value: object) -> str:
    """Exact name normalization used by the submitted experiment 01."""

    return SPACE_RE.sub(" ", str(value or "")).strip().casefold().replace("ё", "е")


def tokens(value: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(value))


def normalized_measures(value: str) -> tuple[str, ...]:
    result = []
    for number, raw_unit in MEASURE_RE.findall(value):
        unit = raw_unit.casefold().replace("ё", "е")
        kind, factor = MEASURE_FACTORS[unit]
        normalized = round(float(number.replace(",", ".")) * factor, 6)
        result.append(f"{kind}:{normalized:g}")
    return tuple(sorted(set(result)))


def parse_attributes(raw: object) -> dict[str, str]:
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    result: dict[str, str] = {}
    for key, value in data.items():
        normalized_key = normalize_name(key)
        normalized_value = normalize_name(value)
        if normalized_key and normalized_value:
            result[normalized_key] = normalized_value
    return result


def values_for_key(attributes: dict[str, str], pattern: re.Pattern[str]) -> tuple[str, ...]:
    return tuple(sorted({value for key, value in attributes.items() if pattern.search(key)}))


def prepare_items(items: pd.DataFrame) -> pd.DataFrame:
    """Create item-level representations used by both notebooks."""

    records = []
    for row in items.itertuples(index=False):
        name = str(row.name or "")
        normalized = normalize_name(name)
        lexical = lexical_name(name)
        item_tokens = tokens(normalized)
        attributes = parse_attributes(row.attributes)
        brand_values = values_for_key(attributes, BRAND_KEY_RE)
        model_values = values_for_key(attributes, MODEL_KEY_RE)
        seller_values = values_for_key(attributes, SELLER_KEY_RE)
        color_values = values_for_key(attributes, COLOR_KEY_RE)
        alnum_ids = tuple(sorted(set(ALNUM_ID_RE.findall(normalized))))
        numbers = tuple(sorted(set(NUMBER_RE.findall(normalized))))
        measures = normalized_measures(normalized + " " + " ".join(attributes.values()))
        # A deliberately conservative family signature.  It is empty when no
        # model-like evidence exists; generic brand-only groups would be huge.
        model_basis = model_values or alnum_ids
        family_signature = ""
        if model_basis:
            family_signature = "|".join(
                (normalize_name(row.category), ",".join(brand_values), ",".join(model_basis))
            )
        records.append(
            {
                "id": row.id,
                "name": name,
                "category": str(row.category),
                "normalized_name": normalized,
                "lexical_name": lexical,
                "tokens": item_tokens,
                "numbers": numbers,
                "measures": measures,
                "alnum_ids": alnum_ids,
                "brand_values": brand_values,
                "model_values": model_values,
                "seller_values": seller_values,
                "color_values": color_values,
                "family_signature": family_signature,
                "name_signature": f"{row.category}|{normalized}",
                "attributes_parsed": attributes,
            }
        )
    return pd.DataFrame.from_records(records)


def pair_positions(items: pd.DataFrame, matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    positions = pd.Series(np.arange(len(items), dtype=np.int32), index=items.id.to_numpy())
    left = positions.loc[matches.id1].to_numpy()
    right = positions.loc[matches.id2].to_numpy()
    return left, right


def lexical_pair_table(items: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Build the exact eight lexical features used by experiment 01 plus audit signals."""

    left, right = pair_positions(items, matches)
    item_names = items.name.to_numpy()
    lexical_names = items.lexical_name.to_numpy()
    normalized_names = items.normalized_name.to_numpy()
    item_categories = items.category.to_numpy()
    item_numbers = items.numbers.to_numpy()
    item_measures = items.measures.to_numpy()
    item_brands = items.brand_values.to_numpy()
    item_models = items.model_values.to_numpy()
    item_ids = items.alnum_ids.to_numpy()
    item_sellers = items.seller_values.to_numpy()
    item_colors = items.color_values.to_numpy()
    rows = []
    for lp, rp in zip(left, right):
        first_name, second_name = lexical_names[lp], lexical_names[rp]
        first_numbers, second_numbers = set(NUMBER_RE.findall(first_name)), set(NUMBER_RE.findall(second_name))
        union = first_numbers | second_numbers
        longest = max(len(first_name), len(second_name))
        first_measures, second_measures = set(item_measures[lp]), set(item_measures[rp])
        first_brands, second_brands = set(item_brands[lp]), set(item_brands[rp])
        first_models = set(item_models[lp] or item_ids[lp])
        second_models = set(item_models[rp] or item_ids[rp])
        first_sellers, second_sellers = set(item_sellers[lp]), set(item_sellers[rp])
        first_colors, second_colors = set(item_colors[lp]), set(item_colors[rp])
        rows.append(
            {
                "name_ratio": fuzz.ratio(first_name, second_name) / 100.0,
                "name_token_set_ratio": fuzz.token_set_ratio(first_name, second_name) / 100.0,
                "name_token_sort_ratio": fuzz.token_sort_ratio(first_name, second_name) / 100.0,
                "name_exact": float(first_name == second_name),
                "name_length_ratio": min(len(first_name), len(second_name)) / longest if longest else 1.0,
                "name_numeric_jaccard": len(first_numbers & second_numbers) / max(1, len(union)),
                "name_numbers_both": float(bool(first_numbers) and bool(second_numbers)),
                "name_length_delta": abs(len(first_name) - len(second_name)),
                "number_conflict": float(bool(first_numbers and second_numbers and first_numbers != second_numbers)),
                "measure_match": float(bool(first_measures & second_measures)),
                "measure_conflict": float(bool(first_measures and second_measures and first_measures != second_measures)),
                "brand_match": float(bool(first_brands & second_brands)),
                "brand_conflict": float(bool(first_brands and second_brands and not first_brands & second_brands)),
                "model_match": float(bool(first_models & second_models)),
                "model_conflict": float(bool(first_models and second_models and not first_models & second_models)),
                "seller_match": float(bool(first_sellers & second_sellers)),
                "seller_conflict": float(bool(first_sellers and second_sellers and not first_sellers & second_sellers)),
                "color_match": float(bool(first_colors & second_colors)),
                "color_conflict": float(bool(first_colors and second_colors and not first_colors & second_colors)),
            }
        )
    result = matches[["id1", "id2", "target"]].reset_index(drop=True).copy()
    result["category"] = item_categories[left]
    result["name_1"] = item_names[left]
    result["name_2"] = item_names[right]
    result["normalized_name_1"] = normalized_names[left]
    result["normalized_name_2"] = normalized_names[right]
    return pd.concat([result, pd.DataFrame.from_records(rows)], axis=1)


def _component_labels(item_count: int, left: np.ndarray, right: np.ndarray, extra_groups: Sequence[str] | None = None) -> np.ndarray:
    parent = np.arange(item_count, dtype=np.int32)
    size = np.ones(item_count, dtype=np.int32)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for a, b in zip(left, right):
        union(int(a), int(b))
    if extra_groups is not None:
        representative: dict[str, int] = {}
        for position, group in enumerate(extra_groups):
            if not group:
                continue
            if group in representative:
                union(position, representative[group])
            else:
                representative[group] = position
    roots = np.fromiter((find(i) for i in range(item_count)), dtype=np.int32, count=item_count)
    return roots


def split_from_groups(
    pair_groups: np.ndarray,
    validation_fraction: float,
    seed: int,
    name: str,
    notes: str,
) -> SplitResult:
    unique = np.unique(pair_groups)
    selected = np.asarray(
        [group for group in unique if stable_hash(int(group), seed) / 2**64 < validation_fraction],
        dtype=unique.dtype,
    )
    valid = np.isin(pair_groups, selected)
    if not valid.any() or valid.all():
        raise RuntimeError(f"Degenerate split {name}: validation={int(valid.sum())}/{len(valid)}")
    return SplitResult(name=name, train_mask=~valid, valid_mask=valid, notes=notes)


def original_component_split(
    matches: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> SplitResult:
    """Reproduce :func:`src.data_pipeline.component_split` exactly."""

    all_ids = pd.unique(matches[["id1", "id2"]].to_numpy().reshape(-1))
    positions = pd.Series(np.arange(len(all_ids), dtype=np.int64), index=all_ids)
    left = positions.loc[matches.id1].to_numpy()
    right = positions.loc[matches.id2].to_numpy()
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
    component = np.fromiter((find(int(node)) for node in left), dtype=np.int64, count=len(left))
    unique_components = np.unique(component)
    rng = np.random.default_rng(seed)
    validation_components = unique_components[
        rng.random(len(unique_components)) < validation_fraction
    ]
    valid = np.isin(component, validation_components)
    return SplitResult(
        name=f"component_seed_{seed}",
        train_mask=~valid,
        valid_mask=valid,
        notes="Exact reproduction of the existing connected-component split.",
    )


def build_split_scenarios(
    items: pd.DataFrame,
    matches: pd.DataFrame,
    validation_fraction: float = 0.15,
    seeds: Sequence[int] = (13, 42, 77, 2026),
) -> list[SplitResult]:
    left, right = pair_positions(items, matches)
    scenarios = [original_component_split(matches, validation_fraction, seed) for seed in seeds]
    exact_name_roots = _component_labels(len(items), left, right, items.name_signature.tolist())
    scenarios.append(
        split_from_groups(
            exact_name_roots[left], validation_fraction, 42,
            "exact_normalized_name_holdout", "Pair components merged by exact category + normalized name.",
        )
    )
    family_roots = _component_labels(len(items), left, right, items.family_signature.tolist())
    scenarios.append(
        split_from_groups(
            family_roots[left], validation_fraction, 42,
            "brand_model_family_holdout", "Pair components merged by conservative category + brand + model/id signature.",
        )
    )
    seller_signatures = ["|".join(value) if len(value) else "" for value in items.seller_values]
    if sum(bool(value) for value in seller_signatures) >= 1000:
        seller_roots = _component_labels(len(items), left, right, seller_signatures)
        scenarios.append(
            split_from_groups(
                seller_roots[left], validation_fraction, 42,
                "seller_holdout", "Pair components merged by exact normalized seller/store/vendor value.",
            )
        )
    return scenarios


def lexical_feature_columns(pair_table: pd.DataFrame) -> list[str]:
    return [
        "name_ratio", "name_token_set_ratio", "name_token_sort_ratio", "name_exact",
        "name_length_ratio", "name_numeric_jaccard", "name_numbers_both", "name_length_delta",
    ]


def macro_ap(target: np.ndarray, scores: np.ndarray, categories: np.ndarray) -> tuple[float, dict[str, float]]:
    per_category = {}
    for category in sorted(pd.unique(categories)):
        mask = categories == category
        per_category[str(category)] = float(average_precision_score(target[mask], scores[mask]))
    return float(np.mean(list(per_category.values()))), per_category


def evaluate_split(
    pair_table: pd.DataFrame,
    split: SplitResult,
    *,
    iterations: int = 1200,
    depth: int = 8,
    seed: int = 42,
) -> tuple[dict[str, object], pd.DataFrame, object]:
    """Train the same lexical CatBoost for one scenario."""

    from catboost import CatBoostClassifier

    feature_columns = lexical_feature_columns(pair_table)
    category_frame = pd.get_dummies(pair_table.category.astype(str), prefix="category", dtype=np.float32)
    features = pd.concat([pair_table[feature_columns].astype(np.float32), category_frame], axis=1)
    target = pair_table.target.to_numpy(np.int8)
    categories = pair_table.category.astype(str).to_numpy()
    counts = pd.Series(categories[split.train_mask]).value_counts()
    weights = np.asarray([1.0 / counts[value] for value in categories[split.train_mask]], dtype=np.float64)
    weights *= len(weights) / weights.sum()
    parameters = dict(
        loss_function="Logloss", eval_metric="AUC", iterations=iterations, depth=depth,
        learning_rate=0.06, l2_leaf_reg=3.0, random_seed=seed, verbose=False,
        allow_writing_files=False, thread_count=-1,
    )
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" or Path("/proc/driver/nvidia/version").exists():
        parameters.update(task_type="GPU", devices="0")
    model = CatBoostClassifier(**parameters)
    try:
        model.fit(
            features.loc[split.train_mask], target[split.train_mask], sample_weight=weights,
            eval_set=(features.loc[split.valid_mask], target[split.valid_mask]),
            early_stopping_rounds=120,
        )
        backend = parameters.get("task_type", "CPU")
    except Exception:
        if parameters.get("task_type") != "GPU":
            raise
        parameters.pop("task_type", None)
        parameters.pop("devices", None)
        model = CatBoostClassifier(**parameters)
        model.fit(
            features.loc[split.train_mask], target[split.train_mask], sample_weight=weights,
            eval_set=(features.loc[split.valid_mask], target[split.valid_mask]),
            early_stopping_rounds=120,
        )
        backend = "CPU fallback"
    scores = model.predict_proba(features.loc[split.valid_mask])[:, 1]
    score, per_category = macro_ap(target[split.valid_mask], scores, categories[split.valid_mask])
    valid = pair_table.loc[split.valid_mask, [
        "id1", "id2", "target", "category", "name_1", "name_2",
        "normalized_name_1", "normalized_name_2",
        *feature_columns,
        "number_conflict", "measure_match", "measure_conflict", "brand_match", "brand_conflict",
        "model_match", "model_conflict", "seller_match", "seller_conflict", "color_match", "color_conflict",
    ]].copy()
    valid["predict"] = scores
    train_ids = set(pair_table.loc[split.train_mask, "id1"]) | set(pair_table.loc[split.train_mask, "id2"])
    valid_ids = set(valid.id1) | set(valid.id2)
    report = {
        "split": split.name,
        "notes": split.notes,
        "train_pairs": int(split.train_mask.sum()),
        "validation_pairs": int(split.valid_mask.sum()),
        "validation_fraction": float(split.valid_mask.mean()),
        "train_positive_rate": float(target[split.train_mask].mean()),
        "validation_positive_rate": float(target[split.valid_mask].mean()),
        "overlapping_item_ids": len(train_ids & valid_ids),
        "macro_average_precision": score,
        "overall_average_precision": float(average_precision_score(target[split.valid_mask], scores)),
        "best_iteration": int(model.get_best_iteration()),
        "catboost_backend": backend,
        "per_category_average_precision": per_category,
    }
    return report, valid, model


def representation_overlap(items: pd.DataFrame, pair_table: pd.DataFrame, split: SplitResult) -> pd.DataFrame:
    train_ids = set(pair_table.loc[split.train_mask, "id1"]) | set(pair_table.loc[split.train_mask, "id2"])
    valid_ids = set(pair_table.loc[split.valid_mask, "id1"]) | set(pair_table.loc[split.valid_mask, "id2"])
    train = items[items.id.isin(train_ids)]
    valid = items[items.id.isin(valid_ids)]
    rows = []
    for label, column in (
        ("normalized_name", "normalized_name"),
        ("brand", "brand_values"),
        ("model_or_id", "model_values"),
        ("seller", "seller_values"),
        ("family_signature", "family_signature"),
    ):
        def explode_values(frame: pd.DataFrame) -> set[str]:
            values: set[str] = set()
            for value in frame[column]:
                if isinstance(value, (tuple, list, np.ndarray)):
                    values.update(item for item in value if item)
                elif value:
                    values.add(str(value))
            return values
        train_values, valid_values = explode_values(train), explode_values(valid)
        rows.append({
            "representation": label,
            "train_unique": len(train_values),
            "validation_unique": len(valid_values),
            "shared_unique": len(train_values & valid_values),
            "validation_seen_share": len(train_values & valid_values) / max(1, len(valid_values)),
        })
    return pd.DataFrame(rows)


def hard_negative_table(predictions: pd.DataFrame) -> pd.DataFrame:
    negatives = predictions[predictions.target.eq(0)].copy()
    negatives["hard_score"] = (
        0.45 * negatives.name_token_set_ratio
        + 0.20 * negatives.name_numeric_jaccard
        + 0.15 * negatives.brand_match
        + 0.15 * negatives.model_match
        + 0.05 * negatives.color_match
    )
    negatives["critical_conflict"] = (
        negatives.number_conflict.astype(bool)
        | negatives.measure_conflict.astype(bool)
        | negatives.model_conflict.astype(bool)
    )
    return negatives.sort_values(["hard_score", "predict"], ascending=False)


def confusion_columns(predictions: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    result = predictions.copy()
    predicted = result.predict.ge(threshold)
    result["is_fp"] = result.target.eq(0) & predicted
    result["is_fn"] = result.target.eq(1) & ~predicted
    result["error_type"] = np.select([result.is_fp, result.is_fn], ["FP", "FN"], default="correct")
    return result


def word_ngrams(text: str, min_n: int = 1, max_n: int = 3) -> set[str]:
    values = tokens(text)
    return {
        " ".join(values[index:index + n])
        for n in range(min_n, max_n + 1)
        for index in range(0, len(values) - n + 1)
    }


def char_ngrams(text: str, min_n: int = 3, max_n: int = 5) -> set[str]:
    compact = text.replace(" ", "")
    return {
        compact[index:index + n]
        for n in range(min_n, max_n + 1)
        for index in range(0, len(compact) - n + 1)
    }


def pattern_error_rates(
    predictions: pd.DataFrame,
    *,
    kind: str,
    min_support: int = 100,
    max_patterns_per_pair: int = 200,
) -> pd.DataFrame:
    """Calculate FPR/FNR for n-grams common to both product names."""

    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    builder = word_ngrams if kind == "word" else char_ngrams
    for row in predictions.itertuples(index=False):
        common = builder(row.normalized_name_1) & builder(row.normalized_name_2)
        if len(common) > max_patterns_per_pair:
            common = set(sorted(common, key=lambda value: (len(value), value))[:max_patterns_per_pair])
        for pattern in common:
            record = stats[pattern]
            if row.target == 0:
                record[0] += 1
                record[1] += int(row.is_fp)
            else:
                record[2] += 1
                record[3] += int(row.is_fn)
    records = []
    for pattern, (negatives, false_positives, positives, false_negatives) in stats.items():
        support = negatives + positives
        if support < min_support:
            continue
        records.append({
            "kind": kind,
            "pattern": pattern,
            "support": support,
            "negative_support": negatives,
            "positive_support": positives,
            "positive_rate": positives / support,
            "fpr": false_positives / negatives if negatives else np.nan,
            "fnr": false_negatives / positives if positives else np.nan,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        })
    if not records:
        return pd.DataFrame(columns=["kind", "pattern", "support", "fpr", "fnr"])
    return pd.DataFrame(records).sort_values(["false_positives", "support"], ascending=False)


def semantic_combination_rates(predictions: pd.DataFrame, min_support: int = 50) -> pd.DataFrame:
    signals = [
        "brand_match", "brand_conflict", "model_match", "model_conflict", "color_match", "color_conflict",
        "number_conflict", "measure_match", "measure_conflict", "seller_match", "seller_conflict",
    ]
    records = []
    for size in (1, 2):
        combinations: Iterable[tuple[str, ...]]
        if size == 1:
            combinations = ((value,) for value in signals)
        else:
            combinations = ((signals[i], signals[j]) for i in range(len(signals)) for j in range(i + 1, len(signals)))
        for combination in combinations:
            mask = np.ones(len(predictions), dtype=bool)
            for column in combination:
                mask &= predictions[column].astype(bool).to_numpy()
            selected = predictions.loc[mask]
            if len(selected) < min_support:
                continue
            negative = selected.target.eq(0)
            positive = ~negative
            records.append({
                "combination": " + ".join(combination),
                "support": len(selected),
                "positive_rate": float(selected.target.mean()),
                "negative_support": int(negative.sum()),
                "positive_support": int(positive.sum()),
                "fpr": float(selected.loc[negative, "is_fp"].mean()) if negative.any() else np.nan,
                "fnr": float(selected.loc[positive, "is_fn"].mean()) if positive.any() else np.nan,
                "mean_prediction": float(selected.predict.mean()),
            })
    return pd.DataFrame(records).sort_values(["fpr", "support"], ascending=False)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def reports_to_frame(reports: Sequence[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame([{key: value for key, value in report.items() if key != "per_category_average_precision"} for report in reports])
