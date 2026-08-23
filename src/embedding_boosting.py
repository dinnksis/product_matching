"""Three ablation experiments: lexical, Qwen embeddings, and attributes."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_pipeline import component_split


SPACE_RE = re.compile(r"\s+")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
FAMILY_PATTERNS = {
    "brand": re.compile(r"бренд|brand|производител"),
    "model": re.compile(r"модель|model|серия|линейка"),
    "identifier": re.compile(r"артикул|партномер|part.?number|sku|mpn|oem|код товара"),
    "size": re.compile(r"размер|длина|ширина|высота|диаметр|толщина|габарит"),
    "quantity": re.compile(r"количеств|комплект|упаков|штук|шт\.?$|объ.м|вес"),
    "color": re.compile(r"цвет|оттенок"),
    "material": re.compile(r"материал|состав|сырь"),
    "country": re.compile(r"страна|производств"),
    "seller_noise": re.compile(r"продав|магазин|поставщик|валюта|цена|достав|гарант"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("embedding_boosting")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(output_dir / "run.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def clean(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return SPACE_RE.sub(" ", str(value)).strip().casefold().replace("ё", "е")


def parse_attributes(raw: object) -> dict[str, str]:
    if not isinstance(raw, str) or not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, item in value.items():
        normalized_key, normalized_value = clean(key), clean(item)
        if normalized_key and normalized_value:
            result[normalized_key] = normalized_value
    return result


def family_for_key(key: str) -> str | None:
    for family, pattern in FAMILY_PATTERNS.items():
        if pattern.search(key):
            return family
    return None


def macro_ap(target: np.ndarray, scores: np.ndarray, categories: np.ndarray) -> tuple[float, dict[str, float]]:
    values = {}
    for category in sorted(pd.unique(categories)):
        mask = categories == category
        values[str(category)] = float(average_precision_score(target[mask], scores[mask]))
    return float(np.mean(list(values.values()))), values


def positions_for_pairs(items: pd.DataFrame, pairs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    lookup = pd.Series(np.arange(len(items), dtype=np.int32), index=items.id.to_numpy())
    return lookup.loc[pairs.id1].to_numpy(), lookup.loc[pairs.id2].to_numpy()


def name_features(names: np.ndarray, left: np.ndarray, right: np.ndarray, categories: np.ndarray) -> pd.DataFrame:
    records = []
    for index, (left_position, right_position) in enumerate(zip(left, right), start=1):
        first, second = clean(names[left_position]), clean(names[right_position])
        first_numbers, second_numbers = set(NUMBER_RE.findall(first)), set(NUMBER_RE.findall(second))
        union = first_numbers | second_numbers
        longest = max(len(first), len(second))
        records.append((
            fuzz.ratio(first, second) / 100.0,
            fuzz.token_set_ratio(first, second) / 100.0,
            fuzz.token_sort_ratio(first, second) / 100.0,
            float(first == second),
            min(len(first), len(second)) / longest if longest else 1.0,
            len(first_numbers & second_numbers) / max(1, len(union)),
            float(bool(first_numbers) and bool(second_numbers)),
            abs(len(first) - len(second)),
        ))
    frame = pd.DataFrame(records, columns=[
        "name_ratio", "name_token_set_ratio", "name_token_sort_ratio", "name_exact",
        "name_length_ratio", "name_numeric_jaccard", "name_numbers_both", "name_length_delta",
    ], dtype=np.float32)
    category_frame = pd.get_dummies(pd.Series(categories, name="category"), prefix="category", dtype=np.float32)
    return pd.concat([frame, category_frame], axis=1)


def encode_items(items: pd.DataFrame, config: dict[str, object], output_dir: Path, logger: logging.Logger) -> np.ndarray:
    cache_path = output_dir / "item_embeddings.f16.npy"
    ids_path = output_dir / "item_embedding_ids.npy"
    if cache_path.exists() and ids_path.exists():
        cached_ids = np.load(ids_path, allow_pickle=True)
        if np.array_equal(cached_ids, items.id.to_numpy()):
            logger.info("Reusing cached item embeddings: %s", cache_path)
            return np.load(cache_path, mmap_mode="r")

    import torch
    from sentence_transformers import SentenceTransformer

    model_name = str(config["embedding_model"])
    max_length = int(config["embedding_max_length"])
    dimension = int(config["embedding_dimension"])
    batch_size = int(config["embedding_batch_size"])
    devices = [f"cuda:{index}" for index in range(torch.cuda.device_count())] or ["cpu"]
    logger.info("Loading %s; devices=%s, max_length=%d, output_dimension=%d", model_name, devices, max_length, dimension)
    started = time.perf_counter()
    model = SentenceTransformer(model_name, model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = max_length
    texts = items.name.fillna("").astype(str).tolist()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
        device=devices if len(devices) > 1 else devices[0],
        truncate_dim=dimension,
    )
    embeddings = embeddings.astype(np.float32, copy=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= np.maximum(norms, 1e-12)
    np.save(cache_path, embeddings.astype(np.float16))
    np.save(ids_path, items.id.to_numpy())
    model_directory = output_dir / "qwen_embedding_model"
    model.save(str(model_directory))
    resolved_revision = getattr(model[0].auto_model.config, "_commit_hash", None)
    (output_dir / "embedding_model_identity.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "resolved_huggingface_revision": resolved_revision,
                "saved_directory": model_directory.name,
                "max_length": max_length,
                "output_dimension": dimension,
                "normalized": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Encoded and cached %,d items in %.1f minutes", len(items), (time.perf_counter() - started) / 60)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.load(cache_path, mmap_mode="r")


def embedding_pair_features(embeddings: np.ndarray, left: np.ndarray, right: np.ndarray) -> pd.DataFrame:
    first = np.asarray(embeddings[left], dtype=np.float32)
    second = np.asarray(embeddings[right], dtype=np.float32)
    absolute = np.abs(first - second)
    product = first * second
    dimension = first.shape[1]
    data = {
        "embedding_cosine": np.einsum("ij,ij->i", first, second),
        "embedding_l1_mean": absolute.mean(axis=1),
        "embedding_l2": np.sqrt(np.square(first - second).sum(axis=1)),
        "embedding_abs_max": absolute.max(axis=1),
    }
    for index in range(dimension):
        data[f"embedding_abs_{index:03d}"] = absolute[:, index]
        data[f"embedding_product_{index:03d}"] = product[:, index]
    return pd.DataFrame(data, dtype=np.float32)


def select_exact_keys(
    pairs: pd.DataFrame,
    categories: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    attributes: list[dict[str, str]],
    per_category: int,
    min_support: int,
) -> dict[str, list[str]]:
    statistics: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    targets = pairs.target.to_numpy(dtype=np.int8)
    for category, target, left_position, right_position in zip(categories, targets, left, right):
        first, second = attributes[left_position], attributes[right_position]
        for key in first.keys() & second.keys():
            record = statistics[(str(category), key)]
            record[0] += 1
            if first[key] == second[key]:
                record[1] += 1
                record[2] += int(target)
            else:
                record[3] += 1
                record[4] += int(target)
    ranked: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for (category, key), (both, equal, positive_equal, conflict, positive_conflict) in statistics.items():
        if both < min_support or not equal or not conflict:
            continue
        equal_rate = positive_equal / equal
        conflict_rate = positive_conflict / conflict
        score = abs(equal_rate - conflict_rate) * math.log1p(both)
        ranked[category].append((score, key))
    return {
        category: [key for _, key in sorted(values, reverse=True)[:per_category]]
        for category, values in ranked.items()
    }


def attribute_features(
    categories: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    attributes: list[dict[str, str]],
    selected_keys: dict[str, list[str]],
    per_category: int,
) -> pd.DataFrame:
    rows = []
    family_names = list(FAMILY_PATTERNS)
    for category, left_position, right_position in zip(categories, left, right):
        first, second = attributes[left_position], attributes[right_position]
        first_keys, second_keys = set(first), set(second)
        common, union = first_keys & second_keys, first_keys | second_keys
        equal = sum(first[key] == second[key] for key in common)
        conflict = len(common) - equal
        first_families: dict[str, set[str]] = defaultdict(set)
        second_families: dict[str, set[str]] = defaultdict(set)
        for key, value in first.items():
            if family := family_for_key(key):
                first_families[family].add(value)
        for key, value in second.items():
            if family := family_for_key(key):
                second_families[family].add(value)
        row = [
            len(common), len(common) / max(1, len(union)), equal, conflict,
            equal / max(1, len(common)), conflict / max(1, len(common)),
            abs(len(first) - len(second)), float(first == second),
        ]
        for family in family_names:
            first_values, second_values = first_families[family], second_families[family]
            row.extend((
                float(bool(first_values and second_values)),
                float(bool(first_values & second_values)),
                float(bool(first_values and second_values and not (first_values & second_values))),
            ))
        keys = selected_keys.get(str(category), [])
        for rank in range(per_category):
            key = keys[rank] if rank < len(keys) else None
            both = bool(key and key in first and key in second)
            row.extend((
                float(both),
                float(both and first[key] == second[key]),
                float(both and first[key] != second[key]),
                float(bool(key) and ((key in first) != (key in second))),
            ))
        rows.append(row)
    columns = [
        "attr_shared_keys", "attr_key_jaccard", "attr_equal_values", "attr_conflicting_values",
        "attr_equal_ratio", "attr_conflict_ratio", "attr_count_delta", "attr_exact",
    ]
    for family in family_names:
        columns.extend((f"{family}_both", f"{family}_match", f"{family}_conflict"))
    for rank in range(per_category):
        columns.extend((f"exact_key_{rank}_both", f"exact_key_{rank}_match", f"exact_key_{rank}_conflict", f"exact_key_{rank}_one_missing"))
    return pd.DataFrame(rows, columns=columns, dtype=np.float32)


def fit_experiment(
    name: str,
    features: pd.DataFrame,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    target: np.ndarray,
    categories: np.ndarray,
    config: dict[str, object],
    output_dir: Path,
    pair_ids: pd.DataFrame,
    logger: logging.Logger,
) -> dict[str, object]:
    from catboost import CatBoostClassifier

    experiment_dir = output_dir / name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    counts = pd.Series(categories[train_mask]).value_counts()
    weights = np.asarray([1.0 / counts[category] for category in categories[train_mask]], dtype=np.float64)
    weights *= len(weights) / weights.sum()
    parameters = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=int(config["catboost_iterations"]),
        depth=int(config["catboost_depth"]),
        learning_rate=float(config["catboost_learning_rate"]),
        l2_leaf_reg=3.0,
        random_seed=int(config["seed"]),
        verbose=50,
        allow_writing_files=False,
    )
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" or Path("/proc/driver/nvidia/version").exists():
        parameters.update(task_type="GPU", devices="0:1")
    logger.info(
        "Training %s: %s features, %s train, %s validation",
        name,
        f"{features.shape[1]:,}",
        f"{int(train_mask.sum()):,}",
        f"{int(valid_mask.sum()):,}",
    )
    started = time.perf_counter()
    model = CatBoostClassifier(**parameters)
    try:
        model.fit(
            features.loc[train_mask], target[train_mask], sample_weight=weights,
            eval_set=(features.loc[valid_mask], target[valid_mask]),
            early_stopping_rounds=int(config["catboost_early_stopping_rounds"]),
        )
        backend = parameters.get("task_type", "CPU")
    except Exception:
        if parameters.get("task_type") != "GPU":
            raise
        logger.exception("GPU CatBoost failed; retrying this experiment on CPU")
        parameters.pop("task_type", None)
        parameters.pop("devices", None)
        model = CatBoostClassifier(**parameters)
        model.fit(
            features.loc[train_mask], target[train_mask], sample_weight=weights,
            eval_set=(features.loc[valid_mask], target[valid_mask]),
            early_stopping_rounds=int(config["catboost_early_stopping_rounds"]),
        )
        backend = "CPU fallback"
    training_seconds = time.perf_counter() - started
    scores = model.predict_proba(features.loc[valid_mask])[:, 1]
    score, per_category = macro_ap(target[valid_mask], scores, categories[valid_mask])
    overall = float(average_precision_score(target[valid_mask], scores))
    model.save_model(experiment_dir / "model.cbm")
    pd.DataFrame({"feature": features.columns, "importance": model.get_feature_importance()}).sort_values("importance", ascending=False).to_csv(experiment_dir / "feature_importance.csv", index=False)
    predictions = pair_ids.loc[valid_mask, ["id1", "id2"]].copy()
    predictions["target"] = target[valid_mask]
    predictions["category"] = categories[valid_mask]
    predictions["predict"] = scores
    predictions.to_parquet(experiment_dir / "validation_predictions.parquet", index=False)
    report = {
        "experiment": name,
        "macro_average_precision": score,
        "overall_average_precision": overall,
        "per_category_average_precision": per_category,
        "training_seconds": training_seconds,
        "best_iteration": int(model.get_best_iteration()),
        "features": int(features.shape[1]),
        "catboost_backend": backend,
    }
    (experiment_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Finished %s: macro AP=%.6f, overall AP=%.6f, %.1fs", name, score, overall, training_seconds)
    return report


def main() -> None:
    args = parse_args()
    logger = configure_logging(args.output_dir)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    started = time.perf_counter()
    logger.info("Loading human-labelled data")
    items = pd.read_parquet(args.items, columns=["id", "name", "attributes", "category"])
    matches = pd.read_parquet(args.matches, columns=["id1", "id2", "target"])
    train, validation, diagnostics = component_split(matches, float(config["validation_fraction"]), int(config["seed"]))
    train_keys = pd.MultiIndex.from_frame(train[["id1", "id2"]])
    all_keys = pd.MultiIndex.from_frame(matches[["id1", "id2"]])
    train_mask = all_keys.isin(train_keys)
    valid_mask = ~train_mask
    if int(valid_mask.sum()) != len(validation):
        raise RuntimeError("Split reconstruction failed")
    left, right = positions_for_pairs(items, matches)
    categories = items.category.to_numpy()[left].astype(str)
    target = matches.target.to_numpy(dtype=np.int8)
    logger.info("Split: %s", diagnostics)

    names = items.name.fillna("").astype(str).to_numpy()
    lexical = name_features(names, left, right, categories)
    reports = [fit_experiment("01_names_lexical", lexical, train_mask, valid_mask, target, categories, config, args.output_dir, matches, logger)]

    embeddings = encode_items(items, config, args.output_dir, logger)
    embedding_features = embedding_pair_features(embeddings, left, right)
    names_embedding = pd.concat([lexical, embedding_features], axis=1)
    reports.append(fit_experiment("02_names_qwen_embedding", names_embedding, train_mask, valid_mask, target, categories, config, args.output_dir, matches, logger))

    logger.info("Parsing structured JSON attributes")
    attributes = [parse_attributes(raw) for raw in items.attributes]
    per_category = int(config["exact_keys_per_category"])
    selected_keys = select_exact_keys(
        matches.loc[train_mask].reset_index(drop=True), categories[train_mask], left[train_mask], right[train_mask],
        attributes, per_category, int(config["exact_key_min_support"]),
    )
    (args.output_dir / "selected_attribute_keys.json").write_text(json.dumps(selected_keys, ensure_ascii=False, indent=2), encoding="utf-8")
    structured = attribute_features(categories, left, right, attributes, selected_keys, per_category)
    all_features = pd.concat([names_embedding, structured], axis=1)
    reports.append(fit_experiment("03_names_qwen_attributes", all_features, train_mask, valid_mask, target, categories, config, args.output_dir, matches, logger))

    comparison = pd.DataFrame(reports).drop(columns="per_category_average_precision")
    comparison.to_csv(args.output_dir / "experiment_comparison.csv", index=False)
    per_category_frame = pd.DataFrame({report["experiment"]: report["per_category_average_precision"] for report in reports})
    per_category_frame.to_csv(args.output_dir / "per_category_comparison.csv")
    manifest = {
        "status": "complete",
        "total_seconds": time.perf_counter() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "config": config,
        "split": diagnostics.__dict__,
        "experiments": reports,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "COMPLETED").write_text("ok\n", encoding="utf-8")
    logger.info("All experiments complete in %.1f minutes", manifest["total_seconds"] / 60)


if __name__ == "__main__":
    main()
