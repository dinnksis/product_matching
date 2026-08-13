"""Second attribute ablation: normalized values, per-category models, rich embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
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
from src.embedding_boosting import (
    FAMILY_PATTERNS,
    attribute_features,
    clean,
    embedding_pair_features,
    family_for_key,
    fit_experiment,
    macro_ap,
    name_features,
    parse_attributes,
    positions_for_pairs,
    select_exact_keys,
)


MEASURE_RE = re.compile(r"(?<!\w)(\d+(?:[.,]\d+)?)\s*(мл|ml|л|l|г|гр|g|кг|kg|мм|mm|см|cm|м|m|шт)(?!\w)")
MEASURE_FACTORS = {
    "мл": ("volume_ml", 1.0), "ml": ("volume_ml", 1.0), "л": ("volume_ml", 1000.0), "l": ("volume_ml", 1000.0),
    "г": ("weight_g", 1.0), "гр": ("weight_g", 1.0), "g": ("weight_g", 1.0), "кг": ("weight_g", 1000.0), "kg": ("weight_g", 1000.0),
    "мм": ("length_mm", 1.0), "mm": ("length_mm", 1.0), "см": ("length_mm", 10.0), "cm": ("length_mm", 10.0), "м": ("length_mm", 1000.0), "m": ("length_mm", 1000.0),
    "шт": ("count", 1.0),
}
FAMILY_PRIORITY = {name: index for index, name in enumerate(("identifier", "brand", "model", "size", "quantity", "color", "country", "material", "seller_noise"))}


def measures(value: str) -> set[str]:
    result = set()
    for raw_number, unit in MEASURE_RE.findall(value):
        kind, factor = MEASURE_FACTORS[unit]
        number = round(float(raw_number.replace(",", ".")) * factor, 4)
        result.add(f"{kind}={number:g}")
    return result


def token_set(value: str) -> set[str]:
    return {token for token in re.findall(r"[\w.-]+", value) if len(token) >= 2}


def enhanced_attribute_features(categories, left, right, attributes, selected_keys, per_category):
    base = attribute_features(categories, left, right, attributes, selected_keys, per_category)
    rows = []
    for left_position, right_position in zip(left, right):
        first, second = attributes[left_position], attributes[right_position]
        common = first.keys() & second.keys()
        fuzzy = [fuzz.token_set_ratio(first[key], second[key]) / 100.0 for key in common]
        first_tokens = token_set(" ".join(first.values()))
        second_tokens = token_set(" ".join(second.values()))
        first_measures = measures(" ".join(first.values()))
        second_measures = measures(" ".join(second.values()))
        family_rows = []
        for family in FAMILY_PATTERNS:
            first_values = " ".join(value for key, value in first.items() if family_for_key(key) == family)
            second_values = " ".join(value for key, value in second.items() if family_for_key(key) == family)
            if first_values and second_values:
                family_rows.append(fuzz.token_set_ratio(first_values, second_values) / 100.0)
            else:
                family_rows.append(0.0)
        rows.append([
            float(np.mean(fuzzy)) if fuzzy else 0.0,
            float(np.max(fuzzy)) if fuzzy else 0.0,
            len(first_tokens & second_tokens) / max(1, len(first_tokens | second_tokens)),
            len(first_measures & second_measures) / max(1, len(first_measures | second_measures)),
            float(bool(first_measures and second_measures and not (first_measures & second_measures))),
            *family_rows,
        ])
    columns = ["attr_fuzzy_mean", "attr_fuzzy_max", "attr_token_jaccard", "measure_jaccard", "measure_conflict"] + [f"{family}_fuzzy" for family in FAMILY_PATTERNS]
    return pd.concat([base, pd.DataFrame(rows, columns=columns, dtype=np.float32)], axis=1)


def rich_text(name: str, attributes: dict[str, str]) -> str:
    fields = []
    for key, value in sorted(
        attributes.items(),
        key=lambda item: (FAMILY_PRIORITY.get(family_for_key(item[0]) or "", 50), item[0]),
    ):
        if family_for_key(key) == "seller_noise":
            continue
        fields.append(f"{key}: {value}")
    return "Название: " + str(name) + "\n" + "\n".join(fields)


def encode_rich(items, config, model_path: Path, output_dir: Path, logger):
    import torch
    from sentence_transformers import SentenceTransformer

    cache = output_dir / "rich_item_embeddings.f16.npy"
    ids_file = output_dir / "rich_item_embedding_ids.npy"
    if cache.exists() and ids_file.exists() and np.array_equal(np.load(ids_file, allow_pickle=True), items.id.to_numpy()):
        return np.load(cache, mmap_mode="r")
    model = SentenceTransformer(str(model_path), model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = int(config["embedding_max_length"])
    parsed = [parse_attributes(raw) for raw in items.attributes]
    texts = [rich_text(name, attrs) for name, attrs in zip(items.name.fillna(""), parsed)]
    devices = [f"cuda:{index}" for index in range(torch.cuda.device_count())] or ["cpu"]
    started = time.perf_counter()
    result = model.encode(
        texts, batch_size=int(config["embedding_batch_size"]), show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=False,
        device=devices if len(devices) > 1 else devices[0], truncate_dim=int(config["embedding_dimension"]),
    ).astype(np.float32, copy=False)
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
    np.save(cache, result.astype(np.float16)); np.save(ids_file, items.id.to_numpy())
    logger.info("Rich embeddings encoded in %.1f minutes", (time.perf_counter() - started) / 60)
    return np.load(cache, mmap_mode="r")


def category_models(features, train_mask, valid_mask, target, categories, config, output_dir, pair_ids, logger):
    from catboost import CatBoostClassifier

    directory = output_dir / "05_per_category_attributes_v2"
    directory.mkdir(parents=True, exist_ok=True)
    scores = np.zeros(valid_mask.sum(), dtype=np.float32)
    valid_positions = np.flatnonzero(valid_mask)
    reports = {}
    for category in sorted(pd.unique(categories)):
        train_index = np.flatnonzero(train_mask & (categories == category))
        valid_index = np.flatnonzero(valid_mask & (categories == category))
        model = CatBoostClassifier(
            loss_function="Logloss", eval_metric="AUC", iterations=int(config["catboost_iterations"]),
            depth=int(config["catboost_depth"]), learning_rate=float(config["catboost_learning_rate"]),
            l2_leaf_reg=4.0, random_seed=int(config["seed"]), verbose=False,
            task_type="GPU", devices="0:1", allow_writing_files=False,
        )
        model.fit(features.iloc[train_index], target[train_index], eval_set=(features.iloc[valid_index], target[valid_index]), early_stopping_rounds=int(config["catboost_early_stopping_rounds"]))
        category_scores = model.predict_proba(features.iloc[valid_index])[:, 1]
        scores[np.searchsorted(valid_positions, valid_index)] = category_scores
        safe_name = hashlib.sha1(str(category).encode("utf-8")).hexdigest()[:12]
        model_file = f"category_{safe_name}.cbm"
        model.save_model(directory / model_file)
        reports[str(category)] = {"ap": float(average_precision_score(target[valid_index], category_scores)), "best_iteration": int(model.get_best_iteration()), "model_file": model_file}
        logger.info("Category %s AP=%.6f", category, reports[str(category)]["ap"])
    score, per_category = macro_ap(target[valid_mask], scores, categories[valid_mask])
    predictions = pair_ids.loc[valid_mask, ["id1", "id2"]].copy(); predictions["target"] = target[valid_mask]; predictions["category"] = categories[valid_mask]; predictions["predict"] = scores
    predictions.to_parquet(directory / "validation_predictions.parquet", index=False)
    report = {"experiment": "05_per_category_attributes_v2", "macro_average_precision": score, "overall_average_precision": float(average_precision_score(target[valid_mask], scores)), "per_category_average_precision": per_category, "category_models": reports}
    (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True); parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--previous-output", type=Path, required=True)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(args.output_dir / "run.log", encoding="utf-8")])
    logger = logging.getLogger("attribute_v2"); config = json.loads(args.config.read_text(encoding="utf-8")); started = time.perf_counter()
    items = pd.read_parquet(args.items, columns=["id", "name", "attributes", "category"]); matches = pd.read_parquet(args.matches, columns=["id1", "id2", "target"])
    train, validation, diagnostics = component_split(matches, float(config["validation_fraction"]), int(config["seed"]))
    all_keys = pd.MultiIndex.from_frame(matches[["id1", "id2"]]); train_mask = all_keys.isin(pd.MultiIndex.from_frame(train[["id1", "id2"]])); valid_mask = ~train_mask
    left, right = positions_for_pairs(items, matches); categories = items.category.to_numpy()[left].astype(str); target = matches.target.to_numpy(np.int8)
    old_ids = np.load(args.previous_output / "item_embedding_ids.npy", allow_pickle=True)
    if not np.array_equal(old_ids, items.id.to_numpy()): raise RuntimeError("Previous embedding ids do not match items")
    old_embeddings = np.load(args.previous_output / "item_embeddings.f16.npy", mmap_mode="r")
    lexical = name_features(items.name.fillna("").astype(str).to_numpy(), left, right, categories); old_pair = embedding_pair_features(old_embeddings, left, right)
    attributes = [parse_attributes(raw) for raw in items.attributes]; per_category = int(config["exact_keys_per_category"])
    selected = select_exact_keys(matches.loc[train_mask].reset_index(drop=True), categories[train_mask], left[train_mask], right[train_mask], attributes, per_category, int(config["exact_key_min_support"]))
    (args.output_dir / "selected_attribute_keys.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    attrs_v2 = enhanced_attribute_features(categories, left, right, attributes, selected, per_category)
    common = pd.concat([lexical, old_pair, attrs_v2], axis=1)
    reports = [fit_experiment("04_global_attributes_v2", common, train_mask, valid_mask, target, categories, config, args.output_dir, matches, logger)]
    reports.append(category_models(common, train_mask, valid_mask, target, categories, config, args.output_dir, matches, logger))
    model_candidates = list(args.previous_output.glob("qwen_embedding_model"))
    if len(model_candidates) != 1: raise RuntimeError(f"Expected Qwen model in previous output, found {model_candidates}")
    rich_embeddings = encode_rich(items, config, model_candidates[0], args.output_dir, logger); rich_pair = embedding_pair_features(rich_embeddings, left, right)
    rich_features = pd.concat([lexical, rich_pair, attrs_v2], axis=1)
    reports.append(fit_experiment("06_rich_embedding_attributes_v2", rich_features, train_mask, valid_mask, target, categories, config, args.output_dir, matches, logger))
    pd.DataFrame([{k: v for k, v in report.items() if k not in {"per_category_average_precision", "category_models"}} for report in reports]).to_csv(args.output_dir / "experiment_comparison.csv", index=False)
    (args.output_dir / "manifest.json").write_text(json.dumps({"status": "complete", "seconds": time.perf_counter()-started, "split": diagnostics.__dict__, "reports": reports}, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "COMPLETED").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__": main()
