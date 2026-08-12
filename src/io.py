from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

ITEM_COLUMNS = ["id", "name", "attributes", "category"]
MATCH_COLUMNS = ["id1", "id2"]


class Scorer(Protocol):
    def predict(self, pairs: pd.DataFrame) -> np.ndarray: ...


def _require_columns(frame: pd.DataFrame, required: list[str], source: Path) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing required columns: {missing}")


def load_inputs(items_path: Path, matches_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Reading items from %s", items_path)
    items = pd.read_parquet(items_path, columns=ITEM_COLUMNS)
    logging.info("Reading pairs from %s", matches_path)
    matches = pd.read_parquet(matches_path, columns=MATCH_COLUMNS)
    _require_columns(items, ITEM_COLUMNS, items_path)
    _require_columns(matches, MATCH_COLUMNS, matches_path)
    if items["id"].duplicated().any():
        raise ValueError(f"{items_path}: item ids must be unique")
    return items, matches


def build_pairs(items: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    item_index = items.set_index("id", verify_integrity=True)
    left = item_index.reindex(matches["id1"].to_numpy()).add_suffix("_1")
    right = item_index.reindex(matches["id2"].to_numpy()).add_suffix("_2")
    left.index = matches.index
    right.index = matches.index
    pairs = pd.concat([left, right], axis=1)
    missing = pairs["name_1"].isna() | pairs["name_2"].isna()
    if missing.any():
        examples = matches.loc[missing, MATCH_COLUMNS].head(5).to_dict("records")
        raise ValueError(f"Pairs reference missing item ids; examples: {examples}")
    return pairs


def validate_predictions(matches: pd.DataFrame, predictions: np.ndarray) -> np.ndarray:
    values = np.asarray(predictions).reshape(-1)
    if len(values) != len(matches):
        raise ValueError(f"Expected {len(matches)} predictions, got {len(values)}")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("Predictions must be numeric")
    if not np.isfinite(values).all():
        raise ValueError("Predictions contain NaN or infinity")
    return values


def run_inference(
    items_path: Path, matches_path: Path, output_path: Path, scorer: Scorer
) -> None:
    items, matches = load_inputs(items_path, matches_path)
    logging.info("Scoring %d pairs over %d items", len(matches), len(items))
    pairs = build_pairs(items, matches)
    predictions = validate_predictions(matches, scorer.predict(pairs))
    output = matches[MATCH_COLUMNS].copy()
    output["predict"] = predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    logging.info("Saved %d predictions to %s", len(output), output_path)

