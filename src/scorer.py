from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_NON_WORD = re.compile(r"[^\w]+", flags=re.UNICODE)


def normalize_text(series: pd.Series) -> pd.Series:
    """Cheap, deterministic normalization suitable for a fallback baseline."""
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace("ё", "е", regex=False)
        .str.replace(_NON_WORD, " ", regex=True)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


@dataclass(frozen=True)
class HeuristicScorer:
    name_weight: float = 0.75
    attributes_weight: float = 0.20
    category_weight: float = 0.05

    @classmethod
    def from_json(cls, path: Path) -> "HeuristicScorer":
        with path.open("r", encoding="utf-8") as stream:
            config = json.load(stream)
        return cls(**config)

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        """Return continuous scores; replace this method for a trained model."""
        name1, name2 = normalize_text(pairs["name_1"]), normalize_text(pairs["name_2"])
        attr1 = normalize_text(pairs["attributes_1"])
        attr2 = normalize_text(pairs["attributes_2"])
        cat1 = pairs["category_1"].fillna("").astype(str)
        cat2 = pairs["category_2"].fillna("").astype(str)

        name_equal = name1.eq(name2) & name1.ne("")
        attr_equal = attr1.eq(attr2) & attr1.ne("")
        category_equal = cat1.eq(cat2) & cat1.ne("")

        # A small length-ratio signal separates obviously different names while
        # preserving raw continuous predictions required by PR-AUC.
        len1 = name1.str.len().to_numpy(dtype=np.float32)
        len2 = name2.str.len().to_numpy(dtype=np.float32)
        length_ratio = np.minimum(len1, len2) / np.maximum(np.maximum(len1, len2), 1)
        scores = (
            self.name_weight * name_equal.to_numpy(dtype=np.float32)
            + self.attributes_weight * attr_equal.to_numpy(dtype=np.float32)
            + self.category_weight * category_equal.to_numpy(dtype=np.float32)
            + np.float32(0.01) * length_ratio
        )
        return scores.astype(np.float32, copy=False)

