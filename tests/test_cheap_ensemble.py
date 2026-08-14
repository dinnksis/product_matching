from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.cheap_ensemble import (
    build_pair_features,
    extract_codes,
    extract_numbers,
    fit_hashed_char_idf,
    prepare_item_records,
)


def test_codes_and_plain_numbers_are_separated() -> None:
    text = "macbook air m3 16/512 sm-s928b rtx4070"
    assert extract_numbers(text) == frozenset({"16", "512"})
    assert {"sms928b", "rtx4070"}.issubset(extract_codes(text))


def test_numeric_conflict_is_contextual_and_one_sided_code_is_not_conflict() -> None:
    raw_items = pd.DataFrame(
        [
            {
                "id": 1,
                "name": "MacBook Air M3 16/512 Midnight",
                "attributes": json.dumps({"Бренд": "Apple", "Цвет": "Midnight"}),
                "category": "Электроника",
            },
            {
                "id": 2,
                "name": "MacBook Air M3 8/512 Midnight",
                "attributes": json.dumps({"Бренд": "Apple", "Цвет": "Midnight"}),
                "category": "Электроника",
            },
            {
                "id": 3,
                "name": "Apple MUW43RU/A",
                "attributes": json.dumps({"Бренд": "Apple"}),
                "category": "Электроника",
            },
        ]
    )
    items = prepare_item_records(raw_items)
    pairs = pd.DataFrame({"id1": [1, 1], "id2": [2, 3]})
    idf, _ = fit_hashed_char_idf(raw_items["name"], n_features=1024, batch_size=2)
    features = build_pair_features(items, pairs, [0.9, 0.2], idf)

    assert features.loc[0, "numeric_context_conflict_count"] >= 1
    assert features.loc[0, "slash_spec_conflict"] == 1
    assert features.loc[1, "sku_human_asymmetry"] == 1
    assert features.loc[1, "code_conflict"] == 0


def test_hashed_char_tfidf_similarity_is_bounded() -> None:
    raw_items = pd.DataFrame(
        [
            {"id": 1, "name": "Samsung Galaxy S24", "attributes": "{}", "category": "x"},
            {"id": 2, "name": "Samsung Galaxy S24", "attributes": "{}", "category": "x"},
        ]
    )
    items = prepare_item_records(raw_items)
    idf, documents = fit_hashed_char_idf(raw_items["name"], n_features=1024)
    features = build_pair_features(
        items, pd.DataFrame({"id1": [1], "id2": [2]}), [0.5], idf
    )
    assert documents == 2
    assert np.isclose(features.loc[0, "title_char_tfidf_cosine"], 1.0)
