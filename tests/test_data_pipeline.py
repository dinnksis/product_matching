from __future__ import annotations

import json
import unittest

import pandas as pd

from src.data_pipeline import component_split, serialize_attributes, serialize_product


class DataPipelineTest(unittest.TestCase):
    def test_priority_and_empty_values(self) -> None:
        raw = json.dumps({"прочее": "x", "модель": "M-1", "бренд": "A", "пусто": ""})
        lines = serialize_attributes(raw, max_chars=None).splitlines()
        self.assertEqual(lines[:2], ["бренд: A", "модель: M-1"])
        self.assertNotIn("пусто: ", lines)

    def test_product_serialization(self) -> None:
        row = pd.Series({"name": "  A   B ", "category": "Demo", "attributes": "{}"})
        self.assertEqual(serialize_product(row), "Категория: Demo\nНазвание: A B")

    def test_product_attributes_are_flat_lines(self) -> None:
        row = pd.Series(
            {
                "name": "Phone",
                "category": "Electronics",
                "attributes": json.dumps({"цвет": "чёрный", "бренд": "Acme"}),
            }
        )
        self.assertEqual(
            serialize_product(row),
            "Категория: Electronics\nНазвание: Phone\nбренд: Acme\nцвет: чёрный",
        )

    def test_component_split_has_no_item_leakage(self) -> None:
        pairs = pd.DataFrame(
            {"id1": [1, 2, 10, 20], "id2": [2, 3, 11, 21], "target": [1, 0, 1, 0]}
        )
        train, val, diagnostics = component_split(pairs, validation_fraction=0.5, seed=7)
        train_ids = set(train.id1) | set(train.id2)
        val_ids = set(val.id1) | set(val.id2)
        self.assertFalse(train_ids & val_ids)
        self.assertEqual(diagnostics.overlapping_items, 0)


if __name__ == "__main__":
    unittest.main()
