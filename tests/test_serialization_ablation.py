from __future__ import annotations

import json
import unittest

from src.serialization_ablation import (
    normalize_text,
    parse_attributes,
    select_frequent_keys,
    serialize_product,
)


class SerializationAblationTest(unittest.TestCase):
    def test_normalization_preserves_model_codes_and_digits(self) -> None:
        value = normalize_text("  RTX 4070 / SM-S928B / WH-1000XM5, 256 ГБ  ")
        self.assertEqual(value, "rtx 4070 / sm-s928b / wh-1000xm5, 256 gb")

    def test_unambiguous_units_are_canonicalized(self) -> None:
        self.assertEqual(normalize_text("500 МЛ, 2 кг, 15 Вт"), "500 ml, 2 kg, 15 w")

    def test_variants_have_no_special_tokens(self) -> None:
        attributes = parse_attributes(json.dumps({"Бренд": "Samsung", "Память": "256 ГБ", "Редкий ключ": "X"}))
        ranks = {"бренд": 0, "память": 1, "редкий ключ": 2}
        frequent = {"бренд", "память"}
        self.assertEqual(
            serialize_product("Телефон", attributes, "S0_TITLE", frequent, ranks),
            "телефон",
        )
        self.assertEqual(
            serialize_product("Телефон", attributes, "S1_KEY_VALUE", frequent, ranks),
            "телефон. бренд: samsung. память: 256 gb. редкий ключ: x",
        )
        self.assertEqual(
            serialize_product("Телефон", attributes, "S2_VALUES_ONLY", frequent, ranks),
            "телефон. samsung. 256 gb. x",
        )
        self.assertEqual(
            serialize_product("Телефон", attributes, "S3_HYBRID", frequent, ranks),
            "телефон. бренд: samsung. память: 256 gb. x",
        )

    def test_frequency_threshold_uses_only_supplied_train_items(self) -> None:
        attributes = [
            [("common", "a"), ("rare", "x")],
            [("common", "b")],
            [("common", "c")],
        ]
        frequent, table, summary = select_frequent_keys(
            attributes,
            {
                "strategy": "occurrence_coverage",
                "target_coverage": 0.7,
                "minimum_item_support": 2,
                "maximum_frequent_keys": 10,
            },
        )
        self.assertEqual(frequent, {"common"})
        self.assertEqual(summary["selected_item_support_threshold"], 3)
        self.assertEqual(int(table.loc[table.attribute_name == "rare", "is_frequent"].iloc[0]), 0)


if __name__ == "__main__":
    unittest.main()
