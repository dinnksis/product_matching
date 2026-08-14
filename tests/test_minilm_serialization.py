from __future__ import annotations

import unittest

from src.minilm_serialization import (
    attribute_frequency,
    normalize_text,
    parse_attributes,
    serialize_product,
)


class MiniLmSerializationTest(unittest.TestCase):
    def test_normalization_matches_ablation(self) -> None:
        self.assertEqual(normalize_text("  Ёлка 512 ГБ  "), "елка 512 gb")

    def test_s1_serializes_title_then_ranked_key_value_fields(self) -> None:
        attributes = parse_attributes(
            '{"Цвет": " Красный ", "Память": ["512 ГБ", "1 ТБ"]}'
        )
        result = serialize_product(
            "  СМАРТФОН Ё  ",
            attributes,
            "S1_KEY_VALUE",
            set(),
            {"память": 0, "цвет": 1},
        )
        self.assertEqual(
            result,
            "смартфон е. память: 1 tb. память: 512 gb. цвет: красный",
        )

    def test_other_ablation_variants_use_the_same_field_order(self) -> None:
        attributes = [("brand", "acme"), ("color", "black")]
        rank = {"color": 0, "brand": 1}
        self.assertEqual(
            serialize_product("Phone", attributes, "S2_VALUES_ONLY", set(), rank),
            "phone. black. acme",
        )
        self.assertEqual(
            serialize_product(
                "Phone", attributes, "S3_HYBRID", {"brand"}, rank
            ),
            "phone. black. brand: acme",
        )

    def test_attribute_frequency_order_is_deterministic(self) -> None:
        rows = attribute_frequency(
            [
                [("color", "red"), ("brand", "a")],
                [("brand", "b"), ("brand", "c")],
                [("size", "m"), ("color", "blue")],
            ]
        )
        self.assertEqual(
            [row.attribute_name for row in rows], ["brand", "color", "size"]
        )


if __name__ == "__main__":
    unittest.main()
