from __future__ import annotations

import json
import unittest

from item_pipeline.normalization import (
    extract_subtype,
    normalize_text,
    parse_attributes,
    title_attribute_token_coverage,
)


class NormalizationTest(unittest.TestCase):
    def test_parse_attributes_preserves_original_keys_and_drops_empty_values(self) -> None:
        raw = json.dumps(
            {"Тип товара": " Картридж ", "цвет": "", "модель": "PG-440XL"},
            ensure_ascii=False,
        )
        self.assertEqual(
            parse_attributes(raw),
            {"Тип товара": "Картридж", "модель": "PG-440XL"},
        )

    def test_extract_subtype_prefers_type_and_has_title_fallback(self) -> None:
        self.assertEqual(
            extract_subtype("длинное название", {"тип": "Картридж; расходник"}),
            "картридж",
        )
        self.assertEqual(
            extract_subtype("Canon картридж PG-440", {}),
            "__title__:canon картридж",
        )

    def test_title_attribute_coverage_is_partial(self) -> None:
        coverage = title_attribute_token_coverage(
            "canon картридж pg-440xl черный",
            {"бренд": "Canon", "модель": "PG-440XL"},
        )
        self.assertEqual(coverage, 0.5)
        self.assertEqual(normalize_text("  Ёлка  XL "), "елка xl")


if __name__ == "__main__":
    unittest.main()
