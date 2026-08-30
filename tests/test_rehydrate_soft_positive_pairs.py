from __future__ import annotations

import copy
import unittest

from scripts.rehydrate_soft_positive_pairs import (
    EXPECTED_QUALITY_CHECKS,
    sanitize_generated_response,
    valid_reference_brand,
    validate_response,
    value_semantic_issues,
)


class RehydrateSoftPositivePairsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.task = {
            "category": "Обувь",
            "product_type": "ботинки",
            "composition_index": 1,
            "application": {
                "generation_rule_id": "rule-1",
                "concept": "width",
                "attribute_key": "Ширина колодки",
                "original_value": "6",
                "new_value": "6F",
            },
            "rule": {
                "generation_rule_id": "rule-1",
                "concept": "width",
                "required_attribute_key": "Ширина колодки",
            },
        }
        self.references = {
            "allowed_observed_brands": ["salomon"],
            "content_references": [
                {
                    "name": "реальные зимние ботинки salomon мужские черные",
                    "attributes": {
                        "Тип товара": "ботинки",
                        "Бренд": "salomon",
                        "Сезон": "зима",
                        "Материал верха": "кожа",
                        "Цвет": "черный",
                    },
                }
            ],
            "seller_pair_template": {
                "pair_id": "rp-test",
                "item_a": {
                    "name": "мужские зимние ботинки salomon из натуральной кожи",
                    "attributes": {
                        "Тип товара": "ботинки",
                        "Бренд": "salomon",
                        "Сезон": "зима",
                        "Материал верха": "натуральная кожа",
                        "Пол": "мужской",
                    },
                },
                "item_b": {
                    "name": "salomon ботинки мужские кожаные для зимы черные",
                    "attributes": {
                        "Вид товара": "ботинки",
                        "Бренд": "salomon",
                        "Сезон": "зимний",
                        "Материал": "кожа",
                        "Цвет": "черный",
                    },
                },
            },
        }

    def response(self) -> dict:
        return {
            "item_a": {
                "name": "мужские зимние ботинки salomon кожаные ширина шесть",
                "attributes": {
                    "Тип товара": "ботинки",
                    "Бренд": "salomon",
                    "Сезон": "зима",
                    "Материал верха": "натуральная кожа",
                    "Ширина колодки": "6",
                    "Пол": "мужской",
                },
                "category": "Обувь",
            },
            "item_b": {
                "name": "кожаные ботинки salomon мужские зимние ширина 6f",
                "attributes": {
                    "Тип товара": "ботинки",
                    "Бренд": "salomon",
                    "Сезон": "зима",
                    "Материал": "натуральная кожа",
                    "Ширина колодки": "6F",
                    "Цвет": "черный",
                    "Назначение": "повседневные",
                },
                "category": "Обувь",
            },
            "application": {
                "generation_rule_id": "rule-1",
                "concept": "width",
                "canonical_attribute_key": "Ширина колодки",
                "attribute_key_a": "Ширина колодки",
                "attribute_key_b": "Ширина колодки",
                "original_value": "6",
                "new_value": "6F",
                "values_repaired": False,
                "repair_reason": "",
            },
            "identity_facts": ["бренд salomon", "зимний сезон"],
            "quality_checks": {key: True for key in EXPECTED_QUALITY_CHECKS},
        }

    def test_accepts_rich_asymmetric_pair(self) -> None:
        value, errors, metrics = validate_response(
            self.response(), self.task, self.references
        )
        self.assertIsNotNone(value, errors)
        self.assertEqual(errors, [])
        self.assertGreaterEqual(metrics["attribute_count_a"], 5)
        self.assertGreaterEqual(metrics["attribute_count_b"], 5)

    def test_rejects_invented_brand(self) -> None:
        response = self.response()
        response["item_b"]["attributes"]["Бренд"] = "фиолетовый"
        value, errors, _ = validate_response(response, self.task, self.references)
        self.assertIsNone(value)
        self.assertIn("invented_or_changed_brand", errors)

    def test_brand_rule_rejects_color_value(self) -> None:
        self.assertIn(
            "brand_is_color",
            value_semantic_issues("brand", "Бренд", "фиолетовый"),
        )
        self.assertIn(
            "brand_is_country",
            value_semantic_issues("producer", "Производитель", "Швейцария"),
        )
        self.assertIn(
            "type_value_is_color",
            value_semantic_issues("type", "Вид товара", "фиолетовая"),
        )

    def test_rejects_identical_attribute_schema(self) -> None:
        response = self.response()
        response["item_b"]["attributes"] = copy.deepcopy(
            response["item_a"]["attributes"]
        )
        response["item_b"]["attributes"]["Ширина колодки"] = "6F"
        value, errors, _ = validate_response(response, self.task, self.references)
        self.assertIsNone(value)
        self.assertIn("attribute_key_sets_identical", errors)

    def test_reference_brand_filter(self) -> None:
        self.assertTrue(valid_reference_brand("Salomon"))
        self.assertFalse(valid_reference_brand("фиолетовый"))
        self.assertFalse(valid_reference_brand("Россия"))
        self.assertFalse(valid_reference_brand("нет"))
        self.assertFalse(valid_reference_brand("12345"))

    def test_sanitizes_explicit_identifiers(self) -> None:
        response = self.response()
        response["item_a"]["name"] += " арт. 12345"
        response["item_a"]["attributes"]["Артикул производителя"] = "12345"
        response["item_a"]["attributes"]["Код изделия"] = "ABC-42"
        cleaned = sanitize_generated_response(response)
        self.assertNotIn("12345", cleaned["item_a"]["name"])
        self.assertNotIn("Артикул производителя", cleaned["item_a"]["attributes"])
        self.assertNotIn("Код изделия", cleaned["item_a"]["attributes"])


if __name__ == "__main__":
    unittest.main()
