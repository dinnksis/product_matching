from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_qwen_semantic_extraction as runner  # noqa: E402


def value(raw: str, normalized: str | None = None, unit: str | None = None) -> dict:
    return {
        "raw_value": raw,
        "normalized_value": normalized if normalized is not None else raw.casefold(),
        "unit": unit,
    }


def title_side(raw: str, normalized: str | None = None) -> dict:
    return {
        "value": value(raw, normalized),
        "evidence": [
            {"source": "title", "raw_attribute_name": None, "raw_fragment": raw}
        ],
    }


def attribute_side(name: str, raw: str, normalized: str | None = None) -> dict:
    return {
        "value": value(raw, normalized),
        "evidence": [
            {"source": "attribute", "raw_attribute_name": name, "raw_fragment": raw}
        ],
    }


def semantic_fact(
    concept: str,
    relation: str,
    side_a: dict | None,
    side_b: dict | None,
    anchor_type: str | None = None,
) -> dict:
    return {
        "concept": concept,
        "relation": relation,
        "anchor_type": anchor_type,
        "a": side_a,
        "b": side_b,
        "direction": None,
        "confidence": "high",
    }


class QwenSemanticExtractionV14Test(unittest.TestCase):
    def errors(self, fact: dict, payload: dict, profile: str = "v1_4") -> list[str]:
        return runner.semantic_validation_errors(
            {"semantic_facts": [fact]}, payload, profile
        )

    def test_missing_is_rejected_when_known_value_occurs_on_missing_side(self) -> None:
        fact = semantic_fact("scent", "missing_b", title_side("лимон"), None)
        payload = {
            "category": "Бытовая химия",
            "item_a": {"title": "Средство лимон", "attributes": {}},
            "item_b": {"title": "Гель для посуды, лимон", "attributes": {}},
        }
        self.assertTrue(any("missing value is present" in error for error in self.errors(fact, payload)))

    def test_absence_marker_is_not_a_known_brand_value(self) -> None:
        fact = semantic_fact(
            "brand", "different_value", title_side("Золушка"), attribute_side("бренд", "нет бренда")
        )
        payload = {
            "category": "Бытовая химия",
            "item_a": {"title": "Золушка", "attributes": {}},
            "item_b": {"title": "Средство", "attributes": {"бренд": "нет бренда"}},
        }
        self.assertTrue(any("absence marker" in error for error in self.errors(fact, payload)))

    def test_country_abbreviation_is_not_a_brand(self) -> None:
        fact = semantic_fact("brand", "missing_a", None, attribute_side("бренд", "КНР"))
        payload = {
            "category": "Детские товары",
            "item_a": {"title": "Игрушка", "attributes": {}},
            "item_b": {"title": "Игрушка", "attributes": {"бренд": "КНР"}},
        }
        self.assertTrue(any("country-like value used as brand" in error for error in self.errors(fact, payload)))

    def test_weight_cannot_supply_package_quantity(self) -> None:
        fact = semantic_fact(
            "package_quantity", "missing_a", None, attribute_side("вес, г", "5")
        )
        payload = {
            "category": "Красота",
            "item_a": {"title": "Татуировка", "attributes": {}},
            "item_b": {"title": "Татуировка", "attributes": {"вес, г": "5"}},
        }
        self.assertTrue(any("measurement attribute" in error for error in self.errors(fact, payload)))

    def test_accessory_without_number_cannot_be_package_quantity(self) -> None:
        fact = semantic_fact(
            "package_quantity", "missing_a", None, attribute_side("комплектация", "с чехлом")
        )
        payload = {
            "category": "Спорт",
            "item_a": {"title": "Бинокль", "attributes": {}},
            "item_b": {"title": "Бинокль", "attributes": {"комплектация": "с чехлом"}},
        }
        self.assertTrue(any("no explicit numeric count" in error for error in self.errors(fact, payload)))

    def test_type_difference_flags_internal_title_attribute_conflict(self) -> None:
        fact = semantic_fact(
            "calculator_type",
            "different_value",
            title_side("настольный"),
            attribute_side("тип", "карманный калькулятор"),
        )
        payload = {
            "category": "Канцелярские товары",
            "item_a": {"title": "Калькулятор настольный", "attributes": {}},
            "item_b": {
                "title": "Калькулятор настольный",
                "attributes": {"тип": "карманный калькулятор"},
            },
        }
        self.assertTrue(any("source conflict" in error for error in self.errors(fact, payload)))

    def test_safe_normalized_brand_equivalence_is_allowed_only_in_v14(self) -> None:
        fact = semantic_fact(
            "brand",
            "identity_same",
            title_side("Fito Cosmetic", "fito cosmetic"),
            title_side("Fito Косметик", "fito cosmetic"),
            anchor_type="brand",
        )
        payload = {
            "category": "Бытовая химия",
            "item_a": {"title": "Fito Cosmetic гель", "attributes": {}},
            "item_b": {"title": "Fito Косметик крем", "attributes": {}},
        }
        legacy = self.errors(fact, payload, "legacy")
        current = self.errors(fact, payload, "v1_4")
        self.assertTrue(any("identity_same values differ" in error for error in legacy))
        self.assertFalse(any("identity_same values differ" in error for error in current))


if __name__ == "__main__":
    unittest.main()
