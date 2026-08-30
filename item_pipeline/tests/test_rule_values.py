from __future__ import annotations

import unittest

from item_pipeline.pair_generate import _semantic_pair_signature
from item_pipeline.pair_rules import MutationRule
from item_pipeline.pair_validation import (
    _target_value_reasons,
    _target_values_semantically_equivalent,
)
from item_pipeline.rule_values import canonical_target_value


def make_rule(
    concept: str,
    *,
    product_type: str = "товар",
    domain: tuple[str, ...] = (),
) -> MutationRule:
    return MutationRule(
        generation_rule_id=f"rule-{concept}",
        source_rule_id=f"source-{concept}",
        generation_tier="TEST",
        label=0,
        concept=concept,
        relation="different_value",
        semantic_family="test",
        attribute_key="Значение",
        anchor_hint="",
        allowed_categories=("Тест",),
        generation_action="change",
        required_postcondition="different",
        source_path="test",
        allowed_product_types=(product_type,),
        target_value_domain=domain,
    )


class RuleValueCanonicalizationTest(unittest.TestCase):
    def test_storage_aliases_and_terabyte_equivalence(self) -> None:
        domain = (
            "8 ГБ", "16 ГБ", "32 ГБ", "64 ГБ", "128 ГБ", "256 ГБ",
            "512 ГБ", "1024 ГБ",
        )
        for raw in ("128", "128 ГБ", "128GB"):
            with self.subTest(raw=raw):
                self.assertEqual(
                    canonical_target_value(
                        "storage_capacity", "смартфон", raw, domain
                    ),
                    "128 ГБ",
                )
        rule = make_rule(
            "storage_capacity", product_type="смартфон", domain=domain
        )
        self.assertTrue(
            _target_values_semantically_equivalent(rule, "1 ТБ", "1024 ГБ")
        )

    def test_dimension_units_convert_to_millimeters(self) -> None:
        self.assertEqual(
            canonical_target_value("width", "стол", "1 м"), "1000 мм"
        )
        self.assertEqual(
            canonical_target_value("width", "стол", "100 см"), "1000 мм"
        )
        self.assertEqual(
            canonical_target_value("width", "стол", "60 см"), "600 мм"
        )
        self.assertEqual(
            canonical_target_value("diameter", "сверло", "6,5 мм"), "6.5 мм"
        )
        rule = make_rule("width", product_type="стол")
        self.assertTrue(
            _target_values_semantically_equivalent(rule, "1 м", "100 см")
        )
        self.assertIsNone(canonical_target_value("width", "стол", "1200"))

    def test_quantity_gold_and_bow_canonical_values(self) -> None:
        self.assertEqual(
            canonical_target_value("package_quantity", "лампочка", "6 штук"),
            "6 шт",
        )
        self.assertIsNone(
            canonical_target_value("package_quantity", "лампочка", "6")
        )
        quantity_rule = make_rule("package_quantity", product_type="лампочка")
        self.assertIn(
            "value:missing_or_invalid_canonical_unit",
            _target_value_reasons(quantity_rule, "6", prefix="value"),
        )
        gold = ("белое", "желтое", "красное", "комбинированное")
        self.assertEqual(
            canonical_target_value(
                "gold_color", "серьги ювелирные", "белое и желтое золото", gold
            ),
            "комбинированное",
        )
        bows = ("1/32", "1/16", "1/10", "1/8", "1/4", "1/2", "3/4", "4/4")
        self.assertEqual(
            canonical_target_value("size", "смычок", "размер ½", bows),
            "1/2",
        )
        bow_rule = make_rule("size", product_type="смычок", domain=bows)
        self.assertIn(
            "value:outside_target_value_domain",
            _target_value_reasons(bow_rule, "7/8", prefix="value"),
        )

    def test_storage_signature_collapses_spelling_aliases(self) -> None:
        def signature(old: str, new: str) -> str:
            return _semantic_pair_signature(
                "Электроника",
                "смартфон",
                [{
                    "concept": "storage_capacity",
                    "attribute_key": "Объем памяти",
                    "original_value": old,
                    "new_value": new,
                }],
            )

        self.assertEqual(signature("128", "256"), signature("128 ГБ", "256 ГБ"))
        self.assertEqual(signature("128", "256"), signature("128GB", "256GB"))

    def test_insert_concept_aliases_are_not_physical_changes(self) -> None:
        for concept in ("insert", "insert_stone", "insert_type"):
            with self.subTest(concept=concept):
                self.assertTrue(
                    _target_values_semantically_equivalent(
                        make_rule(concept), "фианит", "кубический цирконий"
                    )
                )


if __name__ == "__main__":
    unittest.main()
