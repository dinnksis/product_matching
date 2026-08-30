from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from item_pipeline.pair_generate import select_rule_bundle
from item_pipeline.pair_rules import (
    catalog_summary,
    load_mutation_rules,
    rules_are_product_compatible,
    rules_by_category,
)


ROOT = Path(__file__).resolve().parents[2]
CATALOG_DIR = ROOT / "configs" / "generation_rule_catalog_rare_v1"
CATALOG = CATALOG_DIR / "rare_rule_candidates_all.csv"
DEFAULT_CATALOGS = [
    CATALOG,
    CATALOG_DIR / "rare_generation_rules_v1.json",
    CATALOG_DIR / "rare_negative_rules_experimental.csv",
]


class PairRulesTest(unittest.TestCase):
    def test_rich_catalogs_overlay_candidates_without_duplicating_rules(self) -> None:
        rules = load_mutation_rules(DEFAULT_CATALOGS)
        summary = catalog_summary(rules)
        self.assertEqual(summary["loaded_rules"], 75)
        safe = next(rule for rule in rules if rule.generation_tier == "RARE_SAFE")
        self.assertEqual(Path(safe.source_path).name, "rare_generation_rules_v1.json")

    def test_full_rare_catalog_retains_tiers_but_skips_empty_category_scope(self) -> None:
        rules = load_mutation_rules([CATALOG])
        summary = catalog_summary(rules)
        self.assertEqual(summary["loaded_rules"], 75)
        self.assertEqual(summary["selectable_rules"], 72)
        self.assertEqual(summary["rules_without_allowed_categories"], 3)
        self.assertEqual(summary["tier_counts"]["RARE_SAFE"], 24)
        self.assertEqual(set(summary["labels"]), {0})

        by_category = rules_by_category(rules)
        for category, category_rules in by_category.items():
            self.assertTrue(category_rules)
            self.assertTrue(
                all(category in rule.allowed_categories for rule in category_rules)
            )

    def test_two_rule_bundle_uses_distinct_concepts_in_one_category(self) -> None:
        rules = load_mutation_rules([CATALOG])
        category = "Электроника"
        bundle = select_rule_bundle(
            rules_by_category(rules)[category],
            task_seed=17,
            selection_attempt=1,
            two_rule_fraction=1.0,
        )
        self.assertEqual(len(bundle), 2)
        self.assertEqual(len({rule.concept for rule in bundle}), 2)
        self.assertTrue(all(category in rule.allowed_categories for rule in bundle))
        self.assertEqual({rule.label for rule in bundle}, {0})
        self.assertTrue(rules_are_product_compatible(*bundle))

    def test_same_category_does_not_imply_same_product_compatibility(self) -> None:
        rules = load_mutation_rules(DEFAULT_CATALOGS)
        by_concept = {rule.concept: rule for rule in rules}
        self.assertTrue(
            rules_are_product_compatible(
                by_concept["oxygen_permeability"], by_concept["base_curve"]
            )
        )
        self.assertFalse(
            rules_are_product_compatible(
                by_concept["paper_format"], by_concept["model_line"]
            )
        )

    def test_catalog_can_supply_attribute_key_for_new_statistical_concept(self) -> None:
        payload = [
            {
                "source_rule_id": "stat_rule_1",
                "generation_rule_id": "gen_stat_rule_1",
                "generation_tier": "STAT_MIN2_P80",
                "label": 0,
                "concept": "package_quantity",
                "relation": "different_value",
                "attribute_key": "Количество в упаковке",
                "anchor_hint": "Не дублируй количество в других полях.",
                "allowed_categories": ["Продукты питания"],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            rules = load_mutation_rules([path])
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].attribute_key, "Количество в упаковке")
        self.assertEqual(rules[0].anchor_hint, "Не дублируй количество в других полях.")

    def test_shared_type_and_curated_semantics_enable_statistical_bundle(self) -> None:
        payload = [
            {
                "source_rule_id": f"source-{concept}",
                "generation_rule_id": f"rule-{concept}",
                "generation_tier": "STAT_SCOPED",
                "label": 0,
                "concept": concept,
                "relation": "different_value",
                "attribute_key": key,
                "allowed_categories": ["Дом и сад"],
                "allowed_product_types": ["семена"],
            }
            for concept, key in [
                ("package_quantity", "Количество в упаковке"),
                ("variety", "Сорт семян"),
            ]
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            rules = load_mutation_rules([path])
        self.assertTrue(rules_are_product_compatible(rules[0], rules[1]))
        bundle = select_rule_bundle(
            rules,
            task_seed=7,
            selection_attempt=1,
            two_rule_fraction=1.0,
        )
        self.assertEqual(len(bundle), 2)


if __name__ == "__main__":
    unittest.main()
