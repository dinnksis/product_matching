from __future__ import annotations

import json
import unittest
from pathlib import Path

from item_pipeline.pair_rules import load_mutation_rules
from scripts.export_soft_semantic_positive_rules import TIER_A, TIER_B


ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = (
    ROOT
    / "configs"
    / "generation_rule_catalog_statistical_v1"
    / "soft_positive_ab_v1"
)


class SoftSemanticPositiveCatalogTest(unittest.TestCase):
    def test_catalog_counts_and_evidence_contract(self) -> None:
        expected = {
            "tier_a": (325, TIER_A, 20),
            "tier_b": (2793, TIER_B, 10),
        }
        for name, (count, tier, quota) in expected.items():
            path = CATALOG_DIR / f"{name}.json"
            raw = json.loads(path.read_text(encoding="utf-8"))
            loaded = load_mutation_rules([path])
            self.assertEqual(len(raw), count)
            self.assertEqual(len(loaded), count)
            self.assertEqual({rule.label for rule in loaded}, {1})
            self.assertEqual({rule.generation_tier for rule in loaded}, {tier})
            self.assertEqual(
                len({rule.generation_rule_id for rule in loaded}), count
            )
            for rule in raw:
                self.assertEqual(rule["generation_examples_per_rule"], quota)
                self.assertTrue(rule["allowed_categories"])
                self.assertEqual(len(rule["allowed_product_types"]), 1)
                self.assertTrue(rule["attribute_key"].strip())
                self.assertTrue(rule["source_examples"])
                self.assertEqual(rule["target_label"], 1)

    def test_tier_boundaries_are_exact(self) -> None:
        tier_a = json.loads((CATALOG_DIR / "tier_a.json").read_text(encoding="utf-8"))
        tier_b = json.loads((CATALOG_DIR / "tier_b.json").read_text(encoding="utf-8"))
        for rule in tier_a:
            self.assertGreaterEqual(rule["profile_pair_support"], 5)
            self.assertGreaterEqual(rule["profile_target_probability"], 0.80)
        for rule in tier_b:
            self.assertGreaterEqual(rule["profile_pair_support"], 2)
            self.assertGreaterEqual(rule["profile_target_probability"], 0.70)
            self.assertTrue(
                rule["profile_pair_support"] < 5
                or rule["profile_target_probability"] < 0.80
            )

    def test_manifest_planned_counts(self) -> None:
        manifest = json.loads((CATALOG_DIR / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["tier_a_rules"], 325)
        self.assertEqual(manifest["tier_b_rules"], 2793)
        self.assertEqual(manifest["planned_tier_a_pairs"], 6500)
        self.assertEqual(manifest["planned_tier_b_pairs"], 27930)
        self.assertEqual(manifest["planned_total_pairs"], 34430)


if __name__ == "__main__":
    unittest.main()
