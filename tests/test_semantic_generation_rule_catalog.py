from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import unittest
from collections import Counter
from pathlib import Path

from item_pipeline.normalization import normalize_text
from item_pipeline.pair_rules import load_mutation_rules


ROOT = Path(__file__).resolve().parents[1]
CATALOG = (
    ROOT
    / "configs"
    / "generation_rule_catalog_statistical_v1"
    / "semantic_all_pairs_cross_split_p80_support5_scoped_v1.json"
)
MANIFEST = CATALOG.with_suffix(".manifest.json")
EXPORTER = ROOT / "scripts" / "export_semantic_generation_rules.py"
FORBIDDEN = re.compile(
    r"(?:sku|артикул|партномер|part[_ -]?number|oem|код\s+товара)", re.I
)


def load_exporter():
    spec = importlib.util.spec_from_file_location(
        "semantic_generation_rule_exporter", EXPORTER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load semantic generation-rule exporter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SemanticGenerationRuleCatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_rules = json.loads(CATALOG.read_text(encoding="utf-8"))
        cls.rules = load_mutation_rules([CATALOG])
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.exporter = load_exporter()

    def test_catalog_is_loader_compatible_and_fully_source_scoped(self) -> None:
        self.assertEqual(len(self.rules), 14_505)
        self.assertEqual(Counter(rule.label for rule in self.rules), {0: 14_483, 1: 22})
        self.assertEqual(
            len({category for rule in self.rules for category in rule.allowed_categories}),
            18,
        )
        self.assertTrue(all(len(rule.allowed_categories) == 1 for rule in self.rules))
        self.assertTrue(all(len(rule.allowed_product_types) == 1 for rule in self.rules))
        self.assertTrue(all(rule.attribute_key for rule in self.rules))
        self.assertEqual(
            len({rule.generation_rule_id for rule in self.rules}), len(self.rules)
        )

    def test_manifest_pins_sources_output_and_first_experiment_policy(self) -> None:
        output_hash = hashlib.sha256(CATALOG.read_bytes()).hexdigest()
        self.assertEqual(self.manifest["output_sha256"], output_hash)
        self.assertEqual(self.manifest["exported_rules"], 14_505)
        self.assertEqual(self.manifest["label_counts"], {"0": 14_483, "1": 22})
        self.assertEqual(self.manifest["category_coverage"], 18)
        selection = self.manifest["selection"]
        self.assertEqual(selection["semantic_threshold"], 0.8)
        self.assertEqual(selection["minimum_semantic_weighted_support"], 5.0)
        self.assertEqual(selection["minimum_product_type_pair_support"], 2)
        self.assertTrue(selection["require_semantic_cross_split_p80"])
        self.assertTrue(selection["require_product_type_cross_split_p80"])
        self.assertEqual(selection["recommended_first_experiment_two_rule_fraction"], 0.0)
        self.assertEqual(
            selection["evidence_interpretation"],
            "experimental_correlational_not_causal",
        )

    def test_every_profile_passes_the_recorded_probability_contract(self) -> None:
        for rule in self.raw_rules:
            with self.subTest(rule=rule["generation_rule_id"]):
                self.assertEqual(rule["label"], rule["target_label"])
                self.assertGreaterEqual(rule["semantic_weighted_support"], 5.0)
                self.assertGreaterEqual(rule["semantic_target_probability"], 0.8)
                self.assertIs(rule["semantic_cross_split_p80"], True)
                self.assertGreaterEqual(rule["profile_pair_support"], 2)
                self.assertGreaterEqual(rule["profile_target_probability"], 0.8)
                self.assertIs(rule["profile_cross_split_p80"], True)
                for split in ("discovery", "validation"):
                    stats = rule[f"{split}_profile_statistics"]
                    self.assertGreaterEqual(stats["support"], 1)
                    self.assertGreaterEqual(stats["target_probability"], 0.8)

    def test_identifiers_are_not_generation_targets(self) -> None:
        for rule in self.rules:
            target = f"{rule.concept} {rule.attribute_key}"
            with self.subTest(rule=rule.generation_rule_id):
                self.assertIsNone(FORBIDDEN.search(normalize_text(target)))

    def test_positive_rules_remain_explicitly_experimental(self) -> None:
        positives = [rule for rule in self.raw_rules if int(rule["label"]) == 1]
        self.assertEqual(len(positives), 22)
        self.assertTrue(
            all("EXPERIMENTAL_CORRELATIONAL" in rule["generation_tier"] for rule in positives)
        )
        self.assertTrue(all(rule["target_label"] == 1 for rule in positives))


if __name__ == "__main__":
    unittest.main()
