from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

from item_pipeline.pair_rules import load_mutation_rules


ROOT = Path(__file__).resolve().parents[1]
CATALOG = (
    ROOT
    / "configs"
    / "generation_rule_catalog_statistical_v1"
    / "semantic_all_pairs_cross_split_p80_support5_scoped_compact_v2.json"
)
MANIFEST = CATALOG.with_suffix(".manifest.json")
EXPORTER = ROOT / "scripts" / "export_semantic_generation_rules.py"


def load_exporter():
    spec = importlib.util.spec_from_file_location(
        "semantic_generation_rule_exporter_compact", EXPORTER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load semantic generation-rule exporter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CompactSemanticGenerationRuleCatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_rules = json.loads(CATALOG.read_text(encoding="utf-8"))
        cls.rules = load_mutation_rules([CATALOG])
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.exporter = load_exporter()

    def test_loader_counts_and_category_coverage(self) -> None:
        self.assertEqual(len(self.rules), 3_696)
        self.assertEqual(Counter(rule.label for rule in self.rules), {0: 3_674, 1: 22})
        self.assertEqual(
            len({category for rule in self.rules for category in rule.allowed_categories}),
            18,
        )
        self.assertTrue(all(len(rule.allowed_product_types) == 1 for rule in self.rules))

    def test_placeholder_and_non_alphabetic_product_types_are_rejected(self) -> None:
        invalid = ["", "-", "—", "–", "нет", "не указан", "не указано", "123", "..."]
        for value in invalid:
            with self.subTest(value=value):
                self.assertFalse(self.exporter.valid_product_type(value))
        for value in ("настольная игра", "SSD накопитель", "шина R18"):
            with self.subTest(value=value):
                self.assertTrue(self.exporter.valid_product_type(value))
        self.assertTrue(
            all(
                self.exporter.valid_product_type(rule.allowed_product_types[0])
                for rule in self.rules
            )
        )

    def test_label0_has_one_best_profile_per_semantic_center(self) -> None:
        negatives = [rule for rule in self.raw_rules if int(rule["label"]) == 0]
        source_ids = [str(rule["source_rule_id"]) for rule in negatives]
        self.assertEqual(len(source_ids), len(set(source_ids)))
        self.assertEqual(len(negatives), 3_674)
        self.assertEqual(
            self.manifest["selection"]["label0_profile_compaction"],
            "one_best_profile_per_semantic_center_by_support_probability_type",
        )

    def test_all_valid_positive_profiles_are_retained(self) -> None:
        positives = [rule for rule in self.raw_rules if int(rule["label"]) == 1]
        self.assertEqual(len(positives), 22)
        self.assertTrue(all(rule["target_label"] == 1 for rule in positives))
        self.assertEqual(
            self.manifest["selection"]["label1_profile_compaction"],
            "keep_all_valid_profiles",
        )

    def test_manifest_pins_compact_output(self) -> None:
        output_hash = hashlib.sha256(CATALOG.read_bytes()).hexdigest()
        self.assertEqual(output_hash, self.manifest["output_sha256"])
        self.assertEqual(self.manifest["exported_rules"], 3_696)
        self.assertEqual(self.manifest["label_counts"], {"0": 3_674, "1": 22})
        self.assertEqual(self.manifest["category_coverage"], 18)
        self.assertEqual(
            self.manifest["executable_profiles_before_label0_compaction"], 14_493
        )
        self.assertTrue(
            self.manifest["selection"][
                "reject_placeholder_or_non_alphabetic_product_types"
            ]
        )
        self.assertEqual(
            self.manifest["selection"]["recommended_first_experiment_two_rule_fraction"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
