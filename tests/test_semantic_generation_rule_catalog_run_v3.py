from __future__ import annotations

import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

from item_pipeline.pair_rules import load_mutation_rules


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "generation_rule_catalog_statistical_v1"
CATALOG = (
    CONFIG
    / "semantic_all_pairs_cross_split_p80_support5_scoped_compact_safe_positive_v3.json"
)
MANIFEST = CATALOG.with_suffix(".manifest.json")
SOURCE_V2 = (
    CONFIG / "semantic_all_pairs_cross_split_p80_support5_scoped_compact_v2.json"
)
ALLOWLIST = {
    "gen_sem_all_2bd42ae67368b6da139a",
    "gen_sem_all_fc1bf7245474d5979bbc",
    "gen_sem_all_a29a2640f81133199b22",
    "gen_sem_all_bb884e95495f2e4053ad",
    "gen_sem_all_1fd8ee0362b4a69694eb",
    "gen_sem_all_3eab364eedff30a6ccec",
}
POSITIVE_TIER = (
    "SEMANTIC_ALL_PAIRS_LABEL1_MANUAL_ALLOWLIST_COMPACT_SAFE_POSITIVE_V3_"
    "EXPERIMENTAL_CORRELATIONAL"
)


class SemanticSafePositiveRunCatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_rules = json.loads(CATALOG.read_text(encoding="utf-8"))
        cls.rules = load_mutation_rules([CATALOG])
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.source_v2 = json.loads(SOURCE_V2.read_text(encoding="utf-8"))

    def test_loader_counts_and_category_coverage(self) -> None:
        self.assertEqual(len(self.rules), 3_680)
        self.assertEqual(Counter(rule.label for rule in self.rules), {0: 3_674, 1: 6})
        self.assertEqual(
            len({category for rule in self.rules for category in rule.allowed_categories}),
            18,
        )

    def test_only_manually_allowed_positive_ids_remain(self) -> None:
        positives = {
            rule.generation_rule_id for rule in self.rules if rule.label == 1
        }
        self.assertEqual(positives, ALLOWLIST)
        self.assertTrue(
            all(rule.generation_tier == POSITIVE_TIER for rule in self.rules if rule.label == 1)
        )

    def test_all_v2_negative_rules_are_unchanged(self) -> None:
        final_negative = {
            rule["generation_rule_id"]: rule
            for rule in self.raw_rules
            if int(rule["label"]) == 0
        }
        source_negative = {
            rule["generation_rule_id"]: rule
            for rule in self.source_v2
            if int(rule["label"]) == 0
        }
        self.assertEqual(final_negative, source_negative)

    def test_manifest_records_manual_tier_diagnostics(self) -> None:
        diagnostics = self.manifest["manual_positive_tier_diagnostics"]
        self.assertEqual(diagnostics["source_positive_profiles"], 22)
        self.assertEqual(diagnostics["allowed_positive_profiles"], 6)
        self.assertEqual(diagnostics["excluded_positive_profiles"], 16)
        self.assertEqual(set(diagnostics["allowed_generation_rule_ids"]), ALLOWLIST)
        self.assertEqual(len(diagnostics["allowed_profiles"]), 6)
        self.assertEqual(
            set(self.manifest["selection"]["positive_allowlist"]), ALLOWLIST
        )
        self.assertEqual(
            self.manifest["selection"]["positive_manual_tier"], POSITIVE_TIER
        )

    def test_manifest_pins_source_and_output(self) -> None:
        self.assertEqual(
            self.manifest["source_catalog_sha256"],
            hashlib.sha256(SOURCE_V2.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            self.manifest["output_sha256"],
            hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
        )
        self.assertEqual(self.manifest["exported_rules"], 3_680)
        self.assertEqual(self.manifest["label_counts"], {"0": 3_674, "1": 6})
        self.assertEqual(self.manifest["category_coverage"], 18)


if __name__ == "__main__":
    unittest.main()
