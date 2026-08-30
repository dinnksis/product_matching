from __future__ import annotations

import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

from item_pipeline.pair_rules import load_mutation_rules


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "generation_rule_catalog_statistical_v1"
SOURCE = (
    CONFIG
    / "semantic_all_pairs_cross_split_p80_support5_scoped_compact_safe_positive_v3.json"
)
CATALOG = (
    CONFIG
    / "semantic_all_pairs_cross_split_p80_support5_scoped_compact_transition_positive_v4.json"
)
MANIFEST = CATALOG.with_suffix(".manifest.json")
CATALOG_SHA256 = "a0aff8fa850c6b568c867c704b04be28a4f7c47a20dbb79723925c9dc2fd8245"
SOURCE_SHA256 = "569518c074da166dcac74b12a0c8ee313d08840468ac6bfddafcd4d90cfb3be2"
CATALOG_VERSION = (
    "semantic_all_pairs_cross_split_p80_support5_scoped_compact_transition_positive_v4"
)
POSITIVE_TIER = (
    "SEMANTIC_ALL_PAIRS_LABEL1_MANUAL_TRANSITION_ALLOWLIST_COMPACT_"
    "TRANSITION_POSITIVE_V4_EXPERIMENTAL_CORRELATIONAL"
)
EXPECTED_TRANSITIONS = {
    "gen_sem_all_2bd42ae67368b6da139a": [
        ["детская", "3 лет"],
        ["детская", "6 лет"],
    ],
    "gen_sem_all_fc1bf7245474d5979bbc": [
        ["настольная игра", "карточная игра"],
        ["настольная игра", "балансир"],
        ["настольная игра", "обучающая игра"],
        ["обучающая игра", "викторина"],
        ["обучающая игра", "викторина, для двоих"],
        ["бродилка", "лабиринт"],
    ],
    "gen_sem_all_a29a2640f81133199b22": [
        ["детская", "3 лет"],
        ["детская", "4 лет"],
        ["детская", "10 лет"],
        ["детская", "10+"],
        ["детская", "13 лет"],
        ["детская", "школьники (7-16)"],
        ["взрослая", "от 18 лет"],
    ],
    "gen_sem_all_1fd8ee0362b4a69694eb": [
        ["детская", "3 лет"],
        ["детская", "3+"],
        ["детская", "4 лет"],
        ["детская", "4+"],
        ["детская", "6 лет"],
        ["детская", "от 7 лет"],
        ["детская", "10 лет"],
        ["детская", "10+"],
    ],
}
EXCLUDED_POSITIVES = {
    "gen_sem_all_bb884e95495f2e4053ad",
    "gen_sem_all_3eab364eedff30a6ccec",
}


class SemanticTransitionPositiveV4CatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_rules = json.loads(CATALOG.read_text(encoding="utf-8"))
        cls.raw_by_id = {
            str(rule["generation_rule_id"]): rule for rule in cls.raw_rules
        }
        cls.loaded_rules = load_mutation_rules([CATALOG])
        cls.source_rules = json.loads(SOURCE.read_text(encoding="utf-8"))
        cls.source_by_id = {
            str(rule["generation_rule_id"]): rule for rule in cls.source_rules
        }
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_pinned_catalog_and_source_sha256(self) -> None:
        self.assertEqual(hashlib.sha256(CATALOG.read_bytes()).hexdigest(), CATALOG_SHA256)
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), SOURCE_SHA256)
        self.assertEqual(self.manifest["output_sha256"], CATALOG_SHA256)
        self.assertEqual(self.manifest["source_catalog_sha256"], SOURCE_SHA256)

    def test_counts_and_loader_compatibility(self) -> None:
        self.assertEqual(len(self.raw_rules), 3_678)
        self.assertEqual(len(self.loaded_rules), 3_678)
        self.assertEqual(
            Counter(int(rule["label"]) for rule in self.raw_rules),
            {0: 3_674, 1: 4},
        )
        self.assertEqual(Counter(rule.label for rule in self.loaded_rules), {0: 3_674, 1: 4})
        self.assertEqual(
            len({category for rule in self.loaded_rules for category in rule.allowed_categories}),
            18,
        )

    def test_positive_allowlist_and_exact_unordered_transitions(self) -> None:
        positive_ids = {
            rule_id
            for rule_id, rule in self.raw_by_id.items()
            if int(rule["label"]) == 1
        }
        self.assertEqual(positive_ids, set(EXPECTED_TRANSITIONS))
        self.assertTrue(EXCLUDED_POSITIVES.isdisjoint(self.raw_by_id))

        for rule_id, expected in EXPECTED_TRANSITIONS.items():
            with self.subTest(rule_id=rule_id):
                rule = self.raw_by_id[rule_id]
                self.assertEqual(rule["allowed_value_transitions"], expected)
                self.assertIs(rule["allowed_value_transitions_unordered"], True)
                self.assertEqual(rule["value_transition_semantics"], "exact_unordered_pairs")
                canonical = {
                    tuple(sorted((left.casefold(), right.casefold())))
                    for left, right in expected
                }
                self.assertEqual(len(canonical), len(expected))

    def test_positive_domains_caps_context_and_prompt_fields(self) -> None:
        for rule_id, transitions in EXPECTED_TRANSITIONS.items():
            with self.subTest(rule_id=rule_id):
                rule = self.raw_by_id[rule_id]
                expected_domain = list(
                    dict.fromkeys(value for pair in transitions for value in pair)
                )
                self.assertEqual(rule["target_value_domain"], expected_domain)
                self.assertEqual(rule["primary_task_safety_cap"], 2 * len(transitions))
                self.assertEqual(
                    rule["allowed_anchor_context_keys"], ["Бренд", "Название игры"]
                )
                self.assertEqual(
                    rule["required_anchor_context_keys"], ["Бренд", "Название игры"]
                )
                self.assertEqual(rule["generation_tier"], POSITIVE_TIER)
                self.assertEqual(rule["manual_positive_review_version"], CATALOG_VERSION)
                self.assertEqual(rule["transition_positive_review_version"], CATALOG_VERSION)
                self.assertIs(rule["manual_transition_allowlist"], True)
                self.assertIn("allowed_value_transitions", rule["generation_action"])
                self.assertIn("two endpoints", rule["required_postcondition"])
                self.assertIn("«Бренд»", rule["anchor_hint"])
                self.assertIn("«Название игры»", rule["anchor_hint"])

        age_rule = self.raw_by_id["gen_sem_all_2bd42ae67368b6da139a"]
        self.assertEqual(age_rule["attribute_key"], "Возрастная рекомендация")
        self.assertEqual(
            age_rule["attribute_key_source"],
            "manual_transition_positive_v4_override",
        )

    def test_all_negative_profiles_retained_with_only_prompt_guards_changed(self) -> None:
        source_negatives = {
            rule_id: rule
            for rule_id, rule in self.source_by_id.items()
            if int(rule["label"]) == 0
        }
        final_negatives = {
            rule_id: rule
            for rule_id, rule in self.raw_by_id.items()
            if int(rule["label"]) == 0
        }
        self.assertEqual(set(final_negatives), set(source_negatives))
        for rule_id, source in source_negatives.items():
            with self.subTest(rule_id=rule_id):
                final = final_negatives[rule_id]
                changed = {
                    key for key in set(source) | set(final) if source.get(key) != final.get(key)
                }
                self.assertEqual(
                    changed, {"generation_action", "required_postcondition"}
                )
                self.assertTrue(final["generation_action"].startswith(source["generation_action"]))
                self.assertTrue(
                    final["required_postcondition"].startswith(
                        source["required_postcondition"]
                    )
                )
                self.assertIn("unquestionably non-equivalent", final["generation_action"])
                self.assertIn("synonyms", final["generation_action"])
                self.assertIn("same sellable variant", final["required_postcondition"])

    def test_manifest_records_capacity_and_evidence_caveats(self) -> None:
        selection = self.manifest["selection"]
        provenance = self.manifest["transition_provenance"]
        self.assertEqual(self.manifest["catalog_version"], CATALOG_VERSION)
        self.assertEqual(self.manifest["exported_rules"], 3_678)
        self.assertEqual(self.manifest["label_counts"], {"0": 3_674, "1": 4})
        self.assertEqual(self.manifest["category_coverage"], 18)
        self.assertEqual(self.manifest["transition_rule_count"], 4)
        self.assertEqual(self.manifest["transition_count"], 23)
        self.assertEqual(self.manifest["transition_capacity"], 46)
        self.assertEqual(selection["recommended_target1_count"], 46)
        self.assertEqual(selection["recommended_label_one_fraction"], 0.0046)
        self.assertEqual(selection["recommended_two_rule_fraction"], 0.0)
        self.assertIs(selection["label0_non_equivalence_prompt_guard"], True)
        self.assertEqual(provenance["cross_split_validation_scope"], "rule_profile_only")
        self.assertIs(provenance["exact_transitions_cross_split_validated"], False)
        self.assertIn("no individual exact value transition", provenance["caveat"])


if __name__ == "__main__":
    unittest.main()
