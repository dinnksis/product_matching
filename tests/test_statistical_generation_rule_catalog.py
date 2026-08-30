from __future__ import annotations

import collections
import csv
import hashlib
import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path

import pandas as pd

from item_pipeline.normalization import normalize_text, parse_attributes
from item_pipeline.pair_rules import load_mutation_rules
from item_pipeline.pair_validation import _target_value_reasons
from item_pipeline.qwen import _prompt_source_examples
from item_pipeline.rule_schedule import build_balanced_rule_schedule


ROOT = Path(__file__).resolve().parents[1]
CATALOG = (
    ROOT
    / "configs"
    / "generation_rule_catalog_statistical_v1"
    / "statistical_negative_rules_min2_p80_scoped_v3.json"
)
MANIFEST = CATALOG.with_suffix(".manifest.json")
POLICY = CATALOG.parent / "profile_capacity_policy_v1.json"
REJECTED = CATALOG.with_suffix(".rejected.csv")
EXPORTER = ROOT / "scripts" / "export_statistical_generation_rules.py"
PAIR_INPUTS = (
    ROOT / "data" / "qwen_atomic_differences_v2_full_train" / "pilot_inputs.parquet"
)
OCCURRENCES = (
    ROOT
    / "reports"
    / "atomic_rule_statistics_current"
    / "atomic_occurrences.parquet"
)
ITEMS = ROOT / "data" / "items_human.parquet"
MAIN_TIERS = {
    "STAT_LABEL0_CROSS_SPLIT_MIN2_P80_SCOPED",
    "STAT_LABEL0_MIN2_P80_SCOPED",
}


def load_exporter():
    spec = importlib.util.spec_from_file_location(
        "statistical_generation_rule_exporter", EXPORTER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load statistical generation-rule exporter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class StatisticalGenerationRuleCatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rules = json.loads(CATALOG.read_text(encoding="utf-8"))
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.exporter = load_exporter()
        with REJECTED.open(encoding="utf-8-sig", newline="") as stream:
            cls.rejected = list(csv.DictReader(stream))

    def test_final_hash_and_main_tier_counts(self) -> None:
        expected_hash = (
            "017e8ced6035695474007d5cc91e72870d77e5bef6b2348cdd13bde6cbdfdc6c"
        )
        self.assertEqual(hashlib.sha256(CATALOG.read_bytes()).hexdigest(), expected_hash)
        self.assertEqual(self.manifest["output_sha256"], expected_hash)
        self.assertEqual(len(self.rules), 102)
        tier_counts = collections.Counter(
            str(rule["generation_tier"]) for rule in self.rules
        )
        self.assertEqual(
            tier_counts,
            {
                "STAT_LABEL0_CROSS_SPLIT_MIN2_P80_SCOPED": 46,
                "STAT_LABEL0_MIN2_P80_SCOPED": 54,
                "STAT_LABEL0_UNANIMOUS_OVERRIDE": 2,
            },
        )
        self.assertEqual(sum(tier_counts[tier] for tier in MAIN_TIERS), 100)

    def test_manifest_records_source_and_anchor_safety_contract(self) -> None:
        selection = self.manifest["selection"]
        for flag in (
            "source_examples_use_raw_side_aligned_values",
            "restrict_anchor_context_keys",
            "canonical_anchor_title_from_attribute_values",
            "semantic_value_equivalence_validation",
            "forbid_numeric_only_model_values",
            "canonical_product_type_aliases_before_profile_grouping",
            "deduplicate_canonical_profile_source_pairs",
            "finite_target_value_domain_validation",
            "canonical_target_values_in_semantic_signature",
            "canonical_quantity_units_required",
            "canonical_dimension_units_required",
            "prompt_source_examples_satisfy_effective_target_contract",
            "insert_stone_alias_equivalence_validation",
        ):
            with self.subTest(flag=flag):
                self.assertIs(selection[flag], True)
        expected_excluded = {
            "rp_6a0e6c961b827e858cab46f5",
            "rp_eb1e31b61dfa4f86f81697ef",
        }
        self.assertEqual(set(selection["excluded_source_pair_ids"]), expected_excluded)
        self.assertEqual(self.exporter.EXCLUDED_SOURCE_PAIR_IDS, expected_excluded)

    def test_product_type_alias_is_merged_before_profile_selection(self) -> None:
        scoped_keys = [
            (
                rule["category"],
                rule["concept"],
                normalize_text(rule["allowed_product_types"][0]),
            )
            for rule in self.rules
        ]
        self.assertEqual(len(scoped_keys), len(set(scoped_keys)))
        self.assertNotIn(
            ("Ювелирные изделия", "gold_color", "серьги"), scoped_keys
        )
        gold = next(
            rule
            for rule in self.rules
            if rule["generation_rule_id"] == "gen_stat_dbef08c8ee2d88f9"
        )
        profile = gold["anchor_profiles"][0]
        self.assertEqual(gold["allowed_product_types"], ["серьги ювелирные"])
        self.assertEqual(profile["singleton_pair_support"], 5)
        self.assertEqual(len(profile["source_pair_ids"]), 5)
        self.assertEqual(
            len(profile["source_pair_ids"]), len(set(profile["source_pair_ids"]))
        )
        self.assertFalse(
            any(
                rule["generation_rule_id"] == "gen_stat_9525992e06247cbf"
                for rule in self.rules
            )
        )
        # The canonical union also makes this formerly split support reach min2.
        insert_stone = next(
            rule
            for rule in self.rules
            if rule["generation_rule_id"] == "gen_stat_4ca244067234664c"
        )
        self.assertEqual(insert_stone["concept"], "insert_stone")
        self.assertEqual(insert_stone["allowed_product_types"], ["серьги ювелирные"])
        self.assertEqual(insert_stone["singleton_support"], 2)

    def test_finite_domains_and_capacity_policy_are_pinned(self) -> None:
        policy_hash = hashlib.sha256(POLICY.read_bytes()).hexdigest()
        self.assertEqual(
            policy_hash,
            "829f9c458f1a2a72f16b766dfaf3abbdbad9a1874d71feb4457eab7da52b3fee",
        )
        self.assertEqual(self.manifest["profile_capacity_policy_sha256"], policy_hash)
        for rule in self.rules:
            self.assertEqual(
                rule["profile_capacity_policy_version"],
                "profile_capacity_policy_v1",
            )
            self.assertEqual(rule["profile_capacity_policy_sha256"], policy_hash)
        finite = [rule for rule in self.rules if rule["target_value_domain"]]
        self.assertEqual(len(finite), 5)
        observed = {
            (
                rule["concept"],
                normalize_text(rule["allowed_product_types"][0]),
            ): (len(rule["target_value_domain"]), rule["primary_task_safety_cap"])
            for rule in finite
        }
        self.assertEqual(
            observed,
            {
                ("gold_color", "браслет"): (4, 12),
                ("gold_color", "кольцо"): (4, 12),
                ("gold_color", "серьги ювелирные"): (4, 12),
                ("storage_capacity", "смартфон"): (8, 28),
                ("size", "смычок"): (8, 28),
            },
        )

    def test_prompt_examples_obey_effective_target_contract(self) -> None:
        loaded = load_mutation_rules([CATALOG])
        constrained = {
            "case_diameter",
            "diameter",
            "length",
            "length_mm",
            "package_quantity",
            "wheel_diameter",
            "width",
        }
        empty_effective: list[str] = []
        for rule in loaded:
            if rule.concept not in constrained and not rule.target_value_domain:
                continue
            examples = _prompt_source_examples(rule)
            if not examples:
                empty_effective.append(rule.generation_rule_id)
            product_type = rule.allowed_product_types[0]
            for example in examples:
                for side in ("a", "b"):
                    value = example[f"target_value_{side}"]
                    self.assertEqual(
                        _target_value_reasons(
                            rule,
                            value,
                            prefix="prompt",
                            product_type=product_type,
                        ),
                        [],
                    )
        self.assertEqual(
            empty_effective,
            [
                "gen_stat_27a4df2e379281e1",
                "gen_stat_55b5ffa6af53b701",
                "gen_stat_85bd9892423be105",
                "gen_stat_9f89f32556c53254",
                "gen_stat_bb20c05cb4945feb",
                "gen_stat_fa70490e70ccff44",
            ],
        )

    def test_actual_5200_schedule_covers_rules_and_respects_caps(self) -> None:
        rules = load_mutation_rules([CATALOG], tiers=MAIN_TIERS)
        donors = pd.read_parquet(ITEMS, columns=["id", "category"])
        schedule = build_balanced_rule_schedule(
            donors,
            rules,
            count=5_200,
            seed=20_260_910,
            two_rule_fraction=0.15,
            semantic_signature_limit=2,
        )
        summary = schedule.summary()
        self.assertEqual(
            schedule.schedule_sha256,
            "98f31514f1f24540236aa3351c10f5e611cd353c9ec5dc1ed77637d63e16eb96",
        )
        self.assertEqual(summary["primary_rule_profile_coverage"], 100)
        self.assertEqual(summary["scheduled_two_rule_tasks"], 426)
        self.assertAlmostEqual(summary["scheduled_two_rule_fraction"], 426 / 5_200)
        self.assertEqual(summary["balanced_total_rule_profile_usage_skew"], 1)
        for profile_id, cap in summary["profile_primary_task_caps"].items():
            self.assertLessEqual(summary["primary_rule_profile_usage"][profile_id], cap)

        prefix = schedule.entries[:5_000]
        prefix_primary = collections.Counter(
            entry.primary.profile_id for entry in prefix
        )
        self.assertEqual(len(prefix_primary), 100)
        self.assertEqual(sum(entry.secondary is not None for entry in prefix), 404)
        for profile_id, cap in summary["profile_primary_task_caps"].items():
            self.assertLessEqual(prefix_primary[profile_id], cap)

    def test_anchor_context_allowlists_are_nonempty_and_non_target(self) -> None:
        normalized_product_type_key = normalize_text("Тип товара")
        for rule in self.rules:
            with self.subTest(rule=rule["generation_rule_id"]):
                context_keys = rule["allowed_anchor_context_keys"]
                self.assertTrue(context_keys)
                normalized_context_keys = [normalize_text(key) for key in context_keys]
                self.assertTrue(all(normalized_context_keys))
                self.assertEqual(
                    len(normalized_context_keys), len(set(normalized_context_keys))
                )
                self.assertNotIn(
                    normalize_text(rule["attribute_key"]), normalized_context_keys
                )
                self.assertNotIn(normalized_product_type_key, normalized_context_keys)

    def test_source_examples_are_not_excluded_and_are_side_aligned(self) -> None:
        excluded = set(self.manifest["selection"]["excluded_source_pair_ids"])
        examples = [
            (rule, example)
            for rule in self.rules
            for example in rule["source_examples"]
        ]
        source_pair_ids = {example["source_pair_id"] for _, example in examples}
        self.assertTrue(source_pair_ids)
        self.assertTrue(source_pair_ids.isdisjoint(excluded))

        pair_inputs = pd.read_parquet(
            PAIR_INPUTS,
            columns=[
                "pair_id",
                "title_a",
                "attributes_a_json",
                "title_b",
                "attributes_b_json",
            ],
        ).set_index("pair_id")
        occurrences = pd.read_parquet(
            OCCURRENCES,
            columns=[
                "pair_id",
                "category",
                "is_singleton",
                "concept",
                "relation",
                "raw_value_a",
                "raw_value_b",
                "source_a",
                "source_b",
                "raw_attribute_a",
                "raw_attribute_b",
            ],
        ).set_index("pair_id")

        for rule, example in examples:
            pair_id = example["source_pair_id"]
            with self.subTest(rule=rule["generation_rule_id"], pair_id=pair_id):
                self.assertIn(pair_id, pair_inputs.index)
                source_input = pair_inputs.loc[pair_id]
                source_facts = occurrences.loc[[pair_id]]
                source_facts = source_facts[
                    source_facts["category"].eq(rule["category"])
                    & source_facts["concept"].eq(rule["concept"])
                    & source_facts["relation"].eq(rule["relation"])
                    & source_facts["is_singleton"]
                ]
                self.assertEqual(len(source_facts), 1)
                fact = source_facts.iloc[0]
                for side in ("a", "b"):
                    self.assertEqual(
                        example[f"title_{side}"], str(source_input[f"title_{side}"])[:280]
                    )
                    self.assertEqual(
                        example[f"target_value_{side}"],
                        str(fact[f"raw_value_{side}"])[:160],
                    )
                    source_kind = str(fact[f"source_{side}"])
                    self.assertIn(source_kind, {"attribute", "title"})
                    if source_kind == "attribute":
                        input_attribute_keys = {
                            normalize_text(key)
                            for key in parse_attributes(
                                source_input[f"attributes_{side}_json"]
                            )
                        }
                        self.assertIn(
                            normalize_text(fact[f"raw_attribute_{side}"]),
                            input_attribute_keys,
                        )

    def test_every_rule_has_one_profile_and_examples_of_that_type(self) -> None:
        for rule in self.rules:
            with self.subTest(rule=rule["generation_rule_id"]):
                self.assertEqual(len(rule["allowed_product_types"]), 1)
                self.assertEqual(len(rule["anchor_profiles"]), 1)
                product_type = normalize_text(rule["allowed_product_types"][0])
                profile = rule["anchor_profiles"][0]
                self.assertEqual(normalize_text(profile["product_type"]), product_type)
                self.assertEqual(
                    normalize_text(profile["normalized_product_type"]), product_type
                )
                self.assertTrue(rule["source_examples"])
                self.assertEqual(rule["source_examples"], profile["source_examples"])
                self.assertEqual(
                    {
                        normalize_text(example["product_type"])
                        for example in rule["source_examples"]
                    },
                    {product_type},
                )
                self.assertLessEqual(
                    {example["source_pair_id"] for example in rule["source_examples"]},
                    set(profile["source_pair_ids"]),
                )

    def test_profile_singleton_and_split_statistics_are_consistent(self) -> None:
        for rule in self.rules:
            with self.subTest(rule=rule["generation_rule_id"]):
                profile = rule["anchor_profiles"][0]
                self.assertEqual(
                    rule["singleton_support"], profile["singleton_pair_support"]
                )
                self.assertEqual(
                    rule["singleton_target_support"],
                    profile["singleton_target_support"],
                )
                self.assertEqual(
                    rule["singleton_target_probability"],
                    profile["singleton_target_probability"],
                )
                self.assertGreaterEqual(rule["singleton_support"], 2)
                self.assertGreaterEqual(rule["singleton_target_probability"], 0.8)

                for split in ("discovery", "validation"):
                    split_stats = profile["split_statistics"][split]
                    self.assertEqual(
                        rule[f"{split}_singleton_support"], split_stats["support"]
                    )
                    self.assertEqual(
                        rule[f"{split}_singleton_target_support"],
                        split_stats["target_support"],
                    )
                    self.assertEqual(
                        rule[f"{split}_singleton_target_probability"],
                        split_stats["target_probability"],
                    )
                    expected_probability = (
                        split_stats["target_support"] / split_stats["support"]
                        if split_stats["support"]
                        else 0.0
                    )
                    self.assertAlmostEqual(
                        split_stats["target_probability"], expected_probability
                    )

                is_cross_split = all(
                    profile["split_statistics"][split]["support"] >= 1
                    and profile["split_statistics"][split]["target_probability"]
                    >= 0.8
                    for split in ("discovery", "validation")
                )
                self.assertEqual(
                    "CROSS_SPLIT" in rule["generation_tier"], is_cross_split
                )

    def test_identifier_dominated_sources_do_not_leak(self) -> None:
        minimum_key_support = int(
            self.manifest["selection"]["minimum_observed_attribute_key_support"]
        )
        exported_keys = {
            (str(rule["category"]), str(rule["concept"])) for rule in self.rules
        }
        dominated = {
            (str(row["category"]), str(row["concept"]))
            for row in self.rejected
            if row["rejection_reason"]
            == "forbidden_identifier_dominates_source_attributes"
        }
        self.assertIn(("Автотовары", "model"), dominated)
        self.assertTrue(dominated.isdisjoint(exported_keys))
        for rule in self.rules:
            with self.subTest(rule=rule["generation_rule_id"]):
                self.assertIsNone(
                    self.exporter.FORBIDDEN_CONCEPT_RE.search(rule["concept"])
                )
                self.assertIsNone(
                    self.exporter.FORBIDDEN_ATTRIBUTE_RE.search(rule["attribute_key"])
                )
                self.assertLess(
                    int(rule["forbidden_source_attribute_support"]),
                    max(minimum_key_support, int(rule["observed_attribute_key_support"])),
                )
                self.assertGreaterEqual(rule["non_identifier_singleton_support"], 2)
                self.assertGreaterEqual(
                    rule["non_identifier_singleton_target_probability"], 0.8
                )

    def test_numeric_concepts_have_exporter_target_patterns(self) -> None:
        present_numeric_concepts = {
            rule["concept"]
            for rule in self.rules
            if rule["concept"] in self.exporter.TARGET_VALUE_PATTERNS
        }
        self.assertEqual(
            present_numeric_concepts,
            {
                "case_diameter",
                "color_code",
                "diameter",
                "length",
                "optical_power",
                "package_quantity",
                "size",
                "storage_capacity",
                "wheel_diameter",
                "width",
            },
        )
        for rule in self.rules:
            if rule["concept"] not in present_numeric_concepts:
                continue
            with self.subTest(rule=rule["generation_rule_id"]):
                self.assertTrue(rule["target_value_pattern"])
                re.compile(rule["target_value_pattern"])


if __name__ == "__main__":
    unittest.main()
