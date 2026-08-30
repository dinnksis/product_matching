from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_semantic_atomic_rules.py"
SPEC = importlib.util.spec_from_file_location("analyze_semantic_atomic_rules", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SemanticAtomicRulesTest(unittest.TestCase):
    def test_value_family_separates_measurements_and_counts(self) -> None:
        self.assertEqual(MODULE.value_family("storage_capacity", "64 ГБ", "128 GB"), "data")
        self.assertEqual(MODULE.value_family("package_quantity", "4 шт", "8 штук"), "count")
        self.assertEqual(MODULE.value_family("length", "10 см", "20 cm"), "length")
        self.assertEqual(MODULE.value_family("diameter", "10", "20"), "length")
        self.assertEqual(MODULE.value_family("sku", "A10", "A20"), "identifier")
        self.assertEqual(MODULE.value_family("color", "красный", "синий"), "text")

    def test_build_prototypes_keeps_labels_out_of_semantic_text(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "pair_id": "p1",
                    "category": "Аптека",
                    "human_label": 1,
                    "split": "discovery",
                    "is_singleton": True,
                    "concept": "optical_power",
                    "raw_concept": "diopters",
                    "relation": "different_value",
                    "raw_value_a": "-2.5 D",
                    "raw_value_b": "-3.0 D",
                    "raw_attribute_a": "Оптическая сила",
                    "raw_attribute_b": "Диоптрии",
                }
            ]
        )
        prototypes, _ = MODULE.build_prototypes(rows)
        self.assertEqual(len(prototypes), 1)
        self.assertEqual(int(prototypes.iloc[0].singleton_label1), 1)
        text = prototypes.iloc[0].semantic_text
        self.assertNotIn("метка", text.casefold())
        self.assertNotIn("human_label", text)

    def test_all_pair_evidence_counts_one_vote_per_pair_and_prototype(self) -> None:
        base = {
            "pair_id": "p1",
            "category": "Мебель",
            "human_label": 0,
            "split": "discovery",
            "is_singleton": False,
            "concept": "color",
            "raw_concept": "colour",
            "relation": "different_value",
            "raw_value_a": "красный",
            "raw_value_b": "синий",
            "raw_attribute_a": "Цвет",
            "raw_attribute_b": "Цвет товара",
        }
        prototypes, _ = MODULE.build_prototypes(pd.DataFrame([base, dict(base)]))
        self.assertEqual(len(prototypes), 1)
        self.assertEqual(int(prototypes.iloc[0].pair_support), 1)
        self.assertEqual(int(prototypes.iloc[0].label0), 1)
        self.assertEqual(int(prototypes.iloc[0].singleton_support), 0)

    def test_cannot_link_uses_distinct_concepts_in_same_pair(self) -> None:
        frame = pd.DataFrame(
            [
                {"category": "Мебель", "pair_id": "p", "raw_concept_normalized": "color"},
                {"category": "Мебель", "pair_id": "p", "raw_concept_normalized": "frame color"},
            ]
        )
        links = MODULE.cannot_link_concepts(frame)
        self.assertTrue(MODULE.is_cannot_link("Мебель", "color", "frame color", links))

    def test_semantic_pooling_adds_weighted_support(self) -> None:
        prototypes = pd.DataFrame(
            [
                {
                    "prototype_key": "a",
                    "category": "Аптека",
                    "relation": "different_value",
                    "raw_concept": "optical_power",
                    "canonical_concept": "optical_power",
                    "attribute_role": "оптическая сила",
                    "value_family": "optical",
                    "semantic_text": "a",
                    "forbidden_identifier": False,
                    "pair_support": 2,
                    "singleton_support": 1,
                    "label0": 0,
                    "label1": 2,
                    "singleton_label0": 0,
                    "singleton_label1": 1,
                    "discovery_singleton_label0": 0,
                    "discovery_singleton_label1": 1,
                    "validation_singleton_label0": 0,
                    "validation_singleton_label1": 0,
                    "discovery_label0": 0,
                    "discovery_label1": 1,
                    "validation_label0": 0,
                    "validation_label1": 1,
                },
                {
                    "prototype_key": "b",
                    "category": "Аптека",
                    "relation": "different_value",
                    "raw_concept": "diopters",
                    "canonical_concept": "optical_power",
                    "attribute_role": "диоптрии",
                    "value_family": "optical",
                    "semantic_text": "b",
                    "forbidden_identifier": False,
                    "pair_support": 2,
                    "singleton_support": 1,
                    "label0": 1,
                    "label1": 1,
                    "singleton_label0": 0,
                    "singleton_label1": 1,
                    "discovery_singleton_label0": 0,
                    "discovery_singleton_label1": 0,
                    "validation_singleton_label0": 0,
                    "validation_singleton_label1": 1,
                    "discovery_label0": 1,
                    "discovery_label1": 0,
                    "validation_label0": 0,
                    "validation_label1": 1,
                },
            ]
        )
        neighbours = [
            [MODULE.Neighbour(0, 1.0), MODULE.Neighbour(1, 0.9)],
            [MODULE.Neighbour(1, 1.0), MODULE.Neighbour(0, 0.9)],
        ]
        candidates, _ = MODULE.pooled_candidates(
            prototypes,
            neighbours,
            set(),
            threshold=0.8,
            minimum_support=1.5,
            minimum_probability=0.8,
        )
        first = candidates.iloc[0]
        self.assertAlmostEqual(float(first.weighted_evidence_support), 1.9)
        self.assertEqual(int(first.target_label), 1)
        self.assertTrue(bool(first.cross_split_p80))
        self.assertTrue(bool(first.is_candidate))

        all_candidates, _ = MODULE.pooled_candidates(
            prototypes,
            neighbours,
            set(),
            evidence_scope="all",
            threshold=0.8,
            minimum_support=1.5,
            minimum_probability=0.7,
        )
        all_first = all_candidates.iloc[0]
        self.assertAlmostEqual(float(all_first.weighted_evidence_support), 3.8)
        self.assertAlmostEqual(float(all_first.weighted_singleton_support), 1.9)
        self.assertEqual(all_first.evidence_scope, "all")

    def test_assign_rule_tiers_is_nested(self) -> None:
        rules = pd.DataFrame(
            [
                {"cross_split_p80": False, "weighted_evidence_support": 20.0},
                {"cross_split_p80": True, "weighted_evidence_support": 2.5},
                {"cross_split_p80": True, "weighted_evidence_support": 5.0},
            ]
        )
        tiered = MODULE.assign_rule_tiers(rules)
        self.assertEqual(
            tiered["evidence_tier"].tolist(),
            [
                "SEMANTIC_P80_SUPPORT2",
                "SEMANTIC_CROSS_SPLIT_P80",
                "SEMANTIC_CROSS_SPLIT_SUPPORT5_P80",
            ],
        )


if __name__ == "__main__":
    unittest.main()
