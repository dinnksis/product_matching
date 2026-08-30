from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from item_pipeline.pair_rules import MutationRule, load_mutation_rules
from item_pipeline.pair_validation import (
    _metadata_mutation_rule,
    validate_mutation,
    validate_rule_anchor,
)
from item_pipeline.qwen import (
    build_mutation_prompt,
    build_rule_anchor_prompt,
    mutated_item_schema,
    rule_anchor_schema,
)
from item_pipeline.rule_schedule import SCHEDULE_VERSION, build_balanced_rule_schedule


def rule_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "source_rule_id": "source-age",
        "generation_rule_id": "rule-age",
        "generation_tier": "TRANSITION_TEST",
        "label": 1,
        "concept": "age_recommendation",
        "relation": "different_value",
        "semantic_family": "audience",
        "attribute_key": "Возраст",
        "allowed_categories": ["Детские товары"],
        "allowed_product_types": ["настольная игра"],
        "allowed_anchor_context_keys": ["Бренд"],
        "required_anchor_context_keys": ["Бренд"],
        "target_value_domain": ["5+", "6+", "8+"],
        "allowed_value_transitions": [["6+", "5+"], ["6+", "8+"]],
        "primary_task_safety_cap": 3,
    }
    payload.update(overrides)
    return payload


def load_one(payload: dict[str, object]) -> MutationRule:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rules.json"
        path.write_text(
            json.dumps([payload], ensure_ascii=False), encoding="utf-8"
        )
        return load_mutation_rules([path])[0]


def transition_rule(*, label: int = 1) -> MutationRule:
    return load_one(rule_payload(label=label))


def anchor_item(value: str = "6+") -> tuple[dict[str, object], list[dict[str, str]]]:
    item: dict[str, object] = {
        "id": -1,
        "name": f"настольная игра Нейтральный {value}",
        "attributes": {
            "Тип товара": "настольная игра",
            "Бренд": "Нейтральный",
            "Возраст": value,
        },
        "category": "Детские товары",
    }
    evidence = [
        {
            "generation_rule_id": "rule-age",
            "concept": "age_recommendation",
            "attribute_key": "Возраст",
            "attribute_value": value,
        }
    ]
    return item, evidence


class TransitionRuleTest(unittest.TestCase):
    def test_loader_canonicalizes_unordered_pairs_and_preserves_metadata(self) -> None:
        rule = transition_rule()
        self.assertEqual(
            rule.allowed_value_transitions,
            (("5+", "6+"), ("6+", "8+")),
        )
        self.assertEqual(rule.required_anchor_context_keys, ("Бренд",))
        payload = rule.prompt_payload()
        self.assertEqual(
            payload["allowed_value_transitions"],
            [["5+", "6+"], ["6+", "8+"]],
        )
        restored = _metadata_mutation_rule(
            {
                **payload,
                "allowed_categories": list(rule.allowed_categories),
                "source_path": rule.source_path,
            }
        )
        self.assertEqual(restored.allowed_value_transitions, rule.allowed_value_transitions)
        self.assertEqual(
            restored.required_anchor_context_keys,
            rule.required_anchor_context_keys,
        )

    def test_loader_rejects_malformed_duplicate_and_domain_inconsistent_pairs(self) -> None:
        invalid_values = {
            "wrong_pair_size": [["5+"]],
            "empty_member": [["5+", ""]],
            "same_member": [["5+", " 5+ "]],
            "reverse_duplicate": [["5+", "6+"], ["6+", "5+"]],
        }
        for name, transitions in invalid_values.items():
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "allowed_value_transitions"
            ):
                load_one(rule_payload(allowed_value_transitions=transitions))

        with self.assertRaisesRegex(ValueError, "outside target_value_domain"):
            load_one(
                rule_payload(allowed_value_transitions=[["5+", "12+"]])
            )
        with self.assertRaisesRegex(ValueError, "required context outside"):
            load_one(
                rule_payload(required_anchor_context_keys=["Производитель"])
            )

    def test_schema_and_prompts_expose_transition_and_required_context(self) -> None:
        rule = transition_rule()
        anchor_schema = rule_anchor_schema("Детские товары", [rule])
        attributes_schema = anchor_schema["properties"]["item"]["properties"][
            "attributes"
        ]
        self.assertIn("Бренд", attributes_schema["required"])
        self.assertEqual(
            attributes_schema["properties"]["Возраст"]["enum"],
            ["5+", "6+", "8+"],
        )

        mutation_schema = mutated_item_schema(
            "Детские товары", ["Тип товара", "Бренд", "Возраст"], [rule]
        )
        self.assertEqual(
            mutation_schema["properties"]["applications"]["items"]["properties"]
            ["new_value"]["enum"],
            ["5+", "6+", "8+"],
        )

        donor = {
            "id": 17,
            "name": "игра",
            "attributes": "{}",
            "category": "Детские товары",
        }
        anchor_prompt = json.loads(
            build_rule_anchor_prompt(donor, [rule]).split("\n\n", 1)[1]
        )
        self.assertEqual(anchor_prompt["required_anchor_context_keys"], ["Бренд"])
        self.assertEqual(
            anchor_prompt["required_targets"][0]["allowed_value_transitions"],
            [["5+", "6+"], ["6+", "8+"]],
        )
        item, evidence = anchor_item()
        mutation_prompt = json.loads(
            build_mutation_prompt(item, [rule], evidence).split("\n\n", 1)[1]
        )
        self.assertEqual(
            mutation_prompt["rules"][0]["allowed_value_transitions"],
            [["5+", "6+"], ["6+", "8+"]],
        )

    def test_anchor_requires_context_and_exact_transition_endpoint(self) -> None:
        rule = transition_rule()
        item, evidence = anchor_item()
        valid = validate_rule_anchor(
            item,
            "настольная игра",
            evidence,
            category="Детские товары",
            rules=[rule],
        )
        self.assertTrue(valid.valid, valid.reasons)

        missing_context = {
            **item,
            "name": "настольная игра 6+",
            "attributes": {
                "Тип товара": "настольная игра",
                "Возраст": "6+",
            },
        }
        missing = validate_rule_anchor(
            missing_context,
            "настольная игра",
            evidence,
            category="Детские товары",
            rules=[rule],
        )
        self.assertIn(
            "anchor_missing_required_context_key:rule-age:Бренд", missing.reasons
        )

        outside_item, outside_evidence = anchor_item("12+")
        outside = validate_rule_anchor(
            outside_item,
            "настольная игра",
            outside_evidence,
            category="Детские товары",
            rules=[replace(rule, target_value_domain=())],
        )
        self.assertIn(
            "anchor_evidence_0:outside_allowed_value_transitions", outside.reasons
        )

    def test_mutation_accepts_either_direction_and_rejects_other_pairs(self) -> None:
        rule = transition_rule()
        item, evidence = anchor_item("8+")
        anchor = {**item, "attributes": json.dumps(item["attributes"], ensure_ascii=False)}
        valid = validate_mutation(
            {
                **item,
                "name": "настольная игра Нейтральный 6+",
                "attributes": {**item["attributes"], "Возраст": "6+"},
            },
            [
                {
                    "generation_rule_id": "rule-age",
                    "concept": "age_recommendation",
                    "attribute_key": "Возраст",
                    "original_value": "8+",
                    "new_value": "6+",
                }
            ],
            anchor=anchor,
            rules=[rule],
            evidence=evidence,
        )
        self.assertTrue(valid.valid, valid.reasons)

        disallowed = validate_mutation(
            {
                **item,
                "name": "настольная игра Нейтральный 5+",
                "attributes": {**item["attributes"], "Возраст": "5+"},
            },
            [
                {
                    "generation_rule_id": "rule-age",
                    "concept": "age_recommendation",
                    "attribute_key": "Возраст",
                    "original_value": "8+",
                    "new_value": "5+",
                }
            ],
            anchor=anchor,
            rules=[rule],
            evidence=evidence,
        )
        self.assertIn("application_0:disallowed_value_transition", disallowed.reasons)

    def test_semantic_equivalence_is_allowed_only_for_positive_rule(self) -> None:
        base = MutationRule(
            generation_rule_id="rule-material",
            source_rule_id="source-material",
            generation_tier="TRANSITION_TEST",
            label=1,
            concept="material",
            relation="different_value",
            semantic_family="material",
            attribute_key="Материал",
            anchor_hint="",
            allowed_categories=("Аксессуары",),
            generation_action="change material wording",
            required_postcondition="preserve identity",
            source_path="test",
            allowed_product_types=("сумка",),
            allowed_anchor_context_keys=("Бренд",),
            required_anchor_context_keys=("Бренд",),
            allowed_value_transitions=(("искусственная кожа", "экокожа"),),
        )
        anchor_attributes = {
            "Тип товара": "сумка",
            "Бренд": "Нейтральный",
            "Материал": "искусственная кожа",
        }
        anchor = {
            "id": -1,
            "name": "сумка Нейтральный искусственная кожа",
            "attributes": json.dumps(anchor_attributes, ensure_ascii=False),
            "category": "Аксессуары",
        }
        mutation = {
            "name": "сумка Нейтральный экокожа",
            "attributes": {**anchor_attributes, "Материал": "экокожа"},
            "category": "Аксессуары",
        }
        applications = [
            {
                "generation_rule_id": "rule-material",
                "concept": "material",
                "attribute_key": "Материал",
                "original_value": "искусственная кожа",
                "new_value": "экокожа",
            }
        ]
        positive = validate_mutation(
            mutation, applications, anchor=anchor, rules=[base]
        )
        self.assertTrue(positive.valid, positive.reasons)

        negative = validate_mutation(
            mutation,
            applications,
            anchor=anchor,
            rules=[replace(base, label=0)],
        )
        self.assertIn(
            "application_0:semantically_equivalent_target_values", negative.reasons
        )

    def test_schedule_capacity_uses_transition_count_then_explicit_cap(self) -> None:
        rule = transition_rule()
        donors = pd.DataFrame(
            {"id": range(1, 7), "category": ["Детские товары"] * 6}
        )
        uncapped = build_balanced_rule_schedule(
            donors,
            [replace(rule, primary_task_safety_cap=None)],
            count=4,
            seed=23,
            two_rule_fraction=0.0,
            semantic_signature_limit=2,
        )
        profile_id = uncapped.eligible_profiles[0].profile_id
        self.assertEqual(uncapped.summary()["profile_primary_task_caps"][profile_id], 4)

        capped = build_balanced_rule_schedule(
            donors,
            [rule],
            count=3,
            seed=23,
            two_rule_fraction=0.0,
            semantic_signature_limit=2,
        )
        self.assertEqual(capped.summary()["profile_primary_task_caps"][profile_id], 3)
        self.assertIn("transition_capacity", SCHEDULE_VERSION)


if __name__ == "__main__":
    unittest.main()
