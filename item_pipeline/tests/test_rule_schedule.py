from __future__ import annotations

import unittest

import pandas as pd

from item_pipeline.pair_rules import MutationRule, rules_are_product_compatible
from item_pipeline.rule_schedule import (
    balanced_category_quotas,
    build_balanced_rule_schedule,
)


def make_rule(
    rule_id: str,
    *,
    category: str,
    concept: str,
    product_types: tuple[str, ...] = ("товар",),
    target_value_domain: tuple[str, ...] = (),
    primary_task_safety_cap: int | None = None,
    label: int = 0,
) -> MutationRule:
    return MutationRule(
        generation_rule_id=rule_id,
        source_rule_id=f"source-{rule_id}",
        generation_tier="MAIN",
        label=label,
        concept=concept,
        relation="different_value",
        semantic_family="test",
        attribute_key=f"Ключ {concept}",
        anchor_hint="",
        allowed_categories=(category,),
        generation_action=f"change {concept}",
        required_postcondition=f"different {concept}",
        source_path="test-rules.json",
        allowed_product_types=product_types,
        allowed_anchor_context_keys=("Бренд",),
        target_value_domain=target_value_domain,
        primary_task_safety_cap=primary_task_safety_cap,
    )


def donors(category_counts: dict[str, int]) -> pd.DataFrame:
    rows = []
    donor_id = 1
    for category, count in category_counts.items():
        for _ in range(count):
            rows.append({"id": donor_id, "category": category})
            donor_id += 1
    return pd.DataFrame(rows)


class BalancedRuleScheduleTest(unittest.TestCase):
    def test_explicit_label_quota_is_exact_deterministic_and_never_mixed(self) -> None:
        rules = [
            make_rule(
                f"label-{label}-{concept}",
                category="c",
                concept=concept,
                label=label,
            )
            for label in (0, 1)
            for concept in ("color", "material")
        ]
        schedule = build_balanced_rule_schedule(
            donors({"c": 20}),
            rules,
            count=10,
            seed=13,
            two_rule_fraction=0.6,
            label_one_fraction=0.5,
        )
        summary = schedule.summary()
        self.assertTrue(summary["label_quota_enabled"])
        self.assertEqual(summary["requested_label_one_fraction"], 0.5)
        self.assertEqual(summary["planned_target_counts"], {"0": 5, "1": 5})
        self.assertEqual(summary["planned_target_fractions"], {"0": 0.5, "1": 0.5})
        self.assertEqual(summary["scheduled_two_rule_tasks"], 6)
        self.assertTrue(
            all(len({rule.label for rule in entry.rules}) == 1 for entry in schedule.entries)
        )

        reordered = build_balanced_rule_schedule(
            donors({"c": 20}).sample(frac=1.0, random_state=7),
            list(reversed(rules)),
            count=10,
            seed=13,
            two_rule_fraction=0.6,
            label_one_fraction=0.5,
        )
        self.assertEqual(schedule.schedule_sha256, reordered.schedule_sha256)
        self.assertEqual(schedule.provenance_rows(), reordered.provenance_rows())

    def test_explicit_label_quota_reserves_scarce_category_donors(self) -> None:
        rules = [
            make_rule("zero-a", category="a", concept="color", label=0),
            make_rule("zero-b", category="b", concept="color", label=0),
            make_rule("one-a", category="a", concept="material", label=1),
        ]
        schedule = build_balanced_rule_schedule(
            donors({"a": 3, "b": 7}),
            rules,
            count=10,
            seed=19,
            two_rule_fraction=0.0,
            label_one_fraction=0.3,
        )
        summary = schedule.summary()
        self.assertEqual(summary["planned_target_counts"], {"0": 7, "1": 3})
        self.assertEqual(
            summary["label_category_task_quotas"],
            {"0": {"b": 7}, "1": {"a": 3}},
        )

    def test_explicit_label_quota_rejects_missing_requested_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "label quota is infeasible"):
            build_balanced_rule_schedule(
                donors({"c": 10}),
                [make_rule("zero", category="c", concept="color", label=0)],
                count=10,
                seed=21,
                two_rule_fraction=0.0,
                label_one_fraction=0.5,
            )

    def test_category_quota_is_profile_weighted_and_respects_capacity(self) -> None:
        quotas = balanced_category_quotas(
            {"small": 100, "large": 100},
            {"small": 2, "large": 3},
            count=50,
            seed=17,
        )
        self.assertEqual(quotas, {"large": 30, "small": 20})

        capped = balanced_category_quotas(
            {"small": 100, "large": 10},
            {"small": 1, "large": 3},
            count=30,
            seed=17,
        )
        self.assertEqual(capped, {"large": 10, "small": 20})

    def test_primary_profiles_are_covered_with_at_most_one_row_skew(self) -> None:
        rules = [
            make_rule(f"a-{index}", category="a", concept=f"a_concept_{index}")
            for index in range(2)
        ] + [
            make_rule(f"b-{index}", category="b", concept=f"b_concept_{index}")
            for index in range(4)
        ]
        schedule = build_balanced_rule_schedule(
            donors({"a": 30, "b": 50}),
            rules,
            count=60,
            seed=23,
            two_rule_fraction=0.0,
        )
        summary = schedule.summary()
        self.assertEqual(summary["category_task_quotas"], {"a": 20, "b": 40})
        self.assertEqual(summary["primary_rule_coverage"], 6)
        self.assertEqual(summary["primary_rule_profile_coverage"], 6)
        self.assertEqual(
            set(summary["primary_rule_profile_usage"].values()),
            {10},
        )
        self.assertTrue(
            all(
                values["skew"] <= 1
                for values in summary["category_primary_profile_ranges"].values()
            )
        )
        self.assertEqual(len({entry.donor_id for entry in schedule.entries}), 60)

    def test_exact_two_rule_fraction_and_product_compatibility(self) -> None:
        rules = [
            make_rule("color", category="c", concept="color"),
            make_rule("material", category="c", concept="material"),
        ]
        schedule = build_balanced_rule_schedule(
            donors({"c": 20}),
            rules,
            count=10,
            seed=29,
            two_rule_fraction=0.4,
        )
        summary = schedule.summary()
        self.assertEqual(summary["requested_two_rule_tasks"], 4)
        self.assertEqual(summary["eligible_two_rule_primary_slots"], 10)
        self.assertEqual(summary["scheduled_two_rule_tasks"], 4)
        self.assertAlmostEqual(summary["scheduled_two_rule_fraction"], 0.4)
        self.assertFalse(summary["two_rule_target_clipped"])
        for entry in schedule.entries:
            self.assertTrue(all(rule.allowed_categories == ("c",) for rule in entry.rules))
            self.assertTrue(
                all(rule.allowed_product_types == ("товар",) for rule in entry.rules)
            )
            if len(entry.rules) == 2:
                self.assertTrue(rules_are_product_compatible(*entry.rules))

    def test_infeasible_two_rule_target_is_clipped_and_reported(self) -> None:
        rules = [
            make_rule("color", category="pairable", concept="color"),
            make_rule("material", category="pairable", concept="material"),
            make_rule("solo-a", category="solo", concept="model"),
            make_rule("solo-b", category="solo", concept="package_quantity"),
        ]
        schedule = build_balanced_rule_schedule(
            donors({"pairable": 10, "solo": 10}),
            rules,
            count=20,
            seed=31,
            two_rule_fraction=0.8,
        )
        summary = schedule.summary()
        self.assertEqual(summary["requested_two_rule_tasks"], 16)
        self.assertEqual(summary["eligible_two_rule_primary_slots"], 10)
        self.assertEqual(summary["scheduled_two_rule_tasks"], 7)
        self.assertTrue(summary["two_rule_target_clipped"])

    def test_finite_primary_cap_is_resolved_and_residual_is_redistributed(self) -> None:
        finite = make_rule(
            "finite",
            category="finite",
            concept="model",
            target_value_domain=("a", "b", "c"),
            primary_task_safety_cap=2,
        )
        rules = [finite] + [
            make_rule(f"open-{index}", category="open", concept=f"open_{index}")
            for index in range(3)
        ]
        schedule = build_balanced_rule_schedule(
            donors({"finite": 20, "open": 20}),
            rules,
            count=20,
            seed=33,
            two_rule_fraction=0.0,
            semantic_signature_limit=2,
        )
        summary = schedule.summary()
        finite_profile = next(
            profile.profile_id
            for profile in schedule.eligible_profiles
            if profile.rule.generation_rule_id == "finite"
        )
        self.assertEqual(summary["profile_primary_task_caps"], {finite_profile: 2})
        self.assertEqual(summary["primary_rule_profile_usage"][finite_profile], 2)
        self.assertEqual(sum(summary["primary_rule_profile_usage"].values()), 20)
        self.assertEqual(
            summary["capacity_saturated_single_rule_profiles"], [finite_profile]
        )
        self.assertLessEqual(summary["balanced_total_rule_profile_usage_skew"], 1)

    def test_multi_type_rule_is_split_into_scoped_balanced_profiles(self) -> None:
        rules = [
            make_rule(
                "multi",
                category="c",
                concept="model",
                product_types=("телефон", "планшет"),
            )
        ]
        schedule = build_balanced_rule_schedule(
            donors({"c": 8}),
            rules,
            count=8,
            seed=37,
            two_rule_fraction=0.0,
        )
        summary = schedule.summary()
        self.assertEqual(summary["eligible_rules"], 1)
        self.assertEqual(summary["eligible_rule_profiles"], 2)
        self.assertEqual(set(summary["primary_rule_profile_usage"].values()), {4})
        observed_types = {
            entry.primary.rule.allowed_product_types for entry in schedule.entries
        }
        self.assertEqual(observed_types, {("телефон",), ("планшет",)})

    def test_schedule_is_input_order_independent_and_seed_deterministic(self) -> None:
        rule_values = [
            make_rule("color", category="c", concept="color"),
            make_rule("material", category="c", concept="material"),
            make_rule("model", category="c", concept="model"),
        ]
        donor_values = donors({"c": 20})
        first = build_balanced_rule_schedule(
            donor_values,
            rule_values,
            count=15,
            seed=41,
            two_rule_fraction=0.2,
        )
        reordered = build_balanced_rule_schedule(
            donor_values.sample(frac=1.0, random_state=3).reset_index(drop=True),
            list(reversed(rule_values)),
            count=15,
            seed=41,
            two_rule_fraction=0.2,
        )
        self.assertEqual(first.schedule_sha256, reordered.schedule_sha256)
        self.assertEqual(first.provenance_rows(), reordered.provenance_rows())
        self.assertEqual(
            first.rules_for_task(7),
            first.rules_for_task(7),
            "retries must reuse the exact scheduled bundle",
        )

        other_seed = build_balanced_rule_schedule(
            donor_values,
            rule_values,
            count=15,
            seed=42,
            two_rule_fraction=0.2,
        )
        self.assertNotEqual(first.schedule_sha256, other_seed.schedule_sha256)
        self.assertEqual(
            first.summary()["category_task_quotas"],
            other_seed.summary()["category_task_quotas"],
        )


if __name__ == "__main__":
    unittest.main()
