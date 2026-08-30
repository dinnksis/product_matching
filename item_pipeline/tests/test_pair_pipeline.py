from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import pandas as pd

from item_pipeline.cli import build_parser, main as cli_main
from item_pipeline.normalization import parse_attributes
from item_pipeline.pair_generate import (
    ATTEMPT_DIVERSITY_VERSION,
    _canonical_card_key,
    PairGenerationTask,
    _attempt_diversity_nonce,
    _semantic_pair_signature,
    run_pair_generation,
)
from item_pipeline.pair_rules import MutationRule
from item_pipeline.qwen import build_rule_anchor_prompt
from item_pipeline.rule_schedule import SCHEDULE_VERSION
from item_pipeline.pair_validation import (
    validate_mutation,
    validate_pair_dataset,
    validate_rule_anchor,
)


class DeterministicFakePairClient:
    model = "fake-qwen"
    temperature = 0.7
    max_tokens = 1400
    structured_output = False

    def __init__(self, items: pd.DataFrame) -> None:
        self.by_category = {
            str(row.category): parse_attributes(row.attributes)
            for row in items.itertuples(index=False)
        }
        self.names = {
            str(row.category): str(row.name) for row in items.itertuples(index=False)
        }

    def generate_anchor(self, prompt, *, category, rules, seed):
        del prompt
        values = {
            "battery_capacity": "5000 мАч",
            "ram_capacity": "8 ГБ",
        }
        attributes = {
            "Тип товара": "кабель",
            "модель": "cable-test",
            **{rule.attribute_key: values[rule.concept] for rule in rules},
        }
        evidence = [
            {
                "generation_rule_id": rule.generation_rule_id,
                "concept": rule.concept,
                "attribute_key": rule.attribute_key,
                "attribute_value": values[rule.concept],
            }
            for rule in rules
        ]
        generated_name = "кабель cable-test " + " ".join(
            values[rule.concept] for rule in rules
        )
        self.by_category[category] = attributes
        self.names[category] = generated_name
        return {
            "item": {
                "name": generated_name,
                "attributes": attributes,
                "category": category,
            },
            "product_type": "кабель",
            "evidence": evidence,
            "request_attempts": 1,
            "latency_seconds": 0.01,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "response_id": f"anchor-{seed}",
        }

    def mutate(self, prompt, *, category, attribute_keys, rules, seed):
        del prompt
        changed = dict(self.by_category[category])
        name = self.names[category]
        applications = []
        for rule in rules:
            if rule.concept == "battery_capacity":
                new = "6000 мАч"
            elif rule.concept == "ram_capacity":
                new = "16 ГБ"
            else:
                raise AssertionError(rule.concept)
            key = rule.attribute_key
            original = changed[key]
            changed[key] = new
            name = name.replace(original, new)
            applications.append(
                {
                    "generation_rule_id": rule.generation_rule_id,
                    "concept": rule.concept,
                    "original_value": original,
                    "new_value": new,
                    "attribute_key": key,
                }
            )
        self.assert_keys(attribute_keys, changed)
        return {
            "item": {"name": name, "attributes": changed, "category": category},
            "applications": applications,
            "request_attempts": 1,
            "latency_seconds": 0.01,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "response_id": f"response-{seed}",
        }

    @staticmethod
    def assert_keys(expected, actual):
        if set(expected) != set(actual):
            raise AssertionError((expected, actual))


class RepeatedSemanticPairClient:
    model = "fake-qwen"
    temperature = 0.7
    max_tokens = 1400
    structured_output = False

    def __init__(self) -> None:
        self.attributes: dict[str, str] = {}
        self.name = ""

    def generate_anchor(self, prompt, *, category, rules, seed):
        del prompt
        self.attributes = {
            "Тип товара": "чехол",
            "Бренд": f"context-{seed}",
            rules[0].attribute_key: "красный",
        }
        self.name = " ".join(self.attributes.values())
        return {
            "item": {
                "name": self.name,
                "attributes": dict(self.attributes),
                "category": category,
            },
            "product_type": "чехол",
            "evidence": [{
                "generation_rule_id": rules[0].generation_rule_id,
                "concept": rules[0].concept,
                "attribute_key": rules[0].attribute_key,
                "attribute_value": "красный",
            }],
            "request_attempts": 1,
            "latency_seconds": 0.01,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "response_id": f"anchor-{seed}",
        }

    def mutate(self, prompt, *, category, attribute_keys, rules, seed):
        del prompt
        if set(attribute_keys) != set(self.attributes):
            raise AssertionError((attribute_keys, self.attributes))
        changed = {**self.attributes, rules[0].attribute_key: "синий"}
        return {
            "item": {
                "name": self.name.replace("красный", "синий"),
                "attributes": changed,
                "category": category,
            },
            "applications": [{
                "generation_rule_id": rules[0].generation_rule_id,
                "concept": rules[0].concept,
                "attribute_key": rules[0].attribute_key,
                "original_value": "красный",
                "new_value": "синий",
            }],
            "request_attempts": 1,
            "latency_seconds": 0.01,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "response_id": f"mutation-{seed}",
        }


class GlobalFeedbackAwarePairClient:
    model = "fake-qwen"
    temperature = 0.7
    max_tokens = 1400
    structured_output = False

    def __init__(self) -> None:
        self.attributes: dict[str, str] = {}
        self.name = ""
        self.anchor_nonces: list[str] = []
        self.mutation_nonces: list[str] = []
        self.global_feedback_prompts = 0

    @staticmethod
    def _payload(prompt: str) -> dict:
        return json.loads(prompt.split("\n\n", 1)[1])

    def generate_anchor(self, prompt, *, category, rules, seed):
        payload = self._payload(prompt)
        self.anchor_nonces.append(payload["diversity_nonce_not_for_output"])
        has_global_feedback = bool(payload.get("previous_global_rejections"))
        self.global_feedback_prompts += int(has_global_feedback)
        color = "зелёный" if has_global_feedback else "красный"
        self.attributes = {
            "Тип товара": "чехол",
            "Бренд": f"context-{seed}",
            rules[0].attribute_key: color,
        }
        self.name = " ".join(self.attributes.values())
        return {
            "item": {
                "name": self.name,
                "attributes": dict(self.attributes),
                "category": category,
            },
            "product_type": "чехол",
            "evidence": [{
                "generation_rule_id": rules[0].generation_rule_id,
                "concept": rules[0].concept,
                "attribute_key": rules[0].attribute_key,
                "attribute_value": color,
            }],
            "request_attempts": 1,
            "latency_seconds": 0.01,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "response_id": f"anchor-{seed}",
        }

    def mutate(self, prompt, *, category, attribute_keys, rules, seed):
        payload = self._payload(prompt)
        self.mutation_nonces.append(payload["diversity_nonce_not_for_output"])
        has_global_feedback = bool(payload.get("previous_global_rejections"))
        self.global_feedback_prompts += int(has_global_feedback)
        original = self.attributes[rules[0].attribute_key]
        new = "жёлтый" if has_global_feedback else "синий"
        changed = {**self.attributes, rules[0].attribute_key: new}
        if set(attribute_keys) != set(changed):
            raise AssertionError((attribute_keys, changed))
        return {
            "item": {
                "name": self.name.replace(original, new),
                "attributes": changed,
                "category": category,
            },
            "applications": [{
                "generation_rule_id": rules[0].generation_rule_id,
                "concept": rules[0].concept,
                "attribute_key": rules[0].attribute_key,
                "original_value": original,
                "new_value": new,
            }],
            "request_attempts": 1,
            "latency_seconds": 0.01,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "response_id": f"mutation-{seed}",
        }


class PairPipelineTest(unittest.TestCase):
    def test_cli_rejects_zero_http_retries_before_io(self) -> None:
        for command in ("generate", "generate-pairs"):
            with self.subTest(command=command), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    cli_main([command, "--retries", "0"])
            self.assertEqual(raised.exception.code, 2)

    def test_canonical_card_key_normalizes_yo_and_attribute_text(self) -> None:
        left = {
            "name": "Комбинезон Лапочка зелёный",
            "attributes": json.dumps(
                {
                    "Тип товара": "комбинезон",
                    "Бренд": "Лапочка",
                    "Цвет товара": "зелёный",
                },
                ensure_ascii=False,
            ),
            "category": "Товары для животных",
        }
        right = {
            "name": "комбинезон лапочка зеленый",
            "attributes": json.dumps(
                {
                    "тип товара": "КОМБИНЕЗОН",
                    "бренд": "лапочка",
                    "цвет товара": "зеленый",
                },
                ensure_ascii=False,
            ),
            # Category is not part of the visible-card uniqueness contract.
            "category": "Одежда",
        }
        self.assertEqual(_canonical_card_key(left), _canonical_card_key(right))

    def test_attempt_diversity_nonce_covers_every_retry_coordinate(self) -> None:
        task = PairGenerationTask(
            task_index=7,
            mutated_id=-8,
            seed=101,
            anchor={"id": 5},
        )
        base = dict(
            task_seed_offset=0,
            task_retry_round=0,
            selection_attempt=1,
            stage="anchor",
            stage_attempt=1,
        )
        nonce = _attempt_diversity_nonce(task, **base)
        self.assertEqual(nonce, _attempt_diversity_nonce(task, **base))
        variants = []
        for field, value in (
            ("task_seed_offset", 10000),
            ("task_retry_round", 1),
            ("selection_attempt", 2),
            ("stage", "mutation"),
            ("stage_attempt", 2),
        ):
            arguments = {**base, field: value}
            variants.append(_attempt_diversity_nonce(task, **arguments))
        variants.append(
            _attempt_diversity_nonce(
                PairGenerationTask(
                    task_index=8,
                    mutated_id=-9,
                    seed=102,
                    anchor={"id": 6},
                ),
                **base,
            )
        )
        self.assertEqual(len({nonce, *variants}), 1 + len(variants))

    def test_anchor_prompt_forbids_extra_keys_when_rule_contexts_do_not_overlap(self) -> None:
        brand_rule = MutationRule(
            generation_rule_id="rule-brand",
            source_rule_id="source-brand",
            generation_tier="STAT_SCOPED",
            label=0,
            concept="brand",
            relation="different_value",
            semantic_family="identity",
            attribute_key="Бренд",
            anchor_hint="",
            allowed_categories=("Ювелирные изделия",),
            generation_action="change brand",
            required_postcondition="different brand",
            source_path="test",
            allowed_product_types=("кольцо",),
            allowed_anchor_context_keys=("Назначение",),
        )
        size_rule = MutationRule(
            generation_rule_id="rule-size",
            source_rule_id="source-size",
            generation_tier="STAT_SCOPED",
            label=0,
            concept="size",
            relation="different_value",
            semantic_family="size",
            attribute_key="Размер",
            anchor_hint="",
            allowed_categories=("Ювелирные изделия",),
            generation_action="change size",
            required_postcondition="different size",
            source_path="test",
            allowed_product_types=("кольцо",),
            allowed_anchor_context_keys=("Бренд",),
        )

        prompt = build_rule_anchor_prompt(
            {
                "id": 17,
                "name": "кольцо",
                "attributes": "{}",
                "category": "Ювелирные изделия",
            },
            [brand_rule, size_rule],
        )
        payload = json.loads(prompt.split("\n\n", 1)[1])
        contract = payload["local_validation_contract"]
        self.assertEqual(
            contract["attribute_keys_must_equal_exactly"],
            ["Тип товара", "Бренд", "Размер"],
        )
        self.assertIs(contract["additional_attribute_keys_forbidden"], True)
        self.assertEqual(payload["allowed_anchor_context_keys"], [])
        self.assertEqual(
            set(payload["required_output_json"]["item"]["attributes"]),
            {"Тип товара", "Бренд", "Размер"},
        )

    def test_semantic_signature_cli_limit_defaults_to_two_and_must_be_positive(self) -> None:
        parser = build_parser()
        self.assertEqual(
            parser.parse_args(["generate-pairs"]).semantic_signature_limit, 2
        )
        self.assertEqual(
            parser.parse_args(
                ["generate-pairs", "--semantic-signature-limit", "7"]
            ).semantic_signature_limit,
            7,
        )
        self.assertIsNone(
            parser.parse_args(["generate-pairs"]).label_one_fraction
        )
        self.assertEqual(
            parser.parse_args(
                ["generate-pairs", "--label-one-fraction", "0.5"]
            ).label_one_fraction,
            0.5,
        )
        with self.assertRaisesRegex(ValueError, "semantic_signature_limit"):
            run_pair_generation(
                items_path=Path("missing.parquet"),
                rule_paths=[],
                output_dir=Path("unused"),
                client=RepeatedSemanticPairClient(),
                system_prompt="test",
                count=1,
                seed=1,
                mutated_id_start=None,
                categories=None,
                tiers=None,
                workers=1,
                pair_attempts=1,
                checkpoint_every=1,
                two_rule_fraction=0.0,
                semantic_signature_limit=0,
            )

    def test_semantic_signature_is_direction_and_rule_order_invariant(self) -> None:
        applications = [
            {
                "concept": "material",
                "attribute_key": "Материал",
                "original_value": "Хлопок",
                "new_value": "Лён",
                "context_brand": "ignored-a",
            },
            {
                "concept": "color",
                "attribute_key": "Цвет товара",
                "original_value": "Красный",
                "new_value": "Синий",
            },
        ]
        reversed_applications = [
            {
                **application,
                "original_value": application["new_value"],
                "new_value": application["original_value"],
                "context_brand": "ignored-b",
            }
            for application in reversed(applications)
        ]
        signature = _semantic_pair_signature(
            " Одежда ", "Футболка", applications
        )
        self.assertEqual(
            signature,
            _semantic_pair_signature(
                "одежда", " футболка ", reversed_applications
            ),
        )
        self.assertNotEqual(
            signature,
            _semantic_pair_signature("Обувь", "футболка", applications),
        )
        self.assertNotEqual(
            signature,
            _semantic_pair_signature("Одежда", "рубашка", applications),
        )

    def test_semantic_signature_ignores_brand_and_context_fields(self) -> None:
        application = {
            "concept": "color",
            "attribute_key": "Цвет товара",
            "original_value": "красный",
            "new_value": "синий",
        }
        self.assertEqual(
            _semantic_pair_signature(
                "Одежда",
                "футболка",
                [{**application, "brand": "brand-a", "context": {"size": "m"}}],
            ),
            _semantic_pair_signature(
                "Одежда",
                "футболка",
                [{**application, "brand": "brand-b", "context": {"size": "xl"}}],
            ),
        )

    def test_anchor_canonical_title_rejects_hidden_word_or_number(self) -> None:
        rule = MutationRule(
            generation_rule_id="rule-battery",
            source_rule_id="source-battery",
            generation_tier="STAT_SCOPED",
            label=0,
            concept="battery_capacity",
            relation="different_value",
            semantic_family="capacity_spec",
            attribute_key="Емкость аккумулятора",
            anchor_hint="",
            allowed_categories=("Электроника",),
            generation_action="change capacity",
            required_postcondition="different capacity",
            source_path="test",
            allowed_product_types=("смартфон",),
            allowed_anchor_context_keys=("Бренд",),
        )
        attributes = {
            "Тип товара": "смартфон",
            "Бренд": "нейтральный бренд",
            "Емкость аккумулятора": "5000 мАч",
        }
        evidence = [{
            "generation_rule_id": "rule-battery",
            "concept": "battery_capacity",
            "attribute_key": "Емкость аккумулятора",
            "attribute_value": "5000 мАч",
        }]

        exact = validate_rule_anchor(
            {
                "name": "смартфон нейтральный бренд 5000 мАч",
                "attributes": attributes,
                "category": "Электроника",
            },
            "смартфон",
            evidence,
            category="Электроника",
            rules=[rule],
        )
        self.assertTrue(exact.valid, exact.reasons)

        for hidden_fragment in ("скрытое-слово", "128"):
            with self.subTest(hidden_fragment=hidden_fragment):
                invalid = validate_rule_anchor(
                    {
                        "name": (
                            "смартфон нейтральный бренд "
                            f"{hidden_fragment} 5000 мАч"
                        ),
                        "attributes": attributes,
                        "category": "Электроника",
                    },
                    "смартфон",
                    evidence,
                    category="Электроника",
                    rules=[rule],
                )
                self.assertFalse(invalid.valid)
                self.assertIn(
                    "anchor_name_not_exact_attribute_value_concatenation",
                    invalid.reasons,
                )

    def test_anchor_requires_exact_rule_attribute_and_value_in_name(self) -> None:
        rule = MutationRule(
            generation_rule_id="rule-oxygen",
            source_rule_id="source-oxygen",
            generation_tier="RARE_SAFE",
            label=0,
            concept="oxygen_permeability",
            relation="different_value",
            semantic_family="optical_prescription",
            attribute_key="Кислородопроницаемость",
            anchor_hint="contact lenses only",
            allowed_categories=("Аптека",),
            generation_action="change oxygen permeability",
            required_postcondition="same contact lens subtype",
            source_path="test",
        )
        result = validate_rule_anchor(
            {
                "name": "контактные линзы -2.00",
                "attributes": {
                    "Тип товара": "контактные линзы",
                    "Оптическая сила": "-2.00",
                    "Кислородопроницаемость": "100 Dk/t",
                },
                "category": "Аптека",
            },
            "контактные линзы",
            [{
                "generation_rule_id": "rule-oxygen",
                "concept": "oxygen_permeability",
                "attribute_key": "Кислородопроницаемость",
                "attribute_value": "100 Dk/t",
            }],
            category="Аптека",
            rules=[rule],
        )
        self.assertFalse(result.valid)
        self.assertIn("anchor_evidence_0:value_missing_from_name", result.reasons)

    def test_anchor_rejects_sku_or_article_in_attributes_and_name(self) -> None:
        rule = MutationRule(
            generation_rule_id="rule-battery",
            source_rule_id="source-battery",
            generation_tier="RARE_SAFE",
            label=0,
            concept="battery_capacity",
            relation="different_value",
            semantic_family="capacity_spec",
            attribute_key="Емкость аккумулятора",
            anchor_hint="",
            allowed_categories=("Электроника",),
            generation_action="change capacity",
            required_postcondition="different capacity",
            source_path="test",
        )
        result = validate_rule_anchor(
            {
                "name": "смартфон 5000 мач, sku: phone-5000",
                "attributes": {
                    "Тип товара": "смартфон",
                    "Емкость аккумулятора": "5000 мач",
                    "Номер артикула фильтра": "phone-5000",
                },
                "category": "Электроника",
            },
            "смартфон",
            [{
                "generation_rule_id": "rule-battery",
                "concept": "battery_capacity",
                "attribute_key": "Емкость аккумулятора",
                "attribute_value": "5000 мач",
            }],
            category="Электроника",
            rules=[rule],
        )
        self.assertFalse(result.valid)
        self.assertIn("anchor_forbidden_sku_or_article_attribute", result.reasons)
        self.assertIn("anchor_forbidden_sku_or_article_in_name", result.reasons)

    def test_anchor_requires_observed_product_type_and_rejects_dependency(self) -> None:
        rule = MutationRule(
            generation_rule_id="rule-brand",
            source_rule_id="source-brand",
            generation_tier="STAT_SCOPED",
            label=0,
            concept="brand",
            relation="different_value",
            semantic_family="identity",
            attribute_key="Бренд",
            anchor_hint="shoes only",
            allowed_categories=("Обувь",),
            generation_action="change brand",
            required_postcondition="preserve everything else",
            source_path="test",
            allowed_product_types=("ботинки",),
            forbidden_anchor_attribute_patterns=(r"(?:модель|model)",),
        )
        result = validate_rule_anchor(
            {
                "name": "кроссовки nike air zoom",
                "attributes": {
                    "Тип товара": "кроссовки",
                    "Бренд": "nike",
                    "Модель": "air zoom",
                },
                "category": "Обувь",
            },
            "кроссовки",
            [{
                "generation_rule_id": "rule-brand",
                "concept": "brand",
                "attribute_key": "Бренд",
                "attribute_value": "nike",
            }],
            category="Обувь",
            rules=[rule],
        )
        self.assertFalse(result.valid)
        self.assertIn("anchor_product_type_outside_rule_scope", result.reasons)
        self.assertIn(
            "anchor_forbidden_dependent_attribute:rule-brand", result.reasons
        )

    def test_anchor_rejects_allowed_type_used_only_as_compatible_product(self) -> None:
        rule = MutationRule(
            generation_rule_id="rule-model",
            source_rule_id="source-model",
            generation_tier="STAT_SCOPED",
            label=0,
            concept="model",
            relation="different_value",
            semantic_family="identity",
            attribute_key="Модель",
            anchor_hint="acoustic guitars only",
            allowed_categories=("Музыкальные инструменты",),
            generation_action="change model",
            required_postcondition="preserve everything else",
            source_path="test",
            allowed_product_types=("акустическая гитара",),
        )
        result = validate_rule_anchor(
            {
                "name": "струны для акустической гитары модель ag-2000 12-53",
                "attributes": {
                    "Тип товара": "акустическая гитара",
                    "Модель": "ag-2000",
                    "Калибр": "12-53",
                },
                "category": "Музыкальные инструменты",
            },
            "акустическая гитара",
            [
                {
                    "generation_rule_id": "rule-model",
                    "concept": "model",
                    "attribute_key": "Модель",
                    "attribute_value": "ag-2000",
                }
            ],
            category="Музыкальные инструменты",
            rules=[rule],
        )
        self.assertFalse(result.valid)
        self.assertIn("anchor_name_does_not_start_with_product_type", result.reasons)

    def test_mutation_requires_new_target_value_in_name(self) -> None:
        rule = MutationRule(
            generation_rule_id="rule-oxygen",
            source_rule_id="source-oxygen",
            generation_tier="RARE_SAFE",
            label=0,
            concept="oxygen_permeability",
            relation="different_value",
            semantic_family="optical_prescription",
            attribute_key="Кислородопроницаемость",
            anchor_hint="contact lenses only",
            allowed_categories=("Аптека",),
            generation_action="change oxygen permeability",
            required_postcondition="same contact lens subtype",
            source_path="test",
        )
        anchor = {
            "id": -1,
            "name": "контактные линзы 100 Dk/t",
            "attributes": json.dumps({
                "Тип товара": "контактные линзы",
                "Кислородопроницаемость": "100 Dk/t",
                "Бренд": "test",
            }),
            "category": "Аптека",
        }
        result = validate_mutation(
            {
                "name": "контактные линзы",
                "attributes": {
                    "Тип товара": "контактные линзы",
                    "Кислородопроницаемость": "160 Dk/t",
                    "Бренд": "test",
                },
                "category": "Аптека",
            },
            [{
                "generation_rule_id": "rule-oxygen",
                "concept": "oxygen_permeability",
                "attribute_key": "Кислородопроницаемость",
                "original_value": "100 Dk/t",
                "new_value": "160 Dk/t",
            }],
            anchor=anchor,
            rules=[rule],
            evidence=[{
                "generation_rule_id": "rule-oxygen",
                "concept": "oxygen_permeability",
                "attribute_key": "Кислородопроницаемость",
                "attribute_value": "100 Dk/t",
            }],
        )
        self.assertFalse(result.valid)
        self.assertIn("application_0:new_value_missing_from_name", result.reasons)

    def test_mutation_rejects_semantically_equivalent_target_values(self) -> None:
        def validate_change(
            *, concept: str, attribute_key: str, original: str, new: str
        ):
            rule = MutationRule(
                generation_rule_id=f"rule-{concept}",
                source_rule_id=f"source-{concept}",
                generation_tier="STAT_SCOPED",
                label=0,
                concept=concept,
                relation="different_value",
                semantic_family="test",
                attribute_key=attribute_key,
                anchor_hint="",
                allowed_categories=("Тестовая категория",),
                generation_action="change target",
                required_postcondition="different physical value",
                source_path="test",
            )
            anchor_attributes = {
                "Тип товара": "товар",
                "Бренд": "нейтральный",
                attribute_key: original,
            }
            anchor = {
                "id": -1,
                "name": f"товар нейтральный {original}",
                "attributes": json.dumps(anchor_attributes, ensure_ascii=False),
                "category": "Тестовая категория",
            }
            return validate_mutation(
                {
                    "name": f"товар нейтральный {new}",
                    "attributes": {**anchor_attributes, attribute_key: new},
                    "category": "Тестовая категория",
                },
                [{
                    "generation_rule_id": rule.generation_rule_id,
                    "concept": concept,
                    "attribute_key": attribute_key,
                    "original_value": original,
                    "new_value": new,
                }],
                anchor=anchor,
                rules=[rule],
            )

        material_alias = validate_change(
            concept="material",
            attribute_key="Материал",
            original="искусственная кожа",
            new="экокожа",
        )
        self.assertFalse(material_alias.valid)
        self.assertIn(
            "application_0:semantically_equivalent_target_values",
            material_alias.reasons,
        )

        physically_different_material = validate_change(
            concept="material",
            attribute_key="Материал",
            original="искусственная кожа",
            new="натуральная кожа",
        )
        self.assertTrue(
            physically_different_material.valid,
            physically_different_material.reasons,
        )

        unit_only_numeric_change = validate_change(
            concept="package_quantity",
            attribute_key="Количество в упаковке",
            original="1",
            new="1 шт",
        )
        self.assertFalse(unit_only_numeric_change.valid)
        self.assertIn(
            "application_0:semantically_equivalent_target_values",
            unit_only_numeric_change.reasons,
        )

    def test_mutation_rejects_a_rule_from_another_category(self) -> None:
        anchor = {
            "id": -1,
            "name": "кабель красный",
            "attributes": json.dumps({"тип": "кабель", "цвет": "красный"}),
            "category": "Электроника",
        }
        rule = MutationRule(
            generation_rule_id="rule-color",
            source_rule_id="source-color",
            generation_tier="RARE_SAFE",
            label=0,
            concept="color",
            relation="different_value",
            semantic_family="categorical_variant",
            attribute_key="цвет",
            anchor_hint="",
            allowed_categories=("Обувь",),
            generation_action="change color",
            required_postcondition="different explicit color",
            source_path="test",
        )
        result = validate_mutation(
            {
                "name": "кабель синий",
                "attributes": {"тип": "кабель", "цвет": "синий"},
                "category": "Электроника",
            },
            [
                {
                    "generation_rule_id": "rule-color",
                    "concept": "color",
                    "original_value": "красный",
                    "new_value": "синий",
                    "attribute_key": "цвет",
                }
            ],
            anchor=anchor,
            rules=[rule],
        )
        self.assertFalse(result.valid)
        self.assertIn("rule_not_allowed_for_category", result.reasons)

    def test_mutation_rejects_hidden_second_title_change(self) -> None:
        rule = MutationRule(
            generation_rule_id="rule-car-brand",
            source_rule_id="source-car-brand",
            generation_tier="STAT_SCOPED",
            label=0,
            concept="car_brand",
            relation="different_value",
            semantic_family="identity",
            attribute_key="Марка автомобиля",
            anchor_hint="toy car",
            allowed_categories=("Детские товары",),
            generation_action="change brand",
            required_postcondition="exact title substitution",
            source_path="test",
        )
        anchor = {
            "id": -1,
            "name": "машинка toyota camry красная",
            "attributes": json.dumps({
                "Тип товара": "машинка",
                "Марка автомобиля": "toyota",
                "Цвет": "красная",
            }, ensure_ascii=False),
            "category": "Детские товары",
        }
        result = validate_mutation(
            {
                "name": "машинка honda accord красная",
                "attributes": {
                    "Тип товара": "машинка",
                    "Марка автомобиля": "honda",
                    "Цвет": "красная",
                },
                "category": "Детские товары",
            },
            [{
                "generation_rule_id": "rule-car-brand",
                "concept": "car_brand",
                "attribute_key": "Марка автомобиля",
                "original_value": "toyota",
                "new_value": "honda",
            }],
            anchor=anchor,
            rules=[rule],
        )
        self.assertFalse(result.valid)
        self.assertIn("title_not_exact_target_substitution", result.reasons)

    def test_pair_dataset_detects_order_insensitive_duplicate_cards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items_path, pairs_path = root / "items.parquet", root / "pairs.parquet"
            pd.DataFrame([
                {"id": 1, "name": "товар", "attributes": '{"a":"1","b":"2"}', "category": "c"},
                {"id": 2, "name": "другой", "attributes": '{"a":"2"}', "category": "c"},
                {"id": 3, "name": "товар", "attributes": '{"b":"2","a":"1"}', "category": "c"},
                {"id": 4, "name": "еще", "attributes": '{"a":"3"}', "category": "c"},
            ]).to_parquet(items_path, index=False)
            pd.DataFrame([
                {"id1": 1, "id2": 2, "target": 0},
                {"id1": 3, "id2": 4, "target": 0},
            ]).to_parquet(pairs_path, index=False)
            report = validate_pair_dataset(items_path, pairs_path)
        self.assertFalse(report["valid"])
        self.assertEqual(report["duplicate_full_card_rows"], 2)

    def test_pair_generation_is_category_safe_validated_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items_path = root / "base_items.parquet"
            rules_path = root / "rules.json"
            output = root / "pairs"
            attributes = {
                "тип": "кабель",
                "модель": "cable-test",
                "цвет": "красный",
                "длина кабеля": "1 м",
            }
            base_items = pd.DataFrame(
                [
                    {
                        "id": -1,
                        "name": "кабель cable-test красный 1 м",
                        "attributes": json.dumps(attributes, ensure_ascii=False),
                        "category": "Электроника",
                    }
                ]
            )
            base_items.to_parquet(items_path, index=False)
            rules = [
                {
                    "generation_rule_id": "rule-battery",
                    "source_rule_id": "source-battery",
                    "generation_tier": "RARE_SAFE",
                    "label": 0,
                    "concept": "battery_capacity",
                    "relation": "different_value",
                    "semantic_family": "capacity_spec",
                    "allowed_categories": ["Электроника"],
                    "generation_action": "change battery capacity",
                    "required_postcondition": "different explicit capacity",
                },
                {
                    "generation_rule_id": "rule-ram",
                    "source_rule_id": "source-ram",
                    "generation_tier": "RARE_SAFE",
                    "label": 0,
                    "concept": "ram_capacity",
                    "relation": "different_value",
                    "semantic_family": "capacity_spec",
                    "allowed_categories": ["Электроника"],
                    "generation_action": "change RAM capacity",
                    "required_postcondition": "same unit and different capacity",
                },
            ]
            rules_path.write_text(
                json.dumps(rules, ensure_ascii=False), encoding="utf-8"
            )
            arguments = dict(
                items_path=items_path,
                rule_paths=[rules_path],
                output_dir=output,
                client=DeterministicFakePairClient(base_items),
                system_prompt="test prompt",
                count=1,
                seed=17,
                mutated_id_start=None,
                categories=None,
                tiers=None,
                workers=2,
                pair_attempts=2,
                checkpoint_every=1,
                two_rule_fraction=1.0,
            )
            summary = run_pair_generation(**arguments)
            self.assertEqual(summary["generated_pairs"], 1)
            self.assertTrue(summary["validation_valid"])
            self.assertEqual(summary["version"], "rule_first_pair_generation_v3")
            self.assertIs(summary["global_duplicate_card_retry"], True)
            self.assertIs(summary["semantic_signature_retry"], True)
            self.assertEqual(summary["semantic_signature_limit"], 2)
            self.assertIs(summary["balanced_rule_schedule"], True)
            self.assertEqual(
                summary["rule_schedule_version"], SCHEDULE_VERSION
            )
            self.assertEqual(len(summary["rule_schedule_sha256"]), 64)
            self.assertEqual(summary["eligible_rules"], 2)
            self.assertEqual(summary["primary_rule_coverage"], 1)
            self.assertEqual(summary["total_rule_coverage"], 2)
            self.assertEqual(summary["realized_primary_rule_coverage"], 1)
            self.assertEqual(summary["scheduled_two_rule_tasks"], 1)
            self.assertEqual(summary["temperature"], 0.7)
            self.assertEqual(summary["max_tokens"], 1400)
            pairs = pd.read_parquet(output / "pairs.parquet")
            self.assertEqual(pairs.to_dict("records"), [{"id1": -1, "id2": -2, "target": 0}])
            items = pd.read_parquet(output / "items.parquet")
            self.assertEqual(len(items), 2)
            metadata = pd.read_parquet(output / "pair_generation_metadata.parquet")
            self.assertEqual(int(metadata.iloc[0]["rule_count"]), 2)
            self.assertTrue(bool(metadata.iloc[0]["balanced_rule_schedule"]))
            self.assertEqual(
                metadata.iloc[0]["rule_schedule_sha256"],
                summary["rule_schedule_sha256"],
            )
            self.assertEqual(
                json.loads(metadata.iloc[0]["scheduled_rule_ids"]),
                json.loads(metadata.iloc[0]["rule_ids"]),
            )

            resumed = run_pair_generation(**arguments)
            self.assertEqual(resumed["generated_pairs"], 1)
            report = validate_pair_dataset(
                output / "items.parquet",
                output / "pairs.parquet",
                metadata_path=output / "pair_generation_metadata.parquet",
            )
            self.assertTrue(report["valid"])
            mismatched_pairs = pairs.copy()
            mismatched_pairs["target"] = 1
            mismatched_path = root / "mismatched_pairs.parquet"
            mismatched_pairs.to_parquet(mismatched_path, index=False)
            mismatched_report = validate_pair_dataset(
                output / "items.parquet",
                mismatched_path,
                metadata_path=output / "pair_generation_metadata.parquet",
            )
            self.assertFalse(mismatched_report["valid"])
            self.assertGreater(mismatched_report["metadata_errors"], 0)

    def test_pair_generation_enforces_mixed_label_quota_and_resume_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items_path = root / "base_items.parquet"
            rules_path = root / "rules.json"
            output = root / "pairs"
            base_items = pd.DataFrame(
                [
                    {
                        "id": source_id,
                        "name": f"style donor {source_id}",
                        "attributes": json.dumps({"тип": "кабель"}, ensure_ascii=False),
                        "category": "Электроника",
                    }
                    for source_id in (-1, -2)
                ]
            )
            base_items.to_parquet(items_path, index=False)
            rules_path.write_text(
                json.dumps(
                    [
                        {
                            "generation_rule_id": "rule-battery-zero",
                            "source_rule_id": "source-battery-zero",
                            "generation_tier": "STAT_SCOPED",
                            "label": 0,
                            "concept": "battery_capacity",
                            "relation": "different_value",
                            "semantic_family": "capacity_spec",
                            "allowed_categories": ["Электроника"],
                            "generation_action": "change battery capacity",
                            "required_postcondition": "different capacity",
                        },
                        {
                            "generation_rule_id": "rule-ram-one",
                            "source_rule_id": "source-ram-one",
                            "generation_tier": "STAT_SCOPED",
                            "label": 1,
                            "concept": "ram_capacity",
                            "relation": "different_value",
                            "semantic_family": "capacity_spec",
                            "allowed_categories": ["Электроника"],
                            "generation_action": "change RAM capacity",
                            "required_postcondition": "different capacity",
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            arguments = dict(
                items_path=items_path,
                rule_paths=[rules_path],
                output_dir=output,
                client=DeterministicFakePairClient(base_items),
                system_prompt="test prompt",
                count=2,
                seed=47,
                mutated_id_start=-10,
                categories=None,
                tiers=None,
                workers=1,
                pair_attempts=1,
                checkpoint_every=1,
                two_rule_fraction=0.0,
                label_one_fraction=0.5,
            )
            summary = run_pair_generation(**arguments)
            self.assertEqual(summary["planned_target_counts"], {"0": 1, "1": 1})
            self.assertEqual(summary["realized_target_counts"], {"0": 1, "1": 1})
            self.assertEqual(summary["realized_label_one_fraction"], 0.5)
            pairs = pd.read_parquet(output / "pairs.parquet")
            self.assertEqual(pairs["target"].value_counts().to_dict(), {0: 1, 1: 1})
            metadata_path = output / "pair_generation_metadata.parquet"
            metadata = pd.read_parquet(metadata_path)
            self.assertEqual(set(metadata["scheduled_target"].astype(int)), {0, 1})
            self.assertTrue(metadata["label_quota_enabled"].astype(bool).all())
            self.assertEqual(
                set(metadata["requested_label_one_fraction"].astype(float)),
                {0.5},
            )
            self.assertEqual(
                {tuple(json.loads(value).items()) for value in metadata["planned_target_counts_json"]},
                {(('0', 1), ('1', 1))},
            )

            metadata.loc[metadata.index[0], "requested_label_one_fraction"] = 0.4
            metadata.to_parquet(metadata_path, index=False)
            with self.assertRaisesRegex(
                ValueError, "requested_label_one_fraction"
            ):
                run_pair_generation(**arguments)

    def test_saturated_bundle_feedback_diversifies_new_tasks_and_retries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items_path = root / "base_items.parquet"
            rules_path = root / "rules.json"
            output = root / "pairs"
            pd.DataFrame([
                {
                    "id": source_id,
                    "name": f"style donor {source_id}",
                    "attributes": json.dumps({"тип": "чехол"}, ensure_ascii=False),
                    "category": "Электроника",
                }
                for source_id in (101, 102, 103, 104)
            ]).to_parquet(items_path, index=False)
            rules_path.write_text(
                json.dumps(
                    [{
                        "generation_rule_id": "rule-color",
                        "source_rule_id": "source-color",
                        "generation_tier": "STAT_SCOPED",
                        "label": 0,
                        "concept": "color",
                        "relation": "different_value",
                        "semantic_family": "categorical_variant",
                        "attribute_key": "Цвет товара",
                        "allowed_categories": ["Электроника"],
                        "allowed_product_types": ["чехол"],
                        "allowed_anchor_context_keys": ["Бренд"],
                        "generation_action": "change color",
                        "required_postcondition": "different color",
                    }],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            client = GlobalFeedbackAwarePairClient()
            summary = run_pair_generation(
                items_path=items_path,
                rule_paths=[rules_path],
                output_dir=output,
                client=client,
                system_prompt="test prompt",
                count=4,
                seed=17,
                mutated_id_start=None,
                categories=None,
                tiers=None,
                workers=1,
                pair_attempts=1,
                anchor_attempts=1,
                mutation_attempts=1,
                checkpoint_every=1,
                two_rule_fraction=0.0,
                task_retries=2,
                semantic_signature_limit=2,
            )

            self.assertEqual(summary["generated_pairs"], 4)
            self.assertEqual(summary["pending"], 0)
            self.assertEqual(
                summary["attempt_diversity_version"],
                ATTEMPT_DIVERSITY_VERSION,
            )
            self.assertGreater(client.global_feedback_prompts, 0)
            self.assertEqual(
                len(client.anchor_nonces), len(set(client.anchor_nonces))
            )
            self.assertEqual(
                len(client.mutation_nonces), len(set(client.mutation_nonces))
            )
            metadata = pd.read_parquet(
                output / "pair_generation_metadata.parquet"
            )
            self.assertEqual(metadata["semantic_signature"].nunique(), 2)
            self.assertEqual(
                set(metadata["attempt_diversity_version"]),
                {ATTEMPT_DIVERSITY_VERSION},
            )
            self.assertEqual(set(metadata["anchor_attempt"].astype(int)), {1})
            self.assertEqual(set(metadata["mutation_attempt"].astype(int)), {1})
            self.assertGreater(
                int(metadata["global_rejection_feedback_count"].max()), 0
            )
            self.assertTrue(
                metadata["anchor_diversity_nonce_sha256"]
                .astype(str)
                .str.fullmatch(r"[0-9a-f]{64}")
                .all()
            )
            self.assertTrue(
                metadata["mutation_diversity_nonce_sha256"]
                .astype(str)
                .str.fullmatch(r"[0-9a-f]{64}")
                .all()
            )

    def test_semantic_signature_cap_is_restored_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items_path = root / "base_items.parquet"
            rules_path = root / "rules.json"
            output = root / "pairs"
            pd.DataFrame([
                {
                    "id": source_id,
                    "name": f"style donor {source_id}",
                    "attributes": json.dumps({"тип": "чехол"}, ensure_ascii=False),
                    "category": "Электроника",
                }
                for source_id in (101, 102, 103)
            ]).to_parquet(items_path, index=False)
            rules_path.write_text(
                json.dumps(
                    [{
                        "generation_rule_id": "rule-color",
                        "source_rule_id": "source-color",
                        "generation_tier": "STAT_SCOPED",
                        "label": 0,
                        "concept": "color",
                        "relation": "different_value",
                        "semantic_family": "categorical_variant",
                        "attribute_key": "Цвет товара",
                        "allowed_categories": ["Электроника"],
                        "allowed_product_types": ["чехол"],
                        "allowed_anchor_context_keys": ["Бренд"],
                        "generation_action": "change color",
                        "required_postcondition": "different color",
                    }],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            arguments = dict(
                items_path=items_path,
                rule_paths=[rules_path],
                output_dir=output,
                client=RepeatedSemanticPairClient(),
                system_prompt="test prompt",
                count=3,
                seed=17,
                mutated_id_start=None,
                categories=None,
                tiers=None,
                workers=1,
                pair_attempts=1,
                anchor_attempts=1,
                mutation_attempts=1,
                checkpoint_every=1,
                two_rule_fraction=0.0,
                task_retries=1,
                semantic_signature_limit=2,
            )

            first = run_pair_generation(**arguments)
            self.assertEqual(first["generated_pairs"], 2)
            self.assertEqual(first["pending"], 1)
            self.assertEqual(first["semantic_signature_retry_events"], 2)
            self.assertEqual(first["semantic_signature_retry_events_this_run"], 2)
            self.assertEqual(first["semantic_signature_unique_count"], 1)
            self.assertEqual(first["semantic_signature_max_count"], 2)
            metadata = pd.read_parquet(
                output / "pair_generation_metadata.parquet"
            )
            self.assertEqual(metadata["semantic_signature"].nunique(), 1)
            self.assertEqual(len(metadata), 2)

            resumed = run_pair_generation(**arguments)
            self.assertEqual(resumed["generated_pairs"], 2)
            self.assertEqual(resumed["pending"], 1)
            self.assertEqual(resumed["semantic_signature_retry_events"], 4)
            self.assertEqual(
                resumed["semantic_signature_retry_events_this_run"], 2
            )
            self.assertEqual(resumed["semantic_signature_max_count"], 2)
            self.assertEqual(
                len(pd.read_parquet(output / "pair_generation_metadata.parquet")),
                2,
            )


if __name__ == "__main__":
    unittest.main()
