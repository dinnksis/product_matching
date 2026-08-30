from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/build_soft_positive_human_skeleton_pairs.py"


def load_script():
    spec = importlib.util.spec_from_file_location(
        "soft_positive_human_skeleton_test_module", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = load_script()


def attrs(
    brand: str,
    width: str,
    material: str = "кожа",
    country: str = "Россия",
    warranty: str = "12 месяцев",
) -> str:
    return json.dumps(
        {
            "Тип": "ботинки",
            "Бренд": brand,
            "Ширина": width,
            "Материал": material,
            "Страна производства": country,
            "Гарантийный срок": warranty,
            "Сезон": "зима",
        },
        ensure_ascii=False,
    )


def rule(
    rule_id: str,
    pair_id: str,
    value_a: str,
    value_b: str,
    *,
    key: str = "Ширина",
) -> dict[str, object]:
    return {
        "generation_rule_id": rule_id,
        "source_rule_id": f"source-{rule_id}",
        "generation_tier": "TEST_TIER",
        "label": 1,
        "concept": {
            "Ширина": "width",
            "Бренд": "brand",
            "Страна производства": "country_of_origin",
            "Гарантийный срок": "warranty_period",
        }.get(key, "unknown"),
        "required_attribute_key": key,
        "allowed_categories": ["Обувь"],
        "allowed_product_types": ["ботинки"],
        "source_examples": [
            {
                "source_pair_id": pair_id,
                "target_value_a": value_a,
                "target_value_b": value_b,
                "source_is_singleton": True,
            }
        ],
    }


def write_validation_fixture(root: Path, *, overlap_fallback: bool = False) -> Path:
    validation = root / "frozen_validation"
    validation.mkdir(parents=True)
    validation_rows: list[dict[str, object]] = []
    iid_rows: list[dict[str, object]] = []
    if overlap_fallback:
        cards = [
            {
                "id": 9001,
                "name": "фиолетовый короб ботинки Alpha зимние",
                "attributes": attrs("Alpha", "7", "нубук"),
                "category": "Обувь",
            },
            {
                "id": 9002,
                "name": "контрольная карточка без совпадения",
                "attributes": attrs("ValidationOnly", "99", "текстиль"),
                "category": "Обувь",
            },
        ]
        for card in cards:
            validation_rows.append(
                {
                    "id": card["id"],
                    "name": card["name"],
                    "category": card["category"],
                    "product_text": builder.serialize_product(
                        pd.Series(card), max_attribute_chars=6000
                    ),
                }
            )
        iid_rows.append({"id1": 9001, "id2": 9002, "target": 1})
    pd.DataFrame(
        validation_rows,
        columns=["id", "name", "category", "product_text"],
    ).to_parquet(validation / "items.parquet", index=False)
    for split in builder.VALIDATION_SPLITS:
        rows = iid_rows if split == "iid" else []
        pd.DataFrame(rows, columns=["id1", "id2", "target"]).to_parquet(
            validation / f"{split}_validation_pairs.parquet", index=False
        )
    return validation


def write_fixture(
    root: Path, *, overlap_fallback: bool = False
) -> tuple[Path, Path, Path, Path]:
    source = root / "source"
    pilot = root / "pilot"
    source.mkdir()
    pilot.mkdir()
    input_rows = [
        {
            "pair_id": "p-width",
            "item_id_a": 101,
            "item_id_b": 102,
            "category": "Обувь",
            "title_a": "Nord ботинки мужские зимние",
            "attributes_a_json": attrs("Nord", "6", warranty="12 месяцев"),
            "title_b": "ботинки Nord зимние мужские",
            "attributes_b_json": attrs("Nord", "6F", warranty="24 месяца"),
        },
        {
            "pair_id": "p-brand-fallback",
            "item_id_a": 201,
            "item_id_b": 202,
            "category": "Обувь",
            "title_a": "фиолетовый короб ботинки Alpha зимние",
            "attributes_a_json": attrs("Alpha", "7", "нубук"),
            "title_b": "синий короб ботинки Beta зимние",
            "attributes_b_json": attrs("Beta", "7", "нубук"),
        },
        {
            "pair_id": "p-extra",
            "item_id_a": 301,
            "item_id_b": 302,
            "category": "Обувь",
            "title_a": "Gamma ботинки женские зимние",
            "attributes_a_json": attrs("Gamma", "8", "замша"),
            "title_b": "ботинки Gamma женские зимние",
            "attributes_b_json": attrs("Gamma", "8.5", "замша"),
        },
    ]
    inputs = pd.DataFrame(input_rows)
    labels = pd.DataFrame(
        {"pair_id": [row["pair_id"] for row in input_rows], "human_label": [1.0, 1.0, 1.0]}
    )
    inputs_path = pilot / "pilot_inputs.parquet"
    labels_path = pilot / "pilot_labels.parquet"
    inputs.to_parquet(inputs_path, index=False)
    labels.to_parquet(labels_path, index=False)
    (pilot / "manifest.json").write_text(
        json.dumps(
            {
                "source_train": str(root / "prepared/validation_splits_v1/human/train_pairs.parquet"),
                "source_pairs": 3,
                "pilot_pairs": 3,
                "sampling_mode": "all",
                "categories": 1,
                "labels_stored_separately": True,
                "label_counts": {"0": 0, "1": 3},
            }
        ),
        encoding="utf-8",
    )

    strong = rule(
        "rule-warranty",
        "p-width",
        "12 месяцев",
        "24 месяца",
        key="Гарантийный срок",
    )
    unsafe = rule(
        "rule-brand-fallback",
        "p-brand-fallback",
        "фиолетовый",
        "синий",
        key="Бренд",
    )
    metadata = pd.DataFrame(
        [
            {
                "composition_index": 0,
                "category": "Обувь",
                "product_type": "ботинки",
                "component": "tier_a",
                "rules_json": json.dumps([strong], ensure_ascii=False),
                "applications_json": "[]",
            },
            {
                "composition_index": 1,
                "category": "Обувь",
                "product_type": "ботинки",
                "component": "tier_b",
                "rules_json": json.dumps([unsafe], ensure_ascii=False),
                "applications_json": "[]",
            },
        ]
    )
    metadata.to_parquet(source / "pair_generation_metadata.parquet", index=False)
    pd.DataFrame(
        [{"id1": -1, "id2": -2, "target": 1}, {"id1": -3, "id2": -4, "target": 1}]
    ).to_parquet(source / "pairs.parquet", index=False)
    pd.DataFrame(
        [
            {"id": -1, "name": "old a", "attributes": '{"x":"1"}', "category": "Обувь"},
            {"id": -2, "name": "old b", "attributes": '{"x":"2"}', "category": "Обувь"},
            {"id": -3, "name": "old c", "attributes": '{"x":"3"}', "category": "Обувь"},
            {"id": -4, "name": "old d", "attributes": '{"x":"4"}', "category": "Обувь"},
        ]
    ).to_parquet(source / "items.parquet", index=False)
    (source / "summary.json").write_text('{"generated_pairs":2}\n', encoding="utf-8")
    validation_dir = write_validation_fixture(
        root, overlap_fallback=overlap_fallback
    )
    return source, inputs_path, labels_path, validation_dir


class SoftPositiveHumanSkeletonTest(unittest.TestCase):
    def test_semantic_key_gate_rejects_cross_facet_extraction_errors(self) -> None:
        incompatible = (
            ("Тип обложки", "Обложка", "Тип", "cover_type"),
            ("Модель", "Модель", "Тип", "model"),
            ("Пол", "Пол", "Назначение", "gender"),
            (
                "Минимальное число игроков",
                "Максимальное число игроков",
                "Минимальное число игроков",
                "min_players",
            ),
            (
                "Минимальное число игроков",
                "Макс. число игроков",
                "Минимальное число игроков",
                "min_players",
            ),
            ("Ширина упаковки", "Ширина упаковки", "Вес товара", "package_width"),
            ("Цвет", "Материал", "Материал", "color"),
            (
                "Материал обивки",
                "Материал корпуса",
                "Тип материала сиденья без обивки",
                "material_seat",
            ),
            (
                "Материал столешницы",
                "Форма столешницы",
                "Форма столешницы",
                "tabletop_material_detail",
            ),
            (
                "Количество элементов питания",
                "Количество элементов питания",
                "Тип элементов питания",
                "battery_count",
            ),
            (
                "Объем травосборника, л",
                "Травосборник",
                "Объем травосборника, л",
                "grass_catcher_capacity",
            ),
            (
                "Количество мелодий и звуков",
                "Количество мелодий и звуков",
                "Мелодии",
                "melody_count",
            ),
            ("Комплектация", "Комплектация", "Комплектация", "packaging_material"),
            (
                "Материал покрытия",
                "Покрытие поверхности коврика",
                "Характеристика покрытия",
                "surface_treatment",
            ),
            ("Размер животного", "Возраст животного", "Возраст животного", "animal_size"),
            (
                "Материал подошвы",
                "Материал платформы",
                "Материал каблука",
                "sole_material",
            ),
            ("Высота платформы", "Высота каблука", "Высота каблука", "platform_height"),
            ("Тип каблука", "Тип подошвы", "Тип подошвы", "heel_type"),
            ("Бренд", "Бренд", "Бренд товара", "brand"),
            ("Бренд", "Бренд", "Бренд", "brand"),
            ("Цвет", "Цвет", "Цвет товара", "color"),
            ("Цвет", "Название цвета", "Цвет товара", "color"),
            ("Вид товара", "Тип", "Тип товара", "product_type"),
        )
        incompatible += tuple(
            ("Материал стельки", "Стелька", "Материал стельки", concept)
            for concept in ("insole_material", "insole_type", "material_insole")
        )
        for required, key_a, key_b, concept in incompatible:
            with self.subTest(required=required, key_a=key_a, key_b=key_b):
                self.assertFalse(
                    builder.semantic_key_compatible_for_overlay(
                        required, key_a, key_b, concept
                    )
                )

    def test_semantic_key_gate_accepts_only_safe_allowlisted_families(self) -> None:
        compatible = (
            ("Гарантия", "Гарантия", "Гарантия", "warranty"),
            (
                "Гарантийный срок",
                "Гарантийный срок",
                "Срок гарантии",
                "warranty_period",
            ),
        )
        for required, key_a, key_b, concept in compatible:
            with self.subTest(required=required, key_a=key_a, key_b=key_b):
                self.assertTrue(
                    builder.semantic_key_compatible_for_overlay(
                        required, key_a, key_b, concept
                    )
                )

    def test_noisy_metadata_families_are_never_overlayable(self) -> None:
        for required, key_a, key_b, concept in (
            (
                "Страна производства",
                "Страна производства",
                "Страна-изготовитель",
                "country_of_origin",
            ),
            (
                "Условия хранения",
                "Условия хранения",
                "Условия хранения",
                "storage_conditions",
            ),
            ("Состав", "Состав", "Состав продукта", "ingredients"),
            (
                "Сертификат 80 PLUS",
                "Сертификат 80 PLUS",
                "Сертификация 80 PLUS",
                "certification_80_plus",
            ),
        ):
            with self.subTest(required=required, concept=concept):
                self.assertFalse(
                    builder.semantic_key_compatible_for_overlay(
                        required, key_a, key_b, concept
                    )
                )

    def test_brand_family_is_never_overlayable(self) -> None:
        for key_a, key_b in (
            ("Бренд", "Бренд"),
            ("Бренд товара", "Торговая марка"),
            ("Brand", "Brand"),
        ):
            with self.subTest(key_a=key_a, key_b=key_b):
                self.assertFalse(
                    builder.semantic_key_compatible_for_overlay(
                        "Бренд", key_a, key_b, "brand"
                    )
                )

    def test_color_family_is_never_overlayable(self) -> None:
        for key_a, key_b in (
            ("Цвет", "Цвет"),
            ("Название цвета", "Цвет товара"),
            ("Color", "Colour"),
        ):
            with self.subTest(key_a=key_a, key_b=key_b):
                self.assertFalse(
                    builder.semantic_key_compatible_for_overlay(
                        "Цвет", key_a, key_b, "color"
                    )
                )

    def test_generic_and_qualified_product_types_are_not_allowlisted(self) -> None:
        for required, key_a, key_b, concept in (
            ("Тип", "Тип", "Вид товара", "type"),
            ("Вид товара", "Тип товара", "Тип", "product_type"),
        ):
            with self.subTest(required=required):
                self.assertFalse(
                    builder.semantic_key_compatible_for_overlay(
                        required, key_a, key_b, concept
                    )
                )
        self.assertFalse(
            builder.semantic_key_compatible_for_overlay(
                "Тип застежки", "Тип застежки", "Вид застежки", "closure_type"
            )
        )

    def test_non_allowlisted_families_and_unknown_tail_are_vetoed(self) -> None:
        denied = (
            ("Материал", "material"),
            ("Пол", "gender"),
            ("Аромат", "scent"),
            ("Вкус", "flavor"),
            ("Форма", "shape"),
            ("Назначение", "purpose"),
            ("Возраст", "age"),
            ("Размер", "size"),
            ("Количество", "count"),
            ("Минимальное число игроков", "min_players"),
            ("Максимальное число игроков", "max_players"),
            ("Ширина", "width"),
            ("Вес", "weight"),
            ("Объем", "volume"),
            ("Мощность", "power"),
            ("Скорость", "speed"),
            ("Управление", "control"),
            ("Съемная стелька", "removable_insole"),
            ("Редкий хвостовой параметр", "unknown_tail"),
        )
        for key, concept in denied:
            with self.subTest(key=key):
                self.assertFalse(
                    builder.semantic_key_compatible_for_overlay(
                        key, key, key, concept
                    )
                )

    def test_target_aliases_are_candidate_specific(self) -> None:
        package_bank = {
            "required_key": "Ширина упаковки",
            "concept": "package_width",
            "observed_aliases": [
                "ширина упаковки", "вес товара", "высота упаковки"
            ],
        }
        package_evidence = {
            "target_key_a": "Ширина упаковки",
            "target_key_b": "Ширина упаковки",
        }
        self.assertEqual(
            builder.selected_target_aliases(package_bank, package_evidence),
            {"ширина упаковки"},
        )

        cover_bank = {
            "required_key": "Тип обложки",
            "concept": "cover_type",
            "observed_aliases": ["обложка", "тип", "материал", "тип обложки"],
        }
        cover_evidence = {"target_key_a": "Обложка", "target_key_b": "Обложка"}
        cover_aliases = builder.selected_target_aliases(cover_bank, cover_evidence)
        self.assertEqual(cover_aliases, {"обложка"})

        color_bank = {
            "required_key": "Цвет",
            "concept": "color",
            "observed_aliases": ["цвет", "цвет товара", "название цвета", "материал"],
        }
        color_evidence = {"target_key_a": "Цвет", "target_key_b": "Цвет товара"}
        color_aliases = builder.selected_target_aliases(color_bank, color_evidence)
        self.assertEqual(
            color_aliases,
            {"цвет", "цвет товара"},
        )
        retained, removed_target, removed_ids = builder.sanitize_skeleton_attributes(
            {
                "Цвет": "красный",
                "Цвет товара": "бордовый",
                "Название цвета": "вишневый",
                "Материал": "кожа",
            },
            color_aliases,
            "Цвет",
        )
        self.assertEqual(
            retained,
            {"Название цвета": "вишневый", "Материал": "кожа"},
        )
        self.assertEqual(
            set(removed_target), {"Цвет", "Цвет товара"}
        )
        self.assertEqual(removed_ids, [])

    def test_unselected_target_alias_conflict_skips_skeleton(self) -> None:
        bank = {
            "required_key": "Гарантийный срок",
            "concept": "warranty_period",
            "observed_aliases": [
                "гарантийный срок",
                "срок гарантии",
                "гарантия",
            ],
            "candidates": [],
        }
        evidence = {
            "target_key_a": "Гарантийный срок",
            "target_key_b": "Гарантийный срок",
            "target_value_a": "12 месяцев",
            "target_value_b": "24 месяца",
        }
        selected = builder.selected_target_aliases(bank, evidence)
        self.assertEqual(selected, {"гарантийный срок"})
        contradictory = builder.HumanPair(
            pair_id="bad-color-skeleton",
            item_id_a=1,
            item_id_b=2,
            category="Обувь",
            title_a="ботинки Nord",
            attributes_a={"Тип": "ботинки", "Срок гарантии": "6 месяцев"},
            title_b="ботинки Nord",
            attributes_b={"Тип": "ботинки", "Срок гарантии": "36 месяцев"},
        )
        self.assertTrue(
            builder._target_title_conflict(contradictory, bank, evidence, selected)
        )
        compatible = builder.HumanPair(
            pair_id="good-color-skeleton",
            item_id_a=3,
            item_id_b=4,
            category="Обувь",
            title_a="ботинки Nord",
            attributes_a={"Тип": "ботинки", "Сезон": "зима"},
            title_b="ботинки Nord",
            attributes_b={"Тип": "ботинки", "Сезон": "зима"},
        )
        self.assertFalse(builder._target_title_conflict(compatible, bank, evidence, selected))

        evidence_pair = builder.HumanPair(
            pair_id="color-evidence",
            item_id_a=5,
            item_id_b=6,
            category="Обувь",
            title_a="синие ботинки Nord",
            attributes_a={"Тип": "ботинки", "Гарантийный срок": "12 месяцев"},
            title_b="голубые ботинки Nord",
            attributes_b={"Тип": "ботинки", "Гарантийный срок": "24 месяца"},
        )
        candidate = {
            **evidence,
            "source_pair_id": "color-evidence",
            "source_example_index": 0,
            "source_is_singleton": True,
            "grounding_mode": "both_raw_attributes_unique",
            "grounding_rank": 4,
            "schema_safe_for_overlay": True,
            "semantic_keys_compatible_for_overlay": True,
        }
        bank["rule_id"] = "color-rule"
        bank["candidates"] = [candidate]
        task = {
            "composition_index": 0,
            "category": "Обувь",
            "product_type": "ботинки",
            "scope": ("Обувь", "ботинки"),
            "rule": {"generation_rule_id": "color-rule"},
        }
        plan = builder.build_task_plan(
            [task],
            {("Обувь", "ботинки"): [("bad-color-skeleton", "explicit_type")]},
            {
                "bad-color-skeleton": contradictory,
                "color-evidence": evidence_pair,
            },
            {"color-rule": bank},
            seed=9,
        )
        self.assertEqual(plan[0]["construction_mode"], "source_pair_surface")
        self.assertEqual(plan[0]["skeleton_pair_id"], "color-evidence")

    def test_warranty_overlay_rejects_retained_service_life(self) -> None:
        bank = {
            "required_key": "Гарантийный срок",
            "concept": "warranty_period",
        }
        evidence = {
            "target_key_a": "Гарантийный срок",
            "target_key_b": "Срок гарантии",
            "target_value_a": "10 лет",
            "target_value_b": "5 лет",
        }
        aliases = builder.selected_target_aliases(bank, evidence)
        contradictory = builder.HumanPair(
            pair_id="warranty-service-life",
            item_id_a=11,
            item_id_b=12,
            category="Дом и сад",
            title_a="сковорода Nord",
            attributes_a={
                "Гарантийный срок": "1 год",
                "Срок службы": "1 год",
            },
            title_b="сковорода Nord",
            attributes_b={
                "Срок гарантии": "6 месяцев",
                "Ресурс изделия": "2 года",
            },
        )
        self.assertTrue(
            builder._target_title_conflict(
                contradictory, bank, evidence, aliases
            )
        )

    def test_overlay_requires_selected_target_key_on_both_sides(self) -> None:
        brand_aliases = {"бренд"}
        brand_title_only = builder.HumanPair(
            pair_id="brand-title-only",
            item_id_a=1,
            item_id_b=2,
            category="Красота и гигиена",
            title_a="Insight шампунь для сухих волос",
            attributes_a={"Тип": "шампунь", "Бренд": "Insight"},
            title_b="Syoss шампунь для сухих волос",
            attributes_b={"Тип": "шампунь"},
        )
        self.assertFalse(
            builder._selected_target_key_present_on_both_sides(
                brand_title_only, brand_aliases
            )
        )

        width_missing_one_side = builder.HumanPair(
            pair_id="width-missing",
            item_id_a=3,
            item_id_b=4,
            category="Обувь",
            title_a="ботинки Nord",
            attributes_a={"Тип": "ботинки", "Ширина": "6"},
            title_b="ботинки Nord",
            attributes_b={"Тип": "ботинки", "Материал": "кожа"},
        )
        self.assertFalse(
            builder._selected_target_key_present_on_both_sides(
                width_missing_one_side, {"ширина"}
            )
        )

        width_on_both_sides = builder.HumanPair(
            pair_id="width-both",
            item_id_a=5,
            item_id_b=6,
            category="Обувь",
            title_a="ботинки Nord",
            attributes_a={"Тип": "ботинки", "Ширина": "6"},
            title_b="ботинки Nord",
            attributes_b={"Тип": "ботинки", "Ширина": "6F"},
        )
        self.assertTrue(
            builder._selected_target_key_present_on_both_sides(
                width_on_both_sides, {"ширина"}
            )
        )

    def test_exact_source_grounding_and_unsafe_fallback(self) -> None:
        pair = builder.HumanPair(
            pair_id="p",
            item_id_a=1,
            item_id_b=2,
            category="Обувь",
            title_a="Nord ботинки",
            attributes_a={"Тип": "ботинки", "Ширина": "6"},
            title_b="ботинки Nord",
            attributes_b={"Тип": "ботинки", "Ширина": "6F"},
        )
        evidence = builder.build_rule_evidence(
            rule("r", "p", "6", "6F"), {"p": pair}
        )
        candidate = evidence["candidates"][0]
        self.assertEqual(candidate["grounding_mode"], "both_raw_attributes_unique")
        self.assertEqual(candidate["target_key_a"], "Ширина")
        self.assertEqual(candidate["target_value_b"], "6F")

        unsafe_pair = builder.HumanPair(
            pair_id="u",
            item_id_a=3,
            item_id_b=4,
            category="Обувь",
            title_a="фиолетовый короб ботинки",
            attributes_a={"Тип": "ботинки", "Бренд": "Alpha"},
            title_b="синий короб ботинки",
            attributes_b={"Тип": "ботинки", "Бренд": "Beta"},
        )
        unsafe = builder.build_rule_evidence(
            rule("u-rule", "u", "фиолетовый", "синий", key="Бренд"),
            {"u": unsafe_pair},
        )["candidates"][0]
        self.assertFalse(unsafe["schema_safe_for_overlay"])

        cross_key_pair = builder.HumanPair(
            pair_id="cross",
            item_id_a=5,
            item_id_b=6,
            category="Обувь",
            title_a="Nord ботинки настольная игра",
            attributes_a={"Тип": "ботинки", "Максимальное число игроков": "2"},
            title_b="ботинки Nord настольная игра",
            attributes_b={"Тип": "ботинки", "Минимальное число игроков": "3"},
        )
        cross_rule = rule(
            "cross-rule",
            "cross",
            "2",
            "3",
            key="Минимальное число игроков",
        )
        cross_rule["concept"] = "min_players"
        cross_bank = builder.build_rule_evidence(cross_rule, {"cross": cross_key_pair})
        self.assertEqual(
            cross_bank["candidates"][0]["grounding_mode"],
            "both_raw_attributes_unique",
        )
        self.assertFalse(
            cross_bank["candidates"][0]["semantic_keys_compatible_for_overlay"]
        )
        task = {
            "composition_index": 0,
            "category": "Обувь",
            "product_type": "ботинки",
            "scope": ("Обувь", "ботинки"),
            "rule": cross_rule,
        }
        planned = builder.build_task_plan(
            [task],
            {("Обувь", "ботинки"): [("cross", "explicit_type")]},
            {"cross": cross_key_pair},
            {"cross-rule": cross_bank},
            seed=5,
        )[0]
        self.assertEqual(planned["construction_mode"], "source_pair_surface")

    def test_non_target_preservation_determinism_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory(prefix="human-skeleton-") as raw:
            root = Path(raw)
            source, inputs, labels, validation_dir = write_fixture(root)
            first = builder.build(
                source_dir=source,
                pilot_inputs_path=inputs,
                pilot_labels_path=labels,
                validation_dir=validation_dir,
                output_dir=root / "out-one",
                seed=17,
            )
            second = builder.build(
                source_dir=source,
                pilot_inputs_path=inputs,
                pilot_labels_path=labels,
                validation_dir=validation_dir,
                output_dir=root / "out-two",
                seed=17,
            )
            self.assertTrue(first["validation"]["valid"])
            self.assertEqual(first["summary"]["run_signature"], second["summary"]["run_signature"])
            pd.testing.assert_frame_equal(
                pd.read_parquet(root / "out-one/items.parquet"),
                pd.read_parquet(root / "out-two/items.parquet"),
            )
            metadata = pd.read_parquet(root / "out-one/pair_generation_metadata.parquet")
            self.assertEqual(set(metadata["construction_mode"]), {"overlay", "source_pair_surface"})
            self.assertTrue(metadata["observed_label1_transition"].all())
            self.assertTrue(metadata["non_target_values_preserved"].all())
            self.assertTrue(metadata["run_signature"].eq(first["summary"]["run_signature"]).all())
            self.assertTrue(
                metadata["overlay_allowlist_version"].eq(
                    builder.OVERLAY_ALLOWLIST_VERSION
                ).all()
            )
            self.assertEqual(
                set(metadata.loc[metadata["construction_mode"].eq("overlay"), "overlay_allowlist_family"]),
                {"warranty"},
            )

            items = pd.read_parquet(root / "out-one/items.parquet")
            pairs = pd.read_parquet(root / "out-one/pairs.parquet")
            item_map = items.set_index("id")
            overlay = metadata.loc[metadata["construction_mode"].eq("overlay")].iloc[0]
            left = builder.parse_attributes(item_map.loc[int(overlay.id1), "attributes"])
            right = builder.parse_attributes(item_map.loc[int(overlay.id2), "attributes"])
            source_inputs = pd.read_parquet(inputs).set_index("pair_id")
            overlay_source = source_inputs.loc[str(overlay.skeleton_pair_id)]
            self.assertEqual(
                left["Материал"],
                builder.parse_attributes(overlay_source.attributes_a_json)["Материал"],
            )
            self.assertEqual(
                right["Материал"],
                builder.parse_attributes(overlay_source.attributes_b_json)["Материал"],
            )
            self.assertEqual(left[str(overlay.target_key_a)], str(overlay.target_value_a))
            self.assertEqual(right[str(overlay.target_key_b)], str(overlay.target_value_b))

            fallback = metadata.loc[metadata["construction_mode"].eq("source_pair_surface")].iloc[0]
            source_row = source_inputs.loc[str(fallback.skeleton_pair_id)]
            self.assertEqual(
                builder.parse_attributes(item_map.loc[int(fallback.id1), "attributes"]),
                builder.parse_attributes(source_row.attributes_a_json),
            )
            self.assertEqual(
                builder.parse_attributes(item_map.loc[int(fallback.id2), "attributes"]),
                builder.parse_attributes(source_row.attributes_b_json),
            )
            self.assertEqual(set(pairs["target"]), {1})

            provenance = first["summary"]["source_provenance"]
            self.assertIn("builder_script", provenance)
            self.assertIn("pilot_manifest", provenance)
            self.assertIn("validation_items", provenance)
            self.assertIn("validation_iid_pairs", provenance)
            self.assertIn("validation_hard_pairs", provenance)
            self.assertIn("validation_ood_pairs", provenance)
            self.assertEqual(provenance["builder_script"]["sha256"], builder.sha256_file(SCRIPT_PATH))
            self.assertTrue((root / "out-one/build_manifest.json").is_file())
            allowlist = first["summary"]["overlay_allowlist"]
            self.assertEqual(allowlist["version"], builder.OVERLAY_ALLOWLIST_VERSION)
            self.assertEqual(
                set(allowlist["families"]),
                set(builder.SAFE_OVERLAY_SIGNATURES_BY_FAMILY),
            )
            self.assertEqual(sum(allowlist["task_counts_by_family"].values()), 2)
            self.assertEqual(first["manifest"]["overlay_allowlist"], allowlist)
            diagnostics = first["summary"]["fact_clone_diagnostics"]
            self.assertTrue(diagnostics["nonblocking"])
            self.assertEqual(diagnostics["all"]["pairs"]["total"], 2)
            self.assertEqual(
                diagnostics["by_construction_mode"]["source_pair_surface"]["pairs"][
                    "fact_identical_to_human_positive"
                ],
                1,
            )

    def test_assignment_is_deterministic_and_balanced(self) -> None:
        tasks = [
            {"composition_index": index, "scope": ("Обувь", "ботинки")}
            for index in range(7)
        ]
        pools = {("Обувь", "ботинки"): [("p1", "explicit_type"), ("p2", "title_exact_phrase")]}
        first = builder.assign_skeletons(tasks, pools, seed=41)
        second = builder.assign_skeletons(tasks, pools, seed=41)
        self.assertEqual(first, second)
        counts = Counter(row["pair_id"] for row in first.values())
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_frozen_validation_fact_overlap_is_dropped_and_cli_replays(self) -> None:
        with tempfile.TemporaryDirectory(prefix="human-skeleton-leakage-") as raw:
            root = Path(raw)
            source, inputs, labels, validation_dir = write_fixture(
                root, overlap_fallback=True
            )
            output = root / "output"
            command = [
                sys.executable,
                str(SCRIPT_PATH),
                "--source-dir",
                str(source),
                "--pilot-inputs",
                str(inputs),
                "--pilot-labels",
                str(labels),
                "--validation-dir",
                str(validation_dir),
                "--output-dir",
                str(output),
                "--seed",
                "17",
            ]
            completed = subprocess.run(
                command, check=True, capture_output=True, text=True
            )
            result = json.loads(completed.stdout)
            report = result["summary"]["validation_overlap_filter"]
            self.assertEqual(result["summary"]["source_task_count"], 2)
            self.assertEqual(result["summary"]["generated_pairs"], 1)
            self.assertEqual(result["summary"]["config"]["count"], 1)
            self.assertEqual(report["source_task_count"], 2)
            self.assertEqual(report["emitted_pair_count"], 1)
            self.assertEqual(report["dropped_pair_count"], 1)
            self.assertEqual(report["dropped_task_ids"], [1])
            self.assertEqual(report["dropped_pairs_by_split"], {"iid": 1, "hard": 0, "ood": 0})
            self.assertEqual(report["dropped_endpoints_by_side"], {"a": 1, "b": 0})
            self.assertEqual(
                report["dropped_pairs_by_construction_mode"],
                {"source_pair_surface": 1},
            )
            self.assertEqual(report["postfilter_overlapping_card_count"], 0)
            self.assertEqual(len(report["dropped_pairs"]), 1)
            endpoint = report["dropped_pairs"][0]["overlap_endpoints"][0]
            self.assertEqual(endpoint["side"], "a")
            self.assertTrue(endpoint["fact_tokens"])
            self.assertEqual(
                endpoint["validation_matches_by_split"], {"iid": [9001]}
            )

            pairs = pd.read_parquet(output / "pairs.parquet")
            items = pd.read_parquet(output / "items.parquet")
            metadata = pd.read_parquet(output / "pair_generation_metadata.parquet")
            self.assertEqual(len(pairs), 1)
            self.assertEqual(len(items), 2)
            self.assertEqual(metadata["composition_index"].tolist(), [0])
            self.assertEqual(set(pairs["target"]), {1})
            for filename in (
                "summary.json",
                "validation_report.json",
                "distribution_report.json",
                "build_manifest.json",
            ):
                document = json.loads((output / filename).read_text(encoding="utf-8"))
                self.assertEqual(document["validation_overlap_filter"], report)

            validated = subprocess.run(
                [*command, "--validate-only"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(json.loads(validated.stdout)["validation"]["valid"])

            wrong_validation = write_validation_fixture(root / "other")
            failed = subprocess.run(
                [
                    *command[: command.index("--validation-dir") + 1],
                    str(wrong_validation),
                    "--output-dir",
                    str(output),
                    "--seed",
                    "17",
                    "--validate-only",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("source files or builder script changed", failed.stderr)

    def test_validator_rejects_duplicate_card(self) -> None:
        with tempfile.TemporaryDirectory(prefix="human-skeleton-duplicate-") as raw:
            root = Path(raw)
            source, inputs_path, labels_path, validation_dir = write_fixture(root)
            result = builder.build(
                source_dir=source,
                pilot_inputs_path=inputs_path,
                pilot_labels_path=labels_path,
                validation_dir=validation_dir,
                output_dir=root / "output",
                seed=23,
            )
            items = pd.read_parquet(root / "output/items.parquet")
            pairs = pd.read_parquet(root / "output/pairs.parquet")
            metadata = pd.read_parquet(root / "output/pair_generation_metadata.parquet")
            items.loc[1, ["name", "attributes"]] = items.loc[0, ["name", "attributes"]].values
            source_metadata = pd.read_parquet(source / "pair_generation_metadata.parquet")
            tasks = builder.parse_tasks(source_metadata)
            inputs = pd.read_parquet(inputs_path)
            labels = pd.read_parquet(labels_path)
            human_pairs, _ = builder.load_human_pairs(inputs, labels)
            evidence = builder.build_all_evidence(tasks, human_pairs)
            report = builder.validate_dataset(
                tasks=tasks,
                human_pairs=human_pairs,
                evidence_by_rule=evidence,
                items=items,
                pairs=pairs,
                metadata=metadata,
                expected_count=2,
                source_task_count=2,
                validation_facts=builder.load_frozen_validation_facts(
                    validation_dir
                ),
                validation_overlap=result["summary"]["validation_overlap_filter"],
            )
            self.assertFalse(report["valid"])
            self.assertIn("duplicate_global_card", report["errors"])
            self.assertTrue(result["validation"]["valid"])

    def test_rejects_ungrounded_or_non_train_evidence(self) -> None:
        bad = rule("bad", "missing-pair", "6", "7")
        with self.assertRaises(builder.SkeletonBuildError):
            builder.build_rule_evidence(bad, {})

        pair = builder.HumanPair(
            pair_id="wrong-type",
            item_id_a=1,
            item_id_b=2,
            category="Обувь",
            title_a="Nord кроссовки",
            attributes_a={"Тип": "кроссовки", "Ширина": "6"},
            title_b="кроссовки Nord",
            attributes_b={"Тип": "кроссовки", "Ширина": "7"},
        )
        with self.assertRaises(builder.SkeletonBuildError):
            builder.build_rule_evidence(rule("bad-scope", "wrong-type", "6", "7"), {"wrong-type": pair})

    def test_manifest_rejects_validation_source(self) -> None:
        with tempfile.TemporaryDirectory(prefix="human-skeleton-manifest-") as raw:
            root = Path(raw)
            inputs = pd.DataFrame({"pair_id": ["p"], "category": ["Обувь"]})
            labels = pd.DataFrame({"pair_id": ["p"], "human_label": [1.0]})
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "source_train": "/tmp/prepared/validation_splits_v1/human/val_pairs.parquet",
                        "source_pairs": 1,
                        "pilot_pairs": 1,
                        "sampling_mode": "all",
                        "categories": 1,
                        "labels_stored_separately": True,
                        "label_counts": {"1": 1},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(builder.SkeletonBuildError):
                builder.validate_pilot_manifest(manifest, inputs, labels)


if __name__ == "__main__":
    unittest.main()
