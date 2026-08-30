from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/build_surface_positive_pairs.py"


def load_script():
    spec = importlib.util.spec_from_file_location(
        "surface_positive_pairs_test_module", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = load_script()


def attributes(brand: str, product_type: str, model: str, color: str) -> str:
    return json.dumps(
        {
            "Бренд": brand,
            "Тип": product_type,
            "Модель": model,
            "Цвет": color,
            "Материал": "сталь",
            "Мощность": "1200 Вт",
            "Страна производства": "Россия",
        },
        ensure_ascii=False,
    )


def build_fixture(root: Path) -> tuple[Path, Path]:
    item_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    categories = (
        "Автотовары",
        "Красота и гигиена",
        "Детские товары",
    )
    for pair_index, category in enumerate(categories):
        left_id = 100 + pair_index * 2
        right_id = left_id + 1
        brand = f"Brand{pair_index}"
        product_type = f"товар{pair_index}"
        model = f"M{pair_index}"
        for source_id, suffix in ((left_id, "новый"), (right_id, "комплект")):
            item_rows.append(
                {
                    "id": source_id,
                    "name": f"{brand} {product_type} {model} {suffix}",
                    "attributes": attributes(
                        brand, product_type, model, ("красный", "белый")[source_id % 2]
                    ),
                    "category": category,
                }
            )
        pair_rows.append({"id1": left_id, "id2": right_id, "target": 1.0})

    # A target=0 card is not an eligible donor even though its shape is valid.
    item_rows.extend(
        [
            {
                "id": 200,
                "name": "ZeroBrand чайник Z1 новый",
                "attributes": attributes("ZeroBrand", "чайник", "Z1", "белый"),
                "category": "Электроника",
            },
            {
                "id": 201,
                "name": "ZeroBrand чайник Z2 новый",
                "attributes": attributes("ZeroBrand", "чайник", "Z2", "черный"),
                "category": "Электроника",
            },
        ]
    )
    pair_rows.append({"id1": 200, "id2": 201, "target": 0.0})

    # Frozen OOD categories cannot leak into this training augmentation.
    item_rows.extend(
        [
            {
                "id": 300,
                "name": "DressBrand платье D1 новое",
                "attributes": attributes("DressBrand", "платье", "D1", "красный"),
                "category": "Одежда",
            },
            {
                "id": 301,
                "name": "DressBrand платье D1 женское",
                "attributes": attributes("DressBrand", "платье", "D1", "красный"),
                "category": "Одежда",
            },
        ]
    )
    pair_rows.append({"id1": 300, "id2": 301, "target": 1.0})

    # Both endpoints contain the explicit schema/value bug the builder rejects.
    for source_id in (400, 401):
        broken = json.loads(attributes("фиолетовый", "набор", "B1", "белый"))
        item_rows.append(
            {
                "id": source_id,
                "name": "фиолетовый набор B1 детский",
                "attributes": json.dumps(broken, ensure_ascii=False),
                "category": "Детские товары",
            }
        )
    pair_rows.append({"id1": 400, "id2": 401, "target": 1.0})

    items_path, pairs_path = root / "items.parquet", root / "train_pairs.parquet"
    pd.DataFrame(item_rows).to_parquet(items_path, index=False)
    pd.DataFrame(pair_rows).to_parquet(pairs_path, index=False)
    return items_path, pairs_path


class SurfacePositivePairsTest(unittest.TestCase):
    def test_identity_fields_are_never_optional(self) -> None:
        for key in (
            "Бренд",
            "Бренд товара",
            "Модель",
            "Модель производителя",
            "Артикул",
            "Партномер (артикул производителя)",
            "Тип товара",
        ):
            with self.subTest(key=key):
                self.assertTrue(builder.identity_key(key))

    def test_title_transform_keeps_decimal_and_hyphenated_tokens_intact(self) -> None:
        decimal = 'western digital / 1тб 3,5" внутренний жёсткий диск'
        transformed, _ = builder.transform_name(
            decimal, {"Бренд": "western digital"}, 17, 1
        )
        self.assertIn("3,5", transformed)
        self.assertEqual(
            builder.token_multiset(decimal), builder.token_multiset(transformed)
        )

        hyphenated = 'дрель "фикси-дрель" 3 насадки'
        transformed, _ = builder.transform_name(hyphenated, {}, 17, 2)
        self.assertIn("фикси-дрель", transformed)
        self.assertNotIn("фикси, дрель", transformed)
        self.assertEqual(
            builder.token_multiset(hyphenated), builder.token_multiset(transformed)
        )

    def test_build_is_deterministic_fact_preserving_and_leakage_safe(self) -> None:
        with tempfile.TemporaryDirectory(prefix="surface-positive-") as raw:
            root = Path(raw)
            items_path, pairs_path = build_fixture(root)
            first = builder.build(
                items_path=items_path,
                eligible_pairs_path=pairs_path,
                output_dir=root / "first",
                count=3,
                seed=17,
                example_count=2,
            )
            second = builder.build(
                items_path=items_path,
                eligible_pairs_path=pairs_path,
                output_dir=root / "second",
                count=3,
                seed=17,
                example_count=2,
            )

            self.assertTrue(first["validation"]["valid"])
            self.assertTrue(first["validation"]["no_new_or_changed_visible_facts"])
            self.assertEqual(first["summary"]["target_counts"], {"0": 0, "1": 3})
            self.assertEqual(
                first["summary"]["run_signature"],
                second["summary"]["run_signature"],
            )
            pd.testing.assert_frame_equal(
                pd.read_parquet(root / "first/items.parquet"),
                pd.read_parquet(root / "second/items.parquet"),
            )
            pd.testing.assert_frame_equal(
                pd.read_parquet(root / "first/pair_provenance.parquet"),
                pd.read_parquet(root / "second/pair_provenance.parquet"),
            )

            output_pairs = pd.read_parquet(root / "first/pairs.parquet")
            output_items = pd.read_parquet(root / "first/items.parquet")
            provenance = pd.read_parquet(root / "first/pair_provenance.parquet")
            self.assertEqual(set(output_pairs["target"]), {1})
            self.assertEqual(len(output_items), 6)
            self.assertEqual(provenance["source_item_id"].nunique(), 3)
            self.assertFalse(set(provenance["source_item_id"]) & {200, 201, 300, 301, 400, 401})
            self.assertFalse(
                set(output_items["category"]) & {"Одежда", "Бытовая техника"}
            )
            self.assertTrue(provenance["no_atomic_change"].all())

            candidate_report = first["summary"]["candidate_pool"]
            self.assertGreaterEqual(
                candidate_report["rejection_counts"].get("forbidden_ood_category", 0),
                1,
            )
            self.assertGreaterEqual(
                candidate_report["rejection_counts"].get("brand_is_color", 0), 1
            )
            distribution = first["distribution"]
            self.assertEqual(distribution["generated"]["pairs"], 3)
            self.assertEqual(
                distribution["interpretation"]["atomic_attribute_changes"], 0
            )
            self.assertTrue((root / "first/build_manifest.json").is_file())
            self.assertEqual(len((root / "first/examples.jsonl").read_text().splitlines()), 2)

    def test_validator_rejects_changed_attribute_fact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="surface-positive-invalid-") as raw:
            root = Path(raw)
            items_path, pairs_path = build_fixture(root)
            builder.build(
                items_path=items_path,
                eligible_pairs_path=pairs_path,
                output_dir=root / "output",
                count=2,
                seed=31,
            )
            source_items = pd.read_parquet(items_path)
            eligible_pairs = pd.read_parquet(pairs_path)
            pairs = pd.read_parquet(root / "output/pairs.parquet")
            items = pd.read_parquet(root / "output/items.parquet")
            provenance = pd.read_parquet(root / "output/pair_provenance.parquet")

            first_pair = pairs.iloc[0]
            right_index = items.index[items["id"].eq(first_pair["id2"])][0]
            right_attributes = json.loads(items.at[right_index, "attributes"])
            changed_key = next(iter(right_attributes))
            right_attributes[changed_key] = "СОВЕРШЕННО ДРУГОЕ ЗНАЧЕНИЕ"
            items.at[right_index, "attributes"] = json.dumps(
                right_attributes, ensure_ascii=False
            )
            report = builder.validate_surface_dataset(
                source_items=source_items,
                eligible_pairs=eligible_pairs,
                pairs=pairs,
                items=items,
                provenance=provenance,
                expected_count=2,
            )
            self.assertFalse(report["valid"])
            self.assertTrue(
                any("attribute_value_fact_changed" in error for error in report["errors"])
            )
            self.assertTrue(
                any("retained_facts_differ" in error for error in report["errors"])
            )

    def test_validator_rejects_schema_value_swap_even_if_value_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory(prefix="surface-positive-schema-swap-") as raw:
            root = Path(raw)
            items_path, pairs_path = build_fixture(root)
            builder.build(
                items_path=items_path,
                eligible_pairs_path=pairs_path,
                output_dir=root / "output",
                count=1,
                seed=43,
            )
            source_items = pd.read_parquet(items_path)
            eligible_pairs = pd.read_parquet(pairs_path)
            pairs = pd.read_parquet(root / "output/pairs.parquet")
            items = pd.read_parquet(root / "output/items.parquet")
            provenance = pd.read_parquet(root / "output/pair_provenance.parquet")

            right_id = int(pairs.iloc[0]["id2"])
            right_index = items.index[items["id"].eq(right_id)][0]
            right_attributes = json.loads(items.at[right_index, "attributes"])
            mapping = json.loads(provenance.at[0, "right_key_to_source_key_json"])
            old_right_key = next(
                key
                for key, source_key in mapping.items()
                if source_key == "Бренд"
            )
            brand_value = right_attributes.pop(old_right_key)
            right_attributes["Цвет"] = brand_value
            mapping["Цвет"] = mapping.pop(old_right_key)
            items.at[right_index, "attributes"] = json.dumps(
                right_attributes, ensure_ascii=False
            )
            provenance.at[0, "right_key_to_source_key_json"] = json.dumps(
                mapping, ensure_ascii=False
            )

            report = builder.validate_surface_dataset(
                source_items=source_items,
                eligible_pairs=eligible_pairs,
                pairs=pairs,
                items=items,
                provenance=provenance,
                expected_count=1,
            )
            self.assertFalse(report["valid"])
            self.assertTrue(
                any(
                    "unsafe_attribute_key_alias" in error
                    for error in report["errors"]
                )
            )

    def test_requested_count_cannot_exceed_safe_donor_capacity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="surface-positive-capacity-") as raw:
            root = Path(raw)
            items_path, pairs_path = build_fixture(root)
            with self.assertRaisesRegex(builder.SurfacePositiveError, "eligible donors"):
                builder.build(
                    items_path=items_path,
                    eligible_pairs_path=pairs_path,
                    output_dir=root / "output",
                    count=4,
                    seed=11,
                )


if __name__ == "__main__":
    unittest.main()
