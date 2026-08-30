from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nbformat
import pandas as pd

from item_pipeline.pair_rules import MutationRule


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    path = scripts / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MixedGenerationRuleKaggleFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.uploader = load_script(
            "mixed_generation_uploader_test",
            "push_mixed_generation_rule_pairs_dataset.py",
        )
        cls.notebook = load_script(
            "mixed_generation_notebook_test",
            "create_mixed_generation_rule_10k_notebook.py",
        )
        cls.launcher = load_script(
            "mixed_generation_launcher_test",
            "launch_mixed_generation_rule_kaggle_experiment.py",
        )

    def test_generated_target_counts_allow_one_label_and_require_exact_total(self) -> None:
        self.assertEqual(
            self.uploader.expected_target_counts(9_954, 46, 10_000),
            {"0": 9_954, "1": 46},
        )
        self.assertEqual(
            self.uploader.expected_target_counts(0, 6_500, 6_500),
            {"0": 0, "1": 6_500},
        )
        self.assertEqual(
            self.notebook.expected_counts(0, 6_500, 6_500),
            {"0": 0, "1": 6_500},
        )
        for values in ((-1, 11, 10), (0, 0, 0), (7, 2, 10)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.uploader.expected_target_counts(*values)

    def test_final_run_defaults_pin_full_transition_capacity_quota(self) -> None:
        self.assertEqual(self.uploader.DEFAULT_PAIR_COUNT, 10_000)
        self.assertEqual(self.uploader.DEFAULT_TARGET0, 9_954)
        self.assertEqual(self.uploader.DEFAULT_TARGET1, 46)
        quota = self.uploader.target_quota_provenance(
            {"0": 9_954, "1": 46}, 10_000
        )
        self.assertEqual(
            quota["policy"], "transition_positive_v4_full_capacity_v1"
        )
        self.assertEqual(quota["label_one_fraction"], 0.0046)
        self.assertIn("23 manually approved", quota["rationale"])
        with mock.patch.object(
            sys,
            "argv",
            ["push_mixed_generation_rule_pairs_dataset.py"],
        ):
            uploader_args = self.uploader.parse_args()
        self.assertEqual(
            (
                uploader_args.expected_pairs,
                uploader_args.expected_target0,
                uploader_args.expected_target1,
            ),
            (10_000, 9_954, 46),
        )
        with mock.patch.object(
            sys,
            "argv",
            [
                "create_mixed_generation_rule_10k_notebook.py",
                "--upload-manifest-sha256",
                "d" * 64,
            ],
        ):
            notebook_args = self.notebook.parse_args()
        self.assertEqual(
            (
                notebook_args.pair_count,
                notebook_args.expected_target0,
                notebook_args.expected_target1,
            ),
            (10_000, 9_954, 46),
        )
        with mock.patch.object(
            sys,
            "argv",
            ["launch_mixed_generation_rule_kaggle_experiment.py"],
        ):
            launcher_args = self.launcher.parse_args()
        self.assertEqual(
            (
                launcher_args.pair_count,
                launcher_args.expected_target0,
                launcher_args.expected_target1,
            ),
            (10_000, 9_954, 46),
        )
        self.assertEqual(
            launcher_args.expected_rule_catalog,
            self.launcher.DEFAULT_RULE_CATALOG,
        )
        self.assertIsNone(launcher_args.expected_source_items)
        self.assertEqual(
            launcher_args.expected_api_base_url,
            "https://openrouter.ai/api/v1",
        )

    def test_uploader_stages_and_pins_mixed_targets(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mixed-upload-test-") as raw:
            root = Path(raw)
            source, stage = root / "source", root / "stage"
            source.mkdir()
            items = pd.DataFrame(
                [
                    {
                        "id": 1,
                        "name": "чехол красный",
                        "attributes": json.dumps(
                            {"Тип товара": "чехол", "Цвет": "красный"},
                            ensure_ascii=False,
                        ),
                        "category": "Электроника",
                    },
                    {
                        "id": -1,
                        "name": "чехол синий",
                        "attributes": json.dumps(
                            {"Тип товара": "чехол", "Цвет": "синий"},
                            ensure_ascii=False,
                        ),
                        "category": "Электроника",
                    },
                    {
                        "id": 2,
                        "name": "кабель 1 м",
                        "attributes": json.dumps(
                            {"Тип товара": "кабель", "Длина": "1 м"},
                            ensure_ascii=False,
                        ),
                        "category": "Электроника",
                    },
                    {
                        "id": -2,
                        "name": "кабель 2 м",
                        "attributes": json.dumps(
                            {"Тип товара": "кабель", "Длина": "2 м"},
                            ensure_ascii=False,
                        ),
                        "category": "Электроника",
                    },
                ]
            )
            pairs = pd.DataFrame(
                [
                    {"id1": 1, "id2": -1, "target": 0},
                    {"id1": 2, "id2": -2, "target": 1},
                ]
            )
            metadata = pairs.assign(task_index=[0, 1])
            item_path, pair_path, metadata_path = (
                source / "items.parquet",
                source / "pairs.parquet",
                source / "pair_generation_metadata.parquet",
            )
            items.to_parquet(item_path, index=False)
            pairs.to_parquet(pair_path, index=False)
            metadata.to_parquet(metadata_path, index=False)
            summary = {
                "run_signature": "a" * 64,
                "model": "provider/model",
                "api_base_url": "https://openrouter.ai/api/v1",
                "structured_output": False,
                "reasoning_effort": "low",
                "prompt_sha256": "b" * 64,
                "rule_catalogs": [{"path": "rules.json", "sha256": "c" * 64}],
                "rule_tiers": ["SEMANTIC_TEST"],
                "base_items_path": "/tmp/style-donors.parquet",
                "base_items_sha256": "e" * 64,
                "label_one_fraction": 0.5,
                "planned_target_counts": {"0": 1, "1": 1},
                "realized_target_counts": {"0": 1, "1": 1},
            }
            checked = {
                "paths": {
                    "items": item_path,
                    "pairs": pair_path,
                    "metadata": metadata_path,
                },
                "summary": summary,
                "validation": {"valid": True, "pairs": 2},
            }
            with (
                mock.patch.object(
                    self.uploader.legacy_upload,
                    "require_complete_source",
                    return_value=checked,
                ),
                mock.patch.object(
                    self.uploader.legacy_upload,
                    "attempt_diversity_provenance",
                    return_value={"selected_task_count": 2},
                ),
            ):
                manifest = self.uploader.build_payload(
                    source,
                    stage,
                    "owner",
                    2,
                    {"0": 1, "1": 1},
                    dataset_slug="mixed-test",
                    artifact_tag="mixed-test",
                    label_source="mixed_test",
                )

            self.assertEqual(manifest["targets"], {"0": 1, "1": 1})
            self.assertEqual(
                manifest["target_quota"]["policy"], "explicit_exact_counts"
            )
            self.assertEqual(manifest["source_provenance"]["reasoning_effort"], "low")
            self.assertEqual(
                manifest["source_provenance"]["base_items_sha256"], "e" * 64
            )
            staged_pairs = pd.read_parquet(
                stage / "generation_rule_pairs_mixed-test.parquet"
            )
            self.assertEqual(
                staged_pairs.groupby("target").size().to_dict(), {0: 1, 1: 1}
            )
            self.assertEqual(set(staged_pairs["label_source"]), {"mixed_test"})

    def test_notebook_generator_changes_only_editable_cells(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mixed-notebook-test-") as raw:
            output = Path(raw) / "mixed.ipynb"
            args = argparse.Namespace(
                pair_count=10_000,
                expected_target0=9_954,
                expected_target1=46,
                artifact_tag="mixed10",
                output=output,
                experiment_label="mixed_test",
                dataset_ref="owner/mixed-test",
                upload_manifest_sha256="d" * 64,
                label_source="mixed_test",
                notes=None,
            )
            with mock.patch.object(self.notebook, "parse_args", return_value=args):
                self.notebook.main()

            source = nbformat.read(self.notebook.SOURCE, as_version=4)
            generated = nbformat.read(output, as_version=4)
            self.assertEqual(len(source.cells), len(generated.cells))
            changed_tags: list[list[str]] = []
            for before, after in zip(source.cells, generated.cells, strict=True):
                tags = before.get("metadata", {}).get("tags", [])
                if before.source != after.source:
                    changed_tags.append(tags)
                if "frozen" in tags:
                    self.assertEqual(before.source, after.source)
            self.assertEqual(len(changed_tags), 2)
            self.assertTrue(
                all(
                    "experiment-routing" in tags or "data-hook" in tags
                    for tags in changed_tags
                )
            )
            data_cell = next(
                cell
                for cell in generated.cells
                if cell.cell_type == "code"
                and "data-hook" in cell.get("metadata", {}).get("tags", [])
            )
            self.assertIn(
                "expected_targets = {'0': 9954, '1': 46}", data_cell.source
            )
            self.assertIn("extra_pairs[\"target\"].isin([0, 1])", data_cell.source)
            self.assertNotIn("target\"].eq(0).all", data_cell.source)

    def test_schedule_verifier_accepts_one_rule_per_label(self) -> None:
        def rule(rule_id: str, label: int, concept: str) -> MutationRule:
            return MutationRule(
                generation_rule_id=rule_id,
                source_rule_id=f"source-{rule_id}",
                generation_tier="SEMANTIC_TEST",
                label=label,
                concept=concept,
                relation="different_value",
                semantic_family="semantic_test",
                attribute_key=concept,
                anchor_hint="test",
                allowed_categories=("Электроника",),
                generation_action="replace",
                required_postcondition="replace one field",
                source_path="test",
                allowed_product_types=("чехол",),
            )

        rules = [rule("r0", 0, "color"), rule("r1", 1, "material")]
        pairs = pd.DataFrame(
            [
                {"id1": 101, "id2": -1, "target": 0},
                {"id1": 102, "id2": -2, "target": 1},
            ]
        )
        rows = []
        for index, current in enumerate(rules):
            rows.append(
                {
                    "id1": 101 + index,
                    "id2": -1 - index,
                    "target": current.label,
                    "task_index": index,
                    "source_style_id": 201 + index,
                    "category": "Электроника",
                    "product_type": "чехол",
                    "rule_count": 1,
                    "rule_ids": json.dumps([current.generation_rule_id]),
                    "scheduled_rule_ids": json.dumps([current.generation_rule_id]),
                    "scheduled_rule_profile_ids": json.dumps([f"p{index}"]),
                    "rule_schedule_sha256": "pending",
                    "rules_json": json.dumps(
                        [{"generation_rule_id": current.generation_rule_id}]
                    ),
                    "applications_json": json.dumps(
                        [
                            {
                                "generation_rule_id": current.generation_rule_id,
                                "concept": current.concept,
                                "attribute_key": current.attribute_key,
                                "original_value": "old",
                                "new_value": "new",
                            }
                        ]
                    ),
                    "model": "provider/model",
                    "run_signature": "a" * 64,
                }
            )
        metadata = pd.DataFrame(rows)
        computed_sha = self.launcher.schedule_sha256(metadata)
        metadata["rule_schedule_sha256"] = computed_sha
        donors = pd.DataFrame(
            {
                "id": [201, 202],
                "category": ["Электроника", "Электроника"],
            }
        )
        result = self.launcher.verify_schedule_rows(
            metadata,
            pairs,
            rules,
            donors,
            {"rule_schedule_sha256": computed_sha},
            expected_positive_rule_counts={"r1": 1},
        )
        self.assertEqual(result["label_rule_applications"], {"0": 1, "1": 1})
        self.assertEqual(result["primary_rule_coverage"], 2)
        self.assertEqual(
            result["primary_rule_usage_by_label"]["1"], {"r1": 1}
        )

    def test_transition_v4_catalog_is_sha_pinned_with_exact_capacity(self) -> None:
        path = self.launcher.DEFAULT_RULE_CATALOG.resolve()
        manifest_path = path.with_suffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(
            self.launcher.sha256_file(manifest_path),
            self.launcher.DEFAULT_RULE_CATALOG_MANIFEST_SHA256,
        )
        contract = self.launcher.verify_transition_catalog(
            path,
            manifest,
            self.launcher.sha256_file(path),
        )
        self.assertEqual(len(contract["positive_rule_ids"]), 4)
        self.assertEqual(contract["transition_count"], 23)
        self.assertEqual(sum(contract["positive_rule_counts"].values()), 46)
        self.assertEqual(
            contract["positive_rule_counts"],
            self.launcher.DEFAULT_POSITIVE_RULE_CAPACITIES,
        )

    def test_positive_transition_guard_requires_exact_coverage_and_context(
        self,
    ) -> None:
        rule_id = "positive-transition"
        rule = MutationRule(
            generation_rule_id=rule_id,
            source_rule_id="source-positive-transition",
            generation_tier="SEMANTIC_TEST",
            label=1,
            concept="age_rating",
            relation="different_value",
            semantic_family="semantic_test",
            attribute_key="Возрастная рекомендация",
            anchor_hint="test",
            allowed_categories=("Хобби и творчество",),
            generation_action="replace",
            required_postcondition="replace one field",
            source_path="test",
            allowed_product_types=("настольная игра",),
            allowed_anchor_context_keys=("Бренд", "Название игры"),
            required_anchor_context_keys=("Бренд", "Название игры"),
            target_value_domain=("детская", "3 лет", "6 лет"),
            allowed_value_transitions=(
                ("детская", "3 лет"),
                ("детская", "6 лет"),
            ),
            primary_task_safety_cap=4,
        )
        transitions = [
            ("детская", "3 лет"),
            ("3 лет", "детская"),
            ("детская", "6 лет"),
            ("6 лет", "детская"),
        ]
        rows: list[dict[str, object]] = []
        items: list[dict[str, object]] = []
        for index, (original, new) in enumerate(transitions):
            id1, id2 = 100 + index, -100 - index
            common = {
                "Тип товара": "настольная игра",
                "Бренд": f"бренд {index}",
                "Название игры": f"игра {index}",
            }
            items.extend(
                [
                    {
                        "id": id1,
                        "attributes": json.dumps(
                            {
                                **common,
                                "Возрастная рекомендация": original,
                            },
                            ensure_ascii=False,
                        ),
                    },
                    {
                        "id": id2,
                        "attributes": json.dumps(
                            {
                                **common,
                                "Возрастная рекомендация": new,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ]
            )
            rows.append(
                {
                    "id1": id1,
                    "id2": id2,
                    "target": 1,
                    "task_index": index,
                    "scheduled_rule_ids": json.dumps([rule_id]),
                    "applications_json": json.dumps(
                        [
                            {
                                "generation_rule_id": rule_id,
                                "concept": "age_rating",
                                "attribute_key": "Возрастная рекомендация",
                                "original_value": original,
                                "new_value": new,
                            }
                        ],
                        ensure_ascii=False,
                    ),
                }
            )
        contract = {
            "transitions_by_rule": {
                rule_id: frozenset(
                    {
                        ("3 лет", "детская"),
                        ("6 лет", "детская"),
                    }
                )
            },
            "required_context_keys_by_rule": {
                rule_id: ("Бренд", "Название игры")
            },
            "transition_repetitions": 2,
        }
        metadata = pd.DataFrame(rows)
        generated_items = pd.DataFrame(items)
        result = self.launcher.verify_positive_transition_rows(
            metadata, generated_items, [rule], contract
        )
        self.assertEqual(result["positive_rows"], 4)
        self.assertEqual(result["transition_coverage"], 2)
        self.assertEqual(result["maximum_transition_count"], 2)

        disallowed = metadata.copy()
        application = json.loads(disallowed.loc[0, "applications_json"])
        application[0]["new_value"] = "12 лет"
        disallowed.loc[0, "applications_json"] = json.dumps(
            application, ensure_ascii=False
        )
        with self.assertRaisesRegex(RuntimeError, "disallowed value transition"):
            self.launcher.verify_positive_transition_rows(
                disallowed, generated_items, [rule], contract
            )

        missing_context_items = generated_items.copy()
        attributes = json.loads(missing_context_items.loc[0, "attributes"])
        del attributes["Бренд"]
        missing_context_items.loc[0, "attributes"] = json.dumps(
            attributes, ensure_ascii=False
        )
        with self.assertRaisesRegex(RuntimeError, "required context"):
            self.launcher.verify_positive_transition_rows(
                metadata, missing_context_items, [rule], contract
            )

        over_cap = metadata.copy()
        over_cap_items = generated_items.copy()
        third_application = json.loads(over_cap.loc[2, "applications_json"])
        third_application[0]["new_value"] = "3 лет"
        over_cap.loc[2, "applications_json"] = json.dumps(
            third_application, ensure_ascii=False
        )
        third_attributes = json.loads(over_cap_items.loc[5, "attributes"])
        third_attributes["Возрастная рекомендация"] = "3 лет"
        over_cap_items.loc[5, "attributes"] = json.dumps(
            third_attributes, ensure_ascii=False
        )
        with self.assertRaisesRegex(RuntimeError, "repetition cap"):
            self.launcher.verify_positive_transition_rows(
                over_cap, over_cap_items, [rule], contract
            )

    def test_frozen_contract_requires_identity_freeze_and_exact_positive_usage(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="mixed-frozen-test-") as raw:
            frozen_dir = Path(raw)
            pd.DataFrame(
                [
                    {"id1": 1, "id2": -1, "target": 0},
                    {"id1": 2, "id2": -2, "target": 1},
                ]
            ).to_parquet(frozen_dir / "pairs.parquet", index=False)
            pd.DataFrame(
                [
                    {"id": 1, "attributes": "{}"},
                    {"id": -1, "attributes": "{}"},
                    {"id": 2, "attributes": "{}"},
                    {"id": -2, "attributes": "{}"},
                ]
            ).to_parquet(frozen_dir / "items.parquet", index=False)
            pd.DataFrame(
                [
                    {"task_index": 0},
                    {"task_index": 1},
                ]
            ).to_parquet(
                frozen_dir / "pair_generation_metadata.parquet", index=False
            )
            (frozen_dir / "dropped_pairs.json").write_text("[]", encoding="utf-8")
            frozen = {
                "summary": {
                    "source_dir": "/tmp/raw",
                    "source_generated_pairs": 2,
                    "source_realized_target_counts": {"0": 1, "1": 1},
                    "dropped_before_target": 0,
                    "frozen_rule_schedule": {
                        "selected_task_count": 2,
                        "source_rule_schedule_sha256": "a" * 64,
                        "primary_rule_usage": {"negative": 1, "positive": 1},
                        "primary_rule_coverage": 2,
                        "full_primary_rule_coverage": True,
                    },
                }
            }
            with (
                mock.patch.object(
                    self.launcher.strict_checks,
                    "verify_frozen_attempt_diversity",
                ),
                mock.patch.object(
                    self.launcher.strict_checks,
                    "verify_frozen_global_card_uniqueness",
                    return_value={"frozen_unique_card_count": 4},
                ),
                mock.patch.object(
                    self.launcher,
                    "verify_positive_transition_rows",
                    return_value={"transition_coverage": 1},
                ),
            ):
                result = self.launcher.verify_frozen(
                    frozen,
                    frozen_dir,
                    {"0": 1, "1": 1},
                    2,
                    "a" * 64,
                    {"positive": 1},
                    [],
                    {},
                    2,
                )
                self.assertEqual(result["positive_rule_counts"], {"positive": 1})
                frozen["summary"]["dropped_before_target"] = 1
                with self.assertRaisesRegex(RuntimeError, "freeze-time drops"):
                    self.launcher.verify_frozen(
                        frozen,
                        frozen_dir,
                        {"0": 1, "1": 1},
                        2,
                        "a" * 64,
                        {"positive": 1},
                        [],
                        {},
                        2,
                    )


if __name__ == "__main__":
    unittest.main()
