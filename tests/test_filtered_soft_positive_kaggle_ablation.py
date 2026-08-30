from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import create_filtered_soft_positive_notebook as notebook_builder
import launch_filtered_soft_positive_kaggle_ablation as launcher


SMALL_CANDIDATE_COUNT = 4
SMALL_DUPLICATE_COUNT = 1
PRODUCTION_CANDIDATE_COUNT = launcher.CANDIDATE_COUNT
PRODUCTION_DUPLICATE_COUNT = launcher.DUPLICATE_SOURCE_ROW_COUNT
SOURCE_SIGNATURE = "b" * 64


class FilteredSoftPositiveLauncherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.contract_patch = mock.patch.multiple(
            launcher,
            CANDIDATE_COUNT=SMALL_CANDIDATE_COUNT,
            DUPLICATE_SOURCE_ROW_COUNT=SMALL_DUPLICATE_COUNT,
        )
        self.contract_patch.start()
        self._write_valid_source()

    def tearDown(self) -> None:
        self.contract_patch.stop()
        self.temporary.cleanup()

    def _write_json(self, name: str, value: dict) -> None:
        launcher.base.atomic_json(value, self.source / name)

    def _file_record(self, name: str) -> dict:
        return launcher.base._file_record(self.source / name)

    def _common(self, weight_stats: dict) -> dict:
        return {
            "contract_version": launcher.CONTRACT_VERSION,
            "label_source": launcher.LABEL_SOURCE,
            "pair_count": 2,
            "target_counts": {"0": 0, "1": 2},
            "candidate_count": SMALL_CANDIDATE_COUNT,
            "selected_count": 2,
            "rejected_count": 2,
            "sample_weight_stats": weight_stats,
            "validation_fact_overlap_count": 0,
            "forbidden_ood_categories": [],
            "run_signature": self.run_signature,
        }

    def _write_valid_source(self) -> None:
        items = pd.DataFrame(
            [
                {
                    "id": item_id,
                    "name": f"Тестовый товар {item_id}",
                    "attributes": json.dumps(
                        {"Цвет": f"оттенок {item_id}"}, ensure_ascii=False
                    ),
                    "category": "Дом и сад",
                }
                for item_id in range(1, 5)
            ]
        )
        pairs = pd.DataFrame(
            {
                "id1": [1, 3],
                "id2": [2, 4],
                "target": [1, 1],
                "label_source": [launcher.LABEL_SOURCE] * 2,
                "sample_weight": [0.5, 1.5],
            }
        )
        metadata = pd.DataFrame(
            {
                "id1": [1, 3],
                "id2": [2, 4],
                "target": [1, 1],
                "candidate_key": ["candidate-0", "candidate-1"],
                "baseline_score": [0.7, 0.6],
                "score_ab": [0.8, 0.7],
                "score_ba": [0.6, 0.5],
                "score_order_gap": [0.2, 0.2],
                "quality_score": [0.9, 0.8],
                "sample_weight": [0.5, 1.5],
                "source_tier": ["A", "B"],
                "source_run_signature": [SOURCE_SIGNATURE] * 2,
                "source_task_index": [10, 20],
                "category": ["Дом и сад", "Дом и сад"],
                "rule_id": ["rule-0", "rule-1"],
                "concept": ["цвет", "размер"],
                "relation": ["эквивалентно", "эквивалентно"],
                "product_type": ["ваза", "полка"],
                "required_attribute_key": ["Цвет", "Размер"],
                "original_value": ["red", "large"],
                "new_value": ["scarlet", "big"],
                "rule_probability": [0.95, 0.91],
                "rule_support": [5, 4],
                "rule_singleton_support": [2, 2],
                "human_category_original_support": [3, 4],
                "human_category_new_support": [3, 4],
                "human_scope_original_support": [2, 2],
                "human_scope_new_support": [2, 2],
                "evidence_type": [
                    "source_exact_transition",
                    "source_exact_transition",
                ],
                "evidence_source": ["tier_a", "tier_b"],
                "evidence_value": ["red=scarlet", "large=big"],
                "selection_rank": [1, 2],
            }
        )
        decisions = pd.DataFrame(
            {
                "candidate_key": [f"candidate-{index}" for index in range(4)],
                "source_tier": ["A", "B", "A", "B"],
                "source_run_signature": [SOURCE_SIGNATURE] * 4,
                "source_task_index": [10, 20, 30, 40],
                "category": ["Дом и сад"] * 4,
                "product_type": ["ваза", "полка", "стол", "лампа"],
                "rule_id": ["rule-0", "rule-1", "rule-2", "rule-3"],
                "concept": ["цвет", "размер", "материал", "форма"],
                "relation": ["эквивалентно"] * 4,
                "required_attribute_key": ["Цвет", "Размер", "Материал", "Форма"],
                "original_value": ["red", "large", "wood", "round"],
                "new_value": ["scarlet", "big", "oak", "circle"],
                "rule_probability": [0.95, 0.91, 0.85, 0.82],
                "rule_support": [5, 4, 3, 3],
                "rule_singleton_support": [2, 2, 1, 1],
                "human_category_original_support": [3, 4, 2, 2],
                "human_category_new_support": [3, 4, 2, 2],
                "human_scope_original_support": [2, 2, 1, 1],
                "human_scope_new_support": [2, 2, 1, 1],
                "selected": pd.Series([True, True, False, False], dtype="bool"),
                "rejection_reasons": [
                    "[]",
                    "[]",
                    '["low_quality"]',
                    '["too_easy"]',
                ],
                "duplicate_source_keys_json": [
                    '["tier-b-secondary-181"]',
                    "[]",
                    "[]",
                    "[]",
                ],
                "baseline_score": [0.7, 0.6, 0.95, 0.1],
                "score_ab": [0.8, 0.7, 0.95, 0.2],
                "score_ba": [0.6, 0.5, 0.95, 0.0],
                "score_order_gap": [0.2, 0.2, 0.0, 0.2],
                "quality_score": [0.9, 0.8, 0.2, 0.7],
                "evidence_type": ["source_exact_transition"] * 4,
                "evidence_source": ["tier_a", "tier_b", "tier_a", "tier_b"],
                "evidence_value": [
                    "red=scarlet",
                    "large=big",
                    "x=y",
                    "a=b",
                ],
            }
        )
        items.to_parquet(self.source / "items.parquet", index=False)
        pairs.to_parquet(self.source / "pairs.parquet", index=False)
        metadata.to_parquet(
            self.source / "pair_generation_metadata.parquet", index=False
        )
        decisions.to_parquet(
            self.source / "candidate_decisions.parquet", index=False
        )
        weight_stats = launcher.sample_weight_statistics(pairs["sample_weight"])
        self.input_provenance = {
            "fixture": {"path": "/fixture", "sha256": "d" * 64}
        }
        self.validation_reference = {
            "version": "fixture_validation_v1",
            "paths": {},
        }
        self.signature_payload = {
            "contract_version": launcher.CONTRACT_VERSION,
            "builder_sha256": launcher.base.sha256_file(
                launcher.FILTER_BUILDER_PATH
            ),
            "label_source": launcher.LABEL_SOURCE,
            "inputs": self.input_provenance,
            "validation_reference": self.validation_reference,
            "selected_candidate_keys_sha256": launcher.base.canonical_sha256(
                metadata["candidate_key"].tolist()
            ),
            "sample_weight_stats": weight_stats,
        }
        self.run_signature = launcher.base.canonical_sha256(self.signature_payload)
        common = self._common(weight_stats)
        summary = {
            **common,
            "generated_pairs": 2,
            "generated_items": 4,
            "metadata_rows": 2,
            "input_provenance": self.input_provenance,
            "validation_reference": self.validation_reference,
            "signature_payload": self.signature_payload,
        }
        selection = {**common, "valid": True, "selected_pairs": 2}
        validation = {
            **common,
            "valid": True,
            "pairs": 2,
            "items": 4,
            "metadata_rows": 2,
            "one_use_unique_cards": True,
            "validation_reference": self.validation_reference,
        }
        self._write_json("summary.json", summary)
        self._write_json("selection_report.json", selection)
        self._write_json("validation_report.json", validation)
        manifest = {
            **common,
            "valid": True,
            "input_provenance": self.input_provenance,
            "validation_reference": self.validation_reference,
            "signature_payload": self.signature_payload,
            "files": {
                name: self._file_record(name)
                for name in sorted(launcher.BUILD_MANIFEST_FILE_SET)
            },
        }
        self._write_json("build_manifest.json", manifest)

    def _refresh_manifest_file_record(self, filename: str) -> None:
        manifest_path = self.source / "build_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"][filename] = self._file_record(filename)
        self._write_json("build_manifest.json", manifest)

    def test_constants_pin_full_candidate_audit(self) -> None:
        # Compact fixtures patch runtime counts; these capture the release pins.
        self.assertEqual(PRODUCTION_CANDIDATE_COUNT, 31_176)
        self.assertEqual(PRODUCTION_DUPLICATE_COUNT, 181)

    def test_valid_source_staging_and_notebook_preserve_weights(self) -> None:
        checked = launcher.verify_source(self.source, pair_count=2)
        self.assertEqual(checked["sample_weight_stats"]["sum"], 2.0)
        self.assertEqual(
            checked["validation"]["candidate_decisions"]["candidate_count"], 4
        )
        stage = self.root / "stage"
        manifest = launcher.prepare_upload_payload(
            checked,
            stage_dir=stage,
            owner="tester",
            dataset_slug="filtered-soft-positive-test",
            artifact_tag="filtered-test",
        )
        pair_path = stage / "generation_rule_pairs_filtered-test.parquet"
        staged_pairs = pd.read_parquet(pair_path)
        self.assertEqual(staged_pairs["sample_weight"].tolist(), [0.5, 1.5])
        self.assertEqual(
            manifest["sample_weight_stats"], checked["sample_weight_stats"]
        )
        self.assertFalse(
            manifest["source_provenance"]["candidate_decisions_in_train_payload"]
        )
        self.assertIn(
            "candidate_decisions.parquet",
            manifest["source_provenance"]["source_files"],
        )

        manifest_sha = launcher.base.sha256_file(stage / "upload_manifest.json")
        notebook_path = self.root / "filtered.ipynb"
        launcher.generate_notebook(
            notebook=notebook_path,
            pair_count=2,
            artifact_tag="filtered-test",
            experiment_label="filtered_test",
            dataset_ref=manifest["dataset"],
            upload_manifest_sha256=manifest_sha,
            sample_weight_stats=checked["sample_weight_stats"],
            notes="weighted test",
        )
        notebook = nbformat.read(notebook_path, as_version=4)
        hook = next(
            cell.source
            for cell in notebook.cells
            if cell.cell_type == "code"
            and "data-hook" in cell.get("metadata", {}).get("tags", [])
        )
        self.assertIn('extra_pairs["sample_weight"] = pd.to_numeric', hook)
        self.assertNotIn('extra_pairs["sample_weight"] = 1.0', hook)
        self.assertIn("float64_le_pair_order_v1", hook)
        namespace: dict = {}
        exec(compile(hook, "<filtered-data-hook>", "exec"), namespace)
        human_pairs = pd.DataFrame(
            {"id1": [101], "id2": [102], "target": [0]}
        )
        human_items = pd.DataFrame(
            {
                "id": [101, 102],
                "name": ["human a", "human b"],
                "category": ["Дом и сад", "Дом и сад"],
                "product_text": ["human a", "human b"],
            }
        )
        train_pairs, train_items = namespace["build_train_data"](
            human_pairs, human_items, self.root
        )
        self.assertEqual(train_pairs["sample_weight"].tolist(), [1.0, 0.5, 1.5])
        self.assertEqual(
            train_pairs["label_source"].tolist(),
            ["human", launcher.LABEL_SOURCE, launcher.LABEL_SOURCE],
        )
        self.assertEqual(len(train_items), 6)

    def test_source_semantics_fail_closed_after_hash_refresh(self) -> None:
        cases = []

        def wrong_pair_label() -> None:
            pairs = pd.read_parquet(self.source / "pairs.parquet")
            pairs["label_source"] = "wrong"
            pairs.to_parquet(self.source / "pairs.parquet", index=False)
            self._refresh_manifest_file_record("pairs.parquet")

        cases.append(("label_source", wrong_pair_label))

        for label, mutate in cases:
            with self.subTest(label=label):
                mutate()
                with self.assertRaisesRegex(RuntimeError, "label_source"):
                    launcher.verify_source(self.source, pair_count=2)
                self._write_valid_source()

        metadata = pd.read_parquet(
            self.source / "pair_generation_metadata.parquet"
        )
        metadata.loc[0, "evidence_value"] = ""
        metadata.to_parquet(
            self.source / "pair_generation_metadata.parquet", index=False
        )
        self._refresh_manifest_file_record("pair_generation_metadata.parquet")
        with self.assertRaisesRegex(RuntimeError, "evidence_value"):
            launcher.verify_source(self.source, pair_count=2)
        self._write_valid_source()

        decisions = pd.read_parquet(self.source / "candidate_decisions.parquet")
        decisions.loc[2, "rejection_reasons"] = "[]"
        decisions.to_parquet(
            self.source / "candidate_decisions.parquet", index=False
        )
        self._refresh_manifest_file_record("candidate_decisions.parquet")
        with self.assertRaisesRegex(RuntimeError, "rejection_reasons"):
            launcher.verify_source(self.source, pair_count=2)

    def test_provenance_tamper_and_ood_are_rejected(self) -> None:
        summary_path = self.source / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["validation_fact_overlap_count"] = 1
        self._write_json("summary.json", summary)
        self._refresh_manifest_file_record("summary.json")
        with self.assertRaisesRegex(RuntimeError, "validation_fact_overlap_count"):
            launcher.verify_source(self.source, pair_count=2)
        self._write_valid_source()

        summary = json.loads(
            (self.source / "summary.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (self.source / "build_manifest.json").read_text(encoding="utf-8")
        )
        for document in (summary, manifest):
            document["signature_payload"]["selected_candidate_keys_sha256"] = (
                "e" * 64
            )
        self._write_json("summary.json", summary)
        manifest["files"]["summary.json"] = self._file_record("summary.json")
        self._write_json("build_manifest.json", manifest)
        with self.assertRaisesRegex(RuntimeError, "authenticate signature_payload"):
            launcher.verify_source(self.source, pair_count=2)
        self._write_valid_source()

        items = pd.read_parquet(self.source / "items.parquet")
        items.loc[0, "category"] = "Одежда"
        items.to_parquet(self.source / "items.parquet", index=False)
        self._refresh_manifest_file_record("items.parquet")
        with self.assertRaisesRegex(RuntimeError, "OOD"):
            launcher.verify_source(self.source, pair_count=2)

    def test_main_invalid_source_stops_before_env_staging_or_network(self) -> None:
        (self.source / "summary.json").unlink()
        argv = [
            "launcher",
            "--source-dir",
            str(self.source),
            "--pair-count",
            "2",
            "--dataset-slug",
            "filtered-test",
            "--artifact-tag",
            "filtered-test",
            "--experiment-label",
            "filtered_test",
            "--kernel-slug",
            "filtered-test",
            "--dry-run-only",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(launcher.base.kaggle, "load_dotenv") as load_env,
            mock.patch.object(launcher.base, "run") as run,
            mock.patch.object(launcher.base, "upload_dataset") as upload,
        ):
            with self.assertRaisesRegex(RuntimeError, "documents differ"):
                launcher.main()
        load_env.assert_not_called()
        run.assert_not_called()
        upload.assert_not_called()

    def test_valid_main_dry_run_stages_locked_notebook_without_upload(self) -> None:
        stage = self.root / "main-stage"
        notebook = self.root / "main.ipynb"
        report = self.root / "launcher.json"
        argv = [
            "launcher",
            "--source-dir",
            str(self.source),
            "--pair-count",
            "2",
            "--dataset-slug",
            "filtered-soft-positive-test",
            "--artifact-tag",
            "filtered-test",
            "--experiment-label",
            "filtered_test",
            "--kernel-slug",
            "filtered-test",
            "--stage-dir",
            str(stage),
            "--notebook",
            str(notebook),
            "--report",
            str(report),
            "--dry-run-only",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.dict(
                launcher.os.environ,
                {
                    "KAGGLE_USERNAME": "tester",
                    "KAGGLE_IS_PRIVATE": "true",
                    "KAGGLE_ENABLE_INTERNET": "true",
                },
                clear=False,
            ),
            mock.patch.object(launcher.base.kaggle, "load_dotenv") as load_env,
            mock.patch.object(launcher.base, "run") as run,
            mock.patch.object(launcher.base, "upload_dataset") as upload,
            mock.patch("builtins.print"),
        ):
            launcher.main()
        load_env.assert_called_once()
        upload.assert_not_called()
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command[-1], "--dry-run")
        for dataset in (
            launcher.base.VALIDATION_DATASET,
            launcher.base.CHECKPOINT_DATASET,
            launcher.base.SIGNIFICANCE_DATASET,
            "tester/filtered-soft-positive-test",
        ):
            self.assertIn(dataset, command)
        result = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "dry_run_complete")
        self.assertEqual(result["sample_weight_stats"]["sum"], 2.0)
        self.assertTrue(notebook.is_file())

        finalized_argv = [
            value for value in argv if value != "--dry-run-only"
        ] + ["--finalize-local-output"]
        completed = {
            "run_id": "f" * 32,
            "comparison": {"status": "ready"},
            "sync": {"status": "synced"},
            "training_report": {
                "training_sampling": "none",
                "training_loss_weighting": "none",
                "training_source_counts": {
                    "human": 3,
                    launcher.LABEL_SOURCE: 2,
                },
                "training_source_weight_mass": {
                    "human": 3.0,
                    launcher.LABEL_SOURCE: 2.0,
                },
                "training_loss_weight_min": 0.5,
                "training_loss_weight_max": 1.5,
            },
        }
        with (
            mock.patch.object(sys, "argv", finalized_argv),
            mock.patch.object(launcher.base.kaggle, "load_dotenv") as load_env,
            mock.patch.object(launcher.base, "run") as run,
            mock.patch.object(launcher.base, "upload_dataset") as upload,
            mock.patch.object(
                launcher.base, "verify_completion", return_value=completed
            ) as verify_completion,
            mock.patch("builtins.print"),
        ):
            launcher.main()
        load_env.assert_not_called()
        run.assert_not_called()
        upload.assert_not_called()
        verify_completion.assert_called_once()
        result = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["finalized_from_local_output"])
        self.assertEqual(result["run_id"], "f" * 32)

    def test_weighted_completion_requires_source_mass_and_unit_human_weight(self) -> None:
        stats = launcher.sample_weight_statistics(pd.Series([0.5, 1.0]))
        replayed = launcher.np.concatenate(
            [
                launcher.np.ones(306_669, dtype=launcher.np.float32),
                launcher.np.asarray([0.5, 1.0], dtype=launcher.np.float32),
            ]
        )
        replayed /= replayed.mean()
        completed = {
            "training_report": {
                "training_sampling": "none",
                "training_loss_weighting": "none",
                "training_source_counts": {
                    "human": 306_669,
                    launcher.LABEL_SOURCE: 2,
                },
                "training_source_weight_mass": {
                    "human": float(replayed[:306_669].sum()),
                    launcher.LABEL_SOURCE: float(replayed[306_669:].sum()),
                },
                "training_loss_weight_min": float(replayed.min()),
                "training_loss_weight_max": float(replayed.max()),
            }
        }
        launcher.verify_weighted_completion(
            completed,
            pair_count=2,
            weight_stats=stats,
            sample_weights=pd.Series([0.5, 1.0]),
        )
        completed["training_report"]["training_source_weight_mass"][
            launcher.LABEL_SOURCE
        ] = 1.99
        with self.assertRaisesRegex(RuntimeError, "weight mass"):
            launcher.verify_weighted_completion(
                completed,
                pair_count=2,
                weight_stats=stats,
                sample_weights=pd.Series([0.5, 1.0]),
            )


class FilteredNotebookWeightStatsTest(unittest.TestCase):
    def test_weight_stats_contract_rejects_incomplete_or_nonpositive(self) -> None:
        valid = {
            "version": notebook_builder.WEIGHT_STATS_VERSION,
            "count": 2,
            "sum": 2.0,
            "min": 0.5,
            "max": 1.5,
            "sha256": "c" * 64,
        }
        self.assertEqual(
            notebook_builder.validate_weight_stats(valid, pair_count=2), valid
        )
        for broken in (
            {**valid, "count": 3},
            {**valid, "min": 0.0},
            {key: value for key, value in valid.items() if key != "sha256"},
        ):
            with self.subTest(broken=broken):
                with self.assertRaises(ValueError):
                    notebook_builder.validate_weight_stats(broken, pair_count=2)


if __name__ == "__main__":
    unittest.main()
