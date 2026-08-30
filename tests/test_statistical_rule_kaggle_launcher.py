from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from item_pipeline.normalization import stable_hash64
from item_pipeline.pair_generate import PairGenerationTask, _attempt_diversity_nonce
from item_pipeline.pair_rules import MutationRule

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "launch_statistical_rule_kaggle_experiment.py"
CATALOG_V3 = (
    ROOT
    / "configs"
    / "generation_rule_catalog_statistical_v1"
    / "statistical_negative_rules_min2_p80_scoped_v3.json"
)


def load_launcher():
    spec = importlib.util.spec_from_file_location(
        "statistical_rule_kaggle_launcher", LAUNCHER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load statistical-rule Kaggle launcher")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class StatisticalRuleKaggleLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = load_launcher()

    def _local_finalization_fixture(
        self, root: Path
    ) -> dict[str, object]:
        pair_count = 2
        dataset_slug = "generated-pairs-local-test"
        dataset_ref = f"testowner/{dataset_slug}"
        experiment_label = "minilm-local-finalize-test"
        kernel_slug = "minilm-local-finalize-test"
        label_source = "generated_local_test"
        run_id = "run-local-123"
        frozen_dir = root / "frozen"
        stage_dir = root / ".kaggle" / "datasets" / dataset_slug
        output_dir = root / "outputs" / kernel_slug
        model_dir = output_dir / "trained_model"
        for directory in (frozen_dir, stage_dir, model_dir):
            directory.mkdir(parents=True)

        uniqueness = {
            "version": self.launcher.FROZEN_GLOBAL_CARD_UNIQUENESS_VERSION,
            "card_key_version": self.launcher.GLOBAL_CARD_KEY_VERSION,
            "category_agnostic": True,
            "frozen_card_count": pair_count * 2,
            "frozen_unique_card_count": pair_count * 2,
            "frozen_duplicate_card_group_count": 0,
            "frozen_duplicate_card_row_count": 0,
        }
        freeze_manifest_path = frozen_dir / "freeze_manifest.json"
        freeze_manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "count": pair_count,
                    "frozen_global_card_uniqueness": uniqueness,
                }
            ),
            encoding="utf-8",
        )
        staged_freeze_manifest = stage_dir / "freeze_manifest.json"
        staged_freeze_manifest.write_bytes(freeze_manifest_path.read_bytes())
        validation_path = stage_dir / "validation_report.json"
        validation_path.write_text(
            json.dumps({"valid": True, "pairs": pair_count}), encoding="utf-8"
        )

        def inventory(path: Path) -> dict[str, object]:
            return {
                "bytes": path.stat().st_size,
                "sha256": self.launcher.sha256_file(path),
            }

        upload_manifest_path = stage_dir / "upload_manifest.json"
        schedule_sha256 = "b" * 64
        upload_manifest_path.write_text(
            json.dumps(
                {
                    "dataset": dataset_ref,
                    "pairs": pair_count,
                    "items": pair_count * 2,
                    "label_source": label_source,
                    "checkpoint": False,
                    "targets": {"0": pair_count},
                    "source_provenance": {
                        "run_signature": "a" * 64,
                        "semantic_signature": {
                            "semantic_signature_retry": True,
                            "semantic_signature_version": (
                                self.launcher.SEMANTIC_SIGNATURE_VERSION
                            ),
                            "semantic_signature_limit": 2,
                            "semantic_signature_max_count": 2,
                        },
                        "rule_schedule": {
                            "balanced_rule_schedule": True,
                            "rule_schedule_version": self.launcher.SCHEDULE_VERSION,
                            "rule_schedule_sha256": schedule_sha256,
                        },
                        "attempt_diversity": {
                            "attempt_diversity_version": (
                                self.launcher.ATTEMPT_DIVERSITY_VERSION
                            )
                        },
                        "frozen_rule_schedule": {
                            "selected_task_count": pair_count,
                            "source_rule_schedule_version": (
                                self.launcher.SCHEDULE_VERSION
                            ),
                            "source_rule_schedule_sha256": schedule_sha256,
                        },
                        "frozen_semantic_signature": {
                            "semantic_signature_version": (
                                self.launcher.SEMANTIC_SIGNATURE_VERSION
                            ),
                            "semantic_signature_limit": 2,
                            "semantic_signature_max_count": 2,
                        },
                        "frozen_attempt_diversity": {
                            "version": (
                                self.launcher.FROZEN_ATTEMPT_DIVERSITY_VERSION
                            ),
                            "attempt_diversity_version": (
                                self.launcher.ATTEMPT_DIVERSITY_VERSION
                            ),
                            "selected_task_count": pair_count,
                        },
                    },
                    "files": {
                        "freeze_manifest.json": inventory(staged_freeze_manifest),
                        "validation_report.json": inventory(validation_path),
                    },
                }
            ),
            encoding="utf-8",
        )
        upload_manifest_sha256 = self.launcher.sha256_file(upload_manifest_path)

        metrics = {"iid": 0.70, "hard": 0.40, "ood": 0.60}
        validation_splits = {}
        comparison_splits = {}
        for split, metric in metrics.items():
            prediction_name = f"{split}_validation_predictions.parquet"
            (model_dir / prediction_name).write_bytes(f"{split}-predictions".encode())
            validation_splits[split] = {
                "macro_average_precision": metric,
                "predictions_file": prediction_name,
            }
            comparison_splits[split] = {
                "candidate_macro_average_precision": metric,
            }
        training_report = {
            "training_source_counts": {"human": 3, label_source: pair_count},
            "validation_splits": validation_splits,
        }
        (model_dir / "training_report.json").write_text(
            json.dumps(training_report), encoding="utf-8"
        )
        (model_dir / "model.safetensors").write_bytes(b"model")
        comparison = {
            "status": "ready",
            "candidate_run_id": run_id,
            "baseline_run_id": "baseline-run",
            "splits": comparison_splits,
        }
        (output_dir / "baseline_comparison.json").write_text(
            json.dumps(comparison), encoding="utf-8"
        )
        completion = {
            "status": "complete",
            "experiment": experiment_label,
            "experiment_group": "data",
            "run_id": run_id,
            "notes": (
                f"Source dataset {dataset_ref}. "
                f"Upload manifest SHA-256 {upload_manifest_sha256}."
            ),
            "train_data": {
                "label_source_counts": {"human": 3, label_source: pair_count}
            },
            "training_report": training_report,
            "baseline_comparison": comparison,
        }
        (output_dir / "notebook_completed.json").write_text(
            json.dumps(completion), encoding="utf-8"
        )
        (output_dir / "google_sheets_sync.json").write_text(
            json.dumps(
                {
                    "status": "synced",
                    "run_id": run_id,
                    "experiment_group": "data",
                    "comparison_sheet": "data_exps",
                }
            ),
            encoding="utf-8",
        )
        (output_dir / "experiment_run_id.txt").write_text(
            run_id + "\n", encoding="utf-8"
        )
        return {
            "pair_count": pair_count,
            "dataset_slug": dataset_slug,
            "dataset_ref": dataset_ref,
            "experiment_label": experiment_label,
            "kernel_slug": kernel_slug,
            "label_source": label_source,
            "run_id": run_id,
            "frozen_dir": frozen_dir,
            "output_dir": output_dir,
            "upload_manifest_path": upload_manifest_path,
        }

    def _partial_schedule_fixture(self):
        def rule(rule_id: str, concept: str, attribute_key: str) -> MutationRule:
            return MutationRule(
                generation_rule_id=rule_id,
                source_rule_id=f"source-{rule_id}",
                generation_tier="STAT_TEST",
                label=0,
                concept=concept,
                relation="different_value",
                semantic_family="test",
                attribute_key=attribute_key,
                anchor_hint="test",
                allowed_categories=("a",),
                generation_action="replace",
                required_postcondition="replace only target",
                source_path="test",
                allowed_product_types=("widget",),
                profile_capacity_policy_version=(
                    self.launcher.EXPECTED_PROFILE_CAPACITY_POLICY_VERSION
                ),
                profile_capacity_policy_sha256=(
                    self.launcher.EXPECTED_PROFILE_CAPACITY_POLICY_SHA256
                ),
            )

        donors = pd.DataFrame(
            {"id": [101, 102, 103, 104], "category": ["a"] * 4}
        )
        schedule = self.launcher.build_balanced_rule_schedule(
            donors,
            [rule("r0", "color", "Цвет"), rule("r1", "size", "Размер")],
            count=4,
            seed=17,
            two_rule_fraction=0.0,
            semantic_signature_limit=2,
        )
        accepted = [0, 2, 3]
        rows = []
        for task_index in accepted:
            provenance = schedule.bundle_for_task(task_index).provenance(
                schedule.schedule_sha256
            )
            rows.append(
                {
                    **provenance,
                    "product_type": provenance["scheduled_primary_product_type"],
                    "rule_count": len(provenance["scheduled_rule_ids"]),
                    "rule_ids": json.dumps(provenance["scheduled_rule_ids"]),
                    "scheduled_rule_ids": json.dumps(
                        provenance["scheduled_rule_ids"]
                    ),
                    "scheduled_rule_profile_ids": json.dumps(
                        provenance["scheduled_rule_profile_ids"]
                    ),
                    "balanced_rule_schedule": True,
                    "rule_schedule_version": self.launcher.SCHEDULE_VERSION,
                    "profile_capacity_policy_version": (
                        self.launcher.EXPECTED_PROFILE_CAPACITY_POLICY_VERSION
                    ),
                    "profile_capacity_policy_sha256": (
                        self.launcher.EXPECTED_PROFILE_CAPACITY_POLICY_SHA256
                    ),
                }
            )
        planned = schedule.summary()
        realized = schedule.realized_summary(accepted)
        summary = {
            "balanced_rule_schedule": True,
            **planned,
            "planned_rule_schedule": planned,
            **realized,
            "realized_rule_schedule": realized,
            "count": 4,
            "generated_pairs": 3,
            "pending": 1,
            "errors": 1,
        }
        missing = schedule.bundle_for_task(1)
        errors = [
            {
                "task_index": 1,
                "source_id": missing.donor_id,
                "category": missing.category,
                "error": "RuntimeError('synthetic test failure')",
            }
        ]
        return schedule, self.launcher.pd.DataFrame(rows), summary, errors

    def test_custom_env_file_is_forwarded_to_both_child_commands(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stat-rule-launcher-test-") as raw:
            temp_root = Path(raw)
            env_file = temp_root / "custom.env"
            env_file.write_text("", encoding="utf-8")
            dataset_slug = "generated-pairs-test"
            label_source = "generated_test"
            frozen_dir = temp_root / "frozen"
            frozen_dir.mkdir(parents=True)
            freeze_manifest_path = frozen_dir / "freeze_manifest.json"
            freeze_manifest_path.write_text("{}", encoding="utf-8")
            freeze_manifest_sha256 = hashlib.sha256(b"{}").hexdigest()
            manifest_path = (
                temp_root
                / ".kaggle"
                / "datasets"
                / dataset_slug
                / "upload_manifest.json"
            )
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset": f"testowner/{dataset_slug}",
                        "pairs": 20,
                        "label_source": label_source,
                        "files": {
                            "freeze_manifest.json": {
                                "sha256": freeze_manifest_sha256,
                            }
                        },
                        "source_provenance": {
                            "attempt_diversity": {
                                "attempt_diversity_version": (
                                    self.launcher.ATTEMPT_DIVERSITY_VERSION
                                )
                            },
                            "frozen_attempt_diversity": {
                                "version": (
                                    self.launcher.FROZEN_ATTEMPT_DIVERSITY_VERSION
                                ),
                                "attempt_diversity_version": (
                                    self.launcher.ATTEMPT_DIVERSITY_VERSION
                                ),
                                "selected_task_count": 20,
                                "anchor_nonce_hash_valid_count": 20,
                                "anchor_nonce_hash_unique_count": 20,
                                "mutation_nonce_hash_valid_count": 20,
                                "mutation_nonce_hash_unique_count": 20,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                raw_dir=temp_root / "raw",
                frozen_dir=frozen_dir,
                pair_count=20,
                dataset_slug=dataset_slug,
                artifact_tag="test20",
                experiment_label="minilm-test20",
                kernel_slug="minilm-test20",
                title="MiniLM test20",
                notebook=temp_root / "test20.ipynb",
                env_file=env_file,
                expected_rule_catalog=temp_root / "rules.json",
                expected_source_items=temp_root / "items.parquet",
                expected_prompt=temp_root / "prompt.md",
                expected_model="test-model",
                expected_temperature=0.7,
                expected_semantic_signature_limit=2,
                expected_tier=None,
                label_source=label_source,
                minimum_two_rule_fraction=0.15,
                dry_run_only=True,
                finalize_local_output=False,
                local_output_dir=None,
            )

            with (
                mock.patch.object(self.launcher, "ROOT", temp_root),
                mock.patch.object(self.launcher, "parse_args", return_value=args),
                mock.patch.object(
                    self.launcher,
                    "require_complete_raw",
                    return_value={
                        "generated_pairs": 20,
                        "verified_realized_two_rule_fraction": 0.25,
                        "verified_catalog_main_rules": 2,
                        "verified_rule_schedule_statistics": {
                            "schedule_sha256": "a" * 64,
                        },
                    },
                ),
                mock.patch.object(
                    self.launcher,
                    "freeze",
                    return_value={
                        "summary": {
                            "generated_pairs": 20,
                            "frozen_attempt_diversity": {
                                "version": (
                                    self.launcher.FROZEN_ATTEMPT_DIVERSITY_VERSION
                                ),
                                "attempt_diversity_version": (
                                    self.launcher.ATTEMPT_DIVERSITY_VERSION
                                ),
                                "selected_task_count": 20,
                                "anchor_nonce_hash_valid_count": 20,
                                "anchor_nonce_hash_unique_count": 20,
                                "mutation_nonce_hash_valid_count": 20,
                                "mutation_nonce_hash_unique_count": 20,
                            },
                            "frozen_rule_schedule": {
                                "selected_task_count": 20,
                                "source_rule_schedule_sha256": "a" * 64,
                                "primary_rule_coverage": 2,
                                "primary_rule_profile_coverage": 2,
                                "full_primary_rule_coverage": True,
                                "full_primary_rule_profile_coverage": True,
                                "primary_rule_profile_cap_violations": {},
                                "semantic_signature_version": (
                                    self.launcher.SEMANTIC_SIGNATURE_VERSION
                                ),
                                "semantic_signature_limit": 2,
                                "semantic_signature_max_count": 2,
                                "profile_capacity_policy_version": (
                                    self.launcher.EXPECTED_PROFILE_CAPACITY_POLICY_VERSION
                                ),
                                "profile_capacity_policy_sha256": (
                                    self.launcher.EXPECTED_PROFILE_CAPACITY_POLICY_SHA256
                                ),
                                "two_rule_fraction": 0.25,
                            },
                        }
                    },
                ),
                mock.patch.object(
                    self.launcher,
                    "verify_frozen_global_card_uniqueness",
                    return_value={
                        "version": (
                            self.launcher.FROZEN_GLOBAL_CARD_UNIQUENESS_VERSION
                        )
                    },
                ),
                mock.patch.object(self.launcher.kaggle, "load_dotenv"),
                mock.patch.object(self.launcher, "run") as run,
                mock.patch.dict(
                    self.launcher.os.environ,
                    {"KAGGLE_USERNAME": "testowner"},
                    clear=False,
                ),
            ):
                self.launcher.main()

            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(len(commands), 3)
            upload_command, _, notebook_command = commands
            for command in (upload_command, notebook_command):
                self.assertEqual(command.count("--env-file"), 1)
                index = command.index("--env-file")
                self.assertEqual(command[index + 1], str(env_file))
            self.assertIn("scripts/push_generation_rule_pairs_dataset.py", upload_command)
            self.assertIn("scripts/run_kaggle_notebook.py", notebook_command)
            message_index = upload_command.index("--message")
            self.assertTrue(upload_command[message_index + 1].endswith("v3"))
            self.assertEqual(upload_command[-1], "--dry-run")
            self.assertEqual(notebook_command[-1], "--dry-run")

    def test_finalize_local_output_main_is_local_only_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stat-rule-local-finalize-") as raw:
            temp_root = Path(raw)
            fixture = self._local_finalization_fixture(temp_root)
            args = argparse.Namespace(
                raw_dir=temp_root / "unused-raw",
                frozen_dir=fixture["frozen_dir"],
                pair_count=fixture["pair_count"],
                dataset_slug=fixture["dataset_slug"],
                artifact_tag="unused",
                experiment_label=fixture["experiment_label"],
                kernel_slug=fixture["kernel_slug"],
                title="unused",
                notebook=temp_root / "unused.ipynb",
                env_file=temp_root / "unused.env",
                expected_rule_catalog=temp_root / "unused-rules.json",
                expected_source_items=temp_root / "unused-items.parquet",
                expected_prompt=temp_root / "unused-prompt.md",
                expected_model="unused-model",
                expected_temperature=0.7,
                expected_semantic_signature_limit=2,
                expected_tier=None,
                label_source=fixture["label_source"],
                minimum_two_rule_fraction=0.075,
                dry_run_only=False,
                finalize_local_output=True,
                local_output_dir=fixture["output_dir"],
            )
            with (
                mock.patch.object(self.launcher, "ROOT", temp_root),
                mock.patch.object(self.launcher, "parse_args", return_value=args),
                mock.patch.object(self.launcher, "run") as run,
                mock.patch.object(self.launcher, "require_complete_raw") as preflight,
                mock.patch.object(self.launcher, "freeze") as freeze,
                mock.patch.object(self.launcher.kaggle, "load_dotenv") as load_dotenv,
            ):
                self.launcher.main()

            run.assert_not_called()
            preflight.assert_not_called()
            freeze.assert_not_called()
            load_dotenv.assert_not_called()
            report_path = (
                temp_root
                / "reports"
                / f"{fixture['experiment_label']}_launcher.json"
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "complete")
            self.assertTrue(report["finalized_from_local_output"])
            self.assertEqual(report["run_id"], fixture["run_id"])
            self.assertEqual(
                report["local_artifact_validation"]["upload_payload_files_verified"],
                2,
            )

    def test_finalize_local_output_rejects_tampered_payload_and_run_id(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stat-rule-local-tamper-") as raw:
            temp_root = Path(raw)
            fixture = self._local_finalization_fixture(temp_root)
            kwargs = {
                "output_dir": fixture["output_dir"],
                "upload_manifest_path": fixture["upload_manifest_path"],
                "frozen_dir": fixture["frozen_dir"],
                "dataset_ref": fixture["dataset_ref"],
                "pair_count": fixture["pair_count"],
                "label_source": fixture["label_source"],
                "experiment_label": fixture["experiment_label"],
                "kernel_slug": fixture["kernel_slug"],
            }
            result = self.launcher.finalize_local_output(**kwargs)
            self.assertEqual(result["status"], "complete")

            sync_path = Path(fixture["output_dir"]) / "google_sheets_sync.json"
            sync = json.loads(sync_path.read_text(encoding="utf-8"))
            sync["run_id"] = "different-run"
            sync_path.write_text(json.dumps(sync), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "sync run IDs differ"):
                self.launcher.finalize_local_output(**kwargs)

            sync["run_id"] = fixture["run_id"]
            sync_path.write_text(json.dumps(sync), encoding="utf-8")
            validation_path = Path(fixture["upload_manifest_path"]).parent / (
                "validation_report.json"
            )
            validation_path.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "payload file differs"):
                self.launcher.finalize_local_output(**kwargs)

    def test_raw_preflight_requires_generation_v3_and_duplicate_retry(self) -> None:
        base_summary = {
            "generated_pairs": 2,
            "pending": 0,
            "count": 2,
            "scheduled_tasks": 2,
            "validation_valid": True,
            "model": "test-model",
            "structured_output": False,
            "version": "rule_first_pair_generation_v3",
            "attempt_diversity_version": (
                self.launcher.ATTEMPT_DIVERSITY_VERSION
            ),
            "global_duplicate_card_retry": True,
            "semantic_signature_retry": True,
            "semantic_signature_version": self.launcher.SEMANTIC_SIGNATURE_VERSION,
            "semantic_signature_limit": 2,
            "temperature": 0.7,
        }
        cases = (
            (
                {"version": "rule_first_pair_generation_v2"},
                "unexpected pair-generation version",
            ),
            (
                {"attempt_diversity_version": "legacy"},
                "unexpected attempt-diversity version",
            ),
            (
                {"global_duplicate_card_retry": False},
                "lacks global duplicate-card retry",
            ),
            (
                {"semantic_signature_retry": False},
                "lacks semantic-signature retry",
            ),
            (
                {"semantic_signature_version": "legacy"},
                "unexpected semantic-signature version",
            ),
            (
                {"semantic_signature_limit": 3},
                "unexpected semantic-signature limit",
            ),
            (
                {"temperature": 0.25},
                "unexpected Qwen temperature",
            ),
        )
        with tempfile.TemporaryDirectory(prefix="stat-rule-preflight-test-") as raw:
            raw_dir = Path(raw)
            summary_path = raw_dir / "summary.json"
            for changes, expected_error in cases:
                with self.subTest(changes=changes):
                    summary_path.write_text(
                        json.dumps({**base_summary, **changes}), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(RuntimeError, expected_error):
                        self.launcher.require_complete_raw(
                            raw_dir,
                            2,
                            expected_source_items=raw_dir / "items.parquet",
                            expected_rule_catalog=raw_dir / "rules.json",
                            expected_prompt=raw_dir / "prompt.md",
                            expected_model="test-model",
                            expected_temperature=0.7,
                            expected_semantic_signature_limit=2,
                            expected_tiers={"STAT_TEST"},
                            minimum_two_rule_fraction=0.1,
                        )

    def test_semantic_signature_metadata_guard_recomputes_and_caps(self) -> None:
        applications = [{
            "concept": "color",
            "attribute_key": "Цвет товара",
            "original_value": "красный",
            "new_value": "синий",
        }]
        signature = self.launcher._semantic_pair_signature(
            "Электроника", "чехол", applications
        )
        metadata = self.launcher.pd.DataFrame([
            {
                "category": "Электроника",
                "product_type": "чехол",
                "applications_json": json.dumps(applications, ensure_ascii=False),
                "semantic_signature": signature,
                "semantic_signature_version": (
                    self.launcher.SEMANTIC_SIGNATURE_VERSION
                ),
            }
            for _ in range(2)
        ])
        summary = {
            "semantic_signature_unique_count": 1,
            "semantic_signature_max_count": 2,
        }
        self.assertEqual(
            self.launcher.verify_semantic_signature_metadata(metadata, summary, 2),
            {"unique_count": 1, "max_count": 2},
        )
        with self.assertRaisesRegex(RuntimeError, "exceeds semantic-signature limit"):
            self.launcher.verify_semantic_signature_metadata(
                self.launcher.pd.concat([metadata, metadata.iloc[[0]]]),
                {**summary, "semantic_signature_max_count": 3},
                2,
            )
        tampered = metadata.copy()
        tampered.loc[0, "semantic_signature"] = "not-the-derived-signature"
        with self.assertRaisesRegex(RuntimeError, "does not match applications"):
            self.launcher.verify_semantic_signature_metadata(tampered, summary, 2)

    def test_balanced_schedule_guard_recomputes_sha_and_usage(self) -> None:
        rows = [
            {
                "task_index": 0,
                "source_style_id": 101,
                "category": "a",
                "rule_count": 1,
                "rule_ids": json.dumps(["r0"]),
                "balanced_rule_schedule": True,
                "rule_schedule_version": self.launcher.SCHEDULE_VERSION,
                "scheduled_primary_rule_id": "r0",
                "scheduled_primary_profile_id": "p0",
                "scheduled_primary_task_cap": None,
                "scheduled_secondary_rule_id": None,
                "scheduled_secondary_profile_id": None,
                "scheduled_rule_ids": json.dumps(["r0"]),
                "scheduled_rule_profile_ids": json.dumps(["p0"]),
                "profile_capacity_policy_version": (
                    self.launcher.EXPECTED_PROFILE_CAPACITY_POLICY_VERSION
                ),
                "profile_capacity_policy_sha256": (
                    self.launcher.EXPECTED_PROFILE_CAPACITY_POLICY_SHA256
                ),
            },
            {
                "task_index": 1,
                "source_style_id": 102,
                "category": "a",
                "rule_count": 2,
                "rule_ids": json.dumps(["r1", "r0"]),
                "balanced_rule_schedule": True,
                "rule_schedule_version": self.launcher.SCHEDULE_VERSION,
                "scheduled_primary_rule_id": "r1",
                "scheduled_primary_profile_id": "p1",
                "scheduled_primary_task_cap": None,
                "scheduled_secondary_rule_id": "r0",
                "scheduled_secondary_profile_id": "p0",
                "scheduled_rule_ids": json.dumps(["r1", "r0"]),
                "scheduled_rule_profile_ids": json.dumps(["p1", "p0"]),
                "profile_capacity_policy_version": (
                    self.launcher.EXPECTED_PROFILE_CAPACITY_POLICY_VERSION
                ),
                "profile_capacity_policy_sha256": (
                    self.launcher.EXPECTED_PROFILE_CAPACITY_POLICY_SHA256
                ),
            },
        ]
        payload = [
            {
                "task_index": row["task_index"],
                "donor_id": row["source_style_id"],
                "category": row["category"],
                "profiles": json.loads(row["scheduled_rule_profile_ids"]),
            }
            for row in rows
        ]
        schedule_sha256 = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for row in rows:
            row["rule_schedule_sha256"] = schedule_sha256
        planned = {
            "rule_schedule_version": self.launcher.SCHEDULE_VERSION,
            "rule_schedule_sha256": schedule_sha256,
            "scheduled_tasks": 2,
            "eligible_rules": 2,
            "eligible_rule_profiles": 2,
            "primary_rule_coverage": 2,
            "primary_rule_profile_coverage": 2,
            "category_task_quotas": {"a": 2},
            "primary_rule_usage": {"r0": 1, "r1": 1},
            "primary_rule_profile_usage": {"p0": 1, "p1": 1},
            "secondary_rule_usage": {"r0": 1, "r1": 0},
            "secondary_rule_profile_usage": {"p0": 1, "p1": 0},
            "total_rule_usage": {"r0": 2, "r1": 1},
            "total_rule_profile_usage": {"p0": 2, "p1": 1},
            "profile_primary_task_caps": {},
            "capacity_saturated_single_rule_profiles": [],
            "profile_capacity_policy_version": (
                self.launcher.EXPECTED_PROFILE_CAPACITY_POLICY_VERSION
            ),
            "profile_capacity_policy_sha256": (
                self.launcher.EXPECTED_PROFILE_CAPACITY_POLICY_SHA256
            ),
            "primary_rule_profile_usage_min": 1,
            "primary_rule_profile_usage_max": 1,
            "primary_rule_profile_usage_skew": 0,
            "total_rule_profile_usage_min": 1,
            "total_rule_profile_usage_max": 2,
            "total_rule_profile_usage_skew": 1,
            "balanced_total_rule_profile_usage_min": 1,
            "balanced_total_rule_profile_usage_max": 2,
            "balanced_total_rule_profile_usage_skew": 1,
            "scheduled_two_rule_tasks": 1,
            "scheduled_two_rule_fraction": 0.5,
        }
        realized = {
            "completed_scheduled_tasks": 2,
            "pending_scheduled_tasks": 0,
            "realized_primary_rule_coverage": 2,
            "realized_primary_rule_profile_coverage": 2,
            "realized_category_task_counts": {"a": 2},
            "realized_scheduled_two_rule_tasks": 1,
        }
        summary = {
            "balanced_rule_schedule": True,
            **planned,
            "planned_rule_schedule": planned,
            **realized,
            "realized_rule_schedule": realized,
        }
        metadata = self.launcher.pd.DataFrame(rows)
        verified = self.launcher.verify_balanced_rule_schedule_metadata(
            metadata, summary, expected_rule_count=2
        )
        self.assertEqual(verified["schedule_sha256"], schedule_sha256)
        self.assertEqual(verified["primary_rule_coverage"], 2)
        self.assertEqual(verified["two_rule_fraction"], 0.5)

        tampered = metadata.copy()
        tampered.loc[0, "source_style_id"] = 999
        with self.assertRaisesRegex(RuntimeError, "does not reproduce"):
            self.launcher.verify_balanced_rule_schedule_metadata(
                tampered, summary, expected_rule_count=2
            )

    def test_partial_ready_contract_accepts_exact_gaps_and_rejects_tampering(
        self,
    ) -> None:
        schedule, metadata, summary, errors = self._partial_schedule_fixture()
        verified = self.launcher.verify_rebuilt_balanced_rule_schedule_metadata(
            metadata, summary, expected_rule_count=2, expected_schedule=schedule
        )
        self.assertTrue(verified["partial_ready"])
        self.assertEqual(verified["completed_scheduled_tasks"], 3)
        self.assertEqual(verified["pending_scheduled_tasks"], 1)

        with tempfile.TemporaryDirectory(
            prefix="stat-rule-partial-ready-test-"
        ) as raw:
            errors_path = Path(raw) / "errors.json"
            errors_path.write_text(json.dumps(errors), encoding="utf-8")
            pending = self.launcher.verify_pending_task_errors(
                errors_path, metadata, summary, schedule
            )
            self.assertEqual(pending["pending_task_indices"], [1])

            tampered_errors = [{**errors[0], "task_index": 0}]
            errors_path.write_text(
                json.dumps(tampered_errors), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                RuntimeError, "rebuilt schedule|exact complement"
            ):
                self.launcher.verify_pending_task_errors(
                    errors_path, metadata, summary, schedule
                )

            errors_path.write_text(json.dumps(errors), encoding="utf-8")
            extra_gap = metadata[metadata["task_index"].ne(2)].copy()
            with self.assertRaisesRegex(RuntimeError, "pending/error counts"):
                self.launcher.verify_pending_task_errors(
                    errors_path, extra_gap, summary, schedule
                )

        tampered_schedule = metadata.copy()
        tampered_schedule.loc[
            tampered_schedule["task_index"].eq(0), "source_style_id"
        ] = 999
        with self.assertRaisesRegex(RuntimeError, "rebuilt field source_style_id"):
            self.launcher.verify_rebuilt_balanced_rule_schedule_metadata(
                tampered_schedule,
                summary,
                expected_rule_count=2,
                expected_schedule=schedule,
            )

        tampered_enabled = metadata.astype(
            {"balanced_rule_schedule": "object"}
        ).copy()
        tampered_enabled.loc[
            tampered_enabled["task_index"].eq(0), "balanced_rule_schedule"
        ] = "False"
        with self.assertRaisesRegex(RuntimeError, "disables balanced"):
            self.launcher.verify_rebuilt_balanced_rule_schedule_metadata(
                tampered_enabled,
                summary,
                expected_rule_count=2,
                expected_schedule=schedule,
            )

    def test_attempt_provenance_aggregates_resume_offsets_and_configs(self) -> None:
        metadata = self.launcher.pd.DataFrame(
            {
                "task_seed_offset": [0, 100],
                "task_retry_round": [0, 1],
                "selection_attempt": [1, 2],
                "pair_attempts_config": [3, 3],
                "anchor_attempts_config": [2, 2],
                "mutation_attempts_config": [4, 4],
                "task_retries_config": [1, 5],
            }
        )
        summary = {
            "pair_attempts": 3,
            "anchor_attempts": 2,
            "mutation_attempts": 4,
            "realized_task_seed_offsets": [0, 100],
            "realized_task_seed_offset_distribution": {"0": 1, "100": 1},
            "realized_task_retry_round_distribution": {"0": 1, "1": 1},
            "realized_selection_attempt_distribution": {"1": 1, "2": 1},
            "realized_pair_attempts_config_distribution": {"3": 2},
            "realized_anchor_attempts_config_distribution": {"2": 2},
            "realized_mutation_attempts_config_distribution": {"4": 2},
            "realized_task_retries_config_distribution": {"1": 1, "5": 1},
        }
        verified = self.launcher.verify_attempt_provenance_metadata(
            metadata, summary
        )
        self.assertEqual(verified["task_seed_offsets"], [0, 100])
        with self.assertRaisesRegex(RuntimeError, "task seed offsets"):
            self.launcher.verify_attempt_provenance_metadata(
                metadata, {**summary, "realized_task_seed_offsets": [100]}
            )

    def test_attempt_diversity_guard_recomputes_nonces_and_rejects_mix(self) -> None:
        run_seed = 17
        rows = []
        for task_index, source_id in enumerate((101, 102)):
            task = PairGenerationTask(
                task_index=task_index,
                mutated_id=-task_index - 1,
                seed=int(stable_hash64(run_seed, source_id) % (2**31 - 1)),
                anchor={"id": source_id},
            )
            common = {
                "task_seed_offset": 0,
                "task_retry_round": 0,
                "selection_attempt": 1,
            }
            rows.append(
                {
                    "task_index": task_index,
                    "id2": -task_index - 1,
                    "source_style_id": source_id,
                    "attempt_diversity_version": (
                        self.launcher.ATTEMPT_DIVERSITY_VERSION
                    ),
                    **common,
                    "anchor_attempt": 1,
                    "mutation_attempt": 1,
                    "pair_attempts_config": 2,
                    "anchor_attempts_config": 2,
                    "mutation_attempts_config": 2,
                    "anchor_diversity_nonce_sha256": hashlib.sha256(
                        _attempt_diversity_nonce(
                            task,
                            **common,
                            stage="anchor",
                            stage_attempt=1,
                        ).encode("utf-8")
                    ).hexdigest(),
                    "mutation_diversity_nonce_sha256": hashlib.sha256(
                        _attempt_diversity_nonce(
                            task,
                            **common,
                            stage="mutation",
                            stage_attempt=1,
                        ).encode("utf-8")
                    ).hexdigest(),
                    "global_rejection_feedback_count": task_index,
                    "forbidden_semantic_signature_count": task_index,
                    "forbidden_card_key_count": task_index * 2,
                }
            )
        metadata = self.launcher.pd.DataFrame(rows)
        summary = {
            "seed": run_seed,
            "attempt_diversity_version": (
                self.launcher.ATTEMPT_DIVERSITY_VERSION
            ),
        }
        verified = self.launcher.verify_attempt_diversity_metadata(
            metadata, summary
        )
        self.assertEqual(verified["anchor_nonce_hash_unique_count"], 2)
        self.assertEqual(verified["mutation_nonce_hash_unique_count"], 2)

        mixed = metadata.copy()
        mixed.loc[0, "attempt_diversity_version"] = "legacy"
        with self.assertRaisesRegex(RuntimeError, "mixes or uses unexpected"):
            self.launcher.verify_attempt_diversity_metadata(mixed, summary)
        tampered = metadata.copy()
        tampered.loc[0, "anchor_diversity_nonce_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "nonce hash mismatch"):
            self.launcher.verify_attempt_diversity_metadata(tampered, summary)

    def test_frozen_schedule_guard_requires_subset_coverage_and_caps(self) -> None:
        schedule_sha = "b" * 64
        raw = {
            "verified_catalog_main_rules": 2,
            "verified_rule_schedule_statistics": {"schedule_sha256": schedule_sha},
        }
        frozen_schedule = {
            "selected_task_count": 5_000,
            "source_rule_schedule_sha256": schedule_sha,
            "primary_rule_coverage": 2,
            "primary_rule_profile_coverage": 2,
            "full_primary_rule_coverage": True,
            "full_primary_rule_profile_coverage": True,
            "primary_rule_profile_cap_violations": {},
            "semantic_signature_version": self.launcher.SEMANTIC_SIGNATURE_VERSION,
            "semantic_signature_limit": 2,
            "semantic_signature_max_count": 2,
            "profile_capacity_policy_version": (
                self.launcher.EXPECTED_PROFILE_CAPACITY_POLICY_VERSION
            ),
            "profile_capacity_policy_sha256": (
                self.launcher.EXPECTED_PROFILE_CAPACITY_POLICY_SHA256
            ),
            "two_rule_fraction": 0.08,
        }
        verified = self.launcher.verify_frozen_rule_schedule(
            {"summary": {"frozen_rule_schedule": frozen_schedule}},
            raw,
            pair_count=5_000,
            expected_semantic_signature_limit=2,
            minimum_two_rule_fraction=0.075,
        )
        self.assertEqual(verified, frozen_schedule)
        with self.assertRaisesRegex(RuntimeError, "capacity_violations"):
            self.launcher.verify_frozen_rule_schedule(
                {
                    "summary": {
                        "frozen_rule_schedule": {
                            **frozen_schedule,
                            "primary_rule_profile_cap_violations": {"p": 1},
                        }
                    }
                },
                raw,
                pair_count=5_000,
                expected_semantic_signature_limit=2,
                minimum_two_rule_fraction=0.075,
            )

    def test_frozen_global_card_guard_is_category_agnostic(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="stat-rule-global-card-test-"
        ) as raw:
            root = Path(raw)
            raw_dir, frozen_dir = root / "raw", root / "frozen"
            raw_dir.mkdir()
            frozen_dir.mkdir()
            unique_items = pd.DataFrame(
                [
                    {
                        "id": 1,
                        "name": "чехол красный",
                        "attributes": json.dumps(
                            {"Тип": "чехол", "Цвет": "красный"},
                            ensure_ascii=False,
                        ),
                        "category": "Электроника",
                    },
                    {
                        "id": -1,
                        "name": "чехол синий",
                        "attributes": json.dumps(
                            {"Тип": "чехол", "Цвет": "синий"},
                            ensure_ascii=False,
                        ),
                        "category": "Электроника",
                    },
                ]
            )
            unique_items.to_parquet(raw_dir / "items.parquet", index=False)
            unique_items.to_parquet(frozen_dir / "items.parquet", index=False)
            (frozen_dir / "dropped_pairs.json").write_text(
                "[]", encoding="utf-8"
            )
            provenance = self.launcher.global_card_uniqueness_provenance(
                unique_items, unique_items, []
            )
            frozen = {
                "summary": {
                    "dropped_before_target": 0,
                    "frozen_global_card_uniqueness": provenance,
                },
                "manifest": {"frozen_global_card_uniqueness": provenance},
            }
            (frozen_dir / "summary.json").write_text(
                json.dumps(frozen["summary"]), encoding="utf-8"
            )
            (frozen_dir / "freeze_manifest.json").write_text(
                json.dumps(frozen["manifest"]), encoding="utf-8"
            )
            self.assertEqual(
                self.launcher.verify_frozen_global_card_uniqueness(
                    frozen,
                    raw_dir=raw_dir,
                    frozen_dir=frozen_dir,
                    pair_count=1,
                ),
                provenance,
            )

            duplicate_items = unique_items.copy()
            duplicate_items.loc[1, "id"] = 2
            duplicate_items.loc[1, "name"] = "ЧЕХОЛ КРАСНЫЙ"
            duplicate_items.loc[1, "attributes"] = json.dumps(
                {"цвет": "КРАСНЫЙ", "тип": "ЧЕХОЛ"},
                ensure_ascii=False,
            )
            duplicate_items.loc[1, "category"] = "Другая категория"
            duplicate_items.to_parquet(raw_dir / "items.parquet", index=False)
            duplicate_items.to_parquet(
                frozen_dir / "items.parquet", index=False
            )
            duplicate_provenance = (
                self.launcher.global_card_uniqueness_provenance(
                    duplicate_items, duplicate_items, []
                )
            )
            duplicate_frozen = {
                "summary": {
                    "dropped_before_target": 0,
                    "frozen_global_card_uniqueness": duplicate_provenance,
                },
                "manifest": {
                    "frozen_global_card_uniqueness": duplicate_provenance
                },
            }
            (frozen_dir / "summary.json").write_text(
                json.dumps(duplicate_frozen["summary"]), encoding="utf-8"
            )
            (frozen_dir / "freeze_manifest.json").write_text(
                json.dumps(duplicate_frozen["manifest"]), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                RuntimeError, "frozen global card uniqueness"
            ):
                self.launcher.verify_frozen_global_card_uniqueness(
                    duplicate_frozen,
                    raw_dir=raw_dir,
                    frozen_dir=frozen_dir,
                    pair_count=1,
                )

    def test_defaults_and_strict_catalog_contract_target_v3(self) -> None:
        expected_sha256 = (
            "017e8ced6035695474007d5cc91e72870d77e5bef6b2348cdd13bde6cbdfdc6c"
        )
        self.assertEqual(self.launcher.sha256_file(CATALOG_V3), expected_sha256)
        self.assertEqual(
            self.launcher.EXPECTED_RULE_CATALOG_SHA256, expected_sha256
        )
        self.assertEqual(self.launcher.DEFAULT_RULE_CATALOG, CATALOG_V3)
        expected_source_items = (
            ROOT / "item_pipeline" / "artifacts" / "generated" / "items.parquet"
        )
        expected_source_sha256 = (
            "54672a0241b9586563812246be77b24f976a253a9f4e732d65d2484496a13883"
        )
        self.assertEqual(self.launcher.DEFAULT_SOURCE_ITEMS, expected_source_items)
        self.assertEqual(
            self.launcher.EXPECTED_SOURCE_ITEMS_SHA256, expected_source_sha256
        )
        self.assertEqual(
            self.launcher.sha256_file(expected_source_items), expected_source_sha256
        )
        self.assertEqual(self.launcher.DEFAULT_TEMPERATURE, 0.7)
        self.assertEqual(self.launcher.DEFAULT_SEMANTIC_SIGNATURE_LIMIT, 2)
        with mock.patch.object(sys, "argv", ["launcher"]):
            self.assertAlmostEqual(
                self.launcher.parse_args().minimum_two_rule_fraction, 0.075
            )
        self.assertEqual(
            self.launcher.DEFAULT_RAW,
            ROOT
            / "item_pipeline"
            / "artifacts"
            / "rule_first_pairs_stat_p80_scoped_v3_diversity_normalized_v2_raw5200",
        )
        for value in (
            self.launcher.DEFAULT_FROZEN,
            self.launcher.DEFAULT_NOTEBOOK,
            self.launcher.DEFAULT_DATASET_SLUG,
            self.launcher.DEFAULT_ARTIFACT_TAG,
            self.launcher.DEFAULT_EXPERIMENT,
            self.launcher.DEFAULT_KERNEL_SLUG,
            self.launcher.DEFAULT_TITLE,
        ):
            self.assertIn("v3", str(value))
            self.assertNotIn("v2", str(value))

        manifest = json.loads(
            CATALOG_V3.with_suffix(".manifest.json").read_text(encoding="utf-8")
        )
        contract = self.launcher.require_statistical_catalog_contract(
            manifest, set(self.launcher.DEFAULT_TIERS)
        )
        self.assertEqual(contract["main_rule_count"], 100)
        self.assertEqual(
            contract["tier_counts"],
            {
                "STAT_LABEL0_CROSS_SPLIT_MIN2_P80_SCOPED": 46,
                "STAT_LABEL0_MIN2_P80_SCOPED": 54,
            },
        )
        rules = json.loads(CATALOG_V3.read_text(encoding="utf-8"))
        actual_tiers = collections.Counter(
            str(rule["generation_tier"])
            for rule in rules
            if rule["generation_tier"] in self.launcher.DEFAULT_TIERS
        )
        self.assertEqual(dict(actual_tiers), contract["tier_counts"])

        v2_manifest = json.loads(
            CATALOG_V3.with_name(
                "statistical_negative_rules_min2_p80_scoped_v2.manifest.json"
            ).read_text(encoding="utf-8")
        )
        with self.assertRaisesRegex(RuntimeError, "unexpected statistical catalog policy"):
            self.launcher.require_statistical_catalog_contract(
                v2_manifest, set(self.launcher.DEFAULT_TIERS)
            )


if __name__ == "__main__":
    unittest.main()
