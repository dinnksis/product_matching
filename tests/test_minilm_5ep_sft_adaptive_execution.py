from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import create_minilm_5ep_sft_hparam_notebooks as builder
import materialize_minilm_5ep_sft_loss_confirmation as adaptive
import run_minilm_5ep_sft_hparam_kaggle as launcher
import summarize_minilm_5ep_sft_hparams as summarizer
from tests import test_materialize_minilm_5ep_sft_loss_confirmation as fixtures


class AdaptiveExecutionIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = fixtures.LossConfirmationMaterializerTest("runTest")
        self.fixture.setUp()

    def tearDown(self) -> None:
        self.fixture.tearDown()

    @property
    def f(self) -> fixtures.LossConfirmationMaterializerTest:
        return self.fixture

    def _load(self, path: Path) -> dict:
        return builder.load_campaign_lock(
            path,
            plan=self.f.plan,
            base_config=self.f.base_config,
        )

    def _enrich_run(
        self,
        row: dict,
        entry: dict,
        *,
        full_parquets: bool = False,
    ) -> None:
        root = self.f.artifacts / row["kernel_slug"]
        model_dir = root / "model"
        run_id = row["run_id"]
        config = entry["expected_config"]
        split_rows = {"iid": 12_000, "hard": 5_814, "ood": 41_171}
        split_scores = {
            "iid": float(row["iid_macro_ap"]),
            "hard": float(self.f.plan["baseline_metrics"]["hard_macro_ap"]),
            "ood": float(self.f.plan["baseline_metrics"]["ood_macro_ap"]),
        }
        validation_splits = {
            split: {
                "examples": count,
                "macro_average_precision": split_scores[split],
                "overall_average_precision": split_scores[split],
                "recall_at_precision_0_99": 0.1,
                "roc_auc": 0.7,
                "log_loss": 0.5,
            }
            for split, count in split_rows.items()
        }
        report = {
            "args": deepcopy(config),
            "training_seconds": 1000.0,
            "total_pipeline_seconds": 1500.0,
            "peak_vram_gib_by_rank": [10.0, 10.0],
            "training_sampling": "none",
            "training_loss_weighting": "none",
            "training_subset": "all",
            "original_training_examples": 306_669,
            "training_unique_coverage_per_epoch": 1.0,
            "training_loss_weight_min": 1.0,
            "training_loss_weight_median": 1.0,
            "training_loss_weight_max": 1.0,
            "validation_splits": validation_splits,
        }
        baseline_metrics = self.f.plan["baseline_metrics"]
        comparison = {
            "status": "ready",
            "baseline_run_id": self.f.plan["baseline_run_id"],
            "candidate_run_id": run_id,
            "method": "paired_component_permutation",
            "splits": {
                split: {
                    "delta_macro_average_precision": split_scores[split]
                    - float(baseline_metrics[f"{split}_macro_ap"]),
                    "p_value": 0.5,
                    "p_value_holm": 1.0,
                    "ci95_low": -0.001,
                    "ci95_high": 0.001,
                }
                for split in split_rows
            },
        }
        completion_path = root / "notebook_completed.json"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion.update(
            {
                "experiment_group": "sft",
                "dataset_ref": launcher.VALIDATION_DATASET,
                "initial_checkpoint_ref": launcher.CHECKPOINT_DATASET,
                "initial_checkpoint_manifest_sha256": (
                    builder.team_builder.CHECKPOINT_MANIFEST_SHA256
                ),
                "training_report": report,
                "baseline_comparison": comparison,
            }
        )
        completion_path.write_text(json.dumps(completion), encoding="utf-8")
        (root / "baseline_comparison.json").write_text(
            json.dumps(comparison), encoding="utf-8"
        )
        (root / "experiment_run_id.txt").write_text(run_id + "\n", encoding="utf-8")
        (root / "google_sheets_sync.json").write_text(
            json.dumps(
                {
                    "status": "synced",
                    "run_id": run_id,
                    "experiment_group": "sft",
                    "comparison_sheet": "sft_exps",
                    "spreadsheet_id": launcher.qwen_builder.EXPERIMENT_SPREADSHEET_ID,
                }
            ),
            encoding="utf-8",
        )
        (model_dir / "training_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        (root / "cross_encoder_config.json").write_text(
            (model_dir / "training_config.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        selected_splits = split_rows if full_parquets else {"iid": 4}
        for split, count in selected_splits.items():
            pd.DataFrame(
                {
                    "id1": range(count),
                    "id2": range(count, count * 2),
                    "target": [index % 2 for index in range(count)],
                    "score": [0.1 + 0.8 * (index % 2) for index in range(count)],
                    "category": ["test"] * count,
                }
            ).to_parquet(
                model_dir / f"{split}_validation_predictions.parquet",
                index=False,
            )
        row["iid_predictions_sha256"] = adaptive.file_sha256(
            model_dir / "iid_validation_predictions.parquet"
        )

    def _executed_primary(self) -> tuple[dict, dict, dict[str, dict]]:
        primary = self.f._primary()
        summary, rows = self.f._primary_summary(primary)
        loaded = self._load(self.f.locks_dir / "primary.lock.json")
        entries = {
            entry["experiment"]: entry
            for entry in launcher.campaign_variants(
                self.f.plan,
                stage=None,
                only=None,
                stage_lock=loaded,
            )
        }
        for row in rows.values():
            self._enrich_run(row, entries[row["experiment"]])
        return primary, summary, rows

    def test_primary_dispatch_contract_generator_and_metadata(self) -> None:
        primary = self.f._primary()
        path = self.f.locks_dir / "primary.lock.json"
        loaded = self._load(path)
        contract = builder.normalized_campaign_execution_contract(
            self.f.plan,
            loaded,
            base_config=self.f.base_config,
        )
        self.assertEqual(contract["execution_status"], "runnable")
        self.assertEqual(len(contract["variants"]), 4)
        self.assertEqual(contract["hypothesis_family_size"], 5)
        first = contract["variants"][0]
        self.assertEqual(
            first["expected_notes"],
            adaptive.expected_variant_notes(primary, first["variant"]),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            built = builder.build_campaign(
                plan_path=self.f.plan_path,
                output_dir=Path(temp_dir),
                only={first["experiment"]},
                stage_lock_path=path,
            )
            self.assertEqual(len(built), 1)
            notebook = nbformat.read(built[0]["notebook"], as_version=4)
            metadata = notebook.metadata["product_matching_training"]["stage_lock"]
            self.assertEqual(metadata["schema_version"], 2)
            self.assertEqual(metadata["mode"], "loss_primary")
            self.assertEqual(metadata["family"]["maximum_hypotheses"], 5)
            self.assertEqual(
                metadata["parent"], first["parent_provenance"]
            )

    def test_generator_rejects_current_embedded_source_drift_before_write(self) -> None:
        self.f._primary()
        drifted_source_sha256 = "f" * 64
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with mock.patch.object(
                builder.baseline_builder,
                "embedded_sources",
                return_value=({}, drifted_source_sha256),
            ):
                with self.assertRaisesRegex(
                    builder.CampaignConfigError, "current embedded source bundle"
                ):
                    builder.build_campaign(
                        plan_path=self.f.plan_path,
                        output_dir=output_dir,
                        stage_lock_path=self.f.locks_dir / "primary.lock.json",
                    )
            self.assertEqual(list(output_dir.glob("*.ipynb")), [])

    def test_launcher_source_drift_fails_before_env_build_or_kaggle(self) -> None:
        primary = self.f._primary()
        experiment = primary["resolved_stage"]["variants"][0]["experiment"]
        argv = [
            "run_minilm_5ep_sft_hparam_kaggle.py",
            "--plan",
            str(self.f.plan_path),
            "--stage-lock",
            str(self.f.locks_dir / "primary.lock.json"),
            "--only",
            experiment,
            "--submit",
            "--wait",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            builder.baseline_builder,
            "embedded_sources",
            return_value=({}, "f" * 64),
        ), mock.patch.object(
            builder, "build_campaign"
        ) as build_campaign, mock.patch.object(
            launcher.kaggle, "load_dotenv"
        ) as load_env, mock.patch.object(
            launcher.kaggle, "kaggle_command"
        ) as kaggle_command, mock.patch.object(
            launcher.subprocess, "run"
        ) as subprocess_run:
            with self.assertRaisesRegex(
                builder.CampaignConfigError, "current embedded source bundle"
            ):
                launcher.main()
        build_campaign.assert_not_called()
        load_env.assert_not_called()
        kaggle_command.assert_not_called()
        subprocess_run.assert_not_called()

    def test_launcher_rejects_drifted_built_identity_before_env_or_kaggle(self) -> None:
        self.f._primary()
        lock_path = self.f.locks_dir / "primary.lock.json"
        loaded = self._load(lock_path)
        expected = launcher.campaign_variants(
            self.f.plan,
            stage=None,
            only=None,
            stage_lock=loaded,
        )[0]
        built = {
            key: deepcopy(expected[key])
            for key in launcher.LOCKED_BUILD_IDENTITY_FIELDS
        }
        built["source_sha256"] = "f" * 64
        argv = [
            "run_minilm_5ep_sft_hparam_kaggle.py",
            "--plan",
            str(self.f.plan_path),
            "--stage-lock",
            str(lock_path),
            "--only",
            expected["experiment"],
            "--submit",
            "--wait",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            builder, "build_campaign", return_value=[built]
        ), mock.patch.object(
            launcher.kaggle, "load_dotenv"
        ) as load_env, mock.patch.object(
            launcher.kaggle, "kaggle_command"
        ) as kaggle_command, mock.patch.object(
            launcher.subprocess, "run"
        ) as subprocess_run:
            with self.assertRaisesRegex(
                builder.CampaignConfigError, "immutable locked campaign"
            ):
                launcher.main()
        load_env.assert_not_called()
        kaggle_command.assert_not_called()
        subprocess_run.assert_not_called()

    def test_locked_build_identity_checks_every_immutable_field(self) -> None:
        self.f._primary()
        loaded = self._load(self.f.locks_dir / "primary.lock.json")
        expected = launcher.campaign_variants(
            self.f.plan,
            stage=None,
            only=None,
            stage_lock=loaded,
        )[0]
        reference = {
            key: deepcopy(expected[key])
            for key in launcher.LOCKED_BUILD_IDENTITY_FIELDS
        }
        launcher.validate_locked_build_identity(expected, reference)
        for field in launcher.LOCKED_BUILD_IDENTITY_FIELDS:
            with self.subTest(field=field):
                changed = deepcopy(reference)
                value = changed[field]
                if isinstance(value, bool):
                    changed[field] = not value
                elif isinstance(value, int):
                    changed[field] = value + 1
                elif isinstance(value, dict):
                    changed[field] = {**value, "tampered": True}
                elif value is None:
                    changed[field] = "tampered"
                else:
                    changed[field] = f"{value}-tampered"
                with self.assertRaises(builder.CampaignConfigError):
                    launcher.validate_locked_build_identity(expected, changed)

    def test_skipped_overlay_and_lr_are_noop_execution_receipts(self) -> None:
        metrics = {
            "balanced_binary_bce": 0.799,
            "balanced_category_class_sqrt_bce": 0.798,
            "balanced_category_class_bce": 0.797,
            "focal_bce_gamma2_scale4": 0.799,
        }
        primary = self.f._primary()
        primary_summary, _ = self.f._primary_summary(primary, metrics=metrics)
        overlay = self.f._overlay(primary, primary_summary)
        self.assertEqual(overlay["execution_status"], "skipped")
        primary_final, _ = self.f._primary_final_summary(
            primary, overlay, primary_summary
        )
        refine = self.f._refine(primary, overlay, primary_final)
        self.assertEqual(refine["execution_status"], "skipped")
        for name, lock in (("overlay.lock.json", overlay), ("lr.lock.json", refine)):
            with self.subTest(name=name):
                loaded = self._load(self.f.locks_dir / name)
                contract = builder.normalized_campaign_execution_contract(
                    self.f.plan, loaded, base_config=self.f.base_config
                )
                self.assertEqual(contract["execution_status"], "skipped")
                self.assertEqual(contract["variants"], [])
                self.assertEqual(
                    launcher.campaign_variants(
                        self.f.plan,
                        stage=None,
                        only=None,
                        stage_lock=loaded,
                    ),
                    [],
                )
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(
                builder.build_campaign(
                    plan_path=self.f.plan_path,
                    output_dir=Path(temp_dir),
                    source_dir=self.f.root / "intentionally-missing-source",
                    stage_lock_path=self.f.locks_dir / "overlay.lock.json",
                ),
                [],
            )

    def test_skipped_launcher_main_performs_zero_kaggle_actions(self) -> None:
        metrics = {
            "balanced_binary_bce": 0.799,
            "balanced_category_class_sqrt_bce": 0.798,
            "balanced_category_class_bce": 0.797,
            "focal_bce_gamma2_scale4": 0.799,
        }
        primary = self.f._primary()
        primary_summary, _ = self.f._primary_summary(primary, metrics=metrics)
        self.f._overlay(primary, primary_summary)
        argv = [
            "run_minilm_5ep_sft_hparam_kaggle.py",
            "--plan",
            str(self.f.plan_path),
            "--stage-lock",
            str(self.f.locks_dir / "overlay.lock.json"),
            "--submit",
            "--wait",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            launcher.kaggle, "load_dotenv"
        ) as load_env, mock.patch.object(
            launcher.kaggle, "kaggle_command"
        ) as kaggle_command, mock.patch.object(
            launcher.subprocess, "run"
        ) as subprocess_run:
            launcher.main()
        load_env.assert_not_called()
        kaggle_command.assert_not_called()
        subprocess_run.assert_not_called()

    def test_launcher_rejects_output_root_outside_trusted_authority(self) -> None:
        self.f._primary()
        argv = [
            "run_minilm_5ep_sft_hparam_kaggle.py",
            "--plan",
            str(self.f.plan_path),
            "--stage-lock",
            str(self.f.locks_dir / "primary.lock.json"),
            "--submit",
            "--wait",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(
            os.environ,
            {"KAGGLE_OUTPUT_DIR": str(self.f.root / "wrong-artifacts")},
        ), mock.patch.object(
            launcher.kaggle, "load_dotenv"
        ), mock.patch.object(
            launcher.kaggle, "kaggle_command"
        ) as kaggle_command:
            with self.assertRaisesRegex(RuntimeError, "trusted artifacts authority"):
                launcher.main()
        kaggle_command.assert_not_called()

    def test_runnable_overlay_and_lr_contracts(self) -> None:
        chain = self.f._full_chain()
        for name, expected_count, expected_family in (
            ("overlay.lock.json", 1, 5),
            ("lr.lock.json", 2, 2),
        ):
            with self.subTest(name=name):
                lock = self._load(self.f.locks_dir / name)
                contract = builder.normalized_campaign_execution_contract(
                    self.f.plan, lock, base_config=self.f.base_config
                )
                self.assertEqual(contract["execution_status"], "runnable")
                self.assertEqual(len(contract["variants"]), expected_count)
                self.assertEqual(
                    contract["hypothesis_family_size"], expected_family
                )
        self.assertEqual(chain["overlay"]["execution_status"], "runnable")
        self.assertEqual(chain["refine"]["execution_status"], "runnable")

    def _confirmation_rows(self, lock: dict) -> list[dict]:
        origin_by_id = {
            origin["origin_id"]: origin for origin in lock["origins"]
        }
        variants_by_group: dict[str, list[dict]] = {}
        for variant in lock["resolved_stage"]["variants"]:
            variants_by_group.setdefault(variant["recipe_group_id"], []).append(
                variant
            )
        rows = []
        for group in lock["resolved_stage"]["recipe_groups"]:
            roles = group["roles"]
            if "current_protocol_baseline_recipe" in roles:
                score = 0.790
            elif "selected_regularized_bce_recipe" in roles:
                score = 0.7921
            else:
                score = 0.796
            origin = origin_by_id[group["origin_seed42_id"]]
            rows.append(
                {
                    "experiment": origin["experiment"],
                    "run_id": origin["run_id"],
                    "seed": 42,
                    "iid_macro_ap": score,
                    "completed": True,
                }
            )
            for variant in variants_by_group[group["recipe_group_id"]]:
                rows.append(
                    {
                        "experiment": variant["experiment"],
                        "run_id": f"run-{variant['experiment']}",
                        "seed": variant["seed"],
                        "iid_macro_ap": score,
                        "completed": True,
                    }
                )
        return rows

    def _runtime_check(
        self,
        lock: dict,
        path: Path,
        *,
        selected_recipe_group_id: str | None = None,
    ) -> Path:
        selected_recipe_group_id = selected_recipe_group_id or next(
            group["recipe_group_id"]
            for group in lock["resolved_stage"]["recipe_groups"]
            if "loss_finalist_1" in group["roles"]
        )
        selected_group = next(
            group
            for group in lock["resolved_stage"]["recipe_groups"]
            if group["recipe_group_id"] == selected_recipe_group_id
        )
        payload = {
            "schema_version": 1,
            "campaign": self.f.plan["campaign"],
            "confirmation_lock_payload_sha256": lock["lock_payload_sha256"],
            "status": "passed",
            "selected_recipe_group_id": selected_recipe_group_id,
            "checked_recipe_family_sha256": selected_group[
                "recipe_family_sha256"
            ],
            "check_seconds": 40.0,
            "public_seconds": 250.0,
            "private_seconds": 600.0,
        }
        payload["runtime_check_payload_sha256"] = adaptive.canonical_sha256(
            payload
        )
        path.write_text(adaptive.canonical_json_dumps(payload) + "\n", encoding="utf-8")
        return path

    def test_confirmation_six_and_eight_run_contract_acceptance_tie_runtime(self) -> None:
        cases = (
            (None, 8),
            (
                {
                    "balanced_binary_bce": 0.805,
                    "balanced_category_class_sqrt_bce": 0.790,
                    "balanced_category_class_bce": 0.789,
                    "focal_bce_gamma2_scale4": 0.790,
                },
                6,
            ),
        )
        for index, (metrics, expected) in enumerate(cases):
            with self.subTest(expected=expected):
                if index:
                    self.tearDown()
                    self.setUp()
                chain = self.f._full_chain(metrics=metrics)
                confirmation = self.f._confirmation(chain)
                self.assertEqual(
                    len(confirmation["resolved_stage"]["variants"]), expected
                )
                loaded = self._load(self.f.locks_dir / "confirmation.lock.json")
                contract = builder.normalized_campaign_execution_contract(
                    self.f.plan, loaded, base_config=self.f.base_config
                )
                self.assertEqual(len(contract["variants"]), expected)
                runtime_path = self._runtime_check(
                    confirmation, self.f.root / "runtime-check.json"
                )
                with mock.patch.object(
                    summarizer, "verify_confirmation_iid_predictions"
                ):
                    projection = summarizer.adaptive_confirmation_projection(
                        plan=self.f.plan,
                        lock=confirmation,
                        rows=self._confirmation_rows(confirmation),
                        artifacts_dir=self.f.artifacts,
                        runtime_check_path=runtime_path,
                    )
                self.assertEqual(projection["decision_status"], "ready")
                self.assertIsNotNone(projection["selected_recipe_group_id"])
                with mock.patch.object(
                    summarizer, "verify_confirmation_iid_predictions"
                ):
                    pending = summarizer.adaptive_confirmation_projection(
                        plan=self.f.plan,
                        lock=confirmation,
                        rows=self._confirmation_rows(confirmation),
                        artifacts_dir=self.f.artifacts,
                        runtime_check_path=None,
                    )
                self.assertEqual(pending["decision_status"], "runtime_gate_pending")
                self.assertIsNone(pending["selected_recipe_group_id"])
                failed_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
                runtime_path.write_text(
                    json.dumps(failed_runtime, indent=2, sort_keys=False) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(RuntimeError, "runtime check differs"):
                    with mock.patch.object(
                        summarizer, "verify_confirmation_iid_predictions"
                    ):
                        summarizer.adaptive_confirmation_projection(
                            plan=self.f.plan,
                            lock=confirmation,
                            rows=self._confirmation_rows(confirmation),
                            artifacts_dir=self.f.artifacts,
                            runtime_check_path=runtime_path,
                        )
                failed_runtime["public_seconds"] = 288.0
                failed_runtime.pop("runtime_check_payload_sha256")
                failed_runtime["runtime_check_payload_sha256"] = (
                    adaptive.canonical_sha256(failed_runtime)
                )
                runtime_path.write_text(
                    adaptive.canonical_json_dumps(failed_runtime) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(RuntimeError, "runtime check differs"):
                    with mock.patch.object(
                        summarizer, "verify_confirmation_iid_predictions"
                    ):
                        summarizer.adaptive_confirmation_projection(
                            plan=self.f.plan,
                            lock=confirmation,
                            rows=self._confirmation_rows(confirmation),
                            artifacts_dir=self.f.artifacts,
                            runtime_check_path=runtime_path,
                        )

    def test_partial_confirmation_is_pending_with_frozen_missing_seeds(self) -> None:
        chain = self.f._full_chain()
        confirmation = self.f._confirmation(chain)
        rows = summarizer.adaptive_run_rows(
            plan=self.f.plan,
            lock=confirmation,
            artifacts_dir=self.f.artifacts,
        )
        current_experiments = {
            variant["experiment"]
            for variant in confirmation["resolved_stage"]["variants"]
        }
        missing = [
            row for row in rows if row["experiment"] in current_experiments
        ]
        self.assertTrue(missing)
        self.assertTrue(all(row["completed"] is False for row in missing))
        self.assertEqual(
            {row["seed"] for row in missing},
            {17, 2026},
        )
        with mock.patch.object(
            summarizer, "verify_confirmation_iid_predictions"
        ):
            projection = summarizer.adaptive_confirmation_projection(
                plan=self.f.plan,
                lock=confirmation,
                rows=rows,
                artifacts_dir=self.f.artifacts,
                runtime_check_path=None,
            )
        self.assertEqual(projection["decision_status"], "pending_runs")
        self.assertIsNone(projection["selected_recipe_group_id"])

    def test_confirmation_acceptance_and_tie_boundaries_are_decimal_exact(self) -> None:
        chain = self.f._full_chain()
        confirmation = self.f._confirmation(chain)
        rows = self._confirmation_rows(confirmation)
        rows_by_experiment = {row["experiment"]: row for row in rows}
        variants_by_group: dict[str, list[dict]] = {}
        for variant in confirmation["resolved_stage"]["variants"]:
            variants_by_group.setdefault(variant["recipe_group_id"], []).append(
                variant
            )
        origin_by_id = {
            origin["origin_id"]: origin for origin in confirmation["origins"]
        }
        baseline_group = next(
            group
            for group in confirmation["resolved_stage"]["recipe_groups"]
            if "current_protocol_baseline_recipe" in group["roles"]
        )
        tuned_group = next(
            group
            for group in confirmation["resolved_stage"]["recipe_groups"]
            if "selected_regularized_bce_recipe" in group["roles"]
        )
        for group in confirmation["resolved_stage"]["recipe_groups"]:
            group_rows = [
                rows_by_experiment[
                    origin_by_id[group["origin_seed42_id"]]["experiment"]
                ],
                *[
                    rows_by_experiment[variant["experiment"]]
                    for variant in variants_by_group[group["recipe_group_id"]]
                ],
            ]
            if group["recipe_group_id"] == baseline_group["recipe_group_id"]:
                for row in group_rows:
                    row["iid_macro_ap"] = 0.790
            elif group["recipe_group_id"] == tuned_group["recipe_group_id"]:
                deltas = {17: 0.003, 42: 0.003, 2026: 0.0}
                for row in group_rows:
                    row["iid_macro_ap"] = 0.790 + deltas[row["seed"]]
            else:
                for row in group_rows:
                    row["iid_macro_ap"] = 0.789
        with mock.patch.object(
            summarizer, "verify_confirmation_iid_predictions"
        ):
            projection = summarizer.adaptive_confirmation_projection(
                plan=self.f.plan,
                lock=confirmation,
                rows=rows,
                artifacts_dir=self.f.artifacts,
                runtime_check_path=None,
            )
        evaluations = {
            row["recipe_group_id"]: row for row in projection["groups"]
        }
        self.assertTrue(evaluations[tuned_group["recipe_group_id"]]["accepted"])
        self.assertEqual(
            evaluations[tuned_group["recipe_group_id"]]["mean_iid_delta"],
            0.002,
        )
        self.assertEqual(
            set(projection["practical_shortlist_recipe_group_ids"]),
            {
                baseline_group["recipe_group_id"],
                tuned_group["recipe_group_id"],
            },
        )
        self.assertEqual(
            projection["selection_before_runtime_gate"],
            baseline_group["recipe_group_id"],
        )

    def test_confirmation_summary_stays_pending_until_runtime_gate(self) -> None:
        chain = self.f._full_chain()
        confirmation = self.f._confirmation(chain)
        rows = self._confirmation_rows(confirmation)
        output_dir = self.f.root / "confirmation-summary"
        with mock.patch.object(
            summarizer, "adaptive_run_rows", return_value=rows
        ), mock.patch.object(
            summarizer, "adaptive_hypothesis_projections", return_value=[]
        ), mock.patch.object(
            summarizer, "verify_confirmation_iid_predictions"
        ):
            pending = summarizer.summarize_adaptive_campaign_lock(
                plan=self.f.plan,
                lock=confirmation,
                lock_path=self.f.locks_dir / "confirmation.lock.json",
                artifacts_dir=self.f.artifacts,
                output_dir=output_dir,
            )
            runtime_path = self._runtime_check(
                confirmation, self.f.root / "runtime-check.json"
            )
            ready = summarizer.summarize_adaptive_campaign_lock(
                plan=self.f.plan,
                lock=confirmation,
                lock_path=self.f.locks_dir / "confirmation.lock.json",
                artifacts_dir=self.f.artifacts,
                output_dir=output_dir,
                runtime_check_path=runtime_path,
            )
        self.assertEqual(pending["execution_status"], "pending")
        self.assertFalse(
            pending["stages"][confirmation["effective_stage"]]["complete"]
        )
        self.assertTrue(
            pending["stages"][confirmation["effective_stage"]]["runs_complete"]
        )
        self.assertEqual(ready["execution_status"], "complete")
        self.assertTrue(
            ready["stages"][confirmation["effective_stage"]]["complete"]
        )

    def test_six_run_confirmation_notebooks_and_completion_ingestion(self) -> None:
        chain = self.f._full_chain(
            metrics={
                "balanced_binary_bce": 0.805,
                "balanced_category_class_sqrt_bce": 0.790,
                "balanced_category_class_bce": 0.789,
                "focal_bce_gamma2_scale4": 0.790,
            }
        )
        confirmation = self.f._confirmation(chain)
        confirmation_path = self.f.locks_dir / "confirmation.lock.json"
        self.assertEqual(len(confirmation["resolved_stage"]["variants"]), 6)
        with tempfile.TemporaryDirectory() as notebook_dir:
            built = builder.build_campaign(
                plan_path=self.f.plan_path,
                output_dir=Path(notebook_dir),
                stage_lock_path=confirmation_path,
            )
            self.assertEqual(len(built), 6)
        entries = {
            entry["experiment"]: entry
            for entry in builder.normalized_campaign_execution_contract(
                self.f.plan,
                confirmation,
                base_config=self.f.base_config,
            )["variants"]
        }
        for variant in confirmation["resolved_stage"]["variants"]:
            roles = set(variant["confirmation_roles"])
            ap = (
                0.790
                if "current_protocol_baseline_recipe" in roles
                else 0.7921
                if "selected_regularized_bce_recipe" in roles
                else 0.796
            )
            row = self.f._run_from_variant(confirmation, variant, ap)
            self._enrich_run(row, entries[variant["experiment"]])
        rows = summarizer.adaptive_run_rows(
            plan=self.f.plan,
            lock=confirmation,
            artifacts_dir=self.f.artifacts,
        )
        current_experiments = set(entries)
        completed_current = [
            row
            for row in rows
            if row["experiment"] in current_experiments and row["completed"]
        ]
        self.assertEqual(len(completed_current), 6)
        self.assertTrue(
            all(row.get("iid_predictions_sha256") for row in completed_current)
        )
        with mock.patch.object(
            summarizer, "verify_confirmation_iid_predictions"
        ):
            projection = summarizer.adaptive_confirmation_projection(
                plan=self.f.plan,
                lock=confirmation,
                rows=rows,
                artifacts_dir=self.f.artifacts,
                runtime_check_path=None,
            )
        self.assertEqual(projection["decision_status"], "runtime_gate_pending")
        self.assertIsNotNone(projection["selection_before_runtime_gate"])

    def test_projection_families_keep_reserved_holm_and_anchor_statistics(self) -> None:
        chain = self.f._full_chain()
        comparison = {
            "delta_macro_average_precision": 0.003,
            "p_value": 0.01,
            "ci95_low": 0.001,
            "ci95_high": 0.005,
        }
        closure = [
            (chain["primary"], self.f.locks_dir / "primary.lock.json"),
            (chain["overlay"], self.f.locks_dir / "overlay.lock.json"),
            (chain["refine"], self.f.locks_dir / "lr.lock.json"),
        ]
        with mock.patch.object(
            summarizer, "_adaptive_comparison", return_value=comparison
        ):
            projections = summarizer.adaptive_hypothesis_projections(
                closure=closure,
                rows=chain["loss_final"]["runs"],
                artifacts_dir=self.f.artifacts,
                cache_dir=self.f.root / "comparison-cache",
            )
        by_family = {row["family_id"]: row for row in projections}
        primary = by_family[chain["primary"]["family"]["family_id"]]
        refine = by_family[chain["refine"]["family"]["family_id"]]
        self.assertEqual(primary["maximum_hypotheses"], 5)
        self.assertEqual(primary["reserved_p_equals_1"], 0)
        self.assertEqual(refine["maximum_hypotheses"], 2)
        self.assertEqual(refine["reserved_p_equals_1"], 0)
        for candidate in [*primary["candidates"], *refine["candidates"]]:
            self.assertEqual(candidate["iid_delta_vs_anchor"], 0.003)
            self.assertEqual(candidate["iid_p_value_vs_anchor"], 0.01)
            self.assertEqual(candidate["iid_ci95_low_vs_anchor"], 0.001)
            self.assertEqual(candidate["iid_ci95_high_vs_anchor"], 0.005)

    def test_confirmation_iid_verifier_binds_report_and_matched_pairs(self) -> None:
        def write_predictions(slug: str, offset: int) -> Path:
            path = (
                self.f.artifacts
                / slug
                / "model"
                / "iid_validation_predictions.parquet"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "id1": [offset + 1, offset + 2],
                    "id2": [offset + 11, offset + 12],
                    "target": [0, 1],
                    "score": [0.1, 0.9],
                    "category": ["test", "test"],
                }
            ).to_parquet(path, index=False)
            return path

        baseline_path = write_predictions("confirmation-baseline", 0)
        candidate_path = write_predictions("confirmation-candidate", 0)
        baseline_row = {
            "experiment": "baseline",
            "kernel_slug": "confirmation-baseline",
            "run_id": "baseline-run",
            "seed": 17,
            "completed": True,
            "iid_macro_ap": 1.0,
            "iid_predictions_sha256": adaptive.file_sha256(baseline_path),
        }
        candidate_row = {
            "experiment": "candidate",
            "kernel_slug": "confirmation-candidate",
            "run_id": "candidate-run",
            "seed": 17,
            "completed": True,
            "iid_macro_ap": 1.0,
            "iid_predictions_sha256": adaptive.file_sha256(candidate_path),
        }
        groups = [
            ({"recipe_group_id": "baseline"}, [baseline_row]),
            ({"recipe_group_id": "candidate"}, [candidate_row]),
        ]
        summarizer.verify_confirmation_iid_predictions(
            groups=groups,
            baseline_group_id="baseline",
            artifacts_dir=self.f.artifacts,
        )
        candidate_row["iid_macro_ap"] = 0.9
        with self.assertRaisesRegex(RuntimeError, "report/parquet metric differs"):
            summarizer.verify_confirmation_iid_predictions(
                groups=groups,
                baseline_group_id="baseline",
                artifacts_dir=self.f.artifacts,
            )
        candidate_row["iid_macro_ap"] = 1.0
        candidate_path = write_predictions("confirmation-candidate", 100)
        candidate_row["iid_predictions_sha256"] = adaptive.file_sha256(
            candidate_path
        )
        with self.assertRaisesRegex(Exception, "pair sets differ"):
            summarizer.verify_confirmation_iid_predictions(
                groups=groups,
                baseline_group_id="baseline",
                artifacts_dir=self.f.artifacts,
            )

    def test_skipped_receipt_writes_complete_zero_run_closure_summary(self) -> None:
        metrics = {
            "balanced_binary_bce": 0.799,
            "balanced_category_class_sqrt_bce": 0.798,
            "balanced_category_class_bce": 0.797,
            "focal_bce_gamma2_scale4": 0.799,
        }
        primary = self.f._primary()
        primary_summary, rows = self.f._primary_summary(primary, metrics=metrics)
        primary_contract = builder.normalized_campaign_execution_contract(
            self.f.plan, primary, base_config=self.f.base_config
        )
        entries = {
            entry["experiment"]: entry for entry in primary_contract["variants"]
        }
        for row in rows.values():
            self._enrich_run(row, entries[row["experiment"]])
        overlay = self.f._overlay(primary, primary_summary)
        self.assertEqual(overlay["execution_status"], "skipped")
        output_dir = self.f.root / "skipped-summary"
        with mock.patch.object(
            summarizer,
            "adaptive_hypothesis_projections",
            return_value=[
                {
                    "family_id": primary["family"]["family_id"],
                    "maximum_hypotheses": 5,
                    "reserved_p_equals_1": 1,
                }
            ],
        ):
            result = summarizer.summarize_adaptive_campaign_lock(
                plan=self.f.plan,
                lock=overlay,
                lock_path=self.f.locks_dir / "overlay.lock.json",
                artifacts_dir=self.f.artifacts,
                output_dir=output_dir,
            )
        self.assertEqual(result["execution_status"], "complete")
        self.assertEqual(result["execution_lock_sha256s"], [primary["lock_payload_sha256"]])
        self.assertEqual(
            result["execution_receipt_sha256s"], [overlay["lock_payload_sha256"]]
        )
        stage = result["stages"][overlay["effective_stage"]]
        self.assertEqual(stage["expected_new_runs"], 0)
        self.assertEqual(stage["completed_new_runs"], 0)
        adaptive.validate_execution_summary(
            result,
            plan=self.f.plan,
            required_locks=[primary, overlay],
        )
        refine_path = self.f.locks_dir / "integrated-skipped-lr.lock.json"
        refine = adaptive.materialize_loss_lr_refine_lock(
            plan=self.f.plan,
            summary=result,
            summary_path=output_dir / "summary.json",
            artifacts_dir=self.f.artifacts,
            prerequisite_locks=[primary, overlay],
            prerequisite_lock_paths=[
                self.f.locks_dir / "primary.lock.json",
                self.f.locks_dir / "overlay.lock.json",
            ],
            history_documents=[],
            history_document_paths=[],
            output_path=refine_path,
        )
        self.assertEqual(refine["execution_status"], "skipped")
        refine_output = self.f.root / "skipped-lr-summary"
        with mock.patch.object(
            summarizer, "adaptive_hypothesis_projections", return_value=[]
        ):
            refine_result = summarizer.summarize_adaptive_campaign_lock(
                plan=self.f.plan,
                lock=refine,
                lock_path=refine_path,
                artifacts_dir=self.f.artifacts,
                output_dir=refine_output,
            )
        self.assertEqual(refine_result["execution_status"], "complete")
        self.assertEqual(
            refine_result["execution_receipt_sha256s"],
            sorted(
                [overlay["lock_payload_sha256"], refine["lock_payload_sha256"]]
            ),
        )
        adaptive.validate_execution_summary(
            refine_result,
            plan=self.f.plan,
            required_locks=[primary, overlay, refine],
        )

    def test_launcher_resume_validates_schema_v2_exact_provenance(self) -> None:
        primary = self.f._primary()
        summary, rows = self.f._primary_summary(primary)
        loaded = self._load(self.f.locks_dir / "primary.lock.json")
        entry = launcher.campaign_variants(
            self.f.plan, stage=None, only=None, stage_lock=loaded
        )[0]
        row = rows[entry["loss_variant"]]
        self._enrich_run(row, entry, full_parquets=True)
        root = self.f.artifacts / entry["kernel_slug"]
        validated = launcher.validate_run_output(root, entry=entry)
        self.assertEqual(validated["run_id"], row["run_id"])
        completion_path = root / "notebook_completed.json"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["notes"] = "{}"
        completion_path.write_text(json.dumps(completion), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "notes differ"):
            launcher.validate_run_output(root, entry=entry)
        self.assertEqual(summary["campaign"], self.f.plan["campaign"])

    def test_dispatcher_rejects_relocated_or_payload_selected_manifest(self) -> None:
        self.f._primary()
        original = self.f.locks_dir / "primary.lock.json"
        relocated = self.f.locks_dir / "relocated-primary.lock.json"
        shutil.copy2(original, relocated)
        shutil.copy2(
            adaptive.trusted_provenance_manifest_path(original),
            adaptive.trusted_provenance_manifest_path(relocated),
        )
        with self.assertRaisesRegex(builder.CampaignConfigError, "relocated"):
            self._load(relocated)

        payload = json.loads(original.read_text(encoding="utf-8"))
        payload["trusted_manifest_path"] = str(
            adaptive.trusted_provenance_manifest_path(original)
        )
        payload.pop("lock_payload_sha256")
        payload["lock_payload_sha256"] = adaptive.canonical_sha256(payload)
        forged = self.f.locks_dir / "payload-selected-manifest.lock.json"
        forged.write_text(adaptive.canonical_json_dumps(payload) + "\n", encoding="utf-8")
        shutil.copy2(
            adaptive.trusted_provenance_manifest_path(original),
            adaptive.trusted_provenance_manifest_path(forged),
        )
        with self.assertRaises(builder.CampaignConfigError):
            self._load(forged)

        symlink = self.f.locks_dir / "primary-symlink.lock.json"
        symlink.symlink_to(original.name)
        with self.assertRaisesRegex(builder.CampaignConfigError, "symlink"):
            self._load(symlink)

    def test_closure_reloads_and_binds_root_object_to_exact_path(self) -> None:
        primary = self.f._primary()
        changed = deepcopy(primary)
        changed["mode"] = "loss_overlay"
        with self.assertRaisesRegex(RuntimeError, "root lock object differs"):
            summarizer.load_adaptive_prerequisite_closure(
                plan=self.f.plan,
                root_lock=changed,
                root_lock_path=self.f.locks_dir / "primary.lock.json",
                base_config=self.f.base_config,
            )

    def test_schema_v2_summarizer_rejects_unrelated_stage_filter(self) -> None:
        self.f._primary()
        argv = [
            "summarize_minilm_5ep_sft_hparams.py",
            "--plan",
            str(self.f.plan_path),
            "--stage-lock",
            str(self.f.locks_dir / "primary.lock.json"),
            "--stage",
            "epoch_line",
        ]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaisesRegex(
                builder.CampaignConfigError, "differs from locked stage"
            ):
                summarizer.main()

    def test_schema1_dispatcher_is_exact_regression(self) -> None:
        strict = builder.load_stage_lock(
            self.f.source_lock_path,
            plan=self.f.plan,
            base_config=self.f.base_config,
        )
        dispatched = self._load(self.f.source_lock_path)
        self.assertEqual(dispatched, strict)

    def test_adaptive_summary_is_directly_consumable_by_next_materializer(self) -> None:
        primary, _, _ = self._executed_primary()
        output_dir = self.f.root / "adaptive-summary"
        comparison = {
            "delta_macro_average_precision": 0.003,
            "p_value": 0.01,
            "ci95_low": 0.001,
            "ci95_high": 0.005,
        }
        with mock.patch.object(
            summarizer, "_adaptive_comparison", return_value=comparison
        ):
            result = summarizer.summarize_adaptive_campaign_lock(
                plan=self.f.plan,
                lock=primary,
                lock_path=self.f.locks_dir / "primary.lock.json",
                artifacts_dir=self.f.artifacts,
                output_dir=output_dir,
            )
        adaptive.validate_execution_summary(
            result,
            plan=self.f.plan,
            required_locks=[primary],
        )
        self.assertEqual(len({row["run_id"] for row in result["runs"]}), 5)
        self.assertEqual(result["hypothesis_families"][0]["reserved_p_equals_1"], 1)
        overlay = adaptive.materialize_loss_overlay_lock(
            plan=self.f.plan,
            summary=result,
            summary_path=output_dir / "summary.json",
            artifacts_dir=self.f.artifacts,
            prerequisite_locks=[primary],
            prerequisite_lock_paths=[self.f.locks_dir / "primary.lock.json"],
            history_documents=[],
            history_document_paths=[],
            output_path=self.f.locks_dir / "integrated-overlay.lock.json",
        )
        self.assertEqual(overlay["mode"], "loss_overlay")
        self.assertEqual(overlay["execution_status"], "runnable")
        overlay_path = self.f.locks_dir / "integrated-overlay.lock.json"
        with tempfile.TemporaryDirectory() as notebook_dir:
            built = builder.build_campaign(
                plan_path=self.f.plan_path,
                output_dir=Path(notebook_dir),
                stage_lock_path=overlay_path,
            )
            self.assertEqual(len(built), 1)
        overlay_entry = builder.normalized_campaign_execution_contract(
            self.f.plan,
            overlay,
            base_config=self.f.base_config,
        )["variants"][0]
        overlay_row = self.f._run_from_variant(
            overlay, overlay["resolved_stage"]["variants"][0], 0.803
        )
        self._enrich_run(overlay_row, overlay_entry)
        overlay_output = self.f.root / "integrated-overlay-summary"
        with mock.patch.object(
            summarizer, "_adaptive_comparison", return_value=comparison
        ):
            overlay_result = summarizer.summarize_adaptive_campaign_lock(
                plan=self.f.plan,
                lock=overlay,
                lock_path=overlay_path,
                artifacts_dir=self.f.artifacts,
                output_dir=overlay_output,
            )
        self.assertEqual(
            overlay_result["hypothesis_families"][0]["reserved_p_equals_1"],
            0,
        )
        refine_path = self.f.locks_dir / "integrated-lr.lock.json"
        refine = adaptive.materialize_loss_lr_refine_lock(
            plan=self.f.plan,
            summary=overlay_result,
            summary_path=overlay_output / "summary.json",
            artifacts_dir=self.f.artifacts,
            prerequisite_locks=[primary, overlay],
            prerequisite_lock_paths=[
                self.f.locks_dir / "primary.lock.json",
                overlay_path,
            ],
            history_documents=[],
            history_document_paths=[],
            output_path=refine_path,
        )
        self.assertEqual(refine["execution_status"], "runnable")
        with tempfile.TemporaryDirectory() as notebook_dir:
            built = builder.build_campaign(
                plan_path=self.f.plan_path,
                output_dir=Path(notebook_dir),
                stage_lock_path=refine_path,
            )
            self.assertEqual(len(built), 2)
        refine_entries = {
            entry["experiment"]: entry
            for entry in builder.normalized_campaign_execution_contract(
                self.f.plan,
                refine,
                base_config=self.f.base_config,
            )["variants"]
        }
        for variant, ap in zip(
            refine["resolved_stage"]["variants"], (0.806, 0.804), strict=True
        ):
            row = self.f._run_from_variant(refine, variant, ap)
            self._enrich_run(row, refine_entries[variant["experiment"]])
        refine_output = self.f.root / "integrated-lr-summary"
        with mock.patch.object(
            summarizer, "_adaptive_comparison", return_value=comparison
        ):
            refine_result = summarizer.summarize_adaptive_campaign_lock(
                plan=self.f.plan,
                lock=refine,
                lock_path=refine_path,
                artifacts_dir=self.f.artifacts,
                output_dir=refine_output,
            )
        self.assertEqual(len(refine_result["hypothesis_families"]), 2)
        self.assertEqual(
            refine_result["hypothesis_families"][1]["maximum_hypotheses"], 2
        )
        confirmation_path = self.f.locks_dir / "integrated-confirmation.lock.json"
        confirmation = adaptive.materialize_confirmation_lock(
            plan=self.f.plan,
            summary=refine_result,
            summary_path=refine_output / "summary.json",
            baseline_summary=self.f.baseline_summary,
            baseline_summary_path=self.f.baseline_summary_path,
            artifacts_dir=self.f.artifacts,
            prerequisite_locks=[primary, overlay, refine],
            prerequisite_lock_paths=[
                self.f.locks_dir / "primary.lock.json",
                overlay_path,
                refine_path,
            ],
            history_documents=[],
            history_document_paths=[],
            output_path=confirmation_path,
        )
        self.assertEqual(
            len(confirmation["resolved_stage"]["variants"]), 8
        )
        with tempfile.TemporaryDirectory() as notebook_dir:
            built = builder.build_campaign(
                plan_path=self.f.plan_path,
                output_dir=Path(notebook_dir),
                stage_lock_path=confirmation_path,
            )
            self.assertEqual(len(built), 8)
        confirmation_entries = {
            entry["experiment"]: entry
            for entry in builder.normalized_campaign_execution_contract(
                self.f.plan,
                confirmation,
                base_config=self.f.base_config,
            )["variants"]
        }
        for variant in confirmation["resolved_stage"]["variants"]:
            roles = set(variant["confirmation_roles"])
            ap = (
                0.790
                if "current_protocol_baseline_recipe" in roles
                else 0.7921
                if "selected_regularized_bce_recipe" in roles
                else 0.796
            )
            row = self.f._run_from_variant(confirmation, variant, ap)
            self._enrich_run(row, confirmation_entries[variant["experiment"]])
        confirmation_output = self.f.root / "integrated-confirmation-summary"
        with mock.patch.object(
            summarizer, "_adaptive_comparison", return_value=comparison
        ), mock.patch.object(
            summarizer, "verify_confirmation_iid_predictions"
        ):
            pending = summarizer.summarize_adaptive_campaign_lock(
                plan=self.f.plan,
                lock=confirmation,
                lock_path=confirmation_path,
                artifacts_dir=self.f.artifacts,
                output_dir=confirmation_output,
            )
            selected_group_id = pending["confirmation"][
                "selection_before_runtime_gate"
            ]
            runtime_path = self._runtime_check(
                confirmation,
                self.f.root / "integrated-runtime-check.json",
                selected_recipe_group_id=selected_group_id,
            )
            ready = summarizer.summarize_adaptive_campaign_lock(
                plan=self.f.plan,
                lock=confirmation,
                lock_path=confirmation_path,
                artifacts_dir=self.f.artifacts,
                output_dir=confirmation_output,
                runtime_check_path=runtime_path,
            )
        self.assertEqual(pending["execution_status"], "pending")
        self.assertEqual(ready["execution_status"], "complete")
        confirmation_stage = ready["stages"][confirmation["effective_stage"]]
        self.assertEqual(confirmation_stage["completed_new_runs"], 8)


if __name__ == "__main__":
    unittest.main()
