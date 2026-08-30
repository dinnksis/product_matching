from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from unittest import mock
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import materialize_minilm_5ep_sft_hparam_stage as materializer
import create_minilm_5ep_sft_hparam_notebooks as builder
import run_minilm_5ep_sft_hparam_kaggle as launcher
import summarize_minilm_5ep_sft_hparams as summarizer
from summarize_minilm_5ep_sft_hparams import (
    add_stage_holm,
    expected_entries,
    stage_summary,
)

import nbformat
import pandas as pd
from sklearn.metrics import average_precision_score


class Minilm5epStageMaterializerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.plan_path = self.root / "campaign.json"
        self.summary_path = self.root / "summary.json"
        self.artifacts_dir = self.root / "artifacts"
        self.lock_path = self.root / "locks" / "epoch_line.lock.json"

        self.plan = json.loads(
            materializer.DEFAULT_PLAN.read_text(encoding="utf-8")
        )
        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
        source_stage = next(
            stage
            for stage in self.plan["stages"]
            if stage["name"] == "lr_log_line"
        )
        self.parent_variant = next(
            variant
            for variant in source_stage["variants"]
            if variant["overrides"]["learning_rate"] == 1e-5
        )
        base_config = json.loads(
            materializer.BASE_CONFIG_PATH.read_text(encoding="utf-8")
        )
        self.parent_config = deepcopy(base_config)
        self.parent_config.update(self.parent_variant["overrides"])
        self.recipe_sha = materializer.canonical_sha256(self.parent_config)
        self.run_id = "0123456789abcdef0123456789abcdef"
        self.kernel_slug = self.parent_variant["kernel_slug"]

        artifact_root = self.artifacts_dir / self.kernel_slug
        model_dir = artifact_root / "model"
        model_dir.mkdir(parents=True)
        runtime_config = deepcopy(self.parent_config)
        runtime_config["model"] = "/kaggle/input/frozen-checkpoint/model"
        (model_dir / "training_config.json").write_text(
            json.dumps(runtime_config),
            encoding="utf-8",
        )
        self.iid_bytes = b"deterministic-iid-predictions-for-test"
        (model_dir / "iid_validation_predictions.parquet").write_bytes(
            self.iid_bytes
        )
        completion = {
            "status": "complete",
            "experiment": self.parent_variant["experiment"],
            "run_id": self.run_id,
            "frozen_recipe_sha256": self.recipe_sha,
        }
        completion_path = artifact_root / "notebook_completed.json"
        completion_path.write_text(json.dumps(completion), encoding="utf-8")

        self.summary = {
            "schema_version": 1,
            "campaign": self.plan["campaign"],
            "stages": {
                "lr_log_line": {
                    "expected_runs": 4,
                    "completed_runs": 4,
                    "complete": True,
                    "decision_status": "ready",
                    "control_gate": "passed",
                    "recommended_experiment": self.parent_variant["experiment"],
                    "recommended_run_id": self.run_id,
                    "needs_boundary_extension": False,
                }
            },
            "runs": [
                {
                    "stage": "lr_log_line",
                    "experiment": self.parent_variant["experiment"],
                    "kernel_slug": self.kernel_slug,
                    "role": "candidate",
                    "completed": True,
                    "status": "complete",
                    "run_id": self.run_id,
                    "recipe_sha256": self.recipe_sha,
                    "completion_path": str(completion_path),
                }
            ],
        }
        self._write_summary()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_summary(self) -> None:
        self.summary_path.write_text(
            json.dumps(self.summary),
            encoding="utf-8",
        )

    def _materialize(self) -> dict[str, object]:
        return materializer.materialize_stage_lock(
            plan_path=self.plan_path,
            summary_path=self.summary_path,
            artifacts_dir=self.artifacts_dir,
            source_stage="lr_log_line",
            target_stage="epoch_line",
            output_path=self.lock_path,
        )

    def _prepare_lr_boundary_extension(
        self,
    ) -> tuple[dict[str, object], dict[str, str]]:
        source_stage = next(
            stage
            for stage in self.plan["stages"]
            if stage["name"] == "lr_log_line"
        )
        base_config = json.loads(
            materializer.BASE_CONFIG_PATH.read_text(encoding="utf-8")
        )
        _, source_sha256 = builder.baseline_builder.embedded_sources()
        run_ids: dict[str, str] = {}
        rows = []
        for index, variant in enumerate(source_stage["variants"], start=1):
            config = deepcopy(base_config)
            config.update(variant["overrides"])
            recipe_sha256 = materializer.canonical_sha256(config)
            run_id = f"{index:032x}"
            run_ids[variant["experiment"]] = run_id
            artifact_root = self.artifacts_dir / variant["kernel_slug"]
            model_dir = artifact_root / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            runtime_config = deepcopy(config)
            runtime_config["model"] = "/kaggle/input/frozen-checkpoint/model"
            (model_dir / "training_config.json").write_text(
                json.dumps(runtime_config), encoding="utf-8"
            )
            iid_path = model_dir / "iid_validation_predictions.parquet"
            iid_path.write_bytes(f"iid-{variant['experiment']}".encode("utf-8"))
            notes = materializer.canonical_json_dumps(
                {"stage": "lr_log_line", "experiment": variant["experiment"]}
            )
            completion = {
                "status": "complete",
                "experiment": variant["experiment"],
                "run_id": run_id,
                "frozen_recipe_sha256": recipe_sha256,
                "code_bundle_sha256": source_sha256,
                "loss_hook_sha256": builder.FIXED_LOSS_HOOK_SHA256,
                "notes": notes,
            }
            completion_path = artifact_root / "notebook_completed.json"
            completion_path.write_text(json.dumps(completion), encoding="utf-8")
            role = variant.get("role", "candidate")
            rows.append(
                {
                    "stage": "lr_log_line",
                    "experiment": variant["experiment"],
                    "kernel_slug": variant["kernel_slug"],
                    "role": role,
                    "completed": True,
                    "status": "complete",
                    "run_id": run_id,
                    "recipe_sha256": recipe_sha256,
                    "iid_predictions_sha256": hashlib.sha256(
                        iid_path.read_bytes()
                    ).hexdigest(),
                    "completion_path": str(completion_path),
                    "loss_variant": "bce",
                    "is_hypothesis": role != "current_protocol_control",
                    "hypothesis_family_size": 4,
                }
            )
        edge = next(
            variant
            for variant in source_stage["variants"]
            if variant["overrides"]["learning_rate"] == 5e-6
        )
        self.summary = {
            "schema_version": 1,
            "campaign": self.plan["campaign"],
            "stages": {
                "lr_log_line": {
                    "expected_runs": 3,
                    "completed_runs": 3,
                    "complete": True,
                    "decision_status": "ready",
                    "control_gate": "passed",
                    "recommended_experiment": edge["experiment"],
                    "recommended_run_id": run_ids[edge["experiment"]],
                    "needs_boundary_extension": True,
                    "recommended_extension": {
                        "axis": "learning_rate",
                        "direction": "lower",
                        "level": 2.5e-6,
                        "gain_over_nearest_interior": 0.003,
                    },
                }
            },
            "runs": rows,
        }
        self._write_summary()
        return source_stage, run_ids

    def test_lr_line_to_epoch_line_lock_contains_frozen_parent_and_recipes(self) -> None:
        lock = self._materialize()

        self.assertEqual(lock["source_stage"], "lr_log_line")
        self.assertEqual(lock["target_stage"], "epoch_line")
        self.assertEqual(lock["effective_stage"], "epoch_line")
        self.assertEqual(lock["parent"]["experiment"], self.parent_variant["experiment"])
        self.assertEqual(lock["parent"]["run_id"], self.run_id)
        self.assertEqual(lock["parent"]["recipe_sha256"], self.recipe_sha)
        self.assertEqual(lock["parent"]["resolved_config"], self.parent_config)
        self.assertEqual(
            lock["parent"]["iid_predictions_sha256"],
            hashlib.sha256(self.iid_bytes).hexdigest(),
        )

        resolved = lock["resolved_stage"]
        self.assertEqual(resolved["axis"], "epochs")
        self.assertEqual(resolved["parent_level"], 1)
        self.assertEqual(resolved["family"]["maximum_hypotheses"], 3)
        variants = resolved["variants"]
        self.assertEqual([variant["level"] for variant in variants], [2, 3])
        for variant in variants:
            self.assertEqual(
                variant["resolved_config"]["learning_rate"],
                self.parent_config["learning_rate"],
            )
            self.assertEqual(
                variant["resolved_config"]["epochs"], variant["level"]
            )
            self.assertEqual(
                variant["overrides"],
                {
                    "epochs": variant["level"],
                    "learning_rate": 1e-5,
                },
            )
            self.assertEqual(
                variant["expected_recipe_sha256"],
                materializer.canonical_sha256(variant["resolved_config"]),
            )
        self.assertEqual(
            [variant["experiment"] for variant in variants],
            [
                "minilm5_sft_e2_lr1e5_v1",
                "minilm5_sft_e3_lr1e5_v1",
            ],
        )

        parsed = json.loads(self.lock_path.read_text(encoding="utf-8"))
        self.assertEqual(parsed, lock)
        self.assertEqual(
            self.lock_path.read_text(encoding="utf-8"),
            materializer.canonical_json_dumps(lock) + "\n",
        )
        unhashed = dict(lock)
        stored_sha = unhashed.pop("lock_payload_sha256")
        self.assertEqual(stored_sha, materializer.canonical_sha256(unhashed))

    def test_existing_lock_is_reused_without_reading_parent_artifacts_again(self) -> None:
        first = self._materialize()
        preserved_summary = (
            self.summary_path.parent
            / "stages"
            / "lr_log_line"
            / "summary.json"
        )
        preserved_summary.parent.mkdir(parents=True)
        preserved_summary.write_text(
            json.dumps(self.summary),
            encoding="utf-8",
        )
        self.summary_path.write_text(
            json.dumps(
                {
                    "campaign": self.plan["campaign"],
                    "stages": {"epoch_line": {"complete": False}},
                    "runs": [],
                }
            ),
            encoding="utf-8",
        )
        for path in self.artifacts_dir.rglob("*"):
            if path.is_file():
                path.unlink()

        second = self._materialize()

        self.assertEqual(second, first)
        self.assertEqual(
            second["parent"]["experiment"], self.parent_variant["experiment"]
        )

    def test_existing_lock_refuses_a_different_summary_parent(self) -> None:
        self._materialize()
        decision = self.summary["stages"]["lr_log_line"]
        decision["recommended_experiment"] = "different_parent"
        decision["recommended_run_id"] = "fedcba9876543210fedcba9876543210"
        self.summary["runs"].append(
            {
                "stage": "lr_log_line",
                "experiment": "different_parent",
                "kernel_slug": "different-parent",
                "completed": True,
                "status": "complete",
                "run_id": decision["recommended_run_id"],
            }
        )
        self._write_summary()

        with self.assertRaises(materializer.ExistingLockConflictError):
            self._materialize()

    def test_existing_lock_refuses_plan_drift_and_noncanonical_edits(self) -> None:
        self._materialize()
        changed_plan = deepcopy(self.plan)
        changed_plan["objective"] += " changed"
        self.plan_path.write_text(json.dumps(changed_plan), encoding="utf-8")
        with self.assertRaises(materializer.ExistingLockConflictError):
            self._materialize()

        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
        payload = json.loads(self.lock_path.read_text(encoding="utf-8"))
        self.lock_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with self.assertRaises(materializer.ExistingLockConflictError):
            self._materialize()

    def test_incomplete_or_unextended_source_stage_cannot_be_materialized(self) -> None:
        self.summary["stages"]["lr_log_line"]["complete"] = False
        self._write_summary()
        with self.assertRaises(materializer.StageMaterializationError):
            self._materialize()

        self.summary["stages"]["lr_log_line"]["complete"] = True
        self.summary["stages"]["lr_log_line"]["needs_boundary_extension"] = True
        self._write_summary()
        with self.assertRaises(materializer.StageMaterializationError):
            self._materialize()

    def test_coordinate_stage_schema_resolves_effective_batch_recipes(self) -> None:
        target = next(
            stage
            for stage in self.plan["stages"]
            if stage["name"] == "regularization_coordinate_search"
        )
        parent = {
            "experiment": self.parent_variant["experiment"],
            "run_id": self.run_id,
            "resolved_config": self.parent_config,
        }

        resolved = materializer.resolve_target_stage(
            target,
            parent=parent,
            coordinate="effective_batch",
        )

        self.assertEqual(resolved["axis"], "effective_batch")
        self.assertEqual(resolved["parent_level"], 192)
        self.assertEqual([variant["level"] for variant in resolved["variants"]], [96, 384])
        by_level = {variant["level"]: variant for variant in resolved["variants"]}
        self.assertEqual(by_level[96]["coordinate_overrides"], {
            "batch_size": 48,
            "gradient_accumulation": 1,
        })
        self.assertEqual(by_level[384]["coordinate_overrides"], {
            "batch_size": 96,
            "gradient_accumulation": 2,
        })
        weight_decay = materializer.resolve_target_stage(
            target,
            parent=parent,
            coordinate="weight_decay",
        )
        self.assertEqual(
            weight_decay["family"],
            {
                "correction": "holm",
                "planned_candidate_hypotheses": 2,
                "reserved_conditional_extensions": 1,
                "maximum_hypotheses": 3,
            },
        )

    def test_stage_transition_order_is_exact_for_all_coordinate_substages(self) -> None:
        self.assertEqual(
            materializer.expected_source_stage(
                self.plan, target_stage="epoch_line", coordinate=None
            ),
            "lr_log_line",
        )
        expected = {
            "effective_batch": "epoch_line",
            "warmup_ratio": "regularization_coordinate_search__effective_batch",
            "weight_decay": "regularization_coordinate_search__warmup_ratio",
            "label_smoothing": "regularization_coordinate_search__weight_decay",
            "classifier_dropout": "regularization_coordinate_search__label_smoothing",
            "max_grad_norm": "regularization_coordinate_search__classifier_dropout",
        }
        for coordinate, predecessor in expected.items():
            with self.subTest(coordinate=coordinate):
                self.assertEqual(
                    materializer.expected_source_stage(
                        self.plan,
                        target_stage="regularization_coordinate_search",
                        coordinate=coordinate,
                    ),
                    predecessor,
                )
        with self.assertRaisesRegex(
            materializer.StageMaterializationError, "Out-of-order transition"
        ):
            materializer.validate_stage_transition(
                self.plan,
                source_stage="epoch_line",
                target_stage="regularization_coordinate_search",
                coordinate="weight_decay",
            )

    def test_boundary_extension_lock_reuses_full_family_and_is_one_shot(self) -> None:
        source_stage, run_ids = self._prepare_lr_boundary_extension()
        output_path = self.root / "locks" / "lr_log_line_boundary.lock.json"

        lock = materializer.materialize_boundary_extension_lock(
            plan_path=self.plan_path,
            summary_path=self.summary_path,
            artifacts_dir=self.artifacts_dir,
            source_stage="lr_log_line",
            output_path=output_path,
        )

        self.assertEqual(lock["transition_kind"], "conditional_boundary_extension")
        self.assertEqual(lock["source_stage"], "lr_log_line")
        self.assertEqual(lock["effective_stage"], "lr_log_line")
        self.assertEqual(len(lock["prior_entries"]), 4)
        self.assertEqual(
            lock["parent"]["experiment"], source_stage["variants"][0]["experiment"]
        )
        edge = next(
            variant
            for variant in source_stage["variants"]
            if variant["overrides"]["learning_rate"] == 5e-6
        )
        self.assertEqual(lock["extension_source"]["experiment"], edge["experiment"])
        self.assertEqual(
            lock["extension_source"]["run_id"], run_ids[edge["experiment"]]
        )
        resolved = lock["resolved_stage"]
        self.assertEqual(
            resolved["family"],
            {
                "correction": "holm",
                "planned_candidate_hypotheses": 4,
                "reserved_conditional_extensions": 0,
                "maximum_hypotheses": 4,
            },
        )
        self.assertTrue(resolved["conditional_extension_consumed"])
        self.assertEqual(resolved["conditional_extension_levels"], [])
        self.assertEqual(len(resolved["variants"]), 1)
        self.assertEqual(resolved["variants"][0]["level"], 2.5e-6)

        parsed = builder.load_stage_lock(
            output_path, plan=builder.load_plan(self.plan_path)
        )
        entries = expected_entries(self.plan, None, stage_lock=parsed)
        self.assertEqual(len(entries), 5)
        self.assertEqual(sum(bool(row["is_hypothesis"]) for row in entries), 4)
        self.assertEqual(
            sum(row["expected_run_id"] is None for row in entries), 1
        )
        self.assertTrue(
            all(row["extension_consumed_axis"] == "learning_rate" for row in entries)
        )
        launched = launcher.campaign_variants(
            self.plan, stage=None, only=None, stage_lock=parsed
        )
        self.assertEqual(len(launched), 1)
        self.assertEqual(launched[0]["experiment"], resolved["variants"][0]["experiment"])

        # Once the extension summary is final, replay validates its lock pointer
        # and does not try to rematerialize from the now-replaced base summary.
        self.summary["stages"]["lr_log_line"]["needs_boundary_extension"] = False
        self.summary["stages"]["lr_log_line"]["recommended_experiment"] = resolved[
            "variants"
        ][0]["experiment"]
        self.summary["stages"]["lr_log_line"]["recommended_run_id"] = "extension-run"
        self.summary["stage_lock"] = {
            "lock_payload_sha256": lock["lock_payload_sha256"],
            "effective_stage": "lr_log_line",
        }
        self._write_summary()
        replayed = materializer.materialize_boundary_extension_lock(
            plan_path=self.plan_path,
            summary_path=self.summary_path,
            artifacts_dir=self.root / "missing-artifacts",
            source_stage="lr_log_line",
            output_path=output_path,
        )
        self.assertEqual(replayed, lock)

    def test_coordinate_boundary_detection_uses_declared_level_once(self) -> None:
        rows = [
            {
                "stage": "regularization_coordinate_search__weight_decay",
                "experiment": "anchor",
                "role": "stage_anchor",
                "is_hypothesis": False,
                "hypothesis_family_size": 3,
                "completed": True,
                "iid_macro_ap": 0.800,
                "iid_delta": 0.0,
                "iid_delta_vs_anchor": 0.0,
                "iid_p_holm_stage": 1.0,
                "run_id": "anchor-run",
                "axis": "weight_decay",
                "level": 0.01,
                "conditional_extension_levels": [0.1],
                "planned_overrides": {"weight_decay": 0.01},
            },
            {
                "stage": "regularization_coordinate_search__weight_decay",
                "experiment": "zero",
                "role": "candidate",
                "is_hypothesis": True,
                "hypothesis_family_size": 3,
                "completed": True,
                "iid_macro_ap": 0.801,
                "iid_delta": 0.001,
                "iid_delta_vs_anchor": 0.001,
                "iid_p_holm_stage": 0.2,
                "run_id": "zero-run",
                "axis": "weight_decay",
                "level": 0.0,
                "conditional_extension_levels": [0.1],
                "planned_overrides": {"weight_decay": 0.0},
            },
            {
                "stage": "regularization_coordinate_search__weight_decay",
                "experiment": "high-edge",
                "role": "candidate",
                "is_hypothesis": True,
                "hypothesis_family_size": 3,
                "completed": True,
                "iid_macro_ap": 0.806,
                "iid_delta": 0.006,
                "iid_delta_vs_anchor": 0.006,
                "iid_p_holm_stage": 0.03,
                "run_id": "high-run",
                "axis": "weight_decay",
                "level": 0.05,
                "conditional_extension_levels": [0.1],
                "planned_overrides": {"weight_decay": 0.05},
            },
        ]
        decision = stage_summary(
            pd.DataFrame(rows),
            tie_margin=0.002,
            control_gate=self.plan["control_gate"],
        )["regularization_coordinate_search__weight_decay"]
        self.assertTrue(decision["needs_boundary_extension"])
        extension = decision["recommended_extension"]
        self.assertEqual(extension["axis"], "weight_decay")
        self.assertEqual(extension["direction"], "higher")
        self.assertEqual(extension["level"], 0.1)
        self.assertAlmostEqual(extension["gain_over_nearest_interior"], 0.006)

        for row in rows:
            row["extension_consumed_axis"] = "weight_decay"
            row["stage_lock_transition_kind"] = "conditional_boundary_extension"
        consumed = stage_summary(
            pd.DataFrame(rows),
            tie_margin=0.002,
            control_gate=self.plan["control_gate"],
        )["regularization_coordinate_search__weight_decay"]
        self.assertFalse(consumed["needs_boundary_extension"])
        self.assertEqual(summarizer._json_cell([0.1]), [0.1])
        self.assertIsNone(summarizer._json_cell(pd.NA))

    def test_lock_drives_only_resolved_notebooks_and_launcher_variants(self) -> None:
        self._materialize()
        plan = builder.load_plan(self.plan_path)
        lock = builder.load_stage_lock(self.lock_path, plan=plan)

        output_dir = self.root / "notebooks"
        built = builder.build_campaign(
            plan_path=self.plan_path,
            output_dir=output_dir,
            stage_lock_path=self.lock_path,
        )
        self.assertEqual(
            {row["experiment"] for row in built},
            {
                "minilm5_sft_e2_lr1e5_v1",
                "minilm5_sft_e3_lr1e5_v1",
            },
        )
        self.assertTrue(
            all(row["stage_lock_payload_sha256"] == lock["lock_payload_sha256"] for row in built)
        )
        notebook = nbformat.read(built[0]["notebook"], as_version=4)
        metadata = notebook.metadata["product_matching_training"]
        self.assertEqual(metadata["campaign_stage"], "epoch_line")
        self.assertEqual(
            metadata["stage_lock"]["parent"]["run_id"], self.run_id
        )
        notes = json.loads(built[0]["expected_notes"])
        self.assertEqual(notes["stage_parent"]["run_id"], self.run_id)
        self.assertEqual(
            notes["stage_lock_payload_sha256"], lock["lock_payload_sha256"]
        )

        launched = launcher.campaign_variants(
            plan,
            stage=None,
            only=None,
            stage_lock=lock,
        )
        self.assertEqual(
            [row["experiment"] for row in launched],
            [row["experiment"] for row in lock["resolved_stage"]["variants"]],
        )
        self.assertTrue(all(row["role"] == "candidate" for row in launched))
        self.assertTrue(all(row["hypothesis_family_size"] == 3 for row in launched))

    def test_lock_summary_reuses_anchor_but_excludes_it_from_holm_family(self) -> None:
        lock = self._materialize()
        plan = builder.load_plan(self.plan_path)
        lock = builder.load_stage_lock(self.lock_path, plan=plan)
        entries = expected_entries(plan, None, stage_lock=lock)
        self.assertEqual(len(entries), 3)
        anchors = [entry for entry in entries if entry["role"] == "stage_anchor"]
        self.assertEqual(len(anchors), 1)
        self.assertFalse(anchors[0]["is_hypothesis"])
        self.assertEqual(anchors[0]["expected_run_id"], self.run_id)

        frame = pd.DataFrame(
            [
                {
                    "stage": "epoch_line",
                    "experiment": "anchor",
                    "role": "stage_anchor",
                    "is_hypothesis": False,
                    "hypothesis_family_size": 3,
                    "completed": True,
                    "iid_p_value": 0.9,
                    "iid_p_value_vs_anchor": 1.0,
                    "iid_macro_ap": 0.800,
                    "iid_delta": 0.01,
                    "iid_delta_vs_anchor": 0.0,
                    "run_id": "anchor-run",
                    "learning_rate": 1e-5,
                    "epochs": 1,
                    "planned_overrides": {"learning_rate": 1e-5, "epochs": 1},
                },
                {
                    "stage": "epoch_line",
                    "experiment": "epoch-2",
                    "role": "candidate",
                    "is_hypothesis": True,
                    "hypothesis_family_size": 3,
                    "completed": True,
                    "iid_p_value": 0.8,
                    "iid_p_value_vs_anchor": 0.01,
                    "iid_macro_ap": 0.804,
                    "iid_delta": 0.014,
                    "iid_delta_vs_anchor": 0.004,
                    "run_id": "epoch-2-run",
                    "learning_rate": 1e-5,
                    "epochs": 2,
                    "planned_overrides": {"learning_rate": 1e-5, "epochs": 2},
                },
                {
                    "stage": "epoch_line",
                    "experiment": "epoch-3",
                    "role": "candidate",
                    "is_hypothesis": True,
                    "hypothesis_family_size": 3,
                    "completed": True,
                    "iid_p_value": 0.7,
                    "iid_p_value_vs_anchor": 0.04,
                    "iid_macro_ap": 0.801,
                    "iid_delta": 0.011,
                    "iid_delta_vs_anchor": 0.001,
                    "run_id": "epoch-3-run",
                    "learning_rate": 1e-5,
                    "epochs": 3,
                    "planned_overrides": {"learning_rate": 1e-5, "epochs": 3},
                },
            ]
        )
        adjusted = add_stage_holm(frame)
        by_experiment = adjusted.set_index("experiment")
        self.assertTrue(pd.isna(by_experiment.at["anchor", "iid_p_holm_stage"]))
        self.assertAlmostEqual(
            by_experiment.at["epoch-2", "iid_p_holm_stage"], 0.03
        )
        self.assertAlmostEqual(
            by_experiment.at["epoch-3", "iid_p_holm_stage"], 0.08
        )
        decision = stage_summary(
            adjusted,
            tie_margin=0.002,
            control_gate=plan["control_gate"],
        )["epoch_line"]
        self.assertEqual(decision["anchor_reuse"], "validated")
        self.assertEqual(decision["recommended_experiment"], "epoch-2")
        self.assertAlmostEqual(
            decision["best_candidate_gain_over_anchor"], 0.004
        )

    def test_generator_rejects_rehashed_lock_with_unsafe_resolved_drift(self) -> None:
        lock = self._materialize()
        lock["resolved_stage"]["variants"][0]["resolved_config"][
            "sampling"
        ] = "category_label"
        unhashed = dict(lock)
        unhashed.pop("lock_payload_sha256")
        lock["lock_payload_sha256"] = materializer.canonical_sha256(unhashed)
        self.lock_path.write_text(
            materializer.canonical_json_dumps(lock) + "\n",
            encoding="utf-8",
        )

        with self.assertRaises(builder.CampaignConfigError):
            builder.load_stage_lock(
                self.lock_path,
                plan=builder.load_plan(self.plan_path),
            )

    def test_generator_rejects_rehashed_out_of_order_transition(self) -> None:
        lock = self._materialize()
        lock["source_stage"] = "epoch_line"
        unhashed = dict(lock)
        unhashed.pop("lock_payload_sha256")
        lock["lock_payload_sha256"] = materializer.canonical_sha256(unhashed)
        self.lock_path.write_text(
            materializer.canonical_json_dumps(lock) + "\n", encoding="utf-8"
        )

        with self.assertRaisesRegex(builder.CampaignConfigError, "Out-of-order"):
            builder.load_stage_lock(
                self.lock_path,
                plan=builder.load_plan(self.plan_path),
            )

    def test_initial_control_anchor_comparison_is_cached_by_parquet_hash(self) -> None:
        artifacts = self.root / "comparison-artifacts"
        target = [0, 0, 1, 1, 0, 0, 1, 1]
        control_scores = [0.8, 0.7, 0.6, 0.5, 0.8, 0.7, 0.6, 0.5]
        candidate_scores = [0.1, 0.2, 0.9, 0.8, 0.1, 0.2, 0.9, 0.8]
        for slug, scores in (
            ("control", control_scores),
            ("candidate", candidate_scores),
        ):
            model_dir = artifacts / slug / "model"
            model_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "id1": range(0, 16, 2),
                    "id2": range(1, 17, 2),
                    "target": target,
                    "category": ["one"] * 8,
                    "score": scores,
                }
            ).to_parquet(
                model_dir / "iid_validation_predictions.parquet",
                index=False,
            )
        frame = pd.DataFrame(
            [
                {
                    "stage": "lr_log_line",
                    "kernel_slug": "control",
                    "role": "current_protocol_control",
                    "completed": True,
                    "iid_macro_ap": average_precision_score(
                        target, control_scores
                    ),
                },
                {
                    "stage": "lr_log_line",
                    "kernel_slug": "candidate",
                    "role": "candidate",
                    "completed": True,
                    "iid_macro_ap": average_precision_score(
                        target, candidate_scores
                    ),
                },
            ]
        )
        cache_dir = self.root / "anchor-cache"
        compared = summarizer.add_anchor_relative_iid(
            frame,
            artifacts_dir=artifacts,
            permutations=9,
            bootstrap_resamples=9,
            seed=7,
            cache_dir=cache_dir,
        )
        self.assertGreater(compared.iloc[1]["iid_delta_vs_anchor"], 0)
        self.assertEqual(len(list(cache_dir.glob("*.json"))), 1)
        with mock.patch.object(
            summarizer,
            "compare_prediction_frames",
            side_effect=AssertionError("cache was not reused"),
        ):
            repeated = summarizer.add_anchor_relative_iid(
                frame,
                artifacts_dir=artifacts,
                permutations=9,
                bootstrap_resamples=9,
                seed=7,
                cache_dir=cache_dir,
            )
        self.assertEqual(
            repeated.iloc[1]["iid_p_value_vs_anchor"],
            compared.iloc[1]["iid_p_value_vs_anchor"],
        )

    def test_completed_stage_snapshot_refuses_conflicting_overwrite(self) -> None:
        output_dir = self.root / "summary-output"
        incomplete = {
            "runs.csv": "run\nfirst\n",
            "report.md": "in progress\n",
            "summary.json": json.dumps(
                {"stages": {"epoch_line": {"complete": False}}}
            ),
        }
        summarizer.write_stage_snapshot(
            output_dir,
            effective_stage="epoch_line",
            files=incomplete,
        )
        complete = {
            "runs.csv": "run\nfinal\n",
            "report.md": "complete\n",
            "summary.json": json.dumps(
                {
                    "stages": {
                        "epoch_line": {
                            "complete": True,
                            "decision_status": "ready",
                            "needs_boundary_extension": False,
                        }
                    }
                }
            ),
        }
        summarizer.write_stage_snapshot(
            output_dir,
            effective_stage="epoch_line",
            files=complete,
        )
        summarizer.write_stage_snapshot(
            output_dir,
            effective_stage="epoch_line",
            files=complete,
        )
        changed = dict(complete)
        changed["report.md"] = "changed\n"
        with self.assertRaises(RuntimeError):
            summarizer.write_stage_snapshot(
                output_dir,
                effective_stage="epoch_line",
                files=changed,
            )


if __name__ == "__main__":
    unittest.main()
