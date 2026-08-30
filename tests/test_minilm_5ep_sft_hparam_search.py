from __future__ import annotations

import json
from copy import deepcopy
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import nbformat
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import create_cross_encoder_training_notebook as cross_builder
import create_minilm_5ep_sft_hparam_notebooks as builder
import create_minilm_5ep_team_ablation_notebook as team_builder
import create_minilm_validation_baseline_notebook as baseline_builder
import run_minilm_5ep_sft_hparam_kaggle as launcher
from summarize_minilm_5ep_sft_hparams import (
    add_stage_holm,
    holm_adjust,
    stage_summary,
    validate_frozen_training_contract,
)


class Minilm5epSftHparamSearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = builder.load_plan()
        cls.base_config = cross_builder.load_training_config(builder.BASE_CONFIG_PATH)
        cls.dataset = baseline_builder.load_manifest(
            baseline_builder.DEFAULT_SOURCE_DIR,
            team_builder.DATASET_OWNER,
        )
        cls.variants = list(builder.ready_variants(cls.plan))

    def test_stage_one_is_a_logarithmic_lr_line(self) -> None:
        self.assertEqual(len(self.variants), 4)
        points = {
            (
                int(variant["overrides"]["epochs"]),
                float(variant["overrides"]["learning_rate"]),
            )
            for _, variant in self.variants
        }
        self.assertEqual(
            points,
            {
                (1, learning_rate)
                for learning_rate in (5e-6, 1e-5, 2e-5, 4e-5)
            },
        )
        controls = [
            variant
            for _, variant in self.variants
            if variant.get("role") == "current_protocol_control"
        ]
        self.assertEqual(len(controls), 1)
        self.assertEqual(
            controls[0]["overrides"],
            {"epochs": 1, "learning_rate": 2e-5},
        )

    def test_variant_overrides_cannot_change_frozen_data_recipe(self) -> None:
        _, variant = self.variants[0]
        config = builder.variant_config(self.base_config, self.plan, variant)
        allowed = set(self.plan["allowed_override_keys"])
        for key, value in self.base_config.items():
            if key not in allowed:
                self.assertEqual(config[key], value)
        bad = dict(variant)
        bad["overrides"] = {"sampling": "category_label"}
        with self.assertRaises(builder.CampaignConfigError):
            builder.variant_config(self.base_config, self.plan, bad)
        poisoned_plan = deepcopy(self.plan)
        poisoned_plan["allowed_override_keys"].append("sampling")
        with self.assertRaises(builder.CampaignConfigError):
            builder.variant_config(self.base_config, poisoned_plan, bad)

    def test_invalid_numeric_and_model_kwargs_fail_before_kaggle(self) -> None:
        _, original = self.variants[0]
        invalid_overrides = (
            {"epochs": 1.5},
            {"learning_rate": float("nan")},
            {"max_grad_norm": float("nan")},
            {"seed": -1},
            {"model_load_kwargs": "not-an-object"},
            {"model_load_kwargs": {"num_labels": 20}},
            {"model_load_kwargs": {"classifier_dropout": 1.0}},
        )
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                variant = dict(original)
                variant["overrides"] = overrides
                with self.assertRaises(builder.CampaignConfigError):
                    builder.variant_config(self.base_config, self.plan, variant)

    def test_generated_notebook_is_locked_and_routes_to_sft(self) -> None:
        stage, variant = next(
            (stage, variant)
            for stage, variant in self.variants
            if variant.get("role") == "current_protocol_control"
        )
        notebook = builder.build_variant_notebook(
            dataset=self.dataset,
            base_config=self.base_config,
            plan=self.plan,
            stage=stage,
            variant=variant,
        )
        nbformat.validate(notebook)
        editable = [
            cell
            for cell in notebook.cells
            if "team-editable" in cell.metadata.get("tags", [])
            or "run-configurable" in cell.metadata.get("tags", [])
        ]
        self.assertEqual(editable, [])
        routing = [
            cell
            for cell in notebook.cells
            if "experiment-routing" in cell.metadata.get("tags", [])
            and cell.cell_type == "code"
        ]
        self.assertEqual(len(routing), 1)
        self.assertIn('EXPECTED_EXPERIMENT_SHEET = "sft_exps"', routing[0].source)
        self.assertIn("EXPERIMENT_SHEET = EXPECTED_EXPERIMENT_SHEET", routing[0].source)
        recipe = next(
            cell
            for cell in notebook.cells
            if "frozen-recipe" in cell.metadata.get("tags", [])
        )
        self.assertIn("'epochs': 1", recipe.source)
        self.assertIn("'learning_rate': 2e-05", recipe.source)
        source = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        self.assertIn("def build_train_data", source)
        self.assertIn("binary_cross_entropy_with_logits", source)
        self.assertIn("SFT hyperparameter run changed frozen human train pairs", source)
        self.assertIn("Materialized SFT loss hook differs", source)
        self.assertIn("compare_experiment_directories(", source)
        self.assertIn("sync_from_kaggle_credentials(", source)
        self.assertLess(
            source.index("compare_experiment_directories("),
            source.index("sync_from_kaggle_credentials("),
        )
        metadata = notebook.metadata["product_matching_training"]
        self.assertEqual(metadata["experiment_group"], "sft")
        self.assertEqual(metadata["campaign_stage"], "lr_log_line")
        self.assertEqual(metadata["editable_cells"], [])
        self.assertNotIn("run_configurable_cell", metadata)
        self.assertEqual(
            metadata["fixed_loss_hook_sha256"], builder.FIXED_LOSS_HOOK_SHA256
        )

    def test_notes_contain_regularization_fields_missing_from_compact_sheet(self) -> None:
        stage, variant = self.variants[0]
        config = builder.variant_config(self.base_config, self.plan, variant)
        notes = json.loads(
            builder._variant_notes(self.plan["campaign"], stage, variant, config)
        )
        for key in (
            "weight_decay",
            "warmup_ratio",
            "label_smoothing",
            "max_grad_norm",
            "effective_batch",
        ):
            self.assertIn(key, notes)

    def test_special_losses_are_allowlisted_frozen_hooks(self) -> None:
        loss_stage = next(
            stage
            for stage in self.plan["stages"]
            if stage["name"] == "special_loss_screen"
        )
        planned = loss_stage["loss_variants"]
        combinations = list(
            loss_stage["conditional_combination"]["variants_by_balance"].values()
        )
        self.assertIn("balanced_category_class_bce", planned)
        self.assertIn("focal_bce_gamma2_scale4", planned)
        self.assertIn(
            "balanced_category_class_sqrt_focal_gamma2_scale4",
            combinations,
        )
        for name in [*planned, *combinations]:
            with self.subTest(loss_variant=name):
                variant = dict(self.variants[0][1], loss_variant=name)
                resolved, source, digest = builder.variant_loss(variant)
                self.assertEqual(resolved, name)
                compile(source, f"<{name}>", "exec")
                self.assertEqual(digest, builder.LOSS_VARIANT_SHA256[name])
                namespace: dict[str, object] = {}
                exec(source, namespace)
                train_frame = pd.DataFrame(
                    {
                        "target": [0, 0, 0, 1, 0, 1],
                        "category_1": ["a", "a", "a", "a", "b", "b"],
                    }
                )
                namespace["initialize_loss"](
                    train_frame=train_frame,
                    device=torch.device("cpu"),
                    rank=0,
                    world_size=1,
                )
                if name.startswith("balanced_"):
                    weights = namespace["_PAIR_BALANCE_WEIGHTS"]
                    self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
                    self.assertGreater(float(weights.max()), float(weights.min()))
                logits = torch.tensor(
                    [-1.0, 1.0, 0.5, -0.5, 0.25, -0.25],
                    requires_grad=True,
                )
                result = namespace["compute_loss"](
                    logits=logits,
                    targets=torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 1.0]),
                    sample_weights=torch.ones(6),
                    pair_indices=torch.arange(6),
                    orientations=torch.zeros(6, dtype=torch.bool),
                    epoch=0,
                    step=0,
                )
                self.assertTrue(torch.isfinite(result["loss"]))
                result["loss"].backward()
                self.assertTrue(torch.isfinite(logits.grad).all())
        bad = dict(self.variants[0][1], loss_variant="arbitrary_python")
        with self.assertRaises(builder.CampaignConfigError):
            builder.variant_loss(bad)

    def test_registered_loss_cannot_escape_its_declared_stage(self) -> None:
        special_allowed = builder.stage_loss_allowlist(
            self.plan,
            stage_name="special_loss_screen",
        )
        self.assertIn("focal_bce_gamma2_scale4", special_allowed)
        self.assertNotIn("negative_focal_bce_gamma2", special_allowed)
        self.assertEqual(
            builder.stage_loss_allowlist(self.plan, stage_name="lr_log_line"),
            {"bce"},
        )

        for loss_variant in (
            "focal_bce_gamma2_scale4",
            "negative_focal_bce_gamma2",
            "bce_topk_ranknet_lambda025",
        ):
            with self.subTest(loss_variant=loss_variant):
                poisoned = deepcopy(self.plan)
                poisoned["stages"][0]["variants"][0][
                    "loss_variant"
                ] = loss_variant
                with self.assertRaises(builder.CampaignConfigError):
                    list(builder.ready_variants(poisoned, stage_name="lr_log_line"))

    def test_launcher_attaches_all_frozen_inputs_and_keeps_sheets_credentials(self) -> None:
        stage, variant = self.variants[0]
        entry = {
            "stage": stage,
            "experiment": variant["experiment"],
            "kernel_slug": variant["kernel_slug"],
            "title": variant["title"],
            "notebook": str(builder.output_path(builder.DEFAULT_OUTPUT_DIR, variant)),
        }
        command = launcher.runner_command(
            entry,
            env_file=ROOT / ".env",
            dry_run=True,
            no_wait=True,
        )
        joined = " ".join(command)
        for dataset in launcher.REQUIRED_DATASETS:
            self.assertIn(dataset, joined)
        self.assertNotIn("--no-google-sheets-credentials", command)
        self.assertIn("--no-env-sources", command)

    def test_launcher_validates_sft_sheet_and_paired_artifacts(self) -> None:
        entry = launcher.campaign_variants(self.plan, stage=None, only=None)[0]
        run_id = "0123456789abcdef0123456789abcdef"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            model_dir.mkdir()
            split_rows = {"iid": 12_000, "hard": 5_814, "ood": 41_171}
            split_scores = {
                split: float(entry["baseline_metrics"][f"{split}_macro_ap"])
                for split in split_rows
            }
            validation_splits = {
                split: {
                    "examples": rows,
                    "macro_average_precision": split_scores[split],
                    "overall_average_precision": min(1.0, split_scores[split] + 0.01),
                    "recall_at_precision_0_99": 0.1,
                    "roc_auc": 0.7,
                    "log_loss": 0.5,
                }
                for split, rows in split_rows.items()
            }
            report = {
                "training_seconds": 1000.0,
                "total_pipeline_seconds": 1500.0,
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
            comparison = {
                "status": "ready",
                "baseline_run_id": team_builder.SIGNIFICANCE_BASELINE_RUN_ID,
                "candidate_run_id": run_id,
                "method": "paired_component_permutation",
                "splits": {
                    split: {
                        "delta_macro_average_precision": 0.0,
                        "p_value": 1.0,
                        "p_value_holm": 1.0,
                        "ci95_low": -0.001,
                        "ci95_high": 0.001,
                    }
                    for split in split_rows
                },
            }
            completion = {
                "status": "complete",
                "run_id": run_id,
                "experiment": entry["experiment"],
                "experiment_group": "sft",
                "frozen_recipe_sha256": entry["recipe_sha256"],
                "code_bundle_sha256": entry["source_sha256"],
                "dataset_ref": launcher.VALIDATION_DATASET,
                "initial_checkpoint_ref": launcher.CHECKPOINT_DATASET,
                "initial_checkpoint_manifest_sha256": (
                    team_builder.CHECKPOINT_MANIFEST_SHA256
                ),
                "loss_hook_sha256": entry["loss_hook_sha256"],
                "notes": entry["expected_notes"],
                "train_data": {
                    "train_pairs": 306_669,
                    "items": 711_304,
                    "same_size_as_human_baseline": True,
                },
                "training_report": report,
                "baseline_comparison": comparison,
            }
            sync = {
                "status": "synced",
                "run_id": run_id,
                "experiment_group": "sft",
                "comparison_sheet": "sft_exps",
                "spreadsheet_id": (
                    launcher.qwen_builder.EXPERIMENT_SPREADSHEET_ID
                ),
            }
            (root / "notebook_completed.json").write_text(json.dumps(completion))
            (root / "baseline_comparison.json").write_text(json.dumps(comparison))
            (root / "google_sheets_sync.json").write_text(json.dumps(sync))
            (root / "experiment_run_id.txt").write_text(run_id + "\n")
            (model_dir / "training_report.json").write_text(json.dumps(report))
            config_text = json.dumps(entry["expected_config"])
            (model_dir / "training_config.json").write_text(config_text)
            (root / "cross_encoder_config.json").write_text(config_text)
            for split, rows in split_rows.items():
                pd.DataFrame(
                    {
                        "id1": range(rows),
                        "id2": range(rows, 2 * rows),
                        "target": [0] * rows,
                        "score": [0.1] * rows,
                        "category": ["test"] * rows,
                    }
                ).to_parquet(
                    model_dir / f"{split}_validation_predictions.parquet",
                    index=False,
                )
            validated = launcher.validate_run_output(root, entry=entry)
            self.assertEqual(validated["run_id"], run_id)
            bad_sync = dict(sync, comparison_sheet="data_exps")
            (root / "google_sheets_sync.json").write_text(json.dumps(bad_sync))
            with self.assertRaises(RuntimeError):
                launcher.validate_run_output(root, entry=entry)

    def test_kaggle_output_download_retries_in_fresh_staging_directories(self) -> None:
        entry = {
            "kernel_slug": "retry-test-kernel",
            "experiment": "retry_test",
        }
        responses = [
            subprocess.CompletedProcess([], 1, "proxy failure"),
            subprocess.CompletedProcess([], 1, "proxy failure"),
            subprocess.CompletedProcess([], 0, "downloaded"),
        ]
        events: list[str] = []

        def precheck(*args: object, **kwargs: object) -> None:
            events.append("precheck")

        def retry_sync(*args: object, **kwargs: object) -> None:
            events.append("retry_sync")

        def full_validation(*args: object, **kwargs: object) -> dict[str, str]:
            events.append("full_validation")
            return {"run_id": "run", "experiment": "retry_test"}

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with (
                mock.patch.object(launcher, "output_root", return_value=root),
                mock.patch.object(
                    launcher.kaggle,
                    "run_command",
                    side_effect=responses,
                ) as run_command,
                mock.patch.object(launcher.time, "sleep") as sleep,
                mock.patch.object(
                    launcher,
                    "validate_run_payload",
                    side_effect=precheck,
                ),
                mock.patch.object(
                    launcher,
                    "retry_pending_sheets_sync",
                    side_effect=retry_sync,
                ),
                mock.patch.object(
                    launcher,
                    "validate_run_output",
                    side_effect=full_validation,
                ),
            ):
                destination = launcher.download_output(
                    ["kaggle"],
                    username="owner",
                    entry=entry,
                    full_download=False,
                )
            self.assertEqual(run_command.call_count, 3)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [3, 8])
            self.assertEqual(destination, root / entry["kernel_slug"])
            self.assertTrue(destination.is_dir())
            self.assertEqual(
                events,
                ["precheck", "retry_sync", "full_validation"],
            )
            self.assertEqual(
                list(root.glob(f".{entry['kernel_slug']}.download-*")),
                [],
            )

    def test_synced_marker_requires_exact_sheets_identity(self) -> None:
        run_id = "0123456789abcdef0123456789abcdef"
        completion = {
            "status": "complete",
            "run_id": run_id,
            "experiment_group": "sft",
            "baseline_comparison": {"status": "ready"},
        }
        exact_sync = {
            "status": "synced",
            "run_id": run_id,
            "experiment_group": "sft",
            "comparison_sheet": "sft_exps",
            "spreadsheet_id": launcher.qwen_builder.EXPERIMENT_SPREADSHEET_ID,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "notebook_completed.json").write_text(
                json.dumps(completion),
                encoding="utf-8",
            )
            sync_path = root / "google_sheets_sync.json"
            sync_path.write_text(json.dumps(exact_sync), encoding="utf-8")
            launcher.retry_pending_sheets_sync(root, kernel_ref="owner/kernel")

            bad_values = {
                "run_id": "different-run",
                "experiment_group": "data",
                "comparison_sheet": "data_exps",
                "spreadsheet_id": "different-spreadsheet",
            }
            for field, value in bad_values.items():
                with self.subTest(field=field):
                    sync_path.write_text(
                        json.dumps({**exact_sync, field: value}),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(RuntimeError, "stale/mismatched"):
                        launcher.retry_pending_sheets_sync(
                            root,
                            kernel_ref="owner/kernel",
                        )

    def test_summary_rechecks_frozen_sampling_and_weight_contract(self) -> None:
        report = {
            "training_sampling": "none",
            "training_loss_weighting": "none",
            "training_subset": "all",
            "original_training_examples": 306_669,
            "training_unique_coverage_per_epoch": 1.0,
            "training_loss_weight_min": 1.0,
            "training_loss_weight_median": 1.0,
            "training_loss_weight_max": 1.0,
        }
        validate_frozen_training_contract(report, experiment="valid")
        bad_values = {
            "training_sampling": "category_label",
            "training_loss_weighting": "balanced",
            "training_subset": "sampled",
            "original_training_examples": 1,
            "training_unique_coverage_per_epoch": 0.5,
            "training_loss_weight_min": 0.5,
            "training_loss_weight_median": 0.9,
            "training_loss_weight_max": 2.0,
        }
        for field, value in bad_values.items():
            with self.subTest(field=field):
                with self.assertRaisesRegex(RuntimeError, "frozen sampling"):
                    validate_frozen_training_contract(
                        {**report, field: value},
                        experiment="invalid",
                    )

    def test_holm_adjustment_is_monotone_in_sorted_p_values(self) -> None:
        raw = [0.04, 0.001, 0.02, 0.9]
        adjusted = holm_adjust(raw)
        self.assertEqual(adjusted, [0.08, 0.004, 0.06, 0.9])
        ordered = sorted(zip(raw, adjusted))
        self.assertEqual(
            [value for _, value in ordered],
            sorted(value for _, value in ordered),
        )

    def test_incomplete_stage_uses_full_planned_holm_family_and_has_no_winner(self) -> None:
        rows = [
            {
                "stage": "grid",
                "experiment": "control",
                "role": "current_protocol_control",
                "completed": True,
                "iid_p_value": 1.0,
                "iid_macro_ap": 0.789388132774931,
                "iid_delta": 0.0,
                "hard_delta": 0.0,
                "ood_delta": 0.0,
                "total_pipeline_seconds": 1548.96508378,
                "run_id": "control-run",
                "learning_rate": 2e-5,
                "epochs": 1,
                "planned_overrides": {"epochs": 1, "learning_rate": 2e-5},
            }
        ]
        for index in range(11):
            completed = index < 2
            rows.append(
                {
                    "stage": "grid",
                    "experiment": f"candidate-{index}",
                    "role": "candidate",
                    "completed": completed,
                    "iid_p_value": (0.01, 0.02)[index] if completed else None,
                    "iid_macro_ap": 0.80 - index / 1000 if completed else None,
                    "iid_delta": 0.01 - index / 1000 if completed else None,
                    "hard_delta": 0.0 if completed else None,
                    "ood_delta": 0.0 if completed else None,
                    "total_pipeline_seconds": 1600.0 if completed else None,
                    "run_id": f"run-{index}" if completed else None,
                    "learning_rate": 1e-5,
                    "epochs": 1,
                    "planned_overrides": {"epochs": 1, "learning_rate": 1e-5},
                }
            )
        frame = add_stage_holm(pd.DataFrame(rows))
        adjusted = frame.loc[
            frame["role"].eq("candidate") & frame["completed"],
            "iid_p_holm_stage",
        ].tolist()
        self.assertEqual(adjusted, [0.11, 0.2])
        summary = stage_summary(
            frame,
            tie_margin=0.002,
            control_gate=self.plan["control_gate"],
        )["grid"]
        self.assertEqual(summary["decision_status"], "pending")
        self.assertNotIn("best_experiment", summary)
        self.assertNotIn("provisional_leader", summary)

    def test_complete_stage_retains_control_when_all_candidates_are_worse(self) -> None:
        rows = [
            {
                "stage": "grid",
                "experiment": "control",
                "role": "current_protocol_control",
                "completed": True,
                "iid_p_value": 1.0,
                "iid_macro_ap": 0.79,
                "iid_delta": 0.0,
                "hard_delta": 0.0,
                "ood_delta": 0.0,
                "total_pipeline_seconds": 1548.96508378,
                "run_id": "control-run",
                "learning_rate": 2e-5,
                "epochs": 1,
                "planned_overrides": {"epochs": 1, "learning_rate": 2e-5},
            }
        ]
        for index in range(11):
            rows.append(
                {
                    "stage": "grid",
                    "experiment": f"candidate-{index}",
                    "role": "candidate",
                    "completed": True,
                    "iid_p_value": 0.5,
                    "iid_macro_ap": 0.789 - index / 10000,
                    "iid_delta": -0.001 - index / 10000,
                    "hard_delta": 0.0,
                    "ood_delta": 0.0,
                    "total_pipeline_seconds": 1600.0,
                    "run_id": f"run-{index}",
                    "learning_rate": 1e-5,
                    "epochs": 1,
                    "planned_overrides": {
                        "epochs": 1,
                        "learning_rate": 1e-5,
                    },
                }
            )
        frame = add_stage_holm(pd.DataFrame(rows))
        summary = stage_summary(
            frame,
            tie_margin=0.002,
            control_gate=self.plan["control_gate"],
        )["grid"]
        self.assertEqual(summary["decision_status"], "ready")
        self.assertEqual(summary["recommendation"], "retain_current_recipe")
        self.assertEqual(summary["recommended_experiment"], "control")
        self.assertFalse(summary["challenger_selected"])
        self.assertFalse(summary["needs_boundary_extension"])


if __name__ == "__main__":
    unittest.main()
