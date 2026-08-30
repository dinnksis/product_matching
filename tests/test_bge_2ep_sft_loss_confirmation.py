from __future__ import annotations

import contextlib
import io
import json
import math
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

import nbformat as nbf
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import create_bge_2ep_sft_candidate_notebooks as candidate_builder
import create_bge_2ep_sft_loss_confirmation_notebooks as builder
import create_bge_2ep_sft_notebooks as baseline_builder
import create_cross_encoder_training_notebook as cross_builder
import run_bge_2ep_sft_loss_confirmation as runner


class BgeLossConfirmationPolicyTests(unittest.TestCase):
    def test_policy_is_narrow_and_plan_only(self) -> None:
        policy = builder.load_policy()
        self.assertEqual(policy["workflow"], builder.WORKFLOW)
        self.assertEqual(policy["status"], "plan_only_default")
        self.assertEqual(policy["primary_split"], "iid")
        self.assertEqual(policy["diagnostic_splits"], ["hard"])
        self.assertEqual(
            policy["ood"],
            {
                "evaluated": False,
                "metric_sentinel": -1.0,
                "prediction_file": None,
                "comparison": None,
            },
        )
        execution = policy["execution"]
        self.assertTrue(execution["sequential"])
        for forbidden in (
            "fanout",
            "resubmit_terminal_failure",
            "ods",
            "runtime_ablation",
            "checkpoint_export",
            "checkpoint_resume",
        ):
            self.assertFalse(execution[forbidden])
        self.assertTrue(execution["append_only_attempt_ledger"])
        self.assertTrue(execution["remote_loss_prefix_audit_before_push"])
        self.assertEqual(execution["max_total_bge_kernel_slugs"], 10)
        self.assertEqual(len(execution["historical_bge_kernel_slugs_before_lr_e2"]), 4)

    def test_default_main_does_not_load_env_or_touch_kaggle(self) -> None:
        with mock.patch.object(
            runner.kaggle,
            "load_dotenv",
            side_effect=AssertionError("default plan loaded .env"),
        ), mock.patch.object(
            runner.kaggle,
            "kaggle_command",
            side_effect=AssertionError("default plan resolved Kaggle CLI"),
        ), mock.patch.object(
            runner.kaggle,
            "run_command",
            side_effect=AssertionError("default plan contacted Kaggle"),
        ), contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(runner.main([]), 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["mode"], "plan_only")
        self.assertFalse(payload["mutation"])
        self.assertEqual(payload["kernel_budget"]["hard_total_cap"], 10)

    def test_only_four_fixed_variant_specs_exist(self) -> None:
        self.assertEqual(
            set(builder.VARIANT_SPECS),
            {
                "screen_bce_s42",
                "screen_sqrt_s42",
                "confirm_bce_s17",
                "confirm_sqrt_s17",
            },
        )
        observed_losses = {
            spec["loss_variant"] for spec in builder.VARIANT_SPECS.values()
        }
        self.assertEqual(
            observed_losses,
            {builder.PLAIN_BCE, builder.SQRT_BALANCED_BCE},
        )
        with self.assertRaises(builder.LossConfirmationBuildError):
            builder.variant_spec("focal_or_other_loss")

    def test_screen_and_final_boundaries_are_exact(self) -> None:
        self.assertFalse(runner.screen_accepts_challenger(0.002))
        self.assertFalse(runner.screen_accepts_challenger(math.nextafter(0.002, 0.0)))
        self.assertTrue(runner.screen_accepts_challenger(math.nextafter(0.002, 1.0)))
        self.assertEqual(runner.confirmation_variant_keys(False), ["confirm_bce_s17"])
        self.assertEqual(
            runner.confirmation_variant_keys(True),
            ["confirm_bce_s17", "confirm_sqrt_s17"],
        )
        bce = runner.final_loss_decision(0.002, None)
        self.assertEqual(bce["selected_loss_variant"], builder.PLAIN_BCE)
        self.assertFalse(bce["challenger_accepted"])
        negative_second_seed = runner.final_loss_decision(0.003, -0.001)
        self.assertEqual(
            negative_second_seed["selected_loss_variant"], builder.PLAIN_BCE
        )
        mean_too_small = runner.final_loss_decision(0.0021, 0.0018)
        self.assertEqual(mean_too_small["selected_loss_variant"], builder.PLAIN_BCE)
        exact_mean = runner.final_loss_decision(0.0021, 0.0019)
        self.assertEqual(
            exact_mean["selected_loss_variant"], builder.SQRT_BALANCED_BCE
        )
        self.assertTrue(exact_mean["challenger_accepted"])
        with self.assertRaises(runner.LossConfirmationWorkflowError):
            runner.final_loss_decision(0.001, 0.5)
        self.assertEqual(
            runner.exact_bound_delta(0.002, 0.002, label="test"), 0.002
        )
        for adjacent in (
            math.nextafter(0.002, 0.0),
            math.nextafter(0.002, 1.0),
        ):
            with self.assertRaisesRegex(
                runner.LossConfirmationWorkflowError, "bound comparison"
            ):
                runner.exact_bound_delta(adjacent, 0.002, label="test")

    def test_budget_counts_all_four_baseline_attempts_and_caps_at_ten(self) -> None:
        historical = builder.load_policy()["execution"][
            "historical_bge_kernel_slugs_before_lr_e2"
        ]
        authority = {"entry": {"kernel_slug": historical[-1]}}
        selected = {"epoch_receipt": {"e2_entry": {"kernel_slug": "pm-b2-e2"}}}
        lr_entries = [
            {"kernel_slug": "pm-b2-lr1"},
            {"kernel_slug": "pm-b2-lr4"},
        ]
        with mock.patch.object(runner.candidate_runner, "build_lr_entries", return_value=lr_entries):
            receipt = runner.kernel_budget(
                authority=authority,
                selected=selected,
                planned_new_slugs=["pm-b2-lsqrt-111111111111-s42-l1"],
                attempted_loss_slugs_before_stage=[],
                reserve_worst_case=True,
            )
            self.assertEqual(receipt["projected_total"], 10)
            self.assertEqual(len(receipt["prior_unique_slugs"]), 7)
            accepted_branch = runner.kernel_budget(
                authority=authority,
                selected=selected,
                planned_new_slugs=[
                    "pm-b2-lsqrt-111111111111-s42-l1",
                    "pm-b2-lbce-222222222222-s17-l1",
                    "pm-b2-lsqrt-333333333333-s17-l1",
                ],
                attempted_loss_slugs_before_stage=[
                    "pm-b2-lsqrt-111111111111-s42-l1",
                    "pm-b2-lbce-222222222222-s17-l1",
                ],
                reserve_worst_case=False,
            )
            self.assertEqual(accepted_branch["projected_total"], 10)
            with self.assertRaises(runner.LossConfirmationWorkflowError):
                runner.kernel_budget(
                    authority=authority,
                    selected=selected,
                    planned_new_slugs=[
                        "pm-b2-lsqrt-111111111111-s42-l1",
                        "pm-b2-lbce-222222222222-s17-l1",
                        "pm-b2-lsqrt-333333333333-s17-l1",
                        "pm-b2-lbce-444444444444-s42-l1",
                    ],
                    attempted_loss_slugs_before_stage=[],
                    reserve_worst_case=False,
                )

    def test_old_identity_attempt_is_counted_and_blocks_replacement(self) -> None:
        historical = builder.load_policy()["execution"][
            "historical_bge_kernel_slugs_before_lr_e2"
        ]
        authority = {"entry": {"kernel_slug": historical[-1]}}
        selected = {"epoch_receipt": {"e2_entry": {"kernel_slug": "pm-b2-e2"}}}
        lr_entries = [
            {"kernel_slug": "pm-b2-lr1"},
            {"kernel_slug": "pm-b2-lr4"},
        ]
        old_slug = "pm-b2-lsqrt-aaaaaaaaaaaa-s42-l1"
        new_slug = "pm-b2-lsqrt-bbbbbbbbbbbb-s42-l1"
        with mock.patch.object(
            runner.candidate_runner, "build_lr_entries", return_value=lr_entries
        ):
            with self.assertRaisesRegex(
                runner.LossConfirmationWorkflowError, "exceed"
            ):
                runner.kernel_budget(
                    authority=authority,
                    selected=selected,
                    planned_new_slugs=[new_slug],
                    attempted_loss_slugs_before_stage=[old_slug],
                    reserve_worst_case=True,
                )

    def test_remote_old_identity_attempt_blocks_push_before_resubmit(self) -> None:
        entry = {"kernel_slug": "pm-b2-lsqrt-bbbbbbbbbbbb-s42-l1"}
        old_slug = "pm-b2-lsqrt-aaaaaaaaaaaa-s42-l1"
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            runner, "validate_staged_kernel_metadata"
        ), mock.patch.object(
            runner, "remote_loss_kernel_slugs", return_value=[old_slug]
        ), mock.patch.object(
            runner.baseline_launcher, "confirm_remote_absence"
        ) as absence, mock.patch.object(
            runner.kaggle, "run_command"
        ) as push:
            with self.assertRaisesRegex(
                runner.LossConfirmationWorkflowError, "Remote prior loss attempts"
            ):
                runner.push_after_final_gates(
                    ["kaggle"],
                    kernel_ref=f"{runner.OWNER}/{entry['kernel_slug']}",
                    entry=entry,
                    authority={},
                    run_timeout=600,
                    expected_prior_loss_slugs=[],
                    attempt_ledger_path=Path(temp_dir) / "attempts.json",
                )
        absence.assert_not_called()
        push.assert_not_called()

    def test_push_intent_ledger_is_append_only_even_if_remote_push_never_happens(self) -> None:
        first = "pm-b2-lsqrt-aaaaaaaaaaaa-s42-l1"
        replacement = "pm-b2-lsqrt-bbbbbbbbbbbb-s42-l1"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "attempts.json"
            runner.record_kernel_push_intent(
                path=path,
                expected_prior_slugs=[],
                kernel_slug=first,
            )
            self.assertEqual(runner.load_attempt_ledger(path), [first])
            with self.assertRaisesRegex(
                runner.LossConfirmationWorkflowError, "replacement/resubmission"
            ):
                runner.record_kernel_push_intent(
                    path=path,
                    expected_prior_slugs=[],
                    kernel_slug=replacement,
                )

    def test_epoch_selection_cannot_follow_a_rehashed_tampered_summary(self) -> None:
        e1 = {"run_id": "1" * 32, "experiment": "e1"}
        e2 = {"run_id": "2" * 32, "experiment": "e2"}
        comparison = {
            "splits": {
                "iid": {"delta_macro_average_precision": 0.003},
                "hard": {"delta_macro_average_precision": -0.9},
                "ood": runner.comparator._ood_result(),
            }
        }
        summary = {
            "splits": deepcopy(comparison["splits"]),
            "selection": {
                "selected_experiment": "e2",
                "selected_run_id": "2" * 32,
                "selected_epoch": 2,
                "rule": "select e2 iff paired IID delta > 0.002",
            },
        }
        selected_e2, selection = runner._epoch_selection_from_bound_comparison(
            summary=summary,
            comparison=comparison,
            e1_parent=e1,
            e2_parent=e2,
        )
        self.assertTrue(selected_e2)
        self.assertEqual(selection, summary["selection"])

        # Simulates changing summary+selection and then updating the receipt's
        # summary SHA while leaving the Sheets-bound comparison untouched.
        tampered = deepcopy(summary)
        tampered["splits"]["iid"]["delta_macro_average_precision"] = -0.003
        tampered["selection"] = {
            "selected_experiment": "e1",
            "selected_run_id": "1" * 32,
            "selected_epoch": 1,
            "rule": "select e2 iff paired IID delta > 0.002",
        }
        with self.assertRaisesRegex(
            runner.LossConfirmationWorkflowError, "Sheets-bound comparison"
        ):
            runner._epoch_selection_from_bound_comparison(
                summary=tampered,
                comparison=comparison,
                e1_parent=e1,
                e2_parent=e2,
            )

    def test_epoch_embedded_comparison_requires_exact_candidate_and_family(self) -> None:
        anchor = {
            "run_id": "1" * 32,
            "experiment": "e1",
            "report": {
                "validation_splits": {
                    "iid": {"macro_average_precision": 0.5},
                    "hard": {"macro_average_precision": 0.4},
                }
            },
        }
        challenger = {
            "run_id": "2" * 32,
            "experiment": "e2",
            "report": {
                "validation_splits": {
                    "iid": {"macro_average_precision": 0.503},
                    "hard": {"macro_average_precision": 0.39},
                }
            },
        }

        def split_result(split: str, index: int) -> dict[str, object]:
            baseline = anchor["report"]["validation_splits"][split][
                "macro_average_precision"
            ]
            candidate = challenger["report"]["validation_splits"][split][
                "macro_average_precision"
            ]
            return {
                "examples": runner.comparator.EXPECTED_ROWS[split],
                "categories": runner.EXPECTED_COMPARISON_CATEGORIES,
                "components": 100,
                "baseline_macro_average_precision": baseline,
                "candidate_macro_average_precision": candidate,
                "delta_macro_average_precision": candidate - baseline,
                "p_value": 0.25,
                "ci95_low": -0.02,
                "ci95_high": 0.03,
                "permutations": 2_000,
                "bootstrap_resamples": 2_000,
                "seed": 42 + index,
                "p_value_holm": 0.25,
                "holm_family": runner.candidate_runner.EPOCH_FAMILY_NAME,
                "holm_family_size": 1,
            }

        iid = split_result("iid", 0)
        hard = split_result("hard", 1)
        comparison = {
            "schema_version": 1,
            "status": "ready_ood_disabled",
            "baseline_run_id": anchor["run_id"],
            "candidate_run_id": challenger["run_id"],
            "baseline_experiment": anchor["experiment"],
            "candidate_experiment": challenger["experiment"],
            "baseline_manifest_sha256": "a" * 64,
            "method": "paired_component_permutation",
            "confidence_interval_method": "paired_component_bootstrap_percentile",
            "multiple_testing_correction": (
                "holm_within_planned_candidate_family_per_split"
            ),
            "holm_family": runner.candidate_runner.EPOCH_FAMILY_NAME,
            "holm_family_members": [challenger["experiment"]],
            "primary_split": "iid",
            "diagnostic_splits": ["hard"],
            "practical_tie_margin": 0.002,
            "iid_practical_relation": "improves_beyond_margin",
            "ood_policy": "disabled_train_contaminated_no_paired_comparison",
            "splits": {
                "iid": iid,
                "hard": hard,
                "ood": runner.comparator._ood_result(),
            },
        }
        runner._validate_epoch_comparison(
            comparison,
            authority={"context": {"manifest_sha256": "a" * 64}},
            anchor=anchor,
            challenger=challenger,
        )
        tampered = deepcopy(comparison)
        tampered["holm_family"] = "another_family"
        with self.assertRaisesRegex(
            runner.LossConfirmationWorkflowError, "holm_family"
        ):
            runner._validate_epoch_comparison(
                tampered,
                authority={"context": {"manifest_sha256": "a" * 64}},
                anchor=anchor,
                challenger=challenger,
            )

    def test_loss_screen_saved_delta_is_bound_back_to_raw_reports(self) -> None:
        anchor = {
            "run_id": "1" * 32,
            "experiment": "anchor",
            "report": {
                "validation_splits": {
                    "iid": {"macro_average_precision": 0.5},
                    "hard": {"macro_average_precision": 0.4},
                }
            },
        }
        challenger = {
            "run_id": "2" * 32,
            "experiment": "challenger",
            "report": {
                "validation_splits": {
                    "iid": {"macro_average_precision": 0.504},
                    "hard": {"macro_average_precision": 0.39},
                }
            },
        }

        def result(split: str, index: int) -> dict[str, object]:
            baseline_ap = anchor["report"]["validation_splits"][split][
                "macro_average_precision"
            ]
            candidate_ap = challenger["report"]["validation_splits"][split][
                "macro_average_precision"
            ]
            return {
                "examples": runner.comparator.EXPECTED_ROWS[split],
                "categories": runner.EXPECTED_COMPARISON_CATEGORIES,
                "components": 100,
                "baseline_macro_average_precision": baseline_ap,
                "candidate_macro_average_precision": candidate_ap,
                "delta_macro_average_precision": candidate_ap - baseline_ap,
                "p_value": 0.25,
                "ci95_low": -0.02,
                "ci95_high": 0.03,
                "permutations": 2_000,
                "bootstrap_resamples": 2_000,
                "seed": 42 + index,
                "p_value_holm": 0.25,
                "holm_family": runner.SCREEN_FAMILY,
                "holm_family_size": 1,
            }

        comparison = {
            "schema_version": 1,
            "status": "ready_ood_disabled",
            "baseline_run_id": anchor["run_id"],
            "candidate_run_id": challenger["run_id"],
            "baseline_experiment": anchor["experiment"],
            "candidate_experiment": challenger["experiment"],
            "baseline_manifest_sha256": "a" * 64,
            "method": "paired_component_permutation",
            "confidence_interval_method": "paired_component_bootstrap_percentile",
            "multiple_testing_correction": (
                "holm_within_planned_candidate_family_per_split"
            ),
            "holm_family": runner.SCREEN_FAMILY,
            "holm_family_members": [challenger["experiment"]],
            "primary_split": "iid",
            "diagnostic_splits": ["hard"],
            "practical_tie_margin": 0.002,
            "iid_practical_relation": "improves_beyond_margin",
            "ood_policy": "disabled_train_contaminated_no_paired_comparison",
            "splits": {
                "iid": result("iid", 0),
                "hard": result("hard", 1),
                "ood": runner.comparator._ood_result(),
            },
        }
        runner.validate_loss_comparison(
            comparison,
            anchor=anchor,
            challenger=challenger,
            family=runner.SCREEN_FAMILY,
            baseline_manifest_sha256="a" * 64,
        )
        tampered = deepcopy(comparison)
        tampered["splits"]["iid"]["candidate_macro_average_precision"] = 0.51
        tampered["splits"]["iid"]["delta_macro_average_precision"] = 0.01
        with self.assertRaisesRegex(
            runner.LossConfirmationWorkflowError, "raw reports"
        ):
            runner.validate_loss_comparison(
                tampered,
                anchor=anchor,
                challenger=challenger,
                family=runner.SCREEN_FAMILY,
                baseline_manifest_sha256="a" * 64,
            )

    def test_local_receipts_have_separate_non_authority_paths(self) -> None:
        self.assertNotEqual(
            runner.LOCAL_SCREEN_RECEIPT_FILENAME,
            runner.SCREEN_RECEIPT_FILENAME,
        )
        self.assertNotEqual(
            runner.LOCAL_FINAL_RECEIPT_FILENAME,
            runner.FINAL_RECEIPT_FILENAME,
        )
        self.assertTrue(runner.LOCAL_SCREEN_RECEIPT_FILENAME.endswith(".local.json"))

    def test_terminal_failure_is_never_resubmitted(self) -> None:
        terminal = next(iter(runner.kaggle.TERMINAL_FAILURE))
        entry = {"kernel_slug": "pm-b2-terminal-loss"}
        with mock.patch.object(runner, "_local_output", return_value=None), mock.patch.object(
            runner.baseline_launcher, "remote_kernel_status", return_value=terminal
        ), mock.patch.object(runner.kaggle, "run_command") as logs, mock.patch.object(
            runner, "push_after_final_gates"
        ) as push:
            with self.assertRaises(runner.LossConfirmationWorkflowError):
                runner.execute_entry(
                    cli=["kaggle"],
                    env_file=ROOT / ".env",
                    authority={},
                    entry=entry,
                    poll_interval=5,
                    wait_timeout=60,
                    run_timeout=600,
                    full_download=False,
                    expected_prior_loss_slugs=[],
                )
        push.assert_not_called()
        logs.assert_called_once()

    def test_private_baseline_gate_is_last_check_before_push(self) -> None:
        events: list[str] = []
        entry = {
            "kernel_slug": "pm-b2-lsqrt-bbbbbbbbbbbb-s42-l1",
            "baseline_context": {},
        }
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            runner,
            "validate_staged_kernel_metadata",
            side_effect=lambda value: events.append("staged"),
        ), mock.patch.object(
            runner,
            "audit_prior_attempts_before_push",
            side_effect=lambda *args, **kwargs: events.append("loss_audit"),
        ), mock.patch.object(
            runner.baseline_launcher,
            "confirm_remote_absence",
            side_effect=lambda cli, ref: events.append("absence"),
        ), mock.patch.object(
            runner.candidate_runner,
            "verify_remote_baseline_dataset",
            side_effect=lambda cli, authority: events.append("private_dataset"),
        ), mock.patch.object(
            runner.kaggle,
            "run_command",
            side_effect=lambda *args, **kwargs: (
                events.append("push")
                or subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
            ),
        ), mock.patch.object(
            runner.candidate_runner,
            "verify_remote_candidate_sources",
            side_effect=lambda *args, **kwargs: events.append("remote_sources"),
        ), mock.patch.object(runner.kaggle, "STAGE_ROOT", ROOT / ".kaggle" / "staging"):
            runner.push_after_final_gates(
                ["kaggle"],
                kernel_ref=f"alexproger23/{entry['kernel_slug']}",
                entry=entry,
                authority={},
                run_timeout=600,
                expected_prior_loss_slugs=[],
                attempt_ledger_path=Path(temp_dir) / "attempts.json",
            )
        self.assertEqual(
            events,
            [
                "staged",
                "loss_audit",
                "absence",
                "loss_audit",
                "private_dataset",
                "push",
                "remote_sources",
            ],
        )

    def test_ood_prediction_is_rejected_before_other_output_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            (directory / "ood_validation_predictions.parquet").write_bytes(b"forbidden")
            with self.assertRaisesRegex(
                runner.LossConfirmationWorkflowError, "forbidden OOD"
            ):
                runner.validate_loss_output(directory, entry={})

    def test_checkpoint_state_is_rejected_before_other_output_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            (directory / "optimizer.pt").write_bytes(b"forbidden")
            with self.assertRaisesRegex(
                runner.LossConfirmationWorkflowError, "forbidden checkpoint"
            ):
                runner.validate_loss_output(directory, entry={})


class BgeBalancedLossTests(unittest.TestCase):
    def test_validation_comparison_splits_have_exactly_eighteen_categories(self) -> None:
        human = ROOT / "prepared" / "validation_splits_v1" / "human"
        items = pd.read_parquet(human / "items.parquet", columns=["id", "category"])
        category_by_id = items.set_index("id")["category"]
        for filename in (
            "iid_validation_pairs.parquet",
            "hard_validation_pairs.parquet",
        ):
            pairs = pd.read_parquet(human / filename, columns=["id1"])
            self.assertEqual(
                pairs["id1"].map(category_by_id).nunique(),
                runner.EXPECTED_COMPARISON_CATEGORIES,
            )

    def test_transferred_weight_formula_matches_exact_combined_train(self) -> None:
        human = ROOT / "prepared" / "validation_splits_v1" / "human"
        train = pd.read_parquet(human / "train_pairs.parquet")
        former_ood = pd.read_parquet(human / "ood_validation_pairs.parquet")
        items = pd.read_parquet(human / "items.parquet", columns=["id", "category"])
        category_by_id = items.set_index("id")["category"]
        train["category_1"] = train["id1"].map(category_by_id)
        former_ood["category_1"] = former_ood["id1"].map(category_by_id)
        combined = pd.concat([train, former_ood], ignore_index=True)
        namespace: dict[str, object] = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(builder.BALANCED_CATEGORY_CLASS_SQRT_BCE_SOURCE, namespace)
            namespace["initialize_loss"](
                train_frame=combined,
                device=torch.device("cpu"),
                rank=0,
                world_size=2,
            )
        weights = namespace["_PAIR_BALANCE_WEIGHTS"]
        self.assertIsInstance(weights, torch.Tensor)
        self.assertEqual(len(weights), baseline_builder.EXPECTED_TRAIN)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertAlmostEqual(
            float(weights.min()), builder.EXPECTED_BALANCE_WEIGHT_MIN, places=6
        )
        self.assertAlmostEqual(
            float(weights.max()), builder.EXPECTED_BALANCE_WEIGHT_MAX, places=6
        )
        self.assertEqual(namespace["LOSS_VARIANT"], builder.SQRT_BALANCED_BCE)

    def test_hook_hashes_are_distinct_and_plain_bce_is_frozen(self) -> None:
        self.assertEqual(
            builder.LOSS_HOOK_SHA256[builder.PLAIN_BCE],
            baseline_builder.FIXED_LOSS_HOOK_SHA256,
        )
        self.assertNotEqual(
            builder.LOSS_HOOK_SHA256[builder.PLAIN_BCE],
            builder.LOSS_HOOK_SHA256[builder.SQRT_BALANCED_BCE],
        )
        source = builder.BALANCED_CATEGORY_CLASS_SQRT_BCE_SOURCE
        self.assertIn("EXPECTED_CATEGORIES = 20", source)
        self.assertIn("EXPECTED_STRATA = 40", source)
        self.assertNotIn("FOCAL", source)


class BgeLossNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.validation = baseline_builder.load_validation_dataset(
            baseline_builder.DEFAULT_SOURCE_DIR,
            runner.OWNER,
        )
        cls.checkpoint = baseline_builder.load_checkpoint_dataset(
            baseline_builder.DEFAULT_CHECKPOINT_STAGE_DIR,
            runner.OWNER,
            verify_payload=True,
        )
        cls.base_config = cross_builder.load_training_config(
            baseline_builder.DEFAULT_CONFIG
        )
        cls.baseline_notebook, cls.baseline_entry = baseline_builder.build_variant_notebook(
            validation=cls.validation,
            checkpoint=cls.checkpoint,
            base_config=cls.base_config,
            variant=baseline_builder.VARIANT_SPECS[0],
        )
        binding = {
            "baseline_run_id": "a" * 32,
            "baseline_experiment": cls.baseline_entry["experiment"],
            "campaign": baseline_builder.CAMPAIGN,
            "campaign_identity_sha256": cls.baseline_entry["identity_sha256"],
            "source_sha256": cls.baseline_entry["source_sha256"],
            "recipe_sha256": cls.baseline_entry["recipe_sha256"],
            "executable_cells_sha256": cls.baseline_entry["executable_cells_sha256"],
            "loss_hook_sha256": cls.baseline_entry["loss_hook_sha256"],
            "checkpoint_manifest_sha256": cls.baseline_entry[
                "checkpoint_manifest_sha256"
            ],
            "checkpoint_model_sha256": cls.baseline_entry["checkpoint_model_sha256"],
            "validation_manifest_sha256": cls.baseline_entry[
                "validation_manifest_sha256"
            ],
        }
        manifest = {
            "schema_version": 1,
            "dataset": f"{runner.OWNER}/product-matching-bge-2ep-sft-baseline-v1",
            "is_private": True,
            "evaluated_splits": ["iid", "hard"],
            "ood": {
                "evaluated": False,
                "metric_sentinel": -1.0,
                "comparison": None,
                "prediction_file": None,
            },
            "files": {
                "notebook_completed.json": {"bytes": 1, "sha256": "b" * 64},
                "iid_validation_predictions.parquet": {
                    "bytes": 1,
                    "sha256": "c" * 64,
                },
                "hard_validation_predictions.parquet": {
                    "bytes": 1,
                    "sha256": "d" * 64,
                },
            },
            "binding": binding,
        }
        cls.context = candidate_builder.validate_baseline_context(
            {
                "dataset_ref": manifest["dataset"],
                "dataset_slug": "product-matching-bge-2ep-sft-baseline-v1",
                "dataset_version": 1,
                "manifest_sha256": "e" * 64,
                "manifest_canonical_sha256": candidate_builder.canonical_sha256(manifest),
                "manifest": manifest,
                "binding": binding,
            }
        )
        cls.parent = builder.validate_parent_receipt(
            {
                "run_id": binding["baseline_run_id"],
                "experiment": binding["baseline_experiment"],
                "campaign_identity_sha256": binding["campaign_identity_sha256"],
                "source_sha256": binding["source_sha256"],
                "recipe_sha256": binding["recipe_sha256"],
                "checkpoint_manifest_sha256": binding[
                    "checkpoint_manifest_sha256"
                ],
                "checkpoint_model_sha256": binding["checkpoint_model_sha256"],
                "validation_manifest_sha256": binding[
                    "validation_manifest_sha256"
                ],
                "loss_hook_sha256": binding["loss_hook_sha256"],
                "config": deepcopy(cls.baseline_entry["expected_config"]),
            },
            require_seed=42,
            require_plain_bce=True,
        )

    def _build(self, key: str):
        return builder.build_notebook(
            validation=self.validation,
            checkpoint=self.checkpoint,
            base_config=self.base_config,
            spec=builder.variant_spec(key),
            parent=self.parent,
            baseline_context=self.context,
        )

    def test_seed42_screen_changes_only_loss(self) -> None:
        notebook, entry = self._build("screen_sqrt_s42")
        self.assertEqual(entry["expected_config"], self.parent["config"])
        self.assertEqual(entry["seed"], 42)
        self.assertEqual(entry["loss_variant"], builder.SQRT_BALANCED_BCE)
        self.assertNotEqual(entry["loss_hook_sha256"], self.parent["loss_hook_sha256"])
        self.assertTrue(entry["fresh_start"])
        self.assertFalse(entry["checkpoint_resume"])
        validated = builder.validate_notebook(notebook, entry=entry)
        self.assertEqual(validated["loss_variant"], builder.SQRT_BALANCED_BCE)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"loss-notebook-cell-{index}", "exec")
        metadata = notebook.metadata["product_matching_training"]
        self.assertEqual(metadata["primary_split"], "iid")
        self.assertEqual(metadata["diagnostic_splits"], ["hard"])
        self.assertEqual(metadata["ood_metric_sentinel"], -1)
        self.assertIsNone(metadata["ood_comparison"])

    def test_seed17_bce_changes_only_seed(self) -> None:
        _, entry = self._build("confirm_bce_s17")
        expected = deepcopy(self.parent["config"])
        expected["seed"] = 17
        self.assertEqual(entry["expected_config"], expected)
        changed = {
            key
            for key in expected
            if expected[key] != self.parent["config"][key]
        }
        self.assertEqual(changed, {"seed"})
        self.assertEqual(entry["loss_variant"], builder.PLAIN_BCE)
        self.assertEqual(entry["loss_hook_sha256"], self.parent["loss_hook_sha256"])

    def test_seed17_challenger_requires_matched_bce_parent(self) -> None:
        with self.assertRaisesRegex(
            builder.LossConfirmationBuildError, "parent seed differs"
        ):
            self._build("confirm_sqrt_s17")
        matched_parent = deepcopy(self.parent)
        matched_parent["run_id"] = "f" * 32
        matched_parent["experiment"] = builder.VARIANT_SPECS[
            "confirm_bce_s17"
        ]["experiment"]
        matched_parent["config"]["seed"] = 17
        matched_parent["recipe_sha256"] = builder.canonical_sha256(
            matched_parent["config"]
        )
        notebook, entry = builder.build_notebook(
            validation=self.validation,
            checkpoint=self.checkpoint,
            base_config=self.base_config,
            spec=builder.variant_spec("confirm_sqrt_s17"),
            parent=matched_parent,
            baseline_context=self.context,
        )
        self.assertEqual(entry["expected_config"], matched_parent["config"])
        self.assertEqual(entry["parent"]["run_id"], "f" * 32)
        builder.validate_notebook(notebook, entry=entry)

    def test_build_is_deterministic_and_binds_workflow_sources(self) -> None:
        notebook_a, entry_a = self._build("screen_sqrt_s42")
        notebook_b, entry_b = self._build("screen_sqrt_s42")
        self.assertEqual(entry_a, entry_b)
        self.assertEqual(
            nbf.writes(notebook_a, version=4),
            nbf.writes(notebook_b, version=4),
        )
        ledger, digest = builder.workflow_source_ledger()
        self.assertEqual(entry_a["workflow_source_ledger"], ledger)
        self.assertEqual(entry_a["workflow_ledger_sha256"], digest)

    def test_tampered_loss_metadata_and_parent_coordinate_fail(self) -> None:
        notebook, entry = self._build("screen_sqrt_s42")
        tampered_notebook = deepcopy(notebook)
        tampered_notebook.metadata["product_matching_training"]["loss_variant"] = "focal"
        with self.assertRaises(builder.LossConfirmationBuildError):
            builder.validate_notebook(tampered_notebook, entry=entry)
        tampered_parent = deepcopy(self.parent)
        tampered_parent["config"]["weight_decay"] = 0.05
        tampered_parent["recipe_sha256"] = builder.canonical_sha256(
            tampered_parent["config"]
        )
        with self.assertRaisesRegex(
            builder.LossConfirmationBuildError, "weight_decay"
        ):
            builder.validate_parent_receipt(tampered_parent)

    def test_runner_binds_exact_four_private_datasets(self) -> None:
        notebook, entry = self._build("screen_sqrt_s42")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "loss.ipynb"
            nbf.write(notebook, path)
            entry["notebook"] = str(path)
            command = runner.runner_command(entry, env_file=ROOT / ".env")
        attached = [command[index + 1] for index, value in enumerate(command) if value == "--dataset"]
        self.assertEqual(attached, runner.expected_dataset_sources(entry)[:3])
        self.assertEqual(
            runner.expected_dataset_sources(entry),
            [
                entry["validation_dataset"],
                entry["checkpoint_dataset"],
                self.context["dataset_ref"],
                runner.baseline_launcher.CREDENTIALS_DATASET,
            ],
        )
        self.assertIn("--no-env-sources", command)
        self.assertIn("--no-gpu-check", command)


if __name__ == "__main__":
    unittest.main()
