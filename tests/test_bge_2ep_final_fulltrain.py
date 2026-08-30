from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import nbformat
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import create_bge_2ep_final_fulltrain_notebook as builder
import run_bge_2ep_final_fulltrain as launcher
import train_bge_2ep_final_fulltrain as trainer
from src.google_sheets_logger import (
    EXPERIMENT_HEADERS,
    build_experiment_row,
    experiment_group,
)


class FinalFulltrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.entry = builder.build_notebook(owner="alexproger23", write=True)
        cls.generated = builder.load_and_validate_notebook(
            Path(cls.entry["notebook"]), entry=cls.entry
        )

    def test_selected_recipe_is_exact(self) -> None:
        config = json.loads(builder.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config, builder.EXPECTED_CONFIG)
        self.assertEqual(builder.base.canonical_sha256(config), builder.EXPECTED_RECIPE_SHA256)
        self.assertEqual(config["epochs"], 2)
        self.assertEqual(config["learning_rate"], 2e-5)
        self.assertEqual(config["loss_weighting"], "none")
        self.assertEqual(config["seed"], 42)

    def test_assignment_authority_reconstructs_original_matches(self) -> None:
        human = ROOT / "prepared/validation_splits_v1/human"
        assignments = pd.read_parquet(human / "split_assignments.parquet")
        assignments = assignments.sort_values("human_row_id", kind="stable").reset_index(drop=True)
        self.assertTrue(
            np.array_equal(
                assignments["human_row_id"].to_numpy(),
                np.arange(builder.EXPECTED_ROWS, dtype=np.int64),
            )
        )
        original = pd.read_parquet(ROOT / "data/matches.parquet")
        pd.testing.assert_frame_equal(
            assignments[["id1", "id2", "target"]],
            original[["id1", "id2", "target"]],
            check_exact=True,
        )
        self.assertEqual(len(assignments), builder.EXPECTED_ROWS)
        self.assertEqual(int(assignments["target"].sum()), builder.EXPECTED_POSITIVES)
        for spec in builder.SPLITS:
            assigned = assignments.loc[
                assignments["split"] == spec["split"], ["id1", "id2", "target"]
            ].reset_index(drop=True)
            frozen = pd.read_parquet(human / Path(spec["relative"]).name)
            pd.testing.assert_frame_equal(assigned, frozen, check_exact=True)
            self.assertEqual(len(assigned), spec["rows"])
            self.assertEqual(int(assigned.target.sum()), spec["positives"])

    def test_notebook_identity_and_no_validation_training(self) -> None:
        checked = builder.validate_notebook_identity(
            self.generated, entry=self.entry
        )
        self.assertEqual(checked["identity_sha256"], self.entry["identity_sha256"])
        metadata = self.generated.metadata["product_matching_training"]
        self.assertFalse(metadata["quality_evaluation"])
        self.assertTrue(metadata["google_sheets_tracking"])
        self.assertEqual(metadata["experiment_group"], "sft")
        self.assertEqual(metadata["validation_splits"], [])
        training_cells = [
            cell
            for cell in self.generated.cells
            if "training-only" in cell.metadata.get("tags", [])
        ]
        self.assertEqual(len(training_cells), 1)
        self.assertNotIn("--validation-split", training_cells[0].source)
        self.assertIn("train_bge_2ep_final_fulltrain.py", training_cells[0].source)
        self.assertEqual(
            sum("sync_from_kaggle_credentials" in cell.source for cell in self.generated.cells),
            1,
        )
        self.assertEqual(
            sum(
                "unavailable_fulltrain_no_holdout" in cell.source
                for cell in self.generated.cells
            ),
            1,
        )

    def test_sheets_projection_is_explicit_sft_with_only_minus_one_metrics(self) -> None:
        report = {
            "experiment_group": "sft",
            "original_training_examples": builder.EXPECTED_ROWS,
            "args": builder.EXPECTED_CONFIG,
            "validation_splits": {
                split: dict(builder.UNAVAILABLE_METRIC)
                for split in ("iid", "hard", "ood")
            },
        }
        completion = {
            "run_id": "a" * 32,
            "status": "complete",
            "experiment_group": "sft",
            "training_report": report,
        }
        projection = dict(
            zip(
                EXPERIMENT_HEADERS,
                build_experiment_row(completion),
                strict=True,
            )
        )
        self.assertEqual(experiment_group(completion), "sft")
        for field in (
            "iid_macro_ap",
            "hard_macro_ap",
            "ood_macro_ap",
            "hard_recall_at_p99",
            "hard_roc_auc",
            "ood_log_loss",
        ):
            self.assertEqual(projection[field], -1.0)
        self.assertEqual(projection["train_pairs"], builder.EXPECTED_ROWS)

    def test_every_code_cell_compiles(self) -> None:
        for index, cell in enumerate(self.generated.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"final-fulltrain-cell-{index}", "exec")

    def test_executable_cell_tamper_is_rejected(self) -> None:
        notebook = copy.deepcopy(self.generated)
        code_cell = next(cell for cell in notebook.cells if cell.cell_type == "code")
        code_cell.source += "\n# tampered"
        with self.assertRaises(builder.FinalExportConfigError):
            builder.validate_notebook_identity(notebook, entry=self.entry)

    def test_metadata_tamper_is_rejected(self) -> None:
        notebook = copy.deepcopy(self.generated)
        notebook.metadata["product_matching_training"]["withheld_pairs"] = 1
        with self.assertRaises(builder.FinalExportConfigError):
            builder.validate_notebook_identity(notebook, entry=self.entry)

    def test_fulltrain_loss_hook_is_new_and_count_bound(self) -> None:
        self.assertNotEqual(builder.FULLTRAIN_LOSS_HOOK_SHA256, builder.base.FIXED_LOSS_HOOK_SHA256)
        self.assertIn("365654", builder.FULLTRAIN_LOSS_HOOK_SOURCE)
        self.assertIn("human_iid", builder.FULLTRAIN_LOSS_HOOK_SOURCE)
        self.assertIn("human_hard", builder.FULLTRAIN_LOSS_HOOK_SOURCE)

    def test_sample_exact_accumulation_geometry(self) -> None:
        sizes = [8] * 22_854
        sizes[7_777] = 3
        contribution = np.zeros(sum(sizes), dtype=np.float64)
        offset = 0
        for step, size in enumerate(sizes):
            denominator = trainer.accumulation_group_denominator(
                sizes, step=step, gradient_accumulation=12
            )
            contribution[offset : offset + size] = 1.0 / denominator
            offset += size
        self.assertEqual(offset, 182_827)
        for start in range(0, len(sizes), 12):
            lo = sum(sizes[:start])
            hi = sum(sizes[: start + 12])
            self.assertAlmostEqual(float(contribution[lo:hi].sum()), 1.0, places=12)
        partial_denominator = trainer.accumulation_group_denominator(
            sizes, step=7_777, gradient_accumulation=12
        )
        self.assertEqual(partial_denominator, 91)
        self.assertEqual((len(sizes) + 11) // 12, trainer.EXPECTED_UPDATES_PER_EPOCH)
        self.assertEqual(
            2 * ((len(sizes) + 11) // 12), trainer.EXPECTED_TOTAL_UPDATES
        )
        self.assertEqual(
            2 * 2 * sum(sizes),
            builder.EXPECTED_TRAINING_EXAMPLES,
        )

    def test_trainer_contract_binds_all_trajectory_fields(self) -> None:
        config = dict(builder.EXPECTED_CONFIG)
        config["model"] = "/kaggle/temp/model"
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "runtime.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            args = SimpleNamespace(
                **config,
                config=path,
                validation_split=[],
                model_load_kwarg=[],
            )
            trainer.validate_final_contract(args, world_size=2)
            args.bucket_size_multiplier = 51
            with self.assertRaises(trainer.FinalFulltrainContractError):
                trainer.validate_final_contract(args, world_size=2)

    def test_launcher_staging_attaches_exact_three_datasets(self) -> None:
        self.assertEqual(launcher.EXPECTED_VALIDATION_DATASET_VERSION, 3)
        self.assertEqual(launcher.EXPECTED_CHECKPOINT_DATASET_VERSION, 1)
        self.assertIn(
            "pm-b2-final-f8011cbe984a-s42-v1",
            launcher.TOMBSTONED_KERNEL_SLUGS,
        )
        command = launcher.runner_command(
            self.entry, env_file=ROOT / ".env", dry_run=True
        )
        self.assertNotIn("--no-google-sheets-credentials", command)
        self.assertIn("--dry-run", command)
        self.assertEqual(
            launcher.expected_dataset_sources(self.entry),
            [
                launcher.VALIDATION_DATASET,
                launcher.CHECKPOINT_DATASET,
                launcher.CREDENTIALS_DATASET,
            ],
        )

    def test_launcher_loads_frozen_notebook_without_rewriting(self) -> None:
        with mock.patch.object(
            launcher.builder,
            "build_notebook",
            return_value=self.entry,
        ) as build:
            self.assertIs(launcher.load_frozen_entry("alexproger23"), self.entry)
        build.assert_called_once_with(owner="alexproger23", write=False)

    def test_frozen_notebook_gate_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            symlink = Path(raw) / "reviewed.ipynb"
            symlink.symlink_to(Path(self.entry["notebook"]))
            with self.assertRaises(launcher.FinalExportWorkflowError):
                launcher.validate_frozen_notebook_file(symlink, entry=self.entry)

    def test_attempt_ledger_forbids_same_identity_resubmit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "ledger.json"
            launcher.reserve_single_attempt(self.entry, path=path)
            with self.assertRaises(launcher.FinalExportWorkflowError):
                launcher.reserve_single_attempt(self.entry, path=path)

    def test_attempt_ledger_requires_explicit_prior_tombstone(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "ledger.json"
            payload = launcher.empty_attempt_ledger()
            payload["attempts"].append(
                {
                    "kernel_slug": "pm-b2-final-prior-s42-v1",
                    "identity_sha256": "a" * 64,
                    "status": "failed",
                }
            )
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                launcher.FinalExportWorkflowError, "explicitly tombstoned"
            ):
                launcher.reserve_single_attempt(self.entry, path=path)

    def test_exact_ready_status_rejects_substrings(self) -> None:
        authority = {
            key: {
                "dataset_ref": ref,
                "dataset_version": 1,
            }
            for key, ref in (
                ("validation", launcher.VALIDATION_DATASET),
                ("checkpoint", launcher.CHECKPOINT_DATASET),
                ("credentials", launcher.CREDENTIALS_DATASET),
            )
        }
        for invalid in ("not ready", "unsuccessful", "ready later"):
            with self.subTest(status=invalid), mock.patch.object(
                launcher.dataset_push,
                "dataset_status",
                return_value={"status": invalid, "current_version_number": 1},
            ):
                with self.assertRaises(launcher.FinalExportWorkflowError):
                    launcher.recheck_remote_input_statuses(["kaggle"], authority)

    def test_remote_dataset_file_names_must_be_flat_and_exact(self) -> None:
        self.assertEqual(
            launcher._remote_file_names([{"name": "manifest.json"}]),
            {"manifest.json"},
        )
        for unsafe in (
            "nested/manifest.json",
            "../manifest.json",
            "/manifest.json",
            "./manifest.json",
            "nested\\manifest.json",
            ".",
            "..",
        ):
            with self.subTest(name=unsafe), self.assertRaises(
                launcher.FinalExportWorkflowError
            ):
                launcher._remote_file_names([{"name": unsafe}])

    def test_stable_reader_rejects_symlink_and_hardlink(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.bin"
            source.write_bytes(b"safe")
            symlink = root / "symlink.bin"
            symlink.symlink_to(source)
            with self.assertRaises(launcher.FinalExportWorkflowError):
                launcher.stable_regular_file_record(symlink, root=root)
            hardlink = root / "hardlink.bin"
            os.link(source, hardlink)
            with self.assertRaises(launcher.FinalExportWorkflowError):
                launcher.stable_regular_file_record(source, root=root)

    def test_tree_scan_rejects_broken_symlink_and_fifo(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            broken = root / "broken"
            broken.symlink_to(root / "missing")
            with self.assertRaises(launcher.FinalExportWorkflowError):
                launcher.scan_regular_tree(root)
        if hasattr(os, "mkfifo"):
            with tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                os.mkfifo(root / "pipe")
                with self.assertRaises(launcher.FinalExportWorkflowError):
                    launcher.scan_regular_tree(root)

    def test_completion_marker_is_after_all_local_export_checks(self) -> None:
        source = builder._completion_cell(
            recipe_sha256=builder.EXPECTED_RECIPE_SHA256
        ).source
        self.assertLess(
            source.index("Final BGE output contains validation predictions"),
            source.index("completion_path.write_text"),
        )
        self.assertLess(
            source.index("shutil.rmtree(PROJECT_ROOT)"),
            source.index("completion_path.write_text"),
        )
        self.assertIn("special_tokens_map.json", source)
        self.assertIn("local_files_only=True", source)


if __name__ == "__main__":
    unittest.main()
