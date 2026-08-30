from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import nbformat as nbf
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import create_rumodernbert_sft_kaggle_notebooks as builder
import push_rumodernbert_pretrain_checkpoint_dataset as uploader
import rumodernbert_finite_bce_2xt4 as loss_hook
import run_rumodernbert_sft_kaggle as launcher
import train_bge_2ep_sft as distributed_amp
import train_rumodernbert_2xt4 as adapter


class RuModernBertKaggleTest(unittest.TestCase):
    def test_exact_variant_catalog_and_geometry(self) -> None:
        plan = builder.load_plan()
        self.assertEqual([row["key"] for row in plan["experiments"]], list(launcher.ALL_KEYS))
        self.assertEqual(plan["experiments"][-1]["epochs"], 3)
        self.assertIs(plan["training"]["gradient_checkpointing"], False)
        self.assertEqual(plan["training"]["effective_batch"], 192)
        self.assertEqual(
            plan["training"]["batch_size"]
            * plan["training"]["world_size"]
            * plan["training"]["gradient_accumulation"],
            192,
        )

    def test_dynamic_stages_require_selected_lr(self) -> None:
        plan = builder.load_plan()
        with self.assertRaises(builder.CampaignError):
            builder.resolve_variant(plan, "e2_selected_lr", None)
        resolved = builder.resolve_variant(plan, "e3_selected_lr", 4e-5)
        self.assertEqual(resolved["epochs"], 3)
        self.assertEqual(resolved["learning_rate"], 4e-5)

    def test_checkpoint_stage_excludes_optimizer(self) -> None:
        manifest = builder.load_checkpoint("alexproger23")
        self.assertNotIn("optimizer.pt", manifest["manifest"]["files"])
        self.assertEqual(
            manifest["manifest"]["reconstruction"]["sha256"], builder.MODEL_SHA256
        )

    def test_source_bundle_is_exact_and_unique(self) -> None:
        sources, ledger, digest = builder.source_bundle()
        self.assertEqual(len(ledger), len({row["path"] for row in ledger}))
        self.assertEqual(
            digest, builder.canonical_sha256({"schema_version": 1, "files": ledger})
        )
        self.assertIn("scripts/train_rumodernbert_2xt4.py", sources)
        self.assertIn("scripts/rumodernbert_finite_bce_2xt4.py", sources)

    def test_notebook_is_deterministic_and_compiles(self) -> None:
        first, first_entry = builder.build_variant(owner="alexproger23", key="e1_lr8e5")
        second, second_entry = builder.build_variant(owner="alexproger23", key="e1_lr8e5")
        self.assertEqual(first_entry, second_entry)
        self.assertEqual(
            [cell.source for cell in first.cells], [cell.source for cell in second.cells]
        )
        for index, cell in enumerate(first.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"cell-{index}", "exec")
        joined = "\n".join(cell.source for cell in first.cells if cell.cell_type == "code")
        self.assertNotIn("--validation-split', 'ood=", joined)
        self.assertIn("--nproc_per_node=2", joined)
        self.assertIn("trained_model_sha256", joined)
        nbf.validate(first)

    def test_all_slugs_are_unique_and_safe(self) -> None:
        entries = [
            builder.build_variant(
                owner="alexproger23",
                key=key,
                selected_lr=8e-5 if key not in launcher.FIRST_KEYS else None,
            )[1]
            for key in launcher.ALL_KEYS
        ]
        slugs = [entry["kernel_slug"] for entry in entries]
        self.assertEqual(len(slugs), len(set(slugs)))
        self.assertTrue(all(len(slug) <= 50 for slug in slugs))

    def test_lr_selection_prefers_anchor_within_margin(self) -> None:
        results = [
            {"key": "e1_lr8e5", "learning_rate": 8e-5, "iid_macro_average_precision": 0.80},
            {"key": "e1_lr4e5", "learning_rate": 4e-5, "iid_macro_average_precision": 0.8019},
            {"key": "e1_lr1p6e4", "learning_rate": 1.6e-4, "iid_macro_average_precision": 0.79},
        ]
        self.assertEqual(launcher.select_learning_rate(results), 8e-5)

    def test_lr_selection_chooses_lower_practical_challenger(self) -> None:
        results = [
            {"key": "e1_lr8e5", "learning_rate": 8e-5, "iid_macro_average_precision": 0.80},
            {"key": "e1_lr4e5", "learning_rate": 4e-5, "iid_macro_average_precision": 0.805},
            {"key": "e1_lr1p6e4", "learning_rate": 1.6e-4, "iid_macro_average_precision": 0.8055},
        ]
        self.assertEqual(launcher.select_learning_rate(results), 4e-5)

    def test_final_selection_prefers_fewer_epochs_in_tie(self) -> None:
        rows = [
            {"key": "e1_lr8e5", "epochs": 1, "learning_rate": 8e-5, "iid_macro_average_precision": 0.8},
            {"key": "e2_selected_lr", "epochs": 2, "learning_rate": 8e-5, "iid_macro_average_precision": 0.803},
            {"key": "e3_selected_lr", "epochs": 3, "learning_rate": 8e-5, "iid_macro_average_precision": 0.8045},
        ]
        self.assertEqual(launcher.select_final(rows, 8e-5)["epochs"], 2)

    def test_loss_hook_requires_two_ranks_and_is_finite(self) -> None:
        class Frame:
            def __len__(self):
                return builder.EXPECTED_TRAIN

        loss_hook.initialize_loss(train_frame=Frame(), device="cpu", rank=0, world_size=2)
        with self.assertRaises(ValueError):
            loss_hook.initialize_loss(train_frame=Frame(), device="cpu", rank=0, world_size=1)
        logits = torch.tensor([0.1, -0.2])
        result = loss_hook.compute_loss(
            logits=logits,
            targets=torch.tensor([1.0, 0.0]),
            sample_weights=torch.ones(2),
            pair_indices=torch.arange(2),
            orientations=torch.zeros(2),
            epoch=0,
            step=0,
        )
        self.assertTrue(torch.isfinite(result["loss"]))

    def test_adapter_binds_distributed_geometry(self) -> None:
        previous = {
            name: getattr(distributed_amp, name)
            for name in (
                "EXPECTED_PARAMETERS",
                "EXPECTED_TRAINABLE_PARAMETER_TENSORS",
                "EXPECTED_MICROBATCH",
                "EXPECTED_EVAL_BATCH",
                "EXPECTED_GRADIENT_ACCUMULATION",
                "EXPECTED_EFFECTIVE_BATCH",
                "validate_memory_geometry",
            )
        }
        try:
            adapter.configure_adapter()
            self.assertEqual(distributed_amp.EXPECTED_PARAMETERS, 149_605_633)
            self.assertEqual(distributed_amp.EXPECTED_MICROBATCH, 24)
            self.assertEqual(distributed_amp.EXPECTED_EFFECTIVE_BATCH, 192)
        finally:
            for name, value in previous.items():
                setattr(distributed_amp, name, value)

    def test_preflight_classifier_overrides_captured_bge_default(self) -> None:
        outcome = adapter.classify_amp_optimizer_attempt(
            gradients_finite=True,
            scale_before=65536.0,
            scale_after=65536.0,
            optimizer_state_parameters=adapter.EXPECTED_PARAMETER_TENSORS,
        )
        self.assertEqual(outcome, "optimizer_step")
        with self.assertRaises(RuntimeError):
            adapter.classify_amp_optimizer_attempt(
                gradients_finite=True,
                scale_before=65536.0,
                scale_after=65536.0,
                optimizer_state_parameters=393,
            )

    def test_rumodernbert_geometry_disables_activation_checkpointing(self) -> None:
        config = builder.resolved_config(
            builder.load_plan(),
            builder.resolve_variant(builder.load_plan(), "e1_lr8e5", None),
            "/tmp/model",
        )
        with mock.patch.multiple(
            distributed_amp,
            EXPECTED_PARAMETERS=adapter.EXPECTED_PARAMETERS,
            EXPECTED_TRAINABLE_PARAMETER_TENSORS=adapter.EXPECTED_PARAMETER_TENSORS,
            EXPECTED_MICROBATCH=adapter.EXPECTED_MICROBATCH,
            EXPECTED_EVAL_BATCH=adapter.EXPECTED_EVAL_BATCH,
            EXPECTED_GRADIENT_ACCUMULATION=adapter.EXPECTED_GRADIENT_ACCUMULATION,
            EXPECTED_EFFECTIVE_BATCH=adapter.EXPECTED_EFFECTIVE_BATCH,
        ):
            adapter.validate_memory_geometry(config)
            config["gradient_checkpointing"] = True
            with self.assertRaises(distributed_amp.BgeTrainingContractError):
                adapter.validate_memory_geometry(config)
        self.assertIsNone(
            adapter.disable_gradient_checkpointing_enable(
                object(), {"use_reentrant": False}
            )
        )

    def test_failed_first_slug_is_tombstoned_and_new_identity_differs(self) -> None:
        failed = "pm-rmb-e1-lr8e5-06abd6796702-s42-v1"
        self.assertIn(failed, launcher.TOMBSTONED_KERNEL_SLUGS)
        self.assertIn(
            "pm-rmb-e1-lr8e5-c37bbe4011cc-s42-v1",
            launcher.TOMBSTONED_KERNEL_SLUGS,
        )
        self.assertIn(
            "pm-rmb-e1-lr8e5-092f927d6256-s42-v1",
            launcher.TOMBSTONED_KERNEL_SLUGS,
        )
        _, entry = builder.build_variant(owner="alexproger23", key="e1_lr8e5")
        self.assertNotEqual(entry["kernel_slug"], failed)
        self.assertNotIn(entry["kernel_slug"], launcher.TOMBSTONED_KERNEL_SLUGS)

    def test_launcher_invocation_runs_exactly_one_requested_entry(self) -> None:
        entry = {"kernel_slug": "pm-rmb-single-test"}
        argv = [
            "run_rumodernbert_sft_kaggle.py",
            "--key",
            "e1_lr8e5",
            "--stage-only",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(launcher.kaggle, "load_dotenv"),
            mock.patch.dict(
                os.environ,
                {
                    "KAGGLE_USERNAME": "alexproger23",
                    "KAGGLE_ACCELERATOR": "NvidiaTeslaT4",
                    "KAGGLE_IS_PRIVATE": "true",
                },
            ),
            mock.patch.object(launcher.kaggle, "kaggle_command", return_value=["kaggle"]),
            mock.patch.object(launcher, "campaign_lock", return_value=nullcontext()),
            mock.patch.object(
                launcher,
                "build_entry",
                return_value=(entry, Path("single.ipynb")),
            ) as build,
            mock.patch.object(launcher, "run_entry", return_value=None) as run,
        ):
            self.assertEqual(launcher.main(), 0)
        build.assert_called_once_with("alexproger23", "e1_lr8e5", None)
        run.assert_called_once()

    def test_launcher_requires_explicit_key_and_dynamic_lr(self) -> None:
        with mock.patch.object(sys, "argv", ["run_rumodernbert_sft_kaggle.py"]):
            with self.assertRaises(SystemExit):
                launcher.parse_args()
        with mock.patch.object(
            sys,
            "argv",
            ["run_rumodernbert_sft_kaggle.py", "--key", "e2_selected_lr"],
        ):
            with self.assertRaises(SystemExit):
                launcher.main()

    def test_kaggle_not_found_parser_is_exact(self) -> None:
        self.assertEqual(launcher._listed_kernel_refs("Not found\n"), set())
        with self.assertRaises(launcher.CampaignError):
            launcher._listed_kernel_refs("Not found extra")

    def test_macro_ap_uses_canonical_shared_trainer_schema(self) -> None:
        frame = pd.DataFrame(
            {
                "target": [1.0, 0.0, 1.0, 0.0],
                "category_1": ["a", "a", "b", "b"],
                "score": [0.9, 0.1, 0.8, 0.2],
            }
        )
        self.assertEqual(launcher.macro_ap(frame), 1.0)
        with self.assertRaises(KeyError):
            launcher.macro_ap(frame.rename(columns={"score": "score_symmetric"}))

    def test_uploader_contract_restores_shared_globals(self) -> None:
        original = distributed = uploader.hardened.CHECKPOINT_FILES
        with uploader.hardened_contract():
            self.assertEqual(uploader.hardened.CHECKPOINT_FILES, uploader.CHECKPOINT_FILES)
        self.assertEqual(uploader.hardened.CHECKPOINT_FILES, original)


if __name__ == "__main__":
    unittest.main()
