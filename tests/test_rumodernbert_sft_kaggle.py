from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import nbformat as nbf
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
    def test_exact_five_run_plan_and_geometry(self) -> None:
        plan = builder.load_plan()
        self.assertEqual([row["key"] for row in plan["experiments"]], list(launcher.ALL_KEYS))
        self.assertEqual(plan["experiments"][-1]["epochs"], 3)
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

    def test_kaggle_not_found_parser_is_exact(self) -> None:
        self.assertEqual(launcher._listed_kernel_refs("Not found\n"), set())
        with self.assertRaises(launcher.CampaignError):
            launcher._listed_kernel_refs("Not found extra")

    def test_uploader_contract_restores_shared_globals(self) -> None:
        original = distributed = uploader.hardened.CHECKPOINT_FILES
        with uploader.hardened_contract():
            self.assertEqual(uploader.hardened.CHECKPOINT_FILES, uploader.CHECKPOINT_FILES)
        self.assertEqual(uploader.hardened.CHECKPOINT_FILES, original)


if __name__ == "__main__":
    unittest.main()
