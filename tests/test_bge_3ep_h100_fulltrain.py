from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_module(
    "bge_3ep_h100_fulltrain_runner", "scripts/run_bge_3ep_h100_fulltrain.py"
)
trainer = load_module(
    "bge_3ep_h100_fulltrain_trainer", "scripts/train_bge_3ep_h100_fulltrain.py"
)
loss_hook = load_module(
    "bge_h100_fulltrain_finite_bce",
    "scripts/bge_h100_fulltrain_finite_bce.py",
)


class BgeThreeEpochH100FulltrainTest(unittest.TestCase):
    def test_execution_sources_are_frozen(self) -> None:
        self.assertEqual(
            runner.validate_own_execution_sources(), runner.EXPECTED_EXECUTION_FILES
        )

    def test_fulltrain_loss_is_finite_and_exact_row_bound(self) -> None:
        class ExactRows:
            def __len__(self):
                return loss_hook.EXPECTED_TRAIN_ROWS

        loss_hook.initialize_loss(
            train_frame=ExactRows(), device=torch.device("cpu"), rank=0, world_size=1
        )
        result = loss_hook.compute_loss(
            logits=torch.tensor([-1.0, 1.0]),
            targets=torch.tensor([0.0, 1.0]),
            sample_weights=torch.ones(2),
            pair_indices=torch.arange(2),
            orientations=torch.zeros(2),
            epoch=0,
            step=0,
        )
        self.assertTrue(torch.isfinite(result["loss"]))
        with self.assertRaisesRegex(ValueError, "row count"):
            loss_hook.initialize_loss(
                train_frame=[1], device=torch.device("cpu"), rank=0, world_size=1
            )

    def test_partial_accumulation_is_sample_exact(self) -> None:
        tail = [64, 64, 64, 22]
        self.assertEqual(
            trainer.accumulation_group_denominator(
                tail, step=3, gradient_accumulation=3
            ),
            22,
        )
        middle = [64, 22, 64, 64]
        denominator = trainer.accumulation_group_denominator(
            middle, step=1, gradient_accumulation=3
        )
        self.assertEqual(denominator, 150)
        weights = [size / denominator for size in middle[:3]]
        self.assertAlmostEqual(sum(weights), 1.0)

    def test_full_train_frame_has_all_four_sources_in_frozen_order(self) -> None:
        frames = {
            "train_pairs.parquet": pd.DataFrame(
                {"id1": [1], "id2": [2], "target": [1.0]}
            ),
            "iid_validation_pairs.parquet": pd.DataFrame(
                {"id1": [3], "id2": [4], "target": [0.0]}
            ),
            "hard_validation_pairs.parquet": pd.DataFrame(
                {"id1": [5], "id2": [6], "target": [1.0]}
            ),
            "ood_validation_pairs.parquet": pd.DataFrame(
                {"id1": [7], "id2": [8], "target": [0.0]}
            ),
        }
        expected_sources = {
            "human_train": 1,
            "human_iid": 1,
            "human_hard": 1,
            "human_former_ood": 1,
        }
        with mock.patch.object(runner, "EXPECTED_ROWS", 4), mock.patch.object(
            runner, "EXPECTED_POSITIVES", 2
        ), mock.patch.object(
            runner, "EXPECTED_SOURCE_COUNTS", expected_sources
        ), mock.patch.object(runner, "EXPECTED_COVERED_ITEMS", 8):
            result = runner.full_train_frame(frames)
        self.assertEqual(
            result["label_source"].tolist(),
            ["human_train", "human_iid", "human_hard", "human_former_ood"],
        )

    def test_recipe_is_three_epochs_one_h100_and_no_validation(self) -> None:
        payload = json.loads(
            (ROOT / "configs" / "bge_3ep_sft_oodtrain_h100_v1.json").read_text(
                encoding="utf-8"
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.json"
            config.write_text(json.dumps(payload), encoding="utf-8")
            values = dict(payload)
            values.update(
                config=config,
                validation_split=[],
                model_load_kwarg=[],
            )
            trainer.validate_final_contract(
                argparse.Namespace(**values), world_size=1
            )
            with self.assertRaisesRegex(
                trainer.FinalFulltrainContractError, "world_size"
            ):
                trainer.validate_final_contract(
                    argparse.Namespace(**values), world_size=2
                )

    def test_training_command_has_no_validation_arguments(self) -> None:
        command = runner.training_command(
            config_path=Path("/tmp/config.json"),
            prepared_dir=Path("/tmp/prepared"),
            output_dir=Path("/tmp/output"),
            token_cache_dir=Path("/tmp/cache"),
        )
        self.assertNotIn("--validation-split", command)
        self.assertIn(str(runner.LOSS_HOOK), command)


if __name__ == "__main__":
    unittest.main()
