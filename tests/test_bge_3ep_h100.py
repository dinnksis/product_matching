from __future__ import annotations

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


runner = load_module("bge_3ep_h100_runner", "scripts/run_bge_3ep_h100.py")
trainer = load_module("bge_3ep_h100_trainer", "scripts/train_bge_3ep_h100.py")
loss_hook = load_module("bge_h100_finite_bce", "scripts/bge_h100_finite_bce.py")


class BgeThreeEpochH100Test(unittest.TestCase):
    def test_frozen_config_and_hash(self) -> None:
        path = ROOT / "configs" / "bge_3ep_sft_oodtrain_h100_v1.json"
        self.assertEqual(runner.sha256_file(path), runner.EXPECTED_CONFIG_SHA256)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["epochs"], 3)
        self.assertEqual(payload["learning_rate"], 2e-5)
        self.assertEqual(payload["batch_size"] * payload["gradient_accumulation"], 192)
        self.assertEqual(payload["model"], "model/pretrain_bge_2ep")
        self.assertFalse(payload["gradient_checkpointing"])

    def test_execution_sources_match_frozen_hashes(self) -> None:
        self.assertEqual(runner.validate_execution_sources(), runner.EXPECTED_EXECUTION_FILES)

    def test_training_wrapper_accepts_only_exact_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "pretrain_bge_2ep"
            model.mkdir()
            payload = json.loads(
                (ROOT / "configs" / "bge_3ep_sft_oodtrain_h100_v1.json").read_text(
                    encoding="utf-8"
                )
            )
            payload["model"] = str(model)
            config = root / "config.json"
            config.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(trainer.load_training_config(config), payload)
            payload["epochs"] = 2
            config.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(trainer.BgeH100TrainingError, "epochs"):
                trainer.load_training_config(config)

    def test_loss_hook_is_finite_plain_bce(self) -> None:
        class ExactRows:
            def __len__(self):
                return loss_hook.EXPECTED_TRAIN_ROWS

        loss_hook.initialize_loss(
            train_frame=ExactRows(), device=torch.device("cpu"), rank=0, world_size=1
        )
        result = loss_hook.compute_loss(
            logits=torch.tensor([0.0, 1.0]),
            targets=torch.tensor([0.0, 1.0]),
            sample_weights=torch.ones(2),
            pair_indices=torch.arange(2),
            orientations=torch.zeros(2),
            epoch=0,
            step=0,
        )
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertEqual(result["loss"], result["bce"])
        with self.assertRaisesRegex(ValueError, "one rank"):
            loss_hook.initialize_loss(
                train_frame=ExactRows(),
                device=torch.device("cpu"),
                rank=0,
                world_size=2,
            )
        with self.assertRaises(FloatingPointError):
            loss_hook.compute_loss(
                logits=torch.tensor([float("nan")]),
                targets=torch.tensor([1.0]),
                sample_weights=torch.ones(1),
                pair_indices=torch.arange(1),
                orientations=torch.zeros(1),
                epoch=0,
                step=0,
            )

    def test_optimizer_and_clip_guards(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = trainer.adamw_foreach_disabled([parameter], lr=2e-5)
        self.assertFalse(optimizer.defaults["foreach"])
        with self.assertRaisesRegex(trainer.BgeH100TrainingError, "foreach"):
            trainer.adamw_foreach_disabled([parameter], lr=2e-5, foreach=True)
        parameter.grad = torch.tensor([1.0])
        norm = trainer.strict_clip_grad_norm([parameter], 0.5)
        self.assertTrue(torch.isfinite(norm))
        self.assertLessEqual(float(parameter.grad.abs().max()), 0.500001)

    def test_unordered_pair_index_detects_reverse_duplicate(self) -> None:
        frame = pd.DataFrame(
            {"id1": [1, 3, 2], "id2": [2, 4, 1], "target": [0.0, 1.0, 0.0]}
        )
        self.assertTrue(runner.unordered_pair_index(frame).has_duplicates)

    def test_training_command_is_single_process_and_two_split(self) -> None:
        command = runner.training_command(
            config_path=Path("/tmp/config.json"),
            prepared_dir=Path("/tmp/prepared"),
            human_dir=Path("/tmp/human"),
            output_dir=Path("/tmp/output"),
            token_cache_dir=Path("/tmp/cache"),
        )
        self.assertEqual(command[0], __import__("sys").executable)
        self.assertNotIn("torchrun", " ".join(command))
        self.assertEqual(command.count("--validation-split"), 2)
        self.assertNotIn("ood=", " ".join(command))

    def test_static_tokenizer_sidecar_is_copied_and_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "special_tokens_map.json"
            destination = root / "output" / "special_tokens_map.json"
            destination.parent.mkdir()
            source.write_text("{}\n", encoding="utf-8")
            digest = runner.sha256_file(source)
            with mock.patch.dict(
                runner.EXPECTED_CHECKPOINT_FILES,
                {"special_tokens_map.json": digest},
                clear=False,
            ):
                runner.install_static_tokenizer_file(source, destination)
                runner.install_static_tokenizer_file(source, destination)
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertNotEqual(source.stat().st_ino, destination.stat().st_ino)

    def test_default_is_plan_only(self) -> None:
        with mock.patch.object(runner, "load_base_config", return_value={}), mock.patch.object(
            runner, "validate_execution_sources", return_value={}
        ), mock.patch.object(
            runner, "validate_hashed_files", return_value={}
        ), mock.patch.object(
            runner, "validate_data", return_value=({}, {})
        ), mock.patch("builtins.print") as output:
            self.assertEqual(runner.main([]), 0)
        self.assertTrue(any("Plan only" in str(call) for call in output.call_args_list))


if __name__ == "__main__":
    unittest.main()
