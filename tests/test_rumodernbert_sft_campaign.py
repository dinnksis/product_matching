from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import rumodernbert_finite_bce as loss_hook
import run_rumodernbert_sft_campaign as campaign
import train_rumodernbert_sft as trainer


class RuModernBertPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = campaign.load_plan(
            ROOT / "configs" / "rumodernbert_3ep_sft_oodtrain_h100_v1.json"
        )

    def test_exact_five_stage_sequence(self) -> None:
        stages = campaign.stage_specifications(self.plan)
        self.assertEqual([stage["key"] for stage in stages], campaign.EXPECTED_STAGE_KEYS)
        self.assertEqual([stage["epochs"] for stage in stages], [1, 1, 1, 2, 3])
        self.assertEqual(
            [stage["learning_rate"] for stage in stages],
            [8e-5, 4e-5, 1.6e-4, None, None],
        )
        resolved = campaign.stage_specifications(self.plan, selected_lr=4e-5)
        self.assertEqual([resolved[3]["learning_rate"], resolved[4]["learning_rate"]], [4e-5, 4e-5])

    def test_data_policy_has_no_ood_validation(self) -> None:
        source = self.plan["source_data"]
        self.assertEqual(source["validation_splits"], ["iid", "hard"])
        self.assertEqual(source["ood_policy"], "included_in_train_not_evaluated")
        self.assertFalse(self.plan["selection"]["ood_is_evaluated"])
        self.assertEqual(self.plan["selection"]["ood_metric_sentinel"], -1)

    def test_resolved_training_config_is_plain_bce_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / trainer.EXPECTED_MODEL_DIRNAME
            checkpoint.mkdir()
            config = campaign.resolved_config(
                self.plan, checkpoint, epochs=3, learning_rate=8e-5
            )
            self.assertEqual(set(config), campaign.TRAINER_CONFIG_KEYS)
            self.assertEqual(config["epochs"], 3)
            self.assertEqual(config["learning_rate"], 8e-5)
            self.assertEqual(config["batch_size"] * config["gradient_accumulation"], 192)
            self.assertEqual(config["sampling"], "none")
            self.assertEqual(config["loss_weighting"], "none")
            self.assertEqual(config["label_smoothing"], 0.0)
            self.assertFalse(config["gradient_checkpointing"])

    def test_training_command_names_only_iid_and_hard(self) -> None:
        command = campaign.training_command(
            config_path=Path("config.json"),
            prepared_dir=Path("prepared"),
            output_dir=Path("output"),
            token_cache_dir=Path("cache"),
        )
        joined = " ".join(command)
        self.assertIn("iid=iid_validation_pairs.parquet", joined)
        self.assertIn("hard=hard_validation_pairs.parquet", joined)
        self.assertNotIn("ood=", joined)
        self.assertIn(str(campaign.LOSS_HOOK), command)

    def test_default_cli_is_plan_only(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(campaign.main([]), 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["mode"], "plan_only")
        self.assertEqual(payload["training_runs"], 5)
        self.assertFalse(payload["kaggle_used"])


class SelectionRuleTest(unittest.TestCase):
    @staticmethod
    def lr_completions(scores: dict[float, float]) -> dict[str, dict[str, float]]:
        return {
            "e1_lr8e5": {"learning_rate": 8e-5, "iid_macro_average_precision": scores[8e-5]},
            "e1_lr4e5": {"learning_rate": 4e-5, "iid_macro_average_precision": scores[4e-5]},
            "e1_lr1p6e4": {
                "learning_rate": 1.6e-4,
                "iid_macro_average_precision": scores[1.6e-4],
            },
        }

    def test_lr_anchor_wins_practical_tie(self) -> None:
        values = self.lr_completions({4e-5: 0.8100, 8e-5: 0.8110, 1.6e-4: 0.8129})
        self.assertEqual(campaign.select_learning_rate(values, 0.002), 8e-5)

    def test_lr_numeric_winner_outside_margin(self) -> None:
        values = self.lr_completions({4e-5: 0.816, 8e-5: 0.810, 1.6e-4: 0.813})
        self.assertEqual(campaign.select_learning_rate(values, 0.002), 4e-5)

    def test_lr_tie_without_anchor_prefers_smaller(self) -> None:
        values = self.lr_completions({4e-5: 0.8160, 8e-5: 0.810, 1.6e-4: 0.8175})
        self.assertEqual(campaign.select_learning_rate(values, 0.002), 4e-5)

    def test_epoch_rule_selects_fewest_within_margin(self) -> None:
        values = {
            "one": {"epochs": 1, "iid_macro_average_precision": 0.8100},
            "two": {"epochs": 2, "iid_macro_average_precision": 0.8150},
            "three": {"epochs": 3, "iid_macro_average_precision": 0.8165},
        }
        self.assertEqual(campaign.select_epoch(values, 0.002), 2)


class DataAndReceiptTest(unittest.TestCase):
    def test_pair_validator_rejects_unordered_duplicate(self) -> None:
        frame = pd.DataFrame(
            {"id1": [1, 2], "id2": [2, 1], "target": [0.0, 0.0]}
        )
        with self.assertRaisesRegex(campaign.RuModernBertCampaignError, "duplicate unordered"):
            campaign.validate_pair_frame(frame, name="probe", expected_rows=2)

    def test_write_once_json_rejects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            campaign.write_once_json(path, {"value": 1})
            campaign.write_once_json(path, {"value": 1})
            with self.assertRaisesRegex(campaign.RuModernBertCampaignError, "differs"):
                campaign.write_once_json(path, {"value": 2})

    def test_prepared_manifest_hash_is_self_consistent(self) -> None:
        plan = campaign.load_plan(
            ROOT / "configs" / "rumodernbert_3ep_sft_oodtrain_h100_v1.json"
        )
        payload = campaign.prepared_manifest_payload(
            plan=plan,
            source_hashes={"source": "a" * 64},
            output_dir=Path("/tmp/prepared"),
            output_hashes={"output": "b" * 64},
        )
        stored = payload.pop("manifest_payload_sha256")
        self.assertEqual(stored, campaign.canonical_sha256(payload))


class TrainingGuardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = campaign.load_plan(
            ROOT / "configs" / "rumodernbert_3ep_sft_oodtrain_h100_v1.json"
        )

    def test_training_config_accepts_only_frozen_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / trainer.EXPECTED_MODEL_DIRNAME
            checkpoint.mkdir()
            config = campaign.resolved_config(
                self.plan, checkpoint, epochs=3, learning_rate=8e-5
            )
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            self.assertEqual(trainer.load_training_config(path)["epochs"], 3)
            config["label_smoothing"] = 0.05
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(trainer.RuModernBertTrainingError, "label_smoothing"):
                trainer.load_training_config(path)

    def test_finite_loss_hook(self) -> None:
        logits = torch.tensor([0.0, 1.0])
        targets = torch.tensor([0.0, 1.0])
        weights = torch.ones(2)
        result = loss_hook.compute_loss(
            logits=logits,
            targets=targets,
            sample_weights=weights,
            pair_indices=torch.arange(2),
            orientations=torch.zeros(2),
            epoch=0,
            step=0,
        )
        self.assertTrue(torch.isfinite(result["loss"]))
        with self.assertRaises(FloatingPointError):
            loss_hook.compute_loss(
                logits=torch.tensor([float("nan")]),
                targets=torch.tensor([0.0]),
                sample_weights=torch.ones(1),
                pair_indices=torch.arange(1),
                orientations=torch.zeros(1),
                epoch=0,
                step=0,
            )

    def test_strict_gradient_clip_rejects_nonfinite(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        parameter.grad = torch.tensor([float("inf")])
        with self.assertRaises(RuntimeError):
            trainer.strict_clip_grad_norm([parameter], 0.5)


if __name__ == "__main__":
    unittest.main()
