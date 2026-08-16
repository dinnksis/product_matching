from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

@unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed locally")
class CrossEncoderExperimentHooksTest(unittest.TestCase):
    def test_default_hook_is_weighted_bce(self) -> None:
        import torch

        from src.cross_encoder_experiment_hooks import load_loss_hook

        hook = load_loss_hook(None)
        logits = torch.tensor([0.0, 1.0], requires_grad=True)
        targets = torch.tensor([0.0, 1.0])
        weights = torch.tensor([1.0, 2.0])

        loss, metrics = hook.compute(
            logits=logits,
            targets=targets,
            sample_weights=weights,
        )

        self.assertEqual(hook.name, "weighted_bce")
        self.assertGreater(float(loss), 0.0)
        self.assertIn("bce", metrics)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_external_hook_initializes_and_reports_metrics(self) -> None:
        import torch

        from src.cross_encoder_experiment_hooks import load_loss_hook

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom_loss.py"
            path.write_text(
                """
scale = None

def initialize_loss(*, world_size, **kwargs):
    global scale
    scale = world_size

def compute_loss(*, logits, targets, **kwargs):
    loss = ((logits - targets) ** 2).mean() * scale
    return {"loss": loss, "mse": loss.detach()}
""".strip(),
                encoding="utf-8",
            )
            hook = load_loss_hook(path)
            hook.initialize(train_frame=None, device=None, rank=0, world_size=2)
            logits = torch.tensor([0.0, 1.0], requires_grad=True)
            loss, metrics = hook.compute(
                logits=logits,
                targets=torch.tensor([1.0, 1.0]),
            )

        self.assertEqual(float(loss), 1.0)
        self.assertEqual(float(metrics["mse"]), 1.0)
        self.assertIsNotNone(hook.sha256)

    def test_external_hook_requires_compute_loss(self) -> None:
        from src.cross_encoder_experiment_hooks import load_loss_hook

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.py"
            path.write_text("VALUE = 1\n", encoding="utf-8")
            with self.assertRaisesRegex(TypeError, "compute_loss"):
                load_loss_hook(path)


if __name__ == "__main__":
    unittest.main()
