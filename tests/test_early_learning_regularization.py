from __future__ import annotations

import math
import unittest

from src.early_learning_regularization import (
    binary_elr_loss,
    binary_elr_regularization,
    make_binary_elr_targets,
)


try:
    import torch
except ImportError:  # pragma: no cover - server dependency is optional locally
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class BinaryElrTest(unittest.TestCase):
    def test_temporal_target_and_equation_six_binary_adaptation(self) -> None:
        assert torch is not None
        history = make_binary_elr_targets(2, torch.device("cpu"))
        logits = torch.zeros(2, requires_grad=True)
        labels = torch.tensor([0.0, 1.0])
        indices = torch.tensor([0, 1])
        result = binary_elr_loss(
            logits=logits,
            labels=labels,
            example_indices=indices,
            target_history=history,
            beta=0.7,
            regularization_strength=3.0,
        )
        torch.testing.assert_close(history, torch.full((2, 2), 0.15))
        self.assertAlmostEqual(float(result.supervised), math.log(2), places=6)
        self.assertAlmostEqual(float(result.regularizer), math.log(0.85), places=6)
        self.assertAlmostEqual(
            float(result.total), math.log(2) + 3 * math.log(0.85), places=6
        )
        result.total.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_soft_labels_are_accepted_without_rounding(self) -> None:
        assert torch is not None
        history = make_binary_elr_targets(1, torch.device("cpu"))
        result = binary_elr_loss(
            logits=torch.tensor([0.2], requires_grad=True),
            labels=torch.tensor([5 / 9]),
            example_indices=torch.tensor([0]),
            target_history=history,
            beta=0.7,
            regularization_strength=3.0,
        )
        self.assertTrue(torch.isfinite(result.total))

    def test_regularizer_accepts_orientation_averaged_probability(self) -> None:
        assert torch is not None
        history = make_binary_elr_targets(1, torch.device("cpu"))
        directional_logits = torch.tensor([[2.0, -2.0]], requires_grad=True)
        averaged_probability = directional_logits.sigmoid().mean(dim=1)
        result = binary_elr_regularization(
            positive_probabilities=averaged_probability,
            example_indices=torch.tensor([0]),
            target_history=history,
            beta=0.7,
        )

        torch.testing.assert_close(history, torch.tensor([[0.15, 0.15]]))
        self.assertAlmostEqual(float(result.regularizer), math.log(0.85), places=6)
        result.regularizer.backward()
        self.assertTrue(torch.isfinite(directional_logits.grad).all())


if __name__ == "__main__":
    unittest.main()
