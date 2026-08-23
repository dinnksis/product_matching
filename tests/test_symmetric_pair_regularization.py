from __future__ import annotations

import importlib.util
import math
import unittest

from src.symmetric_pair_regularization import symmetric_pair_loss


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class SymmetricPairRegularizationTest(unittest.TestCase):
    def test_identical_logits_have_zero_symmetry_penalty(self) -> None:
        import torch

        logits = torch.tensor([[0.0, 0.0], [1.0, 1.0]], requires_grad=True)
        result = symmetric_pair_loss(
            directional_logits=logits,
            labels=torch.tensor([0.0, 1.0]),
        )

        self.assertAlmostEqual(float(result.regularizer), 0.0, places=7)
        self.assertAlmostEqual(float(result.mean_abs_logit_gap), 0.0, places=7)
        self.assertTrue(torch.isfinite(result.supervised))

    def test_squared_gap_is_non_negative_and_has_expected_scale(self) -> None:
        import torch

        result = symmetric_pair_loss(
            directional_logits=torch.tensor([[2.0, -1.0]], requires_grad=True),
            labels=torch.tensor([0.5]),
        )

        self.assertAlmostEqual(float(result.regularizer), 9.0, places=7)
        self.assertAlmostEqual(float(result.mean_abs_logit_gap), 3.0, places=7)

    def test_two_direction_bce_is_averaged_not_summed(self) -> None:
        import torch

        result = symmetric_pair_loss(
            directional_logits=torch.zeros((1, 2), requires_grad=True),
            labels=torch.tensor([5 / 9]),
        )

        self.assertAlmostEqual(float(result.supervised), math.log(2), places=6)
        self.assertAlmostEqual(
            float(result.mean_positive_probability), 0.5, places=7
        )
        (result.supervised + 0.1 * result.regularizer).backward()


if __name__ == "__main__":
    unittest.main()
