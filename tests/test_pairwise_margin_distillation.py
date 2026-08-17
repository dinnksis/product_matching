from __future__ import annotations

import importlib.util
import unittest

from src.pairwise_margin_distillation import pairwise_margin_huber_loss


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class PairwiseMarginDistillationTest(unittest.TestCase):
    def test_exact_teacher_logits_have_zero_margin_loss(self) -> None:
        import torch

        teacher = torch.tensor([0.95, 0.65, 0.12])
        student = torch.logit(teacher)
        result = pairwise_margin_huber_loss(
            student_logits=student,
            teacher_probabilities=teacher,
            category_ids=torch.tensor([4, 4, 4]),
        )

        self.assertEqual(int(result.pair_count), 1)
        self.assertAlmostEqual(float(result.loss), 0.0, places=7)

    def test_comparisons_never_cross_category_boundaries(self) -> None:
        import torch

        teacher = torch.tensor([0.1, 0.9, 0.2, 0.8])
        student = torch.logit(teacher)
        result = pairwise_margin_huber_loss(
            student_logits=student,
            teacher_probabilities=teacher,
            category_ids=torch.tensor([0, 0, 1, 1]),
        )

        self.assertEqual(int(result.pair_count), 2)
        self.assertAlmostEqual(float(result.loss), 0.0, places=7)

    def test_temperature_scales_teacher_margin(self) -> None:
        import torch

        teacher = torch.tensor([0.1, 0.9])
        student = torch.logit(teacher) / 2.0
        result = pairwise_margin_huber_loss(
            student_logits=student,
            teacher_probabilities=teacher,
            category_ids=torch.tensor([0, 0]),
            temperature=2.0,
        )

        self.assertEqual(int(result.pair_count), 1)
        self.assertAlmostEqual(float(result.loss), 0.0, places=7)

    def test_equal_teacher_scores_produce_differentiable_zero(self) -> None:
        import torch

        student = torch.tensor([0.2, -0.4], requires_grad=True)
        result = pairwise_margin_huber_loss(
            student_logits=student,
            teacher_probabilities=torch.tensor([0.5, 0.5]),
            category_ids=torch.tensor([0, 0]),
        )

        self.assertEqual(int(result.pair_count), 0)
        result.loss.backward()
        torch.testing.assert_close(student.grad, torch.zeros_like(student))

    def test_hard_teacher_probabilities_are_clamped_to_finite_logits(self) -> None:
        import torch

        result = pairwise_margin_huber_loss(
            student_logits=torch.tensor([-2.0, 2.0]),
            teacher_probabilities=torch.tensor([0.0, 1.0]),
            category_ids=torch.tensor([0, 0]),
        )

        self.assertEqual(int(result.pair_count), 1)
        self.assertTrue(torch.isfinite(result.loss))
        self.assertTrue(torch.isfinite(result.mean_teacher_abs_margin))


if __name__ == "__main__":
    unittest.main()
