from __future__ import annotations

import unittest

import numpy as np

from src.validation_metrics import binary_probability_metrics, recall_at_precision


class ValidationMetricsTest(unittest.TestCase):
    def test_recall_at_p99_chooses_maximum_recall(self) -> None:
        target = np.array([1, 1, 0, 1, 0, 0], dtype=np.int8)
        probability = np.array([0.99, 0.90, 0.80, 0.70, 0.60, 0.10])

        recall, threshold = recall_at_precision(target, probability)

        self.assertAlmostEqual(recall, 2 / 3)
        self.assertAlmostEqual(threshold or 0.0, 0.90)

    def test_recall_is_zero_when_p99_is_unavailable(self) -> None:
        target = np.array([0, 1, 0, 1], dtype=np.int8)
        probability = np.array([0.99, 0.9, 0.8, 0.7])

        recall, threshold = recall_at_precision(target, probability)

        self.assertEqual(recall, 0.0)
        self.assertIsNone(threshold)

    def test_probability_metrics_match_perfect_ranking(self) -> None:
        target = np.array([0, 0, 1, 1], dtype=np.int8)
        probability = np.array([0.01, 0.1, 0.9, 0.99])

        metrics = binary_probability_metrics(target, probability)

        self.assertEqual(metrics["recall_at_precision_0_99"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertGreater(metrics["log_loss"], 0.0)
        self.assertLess(metrics["log_loss"], 0.1)

    def test_non_probability_scores_are_rejected_for_log_loss(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            binary_probability_metrics(
                np.array([0, 1]),
                np.array([-1.0, 2.0]),
            )


if __name__ == "__main__":
    unittest.main()
