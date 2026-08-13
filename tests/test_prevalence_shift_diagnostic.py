from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.prevalence_shift_diagnostic import (
    evaluate_target_prevalences,
    grouped_metrics,
    negative_weight,
    sample_weights,
    validate_predictions,
)


def sample_predictions() -> pd.DataFrame:
    rows = []
    pair_id = 0
    for category, scores in {
        "a": [(1, 0.95), (1, 0.70), (0, 0.80), (0, 0.20)],
        "b": [(1, 0.90), (1, 0.60), (0, 0.75), (0, 0.10)],
    }.items():
        for target, score in scores:
            rows.append(
                {
                    "id1": pair_id * 2,
                    "id2": pair_id * 2 + 1,
                    "target": target,
                    "category": category,
                    "predict": score,
                }
            )
            pair_id += 1
    return pd.DataFrame(rows)


class PrevalenceShiftDiagnosticTest(unittest.TestCase):
    def test_negative_weight_reaches_requested_global_prevalence(self) -> None:
        frame = validate_predictions(sample_predictions())
        target = frame.target.to_numpy()
        original = float(target.mean())
        requested = 0.10
        weights = sample_weights(target, negative_weight(original, requested))
        observed = float(np.sum(weights * target) / np.sum(weights))
        self.assertAlmostEqual(observed, requested, places=14)

    def test_constant_class_weight_changes_ap_but_not_roc_auc(self) -> None:
        frame = validate_predictions(sample_predictions())
        baseline, _ = grouped_metrics(frame, np.ones(len(frame)))
        table, _ = evaluate_target_prevalences(frame, [0.10])
        shifted = table.iloc[0]
        self.assertLess(
            shifted["macro_average_precision"],
            baseline["macro_average_precision"],
        )
        self.assertAlmostEqual(
            shifted["macro_roc_auc"], baseline["macro_roc_auc"], places=14
        )
        self.assertAlmostEqual(shifted["effective_prevalence"], 0.10, places=14)

    def test_competition_metric_is_equal_weight_macro_over_categories(self) -> None:
        frame = validate_predictions(sample_predictions())
        summary, categories = grouped_metrics(frame, np.ones(len(frame)))
        self.assertAlmostEqual(
            summary["macro_average_precision"],
            float(categories.average_precision.mean()),
            places=14,
        )


if __name__ == "__main__":
    unittest.main()
