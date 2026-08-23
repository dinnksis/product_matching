from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.experiment_significance import (
    SignificanceError,
    align_predictions,
    compare_prediction_frames,
    holm_adjust,
    read_prediction_artifact,
)


def prediction_frame(scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id1": [1, 3, 5, 7, 11, 13, 15, 17],
            "id2": [2, 4, 6, 8, 12, 14, 16, 18],
            "target": [0, 0, 1, 1, 0, 0, 1, 1],
            "category_1": ["a"] * 4 + ["b"] * 4,
            "score": scores,
        }
    )


class SignificanceTest(unittest.TestCase):
    def test_prediction_reader_skips_heavy_diagnostic_columns(self) -> None:
        frame = prediction_frame([0.4, 0.3, 0.6, 0.5, 0.4, 0.3, 0.6, 0.5])
        frame["product_text_1"] = "large text"
        frame["score_order_gap"] = 0.0
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.parquet"
            frame.to_parquet(path, index=False)

            loaded = read_prediction_artifact(path)

        self.assertEqual(
            list(loaded.columns),
            ["id1", "id2", "target", "category_1", "score"],
        )
        self.assertEqual(len(loaded), len(frame))

    def test_alignment_accepts_reordered_pairs(self) -> None:
        baseline = prediction_frame([0.4, 0.3, 0.6, 0.5, 0.4, 0.3, 0.6, 0.5])
        candidate = prediction_frame([0.1, 0.2, 0.9, 0.8, 0.1, 0.2, 0.9, 0.8])

        aligned = align_predictions(baseline, candidate.iloc[::-1])

        self.assertEqual(len(aligned), 8)
        self.assertEqual(aligned["candidate_score"].max(), 0.9)

    def test_mismatched_pair_sets_are_rejected(self) -> None:
        baseline = prediction_frame([0.4] * 8)
        candidate = prediction_frame([0.5] * 8)
        candidate.loc[0, "id2"] = 999

        with self.assertRaisesRegex(SignificanceError, "pair sets differ"):
            align_predictions(baseline, candidate)

    def test_holm_adjustment_is_monotone_in_sorted_order(self) -> None:
        adjusted = holm_adjust({"iid": 0.01, "hard": 0.04, "ood": 0.03})

        self.assertAlmostEqual(adjusted["iid"], 0.03)
        self.assertAlmostEqual(adjusted["hard"], 0.06)
        self.assertAlmostEqual(adjusted["ood"], 0.06)

    def test_paired_test_is_deterministic_and_reports_positive_delta(self) -> None:
        baseline = prediction_frame([0.8, 0.7, 0.6, 0.5, 0.8, 0.7, 0.6, 0.5])
        candidate = prediction_frame([0.1, 0.2, 0.9, 0.8, 0.1, 0.2, 0.9, 0.8])

        first = compare_prediction_frames(
            baseline,
            candidate,
            permutations=99,
            bootstrap_resamples=99,
            seed=7,
        )
        second = compare_prediction_frames(
            baseline,
            candidate,
            permutations=99,
            bootstrap_resamples=99,
            seed=7,
        )

        self.assertEqual(first, second)
        self.assertGreater(first["delta_macro_average_precision"], 0)
        self.assertGreaterEqual(first["p_value"], 0)
        self.assertLessEqual(first["p_value"], 1)
        self.assertLessEqual(first["ci95_low"], first["ci95_high"])


if __name__ == "__main__":
    unittest.main()
