from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.io import build_pairs, validate_predictions
from src.scorer import HeuristicScorer


class SubmissionContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.items = pd.DataFrame(
            {
                "id": [10, 20, 30],
                "name": ["Телефон X", "телефон-x", "Чехол"],
                "attributes": ['{"brand":"A"}', '{"brand":"A"}', "{}"],
                "category": ["phone", "phone", "case"],
            }
        )
        self.matches = pd.DataFrame({"id1": [10, 10], "id2": [20, 30]})

    def test_pair_order_and_scores(self) -> None:
        pairs = build_pairs(self.items, self.matches)
        scores = HeuristicScorer().predict(pairs)
        self.assertEqual(len(scores), 2)
        self.assertGreater(scores[0], scores[1])
        np.testing.assert_array_equal(self.matches.index, pairs.index)

    def test_missing_item_fails(self) -> None:
        bad_matches = pd.DataFrame({"id1": [10], "id2": [999]})
        with self.assertRaisesRegex(ValueError, "missing item"):
            build_pairs(self.items, bad_matches)

    def test_invalid_prediction_count_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected 2"):
            validate_predictions(self.matches, np.array([0.5]))


if __name__ == "__main__":
    unittest.main()

