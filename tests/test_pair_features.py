from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.pair_features import (
    build_training_loss_weights,
    category_label_downsample,
    name_ngram_cosine,
)


class PairFeaturesTest(unittest.TestCase):
    def test_category_label_downsample_keeps_minority_and_category_sizes(self) -> None:
        frame = pd.DataFrame(
            {
                "row": range(11),
                "category_1": ["a"] * 7 + ["b"] * 4,
                "target": [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
            }
        )
        result = category_label_downsample(frame, seed=7)
        counts = result.groupby(["category_1", "target"]).size()
        self.assertEqual(counts.to_dict(), {("a", 0): 2, ("a", 1): 2, ("b", 0): 1, ("b", 1): 1})
        self.assertEqual(set(result.loc[result["category_1"].eq("a") & result["target"].eq(1), "row"]), {5, 6})
        self.assertEqual(len(result), result["row"].nunique())

    def test_name_ngram_cosine_rewards_matching_names(self) -> None:
        frame = pd.DataFrame(
            {
                "product_text_1": [
                    "Категория: x\nНазвание: iPhone 15 Pro 256 GB",
                    "Категория: x\nНазвание: Корм для кошек",
                ],
                "product_text_2": [
                    "Категория: x\nНазвание: iPhone 15 Pro 256 GB",
                    "Категория: x\nНазвание: Садовая лопата",
                ],
            }
        )
        similarities = name_ngram_cosine(frame, n_features=2**12)
        self.assertGreater(similarities[0], 0.99)
        self.assertLess(similarities[1], similarities[0])

    def test_sqrt_weighting_is_milder_than_inverse_frequency(self) -> None:
        weights = build_training_loss_weights(
            ["a"] * 10,
            [0] * 8 + [1] * 2,
            mode="category_label_sqrt",
        )
        observed_ratio = float(weights[8] / weights[0])
        self.assertAlmostEqual(observed_ratio, 2.0, places=6)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_lexical_weighting_redistributes_only_negative_group_mass(self) -> None:
        categories = ["a", "a", "a", "a", "b", "b"]
        targets = [0, 0, 1, 1, 0, 0]
        similarities = [0.0, 1.0, 0.0, 1.0, 0.2, 0.8]
        base = build_training_loss_weights(
            categories,
            targets,
            mode="category_label_sqrt",
        )
        weighted = build_training_loss_weights(
            categories,
            targets,
            mode="category_label_sqrt",
            lexical_similarities=similarities,
            lexical_hard_negative_strength=1.0,
        )
        self.assertGreater(weighted[1], weighted[0])
        self.assertGreater(weighted[5], weighted[4])
        np.testing.assert_allclose(weighted[[0, 1]].sum(), base[[0, 1]].sum())
        np.testing.assert_allclose(weighted[[4, 5]].sum(), base[[4, 5]].sum())


if __name__ == "__main__":
    unittest.main()
