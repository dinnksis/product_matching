from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "GPU training dependencies are not installed")
class QwenTrainingUtilitiesTest(unittest.TestCase):
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(character) % 251 for character in text]

        def __call__(self, texts, **kwargs):
            return {"input_ids": [self.encode(text) for text in texts]}

    def test_balanced_truncation_preserves_both_products(self) -> None:
        from src.qwen_training import _balanced_prefixes

        first, second = _balanced_prefixes(list(range(20)), list(range(100)), budget=50)
        self.assertEqual(len(first), 20)
        self.assertEqual(len(second), 30)
        self.assertEqual(first[0], 0)
        self.assertEqual(second[0], 0)

    def test_category_label_weights_have_equal_group_mass(self) -> None:
        from src.qwen_training import balanced_sampling_weights

        categories = ["a", "a", "a", "b", "b", "b"]
        targets = [0, 0, 1, 0, 1, 1]
        weights = balanced_sampling_weights(categories, targets, "category_label")
        self.assertIsNotNone(weights)
        masses = [
            weights[[0, 1]].sum(),
            weights[[2]].sum(),
            weights[[3]].sum(),
            weights[[4, 5]].sum(),
        ]
        np.testing.assert_allclose(masses, np.full(4, 0.25))

    def test_ddp_sampler_has_equal_rank_lengths(self) -> None:
        from src.qwen_training import LengthBucketBatchSampler

        lengths = np.arange(1, 12)
        first = LengthBucketBatchSampler(lengths, lengths, batch_size=3, rank=0, world_size=2)
        second = LengthBucketBatchSampler(lengths, lengths, batch_size=3, rank=1, world_size=2)
        self.assertEqual(len(list(first)), len(list(second)))
        self.assertEqual(sum(map(len, first)), sum(map(len, second)))

    def test_token_cache_caps_length_and_keeps_both_orientations(self) -> None:
        from src.qwen_training import build_token_cache

        frame = pd.DataFrame(
            {
                "id1": [1],
                "id2": [2],
                "target": [1.0],
                "product_text_1": ["Категория: A\nНазвание: first"],
                "product_text_2": ["Категория: A\nНазвание: second"],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            cache = build_token_cache(
                frame,
                self.FakeTokenizer(),
                Path(temporary),
                "test",
                "fake",
                max_length=1024,
                tokenization_batch_size=1,
            )
            self.assertEqual(cache.size, 1)
            self.assertLessEqual(len(cache.sequence(0)), 1024)
            self.assertFalse(np.array_equal(cache.sequence(0), cache.sequence(0, reverse=True)))


if __name__ == "__main__":
    unittest.main()
