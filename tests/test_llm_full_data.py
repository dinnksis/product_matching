from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.train_llm_full import (
    BucketBatchSampler,
    PairCollator,
    SymmetricEvaluationBatchSampler,
    infer_pair_template,
    one_logit,
    snapshot_checkpoint,
)
from src.llm_full_data import (
    balanced_prefix_lengths,
    build_full_pair_cache,
    build_pair_category_cache,
)


class FakePairTokenizer:
    model_input_names = ["input_ids", "attention_mask"]
    padding_side = "right"
    pad_token_id = 0

    def __len__(self) -> int:
        return 512

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 4 if pair else 2

    def build_inputs_with_special_tokens(self, first, second=None):
        if second is None:
            return [1, *first, 2]
        return [1, *first, 2, 2, *second, 2]

    def __call__(self, texts, *, max_length, **kwargs):
        return {
            "input_ids": [
                [3 + ord(character) % 500 for character in text][:max_length]
                for text in texts
            ]
        }


class FullLlmDataTest(unittest.TestCase):
    def test_balanced_prefixes_keep_both_products(self) -> None:
        self.assertEqual(balanced_prefix_lengths(10, 100, 40), (10, 30))
        self.assertEqual(balanced_prefix_lengths(100, 10, 40), (30, 10))
        self.assertEqual(balanced_prefix_lengths(12, 8, 40), (12, 8))

    def test_pair_template_is_inferred_without_hardcoding_xlmr_tokens(self) -> None:
        template = infer_pair_template(FakePairTokenizer())
        np.testing.assert_array_equal(template.prefix, [1])
        np.testing.assert_array_equal(template.middle, [2, 2])
        np.testing.assert_array_equal(template.suffix, [2])
        self.assertEqual(template.special_tokens, 4)
        self.assertFalse(template.uses_token_type_ids)

    def test_symmetric_evaluation_visits_both_orientations_once(self) -> None:
        sampler = SymmetricEvaluationBatchSampler([30, 10, 20], batch_size=4)
        rows = [row for batch in sampler for row in batch]
        self.assertEqual(len(sampler), 2)
        self.assertCountEqual(
            rows,
            [
                (0, False),
                (0, True),
                (1, False),
                (1, True),
                (2, False),
                (2, True),
            ],
        )

    def test_symmetric_training_collator_emits_two_directions_per_pair(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")

        tokenizer = FakePairTokenizer()
        collator = PairCollator(
            template=infer_pair_template(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            max_length=16,
            pad_to_multiple_of=1,
            symmetric=True,
        )
        batch = collator(
            [
                {
                    "first": np.asarray([10, 11]),
                    "second": np.asarray([20]),
                    "target": 5 / 9,
                    "pair_index": 7,
                    "reverse": False,
                }
            ]
        )

        torch.testing.assert_close(
            batch["input_ids"],
            torch.tensor(
                [
                    [1, 10, 11, 2, 2, 20, 2],
                    [1, 20, 2, 2, 10, 11, 2],
                ]
            ),
        )
        self.assertEqual(len(batch["targets"]), 1)
        self.assertEqual(len(batch["pair_indices"]), 1)

    def test_distributed_sampler_keeps_every_example_once_without_padding(self) -> None:
        lengths = np.arange(25, dtype=np.int64) + 1
        samplers = [
            BucketBatchSampler(
                lengths,
                batch_size=3,
                bucket_size_multiplier=2,
                seed=17,
                rank=rank,
                world_size=4,
            )
            for rank in range(4)
        ]
        batches_by_rank = [list(sampler) for sampler in samplers]

        self.assertEqual({len(batches) for batches in batches_by_rank}, {3})
        all_indices: list[int] = []
        for step in range(3):
            step_indices: list[int] = []
            for batches in batches_by_rank:
                self.assertGreater(len(batches[step]), 0)
                self.assertLessEqual(len(batches[step]), 3)
                step_indices.extend(index for index, _ in batches[step])
            self.assertEqual(len(step_indices), len(set(step_indices)))
            all_indices.extend(step_indices)
        self.assertCountEqual(all_indices, range(25))

        samplers[2].set_epoch(0, start_batch=1)
        self.assertEqual(list(samplers[2]), batches_by_rank[2][1:])

    def test_distributed_validation_partitions_pairs_without_overlap(self) -> None:
        lengths = [30, 10, 20, 60, 50, 40, 70]
        rows_by_rank = []
        for rank in range(3):
            sampler = SymmetricEvaluationBatchSampler(
                lengths,
                batch_size=4,
                rank=rank,
                world_size=3,
            )
            rows_by_rank.append([row for batch in sampler for row in batch])
        combined = [row for rows in rows_by_rank for row in rows]
        self.assertCountEqual(
            combined,
            [
                (index, reverse)
                for index in range(len(lengths))
                for reverse in (False, True)
            ],
        )
        self.assertEqual(len(combined), len(set(combined)))

    def test_one_logit_contract_rejects_causal_or_two_class_outputs(self) -> None:
        outputs = SimpleNamespace(logits=np.asarray([[0.25], [-0.5]]))
        np.testing.assert_array_equal(one_logit(outputs), [0.25, -0.5])
        with self.assertRaisesRegex(ValueError, "exactly one logit"):
            one_logit(SimpleNamespace(logits=np.zeros((2, 2))))
        with self.assertRaisesRegex(ValueError, "exactly one logit"):
            one_logit(SimpleNamespace(logits=np.zeros((2, 4, 100))))

    def test_checkpoint_snapshot_uses_hard_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "checkpoint-last"
            source.mkdir()
            (source / "model.safetensors").write_bytes(b"weights")
            destination = root / "checkpoint-epoch-01"
            snapshot_checkpoint(source, destination, replace=False)
            self.assertEqual(
                (source / "model.safetensors").stat().st_ino,
                (destination / "model.safetensors").stat().st_ino,
            )

    def test_cache_retains_every_fractional_soft_target(self) -> None:
        targets = [0.0, 5 / 9, 6 / 9, 7 / 9, 1.0]
        items = pd.DataFrame(
            {
                "id": [40, 10, 50, 20, 30],
                "name": ["d", "a", "e", "b", "c"],
                "attributes": ['{"Brand": "Test"}'] * 5,
                "category": ["test"] * 5,
            }
        )
        pairs = pd.DataFrame(
            {
                "id1": [10, 20, 30, 40, 50],
                "id2": [20, 30, 40, 50, 10],
                "target": targets,
            }
        )
        tokenizer = FakePairTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item_path = root / "items.parquet"
            pair_path = root / "pairs.parquet"
            items.to_parquet(item_path, index=False)
            pairs.to_parquet(pair_path, index=False)
            cache = build_full_pair_cache(
                item_paths=[item_path],
                pair_paths=[pair_path],
                tokenizer=tokenizer,
                model_name="fake",
                cache_root=root / "cache",
                max_length=32,
                item_batch_size=2,
                pair_batch_size=2,
            )
            self.assertEqual(cache.pair_count, 5)
            self.assertEqual(cache.item_count, 5)
            np.testing.assert_allclose(cache.targets, targets, rtol=0, atol=1e-6)
            self.assertGreater(len(cache.tokens_for_item(0)), 0)
            self.assertTrue((cache.pair_lengths <= 32).all())
            self.assertEqual(cache.metadata["serialization"]["variant"], "S1_KEY_VALUE")
            self.assertFalse(cache.metadata["serialization"]["category_included"])
            self.assertTrue((cache.directory / "attribute_name_frequency.csv").is_file())

            category_cache = build_pair_category_cache(
                cache,
                item_paths=[item_path],
                batch_size=2,
            )
            np.testing.assert_array_equal(category_cache.values, [0, 0, 0, 0, 0])
            self.assertEqual(category_cache.metadata["category_names"], ["test"])
            self.assertEqual(category_cache.metadata["pair_counts"], {"test": 5})

            reused_categories = build_pair_category_cache(
                cache,
                item_paths=[item_path],
                batch_size=2,
            )
            self.assertEqual(
                reused_categories.metadata["cache_fingerprint"],
                cache.metadata["fingerprint"],
            )

            reused = build_full_pair_cache(
                item_paths=[item_path],
                pair_paths=[pair_path],
                tokenizer=tokenizer,
                model_name="fake",
                cache_root=root / "cache",
                max_length=32,
                item_batch_size=2,
                pair_batch_size=2,
            )
            self.assertEqual(reused.directory, cache.directory)


if __name__ == "__main__":
    unittest.main()
