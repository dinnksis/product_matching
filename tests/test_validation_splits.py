from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prepare_validation_splits import split_llm_data

from src.validation_splits import (
    hard_selection_targets,
    proportional_group_quotas,
    sample_component_anchors,
    sample_components_to_pair_count,
    select_hard_anchors,
    stable_component_ids,
)


class ValidationSplitsTest(unittest.TestCase):
    def test_component_ids_use_minimum_item_and_join_transitively(self) -> None:
        pairs = pd.DataFrame({"id1": [9, 2, 100], "id2": [2, 5, 101]})
        np.testing.assert_array_equal(stable_component_ids(pairs), [2, 2, 100])

    def test_proportional_quotas_have_exact_total(self) -> None:
        frame = pd.DataFrame(
            {
                "category": ["a"] * 7 + ["b"] * 3,
                "target": [0] * 5 + [1] * 2 + [0] * 2 + [1],
            }
        )
        quotas = proportional_group_quotas(frame, 6)
        self.assertEqual(sum(quotas.values()), 6)
        self.assertEqual(quotas[("a", 0)], 3)

    def test_component_sampling_never_uses_two_anchors_from_same_component(self) -> None:
        frame = pd.DataFrame(
            {
                "category": ["a"] * 4 + ["b"] * 4,
                "target": [0, 0, 1, 1] * 2,
                "component_id": [1, 2, 3, 4, 5, 6, 7, 8],
            }
        )
        indices = sample_component_anchors(frame, 4, seed=7)
        self.assertEqual(len(indices), 4)
        self.assertEqual(frame.loc[indices, "component_id"].nunique(), 4)

    def test_component_pair_sampling_hits_exact_requested_size(self) -> None:
        frame = pd.DataFrame(
            {
                "component_id": [1, 1, 1, 2, 2, 3, 4, 5, 6],
            }
        )
        selected = sample_components_to_pair_count(frame, 6, seed=11)
        self.assertEqual(int(frame["component_id"].isin(selected).sum()), 6)

    def test_hard_selection_preserves_requested_source_fractions(self) -> None:
        size = 100
        frame = pd.DataFrame(
            {
                "category": ["a"] * size,
                "target": [0.0] * 50 + [1.0] * 50,
                "component_id": np.arange(size),
                "model_is_error": [True] * size,
                "model_wrongness": np.linspace(0, 1, size),
                "lexical_hardness": np.linspace(1, 0, size),
                "diagnostic_hardness": np.random.default_rng(7).random(size),
            }
        )
        selected = select_hard_anchors(frame, 50)
        expected = hard_selection_targets(50)
        counts = selected["selection_reason"].value_counts()
        self.assertEqual(counts["confident_minilm_v1_error"], expected.model_errors)
        self.assertEqual(counts["lexical_surprise"], expected.lexical_surprises)
        self.assertEqual(
            counts["model_disagreement_or_order_gap"], expected.diagnostic
        )

    def test_llm_catalog_excludes_all_frozen_validation_items(self) -> None:
        items = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "category": ["A", "A", "OOD", "OOD"],
            }
        )
        pairs = pd.DataFrame(
            {
                "id1": [1, 4],
                "id2": [1, 4],
                "target": [0.1, 0.9],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items_path = root / "items.parquet"
            pairs_path = root / "pairs.parquet"
            output = root / "output"
            items.to_parquet(items_path, index=False)
            pairs.to_parquet(pairs_path, index=False)

            report = split_llm_data(
                items_path,
                pairs_path,
                output,
                ("OOD",),
                np.array([2, 3], dtype=np.int64),
            )

            non_ood_ids = set(
                pd.read_parquet(output / "non_ood_items.parquet")["id"]
            )
            ood_ids = set(pd.read_parquet(output / "ood_items.parquet")["id"])

        self.assertEqual(non_ood_ids, {1})
        self.assertEqual(ood_ids, {4})
        self.assertEqual(
            report["excluded_human_validation_items"],
            {"non_ood": 1, "ood": 1},
        )
        self.assertEqual(report["pairs_touching_human_validation_items"], 0)

    def test_llm_pair_touching_validation_item_is_rejected(self) -> None:
        items = pd.DataFrame({"id": [1, 2], "category": ["A", "A"]})
        pairs = pd.DataFrame({"id1": [1], "id2": [2], "target": [0.5]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items_path = root / "items.parquet"
            pairs_path = root / "pairs.parquet"
            items.to_parquet(items_path, index=False)
            pairs.to_parquet(pairs_path, index=False)

            with self.assertRaisesRegex(
                ValueError,
                "touch frozen human validation items",
            ):
                split_llm_data(
                    items_path,
                    pairs_path,
                    root / "output",
                    ("OOD",),
                    np.array([2], dtype=np.int64),
                )


if __name__ == "__main__":
    unittest.main()
