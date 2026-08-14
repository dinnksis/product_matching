from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

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


if __name__ == "__main__":
    unittest.main()
