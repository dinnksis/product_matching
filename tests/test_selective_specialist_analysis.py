from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.selective_specialist_analysis import (
    add_pairwise_proxies,
    choose_oracle_alternative,
    compact_oracle_summary,
    normalized_rank,
    oracle_route_mask,
    oracle_routing_rows,
    safe_macro_average_precision,
)


class SelectiveSpecialistAnalysisTest(unittest.TestCase):
    def test_pairwise_proxy_directions(self) -> None:
        frame = pd.DataFrame(
            {
                "target": [1.0, 0.0, 1.0],
                "bge_probability": [0.2, 0.2, 0.8],
                "minilm_probability": [0.9, 0.8, 0.81],
            }
        )
        result = add_pairwise_proxies(frame, "minilm")
        self.assertEqual(
            result["binary_status"].tolist(),
            ["specialist_better", "bge_better", "approximately_equal"],
        )
        self.assertGreater(result.loc[0, "absolute_error_gain"], 0)
        self.assertLess(result.loc[1, "absolute_error_gain"], 0)

    def test_best_expert_is_chosen_per_row(self) -> None:
        target = np.asarray([1.0, 0.0])
        base = np.asarray([0.2, 0.8])
        selected, benefit, chosen = choose_oracle_alternative(
            target,
            base,
            {
                "minilm": np.asarray([0.9, 0.7]),
                "rumodernbert": np.asarray([0.6, 0.1]),
            },
            "best_expert",
        )
        np.testing.assert_allclose(selected, [0.9, 0.1])
        self.assertEqual(chosen.tolist(), ["minilm", "rumodernbert"])
        self.assertTrue((benefit > 0).all())

    def test_route_budget_and_positive_benefit(self) -> None:
        benefit = np.asarray([0.9, 0.8, 0.7, -1.0, 0.6, 0.5, 0.4, 0.0])
        category = np.asarray(["a"] * 4 + ["b"] * 4)
        global_mask = oracle_route_mask(benefit, category, 0.25, "global_directional")
        self.assertEqual(int(global_mask.sum()), 2)
        balanced = oracle_route_mask(
            benefit, category, 0.25, "category_balanced_directional"
        )
        self.assertEqual(int(balanced.sum()), 2)
        self.assertTrue(balanced[0])
        self.assertTrue(balanced[4])

    def test_oracle_improves_simple_macro_ap(self) -> None:
        frame = pd.DataFrame(
            {
                "target": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                "category": ["a"] * 4 + ["b"] * 4,
                "bge_probability": [0.2, 0.8, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6],
                "minilm_probability": [0.9, 0.1, 0.8, 0.2, 0.8, 0.2, 0.7, 0.3],
                "rumodernbert_probability": [0.7, 0.3, 0.6, 0.4, 0.9, 0.1, 0.8, 0.2],
            }
        )
        rows = oracle_routing_rows(
            frame,
            "iid",
            budgets=(0.5,),
            score_modes=("replace",),
            policies=("category_balanced_directional",),
        )
        result = pd.DataFrame(rows)
        self.assertTrue((result["macro_ap_gain"] > 0).all())
        compact = compact_oracle_summary(result)
        self.assertEqual(len(compact), 3)

    def test_rank_and_macro_ap(self) -> None:
        np.testing.assert_allclose(normalized_rank([0.1, 0.4, 0.4, 0.9]), [0.25, 0.625, 0.625, 1.0])
        value, categories = safe_macro_average_precision(
            [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], ["a", "a", "b", "b"]
        )
        self.assertEqual(value, 1.0)
        self.assertEqual(categories, 2)


if __name__ == "__main__":
    unittest.main()
