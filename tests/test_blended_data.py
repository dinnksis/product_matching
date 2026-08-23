from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.blended_data import category_balance_plan, select_llm_supplement


class BlendedDataTest(unittest.TestCase):
    def test_plan_retains_all_human_rows_and_equalizes_categories(self) -> None:
        human = pd.DataFrame(
            {
                "target": [0, 0, 0, 1, 0, 1, 1, 1],
            }
        )
        categories = pd.Series(["a"] * 4 + ["b"] * 4)
        plan, target = category_balance_plan(human, categories)
        self.assertEqual(target, 3)
        self.assertEqual(plan.loc["a"].to_dict(), {
            "human_negative": 3,
            "human_positive": 1,
            "llm_negative_needed": 0,
            "llm_positive_needed": 2,
        })
        self.assertEqual(plan.loc["b"].to_dict(), {
            "human_negative": 1,
            "human_positive": 3,
            "llm_negative_needed": 2,
            "llm_positive_needed": 0,
        })

    def test_selection_prefers_confident_labels_and_excludes_validation(self) -> None:
        plan = pd.DataFrame(
            {
                "human_negative": [1],
                "human_positive": [1],
                "llm_negative_needed": [1],
                "llm_positive_needed": [2],
            },
            index=pd.Index(["a"], name="category"),
        )
        candidates = pd.DataFrame(
            {
                "id1": [10, 20, 30, 40, 50, 60],
                "id2": [11, 21, 31, 41, 51, 61],
                "target": [0.0, 0.222222, 1.0, 0.888889, 1.0, 1.0],
            }
        )
        selected = select_llm_supplement(
            candidates,
            np.zeros(len(candidates), dtype=np.int8),
            {"a": 0},
            plan,
            forbidden_item_ids={51},
            forbidden_pairs={(60, 61)},
            llm_weight=0.35,
            seed=7,
        )
        self.assertEqual(set(selected["id1"]), {10, 30, 40})
        self.assertEqual(selected["target"].value_counts().to_dict(), {1.0: 2, 0.0: 1})
        exact_positive_weight = selected.loc[selected["id1"].eq(30), "sample_weight"].item()
        weak_positive_weight = selected.loc[selected["id1"].eq(40), "sample_weight"].item()
        self.assertAlmostEqual(exact_positive_weight, 0.35, places=6)
        self.assertLess(weak_positive_weight, exact_positive_weight)


if __name__ == "__main__":
    unittest.main()
