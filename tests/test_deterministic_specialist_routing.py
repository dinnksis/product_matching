import unittest

import numpy as np
import pandas as pd

from src.deterministic_specialist_routing import (
    domain_conflict_priority,
    estimate_private_t4_runtime,
    expert_assignment,
    routed_scores,
    top_budget_mask,
    uncertainty_abs,
    uncertainty_entropy,
)


def frame() -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "id1": np.arange(10),
            "id2": np.arange(100, 110),
            "target": np.arange(10) % 2,
            "bge_probability": np.linspace(0.05, 0.95, 10),
            "minilm_probability": np.linspace(0.15, 0.85, 10),
            "rumodernbert_probability": np.linspace(0.25, 0.75, 10),
            "model_code_conflict": [0, 1] * 5,
            "title_code_conflict": [0] * 10,
            "brand_conflict": [0] * 10,
            "title_token_set": np.linspace(0.3, 1.0, 10),
            "attribute_key_jaccard": np.linspace(0.0, 0.5, 10),
        }
    )
    for column in (
        "size", "ram_storage", "volume", "weight", "dimensions",
        "pack_count", "power", "optical",
    ):
        result[f"num_{column}_conflict"] = 0
    result.loc[1, ["num_size_conflict", "num_weight_conflict"]] = 1
    return result


class DeterministicSpecialistRoutingTest(unittest.TestCase):
    def test_abs_and_entropy_have_same_route_order(self) -> None:
        current = frame()
        for budget in (0.1, 0.3, 0.8):
            by_abs = top_budget_mask(
                uncertainty_abs(current.bge_probability), budget, current.id1, current.id2
            )
            by_entropy = top_budget_mask(
                uncertainty_entropy(current.bge_probability), budget, current.id1, current.id2
            )
            np.testing.assert_array_equal(by_abs, by_entropy)

    def test_route_priority_is_label_invariant(self) -> None:
        current = frame()
        before = domain_conflict_priority(current)
        current["target"] = 1 - current["target"]
        after = domain_conflict_priority(current)
        np.testing.assert_array_equal(before, after)

    def test_budget_is_exact_floor(self) -> None:
        current = frame()
        mask = top_budget_mask(np.arange(10), 0.3, current.id1, current.id2)
        self.assertEqual(int(mask.sum()), 3)

    def test_dynamic_assignment_uses_complex_conflict(self) -> None:
        current = frame()
        routed = np.ones(len(current), dtype=bool)
        assigned = expert_assignment(current, routed, "dynamic_conflict")
        self.assertEqual(assigned[1], "rumodernbert")
        self.assertEqual(assigned[0], "minilm")

    def test_routed_blend_preserves_unrouted_bge(self) -> None:
        current = frame()
        assigned = np.full(len(current), "bge", dtype=object)
        assigned[0] = "minilm"
        score = routed_scores(current, assigned, "blend_50_50")
        self.assertAlmostEqual(score[0], 0.10)
        np.testing.assert_array_equal(score[1:], current.bge_probability.to_numpy()[1:])

    def test_runtime_is_monotonic(self) -> None:
        measurements = pd.DataFrame(
            {
                "model": ["bge", "minilm", "rumodernbert"],
                "pairs_per_second": [50.0, 300.0, 100.0],
                "load_seconds": [20.0, 1.0, 4.0],
            }
        )
        baseline = estimate_private_t4_runtime(1000, 0.0, 0.0, measurements)
        routed = estimate_private_t4_runtime(1000, 0.1, 0.0, measurements)
        self.assertGreater(routed["estimated_private_t4_seconds"], baseline["estimated_private_t4_seconds"])


if __name__ == "__main__":
    unittest.main()
