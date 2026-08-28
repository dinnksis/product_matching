from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.benefit_router import (
    assert_component_disjoint,
    assert_pair_ids_disjoint,
    benefit_targets,
    deterministic_random_priority,
    router_feature_frame,
)


def test_benefit_targets_have_expected_direction_and_margin() -> None:
    target = [1, 0, 1]
    regression, classification = benefit_targets(
        target,
        [0.2, 0.8, 0.7],
        [0.9, 0.1, 0.71],
        classification_margin=0.02,
    )
    assert regression[0] > 0
    assert regression[1] > 0
    assert 0 < regression[2] < 0.02
    assert classification.tolist() == [1, 1, 0]


def test_router_features_use_bge_but_reject_specialist_columns() -> None:
    cheap = pd.DataFrame(
        {
            "category": ["a", "b"],
            "title_token_set": [0.2, 0.8],
            "rule_any_fired": [0.0, 1.0],
        }
    )
    bge = pd.DataFrame({"score": [0.2, 0.6], "score_order_gap": [0.01, 0.03]})
    features, categorical = router_feature_frame(cheap, bge)
    assert categorical == ["category"]
    assert "bge_entropy" in features
    assert "bge_score_order_gap" in features
    assert "rule_any_fired" not in features

    with pytest.raises(ValueError, match="Forbidden"):
        router_feature_frame(cheap.assign(minilm_score=[0.1, 0.2]), bge)


def test_component_disjoint_check() -> None:
    assert_component_disjoint([0, 0, 1], [10, 10, 20])
    with pytest.raises(AssertionError, match="component"):
        assert_component_disjoint([0, 1], [10, 10])

    assert_pair_ids_disjoint(pd.DataFrame({"id1": [1, 3], "id2": [2, 4], "fold": [0, 1]}))
    with pytest.raises(AssertionError, match="product id"):
        assert_pair_ids_disjoint(
            pd.DataFrame({"id1": [1, 2], "id2": [2, 3], "fold": [0, 1]})
        )


def test_random_priority_is_stable_and_row_order_independent() -> None:
    first = deterministic_random_priority([1, 2], [8, 9], 2026)
    second = deterministic_random_priority([2, 1], [9, 8], 2026)
    np.testing.assert_allclose(first, second[::-1])
