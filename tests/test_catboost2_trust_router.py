from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.catboost2_trust_router import (
    assert_component_disjoint,
    confidence_features,
    crossfit_threshold_routing,
    decision_error,
    trust_feature_frame,
)


def test_confidence_features_and_error_target() -> None:
    probability = np.array([0.1, 0.6, 0.5])
    target = np.array([0, 0, 1])
    features = confidence_features(probability)

    assert features["cb1_predicted_class"].tolist() == [0.0, 1.0, 1.0]
    assert features["min_p_cb1_one_minus_p"].tolist() == pytest.approx([0.1, 0.4, 0.5])
    assert decision_error(probability, target).tolist() == [0, 1, 0]


def test_trust_frame_contains_frozen_base_and_confidence() -> None:
    base = pd.DataFrame({
        "title_ratio": [0.8],
        "rule_raw_candidate_count": [2.0],
        "category": ["electronics"],
        "conflict_signature": ["ram_storage"],
    })
    frame, categorical = trust_feature_frame(base, np.array([0.2]))
    assert "title_ratio" in frame
    assert "rule_raw_candidate_count" not in frame
    assert "p_cb1" in frame
    assert categorical == ["category"]


def test_crossfit_router_applies_thresholds_to_unseen_folds() -> None:
    folds = np.repeat(np.arange(5), 100)
    target_error = np.zeros(500, dtype=np.int8)
    target_error[[99, 199, 299, 399, 499]] = 1
    score = np.linspace(0.0, 1.0, 500)
    summary, details, accepted = crossfit_threshold_routing(
        score, target_error, folds, risk_limit=0.1
    )
    assert len(details) == 5
    assert accepted.shape == (500,)
    assert summary["accepted"] == int(accepted.sum())


def test_component_disjoint_assertion() -> None:
    assert_component_disjoint(np.array([0, 0, 1]), np.array([10, 10, 20]))
    with pytest.raises(AssertionError):
        assert_component_disjoint(np.array([0, 1]), np.array([10, 10]))
