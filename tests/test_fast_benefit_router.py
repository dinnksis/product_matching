from __future__ import annotations

import numpy as np

from src.fast_benefit_router import bge_feature_frame, runtime_feature_frame


def test_score_category_has_frozen_schema():
    frame = runtime_feature_frame(
        ["a", "b"], ["x", "y"], ["x", "z"], [0.2, 0.8], [-1.3862944, 1.3862944], "score_category"
    )
    assert frame.columns.tolist() == [
        "category", "bge_probability", "bge_logit", "bge_abs_from_half",
        "bge_uncertainty", "bge_entropy", "bge_raw_logit",
    ]


def test_title_features_are_symmetric_but_inference_is_not():
    forward = runtime_feature_frame(
        ["a"], ["Phone X 128 GB"], ["phone x 256 gb"], [0.4], [-0.4], "score_title"
    )
    reverse = runtime_feature_frame(
        ["a"], ["phone x 256 gb"], ["Phone X 128 GB"], [0.4], [-0.4], "score_title"
    )
    title_columns = [column for column in forward if column.startswith("title_")]
    assert np.allclose(forward[title_columns], reverse[title_columns])


def test_bge_entropy_is_finite_at_probability_edges():
    frame = bge_feature_frame([0.0, 1.0], [-20.0, 20.0])
    assert np.isfinite(frame.to_numpy()).all()

