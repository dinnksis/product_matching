from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.evaluate_compute_allocations import (
    aggregate_scores,
    exclusive_masks,
    restricted_top_mask,
    sequential_features,
)


def test_exclusive_masks_fill_exact_nonoverlapping_quotas():
    size = 20
    pairs = pd.DataFrame({"id1": np.arange(size), "id2": np.arange(size) + 100})
    mini, ru = exclusive_masks(
        np.linspace(0, 1, size),
        np.linspace(1, 0, size),
        0.40,
        0.10,
        pairs["id1"],
        pairs["id2"],
    )
    assert mini.sum() == 8
    assert ru.sum() == 2
    assert not np.any(mini & ru)


def test_hierarchical_mask_is_restricted():
    pairs = pd.DataFrame({"id1": np.arange(10), "id2": np.arange(10) + 20})
    eligible = np.array([True] * 4 + [False] * 6)
    routed = restricted_top_mask(
        np.arange(10), eligible, 0.20, pairs["id1"], pairs["id2"]
    )
    assert routed.sum() == 2
    assert not np.any(routed & ~eligible)


def test_sequential_features_use_only_bge_and_minilm():
    base = pd.DataFrame({"category": ["a", "b"], "cheap": [1.0, 2.0]})
    result = sequential_features(base, np.array([0.2, 0.8]), np.array([0.4, 0.7]))
    assert "rumodernbert_probability" not in result
    assert np.allclose(result["bge_minilm_disagreement"], [0.2, 0.1])


def test_hierarchical_aggregation_blends_with_current_score():
    frame = pd.DataFrame(
        {
            "bge_probability": [0.2, 0.3],
            "minilm_probability": [0.6, 0.7],
            "rumodernbert_probability": [0.9, 0.1],
        }
    )
    mini = np.array([True, False])
    ru = np.array([True, False])
    score = aggregate_scores(frame, mini, ru, 0.5, 0.25, "hierarchical")
    assert np.allclose(score, [0.525, 0.3])
