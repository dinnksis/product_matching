from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analyze_architecture_diversity import (
    MODELS,
    _overlap,
    analyze,
    best_f1_threshold,
    sheet_completions,
)


def test_best_f1_threshold_uses_scores_from_supplied_split() -> None:
    result = best_f1_threshold([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.8])
    assert result["threshold"] == 0.7
    assert result["best_f1"] == 1.0


def test_overlap_counts_four_exclusive_outcomes() -> None:
    result = _overlap(
        np.array([True, True, False, False]),
        np.array([True, False, True, False]),
        np.ones(4, dtype=bool),
    )
    assert result == {
        "n": 4,
        "both_correct": 1,
        "both_wrong": 1,
        "a_correct_b_wrong": 1,
        "b_correct_a_wrong": 1,
    }


def test_full_analysis_row_counts_and_sheet_ids_are_unique() -> None:
    target = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    frames = {}
    for split in ("ordinary", "hard", "OOD"):
        frame = pd.DataFrame({"target": target})
        for index, model in enumerate(MODELS):
            frame[f"{model}_probability"] = np.roll(
                np.array([0.1, 0.2, 0.6, 0.4, 0.8, 0.9]), index
            )
        frames[split] = frame
    _, pairwise, uniqueness, hard, _ = analyze(frames)
    assert len(pairwise) == 18
    assert len(uniqueness) == 12
    assert len(hard) == 3
    completions = sheet_completions(
        pairwise,
        uniqueness,
        hard,
        completed_at="2026-08-25T00:00:00Z",
        source_hash="abc",
    )
    assert len(completions) == 33
    assert len({row["run_id"] for row in completions}) == 33
