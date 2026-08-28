from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import evaluate_architecture_ensembles as ensemble


def test_full_nonempty_combination_space() -> None:
    combinations = ensemble.model_combinations()
    assert len(combinations) == 15
    assert sum(len(item) == 2 for item in combinations) == 6
    assert sum(len(item) == 3 for item in combinations) == 4
    assert combinations[-1] == ensemble.MODELS


def test_normalized_rank_definition_and_macro_ap() -> None:
    values = pd.Series([0.1, 0.4, 0.4, 0.9])
    ranks = values.rank(method="average", ascending=True).to_numpy() / len(values)
    np.testing.assert_allclose(ranks, [0.25, 0.625, 0.625, 1.0])
    frame = pd.DataFrame(
        {
            "target": [0.0, 1.0, 1.0, 0.0],
            "category": ["a", "a", "b", "b"],
        }
    )
    assert ensemble.macro_average_precision(
        frame, np.asarray([0.1, 0.9, 0.8, 0.2])
    ) == 1.0


def test_run_ids_are_stable_and_method_specific() -> None:
    first = ensemble.deterministic_run_id(("gte", "bge"), "mean_probability")
    assert first == ensemble.deterministic_run_id(
        ("gte", "bge"), "mean_probability"
    )
    assert first != ensemble.deterministic_run_id(("gte", "bge"), "mean_rank")
