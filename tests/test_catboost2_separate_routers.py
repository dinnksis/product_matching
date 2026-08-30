from __future__ import annotations

import numpy as np

from src.catboost1_early_exit import threshold_states
from src.catboost2_separate_routers import (
    apply_separate_thresholds,
    best_two_low_tail_state,
    crossfit_separate_routing,
    routing_summary,
    select_separate_thresholds,
)


def test_two_thresholds_can_allocate_coverage_asymmetrically() -> None:
    # The positive route is deliberately unsafe, so the optimum accepts only
    # the clean negative tail instead of forcing equal thresholds/coverage.
    neg_score = np.linspace(0.0, 1.0, 50)
    neg_error = np.zeros(50, dtype=np.int8)
    pos_score = np.linspace(0.0, 1.0, 20)
    pos_error = np.ones(20, dtype=np.int8)
    state = best_two_low_tail_state(
        threshold_states(neg_score, neg_error, "negative"),
        threshold_states(pos_score, pos_error, "negative"),
        risk_limit=0.1,
        total_examples=70,
    )
    assert state["negative_accepted"] == 50
    assert state["positive_accepted"] == 0
    assert state["threshold_neg"] > 1.0
    assert state["threshold_pos"] == 0.0


def test_select_apply_and_summary_use_combined_risk() -> None:
    q = np.r_[np.linspace(0.0, 0.4, 40), np.linspace(0.0, 0.4, 40)]
    predicted = np.r_[np.zeros(40, dtype=np.int8), np.ones(40, dtype=np.int8)]
    errors = np.zeros(80, dtype=np.int8)
    state = select_separate_thresholds(q, errors, predicted, risk_limit=0.1)
    accepted = apply_separate_thresholds(
        q, predicted, state["threshold_neg"], state["threshold_pos"]
    )
    summary = routing_summary(accepted, errors, predicted, risk_limit=0.1)
    assert summary["accepted"] == 80
    assert summary["negative_accepted"] == 40
    assert summary["positive_accepted"] == 40
    assert summary["passes_risk"]


def test_crossfit_separate_router_covers_all_folds() -> None:
    folds = np.repeat(np.arange(5), 100)
    predicted = np.tile(np.r_[np.zeros(50), np.ones(50)], 5).astype(np.int8)
    q = np.tile(np.linspace(0.0, 1.0, 100), 5)
    errors = np.zeros(500, dtype=np.int8)
    summary, details, accepted = crossfit_separate_routing(
        q, errors, predicted, folds, risk_limit=0.05
    )
    assert len(details) == 5
    assert accepted.shape == (500,)
    assert summary["accepted"] == 500
