from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from src.catboost1_rule_portfolio import (
    apply_policy,
    build_candidate_matrix,
    select_portfolio,
)


def test_candidate_matrix_uses_only_conditions_and_builds_compounds() -> None:
    rows = 4
    base = pd.DataFrame({
        "category": ["optics"] * rows,
        "brand_match": [1, 1, 0, 0],
        "model_code_match": [0] * rows,
        "title_code_match": [0] * rows,
        "title_token_set": [0.9, 0.8, 0.7, 0.6],
        "title_exact": [0] * rows,
        "title_code_conflict": [0] * rows,
    })
    definitions = pd.DataFrame({
        "rule_id": ["r1", "r2"],
        "canonical_rule": ["pupillary differs", "lens width differs"],
        "concept": ["pupillary_distance", "lens_width"],
        "relation": ["different_value", "different_value"],
    })
    global_rules = sparse.csr_matrix(np.array([
        [1, 1], [1, 0], [0, 1], [0, 0],
    ], dtype=np.uint8))
    category_rules = global_rules.copy()
    matrix, catalog = build_candidate_matrix(
        base, global_rules, category_rules, definitions,
        ["optics||0", "optics||1"],
    )
    assert matrix.shape[0] == rows
    assert set(catalog["source"]) >= {
        "mined_rule_global", "mined_rule_category", "mined_rule_anchor", "semantic_compound",
    }
    compound = catalog.loc[catalog["concept"].eq("optical_ge2"), "candidate_index"]
    assert len(compound) == 2  # global and category versions
    assert matrix[0, int(compound.iloc[0])] == 1


def test_portfolio_selection_and_application_are_separate() -> None:
    rows = 60
    scores = np.full(rows, 0.5)
    target = np.r_[np.zeros(45, dtype=np.int8), np.ones(15, dtype=np.int8)]
    folds = np.arange(rows) % 3
    matrix = sparse.csr_matrix(np.r_[np.ones(40), np.zeros(20)][:, None])
    catalog = pd.DataFrame({"candidate_index": [0], "name": ["clean_rule"]})
    vetoes = {"none": np.zeros(rows, dtype=bool)}
    policy = select_portfolio(
        scores, target, folds, matrix, catalog, vetoes,
        risk_limit=0.2, score_caps=[0.8], seed_fractions=[1.0],
        minimum_support=10, minimum_folds=3, maximum_gates=2,
    )
    assert policy.calibration_accepted == 40
    assert policy.calibration_errors == 0
    assert [gate.name for gate in policy.gates] == ["clean_rule"]
    accepted, reasons = apply_policy(policy, scores, matrix, vetoes)
    assert accepted.sum() == 40
    assert set(reasons[accepted]) == {"clean_rule"}
