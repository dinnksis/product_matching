from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.catboost1_early_exit import (
    best_side_state,
    build_item_record,
    best_total_state,
    category_balanced_weights,
    extract_pair_features,
    load_label_free_rule_registry,
    semantic_family,
    fold_rule_evidence,
    threshold_states,
    variant_frame,
    wilson_upper,
)


def test_typed_size_conflict_is_not_generic_numeric_conflict() -> None:
    items = pd.DataFrame([
        {
            "id": 1,
            "name": "Футболка Acme размер 42",
            "attributes": json.dumps({"Бренд": "Acme", "Размер": "42"}, ensure_ascii=False),
            "category": "Одежда",
        },
        {
            "id": 2,
            "name": "Футболка Acme размер 44",
            "attributes": json.dumps({"Бренд": "Acme", "Размер": "44"}, ensure_ascii=False),
            "category": "Одежда",
        },
    ])
    pairs = pd.DataFrame({"id1": [1], "id2": [2], "target": [1]})
    features, fired = extract_pair_features(
        pairs, items, learned_concepts={}, rule_registry={("size", "different_value"): 0}
    )

    assert features.loc[0, "num_size_conflict"] == 1
    assert features.loc[0, "num_ram_storage_conflict"] == 0
    assert "numeric_conflict" not in features.columns
    assert features.loc[0, "category_primary_conflict"] == "Одежда||size"
    assert features.loc[0, "matching_regime"] == "variant_tolerant"
    assert fired == [[0]]


def test_rule_train_encoding_removes_the_rows_own_label() -> None:
    # Only row zero fires the rule. Its LOO statistics therefore have zero support
    # and cannot encode its own positive target.
    matrix = sparse.csr_matrix(np.array([[1.0], [0.0], [0.0]], dtype=np.float32))
    target = np.array([1, 0, 1], dtype=np.int8)
    train, valid = fold_rule_evidence(
        matrix, target, np.array([0, 1]), np.array([2]), prior_strength=2.0
    )

    assert train.loc[0, "rule_evidence_support_sum_log"] == pytest.approx(0.0)
    assert train.loc[0, "rule_positive_evidence_sum"] == pytest.approx(0.0)
    assert train.loc[0, "rule_negative_evidence_sum"] == pytest.approx(0.0)
    assert valid.loc[0, "rule_evidence_fired"] == pytest.approx(0.0)


def test_threshold_sweep_keeps_equal_scores_together_and_uses_strict_thresholds() -> None:
    scores = np.array([0.1, 0.1, 0.8, 0.9])
    target = np.array([0, 0, 1, 1])
    negative = threshold_states(scores, target, "negative")
    positive = threshold_states(scores, target, "positive")

    assert negative.accepted.tolist() == [0, 2, 3, 4]
    first_tail = negative.iloc[1]
    assert first_tail.threshold > 0.1
    assert np.sum(scores < first_tail.threshold) == 2
    top_tail = positive.loc[positive.accepted == 2].iloc[0]
    assert top_tail.threshold < 0.8
    assert np.sum(scores > top_tail.threshold) == 2


def test_wilson_and_best_risk_coverage_states() -> None:
    assert float(wilson_upper(0, 4000)) < 0.001
    scores = np.r_[np.full(4000, 0.01), np.full(4000, 0.99)]
    target = np.r_[np.zeros(4000, dtype=np.int8), np.ones(4000, dtype=np.int8)]
    negative = threshold_states(scores, target, "negative")
    positive = threshold_states(scores, target, "positive")

    neg_best = best_side_state(negative, 0.001)
    total_best = best_total_state(negative, positive, 0.001, len(target))
    assert neg_best["accepted"] == 4000
    assert total_best["accepted"] == 8000
    assert total_best["errors"] == 0


def test_variant_feature_contract() -> None:
    base = pd.DataFrame({
        "cheap": [0.5],
        "category": ["Электроника"],
        "primary_conflict_type": ["ram_storage"],
        "conflict_signature": ["ram_storage"],
        "category_primary_conflict": ["Электроника||ram_storage"],
        "category_conflict_signature": ["Электроника||ram_storage"],
        "matching_regime": ["configuration_sensitive"],
        "regime_conflict_signature": ["configuration_sensitive||ram_storage"],
    })
    evidence = pd.DataFrame({"rule_positive_evidence_sum": [0.2]})

    v1, cat1 = variant_frame(base, evidence, None, "V1_global")
    v3, cat3 = variant_frame(base, evidence, evidence, "V3_category_aware")
    assert "category" not in v1
    assert cat1 == []
    assert "category_conflict_signature" in v3
    assert "category_conflict_signature" in cat3
    assert "matching_regime" not in v3


def test_category_balanced_weights_equalize_category_mass() -> None:
    categories = np.array(["a", "a", "a", "b"])
    weights = category_balanced_weights(categories)
    assert weights[:3].sum() == pytest.approx(weights[3:].sum())


def test_dimension_chain_normalizes_cm_and_mm() -> None:
    items = pd.DataFrame([
        {"id": 1, "name": "Сейф 5,5x15,5x24 см", "attributes": "{}", "category": "Дом и сад"},
        {"id": 2, "name": "Сейф 55x155x240 мм", "attributes": "{}", "category": "Дом и сад"},
    ])
    features, _ = extract_pair_features(
        pd.DataFrame({"id1": [1], "id2": [2], "target": [1]}),
        items,
        learned_concepts={},
        rule_registry={},
    )
    assert features.loc[0, "num_dimensions_match"] == 1
    assert features.loc[0, "num_dimensions_conflict"] == 0


def test_partial_numeric_overlap_retains_a_conflict() -> None:
    items = pd.DataFrame([
        {
            "id": 1, "name": "Футболка", "attributes": json.dumps({"Размер": ["42", "44"]}, ensure_ascii=False),
            "category": "Одежда",
        },
        {
            "id": 2, "name": "Футболка", "attributes": json.dumps({"Размер": ["42", "46"]}, ensure_ascii=False),
            "category": "Одежда",
        },
    ])
    features, _ = extract_pair_features(
        pd.DataFrame({"id1": [1], "id2": [2], "target": [1]}), items, {}, {}
    )
    assert features.loc[0, "num_size_match"] == 1
    assert features.loc[0, "num_size_conflict"] == 1


def test_rule_registry_enforces_label_free_role_and_relation_whitelist(tmp_path) -> None:
    path = tmp_path / "definitions.parquet"
    pd.DataFrame([
        {
            "rule_id": "good", "canonical_rule": "size differs", "concept": "size",
            "relation": "different_value", "rule_role": "RULE_CANDIDATE",
        },
        {
            "rule_id": "context", "canonical_rule": "size missing", "concept": "size",
            "relation": "missing_one_side", "rule_role": "CONTEXT_ONLY",
        },
        {
            "rule_id": "review", "canonical_rule": "size unknown", "concept": "size",
            "relation": "unknown", "rule_role": "REVIEW_BEFORE_USE",
        },
    ]).to_parquet(path, index=False)
    registry, definitions = load_label_free_rule_registry(
        path,
        allowed_relations={"different_value", "specificity_difference"},
        allowed_roles={"RULE_CANDIDATE"},
    )
    assert definitions.rule_id.tolist() == ["good"]
    assert registry == {("size", "different_value"): 0}


def test_semantic_family_does_not_use_english_substrings() -> None:
    assert semantic_family("frame_construction") == "optical"
    assert semantic_family("optical_power") == "optical"
    assert semantic_family("country_of_origin") is None


def test_legacy_semantics_reproduces_frozen_catboost1_contract() -> None:
    attributes = json.dumps({
        "frame_construction": "12",
        "country_of_origin": "2024",
        "optical_power": "+3.5",
    })
    current = build_item_record("item", attributes, {}, legacy_semantics=False)
    legacy = build_item_record("item", attributes, {}, legacy_semantics=True)

    assert "12" in current.typed_values["optical"]
    assert "12" in legacy.typed_values["ram_storage"]
    assert "2024" not in current.typed_values.get("pack_count", set())
    assert "2024" in legacy.typed_values["pack_count"]
    assert "3.5" in current.typed_values["optical"]
    assert "3.5" in legacy.typed_values["power"]
