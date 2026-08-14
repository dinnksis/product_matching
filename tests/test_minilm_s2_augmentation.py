from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.minilm_s2_augmentation import group_metrics, serialize_values_shuffled


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from train_minilm_s2_augmentation import ForwardLengthBucketBatchSampler


def test_attribute_shuffle_is_deterministic_and_keeps_title_first() -> None:
    attributes = [("brand", "samsung"), ("memory", "256 gb"), ("color", "black")]
    first = serialize_values_shuffled("Galaxy S24", attributes, seed=7)
    second = serialize_values_shuffled("Galaxy S24", attributes, seed=7)

    assert first == second
    assert first.startswith("galaxy s24. ")
    assert "brand" not in first and "memory" not in first and "color" not in first
    assert all(first.count(value) == 1 for _, value in attributes)


def test_attribute_shuffle_changes_with_epoch_seed() -> None:
    attributes = [(f"key-{index}", f"value-{index}") for index in range(8)]
    assert serialize_values_shuffled("title", attributes, seed=1) != serialize_values_shuffled(
        "title", attributes, seed=2
    )


def test_baseline_sampler_never_reverses_and_visits_every_example_once() -> None:
    sampler = ForwardLengthBucketBatchSampler(
        np.arange(1, 18), batch_size=4, bucket_size_multiplier=2, seed=42
    )
    rows = [entry for batch in sampler for entry in batch]
    assert sorted(index for index, _ in rows) == list(range(17))
    assert not any(reverse for _, reverse in rows)


def test_hard_group_metrics_avoids_positive_only_ap() -> None:
    predictions = pd.DataFrame(
        {
            "id1": [1, 2, 3, 4],
            "id2": [11, 12, 13, 14],
            "target": [1, 0, 1, 0],
            "category": ["a", "a", "a", "a"],
            "score": [0.9, 0.8, 0.7, 0.1],
        }
    )
    groups = pd.DataFrame(
        {
            "id1": [1, 2, 3, 4],
            "id2": [11, 12, 13, 14],
            "target": [1, 0, 1, 0],
            "category": ["a", "a", "a", "a"],
            "high_name_similarity": [True, True, False, False],
            "critical_variant_conflict": [False, True, True, False],
            "hard_looking": [True, True, True, False],
        }
    )

    metrics = group_metrics(predictions, groups)

    assert metrics["positive_only"]["pr_auc"] is None
    assert metrics["positive_only"]["support"] == 2
    assert metrics["hard_looking"]["support"] == 3
    assert metrics["hard_looking"]["macro_average_precision"] is not None
