from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd


def canonical_pair(first: Any, second: Any) -> tuple[int, int]:
    left, right = int(first), int(second)
    return (left, right) if left <= right else (right, left)


def category_balance_plan(
    human_train: pd.DataFrame,
    categories: pd.Series,
    *,
    target_per_class: int | None = None,
) -> tuple[pd.DataFrame, int]:
    """Plan an equal category × binary-label train while retaining all human rows."""
    if len(human_train) != len(categories):
        raise ValueError("Human train and category lengths differ")
    labels = (human_train["target"].to_numpy(dtype=np.float64) >= 0.5).astype(np.int8)
    counts = (
        pd.DataFrame(
            {
                "category": pd.Series(categories, dtype="string").reset_index(drop=True),
                "label": labels,
            }
        )
        .groupby(["category", "label"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1], fill_value=0)
    )
    if (counts == 0).any().any():
        raise ValueError("Every human category must contain both binary labels")
    minimum_target = int(counts.to_numpy().max())
    selected_target = minimum_target if target_per_class is None else target_per_class
    if selected_target < minimum_target:
        raise ValueError(
            f"target_per_class={selected_target} would remove human rows; "
            f"minimum is {minimum_target}"
        )
    plan = pd.DataFrame(
        {
            "human_negative": counts[0].astype(int),
            "human_positive": counts[1].astype(int),
            "llm_negative_needed": (selected_target - counts[0]).astype(int),
            "llm_positive_needed": (selected_target - counts[1]).astype(int),
        }
    )
    plan.index = plan.index.astype(str)
    plan.index.name = "category"
    return plan, selected_target


def select_llm_supplement(
    pairs: pd.DataFrame,
    category_codes: np.ndarray,
    category_to_code: Mapping[str, int],
    plan: pd.DataFrame,
    *,
    forbidden_item_ids: Iterable[int] = (),
    forbidden_pairs: Iterable[tuple[int, int]] = (),
    llm_weight: float = 0.35,
    seed: int = 42,
) -> pd.DataFrame:
    """Select highest-confidence weak labels required by a category balance plan."""
    if len(pairs) != len(category_codes):
        raise ValueError("LLM pair and category-code lengths differ")
    if not 0 < llm_weight <= 1:
        raise ValueError("llm_weight must be in (0, 1]")
    raw_targets = pairs["target"].to_numpy(dtype=np.float64)
    if not np.isfinite(raw_targets).all() or not ((0 <= raw_targets) & (raw_targets <= 1)).all():
        raise ValueError("LLM targets must be finite probabilities in [0, 1]")

    first_ids = pairs["id1"].to_numpy(dtype=np.int64)
    second_ids = pairs["id2"].to_numpy(dtype=np.int64)
    forbidden_ids = np.fromiter(
        (int(value) for value in forbidden_item_ids), dtype=np.int64
    )
    eligible = np.ones(len(pairs), dtype=bool)
    if len(forbidden_ids):
        eligible &= ~np.isin(first_ids, forbidden_ids)
        eligible &= ~np.isin(second_ids, forbidden_ids)

    used_pairs = {canonical_pair(first, second) for first, second in forbidden_pairs}
    labels = (raw_targets >= 0.5).astype(np.int8)
    rng = np.random.default_rng(seed)
    selected_positions: list[int] = []
    selected_categories: list[str] = []

    for category in sorted(plan.index):
        if category not in category_to_code:
            raise ValueError(f"LLM items do not contain category {category!r}")
        code = category_to_code[category]
        for label, needed_column in (
            (0, "llm_negative_needed"),
            (1, "llm_positive_needed"),
        ):
            needed = int(plan.at[category, needed_column])
            if not needed:
                continue
            group_mask = eligible & (category_codes == code) & (labels == label)
            group_positions = np.flatnonzero(group_mask)
            levels = np.unique(raw_targets[group_positions])
            levels.sort()
            if label == 1:
                levels = levels[::-1]

            accepted = 0
            for level in levels:
                level_positions = group_positions[
                    raw_targets[group_positions] == level
                ].copy()
                rng.shuffle(level_positions)
                for position in level_positions:
                    key = canonical_pair(first_ids[position], second_ids[position])
                    if key in used_pairs:
                        continue
                    used_pairs.add(key)
                    selected_positions.append(int(position))
                    selected_categories.append(category)
                    accepted += 1
                    if accepted == needed:
                        break
                if accepted == needed:
                    break
            if accepted != needed:
                raise ValueError(
                    f"Not enough unique LLM pairs for {category!r}, label={label}: "
                    f"needed {needed}, selected {accepted}"
                )

    selected = pairs.iloc[selected_positions][["id1", "id2"]].reset_index(drop=True)
    selected_raw_targets = raw_targets[selected_positions]
    selected_labels = labels[selected_positions].astype(np.float64)
    confidence = np.abs(selected_raw_targets - 0.5) * 2.0
    selected["target"] = selected_labels
    selected["llm_target_raw"] = selected_raw_targets
    selected["label_source"] = "llm"
    selected["sample_weight"] = (llm_weight * confidence).astype(np.float32)
    selected["_category"] = selected_categories
    return selected
