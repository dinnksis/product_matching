from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PairwiseMarginResult:
    loss: Any
    pair_count: Any
    mean_teacher_abs_margin: Any
    mean_student_abs_margin: Any


def soft_probability_logit(
    probabilities: Any,
    *,
    epsilon: float,
) -> Any:
    """Convert soft probabilities to finite teacher logits."""
    import torch

    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be in (0, 0.5)")
    return torch.logit(probabilities.float().clamp(epsilon, 1.0 - epsilon))


def pairwise_margin_huber_loss(
    *,
    student_logits: Any,
    teacher_probabilities: Any,
    category_ids: Any,
    temperature: float = 1.0,
    huber_delta: float = 1.0,
    logit_epsilon: float = 1e-4,
    min_teacher_gap: float = 0.0,
) -> PairwiseMarginResult:
    """Match within-category teacher margins using deterministic extreme pairs.

    Within each category in the local mini-batch, examples are sorted by their
    teacher logits.  The lower half is paired with the upper half.  This avoids
    spending most comparisons on equal zero targets in the highly imbalanced
    weak-label corpus and keeps the number of comparisons linear in batch size.
    """
    import torch
    import torch.nn.functional as F

    if student_logits.ndim != 1 or teacher_probabilities.ndim != 1:
        raise ValueError("student_logits and teacher_probabilities must be vectors")
    if category_ids.ndim != 1:
        raise ValueError("category_ids must be a vector")
    if not (
        len(student_logits) == len(teacher_probabilities) == len(category_ids)
    ):
        raise ValueError("Pairwise margin inputs must have equal lengths")
    if temperature <= 0 or huber_delta <= 0:
        raise ValueError("temperature and huber_delta must be positive")
    if min_teacher_gap < 0:
        raise ValueError("min_teacher_gap must be non-negative")

    teacher_logits = soft_probability_logit(
        teacher_probabilities,
        epsilon=logit_epsilon,
    )
    # Stable sorting twice is a tensor-only lexsort by (category, teacher logit).
    # Keeping this vectorized avoids a CPU/GPU synchronization per category.
    teacher_order = torch.argsort(teacher_logits, stable=True)
    order = teacher_order[
        torch.argsort(category_ids[teacher_order], stable=True)
    ]
    ordered_categories = category_ids[order]
    positions = torch.arange(len(order), device=order.device)

    group_start_mask = torch.ones_like(ordered_categories, dtype=torch.bool)
    group_start_mask[1:] = ordered_categories[1:] != ordered_categories[:-1]
    start_markers = torch.where(
        group_start_mask,
        positions,
        torch.zeros_like(positions),
    )
    group_starts = torch.cummax(start_markers, dim=0).values

    group_end_mask = torch.ones_like(ordered_categories, dtype=torch.bool)
    group_end_mask[:-1] = ordered_categories[:-1] != ordered_categories[1:]
    end_markers = torch.where(
        group_end_mask,
        positions + 1,
        torch.full_like(positions, len(order)),
    )
    group_ends = torch.flip(
        torch.cummin(torch.flip(end_markers, dims=(0,)), dim=0).values,
        dims=(0,),
    )

    group_sizes = group_ends - group_starts
    within_group = positions - group_starts
    half_sizes = torch.div(group_sizes, 2, rounding_mode="floor")
    lower_mask = within_group < half_sizes
    lower_positions = positions[lower_mask]
    upper_positions = (
        group_starts + group_sizes - half_sizes + within_group
    )[lower_mask]
    lower = order[lower_positions]
    upper = order[upper_positions]

    raw_teacher_margin = teacher_logits[upper] - teacher_logits[lower]
    student_margin = student_logits[upper] - student_logits[lower]
    teacher_margin = raw_teacher_margin / temperature
    keep = raw_teacher_margin.abs() > min_teacher_gap
    keep_float = keep.float()
    pair_count = keep.sum()
    denominator = pair_count.clamp_min(1).float()
    per_pair_loss = F.huber_loss(
        student_margin.float(),
        teacher_margin,
        reduction="none",
        delta=huber_delta,
    )
    loss = (per_pair_loss * keep_float).sum() / denominator
    return PairwiseMarginResult(
        loss=loss,
        pair_count=pair_count.detach(),
        mean_teacher_abs_margin=(
            (teacher_margin.abs() * keep_float).sum() / denominator
        ).detach(),
        mean_student_abs_margin=(
            (student_margin.float().abs() * keep_float).sum() / denominator
        ).detach(),
    )
