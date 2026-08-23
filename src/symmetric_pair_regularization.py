from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SymmetricPairLoss:
    supervised: Any
    regularizer: Any
    mean_abs_logit_gap: Any
    mean_positive_probability: Any


def symmetric_pair_loss(
    *,
    directional_logits: Any,
    labels: Any,
) -> SymmetricPairLoss:
    """Return two-direction soft BCE and a non-negative logit-gap penalty."""
    import torch.nn.functional as F

    if directional_logits.ndim != 2 or directional_logits.shape[1] != 2:
        raise ValueError("directional_logits must have shape [pairs, 2]")
    if labels.ndim != 1 or len(labels) != len(directional_logits):
        raise ValueError("labels must contain one target per directional-logit pair")

    float_logits = directional_logits.float()
    expanded_labels = labels[:, None].expand_as(float_logits).float()
    supervised = F.binary_cross_entropy_with_logits(
        float_logits,
        expanded_labels,
    )
    logit_gap = float_logits[:, 0] - float_logits[:, 1]
    return SymmetricPairLoss(
        supervised=supervised,
        regularizer=logit_gap.square().mean(),
        mean_abs_logit_gap=logit_gap.abs().mean().detach(),
        mean_positive_probability=float_logits.sigmoid().mean(dim=1),
    )
