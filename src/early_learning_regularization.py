"""Early-learning regularization for the binary product-pair classifier.

The original ELR formulation stores a two-class temporal target for every
training example.  The cross-encoder emits one logit, so its equivalent class
probability vector is ``[1 - sigmoid(logit), sigmoid(logit)]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BinaryElrLoss:
    total: Any
    supervised: Any
    regularizer: Any
    mean_agreement: Any


@dataclass(frozen=True)
class BinaryElrRegularization:
    regularizer: Any
    mean_agreement: Any


def make_binary_elr_targets(example_count: int, device: Any) -> Any:
    """Allocate the exact two-component ELR history used by the paper."""
    import torch

    if example_count <= 0:
        raise ValueError("ELR example_count must be positive")
    return torch.zeros((example_count, 2), dtype=torch.float32, device=device)


def binary_elr_regularization(
    *,
    positive_probabilities: Any,
    example_indices: Any,
    target_history: Any,
    beta: float,
    epsilon: float = 1e-4,
) -> BinaryElrRegularization:
    """Update temporal targets from one probability per training example."""
    import torch

    if not 0 <= beta < 1:
        raise ValueError("ELR beta must be in [0, 1)")
    if not 0 < epsilon < 0.5:
        raise ValueError("ELR epsilon must be in (0, 0.5)")
    if positive_probabilities.ndim != 1:
        raise ValueError("ELR expects one positive probability per example")
    if example_indices.shape != positive_probabilities.shape:
        raise ValueError("ELR example indices must align with probabilities")
    if target_history.ndim != 2 or target_history.shape[1] != 2:
        raise ValueError("ELR target history must have shape [examples, 2]")

    positive = positive_probabilities.float().clamp(epsilon, 1.0 - epsilon)
    probabilities = torch.stack((1.0 - positive, positive), dim=1)
    with torch.no_grad():
        previous = target_history.index_select(0, example_indices)
        updated = beta * previous + (1.0 - beta) * probabilities.detach()
        target_history.index_copy_(0, example_indices, updated)

    temporal_targets = target_history.index_select(0, example_indices)
    agreement = (temporal_targets * probabilities).sum(dim=1)
    agreement = agreement.clamp(min=0.0, max=1.0 - epsilon)
    return BinaryElrRegularization(
        regularizer=torch.log1p(-agreement).mean(),
        mean_agreement=agreement.mean(),
    )


def binary_elr_loss(
    *,
    logits: Any,
    labels: Any,
    example_indices: Any,
    target_history: Any,
    beta: float,
    regularization_strength: float,
    epsilon: float = 1e-4,
) -> BinaryElrLoss:
    """Update temporal targets and return BCE plus the ELR penalty.

    This is Equation (6) with the temporal ensemble from Equation (9) in
    Liu et al. (NeurIPS 2020), adapted from softmax CE to an equivalent binary
    BCE representation. Soft labels are retained in the supervised term.
    """
    import torch.nn.functional as F

    if regularization_strength < 0:
        raise ValueError("ELR regularization strength must be non-negative")
    if logits.ndim != 1 or labels.shape != logits.shape:
        raise ValueError("ELR expects one logit and one label per example")
    regularization = binary_elr_regularization(
        positive_probabilities=logits.float().sigmoid(),
        example_indices=example_indices,
        target_history=target_history,
        beta=beta,
        epsilon=epsilon,
    )
    supervised = F.binary_cross_entropy_with_logits(logits.float(), labels.float())
    total = supervised + regularization_strength * regularization.regularizer
    return BinaryElrLoss(
        total=total,
        supervised=supervised,
        regularizer=regularization.regularizer,
        mean_agreement=regularization.mean_agreement,
    )
