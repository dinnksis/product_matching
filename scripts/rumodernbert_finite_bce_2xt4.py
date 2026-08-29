"""Finite unweighted BCE for the RuModernBERT two-T4 Kaggle campaign."""

from __future__ import annotations

import torch
import torch.nn.functional as F


LOSS_VARIANT = "rumodernbert_finite_plain_bce_2xt4_v1"
EXPECTED_TRAIN_ROWS = 347_840


def initialize_loss(*, train_frame, device, rank, world_size):
    if len(train_frame) != EXPECTED_TRAIN_ROWS:
        raise ValueError(
            f"unexpected RuModernBERT train rows: {len(train_frame)}"
        )
    if world_size != 2 or rank not in {0, 1}:
        raise ValueError(f"RuModernBERT Kaggle loss requires two ranks, got {rank}/{world_size}")
    return None


def compute_loss(
    *, logits, targets, sample_weights, pair_indices, orientations, epoch, step
):
    for name, tensor in (
        ("logits", logits),
        ("targets", targets),
        ("sample_weights", sample_weights),
    ):
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"non-finite RuModernBERT {name}")
    denominator = sample_weights.sum()
    if not torch.isfinite(denominator) or denominator <= 0:
        raise FloatingPointError("invalid RuModernBERT BCE denominator")
    per_example = F.binary_cross_entropy_with_logits(
        logits.float(), targets, reduction="none"
    )
    loss = (per_example * sample_weights).sum() / denominator
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite RuModernBERT BCE loss")
    return {"loss": loss, "bce": loss.detach()}
