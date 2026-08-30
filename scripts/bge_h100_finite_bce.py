#!/usr/bin/env python3
"""Finite plain BCE contract for the single-H100 BGE three-epoch run."""

from __future__ import annotations

import torch
import torch.nn.functional as F


LOSS_VARIANT = "bge_h100_finite_plain_bce_v1"
EXPECTED_TRAIN_ROWS = 347_840


def initialize_loss(*, train_frame, device, rank, world_size):
    if len(train_frame) != EXPECTED_TRAIN_ROWS:
        raise ValueError(
            "BGE H100 BCE hook received an unexpected training row count: "
            f"{len(train_frame)} != {EXPECTED_TRAIN_ROWS}"
        )
    if world_size != 1 or rank != 0:
        raise ValueError(
            f"BGE H100 campaign requires one rank, got rank={rank}, world_size={world_size}"
        )
    return None


def compute_loss(
    *,
    logits,
    targets,
    sample_weights,
    pair_indices,
    orientations,
    epoch,
    step,
):
    del pair_indices, orientations, epoch, step
    if not torch.isfinite(logits).all():
        raise FloatingPointError("non-finite BGE training logits")
    if not torch.isfinite(targets).all():
        raise FloatingPointError("non-finite BGE training targets")
    if not torch.isfinite(sample_weights).all():
        raise FloatingPointError("non-finite BGE sample weights")
    denominator = sample_weights.sum()
    if not torch.isfinite(denominator) or denominator <= 0:
        raise FloatingPointError("invalid BGE BCE denominator")
    per_example = F.binary_cross_entropy_with_logits(
        logits.float(), targets, reduction="none"
    )
    loss = (per_example * sample_weights).sum() / denominator
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite BGE BCE loss")
    return {"loss": loss, "bce": loss.detach()}
