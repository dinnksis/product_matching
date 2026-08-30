#!/usr/bin/env python3
"""Finite plain BCE for the all-human single-H100 final export."""

from __future__ import annotations

import torch
import torch.nn.functional as F


LOSS_VARIANT = "bge_h100_fulltrain_finite_plain_bce_v1"
EXPECTED_TRAIN_ROWS = 365_654


def initialize_loss(*, train_frame, device, rank, world_size):
    if len(train_frame) != EXPECTED_TRAIN_ROWS:
        raise ValueError(
            "BGE H100 full-train BCE received an unexpected row count: "
            f"{len(train_frame)} != {EXPECTED_TRAIN_ROWS}"
        )
    if world_size != 1 or rank != 0:
        raise ValueError(
            f"BGE H100 full-train requires one rank, got rank={rank}, world_size={world_size}"
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
        raise FloatingPointError("non-finite BGE full-train logits")
    if not torch.isfinite(targets).all():
        raise FloatingPointError("non-finite BGE full-train targets")
    if not torch.isfinite(sample_weights).all():
        raise FloatingPointError("non-finite BGE full-train sample weights")
    denominator = sample_weights.sum()
    if not torch.isfinite(denominator) or denominator <= 0:
        raise FloatingPointError("invalid BGE full-train BCE denominator")
    per_example = F.binary_cross_entropy_with_logits(
        logits.float(), targets, reduction="none"
    )
    loss = (per_example * sample_weights).sum() / denominator
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite BGE full-train BCE loss")
    return {"loss": loss, "bce": loss.detach()}
