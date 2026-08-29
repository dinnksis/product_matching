#!/usr/bin/env python3
"""RuModernBERT 2xT4 adapter over the audited distributed AMP trainer.

The shared BGE preflight helper captures its 393-tensor default in a Python
function signature.  Updating the module constant alone therefore does not
change that comparison.  This adapter passes RuModernBERT's exact 138-tensor
expectation explicitly while keeping the generic full-state and moment-element
checks unchanged.
"""

from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path

from transformers import PreTrainedModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_bge_2ep_sft as distributed_amp


EXPECTED_PARAMETERS = 149_605_633
EXPECTED_PARAMETER_TENSORS = 138
EXPECTED_MICROBATCH = 24
EXPECTED_EVAL_BATCH = 96
EXPECTED_GRADIENT_ACCUMULATION = 4
EXPECTED_EFFECTIVE_BATCH = 192
PREFLIGHT_STATE_POLICY = "explicit_rumodernbert_optimizer_tensor_count_v1"
ACTIVATION_CHECKPOINTING_POLICY = "disabled_after_modernbert_autocast_checkpoint_error_v1"
_GENERIC_CLASSIFY_AMP_ATTEMPT = distributed_amp.classify_amp_optimizer_attempt
_GENERIC_VALIDATE_MEMORY_GEOMETRY = distributed_amp.validate_memory_geometry
_GENERIC_GRADIENT_CHECKPOINTING_ENABLE = PreTrainedModel.gradient_checkpointing_enable


def classify_amp_optimizer_attempt(**kwargs) -> str:
    """Call the shared invariant with RuModernBERT's exact tensor count."""

    if "expected_optimizer_state_parameters" in kwargs:
        raise RuntimeError("caller must not override the RuModernBERT state count")
    return _GENERIC_CLASSIFY_AMP_ATTEMPT(
        **kwargs,
        expected_optimizer_state_parameters=EXPECTED_PARAMETER_TENSORS,
    )


def validate_memory_geometry(config) -> None:
    """Validate the generic geometry with checkpointing explicitly disabled."""

    if config.get("gradient_checkpointing") is not False:
        raise distributed_amp.BgeTrainingContractError(
            "RuModernBERT activation checkpointing must remain disabled"
        )
    projected = deepcopy(config)
    projected["gradient_checkpointing"] = True
    _GENERIC_VALIDATE_MEMORY_GEOMETRY(projected)


def disable_gradient_checkpointing_enable(
    self, gradient_checkpointing_kwargs=None
) -> None:
    """Keep the generic disposable preflight on the exact no-checkpoint path."""

    if gradient_checkpointing_kwargs != {"use_reentrant": False}:
        raise RuntimeError("unexpected generic checkpointing request")
    return None


def configure_adapter(*, preflight_only: bool = False) -> None:
    """Bind the generic audited T4/GradScaler implementation to RuModernBERT."""

    distributed_amp.EXPECTED_PARAMETERS = EXPECTED_PARAMETERS
    distributed_amp.EXPECTED_TRAINABLE_PARAMETER_TENSORS = EXPECTED_PARAMETER_TENSORS
    distributed_amp.EXPECTED_MICROBATCH = EXPECTED_MICROBATCH
    distributed_amp.EXPECTED_EVAL_BATCH = EXPECTED_EVAL_BATCH
    distributed_amp.EXPECTED_GRADIENT_ACCUMULATION = EXPECTED_GRADIENT_ACCUMULATION
    distributed_amp.EXPECTED_EFFECTIVE_BATCH = EXPECTED_EFFECTIVE_BATCH
    distributed_amp.validate_memory_geometry = validate_memory_geometry
    if preflight_only:
        distributed_amp.classify_amp_optimizer_attempt = classify_amp_optimizer_attempt
        PreTrainedModel.gradient_checkpointing_enable = (
            disable_gradient_checkpointing_enable
        )


def annotate_preflight_report(argv: list[str]) -> None:
    """Bind the model-specific state policy to the rank-zero report."""

    if int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return
    try:
        report_index = argv.index("--preflight-report") + 1
        report_path = Path(argv[report_index])
    except (ValueError, IndexError) as error:
        raise RuntimeError("RuModernBERT preflight report path is missing") from error
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["optimizer_state_materialization_policy"] = PREFLIGHT_STATE_POLICY
    report["optimizer_state_expected_parameter_tensors"] = EXPECTED_PARAMETER_TENSORS
    report["gradient_checkpointing"] = False
    report["activation_checkpointing_policy"] = ACTIVATION_CHECKPOINTING_POLICY
    report["full_training_uses_synthetic_zero_gradients"] = False
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    preflight_only = "--memory-preflight-only" in sys.argv[1:]
    previous_classifier = distributed_amp.classify_amp_optimizer_attempt
    previous_validator = distributed_amp.validate_memory_geometry
    previous_checkpointing_enable = PreTrainedModel.gradient_checkpointing_enable
    configure_adapter(preflight_only=preflight_only)
    try:
        result = distributed_amp.main()
        if preflight_only:
            annotate_preflight_report(sys.argv[1:])
        return result
    finally:
        distributed_amp.classify_amp_optimizer_attempt = previous_classifier
        distributed_amp.validate_memory_geometry = previous_validator
        PreTrainedModel.gradient_checkpointing_enable = previous_checkpointing_enable


if __name__ == "__main__":
    raise SystemExit(main())
