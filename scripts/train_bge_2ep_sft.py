#!/usr/bin/env python3
"""Memory-safe entry point for full BGE-2ep supervised fine-tuning.

The shared cross-encoder trainer is intentionally not edited: several frozen
MiniLM campaigns bind its exact source hash.  This wrapper changes only two
runtime details required by the 568M-parameter BGE model on 16 GiB T4 GPUs:

* AdamW foreach kernels are disabled to avoid a parameter-sized temporary;
* gradient clipping keeps ``foreach=False`` while leaving transient FP16
  overflow to ``GradScaler``'s skip-and-backoff protocol.

It also exposes a real two-rank one-step preflight.  The preflight loads the
same model, enables gradient checkpointing, reproduces the full 12-microbatch
accumulation group at batch=8 and length=384, creates Adam moments with a real
optimizer step, and writes a small auditable report.  Initial loss-scale
overflow is retried only under a bounded, fail-closed contract.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_cross_encoder as shared_trainer


EXPECTED_PARAMETERS = 567_755_777
EXPECTED_TRAINABLE_PARAMETER_TENSORS = 393
EXPECTED_WORLD_SIZE = 2
EXPECTED_GPU_MARKER = "T4"
EXPECTED_MICROBATCH = 8
EXPECTED_MAX_LENGTH = 384
EXPECTED_GRADIENT_ACCUMULATION = 12
EXPECTED_EFFECTIVE_BATCH = 192
EXPECTED_EVAL_BATCH = 32
MAX_PREFLIGHT_AMP_ATTEMPTS = 17
AMP_NONFINITE_POLICY = "finite_loss_guard_grad_scaler_bounded_backoff_v1"

_ORIGINAL_ADAMW = shared_trainer.AdamW
_ORIGINAL_CLIP_GRAD_NORM = torch.nn.utils.clip_grad_norm_


class BgeTrainingContractError(ValueError):
    """Raised before training when the frozen T4 geometry was changed."""


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BgeTrainingContractError("BGE config must contain a JSON object")
    return payload


def validate_memory_geometry(config: dict[str, Any]) -> None:
    expected = {
        "batch_size": EXPECTED_MICROBATCH,
        "eval_batch_size": EXPECTED_EVAL_BATCH,
        "gradient_accumulation": EXPECTED_GRADIENT_ACCUMULATION,
        "max_length": EXPECTED_MAX_LENGTH,
        "gradient_checkpointing": True,
        "attention_implementation": "sdpa",
    }
    mismatches = {
        key: {"actual": config.get(key), "expected": value}
        for key, value in expected.items()
        if config.get(key) != value
    }
    effective_batch = (
        int(config.get("batch_size", 0))
        * EXPECTED_WORLD_SIZE
        * int(config.get("gradient_accumulation", 0))
    )
    if effective_batch != EXPECTED_EFFECTIVE_BATCH:
        mismatches["effective_batch"] = {
            "actual": effective_batch,
            "expected": EXPECTED_EFFECTIVE_BATCH,
        }
    if mismatches:
        raise BgeTrainingContractError(
            "BGE 2xT4 memory geometry differs: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def memory_efficient_adamw(
    params: Iterable[torch.nn.Parameter] | Any,
    *args: Any,
    **kwargs: Any,
) -> AdamW:
    requested = kwargs.get("foreach")
    if requested not in (None, False):
        raise BgeTrainingContractError("BGE AdamW foreach must remain disabled")
    kwargs["foreach"] = False
    return _ORIGINAL_ADAMW(params, *args, **kwargs)


def amp_compatible_clip_grad_norm(
    parameters: Iterable[torch.Tensor],
    max_norm: float,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    requested = kwargs.get("error_if_nonfinite")
    if requested not in (None, False):
        raise BgeTrainingContractError(
            "BGE FP16 clipping must leave transient overflow to GradScaler"
        )
    # GradScaler.unscale_ records found_inf before clipping.  Raising here would
    # prevent scaler.step/update from skipping the unsafe optimizer step and
    # reducing the scale, which is exactly what killed the v3 preflight.
    kwargs["error_if_nonfinite"] = False
    requested_foreach = kwargs.get("foreach")
    if requested_foreach not in (None, False):
        raise BgeTrainingContractError(
            "BGE gradient clipping foreach must remain disabled"
        )
    kwargs["foreach"] = False
    return _ORIGINAL_CLIP_GRAD_NORM(parameters, max_norm, *args, **kwargs)


def classify_amp_optimizer_attempt(
    *,
    gradients_finite: bool,
    scale_before: float,
    scale_after: float,
    optimizer_state_parameters: int,
    expected_optimizer_state_parameters: int = EXPECTED_TRAINABLE_PARAMETER_TENSORS,
) -> str:
    """Validate one public-API GradScaler transition without private state."""
    if (
        not np.isfinite(scale_before)
        or not np.isfinite(scale_after)
        or scale_before <= 0
        or scale_after <= 0
    ):
        raise FloatingPointError("BGE AMP attempt reported an invalid loss scale")
    if gradients_finite:
        if scale_after < scale_before:
            raise FloatingPointError(
                "BGE GradScaler reduced scale despite finite gradients"
            )
        if optimizer_state_parameters != expected_optimizer_state_parameters:
            raise RuntimeError(
                "BGE finite AMP attempt did not materialize every AdamW state"
            )
        return "optimizer_step"
    if scale_after >= scale_before:
        raise FloatingPointError(
            "BGE GradScaler did not reduce scale after gradient overflow"
        )
    if optimizer_state_parameters != 0:
        raise RuntimeError(
            "BGE skipped AMP attempt unexpectedly mutated AdamW state"
        )
    return "skipped_gradient_overflow"


def install_full_training_guards() -> tuple[Any, Any]:
    """Patch the worker and return the exact values that must be restored."""
    previous_adamw = shared_trainer.AdamW
    previous_clip_grad_norm = torch.nn.utils.clip_grad_norm_
    shared_trainer.AdamW = memory_efficient_adamw
    # train_cross_encoder dereferences the function through torch.nn.utils.
    shared_trainer.torch.nn.utils.clip_grad_norm_ = amp_compatible_clip_grad_norm
    return previous_adamw, previous_clip_grad_norm


def restore_full_training_guards(previous_adamw: Any, previous_clip_grad_norm: Any) -> None:
    """Restore shared process state after both successful and failed training."""
    shared_trainer.AdamW = previous_adamw
    shared_trainer.torch.nn.utils.clip_grad_norm_ = previous_clip_grad_norm


def _preflight_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BGE 2xT4 one-step memory preflight")
    parser.add_argument("--memory-preflight-only", action="store_true", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    return parser


def _optimizer_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    decay = [
        parameter
        for name, parameter in named
        if parameter.ndim >= 2 and "layer_norm" not in name.lower()
    ]
    no_decay = [
        parameter
        for name, parameter in named
        if parameter.ndim < 2 or "layer_norm" in name.lower()
    ]
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def run_memory_preflight(argv: list[str] | None = None) -> int:
    args = _preflight_parser().parse_args(argv)
    config = load_config(args.config)
    validate_memory_geometry(config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the BGE memory preflight")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != EXPECTED_WORLD_SIZE:
        raise RuntimeError(
            f"BGE memory preflight requires exactly {EXPECTED_WORLD_SIZE} ranks, got {world_size}"
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank()
    try:
        gpu_name = torch.cuda.get_device_name(device)
        if EXPECTED_GPU_MARKER not in gpu_name.upper():
            raise RuntimeError(f"BGE memory preflight requires T4, got {gpu_name!r}")
        seed = int(config.get("seed", 42)) + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        model = AutoModelForSequenceClassification.from_pretrained(
            str(config["model"]),
            num_labels=1,
            attn_implementation=str(config["attention_implementation"]),
            trust_remote_code=False,
        ).to(device)
        model.config.id2label = {0: "MATCH_SCORE"}
        model.config.label2id = {"MATCH_SCORE": 0}
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != EXPECTED_PARAMETERS:
            raise RuntimeError(
                f"Unexpected BGE parameter count: {parameter_count} != {EXPECTED_PARAMETERS}"
            )
        training_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
        optimizer = memory_efficient_adamw(
            _optimizer_groups(training_model, float(config["weight_decay"])),
            lr=float(config["learning_rate"]),
        )

        input_ids = torch.full(
            (EXPECTED_MICROBATCH, EXPECTED_MAX_LENGTH),
            10,
            dtype=torch.long,
            device=device,
        )
        input_ids[:, 0] = 0
        input_ids[:, -1] = 2
        attention_mask = torch.ones_like(input_ids)
        targets = torch.tensor(
            [0.0, 1.0] * (EXPECTED_MICROBATCH // 2),
            dtype=torch.float32,
            device=device,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        training_model.train()
        torch.cuda.reset_peak_memory_stats(device)
        trainable_parameters = [
            parameter
            for parameter in training_model.parameters()
            if parameter.requires_grad
        ]
        attempt_history: list[dict[str, Any]] = []
        successful_loss: float | None = None
        successful_grad_norm: torch.Tensor | None = None
        for attempt in range(1, MAX_PREFLIGHT_AMP_ATTEMPTS + 1):
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = torch.zeros((), dtype=torch.float32, device=device)
            for microstep in range(EXPECTED_GRADIENT_ACCUMULATION):
                sync_context = (
                    training_model.no_sync()
                    if microstep + 1 < EXPECTED_GRADIENT_ACCUMULATION
                    else nullcontext()
                )
                with sync_context:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = training_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        ).logits[:, 0].float()
                        if not torch.isfinite(logits).all():
                            raise FloatingPointError(
                                "BGE memory preflight produced non-finite logits"
                            )
                        raw_loss = F.binary_cross_entropy_with_logits(
                            logits, targets
                        )
                        loss = raw_loss / EXPECTED_GRADIENT_ACCUMULATION
                    if not torch.isfinite(raw_loss) or not torch.isfinite(loss):
                        raise FloatingPointError(
                            "BGE memory preflight produced non-finite loss"
                        )
                    scaler.scale(loss).backward()
                accumulated_loss += loss.detach()

            scaler.unscale_(optimizer)
            grad_norm = amp_compatible_clip_grad_norm(
                trainable_parameters,
                float(config["max_grad_norm"]),
            )
            gradients_finite = bool(torch.isfinite(grad_norm).item())
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            optimizer_state_parameters = len(optimizer.state)
            outcome = classify_amp_optimizer_attempt(
                gradients_finite=gradients_finite,
                scale_before=scale_before,
                scale_after=scale_after,
                optimizer_state_parameters=optimizer_state_parameters,
            )
            attempt_history.append(
                {
                    "attempt": attempt,
                    "accumulated_microbatches": EXPECTED_GRADIENT_ACCUMULATION,
                    "loss_divisor_per_microbatch": EXPECTED_GRADIENT_ACCUMULATION,
                    "accumulated_loss": float(accumulated_loss),
                    "gradient_norm": (
                        float(grad_norm.detach()) if gradients_finite else None
                    ),
                    "gradients_finite": gradients_finite,
                    "scale_before": scale_before,
                    "scale_after": scale_after,
                    "optimizer_state_parameters": optimizer_state_parameters,
                    "outcome": outcome,
                }
            )
            if outcome == "optimizer_step":
                successful_loss = float(accumulated_loss)
                successful_grad_norm = grad_norm.detach()
                break
        else:
            raise FloatingPointError(
                "BGE memory preflight exhausted bounded GradScaler backoff"
            )
        if successful_loss is None or successful_grad_norm is None:
            raise RuntimeError("BGE memory preflight recorded no optimizer step")
        optimizer_state_parameters = 0
        optimizer_state_tensor_elements = 0
        for state in optimizer.state.values():
            if "exp_avg" not in state or "exp_avg_sq" not in state:
                continue
            optimizer_state_parameters += 1
            for name in ("exp_avg", "exp_avg_sq"):
                tensor = state[name]
                flat = tensor.reshape(-1)
                stride = max(1, flat.numel() // 1024)
                if not torch.isfinite(flat[::stride]).all():
                    raise FloatingPointError(
                        f"BGE memory preflight produced non-finite AdamW {name}"
                    )
                optimizer_state_tensor_elements += tensor.numel()
        if optimizer_state_parameters != EXPECTED_TRAINABLE_PARAMETER_TENSORS:
            raise RuntimeError(
                "BGE memory preflight materialized AdamW state for an unexpected "
                f"number of tensors: {optimizer_state_parameters} != "
                f"{EXPECTED_TRAINABLE_PARAMETER_TENSORS}"
            )
        if optimizer_state_tensor_elements != 2 * EXPECTED_PARAMETERS:
            raise RuntimeError(
                "BGE memory preflight AdamW moment elements differ from two full "
                "trainable model copies"
            )

        # Full validation runs after training while Adam moments are still
        # resident.  Exercise its exact per-rank batch now so preflight covers
        # both the training and evaluation peaks rather than only the update.
        optimizer.zero_grad(set_to_none=True)
        eval_input_ids = torch.full(
            (EXPECTED_EVAL_BATCH, EXPECTED_MAX_LENGTH),
            10,
            dtype=torch.long,
            device=device,
        )
        eval_input_ids[:, 0] = 0
        eval_input_ids[:, -1] = 2
        eval_attention_mask = torch.ones_like(eval_input_ids)
        training_model.eval()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            eval_logits = training_model(
                input_ids=eval_input_ids,
                attention_mask=eval_attention_mask,
            ).logits[:, 0].float()
        if (
            tuple(eval_logits.shape) != (EXPECTED_EVAL_BATCH,)
            or not torch.isfinite(eval_logits).all()
        ):
            raise FloatingPointError(
                "BGE memory preflight produced invalid evaluation logits"
            )
        torch.cuda.synchronize(device)
        local_report = {
            "rank": rank,
            "gpu": gpu_name,
            "loss": successful_loss,
            "gradient_norm": float(successful_grad_norm),
            "amp_attempts": attempt_history,
            "amp_overflow_skips": len(attempt_history) - 1,
            "amp_final_scale": attempt_history[-1]["scale_after"],
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        }
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, local_report)
        if rank == 0:
            report = {
                "schema_version": 1,
                "status": "passed",
                "world_size": world_size,
                "model": str(config["model"]),
                "parameters": parameter_count,
                "microbatch_per_gpu": EXPECTED_MICROBATCH,
                "max_length": EXPECTED_MAX_LENGTH,
                "gradient_accumulation": EXPECTED_GRADIENT_ACCUMULATION,
                "accumulated_microbatches": EXPECTED_GRADIENT_ACCUMULATION,
                "loss_divisor_per_microbatch": EXPECTED_GRADIENT_ACCUMULATION,
                "ddp_no_sync_microbatches": EXPECTED_GRADIENT_ACCUMULATION - 1,
                "ddp_sync_microbatches": 1,
                "effective_batch": EXPECTED_EFFECTIVE_BATCH,
                "eval_batch_per_gpu": EXPECTED_EVAL_BATCH,
                "eval_probe_after_optimizer_state": True,
                "gradient_checkpointing": True,
                "attention_implementation": "sdpa",
                "amp_dtype": "float16",
                "adamw_foreach": False,
                "gradient_clip_foreach": False,
                "nonfinite_gradient_policy": AMP_NONFINITE_POLICY,
                "amp_max_attempts": MAX_PREFLIGHT_AMP_ATTEMPTS,
                "optimizer_state": "adamw_exp_avg_and_exp_avg_sq_materialized",
                "optimizer_state_parameters_per_rank": optimizer_state_parameters,
                "optimizer_state_tensor_elements_per_rank": (
                    optimizer_state_tensor_elements
                ),
                "ranks": gathered,
            }
            args.preflight_report.parent.mkdir(parents=True, exist_ok=True)
            args.preflight_report.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(report, ensure_ascii=False), flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


def run_full_training() -> int:
    # Validate the same exact geometry before shared_trainer parses the config.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, required=True)
    known, _ = pre_parser.parse_known_args()
    validate_memory_geometry(load_config(known.config))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != EXPECTED_WORLD_SIZE:
        raise BgeTrainingContractError(
            f"Full BGE training requires exactly {EXPECTED_WORLD_SIZE} ranks, got {world_size}"
        )
    previous_adamw, previous_clip_grad_norm = install_full_training_guards()
    try:
        shared_trainer.main()
    finally:
        restore_full_training_guards(previous_adamw, previous_clip_grad_norm)
    return 0


def main() -> int:
    if "--memory-preflight-only" in sys.argv[1:]:
        return run_memory_preflight()
    return run_full_training()


if __name__ == "__main__":
    raise SystemExit(main())
