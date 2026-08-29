#!/usr/bin/env python3
"""H100 guards and memory preflight for RuModernBERT human SFT."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, __version__ as transformers_version


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_cross_encoder as shared_trainer


EXPECTED_PARAMETERS = 149_605_633
EXPECTED_PARAMETER_TENSORS = 138
EXPECTED_MODEL_DIRNAME = "pretrain_rumodernbert_3ep"
EXPECTED_TRANSFORMERS_VERSION = "4.57.6"
ALLOWED_LEARNING_RATES = {4e-5, 8e-5, 1.6e-4}
EXPECTED_CONFIG = {
    "model_backend": "sequence_classification",
    "trust_remote_code": False,
    "batch_size": 192,
    "eval_batch_size": 512,
    "gradient_accumulation": 1,
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
    "max_length": 384,
    "attention_implementation": "sdpa",
    "sampling": "none",
    "train_subset": "all",
    "loss_weighting": "none",
    "lexical_hard_negative_strength": 0.0,
    "bucket_size_multiplier": 50,
    "dataloader_workers": 16,
    "prefetch_factor": 4,
    "tokenization_batch_size": 1024,
    "tokenization_log_every": 50,
    "gradient_checkpointing": False,
    "symmetric_validation": True,
    "label_smoothing": 0.0,
    "max_grad_norm": 0.5,
    "log_every": 50,
    "seed": 42,
}


class RuModernBertTrainingError(RuntimeError):
    """Raised when the frozen single-H100 contract is violated."""


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_training_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuModernBertTrainingError(f"Could not read training config {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuModernBertTrainingError("Training config must be a JSON object")
    unknown = set(payload) - shared_trainer.CONFIG_KEYS
    if unknown:
        raise RuModernBertTrainingError(f"Unknown training config keys: {sorted(unknown)}")
    missing = (set(EXPECTED_CONFIG) | {"model", "epochs", "learning_rate"}) - set(payload)
    if missing:
        raise RuModernBertTrainingError(f"Training config is missing keys: {sorted(missing)}")
    for key, expected in EXPECTED_CONFIG.items():
        if payload.get(key) != expected:
            raise RuModernBertTrainingError(
                f"Frozen RuModernBERT config differs at {key}: "
                f"{payload.get(key)!r} != {expected!r}"
            )
    model_path = Path(str(payload["model"])).expanduser().resolve()
    if model_path.name != EXPECTED_MODEL_DIRNAME or not model_path.is_dir():
        raise RuModernBertTrainingError(
            f"RuModernBERT model path must end in {EXPECTED_MODEL_DIRNAME!r}: {model_path}"
        )
    epochs = payload.get("epochs")
    if type(epochs) is not int or epochs not in {1, 2, 3}:
        raise RuModernBertTrainingError(f"RuModernBERT epochs must be 1, 2 or 3, got {epochs!r}")
    learning_rate = payload.get("learning_rate")
    if not isinstance(learning_rate, (int, float)) or float(learning_rate) not in ALLOWED_LEARNING_RATES:
        raise RuModernBertTrainingError(
            f"RuModernBERT learning rate is outside the frozen line: {learning_rate!r}"
        )
    if payload.get("model_load_kwargs", {}) != {}:
        raise RuModernBertTrainingError("RuModernBERT model_load_kwargs must remain empty")
    return payload


def require_single_h100() -> dict[str, Any]:
    if transformers_version != EXPECTED_TRANSFORMERS_VERSION:
        raise RuModernBertTrainingError(
            "RuModernBERT campaign requires transformers "
            f"{EXPECTED_TRANSFORMERS_VERSION}, got {transformers_version}"
        )
    if not torch.cuda.is_available():
        raise RuModernBertTrainingError("CUDA is required")
    if torch.cuda.device_count() != 1:
        raise RuModernBertTrainingError(
            f"Exactly one visible GPU is required, got {torch.cuda.device_count()}"
        )
    gpu_name = torch.cuda.get_device_name(0)
    if "H100" not in gpu_name.upper():
        raise RuModernBertTrainingError(f"Expected an H100, got {gpu_name!r}")
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    if total_memory < 75 * 2**30:
        raise RuModernBertTrainingError(
            f"Expected an 80GB-class H100, got {total_memory / 2**30:.2f} GiB"
        )
    major, minor = torch.cuda.get_device_capability(0)
    if major < 9:
        raise RuModernBertTrainingError(
            f"Expected Hopper compute capability, got {major}.{minor}"
        )
    return {
        "name": gpu_name,
        "total_memory_gib": total_memory / 2**30,
        "compute_capability": f"{major}.{minor}",
    }


def h100_bfloat16_dtype() -> torch.dtype:
    require_single_h100()
    return torch.bfloat16


def adamw_foreach_disabled(
    params: Iterable[torch.nn.Parameter] | Any,
    *args: Any,
    **kwargs: Any,
) -> AdamW:
    requested = kwargs.get("foreach")
    if requested not in (None, False):
        raise RuModernBertTrainingError("AdamW foreach must remain disabled")
    kwargs["foreach"] = False
    return AdamW(params, *args, **kwargs)


_ORIGINAL_CLIP_GRAD_NORM = torch.nn.utils.clip_grad_norm_


def strict_clip_grad_norm(
    parameters: Iterable[torch.Tensor],
    max_norm: float,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    requested_nonfinite = kwargs.get("error_if_nonfinite")
    if requested_nonfinite not in (None, True):
        raise RuModernBertTrainingError("Non-finite gradients must fail closed")
    requested_foreach = kwargs.get("foreach")
    if requested_foreach not in (None, False):
        raise RuModernBertTrainingError("Gradient clipping foreach must remain disabled")
    kwargs["error_if_nonfinite"] = True
    kwargs["foreach"] = False
    return _ORIGINAL_CLIP_GRAD_NORM(parameters, max_norm, *args, **kwargs)


def optimizer_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2 and "layer_norm" not in name.lower():
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def run_preflight(config_path: Path, report_path: Path) -> int:
    config = load_training_config(config_path)
    gpu = require_single_h100()
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda", 0)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(config["model"]),
        num_labels=1,
        attn_implementation=str(config["attention_implementation"]),
        trust_remote_code=False,
        local_files_only=True,
    ).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameter_count = sum(parameter.numel() for parameter in parameters)
    if parameter_count != EXPECTED_PARAMETERS or len(parameters) != EXPECTED_PARAMETER_TENSORS:
        raise RuModernBertTrainingError(
            "Unexpected RuModernBERT parameter contract: "
            f"{parameter_count}/{len(parameters)}"
        )
    optimizer = adamw_foreach_disabled(
        optimizer_groups(model, float(config["weight_decay"])),
        lr=float(config["learning_rate"]),
    )
    batch_size = int(config["batch_size"])
    max_length = int(config["max_length"])
    input_ids = torch.full(
        (batch_size, max_length), 10, dtype=torch.long, device=device
    )
    input_ids[:, 0] = 50_281
    input_ids[:, -1] = 50_282
    attention_mask = torch.ones_like(input_ids)
    targets = torch.tensor(
        [0.0, 1.0] * (batch_size // 2), dtype=torch.float32, device=device
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, 0]
        loss = F.binary_cross_entropy_with_logits(logits.float(), targets)
    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
        raise FloatingPointError("RuModernBERT preflight produced non-finite forward values")
    loss.backward()
    gradient_norm = strict_clip_grad_norm(parameters, float(config["max_grad_norm"]))
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError("RuModernBERT preflight produced non-finite gradients")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    state_parameters = 0
    state_moment_elements = 0
    for state in optimizer.state.values():
        if "exp_avg" not in state or "exp_avg_sq" not in state:
            continue
        state_parameters += 1
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = state[name]
            flat = tensor.reshape(-1)
            stride = max(1, flat.numel() // 1024)
            if not torch.isfinite(flat[::stride]).all():
                raise FloatingPointError(f"Non-finite AdamW state {name}")
            state_moment_elements += tensor.numel()
    if state_parameters != EXPECTED_PARAMETER_TENSORS:
        raise RuModernBertTrainingError(
            f"AdamW state tensor count differs: {state_parameters} != {EXPECTED_PARAMETER_TENSORS}"
        )
    if state_moment_elements != 2 * EXPECTED_PARAMETERS:
        raise RuModernBertTrainingError(
            f"AdamW moment elements differ: {state_moment_elements} != {2 * EXPECTED_PARAMETERS}"
        )
    eval_batch_size = int(config["eval_batch_size"])
    eval_ids = input_ids[:1].expand(eval_batch_size, -1).contiguous()
    eval_mask = attention_mask[:1].expand(eval_batch_size, -1).contiguous()
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        eval_logits = model(input_ids=eval_ids, attention_mask=eval_mask).logits[:, 0]
    if tuple(eval_logits.shape) != (eval_batch_size,) or not torch.isfinite(eval_logits).all():
        raise FloatingPointError("RuModernBERT preflight produced invalid evaluation logits")
    torch.cuda.synchronize(device)
    payload = {
        "schema_version": 1,
        "status": "passed",
        "gpu": gpu,
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "amp_dtype": "bfloat16",
        "model": str(Path(str(config["model"])).resolve()),
        "parameters": parameter_count,
        "parameter_tensors": len(parameters),
        "microbatch": batch_size,
        "gradient_accumulation": int(config["gradient_accumulation"]),
        "effective_batch": batch_size * int(config["gradient_accumulation"]),
        "eval_batch": eval_batch_size,
        "max_length": max_length,
        "loss": float(loss.detach()),
        "gradient_norm": float(gradient_norm.detach()),
        "optimizer_state_parameters": state_parameters,
        "optimizer_state_moment_elements": state_moment_elements,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }
    atomic_write_json(report_path, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


def run_training(config_path: Path) -> int:
    load_training_config(config_path)
    require_single_h100()
    previous_adamw = shared_trainer.AdamW
    previous_clip = shared_trainer.torch.nn.utils.clip_grad_norm_
    previous_dtype = shared_trainer.preferred_cuda_dtype
    shared_trainer.AdamW = adamw_foreach_disabled
    shared_trainer.torch.nn.utils.clip_grad_norm_ = strict_clip_grad_norm
    shared_trainer.preferred_cuda_dtype = h100_bfloat16_dtype
    try:
        shared_trainer.main()
    finally:
        shared_trainer.AdamW = previous_adamw
        shared_trainer.torch.nn.utils.clip_grad_norm_ = previous_clip
        shared_trainer.preferred_cuda_dtype = previous_dtype
    return 0


def parse_preflight_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RuModernBERT one-H100 memory preflight")
    parser.add_argument("--preflight-only", action="store_true", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--preflight-only" in values:
        args = parse_preflight_args(values)
        return run_preflight(args.config.resolve(), args.preflight_report.resolve())
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, required=True)
    known, _ = config_parser.parse_known_args(values)
    return run_training(known.config.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
