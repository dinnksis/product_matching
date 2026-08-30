#!/usr/bin/env python3
"""Single-H100 guards and worst-case memory preflight for BGE SFT."""

from __future__ import annotations

import argparse
import hashlib
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
from transformers import AutoModelForSequenceClassification
from transformers import __version__ as transformers_version


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_cross_encoder as shared_trainer


EXPECTED_PARAMETERS = 567_755_777
EXPECTED_PARAMETER_TENSORS = 393
EXPECTED_TRANSFORMERS_VERSION = "4.57.6"
EXPECTED_CHECKPOINT_FILES = {
    "config.json": "3f143138299caf72270c79814d1f0e3c38fc168e2d30f1e6cc4c70f6cdb481f5",
    "model.safetensors": "cdaf66bb271e6cc742267aa0aec0c890be1c898a93c469c137f5174ea9eeba72",
    "special_tokens_map.json": "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835",
    "tokenizer.json": "8bf8afbfd11306bd872018c53bfdf2e160a56f8edbcf49933324404791c148d3",
    "tokenizer_config.json": "b87c8703482b0300d3da30e201519aa641f6a450f5eb5bf1e624afbf70c74d80",
}
EXPECTED_CONFIG = {
    "model_backend": "sequence_classification",
    "model_load_kwargs": {"local_files_only": True},
    "trust_remote_code": False,
    "epochs": 3,
    "batch_size": 64,
    "eval_batch_size": 192,
    "gradient_accumulation": 3,
    "learning_rate": 2e-5,
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


class BgeH100TrainingError(RuntimeError):
    """Raised when the frozen one-H100 contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint_files(model_path: Path) -> None:
    if not model_path.is_dir():
        raise BgeH100TrainingError(f"BGE model directory does not exist: {model_path}")
    for name, expected_hash in EXPECTED_CHECKPOINT_FILES.items():
        path = model_path / name
        if path.is_symlink() or not path.is_file():
            raise BgeH100TrainingError(
                f"BGE checkpoint file must be regular and non-symlink: {path}"
            )
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise BgeH100TrainingError(
                f"BGE checkpoint hash differs for {name}: {actual_hash} != {expected_hash}"
            )


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
        raise BgeH100TrainingError(
            f"Could not read training config {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise BgeH100TrainingError("Training config must be a JSON object")
    if set(payload) != set(EXPECTED_CONFIG) | {"model"}:
        raise BgeH100TrainingError(
            "BGE H100 config keys differ: "
            f"missing={sorted((set(EXPECTED_CONFIG) | {'model'}) - set(payload))}, "
            f"extra={sorted(set(payload) - (set(EXPECTED_CONFIG) | {'model'}))}"
        )
    for key, expected in EXPECTED_CONFIG.items():
        if payload.get(key) != expected:
            raise BgeH100TrainingError(
                f"Frozen BGE H100 config differs at {key}: "
                f"{payload.get(key)!r} != {expected!r}"
            )
    model_path = Path(str(payload["model"])).expanduser().resolve()
    validate_checkpoint_files(model_path)
    return payload


def require_single_h100() -> dict[str, Any]:
    if transformers_version != EXPECTED_TRANSFORMERS_VERSION:
        raise BgeH100TrainingError(
            f"Expected transformers {EXPECTED_TRANSFORMERS_VERSION}, got {transformers_version}"
        )
    if not torch.cuda.is_available():
        raise BgeH100TrainingError("CUDA is required")
    if torch.cuda.device_count() != 1:
        raise BgeH100TrainingError(
            f"Exactly one visible GPU is required, got {torch.cuda.device_count()}"
        )
    gpu_name = torch.cuda.get_device_name(0)
    if "H100" not in gpu_name.upper():
        raise BgeH100TrainingError(f"Expected an H100, got {gpu_name!r}")
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    if total_memory < 75 * 2**30:
        raise BgeH100TrainingError(
            f"Expected an 80GB-class H100, got {total_memory / 2**30:.2f} GiB"
        )
    major, minor = torch.cuda.get_device_capability(0)
    if major < 9:
        raise BgeH100TrainingError(
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
        raise BgeH100TrainingError("AdamW foreach must remain disabled")
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
        raise BgeH100TrainingError("Non-finite gradients must fail closed")
    requested_foreach = kwargs.get("foreach")
    if requested_foreach not in (None, False):
        raise BgeH100TrainingError("Gradient clipping foreach must remain disabled")
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
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
        raise BgeH100TrainingError(
            f"Unexpected BGE parameter contract: {parameter_count}/{len(parameters)}"
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
    input_ids[:, 0] = 0
    input_ids[:, -1] = 2
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
        raise FloatingPointError("BGE preflight produced non-finite forward values")
    loss.backward()
    gradient_norm = strict_clip_grad_norm(parameters, float(config["max_grad_norm"]))
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError("BGE preflight produced non-finite gradients")
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
        raise BgeH100TrainingError(
            f"AdamW state tensor count differs: {state_parameters} != {EXPECTED_PARAMETER_TENSORS}"
        )
    if state_moment_elements != 2 * EXPECTED_PARAMETERS:
        raise BgeH100TrainingError(
            f"AdamW moment elements differ: {state_moment_elements} != {2 * EXPECTED_PARAMETERS}"
        )
    eval_batch_size = int(config["eval_batch_size"])
    eval_ids = input_ids[:1].expand(eval_batch_size, -1).contiguous()
    eval_mask = attention_mask[:1].expand(eval_batch_size, -1).contiguous()
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        eval_logits = model(input_ids=eval_ids, attention_mask=eval_mask).logits[:, 0]
    if tuple(eval_logits.shape) != (eval_batch_size,) or not torch.isfinite(eval_logits).all():
        raise FloatingPointError("BGE preflight produced invalid evaluation logits")
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
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        shared_trainer.main()
    finally:
        shared_trainer.AdamW = previous_adamw
        shared_trainer.torch.nn.utils.clip_grad_norm_ = previous_clip
        shared_trainer.preferred_cuda_dtype = previous_dtype
    return 0


def parse_preflight_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BGE one-H100 memory preflight")
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
