from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import torch
import torch.nn.functional as F


_METRIC_NAME = re.compile(r"[a-zA-Z][a-zA-Z0-9_.-]*")


def _default_compute_loss(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: torch.Tensor,
    **_: Any,
) -> dict[str, torch.Tensor]:
    per_example = F.binary_cross_entropy_with_logits(
        logits.float(), targets, reduction="none"
    )
    loss = (per_example * sample_weights).sum() / sample_weights.sum()
    return {"loss": loss, "bce": loss.detach()}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LoadedLossHook:
    """A small, explicit extension point for controlled loss ablations."""

    name: str
    path: Path | None
    sha256: str | None
    module: ModuleType | None

    @property
    def metadata(self) -> dict[str, str | None]:
        return {
            "name": self.name,
            "path": str(self.path) if self.path is not None else None,
            "sha256": self.sha256,
        }

    def initialize(self, **context: Any) -> None:
        if self.module is None:
            return
        initialize = getattr(self.module, "initialize_loss", None)
        if initialize is not None:
            initialize(**context)

    def compute(self, **context: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        compute = (
            _default_compute_loss
            if self.module is None
            else getattr(self.module, "compute_loss")
        )
        result = compute(**context)
        if isinstance(result, torch.Tensor):
            loss = result
            raw_metrics: Mapping[str, Any] = {}
        elif isinstance(result, Mapping):
            if "loss" not in result:
                raise ValueError("Loss hook mapping must contain a 'loss' tensor")
            loss = result["loss"]
            raw_metrics = {key: value for key, value in result.items() if key != "loss"}
        else:
            raise TypeError("Loss hook must return a scalar tensor or a mapping with 'loss'")
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise ValueError("Loss hook 'loss' must be a scalar torch.Tensor")
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Loss hook returned a non-finite loss")

        metrics: dict[str, torch.Tensor] = {}
        for name, value in raw_metrics.items():
            if not isinstance(name, str) or _METRIC_NAME.fullmatch(name) is None:
                raise ValueError(f"Invalid loss metric name: {name!r}")
            metric = value if isinstance(value, torch.Tensor) else loss.new_tensor(value)
            if metric.numel() != 1:
                raise ValueError(f"Loss metric {name!r} must contain one value")
            metrics[name] = metric.reshape(()).detach()
        return loss, metrics


def load_loss_hook(path: Path | None) -> LoadedLossHook:
    if path is None:
        return LoadedLossHook(
            name="weighted_bce",
            path=None,
            sha256=None,
            module=None,
        )
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Loss hook does not exist: {resolved}")
    source_hash = _file_sha256(resolved)
    module_name = f"product_matching_loss_hook_{source_hash[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load loss hook: {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    if not callable(getattr(module, "compute_loss", None)):
        raise TypeError(f"Loss hook must define compute_loss(...): {resolved}")
    initialize = getattr(module, "initialize_loss", None)
    if initialize is not None and not callable(initialize):
        raise TypeError("Loss hook initialize_loss must be callable when provided")
    return LoadedLossHook(
        name=resolved.stem,
        path=resolved,
        sha256=source_hash,
        module=module,
    )
