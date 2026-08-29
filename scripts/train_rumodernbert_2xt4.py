#!/usr/bin/env python3
"""RuModernBERT 2xT4 adapter over the audited distributed AMP trainer."""

from __future__ import annotations

import sys
from pathlib import Path


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


def configure_adapter() -> None:
    """Bind the generic audited T4/GradScaler implementation to RuModernBERT."""

    distributed_amp.EXPECTED_PARAMETERS = EXPECTED_PARAMETERS
    distributed_amp.EXPECTED_TRAINABLE_PARAMETER_TENSORS = EXPECTED_PARAMETER_TENSORS
    distributed_amp.EXPECTED_MICROBATCH = EXPECTED_MICROBATCH
    distributed_amp.EXPECTED_EVAL_BATCH = EXPECTED_EVAL_BATCH
    distributed_amp.EXPECTED_GRADIENT_ACCUMULATION = EXPECTED_GRADIENT_ACCUMULATION
    distributed_amp.EXPECTED_EFFECTIVE_BATCH = EXPECTED_EFFECTIVE_BATCH


def main() -> int:
    configure_adapter()
    return distributed_amp.main()


if __name__ == "__main__":
    raise SystemExit(main())
