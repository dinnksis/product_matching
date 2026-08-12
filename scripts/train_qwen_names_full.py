"""Full fine-tuning entry point for the product-text Qwen experiment.

Kept separate so it cannot be started accidentally while running the LoRA
experiment. All training mechanics live in train_qwen_names.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_qwen_names import main


if __name__ == "__main__":
    if "--training-mode" not in sys.argv:
        sys.argv.extend(["--training-mode", "full"])
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", "model/qwen_products_full"])
    main()
