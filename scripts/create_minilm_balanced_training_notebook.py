#!/usr/bin/env python3
"""Build the two-epoch MiniLM notebook on the balanced human+LLM split."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

import create_cross_encoder_training_notebook as cross_builder
import create_mxbai_training_notebook as balanced_builder


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = ROOT / "notebooks" / "minilm_balanced_llm_2ep_training_2xt4.ipynb"
DEFAULT_CONFIG = ROOT / "configs" / "cross_encoder_minilm_balanced_llm_2ep.json"


def build_notebook(
    manifest: dict[str, object], training_config: dict[str, object]
) -> nbf.NotebookNode:
    notebook = balanced_builder.build_notebook(manifest, training_config)
    replacements = {
        "# mxbai-rerank-xsmall-v1: balanced product matching на 2×Tesla T4":
            "# mMARCO MiniLM: balanced product matching, 2 эпохи на 2×Tesla T4",
        "Full fine-tuning `mixedbread-ai/mxbai-rerank-xsmall-v1`":
            "Full fine-tuning `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`",
        "/kaggle/temp/product_matching_mxbai":
            "/kaggle/temp/product_matching_minilm_balanced_2ep",
        'WORKING_ROOT / "mxbai_xsmall_balanced"':
            'WORKING_ROOT / "minilm_balanced_llm_2ep"',
        'WORKING_ROOT / "mxbai_xsmall_training.log"':
            'WORKING_ROOT / "minilm_balanced_llm_2ep_training.log"',
        "configs/cross_encoder_mxbai_xsmall_balanced_llm.json":
            "configs/cross_encoder_minilm_balanced_llm_2ep.json",
        "DeBERTa-v2 cross-encoder": "MiniLM cross-encoder",
    }
    for cell in notebook.cells:
        source = cell.source
        for old, new in replacements.items():
            source = source.replace(old, new)
        cell.source = source
    notebook.metadata["product_matching_training"].update(
        {
            "experiment": "minilm_balanced_llm_2ep",
            "prepared_data": "physical category and label balance",
        }
    )
    return notebook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_NOTEBOOK)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    owner = balanced_builder.shared.dotenv_username(args.env_file)
    if not owner:
        raise SystemExit("Set KAGGLE_USERNAME in .env")
    config = cross_builder.load_training_config(args.config)
    manifest = balanced_builder.build_dataset(
        balanced_builder.DEFAULT_DATASET_DIR,
        owner,
    )
    notebook = build_notebook(manifest, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, args.output)
    print(f"Wrote notebook: {args.output}")
    print(f"Dataset reference: {manifest['dataset']}")
    print(f"Code bundle SHA-256: {manifest['code_bundle']['sha256']}")


if __name__ == "__main__":
    main()
