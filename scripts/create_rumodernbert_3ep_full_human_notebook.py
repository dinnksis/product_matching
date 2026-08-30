#!/usr/bin/env python3
"""Build the final RuModernBERT 3ep-pretrain -> all-human Kaggle notebook."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

import create_minilm_5ep_full_human_notebook as common


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "cross_encoder_rumodernbert_3ep_full_human_final.json"
DEFAULT_NOTEBOOK = ROOT / "notebooks" / "rumodernbert_3ep_full_human_final_2xt4.ipynb"
DEFAULT_CHECKPOINT_DATASET_REF = (
    "alexproger23/product-matching-rumodernbert-pretrain-3ep"
)
CHECKPOINT_MANIFEST_SHA256 = (
    "4fbfac78fd55130d593b5f81086edd0fffb53534c1aceeccd7ad1cfa8ccde648"
)
EXPERIMENT_NAME = "rumodernbert_3ep_full_human_final"
LOCKED_VALUES = {
    "epochs": 3,
    "batch_size": 24,
    "gradient_accumulation": 4,
    "learning_rate": 4e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
    "scheduler": "cosine",
    "max_length": 384,
    "attention_implementation": "sdpa",
    "train_subset": "all",
    "sampling": "none",
    "loss_weighting": "none",
    "lexical_hard_negative_strength": 0.0,
    "bucket_size_multiplier": 50,
    "dataloader_workers": 4,
    "prefetch_factor": 2,
    "gradient_checkpointing": False,
    "label_smoothing": 0.0,
    "max_grad_norm": 0.5,
    "seed": 42,
    "skip_validation": True,
}


def load_locked_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    differences = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in LOCKED_VALUES.items()
        if config.get(key) != expected
    }
    if differences:
        raise ValueError(
            "Final RuModernBERT recipe differs from the user-locked values: "
            + json.dumps(differences, ensure_ascii=False, sort_keys=True)
        )
    return config


def write_notebook(
    notebook_path: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
    data_dataset_ref: str,
    checkpoint_dataset_ref: str = DEFAULT_CHECKPOINT_DATASET_REF,
) -> dict[str, object]:
    config = load_locked_config(config_path)
    data_audit = common.validate_local_human_data()
    notebook = common.build_notebook(
        config,
        data_audit,
        data_dataset_ref=data_dataset_ref,
        checkpoint_dataset_ref=checkpoint_dataset_ref,
        checkpoint_manifest_sha256=CHECKPOINT_MANIFEST_SHA256,
        experiment_name=EXPERIMENT_NAME,
        notebook_title=(
            "Final RuModernBERT: 3ep pretrain checkpoint -> all human labels"
        ),
        locked_recipe=LOCKED_VALUES,
        expected_amp_dtype="torch.float16",
        expected_world_size=2,
        expected_effective_batch_size=192,
    )
    notebook.metadata["product_matching_final_training"]["architecture"] = (
        "RuModernBERT"
    )
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, notebook_path)
    return data_audit
