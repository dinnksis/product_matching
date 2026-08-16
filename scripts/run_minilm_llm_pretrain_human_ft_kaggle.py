#!/usr/bin/env python3
"""Fine-tune an LLM-pretrained MiniLM checkpoint on frozen human train."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

import create_cross_encoder_training_notebook as cross_builder
import create_minilm_validation_baseline_notebook as builder
import create_qwen_training_notebook as shared
import push_minilm_pretrain_checkpoint_dataset as checkpoint_push
import run_kaggle_notebook as kaggle
import run_minilm_validation_baseline_kaggle as baseline_runner


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "cross_encoder_minilm_llm_pretrain_human_ft.json"
BASELINE_CONFIG = ROOT / "configs" / "cross_encoder_minilm_validation_baseline.json"
NOTEBOOK = ROOT / "notebooks" / "minilm_llm_pretrain_human_ft_2xt4.ipynb"
EXPERIMENT_NAME = "minilm_llm_pretrain_human_ft_v1"
KERNEL_SLUG = "product-matching-minilm-llm-pretrain-human-ft-v1"
KERNEL_TITLE = "Product Matching MiniLM LLM Pretrain Human FT v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--checkpoint-tag", default="1ep")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source-dir", type=Path, default=builder.DEFAULT_SOURCE_DIR)
    parser.add_argument("--checkpoint-source-dir", type=Path)
    parser.add_argument("--checkpoint-stage-dir", type=Path)
    parser.add_argument("--checkpoint-dataset-slug")
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--experiment-name")
    parser.add_argument("--kernel-slug")
    parser.add_argument("--kernel-title")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def assert_baseline_parameters(config: dict[str, object]) -> None:
    baseline = cross_builder.load_training_config(BASELINE_CONFIG)
    differences = {
        key: {"baseline": baseline.get(key), "experiment": config.get(key)}
        for key in sorted(set(baseline) | set(config))
        if key != "model" and baseline.get(key) != config.get(key)
    }
    if differences:
        raise ValueError(
            "Human fine-tune parameters differ from the previous validation baseline: "
            + json.dumps(differences, ensure_ascii=False, sort_keys=True)
        )


def main() -> None:
    args = parse_args()
    kaggle.load_dotenv(args.env_file)
    owner = shared.dotenv_username(args.env_file)
    if not owner:
        raise SystemExit("Set KAGGLE_USERNAME in .env")

    checkpoint_tag = kaggle.validate_slug(args.checkpoint_tag, "checkpoint tag")
    config_path = args.config or (
        DEFAULT_CONFIG
        if checkpoint_tag == "1ep"
        else ROOT / "configs" / f"cross_encoder_minilm_llm_pretrain_{checkpoint_tag}_human_ft.json"
    )
    checkpoint_source_dir = args.checkpoint_source_dir or (
        ROOT / "model" / f"pretrain_minilm_{checkpoint_tag}"
    )
    checkpoint_dataset_slug = args.checkpoint_dataset_slug or (
        f"product-matching-minilm-llm-pretrain-{checkpoint_tag}"
    )
    checkpoint_stage_dir = args.checkpoint_stage_dir or (
        ROOT / ".kaggle" / "datasets" / checkpoint_dataset_slug
    )
    experiment_name = args.experiment_name or (
        EXPERIMENT_NAME
        if checkpoint_tag == "1ep"
        else f"minilm_llm_pretrain_{checkpoint_tag}_human_ft_v1"
    )
    kernel_slug = args.kernel_slug or (
        KERNEL_SLUG
        if checkpoint_tag == "1ep"
        else f"product-matching-minilm-{checkpoint_tag}-human-ft-v1"
    )
    kernel_title = args.kernel_title or (
        KERNEL_TITLE
        if checkpoint_tag == "1ep"
        else f"Product Matching MiniLM {checkpoint_tag} Human FT v1"
    )
    notebook_path = args.notebook or (
        NOTEBOOK
        if checkpoint_tag == "1ep"
        else ROOT / "notebooks" / f"minilm_llm_pretrain_{checkpoint_tag}_human_ft_2xt4.ipynb"
    )

    dataset = builder.load_manifest(args.source_dir, owner)
    config = cross_builder.load_training_config(config_path)
    assert_baseline_parameters(config)
    checkpoint_manifest = checkpoint_push.build_payload(
        checkpoint_source_dir,
        checkpoint_stage_dir,
        owner,
        dataset_slug=checkpoint_dataset_slug,
    )
    checkpoint = {
        "dataset": checkpoint_manifest["dataset"],
        "manifest_sha256": checkpoint_push.sha256_file(
            checkpoint_stage_dir / checkpoint_push.MANIFEST_NAME
        ),
    }
    notebook = builder.build_notebook(
        dataset,
        config,
        experiment_name=experiment_name,
        experiment_title=(
            f"MiniLM: LLM pretrain {checkpoint_tag} → human fine-tune → IID/hard/OOD"
        ),
        experiment_description=(
            f"Checkpoint после {checkpoint_tag} на non-OOD LLM-разметке получает новую "
            "эпоху fine-tuning только на frozen human train. Optimizer создаётся "
            "заново; все гиперпараметры совпадают с предыдущим human-only baseline. "
            "После обучения один checkpoint оценивается на IID, hard и OOD."
        ),
        initial_checkpoint=checkpoint,
    )
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, notebook_path)

    if not args.dry_run:
        baseline_runner.verify_remote_dataset(
            str(dataset["dataset"]), str(dataset["manifest_sha256"])
        )
        checkpoint_push.verify_remote_dataset(
            kaggle.kaggle_command(),
            str(checkpoint["dataset"]),
            str(checkpoint["manifest_sha256"]),
            set(checkpoint_manifest["files"]),
        )

    command = [
        sys.executable,
        str(ROOT / "scripts/run_kaggle_notebook.py"),
        str(notebook_path),
        "--env-file",
        str(args.env_file),
        "--slug",
        kernel_slug,
        "--title",
        kernel_title,
        "--dataset",
        str(dataset["dataset"]),
        "--dataset",
        str(checkpoint["dataset"]),
        "--no-env-sources",
    ]
    if args.dry_run:
        command.append("--dry-run")
    if args.no_wait:
        command.append("--no-wait")
    if args.no_download:
        command.append("--no-download")
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
