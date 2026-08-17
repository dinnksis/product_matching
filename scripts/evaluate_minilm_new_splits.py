#!/usr/bin/env python3
"""Evaluate one trained serialization checkpoint on IID, hard, and OOD splits."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_serialization_ablation import attach_text, evaluate, loader_options
from src.cross_encoder_training import (
    CrossEncoderBatchCollator,
    CrossEncoderPairDataset,
    build_pair_token_cache,
)
from src.qwen_reranker import preferred_cuda_dtype
from src.qwen_training import FixedLengthBatchSampler
from src.serialization_ablation import VARIANTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token-cache-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    training_report = json.loads(args.training_report.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, local_files_only=True, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint_dir,
        local_files_only=True,
        attn_implementation=str(config["attention_implementation"]),
    ).to(device).eval()
    items = pd.read_parquet(
        args.prepared_dir / "items.parquet",
        columns=["id", "category", f"text_{args.variant.lower()}"],
    )
    collator = CrossEncoderBatchCollator(tokenizer.pad_token_id)
    amp_dtype = preferred_cuda_dtype()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    reports = {}
    for split in config["validation_splits"]:
        split_started = time.perf_counter()
        pairs = pd.read_parquet(args.prepared_dir / f"{split}_pairs.parquet")
        validation = attach_text(pairs, items, f"text_{args.variant.lower()}")
        cache = build_pair_token_cache(
            validation,
            tokenizer,
            args.token_cache_dir,
            f"{args.variant.lower()}-{split}",
            str(args.checkpoint_dir),
            int(config["max_length"]),
            int(config["tokenization_batch_size"]),
            int(config["tokenization_log_every"]),
        )
        targets = validation["target"].to_numpy(dtype=np.float32)
        dataset = CrossEncoderPairDataset(cache, targets)
        sampler = FixedLengthBatchSampler(
            cache,
            np.arange(len(dataset)),
            int(config["eval_batch_size"]),
            both_orientations=True,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collator,
            **loader_options(config, persistent=False),
        )
        metrics, predictions = evaluate(
            model, loader, validation, cache, device, amp_dtype, int(config["max_length"])
        )
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(split_dir / "predictions.parquet", index=False)
        report = {
            "experiment": f"{config['experiment']}_{args.variant.lower()}_{split}",
            "serialization": args.variant,
            "validation_split": split,
            "model": config["model"],
            "model_revision": training_report["model_revision"],
            "human_labels_only": True,
            "training_examples": training_report["training_examples"],
            "training_seconds": training_report["training_seconds"],
            "training_wall_seconds": training_report["training_seconds"],
            "validation_examples": len(dataset),
            "validation_positive_examples": int(targets.sum()),
            "validation_positive_rate": float(targets.mean()),
            "total_evaluation_pipeline_seconds": time.perf_counter() - split_started,
            **metrics,
            "args": {
                **config,
                "variant": args.variant,
                "validation_split": split,
                "validation_pair_order": "MEAN_AB_BA",
                "model_revision": training_report["model_revision"],
                "effective_batch_size": int(config["batch_size"]) * int(config["gradient_accumulation"]),
            },
        }
        (split_dir / "evaluation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        reports[split] = report
        print(json.dumps({"variant": args.variant, "split": split, **metrics}, ensure_ascii=False), flush=True)
    (args.output_dir / "evaluation_reports.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "COMPLETED").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
