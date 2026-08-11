"""Zero-shot/adapter validation and timed inference for Qwen3-Reranker.

This script intentionally is not part of the CPU fallback submission yet.
Run it on a GPU machine after downloading the model files.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline import attach_item_fields
from src.qwen_reranker import QwenBatchCollator, yes_probability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--prepared-dir", type=Path, default=Path("prepared/human"))
    parser.add_argument("--pairs", type=Path, help="Defaults to prepared-dir/val_pairs.parquet")
    parser.add_argument("--output", type=Path, default=Path("predictions/qwen_names_val.csv"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--attention-implementation", default="sdpa", choices=["sdpa", "flash_attention_2"])
    parser.add_argument("--limit", type=int, help="Useful for throughput smoke tests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    pairs_path = args.pairs or args.prepared_dir / "val_pairs.parquet"
    items = pd.read_parquet(args.prepared_dir / "items.parquet", columns=["id", "name", "category"])
    pairs = pd.read_parquet(pairs_path)
    if args.limit:
        pairs = pairs.head(args.limit).copy()
    data = attach_item_fields(pairs, items, fields=("name", "category"))
    data_ready = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attention_implementation
    ).cuda().eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()
    model_ready = time.perf_counter()

    collator = QwenBatchCollator(tokenizer, max_length=args.max_length)
    rows = data.to_dict("records")
    loader = DataLoader(rows, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    predictions: list[np.ndarray] = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            scores = yes_probability(model(**batch).logits, collator.yes_id, collator.no_id)
            predictions.append(scores.float().cpu().numpy())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    scores = np.concatenate(predictions)

    result = pairs[["id1", "id2"]].copy()
    result["predict"] = scores
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    report = {
        "pairs": len(result),
        "data_loading_seconds": data_ready - total_started,
        "model_loading_seconds": model_ready - data_ready,
        "inference_seconds": elapsed,
        "total_seconds": time.perf_counter() - total_started,
        "pairs_per_second": len(result) / elapsed,
        "estimated_seconds_1m_pairs": 1_000_000 * elapsed / len(result),
    }
    if "target" in data:
        scored = pd.DataFrame({"target": data["target"], "predict": scores, "category": data["category_1"]})
        per_category = scored.groupby("category").apply(
            lambda group: average_precision_score(group["target"], group["predict"]),
            include_groups=False,
        )
        report["macro_average_precision"] = float(per_category.mean())
        report["per_category_average_precision"] = {str(k): float(v) for k, v in per_category.items()}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
