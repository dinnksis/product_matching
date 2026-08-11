"""LoRA fine-tuning of Qwen3-Reranker-0.6B using product names only.

Do not run this during repository checks: it downloads a model and requires GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from sklearn.metrics import average_precision_score
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline import attach_item_fields
from src.qwen_reranker import QwenBatchCollator, yes_probability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--prepared-dir", type=Path, default=Path("prepared/human"))
    parser.add_argument("--output-dir", type=Path, default=Path("model/qwen_names_lora"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--attention-implementation", default="sdpa", choices=["sdpa", "flash_attention_2"])
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--training-mode", choices=["lora", "full"], default="lora")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate(model, loader, collator, categories, device) -> tuple[float, dict[str, float]]:
    model.eval()
    scores = []
    with torch.inference_mode():
        for batch in loader:
            targets = batch.pop("labels")[:, -1]
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            logits = model(**batch).logits[:, -2, :]
            probability = torch.stack((logits[:, collator.no_id], logits[:, collator.yes_id]), dim=1).softmax(1)[:, 1]
            scores.extend(probability.float().cpu().tolist())
    frame = pd.DataFrame({"target": loader.dataset.targets, "predict": scores, "category": categories})
    per_category = frame.groupby("category").apply(
        lambda group: average_precision_score(group["target"], group["predict"]),
        include_groups=False,
    )
    return float(per_category.mean()), {str(k): float(v) for k, v in per_category.items()}


class RecordDataset(torch.utils.data.Dataset):
    def __init__(self, frame: pd.DataFrame, random_swap: bool = False):
        self.records = frame.to_dict("records")
        self.targets = frame["target"].astype(float).tolist()
        self.random_swap = random_swap
    def __len__(self): return len(self.records)
    def __getitem__(self, index):
        row = self.records[index].copy()
        if self.random_swap and random.random() < 0.5:
            row["name_1"], row["name_2"] = row["name_2"], row["name_1"]
        return row


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    is_main = not distributed or dist.get_rank() == 0
    random.seed(args.seed + local_rank); np.random.seed(args.seed + local_rank); torch.manual_seed(args.seed + local_rank)
    items = pd.read_parquet(args.prepared_dir / "items.parquet", columns=["id", "name", "category"])
    train = attach_item_fields(pd.read_parquet(args.prepared_dir / "train_pairs.parquet"), items)
    validation = attach_item_fields(pd.read_parquet(args.prepared_dir / "val_pairs.parquet"), items)

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attention_implementation
    )
    if args.training_mode == "lora":
        config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, config)
    model = model.to(device)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    if is_main:
        if args.training_mode == "lora":
            model.print_trainable_parameters()
        else:
            trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            print(f"Full fine-tuning: {trainable:,} trainable parameters")

    train_collator = QwenBatchCollator(tokenizer, args.max_length, include_labels=True)
    val_collator = QwenBatchCollator(tokenizer, args.max_length, include_labels=True)
    train_dataset = RecordDataset(train, random_swap=True)
    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=args.seed) if distributed else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=train_sampler is None, sampler=train_sampler, collate_fn=train_collator, pin_memory=True)
    val_dataset = RecordDataset(validation)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, collate_fn=val_collator, pin_memory=True)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(total_updates * 0.05)), total_updates)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    started = time.perf_counter(); optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            loss = model(**batch).loss / args.gradient_accumulation
            loss.backward()
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            if is_main and step % 200 == 0:
                print(json.dumps({"epoch": epoch + 1, "step": step, "loss": float(loss) * args.gradient_accumulation}))
        if distributed:
            dist.barrier()
        if is_main:
            inference_model = model.module if distributed else model
            macro_ap, per_category = evaluate(inference_model, val_loader, val_collator, validation["category_1"].tolist(), device)
            print(json.dumps({"epoch": epoch + 1, "macro_average_precision": macro_ap, "per_category": per_category}, ensure_ascii=False))
        if distributed:
            dist.barrier()

    if is_main:
        saved_model = model.module if distributed else model
        args.output_dir.mkdir(parents=True, exist_ok=True)
        saved_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        (args.output_dir / "training_report.json").write_text(
            json.dumps({"elapsed_seconds": time.perf_counter() - started, "macro_average_precision": macro_ap, "args": vars(args)}, default=str, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved {args.training_mode} model to {args.output_dir}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
