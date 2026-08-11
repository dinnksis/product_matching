from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


INSTRUCTION = "Определи, являются ли два названия карточками одного и того же товара."
PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    '<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def preferred_cuda_dtype() -> torch.dtype:
    """Use bf16 on Ampere/Hopper and fp16 on older Kaggle GPUs such as T4."""
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def format_pair(name1: str, name2: str, instruction: str = INSTRUCTION) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {name1}\n<Document>: {name2}"


@dataclass
class QwenBatchCollator:
    tokenizer: Any
    max_length: int = 256
    include_labels: bool = False

    def __post_init__(self) -> None:
        self.prefix_ids = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.yes_id = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.no_id = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        reserved = len(self.prefix_ids) + len(self.suffix_ids)
        pair_budget = self.max_length - reserved - (1 if self.include_labels else 0)
        if pair_budget <= 0:
            raise ValueError("max_length is too small for the Qwen prompt")
        sequences, targets = [], []
        for row in rows:
            pair_ids = self.tokenizer.encode(
                format_pair(str(row["name_1"]), str(row["name_2"])),
                add_special_tokens=False,
                truncation=True,
                max_length=pair_budget,
            )
            sequence = self.prefix_ids + pair_ids + self.suffix_ids
            if self.include_labels:
                target = self.yes_id if float(row["target"]) >= 0.5 else self.no_id
                sequence.append(target)
                targets.append(target)
            sequences.append(sequence)

        batch = self.tokenizer.pad(
            {"input_ids": sequences}, padding=True, return_tensors="pt"
        )
        if self.include_labels:
            labels = torch.full_like(batch["input_ids"], -100)
            labels[:, -1] = torch.tensor(targets, dtype=torch.long)
            batch["labels"] = labels
        return batch


def yes_probability(logits: torch.Tensor, yes_id: int, no_id: int) -> torch.Tensor:
    pair_logits = torch.stack((logits[:, -1, no_id], logits[:, -1, yes_id]), dim=1)
    return pair_logits.softmax(dim=1)[:, 1]
