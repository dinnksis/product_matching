from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


INSTRUCTION = (
    "Determine whether Query and Document are listings of exactly the same marketplace "
    "product and the same variant. Ignore wording and attribute-schema differences. "
    "Answer no if the model, part number or SKU, size, color, volume, quantity, material, "
    "or bundle configuration differs."
)
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


def format_pair(product1: str, product2: str, instruction: str = INSTRUCTION) -> str:
    """Format two line-oriented product records for the original reranker prompt."""
    return (
        f"<Instruct>: {instruction}\n"
        f"<Query>:\n{product1}\n"
        f"<Document>:\n{product2}"
    )


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
        pair_budget = self.max_length - reserved
        if pair_budget <= 0:
            raise ValueError("max_length is too small for the Qwen prompt")
        pair_texts, targets = [], []
        for row in rows:
            first = row.get("product_text_1", row.get("text_1", row.get("name_1")))
            second = row.get("product_text_2", row.get("text_2", row.get("name_2")))
            if first is None or second is None:
                raise KeyError(
                    "Rows must contain product_text_1/product_text_2 or name_1/name_2"
                )
            pair_texts.append(format_pair(str(first), str(second)))
            if self.include_labels:
                targets.append(float(row["target"]))

        # One batched Rust-tokenizer call is much faster than one Python call per
        # sample. Training uses a persistent on-disk token cache, while this
        # collator remains useful for inference and small smoke tests.
        encoded = self.tokenizer(
            pair_texts,
            add_special_tokens=False,
            truncation=True,
            max_length=pair_budget,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]
        sequences = [
            self.prefix_ids + pair_ids + self.suffix_ids for pair_ids in encoded
        ]

        batch = self.tokenizer.pad(
            {"input_ids": sequences}, padding=True, return_tensors="pt"
        )
        if self.include_labels:
            batch["targets"] = torch.tensor(targets, dtype=torch.float32)
        return batch


def yes_probability(logits: torch.Tensor, yes_id: int, no_id: int) -> torch.Tensor:
    pair_logits = torch.stack((logits[:, -1, no_id], logits[:, -1, yes_id]), dim=1)
    return pair_logits.softmax(dim=1)[:, 1]
