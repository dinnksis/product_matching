#!/usr/bin/env python3
"""Evaluate an OpenAI-compatible Qwen judge on frozen human validations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation_metrics import binary_probability_metrics  # noqa: E402


DEFAULT_DATA_DIR = ROOT / "prepared" / "validation_splits_v1" / "human"
DEFAULT_PROMPT = ROOT / "prompts" / "qwen3_5_product_match_judge_v1.md"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "qwen3_5_397b_judge_v1"
SPLITS = ("iid", "hard", "ood")
CHOICES = ("0", "1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8193/v1")
    parser.add_argument("--model")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument(
        "--limit-per-split",
        type=int,
        help="Deterministic diagnostic sample per split; omit for the full run.",
    )
    parser.add_argument("--sample-seed", type=int, default=20260816)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs(
    data_dir: Path, limit_per_split: int | None, sample_seed: int
) -> pd.DataFrame:
    items_path = data_dir / "items.parquet"
    items = pd.read_parquet(items_path, columns=["id", "category", "product_text"])
    if items["id"].duplicated().any():
        raise ValueError(f"Duplicate item IDs in {items_path}")
    item_lookup = items.set_index("id")

    parts: list[pd.DataFrame] = []
    for split_offset, split in enumerate(SPLITS):
        path = data_dir / f"{split}_validation_pairs.parquet"
        frame = pd.read_parquet(path, columns=["id1", "id2", "target"])
        if limit_per_split is not None and limit_per_split < len(frame):
            frame = frame.sample(
                n=limit_per_split,
                random_state=sample_seed + split_offset,
            ).sort_index()
        frame = frame.copy()
        frame.insert(0, "split", split)
        parts.append(frame)

    pairs = pd.concat(parts, ignore_index=True)
    if not pairs["target"].isin([0.0, 1.0]).all():
        raise ValueError("Validation targets must be binary")
    if pairs.duplicated(["id1", "id2"]).any():
        raise ValueError("Validation splits contain repeated oriented pairs")

    for side in (1, 2):
        ids = pairs[f"id{side}"]
        pairs[f"category_{side}"] = ids.map(item_lookup["category"])
        pairs[f"product_text_{side}"] = ids.map(item_lookup["product_text"])
    if pairs[["category_1", "category_2", "product_text_1", "product_text_2"]].isna().any().any():
        raise ValueError("At least one validation item is missing from items.parquet")
    if not pairs["category_1"].eq(pairs["category_2"]).all():
        raise ValueError("Cross-category validation pair found")
    pairs["category"] = pairs.pop("category_1")
    pairs = pairs.drop(columns="category_2")
    return pairs


def discover_model(base_url: str, requested_model: str | None, timeout: float) -> str:
    if requested_model:
        return requested_model
    last_error: Exception | None = None
    response_data: dict[str, Any] | None = None
    session = requests.Session()
    session.trust_env = False
    for attempt in range(1, 7):
        try:
            response = session.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
            response.raise_for_status()
            response_data = response.json()
            break
        except Exception as error:
            last_error = error
            if attempt < 6:
                time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))
    if response_data is None:
        raise RuntimeError("Could not discover the served model after 6 attempts") from last_error
    models = [entry["id"] for entry in response_data.get("data", [])]
    if len(models) != 1:
        raise ValueError(f"Expected one served model, found {models}; pass --model")
    return str(models[0])


def build_user_message(row: dict[str, Any]) -> str:
    return (
        f"Категория пары: {row['category']}\n\n"
        f"<CARD_A>\n{row['product_text_1']}\n</CARD_A>\n\n"
        f"<CARD_B>\n{row['product_text_2']}\n</CARD_B>"
    )


def binary_softmax(logprob_0: float, logprob_1: float) -> float:
    difference = logprob_1 - logprob_0
    if difference >= 0:
        return float(1.0 / (1.0 + math.exp(-difference)))
    exponent = math.exp(difference)
    return float(exponent / (1.0 + exponent))


def parse_choice(response: dict[str, Any]) -> dict[str, Any]:
    choice = response["choices"][0]
    answer = str(choice["message"]["content"]).strip()
    if answer not in CHOICES:
        raise ValueError(f"Unexpected model answer: {answer!r}")
    token_rows = choice["logprobs"]["content"]
    answer_token = next(
        (entry for entry in token_rows if str(entry.get("token", "")).strip() in CHOICES),
        None,
    )
    if answer_token is None:
        raise ValueError("No binary answer token in completion logprobs")
    alternatives = {
        str(entry["token"]).strip(): float(entry["logprob"])
        for entry in answer_token["top_logprobs"]
        if str(entry.get("token", "")).strip() in CHOICES
    }
    if set(alternatives) != set(CHOICES):
        raise ValueError(f"Both binary logprobs were not returned: {alternatives}")
    logprob_0, logprob_1 = alternatives["0"], alternatives["1"]
    usage = response.get("usage") or {}
    return {
        "answer": int(answer),
        "predict": binary_softmax(logprob_0, logprob_1),
        "logprob_0": logprob_0,
        "logprob_1": logprob_1,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "response_id": str(response.get("id", "")),
    }


class JudgeClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        system_prompt: str,
        timeout: float,
        retries: int,
    ) -> None:
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.retries = retries
        self.local = threading.local()

    def session(self) -> requests.Session:
        session = getattr(self.local, "session", None)
        if session is None:
            session = requests.Session()
            session.trust_env = False
            self.local.session = session
        return session

    def judge(self, row: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": build_user_message(row)},
            ],
            "temperature": 0,
            "max_tokens": 4,
            "logprobs": True,
            "top_logprobs": 5,
            "chat_template_kwargs": {"enable_thinking": False},
            "structured_outputs": {"choice": list(CHOICES)},
        }
        last_error: Exception | None = None
        started = time.perf_counter()
        for attempt in range(1, self.retries + 1):
            try:
                response = self.session().post(self.url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                parsed = parse_choice(response.json())
                return {
                    "id1": int(row["id1"]),
                    "id2": int(row["id2"]),
                    **parsed,
                    "attempts": attempt,
                    "latency_seconds": time.perf_counter() - started,
                    "completed_at": utc_now(),
                }
            except Exception as error:  # network and schema failures are both retryable
                last_error = error
                if attempt < self.retries:
                    delay = min(12.0, 0.5 * (2 ** (attempt - 1)))
                    time.sleep(delay + random.random() * 0.2)
        raise RuntimeError(
            f"Pair ({row['id1']}, {row['id2']}) failed after {self.retries} attempts: "
            f"{last_error!r}"
        ) from last_error


def load_checkpoint(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    if frame.duplicated(["id1", "id2"]).any():
        raise ValueError(f"Duplicate pair in checkpoint {path}")
    return {
        (int(row["id1"]), int(row["id2"])): row
        for row in frame.to_dict("records")
    }


def write_checkpoint(
    path: Path, responses: dict[tuple[int, int], dict[str, Any]]
) -> None:
    if not responses:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pd.DataFrame(responses.values()).sort_values(["id1", "id2"]).to_parquet(
        temporary, index=False
    )
    os.replace(temporary, path)


def infer_pending(
    rows: list[dict[str, Any]],
    client: JudgeClient,
    responses: dict[tuple[int, int], dict[str, Any]],
    checkpoint_path: Path,
    workers: int,
    checkpoint_every: int,
) -> tuple[list[str], float]:
    errors: list[str] = []
    if not rows:
        return errors, 0.0
    started = time.perf_counter()
    completed = 0
    iterator = iter(rows)
    max_inflight = workers * 2

    with ThreadPoolExecutor(max_workers=workers) as executor:
        inflight: dict[Future[dict[str, Any]], dict[str, Any]] = {}

        def submit_one() -> bool:
            try:
                row = next(iterator)
            except StopIteration:
                return False
            inflight[executor.submit(client.judge, row)] = row
            return True

        for _ in range(min(max_inflight, len(rows))):
            submit_one()

        while inflight:
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for future in done:
                row = inflight.pop(future)
                try:
                    result = future.result()
                    key = (int(result["id1"]), int(result["id2"]))
                    responses[key] = result
                except Exception as error:
                    error_text = f"({row['id1']}, {row['id2']}): {error!r}"
                    errors.append(error_text)
                    print(f"request_error={error_text}", flush=True)
                completed += 1
                submit_one()

                if completed % checkpoint_every == 0 or completed == len(rows):
                    write_checkpoint(checkpoint_path, responses)
                    elapsed = time.perf_counter() - started
                    rate = completed / max(elapsed, 1e-9)
                    remaining = (len(rows) - completed) / max(rate, 1e-9)
                    print(
                        f"completed={completed}/{len(rows)} total_saved={len(responses)} "
                        f"errors={len(errors)} rate={rate:.2f} pairs/s eta={remaining / 60:.1f}m",
                        flush=True,
                    )
    return errors, time.perf_counter() - started


def threshold_metrics(target: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    prediction = score >= 0.5
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "predicted_positive_rate": float(prediction.mean()),
    }


def evaluate_split(frame: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target = frame["target"].to_numpy(dtype=np.int8)
    score = frame["predict"].to_numpy(dtype=np.float64)
    probability_metrics = binary_probability_metrics(target, score)
    per_category: list[dict[str, Any]] = []
    for category, part in frame.groupby("category", sort=True):
        part_target = part["target"].to_numpy(dtype=np.int8)
        part_score = part["predict"].to_numpy(dtype=np.float64)
        per_category.append(
            {
                "category": str(category),
                "pairs": len(part),
                "positives": int(part_target.sum()),
                "positive_rate": float(part_target.mean()),
                "average_precision": float(
                    average_precision_score(part_target, part_score)
                ),
                "roc_auc": (
                    float(roc_auc_score(part_target, part_score))
                    if np.unique(part_target).size == 2
                    else None
                ),
            }
        )
    summary = {
        "pairs": len(frame),
        "positives": int(target.sum()),
        "positive_rate": float(target.mean()),
        "categories": len(per_category),
        "macro_average_precision": float(
            np.mean([row["average_precision"] for row in per_category])
        ),
        "overall_average_precision": float(average_precision_score(target, score)),
        "overall_roc_auc": (
            probability_metrics["roc_auc"]
        ),
        "brier_score": float(brier_score_loss(target, score)),
        "log_loss": probability_metrics["log_loss"],
        "recall_at_precision_0_99": probability_metrics[
            "recall_at_precision_0_99"
        ],
        "threshold_at_precision_0_99": probability_metrics[
            "threshold_at_precision_0_99"
        ],
        "score_min": float(score.min()),
        "score_median": float(np.median(score)),
        "score_max": float(score.max()),
        "threshold_0_5": threshold_metrics(target, score),
    }
    return summary, per_category


def finalize(
    *,
    pairs: pd.DataFrame,
    responses: dict[tuple[int, int], dict[str, Any]],
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    response_frame = pd.DataFrame(responses.values())
    predictions = pairs.merge(
        response_frame,
        on=["id1", "id2"],
        how="left",
        validate="one_to_one",
    )
    if predictions["predict"].isna().any():
        raise RuntimeError("Cannot finalize: at least one pair has no prediction")
    predictions = predictions.drop(columns=["product_text_1", "product_text_2"])
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)

    per_category_rows: list[dict[str, Any]] = []
    split_summaries: dict[str, Any] = {}
    for split in SPLITS:
        part = predictions[predictions["split"].eq(split)]
        summary, categories = evaluate_split(part)
        split_summaries[split] = summary
        per_category_rows.extend({"split": split, **row} for row in categories)
    pd.DataFrame(per_category_rows).to_csv(
        output_dir / "per_category_metrics.csv", index=False
    )

    response_columns = [
        "latency_seconds",
        "attempts",
        "prompt_tokens",
        "completion_tokens",
    ]
    runtime = {
        "mean_request_latency_seconds": float(predictions["latency_seconds"].mean()),
        "p95_request_latency_seconds": float(
            predictions["latency_seconds"].quantile(0.95)
        ),
        "requests_retried": int(predictions["attempts"].gt(1).sum()),
        "total_prompt_tokens_reported": int(predictions["prompt_tokens"].sum()),
        "total_completion_tokens_reported": int(
            predictions["completion_tokens"].sum()
        ),
    }
    assert set(response_columns).issubset(predictions.columns)
    report = {**metadata, "runtime": runtime, "validation_splits": split_summaries}
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.retries <= 0 or args.checkpoint_every <= 0:
        raise ValueError("workers, retries and checkpoint-every must be positive")
    if args.limit_per_split is not None and args.limit_per_split <= 0:
        raise ValueError("--limit-per-split must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = args.prompt.read_text(encoding="utf-8").strip()
    model = discover_model(args.base_url, args.model, args.timeout_seconds)
    pairs = load_inputs(args.data_dir, args.limit_per_split, args.sample_seed)
    checkpoint_path = args.output_dir / "responses.parquet"
    responses = load_checkpoint(checkpoint_path)
    expected_keys = set(zip(pairs["id1"].astype(int), pairs["id2"].astype(int)))
    unexpected = set(responses) - expected_keys
    if unexpected:
        raise ValueError(
            f"Checkpoint contains {len(unexpected)} pairs outside this evaluation"
        )
    pending = [
        row
        for row in pairs.to_dict("records")
        if (int(row["id1"]), int(row["id2"])) not in responses
    ]
    print(
        f"model={model} pairs={len(pairs)} resumed={len(responses)} "
        f"pending={len(pending)} workers={args.workers}",
        flush=True,
    )

    client = JudgeClient(
        base_url=args.base_url,
        model=model,
        system_prompt=system_prompt,
        timeout=args.timeout_seconds,
        retries=args.retries,
    )
    started_at = utc_now()
    errors, inference_seconds = infer_pending(
        pending,
        client,
        responses,
        checkpoint_path,
        args.workers,
        args.checkpoint_every,
    )
    if errors:
        (args.output_dir / "errors.json").write_text(
            json.dumps(errors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise RuntimeError(
            f"{len(errors)} requests failed; successful responses were checkpointed. "
            "Rerun the same command to resume."
        )
    (args.output_dir / "errors.json").unlink(missing_ok=True)

    metadata = {
        "evaluation": args.prompt.stem,
        "model": model,
        "base_url": args.base_url,
        "prompt_path": str(args.prompt),
        "prompt_sha256": sha256_file(args.prompt),
        "data_dir": str(args.data_dir),
        "limit_per_split": args.limit_per_split,
        "sample_seed": args.sample_seed,
        "request_parameters": {
            "temperature": 0,
            "max_tokens": 4,
            "logprobs": True,
            "top_logprobs": 5,
            "enable_thinking": False,
            "structured_choice": list(CHOICES),
        },
        "started_at": started_at,
        "completed_at": utc_now(),
        "new_inference_seconds": inference_seconds,
        "new_pairs_per_second": len(pending) / max(inference_seconds, 1e-9),
        "resumed_pairs": len(pairs) - len(pending),
    }
    finalize(
        pairs=pairs,
        responses=responses,
        output_dir=args.output_dir,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
