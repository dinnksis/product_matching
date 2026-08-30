from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .normalization import json_dumps, normalize_text, parse_attributes, stable_hash64
from .qwen import QwenItemClient, build_generation_prompt
from .retrieval import HybridRetriever
from .validation import CandidateValidation, validate_candidate


@dataclass(frozen=True)
class GenerationTask:
    task_index: int
    synthetic_id: int
    seed: int
    anchor: dict[str, Any]
    examples: list[dict[str, Any]]


class NameRegistry:
    def __init__(self, names: set[str]) -> None:
        self.names = set(names)
        self.lock = threading.Lock()

    def reserve(self, name: str) -> bool:
        normalized = normalize_text(name)
        with self.lock:
            if normalized in self.names:
                return False
            self.names.add(normalized)
            return True


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _atomic_json(value: dict[str, Any] | list[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def build_tasks(
    retriever: HybridRetriever,
    *,
    count: int,
    seed: int,
    id_start: int,
    example_count: int,
    categories: list[str] | None,
    min_attributes: int = 3,
    max_attributes: int = 40,
) -> list[GenerationTask]:
    if count < 1:
        raise ValueError("count must be positive")
    bank = retriever.bank
    allowed_categories = sorted(categories or bank["category"].astype(str).unique().tolist())
    unknown = set(allowed_categories) - set(bank["category"].astype(str))
    if unknown:
        raise ValueError(f"Requested categories are absent from index: {sorted(unknown)}")

    eligible: dict[str, np.ndarray] = {}
    for category in allowed_categories:
        positions: list[int] = []
        for position in retriever.by_category[category]:
            attributes = parse_attributes(bank.iloc[int(position)]["attributes"])
            if min_attributes <= len(attributes) <= max_attributes:
                positions.append(int(position))
        if not positions:
            raise ValueError(f"No eligible schema donors in category {category!r}")
        eligible[category] = np.asarray(positions, dtype=np.int32)

    rng = np.random.default_rng(seed)
    shuffled: dict[str, np.ndarray] = {}
    cursors: dict[str, int] = {}
    for category, positions in eligible.items():
        shuffled[category] = rng.permutation(positions)
        cursors[category] = 0

    tasks: list[GenerationTask] = []
    for task_index in range(count):
        category = allowed_categories[task_index % len(allowed_categories)]
        cursor = cursors[category]
        positions = shuffled[category]
        if cursor >= len(positions):
            positions = rng.permutation(eligible[category])
            shuffled[category] = positions
            cursor = 0
        anchor_position = int(positions[cursor])
        cursors[category] = cursor + 1
        example_positions = retriever.retrieve(anchor_position, k=example_count)
        anchor = bank.iloc[anchor_position].to_dict()
        examples = [bank.iloc[position].to_dict() for position in example_positions]
        task_seed = stable_hash64(seed, task_index) % (2**31 - 1)
        tasks.append(
            GenerationTask(
                task_index=task_index,
                synthetic_id=id_start - task_index,
                seed=int(task_seed),
                anchor=anchor,
                examples=examples,
            )
        )
    return tasks


def generate_task(
    task: GenerationTask,
    *,
    client: QwenItemClient,
    source_names: set[str],
    registry: NameRegistry,
    generation_attempts: int,
    task_retry_round: int,
    task_seed_offset: int,
    run_signature: str,
    prompt_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    anchor_attributes = parse_attributes(task.anchor["attributes"])
    keys = list(anchor_attributes)
    feedback: list[str] = []
    rejection_history: list[list[str]] = []
    total_latency = 0.0
    total_request_attempts = 0
    last_error: Exception | None = None
    for generation_attempt in range(1, generation_attempts + 1):
        prompt = build_generation_prompt(task.anchor, task.examples, feedback=feedback)
        try:
            response = client.generate(
                prompt,
                category=str(task.anchor["category"]),
                attribute_keys=keys,
                seed=(
                    task.seed
                    + task_seed_offset
                    + task_retry_round * generation_attempts
                    + generation_attempt
                    - 1
                ),
            )
        except Exception as error:
            last_error = error
            feedback = [f"request_error:{type(error).__name__}"]
            rejection_history.append(feedback)
            continue
        total_latency += float(response["latency_seconds"])
        total_request_attempts += int(response["request_attempts"])
        validation: CandidateValidation = validate_candidate(
            response["item"],
            anchor=task.anchor,
            examples=task.examples,
            existing_normalized_names=source_names,
        )
        if not validation.valid:
            feedback = validation.reasons
            rejection_history.append(feedback)
            continue
        if not registry.reserve(validation.item["name"]):
            feedback = ["duplicate_generated_name"]
            rejection_history.append(feedback)
            continue

        item_row = {
            "id": int(task.synthetic_id),
            "name": validation.item["name"],
            "attributes": json_dumps(validation.item["attributes"]),
            "category": validation.item["category"],
        }
        metadata_row = {
            "id": int(task.synthetic_id),
            "task_index": int(task.task_index),
            "source_id": int(task.anchor["id"]),
            "subtype": str(task.anchor["subtype"]),
            "retrieved_ids": json_dumps([int(row["id"]) for row in task.examples]),
            "run_signature": run_signature,
            "prompt_sha256": prompt_sha256,
            "model": client.model,
            "task_retry_round": task_retry_round,
            "task_seed_offset": task_seed_offset,
            "generation_attempts": generation_attempt,
            "request_attempts": total_request_attempts,
            "latency_seconds": total_latency,
            "prompt_tokens": int(response["prompt_tokens"]),
            "completion_tokens": int(response["completion_tokens"]),
            "response_id": str(response["response_id"]),
            "rejection_history": json_dumps(rejection_history),
            **validation.metrics,
            "completed_at": utc_now(),
        }
        return item_row, metadata_row
    raise RuntimeError(
        f"task {task.task_index} failed after {generation_attempts} generation attempts; "
        f"last_feedback={feedback}; last_error={last_error!r}"
    )


def _load_checkpoint(
    output_dir: Path,
    run_signature: str,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    items_path = output_dir / "items.parquet"
    metadata_path = output_dir / "generation_metadata.parquet"
    if not items_path.exists() and not metadata_path.exists():
        return {}, {}
    if not items_path.exists() or not metadata_path.exists():
        raise ValueError("Incomplete checkpoint: items and metadata must both exist")
    items = pd.read_parquet(items_path)
    metadata = pd.read_parquet(metadata_path)
    if items["id"].duplicated().any() or metadata["task_index"].duplicated().any():
        raise ValueError("Checkpoint contains duplicate IDs or task indices")
    signatures = set(metadata["run_signature"].astype(str))
    if signatures != {run_signature}:
        raise ValueError(
            "Checkpoint belongs to another configuration; use another output directory"
        )
    item_by_id = {int(row["id"]): row for row in items.to_dict("records")}
    metadata_by_task = {
        int(row["task_index"]): row for row in metadata.to_dict("records")
    }
    if set(item_by_id) != {int(row["id"]) for row in metadata_by_task.values()}:
        raise ValueError("Checkpoint items and metadata IDs do not align")
    items_by_task = {
        task_index: item_by_id[int(metadata_row["id"])]
        for task_index, metadata_row in metadata_by_task.items()
    }
    return items_by_task, metadata_by_task


def _write_checkpoint(
    output_dir: Path,
    items_by_task: dict[int, dict[str, Any]],
    metadata_by_task: dict[int, dict[str, Any]],
) -> None:
    if not items_by_task:
        return
    task_indices = sorted(items_by_task)
    items = pd.DataFrame([items_by_task[index] for index in task_indices])[
        ["id", "name", "attributes", "category"]
    ]
    metadata = pd.DataFrame([metadata_by_task[index] for index in task_indices])
    _atomic_parquet(items, output_dir / "items.parquet")
    _atomic_parquet(metadata, output_dir / "generation_metadata.parquet")


def run_generation(
    *,
    index_dir: Path,
    output_dir: Path,
    client: QwenItemClient,
    system_prompt: str,
    count: int,
    seed: int,
    id_start: int,
    example_count: int,
    categories: list[str] | None,
    workers: int,
    generation_attempts: int,
    checkpoint_every: int,
    task_retries: int = 3,
    task_seed_offset: int = 0,
) -> dict[str, Any]:
    if workers < 1 or generation_attempts < 1 or checkpoint_every < 1:
        raise ValueError("workers, generation_attempts and checkpoint_every must be positive")
    if task_retries < 0:
        raise ValueError("task_retries must be non-negative")
    retriever = HybridRetriever.from_index_dir(index_dir)
    profile_path = index_dir / "profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    prompt_sha256 = sha256_text(system_prompt)
    signature_payload = {
        "version": "item_generation_v1",
        "index_profile": profile,
        "model": client.model,
        "structured_output": bool(getattr(client, "structured_output", True)),
        "prompt_sha256": prompt_sha256,
        "count": count,
        "seed": seed,
        "id_start": id_start,
        "example_count": example_count,
        "categories": categories,
    }
    run_signature = sha256_text(
        json.dumps(signature_payload, ensure_ascii=False, sort_keys=True)
    )
    tasks = build_tasks(
        retriever,
        count=count,
        seed=seed,
        id_start=id_start,
        example_count=example_count,
        categories=categories,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    items_by_task, metadata_by_task = _load_checkpoint(output_dir, run_signature)
    source_names = {normalize_text(value) for value in retriever.bank["name"]}
    registry = NameRegistry(
        source_names
        | {normalize_text(row["name"]) for row in items_by_task.values()}
    )
    pending = [task for task in tasks if task.task_index not in items_by_task]
    resumed_count = len(items_by_task)
    print(
        f"generate requested={count} resumed={len(items_by_task)} pending={len(pending)} "
        f"workers={workers} task_retries={task_retries} model={client.model}",
        flush=True,
    )
    errors: list[dict[str, Any]] = []
    task_retry_events = 0
    started = time.perf_counter()
    completed_since_checkpoint = 0
    iterator = iter(pending)
    retry_queue: deque[tuple[GenerationTask, int]] = deque()
    inflight: dict[
        Future[tuple[dict[str, Any], dict[str, Any]]],
        tuple[GenerationTask, int],
    ] = {}

    def submit_task(
        executor: ThreadPoolExecutor,
        task: GenerationTask,
        task_retry_round: int,
    ) -> None:
        future = executor.submit(
            generate_task,
            task,
            client=client,
            source_names=source_names,
            registry=registry,
            generation_attempts=generation_attempts,
            task_retry_round=task_retry_round,
            task_seed_offset=task_seed_offset,
            run_signature=run_signature,
            prompt_sha256=prompt_sha256,
        )
        inflight[future] = (task, task_retry_round)

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        try:
            task = next(iterator)
        except StopIteration:
            if not retry_queue:
                return False
            task, task_retry_round = retry_queue.popleft()
        else:
            task_retry_round = 0
        submit_task(executor, task, task_retry_round)
        return True

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in range(min(len(pending), workers * 2)):
            submit_next(executor)
        while inflight:
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for future in done:
                task, task_retry_round = inflight.pop(future)
                try:
                    item_row, metadata_row = future.result()
                    items_by_task[task.task_index] = item_row
                    metadata_by_task[task.task_index] = metadata_row
                    completed_since_checkpoint += 1
                except Exception as error:
                    if task_retry_round < task_retries:
                        task_retry_events += 1
                        retry_queue.append((task, task_retry_round + 1))
                        submit_next(executor)
                    else:
                        errors.append(
                            {
                                "task_index": task.task_index,
                                "source_id": int(task.anchor["id"]),
                                "task_retry_rounds": task_retry_round,
                                "error": repr(error),
                            }
                        )
                        submit_next(executor)
                else:
                    submit_next(executor)
                if completed_since_checkpoint >= checkpoint_every:
                    _write_checkpoint(output_dir, items_by_task, metadata_by_task)
                    completed_since_checkpoint = 0
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    generated_this_run = len(items_by_task) - resumed_count
                    print(
                        f"generate saved={len(items_by_task)}/{count} errors={len(errors)} "
                        f"retried={task_retry_events} retry_queue={len(retry_queue)} "
                        f"rate={(generated_this_run / elapsed):.2f}/s",
                        flush=True,
                    )

    _write_checkpoint(output_dir, items_by_task, metadata_by_task)
    elapsed = time.perf_counter() - started
    summary = {
        **signature_payload,
        "run_signature": run_signature,
        "generated": len(items_by_task),
        "pending": count - len(items_by_task),
        "errors": len(errors),
        "task_retries": task_retries,
        "task_seed_offset": task_seed_offset,
        "task_retry_events": task_retry_events,
        "elapsed_seconds_this_run": elapsed,
        "items_path": str((output_dir / "items.parquet").resolve()),
        "metadata_path": str((output_dir / "generation_metadata.parquet").resolve()),
    }
    _atomic_json(errors, output_dir / "errors.json")
    _atomic_json(summary, output_dir / "summary.json")
    return summary
