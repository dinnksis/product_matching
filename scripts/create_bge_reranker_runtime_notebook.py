#!/usr/bin/env python3
"""Generate the 2xT4 BGE reranker runtime benchmark notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

import create_qwen_training_notebook as shared


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "notebooks" / "bge_reranker_v2_m3_runtime_2xt4.ipynb"
MODEL_NAME = "BAAI/bge-reranker-v2-m3"
DATASET_SLUG = "product-matching-qwen-training"
KERNEL_SLUG = "product-matching-bge-reranker-v2-m3-runtime"
MAX_LENGTH = 192
EXPECTED_TRAIN_PAIRS = 310_767


HF_WORKER_SOURCE = r'''
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["transformers", "sentence_transformers"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=192)
    return parser.parse_args()


def representative_sample(frame, size=8192):
    if len(frame) <= size:
        return frame
    positions = np.linspace(0, len(frame) - 1, num=size, dtype=np.int64)
    return frame.iloc[positions].reset_index(drop=True)


def main():
    args = parse_args()
    worker_started = time.perf_counter()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("RAYON_NUM_THREADS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    frame = pd.read_parquet(args.input)
    frame = frame.iloc[args.gpu_id :: args.world_size].copy()
    frame["sort_key"] = (
        frame["text1"].str.len().astype(np.int32)
        + frame["text2"].str.len().astype(np.int32)
    )
    frame.sort_values(["sort_key", "pair_index"], kind="stable", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    tune_frame = representative_sample(frame)
    data_ready_seconds = time.perf_counter() - worker_started

    load_started = time.perf_counter()
    if args.backend == "transformers":
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            local_files_only=True,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        ).to(device).eval()
    else:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(
            str(args.model),
            device="cuda:0",
            local_files_only=True,
            max_length=args.max_length,
            model_kwargs={
                "torch_dtype": torch.float16,
                "attn_implementation": "sdpa",
            },
        )
        tokenizer = None
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started

    def direct_predict(part, batch_size):
        outputs = []
        batch_max_lengths = []
        with torch.inference_mode():
            for start in range(0, len(part), batch_size):
                batch = part.iloc[start : start + batch_size]
                encoded = tokenizer(
                    batch["text1"].tolist(),
                    batch["text2"].tolist(),
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    pad_to_multiple_of=8,
                    return_tensors="pt",
                )
                batch_max_lengths.append(int(encoded["input_ids"].shape[1]))
                encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
                logits = model(**encoded, return_dict=True).logits.reshape(-1)
                outputs.append(logits.float().cpu().numpy())
        torch.cuda.synchronize()
        return np.concatenate(outputs).astype(np.float32), batch_max_lengths

    def sentence_transformers_predict(part, batch_size):
        pairs = list(zip(part["text1"].tolist(), part["text2"].tolist()))
        with torch.inference_mode():
            values = model.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=False,
                activation_fn=torch.nn.Identity(),
                convert_to_numpy=True,
            )
        torch.cuda.synchronize()
        return np.asarray(values, dtype=np.float32).reshape(-1), []

    predict = direct_predict if args.backend == "transformers" else sentence_transformers_predict
    batch_trials = []
    candidates = [128, 256, 384, 512, 768]
    tuning_started = time.perf_counter()
    for batch_size in candidates:
        gc.collect()
        torch.cuda.empty_cache()
        try:
            warm_rows = min(len(tune_frame), max(256, batch_size))
            predict(tune_frame.iloc[:warm_rows], batch_size)
            torch.cuda.synchronize()
            trial_started = time.perf_counter()
            trial_scores, _ = predict(tune_frame, batch_size)
            trial_seconds = time.perf_counter() - trial_started
            if len(trial_scores) != len(tune_frame) or not np.isfinite(trial_scores).all():
                raise RuntimeError("batch trial returned invalid scores")
            batch_trials.append({
                "batch_size_per_gpu": batch_size,
                "status": "ok",
                "seconds": trial_seconds,
                "pairs_per_second_per_gpu": len(tune_frame) / trial_seconds,
            })
        except torch.cuda.OutOfMemoryError as error:
            batch_trials.append({
                "batch_size_per_gpu": batch_size,
                "status": "oom",
                "error": type(error).__name__,
            })
            gc.collect()
            torch.cuda.empty_cache()
            break
    tuning_seconds = time.perf_counter() - tuning_started
    successful = [trial for trial in batch_trials if trial["status"] == "ok"]
    if not successful:
        raise RuntimeError("No batch size completed successfully")
    chosen = max(successful, key=lambda trial: trial["pairs_per_second_per_gpu"])
    batch_size = int(chosen["batch_size_per_gpu"])

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    inference_started = time.perf_counter()
    scores, batch_max_lengths = predict(frame, batch_size)
    inference_seconds = time.perf_counter() - inference_started
    peak_vram_gib = torch.cuda.max_memory_allocated() / 2**30
    if len(scores) != len(frame) or not np.isfinite(scores).all():
        raise RuntimeError("Full inference returned invalid scores")

    output = frame[["pair_index", "id1", "id2", "target", "category"]].copy()
    output["predict"] = scores
    output.to_parquet(args.output, index=False)
    report = {
        "backend": args.backend,
        "gpu_id": args.gpu_id,
        "gpu_name": torch.cuda.get_device_name(0),
        "pairs": len(frame),
        "dtype": "float16",
        "attention_implementation": "sdpa",
        "max_length": args.max_length,
        "length_sorting": True,
        "data_ready_seconds": data_ready_seconds,
        "model_load_seconds": model_load_seconds,
        "batch_tuning_seconds": tuning_seconds,
        "batch_trials": batch_trials,
        "chosen_batch_size_per_gpu": batch_size,
        "inference_seconds": inference_seconds,
        "pairs_per_second_per_gpu": len(frame) / inference_seconds,
        "peak_vram_gib": peak_vram_gib,
        "observed_padded_tokens_max": max(batch_max_lengths) if batch_max_lengths else None,
        "worker_wall_seconds": time.perf_counter() - worker_started,
        "versions": {
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "sentence_transformers": (
                importlib.metadata.version("sentence-transformers")
                if args.backend == "sentence_transformers" else None
            ),
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
'''


VLLM_WORKER_SOURCE = r'''
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=192)
    return parser.parse_args()


def main():
    args = parse_args()
    worker_started = time.perf_counter()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("RAYON_NUM_THREADS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    import torch
    from vllm import LLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    frame = pd.read_parquet(args.input)
    frame = frame.iloc[args.gpu_id :: args.world_size].copy()
    frame["sort_key"] = (
        frame["text1"].str.len().astype(np.int32)
        + frame["text2"].str.len().astype(np.int32)
    )
    frame.sort_values(["sort_key", "pair_index"], kind="stable", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    data_ready_seconds = time.perf_counter() - worker_started

    load_started = time.perf_counter()
    llm = LLM(
        model=str(args.model),
        runner="pooling",
        trust_remote_code=True,
        dtype="float16",
        max_model_len=args.max_length,
        max_num_seqs=1024,
        max_num_batched_tokens=65536,
        gpu_memory_utilization=0.92,
        enforce_eager=False,
        enable_prefix_caching=False,
        seed=42,
    )
    model_load_seconds = time.perf_counter() - load_started

    warmup_rows = min(2048, len(frame))
    warmup_started = time.perf_counter()
    warmup = llm.score(
        frame["text1"].iloc[:warmup_rows].tolist(),
        frame["text2"].iloc[:warmup_rows].tolist(),
        use_tqdm=False,
    )
    warmup_seconds = time.perf_counter() - warmup_started
    if len(warmup) != warmup_rows:
        raise RuntimeError("vLLM warmup returned an unexpected number of outputs")

    inference_started = time.perf_counter()
    outputs = llm.score(
        frame["text1"].tolist(),
        frame["text2"].tolist(),
        use_tqdm=True,
    )
    inference_seconds = time.perf_counter() - inference_started
    scores = np.asarray([item.outputs.score for item in outputs], dtype=np.float32)
    if len(scores) != len(frame) or not np.isfinite(scores).all():
        raise RuntimeError("vLLM full inference returned invalid scores")

    output = frame[["pair_index", "id1", "id2", "target", "category"]].copy()
    output["predict"] = scores
    output.to_parquet(args.output, index=False)
    report = {
        "backend": "vllm",
        "gpu_id": args.gpu_id,
        "gpu_name": torch.cuda.get_device_name(0),
        "pairs": len(frame),
        "dtype": "float16",
        "runner": "pooling",
        "max_length": args.max_length,
        "length_sorting": True,
        "data_parallel_replicas": args.world_size,
        "max_num_seqs_per_replica": 1024,
        "max_num_batched_tokens_per_replica": 65536,
        "gpu_memory_utilization": 0.92,
        "enforce_eager": False,
        "data_ready_seconds": data_ready_seconds,
        "model_load_seconds": model_load_seconds,
        "warmup_pairs": warmup_rows,
        "warmup_seconds": warmup_seconds,
        "inference_seconds": inference_seconds,
        "pairs_per_second_per_gpu": len(frame) / inference_seconds,
        "worker_wall_seconds": time.perf_counter() - worker_started,
        "versions": {
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "vllm": importlib.metadata.version("vllm"),
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
'''


def markdown(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(text).strip())


def source_fingerprint() -> str:
    payload = {
        "model": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "expected_train_pairs": EXPECTED_TRAIN_PAIRS,
        "hf_worker": HF_WORKER_SOURCE,
        "vllm_worker": VLLM_WORKER_SOURCE,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_notebook(owner: str) -> nbf.NotebookNode:
    dataset_ref = f"{owner}/{DATASET_SLUG}"
    fingerprint = source_fingerprint()
    cells = [
        markdown(
            f"""
            # BAAI/bge-reranker-v2-m3: runtime benchmark на 2×T4

            Zero-shot benchmark исходного checkpoint без обучения. На одинаковом
            component-disjoint human train (`seed=42`, {EXPECTED_TRAIN_PAIRS:,} пар)
            сравниваются прямой `transformers`, `sentence-transformers.CrossEncoder`
            и `vLLM.score`.

            Во всех случаях используются только исходные названия товаров,
            FP16, `max_length={MAX_LENGTH}` и две независимые GPU-реплики. Пары
            сортируются по длине внутри каждого shard, чтобы не тратить время на
            padding. Чистый inference, model load, batch tuning, wall time, peak
            VRAM и GPU utilization сохраняются раздельно.
            """
        ),
        code(
            f"""
            import importlib.metadata
            import json
            import os
            import subprocess
            import sys
            import time
            from pathlib import Path

            import numpy as np
            import pandas as pd

            INPUT_ROOT = Path("/kaggle/input")
            WORKING_ROOT = Path("/kaggle/working")
            TEMP_ROOT = Path("/kaggle/temp/bge_reranker_v2_m3_runtime")
            OUTPUT_ROOT = WORKING_ROOT / "bge_reranker_v2_m3_runtime"
            MODEL_NAME = {MODEL_NAME!r}
            EXPECTED_DATASET_REF = {dataset_ref!r}
            CODE_FINGERPRINT = {fingerprint!r}
            EXPECTED_TRAIN_PAIRS = {EXPECTED_TRAIN_PAIRS}
            MAX_LENGTH = {MAX_LENGTH}
            PIPELINE_STARTED = time.perf_counter()
            TEMP_ROOT.mkdir(parents=True, exist_ok=True)
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

            def exactly_one(filename):
                candidates = list(INPUT_ROOT.glob(f"**/{{filename}}"))
                if len(candidates) != 1:
                    raise RuntimeError(
                        f"Expected exactly one {{filename!r}}, found {{candidates}}"
                    )
                return candidates[0]

            print(subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"],
                check=False, capture_output=True, text=True,
            ).stdout)
            """
        ),
        shared.experiment_run_initialization_cell(),
        markdown("## Зависимости и исходный checkpoint"),
        code(
            """
            install_started = time.perf_counter()
            subprocess.run(
                [
                    sys.executable, "-m", "pip", "install", "--quiet",
                    "--disable-pip-version-check",
                    "transformers>=4.56,<5",
                    "sentence-transformers>=5,<6",
                    "accelerate>=1.10,<2",
                ],
                check=True,
            )
            hf_dependencies_install_seconds = time.perf_counter() - install_started

            from huggingface_hub import snapshot_download

            download_started = time.perf_counter()
            model_dir = Path(snapshot_download(
                MODEL_NAME,
                ignore_patterns=["*.onnx", "*.ot", "*.msgpack", "tf_model.h5", "flax_model*"],
            )).resolve()
            model_download_seconds = time.perf_counter() - download_started
            model_revision = model_dir.name
            model_bytes = sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file())
            hf_versions = {
                name: importlib.metadata.version(name)
                for name in ["torch", "transformers", "sentence-transformers", "huggingface-hub"]
            }
            print(json.dumps({
                "model": MODEL_NAME,
                "revision": model_revision,
                "bytes": model_bytes,
                "download_seconds": model_download_seconds,
                "versions": hf_versions,
            }, indent=2))
            """
        ),
        markdown("## Воспроизводимый human train и пары названий"),
        code(
            """
            data_started = time.perf_counter()
            items_path = exactly_one("items_human.parquet")
            matches_path = exactly_one("matches.parquet")
            items = pd.read_parquet(items_path, columns=["id", "name", "category"])
            matches = pd.read_parquet(matches_path, columns=["id1", "id2", "target"])

            all_ids = pd.unique(matches[["id1", "id2"]].to_numpy().reshape(-1))
            positions = pd.Series(np.arange(len(all_ids), dtype=np.int64), index=all_ids)
            left_positions = positions.loc[matches["id1"]].to_numpy()
            right_positions = positions.loc[matches["id2"]].to_numpy()
            parent = np.arange(len(all_ids), dtype=np.int64)
            component_size = np.ones(len(all_ids), dtype=np.int64)

            def find(node):
                while parent[node] != node:
                    parent[node] = parent[parent[node]]
                    node = int(parent[node])
                return node

            for first, second in zip(left_positions, right_positions):
                root1, root2 = find(int(first)), find(int(second))
                if root1 == root2:
                    continue
                if component_size[root1] < component_size[root2]:
                    root1, root2 = root2, root1
                parent[root2] = root1
                component_size[root1] += component_size[root2]
            components = np.fromiter(
                (find(int(node)) for node in left_positions),
                dtype=np.int64,
                count=len(left_positions),
            )
            unique_components = np.unique(components)
            rng = np.random.default_rng(42)
            validation_components = unique_components[
                rng.random(len(unique_components)) < 0.15
            ]
            train_mask = ~np.isin(components, validation_components)
            train = matches.loc[train_mask].copy()
            train["pair_index"] = train.index.astype(np.int64)
            train.reset_index(drop=True, inplace=True)
            if len(train) != EXPECTED_TRAIN_PAIRS:
                raise RuntimeError(f"Expected {{EXPECTED_TRAIN_PAIRS}} train pairs, got {{len(train)}}")

            lookup = items.set_index("id", verify_integrity=True)
            left = lookup.reindex(train["id1"].to_numpy())
            right = lookup.reindex(train["id2"].to_numpy())
            if left["name"].isna().any() or right["name"].isna().any():
                raise RuntimeError("Train contains product IDs absent from items")
            benchmark_pairs = train[["pair_index", "id1", "id2", "target"]].copy()
            benchmark_pairs["category"] = left["category"].astype(str).to_numpy()
            benchmark_pairs["text1"] = left["name"].astype(str).to_numpy()
            benchmark_pairs["text2"] = right["name"].astype(str).to_numpy()

            # vLLM's score API receives two strings rather than an already
            # truncated pair encoding. Pre-truncate both sides with the model's
            # own tokenizer so every backend sees exactly the same bounded text.
            # XLM-R uses four pair-level special tokens, leaving 94 tokens per
            # product at MAX_LENGTH=192.
            from transformers import AutoTokenizer

            prep_tokenizer = AutoTokenizer.from_pretrained(
                model_dir, local_files_only=True, use_fast=True
            )
            pair_special_tokens = prep_tokenizer.num_special_tokens_to_add(pair=True)
            item_token_budget = (MAX_LENGTH - pair_special_tokens) // 2

            def truncate_names(texts, batch_size=4096):
                result = []
                original_lengths = []
                for start in range(0, len(texts), batch_size):
                    batch = texts.iloc[start : start + batch_size].tolist()
                    token_ids = prep_tokenizer(
                        batch,
                        add_special_tokens=False,
                        truncation=False,
                    )["input_ids"]
                    original_lengths.extend(len(values) for values in token_ids)
                    result.extend(prep_tokenizer.batch_decode(
                        [values[:item_token_budget] for values in token_ids],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ))
                return result, np.asarray(original_lengths, dtype=np.int32)

            benchmark_pairs["text1"], token_lengths_1 = truncate_names(benchmark_pairs["text1"])
            benchmark_pairs["text2"], token_lengths_2 = truncate_names(benchmark_pairs["text2"])
            all_item_token_lengths = np.concatenate([token_lengths_1, token_lengths_2])
            benchmark_input = TEMP_ROOT / "human_train_name_pairs.parquet"
            benchmark_pairs.to_parquet(benchmark_input, index=False)
            data_preparation_seconds = time.perf_counter() - data_started
            length_sum = benchmark_pairs["text1"].str.len() + benchmark_pairs["text2"].str.len()
            data_report = {
                "pairs": len(benchmark_pairs),
                "positive_rate": float(benchmark_pairs.target.mean()),
                "categories": int(benchmark_pairs.category.nunique()),
                "name_pair_characters_p50": float(length_sum.quantile(.50)),
                "name_pair_characters_p95": float(length_sum.quantile(.95)),
                "name_pair_characters_p99": float(length_sum.quantile(.99)),
                "name_pair_characters_max": int(length_sum.max()),
                "pair_special_tokens": int(pair_special_tokens),
                "item_token_budget": int(item_token_budget),
                "item_tokens_p50_before_truncation": float(np.quantile(all_item_token_lengths, .50)),
                "item_tokens_p95_before_truncation": float(np.quantile(all_item_token_lengths, .95)),
                "item_tokens_p99_before_truncation": float(np.quantile(all_item_token_lengths, .99)),
                "item_tokens_max_before_truncation": int(all_item_token_lengths.max()),
                "item_truncation_rate": float((all_item_token_lengths > item_token_budget).mean()),
                "preparation_seconds": data_preparation_seconds,
            }
            print(json.dumps(data_report, ensure_ascii=False, indent=2))
            """
        ),
        markdown("## Две реплики, мониторинг GPU и запуск backend’ов"),
        code(
            f"""
            hf_worker_path = TEMP_ROOT / "hf_runtime_worker.py"
            vllm_worker_path = TEMP_ROOT / "vllm_runtime_worker.py"
            hf_worker_path.write_text({HF_WORKER_SOURCE!r}, encoding="utf-8")
            vllm_worker_path.write_text({VLLM_WORKER_SOURCE!r}, encoding="utf-8")

            def gpu_snapshot():
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,utilization.gpu,memory.used,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                rows = []
                for line in result.stdout.strip().splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) != 4:
                        continue
                    rows.append({{
                        "gpu_id": int(parts[0]),
                        "utilization_percent": float(parts[1]),
                        "memory_used_mib": float(parts[2]),
                        "power_watts": float(parts[3]),
                    }})
                return rows

            def run_backend(backend, worker_path):
                backend_dir = OUTPUT_ROOT / backend
                backend_dir.mkdir(parents=True, exist_ok=True)
                processes = []
                log_handles = []
                started = time.perf_counter()
                try:
                    for gpu_id in range(2):
                        command = [
                            sys.executable, str(worker_path),
                            "--input", str(benchmark_input),
                            "--model", str(model_dir),
                            "--output", str(backend_dir / f"predictions_gpu{{gpu_id}}.parquet"),
                            "--report", str(backend_dir / f"worker_gpu{{gpu_id}}.json"),
                            "--gpu-id", str(gpu_id),
                            "--world-size", "2",
                            "--max-length", str(MAX_LENGTH),
                        ]
                        if worker_path == hf_worker_path:
                            command[2:2] = ["--backend", backend]
                        environment = os.environ.copy()
                        environment.update({{
                            "CUDA_VISIBLE_DEVICES": str(gpu_id),
                            "PYTHONUNBUFFERED": "1",
                            "TOKENIZERS_PARALLELISM": "true",
                            "RAYON_NUM_THREADS": "2",
                            "OMP_NUM_THREADS": "1",
                            "MKL_NUM_THREADS": "1",
                        }})
                        log_handle = (backend_dir / f"worker_gpu{{gpu_id}}.log").open(
                            "w", encoding="utf-8", buffering=1
                        )
                        log_handles.append(log_handle)
                        processes.append(subprocess.Popen(
                            command,
                            env=environment,
                            stdout=log_handle,
                            stderr=subprocess.STDOUT,
                            text=True,
                        ))

                    utilization_rows = []
                    while any(process.poll() is None for process in processes):
                        sample_time = time.perf_counter() - started
                        for row in gpu_snapshot():
                            utilization_rows.append({{"elapsed_seconds": sample_time, **row}})
                        time.sleep(1.0)
                    return_codes = [process.wait() for process in processes]
                finally:
                    for handle in log_handles:
                        handle.close()
                wall_seconds = time.perf_counter() - started
                if any(return_codes):
                    for gpu_id, return_code in enumerate(return_codes):
                        if return_code:
                            log_path = backend_dir / f"worker_gpu{{gpu_id}}.log"
                            tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
                            print(f"--- {{backend}} GPU {{gpu_id}} failure ---\\n{{tail}}")
                    raise RuntimeError(f"{{backend}} workers failed: {{return_codes}}")

                worker_reports = [
                    json.loads((backend_dir / f"worker_gpu{{gpu_id}}.json").read_text())
                    for gpu_id in range(2)
                ]
                predictions = pd.concat(
                    [pd.read_parquet(backend_dir / f"predictions_gpu{{gpu_id}}.parquet") for gpu_id in range(2)],
                    ignore_index=True,
                ).sort_values("pair_index", kind="stable")
                if len(predictions) != EXPECTED_TRAIN_PAIRS or predictions.pair_index.duplicated().any():
                    raise RuntimeError(f"{{backend}} did not score every pair exactly once")
                combined_predictions = backend_dir / "predictions.parquet"
                predictions.to_parquet(combined_predictions, index=False)

                utilization = pd.DataFrame(utilization_rows)
                utilization.to_csv(backend_dir / "gpu_utilization.csv", index=False)
                active = utilization[utilization.utilization_percent > 0]
                inference_seconds = max(report["inference_seconds"] for report in worker_reports)
                report = {{
                    "backend": backend,
                    "pairs": len(predictions),
                    "gpu_count": 2,
                    "wall_seconds_including_load_and_tuning": wall_seconds,
                    "inference_seconds": inference_seconds,
                    "aggregate_pairs_per_second": len(predictions) / inference_seconds,
                    "end_to_end_pairs_per_second": len(predictions) / wall_seconds,
                    "peak_vram_gib_by_gpu": [item.get("peak_vram_gib") for item in worker_reports],
                    "chosen_batch_size_per_gpu": [item.get("chosen_batch_size_per_gpu") for item in worker_reports],
                    "gpu_utilization_mean_percent": float(utilization.utilization_percent.mean()),
                    "gpu_utilization_mean_active_percent": (
                        float(active.utilization_percent.mean()) if len(active) else 0.0
                    ),
                    "gpu_utilization_p95_percent": float(utilization.utilization_percent.quantile(.95)),
                    "gpu_utilization_max_percent": float(utilization.utilization_percent.max()),
                    "gpu_memory_used_max_mib": float(utilization.memory_used_mib.max()),
                    "worker_reports": worker_reports,
                    "predictions_file": str(combined_predictions.relative_to(WORKING_ROOT)),
                }}
                (backend_dir / "report.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(json.dumps(report, ensure_ascii=False, indent=2))
                return report
            """
        ),
        markdown("### 1. Прямой Transformers"),
        code("transformers_report = run_backend(\"transformers\", hf_worker_path)"),
        markdown("### 2. sentence-transformers CrossEncoder"),
        code("sentence_transformers_report = run_backend(\"sentence_transformers\", hf_worker_path)"),
        markdown("### 3. vLLM: большой continuous batch на каждой T4"),
        code(
            """
            vllm_install_started = time.perf_counter()
            subprocess.run(
                [
                    sys.executable, "-m", "pip", "install", "--quiet",
                    "--disable-pip-version-check", "vllm==0.19.1",
                ],
                check=True,
            )
            vllm_install_seconds = time.perf_counter() - vllm_install_started
            vllm_version = importlib.metadata.version("vllm")
            print({"vllm": vllm_version, "install_seconds": vllm_install_seconds})
            vllm_report = run_backend("vllm", vllm_worker_path)
            """
        ),
        markdown("## Метрики, согласованность scores и итоговый отчёт"),
        code(
            """
            from datetime import datetime, timezone
            from sklearn.metrics import average_precision_score

            backend_reports = {
                "transformers": transformers_report,
                "sentence_transformers": sentence_transformers_report,
                "vllm": vllm_report,
            }
            metric_rows = []
            prediction_tables = {}
            for backend, backend_report in backend_reports.items():
                path = WORKING_ROOT / backend_report["predictions_file"]
                predictions = pd.read_parquet(path).sort_values("pair_index", kind="stable")
                prediction_tables[backend] = predictions
                per_category = predictions.groupby("category", sort=True).apply(
                    lambda part: average_precision_score(part["target"], part["predict"]),
                    include_groups=False,
                )
                metric_rows.append({
                    "backend": backend,
                    "macro_average_precision": float(per_category.mean()),
                    "overall_average_precision": float(
                        average_precision_score(predictions.target, predictions.predict)
                    ),
                    "score_min": float(predictions.predict.min()),
                    "score_median": float(predictions.predict.median()),
                    "score_max": float(predictions.predict.max()),
                    "per_category_average_precision": {
                        str(key): float(value) for key, value in per_category.items()
                    },
                })
            metrics_by_backend = {row["backend"]: row for row in metric_rows}
            reference = prediction_tables["transformers"].predict.to_numpy()
            agreement = {}
            for backend in ["sentence_transformers", "vllm"]:
                values = prediction_tables[backend].predict.to_numpy()
                comparison_reference = (
                    1.0 / (1.0 + np.exp(-reference))
                    if backend == "vllm" else reference
                )
                agreement[backend] = {
                    "reference_scale": (
                        "sigmoid(transformers_logit)" if backend == "vllm"
                        else "transformers_logit"
                    ),
                    "pearson_with_transformers": float(
                        np.corrcoef(comparison_reference, values)[0, 1]
                    ),
                    "mean_absolute_difference": float(
                        np.mean(np.abs(comparison_reference - values))
                    ),
                    "max_absolute_difference": float(
                        np.max(np.abs(comparison_reference - values))
                    ),
                }

            ranking = sorted(
                backend_reports,
                key=lambda backend: backend_reports[backend]["aggregate_pairs_per_second"],
                reverse=True,
            )
            fastest_backend = ranking[0]
            fastest_metrics = metrics_by_backend[fastest_backend]
            fastest_batch_sizes = backend_reports[fastest_backend]["chosen_batch_size_per_gpu"]
            fastest_batch_size = (
                fastest_batch_sizes[0]
                if isinstance(fastest_batch_sizes, list) else fastest_batch_sizes
            )
            summary_table = pd.DataFrame([
                {
                    "backend": backend,
                    "inference_seconds": backend_reports[backend]["inference_seconds"],
                    "pairs_per_second_2xt4": backend_reports[backend]["aggregate_pairs_per_second"],
                    "wall_seconds": backend_reports[backend]["wall_seconds_including_load_and_tuning"],
                    "batch_per_gpu": backend_reports[backend]["chosen_batch_size_per_gpu"],
                    "gpu_util_active_percent": backend_reports[backend]["gpu_utilization_mean_active_percent"],
                    "macro_ap": metrics_by_backend[backend]["macro_average_precision"],
                }
                for backend in ranking
            ])
            summary_table.to_csv(OUTPUT_ROOT / "summary.csv", index=False)
            display(summary_table)

            pipeline_seconds = time.perf_counter() - PIPELINE_STARTED
            report = {
                "experiment_group": "pretrain",
                "notes": (
                    "Diagnostic zero-shot runtime comparison on component-disjoint human train; "
                    "names only, not the frozen IID/hard/OOD validation protocol."
                ),
                "model": MODEL_NAME,
                "model_revision": model_revision,
                "model_bytes": model_bytes,
                "input": "name pairs only",
                "max_length": MAX_LENGTH,
                "dtype": "float16",
                "data": data_report,
                "backend_ranking_fastest_first": ranking,
                "fastest_backend": fastest_backend,
                "backends": backend_reports,
                "metrics_by_backend": metrics_by_backend,
                "score_agreement": agreement,
                "dependency_install_seconds": {
                    "huggingface_stack": hf_dependencies_install_seconds,
                    "vllm": vllm_install_seconds,
                },
                "model_download_seconds": model_download_seconds,
                "total_pipeline_seconds": pipeline_seconds,
                "original_training_examples": EXPECTED_TRAIN_PAIRS,
                "validation_examples": EXPECTED_TRAIN_PAIRS,
                "macro_average_precision": fastest_metrics["macro_average_precision"],
                "overall_average_precision": fastest_metrics["overall_average_precision"],
                "per_category_average_precision": fastest_metrics["per_category_average_precision"],
                "validation_splits": {
                    "iid": {
                        "examples": EXPECTED_TRAIN_PAIRS,
                        "macro_average_precision": fastest_metrics["macro_average_precision"],
                        "overall_average_precision": fastest_metrics["overall_average_precision"],
                        "per_category_average_precision": fastest_metrics["per_category_average_precision"],
                        "diagnostic_split": "component_seed_42_human_train",
                    }
                },
                "args": {
                    "model": MODEL_NAME,
                    "max_length": MAX_LENGTH,
                    "batch_size": fastest_batch_size,
                    "seed": 42,
                    "experiment_group": "pretrain",
                    "input_fields": ["name"],
                    "gpu_replicas": 2,
                },
            }
            report_path = OUTPUT_ROOT / "training_report.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            (WORKING_ROOT / "training_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            completion = {
                "status": "complete",
                "run_id": EXPERIMENT_RUN_ID,
                "started_at_utc": EXPERIMENT_STARTED_AT_UTC,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ).replace("+00:00", "Z"),
                "experiment": "bge_reranker_v2_m3_zero_shot_runtime_2xt4",
                "experiment_group": "pretrain",
                "notes": report["notes"],
                "model": MODEL_NAME,
                "dataset_ref": EXPECTED_DATASET_REF,
                "kaggle_kernel_ref": (
                    os.getenv("KAGGLE_KERNEL_RUN_ID")
                    or os.getenv("KAGGLE_KERNEL_INFERENCE_RUN_ID")
                    or ""
                ),
                "code_bundle_sha256": CODE_FINGERPRINT,
                "training_wall_seconds": pipeline_seconds,
                "training_report": report,
            }
            completion_path = WORKING_ROOT / "notebook_completed.json"
            completion_path.write_text(
                json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps({
                "fastest_backend": fastest_backend,
                "ranking": ranking,
                "total_pipeline_seconds": pipeline_seconds,
                "completion_path": str(completion_path),
            }, ensure_ascii=False, indent=2))
            """
        ),
        *shared.google_sheets_tracking_cells(),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "product_matching_benchmark": {
                "model": MODEL_NAME,
                "dataset": dataset_ref,
                "expected_train_pairs": EXPECTED_TRAIN_PAIRS,
                "max_length": MAX_LENGTH,
                "expected_gpus": 2,
                "gpu_type": "NVIDIA Tesla T4",
                "source_fingerprint": fingerprint,
            },
        }
    )
    return notebook


def dotenv_username(path: Path) -> str | None:
    return shared.dotenv_username(path)


def main() -> None:
    owner = dotenv_username(ROOT / ".env")
    if not owner:
        raise SystemExit("Set KAGGLE_USERNAME in .env")
    notebook = build_notebook(owner)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT_PATH)
    print(
        f"Wrote {OUTPUT_PATH.relative_to(ROOT)} "
        f"({OUTPUT_PATH.stat().st_size / 2**20:.2f} MiB), "
        f"fingerprint={source_fingerprint()}"
    )


if __name__ == "__main__":
    main()
