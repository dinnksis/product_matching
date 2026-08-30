#!/usr/bin/env python3
"""Prepare and run one frozen three-epoch BGE experiment on one H100."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "bge_3ep_sft_oodtrain_h100_v1.json"
DEFAULT_MODEL_DIR = ROOT / "model" / "pretrain_bge_2ep"
DEFAULT_HUMAN_DIR = ROOT / "prepared" / "validation_splits_v1" / "human"
DEFAULT_WORK_DIR = ROOT / "artifacts" / "bge_3ep_sft_oodtrain_h100_v1"
TRAINER = ROOT / "scripts" / "train_bge_3ep_h100.py"
LOSS_HOOK = ROOT / "scripts" / "bge_h100_finite_bce.py"

EXPECTED_CONFIG_SHA256 = "93cf66d22535b9580af2e26e89d33e0f4f8c7b1ce5023679ddaaa5a68e2946e0"
EXPECTED_CHECKPOINT_FILES = {
    "config.json": "3f143138299caf72270c79814d1f0e3c38fc168e2d30f1e6cc4c70f6cdb481f5",
    "model.safetensors": "cdaf66bb271e6cc742267aa0aec0c890be1c898a93c469c137f5174ea9eeba72",
    "special_tokens_map.json": "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835",
    "tokenizer.json": "8bf8afbfd11306bd872018c53bfdf2e160a56f8edbcf49933324404791c148d3",
    "tokenizer_config.json": "b87c8703482b0300d3da30e201519aa641f6a450f5eb5bf1e624afbf70c74d80",
}
EXPECTED_DATA_FILES = {
    "items.parquet": "a787b58485632bad9039ad04f5b6210080e8d91e0c03e3a5c79d678962a14b1d",
    "train_pairs.parquet": "ffc8ac7283ca0fe1ac8a39e8ca31cb4fa069465576eeb636005c72636935e616",
    "ood_validation_pairs.parquet": "e12eebba6afd6c307bd70475eea29d80df513280b262328cf819bac65e8b22a4",
    "iid_validation_pairs.parquet": "8964422e8c3b254a355d35b5fb60568fed4b6532abb1e8918aba1050f7ecb798",
    "hard_validation_pairs.parquet": "6731ff9c41100b0ea21d6868cd91cbfab1038832d500753cbab8d4e6962fb064",
}
EXPECTED_EXECUTION_FILES = {
    "configs/bge_3ep_sft_oodtrain_h100_v1.json": EXPECTED_CONFIG_SHA256,
    "scripts/train_bge_3ep_h100.py": "134a318bddea37a25513604a797c0e1f2ff5416edc1b8ebc0ff0725723ee5140",
    "scripts/bge_h100_finite_bce.py": "ca805008554f686599882edcb573777ce2182ef60b9de00369540158e7cb260e",
    "requirements-bge-h100.txt": "d6992082889c25c70db2babff8c92f8caab0e5a34df1e72a249ba5ab35625621",
    "scripts/train_cross_encoder.py": "f13e06a42c6517e9674b47c8c95f0d7e41c8e5373cafdb47411aa624e9143a39",
    "src/cross_encoder_training.py": "55ef5cf491ab4f7a1ac885e3e5b6798035cc3d5e63be4ce7313ec6bf87ba21cf",
    "src/cross_encoder_experiment_hooks.py": "c9c6f796b7ec325af250b0387aba08331db46b1eb3ee5e54ab76ded7a08f037e",
    "src/data_pipeline.py": "4f0578906c7693787f38b1fb40d103dc4870524018f9f40a02d694dfd571cd95",
    "src/experiment_protocol.py": "4d743922e8c31961e8b26301944940af12823d5385a71ae2ac649b03fe6a5ef0",
    "src/pair_features.py": "e8051da6916b9030962f1950a21e1916071ed20e2b1159a8d557d13b208013a1",
    "src/qwen_training.py": "ae852a5846105cd1c8aa9ee9fafd37cf8c3b6177ee40382080461ab653bc8757",
    "src/qwen_reranker.py": "6d5b3d688b56557776cefda965e8f9b1758ef03f16c72627fe0a7c0a2329737d",
    "src/validation_metrics.py": "659f972d541c0e0027d1fa2f8b0e380c2e644067e54d89df2bdfe98c613f2ad3",
}
EXPECTED_ROWS = {
    "train_pairs.parquet": (306_669, 80_136),
    "ood_validation_pairs.parquet": (41_171, 9_155),
    "iid_validation_pairs.parquet": (12_000, 3_118),
    "hard_validation_pairs.parquet": (5_814, 1_481),
}
EXPECTED_COMBINED_ROWS = 347_840
EXPECTED_COMBINED_POSITIVES = 89_291
EXPECTED_ITEMS = 711_304
EXPECTED_OUTPUT_FILES = {
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "training_config.json",
    "training_report.json",
    "iid_validation_predictions.parquet",
    "hard_validation_predictions.parquet",
}


class BgeH100RunError(RuntimeError):
    """Raised when local input, execution or output differs from the contract."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BgeH100RunError(f"Could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise BgeH100RunError(f"Expected JSON object in {path}")
    return value


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise BgeH100RunError(f"{label} must be a regular non-symlink file: {path}")


def validate_hashed_files(
    directory: Path,
    expected: Mapping[str, str],
    label: str,
    *,
    exact_names: bool = False,
) -> dict[str, str]:
    if not directory.is_dir():
        raise BgeH100RunError(f"{label} directory does not exist: {directory}")
    if exact_names:
        actual_names = {
            path.name for path in directory.iterdir() if path.is_file() or path.is_symlink()
        }
        if actual_names != set(expected):
            raise BgeH100RunError(
                f"{label} file set differs: expected={sorted(expected)}, actual={sorted(actual_names)}"
            )
    observed: dict[str, str] = {}
    for name, expected_hash in expected.items():
        path = directory / name
        require_regular_file(path, f"{label} {name}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise BgeH100RunError(
                f"{label} hash differs for {name}: {actual_hash} != {expected_hash}"
            )
        observed[name] = actual_hash
    return observed


def validate_execution_sources() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected_hash in EXPECTED_EXECUTION_FILES.items():
        path = ROOT / relative
        require_regular_file(path, f"execution source {relative}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise BgeH100RunError(
                f"Execution source hash differs for {relative}: {actual_hash} != {expected_hash}"
            )
        observed[relative] = actual_hash
    return observed


def load_base_config(path: Path) -> dict[str, Any]:
    require_regular_file(path, "training config")
    if path.resolve() == DEFAULT_CONFIG.resolve() and sha256_file(path) != EXPECTED_CONFIG_SHA256:
        raise BgeH100RunError("Frozen BGE H100 config hash differs")
    config = read_json(path)
    expected = {
        "model_backend": "sequence_classification",
        "model_load_kwargs": {"local_files_only": True},
        "trust_remote_code": False,
        "epochs": 3,
        "batch_size": 64,
        "eval_batch_size": 192,
        "gradient_accumulation": 3,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "max_length": 384,
        "attention_implementation": "sdpa",
        "sampling": "none",
        "train_subset": "all",
        "loss_weighting": "none",
        "lexical_hard_negative_strength": 0.0,
        "bucket_size_multiplier": 50,
        "dataloader_workers": 16,
        "prefetch_factor": 4,
        "tokenization_batch_size": 1024,
        "tokenization_log_every": 50,
        "gradient_checkpointing": False,
        "symmetric_validation": True,
        "label_smoothing": 0.0,
        "max_grad_norm": 0.5,
        "log_every": 50,
        "seed": 42,
    }
    if set(config) != set(expected) | {"model"}:
        raise BgeH100RunError("Training config key set differs")
    for key, value in expected.items():
        if config.get(key) != value:
            raise BgeH100RunError(
                f"Training config differs at {key}: {config.get(key)!r} != {value!r}"
            )
    return config


def validate_pair_frame(frame: pd.DataFrame, name: str) -> None:
    if list(frame.columns) != ["id1", "id2", "target"]:
        raise BgeH100RunError(f"{name} columns differ: {list(frame.columns)}")
    expected_rows, expected_positives = EXPECTED_ROWS[name]
    if len(frame) != expected_rows:
        raise BgeH100RunError(f"{name} row count differs: {len(frame)} != {expected_rows}")
    if frame.isna().any().any():
        raise BgeH100RunError(f"{name} contains nulls")
    if not frame["target"].isin([0.0, 1.0]).all():
        raise BgeH100RunError(f"{name} targets are not binary")
    positives = int(frame["target"].sum())
    if positives != expected_positives:
        raise BgeH100RunError(
            f"{name} positive count differs: {positives} != {expected_positives}"
        )
    if (frame["id1"] == frame["id2"]).any():
        raise BgeH100RunError(f"{name} contains self-pairs")


def unordered_pair_index(frame: pd.DataFrame) -> pd.MultiIndex:
    id1 = frame["id1"].to_numpy(dtype=np.int64, copy=False)
    id2 = frame["id2"].to_numpy(dtype=np.int64, copy=False)
    return pd.MultiIndex.from_arrays([np.minimum(id1, id2), np.maximum(id1, id2)])


def validate_data(human_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    hashes = validate_hashed_files(human_dir, EXPECTED_DATA_FILES, "human data")
    items = pd.read_parquet(
        human_dir / "items.parquet", columns=["id", "product_text", "category"]
    )
    if len(items) != EXPECTED_ITEMS or not items["id"].is_unique:
        raise BgeH100RunError("Item catalog count/uniqueness differs")
    if items.isna().any().any():
        raise BgeH100RunError("Item catalog contains nulls")
    frames = {
        name: pd.read_parquet(human_dir / name)
        for name in EXPECTED_ROWS
    }
    for name, frame in frames.items():
        validate_pair_frame(frame, name)
    all_pairs = pd.concat(list(frames.values()), ignore_index=True)
    unordered = unordered_pair_index(all_pairs)
    if unordered.has_duplicates:
        raise BgeH100RunError("Human split union contains duplicate unordered pairs")
    item_categories = items.set_index("id")["category"]
    if not all_pairs["id1"].isin(item_categories.index).all() or not all_pairs[
        "id2"
    ].isin(item_categories.index).all():
        raise BgeH100RunError("Human split union references missing item IDs")
    categories_1 = all_pairs["id1"].map(item_categories)
    categories_2 = all_pairs["id2"].map(item_categories)
    if not categories_1.equals(categories_2):
        raise BgeH100RunError("Human split union contains cross-category pairs")
    train = pd.concat(
        [
            frames["train_pairs.parquet"].assign(label_source="human_train"),
            frames["ood_validation_pairs.parquet"].assign(
                label_source="human_former_ood"
            ),
        ],
        ignore_index=True,
    )
    if len(train) != EXPECTED_COMBINED_ROWS or int(train["target"].sum()) != EXPECTED_COMBINED_POSITIVES:
        raise BgeH100RunError("Combined BGE train count/positives differ")
    train_items = set(train["id1"]) | set(train["id2"])
    overlap: dict[str, int] = {}
    validation_categories: dict[str, int] = {}
    for split in ("iid_validation_pairs.parquet", "hard_validation_pairs.parquet"):
        frame = frames[split]
        validation_items = set(frame["id1"]) | set(frame["id2"])
        overlap[split] = len(train_items & validation_items)
        validation_categories[split] = int(frame["id1"].map(item_categories).nunique())
    train_categories = int(train["id1"].map(item_categories).nunique())
    if overlap != {"iid_validation_pairs.parquet": 0, "hard_validation_pairs.parquet": 0}:
        raise BgeH100RunError(f"Train/validation item overlap differs: {overlap}")
    if train_categories != 20 or validation_categories != {
        "iid_validation_pairs.parquet": 18,
        "hard_validation_pairs.parquet": 18,
    }:
        raise BgeH100RunError(
            f"Category coverage differs: train={train_categories}, validation={validation_categories}"
        )
    report = {
        "source_sha256": hashes,
        "items": len(items),
        "combined_train_rows": len(train),
        "combined_train_positives": int(train["target"].sum()),
        "combined_train_categories": train_categories,
        "training_source_counts": {
            str(key): int(value)
            for key, value in train["label_source"].value_counts().items()
        },
        "validation_rows": {
            "iid": len(frames["iid_validation_pairs.parquet"]),
            "hard": len(frames["hard_validation_pairs.parquet"]),
        },
        "validation_categories": validation_categories,
        "train_validation_item_overlap": overlap,
        "ood_policy": "included_in_train_not_evaluated",
    }
    frames["combined_train"] = train
    return frames, report


def atomic_copy_or_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".pending")
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def install_static_tokenizer_file(source: Path, destination: Path) -> None:
    """Install a tokenizer sidecar that Transformers may omit on save."""
    require_regular_file(source, "initial tokenizer sidecar")
    expected_hash = EXPECTED_CHECKPOINT_FILES[source.name]
    if sha256_file(source) != expected_hash:
        raise BgeH100RunError(f"Initial tokenizer sidecar hash differs: {source}")
    if destination.exists():
        require_regular_file(destination, "saved tokenizer sidecar")
        if sha256_file(destination) != expected_hash:
            raise BgeH100RunError(f"Saved tokenizer sidecar hash differs: {destination}")
        return
    temporary = destination.with_name(destination.name + ".pending")
    shutil.copy2(source, temporary)
    if sha256_file(temporary) != expected_hash:
        raise BgeH100RunError("Copied tokenizer sidecar hash differs")
    os.replace(temporary, destination)


def prepare_data(human_dir: Path, prepared_dir: Path) -> dict[str, Any]:
    frames, source_report = validate_data(human_dir)
    manifest_path = prepared_dir / "prepared_data_manifest.json"
    train_path = prepared_dir / "train_pairs.parquet"
    items_path = prepared_dir / "items.parquet"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        require_regular_file(train_path, "prepared combined train")
        require_regular_file(items_path, "prepared item catalog")
        if sha256_file(train_path) != manifest.get("combined_train_sha256"):
            raise BgeH100RunError("Prepared combined train hash differs")
        if sha256_file(items_path) != EXPECTED_DATA_FILES["items.parquet"]:
            raise BgeH100RunError("Prepared item catalog hash differs")
        if manifest.get("source_report") != source_report:
            raise BgeH100RunError("Prepared data source report differs")
        return manifest
    if prepared_dir.exists() and any(prepared_dir.iterdir()):
        raise BgeH100RunError(f"Partial prepared directory exists: {prepared_dir}")
    prepared_dir.mkdir(parents=True, exist_ok=True)
    temporary_train = train_path.with_name(train_path.name + ".pending")
    frames["combined_train"].to_parquet(
        temporary_train, index=False, compression="zstd"
    )
    os.replace(temporary_train, train_path)
    atomic_copy_or_link(human_dir / "items.parquet", items_path)
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "source_report": source_report,
        "train_order": ["train_pairs.parquet", "ood_validation_pairs.parquet"],
        "combined_train_sha256": sha256_file(train_path),
        "combined_train_bytes": train_path.stat().st_size,
        "items_sha256": sha256_file(items_path),
    }
    manifest["payload_sha256"] = canonical_sha256(manifest)
    atomic_write_json(manifest_path, manifest)
    return manifest


def resolved_config(base_config: Mapping[str, Any], model_dir: Path) -> dict[str, Any]:
    result = dict(base_config)
    result["model"] = str(model_dir.resolve())
    return result


def run_command_logged(command: Sequence[str], log_path: Path, env: Mapping[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=ROOT,
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        returncode = process.wait()
    if returncode != 0:
        raise BgeH100RunError(
            f"Command failed with exit code {returncode}; inspect {log_path}"
        )


def training_command(
    *,
    config_path: Path,
    prepared_dir: Path,
    human_dir: Path,
    output_dir: Path,
    token_cache_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(TRAINER),
        "--config",
        str(config_path),
        "--prepared-dir",
        str(prepared_dir),
        "--output-dir",
        str(output_dir),
        "--token-cache-dir",
        str(token_cache_dir),
        "--loss-hook",
        str(LOSS_HOOK),
        "--validation-split",
        f"iid={human_dir / 'iid_validation_pairs.parquet'}",
        "--validation-split",
        f"hard={human_dir / 'hard_validation_pairs.parquet'}",
    ]


def validate_prediction_file(path: Path, expected_rows: int) -> str:
    require_regular_file(path, "validation predictions")
    frame = pd.read_parquet(path)
    required = {"id1", "id2", "target", "category_1", "score"}
    if not required.issubset(frame.columns) or len(frame) != expected_rows:
        raise BgeH100RunError(
            f"Prediction schema/count differs for {path}: {len(frame)}, {list(frame.columns)}"
        )
    scores = frame["score"].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise BgeH100RunError(f"Prediction scores are invalid: {path}")
    return sha256_file(path)


def validate_training_output(
    output_dir: Path,
    config_path: Path,
    initial_model_sha256: str,
) -> dict[str, Any]:
    if not output_dir.is_dir():
        raise BgeH100RunError(f"Training output directory is missing: {output_dir}")
    actual_files = {
        path.name for path in output_dir.iterdir() if path.is_file() or path.is_symlink()
    }
    if actual_files != EXPECTED_OUTPUT_FILES:
        raise BgeH100RunError(
            f"Training output file set differs: expected={sorted(EXPECTED_OUTPUT_FILES)}, "
            f"actual={sorted(actual_files)}"
        )
    for path in output_dir.iterdir():
        require_regular_file(path, f"training output {path.name}")
    report = read_json(output_dir / "training_report.json")
    args = report.get("args")
    if not isinstance(args, dict):
        raise BgeH100RunError("Training report args are missing")
    expected_args = {
        "epochs": 3,
        "batch_size": 64,
        "eval_batch_size": 192,
        "gradient_accumulation": 3,
        "learning_rate": 2e-5,
        "sampling": "none",
        "loss_weighting": "none",
        "max_length": 384,
        "seed": 42,
    }
    for key, value in expected_args.items():
        if args.get(key) != value:
            raise BgeH100RunError(
                f"Training report argument differs at {key}: {args.get(key)!r} != {value!r}"
            )
    if report.get("original_training_examples") != EXPECTED_COMBINED_ROWS:
        raise BgeH100RunError("Training report original row count differs")
    if report.get("training_examples") != EXPECTED_COMBINED_ROWS * 3:
        raise BgeH100RunError("Training report processed example count differs")
    if report.get("training_source_counts") != {
        "human_train": 306_669,
        "human_former_ood": 41_171,
    }:
        raise BgeH100RunError("Training source counts differ")
    validations = report.get("validation_splits")
    if not isinstance(validations, dict) or set(validations) != {"iid", "hard"}:
        raise BgeH100RunError("Validation split set differs")
    for name, rows in (("iid", 12_000), ("hard", 5_814)):
        entry = validations[name]
        if not isinstance(entry, dict) or entry.get("examples") != rows:
            raise BgeH100RunError(f"Validation report differs for {name}")
        macro = entry.get("macro_average_precision")
        if not isinstance(macro, (int, float)) or not math.isfinite(macro) or not 0 <= macro <= 1:
            raise BgeH100RunError(f"Invalid validation macro AP for {name}: {macro!r}")
    prediction_hashes = {
        "iid": validate_prediction_file(
            output_dir / "iid_validation_predictions.parquet", 12_000
        ),
        "hard": validate_prediction_file(
            output_dir / "hard_validation_predictions.parquet", 5_814
        ),
    }
    final_model_sha256 = sha256_file(output_dir / "model.safetensors")
    if final_model_sha256 == initial_model_sha256:
        raise BgeH100RunError("Final model bytes equal the initial checkpoint")
    if (output_dir / "training_config.json").read_bytes() != config_path.read_bytes():
        raise BgeH100RunError("Saved training config differs from resolved config")
    return {
        "report_sha256": sha256_file(output_dir / "training_report.json"),
        "final_model_sha256": final_model_sha256,
        "prediction_sha256": prediction_hashes,
        "iid_macro_average_precision": validations["iid"]["macro_average_precision"],
        "hard_macro_average_precision": validations["hard"]["macro_average_precision"],
    }


def deployment_smoke(output_dir: Path, report_path: Path) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available():
        raise BgeH100RunError("CUDA disappeared before deployment smoke")
    tokenizer = AutoTokenizer.from_pretrained(
        output_dir, use_fast=True, local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        output_dir,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 567_755_777:
        raise BgeH100RunError(f"Reloaded parameter count differs: {parameter_count}")
    encoded = tokenizer(
        "Категория: Электроника\nНазвание: тестовый товар",
        "Категория: Электроника\nНазвание: похожий тестовый товар",
        truncation=True,
        max_length=384,
        return_tensors="pt",
    )
    encoded = {key: value.to("cuda") for key, value in encoded.items()}
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logit = model(**encoded).logits[:, 0].float()
    if tuple(logit.shape) != (1,) or not torch.isfinite(logit).all():
        raise BgeH100RunError("Reloaded checkpoint produced invalid logits")
    payload = {
        "schema_version": 1,
        "status": "passed",
        "checked_at": utc_now(),
        "parameters": parameter_count,
        "tokenizer_class": type(tokenizer).__name__,
        "model_class": type(model).__name__,
        "finite_logit": float(logit.item()),
        "max_length": 384,
    }
    atomic_write_json(report_path, payload)
    return payload


def validate_completion(work_dir: Path) -> dict[str, Any]:
    completion_path = work_dir / "run_completed.json"
    completion = read_json(completion_path)
    if completion.get("schema_version") != 1 or completion.get("status") != "complete":
        raise BgeH100RunError("Completion status differs")
    files = completion.get("files")
    if not isinstance(files, dict):
        raise BgeH100RunError("Completion file ledger is missing")
    for relative, expected_hash in files.items():
        path = work_dir / relative
        require_regular_file(path, f"completed artifact {relative}")
        if sha256_file(path) != expected_hash:
            raise BgeH100RunError(f"Completed artifact hash differs: {relative}")
    stored_payload_sha = completion.get("payload_sha256")
    without_hash = dict(completion)
    without_hash.pop("payload_sha256", None)
    if stored_payload_sha != canonical_sha256(without_hash):
        raise BgeH100RunError("Completion payload hash differs")
    return completion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--human-dir", type=Path, default=DEFAULT_HUMAN_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    human_dir = args.human_dir.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    base_config = load_base_config(config_path)
    execution_hashes = validate_execution_sources()
    checkpoint_hashes = validate_hashed_files(
        model_dir, EXPECTED_CHECKPOINT_FILES, "BGE checkpoint"
    )
    _, data_report = validate_data(human_dir)
    plan = {
        "experiment": "bge_3ep_sft_oodtrain_h100_v1",
        "gpu": "one_H100_80GB",
        "fresh_start": True,
        "epochs": 3,
        "learning_rate": 2e-5,
        "effective_batch": 192,
        "amp_dtype": "bfloat16",
        "train_rows": EXPECTED_COMBINED_ROWS,
        "validation": {"iid": 12_000, "hard": 5_814, "ood": -1},
        "model_dir": str(model_dir),
        "human_dir": str(human_dir),
        "work_dir": str(work_dir),
        "checkpoint_sha256": checkpoint_hashes,
        "data": data_report,
        "execution_sha256": execution_hashes,
    }
    if not (args.dry_run or args.prepare_only or args.run):
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        print("Plan only. Pass --dry-run, --prepare-only or --run.")
        return 0
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        print("Dry-run passed; no files were written and no GPU was used.")
        return 0
    completion_path = work_dir / "run_completed.json"
    if completion_path.exists():
        completion = validate_completion(work_dir)
        print(json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True))
        print("Exact completed run already exists; nothing to do.")
        return 0
    prepared_dir = work_dir / "prepared"
    prepared_manifest = prepare_data(human_dir, prepared_dir)
    if args.prepare_only:
        print(json.dumps(prepared_manifest, ensure_ascii=False, indent=2, sort_keys=True))
        print("Prepared data is ready; no GPU was used.")
        return 0
    run_dir = work_dir / "run"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise BgeH100RunError(
            f"Partial output exists and exact optimizer resume is unavailable: {run_dir}"
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = work_dir / "resolved_config.json"
    atomic_write_json(resolved_config_path, resolved_config(base_config, model_dir))
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONHASHSEED": "42",
        }
    )
    preflight_path = work_dir / "h100_preflight.json"
    if preflight_path.exists():
        preflight = read_json(preflight_path)
        if preflight.get("status") != "passed" or preflight.get("parameters") != 567_755_777:
            raise BgeH100RunError("Existing H100 preflight report differs")
    else:
        run_command_logged(
            [
                sys.executable,
                str(TRAINER),
                "--preflight-only",
                "--config",
                str(resolved_config_path),
                "--preflight-report",
                str(preflight_path),
            ],
            work_dir / "preflight.log",
            env,
        )
    started = time.perf_counter()
    run_command_logged(
        training_command(
            config_path=resolved_config_path,
            prepared_dir=prepared_dir,
            human_dir=human_dir,
            output_dir=run_dir,
            token_cache_dir=work_dir / "token_cache",
        ),
        work_dir / "training.log",
        env,
    )
    # XLMRobertaTokenizer in Transformers 4.57.6 may omit this unchanged
    # sidecar from save_pretrained().  It is part of the pinned tokenizer
    # contract, so install the verified original bytes before output audit.
    install_static_tokenizer_file(
        model_dir / "special_tokens_map.json",
        run_dir / "special_tokens_map.json",
    )
    output_summary = validate_training_output(
        run_dir,
        resolved_config_path,
        EXPECTED_CHECKPOINT_FILES["model.safetensors"],
    )
    deployment_smoke(run_dir, work_dir / "deployment_smoke.json")
    artifact_paths = [
        resolved_config_path,
        prepared_dir / "prepared_data_manifest.json",
        preflight_path,
        work_dir / "preflight.log",
        work_dir / "training.log",
        work_dir / "deployment_smoke.json",
        *sorted(run_dir.iterdir()),
    ]
    files = {
        str(path.relative_to(work_dir)): sha256_file(path)
        for path in artifact_paths
        if path.is_file()
    }
    completion: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "experiment": "bge_3ep_sft_oodtrain_h100_v1",
        "elapsed_seconds_including_training_and_validation": time.perf_counter() - started,
        "fresh_start_from_pretrain": True,
        "epochs": 3,
        "learning_rate": 2e-5,
        "checkpoint_model_sha256": EXPECTED_CHECKPOINT_FILES["model.safetensors"],
        "prepared_manifest_sha256": sha256_file(
            prepared_dir / "prepared_data_manifest.json"
        ),
        "output": output_summary,
        "files": files,
    }
    completion["payload_sha256"] = canonical_sha256(completion)
    atomic_write_json(completion_path, completion)
    validate_completion(work_dir)
    print(json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Completed BGE H100 run: {work_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
