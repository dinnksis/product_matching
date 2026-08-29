#!/usr/bin/env python3
"""Prepare data and run five RuModernBERT SFT experiments sequentially."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_significance import compare_prediction_frames, holm_adjust


DEFAULT_PLAN = ROOT / "configs" / "rumodernbert_3ep_sft_oodtrain_h100_v1.json"
DEFAULT_WORK_DIR = ROOT / "artifacts" / "rumodernbert_3ep_sft_oodtrain_h100_v1"
TRAINER = ROOT / "scripts" / "train_rumodernbert_sft.py"
LOSS_HOOK = ROOT / "scripts" / "rumodernbert_finite_bce.py"
PREPARED_MANIFEST = "prepared_data_manifest.json"
COMPLETION_FILE = "run_completed.json"
SUMMARY_FILE = "campaign_summary.json"
STATE_FILE = "campaign_state.json"
EXPECTED_STAGE_KEYS = [
    "e1_lr8e5",
    "e1_lr4e5",
    "e1_lr1p6e4",
    "e2_selected_lr",
    "e3_selected_lr",
]
EXPECTED_PLAN_SHA256 = "faa06604ffb4c1e02cc7e13e6d889be42911e75cb0c30d8ddd666ba3edbed5e2"
EXPECTED_EXECUTION_SHA256 = {
    "scripts/train_rumodernbert_sft.py": "f4f6a5ce9836f147f7dda2a3c4becc2636387cd16620bc2e494a0725962d49e6",
    "scripts/rumodernbert_finite_bce.py": "e6c1f9dff9c1d76034e3d5f7cf485a49b4f299645e44b611882d2edd3e772ba1",
    "requirements-rumodernbert-h100.txt": "d6992082889c25c70db2babff8c92f8caab0e5a34df1e72a249ba5ab35625621",
    "requirements-cross-encoder.txt": "cf237d47a707ec18d429436f6134b74484ff69777e334efd8522cb132a4efce2",
    "scripts/train_cross_encoder.py": "f13e06a42c6517e9674b47c8c95f0d7e41c8e5373cafdb47411aa624e9143a39",
    "src/cross_encoder_training.py": "55ef5cf491ab4f7a1ac885e3e5b6798035cc3d5e63be4ce7313ec6bf87ba21cf",
    "src/cross_encoder_experiment_hooks.py": "c9c6f796b7ec325af250b0387aba08331db46b1eb3ee5e54ab76ded7a08f037e",
    "src/data_pipeline.py": "4f0578906c7693787f38b1fb40d103dc4870524018f9f40a02d694dfd571cd95",
    "src/experiment_protocol.py": "4d743922e8c31961e8b26301944940af12823d5385a71ae2ac649b03fe6a5ef0",
    "src/pair_features.py": "e8051da6916b9030962f1950a21e1916071ed20e2b1159a8d557d13b208013a1",
    "src/qwen_reranker.py": "6d5b3d688b56557776cefda965e8f9b1758ef03f16c72627fe0a7c0a2329737d",
    "src/qwen_training.py": "ae852a5846105cd1c8aa9ee9fafd37cf8c3b6177ee40382080461ab653bc8757",
    "src/validation_metrics.py": "659f972d541c0e0027d1fa2f8b0e380c2e644067e54d89df2bdfe98c613f2ad3",
    "src/experiment_significance.py": "de25acb18f3ec1a51d17fdb9e15d0d018032396e7ad20fe6f1b6617041b5ef70",
}
TRAINER_CONFIG_KEYS = {
    "model",
    "model_backend",
    "model_load_kwargs",
    "trust_remote_code",
    "epochs",
    "batch_size",
    "eval_batch_size",
    "gradient_accumulation",
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "max_length",
    "attention_implementation",
    "sampling",
    "train_subset",
    "loss_weighting",
    "lexical_hard_negative_strength",
    "bucket_size_multiplier",
    "dataloader_workers",
    "prefetch_factor",
    "tokenization_batch_size",
    "tokenization_log_every",
    "gradient_checkpointing",
    "symmetric_validation",
    "label_smoothing",
    "max_grad_norm",
    "log_every",
    "seed",
}


class RuModernBertCampaignError(RuntimeError):
    """Raised when campaign provenance, data or output differs."""


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
        raise RuModernBertCampaignError(f"Could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuModernBertCampaignError(f"Expected JSON object in {path}")
    return value


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_once_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        if read_json(path) != dict(payload):
            raise RuModernBertCampaignError(f"Write-once JSON differs: {path}")
        return
    atomic_write_json(path, payload)


def require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuModernBertCampaignError(f"{label} must be a regular non-symlink file: {path}")


def load_plan(path: Path) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_PLAN_SHA256:
        raise RuModernBertCampaignError("Frozen campaign plan SHA-256 differs")
    plan = read_json(path)
    if set(plan) != {
        "schema_version",
        "campaign",
        "objective",
        "checkpoint",
        "source_data",
        "training",
        "experiments",
        "selection",
    }:
        raise RuModernBertCampaignError("Campaign plan top-level fields differ")
    if plan["schema_version"] != 1 or plan["campaign"] != "rumodernbert_3ep_sft_oodtrain_h100_v1":
        raise RuModernBertCampaignError("Unexpected RuModernBERT campaign identity")
    experiments = plan.get("experiments")
    if not isinstance(experiments, dict) or experiments.get("fixed_count") != 5:
        raise RuModernBertCampaignError("Campaign must contain exactly five experiments")
    stages = experiments.get("ordered_stages")
    if not isinstance(stages, list) or [stage.get("key") for stage in stages] != EXPECTED_STAGE_KEYS:
        raise RuModernBertCampaignError("Campaign experiment order differs")
    if [stage.get("epochs") for stage in stages] != [1, 1, 1, 2, 3]:
        raise RuModernBertCampaignError("Campaign epoch line differs")
    fixed_lrs = [stages[index].get("learning_rate") for index in range(3)]
    if fixed_lrs != [8e-5, 4e-5, 1.6e-4]:
        raise RuModernBertCampaignError("Campaign LR screen differs")
    if [stages[index].get("learning_rate") for index in (3, 4)] != [
        "selected_from_first_three",
        "selected_from_first_three",
    ]:
        raise RuModernBertCampaignError("Epoch extensions must use the selected LR")
    training = plan.get("training")
    if not isinstance(training, dict) or training.get("gpu") != "one_H100_80GB":
        raise RuModernBertCampaignError("Campaign must remain bound to one H100 80GB")
    if training.get("effective_batch") != 192 or training.get("amp_dtype") != "bfloat16":
        raise RuModernBertCampaignError("Campaign H100 geometry differs")
    selection = plan.get("selection")
    if not isinstance(selection, dict) or selection.get("ood_is_evaluated") is not False:
        raise RuModernBertCampaignError("OOD must remain disabled")
    return plan


def validate_execution_sources() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in EXPECTED_EXECUTION_SHA256.items():
        path = ROOT / relative
        require_regular_file(path, f"execution source {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuModernBertCampaignError(
                f"Execution source hash differs for {relative}: {actual} != {expected}"
            )
        observed[relative] = actual
    return observed


def validate_hashed_files(directory: Path, expected: Mapping[str, str], label: str) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, expected_hash in expected.items():
        path = directory / name
        require_regular_file(path, f"{label} {name}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuModernBertCampaignError(
                f"{label} hash differs for {name}: {actual_hash} != {expected_hash}"
            )
        observed[name] = actual_hash
    return observed


def validate_checkpoint(plan: Mapping[str, Any], checkpoint_dir: Path) -> dict[str, Any]:
    checkpoint = plan["checkpoint"]
    hashes = validate_hashed_files(checkpoint_dir, checkpoint["files"], "checkpoint")
    config = read_json(checkpoint_dir / "config.json")
    if config.get("architectures") != [checkpoint["architecture"]]:
        raise RuModernBertCampaignError("Checkpoint architecture differs")
    if config.get("num_hidden_layers") != 22 or config.get("hidden_size") != 768:
        raise RuModernBertCampaignError("Checkpoint encoder shape differs")
    if config.get("id2label") != {"0": "MATCH_SCORE"}:
        raise RuModernBertCampaignError("Checkpoint classification head contract differs")
    return {
        "path": str(checkpoint_dir),
        "architecture": checkpoint["architecture"],
        "parameters": checkpoint["parameters"],
        "files": hashes,
    }


def validate_source_files(plan: Mapping[str, Any], human_dir: Path) -> dict[str, str]:
    return validate_hashed_files(human_dir, plan["source_data"]["files"], "source data")


def unordered_pair_keys(frame: pd.DataFrame) -> pd.MultiIndex:
    left = frame["id1"].to_numpy(dtype=np.int64)
    right = frame["id2"].to_numpy(dtype=np.int64)
    return pd.MultiIndex.from_arrays([np.minimum(left, right), np.maximum(left, right)])


def item_ids(frame: pd.DataFrame) -> set[int]:
    return set(frame["id1"].astype(np.int64)) | set(frame["id2"].astype(np.int64))


def validate_pair_frame(frame: pd.DataFrame, *, name: str, expected_rows: int) -> None:
    required = {"id1", "id2", "target"}
    if not required.issubset(frame.columns):
        raise RuModernBertCampaignError(f"{name} pair columns differ")
    if len(frame) != expected_rows:
        raise RuModernBertCampaignError(f"{name} rows differ: {len(frame)} != {expected_rows}")
    if frame[list(required)].isnull().any().any():
        raise RuModernBertCampaignError(f"{name} contains null pair values")
    if (frame["id1"] == frame["id2"]).any():
        raise RuModernBertCampaignError(f"{name} contains self-pairs")
    if not frame["target"].isin([0.0, 1.0]).all():
        raise RuModernBertCampaignError(f"{name} targets must be binary")
    if unordered_pair_keys(frame).duplicated().any():
        raise RuModernBertCampaignError(f"{name} contains duplicate unordered pairs")


def prepared_manifest_payload(
    *,
    plan: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    output_dir: Path,
    output_hashes: Mapping[str, str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "campaign": plan["campaign"],
        "source_hashes": dict(source_hashes),
        "output_hashes": dict(output_hashes),
        "train": {
            "rows": 347_840,
            "positives": 89_291,
            "categories": 20,
            "source_counts": {
                "human_train": 306_669,
                "human_former_ood": 41_171,
            },
            "order": ["human_train", "human_former_ood"],
        },
        "validation": {
            "iid": {"rows": 12_000, "categories": 18},
            "hard": {"rows": 5_814, "categories": 18},
            "ood": {"evaluated": False, "metric_sentinel": -1},
        },
        "item_overlap": {"train_iid": 0, "train_hard": 0, "iid_hard": 0},
        "output_dir": str(output_dir),
    }
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    return payload


def validate_prepared_data(
    prepared_dir: Path, plan: Mapping[str, Any], source_hashes: Mapping[str, str]
) -> dict[str, Any]:
    manifest_path = prepared_dir / PREPARED_MANIFEST
    require_regular_file(manifest_path, "prepared manifest")
    manifest = read_json(manifest_path)
    stored_hash = manifest.pop("manifest_payload_sha256", None)
    computed_hash = canonical_sha256(manifest)
    manifest["manifest_payload_sha256"] = stored_hash
    if stored_hash != computed_hash:
        raise RuModernBertCampaignError("Prepared manifest payload hash differs")
    if manifest.get("campaign") != plan["campaign"] or manifest.get("source_hashes") != dict(source_hashes):
        raise RuModernBertCampaignError("Prepared manifest source authority differs")
    expected_outputs = {name: sha256_file(prepared_dir / name) for name in (
        "items.parquet",
        "train_pairs.parquet",
        "iid_validation_pairs.parquet",
        "hard_validation_pairs.parquet",
    )}
    if manifest.get("output_hashes") != expected_outputs:
        raise RuModernBertCampaignError("Prepared output hashes differ")
    if (prepared_dir / "ood_validation_pairs.parquet").exists():
        raise RuModernBertCampaignError("Prepared directory must not expose an OOD validation split")
    return manifest


def prepare_data(plan: Mapping[str, Any], human_dir: Path, prepared_dir: Path) -> dict[str, Any]:
    source_hashes = validate_source_files(plan, human_dir)
    if prepared_dir.exists():
        return validate_prepared_data(prepared_dir, plan, source_hashes)
    expected = plan["source_data"]["expected"]
    items = pd.read_parquet(human_dir / "items.parquet")
    train = pd.read_parquet(human_dir / "train_pairs.parquet")
    former_ood = pd.read_parquet(human_dir / "ood_validation_pairs.parquet")
    iid = pd.read_parquet(human_dir / "iid_validation_pairs.parquet")
    hard = pd.read_parquet(human_dir / "hard_validation_pairs.parquet")
    if len(items) != expected["items"] or set(items.columns) != {"id", "name", "category", "product_text"}:
        raise RuModernBertCampaignError("Source item table differs")
    if items["id"].duplicated().any() or items[list(items.columns)].isnull().any().any():
        raise RuModernBertCampaignError("Source item table contains duplicate IDs or nulls")
    validate_pair_frame(train, name="human train", expected_rows=expected["human_train_pairs"])
    validate_pair_frame(former_ood, name="former OOD", expected_rows=expected["former_ood_pairs"])
    validate_pair_frame(iid, name="IID", expected_rows=expected["iid_pairs"])
    validate_pair_frame(hard, name="hard", expected_rows=expected["hard_pairs"])
    train = train.copy()
    former_ood = former_ood.copy()
    train["label_source"] = "human_train"
    former_ood["label_source"] = "human_former_ood"
    combined = pd.concat([train, former_ood], ignore_index=True)
    validate_pair_frame(combined, name="combined train", expected_rows=expected["combined_train_pairs"])
    if int(combined["target"].sum()) != expected["combined_train_positives"]:
        raise RuModernBertCampaignError("Combined train positive count differs")
    categories = items.set_index("id")["category"]
    all_pair_ids = item_ids(combined) | item_ids(iid) | item_ids(hard)
    if not all_pair_ids.issubset(set(items["id"].astype(np.int64))):
        raise RuModernBertCampaignError("A pair references a missing item")
    def pair_categories(frame: pd.DataFrame, label: str) -> set[str]:
        left = frame["id1"].map(categories)
        right = frame["id2"].map(categories)
        if not left.equals(right):
            raise RuModernBertCampaignError(f"{label} contains cross-category pairs")
        return set(left.astype(str))
    if len(pair_categories(combined, "combined train")) != expected["combined_train_categories"]:
        raise RuModernBertCampaignError("Combined train category count differs")
    if len(pair_categories(iid, "IID")) != expected["iid_categories"]:
        raise RuModernBertCampaignError("IID category count differs")
    if len(pair_categories(hard, "hard")) != expected["hard_categories"]:
        raise RuModernBertCampaignError("Hard category count differs")
    train_ids, iid_ids, hard_ids = item_ids(combined), item_ids(iid), item_ids(hard)
    overlaps = {
        "train_iid": len(train_ids & iid_ids),
        "train_hard": len(train_ids & hard_ids),
        "iid_hard": len(iid_ids & hard_ids),
    }
    if overlaps != {"train_iid": 0, "train_hard": 0, "iid_hard": 0}:
        raise RuModernBertCampaignError(f"Prepared split item leakage: {overlaps}")
    prepared_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=prepared_dir.name + ".pending-", dir=prepared_dir.parent))
    try:
        items.to_parquet(temporary / "items.parquet", index=False, compression="zstd")
        combined.to_parquet(temporary / "train_pairs.parquet", index=False, compression="zstd")
        shutil.copy2(human_dir / "iid_validation_pairs.parquet", temporary / "iid_validation_pairs.parquet")
        shutil.copy2(human_dir / "hard_validation_pairs.parquet", temporary / "hard_validation_pairs.parquet")
        output_hashes = {name: sha256_file(temporary / name) for name in (
            "items.parquet",
            "train_pairs.parquet",
            "iid_validation_pairs.parquet",
            "hard_validation_pairs.parquet",
        )}
        manifest = prepared_manifest_payload(
            plan=plan,
            source_hashes=source_hashes,
            output_dir=prepared_dir,
            output_hashes=output_hashes,
        )
        atomic_write_json(temporary / PREPARED_MANIFEST, manifest)
        os.replace(temporary, prepared_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return validate_prepared_data(prepared_dir, plan, source_hashes)


def resolved_config(
    plan: Mapping[str, Any], checkpoint_dir: Path, *, epochs: int, learning_rate: float
) -> dict[str, Any]:
    training = plan["training"]
    values = {
        key: training[key]
        for key in TRAINER_CONFIG_KEYS
        if key in training
    }
    values.update(
        {
            "model": str(checkpoint_dir),
            "model_load_kwargs": {},
            "epochs": epochs,
            "learning_rate": learning_rate,
        }
    )
    if set(values) != TRAINER_CONFIG_KEYS:
        missing = TRAINER_CONFIG_KEYS - set(values)
        extra = set(values) - TRAINER_CONFIG_KEYS
        raise RuModernBertCampaignError(
            f"Resolved trainer config fields differ; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return values


def write_resolved_config(path: Path, payload: Mapping[str, Any]) -> None:
    write_once_json(path, payload)


def stage_specifications(plan: Mapping[str, Any], selected_lr: float | None = None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for stage in plan["experiments"]["ordered_stages"]:
        value = stage["learning_rate"]
        if value == "selected_from_first_three":
            value = selected_lr
        result.append({**stage, "learning_rate": value})
    return result


def metric_from_report(report: Mapping[str, Any], split: str) -> float:
    splits = report.get("validation_splits")
    if not isinstance(splits, dict) or set(splits) != {"iid", "hard"}:
        raise RuModernBertCampaignError("Training report validation split set differs")
    value = splits[split].get("macro_average_precision")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuModernBertCampaignError(f"Invalid {split} macro AP")
    return float(value)


def compare_prediction_keys(predictions: pd.DataFrame, truth: pd.DataFrame, split: str) -> None:
    required = {"id1", "id2", "target", "category_1", "score"}
    if not required.issubset(predictions.columns):
        raise RuModernBertCampaignError(f"{split} prediction columns differ")
    if len(predictions) != len(truth):
        raise RuModernBertCampaignError(f"{split} prediction row count differs")
    for column in ("id1", "id2"):
        if not np.array_equal(predictions[column].to_numpy(), truth[column].to_numpy()):
            raise RuModernBertCampaignError(f"{split} prediction {column} order differs")
    if not np.array_equal(
        predictions["target"].to_numpy(dtype=np.float64),
        truth["target"].to_numpy(dtype=np.float64),
    ):
        raise RuModernBertCampaignError(f"{split} prediction targets differ")
    scores = predictions["score"].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise RuModernBertCampaignError(f"{split} predictions contain invalid scores")


def validate_run_output(
    *,
    key: str,
    run_dir: Path,
    config_path: Path,
    prepared_dir: Path,
    checkpoint_model_sha256: str,
    expected_epochs: int,
    expected_lr: float,
) -> dict[str, Any]:
    completion_path = run_dir / COMPLETION_FILE
    report_path = run_dir / "training_report.json"
    training_config_path = run_dir / "training_config.json"
    model_path = run_dir / "model.safetensors"
    iid_path = run_dir / "iid_validation_predictions.parquet"
    hard_path = run_dir / "hard_validation_predictions.parquet"
    for path, label in (
        (report_path, "training report"),
        (training_config_path, "training config"),
        (model_path, "trained model"),
        (iid_path, "IID predictions"),
        (hard_path, "hard predictions"),
    ):
        require_regular_file(path, label)
    if (run_dir / "ood_validation_predictions.parquet").exists():
        raise RuModernBertCampaignError("OOD predictions are forbidden")
    if training_config_path.read_bytes() != config_path.read_bytes():
        raise RuModernBertCampaignError("Saved training config differs from resolved config")
    config = read_json(config_path)
    if config["epochs"] != expected_epochs or float(config["learning_rate"]) != expected_lr:
        raise RuModernBertCampaignError("Resolved epoch/LR coordinate differs")
    report = read_json(report_path)
    if report.get("primary_validation_split") != "iid":
        raise RuModernBertCampaignError("IID must be the primary validation split")
    if report.get("original_training_examples") != 347_840:
        raise RuModernBertCampaignError("Original training row count differs")
    if report.get("training_examples") != 347_840 * expected_epochs:
        raise RuModernBertCampaignError("Training example coverage differs")
    if report.get("training_sampling") != "none" or report.get("training_loss_weighting") != "none":
        raise RuModernBertCampaignError("Training sampling/loss weighting differs")
    if report.get("training_source_counts") != {
        "human_train": 306_669,
        "human_former_ood": 41_171,
    }:
        raise RuModernBertCampaignError("Training source counts differ")
    if report.get("training_unique_coverage_per_epoch") != 1.0:
        raise RuModernBertCampaignError("Training coverage differs")
    if any(float(report.get(name, float("nan"))) != 1.0 for name in (
        "training_loss_weight_min",
        "training_loss_weight_median",
        "training_loss_weight_max",
    )):
        raise RuModernBertCampaignError("Plain BCE weights differ from one")
    loss_hook = report.get("loss_hook")
    if not isinstance(loss_hook, dict) or loss_hook.get("name") != LOSS_HOOK.stem:
        raise RuModernBertCampaignError("Finite BCE hook name differs")
    if loss_hook.get("sha256") != sha256_file(LOSS_HOOK):
        raise RuModernBertCampaignError("Finite BCE hook hash differs")
    args = report.get("args")
    if not isinstance(args, dict):
        raise RuModernBertCampaignError("Training report args are missing")
    expected_args = {
        "epochs": expected_epochs,
        "learning_rate": expected_lr,
        "max_length": 384,
        "sampling": "none",
        "loss_weighting": "none",
        "gradient_accumulation": 1,
        "batch_size": 192,
        "eval_batch_size": 512,
    }
    for name, expected in expected_args.items():
        if args.get(name) != expected:
            raise RuModernBertCampaignError(f"Training report arg differs at {name}")
    iid_truth = pd.read_parquet(prepared_dir / "iid_validation_pairs.parquet")
    hard_truth = pd.read_parquet(prepared_dir / "hard_validation_pairs.parquet")
    iid_predictions = pd.read_parquet(iid_path)
    hard_predictions = pd.read_parquet(hard_path)
    compare_prediction_keys(iid_predictions, iid_truth, "iid")
    compare_prediction_keys(hard_predictions, hard_truth, "hard")
    for split, predictions in (("iid", iid_predictions), ("hard", hard_predictions)):
        category_count = predictions["category_1"].astype(str).nunique()
        if category_count != 18:
            raise RuModernBertCampaignError(f"{split} category count differs")
        recomputed = float(
            np.mean(
                [
                    average_precision_score(group["target"], group["score"])
                    for _, group in predictions.groupby("category_1", sort=False)
                ]
            )
        )
        reported = metric_from_report(report, split)
        if not math.isclose(recomputed, reported, rel_tol=0.0, abs_tol=1e-12):
            raise RuModernBertCampaignError(
                f"{split} macro AP differs: {recomputed} != {reported}"
            )
    expected_completion = {
        "schema_version": 1,
        "campaign": "rumodernbert_3ep_sft_oodtrain_h100_v1",
        "status": "complete",
        "key": key,
        "epochs": expected_epochs,
        "learning_rate": expected_lr,
        "checkpoint_model_sha256": checkpoint_model_sha256,
        "config_sha256": sha256_file(config_path),
        "training_report_sha256": sha256_file(report_path),
        "trained_model_sha256": sha256_file(model_path),
        "iid_predictions_sha256": sha256_file(iid_path),
        "hard_predictions_sha256": sha256_file(hard_path),
        "iid_macro_average_precision": metric_from_report(report, "iid"),
        "hard_macro_average_precision": metric_from_report(report, "hard"),
        "ood_macro_average_precision": -1,
    }
    expected_completion["completion_payload_sha256"] = canonical_sha256(expected_completion)
    if completion_path.exists():
        if read_json(completion_path) != expected_completion:
            raise RuModernBertCampaignError(f"Completion receipt differs for {key}")
    else:
        atomic_write_json(completion_path, expected_completion)
    return expected_completion


def select_learning_rate(completions: Mapping[str, Mapping[str, Any]], margin: float) -> float:
    keys = ["e1_lr8e5", "e1_lr4e5", "e1_lr1p6e4"]
    if set(completions) != set(keys):
        raise RuModernBertCampaignError("LR selection requires exactly three e1 runs")
    values = {
        float(completions[key]["learning_rate"]): float(
            completions[key]["iid_macro_average_precision"]
        )
        for key in keys
    }
    best_score = max(values.values())
    anchor = 8e-5
    if best_score - values[anchor] <= margin:
        return anchor
    eligible = [lr for lr, score in values.items() if best_score - score <= margin]
    return min(eligible)


def select_epoch(completions: Mapping[str, Mapping[str, Any]], margin: float) -> int:
    values = {
        int(completion["epochs"]): float(completion["iid_macro_average_precision"])
        for completion in completions.values()
    }
    if set(values) != {1, 2, 3}:
        raise RuModernBertCampaignError("Epoch selection requires e1/e2/e3")
    best_score = max(values.values())
    return min(epoch for epoch, score in values.items() if best_score - score <= margin)


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
        raise RuModernBertCampaignError(
            f"Command failed with exit code {returncode}; inspect {log_path}"
        )


def training_command(
    *,
    config_path: Path,
    prepared_dir: Path,
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
        "iid=iid_validation_pairs.parquet",
        "--validation-split",
        "hard=hard_validation_pairs.parquet",
    ]


def run_one(
    *,
    key: str,
    epochs: int,
    learning_rate: float,
    plan: Mapping[str, Any],
    checkpoint_dir: Path,
    prepared_dir: Path,
    work_dir: Path,
    checkpoint_model_sha256: str,
    env: Mapping[str, str],
) -> dict[str, Any]:
    config_path = work_dir / "resolved_configs" / f"{key}.json"
    write_resolved_config(
        config_path,
        resolved_config(plan, checkpoint_dir, epochs=epochs, learning_rate=learning_rate),
    )
    run_dir = work_dir / "runs" / key
    completion_path = run_dir / COMPLETION_FILE
    if completion_path.exists():
        return validate_run_output(
            key=key,
            run_dir=run_dir,
            config_path=config_path,
            prepared_dir=prepared_dir,
            checkpoint_model_sha256=checkpoint_model_sha256,
            expected_epochs=epochs,
            expected_lr=learning_rate,
        )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuModernBertCampaignError(
            f"Partial run directory exists and exact resume is impossible: {run_dir}. "
            "Inspect/archive it before choosing a new work directory."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    command = training_command(
        config_path=config_path,
        prepared_dir=prepared_dir,
        output_dir=run_dir,
        token_cache_dir=work_dir / "token_cache",
    )
    run_command_logged(command, run_dir / "training.log", env)
    return validate_run_output(
        key=key,
        run_dir=run_dir,
        config_path=config_path,
        prepared_dir=prepared_dir,
        checkpoint_model_sha256=checkpoint_model_sha256,
        expected_epochs=epochs,
        expected_lr=learning_rate,
    )


def comparison(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    permutations: int,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    splits: dict[str, dict[str, Any]] = {}
    for index, split in enumerate(("iid", "hard")):
        splits[split] = compare_prediction_frames(
            pd.read_parquet(baseline_dir / f"{split}_validation_predictions.parquet"),
            pd.read_parquet(candidate_dir / f"{split}_validation_predictions.parquet"),
            permutations=permutations,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed + index,
        )
    return {"splits": splits}


def family_comparisons(
    *,
    baseline_key: str,
    candidate_keys: Sequence[str],
    work_dir: Path,
    permutations: int,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    results = {
        key: comparison(
            work_dir / "runs" / baseline_key,
            work_dir / "runs" / key,
            permutations=permutations,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed + 10 * index,
        )
        for index, key in enumerate(candidate_keys)
    }
    for split in ("iid", "hard"):
        adjusted = holm_adjust(
            {key: float(value["splits"][split]["p_value"]) for key, value in results.items()}
        )
        for key, p_value_holm in adjusted.items():
            results[key]["splits"][split]["p_value_holm"] = p_value_holm
    return results


def load_all_completions(
    work_dir: Path,
    *,
    prepared_dir: Path,
    checkpoint_model_sha256: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    fixed = [
        ("e1_lr8e5", 1, 8e-5),
        ("e1_lr4e5", 1, 4e-5),
        ("e1_lr1p6e4", 1, 1.6e-4),
    ]
    for key, epochs, learning_rate in fixed:
        path = work_dir / "runs" / key / COMPLETION_FILE
        require_regular_file(path, f"completion for {key}")
        result[key] = validate_run_output(
            key=key,
            run_dir=work_dir / "runs" / key,
            config_path=work_dir / "resolved_configs" / f"{key}.json",
            prepared_dir=prepared_dir,
            checkpoint_model_sha256=checkpoint_model_sha256,
            expected_epochs=epochs,
            expected_lr=learning_rate,
        )
    selected_lr = select_learning_rate(result, 0.002)
    for key, epochs in (("e2_selected_lr", 2), ("e3_selected_lr", 3)):
        path = work_dir / "runs" / key / COMPLETION_FILE
        require_regular_file(path, f"completion for {key}")
        result[key] = validate_run_output(
            key=key,
            run_dir=work_dir / "runs" / key,
            config_path=work_dir / "resolved_configs" / f"{key}.json",
            prepared_dir=prepared_dir,
            checkpoint_model_sha256=checkpoint_model_sha256,
            expected_epochs=epochs,
            expected_lr=selected_lr,
        )
    return result


def summarize(
    *,
    plan: Mapping[str, Any],
    work_dir: Path,
    permutations: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    completions = load_all_completions(
        work_dir,
        prepared_dir=work_dir / "prepared",
        checkpoint_model_sha256=plan["checkpoint"]["files"]["model.safetensors"],
    )
    margin = float(plan["selection"]["practical_tie_margin"])
    selected_lr = select_learning_rate(
        {key: completions[key] for key in EXPECTED_STAGE_KEYS[:3]}, margin
    )
    selected_e1_key = next(
        key for key in EXPECTED_STAGE_KEYS[:3]
        if float(completions[key]["learning_rate"]) == selected_lr
    )
    selected_epoch = select_epoch(
        {
            selected_e1_key: completions[selected_e1_key],
            "e2_selected_lr": completions["e2_selected_lr"],
            "e3_selected_lr": completions["e3_selected_lr"],
        },
        margin,
    )
    recommended_key = {
        1: selected_e1_key,
        2: "e2_selected_lr",
        3: "e3_selected_lr",
    }[selected_epoch]
    lr_family = family_comparisons(
        baseline_key="e1_lr8e5",
        candidate_keys=["e1_lr4e5", "e1_lr1p6e4"],
        work_dir=work_dir,
        permutations=permutations,
        bootstrap_resamples=bootstrap_resamples,
        seed=42,
    )
    epoch_family = family_comparisons(
        baseline_key=selected_e1_key,
        candidate_keys=["e2_selected_lr", "e3_selected_lr"],
        work_dir=work_dir,
        permutations=permutations,
        bootstrap_resamples=bootstrap_resamples,
        seed=142,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "campaign": plan["campaign"],
        "status": "complete",
        "completed_runs": 5,
        "primary_metric": "iid_macro_average_precision",
        "hard_is_diagnostic_only": True,
        "ood": {"evaluated": False, "metric_sentinel": -1},
        "practical_tie_margin": margin,
        "selected_learning_rate": selected_lr,
        "selected_epoch": selected_epoch,
        "recommended_key": recommended_key,
        "recommended_output_dir": str(work_dir / "runs" / recommended_key),
        "runs": [completions[key] for key in EXPECTED_STAGE_KEYS],
        "comparisons": {
            "lr_family_vs_e1_lr8e5": lr_family,
            "epoch_family_vs_selected_e1": epoch_family,
        },
        "next_decision": (
            "choose_recommended_recipe_or_design_a_new_bounded_axis_only_if_the_five_run_curve_remains_open"
        ),
    }
    payload["summary_payload_sha256"] = canonical_sha256(payload)
    atomic_write_json(work_dir / SUMMARY_FILE, payload)
    return payload


def state_payload(
    *, status: str, plan: Mapping[str, Any], work_dir: Path, completed: Sequence[str], current: str | None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "campaign": plan["campaign"],
        "status": status,
        "completed_keys": list(completed),
        "current_key": current,
        "updated_at_utc": utc_now(),
        "work_dir": str(work_dir),
    }
    payload["state_payload_sha256"] = canonical_sha256(payload)
    return payload


@contextmanager
def exclusive_lock(work_dir: Path):
    work_dir.mkdir(parents=True, exist_ok=True)
    lock_path = work_dir / ".campaign.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuModernBertCampaignError("Another campaign controller is active") from error
        yield


def run_campaign(
    *,
    plan: Mapping[str, Any],
    checkpoint_dir: Path,
    human_dir: Path,
    work_dir: Path,
    permutations: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    with exclusive_lock(work_dir):
        checkpoint = validate_checkpoint(plan, checkpoint_dir)
        prepared_dir = work_dir / "prepared"
        prepare_data(plan, human_dir, prepared_dir)
        env = dict(os.environ)
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        anchor_config = work_dir / "resolved_configs" / "e1_lr8e5.json"
        write_resolved_config(
            anchor_config,
            resolved_config(plan, checkpoint_dir, epochs=1, learning_rate=8e-5),
        )
        preflight_report = work_dir / "h100_preflight.json"
        if not preflight_report.exists():
            run_command_logged(
                [
                    sys.executable,
                    str(TRAINER),
                    "--preflight-only",
                    "--config",
                    str(anchor_config),
                    "--preflight-report",
                    str(preflight_report),
                ],
                work_dir / "h100_preflight.log",
                env,
            )
        preflight = read_json(preflight_report)
        if preflight.get("status") != "passed" or preflight.get("parameters") != 149_605_633:
            raise RuModernBertCampaignError("H100 memory preflight receipt differs")
        completed: dict[str, dict[str, Any]] = {}
        first_three = [("e1_lr8e5", 8e-5), ("e1_lr4e5", 4e-5), ("e1_lr1p6e4", 1.6e-4)]
        for key, learning_rate in first_three:
            atomic_write_json(
                work_dir / STATE_FILE,
                state_payload(
                    status="running", plan=plan, work_dir=work_dir,
                    completed=list(completed), current=key,
                ),
            )
            completed[key] = run_one(
                key=key,
                epochs=1,
                learning_rate=learning_rate,
                plan=plan,
                checkpoint_dir=checkpoint_dir,
                prepared_dir=prepared_dir,
                work_dir=work_dir,
                checkpoint_model_sha256=checkpoint["files"]["model.safetensors"],
                env=env,
            )
        selected_lr = select_learning_rate(completed, float(plan["selection"]["practical_tie_margin"]))
        for key, epochs in (("e2_selected_lr", 2), ("e3_selected_lr", 3)):
            atomic_write_json(
                work_dir / STATE_FILE,
                state_payload(
                    status="running", plan=plan, work_dir=work_dir,
                    completed=list(completed), current=key,
                ),
            )
            completed[key] = run_one(
                key=key,
                epochs=epochs,
                learning_rate=selected_lr,
                plan=plan,
                checkpoint_dir=checkpoint_dir,
                prepared_dir=prepared_dir,
                work_dir=work_dir,
                checkpoint_model_sha256=checkpoint["files"]["model.safetensors"],
                env=env,
            )
        result = summarize(
            plan=plan,
            work_dir=work_dir,
            permutations=permutations,
            bootstrap_resamples=bootstrap_resamples,
        )
        atomic_write_json(
            work_dir / STATE_FILE,
            state_payload(
                status="complete", plan=plan, work_dir=work_dir,
                completed=EXPECTED_STAGE_KEYS, current=None,
            ),
        )
        return result


def plan_payload(
    plan: Mapping[str, Any], checkpoint_dir: Path, human_dir: Path, work_dir: Path
) -> dict[str, Any]:
    return {
        "mode": "plan_only",
        "campaign": plan["campaign"],
        "checkpoint_dir": str(checkpoint_dir),
        "source_human_dir": str(human_dir),
        "work_dir": str(work_dir),
        "gpu": "1 x H100 80GB",
        "training_runs": 5,
        "ordered_stages": stage_specifications(plan),
        "data": {
            "train": "306669 human train + 41171 former OOD = 347840",
            "validation": ["iid", "hard"],
            "ood_metric": -1,
        },
        "fresh_checkpoint_per_run": True,
        "kaggle_used": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--source-human-dir", type=Path)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--summarize", action="store_true")
    parser.add_argument("--permutations", type=int)
    parser.add_argument("--bootstrap-resamples", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan_path = args.plan.expanduser().resolve()
    plan = load_plan(plan_path)
    validate_execution_sources()
    checkpoint_dir = (
        args.checkpoint_dir.expanduser().resolve()
        if args.checkpoint_dir is not None
        else (ROOT / plan["checkpoint"]["path"]).resolve()
    )
    human_dir = (
        args.source_human_dir.expanduser().resolve()
        if args.source_human_dir is not None
        else (ROOT / plan["source_data"]["human_dir"]).resolve()
    )
    work_dir = args.work_dir.expanduser().resolve()
    permutations = args.permutations or int(plan["selection"]["paired_permutations"])
    bootstrap_resamples = args.bootstrap_resamples or int(
        plan["selection"]["paired_bootstrap_resamples"]
    )
    if permutations < 1 or bootstrap_resamples < 1:
        raise SystemExit("Permutation and bootstrap counts must be positive")
    if not (args.dry_run or args.prepare_only or args.run or args.summarize):
        print(json.dumps(plan_payload(plan, checkpoint_dir, human_dir, work_dir), ensure_ascii=False, indent=2))
        return 0
    validate_checkpoint(plan, checkpoint_dir)
    source_hashes = validate_source_files(plan, human_dir)
    if args.dry_run:
        payload = plan_payload(plan, checkpoint_dir, human_dir, work_dir)
        payload["mode"] = "dry_run"
        payload["checkpoint_validated"] = True
        payload["source_hashes"] = source_hashes
        payload["training_started"] = False
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.prepare_only:
        manifest = prepare_data(plan, human_dir, work_dir / "prepared")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if args.summarize:
        result = summarize(
            plan=plan,
            work_dir=work_dir,
            permutations=permutations,
            bootstrap_resamples=bootstrap_resamples,
        )
    else:
        result = run_campaign(
            plan=plan,
            checkpoint_dir=checkpoint_dir,
            human_dir=human_dir,
            work_dir=work_dir,
            permutations=permutations,
            bootstrap_resamples=bootstrap_resamples,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
