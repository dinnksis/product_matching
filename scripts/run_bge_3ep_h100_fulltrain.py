#!/usr/bin/env python3
"""Run one metric-free BGE H100 export on all 365,654 human pairs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_bge_3ep_h100 as base


DEFAULT_CONFIG = ROOT / "configs" / "bge_3ep_sft_oodtrain_h100_v1.json"
DEFAULT_MODEL_DIR = ROOT / "model" / "pretrain_bge_2ep"
DEFAULT_HUMAN_DIR = ROOT / "prepared" / "validation_splits_v1" / "human"
DEFAULT_WORK_DIR = ROOT / "artifacts" / "bge_3ep_fulltrain_h100_v1"
TRAINER = ROOT / "scripts" / "train_bge_3ep_h100_fulltrain.py"
LOSS_HOOK = ROOT / "scripts" / "bge_h100_fulltrain_finite_bce.py"

EXPECTED_EXECUTION_FILES = {
    "scripts/run_bge_3ep_h100.py": "c93d8e72df3738fc2b1b85db0eace65894d48c019c1877889393b15984b1d33c",
    "scripts/train_bge_3ep_h100_fulltrain.py": "2651864b2c0e7ae693408540837e726cd13ecd30351fac6b1e6673006a4f0bfb",
    "scripts/bge_h100_fulltrain_finite_bce.py": "5a2350ddf544e1e2d56cc8aa5255221f936854393a16d8ec4c11e950be2d6251",
}
EXPECTED_ROWS = 365_654
EXPECTED_POSITIVES = 93_890
EXPECTED_COVERED_ITEMS = 711_304
EXPECTED_SOURCE_COUNTS = {
    "human_train": 306_669,
    "human_iid": 12_000,
    "human_hard": 5_814,
    "human_former_ood": 41_171,
}
EXPECTED_OUTPUT_FILES = {
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "training_config.json",
    "training_report.json",
}


class BgeH100FulltrainError(RuntimeError):
    """Raised if the all-human export differs from its one-shot contract."""


def validate_own_execution_sources() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected_hash in EXPECTED_EXECUTION_FILES.items():
        path = ROOT / relative
        base.require_regular_file(path, f"full-train execution source {relative}")
        actual_hash = base.sha256_file(path)
        if actual_hash != expected_hash:
            raise BgeH100FulltrainError(
                f"Full-train execution hash differs for {relative}: "
                f"{actual_hash} != {expected_hash}"
            )
        observed[relative] = actual_hash
    return observed


def full_train_frame(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    train = pd.concat(
        [
            frames["train_pairs.parquet"].assign(label_source="human_train"),
            frames["iid_validation_pairs.parquet"].assign(label_source="human_iid"),
            frames["hard_validation_pairs.parquet"].assign(label_source="human_hard"),
            frames["ood_validation_pairs.parquet"].assign(
                label_source="human_former_ood"
            ),
        ],
        ignore_index=True,
    )
    if len(train) != EXPECTED_ROWS or int(train["target"].sum()) != EXPECTED_POSITIVES:
        raise BgeH100FulltrainError("All-human train count/positives differ")
    source_counts = {
        str(key): int(value)
        for key, value in train["label_source"].value_counts().items()
    }
    if source_counts != EXPECTED_SOURCE_COUNTS:
        raise BgeH100FulltrainError(f"All-human source counts differ: {source_counts}")
    unordered = base.unordered_pair_index(train)
    if unordered.has_duplicates:
        raise BgeH100FulltrainError("All-human train contains unordered duplicates")
    covered_items = set(train["id1"]) | set(train["id2"])
    if len(covered_items) != EXPECTED_COVERED_ITEMS:
        raise BgeH100FulltrainError(
            "All-human train item coverage differs: "
            f"{len(covered_items)} != {EXPECTED_COVERED_ITEMS}"
        )
    return train


def prepare_full_data(human_dir: Path, prepared_dir: Path) -> dict[str, Any]:
    frames, source_report = base.validate_data(human_dir)
    train = full_train_frame(frames)
    manifest_path = prepared_dir / "prepared_data_manifest.json"
    train_path = prepared_dir / "train_pairs.parquet"
    items_path = prepared_dir / "items.parquet"
    if manifest_path.exists():
        manifest = base.read_json(manifest_path)
        base.require_regular_file(train_path, "prepared all-human train")
        base.require_regular_file(items_path, "prepared item catalog")
        if base.sha256_file(train_path) != manifest.get("full_train_sha256"):
            raise BgeH100FulltrainError("Prepared all-human train hash differs")
        if base.sha256_file(items_path) != base.EXPECTED_DATA_FILES["items.parquet"]:
            raise BgeH100FulltrainError("Prepared all-human item catalog hash differs")
        if manifest.get("source_report") != source_report:
            raise BgeH100FulltrainError("Prepared all-human source report differs")
        return manifest
    if prepared_dir.exists() and any(prepared_dir.iterdir()):
        raise BgeH100FulltrainError(f"Partial prepared directory exists: {prepared_dir}")
    prepared_dir.mkdir(parents=True, exist_ok=True)
    temporary = train_path.with_name(train_path.name + ".pending")
    train.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, train_path)
    base.atomic_copy_or_link(human_dir / "items.parquet", items_path)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": base.utc_now(),
        "purpose": "final_deployment_export_no_holdout",
        "quality_evaluation": False,
        "source_report": source_report,
        "train_order": [
            "train_pairs.parquet",
            "iid_validation_pairs.parquet",
            "hard_validation_pairs.parquet",
            "ood_validation_pairs.parquet",
        ],
        "label_source_counts": EXPECTED_SOURCE_COUNTS,
        "full_train_rows": len(train),
        "full_train_positives": int(train["target"].sum()),
        "full_train_sha256": base.sha256_file(train_path),
        "full_train_bytes": train_path.stat().st_size,
        "items_sha256": base.sha256_file(items_path),
        "held_out_splits": [],
        "iid_metric": -1,
        "hard_metric": -1,
        "ood_metric": -1,
    }
    manifest["payload_sha256"] = base.canonical_sha256(manifest)
    base.atomic_write_json(manifest_path, manifest)
    return manifest


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
    ]


def validate_training_output(
    output_dir: Path,
    config_path: Path,
    initial_model_sha256: str,
) -> dict[str, Any]:
    if not output_dir.is_dir():
        raise BgeH100FulltrainError(f"Full-train output is missing: {output_dir}")
    actual_files = {
        path.name for path in output_dir.iterdir() if path.is_file() or path.is_symlink()
    }
    if actual_files != EXPECTED_OUTPUT_FILES:
        raise BgeH100FulltrainError(
            f"Full-train output set differs: expected={sorted(EXPECTED_OUTPUT_FILES)}, "
            f"actual={sorted(actual_files)}"
        )
    for path in output_dir.iterdir():
        base.require_regular_file(path, f"full-train output {path.name}")
    report = base.read_json(output_dir / "training_report.json")
    expected_scalars = {
        "status": "complete",
        "purpose": "final_deployment_export",
        "quality_evaluation": False,
        "validation_splits": [],
        "validation_predictions_written": False,
        "original_training_examples": EXPECTED_ROWS,
        "training_examples": EXPECTED_ROWS * 3,
        "epochs": 3,
        "steps_per_epoch": 5_714,
        "updates_per_epoch": 1_905,
        "planned_optimizer_updates": 5_715,
        "optimizer_step_attempts": 5_715,
        "optimizer_steps_succeeded": 5_715,
        "amp_overflow_skips": 0,
        "warmup_updates": 285,
        "gradient_accumulation_normalization": "sample_exact_group_mean_v1",
    }
    for key, value in expected_scalars.items():
        if report.get(key) != value:
            raise BgeH100FulltrainError(
                f"Full-train report differs at {key}: {report.get(key)!r} != {value!r}"
            )
    if report.get("training_source_counts") != EXPECTED_SOURCE_COUNTS:
        raise BgeH100FulltrainError("Full-train report source counts differ")
    geometry = report.get("epoch_batch_geometry")
    if not isinstance(geometry, list) or len(geometry) != 3:
        raise BgeH100FulltrainError("Full-train epoch geometry differs")
    for expected_epoch, entry in enumerate(geometry, start=1):
        if not isinstance(entry, dict) or entry.get("epoch") != expected_epoch:
            raise BgeH100FulltrainError("Full-train epoch geometry ordering differs")
        if entry.get("partial_batch_size") != 22:
            raise BgeH100FulltrainError("Full-train partial batch size differs")
        denominator = entry.get("partial_group_examples")
        if denominator not in {86, 150}:
            raise BgeH100FulltrainError(
                f"Full-train partial group denominator differs: {denominator}"
            )
    args = report.get("args")
    if not isinstance(args, dict) or args.get("validation_split") != []:
        raise BgeH100FulltrainError("Full-train report unexpectedly contains validation")
    if (output_dir / "training_config.json").read_bytes() != config_path.read_bytes():
        raise BgeH100FulltrainError("Saved full-train config differs")
    final_model_sha256 = base.sha256_file(output_dir / "model.safetensors")
    if final_model_sha256 == initial_model_sha256:
        raise BgeH100FulltrainError("Full-train model equals its initial checkpoint")
    for path in output_dir.iterdir():
        if "prediction" in path.name.lower():
            raise BgeH100FulltrainError("Full-train output contains predictions")
    return {
        "training_report_sha256": base.sha256_file(
            output_dir / "training_report.json"
        ),
        "final_model_sha256": final_model_sha256,
        "quality_metrics": {"iid": -1, "hard": -1, "ood": -1},
        "validation_predictions_written": False,
    }


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
    base_config = base.load_base_config(config_path)
    base_execution = base.validate_execution_sources()
    own_execution = validate_own_execution_sources()
    checkpoint_hashes = base.validate_hashed_files(
        model_dir, base.EXPECTED_CHECKPOINT_FILES, "BGE checkpoint"
    )
    frames, data_report = base.validate_data(human_dir)
    full_train = full_train_frame(frames)
    plan = {
        "experiment": "bge_3ep_fulltrain_h100_v1",
        "purpose": "final_deployment_export_no_holdout",
        "quality_evaluation": False,
        "gpu": "one_H100_80GB",
        "fresh_start": True,
        "epochs": 3,
        "learning_rate": 2e-5,
        "effective_batch": 192,
        "amp_dtype": "bfloat16",
        "train_rows": len(full_train),
        "train_positives": int(full_train["target"].sum()),
        "source_counts": EXPECTED_SOURCE_COUNTS,
        "validation": {"iid": -1, "hard": -1, "ood": -1},
        "model_dir": str(model_dir),
        "human_dir": str(human_dir),
        "work_dir": str(work_dir),
        "checkpoint_sha256": checkpoint_hashes,
        "data": data_report,
        "execution_sha256": {**base_execution, **own_execution},
    }
    if not (args.dry_run or args.prepare_only or args.run):
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        print("Plan only. Pass --dry-run, --prepare-only or --run.")
        return 0
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        print("Full-train dry-run passed; no files were written and no GPU was used.")
        return 0
    completion_path = work_dir / "run_completed.json"
    if completion_path.exists():
        completion = base.validate_completion(work_dir)
        print(json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True))
        print("Exact completed full-train run already exists; nothing to do.")
        return 0
    prepared_dir = work_dir / "prepared"
    prepared_manifest = prepare_full_data(human_dir, prepared_dir)
    if args.prepare_only:
        print(json.dumps(prepared_manifest, ensure_ascii=False, indent=2, sort_keys=True))
        print("All-human data is ready; no GPU was used.")
        return 0
    run_dir = work_dir / "run"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise BgeH100FulltrainError(
            f"Partial full-train output exists and exact resume is unavailable: {run_dir}"
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = work_dir / "resolved_config.json"
    base.atomic_write_json(
        resolved_config_path, base.resolved_config(base_config, model_dir)
    )
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
        preflight = base.read_json(preflight_path)
        if preflight.get("status") != "passed" or preflight.get("parameters") != 567_755_777:
            raise BgeH100FulltrainError("Existing full-train preflight differs")
    else:
        base.run_command_logged(
            [
                sys.executable,
                str(ROOT / "scripts" / "train_bge_3ep_h100.py"),
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
    base.run_command_logged(
        training_command(
            config_path=resolved_config_path,
            prepared_dir=prepared_dir,
            output_dir=run_dir,
            token_cache_dir=work_dir / "token_cache",
        ),
        work_dir / "training.log",
        env,
    )
    output_summary = validate_training_output(
        run_dir,
        resolved_config_path,
        base.EXPECTED_CHECKPOINT_FILES["model.safetensors"],
    )
    base.deployment_smoke(run_dir, work_dir / "deployment_smoke.json")
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
        str(path.relative_to(work_dir)): base.sha256_file(path)
        for path in artifact_paths
        if path.is_file()
    }
    completion: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": base.utc_now(),
        "experiment": "bge_3ep_fulltrain_h100_v1",
        "purpose": "final_deployment_export_no_holdout",
        "quality_evaluation": False,
        "elapsed_seconds_including_training": time.perf_counter() - started,
        "fresh_start_from_pretrain": True,
        "epochs": 3,
        "learning_rate": 2e-5,
        "checkpoint_model_sha256": base.EXPECTED_CHECKPOINT_FILES[
            "model.safetensors"
        ],
        "prepared_manifest_sha256": base.sha256_file(
            prepared_dir / "prepared_data_manifest.json"
        ),
        "output": output_summary,
        "files": files,
    }
    completion["payload_sha256"] = base.canonical_sha256(completion)
    base.atomic_write_json(completion_path, completion)
    base.validate_completion(work_dir)
    print(json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Completed BGE all-human H100 export: {work_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
