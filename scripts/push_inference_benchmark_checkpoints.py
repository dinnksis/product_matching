#!/usr/bin/env python3
"""Stage and upload the three frozen inference-benchmark checkpoints.

Large safetensors files are split losslessly so Kaggle's resumable Dataset
upload does not have to transfer a multi-gigabyte object in one request.  The
benchmark notebook reconstructs the original bytes and verifies every hash
before importing Transformers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import push_kaggle_training_dataset as shared_push
import run_kaggle_notebook as kaggle


DATASET_SLUG = "product-matching-inference-checkpoints-v1"
STAGE_DIR = ROOT / ".kaggle" / "datasets" / DATASET_SLUG
MANIFEST_NAME = "inference_benchmark_manifest.json"
PART_BYTES = 128 * 1024 * 1024

MODEL_SOURCES = {
    "bge": {
        "checkpoint": ROOT / "artifacts/kaggle/product-matching-architecture-bge-v2-m3-v1/bge_reranker_v2_m3_human_ft_v1",
        "tokenizer": ROOT / "artifacts/kaggle/product-matching-architecture-bge-v2-m3-v1/bge_reranker_v2_m3_human_ft_v1",
        "files": (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
        ),
        "model_sha256": "0e4e8e0b9dd5f220fd0423a5a27d8d6c09ff3697e72fbbf16f861f03df02b87e",
    },
    "minilm": {
        "checkpoint": ROOT / "artifacts/kaggle/product-matching-architecture-minilm-5ep-v1/minilm_5ep_synthetic_pretrain_human_ft_s2_v1",
        "tokenizer": ROOT / "submits/minilm-s2-values-only/models/minilm-s2-values-only",
        "files": (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ),
        "model_sha256": "1122ac37bda1be257b743c56468fbf3abba8a341f89c7d34888a6c0f3afbb6ab",
    },
    "rumodernbert": {
        "checkpoint": ROOT / "artifacts/kaggle/product-matching-architecture-rumodernbert-v1/rumodernbert_base_random_head_human_ft_v1",
        "tokenizer": ROOT / "artifacts/kaggle/product-matching-architecture-rumodernbert-v1/rumodernbert_base_random_head_human_ft_v1",
        "files": ("config.json", "tokenizer.json", "tokenizer_config.json"),
        "model_sha256": "e6cda247fe02e615bfc1ddc8849a7b82e207dbe603c533cd6b529ed66eb19bce",
    },
}

REFERENCE_SOURCES = {
    model: {
        split: ROOT / "preds" / f"preds_{model}" / f"{split}_validation_predictions.parquet"
        for split in ("iid", "hard", "ood")
    }
    for model in MODEL_SOURCES
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--stage-dir", type=Path, default=STAGE_DIR)
    parser.add_argument("--dataset-slug", default=DATASET_SLUG)
    parser.add_argument(
        "--message",
        default="Frozen BGE MiniLM RuModernBERT checkpoints for inference benchmarking",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source: Path, destination: Path, digest: str) -> None:
    if destination.is_file() and destination.stat().st_size == source.stat().st_size:
        if sha256_file(destination) == digest:
            return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def split_file(source: Path, stage_dir: Path, prefix: str) -> dict[str, object]:
    source_digest = hashlib.sha256()
    parts: list[dict[str, object]] = []
    with source.open("rb") as stream:
        index = 0
        while chunk := stream.read(PART_BYTES):
            source_digest.update(chunk)
            staged_name = f"{prefix}__model.safetensors.part{index:03d}"
            digest = hashlib.sha256(chunk).hexdigest()
            destination = stage_dir / staged_name
            if not (
                destination.is_file()
                and destination.stat().st_size == len(chunk)
                and sha256_file(destination) == digest
            ):
                temporary = destination.with_name(f".{destination.name}.tmp")
                temporary.unlink(missing_ok=True)
                with temporary.open("wb") as output:
                    output.write(chunk)
                os.replace(temporary, destination)
            parts.append({"staged_name": staged_name, "bytes": len(chunk), "sha256": digest})
            index += 1
    return {
        "filename": "model.safetensors",
        "bytes": source.stat().st_size,
        "sha256": source_digest.hexdigest(),
        "parts": parts,
    }


def build_payload(stage_dir: Path, owner: str, dataset_slug: str) -> dict[str, object]:
    dataset_slug = kaggle.validate_slug(dataset_slug, "dataset slug")
    stage_dir.mkdir(parents=True, exist_ok=True)
    expected_staged: set[str] = set()
    models: dict[str, object] = {}

    for model_name, declaration in MODEL_SOURCES.items():
        checkpoint_dir = Path(declaration["checkpoint"])
        tokenizer_dir = Path(declaration["tokenizer"])
        direct_files: list[dict[str, object]] = []
        for filename in declaration["files"]:
            source = checkpoint_dir / filename if filename == "config.json" else tokenizer_dir / filename
            if not source.is_file() or source.stat().st_size == 0:
                kaggle.fail(f"checkpoint file is missing: {source}")
            digest = sha256_file(source)
            staged_name = f"{model_name}__{filename}"
            link_or_copy(source, stage_dir / staged_name, digest)
            expected_staged.add(staged_name)
            direct_files.append(
                {
                    "filename": filename,
                    "staged_name": staged_name,
                    "bytes": source.stat().st_size,
                    "sha256": digest,
                }
            )

        model_source = checkpoint_dir / "model.safetensors"
        if not model_source.is_file() or model_source.stat().st_size == 0:
            kaggle.fail(f"model weights are missing: {model_source}")
        reconstruction = split_file(model_source, stage_dir, model_name)
        if reconstruction["sha256"] != declaration["model_sha256"]:
            kaggle.fail(
                f"{model_name} model SHA-256 changed: {reconstruction['sha256']}"
            )
        expected_staged.update(part["staged_name"] for part in reconstruction["parts"])
        models[model_name] = {
            "checkpoint_source": str(checkpoint_dir.relative_to(ROOT)).replace("\\", "/"),
            "direct_files": direct_files,
            "model": reconstruction,
        }

    references: dict[str, object] = {}
    for model_name, splits in REFERENCE_SOURCES.items():
        references[model_name] = {}
        for split, source in splits.items():
            if not source.is_file() or source.stat().st_size == 0:
                kaggle.fail(f"reference predictions are missing: {source}")
            digest = sha256_file(source)
            staged_name = f"reference__{model_name}__{split}.parquet"
            link_or_copy(source, stage_dir / staged_name, digest)
            expected_staged.add(staged_name)
            references[model_name][split] = {
                "staged_name": staged_name,
                "bytes": source.stat().st_size,
                "sha256": digest,
            }

    frequency_source = ROOT / "prepared/serialization_ablation/attribute_name_frequency.csv"
    frequency_digest = sha256_file(frequency_source)
    frequency_name = "attribute_name_frequency.csv"
    link_or_copy(frequency_source, stage_dir / frequency_name, frequency_digest)
    expected_staged.add(frequency_name)

    manifest = {
        "schema_version": 1,
        "dataset": f"{owner}/{dataset_slug}",
        "is_private": True,
        "purpose": "frozen inference backend benchmark; no training",
        "serialization": "S2_VALUES_ONLY",
        "max_length": 384,
        "symmetric_scoring": True,
        "models": models,
        "references": references,
        "attribute_frequency": {
            "staged_name": frequency_name,
            "bytes": frequency_source.stat().st_size,
            "sha256": frequency_digest,
        },
    }
    manifest_path = stage_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    expected_staged.add(MANIFEST_NAME)
    metadata = {
        "title": "Product Matching Inference Checkpoints v1",
        "id": manifest["dataset"],
        "licenses": [{"name": "unknown"}],
        "isPrivate": True,
        "description": (
            "Private frozen BGE v2-m3, MiniLM and RuModernBERT human-FT "
            "checkpoints with validation reference scores. Safetensors are "
            "losslessly sharded and verified before use."
        ),
    }
    (stage_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    actual = {
        path.name
        for path in stage_dir.iterdir()
        if path.is_file() and path.name != "dataset-metadata.json"
    }
    stale = actual - expected_staged
    for name in stale:
        (stage_dir / name).unlink()
    missing = expected_staged - {
        path.name
        for path in stage_dir.iterdir()
        if path.is_file() and path.name != "dataset-metadata.json"
    }
    if missing:
        kaggle.fail(f"staged benchmark Dataset is incomplete: {sorted(missing)}")
    return manifest


def main() -> None:
    args = parse_args()
    env_file = args.env_file if args.env_file.is_absolute() else ROOT / args.env_file
    kaggle.load_dotenv(env_file)
    owner = kaggle.validate_slug(os.getenv("KAGGLE_USERNAME", "").strip(), "KAGGLE_USERNAME")
    if not args.dry_run and not os.getenv("KAGGLE_API_TOKEN", "").strip():
        kaggle.fail("set KAGGLE_API_TOKEN in .env")
    stage_dir = args.stage_dir if args.stage_dir.is_absolute() else ROOT / args.stage_dir
    manifest = build_payload(stage_dir.resolve(), owner, args.dataset_slug)
    total = sum(
        int(part["bytes"])
        for model in manifest["models"].values()
        for part in model["model"]["parts"]
    )
    print(f"Prepared private benchmark Dataset at {stage_dir}")
    print(f"Dataset reference: {manifest['dataset']}")
    print(f"Sharded model bytes: {total:,}")
    if args.dry_run:
        print("Dry run complete; Kaggle was not contacted.")
        return

    cli = kaggle.kaggle_command()
    dataset_ref = str(manifest["dataset"])
    previous = shared_push.dataset_status(cli, dataset_ref)
    previous_version = int(previous.get("current_version_number", 0)) if previous else 0
    if previous is None:
        command = cli + ["datasets", "create", "--path", str(stage_dir)]
    else:
        command = cli + [
            "datasets", "version", "--path", str(stage_dir), "--message", args.message
        ]
    print("$", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, env=os.environ.copy())
    if result.returncode:
        kaggle.fail(f"benchmark Dataset upload failed with exit code {result.returncode}")
    status = shared_push.wait_until_ready(
        cli, dataset_ref, minimum_version=previous_version + 1
    )
    print(
        f"Private benchmark Dataset ready at version {status.get('current_version_number')}: "
        f"https://www.kaggle.com/datasets/{dataset_ref}"
    )


if __name__ == "__main__":
    main()
