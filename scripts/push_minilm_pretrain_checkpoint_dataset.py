#!/usr/bin/env python3
"""Create or version the private one-epoch LLM-pretrained MiniLM checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import push_kaggle_training_dataset as shared_push
import run_kaggle_notebook as kaggle


DATASET_SLUG = "product-matching-minilm-llm-pretrain-1ep"
SOURCE_DIR = ROOT / "model" / "pretrain_minilm_1ep"
STAGE_DIR = ROOT / ".kaggle" / "datasets" / DATASET_SLUG
DIRECT_CHECKPOINT_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "training_args.json",
    "training_state.json",
)
MODEL_FILENAME = "model.safetensors"
MODEL_PART_BYTES = 64 * 1024 * 1024
MANIFEST_NAME = "checkpoint_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the LLM-pretrained MiniLM checkpoint as a private Kaggle Dataset"
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--stage-dir", type=Path, default=STAGE_DIR)
    parser.add_argument("--dataset-slug", default=DATASET_SLUG)
    parser.add_argument(
        "--message",
        default="One epoch LLM-pretrained MiniLM checkpoint for human fine-tuning",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source: Path, destination: Path, expected_hash: str) -> None:
    if destination.is_file():
        if (
            destination.stat().st_size == source.stat().st_size
            and sha256_file(destination) == expected_hash
        ):
            return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def split_model(source: Path, stage_dir: Path) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    model_digest = hashlib.sha256()
    parts: dict[str, dict[str, object]] = {}
    with source.open("rb") as stream:
        index = 0
        while chunk := stream.read(MODEL_PART_BYTES):
            model_digest.update(chunk)
            part_name = f"{MODEL_FILENAME}.part{index:03d}"
            part_hash = hashlib.sha256(chunk).hexdigest()
            destination = stage_dir / part_name
            if not (
                destination.is_file()
                and destination.stat().st_size == len(chunk)
                and sha256_file(destination) == part_hash
            ):
                temporary = destination.with_name(f".{destination.name}.tmp")
                temporary.unlink(missing_ok=True)
                with temporary.open("wb") as output:
                    output.write(chunk)
                os.replace(temporary, destination)
            parts[part_name] = {"bytes": len(chunk), "sha256": part_hash}
            index += 1
    return parts, {
        "filename": MODEL_FILENAME,
        "bytes": source.stat().st_size,
        "sha256": model_digest.hexdigest(),
        "parts": list(parts),
    }


def build_payload(
    source_dir: Path,
    stage_dir: Path,
    owner: str,
    *,
    dataset_slug: str = DATASET_SLUG,
) -> dict[str, object]:
    source_dir = source_dir.resolve()
    dataset_slug = kaggle.validate_slug(dataset_slug, "dataset slug")
    stage_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, object]] = {}
    for filename in DIRECT_CHECKPOINT_FILES:
        source = source_dir / filename
        if not source.is_file():
            kaggle.fail(f"checkpoint file is missing: {source}")
        file_hash = sha256_file(source)
        link_or_copy(source, stage_dir / filename, file_hash)
        files[filename] = {
            "bytes": source.stat().st_size,
            "sha256": file_hash,
        }
    model_source = source_dir / MODEL_FILENAME
    if not model_source.is_file():
        kaggle.fail(f"checkpoint model is missing: {model_source}")
    # Never upload the single 449 MiB file: Kaggle's resumable upload can stall
    # indefinitely on it. The notebook reconstructs the exact bytes and checks
    # the original SHA-256 before loading Transformers.
    (stage_dir / MODEL_FILENAME).unlink(missing_ok=True)
    model_parts, model_reconstruction = split_model(model_source, stage_dir)
    files.update(model_parts)

    dataset_ref = f"{owner}/{dataset_slug}"
    try:
        source_label = source_dir.relative_to(ROOT).as_posix()
    except ValueError:
        source_label = str(source_dir)
    manifest = {
        "schema_version": 1,
        "dataset": dataset_ref,
        "is_private": True,
        "source": source_label,
        "purpose": "initialize a fresh human-only fine-tuning run",
        "files": files,
        "reconstruction": model_reconstruction,
    }
    (stage_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "title": f"Product Matching MiniLM {source_dir.name}",
        "id": dataset_ref,
        "licenses": [{"name": "unknown"}],
        "isPrivate": True,
        "description": (
            "Private MiniLM sequence-classification weights after one epoch on "
            "non-OOD LLM-labelled pairs. Optimizer and ELR state are intentionally "
            "excluded because downstream training starts a fresh human-only stage. "
            "The safetensors file is losslessly sharded for reliable upload."
        ),
    }
    (stage_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    expected = set(files) | {MANIFEST_NAME}
    actual = {
        path.name
        for path in stage_dir.iterdir()
        if path.is_file() and path.name != "dataset-metadata.json"
    }
    if actual != expected:
        kaggle.fail(
            f"staged checkpoint file set mismatch; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return manifest


def verify_remote_dataset(
    cli: list[str],
    dataset_ref: str,
    expected_manifest_hash: str,
    expected_files: set[str],
) -> None:
    expected_files = set(expected_files) | {MANIFEST_NAME}
    remote_files = kaggle.run_command(
        cli
        + [
            "datasets",
            "files",
            dataset_ref,
            "--format",
            "json",
            "--page-size",
            "100",
        ]
    )
    missing = sorted(name for name in expected_files if name not in remote_files.stdout)
    if missing:
        kaggle.fail(f"remote checkpoint Dataset is missing files: {missing}", 1)
    with tempfile.TemporaryDirectory(prefix="kaggle-minilm-checkpoint-") as temporary:
        destination = Path(temporary)
        kaggle.run_command(
            cli
            + [
                "datasets",
                "download",
                dataset_ref,
                "-f",
                MANIFEST_NAME,
                "-p",
                str(destination),
                "-o",
                "-q",
            ]
        )
        downloaded = destination / MANIFEST_NAME
        if sha256_file(downloaded) != expected_manifest_hash:
            kaggle.fail("remote checkpoint manifest differs from the local payload", 1)
    with tempfile.TemporaryDirectory(prefix="kaggle-minilm-checkpoint-metadata-") as temporary:
        destination = Path(temporary)
        kaggle.run_command(
            cli + ["datasets", "metadata", dataset_ref, "--path", str(destination)]
        )
        metadata = json.loads(
            (destination / "dataset-metadata.json").read_text(encoding="utf-8")
        )
        info = metadata.get("info", metadata)
        if info.get("isPrivate") is not True:
            kaggle.fail("Kaggle reports that the checkpoint Dataset is not private", 1)


def main() -> None:
    args = parse_args()
    env_file = args.env_file if args.env_file.is_absolute() else ROOT / args.env_file
    kaggle.load_dotenv(env_file)
    owner = kaggle.validate_slug(
        os.getenv("KAGGLE_USERNAME", "").strip(), "KAGGLE_USERNAME"
    )
    if not args.dry_run and not os.getenv("KAGGLE_API_TOKEN", "").strip():
        kaggle.fail("set KAGGLE_API_TOKEN in .env")

    source_dir = args.source_dir if args.source_dir.is_absolute() else ROOT / args.source_dir
    stage_dir = args.stage_dir if args.stage_dir.is_absolute() else ROOT / args.stage_dir
    manifest = build_payload(
        source_dir,
        stage_dir,
        owner,
        dataset_slug=args.dataset_slug,
    )
    dataset_ref = str(manifest["dataset"])
    manifest_hash = sha256_file(stage_dir / MANIFEST_NAME)
    total_bytes = sum(int(item["bytes"]) for item in manifest["files"].values())
    print(f"Prepared private checkpoint Dataset: {stage_dir}")
    print(f"Dataset reference: {dataset_ref}")
    print(f"Checkpoint bytes: {total_bytes:,}; manifest SHA-256: {manifest_hash}")
    if args.dry_run:
        print("Dry run complete; Kaggle was not contacted.")
        return

    cli = kaggle.kaggle_command()
    previous_status = shared_push.dataset_status(cli, dataset_ref)
    previous_version = (
        int(previous_status.get("current_version_number", 0))
        if previous_status
        else 0
    )
    if previous_status is None:
        command = cli + ["datasets", "create", "--path", str(stage_dir)]
    else:
        command = cli + [
            "datasets",
            "version",
            "--path",
            str(stage_dir),
            "--message",
            args.message,
        ]
    # Large resumable uploads can run for several minutes. Inherit stdout/stderr
    # here so Kaggle's progress and retry diagnostics stay visible instead of
    # being buffered until the whole 450+ MiB transfer completes.
    print("$", " ".join(command), flush=True)
    upload = subprocess.run(command, cwd=ROOT, env=os.environ.copy())
    if upload.returncode:
        kaggle.fail(
            f"checkpoint Dataset upload failed with exit code {upload.returncode}",
            upload.returncode,
        )
    status = shared_push.wait_until_ready(
        cli, dataset_ref, minimum_version=previous_version + 1
    )
    verify_remote_dataset(cli, dataset_ref, manifest_hash, set(manifest["files"]))
    print(
        f"Private checkpoint Dataset is ready at version "
        f"{status.get('current_version_number')}: "
        f"https://www.kaggle.com/datasets/{dataset_ref}"
    )


if __name__ == "__main__":
    main()
