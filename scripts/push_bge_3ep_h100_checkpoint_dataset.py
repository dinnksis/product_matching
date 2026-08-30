#!/usr/bin/env python3
"""Upload the exact trained BGE 3ep H100 checkpoint as a private Dataset."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import push_bge_pretrain_checkpoint_dataset as hardened
import push_kaggle_training_dataset as shared_push
import push_minilm_pretrain_checkpoint_dataset as checkpoint_remote
import run_kaggle_notebook as kaggle


DATASET_SLUG = "product-matching-bge-3ep-h100-oodtrain"
SOURCE_DIR = ROOT / "artifacts" / "server_bge_3ep_h100" / "run"
STAGE_DIR = ROOT / ".kaggle" / "datasets" / f"{DATASET_SLUG}-model-only-stage"
MODEL_FILENAME = "model.safetensors"
MANIFEST_NAME = "checkpoint_manifest.json"
DIRECT_CHECKPOINT_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
CHECKPOINT_FILES = DIRECT_CHECKPOINT_FILES + (MODEL_FILENAME,)
IGNORED_SOURCE_FILES = {
    "iid_validation_predictions.parquet",
    "hard_validation_predictions.parquet",
    "training_config.json",
    "training_report.json",
}
EXPECTED_SOURCE_FILES: dict[str, dict[str, object]] = {
    "config.json": {
        "bytes": 764,
        "sha256": "3f143138299caf72270c79814d1f0e3c38fc168e2d30f1e6cc4c70f6cdb481f5",
    },
    "tokenizer.json": {
        "bytes": 17_082_998,
        "sha256": "efcbc2e883c0ef3f67ce522db4e6e4e114532aeb827b5e2cbfa89bac7da8252f",
    },
    "tokenizer_config.json": {
        "bytes": 1_207,
        "sha256": "272dabf15417f68de99532ca6a2f7c402afdf654d4dc4134f35db97af1edc8b0",
    },
    "special_tokens_map.json": {
        "bytes": 964,
        "sha256": "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835",
    },
    MODEL_FILENAME: {
        "bytes": 2_271_071_852,
        "sha256": "d7e899ea3cd305db970aa6f3466eb71a138ad418c74b8b6ac730d1828c4a4ab8",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--stage-dir", type=Path, default=STAGE_DIR)
    parser.add_argument(
        "--message",
        default="BGE 3ep H100 SFT checkpoint trained with former OOD in train",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_source_checkpoint(
    source_dir: Path,
    *,
    expected_files: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    expected = hardened._normalized_expected_files(
        EXPECTED_SOURCE_FILES if expected_files is None else expected_files
    )
    actual = {path.name for path in source_dir.iterdir()}
    allowed = set(CHECKPOINT_FILES) | IGNORED_SOURCE_FILES
    if actual != allowed:
        kaggle.fail(
            "trained BGE output file set mismatch; "
            f"missing={sorted(allowed - actual)}, unexpected={sorted(actual - allowed)}"
        )
    measured: dict[str, dict[str, object]] = {}
    for filename in CHECKPOINT_FILES:
        record, _ = hardened._read_stable_regular_file(source_dir / filename)
        if record != expected[filename]:
            kaggle.fail(
                f"trained BGE bytes differ for {filename}: "
                f"expected {expected[filename]}, got {record}"
            )
        measured[filename] = record
    return measured


@contextmanager
def hardened_contract() -> Iterator[None]:
    names = (
        "DATASET_SLUG",
        "SOURCE_DIR",
        "STAGE_DIR",
        "DIRECT_CHECKPOINT_FILES",
        "CHECKPOINT_FILES",
        "MODEL_FILENAME",
        "EXPECTED_SOURCE_FILES",
        "validate_source_checkpoint",
    )
    previous = {name: getattr(hardened, name) for name in names}
    hardened.DATASET_SLUG = DATASET_SLUG
    hardened.SOURCE_DIR = SOURCE_DIR
    hardened.STAGE_DIR = STAGE_DIR
    hardened.DIRECT_CHECKPOINT_FILES = DIRECT_CHECKPOINT_FILES
    hardened.CHECKPOINT_FILES = CHECKPOINT_FILES
    hardened.MODEL_FILENAME = MODEL_FILENAME
    hardened.EXPECTED_SOURCE_FILES = EXPECTED_SOURCE_FILES
    hardened.validate_source_checkpoint = validate_source_checkpoint
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(hardened, name, value)


def build_payload(source_dir: Path, stage_dir: Path, owner: str) -> dict[str, object]:
    with hardened_contract():
        manifest = hardened.build_payload(
            source_dir,
            stage_dir,
            owner,
            dataset_slug=DATASET_SLUG,
            expected_files=EXPECTED_SOURCE_FILES,
        )
        manifest["purpose"] = "private deployable BGE product-matching checkpoint"
        manifest["provenance"] = {
            "status": "locally_trained_exact_bytes",
            "model_family": "XLM-R/BGE reranker sequence classifier",
            "declared_training": "three-epoch supervised H100 run",
            "training_metadata_included": False,
            "validation_predictions_included": False,
            "verified_scope": "exact final checkpoint, tokenizer and config bytes",
        }
        (stage_dir / MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metadata = {
            "title": "Product Matching BGE 3ep H100 OOD-train",
            "id": f"{owner}/{DATASET_SLUG}",
            "licenses": [{"name": "unknown"}],
            "isPrivate": True,
            "description": (
                "Private exact BGE/XLM-R product-matching checkpoint trained for "
                "three epochs on human train plus former OOD. The safetensors "
                "file is losslessly sharded into 64 MiB pieces."
            ),
        }
        (stage_dir / "dataset-metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        hardened.verify_payload_for_upload(
            stage_dir,
            manifest,
            expected_checkpoint_files=EXPECTED_SOURCE_FILES,
        )
        return manifest


def main() -> int:
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
    with hardened_contract():
        source_dir, stage_dir = hardened.resolve_isolated_paths(source_dir, stage_dir)
    manifest = build_payload(source_dir, stage_dir, owner)
    manifest_hash = hardened.sha256_file(stage_dir / MANIFEST_NAME)
    print(f"Prepared private trained-model Dataset: {stage_dir}")
    print(f"Dataset reference: {manifest['dataset']}")
    print(f"Final model SHA-256: {EXPECTED_SOURCE_FILES[MODEL_FILENAME]['sha256']}")
    print(f"Manifest SHA-256: {manifest_hash}")
    if args.dry_run:
        print("Dry run complete; Kaggle was not contacted.")
        return 0

    with hardened_contract():
        manifest_hash = hardened.verify_payload_for_upload(
            stage_dir, manifest, expected_checkpoint_files=EXPECTED_SOURCE_FILES
        )
    cli = kaggle.kaggle_command()
    dataset_ref = str(manifest["dataset"])
    previous = shared_push.dataset_status(cli, dataset_ref)
    previous_version = int(previous.get("current_version_number", 0)) if previous else 0
    command = (
        cli + ["datasets", "create", "--path", str(stage_dir)]
        if previous is None
        else cli + [
            "datasets",
            "version",
            "--path",
            str(stage_dir),
            "--message",
            args.message,
        ]
    )
    with hardened_contract():
        manifest_hash = hardened.verify_payload_for_upload(
            stage_dir, manifest, expected_checkpoint_files=EXPECTED_SOURCE_FILES
        )
    print("$", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, env=os.environ.copy())
    if result.returncode:
        kaggle.fail(f"trained-model Dataset upload failed with {result.returncode}")
    status = shared_push.wait_until_ready(
        cli, dataset_ref, minimum_version=previous_version + 1
    )
    checkpoint_remote.verify_remote_dataset(
        cli, dataset_ref, manifest_hash, set(manifest["files"])
    )
    print(
        "Private trained-model Dataset ready at version "
        f"{status.get('current_version_number')}: "
        f"https://www.kaggle.com/datasets/{dataset_ref}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
