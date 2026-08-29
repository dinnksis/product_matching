#!/usr/bin/env python3
"""Stage and upload the exact RuModernBERT task-pretrained checkpoint privately."""

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


DATASET_SLUG = "product-matching-rumodernbert-pretrain-3ep"
SOURCE_DIR = ROOT / "model" / "pretrain_rumodernbert_3ep"
STAGE_DIR = ROOT / ".kaggle" / "datasets" / DATASET_SLUG
MODEL_FILENAME = "model.safetensors"
MANIFEST_NAME = "checkpoint_manifest.json"
DIRECT_CHECKPOINT_FILES = (
    "added_tokens.json",
    "config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
CHECKPOINT_FILES = DIRECT_CHECKPOINT_FILES + (MODEL_FILENAME,)
EXPECTED_SOURCE_FILES: dict[str, dict[str, object]] = {
    "added_tokens.json": {"bytes": 1770, "sha256": "b1fc4b8b19f7e5c2264144bf4cfdb8a5a0c62d99edce903970b889ac82f3285f"},
    "config.json": {"bytes": 2254, "sha256": "de2e907e5cd29eef9d19e24c3322abaaff959daa534ddb8041cd4cb2aa0db9d0"},
    "merges.txt": {"bytes": 1042566, "sha256": "53226b9ceaf7a1f79cf0674b94fc5e147411ef52410dadc71037a4572584ccfa"},
    "special_tokens_map.json": {"bytes": 837, "sha256": "2ea3c1eb27baf06d75115d453c067c1832d7101a97243b298a4af8d59c916d62"},
    "tokenizer.json": {"bytes": 4752755, "sha256": "588f17299b8a7c06a895439aee50a2de95e78a03d3d0ae2e30adf23aa81a0f3f"},
    "tokenizer_config.json": {"bytes": 20014, "sha256": "6030cf18a07362d8209899769e3af19cfada7363a1d3056ba47ef2f0dcaf7360"},
    "vocab.json": {"bytes": 1384902, "sha256": "65be01b3421087106b17bed0981826443bfddcba641af6428558be1bdac95a5c"},
    MODEL_FILENAME: {"bytes": 598436708, "sha256": "e8b7ebda4904c2e7f8d2ec42645cfcbda928f90fa8dbd59d03e314318118673d"},
}


def validate_source_checkpoint(
    source_dir: Path,
    *,
    expected_files: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    """Validate inference bytes while explicitly excluding pretrain optimizer state."""

    expected = hardened._normalized_expected_files(
        EXPECTED_SOURCE_FILES if expected_files is None else expected_files
    )
    actual_entries = {path.name for path in source_dir.iterdir()}
    allowed = set(CHECKPOINT_FILES) | {"optimizer.pt"}
    if actual_entries != allowed:
        kaggle.fail(
            "RuModernBERT source file set mismatch; "
            f"missing={sorted(allowed - actual_entries)}, "
            f"unexpected={sorted(actual_entries - allowed)}"
        )
    measured: dict[str, dict[str, object]] = {}
    for filename in CHECKPOINT_FILES:
        record, _ = hardened._read_stable_regular_file(source_dir / filename)
        if record != expected[filename]:
            kaggle.fail(
                f"RuModernBERT checkpoint bytes differ for {filename}: "
                f"expected {expected[filename]}, got {record}"
            )
        measured[filename] = record
    return measured


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--stage-dir", type=Path, default=STAGE_DIR)
    parser.add_argument("--dataset-slug", default=DATASET_SLUG)
    parser.add_argument("--message", default="RuModernBERT task pretrain checkpoint, epoch 3")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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
        manifest["provenance"] = {
            "status": "user_supplied_exact_bytes",
            "declared_model_family": "ruModernBERT sequence classifier",
            "declared_task_pretraining_epochs": 3,
            "declared_pretraining_categories": 20,
            "training_lineage_verified": False,
            "verified_scope": "exact checkpoint file sizes, file set and SHA-256",
        }
        manifest["purpose"] = "initialize fresh RuModernBERT supervised fine-tuning"
        (stage_dir / MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metadata = {
            "title": "Product Matching RuModernBERT Pretrain 3ep",
            "id": f"{owner}/{DATASET_SLUG}",
            "licenses": [{"name": "unknown"}],
            "isPrivate": True,
            "description": "Private exact RuModernBERT 20-category task checkpoint; fresh SFT state.",
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
    owner = kaggle.validate_slug(os.getenv("KAGGLE_USERNAME", "").strip(), "KAGGLE_USERNAME")
    if not args.dry_run and not os.getenv("KAGGLE_API_TOKEN", "").strip():
        kaggle.fail("set KAGGLE_API_TOKEN in .env")
    source_dir = args.source_dir if args.source_dir.is_absolute() else ROOT / args.source_dir
    stage_dir = args.stage_dir if args.stage_dir.is_absolute() else ROOT / args.stage_dir
    with hardened_contract():
        source_dir, stage_dir = hardened.resolve_isolated_paths(source_dir, stage_dir)
    manifest = build_payload(source_dir, stage_dir, owner)
    manifest_hash = hardened.sha256_file(stage_dir / MANIFEST_NAME)
    print(f"Prepared private checkpoint Dataset: {stage_dir}")
    print(f"Dataset reference: {manifest['dataset']}")
    print(f"Manifest SHA-256: {manifest_hash}")
    if args.dry_run:
        print("Dry run complete; Kaggle was not contacted.")
        return 0

    with hardened_contract():
        hardened.verify_payload_for_upload(
            stage_dir, manifest, expected_checkpoint_files=EXPECTED_SOURCE_FILES
        )
    cli = kaggle.kaggle_command()
    dataset_ref = str(manifest["dataset"])
    previous = shared_push.dataset_status(cli, dataset_ref)
    previous_version = int(previous.get("current_version_number", 0)) if previous else 0
    command = (
        cli + ["datasets", "create", "--path", str(stage_dir)]
        if previous is None
        else cli + ["datasets", "version", "--path", str(stage_dir), "--message", args.message]
    )
    with hardened_contract():
        manifest_hash = hardened.verify_payload_for_upload(
            stage_dir, manifest, expected_checkpoint_files=EXPECTED_SOURCE_FILES
        )
    print("$", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, env=os.environ.copy())
    if result.returncode:
        kaggle.fail(f"checkpoint Dataset upload failed with {result.returncode}", result.returncode)
    status = shared_push.wait_until_ready(
        cli, dataset_ref, minimum_version=previous_version + 1
    )
    checkpoint_remote.verify_remote_dataset(
        cli, dataset_ref, manifest_hash, set(manifest["files"])
    )
    print(f"Private checkpoint Dataset ready at version {status.get('current_version_number')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
