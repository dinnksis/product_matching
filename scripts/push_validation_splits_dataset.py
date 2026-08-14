#!/usr/bin/env python3
"""Create or version the private frozen validation-splits Dataset on Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import push_kaggle_training_dataset as shared_push
import run_kaggle_notebook as kaggle


DATASET_SLUG = "product-matching-validation-splits-v1"
SOURCE_DIR = ROOT / "prepared" / "validation_splits_v1"
STAGE_DIR = ROOT / ".kaggle" / "datasets" / DATASET_SLUG
FILE_MAP = {
    "manifest.json": "validation_splits_manifest.json",
    "human/items.parquet": "human_items.parquet",
    "human/hard_selection_details.parquet": "human_hard_selection_details.parquet",
    "human/hard_validation_pairs.parquet": "human_hard_validation_pairs.parquet",
    "human/iid_validation_pairs.parquet": "human_iid_validation_pairs.parquet",
    "human/ood_validation_pairs.parquet": "human_ood_validation_pairs.parquet",
    "human/split_assignments.parquet": "human_split_assignments.parquet",
    "human/train_pairs.parquet": "human_train_pairs.parquet",
    "llm/non_ood_items.parquet": "llm_non_ood_items.parquet",
    "llm/non_ood_pairs.parquet": "llm_non_ood_pairs.parquet",
    "llm/ood_items.parquet": "llm_ood_items.parquet",
    "llm/ood_pairs.parquet": "llm_ood_pairs.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload validation_splits_v1 as a private Kaggle Dataset"
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--stage-dir", type=Path, default=STAGE_DIR)
    parser.add_argument(
        "--message",
        default="Freeze human IID hard OOD and LLM category splits v1",
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


def build_payload(source_dir: Path, stage_dir: Path, owner: str) -> dict[str, object]:
    source_dir = source_dir.resolve()
    source_manifest_path = source_dir / "manifest.json"
    if not source_manifest_path.is_file():
        kaggle.fail(f"validation split manifest is missing: {source_manifest_path}")
    try:
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        kaggle.fail(f"validation split manifest is invalid: {error}")
    if source_manifest.get("version") != "human_v1":
        kaggle.fail("validation split manifest must have version='human_v1'")
    if source_manifest.get("ood_categories") != ["Одежда", "Бытовая техника"]:
        kaggle.fail("unexpected OOD categories in validation split manifest")

    declared_outputs = source_manifest.get("outputs")
    if not isinstance(declared_outputs, dict):
        kaggle.fail("validation split manifest has no output hashes")
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged_files: dict[str, dict[str, object]] = {}
    for relative_source, destination_name in FILE_MAP.items():
        source = source_dir / relative_source
        if not source.is_file():
            kaggle.fail(f"validation split file is missing: {source}")
        if relative_source == "manifest.json":
            expected_hash = sha256_file(source)
        else:
            declaration = declared_outputs.get(relative_source)
            if not isinstance(declaration, dict):
                kaggle.fail(f"manifest has no hash for {relative_source}")
            expected_hash = str(declaration.get("sha256", ""))
            if source.stat().st_size != int(declaration.get("bytes", -1)):
                kaggle.fail(f"size differs from manifest for {relative_source}")
            if sha256_file(source) != expected_hash:
                kaggle.fail(f"SHA-256 differs from manifest for {relative_source}")
        destination = stage_dir / destination_name
        link_or_copy(source, destination, expected_hash)
        staged_files[destination_name] = {
            "source": relative_source,
            "bytes": source.stat().st_size,
            "sha256": expected_hash,
        }

    dataset_ref = f"{owner}/{DATASET_SLUG}"
    upload_manifest = {
        "schema_version": 1,
        "dataset": dataset_ref,
        "is_private": True,
        "source_version": source_manifest["version"],
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "files": staged_files,
    }
    (stage_dir / "upload_manifest.json").write_text(
        json.dumps(upload_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "title": "Product Matching Validation Splits v1",
        "id": dataset_ref,
        "licenses": [{"name": "unknown"}],
        "isPrivate": True,
        "description": (
            "Private frozen E-CUP 2026 product-matching data: component-disjoint "
            "human train, IID, hard and category-OOD splits plus LLM items and pairs "
            "separated by the same OOD categories. Not for publication."
        ),
    }
    (stage_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    expected_names = set(FILE_MAP.values()) | {"upload_manifest.json"}
    actual_names = {
        path.name
        for path in stage_dir.iterdir()
        if path.is_file() and path.name != "dataset-metadata.json"
    }
    unexpected = actual_names - expected_names
    missing = expected_names - actual_names
    if unexpected or missing:
        kaggle.fail(
            f"staged file set mismatch; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    return upload_manifest


def verify_remote_dataset(
    cli: list[str],
    dataset_ref: str,
    expected_files: set[str],
) -> dict[str, object]:
    files = kaggle.run_command(
        cli
        + [
            "datasets",
            "files",
            dataset_ref,
            "--format",
            "json",
            "--page-size",
            "200",
        ]
    )
    missing = sorted(name for name in expected_files if name not in files.stdout)
    if missing:
        kaggle.fail(f"remote Dataset is missing files: {missing}", 1)

    with tempfile.TemporaryDirectory(prefix="kaggle-validation-metadata-") as temporary:
        metadata_dir = Path(temporary)
        kaggle.run_command(
            cli
            + [
                "datasets",
                "metadata",
                dataset_ref,
                "--path",
                str(metadata_dir),
            ]
        )
        metadata_path = metadata_dir / "dataset-metadata.json"
        remote_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        info = remote_metadata.get("info", remote_metadata)
        if info.get("isPrivate") is not True:
            kaggle.fail("Kaggle reports that the Dataset is not private", 1)
    return remote_metadata


def main() -> None:
    args = parse_args()
    env_file = args.env_file if args.env_file.is_absolute() else ROOT / args.env_file
    kaggle.load_dotenv(env_file)
    owner = kaggle.validate_slug(
        os.getenv("KAGGLE_USERNAME", "").strip(),
        "KAGGLE_USERNAME",
    )
    if not args.dry_run and not os.getenv("KAGGLE_API_TOKEN", "").strip():
        kaggle.fail("set KAGGLE_API_TOKEN in .env")

    source_dir = args.source_dir if args.source_dir.is_absolute() else ROOT / args.source_dir
    stage_dir = args.stage_dir if args.stage_dir.is_absolute() else ROOT / args.stage_dir
    upload_manifest = build_payload(source_dir, stage_dir, owner)
    dataset_ref = str(upload_manifest["dataset"])
    expected_files = set(FILE_MAP.values()) | {"upload_manifest.json"}
    print(f"Prepared private Kaggle Dataset payload: {stage_dir}")
    print(f"Dataset reference: {dataset_ref}")
    print(f"Files: {len(expected_files)}; total bytes: "
          f"{sum(int(item['bytes']) for item in upload_manifest['files'].values()):,}")
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
        command = cli + [
            "datasets",
            "create",
            "--path",
            str(stage_dir),
            "--keep-tabular",
        ]
    else:
        command = cli + [
            "datasets",
            "version",
            "--path",
            str(stage_dir),
            "--message",
            args.message,
            "--keep-tabular",
        ]
    # Deliberately omit --public: Kaggle CLI creates datasets privately by default.
    kaggle.run_command(command)
    status = shared_push.wait_until_ready(
        cli,
        dataset_ref,
        minimum_version=previous_version + 1,
    )
    verify_remote_dataset(cli, dataset_ref, expected_files)
    print(
        f"Private Dataset is ready at version "
        f"{status.get('current_version_number')}: "
        f"https://www.kaggle.com/datasets/{dataset_ref}"
    )


if __name__ == "__main__":
    main()
