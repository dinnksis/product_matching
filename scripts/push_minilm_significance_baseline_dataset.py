#!/usr/bin/env python3
"""Create or version the private frozen significance-baseline Dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import push_kaggle_training_dataset as shared_push
import run_kaggle_notebook as kaggle


DATASET_SLUG = "product-matching-minilm-5ep-significance-v1"
BASELINE_RUN_ID = "67f4fe76886b43d6b52ed5cb49068e1e"
SOURCE_DIR = (
    ROOT
    / "artifacts"
    / "kaggle"
    / "product-matching-minilm-5ep-human-ft-v1"
    / "minilm_llm_pretrain_5ep_human_ft_v1"
)
STAGE_DIR = ROOT / ".kaggle" / "datasets" / DATASET_SLUG
MANIFEST_FILENAME = "minilm_5ep_significance_baseline_manifest.json"
PREDICTION_FILENAMES = {
    split: f"{split}_validation_predictions.parquet"
    for split in ("iid", "hard", "ood")
}
COMPACT_COLUMNS = ("id1", "id2", "target", "category_1", "score")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload slim frozen MiniLM baseline predictions for paired "
            "IID/hard/OOD significance tests"
        )
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--stage-dir", type=Path, default=STAGE_DIR)
    parser.add_argument(
        "--message",
        default="Freeze MiniLM 5ep human fine-tune significance baseline v1",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_compact_predictions(source: Path, destination: Path) -> int:
    if not source.is_file():
        kaggle.fail(f"baseline prediction artifact is missing: {source}")
    frame = pd.read_parquet(source, columns=list(COMPACT_COLUMNS))
    if frame.empty:
        kaggle.fail(f"baseline prediction artifact is empty: {source}")
    if frame[list(COMPACT_COLUMNS)].isna().any().any():
        kaggle.fail(f"baseline prediction artifact contains nulls: {source}")
    if not frame["target"].isin([0, 1, 0.0, 1.0]).all():
        kaggle.fail(f"baseline prediction targets are not binary: {source}")
    unordered = pd.DataFrame(
        {
            "left": frame[["id1", "id2"]].min(axis=1),
            "right": frame[["id1", "id2"]].max(axis=1),
        }
    )
    if unordered.duplicated().any():
        kaggle.fail(f"baseline prediction artifact has duplicate pairs: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, destination)
    return len(frame)


def build_payload(source_dir: Path, stage_dir: Path, owner: str) -> dict[str, object]:
    source_dir = source_dir.expanduser().resolve()
    stage_dir = stage_dir.expanduser().resolve()
    stage_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, object]] = {}
    for split, filename in PREDICTION_FILENAMES.items():
        source = source_dir / filename
        destination = stage_dir / filename
        rows = _write_compact_predictions(source, destination)
        files[filename] = {
            "split": split,
            "rows": rows,
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "source_sha256": sha256_file(source),
            "columns": list(COMPACT_COLUMNS),
        }

    dataset_ref = f"{owner}/{DATASET_SLUG}"
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset": dataset_ref,
        "is_private": True,
        "baseline_run_id": BASELINE_RUN_ID,
        "baseline_experiment": "minilm_llm_pretrain_5ep_human_ft_v1",
        "validation_protocol": "product-matching-validation-splits-v1",
        "files": files,
    }
    manifest_path = stage_dir / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "title": "MiniLM 5ep Significance Baseline v1",
        "id": dataset_ref,
        "licenses": [{"name": "unknown"}],
        "isPrivate": True,
        "description": (
            "Private slim IID/hard/OOD predictions for paired statistical "
            "comparison with the frozen MiniLM 5ep human fine-tune baseline."
        ),
    }
    (stage_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    expected_names = set(PREDICTION_FILENAMES.values()) | {MANIFEST_FILENAME}
    actual_names = {
        path.name
        for path in stage_dir.iterdir()
        if path.is_file() and path.name != "dataset-metadata.json"
    }
    if actual_names != expected_names:
        kaggle.fail(
            "staged significance baseline file set differs: "
            f"expected={sorted(expected_names)}, actual={sorted(actual_names)}"
        )
    return manifest


def verify_remote_dataset(
    cli: list[str],
    dataset_ref: str,
    expected_files: set[str],
) -> None:
    files = kaggle.run_command(
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
    missing = sorted(name for name in expected_files if name not in files.stdout)
    if missing:
        kaggle.fail(f"remote significance baseline Dataset is missing: {missing}", 1)
    with tempfile.TemporaryDirectory(prefix="kaggle-significance-metadata-") as temp:
        metadata_dir = Path(temp)
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
        remote_metadata = json.loads(
            (metadata_dir / "dataset-metadata.json").read_text(encoding="utf-8")
        )
        info = remote_metadata.get("info", remote_metadata)
        if info.get("isPrivate") is not True:
            kaggle.fail("Kaggle reports that the significance Dataset is public", 1)


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
    manifest = build_payload(source_dir, stage_dir, owner)
    dataset_ref = str(manifest["dataset"])
    expected_files = set(PREDICTION_FILENAMES.values()) | {MANIFEST_FILENAME}
    total_bytes = sum(
        int(declaration["bytes"])
        for declaration in manifest["files"].values()  # type: ignore[union-attr]
    )
    print(f"Prepared private significance baseline Dataset: {stage_dir}")
    print(f"Dataset reference: {dataset_ref}")
    print(f"Baseline run_id: {BASELINE_RUN_ID}")
    print(f"Prediction bytes: {total_bytes:,}")
    print(f"Manifest SHA-256: {sha256_file(stage_dir / MANIFEST_FILENAME)}")
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
    kaggle.run_command(command)
    shared_push.wait_until_ready(
        cli,
        dataset_ref,
        minimum_version=previous_version + 1,
    )
    verify_remote_dataset(cli, dataset_ref, expected_files)
    print(f"Private Dataset is ready: https://www.kaggle.com/datasets/{dataset_ref}")


if __name__ == "__main__":
    main()
