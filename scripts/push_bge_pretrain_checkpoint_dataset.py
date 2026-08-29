#!/usr/bin/env python3
"""Create or version the private user-supplied BGE pretrain checkpoint Dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import push_kaggle_training_dataset as shared_push
import push_minilm_pretrain_checkpoint_dataset as checkpoint_push
import run_kaggle_notebook as kaggle


DATASET_SLUG = "product-matching-bge-pretrain-2ep"
SOURCE_DIR = ROOT / "model" / "pretrain_bge_2ep"
STAGE_DIR = ROOT / ".kaggle" / "datasets" / DATASET_SLUG
DIRECT_CHECKPOINT_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
MODEL_FILENAME = "model.safetensors"
CHECKPOINT_FILES = DIRECT_CHECKPOINT_FILES + (MODEL_FILENAME,)
MODEL_PART_BYTES = 64 * 1024 * 1024
MANIFEST_NAME = "checkpoint_manifest.json"

# The checkpoint is supplied by the user without its training report or source
# manifest. These constants establish byte identity for the exact local payload;
# they deliberately do not claim that its training lineage has been verified.
EXPECTED_SOURCE_FILES: dict[str, dict[str, object]] = {
    "config.json": {
        "bytes": 764,
        "sha256": "3f143138299caf72270c79814d1f0e3c38fc168e2d30f1e6cc4c70f6cdb481f5",
    },
    "tokenizer.json": {
        "bytes": 17_082_900,
        "sha256": "8bf8afbfd11306bd872018c53bfdf2e160a56f8edbcf49933324404791c148d3",
    },
    "tokenizer_config.json": {
        "bytes": 1_203,
        "sha256": "b87c8703482b0300d3da30e201519aa641f6a450f5eb5bf1e624afbf70c74d80",
    },
    "special_tokens_map.json": {
        "bytes": 964,
        "sha256": "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835",
    },
    MODEL_FILENAME: {
        "bytes": 2_271_071_852,
        "sha256": "c21ccfcd5de310ca0328620bf8ba09e838dbe3f6394be656bd7fec16ad8377d1",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload the user-supplied two-epoch BGE pretrain checkpoint as a "
            "private Kaggle Dataset"
        )
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--stage-dir", type=Path, default=STAGE_DIR)
    parser.add_argument("--dataset-slug", default=DATASET_SLUG)
    parser.add_argument(
        "--message",
        default="User-supplied BGE two-epoch pretrain checkpoint",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


sha256_file = checkpoint_push.sha256_file


@dataclass(frozen=True)
class FileSnapshot:
    device: int
    inode: int
    mode: int
    links: int
    size: int
    mtime_ns: int
    ctime_ns: int


def _snapshot(value: os.stat_result) -> FileSnapshot:
    return FileSnapshot(
        device=value.st_dev,
        inode=value.st_ino,
        mode=value.st_mode,
        links=value.st_nlink,
        size=value.st_size,
        mtime_ns=value.st_mtime_ns,
        ctime_ns=value.st_ctime_ns,
    )


def resolve_isolated_paths(source_dir: Path, stage_dir: Path) -> tuple[Path, Path]:
    """Resolve symlinks and reject every source/staging containment relation."""

    if source_dir.is_symlink():
        kaggle.fail(f"checkpoint source directory must not be a symlink: {source_dir}")
    if stage_dir.is_symlink():
        kaggle.fail(f"checkpoint stage directory must not be a symlink: {stage_dir}")
    try:
        resolved_source = source_dir.resolve(strict=True)
        resolved_stage = stage_dir.resolve(strict=False)
    except OSError as error:
        kaggle.fail(f"could not resolve checkpoint paths safely: {error}")
    if not resolved_source.is_dir():
        kaggle.fail(f"checkpoint source directory does not exist: {resolved_source}")
    if resolved_stage.exists() and not resolved_stage.is_dir():
        kaggle.fail(f"checkpoint stage path is not a directory: {resolved_stage}")
    if (
        resolved_source == resolved_stage
        or resolved_source in resolved_stage.parents
        or resolved_stage in resolved_source.parents
    ):
        kaggle.fail(
            "checkpoint source and stage directories must be disjoint after symlink "
            f"resolution: source={resolved_source}, stage={resolved_stage}"
        )
    return resolved_source, resolved_stage


def _read_stable_regular_file(
    path: Path,
    *,
    consumer: Callable[[bytes], None] | None = None,
    chunk_size: int = 8 * 1024 * 1024,
) -> tuple[dict[str, object], FileSnapshot]:
    """Hash a no-follow regular file and prove its identity stayed stable while read."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    try:
        path_before = os.lstat(path)
    except OSError as error:
        kaggle.fail(f"could not inspect checkpoint file {path}: {error}")
    if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(path_before.st_mode):
        kaggle.fail(f"checkpoint entry must be a non-symlink regular file: {path}")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        kaggle.fail(f"could not open checkpoint file safely {path}: {error}")
    digest = hashlib.sha256()
    total = 0
    opened: FileSnapshot | None = None
    after: FileSnapshot | None = None
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            opened = _snapshot(os.fstat(stream.fileno()))
            if not stat.S_ISREG(opened.mode):
                kaggle.fail(f"opened checkpoint entry is not a regular file: {path}")
            path_snapshot = _snapshot(path_before)
            if (opened.device, opened.inode) != (
                path_snapshot.device,
                path_snapshot.inode,
            ):
                kaggle.fail(f"checkpoint file identity changed while opening: {path}")
            while chunk := stream.read(chunk_size):
                digest.update(chunk)
                total += len(chunk)
                if consumer is not None:
                    consumer(chunk)
            after = _snapshot(os.fstat(stream.fileno()))
    finally:
        # fdopen owns the descriptor on its successful path. If fdopen itself
        # failed, closing an already-closed descriptor is harmlessly ignored.
        try:
            os.close(descriptor)
        except OSError:
            pass
    assert opened is not None and after is not None
    try:
        path_after = _snapshot(os.lstat(path))
    except OSError as error:
        kaggle.fail(f"checkpoint file disappeared while reading {path}: {error}")
    if opened != after or opened != path_after or total != opened.size:
        kaggle.fail(f"checkpoint file changed while being read: {path}")
    return {"bytes": total, "sha256": digest.hexdigest()}, opened


def _atomic_write_bytes(destination: Path, payload: bytes) -> None:
    descriptor, temporary_raw = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(destination: Path, text: str) -> None:
    _atomic_write_bytes(destination, text.encode("utf-8"))


def _copy_direct_file(
    source: Path,
    destination: Path,
    expected: Mapping[str, object],
) -> None:
    descriptor, temporary_raw = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            measured, _ = _read_stable_regular_file(
                source,
                consumer=output.write,
            )
            output.flush()
            os.fsync(output.fileno())
        if measured != dict(expected):
            kaggle.fail(
                f"checkpoint source changed before copy completed for {source.name}: "
                f"expected {dict(expected)}, got {measured}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    staged, _ = _read_stable_regular_file(destination)
    if staged != dict(expected):
        kaggle.fail(
            f"staged checkpoint copy differs for {destination.name}: "
            f"expected {dict(expected)}, got {staged}"
        )


def _normalized_expected_files(
    expected_files: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    if set(expected_files) != set(CHECKPOINT_FILES):
        kaggle.fail(
            "expected checkpoint file contract differs from the exact BGE payload; "
            f"missing={sorted(set(CHECKPOINT_FILES) - set(expected_files))}, "
            f"unexpected={sorted(set(expected_files) - set(CHECKPOINT_FILES))}"
        )
    normalized: dict[str, dict[str, object]] = {}
    for filename in CHECKPOINT_FILES:
        record = expected_files[filename]
        size = record.get("bytes")
        digest = record.get("sha256")
        if not isinstance(size, int) or size <= 0:
            kaggle.fail(f"invalid expected byte count for {filename}: {size!r}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            kaggle.fail(f"invalid expected SHA-256 for {filename}: {digest!r}")
        normalized[filename] = {"bytes": size, "sha256": digest}
    return normalized


def validate_source_checkpoint(
    source_dir: Path,
    *,
    expected_files: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    """Validate the exact five-file checkpoint and return its measured ledger."""

    if not source_dir.is_dir():
        kaggle.fail(f"checkpoint source directory does not exist: {source_dir}")
    expected = _normalized_expected_files(
        EXPECTED_SOURCE_FILES if expected_files is None else expected_files
    )
    actual_entries = {path.name for path in source_dir.iterdir()}
    required = set(CHECKPOINT_FILES)
    if actual_entries != required:
        kaggle.fail(
            "checkpoint source file set mismatch; "
            f"missing={sorted(required - actual_entries)}, "
            f"unexpected={sorted(actual_entries - required)}"
        )

    measured: dict[str, dict[str, object]] = {}
    for filename in CHECKPOINT_FILES:
        source = source_dir / filename
        record, _ = _read_stable_regular_file(source)
        if record != expected[filename]:
            kaggle.fail(
                f"checkpoint bytes differ for {filename}: "
                f"expected {expected[filename]}, got {record}"
            )
        measured[filename] = record
    return measured


def split_model(
    source: Path,
    stage_dir: Path,
    *,
    part_bytes: int = MODEL_PART_BYTES,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    if part_bytes <= 0:
        raise ValueError("part_bytes must be positive")
    parts: dict[str, dict[str, object]] = {}
    index = 0

    def stage_chunk(chunk: bytes) -> None:
        nonlocal index
        part_name = f"{MODEL_FILENAME}.part{index:03d}"
        part_hash = hashlib.sha256(chunk).hexdigest()
        destination = stage_dir / part_name
        _atomic_write_bytes(destination, chunk)
        staged, _ = _read_stable_regular_file(destination)
        expected_part = {"bytes": len(chunk), "sha256": part_hash}
        if staged != expected_part:
            kaggle.fail(
                f"staged model part differs for {part_name}: "
                f"expected {expected_part}, got {staged}"
            )
        parts[part_name] = expected_part
        index += 1

    model_record, _ = _read_stable_regular_file(
        source,
        consumer=stage_chunk,
        chunk_size=part_bytes,
    )
    if not parts:
        kaggle.fail("checkpoint model cannot be empty")
    return parts, {
        "filename": MODEL_FILENAME,
        **model_record,
        "part_bytes": part_bytes,
        "parts": list(parts),
    }


def verify_staged_payload(
    stage_dir: Path,
    *,
    files: Mapping[str, Mapping[str, object]],
    checkpoint_files: Mapping[str, Mapping[str, object]],
    reconstruction: Mapping[str, object],
) -> None:
    """Rehash staged independent copies and reconstruct the original model digest."""

    normalized_checkpoint = _normalized_expected_files(checkpoint_files)
    raw_parts = reconstruction.get("parts")
    if (
        reconstruction.get("filename") != MODEL_FILENAME
        or not isinstance(raw_parts, list)
        or not raw_parts
        or any(not isinstance(name, str) for name in raw_parts)
        or len(set(raw_parts)) != len(raw_parts)
    ):
        kaggle.fail("invalid staged model reconstruction contract")
    parts = [str(name) for name in raw_parts]
    expected_file_names = set(DIRECT_CHECKPOINT_FILES) | set(parts)
    if set(files) != expected_file_names:
        kaggle.fail(
            "staged payload ledger has an unexpected file set; "
            f"missing={sorted(expected_file_names - set(files))}, "
            f"unexpected={sorted(set(files) - expected_file_names)}"
        )

    for filename in DIRECT_CHECKPOINT_FILES:
        expected = normalized_checkpoint[filename]
        if dict(files[filename]) != expected:
            kaggle.fail(f"staged direct-file ledger differs for {filename}")
        measured, _ = _read_stable_regular_file(stage_dir / filename)
        if measured != expected:
            kaggle.fail(
                f"staged direct file differs for {filename}: "
                f"expected {expected}, got {measured}"
            )

    reconstructed_digest = hashlib.sha256()
    reconstructed_bytes = 0
    for part_name in parts:
        expected_part = dict(files[part_name])

        def update_model_digest(chunk: bytes) -> None:
            reconstructed_digest.update(chunk)

        measured, _ = _read_stable_regular_file(
            stage_dir / part_name,
            consumer=update_model_digest,
        )
        if measured != expected_part:
            kaggle.fail(
                f"staged model part differs for {part_name}: "
                f"expected {expected_part}, got {measured}"
            )
        reconstructed_bytes += int(measured["bytes"])

    model_expected = normalized_checkpoint[MODEL_FILENAME]
    measured_reconstruction = {
        "bytes": reconstructed_bytes,
        "sha256": reconstructed_digest.hexdigest(),
    }
    declared_reconstruction = {
        "bytes": reconstruction.get("bytes"),
        "sha256": reconstruction.get("sha256"),
    }
    if (
        measured_reconstruction != model_expected
        or declared_reconstruction != model_expected
    ):
        kaggle.fail(
            "staged model reconstruction differs from the frozen checkpoint: "
            f"expected {model_expected}, measured {measured_reconstruction}, "
            f"declared {declared_reconstruction}"
        )


def _assert_direct_copies_are_isolated(source_dir: Path, stage_dir: Path) -> None:
    for filename in DIRECT_CHECKPOINT_FILES:
        source = _snapshot(os.lstat(source_dir / filename))
        staged = _snapshot(os.lstat(stage_dir / filename))
        if (source.device, source.inode) == (staged.device, staged.inode):
            kaggle.fail(f"staged direct file is a hardlink to its source: {filename}")


def _validate_complete_stage_file_set(
    stage_dir: Path,
    payload_files: Mapping[str, object],
) -> None:
    expected = set(payload_files) | {MANIFEST_NAME, "dataset-metadata.json"}
    actual = {path.name for path in stage_dir.iterdir()}
    if actual != expected:
        kaggle.fail(
            "staged checkpoint file set mismatch; "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    for filename in (MANIFEST_NAME, "dataset-metadata.json"):
        mode = os.lstat(stage_dir / filename).st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            kaggle.fail(f"staged authority file is not a regular file: {filename}")


def verify_payload_for_upload(
    stage_dir: Path,
    manifest: Mapping[str, object],
    *,
    expected_checkpoint_files: Mapping[str, Mapping[str, object]],
) -> str:
    """Final frozen-contract verification immediately before a remote upload."""

    frozen = _normalized_expected_files(expected_checkpoint_files)
    checkpoint_files = manifest.get("checkpoint_files")
    files = manifest.get("files")
    reconstruction = manifest.get("reconstruction")
    if (
        not isinstance(checkpoint_files, Mapping)
        or not isinstance(files, Mapping)
        or not isinstance(reconstruction, Mapping)
        or checkpoint_files != frozen
    ):
        kaggle.fail("upload manifest does not bind the frozen checkpoint contract")
    verify_staged_payload(
        stage_dir,
        files=files,
        checkpoint_files=checkpoint_files,
        reconstruction=reconstruction,
    )
    _validate_complete_stage_file_set(stage_dir, files)
    manifest_path = stage_dir / MANIFEST_NAME
    try:
        disk_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        kaggle.fail(f"could not reload staged checkpoint manifest: {error}")
    if disk_manifest != manifest:
        kaggle.fail("staged checkpoint manifest differs from the verified payload")
    metadata = json.loads(
        (stage_dir / "dataset-metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("isPrivate") is not True or metadata.get("id") != manifest.get(
        "dataset"
    ):
        kaggle.fail("staged Dataset metadata is not the expected private Dataset")
    return sha256_file(manifest_path)


def build_payload(
    source_dir: Path,
    stage_dir: Path,
    owner: str,
    *,
    dataset_slug: str = DATASET_SLUG,
    expected_files: Mapping[str, Mapping[str, object]] | None = None,
    model_part_bytes: int | None = None,
) -> dict[str, object]:
    source_dir, stage_dir = resolve_isolated_paths(source_dir, stage_dir)
    dataset_slug = kaggle.validate_slug(dataset_slug, "dataset slug")
    measured_checkpoint_files = validate_source_checkpoint(
        source_dir,
        expected_files=expected_files,
    )
    part_bytes = MODEL_PART_BYTES if model_part_bytes is None else model_part_bytes
    stage_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, dict[str, object]] = {}
    for filename in DIRECT_CHECKPOINT_FILES:
        source = source_dir / filename
        _copy_direct_file(
            source,
            stage_dir / filename,
            measured_checkpoint_files[filename],
        )
        files[filename] = dict(measured_checkpoint_files[filename])

    # Upload deterministic 64 MiB pieces rather than one 2.1 GiB object. The
    # training notebook must reconstruct and verify the original SHA before load.
    (stage_dir / MODEL_FILENAME).unlink(missing_ok=True)
    model_parts, model_reconstruction = split_model(
        source_dir / MODEL_FILENAME,
        stage_dir,
        part_bytes=part_bytes,
    )
    expected_model = measured_checkpoint_files[MODEL_FILENAME]
    if (
        model_reconstruction["bytes"] != expected_model["bytes"]
        or model_reconstruction["sha256"] != expected_model["sha256"]
    ):
        kaggle.fail("sharded model reconstruction ledger differs from the source model")
    files.update(model_parts)
    _assert_direct_copies_are_isolated(source_dir, stage_dir)

    # This is the first full rehash, immediately before manifest publication.
    verify_staged_payload(
        stage_dir,
        files=files,
        checkpoint_files=measured_checkpoint_files,
        reconstruction=model_reconstruction,
    )

    dataset_ref = f"{owner}/{dataset_slug}"
    try:
        source_label = source_dir.relative_to(ROOT).as_posix()
    except ValueError:
        source_label = "external_user_supplied_checkpoint"
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset": dataset_ref,
        "is_private": True,
        "source": source_label,
        "purpose": "initialize fresh supervised fine-tuning from the supplied checkpoint",
        "checkpoint_files": measured_checkpoint_files,
        "files": files,
        "reconstruction": model_reconstruction,
        "provenance": {
            "status": "user_supplied_unverified",
            "source_type": "local_user_supplied_checkpoint",
            "declared_model_family": "BAAI/bge-reranker-v2-m3",
            "declared_pretraining_epochs": 2,
            "training_lineage_verified": False,
            "pretraining_data_verified": False,
            "base_model_revision_verified": False,
            "verified_scope": "exact checkpoint file sizes and SHA-256 only",
        },
    }
    _atomic_write_text(
        stage_dir / MANIFEST_NAME,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    metadata = {
        "title": "Product Matching BGE Pretrain 2ep",
        "id": dataset_ref,
        "licenses": [{"name": "unknown"}],
        "isPrivate": True,
        "description": (
            "Private user-supplied BGE sequence-classification checkpoint declared "
            "as two-epoch task pretraining. Exact bytes are pinned, while training "
            "lineage, source data and base revision remain unverified. The 2.1 GiB "
            "safetensors file is losslessly sharded into 64 MiB pieces."
        ),
    }
    _atomic_write_text(
        stage_dir / "dataset-metadata.json",
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    )
    _validate_complete_stage_file_set(stage_dir, files)
    # Rehash again after publishing both authority files. Dry-run returns only
    # after this check, so it cannot report a mismatched manifest/stage pair.
    verify_staged_payload(
        stage_dir,
        files=files,
        checkpoint_files=measured_checkpoint_files,
        reconstruction=model_reconstruction,
    )
    return manifest


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
    source_dir, stage_dir = resolve_isolated_paths(source_dir, stage_dir)
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

    # Fail before even resolving the CLI if the local payload drifted after build.
    manifest_hash = verify_payload_for_upload(
        stage_dir,
        manifest,
        expected_checkpoint_files=EXPECTED_SOURCE_FILES,
    )
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
    # The read-only status call above can take time. Bind every staged byte once
    # more immediately before the first mutating Kaggle subprocess.
    manifest_hash = verify_payload_for_upload(
        stage_dir,
        manifest,
        expected_checkpoint_files=EXPECTED_SOURCE_FILES,
    )
    print("$", " ".join(command), flush=True)
    upload = subprocess.run(command, cwd=ROOT, env=os.environ.copy())
    if upload.returncode:
        kaggle.fail(
            f"checkpoint Dataset upload failed with exit code {upload.returncode}",
            upload.returncode,
        )
    status = shared_push.wait_until_ready(
        cli,
        dataset_ref,
        minimum_version=previous_version + 1,
    )
    checkpoint_push.verify_remote_dataset(
        cli,
        dataset_ref,
        manifest_hash,
        set(manifest["files"]),
    )
    print(
        f"Private checkpoint Dataset is ready at version "
        f"{status.get('current_version_number')}: "
        f"https://www.kaggle.com/datasets/{dataset_ref}"
    )


if __name__ == "__main__":
    main()
