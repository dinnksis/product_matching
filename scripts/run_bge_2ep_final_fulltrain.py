#!/usr/bin/env python3
"""One-shot local controller for the final BGE full-human export kernel."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import create_bge_2ep_final_fulltrain_notebook as builder
import push_kaggle_training_dataset as dataset_push
import run_bge_2ep_sft_kaggle as campaign_runner
import run_kaggle_notebook as kaggle


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DATASET = "alexproger23/product-matching-validation-splits-v1"
CHECKPOINT_DATASET = "alexproger23/product-matching-bge-pretrain-2ep"
CREDENTIALS_DATASET = "alexproger23/ecom-matching-google-sheets-credentials"
REQUIRED_DATASETS = (VALIDATION_DATASET, CHECKPOINT_DATASET, CREDENTIALS_DATASET)
RUN_TIMEOUT_SECONDS = 32_400
DEFAULT_WAIT_TIMEOUT_SECONDS = 36_000
POLL_INTERVAL_SECONDS = 30
READY_DATASET_STATUSES = frozenset({"ready", "complete", "successful"})
LOCK_PATH = ROOT / ".kaggle/locks/bge_2ep_final_fulltrain_export_v1.lock"
ATTEMPT_LEDGER_PATH = (
    ROOT / ".kaggle/audit/bge_2ep_final_fulltrain_export_v1/attempted_kernel_slugs.json"
)
TOMBSTONED_KERNEL_SLUGS = frozenset(
    {
        # Local-only provisional identities superseded before any Kaggle push.
        "pm-b2-final-b01068d9adf4-s42-v1",
        "pm-b2-final-22afdecb5a9b-s42-v1",
        "pm-b2-final-3c5b43867641-s42-v1",
        "pm-b2-final-20d1a7ada6d5-s42-v1",
        "pm-b2-final-fb760179c0f3-s42-v1",
        "pm-b2-final-f8011cbe984a-s42-v1",
    }
)
EXPECTED_VALIDATION_DATASET_VERSION = 3
EXPECTED_CHECKPOINT_DATASET_VERSION = 1
EXPECTED_VALIDATION_REMOTE_FILES = frozenset(
    {
        "validation_splits_manifest.json",
        "human_items.parquet",
        "human_hard_selection_details.parquet",
        "human_hard_validation_pairs.parquet",
        "human_iid_validation_pairs.parquet",
        "human_ood_validation_pairs.parquet",
        "human_split_assignments.parquet",
        "human_train_pairs.parquet",
        "llm_non_ood_items.parquet",
        "llm_non_ood_pairs.parquet",
        "llm_ood_items.parquet",
        "llm_ood_pairs.parquet",
        "upload_manifest.json",
    }
)


class FinalExportWorkflowError(RuntimeError):
    """Raised when the single-run export cannot proceed safely."""


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_regular_file_record(path: Path, *, root: Path) -> dict[str, Any]:
    record, _ = _stable_regular_file(path, root=root, collect=False)
    return record


def _stable_regular_file(
    path: Path, *, root: Path, collect: bool
) -> tuple[dict[str, Any], bytes | None]:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise FinalExportWorkflowError(f"Artifact is outside its root: {path}") from error
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise FinalExportWorkflowError(f"Artifact is not an isolated regular file: {path}")
    resolved_root = root.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    if not resolved_path.is_relative_to(resolved_root):
        raise FinalExportWorkflowError(f"Artifact escapes its root: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    chunks: list[bytes] | None = [] if collect else None
    digest = hashlib.sha256()
    try:
        opened_before = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.lstat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_mode,
        before.st_nlink,
    )
    identity_opened_before = (
        opened_before.st_dev,
        opened_before.st_ino,
        opened_before.st_size,
        opened_before.st_mtime_ns,
        opened_before.st_ctime_ns,
        opened_before.st_mode,
        opened_before.st_nlink,
    )
    identity_opened_after = (
        opened_after.st_dev,
        opened_after.st_ino,
        opened_after.st_size,
        opened_after.st_mtime_ns,
        opened_after.st_ctime_ns,
        opened_after.st_mode,
        opened_after.st_nlink,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_mode,
        after.st_nlink,
    )
    if not (
        identity_before
        == identity_opened_before
        == identity_opened_after
        == identity_after
    ):
        raise FinalExportWorkflowError(f"Artifact changed while hashing: {path}")
    record = {
        "path": relative.as_posix(),
        "bytes": before.st_size,
        "sha256": digest.hexdigest(),
    }
    return record, b"".join(chunks) if chunks is not None else None


def load_stable_json(path: Path, *, root: Path) -> dict[str, Any]:
    _, payload = _stable_regular_file(path, root=root, collect=True)
    assert payload is not None
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalExportWorkflowError(f"Invalid stable JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise FinalExportWorkflowError(f"Stable JSON artifact is not an object: {path}")
    return value


def scan_regular_tree(root: Path) -> tuple[set[Path], set[Path]]:
    root_stat = root.lstat()
    if not stat.S_ISDIR(root_stat.st_mode) or root.is_symlink():
        raise FinalExportWorkflowError(f"Artifact root is not a real directory: {root}")
    directories: set[Path] = set()
    files: set[Path] = set()
    for current_raw, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_raw)
        for name in directory_names:
            path = current / name
            mode = path.lstat().st_mode
            if not stat.S_ISDIR(mode) or path.is_symlink():
                raise FinalExportWorkflowError(f"Unsafe directory entry: {path}")
            directories.add(path)
        for name in file_names:
            path = current / name
            details = path.lstat()
            if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
                raise FinalExportWorkflowError(f"Unsafe file entry: {path}")
            files.add(path)
    return directories, files


def runner_command(
    entry: Mapping[str, Any], *, env_file: Path, dry_run: bool
) -> list[str]:
    validate_frozen_notebook_file(Path(entry["notebook"]), entry=entry)
    command = [
        os.fspath(Path(os.sys.executable)),
        str(ROOT / "scripts/run_kaggle_notebook.py"),
        str(entry["notebook"]),
        "--env-file",
        str(env_file),
        "--slug",
        str(entry["kernel_slug"]),
        "--title",
        str(entry["title"]),
        "--dataset",
        str(entry["validation_dataset"]),
        "--dataset",
        str(entry["checkpoint_dataset"]),
        "--no-env-sources",
        "--no-gpu-check",
        "--no-download",
    ]
    if dry_run:
        command.append("--dry-run")
    else:
        command.append("--no-wait")
    return command


def load_frozen_entry(owner: str) -> dict[str, Any]:
    """Resolve current authorities but never rewrite the reviewed notebook."""
    entry = builder.build_notebook(owner=owner, write=False)
    validate_frozen_notebook_file(Path(entry["notebook"]), entry=entry)
    return entry


def validate_frozen_notebook_file(
    path: Path, *, entry: Mapping[str, Any]
) -> dict[str, Any]:
    """Read one reviewed notebook through the no-follow stable-file gate."""
    record, payload = _stable_regular_file(path, root=path.parent, collect=True)
    if record["sha256"] != entry.get("notebook_sha256"):
        raise FinalExportWorkflowError("Frozen notebook file SHA-256 differs")
    assert payload is not None
    try:
        notebook = builder.nbf.reads(payload.decode("utf-8"), as_version=4)
    except Exception as error:
        # nbformat raises several JSON/schema exception types across versions.
        raise FinalExportWorkflowError("Frozen notebook cannot be parsed") from error
    builder.validate_notebook_identity(notebook, entry=entry)
    return record


def expected_dataset_sources(entry: Mapping[str, Any]) -> list[str]:
    return [
        str(entry["validation_dataset"]),
        str(entry["checkpoint_dataset"]),
        CREDENTIALS_DATASET,
    ]


def validate_staged_kernel_metadata(entry: Mapping[str, Any]) -> dict[str, Any]:
    stage_dir = kaggle.STAGE_ROOT / str(entry["kernel_slug"])
    metadata_path = stage_dir / "kernel-metadata.json"
    notebook_path = stage_dir / "notebook.ipynb"
    directories, files = scan_regular_tree(stage_dir)
    if directories or files != {metadata_path, notebook_path}:
        raise FinalExportWorkflowError(
            "Dry-run staging tree differs from the exact two-file allowlist"
        )
    metadata = load_stable_json(metadata_path, root=stage_dir)
    exact = {
        "id": f"alexproger23/{entry['kernel_slug']}",
        "title": entry["title"],
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": expected_dataset_sources(entry),
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    if metadata != exact:
        raise FinalExportWorkflowError(
            "Staged final-export metadata differs: "
            + json.dumps(
                {"actual": metadata, "expected": exact},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    validate_frozen_notebook_file(notebook_path, entry=entry)
    return metadata


def stage_locally(
    entry: Mapping[str, Any], *, env_file: Path
) -> dict[str, Any]:
    variable = "KAGGLE_GOOGLE_SHEETS_CREDENTIALS_DATASET"
    previous = os.environ.get(variable)
    os.environ[variable] = CREDENTIALS_DATASET
    try:
        kaggle.run_command(runner_command(entry, env_file=env_file, dry_run=True))
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous
    return validate_staged_kernel_metadata(entry)


@contextmanager
def exclusive_lock(path: Path = LOCK_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise FinalExportWorkflowError(
                "Another final-export controller holds the local lock"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "time": time.time()}) + "\n")
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def empty_attempt_ledger() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign": builder.CAMPAIGN,
        "policy": "one_push_per_identity_no_automatic_resubmit_v1",
        "attempts": [],
    }


def load_attempt_ledger(path: Path = ATTEMPT_LEDGER_PATH) -> dict[str, Any]:
    if not path.exists():
        return empty_attempt_ledger()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or {
        key: payload.get(key) for key in ("schema_version", "campaign", "policy")
    } != {
        "schema_version": 1,
        "campaign": builder.CAMPAIGN,
        "policy": "one_push_per_identity_no_automatic_resubmit_v1",
    }:
        raise FinalExportWorkflowError("Final-export attempt ledger header differs")
    attempts = payload.get("attempts")
    if not isinstance(attempts, list):
        raise FinalExportWorkflowError("Final-export attempts must be a list")
    seen: set[str] = set()
    for record in attempts:
        if not isinstance(record, dict):
            raise FinalExportWorkflowError("Final-export attempt is not an object")
        slug = record.get("kernel_slug")
        identity = record.get("identity_sha256")
        status = record.get("status")
        if (
            not isinstance(slug, str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug)
            or not isinstance(identity, str)
            or not re.fullmatch(r"[0-9a-f]{64}", identity)
            or status
            not in {
                "push_pending",
                "submitted",
                "running",
                "complete",
                "failed",
            }
        ):
            raise FinalExportWorkflowError("Final-export attempt record differs")
        if slug in seen:
            raise FinalExportWorkflowError("Final-export ledger repeats a kernel slug")
        seen.add(slug)
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(path)


def reserve_single_attempt(
    entry: Mapping[str, Any], *, path: Path = ATTEMPT_LEDGER_PATH
) -> dict[str, Any]:
    ledger = load_attempt_ledger(path)
    attempts = ledger["attempts"]
    current_slug = str(entry["kernel_slug"])
    if any(record["kernel_slug"] == current_slug for record in attempts):
        raise FinalExportWorkflowError(
            "This final-export identity already consumed its only push attempt"
        )
    untombstoned_prior = [
        record["kernel_slug"]
        for record in attempts
        if record["kernel_slug"] not in TOMBSTONED_KERNEL_SLUGS
    ]
    if untombstoned_prior:
        raise FinalExportWorkflowError(
            "A prior identity must be explicitly tombstoned before a new iteration: "
            + repr(untombstoned_prior)
        )
    if current_slug in TOMBSTONED_KERNEL_SLUGS:
        raise FinalExportWorkflowError("Current final-export slug is tombstoned")
    attempts.append(
        {
            "kernel_slug": current_slug,
            "identity_sha256": entry["identity_sha256"],
            "status": "push_pending",
            "reserved_at_unix": time.time(),
        }
    )
    _atomic_write_json(path, ledger)
    return ledger


def update_attempt_status(
    entry: Mapping[str, Any],
    status: str,
    *,
    path: Path = ATTEMPT_LEDGER_PATH,
) -> dict[str, Any]:
    if status not in {"submitted", "running", "complete", "failed"}:
        raise FinalExportWorkflowError(f"Unsupported attempt status: {status}")
    ledger = load_attempt_ledger(path)
    records = [
        record
        for record in ledger["attempts"]
        if record["kernel_slug"] == entry["kernel_slug"]
    ]
    if len(records) != 1:
        raise FinalExportWorkflowError("Cannot update an unreserved final-export attempt")
    records[0]["status"] = status
    records[0]["updated_at_unix"] = time.time()
    _atomic_write_json(path, ledger)
    return ledger


def require_reserved_attempt(entry: Mapping[str, Any]) -> dict[str, Any]:
    ledger = load_attempt_ledger()
    records = [
        record
        for record in ledger["attempts"]
        if record["kernel_slug"] == entry["kernel_slug"]
        and record["identity_sha256"] == entry["identity_sha256"]
    ]
    if len(records) != 1:
        raise FinalExportWorkflowError(
            "Monitor mode requires the exact locally reserved attempt"
        )
    return records[0]


def verify_remote_sources(
    cli: list[str], *, kernel_ref: str, entry: Mapping[str, Any]
) -> None:
    with tempfile.TemporaryDirectory(prefix="bge-final-kernel-metadata-") as temp_dir:
        result = kaggle.run_command(
            cli + ["kernels", "pull", kernel_ref, "-p", temp_dir, "-m"],
            check=False,
        )
        if result.returncode:
            raise FinalExportWorkflowError("Could not pull final-export metadata")
        metadata_path = Path(temp_dir) / "kernel-metadata.json"
        if not metadata_path.is_file():
            raise FinalExportWorkflowError("Remote final-export metadata is missing")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual = metadata.get("dataset_sources")
    expected = expected_dataset_sources(entry)
    if (
        not isinstance(actual, list)
        or len(actual) != len(expected)
        or len(set(actual)) != len(actual)
        or set(actual) != set(expected)
    ):
        raise FinalExportWorkflowError(
            "Remote final-export Dataset attachments differ: "
            + json.dumps({"actual": actual, "expected": expected}, sort_keys=True)
        )
    exact = {
        "id": kernel_ref,
        "title": entry["title"],
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    mismatches = {
        key: {"actual": metadata.get(key), "expected": value}
        for key, value in exact.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise FinalExportWorkflowError(
            "Remote final-export kernel metadata differs: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def _remote_file_names(payload: object) -> set[str]:
    rows: object = payload
    if isinstance(payload, Mapping):
        rows = payload.get("datasetFiles") or payload.get("files") or []
    if not isinstance(rows, list):
        raise FinalExportWorkflowError("Remote Dataset file listing is not a list")
    names: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise FinalExportWorkflowError("Remote Dataset file row is invalid")
        raw_name = row.get("name") or row.get("ref")
        if not isinstance(raw_name, str) or not raw_name:
            raise FinalExportWorkflowError("Remote Dataset file has no name")
        raw_path = Path(raw_name)
        if (
            raw_path.is_absolute()
            or raw_name in {".", ".."}
            or "/" in raw_name
            or "\\" in raw_name
            or raw_path.parent != Path(".")
            or raw_path.name != raw_name
        ):
            raise FinalExportWorkflowError(
                "Remote Dataset file is not an exact flat-root name"
            )
        name = raw_name
        if name in names:
            raise FinalExportWorkflowError("Remote Dataset file listing has duplicates")
        names.add(name)
    return names


def audit_remote_dataset(
    cli: list[str],
    *,
    dataset_ref: str,
    expected_version: int,
    expected_files: set[str] | frozenset[str],
    manifest_filename: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Read status/files/manifest/privacy and re-read status before mutation."""

    def exact_status() -> dict[str, Any]:
        status = dataset_push.dataset_status(cli, dataset_ref)
        if not isinstance(status, dict):
            raise FinalExportWorkflowError(
                f"Could not read remote Dataset status: {dataset_ref}"
            )
        try:
            version = int(status.get("current_version_number", 0))
        except (TypeError, ValueError) as error:
            raise FinalExportWorkflowError("Remote Dataset version is invalid") from error
        state = str(status.get("status", "")).casefold()
        if version != expected_version:
            raise FinalExportWorkflowError(
                f"Remote Dataset version differs for {dataset_ref}: "
                f"{version} != {expected_version}"
            )
        if state not in READY_DATASET_STATUSES:
            raise FinalExportWorkflowError(
                f"Remote Dataset is not ready: {dataset_ref} ({status.get('status')!r})"
            )
        for key in ("ref", "id", "dataset_ref"):
            if key in status and status[key] not in (None, "", dataset_ref):
                raise FinalExportWorkflowError(
                    f"Remote Dataset status identity differs: {dataset_ref}"
                )
        return status

    exact_status()
    listing = kaggle.run_command(
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
    try:
        listing_payload = json.loads(listing.stdout)
    except json.JSONDecodeError as error:
        raise FinalExportWorkflowError("Remote Dataset files are not JSON") from error
    observed_files = _remote_file_names(listing_payload)
    if observed_files != set(expected_files):
        raise FinalExportWorkflowError(
            f"Remote Dataset file set differs for {dataset_ref}: "
            f"missing={sorted(set(expected_files) - observed_files)}, "
            f"unexpected={sorted(observed_files - set(expected_files))}"
        )
    with tempfile.TemporaryDirectory(prefix="bge-final-dataset-manifest-") as raw:
        destination = Path(raw)
        kaggle.run_command(
            cli
            + [
                "datasets",
                "download",
                dataset_ref,
                "-f",
                manifest_filename,
                "-p",
                str(destination),
                "-o",
                "-q",
            ]
        )
        manifest_path = destination / manifest_filename
        if not manifest_path.is_file() or file_sha256(manifest_path) != manifest_sha256:
            raise FinalExportWorkflowError(
                f"Remote Dataset manifest differs for {dataset_ref}"
            )
    with tempfile.TemporaryDirectory(prefix="bge-final-dataset-metadata-") as raw:
        destination = Path(raw)
        kaggle.run_command(
            cli + ["datasets", "metadata", dataset_ref, "--path", str(destination)]
        )
        metadata_path = destination / "dataset-metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        info = metadata.get("info", metadata)
        if not isinstance(info, Mapping) or info.get("isPrivate") is not True:
            raise FinalExportWorkflowError(
                f"Remote Dataset is not private: {dataset_ref}"
            )
    final_status = exact_status()
    return {
        "dataset_ref": dataset_ref,
        "dataset_version": expected_version,
        "manifest_sha256": manifest_sha256,
        "files": sorted(observed_files),
        "private": True,
        "status": final_status.get("status"),
    }


def audit_remote_training_inputs(
    cli: list[str], *, entry: Mapping[str, Any]
) -> dict[str, Any]:
    checkpoint_manifest = builder.base.load_checkpoint_dataset(
        builder.DEFAULT_CHECKPOINT_STAGE_DIR,
        "alexproger23",
        verify_payload=True,
    )["manifest"]
    checkpoint_files = set(checkpoint_manifest["files"]) | {"checkpoint_manifest.json"}
    validation = audit_remote_dataset(
        cli,
        dataset_ref=str(entry["validation_dataset"]),
        expected_version=EXPECTED_VALIDATION_DATASET_VERSION,
        expected_files=EXPECTED_VALIDATION_REMOTE_FILES,
        manifest_filename="validation_splits_manifest.json",
        manifest_sha256=str(entry["validation_manifest_sha256"]),
    )
    checkpoint = audit_remote_dataset(
        cli,
        dataset_ref=str(entry["checkpoint_dataset"]),
        expected_version=EXPECTED_CHECKPOINT_DATASET_VERSION,
        expected_files=checkpoint_files,
        manifest_filename="checkpoint_manifest.json",
        manifest_sha256=str(entry["checkpoint_manifest_sha256"]),
    )
    credentials = audit_remote_credentials_dataset(cli)
    return {
        "validation": validation,
        "checkpoint": checkpoint,
        "credentials": credentials,
    }


def audit_remote_credentials_dataset(cli: list[str]) -> dict[str, Any]:
    """Verify the credential attachment without downloading secret contents."""

    def exact_status() -> dict[str, Any]:
        status = dataset_push.dataset_status(cli, CREDENTIALS_DATASET)
        if not isinstance(status, dict):
            raise FinalExportWorkflowError("Could not read credentials Dataset status")
        try:
            version = int(status.get("current_version_number", 0))
        except (TypeError, ValueError) as error:
            raise FinalExportWorkflowError(
                "Credentials Dataset version is invalid"
            ) from error
        state = str(status.get("status", "")).strip().casefold()
        if version < 1 or state not in READY_DATASET_STATUSES:
            raise FinalExportWorkflowError(
                "Credentials Dataset is not in an exact ready state"
            )
        for key in ("ref", "id", "dataset_ref"):
            if key in status and status[key] not in (None, "", CREDENTIALS_DATASET):
                raise FinalExportWorkflowError("Credentials Dataset identity differs")
        return status

    first_status = exact_status()
    listing = kaggle.run_command(
        cli
        + [
            "datasets",
            "files",
            CREDENTIALS_DATASET,
            "--format",
            "json",
            "--page-size",
            "10",
        ]
    )
    try:
        files = _remote_file_names(json.loads(listing.stdout))
    except json.JSONDecodeError as error:
        raise FinalExportWorkflowError(
            "Credentials Dataset file listing is not JSON"
        ) from error
    if files != {"google-service-account.json"}:
        raise FinalExportWorkflowError("Credentials Dataset file set differs")
    with tempfile.TemporaryDirectory(prefix="bge-final-credentials-metadata-") as raw:
        destination = Path(raw)
        kaggle.run_command(
            cli
            + [
                "datasets",
                "metadata",
                CREDENTIALS_DATASET,
                "--path",
                str(destination),
            ]
        )
        metadata = json.loads(
            (destination / "dataset-metadata.json").read_text(encoding="utf-8")
        )
        info = metadata.get("info", metadata)
        if not isinstance(info, Mapping) or info.get("isPrivate") is not True:
            raise FinalExportWorkflowError("Credentials Dataset is not private")
    final_status = exact_status()
    if int(first_status["current_version_number"]) != int(
        final_status["current_version_number"]
    ):
        raise FinalExportWorkflowError("Credentials Dataset version changed during audit")
    return {
        "dataset_ref": CREDENTIALS_DATASET,
        "dataset_version": int(final_status["current_version_number"]),
        "files": ["google-service-account.json"],
        "private": True,
        "status": final_status["status"],
        "contents_downloaded": False,
    }


def recheck_remote_input_statuses(
    cli: list[str], authority: Mapping[str, Mapping[str, Any]]
) -> None:
    """Compact final readiness/version read after the three full audits."""
    for key in ("validation", "checkpoint", "credentials"):
        expected = authority[key]
        dataset_ref = str(expected["dataset_ref"])
        status = dataset_push.dataset_status(cli, dataset_ref)
        if not isinstance(status, dict):
            raise FinalExportWorkflowError(
                f"Final Dataset status re-read failed: {dataset_ref}"
            )
        try:
            version = int(status.get("current_version_number", 0))
        except (TypeError, ValueError) as error:
            raise FinalExportWorkflowError(
                f"Final Dataset version re-read failed: {dataset_ref}"
            ) from error
        state = str(status.get("status", "")).strip().casefold()
        if version != int(expected["dataset_version"]) or state not in READY_DATASET_STATUSES:
            raise FinalExportWorkflowError(
                f"Dataset changed after full pre-push audit: {dataset_ref}"
            )
        for identity_key in ("ref", "id", "dataset_ref"):
            if identity_key in status and status[identity_key] not in (
                None,
                "",
                dataset_ref,
            ):
                raise FinalExportWorkflowError(
                    f"Dataset identity changed after audit: {dataset_ref}"
                )


def validate_artifact_tree(
    directory: Path, *, entry: Mapping[str, Any]
) -> dict[str, Any]:
    download_directories, download_files = scan_regular_tree(directory)
    completions = [
        path for path in download_files if path.name == "notebook_completed.json"
    ]
    if len(completions) != 1:
        raise FinalExportWorkflowError(
            f"Expected one notebook_completed.json, found {completions}"
        )
    completion_path = completions[0]
    completion = load_stable_json(completion_path, root=directory)
    exact_completion = {
        "schema_version": 1,
        "status": "complete",
        "experiment": builder.EXPERIMENT,
        "campaign": builder.CAMPAIGN,
        "purpose": "final_deployment_export",
        "experiment_group": "sft",
        "quality_evaluation": False,
        "validation_splits": [],
        "validation_predictions_written": False,
        "dataset_ref": entry["validation_dataset"],
        "validation_manifest_sha256": entry["validation_manifest_sha256"],
        "initial_checkpoint_ref": entry["checkpoint_dataset"],
        "initial_checkpoint_manifest_sha256": entry["checkpoint_manifest_sha256"],
        "initial_checkpoint_model_sha256": entry["checkpoint_model_sha256"],
        "code_bundle_sha256": entry["source_sha256"],
        "frozen_recipe_sha256": entry["recipe_sha256"],
        "campaign_identity_sha256": entry["identity_sha256"],
        "executable_cells_sha256": entry["executable_cells_sha256"],
        "loss_variant": "bce_finite_guard_final_fulltrain_v1",
        "loss_hook_sha256": entry["loss_hook_sha256"],
        "artifact_directory": builder.EXPERIMENT,
        "artifact_file_count": 12,
    }
    mismatches = {
        key: {"actual": completion.get(key), "expected": value}
        for key, value in exact_completion.items()
        if completion.get(key) != value
    }
    if mismatches:
        raise FinalExportWorkflowError(
            "Final completion identity differs: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    if not isinstance(completion.get("run_id"), str) or not re.fullmatch(
        r"[0-9a-f]{32}", completion["run_id"]
    ):
        raise FinalExportWorkflowError("Final completion run_id is invalid")
    expected_receipts = {
        path.as_posix(): digest for path, digest in builder.SELECTION_RECEIPTS
    }
    if completion.get("selection_receipt_sha256") != expected_receipts:
        raise FinalExportWorkflowError("Final completion selection receipts differ")

    artifact_dir = completion_path.parent / str(completion["artifact_directory"])
    if artifact_dir not in download_directories:
        raise FinalExportWorkflowError("Final artifact directory is missing")
    artifact_directories, artifact_files = scan_regular_tree(artifact_dir)
    manifest_path = artifact_dir / "artifact_manifest.json"
    if manifest_path not in artifact_files:
        raise FinalExportWorkflowError("Final artifact manifest is missing")
    manifest_record = stable_regular_file_record(manifest_path, root=artifact_dir)
    if manifest_record["sha256"] != completion.get("artifact_manifest_sha256"):
        raise FinalExportWorkflowError("Final artifact manifest hash differs")
    manifest = load_stable_json(manifest_path, root=artifact_dir)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("root") != builder.EXPERIMENT
        or manifest.get("campaign_identity_sha256") != entry["identity_sha256"]
        or manifest.get("file_count") != 12
    ):
        raise FinalExportWorkflowError("Final artifact manifest header differs")
    records = manifest.get("files")
    if not isinstance(records, list) or len(records) != 12:
        raise FinalExportWorkflowError("Final artifact manifest records differ")
    expected_paths = {
        "model/config.json",
        "model/model.safetensors",
        "model/special_tokens_map.json",
        "model/tokenizer.json",
        "model/tokenizer_config.json",
        "training_config.json",
        "training_summary.json",
        "training_report.json",
        "deployment_smoke.json",
        "train_data_report.json",
        "memory_preflight.json",
        "runtime_versions.json",
    }
    declared_paths: set[str] = set()
    normalized_records: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise FinalExportWorkflowError("Final artifact record schema differs")
        relative = record["path"]
        if (
            not isinstance(relative, str)
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in declared_paths
        ):
            raise FinalExportWorkflowError("Final artifact path is unsafe or duplicated")
        declared_paths.add(relative)
        path = artifact_dir / relative
        size = record["bytes"]
        digest = record["sha256"]
        actual_record = stable_regular_file_record(path, root=artifact_dir)
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or actual_record != record
        ):
            raise FinalExportWorkflowError(f"Final artifact differs: {relative}")
        normalized_records.append(dict(record))
    if declared_paths != expected_paths:
        raise FinalExportWorkflowError("Final artifact allowlist differs")
    actual_paths = {
        path.relative_to(artifact_dir).as_posix() for path in artifact_files
    }
    if actual_paths != expected_paths | {"artifact_manifest.json"}:
        raise FinalExportWorkflowError("Final artifact tree has undeclared files")
    actual_directories = {
        path.relative_to(artifact_dir).as_posix() for path in artifact_directories
    }
    if actual_directories != {"model"}:
        raise FinalExportWorkflowError("Final artifact directory tree differs")
    stable_regular_file_record(manifest_path, root=artifact_dir)
    tree_sha256 = canonical_sha256(
        {"schema_version": 1, "files": normalized_records}
    )
    if (
        tree_sha256 != manifest.get("tree_sha256")
        or tree_sha256 != completion.get("artifact_tree_sha256")
        or sum(record["bytes"] for record in normalized_records)
        != manifest.get("total_bytes")
    ):
        raise FinalExportWorkflowError("Final artifact tree digest differs")

    model_record = next(
        record
        for record in normalized_records
        if record["path"] == "model/model.safetensors"
    )
    if (
        model_record["bytes"] < 2_000_000_000
        or model_record["sha256"] == entry["checkpoint_model_sha256"]
    ):
        raise FinalExportWorkflowError("Final model weights are not a trained export")
    model_config = load_stable_json(
        artifact_dir / "model/config.json", root=artifact_dir
    )
    if (
        model_config.get("model_type") != "xlm-roberta"
        or model_config.get("id2label") != {"0": "MATCH_SCORE"}
        or model_config.get("label2id") != {"MATCH_SCORE": 0}
    ):
        raise FinalExportWorkflowError("Final model config differs")
    tokenizer_config = load_stable_json(
        artifact_dir / "model/tokenizer_config.json", root=artifact_dir
    )
    if not isinstance(tokenizer_config, dict) or not tokenizer_config:
        raise FinalExportWorkflowError("Final tokenizer config differs")
    special_tokens_record = next(
        record
        for record in normalized_records
        if record["path"] == "model/special_tokens_map.json"
    )
    expected_special_tokens = builder.checkpoint_push.EXPECTED_SOURCE_FILES[
        "special_tokens_map.json"
    ]
    if {
        "bytes": special_tokens_record["bytes"],
        "sha256": special_tokens_record["sha256"],
    } != expected_special_tokens:
        raise FinalExportWorkflowError("Final special-tokens map differs")
    training_config = load_stable_json(
        artifact_dir / "training_config.json", root=artifact_dir
    )
    if training_config != entry["expected_config"]:
        raise FinalExportWorkflowError("Final frozen training config differs")
    summary = load_stable_json(
        artifact_dir / "training_summary.json", root=artifact_dir
    )
    if (
        summary.get("quality_evaluation") is not False
        or summary.get("validation_splits") != []
        or summary.get("validation_predictions_written") is not False
        or summary.get("original_training_examples") != builder.EXPECTED_ROWS
        or summary.get("training_examples") != builder.EXPECTED_TRAINING_EXAMPLES
        or summary.get("planned_optimizer_updates") != builder.EXPECTED_TOTAL_UPDATES
        or summary.get("gradient_accumulation_normalization")
        != "sample_exact_group_mean_v1"
    ):
        raise FinalExportWorkflowError("Final training summary differs")
    geometry = summary.get("epoch_batch_geometry")
    if not isinstance(geometry, list) or len(geometry) != 2:
        raise FinalExportWorkflowError("Final epoch batch geometry differs")
    for epoch, record in enumerate(geometry, start=1):
        if (
            not isinstance(record, Mapping)
            or record.get("epoch") != epoch
            or record.get("partial_batch_size") != 3
            or record.get("partial_group_examples_per_rank") not in {43, 91}
            or record.get("ranks_geometry_equal") is not True
            or not isinstance(record.get("partial_batch_step_zero_based"), int)
            or not 0
            <= record["partial_batch_step_zero_based"]
            < builder.EXPECTED_STEPS_PER_EPOCH_PER_RANK
            or not isinstance(record.get("batch_sizes_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", record["batch_sizes_sha256"])
        ):
            raise FinalExportWorkflowError("Final sample-exact batch geometry differs")
    sheets_report = load_stable_json(
        artifact_dir / "training_report.json", root=artifact_dir
    )
    expected_sheets_header = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "final_deployment_export",
        "experiment_group": "sft",
        "quality_evaluation": False,
        "original_training_examples": builder.EXPECTED_ROWS,
        "training_examples": builder.EXPECTED_TRAINING_EXAMPLES,
        "training_source_counts": {
            spec["label_source"]: spec["rows"] for spec in builder.SPLITS
        },
        "args": entry["expected_config"],
        "notes": builder.SHEETS_NOTES,
    }
    if any(
        sheets_report.get(key) != value
        for key, value in expected_sheets_header.items()
    ) or set(sheets_report.get("validation_splits", {})) != {"iid", "hard", "ood"}:
        raise FinalExportWorkflowError("Final Sheets training report differs")
    for split in ("iid", "hard", "ood"):
        metrics = sheets_report["validation_splits"][split]
        if metrics != builder.UNAVAILABLE_METRIC:
            raise FinalExportWorkflowError(
                f"Final {split} unavailable-metric sentinel differs"
            )
    if completion.get("training_report") != sheets_report:
        raise FinalExportWorkflowError("Completion/Sheets training reports differ")
    smoke = load_stable_json(
        artifact_dir / "deployment_smoke.json", root=artifact_dir
    )
    if smoke != {
        "schema_version": 1,
        "status": "passed",
        "quality_evaluation": False,
        "check": "load_tokenizer_model_and_one_finite_forward",
        "parameters": builder.base.EXPECTED_PARAMETERS,
        "logit_shape": [1, 1],
        "finite": True,
    }:
        raise FinalExportWorkflowError("Final deployment smoke differs")
    data_report = load_stable_json(
        artifact_dir / "train_data_report.json", root=artifact_dir
    )
    if (
        data_report.get("policy")
        != "all_human_assignments_sorted_by_human_row_id_v1"
        or data_report.get("quality_evaluation") is not False
        or data_report.get("withheld_pairs") != 0
        or data_report.get("train_pairs") != builder.EXPECTED_ROWS
        or data_report.get("train_positives") != builder.EXPECTED_POSITIVES
        or data_report.get("assignment_file", {}).get("sha256")
        != builder.EXPECTED_ASSIGNMENTS_SHA256
    ):
        raise FinalExportWorkflowError("Final train-data report differs")
    if (
        completion.get("training_summary") != summary
        or completion.get("deployment_smoke") != smoke
        or completion.get("train_data") != data_report
        or completion.get("memory_preflight")
        != load_stable_json(
            artifact_dir / "memory_preflight.json", root=artifact_dir
        )
    ):
        raise FinalExportWorkflowError("Final completion embedded reports differ")
    materialized_train = data_report.get("materialized_train_parquet", {})
    if (
        materialized_train.get("readback_frame_exact") is not True
        or materialized_train.get("columns")
        != ["id1", "id2", "target", "label_source"]
        or materialized_train.get("dtypes")
        != {
            "id1": "int64",
            "id2": "int64",
            "target": "float64",
            "label_source": "object",
        }
    ):
        raise FinalExportWorkflowError("Final train materialization proof differs")
    forbidden = [
        path
        for path in download_files
        if path.name.endswith("validation_predictions.parquet")
    ]
    if forbidden:
        raise FinalExportWorkflowError(f"Forbidden final-export files found: {forbidden}")
    sync_paths = [
        path for path in download_files if path.name == "google_sheets_sync.json"
    ]
    if len(sync_paths) != 1:
        raise FinalExportWorkflowError("Final export has no unique Sheets sync receipt")
    sync = load_stable_json(sync_paths[0], root=directory)
    if sync.get("status") not in {"synced", "pending"}:
        raise FinalExportWorkflowError("Final Sheets sync status differs")
    pending_paths = [
        path for path in download_files if path.name == "sheets_sync_pending.json"
    ]
    if sync["status"] == "pending" and len(pending_paths) != 1:
        raise FinalExportWorkflowError("Pending Sheets sync has no retry payload")
    if sync["status"] == "synced" and pending_paths:
        raise FinalExportWorkflowError("Synced Sheets run retained a pending payload")
    if sync.get("run_id") != completion["run_id"]:
        raise FinalExportWorkflowError("Sheets sync run_id differs from completion")
    if sync.get("spreadsheet_id") != builder.shared.EXPERIMENT_SPREADSHEET_ID:
        raise FinalExportWorkflowError("Sheets sync spreadsheet identity differs")
    if sync["status"] == "synced":
        if (
            set(sync)
            != {
                "status",
                "run_id",
                "synced_at_utc",
                "spreadsheet_id",
                "spreadsheet_url",
                "experiment_action",
                "experiment_group",
                "comparison_sheet",
                "comparison_action",
            }
            or sync.get("spreadsheet_url")
            != (
                "https://docs.google.com/spreadsheets/d/"
                f"{builder.shared.EXPERIMENT_SPREADSHEET_ID}/edit"
            )
            or sync.get("experiment_group") != "sft"
            or sync.get("comparison_sheet") != "sft_exps"
            or sync.get("experiment_action") not in {"appended", "updated"}
            or sync.get("comparison_action") not in {"appended", "updated"}
            or not isinstance(sync.get("synced_at_utc"), str)
            or not sync["synced_at_utc"]
        ):
            raise FinalExportWorkflowError("Synced Sheets route/action differs")
    if pending_paths:
        pending = load_stable_json(pending_paths[0], root=directory)
        if (
            set(sync) != {
                "status",
                "run_id",
                "spreadsheet_id",
                "error_type",
                "error",
            }
            or set(pending) != set(sync) | {"completion"}
            or pending.get("run_id") != completion["run_id"]
            or pending.get("completion") != completion
            or not isinstance(pending.get("error_type"), str)
            or not pending["error_type"]
            or not isinstance(pending.get("error"), str)
            or not pending["error"]
        ):
            raise FinalExportWorkflowError("Pending Sheets run_id differs")

    working_root = completion_path.parent
    essential_root_files = {
        "experiment_run_id.txt",
        "experiment_started_at_utc.txt",
        "cross_encoder_config.json",
        "bge_memory_preflight.json",
        "bge_train_data_report.json",
        "bge_runtime_versions.json",
        f"{builder.EXPERIMENT}.log",
        f"{builder.EXPERIMENT}_memory_preflight.log",
        f"{entry['kernel_slug']}.log",
        "notebook_completed.json",
        "google_sheets_sync.json",
        "google_sheets_logger.py",
    }
    missing_root = [
        name
        for name in sorted(essential_root_files)
        if (working_root / name) not in download_files
    ]
    if missing_root:
        raise FinalExportWorkflowError(
            f"Final export is missing essential top-level outputs: {missing_root}"
        )
    for name in essential_root_files:
        path = working_root / name
        if stable_regular_file_record(path, root=directory)["bytes"] <= 0:
            raise FinalExportWorkflowError(f"Final top-level output is empty: {name}")
    if working_root != directory:
        raise FinalExportWorkflowError("Final completion marker is not at download root")
    expected_download_directories = {artifact_dir, artifact_dir / "model"}
    if download_directories != expected_download_directories:
        raise FinalExportWorkflowError("Final download directory topology differs")
    expected_download_files = {
        working_root / name for name in essential_root_files
    } | artifact_files
    if pending_paths:
        expected_download_files.add(pending_paths[0])
    if download_files != expected_download_files:
        raise FinalExportWorkflowError("Final download file topology differs")
    _, run_id_payload = _stable_regular_file(
        working_root / "experiment_run_id.txt", root=directory, collect=True
    )
    assert run_id_payload is not None
    if run_id_payload.decode("utf-8").strip() != completion["run_id"]:
        raise FinalExportWorkflowError("Final run-id marker differs")
    mirrored = {
        "bge_memory_preflight.json": "memory_preflight.json",
        "bge_train_data_report.json": "train_data_report.json",
        "bge_runtime_versions.json": "runtime_versions.json",
    }
    for source_name, artifact_name in mirrored.items():
        source_record = stable_regular_file_record(
            working_root / source_name, root=directory
        )
        artifact_record = stable_regular_file_record(
            artifact_dir / artifact_name, root=artifact_dir
        )
        if source_record["sha256"] != artifact_record["sha256"]:
            raise FinalExportWorkflowError(
                f"Final mirrored audit output differs: {source_name}"
            )
    return {
        "directory": str(directory),
        "completion": str(completion_path),
        "run_id": completion["run_id"],
        "identity_sha256": entry["identity_sha256"],
        "artifact_tree_sha256": tree_sha256,
        "model_sha256": model_record["sha256"],
        "model_bytes": model_record["bytes"],
    }


def output_root() -> Path:
    configured = Path(os.getenv("KAGGLE_OUTPUT_DIR", "artifacts/kaggle")).expanduser()
    return configured.resolve() if configured.is_absolute() else (ROOT / configured).resolve()


def download_and_validate(
    cli: list[str], *, owner: str, entry: Mapping[str, Any]
) -> Path:
    root = output_root()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / str(entry["kernel_slug"])
    if destination.exists():
        validate_artifact_tree(destination, entry=entry)
        return destination
    kernel_ref = f"{owner}/{entry['kernel_slug']}"
    staging: Path | None = None
    for attempt in range(1, 4):
        candidate = Path(
            tempfile.mkdtemp(prefix=f".{entry['kernel_slug']}.download-", dir=root)
        )
        result = kaggle.run_command(
            cli
            + [
                "kernels",
                "output",
                kernel_ref,
                "-p",
                str(candidate),
                "--force",
                "--page-size",
                "200",
            ],
            check=False,
        )
        if result.returncode == 0:
            staging = candidate
            break
        shutil.rmtree(candidate)
        if attempt < 3:
            time.sleep(3 if attempt == 1 else 8)
    if staging is None:
        raise FinalExportWorkflowError("Could not download final export")
    try:
        validation = validate_artifact_tree(staging, entry=entry)
    except Exception:
        print(f"Invalid final-export download preserved at: {staging}")
        raise
    staging.rename(destination)
    print(json.dumps({"validated_final_export": validation}, ensure_ascii=False))
    return destination


def enforce_live_environment() -> None:
    if kaggle.env_bool("KAGGLE_IS_PRIVATE", True) is not True:
        raise FinalExportWorkflowError("Final BGE export must remain private")
    if kaggle.env_bool("KAGGLE_ENABLE_INTERNET", True) is not True:
        raise FinalExportWorkflowError(
            "Final BGE export requires internet for the pinned package bootstrap"
        )
    accelerator = os.getenv("KAGGLE_ACCELERATOR", "NvidiaTeslaT4").strip()
    if accelerator != "NvidiaTeslaT4":
        raise FinalExportWorkflowError("Final BGE export requires NvidiaTeslaT4")


def monitor_existing(
    *, cli: list[str], owner: str, entry: Mapping[str, Any]
) -> int:
    require_reserved_attempt(entry)
    kernel_ref = f"{owner}/{entry['kernel_slug']}"
    status = campaign_runner.remote_kernel_status(cli, kernel_ref)
    if status == "absence_unconfirmed":
        raise FinalExportWorkflowError("Reserved final-export kernel is absent remotely")
    verify_remote_sources(cli, kernel_ref=kernel_ref, entry=entry)
    if status in {"queued", "running"}:
        update_attempt_status(entry, "running")
        wait_timeout = kaggle.env_int(
            "KAGGLE_WAIT_TIMEOUT_SECONDS",
            DEFAULT_WAIT_TIMEOUT_SECONDS,
            minimum=60,
        )
        kaggle.wait_for_kernel(
            cli,
            kernel_ref,
            poll_interval=POLL_INTERVAL_SECONDS,
            wait_timeout=wait_timeout,
        )
        status = "complete"
    if status in kaggle.TERMINAL_FAILURE:
        update_attempt_status(entry, "failed")
        kaggle.run_command(cli + ["kernels", "logs", kernel_ref], check=False)
        raise FinalExportWorkflowError(
            "Final-export kernel failed; automatic resubmission is forbidden"
        )
    if status not in kaggle.TERMINAL_SUCCESS and status != "complete":
        raise FinalExportWorkflowError(f"Unexpected final-export status: {status}")
    download_and_validate(cli, owner=owner, entry=entry)
    update_attempt_status(entry, "complete")
    return 0


def execute_once(
    *,
    cli: list[str],
    owner: str,
    entry: Mapping[str, Any],
    env_file: Path,
    no_wait: bool,
) -> int:
    stage_locally(entry, env_file=env_file)
    kernel_ref = f"{owner}/{entry['kernel_slug']}"
    campaign_runner.confirm_remote_absence(cli, kernel_ref)
    remote_inputs = audit_remote_training_inputs(cli, entry=entry)
    print(json.dumps({"remote_training_input_gate": remote_inputs}, ensure_ascii=False))
    validate_staged_kernel_metadata(entry)
    recheck_remote_input_statuses(cli, remote_inputs)
    # Dataset audits take multiple calls.  Make repeated kernel absence checks
    # the final reads immediately before the one-shot reservation and push.
    campaign_runner.confirm_remote_absence(cli, kernel_ref)
    reserve_single_attempt(entry)
    stage_dir = kaggle.STAGE_ROOT / str(entry["kernel_slug"])
    result = kaggle.run_command(
        cli
        + [
            "kernels",
            "push",
            "-p",
            str(stage_dir),
            "--timeout",
            str(RUN_TIMEOUT_SECONDS),
            "--accelerator",
            "NvidiaTeslaT4",
        ],
        check=False,
    )
    if result.returncode:
        update_attempt_status(entry, "failed")
        raise FinalExportWorkflowError(
            "Final-export push failed; this identity will not be resubmitted"
        )
    verify_remote_sources(cli, kernel_ref=kernel_ref, entry=entry)
    update_attempt_status(entry, "submitted")
    if no_wait:
        print(json.dumps({"submitted_once": kernel_ref, "timeout": RUN_TIMEOUT_SECONDS}))
        return 0
    return monitor_existing(cli=cli, owner=owner, entry=entry)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--monitor", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.no_wait and not args.execute:
        raise SystemExit("--no-wait is valid only with --execute")
    env_file = args.env_file if args.env_file.is_absolute() else ROOT / args.env_file
    kaggle.load_dotenv(env_file)
    owner = os.getenv("KAGGLE_USERNAME", "").strip()
    if not owner:
        raise SystemExit("Set KAGGLE_USERNAME in .env")
    owner = kaggle.validate_slug(owner, "KAGGLE_USERNAME")
    entry = load_frozen_entry(owner)
    if (
        entry["validation_dataset"] != VALIDATION_DATASET
        or entry["checkpoint_dataset"] != CHECKPOINT_DATASET
    ):
        raise FinalExportWorkflowError("Final-export Dataset identities changed")
    if entry["kernel_slug"] in TOMBSTONED_KERNEL_SLUGS:
        raise FinalExportWorkflowError("Generated final-export slug is tombstoned")
    mode = "monitor" if args.monitor else "execute" if args.execute else "dry-run"
    print(
        json.dumps(
            {
                "campaign": builder.CAMPAIGN,
                "mode": mode,
                "execution": "one_kernel_sequential_no_resubmit",
                "kernel_slug": entry["kernel_slug"],
                "identity_sha256": entry["identity_sha256"],
                "recipe_sha256": entry["recipe_sha256"],
                "train_rows": builder.EXPECTED_ROWS,
                "withheld_rows": 0,
                "quality_evaluation": False,
                "google_sheets": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if mode == "dry-run":
        stage_locally(entry, env_file=env_file)
        return 0
    enforce_live_environment()
    token = os.getenv("KAGGLE_API_TOKEN", "").strip()
    if not token:
        raise SystemExit("Set KAGGLE_API_TOKEN in .env")
    cli = kaggle.kaggle_command()
    kaggle.run_command(cli + ["kernels", "list", "--mine", "--page-size", "1"])
    with exclusive_lock():
        if mode == "monitor":
            return monitor_existing(cli=cli, owner=owner, entry=entry)
        return execute_once(
            cli=cli,
            owner=owner,
            entry=entry,
            env_file=env_file,
            no_wait=args.no_wait,
        )


if __name__ == "__main__":
    raise SystemExit(main())
