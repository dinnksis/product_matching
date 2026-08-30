#!/usr/bin/env python3
"""Read-only remote monitor for the already-pushed final BGE v1 kernel.

This file is intentionally outside the final export source ledger.  The
reviewed launcher cannot monitor the pushed version because Kaggle normalizes
``code_file`` from ``notebook.ipynb`` to ``<kernel-slug>.ipynb`` on pull.  This
controller accepts exactly that one server normalization while retaining every
other frozen identity, metadata, source and artifact check.

The default mode is local plan-only.  ``--wait`` may issue only read operations
against the already-existing kernel (get/status/pull/logs/output).  It has no
staging, reservation, push, versioning, Sheets or resubmission path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import tempfile
from pathlib import Path
from typing import Any, Mapping

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

import create_bge_2ep_final_fulltrain_notebook as builder
import run_bge_2ep_final_fulltrain as frozen
import run_kaggle_notebook as kaggle


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_OWNER = "alexproger23"
EXPECTED_KERNEL_VERSION = 1
EXPECTED_KERNEL_SLUG = "pm-b2-final-b501110394a4-s42-v1"
EXPECTED_IDENTITY_SHA256 = (
    "b501110394a4e021c18bbed86d14fdf616e64a4482b1bc2f580285c68768ff57"
)
EXPECTED_SOURCE_SHA256 = (
    "87898d90d2203907098df303b14d403d58afc2f36eea31587bf9aac23eb15dfb"
)
EXPECTED_EXECUTABLE_CELLS_SHA256 = (
    "f1b18efff2fa9b6ef0729f822ec07425cc160700872e7d0dc18deb8647956dc4"
)
EXPECTED_NOTEBOOK_SHA256 = (
    "4c5e627a384f30b9fce5907ec74d57f62e58422071d99fd56f94c26a7629b3ed"
)
EXPECTED_RECIPE_SHA256 = (
    "d46be9217a43396ecc8c594fc1864ee93761d288c30e5a40041adbb28bd7adfe"
)
EXPECTED_REMOTE_ID_NO = 132_549_639
EXPECTED_REMOTE_DOCKER_IMAGE = (
    "gcr.io/kaggle-private-byod/python@sha256:"
    "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
)
POLL_INTERVAL_SECONDS = 60
DEFAULT_WAIT_TIMEOUT_SECONDS = frozen.DEFAULT_WAIT_TIMEOUT_SECONDS
TRANSIENT_ATTEMPTS = 3
TRANSIENT_BACKOFF_SECONDS = (1.0, 3.0)
THIS_SOURCE = Path("scripts/monitor_bge_2ep_final_fulltrain_existing.py")
THIS_TEST = Path("tests/test_monitor_bge_2ep_final_fulltrain_existing.py")


class ExistingFinalExportMonitorError(RuntimeError):
    """Raised when the already-pushed b501/v1 authority is not exact."""


def expected_remote_metadata(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact metadata emitted by the installed Kaggle pull client."""
    slug = str(entry["kernel_slug"])
    return {
        "id": f"{EXPECTED_OWNER}/{slug}",
        "id_no": EXPECTED_REMOTE_ID_NO,
        "title": entry["title"],
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        # Kaggle canonicalizes attachment order on the stored version.
        "dataset_sources": sorted(frozen.expected_dataset_sources(entry)),
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "docker_image": EXPECTED_REMOTE_DOCKER_IMAGE,
        "machine_shape": "NvidiaTeslaT4",
    }


def load_exact_b501_entry(owner: str = EXPECTED_OWNER) -> dict[str, Any]:
    """Load the frozen entry without writing and pin it to the pushed b501 tuple."""
    if owner != EXPECTED_OWNER:
        raise ExistingFinalExportMonitorError("Final b501 owner differs")
    ledger_paths = {path.as_posix() for path in builder.SOURCE_LEDGER_FILES}
    excluded = {THIS_SOURCE.as_posix(), THIS_TEST.as_posix()}
    overlap = sorted(ledger_paths & excluded)
    if overlap:
        raise ExistingFinalExportMonitorError(
            f"Identity-neutral monitor files entered SOURCE_LEDGER_FILES: {overlap}"
        )
    entry = frozen.load_frozen_entry(owner)
    pinned = {
        "kernel_slug": EXPECTED_KERNEL_SLUG,
        "identity_sha256": EXPECTED_IDENTITY_SHA256,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "executable_cells_sha256": EXPECTED_EXECUTABLE_CELLS_SHA256,
        "notebook_sha256": EXPECTED_NOTEBOOK_SHA256,
        "recipe_sha256": EXPECTED_RECIPE_SHA256,
    }
    mismatches = {
        key: {"actual": entry.get(key), "expected": expected}
        for key, expected in pinned.items()
        if entry.get(key) != expected
    }
    if mismatches:
        raise ExistingFinalExportMonitorError(
            "Frozen b501 tuple changed: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    return entry


def require_exact_attempt(
    entry: Mapping[str, Any],
    *,
    path: Path = frozen.ATTEMPT_LEDGER_PATH,
) -> dict[str, Any]:
    """Require the single already-pushed b501 attempt and no other attempt."""
    try:
        if path == frozen.ATTEMPT_LEDGER_PATH:
            frozen.require_reserved_attempt(entry)
        ledger = frozen.load_attempt_ledger(path)
    except frozen.FinalExportWorkflowError as error:
        raise ExistingFinalExportMonitorError(
            "Existing final-export attempt ledger differs"
        ) from error
    attempts = ledger.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != 1:
        raise ExistingFinalExportMonitorError(
            "Existing monitor requires exactly one final-export attempt"
        )
    record = attempts[0]
    allowed_keys = {
        "kernel_slug",
        "identity_sha256",
        "status",
        "reserved_at_unix",
        "updated_at_unix",
    }
    if not isinstance(record, dict) or not {
        "kernel_slug",
        "identity_sha256",
        "status",
        "reserved_at_unix",
    }.issubset(record) or not set(record).issubset(allowed_keys):
        raise ExistingFinalExportMonitorError("Existing attempt record schema differs")
    if (
        record["kernel_slug"] != entry["kernel_slug"]
        or record["identity_sha256"] != entry["identity_sha256"]
        or record["status"]
        not in {"submitted", "running", "complete", "failed"}
    ):
        raise ExistingFinalExportMonitorError("Existing attempt identity/status differs")
    reserved = record["reserved_at_unix"]
    updated = record.get("updated_at_unix", reserved)
    if (
        isinstance(reserved, bool)
        or not isinstance(reserved, (int, float))
        or not math.isfinite(float(reserved))
        or reserved <= 0
        or isinstance(updated, bool)
        or not isinstance(updated, (int, float))
        or not math.isfinite(float(updated))
        or updated < reserved
    ):
        raise ExistingFinalExportMonitorError("Existing attempt timestamps differ")
    return dict(record)


def create_readonly_api() -> KaggleApi:
    """Authenticate a client used only for the get-kernel version authority."""
    api = KaggleApi()
    api.authenticate()
    return api


def _transient_backoff(completed_attempts: int) -> None:
    """Sleep only between bounded transport attempts."""
    if not 1 <= completed_attempts < TRANSIENT_ATTEMPTS:
        raise ExistingFinalExportMonitorError("Invalid transient retry index")
    time.sleep(TRANSIENT_BACKOFF_SECONDS[completed_attempts - 1])


def _get_kernel_with_retry(
    api: KaggleApi, request: ApiGetKernelRequest
) -> Any:
    """Retry only exceptions raised by the remote get transport."""
    last_error: Exception | None = None
    for attempt in range(1, TRANSIENT_ATTEMPTS + 1):
        try:
            with api.build_kaggle_client() as client:
                return client.kernels.kernels_api_client.get_kernel(request)
        except ExistingFinalExportMonitorError:
            raise
        except Exception as error:
            last_error = error
            if attempt < TRANSIENT_ATTEMPTS:
                _transient_backoff(attempt)
    raise ExistingFinalExportMonitorError(
        "Could not read remote b501 get-kernel authority after bounded retries"
    ) from last_error


def read_remote_v1_authority(
    api: KaggleApi, *, owner: str, entry: Mapping[str, Any]
) -> dict[str, Any]:
    """Read the exact current-version authority; never print the returned blob."""
    request = ApiGetKernelRequest()
    request.user_name = owner
    request.kernel_slug = str(entry["kernel_slug"])
    response = _get_kernel_with_retry(api, request)
    metadata = response.metadata
    datasets = list(metadata.dataset_data_sources or [])
    exact = {
        "id_no": metadata.id,
        "ref": metadata.ref,
        "title": metadata.title,
        "slug": metadata.slug,
        "current_version_number": metadata.current_version_number,
        "is_private": metadata.is_private,
        "enable_gpu": metadata.enable_gpu,
        "enable_tpu": metadata.enable_tpu,
        "enable_internet": metadata.enable_internet,
        "dataset_sources": sorted(datasets),
        "kernel_sources": list(metadata.kernel_data_sources or []),
        "competition_sources": list(metadata.competition_data_sources or []),
        "model_sources": list(metadata.model_data_sources or []),
        "docker_image": metadata.docker_image,
        "machine_shape": metadata.machine_shape,
    }
    expected = {
        "id_no": EXPECTED_REMOTE_ID_NO,
        "ref": f"{owner}/{entry['kernel_slug']}",
        "title": entry["title"],
        "slug": entry["kernel_slug"],
        "current_version_number": EXPECTED_KERNEL_VERSION,
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": sorted(frozen.expected_dataset_sources(entry)),
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "docker_image": EXPECTED_REMOTE_DOCKER_IMAGE,
        "machine_shape": "NvidiaTeslaT4",
    }
    if exact != expected:
        raise ExistingFinalExportMonitorError(
            "Remote b501/v1 get-kernel authority differs: "
            + json.dumps(
                {"actual": exact, "expected": expected},
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        )
    return exact


def validate_pulled_notebook(
    path: Path, *, root: Path, entry: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate pulled notebook semantics through a stable no-follow read."""
    record, payload = frozen._stable_regular_file(path, root=root, collect=True)
    assert payload is not None
    try:
        notebook = builder.nbf.reads(payload.decode("utf-8"), as_version=4)
    except Exception as error:
        raise ExistingFinalExportMonitorError(
            "Pulled b501 notebook cannot be parsed"
        ) from error
    checked = builder.validate_notebook_identity(notebook, entry=entry)
    expected = {
        "identity_sha256": EXPECTED_IDENTITY_SHA256,
        "executable_cells_sha256": EXPECTED_EXECUTABLE_CELLS_SHA256,
        "source_sha256": EXPECTED_SOURCE_SHA256,
    }
    if checked != expected:
        raise ExistingFinalExportMonitorError(
            "Pulled b501 notebook identity/source/executable digest differs"
        )
    return {"file": record, **checked}


def pull_and_validate_remote_v1(
    cli: list[str],
    api: KaggleApi,
    *,
    owner: str,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    """Pull current source only while authoritative current version remains one.

    Kaggle currently returns 403 for an explicit private ``owner/slug/1`` pull.
    Therefore the unversioned read is bracketed by authoritative get-kernel
    reads that both require ``current_version_number == 1``.
    """
    kernel_ref = f"{owner}/{entry['kernel_slug']}"
    last_error: Exception | None = None
    for attempt in range(1, TRANSIENT_ATTEMPTS + 1):
        before = read_remote_v1_authority(api, owner=owner, entry=entry)
        with tempfile.TemporaryDirectory(prefix="bge-final-b501-v1-pull-") as raw:
            root = Path(raw)
            result = None
            try:
                result = kaggle.run_command(
                    cli + ["kernels", "pull", kernel_ref, "-p", raw, "-m"],
                    check=False,
                )
            except ExistingFinalExportMonitorError:
                raise
            except Exception as error:
                last_error = error
            after = read_remote_v1_authority(api, owner=owner, entry=entry)
            if before != after:
                raise ExistingFinalExportMonitorError(
                    "Remote b501 authority changed while pulling source"
                )
            if result is None or result.returncode:
                if result is not None:
                    last_error = RuntimeError(
                        f"Kaggle pull exited with status {result.returncode}"
                    )
                if attempt < TRANSIENT_ATTEMPTS:
                    _transient_backoff(attempt)
                    continue
                break

            # A successful transport is validated exactly once.  Semantic,
            # identity and content mismatches are never retried.
            metadata_path = root / "kernel-metadata.json"
            notebook_path = root / f"{entry['kernel_slug']}.ipynb"
            directories, files = frozen.scan_regular_tree(root)
            if directories or files != {metadata_path, notebook_path}:
                raise ExistingFinalExportMonitorError(
                    "Pulled b501 tree differs from exact metadata/notebook allowlist"
                )
            metadata = frozen.load_stable_json(metadata_path, root=root)
            expected = expected_remote_metadata(entry)
            if metadata != expected:
                raise ExistingFinalExportMonitorError(
                    "Pulled b501 metadata differs: "
                    + json.dumps(
                        {"actual": metadata, "expected": expected},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
            notebook = validate_pulled_notebook(
                notebook_path, root=root, entry=entry
            )
            return {
                "authority": after,
                "metadata": metadata,
                "notebook": notebook,
            }
    raise ExistingFinalExportMonitorError(
        "Could not pull existing b501 source after bounded retries"
    ) from last_error


def read_exact_v1_status(
    cli: list[str],
    api: KaggleApi,
    *,
    owner: str,
    entry: Mapping[str, Any],
) -> str:
    """Read latest-session status bracketed by exact current-version-one gates."""
    kernel_ref = f"{owner}/{entry['kernel_slug']}"
    last_error: Exception | None = None
    for attempt in range(1, TRANSIENT_ATTEMPTS + 1):
        before = read_remote_v1_authority(api, owner=owner, entry=entry)
        result = None
        try:
            result = kaggle.run_command(
                cli + ["kernels", "status", kernel_ref], check=False
            )
        except ExistingFinalExportMonitorError:
            raise
        except Exception as error:
            last_error = error
        after = read_remote_v1_authority(api, owner=owner, entry=entry)
        if before != after:
            raise ExistingFinalExportMonitorError(
                "Remote b501 authority changed while reading status"
            )
        if result is None or result.returncode:
            if result is not None:
                last_error = RuntimeError(
                    f"Kaggle status exited with status {result.returncode}"
                )
            if attempt < TRANSIENT_ATTEMPTS:
                _transient_backoff(attempt)
                continue
            break

        # A response that reached Kaggle but has an unknown status is a
        # semantic mismatch, not a transport failure; never retry it.
        status = kaggle.extract_status(result.stdout)
        if status not in kaggle.TERMINAL_SUCCESS | kaggle.TERMINAL_FAILURE | {
            "queued",
            "running",
        }:
            raise ExistingFinalExportMonitorError(
                f"Unexpected existing b501 status: {status!r}"
            )
        return status
    raise ExistingFinalExportMonitorError(
        "Could not read existing b501 status after bounded retries"
    ) from last_error


def update_exact_attempt(entry: Mapping[str, Any], status: str) -> dict[str, Any]:
    frozen.update_attempt_status(entry, status)
    record = require_exact_attempt(entry)
    if record["status"] != status:
        raise ExistingFinalExportMonitorError("Attempt status update did not persist")
    return record


def log_terminal_failure(
    cli: list[str], api: KaggleApi, *, owner: str, entry: Mapping[str, Any]
) -> None:
    before = read_remote_v1_authority(api, owner=owner, entry=entry)
    kernel_ref = f"{owner}/{entry['kernel_slug']}"
    kaggle.run_command(cli + ["kernels", "logs", kernel_ref], check=False)
    after = read_remote_v1_authority(api, owner=owner, entry=entry)
    if before != after:
        raise ExistingFinalExportMonitorError(
            "Remote b501 authority changed while reading failure logs"
        )


def wait_download_and_validate(
    cli: list[str],
    api: KaggleApi,
    *,
    owner: str,
    entry: Mapping[str, Any],
) -> Path:
    """Wait for the one existing v1, then strictly download or mark failure."""
    record = require_exact_attempt(entry)
    if record["status"] == "failed":
        raise ExistingFinalExportMonitorError(
            "The only b501 attempt is already failed; resubmission is forbidden"
        )
    pull_and_validate_remote_v1(cli, api, owner=owner, entry=entry)
    deadline = time.monotonic() + kaggle.env_int(
        "KAGGLE_WAIT_TIMEOUT_SECONDS",
        DEFAULT_WAIT_TIMEOUT_SECONDS,
        minimum=60,
    )
    last_status: str | None = None
    while True:
        status = read_exact_v1_status(
            cli, api, owner=owner, entry=entry
        )
        if status != last_status:
            print(f"Kaggle b501/v1 status: {status}", flush=True)
            last_status = status
        if status in kaggle.TERMINAL_FAILURE:
            update_exact_attempt(entry, "failed")
            log_terminal_failure(cli, api, owner=owner, entry=entry)
            raise ExistingFinalExportMonitorError(
                "Final-export b501/v1 failed; no resubmit is permitted"
            )
        if status in kaggle.TERMINAL_SUCCESS:
            break
        update_exact_attempt(entry, "running")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ExistingFinalExportMonitorError(
                "Local wait timed out while b501/v1 remains active; no remote state changed"
            )
        time.sleep(min(POLL_INTERVAL_SECONDS, remaining))

    pull_and_validate_remote_v1(cli, api, owner=owner, entry=entry)
    before_download = read_remote_v1_authority(api, owner=owner, entry=entry)
    try:
        destination = frozen.download_and_validate(cli, owner=owner, entry=entry)
        after_download = read_remote_v1_authority(api, owner=owner, entry=entry)
        if before_download != after_download:
            raise ExistingFinalExportMonitorError(
                "Remote b501 authority changed while downloading outputs"
            )
        # Replay the strict validator even when download_and_validate reused an
        # already-present destination.
        frozen.validate_artifact_tree(destination, entry=entry)
    except Exception:
        # Transport, filesystem and local validation errors do not change the
        # authoritative terminal-success state.  Preserve the prior attempt
        # status so the same existing v1 output can be retried read-only.
        raise
    update_exact_attempt(entry, "complete")
    return destination


def plan_payload(entry: Mapping[str, Any], attempt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mode": "local_plan_only",
        "owner": EXPECTED_OWNER,
        "kernel_slug": entry["kernel_slug"],
        "kernel_version": EXPECTED_KERNEL_VERSION,
        "identity_sha256": entry["identity_sha256"],
        "source_sha256": entry["source_sha256"],
        "executable_cells_sha256": entry["executable_cells_sha256"],
        "notebook_sha256": entry["notebook_sha256"],
        "attempt_status": attempt["status"],
        "default_remote_calls": [],
        "wait_remote_operations": ["get", "status", "pull", "logs", "output"],
        "remote_mutations": [],
        "resubmit": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument(
        "--wait",
        action="store_true",
        help="read, wait for and strictly download the existing b501/v1 run",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entry = load_exact_b501_entry()
    if not args.wait:
        attempt = require_exact_attempt(entry)
        print(json.dumps(plan_payload(entry, attempt), ensure_ascii=False, indent=2))
        return 0

    env_file = args.env_file if args.env_file.is_absolute() else ROOT / args.env_file
    kaggle.load_dotenv(env_file)
    owner = os.getenv("KAGGLE_USERNAME", "").strip()
    if owner != EXPECTED_OWNER:
        raise ExistingFinalExportMonitorError("KAGGLE_USERNAME is not the b501 owner")
    if not os.getenv("KAGGLE_API_TOKEN", "").strip():
        raise SystemExit("Set KAGGLE_API_TOKEN in .env")
    cli = kaggle.kaggle_command()
    api = create_readonly_api()
    with frozen.exclusive_lock():
        destination = wait_download_and_validate(
            cli, api, owner=owner, entry=entry
        )
    print(
        json.dumps(
            {
                "status": "complete",
                "kernel_ref": f"{owner}/{entry['kernel_slug']}/1",
                "validated_output": str(destination),
                "identity_sha256": entry["identity_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
