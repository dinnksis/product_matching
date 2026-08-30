#!/usr/bin/env python3
"""One-time fail-closed recovery for the pre-creation long-slug Kaggle failure.

The default preflight is local and read-only.  ``--apply`` is also local: it
archives the exact bad primary lock, its owned provenance sidecars, generated
notebooks/staging, and the failed controller state outside ``stage_locks``;
then it rematerializes the same recipes under the reviewed short-identity
adapter.  This script never calls Kaggle.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import stat
import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import materialize_minilm_5ep_sft_loss_fast_track as materializer
import minilm_5ep_sft_loss_fast_track_support as support


ROOT = support.ROOT
AUDIT_RECEIPT = support.DEFAULT_LONG_SLUG_AUDIT_RECEIPT
BAD_LOCK = (
    support.DEFAULT_REPORT_DIR
    / "stage_locks"
    / "special_loss_screen__primary.lock.json"
)
REJECTED_ORIGINAL_PROVENANCE_ARCHIVE = Path(str(BAD_LOCK) + ".trusted-provenance")
FAILED_CONTROLLER_STATE = (
    support.DEFAULT_REPORT_DIR / "fast_track" / "loss_controller_state.json"
)
RECOVERY_ARCHIVE = (
    support.DEFAULT_REPORT_DIR
    / "fast_track"
    / "recovery"
    / "primary_long_slug_dedd2f34"
)
RECOVERY_PREFLIGHT = RECOVERY_ARCHIVE / "preflight.json"
RECOVERY_COMPLETION = RECOVERY_ARCHIVE / "completion.json"
RECOVERY_JOURNAL = RECOVERY_ARCHIVE / "journal"
RECOVERY_MUTEX = RECOVERY_ARCHIVE.parent / ".primary_long_slug_dedd2f34.apply.lock"
RECOVERY_AUDIT_COPY = RECOVERY_ARCHIVE / "evidence" / AUDIT_RECEIPT.name
RECOVERY_FORENSIC_MANIFEST = (
    RECOVERY_ARCHIVE
    / "forensic"
    / f"{BAD_LOCK.name}.trusted-provenance.original.json"
)
RECOVERY_RELOCATED_LOCK = RECOVERY_ARCHIVE / "authority" / BAD_LOCK.name
RECOVERY_RELOCATED_PROVENANCE = Path(
    str(RECOVERY_RELOCATED_LOCK) + ".trusted-provenance.json"
)
RECOVERY_RELOCATED_ARCHIVE = Path(
    str(RECOVERY_RELOCATED_LOCK) + ".trusted-provenance"
)

FaultHook = Callable[[str], None]

AUDIT_RECEIPT_FILE_SHA256 = support.LONG_SLUG_AUDIT_FILE_SHA256
AUDIT_RECEIPT_PAYLOAD_SHA256 = support.LONG_SLUG_AUDIT_PAYLOAD_SHA256
BAD_LOCK_FILE_SHA256 = support.REJECTED_PRIMARY_LOCK_FILE_SHA256
BAD_LOCK_PAYLOAD_SHA256 = support.REJECTED_PRIMARY_LOCK_PAYLOAD_SHA256
BAD_PROVENANCE_MANIFEST_FILE_SHA256 = (
    "30e21f3ada3105ec58549b55f31318d1f1f81487fccc41c8f3c37a0568ab53cc"
)
# The real one-time transaction was completed under this exact reviewed
# recovery generation.  The completion is immutable, so later hardening of
# this validator must preserve (and narrowly pin) its historical freeze anchor
# instead of rewriting already-published recovery evidence.
APPLIED_RECOVERY_FREEZE_MANIFEST_PAYLOAD_SHA256 = (
    "62a16a0c88428f396f71e32e7b39bda2620a93fe67e1e99de789e2e366177a99"
)
APPLIED_RECOVERY_COMPLETION_PAYLOAD_SHA256 = (
    "84a3b2d5ec8e302436d6e63ef43366c5dbd9437856d0b18c1a71aa81c35d71ab"
)
EXPECTED_OWNER = "alexproger23"
BAD_VARIANTS = (
    (
        "balanced_binary_bce",
        "minilm5_sft_loss_balanced_binary_bce_ff031cb1_s42_v1",
        "pm-minilm5-sft-loss-balanced-binary-bce-ff031cb1-s42-v1",
        "ff031cb18cc2c235d9aa3393589327529553d29aa3569f3704b7d9de504512dd",
    ),
    (
        "balanced_category_class_sqrt_bce",
        "minilm5_sft_loss_balanced_category_class_sqrt_bce_62369bd5_s42_v1",
        "pm-minilm5-sft-loss-balanced-category-class-sqrt-bce-62369bd5-s42-v1",
        "62369bd55051e98c88397e9d8dcd11f68145d1e5df43edd81e84186cf72b174b",
    ),
    (
        "balanced_category_class_bce",
        "minilm5_sft_loss_balanced_category_class_bce_d9ae2ba2_s42_v1",
        "pm-minilm5-sft-loss-balanced-category-class-bce-d9ae2ba2-s42-v1",
        "d9ae2ba2e8e9de7798489737ed0df43cc75e2e48a95e82ef09f95d9e182e4a69",
    ),
    (
        "focal_bce_gamma2_scale4",
        "minilm5_sft_loss_focal_bce_gamma2_scale4_3e01e93b_s42_v1",
        "pm-minilm5-sft-loss-focal-bce-gamma2-scale4-3e01e93b-s42-v1",
        "3e01e93b22806a9372f56152cf6fc2028d98c786904e8c0e8850275a1ee19947",
    ),
)
BAD_SLUGS = tuple(row[2] for row in BAD_VARIANTS)


class RecoveryError(support.FastTrackError):
    """The one-time recovery evidence or local state is not exact."""


def _fault(hook: FaultHook | None, point: str) -> None:
    if hook is not None:
        hook(point)


def _fsync_directory(path: Path) -> None:
    """Persist a directory entry when the local filesystem supports it."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_regular_single_link_stat(value: os.stat_result, *, label: str) -> None:
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise RecoveryError(f"{label} must be an ordinary single-link file")


def _open_parent_directory(path: Path, *, label: str) -> int:
    _assert_no_symlink_ancestors(path, label=label)
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path.parent, flags)
    opened = os.fstat(descriptor)
    if not stat.S_ISDIR(opened.st_mode):
        os.close(descriptor)
        raise RecoveryError(f"{label} parent is not an ordinary directory")
    return descriptor


def _open_verified_single_link_file(
    path: Path, *, flags: int, label: str
) -> tuple[int, int, tuple[int, int], tuple[int, int, int]]:
    """Open by parent fd and bind lstat/fstat to one single-link inode."""
    parent_descriptor = _open_parent_directory(path, label=label)
    descriptor: int | None = None
    try:
        before = os.stat(
            path.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        _require_regular_single_link_stat(before, label=label)
        selected_flags = flags
        if hasattr(os, "O_NOFOLLOW"):
            selected_flags |= os.O_NOFOLLOW
        descriptor = os.open(path.name, selected_flags, dir_fd=parent_descriptor)
        opened = os.fstat(descriptor)
        _require_regular_single_link_stat(opened, label=label)
        identity = (before.st_dev, before.st_ino)
        if identity != (opened.st_dev, opened.st_ino):
            raise RecoveryError(f"{label} inode changed while opening")
        stable = (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        return descriptor, parent_descriptor, identity, stable
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)
        raise


def _verify_open_single_link_file(
    path: Path,
    *,
    descriptor: int,
    parent_descriptor: int,
    identity: tuple[int, int],
    label: str,
) -> os.stat_result:
    opened = os.fstat(descriptor)
    current = os.stat(
        path.name, dir_fd=parent_descriptor, follow_symlinks=False
    )
    _require_regular_single_link_stat(opened, label=label)
    _require_regular_single_link_stat(current, label=label)
    if (
        (opened.st_dev, opened.st_ino) != identity
        or (current.st_dev, current.st_ino) != identity
    ):
        raise RecoveryError(f"{label} inode changed during access")
    return opened


def _read_single_link_bytes(path: Path, *, label: str) -> bytes:
    descriptor, parent_descriptor, identity, stable = (
        _open_verified_single_link_file(path, flags=os.O_RDONLY, label=label)
    )
    try:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        opened = _verify_open_single_link_file(
            path,
            descriptor=descriptor,
            parent_descriptor=parent_descriptor,
            identity=identity,
            label=label,
        )
        if (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != stable:
            raise RecoveryError(f"{label} changed while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)


def _single_link_file_sha256(path: Path, *, label: str) -> str:
    return hashlib.sha256(_read_single_link_bytes(path, label=label)).hexdigest()


def _read_single_link_text(path: Path, *, label: str) -> str:
    try:
        return _read_single_link_bytes(path, label=label).decode("utf-8")
    except UnicodeDecodeError as error:
        raise RecoveryError(f"{label} is not UTF-8") from error


def _load_single_link_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(_read_single_link_text(path, label=label))
    except json.JSONDecodeError as error:
        raise RecoveryError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise RecoveryError(f"{label} must contain a JSON object")
    support.canonical_json_dumps(value)
    return value


def _rename_noreplace(source: Path, target: Path) -> None:
    """Atomically rename on one filesystem while refusing an existing target."""
    if target.exists() or target.is_symlink():
        raise RecoveryError(f"Recovery target already exists: {target}")
    if source.is_symlink() or not source.exists():
        raise RecoveryError("Atomic rename source is missing or a symlink")
    if not source.is_file() and not source.is_dir():
        raise RecoveryError("Atomic rename source is not a regular file/directory")
    source_is_file = source.is_file() and not source.is_symlink()
    if source_is_file:
        descriptor, source_parent, identity, _ = _open_verified_single_link_file(
            source, flags=os.O_RDONLY, label="atomic rename source"
        )
        target_parent = _open_parent_directory(
            target, label="atomic rename target"
        )
        try:
            try:
                os.stat(
                    target.name,
                    dir_fd=target_parent,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise RecoveryError(f"Recovery target appeared: {target}")
            _verify_open_single_link_file(
                source,
                descriptor=descriptor,
                parent_descriptor=source_parent,
                identity=identity,
                label="atomic rename source",
            )
            libc = ctypes.CDLL(None, use_errno=True)
            result: int | None = None
            if sys.platform == "darwin" and hasattr(libc, "renameatx_np"):
                function = libc.renameatx_np
                function.argtypes = [
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_uint,
                ]
                function.restype = ctypes.c_int
                result = function(
                    source_parent,
                    os.fsencode(source.name),
                    target_parent,
                    os.fsencode(target.name),
                    0x00000004,  # RENAME_EXCL
                )
            elif hasattr(libc, "renameat2"):
                function = libc.renameat2
                function.argtypes = [
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_uint,
                ]
                function.restype = ctypes.c_int
                result = function(
                    source_parent,
                    os.fsencode(source.name),
                    target_parent,
                    os.fsencode(target.name),
                    1,  # RENAME_NOREPLACE
                )
            else:
                # Hard-link + unlink has no overwrite window and is safe for
                # same-filesystem regular files.  The post-check below also
                # rejects a concurrent extra hard link.
                try:
                    os.link(
                        source.name,
                        target.name,
                        src_dir_fd=source_parent,
                        dst_dir_fd=target_parent,
                        follow_symlinks=False,
                    )
                except FileExistsError as error:
                    raise RecoveryError(
                        f"Recovery target appeared: {target}"
                    ) from error
                except OSError as error:
                    raise RecoveryError(
                        f"Atomic no-replace link failed: {error}"
                    ) from error
                os.unlink(source.name, dir_fd=source_parent)
            if result is not None and result != 0:
                error_number = ctypes.get_errno()
                if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise RecoveryError(f"Recovery target appeared: {target}")
                raise RecoveryError(
                    f"Atomic no-replace rename failed ({error_number}): "
                    f"{os.strerror(error_number)}"
                )
            opened = os.fstat(descriptor)
            installed = os.stat(
                target.name,
                dir_fd=target_parent,
                follow_symlinks=False,
            )
            _require_regular_single_link_stat(
                opened, label="atomic rename source descriptor"
            )
            _require_regular_single_link_stat(
                installed, label="atomic rename target"
            )
            if (
                (opened.st_dev, opened.st_ino) != identity
                or (installed.st_dev, installed.st_ino) != identity
            ):
                raise RecoveryError("Atomic rename installed a different inode")
            os.fsync(target_parent)
            if source_parent != target_parent:
                os.fsync(source_parent)
        finally:
            os.close(descriptor)
            os.close(source_parent)
            os.close(target_parent)
        _read_single_link_bytes(target, label="atomic rename target")
        return
    libc = ctypes.CDLL(None, use_errno=True)
    source_raw = os.fsencode(source)
    target_raw = os.fsencode(target)
    result: int | None = None
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        function = libc.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(source_raw, target_raw, 0x00000004)  # RENAME_EXCL
    elif hasattr(libc, "renameat2"):
        function = libc.renameat2
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(-100, source_raw, -100, target_raw, 1)  # RENAME_NOREPLACE
    else:  # pragma: no cover - supported macOS/Linux execution environments.
        raise RecoveryError("This platform has no atomic no-replace directory rename")
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise RecoveryError(f"Recovery target appeared: {target}")
        raise RecoveryError(
            f"Atomic no-replace rename failed ({error_number}): "
            f"{os.strerror(error_number)}"
        )
    _fsync_directory(target.parent)
    if source.parent != target.parent:
        _fsync_directory(source.parent)


def _pending_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.fast-recovery.pending")


def _assert_no_symlink_ancestors(path: Path, *, label: str) -> None:
    try:
        absolute = Path(os.path.abspath(path))
        relative = absolute.relative_to(ROOT)
        path.resolve(strict=False).relative_to(ROOT)
    except ValueError as error:
        raise RecoveryError(f"{label} escapes the repository") from error
    current = ROOT
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise RecoveryError(f"{label} has a symlink ancestor: {current}")


def _atomic_write_bytes_once(
    path: Path,
    payload: bytes,
    *,
    mode: int,
    label: str,
    fault_hook: FaultHook | None = None,
) -> None:
    """Resume a write through a deterministic pending file, then install once."""
    _assert_no_symlink_ancestors(path, label=label)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RecoveryError(f"{label} target must not be a symlink")
    pending = _pending_path(path)
    if path.exists():
        if _read_single_link_bytes(path, label=f"existing {label}") != payload:
            raise RecoveryError(f"Existing immutable {label} differs")
        if pending.exists() or pending.is_symlink():
            raise RecoveryError(f"Installed {label} has an ambiguous pending copy")
        return
    if pending.is_symlink() or (pending.exists() and not pending.is_file()):
        raise RecoveryError(f"{label} pending path is not an ordinary file")
    if pending.exists():
        prefix = _read_single_link_bytes(
            pending, label=f"interrupted {label} pending file"
        )
        if not payload.startswith(prefix):
            raise RecoveryError(f"Interrupted {label} is not an exact payload prefix")
        if prefix == payload:
            descriptor, parent_descriptor, identity, _ = (
                _open_verified_single_link_file(
                    pending,
                    flags=os.O_RDONLY,
                    label=f"complete interrupted {label} pending file",
                )
            )
            try:
                opened = _verify_open_single_link_file(
                    pending,
                    descriptor=descriptor,
                    parent_descriptor=parent_descriptor,
                    identity=identity,
                    label=f"complete interrupted {label} pending file",
                )
                pending_has_final_mode = stat.S_IMODE(opened.st_mode) == mode
            finally:
                os.close(descriptor)
                os.close(parent_descriptor)
            if pending_has_final_mode:
                # A crash at ``pending_complete`` leaves a read-only file for
                # immutable records.  It is already complete and must never be
                # reopened O_RDWR merely to repeat an already-persisted chmod.
                _rename_noreplace(pending, path)
                if _read_single_link_bytes(
                    path, label=f"installed {label}"
                ) != payload:
                    raise RecoveryError(f"Installed immutable {label} differs")
                _fault(fault_hook, f"{label}:installed")
                return
    else:
        parent_descriptor = _open_parent_directory(pending, label=label)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor: int | None = None
        try:
            descriptor = os.open(
                pending.name,
                flags,
                0o600,
                dir_fd=parent_descriptor,
            )
        except FileExistsError:
            os.close(parent_descriptor)
            return _atomic_write_bytes_once(
                path,
                payload,
                mode=mode,
                label=label,
                fault_hook=fault_hook,
            )
        except BaseException:
            os.close(parent_descriptor)
            raise
        try:
            assert descriptor is not None
            created = os.fstat(descriptor)
            current = os.stat(
                pending.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            _require_regular_single_link_stat(created, label=label)
            _require_regular_single_link_stat(current, label=label)
            if (created.st_dev, created.st_ino) != (current.st_dev, current.st_ino):
                raise RecoveryError(f"{label} pending inode changed at creation")
            os.fsync(descriptor)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent_descriptor)
        prefix = b""
        _fault(fault_hook, f"{label}:pending_created")
    remainder = payload[len(prefix):]
    if remainder:
        split = max(1, len(remainder) // 2)
        for index, chunk in enumerate((remainder[:split], remainder[split:])):
            if not chunk:
                continue
            descriptor, parent_descriptor, identity, _ = (
                _open_verified_single_link_file(
                    pending,
                    flags=os.O_WRONLY | os.O_APPEND,
                    label=f"{label} pending append",
                )
            )
            try:
                written = 0
                while written < len(chunk):
                    count = os.write(descriptor, chunk[written:])
                    if count <= 0:  # pragma: no cover - defensive OS boundary.
                        raise RecoveryError(f"Interrupted zero-byte write for {label}")
                    written += count
                os.fsync(descriptor)
                _verify_open_single_link_file(
                    pending,
                    descriptor=descriptor,
                    parent_descriptor=parent_descriptor,
                    identity=identity,
                    label=f"{label} pending append",
                )
            finally:
                os.close(descriptor)
                os.close(parent_descriptor)
            if index == 0:
                _fault(fault_hook, f"{label}:partial_written")
    descriptor, parent_descriptor, identity, _ = _open_verified_single_link_file(
        pending, flags=os.O_RDWR, label=f"completed {label} pending file"
    )
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        if b"".join(chunks) != payload:
            raise RecoveryError(f"Completed {label} pending payload differs")
        _verify_open_single_link_file(
            pending,
            descriptor=descriptor,
            parent_descriptor=parent_descriptor,
            identity=identity,
            label=f"completed {label} pending file",
        )
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        _verify_open_single_link_file(
            pending,
            descriptor=descriptor,
            parent_descriptor=parent_descriptor,
            identity=identity,
            label=f"completed {label} pending file",
        )
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)
    _fault(fault_hook, f"{label}:pending_complete")
    _rename_noreplace(pending, path)
    if _read_single_link_bytes(path, label=f"installed {label}") != payload:
        raise RecoveryError(f"Installed immutable {label} differs")
    _fault(fault_hook, f"{label}:installed")


def _atomic_write_json_once(
    path: Path,
    document: Mapping[str, Any],
    *,
    label: str,
    mode: int = 0o444,
    fault_hook: FaultHook | None = None,
) -> None:
    _atomic_write_bytes_once(
        path,
        (support.canonical_json_dumps(document) + "\n").encode("utf-8"),
        mode=mode,
        label=label,
        fault_hook=fault_hook,
    )


@contextmanager
def _exclusive_recovery_lock() -> Iterator[None]:
    RECOVERY_MUTEX.parent.mkdir(parents=True, exist_ok=True)
    if RECOVERY_MUTEX.is_symlink():
        raise RecoveryError("Recovery mutex must not be a symlink")
    if not RECOVERY_MUTEX.exists():
        parent_descriptor = _open_parent_directory(
            RECOVERY_MUTEX, label="recovery mutex"
        )
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            created = os.open(
                RECOVERY_MUTEX.name,
                flags,
                0o600,
                dir_fd=parent_descriptor,
            )
        except FileExistsError:
            pass
        else:
            os.close(created)
        finally:
            os.close(parent_descriptor)
    descriptor, parent_descriptor, identity, _ = _open_verified_single_link_file(
        RECOVERY_MUTEX, flags=os.O_RDWR, label="recovery mutex"
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        _verify_open_single_link_file(
            RECOVERY_MUTEX,
            descriptor=descriptor,
            parent_descriptor=parent_descriptor,
            identity=identity,
            label="recovery mutex",
        )
        yield
    finally:
        try:
            _verify_open_single_link_file(
                RECOVERY_MUTEX,
                descriptor=descriptor,
                parent_descriptor=parent_descriptor,
                identity=identity,
                label="recovery mutex",
            )
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
            os.close(parent_descriptor)


def _exact_keys(value: Any, keys: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RecoveryError(f"{label} schema differs from the reviewed recovery")
    return value


def _repo_path(raw: Any, *, label: str, must_exist: bool) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise RecoveryError(f"{label} must be one repository-relative path")
    unresolved = ROOT / raw
    if unresolved.is_symlink():
        raise RecoveryError(f"{label} must not be a symlink")
    path = unresolved.resolve(strict=must_exist)
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise RecoveryError(f"{label} escapes the repository") from error
    return path


def _relative(path: Path) -> str:
    return str(path.resolve(strict=False).relative_to(ROOT))


def _tree_hashes(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_dir():
        raise RecoveryError(f"Expected an ordinary recovery directory: {path}")
    result: dict[str, str] = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise RecoveryError(f"Recovery source tree contains a symlink: {item}")
        if item.is_file():
            result[str(item.relative_to(path))] = _single_link_file_sha256(
                item, label="recovery tree member"
            )
        elif not item.is_dir():
            raise RecoveryError(f"Recovery source tree contains a special file: {item}")
    if not result:
        raise RecoveryError(f"Recovery source tree is empty: {path}")
    return result


def _load_bad_lock_at(path: Path) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or _single_link_file_sha256(path, label="failed primary lock")
        != BAD_LOCK_FILE_SHA256
    ):
        raise RecoveryError("The exact failed primary lock is not present")
    lock = _load_single_link_json(path, label="failed long-slug primary lock")
    unhashed = dict(lock)
    stored = unhashed.pop("lock_payload_sha256", None)
    if (
        stored != BAD_LOCK_PAYLOAD_SHA256
        or stored != support.canonical_sha256(unhashed)
        or lock.get("schema_version") != 2
        or lock.get("mode") != "loss_primary"
        or lock.get("effective_stage") != support.LOSS_PRIMARY_STAGE
        or lock.get("execution_status") != "runnable"
    ):
        raise RecoveryError("Failed primary lock identity/hash differs")
    resolved = lock.get("resolved_stage")
    variants = resolved.get("variants") if isinstance(resolved, Mapping) else None
    if not isinstance(variants, list) or len(variants) != len(BAD_VARIANTS):
        raise RecoveryError("Failed primary lock does not contain four variants")
    for variant, expected in zip(variants, BAD_VARIANTS):
        loss_variant, experiment, slug, family_sha = expected
        if (
            not isinstance(variant, Mapping)
            or variant.get("loss_variant") != loss_variant
            or variant.get("experiment") != experiment
            or variant.get("kernel_slug") != slug
            or variant.get("title") != slug
            or variant.get("seed") != 42
            or variant.get("expected_recipe_family_sha256") != family_sha
            or len(slug) <= support.KAGGLE_REMOTE_IDENTITY_MAX_LENGTH
        ):
            raise RecoveryError("Failed primary variant identity differs")
    return lock


def _load_bad_lock() -> dict[str, Any]:
    return _load_bad_lock_at(BAD_LOCK)


def _validate_bad_provenance() -> tuple[Path, Path]:
    manifest_path = Path(str(BAD_LOCK) + ".trusted-provenance.json")
    archive_dir = Path(str(BAD_LOCK) + ".trusted-provenance")
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or _single_link_file_sha256(
            manifest_path, label="failed trusted-provenance manifest"
        )
        != BAD_PROVENANCE_MANIFEST_FILE_SHA256
    ):
        raise RecoveryError("Failed lock provenance manifest differs")
    manifest = _load_single_link_json(
        manifest_path, label="failed lock trusted provenance manifest"
    )
    if Path(str(manifest.get("archive_dir"))).resolve(strict=True) != archive_dir:
        raise RecoveryError("Failed lock provenance archive path differs")
    sources = manifest.get("source_documents")
    if not isinstance(sources, list) or len(sources) != 2:
        raise RecoveryError("Failed lock provenance source ledger differs")
    expected_files: set[Path] = set()
    for source in sources:
        if not isinstance(source, Mapping):
            raise RecoveryError("Failed lock provenance source is malformed")
        snapshot = Path(str(source.get("snapshot_path"))).resolve(strict=True)
        if snapshot.parent != archive_dir / "sources":
            raise RecoveryError("Failed lock provenance snapshot escaped its archive")
        if _single_link_file_sha256(
            snapshot, label="failed trusted-provenance snapshot"
        ) != source.get("snapshot_file_sha256"):
            raise RecoveryError("Failed lock provenance snapshot hash differs")
        expected_files.add(snapshot)
    actual_files = {path for path in archive_dir.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise RecoveryError("Failed lock provenance archive file set differs")
    return manifest_path, archive_dir


def _validate_failed_controller_state(
    receipt_failed: Mapping[str, Any],
) -> None:
    _exact_keys(
        receipt_failed,
        {
            "controller_state_path", "controller_state_file_sha256",
            "controller_state_payload_sha256", "status", "stop_reason",
            "source_lock_payload_sha256", "submit_wait",
            "savekernel_http400_output_persisted", "savekernel_error_output",
            "evidence_limitation",
        },
        label="failed submit evidence",
    )
    path = _repo_path(
        receipt_failed["controller_state_path"],
        label="failed controller state",
        must_exist=True,
    )
    if (
        path != FAILED_CONTROLLER_STATE.resolve(strict=True)
        or _single_link_file_sha256(path, label="failed controller state")
        != receipt_failed["controller_state_file_sha256"]
    ):
        raise RecoveryError("Failed controller state file binding differs")
    state = _load_single_link_json(path, label="failed loss controller state")
    unhashed = dict(state)
    stored = unhashed.pop("controller_state_payload_sha256", None)
    if (
        stored != receipt_failed["controller_state_payload_sha256"]
        or stored != support.canonical_sha256(unhashed)
        or state.get("status") != "failed_closed"
        or state.get("stop_reason") != "kaggle_or_api_failure"
        or receipt_failed.get("status") != "failed_closed"
        or receipt_failed.get("stop_reason") != "kaggle_or_api_failure"
        or receipt_failed.get("source_lock_payload_sha256")
        != BAD_LOCK_PAYLOAD_SHA256
        or receipt_failed.get("savekernel_http400_output_persisted") is not False
        or receipt_failed.get("savekernel_error_output") is not None
    ):
        raise RecoveryError("Failed controller state semantics differ")
    submit = _exact_keys(
        receipt_failed.get("submit_wait"),
        {"argv", "phase", "returncode"},
        label="failed submit command",
    )
    matching = [
        row
        for row in state.get("executed_commands", [])
        if isinstance(row, Mapping) and row.get("phase") == "submit_wait"
    ]
    if (
        len(matching) != 1
        or dict(submit) != dict(matching[0])
        or submit.get("returncode") != 1
        or submit.get("phase") != "submit_wait"
        or not isinstance(submit.get("argv"), list)
        or submit["argv"][-2:] != ["--submit", "--wait"]
        or "--force-resubmit" in submit["argv"]
        or state.get("planned_action", {}).get("lock_sha256")
        != BAD_LOCK_PAYLOAD_SHA256
    ):
        raise RecoveryError("Failed submit command evidence differs")


def _validate_remote_observations(
    observations: Any, *, owner: str, slugs: Sequence[str]
) -> None:
    if not isinstance(observations, list) or len(observations) != len(slugs):
        raise RecoveryError("Remote absence observation count differs")
    for observation, slug in zip(observations, slugs):
        row = _exact_keys(
            observation,
            {
                "owner_slug", "kernel_slug", "status_command", "list_command",
                "files_command", "remote_kernel_identity_state",
                "remote_outputs_state",
            },
            label="remote absence observation",
        )
        owner_slug = f"{owner}/{slug}"
        if (
            row.get("owner_slug") != owner_slug
            or row.get("kernel_slug") != slug
            or row.get("remote_kernel_identity_state")
            != "absent_from_authenticated_owner_scope"
            or row.get("remote_outputs_state")
            != "unknown_readonly_endpoint_forbidden"
        ):
            raise RecoveryError("Remote absence identity/result differs")
        status = _exact_keys(
            row.get("status_command"),
            {"argv", "returncode", "combined_output", "normalized_result"},
            label="read-only status command",
        )
        listing = _exact_keys(
            row.get("list_command"),
            {
                "argv", "returncode", "combined_output", "exact_ref_matches",
                "normalized_result",
            },
            label="read-only owner-list command",
        )
        files = _exact_keys(
            row.get("files_command"),
            {"argv", "returncode", "combined_output", "normalized_result"},
            label="read-only files command",
        )
        if (
            status.get("argv")
            != [".venv/bin/kaggle", "kernels", "status", owner_slug]
            or status.get("returncode") != 1
            or status.get("normalized_result") != "kernels_get_denied"
            or owner_slug not in str(status.get("combined_output"))
            or listing.get("argv")
            != [
                ".venv/bin/kaggle", "kernels", "list", "--mine",
                "--search", slug, "--page-size", "20",
            ]
            or listing.get("returncode") != 0
            or listing.get("combined_output") != "Not found\n"
            or listing.get("exact_ref_matches") != 0
            or listing.get("normalized_result") != "owner_list_not_found"
            or files.get("argv")
            != [
                ".venv/bin/kaggle", "kernels", "files", owner_slug,
                "--format", "json", "--page-size", "200",
            ]
            or files.get("returncode") != 1
            or files.get("normalized_result")
            != "readonly_output_endpoint_forbidden"
            or "ListKernelFiles" not in str(files.get("combined_output"))
        ):
            raise RecoveryError("Read-only remote command evidence differs")


def _validate_local_evidence(receipt: Mapping[str, Any]) -> None:
    staging = receipt.get("local_staging_metadata")
    artifacts = receipt.get("local_artifacts")
    if (
        not isinstance(staging, list)
        or not isinstance(artifacts, list)
        or len(staging) != len(BAD_SLUGS)
        or len(artifacts) != len(BAD_SLUGS)
    ):
        raise RecoveryError("Local recovery evidence count differs")
    for metadata_ref, artifact_ref, slug in zip(staging, artifacts, BAD_SLUGS):
        metadata = _exact_keys(
            metadata_ref,
            {
                "kernel_slug", "path", "file_sha256", "id", "title",
                "title_length", "slug_length",
            },
            label="local staging metadata evidence",
        )
        metadata_path = _repo_path(
            metadata["path"], label="staged Kaggle metadata", must_exist=True
        )
        document = _load_single_link_json(
            metadata_path, label="staged long-slug Kaggle metadata"
        )
        if (
            metadata.get("kernel_slug") != slug
            or metadata_path
            != (ROOT / ".kaggle" / "staging" / slug / "kernel-metadata.json")
            or _single_link_file_sha256(
                metadata_path, label="staged long-slug Kaggle metadata"
            )
            != metadata.get("file_sha256")
            or metadata.get("id") != f"{EXPECTED_OWNER}/{slug}"
            or metadata.get("title") != slug
            or metadata.get("title_length") != len(slug)
            or metadata.get("slug_length") != len(slug)
            or len(slug) <= support.KAGGLE_REMOTE_IDENTITY_MAX_LENGTH
            or document.get("id") != f"{EXPECTED_OWNER}/{slug}"
            or document.get("title") != slug
        ):
            raise RecoveryError("Staged long-slug metadata binding differs")
        artifact = _exact_keys(
            artifact_ref,
            {"kernel_slug", "path", "exists", "file_count"},
            label="local output evidence",
        )
        artifact_path = _repo_path(
            artifact["path"], label="absent local output", must_exist=False
        )
        if (
            artifact.get("kernel_slug") != slug
            or artifact_path != support.DEFAULT_ARTIFACTS_DIR / slug
            or artifact.get("exists") is not False
            or artifact.get("file_count") != 0
            or artifact_path.exists()
            or artifact_path.is_symlink()
        ):
            raise RecoveryError("A failed long-slug run has local outputs")


def validate_audit_receipt(path: Path = AUDIT_RECEIPT) -> dict[str, Any]:
    if path.resolve(strict=True) != AUDIT_RECEIPT.resolve(strict=True):
        raise RecoveryError("Recovery requires the reviewed audit receipt path")
    if _single_link_file_sha256(path, label="recovery audit receipt") != (
        AUDIT_RECEIPT_FILE_SHA256
    ):
        raise RecoveryError("Recovery audit receipt file SHA differs")
    receipt = _load_single_link_json(
        path, label="long-slug absence audit receipt"
    )
    _exact_keys(
        receipt,
        {
            "schema_version", "kind", "campaign", "audited_at_utc",
            "audit_scope", "source_lock", "failed_submit", "expected_owner",
            "expected_bad_kernel_slugs", "remote_observations",
            "local_staging_metadata", "local_artifacts", "assertions",
            "receipt_hash_contract", "receipt_payload_sha256",
        },
        label="long-slug absence audit receipt",
    )
    unhashed = dict(receipt)
    stored = unhashed.pop("receipt_payload_sha256", None)
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind")
        != "minilm_5ep_sft_long_slug_recovery_absence_receipt"
        or receipt.get("campaign") != "minilm_5ep_sft_hparam_search_v1"
        or stored != AUDIT_RECEIPT_PAYLOAD_SHA256
        or stored != support.canonical_sha256(unhashed)
        or receipt.get("expected_owner") != EXPECTED_OWNER
        or receipt.get("expected_bad_kernel_slugs") != list(BAD_SLUGS)
    ):
        raise RecoveryError("Recovery audit receipt identity/hash differs")
    source = _exact_keys(
        receipt.get("source_lock"),
        {
            "path", "file_sha256", "lock_payload_sha256", "schema_version",
            "execution_status",
        },
        label="audit source lock",
    )
    source_path = _repo_path(
        source["path"], label="audit source lock", must_exist=True
    )
    if (
        source_path != BAD_LOCK.resolve(strict=True)
        or source.get("file_sha256") != BAD_LOCK_FILE_SHA256
        or _single_link_file_sha256(source_path, label="audit source lock")
        != BAD_LOCK_FILE_SHA256
        or source.get("lock_payload_sha256") != BAD_LOCK_PAYLOAD_SHA256
        or source.get("schema_version") != 2
        or source.get("execution_status") != "runnable"
    ):
        raise RecoveryError("Audit source-lock binding differs")
    _validate_failed_controller_state(
        _exact_keys(
            receipt.get("failed_submit"),
            {
                "controller_state_path", "controller_state_file_sha256",
                "controller_state_payload_sha256", "status", "stop_reason",
                "source_lock_payload_sha256", "submit_wait",
                "savekernel_http400_output_persisted", "savekernel_error_output",
                "evidence_limitation",
            },
            label="failed submit evidence",
        )
    )
    _validate_remote_observations(
        receipt.get("remote_observations"),
        owner=EXPECTED_OWNER,
        slugs=BAD_SLUGS,
    )
    _validate_local_evidence(receipt)
    assertions = _exact_keys(
        receipt.get("assertions"),
        {
            "expected_identity_count", "source_lock_identity_set_matches",
            "all_status_requests_failed_as_missing_or_inaccessible",
            "all_authenticated_owner_lists_returned_not_found",
            "all_exact_owner_list_match_counts_zero",
            "all_remote_kernel_identities_absent_from_authenticated_owner_scope",
            "remote_output_file_counts_proven_zero",
            "all_remote_output_metadata_requests_forbidden",
            "all_local_staging_metadata_bound", "all_local_artifact_dirs_absent",
            "all_local_downloaded_output_file_counts_zero", "no_mutating_commands",
        },
        label="recovery audit assertions",
    )
    required_true = set(assertions) - {
        "expected_identity_count", "remote_output_file_counts_proven_zero"
    }
    if (
        assertions.get("expected_identity_count") != 4
        or assertions.get("remote_output_file_counts_proven_zero") is not False
        or any(assertions.get(key) is not True for key in required_true)
    ):
        raise RecoveryError("Recovery audit assertions differ")
    return receipt


def _validate_generated_notebooks(lock: Mapping[str, Any]) -> list[Path]:
    variants = lock["resolved_stage"]["variants"]
    result: list[Path] = []
    for variant in variants:
        path = (
            ROOT
            / "notebooks"
            / "minilm_5ep_team_ablation"
            / "sft_hparams_v1"
            / f"{variant['experiment']}_2xt4.ipynb"
        )
        if path.is_symlink() or not path.is_file():
            raise RecoveryError(f"Failed-attempt notebook is missing: {path}")
        document = _load_single_link_json(path, label="failed-attempt notebook")
        metadata = document.get("metadata", {}).get("product_matching_training", {})
        stage_lock = metadata.get("stage_lock") if isinstance(metadata, Mapping) else None
        if (
            metadata.get("experiment") != variant["experiment"]
            or not isinstance(stage_lock, Mapping)
            or stage_lock.get("lock_payload_sha256") != BAD_LOCK_PAYLOAD_SHA256
        ):
            raise RecoveryError("Failed-attempt notebook lock binding differs")
        result.append(path)
    return result


def _expected_move_paths() -> list[tuple[str, Path, Path, str]]:
    provenance_manifest = Path(str(BAD_LOCK) + ".trusted-provenance.json")
    provenance_dir = Path(str(BAD_LOCK) + ".trusted-provenance")
    notebooks = [
        ROOT
        / "notebooks"
        / "minilm_5ep_team_ablation"
        / "sft_hparams_v1"
        / f"{experiment}_2xt4.ipynb"
        for _, experiment, _, _ in BAD_VARIANTS
    ]
    staging = [ROOT / ".kaggle" / "staging" / slug for slug in BAD_SLUGS]
    result: list[tuple[str, Path, Path, str]] = [
        ("source_lock", BAD_LOCK, RECOVERY_RELOCATED_LOCK, "file"),
        (
            "trusted_provenance_manifest_forensic_original",
            provenance_manifest,
            RECOVERY_FORENSIC_MANIFEST,
            "file",
        ),
        (
            "trusted_provenance_archive",
            provenance_dir,
            RECOVERY_RELOCATED_ARCHIVE,
            "directory",
        ),
        (
            "failed_controller_state",
            FAILED_CONTROLLER_STATE,
            RECOVERY_ARCHIVE / "evidence" / FAILED_CONTROLLER_STATE.name,
            "file",
        ),
    ]
    result.extend(
        (
            "generated_notebook",
            path,
            RECOVERY_ARCHIVE / "notebooks" / path.name,
            "file",
        )
        for path in notebooks
    )
    result.extend(
        (
            "kaggle_staging",
            path,
            RECOVERY_ARCHIVE / "staging" / path.name,
            "directory",
        )
        for path in staging
    )
    return result


def _move_payload(
    *, index: int, role: str, source: Path, target: Path, kind: str
) -> dict[str, Any]:
    if source.is_symlink() or not source.exists():
        raise RecoveryError(f"Recovery source is missing or a symlink: {source}")
    if (kind == "file") != source.is_file() or (kind == "directory") != source.is_dir():
        raise RecoveryError(f"Recovery source kind differs: {source}")
    if target.exists() or target.is_symlink():
        raise RecoveryError(f"Recovery archive target already exists: {target}")
    result: dict[str, Any] = {
        "index": index,
        "role": role,
        "source": _relative(source),
        "target": _relative(target),
        "kind": kind,
        "file_sha256": _single_link_file_sha256(
            source, label="recovery move source"
        )
        if kind == "file"
        else None,
        "tree_file_sha256s": _tree_hashes(source) if kind == "directory" else None,
    }
    result["move_payload_sha256"] = support.canonical_sha256(result)
    return result


def build_preflight() -> dict[str, Any]:
    support.load_freeze_manifest()
    receipt = support.validate_receipt()
    if receipt["budget"]["unique_kernels"] != 18:
        raise RecoveryError("Skip receipt no longer binds the exact prior-18 budget")
    lock = _load_bad_lock()
    _validate_bad_provenance()
    audit = validate_audit_receipt()
    _validate_generated_notebooks(lock)
    moves = [
        _move_payload(
            index=index,
            role=role,
            source=source,
            target=target,
            kind=kind,
        )
        for index, (role, source, target, kind) in enumerate(_expected_move_paths())
    ]
    result: dict[str, Any] = {
        "schema_version": 2,
        "kind": "minilm_5ep_sft_short_slug_recovery_preflight",
        "campaign": "minilm_5ep_sft_hparam_search_v1",
        "transaction_id": BAD_LOCK_PAYLOAD_SHA256[:16],
        "source_lock_payload_sha256": BAD_LOCK_PAYLOAD_SHA256,
        "recovery_archive": _relative(RECOVERY_ARCHIVE),
        "journal_directory": _relative(RECOVERY_JOURNAL),
        "audit_receipt_path": _relative(AUDIT_RECEIPT),
        "audit_receipt_file_sha256": _single_link_file_sha256(
            AUDIT_RECEIPT, label="recovery audit receipt"
        ),
        "audit_receipt_payload_sha256": audit["receipt_payload_sha256"],
        "receipt_summary_payload_sha256": receipt["summary_payload_sha256"],
        "prior_unique_kernels": receipt["budget"]["unique_kernels"],
        "remote_identity_evidence": (
            "four exact owner-list absences; no remote mutation"
        ),
        "remote_output_evidence_limitation": (
            "files endpoint forbidden; no remote identity exists and local outputs "
            "are zero"
        ),
        "transaction_protocol": {
            "archive_moves": "atomic_noreplace_with_exact_source_target_reconciliation",
            "journal": "immutable_one_record_per_completed_transition",
            "materialization_writes": "resumable_pending_then_atomic_noreplace",
            "forensic_manifest": "original_bytes_only_not_active_authority",
            "relocated_old_authority": "core_validated_inside_recovery_archive",
        },
        "moves": moves,
        "kaggle_mutations": 0,
    }
    result["preflight_payload_sha256"] = support.canonical_sha256(result)
    return result


def _validate_preflight_document(preflight: Mapping[str, Any]) -> dict[str, Any]:
    expected_top = {
        "schema_version", "kind", "campaign", "transaction_id",
        "source_lock_payload_sha256", "recovery_archive", "journal_directory",
        "audit_receipt_path", "audit_receipt_file_sha256",
        "audit_receipt_payload_sha256", "receipt_summary_payload_sha256",
        "prior_unique_kernels", "remote_identity_evidence",
        "remote_output_evidence_limitation", "transaction_protocol", "moves",
        "kaggle_mutations", "preflight_payload_sha256",
    }
    payload = dict(_exact_keys(preflight, expected_top, label="recovery preflight"))
    unhashed = dict(payload)
    stored = unhashed.pop("preflight_payload_sha256", None)
    receipt = support.validate_receipt()
    if (
        AUDIT_RECEIPT.is_symlink()
        or not AUDIT_RECEIPT.is_file()
        or _single_link_file_sha256(
            AUDIT_RECEIPT, label="reviewed recovery audit receipt"
        )
        != AUDIT_RECEIPT_FILE_SHA256
    ):
        raise RecoveryError("Reviewed recovery audit receipt differs")
    audit = _load_single_link_json(
        AUDIT_RECEIPT, label="reviewed recovery audit receipt"
    )
    audit_unhashed = dict(audit)
    audit_stored = audit_unhashed.pop("receipt_payload_sha256", None)
    if (
        audit_stored != AUDIT_RECEIPT_PAYLOAD_SHA256
        or audit_stored != support.canonical_sha256(audit_unhashed)
    ):
        raise RecoveryError("Reviewed recovery audit payload differs")
    if (
        payload.get("schema_version") != 2
        or payload.get("kind") != "minilm_5ep_sft_short_slug_recovery_preflight"
        or payload.get("campaign") != "minilm_5ep_sft_hparam_search_v1"
        or payload.get("transaction_id") != BAD_LOCK_PAYLOAD_SHA256[:16]
        or payload.get("source_lock_payload_sha256") != BAD_LOCK_PAYLOAD_SHA256
        or payload.get("recovery_archive") != _relative(RECOVERY_ARCHIVE)
        or payload.get("journal_directory") != _relative(RECOVERY_JOURNAL)
        or payload.get("audit_receipt_path") != _relative(AUDIT_RECEIPT)
        or payload.get("audit_receipt_file_sha256") != AUDIT_RECEIPT_FILE_SHA256
        or payload.get("audit_receipt_payload_sha256")
        != AUDIT_RECEIPT_PAYLOAD_SHA256
        or payload.get("receipt_summary_payload_sha256")
        != receipt.get("summary_payload_sha256")
        or payload.get("prior_unique_kernels") != 18
        or payload.get("kaggle_mutations") != 0
        or stored != support.canonical_sha256(unhashed)
    ):
        raise RecoveryError("Recovery preflight identity/hash differs")
    expected_protocol = {
        "archive_moves": "atomic_noreplace_with_exact_source_target_reconciliation",
        "journal": "immutable_one_record_per_completed_transition",
        "materialization_writes": "resumable_pending_then_atomic_noreplace",
        "forensic_manifest": "original_bytes_only_not_active_authority",
        "relocated_old_authority": "core_validated_inside_recovery_archive",
    }
    if payload.get("transaction_protocol") != expected_protocol:
        raise RecoveryError("Recovery transaction protocol differs")
    moves = payload.get("moves")
    expected_paths = _expected_move_paths()
    if not isinstance(moves, list) or len(moves) != len(expected_paths):
        raise RecoveryError("Recovery move ledger count differs")
    for index, (item, expected) in enumerate(zip(moves, expected_paths)):
        row = dict(
            _exact_keys(
                item,
                {
                    "index", "role", "source", "target", "kind",
                    "file_sha256", "tree_file_sha256s", "move_payload_sha256",
                },
                label="recovery move ledger entry",
            )
        )
        unhashed_move = dict(row)
        move_sha = unhashed_move.pop("move_payload_sha256", None)
        role, source, target, kind = expected
        if (
            row.get("index") != index
            or row.get("role") != role
            or row.get("source") != _relative(source)
            or row.get("target") != _relative(target)
            or row.get("kind") != kind
            or move_sha != support.canonical_sha256(unhashed_move)
        ):
            raise RecoveryError("Recovery move ledger identity/hash differs")
        if kind == "file":
            if not isinstance(row.get("file_sha256"), str) or row.get(
                "tree_file_sha256s"
            ) is not None:
                raise RecoveryError("Recovery file ledger hash differs")
        elif (
            row.get("file_sha256") is not None
            or not isinstance(row.get("tree_file_sha256s"), Mapping)
            or not row["tree_file_sha256s"]
        ):
            raise RecoveryError("Recovery directory ledger hash differs")
    if moves[0]["file_sha256"] != BAD_LOCK_FILE_SHA256 or moves[1][
        "file_sha256"
    ] != BAD_PROVENANCE_MANIFEST_FILE_SHA256:
        raise RecoveryError("Recovery immutable source hashes differ")
    return payload


def _artifact_matches(path: Path, item: Mapping[str, Any]) -> bool:
    if path.is_symlink():
        raise RecoveryError(f"Recovery artifact must not be a symlink: {path}")
    if not path.exists():
        return False
    if item.get("kind") == "file":
        if not path.is_file() or stat.S_ISREG(path.stat().st_mode) is False:
            raise RecoveryError(f"Recovery artifact is not an ordinary file: {path}")
        if _single_link_file_sha256(
            path, label="recovery move file"
        ) != item.get("file_sha256"):
            raise RecoveryError(f"Recovery file hash differs: {path}")
        return True
    if item.get("kind") == "directory":
        if _tree_hashes(path) != item.get("tree_file_sha256s"):
            raise RecoveryError(f"Recovery directory tree hash differs: {path}")
        return True
    raise RecoveryError("Recovery move kind differs")


def _artifact_state(path: Path, item: Mapping[str, Any]) -> str:
    """Return absent/exact/different without treating ordinary drift as absence."""
    if path.is_symlink():
        raise RecoveryError(f"Recovery artifact must not be a symlink: {path}")
    if not path.exists():
        return "absent"
    try:
        return "exact" if _artifact_matches(path, item) else "absent"
    except RecoveryError:
        if path.is_file() or path.is_dir():
            return "different"
        raise


def _move_record(item: Mapping[str, Any], preflight_sha: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "minilm_5ep_sft_short_slug_recovery_move_complete",
        "campaign": "minilm_5ep_sft_hparam_search_v1",
        "transaction_id": BAD_LOCK_PAYLOAD_SHA256[:16],
        "preflight_payload_sha256": preflight_sha,
        "move_index": item["index"],
        "move_payload_sha256": item["move_payload_sha256"],
        "source": item["source"],
        "target": item["target"],
        "final_state": "source_absent_target_exact",
    }
    result["journal_payload_sha256"] = support.canonical_sha256(result)
    return result


def _journal_path_for_move(index: int) -> Path:
    return RECOVERY_JOURNAL / f"move_{index:02d}.complete.json"


def _write_journal_once(
    path: Path,
    document: Mapping[str, Any],
    *,
    label: str,
    fault_hook: FaultHook | None,
) -> None:
    _atomic_write_json_once(
        path, document, label=label, fault_hook=fault_hook
    )


def _reconcile_one_move(
    item: Mapping[str, Any],
    *,
    preflight_sha: str,
    fault_hook: FaultHook | None,
) -> None:
    source = _repo_path(item["source"], label="recovery move source", must_exist=False)
    target = _repo_path(item["target"], label="recovery move target", must_exist=False)
    record_path = _journal_path_for_move(item["index"])
    if record_path.exists() or record_path.is_symlink():
        if record_path.is_symlink() or not record_path.is_file():
            raise RecoveryError("Recovery move journal is not an ordinary file")
        _load_move_record(
            record_path, item=item, preflight_sha=preflight_sha
        )
        if not _artifact_matches(target, item):
            raise RecoveryError("Journaled recovery target no longer matches")
        # A journaled source path may later hold the reviewed replacement lock,
        # its new sidecars/notebooks, or a new controller state.  Never mutate it.
        return
    source_state = _artifact_state(source, item)
    target_state = _artifact_state(target, item)
    if source_state == "exact" and target_state == "exact":
        raise RecoveryError("Recovery move is ambiguous: exact source and target both exist")
    if target_state == "different":
        raise RecoveryError("Recovery move target differs from its immutable ledger")
    if target_state == "exact":
        if source_state != "absent":
            raise RecoveryError("Unjournaled recovery target has a live source collision")
    elif source_state == "exact":
        target.parent.mkdir(parents=True, exist_ok=True)
        _fault(fault_hook, f"move_{item['index']:02d}:before_rename")
        _rename_noreplace(source, target)
        _fault(fault_hook, f"move_{item['index']:02d}:after_rename")
        if source.exists() or source.is_symlink() or not _artifact_matches(target, item):
            raise RecoveryError("Recovery move did not reach its exact terminal state")
    else:
        raise RecoveryError("Recovery move lost its exact source and target")
    record = _move_record(item, preflight_sha)
    _write_journal_once(
        _journal_path_for_move(item["index"]),
        record,
        label=f"move_{item['index']:02d}_journal",
        fault_hook=fault_hook,
    )


def _move_preflight_sources(
    preflight: Mapping[str, Any], *, fault_hook: FaultHook | None = None
) -> None:
    validated = _validate_preflight_document(preflight)
    for item in validated["moves"]:
        _reconcile_one_move(
            item,
            preflight_sha=validated["preflight_payload_sha256"],
            fault_hook=fault_hook,
        )


@contextmanager
def _patched_atomic_materialization_writes(
    fault_hook: FaultHook | None = None,
) -> Iterator[None]:
    """Make the frozen materializer's three write-once boundaries crash-safe."""
    adaptive = support.adaptive
    original_snapshot = adaptive._write_source_snapshot_once
    original_manifest = adaptive._write_trusted_provenance_once
    original_lock = adaptive._write_once

    def snapshot_once(path: Path, document: Mapping[str, Any]) -> None:
        serialized = (adaptive.canonical_json_dumps(document) + "\n").encode("utf-8")
        _atomic_write_bytes_once(
            path,
            serialized,
            mode=0o444,
            label=f"materialization_snapshot_{adaptive._summary_document_sha(document)}",
            fault_hook=fault_hook,
        )

    def manifest_once(
        lock_path: Path,
        trusted: Mapping[str, Any],
        *,
        plan: Mapping[str, Any],
    ) -> Path:
        adaptive.validate_trusted_provenance(trusted, plan=plan)
        path = adaptive.trusted_provenance_manifest_path(lock_path)
        _atomic_write_json_once(
            path,
            trusted,
            label="materialization_provenance_manifest",
            fault_hook=fault_hook,
        )
        loaded = adaptive.load_trusted_provenance(path, plan=plan)
        if loaded != dict(trusted):
            raise RecoveryError("Installed materialization provenance differs")
        return path

    def lock_once(
        path: Path,
        payload: Mapping[str, Any],
        *,
        plan: Mapping[str, Any],
        mode: str,
        trusted_provenance: Mapping[str, Any],
    ) -> dict[str, Any]:
        manifest_once(path, trusted_provenance, plan=plan)
        _atomic_write_json_once(
            path,
            payload,
            mode=0o644,
            label="materialization_primary_lock",
            fault_hook=fault_hook,
        )
        existing = adaptive.read_lock(
            path, plan=plan, trusted_provenance=trusted_provenance
        )
        if existing.get("mode") != mode or existing != dict(payload):
            raise RecoveryError("Installed replacement lock differs")
        return existing

    adaptive._write_source_snapshot_once = snapshot_once
    adaptive._write_trusted_provenance_once = manifest_once
    adaptive._write_once = lock_once
    try:
        yield
    finally:
        adaptive._write_source_snapshot_once = original_snapshot
        adaptive._write_trusted_provenance_once = original_manifest
        adaptive._write_once = original_lock


def _materialize_replacement(
    *, fault_hook: FaultHook | None = None
) -> dict[str, Any]:
    argv = [
        str(materializer.__file__),
        "loss_primary",
        "--fast-track-policy", str(support.DEFAULT_POLICY),
        "--fast-track-receipt", str(support.DEFAULT_RECEIPT),
        "--fast-track-freeze-manifest", str(support.DEFAULT_FREEZE_MANIFEST),
        "--plan", str(support.DEFAULT_PLAN),
        "--summary", str(support.DEFAULT_SUMMARY),
        "--artifacts-dir", str(support.DEFAULT_ARTIFACTS_DIR),
        "--prerequisite-lock",
        str(
            support.DEFAULT_REPORT_DIR
            / "stage_locks"
            / "regularization_coordinate_search_classifier_dropout.lock.json"
        ),
        "--history-summary", str(support.DEFAULT_RECEIPT),
        "--source-stage", support.SOURCE_STAGE,
        "--output", str(BAD_LOCK),
    ]
    previous = list(sys.argv)
    sys.argv = argv
    try:
        with _patched_atomic_materialization_writes(fault_hook):
            materializer.main()
    finally:
        sys.argv = previous
    return support.load_scoped_loss_lock(
        plan_path=support.DEFAULT_PLAN,
        stage_lock_path=BAD_LOCK,
    )


def _validate_replacement(
    new_lock: Mapping[str, Any], old_lock: Mapping[str, Any]
) -> list[dict[str, Any]]:
    support._validate_short_remote_lock(new_lock)
    old_variants = old_lock["resolved_stage"]["variants"]
    new_variants = new_lock["resolved_stage"]["variants"]
    if len(old_variants) != 4 or len(new_variants) != 4:
        raise RecoveryError("Replacement primary variant count differs")
    result = []
    stable_keys = (
        "experiment", "loss_variant", "seed", "expected_recipe_sha256",
        "expected_recipe_family_sha256", "expected_loss_hook_sha256",
    )
    for old, new in zip(old_variants, new_variants):
        if any(old.get(key) != new.get(key) for key in stable_keys):
            raise RecoveryError("Replacement changed experiment/recipe/family identity")
        if old.get("kernel_slug") == new.get("kernel_slug"):
            raise RecoveryError("Replacement retained the rejected long remote identity")
        result.append(
            {
                "experiment": new["experiment"],
                "loss_variant": new["loss_variant"],
                "recipe_sha256": new["expected_recipe_sha256"],
                "recipe_family_sha256": new["expected_recipe_family_sha256"],
                "old_kernel_slug": old["kernel_slug"],
                "new_kernel_slug": new["kernel_slug"],
                "new_title": new["title"],
            }
        )
    return result


def _phase_record(
    *,
    phase: str,
    preflight_sha: str,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "minilm_5ep_sft_short_slug_recovery_journal_record",
        "campaign": "minilm_5ep_sft_hparam_search_v1",
        "transaction_id": BAD_LOCK_PAYLOAD_SHA256[:16],
        "preflight_payload_sha256": preflight_sha,
        "phase": phase,
        "details": dict(details),
    }
    result["journal_payload_sha256"] = support.canonical_sha256(result)
    return result


def _phase_path(phase: str) -> Path:
    return RECOVERY_JOURNAL / f"{phase}.json"


def _write_phase_once(
    *,
    phase: str,
    preflight_sha: str,
    details: Mapping[str, Any],
    fault_hook: FaultHook | None,
) -> dict[str, Any]:
    record = _phase_record(
        phase=phase, preflight_sha=preflight_sha, details=details
    )
    _write_journal_once(
        _phase_path(phase),
        record,
        label=f"{phase}_journal",
        fault_hook=fault_hook,
    )
    return record


def _load_phase(path: Path, *, phase: str, preflight_sha: str) -> dict[str, Any]:
    record = _load_single_link_json(path, label=f"recovery {phase} journal")
    _exact_keys(
        record,
        {
            "schema_version", "kind", "campaign", "transaction_id",
            "preflight_payload_sha256", "phase", "details",
            "journal_payload_sha256",
        },
        label=f"recovery {phase} journal",
    )
    unhashed = dict(record)
    stored = unhashed.pop("journal_payload_sha256", None)
    if (
        record.get("schema_version") != 1
        or record.get("kind")
        != "minilm_5ep_sft_short_slug_recovery_journal_record"
        or record.get("campaign") != "minilm_5ep_sft_hparam_search_v1"
        or record.get("transaction_id") != BAD_LOCK_PAYLOAD_SHA256[:16]
        or record.get("preflight_payload_sha256") != preflight_sha
        or record.get("phase") != phase
        or not isinstance(record.get("details"), Mapping)
        or stored != support.canonical_sha256(unhashed)
        or _read_single_link_text(path, label=f"recovery {phase} journal")
        != support.canonical_json_dumps(record) + "\n"
    ):
        raise RecoveryError(f"Recovery {phase} journal identity/hash differs")
    return record


def _load_move_record(
    path: Path, *, item: Mapping[str, Any], preflight_sha: str
) -> dict[str, Any]:
    record = _load_single_link_json(path, label="recovery move journal")
    expected = _move_record(item, preflight_sha)
    if (
        record != expected
        or _read_single_link_text(path, label="recovery move journal")
        != support.canonical_json_dumps(expected) + "\n"
    ):
        raise RecoveryError("Recovery move journal identity/hash differs")
    return record


def _relocated_old_authority() -> tuple[dict[str, Any], dict[str, Any]]:
    original = _load_single_link_json(
        RECOVERY_FORENSIC_MANIFEST,
        label="forensic original trusted-provenance manifest",
    )
    if _single_link_file_sha256(
        RECOVERY_FORENSIC_MANIFEST,
        label="forensic original trusted-provenance manifest",
    ) != (
        BAD_PROVENANCE_MANIFEST_FILE_SHA256
    ):
        raise RecoveryError("Forensic original provenance bytes differ")
    original_archive = Path(str(original.get("archive_dir"))).resolve(strict=False)
    expected_original_archive = REJECTED_ORIGINAL_PROVENANCE_ARCHIVE.resolve(
        strict=False
    )
    if original_archive != expected_original_archive:
        raise RecoveryError("Forensic manifest did not preserve its original active path")
    documents = original.get("source_documents")
    if not isinstance(documents, list) or len(documents) != 2:
        raise RecoveryError("Forensic provenance document ledger differs")
    relocated = deepcopy(original)
    relocated["archive_dir"] = str(RECOVERY_RELOCATED_ARCHIVE.resolve(strict=True))
    relocated_documents: list[dict[str, Any]] = []
    for source in documents:
        if not isinstance(source, Mapping):
            raise RecoveryError("Forensic provenance source differs")
        original_snapshot = Path(str(source.get("snapshot_path"))).resolve(strict=False)
        document_sha = str(source.get("document_sha256"))
        if (
            original_snapshot.parent != expected_original_archive / "sources"
            or original_snapshot.name != f"{document_sha}.json"
        ):
            raise RecoveryError("Forensic source path is not the exact original path")
        relocated_snapshot = (
            RECOVERY_RELOCATED_ARCHIVE / "sources" / f"{document_sha}.json"
        ).resolve(strict=True)
        if _single_link_file_sha256(
            relocated_snapshot, label="relocated old source snapshot"
        ) != source.get(
            "snapshot_file_sha256"
        ):
            raise RecoveryError("Relocated old source snapshot hash differs")
        relocated_source = dict(source)
        relocated_source["snapshot_path"] = str(relocated_snapshot)
        relocated_documents.append(relocated_source)
    relocated["source_documents"] = relocated_documents
    relocated.pop("trusted_provenance_payload_sha256", None)
    relocated["trusted_provenance_payload_sha256"] = support.canonical_sha256(
        relocated
    )
    plan = support.adaptive.load_plan(support.DEFAULT_PLAN)
    stored_relocated = _load_single_link_json(
        RECOVERY_RELOCATED_PROVENANCE,
        label="relocated old trusted-provenance manifest",
    )
    if stored_relocated != relocated:
        raise RecoveryError("Relocated old trusted provenance differs")
    with support.patched_loss_predecessor():
        short_identity = support.adaptive._variant_identity
        support.adaptive._variant_identity = support._ORIGINAL_VARIANT_IDENTITY
        try:
            loaded = support.adaptive.load_trusted_provenance(
                RECOVERY_RELOCATED_PROVENANCE, plan=plan
            )
            if loaded != relocated:
                raise RecoveryError("Relocated old trusted provenance differs")
            old_lock = _load_bad_lock_at(RECOVERY_RELOCATED_LOCK)
            core_loaded = support.adaptive.read_lock(
                RECOVERY_RELOCATED_LOCK,
                plan=plan,
                trusted_provenance=loaded,
            )
        finally:
            support.adaptive._variant_identity = short_identity
    if core_loaded != old_lock:
        raise RecoveryError("Core could not reload the relocated old authority")
    active_manifest = Path(str(BAD_LOCK) + ".trusted-provenance.json").resolve(
        strict=False
    )
    active_archive = Path(str(BAD_LOCK) + ".trusted-provenance").resolve(
        strict=False
    )
    if (
        RECOVERY_RELOCATED_PROVENANCE.resolve(strict=True) == active_manifest
        or Path(loaded["archive_dir"]).resolve(strict=True)
        != RECOVERY_RELOCATED_ARCHIVE.resolve(strict=True)
        or any(
            Path(row["snapshot_path"]).resolve(strict=True).parent
            != RECOVERY_RELOCATED_ARCHIVE.resolve(strict=True) / "sources"
            for row in loaded["source_documents"]
        )
        or any(
            Path(row["snapshot_path"]).resolve(strict=True) == active_archive
            or active_archive in Path(row["snapshot_path"]).resolve(
                strict=True
            ).parents
            for row in loaded["source_documents"]
        )
        or any(
            Path(row["lock_path"]).resolve(strict=True)
            in {BAD_LOCK.resolve(strict=False), active_manifest}
            for row in loaded["prerequisite_locks"]
        )
    ):
        raise RecoveryError("Relocated old authority still depends on active sidecars")
    evidence = {
        "authority_status": "relocated_core_validated_old_lock_authority",
        "lock_path": _relative(RECOVERY_RELOCATED_LOCK),
        "lock_file_sha256": _single_link_file_sha256(
            RECOVERY_RELOCATED_LOCK, label="relocated old lock"
        ),
        "lock_payload_sha256": old_lock["lock_payload_sha256"],
        "forensic_original_manifest": {
            "path": _relative(RECOVERY_FORENSIC_MANIFEST),
            "file_sha256": _single_link_file_sha256(
                RECOVERY_FORENSIC_MANIFEST,
                label="forensic original trusted-provenance manifest",
            ),
            "status": "original_bytes_only_not_a_reloadable_authority",
            "original_archive_dir": str(expected_original_archive),
        },
        "relocated_manifest": {
            "path": _relative(RECOVERY_RELOCATED_PROVENANCE),
            "file_sha256": _single_link_file_sha256(
                RECOVERY_RELOCATED_PROVENANCE,
                label="relocated old trusted-provenance manifest",
            ),
            "payload_sha256": loaded["trusted_provenance_payload_sha256"],
            "archive_dir": _relative(RECOVERY_RELOCATED_ARCHIVE),
        },
        "core_validation": (
            "frozen_core_read_lock_under_reviewed_fast_predecessor_and_original_"
            "long_identity_passed"
        ),
        "active_sidecar_dependency": False,
    }
    return evidence, old_lock


def _ensure_relocated_old_authority(
    *, fault_hook: FaultHook | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    original = _load_single_link_json(
        RECOVERY_FORENSIC_MANIFEST,
        label="forensic original trusted-provenance manifest",
    )
    relocated = deepcopy(original)
    relocated["archive_dir"] = str(RECOVERY_RELOCATED_ARCHIVE.resolve(strict=True))
    documents = relocated.get("source_documents")
    if not isinstance(documents, list):
        raise RecoveryError("Forensic provenance sources differ")
    for source in documents:
        if not isinstance(source, dict):
            raise RecoveryError("Forensic provenance source differs")
        source["snapshot_path"] = str(
            (
                RECOVERY_RELOCATED_ARCHIVE
                / "sources"
                / f"{source.get('document_sha256')}.json"
            ).resolve(strict=True)
        )
    relocated.pop("trusted_provenance_payload_sha256", None)
    relocated["trusted_provenance_payload_sha256"] = support.canonical_sha256(
        relocated
    )
    _atomic_write_json_once(
        RECOVERY_RELOCATED_PROVENANCE,
        relocated,
        label="relocated_old_provenance_manifest",
        fault_hook=fault_hook,
    )
    return _relocated_old_authority()


def _replacement_evidence(new_lock: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = Path(str(BAD_LOCK) + ".trusted-provenance.json")
    archive_dir = Path(str(BAD_LOCK) + ".trusted-provenance")
    plan = support.adaptive.load_plan(support.DEFAULT_PLAN)
    stored_trusted = _load_single_link_json(
        manifest_path, label="replacement trusted-provenance manifest"
    )
    _single_link_file_sha256(BAD_LOCK, label="replacement primary lock")
    _tree_hashes(archive_dir)
    with support.patched_loss_predecessor():
        trusted = support.adaptive.load_trusted_provenance(manifest_path, plan=plan)
        reloaded = support.adaptive.read_lock(
            BAD_LOCK, plan=plan, trusted_provenance=trusted
        )
    if reloaded != dict(new_lock) or trusted != stored_trusted:
        raise RecoveryError("Replacement lock differs from active provenance")
    expected_snapshots = {
        Path(row["snapshot_path"]).resolve(strict=True)
        for row in trusted["source_documents"]
    }
    actual_files = {path.resolve(strict=True) for path in archive_dir.rglob("*") if path.is_file()}
    if (
        Path(trusted["archive_dir"]).resolve(strict=True)
        != archive_dir.resolve(strict=True)
        or actual_files != expected_snapshots
    ):
        raise RecoveryError("Replacement provenance archive file set differs")
    return {
        "lock_path": _relative(BAD_LOCK),
        "lock_file_sha256": _single_link_file_sha256(
            BAD_LOCK, label="replacement primary lock"
        ),
        "lock_payload_sha256": new_lock["lock_payload_sha256"],
        "trusted_provenance_manifest_path": _relative(manifest_path),
        "trusted_provenance_manifest_file_sha256": _single_link_file_sha256(
            manifest_path, label="replacement trusted-provenance manifest"
        ),
        "trusted_provenance_payload_sha256": trusted[
            "trusted_provenance_payload_sha256"
        ],
        "trusted_provenance_archive_path": _relative(archive_dir),
        "trusted_provenance_tree_file_sha256s": _tree_hashes(archive_dir),
    }


def _validate_active_materialization_paths_before_core() -> None:
    manifest_path = Path(str(BAD_LOCK) + ".trusted-provenance.json")
    archive_dir = Path(str(BAD_LOCK) + ".trusted-provenance")
    for path, label in (
        (BAD_LOCK, "replacement primary lock"),
        (manifest_path, "replacement trusted-provenance manifest"),
    ):
        if path.exists() or path.is_symlink():
            _read_single_link_bytes(path, label=label)
        pending = _pending_path(path)
        if pending.exists() or pending.is_symlink():
            _read_single_link_bytes(pending, label=f"{label} pending file")
    if archive_dir.exists() or archive_dir.is_symlink():
        _tree_hashes(archive_dir)


def _journal_file_hashes(preflight: Mapping[str, Any]) -> dict[str, str]:
    phase_names = (
        "initialization.complete",
        "archive.complete",
        "relocated_old_authority.complete",
        "materialization.intent",
        "materialization.complete",
    )
    paths = [_phase_path(name) for name in phase_names]
    paths.extend(_journal_path_for_move(item["index"]) for item in preflight["moves"])
    expected = {
        path.name: _single_link_file_sha256(path, label="recovery journal file")
        for path in paths
    }
    actual = {path.name for path in RECOVERY_JOURNAL.iterdir() if path.is_file()}
    if actual != set(expected):
        raise RecoveryError("Recovery journal file set differs")
    if any(path.is_symlink() or not path.is_file() for path in RECOVERY_JOURNAL.iterdir()):
        raise RecoveryError("Recovery journal contains a non-file entry")
    return dict(sorted(expected.items()))


def _validate_transaction_journal(preflight: Mapping[str, Any]) -> dict[str, str]:
    preflight_sha = preflight["preflight_payload_sha256"]
    initialization = _load_phase(
        _phase_path("initialization.complete"),
        phase="initialization.complete",
        preflight_sha=preflight_sha,
    )
    if initialization["details"] != {
        "preflight_file_sha256": _single_link_file_sha256(
            RECOVERY_PREFLIGHT, label="recovery preflight"
        ),
        "audit_copy_path": _relative(RECOVERY_AUDIT_COPY),
        "audit_copy_file_sha256": AUDIT_RECEIPT_FILE_SHA256,
    }:
        raise RecoveryError("Recovery initialization journal differs")
    move_hashes: dict[str, str] = {}
    for item in preflight["moves"]:
        path = _journal_path_for_move(item["index"])
        _load_move_record(path, item=item, preflight_sha=preflight_sha)
        target = _repo_path(item["target"], label="moved target", must_exist=False)
        if not _artifact_matches(target, item):
            raise RecoveryError("Recovery move terminal state differs")
        move_hashes[path.name] = _single_link_file_sha256(
            path, label="recovery move journal"
        )
    archive = _load_phase(
        _phase_path("archive.complete"),
        phase="archive.complete",
        preflight_sha=preflight_sha,
    )
    if archive["details"] != {
        "move_count": len(preflight["moves"]),
        "move_journal_file_sha256": dict(sorted(move_hashes.items())),
        "terminal_state": "all_sources_absent_all_targets_exact",
    }:
        raise RecoveryError("Recovery archive-complete journal differs")
    relocated, _ = _relocated_old_authority()
    relocated_phase = _load_phase(
        _phase_path("relocated_old_authority.complete"),
        phase="relocated_old_authority.complete",
        preflight_sha=preflight_sha,
    )
    if relocated_phase["details"] != relocated:
        raise RecoveryError("Relocated old authority journal differs")
    intent = _load_phase(
        _phase_path("materialization.intent"),
        phase="materialization.intent",
        preflight_sha=preflight_sha,
    )
    if intent["details"] != {
        "mode": "loss_primary",
        "output_path": _relative(BAD_LOCK),
        "expected_old_lock_payload_sha256": BAD_LOCK_PAYLOAD_SHA256,
        "write_protocol": "resumable_pending_then_atomic_noreplace",
        "kaggle_mutations": 0,
    }:
        raise RecoveryError("Materialization intent journal differs")
    return _journal_file_hashes(preflight)


def apply_recovery(
    preflight: Mapping[str, Any], *, fault_hook: FaultHook | None = None
) -> dict[str, Any]:
    if RECOVERY_COMPLETION.exists():
        return validate_completed_recovery()
    validated = _validate_preflight_document(preflight)
    if RECOVERY_ARCHIVE.is_symlink():
        raise RecoveryError("Recovery archive must not be a symlink")
    RECOVERY_ARCHIVE.mkdir(parents=True, exist_ok=True)
    if not RECOVERY_ARCHIVE.is_dir():
        raise RecoveryError("Recovery archive is not a directory")
    _fault(fault_hook, "after_archive_mkdir")
    _atomic_write_json_once(
        RECOVERY_PREFLIGHT,
        validated,
        label="recovery_preflight",
        fault_hook=fault_hook,
    )
    _fault(fault_hook, "after_preflight")
    _atomic_write_bytes_once(
        RECOVERY_AUDIT_COPY,
        _read_single_link_bytes(AUDIT_RECEIPT, label="recovery audit receipt"),
        mode=0o444,
        label="audit_copy",
        fault_hook=fault_hook,
    )
    if _single_link_file_sha256(
        RECOVERY_AUDIT_COPY, label="archived recovery audit receipt"
    ) != AUDIT_RECEIPT_FILE_SHA256:
        raise RecoveryError("Archived audit receipt copy differs")
    _fault(fault_hook, "after_audit_copy")
    preflight_sha = validated["preflight_payload_sha256"]
    _write_phase_once(
        phase="initialization.complete",
        preflight_sha=preflight_sha,
        details={
            "preflight_file_sha256": _single_link_file_sha256(
                RECOVERY_PREFLIGHT, label="recovery preflight"
            ),
            "audit_copy_path": _relative(RECOVERY_AUDIT_COPY),
            "audit_copy_file_sha256": _single_link_file_sha256(
                RECOVERY_AUDIT_COPY, label="archived recovery audit receipt"
            ),
        },
        fault_hook=fault_hook,
    )
    _move_preflight_sources(validated, fault_hook=fault_hook)
    move_hashes = {
        _journal_path_for_move(item["index"]).name: _single_link_file_sha256(
            _journal_path_for_move(item["index"]),
            label="recovery move journal",
        )
        for item in validated["moves"]
    }
    _write_phase_once(
        phase="archive.complete",
        preflight_sha=preflight_sha,
        details={
            "move_count": len(validated["moves"]),
            "move_journal_file_sha256": dict(sorted(move_hashes.items())),
            "terminal_state": "all_sources_absent_all_targets_exact",
        },
        fault_hook=fault_hook,
    )
    _fault(fault_hook, "after_archive_complete")
    relocated_evidence, old_lock = _ensure_relocated_old_authority(
        fault_hook=fault_hook
    )
    _write_phase_once(
        phase="relocated_old_authority.complete",
        preflight_sha=preflight_sha,
        details=relocated_evidence,
        fault_hook=fault_hook,
    )
    intent_details = {
        "mode": "loss_primary",
        "output_path": _relative(BAD_LOCK),
        "expected_old_lock_payload_sha256": BAD_LOCK_PAYLOAD_SHA256,
        "write_protocol": "resumable_pending_then_atomic_noreplace",
        "kaggle_mutations": 0,
    }
    _write_phase_once(
        phase="materialization.intent",
        preflight_sha=preflight_sha,
        details=intent_details,
        fault_hook=fault_hook,
    )
    _fault(fault_hook, "before_rematerialize")
    _validate_active_materialization_paths_before_core()
    if BAD_LOCK.exists() or BAD_LOCK.is_symlink():
        new_lock = support.load_scoped_loss_lock(
            plan_path=support.DEFAULT_PLAN, stage_lock_path=BAD_LOCK
        )
    else:
        new_lock = _materialize_replacement(fault_hook=fault_hook)
    variants = _validate_replacement(new_lock, old_lock)
    replacement = _replacement_evidence(new_lock)
    _fault(fault_hook, "after_rematerialize")
    materialization_details = {
        "replacement": replacement,
        "variants": variants,
        "terminal_state": "short_primary_lock_and_provenance_exact",
    }
    _write_phase_once(
        phase="materialization.complete",
        preflight_sha=preflight_sha,
        details=materialization_details,
        fault_hook=fault_hook,
    )
    _fault(fault_hook, "after_materialization_journal")
    journal_hashes = _validate_transaction_journal(validated)
    materialization_phase = _load_phase(
        _phase_path("materialization.complete"),
        phase="materialization.complete",
        preflight_sha=preflight_sha,
    )
    if materialization_phase["details"] != materialization_details:
        raise RecoveryError("Materialization completion journal differs")
    freeze = support.load_freeze_manifest()
    completion: dict[str, Any] = {
        "schema_version": 2,
        "kind": "minilm_5ep_sft_short_slug_recovery_completion",
        "campaign": "minilm_5ep_sft_hparam_search_v1",
        "transaction_id": BAD_LOCK_PAYLOAD_SHA256[:16],
        "preflight_payload_sha256": preflight_sha,
        "audit_receipt_payload_sha256": AUDIT_RECEIPT_PAYLOAD_SHA256,
        "old_lock_file_sha256": BAD_LOCK_FILE_SHA256,
        "old_lock_payload_sha256": BAD_LOCK_PAYLOAD_SHA256,
        "archived_old_authority": relocated_evidence,
        "new_lock_path": _relative(BAD_LOCK),
        "new_lock_file_sha256": replacement["lock_file_sha256"],
        "new_lock_payload_sha256": new_lock["lock_payload_sha256"],
        "new_lock_provenance": replacement,
        "freeze_manifest_payload_sha256": freeze["manifest_payload_sha256"],
        "journal_directory": _relative(RECOVERY_JOURNAL),
        "journal_record_file_sha256": journal_hashes,
        "variants": variants,
        "kaggle_mutations": 0,
        "next_action": "fast-track generator and dry-run preflight before explicit submit",
    }
    completion["completion_payload_sha256"] = support.canonical_sha256(completion)
    _fault(fault_hook, "before_completion")
    _atomic_write_json_once(
        RECOVERY_COMPLETION,
        completion,
        label="recovery_completion",
        fault_hook=fault_hook,
    )
    _fault(fault_hook, "after_completion")
    return validate_completed_recovery()


def validate_completed_recovery() -> dict[str, Any]:
    completion = _load_single_link_json(
        RECOVERY_COMPLETION, label="short-slug recovery completion"
    )
    _exact_keys(
        completion,
        {
            "schema_version", "kind", "campaign", "transaction_id",
            "preflight_payload_sha256", "audit_receipt_payload_sha256",
            "old_lock_file_sha256", "old_lock_payload_sha256",
            "archived_old_authority", "new_lock_path", "new_lock_file_sha256",
            "new_lock_payload_sha256", "new_lock_provenance",
            "freeze_manifest_payload_sha256", "journal_directory",
            "journal_record_file_sha256", "variants", "kaggle_mutations",
            "next_action", "completion_payload_sha256",
        },
        label="short-slug recovery completion",
    )
    unhashed = dict(completion)
    stored = unhashed.pop("completion_payload_sha256", None)
    freeze = support.load_freeze_manifest()
    completion_freeze = completion.get("freeze_manifest_payload_sha256")
    current_freeze = freeze["manifest_payload_sha256"]
    freeze_anchor_is_current = completion_freeze == current_freeze
    freeze_anchor_is_exact_applied_generation = (
        completion_freeze
        == APPLIED_RECOVERY_FREEZE_MANIFEST_PAYLOAD_SHA256
        and stored == APPLIED_RECOVERY_COMPLETION_PAYLOAD_SHA256
    )
    if (
        completion.get("schema_version") != 2
        or completion.get("kind")
        != "minilm_5ep_sft_short_slug_recovery_completion"
        or completion.get("campaign") != "minilm_5ep_sft_hparam_search_v1"
        or completion.get("transaction_id") != BAD_LOCK_PAYLOAD_SHA256[:16]
        or stored != support.canonical_sha256(unhashed)
        or completion.get("audit_receipt_payload_sha256")
        != AUDIT_RECEIPT_PAYLOAD_SHA256
        or completion.get("old_lock_file_sha256") != BAD_LOCK_FILE_SHA256
        or completion.get("old_lock_payload_sha256") != BAD_LOCK_PAYLOAD_SHA256
        or not (
            freeze_anchor_is_current
            or freeze_anchor_is_exact_applied_generation
        )
        or completion.get("journal_directory") != _relative(RECOVERY_JOURNAL)
        or completion.get("kaggle_mutations") != 0
        or _read_single_link_text(
            RECOVERY_COMPLETION, label="short-slug recovery completion"
        )
        != support.canonical_json_dumps(completion) + "\n"
    ):
        raise RecoveryError("Short-slug recovery completion identity/hash differs")
    preflight = _validate_preflight_document(
        _load_single_link_json(
            RECOVERY_PREFLIGHT, label="archived short-slug recovery preflight"
        )
    )
    if preflight["preflight_payload_sha256"] != completion[
        "preflight_payload_sha256"
    ]:
        raise RecoveryError("Archived recovery preflight hash differs")
    if _single_link_file_sha256(
        RECOVERY_AUDIT_COPY, label="archived recovery audit receipt"
    ) != AUDIT_RECEIPT_FILE_SHA256:
        raise RecoveryError("Archived audit receipt differs")
    journal_hashes = _validate_transaction_journal(preflight)
    if completion.get("journal_record_file_sha256") != journal_hashes:
        raise RecoveryError("Recovery completion journal binding differs")
    relocated, old_lock = _relocated_old_authority()
    if completion.get("archived_old_authority") != relocated:
        raise RecoveryError("Archived old authority completion binding differs")
    new_lock_path = _repo_path(
        completion.get("new_lock_path"),
        label="replacement primary lock",
        must_exist=True,
    )
    _validate_active_materialization_paths_before_core()
    new_lock = support.load_scoped_loss_lock(
        plan_path=support.DEFAULT_PLAN,
        stage_lock_path=new_lock_path,
    )
    replacement = _replacement_evidence(new_lock)
    if (
        new_lock_path != BAD_LOCK.resolve(strict=True)
        or _single_link_file_sha256(
            new_lock_path, label="replacement primary lock"
        )
        != completion.get("new_lock_file_sha256")
        or new_lock.get("lock_payload_sha256")
        != completion.get("new_lock_payload_sha256")
        or completion.get("new_lock_provenance") != replacement
    ):
        raise RecoveryError("Replacement primary lock completion binding differs")
    expected_variants = _validate_replacement(new_lock, old_lock)
    if completion.get("variants") != expected_variants:
        raise RecoveryError("Replacement primary variants differ from completion")
    materialization = _load_phase(
        _phase_path("materialization.complete"),
        phase="materialization.complete",
        preflight_sha=preflight["preflight_payload_sha256"],
    )
    if materialization["details"] != {
        "replacement": replacement,
        "variants": expected_variants,
        "terminal_state": "short_primary_lock_and_provenance_exact",
    }:
        raise RecoveryError("Replacement materialization journal differs")
    support.validate_receipt()
    return completion


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--audit-receipt", type=Path, default=AUDIT_RECEIPT)
    return parser.parse_args(argv)


def _load_or_build_preflight() -> dict[str, Any]:
    if RECOVERY_PREFLIGHT.exists() or RECOVERY_PREFLIGHT.is_symlink():
        if RECOVERY_PREFLIGHT.is_symlink() or not RECOVERY_PREFLIGHT.is_file():
            raise RecoveryError("Stored recovery preflight is not an ordinary file")
        payload = _validate_preflight_document(
            _load_single_link_json(
                RECOVERY_PREFLIGHT, label="stored recovery preflight"
            )
        )
        if _read_single_link_text(
            RECOVERY_PREFLIGHT, label="stored recovery preflight"
        ) != (
            support.canonical_json_dumps(payload) + "\n"
        ):
            raise RecoveryError("Stored recovery preflight is not canonical")
        return payload
    return _validate_preflight_document(build_preflight())


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.audit_receipt.resolve(strict=True) != AUDIT_RECEIPT.resolve(strict=True):
        raise RecoveryError("Only the reviewed read-only audit receipt is accepted")
    if args.preflight:
        if RECOVERY_COMPLETION.exists():
            completion = validate_completed_recovery()
            print(
                support.canonical_json_dumps(
                    {"status": "already_recovered", **completion}
                )
            )
            return 0
        preflight = _load_or_build_preflight()
        print(support.canonical_json_dumps(preflight))
        return 0
    with _exclusive_recovery_lock():
        if RECOVERY_COMPLETION.exists():
            completion = validate_completed_recovery()
            print(
                support.canonical_json_dumps(
                    {"status": "already_recovered", **completion}
                )
            )
            return 0
        preflight = _load_or_build_preflight()
        completion = apply_recovery(preflight)
        print(support.canonical_json_dumps(completion))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
