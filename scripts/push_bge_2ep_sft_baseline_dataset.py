#!/usr/bin/env python3
"""Build or upload the private slim BGE SFT significance baseline Dataset.

The safe default is a local dry-run.  A remote Dataset mutation requires the
explicit ``--upload`` flag.  Only the exact baseline completion receipt and
compact IID/hard predictions are copied; OOD predictions and model weights are
forbidden because the former OOD split is part of supervised training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import create_bge_2ep_sft_notebooks as builder
import push_kaggle_training_dataset as shared_push
import run_bge_2ep_sft_kaggle as campaign_launcher
import run_kaggle_notebook as kaggle


DATASET_SLUG = "product-matching-bge-2ep-sft-baseline-v1"
STAGE_DIR = ROOT / ".kaggle" / "datasets" / DATASET_SLUG
MANIFEST_FILENAME = "bge_2ep_sft_baseline_manifest.json"
COMPLETION_FILENAME = "notebook_completed.json"
SPLITS = ("iid", "hard")
PREDICTION_FILENAMES = {
    split: f"{split}_validation_predictions.parquet" for split in SPLITS
}
COMPACT_COLUMNS = ("id1", "id2", "target", "category_1", "score")
EXPECTED_ROWS = {"iid": builder.EXPECTED_IID, "hard": builder.EXPECTED_HARD}
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
SENSITIVE_KEYS = {
    "api_token",
    "kaggle_api_token",
    "private_key",
    "private_key_id",
    "service_account_json",
    "credentials",
    "credential_json",
}


class BaselineDatasetError(ValueError):
    """Raised when a baseline cannot be frozen without ambiguity."""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BaselineDatasetError(f"Could not read JSON object {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise BaselineDatasetError(f"Expected a JSON object in {path.name}")
    return value


def _assert_no_sensitive_values(value: object, *, location: str = "completion") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).strip().lower()
            if key in SENSITIVE_KEYS:
                raise BaselineDatasetError(
                    f"Refusing to stage sensitive field {location}.{raw_key}"
                )
            _assert_no_sensitive_values(item, location=f"{location}.{raw_key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_sensitive_values(item, location=f"{location}[{index}]")


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or HASH_PATTERN.fullmatch(value) is None:
        raise BaselineDatasetError(f"{label} is not an exact lowercase SHA-256")
    return value


def resolve_isolated_paths(source_dir: Path, stage_dir: Path) -> tuple[Path, Path]:
    """Reject source/stage aliases and containment before any stage mutation."""
    if source_dir.is_symlink():
        raise BaselineDatasetError(f"Baseline source must not be a symlink: {source_dir}")
    if stage_dir.is_symlink():
        raise BaselineDatasetError(f"Baseline stage must not be a symlink: {stage_dir}")
    source = source_dir.expanduser().resolve(strict=True)
    stage = stage_dir.expanduser().resolve(strict=False)
    if not source.is_dir():
        raise BaselineDatasetError(f"Baseline source is not a directory: {source}")
    if stage.exists() and not stage.is_dir():
        raise BaselineDatasetError(f"Baseline stage is not a directory: {stage}")
    if source == stage or source in stage.parents or stage in source.parents:
        raise BaselineDatasetError(
            "Baseline source and stage must be disjoint after path resolution"
        )
    return source, stage


def expected_baseline_entry(owner: str) -> dict[str, Any]:
    """Materialize the current exact baseline identity without writing notebooks."""
    entries = builder.build_campaign(
        owner=owner,
        only={campaign_launcher.BASELINE_KEY},
        write=False,
    )
    if len(entries) != 1 or entries[0].get("role") != "baseline":
        raise BaselineDatasetError("Could not resolve one exact BGE baseline entry")
    return entries[0]


def default_source_dir(entry: Mapping[str, Any]) -> Path:
    return campaign_launcher.output_root() / str(entry["kernel_slug"])


def _validate_completion_binding(
    completion: Mapping[str, Any],
    *,
    entry: Mapping[str, Any],
) -> dict[str, str]:
    exact = {
        "status": "complete",
        "experiment": entry["experiment"],
        "experiment_group": "sft",
        "campaign": builder.CAMPAIGN,
        "role": "baseline",
        "model": entry["checkpoint_dataset"],
        "dataset_ref": entry["validation_dataset"],
        "validation_manifest_sha256": entry["validation_manifest_sha256"],
        "initial_checkpoint_ref": entry["checkpoint_dataset"],
        "initial_checkpoint_manifest_sha256": entry["checkpoint_manifest_sha256"],
        "initial_checkpoint_model_sha256": entry["checkpoint_model_sha256"],
        "code_bundle_sha256": entry["source_sha256"],
        "frozen_recipe_sha256": entry["recipe_sha256"],
        "campaign_identity_sha256": entry["identity_sha256"],
        "executable_cells_sha256": entry["executable_cells_sha256"],
        "loss_hook_sha256": entry["loss_hook_sha256"],
        "ood_evaluation_policy": "disabled_train_contaminated",
    }
    for key, expected in exact.items():
        if completion.get(key) != expected:
            raise BaselineDatasetError(f"Baseline completion identity differs at {key}")
    run_id = str(completion.get("run_id", "")).strip()
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise BaselineDatasetError("Baseline completion run_id is not a 32-hex UUID")
    for key in (
        "validation_manifest_sha256",
        "initial_checkpoint_manifest_sha256",
        "initial_checkpoint_model_sha256",
        "code_bundle_sha256",
        "frozen_recipe_sha256",
        "campaign_identity_sha256",
        "executable_cells_sha256",
        "loss_hook_sha256",
    ):
        _require_hash(completion.get(key), f"completion.{key}")
    if completion.get("baseline_comparison") not in (None, {}):
        raise BaselineDatasetError("Baseline completion must not contain a comparison")
    report = completion.get("training_report")
    if not isinstance(report, Mapping):
        raise BaselineDatasetError("Baseline completion has no training_report")
    validation = report.get("validation_splits")
    if not isinstance(validation, Mapping) or set(validation) != {"iid", "hard", "ood"}:
        raise BaselineDatasetError("Baseline report does not contain exact IID/hard/OOD keys")
    if validation.get("ood") != builder.OOD_SENTINEL:
        raise BaselineDatasetError("Baseline report does not preserve exact OOD=-1 sentinel")
    if report.get("evaluated_validation_splits") != ["iid", "hard"]:
        raise BaselineDatasetError("Baseline report claims an unexpected validation split")
    _assert_no_sensitive_values(completion)
    return {
        "baseline_run_id": run_id,
        "baseline_experiment": str(completion["experiment"]),
        "campaign": str(completion["campaign"]),
        "campaign_identity_sha256": str(completion["campaign_identity_sha256"]),
        "source_sha256": str(completion["code_bundle_sha256"]),
        "recipe_sha256": str(completion["frozen_recipe_sha256"]),
        "executable_cells_sha256": str(completion["executable_cells_sha256"]),
        "loss_hook_sha256": str(completion["loss_hook_sha256"]),
        "checkpoint_manifest_sha256": str(
            completion["initial_checkpoint_manifest_sha256"]
        ),
        "checkpoint_model_sha256": str(completion["initial_checkpoint_model_sha256"]),
        "validation_manifest_sha256": str(completion["validation_manifest_sha256"]),
    }


def validate_baseline_source(
    source_dir: Path,
    *,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    """Reuse the campaign's full validator before freezing a slim baseline."""
    campaign_launcher.validate_run_output(source_dir, entry=entry)
    completion = _read_json_object(source_dir / COMPLETION_FILENAME)
    _validate_completion_binding(completion, entry=entry)
    if list(source_dir.rglob("ood_validation_predictions.parquet")):
        raise BaselineDatasetError("Source contains forbidden OOD predictions")
    return completion


def _regular_file(path: Path, label: str) -> None:
    try:
        mode = os.lstat(path).st_mode
    except OSError as error:
        raise BaselineDatasetError(f"Could not inspect {label}: {error}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise BaselineDatasetError(f"{label} must be a non-symlink regular file")


def _find_exactly_one(source_dir: Path, filename: str) -> Path:
    paths = list(source_dir.rglob(filename))
    if len(paths) != 1:
        raise BaselineDatasetError(
            f"Expected exactly one {filename} below {source_dir}, got {paths}"
        )
    _regular_file(paths[0], filename)
    return paths[0]


def _copy_exact_file(source: Path, destination: Path) -> dict[str, object]:
    _regular_file(source, source.name)
    source_before = (source.stat().st_size, sha256_file(source))
    temporary = destination.with_name(f".{destination.name}.{time.time_ns()}.tmp")
    try:
        with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=8 * 1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        source_after = (source.stat().st_size, sha256_file(source))
        if source_after != source_before:
            raise BaselineDatasetError(f"Source changed while copying {source.name}")
        staged = (temporary.stat().st_size, sha256_file(temporary))
        if staged != source_before:
            raise BaselineDatasetError(f"Staged exact copy differs for {source.name}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"bytes": source_before[0], "sha256": source_before[1]}


def _prediction_schema(path: Path) -> set[str]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise BaselineDatasetError("pyarrow is required to inspect predictions") from error
    return set(parquet.ParquetFile(path).schema.names)


def _write_compact_predictions(
    source: Path,
    destination: Path,
    *,
    split: str,
) -> dict[str, object]:
    _regular_file(source, f"{split} predictions")
    available = _prediction_schema(source)
    missing = set(COMPACT_COLUMNS) - available
    if missing:
        raise BaselineDatasetError(
            f"{split} predictions are missing compact columns: {sorted(missing)}"
        )
    source_before = {"bytes": source.stat().st_size, "sha256": sha256_file(source)}
    frame = pd.read_parquet(source, columns=list(COMPACT_COLUMNS))
    source_after = {"bytes": source.stat().st_size, "sha256": sha256_file(source)}
    if source_after != source_before:
        raise BaselineDatasetError(f"{split} predictions changed while being read")
    if len(frame) != EXPECTED_ROWS[split]:
        raise BaselineDatasetError(
            f"{split} predictions have {len(frame)} rows, expected {EXPECTED_ROWS[split]}"
        )
    if frame.isna().any().any():
        raise BaselineDatasetError(f"{split} compact predictions contain nulls")
    target = pd.to_numeric(frame["target"], errors="raise").to_numpy(dtype=np.float64)
    score = pd.to_numeric(frame["score"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(target).all() or not set(np.unique(target)) <= {0.0, 1.0}:
        raise BaselineDatasetError(f"{split} prediction targets are not finite binary values")
    if not np.isfinite(score).all():
        raise BaselineDatasetError(f"{split} prediction scores are not finite")
    if (frame["category_1"].astype(str).str.len() == 0).any():
        raise BaselineDatasetError(f"{split} predictions contain empty categories")
    unordered_pairs = [
        (left, right) if left <= right else (right, left)
        for left, right in zip(
            frame["id1"].astype(str),
            frame["id2"].astype(str),
            strict=True,
        )
    ]
    unordered = pd.DataFrame(unordered_pairs, columns=["left", "right"])
    if unordered.duplicated().any():
        raise BaselineDatasetError(f"{split} predictions contain duplicate unordered pairs")

    temporary = destination.with_name(f".{destination.name}.{time.time_ns()}.tmp")
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        staged = pd.read_parquet(temporary, columns=list(COMPACT_COLUMNS))
        if not staged.equals(frame):
            raise BaselineDatasetError(f"{split} staged compact predictions differ")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "split": split,
        "rows": len(frame),
        "columns": list(COMPACT_COLUMNS),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "source_bytes": source_before["bytes"],
        "source_sha256": source_before["sha256"],
    }


def _write_json(path: Path, value: object, *, sort_keys: bool = False) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=sort_keys,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_stage_file_set(stage_dir: Path) -> None:
    expected = {
        COMPLETION_FILENAME,
        MANIFEST_FILENAME,
        "dataset-metadata.json",
        *PREDICTION_FILENAMES.values(),
    }
    actual = {path.name for path in stage_dir.iterdir()}
    if actual != expected:
        raise BaselineDatasetError(
            "Staged baseline file set differs: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    if any("ood" in name.lower() for name in actual):
        raise BaselineDatasetError("Staged baseline contains an OOD-named artifact")
    for name in actual:
        _regular_file(stage_dir / name, f"staged {name}")


def verify_payload_for_upload(
    stage_dir: Path,
    manifest: Mapping[str, Any],
) -> str:
    """Rehash all staged authority/payload files immediately before upload."""
    _validate_stage_file_set(stage_dir)
    disk_manifest = _read_json_object(stage_dir / MANIFEST_FILENAME)
    if disk_manifest != dict(manifest):
        raise BaselineDatasetError("Staged manifest differs from the verified manifest")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != {
        COMPLETION_FILENAME,
        *PREDICTION_FILENAMES.values(),
    }:
        raise BaselineDatasetError("Manifest file ledger differs from the slim contract")
    for filename, raw_declaration in files.items():
        if not isinstance(raw_declaration, Mapping):
            raise BaselineDatasetError(f"Invalid manifest declaration for {filename}")
        measured = {
            "bytes": (stage_dir / filename).stat().st_size,
            "sha256": sha256_file(stage_dir / filename),
        }
        declared = {
            "bytes": raw_declaration.get("bytes"),
            "sha256": raw_declaration.get("sha256"),
        }
        if measured != declared:
            raise BaselineDatasetError(f"Staged payload drifted for {filename}")
    completion = _read_json_object(stage_dir / COMPLETION_FILENAME)
    if manifest.get("completion_canonical_sha256") != _canonical_sha256(completion):
        raise BaselineDatasetError("Manifest canonical completion hash differs")
    binding = manifest.get("binding")
    if not isinstance(binding, Mapping):
        raise BaselineDatasetError("Manifest has no baseline binding")
    for manifest_key, completion_key in (
        ("baseline_run_id", "run_id"),
        ("baseline_experiment", "experiment"),
        ("campaign", "campaign"),
        ("campaign_identity_sha256", "campaign_identity_sha256"),
        ("source_sha256", "code_bundle_sha256"),
        ("recipe_sha256", "frozen_recipe_sha256"),
        ("executable_cells_sha256", "executable_cells_sha256"),
        ("loss_hook_sha256", "loss_hook_sha256"),
        ("checkpoint_manifest_sha256", "initial_checkpoint_manifest_sha256"),
        ("checkpoint_model_sha256", "initial_checkpoint_model_sha256"),
        ("validation_manifest_sha256", "validation_manifest_sha256"),
    ):
        if binding.get(manifest_key) != completion.get(completion_key):
            raise BaselineDatasetError(f"Manifest/completion binding differs at {manifest_key}")
    metadata = _read_json_object(stage_dir / "dataset-metadata.json")
    if metadata.get("isPrivate") is not True or metadata.get("id") != manifest.get(
        "dataset"
    ):
        raise BaselineDatasetError("Staged Dataset metadata is not exact and private")
    return sha256_file(stage_dir / MANIFEST_FILENAME)


def _promote_stage(temporary: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if destination.exists():
        backup = destination.with_name(f".{destination.name}.previous-{time.time_ns()}")
        destination.rename(backup)
    try:
        temporary.rename(destination)
    except Exception:
        if backup is not None and backup.exists() and not destination.exists():
            backup.rename(destination)
        raise


def build_payload(
    source_dir: Path,
    stage_dir: Path,
    owner: str,
    *,
    entry: Mapping[str, Any],
    dataset_slug: str = DATASET_SLUG,
    source_validator: Callable[..., dict[str, Any]] = validate_baseline_source,
) -> dict[str, Any]:
    source_dir, stage_dir = resolve_isolated_paths(source_dir, stage_dir)
    owner = kaggle.validate_slug(owner, "KAGGLE_USERNAME")
    dataset_slug = kaggle.validate_slug(dataset_slug, "dataset slug")
    if list(source_dir.rglob("ood_validation_predictions.parquet")):
        raise BaselineDatasetError("Source contains forbidden OOD predictions")
    if list(source_dir.rglob("model.safetensors*")):
        raise BaselineDatasetError("Slim baseline source contains forbidden model weights")
    completion_before = source_validator(source_dir, entry=entry)
    binding = _validate_completion_binding(completion_before, entry=entry)

    stage_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{stage_dir.name}.build-", dir=stage_dir.parent)
    )
    promoted = False
    try:
        completion_source = _find_exactly_one(source_dir, COMPLETION_FILENAME)
        prediction_sources = {
            split: _find_exactly_one(source_dir, filename)
            for split, filename in PREDICTION_FILENAMES.items()
        }
        completion_declaration = _copy_exact_file(
            completion_source,
            temporary / COMPLETION_FILENAME,
        )
        prediction_declarations: dict[str, dict[str, object]] = {}
        for split, filename in PREDICTION_FILENAMES.items():
            prediction_declarations[filename] = _write_compact_predictions(
                prediction_sources[split],
                temporary / filename,
                split=split,
            )

        # Re-run the exact campaign validator after every source read.  This
        # closes the validation/copy window before publishing the manifest.
        completion_after = source_validator(source_dir, entry=entry)
        if completion_after != completion_before:
            raise BaselineDatasetError("Baseline completion changed during staging")
        if sha256_file(completion_source) != completion_declaration["sha256"]:
            raise BaselineDatasetError("Baseline completion changed after staging")
        for split, filename in PREDICTION_FILENAMES.items():
            declaration = prediction_declarations[filename]
            if sha256_file(prediction_sources[split]) != declaration["source_sha256"]:
                raise BaselineDatasetError(f"Baseline source changed after staging: {filename}")

        dataset_ref = f"{owner}/{dataset_slug}"
        files: dict[str, dict[str, object]] = {
            COMPLETION_FILENAME: {
                **completion_declaration,
                "exact_source_copy": True,
            },
            **prediction_declarations,
        }
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "dataset": dataset_ref,
            "is_private": True,
            "purpose": "paired IID/hard comparison against the exact BGE SFT baseline",
            "binding": binding,
            "validation_protocol": "product-matching-validation-splits-v1",
            "evaluated_splits": list(SPLITS),
            "primary_split": "iid",
            "diagnostic_splits": ["hard"],
            "ood": {
                "evaluated": False,
                "metric_sentinel": -1.0,
                "comparison": None,
                "reason": "former frozen OOD pairs are part of BGE supervised training",
                "prediction_file": None,
            },
            "files": files,
            "completion_canonical_sha256": _canonical_sha256(completion_before),
        }
        _write_json(temporary / MANIFEST_FILENAME, manifest, sort_keys=True)
        metadata = {
            "title": "Product Matching BGE 2ep SFT Baseline v1",
            "id": dataset_ref,
            "licenses": [{"name": "unknown"}],
            "isPrivate": True,
            "description": (
                "Private slim exact BGE SFT baseline receipt and IID/hard "
                "predictions. Former OOD pairs were promoted to training; no OOD "
                "predictions or model weights are included."
            ),
        }
        _write_json(temporary / "dataset-metadata.json", metadata)
        verify_payload_for_upload(temporary, manifest)
        _promote_stage(temporary, stage_dir)
        promoted = True
        verify_payload_for_upload(stage_dir, manifest)
        return manifest
    finally:
        if not promoted and temporary.exists():
            shutil.rmtree(temporary)


def verify_remote_dataset(
    cli: list[str],
    dataset_ref: str,
    *,
    expected_manifest_sha256: str,
) -> None:
    expected_files = {
        MANIFEST_FILENAME,
        COMPLETION_FILENAME,
        *PREDICTION_FILENAMES.values(),
    }
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
    try:
        remote_payload = json.loads(remote_files.stdout)
    except json.JSONDecodeError as error:
        kaggle.fail(f"remote BGE baseline file listing is invalid JSON: {error}", 1)
    observed: set[str] = set()
    if isinstance(remote_payload, list):
        for row in remote_payload:
            if isinstance(row, Mapping):
                name = row.get("name") or row.get("ref")
                if isinstance(name, str):
                    observed.add(Path(name).name)
    elif isinstance(remote_payload, Mapping):
        rows = remote_payload.get("datasetFiles") or remote_payload.get("files") or []
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, Mapping) and isinstance(row.get("name"), str):
                    observed.add(Path(str(row["name"])).name)
    missing = sorted(expected_files - observed)
    if missing:
        kaggle.fail(f"remote BGE baseline Dataset is missing: {missing}", 1)
    unexpected = sorted(observed - expected_files)
    if unexpected:
        kaggle.fail(
            f"remote BGE baseline Dataset has unexpected files: {unexpected}", 1
        )
    with tempfile.TemporaryDirectory(prefix="kaggle-bge-sft-baseline-") as raw:
        destination = Path(raw)
        kaggle.run_command(
            cli
            + [
                "datasets",
                "download",
                dataset_ref,
                "-f",
                MANIFEST_FILENAME,
                "-p",
                str(destination),
                "-o",
                "-q",
            ]
        )
        if sha256_file(destination / MANIFEST_FILENAME) != expected_manifest_sha256:
            kaggle.fail("remote BGE baseline manifest differs", 1)
    with tempfile.TemporaryDirectory(prefix="kaggle-bge-sft-baseline-metadata-") as raw:
        destination = Path(raw)
        kaggle.run_command(
            cli + ["datasets", "metadata", dataset_ref, "--path", str(destination)]
        )
        metadata = _read_json_object(destination / "dataset-metadata.json")
        info = metadata.get("info", metadata)
        if not isinstance(info, Mapping) or info.get("isPrivate") is not True:
            kaggle.fail("Kaggle reports that the BGE baseline Dataset is public", 1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the private slim BGE-2ep SFT baseline Dataset. The default "
            "is a local dry-run; pass --upload for a Kaggle mutation."
        )
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--stage-dir", type=Path, default=STAGE_DIR)
    parser.add_argument("--dataset-slug", default=DATASET_SLUG)
    parser.add_argument(
        "--message",
        default="Freeze exact BGE 2ep SFT baseline IID/hard predictions v1",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--upload",
        action="store_true",
        help="Explicitly create/version the private Kaggle Dataset",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicit spelling of the default local-only behavior",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    env_file = args.env_file if args.env_file.is_absolute() else ROOT / args.env_file
    kaggle.load_dotenv(env_file)
    owner = kaggle.validate_slug(
        os.getenv("KAGGLE_USERNAME", "").strip(), "KAGGLE_USERNAME"
    )
    entry = expected_baseline_entry(owner)
    source_dir = args.source_dir or default_source_dir(entry)
    if not source_dir.is_absolute():
        source_dir = ROOT / source_dir
    stage_dir = args.stage_dir if args.stage_dir.is_absolute() else ROOT / args.stage_dir
    manifest = build_payload(
        source_dir,
        stage_dir,
        owner,
        entry=entry,
        dataset_slug=args.dataset_slug,
    )
    manifest_sha = verify_payload_for_upload(stage_dir.resolve(), manifest)
    total_bytes = sum(
        int(record["bytes"])
        for record in manifest["files"].values()
        if isinstance(record, Mapping)
    )
    print(
        json.dumps(
            {
                "mode": "upload" if args.upload else "dry_run",
                "dataset": manifest["dataset"],
                "stage_dir": str(stage_dir.resolve()),
                "baseline_run_id": manifest["binding"]["baseline_run_id"],
                "campaign_identity_sha256": manifest["binding"][
                    "campaign_identity_sha256"
                ],
                "payload_bytes": total_bytes,
                "manifest_sha256": manifest_sha,
                "ood_predictions": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not args.upload:
        print("Dry run complete; Kaggle was not contacted.")
        return 0
    if not os.getenv("KAGGLE_API_TOKEN", "").strip():
        kaggle.fail("set KAGGLE_API_TOKEN in .env")

    # No Kaggle CLI resolution or status request occurs before --upload.
    cli = kaggle.kaggle_command()
    dataset_ref = str(manifest["dataset"])
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
    manifest_sha = verify_payload_for_upload(stage_dir.resolve(), manifest)
    print("$", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, env=os.environ.copy())
    if result.returncode:
        kaggle.fail(
            f"BGE baseline Dataset upload failed with exit code {result.returncode}",
            result.returncode,
        )
    status = shared_push.wait_until_ready(
        cli,
        dataset_ref,
        minimum_version=previous_version + 1,
    )
    verify_remote_dataset(
        cli,
        dataset_ref,
        expected_manifest_sha256=manifest_sha,
    )
    print(
        f"Private BGE baseline Dataset ready at version "
        f"{status.get('current_version_number')}: {dataset_ref}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
