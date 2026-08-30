#!/usr/bin/env python3
"""Publish and evaluate an all-positive synthetic-card ablation on Kaggle.

The launcher is intentionally independent of any one generator.  It accepts a
validated local snapshot for the human-skeleton A+B, legacy rehydrated A+B, or
near-duplicate experiment, pins every input and provenance document by SHA-256,
reindexes item IDs into a deterministic synthetic-only namespace, creates a
private Kaggle Dataset, and runs the locked MiniLM 5ep data-ablation notebook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import numbers
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from item_pipeline.normalization import parse_attributes
from scripts.freeze_generated_pair_dataset import canonical_card
from src.data_pipeline import serialize_product

import create_minilm_5ep_team_ablation_notebook as frozen_notebook
import build_soft_positive_human_skeleton_pairs as human_skeleton_builder
import push_generation_rule_pairs_dataset as upload_helpers
import push_kaggle_training_dataset as shared_push
import run_kaggle_notebook as kaggle


DATASET_KINDS = ("rehydrated_ab", "human_skeleton_ab", "near_duplicate")
HARDENED_DATASET_KINDS = {"human_skeleton_ab", "near_duplicate"}
REQUIRED_PROVENANCE_DOCUMENTS = {
    "summary.json",
    "validation_report.json",
    "distribution_report.json",
    "build_manifest.json",
}
EXPECTED_BUILDER_VERSIONS = {
    "human_skeleton_ab": "soft_positive_human_skeleton_overlay_v2",
    "near_duplicate": "surface_positive_augmentation_v1",
}
EXPECTED_VALIDATION_VERSIONS = {
    "human_skeleton_ab": "soft_positive_human_skeleton_validator_v2",
    "near_duplicate": "surface_positive_no_atomic_change_validator_v1",
}
HUMAN_SKELETON_EVIDENCE_VERSION = (
    "label1_source_example_exact_subfacet_grounding_v2"
)
HUMAN_SKELETON_SELECTION_VERSION = "exact_scope_balanced_human_positive_pair_v1"
HUMAN_SKELETON_OVERLAY_ALLOWLIST_VERSION = (
    "non_title_metadata_exact_signatures_v1"
)
HUMAN_SKELETON_OVERLAY_ALLOWLIST_FAMILIES = {
    "warranty": [["warranty"], ["duration", "warranty"]],
}
HUMAN_SKELETON_CONSTRUCTION_MODES = {"overlay", "source_pair_surface"}
FACT_CLONE_DIAGNOSTICS_VERSION = "punctuation_insensitive_fact_clone_diagnostics_v1"
VALIDATION_OVERLAP_FILTER_VERSION = "frozen_validation_serialized_fact_exclusion_v1"
VALIDATION_FACT_KEY_VERSION = "ordered_fact_tokens_serialize_product_6000_v1"
VALIDATION_PROVENANCE_FILENAMES = {
    "validation_items": "items.parquet",
    "validation_iid_pairs": "iid_validation_pairs.parquet",
    "validation_hard_pairs": "hard_validation_pairs.parquet",
    "validation_ood_pairs": "ood_validation_pairs.parquet",
}
SOURCE_SCHEMA_VERSION = "positive_synthetic_source_snapshot_v1"
UPLOAD_SCHEMA_VERSION = 4
ID_REINDEX_VERSION = "positive_synthetic_negative_int64_v1"
SYNTHETIC_ID_START = -8_000_000_000_000_000_000
FROZEN_OOD_CATEGORIES = {"Одежда", "Бытовая техника"}
VALIDATION_DATASET = "alexproger23/product-matching-validation-splits-v1"
CHECKPOINT_DATASET = frozen_notebook.CHECKPOINT_DATASET
SIGNIFICANCE_DATASET = frozen_notebook.SIGNIFICANCE_BASELINE_DATASET
EXPECTED_SPLITS = {"iid", "hard", "ood"}
PROVENANCE_FILENAME = "positive_synthetic_source_provenance.json"
VALIDATION_FILENAME = "positive_synthetic_validation.json"
REPORT_SCHEMA_VERSION = 1
SECRET_LIKE_KEYS = {
    "api_key",
    "apikey",
    "api_token",
    "access_token",
    "refresh_token",
    "authorization",
    "password",
    "secret",
    "private_key",
    "credential",
    "credentials",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-kind", choices=DATASET_KINDS, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--pair-count", type=int, required=True)
    parser.add_argument(
        "--provenance-file",
        action="append",
        type=Path,
        required=True,
        help="repeat for every JSON generation/audit manifest that must be pinned",
    )
    parser.add_argument("--items-file", default="items.parquet")
    parser.add_argument("--pairs-file", default="pairs.parquet")
    parser.add_argument(
        "--metadata-file", default="pair_generation_metadata.parquet"
    )
    parser.add_argument("--dataset-slug", required=True)
    parser.add_argument("--artifact-tag", required=True)
    parser.add_argument("--experiment-label", required=True)
    parser.add_argument("--kernel-slug", required=True)
    parser.add_argument("--title")
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--stage-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--label-source")
    parser.add_argument("--notes")
    parser.add_argument("--dataset-message")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument(
        "--dry-run-only",
        action="store_true",
        help="validate, stage, generate and locally dry-run without contacting Kaggle",
    )
    parser.add_argument(
        "--skip-dataset-upload",
        action="store_true",
        help="reuse an already verified private Dataset version",
    )
    parser.add_argument(
        "--monitor-existing",
        action="store_true",
        help="wait for and download an already pushed kernel instead of pushing again",
    )
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {description}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must be a JSON object: {path}")
    return value


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run(command: list[str]) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _exact_int(value: Any, *, field: str, row: int) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"{field} contains bool at row {row}")
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value.strip())
    raise RuntimeError(f"{field} must contain exact integer IDs at row {row}: {value!r}")


def exact_int_series(series: pd.Series, *, field: str) -> pd.Series:
    values = [
        _exact_int(value, field=field, row=index)
        for index, value in enumerate(series.tolist())
    ]
    minimum, maximum = -(2**63), 2**63 - 1
    if any(value < minimum or value > maximum for value in values):
        raise RuntimeError(f"{field} contains an ID outside signed int64")
    return pd.Series(values, index=series.index, dtype="int64")


def reject_secret_like_keys(value: Any, *, location: str = "$") -> None:
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key).strip().casefold().replace("-", "_")
            if key in SECRET_LIKE_KEYS:
                raise RuntimeError(
                    f"provenance document contains secret-like key at {location}.{raw_key}"
                )
            reject_secret_like_keys(child, location=f"{location}.{raw_key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_secret_like_keys(child, location=f"{location}[{index}]")


def load_provenance_documents(paths: list[Path]) -> list[dict[str, Any]]:
    if not paths:
        raise RuntimeError("at least one JSON provenance document is required")
    documents: list[dict[str, Any]] = []
    labels: set[str] = set()
    for raw_path in paths:
        path = absolute(raw_path)
        if not path.is_file():
            raise RuntimeError(f"missing provenance document: {path}")
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        stability_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        stability_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if stability_before != stability_after or len(raw) != after.st_size:
            raise RuntimeError(f"provenance document changed while reading: {path}")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid provenance document: {path}: {error}") from error
        if not isinstance(payload, dict):
            raise RuntimeError(f"provenance document must be a JSON object: {path}")
        reject_secret_like_keys(payload)
        label = path.name
        if label in labels:
            raise RuntimeError(f"provenance basenames must be unique: {label}")
        labels.add(label)
        documents.append(
            {
                "name": label,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "payload": payload,
            }
        )
    return documents


def _required_exact_int(value: Any, *, field: str, expected: int) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise RuntimeError(f"{field} must be the exact integer {expected}")
    if int(value) != expected:
        raise RuntimeError(f"{field} differs: {int(value)} != {expected}")


def _required_target_counts(value: Any, *, field: str, pair_count: int) -> None:
    if not isinstance(value, dict):
        raise RuntimeError(f"{field} must be an object")
    try:
        observed = {
            str(key): _exact_int(raw, field=f"{field}.{key}", row=0)
            for key, raw in value.items()
        }
    except RuntimeError as error:
        raise RuntimeError(f"invalid {field}: {error}") from error
    expected = {"0": 0, "1": pair_count}
    if observed != expected:
        raise RuntimeError(f"{field} differs: {observed} != {expected}")


def _required_nonempty_sha256(value: Any, *, field: str) -> str:
    result = str(value or "").strip().casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", result):
        raise RuntimeError(f"{field} must be a non-empty SHA-256")
    return result


def _verify_available_source_provenance(
    source_provenance: dict[str, Any], *, field: str
) -> None:
    """Verify referenced local inputs when their recorded paths are available."""

    for name, record in source_provenance.items():
        if isinstance(record, dict) and "path" in record and "sha256" in record:
            expected_sha = _required_nonempty_sha256(
                record["sha256"], field=f"{field}.{name}.sha256"
            )
            path = Path(str(record["path"])).expanduser()
            if path.is_file():
                actual = _file_record(path)
                if actual["sha256"] != expected_sha:
                    raise RuntimeError(
                        f"{field}.{name} local SHA-256 differs from provenance"
                    )
                if "bytes" in record:
                    _required_exact_int(
                        record["bytes"],
                        field=f"{field}.{name}.bytes",
                        expected=int(actual["bytes"]),
                    )
        if not str(name).endswith("_path"):
            continue
        prefix = str(name)[: -len("_path")]
        sha_key = f"{prefix}_sha256"
        if sha_key not in source_provenance:
            continue
        expected_sha = _required_nonempty_sha256(
            source_provenance[sha_key], field=f"{field}.{sha_key}"
        )
        path = Path(str(record)).expanduser()
        if path.is_file() and sha256_file(path) != expected_sha:
            raise RuntimeError(
                f"{field}.{prefix} local SHA-256 differs from provenance"
            )


def _verify_manifest_files(
    manifest: dict[str, Any],
    *,
    source_dir: Path,
    required_filenames: set[str],
) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("build_manifest.files must be an object")
    missing = required_filenames - set(files)
    if missing:
        raise RuntimeError(
            f"build_manifest.files lacks required outputs: {sorted(missing)}"
        )
    for name, record in files.items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or not isinstance(record, dict)
        ):
            raise RuntimeError(f"invalid build_manifest file record: {name!r}")
        path = source_dir / name
        actual = _file_record(path)
        expected_sha = _required_nonempty_sha256(
            record.get("sha256"), field=f"build_manifest.files.{name}.sha256"
        )
        _required_exact_int(
            record.get("bytes"),
            field=f"build_manifest.files.{name}.bytes",
            expected=int(actual["bytes"]),
        )
        if actual["sha256"] != expected_sha:
            raise RuntimeError(f"build_manifest SHA-256 differs for {name}")


def _exact_nonnegative_count_map(value: Any, *, field: str) -> dict[str, int]:
    if not isinstance(value, dict) or not value:
        raise RuntimeError(f"{field} must be a non-empty count object")
    result: dict[str, int] = {}
    for raw_key, raw_count in value.items():
        key = str(raw_key)
        if (
            not key
            or isinstance(raw_count, bool)
            or not isinstance(raw_count, numbers.Integral)
            or int(raw_count) < 0
        ):
            raise RuntimeError(f"{field} contains an invalid count for {raw_key!r}")
        result[key] = int(raw_count)
    return result


def _verify_clone_count_block(
    value: Any, *, field: str, expected_pairs: int
) -> None:
    if not isinstance(value, dict) or set(value) != {"cards", "pairs"}:
        raise RuntimeError(f"{field} must contain exact cards/pairs diagnostics")
    for unit, expected_total in (("cards", expected_pairs * 2), ("pairs", expected_pairs)):
        metrics = value[unit]
        if not isinstance(metrics, dict):
            raise RuntimeError(f"{field}.{unit} must be an object")
        expected_keys = {
            "total",
            "unique",
            "excess_clones",
            "fact_identical_to_human_positive",
        }
        if set(metrics) != expected_keys:
            raise RuntimeError(f"{field}.{unit} has unexpected clone metrics")
        counts = _exact_nonnegative_count_map(metrics, field=f"{field}.{unit}")
        if counts["total"] != expected_total:
            raise RuntimeError(
                f"{field}.{unit}.total differs: {counts['total']} != {expected_total}"
            )
        if counts["unique"] + counts["excess_clones"] != expected_total:
            raise RuntimeError(f"{field}.{unit} clone accounting is inconsistent")
        if counts["fact_identical_to_human_positive"] > expected_total:
            raise RuntimeError(f"{field}.{unit} human-clone count exceeds total")


def _verify_fact_clone_diagnostics(
    summary: dict[str, Any],
    distribution: dict[str, Any],
    *,
    pair_count: int,
    construction_mode_counts: dict[str, int],
) -> dict[str, Any]:
    diagnostics = summary.get("fact_clone_diagnostics")
    if not isinstance(diagnostics, dict) or diagnostics != distribution.get(
        "fact_clone_diagnostics"
    ):
        raise RuntimeError("summary/distribution fact-clone diagnostics mismatch")
    if diagnostics.get("version") != FACT_CLONE_DIAGNOSTICS_VERSION:
        raise RuntimeError("fact-clone diagnostics version is not pinned")
    if diagnostics.get("nonblocking") is not True:
        raise RuntimeError("fact-clone diagnostics must be explicitly nonblocking")
    _verify_clone_count_block(
        diagnostics.get("all"), field="fact_clone_diagnostics.all", expected_pairs=pair_count
    )
    by_mode = diagnostics.get("by_construction_mode")
    if not isinstance(by_mode, dict) or set(by_mode) != set(construction_mode_counts):
        raise RuntimeError("fact-clone diagnostics construction modes differ")
    for mode, count in construction_mode_counts.items():
        _verify_clone_count_block(
            by_mode[mode],
            field=f"fact_clone_diagnostics.by_construction_mode.{mode}",
            expected_pairs=count,
        )
    human_reference = diagnostics.get("human_positive_reference")
    if not isinstance(human_reference, dict) or set(human_reference) != {
        "unique_cards",
        "unique_pairs",
    }:
        raise RuntimeError("fact-clone human reference is incomplete")
    reference_counts = _exact_nonnegative_count_map(
        human_reference, field="fact_clone_diagnostics.human_positive_reference"
    )
    if min(reference_counts.values()) < 1:
        raise RuntimeError("fact-clone human reference must be non-empty")
    for unit in ("cards", "pairs"):
        all_identical = diagnostics["all"][unit][
            "fact_identical_to_human_positive"
        ]
        mode_identical = sum(
            by_mode[mode][unit]["fact_identical_to_human_positive"]
            for mode in construction_mode_counts
        )
        if all_identical != mode_identical:
            raise RuntimeError(
                f"fact-clone {unit} human-reference accounting is inconsistent"
            )
    return diagnostics


def _verify_human_overlay_allowlist(
    summary: dict[str, Any],
    distribution: dict[str, Any],
    manifest: dict[str, Any],
    metadata: pd.DataFrame,
    *,
    pair_count: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    allowlist = summary.get("overlay_allowlist")
    if (
        not isinstance(allowlist, dict)
        or allowlist != distribution.get("overlay_allowlist")
        or allowlist != manifest.get("overlay_allowlist")
    ):
        raise RuntimeError("summary/distribution/manifest overlay allowlist mismatch")
    if set(allowlist) != {
        "version",
        "families",
        "task_counts_by_family",
        "construction_mode_counts_by_family",
    }:
        raise RuntimeError("overlay allowlist report has unexpected fields")
    if allowlist.get("version") != HUMAN_SKELETON_OVERLAY_ALLOWLIST_VERSION:
        raise RuntimeError("overlay allowlist version is not pinned")
    if allowlist.get("families") != HUMAN_SKELETON_OVERLAY_ALLOWLIST_FAMILIES:
        raise RuntimeError("overlay allowlist family definitions are not pinned")

    required_columns = {
        "construction_mode",
        "overlay_allowlist_version",
        "overlay_allowlist_family",
        "semantic_keys_compatible_for_overlay",
        "schema_safe_for_overlay",
        "required_attribute_key",
        "target_key_a",
        "target_key_b",
        "target_value_a",
        "target_value_b",
        "concept",
    }
    if missing := required_columns - set(metadata.columns):
        raise RuntimeError(f"human-skeleton metadata lacks overlay fields: {sorted(missing)}")
    versions = set(metadata["overlay_allowlist_version"].tolist())
    if versions != {HUMAN_SKELETON_OVERLAY_ALLOWLIST_VERSION}:
        raise RuntimeError("metadata overlay allowlist version mismatch")
    modes = metadata["construction_mode"]
    if modes.isna().any() or set(modes.tolist()) != HUMAN_SKELETON_CONSTRUCTION_MODES:
        raise RuntimeError("metadata construction modes are not pinned")
    families = metadata["overlay_allowlist_family"]
    if families.isna().any() or not families.map(lambda value: isinstance(value, str)).all():
        raise RuntimeError("metadata overlay allowlist family must be a string")
    allowed_families = set(HUMAN_SKELETON_OVERLAY_ALLOWLIST_FAMILIES)
    unsupported = set(families.tolist()) - allowed_families - {""}
    if unsupported:
        raise RuntimeError(f"metadata contains unsupported overlay families: {sorted(unsupported)}")
    expected_builder_allowlist = {
        "version": HUMAN_SKELETON_OVERLAY_ALLOWLIST_VERSION,
        "families": HUMAN_SKELETON_OVERLAY_ALLOWLIST_FAMILIES,
    }
    if human_skeleton_builder.overlay_allowlist_definition() != expected_builder_allowlist:
        raise RuntimeError("launcher/builder overlay allowlist pins differ")
    for column in (
        "semantic_keys_compatible_for_overlay",
        "schema_safe_for_overlay",
    ):
        if (
            not pd.api.types.is_bool_dtype(metadata[column].dtype)
            or metadata[column].isna().any()
        ):
            raise RuntimeError(f"metadata {column} must contain exact booleans")
    for index, row in metadata.iterrows():
        semantic_replay = human_skeleton_builder.semantic_key_compatible_for_overlay(
            row["required_attribute_key"],
            row["target_key_a"],
            row["target_key_b"],
            row["concept"],
        )
        expected_family = ""
        if semantic_replay:
            expected_family = (
                human_skeleton_builder._safe_overlay_family(
                    human_skeleton_builder._semantic_subfacet_signature(
                        row["required_attribute_key"]
                    )
                )
                or ""
            )
        schema_replay = bool(
            semantic_replay
            and human_skeleton_builder._fallback_schema_value_safe(
                row["target_key_a"], row["target_value_a"]
            )
            and human_skeleton_builder._fallback_schema_value_safe(
                row["target_key_b"], row["target_value_b"]
            )
        )
        if str(row["overlay_allowlist_family"]) != expected_family:
            raise RuntimeError(
                "metadata overlay family does not replay from exact key signature "
                f"at row {index}"
            )
        if bool(row["semantic_keys_compatible_for_overlay"]) != semantic_replay:
            raise RuntimeError(
                f"metadata semantic overlay proof does not replay at row {index}"
            )
        if bool(row["schema_safe_for_overlay"]) != schema_replay:
            raise RuntimeError(
                f"metadata schema overlay proof does not replay at row {index}"
            )
    overlay_mask = modes.eq("overlay")
    if not overlay_mask.any() or not modes.eq("source_pair_surface").any():
        raise RuntimeError("human-skeleton source must contain overlay and fallback rows")
    if families.loc[overlay_mask].eq("").any():
        raise RuntimeError("overlay row is not assigned to an allowlisted family")
    if (
        not metadata.loc[overlay_mask, "semantic_keys_compatible_for_overlay"].eq(True).all()
        or not metadata.loc[overlay_mask, "schema_safe_for_overlay"].eq(True).all()
    ):
        raise RuntimeError("overlay row lacks semantic/schema safety proof")

    report_families = families.replace("", "not_allowlisted")
    task_counts = {
        str(key): int(value)
        for key, value in report_families.value_counts().sort_index().items()
    }
    if task_counts != _exact_nonnegative_count_map(
        allowlist.get("task_counts_by_family"),
        field="overlay_allowlist.task_counts_by_family",
    ):
        raise RuntimeError("overlay allowlist task counts differ from metadata")
    mode_family_counts = {
        str(mode): {
            str(key): int(value)
            for key, value in report_families.loc[modes.eq(mode)]
            .value_counts()
            .sort_index()
            .items()
        }
        for mode in sorted(HUMAN_SKELETON_CONSTRUCTION_MODES)
    }
    stored_mode_family_counts = allowlist.get("construction_mode_counts_by_family")
    if not isinstance(stored_mode_family_counts, dict):
        raise RuntimeError("overlay allowlist construction-mode counts are missing")
    normalized_stored_counts = {
        str(mode): _exact_nonnegative_count_map(
            counts,
            field=f"overlay_allowlist.construction_mode_counts_by_family.{mode}",
        )
        for mode, counts in stored_mode_family_counts.items()
    }
    if normalized_stored_counts != mode_family_counts:
        raise RuntimeError("overlay allowlist construction-mode counts differ from metadata")

    construction_mode_counts = {
        str(key): int(value) for key, value in modes.value_counts().sort_index().items()
    }
    if sum(construction_mode_counts.values()) != pair_count:
        raise RuntimeError("metadata construction-mode counts differ from pair count")
    skeleton_distribution = summary.get("skeleton_distribution")
    skeletons = distribution.get("skeletons")
    if not isinstance(skeleton_distribution, dict) or skeleton_distribution != skeletons:
        raise RuntimeError("summary/distribution skeleton counts mismatch")
    if _exact_nonnegative_count_map(
        skeleton_distribution.get("construction_modes"),
        field="skeleton_distribution.construction_modes",
    ) != construction_mode_counts:
        raise RuntimeError("skeleton construction-mode counts differ from metadata")
    _required_exact_int(
        skeleton_distribution.get("total_assignments"),
        field="skeleton_distribution.total_assignments",
        expected=pair_count,
    )
    return allowlist, construction_mode_counts


def _required_local_provenance_file(
    source_provenance: dict[str, Any],
    *,
    key: str,
    expected_filename: str,
) -> Path:
    record = source_provenance.get(key)
    if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
        raise RuntimeError(f"summary.source_provenance.{key} is not an exact file record")
    path = Path(str(record["path"])).expanduser().resolve()
    if path.name != expected_filename or not path.is_file():
        raise RuntimeError(
            f"summary.source_provenance.{key} must resolve to {expected_filename}"
        )
    actual = _file_record(path)
    _required_exact_int(
        record["bytes"],
        field=f"summary.source_provenance.{key}.bytes",
        expected=int(actual["bytes"]),
    )
    if _required_nonempty_sha256(
        record["sha256"], field=f"summary.source_provenance.{key}.sha256"
    ) != actual["sha256"]:
        raise RuntimeError(f"summary.source_provenance.{key} SHA-256 differs")
    return path


def _verify_validation_overlap_filter(
    *,
    summary: dict[str, Any],
    validation: dict[str, Any],
    distribution: dict[str, Any],
    manifest: dict[str, Any],
    source_provenance: dict[str, Any],
    items: pd.DataFrame,
    metadata: pd.DataFrame,
    pair_count: int,
) -> dict[str, Any]:
    report = summary.get("validation_overlap_filter")
    if not isinstance(report, dict) or any(
        payload.get("validation_overlap_filter") != report
        for payload in (validation, distribution, manifest)
    ):
        raise RuntimeError(
            "summary/validation/distribution/manifest validation-overlap reports differ"
        )
    expected_report_fields = {
        "version",
        "fact_key_version",
        "source_task_count",
        "emitted_pair_count",
        "dropped_pair_count",
        "dropped_task_ids",
        "dropped_pairs_by_split",
        "dropped_endpoints_by_side",
        "dropped_pairs_by_construction_mode",
        "unique_dropped_card_facts",
        "postfilter_overlapping_card_count",
        "validation_reference",
        "dropped_pairs",
    }
    if set(report) != expected_report_fields:
        raise RuntimeError("validation-overlap report fields are not pinned")
    if report.get("version") != VALIDATION_OVERLAP_FILTER_VERSION:
        raise RuntimeError("validation-overlap filter version is not pinned")
    if report.get("fact_key_version") != VALIDATION_FACT_KEY_VERSION:
        raise RuntimeError("validation-overlap fact-key version is not pinned")

    dropped_count = _exact_int(
        report.get("dropped_pair_count"),
        field="validation_overlap_filter.dropped_pair_count",
        row=0,
    )
    source_task_count = _exact_int(
        report.get("source_task_count"),
        field="validation_overlap_filter.source_task_count",
        row=0,
    )
    _required_exact_int(
        report.get("emitted_pair_count"),
        field="validation_overlap_filter.emitted_pair_count",
        expected=pair_count,
    )
    if dropped_count < 0 or source_task_count != pair_count + dropped_count:
        raise RuntimeError("validation-overlap source/emitted/dropped arithmetic differs")
    for field, payload in (
        ("summary.source_task_count", summary),
        ("validation_report.source_task_count", validation),
        ("build_manifest.source_task_count", manifest),
    ):
        _required_exact_int(
            payload.get("source_task_count"), field=field, expected=source_task_count
        )
    for field, config in (
        ("summary.config", summary.get("config")),
        ("build_manifest.config", manifest.get("config")),
    ):
        if not isinstance(config, dict):
            raise RuntimeError(f"{field} must be an object")
        _required_exact_int(
            config.get("source_task_count"),
            field=f"{field}.source_task_count",
            expected=source_task_count,
        )
        _required_exact_int(
            config.get("dropped_validation_overlap"),
            field=f"{field}.dropped_validation_overlap",
            expected=dropped_count,
        )

    validation_paths = {
        key: _required_local_provenance_file(
            source_provenance,
            key=key,
            expected_filename=filename,
        )
        for key, filename in VALIDATION_PROVENANCE_FILENAMES.items()
    }
    validation_dir = validation_paths["validation_items"].parent
    for key, filename in VALIDATION_PROVENANCE_FILENAMES.items():
        if validation_paths[key] != (validation_dir / filename).resolve():
            raise RuntimeError("frozen validation provenance files do not share one directory")
    try:
        validation_facts = human_skeleton_builder.load_frozen_validation_facts(
            validation_dir
        )
    except Exception as error:
        raise RuntimeError("could not replay frozen validation facts") from error
    expected_reference = {
        "split_pair_counts": dict(validation_facts.split_pair_counts),
        "split_item_counts": dict(validation_facts.split_item_counts),
        "unique_item_ids": int(validation_facts.unique_item_ids),
        "unique_fact_keys": int(validation_facts.unique_fact_keys),
    }
    if report.get("validation_reference") != expected_reference:
        raise RuntimeError("validation-overlap frozen reference statistics differ")

    source_metadata_path = _required_local_provenance_file(
        source_provenance,
        key="source_metadata",
        expected_filename="pair_generation_metadata.parquet",
    )
    try:
        source_metadata = pd.read_parquet(
            source_metadata_path, columns=["composition_index", "task_index"]
        )
    except Exception as error:
        raise RuntimeError("could not read signed source task metadata") from error
    for frame, label in (
        (source_metadata, "signed source metadata"),
        (metadata, "emitted metadata"),
    ):
        if "composition_index" not in frame:
            raise RuntimeError(f"{label} lacks composition_index")
        values = exact_int_series(
            frame["composition_index"], field=f"{label}.composition_index"
        )
        if values.duplicated().any():
            raise RuntimeError(f"{label} composition_index is not unique")
    source_compositions = set(
        exact_int_series(
            source_metadata["composition_index"],
            field="signed source metadata.composition_index",
        ).tolist()
    )
    emitted_compositions = set(
        exact_int_series(
            metadata["composition_index"], field="emitted metadata.composition_index"
        ).tolist()
    )
    if len(source_compositions) != source_task_count:
        raise RuntimeError("signed source composition count differs")
    if not emitted_compositions <= source_compositions:
        raise RuntimeError("emitted composition IDs are absent from signed source")
    source_task_by_composition = dict(
        zip(
            exact_int_series(
                source_metadata["composition_index"],
                field="signed source metadata.composition_index",
            ).tolist(),
            exact_int_series(
                source_metadata["task_index"], field="signed source metadata.task_index"
            ).tolist(),
            strict=True,
        )
    )

    raw_dropped_ids = report.get("dropped_task_ids")
    dropped_pairs = report.get("dropped_pairs")
    if not isinstance(raw_dropped_ids, list) or not isinstance(dropped_pairs, list):
        raise RuntimeError("validation-overlap dropped task evidence must be lists")
    dropped_ids = [
        _exact_int(value, field="validation_overlap_filter.dropped_task_ids", row=index)
        for index, value in enumerate(raw_dropped_ids)
    ]
    if len(dropped_ids) != dropped_count or len(set(dropped_ids)) != dropped_count:
        raise RuntimeError("validation-overlap dropped composition IDs are not unique/exact")
    if set(dropped_ids) != source_compositions - emitted_compositions:
        raise RuntimeError("validation-overlap dropped composition set differs")
    if len(dropped_pairs) != dropped_count:
        raise RuntimeError("validation-overlap dropped-pair evidence count differs")

    emitted_item_ids = set(int(value) for value in items["id"])
    seen_dropped_item_ids: set[int] = set()
    replayed_dropped_ids: list[int] = []
    replayed_split_counts: Counter[str] = Counter()
    replayed_side_counts: Counter[str] = Counter()
    replayed_mode_counts: Counter[str] = Counter()
    replayed_fact_hashes: set[str] = set()
    for pair_index, pair_record in enumerate(dropped_pairs):
        if not isinstance(pair_record, dict) or set(pair_record) != {
            "composition_index",
            "source_task_index",
            "generated_id1",
            "generated_id2",
            "construction_mode",
            "overlap_endpoints",
        }:
            raise RuntimeError("validation-overlap dropped-pair record is not exact")
        composition_index = _exact_int(
            pair_record["composition_index"],
            field="validation_overlap_filter.dropped_pairs.composition_index",
            row=pair_index,
        )
        replayed_dropped_ids.append(composition_index)
        source_task_index = _exact_int(
            pair_record["source_task_index"],
            field="validation_overlap_filter.dropped_pairs.source_task_index",
            row=pair_index,
        )
        if source_task_by_composition.get(composition_index) != source_task_index:
            raise RuntimeError("dropped pair source task does not match signed metadata")
        generated_ids = {
            "a": _exact_int(
                pair_record["generated_id1"],
                field="validation_overlap_filter.dropped_pairs.generated_id1",
                row=pair_index,
            ),
            "b": _exact_int(
                pair_record["generated_id2"],
                field="validation_overlap_filter.dropped_pairs.generated_id2",
                row=pair_index,
            ),
        }
        if (
            len(set(generated_ids.values())) != 2
            or set(generated_ids.values()) & emitted_item_ids
            or set(generated_ids.values()) & seen_dropped_item_ids
        ):
            raise RuntimeError("dropped generated item IDs are reused or still emitted")
        seen_dropped_item_ids.update(generated_ids.values())
        mode = str(pair_record["construction_mode"])
        if mode not in HUMAN_SKELETON_CONSTRUCTION_MODES:
            raise RuntimeError("dropped pair construction mode is unsupported")
        replayed_mode_counts[mode] += 1
        endpoints = pair_record["overlap_endpoints"]
        if not isinstance(endpoints, list) or not endpoints:
            raise RuntimeError("dropped pair lacks validation-overlap endpoints")
        seen_sides: set[str] = set()
        pair_splits: set[str] = set()
        for endpoint in endpoints:
            if not isinstance(endpoint, dict) or set(endpoint) != {
                "side",
                "generated_item_id",
                "fact_key_sha256",
                "fact_tokens",
                "validation_matches_by_split",
            }:
                raise RuntimeError("validation-overlap endpoint record is not exact")
            side = str(endpoint["side"])
            if side not in {"a", "b"} or side in seen_sides:
                raise RuntimeError("validation-overlap endpoint side is invalid or duplicated")
            seen_sides.add(side)
            replayed_side_counts[side] += 1
            if _exact_int(
                endpoint["generated_item_id"],
                field="validation_overlap_filter.endpoint.generated_item_id",
                row=pair_index,
            ) != generated_ids[side]:
                raise RuntimeError("validation-overlap endpoint item ID differs")
            tokens_value = endpoint["fact_tokens"]
            if (
                not isinstance(tokens_value, list)
                or not tokens_value
                or any(not isinstance(token, str) or not token for token in tokens_value)
            ):
                raise RuntimeError("validation-overlap fact tokens are invalid")
            tokens = tuple(tokens_value)
            fact_sha = human_skeleton_builder.sha256_json(list(tokens))
            if endpoint["fact_key_sha256"] != fact_sha:
                raise RuntimeError("validation-overlap fact hash does not match tokens")
            replayed_fact_hashes.add(fact_sha)
            expected_sources = validation_facts.fact_sources.get(tokens)
            if not expected_sources:
                raise RuntimeError("reported dropped fact is absent from frozen validation")
            expected_source_payload = {
                split: list(ids) for split, ids in sorted(expected_sources.items())
            }
            if endpoint["validation_matches_by_split"] != expected_source_payload:
                raise RuntimeError("reported validation matches do not replay exactly")
            pair_splits.update(expected_sources)
        replayed_split_counts.update(pair_splits)

    if replayed_dropped_ids != dropped_ids:
        raise RuntimeError("dropped composition ID list and evidence order differ")
    expected_split_counts = {
        split: int(replayed_split_counts[split])
        for split in human_skeleton_builder.VALIDATION_SPLITS
    }
    if report.get("dropped_pairs_by_split") != expected_split_counts:
        raise RuntimeError("validation-overlap split counts do not replay")
    if report.get("dropped_endpoints_by_side") != {
        side: int(replayed_side_counts[side]) for side in ("a", "b")
    }:
        raise RuntimeError("validation-overlap endpoint-side counts do not replay")
    if report.get("dropped_pairs_by_construction_mode") != dict(
        sorted(replayed_mode_counts.items())
    ):
        raise RuntimeError("validation-overlap construction-mode counts do not replay")
    _required_exact_int(
        report.get("unique_dropped_card_facts"),
        field="validation_overlap_filter.unique_dropped_card_facts",
        expected=len(replayed_fact_hashes),
    )

    postfilter_overlap = 0
    for row in items.itertuples(index=False):
        tokens = human_skeleton_builder._fact_card_tokens(
            str(row.name), parse_attributes(row.attributes), str(row.category)
        )
        postfilter_overlap += tokens in validation_facts.fact_sources
    if postfilter_overlap != 0:
        raise RuntimeError("emitted cards still overlap frozen validation facts")
    _required_exact_int(
        report.get("postfilter_overlapping_card_count"),
        field="validation_overlap_filter.postfilter_overlapping_card_count",
        expected=0,
    )
    return report


def _verify_hardened_provenance(
    *,
    dataset_kind: str,
    source_dir: Path,
    pair_count: int,
    items_filename: str,
    pairs_filename: str,
    metadata_filename: str,
    items: pd.DataFrame,
    metadata: pd.DataFrame,
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    if dataset_kind not in HARDENED_DATASET_KINDS:
        return {}
    document_by_name = {document["name"]: document["payload"] for document in documents}
    document_record_by_name = {
        document["name"]: {
            "bytes": document["bytes"],
            "sha256": document["sha256"],
        }
        for document in documents
    }
    observed_names = set(document_by_name)
    if observed_names != REQUIRED_PROVENANCE_DOCUMENTS:
        raise RuntimeError(
            "kind-specific provenance documents differ: "
            f"{sorted(observed_names)} != {sorted(REQUIRED_PROVENANCE_DOCUMENTS)}"
        )
    for name in REQUIRED_PROVENANCE_DOCUMENTS:
        if _file_record(source_dir / name) != document_record_by_name[name]:
            raise RuntimeError(
                f"provenance document is not bound to source snapshot: {name}"
            )
    summary = document_by_name["summary.json"]
    validation = document_by_name["validation_report.json"]
    distribution = document_by_name["distribution_report.json"]
    manifest = document_by_name["build_manifest.json"]
    expected_items = pair_count * 2
    expected_builder = EXPECTED_BUILDER_VERSIONS[dataset_kind]
    expected_validation = EXPECTED_VALIDATION_VERSIONS[dataset_kind]

    for field, payload in (
        ("summary", summary),
        ("distribution_report", distribution),
        ("build_manifest", manifest),
    ):
        if payload.get("builder_version") != expected_builder:
            raise RuntimeError(f"{field}.builder_version is not pinned")
    if summary.get("validation_version") != expected_validation:
        raise RuntimeError("summary.validation_version is not pinned")
    if validation.get("version") != expected_validation:
        raise RuntimeError("validation_report.version is not pinned")
    if validation.get("valid") is not True:
        raise RuntimeError("validation_report.valid must be true")
    if validation.get("errors") not in ({}, []):
        raise RuntimeError("validation_report.errors must be empty")

    _required_exact_int(
        summary.get("generated_pairs"), field="summary.generated_pairs", expected=pair_count
    )
    _required_exact_int(
        summary.get("generated_items"), field="summary.generated_items", expected=expected_items
    )
    _required_target_counts(
        summary.get("target_counts"), field="summary.target_counts", pair_count=pair_count
    )
    for field in ("checked_pairs", "pairs"):
        _required_exact_int(
            validation.get(field), field=f"validation_report.{field}", expected=pair_count
        )
    _required_exact_int(
        validation.get("items"), field="validation_report.items", expected=expected_items
    )
    _required_target_counts(
        validation.get("target_counts"),
        field="validation_report.target_counts",
        pair_count=pair_count,
    )
    _required_exact_int(manifest.get("pairs"), field="build_manifest.pairs", expected=pair_count)
    _required_exact_int(manifest.get("items"), field="build_manifest.items", expected=expected_items)
    _required_target_counts(
        manifest.get("targets"), field="build_manifest.targets", pair_count=pair_count
    )
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("build_manifest.config must be an object")
    _required_exact_int(config.get("count"), field="build_manifest.config.count", expected=pair_count)
    generated_distribution = distribution.get("generated")
    if not isinstance(generated_distribution, dict):
        raise RuntimeError("distribution_report.generated must be an object")
    _required_exact_int(
        generated_distribution.get("pairs"),
        field="distribution_report.generated.pairs",
        expected=pair_count,
    )
    if summary.get("validation") != validation:
        raise RuntimeError("summary.validation differs from validation_report.json")

    run_signature = _required_nonempty_sha256(
        summary.get("run_signature"), field="summary.run_signature"
    )
    if _required_nonempty_sha256(
        manifest.get("run_signature"), field="build_manifest.run_signature"
    ) != run_signature:
        raise RuntimeError("summary/build_manifest run_signature mismatch")
    if "run_signature" in metadata:
        metadata_signatures = {
            str(value).strip().casefold() for value in metadata["run_signature"].tolist()
        }
        if metadata_signatures != {run_signature}:
            raise RuntimeError("metadata run_signature mismatch")
    elif dataset_kind == "human_skeleton_ab":
        raise RuntimeError("human_skeleton_ab metadata lacks run_signature")

    source_provenance = summary.get("source_provenance")
    if not isinstance(source_provenance, dict) or not source_provenance:
        raise RuntimeError("summary.source_provenance must be a non-empty object")
    if manifest.get("source_provenance") != source_provenance:
        raise RuntimeError("summary/build_manifest source_provenance mismatch")
    _verify_available_source_provenance(
        source_provenance, field="summary.source_provenance"
    )

    label_source = str(summary.get("label_source") or "").strip()
    if not label_source or manifest.get("label_source") != label_source:
        raise RuntimeError("summary/build_manifest label_source mismatch")
    required_manifest_files = {
        items_filename,
        pairs_filename,
        metadata_filename,
        "summary.json",
        "validation_report.json",
        "distribution_report.json",
    }
    _verify_manifest_files(
        manifest,
        source_dir=source_dir,
        required_filenames=required_manifest_files,
    )

    overlay_allowlist: dict[str, Any] = {}
    construction_mode_counts: dict[str, int] = {}
    fact_clone_diagnostics: dict[str, Any] = {}
    validation_overlap_filter: dict[str, Any] = {}
    if "builder_version" in metadata:
        builder_versions = {str(value) for value in metadata["builder_version"].tolist()}
        if builder_versions != {expected_builder}:
            raise RuntimeError("metadata builder_version mismatch")
    if dataset_kind == "human_skeleton_ab":
        if summary.get("evidence_version") != HUMAN_SKELETON_EVIDENCE_VERSION:
            raise RuntimeError("human-skeleton summary.evidence_version is not pinned")
        if summary.get("selection_version") != HUMAN_SKELETON_SELECTION_VERSION:
            raise RuntimeError("human-skeleton summary.selection_version is not pinned")
        if validation.get("non_target_values_preserved") is not True:
            raise RuntimeError("human-skeleton non-target values were not preserved")
        _required_exact_int(
            validation.get("unique_cards"),
            field="validation_report.unique_cards",
            expected=expected_items,
        )
        gates = distribution.get("gates")
        if not isinstance(gates, dict) or not gates or any(value is not True for value in gates.values()):
            raise RuntimeError("human-skeleton distribution gates must all be true")
        if distribution.get("valid") is not True:
            raise RuntimeError("human-skeleton distribution_report.valid must be true")
        if summary.get("distribution_gates") != gates:
            raise RuntimeError("summary/distribution gate mismatch")
        overlay_allowlist, construction_mode_counts = (
            _verify_human_overlay_allowlist(
                summary,
                distribution,
                manifest,
                metadata,
                pair_count=pair_count,
            )
        )
        fact_clone_diagnostics = _verify_fact_clone_diagnostics(
            summary,
            distribution,
            pair_count=pair_count,
            construction_mode_counts=construction_mode_counts,
        )
        validation_overlap_filter = _verify_validation_overlap_filter(
            summary=summary,
            validation=validation,
            distribution=distribution,
            manifest=manifest,
            source_provenance=source_provenance,
            items=items,
            metadata=metadata,
            pair_count=pair_count,
        )
        for column in ("non_target_values_preserved", "observed_label1_transition"):
            if (
                column not in metadata
                or not pd.api.types.is_bool_dtype(metadata[column].dtype)
                or metadata[column].isna().any()
                or not metadata[column].all()
            ):
                raise RuntimeError(f"human-skeleton metadata {column} must be all true")
        if "evidence_human_label" not in metadata or not metadata["evidence_human_label"].eq(1).all():
            raise RuntimeError("human-skeleton metadata evidence_human_label must be all one")
    else:
        if manifest.get("no_atomic_change") is not True:
            raise RuntimeError("near-duplicate build_manifest.no_atomic_change must be true")
        if validation.get("no_new_or_changed_visible_facts") is not True:
            raise RuntimeError("near-duplicate validation must preserve visible facts")
        if summary.get("no_atomic_change") is not True:
            raise RuntimeError("near-duplicate summary.no_atomic_change must be true")
        interpretation = distribution.get("interpretation")
        if not isinstance(interpretation, dict) or interpretation.get("atomic_attribute_changes") != 0:
            raise RuntimeError("near-duplicate distribution must report zero atomic changes")
        if (
            "no_atomic_change" not in metadata
            or not pd.api.types.is_bool_dtype(metadata["no_atomic_change"].dtype)
            or metadata["no_atomic_change"].isna().any()
            or not metadata["no_atomic_change"].all()
        ):
            raise RuntimeError("near-duplicate metadata no_atomic_change must be all true")

    contract = {
        "builder_version": expected_builder,
        "validation_version": expected_validation,
        "run_signature": run_signature,
        "label_source": label_source,
        "source_provenance": source_provenance,
    }
    if dataset_kind == "human_skeleton_ab":
        contract.update(
            {
                "overlay_allowlist": overlay_allowlist,
                "construction_mode_counts": construction_mode_counts,
                "fact_clone_diagnostics": fact_clone_diagnostics,
                "validation_overlap_filter": validation_overlap_filter,
            }
        )
    return contract


def _required_columns(
    frame: pd.DataFrame, required: set[str], *, description: str
) -> None:
    if missing := required - set(frame.columns):
        raise RuntimeError(f"{description} lacks columns: {sorted(missing)}")


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing source file: {path}")
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    stability_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    stability_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if stability_before != stability_after:
        raise RuntimeError(f"source file changed while hashing: {path}")
    return {"bytes": after.st_size, "sha256": digest}


def _target_one(frame: pd.DataFrame, *, description: str) -> None:
    values = [
        _exact_int(value, field=f"{description}.target", row=index)
        for index, value in enumerate(frame["target"].tolist())
    ]
    if any(value != 1 for value in values):
        counts = Counter(values)
        raise RuntimeError(
            f"{description} must be all target=1, observed={dict(sorted(counts.items()))}"
        )


def verify_source(
    source_dir: Path,
    *,
    pair_count: int,
    dataset_kind: str,
    provenance_paths: list[Path],
    items_filename: str = "items.parquet",
    pairs_filename: str = "pairs.parquet",
    metadata_filename: str = "pair_generation_metadata.parquet",
) -> dict[str, Any]:
    if dataset_kind not in DATASET_KINDS:
        raise ValueError(f"unsupported dataset kind: {dataset_kind!r}")
    if pair_count < 1:
        raise ValueError("pair_count must be positive")
    source_dir = source_dir.resolve()
    paths = {
        "items": source_dir / items_filename,
        "pairs": source_dir / pairs_filename,
        "metadata": source_dir / metadata_filename,
    }
    file_records = {name: _file_record(path) for name, path in paths.items()}
    documents = load_provenance_documents(provenance_paths)
    provenance_path_by_name = {
        absolute(path).name: absolute(path) for path in provenance_paths
    }

    items = pd.read_parquet(paths["items"])
    pairs = pd.read_parquet(paths["pairs"])
    metadata = pd.read_parquet(paths["metadata"])
    _required_columns(
        items, {"id", "name", "attributes", "category"}, description="items"
    )
    _required_columns(pairs, {"id1", "id2", "target"}, description="pairs")
    _required_columns(
        metadata, {"id1", "id2", "target"}, description="generation metadata"
    )
    if len(pairs) != pair_count or len(items) != pair_count * 2:
        raise RuntimeError(
            f"source dimensions differ: pairs={len(pairs):,}, items={len(items):,}, "
            f"expected={pair_count:,}/{pair_count * 2:,}"
        )
    if len(metadata) != pair_count:
        raise RuntimeError(
            f"metadata rows differ: {len(metadata):,} != {pair_count:,}"
        )

    items = items.copy()
    pairs = pairs.copy()
    metadata = metadata.copy()
    items["id"] = exact_int_series(items["id"], field="items.id")
    for frame, label in ((pairs, "pairs"), (metadata, "metadata")):
        frame["id1"] = exact_int_series(frame["id1"], field=f"{label}.id1")
        frame["id2"] = exact_int_series(frame["id2"], field=f"{label}.id2")
    _target_one(pairs, description="pairs")
    _target_one(metadata, description="metadata")
    pairs["target"] = 1
    metadata["target"] = 1

    if items["id"].duplicated().any():
        raise RuntimeError("source item IDs are not unique")
    if pairs["id1"].eq(pairs["id2"]).any():
        raise RuntimeError("source contains self-pairs")
    endpoints = pd.concat([pairs["id1"], pairs["id2"]], ignore_index=True)
    if endpoints.duplicated().any():
        raise RuntimeError("every synthetic item must occur in exactly one pair")
    if set(endpoints) != set(items["id"]):
        raise RuntimeError("source item catalogue does not exactly match pair endpoints")
    unordered_keys = [
        tuple(sorted((int(row.id1), int(row.id2))))
        for row in pairs.itertuples(index=False)
    ]
    if len(unordered_keys) != len(set(unordered_keys)):
        raise RuntimeError("source contains duplicate unordered pairs")

    pair_keys = [
        (int(row.id1), int(row.id2)) for row in pairs.itertuples(index=False)
    ]
    metadata_keys = [
        (int(row.id1), int(row.id2)) for row in metadata.itertuples(index=False)
    ]
    if len(metadata_keys) != len(set(metadata_keys)) or set(metadata_keys) != set(
        pair_keys
    ):
        raise RuntimeError("metadata is not a one-to-one map of oriented source pairs")
    if "task_index" in metadata:
        task_indices = exact_int_series(metadata["task_index"], field="metadata.task_index")
        if task_indices.duplicated().any():
            raise RuntimeError("metadata task_index is not unique")
        metadata["task_index"] = task_indices

    if items[["name", "attributes", "category"]].isna().any().any():
        raise RuntimeError("source cards contain null name/attributes/category")
    category_values = items["category"].astype(str).str.strip()
    name_values = items["name"].astype(str).str.strip()
    if category_values.eq("").any() or name_values.eq("").any():
        raise RuntimeError("source cards contain empty names or categories")
    forbidden = set(category_values) & FROZEN_OOD_CATEGORIES
    if forbidden:
        raise RuntimeError(
            f"source leaks frozen OOD categories: {sorted(forbidden)}"
        )
    parsed_attributes: list[dict[str, Any]] = []
    for index, raw in enumerate(items["attributes"].tolist()):
        try:
            attributes = parse_attributes(raw)
        except Exception as error:
            raise RuntimeError(f"invalid attributes at item row {index}") from error
        if not attributes:
            raise RuntimeError(f"empty attributes at item row {index}")
        parsed_attributes.append(attributes)

    category_by_id = items.set_index("id")["category"].astype(str)
    left_category = pairs["id1"].map(category_by_id)
    right_category = pairs["id2"].map(category_by_id)
    if not left_category.equals(right_category):
        raise RuntimeError("source contains cross-category pairs")

    card_keys = [canonical_card(row) for _, row in items.iterrows()]
    card_counts = Counter(card_keys)
    duplicate_card_count = sum(count - 1 for count in card_counts.values())
    if duplicate_card_count:
        raise RuntimeError(
            "source contains category-agnostic duplicate cards: "
            f"duplicates={duplicate_card_count:,}"
        )
    card_key_by_id = dict(zip(items["id"], card_keys, strict=True))
    identical_pair_cards = sum(
        card_key_by_id[int(row.id1)] == card_key_by_id[int(row.id2)]
        for row in pairs.itertuples(index=False)
    )
    if identical_pair_cards:
        raise RuntimeError(
            f"source contains {identical_pair_cards:,} canonically identical pairs"
        )

    provenance_contract = _verify_hardened_provenance(
        dataset_kind=dataset_kind,
        source_dir=source_dir,
        pair_count=pair_count,
        items_filename=items_filename,
        pairs_filename=pairs_filename,
        metadata_filename=metadata_filename,
        items=items,
        metadata=metadata,
        documents=documents,
    )

    source_fingerprint_payload = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "dataset_kind": dataset_kind,
        "pairs": pair_count,
        "targets": {"0": 0, "1": pair_count},
        "source_files": file_records,
        "provenance_documents": [
            {key: document[key] for key in ("name", "bytes", "sha256")}
            for document in documents
        ],
        "canonical_card_multiset_sha256": canonical_sha256(sorted(card_keys)),
        "category_counts": {
            str(key): int(value)
            for key, value in left_category.value_counts().sort_index().items()
        },
    }
    source_fingerprint = canonical_sha256(source_fingerprint_payload)
    validation = {
        **source_fingerprint_payload,
        "source_fingerprint": source_fingerprint,
        "valid": True,
        "items": len(items),
        "metadata_rows": len(metadata),
        "unique_item_ids": int(items["id"].nunique()),
        "unique_card_keys": len(card_counts),
        "unique_pair_endpoints": int(endpoints.nunique()),
        "identical_pair_cards": identical_pair_cards,
        "duplicate_card_count": duplicate_card_count,
        "forbidden_ood_categories": [],
        "provenance_contract": provenance_contract,
    }
    return {
        "paths": paths,
        "items": items,
        "pairs": pairs,
        "metadata": metadata,
        "documents": documents,
        "provenance_path_by_name": provenance_path_by_name,
        "file_records": file_records,
        "provenance_contract": provenance_contract,
        "validation": validation,
        "source_fingerprint": source_fingerprint,
    }


def reindex_source(checked: dict[str, Any], *, dataset_kind: str) -> dict[str, Any]:
    pairs: pd.DataFrame = checked["pairs"]
    items: pd.DataFrame = checked["items"]
    metadata: pd.DataFrame = checked["metadata"]
    count = len(pairs)
    final_id = SYNTHETIC_ID_START - (count * 2 - 1)
    if final_id < -(2**63):
        raise RuntimeError("synthetic reindexing exceeds signed int64")

    item_by_id = items.set_index("id", drop=False)
    metadata_by_pair = {
        (int(row["id1"]), int(row["id2"])): row
        for row in metadata.to_dict("records")
    }
    output_items: list[dict[str, Any]] = []
    output_pairs: list[dict[str, Any]] = []
    output_metadata: list[dict[str, Any]] = []
    source_to_output: dict[int, int] = {}
    for pair_index, pair in enumerate(pairs.itertuples(index=False)):
        source_id1, source_id2 = int(pair.id1), int(pair.id2)
        id1 = SYNTHETIC_ID_START - pair_index * 2
        id2 = id1 - 1
        source_to_output[source_id1] = id1
        source_to_output[source_id2] = id2
        for source_id, output_id in ((source_id1, id1), (source_id2, id2)):
            item = item_by_id.loc[source_id].to_dict()
            item["id"] = output_id
            output_items.append(item)
        output_pairs.append({"id1": id1, "id2": id2, "target": 1})
        source_metadata = dict(metadata_by_pair[(source_id1, source_id2)])
        source_metadata.update(
            {
                "ablation_pair_index": pair_index,
                "ablation_source_id1": source_id1,
                "ablation_source_id2": source_id2,
                "id1": id1,
                "id2": id2,
                "target": 1,
                "dataset_kind": dataset_kind,
                "source_fingerprint": checked["source_fingerprint"],
            }
        )
        output_metadata.append(source_metadata)

    reindexed_items = pd.DataFrame(output_items)
    reindexed_pairs = pd.DataFrame(output_pairs).astype(
        {"id1": "int64", "id2": "int64", "target": "int8"}
    )
    reindexed_metadata = pd.DataFrame(output_metadata)
    reindexed_metadata["id1"] = reindexed_metadata["id1"].astype("int64")
    reindexed_metadata["id2"] = reindexed_metadata["id2"].astype("int64")
    reindexed_metadata["target"] = reindexed_metadata["target"].astype("int8")
    reindexed_items["id"] = reindexed_items["id"].astype("int64")
    if (
        reindexed_items["id"].duplicated().any()
        or not reindexed_items["id"].lt(0).all()
        or set(reindexed_items["id"])
        != set(reindexed_pairs["id1"]) | set(reindexed_pairs["id2"])
    ):
        raise RuntimeError("deterministic synthetic ID reindexing failed")
    return {
        "items": reindexed_items,
        "pairs": reindexed_pairs,
        "metadata": reindexed_metadata,
        "provenance": {
            "version": ID_REINDEX_VERSION,
            "start": SYNTHETIC_ID_START,
            "end": final_id,
            "pairs": count,
            "items": count * 2,
            "source_id_to_output_id_sha256": canonical_sha256(
                sorted(source_to_output.items())
            ),
        },
    }


def prepare_upload_payload(
    checked: dict[str, Any],
    *,
    stage_dir: Path,
    owner: str,
    dataset_slug: str,
    artifact_tag: str,
    label_source: str,
    dataset_kind: str,
) -> dict[str, Any]:
    if not label_source or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", label_source):
        raise ValueError("label_source must contain lowercase letters, digits, _ or -")
    expected_label_source = (checked.get("provenance_contract") or {}).get(
        "label_source"
    )
    if expected_label_source and label_source != expected_label_source:
        raise ValueError(
            "label_source differs from the pinned source provenance: "
            f"{label_source!r} != {expected_label_source!r}"
        )
    dataset_slug = kaggle.validate_slug(dataset_slug, "dataset slug")
    filenames = upload_helpers.artifact_filenames(artifact_tag)
    reindexed = reindex_source(checked, dataset_kind=dataset_kind)
    pairs = reindexed["pairs"].copy()
    pairs["label_source"] = label_source
    items = reindexed["items"].copy()
    items["product_text"] = items.apply(serialize_product, axis=1)
    if items["product_text"].astype(str).str.strip().eq("").any():
        raise RuntimeError("serialized synthetic cards contain empty product_text")
    metadata = reindexed["metadata"]

    stage_dir.mkdir(parents=True, exist_ok=True)
    pair_path = stage_dir / filenames["pairs"]
    item_path = stage_dir / filenames["items"]
    metadata_path = stage_dir / filenames["metadata"]
    upload_helpers.atomic_parquet(
        pairs[["id1", "id2", "target", "label_source"]], pair_path
    )
    upload_helpers.atomic_parquet(
        items[["id", "name", "category", "product_text"]], item_path
    )
    upload_helpers.atomic_parquet(metadata, metadata_path)

    source_provenance = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "dataset_kind": dataset_kind,
        "source_fingerprint": checked["source_fingerprint"],
        "source_files": checked["file_records"],
        "provenance_documents": checked["documents"],
        "verified_contract": checked.get("provenance_contract") or {},
        "id_reindexing": reindexed["provenance"],
    }
    atomic_json(source_provenance, stage_dir / PROVENANCE_FILENAME)
    atomic_json(checked["validation"], stage_dir / VALIDATION_FILENAME)

    staged_names = [
        filenames["pairs"],
        filenames["items"],
        filenames["metadata"],
        PROVENANCE_FILENAME,
        VALIDATION_FILENAME,
    ]
    files = {
        name: {
            "bytes": (stage_dir / name).stat().st_size,
            "sha256": sha256_file(stage_dir / name),
        }
        for name in staged_names
    }
    pair_count = len(pairs)
    dataset_ref = f"{owner}/{dataset_slug}"
    manifest = {
        "schema_version": UPLOAD_SCHEMA_VERSION,
        "dataset": dataset_ref,
        "is_private": True,
        "generation_kind": dataset_kind,
        "pairs": pair_count,
        "items": pair_count * 2,
        "label_source": label_source,
        "targets": {"0": 0, "1": pair_count},
        "sample_weight": 1.0,
        "source_provenance": {
            "schema_version": SOURCE_SCHEMA_VERSION,
            "source_fingerprint": checked["source_fingerprint"],
            "source_files": checked["file_records"],
            "provenance_documents": [
                {key: document[key] for key in ("name", "bytes", "sha256")}
                for document in checked["documents"]
            ],
            "verified_contract": checked.get("provenance_contract") or {},
            "id_reindexing": reindexed["provenance"],
            "validation_sha256": sha256_file(stage_dir / VALIDATION_FILENAME),
        },
        "files": files,
    }
    atomic_json(manifest, stage_dir / "upload_manifest.json")
    metadata_json = {
        "title": f"Product matching {dataset_kind.replace('_', ' ')} positives",
        "id": dataset_ref,
        "licenses": [{"name": "unknown"}],
        "isPrivate": True,
        "description": (
            f"Private E-CUP 2026 data ablation: {pair_count:,} validated "
            f"{dataset_kind.replace('_', ' ')} target=1 pairs."
        ),
    }
    atomic_json(metadata_json, stage_dir / "dataset-metadata.json")
    expected_names = set(staged_names) | {"upload_manifest.json"}
    actual_names = {
        path.name
        for path in stage_dir.iterdir()
        if path.is_file() and path.name != "dataset-metadata.json"
    }
    if actual_names != expected_names:
        raise RuntimeError(
            f"staged payload file set differs: {sorted(actual_names)} != "
            f"{sorted(expected_names)}"
        )
    for name, initial in checked["file_records"].items():
        current = _file_record(checked["paths"][name])
        if current != initial:
            raise RuntimeError(f"source file changed while staging: {name}")
    for document in checked["documents"]:
        name = document["name"]
        current = _file_record(checked["provenance_path_by_name"][name])
        expected = {key: document[key] for key in ("bytes", "sha256")}
        if current != expected:
            raise RuntimeError(
                f"provenance document changed while staging: {name}"
            )
    return manifest


def upload_dataset(
    stage_dir: Path,
    manifest: dict[str, Any],
    *,
    message: str | None,
) -> None:
    cli = kaggle.kaggle_command()
    dataset_ref = str(manifest["dataset"])
    previous = shared_push.dataset_status(cli, dataset_ref)
    previous_version = int(previous.get("current_version_number", 0)) if previous else 0
    if previous is None:
        command = cli + [
            "datasets",
            "create",
            "--path",
            str(stage_dir),
            "--keep-tabular",
        ]
    else:
        upload_helpers.verify_remote_dataset(cli, dataset_ref, set())
        command = cli + [
            "datasets",
            "version",
            "--path",
            str(stage_dir),
            "--message",
            message
            or (
                f"Add {manifest['pairs']:,} {manifest['generation_kind']} "
                "positive pairs"
            ),
            "--keep-tabular",
        ]
    kaggle.run_command(command)
    shared_push.wait_until_ready(
        cli, dataset_ref, minimum_version=previous_version + 1
    )
    upload_helpers.verify_remote_dataset(
        cli, dataset_ref, set(manifest["files"]) | {"upload_manifest.json"}
    )


def default_notebook_path(experiment_label: str) -> Path:
    return ROOT / (
        "notebooks/minilm_5ep_team_ablation/"
        f"{experiment_label}_2xt4.ipynb"
    )


def default_report_path(experiment_label: str) -> Path:
    return ROOT / "reports" / f"{experiment_label}_launcher.json"


def build_notes(
    *,
    dataset_kind: str,
    pair_count: int,
    dataset_ref: str,
    upload_manifest_sha256: str,
    source_fingerprint: str,
    extra_notes: str | None,
    provenance_contract: dict[str, Any] | None = None,
) -> str:
    description = (
        f"Frozen MiniLM 5ep human baseline plus {pair_count:,} validated "
        f"{dataset_kind.replace('_', ' ')} synthetic pairs, all target=1 with unit "
        "sample weight. The launcher's exact canonical-card duplicate guard passes "
        "within this snapshot, every item is used once, IDs are deterministically "
        "reindexed into a synthetic-only int64 namespace, and frozen OOD categories "
        "are excluded. Frozen checkpoint, "
        "one-epoch recipe and IID/hard/OOD validation are unchanged; paired component "
        "permutation, component bootstrap and Holm correction use the shared baseline. "
        f"Source dataset {dataset_ref}. Upload manifest SHA-256 "
        f"{upload_manifest_sha256}. Source fingerprint {source_fingerprint}."
    )
    if dataset_kind == "human_skeleton_ab":
        clone_diagnostics = (provenance_contract or {}).get(
            "fact_clone_diagnostics"
        )
        if (
            not isinstance(clone_diagnostics, dict)
            or clone_diagnostics.get("nonblocking") is not True
        ):
            raise RuntimeError(
                "human-skeleton notes require verified nonblocking clone diagnostics"
            )
        all_diagnostics = clone_diagnostics.get("all") or {}
        card_diagnostics = all_diagnostics.get("cards") or {}
        pair_diagnostics = all_diagnostics.get("pairs") or {}
        description += (
            " This canonical uniqueness guard is not a factual-novelty claim. "
            "Punctuation-insensitive source-clone diagnostics are explicitly "
            "nonblocking and report "
            f"{int(card_diagnostics['fact_identical_to_human_positive']):,}/"
            f"{int(card_diagnostics['total']):,} cards and "
            f"{int(pair_diagnostics['fact_identical_to_human_positive']):,}/"
            f"{int(pair_diagnostics['total']):,} pairs as fact-identical to human "
            "positives."
        )
        overlap_filter = (provenance_contract or {}).get(
            "validation_overlap_filter"
        )
        if (
            not isinstance(overlap_filter, dict)
            or overlap_filter.get("postfilter_overlapping_card_count") != 0
        ):
            raise RuntimeError(
                "human-skeleton notes require a verified zero-overlap filter"
            )
        description += (
            " The signed frozen-validation fact filter excluded "
            f"{int(overlap_filter['dropped_pair_count']):,} of "
            f"{int(overlap_filter['source_task_count']):,} source tasks and the "
            "independent post-filter replay found zero overlapping cards."
        )
    if extra_notes:
        description += f" {extra_notes.strip()}"
    return description


def generate_notebook(
    *,
    notebook: Path,
    pair_count: int,
    artifact_tag: str,
    experiment_label: str,
    dataset_ref: str,
    upload_manifest_sha256: str,
    label_source: str,
    notes: str,
) -> None:
    run(
        [
            sys.executable,
            "scripts/create_mixed_generation_rule_10k_notebook.py",
            "--pair-count",
            str(pair_count),
            "--expected-target0",
            "0",
            "--expected-target1",
            str(pair_count),
            "--artifact-tag",
            artifact_tag,
            "--output",
            str(notebook),
            "--experiment-label",
            experiment_label,
            "--dataset-ref",
            dataset_ref,
            "--upload-manifest-sha256",
            upload_manifest_sha256,
            "--label-source",
            label_source,
            "--notes",
            notes,
        ]
    )
    verify_notebook(
        notebook,
        pair_count=pair_count,
        experiment_label=experiment_label,
        dataset_ref=dataset_ref,
        manifest_sha256=upload_manifest_sha256,
        label_source=label_source,
    )


def verify_notebook(
    notebook_path: Path,
    *,
    pair_count: int,
    experiment_label: str,
    dataset_ref: str,
    manifest_sha256: str,
    label_source: str,
) -> None:
    notebook = nbformat.read(notebook_path, as_version=4)
    data_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code"
        and "data-hook" in cell.get("metadata", {}).get("tags", [])
    ]
    routing_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code"
        and "experiment-routing" in cell.get("metadata", {}).get("tags", [])
    ]
    if len(data_cells) != 1 or len(routing_cells) != 1:
        raise RuntimeError("generated notebook lacks its unique data/routing cells")
    data_source = data_cells[0].source
    routing_source = routing_cells[0].source
    required_data_fragments = (
        repr(dataset_ref),
        repr(manifest_sha256),
        repr(label_source),
        f"upload_manifest.get(\"pairs\") != {pair_count}",
        f"expected_targets = {{'0': 0, '1': {pair_count}}}",
    )
    if any(fragment not in data_source for fragment in required_data_fragments):
        raise RuntimeError("generated notebook data hook is not exactly pinned")
    if (
        f"EXPERIMENT_LABEL = {experiment_label!r}" not in routing_source
        or "EXPERIMENT_SHEET = 'data_exps'" not in routing_source
    ):
        raise RuntimeError("generated notebook is not routed to data_exps")
    frozen_source = "\n".join(
        cell.source
        for cell in notebook.cells
        if "frozen" in cell.get("metadata", {}).get("tags", [])
    )
    if (
        frozen_notebook.SIGNIFICANCE_BASELINE_RUN_ID not in frozen_source
        or "paired_component_permutation" not in frozen_source
        or not all(repr(split) in frozen_source for split in EXPECTED_SPLITS)
        or "google_sheets_sync.json" not in frozen_source
    ):
        raise RuntimeError(
            "generated notebook lost frozen baseline/significance/Sheets machinery"
        )
    nbformat.validate(notebook)


def notebook_command(
    *,
    notebook: Path,
    env_file: Path,
    kernel_slug: str,
    title: str,
    dataset_ref: str,
) -> list[str]:
    return [
        sys.executable,
        "scripts/run_kaggle_notebook.py",
        str(notebook),
        "--env-file",
        str(env_file),
        "--slug",
        kernel_slug,
        "--title",
        title,
        "--dataset",
        VALIDATION_DATASET,
        "--dataset",
        CHECKPOINT_DATASET,
        "--dataset",
        SIGNIFICANCE_DATASET,
        "--dataset",
        dataset_ref,
        "--no-env-sources",
    ]


def monitor_existing_kernel(owner: str, slug: str) -> Path:
    cli = kaggle.kaggle_command()
    kernel_ref = f"{owner}/{slug}"
    kaggle.wait_for_kernel(
        cli,
        kernel_ref,
        poll_interval=kaggle.env_int("KAGGLE_POLL_INTERVAL_SECONDS", 30, minimum=5),
        wait_timeout=kaggle.env_int(
            "KAGGLE_WAIT_TIMEOUT_SECONDS", 45_000, minimum=60
        ),
    )
    output_root = Path(
        os.getenv("KAGGLE_OUTPUT_DIR", "") or ROOT / "artifacts/kaggle"
    ).expanduser()
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_dir = output_root.resolve() / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    kaggle.run_command(
        cli
        + [
            "kernels",
            "output",
            kernel_ref,
            "-p",
            str(output_dir),
            "--force",
            "--page-size",
            "200",
        ]
    )
    return output_dir


def output_directory(kernel_slug: str) -> Path:
    output_root = Path(
        os.getenv("KAGGLE_OUTPUT_DIR", "") or ROOT / "artifacts/kaggle"
    ).expanduser()
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    return output_root.resolve() / kernel_slug


def verify_completion(
    output_dir: Path,
    *,
    experiment_label: str,
    dataset_ref: str,
    upload_manifest_sha256: str,
    source_fingerprint: str,
    pair_count: int,
    label_source: str,
) -> dict[str, Any]:
    completion = read_json(output_dir / "notebook_completed.json", "completion marker")
    sync = read_json(output_dir / "google_sheets_sync.json", "Sheets sync marker")
    comparison = read_json(
        output_dir / "baseline_comparison.json", "baseline comparison"
    )
    embedded_training_report = completion.get("training_report")
    if not isinstance(embedded_training_report, dict):
        raise RuntimeError("completion marker has no embedded training report")
    training_report_paths = list(output_dir.glob("**/training_report.json"))
    if len(training_report_paths) != 1:
        raise RuntimeError(
            "expected exactly one downloaded training_report.json, got "
            f"{[str(path) for path in training_report_paths]}"
        )
    training_report = read_json(training_report_paths[0], "training report")
    if training_report != embedded_training_report:
        raise RuntimeError("downloaded and embedded training reports differ")
    run_id = str(completion.get("run_id") or "")
    if (
        completion.get("status") != "complete"
        or completion.get("experiment") != experiment_label
        or completion.get("experiment_group") != "data"
        or not re.fullmatch(r"[0-9a-f]{32}", run_id)
    ):
        raise RuntimeError("Kaggle completion marker contract failed")
    if (
        sync.get("status") != "synced"
        or sync.get("run_id") != run_id
        or sync.get("experiment_group") != "data"
        or sync.get("comparison_sheet") != "data_exps"
    ):
        raise RuntimeError("mandatory data_exps synchronization did not complete")
    if (
        comparison.get("status") != "ready"
        or comparison.get("baseline_run_id")
        != frozen_notebook.SIGNIFICANCE_BASELINE_RUN_ID
        or comparison.get("candidate_run_id") != run_id
        or set((comparison.get("splits") or {})) != EXPECTED_SPLITS
    ):
        raise RuntimeError("paired baseline comparison contract failed")
    for split, values in comparison["splits"].items():
        required = {
            "baseline_macro_average_precision",
            "candidate_macro_average_precision",
            "delta_macro_average_precision",
            "p_value",
            "p_value_holm",
            "ci95_low",
            "ci95_high",
        }
        if required - set(values):
            raise RuntimeError(f"baseline comparison lacks fields for {split}")
    validation_splits = training_report.get("validation_splits") or {}
    if set(validation_splits) != EXPECTED_SPLITS:
        raise RuntimeError("training report does not contain exact IID/hard/OOD splits")
    label_counts = (completion.get("train_data") or {}).get(
        "label_source_counts"
    ) or {}
    if int(label_counts.get(label_source, -1)) != pair_count:
        raise RuntimeError("Kaggle train data has the wrong synthetic count")
    notes = str(completion.get("notes") or "")
    for pin in (dataset_ref, upload_manifest_sha256, source_fingerprint):
        if pin not in notes:
            raise RuntimeError(f"Kaggle completion notes do not pin {pin}")
    return {
        "run_id": run_id,
        "completion": completion,
        "sync": sync,
        "comparison": comparison,
        "training_report": training_report,
    }


def write_report(path: Path, result: dict[str, Any]) -> Path:
    atomic_json(
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            **result,
        },
        path,
    )
    return path


def main() -> None:
    args = parse_args()
    source_dir = absolute(args.source_dir)
    env_file = absolute(args.env_file)
    notebook = absolute(
        args.notebook or default_notebook_path(args.experiment_label)
    )
    report_path = absolute(args.report or default_report_path(args.experiment_label))
    if args.pair_count < 1:
        raise ValueError("--pair-count must be positive")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.experiment_label):
        raise ValueError("--experiment-label must be a lowercase slug")
    checked = verify_source(
        source_dir,
        pair_count=args.pair_count,
        dataset_kind=args.dataset_kind,
        provenance_paths=args.provenance_file,
        items_filename=args.items_file,
        pairs_filename=args.pairs_file,
        metadata_filename=args.metadata_file,
    )
    label_source = (
        args.label_source
        or (checked.get("provenance_contract") or {}).get("label_source")
        or f"{args.dataset_kind}_positive_v1"
    )

    kaggle.load_dotenv(env_file)
    owner = kaggle.validate_slug(
        os.getenv("KAGGLE_USERNAME", "").strip(), "KAGGLE_USERNAME"
    )
    if not kaggle.env_bool("KAGGLE_IS_PRIVATE", True):
        kaggle.fail("positive synthetic ablations must use a private Kaggle notebook")
    if not kaggle.env_bool("KAGGLE_ENABLE_INTERNET", True):
        kaggle.fail("Kaggle internet must be enabled for mandatory data_exps sync")
    if not args.dry_run_only and not os.getenv("KAGGLE_API_TOKEN", "").strip():
        kaggle.fail("set KAGGLE_API_TOKEN in .env")
    stage_dir = absolute(
        args.stage_dir
        or (ROOT / ".kaggle" / "datasets" / args.dataset_slug)
    )
    manifest = prepare_upload_payload(
        checked,
        stage_dir=stage_dir,
        owner=owner,
        dataset_slug=args.dataset_slug,
        artifact_tag=args.artifact_tag,
        label_source=label_source,
        dataset_kind=args.dataset_kind,
    )
    dataset_ref = str(manifest["dataset"])
    upload_manifest_sha = sha256_file(stage_dir / "upload_manifest.json")
    notes = build_notes(
        dataset_kind=args.dataset_kind,
        pair_count=args.pair_count,
        dataset_ref=dataset_ref,
        upload_manifest_sha256=upload_manifest_sha,
        source_fingerprint=checked["source_fingerprint"],
        extra_notes=args.notes,
        provenance_contract=checked.get("provenance_contract") or {},
    )
    generate_notebook(
        notebook=notebook,
        pair_count=args.pair_count,
        artifact_tag=args.artifact_tag,
        experiment_label=args.experiment_label,
        dataset_ref=dataset_ref,
        upload_manifest_sha256=upload_manifest_sha,
        label_source=label_source,
        notes=notes,
    )
    command = notebook_command(
        notebook=notebook,
        env_file=env_file,
        kernel_slug=args.kernel_slug,
        title=args.title
        or f"MiniLM 5ep: {args.dataset_kind.replace('_', ' ')} positives",
        dataset_ref=dataset_ref,
    )
    run(command + ["--dry-run"])
    if (
        not args.dry_run_only
        and not args.skip_dataset_upload
        and not args.monitor_existing
    ):
        upload_dataset(stage_dir, manifest, message=args.dataset_message)

    common_result = {
        "dataset_kind": args.dataset_kind,
        "dataset_ref": dataset_ref,
        "pair_count": args.pair_count,
        "target_counts": {"0": 0, "1": args.pair_count},
        "label_source": label_source,
        "source_fingerprint": checked["source_fingerprint"],
        "source_contract": checked.get("provenance_contract") or {},
        "upload_manifest_sha256": upload_manifest_sha,
        "notebook": str(notebook),
        "kernel_slug": args.kernel_slug,
        "experiment": args.experiment_label,
    }
    if args.dry_run_only:
        result = {"status": "dry_run_complete", **common_result}
    else:
        if args.monitor_existing:
            output_dir = monitor_existing_kernel(owner, args.kernel_slug)
        else:
            run(command)
            output_dir = output_directory(args.kernel_slug)
        completed = verify_completion(
            output_dir,
            experiment_label=args.experiment_label,
            dataset_ref=dataset_ref,
            upload_manifest_sha256=upload_manifest_sha,
            source_fingerprint=checked["source_fingerprint"],
            pair_count=args.pair_count,
            label_source=label_source,
        )
        result = {
            "status": "complete",
            **common_result,
            "run_id": completed["run_id"],
            "baseline_comparison": completed["comparison"],
            "google_sheets_sync": completed["sync"],
        }
    report = write_report(report_path, result)
    print(
        json.dumps(
            {**result, "launcher_report": str(report)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
