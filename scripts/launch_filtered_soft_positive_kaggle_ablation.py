#!/usr/bin/env python3
"""Validate, publish and evaluate weighted filtered soft-positive pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from item_pipeline.normalization import parse_attributes
from src.data_pipeline import serialize_product

import create_filtered_soft_positive_notebook as notebook_builder
import create_minilm_5ep_team_ablation_notebook as frozen_notebook
import launch_positive_synthetic_kaggle_ablation as base


CONTRACT_VERSION = "soft_positive_quality_hardness_filter_v1"
LABEL_SOURCE = "qwen_soft_positive_ab_quality_hardness_v1"
WEIGHT_STATS_VERSION = "float64_le_pair_order_v1"
SOURCE_SCHEMA_VERSION = "filtered_soft_positive_source_snapshot_v1"
UPLOAD_SCHEMA_VERSION = 1
CANDIDATE_COUNT = 31_176
DUPLICATE_SOURCE_ROW_COUNT = 181
DOC_FILENAMES = (
    "summary.json",
    "selection_report.json",
    "validation_report.json",
    "build_manifest.json",
)
PARQUET_FILENAMES = (
    "items.parquet",
    "pairs.parquet",
    "pair_generation_metadata.parquet",
    "candidate_decisions.parquet",
)
BUILD_MANIFEST_FILE_SET = set(PARQUET_FILENAMES) | set(DOC_FILENAMES[:-1])
FROZEN_OOD_CATEGORIES = set(base.FROZEN_OOD_CATEGORIES)
SYNTHETIC_ID_START = base.SYNTHETIC_ID_START
ID_REINDEX_VERSION = "filtered_soft_positive_negative_int64_v1"
FILTER_BUILDER_PATH = ROOT / "scripts/filter_soft_positive_ab_pairs.py"
SOURCE_PROVENANCE_FILENAME = "filtered_soft_positive_source_provenance.json"
SOURCE_VALIDATION_FILENAME = "filtered_soft_positive_validation.json"
REQUIRED_METADATA_COLUMNS = {
    "id1",
    "id2",
    "target",
    "candidate_key",
    "baseline_score",
    "score_ab",
    "score_ba",
    "score_order_gap",
    "quality_score",
    "sample_weight",
    "source_tier",
    "source_run_signature",
    "source_task_index",
    "category",
    "rule_id",
    "concept",
    "relation",
    "product_type",
    "required_attribute_key",
    "original_value",
    "new_value",
    "rule_probability",
    "rule_support",
    "rule_singleton_support",
    "human_category_original_support",
    "human_category_new_support",
    "human_scope_original_support",
    "human_scope_new_support",
    "evidence_type",
    "evidence_source",
    "evidence_value",
    "selection_rank",
}
REQUIRED_DECISION_COLUMNS = {
    "candidate_key",
    "source_tier",
    "source_run_signature",
    "source_task_index",
    "category",
    "product_type",
    "rule_id",
    "concept",
    "relation",
    "required_attribute_key",
    "original_value",
    "new_value",
    "rule_probability",
    "rule_support",
    "rule_singleton_support",
    "human_category_original_support",
    "human_category_new_support",
    "human_scope_original_support",
    "human_scope_new_support",
    "selected",
    "rejection_reasons",
    "duplicate_source_keys_json",
    "baseline_score",
    "score_ab",
    "score_ba",
    "score_order_gap",
    "quality_score",
    "evidence_type",
    "evidence_source",
    "evidence_value",
}
EVIDENCE_TYPES = {
    "source_exact_transition",
    "source_endpoint_values",
    "human_scope_attribute_values",
    "human_category_attribute_values",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--pair-count", type=int, required=True)
    parser.add_argument("--dataset-slug", required=True)
    parser.add_argument("--artifact-tag", required=True)
    parser.add_argument("--experiment-label", required=True)
    parser.add_argument("--kernel-slug", required=True)
    parser.add_argument("--title")
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--stage-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--notes")
    parser.add_argument("--dataset-message")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--skip-dataset-upload", action="store_true")
    parser.add_argument("--monitor-existing", action="store_true")
    parser.add_argument(
        "--finalize-local-output",
        action="store_true",
        help=(
            "validate an already downloaded completion against the local source "
            "and upload manifest, then write status=complete without Kaggle calls"
        ),
    )
    return parser.parse_args()


def _required_columns(
    frame: pd.DataFrame, required: set[str], *, description: str
) -> None:
    if missing := required - set(frame.columns):
        raise RuntimeError(f"{description} lacks columns: {sorted(missing)}")


def _nonempty_text(frame: pd.DataFrame, columns: set[str], *, description: str) -> None:
    for column in sorted(columns):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise RuntimeError(f"{description}.{column} must be nonempty")


def _finite_column(
    frame: pd.DataFrame,
    column: str,
    *,
    description: str,
    lower: float | None = None,
    upper: float | None = None,
) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{description}.{column} must be finite numeric")
    if lower is not None and (values < lower).any():
        raise RuntimeError(f"{description}.{column} is below {lower}")
    if upper is not None and (values > upper).any():
        raise RuntimeError(f"{description}.{column} exceeds {upper}")
    return values


def _nonnegative_int_column(
    frame: pd.DataFrame, column: str, *, description: str
) -> pd.Series:
    values = base.exact_int_series(
        frame[column], field=f"{description}.{column}"
    )
    if values.lt(0).any():
        raise RuntimeError(f"{description}.{column} must be nonnegative")
    return values


def sample_weight_statistics(values: pd.Series) -> dict[str, Any]:
    weights = pd.to_numeric(values, errors="coerce").to_numpy(dtype="<f8")
    if not len(weights):
        raise RuntimeError("sample weights must not be empty")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise RuntimeError("sample weights must be finite and positive")
    return {
        "version": WEIGHT_STATS_VERSION,
        "count": int(len(weights)),
        "sum": float(math.fsum(float(value) for value in weights)),
        "min": float(weights.min()),
        "max": float(weights.max()),
        "sha256": hashlib.sha256(weights.tobytes(order="C")).hexdigest(),
    }


def _canonical_json_array(
    value: Any, *, field: str, row: int, require_nonempty: bool | None
) -> list[str]:
    if not isinstance(value, str):
        raise RuntimeError(f"{field} must be a canonical JSON array string at row {row}")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{field} is invalid JSON at row {row}") from error
    if (
        not isinstance(parsed, list)
        or any(not isinstance(item, str) or not item.strip() for item in parsed)
        or len(parsed) != len(set(parsed))
    ):
        raise RuntimeError(f"{field} must contain unique nonempty strings at row {row}")
    canonical = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    if value != canonical:
        raise RuntimeError(f"{field} is not canonically serialized at row {row}")
    if require_nonempty is not None and require_nonempty != bool(parsed):
        expectation = "nonempty" if require_nonempty else "empty"
        raise RuntimeError(f"{field} must be {expectation} at row {row}")
    return parsed


def _exact_common_document_contract(
    payload: dict[str, Any], *, name: str, pair_count: int, weight_stats: dict[str, Any]
) -> str:
    expected = {
        "contract_version": CONTRACT_VERSION,
        "label_source": LABEL_SOURCE,
        "pair_count": pair_count,
        "target_counts": {"0": 0, "1": pair_count},
        "candidate_count": CANDIDATE_COUNT,
        "selected_count": pair_count,
        "rejected_count": CANDIDATE_COUNT - pair_count,
        "sample_weight_stats": weight_stats,
        "validation_fact_overlap_count": 0,
        "forbidden_ood_categories": [],
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise RuntimeError(
                f"{name}.{field} differs: {payload.get(field)!r} != {value!r}"
            )
    signature = payload.get("run_signature")
    if not isinstance(signature, str) or not re.fullmatch(r"[0-9a-f]{64}", signature):
        raise RuntimeError(f"{name}.run_signature must be a lowercase SHA-256")
    return signature


def _verify_build_manifest_files(
    manifest: dict[str, Any], source_dir: Path
) -> None:
    records = manifest.get("files")
    if not isinstance(records, dict) or set(records) != BUILD_MANIFEST_FILE_SET:
        raise RuntimeError(
            "build_manifest files differ: "
            f"{sorted(records) if isinstance(records, dict) else records!r} != "
            f"{sorted(BUILD_MANIFEST_FILE_SET)}"
        )
    for filename in sorted(BUILD_MANIFEST_FILE_SET):
        expected = records.get(filename)
        actual = base._file_record(source_dir / filename)
        if expected != actual:
            raise RuntimeError(
                f"build_manifest file record differs for {filename}: "
                f"{expected!r} != {actual!r}"
            )


def _verify_signed_source_provenance(
    *,
    summary: dict[str, Any],
    validation: dict[str, Any],
    manifest: dict[str, Any],
    run_signature: str,
    metadata: pd.DataFrame,
    weight_stats: dict[str, Any],
) -> dict[str, Any]:
    signature_payload = manifest.get("signature_payload")
    if not isinstance(signature_payload, dict) or not signature_payload:
        raise RuntimeError("build_manifest.signature_payload must be a nonempty object")
    if summary.get("signature_payload") != signature_payload:
        raise RuntimeError("summary and build manifest signature_payload differ")
    if base.canonical_sha256(signature_payload) != run_signature:
        raise RuntimeError("run_signature does not authenticate signature_payload")
    if (
        signature_payload.get("contract_version") != CONTRACT_VERSION
        or signature_payload.get("label_source") != LABEL_SOURCE
        or signature_payload.get("sample_weight_stats") != weight_stats
    ):
        raise RuntimeError("signature_payload core contract differs")
    if not FILTER_BUILDER_PATH.is_file():
        raise RuntimeError(f"filtered source builder is missing: {FILTER_BUILDER_PATH}")
    if signature_payload.get("builder_sha256") != base.sha256_file(FILTER_BUILDER_PATH):
        raise RuntimeError("signature_payload builder SHA differs from local builder")
    expected_selected_sha = base.canonical_sha256(
        metadata["candidate_key"].astype(str).tolist()
    )
    if signature_payload.get("selected_candidate_keys_sha256") != expected_selected_sha:
        raise RuntimeError("signature_payload selected candidate keys differ")

    input_provenance = signature_payload.get("inputs")
    validation_reference = signature_payload.get("validation_reference")
    if not isinstance(input_provenance, dict) or not input_provenance:
        raise RuntimeError("signature_payload inputs are missing")
    if not isinstance(validation_reference, dict) or not validation_reference:
        raise RuntimeError("signature_payload validation reference is missing")
    if (
        summary.get("input_provenance") != input_provenance
        or manifest.get("input_provenance") != input_provenance
    ):
        raise RuntimeError("summary/manifest input provenance differs from signature")
    if any(
        payload.get("validation_reference") != validation_reference
        for payload in (summary, validation, manifest)
    ):
        raise RuntimeError(
            "summary/validation/manifest validation reference differs from signature"
        )
    return {
        "signature_payload_sha256": run_signature,
        "builder_sha256": signature_payload["builder_sha256"],
        "selected_candidate_keys_sha256": expected_selected_sha,
        "input_provenance": input_provenance,
        "validation_reference": validation_reference,
    }


def _verify_candidate_decisions(
    decisions: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    pair_count: int,
) -> dict[str, Any]:
    _required_columns(
        decisions, REQUIRED_DECISION_COLUMNS, description="candidate decisions"
    )
    if len(decisions) != CANDIDATE_COUNT:
        raise RuntimeError(
            f"candidate decisions rows differ: {len(decisions)} != {CANDIDATE_COUNT}"
        )
    _nonempty_text(
        decisions,
        {
            "candidate_key",
            "source_tier",
            "source_run_signature",
            "category",
            "product_type",
            "rule_id",
            "concept",
            "relation",
            "required_attribute_key",
            "original_value",
            "new_value",
            "evidence_type",
            "evidence_source",
            "evidence_value",
        },
        description="candidate decisions",
    )
    if decisions["candidate_key"].duplicated().any():
        raise RuntimeError("candidate decisions candidate_key is not unique")
    if not pd.api.types.is_bool_dtype(decisions["selected"].dtype):
        raise RuntimeError("candidate decisions selected must have boolean dtype")
    if decisions["selected"].isna().any():
        raise RuntimeError("candidate decisions selected contains nulls")
    selected = decisions["selected"].astype(bool)
    if int(selected.sum()) != pair_count:
        raise RuntimeError("candidate decisions selected count differs")
    if set(decisions["source_tier"].astype(str)) - {"A", "B"}:
        raise RuntimeError("candidate decisions source_tier must be A or B")
    if not decisions["source_run_signature"].astype(str).str.fullmatch(
        r"[0-9a-f]{64}"
    ).all():
        raise RuntimeError(
            "candidate decisions source_run_signature must be lowercase SHA-256"
        )
    decision_tasks = _nonnegative_int_column(
        decisions, "source_task_index", description="candidate decisions"
    )
    decision_identity = list(
        zip(
            decisions["source_run_signature"].astype(str),
            decision_tasks,
            strict=True,
        )
    )
    if len(decision_identity) != len(set(decision_identity)):
        raise RuntimeError("candidate decisions source identity is not unique")
    _finite_column(
        decisions,
        "rule_probability",
        description="candidate decisions",
        lower=0.0,
        upper=1.0,
    )
    rule_support = _nonnegative_int_column(
        decisions, "rule_support", description="candidate decisions"
    )
    singleton_support = _nonnegative_int_column(
        decisions, "rule_singleton_support", description="candidate decisions"
    )
    if rule_support.eq(0).any() or (singleton_support > rule_support).any():
        raise RuntimeError("candidate decisions rule support contract failed")
    for column in (
        "human_category_original_support",
        "human_category_new_support",
        "human_scope_original_support",
        "human_scope_new_support",
    ):
        _nonnegative_int_column(decisions, column, description="candidate decisions")
    if set(decisions["evidence_type"].astype(str)) - EVIDENCE_TYPES:
        raise RuntimeError("candidate decisions evidence_type is unsupported")

    duplicate_keys: list[str] = []
    for row, (is_selected, reasons, duplicates) in enumerate(
        zip(
            selected.tolist(),
            decisions["rejection_reasons"].tolist(),
            decisions["duplicate_source_keys_json"].tolist(),
            strict=True,
        )
    ):
        _canonical_json_array(
            reasons,
            field="candidate decisions.rejection_reasons",
            row=row,
            require_nonempty=not is_selected,
        )
        duplicate_keys.extend(
            _canonical_json_array(
                duplicates,
                field="candidate decisions.duplicate_source_keys_json",
                row=row,
                require_nonempty=None,
            )
            if duplicates != "[]"
            else []
        )
    if (
        len(duplicate_keys) != DUPLICATE_SOURCE_ROW_COUNT
        or len(duplicate_keys) != len(set(duplicate_keys))
    ):
        raise RuntimeError(
            "candidate decisions do not preserve exactly 181 unique secondary "
            "duplicate source keys"
        )

    arrays = {
        column: _finite_column(
            decisions,
            column,
            description="candidate decisions",
            lower=0.0,
            upper=1.0,
        )
        for column in (
            "baseline_score",
            "score_ab",
            "score_ba",
            "score_order_gap",
            "quality_score",
        )
    }
    if not np.allclose(
        arrays["baseline_score"],
        (arrays["score_ab"] + arrays["score_ba"]) / 2.0,
        rtol=0.0,
        atol=1e-7,
    ):
        raise RuntimeError("candidate decisions baseline_score is not directional mean")
    if not np.allclose(
        arrays["score_order_gap"],
        np.abs(arrays["score_ab"] - arrays["score_ba"]),
        rtol=0.0,
        atol=1e-7,
    ):
        raise RuntimeError("candidate decisions score_order_gap differs")

    selected_decisions = decisions.loc[selected].set_index("candidate_key")
    selected_metadata = metadata.set_index("candidate_key")
    if set(selected_decisions.index) != set(selected_metadata.index):
        raise RuntimeError("selected candidate keys differ from generation metadata")
    compare_columns = (
        "source_tier",
        "source_run_signature",
        "source_task_index",
        "category",
        "product_type",
        "rule_id",
        "concept",
        "relation",
        "required_attribute_key",
        "original_value",
        "new_value",
        "rule_probability",
        "rule_support",
        "rule_singleton_support",
        "human_category_original_support",
        "human_category_new_support",
        "human_scope_original_support",
        "human_scope_new_support",
        "baseline_score",
        "score_ab",
        "score_ba",
        "score_order_gap",
        "quality_score",
        "evidence_type",
        "evidence_source",
        "evidence_value",
    )
    selected_decisions = selected_decisions.sort_index()
    selected_metadata = selected_metadata.sort_index()
    for column in compare_columns:
        left = selected_decisions[column]
        right = selected_metadata[column]
        if column in {
            "rule_probability",
            "baseline_score",
            "score_ab",
            "score_ba",
            "score_order_gap",
            "quality_score",
        }:
            if not np.allclose(
                pd.to_numeric(left, errors="coerce"),
                pd.to_numeric(right, errors="coerce"),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    f"selected candidate decisions differ from metadata: {column}"
                )
        elif not left.astype(str).equals(right.astype(str)):
            raise RuntimeError(
                f"selected candidate decisions differ from metadata: {column}"
            )
    return {
        "candidate_count": len(decisions),
        "selected_count": int(selected.sum()),
        "rejected_count": int((~selected).sum()),
        "duplicate_secondary_source_keys": len(duplicate_keys),
    }


def verify_source(source_dir: Path, *, pair_count: int) -> dict[str, Any]:
    if pair_count < 1 or pair_count > CANDIDATE_COUNT:
        raise ValueError(f"pair_count must be in [1, {CANDIDATE_COUNT}]")
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        raise RuntimeError(f"source directory is missing: {source_dir}")
    actual_docs = {path.name for path in source_dir.glob("*.json") if path.is_file()}
    if actual_docs != set(DOC_FILENAMES):
        raise RuntimeError(
            f"source JSON documents differ: {sorted(actual_docs)} != "
            f"{sorted(DOC_FILENAMES)}"
        )
    actual_parquets = {
        path.name for path in source_dir.glob("*.parquet") if path.is_file()
    }
    if actual_parquets != set(PARQUET_FILENAMES):
        raise RuntimeError(
            f"source Parquet files differ: {sorted(actual_parquets)} != "
            f"{sorted(PARQUET_FILENAMES)}"
        )

    source_file_records = {
        name: base._file_record(source_dir / name)
        for name in PARQUET_FILENAMES
    }
    documents = base.load_provenance_documents(
        [source_dir / name for name in DOC_FILENAMES]
    )
    payload_by_name = {
        document["name"]: document["payload"] for document in documents
    }

    items = pd.read_parquet(source_dir / "items.parquet")
    pairs = pd.read_parquet(source_dir / "pairs.parquet")
    metadata = pd.read_parquet(source_dir / "pair_generation_metadata.parquet")
    decisions = pd.read_parquet(source_dir / "candidate_decisions.parquet")
    _required_columns(
        items, {"id", "name", "attributes", "category"}, description="items"
    )
    _required_columns(
        pairs,
        {"id1", "id2", "target", "sample_weight", "label_source"},
        description="pairs",
    )
    _required_columns(metadata, REQUIRED_METADATA_COLUMNS, description="metadata")
    if len(pairs) != pair_count or len(items) != pair_count * 2 or len(metadata) != pair_count:
        raise RuntimeError(
            "source dimensions differ: "
            f"pairs={len(pairs)}, items={len(items)}, metadata={len(metadata)}"
        )

    items = items.copy()
    pairs = pairs.copy()
    metadata = metadata.copy()
    items["id"] = base.exact_int_series(items["id"], field="items.id")
    for frame, label in ((pairs, "pairs"), (metadata, "metadata")):
        frame["id1"] = base.exact_int_series(frame["id1"], field=f"{label}.id1")
        frame["id2"] = base.exact_int_series(frame["id2"], field=f"{label}.id2")
        base._target_one(frame, description=label)
        frame["target"] = 1
    weight_stats = sample_weight_statistics(pairs["sample_weight"])
    metadata_weight_stats = sample_weight_statistics(metadata["sample_weight"])
    if metadata_weight_stats != weight_stats:
        raise RuntimeError("metadata sample weights differ from pairs or pair order")
    if not np.array_equal(
        pd.to_numeric(pairs["sample_weight"]).to_numpy(dtype="<f8"),
        pd.to_numeric(metadata["sample_weight"]).to_numpy(dtype="<f8"),
    ):
        raise RuntimeError("metadata sample weights are not aligned with pair order")
    if set(pairs["label_source"].astype(str)) != {LABEL_SOURCE}:
        raise RuntimeError("source pairs label_source differs from the fixed contract")

    if items["id"].duplicated().any():
        raise RuntimeError("source item IDs are not unique")
    if pairs["id1"].eq(pairs["id2"]).any():
        raise RuntimeError("source contains self-pairs")
    endpoints = pd.concat([pairs["id1"], pairs["id2"]], ignore_index=True)
    if endpoints.duplicated().any() or set(endpoints) != set(items["id"]):
        raise RuntimeError("every source card must occur in exactly one pair")
    unordered = [
        tuple(sorted((int(row.id1), int(row.id2))))
        for row in pairs.itertuples(index=False)
    ]
    if len(unordered) != len(set(unordered)):
        raise RuntimeError("source contains duplicate unordered pairs")
    oriented_pairs = list(zip(pairs["id1"], pairs["id2"], strict=True))
    oriented_metadata = list(zip(metadata["id1"], metadata["id2"], strict=True))
    if len(set(oriented_metadata)) != pair_count or oriented_metadata != oriented_pairs:
        raise RuntimeError("metadata must be pair-order aligned one-to-one")

    _nonempty_text(
        metadata,
        {
            "candidate_key",
            "source_tier",
            "source_run_signature",
            "category",
            "rule_id",
            "concept",
            "relation",
            "product_type",
            "required_attribute_key",
            "original_value",
            "new_value",
            "evidence_type",
            "evidence_source",
            "evidence_value",
        },
        description="metadata",
    )
    if metadata["candidate_key"].duplicated().any():
        raise RuntimeError("metadata candidate_key is not unique")
    if set(metadata["source_tier"].astype(str)) - {"A", "B"}:
        raise RuntimeError("metadata source_tier must be A or B")
    if not metadata["source_run_signature"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise RuntimeError("metadata source_run_signature must be lowercase SHA-256")
    source_task = base.exact_int_series(
        metadata["source_task_index"], field="metadata.source_task_index"
    )
    if source_task.lt(0).any():
        raise RuntimeError("metadata source_task_index must be nonnegative")
    source_identity = list(
        zip(metadata["source_run_signature"].astype(str), source_task, strict=True)
    )
    if len(source_identity) != len(set(source_identity)):
        raise RuntimeError("metadata source run/task identity is not unique")
    _finite_column(
        metadata,
        "rule_probability",
        description="metadata",
        lower=0.0,
        upper=1.0,
    )
    rule_support = _nonnegative_int_column(
        metadata, "rule_support", description="metadata"
    )
    singleton_support = _nonnegative_int_column(
        metadata, "rule_singleton_support", description="metadata"
    )
    if rule_support.eq(0).any() or (singleton_support > rule_support).any():
        raise RuntimeError("metadata rule support contract failed")
    for column in (
        "human_category_original_support",
        "human_category_new_support",
        "human_scope_original_support",
        "human_scope_new_support",
    ):
        _nonnegative_int_column(metadata, column, description="metadata")
    if set(metadata["evidence_type"].astype(str)) - EVIDENCE_TYPES:
        raise RuntimeError("metadata evidence_type is unsupported")
    scope_evidence = metadata["evidence_type"].eq("human_scope_attribute_values")
    if scope_evidence.any() and (
        metadata.loc[scope_evidence, "human_scope_original_support"].astype(int).le(0).any()
        or metadata.loc[scope_evidence, "human_scope_new_support"].astype(int).le(0).any()
    ):
        raise RuntimeError("metadata human-scope evidence lacks positive grounding")
    category_evidence = metadata["evidence_type"].eq(
        "human_category_attribute_values"
    )
    if category_evidence.any() and (
        metadata.loc[
            category_evidence, "human_category_original_support"
        ].astype(int).le(0).any()
        or metadata.loc[
            category_evidence, "human_category_new_support"
        ].astype(int).le(0).any()
    ):
        raise RuntimeError("metadata human-category evidence lacks positive grounding")
    ranks = base.exact_int_series(metadata["selection_rank"], field="metadata.selection_rank")
    if set(ranks) != set(range(1, pair_count + 1)):
        raise RuntimeError("metadata selection_rank must be exactly 1..pair_count")
    numeric = {
        column: _finite_column(
            metadata,
            column,
            description="metadata",
            lower=0.0,
            upper=1.0,
        )
        for column in (
            "baseline_score",
            "score_ab",
            "score_ba",
            "score_order_gap",
            "quality_score",
        )
    }
    if not np.allclose(
        numeric["baseline_score"],
        (numeric["score_ab"] + numeric["score_ba"]) / 2.0,
        rtol=0.0,
        atol=1e-7,
    ):
        raise RuntimeError("metadata baseline_score is not directional mean")
    if not np.allclose(
        numeric["score_order_gap"],
        np.abs(numeric["score_ab"] - numeric["score_ba"]),
        rtol=0.0,
        atol=1e-7,
    ):
        raise RuntimeError("metadata score_order_gap differs")

    if items[["name", "attributes", "category"]].isna().any().any():
        raise RuntimeError("source cards contain null text/category")
    _nonempty_text(items, {"name", "category"}, description="items")
    for row, raw in enumerate(items["attributes"].tolist()):
        try:
            parsed = parse_attributes(raw)
        except Exception as error:
            raise RuntimeError(f"invalid attributes at item row {row}") from error
        if not parsed:
            raise RuntimeError(f"empty attributes at item row {row}")
    categories = items["category"].astype(str).str.strip()
    forbidden = set(categories) & FROZEN_OOD_CATEGORIES
    if forbidden:
        raise RuntimeError(f"source leaks frozen OOD categories: {sorted(forbidden)}")
    category_by_id = items.set_index("id")["category"].astype(str)
    left_category = pairs["id1"].map(category_by_id)
    if not left_category.equals(pairs["id2"].map(category_by_id)):
        raise RuntimeError("source contains cross-category pairs")
    if not left_category.reset_index(drop=True).equals(
        metadata["category"].astype(str).reset_index(drop=True)
    ):
        raise RuntimeError("metadata category differs from source cards")
    card_keys = [base.canonical_card(row) for _, row in items.iterrows()]
    if len(card_keys) != len(set(card_keys)):
        raise RuntimeError("source contains category-agnostic duplicate cards")
    card_by_id = dict(zip(items["id"], card_keys, strict=True))
    if any(
        card_by_id[int(row.id1)] == card_by_id[int(row.id2)]
        for row in pairs.itertuples(index=False)
    ):
        raise RuntimeError("source contains canonically identical within-pair cards")

    candidate_contract = _verify_candidate_decisions(
        decisions, metadata, pair_count=pair_count
    )
    signatures = {
        _exact_common_document_contract(
            payload_by_name[name],
            name=name,
            pair_count=pair_count,
            weight_stats=weight_stats,
        )
        for name in DOC_FILENAMES
    }
    if len(signatures) != 1:
        raise RuntimeError("provenance run_signature differs across documents")
    run_signature = signatures.pop()
    summary = payload_by_name["summary.json"]
    selection = payload_by_name["selection_report.json"]
    validation = payload_by_name["validation_report.json"]
    manifest = payload_by_name["build_manifest.json"]
    if (
        summary.get("generated_pairs") != pair_count
        or summary.get("generated_items") != pair_count * 2
        or summary.get("metadata_rows") != pair_count
    ):
        raise RuntimeError("summary dimensions differ")
    if (
        selection.get("valid") is not True
        or selection.get("selected_pairs") != pair_count
    ):
        raise RuntimeError("selection report is not valid/exact")
    if (
        validation.get("valid") is not True
        or validation.get("pairs") != pair_count
        or validation.get("items") != pair_count * 2
        or validation.get("metadata_rows") != pair_count
        or validation.get("one_use_unique_cards") is not True
    ):
        raise RuntimeError("validation report is not valid/exact")
    if manifest.get("valid") is not True:
        raise RuntimeError("build manifest is not valid")
    _verify_build_manifest_files(manifest, source_dir)
    signed_provenance = _verify_signed_source_provenance(
        summary=summary,
        validation=validation,
        manifest=manifest,
        run_signature=run_signature,
        metadata=metadata,
        weight_stats=weight_stats,
    )

    document_records = {
        document["name"]: {
            "bytes": document["bytes"],
            "sha256": document["sha256"],
        }
        for document in documents
    }
    fingerprint_payload = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "run_signature": run_signature,
        "pairs": pair_count,
        "items": pair_count * 2,
        "targets": {"0": 0, "1": pair_count},
        "sample_weight_stats": weight_stats,
        "candidate_decisions": candidate_contract,
        "signed_provenance": signed_provenance,
        "source_files": source_file_records,
        "provenance_documents": document_records,
        "canonical_card_multiset_sha256": base.canonical_sha256(sorted(card_keys)),
        "category_counts": {
            str(key): int(value)
            for key, value in left_category.value_counts().sort_index().items()
        },
    }
    source_fingerprint = base.canonical_sha256(fingerprint_payload)
    checked_validation = {
        **fingerprint_payload,
        "source_fingerprint": source_fingerprint,
        "valid": True,
        "metadata_rows": len(metadata),
        "unique_item_ids": int(items["id"].nunique()),
        "unique_pair_endpoints": int(endpoints.nunique()),
        "unique_card_keys": len(set(card_keys)),
        "validation_fact_overlap_count": 0,
        "forbidden_ood_categories": [],
    }
    return {
        "source_dir": source_dir,
        "paths": {name: source_dir / name for name in PARQUET_FILENAMES},
        "items": items,
        "pairs": pairs,
        "metadata": metadata,
        "decisions": decisions,
        "documents": documents,
        "source_file_records": source_file_records,
        "run_signature": run_signature,
        "sample_weight_stats": weight_stats,
        "source_fingerprint": source_fingerprint,
        "validation": checked_validation,
    }


def reindex_source(checked: dict[str, Any]) -> dict[str, Any]:
    pairs: pd.DataFrame = checked["pairs"]
    items: pd.DataFrame = checked["items"]
    metadata: pd.DataFrame = checked["metadata"]
    count = len(pairs)
    final_id = SYNTHETIC_ID_START - (count * 2 - 1)
    if final_id < -(2**63):
        raise RuntimeError("synthetic reindexing exceeds signed int64")
    item_by_id = items.set_index("id", drop=False)
    output_items: list[dict[str, Any]] = []
    output_pairs: list[dict[str, Any]] = []
    output_metadata: list[dict[str, Any]] = []
    source_to_output: dict[int, int] = {}
    for pair_index, (pair, source_metadata) in enumerate(
        zip(pairs.to_dict("records"), metadata.to_dict("records"), strict=True)
    ):
        source_id1, source_id2 = int(pair["id1"]), int(pair["id2"])
        id1 = SYNTHETIC_ID_START - pair_index * 2
        id2 = id1 - 1
        source_to_output[source_id1] = id1
        source_to_output[source_id2] = id2
        for source_id, output_id in ((source_id1, id1), (source_id2, id2)):
            item = item_by_id.loc[source_id].to_dict()
            item["id"] = output_id
            output_items.append(item)
        weight = float(pair["sample_weight"])
        output_pairs.append(
            {"id1": id1, "id2": id2, "target": 1, "sample_weight": weight}
        )
        output_metadata.append(
            {
                **source_metadata,
                "ablation_pair_index": pair_index,
                "ablation_source_id1": source_id1,
                "ablation_source_id2": source_id2,
                "id1": id1,
                "id2": id2,
                "target": 1,
                "sample_weight": weight,
                "source_fingerprint": checked["source_fingerprint"],
            }
        )
    output = {
        "items": pd.DataFrame(output_items),
        "pairs": pd.DataFrame(output_pairs),
        "metadata": pd.DataFrame(output_metadata),
        "provenance": {
            "version": ID_REINDEX_VERSION,
            "start": SYNTHETIC_ID_START,
            "end": final_id,
            "pairs": count,
            "items": count * 2,
            "source_id_to_output_id_sha256": base.canonical_sha256(
                sorted(source_to_output.items())
            ),
        },
    }
    output["items"]["id"] = output["items"]["id"].astype("int64")
    for frame in (output["pairs"], output["metadata"]):
        frame["id1"] = frame["id1"].astype("int64")
        frame["id2"] = frame["id2"].astype("int64")
        frame["target"] = frame["target"].astype("int8")
        frame["sample_weight"] = frame["sample_weight"].astype("float64")
    if sample_weight_statistics(output["pairs"]["sample_weight"]) != checked["sample_weight_stats"]:
        raise RuntimeError("deterministic reindexing changed sample weights")
    return output


def prepare_upload_payload(
    checked: dict[str, Any],
    *,
    stage_dir: Path,
    owner: str,
    dataset_slug: str,
    artifact_tag: str,
) -> dict[str, Any]:
    owner = base.kaggle.validate_slug(owner, "dataset owner")
    dataset_slug = base.kaggle.validate_slug(dataset_slug, "dataset slug")
    filenames = base.upload_helpers.artifact_filenames(artifact_tag)
    reindexed = reindex_source(checked)
    pairs = reindexed["pairs"].copy()
    pairs["label_source"] = LABEL_SOURCE
    items = reindexed["items"].copy()
    items["product_text"] = items.apply(serialize_product, axis=1)
    if items["product_text"].astype(str).str.strip().eq("").any():
        raise RuntimeError("serialized synthetic cards contain empty product_text")
    metadata = reindexed["metadata"].copy()
    metadata["label_source"] = LABEL_SOURCE

    stage_dir.mkdir(parents=True, exist_ok=True)
    pair_path = stage_dir / filenames["pairs"]
    item_path = stage_dir / filenames["items"]
    metadata_path = stage_dir / filenames["metadata"]
    base.upload_helpers.atomic_parquet(
        pairs[["id1", "id2", "target", "sample_weight", "label_source"]],
        pair_path,
    )
    base.upload_helpers.atomic_parquet(
        items[["id", "name", "category", "product_text"]], item_path
    )
    base.upload_helpers.atomic_parquet(metadata, metadata_path)
    staged_weight_stats = sample_weight_statistics(
        pd.read_parquet(pair_path, columns=["sample_weight"])["sample_weight"]
    )
    if staged_weight_stats != checked["sample_weight_stats"]:
        raise RuntimeError("staged pair sample weights differ from source")

    source_provenance = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "run_signature": checked["run_signature"],
        "source_fingerprint": checked["source_fingerprint"],
        "source_files": checked["source_file_records"],
        "provenance_documents": {
            document["name"]: {
                "bytes": document["bytes"],
                "sha256": document["sha256"],
            }
            for document in checked["documents"]
        },
        "candidate_decisions_in_train_payload": False,
        "id_reindexing": reindexed["provenance"],
    }
    base.atomic_json(source_provenance, stage_dir / SOURCE_PROVENANCE_FILENAME)
    base.atomic_json(checked["validation"], stage_dir / SOURCE_VALIDATION_FILENAME)
    staged_names = [
        filenames["pairs"],
        filenames["items"],
        filenames["metadata"],
        SOURCE_PROVENANCE_FILENAME,
        SOURCE_VALIDATION_FILENAME,
    ]
    files = {
        name: base._file_record(stage_dir / name) for name in staged_names
    }
    dataset_ref = f"{owner}/{dataset_slug}"
    manifest = {
        "schema_version": UPLOAD_SCHEMA_VERSION,
        "dataset": dataset_ref,
        "is_private": True,
        "generation_kind": CONTRACT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "pairs": len(pairs),
        "items": len(items),
        "label_source": LABEL_SOURCE,
        "targets": {"0": 0, "1": len(pairs)},
        "sample_weight_stats": staged_weight_stats,
        "source_provenance": source_provenance,
        "files": files,
    }
    base.atomic_json(manifest, stage_dir / "upload_manifest.json")
    base.atomic_json(
        {
            "title": "Product matching filtered soft positives",
            "id": dataset_ref,
            "licenses": [{"name": "unknown"}],
            "isPrivate": True,
            "description": (
                f"Private E-CUP 2026 weighted data ablation: {len(pairs):,} "
                "quality/hardness-filtered target=1 pairs."
            ),
        },
        stage_dir / "dataset-metadata.json",
    )
    expected_payload_names = set(staged_names) | {"upload_manifest.json"}
    actual_payload_names = {
        path.name
        for path in stage_dir.iterdir()
        if path.is_file() and path.name != "dataset-metadata.json"
    }
    if actual_payload_names != expected_payload_names:
        raise RuntimeError(
            f"staged payload files differ: {sorted(actual_payload_names)} != "
            f"{sorted(expected_payload_names)}"
        )
    for filename, expected in checked["source_file_records"].items():
        if base._file_record(checked["source_dir"] / filename) != expected:
            raise RuntimeError(f"source file changed while staging: {filename}")
    for document in checked["documents"]:
        expected = {key: document[key] for key in ("bytes", "sha256")}
        if base._file_record(checked["source_dir"] / document["name"]) != expected:
            raise RuntimeError(
                f"provenance document changed while staging: {document['name']}"
            )
    return manifest


def verify_local_upload_payload(
    checked: dict[str, Any],
    *,
    stage_dir: Path,
    dataset_slug: str,
    artifact_tag: str,
) -> tuple[dict[str, Any], str]:
    """Verify an existing staged Dataset without rewriting or contacting Kaggle."""

    dataset_slug = base.kaggle.validate_slug(dataset_slug, "dataset slug")
    filenames = base.upload_helpers.artifact_filenames(artifact_tag)
    manifest_path = stage_dir / "upload_manifest.json"
    manifest = base.read_json(manifest_path, "local upload manifest")
    dataset_ref = str(manifest.get("dataset") or "")
    parts = dataset_ref.split("/", 1)
    if (
        len(parts) != 2
        or parts[1] != dataset_slug
        or base.kaggle.validate_slug(parts[0], "dataset owner") != parts[0]
    ):
        raise RuntimeError("local upload manifest Dataset reference differs")
    expected_core = {
        "is_private": True,
        "generation_kind": CONTRACT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "pairs": len(checked["pairs"]),
        "items": len(checked["items"]),
        "label_source": LABEL_SOURCE,
        "targets": {"0": 0, "1": len(checked["pairs"])},
        "sample_weight_stats": checked["sample_weight_stats"],
    }
    for field, expected in expected_core.items():
        if manifest.get(field) != expected:
            raise RuntimeError(
                f"local upload manifest {field} differs: "
                f"{manifest.get(field)!r} != {expected!r}"
            )
    provenance = manifest.get("source_provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("local upload manifest source provenance is missing")
    expected_documents = {
        document["name"]: {
            "bytes": document["bytes"],
            "sha256": document["sha256"],
        }
        for document in checked["documents"]
    }
    if (
        provenance.get("contract_version") != CONTRACT_VERSION
        or provenance.get("run_signature") != checked["run_signature"]
        or provenance.get("source_fingerprint") != checked["source_fingerprint"]
        or provenance.get("source_files") != checked["source_file_records"]
        or provenance.get("provenance_documents") != expected_documents
        or provenance.get("candidate_decisions_in_train_payload") is not False
    ):
        raise RuntimeError("local upload manifest source provenance differs")
    expected_files = {
        filenames["pairs"],
        filenames["items"],
        filenames["metadata"],
        SOURCE_PROVENANCE_FILENAME,
        SOURCE_VALIDATION_FILENAME,
    }
    records = manifest.get("files")
    if not isinstance(records, dict) or set(records) != expected_files:
        raise RuntimeError("local upload manifest file set differs")
    for filename in sorted(expected_files):
        if records[filename] != base._file_record(stage_dir / filename):
            raise RuntimeError(f"local staged file differs: {filename}")
    return manifest, base.sha256_file(manifest_path)


def default_notebook_path(experiment_label: str) -> Path:
    return ROOT / "notebooks/minilm_5ep_team_ablation" / f"{experiment_label}_2xt4.ipynb"


def default_report_path(experiment_label: str) -> Path:
    return ROOT / "reports" / f"{experiment_label}_launcher.json"


def build_notes(
    *,
    pair_count: int,
    dataset_ref: str,
    upload_manifest_sha256: str,
    source_fingerprint: str,
    weight_stats: dict[str, Any],
    extra_notes: str | None,
) -> str:
    notes = (
        f"Frozen MiniLM 5ep human baseline plus {pair_count:,} validated "
        "quality/hardness-filtered Qwen soft-positive pairs, all target=1. "
        "Synthetic sample weights are preserved exactly and human sample weight "
        "is one. Every synthetic card is unique and used once; frozen OOD "
        "categories and frozen-validation fact overlaps are absent. Frozen "
        "checkpoint, recipe and IID/hard/OOD paired significance are unchanged. "
        f"Source dataset {dataset_ref}. Upload manifest SHA-256 "
        f"{upload_manifest_sha256}. Source fingerprint {source_fingerprint}. "
        f"Source weight SHA-256 {weight_stats['sha256']}."
    )
    if extra_notes:
        notes += f" {extra_notes.strip()}"
    return notes


def generate_notebook(
    *,
    notebook: Path,
    pair_count: int,
    artifact_tag: str,
    experiment_label: str,
    dataset_ref: str,
    upload_manifest_sha256: str,
    sample_weight_stats: dict[str, Any],
    notes: str,
) -> None:
    generated = notebook_builder.build_notebook(
        pair_count=pair_count,
        artifact_tag=artifact_tag,
        experiment_label=experiment_label,
        dataset_ref=dataset_ref,
        upload_manifest_sha256=upload_manifest_sha256,
        sample_weight_stats=sample_weight_stats,
        notes=notes,
    )
    notebook.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(generated, notebook)
    verify_notebook(
        notebook,
        pair_count=pair_count,
        experiment_label=experiment_label,
        dataset_ref=dataset_ref,
        manifest_sha256=upload_manifest_sha256,
        sample_weight_stats=sample_weight_stats,
    )


def verify_notebook(
    notebook_path: Path,
    *,
    pair_count: int,
    experiment_label: str,
    dataset_ref: str,
    manifest_sha256: str,
    sample_weight_stats: dict[str, Any],
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
        raise RuntimeError("generated notebook lacks unique data/routing cells")
    data_source = data_cells[0].source
    routing_source = routing_cells[0].source
    required_data = (
        repr(dataset_ref),
        repr(manifest_sha256),
        repr(LABEL_SOURCE),
        repr(CONTRACT_VERSION),
        repr(WEIGHT_STATS_VERSION),
        repr(sample_weight_stats["sha256"]),
        f'upload_manifest.get("pairs") != {pair_count}',
        'extra_pairs["sample_weight"] = pd.to_numeric',
    )
    if any(fragment not in data_source for fragment in required_data):
        raise RuntimeError("generated notebook data hook is not exactly pinned")
    if 'extra_pairs["sample_weight"] = 1.0' in data_source:
        raise RuntimeError("generated notebook resets synthetic sample weights")
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
        frozen_notebook.CHECKPOINT_MANIFEST_SHA256 not in frozen_source
        or frozen_notebook.SIGNIFICANCE_BASELINE_RUN_ID not in frozen_source
        or "paired_component_permutation" not in frozen_source
        or not all(repr(split) in frozen_source for split in base.EXPECTED_SPLITS)
        or "google_sheets_sync.json" not in frozen_source
    ):
        raise RuntimeError(
            "generated notebook lost frozen checkpoint/significance/Sheets machinery"
        )
    nbformat.validate(notebook)


def verify_weighted_completion(
    completed: dict[str, Any],
    *,
    pair_count: int,
    weight_stats: dict[str, Any],
    sample_weights: pd.Series,
) -> None:
    training = completed["training_report"]
    if (
        training.get("training_sampling") != "none"
        or training.get("training_loss_weighting") != "none"
    ):
        raise RuntimeError("weighted completion requires the locked unweighted recipe")
    counts = training.get("training_source_counts") or {}
    masses = training.get("training_source_weight_mass") or {}
    if int(counts.get(LABEL_SOURCE, -1)) != pair_count:
        raise RuntimeError("training report synthetic source count differs")
    human_count = counts.get("human")
    if isinstance(human_count, bool) or not isinstance(human_count, int) or human_count < 1:
        raise RuntimeError("training report human source count differs")
    if sample_weight_statistics(sample_weights) != weight_stats:
        raise RuntimeError("completion source sample weights differ from provenance")
    synthetic_weights = pd.to_numeric(
        sample_weights, errors="coerce"
    ).to_numpy(dtype=np.float32)
    if len(synthetic_weights) != pair_count:
        raise RuntimeError("completion source sample weight count differs")

    # Replay train_cross_encoder exactly. build_training_loss_weights(mode="none")
    # returns float32 ones; data weights are loaded as float32, multiplied
    # in-place, and normalized with a float32 mean.  Reconstructing the factor
    # from the float64 manifest sum is measurably different for 308k rows and
    # caused the completed 1,400-pair run to be rejected by the verifier.
    replayed_weights = np.concatenate(
        [np.ones(human_count, dtype=np.float32), synthetic_weights]
    )
    replayed_weights /= replayed_weights.mean()
    expected_human_mass = float(replayed_weights[:human_count].sum())
    expected_synthetic_mass = float(replayed_weights[human_count:].sum())
    observed_mass = masses.get(LABEL_SOURCE)
    if (
        isinstance(observed_mass, bool)
        or not isinstance(observed_mass, (int, float))
        or not math.isfinite(float(observed_mass))
        or not math.isclose(
            float(observed_mass),
            expected_synthetic_mass,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise RuntimeError("training report synthetic weight mass differs")
    human_mass = masses.get("human")
    if (
        isinstance(human_mass, bool)
        or not isinstance(human_mass, (int, float))
        or not math.isclose(
            float(human_mass),
            expected_human_mass,
            rel_tol=0.0,
            abs_tol=1e-5,
        )
    ):
        raise RuntimeError("human pre-normalization sample weight is not one")
    expected_min = float(replayed_weights.min())
    expected_max = float(replayed_weights.max())
    if not math.isclose(
        float(training.get("training_loss_weight_min", math.nan)),
        expected_min,
        rel_tol=0.0,
        abs_tol=1e-7,
    ) or not math.isclose(
        float(training.get("training_loss_weight_max", math.nan)),
        expected_max,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise RuntimeError("training report sample weight range differs")


def main() -> None:
    args = parse_args()
    source_dir = base.absolute(args.source_dir)
    env_file = base.absolute(args.env_file)
    notebook = base.absolute(
        args.notebook or default_notebook_path(args.experiment_label)
    )
    report_path = base.absolute(
        args.report or default_report_path(args.experiment_label)
    )
    if args.pair_count < 1:
        raise ValueError("--pair-count must be positive")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.experiment_label):
        raise ValueError("--experiment-label must be a lowercase slug")

    # Fail closed on local data before reading credentials or staging anything.
    checked = verify_source(source_dir, pair_count=args.pair_count)
    stage_dir = base.absolute(
        args.stage_dir or (ROOT / ".kaggle" / "datasets" / args.dataset_slug)
    )
    if args.finalize_local_output:
        if args.dry_run_only or args.skip_dataset_upload or args.monitor_existing:
            raise ValueError(
                "--finalize-local-output cannot be combined with launch-mode flags"
            )
        manifest, upload_manifest_sha = verify_local_upload_payload(
            checked,
            stage_dir=stage_dir,
            dataset_slug=args.dataset_slug,
            artifact_tag=args.artifact_tag,
        )
        dataset_ref = str(manifest["dataset"])
        verify_notebook(
            notebook,
            pair_count=args.pair_count,
            experiment_label=args.experiment_label,
            dataset_ref=dataset_ref,
            manifest_sha256=upload_manifest_sha,
            sample_weight_stats=checked["sample_weight_stats"],
        )
        completed = base.verify_completion(
            base.output_directory(args.kernel_slug),
            experiment_label=args.experiment_label,
            dataset_ref=dataset_ref,
            upload_manifest_sha256=upload_manifest_sha,
            source_fingerprint=checked["source_fingerprint"],
            pair_count=args.pair_count,
            label_source=LABEL_SOURCE,
        )
        verify_weighted_completion(
            completed,
            pair_count=args.pair_count,
            weight_stats=checked["sample_weight_stats"],
            sample_weights=checked["pairs"]["sample_weight"],
        )
        common = {
            "contract_version": CONTRACT_VERSION,
            "dataset_ref": dataset_ref,
            "pair_count": args.pair_count,
            "target_counts": {"0": 0, "1": args.pair_count},
            "label_source": LABEL_SOURCE,
            "sample_weight_stats": checked["sample_weight_stats"],
            "source_fingerprint": checked["source_fingerprint"],
            "source_run_signature": checked["run_signature"],
            "upload_manifest_sha256": upload_manifest_sha,
            "notebook": str(notebook),
            "kernel_slug": args.kernel_slug,
            "experiment": args.experiment_label,
        }
        if report_path.is_file():
            previous = base.read_json(report_path, "existing launcher report")
            for field in (
                "dataset_ref",
                "pair_count",
                "label_source",
                "source_fingerprint",
                "source_run_signature",
                "upload_manifest_sha256",
                "kernel_slug",
                "experiment",
            ):
                if previous.get(field) != common[field]:
                    raise RuntimeError(
                        f"existing launcher report {field} differs; refusing overwrite"
                    )
        result = {
            "status": "complete",
            **common,
            "run_id": completed["run_id"],
            "baseline_comparison": completed["comparison"],
            "google_sheets_sync": completed["sync"],
            "finalized_from_local_output": True,
        }
        report = base.write_report(report_path, result)
        print(
            json.dumps(
                {**result, "launcher_report": str(report)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    base.kaggle.load_dotenv(env_file)
    owner = base.kaggle.validate_slug(
        os.getenv("KAGGLE_USERNAME", "").strip(), "KAGGLE_USERNAME"
    )
    if not base.kaggle.env_bool("KAGGLE_IS_PRIVATE", True):
        base.kaggle.fail("filtered synthetic ablations must use a private notebook")
    if not base.kaggle.env_bool("KAGGLE_ENABLE_INTERNET", True):
        base.kaggle.fail("Kaggle internet must be enabled for mandatory data_exps sync")
    if not args.dry_run_only and not os.getenv("KAGGLE_API_TOKEN", "").strip():
        base.kaggle.fail("set KAGGLE_API_TOKEN in .env")
    manifest = prepare_upload_payload(
        checked,
        stage_dir=stage_dir,
        owner=owner,
        dataset_slug=args.dataset_slug,
        artifact_tag=args.artifact_tag,
    )
    dataset_ref = str(manifest["dataset"])
    upload_manifest_sha = base.sha256_file(stage_dir / "upload_manifest.json")
    notes = build_notes(
        pair_count=args.pair_count,
        dataset_ref=dataset_ref,
        upload_manifest_sha256=upload_manifest_sha,
        source_fingerprint=checked["source_fingerprint"],
        weight_stats=checked["sample_weight_stats"],
        extra_notes=args.notes,
    )
    generate_notebook(
        notebook=notebook,
        pair_count=args.pair_count,
        artifact_tag=args.artifact_tag,
        experiment_label=args.experiment_label,
        dataset_ref=dataset_ref,
        upload_manifest_sha256=upload_manifest_sha,
        sample_weight_stats=checked["sample_weight_stats"],
        notes=notes,
    )
    command = base.notebook_command(
        notebook=notebook,
        env_file=env_file,
        kernel_slug=args.kernel_slug,
        title=args.title or "MiniLM 5ep: filtered soft positives",
        dataset_ref=dataset_ref,
    )
    base.run(command + ["--dry-run"])
    if (
        not args.dry_run_only
        and not args.skip_dataset_upload
        and not args.monitor_existing
    ):
        base.upload_dataset(stage_dir, manifest, message=args.dataset_message)

    common = {
        "contract_version": CONTRACT_VERSION,
        "dataset_ref": dataset_ref,
        "pair_count": args.pair_count,
        "target_counts": {"0": 0, "1": args.pair_count},
        "label_source": LABEL_SOURCE,
        "sample_weight_stats": checked["sample_weight_stats"],
        "source_fingerprint": checked["source_fingerprint"],
        "source_run_signature": checked["run_signature"],
        "upload_manifest_sha256": upload_manifest_sha,
        "notebook": str(notebook),
        "kernel_slug": args.kernel_slug,
        "experiment": args.experiment_label,
    }
    if args.dry_run_only:
        result = {"status": "dry_run_complete", **common}
    else:
        if args.monitor_existing:
            output_dir = base.monitor_existing_kernel(owner, args.kernel_slug)
        else:
            base.run(command)
            output_dir = base.output_directory(args.kernel_slug)
        completed = base.verify_completion(
            output_dir,
            experiment_label=args.experiment_label,
            dataset_ref=dataset_ref,
            upload_manifest_sha256=upload_manifest_sha,
            source_fingerprint=checked["source_fingerprint"],
            pair_count=args.pair_count,
            label_source=LABEL_SOURCE,
        )
        verify_weighted_completion(
            completed,
            pair_count=args.pair_count,
            weight_stats=checked["sample_weight_stats"],
            sample_weights=checked["pairs"]["sample_weight"],
        )
        result = {
            "status": "complete",
            **common,
            "run_id": completed["run_id"],
            "baseline_comparison": completed["comparison"],
            "google_sheets_sync": completed["sync"],
        }
    report = base.write_report(report_path, result)
    print(
        json.dumps(
            {**result, "launcher_report": str(report)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
