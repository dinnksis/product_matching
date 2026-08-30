#!/usr/bin/env python3
"""Create the locked MiniLM ablation for weighted filtered soft positives."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import nbformat


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / (
    "notebooks/minilm_5ep_team_ablation/minilm_5ep_team_ablation_2xt4.ipynb"
)
CONTRACT_VERSION = "soft_positive_quality_hardness_filter_v1"
LABEL_SOURCE = "qwen_soft_positive_ab_quality_hardness_v1"
WEIGHT_STATS_VERSION = "float64_le_pair_order_v1"
FROZEN_OOD_CATEGORIES = {"Одежда", "Бытовая техника"}
SOURCE_PROVENANCE_FILENAME = "filtered_soft_positive_source_provenance.json"
SOURCE_VALIDATION_FILENAME = "filtered_soft_positive_validation.json"


def _sha256(value: str, *, field: str) -> str:
    normalized = str(value).strip().casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"{field} must be a 64-character SHA-256")
    return normalized


def validate_weight_stats(value: Any, *, pair_count: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("sample weight stats must be a JSON object")
    required = {"version", "count", "sum", "min", "max", "sha256"}
    if set(value) != required:
        raise ValueError(
            "sample weight stats fields differ: "
            f"{sorted(value)} != {sorted(required)}"
        )
    if value.get("version") != WEIGHT_STATS_VERSION:
        raise ValueError("unsupported sample weight stats version")
    count = value.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count != pair_count:
        raise ValueError("sample weight count differs from pair-count")
    result: dict[str, Any] = {
        "version": WEIGHT_STATS_VERSION,
        "count": count,
        "sha256": _sha256(value.get("sha256", ""), field="weight SHA-256"),
    }
    for field in ("sum", "min", "max"):
        raw = value.get(field)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"sample weight {field} must be numeric")
        parsed = float(raw)
        if not math.isfinite(parsed) or parsed <= 0:
            raise ValueError(f"sample weight {field} must be finite and positive")
        result[field] = parsed
    if result["min"] > result["max"]:
        raise ValueError("sample weight min exceeds max")
    lower = result["min"] * pair_count
    upper = result["max"] * pair_count
    if not lower - 1e-12 <= result["sum"] <= upper + 1e-12:
        raise ValueError("sample weight sum lies outside the min/max bounds")
    return result


def data_hook(
    *,
    pair_count: int,
    artifact_tag: str,
    dataset_ref: str,
    upload_manifest_sha256: str,
    sample_weight_stats: dict[str, Any],
) -> str:
    pair_filename = f"generation_rule_pairs_{artifact_tag}.parquet"
    item_filename = f"generation_rule_items_{artifact_tag}.parquet"
    metadata_filename = f"generation_rule_pair_metadata_{artifact_tag}.parquet"
    expected_payload_files = {
        pair_filename,
        item_filename,
        metadata_filename,
        SOURCE_PROVENANCE_FILENAME,
        SOURCE_VALIDATION_FILENAME,
    }
    expected_stats = validate_weight_stats(sample_weight_stats, pair_count=pair_count)
    return f'''def build_train_data(human_train_pairs, human_items, input_root):
    import hashlib
    import json
    import math
    import numpy as np
    import pandas as pd

    def sha256_file(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def exact_weight_stats(values):
        weights = pd.to_numeric(values, errors="coerce").to_numpy(dtype="<f8")
        if len(weights) != {pair_count}:
            raise ValueError("Filtered synthetic sample weight count mismatch")
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Filtered synthetic sample weights must be finite and positive")
        return {{
            "version": {WEIGHT_STATS_VERSION!r},
            "count": int(len(weights)),
            "sum": float(math.fsum(float(value) for value in weights)),
            "min": float(weights.min()),
            "max": float(weights.max()),
            "sha256": hashlib.sha256(weights.tobytes(order="C")).hexdigest(),
        }}

    matching_manifests = []
    for manifest_path in input_root.glob("**/upload_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if manifest.get("dataset") == {dataset_ref!r}:
            matching_manifests.append((manifest_path, manifest))
    if len(matching_manifests) != 1:
        raise RuntimeError(
            "Expected exactly one pinned filtered-soft-positive manifest; "
            f"matches={{[str(path) for path, _ in matching_manifests]}}"
        )
    manifest_path, upload_manifest = matching_manifests[0]
    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != {upload_manifest_sha256!r}:
        raise RuntimeError(
            "Filtered synthetic upload manifest SHA mismatch: "
            f"{{actual_manifest_sha256}} != " {upload_manifest_sha256!r}
        )
    expected_weight_stats = {expected_stats!r}
    expected_targets = {{"0": 0, "1": {pair_count}}}
    if (
        upload_manifest.get("contract_version") != {CONTRACT_VERSION!r}
        or upload_manifest.get("generation_kind") != {CONTRACT_VERSION!r}
        or upload_manifest.get("label_source") != {LABEL_SOURCE!r}
        or upload_manifest.get("pairs") != {pair_count}
        or upload_manifest.get("items") != {pair_count * 2}
        or upload_manifest.get("targets") != expected_targets
        or upload_manifest.get("sample_weight_stats") != expected_weight_stats
    ):
        raise ValueError("Filtered synthetic upload manifest contract mismatch")
    source_provenance = upload_manifest.get("source_provenance") or {{}}
    if (
        source_provenance.get("contract_version") != {CONTRACT_VERSION!r}
        or source_provenance.get("candidate_decisions_in_train_payload") is not False
        or "candidate_decisions.parquet"
        not in (source_provenance.get("source_files") or {{}})
    ):
        raise ValueError("Filtered synthetic source provenance contract mismatch")

    generated_root = manifest_path.parent
    pair_path = generated_root / {pair_filename!r}
    item_path = generated_root / {item_filename!r}
    manifest_files = upload_manifest.get("files") or {{}}
    expected_payload_files = {expected_payload_files!r}
    if set(manifest_files) != expected_payload_files:
        raise RuntimeError("Filtered synthetic payload file set differs")
    for filename in sorted(expected_payload_files):
        path = generated_root / filename
        record = manifest_files.get(filename)
        if (
            not isinstance(record, dict)
            or not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or sha256_file(path) != record.get("sha256")
        ):
            raise RuntimeError(f"Filtered synthetic file differs: {{path.name}}")

    extra_pairs = pd.read_parquet(pair_path)
    extra_items = pd.read_parquet(item_path)
    required_pair_columns = {{"id1", "id2", "target", "sample_weight", "label_source"}}
    required_item_columns = {{"id", "name", "category", "product_text"}}
    if missing := required_pair_columns - set(extra_pairs.columns):
        raise ValueError(f"Filtered synthetic pairs missing columns: {{sorted(missing)}}")
    if missing := required_item_columns - set(extra_items.columns):
        raise ValueError(f"Filtered synthetic items missing columns: {{sorted(missing)}}")
    if len(extra_pairs) != {pair_count} or len(extra_items) != {pair_count * 2}:
        raise ValueError("Filtered synthetic dimensions differ from the pinned counts")
    if extra_pairs[["id1", "id2", "target"]].isna().any().any():
        raise ValueError("Filtered synthetic pairs contain null IDs/targets")
    if not extra_pairs["target"].eq(1).all():
        raise ValueError("Filtered synthetic pairs must be all target=1")
    if set(extra_pairs["label_source"].astype(str)) != {{{LABEL_SOURCE!r}}}:
        raise ValueError("Filtered synthetic label_source differs")
    actual_weight_stats = exact_weight_stats(extra_pairs["sample_weight"])
    if actual_weight_stats != expected_weight_stats:
        raise ValueError(
            f"Filtered synthetic sample weights differ: "
            f"{{actual_weight_stats}} != {{expected_weight_stats}}"
        )
    if extra_items["id"].duplicated().any():
        raise ValueError("Filtered synthetic item IDs are not unique")
    endpoints = pd.concat([extra_pairs["id1"], extra_pairs["id2"]], ignore_index=True)
    if endpoints.duplicated().any() or set(endpoints) != set(extra_items["id"]):
        raise ValueError("Every filtered synthetic card must be used exactly once")
    forbidden = set(extra_items["category"].astype(str)) & {FROZEN_OOD_CATEGORIES!r}
    if forbidden:
        raise ValueError(f"Filtered synthetic train uses frozen OOD categories: {{sorted(forbidden)}}")
    category_by_id = extra_items.set_index("id")["category"].astype(str)
    if not extra_pairs["id1"].map(category_by_id).equals(
        extra_pairs["id2"].map(category_by_id)
    ):
        raise ValueError("Filtered synthetic source contains cross-category pairs")

    base_pairs = human_train_pairs.copy()
    base_pairs["sample_weight"] = 1.0
    base_pairs["label_source"] = "human"
    extra_pairs = extra_pairs.copy()
    extra_pairs["sample_weight"] = pd.to_numeric(
        extra_pairs["sample_weight"], errors="raise"
    ).astype("float64")
    extra_pairs["label_source"] = extra_pairs["label_source"].astype(str)
    train_pairs = pd.concat(
        [base_pairs, extra_pairs[list(base_pairs.columns)]], ignore_index=True
    )
    items = pd.concat(
        [human_items, extra_items[list(human_items.columns)]], ignore_index=True
    )
    if items["id"].duplicated().any():
        raise ValueError("Human and filtered synthetic item IDs overlap")
    print({{
        "human_pairs": len(base_pairs),
        "generated_pairs": len(extra_pairs),
        "generated_target_counts": expected_targets,
        "generated_sample_weight_stats": actual_weight_stats,
        "total_pairs": len(train_pairs),
        "label_sources": train_pairs["label_source"].value_counts().to_dict(),
        "generated_dataset_ref": {dataset_ref!r},
        "generated_upload_manifest_sha256": actual_manifest_sha256,
    }})
    return train_pairs, items
'''


def routing_cell(
    *,
    experiment_label: str,
    pair_count: int,
    dataset_ref: str,
    upload_manifest_sha256: str,
    sample_weight_stats: dict[str, Any],
    notes: str | None,
) -> str:
    stats = validate_weight_stats(sample_weight_stats, pair_count=pair_count)
    effective_notes = notes or (
        f"Frozen MiniLM 5ep human baseline plus {pair_count:,} filtered Qwen "
        "soft-positive pairs. Synthetic sample weights are preserved exactly; "
        "human sample weight is one. Frozen recipe and IID/hard/OOD paired "
        f"significance are unchanged. Source {dataset_ref}; upload manifest "
        f"SHA-256 {upload_manifest_sha256}; source weight SHA-256 "
        f"{stats['sha256']}."
    )
    return (
        f"EXPERIMENT_LABEL = {experiment_label!r}\n"
        "EXPERIMENT_SHEET = 'data_exps'  # pretrain_exps | sft_exps | data_exps\n"
        f"EXPERIMENT_NOTES = {effective_notes!r}\n"
    )


def build_notebook(
    *,
    pair_count: int,
    artifact_tag: str,
    experiment_label: str,
    dataset_ref: str,
    upload_manifest_sha256: str,
    sample_weight_stats: dict[str, Any],
    notes: str | None = None,
) -> nbformat.NotebookNode:
    if pair_count < 1:
        raise ValueError("pair_count must be positive")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", artifact_tag):
        raise ValueError("artifact_tag must be a lowercase slug")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", experiment_label):
        raise ValueError("experiment_label must be a lowercase slug")
    manifest_sha = _sha256(
        upload_manifest_sha256, field="upload manifest SHA-256"
    )
    weights = validate_weight_stats(sample_weight_stats, pair_count=pair_count)
    source = nbformat.read(SOURCE, as_version=4)
    notebook = nbformat.from_dict(source)
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
        raise RuntimeError(
            "locked notebook must contain one data hook and one routing cell"
        )
    data_cells[0].source = data_hook(
        pair_count=pair_count,
        artifact_tag=artifact_tag,
        dataset_ref=dataset_ref,
        upload_manifest_sha256=manifest_sha,
        sample_weight_stats=weights,
    )
    routing_cells[0].source = routing_cell(
        experiment_label=experiment_label,
        pair_count=pair_count,
        dataset_ref=dataset_ref,
        upload_manifest_sha256=manifest_sha,
        sample_weight_stats=weights,
        notes=notes,
    )
    for original, generated in zip(source.cells, notebook.cells, strict=True):
        tags = original.get("metadata", {}).get("tags", [])
        if "frozen" in tags and original.source != generated.source:
            raise RuntimeError("a frozen notebook cell was modified")
    nbformat.validate(notebook)
    return notebook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-count", type=int, required=True)
    parser.add_argument("--artifact-tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-label", required=True)
    parser.add_argument("--dataset-ref", required=True)
    parser.add_argument("--upload-manifest-sha256", required=True)
    parser.add_argument(
        "--sample-weight-stats-json",
        required=True,
        help="exact JSON object with version/count/sum/min/max/sha256",
    )
    parser.add_argument("--notes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        weight_stats = json.loads(args.sample_weight_stats_json)
    except json.JSONDecodeError as error:
        raise ValueError("--sample-weight-stats-json is invalid JSON") from error
    output = args.output if args.output.is_absolute() else ROOT / args.output
    notebook = build_notebook(
        pair_count=args.pair_count,
        artifact_tag=args.artifact_tag,
        experiment_label=args.experiment_label,
        dataset_ref=args.dataset_ref,
        upload_manifest_sha256=args.upload_manifest_sha256,
        sample_weight_stats=weight_stats,
        notes=args.notes,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    print(f"Created {output}")


if __name__ == "__main__":
    main()
