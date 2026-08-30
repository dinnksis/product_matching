#!/usr/bin/env python3
"""Prepare component-disjoint OOF folds and label-audit flags for S2 hard mining."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cheap_ensemble import prepare_item_records
from src.serialization_ablation import (
    _union_find_groups,
    normalize_text,
    parse_attributes,
    stable_hash,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--train-pairs", type=Path, required=True)
    parser.add_argument("--hard-audit-assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def full_representation_hash(row: Any) -> str:
    payload = json.dumps(
        [
            normalize_text(row.category),
            normalize_text(row.name),
            tuple(sorted(parse_attributes(row.attributes))),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def set_conflict(left: frozenset[str], right: frozenset[str]) -> bool:
    return bool(left and right and not left & right)


def pair_flags(
    records: pd.DataFrame,
    left_positions: np.ndarray,
    right_positions: np.ndarray,
    full_hashes: np.ndarray,
) -> pd.DataFrame:
    columns = {
        column: records[column].to_numpy(dtype=object)
        for column in (
            "title",
            "numbers",
            "measures",
            "slash_specs",
            "codes",
            "brand_values",
            "model_values",
            "memory_values",
            "color_values",
        )
    }
    output: list[dict[str, Any]] = []
    for left, right in zip(left_positions, right_positions):
        left = int(left)
        right = int(right)
        left_numbers = columns["numbers"][left]
        right_numbers = columns["numbers"][right]
        number_overlap = bool(left_numbers & right_numbers)
        unit_conflicts = 0
        left_measures = columns["measures"][left]
        right_measures = columns["measures"][right]
        for family in set(left_measures) & set(right_measures):
            if not left_measures[family] & right_measures[family]:
                unit_conflicts += 1
        left_slash = columns["slash_specs"][left]
        right_slash = columns["slash_specs"][right]
        slash_conflict = bool(left_slash and right_slash and not left_slash & right_slash)
        numeric_conflict_count = (
            min(
                len(left_numbers - right_numbers),
                len(right_numbers - left_numbers),
            )
            if number_overlap
            else 0
        ) + unit_conflicts + int(slash_conflict)
        left_codes = columns["codes"][left]
        right_codes = columns["codes"][right]
        code_conflict = set_conflict(left_codes, right_codes)
        brand_conflict = set_conflict(
            columns["brand_values"][left], columns["brand_values"][right]
        )
        model_conflict = set_conflict(
            columns["model_values"][left], columns["model_values"][right]
        )
        memory_conflict = set_conflict(
            columns["memory_values"][left], columns["memory_values"][right]
        )
        color_conflict = set_conflict(
            columns["color_values"][left], columns["color_values"][right]
        )
        output.append(
            {
                "exact_normalized_title_match": (
                    columns["title"][left] == columns["title"][right]
                ),
                "identical_full_representation": full_hashes[left] == full_hashes[right],
                "numeric_conflict": numeric_conflict_count > 0,
                "numeric_context_conflict_count": numeric_conflict_count,
                "code_conflict": code_conflict,
                "model_code_conflict": model_conflict,
                "critical_attribute_conflict": (
                    brand_conflict or model_conflict or memory_conflict or color_conflict
                ),
                "brand_conflict": brand_conflict,
                "model_conflict": model_conflict,
                "memory_conflict": memory_conflict,
                "color_conflict": color_conflict,
                "sku_vs_human_title": bool(left_codes) != bool(right_codes),
            }
        )
    return pd.DataFrame.from_records(output)


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(args.train_pairs).reset_index(drop=True)
    if len(train) != int(config["expected_pairs"]["train"]):
        raise ValueError("unexpected train row count")
    train.insert(0, "row_id", np.arange(len(train), dtype=np.int32))
    required_ids = set(train["id1"].tolist()) | set(train["id2"].tolist())
    raw_items = pd.read_parquet(
        args.items, columns=["id", "name", "attributes", "category"]
    )
    selected = raw_items[raw_items["id"].isin(required_ids)].reset_index(drop=True)
    if len(selected) != len(required_ids):
        raise ValueError("items file does not contain every train product")
    positions = pd.Series(
        np.arange(len(selected), dtype=np.int32), index=selected["id"].to_numpy()
    )
    left = positions.loc[train["id1"]].to_numpy(dtype=np.int32)
    right = positions.loc[train["id2"]].to_numpy(dtype=np.int32)
    records = prepare_item_records(selected)
    full_hashes = np.asarray(
        [full_representation_hash(row) for row in selected.itertuples(index=False)],
        dtype=object,
    )
    flags = pair_flags(records, left, right, full_hashes)
    audit = pd.concat([train, flags], axis=1)

    canonical_left = np.minimum(
        audit["id1"].to_numpy(dtype=np.int64), audit["id2"].to_numpy(dtype=np.int64)
    )
    canonical_right = np.maximum(
        audit["id1"].to_numpy(dtype=np.int64), audit["id2"].to_numpy(dtype=np.int64)
    )
    audit["canonical_id1"] = canonical_left
    audit["canonical_id2"] = canonical_right
    audit["unordered_id_pair_target_conflict"] = audit.groupby(
        ["canonical_id1", "canonical_id2"], sort=False
    )["target"].transform("nunique").gt(1)
    rep_left = full_hashes[left].astype(str)
    rep_right = full_hashes[right].astype(str)
    left_first = rep_left <= rep_right
    audit["canonical_representation1"] = np.where(left_first, rep_left, rep_right)
    audit["canonical_representation2"] = np.where(left_first, rep_right, rep_left)
    audit["representation_pair_target_conflict"] = audit.groupby(
        ["canonical_representation1", "canonical_representation2"], sort=False
    )["target"].transform("nunique").gt(1)
    audit["definite_label_conflict"] = audit[
        ["unordered_id_pair_target_conflict", "representation_pair_target_conflict"]
    ].any(axis=1)
    negative = audit["target"].eq(0)
    positive = audit["target"].eq(1)
    audit["suspicious_negative_identity"] = negative & audit[
        ["exact_normalized_title_match", "identical_full_representation"]
    ].any(axis=1)
    audit["suspicious_positive_conflict"] = positive & audit[
        [
            "numeric_conflict",
            "code_conflict",
            "model_code_conflict",
            "critical_attribute_conflict",
        ]
    ].any(axis=1)
    audit["strong_label_suspicion"] = audit[
        ["suspicious_negative_identity", "suspicious_positive_conflict"]
    ].any(axis=1)
    audit["eligible_for_hard_mining"] = ~audit[
        ["definite_label_conflict", "strong_label_suspicion"]
    ].any(axis=1)

    signatures = records["family_signature"].astype(str).to_numpy(dtype=object)
    roots = _union_find_groups(len(selected), left, right, signatures)
    if not np.array_equal(roots[left], roots[right]):
        raise RuntimeError("pair endpoints were assigned to different components")
    pair_roots = roots[left]
    folds = int(config["oof_folds"])
    seed = int(config["oof_group_seed"])
    root_to_fold = {
        int(root): stable_hash(int(root), seed) % folds for root in np.unique(pair_roots)
    }
    audit["oof_fold"] = np.asarray(
        [root_to_fold[int(root)] for root in pair_roots], dtype=np.int8
    )

    audit.to_parquet(args.output_dir / "train_label_audit_and_folds.parquet", index=False)
    pair_columns = ["id1", "id2", "target"]
    for fold in range(folds):
        audit.loc[audit["oof_fold"].ne(fold), pair_columns].to_parquet(
            args.output_dir / f"oof_fold_{fold}_train_pairs.parquet", index=False
        )
        audit.loc[audit["oof_fold"].eq(fold), pair_columns].to_parquet(
            args.output_dir / f"oof_fold_{fold}_validation_pairs.parquet", index=False
        )

    assignments = pd.read_csv(args.hard_audit_assignments)
    required_assignment_columns = {"id1", "id2", "target", "hard_subset"}
    if required_assignment_columns - set(assignments.columns):
        raise ValueError("hard audit assignments have an unexpected schema")
    hard_clean = assignments[assignments["hard_subset"].eq("hard_clean")]
    if len(hard_clean) != int(config["expected_pairs"]["hard_clean"]):
        raise ValueError("unexpected hard-clean row count")
    hard_clean[pair_columns].to_parquet(
        args.output_dir / "hard_clean_pairs.parquet", index=False
    )
    audit_item_ids = set(assignments["id1"].tolist()) | set(assignments["id2"].tolist())
    audit_train_item_overlap = len(required_ids & audit_item_ids)
    if audit_train_item_overlap:
        raise RuntimeError("hard audit and train unexpectedly share items")

    fold_rows = []
    for fold, part in audit.groupby("oof_fold", sort=True):
        fold_rows.append(
            {
                "fold": int(fold),
                "pairs": len(part),
                "positives": int(part["target"].sum()),
                "prevalence": float(part["target"].mean()),
                "items": len(set(part["id1"].tolist()) | set(part["id2"].tolist())),
                "eligible_for_hard_mining": int(part["eligible_for_hard_mining"].sum()),
            }
        )
    report = {
        "experiment": config["experiment"],
        "train_pairs": len(audit),
        "train_items": len(required_ids),
        "oof_folds": fold_rows,
        "audit": {
            "definite_label_conflicts": int(audit["definite_label_conflict"].sum()),
            "strong_label_suspicion": int(audit["strong_label_suspicion"].sum()),
            "eligible_for_hard_mining": int(audit["eligible_for_hard_mining"].sum()),
        },
        "hard_audit_train_item_overlap": audit_train_item_overlap,
        "hard_clean_pairs": len(hard_clean),
        "selection_uses_predictions": False,
    }
    (args.output_dir / "preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
