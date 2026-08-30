#!/usr/bin/env python3
"""Compose the completed Tier A and a clean 10k Tier B checkpoint snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from item_pipeline.pair_validation import validate_pair_dataset
from scripts.freeze_generated_pair_dataset import canonical_card


VERSION = "soft_positive_tier_ab_composition_v1"
SELECTION_POLICY = "all_tier_a_then_b_rule_coverage_then_task_order_v1"
REINDEX_VERSION = "global_negative_pair_ids_v1"
A_COUNT = 6_351
B_COUNT = 10_000
TOTAL_COUNT = A_COUNT + B_COUNT
SEMANTIC_SIGNATURE_LIMIT = 5
MODEL = "qwen3.5-397b-a17b-fp8"
API_BASE_URL = "http://0.0.0.0:8994/v1"
A_CATALOG_SHA256 = "44d2f623958cabc6c61fdf6c837faab8bbcffd77e0bfd5ed5a96f7eaa951fa7f"
B_CATALOG_SHA256 = "11a086376261b7ab058544eb31073e0b33e175453841e2a2801104eef8df38d1"
A_DIR = ROOT / "item_pipeline/artifacts/soft_positive_tier_a_qwen_v1_frozen"
B_DIR = ROOT / "item_pipeline/artifacts/soft_positive_tier_b_27930_qwen_v1_raw"
OUTPUT_DIR = ROOT / "item_pipeline/artifacts/soft_positive_tier_ab_16351_qwen_v1_composed"
A_CATALOG = ROOT / "configs/generation_rule_catalog_statistical_v1/soft_positive_ab_v1/tier_a.json"
B_CATALOG = ROOT / "configs/generation_rule_catalog_statistical_v1/soft_positive_ab_v1/tier_b.json"
FORBIDDEN_OOD_CATEGORIES = {"Одежда", "Бытовая техника"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier-a-dir", type=Path, default=A_DIR)
    parser.add_argument("--tier-b-dir", type=Path, default=B_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--tier-b-count", type=int, default=B_COUNT)
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_source(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = [
        source_dir / "pair_generation_metadata.parquet",
        source_dir / "pairs.parquet",
        source_dir / "items.parquet",
    ]
    if missing := [str(path) for path in paths if not path.is_file()]:
        raise RuntimeError(f"missing source artifacts: {missing}")
    metadata = pd.read_parquet(paths[0]).sort_values("task_index", kind="stable")
    pairs = pd.read_parquet(paths[1])
    items = pd.read_parquet(paths[2])
    required_metadata = {
        "task_index",
        "id1",
        "id2",
        "target",
        "rule_count",
        "scheduled_primary_rule_id",
        "semantic_signature",
        "run_signature",
        "model",
    }
    if missing := required_metadata - set(metadata.columns):
        raise RuntimeError(f"source metadata lacks columns: {sorted(missing)}")
    if metadata["task_index"].duplicated().any():
        raise RuntimeError("source metadata task_index is not unique")
    return metadata, pairs, items


def source_maps(
    pairs: pd.DataFrame, items: pd.DataFrame
) -> tuple[dict[tuple[int, int], Any], dict[int, Any]]:
    pair_map = {
        (int(row.id1), int(row.id2)): row for row in pairs.itertuples(index=False)
    }
    item_map = {int(row.id): row for row in items.itertuples(index=False)}
    return pair_map, item_map


def validate_common_source(
    name: str,
    metadata: pd.DataFrame,
    pair_map: dict[tuple[int, int], Any],
    item_map: dict[int, Any],
    catalog_ids: set[str],
) -> None:
    if not metadata["target"].eq(1).all() or not metadata["rule_count"].eq(1).all():
        raise RuntimeError(f"{name} must contain atomic target=1 pairs only")
    if set(metadata["model"].astype(str)) != {MODEL}:
        raise RuntimeError(f"{name} model differs from {MODEL}")
    observed_rules = set(metadata["scheduled_primary_rule_id"].astype(str))
    if observed_rules - catalog_ids:
        raise RuntimeError(f"{name} metadata contains rules outside its catalog")
    for row in metadata.itertuples(index=False):
        key = (int(row.id1), int(row.id2))
        if key not in pair_map or key[0] not in item_map or key[1] not in item_map:
            raise RuntimeError(f"{name} source artifacts are not a consistent checkpoint")


def row_card_keys(row: Any, item_map: dict[int, Any]) -> tuple[str, str]:
    left = canonical_card(pd.Series(item_map[int(row.id1)]._asdict()))
    right = canonical_card(pd.Series(item_map[int(row.id2)]._asdict()))
    return left, right


def add_if_clean(
    row: Any,
    *,
    item_map: dict[int, Any],
    seen_cards: set[str],
    signature_counts: Counter[str],
) -> tuple[bool, str]:
    left, right = row_card_keys(row, item_map)
    if left == right:
        return False, "identical_pair_card"
    if left in seen_cards or right in seen_cards:
        return False, "duplicate_global_card"
    signature = str(row.semantic_signature)
    if signature_counts[signature] >= SEMANTIC_SIGNATURE_LIMIT:
        return False, "semantic_signature_limit"
    seen_cards.update((left, right))
    signature_counts[signature] += 1
    return True, ""


def select_rows(
    a_metadata: pd.DataFrame,
    a_items: dict[int, Any],
    b_metadata: pd.DataFrame,
    b_items: dict[int, Any],
    b_count: int,
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    seen_cards: set[str] = set()
    signatures: Counter[str] = Counter()
    a_selected: list[Any] = []
    b_selected: list[Any] = []
    selected_b_keys: set[tuple[int, int]] = set()
    selected_b_rules: set[str] = set()
    drop_reasons: Counter[str] = Counter()

    for row in a_metadata.itertuples(index=False):
        accepted, reason = add_if_clean(
            row,
            item_map=a_items,
            seen_cards=seen_cards,
            signature_counts=signatures,
        )
        if not accepted:
            raise RuntimeError(f"Tier A frozen source violates {reason}")
        a_selected.append(row)
    if len(a_selected) != A_COUNT:
        raise RuntimeError(f"Tier A must contain exactly {A_COUNT} rows")

    b_rows = list(b_metadata.itertuples(index=False))
    # First retain one clean example for every B rule already represented in the
    # live checkpoint. This maximizes rule coverage without waiting for all 27,930.
    for row in b_rows:
        rule_id = str(row.scheduled_primary_rule_id)
        if rule_id in selected_b_rules:
            continue
        accepted, reason = add_if_clean(
            row,
            item_map=b_items,
            seen_cards=seen_cards,
            signature_counts=signatures,
        )
        if not accepted:
            drop_reasons[reason] += 1
            continue
        b_selected.append(row)
        selected_b_keys.add((int(row.id1), int(row.id2)))
        selected_b_rules.add(rule_id)

    for row in b_rows:
        if len(b_selected) >= b_count:
            break
        pair_key = (int(row.id1), int(row.id2))
        if pair_key in selected_b_keys:
            continue
        accepted, reason = add_if_clean(
            row,
            item_map=b_items,
            seen_cards=seen_cards,
            signature_counts=signatures,
        )
        if not accepted:
            drop_reasons[reason] += 1
            continue
        b_selected.append(row)
        selected_b_keys.add(pair_key)
        selected_b_rules.add(str(row.scheduled_primary_rule_id))

    if len(b_selected) != b_count:
        raise RuntimeError(
            f"only {len(b_selected):,}/{b_count:,} clean Tier B rows are available"
        )
    provenance = {
        "selection_policy": SELECTION_POLICY,
        "semantic_signature_limit": SEMANTIC_SIGNATURE_LIMIT,
        "semantic_signature_unique_count": len(signatures),
        "semantic_signature_max_count": max(signatures.values(), default=0),
        "tier_b_checkpoint_rows_seen": len(b_metadata),
        "tier_b_selected_rule_coverage": len(selected_b_rules),
        "tier_b_selected_task_index_sha256": sha256_json(
            [int(row.task_index) for row in b_selected]
        ),
        "tier_b_selected_max_task_index": max(int(row.task_index) for row in b_selected),
        "selection_rejection_counts": dict(sorted(drop_reasons.items())),
    }
    return a_selected, b_selected, provenance


def compose_rows(
    selected: list[tuple[str, Any, dict[int, Any], dict[tuple[int, int], Any]]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    item_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    next_id = -1_000_000_000
    for composition_index, (component, row, item_map, pair_map) in enumerate(selected):
        source_key = (int(row.id1), int(row.id2))
        source_pair = pair_map[source_key]._asdict()
        source_left = item_map[source_key[0]]._asdict()
        source_right = item_map[source_key[1]]._asdict()
        id1, id2 = next_id, next_id - 1
        next_id -= 2
        source_left["id"] = id1
        source_right["id"] = id2
        item_rows.extend((source_left, source_right))
        pair_rows.append({"id1": id1, "id2": id2, "target": int(source_pair["target"])})
        metadata = row._asdict()
        metadata.update(
            {
                "composition_index": composition_index,
                "component": component,
                "source_task_index": int(row.task_index),
                "source_id1": source_key[0],
                "source_id2": source_key[1],
                "source_run_signature": str(row.run_signature),
                "task_index": composition_index,
                "id1": id1,
                "id2": id2,
                "target": 1,
            }
        )
        metadata_rows.append(metadata)
    return (
        pd.DataFrame(item_rows),
        pd.DataFrame(pair_rows),
        pd.DataFrame(metadata_rows),
    )


def compose(a_dir: Path, b_dir: Path, output_dir: Path, b_count: int) -> dict[str, Any]:
    if b_count != B_COUNT:
        raise ValueError(f"this pinned ablation requires exactly {B_COUNT} Tier B rows")
    if sha256_file(A_CATALOG) != A_CATALOG_SHA256 or sha256_file(B_CATALOG) != B_CATALOG_SHA256:
        raise RuntimeError("soft-positive catalog SHA-256 differs from the pinned inputs")
    a_catalog = json.loads(A_CATALOG.read_text(encoding="utf-8"))
    b_catalog = json.loads(B_CATALOG.read_text(encoding="utf-8"))
    a_ids = {str(row["generation_rule_id"]) for row in a_catalog}
    b_ids = {str(row["generation_rule_id"]) for row in b_catalog}
    if len(a_ids) != 325 or len(b_ids) != 2_793 or a_ids & b_ids:
        raise RuntimeError("soft-positive A/B catalog dimensions or disjointness differ")

    a_metadata, a_pairs, a_items_frame = read_source(a_dir)
    b_metadata, b_pairs, b_items_frame = read_source(b_dir)
    a_pair_map, a_item_map = source_maps(a_pairs, a_items_frame)
    b_pair_map, b_item_map = source_maps(b_pairs, b_items_frame)
    validate_common_source("Tier A", a_metadata, a_pair_map, a_item_map, a_ids)
    validate_common_source("Tier B", b_metadata, b_pair_map, b_item_map, b_ids)
    a_selected, b_selected, selection = select_rows(
        a_metadata, a_item_map, b_metadata, b_item_map, b_count
    )
    selected = [
        *(('tier_a', row, a_item_map, a_pair_map) for row in a_selected),
        *(('tier_b', row, b_item_map, b_pair_map) for row in b_selected),
    ]
    items, pairs, metadata = compose_rows(selected)
    if len(pairs) != TOTAL_COUNT or len(items) != TOTAL_COUNT * 2:
        raise RuntimeError("composed A+B dimensions differ")
    if items["id"].duplicated().any() or not pairs["target"].eq(1).all():
        raise RuntimeError("composed A+B IDs or targets differ")
    if set(items["category"].astype(str)) & FORBIDDEN_OOD_CATEGORIES:
        raise RuntimeError("composed A+B data leaks frozen OOD categories")
    card_keys = [canonical_card(row) for _, row in items.iterrows()]
    if len(card_keys) != len(set(card_keys)):
        raise RuntimeError("composed A+B cards are not globally unique")
    signature_counts = metadata["semantic_signature"].astype(str).value_counts()
    if int(signature_counts.max()) > SEMANTIC_SIGNATURE_LIMIT:
        raise RuntimeError("composed A+B semantic signature cap is exceeded")

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_parquet(items, output_dir / "items.parquet")
    atomic_parquet(pairs, output_dir / "pairs.parquet")
    atomic_parquet(metadata, output_dir / "pair_generation_metadata.parquet")
    validation = validate_pair_dataset(
        output_dir / "items.parquet",
        output_dir / "pairs.parquet",
        metadata_path=output_dir / "pair_generation_metadata.parquet",
    )
    if validation.get("valid") is not True or int(validation.get("pairs", -1)) != TOTAL_COUNT:
        raise RuntimeError(f"composed A+B validation failed: {validation}")
    atomic_json(validation, output_dir / "validation_report.json")

    source_runs = {
        component: {
            "run_signatures": sorted(
                set(metadata.loc[metadata["component"].eq(component), "source_run_signature"].astype(str))
            ),
            "selected_pairs": int(metadata["component"].eq(component).sum()),
            "selected_task_index_sha256": sha256_json(
                metadata.loc[metadata["component"].eq(component), "source_task_index"]
                .astype(int)
                .tolist()
            ),
        }
        for component in ("tier_a", "tier_b")
    }
    summary = {
        "version": VERSION,
        "status": "complete",
        "count": TOTAL_COUNT,
        "generated_pairs": TOTAL_COUNT,
        "completed": TOTAL_COUNT,
        "pending": 0,
        "errors": 0,
        "target_counts": {"0": 0, "1": TOTAL_COUNT},
        "model": MODEL,
        "api_base_url": API_BASE_URL,
        "structured_output": False,
        "reasoning_effort": None,
        "temperature": 0.7,
        "max_tokens": 1_400,
        "two_rule_fraction": 0.0,
        "label_one_fraction": 1.0,
        "rule_tiers": ["SOFT_POSITIVE_TIER_A", "SOFT_POSITIVE_TIER_B"],
        "rule_catalogs": [
            {"path": str(A_CATALOG), "sha256": A_CATALOG_SHA256},
            {"path": str(B_CATALOG), "sha256": B_CATALOG_SHA256},
        ],
        "semantic_signature_version": str(metadata["semantic_signature_version"].iloc[0]),
        "semantic_signature_limit": SEMANTIC_SIGNATURE_LIMIT,
        "semantic_signature_unique_count": int(signature_counts.size),
        "semantic_signature_max_count": int(signature_counts.max()),
        "attempt_diversity_versions": sorted(set(metadata["attempt_diversity_version"].astype(str))),
        "composition": {
            "version": VERSION,
            "selection_policy": SELECTION_POLICY,
            "reindex_version": REINDEX_VERSION,
            "tier_a_source_dir": str(a_dir),
            "tier_b_source_dir": str(b_dir),
            "tier_a_catalog_sha256": A_CATALOG_SHA256,
            "tier_b_catalog_sha256": B_CATALOG_SHA256,
            "source_runs": source_runs,
            **selection,
        },
        "created_at": datetime.now(UTC).isoformat(),
    }
    summary["run_signature"] = sha256_json(
        {key: value for key, value in summary.items() if key != "created_at"}
    )
    metadata["run_signature"] = summary["run_signature"]
    atomic_parquet(metadata, output_dir / "pair_generation_metadata.parquet")
    atomic_json(summary, output_dir / "summary.json")
    atomic_json([], output_dir / "errors.json")

    files = {
        name: {"bytes": (output_dir / name).stat().st_size, "sha256": sha256_file(output_dir / name)}
        for name in (
            "items.parquet",
            "pairs.parquet",
            "pair_generation_metadata.parquet",
            "summary.json",
            "validation_report.json",
            "errors.json",
        )
    }
    manifest = {
        "version": VERSION,
        "pairs": TOTAL_COUNT,
        "items": TOTAL_COUNT * 2,
        "targets": {"0": 0, "1": TOTAL_COUNT},
        "globally_unique_cards": len(set(card_keys)),
        "semantic_signature_limit": SEMANTIC_SIGNATURE_LIMIT,
        "semantic_signature_max_count": int(signature_counts.max()),
        "selection": selection,
        "source_runs": source_runs,
        "files": files,
    }
    atomic_json(manifest, output_dir / "composition_manifest.json")
    return {"summary": summary, "manifest": manifest, "validation": validation}


def main() -> None:
    args = parse_args()
    result = compose(
        absolute(args.tier_a_dir),
        absolute(args.tier_b_dir),
        absolute(args.output_dir),
        args.tier_b_count,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
