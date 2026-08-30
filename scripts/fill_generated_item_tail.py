#!/usr/bin/env python3
"""Fill missing primary generation tasks from a fresh supplemental run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from item_pipeline.normalization import normalize_text
from item_pipeline.validation import validate_generated_dataset


PRIMARY = ROOT / "item_pipeline" / "artifacts" / "generated"
SUPPLEMENT = ROOT / "item_pipeline" / "artifacts" / "generated_supplement_plain_json"
REFERENCE = ROOT / "item_pipeline" / "artifacts" / "index" / "exemplar_bank.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-dir", type=Path, default=PRIMARY)
    parser.add_argument("--supplement-dir", type=Path, default=SUPPLEMENT)
    parser.add_argument("--count", type=int, default=10_000)
    return parser.parse_args()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def atomic_json(value: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    primary_dir = args.primary_dir.resolve()
    supplement_dir = args.supplement_dir.resolve()
    primary_items = pd.read_parquet(primary_dir / "items.parquet")
    primary_metadata = pd.read_parquet(primary_dir / "generation_metadata.parquet")
    supplement_items = pd.read_parquet(supplement_dir / "items.parquet")
    supplement_metadata = pd.read_parquet(
        supplement_dir / "generation_metadata.parquet"
    )
    primary_summary = json.loads(
        (primary_dir / "summary.json").read_text(encoding="utf-8")
    )
    supplement_summary = json.loads(
        (supplement_dir / "summary.json").read_text(encoding="utf-8")
    )

    if primary_items["id"].duplicated().any() or primary_metadata["task_index"].duplicated().any():
        raise ValueError("Primary checkpoint contains duplicate ids/task indices")
    if supplement_items["id"].duplicated().any() or supplement_metadata["task_index"].duplicated().any():
        raise ValueError("Supplement checkpoint contains duplicate ids/task indices")
    missing_tasks = sorted(set(range(args.count)) - set(primary_metadata["task_index"]))
    if not missing_tasks:
        print("Primary item dataset is already complete")
        return

    supplement_by_id = supplement_items.set_index("id", drop=False)
    existing_names = {normalize_text(value) for value in primary_items["name"]}
    selected_metadata: list[pd.Series] = []
    selected_items: list[pd.Series] = []
    for row in supplement_metadata.sort_values("task_index").itertuples(index=False):
        item = supplement_by_id.loc[int(row.id)]
        normalized_name = normalize_text(item["name"])
        if not normalized_name or normalized_name in existing_names:
            continue
        existing_names.add(normalized_name)
        selected_metadata.append(pd.Series(row._asdict()))
        selected_items.append(item.copy())
        if len(selected_items) == len(missing_tasks):
            break
    if len(selected_items) < len(missing_tasks):
        raise ValueError(
            f"Supplement has only {len(selected_items)} usable unique items; "
            f"need {len(missing_tasks)}"
        )

    primary_signature = str(primary_summary["run_signature"])
    primary_metadata = primary_metadata.copy()
    primary_metadata["composition_source"] = "primary"
    primary_metadata["source_run_signature"] = primary_metadata["run_signature"].astype(str)
    remapped_items: list[pd.Series] = []
    remapped_metadata: list[pd.Series] = []
    for task_index, item, metadata in zip(
        missing_tasks, selected_items, selected_metadata, strict=True
    ):
        synthetic_id = -1 - int(task_index)
        item["id"] = synthetic_id
        source_signature = str(metadata["run_signature"])
        metadata["id"] = synthetic_id
        metadata["task_index"] = int(task_index)
        metadata["composition_source"] = "supplement"
        metadata["source_run_signature"] = source_signature
        metadata["run_signature"] = primary_signature
        remapped_items.append(item)
        remapped_metadata.append(metadata)

    combined_items = pd.concat(
        [primary_items, pd.DataFrame(remapped_items)], ignore_index=True
    ).sort_values("id", ascending=False, ignore_index=True)
    combined_metadata = pd.concat(
        [primary_metadata, pd.DataFrame(remapped_metadata)], ignore_index=True
    ).sort_values("task_index", ignore_index=True)
    if len(combined_items) != args.count or len(combined_metadata) != args.count:
        raise ValueError("Composed dataset does not have the requested row count")
    if combined_items["id"].duplicated().any() or combined_metadata["task_index"].duplicated().any():
        raise ValueError("Composed dataset contains duplicate ids/task indices")

    atomic_parquet(combined_items, primary_dir / "items.parquet")
    atomic_parquet(combined_metadata, primary_dir / "generation_metadata.parquet")
    composition = {
        "version": "item_generation_composition_v1",
        "primary_rows": int(len(primary_items)),
        "supplement_rows": int(len(remapped_items)),
        "primary_run_signature": primary_signature,
        "supplement_run_signature": supplement_summary.get("run_signature"),
        "supplement_source_dir": str(supplement_dir),
    }
    summary = {
        **primary_summary,
        "generated": args.count,
        "pending": 0,
        "errors": 0,
        "composition": composition,
    }
    atomic_json(summary, primary_dir / "summary.json")
    report = validate_generated_dataset(
        primary_dir / "items.parquet",
        reference_path=REFERENCE,
        metadata_path=primary_dir / "generation_metadata.parquet",
    )
    atomic_json(report, primary_dir / "validation_report.json")
    if report.get("valid") is not True or int(report.get("rows", -1)) != args.count:
        raise RuntimeError(f"Composed item validation failed: {report}")
    print(json.dumps({**composition, "validation_valid": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
