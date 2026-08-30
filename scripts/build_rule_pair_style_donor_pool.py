from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "item_pipeline" / "artifacts" / "generated" / "items.parquet"
DEFAULT_OUTPUT_DIR = (
    ROOT / "item_pipeline" / "artifacts" / "generated_style_donors_x2_v1"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_pool(source_path: Path, output_dir: Path, copies: int) -> dict[str, object]:
    if copies < 1:
        raise ValueError("copies must be positive")
    source = pd.read_parquet(source_path)
    required = ["id", "name", "attributes", "category"]
    if source.columns.tolist() != required:
        raise ValueError(f"Unexpected source columns: {source.columns.tolist()}")
    if source.empty or source["id"].duplicated().any():
        raise ValueError("Source donors must be non-empty with unique IDs")

    minimum_id = int(source["id"].min())
    maximum_id = int(source["id"].max())
    span = maximum_id - minimum_id + 1
    frames: list[pd.DataFrame] = []
    for copy_index in range(copies):
        frame = source.copy()
        frame["id"] = frame["id"].astype("int64") - copy_index * span
        frames.append(frame)
    donors = pd.concat(frames, ignore_index=True)
    if donors["id"].duplicated().any():
        raise ValueError("Virtual donor IDs overlap")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "items.parquet"
    donors.to_parquet(output_path, index=False)
    manifest = {
        "version": "generated_style_donors_virtual_id_copies_v1",
        "source_path": str(source_path.resolve()),
        "source_sha256": sha256_file(source_path),
        "source_rows": int(len(source)),
        "copies": int(copies),
        "id_offset_span": int(span),
        "rows": int(len(donors)),
        "unique_ids": int(donors["id"].nunique()),
        "category_counts": {
            str(key): int(value)
            for key, value in donors["category"].value_counts().sort_index().items()
        },
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create virtual-ID copies of style donors for balanced rule schedules"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--copies", type=int, default=2)
    args = parser.parse_args()
    manifest = build_pool(args.source, args.output_dir, args.copies)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
