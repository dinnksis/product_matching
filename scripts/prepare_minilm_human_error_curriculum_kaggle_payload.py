"""Prepare the minimal Kaggle Dataset payload for the MiniLM curriculum run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "data"
    / "minilm_human_error_curriculum_v1"
    / "human_error_curriculum_pairs.parquet"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "kaggle_datasets" / "minilm_human_error_curriculum_v1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a minimal Kaggle curriculum Dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--owner", default="dinakepecheva")
    parser.add_argument("--slug", default="product-matching-minilm-human-error-curriculum-v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(source)
    if list(frame.columns) != ["id1", "id2", "target"]:
        raise ValueError(f"Unexpected curriculum columns: {frame.columns.tolist()}")
    if len(frame) != 9311:
        raise ValueError(f"Expected 9311 curriculum rows, got {len(frame)}")
    destination = output / "human_error_curriculum_pairs.parquet"
    shutil.copy2(source, destination)
    metadata = {
        "title": "Product Matching MiniLM Human Error Curriculum v1",
        "id": f"{args.owner}/{args.slug}",
        "licenses": [{"name": "CC0-1.0"}],
        "description": (
            "Human-only RULE_DISCOVERY OOF-error and hard-pair curriculum for a "
            "frozen MiniLM 5ep data ablation; no synthetic data or Qwen labels."
        ),
    }
    (output / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "version": "minilm_human_error_curriculum_v1",
        "pairs": len(frame),
        "positive_pairs": int(frame["target"].sum()),
        "negative_pairs": int(frame["target"].eq(0).sum()),
        "human_labels_only": True,
        "synthetic_pairs_used": False,
        "qwen_labels_used": False,
        "file": destination.name,
        "sha256": sha256(destination),
    }
    # Keep the audit manifest beside the upload directory.  On Windows some
    # Kaggle CLI versions incorrectly stage arbitrary JSON payload files as
    # ``*.json.json``; the Dataset itself needs only parquet + dataset metadata.
    (output.parent / "minilm_human_error_curriculum_v1_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"payload": str(output), "dataset": metadata["id"], **manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
