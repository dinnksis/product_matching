#!/usr/bin/env python3
"""Download and validate large runtime assets before Docker build."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "docker/embedding-catboost-runtime/models/qwen3-embedding-0.6b"
REPOSITORY = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
REQUIRED = (
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "modules.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
    "1_Pooling/config.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate() -> None:
    missing = [name for name in REQUIRED if not (MODEL_DIR / name).is_file()]
    empty = [name for name in REQUIRED if (MODEL_DIR / name).is_file() and not (MODEL_DIR / name).stat().st_size]
    if missing or empty:
        raise RuntimeError(f"Incomplete Qwen snapshot; missing={missing}, empty={empty}")
    config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3" or config.get("hidden_size") != 1024:
        raise RuntimeError("Unexpected Qwen config")
    weights = MODEL_DIR / "model.safetensors"
    if weights.stat().st_size < 1_000_000_000:
        raise RuntimeError(f"Qwen weights are too small: {weights.stat().st_size:,} bytes")
    pooling = json.loads((MODEL_DIR / "1_Pooling/config.json").read_text(encoding="utf-8"))
    if not (
        pooling.get("pooling_mode_lasttoken")
        or pooling.get("pooling_mode") == "lasttoken"
    ):
        raise RuntimeError("Qwen SentenceTransformer snapshot does not use last-token pooling")
    manifest = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "files": {
            name: {"bytes": (MODEL_DIR / name).stat().st_size, "sha256": sha256(MODEL_DIR / name)}
            for name in REQUIRED
        },
    }
    (MODEL_DIR / "runtime_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Validated Qwen snapshot: {MODEL_DIR}")
    print(f"Weights: {weights.stat().st_size:,} bytes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=REPOSITORY,
            revision=REVISION,
            local_dir=MODEL_DIR,
            allow_patterns=list(REQUIRED),
            max_workers=8,
        )
    validate()


if __name__ == "__main__":
    main()
