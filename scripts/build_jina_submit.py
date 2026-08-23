#!/usr/bin/env python3
"""Download pinned Jina assets and build an autonomous submission ZIP."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMIT_DIR = ROOT / "submits" / "jina-reranker-v2-zero-shot"
MODEL_DIR = SUBMIT_DIR / "models" / "jina-reranker-v2-base-multilingual"
ARCHIVE_PATH = ROOT / "submits" / "jina-reranker-v2-zero-shot.zip"
REPOSITORY = "jinaai/jina-reranker-v2-base-multilingual"
REVISION = "9cfeff2df7d40d1b78e75e5e9cebec92a99813c9"
WEIGHT_SIZE = 556_892_306
WEIGHT_SHA256 = "ab2595ab9f34bdeffe645431d64c6e4aabe2ff5a57cfcacfef0727a97434238f"
MODEL_FILES = (
    "README.md", "block.py", "config.json", "configuration_xlm_roberta.py",
    "embedding.py", "mha.py", "mlp.py", "model.safetensors", "modeling_xlm_roberta.py",
    "special_tokens_map.json", "stochastic_depth.py", "tokenizer.json",
    "tokenizer_config.json", "xlm_padding.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_assets(force: bool) -> None:
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        destination = MODEL_DIR / name
        complete = destination.is_file() and (
            name != "model.safetensors" or destination.stat().st_size == WEIGHT_SIZE
        )
        if complete and not force:
            continue
        if force and destination.exists():
            destination.unlink()
        if name == "model.safetensors" and destination.exists() and destination.stat().st_size > WEIGHT_SIZE:
            destination.unlink()
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
        partial = destination.with_suffix(destination.suffix + ".part")
        if destination.exists():
            destination.replace(partial)
        if partial.exists() and name == "model.safetensors" and partial.stat().st_size > WEIGHT_SIZE:
            partial.unlink()
        result = subprocess.run(
            [curl, "-fL", "--retry", "8", "--retry-all-errors", "--retry-delay", "3", "--continue-at", "-", "-o", str(partial), url],
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"failed to download {name}")
        if name == "model.safetensors" and partial.stat().st_size != WEIGHT_SIZE:
            raise RuntimeError(
                f"incomplete Jina weights: {partial.stat().st_size}; expected {WEIGHT_SIZE}"
            )
        partial.replace(destination)


def verify() -> None:
    missing = [name for name in MODEL_FILES if not (MODEL_DIR / name).is_file()]
    if missing:
        raise RuntimeError(f"missing model files: {missing}")
    weights = MODEL_DIR / "model.safetensors"
    if weights.stat().st_size != WEIGHT_SIZE or sha256(weights) != WEIGHT_SHA256:
        raise RuntimeError("Jina model.safetensors integrity check failed")
    json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))


def build_archive() -> None:
    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()
    with zipfile.ZipFile(ARCHIVE_PATH, "w", allowZip64=True) as archive:
        for path in sorted(SUBMIT_DIR.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            compression = zipfile.ZIP_STORED if path.suffix == ".safetensors" else zipfile.ZIP_DEFLATED
            archive.write(path, path.relative_to(SUBMIT_DIR).as_posix(), compress_type=compression)
    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        broken = archive.testzip()
        names = set(archive.namelist())
    if broken:
        raise RuntimeError(f"broken ZIP member: {broken}")
    if not {"metadata.json", "run.py"}.issubset(names):
        raise RuntimeError("metadata.json and run.py must be at ZIP root")
    print(f"Created {ARCHIVE_PATH} ({ARCHIVE_PATH.stat().st_size:,} bytes)")
    print(f"SHA-256: {sha256(ARCHIVE_PATH)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-only", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    if not args.archive_only:
        download_assets(args.force_download)
    verify()
    build_archive()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
