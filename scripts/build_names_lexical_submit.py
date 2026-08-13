#!/usr/bin/env python3
"""Build the names-only lexical CatBoost control submission."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODEL = (
    ROOT
    / "artifacts/kaggle/product-matching-qwen-embedding-boosting/embedding_boosting"
    / "01_names_lexical/model.cbm"
)
SUBMIT = ROOT / "submits/names-lexical-catboost"
ARCHIVE = ROOT / "submits/names-lexical-catboost.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not SOURCE_MODEL.is_file():
        raise FileNotFoundError(SOURCE_MODEL)
    model_dir = SUBMIT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_MODEL, model_dir / "matching_model.cbm")
    metadata = json.loads((SUBMIT / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("image") != "dinakepech/ecup26-embedding-catboost:1.1":
        raise RuntimeError("unexpected Docker image")
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with zipfile.ZipFile(ARCHIVE, "w", allowZip64=True) as output:
        for path in sorted(SUBMIT.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            compression = zipfile.ZIP_STORED if path.suffix == ".cbm" else zipfile.ZIP_DEFLATED
            output.write(path, path.relative_to(SUBMIT).as_posix(), compress_type=compression)
    with zipfile.ZipFile(ARCHIVE) as source:
        if source.testzip():
            raise RuntimeError("broken ZIP")
        required = {"metadata.json", "run.py", "models/matching_model.cbm"}
        if not required.issubset(source.namelist()):
            raise RuntimeError("submission files are missing")
    print(f"Created {ARCHIVE} ({ARCHIVE.stat().st_size:,} bytes)")
    print(f"SHA-256: {sha256(ARCHIVE)}")


if __name__ == "__main__":
    main()
