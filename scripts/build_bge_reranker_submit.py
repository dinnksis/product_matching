#!/usr/bin/env python3
"""Validate and package the downloaded fine-tuned BGE checkpoint."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODEL = (
    ROOT
    / "artifacts/kaggle/product-matching-architecture-bge-v2-m3-v1"
    / "bge_reranker_v2_m3_human_ft_v1"
)
SOURCE_FREQUENCY = ROOT / "prepared/serialization_ablation/attribute_name_frequency.csv"
SOURCE_SERIALIZER = ROOT / "src/serialization_ablation.py"
SUBMIT = ROOT / "submits/bge-reranker-v2-m3-human-ft-v1"
MODEL = SUBMIT / "models/bge-reranker-v2-m3-human-ft-v1"
ARCHIVE = ROOT / "submits/bge-reranker-v2-m3-human-ft-v1.zip"
EXPECTED_IMAGE = "dinakepech/ecup26-bge-reranker-v2-m3:1.0"
EXPECTED_WEIGHT_BYTES = 2_271_071_852
EXPECTED_WEIGHT_SHA256 = "0e4e8e0b9dd5f220fd0423a5a27d8d6c09ff3697e72fbbf16f861f03df02b87e"
EXPECTED_FREQUENCY_ROWS = 24_916
MAX_ARCHIVE_BYTES = 5 * 1024**3
REQUIRED_MODEL_FILES = {
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "special_tokens_map.json",
}
OPTIONAL_MODEL_FILES = {
    "added_tokens.json",
    "sentencepiece.bpe.model",
    "tokenizer.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_sources() -> None:
    SUBMIT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_FREQUENCY, SUBMIT / "attribute_name_frequency.csv")
    shutil.copy2(SOURCE_SERIALIZER, SUBMIT / "serialization_ablation.py")
    missing = sorted(name for name in REQUIRED_MODEL_FILES if not (SOURCE_MODEL / name).is_file())
    if missing:
        raise FileNotFoundError(
            f"BGE checkpoint is incomplete; missing files in {SOURCE_MODEL}: {missing}"
        )
    if not any((SOURCE_MODEL / name).is_file() for name in {"tokenizer.json", "sentencepiece.bpe.model"}):
        raise FileNotFoundError(
            "BGE tokenizer vocabulary is missing; expected tokenizer.json or sentencepiece.bpe.model"
        )
    weights = SOURCE_MODEL / "model.safetensors"
    if weights.stat().st_size != EXPECTED_WEIGHT_BYTES:
        raise RuntimeError(
            f"unexpected BGE weight size: {weights.stat().st_size}; expected {EXPECTED_WEIGHT_BYTES}"
        )
    actual_hash = sha256(weights)
    if actual_hash != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError(
            f"unexpected BGE weight SHA-256: {actual_hash}; expected {EXPECTED_WEIGHT_SHA256}"
        )
    MODEL.mkdir(parents=True, exist_ok=True)
    for name in sorted(REQUIRED_MODEL_FILES | OPTIONAL_MODEL_FILES):
        source = SOURCE_MODEL / name
        if source.is_file():
            shutil.copy2(source, MODEL / name)


def verify_and_manifest() -> dict[str, object]:
    metadata = json.loads((SUBMIT / "metadata.json").read_text(encoding="utf-8"))
    expected_metadata = {"image": EXPECTED_IMAGE, "entry_point": "python -u run.py"}
    if metadata != expected_metadata:
        raise RuntimeError("metadata.json does not match the expected image/entry point")
    with (SUBMIT / "attribute_name_frequency.csv").open("r", encoding="utf-8") as source:
        frequency_rows = sum(1 for _ in source) - 1
    if frequency_rows != EXPECTED_FREQUENCY_ROWS:
        raise RuntimeError(f"unexpected attribute frequency row count: {frequency_rows}")
    files: dict[str, dict[str, object]] = {}
    for path in sorted(SUBMIT.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            relative = path.relative_to(SUBMIT).as_posix()
            if relative == "SUBMISSION_MANIFEST.json":
                continue
            files[relative] = {"size": path.stat().st_size, "sha256": sha256(path)}
    manifest: dict[str, object] = {
        "experiment": "bge_reranker_v2_m3_human_ft_v1",
        "training_data": "306669 frozen human training pairs, one epoch",
        "validation_macro_average_precision": {
            "ordinary": 0.7822219461825138,
            "hard": 0.3759746515593398,
            "ood": 0.6412704537924441,
        },
        "base_model": "BAAI/bge-reranker-v2-m3",
        "serialization": "S2_VALUES_ONLY",
        "max_length": 384,
        "symmetric_inference": True,
        "precision": "FP16 autocast",
        "docker_image": EXPECTED_IMAGE,
        "files": files,
    }
    (SUBMIT / "SUBMISSION_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_archive() -> None:
    ARCHIVE.unlink(missing_ok=True)
    with zipfile.ZipFile(ARCHIVE, "w", allowZip64=True) as output:
        for path in sorted(SUBMIT.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            compression = zipfile.ZIP_STORED if path.suffix == ".safetensors" else zipfile.ZIP_DEFLATED
            output.write(
                path,
                path.relative_to(SUBMIT).as_posix(),
                compress_type=compression,
            )
    if ARCHIVE.stat().st_size > MAX_ARCHIVE_BYTES:
        raise RuntimeError(f"submission ZIP exceeds 5 GiB: {ARCHIVE}")
    with zipfile.ZipFile(ARCHIVE) as source:
        broken = source.testzip()
        names = set(source.namelist())
    if broken:
        raise RuntimeError(f"broken ZIP member: {broken}")
    required = {
        "metadata.json",
        "run.py",
        "serialization_ablation.py",
        "attribute_name_frequency.csv",
        "SUBMISSION_MANIFEST.json",
        "models/bge-reranker-v2-m3-human-ft-v1/model.safetensors",
    }
    if not required.issubset(names):
        raise RuntimeError(f"submission ZIP is missing: {sorted(required - names)}")
    print(f"Created {ARCHIVE} ({ARCHIVE.stat().st_size:,} bytes)")
    print(f"SHA-256: {sha256(ARCHIVE)}")


def main() -> int:
    copy_sources()
    verify_and_manifest()
    build_archive()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
