#!/usr/bin/env python3
"""Assemble the trained MiniLM S2 checkpoint into a competition ZIP."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODEL = (
    ROOT
    / "artifacts/kaggle/product-matching-minilm-serialization-ablation"
    / "serialization_ablation/checkpoints/S2_VALUES_ONLY"
)
SOURCE_FREQUENCY = ROOT / "prepared/serialization_ablation/attribute_name_frequency.csv"
SOURCE_SERIALIZER = ROOT / "src/serialization_ablation.py"
SUBMIT = ROOT / "submits/minilm-s2-values-only"
MODEL = SUBMIT / "models/minilm-s2-values-only"
ARCHIVE = ROOT / "submits/minilm-s2-values-only.zip"
EXPECTED_IMAGE = "dinakepech/ecup26-minilm-s2:1.0"
MAX_ARCHIVE_BYTES = 5 * 1024**3
REQUIRED_MODEL_FILES = {
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_sources() -> None:
    missing = [name for name in REQUIRED_MODEL_FILES if not (SOURCE_MODEL / name).is_file()]
    if missing:
        raise FileNotFoundError(f"S2 checkpoint is incomplete; missing: {sorted(missing)}")
    weights = SOURCE_MODEL / "model.safetensors"
    if weights.stat().st_size < 100 * 1024 * 1024:
        raise RuntimeError(f"S2 model.safetensors is incomplete: {weights.stat().st_size} bytes")
    MODEL.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_MODEL_FILES:
        shutil.copy2(SOURCE_MODEL / name, MODEL / name)
    shutil.copy2(SOURCE_FREQUENCY, SUBMIT / SOURCE_FREQUENCY.name)
    shutil.copy2(SOURCE_SERIALIZER, SUBMIT / "serialization_ablation.py")


def verify_and_manifest() -> dict[str, object]:
    metadata = json.loads((SUBMIT / "metadata.json").read_text(encoding="utf-8"))
    if metadata != {"image": EXPECTED_IMAGE, "entry_point": "python -u run.py"}:
        raise RuntimeError("metadata.json does not match the expected image/entry point")
    with (SUBMIT / "attribute_name_frequency.csv").open("r", encoding="utf-8") as source:
        row_count = sum(1 for _ in source) - 1
    if row_count != 24_916:
        raise RuntimeError(f"unexpected attribute frequency row count: {row_count}")
    files: dict[str, dict[str, object]] = {}
    for path in sorted(SUBMIT.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            relative = path.relative_to(SUBMIT).as_posix()
            if relative == "SUBMISSION_MANIFEST.json":
                continue
            files[relative] = {"size": path.stat().st_size, "sha256": sha256(path)}
    manifest: dict[str, object] = {
        "experiment": "minilm_serialization_ablation_s2_values_only",
        "training_data": "120000 human-labelled screening pairs",
        "validation_macro_average_precision": 0.6901528206662118,
        "base_model": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "base_model_revision": "1427fd652930e4ba29e8149678df786c240d8825",
        "docker_image": EXPECTED_IMAGE,
        "files": files,
    }
    (SUBMIT / "SUBMISSION_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_archive() -> None:
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with zipfile.ZipFile(ARCHIVE, "w", allowZip64=True) as output:
        for path in sorted(SUBMIT.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            compression = zipfile.ZIP_STORED if path.suffix == ".safetensors" else zipfile.ZIP_DEFLATED
            output.write(path, path.relative_to(SUBMIT).as_posix(), compress_type=compression)
    if ARCHIVE.stat().st_size > MAX_ARCHIVE_BYTES:
        raise RuntimeError(f"submission ZIP exceeds 5 GiB: {ARCHIVE.stat().st_size}")
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
        "models/minilm-s2-values-only/model.safetensors",
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
