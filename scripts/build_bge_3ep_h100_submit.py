#!/usr/bin/env python3
"""Assemble and verify the exact trained BGE checkpoint submission ZIP."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODEL = ROOT / "artifacts" / "server_bge_3ep_h100" / "run"
SUBMIT = ROOT / "submits" / "bge-reranker-v2-m3-3ep-h100"
MODEL = SUBMIT / "models" / "bge-reranker-v2-m3-3ep-h100"
ARCHIVE = ROOT / "submits" / "bge-reranker-v2-m3-3ep-h100.zip"
EXPECTED_IMAGE = "dinakepech/ecup26-minilm-s2:1.0"
MAX_ARCHIVE_BYTES = 5 * 1024**3
EXPECTED_MODEL_FILES: dict[str, dict[str, object]] = {
    "config.json": {
        "size": 764,
        "sha256": "3f143138299caf72270c79814d1f0e3c38fc168e2d30f1e6cc4c70f6cdb481f5",
    },
    "tokenizer.json": {
        "size": 17_082_998,
        "sha256": "efcbc2e883c0ef3f67ce522db4e6e4e114532aeb827b5e2cbfa89bac7da8252f",
    },
    "tokenizer_config.json": {
        "size": 1_207,
        "sha256": "272dabf15417f68de99532ca6a2f7c402afdf654d4dc4134f35db97af1edc8b0",
    },
    "special_tokens_map.json": {
        "size": 964,
        "sha256": "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835",
    },
    "model.safetensors": {
        "size": 2_271_071_852,
        "sha256": "d7e899ea3cd305db970aa6f3466eb71a138ad418c74b8b6ac730d1828c4a4ab8",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source() -> None:
    for name, expected in EXPECTED_MODEL_FILES.items():
        path = SOURCE_MODEL / name
        if not path.is_file():
            raise FileNotFoundError(f"trained BGE checkpoint is missing {name}")
        actual = {"size": path.stat().st_size, "sha256": sha256(path)}
        if actual != expected:
            raise RuntimeError(f"trained BGE bytes changed for {name}: {actual}")


def copy_model() -> None:
    MODEL.mkdir(parents=True, exist_ok=True)
    for name in EXPECTED_MODEL_FILES:
        shutil.copy2(SOURCE_MODEL / name, MODEL / name)


def verify_and_manifest() -> dict[str, object]:
    metadata = json.loads((SUBMIT / "metadata.json").read_text(encoding="utf-8"))
    expected_metadata = {
        "image": EXPECTED_IMAGE,
        "entry_point": "python -u run.py",
    }
    if metadata != expected_metadata:
        raise RuntimeError("metadata.json does not match the verified runtime image")

    config = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
    if config.get("architectures") != ["XLMRobertaForSequenceClassification"]:
        raise RuntimeError("checkpoint architecture changed")
    if config.get("num_hidden_layers") != 24 or config.get("hidden_size") != 1024:
        raise RuntimeError("checkpoint geometry changed")

    files: dict[str, dict[str, object]] = {}
    for path in sorted(SUBMIT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(SUBMIT).as_posix()
        if relative == "SUBMISSION_MANIFEST.json":
            continue
        files[relative] = {"size": path.stat().st_size, "sha256": sha256(path)}

    manifest: dict[str, object] = {
        "experiment": "server_bge_3ep_h100",
        "checkpoint_source": "artifacts/server_bge_3ep_h100/run",
        "model_family": "BAAI/bge-reranker-v2-m3",
        "base_model_revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "training": "3 supervised epochs on 347840 human-labelled pairs",
        "validation": {
            "saved_symmetric_iid_macro_ap": 0.8249745146613486,
            "saved_symmetric_hard_macro_ap": 0.46114816821596377,
            "single_pass_longer_first_iid_macro_ap": 0.8236290179077869,
            "single_pass_longer_first_hard_macro_ap": 0.46263071720675114,
        },
        "inference": {
            "orientations": 1,
            "orientation_rule": "longer serialized product first",
            "max_length": 384,
            "initial_pair_batch_size": 512,
            "dtype_on_h100": "bfloat16",
            "attention": "transformers SDPA with native PyTorch Flash-SDPA enabled",
            "length_sorted": True,
            "tokenization_gpu_overlap": True,
        },
        "docker_image": EXPECTED_IMAGE,
        "files": files,
    }
    (SUBMIT / "SUBMISSION_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_archive() -> None:
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with zipfile.ZipFile(ARCHIVE, "w", allowZip64=True) as output:
        for path in sorted(SUBMIT.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            compression = (
                zipfile.ZIP_STORED
                if path.suffix == ".safetensors"
                else zipfile.ZIP_DEFLATED
            )
            output.write(
                path,
                path.relative_to(SUBMIT).as_posix(),
                compress_type=compression,
            )
    if ARCHIVE.stat().st_size > MAX_ARCHIVE_BYTES:
        raise RuntimeError(f"submission ZIP exceeds 5 GiB: {ARCHIVE.stat().st_size}")

    with zipfile.ZipFile(ARCHIVE) as source:
        broken = source.testzip()
        names = set(source.namelist())
        archived_weight = source.getinfo(
            "models/bge-reranker-v2-m3-3ep-h100/model.safetensors"
        )
    if broken:
        raise RuntimeError(f"broken ZIP member: {broken}")
    required = {
        "metadata.json",
        "run.py",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "SUBMISSION_MANIFEST.json",
        "models/bge-reranker-v2-m3-3ep-h100/config.json",
        "models/bge-reranker-v2-m3-3ep-h100/model.safetensors",
        "models/bge-reranker-v2-m3-3ep-h100/tokenizer.json",
        "models/bge-reranker-v2-m3-3ep-h100/tokenizer_config.json",
        "models/bge-reranker-v2-m3-3ep-h100/special_tokens_map.json",
    }
    if not required.issubset(names):
        raise RuntimeError(f"submission ZIP is missing: {sorted(required - names)}")
    if archived_weight.compress_type != zipfile.ZIP_STORED:
        raise RuntimeError("model.safetensors must be stored without lossy/redundant compression")

    print(f"Created {ARCHIVE} ({ARCHIVE.stat().st_size:,} bytes)")
    print(f"SHA-256: {sha256(ARCHIVE)}")


def main() -> int:
    verify_source()
    copy_model()
    verify_and_manifest()
    build_archive()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
