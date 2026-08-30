#!/usr/bin/env python3
"""Package the frozen fast MiniLM-gated BGE ensemble."""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

import build_bge_minilm_ensemble_submit as common


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "submits/bge-minilm-minilm-gated-v1"
ARCHIVE = ROOT / "submits/bge-minilm-minilm-gated-v1.zip"
BGE_MODEL = SUBMIT / "models/bge-reranker-v2-m3-human-ft-v1"
MINILM_MODEL = SUBMIT / "models/minilm-5ep-human-ft-v1"
MAX_ARCHIVE_BYTES = 5 * 1024**3


def stage() -> None:
    common.validate_weight(common.BGE_SOURCE / "model.safetensors", "bge")
    common.validate_weight(common.MINILM_SOURCE / "model.safetensors", "minilm")
    common.copy_model(common.BGE_SOURCE, BGE_MODEL)
    common.copy_model(
        common.MINILM_SOURCE, MINILM_MODEL, common.MINILM_TOKENIZER_SOURCE
    )
    shutil.copy2(common.FREQUENCY_SOURCE, SUBMIT / "attribute_name_frequency.csv")
    shutil.copy2(common.SERIALIZER_SOURCE, SUBMIT / "serialization_ablation.py")
    shutil.copy2(
        ROOT / "submits/bge-minilm-rank-ensemble-v1/run.py",
        SUBMIT / "ensemble_base.py",
    )


def write_manifest() -> None:
    metadata = json.loads((SUBMIT / "metadata.json").read_text(encoding="utf-8"))
    expected_metadata = {
        "image": common.EXPECTED_IMAGE,
        "entry_point": "python -u run.py",
    }
    if metadata != expected_metadata:
        raise RuntimeError("unexpected metadata.json")
    files = {}
    for path in sorted(SUBMIT.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            relative = path.relative_to(SUBMIT).as_posix()
            if relative != "SUBMISSION_MANIFEST.json":
                files[relative] = {
                    "size": path.stat().st_size,
                    "sha256": common.sha256(path),
                }
    manifest = {
        "experiment": "bge_minilm_minilm_gated_v1",
        "routing": {
            "first_stage": "MiniLM on every pair",
            "bge_condition": "minilm_probability > 0.0602078252",
            "threshold_source": "ordinary validation MiniLM median",
        },
        "aggregation": {
            "not_routed": "MiniLM probability",
            "routed": "mean probability of MiniLM and BGE",
        },
        "serialization": "S2_VALUES_ONLY",
        "max_length": 384,
        "symmetric_inference": True,
        "precision": "FP16 autocast",
        "inference_pipeline": "pretokenized exact-length bucketing with dynamic padding",
        "offline_validation": {
            "ordinary_macro_ap": 0.7918406030,
            "hard_macro_ap": 0.3731689689,
            "ood_macro_ap": 0.6392013654,
            "mean_macro_ap": 0.6014036458,
            "bge_fraction": {
                "ordinary": 0.5,
                "hard": 0.5663914689,
                "ood": 0.4664448277,
            },
        },
        "docker_image": common.EXPECTED_IMAGE,
        "weights": common.EXPECTED,
        "files": files,
    }
    (SUBMIT / "SUBMISSION_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def archive() -> None:
    ARCHIVE.unlink(missing_ok=True)
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
                path, path.relative_to(SUBMIT).as_posix(), compress_type=compression
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
        "ensemble_base.py",
        "SUBMISSION_MANIFEST.json",
        "models/bge-reranker-v2-m3-human-ft-v1/model.safetensors",
        "models/minilm-5ep-human-ft-v1/model.safetensors",
    }
    if not required.issubset(names):
        raise RuntimeError(f"submission ZIP is missing: {sorted(required - names)}")
    print(f"Created {ARCHIVE} ({ARCHIVE.stat().st_size:,} bytes)")
    print(f"SHA-256: {common.sha256(ARCHIVE)}")


def main() -> int:
    stage()
    write_manifest()
    archive()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
