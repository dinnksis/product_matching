#!/usr/bin/env python3
"""Package BGE plus the frozen CatBoost-routed MiniLM 40% submission."""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd

import build_bge_minilm_ensemble_submit as common


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "submits/bge-minilm-benefit-router-40-st-v1"
ARCHIVE = ROOT / "submits/bge-minilm-benefit-router-40-st-v1.zip"
BGE_MODEL = SUBMIT / "models/bge-reranker-v2-m3-human-ft-v1"
MINILM_MODEL = SUBMIT / "models/minilm-5ep-human-ft-v1"
ROUTER_DIR = ROOT / "artifacts/benefit_router_bge_minilm_v1"
ROUTER_MODEL = ROUTER_DIR / "models/router_minilm_classification.cbm"
ROUTER_MANIFEST = ROUTER_DIR / "manifest.json"
CONCEPT_AUDIT = (
    ROOT
    / "artifacts/catboost1_negative_router_v2/feature_cache/attribute_concept_map_audit.parquet"
)
EXPECTED_IMAGE = "dinakepech/ecup26-bge-minilm-benefit-router-sdpa:1.0"
MAX_ARCHIVE_BYTES = 5 * 1024**3


def copy_exact(source: Path, destination: Path) -> None:
    if not source.is_file() or not source.stat().st_size:
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def stage() -> None:
    common.validate_weight(common.BGE_SOURCE / "model.safetensors", "bge")
    common.validate_weight(common.MINILM_SOURCE / "model.safetensors", "minilm")
    common.copy_model(common.BGE_SOURCE, BGE_MODEL)
    common.copy_model(common.MINILM_SOURCE, MINILM_MODEL, common.MINILM_TOKENIZER_SOURCE)
    copy_exact(common.FREQUENCY_SOURCE, SUBMIT / "attribute_name_frequency.csv")
    copy_exact(common.SERIALIZER_SOURCE, SUBMIT / "serialization_ablation.py")
    copy_exact(
        ROOT / "submits/bge-minilm-sentence-transformers-fa2-v1/ensemble_base.py",
        SUBMIT / "ensemble_base.py",
    )
    copy_exact(ROUTER_MODEL, SUBMIT / "router_minilm_classification.cbm")
    copy_exact(ROOT / "src/catboost1_early_exit.py", SUBMIT / "src/catboost1_early_exit.py")
    (SUBMIT / "src/__init__.py").touch()

    source_manifest = json.loads(ROUTER_MANIFEST.read_text(encoding="utf-8"))
    router_manifest = {
        "training_source": source_manifest["training_source"],
        "component_disjoint_router_oof": source_manifest[
            "component_disjoint_router_oof"
        ],
        "specialist": "minilm",
        "router_kind": "classification benefit",
        "feature_columns": source_manifest["feature_columns"],
        "categorical_columns": source_manifest["categorical_columns"],
        "source_hashes": source_manifest["source_hashes"],
    }
    (SUBMIT / "router_manifest.json").write_text(
        json.dumps(router_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    audit = pd.read_parquet(CONCEPT_AUDIT)
    required = {"attribute_key", "concept", "accepted"}
    if required - set(audit.columns):
        raise RuntimeError("attribute concept audit has an unexpected schema")
    accepted = audit.loc[audit["accepted"].astype(bool), ["attribute_key", "concept"]]
    if accepted["attribute_key"].duplicated().any() or len(accepted) != 1_581:
        raise RuntimeError(f"unexpected accepted concept aliases: {len(accepted)}")
    concept_map = dict(
        zip(accepted["attribute_key"].astype(str), accepted["concept"].astype(str))
    )
    (SUBMIT / "attribute_concept_map.json").write_text(
        json.dumps(concept_map, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_manifest() -> None:
    metadata = json.loads((SUBMIT / "metadata.json").read_text(encoding="utf-8"))
    if metadata != {"image": EXPECTED_IMAGE, "entry_point": "python -u run.py"}:
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
        "experiment": "bge_minilm_benefit_router_40_st_v1",
        "backend": "sentence-transformers CrossEncoder 5.1.2",
        "attention": "sdpa without fallback",
        "precision": "FP16",
        "serialization": "S2_VALUES_ONLY",
        "max_length": 384,
        "batch_size": 1024,
        "symmetric_inference": True,
        "routing": {
            "model": "frozen CatBoost classification benefit-router",
            "coverage": 0.40,
            "selection": "top predicted MiniLM benefit with deterministic ID tie-break",
            "features": 134,
        },
        "aggregation": "BGE probability; routed rows use 50/50 probability blend",
        "docker_image": EXPECTED_IMAGE,
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
            compression = zipfile.ZIP_STORED if path.suffix == ".safetensors" else zipfile.ZIP_DEFLATED
            output.write(path, path.relative_to(SUBMIT).as_posix(), compress_type=compression)
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
        "router_minilm_classification.cbm",
        "router_manifest.json",
        "attribute_concept_map.json",
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

