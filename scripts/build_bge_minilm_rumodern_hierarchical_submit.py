#!/usr/bin/env python3
"""Package the hierarchical BGE/MiniLM/RuModernBERT submission."""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd

import build_bge_minilm_ensemble_submit as common


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "submits/bge-minilm-rumodern-hierarchical-60-5-st-v1"
ARCHIVE = ROOT / "submits/bge-minilm-rumodern-hierarchical-60-5-st-v1.zip"
BGE_MODEL = SUBMIT / "models/bge-reranker-v2-m3-human-ft-v1"
MINILM_MODEL = SUBMIT / "models/minilm-5ep-human-ft-v1"
RU_SOURCE = (
    ROOT
    / "artifacts/kaggle/product-matching-architecture-rumodernbert-v1"
    / "rumodernbert_base_random_head_human_ft_v1"
)
RU_MODEL = SUBMIT / "models/rumodernbert-base-human-ft-v1"
BASE_ROUTER_DIR = ROOT / "artifacts/benefit_router_all_experts_v1"
SEQUENTIAL_ROUTER_DIR = ROOT / "artifacts/rumodern_over_bge_minilm_router_v1"
CONCEPT_AUDIT = (
    ROOT
    / "artifacts/catboost1_negative_router_v2/feature_cache/attribute_concept_map_audit.parquet"
)
EXPECTED_IMAGE = "dinakepech/ecup26-bge-minilm-rumodern-router-sdpa:1.0"
MAX_ARCHIVE_BYTES = 5 * 1024**3
RU_FILES = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}
RU_EXPECTED = {
    "bytes": 598_436_708,
    "sha256": "e6cda247fe02e615bfc1ddc8849a7b82e207dbe603c533cd6b529ed66eb19bce",
}


def copy_exact(source: Path, destination: Path) -> None:
    if not source.is_file() or not source.stat().st_size:
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_ru_model() -> None:
    weight = RU_SOURCE / "model.safetensors"
    if weight.stat().st_size != RU_EXPECTED["bytes"] or common.sha256(weight) != RU_EXPECTED["sha256"]:
        raise RuntimeError("RuModernBERT checkpoint identity mismatch")
    RU_MODEL.mkdir(parents=True, exist_ok=True)
    for name in RU_FILES:
        source = RU_SOURCE / name
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = RU_MODEL / name
        if name == "model.safetensors":
            destination.unlink(missing_ok=True)
            os.link(source, destination)
        else:
            shutil.copy2(source, destination)


def stage() -> None:
    common.validate_weight(common.BGE_SOURCE / "model.safetensors", "bge")
    common.validate_weight(common.MINILM_SOURCE / "model.safetensors", "minilm")
    common.copy_model(common.BGE_SOURCE, BGE_MODEL)
    common.copy_model(common.MINILM_SOURCE, MINILM_MODEL, common.MINILM_TOKENIZER_SOURCE)
    copy_ru_model()
    copy_exact(common.FREQUENCY_SOURCE, SUBMIT / "attribute_name_frequency.csv")
    copy_exact(common.SERIALIZER_SOURCE, SUBMIT / "serialization_ablation.py")
    copy_exact(
        ROOT / "submits/bge-minilm-sentence-transformers-fa2-v1/ensemble_base.py",
        SUBMIT / "ensemble_base.py",
    )
    copy_exact(
        ROOT / "submits/bge-minilm-benefit-router-40-st-v1/run.py",
        SUBMIT / "router_base.py",
    )
    copy_exact(ROOT / "src/catboost1_early_exit.py", SUBMIT / "src/catboost1_early_exit.py")
    (SUBMIT / "src/__init__.py").touch()
    copy_exact(
        BASE_ROUTER_DIR / "models/router_minilm_classification.cbm",
        SUBMIT / "router_minilm_classification.cbm",
    )
    copy_exact(
        SEQUENTIAL_ROUTER_DIR / "router_rumodernbert_over_bge_minilm_classification.cbm",
        SUBMIT / "router_rumodernbert_over_bge_minilm_classification.cbm",
    )

    base_manifest = json.loads((BASE_ROUTER_DIR / "manifest.json").read_text(encoding="utf-8"))
    (SUBMIT / "router_manifest.json").write_text(
        json.dumps(
            {
                "feature_columns": base_manifest["feature_columns"],
                "categorical_columns": base_manifest["categorical_columns"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    sequential_manifest = json.loads(
        (SEQUENTIAL_ROUTER_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    (SUBMIT / "sequential_router_manifest.json").write_text(
        json.dumps(
            {
                "target": sequential_manifest["target"],
                "feature_columns": sequential_manifest["feature_columns"],
                "categorical_columns": sequential_manifest["categorical_columns"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    audit = pd.read_parquet(CONCEPT_AUDIT)
    accepted = audit.loc[audit["accepted"].astype(bool), ["attribute_key", "concept"]]
    if len(accepted) != 1_581 or accepted["attribute_key"].duplicated().any():
        raise RuntimeError("unexpected accepted attribute concept map")
    concept_map = dict(zip(accepted["attribute_key"].astype(str), accepted["concept"].astype(str)))
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
                files[relative] = {"size": path.stat().st_size, "sha256": common.sha256(path)}
    manifest = {
        "experiment": "bge_minilm_rumodern_hierarchical_60_5_st_v1",
        "backend": "sentence-transformers CrossEncoder 5.1.2",
        "attention": "sdpa without fallback",
        "precision": "FP16",
        "serialization": "S2_VALUES_ONLY",
        "max_length": 384,
        "batch_size": 1024,
        "symmetric_inference": True,
        "routing": {
            "mini_coverage": 0.60,
            "rumodern_coverage": 0.05,
            "rumodern_subset_of_minilm": True,
            "mini_weight": 0.50,
            "rumodern_weight_on_current_score": 0.50,
        },
        "offline_validation": {
            "iid_macro_ap": 0.7940336037270487,
            "hard_macro_ap": 0.37525634730565444,
            "ood_macro_ap": 0.6434387308733526,
            "mean_macro_ap": 0.6042428939686851,
        },
        "docker_image": EXPECTED_IMAGE,
        "weights": {**common.EXPECTED, "rumodernbert": RU_EXPECTED},
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
        "metadata.json", "run.py", "router_base.py", "SUBMISSION_MANIFEST.json",
        "router_minilm_classification.cbm",
        "router_rumodernbert_over_bge_minilm_classification.cbm",
        "models/bge-reranker-v2-m3-human-ft-v1/model.safetensors",
        "models/minilm-5ep-human-ft-v1/model.safetensors",
        "models/rumodernbert-base-human-ft-v1/model.safetensors",
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

