#!/usr/bin/env python3
"""Package the compact one-way BGE plus routed MiniLM submission."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import build_bge_minilm_ensemble_submit as common


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "submits/bge-minilm-fast-oneway-router40-st-v2"
ARCHIVE = ROOT / "submits/bge-minilm-fast-oneway-router40-st-v2.zip"
BGE_MODEL = SUBMIT / "models/bge-reranker-v2-m3-human-ft-v1"
MINILM_MODEL = SUBMIT / "models/minilm-5ep-human-ft-v1"
ROUTER_MODEL = ROOT / "artifacts/fast_oneway_benefit_router_v1/models/router_score_title.cbm"
ROUTER_REPORT = ROOT / "reports/fast_oneway_benefit_router_v1/main_table.csv"
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
    copy_exact(ROOT / "src/fast_benefit_router.py", SUBMIT / "fast_benefit_router.py")
    copy_exact(ROUTER_MODEL, SUBMIT / "router_score_title.cbm")
    manifest = json.loads(
        (ROOT / "artifacts/fast_oneway_benefit_router_v1/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    variant = manifest["variants"]["score_title"]
    (SUBMIT / "router_manifest.json").write_text(
        json.dumps(
            {
                "variant": "score_title",
                "target": "classification logloss benefit, MiniLM_AB over BGE_AB",
                "coverage": 0.40,
                "feature_columns": variant["feature_columns"],
                "categorical_columns": ["category"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
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
        "experiment": "bge_minilm_fast_oneway_router40_st_v2",
        "backend": "sentence-transformers CrossEncoder 5.1.2",
        "attention": "sdpa without fallback",
        "precision": "FP16",
        "serialization": "S2_VALUES_ONLY",
        "max_length": 384,
        "batch_size": 1024,
        "direction": "AB only",
        "routing": {
            "model": "compact CatBoost classification benefit-router",
            "features": 15,
            "coverage": 0.40,
        },
        "aggregation": "BGE probability; routed rows use 50/50 probability blend",
        "offline_validation": {
            "iid_macro_ap": 0.7881950552432775,
            "hard_macro_ap": 0.37708810054523534,
            "ood_macro_ap": 0.6392545436659138,
            "mean_macro_ap": 0.6015125664848089,
        },
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
        "fast_benefit_router.py",
        "router_score_title.cbm",
        "router_manifest.json",
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
