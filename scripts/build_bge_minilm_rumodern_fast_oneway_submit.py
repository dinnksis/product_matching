#!/usr/bin/env python3
"""Package the compact one-way BGE/MiniLM/RuModernBERT submission."""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

import build_bge_minilm_ensemble_submit as common


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "submits/bge-minilm-rumodern-fast-oneway-40-5-st-v2"
ARCHIVE = ROOT / "submits/bge-minilm-rumodern-fast-oneway-40-5-st-v2.zip"
BGE_MODEL = SUBMIT / "models/bge-reranker-v2-m3-human-ft-v1"
MINILM_MODEL = SUBMIT / "models/minilm-5ep-human-ft-v1"
RU_SOURCE = (
    ROOT
    / "artifacts/kaggle/product-matching-architecture-rumodernbert-v1"
    / "rumodernbert_base_random_head_human_ft_v1"
)
RU_MODEL = SUBMIT / "models/rumodernbert-base-human-ft-v1"
MINI_ROUTER_DIR = ROOT / "artifacts/fast_oneway_benefit_router_v1"
RU_ROUTER_DIR = ROOT / "artifacts/fast_oneway_rumodern_router_v1"
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
    if not weight.is_file():
        raise FileNotFoundError(weight)
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
        elif name == "config.json":
            config = json.loads(source.read_text(encoding="utf-8"))
            # The checkpoint was trained with compile_model=true, but the
            # offline competition image has no C compiler for Triton. SDPA
            # inference works without the ModernBERT torch.compile wrapper.
            config["compile_model"] = False
            config["reference_compile"] = False
            destination.write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
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
        ROOT / "submits/bge-minilm-fast-oneway-router40-st-v2/run.py",
        SUBMIT / "runtime_base.py",
    )
    copy_exact(ROOT / "src/fast_benefit_router.py", SUBMIT / "fast_benefit_router.py")
    copy_exact(ROOT / "src/fast_sequential_router.py", SUBMIT / "fast_sequential_router.py")
    copy_exact(
        MINI_ROUTER_DIR / "models/router_score_category.cbm",
        SUBMIT / "router_minilm_oneway_score_category.cbm",
    )
    copy_exact(
        RU_ROUTER_DIR / "router_rumodern_oneway_score_category.cbm",
        SUBMIT / "router_rumodern_oneway_score_category.cbm",
    )
    mini_source = json.loads((MINI_ROUTER_DIR / "manifest.json").read_text(encoding="utf-8"))
    mini_manifest = {
        "variant": "score_category",
        "feature_columns": mini_source["variants"]["score_category"]["feature_columns"],
        "categorical_columns": ["category"],
    }
    (SUBMIT / "mini_router_manifest.json").write_text(
        json.dumps(mini_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    ru_source = json.loads((RU_ROUTER_DIR / "manifest.json").read_text(encoding="utf-8"))
    ru_manifest = {
        "target": ru_source["target"],
        "feature_columns": ru_source["feature_columns"],
        "categorical_columns": ru_source["categorical_columns"],
    }
    (SUBMIT / "rumodern_router_manifest.json").write_text(
        json.dumps(ru_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
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
        "experiment": "bge_minilm_rumodern_fast_oneway_40_5_st_v2",
        "backend": "sentence-transformers CrossEncoder 5.1.2",
        "attention": "sdpa without fallback",
        "precision": "FP16",
        "serialization": "S2_VALUES_ONLY",
        "max_length": 384,
        "batch_size": 1024,
        "symmetric_inference": False,
        "routing": {
            "mini_coverage": 0.40,
            "rumodern_coverage": 0.05,
            "rumodern_subset_of_minilm": True,
            "mini_router_features": 7,
            "rumodern_router_features": 14,
            "attribute_features": False,
            "title_fuzzy_features": False,
            "mini_weight": 0.50,
            "rumodern_weight_on_current_score": 0.50,
        },
        "offline_validation": {
            "iid_macro_ap": 0.7892377462021465,
            "hard_macro_ap": 0.3767918087586734,
            "ood_macro_ap": 0.6401140735427291,
            "mean_macro_ap": 0.6020478761678497,
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
        "metadata.json", "run.py", "runtime_base.py", "SUBMISSION_MANIFEST.json",
        "router_minilm_oneway_score_category.cbm",
        "router_rumodern_oneway_score_category.cbm",
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
