#!/usr/bin/env python3
"""Package the full one-way triple-model submission without CatBoost."""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

import build_bge_minilm_ensemble_submit as common


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "submits/bge-minilm-rumodern-full-oneway-st-v1"
ARCHIVE = ROOT / "submits/bge-minilm-rumodern-full-oneway-st-v1.zip"
EXPECTED_IMAGE = "dinakepech/ecup26-bge-minilm-rumodern-router-sdpa:1.0"
RU_SOURCE = ROOT / "artifacts/kaggle/product-matching-architecture-rumodernbert-v1/rumodernbert_base_random_head_human_ft_v1"
RU_MODEL = SUBMIT / "models/rumodernbert-base-human-ft-v1"
RU_FILES = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}
RU_EXPECTED = {"bytes": 598_436_708, "sha256": "e6cda247fe02e615bfc1ddc8849a7b82e207dbe603c533cd6b529ed66eb19bce"}


def copy_exact(source: Path, destination: Path) -> None:
    if not source.is_file() or not source.stat().st_size:
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def stage() -> None:
    common.validate_weight(common.BGE_SOURCE / "model.safetensors", "bge")
    common.validate_weight(common.MINILM_SOURCE / "model.safetensors", "minilm")
    common.copy_model(common.BGE_SOURCE, SUBMIT / "models/bge-reranker-v2-m3-human-ft-v1")
    common.copy_model(common.MINILM_SOURCE, SUBMIT / "models/minilm-5ep-human-ft-v1", common.MINILM_TOKENIZER_SOURCE)
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
        elif name == "config.json":
            config = json.loads(source.read_text(encoding="utf-8"))
            config["compile_model"] = False
            config["reference_compile"] = False
            destination.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            shutil.copy2(source, destination)
    copy_exact(common.FREQUENCY_SOURCE, SUBMIT / "attribute_name_frequency.csv")
    copy_exact(common.SERIALIZER_SOURCE, SUBMIT / "serialization_ablation.py")
    copy_exact(ROOT / "submits/bge-minilm-sentence-transformers-fa2-v1/ensemble_base.py", SUBMIT / "ensemble_base.py")
    copy_exact(ROOT / "submits/bge-minilm-fast-oneway-router40-st-v2/run.py", SUBMIT / "runtime_base.py")
    copy_exact(ROOT / "src/fast_benefit_router.py", SUBMIT / "fast_benefit_router.py")


def write_manifest() -> None:
    files = {}
    for path in sorted(SUBMIT.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            relative = path.relative_to(SUBMIT).as_posix()
            if relative != "SUBMISSION_MANIFEST.json":
                files[relative] = {"size": path.stat().st_size, "sha256": common.sha256(path)}
    manifest = {
        "experiment": "bge_minilm_rumodern_full_oneway_st_v1",
        "backend": "sentence-transformers CrossEncoder 5.1.2",
        "attention": "sdpa without fallback",
        "precision": "FP16",
        "serialization": "S2_VALUES_ONLY",
        "max_length": 384,
        "batch_size": 1024,
        "symmetric_inference": False,
        "routing": "none",
        "blend": {"bge": 1 / 3, "minilm": 1 / 3, "rumodernbert": 1 / 3, "source": "human-train OOF"},
        "validation": {"iid_macro_ap": 0.7956066390642699, "hard_macro_ap": 0.36559795928541744, "ood_macro_ap": 0.6452300227278635, "mean_macro_ap": 0.6021448736925169},
        "docker_image": EXPECTED_IMAGE,
        "weights": {**common.EXPECTED, "rumodernbert": RU_EXPECTED},
        "files": files,
    }
    (SUBMIT / "SUBMISSION_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def archive() -> None:
    ARCHIVE.unlink(missing_ok=True)
    with zipfile.ZipFile(ARCHIVE, "w", allowZip64=True) as output:
        for path in sorted(SUBMIT.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            compression = zipfile.ZIP_STORED if path.suffix == ".safetensors" else zipfile.ZIP_DEFLATED
            output.write(path, path.relative_to(SUBMIT).as_posix(), compress_type=compression)
    with zipfile.ZipFile(ARCHIVE) as source:
        assert source.testzip() is None
    print(f"Created {ARCHIVE} ({ARCHIVE.stat().st_size:,} bytes)")
    print(f"SHA-256: {common.sha256(ARCHIVE)}")


def main() -> int:
    stage()
    write_manifest()
    archive()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
