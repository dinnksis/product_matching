#!/usr/bin/env python3
"""Build LogisticRegression and CatBoost S2 ensemble competition ZIPs."""

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
ARTIFACTS = ROOT / "artifacts/cheap_ensemble_s2"
IMAGE = "dinakepech/ecup26-minilm-s2-ensemble:1.0"
MAX_ARCHIVE_BYTES = 5 * 1024**3
VARIANTS = {
    "logistic": {
        "directory": "minilm-s2-logistic-ensemble",
        "model_file": "logistic_pipeline.joblib",
        "macro_ap": 0.6907077804849908,
    },
    "catboost": {
        "directory": "minilm-s2-catboost-ensemble",
        "model_file": "catboost_model.cbm",
        "macro_ap": 0.7031232963649957,
    },
}
MODEL_FILES = {
    "config.json",
    "model.safetensors",
    "sentencepiece.bpe.model",
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


def copy_common(destination: Path) -> None:
    missing = [name for name in MODEL_FILES if not (SOURCE_MODEL / name).is_file()]
    if missing:
        raise FileNotFoundError(f"S2 checkpoint is incomplete: {sorted(missing)}")
    model_destination = destination / "models/minilm-s2-values-only"
    model_destination.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        shutil.copy2(SOURCE_MODEL / name, model_destination / name)
    shutil.copy2(SOURCE_FREQUENCY, destination / "attribute_name_frequency.csv")
    shutil.copy2(ROOT / "src/serialization_ablation.py", destination / "serialization_ablation.py")
    shutil.copy2(ROOT / "src/cheap_ensemble.py", destination / "cheap_ensemble.py")
    shutil.copy2(ROOT / "src/cheap_ensemble_submission.py", destination / "run.py")
    shutil.copy2(ARTIFACTS / "char_idf.npy", destination / "char_idf.npy")
    shutil.copy2(ARTIFACTS / "feature_config.json", destination / "feature_config.json")


def write_metadata(destination: Path, model_type: str) -> None:
    (destination / "metadata.json").write_text(
        json.dumps({"image": IMAGE, "entry_point": "python -u run.py"}, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "ensemble_config.json").write_text(
        json.dumps({"model_type": model_type}, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "README.md").write_text(
        "# MiniLM S2 cheap ensemble submission\n\n"
        f"Meta-model: `{model_type}`. The frozen S2 score is augmented with "
        "symmetric lexical, numeric, product-code and attribute-comparison features.\n\n"
        "Build the shared runtime image from `docker/minilm-s2-ensemble-runtime/Dockerfile`.\n",
        encoding="utf-8",
    )
    (destination / "THIRD_PARTY_NOTICES.md").write_text(
        "# Third-party notices\n\n"
        "The checkpoint derives from `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` "
        "at revision `1427fd652930e4ba29e8149678df786c240d8825` (Apache-2.0). "
        "The runtime also uses scikit-learn, SciPy, RapidFuzz and CatBoost under "
        "their respective open-source licenses.\n",
        encoding="utf-8",
    )


def manifest(destination: Path, model_type: str, macro_ap: float) -> None:
    files = {}
    for path in sorted(destination.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            relative = path.relative_to(destination).as_posix()
            if relative != "SUBMISSION_MANIFEST.json":
                files[relative] = {"size": path.stat().st_size, "sha256": sha256(path)}
    payload = {
        "experiment": f"minilm_s2_cheap_ensemble_{model_type}",
        "validation_protocol": "5-fold item+family-component-disjoint OOF on frozen S2 human holdout",
        "validation_macro_average_precision": macro_ap,
        "base_model": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "base_model_revision": "1427fd652930e4ba29e8149678df786c240d8825",
        "docker_image": IMAGE,
        "files": files,
    }
    (destination / "SUBMISSION_MANIFEST.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def archive(destination: Path) -> Path:
    archive_path = destination.with_suffix(".zip")
    archive_path.unlink(missing_ok=True)
    with zipfile.ZipFile(archive_path, "w", allowZip64=True) as output:
        for path in sorted(destination.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            compression = zipfile.ZIP_STORED if path.suffix == ".safetensors" else zipfile.ZIP_DEFLATED
            output.write(path, path.relative_to(destination).as_posix(), compress_type=compression)
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise RuntimeError(f"submission ZIP exceeds 5 GiB: {archive_path}")
    with zipfile.ZipFile(archive_path) as source:
        broken = source.testzip()
        names = set(source.namelist())
    if broken:
        raise RuntimeError(f"broken ZIP member: {broken}")
    required = {
        "metadata.json",
        "run.py",
        "cheap_ensemble.py",
        "serialization_ablation.py",
        "feature_config.json",
        "char_idf.npy",
        "SUBMISSION_MANIFEST.json",
        "models/minilm-s2-values-only/model.safetensors",
    }
    if not required.issubset(names):
        raise RuntimeError(f"submission ZIP is missing: {sorted(required - names)}")
    return archive_path


def main() -> int:
    for model_type, settings in VARIANTS.items():
        destination = ROOT / "submits" / settings["directory"]
        destination.mkdir(parents=True, exist_ok=True)
        copy_common(destination)
        model_name = str(settings["model_file"])
        shutil.copy2(ARTIFACTS / "models" / model_name, destination / "models" / model_name)
        write_metadata(destination, model_type)
        manifest(destination, model_type, float(settings["macro_ap"]))
        archive_path = archive(destination)
        print(f"Created {archive_path} ({archive_path.stat().st_size:,} bytes)")
        print(f"SHA-256: {sha256(archive_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
