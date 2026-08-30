#!/usr/bin/env python3
"""Build the final-checkpoint full one-way BGE + MiniLM submission."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import build_bge_minilm_rumodern_full_oneway_submit as final_common


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "submits/bge-minilm-full-oneway-final"
ARCHIVE = ROOT / "submits/bge-minilm-full-oneway-final.zip"
EXPECTED_IMAGE = "dinakepech/ecup26-bge-minilm-rumodern-router-sdpa:1.0"
MODEL_DESTINATIONS = {
    "bge": SUBMIT / "models/bge_final",
    "minilm": SUBMIT / "models/minilm_final",
}
ROOT_PAYLOAD = (
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "metadata.json",
    "run.py",
    "runtime_base.py",
    "ensemble_base.py",
    "data_pipeline.py",
    "SUBMISSION_MANIFEST.json",
)


def stage() -> dict[str, Path]:
    sources = {
        "bge": final_common.find_checkpoint_root("bge", final_common.BGE_SOURCE),
        "minilm": final_common.find_checkpoint_root("minilm", final_common.MINILM_SOURCE),
    }
    for label, source in sources.items():
        final_common.model_files(label, source)

    models_root = SUBMIT / "models"
    if models_root.exists():
        shutil.rmtree(models_root)
    for label, source in sources.items():
        final_common.copy_model(label, source, MODEL_DESTINATIONS[label])
    shutil.copy2(ROOT / "src/data_pipeline.py", SUBMIT / "data_pipeline.py")
    return sources


def payload_paths(include_manifest: bool = True) -> list[Path]:
    roots = ROOT_PAYLOAD if include_manifest else ROOT_PAYLOAD[:-1]
    paths = [SUBMIT / name for name in roots]
    for destination in MODEL_DESTINATIONS.values():
        paths.extend(path for path in destination.rglob("*") if path.is_file())
    return sorted(paths, key=lambda path: path.relative_to(SUBMIT).as_posix())


def write_manifest(sources: dict[str, Path]) -> None:
    metadata = json.loads((SUBMIT / "metadata.json").read_text(encoding="utf-8"))
    if metadata != {"image": EXPECTED_IMAGE, "entry_point": "python -u run.py"}:
        raise RuntimeError("unexpected metadata.json")
    files = {}
    for path in payload_paths(include_manifest=False):
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"submission payload is incomplete: {path}")
        relative = path.relative_to(SUBMIT).as_posix()
        files[relative] = {
            "size": path.stat().st_size,
            "sha256": final_common.sha256(path),
        }
    weights = {}
    for label, destination in MODEL_DESTINATIONS.items():
        weight_files = final_common.checkpoint_weight_files(destination)
        weights[label] = {
            "source": str(sources[label].relative_to(ROOT)).replace("\\", "/"),
            "files": {
                path.name: {
                    "bytes": path.stat().st_size,
                    "sha256": final_common.sha256(path),
                }
                for path in weight_files
            },
        }
    manifest = {
        "experiment": "bge_minilm_full_oneway_final",
        "backend": "sentence-transformers CrossEncoder 5.1.2",
        "attention": "sdpa",
        "precision": "float16",
        "serialization": {
            "implementation": "src/data_pipeline.py:serialize_product",
            "format": "Категория, Название, then deterministic key: value lines",
            "max_attribute_chars": 6000,
        },
        "max_length": 384,
        "max_length_scope": "combined cross-encoder pair including special tokens",
        "batch_size": 1024,
        "direction": "AB only",
        "routing": "none; BGE and MiniLM both score every pair",
        "blend": {"bge": 0.5, "minilm": 0.5, "space": "sigmoid probability"},
        "offline_validation": "final checkpoints require fresh leaderboard evaluation",
        "docker_image": EXPECTED_IMAGE,
        "weights": weights,
        "files": files,
    }
    (SUBMIT / "SUBMISSION_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def archive() -> None:
    ARCHIVE.unlink(missing_ok=True)
    with zipfile.ZipFile(ARCHIVE, "w", allowZip64=True) as output:
        for path in payload_paths():
            relative = path.relative_to(SUBMIT).as_posix()
            compression = zipfile.ZIP_STORED if path.suffix == ".safetensors" else zipfile.ZIP_DEFLATED
            output.write(path, relative, compress_type=compression)
    if ARCHIVE.stat().st_size > 5 * 1024**3:
        raise RuntimeError(f"submission ZIP exceeds 5 GiB: {ARCHIVE}")
    with zipfile.ZipFile(ARCHIVE) as source:
        if source.testzip() is not None:
            raise RuntimeError("archive CRC validation failed")
        names = set(source.namelist())
    required = {
        "metadata.json",
        "run.py",
        "SUBMISSION_MANIFEST.json",
        "models/bge_final/model.safetensors",
        "models/minilm_final/model.safetensors",
    }
    if not required.issubset(names):
        raise RuntimeError(f"submission ZIP is missing: {sorted(required - names)}")
    if any(name.startswith("models/rumodernbert") or name.endswith(".cbm") for name in names):
        raise RuntimeError("BGE+MiniLM archive unexpectedly contains RuModern or CatBoost")
    print(f"Created {ARCHIVE} ({ARCHIVE.stat().st_size:,} bytes)")
    print(f"SHA-256: {final_common.sha256(ARCHIVE)}")


def main() -> int:
    sources = stage()
    write_manifest(sources)
    archive()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
