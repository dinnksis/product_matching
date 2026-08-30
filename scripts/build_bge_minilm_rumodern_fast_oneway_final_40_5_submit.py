#!/usr/bin/env python3
"""Build the final-checkpoint BGE100 -> MiniLM40 -> RuModern5 submission."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import build_bge_minilm_rumodern_full_oneway_submit as final_common


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "submits/bge-minilm-rumodern-fast-oneway-40-5-st-final-v1"
ARCHIVE = ROOT / "submits/bge-minilm-rumodern-fast-oneway-40-5-st-final-v1.zip"
EXPECTED_IMAGE = "dinakepech/ecup26-bge-minilm-rumodern-router-sdpa:1.0"
MINI_ROUTER_DIR = ROOT / "artifacts/fast_oneway_benefit_router_v1"
RU_ROUTER_DIR = ROOT / "artifacts/fast_oneway_rumodern_router_v1"
MODEL_DESTINATIONS = {
    "bge": SUBMIT / "models/bge_final",
    "minilm": SUBMIT / "models/minilm_final",
    "rumodernbert": SUBMIT / "models/rumodernbert_final",
}
ROOT_PAYLOAD = (
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "metadata.json",
    "run.py",
    "runtime_base.py",
    "ensemble_base.py",
    "data_pipeline.py",
    "fast_benefit_router.py",
    "fast_sequential_router.py",
    "router_minilm_oneway_score_category.cbm",
    "router_rumodern_oneway_score_category.cbm",
    "mini_router_manifest.json",
    "rumodern_router_manifest.json",
    "SUBMISSION_MANIFEST.json",
)


def copy_required(source: Path, destination: Path) -> None:
    if not source.is_file() or not source.stat().st_size:
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def stage() -> dict[str, Path]:
    sources = final_common.resolve_sources()
    for label, source in sources.items():
        final_common.model_files(label, source)
    required_assets = (
        MINI_ROUTER_DIR / "models/router_score_category.cbm",
        MINI_ROUTER_DIR / "manifest.json",
        RU_ROUTER_DIR / "router_rumodern_oneway_score_category.cbm",
        RU_ROUTER_DIR / "manifest.json",
    )
    for path in required_assets:
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(path)

    models_root = SUBMIT / "models"
    if models_root.exists():
        shutil.rmtree(models_root)
    for label, source in sources.items():
        final_common.copy_model(label, source, MODEL_DESTINATIONS[label])

    copy_required(ROOT / "src/data_pipeline.py", SUBMIT / "data_pipeline.py")
    copy_required(ROOT / "src/fast_benefit_router.py", SUBMIT / "fast_benefit_router.py")
    copy_required(ROOT / "src/fast_sequential_router.py", SUBMIT / "fast_sequential_router.py")
    copy_required(
        MINI_ROUTER_DIR / "models/router_score_category.cbm",
        SUBMIT / "router_minilm_oneway_score_category.cbm",
    )
    copy_required(
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
        "experiment": "bge_minilm_rumodern_fast_oneway_40_5_st_final_v1",
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
        "routing": {
            "minilm_coverage": 0.40,
            "rumodernbert_coverage": 0.05,
            "rumodern_subset_of_minilm": True,
            "mini_router_features": 7,
            "rumodern_router_features": 14,
            "router_checkpoint_status": "reused; trained on previous-checkpoint OOF scores",
        },
        "blend": {
            "minilm": "50% current BGE score + 50% MiniLM on routed pairs",
            "rumodernbert": "50% current BGE/MiniLM score + 50% RuModernBERT",
        },
        "offline_validation": "not transferable to final checkpoints; fresh evaluation required",
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
        "router_minilm_oneway_score_category.cbm",
        "router_rumodern_oneway_score_category.cbm",
        "models/bge_final/model.safetensors",
        "models/minilm_final/model.safetensors",
        "models/rumodernbert_final/model.safetensors",
    }
    if not required.issubset(names):
        raise RuntimeError(f"submission ZIP is missing: {sorted(required - names)}")
    print(f"Created {ARCHIVE} ({ARCHIVE.stat().st_size:,} bytes)")
    print(f"SHA-256: {final_common.sha256(ARCHIVE)}")


def main() -> int:
    sources = stage()
    write_manifest(sources)
    archive()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
