#!/usr/bin/env python3
"""Build the final one-way BGE + MiniLM + RuModernBERT submission archive."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "submits/bge-minilm-rumodern-full-oneway-st-final-v1"
ARCHIVE = ROOT / "submits/bge-minilm-rumodern-full-oneway-final.zip"
EXPECTED_IMAGE = "dinakepech/ecup26-bge-minilm-rumodern-router-sdpa:1.0"

BGE_SOURCE = ROOT / "configs/bge_final"
MINILM_SOURCE = ROOT / "configs/minilm_final"
RUMODERN_SOURCE = ROOT / "configs/rumodernbert_final"
# Compatibility with the directory that already exists locally.
RUMODERN_LEGACY_SOURCE = ROOT / "configs/rubertmodern_final"

MODEL_DESTINATIONS = {
    "bge": SUBMIT / "models/bge_final",
    "minilm": SUBMIT / "models/minilm_final",
    "rumodernbert": SUBMIT / "models/rumodernbert_final",
}
TOKENIZER_FILES = {
    "added_tokens.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "vocab.txt",
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_weight_files(directory: Path) -> list[Path]:
    single = directory / "model.safetensors"
    if single.is_file() and single.stat().st_size:
        return [single]

    index = directory / "model.safetensors.index.json"
    if not index.is_file() or not index.stat().st_size:
        return []
    payload = json.loads(index.read_text(encoding="utf-8"))
    shard_names = sorted(set(payload.get("weight_map", {}).values()))
    if not shard_names:
        raise RuntimeError(f"empty safetensors weight map: {index}")
    shards = [directory / name for name in shard_names]
    missing = [path.name for path in shards if not path.is_file() or not path.stat().st_size]
    if missing:
        raise FileNotFoundError(f"missing safetensors shards referenced by {index}: {missing}")
    return [index, *shards]


def transport_split(directory: Path) -> tuple[dict, list[Path]] | None:
    """Return the Kaggle transport manifest and ordered byte chunks, if present."""
    manifest_path = directory / "checkpoint_manifest.json"
    if not manifest_path.is_file() or not manifest_path.stat().st_size:
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reconstruction = manifest.get("reconstruction", {})
    if reconstruction.get("filename") != "model.safetensors":
        return None
    part_names = reconstruction.get("parts", [])
    if not part_names:
        raise RuntimeError(f"empty checkpoint reconstruction list: {manifest_path}")
    parts = [directory / name for name in part_names]
    missing = [path.name for path in parts if not path.is_file() or not path.stat().st_size]
    if missing:
        raise FileNotFoundError(f"missing transport chunks referenced by {manifest_path}: {missing}")
    return manifest, parts


def has_checkpoint_weights(directory: Path) -> bool:
    return bool(checkpoint_weight_files(directory) or transport_split(directory))


def is_checkpoint_root(directory: Path) -> bool:
    return (
        (directory / "config.json").is_file()
        and (directory / "tokenizer_config.json").is_file()
        and has_checkpoint_weights(directory)
    )


def find_checkpoint_root(label: str, source: Path) -> Path:
    if not source.is_dir():
        raise FileNotFoundError(f"{label}: model directory does not exist: {source}")
    candidates = []
    if is_checkpoint_root(source):
        candidates.append(source)
    candidates.extend(
        path.parent
        for path in source.rglob("config.json")
        if path.parent != source and is_checkpoint_root(path.parent)
    )
    candidates = sorted(set(candidates), key=lambda path: (len(path.parts), path.as_posix()))
    if not candidates:
        raise FileNotFoundError(
            f"{label}: no complete Hugging Face checkpoint under {source}; expected config.json, "
            "tokenizer_config.json and model.safetensors, Hugging Face shards, or "
            "checkpoint_manifest.json plus transport chunks"
        )
    if len(candidates) > 1:
        rendered = ", ".join(str(path.relative_to(source)) for path in candidates)
        raise RuntimeError(f"{label}: multiple checkpoint roots under {source}: {rendered}")
    return candidates[0]


def resolve_sources() -> dict[str, Path]:
    rumodern = RUMODERN_SOURCE
    if not rumodern.is_dir() and RUMODERN_LEGACY_SOURCE.is_dir():
        rumodern = RUMODERN_LEGACY_SOURCE
        print(
            "Note: configs/rumodernbert_final is absent; using the existing "
            "configs/rubertmodern_final directory."
        )
    roots = {
        "bge": BGE_SOURCE,
        "minilm": MINILM_SOURCE,
        "rumodernbert": rumodern,
    }
    return {label: find_checkpoint_root(label, source) for label, source in roots.items()}


def model_files(label: str, source: Path) -> list[Path]:
    required = [source / "config.json", source / "tokenizer_config.json"]
    tokenizer_assets = [source / name for name in sorted(TOKENIZER_FILES) if (source / name).is_file()]
    if not tokenizer_assets:
        raise FileNotFoundError(f"{label}: no offline tokenizer vocabulary/model found in {source}")
    split = transport_split(source)
    transport_files = ([source / "checkpoint_manifest.json", *split[1]] if split else [])
    files = required + checkpoint_weight_files(source) + transport_files + tokenizer_assets
    empty = [path.name for path in files if not path.is_file() or not path.stat().st_size]
    if empty:
        raise RuntimeError(f"{label}: missing or empty inference files in {source}: {empty}")
    return files


def reconstruct_transport_checkpoint(source: Path, destination: Path) -> None:
    split = transport_split(source)
    if split is None:
        raise RuntimeError(f"no transport-split checkpoint in {source}")
    manifest, parts = split
    reconstruction = manifest["reconstruction"]
    file_manifest = manifest.get("files", {})
    expected_bytes = int(reconstruction["bytes"])
    expected_sha = str(reconstruction["sha256"])
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.unlink(missing_ok=True)
    total = 0
    combined_digest = hashlib.sha256()
    try:
        with temporary.open("wb") as output:
            for number, part in enumerate(parts, start=1):
                part_digest = hashlib.sha256()
                part_bytes = 0
                with part.open("rb") as chunk_file:
                    for chunk in iter(lambda: chunk_file.read(8 * 1024 * 1024), b""):
                        output.write(chunk)
                        combined_digest.update(chunk)
                        part_digest.update(chunk)
                        part_bytes += len(chunk)
                expected_part = file_manifest.get(part.name, {})
                if part_bytes != int(expected_part.get("bytes", -1)):
                    raise RuntimeError(f"transport chunk size mismatch: {part}")
                if part_digest.hexdigest() != expected_part.get("sha256"):
                    raise RuntimeError(f"transport chunk SHA-256 mismatch: {part}")
                total += part_bytes
                print(f"Reconstructed BGE chunk {number}/{len(parts)}", flush=True)
        if total != expected_bytes or combined_digest.hexdigest() != expected_sha:
            raise RuntimeError("reconstructed BGE model.safetensors identity mismatch")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def copy_model(label: str, source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    split = transport_split(source)
    direct_weights = set(checkpoint_weight_files(source))
    transport_files = set(([source / "checkpoint_manifest.json", *split[1]]) if split else [])
    for path in model_files(label, source):
        if path in transport_files:
            continue
        target = destination / path.name
        if label == "rumodernbert" and path.name == "config.json":
            config = json.loads(path.read_text(encoding="utf-8"))
            config["compile_model"] = False
            config["reference_compile"] = False
            target.write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        else:
            shutil.copy2(path, target)
    if split:
        if direct_weights:
            raise RuntimeError(f"ambiguous direct and transport-split weights in {source}")
        reconstruct_transport_checkpoint(source, destination / "model.safetensors")


def stage() -> dict[str, Path]:
    # Discover and validate all three sources before replacing an old staged payload.
    sources = resolve_sources()
    for label, source in sources.items():
        model_files(label, source)

    models_root = SUBMIT / "models"
    if models_root.exists():
        shutil.rmtree(models_root)
    for label, source in sources.items():
        copy_model(label, source, MODEL_DESTINATIONS[label])
    shutil.copy2(ROOT / "src/data_pipeline.py", SUBMIT / "data_pipeline.py")
    return sources


def payload_paths(include_manifest: bool = True) -> list[Path]:
    root_files = ROOT_PAYLOAD if include_manifest else ROOT_PAYLOAD[:-1]
    paths = [SUBMIT / name for name in root_files]
    for destination in MODEL_DESTINATIONS.values():
        paths.extend(path for path in destination.rglob("*") if path.is_file())
    return sorted(paths, key=lambda path: path.relative_to(SUBMIT).as_posix())


def write_manifest(sources: dict[str, Path]) -> None:
    files = {}
    for path in payload_paths(include_manifest=False):
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"submission payload is incomplete: {path}")
        relative = path.relative_to(SUBMIT).as_posix()
        files[relative] = {"size": path.stat().st_size, "sha256": sha256(path)}

    weights = {}
    for label, destination in MODEL_DESTINATIONS.items():
        checkpoint_files = checkpoint_weight_files(destination)
        weights[label] = {
            "source": str(sources[label].relative_to(ROOT)).replace("\\", "/"),
            "files": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in checkpoint_files
            },
        }
    manifest = {
        "experiment": "bge_minilm_rumodern_full_oneway_st_v1_final_checkpoints",
        "backend": "sentence-transformers CrossEncoder",
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
        "routing": "none; all three models score every pair",
        "blend": {"bge": 1 / 3, "minilm": 1 / 3, "rumodernbert": 1 / 3},
        "validation": "not recorded: final checkpoints require a fresh evaluation",
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
    with zipfile.ZipFile(ARCHIVE) as source:
        if source.testzip() is not None:
            raise RuntimeError("archive CRC validation failed")
        names = set(source.namelist())
        for required in ("metadata.json", "run.py", "SUBMISSION_MANIFEST.json"):
            if required not in names:
                raise RuntimeError(f"archive root is missing {required}")
    print(f"Created {ARCHIVE} ({ARCHIVE.stat().st_size:,} bytes)")
    print(f"SHA-256: {sha256(ARCHIVE)}")


def main() -> int:
    sources = stage()
    write_manifest(sources)
    archive()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
