#!/usr/bin/env python3
"""Package the frozen BGE + MiniLM normalized-rank ensemble."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BGE_SOURCE = (
    ROOT / "artifacts/kaggle/product-matching-architecture-bge-v2-m3-v1"
    / "bge_reranker_v2_m3_human_ft_v1"
)
MINILM_SOURCE = (
    ROOT / "artifacts/kaggle/product-matching-architecture-minilm-5ep-v1"
    / "minilm_5ep_synthetic_pretrain_human_ft_s2_v1"
)
MINILM_TOKENIZER_SOURCE = ROOT / "submits/minilm-s2-values-only/models/minilm-s2-values-only"
FREQUENCY_SOURCE = ROOT / "prepared/serialization_ablation/attribute_name_frequency.csv"
SERIALIZER_SOURCE = ROOT / "src/serialization_ablation.py"
SUBMIT = ROOT / "submits/bge-minilm-rank-ensemble-v1"
BGE_MODEL = SUBMIT / "models/bge-reranker-v2-m3-human-ft-v1"
MINILM_MODEL = SUBMIT / "models/minilm-5ep-human-ft-v1"
ARCHIVE = ROOT / "submits/bge-minilm-rank-ensemble-optimized-v2.zip"
EXPECTED_IMAGE = "dinakepech/ecup26-bge-reranker-v2-m3:1.0"
MAX_ARCHIVE_BYTES = 5 * 1024**3
MODEL_FILES = {
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
}
OPTIONAL_MODEL_FILES = {"added_tokens.json", "sentencepiece.bpe.model"}
EXPECTED = {
    "bge": {
        "bytes": 2_271_071_852,
        "sha256": "0e4e8e0b9dd5f220fd0423a5a27d8d6c09ff3697e72fbbf16f861f03df02b87e",
    },
    "minilm": {
        "bytes": 470_588_492,
        "sha256": "1122ac37bda1be257b743c56468fbf3abba8a341f89c7d34888a6c0f3afbb6ab",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_weight(path: Path, label: str) -> None:
    expected = EXPECTED[label]
    if not path.is_file() or path.stat().st_size != expected["bytes"]:
        raise RuntimeError(f"{label} weight is missing or has an unexpected size: {path}")
    actual = sha256(path)
    if actual != expected["sha256"]:
        raise RuntimeError(f"{label} weight SHA-256 mismatch: {actual}")


def copy_model(source: Path, destination: Path, tokenizer_source: Path | None = None) -> None:
    tokenizer_source = tokenizer_source or source
    destination.mkdir(parents=True, exist_ok=True)
    for name in sorted(MODEL_FILES | OPTIONAL_MODEL_FILES):
        candidate = source / name
        if name in {"special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"}:
            candidate = tokenizer_source / name
        if candidate.is_file():
            target = destination / name
            if name == "model.safetensors":
                target.unlink(missing_ok=True)
                os.link(candidate, target)
            else:
                shutil.copy2(candidate, target)
    missing = sorted(name for name in MODEL_FILES if not (destination / name).is_file())
    if missing:
        raise FileNotFoundError(f"incomplete staged model {destination}: {missing}")


def stage() -> None:
    validate_weight(BGE_SOURCE / "model.safetensors", "bge")
    validate_weight(MINILM_SOURCE / "model.safetensors", "minilm")
    copy_model(BGE_SOURCE, BGE_MODEL)
    copy_model(MINILM_SOURCE, MINILM_MODEL, MINILM_TOKENIZER_SOURCE)
    shutil.copy2(FREQUENCY_SOURCE, SUBMIT / "attribute_name_frequency.csv")
    shutil.copy2(SERIALIZER_SOURCE, SUBMIT / "serialization_ablation.py")


def write_manifest() -> None:
    metadata = json.loads((SUBMIT / "metadata.json").read_text(encoding="utf-8"))
    if metadata != {"image": EXPECTED_IMAGE, "entry_point": "python -u run.py"}:
        raise RuntimeError("unexpected metadata.json")
    with (SUBMIT / "attribute_name_frequency.csv").open("r", encoding="utf-8") as source:
        if sum(1 for _ in source) - 1 != 24_916:
            raise RuntimeError("unexpected attribute frequency table")
    files = {}
    for path in sorted(SUBMIT.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            relative = path.relative_to(SUBMIT).as_posix()
            if relative != "SUBMISSION_MANIFEST.json":
                files[relative] = {"size": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "experiment": "bge_minilm_rank_ensemble_optimized_v2",
        "aggregation": "mean normalized rank",
        "serialization": "S2_VALUES_ONLY",
        "max_length": 384,
        "symmetric_inference": True,
        "precision": "FP16 autocast",
        "inference_pipeline": "pretokenized exact-length bucketing with dynamic padding",
        "h100_batching": {
            "bge_max_pairs": 2048,
            "bge_pair_token_budget": 786432,
            "minilm_max_pairs": 4096,
            "minilm_pair_token_budget": 1572864,
            "tokenization_batch_size": 2048,
        },
        "t4_equivalence_benchmark": {
            "bge_pearson": 0.9999999774,
            "bge_ap_delta": -0.00000376,
            "minilm_pearson": 0.999999999999,
            "minilm_ap_delta": 0.0,
        },
        "offline_validation": {
            "ordinary_macro_ap": 0.792348,
            "hard_macro_ap": 0.374699,
            "ood_macro_ap": 0.640976,
            "mean_macro_ap": 0.602674,
        },
        "bge_leaderboard_pr_auc": 0.43,
        "docker_image": EXPECTED_IMAGE,
        "weights": EXPECTED,
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
        "SUBMISSION_MANIFEST.json",
        "models/bge-reranker-v2-m3-human-ft-v1/model.safetensors",
        "models/minilm-5ep-human-ft-v1/model.safetensors",
    }
    if not required.issubset(names):
        raise RuntimeError(f"submission ZIP is missing: {sorted(required - names)}")
    print(f"Created {ARCHIVE} ({ARCHIVE.stat().st_size:,} bytes)")
    print(f"SHA-256: {sha256(ARCHIVE)}")


def main() -> int:
    stage()
    write_manifest()
    archive()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
