#!/usr/bin/env python3
"""Download pinned offline assets and build the Qwen3 vLLM submission ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import NoReturn


ROOT = Path(__file__).resolve().parents[1]
SUBMIT_DIR = ROOT / "submits" / "qwen3-reranker-vllm-zero-shot"
ARCHIVE_PATH = ROOT / "submits" / "qwen3-reranker-vllm-zero-shot.zip"
MODEL_DIR = SUBMIT_DIR / "models" / "Qwen3-Reranker-0.6B"
PYARROW_RUNTIME = SUBMIT_DIR / "vendor" / "pyarrow_runtime"

MODEL_REPOSITORY = "Qwen/Qwen3-Reranker-0.6B"
MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
MODEL_BYTES = 1_191_588_280
MODEL_SHA256 = "27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b"
MODEL_FILES = (
    "1_LogitScore/config.json",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "config_sentence_transformers.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)

PYARROW_VERSION = "21.0.0"
PYARROW_WHEEL = "pyarrow-21.0.0-cp312-cp312-manylinux_2_28_x86_64.whl"
PYARROW_SHA256 = "b7ae0bbdc8c6674259b25bef5d2a1d6af5d39d7200c819cf99e07f7dfef1c51e"
MAX_ARCHIVE_BYTES = 5_000_000_000


def fail(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    curl = shutil.which("curl")
    if curl is None:
        fail("curl is required to download submission assets")
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {destination.name}", flush=True)
    result = subprocess.run(
        [
            curl,
            "-fL",
            "--retry",
            "5",
            "--retry-delay",
            "3",
            "--continue-at",
            "-",
            "--output",
            str(destination),
            url,
        ],
        check=False,
    )
    if result.returncode:
        fail(f"download failed with exit code {result.returncode}: {url}")


def download_model(*, force: bool) -> None:
    for relative_name in MODEL_FILES:
        destination = MODEL_DIR / relative_name
        if destination.is_file() and not force:
            continue
        if force and destination.exists():
            destination.unlink()
        url = (
            f"https://huggingface.co/{MODEL_REPOSITORY}/resolve/"
            f"{MODEL_REVISION}/{relative_name}"
        )
        download(url, destination)


def verify_model() -> None:
    missing = [name for name in MODEL_FILES if not (MODEL_DIR / name).is_file()]
    if missing:
        fail(f"model checkpoint is incomplete; missing: {missing}")

    weights = MODEL_DIR / "model.safetensors"
    if weights.stat().st_size != MODEL_BYTES:
        fail(
            f"unexpected model size: {weights.stat().st_size}; expected {MODEL_BYTES}"
        )
    actual_hash = sha256(weights)
    if actual_hash != MODEL_SHA256:
        fail(f"model SHA-256 mismatch: {actual_hash}")

    for name in ("config.json", "tokenizer_config.json", "modules.json"):
        with (MODEL_DIR / name).open(encoding="utf-8") as source:
            json.load(source)
    print("Model checkpoint verified", flush=True)


def resolve_pyarrow_wheel_url() -> str:
    url = f"https://pypi.org/pypi/pyarrow/{PYARROW_VERSION}/json"
    print(f"Resolving {PYARROW_WHEEL} from PyPI", flush=True)
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
        metadata = json.load(response)
    for file_info in metadata.get("urls", []):
        if file_info.get("filename") != PYARROW_WHEEL:
            continue
        remote_hash = file_info.get("digests", {}).get("sha256")
        if remote_hash != PYARROW_SHA256:
            fail(f"PyPI reports an unexpected wheel SHA-256: {remote_hash}")
        return str(file_info["url"])
    fail(f"wheel is absent from PyPI metadata: {PYARROW_WHEEL}")


def install_pyarrow_runtime(*, force: bool) -> None:
    marker = PYARROW_RUNTIME / "pyarrow" / "__init__.py"
    if marker.is_file() and not force:
        print("Vendored PyArrow runtime already exists", flush=True)
        return

    if PYARROW_RUNTIME.exists():
        shutil.rmtree(PYARROW_RUNTIME)
    PYARROW_RUNTIME.mkdir(parents=True)

    with tempfile.TemporaryDirectory(prefix="product-matching-pyarrow-") as temp_dir:
        wheel = Path(temp_dir) / PYARROW_WHEEL
        download(resolve_pyarrow_wheel_url(), wheel)
        actual_hash = sha256(wheel)
        if actual_hash != PYARROW_SHA256:
            fail(f"PyArrow wheel SHA-256 mismatch: {actual_hash}")
        with zipfile.ZipFile(wheel) as archive:
            members = [
                info
                for info in archive.infolist()
                if info.filename.startswith("pyarrow/")
                and ".." not in Path(info.filename).parts
            ]
            archive.extractall(PYARROW_RUNTIME, members=members)

    if not marker.is_file():
        fail("PyArrow extraction did not create pyarrow/__init__.py")
    print("Vendored PyArrow runtime prepared", flush=True)


def include_in_archive(path: Path) -> bool:
    relative = path.relative_to(SUBMIT_DIR)
    if any(part in {"__pycache__", "__MACOSX"} for part in relative.parts):
        return False
    if path.name == ".DS_Store" or path.suffix == ".pyc":
        return False
    return path.is_file()


def build_archive() -> None:
    required = (
        SUBMIT_DIR / "metadata.json",
        SUBMIT_DIR / "run.py",
        SUBMIT_DIR / "run.sh",
        MODEL_DIR / "model.safetensors",
        PYARROW_RUNTIME / "pyarrow" / "__init__.py",
    )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        fail(f"submission payload is incomplete; missing: {missing}")

    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()
    print(f"Building {ARCHIVE_PATH.relative_to(ROOT)}", flush=True)
    with zipfile.ZipFile(ARCHIVE_PATH, "w", allowZip64=True) as archive:
        for path in sorted(SUBMIT_DIR.rglob("*")):
            if not include_in_archive(path):
                continue
            relative = path.relative_to(SUBMIT_DIR).as_posix()
            compression = (
                zipfile.ZIP_STORED
                if path.suffix == ".safetensors"
                else zipfile.ZIP_DEFLATED
            )
            archive.write(path, relative, compress_type=compression, compresslevel=9)

    archive_size = ARCHIVE_PATH.stat().st_size
    if archive_size > MAX_ARCHIVE_BYTES:
        fail(f"archive exceeds the 5 GB competition limit: {archive_size} bytes")
    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        broken_member = archive.testzip()
    if broken_member:
        fail(f"archive CRC validation failed: {broken_member}")
    print(f"Archive size: {archive_size:,} bytes", flush=True)
    print(f"Archive SHA-256: {sha256(ARCHIVE_PATH)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-only",
        action="store_true",
        help="do not access the network; verify existing assets and rebuild ZIP",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="redownload model files and rebuild the vendored PyArrow runtime",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.archive_only and args.force_download:
        fail("--archive-only and --force-download cannot be used together")
    if not args.archive_only:
        download_model(force=args.force_download)
        install_pyarrow_runtime(force=args.force_download)
    verify_model()
    build_archive()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
