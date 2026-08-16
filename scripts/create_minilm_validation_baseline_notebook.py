#!/usr/bin/env python3
"""Build the human-only MiniLM baseline with IID, hard and OOD evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

import create_cross_encoder_training_notebook as cross_builder
import create_qwen_training_notebook as shared


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = ROOT / "notebooks" / "minilm_validation_baseline_2xt4.ipynb"
DEFAULT_CONFIG = ROOT / "configs" / "cross_encoder_minilm_validation_baseline.json"
DEFAULT_SOURCE_DIR = ROOT / "prepared" / "validation_splits_v1"
DATASET_SLUG = "product-matching-validation-splits-v1"
REMOTE_FILES = {
    "human/items.parquet": "human_items.parquet",
    "human/train_pairs.parquet": "human_train_pairs.parquet",
    "human/iid_validation_pairs.parquet": "human_iid_validation_pairs.parquet",
    "human/hard_validation_pairs.parquet": "human_hard_validation_pairs.parquet",
    "human/ood_validation_pairs.parquet": "human_ood_validation_pairs.parquet",
}
EMBEDDED_FILES = (
    Path("requirements-cross-encoder.txt"),
    Path("src/__init__.py"),
    Path("src/cross_encoder_training.py"),
    Path("src/cross_encoder_experiment_hooks.py"),
    Path("src/data_pipeline.py"),
    Path("src/experiment_protocol.py"),
    Path("src/pair_features.py"),
    Path("src/qwen_reranker.py"),
    Path("src/qwen_training.py"),
    Path("scripts/train_cross_encoder.py"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embedded_sources() -> tuple[dict[str, str], str]:
    sources: dict[str, str] = {}
    digest = hashlib.sha256()
    for relative in EMBEDDED_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"Required source is missing: {source}")
        content = source.read_text(encoding="utf-8")
        sources[relative.as_posix()] = content
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(content.encode("utf-8"))
        digest.update(b"\0")
    return sources, digest.hexdigest()


def load_manifest(source_dir: Path, owner: str) -> dict[str, object]:
    manifest_path = source_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != "human_v1":
        raise ValueError("Validation manifest must have version='human_v1'")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Validation manifest has no output hashes")
    for relative in REMOTE_FILES:
        declaration = outputs.get(relative)
        path = source_dir / relative
        if not isinstance(declaration, dict) or not path.is_file():
            raise FileNotFoundError(f"Validation Dataset input is missing: {relative}")
        if int(declaration.get("bytes", -1)) != path.stat().st_size:
            raise ValueError(f"Validation Dataset size mismatch: {relative}")
        if str(declaration.get("sha256")) != sha256(path):
            raise ValueError(f"Validation Dataset hash mismatch: {relative}")
    return {
        "dataset": f"{owner}/{DATASET_SLUG}",
        "manifest_sha256": sha256(manifest_path),
        "manifest": manifest,
    }


def markdown(value: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(value).strip())


def code(value: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(value).strip())


def build_notebook(
    dataset: dict[str, object],
    training_config: dict[str, object],
    *,
    experiment_name: str = "minilm_validation_baseline_v1",
    experiment_title: str = "MiniLM human-only baseline: IID + hard + OOD",
    experiment_description: str | None = None,
    initial_checkpoint: dict[str, object] | None = None,
) -> nbf.NotebookNode:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", experiment_name):
        raise ValueError(f"Invalid experiment name: {experiment_name!r}")
    dataset_ref = str(dataset["dataset"])
    dataset_slug = dataset_ref.rsplit("/", 1)[-1]
    expected_manifest_hash = str(dataset["manifest_sha256"])
    checkpoint_ref = (
        str(initial_checkpoint["dataset"]) if initial_checkpoint is not None else ""
    )
    checkpoint_slug = checkpoint_ref.rsplit("/", 1)[-1] if checkpoint_ref else ""
    checkpoint_manifest_hash = (
        str(initial_checkpoint["manifest_sha256"])
        if initial_checkpoint is not None
        else ""
    )
    if experiment_description is None:
        experiment_description = (
            "`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` обучается только на "
            "frozen human train. Категории «Одежда» и «Бытовая техника» отсутствуют "
            "в train и используются только в OOD. После одной эпохи одна и та же "
            "модель считается сразу на IID, hard и OOD."
        )
    sources, source_hash = embedded_sources()
    # Single-line repr keeps dedent stable even though file contents themselves
    # contain escaped newlines.
    source_literal = repr(sources)
    config_literal = repr(training_config)
    remote_literal = repr(REMOTE_FILES)

    cells = [
        markdown(
            f"""
            # {experiment_title}

            {experiment_description}

            Dataset: `{dataset_ref}`. Manifest SHA-256: `{expected_manifest_hash}`.
            Initial checkpoint: `{checkpoint_ref or "upstream model"}`.
            Embedded source SHA-256: `{source_hash}`.
            """
        ),
        code(
            f"""
            import hashlib
            import json
            import os
            import subprocess
            import sys
            import time
            from pathlib import Path

            INPUT_ROOT = Path("/kaggle/input")
            WORKING_ROOT = Path("/kaggle/working")
            TEMP_ROOT = Path("/kaggle/temp/{experiment_name}")
            PROJECT_ROOT = WORKING_ROOT / "product_matching"
            PREPARED_DIR = TEMP_ROOT / "prepared"
            TOKEN_CACHE_DIR = TEMP_ROOT / "token_cache"
            OUTPUT_DIR = WORKING_ROOT / {experiment_name!r}
            RUNTIME_CONFIG_PATH = WORKING_ROOT / "cross_encoder_config.json"
            TRAIN_LOG = WORKING_ROOT / {f'{experiment_name}.log'!r}
            EXPECTED_DATASET_REF = {dataset_ref!r}
            EXPECTED_DATASET_SLUG = {dataset_slug!r}
            EXPECTED_MANIFEST_SHA256 = {expected_manifest_hash!r}
            INITIAL_CHECKPOINT_REF = {checkpoint_ref!r}
            INITIAL_CHECKPOINT_SLUG = {checkpoint_slug!r}
            EXPECTED_CHECKPOINT_MANIFEST_SHA256 = {checkpoint_manifest_hash!r}
            EMBEDDED_SOURCE_SHA256 = {source_hash!r}
            REMOTE_FILES = {remote_literal}

            def exactly_one(filename):
                direct_candidates = [
                    INPUT_ROOT / EXPECTED_DATASET_SLUG / filename,
                    INPUT_ROOT / "datasets" / EXPECTED_DATASET_REF / filename,
                ]
                for candidate in direct_candidates:
                    if candidate.is_file():
                        return candidate
                candidates = list(INPUT_ROOT.glob(f"**/{{filename}}"))
                if len(candidates) != 1:
                    raise RuntimeError(
                        f"Expected exactly one {{filename!r}} from "
                        f"{{EXPECTED_DATASET_REF}}, found {{candidates}}"
                    )
                return candidates[0]

            def file_sha256(path):
                digest = hashlib.sha256()
                with path.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                return digest.hexdigest()

            manifest_path = exactly_one("validation_splits_manifest.json")
            if file_sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
                raise RuntimeError("Attached validation Dataset manifest has changed")
            validation_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if validation_manifest["ood_categories"] != ["Одежда", "Бытовая техника"]:
                raise RuntimeError("Unexpected OOD categories in attached Dataset")
            attached_files = {{
                relative: exactly_one(remote_name)
                for relative, remote_name in REMOTE_FILES.items()
            }}
            for relative, path in attached_files.items():
                expected = validation_manifest["outputs"][relative]
                if path.stat().st_size != expected["bytes"] or file_sha256(path) != expected["sha256"]:
                    raise RuntimeError(f"Attached Dataset file differs from manifest: {{relative}}")

            INITIAL_MODEL_PATH = None
            if INITIAL_CHECKPOINT_REF:
                checkpoint_manifest_path = exactly_one("checkpoint_manifest.json")
                if file_sha256(checkpoint_manifest_path) != EXPECTED_CHECKPOINT_MANIFEST_SHA256:
                    raise RuntimeError("Attached initial checkpoint manifest has changed")
                checkpoint_manifest = json.loads(
                    checkpoint_manifest_path.read_text(encoding="utf-8")
                )
                if checkpoint_manifest.get("dataset") != INITIAL_CHECKPOINT_REF:
                    raise RuntimeError("Unexpected initial checkpoint Dataset")
                checkpoint_root = checkpoint_manifest_path.parent
                for filename, declaration in checkpoint_manifest["files"].items():
                    checkpoint_file = checkpoint_root / filename
                    if not checkpoint_file.is_file():
                        raise RuntimeError(f"Initial checkpoint file is missing: {{filename}}")
                    if (
                        checkpoint_file.stat().st_size != declaration["bytes"]
                        or file_sha256(checkpoint_file) != declaration["sha256"]
                    ):
                        raise RuntimeError(f"Initial checkpoint file differs from manifest: {{filename}}")
                reconstruction = checkpoint_manifest.get("reconstruction")
                if reconstruction:
                    reconstructed_root = TEMP_ROOT / "initial_checkpoint"
                    reconstructed_root.mkdir(parents=True, exist_ok=True)
                    part_names = set(reconstruction["parts"])
                    for filename in checkpoint_manifest["files"]:
                        if filename in part_names:
                            continue
                        destination = reconstructed_root / filename
                        if destination.exists() or destination.is_symlink():
                            destination.unlink()
                        destination.symlink_to(checkpoint_root / filename)
                    reconstructed_model = reconstructed_root / reconstruction["filename"]
                    model_digest = hashlib.sha256()
                    with reconstructed_model.open("wb") as destination:
                        for part_name in reconstruction["parts"]:
                            part = checkpoint_root / part_name
                            with part.open("rb") as source:
                                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                                    destination.write(chunk)
                                    model_digest.update(chunk)
                    if (
                        reconstructed_model.stat().st_size != reconstruction["bytes"]
                        or model_digest.hexdigest() != reconstruction["sha256"]
                    ):
                        raise RuntimeError("Reconstructed initial model differs from source")
                    INITIAL_MODEL_PATH = reconstructed_root
                else:
                    INITIAL_MODEL_PATH = checkpoint_root
            print(json.dumps({{
                "dataset": EXPECTED_DATASET_REF,
                "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                "initial_checkpoint": INITIAL_CHECKPOINT_REF or None,
                "initial_checkpoint_manifest_sha256": (
                    EXPECTED_CHECKPOINT_MANIFEST_SHA256 or None
                ),
                "files": {{name: str(path) for name, path in attached_files.items()}},
            }}, ensure_ascii=False, indent=2))
            print(subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True).stdout)
            """
        ),
        shared.experiment_run_initialization_cell(),
        markdown("## Базовый конфиг"),
        code(
            f"""
            TRAIN_CONFIG = {config_literal}
            if INITIAL_MODEL_PATH is not None:
                TRAIN_CONFIG["model"] = str(INITIAL_MODEL_PATH)
            RUNTIME_CONFIG_PATH.write_text(
                json.dumps(TRAIN_CONFIG, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(json.dumps(TRAIN_CONFIG, ensure_ascii=False, indent=2))
            """
        ),
        markdown("## Код и frozen data"),
        code(
            f"""
            EMBEDDED_SOURCES = {source_literal}
            PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
            for relative, content in EMBEDDED_SOURCES.items():
                destination = PROJECT_ROOT / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(content, encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--quiet",
                    "--disable-pip-version-check",
                    "--upgrade-strategy",
                    "only-if-needed",
                    "-r",
                    str(PROJECT_ROOT / "requirements-cross-encoder.txt"),
                ],
                check=True,
            )
            PREPARED_DIR.mkdir(parents=True, exist_ok=True)
            prepared_names = {{
                "human/items.parquet": "items.parquet",
                "human/train_pairs.parquet": "train_pairs.parquet",
                "human/iid_validation_pairs.parquet": "iid_validation_pairs.parquet",
                "human/hard_validation_pairs.parquet": "hard_validation_pairs.parquet",
                "human/ood_validation_pairs.parquet": "ood_validation_pairs.parquet",
            }}
            for relative, destination_name in prepared_names.items():
                destination = PREPARED_DIR / destination_name
                if destination.exists() or destination.is_symlink():
                    destination.unlink()
                destination.symlink_to(attached_files[relative])
            """
        ),
        markdown("## Обучение и три validation-протокола"),
        code(
            """
            train_command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                str(PROJECT_ROOT / "scripts/train_cross_encoder.py"),
                "--config", str(RUNTIME_CONFIG_PATH),
                "--prepared-dir", str(PREPARED_DIR),
                "--output-dir", str(OUTPUT_DIR),
                "--token-cache-dir", str(TOKEN_CACHE_DIR),
                "--validation-split", "iid=iid_validation_pairs.parquet",
                "--validation-split", "hard=hard_validation_pairs.parquet",
                "--validation-split", "ood=ood_validation_pairs.parquet",
            ]
            training_environment = os.environ.copy()
            training_environment.update({
                "OMP_NUM_THREADS": "2",
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "PYTHONUNBUFFERED": "1",
            })
            print("$", " ".join(train_command), flush=True)
            training_started = time.perf_counter()
            with TRAIN_LOG.open("w", encoding="utf-8", buffering=1) as log_file:
                process = subprocess.Popen(
                    train_command,
                    cwd=PROJECT_ROOT,
                    env=training_environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log_file.write(line)
                return_code = process.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, train_command)
            training_wall_seconds = time.perf_counter() - training_started
            """
        ),
        markdown("## Артефакты и completion report"),
        code(
            """
            report_path = OUTPUT_DIR / "training_report.json"
            if not report_path.is_file():
                raise RuntimeError(f"Training finished without report: {report_path}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            expected_splits = {"iid", "hard", "ood"}
            if set(report.get("validation_splits", {})) != expected_splits:
                raise RuntimeError(
                    f"Expected three validation splits, got {sorted(report.get('validation_splits', {}))}"
                )
            completion = {
                "status": "complete",
                "run_id": EXPERIMENT_RUN_ID,
                "started_at_utc": EXPERIMENT_STARTED_AT_UTC,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ).replace("+00:00", "Z"),
                "experiment": OUTPUT_DIR.name,
                "model": INITIAL_CHECKPOINT_REF or TRAIN_CONFIG["model"],
                "dataset_ref": EXPECTED_DATASET_REF,
                "initial_checkpoint_ref": INITIAL_CHECKPOINT_REF or None,
                "initial_checkpoint_manifest_sha256": (
                    EXPECTED_CHECKPOINT_MANIFEST_SHA256 or None
                ),
                "kaggle_kernel_ref": (
                    os.getenv("KAGGLE_KERNEL_RUN_ID")
                    or os.getenv("KAGGLE_KERNEL_INFERENCE_RUN_ID")
                    or ""
                ),
                "code_bundle_sha256": EMBEDDED_SOURCE_SHA256,
                "training_wall_seconds": training_wall_seconds,
                "training_report": report,
            }
            completion_path = WORKING_ROOT / "notebook_completed.json"
            completion_path.write_text(
                json.dumps(completion, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            summary = {
                name: {
                    "macro_ap": metrics["macro_average_precision"],
                    "overall_ap": metrics["overall_average_precision"],
                    "examples": metrics["examples"],
                }
                for name, metrics in report["validation_splits"].items()
            }
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            """
        ),
        *shared.google_sheets_tracking_cells(),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "product_matching_training": {
                "dataset": dataset_ref,
                "manifest_sha256": expected_manifest_hash,
                "source_sha256": source_hash,
                "experiment": experiment_name,
                "initial_checkpoint": checkpoint_ref or None,
                "initial_checkpoint_manifest_sha256": checkpoint_manifest_hash or None,
                "validation_splits": ["iid", "hard", "ood"],
                "expected_gpus": 2,
            },
        }
    )
    return notebook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_NOTEBOOK)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    owner = shared.dotenv_username(args.env_file)
    if not owner:
        raise SystemExit("Set KAGGLE_USERNAME in .env")
    dataset = load_manifest(args.source_dir, owner)
    config = cross_builder.load_training_config(args.config)
    notebook = build_notebook(dataset, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, args.output)
    print(f"Wrote notebook: {args.output}")
    print(f"Dataset reference: {dataset['dataset']}")
    print(f"Manifest SHA-256: {dataset['manifest_sha256']}")


if __name__ == "__main__":
    main()
