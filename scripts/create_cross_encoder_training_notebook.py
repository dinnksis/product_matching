#!/usr/bin/env python3
"""Build a configurable MiniLM cross-encoder Kaggle training notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from pprint import pformat
from textwrap import dedent

import nbformat as nbf

import create_qwen_training_notebook as shared


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = ROOT / "notebooks" / "cross_encoder_minilm_training_2xt4.ipynb"
DEFAULT_CONFIG = ROOT / "configs" / "cross_encoder_minilm.json"


def markdown(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(text).strip())


def load_training_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Cross-encoder config must contain a JSON object")
    required = {"model", "epochs", "batch_size", "max_length", "learning_rate"}
    if missing := required - set(config):
        raise ValueError(f"Cross-encoder config is missing keys: {sorted(missing)}")
    return config


def build_notebook(
    manifest: dict[str, object], training_config: dict[str, object]
) -> nbf.NotebookNode:
    bundle = manifest["code_bundle"]
    assert isinstance(bundle, dict)
    expected_bundle_hash = str(bundle["sha256"])
    dataset_reference = str(manifest["dataset"])
    config_literal = pformat(training_config, sort_dicts=False, width=100)
    config_source = (
        "# Менять параметры будущих запусков нужно только здесь или в локальном\n"
        "# configs/cross_encoder_minilm.json перед повторной генерацией notebook.\n"
        f"TRAIN_CONFIG = {config_literal}\n"
        "RUNTIME_CONFIG_PATH.write_text(\n"
        "    json.dumps(TRAIN_CONFIG, ensure_ascii=False, indent=2), encoding='utf-8'\n"
        ")\n"
        "print(json.dumps(TRAIN_CONFIG, ensure_ascii=False, indent=2))"
    )

    cells = [
        markdown(
            f"""
            # mMARCO MiniLM cross-encoder: fine-tuning на 2×Tesla T4

            Full fine-tuning модели
            `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` на ручной разметке
            одинаковых товарных пар. PEFT и LoRA не используются, поэтому запуск не
            зависит от установленной в Kaggle версии `torchao`.

            Dataset: `{dataset_reference}`. Код проверяется по manifest до запуска.
            Source fingerprint: `{expected_bundle_hash}`.

            Все изменяемые гиперпараметры находятся в одной следующей ячейке
            `TRAIN_CONFIG`. Локальный источник этой ячейки —
            `configs/cross_encoder_minilm.json`.
            """
        ),
        code(
            f"""
            import hashlib
            import json
            import os
            import shutil
            import subprocess
            import sys
            import time
            import zipfile
            from pathlib import Path, PurePosixPath

            INPUT_ROOT = Path("/kaggle/input")
            WORKING_ROOT = Path("/kaggle/working")
            TEMP_ROOT = Path("/kaggle/temp/product_matching_minilm")
            PROJECT_ROOT = WORKING_ROOT / "product_matching"
            OUTPUT_DIR = WORKING_ROOT / "minilm_cross_encoder"
            PREPARED_DIR = TEMP_ROOT / "prepared"
            TOKEN_CACHE_DIR = TEMP_ROOT / "token_cache"
            RUNTIME_CONFIG_PATH = WORKING_ROOT / "cross_encoder_config.json"
            TRAIN_LOG = WORKING_ROOT / "minilm_training.log"
            EXPECTED_BUNDLE_SHA256 = {expected_bundle_hash!r}

            def exactly_one(filename):
                candidates = list(INPUT_ROOT.glob(f"**/{{filename}}"))
                if len(candidates) != 1:
                    raise RuntimeError(
                        f"Expected exactly one {{filename!r}} in attached datasets, "
                        f"found {{candidates}}"
                    )
                return candidates[0]

            def file_sha256(path):
                digest = hashlib.sha256()
                with path.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                return digest.hexdigest()

            items_path = exactly_one("items_human.parquet")
            matches_path = exactly_one("matches.parquet")
            manifest_path = exactly_one({shared.MANIFEST_NAME!r})
            attached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            bundle_candidates = list(INPUT_ROOT.glob("**/{shared.BUNDLE_NAME}"))
            bundle_candidates.extend(
                path
                for path in INPUT_ROOT.glob("**/{Path(shared.BUNDLE_NAME).stem}")
                if path.is_dir()
            )
            if len(bundle_candidates) != 1:
                raise RuntimeError(
                    "Expected exactly one source bundle as ZIP or expanded directory, "
                    f"found {{bundle_candidates}}"
                )
            bundle_path = bundle_candidates[0]

            TEMP_ROOT.mkdir(parents=True, exist_ok=True)
            PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
            if bundle_path.is_file():
                actual_bundle_hash = file_sha256(bundle_path)
                if actual_bundle_hash != EXPECTED_BUNDLE_SHA256:
                    raise RuntimeError(
                        "Attached source bundle does not match this notebook: "
                        f"expected {{EXPECTED_BUNDLE_SHA256}}, got {{actual_bundle_hash}}"
                    )
                with zipfile.ZipFile(bundle_path) as archive:
                    for member in archive.namelist():
                        member_path = PurePosixPath(member)
                        if member_path.is_absolute() or ".." in member_path.parts:
                            raise RuntimeError(f"Unsafe code-bundle member: {{member!r}}")
                    archive.extractall(PROJECT_ROOT)
                bundle_layout = "zip"
            else:
                shutil.copytree(bundle_path, PROJECT_ROOT, dirs_exist_ok=True)
                bundle_layout = "expanded_directory"

            expected_source_manifest = attached_manifest["code_bundle"]["source"]
            bundled_source_manifest = json.loads(
                (PROJECT_ROOT / "source_manifest.json").read_text(encoding="utf-8")
            )
            if bundled_source_manifest != expected_source_manifest:
                raise RuntimeError("source_manifest.json does not match dataset manifest")
            for relative_name, expected in expected_source_manifest["files"].items():
                relative_path = PurePosixPath(relative_name)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise RuntimeError(f"Unsafe source manifest path: {{relative_name!r}}")
                source_path = PROJECT_ROOT.joinpath(*relative_path.parts)
                if not source_path.is_file():
                    raise RuntimeError(f"Bundled source file is missing: {{relative_name}}")
                actual_hash = file_sha256(source_path)
                if actual_hash != expected["sha256"]:
                    raise RuntimeError(
                        f"Source hash mismatch for {{relative_name}}: "
                        f"expected {{expected['sha256']}}, got {{actual_hash}}"
                    )

            print(f"items:   {{items_path}} ({{items_path.stat().st_size / 2**20:.1f}} MiB)")
            print(f"matches: {{matches_path}} ({{matches_path.stat().st_size / 2**20:.1f}} MiB)")
            print(
                f"source:  {{bundle_path}} (layout={{bundle_layout}}, "
                f"fingerprint={{EXPECTED_BUNDLE_SHA256}})"
            )
            print(subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True).stdout)
            """
        ),
        markdown("## Гиперпараметры эксперимента"),
        code(config_source),
        markdown(
            """
            ## Зависимости и component-disjoint split

            Для этого trainer не устанавливается PEFT. Подготовленные данные и token
            cache пишутся в `/kaggle/temp`; в outputs останутся только модель, конфиг,
            training report и лог.
            """
        ),
        code(
            """
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
            prepare_command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts/prepare_human_data.py"),
                "--items",
                str(items_path),
                "--matches",
                str(matches_path),
                "--output-dir",
                str(PREPARED_DIR),
            ]
            print("$", " ".join(prepare_command), flush=True)
            subprocess.run(prepare_command, check=True, cwd=PROJECT_ROOT)
            """
        ),
        markdown(
            """
            ## DDP training и финальная validation

            Два процесса используют по одной T4. Training script строит компактный
            mmap token cache, логирует ход токенизации, а затем каждые `log_every`
            шагов печатает loss, throughput, VRAM и ETA эпохи. Validation выполняется
            только один раз после всех эпох.
            """
        ),
        code(
            """
            train_command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                str(PROJECT_ROOT / "scripts/train_cross_encoder.py"),
                "--config",
                str(RUNTIME_CONFIG_PATH),
                "--prepared-dir",
                str(PREPARED_DIR),
                "--output-dir",
                str(OUTPUT_DIR),
                "--token-cache-dir",
                str(TOKEN_CACHE_DIR),
            ]
            training_environment = os.environ.copy()
            training_environment.update(
                {
                    "OMP_NUM_THREADS": "2",
                    # Keep the Rust tokenizer parallel while rank 0 builds the
                    # cache. The training script disables it before DataLoader
                    # workers are forked.
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "PYTHONUNBUFFERED": "1",
                }
            )
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
            print(f"Training process finished in {training_wall_seconds / 3600:.2f} hours")
            """
        ),
        markdown("## Итоговые outputs"),
        code(
            """
            report_path = OUTPUT_DIR / "training_report.json"
            if not report_path.is_file():
                raise RuntimeError(f"Training finished without report: {report_path}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            completion = {
                "status": "complete",
                "model": TRAIN_CONFIG["model"],
                "code_bundle_sha256": EXPECTED_BUNDLE_SHA256,
                "training_wall_seconds": training_wall_seconds,
                "training_report": report,
            }
            completion_path = WORKING_ROOT / "notebook_completed.json"
            completion_path.write_text(
                json.dumps(completion, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            print(json.dumps(completion, ensure_ascii=False, indent=2, default=str))
            print("Saved outputs:")
            for path in sorted(OUTPUT_DIR.rglob("*")):
                if path.is_file():
                    print(f"  {path.relative_to(WORKING_ROOT)}: {path.stat().st_size / 2**20:.2f} MiB")
            print(f"  {TRAIN_LOG.name}: {TRAIN_LOG.stat().st_size / 2**20:.2f} MiB")
            print(f"  {completion_path.name}: {completion_path.stat().st_size / 2**20:.2f} MiB")
            """
        ),
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
                "dataset": dataset_reference,
                "bundle_sha256": expected_bundle_hash,
                "config": training_config,
                "expected_gpus": 2,
                "gpu_type": "NVIDIA Tesla T4",
                "validation_schedule": "after_training",
            },
        }
    )
    return notebook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the configurable MiniLM cross-encoder Kaggle notebook"
    )
    parser.add_argument("--owner", help="Kaggle username; defaults to KAGGLE_USERNAME in .env")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-dir", type=Path, default=shared.DEFAULT_DATASET_DIR)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_file = args.env_file if args.env_file.is_absolute() else ROOT / args.env_file
    owner = args.owner or shared.dotenv_username(env_file)
    if not owner:
        raise SystemExit("Set KAGGLE_USERNAME in .env or pass --owner")
    dataset_dir = args.dataset_dir if args.dataset_dir.is_absolute() else ROOT / args.dataset_dir
    notebook_path = args.notebook if args.notebook.is_absolute() else ROOT / args.notebook
    config_path = args.config if args.config.is_absolute() else ROOT / args.config

    training_config = load_training_config(config_path)
    manifest = shared.build_dataset(dataset_dir, owner)
    notebook = build_notebook(manifest, training_config)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, notebook_path)
    bundle = manifest["code_bundle"]
    assert isinstance(bundle, dict)
    print(f"Wrote notebook: {notebook_path.relative_to(ROOT)}")
    print(f"Dataset reference: {manifest['dataset']}")
    print(f"Code bundle SHA-256: {bundle['sha256']}")
    print(f"Training config: {config_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
