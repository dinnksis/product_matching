#!/usr/bin/env python3
"""Build the private Kaggle training dataset payload and 2xT4 notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = ROOT / "notebooks" / "qwen3_reranker_training_2xt4.ipynb"
DEFAULT_DATASET_DIR = (
    ROOT / ".kaggle" / "datasets" / "product-matching-qwen-training"
)
DATASET_SLUG = "product-matching-qwen-training"
BUNDLE_NAME = "product_matching_training_code.zip"
MANIFEST_NAME = "training_bundle_manifest.json"
EXPERIMENT_SPREADSHEET_ID = "1CtqT52XOrFyHfFt6rCiOMlnq6snJMlsMOJ0ubH79ikA"
RAW_DATA_FILES = (
    Path("data/items_human.parquet"),
    Path("data/matches.parquet"),
)
BUNDLE_FILES = (
    Path("configs/attribute_boosting_v2.json"),
    Path("configs/embedding_boosting.json"),
    Path("requirements-embedding-boosting.txt"),
    Path("requirements-gpu.txt"),
    Path("requirements-cross-encoder.txt"),
    Path("src/__init__.py"),
    Path("src/cross_encoder_training.py"),
    Path("src/blended_data.py"),
    Path("src/data_pipeline.py"),
    Path("src/pair_features.py"),
    Path("src/qwen_reranker.py"),
    Path("src/qwen_training.py"),
    Path("src/embedding_boosting.py"),
    Path("src/attribute_boosting_v2.py"),
    Path("scripts/prepare_human_data.py"),
    Path("scripts/prepare_balanced_llm_data.py"),
    Path("scripts/train_cross_encoder.py"),
    Path("scripts/train_qwen_names.py"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dotenv_username(path: Path) -> str | None:
    if not path.exists():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != "KAGGLE_USERNAME":
            continue
        value = value.strip().split(" #", 1)[0].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        return value or None
    return None


def copy_if_changed(source: Path, destination: Path, source_hash: str) -> None:
    if (
        destination.is_file()
        and destination.stat().st_size == source.stat().st_size
        and sha256(destination) == source_hash
    ):
        return
    shutil.copy2(source, destination)


def write_code_bundle(destination: Path) -> dict[str, object]:
    files: dict[str, dict[str, object]] = {}
    for relative_path in BUNDLE_FILES:
        source = ROOT / relative_path
        if not source.is_file():
            raise FileNotFoundError(f"Required bundle file is missing: {source}")
        files[relative_path.as_posix()] = {
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
        }
    source_manifest = {
        "schema_version": 1,
        "files": files,
    }
    manifest_bytes = (
        json.dumps(source_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    # Fixed timestamps and sorted paths make an unchanged source bundle byte-for-byte
    # reproducible, which also gives the Kaggle dataset version a stable fingerprint.
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        payloads = {
            **{
                path.as_posix(): (ROOT / path).read_bytes()
                for path in BUNDLE_FILES
            },
            "source_manifest.json": manifest_bytes,
        }
        for archive_name, payload in sorted(payloads.items()):
            info = zipfile.ZipInfo(archive_name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload)
    return source_manifest


def build_dataset(dataset_dir: Path, owner: str) -> dict[str, object]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    raw_manifest: dict[str, dict[str, object]] = {}
    for relative_path in RAW_DATA_FILES:
        source = ROOT / relative_path
        if not source.is_file():
            raise FileNotFoundError(
                f"Required training data is missing: {source}. "
                "Download the competition data before building the Kaggle payload."
            )
        file_hash = sha256(source)
        destination = dataset_dir / source.name
        copy_if_changed(source, destination, file_hash)
        raw_manifest[source.name] = {
            "bytes": source.stat().st_size,
            "sha256": file_hash,
        }

    bundle_path = dataset_dir / BUNDLE_NAME
    source_manifest = write_code_bundle(bundle_path)
    manifest = {
        "schema_version": 1,
        "dataset": f"{owner}/{DATASET_SLUG}",
        "raw_data": raw_manifest,
        "code_bundle": {
            "filename": BUNDLE_NAME,
            "bytes": bundle_path.stat().st_size,
            "sha256": sha256(bundle_path),
            "source": source_manifest,
        },
    }
    (dataset_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "title": "Product Matching Training Bundle",
        "id": f"{owner}/{DATASET_SLUG}",
        "licenses": [{"name": "unknown"}],
        "description": (
            "Private human-labelled product-matching parquet data plus exact source "
            "bundles for cross-encoder training. Not for publication."
        ),
    }
    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def markdown(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(text).strip())


def google_sheets_tracking_cells() -> list[nbf.NotebookNode]:
    """Cells shared by every generated training notebook."""
    logger_source = (ROOT / "src/google_sheets_logger.py").read_text(encoding="utf-8")
    return [
        markdown(
            """
            ## Автоматическая запись результатов в Google Sheets

            Успешный отчёт синхронизируется по `run_id`: повторный запуск этой
            ячейки обновит те же строки, а не создаст дубли. Сначала используется
            Kaggle Secret `GOOGLE_SERVICE_ACCOUNT_JSON`, а если он не подключён —
            ключ из приватного Dataset `ecom-matching-google-sheets-credentials`.
            """
        ),
        code(
            f"""
            spreadsheet_id = {EXPERIMENT_SPREADSHEET_ID!r}
            sync_path = WORKING_ROOT / "google_sheets_sync.json"
            pending_path = WORKING_ROOT / "sheets_sync_pending.json"
            logger_module = None
            try:
                import importlib.util

                embedded_logger_path = WORKING_ROOT / "google_sheets_logger.py"
                embedded_logger_path.write_text({logger_source!r}, encoding="utf-8")
                logger_spec = importlib.util.spec_from_file_location(
                    "product_matching_google_sheets_logger", embedded_logger_path
                )
                if logger_spec is None or logger_spec.loader is None:
                    raise RuntimeError("Could not load the embedded Google Sheets logger")
                logger_module = importlib.util.module_from_spec(logger_spec)
                sys.modules[logger_spec.name] = logger_module
                logger_spec.loader.exec_module(logger_module)
                try:
                    import google.auth  # noqa: F401
                    import requests  # noqa: F401
                except ImportError:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            "--quiet",
                            "--disable-pip-version-check",
                            "google-auth>=2.38,<3",
                            "requests>=2.32,<3",
                        ],
                        check=True,
                    )
                sync_result = logger_module.sync_from_kaggle_credentials(
                    spreadsheet_id=spreadsheet_id,
                    completion=completion,
                )
            except Exception as error:
                error_message = (
                    logger_module.safe_error_message(error)
                    if logger_module is not None
                    else f"{{type(error).__name__}}: logger initialization failed"
                )
                sync_result = {{
                    "status": "pending",
                    "run_id": completion["run_id"],
                    "spreadsheet_id": spreadsheet_id,
                    "error_type": type(error).__name__,
                    "error": error_message,
                }}
                pending_path.write_text(
                    json.dumps(
                        {{**sync_result, "completion": completion}},
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    ),
                    encoding="utf-8",
                )
                print(
                    "Google Sheets sync is pending; training outputs are still complete:\\n"
                    + json.dumps(sync_result, ensure_ascii=False, indent=2)
                )
            else:
                sync_result = {{"status": "synced", **sync_result}}
                pending_path.unlink(missing_ok=True)
                print(json.dumps(sync_result, ensure_ascii=False, indent=2))
            sync_path.write_text(
                json.dumps(sync_result, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            print(f"Saved {{sync_path.name}}")
            """
        ),
    ]


def experiment_run_initialization_cell() -> nbf.NotebookNode:
    """Create a stable run identity before any expensive training starts."""
    return code(
        """
        from datetime import datetime, timezone
        import uuid

        RUN_ID_PATH = WORKING_ROOT / "experiment_run_id.txt"
        RUN_STARTED_PATH = WORKING_ROOT / "experiment_started_at_utc.txt"
        if RUN_ID_PATH.is_file():
            EXPERIMENT_RUN_ID = RUN_ID_PATH.read_text(encoding="utf-8").strip()
        else:
            EXPERIMENT_RUN_ID = uuid.uuid4().hex
            RUN_ID_PATH.write_text(EXPERIMENT_RUN_ID + "\\n", encoding="utf-8")
        if RUN_STARTED_PATH.is_file():
            EXPERIMENT_STARTED_AT_UTC = RUN_STARTED_PATH.read_text(
                encoding="utf-8"
            ).strip()
        else:
            EXPERIMENT_STARTED_AT_UTC = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ).replace("+00:00", "Z")
            RUN_STARTED_PATH.write_text(
                EXPERIMENT_STARTED_AT_UTC + "\\n", encoding="utf-8"
            )
        print(
            json.dumps(
                {
                    "run_id": EXPERIMENT_RUN_ID,
                    "started_at_utc": EXPERIMENT_STARTED_AT_UTC,
                },
                ensure_ascii=False,
            )
        )
        """
    )


def build_notebook(manifest: dict[str, object]) -> nbf.NotebookNode:
    bundle = manifest["code_bundle"]
    assert isinstance(bundle, dict)
    expected_bundle_hash = str(bundle["sha256"])
    dataset_reference = str(manifest["dataset"])

    cells = [
        markdown(
            f"""
            # Qwen3-Reranker-0.6B: fine-tuning на 2×Tesla T4

            Notebook обучает reranker на ручной разметке одинаковых товарных пар.
            Он использует DDP — отдельный процесс и копию модели на каждой T4 — и
            выполняет валидацию один раз, только после всех эпох.

            Входной приватный Dataset: `{dataset_reference}`. В нём находятся два
            исходных parquet-файла и ровно та версия `src/` и training scripts, для
            которой создан notebook. Kaggle может смонтировать code bundle как ZIP
            или как автоматически распакованный каталог; оба варианта проверяются.
            SHA-256 исходного code bundle:
            `{expected_bundle_hash}`.

            Основные параметры запуска находятся в ячейке `TRAIN_CONFIG`. Значение
            `batch_size=32` задаётся на одну GPU, поэтому эффективный batch для двух
            T4 равен 64.
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
            EXPECTED_DATASET_REF = {dataset_reference!r}
            WORKING_ROOT = Path("/kaggle/working")
            TEMP_ROOT = Path("/kaggle/temp/product_matching_training")
            PROJECT_ROOT = WORKING_ROOT / "product_matching"
            OUTPUT_DIR = WORKING_ROOT / "qwen_products_lora"
            PREPARED_DIR = TEMP_ROOT / "prepared"
            TOKEN_CACHE_DIR = TEMP_ROOT / "token_cache"
            TRAIN_LOG = WORKING_ROOT / "qwen_training.log"
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
            manifest_path = exactly_one({MANIFEST_NAME!r})
            attached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            attached_bundle_hash = attached_manifest["code_bundle"]["sha256"]
            if attached_bundle_hash != EXPECTED_BUNDLE_SHA256:
                raise RuntimeError(
                    "Attached Kaggle Dataset version does not match this notebook: "
                    f"expected bundle {{EXPECTED_BUNDLE_SHA256}}, "
                    f"got {{attached_bundle_hash}}. Re-submit the kernel only after "
                    "the new Dataset version reports ready."
                )
            bundle_candidates = list(INPUT_ROOT.glob("**/{BUNDLE_NAME}"))
            bundle_candidates.extend(
                path
                for path in INPUT_ROOT.glob("**/{Path(BUNDLE_NAME).stem}")
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
                raise RuntimeError("Expanded source_manifest.json does not match dataset manifest")
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
        experiment_run_initialization_cell(),
        markdown(
            """
            ## Зависимости и подготовка данных

            PyTorch из Kaggle image сохраняется, если он удовлетворяет диапазону в
            `requirements-gpu.txt`; доустанавливаются Transformers и PEFT. Подготовленные
            parquet-файлы и mmap token cache пишутся в `/kaggle/temp`, поэтому они не
            раздувают скачиваемый output notebook.
            """
        ),
        code(
            """
            requirements_path = PROJECT_ROOT / "requirements-gpu.txt"
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
                    str(requirements_path),
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
            ## Параметры обучения

            `batch_size` и `eval_batch_size` указаны на один DDP process/GPU.
            Четыре DataLoader worker'а работают параллельно: по два рядом с каждой T4.
            `attention_mlp` обучает LoRA не только на attention projections, но и на MLP.
            """
        ),
        code(
            """
            TRAIN_CONFIG = {
                "model": "Qwen/Qwen3-Reranker-0.6B",
                "epochs": 2,
                "batch_size": 32,
                "eval_batch_size": 32,
                "gradient_accumulation": 1,
                "max_length": 384,
                "learning_rate": 2e-4,
                "lora_rank": 16,
                "lora_targets": "attention_mlp",
                "sampling": "category_label",
                "dataloader_workers": 2,
                "prefetch_factor": 2,
                "tokenization_batch_size": 512,
                "log_every": 25,
                "seed": 42,
            }
            print(json.dumps(TRAIN_CONFIG, ensure_ascii=False, indent=2))
            """
        ),
        markdown(
            """
            ## DDP fine-tuning

            `torch.distributed.run` создаёт два процесса, поэтому обе T4 вычисляют
            разные части глобального batch. Вывод одновременно виден в notebook и
            сохраняется в `qwen_training.log`. Валидация запускается training script
            один раз — после завершения последней эпохи.
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
                str(PROJECT_ROOT / "scripts/train_qwen_names.py"),
                "--prepared-dir",
                str(PREPARED_DIR),
                "--output-dir",
                str(OUTPUT_DIR),
                "--token-cache-dir",
                str(TOKEN_CACHE_DIR),
                "--model",
                TRAIN_CONFIG["model"],
                "--epochs",
                str(TRAIN_CONFIG["epochs"]),
                "--batch-size",
                str(TRAIN_CONFIG["batch_size"]),
                "--eval-batch-size",
                str(TRAIN_CONFIG["eval_batch_size"]),
                "--gradient-accumulation",
                str(TRAIN_CONFIG["gradient_accumulation"]),
                "--max-length",
                str(TRAIN_CONFIG["max_length"]),
                "--learning-rate",
                str(TRAIN_CONFIG["learning_rate"]),
                "--lora-rank",
                str(TRAIN_CONFIG["lora_rank"]),
                "--lora-targets",
                TRAIN_CONFIG["lora_targets"],
                "--sampling",
                TRAIN_CONFIG["sampling"],
                "--dataloader-workers",
                str(TRAIN_CONFIG["dataloader_workers"]),
                "--prefetch-factor",
                str(TRAIN_CONFIG["prefetch_factor"]),
                "--tokenization-batch-size",
                str(TRAIN_CONFIG["tokenization_batch_size"]),
                "--log-every",
                str(TRAIN_CONFIG["log_every"]),
                "--seed",
                str(TRAIN_CONFIG["seed"]),
            ]
            training_environment = os.environ.copy()
            training_environment.update(
                {
                    "OMP_NUM_THREADS": "2",
                    "TOKENIZERS_PARALLELISM": "false",
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
        markdown("## Итоговый отчёт и сохраняемые outputs"),
        code(
            """
            report_path = OUTPUT_DIR / "training_report.json"
            if not report_path.is_file():
                raise RuntimeError(f"Training finished without report: {report_path}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            completion = {
                "status": "complete",
                "run_id": EXPERIMENT_RUN_ID,
                "started_at_utc": EXPERIMENT_STARTED_AT_UTC,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ).replace("+00:00", "Z"),
                "experiment": OUTPUT_DIR.name,
                "model": report.get("args", {}).get("model", ""),
                "dataset_ref": EXPECTED_DATASET_REF,
                "kaggle_kernel_ref": (
                    os.getenv("KAGGLE_KERNEL_RUN_ID")
                    or os.getenv("KAGGLE_KERNEL_INFERENCE_RUN_ID")
                    or ""
                ),
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
        *google_sheets_tracking_cells(),
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
                "expected_gpus": 2,
                "gpu_type": "NVIDIA Tesla T4",
                "validation_schedule": "after_training",
            },
        }
    )
    return notebook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the private training Dataset payload and Kaggle notebook"
    )
    parser.add_argument("--owner", help="Kaggle username; defaults to KAGGLE_USERNAME in .env")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_file = args.env_file if args.env_file.is_absolute() else ROOT / args.env_file
    owner = args.owner or dotenv_username(env_file)
    if not owner:
        raise SystemExit("Set KAGGLE_USERNAME in .env or pass --owner")
    dataset_dir = (
        args.dataset_dir if args.dataset_dir.is_absolute() else ROOT / args.dataset_dir
    )
    notebook_path = args.notebook if args.notebook.is_absolute() else ROOT / args.notebook

    manifest = build_dataset(dataset_dir, owner)
    notebook = build_notebook(manifest)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, notebook_path)
    payload_size = sum(path.stat().st_size for path in dataset_dir.iterdir() if path.is_file())
    bundle = manifest["code_bundle"]
    assert isinstance(bundle, dict)
    print(f"Wrote notebook: {notebook_path.relative_to(ROOT)}")
    print(
        f"Wrote private dataset payload: {dataset_dir.relative_to(ROOT)} "
        f"({payload_size / 2**20:.1f} MiB)"
    )
    print(f"Dataset reference: {manifest['dataset']}")
    print(f"Code bundle SHA-256: {bundle['sha256']}")


if __name__ == "__main__":
    main()
