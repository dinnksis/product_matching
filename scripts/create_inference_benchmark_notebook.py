#!/usr/bin/env python3
"""Generate the single background Kaggle inference benchmark notebook."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "notebooks" / "inference_benchmarks"
OUTPUT_PATH = OUTPUT_DIR / "three_model_inference_backend_benchmark_2xt4.ipynb"
CHECKPOINT_DATASET_SLUG = "product-matching-inference-checkpoints-v1"
RAW_DATASET_REF = "dinakepecheva/e-cup-human-data"
SPREADSHEET_ID = "1CtqT52XOrFyHfFt6rCiOMlnq6snJMlsMOJ0ubH79ikA"
WORKSHEET = "inference_benchmarks"


def markdown(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(source).strip())


def source_bundle() -> tuple[str, str]:
    sources = {
        "runtime_inference_benchmark.py": (
            ROOT / "scripts/runtime_inference_benchmark.py"
        ).read_text(encoding="utf-8"),
        "src/serialization_ablation.py": (
            ROOT / "src/serialization_ablation.py"
        ).read_text(encoding="utf-8"),
        "google_sheets_logger.py": (
            ROOT / "src/google_sheets_logger.py"
        ).read_text(encoding="utf-8"),
        "requirements-inference-benchmark.txt": (
            ROOT / "requirements-inference-benchmark.txt"
        ).read_text(encoding="utf-8"),
    }
    canonical = json.dumps(
        sources, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return (
        base64.b64encode(gzip.compress(canonical, compresslevel=9, mtime=0)).decode("ascii"),
        hashlib.sha256(canonical).hexdigest(),
    )


def build_notebook() -> nbf.NotebookNode:
    bundle, bundle_sha256 = source_bundle()
    cells = [
        markdown(
            """
            # Frozen BGE + MiniLM + RuModernBERT inference benchmark

            No training and no checkpoint modification. The notebook benchmarks one
            GPU at a time so backend timings remain comparable. Kaggle T4 timings are
            used to select implementations; the final 6/13 minute verdict must still
            be measured in the competition H100 Docker runner.
            """
        ),
        code(
            f"""
            import json
            import os
            import subprocess
            import sys
            import time
            import uuid
            from datetime import datetime, timezone
            from pathlib import Path

            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
            os.environ.setdefault("OMP_NUM_THREADS", "20")
            os.environ.setdefault("MKL_NUM_THREADS", "20")
            os.environ.setdefault("RAYON_NUM_THREADS", "20")
            os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

            INPUT_ROOT = Path("/kaggle/input")
            WORKING_ROOT = Path("/kaggle/working")
            PROJECT_ROOT = WORKING_ROOT / "inference_benchmark_project"
            CHECKPOINT_ROOT = WORKING_ROOT / "inference_benchmark_checkpoints"
            OUTPUT_DIR = WORKING_ROOT / "inference_benchmark_results"
            MANIFEST_NAME = "inference_benchmark_manifest.json"
            CODE_BUNDLE_SHA256 = {bundle_sha256!r}
            EXPERIMENT_RUN_ID = uuid.uuid4().hex
            EXPERIMENT_STARTED_AT_UTC = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ).replace("+00:00", "Z")
            PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
            CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for this benchmark")
            compute_capability = torch.cuda.get_device_capability(0)
            native_bf16 = torch.cuda.is_bf16_supported() and compute_capability[0] >= 8
            gpu_report = {{
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "visible_gpus": torch.cuda.device_count(),
                "gpu_0": torch.cuda.get_device_name(0),
                "compute_capability": list(compute_capability),
                "torch_reports_bf16": torch.cuda.is_bf16_supported(),
                "native_bf16_enabled": native_bf16,
                "benchmark_precision": "bfloat16" if native_bf16 else "float16",
            }}
            print(json.dumps(gpu_report, ensure_ascii=False, indent=2))
            """
        ),
        markdown("## Reconstruct and verify the frozen inputs"),
        code(
            f"""
            import base64
            import gzip
            import hashlib

            EMBEDDED_BUNDLE_B64 = {bundle!r}
            sources = json.loads(
                gzip.decompress(base64.b64decode(EMBEDDED_BUNDLE_B64)).decode("utf-8")
            )
            canonical = json.dumps(
                sources, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            if hashlib.sha256(canonical).hexdigest() != CODE_BUNDLE_SHA256:
                raise RuntimeError("Embedded benchmark source bundle changed")
            for relative, content in sources.items():
                destination = PROJECT_ROOT / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(content, encoding="utf-8")

            subprocess.run(
                [
                    sys.executable, "-m", "pip", "install", "--quiet",
                    "--disable-pip-version-check", "--upgrade-strategy", "only-if-needed",
                    "-r", str(PROJECT_ROOT / "requirements-inference-benchmark.txt"),
                ],
                check=True,
            )

            def sha256_file(path, chunk_size=8 * 1024 * 1024):
                digest = hashlib.sha256()
                with Path(path).open("rb") as source:
                    while chunk := source.read(chunk_size):
                        digest.update(chunk)
                return digest.hexdigest()

            manifest_matches = list(INPUT_ROOT.glob(f"**/{{MANIFEST_NAME}}"))
            if len(manifest_matches) != 1:
                raise RuntimeError(f"Expected one {{MANIFEST_NAME}}, found {{manifest_matches}}")
            manifest_path = manifest_matches[0]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != 1:
                raise RuntimeError("Unexpected checkpoint manifest schema")

            input_by_name = {{path.name: path for path in manifest_path.parent.iterdir() if path.is_file()}}
            reconstructed = {{}}
            for model_name, declaration in manifest["models"].items():
                model_dir = CHECKPOINT_ROOT / model_name
                model_dir.mkdir(parents=True, exist_ok=True)
                for item in declaration["direct_files"]:
                    source = input_by_name[item["staged_name"]]
                    if source.stat().st_size != item["bytes"] or sha256_file(source) != item["sha256"]:
                        raise RuntimeError(f"Changed direct file: {{source}}")
                    destination = model_dir / item["filename"]
                    if destination.exists() or destination.is_symlink():
                        destination.unlink()
                    destination.symlink_to(source)
                model_info = declaration["model"]
                destination = model_dir / model_info["filename"]
                with destination.open("wb") as output:
                    for part in model_info["parts"]:
                        source = input_by_name[part["staged_name"]]
                        if source.stat().st_size != part["bytes"] or sha256_file(source) != part["sha256"]:
                            raise RuntimeError(f"Changed model shard: {{source}}")
                        with source.open("rb") as stream:
                            while chunk := stream.read(8 * 1024 * 1024):
                                output.write(chunk)
                if destination.stat().st_size != model_info["bytes"] or sha256_file(destination) != model_info["sha256"]:
                    raise RuntimeError(f"Reconstructed model differs: {{model_name}}")
                reconstructed[model_name] = {{
                    "path": str(model_dir), "bytes": destination.stat().st_size,
                    "sha256": model_info["sha256"],
                }}
            print(json.dumps({{"reconstructed": reconstructed}}, ensure_ascii=False, indent=2))
            """
        ),
        markdown("## Run the controlled benchmark"),
        code(
            """
            benchmark_started = time.perf_counter()
            command = [
                sys.executable,
                str(PROJECT_ROOT / "runtime_inference_benchmark.py"),
                "--input-root", str(INPUT_ROOT),
                "--checkpoint-root", str(CHECKPOINT_ROOT),
                "--manifest", str(manifest_path),
                "--output-dir", str(OUTPUT_DIR),
                "--sample-size", "5000",
                "--probe-size", "512",
                "--quick",
                "--skip-compile",
            ]
            print("$", " ".join(command), flush=True)
            subprocess.run(command, check=True, cwd=PROJECT_ROOT)
            benchmark_wall_seconds = time.perf_counter() - benchmark_started
            summary_path = OUTPUT_DIR / "inference_benchmark_summary.json"
            results_path = OUTPUT_DIR / "inference_benchmark_results.csv"
            if not summary_path.is_file() or not results_path.is_file():
                raise RuntimeError("Benchmark did not write its required artifacts")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            """
        ),
        markdown("## Completion report"),
        code(
            f"""
            completion = {{
                "status": "complete",
                "run_id": EXPERIMENT_RUN_ID,
                "started_at_utc": EXPERIMENT_STARTED_AT_UTC,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ).replace("+00:00", "Z"),
                "experiment": "three_model_inference_backend_benchmark_v1",
                "models": ["bge", "minilm", "rumodernbert"],
                "checkpoint_dataset": manifest["dataset"],
                "raw_dataset": {RAW_DATASET_REF!r},
                "serialization": "S2_VALUES_ONLY",
                "max_length": 384,
                "sample_size": 5000,
                "code_bundle_sha256": CODE_BUNDLE_SHA256,
                "gpu": gpu_report,
                "benchmark_wall_seconds": benchmark_wall_seconds,
                "summary": summary,
                "artifacts": {{
                    "results_csv": "inference_benchmark_results/inference_benchmark_results.csv",
                    "summary_json": "inference_benchmark_results/inference_benchmark_summary.json",
                }},
            }}
            completion_path = WORKING_ROOT / "notebook_completed.json"
            completion_path.write_text(
                json.dumps(completion, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            print(f"Saved {{completion_path}}")
            """
        ),
        markdown("## Sync every benchmark row to the same Google spreadsheet"),
        code(
            f"""
            sync_path = WORKING_ROOT / "google_sheets_sync.json"
            pending_path = WORKING_ROOT / "sheets_sync_pending.json"
            try:
                import importlib.util
                import requests

                logger_path = PROJECT_ROOT / "google_sheets_logger.py"
                spec = importlib.util.spec_from_file_location("benchmark_sheets_logger", logger_path)
                logger = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = logger
                spec.loader.exec_module(logger)
                credentials_json = logger.kaggle_service_account_json(input_root=INPUT_ROOT)
                client = logger.SheetsRestClient(
                    spreadsheet_id={SPREADSHEET_ID!r},
                    access_token=logger.service_account_token(credentials_json),
                    request=requests.request,
                )
                worksheet = {WORKSHEET!r}
                headers = [
                    "row_id", "run_id", "completed_at_utc", "model", "backend", "phase",
                    "status", "pairs", "batch", "max_length", "precision", "load_seconds",
                    "tokenize_seconds", "inference_seconds", "total_seconds", "pairs_per_second",
                    "peak_vram_gib", "pearson", "spearman", "mean_abs_difference",
                    "max_abs_difference", "macro_ap", "reference_macro_ap", "ap_delta", "error",
                ]
                metadata = client.metadata()
                titles = {{
                    sheet["properties"]["title"]: sheet["properties"]
                    for sheet in metadata.get("sheets", [])
                }}
                if worksheet not in titles:
                    client.batch_update_spreadsheet([{{
                        "addSheet": {{"properties": {{
                            "title": worksheet,
                            "gridProperties": {{"rowCount": 5000, "columnCount": len(headers)}},
                        }}}}
                    }}])
                client.update_values(f"'{{worksheet}}'!A1:Y1", [headers])
                existing = {{
                    str(row[0]) for row in client.get_values(f"'{{worksheet}}'!A2:A") if row
                }}
                result_rows = __import__("pandas").read_csv(results_path).fillna("")
                rows = []
                for index, record in result_rows.iterrows():
                    row_id = f"{{EXPERIMENT_RUN_ID}}:{{index}}"
                    if row_id in existing:
                        continue
                    values = {{"row_id": row_id, "run_id": EXPERIMENT_RUN_ID,
                              "completed_at_utc": completion["completed_at_utc"], **record.to_dict()}}
                    rows.append([values.get(header, "") for header in headers])
                if rows:
                    client.append_values_once(f"'{{worksheet}}'!A:Y", rows)
                sync_result = {{
                    "status": "synced", "spreadsheet_id": {SPREADSHEET_ID!r},
                    "worksheet": worksheet, "rows_appended": len(rows),
                }}
                pending_path.unlink(missing_ok=True)
            except Exception as error:
                sync_result = {{
                    "status": "pending", "run_id": EXPERIMENT_RUN_ID,
                    "error_type": type(error).__name__, "error": str(error)[:500],
                }}
                pending_path.write_text(
                    json.dumps({{"sync": sync_result, "completion": completion}}, ensure_ascii=False,
                               indent=2, default=str), encoding="utf-8"
                )
            sync_path.write_text(
                json.dumps(sync_result, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            print(json.dumps(sync_result, ensure_ascii=False, indent=2))
            """
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    for index, cell in enumerate(notebook.cells):
        cell.id = hashlib.sha256(f"inference-benchmark-v1:{index}".encode()).hexdigest()[:12]
        cell.metadata["tags"] = ["inference-benchmark"]
    notebook.metadata.update(
        {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "product_matching": {
                "template": "three_model_inference_backend_benchmark_v1",
                "checkpoint_dataset_slug": CHECKPOINT_DATASET_SLUG,
                "raw_dataset": RAW_DATASET_REF,
                "sample_size": 5000,
                "worksheet": WORKSHEET,
            },
        }
    )
    nbf.validate(notebook)
    return notebook


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    nbf.write(notebook, OUTPUT_PATH)
    size = OUTPUT_PATH.stat().st_size
    if size >= 1_000_000:
        raise RuntimeError(f"Generated notebook exceeds Kaggle's 1 MB source limit: {size:,}")
    print(f"Created {OUTPUT_PATH} ({size:,} bytes)")


if __name__ == "__main__":
    main()
