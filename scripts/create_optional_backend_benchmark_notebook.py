#!/usr/bin/env python3
"""Generate a short Kaggle benchmark for SentenceTransformers and vLLM."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "notebooks/inference_benchmarks"
OUTPUT_PATH = OUTPUT_DIR / "bge_minilm_vllm_retry_2xt4.ipynb"
CHECKPOINT_DATASET_SLUG = "product-matching-inference-checkpoints-v1"
SPREADSHEET_ID = "1CtqT52XOrFyHfFt6rCiOMlnq6snJMlsMOJ0ubH79ikA"
WORKSHEET = "inference_benchmarks"


def cell(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(source).strip())


def markdown(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def source_bundle() -> tuple[str, str]:
    sources = {
        "runtime_optional_backend_benchmark.py": (
            ROOT / "scripts/runtime_optional_backend_benchmark.py"
        ).read_text(encoding="utf-8"),
        "runtime_inference_benchmark.py": (
            ROOT / "scripts/runtime_inference_benchmark.py"
        ).read_text(encoding="utf-8"),
        "google_sheets_logger.py": (
            ROOT / "src/google_sheets_logger.py"
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
            # BGE + MiniLM: vLLM scoring retry

            Targeted 5k-pair vLLM benchmark after the first run established that
            SentenceTransformers is slower than native Transformers. It does not
            train models or repeat either completed backend benchmark.
            """
        ),
        cell(
            f"""
            import base64, gzip, hashlib, json, os, subprocess, sys, time, uuid
            from datetime import datetime, timezone
            from pathlib import Path

            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
            os.environ["DO_NOT_TRACK"] = "1"

            INPUT_ROOT = Path("/kaggle/input")
            WORKING_ROOT = Path("/kaggle/working")
            PROJECT_ROOT = WORKING_ROOT / "optional_backend_project"
            CHECKPOINT_ROOT = WORKING_ROOT / "optional_backend_checkpoints"
            OUTPUT_DIR = WORKING_ROOT / "optional_backend_results"
            MANIFEST_NAME = "inference_benchmark_manifest.json"
            RUN_ID = uuid.uuid4().hex
            STARTED = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
            BUNDLE_SHA256 = {bundle_sha256!r}
            PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
            CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            sources = json.loads(gzip.decompress(base64.b64decode({bundle!r})).decode("utf-8"))
            canonical = json.dumps(sources, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode("utf-8")
            if hashlib.sha256(canonical).hexdigest() != BUNDLE_SHA256:
                raise RuntimeError("Embedded source bundle changed")
            for relative, content in sources.items():
                destination = PROJECT_ROOT / relative
                destination.write_text(content, encoding="utf-8")
            """
        ),
        markdown("## Install vLLM"),
        cell(
            """
            install_started = time.perf_counter()
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet",
                 "--disable-pip-version-check", "--upgrade-strategy", "only-if-needed",
                 "vllm==0.14.0"],
                check=True,
            )
            install_seconds = time.perf_counter() - install_started
            print(f"vLLM installed in {install_seconds:.1f}s", flush=True)
            """
        ),
        markdown("## Reconstruct BGE and MiniLM byte-for-byte"),
        cell(
            """
            def sha256_file(path, chunk_size=8 * 1024 * 1024):
                digest = hashlib.sha256()
                with Path(path).open("rb") as source:
                    while chunk := source.read(chunk_size):
                        digest.update(chunk)
                return digest.hexdigest()

            manifest_matches = list(INPUT_ROOT.glob(f"**/{MANIFEST_NAME}"))
            if len(manifest_matches) != 1:
                raise RuntimeError(f"Expected one manifest, found {manifest_matches}")
            manifest_path = manifest_matches[0]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            input_by_name = {path.name: path for path in manifest_path.parent.iterdir() if path.is_file()}
            for model_name in ("bge", "minilm"):
                declaration = manifest["models"][model_name]
                model_dir = CHECKPOINT_ROOT / model_name
                model_dir.mkdir(parents=True, exist_ok=True)
                for item in declaration["direct_files"]:
                    source = input_by_name[item["staged_name"]]
                    if source.stat().st_size != item["bytes"] or sha256_file(source) != item["sha256"]:
                        raise RuntimeError(f"Changed direct file: {source}")
                    destination = model_dir / item["filename"]
                    destination.symlink_to(source)
                model_info = declaration["model"]
                destination = model_dir / model_info["filename"]
                with destination.open("wb") as output:
                    for part in model_info["parts"]:
                        source = input_by_name[part["staged_name"]]
                        with source.open("rb") as stream:
                            while chunk := stream.read(8 * 1024 * 1024):
                                output.write(chunk)
                if destination.stat().st_size != model_info["bytes"] or sha256_file(destination) != model_info["sha256"]:
                    raise RuntimeError(f"Reconstructed model differs: {model_name}")
                print(f"Reconstructed {model_name}: {destination.stat().st_size:,} bytes")
            """
        ),
        markdown("## Run two isolated vLLM benchmarks"),
        cell(
            """
            benchmark_started = time.perf_counter()
            commands = []
            for model_name in ("bge", "minilm"):
                commands.append([
                    sys.executable, str(PROJECT_ROOT / "runtime_optional_backend_benchmark.py"),
                    "--backend", "vllm", "--model", model_name,
                    "--input-root", str(INPUT_ROOT),
                    "--checkpoint-root", str(CHECKPOINT_ROOT),
                    "--manifest", str(manifest_path),
                    "--output-dir", str(OUTPUT_DIR),
                    "--sample-size", "5000",
                ])
            failures = []
            for command in commands:
                print("$", " ".join(command), flush=True)
                result = subprocess.run(command, cwd=PROJECT_ROOT)
                if result.returncode:
                    failures.append({"command": command, "exit_code": result.returncode})
            benchmark_seconds = time.perf_counter() - benchmark_started
            reports = []
            for path in sorted(OUTPUT_DIR.glob("*.json")):
                reports.append(json.loads(path.read_text(encoding="utf-8")))
            if len(reports) != 2:
                raise RuntimeError(f"Expected two reports, found {len(reports)}; failures={failures}")

            import pandas as pd
            results = pd.DataFrame(reports)
            results.to_csv(OUTPUT_DIR / "optional_backend_results.csv", index=False)
            native = {
                "bge": {"load_seconds": 22.406188576, "inference_pipeline_seconds": 84.937805272,
                        "backend_wall_seconds": 107.343993848, "pairs_per_second": 58.866602263},
                "minilm": {"load_seconds": 0.884644674, "inference_pipeline_seconds": 14.463021911,
                           "backend_wall_seconds": 15.347666585, "pairs_per_second": 345.709218362},
            }
            summary = {
                "status": "complete",
                "sample_size": 5000,
                "install_seconds": install_seconds,
                "benchmark_seconds": benchmark_seconds,
                "native_t4_reference": native,
                "sentence_transformers_t4_reference": {
                    "bge": {"backend_wall_seconds": 121.416213019,
                            "pairs_per_second": 41.960163559, "equivalent": True},
                    "minilm": {"backend_wall_seconds": 20.171018391,
                               "pairs_per_second": 264.047816218, "equivalent": True},
                },
                "results": reports,
                "failures": failures,
                "note": "T4 backend selection only; final H100 timing still requires Docker",
            }
            (OUTPUT_DIR / "optional_backend_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
            display_columns = [
                "model", "backend", "status", "load_seconds", "inference_seconds",
                "backend_wall_seconds", "pairs_per_second", "pearson", "ap_delta", "error",
            ]
            print(results.reindex(columns=display_columns).fillna("").to_string(index=False))
            """
        ),
        markdown("## Completion marker and Google Sheets"),
        cell(
            f"""
            completed_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
            completion = {{
                "status": "complete", "run_id": RUN_ID, "started_at_utc": STARTED,
                "completed_at_utc": completed_at,
                "experiment": "bge_minilm_vllm_retry_v2",
                "sample_size": 5000, "install_seconds": install_seconds,
                "benchmark_seconds": benchmark_seconds,
                "artifacts": {{"csv": "optional_backend_results/optional_backend_results.csv",
                              "summary": "optional_backend_results/optional_backend_summary.json"}},
            }}
            (WORKING_ROOT / "notebook_completed.json").write_text(
                json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            sync_result = {{"status": "not_attempted"}}
            try:
                import importlib.util, requests
                spec = importlib.util.spec_from_file_location(
                    "optional_backend_sheets", PROJECT_ROOT / "google_sheets_logger.py"
                )
                logger = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = logger
                spec.loader.exec_module(logger)
                credentials_json = logger.kaggle_service_account_json(input_root=INPUT_ROOT)
                client = logger.SheetsRestClient(
                    spreadsheet_id={SPREADSHEET_ID!r},
                    access_token=logger.service_account_token(credentials_json),
                    request=requests.request,
                )
                headers = ["row_id", "run_id", "completed_at_utc", "model", "backend", "phase",
                           "status", "pairs", "batch", "max_length", "precision", "load_seconds",
                           "tokenize_seconds", "inference_seconds", "total_seconds", "pairs_per_second",
                           "peak_vram_gib", "pearson", "spearman", "mean_abs_difference",
                           "max_abs_difference", "macro_ap", "reference_macro_ap", "ap_delta", "error"]
                worksheet = {WORKSHEET!r}
                metadata = client.metadata()
                titles = {{sheet["properties"]["title"] for sheet in metadata.get("sheets", [])}}
                if worksheet not in titles:
                    client.batch_update_spreadsheet([{{"addSheet": {{"properties": {{
                        "title": worksheet, "gridProperties": {{"rowCount": 5000,
                                                                  "columnCount": len(headers)}}
                    }}}}}}])
                client.update_values(f"'{{worksheet}}'!A1:Y1", [headers])
                sheet_rows = []
                for index, report in enumerate(reports):
                    values = {{
                        "row_id": f"{{RUN_ID}}:{{index}}", "run_id": RUN_ID,
                        "completed_at_utc": completed_at, "phase": "sample",
                        "tokenize_seconds": "", "total_seconds": report.get("backend_wall_seconds", ""),
                        **report,
                    }}
                    sheet_rows.append([values.get(header, "") for header in headers])
                client.append_values_once(f"'{{worksheet}}'!A:Y", sheet_rows)
                sync_result = {{"status": "synced", "rows_appended": len(sheet_rows)}}
            except Exception as error:
                sync_result = {{"status": "pending", "error": f"{{type(error).__name__}}: {{error}}"[:500]}}
                (WORKING_ROOT / "sheets_sync_pending.json").write_text(
                    json.dumps(sync_result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            (WORKING_ROOT / "google_sheets_sync.json").write_text(
                json.dumps(sync_result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps({{"completion": completion, "sheets": sync_result}},
                             ensure_ascii=False, indent=2))
            """
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    for index, notebook_cell in enumerate(notebook.cells):
        notebook_cell.id = hashlib.sha256(f"optional-backends-v1:{index}".encode()).hexdigest()[:12]
        notebook_cell.metadata["tags"] = ["optional-backend-benchmark"]
    notebook.metadata.update(
        {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "product_matching": {
                "template": "bge_minilm_vllm_retry_v2",
                "sample_size": 5000,
                "worksheet": WORKSHEET,
            },
        }
    )
    nbf.validate(notebook)
    return notebook


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), OUTPUT_PATH)
    size = OUTPUT_PATH.stat().st_size
    if size >= 1_000_000:
        raise RuntimeError(f"Generated notebook exceeds Kaggle's 1 MB source limit: {size:,}")
    print(f"Created {OUTPUT_PATH} ({size:,} bytes)")


if __name__ == "__main__":
    main()
