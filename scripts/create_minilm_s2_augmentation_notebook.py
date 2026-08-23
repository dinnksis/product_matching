#!/usr/bin/env python3
"""Build the autonomous 2xT4 MiniLM S2 combined-augmentation notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

import create_qwen_training_notebook as shared


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = ROOT / "notebooks" / "minilm_s2_combined_augmentation_2xt4.ipynb"
DEFAULT_CODE_DATASET_DIR = ROOT / ".kaggle" / "datasets" / "product-matching-minilm-s2-augmentation-code"
CODE_DATASET_SLUG = "product-matching-minilm-s2-augmentation-code"
BUNDLE_NAME = "minilm_s2_augmentation_code.zip"
MANIFEST_NAME = "minilm_s2_augmentation_manifest.json"
BUNDLE_FILES = (
    Path("configs/minilm_s2_combined_augmentation.json"),
    Path("requirements-minilm-augmentation.txt"),
    Path("src/__init__.py"),
    Path("src/cross_encoder_training.py"),
    Path("src/minilm_s2_augmentation.py"),
    Path("src/qwen_reranker.py"),
    Path("src/qwen_training.py"),
    Path("src/serialization_ablation.py"),
    Path("src/validation_audit.py"),
    Path("scripts/prepare_minilm_s2_augmentation.py"),
    Path("scripts/train_minilm_s2_augmentation.py"),
    Path("scripts/train_serialization_ablation.py"),
    Path("scripts/summarize_minilm_s2_augmentation.py"),
)


def markdown(value: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(value).strip())


def code(value: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(value).strip())


def build_code_dataset(directory: Path, owner: str) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, object]] = {}
    payloads: dict[str, bytes] = {}
    for relative in BUNDLE_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"Required source is missing: {source}")
        payload = source.read_bytes()
        files[relative.as_posix()] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        payloads[relative.as_posix()] = payload
    source_manifest = {"schema_version": 1, "files": files}
    payloads["source_manifest.json"] = (
        json.dumps(source_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    bundle_path = directory / BUNDLE_NAME
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(payloads.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload)
    manifest = {
        "schema_version": 1,
        "dataset": f"{owner}/{CODE_DATASET_SLUG}",
        "code_bundle": {
            "filename": BUNDLE_NAME,
            "bytes": bundle_path.stat().st_size,
            "sha256": shared.sha256(bundle_path),
            "source": source_manifest,
        },
    }
    (directory / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (directory / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "MiniLM S2 Combined Augmentation Code",
                "id": f"{owner}/{CODE_DATASET_SLUG}",
                "licenses": [{"name": "unknown"}],
                "description": "Private exact source bundle for the human-only two-run augmentation test.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def build_notebook(manifest: dict[str, object]) -> nbf.NotebookNode:
    bundle = manifest["code_bundle"]
    assert isinstance(bundle, dict)
    expected_hash = str(bundle["sha256"])
    expected_source_manifest = bundle["source"]
    config = json.loads(
        (ROOT / "configs/minilm_s2_combined_augmentation.json").read_text(encoding="utf-8")
    )
    cells: list[nbf.NotebookNode] = [
        markdown(
            """
            # MiniLM S2: combined training augmentation

            Two fresh one-epoch runs use the same full leakage-safe human train pool and
            deterministic validation. A is plain S2; B adds intact attribute-entry shuffle
            plus random A/B swap. Each run owns one T4. Validation uses one A-to-B forward.
            No LLM-labelled data is attached or read.
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
            import uuid
            import zipfile
            from datetime import datetime, timezone
            from pathlib import Path, PurePosixPath

            import pandas as pd

            INPUT_ROOT = Path('/kaggle/input')
            WORKING_ROOT = Path('/kaggle/working')
            TEMP_ROOT = Path('/kaggle/temp/minilm_s2_combined_augmentation')
            PROJECT_ROOT = WORKING_ROOT / 'product_matching'
            OUTPUT_DIR = WORKING_ROOT / 'minilm_s2_combined_augmentation'
            PREPARED_DIR = TEMP_ROOT / 'prepared'
            MODEL_DIR = TEMP_ROOT / 'base_model'
            CHECKPOINTS_TEMP = TEMP_ROOT / 'checkpoints'
            TOKEN_CACHE_ROOT = TEMP_ROOT / 'token_cache'
            RUNS_DIR = OUTPUT_DIR / 'runs'
            CONFIG_PATH = PROJECT_ROOT / 'configs/minilm_s2_combined_augmentation.json'
            EXPECTED_BUNDLE_SHA256 = {expected_hash!r}
            EXPECTED_SOURCE_MANIFEST = {expected_source_manifest!r}
            EXPECTED_CODE_DATASET_REF = {str(manifest['dataset'])!r}

            def exactly_one(filename):
                candidates = list(INPUT_ROOT.glob(f'**/{{filename}}'))
                if len(candidates) != 1:
                    raise RuntimeError(f'Expected exactly one {{filename!r}}, found {{candidates}}')
                return candidates[0]

            def sha256(path):
                digest = hashlib.sha256()
                with path.open('rb') as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b''):
                        digest.update(chunk)
                return digest.hexdigest()

            items_path = exactly_one('items_human.parquet')
            matches_path = exactly_one('matches.parquet')
            bundle_candidates = list(INPUT_ROOT.glob('**/{BUNDLE_NAME}'))
            bundle_candidates.extend(path for path in INPUT_ROOT.glob('**/{Path(BUNDLE_NAME).stem}') if path.is_dir())
            if len(bundle_candidates) != 1:
                raise RuntimeError(f'Expected one augmentation code bundle, found {{bundle_candidates}}')
            bundle_path = bundle_candidates[0]
            PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
            if bundle_path.is_file():
                if sha256(bundle_path) != EXPECTED_BUNDLE_SHA256:
                    raise RuntimeError('Attached code bundle hash mismatch')
                with zipfile.ZipFile(bundle_path) as archive:
                    for member in archive.namelist():
                        relative = PurePosixPath(member)
                        if relative.is_absolute() or '..' in relative.parts:
                            raise RuntimeError(f'Unsafe bundle member: {{member!r}}')
                    archive.extractall(PROJECT_ROOT)
                bundle_layout = 'zip'
            else:
                shutil.copytree(bundle_path, PROJECT_ROOT, dirs_exist_ok=True)
                bundle_layout = 'expanded_directory'
            source_manifest = json.loads((PROJECT_ROOT / 'source_manifest.json').read_text(encoding='utf-8'))
            if source_manifest != EXPECTED_SOURCE_MANIFEST:
                raise RuntimeError('Expanded source manifest mismatch')
            for relative, expected in source_manifest['files'].items():
                source = PROJECT_ROOT.joinpath(*PurePosixPath(relative).parts)
                if not source.is_file() or sha256(source) != expected['sha256']:
                    raise RuntimeError(f'Source verification failed: {{relative}}')
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            TEMP_ROOT.mkdir(parents=True, exist_ok=True)
            print('items:', items_path)
            print('matches:', matches_path)
            print('bundle layout:', bundle_layout)
            print('human pairs:', len(pd.read_parquet(matches_path, columns=['target'])))
            print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
            """
        ),
        shared.experiment_run_initialization_cell(),
        markdown("## Frozen configuration"),
        code(
            f"""
            EXPECTED_CONFIG = {config!r}
            TRAIN_CONFIG = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
            if TRAIN_CONFIG != EXPECTED_CONFIG:
                raise RuntimeError('Notebook config and bundled config differ')
            if list(TRAIN_CONFIG['runs']) != ['A_BASELINE', 'B_SHUFFLE_SWAP']:
                raise RuntimeError('Exactly the two planned runs are required')
            print(json.dumps(TRAIN_CONFIG, ensure_ascii=False, indent=2))
            """
        ),
        markdown("## Install and pin the single base model snapshot"),
        code(
            """
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',
                 '--upgrade-strategy', 'only-if-needed', '-r',
                 str(PROJECT_ROOT / 'requirements-minilm-augmentation.txt')],
                check=True,
            )
            from huggingface_hub import HfApi, snapshot_download

            model_revision = HfApi().model_info(TRAIN_CONFIG['model']).sha
            snapshot_download(repo_id=TRAIN_CONFIG['model'], revision=model_revision, local_dir=MODEL_DIR)
            print('model revision:', model_revision)
            """
        ),
        markdown("## One full human-only grouped split and hard-looking groups"),
        code(
            """
            command = [
                sys.executable, '-u', str(PROJECT_ROOT / 'scripts/prepare_minilm_s2_augmentation.py'),
                '--items', str(items_path), '--matches', str(matches_path),
                '--config', str(CONFIG_PATH), '--output-dir', str(PREPARED_DIR),
            ]
            print('$', ' '.join(command), flush=True)
            subprocess.run(command, check=True, cwd=PROJECT_ROOT)
            preparation_report = json.loads((PREPARED_DIR / 'preparation_report.json').read_text(encoding='utf-8'))
            print(json.dumps({'split': preparation_report['split'], 'hard_groups': preparation_report['hard_groups']}, ensure_ascii=False, indent=2))
            """
        ),
        markdown("## A/B training concurrently on two T4 GPUs"),
        code(
            """
            runs = list(TRAIN_CONFIG['runs'])

            def launch(run_name, gpu):
                run_dir = RUNS_DIR / run_name
                run_dir.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable, '-u', str(PROJECT_ROOT / 'scripts/train_minilm_s2_augmentation.py'),
                    '--config', str(CONFIG_PATH), '--prepared-dir', str(PREPARED_DIR),
                    '--model-path', str(MODEL_DIR), '--model-revision', model_revision,
                    '--run-name', run_name, '--output-dir', str(run_dir),
                    '--checkpoint-dir', str(CHECKPOINTS_TEMP / run_name),
                    '--token-cache-dir', str(TOKEN_CACHE_ROOT / run_name),
                ]
                environment = os.environ.copy()
                environment.update({
                    'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONUNBUFFERED': '1',
                    'TOKENIZERS_PARALLELISM': 'true', 'RAYON_NUM_THREADS': '2',
                    'OMP_NUM_THREADS': '2',
                })
                log_path = run_dir / 'console.log'
                handle = log_path.open('w', encoding='utf-8', buffering=1)
                process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, text=True)
                print(f'launched {run_name} on GPU {gpu}: pid={process.pid}')
                return run_name, process, handle, log_path

            experiment_started = time.perf_counter()
            jobs = [launch(run_name, gpu) for gpu, run_name in enumerate(runs)]
            while any(process.poll() is None for _, process, _, _ in jobs):
                time.sleep(30)
                print('run status:', {name: process.poll() for name, process, _, _ in jobs}, flush=True)
            failures = []
            for run_name, process, handle, log_path in jobs:
                handle.close()
                if process.returncode:
                    failures.append((run_name, process.returncode, log_path.read_text(encoding='utf-8', errors='replace')[-10000:]))
            if failures:
                raise RuntimeError('Training failure:\\n' + json.dumps(failures, ensure_ascii=False, indent=2))
            experiment_wall_seconds = time.perf_counter() - experiment_started
            """
        ),
        markdown("## Comparison and winner checkpoint"),
        code(
            """
            command = [
                sys.executable, '-u', str(PROJECT_ROOT / 'scripts/summarize_minilm_s2_augmentation.py'),
                '--runs-dir', str(RUNS_DIR), '--prepared-dir', str(PREPARED_DIR), '--output-dir', str(OUTPUT_DIR),
            ]
            subprocess.run(command, check=True, cwd=PROJECT_ROOT)
            comparison = pd.read_csv(OUTPUT_DIR / 'augmentation_comparison.csv')
            display(comparison)
            best_run = (OUTPUT_DIR / 'best_run.txt').read_text(encoding='utf-8').strip()
            shutil.copytree(CHECKPOINTS_TEMP / best_run, OUTPUT_DIR / 'checkpoint' / best_run, dirs_exist_ok=True)
            reports = {
                run_name: json.loads((RUNS_DIR / run_name / 'training_report.json').read_text(encoding='utf-8'))
                for run_name in runs
            }
            completed_at = datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')
            run_completions = {}
            for run_name, report in reports.items():
                run_completions[run_name] = {
                    'status': 'complete',
                    'run_id': uuid.uuid5(uuid.UUID(hex=EXPERIMENT_RUN_ID), run_name).hex,
                    'started_at_utc': EXPERIMENT_STARTED_AT_UTC,
                    'completed_at_utc': completed_at,
                    'experiment': report['experiment'], 'model': TRAIN_CONFIG['model'],
                    'dataset_ref': EXPECTED_CODE_DATASET_REF,
                    'kaggle_kernel_ref': os.getenv('KAGGLE_KERNEL_RUN_ID') or os.getenv('KAGGLE_KERNEL_INFERENCE_RUN_ID') or '',
                    'code_bundle_sha256': EXPECTED_BUNDLE_SHA256,
                    'training_wall_seconds': report['training_seconds'],
                    'training_report': report,
                }
            (OUTPUT_DIR / 'run_completions.json').write_text(json.dumps(run_completions, ensure_ascii=False, indent=2), encoding='utf-8')
            """
        ),
    ]
    for run_name in config["runs"]:
        cells.append(markdown(f"## Google Sheets: {run_name}"))
        cells.append(code(f"completion = run_completions[{run_name!r}]"))
        cells.extend(shared.google_sheets_tracking_cells(sync_stem=f"google_sheets_sync_{run_name.lower()}"))
    cells.extend(
        [
            markdown("## Final completion marker"),
            code(
                """
                sync_statuses = {}
                for run_name in runs:
                    path = WORKING_ROOT / f'google_sheets_sync_{run_name.lower()}.json'
                    sync_statuses[run_name] = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {'status': 'missing'}
                aggregate = json.loads((OUTPUT_DIR / 'augmentation_report.json').read_text(encoding='utf-8'))
                completion = {
                    'status': 'complete', 'run_id': EXPERIMENT_RUN_ID,
                    'started_at_utc': EXPERIMENT_STARTED_AT_UTC,
                    'completed_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
                    'experiment': TRAIN_CONFIG['experiment'], 'model': TRAIN_CONFIG['model'],
                    'dataset_ref': EXPECTED_CODE_DATASET_REF,
                    'code_bundle_sha256': EXPECTED_BUNDLE_SHA256,
                    'experiment_wall_seconds': experiment_wall_seconds,
                    'best_run': best_run, 'google_sheets': sync_statuses,
                    'augmentation_report': aggregate,
                }
                (WORKING_ROOT / 'notebook_completed.json').write_text(json.dumps(completion, ensure_ascii=False, indent=2), encoding='utf-8')
                (OUTPUT_DIR / 'COMPLETED').write_text('complete\\n', encoding='utf-8')
                print(json.dumps(completion, ensure_ascii=False, indent=2))
                """
            ),
        ]
    )
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update(
        {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "minilm_s2_augmentation": {
                "code_bundle_sha256": expected_hash,
                "model": config["model"],
                "runs": list(config["runs"]),
                "parallel_single_gpu_runs": 2,
                "human_labels_only": True,
                "validation_pair_orders": 1,
            },
        }
    )
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--output", type=Path, default=DEFAULT_NOTEBOOK)
    args = parser.parse_args()
    owner = shared.dotenv_username(args.env_file)
    if not owner:
        raise SystemExit("Set KAGGLE_USERNAME in .env")
    manifest = build_code_dataset(DEFAULT_CODE_DATASET_DIR, owner)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(manifest), args.output)
    print(args.output)


if __name__ == "__main__":
    main()
