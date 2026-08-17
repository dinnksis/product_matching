#!/usr/bin/env python3
"""Build the autonomous 2xT4 S0/S2 experiment on fixed validation splits."""

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
DEFAULT_NOTEBOOK = ROOT / "notebooks/minilm_s0_s2_new_validation_splits_2xt4.ipynb"
DEFAULT_CODE_DATASET_DIR = ROOT / ".kaggle/datasets/product-matching-minilm-s0-s2-new-splits-code"
CODE_DATASET_SLUG = "product-matching-minilm-s0-s2-new-splits-code"
BUNDLE_NAME = "minilm_s0_s2_new_splits_code.zip"
MANIFEST_NAME = "minilm_s0_s2_new_splits_manifest.json"
BUNDLE_FILES = (
    Path("configs/minilm_s0_s2_new_splits.json"),
    Path("requirements-serialization-ablation.txt"),
    Path("src/__init__.py"),
    Path("src/cross_encoder_training.py"),
    Path("src/qwen_reranker.py"),
    Path("src/qwen_training.py"),
    Path("src/serialization_ablation.py"),
    Path("scripts/train_serialization_ablation.py"),
    Path("scripts/prepare_minilm_s0_s2_new_splits.py"),
    Path("scripts/evaluate_minilm_new_splits.py"),
    Path("scripts/summarize_minilm_s0_s2_new_splits.py"),
)


def markdown(value: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(value).strip())


def code(value: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(value).strip())


def build_code_dataset(directory: Path, owner: str) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    payloads = {}
    files = {}
    for relative in BUNDLE_FILES:
        payload = (ROOT / relative).read_bytes()
        payloads[relative.as_posix()] = payload
        files[relative.as_posix()] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
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
            "sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
            "source": source_manifest,
        },
    }
    (directory / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "Product Matching MiniLM S0 S2 New Splits Code",
                "id": f"{owner}/{CODE_DATASET_SLUG}",
                "licenses": [{"name": "other"}],
                "isPrivate": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def build_notebook(manifest: dict[str, object]) -> nbf.NotebookNode:
    bundle = manifest["code_bundle"]
    assert isinstance(bundle, dict)
    config = json.loads((ROOT / "configs/minilm_s0_s2_new_splits.json").read_text(encoding="utf-8"))
    expected_hash = str(bundle["sha256"])
    expected_source = bundle["source"]
    validation_dataset_ref = "alexproger23/product-matching-validation-splits-v1"
    cells = [
        markdown(
            """
            # MiniLM S0 vs S2 on IID, hard, and OOD human splits

            S0 and S2 are trained independently for one epoch on the fixed 306,669-pair
            human train partition, one run per T4. Both checkpoints are evaluated with
            symmetric A/B inference on all three item-disjoint validation splits.
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
            TEMP_ROOT = Path('/kaggle/temp/minilm_s0_s2_new_splits')
            PROJECT_ROOT = TEMP_ROOT / 'product_matching'
            PREPARED_DIR = TEMP_ROOT / 'prepared'
            MODEL_DIR = TEMP_ROOT / 'base_model'
            CHECKPOINTS_DIR = TEMP_ROOT / 'checkpoints'
            TOKEN_CACHE_ROOT = TEMP_ROOT / 'token_cache'
            OUTPUT_DIR = WORKING_ROOT / 'minilm_s0_s2_new_splits'
            TRAINING_DIR = OUTPUT_DIR / 'training'
            EVALUATIONS_DIR = OUTPUT_DIR / 'evaluations'
            CONFIG_PATH = PROJECT_ROOT / 'configs/minilm_s0_s2_new_splits.json'
            EXPECTED_BUNDLE_SHA256 = {expected_hash!r}
            EXPECTED_SOURCE_MANIFEST = {expected_source!r}
            CODE_DATASET_REF = {str(manifest['dataset'])!r}
            VALIDATION_DATASET_REF = {validation_dataset_ref!r}

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
            split_manifest_path = exactly_one('validation_splits_manifest.json')
            train_path = exactly_one('human_train_pairs.parquet')
            iid_path = exactly_one('human_iid_validation_pairs.parquet')
            hard_path = exactly_one('human_hard_validation_pairs.parquet')
            ood_path = exactly_one('human_ood_validation_pairs.parquet')
            bundle_candidates = list(INPUT_ROOT.glob('**/{BUNDLE_NAME}'))
            bundle_candidates.extend(
                path for path in INPUT_ROOT.glob('**/{Path(BUNDLE_NAME).stem}')
                if path.is_dir()
            )
            if len(bundle_candidates) != 1:
                raise RuntimeError(f'Expected one code bundle, found {{bundle_candidates}}')
            bundle_path = bundle_candidates[0]
            PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
            if bundle_path.is_file():
                if sha256(bundle_path) != EXPECTED_BUNDLE_SHA256:
                    raise RuntimeError('Code bundle SHA-256 mismatch')
                with zipfile.ZipFile(bundle_path) as archive:
                    for member in archive.namelist():
                        relative = PurePosixPath(member)
                        if relative.is_absolute() or '..' in relative.parts:
                            raise RuntimeError(f'Unsafe bundle member: {{member}}')
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
                if sha256(source) != expected['sha256']:
                    raise RuntimeError(f'Source hash mismatch: {{relative}}')
            split_manifest = json.loads(split_manifest_path.read_text(encoding='utf-8'))
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            TEMP_ROOT.mkdir(parents=True, exist_ok=True)
            print('items:', items_path)
            print('validation dataset:', VALIDATION_DATASET_REF)
            print('bundle layout:', bundle_layout)
            print('source pairs:', split_manifest['human']['source_pairs'])
            print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
            """
        ),
        shared.experiment_run_initialization_cell(),
        markdown("## Frozen configuration and dependencies"),
        code(
            f"""
            EXPECTED_CONFIG = {config!r}
            TRAIN_CONFIG = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
            if TRAIN_CONFIG != EXPECTED_CONFIG:
                raise RuntimeError('Notebook config and bundled config differ')
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',
                 '--upgrade-strategy', 'only-if-needed', '-r',
                 str(PROJECT_ROOT / 'requirements-serialization-ablation.txt')],
                check=True,
            )
            from huggingface_hub import HfApi, snapshot_download
            model_revision = HfApi().model_info(TRAIN_CONFIG['model']).sha
            snapshot_download(repo_id=TRAIN_CONFIG['model'], revision=model_revision, local_dir=MODEL_DIR)
            print('model revision:', model_revision)
            """
        ),
        markdown("## Prepare leakage-free S0/S2 texts"),
        code(
            """
            prepare_command = [
                sys.executable, '-u', str(PROJECT_ROOT / 'scripts/prepare_minilm_s0_s2_new_splits.py'),
                '--items', str(items_path), '--train', str(train_path),
                '--iid', str(iid_path), '--hard', str(hard_path), '--ood', str(ood_path),
                '--config', str(CONFIG_PATH), '--output-dir', str(PREPARED_DIR),
            ]
            subprocess.run(prepare_command, check=True, cwd=PROJECT_ROOT)
            preparation_report = json.loads((PREPARED_DIR / 'preparation_report.json').read_text(encoding='utf-8'))
            print(json.dumps(preparation_report, ensure_ascii=False, indent=2))
            """
        ),
        markdown("## Train S0 and S2 concurrently on two T4 GPUs"),
        code(
            """
            variants = list(TRAIN_CONFIG['variants'])

            def launch_training(variant, gpu):
                run_dir = TRAINING_DIR / variant
                run_dir.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable, '-u', str(PROJECT_ROOT / 'scripts/train_serialization_ablation.py'),
                    '--config', str(CONFIG_PATH), '--prepared-dir', str(PREPARED_DIR),
                    '--model-path', str(MODEL_DIR), '--model-revision', model_revision,
                    '--variant', variant, '--output-dir', str(run_dir),
                    '--checkpoint-dir', str(CHECKPOINTS_DIR / variant),
                    '--token-cache-dir', str(TOKEN_CACHE_ROOT / 'training' / variant),
                ]
                environment = os.environ.copy()
                environment.update({'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONUNBUFFERED': '1',
                                    'TOKENIZERS_PARALLELISM': 'true', 'RAYON_NUM_THREADS': '2',
                                    'OMP_NUM_THREADS': '2'})
                log_path = run_dir / 'console.log'
                handle = log_path.open('w', encoding='utf-8', buffering=1)
                process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment,
                                           stdout=handle, stderr=subprocess.STDOUT, text=True)
                print(f'launched {variant} on GPU {gpu}: pid={process.pid}')
                return variant, process, handle, log_path

            training_started = time.perf_counter()
            jobs = [launch_training(variant, gpu) for gpu, variant in enumerate(variants)]
            while any(process.poll() is None for _, process, _, _ in jobs):
                time.sleep(30)
                print('training status:', {variant: process.poll() for variant, process, _, _ in jobs}, flush=True)
            failures = []
            for variant, process, handle, log_path in jobs:
                handle.close()
                if process.returncode:
                    failures.append((variant, process.returncode, log_path.read_text(encoding='utf-8', errors='replace')[-12000:]))
            if failures:
                raise RuntimeError('Training failure:\\n' + json.dumps(failures, ensure_ascii=False, indent=2))
            training_wall_seconds = time.perf_counter() - training_started
            """
        ),
        markdown("## Symmetric evaluation on IID, hard, and OOD"),
        code(
            """
            def launch_evaluation(variant, gpu):
                output_dir = EVALUATIONS_DIR / variant
                output_dir.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable, '-u', str(PROJECT_ROOT / 'scripts/evaluate_minilm_new_splits.py'),
                    '--config', str(CONFIG_PATH), '--prepared-dir', str(PREPARED_DIR),
                    '--checkpoint-dir', str(CHECKPOINTS_DIR / variant),
                    '--training-report', str(TRAINING_DIR / variant / 'training_report.json'),
                    '--variant', variant, '--output-dir', str(output_dir),
                    '--token-cache-dir', str(TOKEN_CACHE_ROOT / 'evaluation' / variant),
                ]
                environment = os.environ.copy()
                environment.update({'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONUNBUFFERED': '1',
                                    'TOKENIZERS_PARALLELISM': 'true', 'RAYON_NUM_THREADS': '2',
                                    'OMP_NUM_THREADS': '2'})
                log_path = output_dir / 'console.log'
                handle = log_path.open('w', encoding='utf-8', buffering=1)
                process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment,
                                           stdout=handle, stderr=subprocess.STDOUT, text=True)
                print(f'evaluating {variant} on GPU {gpu}: pid={process.pid}')
                return variant, process, handle, log_path

            evaluation_started = time.perf_counter()
            jobs = [launch_evaluation(variant, gpu) for gpu, variant in enumerate(variants)]
            while any(process.poll() is None for _, process, _, _ in jobs):
                time.sleep(30)
                print('evaluation status:', {variant: process.poll() for variant, process, _, _ in jobs}, flush=True)
            failures = []
            for variant, process, handle, log_path in jobs:
                handle.close()
                if process.returncode:
                    failures.append((variant, process.returncode, log_path.read_text(encoding='utf-8', errors='replace')[-12000:]))
            if failures:
                raise RuntimeError('Evaluation failure:\\n' + json.dumps(failures, ensure_ascii=False, indent=2))
            evaluation_wall_seconds = time.perf_counter() - evaluation_started
            """
        ),
        markdown("## Aggregate metrics and preserve S2 checkpoint"),
        code(
            """
            subprocess.run(
                [sys.executable, '-u', str(PROJECT_ROOT / 'scripts/summarize_minilm_s0_s2_new_splits.py'),
                 '--config', str(CONFIG_PATH), '--evaluations-dir', str(EVALUATIONS_DIR),
                 '--output-dir', str(OUTPUT_DIR)],
                check=True, cwd=PROJECT_ROOT,
            )
            shutil.copytree(CHECKPOINTS_DIR / 'S2_VALUES_ONLY',
                            OUTPUT_DIR / 'checkpoint' / 'S2_VALUES_ONLY', dirs_exist_ok=True)
            comparison = pd.read_csv(OUTPUT_DIR / 's0_s2_split_comparison.csv')
            display(comparison)
            aggregate = json.loads((OUTPUT_DIR / 'aggregate_report.json').read_text(encoding='utf-8'))
            completed_at = datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')
            completions = {}
            for variant in variants:
                for split in TRAIN_CONFIG['validation_splits']:
                    report = aggregate['reports'][variant][split]
                    key = f'{variant}_{split}'
                    completions[key] = {
                        'status': 'complete',
                        'run_id': uuid.uuid5(uuid.UUID(hex=EXPERIMENT_RUN_ID), key).hex,
                        'started_at_utc': EXPERIMENT_STARTED_AT_UTC,
                        'completed_at_utc': completed_at,
                        'experiment': report['experiment'], 'model': TRAIN_CONFIG['model'],
                        'dataset_ref': VALIDATION_DATASET_REF,
                        'kaggle_kernel_ref': os.getenv('KAGGLE_KERNEL_RUN_ID') or os.getenv('KAGGLE_KERNEL_INFERENCE_RUN_ID') or '',
                        'code_bundle_sha256': EXPECTED_BUNDLE_SHA256,
                        'training_wall_seconds': report['training_seconds'],
                        'training_report': report,
                    }
            (OUTPUT_DIR / 'run_completions.json').write_text(
                json.dumps(completions, ensure_ascii=False, indent=2), encoding='utf-8')
            """
        ),
    ]
    for variant in config["variants"]:
        for split in config["validation_splits"]:
            key = f"{variant}_{split}"
            cells.append(markdown(f"## Google Sheets: {variant} / {split}"))
            cells.append(code(f"completion = completions[{key!r}]"))
            cells.extend(shared.google_sheets_tracking_cells(sync_stem=f"google_sheets_sync_{variant.lower()}_{split}"))
    cells.extend(
        [
            markdown("## Completion marker"),
            code(
                """
                completion = {
                    'status': 'complete', 'run_id': EXPERIMENT_RUN_ID,
                    'started_at_utc': EXPERIMENT_STARTED_AT_UTC,
                    'completed_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
                    'experiment': TRAIN_CONFIG['experiment'], 'model': TRAIN_CONFIG['model'],
                    'dataset_ref': VALIDATION_DATASET_REF,
                    'code_bundle_sha256': EXPECTED_BUNDLE_SHA256,
                    'training_wall_seconds': training_wall_seconds,
                    'evaluation_wall_seconds': evaluation_wall_seconds,
                    'catboost_gate': aggregate['catboost_gate'],
                }
                (WORKING_ROOT / 'notebook_completed.json').write_text(
                    json.dumps(completion, ensure_ascii=False, indent=2), encoding='utf-8')
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
            "minilm_s0_s2_new_splits": {
                "code_bundle_sha256": expected_hash,
                "variants": config["variants"],
                "validation_splits": list(config["validation_splits"]),
                "parallel_single_gpu_runs": 2,
                "symmetric_validation": True,
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
