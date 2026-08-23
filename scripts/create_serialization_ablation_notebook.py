#!/usr/bin/env python3
"""Build the autonomous 2xT4 MiniLM serialization ablation notebook."""

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
DEFAULT_NOTEBOOK = ROOT / "notebooks" / "minilm_serialization_ablation_2xt4.ipynb"
DEFAULT_CODE_DATASET_DIR = ROOT / ".kaggle" / "datasets" / "product-matching-serialization-ablation-code"
CODE_DATASET_SLUG = "product-matching-serialization-ablation-code"
BUNDLE_NAME = "serialization_ablation_code.zip"
MANIFEST_NAME = "serialization_ablation_manifest.json"
BUNDLE_FILES = (
    Path("configs/serialization_ablation_minilm.json"),
    Path("requirements-serialization-ablation.txt"),
    Path("src/__init__.py"),
    Path("src/cross_encoder_training.py"),
    Path("src/qwen_reranker.py"),
    Path("src/qwen_training.py"),
    Path("src/serialization_ablation.py"),
    Path("scripts/prepare_serialization_ablation.py"),
    Path("scripts/train_serialization_ablation.py"),
    Path("scripts/summarize_serialization_ablation.py"),
)


def markdown(value: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(value).strip())


def code(value: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(value).strip())


def build_code_dataset(directory: Path, owner: str) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    source_files: dict[str, dict[str, object]] = {}
    payloads: dict[str, bytes] = {}
    for relative in BUNDLE_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"Required ablation source is missing: {source}")
        payload = source.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        source_files[relative.as_posix()] = {"bytes": len(payload), "sha256": digest}
        payloads[relative.as_posix()] = payload
    source_manifest = {"schema_version": 1, "files": source_files}
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
    bundle_hash = shared.sha256(bundle_path)
    manifest = {
        "schema_version": 1,
        "dataset": f"{owner}/{CODE_DATASET_SLUG}",
        "code_bundle": {
            "filename": BUNDLE_NAME,
            "bytes": bundle_path.stat().st_size,
            "sha256": bundle_hash,
            "source": source_manifest,
        },
    }
    (directory / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "title": "MiniLM Serialization Ablation Code",
        "id": f"{owner}/{CODE_DATASET_SLUG}",
        "licenses": [{"name": "unknown"}],
        "description": "Private exact source bundle for the human-only MiniLM serialization ablation.",
    }
    (directory / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_notebook(manifest: dict[str, object]) -> nbf.NotebookNode:
    bundle = manifest["code_bundle"]
    assert isinstance(bundle, dict)
    expected_hash = str(bundle["sha256"])
    expected_source_manifest = bundle["source"]
    config = json.loads((ROOT / "configs/serialization_ablation_minilm.json").read_text(encoding="utf-8"))
    config_literal = repr(config)
    cells: list[nbf.NotebookNode] = [
        markdown(
            """
            # MiniLM product serialization ablation (human labels only)

            Four controlled representations are screened with exactly the same
            grouped split, 120k train subset, normalization, optimizer, batch,
            max length, seed and one-epoch budget. Two independent single-GPU
            runs execute concurrently, one on each NVIDIA T4.

            - S0_TITLE: title only;
            - S1_KEY_VALUE: title plus all `key: value` attributes;
            - S2_VALUES_ONLY: title plus attribute values;
            - S3_HYBRID: frequent train-derived keys keep `key: value`, rare keys keep only values.

            No LLM-labelled pairs are read or attached.
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
            TEMP_ROOT = Path('/kaggle/temp/minilm_serialization_ablation')
            PROJECT_ROOT = WORKING_ROOT / 'product_matching'
            OUTPUT_DIR = WORKING_ROOT / 'serialization_ablation'
            PREPARED_DIR = TEMP_ROOT / 'prepared'
            MODEL_DIR = TEMP_ROOT / 'base_model'
            TEMP_CHECKPOINTS = TEMP_ROOT / 'checkpoints'
            TOKEN_CACHE_ROOT = TEMP_ROOT / 'token_cache'
            RUNS_DIR = OUTPUT_DIR / 'runs'
            CONFIG_PATH = PROJECT_ROOT / 'configs/serialization_ablation_minilm.json'
            EXPECTED_BUNDLE_SHA256 = {expected_hash!r}
            EXPECTED_SOURCE_MANIFEST = {expected_source_manifest!r}
            EXPECTED_CODE_DATASET_REF = {str(manifest['dataset'])!r}
            EXPECTED_CODE_DATASET_SLUG = EXPECTED_CODE_DATASET_REF.rsplit('/', 1)[-1]

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
            bundle_candidates.extend(
                path for path in INPUT_ROOT.glob('**/{Path(BUNDLE_NAME).stem}') if path.is_dir()
            )
            if len(bundle_candidates) != 1:
                raise RuntimeError(f'Expected one serialization code bundle, found {{bundle_candidates}}')
            bundle_path = bundle_candidates[0]
            PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
            if bundle_path.is_file():
                if sha256(bundle_path) != EXPECTED_BUNDLE_SHA256:
                    raise RuntimeError('Attached serialization code bundle hash does not match notebook')
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
                raise RuntimeError('Expanded source manifest does not match notebook')
            for relative, expected in source_manifest['files'].items():
                source = PROJECT_ROOT.joinpath(*PurePosixPath(relative).parts)
                if not source.is_file() or sha256(source) != expected['sha256']:
                    raise RuntimeError(f'Source verification failed: {{relative}}')
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            TEMP_ROOT.mkdir(parents=True, exist_ok=True)
            print('items:', items_path)
            print('matches:', matches_path)
            print('code bundle layout:', bundle_layout)
            print('human matches rows:', len(pd.read_parquet(matches_path, columns=['target'])))
            print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
            """
        ),
        shared.experiment_run_initialization_cell(),
        markdown("## Frozen experiment configuration"),
        code(
            f"""
            EXPECTED_CONFIG = {config_literal}
            observed_config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
            if observed_config != EXPECTED_CONFIG:
                raise RuntimeError('Embedded notebook config and bundled config differ')
            TRAIN_CONFIG = observed_config
            print(json.dumps(TRAIN_CONFIG, ensure_ascii=False, indent=2))
            """
        ),
        markdown("## Install environment and download exactly one base model snapshot"),
        code(
            """
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',
                 '--upgrade-strategy', 'only-if-needed', '-r',
                 str(PROJECT_ROOT / 'requirements-serialization-ablation.txt')],
                check=True,
            )
            from huggingface_hub import HfApi, snapshot_download

            model_revision = HfApi().model_info(TRAIN_CONFIG['model']).sha
            snapshot_path = Path(snapshot_download(
                repo_id=TRAIN_CONFIG['model'],
                revision=model_revision,
                local_dir=MODEL_DIR,
            ))
            print('model snapshot:', snapshot_path)
            print('model revision:', model_revision)
            """
        ),
        markdown("## Build the single grouped split and train-derived HYBRID threshold"),
        code(
            """
            prepare_command = [
                sys.executable, '-u', str(PROJECT_ROOT / 'scripts/prepare_serialization_ablation.py'),
                '--items', str(items_path), '--matches', str(matches_path),
                '--config', str(CONFIG_PATH), '--output-dir', str(PREPARED_DIR),
            ]
            print('$', ' '.join(prepare_command), flush=True)
            subprocess.run(prepare_command, check=True, cwd=PROJECT_ROOT)
            preparation_report = json.loads((PREPARED_DIR / 'preparation_report.json').read_text(encoding='utf-8'))
            threshold = preparation_report['frequency_threshold']
            print(json.dumps({'split': preparation_report['split'], 'frequency_threshold': threshold}, ensure_ascii=False, indent=2))
            frequency = pd.read_csv(PREPARED_DIR / 'attribute_name_frequency.csv')
            display(frequency.head(50))
            """
        ),
        markdown("## Four one-epoch runs in two parallel waves"),
        code(
            """
            variants = list(TRAIN_CONFIG['variants'])
            if len(variants) != 4 or int(TRAIN_CONFIG['parallel_runs']) != 2:
                raise RuntimeError('This notebook expects four variants and two parallel GPU slots')

            def launch_variant(variant, gpu):
                run_dir = RUNS_DIR / variant
                checkpoint_dir = TEMP_CHECKPOINTS / variant
                cache_dir = TOKEN_CACHE_ROOT / variant
                run_dir.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable, '-u', str(PROJECT_ROOT / 'scripts/train_serialization_ablation.py'),
                    '--config', str(CONFIG_PATH), '--prepared-dir', str(PREPARED_DIR),
                    '--model-path', str(MODEL_DIR), '--model-revision', model_revision,
                    '--variant', variant,
                    '--output-dir', str(run_dir), '--checkpoint-dir', str(checkpoint_dir),
                    '--token-cache-dir', str(cache_dir),
                ]
                environment = os.environ.copy()
                environment.update({
                    'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONUNBUFFERED': '1',
                    'TOKENIZERS_PARALLELISM': 'true', 'RAYON_NUM_THREADS': '2',
                    'OMP_NUM_THREADS': '2',
                })
                log_path = run_dir / 'console.log'
                log_handle = log_path.open('w', encoding='utf-8', buffering=1)
                process = subprocess.Popen(
                    command, cwd=PROJECT_ROOT, env=environment,
                    stdout=log_handle, stderr=subprocess.STDOUT, text=True,
                )
                print(f'launched {{variant}} on GPU {{gpu}} pid={{process.pid}}')
                return variant, process, log_handle, log_path

            ablation_started = time.perf_counter()
            for wave_start in range(0, len(variants), 2):
                wave = [launch_variant(variant, gpu) for gpu, variant in enumerate(variants[wave_start:wave_start + 2])]
                while any(process.poll() is None for _, process, _, _ in wave):
                    time.sleep(30)
                    status = {variant: process.poll() for variant, process, _, _ in wave}
                    print('wave status:', status, flush=True)
                failures = []
                for variant, process, handle, log_path in wave:
                    handle.close()
                    if process.returncode:
                        failures.append((variant, process.returncode, log_path.read_text(encoding='utf-8', errors='replace')[-8000:]))
                if failures:
                    raise RuntimeError('Serialization run failure:\\n' + json.dumps(failures, ensure_ascii=False, indent=2))
                print('completed wave:', [variant for variant, *_ in wave])
            ablation_wall_seconds = time.perf_counter() - ablation_started
            """
        ),
        markdown("## Compare variants and retain only baseline plus winner checkpoints"),
        code(
            """
            summary_command = [
                sys.executable, '-u', str(PROJECT_ROOT / 'scripts/summarize_serialization_ablation.py'),
                '--runs-dir', str(RUNS_DIR), '--prepared-dir', str(PREPARED_DIR),
                '--output-dir', str(OUTPUT_DIR),
            ]
            subprocess.run(summary_command, check=True, cwd=PROJECT_ROOT)
            comparison = pd.read_csv(OUTPUT_DIR / 'serialization_comparison.csv')
            display(comparison)
            best_variant = (OUTPUT_DIR / 'best_variant.txt').read_text(encoding='utf-8').strip()
            retained = sorted({'S0_TITLE', best_variant})
            checkpoint_output = OUTPUT_DIR / 'checkpoints'
            for variant in retained:
                shutil.copytree(TEMP_CHECKPOINTS / variant, checkpoint_output / variant, dirs_exist_ok=True)
            print('best serialization:', best_variant)
            print('retained checkpoints:', retained)

            reports = {
                variant: json.loads((RUNS_DIR / variant / 'training_report.json').read_text(encoding='utf-8'))
                for variant in variants
            }
            completed_at = datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')
            variant_completions = {}
            for variant, report in reports.items():
                variant_run_id = uuid.uuid5(uuid.UUID(hex=EXPERIMENT_RUN_ID), variant).hex
                variant_completions[variant] = {
                    'status': 'complete', 'run_id': variant_run_id,
                    'started_at_utc': EXPERIMENT_STARTED_AT_UTC,
                    'completed_at_utc': completed_at,
                    'experiment': f"{{TRAIN_CONFIG['experiment']}}_{{variant.lower()}}",
                    'model': TRAIN_CONFIG['model'],
                    'dataset_ref': EXPECTED_CODE_DATASET_REF,
                    'kaggle_kernel_ref': os.getenv('KAGGLE_KERNEL_RUN_ID') or os.getenv('KAGGLE_KERNEL_INFERENCE_RUN_ID') or '',
                    'code_bundle_sha256': EXPECTED_BUNDLE_SHA256,
                    'training_wall_seconds': report['training_seconds'],
                    'training_report': report,
                }
            (OUTPUT_DIR / 'variant_completions.json').write_text(
                json.dumps(variant_completions, ensure_ascii=False, indent=2), encoding='utf-8'
            )
            """
        ),
    ]
    for variant in config["variants"]:
        cells.append(markdown(f"## Google Sheets: {variant}"))
        cells.append(code(f"completion = variant_completions[{variant!r}]"))
        cells.extend(shared.google_sheets_tracking_cells(sync_stem=f"google_sheets_sync_{str(variant).lower()}"))
    cells.extend(
        [
            markdown("## Final completion marker"),
            code(
                """
                sync_statuses = {}
                for variant in variants:
                    path = WORKING_ROOT / f'google_sheets_sync_{variant.lower()}.json'
                    sync_statuses[variant] = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {'status': 'missing'}
                aggregate_report = json.loads((OUTPUT_DIR / 'ablation_report.json').read_text(encoding='utf-8'))
                completion = {
                    'status': 'complete', 'run_id': EXPERIMENT_RUN_ID,
                    'started_at_utc': EXPERIMENT_STARTED_AT_UTC,
                    'completed_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
                    'experiment': TRAIN_CONFIG['experiment'], 'model': TRAIN_CONFIG['model'],
                    'dataset_ref': EXPECTED_CODE_DATASET_REF,
                    'code_bundle_sha256': EXPECTED_BUNDLE_SHA256,
                    'ablation_wall_seconds': ablation_wall_seconds,
                    'best_serialization': best_variant,
                    'retained_checkpoints': retained,
                    'google_sheets': sync_statuses,
                    'ablation_report': aggregate_report,
                }
                (WORKING_ROOT / 'notebook_completed.json').write_text(
                    json.dumps(completion, ensure_ascii=False, indent=2), encoding='utf-8'
                )
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
            "serialization_ablation": {
                "code_bundle_sha256": expected_hash,
                "model": config["model"],
                "variants": config["variants"],
                "parallel_single_gpu_runs": 2,
                "human_labels_only": True,
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
