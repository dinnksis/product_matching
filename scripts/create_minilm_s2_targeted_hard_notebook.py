#!/usr/bin/env python3
"""Build the autonomous 2xT4 S2 targeted-hard training notebook and code Dataset."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath
from textwrap import dedent

import nbformat as nbf
import pandas as pd

import create_qwen_training_notebook as shared


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = ROOT / "notebooks/minilm_s2_targeted_hard_training_2xt4.ipynb"
DEFAULT_CODE_DATASET_DIR = ROOT / ".kaggle/datasets/product-matching-minilm-s2-targeted-hard-code"
CODE_DATASET_SLUG = "product-matching-minilm-s2-targeted-hard-code"
BUNDLE_NAME = "minilm_s2_targeted_hard_code.zip"
MANIFEST_NAME = "minilm_s2_targeted_hard_manifest.json"
BUNDLE_FILES = (
    Path("configs/minilm_s2_targeted_hard_training.json"),
    Path("requirements-serialization-ablation.txt"),
    Path("requirements-hard-mining.txt"),
    Path("src/__init__.py"),
    Path("src/cheap_ensemble.py"),
    Path("src/cross_encoder_training.py"),
    Path("src/qwen_reranker.py"),
    Path("src/qwen_training.py"),
    Path("src/serialization_ablation.py"),
    Path("scripts/train_serialization_ablation.py"),
    Path("scripts/prepare_minilm_s0_s2_new_splits.py"),
    Path("scripts/prepare_minilm_s2_hard_mining.py"),
    Path("scripts/evaluate_minilm_new_splits.py"),
    Path("scripts/mine_minilm_s2_oof_hard_examples.py"),
    Path("scripts/summarize_minilm_s2_hard_training.py"),
)


def markdown(value: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(value).strip())


def code(value: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(value).strip())


def hard_audit_assignments() -> pd.DataFrame:
    directory = ROOT / "reports/minilm_s2_hard_clean_audit"
    frames = []
    for subset in ("hard_clean", "hard_suspicious", "hard_conflicting"):
        frame = pd.read_csv(
            directory / f"{subset}.csv",
            usecols=["id1", "id2", "target", "hard_subset"],
        )
        if not frame["hard_subset"].eq(subset).all():
            raise ValueError(f"{subset}: hard_subset column differs from filename")
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if len(result) != 5814:
        raise ValueError(f"expected 5814 hard audit rows, got {len(result)}")
    if result.duplicated(["id1", "id2"]).any():
        raise ValueError("hard audit assignments contain duplicate ordered pairs")
    return result


def hard_clean_slice_flags() -> pd.DataFrame:
    return pd.read_csv(
        ROOT / "reports/minilm_s2_hard_clean_audit/hard_clean.csv",
        usecols=[
            "id1",
            "id2",
            "target",
            "numeric_conflict",
            "code_conflict",
            "model_code_conflict",
            "sku_vs_human_title",
        ],
    )


def build_code_dataset(directory: Path, owner: str) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, bytes] = {}
    files: dict[str, dict[str, object]] = {}
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
    assignments_path = directory / "hard_audit_assignments.csv"
    hard_audit_assignments().to_csv(assignments_path, index=False)
    slice_flags_path = directory / "hard_clean_slice_flags.csv"
    hard_clean_slice_flags().to_csv(slice_flags_path, index=False)
    manifest = {
        "schema_version": 1,
        "dataset": f"{owner}/{CODE_DATASET_SLUG}",
        "code_bundle": {
            "filename": BUNDLE_NAME,
            "bytes": bundle_path.stat().st_size,
            "sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
            "source": source_manifest,
        },
        "hard_audit_assignments": {
            "filename": assignments_path.name,
            "rows": 5814,
            "bytes": assignments_path.stat().st_size,
            "sha256": hashlib.sha256(assignments_path.read_bytes()).hexdigest(),
        },
        "hard_clean_slice_flags": {
            "filename": slice_flags_path.name,
            "rows": 4929,
            "bytes": slice_flags_path.stat().st_size,
            "sha256": hashlib.sha256(slice_flags_path.read_bytes()).hexdigest(),
        },
    }
    (directory / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "Product Matching MiniLM S2 Targeted Hard Code",
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
    assignments = manifest["hard_audit_assignments"]
    slice_flags = manifest["hard_clean_slice_flags"]
    assert isinstance(bundle, dict) and isinstance(assignments, dict) and isinstance(slice_flags, dict)
    config = json.loads(
        (ROOT / "configs/minilm_s2_targeted_hard_training.json").read_text(encoding="utf-8")
    )
    expected_hash = str(bundle["sha256"])
    expected_source = bundle["source"]
    expected_assignments_hash = str(assignments["sha256"])
    expected_slice_flags_hash = str(slice_flags["sha256"])
    cells = [
        markdown(
            """
            # MiniLM S2 targeted training on OOF-hard human examples

            One causal change only: the p85 hardest audit-eligible human train
            positives and negatives, mined from 3-fold component/family-disjoint OOF
            S2 scores, are deterministically duplicated once. The model,
            serialization, optimizer, LR, max length, normalization, and one-epoch
            schedule remain fixed. No LLM-labelled data are used.
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
            from IPython.display import display

            INPUT_ROOT = Path('/kaggle/input')
            WORKING_ROOT = Path('/kaggle/working')
            TEMP_ROOT = Path('/kaggle/temp/minilm_s2_targeted_hard')
            PROJECT_ROOT = TEMP_ROOT / 'product_matching'
            PREPARED_DIR = TEMP_ROOT / 'prepared'
            MINING_PREP_DIR = TEMP_ROOT / 'mining_prepared'
            MODEL_DIR = TEMP_ROOT / 'base_model'
            TOKEN_CACHE_ROOT = TEMP_ROOT / 'token_cache'
            OOF_RUNS_DIR = TEMP_ROOT / 'oof_runs'
            VIEWS_DIR = TEMP_ROOT / 'prepared_views'
            BASELINE_TRAINING_DIR = TEMP_ROOT / 'baseline_training'
            BASELINE_CHECKPOINT_DIR = TEMP_ROOT / 'baseline_checkpoint'
            BASELINE_EVALUATIONS_DIR = WORKING_ROOT / 'minilm_s2_targeted_hard/evaluations/baseline_s2'
            HARD_TRAINING_DIR = WORKING_ROOT / 'minilm_s2_targeted_hard/hard_training'
            HARD_CHECKPOINT_DIR = WORKING_ROOT / 'minilm_s2_targeted_hard/checkpoint'
            HARD_EVALUATIONS_DIR = WORKING_ROOT / 'minilm_s2_targeted_hard/evaluations/targeted_hard_s2'
            MINING_OUTPUT_DIR = WORKING_ROOT / 'minilm_s2_targeted_hard/mining'
            OUTPUT_DIR = WORKING_ROOT / 'minilm_s2_targeted_hard'
            LOGS_DIR = OUTPUT_DIR / 'logs'
            CONFIG_PATH = PROJECT_ROOT / 'configs/minilm_s2_targeted_hard_training.json'
            EXPECTED_BUNDLE_SHA256 = {expected_hash!r}
            EXPECTED_SOURCE_MANIFEST = {expected_source!r}
            EXPECTED_ASSIGNMENTS_SHA256 = {expected_assignments_hash!r}
            EXPECTED_SLICE_FLAGS_SHA256 = {expected_slice_flags_hash!r}
            RAW_DATASET_REF = 'dinakepecheva/e-cup-human-data'
            VALIDATION_DATASET_REF = 'alexproger23/product-matching-validation-splits-v1'
            CODE_DATASET_REF = {str(manifest['dataset'])!r}

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
            train_path = exactly_one('human_train_pairs.parquet')
            iid_path = exactly_one('human_iid_validation_pairs.parquet')
            hard_path = exactly_one('human_hard_validation_pairs.parquet')
            ood_path = exactly_one('human_ood_validation_pairs.parquet')
            assignments_path = exactly_one('hard_audit_assignments.csv')
            if sha256(assignments_path) != EXPECTED_ASSIGNMENTS_SHA256:
                raise RuntimeError('Hard audit assignments SHA-256 mismatch')
            hard_clean_slice_flags_path = exactly_one('hard_clean_slice_flags.csv')
            if sha256(hard_clean_slice_flags_path) != EXPECTED_SLICE_FLAGS_SHA256:
                raise RuntimeError('Hard-clean slice flags SHA-256 mismatch')
            bundle_candidates = list(INPUT_ROOT.glob('**/{BUNDLE_NAME}'))
            bundle_candidates.extend(
                path for path in INPUT_ROOT.glob('**/{Path(BUNDLE_NAME).stem}') if path.is_dir()
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
            else:
                shutil.copytree(bundle_path, PROJECT_ROOT, dirs_exist_ok=True)
            source_manifest = json.loads((PROJECT_ROOT / 'source_manifest.json').read_text(encoding='utf-8'))
            if source_manifest != EXPECTED_SOURCE_MANIFEST:
                raise RuntimeError('Expanded source manifest mismatch')
            for relative, expected in source_manifest['files'].items():
                source = PROJECT_ROOT.joinpath(*PurePosixPath(relative).parts)
                if sha256(source) != expected['sha256']:
                    raise RuntimeError(f'Source hash mismatch: {{relative}}')
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            TEMP_ROOT.mkdir(parents=True, exist_ok=True)
            print('items:', items_path)
            print('train:', train_path)
            print('code dataset:', CODE_DATASET_REF)
            print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
            """
        ),
        shared.experiment_run_initialization_cell(),
        markdown("## Frozen configuration and base model"),
        code(
            f"""
            EXPECTED_CONFIG = {config!r}
            TRAIN_CONFIG = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
            if TRAIN_CONFIG != EXPECTED_CONFIG:
                raise RuntimeError('Notebook config and bundled config differ')
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',
                 '--upgrade-strategy', 'only-if-needed', '-r',
                 str(PROJECT_ROOT / 'requirements-hard-mining.txt')],
                check=True,
            )
            from huggingface_hub import HfApi, snapshot_download
            model_revision = HfApi().model_info(TRAIN_CONFIG['model']).sha
            snapshot_download(
                repo_id=TRAIN_CONFIG['model'], revision=model_revision, local_dir=MODEL_DIR
            )
            print('model revision:', model_revision)
            """
        ),
        markdown("## Prepare fixed S2 texts, clean-hard, train audit, and 3 OOF folds"),
        code(
            """
            preparation_started = time.perf_counter()
            subprocess.run(
                [sys.executable, '-u', str(PROJECT_ROOT / 'scripts/prepare_minilm_s0_s2_new_splits.py'),
                 '--items', str(items_path), '--train', str(train_path), '--iid', str(iid_path),
                 '--hard', str(hard_path), '--ood', str(ood_path), '--config', str(CONFIG_PATH),
                 '--output-dir', str(PREPARED_DIR)],
                check=True, cwd=PROJECT_ROOT,
            )
            subprocess.run(
                [sys.executable, '-u', str(PROJECT_ROOT / 'scripts/prepare_minilm_s2_hard_mining.py'),
                 '--config', str(CONFIG_PATH), '--items', str(items_path),
                 '--train-pairs', str(train_path), '--hard-audit-assignments', str(assignments_path),
                 '--output-dir', str(MINING_PREP_DIR)],
                check=True, cwd=PROJECT_ROOT,
            )
            shutil.copy2(MINING_PREP_DIR / 'hard_clean_pairs.parquet', PREPARED_DIR / 'hard_clean_pairs.parquet')
            preparation_seconds = time.perf_counter() - preparation_started
            prep_report = json.loads((MINING_PREP_DIR / 'preparation_report.json').read_text(encoding='utf-8'))
            display(pd.DataFrame(prep_report['oof_folds']))
            print('preparation seconds:', preparation_seconds)
            """
        ),
        markdown("## Helpers for isolated single-GPU jobs"),
        code(
            """
            def make_view(name, train_pairs, validation_pairs):
                directory = VIEWS_DIR / name
                directory.mkdir(parents=True, exist_ok=True)
                links = {
                    'items.parquet': PREPARED_DIR / 'items.parquet',
                    'train_pairs.parquet': Path(train_pairs),
                    'validation_pairs.parquet': Path(validation_pairs),
                }
                for filename, source in links.items():
                    target = directory / filename
                    target.unlink(missing_ok=True)
                    target.symlink_to(source.resolve())
                return directory

            def launch_training(name, gpu, prepared_view, output_dir, checkpoint_dir):
                log_path = LOGS_DIR / f'{name}.log'
                handle = log_path.open('w', encoding='utf-8', buffering=1)
                command = [
                    sys.executable, '-u', str(PROJECT_ROOT / 'scripts/train_serialization_ablation.py'),
                    '--config', str(CONFIG_PATH), '--prepared-dir', str(prepared_view),
                    '--model-path', str(MODEL_DIR), '--model-revision', model_revision,
                    '--variant', TRAIN_CONFIG['variant'], '--output-dir', str(output_dir),
                    '--checkpoint-dir', str(checkpoint_dir),
                    '--token-cache-dir', str(TOKEN_CACHE_ROOT / name),
                ]
                environment = os.environ.copy()
                environment['CUDA_VISIBLE_DEVICES'] = str(gpu)
                process = subprocess.Popen(
                    command, cwd=PROJECT_ROOT, env=environment,
                    stdout=handle, stderr=subprocess.STDOUT, text=True,
                )
                print(f'launched {name} on GPU {gpu}: pid={process.pid}', flush=True)
                return name, process, handle, log_path

            def launch_evaluation(name, gpu, checkpoint_dir, training_report, output_dir):
                log_path = LOGS_DIR / f'{name}.log'
                handle = log_path.open('w', encoding='utf-8', buffering=1)
                command = [
                    sys.executable, '-u', str(PROJECT_ROOT / 'scripts/evaluate_minilm_new_splits.py'),
                    '--config', str(CONFIG_PATH), '--prepared-dir', str(PREPARED_DIR),
                    '--checkpoint-dir', str(checkpoint_dir), '--training-report', str(training_report),
                    '--variant', TRAIN_CONFIG['variant'], '--output-dir', str(output_dir),
                    '--token-cache-dir', str(TOKEN_CACHE_ROOT / name),
                ]
                environment = os.environ.copy()
                environment['CUDA_VISIBLE_DEVICES'] = str(gpu)
                process = subprocess.Popen(
                    command, cwd=PROJECT_ROOT, env=environment,
                    stdout=handle, stderr=subprocess.STDOUT, text=True,
                )
                print(f'launched {name} on GPU {gpu}: pid={process.pid}', flush=True)
                return name, process, handle, log_path

            def wait_jobs(jobs, poll_seconds=60):
                while any(process.poll() is None for _, process, _, _ in jobs):
                    time.sleep(poll_seconds)
                    print('job status:', {name: process.poll() for name, process, _, _ in jobs}, flush=True)
                failures = []
                for name, process, handle, log_path in jobs:
                    handle.close()
                    if process.returncode:
                        failures.append({
                            'name': name, 'returncode': process.returncode,
                            'tail': log_path.read_text(encoding='utf-8', errors='replace')[-16000:],
                        })
                if failures:
                    raise RuntimeError('GPU job failure:\\n' + json.dumps(failures, ensure_ascii=False, indent=2))
            """
        ),
        markdown("## OOF wave 1: folds 0 and 1"),
        code(
            """
            experiment_started = time.perf_counter()
            fold_jobs = []
            for fold, gpu in ((0, 0), (1, 1)):
                view = make_view(
                    f'oof_fold_{fold}',
                    MINING_PREP_DIR / f'oof_fold_{fold}_train_pairs.parquet',
                    MINING_PREP_DIR / f'oof_fold_{fold}_validation_pairs.parquet',
                )
                fold_jobs.append(
                    launch_training(
                        f'oof_fold_{fold}', gpu, view,
                        OOF_RUNS_DIR / f'fold_{fold}', TEMP_ROOT / f'oof_checkpoint_{fold}'
                    )
                )
            wait_jobs(fold_jobs)
            for fold in (0, 1):
                shutil.rmtree(TEMP_ROOT / f'oof_checkpoint_{fold}', ignore_errors=True)
            """
        ),
        markdown("## OOF wave 2: fold 2 and causal baseline A in parallel"),
        code(
            """
            fold2_view = make_view(
                'oof_fold_2', MINING_PREP_DIR / 'oof_fold_2_train_pairs.parquet',
                MINING_PREP_DIR / 'oof_fold_2_validation_pairs.parquet'
            )
            baseline_view = make_view(
                'baseline', PREPARED_DIR / 'train_pairs.parquet', PREPARED_DIR / 'iid_pairs.parquet'
            )
            wave2 = [
                launch_training(
                    'oof_fold_2', 0, fold2_view, OOF_RUNS_DIR / 'fold_2',
                    TEMP_ROOT / 'oof_checkpoint_2'
                ),
                launch_training(
                    'baseline_s2', 1, baseline_view, BASELINE_TRAINING_DIR,
                    BASELINE_CHECKPOINT_DIR
                ),
            ]
            wait_jobs(wave2)
            shutil.rmtree(TEMP_ROOT / 'oof_checkpoint_2', ignore_errors=True)
            """
        ),
        markdown("## OOF hardness mining and deterministic x2 oversampling"),
        code(
            """
            subprocess.run(
                [sys.executable, '-u', str(PROJECT_ROOT / 'scripts/mine_minilm_s2_oof_hard_examples.py'),
                 '--config', str(CONFIG_PATH),
                 '--train-audit', str(MINING_PREP_DIR / 'train_label_audit_and_folds.parquet'),
                 '--oof-runs-dir', str(OOF_RUNS_DIR), '--output-dir', str(MINING_OUTPUT_DIR)],
                check=True, cwd=PROJECT_ROOT,
            )
            mining_report = json.loads((MINING_OUTPUT_DIR / 'hard_mining_report.json').read_text(encoding='utf-8'))
            display(pd.DataFrame(mining_report['counts']))
            display(pd.read_csv(MINING_OUTPUT_DIR / 'hardness_distribution_quantiles.csv'))
            """
        ),
        markdown("## Train B and evaluate A in parallel"),
        code(
            """
            hard_view = make_view(
                'targeted_hard', MINING_OUTPUT_DIR / 'train_pairs_hard_x2.parquet',
                PREPARED_DIR / 'iid_pairs.parquet'
            )
            wave3 = [
                launch_training(
                    'targeted_hard_s2', 0, hard_view, HARD_TRAINING_DIR, HARD_CHECKPOINT_DIR
                ),
                launch_evaluation(
                    'baseline_evaluation', 1, BASELINE_CHECKPOINT_DIR,
                    BASELINE_TRAINING_DIR / 'training_report.json', BASELINE_EVALUATIONS_DIR
                ),
            ]
            wait_jobs(wave3)
            """
        ),
        markdown("## Evaluate B and summarize the causal comparison"),
        code(
            """
            hard_eval = launch_evaluation(
                'targeted_hard_evaluation', 0, HARD_CHECKPOINT_DIR,
                HARD_TRAINING_DIR / 'training_report.json', HARD_EVALUATIONS_DIR
            )
            wait_jobs([hard_eval])
            subprocess.run(
                [sys.executable, '-u', str(PROJECT_ROOT / 'scripts/summarize_minilm_s2_hard_training.py'),
                 '--config', str(CONFIG_PATH),
                 '--baseline-evaluations', str(BASELINE_EVALUATIONS_DIR),
                 '--hard-evaluations', str(HARD_EVALUATIONS_DIR),
                 '--hard-clean-audit', str(hard_clean_slice_flags_path),
                 '--mining-report', str(MINING_OUTPUT_DIR / 'hard_mining_report.json'),
                 '--output-dir', str(OUTPUT_DIR)],
                check=True, cwd=PROJECT_ROOT,
            )
            """
        ),
    ]
    cells.extend(
        [
            markdown("## Results"),
            code(
                """
                aggregate = json.loads((OUTPUT_DIR / 'training_report.json').read_text(encoding='utf-8'))
                display(pd.read_csv(OUTPUT_DIR / 'main_metrics.csv'))
                display(pd.read_csv(OUTPUT_DIR / 'metric_deltas.csv'))
                display(pd.read_csv(OUTPUT_DIR / 'hard_clean_slice_comparison.csv'))
                experiment_wall_seconds = time.perf_counter() - experiment_started
                completed_at = datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')
                sheets_report = dict(aggregate['reports']['targeted_hard_s2']['hard_clean'])
                sheets_report['full_experiment_report'] = aggregate
                completion = {
                    'status': 'complete', 'run_id': EXPERIMENT_RUN_ID,
                    'started_at_utc': EXPERIMENT_STARTED_AT_UTC,
                    'completed_at_utc': completed_at,
                    'experiment': TRAIN_CONFIG['experiment'], 'model': TRAIN_CONFIG['model'],
                    'dataset_ref': VALIDATION_DATASET_REF,
                    'kaggle_kernel_ref': os.getenv('KAGGLE_KERNEL_RUN_ID') or os.getenv('KAGGLE_KERNEL_INFERENCE_RUN_ID') or '',
                    'code_bundle_sha256': EXPECTED_BUNDLE_SHA256,
                    'training_wall_seconds': experiment_wall_seconds,
                    'configuration': TRAIN_CONFIG,
                    'training_report': sheets_report,
                }
                (OUTPUT_DIR / 'run_completion.json').write_text(
                    json.dumps(completion, ensure_ascii=False, indent=2), encoding='utf-8'
                )
                """
            ),
        ]
    )
    cells.extend(
        [
            markdown("## Completion marker"),
            code(
                """
                notebook_completed = {
                    **completion,
                    'success_gate': aggregate['success_gate'],
                    'deltas_macro_average_precision': aggregate['deltas_macro_average_precision'],
                    'google_sheets_status': 'disabled_by_user',
                }
                (WORKING_ROOT / 'notebook_completed.json').write_text(
                    json.dumps(notebook_completed, ensure_ascii=False, indent=2), encoding='utf-8'
                )
                (OUTPUT_DIR / 'COMPLETED').write_text('complete\\n', encoding='utf-8')
                shutil.rmtree(BASELINE_CHECKPOINT_DIR, ignore_errors=True)
                print(json.dumps(notebook_completed, ensure_ascii=False, indent=2))
                """
            ),
        ]
    )
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update(
        {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        }
    )
    return notebook


if __name__ == "__main__":
    raise SystemExit("Use scripts/run_minilm_s2_targeted_hard_kaggle.py")
