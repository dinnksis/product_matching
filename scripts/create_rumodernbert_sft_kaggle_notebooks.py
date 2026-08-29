#!/usr/bin/env python3
"""Build exact RuModernBERT SFT notebooks for the sequential Kaggle campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from textwrap import dedent
from typing import Any, Mapping, Sequence

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import create_minilm_validation_baseline_notebook as validation_builder
import create_qwen_training_notebook as shared
import push_rumodernbert_pretrain_checkpoint_dataset as checkpoint_uploader


DEFAULT_CONFIG = ROOT / "configs" / "rumodernbert_3ep_sft_oodtrain_kaggle_v1.json"
DEFAULT_SOURCE_DIR = ROOT / "prepared" / "validation_splits_v1"
DEFAULT_CHECKPOINT_STAGE = ROOT / ".kaggle" / "datasets" / checkpoint_uploader.DATASET_SLUG
DEFAULT_OUTPUT_DIR = ROOT / "notebooks" / "rumodernbert_3ep_sft_kaggle_v1"
CAMPAIGN = "rumodernbert_3ep_sft_oodtrain_kaggle_v1"
EXPECTED_VALIDATION_MANIFEST_SHA256 = "b64a1902d86c9ad896a626b2a17bf018341f1d9c5fefa124834b525c84808f3c"
EXPECTED_ITEMS = 711_304
EXPECTED_HUMAN_TRAIN = 306_669
EXPECTED_FORMER_OOD = 41_171
EXPECTED_TRAIN = 347_840
EXPECTED_TRAIN_POSITIVES = 89_291
EXPECTED_IID = 12_000
EXPECTED_HARD = 5_814
EXPECTED_PARAMETERS = 149_605_633
EXPECTED_PARAMETER_TENSORS = 138
EXPECTED_MICROBATCH = 24
EXPECTED_EVAL_BATCH = 96
EXPECTED_ACCUMULATION = 4
EXPECTED_EFFECTIVE_BATCH = 192
EXPECTED_PREFLIGHT_STATE_POLICY = "explicit_rumodernbert_optimizer_tensor_count_v1"
EXPECTED_ACTIVATION_CHECKPOINTING_POLICY = "disabled_after_modernbert_autocast_checkpoint_error_v1"
MODEL_SHA256 = "e8b7ebda4904c2e7f8d2ec42645cfcbda928f90fa8dbd59d03e314318118673d"

RUNTIME_FILES = validation_builder.EMBEDDED_FILES + (
    Path("requirements-rumodernbert-kaggle.txt"),
    Path("scripts/train_bge_2ep_sft.py"),
    Path("scripts/train_rumodernbert_2xt4.py"),
    Path("scripts/rumodernbert_finite_bce_2xt4.py"),
)
LEDGER_ONLY_FILES = (
    Path("src/google_sheets_logger.py"),
    Path("scripts/create_qwen_training_notebook.py"),
    Path("scripts/create_minilm_validation_baseline_notebook.py"),
    Path("scripts/push_rumodernbert_pretrain_checkpoint_dataset.py"),
    Path("scripts/create_rumodernbert_sft_kaggle_notebooks.py"),
    DEFAULT_CONFIG.relative_to(ROOT),
)

OOD_SENTINEL = {
    "evaluated": False,
    "status": "disabled_train_contaminated",
    "reason": "former OOD pairs are part of RuModernBERT supervised training",
    "examples": 0,
    "source_training_examples": EXPECTED_FORMER_OOD,
    "macro_average_precision": -1.0,
    "overall_average_precision": -1.0,
    "recall_at_precision_0_99": -1.0,
    "threshold_at_precision_0_99": -1.0,
    "roc_auc": -1.0,
    "log_loss": -1.0,
    "per_category_average_precision": {},
    "predictions_file": None,
}


class CampaignError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_plan(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1 or plan.get("campaign") != CAMPAIGN:
        raise CampaignError("unexpected RuModernBERT Kaggle plan")
    experiments = plan.get("experiments")
    if not isinstance(experiments, list) or [x.get("key") for x in experiments] != [
        "e1_lr8e5", "e1_lr4e5", "e1_lr1p6e4", "e2_selected_lr", "e3_selected_lr"
    ]:
        raise CampaignError("campaign must contain the exact ordered five-run plan")
    training = plan.get("training")
    required_geometry = {
        "batch_size": EXPECTED_MICROBATCH,
        "eval_batch_size": EXPECTED_EVAL_BATCH,
        "gradient_accumulation": EXPECTED_ACCUMULATION,
        "effective_batch": EXPECTED_EFFECTIVE_BATCH,
        "max_length": 384,
        "gradient_checkpointing": False,
        "world_size": 2,
        "gpu": "two_Tesla_T4",
    }
    if not isinstance(training, dict) or any(training.get(k) != v for k, v in required_geometry.items()):
        raise CampaignError("frozen two-T4 geometry changed")
    return plan


def source_bundle() -> tuple[dict[str, str], list[dict[str, Any]], str]:
    runtime_paths = {path.as_posix() for path in RUNTIME_FILES}
    all_paths = list(dict.fromkeys((*RUNTIME_FILES, *LEDGER_ONLY_FILES)))
    sources: dict[str, str] = {}
    ledger: list[dict[str, Any]] = []
    for relative in all_paths:
        path = ROOT / relative
        if not path.is_file():
            raise CampaignError(f"required campaign source is missing: {relative}")
        content = path.read_text(encoding="utf-8")
        name = relative.as_posix()
        embedded = name in runtime_paths
        if embedded:
            sources[name] = content
        ledger.append(
            {
                "path": name,
                "bytes": len(content.encode("utf-8")),
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "runtime_embedded": embedded,
            }
        )
    ledger_hash = canonical_sha256({"schema_version": 1, "files": ledger})
    return sources, ledger, ledger_hash


def load_validation(owner: str, source_dir: Path = DEFAULT_SOURCE_DIR) -> dict[str, Any]:
    value = validation_builder.load_manifest(source_dir, owner)
    if value["manifest_sha256"] != EXPECTED_VALIDATION_MANIFEST_SHA256:
        raise CampaignError("validation manifest SHA-256 changed")
    return value


def load_checkpoint(owner: str, stage_dir: Path = DEFAULT_CHECKPOINT_STAGE) -> dict[str, Any]:
    manifest_path = stage_dir / checkpoint_uploader.MANIFEST_NAME
    if not manifest_path.is_file():
        raise CampaignError("checkpoint Dataset stage is missing; run uploader --dry-run")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_ref = f"{owner}/{checkpoint_uploader.DATASET_SLUG}"
    if manifest.get("dataset") != expected_ref or manifest.get("is_private") is not True:
        raise CampaignError("checkpoint Dataset identity changed")
    if manifest.get("checkpoint_files") != checkpoint_uploader.EXPECTED_SOURCE_FILES:
        raise CampaignError("checkpoint file ledger changed")
    if manifest.get("reconstruction", {}).get("sha256") != MODEL_SHA256:
        raise CampaignError("checkpoint reconstruction SHA-256 changed")
    with checkpoint_uploader.hardened_contract():
        checkpoint_uploader.hardened.verify_payload_for_upload(
            stage_dir,
            manifest,
            expected_checkpoint_files=checkpoint_uploader.EXPECTED_SOURCE_FILES,
        )
    return {
        "dataset": expected_ref,
        "manifest": manifest,
        "manifest_sha256": file_sha256(manifest_path),
    }


def resolve_variant(plan: Mapping[str, Any], key: str, selected_lr: float | None) -> dict[str, Any]:
    spec = next((dict(x) for x in plan["experiments"] if x["key"] == key), None)
    if spec is None:
        raise CampaignError(f"unknown campaign key: {key}")
    learning_rate = spec["learning_rate"]
    if learning_rate == "selected_from_first_three":
        if selected_lr not in {4e-5, 8e-5, 1.6e-4}:
            raise CampaignError(f"{key} requires a selected LR from the frozen line")
        learning_rate = selected_lr
    elif selected_lr is not None:
        raise CampaignError("selected_lr is allowed only for e2/e3")
    spec["learning_rate"] = float(learning_rate)
    spec["experiment"] = f"rumodernbert_sft_{key}_v1"
    return spec


def resolved_config(plan: Mapping[str, Any], variant: Mapping[str, Any], runtime_model: str) -> dict[str, Any]:
    config = deepcopy(dict(plan["training"]))
    for key in ("effective_batch", "loss", "amp_dtype", "world_size", "gpu"):
        config.pop(key)
    config.update(
        {
            "model": runtime_model,
            "model_load_kwargs": {},
            "epochs": int(variant["epochs"]),
            "learning_rate": float(variant["learning_rate"]),
        }
    )
    return config


def markdown(text: str, *tags: str) -> nbf.NotebookNode:
    cell = nbf.v4.new_markdown_cell(dedent(text).strip())
    if tags:
        cell.metadata["tags"] = list(tags)
    return cell


def code(text: str, *tags: str) -> nbf.NotebookNode:
    cell = nbf.v4.new_code_cell(dedent(text).strip())
    if tags:
        cell.metadata["tags"] = list(tags)
    return cell


def _setup_cell(entry: Mapping[str, Any], checkpoint: Mapping[str, Any], validation: Mapping[str, Any]) -> nbf.NotebookNode:
    return code(
        f"""
        import hashlib
        import json
        import math
        import os
        import shutil
        import subprocess
        import sys
        import time
        from pathlib import Path

        INPUT_ROOT = Path('/kaggle/input')
        WORKING_ROOT = Path('/kaggle/working')
        TEMP_ROOT = Path({('/kaggle/temp/' + str(entry['experiment']))!r})
        PROJECT_ROOT = WORKING_ROOT / 'product_matching'
        PREPARED_DIR = TEMP_ROOT / 'prepared'
        TOKEN_CACHE_DIR = TEMP_ROOT / 'token_cache'
        TRAINER_OUTPUT_DIR = TEMP_ROOT / 'trainer_output'
        OUTPUT_DIR = WORKING_ROOT / {str(entry['experiment'])!r}
        CONFIG_PATH = WORKING_ROOT / 'rumodernbert_training_config.json'
        PREFLIGHT_REPORT_PATH = WORKING_ROOT / 'rumodernbert_memory_preflight.json'
        PREFLIGHT_LOG = WORKING_ROOT / 'rumodernbert_memory_preflight.log'
        TRAIN_LOG = WORKING_ROOT / 'rumodernbert_training.log'
        TRAIN_DATA_REPORT_PATH = WORKING_ROOT / 'rumodernbert_train_data_report.json'
        RUNTIME_VERSIONS_PATH = WORKING_ROOT / 'rumodernbert_runtime_versions.json'
        EXPECTED_IDENTITY = {str(entry['identity_sha256'])!r}
        EXPECTED_SOURCE_SHA256 = {str(entry['source_sha256'])!r}
        EXPECTED_VALIDATION_REF = {str(validation['dataset'])!r}
        EXPECTED_VALIDATION_MANIFEST_SHA256 = {str(validation['manifest_sha256'])!r}
        EXPECTED_CHECKPOINT_REF = {str(checkpoint['dataset'])!r}
        EXPECTED_CHECKPOINT_MANIFEST_SHA256 = {str(checkpoint['manifest_sha256'])!r}

        def file_sha256(path):
            digest = hashlib.sha256()
            with path.open('rb') as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            return digest.hexdigest()

        def dataset_file(slug, filename):
            candidates = [p for p in INPUT_ROOT.glob(f'**/{{filename}}') if slug in p.parts]
            if len(candidates) != 1:
                raise RuntimeError(f'expected one {{filename}} under {{slug}}, found {{candidates}}')
            return candidates[0]

        print(subprocess.run(['nvidia-smi'], check=False, capture_output=True, text=True).stdout)
        """,
        "frozen", "environment",
    )


def _bootstrap_cell(
    entry: Mapping[str, Any],
    sources: Mapping[str, str],
    ledger: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
) -> nbf.NotebookNode:
    validation_slug = validation_builder.DATASET_SLUG
    checkpoint_slug = checkpoint_uploader.DATASET_SLUG
    return code(
        f"""
        SOURCES = {dict(sources)!r}
        SOURCE_LEDGER = {list(ledger)!r}
        ledger_payload = json.dumps(
            {{'schema_version': 1, 'files': SOURCE_LEDGER}},
            ensure_ascii=False, sort_keys=True, separators=(',', ':')
        ).encode('utf-8')
        if hashlib.sha256(ledger_payload).hexdigest() != EXPECTED_SOURCE_SHA256:
            raise RuntimeError('source ledger changed')
        records = {{row['path']: row for row in SOURCE_LEDGER}}
        if set(SOURCES) != {{row['path'] for row in SOURCE_LEDGER if row['runtime_embedded']}}:
            raise RuntimeError('runtime source set changed')
        PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
        for relative, content in SOURCES.items():
            payload = content.encode('utf-8')
            declaration = records[relative]
            if len(payload) != declaration['bytes'] or hashlib.sha256(payload).hexdigest() != declaration['sha256']:
                raise RuntimeError(f'embedded source changed: {{relative}}')
            destination = PROJECT_ROOT / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)

        validation_manifest_path = dataset_file({validation_slug!r}, 'validation_splits_manifest.json')
        if file_sha256(validation_manifest_path) != EXPECTED_VALIDATION_MANIFEST_SHA256:
            raise RuntimeError('validation manifest changed')
        validation_manifest = json.loads(validation_manifest_path.read_text(encoding='utf-8'))
        REMOTE_FILES = {validation_builder.REMOTE_FILES!r}
        attached_files = {{
            relative: dataset_file({validation_slug!r}, remote)
            for relative, remote in REMOTE_FILES.items()
        }}
        for relative, path in attached_files.items():
            expected = validation_manifest['outputs'][relative]
            if path.stat().st_size != expected['bytes'] or file_sha256(path) != expected['sha256']:
                raise RuntimeError(f'validation file changed: {{relative}}')

        checkpoint_manifest_path = dataset_file({checkpoint_slug!r}, 'checkpoint_manifest.json')
        if file_sha256(checkpoint_manifest_path) != EXPECTED_CHECKPOINT_MANIFEST_SHA256:
            raise RuntimeError('checkpoint manifest changed')
        checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding='utf-8'))
        if checkpoint_manifest.get('dataset') != EXPECTED_CHECKPOINT_REF or checkpoint_manifest.get('is_private') is not True:
            raise RuntimeError('checkpoint Dataset identity changed')
        checkpoint_root = checkpoint_manifest_path.parent
        for filename, declaration in checkpoint_manifest['files'].items():
            path = checkpoint_root / filename
            if not path.is_file() or path.stat().st_size != declaration['bytes'] or file_sha256(path) != declaration['sha256']:
                raise RuntimeError(f'checkpoint shard changed: {{filename}}')
        reconstruction = checkpoint_manifest['reconstruction']
        initial_model = TEMP_ROOT / 'pretrain_rumodernbert_3ep'
        initial_model.mkdir(parents=True, exist_ok=True)
        part_names = set(reconstruction['parts'])
        for filename in checkpoint_manifest['files']:
            if filename in part_names:
                continue
            destination = initial_model / filename
            destination.unlink(missing_ok=True)
            destination.symlink_to(checkpoint_root / filename)
        reconstructed = initial_model / reconstruction['filename']
        digest = hashlib.sha256()
        with reconstructed.open('wb') as output:
            for part in reconstruction['parts']:
                with (checkpoint_root / part).open('rb') as source:
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                        output.write(chunk)
                        digest.update(chunk)
        if reconstructed.stat().st_size != reconstruction['bytes'] or digest.hexdigest() != {MODEL_SHA256!r}:
            raise RuntimeError('reconstructed RuModernBERT model changed')
        INITIAL_MODEL_PATH = initial_model

        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',
             '--upgrade-strategy', 'only-if-needed', '-r',
             str(PROJECT_ROOT / 'requirements-rumodernbert-kaggle.txt')],
            check=True,
        )
        from importlib import metadata as importlib_metadata
        import numpy as np
        import pandas as pd
        RUNTIME_VERSIONS = {{
            'schema_version': 1,
            'python': sys.version.split()[0],
            'packages': {{name: importlib_metadata.version(name) for name in (
                'numpy', 'pandas', 'pyarrow', 'scikit-learn', 'torch', 'transformers'
            )}},
        }}
        if RUNTIME_VERSIONS['packages']['transformers'] != '4.57.6':
            raise RuntimeError('unexpected transformers version')
        RUNTIME_VERSIONS_PATH.write_text(json.dumps(RUNTIME_VERSIONS, indent=2) + '\\n')
        """,
        "frozen", "bootstrap",
    )


def _data_recipe_cell(entry: Mapping[str, Any]) -> nbf.NotebookNode:
    config = entry["expected_config"]
    return code(
        f"""
        items = pd.read_parquet(attached_files['human/items.parquet'])
        human_train = pd.read_parquet(attached_files['human/train_pairs.parquet'])
        former_ood = pd.read_parquet(attached_files['human/ood_validation_pairs.parquet'])
        iid = pd.read_parquet(attached_files['human/iid_validation_pairs.parquet'])
        hard = pd.read_parquet(attached_files['human/hard_validation_pairs.parquet'])
        if len(items) != {EXPECTED_ITEMS} or not items['id'].is_unique:
            raise RuntimeError('item table changed')
        item_categories = items.set_index('id')['category']
        expected_rows = {{'human_train': {EXPECTED_HUMAN_TRAIN}, 'former_ood': {EXPECTED_FORMER_OOD}, 'iid': {EXPECTED_IID}, 'hard': {EXPECTED_HARD}}}

        def validate_pairs(name, frame):
            if len(frame) != expected_rows[name] or {{'id1','id2','target'}} - set(frame):
                raise RuntimeError(f'pair contract changed: {{name}}')
            if frame[['id1','id2','target']].isnull().any().any() or not frame['target'].isin([0.0,1.0]).all():
                raise RuntimeError(f'invalid pairs: {{name}}')
            lower = np.minimum(frame['id1'].to_numpy(), frame['id2'].to_numpy())
            upper = np.maximum(frame['id1'].to_numpy(), frame['id2'].to_numpy())
            if pd.MultiIndex.from_arrays([lower, upper]).duplicated().any():
                raise RuntimeError(f'duplicate unordered pairs: {{name}}')
            left = frame['id1'].map(item_categories)
            right = frame['id2'].map(item_categories)
            if left.isnull().any() or right.isnull().any() or not left.equals(right):
                raise RuntimeError(f'bad item/category binding: {{name}}')
            return set(frame['id1']) | set(frame['id2']), set(left.astype(str))

        contracts = {{name: validate_pairs(name, frame) for name, frame in (
            ('human_train', human_train), ('former_ood', former_ood), ('iid', iid), ('hard', hard)
        )}}
        if contracts['former_ood'][1] != {{'Одежда', 'Бытовая техника'}}:
            raise RuntimeError('former OOD categories changed')
        train_ids = contracts['human_train'][0] | contracts['former_ood'][0]
        if train_ids & contracts['iid'][0] or train_ids & contracts['hard'][0]:
            raise RuntimeError('train/validation item leakage')
        human_train = human_train.copy(); human_train['label_source'] = 'human_train'
        former_ood = former_ood.copy(); former_ood['label_source'] = 'human_former_ood'
        train = pd.concat([human_train, former_ood], ignore_index=True, sort=False)
        if len(train) != {EXPECTED_TRAIN} or int(train['target'].sum()) != {EXPECTED_TRAIN_POSITIVES}:
            raise RuntimeError('combined train changed')
        PREPARED_DIR.mkdir(parents=True, exist_ok=True)
        train.to_parquet(PREPARED_DIR / 'train_pairs.parquet', index=False, compression='zstd')
        for relative, name in (
            ('human/items.parquet','items.parquet'),
            ('human/iid_validation_pairs.parquet','iid_validation_pairs.parquet'),
            ('human/hard_validation_pairs.parquet','hard_validation_pairs.parquet'),
        ):
            (PREPARED_DIR / name).symlink_to(attached_files[relative])
        TRAIN_DATA_REPORT = {{
            'policy': 'human_train_plus_former_ood_exact_concat_v1',
            'items': len(items), 'train_pairs': len(train),
            'train_positives': int(train['target'].sum()),
            'source_counts': {{str(k): int(v) for k,v in train['label_source'].value_counts().items()}},
            'validation_rows': {{'iid': len(iid), 'hard': len(hard)}},
            'ood_evaluation': 'disabled_train_contaminated',
        }}
        TRAIN_DATA_REPORT_PATH.write_text(json.dumps(TRAIN_DATA_REPORT, ensure_ascii=False, indent=2) + '\\n')

        TRAIN_CONFIG = {dict(config)!r}
        TRAIN_CONFIG['model'] = str(INITIAL_MODEL_PATH)
        if hashlib.sha256(json.dumps({dict(config)!r}, ensure_ascii=False, sort_keys=True, separators=(',',':')).encode()).hexdigest() != {str(entry['recipe_sha256'])!r}:
            raise RuntimeError('frozen recipe changed')
        CONFIG_PATH.write_text(json.dumps(TRAIN_CONFIG, ensure_ascii=False, indent=2) + '\\n')
        """,
        "frozen", "data-and-recipe",
    )


def _process_helper() -> str:
    return dedent(
        """
        def run_logged(command, log_path):
            print('$', ' '.join(str(x) for x in command), flush=True)
            environment = os.environ.copy()
            environment.update({'OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'true','NCCL_DEBUG':'WARN','PYTHONUNBUFFERED':'1'})
            with log_path.open('w', encoding='utf-8', buffering=1) as log:
                process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in process.stdout:
                    print(line, end='', flush=True); log.write(line)
                return_code = process.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, command)
        """
    ).strip()


def _preflight_training_cell() -> nbf.NotebookNode:
    return code(
        _process_helper()
        + "\n\n"
        + dedent(
            f"""
            preflight_command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                str(PROJECT_ROOT / 'scripts/train_rumodernbert_2xt4.py'), '--memory-preflight-only',
                '--config', str(CONFIG_PATH), '--preflight-report', str(PREFLIGHT_REPORT_PATH)]
            run_logged(preflight_command, PREFLIGHT_LOG)
            memory_preflight = json.loads(PREFLIGHT_REPORT_PATH.read_text())
            required = {{
                'status':'passed', 'world_size':2, 'parameters':{EXPECTED_PARAMETERS},
                'microbatch_per_gpu':{EXPECTED_MICROBATCH}, 'gradient_accumulation':{EXPECTED_ACCUMULATION},
                'effective_batch':{EXPECTED_EFFECTIVE_BATCH}, 'eval_batch_per_gpu':{EXPECTED_EVAL_BATCH},
                'optimizer_state_parameters_per_rank':{EXPECTED_PARAMETER_TENSORS},
                'optimizer_state_tensor_elements_per_rank':{2 * EXPECTED_PARAMETERS},
                'amp_dtype':'float16', 'gradient_checkpointing':False,
                'optimizer_state_materialization_policy':{EXPECTED_PREFLIGHT_STATE_POLICY!r},
                'optimizer_state_expected_parameter_tensors':{EXPECTED_PARAMETER_TENSORS},
                'activation_checkpointing_policy':{EXPECTED_ACTIVATION_CHECKPOINTING_POLICY!r},
                'full_training_uses_synthetic_zero_gradients':False,
            }}
            if any(memory_preflight.get(k) != v for k,v in required.items()):
                raise RuntimeError('RuModernBERT two-T4 memory preflight contract failed')
            train_command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                str(PROJECT_ROOT / 'scripts/train_rumodernbert_2xt4.py'), '--config', str(CONFIG_PATH),
                '--prepared-dir', str(PREPARED_DIR), '--output-dir', str(TRAINER_OUTPUT_DIR),
                '--token-cache-dir', str(TOKEN_CACHE_DIR),
                '--loss-hook', str(PROJECT_ROOT / 'scripts/rumodernbert_finite_bce_2xt4.py'),
                '--validation-split', 'iid=iid_validation_pairs.parquet',
                '--validation-split', 'hard=hard_validation_pairs.parquet']
            started = time.perf_counter()
            run_logged(train_command, TRAIN_LOG)
            training_wall_seconds = time.perf_counter() - started
            """
        ),
        "frozen", "preflight-and-training",
    )


def _completion_cell(entry: Mapping[str, Any]) -> nbf.NotebookNode:
    return code(
        f"""
        from datetime import datetime, timezone
        raw_report = json.loads((TRAINER_OUTPUT_DIR / 'training_report.json').read_text())
        if set(raw_report.get('validation_splits', {{}})) != {{'iid','hard'}}:
            raise RuntimeError('trainer evaluated an unexpected split')
        if raw_report.get('original_training_examples') != {EXPECTED_TRAIN}:
            raise RuntimeError('trainer row count changed')
        for split, rows in (('iid',{EXPECTED_IID}),('hard',{EXPECTED_HARD})):
            metrics = raw_report['validation_splits'][split]
            if metrics.get('examples') != rows or not math.isfinite(float(metrics['macro_average_precision'])):
                raise RuntimeError(f'invalid {{split}} metrics')
        raw_report['validation_splits']['ood'] = {OOD_SENTINEL!r}
        raw_report['experiment_group'] = 'sft'
        raw_report['ood_evaluation_policy'] = 'disabled_train_contaminated'
        OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(TRAINER_OUTPUT_DIR, OUTPUT_DIR)
        (OUTPUT_DIR / 'training_report.json').write_text(json.dumps(raw_report, ensure_ascii=False, indent=2, default=str) + '\\n')
        for source in (TRAIN_LOG, PREFLIGHT_LOG, PREFLIGHT_REPORT_PATH, TRAIN_DATA_REPORT_PATH, RUNTIME_VERSIONS_PATH):
            shutil.copy2(source, OUTPUT_DIR / source.name)
        model_path = OUTPUT_DIR / 'model.safetensors'
        if not model_path.is_file() or model_path.stat().st_size == 0:
            raise RuntimeError('trained model was not saved')
        completion = {{
            'schema_version': 1, 'status':'complete',
            'run_id': EXPERIMENT_RUN_ID, 'started_at_utc': EXPERIMENT_STARTED_AT_UTC,
            'completed_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z'),
            'experiment': {str(entry['experiment'])!r}, 'experiment_group':'sft',
            'campaign': {CAMPAIGN!r}, 'key': {str(entry['key'])!r}, 'role': {str(entry['role'])!r},
            'model': EXPECTED_CHECKPOINT_REF, 'dataset_ref': EXPECTED_VALIDATION_REF,
            'campaign_identity_sha256': EXPECTED_IDENTITY,
            'code_bundle_sha256': EXPECTED_SOURCE_SHA256,
            'frozen_recipe_sha256': {str(entry['recipe_sha256'])!r},
            'initial_checkpoint_manifest_sha256': EXPECTED_CHECKPOINT_MANIFEST_SHA256,
            'initial_checkpoint_model_sha256': {MODEL_SHA256!r},
            'trained_model_relative_path': {str(entry['experiment'])!r} + '/model.safetensors',
            'trained_model_bytes': model_path.stat().st_size,
            'trained_model_sha256': file_sha256(model_path),
            'train_data': TRAIN_DATA_REPORT, 'memory_preflight': memory_preflight,
            'runtime_versions': RUNTIME_VERSIONS, 'training_wall_seconds': training_wall_seconds,
            'training_report': raw_report,
            'kaggle_kernel_ref': os.getenv('KAGGLE_KERNEL_RUN_ID') or '',
            'notes': json.dumps({{'epochs':{int(entry['epochs'])},'learning_rate':{float(entry['learning_rate'])},'ood_metric':-1}}, sort_keys=True),
        }}
        completion_path = WORKING_ROOT / 'notebook_completed.json'
        completion_path.write_text(json.dumps(completion, ensure_ascii=False, indent=2, default=str) + '\\n')
        (OUTPUT_DIR / 'notebook_completed.json').write_text(completion_path.read_text())
        print(json.dumps({{'iid_macro_ap':raw_report['validation_splits']['iid']['macro_average_precision'],
            'hard_macro_ap':raw_report['validation_splits']['hard']['macro_average_precision'],
            'ood_macro_ap':-1.0,'model_sha256':completion['trained_model_sha256']}}, indent=2))
        """,
        "frozen", "completion",
    )


def build_variant(
    *, owner: str, key: str, selected_lr: float | None = None,
    plan_path: Path = DEFAULT_CONFIG, source_dir: Path = DEFAULT_SOURCE_DIR,
    checkpoint_stage: Path = DEFAULT_CHECKPOINT_STAGE,
) -> tuple[nbf.NotebookNode, dict[str, Any]]:
    plan = load_plan(plan_path)
    validation = load_validation(owner, source_dir)
    checkpoint = load_checkpoint(owner, checkpoint_stage)
    variant = resolve_variant(plan, key, selected_lr)
    sources, ledger, source_sha = source_bundle()
    runtime_model = f"/kaggle/temp/{variant['experiment']}/pretrain_rumodernbert_3ep"
    config = resolved_config(plan, variant, runtime_model)
    recipe_sha = canonical_sha256(config)
    identity = canonical_sha256({
        'schema_version':1, 'campaign':CAMPAIGN, 'variant':variant,
        'config':config, 'source_sha256':source_sha,
        'validation_manifest_sha256':validation['manifest_sha256'],
        'checkpoint_manifest_sha256':checkpoint['manifest_sha256'],
    })
    slug = f"pm-rmb-{variant['slug_token']}-{identity[:12]}-s42-v1"
    if len(slug) > 50 or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug):
        raise CampaignError(f"unsafe Kaggle slug: {slug}")
    entry = {**variant, 'kernel_slug':slug, 'title':slug,
        'identity_sha256':identity, 'source_sha256':source_sha,
        'recipe_sha256':recipe_sha, 'expected_config':config,
        'validation_dataset':validation['dataset'],
        'validation_manifest_sha256':validation['manifest_sha256'],
        'checkpoint_dataset':checkpoint['dataset'],
        'checkpoint_manifest_sha256':checkpoint['manifest_sha256'],
        'checkpoint_model_sha256':MODEL_SHA256}
    run_identity = shared.experiment_run_initialization_cell(); run_identity.metadata['tags']=['frozen','run-identity']
    sheet_cells = shared.google_sheets_tracking_cells()
    for cell in sheet_cells: cell.metadata['tags']=['frozen','sheets-sync']
    cells = [
        markdown(f"# RuModernBERT Kaggle SFT: `{variant['key']}`\n\nIdentity `{identity}`.", 'frozen'),
        _setup_cell(entry, checkpoint, validation), run_identity,
        _bootstrap_cell(entry, sources, ledger, checkpoint),
        _data_recipe_cell(entry), _preflight_training_cell(), _completion_cell(entry),
        *sheet_cells,
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    for index, cell in enumerate(notebook.cells):
        cell['id'] = f"rmb-{index:02d}-{hashlib.sha256((str(index)+str(cell.source)).encode()).hexdigest()[:12]}"
    notebook.metadata.update({
        'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'},
        'language_info': {'name':'python','version':'3.11'},
        'product_matching_training': {
            'template':'rumodernbert_2xt4_v1','campaign':CAMPAIGN,'key':key,
            'kernel_slug':slug,'identity_sha256':identity,'source_sha256':source_sha,
            'recipe_sha256':recipe_sha,'validation_dataset':validation['dataset'],
            'checkpoint_dataset':checkpoint['dataset'],'expected_gpus':2,
            'train_pairs':EXPECTED_TRAIN,'validation_splits':['iid','hard'],'ood_metric_sentinel':-1,
        },
    })
    nbf.validate(notebook)
    entry['notebook_code_sha256'] = canonical_sha256([
        str(cell.source) for cell in notebook.cells if cell.cell_type == 'code'
    ])
    return notebook, entry


def write_variant(notebook: nbf.NotebookNode, entry: Mapping[str, Any], output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{entry['key']}_{entry['identity_sha256'][:12]}_2xt4.ipynb"
    nbf.write(notebook, destination)
    reloaded = nbf.read(destination, as_version=4)
    if canonical_sha256([str(c.source) for c in reloaded.cells if c.cell_type == 'code']) != entry['notebook_code_sha256']:
        raise CampaignError('written notebook code changed')
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-file', type=Path, default=ROOT / '.env')
    parser.add_argument('--owner')
    parser.add_argument('--key', required=True)
    parser.add_argument('--selected-lr', type=float)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    owner = args.owner or shared.dotenv_username(args.env_file)
    if not owner:
        raise SystemExit('Set KAGGLE_USERNAME or pass --owner')
    notebook, entry = build_variant(owner=owner, key=args.key, selected_lr=args.selected_lr)
    destination = write_variant(notebook, entry, args.output_dir)
    print(json.dumps({**entry, 'notebook':str(destination)}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
