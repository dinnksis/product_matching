#!/usr/bin/env python3
"""Build the CPU Kaggle notebook for leakage-safe S2 + CatBoost evaluation."""

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
DEFAULT_NOTEBOOK = ROOT / "notebooks/s2_catboost_new_validation_splits.ipynb"
DEFAULT_CODE_DATASET_DIR = ROOT / ".kaggle/datasets/product-matching-s2-catboost-new-splits"
CODE_DATASET_SLUG = "product-matching-s2-catboost-new-splits"
BUNDLE_NAME = "s2_catboost_new_splits_bundle.zip"
MANIFEST_NAME = "s2_catboost_new_splits_manifest.json"
S2_EVALUATION_ROOT = Path(
    "artifacts/kaggle/product-matching-minilm-s0-s2-new-splits/"
    "minilm_s0_s2_new_splits/evaluations/S2_VALUES_ONLY"
)
BUNDLE_FILES = (
    Path("configs/cheap_ensemble_s2.json"),
    Path("src/__init__.py"),
    Path("src/cheap_ensemble.py"),
    Path("scripts/train_s2_cheap_ensemble.py"),
    Path("scripts/evaluate_s2_catboost_new_splits.py"),
    Path("artifacts/manual/S2_VALUES_ONLY/validation_predictions.parquet"),
    S2_EVALUATION_ROOT / "iid/predictions.parquet",
    S2_EVALUATION_ROOT / "hard/predictions.parquet",
    S2_EVALUATION_ROOT / "ood/predictions.parquet",
)


def markdown(value: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(value).strip())


def code(value: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(value).strip())


def build_code_dataset(directory: Path, owner: str) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, bytes] = {}
    files: dict[str, dict[str, object]] = {}
    for relative in BUNDLE_FILES:
        source = ROOT / relative
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f"Missing non-empty bundle input: {source}")
        payload = source.read_bytes()
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
                "title": "Product Matching S2 CatBoost New Splits",
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
    expected_hash = str(bundle["sha256"])
    expected_source = bundle["source"]
    cells = [
        markdown(
            """
            # MiniLM S2 + leakage-safe CatBoost on three external splits

            The transformer is frozen. CatBoost is fit on old leakage-safe S2
            holdout predictions after removing every product present in IID,
            hard, or OOD. Evaluation is external and item-disjoint.
            """
        ),
        code(
            f"""
            import hashlib
            import json
            import shutil
            import subprocess
            import sys
            import zipfile
            from pathlib import Path, PurePosixPath

            import pandas as pd

            INPUT_ROOT = Path('/kaggle/input')
            WORKING_ROOT = Path('/kaggle/working')
            PROJECT_ROOT = Path('/kaggle/temp/s2_catboost_new_splits/project')
            OUTPUT_DIR = WORKING_ROOT / 's2_catboost_new_splits'
            EXPECTED_BUNDLE_SHA256 = {expected_hash!r}
            EXPECTED_SOURCE_MANIFEST = {expected_source!r}

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
            split_manifest = exactly_one('validation_splits_manifest.json')
            bundle_candidates = list(INPUT_ROOT.glob('**/{BUNDLE_NAME}'))
            bundle_candidates.extend(
                path for path in INPUT_ROOT.glob('**/{Path(BUNDLE_NAME).stem}') if path.is_dir()
            )
            if len(bundle_candidates) != 1:
                raise RuntimeError(f'Expected one bundle, found {{bundle_candidates}}')
            bundle_path = bundle_candidates[0]
            PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
            if bundle_path.is_file():
                if sha256(bundle_path) != EXPECTED_BUNDLE_SHA256:
                    raise RuntimeError('Bundle SHA-256 mismatch')
                with zipfile.ZipFile(bundle_path) as archive:
                    for member in archive.namelist():
                        relative = PurePosixPath(member)
                        if relative.is_absolute() or '..' in relative.parts:
                            raise RuntimeError(f'Unsafe bundle member: {{member}}')
                    archive.extractall(PROJECT_ROOT)
            else:
                shutil.copytree(bundle_path, PROJECT_ROOT, dirs_exist_ok=True)
            source_manifest = json.loads(
                (PROJECT_ROOT / 'source_manifest.json').read_text(encoding='utf-8')
            )
            if source_manifest != EXPECTED_SOURCE_MANIFEST:
                raise RuntimeError('Source manifest mismatch')
            for relative, expected in source_manifest['files'].items():
                source = PROJECT_ROOT.joinpath(*PurePosixPath(relative).parts)
                if sha256(source) != expected['sha256']:
                    raise RuntimeError(f'Source hash mismatch: {{relative}}')
            split_dir = split_manifest.parent
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            print('items:', items_path)
            print('split_dir:', split_dir)
            print('bundle:', bundle_path)
            """
        ),
        markdown("## Dependencies and evaluation"),
        code(
            """
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--quiet',
                 '--disable-pip-version-check', 'catboost==1.2.8',
                 'rapidfuzz==3.14.5'],
                check=True,
            )
            command = [
                sys.executable, '-u',
                str(PROJECT_ROOT / 'scripts/evaluate_s2_catboost_new_splits.py'),
                '--items', str(items_path),
                '--meta-predictions', str(
                    PROJECT_ROOT / 'artifacts/manual/S2_VALUES_ONLY/validation_predictions.parquet'
                ),
                '--split-dir', str(split_dir),
                '--s2-evaluations-dir', str(
                    PROJECT_ROOT / 'artifacts/kaggle/product-matching-minilm-s0-s2-new-splits/'
                    'minilm_s0_s2_new_splits/evaluations/S2_VALUES_ONLY'
                ),
                '--config', str(PROJECT_ROOT / 'configs/cheap_ensemble_s2.json'),
                '--output-dir', str(OUTPUT_DIR),
            ]
            subprocess.run(command, check=True, cwd=PROJECT_ROOT)
            """
        ),
        markdown("## Result"),
        code(
            """
            report = json.loads((OUTPUT_DIR / 'evaluation_report.json').read_text(encoding='utf-8'))
            rows = []
            for split, values in report['split_reports'].items():
                rows.append({
                    'split': split,
                    'S2_macro_AP': values['transformer']['macro_average_precision'],
                    'S2_CatBoost_macro_AP': values['catboost']['macro_average_precision'],
                    'delta': values['absolute_delta_macro_ap'],
                })
            display(pd.DataFrame(rows))
            display(pd.DataFrame(report['top_catboost_features']))
            if not (OUTPUT_DIR / 'COMPLETED').is_file():
                raise RuntimeError('Completion marker missing')
            """
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update(
        {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        }
    )
    return notebook


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner")
    parser.add_argument("--output", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    args = parser.parse_args()
    owner = args.owner or shared.dotenv_username(args.env_file)
    if not owner:
        raise SystemExit("Set KAGGLE_USERNAME in .env or pass --owner")
    manifest = build_code_dataset(DEFAULT_CODE_DATASET_DIR, owner)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(manifest), args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
