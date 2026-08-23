#!/usr/bin/env python3
"""Build the autonomous 2xT4 Qwen-embedding + CatBoost ablation notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

import create_qwen_training_notebook as shared


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = ROOT / "notebooks" / "embedding_boosting_2xt4.ipynb"
DEFAULT_CODE_DATASET_DIR = ROOT / ".kaggle" / "datasets" / "product-matching-embedding-boosting-code"
CODE_DATASET_SLUG = "product-matching-embedding-boosting-code"


def build_code_dataset(dataset_dir: Path, owner: str) -> dict[str, object]:
    """Build a small Kaggle Dataset with only exact experiment source code."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = dataset_dir / shared.BUNDLE_NAME
    source_manifest = shared.write_code_bundle(bundle_path)
    manifest = {
        "schema_version": 1,
        "dataset": f"{owner}/{CODE_DATASET_SLUG}",
        "code_bundle": {
            "filename": shared.BUNDLE_NAME,
            "bytes": bundle_path.stat().st_size,
            "sha256": shared.sha256(bundle_path),
            "source": source_manifest,
        },
    }
    (dataset_dir / shared.MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "title": "Product Matching Embedding Boosting Code",
        "id": f"{owner}/{CODE_DATASET_SLUG}",
        "licenses": [{"name": "unknown"}],
        "description": "Private reproducible source bundle for the embedding boosting experiments.",
    }
    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def markdown(value: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(value).strip())


def code(value: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(value).strip())


def build_notebook(manifest: dict[str, object]) -> nbf.NotebookNode:
    bundle = manifest["code_bundle"]
    assert isinstance(bundle, dict)
    expected_hash = str(bundle["sha256"])
    cells = [
        markdown(
            """
            # Qwen3-Embedding-0.6B + CatBoost: three controlled experiments

            This notebook runs end-to-end without manual cells:

            1. lexical name features;
            2. names plus cached Qwen item embeddings;
            3. names, embeddings, and structured JSON attributes.

            All three models use the same component-disjoint validation split.
            Final models, predictions, reports, logs, the selected attribute
            keys, and the float16 embedding cache are saved in
            `/kaggle/working/embedding_boosting`.
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

            INPUT_ROOT = Path('/kaggle/input')
            WORKING_ROOT = Path('/kaggle/working')
            PROJECT_ROOT = WORKING_ROOT / 'product_matching'
            OUTPUT_DIR = WORKING_ROOT / 'embedding_boosting'
            EXPECTED_BUNDLE_SHA256 = {expected_hash!r}

            def exactly_one(filename):
                candidates = list(INPUT_ROOT.glob(f'**/{{filename}}'))
                if len(candidates) != 1:
                    raise RuntimeError(f'Expected exactly one {{filename}}, found {{candidates}}')
                return candidates[0]

            def sha256(path):
                digest = hashlib.sha256()
                with path.open('rb') as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b''):
                        digest.update(chunk)
                return digest.hexdigest()

            items_path = exactly_one('items_human.parquet')
            matches_path = exactly_one('matches.parquet')
            PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
            bundle_files = list(INPUT_ROOT.glob('**/{shared.BUNDLE_NAME}'))
            bundle_directories = [
                path for path in INPUT_ROOT.glob('**/{Path(shared.BUNDLE_NAME).stem}')
                if path.is_dir()
            ]
            bundle_candidates = bundle_files + bundle_directories
            if len(bundle_candidates) != 1:
                raise RuntimeError(f'Expected one code bundle, found {{bundle_candidates}}')
            bundle_path = bundle_candidates[0]
            if bundle_path.is_file():
                if sha256(bundle_path) != EXPECTED_BUNDLE_SHA256:
                    raise RuntimeError('Attached source bundle hash does not match this notebook')
                with zipfile.ZipFile(bundle_path) as archive:
                    for member in archive.namelist():
                        path = PurePosixPath(member)
                        if path.is_absolute() or '..' in path.parts:
                            raise RuntimeError(f'Unsafe bundle member: {{member}}')
                    archive.extractall(PROJECT_ROOT)
            else:
                shutil.copytree(bundle_path, PROJECT_ROOT, dirs_exist_ok=True)
            print('items:', items_path)
            print('matches:', matches_path)
            print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
            """
        ),
        markdown("## Install the pinned experiment environment"),
        code(
            """
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--quiet',
                 '--disable-pip-version-check', '--upgrade-strategy', 'only-if-needed',
                 '-r', str(PROJECT_ROOT / 'requirements-embedding-boosting.txt')],
                check=True,
            )
            """
        ),
        markdown("## Run all experiments and persist every artifact"),
        code(
            """
            command = [
                sys.executable, '-u', str(PROJECT_ROOT / 'src/embedding_boosting.py'),
                '--items', str(items_path),
                '--matches', str(matches_path),
                '--config', str(PROJECT_ROOT / 'configs/embedding_boosting.json'),
                '--output-dir', str(OUTPUT_DIR),
            ]
            print('$', ' '.join(command), flush=True)
            with (WORKING_ROOT / 'embedding_boosting_console.log').open('w', encoding='utf-8') as log:
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                return_code = process.wait()
            if return_code:
                raise RuntimeError(f'Experiment process failed with exit code {return_code}')
            print((OUTPUT_DIR / 'experiment_comparison.csv').read_text(encoding='utf-8'))
            print('COMPLETED:', OUTPUT_DIR)
            """
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python", "version": "3"}
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
