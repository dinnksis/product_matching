#!/usr/bin/env python3
"""Build the second autonomous attribute/embedding ablation notebook."""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

import create_embedding_boosting_notebook as shared
import create_qwen_training_notebook as bundle_shared


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = ROOT / "notebooks" / "attribute_boosting_v2_2xt4.ipynb"


def cell(value: str): return nbf.v4.new_code_cell(dedent(value).strip())
def markdown(value: str): return nbf.v4.new_markdown_cell(dedent(value).strip())


def build_notebook(manifest):
    expected_hash = str(manifest["code_bundle"]["sha256"])
    cells = [
        markdown("""
        # Attribute boosting v2

        Reuses the first run's name embeddings and tests normalized/fuzzy
        attributes, per-category CatBoost, and rich `name + attributes` Qwen
        embeddings. All cells run sequentially and persist outputs.
        """),
        cell(f"""
        import hashlib, shutil, subprocess, sys, zipfile
        from pathlib import Path, PurePosixPath

        INPUT = Path('/kaggle/input'); WORK = Path('/kaggle/working')
        PROJECT = WORK / 'product_matching'; OUTPUT = WORK / 'attribute_boosting_v2'
        EXPECTED_HASH = {expected_hash!r}

        def exactly_one(pattern):
            values = list(INPUT.glob(pattern))
            if len(values) != 1: raise RuntimeError(f'Expected one {{pattern}}, found {{values}}')
            return values[0]

        def sha256(path):
            digest = hashlib.sha256()
            with path.open('rb') as source:
                for chunk in iter(lambda: source.read(1024*1024), b''): digest.update(chunk)
            return digest.hexdigest()

        items = exactly_one('**/items_human.parquet'); matches = exactly_one('**/matches.parquet')
        embeddings = exactly_one('**/embedding_boosting/item_embeddings.f16.npy')
        previous = embeddings.parent
        bundles = list(INPUT.glob('**/{bundle_shared.BUNDLE_NAME}')) + [p for p in INPUT.glob('**/{Path(bundle_shared.BUNDLE_NAME).stem}') if p.is_dir()]
        if len(bundles) != 1: raise RuntimeError(f'Expected one code bundle, found {{bundles}}')
        bundle = bundles[0]; PROJECT.mkdir(parents=True, exist_ok=True)
        if bundle.is_file():
            if sha256(bundle) != EXPECTED_HASH: raise RuntimeError('Code bundle hash mismatch')
            with zipfile.ZipFile(bundle) as archive:
                for member in archive.namelist():
                    path = PurePosixPath(member)
                    if path.is_absolute() or '..' in path.parts: raise RuntimeError(member)
                archive.extractall(PROJECT)
        else: shutil.copytree(bundle, PROJECT, dirs_exist_ok=True)
        print(items, matches, previous, sep='\\n')
        print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
        """),
        cell("""
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet',
                        '-r', str(PROJECT/'requirements-embedding-boosting.txt')], check=True)
        """),
        cell("""
        command = [sys.executable, '-u', str(PROJECT/'src/attribute_boosting_v2.py'),
                   '--items', str(items), '--matches', str(matches),
                   '--config', str(PROJECT/'configs/attribute_boosting_v2.json'),
                   '--previous-output', str(previous), '--output-dir', str(OUTPUT)]
        print('$', ' '.join(command), flush=True)
        with (WORK/'attribute_boosting_v2_console.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end='', flush=True); log.write(line); log.flush()
            code = process.wait()
        if code: raise RuntimeError(f'Experiment failed: {{code}}')
        print((OUTPUT/'experiment_comparison.csv').read_text(encoding='utf-8'))
        """),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return notebook


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--env-file', type=Path, default=ROOT/'.env'); parser.add_argument('--output', type=Path, default=DEFAULT_NOTEBOOK); args=parser.parse_args()
    owner = bundle_shared.dotenv_username(args.env_file)
    if not owner: raise SystemExit('Set KAGGLE_USERNAME in .env')
    manifest = shared.build_code_dataset(shared.DEFAULT_CODE_DATASET_DIR, owner)
    args.output.parent.mkdir(parents=True, exist_ok=True); nbf.write(build_notebook(manifest), args.output); print(args.output)


if __name__ == '__main__': main()
