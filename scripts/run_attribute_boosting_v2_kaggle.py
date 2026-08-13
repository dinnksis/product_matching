#!/usr/bin/env python3
"""Publish and run the second attribute boosting experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

import create_attribute_boosting_v2_notebook as notebook_builder
import create_embedding_boosting_notebook as code_builder
import create_qwen_training_notebook as shared
import run_embedding_boosting_kaggle as first_runner


ROOT = Path(__file__).resolve().parents[1]
SLUG = "product-matching-attribute-boosting-v2"


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--env-file', type=Path, default=ROOT/'.env'); parser.add_argument('--dry-run', action='store_true'); parser.add_argument('--wait', action='store_true'); args=parser.parse_args()
    env_file = args.env_file if args.env_file.is_absolute() else ROOT/args.env_file
    owner=shared.dotenv_username(env_file)
    if not owner: raise SystemExit('Set KAGGLE_USERNAME in .env')
    manifest=first_runner.publish_code_dataset(env_file, owner, args.dry_run)
    notebook=notebook_builder.build_notebook(manifest); notebook_builder.DEFAULT_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True); nbf.write(notebook, notebook_builder.DEFAULT_NOTEBOOK)
    command=[sys.executable, str(ROOT/'scripts/run_kaggle_notebook.py'), str(notebook_builder.DEFAULT_NOTEBOOK), '--env-file', str(env_file), '--slug', SLUG, '--title', 'product-matching-attribute-boosting-v2', '--dataset', f'{owner}/e-cup-human-data', '--dataset', str(manifest['dataset']), '--kernel', f'{owner}/product-matching-qwen-embedding-boosting', '--no-env-sources']
    if args.dry_run: command.append('--dry-run')
    elif not args.wait: command.append('--no-wait')
    print('$', ' '.join(command), flush=True); subprocess.run(command, check=True, cwd=ROOT)


if __name__ == '__main__': main()
