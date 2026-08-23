#!/usr/bin/env python3
"""Build the prepared balanced-data payload and mxbai xsmall Kaggle notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat as nbf

import create_cross_encoder_training_notebook as cross_builder
import create_qwen_training_notebook as shared


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = ROOT / "notebooks" / "mxbai_xsmall_balanced_training_2xt4.ipynb"
DEFAULT_CONFIG = ROOT / "configs" / "cross_encoder_mxbai_xsmall_balanced_llm.json"
DEFAULT_DATASET_DIR = ROOT / ".kaggle" / "datasets" / "product-matching-mxbai-balanced-training"
DATASET_SLUG = "product-matching-mxbai-balanced-training"
PREPARED_FILES = {
    Path("prepared/mxbai_balanced/items.parquet"): "mxbai_balanced_items.parquet",
    Path("prepared/mxbai_balanced/train_pairs.parquet"): "mxbai_balanced_train_pairs.parquet",
    Path("prepared/mxbai_balanced/val_pairs.parquet"): "mxbai_balanced_val_pairs.parquet",
    Path("prepared/mxbai_balanced/report.json"): "mxbai_balanced_report.json",
}


def build_dataset(dataset_dir: Path, owner: str) -> dict[str, object]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    prepared_manifest: dict[str, dict[str, object]] = {}
    for source_relative, destination_name in PREPARED_FILES.items():
        source = ROOT / source_relative
        if not source.is_file():
            raise FileNotFoundError(
                f"Prepared blended data is missing: {source}. Run "
                "scripts/prepare_balanced_llm_data.py first."
            )
        source_hash = shared.sha256(source)
        shared.copy_if_changed(source, dataset_dir / destination_name, source_hash)
        prepared_manifest[destination_name] = {
            "bytes": source.stat().st_size,
            "sha256": source_hash,
        }

    bundle_path = dataset_dir / shared.BUNDLE_NAME
    source_manifest = shared.write_code_bundle(bundle_path)
    manifest = {
        "schema_version": 1,
        "dataset": f"{owner}/{DATASET_SLUG}",
        "prepared_data": prepared_manifest,
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
        "title": "Product Matching mxbai Balanced Training",
        "id": f"{owner}/{DATASET_SLUG}",
        "licenses": [{"name": "unknown"}],
        "description": (
            "Private component-disjoint human validation plus an equal category-label "
            "training set supplemented with high-confidence LLM labels."
        ),
    }
    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_notebook(
    manifest: dict[str, object], training_config: dict[str, object]
) -> nbf.NotebookNode:
    notebook = cross_builder.build_notebook(manifest, training_config)
    bundle = manifest["code_bundle"]
    assert isinstance(bundle, dict)
    fingerprint = str(bundle["sha256"])
    dataset_reference = str(manifest["dataset"])

    notebook.cells[0].source = f"""# mxbai-rerank-xsmall-v1: balanced product matching на 2×Tesla T4

Full fine-tuning `mixedbread-ai/mxbai-rerank-xsmall-v1` на физически
сбалансированном наборе: одинаковый размер каждой категории и 50/50 классов.
Human validation остаётся чистой и component-disjoint; LLM-метки используются
только в train с меньшим весом.

Dataset: `{dataset_reference}`. Source fingerprint: `{fingerprint}`.
"""
    bootstrap = notebook.cells[1].source
    bootstrap = bootstrap.replace(
        'TEMP_ROOT = Path("/kaggle/temp/product_matching_minilm")',
        'TEMP_ROOT = Path("/kaggle/temp/product_matching_mxbai")',
    ).replace(
        'OUTPUT_DIR = WORKING_ROOT / "minilm_cross_encoder"',
        'OUTPUT_DIR = WORKING_ROOT / "mxbai_xsmall_balanced"',
    ).replace(
        'TRAIN_LOG = WORKING_ROOT / "minilm_training.log"',
        'TRAIN_LOG = WORKING_ROOT / "mxbai_xsmall_training.log"',
    )
    bootstrap = bootstrap.replace(
        'items_path = exactly_one("items_human.parquet")\n'
        'matches_path = exactly_one("matches.parquet")',
        'prepared_items_path = exactly_one("mxbai_balanced_items.parquet")\n'
        'prepared_train_path = exactly_one("mxbai_balanced_train_pairs.parquet")\n'
        'prepared_validation_path = exactly_one("mxbai_balanced_val_pairs.parquet")\n'
        'prepared_report_path = exactly_one("mxbai_balanced_report.json")',
    )
    bootstrap = bootstrap.replace(
        'print(f"items:   {items_path} ({items_path.stat().st_size / 2**20:.1f} MiB)")\n'
        'print(f"matches: {matches_path} ({matches_path.stat().st_size / 2**20:.1f} MiB)")',
        'print(f"items: {prepared_items_path} "\n'
        '      f"({prepared_items_path.stat().st_size / 2**20:.1f} MiB)")\n'
        'print(f"train: {prepared_train_path} "\n'
        '      f"({prepared_train_path.stat().st_size / 2**20:.1f} MiB)")\n'
        'print(f"validation: {prepared_validation_path} "\n'
        '      f"({prepared_validation_path.stat().st_size / 2**20:.1f} MiB)")',
    )
    if 'items_path = exactly_one("items_human.parquet")' in bootstrap:
        raise RuntimeError("Could not adapt cross-encoder bootstrap cell")
    notebook.cells[1].source = bootstrap
    notebook.cells[4].source = notebook.cells[4].source.replace(
        "configs/cross_encoder_minilm.json",
        "configs/cross_encoder_mxbai_xsmall_balanced_llm.json",
    )
    notebook.cells[5].source = """## Зависимости и готовый balanced split

Kaggle получает только компактный prepared dataset. Исходные 11,19 млн LLM-пар
и 13,4 млн карточек не загружаются. Validation полностью human и не пересекается
с train по item ID.
"""
    notebook.cells[6].source = """subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--disable-pip-version-check",
        "--upgrade-strategy",
        "only-if-needed",
        "-r",
        str(PROJECT_ROOT / "requirements-cross-encoder.txt"),
    ],
    check=True,
)
PREPARED_DIR.mkdir(parents=True, exist_ok=True)
prepared_sources = {
    prepared_items_path: PREPARED_DIR / "items.parquet",
    prepared_train_path: PREPARED_DIR / "train_pairs.parquet",
    prepared_validation_path: PREPARED_DIR / "val_pairs.parquet",
    prepared_report_path: PREPARED_DIR / "report.json",
}
for source, destination in prepared_sources.items():
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source)
prepared_report = json.loads(prepared_report_path.read_text(encoding="utf-8"))
print(json.dumps(prepared_report, ensure_ascii=False, indent=2))
"""
    notebook.cells[7].source = """## DDP full fine-tuning и финальная validation

DeBERTa-v2 cross-encoder получает пару `(item A, item B)` и один ranking logit.
На каждой эпохе порядок A/B случайный, validation усредняет обе ориентации.
BCE учитывает `sample_weight`: human-разметка приоритетнее LLM.
"""
    notebook.metadata["product_matching_training"].update(
        {
            "dataset": dataset_reference,
            "bundle_sha256": fingerprint,
            "experiment": "mxbai_xsmall_balanced_llm",
            "prepared_data": "physical category and label balance",
        }
    )
    return notebook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_NOTEBOOK)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    owner = shared.dotenv_username(args.env_file)
    if not owner:
        raise SystemExit("Set KAGGLE_USERNAME in .env")
    config = cross_builder.load_training_config(args.config)
    manifest = build_dataset(args.dataset_dir, owner)
    notebook = build_notebook(manifest, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, args.output)
    print(f"Wrote notebook: {args.output}")
    print(f"Prepared dataset payload: {args.dataset_dir}")
    print(f"Dataset reference: {manifest['dataset']}")
    print(f"Code bundle SHA-256: {manifest['code_bundle']['sha256']}")


if __name__ == "__main__":
    main()
