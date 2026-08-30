#!/usr/bin/env python3
"""Upload the composed soft-positive A+B data and run the frozen MiniLM ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scripts.compose_soft_positive_tier_ab_dataset import (
    A_COUNT,
    B_COUNT,
    SEMANTIC_SIGNATURE_LIMIT,
    TOTAL_COUNT,
    VERSION,
    canonical_card,
    compose,
)
from item_pipeline.pair_validation import validate_pair_dataset
from src.data_pipeline import serialize_product

import create_mixed_generation_rule_10k_notebook as notebook_builder
import push_generation_rule_pairs_dataset as upload_helpers
import push_kaggle_training_dataset as shared_push
import run_kaggle_notebook as kaggle


SOURCE_DIR = ROOT / "item_pipeline/artifacts/soft_positive_tier_ab_16351_qwen_v1_composed"
A_SOURCE_DIR = ROOT / "item_pipeline/artifacts/soft_positive_tier_a_qwen_v1_frozen"
B_SOURCE_DIR = ROOT / "item_pipeline/artifacts/soft_positive_tier_b_27930_qwen_v1_raw"
DATASET_SLUG = "pm-soft-positive-tier-ab-qwen-16351-20260828"
ARTIFACT_TAG = "soft-positive-tier-ab-qwen-v1"
LABEL_SOURCE = "qwen_soft_positive_tier_ab_generation_v1"
EXPERIMENT = "minilm_5ep_soft_positive_tier_ab_qwen_v1"
KERNEL_SLUG = "minilm-5ep-soft-positive-tier-a-b-qwen-v1"
TITLE = "MiniLM 5ep: soft-positive Tier A+B Qwen v1"
NOTEBOOK = ROOT / (
    "notebooks/minilm_5ep_team_ablation/"
    "minilm_5ep_soft_positive_tier_ab_qwen_v1_2xt4.ipynb"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--dataset-slug", default=DATASET_SLUG)
    parser.add_argument("--artifact-tag", default=ARTIFACT_TAG)
    parser.add_argument("--label-source", default=LABEL_SOURCE)
    parser.add_argument("--experiment-label", default=EXPERIMENT)
    parser.add_argument("--kernel-slug", default=KERNEL_SLUG)
    parser.add_argument("--title", default=TITLE)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument(
        "--skip-dataset-upload",
        action="store_true",
        help="reuse the already uploaded private Dataset after a verified upload",
    )
    parser.add_argument(
        "--monitor-existing",
        action="store_true",
        help="wait for and download the already pushed kernel without another push",
    )
    parser.add_argument(
        "--reuse-composed",
        action="store_true",
        help="verify and reuse the immutable composed snapshot instead of rebuilding it",
    )
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing {description}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must be a JSON object")
    return value


def run(command: list[str]) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def verify_source(source_dir: Path) -> dict[str, Any]:
    summary = read_json(source_dir / "summary.json", "composition summary")
    manifest = read_json(
        source_dir / "composition_manifest.json", "composition manifest"
    )
    validation = read_json(source_dir / "validation_report.json", "validation")
    if (
        summary.get("version") != VERSION
        or summary.get("status") != "complete"
        or int(summary.get("generated_pairs", -1)) != TOTAL_COUNT
        or int(summary.get("pending", -1)) != 0
        or summary.get("target_counts") != {"0": 0, "1": TOTAL_COUNT}
        or manifest.get("version") != VERSION
        or int(manifest.get("pairs", -1)) != TOTAL_COUNT
        or manifest.get("targets") != {"0": 0, "1": TOTAL_COUNT}
        or int(manifest.get("globally_unique_cards", -1)) != TOTAL_COUNT * 2
        or validation.get("valid") is not True
        or int(validation.get("pairs", -1)) != TOTAL_COUNT
    ):
        raise RuntimeError("composed soft-positive A+B contract differs")
    files = manifest.get("files") or {}
    for name, expected in files.items():
        path = source_dir / name
        if (
            not path.is_file()
            or int(expected.get("bytes", -1)) != path.stat().st_size
            or expected.get("sha256") != sha256_file(path)
        ):
            raise RuntimeError(f"composed source file differs from manifest: {name}")
    pairs = pd.read_parquet(source_dir / "pairs.parquet")
    items = pd.read_parquet(source_dir / "items.parquet")
    metadata = pd.read_parquet(source_dir / "pair_generation_metadata.parquet")
    if (
        len(pairs) != TOTAL_COUNT
        or len(items) != TOTAL_COUNT * 2
        or len(metadata) != TOTAL_COUNT
        or not pairs["target"].eq(1).all()
        or not metadata["target"].eq(1).all()
        or metadata["component"].value_counts().to_dict()
        != {"tier_b": B_COUNT, "tier_a": A_COUNT}
        or items["id"].duplicated().any()
        or set(items["id"]) != set(pairs["id1"]) | set(pairs["id2"])
    ):
        raise RuntimeError("composed source dimensions, components, labels or IDs differ")
    card_keys = [canonical_card(row) for _, row in items.iterrows()]
    if len(card_keys) != len(set(card_keys)):
        raise RuntimeError("composed source contains category-agnostic duplicate cards")
    signature_counts = metadata["semantic_signature"].astype(str).value_counts()
    if int(signature_counts.max()) > SEMANTIC_SIGNATURE_LIMIT:
        raise RuntimeError("composed source exceeds the semantic signature cap")
    fresh = validate_pair_dataset(
        source_dir / "items.parquet",
        source_dir / "pairs.parquet",
        metadata_path=source_dir / "pair_generation_metadata.parquet",
    )
    if fresh.get("valid") is not True or int(fresh.get("pairs", -1)) != TOTAL_COUNT:
        raise RuntimeError(f"fresh composed-source validation failed: {fresh}")
    return {
        "summary": summary,
        "manifest": manifest,
        "validation": fresh,
        "pairs": pairs,
        "items": items,
        "metadata": metadata,
        "semantic_signature_unique_count": int(signature_counts.size),
        "semantic_signature_max_count": int(signature_counts.max()),
    }


def prepare_upload_payload(
    checked: dict[str, Any],
    *,
    stage_dir: Path,
    owner: str,
    dataset_slug: str,
    artifact_tag: str,
    label_source: str,
) -> dict[str, Any]:
    filenames = upload_helpers.artifact_filenames(artifact_tag)
    pairs = checked["pairs"].copy()
    pairs["target"] = pairs["target"].astype("int8")
    pairs["label_source"] = label_source
    items = checked["items"].copy()
    items["product_text"] = items.apply(serialize_product, axis=1)
    metadata = checked["metadata"]
    stage_dir.mkdir(parents=True, exist_ok=True)
    pair_path = stage_dir / filenames["pairs"]
    item_path = stage_dir / filenames["items"]
    metadata_path = stage_dir / filenames["metadata"]
    upload_helpers.atomic_parquet(
        pairs[["id1", "id2", "target", "label_source"]], pair_path
    )
    upload_helpers.atomic_parquet(
        items[["id", "name", "category", "product_text"]], item_path
    )
    upload_helpers.atomic_parquet(metadata, metadata_path)
    copied = {
        "generation_summary.json": SOURCE_DIR / "summary.json",
        "validation_report.json": SOURCE_DIR / "validation_report.json",
        "composition_manifest.json": SOURCE_DIR / "composition_manifest.json",
    }
    for destination_name, source in copied.items():
        upload_helpers.copy_file(source, stage_dir / destination_name)
    staged_names = [
        filenames["pairs"],
        filenames["items"],
        filenames["metadata"],
        *copied,
    ]
    files = {
        name: {
            "bytes": (stage_dir / name).stat().st_size,
            "sha256": sha256_file(stage_dir / name),
        }
        for name in staged_names
    }
    dataset_ref = f"{owner}/{dataset_slug}"
    summary = checked["summary"]
    manifest = {
        "schema_version": 3,
        "dataset": dataset_ref,
        "is_private": True,
        "pairs": TOTAL_COUNT,
        "items": TOTAL_COUNT * 2,
        "label_source": label_source,
        "targets": {"0": 0, "1": TOTAL_COUNT},
        "source_provenance": {
            "composition_version": VERSION,
            "composition_run_signature": summary["run_signature"],
            "composition_manifest_sha256": sha256_file(
                SOURCE_DIR / "composition_manifest.json"
            ),
            "model": summary["model"],
            "api_base_url": summary["api_base_url"],
            "structured_output": summary["structured_output"],
            "temperature": summary["temperature"],
            "rule_catalogs": summary["rule_catalogs"],
            "source_runs": summary["composition"]["source_runs"],
            "selection": checked["manifest"]["selection"],
            "semantic_signature_limit": SEMANTIC_SIGNATURE_LIMIT,
            "semantic_signature_unique_count": checked[
                "semantic_signature_unique_count"
            ],
            "semantic_signature_max_count": checked[
                "semantic_signature_max_count"
            ],
        },
        "files": files,
    }
    upload_manifest_path = stage_dir / "upload_manifest.json"
    upload_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata_json = {
        "title": "Soft-positive Tier A+B Qwen pairs 16351 v1",
        "id": dataset_ref,
        "licenses": [{"name": "unknown"}],
        "isPrivate": True,
        "description": (
            "Private E-CUP 2026 ablation data: 6,351 Tier A and 10,000 Tier B "
            "Qwen-generated atomic target=1 pairs, globally reindexed and deduplicated."
        ),
    }
    (stage_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata_json, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    expected_names = set(staged_names) | {"upload_manifest.json"}
    actual_names = {
        path.name
        for path in stage_dir.iterdir()
        if path.is_file() and path.name != "dataset-metadata.json"
    }
    if actual_names != expected_names:
        raise RuntimeError(
            f"staged payload file set differs: {sorted(actual_names)} != {sorted(expected_names)}"
        )
    return manifest


def upload_dataset(stage_dir: Path, manifest: dict[str, Any]) -> None:
    cli = kaggle.kaggle_command()
    dataset_ref = str(manifest["dataset"])
    previous = shared_push.dataset_status(cli, dataset_ref)
    previous_version = int(previous.get("current_version_number", 0)) if previous else 0
    if previous is None:
        command = cli + ["datasets", "create", "--path", str(stage_dir), "--keep-tabular"]
    else:
        command = cli + [
            "datasets",
            "version",
            "--path",
            str(stage_dir),
            "--message",
            "Add composed Qwen soft-positive Tier A+B 16,351 pairs",
            "--keep-tabular",
        ]
    kaggle.run_command(command)
    shared_push.wait_until_ready(cli, dataset_ref, minimum_version=previous_version + 1)
    upload_helpers.verify_remote_dataset(
        cli, dataset_ref, set(manifest["files"]) | {"upload_manifest.json"}
    )


def monitor_existing_kernel(owner: str, slug: str) -> None:
    cli = kaggle.kaggle_command()
    kernel_ref = f"{owner}/{slug}"
    kaggle.wait_for_kernel(
        cli,
        kernel_ref,
        poll_interval=kaggle.env_int("KAGGLE_POLL_INTERVAL_SECONDS", 30, minimum=5),
        wait_timeout=kaggle.env_int("KAGGLE_WAIT_TIMEOUT_SECONDS", 45_000, minimum=60),
    )
    output_root = Path(
        os.getenv("KAGGLE_OUTPUT_DIR", "") or ROOT / "artifacts/kaggle"
    ).expanduser()
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_dir = output_root.resolve() / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    kaggle.run_command(
        cli
        + [
            "kernels",
            "output",
            kernel_ref,
            "-p",
            str(output_dir),
            "--force",
            "--page-size",
            "200",
        ]
    )


def write_report(experiment: str, result: dict[str, Any]) -> Path:
    path = ROOT / "reports" / f"{experiment}_launcher.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"created_at": datetime.now(UTC).isoformat(), **result},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    args = parse_args()
    source_dir = absolute(args.source_dir)
    env_file = absolute(args.env_file)
    notebook = absolute(args.notebook)
    if not args.reuse_composed:
        compose(A_SOURCE_DIR, B_SOURCE_DIR, source_dir, B_COUNT)
    checked = verify_source(source_dir)
    kaggle.load_dotenv(env_file)
    owner = kaggle.validate_slug(os.getenv("KAGGLE_USERNAME", "").strip(), "KAGGLE_USERNAME")
    dataset_ref = f"{owner}/{args.dataset_slug}"
    stage_dir = (ROOT / ".kaggle" / "datasets" / args.dataset_slug).resolve()
    manifest = prepare_upload_payload(
        checked,
        stage_dir=stage_dir,
        owner=owner,
        dataset_slug=args.dataset_slug,
        artifact_tag=args.artifact_tag,
        label_source=args.label_source,
    )
    if (
        not args.dry_run_only
        and not args.skip_dataset_upload
        and not args.monitor_existing
    ):
        upload_dataset(stage_dir, manifest)
    upload_manifest_path = stage_dir / "upload_manifest.json"
    upload_manifest_sha = sha256_file(upload_manifest_path)

    notes = (
        f"Frozen MiniLM 5ep human baseline plus {TOTAL_COUNT:,} Qwen-generated "
        f"soft-positive atomic pairs: {A_COUNT:,} Tier A and {B_COUNT:,} Tier B, "
        "all target=1 and unit sample weight. Tier B selection maximizes represented "
        "rules from the completed checkpoint before task-order filling. Cards are "
        "category-agnostically deduplicated, IDs globally reindexed, and identical "
        f"semantic transitions capped at {SEMANTIC_SIGNATURE_LIMIT}. These are soft "
        "statistical positives, not manually verified equivalences. Frozen checkpoint, "
        "recipe and IID/hard/OOD validation unchanged. This is a data+compute ablation. "
        f"Source dataset {dataset_ref}. Upload manifest SHA-256 {upload_manifest_sha}."
    )
    run(
        [
            sys.executable,
            "scripts/create_mixed_generation_rule_10k_notebook.py",
            "--pair-count",
            str(TOTAL_COUNT),
            "--expected-target0",
            "0",
            "--expected-target1",
            str(TOTAL_COUNT),
            "--artifact-tag",
            args.artifact_tag,
            "--output",
            str(notebook),
            "--experiment-label",
            args.experiment_label,
            "--dataset-ref",
            dataset_ref,
            "--upload-manifest-sha256",
            upload_manifest_sha,
            "--label-source",
            args.label_source,
            "--notes",
            notes,
        ]
    )
    notebook_command = [
        sys.executable,
        "scripts/run_kaggle_notebook.py",
        str(notebook),
        "--env-file",
        str(env_file),
        "--slug",
        args.kernel_slug,
        "--title",
        args.title,
        "--dataset",
        "alexproger23/product-matching-validation-splits-v1",
        "--dataset",
        "alexproger23/product-matching-minilm-llm-pretrain-5ep",
        "--dataset",
        "alexproger23/product-matching-minilm-5ep-significance-v1",
        "--dataset",
        dataset_ref,
        "--no-env-sources",
    ]
    run(notebook_command + ["--dry-run"])
    if args.dry_run_only:
        result = {
            "status": "dry_run_complete",
            "dataset_ref": dataset_ref,
            "pair_count": TOTAL_COUNT,
            "target_counts": {"0": 0, "1": TOTAL_COUNT},
            "upload_manifest_sha256": upload_manifest_sha,
            "composition_run_signature": checked["summary"]["run_signature"],
            "source_components": {"tier_a": A_COUNT, "tier_b": B_COUNT},
            "semantic_signature_unique_count": checked[
                "semantic_signature_unique_count"
            ],
            "semantic_signature_max_count": checked[
                "semantic_signature_max_count"
            ],
            "notebook": str(notebook),
            "kernel_slug": args.kernel_slug,
        }
    else:
        if args.monitor_existing:
            monitor_existing_kernel(owner, args.kernel_slug)
        else:
            run(notebook_command)
        output_root = Path(
            os.getenv("KAGGLE_OUTPUT_DIR", "") or ROOT / "artifacts/kaggle"
        ).expanduser()
        if not output_root.is_absolute():
            output_root = ROOT / output_root
        output_dir = output_root.resolve() / args.kernel_slug
        completion = read_json(output_dir / "notebook_completed.json", "completion marker")
        sync = read_json(output_dir / "google_sheets_sync.json", "Sheets sync marker")
        comparison = read_json(output_dir / "baseline_comparison.json", "baseline comparison")
        run_id = str(completion.get("run_id") or "")
        if (
            completion.get("status") != "complete"
            or completion.get("experiment") != args.experiment_label
            or not run_id
            or sync.get("status") != "synced"
            or sync.get("comparison_sheet") != "data_exps"
            or sync.get("run_id") != run_id
            or comparison.get("status") != "ready"
        ):
            raise RuntimeError("A+B Kaggle completion/significance/data_exps contract failed")
        label_counts = (completion.get("train_data") or {}).get("label_source_counts") or {}
        if int(label_counts.get(args.label_source, -1)) != TOTAL_COUNT:
            raise RuntimeError("Kaggle train data has the wrong A+B synthetic count")
        completion_notes = str(completion.get("notes") or "")
        if dataset_ref not in completion_notes or upload_manifest_sha not in completion_notes:
            raise RuntimeError("Kaggle completion does not pin the A+B Dataset")
        result = {
            "status": "complete",
            "run_id": run_id,
            "dataset_ref": dataset_ref,
            "pair_count": TOTAL_COUNT,
            "target_counts": {"0": 0, "1": TOTAL_COUNT},
            "upload_manifest_sha256": upload_manifest_sha,
            "composition_run_signature": checked["summary"]["run_signature"],
            "source_components": {"tier_a": A_COUNT, "tier_b": B_COUNT},
            "semantic_signature_unique_count": checked[
                "semantic_signature_unique_count"
            ],
            "semantic_signature_max_count": checked[
                "semantic_signature_max_count"
            ],
            "kernel_slug": args.kernel_slug,
            "baseline_comparison": comparison,
        }
    report = write_report(args.experiment_label, result)
    print(json.dumps({**result, "launcher_report": str(report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
