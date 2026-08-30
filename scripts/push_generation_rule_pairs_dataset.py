#!/usr/bin/env python3
"""Build and upload the private 10k generation-rule pair Dataset to Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from item_pipeline.pair_validation import validate_pair_dataset
from src.data_pipeline import serialize_product

import push_kaggle_training_dataset as shared_push
import run_kaggle_notebook as kaggle
from freeze_generated_pair_dataset import (
    attempt_diversity_provenance,
    source_generation_provenance as raw_generation_provenance,
)


DEFAULT_DATASET_SLUG = "product-matching-generation-rule-pairs-10k-v2"
SOURCE_DIR = ROOT / "item_pipeline" / "artifacts" / "rule_first_pairs"


def default_artifact_tag(pair_count: int) -> str:
    return "10k" if pair_count == 10_000 else str(pair_count)


def artifact_filenames(tag: str) -> dict[str, str]:
    if not tag or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in tag):
        raise ValueError(f"invalid artifact tag: {tag!r}")
    return {
        "pairs": f"generation_rule_pairs_{tag}.parquet",
        "items": f"generation_rule_items_{tag}.parquet",
        "metadata": f"generation_rule_pair_metadata_{tag}.parquet",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload validated generation-rule pairs as a private Kaggle Dataset"
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--stage-dir", type=Path)
    parser.add_argument("--expected-pairs", type=int, default=10_000)
    parser.add_argument("--dataset-slug", default=DEFAULT_DATASET_SLUG)
    parser.add_argument(
        "--artifact-tag",
        help="filename suffix; defaults to 10k for 10,000 pairs, otherwise the exact count",
    )
    parser.add_argument(
        "--allow-checkpoint",
        action="store_true",
        help="publish the current structurally valid checkpoint without requiring a completed generation summary",
    )
    parser.add_argument(
        "--message", default="Add 10k validated rule-first generation pairs v2"
    )
    parser.add_argument(
        "--label-source", default="qwen_rule_first_generation_v2"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, destination)


def copy_file(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def source_summary_value(summary: dict[str, object], key: str) -> object:
    source_key = f"source_{key}"
    return summary[source_key] if source_key in summary else summary.get(key)


def source_generation_provenance(summary: dict[str, object]) -> dict[str, object]:
    raw_provenance = raw_generation_provenance(summary)
    nested = summary.get("source_generation_provenance")
    nested = nested if isinstance(nested, dict) else {}
    semantic = nested.get("semantic_signature")
    if not isinstance(semantic, dict):
        semantic = {
            key: source_summary_value(summary, key)
            for key in (
                "semantic_signature_retry",
                "semantic_signature_version",
                "semantic_signature_limit",
                "semantic_signature_unique_count",
                "semantic_signature_max_count",
                "semantic_signature_retry_events",
                "semantic_signature_retry_events_this_run",
            )
            if source_summary_value(summary, key) is not None
        }
    attempt_diversity = nested.get("attempt_diversity")
    if not isinstance(attempt_diversity, dict):
        attempt_diversity = raw_provenance.get("attempt_diversity")
    if not isinstance(attempt_diversity, dict) or not attempt_diversity:
        version = source_summary_value(summary, "attempt_diversity_version")
        attempt_diversity = (
            {"attempt_diversity_version": version}
            if version is not None
            else {}
        )
    schedule = nested.get("rule_schedule")
    if not isinstance(schedule, dict):
        schedule = summary.get("source_rule_schedule")
    if not isinstance(schedule, dict):
        schedule = raw_provenance["rule_schedule"]
    provenance: dict[str, object] = {
        "semantic_signature": semantic,
        "attempt_diversity": attempt_diversity,
        "rule_schedule": schedule if isinstance(schedule, dict) else {},
    }
    frozen_schedule = summary.get("frozen_rule_schedule")
    if isinstance(frozen_schedule, dict):
        provenance["frozen_rule_schedule"] = frozen_schedule
    frozen_semantic = summary.get("frozen_semantic_signature")
    if isinstance(frozen_semantic, dict):
        provenance["frozen_semantic_signature"] = frozen_semantic
    frozen_attempt_diversity = summary.get("frozen_attempt_diversity")
    if isinstance(frozen_attempt_diversity, dict):
        provenance["frozen_attempt_diversity"] = frozen_attempt_diversity
    return provenance


def require_complete_source(
    source_dir: Path,
    expected_pairs: int,
    *,
    allow_checkpoint: bool = False,
) -> dict[str, object]:
    required = {
        "items": source_dir / "items.parquet",
        "pairs": source_dir / "pairs.parquet",
        "metadata": source_dir / "pair_generation_metadata.parquet",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        kaggle.fail(f"generation-rule outputs are incomplete; missing={missing}")
    summary_path = source_dir / "summary.json"
    validation_path = source_dir / "validation_report.json"
    if not allow_checkpoint:
        if not summary_path.is_file() or not validation_path.is_file():
            kaggle.fail(
                "generation-rule outputs are incomplete; missing completed summary or validation report"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        generated = summary.get("generated_pairs", summary.get("generated", -1))
        if int(summary.get("pending", -1)) != 0 or int(generated) != expected_pairs:
            kaggle.fail(f"generation summary is not complete for {expected_pairs} pairs: {summary}")
        if validation.get("valid") is not True or int(validation.get("pairs", -1)) != expected_pairs:
            kaggle.fail(f"pair validation is not valid for {expected_pairs} pairs: {validation}")
    else:
        summary = {
            "status": "frozen_checkpoint",
            "generated": expected_pairs,
            "published_pairs": expected_pairs,
        }
    fresh = validate_pair_dataset(
        required["items"], required["pairs"], metadata_path=required["metadata"]
    )
    if fresh.get("valid") is not True or int(fresh.get("pairs", -1)) != expected_pairs:
        kaggle.fail(f"fresh pair validation failed: {fresh}")
    return {"paths": required, "summary": summary, "validation": fresh}


def build_payload(
    source_dir: Path,
    stage_dir: Path,
    owner: str,
    expected_pairs: int,
    *,
    dataset_slug: str = DEFAULT_DATASET_SLUG,
    artifact_tag: str | None = None,
    allow_checkpoint: bool = False,
    label_source: str = "qwen_rule_first_generation_v2",
) -> dict[str, object]:
    if not label_source or any(character.isspace() for character in label_source):
        raise ValueError("label_source must be one non-empty token")
    checked = require_complete_source(
        source_dir,
        expected_pairs,
        allow_checkpoint=allow_checkpoint,
    )
    paths = checked["paths"]
    assert isinstance(paths, dict)
    filenames = artifact_filenames(artifact_tag or default_artifact_tag(expected_pairs))
    pairs = pd.read_parquet(paths["pairs"], columns=["id1", "id2", "target"])
    items = pd.read_parquet(
        paths["items"], columns=["id", "name", "attributes", "category"]
    )
    metadata = pd.read_parquet(paths["metadata"])

    attempt_summary = {
        "attempt_diversity_version": source_summary_value(
            checked["summary"], "attempt_diversity_version"
        ),
        "seed": source_summary_value(checked["summary"], "seed"),
    }
    try:
        verified_attempt_diversity = attempt_diversity_provenance(
            metadata, attempt_summary
        )
    except ValueError as error:
        kaggle.fail(f"attempt-diversity provenance verification failed: {error}")
    frozen_attempt_diversity = checked["summary"].get(
        "frozen_attempt_diversity"
    )
    if (
        isinstance(frozen_attempt_diversity, dict)
        and frozen_attempt_diversity != verified_attempt_diversity
    ):
        kaggle.fail(
            "frozen attempt-diversity provenance differs from staged metadata"
        )

    if len(pairs) != expected_pairs or len(items) != expected_pairs * 2:
        kaggle.fail(
            f"unexpected source dimensions: pairs={len(pairs)}, items={len(items)}"
        )
    if not pairs["target"].eq(0).all():
        kaggle.fail("generation-rule v1 must contain only negative targets")
    pair_ids = set(pairs["id1"]) | set(pairs["id2"])
    if pair_ids != set(items["id"]):
        kaggle.fail("source item catalogue does not exactly match pair ids")
    pairs = pairs.copy()
    pairs["label_source"] = label_source
    items = items.copy()
    items["product_text"] = items.apply(serialize_product, axis=1)

    stage_dir.mkdir(parents=True, exist_ok=True)
    pair_path = stage_dir / filenames["pairs"]
    item_path = stage_dir / filenames["items"]
    metadata_path = stage_dir / filenames["metadata"]
    atomic_parquet(pairs[["id1", "id2", "target", "label_source"]], pair_path)
    atomic_parquet(items[["id", "name", "category", "product_text"]], item_path)
    atomic_parquet(metadata, metadata_path)
    (stage_dir / "generation_summary.json").write_text(
        json.dumps(checked["summary"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (stage_dir / "validation_report.json").write_text(
        json.dumps(checked["validation"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    freeze_manifest_source = source_dir / "freeze_manifest.json"
    if freeze_manifest_source.is_file():
        copy_file(freeze_manifest_source, stage_dir / "freeze_manifest.json")

    staged_names = [
        filenames["pairs"],
        filenames["items"],
        filenames["metadata"],
        "generation_summary.json",
        "validation_report.json",
    ]
    if freeze_manifest_source.is_file():
        staged_names.append("freeze_manifest.json")
    files = {
        name: {
            "bytes": (stage_dir / name).stat().st_size,
            "sha256": sha256_file(stage_dir / name),
        }
        for name in staged_names
    }
    dataset_ref = f"{owner}/{dataset_slug}"
    checked_summary = checked["summary"]
    assert isinstance(checked_summary, dict)
    generation_provenance = source_generation_provenance(checked_summary)
    upload_manifest = {
        "schema_version": 1,
        "dataset": dataset_ref,
        "is_private": True,
        "pairs": expected_pairs,
        "items": expected_pairs * 2,
        "checkpoint": allow_checkpoint,
        "label_source": label_source,
        "targets": {"0": expected_pairs},
        "source_provenance": {
            "run_signature": source_summary_value(
                checked_summary, "run_signature"
            ),
            "model": source_summary_value(checked_summary, "model"),
            "structured_output": source_summary_value(
                checked_summary, "structured_output"
            ),
            "prompt_sha256": source_summary_value(
                checked_summary, "prompt_sha256"
            ),
            "rule_catalogs": source_summary_value(
                checked_summary, "rule_catalogs"
            ),
            "rule_tiers": source_summary_value(checked_summary, "rule_tiers"),
            **generation_provenance,
        },
        "files": files,
    }
    (stage_dir / "upload_manifest.json").write_text(
        json.dumps(upload_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata_json = {
        "title": f"Rule-First Product Pairs {expected_pairs} v2",
        "id": dataset_ref,
        "licenses": [{"name": "unknown"}],
        "isPrivate": True,
        "description": (
            f"Private E-CUP 2026 training data: {expected_pairs:,} Qwen-generated negative "
            "pairs. Each anchor is source-scoped to an observed product type, then "
            "mutated with exact attribute and literal title-substitution checks."
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
    if expected_names != actual_names:
        kaggle.fail(
            f"staged file set mismatch; expected={sorted(expected_names)}, "
            f"actual={sorted(actual_names)}"
        )
    return upload_manifest


def verify_remote_dataset(
    cli: list[str], dataset_ref: str, expected_files: set[str]
) -> None:
    files = kaggle.run_command(
        cli
        + ["datasets", "files", dataset_ref, "--format", "json", "--page-size", "200"]
    )
    missing = sorted(name for name in expected_files if name not in files.stdout)
    if missing:
        kaggle.fail(f"remote Dataset is missing files: {missing}", 1)
    with tempfile.TemporaryDirectory(prefix="kaggle-generation-rule-metadata-") as temporary:
        destination = Path(temporary)
        kaggle.run_command(
            cli + ["datasets", "metadata", dataset_ref, "--path", str(destination)]
        )
        remote = json.loads(
            (destination / "dataset-metadata.json").read_text(encoding="utf-8")
        )
        if remote.get("info", remote).get("isPrivate") is not True:
            kaggle.fail("Kaggle reports that the generation-rule Dataset is not private", 1)


def main() -> None:
    args = parse_args()
    env_file = args.env_file if args.env_file.is_absolute() else ROOT / args.env_file
    kaggle.load_dotenv(env_file)
    owner = kaggle.validate_slug(os.getenv("KAGGLE_USERNAME", "").strip(), "KAGGLE_USERNAME")
    if not args.dry_run and not os.getenv("KAGGLE_API_TOKEN", "").strip():
        kaggle.fail("set KAGGLE_API_TOKEN in .env")
    source_dir = args.source_dir if args.source_dir.is_absolute() else ROOT / args.source_dir
    if args.stage_dir is None:
        stage_dir = ROOT / ".kaggle" / "datasets" / args.dataset_slug
    else:
        stage_dir = args.stage_dir if args.stage_dir.is_absolute() else ROOT / args.stage_dir
    manifest = build_payload(
        source_dir.resolve(),
        stage_dir.resolve(),
        owner,
        args.expected_pairs,
        dataset_slug=args.dataset_slug,
        artifact_tag=args.artifact_tag,
        allow_checkpoint=args.allow_checkpoint,
        label_source=args.label_source,
    )
    dataset_ref = str(manifest["dataset"])
    expected_files = set(manifest["files"]) | {"upload_manifest.json"}
    print(f"Prepared private Kaggle Dataset payload: {stage_dir}")
    print(f"Dataset reference: {dataset_ref}")
    print(f"Pairs: {args.expected_pairs:,}; files: {len(expected_files)}")
    if args.dry_run:
        print("Dry run complete; Kaggle was not contacted.")
        return

    cli = kaggle.kaggle_command()
    previous_status = shared_push.dataset_status(cli, dataset_ref)
    previous_version = int(previous_status.get("current_version_number", 0)) if previous_status else 0
    if previous_status is None:
        command = cli + ["datasets", "create", "--path", str(stage_dir), "--keep-tabular"]
    else:
        command = cli + [
            "datasets",
            "version",
            "--path",
            str(stage_dir),
            "--message",
            args.message,
            "--keep-tabular",
        ]
    kaggle.run_command(command)
    status = shared_push.wait_until_ready(cli, dataset_ref, minimum_version=previous_version + 1)
    verify_remote_dataset(cli, dataset_ref, expected_files)
    print(
        f"Private Dataset ready at version {status.get('current_version_number')}: "
        f"https://www.kaggle.com/datasets/{dataset_ref}"
    )


if __name__ == "__main__":
    main()
