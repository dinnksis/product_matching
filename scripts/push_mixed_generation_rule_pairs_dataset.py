#!/usr/bin/env python3
"""Build and upload a private mixed-label generated-pair Dataset to Kaggle."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from src.data_pipeline import serialize_product

import push_kaggle_training_dataset as shared_push
import push_generation_rule_pairs_dataset as legacy_upload
import run_kaggle_notebook as kaggle


DEFAULT_PAIR_COUNT = 10_000
DEFAULT_TARGET0 = 9_954
DEFAULT_TARGET1 = 46
DEFAULT_DATASET_SLUG = (
    "product-matching-semantic-rule-pairs-transition-positive-10k-v1"
)
DEFAULT_SOURCE_DIR = (
    ROOT
    / "item_pipeline"
    / "artifacts"
    / "semantic_rule_pairs_transition_positive_10k"
)
DEFAULT_QUOTA_POLICY = "transition_positive_v4_full_capacity_v1"
DEFAULT_QUOTA_RATIONALE = (
    "Use the complete bounded positive-transition capacity exactly once: 46 "
    "target=1 pairs cover all 23 manually approved unordered value transitions "
    "twice across four context-constrained rules; 9,954 target=0 pairs retain "
    "broad negative-rule coverage without extrapolating positive changes."
)
FROZEN_OOD_CATEGORIES = {"Одежда", "Бытовая техника"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--stage-dir", type=Path)
    parser.add_argument("--expected-pairs", type=int, default=DEFAULT_PAIR_COUNT)
    parser.add_argument("--expected-target0", type=int, default=DEFAULT_TARGET0)
    parser.add_argument("--expected-target1", type=int, default=DEFAULT_TARGET1)
    parser.add_argument("--dataset-slug", default=DEFAULT_DATASET_SLUG)
    parser.add_argument(
        "--artifact-tag", default="semantic-transition-positive-10k-v1"
    )
    parser.add_argument(
        "--label-source",
        default="openrouter_semantic_transition_rule_generation_v1",
    )
    parser.add_argument(
        "--message",
        default="Add 10k transition-positive semantic-rule pairs v1",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def expected_target_counts(target0: int, target1: int, total: int) -> dict[str, int]:
    counts = {"0": int(target0), "1": int(target1)}
    if min(counts.values()) < 0 or int(total) < 1:
        raise ValueError("generated target counts must be non-negative and non-empty")
    if sum(counts.values()) != int(total):
        raise ValueError(
            f"target counts must sum to expected pairs: {counts} != {total}"
        )
    return counts


def observed_target_counts(frame: pd.DataFrame) -> dict[str, int]:
    numeric = pd.to_numeric(frame["target"], errors="coerce")
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        raise ValueError("generated pair targets must be binary integers")
    return {
        str(label): int(numeric.eq(label).sum())
        for label in (0, 1)
    }


def target_quota_provenance(
    counts: dict[str, int], total: int
) -> dict[str, object]:
    normalized = expected_target_counts(counts["0"], counts["1"], total)
    is_default = normalized == {"0": DEFAULT_TARGET0, "1": DEFAULT_TARGET1} and (
        int(total) == DEFAULT_PAIR_COUNT
    )
    return {
        "policy": DEFAULT_QUOTA_POLICY if is_default else "explicit_exact_counts",
        "counts": normalized,
        "label_one_fraction": normalized["1"] / int(total),
        "positive_rule_policy": (
            "four_context_constrained_transition_positive_v4_rules"
            if is_default
            else "caller_pinned_catalog"
        ),
        "rationale": (
            DEFAULT_QUOTA_RATIONALE
            if is_default
            else "Use the caller-supplied exact mixed-label counts as a pinned ablation."
        ),
    }


def build_payload(
    source_dir: Path,
    stage_dir: Path,
    owner: str,
    expected_pairs: int,
    target_counts: dict[str, int],
    *,
    dataset_slug: str,
    artifact_tag: str,
    label_source: str,
) -> dict[str, object]:
    if not label_source or any(character.isspace() for character in label_source):
        raise ValueError("label_source must be one non-empty token")
    checked = legacy_upload.require_complete_source(source_dir, expected_pairs)
    paths = checked["paths"]
    assert isinstance(paths, dict)
    summary = checked["summary"]
    assert isinstance(summary, dict)

    pairs = pd.read_parquet(paths["pairs"], columns=["id1", "id2", "target"])
    items = pd.read_parquet(
        paths["items"], columns=["id", "name", "attributes", "category"]
    )
    metadata = pd.read_parquet(paths["metadata"])
    attempt_summary = {
        "attempt_diversity_version": legacy_upload.source_summary_value(
            summary, "attempt_diversity_version"
        ),
        "seed": legacy_upload.source_summary_value(summary, "seed"),
    }
    try:
        verified_attempt_diversity = legacy_upload.attempt_diversity_provenance(
            metadata, attempt_summary
        )
    except ValueError as error:
        kaggle.fail(f"attempt-diversity provenance verification failed: {error}")
    frozen_attempt_diversity = summary.get("frozen_attempt_diversity")
    if (
        isinstance(frozen_attempt_diversity, dict)
        and frozen_attempt_diversity != verified_attempt_diversity
    ):
        kaggle.fail(
            "frozen attempt-diversity provenance differs from staged metadata"
        )
    actual_counts = observed_target_counts(pairs)
    if actual_counts != target_counts:
        kaggle.fail(
            f"generated target counts differ: {actual_counts} != {target_counts}"
        )
    if "target" not in metadata:
        kaggle.fail("generation metadata has no target column")
    metadata_counts = observed_target_counts(metadata)
    if metadata_counts != target_counts:
        kaggle.fail(
            f"metadata target counts differ: {metadata_counts} != {target_counts}"
        )
    if len(pairs) != expected_pairs or len(items) != expected_pairs * 2:
        kaggle.fail(
            f"unexpected source dimensions: pairs={len(pairs)}, items={len(items)}"
        )
    pair_ids = set(pairs["id1"]) | set(pairs["id2"])
    if pair_ids != set(items["id"]):
        kaggle.fail("source item catalogue does not exactly match pair ids")
    forbidden_categories = set(items["category"].astype(str)) & FROZEN_OOD_CATEGORIES
    if forbidden_categories:
        kaggle.fail(
            "mixed train data uses frozen OOD categories: "
            f"{sorted(forbidden_categories)}"
        )

    filenames = legacy_upload.artifact_filenames(artifact_tag)
    pairs = pairs.copy()
    pairs["target"] = pairs["target"].astype("int8")
    pairs["label_source"] = label_source
    items = items.copy()
    items["product_text"] = items.apply(serialize_product, axis=1)

    stage_dir.mkdir(parents=True, exist_ok=True)
    pair_path = stage_dir / filenames["pairs"]
    item_path = stage_dir / filenames["items"]
    metadata_path = stage_dir / filenames["metadata"]
    legacy_upload.atomic_parquet(
        pairs[["id1", "id2", "target", "label_source"]], pair_path
    )
    legacy_upload.atomic_parquet(
        items[["id", "name", "category", "product_text"]], item_path
    )
    legacy_upload.atomic_parquet(metadata, metadata_path)
    (stage_dir / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation = checked["validation"]
    (stage_dir / "validation_report.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    freeze_manifest_source = source_dir / "freeze_manifest.json"
    if freeze_manifest_source.is_file():
        legacy_upload.copy_file(
            freeze_manifest_source, stage_dir / "freeze_manifest.json"
        )

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
            "sha256": legacy_upload.sha256_file(stage_dir / name),
        }
        for name in staged_names
    }
    dataset_ref = f"{owner}/{dataset_slug}"
    upload_manifest = {
        "schema_version": 2,
        "dataset": dataset_ref,
        "is_private": True,
        "pairs": expected_pairs,
        "items": expected_pairs * 2,
        "checkpoint": False,
        "label_source": label_source,
        "targets": target_counts,
        "target_quota": target_quota_provenance(target_counts, expected_pairs),
        "source_provenance": {
            "run_signature": legacy_upload.source_summary_value(
                summary, "run_signature"
            ),
            "model": legacy_upload.source_summary_value(summary, "model"),
            "api_base_url": legacy_upload.source_summary_value(
                summary, "api_base_url"
            ),
            "structured_output": legacy_upload.source_summary_value(
                summary, "structured_output"
            ),
            "reasoning_effort": legacy_upload.source_summary_value(
                summary, "reasoning_effort"
            ),
            "prompt_sha256": legacy_upload.source_summary_value(
                summary, "prompt_sha256"
            ),
            "rule_catalogs": legacy_upload.source_summary_value(
                summary, "rule_catalogs"
            ),
            "rule_tiers": legacy_upload.source_summary_value(
                summary, "rule_tiers"
            ),
            "base_items_path": legacy_upload.source_summary_value(
                summary, "base_items_path"
            ),
            "base_items_sha256": legacy_upload.source_summary_value(
                summary, "base_items_sha256"
            ),
            "label_one_fraction": legacy_upload.source_summary_value(
                summary, "label_one_fraction"
            ),
            "planned_target_counts": legacy_upload.source_summary_value(
                summary, "planned_target_counts"
            ),
            "realized_target_counts": legacy_upload.source_summary_value(
                summary, "realized_target_counts"
            ),
            **legacy_upload.source_generation_provenance(summary),
        },
        "files": files,
    }
    (stage_dir / "upload_manifest.json").write_text(
        json.dumps(upload_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (stage_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": f"Semantic Rule Product Pairs Mixed {expected_pairs} v1",
                "id": dataset_ref,
                "licenses": [{"name": "unknown"}],
                "isPrivate": True,
                "description": (
                    "Private E-CUP 2026 training data: validated generated product "
                    f"pairs with exact target counts {target_counts}."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    expected_names = set(staged_names) | {"upload_manifest.json"}
    actual_names = {
        path.name
        for path in stage_dir.iterdir()
        if path.is_file() and path.name != "dataset-metadata.json"
    }
    if actual_names != expected_names:
        kaggle.fail(
            f"staged file set mismatch; expected={sorted(expected_names)}, "
            f"actual={sorted(actual_names)}"
        )
    return upload_manifest


def main() -> None:
    args = parse_args()
    counts = expected_target_counts(
        args.expected_target0, args.expected_target1, args.expected_pairs
    )
    env_file = args.env_file if args.env_file.is_absolute() else ROOT / args.env_file
    kaggle.load_dotenv(env_file)
    owner = kaggle.validate_slug(
        os.getenv("KAGGLE_USERNAME", "").strip(), "KAGGLE_USERNAME"
    )
    if not args.dry_run and not os.getenv("KAGGLE_API_TOKEN", "").strip():
        kaggle.fail("set KAGGLE_API_TOKEN in .env")
    source_dir = (
        args.source_dir if args.source_dir.is_absolute() else ROOT / args.source_dir
    ).resolve()
    stage_dir = (
        args.stage_dir
        if args.stage_dir is not None
        else ROOT / ".kaggle" / "datasets" / args.dataset_slug
    )
    if not stage_dir.is_absolute():
        stage_dir = ROOT / stage_dir
    manifest = build_payload(
        source_dir,
        stage_dir.resolve(),
        owner,
        args.expected_pairs,
        counts,
        dataset_slug=args.dataset_slug,
        artifact_tag=args.artifact_tag,
        label_source=args.label_source,
    )
    dataset_ref = str(manifest["dataset"])
    expected_files = set(manifest["files"]) | {"upload_manifest.json"}
    print(f"Prepared private mixed-label Kaggle Dataset payload: {stage_dir}")
    print(f"Dataset reference: {dataset_ref}; targets={counts}")
    if args.dry_run:
        print("Dry run complete; Kaggle was not contacted.")
        return

    cli = kaggle.kaggle_command()
    previous_status = shared_push.dataset_status(cli, dataset_ref)
    previous_version = (
        int(previous_status.get("current_version_number", 0))
        if previous_status
        else 0
    )
    if previous_status is None:
        command = cli + [
            "datasets",
            "create",
            "--path",
            str(stage_dir),
            "--keep-tabular",
        ]
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
    status = shared_push.wait_until_ready(
        cli, dataset_ref, minimum_version=previous_version + 1
    )
    legacy_upload.verify_remote_dataset(cli, dataset_ref, expected_files)
    print(
        f"Private Dataset ready at version {status.get('current_version_number')}: "
        f"https://www.kaggle.com/datasets/{dataset_ref}"
    )


if __name__ == "__main__":
    main()
