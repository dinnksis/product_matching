#!/usr/bin/env python3
"""Run the frozen MiniLM baseline ablation with completed soft-positive Tier A."""

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

from freeze_generated_pair_dataset import canonical_card, freeze
from item_pipeline.pair_generate import (
    ATTEMPT_DIVERSITY_VERSION,
    SEMANTIC_SIGNATURE_VERSION,
)
from item_pipeline.pair_rules import load_mutation_rules
from item_pipeline.pair_validation import validate_pair_dataset
from item_pipeline.rule_schedule import SCHEDULE_VERSION, build_balanced_rule_schedule

import launch_statistical_rule_kaggle_experiment as strict_checks
import push_mixed_generation_rule_pairs_dataset as generated_upload
import run_kaggle_notebook as kaggle


RAW_COUNT = 6_500
RULE_COUNT = 325
EXAMPLES_PER_RULE = 20
EXPECTED_MODEL = "qwen3.5-397b-a17b-fp8"
EXPECTED_API_BASE_URL = "http://0.0.0.0:8994/v1"
EXPECTED_SEED = 20260828
EXPECTED_SEMANTIC_SIGNATURE_LIMIT = 5
EXPECTED_CATALOG_SHA256 = (
    "44d2f623958cabc6c61fdf6c837faab8bbcffd77e0bfd5ed5a96f7eaa951fa7f"
)
EXPECTED_DONOR_SHA256 = (
    "8421c606742d34ef32f3241d6711e1dd412cb8853c2fc16211d91e6cc87fcc14"
)
EXPECTED_DONOR_SOURCE_SHA256 = (
    "54672a0241b9586563812246be77b24f976a253a9f4e732d65d2484496a13883"
)
RAW_DIR = ROOT / "item_pipeline/artifacts/soft_positive_tier_a_6500_qwen_v1_raw"
FROZEN_DIR = ROOT / "item_pipeline/artifacts/soft_positive_tier_a_qwen_v1_frozen"
CATALOG = ROOT / (
    "configs/generation_rule_catalog_statistical_v1/soft_positive_ab_v1/tier_a.json"
)
CATALOG_MANIFEST = CATALOG.parent / "manifest.json"
DONORS = ROOT / (
    "item_pipeline/artifacts/generated_style_donors_x6_soft_positive_v1/items.parquet"
)
PROMPT = ROOT / "item_pipeline/prompts/mutate_item_by_rules.md"
NOTEBOOK = ROOT / (
    "notebooks/minilm_5ep_team_ablation/"
    "minilm_5ep_soft_positive_tier_a_qwen_v1_2xt4.ipynb"
)
DATASET_SLUG = "pm-soft-positive-tier-a-qwen-20260827"
ARTIFACT_TAG = "soft-positive-tier-a-qwen-v1"
EXPERIMENT = "minilm_5ep_soft_positive_tier_a_qwen_v1"
KERNEL_SLUG = "minilm-5ep-soft-positive-tier-a-qwen-v1"
TITLE = "MiniLM 5ep: soft-positive Tier A Qwen v1"
LABEL_SOURCE = "qwen_soft_positive_tier_a_generation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--frozen-dir", type=Path, default=FROZEN_DIR)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--minimum-raw-completion", type=float, default=0.95)
    parser.add_argument("--minimum-retention", type=float, default=0.98)
    parser.add_argument("--dataset-slug", default=DATASET_SLUG)
    parser.add_argument("--artifact-tag", default=ARTIFACT_TAG)
    parser.add_argument("--experiment-label", default=EXPERIMENT)
    parser.add_argument("--kernel-slug", default=KERNEL_SLUG)
    parser.add_argument("--title", default=TITLE)
    parser.add_argument("--label-source", default=LABEL_SOURCE)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    parser.add_argument(
        "--dry-run-only",
        action="store_true",
        help="freeze and stage locally without changing Kaggle",
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
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {description}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must be an object: {path}")
    return value


def run(command: list[str]) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def expected_target_counts(count: int) -> dict[str, int]:
    return {"0": 0, "1": int(count)}


def observed_target_counts(frame: pd.DataFrame) -> dict[str, int]:
    return generated_upload.observed_target_counts(frame)


def verify_catalog() -> tuple[list[Any], dict[str, Any]]:
    if sha256_file(CATALOG) != EXPECTED_CATALOG_SHA256:
        raise RuntimeError("Tier A catalog SHA-256 differs from the pinned catalog")
    manifest = read_json(CATALOG_MANIFEST, "soft-positive A/B catalog manifest")
    tier_entry = (manifest.get("catalogs") or {}).get("tier_a") or {}
    if (
        manifest.get("catalog_version")
        != "soft_semantic_label1_ab_singleton2_multi1_scoped_v1"
        or int(manifest.get("tier_a_rules", -1)) != RULE_COUNT
        or int(manifest.get("tier_a_examples_per_rule", -1)) != EXAMPLES_PER_RULE
        or int(manifest.get("planned_tier_a_pairs", -1)) != RAW_COUNT
        or tier_entry.get("sha256") != EXPECTED_CATALOG_SHA256
    ):
        raise RuntimeError("Tier A catalog manifest contract differs")
    rules = load_mutation_rules([CATALOG], labels={1})
    if len(rules) != RULE_COUNT or {rule.label for rule in rules} != {1}:
        raise RuntimeError("Tier A executable catalog must contain 325 label=1 rules")
    if any(
        len(rule.allowed_categories) != 1
        or len(rule.allowed_product_types) != 1
        for rule in rules
    ):
        raise RuntimeError("every Tier A rule must have one category/product scope")
    raw_rules = json.loads(CATALOG.read_text(encoding="utf-8"))
    if not isinstance(raw_rules, list) or any(
        int(row.get("generation_examples_per_rule", -1)) != EXAMPLES_PER_RULE
        or int(row.get("label", -1)) != 1
        for row in raw_rules
    ):
        raise RuntimeError("Tier A generation quota or labels differ")
    return rules, manifest


def verify_donors(summary: dict[str, Any]) -> pd.DataFrame:
    if sha256_file(DONORS) != EXPECTED_DONOR_SHA256:
        raise RuntimeError("Tier A donor pool SHA-256 differs")
    if summary.get("base_items_path") != str(DONORS):
        raise RuntimeError("raw summary points to a different donor pool")
    if summary.get("base_items_sha256") != EXPECTED_DONOR_SHA256:
        raise RuntimeError("raw summary donor SHA-256 differs")
    manifest = read_json(DONORS.parent / "manifest.json", "x6 donor manifest")
    if (
        manifest.get("version") != "generated_style_donors_virtual_id_copies_v1"
        or manifest.get("output_sha256") != EXPECTED_DONOR_SHA256
        or manifest.get("source_sha256") != EXPECTED_DONOR_SOURCE_SHA256
        or int(manifest.get("copies", -1)) != 6
        or int(manifest.get("rows", -1)) != 60_000
        or int(manifest.get("unique_ids", -1)) != 60_000
    ):
        raise RuntimeError("x6 Tier A donor manifest differs")
    donors = pd.read_parquet(DONORS)
    if len(donors) != 60_000 or donors["id"].duplicated().any():
        raise RuntimeError("Tier A donor pool dimensions differ")
    if set(donors["category"].astype(str)) & {"Одежда", "Бытовая техника"}:
        raise RuntimeError("Tier A donors leak frozen OOD categories")
    return donors


def require_ready_raw(raw_dir: Path) -> dict[str, Any]:
    paths = {
        "items": raw_dir / "items.parquet",
        "pairs": raw_dir / "pairs.parquet",
        "metadata": raw_dir / "pair_generation_metadata.parquet",
        "summary": raw_dir / "summary.json",
        "validation": raw_dir / "validation_report.json",
        "errors": raw_dir / "errors.json",
    }
    if missing := [str(path) for path in paths.values() if not path.is_file()]:
        raise RuntimeError(f"Tier A raw generation is incomplete: {missing}")
    summary = read_json(paths["summary"], "Tier A generation summary")
    validation = read_json(paths["validation"], "Tier A validation report")
    errors = json.loads(paths["errors"].read_text(encoding="utf-8"))
    generated = int(summary.get("generated_pairs", -1))
    pending = int(summary.get("pending", -1))
    if (
        int(summary.get("count", -1)) != RAW_COUNT
        or generated < 1
        or pending < 0
        or generated + pending != RAW_COUNT
        or int(summary.get("errors", -1)) != pending
        or not isinstance(errors, list)
        or len(errors) != pending
    ):
        raise RuntimeError("Tier A generated/pending/error accounting differs")
    if validation.get("valid") is not True or int(validation.get("pairs", -1)) != generated:
        raise RuntimeError("persisted Tier A validation is not complete")
    if (
        summary.get("model") != EXPECTED_MODEL
        or str(summary.get("api_base_url") or "").rstrip("/")
        != EXPECTED_API_BASE_URL.rstrip("/")
        or summary.get("structured_output") is not False
        or summary.get("reasoning_effort") is not None
        or float(summary.get("temperature", -1)) != 0.7
        or int(summary.get("max_tokens", -1)) != 1_400
        or int(summary.get("seed", -1)) != EXPECTED_SEED
        or float(summary.get("two_rule_fraction", -1)) != 0.0
        or float(summary.get("label_one_fraction", -1)) != 1.0
        or int(summary.get("semantic_signature_limit", -1))
        != EXPECTED_SEMANTIC_SIGNATURE_LIMIT
    ):
        raise RuntimeError("Tier A generation configuration differs from the pinned run")
    if summary.get("rule_catalogs") != [
        {"path": str(CATALOG), "sha256": EXPECTED_CATALOG_SHA256}
    ]:
        raise RuntimeError("Tier A raw summary does not pin the expected catalog")
    if summary.get("prompt_sha256") != hashlib.sha256(
        PROMPT.read_text(encoding="utf-8").strip().encode("utf-8")
    ).hexdigest():
        raise RuntimeError("Tier A prompt SHA-256 differs")

    rules, manifest = verify_catalog()
    donors = verify_donors(summary)
    rebuilt = build_balanced_rule_schedule(
        donors,
        rules,
        count=RAW_COUNT,
        seed=EXPECTED_SEED,
        two_rule_fraction=0.0,
        semantic_signature_limit=EXPECTED_SEMANTIC_SIGNATURE_LIMIT,
        label_one_fraction=1.0,
    )
    metadata = pd.read_parquet(paths["metadata"])
    pairs = pd.read_parquet(paths["pairs"])
    items = pd.read_parquet(paths["items"])
    if len(metadata) != generated or len(pairs) != generated or len(items) != generated * 2:
        raise RuntimeError("Tier A raw artifact dimensions differ")
    if observed_target_counts(pairs) != expected_target_counts(generated):
        raise RuntimeError("Tier A raw pairs are not all target=1")
    if observed_target_counts(metadata) != expected_target_counts(generated):
        raise RuntimeError("Tier A metadata is not all target=1")
    if not metadata["rule_count"].eq(1).all():
        raise RuntimeError("Tier A raw data contains non-atomic rule bundles")
    schedule = strict_checks.verify_rebuilt_balanced_rule_schedule_metadata(
        metadata, summary, RULE_COUNT, rebuilt
    )
    accepted_tasks = set(metadata["task_index"].astype(int))
    error_tasks: set[int] = set()
    for error in errors:
        if not isinstance(error, dict):
            raise RuntimeError("Tier A errors.json contains a non-object row")
        try:
            task_index = int(error["task_index"])
            source_id = int(error["source_id"])
        except (KeyError, TypeError, ValueError, OverflowError) as cause:
            raise RuntimeError("Tier A errors.json has invalid task provenance") from cause
        if task_index in error_tasks or task_index < 0 or task_index >= RAW_COUNT:
            raise RuntimeError("Tier A errors.json has duplicate/out-of-range tasks")
        scheduled = rebuilt.bundle_for_task(task_index)
        if source_id != scheduled.donor_id or str(error.get("category")) != scheduled.category:
            raise RuntimeError("Tier A error row differs from the rebuilt schedule")
        error_tasks.add(task_index)
    if accepted_tasks & error_tasks or accepted_tasks | error_tasks != set(range(RAW_COUNT)):
        raise RuntimeError("Tier A accepted/error tasks are not an exact schedule partition")
    planned_usage = summary.get("primary_rule_usage") or {}
    realized_usage = summary.get("realized_primary_rule_usage") or {}
    if (
        set(planned_usage) != {rule.generation_rule_id for rule in rules}
        or {int(value) for value in planned_usage.values()} != {EXAMPLES_PER_RULE}
        or set(realized_usage) != set(planned_usage)
        or any(
            int(realized_usage[rule_id]) < 1
            or int(realized_usage[rule_id]) > int(planned_usage[rule_id])
            for rule_id in planned_usage
        )
    ):
        raise RuntimeError("Tier A did not preserve complete bounded rule coverage")
    fresh_validation = validate_pair_dataset(
        paths["items"], paths["pairs"], metadata_path=paths["metadata"]
    )
    if fresh_validation.get("valid") is not True:
        raise RuntimeError(f"fresh Tier A validation failed: {fresh_validation}")
    signatures = strict_checks.verify_semantic_signature_metadata(
        metadata, summary, EXPECTED_SEMANTIC_SIGNATURE_LIMIT
    )
    attempts = strict_checks.verify_attempt_provenance_metadata(metadata, summary)
    diversity = strict_checks.verify_attempt_diversity_metadata(metadata, summary)
    return {
        "summary": summary,
        "generated_pairs": generated,
        "pending_pairs": pending,
        "paths": paths,
        "rules": rules,
        "manifest": manifest,
        "schedule": schedule,
        "schedule_sha256": rebuilt.schedule_sha256,
        "signatures": signatures,
        "attempts": attempts,
        "diversity": diversity,
        "fresh_validation": fresh_validation,
    }


def maximum_globally_unique_pair_count(raw_dir: Path) -> int:
    items = pd.read_parquet(raw_dir / "items.parquet")
    pairs = pd.read_parquet(raw_dir / "pairs.parquet")
    item_by_id = items.set_index("id", drop=False)
    card_keys = {
        int(row["id"]): canonical_card(row) for _, row in items.iterrows()
    }
    seen_cards: set[str] = set()
    seen_ids: set[int] = set()
    retained = 0
    for pair in pairs.itertuples(index=False):
        id1, id2 = int(pair.id1), int(pair.id2)
        if id1 not in item_by_id.index or id2 not in item_by_id.index:
            continue
        left, right = card_keys[id1], card_keys[id2]
        if (
            id1 in seen_ids
            or id2 in seen_ids
            or left == right
            or left in seen_cards
            or right in seen_cards
        ):
            continue
        retained += 1
        seen_ids.update((id1, id2))
        seen_cards.update((left, right))
    return retained


def verify_frozen(
    frozen: dict[str, Any],
    *,
    raw_dir: Path,
    frozen_dir: Path,
    pair_count: int,
    source_schedule_sha256: str,
) -> dict[str, Any]:
    summary = frozen.get("summary") or {}
    schedule = summary.get("frozen_rule_schedule") or {}
    pairs = pd.read_parquet(frozen_dir / "pairs.parquet")
    metadata = pd.read_parquet(frozen_dir / "pair_generation_metadata.parquet")
    if (
        len(pairs) != pair_count
        or len(metadata) != pair_count
        or observed_target_counts(pairs) != expected_target_counts(pair_count)
        or observed_target_counts(metadata) != expected_target_counts(pair_count)
    ):
        raise RuntimeError("frozen Tier A dimensions or labels differ")
    if (
        int(schedule.get("selected_task_count", -1)) != pair_count
        or schedule.get("source_rule_schedule_sha256") != source_schedule_sha256
        or int(schedule.get("primary_rule_coverage", -1)) != RULE_COUNT
        or int(schedule.get("primary_rule_profile_coverage", -1)) != RULE_COUNT
        or schedule.get("full_primary_rule_coverage") is not True
        or schedule.get("full_primary_rule_profile_coverage") is not True
        or schedule.get("primary_rule_profile_cap_violations") != {}
        or int(schedule.get("semantic_signature_limit", -1))
        != EXPECTED_SEMANTIC_SIGNATURE_LIMIT
        or int(schedule.get("semantic_signature_max_count", -1))
        > EXPECTED_SEMANTIC_SIGNATURE_LIMIT
        or float(schedule.get("two_rule_fraction", -1)) != 0.0
    ):
        raise RuntimeError("frozen Tier A schedule contract differs")
    attempt_diversity = strict_checks.verify_frozen_attempt_diversity(
        frozen, pair_count=pair_count
    )
    uniqueness = strict_checks.verify_frozen_global_card_uniqueness(
        frozen,
        raw_dir=raw_dir,
        frozen_dir=frozen_dir,
        pair_count=pair_count,
    )
    validation = read_json(frozen_dir / "validation_report.json", "frozen validation")
    if validation.get("valid") is not True or int(validation.get("pairs", -1)) != pair_count:
        raise RuntimeError("frozen Tier A validation differs")
    return {
        "pair_count": pair_count,
        "target_counts": expected_target_counts(pair_count),
        "schedule": schedule,
        "attempt_diversity": attempt_diversity,
        "global_card_uniqueness": uniqueness,
        "validation": validation,
    }


def write_report(experiment: str, result: dict[str, Any]) -> Path:
    path = ROOT / "reports" / f"{experiment}_launcher.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"created_at": datetime.now(UTC).isoformat(), **result},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def main() -> None:
    args = parse_args()
    if not 0.0 < args.minimum_raw_completion <= 1.0:
        raise ValueError("--minimum-raw-completion must be in (0, 1]")
    if not 0.0 < args.minimum_retention <= 1.0:
        raise ValueError("--minimum-retention must be in (0, 1]")
    raw_dir = absolute(args.raw_dir)
    frozen_dir = absolute(args.frozen_dir)
    notebook = absolute(args.notebook)
    env_file = absolute(args.env_file)
    raw = require_ready_raw(raw_dir)
    if raw["generated_pairs"] / RAW_COUNT < args.minimum_raw_completion:
        raise RuntimeError(
            f"only {raw['generated_pairs']}/{RAW_COUNT} Tier A tasks completed; "
            f"minimum raw completion={args.minimum_raw_completion:.3f}"
        )
    clean_count = maximum_globally_unique_pair_count(raw_dir)
    if clean_count / raw["generated_pairs"] < args.minimum_retention:
        raise RuntimeError(
            f"only {clean_count}/{raw['generated_pairs']} Tier A pairs survive global-card "
            f"deduplication; minimum retention={args.minimum_retention:.3f}"
        )
    frozen = freeze(raw_dir, frozen_dir, clean_count)
    frozen_checks = verify_frozen(
        frozen,
        raw_dir=raw_dir,
        frozen_dir=frozen_dir,
        pair_count=clean_count,
        source_schedule_sha256=raw["schedule_sha256"],
    )

    kaggle.load_dotenv(env_file)
    owner = kaggle.validate_slug(
        os.getenv("KAGGLE_USERNAME", "").strip(), "KAGGLE_USERNAME"
    )
    dataset_ref = f"{owner}/{args.dataset_slug}"
    counts = expected_target_counts(clean_count)
    python = sys.executable
    upload_command = [
        python,
        "scripts/push_mixed_generation_rule_pairs_dataset.py",
        "--env-file",
        str(env_file),
        "--source-dir",
        str(frozen_dir),
        "--expected-pairs",
        str(clean_count),
        "--expected-target0",
        "0",
        "--expected-target1",
        str(clean_count),
        "--dataset-slug",
        args.dataset_slug,
        "--artifact-tag",
        args.artifact_tag,
        "--label-source",
        args.label_source,
        "--message",
        f"Add {clean_count} Qwen soft-positive Tier A pairs",
    ]
    run(upload_command + ["--dry-run"])
    if not args.dry_run_only:
        run(upload_command)
    upload_manifest_path = (
        ROOT / ".kaggle" / "datasets" / args.dataset_slug / "upload_manifest.json"
    )
    upload_manifest = read_json(upload_manifest_path, "Tier A upload manifest")
    manifest_counts = {
        str(key): int(value)
        for key, value in (upload_manifest.get("targets") or {}).items()
    }
    if (
        upload_manifest.get("dataset") != dataset_ref
        or int(upload_manifest.get("pairs", -1)) != clean_count
        or manifest_counts != counts
        or upload_manifest.get("label_source") != args.label_source
    ):
        raise RuntimeError("Tier A upload manifest differs")
    upload_manifest_sha = sha256_file(upload_manifest_path)

    notes = (
        f"Frozen MiniLM 5ep human baseline plus {clean_count:,} generated Tier A "
        "soft-positive atomic pairs (target0=0, target1="
        f"{clean_count}). Local {EXPECTED_MODEL}, 30 workers, thinking disabled, "
        "prompt-only JSON. Statistical profile policy: support>=5 and weighted "
        "label-1 confidence>=0.8, singleton votes weight 2 and multi-atom votes "
        "weight 1; exactly 20 scheduled examples per 325 category/product-scoped "
        f"rules in the full plan before global-card deduplication. Planned={RAW_COUNT}, "
        f"generated={raw['generated_pairs']}, pending={raw['pending_pairs']}, retained="
        f"{clean_count}. Unit sample weight; frozen checkpoint, recipe and "
        "IID/hard/OOD validation unchanged. This is a data+compute ablation. "
        f"Source dataset {dataset_ref}. Upload manifest SHA-256 "
        f"{upload_manifest_sha}."
    )
    run(
        [
            python,
            "scripts/create_mixed_generation_rule_10k_notebook.py",
            "--pair-count",
            str(clean_count),
            "--expected-target0",
            "0",
            "--expected-target1",
            str(clean_count),
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
        python,
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
            "planned_pair_count": RAW_COUNT,
            "raw_pair_count": raw["generated_pairs"],
            "raw_pending_count": raw["pending_pairs"],
            "frozen_pair_count": clean_count,
            "target_counts": counts,
            "catalog_sha256": EXPECTED_CATALOG_SHA256,
            "schedule_sha256": raw["schedule_sha256"],
            "upload_manifest_sha256": upload_manifest_sha,
            "frozen_checks": frozen_checks,
            "notebook": str(notebook),
            "kernel_slug": args.kernel_slug,
        }
    else:
        run(notebook_command)
        output_root = Path(
            os.getenv("KAGGLE_OUTPUT_DIR", "") or ROOT / "artifacts" / "kaggle"
        ).expanduser()
        if not output_root.is_absolute():
            output_root = ROOT / output_root
        output_dir = output_root.resolve() / args.kernel_slug
        completion = read_json(
            output_dir / "notebook_completed.json", "Kaggle completion marker"
        )
        sync = read_json(
            output_dir / "google_sheets_sync.json", "Google Sheets sync marker"
        )
        comparison = read_json(
            output_dir / "baseline_comparison.json", "baseline comparison"
        )
        run_id = str(completion.get("run_id") or "")
        if (
            completion.get("status") != "complete"
            or completion.get("experiment") != args.experiment_label
            or not run_id
            or sync.get("run_id") != run_id
            or sync.get("status") != "synced"
            or sync.get("comparison_sheet") != "data_exps"
            or comparison.get("status") != "ready"
        ):
            raise RuntimeError(
                "Tier A Kaggle completion/significance/data_exps contract failed"
            )
        label_counts = (completion.get("train_data") or {}).get(
            "label_source_counts"
        ) or {}
        if int(label_counts.get(args.label_source, -1)) != clean_count:
            raise RuntimeError("Kaggle train data has the wrong Tier A count")
        completion_notes = str(completion.get("notes") or "")
        if dataset_ref not in completion_notes or upload_manifest_sha not in completion_notes:
            raise RuntimeError("Kaggle completion does not pin the Tier A Dataset")
        result = {
            "status": "complete",
            "run_id": run_id,
            "dataset_ref": dataset_ref,
            "planned_pair_count": RAW_COUNT,
            "raw_pair_count": raw["generated_pairs"],
            "raw_pending_count": raw["pending_pairs"],
            "frozen_pair_count": clean_count,
            "target_counts": counts,
            "catalog_sha256": EXPECTED_CATALOG_SHA256,
            "schedule_sha256": raw["schedule_sha256"],
            "upload_manifest_sha256": upload_manifest_sha,
            "kernel_slug": args.kernel_slug,
            "baseline_comparison": comparison,
            "frozen_checks": frozen_checks,
        }
    report_path = write_report(args.experiment_label, result)
    print(
        json.dumps(
            {**result, "launcher_report": str(report_path)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
