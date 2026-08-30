#!/usr/bin/env python3
"""Generate, submit, monitor and download the MiniLM-5ep SFT campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import create_minilm_5ep_sft_hparam_notebooks as builder
import create_qwen_training_notebook as qwen_builder
import run_kaggle_notebook as kaggle


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DATASET = "alexproger23/product-matching-validation-splits-v1"
CHECKPOINT_DATASET = "alexproger23/product-matching-minilm-llm-pretrain-5ep"
SIGNIFICANCE_DATASET = "alexproger23/product-matching-minilm-5ep-significance-v1"
REQUIRED_DATASETS = (
    VALIDATION_DATASET,
    CHECKPOINT_DATASET,
    SIGNIFICANCE_DATASET,
)
SLIM_OUTPUT_PATTERN = (
    r"(^|/)(notebook_completed\.json|baseline_comparison\.json|"
    r"google_sheets_sync\.json|sheets_sync_pending\.json|"
    r"experiment_run_id\.txt|experiment_started_at_utc\.txt|"
    r"cross_encoder_config\.json|training_report\.json|training_config\.json|"
    r".*validation_predictions\.parquet|.*\.log)$"
)


def campaign_variants(
    plan: Mapping[str, Any],
    *,
    stage: str | None,
    only: set[str] | None,
    stage_lock: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    base_config = builder.cross_builder.load_training_config(builder.BASE_CONFIG_PATH)
    _, source_sha256 = builder.baseline_builder.embedded_sources()
    selected = []
    if stage_lock is not None:
        contract = builder.normalized_campaign_execution_contract(
            plan,
            stage_lock,
            base_config=base_config,
        )
        effective_stage = str(contract["effective_stage"])
        if stage is not None and stage not in set(
            contract["accepted_stage_filters"]
        ):
            raise builder.CampaignConfigError(
                f"--stage {stage!r} differs from locked stage {effective_stage!r}"
            )
        for normalized in contract["variants"]:
            experiment = str(normalized["experiment"])
            if only and experiment not in only:
                continue
            selected.append(
                {
                    "stage": effective_stage,
                    "experiment": experiment,
                    "role": normalized["role"],
                    "kernel_slug": normalized["kernel_slug"],
                    "title": normalized["title"],
                    "notebook": str(
                        builder.output_path(
                            builder.DEFAULT_OUTPUT_DIR,
                            normalized["variant"],
                        )
                    ),
                    "recipe_sha256": normalized["recipe_sha256"],
                    "source_sha256": normalized["source_sha256"],
                    "loss_variant": normalized["loss_variant"],
                    "loss_hook_sha256": normalized["loss_hook_sha256"],
                    "expected_config": normalized["expected_config"],
                    "baseline_metrics": dict(plan["baseline_metrics"]),
                    "expected_notes": normalized["expected_notes"],
                    "provenance_alias": None,
                    "is_hypothesis": normalized["is_hypothesis"],
                    "family_id": normalized["family_id"],
                    "hypothesis_family_size": normalized[
                        "hypothesis_family_size"
                    ],
                    "stage_lock_payload_sha256": contract[
                        "lock_payload_sha256"
                    ],
                    "parent_provenance": normalized["parent_provenance"],
                }
            )
        if only:
            missing = only - {row["experiment"] for row in selected}
            if missing:
                raise builder.CampaignConfigError(
                    "Requested variants are not in the runnable lock: "
                    f"{sorted(missing)}"
                )
        if not selected and contract["execution_status"] != "skipped":
            raise builder.CampaignConfigError("No campaign variants matched")
        return selected
    else:
        raw_variants = builder.ready_variants(plan, stage_name=stage)
        family_size = 0
    for stage_name, raw in raw_variants:
        experiment = str(raw["experiment"])
        if only and experiment not in only:
            continue
        expected_config = builder.variant_config(base_config, plan, raw)
        loss_variant, _, loss_hook_sha256 = builder.variant_loss(raw)
        if stage_lock is None:
            stage_definition = builder.stage_materializer.stage_by_name(
                plan, stage_name
            )
            declared_family = stage_definition.get("family", {})
            family_size = int(
                declared_family.get(
                    "maximum_hypotheses",
                    sum(
                        candidate.get("role", "candidate")
                        != "current_protocol_control"
                        for candidate in stage_definition.get("variants", [])
                    ),
                )
            )
        expected_notes = builder._variant_notes(
            str(plan["campaign"]),
            stage_name,
            raw,
            expected_config,
            stage_lock=stage_lock,
        )
        selected.append(
            {
                "stage": stage_name,
                "experiment": experiment,
                "role": str(raw.get("role", "candidate")),
                "kernel_slug": str(raw["kernel_slug"]),
                "title": str(raw["title"]),
                "notebook": str(builder.output_path(builder.DEFAULT_OUTPUT_DIR, raw)),
                "recipe_sha256": builder.team_builder.canonical_sha256(
                    expected_config
                ),
                "source_sha256": source_sha256,
                "loss_variant": loss_variant,
                "loss_hook_sha256": loss_hook_sha256,
                "expected_config": expected_config,
                "baseline_metrics": dict(plan["baseline_metrics"]),
                "expected_notes": expected_notes,
                "provenance_alias": (
                    None
                    if stage_lock is not None
                    else builder.variant_provenance_alias(raw, stage=stage_name)
                ),
                "is_hypothesis": True
                if stage_lock is not None
                else str(raw.get("role", "candidate"))
                != "current_protocol_control",
                "hypothesis_family_size": family_size,
            }
        )
    if only:
        missing = only - {row["experiment"] for row in selected}
        if missing:
            raise builder.CampaignConfigError(
                f"Requested variants are not ready or do not exist: {sorted(missing)}"
            )
    if not selected:
        raise builder.CampaignConfigError("No campaign variants matched")
    return selected


LOCKED_BUILD_IDENTITY_FIELDS = (
    "stage",
    "experiment",
    "role",
    "kernel_slug",
    "title",
    "notebook",
    "recipe_sha256",
    "source_sha256",
    "loss_variant",
    "loss_hook_sha256",
    "expected_config",
    "expected_notes",
    "is_hypothesis",
    "family_id",
    "hypothesis_family_size",
    "stage_lock_payload_sha256",
    "parent_provenance",
)


def validate_locked_build_identity(
    expected: Mapping[str, Any],
    built: Mapping[str, Any],
) -> None:
    """Refuse any generator result that drifts from the loaded stage lock."""
    missing = [key for key in LOCKED_BUILD_IDENTITY_FIELDS if key not in built]
    mismatches = {
        key: {"expected": expected.get(key), "actual": built.get(key)}
        for key in LOCKED_BUILD_IDENTITY_FIELDS
        if key in built and built.get(key) != expected.get(key)
    }
    if missing or mismatches:
        raise builder.CampaignConfigError(
            "Built notebook identity differs from the immutable locked campaign: "
            + json.dumps(
                {"missing": missing, "mismatches": mismatches},
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        )


def runner_command(
    entry: Mapping[str, Any],
    *,
    env_file: Path,
    dry_run: bool,
    no_wait: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_kaggle_notebook.py"),
        str(entry["notebook"]),
        "--env-file",
        str(env_file),
        "--slug",
        str(entry["kernel_slug"]),
        "--title",
        str(entry["title"]),
    ]
    for dataset in REQUIRED_DATASETS:
        command.extend(["--dataset", dataset])
    command.append("--no-env-sources")
    if dry_run:
        command.append("--dry-run")
    elif no_wait:
        command.append("--no-wait")
    return command


def output_root() -> Path:
    root = Path(os.getenv("KAGGLE_OUTPUT_DIR", "artifacts/kaggle")).expanduser()
    return root.resolve() if root.is_absolute() else (ROOT / root).resolve()


def _validate_run_output(
    directory: Path,
    *,
    entry: Mapping[str, Any],
    require_sheets: bool,
) -> dict[str, Any]:
    """Validate the completed experiment, paired comparison and Sheets upsert."""
    required_root_files = {
        "notebook_completed.json",
        "baseline_comparison.json",
        "experiment_run_id.txt",
    }
    if require_sheets:
        required_root_files.add("google_sheets_sync.json")
    missing = [name for name in required_root_files if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(f"Run output is missing files: {sorted(missing)}")
    if require_sheets and (directory / "sheets_sync_pending.json").exists():
        raise RuntimeError("Google Sheets synchronization is still pending")

    completion = json.loads(
        (directory / "notebook_completed.json").read_text(encoding="utf-8")
    )
    run_id = str(completion.get("run_id", "")).strip()
    if completion.get("status") != "complete" or not run_id:
        raise RuntimeError("Completion artifact is not a successful run")
    if completion.get("experiment") != entry["experiment"]:
        raise RuntimeError("Completion experiment label differs from the campaign")
    if completion.get("experiment_group") != "sft":
        raise RuntimeError("Completion was not routed to experiment_group=sft")
    if completion.get("frozen_recipe_sha256") != entry["recipe_sha256"]:
        raise RuntimeError("Completion recipe hash differs from the campaign")
    if completion.get("code_bundle_sha256") != entry["source_sha256"]:
        raise RuntimeError("Completion source hash differs from the campaign")
    if completion.get("dataset_ref") != VALIDATION_DATASET:
        raise RuntimeError("Completion used a different validation Dataset")
    if completion.get("initial_checkpoint_ref") != CHECKPOINT_DATASET:
        raise RuntimeError("Completion used a different initial checkpoint")
    if (
        completion.get("initial_checkpoint_manifest_sha256")
        != builder.team_builder.CHECKPOINT_MANIFEST_SHA256
    ):
        raise RuntimeError("Completion initial-checkpoint manifest differs")
    if completion.get("loss_hook_sha256") != entry["loss_hook_sha256"]:
        raise RuntimeError("Completion did not use the planned frozen loss hook")
    completion_notes = completion.get("notes")
    if not isinstance(completion_notes, str):
        raise RuntimeError("Completion has no exact experiment notes string")
    expected_notes = entry.get("expected_notes")
    if not isinstance(expected_notes, str):
        raise RuntimeError("Campaign entry has no exact expected notes string")
    if completion_notes != expected_notes:
        alias = entry.get("provenance_alias")
        if not isinstance(alias, Mapping):
            raise RuntimeError("Completion notes differ from the frozen campaign notes")
        if entry.get("role") != "current_protocol_control":
            raise RuntimeError("Only the protocol control may use a notes alias")
        notes_sha256 = hashlib.sha256(completion_notes.encode("utf-8")).hexdigest()
        if notes_sha256 != alias["accepted_completion_notes_sha256"]:
            raise RuntimeError("Completion notes differ from the exact control alias")
        try:
            legacy_notes = json.loads(completion_notes)
        except json.JSONDecodeError as error:
            raise RuntimeError("Aliased control notes are not JSON") from error
        if (
            not isinstance(legacy_notes, Mapping)
            or legacy_notes.get("stage") != alias["recorded_stage"]
            or entry.get("stage") != alias["canonical_stage"]
        ):
            raise RuntimeError("Aliased control notes have unexpected stage provenance")
    if (directory / "experiment_run_id.txt").read_text(encoding="utf-8").strip() != run_id:
        raise RuntimeError("experiment_run_id.txt differs from completion run_id")
    report = completion.get("training_report", {})
    if set(report.get("validation_splits", {})) != {"iid", "hard", "ood"}:
        raise RuntimeError("Completion does not contain all three validation splits")
    train_data = completion.get("train_data", {})
    if (
        train_data.get("train_pairs") != 306_669
        or train_data.get("items") != 711_304
        or train_data.get("same_size_as_human_baseline") is not True
    ):
        raise RuntimeError("Completion did not use the frozen human train data")
    if (
        report.get("training_sampling") != "none"
        or report.get("training_loss_weighting") != "none"
        or report.get("training_subset") != "all"
        or report.get("original_training_examples") != 306_669
        or report.get("training_unique_coverage_per_epoch") != 1.0
        or any(
            report.get(name) != 1.0
            for name in (
                "training_loss_weight_min",
                "training_loss_weight_median",
                "training_loss_weight_max",
            )
        )
    ):
        raise RuntimeError(
            "Completion changed frozen sampling or external sample-weight semantics"
        )

    comparison = json.loads(
        (directory / "baseline_comparison.json").read_text(encoding="utf-8")
    )
    if completion.get("baseline_comparison") != comparison:
        raise RuntimeError("Completion and standalone comparison artifacts differ")
    if comparison.get("status") != "ready":
        raise RuntimeError("Paired baseline comparison is not ready")
    if comparison.get("baseline_run_id") != builder.team_builder.SIGNIFICANCE_BASELINE_RUN_ID:
        raise RuntimeError("Paired comparison used a different baseline")
    if comparison.get("candidate_run_id") != run_id:
        raise RuntimeError("Paired comparison candidate_run_id differs")
    if comparison.get("method") != "paired_component_permutation":
        raise RuntimeError("Paired comparison method differs from the protocol")
    if set(comparison.get("splits", {})) != {"iid", "hard", "ood"}:
        raise RuntimeError("Paired comparison is missing validation splits")

    if require_sheets:
        sync = json.loads(
            (directory / "google_sheets_sync.json").read_text(encoding="utf-8")
        )
        if sync.get("status") != "synced" or sync.get("run_id") != run_id:
            raise RuntimeError(
                "Google Sheets synchronization did not finish for this run_id"
            )
        if (
            sync.get("experiment_group") != "sft"
            or sync.get("comparison_sheet") != "sft_exps"
        ):
            raise RuntimeError("Google Sheets synchronization did not target sft_exps")
        if sync.get("spreadsheet_id") != qwen_builder.EXPERIMENT_SPREADSHEET_ID:
            raise RuntimeError(
                "Google Sheets synchronization targeted another spreadsheet"
            )

    reports = list(directory.rglob("training_report.json"))
    configs = list(directory.rglob("training_config.json"))
    if len(reports) != 1 or len(configs) != 1:
        raise RuntimeError("Slim output must contain one training report and config")
    standalone_report = json.loads(reports[0].read_text(encoding="utf-8"))
    if standalone_report != report:
        raise RuntimeError("Completion and standalone training reports differ")
    expected_config = entry["expected_config"]
    config_paths = [directory / "cross_encoder_config.json", configs[0]]
    for config_path in config_paths:
        if not config_path.is_file():
            raise RuntimeError(f"Slim output is missing config: {config_path.name}")
        actual_config = json.loads(config_path.read_text(encoding="utf-8"))
        if set(actual_config) != set(expected_config):
            raise RuntimeError(f"Training config keys differ in {config_path}")
        for key, expected in expected_config.items():
            if key != "model" and actual_config.get(key) != expected:
                raise RuntimeError(
                    f"Training config {key} differs in {config_path}: "
                    f"{actual_config.get(key)!r} != {expected!r}"
                )

    split_rows = {"iid": 12_000, "hard": 5_814, "ood": 41_171}
    baseline_metrics = entry["baseline_metrics"]
    comparison_splits = comparison["splits"]
    for split, expected_rows in split_rows.items():
        metrics = report["validation_splits"][split]
        score = metrics.get("macro_average_precision")
        overall = metrics.get("overall_average_precision")
        if metrics.get("examples") != expected_rows:
            raise RuntimeError(f"Unexpected {split} validation row count")
        if any(
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 <= float(value) <= 1
            for value in (score, overall)
        ):
            raise RuntimeError(f"Invalid {split} AP metrics")
        paired = comparison_splits[split]
        statistics = {
            name: paired.get(name)
            for name in (
                "delta_macro_average_precision",
                "p_value",
                "p_value_holm",
                "ci95_low",
                "ci95_high",
            )
        }
        if any(
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in statistics.values()
        ):
            raise RuntimeError(f"Invalid {split} paired statistics")
        if not 0 <= float(statistics["p_value"]) <= 1 or not 0 <= float(
            statistics["p_value_holm"]
        ) <= 1:
            raise RuntimeError(f"Invalid {split} p-value")
        if float(statistics["ci95_low"]) > float(statistics["ci95_high"]):
            raise RuntimeError(f"Reversed {split} confidence interval")
        expected_delta = float(score) - float(
            baseline_metrics[f"{split}_macro_ap"]
        )
        if not math.isclose(
            float(statistics["delta_macro_average_precision"]),
            expected_delta,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"Inconsistent {split} paired delta")

    diagnostic_values = (
        report["validation_splits"]["hard"].get("recall_at_precision_0_99"),
        report["validation_splits"]["hard"].get("roc_auc"),
        report["validation_splits"]["ood"].get("log_loss"),
        report.get("training_seconds"),
        report.get("total_pipeline_seconds"),
    )
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(float(value))
        for value in diagnostic_values
    ):
        raise RuntimeError("Training report has non-finite diagnostics/runtime")

    import pyarrow.parquet as pq

    for split in ("iid", "hard", "ood"):
        predictions = list(directory.rglob(f"{split}_validation_predictions.parquet"))
        if len(predictions) != 1 or predictions[0].stat().st_size == 0:
            raise RuntimeError(f"Slim output is missing {split} predictions")
        parquet = pq.ParquetFile(predictions[0])
        if parquet.metadata.num_rows != split_rows[split]:
            raise RuntimeError(f"Unexpected {split} prediction parquet row count")
        required_columns = {"id1", "id2", "target", "score"}
        if missing_columns := required_columns - set(parquet.schema.names):
            raise RuntimeError(
                f"{split} prediction parquet is missing {sorted(missing_columns)}"
            )
        if not {"category", "category_1"} & set(parquet.schema.names):
            raise RuntimeError(f"{split} prediction parquet has no category column")
    return {
        "run_id": run_id,
        "experiment": entry["experiment"],
        "comparison_sheet": "sft_exps",
        "directory": str(directory),
    }


def validate_run_payload(
    directory: Path,
    *,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    """Read-only provenance/config/prediction check before any Sheets mutation."""
    return _validate_run_output(
        directory,
        entry=entry,
        require_sheets=False,
    )


def validate_run_output(
    directory: Path,
    *,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate payload plus the exact final Google Sheets synchronization."""
    return _validate_run_output(
        directory,
        entry=entry,
        require_sheets=True,
    )


def validate_control_gate(
    directory: Path,
    *,
    entry: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Refuse a sweep when the current code no longer reproduces the baseline."""
    validated = validate_run_output(directory, entry=entry)
    completion = json.loads(
        (directory / "notebook_completed.json").read_text(encoding="utf-8")
    )
    comparison = completion["baseline_comparison"]
    deltas = {
        split: float(
            comparison["splits"][split]["delta_macro_average_precision"]
        )
        for split in ("iid", "hard", "ood")
    }
    runtime = float(completion["training_report"]["total_pipeline_seconds"])
    numeric = [*deltas.values(), runtime]
    if any(not math.isfinite(value) for value in numeric) or runtime <= 0:
        raise RuntimeError("Control gate contains invalid metrics or runtime")
    runtime_drift = abs(
        runtime / float(gate["baseline_total_pipeline_seconds"]) - 1.0
    )
    checks = {
        "iid_delta": abs(deltas["iid"]) <= float(gate["max_abs_iid_delta"]),
        "hard_delta": abs(deltas["hard"]) <= float(gate["max_abs_hard_delta"]),
        "ood_delta": abs(deltas["ood"]) <= float(gate["max_abs_ood_delta"]),
        "runtime_drift": runtime_drift
        <= float(gate["max_runtime_relative_drift"]),
    }
    result = {
        **validated,
        "deltas": deltas,
        "runtime_relative_drift": runtime_drift,
        "checks": checks,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Current-protocol control failed; refusing to launch SFT candidates: "
            + json.dumps(result, ensure_ascii=False, sort_keys=True)
        )
    return result


def retry_pending_sheets_sync(directory: Path, *, kernel_ref: str) -> None:
    """Retry a non-fatal Kaggle Sheets failure with the local service account."""
    pending_path = directory / "sheets_sync_pending.json"
    sync_path = directory / "google_sheets_sync.json"
    completion_path = directory / "notebook_completed.json"
    if not completion_path.is_file():
        raise RuntimeError("Cannot retry Sheets sync without notebook_completed.json")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion_run_id = str(completion.get("run_id", "")).strip()
    if (
        completion.get("status") != "complete"
        or not completion_run_id
        or completion.get("experiment_group") != "sft"
        or completion.get("baseline_comparison", {}).get("status") != "ready"
    ):
        raise RuntimeError("Refusing to sync an incomplete SFT completion artifact")

    sync_payload: Mapping[str, Any] = {}
    if sync_path.is_file():
        sync_payload = json.loads(sync_path.read_text(encoding="utf-8"))
    if not pending_path.exists() and sync_payload.get("status") == "synced":
        expected_identity = {
            "run_id": completion_run_id,
            "experiment_group": "sft",
            "comparison_sheet": "sft_exps",
            "spreadsheet_id": qwen_builder.EXPERIMENT_SPREADSHEET_ID,
        }
        mismatches = {
            key: {"actual": sync_payload.get(key), "expected": expected}
            for key, expected in expected_identity.items()
            if sync_payload.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(
                "Refusing to trust a stale/mismatched synced Sheets marker: "
                + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
            )
        return

    from push_google_sheets_credentials_dataset import DEFAULT_KEY_PATH
    from src.google_sheets_logger import (
        load_service_account_info,
        sync_experiment,
    )

    raw_credentials = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if not raw_credentials:
        configured_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_PATH", "").strip()
        key_path = (
            Path(configured_path).expanduser().resolve()
            if configured_path
            else DEFAULT_KEY_PATH.expanduser().resolve()
        )
        raw_credentials = key_path.read_text(encoding="utf-8")
    load_service_account_info(raw_credentials)
    result = sync_experiment(
        spreadsheet_id=qwen_builder.EXPERIMENT_SPREADSHEET_ID,
        service_account_json=raw_credentials,
        completion=completion,
    )
    synced = {"status": "synced", **result}
    sync_path.write_text(
        json.dumps(synced, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if pending_path.exists():
        preserved = directory / "sheets_sync_pending.remote.json"
        if preserved.exists():
            preserved = directory / f"sheets_sync_pending.remote-{time.time_ns()}.json"
        pending_path.rename(preserved)
    print(
        json.dumps(
            {"local_sheets_retry": "synced", "kernel_ref": kernel_ref},
            ensure_ascii=False,
        )
    )


def remote_kernel_status(
    cli: list[str],
    kernel_ref: str,
) -> str:
    result = kaggle.run_command(
        cli + ["kernels", "status", kernel_ref],
        check=False,
    )
    if result.returncode == 0:
        return kaggle.extract_status(result.stdout) or "status_error"
    lowered = result.stdout.lower()
    if any(marker in lowered for marker in ("not found", "does not exist", "404")):
        return "missing"
    # The Kaggle API currently reports a missing private slug as a permission
    # denial. Resolve that ambiguity against the authenticated user's own list.
    slug = kernel_ref.rsplit("/", 1)[-1]
    listing = kaggle.run_command(
        cli
        + [
            "kernels",
            "list",
            "--mine",
            "--search",
            slug,
            "--page-size",
            "100",
            "--format",
            "json",
        ],
        check=False,
    )
    if listing.returncode:
        raise RuntimeError(
            f"Could not list authenticated kernels while resolving {kernel_ref}"
        )
    if kernel_ref not in listing.stdout:
        return "missing"
    raise RuntimeError(
        f"Could not determine remote status for {kernel_ref}; refusing to resubmit"
    )


def download_output(
    cli: list[str],
    *,
    username: str,
    entry: Mapping[str, Any],
    full_download: bool,
) -> Path:
    root = output_root()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / str(entry["kernel_slug"])
    kernel_ref = f"{username}/{entry['kernel_slug']}"
    staging: Path | None = None
    for attempt in range(1, 4):
        attempt_staging = Path(
            tempfile.mkdtemp(
                prefix=f".{entry['kernel_slug']}.download-",
                dir=root,
            )
        )
        command = cli + [
            "kernels",
            "output",
            kernel_ref,
            "-p",
            str(attempt_staging),
            "--force",
            "--page-size",
            "200",
        ]
        if not full_download:
            command.extend(["--file-pattern", SLIM_OUTPUT_PATTERN])
        result = kaggle.run_command(command, check=False)
        if result.returncode == 0:
            staging = attempt_staging
            break
        shutil.rmtree(attempt_staging)
        if attempt < 3:
            delay = 3 if attempt == 1 else 8
            print(
                f"Kaggle output download failed ({attempt}/3); retrying in "
                f"{delay}s",
                flush=True,
            )
            time.sleep(delay)
    if staging is None:
        raise RuntimeError(
            f"Kaggle output download failed after 3 attempts: {kernel_ref}"
        )
    # Never let a stale/mismatched completion artifact mutate the shared Sheet.
    validate_run_payload(staging, entry=entry)
    retry_pending_sheets_sync(staging, kernel_ref=kernel_ref)
    validation = validate_run_output(staging, entry=entry)
    if destination.exists():
        backup = root / f".{entry['kernel_slug']}.previous-{time.time_ns()}"
        destination.rename(backup)
        print(f"Preserved previous output at: {backup}")
    staging.rename(destination)
    validation["directory"] = str(destination)
    print(json.dumps({"validated_output": validation}, ensure_ascii=False))
    return destination


def status_rows(
    cli: list[str],
    *,
    username: str,
    entries: list[dict[str, Any]],
    download_complete: bool,
    full_download: bool,
    logs_on_failure: bool,
) -> list[dict[str, Any]]:
    rows = []
    for entry in entries:
        kernel_ref = f"{username}/{entry['kernel_slug']}"
        result = kaggle.run_command(
            cli + ["kernels", "status", kernel_ref],
            check=False,
        )
        status = kaggle.extract_status(result.stdout) if result.returncode == 0 else None
        row = dict(entry)
        row.update(
            {
                "kernel_ref": kernel_ref,
                "status": status or "status_error",
                "status_return_code": result.returncode,
            }
        )
        if status in kaggle.TERMINAL_SUCCESS and download_complete:
            row["output_dir"] = str(
                download_output(
                    cli,
                    username=username,
                    entry=entry,
                    full_download=full_download,
                )
            )
        elif status in kaggle.TERMINAL_FAILURE and logs_on_failure:
            kaggle.run_command(
                cli + ["kernels", "logs", kernel_ref],
                check=False,
            )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=builder.DEFAULT_PLAN)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--stage")
    parser.add_argument("--stage-lock", type=Path)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--download-complete", action="store_true")
    parser.add_argument(
        "--full-download",
        action="store_true",
        help="download model weights too; the default keeps only reports/logs/predictions",
    )
    parser.add_argument("--logs-on-failure", action="store_true")
    parser.add_argument(
        "--allow-background-fanout",
        action="store_true",
        help="explicitly allow submitting more than one kernel without --wait",
    )
    parser.add_argument(
        "--force-resubmit",
        action="store_true",
        help="push a new kernel version even when a prior version exists",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="allow a new version when the current remote kernel failed",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.wait and not args.submit:
        raise SystemExit("--wait requires --submit")
    if sum(bool(value) for value in (args.dry_run, args.submit, args.status)) != 1:
        raise SystemExit("Choose exactly one of --dry-run, --submit or --status")
    if args.only and args.limit is not None:
        raise SystemExit("Use either explicit --only values or --limit, not both")
    if args.download_complete and not args.status:
        raise SystemExit("--download-complete requires --status")
    if args.logs_on_failure and not args.status:
        raise SystemExit("--logs-on-failure requires --status")

    env_file = args.env_file if args.env_file.is_absolute() else ROOT / args.env_file
    plan_path = args.plan if args.plan.is_absolute() else ROOT / args.plan
    stage_lock_path = args.stage_lock
    if stage_lock_path is not None and not stage_lock_path.is_absolute():
        stage_lock_path = ROOT / stage_lock_path
    only = set(args.only) or None
    plan = builder.load_plan(plan_path)
    base_config = builder.cross_builder.load_training_config(builder.BASE_CONFIG_PATH)
    stage_lock = (
        builder.load_campaign_lock(
            stage_lock_path,
            plan=plan,
            base_config=base_config,
        )
        if stage_lock_path is not None
        else None
    )
    trusted_artifacts_dir: Path | None = None
    if stage_lock is not None and stage_lock.get("schema_version") == 2:
        adaptive = builder._adaptive_lock_module()
        trusted = adaptive.load_trusted_provenance(
            adaptive.trusted_provenance_manifest_path(
                stage_lock_path.resolve(strict=True)
            ),
            plan=plan,
        )
        trusted_artifacts_dir = Path(str(trusted["artifacts_dir"]))
    entries = campaign_variants(
        plan,
        stage=args.stage,
        only=only,
        stage_lock=stage_lock,
    )
    if (
        stage_lock is not None
        and stage_lock.get("schema_version") == 2
        and stage_lock.get("execution_status") == "skipped"
    ):
        print(
            json.dumps(
                {
                    "status": "validated_skipped_receipt",
                    "mode": stage_lock["mode"],
                    "effective_stage": stage_lock["effective_stage"],
                    "lock_payload_sha256": stage_lock["lock_payload_sha256"],
                    "kaggle_actions": 0,
                },
                ensure_ascii=False,
            )
        )
        return
    control_role = str(plan["control_gate"]["role"])
    all_entries = campaign_variants(plan, stage=None, only=None)
    controls = [entry for entry in all_entries if entry["role"] == control_role]
    if len(controls) != 1:
        raise builder.CampaignConfigError(
            f"Campaign must contain exactly one {control_role!r}, found {len(controls)}"
        )
    control_entry = controls[0]
    entries.sort(key=lambda entry: entry["role"] != control_role)
    if args.limit is not None:
        entries = entries[: args.limit]
    if args.submit and len(entries) > 1 and not args.wait and not args.allow_background_fanout:
        raise SystemExit(
            "Background fan-out is disabled: use --wait, --limit 1, --only, "
            "or explicitly pass --allow-background-fanout"
        )

    if args.dry_run or args.submit:
        built = builder.build_campaign(
            plan_path=plan_path,
            output_dir=builder.DEFAULT_OUTPUT_DIR,
            stage_name=args.stage,
            only={entry["experiment"] for entry in entries},
            stage_lock_path=stage_lock_path,
        )
        built_by_experiment = {entry["experiment"]: entry for entry in built}
        if len(built_by_experiment) != len(built) or set(built_by_experiment) != {
            str(entry["experiment"]) for entry in entries
        }:
            raise builder.CampaignConfigError(
                "Built notebook set differs from the selected campaign variants"
            )
        for entry in entries:
            built_entry = built_by_experiment[entry["experiment"]]
            if stage_lock is not None and stage_lock.get("schema_version") == 2:
                validate_locked_build_identity(entry, built_entry)
            else:
                entry.update(built_entry)

    cli: list[str] | None = None
    username = ""
    if args.submit or args.status:
        kaggle.load_dotenv(env_file)
        if (
            trusted_artifacts_dir is not None
            and output_root() != trusted_artifacts_dir
        ):
            raise RuntimeError(
                "KAGGLE_OUTPUT_DIR differs from the schema-v2 trusted "
                "artifacts authority"
            )
        username = os.getenv("KAGGLE_USERNAME", "").strip()
        if not username:
            kaggle.fail("set KAGGLE_USERNAME in .env")
        if not kaggle.env_bool("KAGGLE_IS_PRIVATE", True):
            kaggle.fail("SFT experiments must use KAGGLE_IS_PRIVATE=true")
        if not os.getenv("KAGGLE_API_TOKEN", "").strip():
            kaggle.fail("set KAGGLE_API_TOKEN in .env")
        cli = kaggle.kaggle_command()

    if args.dry_run:
        for entry in entries:
            command = runner_command(
                entry,
                env_file=env_file,
                dry_run=True,
                no_wait=True,
            )
            print("$", " ".join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)

    if args.submit:
        assert cli is not None
        control_gate_verified = False
        for entry in entries:
            if entry["role"] != control_role and not control_gate_verified:
                gate_result = validate_control_gate(
                    output_root() / str(control_entry["kernel_slug"]),
                    entry=control_entry,
                    gate=plan["control_gate"],
                )
                print(json.dumps({"control_gate": gate_result}, ensure_ascii=False))
                control_gate_verified = True
            destination = output_root() / str(entry["kernel_slug"])
            if not args.force_resubmit:
                try:
                    validation = validate_run_output(destination, entry=entry)
                except (FileNotFoundError, RuntimeError, json.JSONDecodeError):
                    validation = None
                if validation is not None:
                    print(json.dumps({"resume_skip_local": validation}, ensure_ascii=False))
                    continue

                kernel_ref = f"{username}/{entry['kernel_slug']}"
                remote_status = remote_kernel_status(cli, kernel_ref)
                if remote_status in kaggle.TERMINAL_SUCCESS:
                    download_output(
                        cli,
                        username=username,
                        entry=entry,
                        full_download=args.full_download,
                    )
                    print(f"Skipping completed remote run: {kernel_ref}")
                    continue
                if remote_status in {"queued", "running"}:
                    if not args.wait:
                        print(f"Skipping active remote run: {kernel_ref} ({remote_status})")
                        continue
                    kaggle.wait_for_kernel(
                        cli,
                        kernel_ref,
                        poll_interval=kaggle.env_int(
                            "KAGGLE_POLL_INTERVAL_SECONDS", 30, 5
                        ),
                        wait_timeout=kaggle.env_int(
                            "KAGGLE_WAIT_TIMEOUT_SECONDS", 45000, 60
                        ),
                    )
                    download_output(
                        cli,
                        username=username,
                        entry=entry,
                        full_download=args.full_download,
                    )
                    continue
                if remote_status in kaggle.TERMINAL_FAILURE and not args.retry_failed:
                    raise RuntimeError(
                        f"Remote run {kernel_ref} failed; inspect logs and pass "
                        "--retry-failed only after fixing the cause"
                    )
                if remote_status not in {"missing", *kaggle.TERMINAL_FAILURE}:
                    raise RuntimeError(
                        f"Unexpected remote status {remote_status!r} for {kernel_ref}"
                    )
            command = runner_command(
                entry,
                env_file=env_file,
                dry_run=False,
                no_wait=True,
            )
            print("$", " ".join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            if args.wait:
                kernel_ref = f"{username}/{entry['kernel_slug']}"
                kaggle.wait_for_kernel(
                    cli,
                    kernel_ref,
                    poll_interval=kaggle.env_int(
                        "KAGGLE_POLL_INTERVAL_SECONDS", 30, 5
                    ),
                    wait_timeout=kaggle.env_int(
                        "KAGGLE_WAIT_TIMEOUT_SECONDS", 45000, 60
                    ),
                )
                download_output(
                    cli,
                    username=username,
                    entry=entry,
                    full_download=args.full_download,
                )
        if args.wait and any(entry["role"] == control_role for entry in entries):
            gate_result = validate_control_gate(
                output_root() / str(control_entry["kernel_slug"]),
                entry=control_entry,
                gate=plan["control_gate"],
            )
            print(json.dumps({"control_gate": gate_result}, ensure_ascii=False))

    if args.status:
        assert cli is not None
        rows = status_rows(
            cli,
            username=username,
            entries=entries,
            download_complete=args.download_complete,
            full_download=args.full_download,
            logs_on_failure=args.logs_on_failure,
        )
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        bad_statuses = kaggle.TERMINAL_FAILURE | {"status_error"}
        if any(row["status"] in bad_statuses for row in rows):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
