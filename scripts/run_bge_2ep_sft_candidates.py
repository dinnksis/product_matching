#!/usr/bin/env python3
"""Plan, stage, execute and compare the guarded BGE SFT candidates.

The default mode is plan-only and performs no Kaggle CLI resolution.  Every
mutating execution is gated by the exact local baseline output, its private
slim Dataset payload, the declared remote Dataset version, and a read-only
remote re-verification immediately before a first kernel push.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import create_bge_2ep_sft_candidate_notebooks as candidate_builder
import create_bge_2ep_sft_notebooks as baseline_builder
import create_qwen_training_notebook as shared
import push_bge_2ep_sft_baseline_dataset as baseline_uploader
import push_kaggle_training_dataset as dataset_push
import run_bge_2ep_sft_kaggle as baseline_launcher
import run_kaggle_notebook as kaggle
import summarize_bge_2ep_sft_comparisons as comparator


ROOT = Path(__file__).resolve().parents[1]
OWNER = "alexproger23"
LR_FAMILY_NAME = "bge2_sft_lr_log_line_core_v1"
EPOCH_FAMILY_NAME = "bge2_sft_selected_lr_epoch_line_v1"
DEFAULT_REPORT_ROOT = ROOT / "reports" / candidate_builder.TEMPLATE
DEFAULT_LR_RECEIPT = DEFAULT_REPORT_ROOT / "lr" / "lr_selection_receipt.json"
COMPARISON_SYNC_FILENAME = "google_sheets_comparison_sync.json"
LR_RECEIPT_FILENAME = "lr_selection_receipt.json"
EPOCH_RECEIPT_FILENAME = "epoch_selection_receipt.json"
READY_MARKERS = ("ready", "complete", "successful")
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
SLIM_OUTPUT_PATTERN = (
    r"(^|/)(notebook_completed\.json|google_sheets_sync\.json|"
    r"sheets_sync_pending\.json|experiment_run_id\.txt|"
    r"experiment_started_at_utc\.txt|cross_encoder_config\.json|"
    r"bge_memory_preflight\.json|bge_runtime_versions\.json|"
    r"bge_train_data_report\.json|bge_baseline_dataset_gate\.json|"
    r"training_report\.json|training_config\.json|"
    r"(?:iid|hard)_validation_predictions\.parquet|.*\.log)$"
)


class CandidateWorkflowError(RuntimeError):
    """Raised when the fail-closed candidate workflow cannot continue."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CandidateWorkflowError(f"Could not read JSON object {path}: {error}") from error
    if not isinstance(payload, dict):
        raise CandidateWorkflowError(f"Expected a JSON object in {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    if path.exists():
        if path.read_text(encoding="utf-8") == serialized:
            return
        raise CandidateWorkflowError(f"Refusing to overwrite a differing receipt: {path}")
    path.write_text(serialized, encoding="utf-8")


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or HASH_PATTERN.fullmatch(value) is None:
        raise CandidateWorkflowError(f"{label} is not an exact lowercase SHA-256")
    return value


def plan() -> dict[str, Any]:
    """Return the network-free, identity-free execution design."""
    return {
        "schema_version": 1,
        "mode": "plan_only",
        "campaign": baseline_builder.CAMPAIGN,
        "baseline_identity": "dynamic; must equal final frozen baseline Dataset binding",
        "blocked_until": [
            "one strictly validated local final-v4 baseline output with exact Sheets marker",
            "one locally verified private slim baseline Dataset payload",
            "an explicit positive remote Dataset version",
            "remote version/manifest/file-set/private equality immediately before each first push",
        ],
        "stages": [
            {
                "name": "lr",
                "execution": "sequential",
                "candidates": ["lr1e5", "lr4e5"],
                "comparison": (
                    "paired IID primary; paired hard diagnostic; "
                    "Holm within two-candidate family"
                ),
                "selection": "IID practical tie margin 0.002; baseline then lower LR then upper LR",
            },
            {
                "name": "e2",
                "execution": "only after exact LR selection receipt",
                "parent": "selected e1 run and recipe (baseline or LR candidate)",
                "comparison": "paired IID primary; paired hard diagnostic; one-member Holm family",
                "selection": "keep e1 unless e2 IID delta is strictly greater than 0.002",
            },
        ],
        "ood": {
            "macro_average_precision": -1.0,
            "compared": False,
            "delta_p_ci": None,
        },
        "deferred": ["LR boundary expansion", "epoch 3"],
        "mutation": False,
    }


def load_local_baseline_authority(
    *,
    owner: str,
    dataset_version: int,
    source_dir: Path | None = None,
    stage_dir: Path = baseline_uploader.STAGE_DIR,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Load the exact current baseline, local output and staged Dataset."""
    if owner != OWNER:
        raise CandidateWorkflowError(f"Frozen BGE campaign owner must remain {OWNER}")
    if (
        isinstance(dataset_version, bool)
        or not isinstance(dataset_version, int)
        or dataset_version < 1
    ):
        raise CandidateWorkflowError("A positive frozen baseline Dataset version is required")
    entry = baseline_uploader.expected_baseline_entry(owner)
    resolved_source = (
        source_dir.expanduser().resolve()
        if source_dir is not None
        else baseline_uploader.default_source_dir(entry).resolve()
    )
    if not resolved_source.is_dir():
        raise CandidateWorkflowError(
            f"Exact local baseline output is missing: {resolved_source}"
        )
    completion = baseline_uploader.validate_baseline_source(
        resolved_source,
        entry=entry,
    )
    resolved_stage = stage_dir.expanduser().resolve()
    manifest_path = resolved_stage / baseline_uploader.MANIFEST_FILENAME
    manifest = _read_json(manifest_path)
    manifest_sha256 = baseline_uploader.verify_payload_for_upload(
        resolved_stage,
        manifest,
    )
    if expected_manifest_sha256 is not None:
        _require_hash(expected_manifest_sha256, "expected baseline manifest SHA-256")
        if manifest_sha256 != expected_manifest_sha256:
            raise CandidateWorkflowError("Local baseline manifest differs from requested hash")
    frozen = comparator.load_frozen_baseline(resolved_stage)
    if frozen["manifest_sha256"] != manifest_sha256:
        raise CandidateWorkflowError("Baseline comparator and uploader hashes differ")
    if frozen["completion"] != completion:
        raise CandidateWorkflowError("Frozen Dataset completion differs from local output")
    context = candidate_builder.validate_baseline_context(
        {
            "dataset_ref": manifest["dataset"],
            "dataset_slug": str(manifest["dataset"]).rsplit("/", 1)[-1],
            "dataset_version": dataset_version,
            "manifest_sha256": manifest_sha256,
            "manifest_canonical_sha256": candidate_builder.canonical_sha256(manifest),
            "manifest": manifest,
            "binding": manifest["binding"],
        }
    )
    parent = candidate_builder.baseline_parent_receipt(context, entry)
    return {
        "owner": owner,
        "entry": entry,
        "source_dir": resolved_source,
        "stage_dir": resolved_stage,
        "completion": completion,
        "frozen": frozen,
        "context": context,
        "baseline_parent": parent,
    }


def expected_dataset_sources(entry: Mapping[str, Any]) -> list[str]:
    context = candidate_builder.validate_baseline_context(entry["baseline_context"])
    return [
        str(entry["validation_dataset"]),
        str(entry["checkpoint_dataset"]),
        context["dataset_ref"],
        baseline_launcher.CREDENTIALS_DATASET,
    ]


def runner_command(
    entry: Mapping[str, Any],
    *,
    env_file: Path,
) -> list[str]:
    candidate_builder.load_candidate_notebook(Path(entry["notebook"]), entry=entry)
    context = candidate_builder.validate_baseline_context(entry["baseline_context"])
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_kaggle_notebook.py"),
        str(entry["notebook"]),
        "--env-file",
        str(env_file),
        "--slug",
        str(entry["kernel_slug"]),
        "--title",
        str(entry["title"]),
        "--dataset",
        str(entry["validation_dataset"]),
        "--dataset",
        str(entry["checkpoint_dataset"]),
        "--dataset",
        context["dataset_ref"],
        "--no-env-sources",
        "--no-gpu-check",
        "--no-download",
        "--dry-run",
    ]


def run_inner_runner(command: list[str]) -> None:
    variable = "KAGGLE_GOOGLE_SHEETS_CREDENTIALS_DATASET"
    previous = os.environ.get(variable)
    os.environ[variable] = baseline_launcher.CREDENTIALS_DATASET
    try:
        kaggle.run_command(command)
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous


def validate_staged_kernel_metadata(entry: Mapping[str, Any]) -> dict[str, Any]:
    stage_dir = kaggle.STAGE_ROOT / str(entry["kernel_slug"])
    path = stage_dir / "kernel-metadata.json"
    if not path.is_file():
        raise CandidateWorkflowError("Candidate dry-run produced no kernel metadata")
    metadata = _read_json(path)
    expected_sources = expected_dataset_sources(entry)
    if metadata.get("dataset_sources") != expected_sources:
        raise CandidateWorkflowError("Staged candidate Dataset attachments differ")
    exact = {
        "id": f"{OWNER}/{entry['kernel_slug']}",
        "title": entry["title"],
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    for key, expected in exact.items():
        if metadata.get(key) != expected:
            raise CandidateWorkflowError(f"Staged candidate metadata differs at {key}")
    candidate_builder.load_candidate_notebook(
        stage_dir / "notebook.ipynb",
        entry=entry,
    )
    return metadata


def verify_remote_baseline_dataset(
    cli: list[str],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact remote version, readiness, manifest, file set and privacy."""
    context = candidate_builder.validate_baseline_context(authority["context"])
    dataset_ref = context["dataset_ref"]

    def exact_status() -> dict[str, Any]:
        status = dataset_push.dataset_status(cli, dataset_ref)
        if not isinstance(status, dict):
            raise CandidateWorkflowError("Could not read remote baseline Dataset status")
        try:
            version = int(status.get("current_version_number", 0))
        except (TypeError, ValueError) as error:
            raise CandidateWorkflowError("Remote baseline Dataset version is invalid") from error
        if version != context["dataset_version"]:
            raise CandidateWorkflowError(
                "Remote baseline Dataset version differs: "
                f"observed={version}, expected={context['dataset_version']}"
            )
        state = str(status.get("status", "")).casefold()
        if not any(marker in state for marker in READY_MARKERS):
            raise CandidateWorkflowError(
                f"Remote baseline Dataset is not ready: {status.get('status')!r}"
            )
        for key in ("ref", "id", "dataset_ref"):
            if key in status and status[key] not in (None, "", dataset_ref):
                raise CandidateWorkflowError(
                    f"Remote baseline Dataset status differs at {key}"
                )
        return status

    exact_status()
    baseline_uploader.verify_remote_dataset(
        cli,
        dataset_ref,
        expected_manifest_sha256=context["manifest_sha256"],
    )
    final_status = exact_status()
    return {
        "dataset_ref": dataset_ref,
        "dataset_version": context["dataset_version"],
        "manifest_sha256": context["manifest_sha256"],
        "private": True,
        "status": final_status.get("status"),
    }


def verify_remote_candidate_sources(
    cli: list[str],
    *,
    kernel_ref: str,
    entry: Mapping[str, Any],
) -> None:
    with tempfile.TemporaryDirectory(prefix="bge-candidate-metadata-") as temp_dir:
        result = kaggle.run_command(
            cli + ["kernels", "pull", kernel_ref, "-p", temp_dir, "-m"],
            check=False,
        )
        if result.returncode:
            raise CandidateWorkflowError("Could not verify remote candidate metadata")
        metadata_path = Path(temp_dir) / "kernel-metadata.json"
        if not metadata_path.is_file():
            raise CandidateWorkflowError("Remote candidate metadata is missing")
        metadata = _read_json(metadata_path)
    actual = metadata.get("dataset_sources")
    expected = expected_dataset_sources(entry)
    if (
        not isinstance(actual, list)
        or len(actual) != len(expected)
        or len(set(actual)) != len(actual)
        or set(actual) != set(expected)
    ):
        raise CandidateWorkflowError(
            f"Remote candidate attachments differ: actual={actual}, expected={expected}"
        )
    if metadata.get("is_private") is not True:
        raise CandidateWorkflowError("Remote candidate kernel is not private")


def push_candidate_after_final_gates(
    cli: list[str],
    *,
    kernel_ref: str,
    entry: Mapping[str, Any],
    authority: Mapping[str, Any],
    run_timeout: int,
) -> None:
    """Push once; the remote baseline audit is the final pre-mutation action."""
    validate_staged_kernel_metadata(entry)
    baseline_launcher.confirm_remote_absence(cli, kernel_ref)
    verify_remote_baseline_dataset(cli, authority)
    stage_dir = kaggle.STAGE_ROOT / str(entry["kernel_slug"])
    result = kaggle.run_command(
        cli
        + [
            "kernels",
            "push",
            "-p",
            str(stage_dir),
            "--timeout",
            str(run_timeout),
            "--accelerator",
            "NvidiaTeslaT4",
        ]
    )
    if "not valid dataset sources" in result.stdout.casefold():
        raise CandidateWorkflowError("Kaggle rejected a candidate Dataset attachment")
    verify_remote_candidate_sources(cli, kernel_ref=kernel_ref, entry=entry)


def validate_candidate_output(
    directory: Path,
    *,
    entry: Mapping[str, Any],
    require_sheets: bool = True,
) -> dict[str, Any]:
    validator = (
        baseline_launcher.validate_run_output
        if require_sheets
        else baseline_launcher.validate_run_payload
    )
    result = validator(directory, entry=entry)
    gate_paths = list(directory.rglob(candidate_builder.BASELINE_GATE_FILENAME))
    if len(gate_paths) != 1 or gate_paths[0] != (
        directory / candidate_builder.BASELINE_GATE_FILENAME
    ):
        raise CandidateWorkflowError("Candidate has no exact root baseline gate artifact")
    gate = _read_json(gate_paths[0])
    completion = _read_json(directory / baseline_uploader.COMPLETION_FILENAME)
    context = candidate_builder.validate_baseline_context(entry["baseline_context"])
    expected_gate = {
        "status": "passed",
        "dataset_ref": context["dataset_ref"],
        "dataset_version": context["dataset_version"],
        "manifest_sha256": context["manifest_sha256"],
        "baseline_run_id": context["binding"]["baseline_run_id"],
        "baseline_identity_sha256": context["binding"]["campaign_identity_sha256"],
        "baseline_source_sha256": context["binding"]["source_sha256"],
        "ood_predictions": False,
    }
    if gate != expected_gate or completion.get("frozen_baseline_dataset") != expected_gate:
        raise CandidateWorkflowError("Candidate baseline Dataset gate differs")
    parent = candidate_builder.validate_parent_receipt(entry["parent"])
    expected_parent = {
        "run_id": parent["run_id"],
        "experiment": parent["experiment"],
        "campaign_identity_sha256": parent["campaign_identity_sha256"],
        "source_sha256": parent["source_sha256"],
        "recipe_sha256": parent["recipe_sha256"],
        "config": parent["config"],
    }
    if completion.get("stage_parent") != expected_parent:
        raise CandidateWorkflowError("Candidate completion parent differs")
    if completion.get("candidate_generator_sha256") != entry[
        "candidate_generator_sha256"
    ]:
        raise CandidateWorkflowError("Candidate generator receipt differs")
    if completion.get("baseline_comparison") not in (None, {}):
        raise CandidateWorkflowError("Raw candidate output contains a stale comparison")
    return {**result, "baseline_dataset_gate": gate, "parent_run_id": parent["run_id"]}


def _local_output(entry: Mapping[str, Any]) -> Path | None:
    directory = baseline_launcher.output_root() / str(entry["kernel_slug"])
    if not directory.exists():
        return None
    if not directory.is_dir():
        raise CandidateWorkflowError(f"Candidate output path is not a directory: {directory}")
    validate_candidate_output(directory, entry=entry)
    return directory


def download_candidate_output(
    cli: list[str],
    *,
    entry: Mapping[str, Any],
    full_download: bool,
) -> Path:
    root = baseline_launcher.output_root()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / str(entry["kernel_slug"])
    if destination.exists():
        raise CandidateWorkflowError(
            f"Refusing to replace an existing candidate output: {destination}"
        )
    kernel_ref = f"{OWNER}/{entry['kernel_slug']}"
    staging: Path | None = None
    for attempt in range(1, 4):
        candidate = Path(
            tempfile.mkdtemp(prefix=f".{entry['kernel_slug']}.download-", dir=root)
        )
        command = cli + [
            "kernels",
            "output",
            kernel_ref,
            "-p",
            str(candidate),
            "--force",
            "--page-size",
            "200",
        ]
        if not full_download:
            command.extend(["--file-pattern", SLIM_OUTPUT_PATTERN])
        result = kaggle.run_command(command, check=False)
        if result.returncode == 0:
            staging = candidate
            break
        shutil.rmtree(candidate)
        if attempt < 3:
            time.sleep(3 if attempt == 1 else 8)
    if staging is None:
        raise CandidateWorkflowError(f"Could not download candidate output: {kernel_ref}")
    try:
        validate_candidate_output(staging, entry=entry, require_sheets=False)
        baseline_launcher.retry_pending_sheets_sync(staging, kernel_ref=kernel_ref)
        validate_candidate_output(staging, entry=entry, require_sheets=True)
        staging.rename(destination)
    except Exception:
        print(f"Invalid candidate download preserved at: {staging}", file=sys.stderr)
        raise
    return destination


def execute_candidate(
    *,
    cli: list[str],
    env_file: Path,
    authority: Mapping[str, Any],
    entry: Mapping[str, Any],
    poll_interval: int,
    wait_timeout: int,
    run_timeout: int,
    full_download: bool,
) -> Path:
    existing = _local_output(entry)
    if existing is not None:
        return existing
    kernel_ref = f"{OWNER}/{entry['kernel_slug']}"
    status = baseline_launcher.remote_kernel_status(cli, kernel_ref)
    if status == "absence_unconfirmed":
        run_inner_runner(runner_command(entry, env_file=env_file))
        validate_staged_kernel_metadata(entry)
        push_candidate_after_final_gates(
            cli,
            kernel_ref=kernel_ref,
            entry=entry,
            authority=authority,
            run_timeout=run_timeout,
        )
        kaggle.wait_for_kernel(
            cli,
            kernel_ref,
            poll_interval=poll_interval,
            wait_timeout=wait_timeout,
        )
    elif status in {"queued", "running"}:
        verify_remote_candidate_sources(cli, kernel_ref=kernel_ref, entry=entry)
        kaggle.wait_for_kernel(
            cli,
            kernel_ref,
            poll_interval=poll_interval,
            wait_timeout=wait_timeout,
        )
    elif status in kaggle.TERMINAL_FAILURE:
        kaggle.run_command(cli + ["kernels", "logs", kernel_ref], check=False)
        raise CandidateWorkflowError(
            f"Candidate {kernel_ref} is terminally failed; resubmission is forbidden"
        )
    elif status in kaggle.TERMINAL_SUCCESS:
        verify_remote_candidate_sources(cli, kernel_ref=kernel_ref, entry=entry)
    else:
        raise CandidateWorkflowError(f"Unexpected candidate kernel status: {status!r}")
    return download_candidate_output(
        cli,
        entry=entry,
        full_download=full_download,
    )


def build_lr_entries(
    authority: Mapping[str, Any],
    *,
    write: bool,
) -> list[dict[str, Any]]:
    specs = [candidate_builder.lr_variant_spec(spec["key"]) for spec in candidate_builder.LR_SPECS]
    parents = {spec["key"]: authority["baseline_parent"] for spec in specs}
    entries = candidate_builder.build_candidate_campaign(
        owner=OWNER,
        baseline_context=authority["context"],
        baseline_entry=authority["entry"],
        specs=specs,
        parents=parents,
        write=write,
    )
    if [entry["key"] for entry in entries] != ["lr1e5", "lr4e5"]:
        raise CandidateWorkflowError("LR execution order differs from 1e-5 then 4e-5")
    return entries


def _parent_from_candidate(
    *,
    entry: Mapping[str, Any],
    completion: Mapping[str, Any],
) -> dict[str, Any]:
    return candidate_builder.validate_parent_receipt(
        {
            "run_id": completion["run_id"],
            "experiment": completion["experiment"],
            "campaign_identity_sha256": completion["campaign_identity_sha256"],
            "source_sha256": completion["code_bundle_sha256"],
            "recipe_sha256": completion["frozen_recipe_sha256"],
            "checkpoint_manifest_sha256": completion[
                "initial_checkpoint_manifest_sha256"
            ],
            "checkpoint_model_sha256": completion[
                "initial_checkpoint_model_sha256"
            ],
            "validation_manifest_sha256": completion["validation_manifest_sha256"],
            "loss_hook_sha256": completion["loss_hook_sha256"],
            "config": entry["expected_config"],
        }
    )


def _baseline_dataset_receipt(authority: Mapping[str, Any]) -> dict[str, Any]:
    context = candidate_builder.validate_baseline_context(authority["context"])
    return {
        "dataset_ref": context["dataset_ref"],
        "dataset_version": context["dataset_version"],
        "manifest_sha256": context["manifest_sha256"],
        "baseline_run_id": context["binding"]["baseline_run_id"],
        "campaign_identity_sha256": context["binding"]["campaign_identity_sha256"],
        "source_sha256": context["binding"]["source_sha256"],
    }


def validate_augmented_completion(
    completion: Mapping[str, Any],
    *,
    expected_baseline_run_id: str,
) -> dict[str, Any]:
    comparison = completion.get("baseline_comparison")
    if not isinstance(comparison, Mapping):
        raise CandidateWorkflowError("Augmented completion has no paired comparison")
    if comparison.get("baseline_run_id") != expected_baseline_run_id:
        raise CandidateWorkflowError("Augmented comparison uses another parent run")
    splits = comparison.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"iid", "hard", "ood"}:
        raise CandidateWorkflowError("Augmented comparison split contract differs")
    ood = splits["ood"]
    if not isinstance(ood, Mapping) or ood != comparator._ood_result():
        raise CandidateWorkflowError("Augmented comparison fabricated OOD evidence")
    from src.google_sheets_logger import COMPARISON_HEADERS, build_comparison_row

    projection = dict(
        zip(COMPARISON_HEADERS, build_comparison_row(completion), strict=True)
    )
    exact = {
        "baseline_run_id": expected_baseline_run_id,
        "ood_macro_ap": -1.0,
        "ood_delta": "",
        "ood_p_value": "",
        "ood_p_holm": "",
        "ood_ci95_low": "",
        "ood_ci95_high": "",
        "comparison_status": "ready_ood_disabled",
        "comparison_method": "paired_component_permutation",
    }
    for key, expected in exact.items():
        if projection.get(key) != expected:
            raise CandidateWorkflowError(f"Augmented Sheet projection differs at {key}")
    return projection


def _local_credentials_json() -> str:
    from push_google_sheets_credentials_dataset import DEFAULT_KEY_PATH
    from src.google_sheets_logger import load_service_account_info

    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if not raw:
        configured = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_PATH", "").strip()
        path = (
            Path(configured).expanduser().resolve()
            if configured
            else DEFAULT_KEY_PATH.expanduser().resolve()
        )
        raw = path.read_text(encoding="utf-8")
    load_service_account_info(raw)
    return raw


def sync_augmented_completion(
    completion: Mapping[str, Any],
    *,
    output_dir: Path,
    expected_baseline_run_id: str,
) -> dict[str, Any]:
    validate_augmented_completion(
        completion,
        expected_baseline_run_id=expected_baseline_run_id,
    )
    marker_path = output_dir / COMPARISON_SYNC_FILENAME
    if marker_path.is_file():
        marker = _read_json(marker_path)
        validate_comparison_sync_marker(marker, completion=completion)
        return marker

    from src.google_sheets_logger import sync_experiment

    result = sync_experiment(
        spreadsheet_id=shared.EXPERIMENT_SPREADSHEET_ID,
        service_account_json=_local_credentials_json(),
        completion=completion,
    )
    completion_sha256 = candidate_builder.canonical_sha256(completion)
    marker = {
        "status": "synced_comparison",
        **result,
        "completion_canonical_sha256": completion_sha256,
    }
    validate_comparison_sync_marker(marker, completion=completion)
    _write_json(marker_path, marker)
    return marker


def validate_comparison_sync_marker(
    marker: Mapping[str, Any],
    *,
    completion: Mapping[str, Any],
) -> None:
    expected = {
        "status": "synced_comparison",
        "run_id": completion["run_id"],
        "experiment_group": "sft",
        "comparison_sheet": "sft_exps",
        "spreadsheet_id": shared.EXPERIMENT_SPREADSHEET_ID,
        "completion_canonical_sha256": candidate_builder.canonical_sha256(completion),
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise CandidateWorkflowError(f"Comparison Sheets marker differs at {key}")


def summarize_lr_stage(
    authority: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    directories: Sequence[Path],
    *,
    output_dir: Path,
    sync_sheets: bool,
    permutations: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    if len(entries) != 2 or len(directories) != 2:
        raise CandidateWorkflowError("The complete LR family requires exactly two candidates")
    for entry, directory in zip(entries, directories, strict=True):
        validate_candidate_output(directory, entry=entry)
    output_dir = comparator.validate_output_isolation(
        output_dir,
        [authority["source_dir"], authority["stage_dir"], *directories],
    )
    planned = [str(entry["experiment"]) for entry in entries]
    tie_order = [authority["entry"]["experiment"], *planned]
    summary = comparator.summarize_candidate_family(
        authority["stage_dir"],
        directories,
        planned_experiments=planned,
        family_name=LR_FAMILY_NAME,
        tie_break_order=tie_order,
        practical_tie_margin=comparator.PRACTICAL_TIE_MARGIN,
        permutations=permutations,
        bootstrap_resamples=bootstrap_resamples,
        seed=42,
    )
    outputs = comparator.materialize_summary(summary, output_dir)
    selected_experiment = summary["selection"]["selected_with_practical_tie_break"]
    if selected_experiment == authority["entry"]["experiment"]:
        selected_parent = authority["baseline_parent"]
        selected_source = "baseline"
        selected_directory = authority["source_dir"]
    else:
        index = planned.index(selected_experiment)
        completion = _read_json(directories[index] / baseline_uploader.COMPLETION_FILENAME)
        selected_parent = _parent_from_candidate(entry=entries[index], completion=completion)
        selected_source = "candidate"
        selected_directory = directories[index]
    summary_path = Path(outputs["summary"])
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "campaign": baseline_builder.CAMPAIGN,
        "stage": "lr_log_line",
        "family_name": LR_FAMILY_NAME,
        "planned_experiments": planned,
        "execution_order": planned,
        "primary_split": "iid",
        "diagnostic_splits": ["hard"],
        "practical_tie_margin": comparator.PRACTICAL_TIE_MARGIN,
        "ood": {"macro_average_precision": -1.0, "comparison": None},
        "frozen_baseline_dataset": _baseline_dataset_receipt(authority),
        "family_summary_path": str(summary_path),
        "family_summary_sha256": comparator.sha256_file(summary_path),
        "selected_source": selected_source,
        "selected_directory": str(selected_directory),
        "selected_parent": selected_parent,
        "comparison_sheets_synced": sync_sheets,
        "comparison_sync_markers": {},
        "boundary_expansion": "deferred",
    }
    if sync_sheets:
        sync_markers: dict[str, dict[str, Any]] = {}
        for experiment in planned:
            raw = summary["candidate_outputs"][experiment]
            candidate_output = Path(outputs["candidates"][experiment]["completion"]).parent
            marker = sync_augmented_completion(
                raw["augmented_completion"],
                output_dir=candidate_output,
                expected_baseline_run_id=authority["baseline_parent"]["run_id"],
            )
            marker_path = candidate_output / COMPARISON_SYNC_FILENAME
            sync_markers[experiment] = {
                "path": str(marker_path),
                "sha256": comparator.sha256_file(marker_path),
                "completion_canonical_sha256": marker[
                    "completion_canonical_sha256"
                ],
            }
        receipt["comparison_sync_markers"] = sync_markers
    receipt_path = output_dir / LR_RECEIPT_FILENAME
    _write_json(receipt_path, receipt)
    return {
        "summary": summary,
        "outputs": outputs,
        "receipt": receipt,
        "receipt_path": receipt_path,
    }


def load_lr_selection_receipt(
    path: Path,
    *,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_path = path.expanduser().resolve(strict=True)
    receipt = _read_json(receipt_path)
    exact_keys = {
        "schema_version",
        "status",
        "campaign",
        "stage",
        "family_name",
        "planned_experiments",
        "execution_order",
        "primary_split",
        "diagnostic_splits",
        "practical_tie_margin",
        "ood",
        "frozen_baseline_dataset",
        "family_summary_path",
        "family_summary_sha256",
        "selected_source",
        "selected_directory",
        "selected_parent",
        "comparison_sheets_synced",
        "comparison_sync_markers",
        "boundary_expansion",
    }
    if set(receipt) != exact_keys:
        raise CandidateWorkflowError("LR selection receipt fields differ")
    expected = {
        "schema_version": 1,
        "status": "complete",
        "campaign": baseline_builder.CAMPAIGN,
        "stage": "lr_log_line",
        "family_name": LR_FAMILY_NAME,
        "primary_split": "iid",
        "diagnostic_splits": ["hard"],
        "practical_tie_margin": comparator.PRACTICAL_TIE_MARGIN,
        "ood": {"macro_average_precision": -1.0, "comparison": None},
        "frozen_baseline_dataset": _baseline_dataset_receipt(authority),
        "boundary_expansion": "deferred",
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise CandidateWorkflowError(f"LR selection receipt differs at {key}")
    expected_experiments = [spec["experiment"] for spec in candidate_builder.LR_SPECS]
    if receipt["planned_experiments"] != expected_experiments or receipt[
        "execution_order"
    ] != expected_experiments:
        raise CandidateWorkflowError("LR receipt does not cover the exact ordered family")
    summary_path = Path(receipt["family_summary_path"]).expanduser().resolve(strict=True)
    expected_summary_path = receipt_path.parent / "family_summary.json"
    if summary_path != expected_summary_path:
        raise CandidateWorkflowError("LR family summary is not beside its receipt")
    if comparator.sha256_file(summary_path) != _require_hash(
        receipt["family_summary_sha256"], "LR family summary SHA-256"
    ):
        raise CandidateWorkflowError("LR family summary changed after selection")
    summary = _read_json(summary_path)
    parent = candidate_builder.validate_parent_receipt(receipt["selected_parent"])
    selected = summary.get("selection", {}).get("selected_with_practical_tie_break")
    if selected != parent["experiment"]:
        raise CandidateWorkflowError("LR receipt parent differs from family selection")
    if (
        summary.get("schema_version") != 1
        or summary.get("status") != "complete"
        or summary.get("campaign") != baseline_builder.CAMPAIGN
        or summary.get("family_name") != LR_FAMILY_NAME
        or summary.get("planned_family_complete") is not True
        or summary.get("planned_experiments") != expected_experiments
        or summary.get("primary_split") != "iid"
        or summary.get("diagnostic_splits") != ["hard"]
        or summary.get("practical_tie_margin") != comparator.PRACTICAL_TIE_MARGIN
    ):
        raise CandidateWorkflowError("LR family summary protocol differs")
    expected_baseline = {
        "run_id": authority["baseline_parent"]["run_id"],
        "experiment": authority["baseline_parent"]["experiment"],
        "manifest_sha256": authority["context"]["manifest_sha256"],
        "campaign_identity_sha256": authority["baseline_parent"][
            "campaign_identity_sha256"
        ],
        "source_sha256": authority["baseline_parent"]["source_sha256"],
        "recipe_sha256": authority["baseline_parent"]["recipe_sha256"],
        "checkpoint_manifest_sha256": authority["baseline_parent"][
            "checkpoint_manifest_sha256"
        ],
    }
    summary_baseline = summary.get("baseline")
    if not isinstance(summary_baseline, Mapping) or any(
        summary_baseline.get(key) != value
        for key, value in expected_baseline.items()
    ):
        raise CandidateWorkflowError("LR family summary baseline binding differs")

    entries = build_lr_entries(authority, write=False)
    entries_by_experiment = {entry["experiment"]: entry for entry in entries}
    summary_candidates = summary.get("candidates")
    if not isinstance(summary_candidates, Mapping) or set(summary_candidates) != set(
        expected_experiments
    ):
        raise CandidateWorkflowError("LR family summary candidate set differs")
    raw_parents: dict[str, dict[str, Any]] = {}
    comparison_root = summary_path.parent
    declared_markers = receipt.get("comparison_sync_markers")
    if not isinstance(declared_markers, Mapping):
        raise CandidateWorkflowError("LR comparison marker ledger is invalid")
    sheets_synced = receipt.get("comparison_sheets_synced")
    expected_marker_members = set(expected_experiments) if sheets_synced else set()
    if set(declared_markers) != expected_marker_members:
        raise CandidateWorkflowError("LR comparison marker family differs")
    for experiment in expected_experiments:
        entry = entries_by_experiment[experiment]
        raw_dir = baseline_launcher.output_root() / str(entry["kernel_slug"])
        validate_candidate_output(raw_dir, entry=entry)
        raw_completion = _read_json(raw_dir / baseline_uploader.COMPLETION_FILENAME)
        raw_parent = _parent_from_candidate(entry=entry, completion=raw_completion)
        raw_parents[experiment] = raw_parent
        summary_candidate = summary_candidates[experiment]
        expected_candidate_binding = {
            "run_id": raw_parent["run_id"],
            "campaign_identity_sha256": raw_parent["campaign_identity_sha256"],
            "source_sha256": raw_parent["source_sha256"],
            "recipe_sha256": raw_parent["recipe_sha256"],
        }
        if not isinstance(summary_candidate, Mapping) or any(
            summary_candidate.get(key) != value
            for key, value in expected_candidate_binding.items()
        ):
            raise CandidateWorkflowError(
                f"LR family summary binding differs for {experiment}"
            )
        comparison_dir = comparison_root / experiment
        augmented_path = comparison_dir / "completion_with_comparison.json"
        augmented = _read_json(augmented_path)
        without_comparison = dict(augmented)
        comparison = without_comparison.pop("baseline_comparison", None)
        if without_comparison != raw_completion or not isinstance(comparison, Mapping):
            raise CandidateWorkflowError(
                f"LR augmented completion differs from raw output: {experiment}"
            )
        validate_augmented_completion(
            augmented,
            expected_baseline_run_id=authority["baseline_parent"]["run_id"],
        )
        if (
            comparison.get("candidate_run_id") != raw_parent["run_id"]
            or comparison.get("candidate_experiment") != experiment
            or comparison.get("baseline_manifest_sha256")
            != authority["context"]["manifest_sha256"]
            or comparison.get("holm_family") != LR_FAMILY_NAME
            or comparison.get("holm_family_members") != expected_experiments
        ):
            raise CandidateWorkflowError(
                f"LR augmented comparison binding differs: {experiment}"
            )
        marker_path = comparison_dir / COMPARISON_SYNC_FILENAME
        if sheets_synced:
            declaration = declared_markers[experiment]
            if not isinstance(declaration, Mapping) or declaration.get("path") != str(
                marker_path
            ):
                raise CandidateWorkflowError(
                    f"LR comparison marker path differs: {experiment}"
                )
            if comparator.sha256_file(marker_path) != _require_hash(
                declaration.get("sha256"),
                f"LR comparison marker SHA-256 for {experiment}",
            ):
                raise CandidateWorkflowError(
                    f"LR comparison marker changed: {experiment}"
                )
            marker = _read_json(marker_path)
            validate_comparison_sync_marker(marker, completion=augmented)
            if declaration.get("completion_canonical_sha256") != marker.get(
                "completion_canonical_sha256"
            ):
                raise CandidateWorkflowError(
                    f"LR comparison marker completion differs: {experiment}"
                )
        elif marker_path.exists():
            raise CandidateWorkflowError(
                f"Unsynced LR receipt has a stale comparison marker: {experiment}"
            )

    source = receipt.get("selected_source")
    selected_dir = Path(receipt["selected_directory"]).expanduser().resolve(strict=True)
    if source == "baseline":
        if parent != authority["baseline_parent"] or selected_dir != authority["source_dir"]:
            raise CandidateWorkflowError("LR receipt baseline parent differs")
    elif source == "candidate":
        selected_entry = entries_by_experiment.get(parent["experiment"])
        if selected_entry is None:
            raise CandidateWorkflowError("LR selected candidate is outside the family")
        expected_selected_dir = baseline_launcher.output_root() / str(
            selected_entry["kernel_slug"]
        )
        if selected_dir != expected_selected_dir:
            raise CandidateWorkflowError("LR selected candidate directory differs")
        if raw_parents[parent["experiment"]] != parent:
            raise CandidateWorkflowError("LR selected candidate artifact differs")
    else:
        raise CandidateWorkflowError("LR selected_source is invalid")
    if not isinstance(sheets_synced, bool):
        raise CandidateWorkflowError("LR comparison Sheets receipt is invalid")
    return {
        **receipt,
        "selected_parent": parent,
        "selected_directory": selected_dir,
        "_receipt_path": receipt_path,
    }


def build_e2_entry(
    authority: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    write: bool,
) -> dict[str, Any]:
    parent = candidate_builder.validate_parent_receipt(receipt["selected_parent"])
    spec = candidate_builder.e2_variant_spec(parent)
    entries = candidate_builder.build_candidate_campaign(
        owner=OWNER,
        baseline_context=authority["context"],
        baseline_entry=authority["entry"],
        specs=[spec],
        parents={"e2": parent},
        write=write,
    )
    if len(entries) != 1 or entries[0]["parent"] != parent:
        raise CandidateWorkflowError("e2 was not bound to the selected e1 parent")
    return entries[0]


def _paired_epoch_split(
    anchor: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    split: str,
    permutations: int,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    result = comparator.compare_prediction_frames(
        comparator.read_prediction_artifact(anchor["predictions"][split]),
        comparator.read_prediction_artifact(candidate["predictions"][split]),
        permutations=permutations,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
    )
    if result["examples"] != comparator.EXPECTED_ROWS[split]:
        raise CandidateWorkflowError(f"Epoch comparison has unexpected {split} rows")
    anchor_ap = comparator._metric_from_report(
        anchor["report"], split=split, label="selected e1"
    )
    candidate_ap = comparator._metric_from_report(
        candidate["report"], split=split, label="e2"
    )
    if not math.isclose(
        float(result["baseline_macro_average_precision"]),
        anchor_ap,
        abs_tol=1e-12,
        rel_tol=0,
    ) or not math.isclose(
        float(result["candidate_macro_average_precision"]),
        candidate_ap,
        abs_tol=1e-12,
        rel_tol=0,
    ):
        raise CandidateWorkflowError(f"Epoch {split} report/predictions differ")
    result["p_value_holm"] = result["p_value"]
    result["holm_family"] = EPOCH_FAMILY_NAME
    result["holm_family_size"] = 1
    return result


def summarize_epoch_stage(
    authority: Mapping[str, Any],
    lr_receipt: Mapping[str, Any],
    entry: Mapping[str, Any],
    directory: Path,
    *,
    output_dir: Path,
    sync_sheets: bool,
    permutations: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    validate_candidate_output(directory, entry=entry)
    output_dir = comparator.validate_output_isolation(
        output_dir,
        [
            authority["source_dir"],
            authority["stage_dir"],
            Path(lr_receipt["selected_directory"]),
            directory,
        ],
    )
    baseline = comparator.load_frozen_baseline(authority["stage_dir"])
    parent = candidate_builder.validate_parent_receipt(lr_receipt["selected_parent"])
    if parent["run_id"] == baseline["run_id"]:
        anchor = baseline
    else:
        anchor = comparator.load_candidate(
            Path(lr_receipt["selected_directory"]),
            baseline=baseline,
        )
    if anchor["run_id"] != parent["run_id"] or anchor["experiment"] != parent[
        "experiment"
    ]:
        raise CandidateWorkflowError("Epoch anchor differs from selected e1 receipt")
    e2 = comparator.load_candidate(directory, baseline=baseline)
    split_results = {
        split: _paired_epoch_split(
            anchor,
            e2,
            split=split,
            permutations=permutations,
            bootstrap_resamples=bootstrap_resamples,
            seed=42 + index,
        )
        for index, split in enumerate(comparator.SPLITS)
    }
    iid_delta = float(split_results["iid"]["delta_macro_average_precision"])
    select_e2 = iid_delta > comparator.PRACTICAL_TIE_MARGIN
    comparison = {
        "schema_version": 1,
        "status": "ready_ood_disabled",
        "baseline_run_id": anchor["run_id"],
        "candidate_run_id": e2["run_id"],
        "baseline_experiment": anchor["experiment"],
        "candidate_experiment": e2["experiment"],
        "baseline_manifest_sha256": baseline["manifest_sha256"],
        "method": "paired_component_permutation",
        "confidence_interval_method": "paired_component_bootstrap_percentile",
        "multiple_testing_correction": "holm_within_planned_candidate_family_per_split",
        "holm_family": EPOCH_FAMILY_NAME,
        "holm_family_members": [e2["experiment"]],
        "primary_split": "iid",
        "diagnostic_splits": ["hard"],
        "practical_tie_margin": comparator.PRACTICAL_TIE_MARGIN,
        "iid_practical_relation": comparator._practical_relation(
            iid_delta, comparator.PRACTICAL_TIE_MARGIN
        ),
        "ood_policy": "disabled_train_contaminated_no_paired_comparison",
        "splits": {
            "iid": split_results["iid"],
            "hard": split_results["hard"],
            "ood": comparator._ood_result(),
        },
    }
    augmented = {**e2["completion"], "experiment_group": "sft", "baseline_comparison": comparison}
    validate_augmented_completion(augmented, expected_baseline_run_id=anchor["run_id"])
    summary = {
        "schema_version": 1,
        "status": "complete",
        "campaign": baseline_builder.CAMPAIGN,
        "family_name": EPOCH_FAMILY_NAME,
        "primary_split": "iid",
        "diagnostic_splits": ["hard"],
        "hard_used_for_selection": False,
        "ood_used_for_selection": False,
        "practical_tie_margin": comparator.PRACTICAL_TIE_MARGIN,
        "anchor": parent,
        "candidate": {
            "run_id": e2["run_id"],
            "experiment": e2["experiment"],
            "recipe_sha256": e2["completion"]["frozen_recipe_sha256"],
        },
        "splits": comparison["splits"],
        "selection": {
            "selected_experiment": e2["experiment"] if select_e2 else anchor["experiment"],
            "selected_run_id": e2["run_id"] if select_e2 else anchor["run_id"],
            "selected_epoch": 2 if select_e2 else 1,
            "rule": "select e2 iff paired IID delta > 0.002",
        },
    }
    summary_path = output_dir / "epoch_family_summary.json"
    comparison_path = output_dir / "baseline_comparison.json"
    completion_path = output_dir / "completion_with_comparison.json"
    _write_json(summary_path, summary)
    _write_json(comparison_path, comparison)
    _write_json(completion_path, augmented)
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "campaign": baseline_builder.CAMPAIGN,
        "stage": "epoch_line",
        "family_name": EPOCH_FAMILY_NAME,
        "frozen_baseline_dataset": _baseline_dataset_receipt(authority),
        "lr_selection_receipt_sha256": comparator.sha256_file(
            Path(lr_receipt["_receipt_path"])
        ),
        "parent": parent,
        "candidate_run_id": e2["run_id"],
        "candidate_experiment": e2["experiment"],
        "comparison_sheets_synced": sync_sheets,
        "comparison_sync_marker": None,
        "epoch_summary_path": str(summary_path),
        "epoch_summary_sha256": comparator.sha256_file(summary_path),
        "selection": summary["selection"],
        "epoch_3": "deferred",
    }
    if sync_sheets:
        marker = sync_augmented_completion(
            augmented,
            output_dir=output_dir,
            expected_baseline_run_id=anchor["run_id"],
        )
        marker_path = output_dir / COMPARISON_SYNC_FILENAME
        receipt["comparison_sync_marker"] = {
            "path": str(marker_path),
            "sha256": comparator.sha256_file(marker_path),
            "completion_canonical_sha256": marker[
                "completion_canonical_sha256"
            ],
        }
    receipt_path = output_dir / EPOCH_RECEIPT_FILENAME
    _write_json(receipt_path, receipt)
    return {
        "summary": summary,
        "comparison": comparison,
        "augmented_completion": augmented,
        "receipt": receipt,
        "receipt_path": receipt_path,
    }


def _stage_entries(entries: Sequence[Mapping[str, Any]], env_file: Path) -> None:
    for entry in entries:
        run_inner_runner(runner_command(entry, env_file=env_file))
        validate_staged_kernel_metadata(entry)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guarded post-baseline BGE SFT LR/e2 candidate workflow"
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--stage", choices=("lr", "e2"), default="lr")
    parser.add_argument("--baseline-dataset-version", type=int)
    parser.add_argument("--baseline-manifest-sha256")
    parser.add_argument("--baseline-source-dir", type=Path)
    parser.add_argument("--baseline-stage-dir", type=Path, default=baseline_uploader.STAGE_DIR)
    parser.add_argument("--lr-receipt", type=Path, default=DEFAULT_LR_RECEIPT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--permutations", type=int, default=2_000)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--full-download", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="stage locally; no Kaggle calls")
    mode.add_argument(
        "--summarize-local",
        action="store_true",
        help="compare already validated local outputs; no Kaggle or Sheets calls",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="explicitly permit sequential candidate kernel and Sheets mutations",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not (args.dry_run or args.summarize_local or args.execute):
        print(json.dumps(plan(), ensure_ascii=False, indent=2))
        return 0
    if args.baseline_dataset_version is None:
        raise SystemExit("--baseline-dataset-version is required outside plan-only mode")
    if args.baseline_manifest_sha256 is None:
        raise SystemExit("--baseline-manifest-sha256 is required outside plan-only mode")
    env_file = args.env_file if args.env_file.is_absolute() else ROOT / args.env_file
    kaggle.load_dotenv(env_file)
    owner = os.getenv("KAGGLE_USERNAME", OWNER).strip()
    owner = kaggle.validate_slug(owner, "KAGGLE_USERNAME")
    authority = load_local_baseline_authority(
        owner=owner,
        dataset_version=args.baseline_dataset_version,
        source_dir=args.baseline_source_dir,
        stage_dir=args.baseline_stage_dir,
        expected_manifest_sha256=args.baseline_manifest_sha256,
    )
    report_root = (
        args.report_root.expanduser().resolve()
        if args.report_root.is_absolute()
        else (ROOT / args.report_root).resolve()
    )
    allowed_report_root = (ROOT / "reports").resolve()
    if report_root != allowed_report_root and allowed_report_root not in report_root.parents:
        raise CandidateWorkflowError(
            f"Candidate reports must remain below {allowed_report_root}"
        )
    if args.stage == "lr":
        entries = build_lr_entries(authority, write=True)
    else:
        lr_receipt = load_lr_selection_receipt(args.lr_receipt, authority=authority)
        if args.execute and lr_receipt["comparison_sheets_synced"] is not True:
            raise CandidateWorkflowError(
                "Live e2 requires an LR receipt with exact comparison Sheets markers"
            )
        entries = [build_e2_entry(authority, lr_receipt, write=True)]
    if args.dry_run:
        _stage_entries(entries, env_file)
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "stage": args.stage,
                    "baseline_dataset": _baseline_dataset_receipt(authority),
                    "staged": [entry["kernel_slug"] for entry in entries],
                    "kaggle_contacted": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.execute:
        baseline_launcher._enforce_live_environment()
        if not os.getenv("KAGGLE_API_TOKEN", "").strip():
            raise SystemExit("Set KAGGLE_API_TOKEN in .env")
        cli = kaggle.kaggle_command()
        poll_interval = kaggle.env_int("KAGGLE_POLL_INTERVAL_SECONDS", 30, minimum=5)
        wait_timeout = kaggle.env_int("KAGGLE_WAIT_TIMEOUT_SECONDS", 45000, minimum=60)
        run_timeout = kaggle.env_int("KAGGLE_RUN_TIMEOUT_SECONDS", 43200, minimum=60)
    else:
        cli = []
        poll_interval = wait_timeout = run_timeout = 0

    with baseline_launcher.exclusive_campaign_lock():
        directories: list[Path] = []
        for entry in entries:
            if args.execute:
                directory = execute_candidate(
                    cli=cli,
                    env_file=env_file,
                    authority=authority,
                    entry=entry,
                    poll_interval=poll_interval,
                    wait_timeout=wait_timeout,
                    run_timeout=run_timeout,
                    full_download=args.full_download,
                )
            else:
                directory = _local_output(entry)
                if directory is None:
                    raise CandidateWorkflowError(
                        f"Local candidate output is missing: {entry['kernel_slug']}"
                    )
            directories.append(directory)
        if args.stage == "lr":
            result = summarize_lr_stage(
                authority,
                entries,
                directories,
                output_dir=report_root / "lr",
                sync_sheets=args.execute,
                permutations=args.permutations,
                bootstrap_resamples=args.bootstrap_resamples,
            )
        else:
            result = summarize_epoch_stage(
                authority,
                lr_receipt,
                entries[0],
                directories[0],
                output_dir=report_root / "e2",
                sync_sheets=args.execute,
                permutations=args.permutations,
                bootstrap_resamples=args.bootstrap_resamples,
            )
    print(
        json.dumps(
            {
                "mode": "execute" if args.execute else "summarize_local",
                "stage": args.stage,
                "receipt": str(result["receipt_path"]),
                "selection": result["summary"]["selection"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
