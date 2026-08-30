#!/usr/bin/env python3
"""Build guarded BGE SFT LR and parented-epoch candidate notebooks.

This module is intentionally outside the frozen baseline source ledger.  It
reuses the exact baseline runtime cells and runtime sources without modifying
them, while binding each candidate executable to the already frozen private
baseline Dataset and to an explicit stage parent receipt.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import create_bge_2ep_sft_notebooks as baseline_builder
import create_cross_encoder_training_notebook as cross_builder


TEMPLATE = "bge_2ep_sft_candidate_v1"
DEFAULT_OUTPUT_DIR = ROOT / "notebooks" / TEMPLATE
BASELINE_MANIFEST_FILENAME = "bge_2ep_sft_baseline_manifest.json"
BASELINE_GATE_FILENAME = "bge_baseline_dataset_gate.json"
LR_SPECS: tuple[dict[str, Any], ...] = (
    {
        "key": "lr1e5",
        "stage": "lr_log_line",
        "experiment": "bge2_sft_oodtrain_e1_lr1e5_v1",
        "slug_token": "lr1e5",
        "learning_rate": 1e-5,
        "epochs": 1,
    },
    {
        "key": "lr4e5",
        "stage": "lr_log_line",
        "experiment": "bge2_sft_oodtrain_e1_lr4e5_v1",
        "slug_token": "lr4e5",
        "learning_rate": 4e-5,
        "epochs": 1,
    },
)
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
RUN_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
EXPERIMENT_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")


class CandidateBuildError(ValueError):
    """Raised when a candidate cannot be bound to the frozen campaign."""


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def generator_sha256() -> str:
    return file_sha256(Path(__file__).resolve())


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or HASH_PATTERN.fullmatch(value) is None:
        raise CandidateBuildError(f"{label} is not an exact lowercase SHA-256")
    return value


def _require_run_id(value: object, label: str) -> str:
    if not isinstance(value, str) or RUN_ID_PATTERN.fullmatch(value) is None:
        raise CandidateBuildError(f"{label} is not a 32-hex run_id")
    return value


def validate_baseline_context(context: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the exact local/remote baseline Dataset authority."""
    dataset_ref = str(context.get("dataset_ref", "")).strip()
    dataset_slug = str(context.get("dataset_slug", "")).strip()
    dataset_version = context.get("dataset_version")
    manifest = context.get("manifest")
    binding = context.get("binding")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*/[a-z0-9][a-z0-9-]*", dataset_ref):
        raise CandidateBuildError("Invalid frozen baseline Dataset reference")
    if dataset_slug != dataset_ref.rsplit("/", 1)[-1]:
        raise CandidateBuildError("Frozen baseline Dataset slug/reference differ")
    if (
        isinstance(dataset_version, bool)
        or not isinstance(dataset_version, int)
        or dataset_version < 1
    ):
        raise CandidateBuildError("Frozen baseline Dataset version must be positive")
    if not isinstance(manifest, Mapping) or not isinstance(binding, Mapping):
        raise CandidateBuildError("Frozen baseline Dataset manifest/binding is missing")
    if manifest.get("schema_version") != 1 or manifest.get("is_private") is not True:
        raise CandidateBuildError("Frozen baseline Dataset manifest is not private v1")
    if manifest.get("dataset") != dataset_ref or manifest.get("binding") != binding:
        raise CandidateBuildError("Frozen baseline Dataset manifest identity differs")
    if manifest.get("evaluated_splits") != ["iid", "hard"]:
        raise CandidateBuildError("Frozen baseline Dataset split contract differs")
    ood = manifest.get("ood")
    if (
        not isinstance(ood, Mapping)
        or ood.get("evaluated") is not False
        or ood.get("metric_sentinel") != -1.0
        or ood.get("comparison") is not None
        or ood.get("prediction_file") is not None
    ):
        raise CandidateBuildError("Frozen baseline Dataset fabricated OOD evidence")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != {
        "notebook_completed.json",
        "iid_validation_predictions.parquet",
        "hard_validation_predictions.parquet",
    }:
        raise CandidateBuildError("Frozen baseline Dataset file ledger differs")
    for filename, raw in files.items():
        if not isinstance(raw, Mapping):
            raise CandidateBuildError(f"Invalid baseline file declaration: {filename}")
        size = raw.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise CandidateBuildError(f"Invalid baseline file bytes: {filename}")
        _require_hash(raw.get("sha256"), f"baseline files.{filename}.sha256")
    exact_binding_keys = {
        "baseline_run_id",
        "baseline_experiment",
        "campaign",
        "campaign_identity_sha256",
        "source_sha256",
        "recipe_sha256",
        "executable_cells_sha256",
        "loss_hook_sha256",
        "checkpoint_manifest_sha256",
        "checkpoint_model_sha256",
        "validation_manifest_sha256",
    }
    if set(binding) != exact_binding_keys:
        raise CandidateBuildError("Frozen baseline binding has unexpected fields")
    _require_run_id(binding.get("baseline_run_id"), "baseline_run_id")
    if binding.get("campaign") != baseline_builder.CAMPAIGN:
        raise CandidateBuildError("Frozen baseline belongs to another campaign")
    for key in exact_binding_keys - {
        "baseline_run_id",
        "baseline_experiment",
        "campaign",
    }:
        _require_hash(binding.get(key), f"baseline binding.{key}")
    manifest_sha256 = _require_hash(
        context.get("manifest_sha256"), "baseline manifest_sha256"
    )
    if canonical_sha256(manifest) != str(
        context.get("manifest_canonical_sha256", canonical_sha256(manifest))
    ):
        raise CandidateBuildError("Frozen baseline canonical manifest hash differs")
    return {
        "dataset_ref": dataset_ref,
        "dataset_slug": dataset_slug,
        "dataset_version": dataset_version,
        "manifest_filename": BASELINE_MANIFEST_FILENAME,
        "manifest_sha256": manifest_sha256,
        "manifest_canonical_sha256": canonical_sha256(manifest),
        "manifest": dict(manifest),
        "binding": dict(binding),
    }


def validate_parent_receipt(parent: Mapping[str, Any]) -> dict[str, Any]:
    exact_keys = {
        "run_id",
        "experiment",
        "campaign_identity_sha256",
        "source_sha256",
        "recipe_sha256",
        "checkpoint_manifest_sha256",
        "checkpoint_model_sha256",
        "validation_manifest_sha256",
        "loss_hook_sha256",
        "config",
    }
    if set(parent) != exact_keys:
        raise CandidateBuildError("Stage parent receipt fields differ")
    _require_run_id(parent.get("run_id"), "parent.run_id")
    experiment = str(parent.get("experiment", ""))
    if EXPERIMENT_PATTERN.fullmatch(experiment) is None:
        raise CandidateBuildError("Stage parent experiment is invalid")
    for key in exact_keys - {"run_id", "experiment", "config"}:
        _require_hash(parent.get(key), f"parent.{key}")
    config = parent.get("config")
    if not isinstance(config, Mapping):
        raise CandidateBuildError("Stage parent config is missing")
    if canonical_sha256(config) != parent.get("recipe_sha256"):
        raise CandidateBuildError("Stage parent config/recipe SHA differ")
    if config.get("epochs") != 1 or config.get("seed") != 42:
        raise CandidateBuildError("Stage parent must be a seed-42 epoch-1 recipe")
    if config.get("learning_rate") not in {1e-5, 2e-5, 4e-5}:
        raise CandidateBuildError("Stage parent LR is outside the core log line")
    return {**dict(parent), "config": dict(config)}


def baseline_parent_receipt(
    baseline_context: Mapping[str, Any],
    baseline_entry: Mapping[str, Any],
) -> dict[str, Any]:
    context = validate_baseline_context(baseline_context)
    binding = context["binding"]
    expected = {
        "experiment": binding["baseline_experiment"],
        "campaign_identity_sha256": binding["campaign_identity_sha256"],
        "source_sha256": binding["source_sha256"],
        "recipe_sha256": binding["recipe_sha256"],
        "checkpoint_manifest_sha256": binding["checkpoint_manifest_sha256"],
        "checkpoint_model_sha256": binding["checkpoint_model_sha256"],
        "validation_manifest_sha256": binding["validation_manifest_sha256"],
        "loss_hook_sha256": binding["loss_hook_sha256"],
    }
    for key, value in expected.items():
        entry_key = {
            "campaign_identity_sha256": "identity_sha256",
            "frozen_recipe_sha256": "recipe_sha256",
        }.get(key, key)
        if key == "experiment":
            actual = baseline_entry.get("experiment")
        elif key == "campaign_identity_sha256":
            actual = baseline_entry.get("identity_sha256")
        else:
            actual = baseline_entry.get(entry_key)
        if actual != value:
            raise CandidateBuildError(f"Baseline entry/context differ at {key}")
    return validate_parent_receipt(
        {
            "run_id": binding["baseline_run_id"],
            **expected,
            "config": dict(baseline_entry["expected_config"]),
        }
    )


def lr_variant_spec(key: str) -> dict[str, Any]:
    for spec in LR_SPECS:
        if spec["key"] == key:
            return dict(spec)
    raise CandidateBuildError(f"Unknown LR candidate key: {key}")


def e2_variant_spec(parent: Mapping[str, Any]) -> dict[str, Any]:
    normalized = validate_parent_receipt(parent)
    learning_rate = float(normalized["config"]["learning_rate"])
    token_by_lr = {1e-5: "lr1e5", 2e-5: "lr2e5", 4e-5: "lr4e5"}
    if learning_rate not in token_by_lr:
        raise CandidateBuildError("Cannot materialize e2 outside the selected LR line")
    token = token_by_lr[learning_rate]
    return {
        "key": "e2",
        "stage": "epoch_line",
        "experiment": f"bge2_sft_oodtrain_e2_{token}_v1",
        "slug_token": f"e2-{token}",
        "learning_rate": learning_rate,
        "epochs": 2,
    }


def resolve_candidate_config(
    base_config: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_builder.validate_base_config(base_config)
    normalized_parent = validate_parent_receipt(parent)
    stage = spec.get("stage")
    if stage == "lr_log_line":
        config = deepcopy(dict(base_config))
        config["learning_rate"] = spec["learning_rate"]
        config["epochs"] = 1
        if normalized_parent["config"] != dict(base_config):
            raise CandidateBuildError("LR candidates must parent the exact baseline recipe")
    elif stage == "epoch_line":
        config = deepcopy(dict(normalized_parent["config"]))
        config["epochs"] = 2
        if spec.get("learning_rate") != normalized_parent["config"].get(
            "learning_rate"
        ):
            raise CandidateBuildError("e2 LR differs from its selected e1 parent")
    else:
        raise CandidateBuildError(f"Unsupported candidate stage: {stage!r}")
    allowed_changes = {"learning_rate", "epochs"}
    if {
        key
        for key in config
        if config.get(key) != base_config.get(key)
    } - allowed_changes:
        raise CandidateBuildError("Candidate changed a non-core recipe coordinate")
    exact_geometry = {
        "batch_size": 8,
        "eval_batch_size": 32,
        "gradient_accumulation": 12,
        "max_length": 384,
        "gradient_checkpointing": True,
        "attention_implementation": "sdpa",
        "seed": 42,
        "sampling": "none",
        "train_subset": "all",
        "loss_weighting": "none",
    }
    for key, expected in exact_geometry.items():
        if config.get(key) != expected:
            raise CandidateBuildError(f"Candidate changed frozen geometry/policy at {key}")
    if config["epochs"] not in {1, 2}:
        raise CandidateBuildError("Core candidate epochs must be one or two")
    if config["learning_rate"] not in {1e-5, 2e-5, 4e-5}:
        raise CandidateBuildError("Core candidate LR is outside the log line")
    return config


def candidate_notes(
    *,
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    parent: Mapping[str, Any],
    baseline_context: Mapping[str, Any],
    identity_sha256: str,
    generator_digest: str,
) -> str:
    return canonical_json(
        {
            "campaign": baseline_builder.CAMPAIGN,
            "stage": spec["stage"],
            "role": "candidate",
            "identity_sha256": identity_sha256,
            "generator_sha256": generator_digest,
            "initial_checkpoint": "model/pretrain_bge_2ep",
            "fresh_start": True,
            "parent_run_id": parent["run_id"],
            "parent_experiment": parent["experiment"],
            "parent_recipe_sha256": parent["recipe_sha256"],
            "baseline_dataset": baseline_context["dataset_ref"],
            "baseline_dataset_version": baseline_context["dataset_version"],
            "baseline_manifest_sha256": baseline_context["manifest_sha256"],
            "train_policy": "306669 human train + 41171 former OOD; exact concat",
            "validation_policy": "IID and hard only; OOD macro=-1 and no comparison",
            "loss_variant": "bce_finite_guard_v1",
            "epochs": config["epochs"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config["weight_decay"],
            "warmup_ratio": config["warmup_ratio"],
            "batch_size_per_gpu": config["batch_size"],
            "gradient_accumulation": config["gradient_accumulation"],
            "effective_batch": 192,
            "max_length": config["max_length"],
            "seed": config["seed"],
        }
    )


def candidate_identity(
    *,
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    parent: Mapping[str, Any],
    baseline_context: Mapping[str, Any],
    source_sha256: str,
    validation_manifest_sha256: str,
    checkpoint_manifest_sha256: str,
    executable_cells_sha256: str,
    generator_digest: str,
) -> str:
    return canonical_sha256(
        {
            "schema_version": 1,
            "template": TEMPLATE,
            "campaign": baseline_builder.CAMPAIGN,
            "spec": dict(spec),
            "config": dict(config),
            "parent": dict(parent),
            "baseline_dataset": {
                "dataset_ref": baseline_context["dataset_ref"],
                "dataset_version": baseline_context["dataset_version"],
                "manifest_sha256": baseline_context["manifest_sha256"],
                "binding": baseline_context["binding"],
            },
            "source_sha256": source_sha256,
            "validation_manifest_sha256": validation_manifest_sha256,
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "executable_cells_sha256": executable_cells_sha256,
            "loss_hook_sha256": baseline_builder.FIXED_LOSS_HOOK_SHA256,
            "generator_sha256": generator_digest,
            "fresh_start": True,
            "train_policy": "human_train_plus_former_ood_exact_concat_v1",
            "validation_policy": "iid_hard_only_ood_minus_one_v1",
        }
    )


def candidate_kernel_slug(spec: Mapping[str, Any], identity_sha256: str) -> str:
    slug = f"pm-b2-{spec['slug_token']}-{identity_sha256[:12]}-s42-c1"
    if len(slug) > 50 or re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug) is None:
        raise CandidateBuildError(f"Unsafe BGE candidate slug: {slug!r}")
    return slug


def _baseline_dataset_guard_cell(
    baseline_context: Mapping[str, Any],
) -> nbf.NotebookNode:
    expected = {
        "dataset_ref": baseline_context["dataset_ref"],
        "dataset_slug": baseline_context["dataset_slug"],
        "dataset_version": baseline_context["dataset_version"],
        "manifest_filename": baseline_context["manifest_filename"],
        "manifest_sha256": baseline_context["manifest_sha256"],
        "manifest": baseline_context["manifest"],
    }
    return baseline_builder.code(
        f"""
        EXPECTED_FROZEN_BASELINE = {expected!r}
        baseline_manifest_path = dataset_file(
            EXPECTED_FROZEN_BASELINE["dataset_slug"],
            EXPECTED_FROZEN_BASELINE["manifest_filename"],
        )
        if file_sha256(baseline_manifest_path) != EXPECTED_FROZEN_BASELINE["manifest_sha256"]:
            raise RuntimeError("Attached frozen BGE baseline manifest has changed")
        attached_baseline_manifest = json.loads(
            baseline_manifest_path.read_text(encoding="utf-8")
        )
        if attached_baseline_manifest != EXPECTED_FROZEN_BASELINE["manifest"]:
            raise RuntimeError("Attached frozen BGE baseline manifest payload differs")
        if (
            attached_baseline_manifest.get("dataset")
            != EXPECTED_FROZEN_BASELINE["dataset_ref"]
            or attached_baseline_manifest.get("is_private") is not True
            or attached_baseline_manifest.get("evaluated_splits") != ["iid", "hard"]
        ):
            raise RuntimeError("Attached frozen BGE baseline identity differs")
        baseline_files = attached_baseline_manifest["files"]
        if set(baseline_files) != {{
            "notebook_completed.json",
            "iid_validation_predictions.parquet",
            "hard_validation_predictions.parquet",
        }}:
            raise RuntimeError("Attached frozen BGE baseline file set differs")
        for filename, declaration in baseline_files.items():
            path = dataset_file(EXPECTED_FROZEN_BASELINE["dataset_slug"], filename)
            if (
                path.stat().st_size != declaration["bytes"]
                or file_sha256(path) != declaration["sha256"]
            ):
                raise RuntimeError(f"Attached frozen BGE baseline file differs: {{filename}}")
        if list(INPUT_ROOT.glob(
            f"**/{{EXPECTED_FROZEN_BASELINE['dataset_slug']}}/**/ood_validation_predictions.parquet"
        )):
            raise RuntimeError("Attached frozen BGE baseline contains OOD predictions")
        baseline_completion_path = dataset_file(
            EXPECTED_FROZEN_BASELINE["dataset_slug"],
            "notebook_completed.json",
        )
        baseline_completion = json.loads(
            baseline_completion_path.read_text(encoding="utf-8")
        )
        baseline_binding = attached_baseline_manifest["binding"]
        exact_completion_binding = {{
            "run_id": baseline_binding["baseline_run_id"],
            "experiment": baseline_binding["baseline_experiment"],
            "campaign": baseline_binding["campaign"],
            "campaign_identity_sha256": baseline_binding["campaign_identity_sha256"],
            "code_bundle_sha256": baseline_binding["source_sha256"],
            "frozen_recipe_sha256": baseline_binding["recipe_sha256"],
            "executable_cells_sha256": baseline_binding["executable_cells_sha256"],
            "loss_hook_sha256": baseline_binding["loss_hook_sha256"],
            "initial_checkpoint_manifest_sha256": baseline_binding[
                "checkpoint_manifest_sha256"
            ],
            "initial_checkpoint_model_sha256": baseline_binding[
                "checkpoint_model_sha256"
            ],
            "validation_manifest_sha256": baseline_binding[
                "validation_manifest_sha256"
            ],
        }}
        for key, expected_value in exact_completion_binding.items():
            if baseline_completion.get(key) != expected_value:
                raise RuntimeError(f"Attached baseline completion differs at {{key}}")
        if (
            baseline_completion.get("status") != "complete"
            or baseline_completion.get("role") != "baseline"
            or baseline_completion.get("experiment_group") != "sft"
            or baseline_completion.get("ood_evaluation_policy")
            != "disabled_train_contaminated"
        ):
            raise RuntimeError("Attached baseline completion policy differs")
        baseline_report = baseline_completion.get("training_report", {{}})
        if (
            baseline_report.get("evaluated_validation_splits") != ["iid", "hard"]
            or baseline_report.get("validation_splits", {{}}).get("ood", {{}}).get(
                "macro_average_precision"
            ) != -1.0
            or baseline_report.get("validation_splits", {{}}).get("ood", {{}}).get(
                "predictions_file"
            ) is not None
        ):
            raise RuntimeError("Attached baseline OOD sentinel differs")
        BASELINE_DATASET_GATE = {{
            "status": "passed",
            "dataset_ref": EXPECTED_FROZEN_BASELINE["dataset_ref"],
            "dataset_version": EXPECTED_FROZEN_BASELINE["dataset_version"],
            "manifest_sha256": EXPECTED_FROZEN_BASELINE["manifest_sha256"],
            "baseline_run_id": baseline_binding["baseline_run_id"],
            "baseline_identity_sha256": baseline_binding[
                "campaign_identity_sha256"
            ],
            "baseline_source_sha256": baseline_binding["source_sha256"],
            "ood_predictions": False,
        }}
        (WORKING_ROOT / {BASELINE_GATE_FILENAME!r}).write_text(
            json.dumps(BASELINE_DATASET_GATE, ensure_ascii=False, indent=2) + "\\n",
            encoding="utf-8",
        )
        print(json.dumps({{"frozen_baseline_dataset_gate": BASELINE_DATASET_GATE}}, indent=2))
        """,
        "frozen",
        "baseline-dataset-gate",
    )


def _augment_completion_cell(
    *,
    parent: Mapping[str, Any],
    baseline_context: Mapping[str, Any],
    generator_digest: str,
) -> nbf.NotebookNode:
    parent_public = {
        "run_id": parent["run_id"],
        "experiment": parent["experiment"],
        "campaign_identity_sha256": parent["campaign_identity_sha256"],
        "source_sha256": parent["source_sha256"],
        "recipe_sha256": parent["recipe_sha256"],
        "config": parent["config"],
    }
    expected_gate = {
        "dataset_ref": baseline_context["dataset_ref"],
        "dataset_version": baseline_context["dataset_version"],
        "manifest_sha256": baseline_context["manifest_sha256"],
        "baseline_run_id": baseline_context["binding"]["baseline_run_id"],
    }
    return baseline_builder.code(
        f"""
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        baseline_gate = json.loads(
            (WORKING_ROOT / {BASELINE_GATE_FILENAME!r}).read_text(encoding="utf-8")
        )
        EXPECTED_BASELINE_GATE_SUMMARY = {expected_gate!r}
        for key, expected_value in EXPECTED_BASELINE_GATE_SUMMARY.items():
            if baseline_gate.get(key) != expected_value:
                raise RuntimeError(f"Candidate completion baseline gate differs at {{key}}")
        if baseline_gate.get("status") != "passed" or baseline_gate.get(
            "ood_predictions"
        ) is not False:
            raise RuntimeError("Candidate completion baseline gate did not pass")
        completion["frozen_baseline_dataset"] = baseline_gate
        completion["stage_parent"] = {parent_public!r}
        completion["candidate_generator_sha256"] = {generator_digest!r}
        completion_path.write_text(
            json.dumps(completion, ensure_ascii=False, indent=2, default=str) + "\\n",
            encoding="utf-8",
        )
        print(json.dumps({{
            "candidate_parent_run_id": completion["stage_parent"]["run_id"],
            "baseline_dataset_version": baseline_gate["dataset_version"],
        }}, ensure_ascii=False, indent=2))
        """,
        "frozen",
        "candidate-receipt",
    )


def build_candidate_notebook(
    *,
    validation: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    base_config: Mapping[str, Any],
    spec: Mapping[str, Any],
    parent: Mapping[str, Any],
    baseline_context: Mapping[str, Any],
) -> tuple[nbf.NotebookNode, dict[str, Any]]:
    context = validate_baseline_context(baseline_context)
    normalized_parent = validate_parent_receipt(parent)
    experiment = str(spec.get("experiment", ""))
    if EXPERIMENT_PATTERN.fullmatch(experiment) is None:
        raise CandidateBuildError("Invalid BGE candidate experiment name")
    config = resolve_candidate_config(
        base_config,
        spec=spec,
        parent=normalized_parent,
    )
    sources, source_ledger, source_sha256 = baseline_builder.source_bundle()
    if source_sha256 != context["binding"]["source_sha256"]:
        raise CandidateBuildError(
            "Current runtime/source ledger differs from the frozen baseline Dataset"
        )
    if checkpoint["manifest_sha256"] != context["binding"][
        "checkpoint_manifest_sha256"
    ]:
        raise CandidateBuildError("Candidate checkpoint differs from frozen baseline")
    if validation["manifest_sha256"] != context["binding"][
        "validation_manifest_sha256"
    ]:
        raise CandidateBuildError("Candidate validation differs from frozen baseline")
    generator_digest = generator_sha256()
    recipe_sha256 = canonical_sha256(config)
    provisional_notes = candidate_notes(
        spec=spec,
        config=config,
        parent=normalized_parent,
        baseline_context=context,
        identity_sha256=baseline_builder.IDENTITY_PLACEHOLDER,
        generator_digest=generator_digest,
    )
    runtime_model_path = f"/kaggle/temp/{experiment}/initial_checkpoint"

    run_identity = baseline_builder.shared.experiment_run_initialization_cell()
    run_identity.metadata["tags"] = ["frozen", "run-identity"]
    sheet_cells = baseline_builder.shared.google_sheets_tracking_cells()
    for cell in sheet_cells:
        cell.metadata["tags"] = ["frozen", "sheets-sync"]
    cells = [
        baseline_builder.markdown(
            f"""
            # BGE 2ep SFT candidate: `{experiment}`

            This candidate starts fresh from the exact BGE two-epoch pretrain
            checkpoint.  It trains on human train plus former OOD, evaluates
            IID/hard only, and requires the exact private frozen BGE baseline
            Dataset before any training cell can run.

            Candidate identity: `{baseline_builder.IDENTITY_PLACEHOLDER}`.
            """,
            "frozen",
            "campaign-description",
        ),
        baseline_builder._setup_cell(
            experiment=experiment,
            validation=validation,
            checkpoint=checkpoint,
            source_sha256=source_sha256,
            identity_sha256=baseline_builder.IDENTITY_PLACEHOLDER,
            executable_sha256=baseline_builder.EXECUTABLE_CELLS_PLACEHOLDER,
        ),
        _baseline_dataset_guard_cell(context),
        run_identity,
        baseline_builder.markdown("## Frozen recipe", "frozen"),
        baseline_builder._recipe_cell(config, recipe_sha256),
        baseline_builder.markdown("## Exact baseline runtime sources", "frozen"),
        baseline_builder._sources_cell(sources, source_ledger, source_sha256),
        baseline_builder.markdown(
            "## Exact train concat and IID/hard isolation", "frozen"
        ),
        baseline_builder._data_cell(),
        baseline_builder._loss_cell(),
        baseline_builder.markdown(
            "## Real two-T4 optimizer-step memory preflight", "frozen"
        ),
        baseline_builder._preflight_cell(),
        baseline_builder.markdown("## Fresh full DDP fine-tuning", "frozen"),
        baseline_builder._training_cell(),
        baseline_builder.markdown(
            "## Slim completion with explicit OOD=-1", "frozen"
        ),
        baseline_builder._completion_cell(
            experiment=experiment,
            role="candidate",
            notes=provisional_notes,
            recipe_sha256=recipe_sha256,
        ),
        _augment_completion_cell(
            parent=normalized_parent,
            baseline_context=context,
            generator_digest=generator_digest,
        ),
        *sheet_cells,
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    executable_sha256 = baseline_builder.executable_cells_sha256(notebook)
    identity_sha256 = candidate_identity(
        spec=spec,
        config=config,
        parent=normalized_parent,
        baseline_context=context,
        source_sha256=source_sha256,
        validation_manifest_sha256=str(validation["manifest_sha256"]),
        checkpoint_manifest_sha256=str(checkpoint["manifest_sha256"]),
        executable_cells_sha256=executable_sha256,
        generator_digest=generator_digest,
    )
    notes = candidate_notes(
        spec=spec,
        config=config,
        parent=normalized_parent,
        baseline_context=context,
        identity_sha256=identity_sha256,
        generator_digest=generator_digest,
    )
    if provisional_notes.replace(
        baseline_builder.IDENTITY_PLACEHOLDER, identity_sha256
    ) != notes:
        raise CandidateBuildError("Candidate notes substitution is not deterministic")
    baseline_builder._replace_notebook_placeholders(
        notebook,
        identity_sha256=identity_sha256,
        executable_sha256=executable_sha256,
    )
    baseline_builder._assign_deterministic_cell_ids(notebook)
    if baseline_builder.executable_cells_sha256(
        notebook,
        identity_sha256=identity_sha256,
        expected_sha256=executable_sha256,
    ) != executable_sha256:
        raise CandidateBuildError("Final candidate executable cells changed")
    slug = candidate_kernel_slug(spec, identity_sha256)
    metadata = {
        "template": TEMPLATE,
        "campaign": baseline_builder.CAMPAIGN,
        "stage": spec["stage"],
        "experiment": experiment,
        "role": "candidate",
        "experiment_group": "sft",
        "validation_dataset": validation["dataset"],
        "validation_manifest_sha256": validation["manifest_sha256"],
        "initial_checkpoint": checkpoint["dataset"],
        "initial_checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
        "source_sha256": source_sha256,
        "frozen_recipe_sha256": recipe_sha256,
        "campaign_identity_sha256": identity_sha256,
        "executable_cells_sha256": executable_sha256,
        "loss_hook_sha256": baseline_builder.FIXED_LOSS_HOOK_SHA256,
        "candidate_generator_sha256": generator_digest,
        "runtime_model_path": runtime_model_path,
        "frozen_baseline_dataset": {
            "dataset_ref": context["dataset_ref"],
            "dataset_version": context["dataset_version"],
            "manifest_sha256": context["manifest_sha256"],
            "baseline_run_id": context["binding"]["baseline_run_id"],
        },
        "stage_parent": {
            "run_id": normalized_parent["run_id"],
            "experiment": normalized_parent["experiment"],
            "campaign_identity_sha256": normalized_parent[
                "campaign_identity_sha256"
            ],
            "recipe_sha256": normalized_parent["recipe_sha256"],
        },
        "fresh_start": True,
        "train_pairs": baseline_builder.EXPECTED_TRAIN,
        "validation_splits": ["iid", "hard"],
        "ood_metric_sentinel": -1,
        "ood_comparison": None,
        "expected_gpus": 2,
        "kernel_slug": slug,
        "editable_cells": [],
    }
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "product_matching_training": metadata,
        }
    )
    nbf.validate(notebook)
    entry = {
        "key": spec["key"],
        "stage": spec["stage"],
        "experiment": experiment,
        "role": "candidate",
        "default": False,
        "kernel_slug": slug,
        "title": slug,
        "recipe_sha256": recipe_sha256,
        "identity_sha256": identity_sha256,
        "executable_cells_sha256": executable_sha256,
        "source_sha256": source_sha256,
        "validation_dataset": validation["dataset"],
        "validation_manifest_sha256": validation["manifest_sha256"],
        "checkpoint_dataset": checkpoint["dataset"],
        "checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
        "checkpoint_model_sha256": context["binding"]["checkpoint_model_sha256"],
        "expected_config": config,
        "expected_runtime_model_path": runtime_model_path,
        "expected_notes": notes,
        "loss_hook_sha256": baseline_builder.FIXED_LOSS_HOOK_SHA256,
        "slug_token": spec["slug_token"],
        "candidate_generator_sha256": generator_digest,
        "baseline_context": context,
        "parent": normalized_parent,
        "fresh_start": True,
    }
    validate_candidate_notebook(notebook, entry=entry)
    return notebook, entry


def validate_candidate_notebook(
    notebook: nbf.NotebookNode,
    *,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    nbf.validate(notebook)
    metadata = notebook.metadata.get("product_matching_training")
    if not isinstance(metadata, Mapping) or metadata.get("template") != TEMPLATE:
        raise CandidateBuildError("Candidate notebook metadata is missing")
    context = validate_baseline_context(entry["baseline_context"])
    parent = validate_parent_receipt(entry["parent"])
    exact = {
        "template": TEMPLATE,
        "campaign": baseline_builder.CAMPAIGN,
        "stage": entry["stage"],
        "experiment": entry["experiment"],
        "role": "candidate",
        "experiment_group": "sft",
        "validation_dataset": entry["validation_dataset"],
        "validation_manifest_sha256": entry["validation_manifest_sha256"],
        "initial_checkpoint": entry["checkpoint_dataset"],
        "initial_checkpoint_manifest_sha256": entry[
            "checkpoint_manifest_sha256"
        ],
        "source_sha256": entry["source_sha256"],
        "frozen_recipe_sha256": entry["recipe_sha256"],
        "campaign_identity_sha256": entry["identity_sha256"],
        "executable_cells_sha256": entry["executable_cells_sha256"],
        "loss_hook_sha256": entry["loss_hook_sha256"],
        "candidate_generator_sha256": entry["candidate_generator_sha256"],
        "runtime_model_path": entry["expected_runtime_model_path"],
        "frozen_baseline_dataset": {
            "dataset_ref": context["dataset_ref"],
            "dataset_version": context["dataset_version"],
            "manifest_sha256": context["manifest_sha256"],
            "baseline_run_id": context["binding"]["baseline_run_id"],
        },
        "stage_parent": {
            "run_id": parent["run_id"],
            "experiment": parent["experiment"],
            "campaign_identity_sha256": parent["campaign_identity_sha256"],
            "recipe_sha256": parent["recipe_sha256"],
        },
        "kernel_slug": entry["kernel_slug"],
        "fresh_start": True,
        "train_pairs": baseline_builder.EXPECTED_TRAIN,
        "validation_splits": ["iid", "hard"],
        "ood_metric_sentinel": -1,
        "ood_comparison": None,
        "expected_gpus": 2,
        "editable_cells": [],
    }
    if dict(metadata) != exact:
        raise CandidateBuildError("Candidate notebook metadata differs from its entry")
    expected_baseline_metadata = {
        "dataset_ref": context["dataset_ref"],
        "dataset_version": context["dataset_version"],
        "manifest_sha256": context["manifest_sha256"],
        "baseline_run_id": context["binding"]["baseline_run_id"],
    }
    if metadata.get("frozen_baseline_dataset") != expected_baseline_metadata:
        raise CandidateBuildError("Candidate baseline Dataset metadata differs")
    if metadata.get("stage_parent") != {
        "run_id": parent["run_id"],
        "experiment": parent["experiment"],
        "campaign_identity_sha256": parent["campaign_identity_sha256"],
        "recipe_sha256": parent["recipe_sha256"],
    }:
        raise CandidateBuildError("Candidate parent metadata differs")
    _, _, source_sha256 = baseline_builder.source_bundle()
    if source_sha256 != entry["source_sha256"]:
        raise CandidateBuildError("Candidate source ledger is stale")
    if canonical_sha256(entry["expected_config"]) != entry["recipe_sha256"]:
        raise CandidateBuildError("Candidate recipe SHA differs")
    executable = baseline_builder.executable_cells_sha256(
        notebook,
        identity_sha256=str(entry["identity_sha256"]),
        expected_sha256=str(entry["executable_cells_sha256"]),
    )
    if executable != entry["executable_cells_sha256"]:
        raise CandidateBuildError("Candidate executable SHA differs")
    expected_identity = candidate_identity(
        spec={
            "key": entry["key"],
            "stage": entry["stage"],
            "experiment": entry["experiment"],
            "slug_token": entry["slug_token"],
            "learning_rate": entry["expected_config"]["learning_rate"],
            "epochs": entry["expected_config"]["epochs"],
        },
        config=entry["expected_config"],
        parent=parent,
        baseline_context=context,
        source_sha256=str(entry["source_sha256"]),
        validation_manifest_sha256=str(entry["validation_manifest_sha256"]),
        checkpoint_manifest_sha256=str(entry["checkpoint_manifest_sha256"]),
        executable_cells_sha256=executable,
        generator_digest=str(entry["candidate_generator_sha256"]),
    )
    if expected_identity != entry["identity_sha256"]:
        raise CandidateBuildError("Candidate campaign identity differs")
    if candidate_kernel_slug(entry, expected_identity) != entry["kernel_slug"]:
        raise CandidateBuildError("Candidate slug differs")
    code_tags = [
        tuple(cell.metadata.get("tags", []))
        for cell in notebook.cells
        if cell.cell_type == "code"
    ]
    for required in (
        ("frozen", "baseline-dataset-gate"),
        ("frozen", "candidate-receipt"),
        ("frozen", "sheets-sync"),
    ):
        if required not in code_tags:
            raise CandidateBuildError(f"Candidate notebook is missing cell {required}")
    return {
        "experiment": entry["experiment"],
        "kernel_slug": entry["kernel_slug"],
        "identity_sha256": expected_identity,
        "source_sha256": source_sha256,
        "baseline_dataset": expected_baseline_metadata,
        "parent_run_id": parent["run_id"],
    }


def load_candidate_notebook(
    path: Path,
    *,
    entry: Mapping[str, Any],
) -> nbf.NotebookNode:
    notebook = nbf.read(path, as_version=4)
    validate_candidate_notebook(notebook, entry=entry)
    return notebook


def build_candidate_campaign(
    *,
    owner: str,
    baseline_context: Mapping[str, Any],
    baseline_entry: Mapping[str, Any],
    specs: list[Mapping[str, Any]],
    parents: Mapping[str, Mapping[str, Any]],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write: bool = True,
) -> list[dict[str, Any]]:
    context = validate_baseline_context(baseline_context)
    validation = baseline_builder.load_validation_dataset(
        baseline_builder.DEFAULT_SOURCE_DIR, owner
    )
    checkpoint = baseline_builder.load_checkpoint_dataset(
        baseline_builder.DEFAULT_CHECKPOINT_STAGE_DIR,
        owner,
        verify_payload=True,
    )
    if file_sha256(baseline_builder.DEFAULT_CONFIG) != (
        baseline_builder.EXPECTED_BASE_CONFIG_FILE_SHA256
    ):
        raise CandidateBuildError("Frozen BGE base config file changed")
    base_config = cross_builder.load_training_config(baseline_builder.DEFAULT_CONFIG)
    baseline_builder.validate_base_config(base_config)
    if baseline_entry["identity_sha256"] != context["binding"][
        "campaign_identity_sha256"
    ]:
        raise CandidateBuildError("Current baseline entry differs from frozen Dataset")
    built: list[dict[str, Any]] = []
    for raw_spec in specs:
        spec = dict(raw_spec)
        key = str(spec["key"])
        if key not in parents:
            raise CandidateBuildError(f"Candidate {key} has no exact parent receipt")
        notebook, entry = build_candidate_notebook(
            validation=validation,
            checkpoint=checkpoint,
            base_config=base_config,
            spec=spec,
            parent=parents[key],
            baseline_context=context,
        )
        destination = output_dir / f"{entry['experiment']}_2xt4.ipynb"
        entry["notebook"] = str(destination)
        if write:
            destination.parent.mkdir(parents=True, exist_ok=True)
            nbf.write(notebook, destination)
            load_candidate_notebook(destination, entry=entry)
        built.append(entry)
    if len({entry["kernel_slug"] for entry in built}) != len(built):
        raise CandidateBuildError("Candidate campaign generated duplicate slugs")
    return built
