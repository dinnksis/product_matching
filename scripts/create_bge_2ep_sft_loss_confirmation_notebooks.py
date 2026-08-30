#!/usr/bin/env python3
"""Build the guarded BGE loss-screen and seed-confirmation notebooks.

This is a post-LR/e2 extension.  It never edits or widens the frozen BGE
baseline/candidate builders: the exact frozen runtime is reused, while this
module owns the one allowlisted transferred loss and the seed-17 variants.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from copy import deepcopy
from pathlib import Path
from textwrap import dedent
from typing import Any, Mapping, Sequence

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import create_bge_2ep_sft_candidate_notebooks as candidate_builder
import create_bge_2ep_sft_notebooks as baseline_builder
import create_cross_encoder_training_notebook as cross_builder


WORKFLOW = "bge_2ep_sft_loss_confirmation_v1"
TEMPLATE = WORKFLOW
DEFAULT_POLICY_PATH = ROOT / "configs" / f"{WORKFLOW}.json"
DEFAULT_OUTPUT_DIR = ROOT / "notebooks" / WORKFLOW
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
RUN_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
EXPERIMENT_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")
PLAIN_BCE = "bce_finite_guard_v1"
SQRT_BALANCED_BCE = "balanced_category_class_sqrt_bce"
EXPECTED_BALANCE_STRATA = 40
EXPECTED_BALANCE_CATEGORIES = 20
EXPECTED_BALANCE_WEIGHT_MIN = 0.6814300041898296
EXPECTED_BALANCE_WEIGHT_MAX = 2.753521186636443
EXPECTED_HISTORICAL_BGE_KERNEL_SLUGS = [
    "pm-b2-base-9c1f4648466b-s42-v1",
    "pm-b2-base-6ad383889383-s42-v1",
    "pm-b2-base-97335fa432bd-s42-v1",
    "pm-b2-base-de25c35eabf4-s42-v1",
]

WORKFLOW_SOURCE_FILES = (
    Path("configs/bge_2ep_sft_loss_confirmation_v1.json"),
    Path("scripts/create_bge_2ep_sft_loss_confirmation_notebooks.py"),
    Path("scripts/run_bge_2ep_sft_loss_confirmation.py"),
)

VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "screen_bce_s42": {
        "key": "screen_bce_s42",
        "stage": "loss_screen",
        "experiment": "bge2_sft_loss_bce_s42_v1",
        "slug_token": "lbce",
        "seed": 42,
        "loss_variant": PLAIN_BCE,
        "role_in_stage": "matched_anchor_only_if_required",
    },
    "screen_sqrt_s42": {
        "key": "screen_sqrt_s42",
        "stage": "loss_screen",
        "experiment": "bge2_sft_loss_sqrt_s42_v1",
        "slug_token": "lsqrt",
        "seed": 42,
        "loss_variant": SQRT_BALANCED_BCE,
        "role_in_stage": "single_challenger",
    },
    "confirm_bce_s17": {
        "key": "confirm_bce_s17",
        "stage": "seed_confirmation",
        "experiment": "bge2_sft_loss_bce_s17_v1",
        "slug_token": "lbce",
        "seed": 17,
        "loss_variant": PLAIN_BCE,
        "role_in_stage": "matched_seed_anchor",
    },
    "confirm_sqrt_s17": {
        "key": "confirm_sqrt_s17",
        "stage": "seed_confirmation",
        "experiment": "bge2_sft_loss_sqrt_s17_v1",
        "slug_token": "lsqrt",
        "seed": 17,
        "loss_variant": SQRT_BALANCED_BCE,
        "role_in_stage": "conditional_challenger",
    },
}


class LossConfirmationBuildError(ValueError):
    """Raised when a loss-confirmation notebook is not exact."""


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise LossConfirmationBuildError(f"{label} is not an exact lowercase SHA-256")
    return value


def _require_run_id(value: object, label: str) -> str:
    if not isinstance(value, str) or RUN_ID_PATTERN.fullmatch(value) is None:
        raise LossConfirmationBuildError(f"{label} is not a 32-hex run_id")
    return value


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LossConfirmationBuildError(f"Could not load loss policy: {error}") from error
    if not isinstance(policy, dict):
        raise LossConfirmationBuildError("Loss policy must be a JSON object")
    exact_top = {
        "schema_version",
        "campaign",
        "workflow",
        "status",
        "primary_split",
        "diagnostic_splits",
        "ood",
        "selected_recipe_source",
        "screen",
        "confirmation",
        "execution",
    }
    if set(policy) != exact_top:
        raise LossConfirmationBuildError("Loss policy top-level fields differ")
    fixed = {
        "schema_version": 1,
        "campaign": baseline_builder.CAMPAIGN,
        "workflow": WORKFLOW,
        "status": "plan_only_default",
        "primary_split": "iid",
        "diagnostic_splits": ["hard"],
        "ood": {
            "evaluated": False,
            "metric_sentinel": -1.0,
            "prediction_file": None,
            "comparison": None,
        },
        "selected_recipe_source": (
            "exact completed and Sheets-synced LR/e2 workflow receipt"
        ),
    }
    for key, expected in fixed.items():
        if policy.get(key) != expected:
            raise LossConfirmationBuildError(f"Loss policy differs at {key}")
    screen = policy.get("screen")
    if not isinstance(screen, Mapping) or dict(screen) != {
        "seed": 42,
        "anchor_loss": PLAIN_BCE,
        "challenger_loss": SQRT_BALANCED_BCE,
        "accept_challenger_only_if_iid_delta_strictly_greater_than": 0.002,
        "reuse_exact_selected_bce_anchor": True,
        "matched_bce_anchor_only_if_reuse_is_impossible": True,
    }:
        raise LossConfirmationBuildError("Loss-screen policy differs")
    confirmation = policy.get("confirmation")
    if not isinstance(confirmation, Mapping) or dict(confirmation) != {
        "seed": 17,
        "if_seed42_winner_is_bce": [PLAIN_BCE],
        "if_seed42_winner_is_challenger": [PLAIN_BCE, SQRT_BALANCED_BCE],
        "challenger_final_acceptance": {
            "seed42_delta_strictly_positive": True,
            "seed17_delta_strictly_positive": True,
            "mean_iid_delta_at_least": 0.002,
        },
    }:
        raise LossConfirmationBuildError("Seed-confirmation policy differs")
    execution = policy.get("execution")
    if not isinstance(execution, Mapping):
        raise LossConfirmationBuildError("Loss execution policy is missing")
    fixed_execution = {
        "sequential": True,
        "fanout": False,
        "resubmit_terminal_failure": False,
        "ods": False,
        "runtime_ablation": False,
        "checkpoint_export": False,
        "checkpoint_resume": False,
        "append_only_attempt_ledger": True,
        "remote_loss_prefix_audit_before_push": True,
        "max_total_bge_kernel_slugs": 10,
    }
    if set(execution) != {
        *fixed_execution,
        "historical_bge_kernel_slugs_before_lr_e2",
    }:
        raise LossConfirmationBuildError("Loss execution policy fields differ")
    for key, expected in fixed_execution.items():
        if execution.get(key) != expected:
            raise LossConfirmationBuildError(f"Loss execution policy differs at {key}")
    historical = execution.get("historical_bge_kernel_slugs_before_lr_e2")
    if historical != EXPECTED_HISTORICAL_BGE_KERNEL_SLUGS:
        raise LossConfirmationBuildError("Historical BGE kernel ledger differs")
    if any(re.fullmatch(r"[a-z0-9][a-z0-9-]*", str(slug)) is None for slug in historical):
        raise LossConfirmationBuildError("Historical BGE kernel ledger has an unsafe slug")
    return policy


def workflow_source_ledger() -> tuple[list[dict[str, Any]], str]:
    ledger: list[dict[str, Any]] = []
    for relative in WORKFLOW_SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise LossConfirmationBuildError(f"Workflow source is missing: {relative}")
        ledger.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return ledger, canonical_sha256(ledger)


def variant_spec(key: str) -> dict[str, Any]:
    try:
        return dict(VARIANT_SPECS[key])
    except KeyError as error:
        raise LossConfirmationBuildError(f"Unknown loss-confirmation variant: {key}") from error


BALANCED_CATEGORY_CLASS_SQRT_BCE_SOURCE = dedent(
    f"""
    from __future__ import annotations

    from collections import Counter
    import json
    import math

    import torch
    import torch.nn.functional as F


    _PAIR_BALANCE_WEIGHTS = None
    BALANCE_BY_CATEGORY = True
    BALANCE_POWER = 0.5
    LOSS_VARIANT = {SQRT_BALANCED_BCE!r}
    EXPECTED_ROWS = {baseline_builder.EXPECTED_TRAIN}
    EXPECTED_CATEGORIES = {EXPECTED_BALANCE_CATEGORIES}
    EXPECTED_STRATA = {EXPECTED_BALANCE_STRATA}
    EXPECTED_WEIGHT_MIN = {EXPECTED_BALANCE_WEIGHT_MIN!r}
    EXPECTED_WEIGHT_MAX = {EXPECTED_BALANCE_WEIGHT_MAX!r}


    def initialize_loss(*, train_frame, device, rank, world_size):
        global _PAIR_BALANCE_WEIGHTS
        if world_size != 2 or rank not in {{0, 1}}:
            raise ValueError("BGE sqrt-balanced hook requires exact two-rank DDP")
        if len(train_frame) != EXPECTED_ROWS:
            raise ValueError("BGE sqrt-balanced hook received an unexpected row count")
        labels = (train_frame["target"].astype(float).to_numpy() >= 0.5).astype(int)
        categories = train_frame["category_1"].astype(str).tolist()
        unique_categories = set(categories)
        keys = list(zip(categories, labels.tolist()))
        observed = set(keys)
        expected = {{
            (category, label)
            for category in unique_categories
            for label in (0, 1)
        }}
        if (
            len(unique_categories) != EXPECTED_CATEGORIES
            or len(observed) != EXPECTED_STRATA
            or observed != expected
        ):
            raise ValueError("Every one of 20 BGE training categories must contain both classes")
        counts = Counter(keys)
        raw = [float(counts[key]) ** (-BALANCE_POWER) for key in keys]
        normalizer = sum(raw) / len(raw)
        weights = [value / normalizer for value in raw]
        if (
            not math.isclose(sum(weights) / len(weights), 1.0, abs_tol=1e-12)
            or not math.isclose(min(weights), EXPECTED_WEIGHT_MIN, abs_tol=1e-12)
            or not math.isclose(max(weights), EXPECTED_WEIGHT_MAX, abs_tol=1e-12)
        ):
            raise ValueError("BGE category/class sqrt-balance weights changed")
        _PAIR_BALANCE_WEIGHTS = torch.tensor(
            weights, dtype=torch.float32, device=device
        )
        print(json.dumps({{
            "loss_variant": LOSS_VARIANT,
            "balance_by_category": BALANCE_BY_CATEGORY,
            "balance_power": BALANCE_POWER,
            "categories": len(unique_categories),
            "strata": len(observed),
            "weight_min": min(weights),
            "weight_max": max(weights),
            "weight_mean": sum(weights) / len(weights),
            "rank": rank,
            "world_size": world_size,
        }}, ensure_ascii=False), flush=True)


    def compute_loss(
        *,
        logits,
        targets,
        sample_weights,
        pair_indices,
        orientations,
        epoch,
        step,
    ):
        if _PAIR_BALANCE_WEIGHTS is None:
            raise RuntimeError("BGE sqrt-balanced loss was not initialized")
        if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
            raise FloatingPointError("non-finite BGE sqrt-balanced inputs")
        if not torch.isfinite(sample_weights).all():
            raise FloatingPointError("non-finite BGE external sample weights")
        per_example_bce = F.binary_cross_entropy_with_logits(
            logits.float(), targets, reduction="none"
        )
        denominator = sample_weights.sum()
        if not torch.isfinite(denominator) or denominator <= 0:
            raise FloatingPointError("invalid BGE sqrt-balanced denominator")
        bce = (per_example_bce * sample_weights).sum() / denominator
        balance_weights = _PAIR_BALANCE_WEIGHTS.index_select(0, pair_indices)
        combined_weights = sample_weights * balance_weights
        balanced_bce = (per_example_bce * combined_weights).sum() / denominator
        if not torch.isfinite(balanced_bce):
            raise FloatingPointError("non-finite BGE sqrt-balanced BCE")
        return {{
            "loss": balanced_bce,
            "bce": bce.detach(),
            "balanced_bce": balanced_bce.detach(),
            "batch_balance_weight": balance_weights.mean().detach(),
        }}
    """
).strip() + "\n"

LOSS_HOOK_SOURCES = {
    PLAIN_BCE: baseline_builder.FIXED_LOSS_HOOK_SOURCE,
    SQRT_BALANCED_BCE: BALANCED_CATEGORY_CLASS_SQRT_BCE_SOURCE,
}
LOSS_HOOK_SHA256 = {
    name: hashlib.sha256(source.encode("utf-8")).hexdigest()
    for name, source in LOSS_HOOK_SOURCES.items()
}


def validate_parent_receipt(
    parent: Mapping[str, Any],
    *,
    require_seed: int | None = None,
    require_plain_bce: bool = False,
) -> dict[str, Any]:
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
        raise LossConfirmationBuildError("Loss stage parent receipt fields differ")
    _require_run_id(parent.get("run_id"), "parent.run_id")
    if EXPERIMENT_PATTERN.fullmatch(str(parent.get("experiment", ""))) is None:
        raise LossConfirmationBuildError("Loss stage parent experiment is invalid")
    for key in exact_keys - {"run_id", "experiment", "config"}:
        _require_hash(parent.get(key), f"parent.{key}")
    config = parent.get("config")
    if not isinstance(config, Mapping):
        raise LossConfirmationBuildError("Loss stage parent config is missing")
    normalized_config = dict(config)
    if canonical_sha256(normalized_config) != parent.get("recipe_sha256"):
        raise LossConfirmationBuildError("Loss stage parent config/recipe SHA differ")
    if normalized_config.get("epochs") not in {1, 2}:
        raise LossConfirmationBuildError("Loss stage parent must use selected e1/e2 epochs")
    if normalized_config.get("learning_rate") not in {1e-5, 2e-5, 4e-5}:
        raise LossConfirmationBuildError("Loss stage parent LR is outside the selected log line")
    if normalized_config.get("seed") not in {17, 42}:
        raise LossConfirmationBuildError("Loss stage parent seed is not allowlisted")
    if require_seed is not None and normalized_config.get("seed") != require_seed:
        raise LossConfirmationBuildError("Loss stage parent seed differs")
    frozen = baseline_builder.EXPECTED_BASE_CONFIG
    if set(normalized_config) != set(frozen):
        raise LossConfirmationBuildError("Loss stage parent config fields differ")
    selected_coordinates = {"learning_rate", "epochs", "seed"}
    for key, expected in frozen.items():
        if key not in selected_coordinates and normalized_config.get(key) != expected:
            raise LossConfirmationBuildError(f"Loss stage parent changed {key}")
    if require_plain_bce and parent.get("loss_hook_sha256") != LOSS_HOOK_SHA256[PLAIN_BCE]:
        raise LossConfirmationBuildError("Loss stage anchor is not exact plain BCE")
    return {**dict(parent), "config": normalized_config}


def resolve_config(parent: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    key = str(spec.get("key", ""))
    if spec != VARIANT_SPECS.get(key):
        raise LossConfirmationBuildError("Loss variant spec differs from its allowlist")
    expected_parent_seed = 17 if key == "confirm_sqrt_s17" else 42
    normalized_parent = validate_parent_receipt(
        parent,
        require_seed=expected_parent_seed,
        require_plain_bce=True,
    )
    if (
        key == "confirm_sqrt_s17"
        and normalized_parent["experiment"]
        != VARIANT_SPECS["confirm_bce_s17"]["experiment"]
    ):
        raise LossConfirmationBuildError(
            "Seed17 challenger must parent the exact matched seed17 BCE run"
        )
    config = deepcopy(normalized_parent["config"])
    config["seed"] = int(spec["seed"])
    changed = {
        name for name in config if config.get(name) != normalized_parent["config"].get(name)
    }
    allowed = set() if spec["seed"] == normalized_parent["config"]["seed"] else {"seed"}
    if changed != allowed:
        raise LossConfirmationBuildError("Loss candidate changed a non-seed recipe field")
    if spec["loss_variant"] not in LOSS_HOOK_SOURCES:
        raise LossConfirmationBuildError("Loss candidate is not allowlisted")
    return config


def candidate_notes(
    *,
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    parent: Mapping[str, Any],
    baseline_context: Mapping[str, Any],
    identity_sha256: str,
    workflow_ledger_sha256: str,
) -> str:
    return canonical_json(
        {
            "workflow": WORKFLOW,
            "campaign": baseline_builder.CAMPAIGN,
            "stage": spec["stage"],
            "role": "candidate",
            "role_in_stage": spec["role_in_stage"],
            "identity_sha256": identity_sha256,
            "workflow_ledger_sha256": workflow_ledger_sha256,
            "initial_checkpoint": "model/pretrain_bge_2ep",
            "fresh_start": True,
            "checkpoint_resume": False,
            "parent_run_id": parent["run_id"],
            "parent_experiment": parent["experiment"],
            "parent_recipe_sha256": parent["recipe_sha256"],
            "baseline_dataset": baseline_context["dataset_ref"],
            "baseline_dataset_version": baseline_context["dataset_version"],
            "baseline_manifest_sha256": baseline_context["manifest_sha256"],
            "train_policy": "306669 human train + 41171 former OOD; exact concat",
            "validation_policy": "IID primary; hard diagnostic; OOD=-1/no parquet",
            "loss_variant": spec["loss_variant"],
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
    loss_hook_sha256: str,
    workflow_ledger: Sequence[Mapping[str, Any]],
    workflow_ledger_sha256: str,
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
            "loss_variant": spec["loss_variant"],
            "loss_hook_sha256": loss_hook_sha256,
            "workflow_source_ledger": list(workflow_ledger),
            "workflow_ledger_sha256": workflow_ledger_sha256,
            "fresh_start": True,
            "checkpoint_resume": False,
            "train_policy": "human_train_plus_former_ood_exact_concat_v1",
            "validation_policy": "iid_hard_only_ood_minus_one_v1",
        }
    )


def candidate_kernel_slug(spec: Mapping[str, Any], identity_sha256: str) -> str:
    slug = (
        f"pm-b2-{spec['slug_token']}-{identity_sha256[:12]}-"
        f"s{spec['seed']}-l1"
    )
    if len(slug) > 50 or re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug) is None:
        raise LossConfirmationBuildError(f"Unsafe loss-confirmation slug: {slug!r}")
    return slug


def _loss_cell(loss_variant: str, loss_source: str, loss_sha256: str) -> nbf.NotebookNode:
    return baseline_builder.code(
        f"""
        LOSS_CONFIRMATION_VARIANT = {loss_variant!r}
        LOSS_CONFIRMATION_HOOK_SOURCE = {loss_source!r}
        EXPECTED_LOSS_HOOK_SHA256 = {loss_sha256!r}
        if hashlib.sha256(LOSS_CONFIRMATION_HOOK_SOURCE.encode("utf-8")).hexdigest() != EXPECTED_LOSS_HOOK_SHA256:
            raise RuntimeError("BGE loss-confirmation hook was edited")
        LOSS_HOOK_PATH.write_text(LOSS_CONFIRMATION_HOOK_SOURCE, encoding="utf-8")
        if file_sha256(LOSS_HOOK_PATH) != EXPECTED_LOSS_HOOK_SHA256:
            raise RuntimeError("Materialized BGE loss-confirmation hook differs")
        """,
        "frozen",
        "loss-confirmation-hook",
    )


def _loss_receipt_cell(
    *,
    spec: Mapping[str, Any],
    parent: Mapping[str, Any],
    workflow_ledger: Sequence[Mapping[str, Any]],
    workflow_ledger_sha256: str,
) -> nbf.NotebookNode:
    public_parent = {
        "run_id": parent["run_id"],
        "experiment": parent["experiment"],
        "campaign_identity_sha256": parent["campaign_identity_sha256"],
        "recipe_sha256": parent["recipe_sha256"],
        "loss_hook_sha256": parent["loss_hook_sha256"],
    }
    receipt = {
        "schema_version": 1,
        "workflow": WORKFLOW,
        "stage": spec["stage"],
        "role_in_stage": spec["role_in_stage"],
        "seed": spec["seed"],
        "loss_variant": spec["loss_variant"],
        "parent": public_parent,
        "fresh_start": True,
        "checkpoint_resume": False,
        "workflow_source_ledger": list(workflow_ledger),
        "workflow_ledger_sha256": workflow_ledger_sha256,
        "primary_split": "iid",
        "diagnostic_splits": ["hard"],
        "ood_macro_average_precision": -1.0,
        "ood_comparison": None,
    }
    return baseline_builder.code(
        f"""
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if (
            completion.get("loss_variant") != {PLAIN_BCE!r}
            or completion.get("loss_hook_sha256") != EXPECTED_LOSS_HOOK_SHA256
        ):
            raise RuntimeError("Base completion did not bind the executed loss hook")
        completion["loss_variant"] = {spec['loss_variant']!r}
        completion["loss_confirmation"] = {receipt!r}
        completion_path.write_text(
            json.dumps(completion, ensure_ascii=False, indent=2, default=str) + "\\n",
            encoding="utf-8",
        )
        """,
        "frozen",
        "loss-confirmation-receipt",
    )


def build_notebook(
    *,
    validation: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    base_config: Mapping[str, Any],
    spec: Mapping[str, Any],
    parent: Mapping[str, Any],
    baseline_context: Mapping[str, Any],
) -> tuple[nbf.NotebookNode, dict[str, Any]]:
    load_policy()
    baseline_builder.validate_base_config(base_config)
    context = candidate_builder.validate_baseline_context(baseline_context)
    normalized_parent = validate_parent_receipt(parent)
    exact_spec = variant_spec(str(spec.get("key", "")))
    if dict(spec) != exact_spec:
        raise LossConfirmationBuildError("Loss-confirmation spec changed")
    config = resolve_config(normalized_parent, exact_spec)
    experiment = exact_spec["experiment"]
    sources, source_ledger, source_sha256 = baseline_builder.source_bundle()
    if source_sha256 != context["binding"]["source_sha256"]:
        raise LossConfirmationBuildError("Frozen BGE runtime differs from baseline Dataset")
    if validation["manifest_sha256"] != context["binding"]["validation_manifest_sha256"]:
        raise LossConfirmationBuildError("Loss candidate validation differs from baseline")
    if checkpoint["manifest_sha256"] != context["binding"]["checkpoint_manifest_sha256"]:
        raise LossConfirmationBuildError("Loss candidate checkpoint differs from baseline")
    workflow_ledger, workflow_ledger_sha256 = workflow_source_ledger()
    loss_variant = exact_spec["loss_variant"]
    loss_source = LOSS_HOOK_SOURCES[loss_variant]
    loss_hook_sha256 = LOSS_HOOK_SHA256[loss_variant]
    recipe_sha256 = canonical_sha256(config)
    provisional_notes = candidate_notes(
        spec=exact_spec,
        config=config,
        parent=normalized_parent,
        baseline_context=context,
        identity_sha256=baseline_builder.IDENTITY_PLACEHOLDER,
        workflow_ledger_sha256=workflow_ledger_sha256,
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
            # BGE loss confirmation: `{experiment}`

            Fresh-start post-LR/e2 loss/seed evidence. IID is primary, hard is
            diagnostic, and former OOD remains in train with metric sentinel -1.

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
        candidate_builder._baseline_dataset_guard_cell(context),
        run_identity,
        baseline_builder.markdown("## Exact selected recipe", "frozen"),
        baseline_builder._recipe_cell(config, recipe_sha256),
        baseline_builder.markdown("## Exact frozen runtime sources", "frozen"),
        baseline_builder._sources_cell(sources, source_ledger, source_sha256),
        baseline_builder.markdown("## Exact train concat and split isolation", "frozen"),
        baseline_builder._data_cell(),
        _loss_cell(loss_variant, loss_source, loss_hook_sha256),
        baseline_builder.markdown("## Real two-T4 optimizer-step preflight", "frozen"),
        baseline_builder._preflight_cell(),
        baseline_builder.markdown("## Fresh full DDP fine-tuning", "frozen"),
        baseline_builder._training_cell(),
        baseline_builder.markdown("## Slim IID/hard completion", "frozen"),
        baseline_builder._completion_cell(
            experiment=experiment,
            role="candidate",
            notes=provisional_notes,
            recipe_sha256=recipe_sha256,
        ),
        _loss_receipt_cell(
            spec=exact_spec,
            parent=normalized_parent,
            workflow_ledger=workflow_ledger,
            workflow_ledger_sha256=workflow_ledger_sha256,
        ),
        candidate_builder._augment_completion_cell(
            parent=normalized_parent,
            baseline_context=context,
            generator_digest=file_sha256(Path(__file__).resolve()),
        ),
        *sheet_cells,
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    executable_sha256 = baseline_builder.executable_cells_sha256(notebook)
    identity_sha256 = candidate_identity(
        spec=exact_spec,
        config=config,
        parent=normalized_parent,
        baseline_context=context,
        source_sha256=source_sha256,
        validation_manifest_sha256=str(validation["manifest_sha256"]),
        checkpoint_manifest_sha256=str(checkpoint["manifest_sha256"]),
        executable_cells_sha256=executable_sha256,
        loss_hook_sha256=loss_hook_sha256,
        workflow_ledger=workflow_ledger,
        workflow_ledger_sha256=workflow_ledger_sha256,
    )
    notes = candidate_notes(
        spec=exact_spec,
        config=config,
        parent=normalized_parent,
        baseline_context=context,
        identity_sha256=identity_sha256,
        workflow_ledger_sha256=workflow_ledger_sha256,
    )
    if provisional_notes.replace(baseline_builder.IDENTITY_PLACEHOLDER, identity_sha256) != notes:
        raise LossConfirmationBuildError("Loss candidate notes substitution is unstable")
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
        raise LossConfirmationBuildError("Final loss candidate executable cells changed")
    slug = candidate_kernel_slug(exact_spec, identity_sha256)
    generator_sha256 = file_sha256(Path(__file__).resolve())
    metadata = {
        "template": TEMPLATE,
        "workflow": WORKFLOW,
        "campaign": baseline_builder.CAMPAIGN,
        "stage": exact_spec["stage"],
        "experiment": experiment,
        "role": "candidate",
        "role_in_stage": exact_spec["role_in_stage"],
        "experiment_group": "sft",
        "loss_variant": loss_variant,
        "loss_hook_sha256": loss_hook_sha256,
        "seed": exact_spec["seed"],
        "validation_dataset": validation["dataset"],
        "validation_manifest_sha256": validation["manifest_sha256"],
        "initial_checkpoint": checkpoint["dataset"],
        "initial_checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
        "source_sha256": source_sha256,
        "frozen_recipe_sha256": recipe_sha256,
        "campaign_identity_sha256": identity_sha256,
        "executable_cells_sha256": executable_sha256,
        "candidate_generator_sha256": generator_sha256,
        "workflow_ledger_sha256": workflow_ledger_sha256,
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
            "campaign_identity_sha256": normalized_parent["campaign_identity_sha256"],
            "recipe_sha256": normalized_parent["recipe_sha256"],
        },
        "fresh_start": True,
        "checkpoint_resume": False,
        "train_pairs": baseline_builder.EXPECTED_TRAIN,
        "validation_splits": ["iid", "hard"],
        "primary_split": "iid",
        "diagnostic_splits": ["hard"],
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
        "key": exact_spec["key"],
        "stage": exact_spec["stage"],
        "role_in_stage": exact_spec["role_in_stage"],
        "experiment": experiment,
        "role": "candidate",
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
        "loss_variant": loss_variant,
        "loss_hook_sha256": loss_hook_sha256,
        "seed": exact_spec["seed"],
        "slug_token": exact_spec["slug_token"],
        "candidate_generator_sha256": generator_sha256,
        "workflow_source_ledger": workflow_ledger,
        "workflow_ledger_sha256": workflow_ledger_sha256,
        "baseline_context": context,
        "parent": normalized_parent,
        "fresh_start": True,
        "checkpoint_resume": False,
    }
    validate_notebook(notebook, entry=entry)
    return notebook, entry


def validate_notebook(
    notebook: nbf.NotebookNode,
    *,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    nbf.validate(notebook)
    metadata = notebook.metadata.get("product_matching_training")
    if not isinstance(metadata, Mapping):
        raise LossConfirmationBuildError("Loss notebook metadata is missing")
    context = candidate_builder.validate_baseline_context(entry["baseline_context"])
    parent = validate_parent_receipt(entry["parent"])
    spec = variant_spec(str(entry["key"]))
    config = resolve_config(parent, spec)
    workflow_ledger, workflow_ledger_sha256 = workflow_source_ledger()
    exact_metadata = {
        "template": TEMPLATE,
        "workflow": WORKFLOW,
        "campaign": baseline_builder.CAMPAIGN,
        "stage": spec["stage"],
        "experiment": spec["experiment"],
        "role": "candidate",
        "role_in_stage": spec["role_in_stage"],
        "experiment_group": "sft",
        "loss_variant": spec["loss_variant"],
        "loss_hook_sha256": LOSS_HOOK_SHA256[spec["loss_variant"]],
        "seed": spec["seed"],
        "validation_dataset": entry["validation_dataset"],
        "validation_manifest_sha256": entry["validation_manifest_sha256"],
        "initial_checkpoint": entry["checkpoint_dataset"],
        "initial_checkpoint_manifest_sha256": entry["checkpoint_manifest_sha256"],
        "source_sha256": entry["source_sha256"],
        "frozen_recipe_sha256": entry["recipe_sha256"],
        "campaign_identity_sha256": entry["identity_sha256"],
        "executable_cells_sha256": entry["executable_cells_sha256"],
        "candidate_generator_sha256": entry["candidate_generator_sha256"],
        "workflow_ledger_sha256": workflow_ledger_sha256,
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
        "fresh_start": True,
        "checkpoint_resume": False,
        "train_pairs": baseline_builder.EXPECTED_TRAIN,
        "validation_splits": ["iid", "hard"],
        "primary_split": "iid",
        "diagnostic_splits": ["hard"],
        "ood_metric_sentinel": -1,
        "ood_comparison": None,
        "expected_gpus": 2,
        "kernel_slug": entry["kernel_slug"],
        "editable_cells": [],
    }
    if dict(metadata) != exact_metadata:
        raise LossConfirmationBuildError("Loss notebook metadata differs from its entry")
    if entry.get("workflow_source_ledger") != workflow_ledger:
        raise LossConfirmationBuildError("Loss workflow source ledger differs")
    if entry.get("workflow_ledger_sha256") != workflow_ledger_sha256:
        raise LossConfirmationBuildError("Loss workflow ledger SHA differs")
    generator_ledger = next(
        row
        for row in workflow_ledger
        if row["path"] == "scripts/create_bge_2ep_sft_loss_confirmation_notebooks.py"
    )
    if entry.get("candidate_generator_sha256") != generator_ledger["sha256"]:
        raise LossConfirmationBuildError("Loss candidate generator SHA differs")
    if entry.get("expected_config") != config or canonical_sha256(config) != entry.get("recipe_sha256"):
        raise LossConfirmationBuildError("Loss notebook recipe differs")
    if entry.get("loss_hook_sha256") != LOSS_HOOK_SHA256[spec["loss_variant"]]:
        raise LossConfirmationBuildError("Loss notebook hook SHA differs")
    executable = baseline_builder.executable_cells_sha256(
        notebook,
        identity_sha256=str(entry["identity_sha256"]),
        expected_sha256=str(entry["executable_cells_sha256"]),
    )
    expected_identity = candidate_identity(
        spec=spec,
        config=config,
        parent=parent,
        baseline_context=context,
        source_sha256=str(entry["source_sha256"]),
        validation_manifest_sha256=str(entry["validation_manifest_sha256"]),
        checkpoint_manifest_sha256=str(entry["checkpoint_manifest_sha256"]),
        executable_cells_sha256=executable,
        loss_hook_sha256=str(entry["loss_hook_sha256"]),
        workflow_ledger=workflow_ledger,
        workflow_ledger_sha256=workflow_ledger_sha256,
    )
    if expected_identity != entry.get("identity_sha256"):
        raise LossConfirmationBuildError("Loss notebook campaign identity differs")
    if candidate_kernel_slug(spec, expected_identity) != entry.get("kernel_slug"):
        raise LossConfirmationBuildError("Loss notebook kernel slug differs")
    code_tags = [
        tuple(cell.metadata.get("tags", []))
        for cell in notebook.cells
        if cell.cell_type == "code"
    ]
    for required in (
        ("frozen", "baseline-dataset-gate"),
        ("frozen", "loss-confirmation-hook"),
        ("frozen", "loss-confirmation-receipt"),
        ("frozen", "candidate-receipt"),
        ("frozen", "sheets-sync"),
    ):
        if required not in code_tags:
            raise LossConfirmationBuildError(f"Loss notebook is missing cell {required}")
    if any("team-editable" in cell.metadata.get("tags", []) for cell in notebook.cells):
        raise LossConfirmationBuildError("Loss notebook contains editable cells")
    if entry.get("fresh_start") is not True or entry.get("checkpoint_resume") is not False:
        raise LossConfirmationBuildError("Loss notebook attempted checkpoint continuation")
    return {
        "experiment": entry["experiment"],
        "kernel_slug": entry["kernel_slug"],
        "identity_sha256": expected_identity,
        "loss_variant": spec["loss_variant"],
        "seed": spec["seed"],
        "parent_run_id": parent["run_id"],
    }


def load_notebook(path: Path, *, entry: Mapping[str, Any]) -> nbf.NotebookNode:
    notebook = nbf.read(path, as_version=4)
    validate_notebook(notebook, entry=entry)
    return notebook


def build_campaign(
    *,
    owner: str,
    baseline_context: Mapping[str, Any],
    baseline_entry: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    parents: Mapping[str, Mapping[str, Any]],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write: bool = True,
) -> list[dict[str, Any]]:
    load_policy()
    context = candidate_builder.validate_baseline_context(baseline_context)
    validation = baseline_builder.load_validation_dataset(
        baseline_builder.DEFAULT_SOURCE_DIR, owner
    )
    checkpoint = baseline_builder.load_checkpoint_dataset(
        baseline_builder.DEFAULT_CHECKPOINT_STAGE_DIR,
        owner,
        verify_payload=True,
    )
    if file_sha256(baseline_builder.DEFAULT_CONFIG) != baseline_builder.EXPECTED_BASE_CONFIG_FILE_SHA256:
        raise LossConfirmationBuildError("Frozen BGE base config file changed")
    base_config = cross_builder.load_training_config(baseline_builder.DEFAULT_CONFIG)
    baseline_builder.validate_base_config(base_config)
    if baseline_entry.get("identity_sha256") != context["binding"]["campaign_identity_sha256"]:
        raise LossConfirmationBuildError("Current baseline entry differs from frozen v4 Dataset")
    built: list[dict[str, Any]] = []
    for raw_spec in specs:
        spec = variant_spec(str(raw_spec.get("key", "")))
        if dict(raw_spec) != spec:
            raise LossConfirmationBuildError("Loss campaign spec is not exact")
        key = spec["key"]
        if key not in parents:
            raise LossConfirmationBuildError(f"Loss candidate {key} has no exact parent")
        notebook, entry = build_notebook(
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
            load_notebook(destination, entry=entry)
        built.append(entry)
    if len({entry["kernel_slug"] for entry in built}) != len(built):
        raise LossConfirmationBuildError("Loss campaign generated duplicate slugs")
    return built


if __name__ == "__main__":
    print(json.dumps(load_policy(), ensure_ascii=False, indent=2))
