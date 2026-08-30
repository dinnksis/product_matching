#!/usr/bin/env python3
"""Build the guarded BGE-2ep SFT baseline and initial log-LR notebooks.

The supplied checkpoint was pretrained with the two categories that formed the
MiniLM OOD split.  Those 41,171 human pairs are therefore deliberately folded
into supervised training here.  Only IID and hard are evaluated; the final
report carries an explicit, machine-readable OOD=-1 sentinel so the shared
``sft_exps`` worksheet cannot mistake the contaminated split for an evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from textwrap import dedent
from typing import Any, Iterable, Mapping

import nbformat as nbf

import create_cross_encoder_training_notebook as cross_builder
import create_minilm_validation_baseline_notebook as validation_builder
import create_qwen_training_notebook as shared
import push_bge_pretrain_checkpoint_dataset as checkpoint_push


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = "bge_2ep_sft_oodtrain_v1"
DEFAULT_CONFIG = ROOT / "configs" / "cross_encoder_bge_2ep_oodtrain_baseline.json"
DEFAULT_SOURCE_DIR = validation_builder.DEFAULT_SOURCE_DIR
DEFAULT_CHECKPOINT_STAGE_DIR = checkpoint_push.STAGE_DIR
DEFAULT_OUTPUT_DIR = ROOT / "notebooks" / CAMPAIGN
VALIDATION_DATASET_SLUG = validation_builder.DATASET_SLUG
CHECKPOINT_DATASET_SLUG = checkpoint_push.DATASET_SLUG
EXPECTED_VALIDATION_MANIFEST_SHA256 = (
    "b64a1902d86c9ad896a626b2a17bf018341f1d9c5fefa124834b525c84808f3c"
)
EXPECTED_OOD_CATEGORIES = ("Одежда", "Бытовая техника")

EXPECTED_ITEMS = 711_304
EXPECTED_HUMAN_TRAIN = 306_669
EXPECTED_FORMER_OOD = 41_171
EXPECTED_TRAIN = 347_840
EXPECTED_TRAIN_POSITIVES = 89_291
EXPECTED_TRAIN_POSITIVE_RATE = EXPECTED_TRAIN_POSITIVES / EXPECTED_TRAIN
EXPECTED_IID = 12_000
EXPECTED_HARD = 5_814
EXPECTED_PARAMETERS = 567_755_777
EXPECTED_TRAINABLE_PARAMETER_TENSORS = 393
EXPECTED_WORLD_SIZE = 2
EXPECTED_EFFECTIVE_BATCH = 192
EXPECTED_PREFLIGHT_AMP_ATTEMPTS = 17
EXPECTED_AMP_NONFINITE_POLICY = (
    "finite_loss_guard_grad_scaler_bounded_backoff_v1"
)
EXPECTED_BASE_CONFIG_FILE_SHA256 = (
    "350748597f2c97deaeb814a72337992e3344ab908dffac426a3404205ef41fba"
)
EXPECTED_BASE_CONFIG_CANONICAL_SHA256 = (
    "2071b5596695033a2f122a68e4e059bc144cbcace2f569340ad4378e77e813d9"
)
EXPECTED_BASE_CONFIG: dict[str, Any] = {
    "model": "model/pretrain_bge_2ep",
    "model_backend": "sequence_classification",
    "model_load_kwargs": {},
    "trust_remote_code": False,
    "epochs": 1,
    "batch_size": 8,
    "eval_batch_size": 32,
    "gradient_accumulation": 12,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
    "max_length": 384,
    "attention_implementation": "sdpa",
    "sampling": "none",
    "train_subset": "all",
    "loss_weighting": "none",
    "lexical_hard_negative_strength": 0.0,
    "bucket_size_multiplier": 50,
    "dataloader_workers": 2,
    "prefetch_factor": 2,
    "tokenization_batch_size": 1024,
    "tokenization_log_every": 20,
    "gradient_checkpointing": True,
    "symmetric_validation": True,
    "label_smoothing": 0.0,
    "max_grad_norm": 0.5,
    "log_every": 50,
    "seed": 42,
}
IDENTITY_PLACEHOLDER = "__BGE_CAMPAIGN_IDENTITY_SHA256__"
EXECUTABLE_CELLS_PLACEHOLDER = "__BGE_EXECUTABLE_CELLS_SHA256__"

OOD_SENTINEL: dict[str, Any] = {
    "evaluated": False,
    "status": "disabled_train_contaminated",
    "reason": "former frozen OOD pairs are part of BGE supervised training",
    "examples": 0,
    "source_training_examples": EXPECTED_FORMER_OOD,
    "macro_average_precision": -1.0,
    "overall_average_precision": -1.0,
    "recall_at_precision_0_99": -1.0,
    "threshold_at_precision_0_99": -1.0,
    "roc_auc": -1.0,
    "log_loss": -1.0,
    "per_category_average_precision": {},
    "predictions_file": None,
}

VARIANT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "key": "baseline",
        "experiment": "bge2_sft_oodtrain_e1_lr2e5_baseline_v1",
        "role": "baseline",
        "slug_token": "base",
        "overrides": {},
        "default": True,
    },
    {
        "key": "lr1e5",
        "experiment": "bge2_sft_oodtrain_e1_lr1e5_v1",
        "role": "candidate",
        "slug_token": "lr1e5",
        "overrides": {"learning_rate": 1e-5},
        "default": False,
    },
    {
        "key": "lr4e5",
        "experiment": "bge2_sft_oodtrain_e1_lr4e5_v1",
        "role": "candidate",
        "slug_token": "lr4e5",
        "overrides": {"learning_rate": 4e-5},
        "default": False,
    },
)

FIXED_LOSS_HOOK_SOURCE = dedent(
    """
    from __future__ import annotations

    import torch
    import torch.nn.functional as F


    LOSS_VARIANT = "bce_finite_guard_v1"


    def initialize_loss(*, train_frame, device, rank, world_size):
        if len(train_frame) != 347840:
            raise ValueError("BGE BCE hook received an unexpected training row count")
        return None


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
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite BGE training logits")
        if not torch.isfinite(targets).all():
            raise FloatingPointError("non-finite BGE training targets")
        if not torch.isfinite(sample_weights).all():
            raise FloatingPointError("non-finite BGE training sample weights")
        denominator = sample_weights.sum()
        if not torch.isfinite(denominator) or denominator <= 0:
            raise FloatingPointError("invalid BGE loss denominator")
        per_example = F.binary_cross_entropy_with_logits(
            logits.float(), targets, reduction="none"
        )
        loss = (per_example * sample_weights).sum() / denominator
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite BGE BCE loss")
        return {"loss": loss, "bce": loss.detach()}
    """
).strip() + "\n"
FIXED_LOSS_HOOK_SHA256 = hashlib.sha256(
    FIXED_LOSS_HOOK_SOURCE.encode("utf-8")
).hexdigest()

RUNTIME_EMBEDDED_FILES = validation_builder.EMBEDDED_FILES + (
    Path("scripts/train_bge_2ep_sft.py"),
)

# Generation-only dependencies are identity authority, but are deliberately
# not embedded as runtime source literals.  In particular, embedding this
# generator inside its own notebook made placeholder substitution rewrite the
# literal after its digest had been computed.  The ordered per-file ledger
# binds every dependency without that self-reference.
SOURCE_LEDGER_FILES = RUNTIME_EMBEDDED_FILES + (
    Path("src/google_sheets_logger.py"),
    Path("scripts/create_cross_encoder_training_notebook.py"),
    Path("scripts/create_minilm_validation_baseline_notebook.py"),
    Path("scripts/create_qwen_training_notebook.py"),
    Path("scripts/push_bge_pretrain_checkpoint_dataset.py"),
    Path("scripts/push_kaggle_training_dataset.py"),
    Path("scripts/push_minilm_pretrain_checkpoint_dataset.py"),
    Path("scripts/run_kaggle_notebook.py"),
    Path("scripts/create_bge_2ep_sft_notebooks.py"),
)


class CampaignConfigError(ValueError):
    """Raised locally when a frozen BGE campaign input has drifted."""


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return checkpoint_push.sha256_file(path)


def source_bundle() -> tuple[dict[str, str], list[dict[str, Any]], str]:
    """Return runtime contents plus an ordered ledger for every identity source."""
    runtime_paths = {relative.as_posix() for relative in RUNTIME_EMBEDDED_FILES}
    sources: dict[str, str] = {}
    ledger: list[dict[str, Any]] = []
    seen: set[str] = set()
    for relative in SOURCE_LEDGER_FILES:
        relative_name = relative.as_posix()
        if relative_name in seen:
            raise CampaignConfigError(f"Duplicate BGE source ledger path: {relative_name}")
        seen.add(relative_name)
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"Required BGE source is missing: {source}")
        payload = source.read_bytes()
        try:
            content = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise CampaignConfigError(
                f"BGE source ledger file is not UTF-8: {relative_name}"
            ) from error
        ledger.append(
            {
                "path": relative_name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "runtime_embedded": relative_name in runtime_paths,
            }
        )
        if relative_name in runtime_paths:
            if (
                IDENTITY_PLACEHOLDER in content
                or EXECUTABLE_CELLS_PLACEHOLDER in content
            ):
                raise CampaignConfigError(
                    f"Runtime BGE source contains an identity placeholder: {relative_name}"
                )
            sources[relative_name] = content
    if set(sources) != runtime_paths:
        raise CampaignConfigError("BGE runtime source set differs from its ledger")
    source_sha256 = canonical_sha256(
        {"schema_version": 1, "files": ledger}
    )
    return sources, ledger, source_sha256


def embedded_sources() -> tuple[dict[str, str], str]:
    """Compatibility view: runtime sources and full-ledger identity digest."""
    sources, _, source_sha256 = source_bundle()
    return sources, source_sha256


def load_validation_dataset(source_dir: Path, owner: str) -> dict[str, Any]:
    dataset = validation_builder.load_manifest(source_dir, owner)
    if dataset["manifest_sha256"] != EXPECTED_VALIDATION_MANIFEST_SHA256:
        raise CampaignConfigError("Frozen validation manifest SHA-256 has changed")
    manifest = dataset["manifest"]
    if tuple(manifest.get("ood_categories", [])) != EXPECTED_OOD_CATEGORIES:
        raise CampaignConfigError("Frozen validation OOD categories have changed")
    split_manifest = manifest.get("human", {}).get("splits", {})
    expected_rows = {
        "train": EXPECTED_HUMAN_TRAIN,
        "iid_validation": EXPECTED_IID,
        "hard_validation": EXPECTED_HARD,
        "ood_validation": EXPECTED_FORMER_OOD,
    }
    for split, rows in expected_rows.items():
        if split_manifest.get(split, {}).get("pairs") != rows:
            raise CampaignConfigError(
                f"Frozen validation split {split!r} has an unexpected size"
            )
    return dict(dataset)


def _validate_file_declaration(name: str, declaration: object) -> dict[str, Any]:
    if not isinstance(declaration, Mapping):
        raise CampaignConfigError(f"Checkpoint declaration is invalid: {name}")
    size = declaration.get("bytes")
    digest = declaration.get("sha256")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise CampaignConfigError(f"Checkpoint byte count is invalid: {name}")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise CampaignConfigError(f"Checkpoint SHA-256 is invalid: {name}")
    return {"bytes": size, "sha256": digest}


def load_checkpoint_dataset(
    stage_dir: Path,
    owner: str,
    *,
    verify_payload: bool = True,
) -> dict[str, Any]:
    """Load the exact staged uploader manifest without creating a new payload."""
    manifest_path = stage_dir / checkpoint_push.MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "BGE checkpoint staging manifest is missing. Run "
            "scripts/push_bge_pretrain_checkpoint_dataset.py --dry-run first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise CampaignConfigError("Unsupported BGE checkpoint manifest schema")
    expected_ref = f"{owner}/{checkpoint_push.DATASET_SLUG}"
    if manifest.get("dataset") != expected_ref or manifest.get("is_private") is not True:
        raise CampaignConfigError("BGE checkpoint manifest has unexpected Dataset identity")

    expected_checkpoint_files = {
        name: dict(declaration)
        for name, declaration in checkpoint_push.EXPECTED_SOURCE_FILES.items()
    }
    if manifest.get("checkpoint_files") != expected_checkpoint_files:
        raise CampaignConfigError("BGE checkpoint manifest changed the exact source ledger")
    reconstruction = manifest.get("reconstruction")
    if not isinstance(reconstruction, Mapping):
        raise CampaignConfigError("BGE checkpoint manifest has no reconstruction ledger")
    expected_model = expected_checkpoint_files[checkpoint_push.MODEL_FILENAME]
    if (
        reconstruction.get("filename") != checkpoint_push.MODEL_FILENAME
        or reconstruction.get("bytes") != expected_model["bytes"]
        or reconstruction.get("sha256") != expected_model["sha256"]
        or reconstruction.get("part_bytes") != checkpoint_push.MODEL_PART_BYTES
    ):
        raise CampaignConfigError("BGE checkpoint reconstruction contract has changed")
    parts = reconstruction.get("parts")
    if not isinstance(parts, list) or not parts:
        raise CampaignConfigError("BGE checkpoint reconstruction has no ordered parts")
    if parts != [f"model.safetensors.part{index:03d}" for index in range(len(parts))]:
        raise CampaignConfigError("BGE checkpoint part order is not contiguous")

    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise CampaignConfigError("BGE checkpoint manifest has no uploaded file ledger")
    expected_names = set(checkpoint_push.DIRECT_CHECKPOINT_FILES) | set(parts)
    if set(files) != expected_names:
        raise CampaignConfigError("BGE checkpoint uploaded file set has changed")
    normalized_files = {
        name: _validate_file_declaration(name, declaration)
        for name, declaration in files.items()
    }
    if sum(normalized_files[name]["bytes"] for name in parts) != expected_model["bytes"]:
        raise CampaignConfigError("BGE checkpoint part sizes do not reconstruct the model")
    for name in checkpoint_push.DIRECT_CHECKPOINT_FILES:
        if normalized_files[name] != expected_checkpoint_files[name]:
            raise CampaignConfigError(f"BGE direct checkpoint file changed: {name}")

    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping) or (
        provenance.get("status") != "user_supplied_unverified"
        or provenance.get("training_lineage_verified") is not False
        or provenance.get("declared_pretraining_epochs") != 2
    ):
        raise CampaignConfigError("BGE checkpoint provenance declaration has changed")

    if verify_payload:
        for name, declaration in normalized_files.items():
            path = stage_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"Staged BGE checkpoint file is missing: {name}")
            if path.stat().st_size != declaration["bytes"]:
                raise CampaignConfigError(f"Staged BGE checkpoint size differs: {name}")
            if file_sha256(path) != declaration["sha256"]:
                raise CampaignConfigError(f"Staged BGE checkpoint SHA-256 differs: {name}")

    return {
        "dataset": expected_ref,
        "manifest_sha256": file_sha256(manifest_path),
        "manifest": manifest,
    }


def validate_base_config(config: Mapping[str, Any]) -> None:
    actual = dict(config)
    expected = EXPECTED_BASE_CONFIG
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatches = {
        key: {"actual": actual.get(key), "expected": value}
        for key, value in expected.items()
        if key in actual and actual[key] != value
    }
    effective_batch = (
        int(actual.get("batch_size", 0))
        * EXPECTED_WORLD_SIZE
        * int(actual.get("gradient_accumulation", 0))
    )
    if effective_batch != EXPECTED_EFFECTIVE_BATCH:
        mismatches["effective_batch"] = {
            "actual": effective_batch,
            "expected": EXPECTED_EFFECTIVE_BATCH,
        }
    actual_canonical_sha256 = canonical_sha256(actual)
    if actual_canonical_sha256 != EXPECTED_BASE_CONFIG_CANONICAL_SHA256:
        mismatches["canonical_sha256"] = {
            "actual": actual_canonical_sha256,
            "expected": EXPECTED_BASE_CONFIG_CANONICAL_SHA256,
        }
    if missing or unexpected or mismatches:
        raise CampaignConfigError(
            "Frozen BGE baseline config differs: "
            + json.dumps(
                {
                    "missing": missing,
                    "unexpected": unexpected,
                    "mismatches": mismatches,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )


def resolve_variant_config(
    base_config: Mapping[str, Any], variant: Mapping[str, Any]
) -> dict[str, Any]:
    config = deepcopy(dict(base_config))
    overrides = variant.get("overrides", {})
    if not isinstance(overrides, Mapping) or set(overrides) - {"learning_rate"}:
        raise CampaignConfigError("Initial BGE variants may change only learning_rate")
    config.update(overrides)
    # Every initial candidate keeps the same memory-safe geometry and one epoch.
    geometry = {
        "epochs": 1,
        "batch_size": 8,
        "eval_batch_size": 32,
        "gradient_accumulation": 12,
        "max_length": 384,
        "gradient_checkpointing": True,
        "attention_implementation": "sdpa",
        "seed": 42,
    }
    if any(config.get(key) != value for key, value in geometry.items()):
        raise CampaignConfigError("Initial BGE log-LR line changed frozen geometry")
    learning_rate = config.get("learning_rate")
    if learning_rate not in {1e-5, 2e-5, 4e-5}:
        raise CampaignConfigError("Initial BGE learning rate is outside the frozen log line")
    return config


def variant_identity(
    *,
    variant: Mapping[str, Any],
    config: Mapping[str, Any],
    source_sha256: str,
    validation_manifest_sha256: str,
    checkpoint_manifest_sha256: str,
    executable_cells_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "schema_version": 1,
            "campaign": CAMPAIGN,
            "experiment": variant["experiment"],
            "role": variant["role"],
            "config": dict(config),
            "source_sha256": source_sha256,
            "validation_manifest_sha256": validation_manifest_sha256,
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "executable_cells_sha256": executable_cells_sha256,
            "train_policy": "human_train_plus_former_ood_exact_concat_v1",
            "validation_policy": "iid_hard_only_ood_minus_one_v1",
            "loss_hook_sha256": FIXED_LOSS_HOOK_SHA256,
        }
    )


def kernel_slug(variant: Mapping[str, Any], identity_sha256: str) -> str:
    slug = f"pm-b2-{variant['slug_token']}-{identity_sha256[:12]}-s42-v1"
    if len(slug) > 50 or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug):
        raise CampaignConfigError(f"Unsafe BGE Kaggle slug: {slug!r}")
    return slug


def variant_notes(
    *,
    variant: Mapping[str, Any],
    config: Mapping[str, Any],
    identity_sha256: str,
) -> str:
    return canonical_json(
        {
            "campaign": CAMPAIGN,
            "role": variant["role"],
            "identity_sha256": identity_sha256,
            "model_family": "bge-reranker-v2-m3",
            "initial_checkpoint": "model/pretrain_bge_2ep",
            "train_policy": "306669 human train + 41171 former OOD; exact concat",
            "validation_policy": "IID and hard only; OOD metrics forced to -1",
            "search_strategy": "baseline_then_one_dimensional_log2_lr_line",
            "loss_variant": "bce_finite_guard_v1",
            "epochs": config["epochs"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config["weight_decay"],
            "warmup_ratio": config["warmup_ratio"],
            "label_smoothing": config["label_smoothing"],
            "max_grad_norm": config["max_grad_norm"],
            "batch_size_per_gpu": config["batch_size"],
            "world_size": EXPECTED_WORLD_SIZE,
            "gradient_accumulation": config["gradient_accumulation"],
            "effective_batch": EXPECTED_EFFECTIVE_BATCH,
            "max_length": config["max_length"],
            "gradient_checkpointing": config["gradient_checkpointing"],
            "adamw_foreach": False,
            "seed": config["seed"],
        }
    )


def markdown(source: str, *tags: str) -> nbf.NotebookNode:
    cell = nbf.v4.new_markdown_cell(dedent(source).strip())
    if tags:
        cell.metadata["tags"] = list(tags)
    return cell


def code(source: str, *tags: str) -> nbf.NotebookNode:
    cell = nbf.v4.new_code_cell(dedent(source).strip())
    if tags:
        cell.metadata["tags"] = list(tags)
    return cell


def executable_cells_sha256(
    notebook: nbf.NotebookNode,
    *,
    identity_sha256: str = IDENTITY_PLACEHOLDER,
    expected_sha256: str = EXECUTABLE_CELLS_PLACEHOLDER,
) -> str:
    """Hash final executable cell sources while normalizing self references."""
    if identity_sha256 != IDENTITY_PLACEHOLDER and not re.fullmatch(
        r"[0-9a-f]{64}", identity_sha256
    ):
        raise CampaignConfigError("Invalid campaign identity used for cell hashing")
    if expected_sha256 != EXECUTABLE_CELLS_PLACEHOLDER and not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise CampaignConfigError("Invalid executable-cell hash used for normalization")
    payload: list[dict[str, Any]] = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        source = str(cell.source)
        if identity_sha256 != IDENTITY_PLACEHOLDER:
            source = source.replace(identity_sha256, IDENTITY_PLACEHOLDER)
        if expected_sha256 != EXECUTABLE_CELLS_PLACEHOLDER:
            source = source.replace(expected_sha256, EXECUTABLE_CELLS_PLACEHOLDER)
        payload.append(
            {
                "cell_index": index,
                "source": source,
                "tags": list(cell.metadata.get("tags", [])),
            }
        )
    return canonical_sha256({"schema_version": 1, "code_cells": payload})


def _replace_notebook_placeholders(
    notebook: nbf.NotebookNode,
    *,
    identity_sha256: str,
    executable_sha256: str,
) -> None:
    for cell in notebook.cells:
        cell.source = str(cell.source).replace(
            IDENTITY_PLACEHOLDER, identity_sha256
        ).replace(EXECUTABLE_CELLS_PLACEHOLDER, executable_sha256)


def _assign_deterministic_cell_ids(notebook: nbf.NotebookNode) -> None:
    """Keep byte-level notebook freezes stable across repeated campaign builds."""
    for index, cell in enumerate(notebook.cells):
        payload = (
            f"{index}\0{cell.cell_type}\0{str(cell.source)}"
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()[:12]
        cell["id"] = f"bge-{index:02d}-{digest}"


def _setup_cell(
    *,
    experiment: str,
    validation: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    source_sha256: str,
    identity_sha256: str,
    executable_sha256: str,
) -> nbf.NotebookNode:
    validation_ref = str(validation["dataset"])
    checkpoint_ref = str(checkpoint["dataset"])
    return code(
        f"""
        import hashlib
        import json
        import math
        import os
        import shutil
        import subprocess
        import sys
        import time
        from pathlib import Path

        INPUT_ROOT = Path("/kaggle/input")
        WORKING_ROOT = Path("/kaggle/working")
        TEMP_ROOT = Path({f'/kaggle/temp/{experiment}'!r})
        PROJECT_ROOT = WORKING_ROOT / "product_matching"
        PREPARED_DIR = TEMP_ROOT / "prepared"
        TOKEN_CACHE_DIR = TEMP_ROOT / "token_cache"
        TRAINER_OUTPUT_DIR = TEMP_ROOT / "trainer_output"
        OUTPUT_DIR = WORKING_ROOT / {experiment!r}
        RUNTIME_CONFIG_PATH = WORKING_ROOT / "cross_encoder_config.json"
        LOSS_HOOK_PATH = PROJECT_ROOT / "bge_sft_loss_hook.py"
        TRAIN_LOG = WORKING_ROOT / {f'{experiment}.log'!r}
        PREFLIGHT_LOG = WORKING_ROOT / {f'{experiment}_memory_preflight.log'!r}
        PREFLIGHT_REPORT_PATH = WORKING_ROOT / "bge_memory_preflight.json"
        TRAIN_DATA_REPORT_PATH = WORKING_ROOT / "bge_train_data_report.json"
        RUNTIME_VERSIONS_PATH = WORKING_ROOT / "bge_runtime_versions.json"

        EXPECTED_VALIDATION_DATASET_REF = {validation_ref!r}
        EXPECTED_VALIDATION_DATASET_SLUG = {VALIDATION_DATASET_SLUG!r}
        EXPECTED_VALIDATION_MANIFEST_SHA256 = {str(validation['manifest_sha256'])!r}
        EXPECTED_CHECKPOINT_REF = {checkpoint_ref!r}
        EXPECTED_CHECKPOINT_SLUG = {CHECKPOINT_DATASET_SLUG!r}
        EXPECTED_CHECKPOINT_MANIFEST_SHA256 = {str(checkpoint['manifest_sha256'])!r}
        EXPECTED_SOURCE_SHA256 = {source_sha256!r}
        EXPECTED_CAMPAIGN_IDENTITY_SHA256 = {identity_sha256!r}
        EXPECTED_EXECUTABLE_CELLS_SHA256 = {executable_sha256!r}
        REMOTE_FILES = {validation_builder.REMOTE_FILES!r}

        def file_sha256(path):
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        def dataset_file(dataset_slug, filename):
            direct = INPUT_ROOT / dataset_slug / filename
            if direct.is_file():
                return direct
            candidates = [
                path for path in INPUT_ROOT.glob(f"**/{{filename}}")
                if dataset_slug in path.parts
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Expected one {{filename!r}} in attached Dataset "
                    f"{{dataset_slug!r}}, found {{candidates}}"
                )
            return candidates[0]

        validation_manifest_path = dataset_file(
            EXPECTED_VALIDATION_DATASET_SLUG,
            "validation_splits_manifest.json",
        )
        if file_sha256(validation_manifest_path) != EXPECTED_VALIDATION_MANIFEST_SHA256:
            raise RuntimeError("Attached validation manifest has changed")
        validation_manifest = json.loads(
            validation_manifest_path.read_text(encoding="utf-8")
        )
        if validation_manifest.get("ood_categories") != ["Одежда", "Бытовая техника"]:
            raise RuntimeError("Unexpected former OOD categories")
        attached_files = {{
            relative: dataset_file(EXPECTED_VALIDATION_DATASET_SLUG, remote_name)
            for relative, remote_name in REMOTE_FILES.items()
        }}
        for relative, path in attached_files.items():
            expected = validation_manifest["outputs"][relative]
            if path.stat().st_size != expected["bytes"] or file_sha256(path) != expected["sha256"]:
                raise RuntimeError(f"Attached validation file differs: {{relative}}")

        checkpoint_manifest_path = dataset_file(
            EXPECTED_CHECKPOINT_SLUG,
            "checkpoint_manifest.json",
        )
        if file_sha256(checkpoint_manifest_path) != EXPECTED_CHECKPOINT_MANIFEST_SHA256:
            raise RuntimeError("Attached BGE checkpoint manifest has changed")
        checkpoint_manifest = json.loads(
            checkpoint_manifest_path.read_text(encoding="utf-8")
        )
        if (
            checkpoint_manifest.get("dataset") != EXPECTED_CHECKPOINT_REF
            or checkpoint_manifest.get("is_private") is not True
            or checkpoint_manifest.get("schema_version") != 1
        ):
            raise RuntimeError("Attached BGE checkpoint identity differs")
        checkpoint_root = checkpoint_manifest_path.parent
        for filename, declaration in checkpoint_manifest["files"].items():
            checkpoint_file = checkpoint_root / filename
            if not checkpoint_file.is_file():
                raise RuntimeError(f"BGE checkpoint shard is missing: {{filename}}")
            if (
                checkpoint_file.stat().st_size != declaration["bytes"]
                or file_sha256(checkpoint_file) != declaration["sha256"]
            ):
                raise RuntimeError(f"BGE checkpoint shard differs: {{filename}}")

        reconstruction = checkpoint_manifest["reconstruction"]
        reconstructed_root = TEMP_ROOT / "initial_checkpoint"
        reconstructed_root.mkdir(parents=True, exist_ok=True)
        part_names = set(reconstruction["parts"])
        for filename in checkpoint_manifest["files"]:
            if filename in part_names:
                continue
            destination = reconstructed_root / filename
            destination.unlink(missing_ok=True)
            destination.symlink_to(checkpoint_root / filename)
        reconstructed_model = reconstructed_root / reconstruction["filename"]
        reconstructed_digest = hashlib.sha256()
        with reconstructed_model.open("wb") as destination:
            for part_name in reconstruction["parts"]:
                with (checkpoint_root / part_name).open("rb") as source:
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        destination.write(chunk)
                        reconstructed_digest.update(chunk)
        if (
            reconstructed_model.stat().st_size != reconstruction["bytes"]
            or reconstructed_digest.hexdigest() != reconstruction["sha256"]
        ):
            raise RuntimeError("Reconstructed BGE model differs from the source checkpoint")
        INITIAL_MODEL_PATH = reconstructed_root

        print(json.dumps({{
            "validation_dataset": EXPECTED_VALIDATION_DATASET_REF,
            "validation_manifest_sha256": EXPECTED_VALIDATION_MANIFEST_SHA256,
            "checkpoint_dataset": EXPECTED_CHECKPOINT_REF,
            "checkpoint_manifest_sha256": EXPECTED_CHECKPOINT_MANIFEST_SHA256,
            "reconstructed_model": str(reconstructed_model),
        }}, ensure_ascii=False, indent=2))
        print(subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True).stdout)
        """,
        "frozen",
        "environment-guard",
    )


def _recipe_cell(config: Mapping[str, Any], recipe_sha256: str) -> nbf.NotebookNode:
    return code(
        f"""
        CANONICAL_TRAIN_CONFIG = {dict(config)!r}
        EXPECTED_TRAIN_RECIPE_SHA256 = {recipe_sha256!r}

        def canonical_json_sha256(value):
            payload = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            return hashlib.sha256(payload).hexdigest()

        if canonical_json_sha256(CANONICAL_TRAIN_CONFIG) != EXPECTED_TRAIN_RECIPE_SHA256:
            raise RuntimeError("Frozen BGE SFT recipe was edited")
        if (
            CANONICAL_TRAIN_CONFIG["batch_size"] != 8
            or CANONICAL_TRAIN_CONFIG["eval_batch_size"] != 32
            or CANONICAL_TRAIN_CONFIG["gradient_accumulation"] != 12
            or CANONICAL_TRAIN_CONFIG["max_length"] != 384
            or CANONICAL_TRAIN_CONFIG["gradient_checkpointing"] is not True
            or 8 * 2 * 12 != 192
        ):
            raise RuntimeError("Frozen BGE 2xT4 memory geometry was edited")
        TRAIN_CONFIG = dict(CANONICAL_TRAIN_CONFIG)
        TRAIN_CONFIG["model"] = str(INITIAL_MODEL_PATH)
        RUNTIME_CONFIG_PATH.write_text(
            json.dumps(TRAIN_CONFIG, ensure_ascii=False, indent=2) + "\\n",
            encoding="utf-8",
        )
        print(json.dumps({{
            "frozen_recipe_sha256": EXPECTED_TRAIN_RECIPE_SHA256,
            "runtime_config": TRAIN_CONFIG,
        }}, ensure_ascii=False, indent=2))
        """,
        "frozen",
        "frozen-recipe",
    )


def _sources_cell(
    sources: Mapping[str, str],
    source_ledger: list[dict[str, Any]],
    source_sha256: str,
) -> nbf.NotebookNode:
    return code(
        f"""
        EMBEDDED_SOURCES = {dict(sources)!r}
        SOURCE_LEDGER = {source_ledger!r}
        EXPECTED_SOURCE_LEDGER_SHA256 = {source_sha256!r}
        if EXPECTED_SOURCE_LEDGER_SHA256 != EXPECTED_SOURCE_SHA256:
            raise RuntimeError("BGE setup/source-ledger identities differ")
        source_ledger_payload = json.dumps(
            {{"schema_version": 1, "files": SOURCE_LEDGER}},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if hashlib.sha256(source_ledger_payload).hexdigest() != EXPECTED_SOURCE_LEDGER_SHA256:
            raise RuntimeError("BGE ordered source ledger was edited")
        ledger_by_path = {{record["path"]: record for record in SOURCE_LEDGER}}
        if len(ledger_by_path) != len(SOURCE_LEDGER):
            raise RuntimeError("BGE source ledger contains duplicate paths")
        expected_runtime_paths = {{
            record["path"]
            for record in SOURCE_LEDGER
            if record["runtime_embedded"] is True
        }}
        if set(EMBEDDED_SOURCES) != expected_runtime_paths:
            raise RuntimeError("Embedded BGE runtime source set differs from its ledger")
        for relative, content in EMBEDDED_SOURCES.items():
            declaration = ledger_by_path[relative]
            payload = content.encode("utf-8")
            if (
                len(payload) != declaration["bytes"]
                or hashlib.sha256(payload).hexdigest() != declaration["sha256"]
            ):
                raise RuntimeError(f"Embedded BGE runtime source differs: {{relative}}")

        PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
        for relative, content in EMBEDDED_SOURCES.items():
            destination = PROJECT_ROOT / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
            declaration = ledger_by_path[relative]
            if (
                destination.stat().st_size != declaration["bytes"]
                or file_sha256(destination) != declaration["sha256"]
            ):
                raise RuntimeError(f"Materialized BGE runtime source differs: {{relative}}")
        pip_result = subprocess.run(
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
        if pip_result.returncode != 0:
            raise RuntimeError("BGE dependency bootstrap did not complete")

        # Import extension-backed scientific packages only after pip has fully
        # returned. Replacing an already imported NumPy/Pandas/PyArrow binary in
        # the live notebook process can terminate the Kaggle kernel.
        from importlib import metadata as importlib_metadata

        import numpy as np
        import pandas as pd

        RUNTIME_VERSIONS = {{
            "schema_version": 1,
            "python": sys.version.split()[0],
            "packages": {{
                distribution: importlib_metadata.version(distribution)
                for distribution in (
                    "numpy",
                    "pandas",
                    "pyarrow",
                    "scikit-learn",
                    "torch",
                    "transformers",
                )
            }},
        }}
        if (
            RUNTIME_VERSIONS["packages"]["numpy"] != np.__version__
            or RUNTIME_VERSIONS["packages"]["pandas"] != pd.__version__
        ):
            raise RuntimeError("BGE imported scientific versions differ from metadata")
        RUNTIME_VERSIONS_PATH.write_text(
            json.dumps(RUNTIME_VERSIONS, ensure_ascii=False, indent=2) + "\\n",
            encoding="utf-8",
        )
        print(json.dumps({{"runtime_versions": RUNTIME_VERSIONS}}, sort_keys=True))
        """,
        "frozen",
        "embedded-sources",
    )


def _data_cell() -> nbf.NotebookNode:
    return code(
        f"""
        items = pd.read_parquet(attached_files["human/items.parquet"])
        human_train = pd.read_parquet(attached_files["human/train_pairs.parquet"])
        iid_validation = pd.read_parquet(
            attached_files["human/iid_validation_pairs.parquet"]
        )
        hard_validation = pd.read_parquet(
            attached_files["human/hard_validation_pairs.parquet"]
        )
        former_ood = pd.read_parquet(
            attached_files["human/ood_validation_pairs.parquet"]
        )

        if len(items) != {EXPECTED_ITEMS} or not items["id"].is_unique:
            raise RuntimeError("Frozen human item table changed")
        expected_pair_rows = {{
            "human_train": {EXPECTED_HUMAN_TRAIN},
            "iid": {EXPECTED_IID},
            "hard": {EXPECTED_HARD},
            "former_ood": {EXPECTED_FORMER_OOD},
        }}
        pair_frames = {{
            "human_train": human_train,
            "iid": iid_validation,
            "hard": hard_validation,
            "former_ood": former_ood,
        }}
        item_categories = items.set_index("id")["category"]

        def validate_pairs(name, frame):
            required = {{"id1", "id2", "target"}}
            if required - set(frame):
                raise RuntimeError(f"{{name}} is missing pair columns")
            if len(frame) != expected_pair_rows[name]:
                raise RuntimeError(f"{{name}} row count changed")
            if frame[list(required)].isnull().any().any():
                raise RuntimeError(f"{{name}} contains null pair fields")
            if (frame["id1"] == frame["id2"]).any():
                raise RuntimeError(f"{{name}} contains self-pairs")
            if not frame["target"].isin([0.0, 1.0]).all():
                raise RuntimeError(f"{{name}} targets are not binary")
            lower = np.minimum(frame["id1"].to_numpy(), frame["id2"].to_numpy())
            upper = np.maximum(frame["id1"].to_numpy(), frame["id2"].to_numpy())
            if pd.MultiIndex.from_arrays([lower, upper]).duplicated().any():
                raise RuntimeError(f"{{name}} contains duplicate unordered pairs")
            left_category = frame["id1"].map(item_categories)
            right_category = frame["id2"].map(item_categories)
            if left_category.isnull().any() or right_category.isnull().any():
                raise RuntimeError(f"{{name}} references missing items")
            if not left_category.equals(right_category):
                raise RuntimeError(f"{{name}} contains cross-category pairs")
            return set(frame["id1"]) | set(frame["id2"]), set(left_category.astype(str))

        ids_and_categories = {{
            name: validate_pairs(name, frame)
            for name, frame in pair_frames.items()
        }}
        if ids_and_categories["former_ood"][1] != {{"Одежда", "Бытовая техника"}}:
            raise RuntimeError("Former OOD split no longer has exactly two categories")

        human_train = human_train.copy()
        human_train["label_source"] = "human_train"
        former_ood = former_ood.copy()
        former_ood["label_source"] = "human_former_ood"
        train_pairs = pd.concat(
            [human_train, former_ood],
            axis=0,
            ignore_index=True,
            sort=False,
        )
        train_ids, train_categories = validate_pairs("human_train", human_train)
        former_ood_ids, _ = validate_pairs("former_ood", former_ood)
        combined_train_ids = train_ids | former_ood_ids
        for validation_name in ("iid", "hard"):
            overlap = combined_train_ids & ids_and_categories[validation_name][0]
            if overlap:
                raise RuntimeError(
                    f"Combined BGE train leaks {{len(overlap)}} item IDs into {{validation_name}}"
                )
        lower = np.minimum(train_pairs["id1"].to_numpy(), train_pairs["id2"].to_numpy())
        upper = np.maximum(train_pairs["id1"].to_numpy(), train_pairs["id2"].to_numpy())
        if pd.MultiIndex.from_arrays([lower, upper]).duplicated().any():
            raise RuntimeError("Combined BGE train contains duplicate unordered pairs")
        if len(train_pairs) != {EXPECTED_TRAIN}:
            raise RuntimeError("Combined BGE train row count changed")
        train_positives = int(train_pairs["target"].sum())
        if train_positives != {EXPECTED_TRAIN_POSITIVES}:
            raise RuntimeError("Combined BGE train positive count changed")
        source_counts = {{
            str(key): int(value)
            for key, value in train_pairs["label_source"].value_counts().items()
        }}
        expected_source_counts = {{
            "human_train": {EXPECTED_HUMAN_TRAIN},
            "human_former_ood": {EXPECTED_FORMER_OOD},
        }}
        if source_counts != expected_source_counts:
            raise RuntimeError("Combined BGE train source counts changed")

        PREPARED_DIR.mkdir(parents=True, exist_ok=True)
        train_pairs.to_parquet(
            PREPARED_DIR / "train_pairs.parquet", index=False, compression="zstd"
        )
        for source, destination_name in (
            (attached_files["human/items.parquet"], "items.parquet"),
            (attached_files["human/iid_validation_pairs.parquet"], "iid_validation_pairs.parquet"),
            (attached_files["human/hard_validation_pairs.parquet"], "hard_validation_pairs.parquet"),
        ):
            destination = PREPARED_DIR / destination_name
            destination.unlink(missing_ok=True)
            destination.symlink_to(source)
        if (PREPARED_DIR / "ood_validation_pairs.parquet").exists():
            raise RuntimeError("OOD validation parquet must not enter the BGE trainer")

        TRAIN_DATA_REPORT = {{
            "schema_version": 1,
            "policy": "human_train_plus_former_ood_exact_concat_v1",
            "items": len(items),
            "train_pairs": len(train_pairs),
            "train_positives": train_positives,
            "train_positive_rate": float(train_pairs["target"].mean()),
            "source_counts": source_counts,
            "former_ood_categories": sorted(ids_and_categories["former_ood"][1]),
            "validation_rows": {{"iid": len(iid_validation), "hard": len(hard_validation)}},
            "validation_item_overlap": {{"iid": 0, "hard": 0}},
            "ood_evaluation": "disabled_train_contaminated",
        }}
        TRAIN_DATA_REPORT_PATH.write_text(
            json.dumps(TRAIN_DATA_REPORT, ensure_ascii=False, indent=2) + "\\n",
            encoding="utf-8",
        )
        print(json.dumps(TRAIN_DATA_REPORT, ensure_ascii=False, indent=2))
        del items, human_train, former_ood, train_pairs, iid_validation, hard_validation
        """,
        "frozen",
        "frozen-data",
    )


def _loss_cell() -> nbf.NotebookNode:
    return code(
        f"""
        FIXED_LOSS_HOOK_SOURCE = {FIXED_LOSS_HOOK_SOURCE!r}
        EXPECTED_LOSS_HOOK_SHA256 = {FIXED_LOSS_HOOK_SHA256!r}
        if hashlib.sha256(FIXED_LOSS_HOOK_SOURCE.encode("utf-8")).hexdigest() != EXPECTED_LOSS_HOOK_SHA256:
            raise RuntimeError("Frozen BGE BCE hook was edited")
        LOSS_HOOK_PATH.write_text(FIXED_LOSS_HOOK_SOURCE, encoding="utf-8")
        if file_sha256(LOSS_HOOK_PATH) != EXPECTED_LOSS_HOOK_SHA256:
            raise RuntimeError("Materialized BGE BCE hook differs")
        """,
        "frozen",
        "fixed-loss-hook",
    )


def _streaming_process_helper() -> str:
    return dedent(
        """
        def run_logged(command, log_path):
            print("$", " ".join(str(value) for value in command), flush=True)
            environment = os.environ.copy()
            environment.update({
                "OMP_NUM_THREADS": "2",
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "PYTHONUNBUFFERED": "1",
            })
            with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log_file.write(line)
                return_code = process.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, command)
        """
    ).strip()


def _preflight_cell() -> nbf.NotebookNode:
    helper = _streaming_process_helper()
    return code(
        helper
        + "\n\n"
        + dedent(
            f"""
            preflight_command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                str(PROJECT_ROOT / "scripts/train_bge_2ep_sft.py"),
                "--memory-preflight-only",
                "--config", str(RUNTIME_CONFIG_PATH),
                "--preflight-report", str(PREFLIGHT_REPORT_PATH),
            ]
            run_logged(preflight_command, PREFLIGHT_LOG)
            if not PREFLIGHT_REPORT_PATH.is_file():
                raise RuntimeError("BGE one-step memory preflight produced no report")
            memory_preflight = json.loads(
                PREFLIGHT_REPORT_PATH.read_text(encoding="utf-8")
            )
            if (
                memory_preflight.get("status") != "passed"
                or memory_preflight.get("model") != str(INITIAL_MODEL_PATH)
                or memory_preflight.get("world_size") != 2
                or memory_preflight.get("parameters") != {EXPECTED_PARAMETERS}
                or memory_preflight.get("microbatch_per_gpu") != 8
                or memory_preflight.get("max_length") != 384
                or memory_preflight.get("gradient_accumulation") != 12
                or memory_preflight.get("accumulated_microbatches") != 12
                or memory_preflight.get("loss_divisor_per_microbatch") != 12
                or memory_preflight.get("ddp_no_sync_microbatches") != 11
                or memory_preflight.get("ddp_sync_microbatches") != 1
                or memory_preflight.get("effective_batch") != 192
                or memory_preflight.get("eval_batch_per_gpu") != 32
                or memory_preflight.get("eval_probe_after_optimizer_state") is not True
                or memory_preflight.get("gradient_checkpointing") is not True
                or memory_preflight.get("adamw_foreach") is not False
                or memory_preflight.get("gradient_clip_foreach") is not False
                or memory_preflight.get("nonfinite_gradient_policy")
                    != {EXPECTED_AMP_NONFINITE_POLICY!r}
                or memory_preflight.get("amp_max_attempts")
                    != {EXPECTED_PREFLIGHT_AMP_ATTEMPTS}
                or memory_preflight.get("optimizer_state")
                    != "adamw_exp_avg_and_exp_avg_sq_materialized"
                or memory_preflight.get("optimizer_state_parameters_per_rank")
                    != {EXPECTED_TRAINABLE_PARAMETER_TENSORS}
                or memory_preflight.get("optimizer_state_tensor_elements_per_rank")
                    != 2 * {EXPECTED_PARAMETERS}
                or len(memory_preflight.get("ranks", [])) != 2
            ):
                raise RuntimeError("BGE one-step memory preflight contract failed")
            print(json.dumps(memory_preflight, ensure_ascii=False, indent=2))
            """
        ).strip(),
        "frozen",
        "memory-preflight",
    )


def _training_cell() -> nbf.NotebookNode:
    return code(
        """
        if memory_preflight.get("status") != "passed":
            raise RuntimeError("Full BGE training cannot start before memory preflight")
        train_command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(PROJECT_ROOT / "scripts/train_bge_2ep_sft.py"),
            "--config", str(RUNTIME_CONFIG_PATH),
            "--prepared-dir", str(PREPARED_DIR),
            "--output-dir", str(TRAINER_OUTPUT_DIR),
            "--token-cache-dir", str(TOKEN_CACHE_DIR),
            "--loss-hook", str(LOSS_HOOK_PATH),
            "--validation-split", "iid=iid_validation_pairs.parquet",
            "--validation-split", "hard=hard_validation_pairs.parquet",
        ]
        training_started = time.perf_counter()
        run_logged(train_command, TRAIN_LOG)
        training_wall_seconds = time.perf_counter() - training_started
        """,
        "frozen",
        "training",
    )


def _completion_cell(
    *,
    experiment: str,
    role: str,
    notes: str,
    recipe_sha256: str,
) -> nbf.NotebookNode:
    return code(
        f"""
        from datetime import datetime, timezone

        raw_report_path = TRAINER_OUTPUT_DIR / "training_report.json"
        raw_config_path = TRAINER_OUTPUT_DIR / "training_config.json"
        if not raw_report_path.is_file() or not raw_config_path.is_file():
            raise RuntimeError("BGE trainer finished without report/config")
        report = json.loads(raw_report_path.read_text(encoding="utf-8"))
        if set(report.get("validation_splits", {{}})) != {{"iid", "hard"}}:
            raise RuntimeError("BGE trainer evaluated a split other than IID/hard")
        if (
            report.get("original_training_examples") != {EXPECTED_TRAIN}
            or report.get("training_sampling") != "none"
            or report.get("training_subset") != "all"
            or report.get("training_loss_weighting") != "none"
            or report.get("training_source_counts") != {{
                "human_train": {EXPECTED_HUMAN_TRAIN},
                "human_former_ood": {EXPECTED_FORMER_OOD},
            }}
        ):
            raise RuntimeError("BGE trainer report changed the frozen train policy")
        for split, expected_rows in (("iid", {EXPECTED_IID}), ("hard", {EXPECTED_HARD})):
            metrics = report["validation_splits"][split]
            if metrics.get("examples") != expected_rows:
                raise RuntimeError(f"Unexpected {{split}} validation rows")
            for metric_name in ("macro_average_precision", "overall_average_precision", "log_loss"):
                value = metrics.get(metric_name)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise RuntimeError(f"Non-finite {{split}} {{metric_name}}")

        ood_sentinel = {OOD_SENTINEL!r}
        report["validation_splits"]["ood"] = ood_sentinel
        report["experiment_group"] = "sft"
        report["ood_evaluation_policy"] = "disabled_train_contaminated"
        report["evaluated_validation_splits"] = ["iid", "hard"]

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for filename in (
            "training_config.json",
            "iid_validation_predictions.parquet",
            "hard_validation_predictions.parquet",
        ):
            source = TRAINER_OUTPUT_DIR / filename
            if not source.is_file() or source.stat().st_size == 0:
                raise RuntimeError(f"BGE trainer artifact is missing: {{filename}}")
            shutil.copy2(source, OUTPUT_DIR / filename)
        if list(TRAINER_OUTPUT_DIR.glob("ood*validation_predictions.parquet")):
            raise RuntimeError("BGE trainer unexpectedly emitted OOD predictions")
        (OUTPUT_DIR / "training_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\\n",
            encoding="utf-8",
        )

        if not RUNTIME_VERSIONS_PATH.is_file():
            raise RuntimeError("BGE runtime version report is missing")
        runtime_versions = json.loads(
            RUNTIME_VERSIONS_PATH.read_text(encoding="utf-8")
        )
        if runtime_versions != RUNTIME_VERSIONS:
            raise RuntimeError("BGE runtime version report changed after bootstrap")

        completion = {{
            "status": "complete",
            "run_id": EXPERIMENT_RUN_ID,
            "started_at_utc": EXPERIMENT_STARTED_AT_UTC,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ).replace("+00:00", "Z"),
            "experiment": {experiment!r},
            "experiment_group": "sft",
            "campaign": {CAMPAIGN!r},
            "role": {role!r},
            "notes": {notes!r},
            "model": EXPECTED_CHECKPOINT_REF,
            "dataset_ref": EXPECTED_VALIDATION_DATASET_REF,
            "validation_manifest_sha256": EXPECTED_VALIDATION_MANIFEST_SHA256,
            "initial_checkpoint_ref": EXPECTED_CHECKPOINT_REF,
            "initial_checkpoint_manifest_sha256": EXPECTED_CHECKPOINT_MANIFEST_SHA256,
            "initial_checkpoint_model_sha256": checkpoint_manifest["reconstruction"]["sha256"],
            "code_bundle_sha256": EXPECTED_SOURCE_SHA256,
            "frozen_recipe_sha256": {recipe_sha256!r},
            "campaign_identity_sha256": EXPECTED_CAMPAIGN_IDENTITY_SHA256,
            "executable_cells_sha256": EXPECTED_EXECUTABLE_CELLS_SHA256,
            "loss_variant": "bce_finite_guard_v1",
            "loss_hook_sha256": EXPECTED_LOSS_HOOK_SHA256,
            "ood_evaluation_policy": "disabled_train_contaminated",
            "train_data": TRAIN_DATA_REPORT,
            "memory_preflight": memory_preflight,
            "runtime_versions": runtime_versions,
            "training_wall_seconds": training_wall_seconds,
            "training_report": report,
            "kaggle_kernel_ref": (
                os.getenv("KAGGLE_KERNEL_RUN_ID")
                or os.getenv("KAGGLE_KERNEL_INFERENCE_RUN_ID")
                or ""
            ),
        }}
        completion_path = WORKING_ROOT / "notebook_completed.json"
        completion_path.write_text(
            json.dumps(completion, ensure_ascii=False, indent=2, default=str) + "\\n",
            encoding="utf-8",
        )

        shutil.rmtree(TRAINER_OUTPUT_DIR)
        if list(WORKING_ROOT.rglob("model.safetensors")):
            raise RuntimeError("Slim BGE output still contains model weights")
        print(json.dumps({{
            "experiment": completion["experiment"],
            "iid_macro_ap": report["validation_splits"]["iid"]["macro_average_precision"],
            "hard_macro_ap": report["validation_splits"]["hard"]["macro_average_precision"],
            "ood_macro_ap": -1.0,
            "trainer_weights_deleted": True,
        }}, ensure_ascii=False, indent=2))
        """,
        "frozen",
        "completion",
    )


def build_variant_notebook(
    *,
    validation: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    base_config: Mapping[str, Any],
    variant: Mapping[str, Any],
) -> tuple[nbf.NotebookNode, dict[str, Any]]:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", str(variant["experiment"])):
        raise CampaignConfigError("Invalid BGE experiment name")
    validate_base_config(base_config)
    config = resolve_variant_config(base_config, variant)
    sources, source_ledger, source_sha256 = source_bundle()
    recipe_sha256 = canonical_sha256(config)
    provisional_notes = variant_notes(
        variant=variant,
        config=config,
        identity_sha256=IDENTITY_PLACEHOLDER,
    )
    experiment = str(variant["experiment"])
    role = str(variant["role"])
    runtime_model_path = f"/kaggle/temp/{experiment}/initial_checkpoint"

    run_identity = shared.experiment_run_initialization_cell()
    run_identity.metadata["tags"] = ["frozen", "run-identity"]
    sheet_cells = shared.google_sheets_tracking_cells()
    for cell in sheet_cells:
        cell.metadata["tags"] = ["frozen", "sheets-sync"]

    cells = [
        markdown(
            f"""
            # BGE 2ep SFT: `{experiment}`

            This run starts from the exact user-supplied BGE two-epoch checkpoint.
            Human train and all 41,171 former OOD pairs are concatenated into a
            347,840-row SFT train.  IID and hard remain component-disjoint
            validation; OOD is intentionally not evaluated and is logged as -1.

            Campaign identity: `{IDENTITY_PLACEHOLDER}`.  Results route to `sft_exps`.
            """,
            "frozen",
            "campaign-description",
        ),
        _setup_cell(
            experiment=experiment,
            validation=validation,
            checkpoint=checkpoint,
            source_sha256=source_sha256,
            identity_sha256=IDENTITY_PLACEHOLDER,
            executable_sha256=EXECUTABLE_CELLS_PLACEHOLDER,
        ),
        run_identity,
        markdown("## Frozen recipe", "frozen"),
        _recipe_cell(config, recipe_sha256),
        markdown("## Embedded trainer", "frozen"),
        _sources_cell(sources, source_ledger, source_sha256),
        markdown("## Exact train concat and IID/hard isolation", "frozen"),
        _data_cell(),
        _loss_cell(),
        markdown("## Real two-T4 optimizer-step memory preflight", "frozen"),
        _preflight_cell(),
        markdown("## Full DDP fine-tuning", "frozen"),
        _training_cell(),
        markdown("## Slim completion with explicit OOD=-1", "frozen"),
        _completion_cell(
            experiment=experiment,
            role=role,
            notes=provisional_notes,
            recipe_sha256=recipe_sha256,
        ),
        *sheet_cells,
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    executable_sha256 = executable_cells_sha256(notebook)
    identity_sha256 = variant_identity(
        variant=variant,
        config=config,
        source_sha256=source_sha256,
        validation_manifest_sha256=str(validation["manifest_sha256"]),
        checkpoint_manifest_sha256=str(checkpoint["manifest_sha256"]),
        executable_cells_sha256=executable_sha256,
    )
    notes = variant_notes(
        variant=variant,
        config=config,
        identity_sha256=identity_sha256,
    )
    if provisional_notes.replace(IDENTITY_PLACEHOLDER, identity_sha256) != notes:
        raise CampaignConfigError("BGE notes identity substitution is not deterministic")
    _replace_notebook_placeholders(
        notebook,
        identity_sha256=identity_sha256,
        executable_sha256=executable_sha256,
    )
    _assign_deterministic_cell_ids(notebook)
    if executable_cells_sha256(
        notebook,
        identity_sha256=identity_sha256,
        expected_sha256=executable_sha256,
    ) != executable_sha256:
        raise CampaignConfigError("Final executable BGE notebook cells changed after freeze")
    slug = kernel_slug(variant, identity_sha256)
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "product_matching_training": {
                "template": "bge_2ep_sft_oodtrain_v1",
                "campaign": CAMPAIGN,
                "experiment": experiment,
                "role": role,
                "experiment_group": "sft",
                "validation_dataset": validation["dataset"],
                "validation_manifest_sha256": validation["manifest_sha256"],
                "initial_checkpoint": checkpoint["dataset"],
                "initial_checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
                "source_sha256": source_sha256,
                "frozen_recipe_sha256": recipe_sha256,
                "campaign_identity_sha256": identity_sha256,
                "executable_cells_sha256": executable_sha256,
                "loss_hook_sha256": FIXED_LOSS_HOOK_SHA256,
                "runtime_model_path": runtime_model_path,
                "train_pairs": EXPECTED_TRAIN,
                "validation_splits": ["iid", "hard"],
                "ood_metric_sentinel": -1,
                "expected_gpus": 2,
                "kernel_slug": slug,
                "editable_cells": [],
            },
        }
    )
    nbf.validate(notebook)
    entry = {
        "key": variant["key"],
        "experiment": experiment,
        "role": role,
        "default": bool(variant.get("default", False)),
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
        "checkpoint_model_sha256": checkpoint_push.EXPECTED_SOURCE_FILES[
            checkpoint_push.MODEL_FILENAME
        ]["sha256"],
        "expected_config": config,
        "expected_runtime_model_path": runtime_model_path,
        "expected_notes": notes,
        "loss_hook_sha256": FIXED_LOSS_HOOK_SHA256,
        "slug_token": variant["slug_token"],
    }
    validate_notebook_identity(notebook, entry=entry)
    return notebook, entry


def validate_notebook_identity(
    notebook: nbf.NotebookNode,
    *,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed when any executable cell or linked identity field drifts."""
    nbf.validate(notebook)
    metadata = notebook.metadata.get("product_matching_training")
    if not isinstance(metadata, Mapping):
        raise CampaignConfigError("BGE notebook has no training metadata")
    exact_metadata = {
        "template": "bge_2ep_sft_oodtrain_v1",
        "campaign": CAMPAIGN,
        "experiment": entry["experiment"],
        "role": entry["role"],
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
        "runtime_model_path": entry["expected_runtime_model_path"],
        "train_pairs": EXPECTED_TRAIN,
        "validation_splits": ["iid", "hard"],
        "ood_metric_sentinel": -1,
        "expected_gpus": 2,
        "kernel_slug": entry["kernel_slug"],
        "editable_cells": [],
    }
    if dict(metadata) != exact_metadata:
        raise CampaignConfigError("BGE notebook metadata differs from its frozen entry")
    _, current_source_sha256 = embedded_sources()
    if current_source_sha256 != entry["source_sha256"]:
        raise CampaignConfigError("BGE notebook entry uses a stale embedded source hash")
    if canonical_sha256(entry["expected_config"]) != entry["recipe_sha256"]:
        raise CampaignConfigError("BGE notebook entry recipe hash differs")
    actual_executable_sha256 = executable_cells_sha256(
        notebook,
        identity_sha256=str(entry["identity_sha256"]),
        expected_sha256=str(entry["executable_cells_sha256"]),
    )
    if actual_executable_sha256 != entry["executable_cells_sha256"]:
        raise CampaignConfigError("BGE executable notebook cell payload differs")
    expected_identity = variant_identity(
        variant=entry,
        config=entry["expected_config"],
        source_sha256=str(entry["source_sha256"]),
        validation_manifest_sha256=str(entry["validation_manifest_sha256"]),
        checkpoint_manifest_sha256=str(entry["checkpoint_manifest_sha256"]),
        executable_cells_sha256=actual_executable_sha256,
    )
    if expected_identity != entry["identity_sha256"]:
        raise CampaignConfigError("BGE campaign identity does not bind executable cells")
    if kernel_slug(entry, expected_identity) != entry["kernel_slug"]:
        raise CampaignConfigError("BGE kernel slug differs from campaign identity")
    return {
        "experiment": entry["experiment"],
        "identity_sha256": expected_identity,
        "executable_cells_sha256": actual_executable_sha256,
        "source_sha256": current_source_sha256,
    }


def load_and_validate_notebook(
    path: Path,
    *,
    entry: Mapping[str, Any],
) -> nbf.NotebookNode:
    notebook = nbf.read(path, as_version=4)
    validate_notebook_identity(notebook, entry=entry)
    return notebook


def output_path(output_dir: Path, entry: Mapping[str, Any]) -> Path:
    return output_dir / f"{entry['experiment']}_2xt4.ipynb"


def selected_variants(only: set[str] | None) -> Iterable[Mapping[str, Any]]:
    selected = [
        variant
        for variant in VARIANT_SPECS
        if only is None
        or variant["key"] in only
        or variant["experiment"] in only
    ]
    if only:
        matched = {
            token
            for token in only
            if any(token in {variant["key"], variant["experiment"]} for variant in selected)
        }
        if missing := only - matched:
            raise CampaignConfigError(f"Unknown BGE variants: {sorted(missing)}")
    if not selected:
        raise CampaignConfigError("No BGE variants selected")
    return selected


def build_campaign(
    *,
    owner: str,
    config_path: Path = DEFAULT_CONFIG,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    checkpoint_stage_dir: Path = DEFAULT_CHECKPOINT_STAGE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    only: set[str] | None = None,
    write: bool = True,
) -> list[dict[str, Any]]:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", owner):
        raise CampaignConfigError(f"Invalid Kaggle owner: {owner!r}")
    validation = load_validation_dataset(source_dir, owner)
    checkpoint = load_checkpoint_dataset(
        checkpoint_stage_dir,
        owner,
        verify_payload=True,
    )
    if file_sha256(config_path) != EXPECTED_BASE_CONFIG_FILE_SHA256:
        raise CampaignConfigError("Frozen BGE baseline config file SHA-256 has changed")
    base_config = cross_builder.load_training_config(config_path)
    validate_base_config(base_config)
    built: list[dict[str, Any]] = []
    for variant in selected_variants(only):
        notebook, entry = build_variant_notebook(
            validation=validation,
            checkpoint=checkpoint,
            base_config=base_config,
            variant=variant,
        )
        destination = output_path(output_dir, entry)
        if write:
            destination.parent.mkdir(parents=True, exist_ok=True)
            nbf.write(notebook, destination)
            load_and_validate_notebook(destination, entry=entry)
        entry["notebook"] = str(destination)
        built.append(entry)
    if len({entry["kernel_slug"] for entry in built}) != len(built):
        raise CampaignConfigError("BGE campaign generated duplicate Kaggle slugs")
    return built


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--owner")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument(
        "--checkpoint-stage-dir", type=Path, default=DEFAULT_CHECKPOINT_STAGE_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--only", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    owner = args.owner or shared.dotenv_username(args.env_file)
    if not owner:
        raise SystemExit("Set KAGGLE_USERNAME in .env or pass --owner")
    entries = build_campaign(
        owner=owner,
        config_path=args.config,
        source_dir=args.source_dir,
        checkpoint_stage_dir=args.checkpoint_stage_dir,
        output_dir=args.output_dir,
        only=set(args.only) or None,
    )
    print(json.dumps(entries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
