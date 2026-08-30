#!/usr/bin/env python3
"""Freeze one adaptive MiniLM-5ep SFT campaign transition into a stage lock.

The lock is the only mutable-to-immutable boundary in the sequential search.  A
completed source-stage decision is resolved once, tied to its downloaded
artifacts, and expanded into complete recipes for the next one-dimensional
line.  A requested boundary extension instead freezes the full completed family
and adds only its one predeclared outer point.  An existing lock is never
replaced or re-selected from a newer summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "configs" / "minilm_5ep_sft_hparam_search_v1.json"
DEFAULT_SUMMARY = (
    ROOT / "reports" / "minilm_5ep_sft_hparam_search_v1" / "summary.json"
)
DEFAULT_ARTIFACTS_DIR = ROOT / "artifacts" / "kaggle"
DEFAULT_LOCKS_DIR = (
    ROOT / "reports" / "minilm_5ep_sft_hparam_search_v1" / "stage_locks"
)
BASE_CONFIG_PATH = (
    ROOT / "configs" / "cross_encoder_minilm_llm_pretrain_5ep_human_ft.json"
)


class StageMaterializationError(RuntimeError):
    """Raised when a stage cannot be resolved without guessing."""


class ExistingLockConflictError(StageMaterializationError):
    """Raised when an immutable lock differs from the requested transition."""


def _optional_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def canonical_json_dumps(value: object) -> str:
    """Serialize finite JSON deterministically for hashing and lock files."""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise StageMaterializationError(
            f"Value is not canonical finite JSON: {error}"
        ) from error


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_dumps(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise StageMaterializationError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise StageMaterializationError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise StageMaterializationError(f"{label} must contain a JSON object: {path}")
    # Reject NaN/Infinity accepted by Python's permissive JSON parser.
    canonical_json_dumps(payload)
    return payload


def stage_by_name(plan: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    stages = plan.get("stages")
    if not isinstance(stages, list):
        raise StageMaterializationError("Campaign plan has no stages list")
    matches = [
        stage
        for stage in stages
        if isinstance(stage, Mapping) and stage.get("name") == name
    ]
    if len(matches) != 1:
        raise StageMaterializationError(
            f"Expected exactly one campaign stage {name!r}, found {len(matches)}"
        )
    return matches[0]


def plan_stage_for_effective_name(
    plan: Mapping[str, Any], name: str
) -> Mapping[str, Any]:
    """Resolve ``stage__coordinate`` summaries back to their plan stage."""
    try:
        return stage_by_name(plan, name)
    except StageMaterializationError:
        pass
    if "__" not in name:
        return stage_by_name(plan, name)
    base_name, coordinate = name.rsplit("__", 1)
    stage = stage_by_name(plan, base_name)
    axes = stage.get("axes")
    if not isinstance(axes, Mapping) or coordinate not in axes:
        raise StageMaterializationError(
            f"Effective stage {name!r} is not a declared coordinate"
        )
    return stage


def expected_source_stage(
    plan: Mapping[str, Any],
    *,
    target_stage: str,
    coordinate: str | None,
) -> str:
    """Return the only predecessor allowed by the declared staged search."""
    stages = plan.get("stages")
    if not isinstance(stages, list):
        raise StageMaterializationError("Campaign plan has no stages list")
    target = stage_by_name(plan, target_stage)
    target_positions = [
        index
        for index, stage in enumerate(stages)
        if isinstance(stage, Mapping) and stage.get("name") == target_stage
    ]
    if len(target_positions) != 1:
        raise StageMaterializationError(
            f"Could not locate one target stage {target_stage!r} in execution order"
        )

    axes = target.get("axes")
    order = target.get("execution_order")
    if isinstance(axes, Mapping) or isinstance(order, list):
        if not isinstance(axes, Mapping) or not isinstance(order, list):
            raise StageMaterializationError(
                f"Coordinate stage {target_stage!r} has an incomplete execution schema"
            )
        if coordinate not in order or coordinate not in axes:
            raise StageMaterializationError(
                f"Unknown or unordered coordinate {coordinate!r}"
            )
        coordinate_position = order.index(coordinate)
        if coordinate_position:
            return f"{target_stage}__{order[coordinate_position - 1]}"
        target_position = target_positions[0]
        if target_position == 0:
            raise StageMaterializationError(
                f"First campaign stage {target_stage!r} has no predecessor"
            )
        previous = stages[target_position - 1]
    else:
        if coordinate is not None:
            raise StageMaterializationError(
                "--coordinate is only valid for a multi-coordinate target stage"
            )
        target_position = target_positions[0]
        if target_position == 0:
            raise StageMaterializationError(
                f"First campaign stage {target_stage!r} has no predecessor"
            )
        previous = stages[target_position - 1]

    if not isinstance(previous, Mapping) or not str(previous.get("name", "")).strip():
        raise StageMaterializationError("Previous campaign stage is malformed")
    previous_name = str(previous["name"])
    previous_order = previous.get("execution_order")
    previous_axes = previous.get("axes")
    if isinstance(previous_order, list) or isinstance(previous_axes, Mapping):
        if (
            not isinstance(previous_order, list)
            or not previous_order
            or not isinstance(previous_axes, Mapping)
            or previous_order[-1] not in previous_axes
        ):
            raise StageMaterializationError(
                f"Previous coordinate stage {previous_name!r} has an invalid order"
            )
        return f"{previous_name}__{previous_order[-1]}"
    return previous_name


def validate_stage_transition(
    plan: Mapping[str, Any],
    *,
    source_stage: str,
    target_stage: str,
    coordinate: str | None,
) -> None:
    expected = expected_source_stage(
        plan,
        target_stage=target_stage,
        coordinate=coordinate,
    )
    if source_stage != expected:
        raise StageMaterializationError(
            f"Out-of-order transition {source_stage!r} -> "
            f"{target_stage!r}/{coordinate!r}; expected source {expected!r}"
        )


def source_snapshot(
    summary: Mapping[str, Any],
    *,
    campaign_name: str,
    source_stage: str,
    expected_parent: Mapping[str, Any] | None = None,
    allow_boundary_extension: bool = False,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    if summary.get("campaign") != campaign_name:
        raise StageMaterializationError("Stage summary belongs to a different campaign")
    stages = summary.get("stages")
    if not isinstance(stages, Mapping) or not isinstance(
        stages.get(source_stage), Mapping
    ):
        raise StageMaterializationError(
            f"Stage summary has no decision for {source_stage!r}"
        )
    decision = stages[source_stage]
    if decision.get("decision_status") != "ready" or decision.get("complete") is not True:
        raise StageMaterializationError(
            f"Source stage {source_stage!r} is not complete and decision-ready"
        )
    if decision.get("control_gate") not in (None, "passed"):
        raise StageMaterializationError(
            f"Source stage {source_stage!r} did not pass its control gate"
        )
    if (
        decision.get("needs_boundary_extension") is True
        and not allow_boundary_extension
    ):
        raise StageMaterializationError(
            f"Source stage {source_stage!r} still requires a boundary extension"
        )
    experiment = str(decision.get("recommended_experiment", "")).strip()
    run_id = str(decision.get("recommended_run_id", "")).strip()
    if not experiment or not run_id:
        raise StageMaterializationError(
            f"Source stage {source_stage!r} has no immutable recommendation"
        )
    if expected_parent is not None and (
        experiment != expected_parent.get("experiment")
        or run_id != expected_parent.get("run_id")
    ):
        raise ExistingLockConflictError(
            "The completed stage summary now points at a different parent than "
            "the immutable lock"
        )

    runs = summary.get("runs")
    if not isinstance(runs, list):
        raise StageMaterializationError("Stage summary has no runs list")
    matches = [
        row
        for row in runs
        if isinstance(row, Mapping)
        and row.get("experiment") == experiment
        and row.get("run_id") == run_id
    ]
    if len(matches) != 1:
        raise StageMaterializationError(
            "Recommended parent must match exactly one row in the stage summary"
        )
    row = matches[0]
    if row.get("completed") is not True or row.get("status") != "complete":
        raise StageMaterializationError("Recommended parent run is not complete")
    if row.get("stage") != source_stage:
        raise StageMaterializationError("Recommended parent row belongs to another stage")
    snapshot = {
        "campaign": campaign_name,
        "source_stage": source_stage,
        "decision": dict(decision),
        "recommended_run": dict(row),
    }
    return snapshot, row


def load_source_summary(path: Path, *, source_stage: str) -> dict[str, Any]:
    """Use root latest when possible, otherwise its preserved stage snapshot."""
    summary = load_json_object(path, label="stage summary")
    stages = summary.get("stages")
    if isinstance(stages, Mapping) and isinstance(
        stages.get(source_stage), Mapping
    ):
        return summary
    preserved_path = path.parent / "stages" / source_stage / "summary.json"
    if preserved_path.is_file():
        return load_json_object(
            preserved_path,
            label=f"preserved {source_stage} stage summary",
        )
    return summary


def _artifact_root(
    artifacts_dir: Path,
    row: Mapping[str, Any],
) -> Path:
    kernel_slug = _optional_text(row.get("kernel_slug"))
    if kernel_slug:
        root = artifacts_dir / kernel_slug
        if (root / "notebook_completed.json").is_file():
            return root

    completion_value = _optional_text(row.get("completion_path"))
    if completion_value:
        completion_path = Path(completion_value)
        if not completion_path.is_absolute():
            completion_path = ROOT / completion_path
        try:
            completion_path.resolve().relative_to(artifacts_dir.resolve())
        except ValueError as error:
            raise StageMaterializationError(
                "Summary completion_path escapes the selected artifacts directory"
            ) from error
        if completion_path.name == "notebook_completed.json" and completion_path.is_file():
            return completion_path.parent

    candidates: list[Path] = []
    for path in artifacts_dir.glob("*/notebook_completed.json"):
        try:
            completion = load_json_object(path, label="completion artifact")
        except StageMaterializationError:
            continue
        if (
            completion.get("experiment") == row.get("experiment")
            and completion.get("run_id") == row.get("run_id")
        ):
            candidates.append(path.parent)
    if len(candidates) != 1:
        raise StageMaterializationError(
            "Could not resolve exactly one downloaded artifact directory for the parent"
        )
    return candidates[0]


def _exactly_one(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if len(matches) != 1:
        raise StageMaterializationError(
            f"Expected exactly one {filename!r} below {root}, found {len(matches)}"
        )
    return matches[0]


def _planned_parent_config(
    plan: Mapping[str, Any],
    *,
    source_stage: str,
    experiment: str,
) -> dict[str, Any] | None:
    stage = plan_stage_for_effective_name(plan, source_stage)
    variants = stage.get("variants")
    if not isinstance(variants, list):
        return None
    matches = [
        variant
        for variant in variants
        if isinstance(variant, Mapping) and variant.get("experiment") == experiment
    ]
    if len(matches) != 1:
        return None
    base = load_json_object(BASE_CONFIG_PATH, label="base SFT config")
    overrides = matches[0].get("overrides")
    if not isinstance(overrides, Mapping):
        raise StageMaterializationError("Parent variant has no overrides object")
    resolved = deepcopy(base)
    resolved.update(deepcopy(dict(overrides)))
    return resolved


def resolve_parent(
    plan: Mapping[str, Any],
    *,
    source_stage: str,
    row: Mapping[str, Any],
    artifacts_dir: Path,
) -> dict[str, Any]:
    artifact_root = _artifact_root(artifacts_dir, row)
    completion_path = artifact_root / "notebook_completed.json"
    completion = load_json_object(completion_path, label="completion artifact")
    experiment = str(row["experiment"])
    run_id = str(row["run_id"])
    if (
        completion.get("status") != "complete"
        or completion.get("experiment") != experiment
        or completion.get("run_id") != run_id
    ):
        raise StageMaterializationError(
            "Downloaded completion artifact does not match the selected parent"
        )
    recipe_sha256 = str(completion.get("frozen_recipe_sha256", "")).strip()
    if not re.fullmatch(r"[0-9a-f]{64}", recipe_sha256):
        raise StageMaterializationError("Parent completion has no valid recipe SHA-256")
    row_recipe = _optional_text(row.get("recipe_sha256"))
    if row_recipe and row_recipe != recipe_sha256:
        raise StageMaterializationError("Summary and completion recipe hashes differ")

    training_config_path = _exactly_one(artifact_root, "training_config.json")
    artifact_config = load_json_object(training_config_path, label="training config")
    resolved_config = _planned_parent_config(
        plan,
        source_stage=source_stage,
        experiment=experiment,
    )
    if resolved_config is None:
        resolved_config = deepcopy(artifact_config)
        base_config = load_json_object(BASE_CONFIG_PATH, label="base SFT config")
        if "model" in base_config:
            resolved_config["model"] = base_config["model"]
    if set(artifact_config) != set(resolved_config):
        raise StageMaterializationError(
            "Parent artifact training config keys differ from the resolved recipe"
        )
    for key, expected in resolved_config.items():
        if key != "model" and artifact_config.get(key) != expected:
            raise StageMaterializationError(
                f"Parent artifact config differs at {key!r}"
            )
    if canonical_sha256(resolved_config) != recipe_sha256:
        raise StageMaterializationError(
            "Resolved parent config does not reproduce the frozen recipe SHA-256"
        )

    iid_predictions_path = _exactly_one(
        artifact_root, "iid_validation_predictions.parquet"
    )
    try:
        iid_relative_path = str(iid_predictions_path.relative_to(artifact_root))
    except ValueError as error:  # pragma: no cover - rglob guarantees containment.
        raise StageMaterializationError("IID predictions escape artifact root") from error

    iid_predictions_sha256 = file_sha256(iid_predictions_path)
    row_iid_sha256 = _optional_text(row.get("iid_predictions_sha256"))
    if row_iid_sha256 and row_iid_sha256 != iid_predictions_sha256:
        raise StageMaterializationError(
            "Summary and parent IID prediction hashes differ"
        )

    kernel_slug = _optional_text(row.get("kernel_slug")) or artifact_root.name
    result = {
        "experiment": experiment,
        "run_id": run_id,
        "kernel_slug": kernel_slug,
        "recipe_sha256": recipe_sha256,
        "iid_predictions_sha256": iid_predictions_sha256,
        "iid_predictions_relative_path": iid_relative_path,
        "completion_sha256": file_sha256(completion_path),
        "training_config_artifact_sha256": file_sha256(training_config_path),
        "resolved_config": resolved_config,
    }
    code_bundle_sha256 = _optional_text(completion.get("code_bundle_sha256"))
    if code_bundle_sha256:
        if not re.fullmatch(r"[0-9a-f]{64}", code_bundle_sha256):
            raise StageMaterializationError(
                "Parent completion has an invalid code-bundle SHA-256"
            )
        result["code_bundle_sha256"] = code_bundle_sha256
    loss_hook_sha256 = _optional_text(completion.get("loss_hook_sha256"))
    if loss_hook_sha256:
        if not re.fullmatch(r"[0-9a-f]{64}", loss_hook_sha256):
            raise StageMaterializationError(
                "Parent completion has an invalid loss-hook SHA-256"
            )
        result["loss_hook_sha256"] = loss_hook_sha256
    loss_variant = _optional_text(row.get("loss_variant"))
    if loss_variant:
        result["loss_variant"] = loss_variant
    notes = completion.get("notes")
    if isinstance(notes, str):
        result["notes"] = notes
    return result


def _axis_definition(
    target_stage: Mapping[str, Any],
    *,
    coordinate: str | None,
) -> tuple[str, list[Any], Mapping[str, Mapping[str, Any]]]:
    direct_axis = target_stage.get("axis")
    if isinstance(direct_axis, Mapping):
        if coordinate is not None:
            raise StageMaterializationError(
                "--coordinate is only valid for a multi-coordinate target stage"
            )
        if len(direct_axis) != 1:
            raise StageMaterializationError("A direct stage axis must have one key")
        axis_name, raw_levels = next(iter(direct_axis.items()))
    else:
        axes = target_stage.get("axes")
        order = target_stage.get("execution_order")
        if not isinstance(axes, Mapping) or not isinstance(order, list):
            raise StageMaterializationError(
                "Target stage has neither a direct axis nor coordinate axes"
            )
        if coordinate is None:
            raise StageMaterializationError(
                "A multi-coordinate target stage requires --coordinate"
            )
        if coordinate not in order or coordinate not in axes:
            raise StageMaterializationError(
                f"Unknown or unordered coordinate {coordinate!r}"
            )
        axis_name = coordinate
        raw_levels = axes[coordinate]
    if not isinstance(raw_levels, list) or not raw_levels:
        raise StageMaterializationError(f"Axis {axis_name!r} has no levels")
    if len({canonical_json_dumps(level) for level in raw_levels}) != len(raw_levels):
        raise StageMaterializationError(f"Axis {axis_name!r} contains duplicate levels")
    recipes = target_stage.get("effective_batch_recipes", {})
    return str(axis_name), list(raw_levels), recipes if isinstance(recipes, Mapping) else {}


def _declared_extension_levels(
    target_stage: Mapping[str, Any],
    *,
    axis_name: str,
) -> list[Any]:
    direct = target_stage.get("conditional_extension")
    if isinstance(direct, Mapping):
        values = direct.get(axis_name, [])
        if not isinstance(values, list):
            raise StageMaterializationError(
                f"Conditional extension for {axis_name!r} must be a list"
            )
        return deepcopy(values)
    boundary = target_stage.get("boundary_extension")
    if isinstance(boundary, Mapping):
        values = [
            boundary[key]
            for key in ("lower", "upper")
            if key in boundary
        ]
        return deepcopy(values)
    coordinates = target_stage.get("conditional_extensions")
    if isinstance(coordinates, Mapping) and axis_name in coordinates:
        value = coordinates[axis_name]
        return deepcopy(value if isinstance(value, list) else [value])
    return []


def _level_overrides(
    axis_name: str,
    level: Any,
    *,
    effective_batch_recipes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if axis_name == "effective_batch":
        overrides = effective_batch_recipes.get(str(level))
        if not isinstance(overrides, Mapping):
            raise StageMaterializationError(
                f"No effective-batch recipe for level {level!r}"
            )
        return deepcopy(dict(overrides))
    if axis_name == "classifier_dropout":
        return {"model_load_kwargs": {"classifier_dropout": level}}
    return {axis_name: level}


def _effective_axis_value(config: Mapping[str, Any], axis_name: str) -> Any:
    if axis_name == "effective_batch":
        return int(config["batch_size"]) * 2 * int(config["gradient_accumulation"])
    if axis_name == "classifier_dropout":
        kwargs = config.get("model_load_kwargs")
        if isinstance(kwargs, Mapping) and kwargs.get("classifier_dropout") is not None:
            return kwargs["classifier_dropout"]
        # Frozen XLM-R checkpoint has classifier_dropout=null and
        # hidden_dropout_prob=0.1, so the classification head uses 0.1.
        return 0.1
    return config.get(axis_name)


def _number_token(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StageMaterializationError(f"Axis level must be numeric, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise StageMaterializationError("Axis levels must be finite")
    if number == 0:
        return "0"
    if number.is_integer():
        return str(int(number))
    scientific = f"{number:.12g}".lower()
    if "e" in scientific:
        mantissa, exponent = scientific.split("e", 1)
        exponent = str(abs(int(exponent)))
        sign = "m" if "e-" in scientific else "p"
        return mantissa.replace(".", "p") + sign + exponent
    return scientific.replace(".", "p")


def _learning_rate_token(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StageMaterializationError("Learning rate must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise StageMaterializationError("Learning rate must be finite and positive")
    scientific = f"{number:.12g}".lower()
    if "e" not in scientific:
        return scientific.replace(".", "p")
    mantissa, exponent = scientific.split("e", 1)
    exponent_value = int(exponent)
    exponent_token = (
        f"e{abs(exponent_value)}"
        if exponent_value < 0
        else f"ep{exponent_value}"
    )
    return mantissa.replace(".", "p") + exponent_token


def _variant_identity(
    *,
    axis_name: str,
    level: Any,
    config: Mapping[str, Any],
) -> tuple[str, str, str]:
    recipe_sha = canonical_sha256(config)
    if axis_name == "epochs":
        lr_token = _learning_rate_token(config["learning_rate"])
        stem = f"minilm5_sft_e{int(level)}_lr{lr_token}_v1"
    else:
        axis_token = {
            "weight_decay": "wd",
            "warmup_ratio": "wu",
            "label_smoothing": "ls",
            "classifier_dropout": "drop",
            "max_grad_norm": "clip",
            "effective_batch": "eb",
        }.get(axis_name, re.sub(r"[^a-z0-9]+", "", axis_name.lower()))
        stem = (
            f"minilm5_sft_{axis_token}{_number_token(level)}_"
            f"{recipe_sha[:8]}_v1"
        )
    if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", stem):
        raise StageMaterializationError(f"Could not form a safe experiment name: {stem}")
    slug = "pm-" + stem.replace("_", "-")
    return stem, slug, recipe_sha


def _overrides_from_base(config: Mapping[str, Any]) -> dict[str, Any]:
    base = load_json_object(BASE_CONFIG_PATH, label="base SFT config")
    removed = set(base) - set(config)
    if removed:
        raise StageMaterializationError(
            f"Resolved config removed base keys: {sorted(removed)}"
        )
    return {
        key: deepcopy(value)
        for key, value in config.items()
        if key not in base or base.get(key) != value
    }


def resolve_target_stage(
    target_stage: Mapping[str, Any],
    *,
    parent: Mapping[str, Any],
    coordinate: str | None,
    allowed_override_keys: set[str] | None = None,
) -> dict[str, Any]:
    axis_name, levels, effective_batch_recipes = _axis_definition(
        target_stage,
        coordinate=coordinate,
    )
    conditional_extension_levels = _declared_extension_levels(
        target_stage,
        axis_name=axis_name,
    )
    parent_config = parent.get("resolved_config")
    if not isinstance(parent_config, Mapping):
        raise StageMaterializationError("Parent lock data has no resolved config")
    parent_level = _effective_axis_value(parent_config, axis_name)
    matching_parent_levels = [
        level
        for level in levels
        if canonical_json_dumps(level) == canonical_json_dumps(parent_level)
    ]
    if len(matching_parent_levels) != 1:
        raise StageMaterializationError(
            f"Parent value {parent_level!r} is not exactly one level on axis {axis_name!r}"
        )

    variants: list[dict[str, Any]] = []
    for level in levels:
        if canonical_json_dumps(level) == canonical_json_dumps(parent_level):
            continue
        coordinate_overrides = _level_overrides(
            axis_name,
            level,
            effective_batch_recipes=effective_batch_recipes,
        )
        resolved_config = deepcopy(dict(parent_config))
        resolved_config.update(deepcopy(coordinate_overrides))
        experiment, kernel_slug, recipe_sha = _variant_identity(
            axis_name=axis_name,
            level=level,
            config=resolved_config,
        )
        overrides = _overrides_from_base(resolved_config)
        if allowed_override_keys is not None:
            forbidden = set(overrides) - allowed_override_keys
            if forbidden:
                raise StageMaterializationError(
                    "Resolved stage changes forbidden config keys: "
                    f"{sorted(forbidden)}"
                )
        variants.append(
            {
                "experiment": experiment,
                "kernel_slug": kernel_slug,
                "title": kernel_slug,
                "role": "candidate",
                "axis": axis_name,
                "level": level,
                "coordinate_overrides": coordinate_overrides,
                "overrides": overrides,
                "resolved_config": resolved_config,
                "expected_recipe_sha256": recipe_sha,
            }
        )

    declared_family = target_stage.get("family")
    if isinstance(declared_family, Mapping):
        family = deepcopy(dict(declared_family))
        if (
            family.get("correction") != "holm"
            or family.get("planned_candidate_hypotheses") != len(variants)
            or isinstance(family.get("reserved_conditional_extensions"), bool)
            or not isinstance(family.get("reserved_conditional_extensions"), int)
            or family["reserved_conditional_extensions"] < 0
            or family.get("maximum_hypotheses")
            != len(variants) + family["reserved_conditional_extensions"]
        ):
            raise StageMaterializationError(
                "Declared direct-stage hypothesis family is inconsistent"
            )
        if isinstance(target_stage.get("conditional_extension"), Mapping):
            declared_extensions = len(conditional_extension_levels)
        else:
            boundary_extension = target_stage.get("boundary_extension", {})
            declared_extensions = (
                boundary_extension.get("max_new_runs", 0)
                if isinstance(boundary_extension, Mapping)
                else -1
            )
        if declared_extensions != family["reserved_conditional_extensions"]:
            raise StageMaterializationError(
                "Declared direct-stage family differs from conditional_extension"
            )
    else:
        coordinate_maxima = target_stage.get("coordinate_maximum_hypotheses")
        if not isinstance(coordinate_maxima, Mapping) or axis_name not in coordinate_maxima:
            raise StageMaterializationError(
                f"No maximum hypothesis family declared for coordinate {axis_name!r}"
            )
        maximum_hypotheses = coordinate_maxima[axis_name]
        if (
            isinstance(maximum_hypotheses, bool)
            or not isinstance(maximum_hypotheses, int)
            or maximum_hypotheses < len(variants)
        ):
            raise StageMaterializationError(
                f"Invalid maximum hypothesis family for coordinate {axis_name!r}"
            )
        reserved_extensions = maximum_hypotheses - len(variants)
        conditional_extensions = target_stage.get("conditional_extensions", {})
        if not isinstance(conditional_extensions, Mapping):
            raise StageMaterializationError("conditional_extensions must be an object")
        expected_reserved = len(conditional_extension_levels)
        if reserved_extensions != expected_reserved:
            raise StageMaterializationError(
                f"Coordinate {axis_name!r} family does not match its conditional extension"
            )
        family = {
            "correction": "holm",
            "planned_candidate_hypotheses": len(variants),
            "reserved_conditional_extensions": reserved_extensions,
            "maximum_hypotheses": maximum_hypotheses,
        }
    return {
        "strategy": target_stage.get("strategy"),
        "axis": axis_name,
        "levels": levels,
        "conditional_extension_levels": conditional_extension_levels,
        "parent_level": parent_level,
        "reused_parent": {
            "experiment": parent["experiment"],
            "run_id": parent["run_id"],
            "level": parent_level,
        },
        "family": family,
        "variants": variants,
    }


def _stage_rows_for_extension(
    summary: Mapping[str, Any],
    *,
    campaign_name: str,
    source_stage: str,
) -> tuple[dict[str, Any], list[Mapping[str, Any]], Mapping[str, Any]]:
    _, recommended = source_snapshot(
        summary,
        campaign_name=campaign_name,
        source_stage=source_stage,
        allow_boundary_extension=True,
    )
    decision = summary["stages"][source_stage]
    if decision.get("needs_boundary_extension") is not True:
        raise StageMaterializationError(
            f"Source stage {source_stage!r} does not request a boundary extension"
        )
    runs = summary.get("runs")
    if not isinstance(runs, list):
        raise StageMaterializationError("Stage summary has no runs list")
    rows = [
        row
        for row in runs
        if isinstance(row, Mapping) and row.get("stage") == source_stage
    ]
    if not rows or any(
        row.get("completed") is not True or row.get("status") != "complete"
        for row in rows
    ):
        raise StageMaterializationError(
            "A conditional extension requires every base-stage entry to be complete"
        )
    snapshot = {
        "campaign": campaign_name,
        "source_stage": source_stage,
        "decision": dict(decision),
        "runs": [dict(row) for row in rows],
    }
    return snapshot, rows, recommended


def _row_is_hypothesis(row: Mapping[str, Any]) -> bool:
    if isinstance(row.get("is_hypothesis"), bool):
        return bool(row["is_hypothesis"])
    return row.get("role") not in {
        "current_protocol_control",
        "stage_anchor",
    }


def _extension_level(
    target_stage: Mapping[str, Any],
    *,
    axis_name: str,
    base_levels: list[Any],
    edge_config: Mapping[str, Any],
    recommendation: Mapping[str, Any],
) -> tuple[Any, str]:
    if recommendation.get("axis") != axis_name:
        raise StageMaterializationError(
            "Stage recommendation axis differs from the materialized axis"
        )
    direction = str(recommendation.get("direction", "")).strip()
    if direction not in {"lower", "higher"}:
        raise StageMaterializationError(
            "Boundary extension recommendation needs lower/higher direction"
        )
    if any(
        isinstance(level, bool)
        or not isinstance(level, (int, float))
        or not math.isfinite(float(level))
        for level in base_levels
    ):
        raise StageMaterializationError("Boundary-extension axes must be numeric")
    numeric_levels = [float(level) for level in base_levels]
    edge_level = _effective_axis_value(edge_config, axis_name)
    if isinstance(edge_level, bool) or not isinstance(edge_level, (int, float)):
        raise StageMaterializationError("Selected edge has no numeric axis value")
    expected_edge = min(numeric_levels) if direction == "lower" else max(numeric_levels)
    if not math.isclose(float(edge_level), expected_edge, rel_tol=0.0, abs_tol=0.0):
        raise StageMaterializationError(
            f"Selected {axis_name!r} value {edge_level!r} is not the {direction} edge"
        )

    declared = _declared_extension_levels(target_stage, axis_name=axis_name)
    if any(
        isinstance(level, bool)
        or not isinstance(level, (int, float))
        or not math.isfinite(float(level))
        for level in declared
    ):
        raise StageMaterializationError(
            "Declared boundary-extension levels must be finite numbers"
        )
    explicit_level = recommendation.get("level")
    if explicit_level is not None:
        declared = [
            level
            for level in declared
            if canonical_json_dumps(level) == canonical_json_dumps(explicit_level)
        ]
    if direction == "lower":
        candidates = [level for level in declared if float(level) < min(numeric_levels)]
    else:
        candidates = [level for level in declared if float(level) > max(numeric_levels)]
    if len(candidates) != 1:
        raise StageMaterializationError(
            f"Expected exactly one declared {direction} extension for {axis_name!r}, "
            f"found {candidates!r}"
        )
    return candidates[0], direction


def _prior_extension_entry(
    *,
    row: Mapping[str, Any],
    artifact: Mapping[str, Any],
    source_stage: str,
    axis_name: str,
    conditional_extension_levels: list[Any],
    family_size: int,
) -> dict[str, Any]:
    code_bundle_sha256 = _optional_text(artifact.get("code_bundle_sha256"))
    loss_hook_sha256 = _optional_text(artifact.get("loss_hook_sha256"))
    loss_variant = _optional_text(artifact.get("loss_variant"))
    notes = artifact.get("notes")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", code_bundle_sha256)
        or not re.fullmatch(r"[0-9a-f]{64}", loss_hook_sha256)
        or not loss_variant
        or not isinstance(notes, str)
    ):
        raise StageMaterializationError(
            "Conditional extension requires frozen source/loss/notes provenance "
            f"for prior run {artifact.get('experiment')!r}"
        )
    config = artifact["resolved_config"]
    return {
        "stage": source_stage,
        "experiment": str(artifact["experiment"]),
        "kernel_slug": str(artifact["kernel_slug"]),
        "role": str(row.get("role", "candidate")),
        "planned_overrides": _overrides_from_base(config),
        "expected_config": deepcopy(dict(config)),
        "expected_recipe_sha256": str(artifact["recipe_sha256"]),
        "expected_source_sha256": code_bundle_sha256,
        "loss_variant": loss_variant,
        "expected_loss_hook_sha256": loss_hook_sha256,
        "expected_run_id": str(artifact["run_id"]),
        "expected_iid_predictions_sha256": str(
            artifact["iid_predictions_sha256"]
        ),
        "expected_completion_sha256": str(artifact["completion_sha256"]),
        "expected_notes": notes,
        "provenance_alias": None,
        "is_hypothesis": _row_is_hypothesis(row),
        "hypothesis_family_size": family_size,
        "axis": axis_name,
        "level": _effective_axis_value(config, axis_name),
        "conditional_extension_levels": deepcopy(conditional_extension_levels),
    }


def _validate_existing_extension_identity(
    lock: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    summary: Mapping[str, Any],
    source_stage: str,
) -> None:
    target = plan_stage_for_effective_name(plan, source_stage)
    coordinate = (
        source_stage.rsplit("__", 1)[1] if "__" in source_stage else None
    )
    expected = {
        "campaign": plan.get("campaign"),
        "source_stage": source_stage,
        "target_stage": target.get("name"),
        "coordinate": coordinate,
        "effective_stage": source_stage,
        "source_plan_sha256": canonical_sha256(plan),
        "transition_kind": "conditional_boundary_extension",
    }
    for key, value in expected.items():
        if lock.get(key) != value:
            raise ExistingLockConflictError(
                f"Existing extension lock has {key}={lock.get(key)!r}, "
                f"expected {value!r}"
            )
    stages = summary.get("stages")
    decision = stages.get(source_stage) if isinstance(stages, Mapping) else None
    if (
        isinstance(decision, Mapping)
        and decision.get("complete") is True
        and decision.get("decision_status") == "ready"
        and decision.get("needs_boundary_extension") is not True
    ):
        summary_lock = summary.get("stage_lock")
        if (
            not isinstance(summary_lock, Mapping)
            or summary_lock.get("lock_payload_sha256")
            != lock.get("lock_payload_sha256")
            or summary_lock.get("effective_stage") != source_stage
        ):
            raise ExistingLockConflictError(
                "Completed extension summary does not point at this immutable lock"
            )
        return
    snapshot, _, recommended = _stage_rows_for_extension(
        summary,
        campaign_name=str(plan.get("campaign", "")),
        source_stage=source_stage,
    )
    extension_source = lock.get("extension_source")
    if not isinstance(extension_source, Mapping) or (
        extension_source.get("experiment") != recommended.get("experiment")
        or extension_source.get("run_id") != recommended.get("run_id")
    ):
        raise ExistingLockConflictError(
            "Existing extension lock points at another selected edge run"
        )
    if lock.get("source_stage_snapshot_sha256") != canonical_sha256(snapshot):
        raise ExistingLockConflictError(
            "Existing extension lock differs from the completed stage snapshot"
        )


def materialize_boundary_extension_lock(
    *,
    plan_path: Path = DEFAULT_PLAN,
    summary_path: Path = DEFAULT_SUMMARY,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    source_stage: str = "lr_log_line",
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Freeze the one predeclared outer point requested by a stage decision."""
    plan = load_json_object(plan_path, label="campaign plan")
    if plan.get("schema_version") != 1 or not str(plan.get("campaign", "")).strip():
        raise StageMaterializationError("Unsupported campaign plan schema")
    target = plan_stage_for_effective_name(plan, source_stage)
    target_name = str(target["name"])
    coordinate = (
        source_stage.rsplit("__", 1)[1] if "__" in source_stage else None
    )
    axis_name, base_levels, effective_batch_recipes = _axis_definition(
        target,
        coordinate=coordinate,
    )
    declared_extension_levels = _declared_extension_levels(
        target,
        axis_name=axis_name,
    )
    if not declared_extension_levels:
        raise StageMaterializationError(
            f"Stage {source_stage!r} has no declared conditional extension"
        )
    if output_path is None:
        output_path = DEFAULT_LOCKS_DIR / f"{source_stage}_boundary.lock.json"
    summary = load_source_summary(summary_path, source_stage=source_stage)
    if output_path.exists():
        existing = read_existing_lock(output_path)
        _validate_existing_extension_identity(
            existing,
            plan=plan,
            summary=summary,
            source_stage=source_stage,
        )
        return existing

    snapshot, rows, recommended_row = _stage_rows_for_extension(
        summary,
        campaign_name=str(plan["campaign"]),
        source_stage=source_stage,
    )
    anchors = [
        row
        for row in rows
        if row.get("role") in {"current_protocol_control", "stage_anchor"}
    ]
    if len(anchors) != 1:
        raise StageMaterializationError(
            "Conditional extension requires exactly one frozen selection anchor"
        )
    hypotheses = [row for row in rows if _row_is_hypothesis(row)]
    declared_sizes = {
        int(row["hypothesis_family_size"])
        for row in rows
        if row.get("hypothesis_family_size") is not None
    }
    if len(declared_sizes) != 1:
        raise StageMaterializationError(
            "Base-stage rows have no single predeclared hypothesis family"
        )
    maximum_hypotheses = declared_sizes.pop()
    if maximum_hypotheses - len(hypotheses) != 1:
        raise StageMaterializationError(
            "Conditional extension must consume exactly one reserved hypothesis slot"
        )

    resolved_rows = [
        (
            row,
            resolve_parent(
                plan,
                source_stage=source_stage,
                row=row,
                artifacts_dir=artifacts_dir,
            ),
        )
        for row in rows
    ]
    by_identity = {
        (artifact["experiment"], artifact["run_id"]): (row, artifact)
        for row, artifact in resolved_rows
    }
    anchor_artifact = by_identity[
        (str(anchors[0]["experiment"]), str(anchors[0]["run_id"]))
    ][1]
    edge_artifact = by_identity[
        (str(recommended_row["experiment"]), str(recommended_row["run_id"]))
    ][1]
    recommendation = summary["stages"][source_stage].get(
        "recommended_extension"
    )
    if not isinstance(recommendation, Mapping):
        raise StageMaterializationError(
            "Stage requests an extension but has no structured recommendation"
        )
    extension_level, direction = _extension_level(
        target,
        axis_name=axis_name,
        base_levels=base_levels,
        edge_config=edge_artifact["resolved_config"],
        recommendation=recommendation,
    )
    coordinate_overrides = _level_overrides(
        axis_name,
        extension_level,
        effective_batch_recipes=effective_batch_recipes,
    )
    resolved_config = deepcopy(dict(edge_artifact["resolved_config"]))
    resolved_config.update(deepcopy(coordinate_overrides))
    overrides = _overrides_from_base(resolved_config)
    forbidden = set(overrides) - set(plan.get("allowed_override_keys", []))
    if forbidden:
        raise StageMaterializationError(
            f"Conditional extension changes forbidden keys: {sorted(forbidden)}"
        )
    experiment, kernel_slug, recipe_sha = _variant_identity(
        axis_name=axis_name,
        level=extension_level,
        config=resolved_config,
    )
    if experiment in {str(row["experiment"]) for row in rows}:
        raise StageMaterializationError(
            f"Conditional extension experiment {experiment!r} is not unique"
        )
    loss_variant = _optional_text(edge_artifact.get("loss_variant")) or "bce"
    variant = {
        "experiment": experiment,
        "kernel_slug": kernel_slug,
        "title": kernel_slug,
        "role": "candidate",
        "axis": axis_name,
        "level": extension_level,
        "conditional_extension": True,
        "coordinate_overrides": coordinate_overrides,
        "overrides": overrides,
        "resolved_config": resolved_config,
        "expected_recipe_sha256": recipe_sha,
        "loss_variant": loss_variant,
    }
    final_family = {
        "correction": "holm",
        "planned_candidate_hypotheses": len(hypotheses) + 1,
        "reserved_conditional_extensions": 0,
        "maximum_hypotheses": maximum_hypotheses,
    }
    prior_entries = [
        _prior_extension_entry(
            row=row,
            artifact=artifact,
            source_stage=source_stage,
            axis_name=axis_name,
            conditional_extension_levels=declared_extension_levels,
            family_size=maximum_hypotheses,
        )
        for row, artifact in resolved_rows
    ]
    payload = _with_payload_hash(
        {
            "schema_version": 1,
            "kind": "minilm_5ep_sft_stage_lock",
            "transition_kind": "conditional_boundary_extension",
            "campaign": plan["campaign"],
            "source_stage": source_stage,
            "target_stage": target_name,
            "coordinate": coordinate,
            "effective_stage": source_stage,
            "source_plan_sha256": canonical_sha256(plan),
            "source_stage_snapshot_sha256": canonical_sha256(snapshot),
            "selection_metric": plan.get("selection_protocol", {}).get(
                "primary_metric"
            ),
            "parent": anchor_artifact,
            "extension_source": edge_artifact,
            "prior_entries": prior_entries,
            "resolved_stage": {
                "strategy": "conditional_boundary_extension",
                "axis": axis_name,
                "levels": [*base_levels, extension_level],
                "conditional_extension_levels": [],
                "conditional_extension_consumed": True,
                "extension_direction": direction,
                "parent_level": _effective_axis_value(
                    anchor_artifact["resolved_config"], axis_name
                ),
                "reused_parent": {
                    "experiment": anchor_artifact["experiment"],
                    "run_id": anchor_artifact["run_id"],
                    "level": _effective_axis_value(
                        anchor_artifact["resolved_config"], axis_name
                    ),
                },
                "family": final_family,
                "variants": [variant],
            },
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = canonical_json_dumps(payload) + "\n"
    try:
        file_descriptor = os.open(
            output_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError:
        existing = read_existing_lock(output_path)
        if existing != payload:
            raise ExistingLockConflictError(
                "Another process created a different immutable extension lock"
            )
        return existing
    with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    return payload


def _with_payload_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    result["lock_payload_sha256"] = canonical_sha256(result)
    return result


def validate_lock_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != 1 or payload.get("kind") != "minilm_5ep_sft_stage_lock":
        raise ExistingLockConflictError("Existing file is not a supported stage lock")
    stored = str(payload.get("lock_payload_sha256", ""))
    unhashed = dict(payload)
    unhashed.pop("lock_payload_sha256", None)
    if stored != canonical_sha256(unhashed):
        raise ExistingLockConflictError("Existing stage lock payload SHA-256 is invalid")


def read_existing_lock(path: Path) -> dict[str, Any]:
    payload = load_json_object(path, label="existing stage lock")
    validate_lock_payload(payload)
    expected_text = canonical_json_dumps(payload) + "\n"
    if path.read_text(encoding="utf-8") != expected_text:
        raise ExistingLockConflictError("Existing stage lock is not canonical JSON")
    return payload


def _validate_existing_identity(
    lock: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    summary: Mapping[str, Any],
    source_stage: str,
    target_stage: str,
    coordinate: str | None,
) -> None:
    expected = {
        "campaign": plan.get("campaign"),
        "source_stage": source_stage,
        "target_stage": target_stage,
        "coordinate": coordinate,
        "source_plan_sha256": canonical_sha256(plan),
        "effective_stage": (
            f"{target_stage}__{coordinate}" if coordinate else target_stage
        ),
    }
    for key, value in expected.items():
        if lock.get(key) != value:
            raise ExistingLockConflictError(
                f"Existing lock has {key}={lock.get(key)!r}, expected {value!r}"
            )
    parent = lock.get("parent")
    if not isinstance(parent, Mapping):
        raise ExistingLockConflictError("Existing lock has no parent object")
    snapshot, _ = source_snapshot(
        summary,
        campaign_name=str(plan.get("campaign", "")),
        source_stage=source_stage,
        expected_parent=parent,
    )
    if lock.get("source_stage_snapshot_sha256") != canonical_sha256(snapshot):
        raise ExistingLockConflictError(
            "Existing lock differs from the completed source-stage snapshot"
        )


def materialize_stage_lock(
    *,
    plan_path: Path = DEFAULT_PLAN,
    summary_path: Path = DEFAULT_SUMMARY,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    source_stage: str = "lr_log_line",
    target_stage: str = "epoch_line",
    output_path: Path | None = None,
    coordinate: str | None = None,
) -> dict[str, Any]:
    plan = load_json_object(plan_path, label="campaign plan")
    if plan.get("schema_version") != 1 or not str(plan.get("campaign", "")).strip():
        raise StageMaterializationError("Unsupported campaign plan schema")
    # Validate both names even on replay so a lock cannot silently outlive the
    # campaign structure it identifies.
    plan_stage_for_effective_name(plan, source_stage)
    target = stage_by_name(plan, target_stage)
    validate_stage_transition(
        plan,
        source_stage=source_stage,
        target_stage=target_stage,
        coordinate=coordinate,
    )
    if output_path is None:
        suffix = f"_{coordinate}" if coordinate else ""
        output_path = DEFAULT_LOCKS_DIR / f"{target_stage}{suffix}.lock.json"
    summary = load_source_summary(summary_path, source_stage=source_stage)

    if output_path.exists():
        existing = read_existing_lock(output_path)
        _validate_existing_identity(
            existing,
            plan=plan,
            summary=summary,
            source_stage=source_stage,
            target_stage=target_stage,
            coordinate=coordinate,
        )
        return existing

    snapshot, row = source_snapshot(
        summary,
        campaign_name=str(plan["campaign"]),
        source_stage=source_stage,
    )
    parent = resolve_parent(
        plan,
        source_stage=source_stage,
        row=row,
        artifacts_dir=artifacts_dir,
    )
    resolved_stage = resolve_target_stage(
        target,
        parent=parent,
        coordinate=coordinate,
        allowed_override_keys=set(plan.get("allowed_override_keys", [])),
    )
    payload = _with_payload_hash(
        {
            "schema_version": 1,
            "kind": "minilm_5ep_sft_stage_lock",
            "campaign": plan["campaign"],
            "source_stage": source_stage,
            "target_stage": target_stage,
            "coordinate": coordinate,
            "effective_stage": (
                f"{target_stage}__{coordinate}" if coordinate else target_stage
            ),
            "source_plan_sha256": canonical_sha256(plan),
            "source_stage_snapshot_sha256": canonical_sha256(snapshot),
            "selection_metric": plan.get("selection_protocol", {}).get(
                "primary_metric"
            ),
            "parent": parent,
            "resolved_stage": resolved_stage,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = canonical_json_dumps(payload) + "\n"
    try:
        file_descriptor = os.open(
            output_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError:
        existing = read_existing_lock(output_path)
        if existing != payload:
            raise ExistingLockConflictError(
                "Another process created a different immutable stage lock"
            )
        return existing
    with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--from-stage", default="lr_log_line")
    parser.add_argument("--to-stage", default="epoch_line")
    parser.add_argument("--coordinate")
    parser.add_argument(
        "--boundary-extension",
        action="store_true",
        help=(
            "materialize the one extension requested by --from-stage instead "
            "of transitioning to --to-stage"
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output
    if output_path is None:
        if args.boundary_extension:
            output_path = (
                DEFAULT_LOCKS_DIR / f"{args.from_stage}_boundary.lock.json"
            )
        else:
            suffix = f"_{args.coordinate}" if args.coordinate else ""
            output_path = DEFAULT_LOCKS_DIR / f"{args.to_stage}{suffix}.lock.json"
    existed = output_path.exists()
    if args.boundary_extension:
        if args.coordinate is not None:
            raise SystemExit(
                "Pass the effective --from-stage (stage__coordinate) for a "
                "boundary extension; do not also pass --coordinate"
            )
        lock = materialize_boundary_extension_lock(
            plan_path=args.plan,
            summary_path=args.summary,
            artifacts_dir=args.artifacts_dir,
            source_stage=args.from_stage,
            output_path=output_path,
        )
    else:
        lock = materialize_stage_lock(
            plan_path=args.plan,
            summary_path=args.summary,
            artifacts_dir=args.artifacts_dir,
            source_stage=args.from_stage,
            target_stage=args.to_stage,
            output_path=output_path,
            coordinate=args.coordinate,
        )
    print(
        canonical_json_dumps(
            {
                "status": "existing" if existed else "created",
                "path": str(output_path),
                "lock_payload_sha256": lock["lock_payload_sha256"],
                "parent_experiment": lock["parent"]["experiment"],
                "parent_run_id": lock["parent"]["run_id"],
                "variants": len(lock["resolved_stage"]["variants"]),
            }
        )
    )


if __name__ == "__main__":
    main()
