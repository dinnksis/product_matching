#!/usr/bin/env python3
"""Build locked MiniLM-5ep notebooks for supervised recipe ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from textwrap import dedent
from typing import Any, Iterable, Mapping

import nbformat as nbf

import create_cross_encoder_training_notebook as cross_builder
import create_minilm_5ep_team_ablation_notebook as team_builder
import create_minilm_validation_baseline_notebook as baseline_builder
import materialize_minilm_5ep_sft_hparam_stage as stage_materializer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "configs" / "minilm_5ep_sft_hparam_search_v1.json"
DEFAULT_OUTPUT_DIR = (
    ROOT / "notebooks" / "minilm_5ep_team_ablation" / "sft_hparams_v1"
)
BASE_CONFIG_PATH = (
    ROOT / "configs" / "cross_encoder_minilm_llm_pretrain_5ep_human_ft.json"
)
READY_STATUS = "ready"
SAFE_OVERRIDE_KEYS = frozenset(
    {
        "epochs",
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "label_smoothing",
        "max_grad_norm",
        "batch_size",
        "gradient_accumulation",
        "model_load_kwargs",
        "seed",
    }
)
SAFE_MODEL_LOAD_KWARGS = frozenset({"classifier_dropout"})
FIXED_LOSS_HOOK_SOURCE = dedent(
    """
    from __future__ import annotations

    import torch
    import torch.nn.functional as F


    def initialize_loss(*, train_frame, device, rank, world_size):
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
        per_example_bce = F.binary_cross_entropy_with_logits(
            logits.float(), targets, reduction="none"
        )
        bce = (per_example_bce * sample_weights).sum() / sample_weights.sum()
        return {
            "loss": bce,
            "bce": bce.detach(),
        }
    """
).strip() + "\n"
FIXED_LOSS_HOOK_SHA256 = hashlib.sha256(
    FIXED_LOSS_HOOK_SOURCE.encode("utf-8")
).hexdigest()
DEFAULT_LOSS_VARIANT = "bce"


def _balanced_bce_source(*, by_category: bool, power: float, name: str) -> str:
    return dedent(
        f"""
        from __future__ import annotations

        from collections import Counter
        import json

        import torch
        import torch.nn.functional as F


        _PAIR_BALANCE_WEIGHTS = None
        BALANCE_BY_CATEGORY = {by_category!r}
        BALANCE_POWER = {power!r}
        LOSS_VARIANT = {name!r}


        def initialize_loss(*, train_frame, device, rank, world_size):
            global _PAIR_BALANCE_WEIGHTS
            labels = (train_frame["target"].astype(float).to_numpy() >= 0.5).astype(int)
            if BALANCE_BY_CATEGORY:
                categories = train_frame["category_1"].astype(str).tolist()
                keys = list(zip(categories, labels.tolist()))
                observed = set(keys)
                expected = {{(category, label) for category in set(categories) for label in (0, 1)}}
                if observed != expected:
                    raise ValueError("Every training category must contain both binary classes")
            else:
                keys = labels.tolist()
                if set(keys) != {{0, 1}}:
                    raise ValueError("Balanced binary BCE requires both classes")
            counts = Counter(keys)
            raw = [float(counts[key]) ** (-BALANCE_POWER) for key in keys]
            normalizer = sum(raw) / len(raw)
            weights = [value / normalizer for value in raw]
            _PAIR_BALANCE_WEIGHTS = torch.tensor(
                weights, dtype=torch.float32, device=device
            )
            print(json.dumps({{
                "loss_variant": LOSS_VARIANT,
                "balance_by_category": BALANCE_BY_CATEGORY,
                "balance_power": BALANCE_POWER,
                "weight_min": min(weights),
                "weight_max": max(weights),
                "rank": rank,
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
            per_example_bce = F.binary_cross_entropy_with_logits(
                logits.float(), targets, reduction="none"
            )
            bce = (per_example_bce * sample_weights).sum() / sample_weights.sum()
            balance_weights = _PAIR_BALANCE_WEIGHTS.index_select(0, pair_indices)
            combined_weights = sample_weights * balance_weights
            balanced_bce = (
                (per_example_bce * combined_weights).sum() / sample_weights.sum()
            )
            return {{
                "loss": balanced_bce,
                "bce": bce.detach(),
                "balanced_bce": balanced_bce.detach(),
                "batch_balance_weight": balance_weights.mean().detach(),
            }}
        """
    ).strip() + "\n"


def _negative_focal_source(*, gamma: float, name: str) -> str:
    return dedent(
        f"""
        from __future__ import annotations

        import json

        import torch
        import torch.nn.functional as F


        FOCAL_GAMMA = {gamma!r}
        LOSS_VARIANT = {name!r}


        def initialize_loss(*, train_frame, device, rank, world_size):
            print(json.dumps({{
                "loss_variant": LOSS_VARIANT,
                "gamma": FOCAL_GAMMA,
                "rank": rank,
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
            per_example_bce = F.binary_cross_entropy_with_logits(
                logits.float(), targets, reduction="none"
            )
            bce = (per_example_bce * sample_weights).sum() / sample_weights.sum()
            negative_factor = logits.float().sigmoid().pow(FOCAL_GAMMA)
            focal_per_example = torch.where(
                targets >= 0.5,
                per_example_bce,
                per_example_bce * negative_factor,
            )
            focal_bce = (
                (focal_per_example * sample_weights).sum() / sample_weights.sum()
            )
            negative_mask = targets < 0.5
            mean_factor = (
                negative_factor[negative_mask].mean()
                if negative_mask.any()
                else negative_factor.new_zeros(())
            )
            return {{
                "loss": focal_bce,
                "bce": bce.detach(),
                "negative_focal_bce": focal_bce.detach(),
                "negative_focal_factor": mean_factor.detach(),
            }}
        """
    ).strip() + "\n"


FOCAL_GAMMA2_SCALE4_SOURCE = dedent(
    """
    from __future__ import annotations

    import json

    import torch
    import torch.nn.functional as F


    FOCAL_GAMMA = 2.0
    FOCAL_SCALE = 4.0
    LOSS_VARIANT = "focal_bce_gamma2_scale4"


    def initialize_loss(*, train_frame, device, rank, world_size):
        print(json.dumps({
            "loss_variant": LOSS_VARIANT,
            "gamma": FOCAL_GAMMA,
            "scale": FOCAL_SCALE,
            "scale_reference": "neutral_probability_0.5",
            "rank": rank,
        }, ensure_ascii=False), flush=True)


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
        per_example_bce = F.binary_cross_entropy_with_logits(
            logits.float(), targets, reduction="none"
        )
        bce = (per_example_bce * sample_weights).sum() / sample_weights.sum()
        probabilities = logits.float().sigmoid()
        hard_positive = targets >= 0.5
        probability_of_target = torch.where(
            hard_positive, probabilities, 1.0 - probabilities
        )
        focal_factor = FOCAL_SCALE * (
            1.0 - probability_of_target
        ).pow(FOCAL_GAMMA)
        focal_bce = (
            (per_example_bce * sample_weights * focal_factor).sum()
            / sample_weights.sum()
        )
        return {
            "loss": focal_bce,
            "bce": bce.detach(),
            "focal_bce": focal_bce.detach(),
            "focal_factor_mean": focal_factor.mean().detach(),
        }
    """
).strip() + "\n"


def _balanced_focal_source(*, by_category: bool, power: float, name: str) -> str:
    """Build an allowlisted balance-plus-focal hook with stable loss scaling."""
    return dedent(
        f"""
        from __future__ import annotations

        from collections import Counter
        import json

        import torch
        import torch.nn.functional as F


        _PAIR_BALANCE_WEIGHTS = None
        BALANCE_BY_CATEGORY = {by_category!r}
        BALANCE_POWER = {power!r}
        FOCAL_GAMMA = 2.0
        FOCAL_SCALE = 4.0
        LOSS_VARIANT = {name!r}


        def initialize_loss(*, train_frame, device, rank, world_size):
            global _PAIR_BALANCE_WEIGHTS
            labels = (train_frame["target"].astype(float).to_numpy() >= 0.5).astype(int)
            if BALANCE_BY_CATEGORY:
                categories = train_frame["category_1"].astype(str).tolist()
                keys = list(zip(categories, labels.tolist()))
                observed = set(keys)
                expected = {{(category, label) for category in set(categories) for label in (0, 1)}}
                if observed != expected:
                    raise ValueError("Every training category must contain both binary classes")
            else:
                keys = labels.tolist()
                if set(keys) != {{0, 1}}:
                    raise ValueError("Balanced focal BCE requires both classes")
            counts = Counter(keys)
            raw = [float(counts[key]) ** (-BALANCE_POWER) for key in keys]
            normalizer = sum(raw) / len(raw)
            weights = [value / normalizer for value in raw]
            _PAIR_BALANCE_WEIGHTS = torch.tensor(
                weights, dtype=torch.float32, device=device
            )
            print(json.dumps({{
                "loss_variant": LOSS_VARIANT,
                "balance_by_category": BALANCE_BY_CATEGORY,
                "balance_power": BALANCE_POWER,
                "gamma": FOCAL_GAMMA,
                "scale": FOCAL_SCALE,
                "scale_reference": "neutral_probability_0.5",
                "weight_min": min(weights),
                "weight_max": max(weights),
                "rank": rank,
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
            per_example_bce = F.binary_cross_entropy_with_logits(
                logits.float(), targets, reduction="none"
            )
            bce = (per_example_bce * sample_weights).sum() / sample_weights.sum()
            balance_weights = _PAIR_BALANCE_WEIGHTS.index_select(0, pair_indices)
            combined_weights = sample_weights * balance_weights
            probabilities = logits.float().sigmoid()
            hard_positive = targets >= 0.5
            probability_of_target = torch.where(
                hard_positive, probabilities, 1.0 - probabilities
            )
            focal_factor = FOCAL_SCALE * (
                1.0 - probability_of_target
            ).pow(FOCAL_GAMMA)
            balanced_focal_bce = (
                (per_example_bce * combined_weights * focal_factor).sum()
                / sample_weights.sum()
            )
            return {{
                "loss": balanced_focal_bce,
                "bce": bce.detach(),
                "balanced_focal_bce": balanced_focal_bce.detach(),
                "balance_weight_mean": balance_weights.mean().detach(),
                "focal_factor_mean": focal_factor.mean().detach(),
            }}
        """
    ).strip() + "\n"


TOPK_RANKNET_LOSS_SOURCE = dedent(
    """
    from __future__ import annotations

    import json

    import torch
    import torch.nn.functional as F


    _CATEGORY_CODES = None
    LAMBDA_RANK = 0.25
    LOSS_VARIANT = "bce_topk_ranknet_lambda025"


    def initialize_loss(*, train_frame, device, rank, world_size):
        global _CATEGORY_CODES
        categories = train_frame["category_1"].astype(str).tolist()
        names = sorted(set(categories))
        lookup = {name: index for index, name in enumerate(names)}
        _CATEGORY_CODES = torch.tensor(
            [lookup[name] for name in categories], dtype=torch.long, device=device
        )
        print(json.dumps({
            "loss_variant": LOSS_VARIANT,
            "lambda_rank": LAMBDA_RANK,
            "rank": rank,
            "category_codes": lookup,
        }, ensure_ascii=False), flush=True)


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
        per_example_bce = F.binary_cross_entropy_with_logits(
            logits.float(), targets, reduction="none"
        )
        bce = (per_example_bce * sample_weights).sum() / sample_weights.sum()
        codes = _CATEGORY_CODES.index_select(0, pair_indices)
        positive = targets >= 0.5
        rank_parts = []
        active_categories = logits.new_zeros(())
        for category in torch.unique(codes):
            category_mask = codes == category
            positive_logits = logits.float()[category_mask & positive]
            negative_logits = logits.float()[category_mask & ~positive]
            if positive_logits.numel() and negative_logits.numel():
                top_count = max(1, (negative_logits.numel() + 1) // 2)
                hard_negatives = torch.topk(negative_logits, k=top_count).values
                rank_parts.append(
                    F.softplus(
                        hard_negatives[None, :] - positive_logits[:, None]
                    ).reshape(-1)
                )
                active_categories = active_categories + 1
        rank_loss = (
            torch.cat(rank_parts).mean()
            if rank_parts
            else logits.float().sum() * 0.0
        )
        loss = bce + LAMBDA_RANK * rank_loss
        return {
            "loss": loss,
            "bce": bce.detach(),
            "topk_ranknet": rank_loss.detach(),
            "ranknet_active_categories": active_categories.detach(),
        }
    """
).strip() + "\n"


LOSS_VARIANT_SOURCES = {
    DEFAULT_LOSS_VARIANT: FIXED_LOSS_HOOK_SOURCE,
    "balanced_binary_bce": _balanced_bce_source(
        by_category=False,
        power=1.0,
        name="balanced_binary_bce",
    ),
    "balanced_category_class_sqrt_bce": _balanced_bce_source(
        by_category=True,
        power=0.5,
        name="balanced_category_class_sqrt_bce",
    ),
    "balanced_category_class_bce": _balanced_bce_source(
        by_category=True,
        power=1.0,
        name="balanced_category_class_bce",
    ),
    "negative_focal_bce_gamma1": _negative_focal_source(
        gamma=1.0,
        name="negative_focal_bce_gamma1",
    ),
    "negative_focal_bce_gamma2": _negative_focal_source(
        gamma=2.0,
        name="negative_focal_bce_gamma2",
    ),
    "focal_bce_gamma2_scale4": FOCAL_GAMMA2_SCALE4_SOURCE,
    "balanced_binary_focal_gamma2_scale4": _balanced_focal_source(
        by_category=False,
        power=1.0,
        name="balanced_binary_focal_gamma2_scale4",
    ),
    "balanced_category_class_sqrt_focal_gamma2_scale4": _balanced_focal_source(
        by_category=True,
        power=0.5,
        name="balanced_category_class_sqrt_focal_gamma2_scale4",
    ),
    "balanced_category_class_focal_gamma2_scale4": _balanced_focal_source(
        by_category=True,
        power=1.0,
        name="balanced_category_class_focal_gamma2_scale4",
    ),
    "bce_topk_ranknet_lambda025": TOPK_RANKNET_LOSS_SOURCE,
}
LOSS_VARIANT_SHA256 = {
    name: hashlib.sha256(source.encode("utf-8")).hexdigest()
    for name, source in LOSS_VARIANT_SOURCES.items()
}


class CampaignConfigError(ValueError):
    """Raised when a campaign could silently violate the frozen protocol."""


def variant_loss(variant: Mapping[str, Any]) -> tuple[str, str, str]:
    name = str(variant.get("loss_variant", DEFAULT_LOSS_VARIANT)).strip()
    if name not in LOSS_VARIANT_SOURCES:
        raise CampaignConfigError(
            f"Unknown loss_variant {name!r}; expected one of "
            f"{sorted(LOSS_VARIANT_SOURCES)}"
        )
    return name, LOSS_VARIANT_SOURCES[name], LOSS_VARIANT_SHA256[name]


def _declared_stage_loss_variants(stage: Mapping[str, Any]) -> set[str]:
    """Return only loss names explicitly admitted by one campaign stage."""
    raw_losses = stage.get("loss_variants", [])
    if not isinstance(raw_losses, list) or any(
        not isinstance(name, str) or not name.strip() for name in raw_losses
    ):
        raise CampaignConfigError("loss_variants must be a list of non-empty strings")
    declared = set(raw_losses)
    combination = stage.get("conditional_combination")
    if isinstance(combination, Mapping):
        variants_by_balance = combination.get("variants_by_balance", {})
        if not isinstance(variants_by_balance, Mapping):
            raise CampaignConfigError(
                "conditional_combination.variants_by_balance must be an object"
            )
        if any(
            not isinstance(name, str) or not name.strip()
            for pair in variants_by_balance.items()
            for name in pair
        ):
            raise CampaignConfigError(
                "conditional loss names must be non-empty strings"
            )
        declared.update(variants_by_balance)
        declared.update(variants_by_balance.values())
    return declared


def stage_loss_allowlist(
    plan: Mapping[str, Any],
    *,
    stage_name: str,
) -> set[str]:
    """Bind executable losses to the reviewed plan, not the larger registry.

    The registry intentionally contains a few historical hooks so old artifacts
    remain inspectable.  They must not become runnable merely by putting their
    name into a ready variant or a hand-edited stage lock.
    """
    stages = plan.get("stages")
    if not isinstance(stages, list):
        raise CampaignConfigError("SFT campaign must declare stages")
    matches = [
        stage
        for stage in stages
        if isinstance(stage, Mapping) and stage.get("name") == stage_name
    ]
    if len(matches) != 1:
        raise CampaignConfigError(
            f"Expected exactly one campaign stage {stage_name!r}, found {len(matches)}"
        )
    declared = _declared_stage_loss_variants(matches[0])
    if declared:
        return declared
    if stage_name == "confirmation":
        inherited: set[str] = set()
        for stage in stages:
            if isinstance(stage, Mapping):
                inherited.update(_declared_stage_loss_variants(stage))
        if inherited:
            return inherited
    return {DEFAULT_LOSS_VARIANT}


def load_plan(path: Path = DEFAULT_PLAN) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise CampaignConfigError("SFT campaign must use schema_version=1")
    if payload.get("baseline_run_id") != team_builder.SIGNIFICANCE_BASELINE_RUN_ID:
        raise CampaignConfigError("Campaign baseline_run_id differs from the frozen baseline")
    declared_overrides = set(payload.get("allowed_override_keys", []))
    if declared_overrides != SAFE_OVERRIDE_KEYS:
        raise CampaignConfigError(
            "Campaign allowed_override_keys must exactly match the generator's "
            f"safe SFT allowlist: {sorted(SAFE_OVERRIDE_KEYS)}"
        )
    stages = payload.get("stages")
    if not isinstance(stages, list) or not stages:
        raise CampaignConfigError("SFT campaign must declare at least one stage")
    declared_loss_variants: set[str] = set()
    for stage in stages:
        if not isinstance(stage, Mapping):
            raise CampaignConfigError("Every campaign stage must be an object")
        declared_loss_variants.update(_declared_stage_loss_variants(stage))
    unknown_losses = declared_loss_variants - set(LOSS_VARIANT_SOURCES)
    if unknown_losses:
        raise CampaignConfigError(
            "Campaign references loss variants outside the hard-coded registry: "
            f"{sorted(unknown_losses)}"
        )
    return payload


def _validate_variant_identity(
    variant: Mapping[str, Any],
    *,
    seen_experiments: set[str],
    seen_slugs: set[str],
    allowed_loss_variants: set[str] | None = None,
) -> None:
    experiment = str(variant.get("experiment", ""))
    kernel_slug = str(variant.get("kernel_slug", ""))
    title = str(variant.get("title", "")).strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", experiment):
        raise CampaignConfigError(f"Invalid experiment label: {experiment!r}")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", kernel_slug):
        raise CampaignConfigError(f"Invalid Kaggle kernel slug: {kernel_slug!r}")
    title_slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    if title_slug != kernel_slug:
        raise CampaignConfigError(
            f"Kaggle title {title!r} resolves to {title_slug!r}, not "
            f"declared kernel_slug {kernel_slug!r}"
        )
    if experiment in seen_experiments:
        raise CampaignConfigError(f"Duplicate experiment label: {experiment}")
    if kernel_slug in seen_slugs:
        raise CampaignConfigError(f"Duplicate Kaggle kernel slug: {kernel_slug}")
    loss_variant, _, _ = variant_loss(variant)
    if (
        allowed_loss_variants is not None
        and loss_variant not in allowed_loss_variants
    ):
        raise CampaignConfigError(
            f"Loss variant {loss_variant!r} is registered but not declared for "
            f"this campaign stage; allowed: {sorted(allowed_loss_variants)}"
        )
    seen_experiments.add(experiment)
    seen_slugs.add(kernel_slug)


def variant_provenance_alias(
    variant: Mapping[str, Any],
    *,
    stage: str,
) -> dict[str, str] | None:
    """Validate the one narrow alias allowed for the legacy protocol control."""
    raw = variant.get("provenance_alias")
    if raw is None:
        return None
    if variant.get("role") != "current_protocol_control":
        raise CampaignConfigError(
            "provenance_alias is allowed only on current_protocol_control"
        )
    if not isinstance(raw, Mapping):
        raise CampaignConfigError("provenance_alias must be an object")
    required = {
        "type",
        "recorded_stage",
        "canonical_stage",
        "accepted_completion_notes_sha256",
        "reason",
    }
    if set(raw) != required:
        raise CampaignConfigError(
            "provenance_alias must contain exactly " + repr(sorted(required))
        )
    result = {key: str(raw[key]).strip() for key in required}
    if result["type"] != "control_submitted_before_search_strategy_revision":
        raise CampaignConfigError("Unsupported provenance_alias type")
    if result["canonical_stage"] != stage:
        raise CampaignConfigError(
            "provenance_alias canonical_stage differs from its campaign stage"
        )
    if not result["recorded_stage"] or not result["reason"]:
        raise CampaignConfigError("provenance_alias needs recorded_stage and reason")
    if result["recorded_stage"] == result["canonical_stage"]:
        raise CampaignConfigError("provenance_alias must describe an actual stage rename")
    if not re.fullmatch(
        r"[0-9a-f]{64}", result["accepted_completion_notes_sha256"]
    ):
        raise CampaignConfigError(
            "provenance_alias accepted_completion_notes_sha256 is invalid"
        )
    return result


def ready_variants(
    plan: Mapping[str, Any],
    *,
    stage_name: str | None = None,
) -> Iterable[tuple[str, Mapping[str, Any]]]:
    seen_experiments: set[str] = set()
    seen_slugs: set[str] = set()
    selected_stage = False
    for raw_stage in plan["stages"]:
        if not isinstance(raw_stage, Mapping):
            raise CampaignConfigError("Every campaign stage must be an object")
        name = str(raw_stage.get("name", "")).strip()
        if not name:
            raise CampaignConfigError("Every campaign stage needs a name")
        if stage_name is not None and name != stage_name:
            continue
        selected_stage = True
        if raw_stage.get("status") != READY_STATUS:
            continue
        variants = raw_stage.get("variants")
        if not isinstance(variants, list) or not variants:
            raise CampaignConfigError(f"Ready stage {name!r} has no variants")
        allowed_losses = stage_loss_allowlist(plan, stage_name=name)
        for variant in variants:
            if not isinstance(variant, Mapping):
                raise CampaignConfigError(f"Stage {name!r} contains a non-object variant")
            _validate_variant_identity(
                variant,
                seen_experiments=seen_experiments,
                seen_slugs=seen_slugs,
                allowed_loss_variants=allowed_losses,
            )
            variant_provenance_alias(variant, stage=name)
            yield name, variant
    if stage_name is not None and not selected_stage:
        raise CampaignConfigError(f"Unknown campaign stage: {stage_name!r}")


def variant_config(
    base_config: Mapping[str, Any],
    plan: Mapping[str, Any],
    variant: Mapping[str, Any],
) -> dict[str, Any]:
    declared_allowed = set(plan.get("allowed_override_keys", []))
    if declared_allowed != SAFE_OVERRIDE_KEYS:
        raise CampaignConfigError("Campaign override allowlist was changed after validation")
    allowed = SAFE_OVERRIDE_KEYS
    overrides = variant.get("overrides")
    if not isinstance(overrides, Mapping):
        raise CampaignConfigError(
            f"Variant {variant.get('experiment')!r} must declare overrides"
        )
    unknown = set(overrides) - allowed
    if unknown:
        raise CampaignConfigError(
            f"Variant {variant.get('experiment')!r} changes forbidden keys: "
            f"{sorted(unknown)}"
        )
    result = deepcopy(dict(base_config))
    result.update(overrides)
    integer_fields = {
        "epochs",
        "batch_size",
        "eval_batch_size",
        "gradient_accumulation",
        "max_length",
        "seed",
    }
    invalid_integer_types = [
        key
        for key in integer_fields
        if isinstance(result[key], bool) or not isinstance(result[key], int)
    ]
    if invalid_integer_types:
        raise CampaignConfigError(
            f"Variant {variant.get('experiment')!r} needs exact integers for: "
            f"{sorted(invalid_integer_types)}"
        )
    invalid = [
        key for key in integer_fields - {"seed"} if int(result[key]) <= 0
    ]
    if invalid:
        raise CampaignConfigError(
            f"Variant {variant.get('experiment')!r} has non-positive values: "
            f"{sorted(invalid)}"
        )
    if not 0 <= int(result["seed"]) <= 2**32 - 1:
        raise CampaignConfigError("seed must be in [0, 2**32 - 1]")
    finite_fields = {
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "label_smoothing",
        "max_grad_norm",
    }
    invalid_floats = [
        key
        for key in finite_fields
        if isinstance(result[key], bool)
        or not isinstance(result[key], (int, float))
        or not math.isfinite(float(result[key]))
    ]
    if invalid_floats:
        raise CampaignConfigError(
            f"Variant {variant.get('experiment')!r} needs finite numbers for: "
            f"{sorted(invalid_floats)}"
        )
    if float(result["learning_rate"]) <= 0 or float(result["max_grad_norm"]) <= 0:
        raise CampaignConfigError("learning_rate and max_grad_norm must be positive")
    if float(result["weight_decay"]) < 0:
        raise CampaignConfigError("weight_decay must be non-negative")
    if not 0 <= float(result["warmup_ratio"]) < 1:
        raise CampaignConfigError("warmup_ratio must be in [0, 1)")
    if not 0 <= float(result["label_smoothing"]) < 1:
        raise CampaignConfigError("label_smoothing must be in [0, 1)")
    model_load_kwargs = result.get("model_load_kwargs", {})
    if not isinstance(model_load_kwargs, Mapping):
        raise CampaignConfigError("model_load_kwargs must be an object")
    if unknown_model_kwargs := set(model_load_kwargs) - SAFE_MODEL_LOAD_KWARGS:
        raise CampaignConfigError(
            "Unsafe model_load_kwargs are forbidden: "
            f"{sorted(unknown_model_kwargs)}"
        )
    for key, value in model_load_kwargs.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 <= float(value) < 1
        ):
            raise CampaignConfigError(f"{key} must be a finite number in [0, 1)")
    for key, value in base_config.items():
        if key not in allowed and result.get(key) != value:
            raise CampaignConfigError(f"Frozen config key changed: {key}")
    return result


def resolved_config_overrides(
    base_config: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    removed = set(base_config) - set(resolved_config)
    if removed:
        raise CampaignConfigError(
            f"Resolved stage config removed frozen keys: {sorted(removed)}"
        )
    return {
        key: deepcopy(value)
        for key, value in resolved_config.items()
        if key not in base_config or base_config.get(key) != value
    }


def load_stage_lock(
    path: Path,
    *,
    plan: Mapping[str, Any],
    base_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a canonical lock and revalidate every recipe through SAFE overrides."""
    try:
        lock = stage_materializer.read_existing_lock(path)
    except stage_materializer.StageMaterializationError as error:
        raise CampaignConfigError(f"Invalid stage lock {path}: {error}") from error
    if lock.get("campaign") != plan.get("campaign"):
        raise CampaignConfigError("Stage lock belongs to a different campaign")
    expected_plan_sha = stage_materializer.canonical_sha256(plan)
    if lock.get("source_plan_sha256") != expected_plan_sha:
        raise CampaignConfigError("Stage lock was materialized from a different plan")
    if lock.get("selection_metric") != plan.get("selection_protocol", {}).get(
        "primary_metric"
    ):
        raise CampaignConfigError("Stage lock changed the primary selection metric")
    target_stage = str(lock.get("target_stage", "")).strip()
    if not target_stage:
        raise CampaignConfigError("Stage lock has no target_stage")
    try:
        target_definition = stage_materializer.stage_by_name(plan, target_stage)
    except stage_materializer.StageMaterializationError as error:
        raise CampaignConfigError(str(error)) from error
    transition_kind = str(lock.get("transition_kind", "stage_transition"))
    coordinate = lock.get("coordinate")
    if transition_kind == "stage_transition":
        try:
            stage_materializer.validate_stage_transition(
                plan,
                source_stage=str(lock.get("source_stage", "")),
                target_stage=target_stage,
                coordinate=coordinate,
            )
        except stage_materializer.StageMaterializationError as error:
            raise CampaignConfigError(str(error)) from error
    elif transition_kind == "conditional_boundary_extension":
        expected_effective = (
            f"{target_stage}__{coordinate}"
            if coordinate is not None
            else target_stage
        )
        if (
            lock.get("source_stage") != expected_effective
            or lock.get("effective_stage") != expected_effective
        ):
            raise CampaignConfigError(
                "Conditional extension lock must stay within its effective stage"
            )
    else:
        raise CampaignConfigError(
            f"Unknown stage-lock transition_kind {transition_kind!r}"
        )
    base = (
        dict(base_config)
        if base_config is not None
        else cross_builder.load_training_config(BASE_CONFIG_PATH)
    )

    parent = lock.get("parent")
    if not isinstance(parent, Mapping):
        raise CampaignConfigError("Stage lock has no parent provenance")
    parent_config = parent.get("resolved_config")
    if not isinstance(parent_config, Mapping):
        raise CampaignConfigError("Stage lock parent has no resolved config")
    parent_variant = {
        "experiment": parent.get("experiment"),
        "overrides": resolved_config_overrides(base, parent_config),
    }
    validated_parent = variant_config(base, plan, parent_variant)
    if validated_parent != dict(parent_config):
        raise CampaignConfigError("Stage lock parent config is not reproducible")
    parent_recipe_sha = team_builder.canonical_sha256(validated_parent)
    if parent.get("recipe_sha256") != parent_recipe_sha:
        raise CampaignConfigError("Stage lock parent recipe SHA-256 differs")
    if not re.fullmatch(
        r"[0-9a-f]{64}", str(parent.get("iid_predictions_sha256", ""))
    ):
        raise CampaignConfigError("Stage lock parent IID predictions SHA-256 is invalid")
    if not str(parent.get("experiment", "")).strip() or not str(
        parent.get("run_id", "")
    ).strip():
        raise CampaignConfigError("Stage lock parent identity is incomplete")
    parent_loss_variant = str(
        parent.get("loss_variant", DEFAULT_LOSS_VARIANT)
    ).strip()
    _, _, parent_loss_sha256 = variant_loss(
        {"loss_variant": parent_loss_variant}
    )
    stored_parent_loss_sha256 = parent.get("loss_hook_sha256")
    if (
        stored_parent_loss_sha256 is not None
        and stored_parent_loss_sha256 != parent_loss_sha256
    ):
        raise CampaignConfigError("Stage lock parent loss-hook SHA-256 differs")

    resolved_stage = lock.get("resolved_stage")
    if not isinstance(resolved_stage, Mapping):
        raise CampaignConfigError("Stage lock has no resolved_stage object")
    reused_parent = resolved_stage.get("reused_parent")
    if not isinstance(reused_parent, Mapping) or (
        reused_parent.get("experiment") != parent.get("experiment")
        or reused_parent.get("run_id") != parent.get("run_id")
    ):
        raise CampaignConfigError("Stage lock reused_parent differs from parent")
    variants = resolved_stage.get("variants")
    if not isinstance(variants, list) or not variants:
        raise CampaignConfigError("Stage lock resolved_stage has no variants")
    prior_entries = lock.get("prior_entries", [])
    if not isinstance(prior_entries, list):
        raise CampaignConfigError("Stage lock prior_entries must be a list")
    if transition_kind == "conditional_boundary_extension":
        if not prior_entries:
            raise CampaignConfigError(
                "Conditional extension lock has no frozen prior entries"
            )
    elif prior_entries:
        raise CampaignConfigError(
            "A normal stage-transition lock cannot contain prior_entries"
        )
    prior_hypotheses = sum(
        isinstance(entry, Mapping) and entry.get("is_hypothesis") is True
        for entry in prior_entries
    )
    family = resolved_stage.get("family")
    if not isinstance(family, Mapping) or family.get(
        "planned_candidate_hypotheses"
    ) != len(variants) + prior_hypotheses:
        raise CampaignConfigError("Stage lock hypothesis family differs from variants")
    if family.get("correction") != "holm":
        raise CampaignConfigError("Stage lock hypothesis family must use Holm correction")
    maximum_hypotheses = family.get("maximum_hypotheses")
    reserved_hypotheses = family.get("reserved_conditional_extensions")
    if (
        isinstance(maximum_hypotheses, bool)
        or not isinstance(maximum_hypotheses, int)
        or isinstance(reserved_hypotheses, bool)
        or not isinstance(reserved_hypotheses, int)
        or reserved_hypotheses < 0
        or maximum_hypotheses
        != len(variants) + prior_hypotheses + reserved_hypotheses
    ):
        raise CampaignConfigError("Stage lock maximum hypothesis family is invalid")
    declared_family = target_definition.get("family")
    if isinstance(declared_family, Mapping):
        if transition_kind == "stage_transition":
            if dict(family) != dict(declared_family):
                raise CampaignConfigError(
                    "Stage lock family differs from the campaign plan"
                )
        else:
            expected_family = dict(declared_family)
            expected_family["planned_candidate_hypotheses"] = (
                int(expected_family["planned_candidate_hypotheses"]) + 1
            )
            expected_family["reserved_conditional_extensions"] = (
                int(expected_family["reserved_conditional_extensions"]) - 1
            )
            if expected_family["reserved_conditional_extensions"] < 0 or dict(
                family
            ) != expected_family:
                raise CampaignConfigError(
                    "Conditional extension family does not consume one reserved slot"
                )
    expected_effective_stage = (
        f"{target_stage}__{coordinate}" if coordinate is not None else target_stage
    )
    if lock.get("effective_stage") != expected_effective_stage:
        raise CampaignConfigError("Stage lock effective_stage is invalid")
    axis_schema = isinstance(target_definition.get("axis"), Mapping) or isinstance(
        target_definition.get("axes"), Mapping
    )
    expected_variant_levels: list[Any] | None = None
    declared_extension_levels: list[Any] = []
    axis_name: str | None = None
    if axis_schema:
        try:
            axis_name, base_levels, effective_batch_recipes = (
                stage_materializer._axis_definition(
                    target_definition,
                    coordinate=coordinate,
                )
            )
            declared_extension_levels = (
                stage_materializer._declared_extension_levels(
                    target_definition,
                    axis_name=axis_name,
                )
            )
            parent_level = stage_materializer._effective_axis_value(
                validated_parent, axis_name
            )
        except stage_materializer.StageMaterializationError as error:
            raise CampaignConfigError(str(error)) from error
        if resolved_stage.get("axis") != axis_name or (
            stage_materializer.canonical_json_dumps(
                resolved_stage.get("parent_level")
            )
            != stage_materializer.canonical_json_dumps(parent_level)
        ):
            raise CampaignConfigError("Stage lock axis or parent level differs")
        parent_matches = [
            level
            for level in base_levels
            if stage_materializer.canonical_json_dumps(level)
            == stage_materializer.canonical_json_dumps(parent_level)
        ]
        if len(parent_matches) != 1:
            raise CampaignConfigError("Stage lock parent is not one planned axis level")
        base_candidate_levels = [
            level
            for level in base_levels
            if stage_materializer.canonical_json_dumps(level)
            != stage_materializer.canonical_json_dumps(parent_level)
        ]
        if transition_kind == "stage_transition":
            expected_variant_levels = base_candidate_levels
            if resolved_stage.get("levels") != base_levels:
                raise CampaignConfigError("Stage lock levels differ from the campaign plan")
            frozen_extensions = resolved_stage.get("conditional_extension_levels")
            if frozen_extensions is not None and frozen_extensions != declared_extension_levels:
                raise CampaignConfigError(
                    "Stage lock conditional extensions differ from the campaign plan"
                )
        else:
            extension_variants = [
                raw for raw in variants if isinstance(raw, Mapping)
            ]
            if len(extension_variants) != 1:
                raise CampaignConfigError(
                    "Conditional extension lock must contain exactly one new variant"
                )
            extension_level = extension_variants[0].get("level")
            if not any(
                stage_materializer.canonical_json_dumps(extension_level)
                == stage_materializer.canonical_json_dumps(level)
                for level in declared_extension_levels
            ):
                raise CampaignConfigError(
                    "Conditional extension level was not predeclared in the plan"
                )
            expected_variant_levels = [extension_level]
            if resolved_stage.get("levels") != [*base_levels, extension_level]:
                raise CampaignConfigError(
                    "Conditional extension did not append exactly one planned level"
                )
            if prior_hypotheses != len(base_candidate_levels):
                raise CampaignConfigError(
                    "Conditional extension did not freeze the full base family"
                )
            if (
                family.get("planned_candidate_hypotheses")
                != len(base_candidate_levels) + 1
                or family.get("reserved_conditional_extensions") != 0
            ):
                raise CampaignConfigError(
                    "Conditional extension family arithmetic is invalid"
                )
    if coordinate is not None:
        coordinate_sizes = target_definition.get("coordinate_maximum_hypotheses")
        if (
            not isinstance(coordinate_sizes, Mapping)
            or coordinate not in coordinate_sizes
            or maximum_hypotheses != coordinate_sizes[coordinate]
        ):
            raise CampaignConfigError(
                "Stage lock coordinate family differs from the campaign plan"
            )

    _, current_source_sha256 = baseline_builder.embedded_sources()
    prior_anchors = [
        entry
        for entry in prior_entries
        if isinstance(entry, Mapping)
        and entry.get("role") in {"current_protocol_control", "stage_anchor"}
    ]
    if transition_kind == "conditional_boundary_extension":
        if len(prior_anchors) != 1 or (
            prior_anchors[0].get("experiment") != parent.get("experiment")
            or prior_anchors[0].get("expected_run_id") != parent.get("run_id")
        ):
            raise CampaignConfigError(
                "Conditional extension prior anchor differs from lock parent"
            )
    seen_experiments: set[str] = set()
    seen_slugs: set[str] = set()
    seen_run_ids: set[str] = set()
    seen_prior_levels: set[str] = set()
    for entry in prior_entries:
        if not isinstance(entry, Mapping):
            raise CampaignConfigError("Stage lock contains a non-object prior entry")
        if entry.get("stage") != lock.get("effective_stage"):
            raise CampaignConfigError("Prior entry belongs to another effective stage")
        experiment = str(entry.get("experiment", ""))
        kernel_slug = str(entry.get("kernel_slug", ""))
        run_id = str(entry.get("expected_run_id", ""))
        if (
            not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", experiment)
            or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", kernel_slug)
            or not run_id
            or experiment in seen_experiments
            or kernel_slug in seen_slugs
            or run_id in seen_run_ids
        ):
            raise CampaignConfigError("Conditional extension prior identity is invalid")
        expected_config = entry.get("expected_config")
        planned_overrides = entry.get("planned_overrides")
        if not isinstance(expected_config, Mapping) or not isinstance(
            planned_overrides, Mapping
        ):
            raise CampaignConfigError("Prior entry has no frozen config")
        prior_variant = {
            "experiment": experiment,
            "overrides": dict(planned_overrides),
            "loss_variant": entry.get("loss_variant", DEFAULT_LOSS_VARIANT),
        }
        validated_prior = variant_config(base, plan, prior_variant)
        if validated_prior != dict(expected_config) or entry.get(
            "expected_recipe_sha256"
        ) != team_builder.canonical_sha256(validated_prior):
            raise CampaignConfigError("Prior entry frozen recipe differs")
        _, _, prior_loss_sha256 = variant_loss(prior_variant)
        if entry.get("expected_loss_hook_sha256") != prior_loss_sha256:
            raise CampaignConfigError("Prior entry frozen loss differs")
        if entry.get("expected_source_sha256") != current_source_sha256:
            raise CampaignConfigError("Prior entry embedded source differs")
        for hash_key in (
            "expected_iid_predictions_sha256",
            "expected_completion_sha256",
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get(hash_key, ""))):
                raise CampaignConfigError(f"Prior entry {hash_key} is invalid")
        if not isinstance(entry.get("expected_notes"), str):
            raise CampaignConfigError("Prior entry has no exact frozen notes")
        if entry.get("hypothesis_family_size") != maximum_hypotheses:
            raise CampaignConfigError("Prior entry hypothesis family differs")
        role = entry.get("role")
        if role not in {"current_protocol_control", "stage_anchor", "candidate"} or (
            entry.get("is_hypothesis")
            is not (role not in {"current_protocol_control", "stage_anchor"})
        ):
            raise CampaignConfigError("Prior entry role/hypothesis flag differs")
        if axis_name is not None:
            prior_level = stage_materializer._effective_axis_value(
                validated_prior, axis_name
            )
            if (
                entry.get("axis") != axis_name
                or stage_materializer.canonical_json_dumps(entry.get("level"))
                != stage_materializer.canonical_json_dumps(prior_level)
                or entry.get("conditional_extension_levels")
                != declared_extension_levels
            ):
                raise CampaignConfigError(
                    "Prior entry axis/level/extension declaration differs"
                )
            prior_level_token = stage_materializer.canonical_json_dumps(prior_level)
            if prior_level_token in seen_prior_levels:
                raise CampaignConfigError("Prior entries repeat an axis level")
            seen_prior_levels.add(prior_level_token)
        seen_experiments.add(experiment)
        seen_slugs.add(kernel_slug)
        seen_run_ids.add(run_id)
    if transition_kind == "conditional_boundary_extension":
        if axis_name is not None and seen_prior_levels != {
            stage_materializer.canonical_json_dumps(level) for level in base_levels
        }:
            raise CampaignConfigError(
                "Conditional extension prior entries do not cover the base levels"
            )
        prior_anchor = prior_anchors[0]
        parent_checks = {
            "kernel_slug": parent.get("kernel_slug"),
            "expected_recipe_sha256": parent.get("recipe_sha256"),
            "expected_iid_predictions_sha256": parent.get(
                "iid_predictions_sha256"
            ),
            "expected_completion_sha256": parent.get("completion_sha256"),
            "expected_config": parent.get("resolved_config"),
            "expected_source_sha256": parent.get("code_bundle_sha256"),
            "loss_variant": parent_loss_variant,
            "expected_loss_hook_sha256": parent_loss_sha256,
            "expected_notes": parent.get("notes"),
        }
        if any(prior_anchor.get(key) != value for key, value in parent_checks.items()):
            raise CampaignConfigError(
                "Conditional extension prior anchor provenance differs from parent"
            )
        extension_source = lock.get("extension_source")
        matching_sources = [
            entry
            for entry in prior_entries
            if isinstance(entry, Mapping)
            and isinstance(extension_source, Mapping)
            and entry.get("experiment") == extension_source.get("experiment")
            and entry.get("expected_run_id") == extension_source.get("run_id")
        ]
        if (
            resolved_stage.get("conditional_extension_consumed") is not True
            or resolved_stage.get("conditional_extension_levels") != []
            or not isinstance(extension_source, Mapping)
            or len(matching_sources) != 1
        ):
            raise CampaignConfigError(
                "Conditional extension source/consumption provenance is invalid"
            )
        source_entry = matching_sources[0]
        source_checks = {
            "kernel_slug": extension_source.get("kernel_slug"),
            "expected_recipe_sha256": extension_source.get("recipe_sha256"),
            "expected_iid_predictions_sha256": extension_source.get(
                "iid_predictions_sha256"
            ),
            "expected_completion_sha256": extension_source.get(
                "completion_sha256"
            ),
            "expected_config": extension_source.get("resolved_config"),
            "expected_source_sha256": extension_source.get("code_bundle_sha256"),
            "loss_variant": extension_source.get("loss_variant"),
            "expected_loss_hook_sha256": extension_source.get(
                "loss_hook_sha256"
            ),
            "expected_notes": extension_source.get("notes"),
        }
        if any(source_entry.get(key) != value for key, value in source_checks.items()):
            raise CampaignConfigError(
                "Conditional extension selected-edge provenance differs"
            )
        if axis_name is not None:
            direction = resolved_stage.get("extension_direction")
            source_level = float(source_entry["level"])
            numeric_base_levels = [float(level) for level in base_levels]
            extension_level = float(variants[0]["level"])
            valid_direction = (
                direction == "lower"
                and source_level == min(numeric_base_levels)
                and extension_level < source_level
            ) or (
                direction == "higher"
                and source_level == max(numeric_base_levels)
                and extension_level > source_level
            )
            if not valid_direction:
                raise CampaignConfigError(
                    "Conditional extension does not continue its selected edge"
                )
    if not prior_entries:
        seen_experiments.add(str(parent["experiment"]))
        parent_slug = str(parent.get("kernel_slug", ""))
        if parent_slug:
            seen_slugs.add(parent_slug)
    allowed_losses = stage_loss_allowlist(plan, stage_name=target_stage)
    seen_variant_levels: set[str] = set()
    for raw in variants:
        if not isinstance(raw, Mapping):
            raise CampaignConfigError("Stage lock contains a non-object variant")
        if raw.get("role", "candidate") != "candidate":
            raise CampaignConfigError("Stage lock may contain only candidate variants")
        if "provenance_alias" in raw:
            raise CampaignConfigError("Locked candidates cannot use provenance aliases")
        _validate_variant_identity(
            raw,
            seen_experiments=seen_experiments,
            seen_slugs=seen_slugs,
            allowed_loss_variants=allowed_losses,
        )
        resolved_config = raw.get("resolved_config")
        if not isinstance(resolved_config, Mapping):
            raise CampaignConfigError("Locked variant has no resolved_config")
        expected_overrides = resolved_config_overrides(base, resolved_config)
        if raw.get("overrides") != expected_overrides:
            raise CampaignConfigError(
                f"Locked variant {raw.get('experiment')!r} overrides do not "
                "reproduce its resolved config"
            )
        validated = variant_config(base, plan, raw)
        if validated != dict(resolved_config):
            raise CampaignConfigError(
                f"Locked variant {raw.get('experiment')!r} resolved config differs"
            )
        recipe_sha = team_builder.canonical_sha256(validated)
        if raw.get("expected_recipe_sha256") != recipe_sha:
            raise CampaignConfigError(
                f"Locked variant {raw.get('experiment')!r} recipe SHA-256 differs"
            )
        if axis_name is not None:
            level = raw.get("level")
            level_token = stage_materializer.canonical_json_dumps(level)
            if raw.get("axis") != axis_name or level_token in seen_variant_levels:
                raise CampaignConfigError("Locked variant axis/level is invalid")
            try:
                expected_coordinate_overrides = stage_materializer._level_overrides(
                    axis_name,
                    level,
                    effective_batch_recipes=effective_batch_recipes,
                )
            except stage_materializer.StageMaterializationError as error:
                raise CampaignConfigError(str(error)) from error
            expected_from_parent = deepcopy(validated_parent)
            expected_from_parent.update(deepcopy(expected_coordinate_overrides))
            if (
                raw.get("coordinate_overrides") != expected_coordinate_overrides
                or dict(resolved_config) != expected_from_parent
            ):
                raise CampaignConfigError(
                    "Locked variant changes more than its declared coordinate"
                )
            seen_variant_levels.add(level_token)
    if expected_variant_levels is not None and seen_variant_levels != {
        stage_materializer.canonical_json_dumps(level)
        for level in expected_variant_levels
    }:
        raise CampaignConfigError(
            "Stage lock variants do not cover exactly the planned coordinate levels"
        )
    return lock


def stage_lock_parent_provenance(lock: Mapping[str, Any]) -> dict[str, str]:
    parent = lock["parent"]
    result = {
        "experiment": str(parent["experiment"]),
        "run_id": str(parent["run_id"]),
        "recipe_sha256": str(parent["recipe_sha256"]),
        "iid_predictions_sha256": str(parent["iid_predictions_sha256"]),
    }
    for key in ("loss_variant", "loss_hook_sha256"):
        if parent.get(key) is not None:
            result[key] = str(parent[key])
    return result


def _adaptive_lock_module():
    """Import schema-v2 support lazily; its validator imports this module."""
    import materialize_minilm_5ep_sft_loss_confirmation as adaptive

    return adaptive


def load_campaign_lock(
    path: Path,
    *,
    plan: Mapping[str, Any],
    base_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Dispatch strict schema-v1/v2 loading without a payload-selected authority."""
    try:
        resolved_path = path.resolve(strict=True)
        probe = json.loads(resolved_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignConfigError(f"Invalid campaign lock {path}: {error}") from error
    if not isinstance(probe, Mapping):
        raise CampaignConfigError("Campaign lock must be a JSON object")
    schema_version = probe.get("schema_version")
    if schema_version == 1:
        return load_stage_lock(
            resolved_path,
            plan=plan,
            base_config=base_config,
        )
    if schema_version != 2:
        raise CampaignConfigError(
            f"Unsupported campaign-lock schema_version {schema_version!r}"
        )
    if path.is_symlink():
        raise CampaignConfigError("Schema-v2 campaign lock must not be a symlink")

    adaptive = _adaptive_lock_module()
    manifest_path = adaptive.trusted_provenance_manifest_path(resolved_path)
    expected_archive = adaptive.trusted_provenance_archive_dir(
        resolved_path
    ).resolve(strict=False)
    if manifest_path.is_symlink():
        raise CampaignConfigError(
            "Schema-v2 trusted-provenance sidecar must not be a symlink"
        )
    try:
        trusted = adaptive.load_trusted_provenance(manifest_path, plan=plan)
        if Path(str(trusted["archive_dir"])) != expected_archive:
            raise CampaignConfigError(
                "Schema-v2 lock was relocated or its archive path differs"
            )
        lock = adaptive.read_lock(
            resolved_path,
            plan=plan,
            trusted_provenance=trusted,
        )
    except adaptive.AdaptiveMaterializationError as error:
        raise CampaignConfigError(
            f"Invalid schema-v2 campaign lock {resolved_path}: {error}"
        ) from error
    _, current_source_sha256 = baseline_builder.embedded_sources()
    if lock.get("expected_source_sha256") != current_source_sha256:
        raise CampaignConfigError(
            "Schema-v2 lock execution source differs from the current embedded "
            "source bundle"
        )
    for variant in lock["resolved_stage"]["variants"]:
        if variant.get("expected_source_sha256") != current_source_sha256:
            raise CampaignConfigError(
                f"Schema-v2 variant {variant.get('experiment')!r} source differs "
                "from the current embedded source bundle"
            )
    return lock


def campaign_variant_parent_provenance(
    lock: Mapping[str, Any],
    variant: Mapping[str, Any],
) -> dict[str, Any]:
    """Return exact frozen lineage for one schema-v1/v2 executable variant."""
    if lock.get("schema_version") == 1:
        return stage_lock_parent_provenance(lock)
    origin_by_id = {
        str(origin["origin_id"]): origin
        for origin in lock.get("origins", [])
        if isinstance(origin, Mapping) and origin.get("origin_id") is not None
    }
    lineage = []
    origin_ids = variant.get("origin_ids")
    if not isinstance(origin_ids, list) or not origin_ids:
        raise CampaignConfigError("Schema-v2 variant has no origin lineage")
    for origin_id in origin_ids:
        origin = origin_by_id.get(str(origin_id))
        if origin is None:
            raise CampaignConfigError(
                f"Schema-v2 variant references unknown origin {origin_id!r}"
            )
        lineage.append(
            {
                "origin_id": str(origin["origin_id"]),
                "experiment": str(origin["experiment"]),
                "run_id": str(origin["run_id"]),
                "recipe_sha256": str(origin["recipe_sha256"]),
                "recipe_family_sha256": str(origin["recipe_family_sha256"]),
                "loss_variant": str(origin["loss_variant"]),
                "loss_hook_sha256": str(origin["loss_hook_sha256"]),
                "expected_source_sha256": str(
                    origin["expected_source_sha256"]
                ),
                "iid_predictions_sha256": str(
                    origin["iid_predictions_sha256"]
                ),
                "completion_sha256": str(origin["completion_sha256"]),
                "completion_notes_sha256": str(
                    origin["completion_notes_sha256"]
                ),
            }
        )
    return {"origin_lineage": lineage}


def normalized_campaign_execution_contract(
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    *,
    base_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize executable expectations shared by build, launch and summary."""
    base = (
        dict(base_config)
        if base_config is not None
        else cross_builder.load_training_config(BASE_CONFIG_PATH)
    )
    schema_version = lock.get("schema_version")
    if schema_version not in {1, 2}:
        raise CampaignConfigError("Unsupported normalized campaign-lock schema")
    status = (
        "runnable" if schema_version == 1 else str(lock.get("execution_status"))
    )
    if status not in {"runnable", "skipped"}:
        raise CampaignConfigError("Campaign-lock execution status is invalid")
    resolved = lock.get("resolved_stage")
    if not isinstance(resolved, Mapping):
        raise CampaignConfigError("Campaign lock has no resolved stage")
    raw_variants = resolved.get("variants")
    if not isinstance(raw_variants, list):
        raise CampaignConfigError("Campaign lock variants are malformed")
    if (status == "runnable") != bool(raw_variants):
        raise CampaignConfigError("Campaign lock status/variants disagree")
    effective_stage = stage_lock_effective_stage(lock)
    target_stage = str(lock.get("target_stage", "")).strip()
    allowed_losses = stage_loss_allowlist(plan, stage_name=target_stage)
    _, embedded_source_sha256 = baseline_builder.embedded_sources()
    if (
        schema_version == 2
        and lock.get("expected_source_sha256") != embedded_source_sha256
    ):
        raise CampaignConfigError(
            "Schema-v2 lock execution source differs from the current embedded "
            "source bundle"
        )
    expected_execution_source = embedded_source_sha256
    family = (
        lock.get("family")
        if schema_version == 2
        else resolved.get("family")
    )
    if not isinstance(family, Mapping):
        raise CampaignConfigError("Campaign lock has no hypothesis family")
    family_id = str(
        family.get("family_id")
        or f"{effective_stage}__schema_v1_family"
    )
    maximum_hypotheses = family.get("maximum_hypotheses")
    if maximum_hypotheses is not None and (
        isinstance(maximum_hypotheses, bool)
        or not isinstance(maximum_hypotheses, int)
        or maximum_hypotheses < 1
    ):
        raise CampaignConfigError("Campaign lock maximum hypotheses is invalid")

    seen_experiments: set[str] = set()
    seen_slugs: set[str] = set()
    variants = []
    for raw in raw_variants:
        if not isinstance(raw, Mapping):
            raise CampaignConfigError("Campaign lock contains a non-object variant")
        _validate_variant_identity(
            raw,
            seen_experiments=seen_experiments,
            seen_slugs=seen_slugs,
            allowed_loss_variants=allowed_losses,
        )
        config = variant_config(base, plan, raw)
        if raw.get("resolved_config") not in (None, config):
            raise CampaignConfigError("Locked resolved config differs after validation")
        recipe_sha256 = team_builder.canonical_sha256(config)
        if raw.get("expected_recipe_sha256") not in (None, recipe_sha256):
            raise CampaignConfigError("Locked recipe SHA differs after validation")
        loss_variant, _, loss_hook_sha256 = variant_loss(raw)
        if raw.get("expected_loss_hook_sha256") not in (None, loss_hook_sha256):
            raise CampaignConfigError("Locked loss-hook SHA differs after validation")
        source_sha256 = str(
            raw.get("expected_source_sha256", expected_execution_source)
        )
        if source_sha256 != expected_execution_source:
            raise CampaignConfigError("Locked execution source SHA differs")
        role = str(raw.get("role", "candidate"))
        is_hypothesis = bool(
            raw.get(
                "is_hypothesis",
                role not in {"current_protocol_control", "stage_anchor"},
            )
        )
        notes = _variant_notes(
            str(plan["campaign"]),
            effective_stage,
            raw,
            config,
            stage_lock=lock,
        )
        variants.append(
            {
                "stage": effective_stage,
                "experiment": str(raw["experiment"]),
                "kernel_slug": str(raw["kernel_slug"]),
                "title": str(raw["title"]),
                "role": role,
                "is_hypothesis": is_hypothesis,
                "family_id": family_id,
                "hypothesis_family_size": maximum_hypotheses,
                "expected_config": config,
                "planned_overrides": dict(raw["overrides"]),
                "recipe_sha256": recipe_sha256,
                "source_sha256": source_sha256,
                "loss_variant": loss_variant,
                "loss_hook_sha256": loss_hook_sha256,
                "expected_notes": notes,
                "parent_provenance": campaign_variant_parent_provenance(
                    lock, raw
                ),
                "origin_ids": deepcopy(raw.get("origin_ids", [])),
                "variant": raw,
            }
        )
    return {
        "schema_version": int(schema_version),
        "lock_payload_sha256": str(lock["lock_payload_sha256"]),
        "execution_status": status,
        "mode": lock.get("mode"),
        "source_stage": str(lock["source_stage"]),
        "target_stage": target_stage,
        "effective_stage": effective_stage,
        "accepted_stage_filters": sorted({target_stage, effective_stage}),
        "family": deepcopy(dict(family)),
        "family_id": family_id,
        "hypothesis_family_size": maximum_hypotheses,
        "prerequisites": deepcopy(lock.get("prerequisites", [])),
        "origins": deepcopy(lock.get("origins", [])),
        "budget": deepcopy(lock.get("budget")),
        "variants": variants,
    }


def stage_lock_effective_stage(lock: Mapping[str, Any]) -> str:
    return str(lock["effective_stage"])


def _variant_notes(
    campaign: str,
    stage: str,
    variant: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    stage_lock: Mapping[str, Any] | None = None,
) -> str:
    if stage_lock is not None and stage_lock.get("schema_version") == 2:
        adaptive = _adaptive_lock_module()
        return adaptive.expected_variant_notes(stage_lock, variant)
    effective_batch = (
        int(config["batch_size"])
        * 2
        * int(config["gradient_accumulation"])
    )
    details = {
        "campaign": campaign,
        "stage": stage,
        "role": variant.get("role", "candidate"),
        "loss_variant": variant.get("loss_variant", DEFAULT_LOSS_VARIANT),
        "epochs": config["epochs"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "warmup_ratio": config["warmup_ratio"],
        "label_smoothing": config["label_smoothing"],
        "max_grad_norm": config["max_grad_norm"],
        "batch_size_per_gpu": config["batch_size"],
        "gradient_accumulation": config["gradient_accumulation"],
        "effective_batch": effective_batch,
        "model_load_kwargs": config.get("model_load_kwargs", {}),
        "seed": config["seed"],
    }
    if stage_lock is not None:
        details["stage_lock_payload_sha256"] = stage_lock[
            "lock_payload_sha256"
        ]
        details["stage_parent"] = stage_lock_parent_provenance(stage_lock)
    return json.dumps(
        details,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _frozen_recipe_source(config: Mapping[str, Any]) -> str:
    recipe_hash = team_builder.canonical_sha256(config)
    return dedent(
        f"""
        CANONICAL_TRAIN_CONFIG = {dict(config)!r}
        EXPECTED_TRAIN_RECIPE_SHA256 = {recipe_hash!r}

        def canonical_json_sha256(value):
            payload = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            return hashlib.sha256(payload).hexdigest()

        actual_recipe_hash = canonical_json_sha256(CANONICAL_TRAIN_CONFIG)
        if actual_recipe_hash != EXPECTED_TRAIN_RECIPE_SHA256:
            raise RuntimeError(
                "Frozen SFT variant recipe was edited. Regenerate this notebook "
                "from the committed campaign plan."
            )
        TRAIN_CONFIG = dict(CANONICAL_TRAIN_CONFIG)
        if INITIAL_MODEL_PATH is not None:
            TRAIN_CONFIG["model"] = str(INITIAL_MODEL_PATH)
        RUNTIME_CONFIG_PATH.write_text(
            json.dumps(TRAIN_CONFIG, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({{
            "frozen_recipe_sha256": EXPECTED_TRAIN_RECIPE_SHA256,
            "training_config": TRAIN_CONFIG,
        }}, ensure_ascii=False, indent=2))
        """
    ).strip()


def build_variant_notebook(
    *,
    dataset: Mapping[str, Any],
    base_config: Mapping[str, Any],
    plan: Mapping[str, Any],
    stage: str,
    variant: Mapping[str, Any],
    stage_lock: Mapping[str, Any] | None = None,
) -> nbf.NotebookNode:
    config = variant_config(base_config, plan, variant)
    loss_variant, loss_hook_source, loss_hook_sha256 = variant_loss(variant)
    experiment = str(variant["experiment"])
    campaign = str(plan["campaign"])
    notes = _variant_notes(
        campaign,
        stage,
        variant,
        config,
        stage_lock=stage_lock,
    )
    notebook = team_builder.build_team_notebook(
        dict(dataset),
        dict(base_config),
        experiment_label=experiment,
        experiment_sheet="sft_exps",
        experiment_notes=notes,
    )

    recipe_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code"
        and "frozen-recipe" in cell.metadata.get("tags", [])
    ]
    if len(recipe_cells) != 1:
        raise RuntimeError("Team template has no unique frozen-recipe cell")
    recipe_cells[0].source = _frozen_recipe_source(config)

    for cell in notebook.cells:
        tags = list(cell.metadata.get("tags", []))
        if "experiment-routing" in tags:
            cell.metadata["tags"] = ["frozen", "experiment-routing"]
            continue
        if "team-editable" not in tags:
            continue
        hook_tag = next(
            (tag for tag in tags if tag in {"data-hook", "loss-hook"}),
            "fixed-hook",
        )
        cell.metadata["tags"] = ["frozen", f"sft-fixed-{hook_tag}"]
        if cell.cell_type == "markdown":
            cell.source = cell.source.replace("✏️ EDIT", "🔒 FIXED")
            cell.source += (
                "\n\nThis hook is fixed by the SFT campaign plan for this "
                "experiment."
            )

    routing_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code"
        and "experiment-routing" in cell.metadata.get("tags", [])
    ]
    if len(routing_cells) != 1:
        raise RuntimeError("SFT notebook must contain one fixed routing cell")
    routing_cells[0].source = dedent(
        f"""
        EXPECTED_EXPERIMENT_LABEL = {experiment!r}
        EXPECTED_EXPERIMENT_SHEET = "sft_exps"
        EXPECTED_EXPERIMENT_NOTES = {notes!r}

        EXPERIMENT_LABEL = EXPECTED_EXPERIMENT_LABEL
        EXPERIMENT_SHEET = EXPECTED_EXPERIMENT_SHEET
        EXPERIMENT_NOTES = EXPECTED_EXPERIMENT_NOTES
        if (
            EXPERIMENT_LABEL != EXPECTED_EXPERIMENT_LABEL
            or EXPERIMENT_SHEET != EXPECTED_EXPERIMENT_SHEET
            or EXPERIMENT_NOTES != EXPECTED_EXPERIMENT_NOTES
        ):
            raise RuntimeError("Frozen SFT experiment routing was edited")
        """
    ).strip()

    data_hook_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code"
        and "sft-fixed-data-hook" in cell.metadata.get("tags", [])
    ]
    loss_hook_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code"
        and "sft-fixed-loss-hook" in cell.metadata.get("tags", [])
    ]
    if len(data_hook_cells) != 1 or len(loss_hook_cells) != 1:
        raise RuntimeError("SFT notebook must contain one fixed data and loss hook")
    data_hook_cells[0].source = dedent(
        """
        SFT_FIXED_DATA_HOOK_ID = "human_identity_v1"


        def build_train_data(human_train_pairs, human_items, input_root):
            return human_train_pairs.copy(), human_items.copy()
        """
    ).strip()
    loss_hook_cells[0].source = dedent(
        f"""
        SFT_FIXED_LOSS_VARIANT = {loss_variant!r}
        SFT_FIXED_LOSS_HOOK_SOURCE = {loss_hook_source!r}
        EXPECTED_SFT_LOSS_HOOK_SHA256 = {loss_hook_sha256!r}
        fixed_loss_payload_hash = hashlib.sha256(
            SFT_FIXED_LOSS_HOOK_SOURCE.encode("utf-8")
        ).hexdigest()
        if fixed_loss_payload_hash != EXPECTED_SFT_LOSS_HOOK_SHA256:
            raise RuntimeError("Frozen SFT loss hook payload was edited")
        fixed_loss_hook_path = PROJECT_ROOT / "team_loss_hook.py"
        fixed_loss_hook_path.write_text(
            SFT_FIXED_LOSS_HOOK_SOURCE,
            encoding="utf-8",
        )
        """
    ).strip()

    data_guard_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code"
        and "data-guard" in cell.metadata.get("tags", [])
    ]
    if len(data_guard_cells) != 1:
        raise RuntimeError("SFT notebook must contain one data guard")
    data_guard_needle = "train_pairs, train_items = hook_result\n"
    data_guard_replacement = dedent(
        """
        train_pairs, train_items = hook_result
        if SFT_FIXED_DATA_HOOK_ID != "human_identity_v1":
            raise RuntimeError("Frozen human identity data hook was edited")
        if not train_pairs.reset_index(drop=True).equals(
            human_train_pairs.reset_index(drop=True)
        ):
            raise RuntimeError("SFT hyperparameter run changed frozen human train pairs")
        if not train_items.reset_index(drop=True).equals(
            human_items.reset_index(drop=True)
        ):
            raise RuntimeError("SFT hyperparameter run changed frozen human items")
        """
    ).strip() + "\n"
    if data_guard_needle not in data_guard_cells[0].source:
        raise RuntimeError("Could not strengthen the frozen SFT data guard")
    data_guard_cells[0].source = data_guard_cells[0].source.replace(
        data_guard_needle,
        data_guard_replacement,
        1,
    )

    protocol_guard_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code"
        and "protocol-guard" in cell.metadata.get("tags", [])
    ]
    if len(protocol_guard_cells) != 1:
        raise RuntimeError("SFT notebook must contain one final protocol guard")
    loss_hash_needle = "LOSS_HOOK_SHA256 = file_sha256(LOSS_HOOK_PATH)\n"
    loss_hash_replacement = dedent(
        """
        LOSS_HOOK_SHA256 = file_sha256(LOSS_HOOK_PATH)
        if LOSS_HOOK_SHA256 != EXPECTED_SFT_LOSS_HOOK_SHA256:
            raise RuntimeError("Materialized SFT loss hook differs from the frozen payload")
        """
    ).strip() + "\n"
    if loss_hash_needle not in protocol_guard_cells[0].source:
        raise RuntimeError("Could not strengthen the frozen SFT loss guard")
    protocol_guard_cells[0].source = protocol_guard_cells[0].source.replace(
        loss_hash_needle,
        loss_hash_replacement,
        1,
    )
    protocol_guard_cells[0].source += dedent(
        """

        if (
            EXPERIMENT_LABEL != EXPECTED_EXPERIMENT_LABEL
            or EXPERIMENT_SHEET != "sft_exps"
            or EXPERIMENT_NOTES != EXPECTED_EXPERIMENT_NOTES
            or EXPERIMENT_GROUP != "sft"
        ):
            raise RuntimeError("SFT routing changed after initialization")
        """
    )

    notebook.cells[0].source = dedent(
        f"""
        # MiniLM 5ep SFT hyperparameter ablation: `{experiment}`

        Campaign `{campaign}`, stage `{stage}`. Every run starts from the same
        verified five-epoch LLM-pretraining checkpoint, uses the same frozen human
        train, planned `{loss_variant}` loss and identical IID/hard/OOD validation
        pairs.

        The complete immutable recipe for this variant is embedded below. Results
        are compared with baseline `{team_builder.SIGNIFICANCE_BASELINE_RUN_ID}` and
        routed to `sft_exps`.
        """
    ).strip()

    metadata = notebook.metadata["product_matching_training"]
    metadata.update(
        {
            "template": "minilm_5ep_sft_hparam_ablation_v1",
            "experiment": experiment,
            "experiment_group": "sft",
            "campaign": campaign,
            "campaign_stage": stage,
            "frozen_recipe_sha256": team_builder.canonical_sha256(config),
            "editable_cells": [],
            "fixed_data_hook": "human_identity_v1",
            "loss_variant": loss_variant,
            "fixed_loss_hook_sha256": loss_hook_sha256,
            "sft_overrides": dict(variant["overrides"]),
        }
    )
    if stage_lock is not None:
        lock_metadata = {
            "schema_version": stage_lock["schema_version"],
            "lock_payload_sha256": stage_lock["lock_payload_sha256"],
            "source_plan_sha256": stage_lock["source_plan_sha256"],
            "source_stage": stage_lock["source_stage"],
            "target_stage": stage_lock["target_stage"],
            "effective_stage": stage_lock_effective_stage(stage_lock),
            "coordinate": stage_lock.get("coordinate"),
            "parent": campaign_variant_parent_provenance(stage_lock, variant),
        }
        if stage_lock.get("schema_version") == 2:
            lock_metadata.update(
                {
                    "mode": stage_lock["mode"],
                    "execution_status": stage_lock["execution_status"],
                    "family": deepcopy(dict(stage_lock["family"])),
                }
            )
        metadata["stage_lock"] = lock_metadata
    metadata.pop("run_configurable_cell", None)
    nbf.validate(notebook)
    return notebook


def output_path(output_dir: Path, variant: Mapping[str, Any]) -> Path:
    return output_dir / f"{variant['experiment']}_2xt4.ipynb"


def build_campaign(
    *,
    plan_path: Path = DEFAULT_PLAN,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    source_dir: Path = baseline_builder.DEFAULT_SOURCE_DIR,
    stage_name: str | None = None,
    only: set[str] | None = None,
    stage_lock_path: Path | None = None,
) -> list[dict[str, Any]]:
    plan = load_plan(plan_path)
    base_config = cross_builder.load_training_config(BASE_CONFIG_PATH)
    team_builder.assert_frozen_recipe(base_config)
    stage_lock: Mapping[str, Any] | None = None
    execution_contract: Mapping[str, Any] | None = None
    normalized_by_experiment: dict[str, Mapping[str, Any]] = {}
    selected_variants: Iterable[tuple[str, Mapping[str, Any]]]
    if stage_lock_path is not None:
        stage_lock = load_campaign_lock(
            stage_lock_path,
            plan=plan,
            base_config=base_config,
        )
        execution_contract = normalized_campaign_execution_contract(
            plan,
            stage_lock,
            base_config=base_config,
        )
        effective_stage = str(execution_contract["effective_stage"])
        accepted_stage_filters = set(execution_contract["accepted_stage_filters"])
        if stage_name is not None and stage_name not in accepted_stage_filters:
            raise CampaignConfigError(
                f"--stage {stage_name!r} differs from locked stage "
                f"{effective_stage!r}"
            )
        normalized_by_experiment = {
            str(entry["experiment"]): entry
            for entry in execution_contract["variants"]
        }
        if execution_contract["execution_status"] == "skipped":
            if only:
                raise CampaignConfigError(
                    "A skipped adaptive receipt has no runnable variants"
                )
            return []
        selected_variants = (
            (effective_stage, entry["variant"])
            for entry in execution_contract["variants"]
        )
    else:
        selected_variants = ready_variants(plan, stage_name=stage_name)
    dataset = baseline_builder.load_manifest(source_dir, team_builder.DATASET_OWNER)
    built: list[dict[str, Any]] = []
    for stage, variant in selected_variants:
        experiment = str(variant["experiment"])
        if only and experiment not in only:
            continue
        notebook = build_variant_notebook(
            dataset=dataset,
            base_config=base_config,
            plan=plan,
            stage=stage,
            variant=variant,
            stage_lock=stage_lock,
        )
        expected_config = (
            deepcopy(normalized_by_experiment[experiment]["expected_config"])
            if stage_lock is not None
            else variant_config(base_config, plan, variant)
        )
        training_metadata = notebook.metadata["product_matching_training"]
        if stage_lock is not None:
            normalized = normalized_by_experiment[experiment]
            expected_metadata = {
                "frozen_recipe_sha256": normalized["recipe_sha256"],
                "source_sha256": normalized["source_sha256"],
                "loss_variant": normalized["loss_variant"],
                "fixed_loss_hook_sha256": normalized["loss_hook_sha256"],
            }
            mismatches = {
                key: {
                    "actual": training_metadata.get(key),
                    "expected": expected,
                }
                for key, expected in expected_metadata.items()
                if training_metadata.get(key) != expected
            }
            if mismatches:
                raise CampaignConfigError(
                    "Built notebook metadata differs from the immutable campaign "
                    f"contract: {mismatches}"
                )
        destination = output_path(output_dir, variant)
        destination.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(notebook, destination)
        built.append(
            {
                "stage": stage,
                "experiment": experiment,
                "kernel_slug": str(variant["kernel_slug"]),
                "title": str(variant["title"]),
                "notebook": str(destination),
                "recipe_sha256": notebook.metadata[
                    "product_matching_training"
                ]["frozen_recipe_sha256"],
                "source_sha256": notebook.metadata[
                    "product_matching_training"
                ]["source_sha256"],
                "loss_variant": notebook.metadata[
                    "product_matching_training"
                ]["loss_variant"],
                "loss_hook_sha256": notebook.metadata[
                    "product_matching_training"
                ]["fixed_loss_hook_sha256"],
                "expected_config": expected_config,
                "expected_notes": (
                    normalized_by_experiment[experiment]["expected_notes"]
                    if stage_lock is not None
                    else _variant_notes(
                        str(plan["campaign"]),
                        stage,
                        variant,
                        variant_config(base_config, plan, variant),
                        stage_lock=None,
                    )
                ),
            }
        )
        if stage_lock is not None:
            built[-1].update(
                {
                    "stage_lock_payload_sha256": stage_lock[
                        "lock_payload_sha256"
                    ],
                    "parent_provenance": normalized_by_experiment[experiment][
                        "parent_provenance"
                    ],
                    "family_id": normalized_by_experiment[experiment][
                        "family_id"
                    ],
                    "hypothesis_family_size": normalized_by_experiment[
                        experiment
                    ]["hypothesis_family_size"],
                    "role": normalized_by_experiment[experiment]["role"],
                    "is_hypothesis": normalized_by_experiment[experiment][
                        "is_hypothesis"
                    ],
                }
            )
    if only:
        missing = only - {entry["experiment"] for entry in built}
        if missing:
            raise CampaignConfigError(
                f"Requested variants are not ready or do not exist: {sorted(missing)}"
            )
    if not built:
        raise CampaignConfigError("No ready variants matched the requested filters")
    return built


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-dir", type=Path, default=baseline_builder.DEFAULT_SOURCE_DIR)
    parser.add_argument("--stage")
    parser.add_argument("--stage-lock", type=Path)
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()
    stage_lock_path = args.stage_lock
    if stage_lock_path is not None and not stage_lock_path.is_absolute():
        stage_lock_path = ROOT / stage_lock_path
    built = build_campaign(
        plan_path=args.plan,
        output_dir=args.output_dir,
        source_dir=args.source_dir,
        stage_name=args.stage,
        only=set(args.only) or None,
        stage_lock_path=stage_lock_path,
    )
    print(json.dumps(built, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
