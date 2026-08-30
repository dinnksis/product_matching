"""Export executable product-scoped rules from semantic all-pair statistics.

The semantic analysis intentionally treats multi-difference pairs as
correlational evidence.  This exporter therefore keeps that provenance in the
generation tier and applies a second, product-type-specific cross-split check
before a rule becomes executable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from item_pipeline.normalization import normalize_text
from scripts.analyze_semantic_atomic_rules import build_prototypes
from scripts.export_statistical_generation_rules import (
    ALLOWED_ANCHOR_CONTEXT_KEYS,
    ATTRIBUTE_KEY_OVERRIDES,
    CATEGORY_ATTRIBUTE_KEY_OVERRIDES,
    CATEGORY_DEPENDENT_ATTRIBUTE_PATTERNS,
    DEFAULT_ALLOWED_ANCHOR_CONTEXT_KEYS,
    DEPENDENT_ATTRIBUTE_PATTERNS,
    FORBIDDEN_ATTRIBUTE_RE,
    FORBIDDEN_CONCEPT_RE,
    FORBIDDEN_TARGET_VALUE_PATTERNS,
    SEMANTIC_REVIEW_CATEGORY_CONCEPTS,
    SEMANTIC_REVIEW_CONCEPTS,
    TARGET_VALUE_PATTERNS,
    atomic_write_bytes,
    explicit_product_type,
    generation_anchor_hint,
    load_profile_capacity_policy,
    product_type_alias_map,
    resolved_profile_policy,
    sha256,
)


DEFAULT_REPORT = (
    ROOT / "reports" / "semantic_atomic_rule_statistics_all_pairs_20260827"
)
DEFAULT_OCCURRENCES = (
    ROOT
    / "reports"
    / "atomic_rule_statistics_semantic_snapshot_20260826"
    / "atomic_occurrences.parquet"
)
DEFAULT_PAIR_INPUTS = (
    ROOT / "data" / "qwen_atomic_differences_v2_full_train" / "pilot_inputs.parquet"
)
DEFAULT_POLICY = (
    ROOT
    / "configs"
    / "generation_rule_catalog_statistical_v1"
    / "profile_capacity_policy_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "configs"
    / "generation_rule_catalog_statistical_v1"
    / "semantic_all_pairs_cross_split_p80_support5_scoped_compact_v2.json"
)

CATALOG_VERSION = "semantic_all_pairs_cross_split_p80_support5_scoped_compact_v2"
TIER_TEMPLATE = (
    "SEMANTIC_ALL_PAIRS_LABEL{label}_CROSS_SPLIT_SUPPORT5_P80_"
    "SCOPED_COMPACT_V2_EXPERIMENTAL_CORRELATIONAL"
)
PLACEHOLDER_PRODUCT_TYPES = {
    "-",
    "--",
    "—",
    "–",
    "n/a",
    "na",
    "none",
    "unknown",
    "нет",
    "не указан",
    "не указана",
    "не указано",
    "не указаны",
    "не определен",
    "не определена",
    "не определено",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--occurrences", type=Path, default=DEFAULT_OCCURRENCES)
    parser.add_argument("--pair-inputs", type=Path, default=DEFAULT_PAIR_INPUTS)
    parser.add_argument("--profile-capacity-policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--minimum-weighted-support", type=float, default=5.0)
    parser.add_argument("--minimum-probability", type=float, default=0.8)
    parser.add_argument("--minimum-product-type-pairs", type=int, default=2)
    parser.add_argument("--minimum-attribute-key-support", type=int, default=2)
    parser.add_argument("--maximum-source-examples", type=int, default=4)
    return parser.parse_args()


def _stable_id(parts: list[Any], *, prefix: str) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    suffix = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}{suffix}"


def _dominant_attribute_keys(evidence: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for side in ("a", "b"):
        frame = evidence[evidence[f"source_{side}"].eq("attribute")][
            ["center_index", f"raw_attribute_{side}"]
        ].rename(columns={f"raw_attribute_{side}": "attribute_key"})
        rows.append(frame)
    attributes = pd.concat(rows, ignore_index=True)
    attributes["attribute_key"] = attributes["attribute_key"].fillna("").map(
        lambda value: " ".join(str(value).split())
    )
    attributes = attributes[
        attributes["attribute_key"].ne("")
        & ~attributes["attribute_key"].str.contains(FORBIDDEN_ATTRIBUTE_RE)
    ]
    return (
        attributes.groupby(["center_index", "attribute_key"], observed=True)
        .size()
        .rename("attribute_key_support")
        .reset_index()
        .sort_values(
            ["center_index", "attribute_key_support", "attribute_key"],
            ascending=[True, False, True],
        )
        .drop_duplicates("center_index")
    )


def _canonical_attribute_key(row: Any) -> tuple[str, str]:
    category = str(row.category)
    concept = str(row.canonical_concept)
    if (category, concept) in CATEGORY_ATTRIBUTE_KEY_OVERRIDES:
        return CATEGORY_ATTRIBUTE_KEY_OVERRIDES[(category, concept)], "category_override"
    if concept in ATTRIBUTE_KEY_OVERRIDES:
        return ATTRIBUTE_KEY_OVERRIDES[concept], "concept_override"
    return str(row.attribute_key or ""), "semantic_neighbour_dominant_source_key"


def _pair_product_types(
    pair_inputs: pd.DataFrame,
    required_pair_ids: set[str],
    aliases: dict[tuple[str, str], str],
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    frame = pair_inputs[pair_inputs["pair_id"].astype(str).isin(required_pair_ids)]
    for source in frame.itertuples(index=False):
        category = str(source.category)
        values = {
            explicit_product_type(source.attributes_a_json, source.title_a),
            explicit_product_type(source.attributes_b_json, source.title_b),
        }
        values.discard("")
        for value in values:
            canonical = aliases.get((category, normalize_text(value)), value)
            normalized = normalize_text(canonical)
            if valid_product_type(normalized):
                rows.append(
                    {
                        "pair_id": str(source.pair_id),
                        "category": category,
                        "product_type": str(canonical),
                        "normalized_product_type": normalized,
                    }
                )
    return pd.DataFrame(
        rows,
        columns=["pair_id", "category", "product_type", "normalized_product_type"],
    ).drop_duplicates(
        ["pair_id", "category", "normalized_product_type"]
    )


def valid_product_type(value: Any) -> bool:
    """Reject missing-value markers and values with no alphabetic product name."""

    normalized = normalize_text(value)
    return bool(
        normalized
        and normalized not in PLACEHOLDER_PRODUCT_TYPES
        and re.search(r"[a-zа-яё]", normalized, flags=re.IGNORECASE)
    )


def _profile_statistics(
    evidence: pd.DataFrame,
    pair_types: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    minimum_support: int,
    minimum_probability: float,
) -> pd.DataFrame:
    pair_evidence = evidence[
        ["center_index", "pair_id", "category", "human_label", "split"]
    ].drop_duplicates(["center_index", "pair_id"])
    pair_evidence = pair_evidence.merge(pair_types, on=["pair_id", "category"])
    pair_evidence = pair_evidence.drop_duplicates(
        ["center_index", "pair_id", "normalized_product_type"]
    )
    target_by_center = selected.set_index("center_index")["target_label"].to_dict()
    grouped = (
        pair_evidence.groupby(
            ["center_index", "normalized_product_type"], observed=True
        )
        .agg(
            product_type=("product_type", "first"),
            profile_pair_support=("pair_id", "size"),
            profile_label0=("human_label", lambda values: int(values.eq(0).sum())),
            profile_label1=("human_label", lambda values: int(values.eq(1).sum())),
        )
        .reset_index()
    )
    grouped["target_label"] = grouped["center_index"].map(target_by_center).astype(int)
    grouped["profile_target_support"] = [
        int(label0 if target == 0 else label1)
        for target, label0, label1 in zip(
            grouped["target_label"], grouped["profile_label0"], grouped["profile_label1"]
        )
    ]
    grouped["profile_target_probability"] = (
        grouped["profile_target_support"] / grouped["profile_pair_support"]
    )

    split = (
        pair_evidence.groupby(
            ["center_index", "normalized_product_type", "split"], observed=True
        )
        .agg(
            support=("pair_id", "size"),
            label0=("human_label", lambda values: int(values.eq(0).sum())),
            label1=("human_label", lambda values: int(values.eq(1).sum())),
        )
        .reset_index()
    )
    split["target_label"] = split["center_index"].map(target_by_center).astype(int)
    split["target_support"] = [
        int(label0 if target == 0 else label1)
        for target, label0, label1 in zip(
            split["target_label"], split["label0"], split["label1"]
        )
    ]
    split["target_probability"] = split["target_support"] / split["support"]
    split_lookup = {
        (int(row.center_index), str(row.normalized_product_type), str(row.split)): {
            "support": int(row.support),
            "target_support": int(row.target_support),
            "target_probability": float(row.target_probability),
        }
        for row in split.itertuples(index=False)
    }
    zero = {"support": 0, "target_support": 0, "target_probability": 0.0}
    grouped["discovery_statistics"] = [
        split_lookup.get((int(center), str(product_type), "discovery"), zero)
        for center, product_type in zip(
            grouped["center_index"], grouped["normalized_product_type"]
        )
    ]
    grouped["validation_statistics"] = [
        split_lookup.get((int(center), str(product_type), "validation"), zero)
        for center, product_type in zip(
            grouped["center_index"], grouped["normalized_product_type"]
        )
    ]
    grouped["profile_cross_split_p80"] = [
        discovery["support"] >= 1
        and validation["support"] >= 1
        and discovery["target_probability"] >= minimum_probability
        and validation["target_probability"] >= minimum_probability
        for discovery, validation in zip(
            grouped["discovery_statistics"], grouped["validation_statistics"]
        )
    ]
    profiles = grouped[
        grouped["profile_pair_support"].ge(minimum_support)
        & grouped["profile_target_probability"].ge(minimum_probability)
        & grouped["profile_cross_split_p80"]
    ].copy()
    return profiles, pair_evidence


def _source_examples(
    profiles: pd.DataFrame,
    evidence: pd.DataFrame,
    pair_inputs: pd.DataFrame,
    *,
    maximum: int,
) -> dict[tuple[int, str], list[dict[str, Any]]]:
    profile_keys = profiles[
        ["center_index", "normalized_product_type", "target_label", "product_type"]
    ]
    candidates = profile_keys.merge(
        evidence,
        on=["center_index", "normalized_product_type"],
        how="inner",
        suffixes=("", "_evidence"),
    )
    candidates = candidates[candidates["human_label"].eq(candidates["target_label"])]
    candidates = candidates.sort_values(
        ["center_index", "normalized_product_type", "similarity", "pair_id"],
        ascending=[True, True, False, True],
    ).drop_duplicates(["center_index", "normalized_product_type", "pair_id"])
    inputs = pair_inputs.set_index("pair_id", drop=False)
    result: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for key, frame in candidates.groupby(
        ["center_index", "normalized_product_type"], observed=True
    ):
        examples: list[dict[str, Any]] = []
        product_type = str(frame.iloc[0]["product_type"])
        for row in frame.head(maximum).itertuples(index=False):
            pair_id = str(row.pair_id)
            if pair_id not in inputs.index:
                continue
            source = inputs.loc[pair_id]
            if isinstance(source, pd.DataFrame):
                source = source.iloc[0]
            examples.append(
                {
                    "source_pair_id": pair_id,
                    "product_type": product_type,
                    "title_a": str(source["title_a"])[:280],
                    "title_b": str(source["title_b"])[:280],
                    "target_value_a": str(row.raw_value_a)[:160],
                    "target_value_b": str(row.raw_value_b)[:160],
                }
            )
        result[(int(key[0]), str(key[1]))] = examples
    return result


def main() -> None:
    args = parse_args()
    if args.minimum_product_type_pairs < 2:
        raise ValueError("minimum-product-type-pairs must be at least 2")
    if not 0.5 <= args.minimum_probability <= 1.0:
        raise ValueError("minimum-probability must be in [0.5, 1.0]")

    report_dir = args.report_dir.resolve()
    rules_path = report_dir / "recommended_semantic_rules.parquet"
    prototypes_path = report_dir / "semantic_prototypes.parquet"
    neighbours_path = report_dir / "semantic_neighbours.parquet"
    occurrences_path = args.occurrences.resolve()
    pair_inputs_path = args.pair_inputs.resolve()
    policy_path = args.profile_capacity_policy.resolve()
    output_path = args.output.resolve()

    candidates = pd.read_parquet(rules_path)
    selected = candidates[
        candidates["threshold"].eq(args.threshold)
        & candidates["relation"].eq("different_value")
        & candidates["cross_split_p80"]
        & candidates["weighted_evidence_support"].ge(args.minimum_weighted_support)
        & candidates["target_probability"].ge(args.minimum_probability)
        & ~candidates["forbidden_identifier"]
    ].copy()
    selected = selected[
        ~selected["canonical_concept"].fillna("").str.contains(FORBIDDEN_CONCEPT_RE)
    ]
    selected["semantic_review_required"] = [
        concept in SEMANTIC_REVIEW_CONCEPTS
        or (category, concept) in SEMANTIC_REVIEW_CATEGORY_CONCEPTS
        for category, concept in zip(selected["category"], selected["canonical_concept"])
    ]
    selected = selected[~selected["semantic_review_required"]].copy()

    occurrences = pd.read_parquet(occurrences_path)
    rebuilt_prototypes, occurrence_frame = build_prototypes(occurrences)
    stored_prototypes = pd.read_parquet(prototypes_path)
    if rebuilt_prototypes["prototype_key"].tolist() != stored_prototypes[
        "prototype_key"
    ].tolist():
        raise RuntimeError("Semantic prototypes do not match the occurrence snapshot")
    prototype_index = pd.Series(
        rebuilt_prototypes.index, index=rebuilt_prototypes["prototype_key"]
    )
    occurrence_evidence = occurrence_frame.drop_duplicates(
        ["pair_id", "prototype_key"]
    ).copy()
    occurrence_evidence["neighbour_index"] = occurrence_evidence[
        "prototype_key"
    ].map(prototype_index).astype("int32")

    neighbours = pd.read_parquet(neighbours_path)
    neighbours = neighbours[
        neighbours["threshold"].eq(args.threshold)
        & neighbours["center_index"].isin(set(selected["center_index"]))
    ][["center_index", "neighbour_index", "similarity"]]
    evidence = neighbours.merge(occurrence_evidence, on="neighbour_index")

    dominant_keys = _dominant_attribute_keys(evidence)
    selected = selected.merge(dominant_keys, on="center_index", how="left")
    mapped = [_canonical_attribute_key(row) for row in selected.itertuples(index=False)]
    selected["attribute_key"] = [value[0] for value in mapped]
    selected["attribute_key_source"] = [value[1] for value in mapped]
    selected = selected[
        selected["attribute_key"].fillna("").ne("")
        & ~selected["attribute_key"].fillna("").str.contains(FORBIDDEN_ATTRIBUTE_RE)
        & (
            selected["attribute_key_source"].ne(
                "semantic_neighbour_dominant_source_key"
            )
            | selected["attribute_key_support"].fillna(0).ge(
                args.minimum_attribute_key_support
            )
        )
    ].copy()
    evidence = evidence[
        evidence["center_index"].isin(set(selected["center_index"]))
    ].copy()

    policy, policy_sha256 = load_profile_capacity_policy(policy_path)
    aliases = product_type_alias_map(policy)
    pair_inputs = pd.read_parquet(pair_inputs_path)
    pair_types = _pair_product_types(
        pair_inputs, set(evidence["pair_id"].astype(str)), aliases
    )
    profiles, pair_evidence = _profile_statistics(
        evidence,
        pair_types,
        selected,
        minimum_support=args.minimum_product_type_pairs,
        minimum_probability=args.minimum_probability,
    )
    profiles = profiles[profiles["center_index"].isin(set(selected["center_index"]))]
    executable_profiles_before_compaction = len(profiles)
    # A semantic center with many observed product types must not receive more
    # training weight merely because its source metadata is more fragmented.
    # Keep every scarce positive profile, but exactly one deterministic best
    # profile for each label-0 center.
    negative_profiles = (
        profiles[profiles["target_label"].eq(0)]
        .sort_values(
            [
                "center_index",
                "profile_pair_support",
                "profile_target_probability",
                "normalized_product_type",
            ],
            ascending=[True, False, False, True],
        )
        .drop_duplicates("center_index")
    )
    positive_profiles = profiles[profiles["target_label"].eq(1)]
    profiles = pd.concat(
        [negative_profiles, positive_profiles], ignore_index=True
    ).sort_values(["center_index", "normalized_product_type"])

    # Add the validated profile type to each occurrence before choosing examples.
    example_evidence = evidence.merge(
        pair_types[["pair_id", "normalized_product_type"]], on="pair_id"
    )
    example_lookup = _source_examples(
        profiles,
        example_evidence,
        pair_inputs,
        maximum=args.maximum_source_examples,
    )
    selected_by_center = selected.set_index("center_index", drop=False)

    exported: list[dict[str, Any]] = []
    for profile in profiles.sort_values(
        ["center_index", "normalized_product_type"]
    ).itertuples(index=False):
        center = selected_by_center.loc[int(profile.center_index)]
        if isinstance(center, pd.DataFrame):
            raise RuntimeError(f"Duplicate semantic center: {profile.center_index}")
        category = str(center["category"])
        concept = str(center["canonical_concept"])
        product_type = str(profile.product_type)
        label = int(center["target_label"])
        capacity = resolved_profile_policy(
            policy,
            category=category,
            concept=concept,
            product_type=product_type,
        )
        source_rule_id = str(center["generation_rule_id"])
        generation_rule_id = _stable_id(
            [CATALOG_VERSION, source_rule_id, category, label, profile.normalized_product_type],
            prefix="gen_sem_all_",
        )
        dependent_patterns = list(DEPENDENT_ATTRIBUTE_PATTERNS.get(concept, ()))
        dependent_patterns.extend(
            CATEGORY_DEPENDENT_ATTRIBUTE_PATTERNS.get((category, concept), ())
        )
        discovery = dict(profile.discovery_statistics)
        validation = dict(profile.validation_statistics)
        examples = example_lookup.get(
            (int(profile.center_index), str(profile.normalized_product_type)), []
        )
        anchor_profile = {
            "product_type": product_type,
            "normalized_product_type": str(profile.normalized_product_type),
            "pair_support": int(profile.profile_pair_support),
            "target_support": int(profile.profile_target_support),
            "target_probability": float(profile.profile_target_probability),
            "split_statistics": {
                "discovery": discovery,
                "validation": validation,
            },
            "source_pair_ids": [example["source_pair_id"] for example in examples],
        }
        exported.append(
            {
                "generation_rule_id": generation_rule_id,
                "source_rule_id": source_rule_id,
                "generation_tier": TIER_TEMPLATE.format(label=label),
                "label": label,
                "target_label": label,
                "concept": concept,
                "raw_concept": str(center["raw_concept"]),
                "relation": str(center["relation"]),
                "semantic_family": "semantic_atomic_difference_all_pairs_correlational",
                "attribute_key": str(center["attribute_key"]),
                "attribute_key_source": str(center["attribute_key_source"]),
                "anchor_hint": generation_anchor_hint(
                    concept, str(center["attribute_key"])
                ),
                "allowed_categories": [category],
                "allowed_product_types": [product_type],
                "allowed_anchor_context_keys": list(
                    ALLOWED_ANCHOR_CONTEXT_KEYS.get(
                        concept, DEFAULT_ALLOWED_ANCHOR_CONTEXT_KEYS
                    )
                ),
                "forbidden_anchor_attribute_patterns": list(
                    dict.fromkeys(dependent_patterns)
                ),
                "target_value_pattern": TARGET_VALUE_PATTERNS.get(concept, ""),
                "forbidden_target_value_pattern": FORBIDDEN_TARGET_VALUE_PATTERNS.get(
                    concept, ""
                ),
                "target_value_domain": list(capacity.get("target_value_domain") or []),
                "primary_task_safety_cap": capacity.get("primary_task_safety_cap"),
                "profile_capacity_policy_version": str(policy["policy_version"]),
                "profile_capacity_policy_sha256": policy_sha256,
                "source_examples": examples,
                "generation_action": (
                    "Replace exactly this explicit attribute with a different, realistic "
                    "value for the same concrete product subtype."
                ),
                "required_postcondition": (
                    "Change only this attribute; preserve the product subtype and all "
                    "unrelated facts, and update every title mention consistently."
                ),
                "evidence_scope": "all_pairs_correlational",
                "semantic_threshold": float(center["threshold"]),
                "semantic_weighted_support": float(
                    center["weighted_evidence_support"]
                ),
                "semantic_target_probability": float(center["target_probability"]),
                "semantic_cross_split_p80": bool(center["cross_split_p80"]),
                "profile_pair_support": int(profile.profile_pair_support),
                "profile_target_support": int(profile.profile_target_support),
                "profile_target_probability": float(
                    profile.profile_target_probability
                ),
                "profile_cross_split_p80": bool(profile.profile_cross_split_p80),
                "discovery_profile_statistics": discovery,
                "validation_profile_statistics": validation,
                "anchor_profiles": [anchor_profile],
            }
        )

    ids = [rule["generation_rule_id"] for rule in exported]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Generated semantic profile IDs are not unique")
    if not exported:
        raise RuntimeError("No semantic rules survived executable profile filtering")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(
        output_path,
        json.dumps(exported, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    manifest = {
        "schema_version": 1,
        "catalog_version": CATALOG_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "semantic_rules": str(rules_path),
        "semantic_rules_sha256": sha256(rules_path),
        "semantic_prototypes": str(prototypes_path),
        "semantic_prototypes_sha256": sha256(prototypes_path),
        "semantic_neighbours": str(neighbours_path),
        "semantic_neighbours_sha256": sha256(neighbours_path),
        "occurrences": str(occurrences_path),
        "occurrences_sha256": sha256(occurrences_path),
        "pair_inputs": str(pair_inputs_path),
        "pair_inputs_sha256": sha256(pair_inputs_path),
        "profile_capacity_policy": str(policy_path),
        "profile_capacity_policy_sha256": policy_sha256,
        "selection": {
            "semantic_threshold": args.threshold,
            "minimum_semantic_weighted_support": args.minimum_weighted_support,
            "minimum_semantic_target_probability": args.minimum_probability,
            "require_semantic_cross_split_p80": True,
            "minimum_attribute_key_support": args.minimum_attribute_key_support,
            "minimum_product_type_pair_support": args.minimum_product_type_pairs,
            "minimum_product_type_target_probability": args.minimum_probability,
            "require_product_type_cross_split_p80": True,
            "forbid_identifiers": True,
            "exclude_existing_semantic_review_list": True,
            "evidence_interpretation": "experimental_correlational_not_causal",
            "one_executable_rule_per_semantic_center_product_type_profile": True,
            "label0_profile_compaction": (
                "one_best_profile_per_semantic_center_by_support_probability_type"
            ),
            "label1_profile_compaction": "keep_all_valid_profiles",
            "reject_placeholder_or_non_alphabetic_product_types": True,
            "placeholder_product_types": sorted(PLACEHOLDER_PRODUCT_TYPES),
            "recommended_first_experiment_two_rule_fraction": 0.0,
        },
        "semantic_centers_after_rule_filters": int(len(selected)),
        "semantic_centers_with_executable_profiles": int(
            profiles["center_index"].nunique()
        ),
        "executable_profiles_before_label0_compaction": int(
            executable_profiles_before_compaction
        ),
        "exported_rules": len(exported),
        "label_counts": dict(
            sorted(Counter(str(rule["label"]) for rule in exported).items())
        ),
        "category_counts": dict(
            sorted(Counter(rule["allowed_categories"][0] for rule in exported).items())
        ),
        "category_coverage": len(
            {rule["allowed_categories"][0] for rule in exported}
        ),
        "tier_counts": dict(
            sorted(Counter(rule["generation_tier"] for rule in exported).items())
        ),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    atomic_write_bytes(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
