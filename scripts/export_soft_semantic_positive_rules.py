"""Export lenient product-scoped label-1 semantic rules in evidence tiers A/B.

This exporter intentionally models annotation statistics rather than causal
product identity.  Singleton observations receive weight 2, observations from
multi-atom pairs receive weight 1, and every semantic neighbourhood is scored
inside a concrete category/product-type profile.

Tier A: pair support >= 5 and weighted label-1 probability >= 0.80.
Tier B: pair support >= 2 and weighted label-1 probability >= 0.70, excluding A.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from item_pipeline.normalization import normalize_text
from item_pipeline.pair_rules import CONCEPT_ATTRIBUTE_KEYS
from scripts.analyze_semantic_atomic_rules import (
    build_prototypes,
    cannot_link_concepts,
    embed_prototypes,
    nearest_neighbours,
    pooled_candidates,
    select_representative_rules,
    stable_hash,
)
from scripts.export_semantic_generation_rules import (
    _dominant_attribute_keys,
    _pair_product_types,
)
from scripts.export_statistical_generation_rules import (
    ATTRIBUTE_KEY_OVERRIDES,
    CATEGORY_ATTRIBUTE_KEY_OVERRIDES,
    FORBIDDEN_ATTRIBUTE_RE,
    atomic_write_bytes,
    sha256,
)


DEFAULT_REPORT = ROOT / "reports" / "semantic_atomic_rule_statistics_all_pairs_20260827"
DEFAULT_OCCURRENCES = (
    ROOT
    / "reports"
    / "atomic_rule_statistics_semantic_snapshot_20260826"
    / "atomic_occurrences.parquet"
)
DEFAULT_PAIR_INPUTS = (
    ROOT / "data" / "qwen_atomic_differences_v2_full_train" / "pilot_inputs.parquet"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "configs" / "generation_rule_catalog_statistical_v1" / "soft_positive_ab_v1"
)

CATALOG_VERSION = "soft_semantic_label1_ab_singleton2_multi1_scoped_v1"
SEMANTIC_THRESHOLD = 0.8
TIER_A = "SOFT_SEMANTIC_LABEL1_TIER_A_SUPPORT5_P80"
TIER_B = "SOFT_SEMANTIC_LABEL1_TIER_B_SUPPORT2_P70"
RELATIONS = ("different_value", "incompatible", "more_specific")
FALLBACK_ATTRIBUTE_KEYS = {
    "applicator_type": "Тип аппликатора",
    "car_body": "Тип кузова",
    "dimensions": "Размеры",
    "dimensions_order": "Порядок размеров",
    "game_edition": "Издание игры",
    "lock": "Тип замка",
    "name_on_product": "Название на товаре",
    "product_name": "Название товара",
    "product_scope": "Назначение товара",
    "scent_variant": "Вариант аромата",
    "title_content": "Содержание названия",
}


def _sha_id(parts: list[Any], *, prefix: str) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return prefix + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _relation_analysis(
    relation: str,
    occurrences: pd.DataFrame,
    report_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return prototypes, occurrence frame, and accepted semantic edges."""

    if relation == "different_value":
        stored_prototypes = pd.read_parquet(report_dir / "semantic_prototypes.parquet")
        candidates = pd.read_parquet(report_dir / "semantic_rule_candidates.parquet")
        edges = pd.read_parquet(report_dir / "semantic_neighbours.parquet")
        rebuilt, frame = build_prototypes(occurrences)
        if rebuilt["prototype_key"].tolist() != stored_prototypes["prototype_key"].tolist():
            raise RuntimeError("Stored semantic prototypes do not match occurrences")
        prototypes = stored_prototypes
    else:
        source = occurrences[occurrences["relation"].eq(relation)].copy()
        # build_prototypes currently selects different_value rows.  The relation
        # is a hard block and is absent from embedding text, so analyse each
        # alternate relation independently through the same implementation.
        source["relation"] = "different_value"
        prototypes, frame = build_prototypes(source)
        prototypes["relation"] = relation
        prototypes["block_key"] = [
            stable_hash((category, relation, family))
            for category, family in zip(prototypes["category"], prototypes["value_family"])
        ]
        embeddings, _ = embed_prototypes(
            prototypes,
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            256,
        )
        neighbours = nearest_neighbours(prototypes, embeddings, 24)
        candidates, edges = pooled_candidates(
            prototypes,
            neighbours,
            cannot_link_concepts(frame),
            evidence_scope="all",
            threshold=SEMANTIC_THRESHOLD,
            minimum_support=0.0,
            minimum_probability=0.5,
        )

    scored = candidates.copy()
    scored["score0"] = scored["weighted_label0"] + scored["weighted_singleton_label0"]
    scored["score1"] = scored["weighted_label1"] + scored["weighted_singleton_label1"]
    scored["target_label"] = scored["score1"].gt(scored["score0"]).astype("int8")
    scored["target_probability"] = scored[["score0", "score1"]].max(axis=1) / (
        scored["score0"] + scored["score1"]
    )
    scored["is_candidate"] = ~scored["forbidden_identifier"].astype(bool)
    representatives = select_representative_rules(scored, edges, prototypes)
    accepted_centers = set(representatives["center_index"].astype(int))
    accepted_edges = edges[edges["center_index"].isin(accepted_centers)][
        ["center_index", "neighbour_index", "similarity"]
    ].copy()
    center_metadata = representatives[
        [
            "center_index",
            "generation_rule_id",
            "canonical_concept",
            "raw_concept",
            "attribute_role",
            "value_family",
            "semantic_text",
        ]
    ].rename(
        columns={
            "raw_concept": "center_raw_concept",
            "attribute_role": "center_attribute_role",
            "value_family": "center_value_family",
            "semantic_text": "center_semantic_text",
        }
    )
    return prototypes, frame, accepted_edges.merge(
        center_metadata,
        on="center_index",
        how="inner",
    )


def _fallback_attribute_key(center: pd.Series, dominant: Any) -> tuple[str, str]:
    category = str(center["category"])
    concept = str(center["canonical_concept"])
    if (category, concept) in CATEGORY_ATTRIBUTE_KEY_OVERRIDES:
        return CATEGORY_ATTRIBUTE_KEY_OVERRIDES[(category, concept)], "category_override"
    if concept in ATTRIBUTE_KEY_OVERRIDES:
        return ATTRIBUTE_KEY_OVERRIDES[concept], "concept_override"
    dominant_text = str(dominant or "").strip()
    if dominant_text and not FORBIDDEN_ATTRIBUTE_RE.search(dominant_text):
        return dominant_text, "semantic_neighbour_dominant_source_key"
    if concept in CONCEPT_ATTRIBUTE_KEYS:
        return CONCEPT_ATTRIBUTE_KEYS[concept], "runtime_concept_override"
    if concept in FALLBACK_ATTRIBUTE_KEYS:
        return FALLBACK_ATTRIBUTE_KEYS[concept], "localized_concept_fallback"
    role = str(center.get("center_attribute_role") or "")
    if role and role != "из названия товара":
        for candidate in (part.strip() for part in role.split("|")):
            if candidate and not FORBIDDEN_ATTRIBUTE_RE.search(candidate):
                return candidate, "semantic_attribute_role_fallback"
    fallback = " ".join(part for part in concept.replace("_", " ").split() if part)
    return fallback or "характеристика", "canonical_concept_fallback"


def _profile_rows(
    *,
    relation: str,
    occurrences: pd.DataFrame,
    pair_inputs: pd.DataFrame,
    report_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prototypes, frame, edges = _relation_analysis(relation, occurrences, report_dir)
    prototype_index = pd.Series(prototypes.index, index=prototypes["prototype_key"])
    occurrence_evidence = frame.drop_duplicates(["pair_id", "prototype_key"]).copy()
    occurrence_evidence["neighbour_index"] = occurrence_evidence["prototype_key"].map(
        prototype_index
    ).astype("int32")
    evidence = edges.merge(occurrence_evidence, on="neighbour_index", how="inner")
    pair_types = _pair_product_types(pair_inputs, set(evidence["pair_id"].astype(str)), {})
    evidence = evidence.merge(pair_types, on=["pair_id", "category"], how="inner")
    evidence = (
        evidence.sort_values(
            ["center_index", "pair_id", "normalized_product_type", "similarity"],
            ascending=[True, True, True, False],
        )
        .drop_duplicates(["center_index", "pair_id", "normalized_product_type"])
        .reset_index(drop=True)
    )
    evidence["vote_weight"] = evidence["similarity"] * np.where(
        evidence["is_singleton"].astype(bool), 2.0, 1.0
    )
    evidence["profile_score0"] = evidence["vote_weight"] * evidence["human_label"].eq(0)
    evidence["profile_score1"] = evidence["vote_weight"] * evidence["human_label"].eq(1)
    profiles = (
        evidence.groupby(
            ["center_index", "category", "normalized_product_type"], observed=True
        )
        .agg(
            product_type=("product_type", "first"),
            profile_pair_support=("pair_id", "nunique"),
            profile_singleton_pairs=("is_singleton", "sum"),
            profile_score0=("profile_score0", "sum"),
            profile_score1=("profile_score1", "sum"),
            profile_label0=("human_label", lambda values: int(values.eq(0).sum())),
            profile_label1=("human_label", lambda values: int(values.eq(1).sum())),
        )
        .reset_index()
    )
    profiles["profile_target_label"] = profiles["profile_score1"].gt(
        profiles["profile_score0"]
    ).astype("int8")
    profiles["profile_target_probability"] = profiles[
        ["profile_score0", "profile_score1"]
    ].max(axis=1) / (profiles["profile_score0"] + profiles["profile_score1"])
    profiles["tier"] = ""
    tier_a = (
        profiles["profile_target_label"].eq(1)
        & profiles["profile_pair_support"].ge(5)
        & profiles["profile_target_probability"].ge(0.80)
    )
    tier_b = (
        profiles["profile_target_label"].eq(1)
        & profiles["profile_pair_support"].ge(2)
        & profiles["profile_target_probability"].ge(0.70)
        & ~tier_a
    )
    profiles.loc[tier_a, "tier"] = "A"
    profiles.loc[tier_b, "tier"] = "B"
    profiles = profiles[profiles["tier"].ne("")].copy()
    evidence = evidence[evidence["center_index"].isin(set(profiles["center_index"]))]
    dominant = _dominant_attribute_keys(evidence)
    return profiles, evidence, dominant


def _examples(
    evidence: pd.DataFrame,
    inputs_by_pair_id: pd.DataFrame,
    center_index: int,
    normalized_product_type: str,
    maximum: int = 4,
) -> list[dict[str, Any]]:
    candidates = evidence[
        evidence["center_index"].eq(center_index)
        & evidence["normalized_product_type"].eq(normalized_product_type)
        & evidence["human_label"].eq(1)
    ].sort_values(["is_singleton", "similarity", "pair_id"], ascending=[False, False, True])
    result: list[dict[str, Any]] = []
    for row in candidates.drop_duplicates("pair_id").head(maximum).itertuples(index=False):
        source = inputs_by_pair_id.loc[str(row.pair_id)]
        if isinstance(source, pd.DataFrame):
            source = source.iloc[0]
        result.append(
            {
                "source_pair_id": str(row.pair_id),
                "title_a": str(source["title_a"])[:320],
                "title_b": str(source["title_b"])[:320],
                "target_value_a": str(row.raw_value_a)[:160],
                "target_value_b": str(row.raw_value_b)[:160],
                "source_is_singleton": bool(row.is_singleton),
            }
        )
    return result


def build_catalogs(
    *,
    occurrences_path: Path,
    pair_inputs_path: Path,
    report_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    occurrences = pd.read_parquet(occurrences_path)
    pair_inputs = pd.read_parquet(pair_inputs_path)
    inputs_by_pair_id = pair_inputs.set_index("pair_id", drop=False)
    catalog: list[dict[str, Any]] = []
    relation_counts: Counter[tuple[str, str]] = Counter()
    for relation in RELATIONS:
        profiles, evidence, dominant = _profile_rows(
            relation=relation,
            occurrences=occurrences,
            pair_inputs=pair_inputs,
            report_dir=report_dir,
        )
        dominant_lookup = dominant.set_index("center_index")["attribute_key"].to_dict()
        center_rows = evidence.sort_values("similarity").drop_duplicates(
            "center_index", keep="last"
        ).set_index("center_index")
        for profile in profiles.sort_values(
            ["tier", "category", "normalized_product_type", "center_index"]
        ).itertuples(index=False):
            center = center_rows.loc[int(profile.center_index)]
            if isinstance(center, pd.DataFrame):
                center = center.iloc[-1]
            attribute_key, attribute_key_source = _fallback_attribute_key(
                center, dominant_lookup.get(int(profile.center_index), "")
            )
            context_key = "Модель" if normalize_text(attribute_key) == normalize_text("Бренд") else "Бренд"
            tier_name = TIER_A if profile.tier == "A" else TIER_B
            source_rule_id = str(center["generation_rule_id"])
            generation_rule_id = _sha_id(
                [
                    CATALOG_VERSION,
                    relation,
                    source_rule_id,
                    profile.category,
                    profile.normalized_product_type,
                    profile.tier,
                ],
                prefix="gen_soft_pos_",
            )
            examples = _examples(
                evidence,
                inputs_by_pair_id,
                int(profile.center_index),
                str(profile.normalized_product_type),
            )
            catalog.append(
                {
                    "generation_rule_id": generation_rule_id,
                    "source_rule_id": source_rule_id,
                    "generation_tier": tier_name,
                    "label": 1,
                    "target_label": 1,
                    "concept": str(center["canonical_concept"]),
                    "raw_concept": str(center["center_raw_concept"]),
                    "relation": relation,
                    "semantic_family": "soft_semantic_atomic_annotation_statistic",
                    "attribute_key": attribute_key,
                    "attribute_key_source": attribute_key_source,
                    "anchor_hint": (
                        "Создай реалистичный товар строго указанного типа. Целевой атрибут "
                        "должен быть естественным для этого товара и явно присутствовать в названии."
                    ),
                    "allowed_categories": [str(profile.category)],
                    "allowed_product_types": [str(profile.product_type)],
                    "allowed_anchor_context_keys": [context_key],
                    "required_anchor_context_keys": [context_key],
                    "forbidden_anchor_attribute_patterns": [],
                    "generation_action": (
                        "Замени значение только этого атомарного атрибута на другое "
                        "реалистичное значение, сохраняя тот же конкретный товар."
                    ),
                    "required_postcondition": (
                        "Измени только целевой атрибут и его точное упоминание в названии; "
                        "тип товара и идентифицирующий контекст должны остаться неизменными."
                    ),
                    "source_examples": examples,
                    "evidence_scope": "singleton_weight2_plus_multi_weight1",
                    "semantic_threshold": SEMANTIC_THRESHOLD,
                    "profile_pair_support": int(profile.profile_pair_support),
                    "profile_singleton_pairs": int(profile.profile_singleton_pairs),
                    "profile_label0": int(profile.profile_label0),
                    "profile_label1": int(profile.profile_label1),
                    "profile_score0": float(profile.profile_score0),
                    "profile_score1": float(profile.profile_score1),
                    "profile_target_probability": float(profile.profile_target_probability),
                    "generation_examples_per_rule": 20 if profile.tier == "A" else 10,
                }
            )
            relation_counts[(profile.tier, relation)] += 1
    tier_a_rules = [rule for rule in catalog if rule["generation_tier"] == TIER_A]
    tier_b_rules = [rule for rule in catalog if rule["generation_tier"] == TIER_B]
    ids = [rule["generation_rule_id"] for rule in catalog]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Generated rule IDs are not unique")
    summary = {
        "catalog_version": CATALOG_VERSION,
        "semantic_threshold": SEMANTIC_THRESHOLD,
        "singleton_vote_weight": 2,
        "multi_atom_vote_weight": 1,
        "relation_counts": {
            f"{tier}:{relation}": int(count)
            for (tier, relation), count in sorted(relation_counts.items())
        },
        "tier_a_rules": len(tier_a_rules),
        "tier_b_rules": len(tier_b_rules),
        "tier_a_examples_per_rule": 20,
        "tier_b_examples_per_rule": 10,
        "planned_tier_a_pairs": len(tier_a_rules) * 20,
        "planned_tier_b_pairs": len(tier_b_rules) * 10,
        "planned_total_pairs": len(tier_a_rules) * 20 + len(tier_b_rules) * 10,
    }
    return tier_a_rules, tier_b_rules, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--occurrences", type=Path, default=DEFAULT_OCCURRENCES)
    parser.add_argument("--pair-inputs", type=Path, default=DEFAULT_PAIR_INPUTS)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tier_a, tier_b, summary = build_catalogs(
        occurrences_path=args.occurrences.resolve(),
        pair_inputs_path=args.pair_inputs.resolve(),
        report_dir=args.report_dir.resolve(),
    )
    paths: dict[str, Path] = {}
    for name, rules in (("tier_a", tier_a), ("tier_b", tier_b)):
        path = output_dir / f"{name}.json"
        atomic_write_bytes(
            path,
            (json.dumps(rules, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )
        paths[name] = path
    manifest = {
        **summary,
        "occurrences": str(args.occurrences.resolve()),
        "occurrences_sha256": sha256(args.occurrences.resolve()),
        "pair_inputs": str(args.pair_inputs.resolve()),
        "pair_inputs_sha256": sha256(args.pair_inputs.resolve()),
        "semantic_report": str(args.report_dir.resolve()),
        "catalogs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }
    manifest_path = output_dir / "manifest.json"
    atomic_write_bytes(
        manifest_path,
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
