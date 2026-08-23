"""Build a generation-ready rule catalog from frozen discovery evidence.

No API/LLM calls are made.  Human labels are attached only to already frozen
semantic facts and anchors for statistical summaries.  The resulting labels are
deterministic only under the explicit controlled-generation contract written to
generation_policy_v0.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = (
    ROOT
    / "artifacts"
    / "qwen_semantic_extraction_v1_3_sanitized_checkpoint_60000"
    / "sanitized_pairs.parquet"
)
DEFAULT_ASSIGNMENTS = (
    ROOT / "reports" / "qwen_rules_checkpoint_60000" / "rule_assignments_label_free.parquet"
)
DEFAULT_LABELS = ROOT / "data" / "qwen_rule_discovery_full_v1" / "pilot_labels.parquet"
DEFAULT_RULES = (
    ROOT
    / "reports"
    / "qwen_rule_internal_validation_sample3000_v1"
    / "frozen_rule_validation_results.parquet"
)
DEFAULT_CATEGORIES = (
    ROOT / "artifacts" / "qwen_rules_frozen_v1_60000" / "frozen_category_effects.parquet"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "generation_rule_catalog_v0"


POSITIVE_TRANSFORMATIONS = [
    {
        "generation_rule_id": "gen_pos_surface_paraphrase_v0",
        "rule_kind": "POSITIVE_TRANSFORMATION",
        "label": 1,
        "concept": "surface_form",
        "relation": "semantically_same",
        "generation_action": "Paraphrase or reorder title tokens without changing any semantic fact.",
        "required_postcondition": "Canonical semantic signature and latent item identity are unchanged.",
    },
    {
        "generation_rule_id": "gen_pos_equivalent_format_v0",
        "rule_kind": "POSITIVE_TRANSFORMATION",
        "label": 1,
        "concept": "value_format",
        "relation": "compatible",
        "generation_action": "Change units, punctuation, spelling or numeric formatting to an equivalent representation.",
        "required_postcondition": "Every normalized value remains equal after unit conversion.",
    },
    {
        "generation_rule_id": "gen_pos_source_redistribution_v0",
        "rule_kind": "POSITIVE_TRANSFORMATION",
        "label": 1,
        "concept": "evidence_source",
        "relation": "semantically_same",
        "generation_action": "Move redundant evidence between title and attributes or remove one duplicate source.",
        "required_postcondition": "No semantic value changes; title and attributes do not conflict.",
    },
    {
        "generation_rule_id": "gen_pos_information_omission_v0",
        "rule_kind": "POSITIVE_TRANSFORMATION",
        "label": 1,
        "concept": "information_completeness",
        "relation": "missing_one_side",
        "generation_action": "Remove a subset of facts from one representation of the same latent item.",
        "required_postcondition": "Known values never become different_value; all identity values that remain are unchanged.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create machine-readable generation rules and Russian handoff reports."
    )
    parser.add_argument("--discovery-pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--discovery-assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--discovery-labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--validated-rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--category-effects", type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-clean-discovery-support", type=int, default=20)
    parser.add_argument("--min-clean-validation-support", type=int, default=2)
    parser.add_argument("--min-category-support", type=int, default=10)
    return parser.parse_args()


def wilson(positive: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    p = positive / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return float(center - radius), float(center + radius)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        rendered = [str(value).replace("|", "\\|").replace("\n", " ") for value in values]
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def parse_anchor_rows(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair in pairs.itertuples(index=False):
        anchors = json.loads(pair.identity_anchors_json)
        seen: set[tuple[str, str]] = set()
        for anchor in anchors:
            anchor_type = str(anchor.get("anchor_type") or "other_identity")
            concept = str(anchor.get("concept") or anchor_type)
            key = (anchor_type, concept)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "pair_id": str(pair.pair_id),
                    "category": str(pair.category),
                    "anchor_type": anchor_type,
                    "concept": concept,
                    "strength": str(anchor.get("strength") or "unknown"),
                }
            )
    return pd.DataFrame(rows)


def anchor_statistics(anchors: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    labeled = anchors.merge(labels, on="pair_id", validate="many_to_one")
    rows: list[dict[str, Any]] = []
    for keys, group in labeled.groupby(["anchor_type", "concept", "strength"], sort=True):
        positive = int(group["human_label"].sum())
        support = int(group["pair_id"].nunique())
        low, high = wilson(positive, support)
        rows.append(
            {
                "anchor_type": keys[0],
                "concept": keys[1],
                "strength": keys[2],
                "support": support,
                "positive": positive,
                "negative": support - positive,
                "match_prevalence": positive / support,
                "wilson_low_95": low,
                "wilson_high_95": high,
                "category_count": int(group["category"].nunique()),
                "generation_role": "IDENTITY_GUARD",
                "label_if_seen_in_natural_pair": None,
                "warning": "Same anchor alone never guarantees label=1.",
            }
        )
    return pd.DataFrame(rows).sort_values("support", ascending=False)


def category_lists(
    rule_ids: set[str], categories: pd.DataFrame, min_support: int
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    selected = categories[categories["rule_id"].isin(rule_ids)].copy()
    allowed: dict[str, list[str]] = {}
    excluded: dict[str, list[str]] = {}
    for rid, group in selected.groupby("rule_id"):
        scope = str(group["scope_candidate_frozen"].iloc[0])
        if scope in {"GLOBAL", "GLOBAL_WITH_EXCEPTIONS"}:
            mask = (
                group["category_support"].ge(min_support)
                & group["category_effect_median"].lt(0)
                & ~group["category_discovery_class"].eq("POSITIVE")
            )
        elif scope == "CATEGORY_SPECIFIC":
            mask = group["category_discovery_class"].eq("NEGATIVE")
        else:
            mask = pd.Series(False, index=group.index)
        allowed[rid] = sorted(group.loc[mask, "category"].astype(str).unique().tolist())
        excluded[rid] = sorted(group.loc[~mask, "category"].astype(str).unique().tolist())
    return allowed, excluded


def conditional_anchor_differences(
    anchors: pd.DataFrame,
    assignments: pd.DataFrame,
    labels: pd.DataFrame,
    rule_lookup: pd.DataFrame,
) -> pd.DataFrame:
    anchor_pairs = anchors[["pair_id", "anchor_type", "concept", "strength"]].drop_duplicates()
    anchor_labeled = anchor_pairs.merge(labels, on="pair_id", validate="many_to_one")
    anchor_base = (
        anchor_labeled.groupby(["anchor_type", "concept", "strength"])["human_label"]
        .agg(["size", "mean"])
        .rename(columns={"size": "anchor_support", "mean": "anchor_match_prevalence"})
        .reset_index()
    )
    joined = anchor_pairs.merge(
        assignments[
            ["pair_id", "rule_id", "canonical_rule", "relation", "semantic_fact_count"]
        ].drop_duplicates(["pair_id", "rule_id"]),
        on="pair_id",
        validate="many_to_many",
    ).merge(labels, on="pair_id", validate="many_to_one")
    rows: list[dict[str, Any]] = []
    keys = ["anchor_type", "concept", "strength", "rule_id", "canonical_rule", "relation"]
    for values, group in joined.groupby(keys, sort=True):
        support = len(group)
        positive = int(group["human_label"].sum())
        low, high = wilson(positive, support)
        rows.append(
            {
                **dict(zip(keys, values)),
                "conditional_support": support,
                "conditional_positive": positive,
                "conditional_negative": support - positive,
                "conditional_match_prevalence": positive / support,
                "conditional_wilson_low_95": low,
                "conditional_wilson_high_95": high,
                "conditional_max_2_support": int(group["semantic_fact_count"].le(2).sum()),
            }
        )
    result = pd.DataFrame(rows).merge(
        anchor_base, on=["anchor_type", "concept", "strength"], validate="many_to_one"
    )
    result["prevalence_delta_vs_anchor"] = (
        result["conditional_match_prevalence"] - result["anchor_match_prevalence"]
    )
    result = result.merge(
        rule_lookup[
            ["rule_id", "discovery_effect_class", "validation_stability", "generation_status"]
        ],
        on="rule_id",
        how="left",
        validate="many_to_one",
    )
    result["interpretation"] = "DIAGNOSTIC_ONLY"
    result.loc[
        result["generation_status"].eq("READY_LABEL_0"), "interpretation"
    ] = "HARD_NEGATIVE_GENERATION_SUPPORT"
    tolerated = (
        result["relation"].eq("different_value")
        & result["conditional_support"].ge(20)
        & result["prevalence_delta_vs_anchor"].ge(-0.05)
    )
    result.loc[tolerated, "interpretation"] = "TOLERATED_DIFFERENCE_CANDIDATE_NOT_DETERMINISTIC"
    return result.sort_values("conditional_support", ascending=False)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_parquet(
        args.discovery_pairs.resolve(),
        columns=[
            "pair_id", "category", "identity_anchors_json", "difference_count",
            "missing_information_count",
        ],
    )
    pairs["pair_id"] = pairs["pair_id"].astype(str)
    labels = pd.read_parquet(args.discovery_labels.resolve())
    labels["pair_id"] = labels["pair_id"].astype(str)
    labels = labels[labels["pair_id"].isin(pairs["pair_id"])].copy()
    if len(labels) != len(pairs) or labels["pair_id"].duplicated().any():
        raise RuntimeError("Discovery labels do not match the 60k extraction one-to-one")
    assignments = pd.read_parquet(args.discovery_assignments.resolve())
    rules = pd.read_parquet(args.validated_rules.resolve()).copy()
    categories = pd.read_parquet(args.category_effects.resolve()).merge(
        rules[["rule_id", "scope_candidate_frozen"]], on="rule_id", validate="many_to_one"
    )

    strict_negative = rules[
        rules["rule_role"].eq("RULE_CANDIDATE")
        & rules["discovery_effect_class"].eq("NEGATIVE")
        & rules["global_support"].ge(50)
        & rules["validation_stability"].eq("REPLICATED_95")
        & rules["scope_candidate_frozen"].isin(
            ["GLOBAL", "GLOBAL_WITH_EXCEPTIONS", "CATEGORY_SPECIFIC"]
        )
    ].copy()
    strict_negative["generation_status"] = np.where(
        strict_negative["max_2_differences_support"].ge(args.min_clean_discovery_support)
        & strict_negative["validation_max_2_differences_support"].ge(
            args.min_clean_validation_support
        ),
        "READY_LABEL_0",
        "STRICT_ASSOCIATION_NEEDS_ISOLATION",
    )
    ready = strict_negative[strict_negative["generation_status"].eq("READY_LABEL_0")].copy()
    allowed, excluded = category_lists(
        set(strict_negative["rule_id"]), categories, args.min_category_support
    )
    strict_negative["allowed_categories"] = strict_negative["rule_id"].map(
        lambda rid: json.dumps(allowed.get(rid, []), ensure_ascii=False)
    )
    strict_negative["excluded_or_unsupported_categories"] = strict_negative["rule_id"].map(
        lambda rid: json.dumps(excluded.get(rid, []), ensure_ascii=False)
    )
    strict_negative["generated_label"] = np.where(
        strict_negative["generation_status"].eq("READY_LABEL_0"), 0, np.nan
    )
    strict_negative["generation_action"] = (
        "Replace the explicit value with a different category-valid value for the same canonical concept."
    )
    strict_negative["generation_contract"] = (
        "Apply only in an allowed category; change one semantic concept; preserve all other latent facts; "
        "synchronize title and attributes; reject conflicting or unknown outputs."
    )
    strict_negative.to_csv(
        output / "negative_rule_evidence_all.csv", index=False, encoding="utf-8-sig"
    )

    ready = strict_negative[
        strict_negative["generation_status"].eq("READY_LABEL_0")
        & strict_negative["allowed_categories"].ne("[]")
    ].copy()
    ready.to_csv(output / "negative_mutation_rules_ready.csv", index=False, encoding="utf-8-sig")

    positive_rows = []
    all_categories = sorted(pairs["category"].astype(str).unique().tolist())
    for row in POSITIVE_TRANSFORMATIONS:
        positive_rows.append(
            {
                **row,
                "generation_status": "READY_LABEL_1_BY_CONSTRUCTION",
                "allowed_categories": json.dumps(all_categories, ensure_ascii=False),
                "generation_contract": (
                    "Both records must be generated from the same latent item; no canonical explicit value "
                    "may change; strong identity values must be preserved; missing is not mismatch."
                ),
                "evidence_source": "generation invariant checked against RULE_DISCOVERY semantics",
            }
        )
    positive = pd.DataFrame(positive_rows)
    positive.to_csv(
        output / "positive_transformation_rules_ready.csv", index=False, encoding="utf-8-sig"
    )

    machine_rules: list[dict[str, Any]] = []
    for row in ready.itertuples(index=False):
        machine_rules.append(
            {
                "generation_rule_id": f"gen_neg_{row.rule_id}",
                "source_rule_id": row.rule_id,
                "rule_kind": "NEGATIVE_MUTATION",
                "label": 0,
                "concept": row.concept,
                "relation": row.relation,
                "scope": row.scope_candidate_frozen,
                "allowed_categories": json.loads(row.allowed_categories),
                "generation_action": row.generation_action,
                "required_postcondition": row.generation_contract,
                "discovery_support": int(row.global_support),
                "discovery_max_2_support": int(row.max_2_differences_support),
                "validation_support": int(row.validation_support),
                "validation_max_2_support": int(row.validation_max_2_differences_support),
                "validation_stability": row.validation_stability,
            }
        )
    for row in positive_rows:
        machine_rules.append(
            {**row, "allowed_categories": json.loads(row["allowed_categories"])}
        )
    write_jsonl(output / "generation_rules_v0.jsonl", machine_rules)
    (output / "generation_rules_v0.json").write_text(
        json.dumps(machine_rules, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    anchors = parse_anchor_rows(pairs)
    anchor_stats = anchor_statistics(anchors, labels)
    anchor_stats.to_csv(output / "positive_anchor_statistics.csv", index=False, encoding="utf-8-sig")
    anchor_category = anchors.merge(labels, on="pair_id", validate="many_to_one").groupby(
        ["anchor_type", "concept", "strength", "category"]
    )["human_label"].agg(["size", "sum"]).reset_index()
    anchor_category = anchor_category.rename(columns={"size": "support", "sum": "positive"})
    anchor_category["negative"] = anchor_category["support"] - anchor_category["positive"]
    anchor_category["match_prevalence"] = anchor_category["positive"] / anchor_category["support"]
    anchor_category.to_csv(
        output / "positive_anchor_statistics_by_category.csv", index=False, encoding="utf-8-sig"
    )

    rule_lookup = rules[["rule_id", "discovery_effect_class", "validation_stability"]].merge(
        strict_negative[["rule_id", "generation_status"]], on="rule_id", how="left"
    )
    conditional = conditional_anchor_differences(
        anchors, assignments, labels, rule_lookup
    )
    conditional.to_csv(
        output / "anchor_conditioned_difference_statistics.csv", index=False, encoding="utf-8-sig"
    )

    labeled_pairs = pairs.merge(labels, on="pair_id", validate="one_to_one")
    positive_pairs = labeled_pairs[labeled_pairs["human_label"].eq(1)]
    negative_pairs = labeled_pairs[labeled_pairs["human_label"].eq(0)]
    anchor_pair_ids = set(anchors["pair_id"])
    positive_anchor_coverage = positive_pairs["pair_id"].isin(anchor_pair_ids).mean()
    negative_anchor_coverage = negative_pairs["pair_id"].isin(anchor_pair_ids).mean()
    allowed_rows = pd.DataFrame(
        [
            {"rule_id": rid, "category": category}
            for rid, category_values in allowed.items()
            if rid in set(ready["rule_id"])
            for category in category_values
        ]
    )
    allowed_rows.to_csv(
        output / "ready_rule_category_scope.csv", index=False, encoding="utf-8-sig"
    )
    ready_assignments = assignments[
        assignments["rule_id"].isin(ready["rule_id"])
    ].merge(allowed_rows, on=["rule_id", "category"], how="inner", validate="many_to_one")
    category_coverage = (
        ready_assignments.groupby("category").agg(
            observed_pairs=("pair_id", "nunique"), ready_rules=("rule_id", "nunique")
        ).reset_index()
    )
    category_coverage.to_csv(
        output / "ready_negative_coverage_by_category.csv", index=False, encoding="utf-8-sig"
    )
    min_ready_per_category = int(category_coverage["ready_rules"].min())

    policy = {
        "version": "generation_rule_catalog_v0",
        "deterministic_label_contract": {
            "label_0": (
                "Start from one structured item and intervene on exactly one READY_LABEL_0 concept "
                "using a different valid value in an allowed category. Keep every other latent fact fixed "
                "and render title/attributes consistently."
            ),
            "label_1": (
                "Render two records from the same latent item and apply only a READY_LABEL_1_BY_CONSTRUCTION "
                "surface/completeness transformation. Never change a normalized explicit semantic value."
            ),
            "natural_pairs": (
                "The catalog does not guarantee a deterministic label for arbitrary natural pairs; use a "
                "trained matcher or human label there."
            ),
        },
        "hard_guards": [
            "missing_a/missing_b never creates label 0",
            "same SKU/MPN/model is an identity guard, not sufficient proof of label 1",
            "do not use a rule outside allowed_categories",
            "do not use STRICT_ASSOCIATION_NEEDS_ISOLATION for deterministic generation",
            "do not infer new OOD category rules from this catalog",
        ],
    }
    (output / "generation_policy_v0.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Generation rule v0",
        "type": "object",
        "required": [
            "generation_rule_id", "rule_kind", "label", "concept", "relation",
            "allowed_categories", "generation_action", "required_postcondition",
        ],
        "properties": {
            "generation_rule_id": {"type": "string"},
            "source_rule_id": {"type": "string"},
            "rule_kind": {"enum": ["NEGATIVE_MUTATION", "POSITIVE_TRANSFORMATION"]},
            "label": {"enum": [0, 1]},
            "concept": {"type": "string"},
            "relation": {"type": "string"},
            "scope": {"type": "string"},
            "allowed_categories": {"type": "array", "items": {"type": "string"}},
            "generation_action": {"type": "string"},
            "required_postcondition": {"type": "string"},
        },
        "additionalProperties": True,
    }
    (output / "generation_rule_v0.schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    decision = {
        "discovery_pairs": len(pairs),
        "discovery_positive_pairs": int(positive_pairs.shape[0]),
        "discovery_negative_pairs": int(negative_pairs.shape[0]),
        "positive_anchor_pair_coverage": float(positive_anchor_coverage),
        "negative_anchor_pair_coverage": float(negative_anchor_coverage),
        "strict_validated_negative_rules": len(strict_negative),
        "ready_clean_negative_rules": len(ready),
        "ready_positive_transformations": len(positive),
        "categories_with_ready_negative_evidence": len(category_coverage),
        "minimum_ready_negative_rules_per_category": min_ready_per_category,
        "additional_qwen_recommended_before_v0_generation": False,
        "reason": (
            "The current catalog has clean validated negative interventions in every current-train category, "
            "four positive transformations with deterministic same-latent-item labels, and 15515 existing "
            "positive discovery pairs for anchor diagnostics. More Qwen would mostly expand the long tail."
        ),
        "targeted_qwen_later_only_if": [
            "a required category/concept has no READY rule",
            "the generator needs semantic value changes for label 1 rather than safe surface transformations",
            "STRICT_ASSOCIATION_NEEDS_ISOLATION rules must be promoted using more clean single-rule evidence",
        ],
        "hard_ood_used": False,
    }
    (output / "qwen_sufficiency_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    ready_report = ready[
        [
            "concept", "scope_candidate_frozen", "global_support",
            "validation_support", "allowed_categories",
        ]
    ].copy()
    ready_report["категорий"] = ready_report["allowed_categories"].map(
        lambda value: len(json.loads(value))
    )
    ready_report = ready_report.drop(columns="allowed_categories").rename(
        columns={
            "scope_candidate_frozen": "scope",
            "global_support": "discovery support",
            "validation_support": "validation support",
        }
    )
    anchor_report = (
        anchors.merge(labels, on="pair_id", validate="many_to_one")
        .groupby(["anchor_type", "strength"])["human_label"]
        .agg(["size", "sum", "mean"])
        .reset_index()
        .rename(
            columns={
                "anchor_type": "anchor",
                "strength": "сила",
                "size": "support",
                "sum": "positive",
                "mean": "P(match)",
            }
        )
    )
    anchor_report["P(match)"] = anchor_report["P(match)"].map(lambda value: f"{value:.3f}")

    human_report = f"""# Готовые правила для генерации пар — v0

## Решение

Дополнительный запуск Qwen перед первой генерацией **не нужен**. На 60k
`RULE_DISCOVERY` уже есть **{len(positive_pairs)} positive** и
**{len(negative_pairs)} negative** пар. После проверки на отдельной выборке
получено **{len(strict_negative)}** строгих negative associations; из них
**{len(ready)}** имеют достаточную поддержку в относительно чистых парах и
разрешены для контролируемой генерации label=0.

Во всех **{len(category_coverage)}** current-train категориях наблюдаются готовые
negative rules; минимум на категорию — **{min_ready_per_category}**.

## Как получать однозначный label

- `label=0`: взять структурированный item, изменить ровно один concept из
  `negative_mutation_rules_ready.csv` на другое валидное значение, не менять
  остальные latent facts и синхронно обновить title/attributes.
- `label=1`: создать два представления одного latent item и применять только
  операции из `positive_transformation_rules_ready.csv`. Явные semantic values
  изменять нельзя.

Это обеспечивает label **по конструкции генератора**. Статистические rules сами
по себе не дают логической гарантии label для произвольной естественной пары.

## Позитивы и identity anchors

Хотя бы один anchor найден в **{positive_anchor_coverage:.1%}** positive-пар.
Anchor сохраняется как guard/context: даже одинаковый SKU, MPN или exact model не
является самостоятельной гарантией match, если присутствует критическое
различие. Missing information также никогда не является mismatch.

### Статистика identity anchors

{markdown_table(anchor_report)}

Одинаковый MPN/SKU даёт сильный сигнал, но не 100% гарантию на естественных
парах. Для synthetic label=1 гарантия возникает из общего latent item, а anchors
служат проверкой сохранения identity.

## Готовые negative mutations

{markdown_table(ready_report)}

Полный список разрешённых категорий для каждого правила хранится в
`negative_mutation_rules_ready.csv` и `ready_rule_category_scope.csv`.

## Что пока не использовать

- rules со статусом `STRICT_ASSOCIATION_NEEDS_ISOLATION`;
- provisional/uncertain/heterogeneous rules;
- positive statistical associations как автоматический label=1;
- категории, отсутствующие в `allowed_categories`;
- hard и OOD для дополнения или исправления catalog.
"""
    (output / "REPORT_FOR_HUMAN_RU.md").write_text(human_report, encoding="utf-8")

    agent_report = f"""# Передача другому агенту: generation rule catalog v0

## Границы данных

Definitions правил и анализ positive anchors построены только по 60k
`RULE_DISCOVERY`. Stability взята из ранее замороженной проверки на 3k
`rule_internal_validation`; validation не создавала и не меняла definitions.
Ordinary, hard и OOD не читались.

## Машиночитаемые входы

- `generation_rules_v0.json/jsonl`: только исполняемые deterministic actions;
- `generation_rule_v0.schema.json`: JSON Schema одной записи;
- `generation_policy_v0.json`: обязательные preconditions/postconditions;
- `negative_mutation_rules_ready.csv`: {len(ready)} interventions для label=0;
- `positive_transformation_rules_ready.csv`: {len(positive)} transformations для label=1;
- `positive_anchor_statistics.csv`: identity guards, не самостоятельные labels;
- `anchor_conditioned_difference_statistics.csv`: диагностика; запись исполнима
  только при статусе связанного правила `READY_LABEL_0`;
- `negative_rule_evidence_all.csv`: строгие associations, включая неисполняемые
  строки `STRICT_ASSOCIATION_NEEDS_ISOLATION`.

## Обязательное поведение генератора

Для label=0 применить ровно одну intervention и взять категорию из
`allowed_categories`. Новое значение должно быть валидным для категории;
остальные latent facts сохраняются, а все упоминания в title/attributes
обновляются согласованно. Для label=1 обе записи создаются из одного latent item;
разрешены только surface, equivalent-format, source-redistribution и
information-omission transformations. Positive pair отклоняется, если появился
`different_value`, `incompatible` или source conflict.

Нельзя считать observational probabilities детерминированными labels для
естественных пар. Нельзя расширять definitions по validation/hard/OOD.
Дополнительный Qwen для v0 не требуется; позднее разрешён targeted pass только
по неиспользованным `RULE_DISCOVERY` pairs для конкретных непокрытых concepts.
"""
    (output / "HANDOFF_FOR_AGENT_RU.md").write_text(agent_report, encoding="utf-8")

    validation_checks = {
        "unique_generation_rule_ids": len({row["generation_rule_id"] for row in machine_rules})
        == len(machine_rules),
        "all_labels_binary": all(row["label"] in {0, 1} for row in machine_rules),
        "all_allowed_categories_are_lists": all(
            isinstance(row["allowed_categories"], list) for row in machine_rules
        ),
        "all_negative_rules_have_allowed_categories": all(
            row["allowed_categories"]
            for row in machine_rules
            if row["rule_kind"] == "NEGATIVE_MUTATION"
        ),
        "all_current_train_categories_have_ready_negative_rule": len(category_coverage)
        == len(all_categories),
        "hard_ood_used": False,
    }
    validation_checks["all_checks_passed"] = all(
        value is True
        for key, value in validation_checks.items()
        if key != "hard_ood_used"
    ) and validation_checks["hard_ood_used"] is False
    (output / "catalog_validation.json").write_text(
        json.dumps(validation_checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not validation_checks["all_checks_passed"]:
        raise RuntimeError(f"Generation catalog validation failed: {validation_checks}")

    manifest = {**decision, "files": sorted(path.name for path in output.iterdir())}
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    print(f"Catalog: {output}", flush=True)


if __name__ == "__main__":
    main()
