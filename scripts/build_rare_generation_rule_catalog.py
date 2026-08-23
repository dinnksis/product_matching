"""Build a rare, category-specific rule catalog for controlled pair generation.

The script uses only frozen RULE_DISCOVERY statistics and the already frozen
RULE_INTERNAL_VALIDATION check.  Ordinary, hard and OOD data are not read.

The output separates executable RARE_SAFE rules from candidates that still
need stronger clean evidence or a semantic guard.  Statistical associations
are never treated as deterministic labels for arbitrary natural pairs: label 0
is valid only under the controlled one-concept intervention contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULES = (
    ROOT
    / "reports"
    / "qwen_rule_internal_validation_sample3000_v1"
    / "frozen_rule_validation_results.parquet"
)
DEFAULT_CATEGORY_RESULTS = (
    ROOT
    / "reports"
    / "qwen_rule_internal_validation_sample3000_v1"
    / "frozen_category_validation_results.parquet"
)
DEFAULT_CORE_RULES = (
    ROOT / "configs" / "generation_rule_catalog_v0" / "negative_mutation_rules_ready.csv"
)
DEFAULT_ASSIGNMENTS = (
    ROOT / "reports" / "qwen_rules_checkpoint_60000" / "rule_assignments_label_free.parquet"
)
DEFAULT_LABELS = ROOT / "data" / "qwen_rule_discovery_full_v1" / "pilot_labels.parquet"
DEFAULT_INPUTS = ROOT / "data" / "qwen_rule_discovery_full_v1" / "pilot_inputs.parquet"
DEFAULT_OUTPUT = ROOT / "configs" / "generation_rule_catalog_rare_v1"


# These concepts describe a concrete, controllable product variant/specification.
# The allow-list is deliberately explicit: adding a concept is a versioned semantic
# decision, not an accidental consequence of a changed statistical threshold.
SAFE_CONCEPT_FAMILIES = {
    "gemstone_type": "categorical_variant",
    "ram_capacity": "capacity_spec",
    "heel_height": "dimensional_spec",
    "oxygen_permeability": "optical_prescription",
    "vehicle_compatibility": "compatibility_target",
    "case_diameter": "dimensional_spec",
    "axis": "optical_prescription",
    "instrument_type": "instrument_spec",
    "cylinder_power": "optical_prescription",
    "flavor_profile": "consumable_variant",
    "target_animal": "target_compatibility",
    "load_index": "vehicle_spec",
    "compatible_model": "compatibility_target",
    "battery_capacity": "capacity_spec",
    "ssd_capacity": "capacity_spec",
    "frame_size": "dimensional_spec",
    "lens_type": "optical_prescription",
    "string_gauge": "instrument_spec",
    "cable_length": "dimensional_spec",
    "wheel_diameter": "dimensional_spec",
    "design_pattern": "categorical_variant",
    "active_ingredient": "composition_identity",
    "storage_capacity": "capacity_spec",
    "curtain_height": "dimensional_spec",
    "curtain_width": "dimensional_spec",
    "thickness": "dimensional_spec",
    "breaking_load": "technical_spec",
    "needle_diameter": "dimensional_spec",
    "curl_type": "beauty_variant",
    "lash_length": "beauty_variant",
    "lash_curl": "beauty_variant",
    "page_yield": "consumable_compatibility",
    "instrument_size": "instrument_spec",
    "bed_width": "dimensional_spec",
    "lash_thickness": "beauty_variant",
    "key_tuning": "instrument_spec",
    "base_curve": "optical_prescription",
    "sheet_size": "dimensional_spec",
    "hook_size": "dimensional_spec",
    "paper_format": "dimensional_spec",
    "engraving_text": "personalization",
    "custom_name": "personalization",
    "compatible_phone_model": "compatibility_target",
    "pasta_shape": "consumable_variant",
    "blade_length": "dimensional_spec",
    "line_diameter": "dimensional_spec",
    "target_surface": "target_compatibility",
    "tea_type": "consumable_variant",
}


# These are statistically promising, but their meaning can depend on packaging,
# measurement conventions, subsets/synonyms or category-specific annotation policy.
SEMANTIC_GUARD_CONCEPTS = {
    "product_form",
    "diameter",
    "material_composition",
    "total_weight",
    "max_load",
    "material_type",
    "fragrance_family",
    "design_theme",
    "height_cm",
    "fabric_type",
    "orientation",
    "filler_material",
    "finish_type",
    "package_length",
    "color_pattern",
}


# These concepts mostly duplicate an already executable core rule.  Keeping them
# would inflate the apparent diversity without creating a genuinely new mutation.
CORE_ALIAS = {
    "gold_color": "color",
    "metal_color": "color",
    "total_volume": "volume",
    "width_mm": "width",
    "model_line": "product_line",
    "model_series": "model_family",
    "sphere_power": "optical_power",
    "lens_power": "optical_power",
    "compatible_models": "compatible_model",
    "target_species": "target_animal",
    "string_gauge_range": "string_gauge",
    "main_ingredient": "active_ingredient",
}


FAMILY_GUARDS = {
    "capacity_spec": (
        "Use two explicit normalized capacities for the same component and unit; "
        "do not confuse device storage, RAM, card capacity or bundle capacity."
    ),
    "compatibility_target": (
        "Replace the exact compatible model with a different valid model; reject "
        "universal accessories and compatibility lists containing both models."
    ),
    "target_compatibility": (
        "Use mutually exclusive explicit targets; reject broad/subset pairs and "
        "items explicitly suitable for both targets."
    ),
    "optical_prescription": (
        "Change one normalized prescription parameter only, preserve sign and unit "
        "semantics, and keep all other prescription fields fixed."
    ),
    "dimensional_spec": (
        "Change the named product dimension in the same unit; reject package "
        "dimensions, approximate measurements and missing values."
    ),
    "instrument_spec": (
        "Choose two valid but distinct values for the same instrument subtype and "
        "keep instrument family, brand and model context fixed."
    ),
    "vehicle_spec": (
        "Change one normalized vehicle/tyre specification; do not mix index, size, "
        "speed rating or compatibility fields."
    ),
    "beauty_variant": (
        "Choose two explicit, mutually distinct variant values and update all title "
        "and attribute mentions consistently."
    ),
    "personalization": (
        "Change the explicit personalized text/name while preserving the base item; "
        "both sides must contain non-empty concrete values."
    ),
    "consumable_variant": (
        "Choose mutually exclusive explicit variants, not synonyms, mixtures, "
        "assortments or a broad value containing the narrow value."
    ),
    "consumable_compatibility": (
        "Change the exact consumable specification while preserving compatible "
        "device family and all unrelated facts."
    ),
    "composition_identity": (
        "Replace the principal explicit ingredient, not concentration, additive or "
        "free-from claim; reject multi-ingredient subset comparisons."
    ),
    "technical_spec": (
        "Change one normalized technical specification with the same unit and "
        "measurement definition; reject ranges that overlap."
    ),
    "categorical_variant": (
        "Choose mutually exclusive explicit category-valid values; reject synonyms, "
        "subsets and multi-value lists containing both variants."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rare generation rules from frozen evidence.")
    parser.add_argument("--validated-rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--category-results", type=Path, default=DEFAULT_CATEGORY_RESULTS)
    parser.add_argument("--core-rules", type=Path, default=DEFAULT_CORE_RULES)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--max-support", type=int, default=500)
    parser.add_argument("--min-clean-discovery-support", type=int, default=5)
    parser.add_argument("--min-validation-support", type=int, default=3)
    parser.add_argument("--min-clean-validation-support", type=int, default=1)
    parser.add_argument("--min-category-support", type=int, default=5)
    parser.add_argument("--examples-per-rule", type=int, default=3)
    return parser.parse_args()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_Нет строк._"
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        rendered = [str(value).replace("|", "\\|").replace("\n", " ") for value in values]
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def allowed_category_rows(
    rule_ids: set[str], category_results: pd.DataFrame, min_support: int
) -> pd.DataFrame:
    selected = category_results[category_results["rule_id"].isin(rule_ids)].copy()
    discovery_ok = (
        selected["category_support"].ge(min_support)
        & selected["category_discovery_class"].eq("NEGATIVE")
        & selected["category_effect_median"].lt(0)
    )
    # A category with at least two validation observations in the opposite median
    # direction is excluded.  Absence of category-level validation is not promoted
    # as evidence: global validation direction and discovery category evidence are
    # still mandatory, and the category remains explicitly scoped.
    validation_opposite = (
        selected["validation_category_support"].fillna(0).ge(2)
        & selected["validation_category_effect_median"].fillna(-1).ge(0)
    )
    result = selected[discovery_ok & ~validation_opposite].copy()
    return result.sort_values(["rule_id", "category"])


def classify_candidates(rules: pd.DataFrame, core_ids: set[str], args: argparse.Namespace) -> pd.DataFrame:
    candidates = rules[
        rules["rule_role"].eq("RULE_CANDIDATE")
        & rules["relation"].eq("different_value")
        & rules["discovery_effect_class"].eq("NEGATIVE")
        & rules["global_support"].between(args.min_support, args.max_support)
        & rules["max_2_differences_support"].ge(args.min_clean_discovery_support)
        & rules["validation_support"].ge(args.min_validation_support)
        & rules["validation_effect_median"].lt(0)
        & ~rules["validation_opposite_95"].fillna(False)
        & ~rules["rule_id"].isin(core_ids)
    ].copy()

    candidates["semantic_family"] = candidates["concept"].map(SAFE_CONCEPT_FAMILIES)
    candidates["core_alias_of"] = candidates["concept"].map(CORE_ALIAS)
    candidates["selection_tier"] = "RARE_REVIEW"
    candidates.loc[candidates["concept"].isin(SEMANTIC_GUARD_CONCEPTS), "selection_tier"] = (
        "RARE_EXPERIMENTAL_SEMANTIC_GUARD"
    )
    candidates.loc[candidates["core_alias_of"].notna(), "selection_tier"] = "EXCLUDED_CORE_ALIAS"
    safe_semantic = candidates["semantic_family"].notna() & candidates["core_alias_of"].isna()
    candidates.loc[safe_semantic, "selection_tier"] = (
        "RARE_EXPERIMENTAL_NEEDS_CLEAN_VALIDATION"
    )
    candidates.loc[
        safe_semantic
        & candidates["validation_max_2_differences_support"].ge(
            args.min_clean_validation_support
        ),
        "selection_tier",
    ] = "RARE_SAFE"
    candidates["generation_guard"] = candidates["semantic_family"].map(FAMILY_GUARDS)
    return candidates.sort_values(["selection_tier", "global_support", "concept"])


def representative_examples(
    safe: pd.DataFrame,
    safe_category_scope: pd.DataFrame,
    assignments: pd.DataFrame,
    labels: pd.DataFrame,
    inputs: pd.DataFrame,
    examples_per_rule: int,
) -> pd.DataFrame:
    selected = assignments[assignments["rule_id"].isin(safe["rule_id"])].copy()
    selected = selected.merge(
        safe_category_scope[["rule_id", "category"]].drop_duplicates(),
        on=["rule_id", "category"],
        how="inner",
        validate="many_to_one",
    )
    selected = selected.merge(labels, on="pair_id", validate="many_to_one")
    selected["clean_rank"] = selected["semantic_fact_count"].gt(2).astype(int)
    selected["label_rank"] = selected["human_label"].ne(0).astype(int)
    selected = selected.sort_values(
        ["rule_id", "clean_rank", "label_rank", "semantic_fact_count", "pair_id"]
    ).groupby("rule_id", as_index=False).head(examples_per_rule)
    selected = selected.merge(
        inputs[["pair_id", "title_a", "title_b"]], on="pair_id", validate="many_to_one"
    )
    return selected[
        [
            "rule_id",
            "concept",
            "category",
            "pair_id",
            "human_label",
            "semantic_fact_count",
            "value_a",
            "value_b",
            "title_a",
            "title_b",
        ]
    ]


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rules = pd.read_parquet(args.validated_rules.resolve())
    category_results = pd.read_parquet(args.category_results.resolve())
    core = pd.read_csv(args.core_rules.resolve())
    core_ids = set(core["rule_id"].astype(str))

    candidates = classify_candidates(rules, core_ids, args)
    category_scope = allowed_category_rows(
        set(candidates["rule_id"]), category_results, args.min_category_support
    )
    allowed = category_scope.groupby("rule_id")["category"].apply(
        lambda values: sorted(values.astype(str).unique().tolist())
    ).to_dict()
    candidates["allowed_categories"] = candidates["rule_id"].map(
        lambda rule_id: json.dumps(allowed.get(rule_id, []), ensure_ascii=False)
    )
    candidates.loc[
        candidates["selection_tier"].eq("RARE_SAFE")
        & candidates["allowed_categories"].eq("[]"),
        "selection_tier",
    ] = "RARE_EXPERIMENTAL_NO_CATEGORY_SCOPE"

    safe = candidates[candidates["selection_tier"].eq("RARE_SAFE")].copy()
    experimental = candidates[
        candidates["selection_tier"].str.startswith("RARE_EXPERIMENTAL")
    ].copy()

    columns = [
        "rule_id",
        "canonical_rule",
        "concept",
        "relation",
        "semantic_family",
        "selection_tier",
        "global_support",
        "global_positive",
        "global_negative",
        "single_difference_support",
        "max_2_differences_support",
        "global_effect_median",
        "scope_candidate_frozen",
        "validation_support",
        "validation_positive",
        "validation_negative",
        "validation_max_2_differences_support",
        "validation_effect_median",
        "validation_stability",
        "allowed_categories",
        "core_alias_of",
        "generation_guard",
    ]
    candidates[columns].to_csv(
        output / "rare_rule_candidates_all.csv", index=False, encoding="utf-8-sig"
    )
    safe[columns].to_csv(output / "rare_negative_rules_safe.csv", index=False, encoding="utf-8-sig")
    safe[columns].to_parquet(output / "rare_negative_rules_safe.parquet", index=False)
    experimental[columns].to_csv(
        output / "rare_negative_rules_experimental.csv", index=False, encoding="utf-8-sig"
    )
    safe_category_scope = category_scope[category_scope["rule_id"].isin(safe["rule_id"])].copy()
    safe_category_scope.to_csv(
        output / "rare_safe_category_scope.csv", index=False, encoding="utf-8-sig"
    )

    machine_rules: list[dict[str, Any]] = []
    for row in safe.itertuples(index=False):
        machine_rules.append(
            {
                "generation_rule_id": f"gen_rare_neg_{row.rule_id}",
                "source_rule_id": row.rule_id,
                "rule_kind": "NEGATIVE_MUTATION",
                "generation_tier": "RARE_SAFE",
                "label": 0,
                "concept": row.concept,
                "relation": row.relation,
                "semantic_family": row.semantic_family,
                "scope": "CATEGORY_SCOPED",
                "allowed_categories": json.loads(row.allowed_categories),
                "generation_action": (
                    "Replace exactly one explicit canonical value with a different "
                    "category-valid value for the same concept."
                ),
                "required_postcondition": row.generation_guard,
                "discovery_support": int(row.global_support),
                "discovery_max_2_support": int(row.max_2_differences_support),
                "validation_support": int(row.validation_support),
                "validation_max_2_support": int(row.validation_max_2_differences_support),
                "validation_direction": "NEGATIVE",
            }
        )
    write_jsonl(output / "rare_generation_rules_v1.jsonl", machine_rules)
    (output / "rare_generation_rules_v1.json").write_text(
        json.dumps(machine_rules, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    assignments = pd.read_parquet(args.assignments.resolve())
    labels = pd.read_parquet(args.labels.resolve())[["pair_id", "human_label"]]
    inputs = pd.read_parquet(args.inputs.resolve(), columns=["pair_id", "title_a", "title_b"])
    examples = representative_examples(
        safe, safe_category_scope, assignments, labels, inputs, args.examples_per_rule
    )
    examples.to_csv(output / "rare_safe_examples.csv", index=False, encoding="utf-8-sig")

    tier_counts = candidates["selection_tier"].value_counts().sort_index().to_dict()
    summary = {
        "version": "generation_rule_catalog_rare_v1",
        "source": "RULE_DISCOVERY_60000_PLUS_FROZEN_INTERNAL_VALIDATION_3000",
        "ordinary_hard_ood_used": False,
        "rare_support_range": [args.min_support, args.max_support],
        "candidate_count": int(len(candidates)),
        "rare_safe_rule_count": int(len(safe)),
        "rare_experimental_rule_count": int(len(experimental)),
        "rare_safe_rule_category_combinations": int(
            sum(len(json.loads(value)) for value in safe["allowed_categories"])
        ),
        "tier_counts": {str(key): int(value) for key, value in tier_counts.items()},
        "safe_requirements": {
            "different_value_only": True,
            "discovery_direction": "NEGATIVE",
            "validation_median_direction": "NEGATIVE",
            "min_clean_discovery_support": args.min_clean_discovery_support,
            "min_validation_support": args.min_validation_support,
            "min_clean_validation_support": args.min_clean_validation_support,
            "min_discovery_category_support": args.min_category_support,
            "explicit_semantic_allow_list": True,
            "category_scoped_only": True,
        },
        "deterministic_label_warning": (
            "Rules assign label 0 only to controlled one-concept mutations satisfying "
            "the rule guard, never to arbitrary natural pairs."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_table = safe[
        [
            "concept",
            "semantic_family",
            "global_support",
            "validation_support",
            "validation_max_2_differences_support",
            "allowed_categories",
        ]
    ].copy()
    report_table["allowed_categories"] = report_table["allowed_categories"].map(
        lambda value: ", ".join(json.loads(value))
    )
    report_table = report_table.rename(
        columns={
            "concept": "концепт",
            "semantic_family": "семейство",
            "global_support": "discovery support",
            "validation_support": "validation support",
            "validation_max_2_differences_support": "clean validation",
            "allowed_categories": "разрешённые категории",
        }
    ).sort_values("discovery support")
    report = f"""# Редкие правила генерации — v1

## Результат

Из frozen-таблиц отобрано **{len(safe)} редких безопасных правил** и
**{len(experimental)} экспериментальных кандидатов**. Безопасные правила дают
**{summary['rare_safe_rule_category_combinations']}** разрешённых комбинаций
`правило × категория` сверх основного каталога v0.

Ordinary, hard и OOD не читались. Определения правил не менялись по validation:
internal validation использована только для проверки направления и наличия clean-support.

## Что означает RARE_SAFE

- support в 60k RULE_DISCOVERY находится в диапазоне {args.min_support}–{args.max_support};
- relation — только `different_value`;
- направление относительно category baseline отрицательное в discovery и validation;
- есть минимум {args.min_clean_discovery_support} discovery-пар с максимум двумя различиями;
- есть минимум {args.min_clean_validation_support} такая validation-пара;
- концепт допускает контролируемое однозначное вмешательство;
- правило применяется только в перечисленных категориях.

Это не означает, что правило детерминированно классифицирует произвольные естественные
пары. Label=0 допустим только при генерации из одного исходного item: меняется ровно
один явный факт, остальные latent facts сохраняются, а title и attributes обновляются
согласованно.

## Готовые редкие правила

{markdown_table(report_table)}

## Что осталось экспериментальным

Кандидаты в `rare_negative_rules_experimental.csv` не потеряны. Они разделены на:

- недостаточно clean-support на internal validation;
- неоднозначную семантику (упаковка, общий размер, материал, ориентация и т. п.);
- отсутствие надёжного category scope;
- алиасы уже существующих core-правил.

Их не следует смешивать с `RARE_SAFE` при автоматическом назначении label=0.
"""
    (output / "REPORT_RU.md").write_text(report, encoding="utf-8")

    manifest = {
        **summary,
        "files": sorted(path.name for path in output.iterdir()),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
