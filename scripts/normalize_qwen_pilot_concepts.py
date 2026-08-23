"""Build a conservative label-free concept map for the Qwen semantic pilot."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "artifacts"
    / "qwen_semantic_extraction_v1_3_sanitized500"
    / "sanitized_pairs.jsonl"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "qwen_concept_normalization_v1_500"

# Only reviewed, meaning-preserving aliases belong here. Contextually different
# concepts (package_weight vs net_weight, model_name vs model_family, etc.) stay
# separate even when their names look similar.
ALIASES: dict[str, tuple[str, str]] = {
    "product_weight": ("weight", "generic_product_prefix"),
    "item_weight": ("weight", "generic_product_prefix"),
    "unit_weight": ("weight", "generic_product_prefix"),
    "weight_g": ("weight", "unit_moved_to_value"),
    "volume_ml": ("volume", "unit_moved_to_value"),
    "length_cm": ("length", "unit_moved_to_value"),
    "length_mm": ("length", "unit_moved_to_value"),
    "width_cm": ("width", "unit_moved_to_value"),
    "heel_height_cm": ("heel_height", "unit_moved_to_value"),
    "sole_height_cm": ("sole_height", "unit_moved_to_value"),
    "insole_length_cm": ("insole_length", "unit_moved_to_value"),
    "color_name": ("color", "lexical_alias"),
    "color_description": ("color", "lexical_alias"),
    "color_variant": ("color", "lexical_alias"),
    "color_variation": ("color", "lexical_alias"),
    "size_ru": ("russian_size", "size_system_alias"),
    "ru_size": ("russian_size", "size_system_alias"),
    "russian_shoe_size": ("russian_size", "size_system_alias"),
    "shoe_size_ru": ("russian_size", "size_system_alias"),
    "size_eu": ("european_size", "size_system_alias"),
    "eu_size": ("european_size", "size_system_alias"),
    "size_eur": ("european_size", "size_system_alias"),
    "shoe_size_eu": ("european_size", "size_system_alias"),
    "origin_country": ("country_of_origin", "lexical_alias"),
    "part_number": ("manufacturer_part_number", "identifier_alias"),
    "oem_number": ("manufacturer_part_number", "identifier_alias"),
    "oem_part_number": ("manufacturer_part_number", "identifier_alias"),
    "oe_code": ("manufacturer_part_number", "identifier_alias"),
    "package_type": ("packaging_type", "lexical_alias"),
    "material_upper": ("upper_material", "token_order_alias"),
    "main_material": ("material", "generic_main_alias"),
    "shelf_life_days": ("shelf_life", "unit_moved_to_value"),
    "shelf_life_months": ("shelf_life", "unit_moved_to_value"),
    "target_gender": ("gender", "lexical_alias"),
    "target_audience_gender": ("gender", "lexical_alias"),
    "target_age_min": ("min_age", "lexical_alias"),
    "target_age_max": ("max_age", "lexical_alias"),
    "screen_diagonal": ("screen_size", "lexical_alias"),
    "clasp_type": ("closure_type", "lexical_alias"),
    "included_items": ("package_contents", "lexical_alias"),
    "included_accessories": ("package_contents", "lexical_alias"),
    "piece_count": ("package_quantity", "quantity_alias"),
    "units_per_item": ("package_quantity", "quantity_alias"),
    "seasonality": ("season", "lexical_alias"),
    "water_resistance_rating": ("water_resistance", "lexical_alias"),
    "water_resistance_class": ("water_resistance", "lexical_alias"),
    "psu_wattage": ("psu_power", "unit_moved_to_value"),
    "scent_name": ("scent", "lexical_alias"),
    "scent_variant": ("scent", "lexical_alias"),
    "flavor_variant": ("flavor", "lexical_alias"),
    "primary_flavor": ("flavor", "lexical_alias"),
    "special_need": ("special_needs", "singular_plural_alias"),
    "modes_count": ("mode_count", "singular_plural_alias"),
    "compatible_device_model": ("compatible_model", "reviewed_qwen_alias"),
}

RELATION_GROUPS = {
    "missing_a": "missing_one_side",
    "missing_b": "missing_one_side",
    "subset": "specificity_difference",
    "more_specific": "specificity_difference",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a conservative label-free concept map.")
    parser.add_argument("--extractions", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def semantic_facts(pair: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        *pair.get("identity_anchors", []),
        *pair.get("differences", []),
        *pair.get("missing_information", []),
    ]


def relation_group(relation: str) -> str:
    return RELATION_GROUPS.get(relation, relation)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    support: Counter[str] = Counter()
    categories: dict[str, set[str]] = defaultdict(set)
    relations: dict[str, set[str]] = defaultdict(set)
    pairs = 0
    pair_rule_before = 0
    pair_rule_after = 0

    with args.extractions.resolve().open(encoding="utf-8") as stream:
        for line in stream:
            pair = json.loads(line)
            # human_label may exist in the post-inference artifact, but is
            # deliberately never read by this label-free normalization pass.
            pairs += 1
            category = str(pair["category"])
            before: set[tuple[str, str]] = set()
            after: set[tuple[str, str]] = set()
            for fact in semantic_facts(pair):
                concept = str(fact["concept"])
                relation = relation_group(str(fact.get("relation", "same")))
                support[concept] += 1
                categories[concept].add(category)
                relations[concept].add(relation)
                if relation != "identity_same" and relation != "same":
                    before.add((concept, relation))
                    after.add((ALIASES.get(concept, (concept, "unchanged"))[0], relation))
            pair_rule_before += len(before)
            pair_rule_after += len(after)

    rows: list[dict[str, Any]] = []
    for concept in sorted(support):
        target, reason = ALIASES.get(concept, (concept, "unchanged"))
        rows.append(
            {
                "source_concept": concept,
                "target_concept": target,
                "changed": concept != target,
                "reason": reason,
                "fact_support": support[concept],
                "category_count": len(categories[concept]),
                "relation_count": len(relations[concept]),
                "relations": json.dumps(sorted(relations[concept]), ensure_ascii=False),
            }
        )

    mapping = pd.DataFrame(rows)
    mapping.to_csv(output_dir / "concept_normalization_map.csv", index=False, encoding="utf-8-sig")
    mapping.to_parquet(output_dir / "concept_normalization_map.parquet", index=False)
    changed = mapping[mapping["changed"]].copy()
    changed.to_csv(output_dir / "applied_aliases.csv", index=False, encoding="utf-8-sig")

    stats = {
        "pairs": pairs,
        "observed_concepts_before": int(mapping["source_concept"].nunique()),
        "concepts_after": int(mapping["target_concept"].nunique()),
        "observed_aliases_applied": len(changed),
        "facts_affected": int(changed["fact_support"].sum()),
        "pair_rule_observations_before": pair_rule_before,
        "pair_rule_observations_after": pair_rule_after,
        "pair_rule_duplicates_collapsed": pair_rule_before - pair_rule_after,
        "labels_read": False,
        "normalization_version": "qwen_concept_normalization_v1",
    }
    (output_dir / "normalization_statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# Label-free нормализация canonical concepts

Обработано пар: **{pairs}**. Human labels не читались.

- concepts до нормализации: **{stats['observed_concepts_before']}**;
- concepts после безопасных aliases: **{stats['concepts_after']}**;
- применено наблюдаемых aliases: **{stats['observed_aliases_applied']}**;
- затронуто фактов: **{stats['facts_affected']}**;
- схлопнуто повторных pair-rule observations: **{stats['pair_rule_duplicates_collapsed']}**.

Нормализация намеренно консервативна. Например, `package_weight`, `net_weight`
и `weight` не объединяются; `model_name`, `model_family` и `model_number` также
остаются разными concepts. Полная карта сохранена до присоединения labels.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
