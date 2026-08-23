"""Freeze rule definitions and discovery-only statistical classifications.

The script never reads internal-validation or test data.  Its output is the
immutable registry that later validation code is allowed to score, but not to
extend or redefine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GLOBAL = ROOT / "reports" / "qwen_rules_checkpoint_60000" / "global_rule_summary.parquet"
DEFAULT_CATEGORY = ROOT / "reports" / "qwen_rules_checkpoint_60000" / "pilot_rule_table.parquet"
DEFAULT_CONCEPT_MAP = (
    ROOT
    / "artifacts"
    / "qwen_concept_normalization_v1_checkpoint_60000"
    / "concept_normalization_map.parquet"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "qwen_rules_frozen_v1_60000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze candidate rules created on the 60k RULE_DISCOVERY checkpoint."
    )
    parser.add_argument("--global-rules", type=Path, default=DEFAULT_GLOBAL)
    parser.add_argument("--category-rules", type=Path, default=DEFAULT_CATEGORY)
    parser.add_argument("--concept-map", type=Path, default=DEFAULT_CONCEPT_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--direction-min-support", type=int, default=20)
    parser.add_argument("--neutral-min-support", type=int, default=50)
    parser.add_argument("--category-min-support", type=int, default=10)
    parser.add_argument("--neutral-log-odds-margin", type=float, default=0.25)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def support_tier(support: int) -> str:
    if support < 10:
        return "RARE_CANDIDATE"
    if support < 20:
        return "EXPLORATORY"
    if support < 50:
        return "ESTIMABLE"
    if support < 100:
        return "ROBUST_DISCOVERY"
    return "HIGH_CONFIDENCE_DISCOVERY"


def effect_class(
    support: int,
    low: float,
    high: float,
    direction_min_support: int,
    neutral_min_support: int,
    neutral_margin: float,
) -> str:
    # Equivalence is checked before direction: a tiny but precisely estimated
    # effect is practically neutral even if its interval misses exactly zero.
    if support >= neutral_min_support and low >= -neutral_margin and high <= neutral_margin:
        return "NEUTRAL"
    if support >= direction_min_support and low > 0:
        return "POSITIVE"
    if support >= direction_min_support and high < 0:
        return "NEGATIVE"
    return "UNCERTAIN"


def rule_role(relation: str) -> str:
    if relation == "missing_one_side":
        return "CONTEXT_ONLY"
    if relation in {"unknown", "conflicting_sources"}:
        return "REVIEW_BEFORE_USE"
    return "RULE_CANDIDATE"


def classify_scope(global_row: pd.Series, category_rows: pd.DataFrame) -> str:
    eligible = category_rows[category_rows["category_support_eligible"]].copy()
    positive = int(eligible["category_discovery_class"].eq("POSITIVE").sum())
    negative = int(eligible["category_discovery_class"].eq("NEGATIVE").sum())
    global_class = str(global_row["discovery_effect_class"])

    if global_class in {"POSITIVE", "NEGATIVE"}:
        same = positive if global_class == "POSITIVE" else negative
        opposite = negative if global_class == "POSITIVE" else positive
        if opposite:
            return (
                "GLOBAL_WITH_EXCEPTIONS"
                if same >= 2 and same > opposite
                else "HETEROGENEOUS"
            )
        if same >= 2:
            return "GLOBAL"
        if same == 1:
            return "CATEGORY_SPECIFIC"
        return "UNCERTAIN"
    if global_class == "NEUTRAL":
        if len(eligible) >= 2 and not positive and not negative:
            return "GLOBAL"
        return "UNCERTAIN"
    if positive and negative:
        return "HETEROGENEOUS"
    if positive or negative:
        return "CATEGORY_SPECIFIC"
    return "UNCERTAIN"


def markdown_counts(series: pd.Series, name: str) -> str:
    rows = [f"| {name} | count |", "| --- | ---: |"]
    rows.extend(f"| {key} | {value} |" for key, value in series.value_counts().items())
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    if args.direction_min_support < 1 or args.neutral_min_support < 1:
        raise ValueError("Support thresholds must be positive")
    if args.category_min_support < 1 or args.neutral_log_odds_margin <= 0:
        raise ValueError("Category support and neutral margin must be positive")

    global_path = args.global_rules.resolve()
    category_path = args.category_rules.resolve()
    concept_map_path = args.concept_map.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    global_rules = pd.read_parquet(global_path).copy()
    category_rules = pd.read_parquet(category_path).copy()
    definitions = ["rule_id", "canonical_rule", "concept", "relation"]
    if global_rules["rule_id"].duplicated().any():
        raise RuntimeError("Global source contains duplicate rule_id")
    if global_rules[definitions].isna().any().any():
        raise RuntimeError("A frozen rule definition contains null values")

    global_rules["support_tier"] = global_rules["global_support"].map(
        lambda value: support_tier(int(value))
    )
    global_rules["discovery_effect_class"] = global_rules.apply(
        lambda row: effect_class(
            int(row.global_support),
            float(row.global_effect_low_95),
            float(row.global_effect_high_95),
            args.direction_min_support,
            args.neutral_min_support,
            args.neutral_log_odds_margin,
        ),
        axis=1,
    )
    global_rules["rule_role"] = global_rules["relation"].map(rule_role)
    global_rules["definition_frozen"] = True
    global_rules["definition_source"] = "RULE_DISCOVERY_CHECKPOINT_60000"

    category_columns = [
        "rule_id", "category", "category_support", "category_positive",
        "category_negative", "category_baseline", "category_effect_median",
        "category_effect_low_95", "category_effect_high_95", "category_effect_sd",
        "category_shrinkage_weight", "category_effect_shrunk",
    ]
    category = category_rules[category_columns].drop_duplicates(
        ["rule_id", "category"]
    ).copy()
    category["category_support_eligible"] = category["category_support"].ge(
        args.category_min_support
    )
    category["category_discovery_class"] = category.apply(
        lambda row: effect_class(
            int(row.category_support),
            float(row.category_effect_low_95),
            float(row.category_effect_high_95),
            args.category_min_support,
            max(args.category_min_support, args.neutral_min_support),
            args.neutral_log_odds_margin,
        ),
        axis=1,
    )

    scopes: dict[str, str] = {}
    category_by_rule = {rid: rows for rid, rows in category.groupby("rule_id", sort=False)}
    for row in global_rules.itertuples(index=False):
        scopes[row.rule_id] = classify_scope(
            pd.Series(row._asdict()), category_by_rule.get(row.rule_id, category.iloc[0:0])
        )
    global_rules["scope_candidate_frozen"] = global_rules["rule_id"].map(scopes)

    global_rules.to_parquet(output / "frozen_rule_definitions.parquet", index=False)
    global_rules.to_csv(
        output / "frozen_rule_definitions.csv", index=False, encoding="utf-8-sig"
    )
    category.to_parquet(output / "frozen_category_effects.parquet", index=False)
    category.to_csv(
        output / "frozen_category_effects.csv", index=False, encoding="utf-8-sig"
    )

    policy: dict[str, Any] = {
        "version": "qwen_rules_frozen_v1_60000",
        "created_at": datetime.now(UTC).isoformat(),
        "definition": "canonical concept + orientation-invariant relation",
        "definition_source": "RULE_DISCOVERY checkpoint of 60000 pairs only",
        "direction_min_support": args.direction_min_support,
        "neutral_min_support": args.neutral_min_support,
        "category_min_support": args.category_min_support,
        "neutral_log_odds_margin": args.neutral_log_odds_margin,
        "support_tiers": {
            "RARE_CANDIDATE": "support < 10",
            "EXPLORATORY": "10 <= support < 20",
            "ESTIMABLE": "20 <= support < 50",
            "ROBUST_DISCOVERY": "50 <= support < 100",
            "HIGH_CONFIDENCE_DISCOVERY": "support >= 100",
        },
        "classification": {
            "POSITIVE": "support >= 20 and 95% effect interval is above zero",
            "NEGATIVE": "support >= 20 and 95% effect interval is below zero",
            "NEUTRAL": "support >= 50 and the full 95% interval is inside +/-0.25 log-odds",
            "UNCERTAIN": "all other cases",
        },
        "important": [
            "These are discovery evidence tiers, not product-matching decision thresholds.",
            "missing_one_side remains context-only even when statistically associated with label.",
            "Validation may score only these frozen rule_ids and cannot create new definitions.",
        ],
    }
    (output / "classification_policy.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        **policy,
        "rule_count": len(global_rules),
        "category_rule_count": len(category),
        "global_source": str(global_path),
        "global_source_sha256": sha256(global_path),
        "category_source": str(category_path),
        "category_source_sha256": sha256(category_path),
        "concept_map": str(concept_map_path),
        "concept_map_sha256": sha256(concept_map_path),
        "effect_class_counts": global_rules["discovery_effect_class"].value_counts().to_dict(),
        "scope_counts": global_rules["scope_candidate_frozen"].value_counts().to_dict(),
        "support_tier_counts": global_rules["support_tier"].value_counts().to_dict(),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = f"""# Замороженные candidate rules: RULE_DISCOVERY 60k

Definitions всех **{len(global_rules)}** правил зафиксированы до обращения к
`rule_internal_validation`. Пороги ниже являются уровнями надёжности discovery,
а не порогами применения правил к product matching.

## Discovery-классы

{markdown_counts(global_rules['discovery_effect_class'], 'класс')}

## Scope candidates

{markdown_counts(global_rules['scope_candidate_frozen'], 'scope')}

`missing_one_side` всегда помечен `CONTEXT_ONLY`: наличие информации только с
одной стороны не является доказательством match или non-match.

Internal validation разрешено только сопоставлять с `rule_id` из
`frozen_rule_definitions.parquet`. Новые concepts/rules сохраняются отдельно как
coverage diagnostics и не меняют этот registry.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
