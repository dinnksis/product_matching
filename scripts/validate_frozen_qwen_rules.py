"""Score frozen RULE_DISCOVERY definitions on rule_internal_validation.

Rule boundaries and the concept map are loaded from frozen discovery artifacts.
Any validation-only semantic formulation is reported as coverage loss and is
never promoted into the registry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_qwen_pilot_rules import (  # noqa: E402
    build_label_free_assignments,
    effect_posterior,
    global_effect_posterior,
    load_concept_map,
    stable_seed,
)


DEFAULT_EXTRACTIONS = (
    ROOT
    / "artifacts"
    / "qwen_rule_internal_validation_v1_sanitized"
    / "sanitized_pairs.jsonl"
)
DEFAULT_LABELS = (
    ROOT / "data" / "qwen_rule_internal_validation_v1" / "validation_labels.parquet"
)
DEFAULT_FROZEN = ROOT / "artifacts" / "qwen_rules_frozen_v1_60000"
DEFAULT_OUTPUT = ROOT / "reports" / "qwen_rule_internal_validation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate immutable discovery rules on internal validation."
    )
    parser.add_argument("--extractions", type=Path, default=DEFAULT_EXTRACTIONS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--frozen-dir", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--posterior-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2031)
    return parser.parse_args()


def validation_effect_class(
    support: int, low: float, high: float, policy: dict[str, Any]
) -> str:
    direction_min = int(policy["direction_min_support"])
    neutral_min = int(policy["neutral_min_support"])
    margin = float(policy["neutral_log_odds_margin"])
    if support >= neutral_min and low >= -margin and high <= margin:
        return "NEUTRAL"
    if support >= direction_min and low > 0:
        return "POSITIVE"
    if support >= direction_min and high < 0:
        return "NEGATIVE"
    return "UNCERTAIN"


def direction_from_median(value: float) -> str:
    if pd.isna(value):
        return "NO_SUPPORT"
    if value > 0:
        return "POSITIVE"
    if value < 0:
        return "NEGATIVE"
    return "ZERO"


def stability(row: pd.Series) -> str:
    discovery = str(row["discovery_effect_class"])
    support = int(row["validation_support"])
    validation = str(row["validation_effect_class"])
    median_direction = direction_from_median(float(row["validation_effect_median"]))
    if support == 0:
        return "NO_VALIDATION_SUPPORT"
    if discovery == "UNCERTAIN":
        return "DISCOVERY_UNCERTAIN"
    if discovery == "NEUTRAL":
        return (
            "REPLICATED_95"
            if validation == "NEUTRAL"
            else "NOT_REPLICATED"
        )
    if validation == discovery:
        return "REPLICATED_95"
    if median_direction == discovery:
        return "SAME_DIRECTION_WEAK"
    if median_direction in {"POSITIVE", "NEGATIVE"}:
        return "OPPOSITE_DIRECTION"
    return "INCONCLUSIVE"


def markdown_counts(series: pd.Series, name: str) -> str:
    rows = [f"| {name} | count |", "| --- | ---: |"]
    rows.extend(f"| {key} | {value} |" for key, value in series.value_counts().items())
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    if args.posterior_draws < 1000:
        raise ValueError("posterior-draws must be at least 1000")
    extractions = args.extractions.resolve()
    labels_path = args.labels.resolve()
    frozen_dir = args.frozen_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    frozen = pd.read_parquet(frozen_dir / "frozen_rule_definitions.parquet")
    frozen_category = pd.read_parquet(frozen_dir / "frozen_category_effects.parquet")
    policy = json.loads((frozen_dir / "classification_policy.json").read_text(encoding="utf-8"))
    manifest = json.loads((frozen_dir / "manifest.json").read_text(encoding="utf-8"))
    concept_map = load_concept_map(Path(manifest["concept_map"]))

    # This creates assignments without reading the human_label field embedded in
    # post-inference sanitized objects.
    assignments, pairs = build_label_free_assignments(extractions, concept_map)
    frozen_ids = set(frozen["rule_id"].astype(str))
    matched = assignments[assignments["rule_id"].isin(frozen_ids)].copy()
    unseen = assignments[~assignments["rule_id"].isin(frozen_ids)].copy()
    matched.to_parquet(output / "validation_rule_assignments_label_free.parquet", index=False)
    unseen.to_parquet(output / "validation_unseen_assignments_label_free.parquet", index=False)

    # Labels are loaded only after rule assignment and the frozen/unseen split.
    labels = pd.read_parquet(labels_path)
    if set(labels.columns) != {"pair_id", "human_label"}:
        raise ValueError("Labels must contain exactly pair_id,human_label")
    labels["pair_id"] = labels["pair_id"].astype(str)
    labels["human_label"] = labels["human_label"].astype(int)
    if labels["pair_id"].duplicated().any() or set(pairs["pair_id"]) != set(labels["pair_id"]):
        raise RuntimeError("Validation labels and extracted pairs do not match one-to-one")
    pair_labels = pairs[["pair_id", "category"]].merge(
        labels, on="pair_id", validate="one_to_one"
    )
    labeled = matched.merge(labels, on="pair_id", validate="many_to_one")

    baselines = pair_labels.groupby("category")["human_label"].agg(["size", "sum"])
    baselines = baselines.rename(columns={"size": "pair_count", "sum": "positive_count"})
    baselines["negative_count"] = baselines["pair_count"] - baselines["positive_count"]
    baselines["category_baseline"] = (
        baselines["positive_count"] + 0.5
    ) / (baselines["pair_count"] + 1)
    baselines.reset_index().to_csv(
        output / "validation_category_baselines.csv", index=False, encoding="utf-8-sig"
    )

    global_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    for rid, group in labeled.groupby("rule_id", sort=True):
        group = group.drop_duplicates("pair_id")
        positive = int(group["human_label"].sum())
        posterior = global_effect_posterior(
            group,
            baselines,
            args.posterior_draws,
            stable_seed(args.seed, rid, "validation_global"),
        )
        global_rows.append(
            {
                "rule_id": rid,
                "validation_support": len(group),
                "validation_positive": positive,
                "validation_negative": len(group) - positive,
                "validation_category_count": int(group["category"].nunique()),
                "validation_single_difference_support": int(
                    group["semantic_fact_count"].eq(1).sum()
                ),
                "validation_max_2_differences_support": int(
                    group["semantic_fact_count"].le(2).sum()
                ),
                "validation_effect": posterior["global_effect"],
                "validation_effect_median": posterior["global_effect_median"],
                "validation_effect_low_95": posterior["global_effect_low_95"],
                "validation_effect_high_95": posterior["global_effect_high_95"],
                "validation_effect_sd": posterior["global_effect_sd"],
                "validation_probability_positive": posterior["global_probability_positive"],
                "validation_probability_negative": posterior["global_probability_negative"],
            }
        )
        for category_name, category_group in group.groupby("category"):
            base = baselines.loc[category_name]
            category_positive = int(category_group["human_label"].sum())
            category_support = len(category_group)
            effect = effect_posterior(
                category_positive,
                category_support - category_positive,
                int(base["positive_count"]),
                int(base["negative_count"]),
                args.posterior_draws,
                stable_seed(args.seed, rid, category_name, "validation_category"),
            )
            category_rows.append(
                {
                    "rule_id": rid,
                    "category": category_name,
                    "validation_category_support": category_support,
                    "validation_category_positive": category_positive,
                    "validation_category_negative": category_support - category_positive,
                    "validation_category_baseline": float(base["category_baseline"]),
                    "validation_category_effect_median": effect["effect_median"],
                    "validation_category_effect_low_95": effect["effect_low_95"],
                    "validation_category_effect_high_95": effect["effect_high_95"],
                    "validation_category_effect_sd": effect["effect_sd"],
                }
            )

    validation_global = pd.DataFrame(global_rows)
    if validation_global.empty:
        raise RuntimeError("No validation semantic facts matched frozen rules")
    result = frozen.merge(validation_global, on="rule_id", how="left", validate="one_to_one")
    integer_columns = [
        "validation_support", "validation_positive", "validation_negative",
        "validation_category_count", "validation_single_difference_support",
        "validation_max_2_differences_support",
    ]
    result[integer_columns] = result[integer_columns].fillna(0).astype(int)
    result["validation_effect_class"] = result.apply(
        lambda row: (
            validation_effect_class(
                int(row.validation_support),
                float(row.validation_effect_low_95),
                float(row.validation_effect_high_95),
                policy,
            )
            if row.validation_support > 0
            else "NO_SUPPORT"
        ),
        axis=1,
    )
    result["validation_stability"] = result.apply(stability, axis=1)
    result["validation_opposite_95"] = (
        (result["discovery_effect_class"].eq("POSITIVE") & result["validation_effect_class"].eq("NEGATIVE"))
        | (result["discovery_effect_class"].eq("NEGATIVE") & result["validation_effect_class"].eq("POSITIVE"))
    )
    result["definition_changed_after_validation"] = False
    result.to_parquet(output / "frozen_rule_validation_results.parquet", index=False)
    result.to_csv(
        output / "frozen_rule_validation_results.csv", index=False, encoding="utf-8-sig"
    )

    validation_category = pd.DataFrame(category_rows)
    category_result = frozen_category.merge(
        validation_category,
        on=["rule_id", "category"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    category_result["category_registry_coverage"] = category_result["_merge"].map(
        {
            "left_only": "DISCOVERY_ONLY",
            "right_only": "VALIDATION_CATEGORY_NEW_FOR_FROZEN_RULE",
            "both": "BOTH",
        }
    )
    category_result = category_result.drop(columns="_merge")
    category_result.to_parquet(output / "frozen_category_validation_results.parquet", index=False)
    category_result.to_csv(
        output / "frozen_category_validation_results.csv", index=False, encoding="utf-8-sig"
    )

    discovery_directional = result[
        result["discovery_effect_class"].isin(["POSITIVE", "NEGATIVE"])
    ]
    robust = discovery_directional[discovery_directional["global_support"].ge(50)]
    supported_robust = robust[robust["validation_support"].gt(0)]
    strict_candidates = supported_robust[
        supported_robust["rule_role"].eq("RULE_CANDIDATE")
        & supported_robust["validation_stability"].eq("REPLICATED_95")
    ].copy()
    provisional_candidates = supported_robust[
        supported_robust["rule_role"].eq("RULE_CANDIDATE")
        & supported_robust["validation_support"].ge(10)
        & supported_robust["validation_stability"].eq("SAME_DIRECTION_WEAK")
    ].copy()
    median_flip_diagnostics = supported_robust[
        supported_robust["validation_stability"].eq("OPPOSITE_DIRECTION")
    ].copy()
    strict_candidates.to_csv(
        output / "strictly_replicated_rule_candidates.csv", index=False, encoding="utf-8-sig"
    )
    provisional_candidates.to_csv(
        output / "provisionally_stable_rule_candidates.csv", index=False, encoding="utf-8-sig"
    )
    median_flip_diagnostics.to_csv(
        output / "median_flip_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "validation_pairs": len(pairs),
        "validation_semantic_assignments": len(assignments),
        "matched_frozen_assignments": len(matched),
        "unseen_validation_assignments": len(unseen),
        "assignment_coverage": len(matched) / max(1, len(assignments)),
        "frozen_rules": len(frozen),
        "frozen_rules_with_validation_support": int(result["validation_support"].gt(0).sum()),
        "discovery_directional_rules": len(discovery_directional),
        "robust_directional_rules": len(robust),
        "robust_directional_with_validation_support": len(supported_robust),
        "strict_rule_candidates": len(strict_candidates),
        "provisionally_stable_rule_candidates": len(provisional_candidates),
        "robust_median_flips": len(median_flip_diagnostics),
        "robust_statistically_confirmed_opposites": int(
            supported_robust["validation_opposite_95"].sum()
        ),
        "robust_stability_counts": supported_robust["validation_stability"].value_counts().to_dict(),
        "all_stability_counts": result["validation_stability"].value_counts().to_dict(),
        "definitions_changed_after_validation": False,
        "ordinary_hard_ood_used": False,
    }
    (output / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# Internal validation замороженных Qwen rules

Definitions были созданы только на `RULE_DISCOVERY 60k` и не менялись после
просмотра internal-validation labels.

- validation pairs: **{len(pairs)}**;
- label-free semantic assignments: **{len(assignments)}**;
- assignments, совпавшие с frozen registry: **{len(matched)}**
  ({summary['assignment_coverage']:.1%});
- validation-only formulations: **{len(unseen)}** — сохранены только как coverage
  diagnostics и не превращены в новые rules;
- frozen rules с validation support: **{summary['frozen_rules_with_validation_support']}**.

## Stability: robust directional discovery rules

{markdown_counts(supported_robust['validation_stability'], 'результат')}

`REPLICATED_95` означает совпадение discovery-класса и validation-класса при
заранее зафиксированной статистической политике. `SAME_DIRECTION_WEAK` означает
совпавшее направление posterior median, но недостаточную validation certainty.
`OPPOSITE_DIRECTION` здесь означает только смену знака posterior median. Число
robust rules со статистически подтверждённым противоположным 95% эффектом:
**{int(supported_robust['validation_opposite_95'].sum())}**. Definition правила
при этом не меняется.

Для следующего этапа сохранены **{len(strict_candidates)}** строго подтверждённых
rule candidates и **{len(provisional_candidates)}** предварительно стабильных
rule candidates с validation support не меньше 10. `CONTEXT_ONLY` в эти списки
не включается.

Ordinary, hard и OOD не использовались.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
