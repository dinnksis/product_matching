"""Map human-only MiniLM OOF errors to frozen Qwen semantic rules.

No synthetic data, Qwen labels, ordinary, hard or OOD labels are used.  Qwen
outputs are used only as label-free semantic assignments for RULE_DISCOVERY;
all targets come from the existing human annotation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OOF = (
    ROOT
    / "artifacts/kaggle/product-matching-minilm-s2-targeted-hard-results2"
    / "minilm_s2_targeted_hard/mining/oof_predictions_and_hardness.parquet"
)
DEFAULT_SPLITS = ROOT / "data/rule_discovery_split_v1/split_assignments.parquet"
DEFAULT_ASSIGNMENTS = ROOT / "reports/qwen_rules_checkpoint_60000/rule_assignments_label_free.parquet"
DEFAULT_INPUTS = ROOT / "data/qwen_rule_discovery_full_v1/pilot_inputs.parquet"
DEFAULT_FROZEN_RULES = (
    ROOT
    / "reports/qwen_rule_internal_validation_sample3000_v1"
    / "frozen_rule_validation_results.parquet"
)
DEFAULT_CORE = ROOT / "configs/generation_rule_catalog_v0/negative_mutation_rules_ready.csv"
DEFAULT_RARE_SAFE = (
    ROOT / "configs/generation_rule_catalog_rare_v1/rare_negative_rules_safe.csv"
)
DEFAULT_RARE_EXPERIMENTAL = (
    ROOT / "configs/generation_rule_catalog_rare_v1/rare_negative_rules_experimental.csv"
)
DEFAULT_OUTPUT = ROOT / "reports/minilm_human_errors_by_rules_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MiniLM human OOF errors by semantic rule.")
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--split-assignments", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--rule-assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--frozen-rules", type=Path, default=DEFAULT_FROZEN_RULES)
    parser.add_argument("--core-rules", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--rare-safe-rules", type=Path, default=DEFAULT_RARE_SAFE)
    parser.add_argument("--rare-experimental-rules", type=Path, default=DEFAULT_RARE_EXPERIMENTAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--examples-per-rule", type=int, default=3)
    return parser.parse_args()


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


def load_catalog_membership(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, set[str]]]:
    membership: dict[str, str] = {}
    allowed_categories: dict[str, set[str]] = {}
    for path, tier in [
        (args.core_rules, "CORE"),
        (args.rare_safe_rules, "RARE_SAFE"),
        (args.rare_experimental_rules, "RARE_EXPERIMENTAL"),
    ]:
        frame = pd.read_csv(path.resolve())
        for row in frame.itertuples(index=False):
            rule_id = str(row.rule_id)
            membership[rule_id] = tier
            raw_categories = getattr(row, "allowed_categories", "[]")
            try:
                allowed_categories[rule_id] = set(json.loads(raw_categories))
            except (TypeError, json.JSONDecodeError):
                allowed_categories[rule_id] = set()
    return membership, allowed_categories


def aggregate(group: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    result = group.groupby(keys, dropna=False).agg(
        human_pairs=("pair_id", "nunique"),
        human_positive_pairs=("is_positive", "sum"),
        human_negative_pairs=("is_negative", "sum"),
        false_positive_pairs=("is_false_positive", "sum"),
        false_negative_pairs=("is_false_negative", "sum"),
        model_error_pairs=("is_model_error", "sum"),
        oof_hard_pairs=("is_oof_hard", "sum"),
        hard_negative_pairs=("is_hard_negative", "sum"),
        hard_positive_pairs=("is_hard_positive", "sum"),
        clean_pairs=("is_clean", "sum"),
        clean_negative_pairs=("is_clean_negative", "sum"),
        clean_false_positive_pairs=("is_clean_false_positive", "sum"),
        mean_oof_score=("score", "mean"),
        mean_hardness=("hardness", "mean"),
    ).reset_index()
    result["error_rate"] = result["model_error_pairs"] / result["human_pairs"]
    negative_denominator = result["human_negative_pairs"].where(
        result["human_negative_pairs"].ne(0)
    )
    positive_denominator = result["human_positive_pairs"].where(
        result["human_positive_pairs"].ne(0)
    )
    result["false_positive_rate"] = result["false_positive_pairs"] / negative_denominator
    result["false_negative_rate"] = result["false_negative_pairs"] / positive_denominator
    result["hard_rate"] = result["oof_hard_pairs"] / result["human_pairs"]
    return result


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    oof = pd.read_parquet(args.oof.resolve())
    splits = pd.read_parquet(args.split_assignments.resolve()).reset_index(names="row_id")
    if len(oof) != len(splits):
        raise RuntimeError("OOF rows and split assignments have different lengths")
    pairs = oof.merge(splits, on="row_id", suffixes=("_oof", "_split"), validate="one_to_one")
    checks = {
        "id1": pairs["id1_oof"].astype(str).eq(pairs["id1_split"].astype(str)).all(),
        "id2": pairs["id2_oof"].astype(str).eq(pairs["id2_split"].astype(str)).all(),
        "target": pairs["target_oof"].eq(pairs["target_split"]).all(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"OOF/split row identity check failed: {checks}")
    pairs = pairs.rename(columns={"target_oof": "human_label", "category": "pair_category"})

    assignments = pd.read_parquet(args.rule_assignments.resolve()).drop_duplicates(
        ["pair_id", "rule_id"]
    )
    frozen = pd.read_parquet(args.frozen_rules.resolve())[
        ["rule_id", "canonical_rule", "concept", "relation"]
    ].drop_duplicates("rule_id")
    membership, allowed_categories = load_catalog_membership(args)
    assignments = assignments.drop(
        columns=["canonical_rule", "concept", "relation"], errors="ignore"
    ).merge(frozen, on="rule_id", how="left", validate="many_to_one")
    pair_columns = [
        "pair_id", "human_label", "score", "hardness", "is_mined_hard", "split",
        "pair_category", "id1_oof", "id2_oof",
    ]
    joined = assignments.merge(pairs[pair_columns], on="pair_id", validate="many_to_one")
    if not joined["split"].eq("rule_discovery").all():
        raise RuntimeError("Semantic assignments unexpectedly contain non-discovery pairs")

    joined["catalog_tier"] = joined["rule_id"].map(membership).fillna("OTHER_CANDIDATE")
    joined["catalog_scope_allowed"] = joined.apply(
        lambda row: (
            not allowed_categories.get(str(row.rule_id))
            or str(row.category) in allowed_categories[str(row.rule_id)]
        ),
        axis=1,
    )
    joined["is_positive"] = joined["human_label"].eq(1).astype(int)
    joined["is_negative"] = joined["human_label"].eq(0).astype(int)
    joined["is_false_positive"] = (
        joined["human_label"].eq(0) & joined["score"].ge(args.decision_threshold)
    ).astype(int)
    joined["is_false_negative"] = (
        joined["human_label"].eq(1) & joined["score"].lt(args.decision_threshold)
    ).astype(int)
    joined["is_model_error"] = (joined["is_false_positive"] | joined["is_false_negative"]).astype(int)
    joined["is_oof_hard"] = joined["is_mined_hard"].astype(int)
    joined["is_hard_negative"] = (
        joined["human_label"].eq(0) & joined["is_mined_hard"]
    ).astype(int)
    joined["is_hard_positive"] = (
        joined["human_label"].eq(1) & joined["is_mined_hard"]
    ).astype(int)
    joined["is_clean"] = joined["semantic_fact_count"].le(2).astype(int)
    joined["is_clean_negative"] = (
        joined["human_label"].eq(0) & joined["semantic_fact_count"].le(2)
    ).astype(int)
    joined["is_clean_false_positive"] = (
        joined["is_false_positive"].eq(1) & joined["semantic_fact_count"].le(2)
    ).astype(int)

    rule_stats = aggregate(
        joined,
        ["rule_id", "canonical_rule", "concept", "relation", "catalog_tier"],
    ).sort_values(["model_error_pairs", "human_pairs"], ascending=False)
    category_stats = aggregate(
        joined,
        ["rule_id", "concept", "relation", "catalog_tier", "category"],
    ).sort_values(["model_error_pairs", "human_pairs"], ascending=False)
    rule_stats.to_csv(output / "rule_error_statistics.csv", index=False, encoding="utf-8-sig")
    rule_stats.to_parquet(output / "rule_error_statistics.parquet", index=False)
    category_stats.to_csv(
        output / "category_rule_error_statistics.csv", index=False, encoding="utf-8-sig"
    )

    rare_joined = joined[
        joined["catalog_tier"].eq("RARE_SAFE") & joined["catalog_scope_allowed"]
    ].copy()
    rare_stats = aggregate(
        rare_joined,
        ["rule_id", "canonical_rule", "concept", "relation", "catalog_tier"],
    ).sort_values(["false_positive_pairs", "hard_negative_pairs", "human_negative_pairs"], ascending=False)
    rare_stats["human_training_priority"] = "LOW_OBSERVED_MODEL_ERROR"
    rare_stats.loc[rare_stats["hard_negative_pairs"].ge(5), "human_training_priority"] = (
        "USE_EXISTING_HUMAN_HARD"
    )
    rare_stats.loc[rare_stats["false_positive_pairs"].ge(5), "human_training_priority"] = (
        "USE_EXISTING_HUMAN_ERRORS_FIRST"
    )
    rare_stats.loc[rare_stats["human_negative_pairs"].lt(10), "human_training_priority"] = (
        "HUMAN_DATA_SCARCE"
    )
    rare_stats.to_csv(
        output / "rare_safe_human_training_priority.csv", index=False, encoding="utf-8-sig"
    )

    inputs = pd.read_parquet(
        args.inputs.resolve(), columns=["pair_id", "title_a", "title_b"]
    )
    examples = joined[joined["is_model_error"].eq(1)].sort_values(
        ["rule_id", "hardness"], ascending=[True, False]
    ).groupby("rule_id", as_index=False).head(args.examples_per_rule)
    examples = examples.merge(inputs, on="pair_id", validate="many_to_one")
    examples[
        [
            "rule_id", "concept", "relation", "catalog_tier", "pair_id", "category",
            "human_label", "score", "hardness", "is_false_positive", "is_false_negative",
            "semantic_fact_count", "value_a", "value_b", "title_a", "title_b",
        ]
    ].to_csv(output / "representative_human_errors.csv", index=False, encoding="utf-8-sig")

    covered_pairs = pairs[pairs["pair_id"].isin(assignments["pair_id"].unique())].copy()
    pair_error = (
        (covered_pairs["human_label"].eq(0) & covered_pairs["score"].ge(args.decision_threshold))
        | (covered_pairs["human_label"].eq(1) & covered_pairs["score"].lt(args.decision_threshold))
    )
    summary = {
        "source_model": "MiniLM S2 values-only baseline OOF, 3 folds",
        "human_labels_only": True,
        "synthetic_pairs_used": False,
        "qwen_labels_used": False,
        "qwen_role": "label-free semantic rule assignment only",
        "ordinary_hard_ood_used_for_rule_selection": False,
        "all_oof_human_pairs": int(len(oof)),
        "qwen_semantic_covered_pairs": int(len(covered_pairs)),
        "covered_positive_pairs": int(covered_pairs["human_label"].sum()),
        "covered_negative_pairs": int(covered_pairs["human_label"].eq(0).sum()),
        "diagnostic_threshold": args.decision_threshold,
        "covered_false_positives": int(
            (covered_pairs["human_label"].eq(0) & covered_pairs["score"].ge(args.decision_threshold)).sum()
        ),
        "covered_false_negatives": int(
            (covered_pairs["human_label"].eq(1) & covered_pairs["score"].lt(args.decision_threshold)).sum()
        ),
        "covered_model_errors": int(pair_error.sum()),
        "covered_oof_hard_pairs": int(covered_pairs["is_mined_hard"].sum()),
        "rules_observed": int(rule_stats["rule_id"].nunique()),
        "rare_safe_rules_observed": int(rare_stats["rule_id"].nunique()),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    top = rule_stats[
        ["concept", "catalog_tier", "human_pairs", "false_positive_pairs", "false_negative_pairs", "oof_hard_pairs"]
    ].head(15).rename(
        columns={
            "concept": "концепт",
            "catalog_tier": "каталог",
            "human_pairs": "human-пар",
            "false_positive_pairs": "FP",
            "false_negative_pairs": "FN",
            "oof_hard_pairs": "OOF-hard",
        }
    )
    rare_view = rare_stats[
        [
            "concept", "human_negative_pairs", "false_positive_pairs", "hard_negative_pairs",
            "clean_negative_pairs", "human_training_priority",
        ]
    ].rename(
        columns={
            "concept": "редкое правило",
            "human_negative_pairs": "human negative",
            "false_positive_pairs": "FP MiniLM",
            "hard_negative_pairs": "OOF-hard negative",
            "clean_negative_pairs": "clean human negative",
            "human_training_priority": "рекомендация",
        }
    )
    report = f"""# Ошибки MiniLM по semantic rules — human-only

## Что сделано

OOF-предсказания MiniLM S2 сопоставлены с label-free semantic assignments Qwen.
Целевые значения везде взяты только из human-разметки. Синтетика и Qwen-labels
не использовались; ordinary, hard и OOD не участвовали в выборе правил.

Из {summary['all_oof_human_pairs']} OOF human-пар semantic extraction покрывает
{summary['qwen_semantic_covered_pairs']} пар. При диагностическом пороге
{args.decision_threshold}: FP = {summary['covered_false_positives']},
FN = {summary['covered_false_negatives']}, OOF-hard = {summary['covered_oof_hard_pairs']}.

## Самые частые ошибки по правилам

{markdown_table(top)}

## Готовые редкие правила: достаточно ли human-примеров

{markdown_table(rare_view)}

`USE_EXISTING_HUMAN_ERRORS_FIRST` означает, что для первого эксперимента уже есть
минимум пять реальных FP MiniLM: сначала следует дообучаться на них, а не генерировать.
`USE_EXISTING_HUMAN_HARD` означает, что ошибок меньше, но есть минимум пять трудных
human-negative. `HUMAN_DATA_SCARCE` — меньше десяти human-negative в доступных 60k.

## Следующий эксперимент

Сформировать только из human-разметки небольшой targeted train subset: реальные
OOF-ошибки и OOF-hard пары по приоритетным правилам. Затем продолжить fine-tuning
из весов frozen MiniLM S2 baseline и сравнить с тем же baseline на неизменённых
IID/hard/OOD. Qwen может отдельно объяснить выбранные ошибки, но не задаёт label.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
