"""Prepare a human-only MiniLM error curriculum with Qwen semantic explanations.

Every target is an existing human label from RULE_DISCOVERY.  Qwen outputs are
attached only as label-free structured explanations.  Suspicious/conflicting
human rows are excluded using the frozen OOF audit flags.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OOF = (
    ROOT
    / "artifacts/kaggle/product-matching-minilm-s2-targeted-hard-results2"
    / "minilm_s2_targeted_hard/mining/oof_predictions_and_hardness.parquet"
)
DEFAULT_SPLITS = ROOT / "data/rule_discovery_split_v1/split_assignments.parquet"
DEFAULT_ASSIGNMENTS = ROOT / "reports/qwen_rules_checkpoint_60000/rule_assignments_label_free.parquet"
DEFAULT_SEMANTICS = (
    ROOT
    / "artifacts/qwen_semantic_extraction_v1_3_sanitized_checkpoint_60000"
    / "sanitized_pairs.parquet"
)
DEFAULT_OUTPUT = ROOT / "data/minilm_human_error_curriculum_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a human-only MiniLM targeted curriculum.")
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--split-assignments", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--rule-assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--semantic-pairs", type=Path, default=DEFAULT_SEMANTICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    return parser.parse_args()


def parse_json_object(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return str(value)


def semantic_signature(group: pd.DataFrame) -> str:
    rows = []
    for row in group.sort_values(["concept", "relation", "rule_id"]).itertuples(index=False):
        rows.append(
            {
                "rule_id": str(row.rule_id),
                "concept": str(row.concept),
                "relation": str(row.relation),
                "value_a": parse_json_object(row.value_a),
                "value_b": parse_json_object(row.value_b),
            }
        )
    return json.dumps(rows, ensure_ascii=False)


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


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    splits = pd.read_parquet(args.split_assignments.resolve()).reset_index(names="row_id")
    oof = pd.read_parquet(args.oof.resolve())
    if len(splits) != len(oof):
        raise RuntimeError("OOF rows and split assignments differ")
    pairs = oof.merge(splits, on="row_id", suffixes=("_oof", "_split"), validate="one_to_one")
    identity_ok = (
        pairs["id1_oof"].astype(str).eq(pairs["id1_split"].astype(str)).all()
        and pairs["id2_oof"].astype(str).eq(pairs["id2_split"].astype(str)).all()
        and pairs["target_oof"].eq(pairs["target_split"]).all()
    )
    if not identity_ok:
        raise RuntimeError("OOF rows do not map one-to-one to human pairs")
    pairs = pairs.rename(
        columns={
            "id1_oof": "id1",
            "id2_oof": "id2",
            "target_oof": "target",
            "category": "category",
        }
    )

    assignments = pd.read_parquet(args.rule_assignments.resolve()).drop_duplicates(
        ["pair_id", "rule_id"]
    )
    covered_pair_ids = set(assignments["pair_id"].astype(str))
    pairs = pairs[pairs["pair_id"].astype(str).isin(covered_pair_ids)].copy()
    if not pairs["split"].eq("rule_discovery").all():
        raise RuntimeError("Targeted set contains a non-RULE_DISCOVERY row")

    pairs["is_false_positive"] = pairs["target"].eq(0) & pairs["score"].ge(
        args.decision_threshold
    )
    pairs["is_false_negative"] = pairs["target"].eq(1) & pairs["score"].lt(
        args.decision_threshold
    )
    pairs["is_model_error"] = pairs["is_false_positive"] | pairs["is_false_negative"]
    pairs["audit_eligible"] = (
        pairs["eligible_for_hard_mining"]
        & ~pairs["definite_label_conflict"]
        & ~pairs["strong_label_suspicion"]
        & ~pairs["duplicate_label_conflict"]
    )

    eligible = pairs[pairs["audit_eligible"]].copy()
    curriculum = eligible[eligible["is_model_error"] | eligible["is_mined_hard"]].copy()
    curriculum["curriculum_group"] = "HARD_ONLY"
    curriculum.loc[curriculum["is_model_error"], "curriculum_group"] = "ERROR_ONLY"
    curriculum.loc[
        curriculum["is_model_error"] & curriculum["is_mined_hard"], "curriculum_group"
    ] = "ERROR_AND_HARD"
    curriculum["error_type"] = "CORRECT_BUT_HARD"
    curriculum.loc[curriculum["is_false_positive"], "error_type"] = "FALSE_POSITIVE"
    curriculum.loc[curriculum["is_false_negative"], "error_type"] = "FALSE_NEGATIVE"
    curriculum["source_split"] = "RULE_DISCOVERY"
    curriculum["label_source"] = "HUMAN"
    curriculum["qwen_label_used"] = False
    curriculum["synthetic"] = False

    curriculum_pair_ids = set(curriculum["pair_id"].astype(str))
    curriculum_assignments = assignments[
        assignments["pair_id"].astype(str).isin(curriculum_pair_ids)
    ].copy()
    signatures = curriculum_assignments.groupby("pair_id", sort=False).apply(
        semantic_signature, include_groups=False
    ).rename("qwen_semantic_signature_json").reset_index()
    semantic_pairs = pd.read_parquet(
        args.semantic_pairs.resolve(),
        columns=[
            "pair_id",
            "identity_anchors_json",
            "differences_json",
            "missing_information_json",
            "pair_summary_json",
            "sanitization_warnings_json",
        ],
    )
    curriculum = curriculum.merge(signatures, on="pair_id", validate="one_to_one").merge(
        semantic_pairs, on="pair_id", validate="one_to_one"
    )

    columns = [
        "pair_id", "id1", "id2", "target", "category", "source_split", "label_source",
        "synthetic", "qwen_label_used", "score", "score_ab", "score_ba", "score_order_gap",
        "hardness", "is_mined_hard", "is_false_positive", "is_false_negative",
        "is_model_error", "error_type", "curriculum_group", "oof_fold",
        "qwen_semantic_signature_json", "identity_anchors_json", "differences_json",
        "missing_information_json", "pair_summary_json", "sanitization_warnings_json",
    ]
    curriculum = curriculum[columns].sort_values(
        ["is_model_error", "hardness", "pair_id"], ascending=[False, False, True]
    )
    errors = curriculum[curriculum["is_model_error"]].copy()
    training_pairs = curriculum[["id1", "id2", "target"]].copy()

    curriculum.to_parquet(output / "human_error_curriculum_with_semantics.parquet", index=False)
    errors.to_parquet(output / "human_errors_with_semantics.parquet", index=False)
    training_pairs.to_parquet(output / "human_error_curriculum_pairs.parquet", index=False)
    curriculum[["pair_id", "curriculum_group", "error_type"]].to_csv(
        output / "human_error_curriculum_pair_ids.csv", index=False, encoding="utf-8-sig"
    )

    exploded = curriculum[["pair_id", "error_type", "category"]].merge(
        curriculum_assignments[["pair_id", "rule_id", "concept", "relation"]],
        on="pair_id",
        validate="one_to_many",
    )
    rule_counts = exploded.groupby(["concept", "relation"]).agg(
        targeted_pairs=("pair_id", "nunique"),
        false_positive_pairs=("error_type", lambda values: int((values == "FALSE_POSITIVE").sum())),
        false_negative_pairs=("error_type", lambda values: int((values == "FALSE_NEGATIVE").sum())),
        hard_only_pairs=("error_type", lambda values: int((values == "CORRECT_BUT_HARD").sum())),
    ).reset_index().sort_values(["false_positive_pairs", "false_negative_pairs"], ascending=False)
    rule_counts.to_csv(output / "curriculum_rule_counts.csv", index=False, encoding="utf-8-sig")
    category_counts = curriculum.groupby(["category", "error_type"]).size().rename("pairs").reset_index()
    category_counts.to_csv(output / "curriculum_category_counts.csv", index=False, encoding="utf-8-sig")

    excluded_errors = pairs[pairs["is_model_error"] & ~pairs["audit_eligible"]]
    excluded_errors[
        [
            "pair_id", "id1", "id2", "target", "score", "hardness",
            "definite_label_conflict", "strong_label_suspicion", "duplicate_label_conflict",
        ]
    ].to_csv(output / "excluded_suspicious_human_errors.csv", index=False, encoding="utf-8-sig")

    summary = {
        "version": "minilm_human_error_curriculum_v1",
        "source_model": "MiniLM S2 values-only baseline OOF, 3 folds",
        "source_split": "RULE_DISCOVERY only",
        "human_labels_only": True,
        "synthetic_pairs_used": False,
        "qwen_labels_used": False,
        "qwen_reasoning_available": False,
        "qwen_structured_semantics_attached": True,
        "ordinary_test_rows": 0,
        "hard_test_rows": 0,
        "ood_test_rows": 0,
        "semantic_covered_human_pairs": int(len(pairs)),
        "audit_eligible_human_pairs": int(len(eligible)),
        "curriculum_unique_pairs": int(len(curriculum)),
        "human_model_errors": int(len(errors)),
        "human_false_positives": int(errors["is_false_positive"].sum()),
        "human_false_negatives": int(errors["is_false_negative"].sum()),
        "correct_but_oof_hard_pairs": int((~curriculum["is_model_error"]).sum()),
        "errors_excluded_as_suspicious_or_conflicting": int(len(excluded_errors)),
        "recommended_training": (
            "Continue fine-tuning from frozen MiniLM S2 baseline on the original human train "
            "with this unique curriculum used only for deterministic oversampling."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    top_rules = rule_counts.head(15).rename(
        columns={
            "concept": "концепт",
            "relation": "relation",
            "targeted_pairs": "пар",
            "false_positive_pairs": "FP",
            "false_negative_pairs": "FN",
            "hard_only_pairs": "корректные OOF-hard",
        }
    )
    report = f"""# Human-only curriculum ошибок MiniLM

## Состав

Собрано **{len(curriculum)} уникальных human-пар** только из `RULE_DISCOVERY`:

- ошибок MiniLM: **{len(errors)}**;
- false positive: **{int(errors['is_false_positive'].sum())}**;
- false negative: **{int(errors['is_false_negative'].sum())}**;
- корректных, но OOF-hard: **{int((~curriculum['is_model_error']).sum())}**.

Ещё **{len(excluded_errors)}** ошибок исключены как подозрительная или конфликтная
human-разметка. Ordinary, существующий hard test и OOD не использовались.

## Что означает «объяснение Qwen»

Thinking/reasoning Qwen был выключен, поэтому скрытой цепочки рассуждений нет.
К каждой паре приложена проверяемая структурированная семантика: identity anchors,
differences, missing information, canonical concepts, relation и значения A/B.
Qwen-label отсутствует; `target` всегда human.

## Частые semantic patterns в curriculum

{markdown_table(top_rules)}

## Как использовать

Начать с весов frozen MiniLM S2 baseline. Полный human train оставить основой,
а строки из `human_error_curriculum_pairs.parquet` только детерминированно
переэкспонировать. Не обучаться только на ошибках: это вызовет забывание простых
примеров и сдвиг prevalence.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
