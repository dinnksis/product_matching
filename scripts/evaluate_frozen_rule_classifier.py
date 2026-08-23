"""Fit a discovery-only interpretable rule classifier and score validation.

Validation labels are deliberately loaded only after validation probabilities
have been computed and written to disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DISCOVERY_PAIRS = (
    ROOT
    / "artifacts"
    / "qwen_semantic_extraction_v1_3_sanitized_checkpoint_60000"
    / "sanitized_pairs.parquet"
)
DEFAULT_DISCOVERY_ASSIGNMENTS = (
    ROOT / "reports" / "qwen_rules_checkpoint_60000" / "rule_assignments_label_free.parquet"
)
DEFAULT_DISCOVERY_LABELS = ROOT / "data" / "qwen_rule_discovery_full_v1" / "pilot_labels.parquet"
DEFAULT_FROZEN = ROOT / "artifacts" / "qwen_rules_frozen_v1_60000" / "frozen_rule_definitions.parquet"
DEFAULT_VALIDATION_INPUTS = (
    ROOT / "data" / "qwen_rule_internal_validation_sample3000_v1" / "validation_inputs.parquet"
)
DEFAULT_VALIDATION_ASSIGNMENTS = (
    ROOT
    / "reports"
    / "qwen_rule_internal_validation_sample3000_v1"
    / "validation_rule_assignments_label_free.parquet"
)
DEFAULT_VALIDATION_LABELS = (
    ROOT / "data" / "qwen_rule_internal_validation_sample3000_v1" / "validation_labels.parquet"
)
DEFAULT_OUTPUT = ROOT / "reports" / "qwen_rule_classifier_sample3000_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen-rule logistic baseline without validation leakage."
    )
    parser.add_argument("--discovery-pairs", type=Path, default=DEFAULT_DISCOVERY_PAIRS)
    parser.add_argument("--discovery-assignments", type=Path, default=DEFAULT_DISCOVERY_ASSIGNMENTS)
    parser.add_argument("--discovery-labels", type=Path, default=DEFAULT_DISCOVERY_LABELS)
    parser.add_argument("--frozen-rules", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--validation-inputs", type=Path, default=DEFAULT_VALIDATION_INPUTS)
    parser.add_argument("--validation-assignments", type=Path, default=DEFAULT_VALIDATION_ASSIGNMENTS)
    parser.add_argument("--validation-labels", type=Path, default=DEFAULT_VALIDATION_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-discovery-support", type=int, default=10)
    parser.add_argument("--regularization-c", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2032)
    return parser.parse_args()


def build_matrix(
    pairs: pd.DataFrame,
    assignments: pd.DataFrame,
    categories: list[str],
    rule_ids: list[str],
) -> tuple[sparse.csr_matrix, np.ndarray]:
    pair_index = pd.Series(np.arange(len(pairs)), index=pairs["pair_id"].astype(str))
    category_index = {value: index for index, value in enumerate(categories)}
    rule_index = {value: index for index, value in enumerate(rule_ids)}

    category_rows: list[int] = []
    category_columns: list[int] = []
    for row_index, category in enumerate(pairs["category"].astype(str)):
        if category in category_index:
            category_rows.append(row_index)
            category_columns.append(category_index[category])
    category_matrix = sparse.coo_matrix(
        (
            np.ones(len(category_rows), dtype=np.float32),
            (category_rows, category_columns),
        ),
        shape=(len(pairs), len(categories)),
    ).tocsr()

    selected = assignments[
        assignments["rule_id"].astype(str).isin(rule_index)
        & assignments["pair_id"].astype(str).isin(pair_index.index)
    ][["pair_id", "rule_id"]].drop_duplicates()
    rule_rows = selected["pair_id"].astype(str).map(pair_index).to_numpy(int)
    rule_columns = selected["rule_id"].astype(str).map(rule_index).to_numpy(int)
    rule_matrix = sparse.coo_matrix(
        (
            np.ones(len(selected), dtype=np.float32),
            (rule_rows, rule_columns),
        ),
        shape=(len(pairs), len(rule_ids)),
    ).tocsr()
    return sparse.hstack([category_matrix, rule_matrix], format="csr"), np.asarray(
        rule_matrix.getnnz(axis=1)
    )


def metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    prediction = (probability >= 0.5).astype(int)
    matrix = confusion_matrix(labels, prediction, labels=[0, 1])
    return {
        "roc_auc": float(roc_auc_score(labels, probability)),
        "average_precision": float(average_precision_score(labels, probability)),
        "log_loss": float(log_loss(labels, probability)),
        "brier": float(brier_score_loss(labels, probability)),
        "accuracy_at_0_5": float(accuracy_score(labels, prediction)),
        "precision_at_0_5": float(precision_score(labels, prediction, zero_division=0)),
        "recall_at_0_5": float(recall_score(labels, prediction, zero_division=0)),
        "f1_at_0_5": float(f1_score(labels, prediction, zero_division=0)),
        "confusion_matrix_tn_fp_fn_tp": [
            int(matrix[0, 0]), int(matrix[0, 1]), int(matrix[1, 0]), int(matrix[1, 1])
        ],
    }


def main() -> None:
    args = parse_args()
    if args.min_discovery_support < 1 or args.regularization_c <= 0:
        raise ValueError("min-discovery-support and regularization-c must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    discovery_pairs = pd.read_parquet(
        args.discovery_pairs.resolve(), columns=["pair_id", "category"]
    )
    discovery_pairs["pair_id"] = discovery_pairs["pair_id"].astype(str)
    discovery_assignments = pd.read_parquet(
        args.discovery_assignments.resolve(), columns=["pair_id", "rule_id"]
    )
    frozen = pd.read_parquet(args.frozen_rules.resolve())
    selected_rules = frozen[frozen["global_support"].ge(args.min_discovery_support)].copy()
    rule_ids = sorted(selected_rules["rule_id"].astype(str).tolist())
    categories = sorted(discovery_pairs["category"].astype(str).unique().tolist())

    # Discovery labels are allowed for fitting the model.
    discovery_labels = pd.read_parquet(args.discovery_labels.resolve())
    discovery_labels["pair_id"] = discovery_labels["pair_id"].astype(str)
    discovery = discovery_pairs.merge(
        discovery_labels, on="pair_id", how="left", validate="one_to_one"
    )
    if discovery["human_label"].isna().any():
        raise RuntimeError("Missing RULE_DISCOVERY labels")
    discovery_y = discovery["human_label"].astype(int).to_numpy()
    discovery_x, discovery_rule_count = build_matrix(
        discovery, discovery_assignments, categories, rule_ids
    )

    classifier = LogisticRegression(
        C=args.regularization_c,
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
        random_state=args.seed,
    )
    classifier.fit(discovery_x, discovery_y)

    # Build validation features and probabilities before loading validation labels.
    validation = pd.read_parquet(
        args.validation_inputs.resolve(), columns=["pair_id", "category"]
    )
    validation["pair_id"] = validation["pair_id"].astype(str)
    validation_assignments = pd.read_parquet(
        args.validation_assignments.resolve(), columns=["pair_id", "rule_id"]
    )
    validation_x, validation_rule_count = build_matrix(
        validation, validation_assignments, categories, rule_ids
    )
    probability = classifier.predict_proba(validation_x)[:, 1]
    label_free_predictions = validation.copy()
    label_free_predictions["selected_rule_count"] = validation_rule_count
    label_free_predictions["match_probability"] = probability
    label_free_predictions["predicted_label_at_0_5"] = (probability >= 0.5).astype(int)
    label_free_predictions.to_parquet(
        output / "validation_predictions_before_labels.parquet", index=False
    )

    # Category-only Jeffreys baseline is also learned exclusively on discovery.
    category_stats = discovery.groupby("category")["human_label"].agg(["size", "sum"])
    category_probability = (
        (category_stats["sum"] + 0.5) / (category_stats["size"] + 1)
    ).to_dict()
    baseline_probability = validation["category"].map(category_probability).to_numpy(float)

    # Validation labels are opened only here, after both probability vectors exist.
    validation_labels = pd.read_parquet(args.validation_labels.resolve())
    validation_labels["pair_id"] = validation_labels["pair_id"].astype(str)
    scored = label_free_predictions.merge(
        validation_labels, on="pair_id", how="left", validate="one_to_one"
    )
    if scored["human_label"].isna().any():
        raise RuntimeError("Missing validation labels")
    validation_y = scored["human_label"].astype(int).to_numpy()
    scored["category_baseline_probability"] = baseline_probability
    scored.to_parquet(output / "validation_predictions_scored.parquet", index=False)
    scored.to_csv(
        output / "validation_predictions_scored.csv", index=False, encoding="utf-8-sig"
    )

    coefficients = pd.DataFrame(
        {
            "feature": [f"category::{value}" for value in categories]
            + [f"rule::{value}" for value in rule_ids],
            "coefficient": classifier.coef_[0],
        }
    )
    rule_coefficients = coefficients[coefficients["feature"].str.startswith("rule::")].copy()
    rule_coefficients["rule_id"] = rule_coefficients["feature"].str.removeprefix("rule::")
    rule_coefficients = rule_coefficients.merge(
        selected_rules[
            [
                "rule_id", "canonical_rule", "relation", "rule_role", "global_support",
                "discovery_effect_class", "scope_candidate_frozen",
            ]
        ],
        on="rule_id",
        validate="one_to_one",
    ).drop(columns="feature")
    rule_coefficients.to_csv(
        output / "rule_model_coefficients.csv", index=False, encoding="utf-8-sig"
    )

    summary = {
        "method": "L2 logistic regression: category one-hot + frozen semantic rule incidence",
        "training_source": "60000 RULE_DISCOVERY pairs only",
        "validation_pairs": len(validation),
        "selected_rules": len(rule_ids),
        "min_discovery_support": args.min_discovery_support,
        "regularization_c": args.regularization_c,
        "discovery_pairs_with_selected_rule": int((discovery_rule_count > 0).sum()),
        "validation_pairs_with_selected_rule": int((validation_rule_count > 0).sum()),
        "validation_prevalence": float(validation_y.mean()),
        "category_only_baseline": metrics(validation_y, baseline_probability),
        "rule_classifier": metrics(validation_y, probability),
        "validation_labels_used_for_training_or_threshold_selection": False,
        "threshold": 0.5,
        "ordinary_hard_ood_used": False,
    }
    (output / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rule_metrics = summary["rule_classifier"]
    baseline_metrics = summary["category_only_baseline"]
    report = f"""# Быстрая проверка предсказания label по frozen rules

Модель обучена только на 60k `RULE_DISCOVERY`: L2 logistic regression с
категорией и бинарными признаками замороженных rules с discovery support не
меньше {args.min_discovery_support}. Validation labels не использовались для
обучения, отбора features или выбора threshold.

| метрика | category-only baseline | frozen-rule classifier |
| --- | ---: | ---: |
| ROC-AUC | {baseline_metrics['roc_auc']:.4f} | {rule_metrics['roc_auc']:.4f} |
| Average precision | {baseline_metrics['average_precision']:.4f} | {rule_metrics['average_precision']:.4f} |
| Log loss | {baseline_metrics['log_loss']:.4f} | {rule_metrics['log_loss']:.4f} |
| Brier | {baseline_metrics['brier']:.4f} | {rule_metrics['brier']:.4f} |
| Accuracy, threshold=0.5 | {baseline_metrics['accuracy_at_0_5']:.4f} | {rule_metrics['accuracy_at_0_5']:.4f} |
| F1, threshold=0.5 | {baseline_metrics['f1_at_0_5']:.4f} | {rule_metrics['f1_at_0_5']:.4f} |

Это диагностический интерпретируемый baseline, а не финальный matcher. Его
коэффициенты учитывают совместное появление rules, но не доказывают причинность
каждого отдельного difference. Ordinary, hard и OOD не использовались.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
