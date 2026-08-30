#!/usr/bin/env python3
"""Build a prediction-independent clean-hard benchmark from frozen predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.serialization_ablation import normalize_text, parse_attributes


DEFAULT_MINILM_ROOT = (
    ROOT
    / "artifacts/kaggle/product-matching-minilm-s0-s2-new-splits"
    / "minilm_s0_s2_new_splits/evaluations"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, default=ROOT / "data/items_human.parquet")
    parser.add_argument(
        "--hard-diagnostics",
        type=Path,
        default=ROOT / "reports/minilm_s2_hard_audit/hard_diagnostics.parquet",
    )
    parser.add_argument(
        "--s0-evaluations", type=Path, default=DEFAULT_MINILM_ROOT / "S0_TITLE"
    )
    parser.add_argument(
        "--s2-evaluations", type=Path, default=DEFAULT_MINILM_ROOT / "S2_VALUES_ONLY"
    )
    parser.add_argument(
        "--catboost-evaluations",
        type=Path,
        default=ROOT / "artifacts/s2_catboost_new_splits_local_smoke",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports/minilm_s2_hard_clean_audit",
    )
    parser.add_argument("--examples-per-issue", type=int, default=100)
    parser.add_argument("--very-high-title-jaccard", type=float, default=0.90)
    parser.add_argument("--very-low-title-jaccard", type=float, default=0.20)
    return parser.parse_args()


def canonical_pair_columns(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    left = frame["id1"].to_numpy(dtype=np.int64)
    right = frame["id2"].to_numpy(dtype=np.int64)
    return np.minimum(left, right), np.maximum(left, right)


def add_canonical_pair(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["canonical_id1"], result["canonical_id2"] = canonical_pair_columns(result)
    return result


def validate_prediction_frame(frame: pd.DataFrame, name: str) -> None:
    required = {"id1", "id2", "target", "category"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    if not frame["target"].isin([0, 1]).all():
        raise ValueError(f"{name}: target is not binary")


def align_score(
    base: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    source_column: str,
    output_column: str,
    name: str,
) -> pd.Series:
    validate_prediction_frame(predictions, name)
    source = add_canonical_pair(predictions)
    base_left = base["canonical_id1"].to_numpy(dtype=np.int64)
    base_right = base["canonical_id2"].to_numpy(dtype=np.int64)
    if (
        len(base) == len(source)
        and np.array_equal(base_left, source["canonical_id1"].to_numpy(dtype=np.int64))
        and np.array_equal(base_right, source["canonical_id2"].to_numpy(dtype=np.int64))
        and np.array_equal(
            base["target"].to_numpy(dtype=np.int8),
            source["target"].to_numpy(dtype=np.int8),
        )
        and base["category"].astype(str).reset_index(drop=True).equals(
            source["category"].astype(str).reset_index(drop=True)
        )
    ):
        return source[source_column].reset_index(drop=True).astype(np.float32)
    if base.duplicated(["canonical_id1", "canonical_id2"]).any() or source.duplicated(
        ["canonical_id1", "canonical_id2"]
    ).any():
        raise ValueError(
            f"{name}: duplicate unordered ID pairs require identical row order for alignment"
        )
    source = source[
        ["canonical_id1", "canonical_id2", "target", "category", source_column]
    ].rename(
        columns={
            "target": "source_target",
            "category": "source_category",
            source_column: output_column,
        }
    )
    aligned = base[["canonical_id1", "canonical_id2", "target", "category"]].merge(
        source,
        on=["canonical_id1", "canonical_id2"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if aligned[output_column].isna().any():
        raise ValueError(f"{name}: predictions are missing aligned pairs")
    if not np.array_equal(
        aligned["target"].to_numpy(dtype=np.int8),
        aligned["source_target"].to_numpy(dtype=np.int8),
    ):
        raise ValueError(f"{name}: targets differ from the audit frame")
    if not aligned["category"].astype(str).equals(aligned["source_category"].astype(str)):
        raise ValueError(f"{name}: categories differ from the audit frame")
    return aligned[output_column].astype(np.float32)


def normalized_item_representation(row: Any) -> tuple[str, str]:
    title = normalize_text(row.name)
    attributes = tuple(sorted(parse_attributes(row.attributes)))
    payload = json.dumps(
        [normalize_text(row.category), title, attributes],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return payload, digest


def attach_item_context(frame: pd.DataFrame, items_path: Path) -> pd.DataFrame:
    required_ids = set(frame["id1"].tolist()) | set(frame["id2"].tolist())
    items = pd.read_parquet(
        items_path, columns=["id", "name", "attributes", "category"]
    )
    items = items[items["id"].isin(required_ids)].reset_index(drop=True)
    if len(items) != len(required_ids):
        raise ValueError("items file does not contain every hard-test product")
    representations = [
        normalized_item_representation(row) for row in items.itertuples(index=False)
    ]
    items["normalized_title"] = items["name"].map(normalize_text)
    items["full_representation"] = [value[0] for value in representations]
    items["full_representation_sha256"] = [value[1] for value in representations]
    left = items.rename(
        columns={
            "id": "id1",
            "name": "title1",
            "attributes": "attributes1",
            "category": "item_category1",
            "normalized_title": "normalized_title1",
            "full_representation": "full_representation1",
            "full_representation_sha256": "full_representation_sha256_1",
        }
    )
    right = items.rename(
        columns={
            "id": "id2",
            "name": "title2",
            "attributes": "attributes2",
            "category": "item_category2",
            "normalized_title": "normalized_title2",
            "full_representation": "full_representation2",
            "full_representation_sha256": "full_representation_sha256_2",
        }
    )
    result = frame.merge(left, on="id1", how="left", validate="many_to_one")
    result = result.merge(right, on="id2", how="left", validate="many_to_one")
    if not result["category"].astype(str).equals(result["item_category1"].astype(str)):
        raise ValueError("left item category differs from pair category")
    if not result["category"].astype(str).equals(result["item_category2"].astype(str)):
        raise ValueError("right item category differs from pair category")
    return result.drop(columns=["item_category1", "item_category2"])


def add_label_audit_flags(
    frame: pd.DataFrame,
    *,
    high_jaccard: float,
    low_jaccard: float,
) -> pd.DataFrame:
    result = frame.copy()
    result["exact_normalized_title_match"] = result["normalized_title1"].eq(
        result["normalized_title2"]
    )
    result["identical_full_representation"] = result[
        "full_representation_sha256_1"
    ].eq(result["full_representation_sha256_2"])
    result["numeric_conflict"] = result["numeric_context_conflict_count"].gt(0)
    result["code_conflict"] = result["code_conflict"].astype(bool)
    result["model_code_conflict"] = result["model_conflict"].astype(bool)
    result["critical_attribute_conflict"] = result[
        ["brand_conflict", "model_conflict", "memory_conflict", "color_conflict"]
    ].astype(bool).any(axis=1)
    result["sku_vs_human_title"] = result["sku_human_asymmetry"].astype(bool)

    id_group = result.groupby(["canonical_id1", "canonical_id2"], sort=False)[
        "target"
    ].transform("nunique")
    result["unordered_id_pair_target_conflict"] = id_group.gt(1)
    rep_left = result["full_representation_sha256_1"].to_numpy(dtype=str)
    rep_right = result["full_representation_sha256_2"].to_numpy(dtype=str)
    left_first = rep_left <= rep_right
    result["canonical_representation1"] = np.where(left_first, rep_left, rep_right)
    result["canonical_representation2"] = np.where(left_first, rep_right, rep_left)
    representation_group = result.groupby(
        ["canonical_representation1", "canonical_representation2"], sort=False
    )["target"].transform("nunique")
    result["representation_pair_target_conflict"] = representation_group.gt(1)
    result["definite_label_conflict"] = result[
        ["unordered_id_pair_target_conflict", "representation_pair_target_conflict"]
    ].any(axis=1)

    negative = result["target"].eq(0)
    positive = result["target"].eq(1)
    result["suspicious_negative_identity"] = negative & result[
        ["exact_normalized_title_match", "identical_full_representation"]
    ].any(axis=1)
    result["suspicious_positive_conflict"] = positive & result[
        [
            "numeric_conflict",
            "code_conflict",
            "model_code_conflict",
            "critical_attribute_conflict",
        ]
    ].any(axis=1)
    result["strong_label_suspicion"] = result[
        ["suspicious_negative_identity", "suspicious_positive_conflict"]
    ].any(axis=1)
    result["hard_subset"] = np.select(
        [result["definite_label_conflict"], result["strong_label_suspicion"]],
        ["hard_conflicting", "hard_suspicious"],
        default="hard_clean",
    )
    result["very_high_lexical_similarity_negative"] = negative & result[
        "title_token_jaccard"
    ].ge(high_jaccard)
    result["very_low_lexical_similarity_positive"] = positive & result[
        "title_token_jaccard"
    ].le(low_jaccard)
    return result


def metric_values(frame: pd.DataFrame, score_column: str) -> dict[str, Any]:
    category_aps: dict[str, float] = {}
    category_rocs: dict[str, float] = {}
    for category, part in frame.groupby("category", sort=True):
        if part["target"].nunique() != 2:
            continue
        category_aps[str(category)] = float(
            average_precision_score(part["target"], part[score_column])
        )
        category_rocs[str(category)] = float(
            roc_auc_score(part["target"], part[score_column])
        )
    has_both = frame["target"].nunique() == 2
    positive = frame["target"].eq(1)
    negative = frame["target"].eq(0)
    return {
        "eligible_categories": len(category_aps),
        "macro_average_precision": (
            float(np.mean(list(category_aps.values()))) if category_aps else None
        ),
        "macro_roc_auc": (
            float(np.mean(list(category_rocs.values()))) if category_rocs else None
        ),
        "overall_average_precision": (
            float(average_precision_score(frame["target"], frame[score_column]))
            if has_both
            else None
        ),
        "overall_roc_auc": (
            float(roc_auc_score(frame["target"], frame[score_column]))
            if has_both
            else None
        ),
        "positive_mean_score": (
            float(frame.loc[positive, score_column].mean()) if positive.any() else None
        ),
        "negative_mean_score": (
            float(frame.loc[negative, score_column].mean()) if negative.any() else None
        ),
        "fnr_at_0_5": (
            float(frame.loc[positive, score_column].lt(0.5).mean())
            if positive.any()
            else None
        ),
        "fpr_at_0_5": (
            float(frame.loc[negative, score_column].ge(0.5).mean())
            if negative.any()
            else None
        ),
    }


def metric_rows(frame: pd.DataFrame, subset: str) -> list[dict[str, Any]]:
    rows = []
    for model, score_column in (
        ("S0", "s0_score"),
        ("S2", "s2_score"),
        ("S2_CATBOOST", "catboost_score"),
    ):
        rows.append(
            {
                "subset": subset,
                "model": model,
                "pairs": len(frame),
                "positives": int(frame["target"].sum()),
                "prevalence": float(frame["target"].mean()) if len(frame) else None,
                "categories": int(frame["category"].nunique()),
                **metric_values(frame, score_column),
            }
        )
    return rows


def load_split_with_scores(args: argparse.Namespace, split: str) -> pd.DataFrame:
    s2 = pd.read_parquet(args.s2_evaluations / split / "predictions.parquet")
    validate_prediction_frame(s2, f"S2 {split}")
    result = add_canonical_pair(s2[["id1", "id2", "target", "category"]])
    result["s2_score"] = s2["score"].to_numpy(dtype=np.float32)
    s0 = pd.read_parquet(args.s0_evaluations / split / "predictions.parquet")
    catboost = pd.read_parquet(args.catboost_evaluations / split / "predictions.parquet")
    result["s0_score"] = align_score(
        result, s0, source_column="score", output_column="s0_score", name=f"S0 {split}"
    ).to_numpy()
    result["catboost_score"] = align_score(
        result,
        catboost,
        source_column="catboost_score",
        output_column="catboost_score",
        name=f"CatBoost {split}",
    ).to_numpy()
    return result


def subset_summary(frame: pd.DataFrame, subset: str) -> dict[str, Any]:
    return {
        "subset": subset,
        "pairs": len(frame),
        "positives": int(frame["target"].sum()),
        "prevalence": float(frame["target"].mean()) if len(frame) else None,
        "categories": int(frame["category"].nunique()),
        "definite_label_conflicts": int(frame["definite_label_conflict"].sum()),
        "unordered_id_pair_target_conflicts": int(
            frame["unordered_id_pair_target_conflict"].sum()
        ),
        "representation_pair_target_conflicts": int(
            frame["representation_pair_target_conflict"].sum()
        ),
        "suspicious_negative_identity": int(
            frame["suspicious_negative_identity"].sum()
        ),
        "suspicious_positive_conflict": int(
            frame["suspicious_positive_conflict"].sum()
        ),
        "sku_vs_human_title": int(frame["sku_vs_human_title"].sum()),
    }


def issue_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    negative = frame["target"].eq(0)
    positive = frame["target"].eq(1)
    return {
        "definite_label_conflict": frame["definite_label_conflict"],
        "negative_identical_full_representation": negative
        & frame["identical_full_representation"],
        "negative_exact_normalized_title": negative
        & frame["exact_normalized_title_match"],
        "positive_numeric_conflict": positive & frame["numeric_conflict"],
        "positive_code_conflict": positive & frame["code_conflict"],
        "positive_model_code_conflict": positive & frame["model_code_conflict"],
        "positive_critical_attribute_conflict": positive
        & frame["critical_attribute_conflict"],
        "sku_vs_human_title": frame["sku_vs_human_title"],
        "very_high_lexical_similarity_negative": frame[
            "very_high_lexical_similarity_negative"
        ],
        "very_low_lexical_similarity_positive": frame[
            "very_low_lexical_similarity_positive"
        ],
    }


def example_table(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    records = []
    severity_columns = [
        "definite_label_conflict",
        "identical_full_representation",
        "exact_normalized_title_match",
        "numeric_conflict",
        "code_conflict",
        "model_code_conflict",
        "critical_attribute_conflict",
        "sku_vs_human_title",
    ]
    severity = frame[severity_columns].astype(int).sum(axis=1)
    ranked = frame.assign(_audit_severity=severity).sort_values(
        ["_audit_severity", "title_token_jaccard", "canonical_id1", "canonical_id2"],
        ascending=[False, False, True, True],
    )
    for issue, mask in issue_masks(ranked).items():
        part = ranked.loc[mask].head(limit).copy()
        part.insert(0, "audit_issue", issue)
        records.append(part)
    if not records:
        return frame.head(0).copy()
    result = pd.concat(records, ignore_index=True)
    return result.drop(columns=["_audit_severity"], errors="ignore")


def clean_slice_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    masks = {
        "numeric_conflict": frame["numeric_conflict"],
        "code_conflict": frame["code_conflict"],
        "model_conflict": frame["model_code_conflict"],
        "sku_vs_human_title": frame["sku_vs_human_title"],
        "very_high_lexical_similarity_negatives": frame[
            "very_high_lexical_similarity_negative"
        ],
        "very_low_lexical_similarity_positives": frame[
            "very_low_lexical_similarity_positive"
        ],
    }
    rows = []
    for slice_name, mask in masks.items():
        part = frame.loc[mask]
        for row in metric_rows(part, f"hard_clean::{slice_name}"):
            row["slice"] = slice_name
            rows.append(row)
    return rows


def write_report(
    path: Path,
    subset_summary_frame: pd.DataFrame,
    metrics: pd.DataFrame,
    flag_summary: pd.DataFrame,
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        display = frame.copy()
        for column in display.select_dtypes(include=["float"]).columns:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.6f}"
            )
        headers = [str(column) for column in display.columns]
        rows = [headers, ["---"] * len(headers)]
        rows.extend(
            [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
            for row in display.itertuples(index=False, name=None)
        )
        return "\n".join("| " + " | ".join(row) + " |" for row in rows)

    pivot = metrics.pivot(index="subset", columns="model", values="macro_average_precision")
    lines = [
        "# Clean-hard label audit",
        "",
        "Hard subsets are selected only from targets, canonical duplicates, and fixed",
        "lexical/attribute conflict flags. Model predictions never participate in selection.",
        "",
        "## Subsets",
        "",
        markdown_table(subset_summary_frame),
        "",
        "## Competition macro AP",
        "",
        markdown_table(pivot.reset_index()),
        "",
        "## Flag coverage",
        "",
        markdown_table(flag_summary),
        "",
        "AP values from subsets with different prevalence are not directly comparable.",
        "The reliable comparison is between models within the same subset.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hard = pd.read_parquet(args.hard_diagnostics).reset_index(drop=True)
    hard = add_canonical_pair(hard)
    s0_hard = pd.read_parquet(args.s0_evaluations / "hard/predictions.parquet")
    hard["s0_score"] = align_score(
        hard,
        s0_hard,
        source_column="score",
        output_column="s0_score",
        name="S0 hard",
    ).to_numpy()
    hard = attach_item_context(hard, args.items)
    hard = add_label_audit_flags(
        hard,
        high_jaccard=args.very_high_title_jaccard,
        low_jaccard=args.very_low_title_jaccard,
    )

    legacy_title_exact = hard["title_exact"].astype(bool)
    normalized_title_disagreements = int(
        legacy_title_exact.ne(hard["exact_normalized_title_match"]).sum()
    )

    output_columns = [
        "id1",
        "id2",
        "canonical_id1",
        "canonical_id2",
        "target",
        "category",
        "hard_subset",
        "title1",
        "title2",
        "normalized_title1",
        "normalized_title2",
        "attributes1",
        "attributes2",
        "full_representation_sha256_1",
        "full_representation_sha256_2",
        "exact_normalized_title_match",
        "numeric_conflict",
        "code_conflict",
        "model_code_conflict",
        "critical_attribute_conflict",
        "sku_vs_human_title",
        "identical_full_representation",
        "unordered_id_pair_target_conflict",
        "representation_pair_target_conflict",
        "definite_label_conflict",
        "suspicious_negative_identity",
        "suspicious_positive_conflict",
        "strong_label_suspicion",
        "very_high_lexical_similarity_negative",
        "very_low_lexical_similarity_positive",
        "title_token_jaccard",
        "title_char_tfidf_cosine",
        "numeric_context_conflict_count",
        "unit_conflict_count",
        "brand_conflict",
        "model_conflict",
        "memory_conflict",
        "color_conflict",
        "s0_score",
        "s2_score",
        "catboost_score",
    ]
    for subset in ("hard_clean", "hard_suspicious", "hard_conflicting"):
        hard.loc[hard["hard_subset"].eq(subset), output_columns].to_csv(
            args.output_dir / f"{subset}.csv", index=False
        )

    summaries = [subset_summary(hard, "hard_all")]
    for subset in ("hard_clean", "hard_suspicious", "hard_conflicting"):
        summaries.append(
            subset_summary(hard.loc[hard["hard_subset"].eq(subset)], subset)
        )
    subset_summary_frame = pd.DataFrame(summaries)
    subset_summary_frame.to_csv(args.output_dir / "label_audit_summary.csv", index=False)

    metrics_rows = metric_rows(hard, "hard_all")
    for subset in ("hard_clean", "hard_suspicious", "hard_conflicting"):
        metrics_rows.extend(
            metric_rows(hard.loc[hard["hard_subset"].eq(subset)], subset)
        )
    for split in ("iid", "ood"):
        metrics_rows.extend(metric_rows(load_split_with_scores(args, split), split))
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(args.output_dir / "benchmark_metrics.csv", index=False)

    clean = hard.loc[hard["hard_subset"].eq("hard_clean")]
    clean_slices = pd.DataFrame(clean_slice_rows(clean))
    clean_slices.to_csv(args.output_dir / "hard_clean_slice_metrics.csv", index=False)

    examples = example_table(hard, args.examples_per_issue)
    example_columns = ["audit_issue", *output_columns]
    examples[example_columns].to_csv(
        args.output_dir / "label_contradictions.csv", index=False
    )
    flag_summary = pd.DataFrame(
        [
            {
                "flag": flag,
                "support": int(mask.sum()),
                "positive_count": int(hard.loc[mask, "target"].sum()),
                "prevalence": float(hard.loc[mask, "target"].mean())
                if mask.any()
                else None,
            }
            for flag, mask in issue_masks(hard).items()
        ]
    )
    flag_summary.to_csv(args.output_dir / "label_audit_flag_summary.csv", index=False)

    write_report(args.output_dir / "report.md", subset_summary_frame, metrics, flag_summary)
    audit_report = {
        "experiment": "minilm_s0_s2_catboost_clean_hard_audit",
        "selection_uses_predictions": False,
        "hard_pairs": len(hard),
        "subsets": subset_summary_frame.to_dict("records"),
        "normalized_title_flag_disagreements_with_existing_title_exact": (
            normalized_title_disagreements
        ),
        "thresholds": {
            "very_high_title_token_jaccard": args.very_high_title_jaccard,
            "very_low_title_token_jaccard": args.very_low_title_jaccard,
        },
        "classification_policy": {
            "conflicting": (
                "opposite targets for the same unordered ID pair or the same unordered "
                "pair of normalized full item representations"
            ),
            "suspicious": (
                "non-conflicting negative with exact normalized title/full representation, "
                "or non-conflicting positive with numeric/code/model/critical-attribute conflict"
            ),
            "clean": "neither conflicting nor strongly suspicious",
            "sku_vs_human_title": (
                "reported as a secondary audit flag but not used alone to exclude a pair"
            ),
        },
        "metrics": metrics.to_dict("records"),
    }
    (args.output_dir / "audit_report.json").write_text(
        json.dumps(audit_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "COMPLETED").write_text("complete\n", encoding="utf-8")
    print(json.dumps(audit_report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
