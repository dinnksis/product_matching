#!/usr/bin/env python3
"""Audit S2 and S2+CatBoost errors on the fixed hard human validation split."""

from __future__ import annotations

import argparse
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

from src.cheap_ensemble import build_pair_features, fit_hashed_char_idf, prepare_item_records
from src.serialization_ablation import parse_attributes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, default=ROOT / "data/items_human.parquet")
    parser.add_argument(
        "--s2-predictions",
        type=Path,
        default=(
            ROOT
            / "artifacts/kaggle/product-matching-minilm-s0-s2-new-splits"
            / "minilm_s0_s2_new_splits/evaluations/S2_VALUES_ONLY/hard/predictions.parquet"
        ),
    )
    parser.add_argument(
        "--catboost-predictions",
        type=Path,
        default=ROOT / "artifacts/s2_catboost_new_splits_local_smoke/hard/predictions.parquet",
    )
    parser.add_argument(
        "--iid-s2-predictions",
        type=Path,
        default=(
            ROOT
            / "artifacts/kaggle/product-matching-minilm-s0-s2-new-splits"
            / "minilm_s0_s2_new_splits/evaluations/S2_VALUES_ONLY/iid/predictions.parquet"
        ),
    )
    parser.add_argument(
        "--iid-catboost-predictions",
        type=Path,
        default=ROOT / "artifacts/s2_catboost_new_splits_local_smoke/iid/predictions.parquet",
    )
    parser.add_argument(
        "--char-idf",
        type=Path,
        default=ROOT / "artifacts/s2_catboost_new_splits_local_smoke/char_idf.npy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports/minilm_s2_hard_audit",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--top-examples", type=int, default=100)
    return parser.parse_args()


def macro_metrics(frame: pd.DataFrame, score_column: str) -> dict[str, Any]:
    per_category_ap: dict[str, float] = {}
    per_category_roc: dict[str, float] = {}
    for category, part in frame.groupby("category", sort=True):
        if part["target"].nunique() != 2:
            continue
        per_category_ap[str(category)] = float(
            average_precision_score(part["target"], part[score_column])
        )
        per_category_roc[str(category)] = float(
            roc_auc_score(part["target"], part[score_column])
        )
    return {
        "eligible_categories": len(per_category_ap),
        "macro_average_precision": (
            float(np.mean(list(per_category_ap.values()))) if per_category_ap else None
        ),
        "macro_roc_auc": (
            float(np.mean(list(per_category_roc.values()))) if per_category_roc else None
        ),
        "overall_average_precision": (
            float(average_precision_score(frame["target"], frame[score_column]))
            if frame["target"].nunique() == 2
            else None
        ),
        "per_category_average_precision": per_category_ap,
    }


def diagnostic_rates(frame: pd.DataFrame, score_column: str) -> dict[str, float | None]:
    positive = frame["target"].eq(1)
    negative = ~positive
    return {
        "positive_mean_score": float(frame.loc[positive, score_column].mean()) if positive.any() else None,
        "negative_mean_score": float(frame.loc[negative, score_column].mean()) if negative.any() else None,
        "fnr_at_0_5": float(frame.loc[positive, score_column].lt(0.5).mean()) if positive.any() else None,
        "fpr_at_0_5": float(frame.loc[negative, score_column].ge(0.5).mean()) if negative.any() else None,
    }


def slice_row(frame: pd.DataFrame, name: str, mask: pd.Series) -> dict[str, Any]:
    part = frame.loc[mask]
    s2 = macro_metrics(part, "s2_score")
    catboost = macro_metrics(part, "catboost_score")
    s2_ap = s2["macro_average_precision"]
    catboost_ap = catboost["macro_average_precision"]
    return {
        "slice": name,
        "support": len(part),
        "positive_count": int(part["target"].sum()),
        "positive_rate": float(part["target"].mean()) if len(part) else None,
        "eligible_categories": s2["eligible_categories"],
        "s2_macro_ap": s2_ap,
        "catboost_macro_ap": catboost_ap,
        "catboost_minus_s2_macro_ap": (
            float(catboost_ap - s2_ap) if s2_ap is not None and catboost_ap is not None else None
        ),
        **{f"s2_{key}": value for key, value in diagnostic_rates(part, "s2_score").items()},
        **{
            f"catboost_{key}": value
            for key, value in diagnostic_rates(part, "catboost_score").items()
        },
    }


def stratified_paired_bootstrap(
    frame: pd.DataFrame, *, samples: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    strata: list[tuple[np.ndarray, np.ndarray]] = []
    for _, part in frame.groupby("category", sort=True):
        positive = part.index[part["target"].eq(1)].to_numpy(dtype=np.int64)
        negative = part.index[part["target"].eq(0)].to_numpy(dtype=np.int64)
        if len(positive) and len(negative):
            strata.append((positive, negative))
    deltas = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        category_deltas = []
        for positive, negative in strata:
            indices = np.concatenate(
                [
                    rng.choice(positive, size=len(positive), replace=True),
                    rng.choice(negative, size=len(negative), replace=True),
                ]
            )
            target = frame.loc[indices, "target"].to_numpy(dtype=np.int8)
            s2 = frame.loc[indices, "s2_score"].to_numpy(dtype=np.float64)
            catboost = frame.loc[indices, "catboost_score"].to_numpy(dtype=np.float64)
            category_deltas.append(
                average_precision_score(target, catboost)
                - average_precision_score(target, s2)
            )
        deltas[sample] = np.mean(category_deltas)
    return {
        "samples": samples,
        "seed": seed,
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "probability_delta_positive": float(np.mean(deltas > 0)),
    }


def category_delta_bootstrap(
    category_deltas: np.ndarray, *, samples: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        category_deltas,
        size=(samples, len(category_deltas)),
        replace=True,
    ).mean(axis=1)
    return {
        "samples": samples,
        "seed": seed,
        "mean_delta": float(draws.mean()),
        "median_delta": float(np.median(draws)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "probability_delta_positive": float(np.mean(draws > 0)),
    }


def attribute_preview(raw: object, limit: int = 500) -> str:
    values = [f"{key}: {value}" for key, value in parse_attributes(raw)]
    text = "; ".join(values)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def add_example_context(frame: pd.DataFrame, raw_items: pd.DataFrame) -> pd.DataFrame:
    context = raw_items[["id", "name", "attributes"]].copy()
    context["attribute_preview"] = context["attributes"].map(attribute_preview)
    context = context.drop(columns="attributes")
    left = context.rename(
        columns={"id": "id1", "name": "title1", "attribute_preview": "attributes1"}
    )
    right = context.rename(
        columns={"id": "id2", "name": "title2", "attribute_preview": "attributes2"}
    )
    return frame.merge(left, on="id1", how="left", validate="many_to_one").merge(
        right, on="id2", how="left", validate="many_to_one"
    )


def validate_aligned_predictions(
    s2: pd.DataFrame, catboost: pd.DataFrame, *, name: str
) -> None:
    expected_columns = {"id1", "id2", "target", "category", "score"}
    if not expected_columns.issubset(s2.columns):
        raise ValueError(f"{name}: S2 predictions have an unexpected schema")
    if len(s2) != len(catboost) or not s2[["id1", "id2"]].equals(
        catboost[["id1", "id2"]]
    ):
        raise ValueError(f"{name}: S2 and CatBoost predictions are not row-aligned")
    if not np.allclose(s2["score"], catboost["transformer_score"], atol=1e-7, rtol=0):
        raise ValueError(f"{name}: transformer scores differ between artifacts")


def common_slice_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "all": pd.Series(True, index=frame.index),
        "title_exact": frame["title_exact"].astype(bool),
        "high_title_similarity": frame["title_token_jaccard"].ge(0.6),
        "token_budget_hit": frame["token_budget_hit"],
        "critical_conflict": frame["critical_conflict"].astype(bool),
        "numeric_context_conflict": frame["numeric_context_conflict_count"].gt(0),
        "unit_conflict": frame["unit_conflict_count"].gt(0),
        "code_exact": frame["code_exact"].astype(bool),
        "code_conflict": frame["code_conflict"].astype(bool),
        "brand_match": frame["brand_match"].astype(bool),
        "brand_conflict": frame["brand_conflict"].astype(bool),
        "model_match": frame["model_match"].astype(bool),
        "model_conflict": frame["model_conflict"].astype(bool),
        "memory_match": frame["memory_match"].astype(bool),
        "memory_conflict": frame["memory_conflict"].astype(bool),
        "color_match": frame["color_match"].astype(bool),
        "color_conflict": frame["color_conflict"].astype(bool),
        "sku_human_asymmetry": frame["sku_human_asymmetry"].astype(bool),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    s2 = pd.read_parquet(args.s2_predictions).reset_index(drop=True)
    catboost = pd.read_parquet(args.catboost_predictions).reset_index(drop=True)
    iid_s2 = pd.read_parquet(args.iid_s2_predictions).reset_index(drop=True)
    iid_catboost = pd.read_parquet(args.iid_catboost_predictions).reset_index(drop=True)
    validate_aligned_predictions(s2, catboost, name="hard")
    validate_aligned_predictions(iid_s2, iid_catboost, name="iid")

    required_ids = set(s2["id1"].tolist()) | set(s2["id2"].tolist())
    required_ids.update(iid_s2["id1"].tolist())
    required_ids.update(iid_s2["id2"].tolist())
    raw_items = pd.read_parquet(
        args.items, columns=["id", "name", "attributes", "category"]
    )
    selected_raw = raw_items[raw_items["id"].isin(required_ids)].reset_index(drop=True)
    if len(selected_raw) != len(required_ids):
        raise ValueError("Missing products for hard predictions")
    items = prepare_item_records(selected_raw)
    if args.char_idf.is_file():
        char_idf = np.load(args.char_idf)
        idf_source = str(args.char_idf)
    else:
        char_idf, _ = fit_hashed_char_idf(
            raw_items["name"].fillna("").astype(str), n_features=65_536
        )
        idf_source = "recomputed_from_all_human_titles"
    features = build_pair_features(items, s2, s2["score"], char_idf)

    diagnostics = s2[["id1", "id2", "target", "category"]].copy()
    diagnostics["s2_score"] = s2["score"].to_numpy(dtype=np.float32)
    diagnostics["catboost_score"] = catboost["catboost_score"].to_numpy(dtype=np.float32)
    diagnostics["score_ab"] = s2["score_ab"].to_numpy(dtype=np.float32)
    diagnostics["score_ba"] = s2["score_ba"].to_numpy(dtype=np.float32)
    diagnostics["score_order_gap"] = s2["score_order_gap"].to_numpy(dtype=np.float32)
    diagnostics["tokens_ab"] = s2["tokens_ab"].to_numpy(dtype=np.int16)
    diagnostics["tokens_ba"] = s2["tokens_ba"].to_numpy(dtype=np.int16)
    diagnostics = pd.concat(
        [diagnostics, features.drop(columns=["transformer_score", "category"])], axis=1
    )
    diagnostics["token_budget_hit"] = diagnostics[["tokens_ab", "tokens_ba"]].max(axis=1).ge(256)
    order_gap_threshold = float(diagnostics["score_order_gap"].quantile(0.9))
    diagnostics["high_order_instability"] = diagnostics["score_order_gap"].ge(
        order_gap_threshold
    )
    diagnostics["high_title_similarity"] = diagnostics["title_token_jaccard"].ge(0.6)

    for score_column in ["s2_score", "catboost_score"]:
        diagnostics[f"{score_column}_rank_pct"] = diagnostics.groupby("category")[
            score_column
        ].rank(method="average", pct=True)
        diagnostics[f"{score_column}_rank_utility"] = np.where(
            diagnostics["target"].eq(1),
            diagnostics[f"{score_column}_rank_pct"],
            1.0 - diagnostics[f"{score_column}_rank_pct"],
        )
    diagnostics["catboost_rank_utility_delta"] = (
        diagnostics["catboost_score_rank_utility"]
        - diagnostics["s2_score_rank_utility"]
    )

    slice_masks = {
        "all_hard": pd.Series(True, index=diagnostics.index),
        "title_exact": diagnostics["title_exact"].astype(bool),
        "token_budget_hit": diagnostics["token_budget_hit"],
        "no_token_budget_hit": ~diagnostics["token_budget_hit"],
        "high_order_instability": diagnostics["high_order_instability"],
        "high_title_similarity": diagnostics["high_title_similarity"],
        "critical_conflict": diagnostics["critical_conflict"].astype(bool),
        "numeric_context_conflict": diagnostics["numeric_context_conflict_count"].gt(0),
        "unit_conflict": diagnostics["unit_conflict_count"].gt(0),
        "code_conflict": diagnostics["code_conflict"].astype(bool),
        "model_conflict": diagnostics["model_conflict"].astype(bool),
        "memory_conflict": diagnostics["memory_conflict"].astype(bool),
        "sku_human_asymmetry": diagnostics["sku_human_asymmetry"].astype(bool),
        "high_attribute_value_overlap": diagnostics["attribute_value_overlap_ratio"].ge(0.5),
        "low_attribute_value_overlap": diagnostics["attribute_value_overlap_ratio"].lt(0.1),
        "negative_exact_title": diagnostics["target"].eq(0)
        & diagnostics["title_exact"].astype(bool),
        "positive_critical_conflict": diagnostics["target"].eq(1)
        & diagnostics["critical_conflict"].astype(bool),
        "positive_numeric_conflict": diagnostics["target"].eq(1)
        & diagnostics["numeric_context_conflict_count"].gt(0),
        "positive_code_conflict": diagnostics["target"].eq(1)
        & diagnostics["code_conflict"].astype(bool),
    }
    slice_metrics = pd.DataFrame(
        [slice_row(diagnostics, name, mask) for name, mask in slice_masks.items()]
    ).sort_values(["support", "slice"], ascending=[False, True])
    slice_metrics.to_csv(args.output_dir / "slice_metrics.csv", index=False)

    label_rows = []
    for name, mask in slice_masks.items():
        for target in [0, 1]:
            part = diagnostics.loc[mask & diagnostics["target"].eq(target)]
            label_rows.append(
                {
                    "slice": name,
                    "target": target,
                    "support": len(part),
                    "s2_mean_score": float(part["s2_score"].mean()) if len(part) else None,
                    "s2_median_score": float(part["s2_score"].median()) if len(part) else None,
                    "catboost_mean_score": float(part["catboost_score"].mean()) if len(part) else None,
                    "catboost_median_score": float(part["catboost_score"].median()) if len(part) else None,
                    "mean_rank_utility_delta": (
                        float(part["catboost_rank_utility_delta"].mean()) if len(part) else None
                    ),
                }
            )
    pd.DataFrame(label_rows).to_csv(args.output_dir / "label_slice_scores.csv", index=False)

    category_rows = []
    for category, part in diagnostics.groupby("category", sort=True):
        s2_ap = float(average_precision_score(part["target"], part["s2_score"]))
        catboost_ap = float(average_precision_score(part["target"], part["catboost_score"]))
        category_rows.append(
            {
                "category": category,
                "support": len(part),
                "positives": int(part["target"].sum()),
                "positive_rate": float(part["target"].mean()),
                "s2_ap": s2_ap,
                "catboost_ap": catboost_ap,
                "catboost_minus_s2_ap": catboost_ap - s2_ap,
                "token_budget_hit_rate": float(part["token_budget_hit"].mean()),
                "critical_conflict_rate": float(part["critical_conflict"].mean()),
                "sku_human_asymmetry_rate": float(part["sku_human_asymmetry"].mean()),
                "mean_order_gap": float(part["score_order_gap"].mean()),
            }
        )
    category_metrics = pd.DataFrame(category_rows).sort_values("catboost_minus_s2_ap")
    category_metrics.to_csv(args.output_dir / "category_metrics.csv", index=False)

    context = add_example_context(diagnostics, selected_raw)
    columns = [
        "id1", "id2", "target", "category", "title1", "title2",
        "attributes1", "attributes2", "s2_score", "catboost_score",
        "catboost_rank_utility_delta", "tokens_ab", "tokens_ba",
        "token_budget_hit", "score_order_gap", "title_token_jaccard",
        "numeric_context_conflict_count", "unit_conflict_count", "code_conflict",
        "model_conflict", "memory_conflict", "sku_human_asymmetry",
        "attribute_value_overlap_ratio",
    ]
    top = args.top_examples
    context.loc[context["target"].eq(0)].nlargest(top, "s2_score")[columns].to_csv(
        args.output_dir / "top_s2_false_positives.csv", index=False
    )
    context.loc[context["target"].eq(1)].nsmallest(top, "s2_score")[columns].to_csv(
        args.output_dir / "top_s2_false_negatives.csv", index=False
    )
    context.nlargest(top, "catboost_rank_utility_delta")[columns].to_csv(
        args.output_dir / "catboost_rank_improvements.csv", index=False
    )
    context.nsmallest(top, "catboost_rank_utility_delta")[columns].to_csv(
        args.output_dir / "catboost_rank_regressions.csv", index=False
    )
    diagnostics.to_parquet(args.output_dir / "hard_diagnostics.parquet", index=False)

    iid_features = build_pair_features(items, iid_s2, iid_s2["score"], char_idf)
    iid_diagnostics = iid_s2[["id1", "id2", "target", "category"]].copy()
    iid_diagnostics["s2_score"] = iid_s2["score"].to_numpy(dtype=np.float32)
    iid_diagnostics["catboost_score"] = iid_catboost["catboost_score"].to_numpy(
        dtype=np.float32
    )
    iid_diagnostics = pd.concat(
        [
            iid_diagnostics,
            iid_features.drop(columns=["transformer_score", "category"]),
        ],
        axis=1,
    )
    iid_diagnostics["token_budget_hit"] = iid_s2[["tokens_ab", "tokens_ba"]].max(
        axis=1
    ).ge(256)
    comparison_rows = []
    common_masks = {
        "hard": common_slice_masks(diagnostics),
        "iid": common_slice_masks(iid_diagnostics),
    }
    for dataset, frame in [("hard", diagnostics), ("iid", iid_diagnostics)]:
        for name, mask in common_masks[dataset].items():
            part = frame.loc[mask]
            comparison_rows.append(
                {
                    "dataset": dataset,
                    "slice": name,
                    "support": len(part),
                    "support_rate": float(mask.mean()),
                    "positives": int(part["target"].sum()),
                    "positive_rate": float(part["target"].mean()) if len(part) else None,
                    "negative_support": int(part["target"].eq(0).sum()),
                    "positive_support": int(part["target"].eq(1).sum()),
                }
            )
    comparison_long = pd.DataFrame(comparison_rows)
    comparison_long.to_csv(args.output_dir / "hard_vs_iid_slices_long.csv", index=False)
    hard_rates = comparison_long[comparison_long["dataset"].eq("hard")].set_index("slice")
    iid_rates = comparison_long[comparison_long["dataset"].eq("iid")].set_index("slice")
    comparison = pd.DataFrame(
        {
            "hard_support": hard_rates["support"],
            "hard_support_rate": hard_rates["support_rate"],
            "hard_positive_rate": hard_rates["positive_rate"],
            "iid_support": iid_rates["support"],
            "iid_support_rate": iid_rates["support_rate"],
            "iid_positive_rate": iid_rates["positive_rate"],
        }
    ).reset_index()
    comparison["hard_vs_iid_support_enrichment"] = (
        comparison["hard_support_rate"] / comparison["iid_support_rate"].replace(0, np.nan)
    )
    comparison.to_csv(args.output_dir / "hard_vs_iid_slices.csv", index=False)

    s2_metrics = macro_metrics(diagnostics, "s2_score")
    catboost_metrics = macro_metrics(diagnostics, "catboost_score")
    bootstrap = stratified_paired_bootstrap(
        diagnostics, samples=args.bootstrap_samples, seed=args.seed
    )
    category_bootstrap = category_delta_bootstrap(
        category_metrics["catboost_minus_s2_ap"].to_numpy(dtype=np.float64),
        samples=max(10_000, args.bootstrap_samples * 10),
        seed=args.seed,
    )
    sorted_category_deltas = category_metrics.sort_values(
        "catboost_minus_s2_ap", ascending=False
    )
    total_category_delta = float(sorted_category_deltas["catboost_minus_s2_ap"].sum())
    summary = {
        "experiment": "minilm_s2_hard_validation_audit",
        "pairs": len(diagnostics),
        "positives": int(diagnostics["target"].sum()),
        "categories": int(diagnostics["category"].nunique()),
        "idf_source": idf_source,
        "order_gap_p90_threshold": order_gap_threshold,
        "s2": s2_metrics,
        "catboost": catboost_metrics,
        "catboost_minus_s2_macro_ap": (
            catboost_metrics["macro_average_precision"] - s2_metrics["macro_average_precision"]
        ),
        "catboost_delta_bootstrap": bootstrap,
        "catboost_delta_category_bootstrap": category_bootstrap,
        "catboost_category_concentration": {
            "improved_categories": int(
                category_metrics["catboost_minus_s2_ap"].gt(0).sum()
            ),
            "degraded_categories": int(
                category_metrics["catboost_minus_s2_ap"].lt(0).sum()
            ),
            "top_3_categories": sorted_category_deltas.head(3)["category"].tolist(),
            "top_3_share_of_summed_delta": float(
                sorted_category_deltas.head(3)["catboost_minus_s2_ap"].sum()
                / total_category_delta
            ),
            "top_5_categories": sorted_category_deltas.head(5)["category"].tolist(),
            "top_5_share_of_summed_delta": float(
                sorted_category_deltas.head(5)["catboost_minus_s2_ap"].sum()
                / total_category_delta
            ),
        },
        "coverage": {
            name: int(mask.sum()) for name, mask in slice_masks.items()
        },
        "hard_vs_iid_slice_comparison": comparison.to_dict("records"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "COMPLETED").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
