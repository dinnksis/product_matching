"""Profile the human and probabilistic LLM product-pair datasets.

The analysis deliberately treats the LLM target as a weak score.  A binary
proxy at 0.5 is reported for readability, but it is not presented as ground
truth.  Outputs are compact tables that can be reviewed without loading the
11M-row parquet again.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from sklearn.metrics import average_precision_score, roc_auc_score


TARGET_DENOMINATOR = 9
NAME_RE = re.compile(r"[^0-9a-zа-яё]+", re.IGNORECASE)
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, default=Path("data/items.parquet"))
    parser.add_argument(
        "--human-items", type=Path, default=Path("data/items_human.parquet")
    )
    parser.add_argument(
        "--human-matches", type=Path, default=Path("data/matches.parquet")
    )
    parser.add_argument(
        "--llm-matches", type=Path, default=Path("data/matches_llm.parquet")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/llm_label_distribution"),
    )
    parser.add_argument("--sample-per-category", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--frozen-predictions",
        type=Path,
        default=Path(
            "artifacts/kaggle/product-matching-qwen-embedding-boosting/"
            "embedding_boosting/01_names_lexical/validation_predictions.parquet"
        ),
    )
    return parser.parse_args()


def normalized_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower().replace("ё", "е")
    return " ".join(NAME_RE.sub(" ", value).split())


def load_item_index(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    table = pq.read_table(path, columns=["id", "category"], memory_map=True)
    ids = table["id"].combine_chunks().to_numpy(zero_copy_only=True)
    category_array = table["category"].combine_chunks()
    categories = sorted(value.as_py() for value in pc.unique(category_array))
    codes = pc.index_in(
        category_array, value_set=pa.array(categories)
    ).to_numpy(zero_copy_only=False).astype(np.int16, copy=False)
    order = np.argsort(ids, kind="mergesort")
    sorted_ids = np.asarray(ids[order])
    sorted_codes = np.asarray(codes[order])
    if np.any(sorted_ids[1:] == sorted_ids[:-1]):
        raise ValueError(f"Duplicate item IDs in {path}")
    item_counts = np.bincount(codes, minlength=len(categories))
    item_frame = pd.DataFrame(
        {"category": categories, "all_items": item_counts.astype(np.int64)}
    )
    return sorted_ids, sorted_codes, categories, item_frame


def lookup_codes(
    sorted_ids: np.ndarray, sorted_codes: np.ndarray, query: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(sorted_ids, query)
    valid = positions < len(sorted_ids)
    safe = np.minimum(positions, len(sorted_ids) - 1)
    valid &= sorted_ids[safe] == query
    result = np.full(len(query), -1, dtype=np.int16)
    result[valid] = sorted_codes[safe[valid]]
    return result, valid


def load_pairs(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    table = pq.read_table(path, columns=["id1", "id2", "target"], memory_map=True)
    nulls = {name: table[name].null_count for name in ("id1", "id2", "target")}
    return (
        table["id1"].combine_chunks().to_numpy(zero_copy_only=True),
        table["id2"].combine_chunks().to_numpy(zero_copy_only=True),
        table["target"].combine_chunks().to_numpy(zero_copy_only=True),
        nulls,
    )


def category_table(
    categories: list[str], codes: np.ndarray, targets: np.ndarray, source: str
) -> tuple[pd.DataFrame, np.ndarray]:
    if source == "llm":
        target_codes = np.rint(targets * TARGET_DENOMINATOR).astype(np.int8)
        if not np.allclose(targets, target_codes / TARGET_DENOMINATOR):
            raise ValueError("LLM targets are not on the expected k/9 grid")
        histogram = np.zeros((len(categories), TARGET_DENOMINATOR + 1), dtype=np.int64)
        np.add.at(histogram, (codes, target_codes), 1)
    else:
        if not np.isin(targets, [0.0, 1.0]).all():
            raise ValueError("Human targets must be binary")
        histogram = np.zeros((len(categories), TARGET_DENOMINATOR + 1), dtype=np.int64)
        np.add.at(
            histogram,
            (codes, targets.astype(np.int8) * TARGET_DENOMINATOR),
            1,
        )

    rows: list[dict[str, float | int | str]] = []
    grid = np.arange(TARGET_DENOMINATOR + 1) / TARGET_DENOMINATOR
    for index, category in enumerate(categories):
        counts = histogram[index]
        pairs = int(counts.sum())
        rows.append(
            {
                "category": category,
                f"{source}_pairs": pairs,
                f"{source}_pair_share": pairs / len(targets),
                f"{source}_soft_positive_mass": float(np.dot(counts, grid) / pairs),
                f"{source}_threshold_positive_rate": float(counts[5:].sum() / pairs),
                f"{source}_exact_zero_rate": float(counts[0] / pairs),
                f"{source}_exact_one_rate": float(counts[-1] / pairs),
                f"{source}_uncertain_rate": float(counts[1:-1].sum() / pairs),
            }
        )
    return pd.DataFrame(rows), histogram


def duplicate_summary(id1: np.ndarray, id2: np.ndarray, target: np.ndarray) -> dict[str, int]:
    records = np.empty(
        len(id1), dtype=[("lo", "<i8"), ("hi", "<i8"), ("target", "<f8")]
    )
    np.minimum(id1, id2, out=records["lo"])
    np.maximum(id1, id2, out=records["hi"])
    records["target"] = target
    records.sort(order=["lo", "hi", "target"])
    same_pair = (records["lo"][1:] == records["lo"][:-1]) & (
        records["hi"][1:] == records["hi"][:-1]
    )
    conflicting = same_pair & (records["target"][1:] != records["target"][:-1])
    return {
        "duplicate_unordered_rows_after_first": int(same_pair.sum()),
        "adjacent_conflicting_duplicate_links": int(conflicting.sum()),
    }


def degree_summary(
    id1: np.ndarray,
    id2: np.ndarray,
    sorted_ids: np.ndarray,
    sorted_codes: np.ndarray,
    categories: list[str],
) -> tuple[dict[str, float | int | dict[str, int]], pd.DataFrame, np.ndarray]:
    endpoints = np.empty(2 * len(id1), dtype=np.int64)
    endpoints[: len(id1)] = id1
    endpoints[len(id1) :] = id2
    endpoints.sort()
    starts = np.empty(len(endpoints), dtype=bool)
    starts[0] = True
    starts[1:] = endpoints[1:] != endpoints[:-1]
    boundaries = np.flatnonzero(starts)
    unique_ids = endpoints[boundaries]
    degrees = np.diff(np.append(boundaries, len(endpoints)))
    quantiles = np.quantile(
        degrees, [0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0], method="nearest"
    )
    summary: dict[str, float | int | dict[str, int]] = {
        "unique_items": int(len(unique_ids)),
        "mean_degree": float(degrees.mean()),
        "degree_one_items": int((degrees == 1).sum()),
        "degree_one_rate": float((degrees == 1).mean()),
        "degree_quantiles": dict(
            zip(
                ("min", "p50", "p90", "p95", "p99", "p999", "max"),
                map(int, quantiles),
            )
        ),
    }
    codes, valid = lookup_codes(sorted_ids, sorted_codes, unique_ids)
    if not valid.all():
        raise ValueError("Pair endpoint is absent from the item table")
    rows = []
    for index, category in enumerate(categories):
        selected = codes == index
        category_degrees = degrees[selected]
        rows.append(
            {
                "category": category,
                "llm_unique_items": int(selected.sum()),
                "llm_mean_degree": float(category_degrees.mean()),
                "llm_degree_one_rate": float((category_degrees == 1).mean()),
                "llm_max_degree": int(category_degrees.max()),
            }
        )
    return summary, pd.DataFrame(rows), unique_ids


def membership(sorted_values: np.ndarray, query: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(sorted_values, query)
    safe = np.minimum(positions, len(sorted_values) - 1)
    return (positions < len(sorted_values)) & (sorted_values[safe] == query)


def sample_pairs(
    source: str,
    id1: np.ndarray,
    id2: np.ndarray,
    targets: np.ndarray,
    codes: np.ndarray,
    categories: list[str],
    sample_per_category: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    selected = []
    for index in range(len(categories)):
        candidates = np.flatnonzero(codes == index)
        selected.append(
            rng.choice(
                candidates, size=min(sample_per_category, len(candidates)), replace=False
            )
        )
    indices = np.concatenate(selected)
    return pd.DataFrame(
        {
            "source": source,
            "category": [categories[index] for index in codes[indices]],
            "id1": id1[indices],
            "id2": id2[indices],
            "target": targets[indices],
        }
    )


def selected_names(items_path: Path, required_ids: np.ndarray) -> dict[int, str]:
    result: dict[int, str] = {}
    parquet = pq.ParquetFile(items_path)
    for row_group in range(parquet.metadata.num_row_groups):
        table = parquet.read_row_group(row_group, columns=["id", "name"])
        ids = table["id"].combine_chunks().to_numpy(zero_copy_only=True)
        positions = np.searchsorted(required_ids, ids)
        safe = np.minimum(positions, len(required_ids) - 1)
        selected = (positions < len(required_ids)) & (required_ids[safe] == ids)
        indices = np.flatnonzero(selected)
        names = table["name"].combine_chunks().take(pa.array(indices)).to_pylist()
        result.update(zip(ids[indices].tolist(), names))
    if len(result) != len(required_ids):
        raise ValueError(
            f"Resolved {len(result):,} names for {len(required_ids):,} requested IDs"
        )
    return result


def lexical_sample_tables(
    pairs: pd.DataFrame, names: dict[int, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    size = len(pairs)
    ratio = np.empty(size, dtype=np.float32)
    token_set = np.empty(size, dtype=np.float32)
    exact = np.empty(size, dtype=np.int8)
    number_jaccard = np.empty(size, dtype=np.float32)
    for index, (left_id, right_id) in enumerate(zip(pairs.id1, pairs.id2)):
        left = normalized_name(names[int(left_id)])
        right = normalized_name(names[int(right_id)])
        ratio[index] = fuzz.ratio(left, right)
        token_set[index] = fuzz.token_set_ratio(left, right)
        exact[index] = left == right
        left_numbers = set(NUMBER_RE.findall(left))
        right_numbers = set(NUMBER_RE.findall(right))
        union = left_numbers | right_numbers
        number_jaccard[index] = (
            len(left_numbers & right_numbers) / len(union) if union else np.nan
        )
    pairs = pairs.copy()
    pairs["hard_label"] = (pairs.target >= 0.5).astype(np.int8)
    pairs["ratio"] = ratio
    pairs["token_set_ratio"] = token_set
    pairs["exact_name"] = exact
    pairs["number_jaccard"] = number_jaccard

    def summarize(group: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "pairs": len(group),
                "target_mean": group.target.mean(),
                "ratio_mean": group.ratio.mean(),
                "ratio_p50": group.ratio.quantile(0.5),
                "ratio_p90": group.ratio.quantile(0.9),
                "token_set_mean": group.token_set_ratio.mean(),
                "token_set_p50": group.token_set_ratio.quantile(0.5),
                "token_set_p90": group.token_set_ratio.quantile(0.9),
                "exact_name_rate": group.exact_name.mean(),
                "number_jaccard_mean": group.number_jaccard.mean(),
            }
        )

    by_label = (
        pairs.groupby(["source", "hard_label"], sort=True)
        .apply(summarize, include_groups=False)
        .reset_index()
    )
    by_category = (
        pairs.groupby(["source", "category"], sort=True)
        .apply(summarize, include_groups=False)
        .reset_index()
    )
    return by_label, by_category


def save_plot(category: pd.DataFrame, output_path: Path) -> None:
    ordered = category.sort_values("human_threshold_positive_rate")
    positions = np.arange(len(ordered))
    figure, axis = plt.subplots(figsize=(10, 8))
    axis.scatter(
        100 * ordered.human_threshold_positive_rate,
        positions,
        label="Human binary labels",
        color="#2563eb",
    )
    axis.scatter(
        100 * ordered.llm_threshold_positive_rate,
        positions,
        label="LLM score >= 0.5",
        color="#ea580c",
    )
    axis.scatter(
        100 * ordered.llm_soft_positive_mass,
        positions,
        label="LLM mean soft score",
        color="#16a34a",
        marker="x",
    )
    axis.set_yticks(positions, ordered.category)
    axis.set_xlabel("Positive share / mean score, %")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def llm_prior_shift_diagnostic(
    predictions_path: Path, category: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float]]:
    predictions = pd.read_parquet(predictions_path)
    required = {"target", "predict", "category"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Frozen predictions lack columns: {sorted(required - set(predictions))}")
    category_index = category.set_index("category")
    rows = []
    for prior_name, prior_column in (
        ("llm_threshold", "llm_threshold_positive_rate"),
        ("llm_soft_mean", "llm_soft_positive_mass"),
    ):
        for category_name, part in predictions.groupby("category", sort=True):
            original = float(part.target.mean())
            target = float(category_index.loc[category_name, prior_column])
            negative_weight = original * (1.0 - target) / (
                target * (1.0 - original)
            )
            weights = np.where(part.target.to_numpy() == 1, 1.0, negative_weight)
            rows.append(
                {
                    "prior": prior_name,
                    "category": category_name,
                    "human_validation_prevalence": original,
                    "target_llm_prevalence": target,
                    "negative_weight": negative_weight,
                    "weighted_average_precision": average_precision_score(
                        part.target, part.predict, sample_weight=weights
                    ),
                    "weighted_roc_auc": roc_auc_score(
                        part.target, part.predict, sample_weight=weights
                    ),
                }
            )
    table = pd.DataFrame(rows)
    baseline_macro_ap = float(
        predictions.groupby("category")
        .apply(
            lambda part: average_precision_score(part.target, part.predict),
            include_groups=False,
        )
        .mean()
    )
    summary = {"baseline_macro_ap": baseline_macro_ap}
    for prior_name, part in table.groupby("prior"):
        summary[f"{prior_name}_macro_ap"] = float(
            part.weighted_average_precision.mean()
        )
        summary[f"{prior_name}_macro_roc_auc"] = float(part.weighted_roc_auc.mean())
    return table, summary


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sorted_ids, sorted_codes, categories, all_item_counts = load_item_index(args.items)

    human_id1, human_id2, human_target, human_nulls = load_pairs(args.human_matches)
    llm_id1, llm_id2, llm_target, llm_nulls = load_pairs(args.llm_matches)
    human_codes_1, human_valid_1 = lookup_codes(sorted_ids, sorted_codes, human_id1)
    human_codes_2, human_valid_2 = lookup_codes(sorted_ids, sorted_codes, human_id2)
    llm_codes_1, llm_valid_1 = lookup_codes(sorted_ids, sorted_codes, llm_id1)
    llm_codes_2, llm_valid_2 = lookup_codes(sorted_ids, sorted_codes, llm_id2)

    human_category, human_histogram = category_table(
        categories, human_codes_1, human_target, "human"
    )
    llm_category, llm_histogram = category_table(
        categories, llm_codes_1, llm_target, "llm"
    )
    category = all_item_counts.merge(human_category, on="category").merge(
        llm_category, on="category"
    )
    category["threshold_rate_delta_llm_minus_human"] = (
        category.llm_threshold_positive_rate
        - category.human_threshold_positive_rate
    )
    category["soft_mass_delta_llm_minus_human"] = (
        category.llm_soft_positive_mass
        - category.human_soft_positive_mass
    )

    target_counts = llm_histogram.sum(axis=0)
    target_distribution = pd.DataFrame(
        {
            "target": np.arange(TARGET_DENOMINATOR + 1) / TARGET_DENOMINATOR,
            "count": target_counts,
            "share": target_counts / len(llm_target),
        }
    )
    target_by_category = pd.DataFrame(
        [
            {
                "category": category_name,
                "target": target_code / TARGET_DENOMINATOR,
                "count": int(llm_histogram[category_index, target_code]),
                "share_within_category": float(
                    llm_histogram[category_index, target_code]
                    / llm_histogram[category_index].sum()
                ),
            }
            for category_index, category_name in enumerate(categories)
            for target_code in range(TARGET_DENOMINATOR + 1)
        ]
    )

    graph, graph_by_category, llm_unique_ids = degree_summary(
        llm_id1, llm_id2, sorted_ids, sorted_codes, categories
    )
    category = category.merge(graph_by_category, on="category")
    human_item_ids = np.sort(
        pq.read_table(args.human_items, columns=["id"], memory_map=True)["id"]
        .combine_chunks()
        .to_numpy(zero_copy_only=True)
    )
    human_in_all = membership(sorted_ids, human_item_ids)
    llm_left_human = membership(human_item_ids, llm_id1)
    llm_right_human = membership(human_item_ids, llm_id2)

    integrity = {
        "human_nulls": human_nulls,
        "llm_nulls": llm_nulls,
        "human_missing_id1": int((~human_valid_1).sum()),
        "human_missing_id2": int((~human_valid_2).sum()),
        "llm_missing_id1": int((~llm_valid_1).sum()),
        "llm_missing_id2": int((~llm_valid_2).sum()),
        "human_cross_category_pairs": int(
            ((human_codes_1 != human_codes_2) & human_valid_1 & human_valid_2).sum()
        ),
        "llm_cross_category_pairs": int(
            ((llm_codes_1 != llm_codes_2) & llm_valid_1 & llm_valid_2).sum()
        ),
        "human_self_pairs": int((human_id1 == human_id2).sum()),
        "llm_self_pairs": int((llm_id1 == llm_id2).sum()),
        **duplicate_summary(llm_id1, llm_id2, llm_target),
    }

    rng = np.random.default_rng(args.seed)
    sampled_pairs = pd.concat(
        [
            sample_pairs(
                "human",
                human_id1,
                human_id2,
                human_target,
                human_codes_1,
                categories,
                args.sample_per_category,
                rng,
            ),
            sample_pairs(
                "llm",
                llm_id1,
                llm_id2,
                llm_target,
                llm_codes_1,
                categories,
                args.sample_per_category,
                rng,
            ),
        ],
        ignore_index=True,
    )
    required_ids = np.unique(
        np.concatenate([sampled_pairs.id1.to_numpy(), sampled_pairs.id2.to_numpy()])
    )
    names = selected_names(args.items, required_ids)
    lexical_by_label, lexical_by_category = lexical_sample_tables(sampled_pairs, names)

    prior_shift_table = None
    prior_shift_summary = None
    if args.frozen_predictions.is_file():
        prior_shift_table, prior_shift_summary = llm_prior_shift_diagnostic(
            args.frozen_predictions, category
        )

    correlation = category[
        ["human_threshold_positive_rate", "llm_threshold_positive_rate"]
    ].corr(method="pearson").iloc[0, 1]
    rank_correlation = category[
        ["human_threshold_positive_rate", "llm_threshold_positive_rate"]
    ].corr(method="spearman").iloc[0, 1]
    summary = {
        "inputs": {
            "items": str(args.items),
            "human_items": str(args.human_items),
            "human_matches": str(args.human_matches),
            "llm_matches": str(args.llm_matches),
        },
        "human": {
            "pairs": int(len(human_target)),
            "positive_examples": int(human_target.sum()),
            "global_positive_rate": float(human_target.mean()),
            "mean_category_positive_rate": float(
                category.human_threshold_positive_rate.mean()
            ),
        },
        "llm": {
            "pairs": int(len(llm_target)),
            "soft_positive_mass": float(llm_target.sum()),
            "global_mean_soft_score": float(llm_target.mean()),
            "mean_category_soft_score": float(category.llm_soft_positive_mass.mean()),
            "threshold_positive_examples": int((llm_target >= 0.5).sum()),
            "global_threshold_positive_rate": float((llm_target >= 0.5).mean()),
            "mean_category_threshold_positive_rate": float(
                category.llm_threshold_positive_rate.mean()
            ),
            "exact_zero_examples": int((llm_target == 0).sum()),
            "exact_one_examples": int((llm_target == 1).sum()),
            "uncertain_examples": int(((llm_target > 0) & (llm_target < 1)).sum()),
            "target_values": target_distribution.target.tolist(),
        },
        "item_universes": {
            "all_items": int(len(sorted_ids)),
            "human_items": int(len(human_item_ids)),
            "human_items_present_in_all_items": int(human_in_all.sum()),
            "llm_unique_items": int(len(llm_unique_ids)),
            "llm_pairs_with_any_human_item": int(
                (llm_left_human | llm_right_human).sum()
            ),
            "llm_pairs_with_two_human_items": int(
                (llm_left_human & llm_right_human).sum()
            ),
            "all_items_unused_by_llm_pairs": int(len(sorted_ids) - len(llm_unique_ids)),
        },
        "integrity": integrity,
        "llm_graph": graph,
        "human_llm_category_prevalence_pearson": float(correlation),
        "human_llm_category_prevalence_spearman": float(rank_correlation),
        "lexical_sample": {
            "seed": args.seed,
            "requested_pairs_per_source_category": args.sample_per_category,
            "sampled_pairs": int(len(sampled_pairs)),
        },
        "llm_prior_shift_diagnostic": prior_shift_summary,
        "runtime_seconds": float(time.perf_counter() - started),
    }

    category.to_csv(args.output_dir / "category_distribution.csv", index=False)
    target_distribution.to_csv(args.output_dir / "target_distribution.csv", index=False)
    target_by_category.to_csv(args.output_dir / "target_by_category.csv", index=False)
    lexical_by_label.to_csv(args.output_dir / "lexical_similarity_by_label.csv", index=False)
    lexical_by_category.to_csv(
        args.output_dir / "lexical_similarity_by_category.csv", index=False
    )
    if prior_shift_table is not None:
        prior_shift_table.to_csv(
            args.output_dir / "llm_prior_weighting_by_category.csv", index=False
        )
    save_plot(category, args.output_dir / "category_prevalence_comparison.png")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
