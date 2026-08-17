#!/usr/bin/env python3
"""Match analogous hard pairs with opposite human targets for policy review."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cheap_ensemble import extract_codes, extract_numbers, normalized_tokens
from src.serialization_ablation import normalize_text, parse_attributes


STOP_TOKENS = {
    "и", "в", "на", "для", "с", "из", "от", "по", "к", "а", "или",
    "the", "of", "for", "with", "шт", "cm", "mm", "см", "мм", "г", "кг",
    "gb", "гб", "ml", "мл", "l", "л", "не", "определен", "nobrand",
}
STRUCTURE_COLUMNS = (
    "title_token_jaccard",
    "numeric_context_conflict_count",
    "unit_conflict_count",
    "code_conflict",
    "model_conflict",
    "memory_conflict",
    "color_conflict",
    "sku_human_asymmetry",
    "attribute_values_similarity",
)
PATTERN_PRIORITY = (
    "memory_conflict",
    "model_conflict",
    "numeric_or_unit_conflict",
    "code_conflict",
    "color_conflict",
    "sku_human_asymmetry",
    "exact_title_other_difference",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, default=ROOT / "data/items_human.parquet")
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=ROOT / "reports/minilm_s2_hard_audit/hard_diagnostics.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "reports/minilm_s2_hard_audit/label_policy_contradiction_pairs.csv"
        ),
    )
    parser.add_argument("--per-pattern", type=int, default=20)
    parser.add_argument("--minimum-analogy-score", type=float, default=0.30)
    return parser.parse_args()


def attr_preview(raw: object, limit: int = 700) -> str:
    text = "; ".join(f"{key}: {value}" for key, value in parse_attributes(raw))
    return text if len(text) <= limit else text[: limit - 1] + "…"


def item_context(row: Any) -> dict[str, Any]:
    title = normalize_text(row.name)
    attributes = parse_attributes(row.attributes)
    values_text = " ".join(value for _, value in attributes)
    return {
        "id": row.id,
        "title": str(row.name),
        "normalized_title": title,
        "title_tokens": normalized_tokens(title),
        "title_numbers": extract_numbers(title),
        "title_codes": extract_codes(title),
        "all_numbers": extract_numbers(title + " " + values_text),
        "all_codes": extract_codes(title + " " + values_text),
        "attributes": attr_preview(row.attributes),
    }


def clean_anchor_tokens(tokens: set[str] | frozenset[str]) -> list[str]:
    return sorted(
        token
        for token in tokens
        if token not in STOP_TOKENS
        and not token.isdigit()
        and len(token) > 1
        and not re.fullmatch(r"\d+(?:[a-zа-я]+)?", token)
    )


def serialize_set(values: Any) -> str:
    return " | ".join(map(str, sorted(values)))


def enrich_pairs(diagnostics: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    position = pd.Series(np.arange(len(items), dtype=np.int32), index=items["id"])
    left = position.loc[diagnostics["id1"]].to_numpy(dtype=np.int32)
    right = position.loc[diagnostics["id2"]].to_numpy(dtype=np.int32)
    records: list[dict[str, Any]] = []
    for pair_index, (left_index, right_index) in enumerate(zip(left, right)):
        first = items.iloc[int(left_index)]
        second = items.iloc[int(right_index)]
        shared = first.title_tokens & second.title_tokens
        union = first.title_tokens | second.title_tokens
        anchor_tokens = clean_anchor_tokens(shared)
        if len(anchor_tokens) < 2:
            anchor_tokens = clean_anchor_tokens(union)
        records.append(
            {
                "pair_index": pair_index,
                "pair_anchor": " ".join(anchor_tokens),
                "title1": first.title,
                "title2": second.title,
                "attributes1": first.attributes,
                "attributes2": second.attributes,
                "numbers1": serialize_set(first.all_numbers),
                "numbers2": serialize_set(second.all_numbers),
                "codes1": serialize_set(first.all_codes),
                "codes2": serialize_set(second.all_codes),
                "shared_title_tokens": serialize_set(shared),
            }
        )
    return pd.concat([diagnostics.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def pattern_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "memory_conflict": frame["memory_conflict"].astype(bool),
        "model_conflict": frame["model_conflict"].astype(bool),
        "numeric_or_unit_conflict": frame["numeric_context_conflict_count"].gt(0)
        | frame["unit_conflict_count"].gt(0),
        "code_conflict": frame["code_conflict"].astype(bool),
        "color_conflict": frame["color_conflict"].astype(bool),
        "sku_human_asymmetry": frame["sku_human_asymmetry"].astype(bool),
        "exact_title_other_difference": frame["title_exact"].astype(bool),
    }


def structure_similarity(positive: pd.DataFrame, negative: pd.DataFrame) -> np.ndarray:
    combined = pd.concat([positive[list(STRUCTURE_COLUMNS)], negative[list(STRUCTURE_COLUMNS)]])
    scale = combined.std(axis=0, ddof=0).replace(0, 1.0).to_numpy(dtype=np.float64)
    positive_values = positive[list(STRUCTURE_COLUMNS)].to_numpy(dtype=np.float64) / scale
    negative_values = negative[list(STRUCTURE_COLUMNS)].to_numpy(dtype=np.float64) / scale
    distances = np.abs(
        positive_values[:, None, :] - negative_values[None, :, :]
    ).mean(axis=2)
    return np.exp(-distances)


def candidate_matches(
    frame: pd.DataFrame,
    pattern: str,
    mask: pd.Series,
    *,
    limit: int,
    minimum_score: float,
) -> list[dict[str, Any]]:
    candidates = frame.loc[
        mask & frame["title_token_jaccard"].ge(0.20)
    ].copy()
    positive = candidates[candidates["target"].eq(1)]
    negative = candidates[candidates["target"].eq(0)]
    output: list[dict[str, Any]] = []
    for category in sorted(set(positive["category"]) & set(negative["category"])):
        pos = positive[positive["category"].eq(category)].copy()
        neg = negative[negative["category"].eq(category)].copy()
        if pos.empty or neg.empty:
            continue
        texts = pd.concat([pos["pair_anchor"], neg["pair_anchor"]]).fillna("").tolist()
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True
        )
        matrix = vectorizer.fit_transform(texts)
        lexical = cosine_similarity(matrix[: len(pos)], matrix[len(pos) :])
        structural = structure_similarity(pos, neg)
        analogy = 0.75 * lexical + 0.25 * structural
        for pos_index, neg_index in np.argwhere(analogy >= minimum_score):
            output.append(
                {
                    "pattern": pattern,
                    "category": category,
                    "analogy_score": float(analogy[pos_index, neg_index]),
                    "lexical_analogy": float(lexical[pos_index, neg_index]),
                    "structure_analogy": float(structural[pos_index, neg_index]),
                    "positive_row": pos.iloc[int(pos_index)],
                    "negative_row": neg.iloc[int(neg_index)],
                }
            )
    output.sort(key=lambda row: row["analogy_score"], reverse=True)
    selected: list[dict[str, Any]] = []
    used_positive: set[int] = set()
    used_negative: set[int] = set()
    for row in output:
        positive_index = int(row["positive_row"]["pair_index"])
        negative_index = int(row["negative_row"]["pair_index"])
        if positive_index in used_positive or negative_index in used_negative:
            continue
        selected.append(row)
        used_positive.add(positive_index)
        used_negative.add(negative_index)
        if len(selected) >= limit:
            break
    return selected


def prefixed_pair(row: pd.Series, prefix: str) -> dict[str, Any]:
    columns = (
        "id1", "id2", "target", "title1", "title2", "attributes1", "attributes2",
        "numbers1", "numbers2", "codes1", "codes2", "shared_title_tokens",
        "s2_score", "catboost_score", "title_token_jaccard",
        "numeric_context_conflict_count", "unit_conflict_count", "code_conflict",
        "model_conflict", "memory_conflict", "color_conflict",
        "sku_human_asymmetry", "token_budget_hit",
    )
    return {f"{prefix}_{column}": row[column] for column in columns}


def main() -> int:
    args = parse_args()
    diagnostics = pd.read_parquet(args.diagnostics)
    required_ids = set(diagnostics["id1"].tolist()) | set(diagnostics["id2"].tolist())
    raw_items = pd.read_parquet(
        args.items, columns=["id", "name", "attributes", "category"]
    )
    raw_items = raw_items[raw_items["id"].isin(required_ids)].reset_index(drop=True)
    if len(raw_items) != len(required_ids):
        raise ValueError("items file does not contain every hard product")
    item_records = pd.DataFrame.from_records(
        item_context(row) for row in raw_items.itertuples(index=False)
    )
    frame = enrich_pairs(diagnostics, item_records)
    masks = pattern_masks(frame)
    records: list[dict[str, Any]] = []
    seen_pair_of_pairs: set[tuple[int, int]] = set()
    for pattern in PATTERN_PRIORITY:
        matches = candidate_matches(
            frame,
            pattern,
            masks[pattern],
            limit=args.per_pattern,
            minimum_score=args.minimum_analogy_score,
        )
        for match in matches:
            positive = match.pop("positive_row")
            negative = match.pop("negative_row")
            key = (int(positive["pair_index"]), int(negative["pair_index"]))
            if key in seen_pair_of_pairs:
                continue
            seen_pair_of_pairs.add(key)
            records.append(
                {
                    **match,
                    **prefixed_pair(positive, "target1"),
                    **prefixed_pair(negative, "target0"),
                    "manual_policy_verdict": "",
                    "manual_comment": "",
                }
            )
    result = pd.DataFrame.from_records(records).sort_values(
        ["pattern", "analogy_score"], ascending=[True, False]
    )
    if result.empty:
        raise RuntimeError("No opposite-target analogues passed the threshold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    full_output = args.output.with_name(
        f"{args.output.stem}_full{args.output.suffix}"
    )
    result.to_csv(full_output, index=False)
    compact = result[
        [
            "pattern", "category", "analogy_score", "lexical_analogy",
            "structure_analogy", "target1_target", "target1_title1",
            "target1_title2", "target1_numbers1", "target1_numbers2",
            "target1_codes1", "target1_codes2", "target1_s2_score",
            "target1_catboost_score", "target0_target", "target0_title1",
            "target0_title2", "target0_numbers1", "target0_numbers2",
            "target0_codes1", "target0_codes2", "target0_s2_score",
            "target0_catboost_score", "manual_policy_verdict", "manual_comment",
        ]
    ].copy()
    compact.insert(
        6,
        "target1_pair",
        "[A] " + compact["target1_title1"] + " || [B] " + compact["target1_title2"],
    )
    compact.insert(
        17,
        "target0_pair",
        "[A] " + compact["target0_title1"] + " || [B] " + compact["target0_title2"],
    )
    compact.to_csv(args.output, index=False)
    comparison_ids = pd.Series(
        [f"C{index:03d}" for index in range(1, len(result) + 1)],
        index=result.index,
    )

    def simple_target_file(prefix: str, target: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "comparison_id": comparison_ids,
                "pattern": result["pattern"],
                "category": result["category"],
                "target": target,
                "title_a": result[f"{prefix}_title1"],
                "title_b": result[f"{prefix}_title2"],
                "numbers_a": result[f"{prefix}_numbers1"],
                "numbers_b": result[f"{prefix}_numbers2"],
                "codes_a": result[f"{prefix}_codes1"],
                "codes_b": result[f"{prefix}_codes2"],
            }
        )

    target0_output = args.output.with_name("label_policy_target_0.csv")
    target1_output = args.output.with_name("label_policy_target_1.csv")
    simple_target_file("target0", 0).to_csv(target0_output, index=False)
    simple_target_file("target1", 1).to_csv(target1_output, index=False)
    summary = {
        "rows": len(result),
        "patterns": result["pattern"].value_counts().sort_index().to_dict(),
        "minimum_analogy_score": args.minimum_analogy_score,
        "mean_analogy_score": float(result["analogy_score"].mean()),
        "output": str(args.output),
        "full_output": str(full_output),
        "target0_output": str(target0_output),
        "target1_output": str(target1_output),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
