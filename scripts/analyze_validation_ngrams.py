"""Rank 3-5 word n-grams associated with MiniLM FP/FN validation errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def analyze(path: Path, split: str, threshold: float, min_support: int, max_features: int):
    frame = pd.read_parquet(path)
    texts = (frame["product_text_1"].fillna("") + " " + frame["product_text_2"].fillna("")).tolist()
    vectorizer = CountVectorizer(
        lowercase=True, ngram_range=(3, 5), min_df=min_support,
        max_features=max_features, binary=True, dtype=np.uint8,
        token_pattern=r"(?u)\b[\w][\w.+/-]*\b",
    )
    matrix = vectorizer.fit_transform(texts)
    names = vectorizer.get_feature_names_out()
    target = frame["target"].to_numpy(dtype=float) >= 0.5
    predicted = frame["score"].to_numpy(dtype=float) >= threshold
    rows, example_rows = [], []
    for label, kind in ((False, "fp"), (True, "fn")):
        eligible = target == label
        mistakes = eligible & (predicted != target)
        support = np.asarray(matrix[eligible].sum(axis=0)).ravel()
        errors = np.asarray(matrix[mistakes].sum(axis=0)).ravel()
        base_rate = mistakes.sum() / max(1, eligible.sum())
        valid = np.flatnonzero(support >= min_support)
        rates = errors[valid] / support[valid]
        lifts = rates / max(base_rate, 1e-12)
        order = valid[np.lexsort((-support[valid], -errors[valid], -lifts))][::-1]
        # lexsort direction is easy to misread; use an explicit DataFrame sort below.
        part = pd.DataFrame({
            "split": split, "error_type": kind, "ngram": names[valid],
            "n_words": [x.count(" ") + 1 for x in names[valid]],
            "support": support[valid].astype(int), "errors": errors[valid].astype(int),
            "error_rate": rates, "split_base_error_rate": base_rate, "error_lift": lifts,
        }).sort_values(["error_lift", "errors", "support"], ascending=[False, False, False])
        rows.append(part)
        for gram_row in part.head(500).itertuples(index=False):
            col = vectorizer.vocabulary_[gram_row.ngram]
            positions = np.flatnonzero(mistakes & (matrix[:, col].toarray().ravel() > 0))[:3]
            examples = []
            for pos in positions:
                r = frame.iloc[pos]
                examples.append({
                    "id1": str(r.id1), "id2": str(r.id2), "target": float(r.target),
                    "score": float(r.score), "category": str(r.category_1),
                    "text1": str(r.product_text_1)[:500], "text2": str(r.product_text_2)[:500],
                })
            example_rows.append({"split": split, "error_type": kind,
                                 "ngram": gram_row.ngram, "examples": examples})
    result = pd.concat(rows, ignore_index=True)
    summary = {
        "split": split, "pairs": len(frame), "threshold": threshold,
        "fp": int((~target & predicted).sum()), "fn": int((target & ~predicted).sum()),
        "vocabulary": len(names), "ranked_ngrams": len(result),
    }
    return result, example_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/ngram_errors"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--max-features", type=int, default=100_000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables, examples, summaries = [], [], []
    for split in ("iid", "hard", "ood"):
        table, split_examples, summary = analyze(
            args.predictions_dir / f"{split}_validation_predictions.parquet",
            split, args.threshold, args.min_support, args.max_features,
        )
        tables.append(table); examples.extend(split_examples); summaries.append(summary)
    pd.concat(tables, ignore_index=True).to_csv(
        args.output_dir / "ngram_error_ranking.csv", index=False, encoding="utf-8-sig")
    with (args.output_dir / "ngram_error_examples.jsonl").open("w", encoding="utf-8") as fh:
        for row in examples:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
