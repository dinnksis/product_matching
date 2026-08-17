#!/usr/bin/env python3
"""Combine 3-fold S2 OOF scores and build deterministic x2 hard oversampling."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "product_matching_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.serialization_ablation import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-audit", type=Path, required=True)
    parser.add_argument("--oof-runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def canonical(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    left = result["id1"].to_numpy(dtype=np.int64)
    right = result["id2"].to_numpy(dtype=np.int64)
    result["canonical_id1"] = np.minimum(left, right)
    result["canonical_id2"] = np.maximum(left, right)
    return result


def combine_oof(config: dict[str, Any], directory: Path) -> pd.DataFrame:
    frames = []
    for fold in range(int(config["oof_folds"])):
        path = directory / f"fold_{fold}/validation_predictions.parquet"
        frame = pd.read_parquet(path)
        frame["oof_fold"] = fold
        frames.append(frame)
    result = canonical(pd.concat(frames, ignore_index=True))
    if result.duplicated(["canonical_id1", "canonical_id2"]).any():
        raise ValueError("OOF predictions contain duplicate unordered pairs")
    if not np.isfinite(result["score"]).all():
        raise ValueError("OOF scores contain non-finite values")
    return result


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = pd.read_parquet(args.train_audit)
    oof = combine_oof(config, args.oof_runs_dir)
    score_columns = [
        "canonical_id1",
        "canonical_id2",
        "target",
        "oof_fold",
        "score_ab",
        "score_ba",
        "score",
        "score_order_gap",
    ]
    oof = oof[score_columns].rename(
        columns={"target": "oof_target", "oof_fold": "predicted_oof_fold"}
    )
    merged = audit.merge(
        oof,
        on=["canonical_id1", "canonical_id2"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if merged["score"].isna().any() or len(merged) != len(audit):
        raise ValueError("OOF predictions do not cover every train row exactly once")
    if not np.array_equal(
        merged["target"].to_numpy(dtype=np.int8),
        merged["oof_target"].to_numpy(dtype=np.int8),
    ):
        raise ValueError("OOF targets differ from train audit")
    if not np.array_equal(
        merged["oof_fold"].to_numpy(dtype=np.int8),
        merged["predicted_oof_fold"].to_numpy(dtype=np.int8),
    ):
        raise ValueError("a fold predicted rows outside its held-out partition")

    positive = merged["target"].eq(1)
    negative = ~positive
    merged["hardness"] = np.where(positive, 1.0 - merged["score"], merged["score"])
    eligible = merged["eligible_for_hard_mining"].astype(bool)
    quantile = float(config["hard_quantile"])
    thresholds = {}
    merged["is_mined_hard"] = False
    for label, label_mask in ((0, negative), (1, positive)):
        candidates = merged.loc[eligible & label_mask, "hardness"]
        if candidates.empty:
            raise ValueError(f"no eligible label={label} candidates")
        threshold = float(candidates.quantile(quantile))
        thresholds[str(label)] = threshold
        merged.loc[eligible & label_mask & merged["hardness"].ge(threshold), "is_mined_hard"] = True

    hard = merged[merged["is_mined_hard"]].copy()
    factor = int(config["hard_oversample_factor"])
    if factor != 2:
        raise ValueError("this causal experiment is fixed to x2 hard oversampling")
    pair_columns = ["id1", "id2", "target"]
    original = merged[["row_id", *pair_columns]].copy()
    original["oversample_copy"] = 0
    duplicate = hard[["row_id", *pair_columns]].copy()
    duplicate["oversample_copy"] = 1
    oversampled = pd.concat([original, duplicate], ignore_index=True)
    seed = int(config["seed"])
    oversampled["shuffle_key"] = [
        stable_hash(f"{row_id}:{copy}", seed)
        for row_id, copy in zip(oversampled["row_id"], oversampled["oversample_copy"])
    ]
    oversampled = oversampled.sort_values(
        ["shuffle_key", "row_id", "oversample_copy"], kind="mergesort"
    )
    oversampled[pair_columns].to_parquet(
        args.output_dir / "train_pairs_hard_x2.parquet", index=False
    )
    merged.to_parquet(args.output_dir / "oof_predictions_and_hardness.parquet", index=False)

    quantile_rows = []
    for label, part in merged.groupby("target", sort=True):
        for eligibility, eligible_part in part.groupby("eligible_for_hard_mining", sort=True):
            values = eligible_part["hardness"]
            quantile_rows.append(
                {
                    "target": int(label),
                    "eligible_for_hard_mining": bool(eligibility),
                    "pairs": len(eligible_part),
                    "score_mean": float(eligible_part["score"].mean()),
                    "hardness_mean": float(values.mean()),
                    "hardness_p50": float(values.quantile(0.50)),
                    "hardness_p80": float(values.quantile(0.80)),
                    "hardness_p85": float(values.quantile(0.85)),
                    "hardness_p90": float(values.quantile(0.90)),
                    "hardness_p95": float(values.quantile(0.95)),
                    "mining_threshold": thresholds[str(int(label))] if eligibility else None,
                }
            )
    pd.DataFrame(quantile_rows).to_csv(
        args.output_dir / "hardness_distribution_quantiles.csv", index=False
    )

    count_rows = [
        {"slice": "hard_positives", "pairs": int((hard["target"].eq(1)).sum())},
        {"slice": "hard_negatives", "pairs": int((hard["target"].eq(0)).sum())},
        {
            "slice": "numeric_conflict_hard_negatives",
            "pairs": int((hard["target"].eq(0) & hard["numeric_conflict"]).sum()),
        },
        {
            "slice": "code_or_model_hard_negatives",
            "pairs": int(
                (
                    hard["target"].eq(0)
                    & (hard["code_conflict"] | hard["model_code_conflict"])
                ).sum()
            ),
        },
        {
            "slice": "sku_human_title_hard_positives",
            "pairs": int(
                (hard["target"].eq(1) & hard["sku_vs_human_title"]).sum()
            ),
        },
    ]
    pd.DataFrame(count_rows).to_csv(args.output_dir / "hard_mining_counts.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, label, title in zip(axes, (0, 1), ("Negative OOF score", "Positive 1 - OOF score")):
        part = merged[eligible & merged["target"].eq(label)]
        axis.hist(part["hardness"], bins=50, color="#4c78a8", alpha=0.85)
        axis.axvline(thresholds[str(label)], color="#e45756", linestyle="--", label="p85")
        axis.set_title(title)
        axis.set_xlabel("hardness")
        axis.legend()
    axes[0].set_ylabel("pairs")
    figure.tight_layout()
    figure.savefig(args.output_dir / "hardness_distributions.png", dpi=160)
    plt.close(figure)

    report = {
        "experiment": config["experiment"],
        "oof_pairs": len(merged),
        "oof_folds": int(config["oof_folds"]),
        "hard_quantile": quantile,
        "thresholds": thresholds,
        "eligible_pairs": int(eligible.sum()),
        "excluded_definite_conflicts": int(merged["definite_label_conflict"].sum()),
        "excluded_strong_suspicion": int(merged["strong_label_suspicion"].sum()),
        "mined_hard_pairs": len(hard),
        "mined_hard_positive_pairs": int((hard["target"].eq(1)).sum()),
        "mined_hard_negative_pairs": int((hard["target"].eq(0)).sum()),
        "original_training_pairs": len(merged),
        "oversampled_training_pairs": len(oversampled),
        "hard_exposure_fraction_after_oversampling": float(2 * len(hard) / len(oversampled)),
        "selection_uses_only_oof_scores": True,
        "label_audit_filter_precedes_hardness": True,
        "counts": count_rows,
    }
    (args.output_dir / "hard_mining_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
