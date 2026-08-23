"""Compare baseline and hard-negative validation predictions by category."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/kaggle/minilm-5ep-human-baseline/minilm_5ep_team_data_loss_ablation"
EXPERIMENT = ROOT / "artifacts/kaggle/minilm-5ep-hard-negatives-v1/minilm_5ep_team_data_loss_ablation"
TRANSFORMATIONS = ROOT / "prepared/hard_negatives_v1/transformations.parquet"
OUTPUT = ROOT / "reports/hard_negatives_v1"


def category_metrics(path: Path, split: str, prefix: str) -> pd.DataFrame:
    frame = pd.read_parquet(path / f"{split}_validation_predictions.parquet")
    rows = []
    for category, group in frame.groupby("category_1", dropna=False):
        if group["target"].nunique() < 2:
            continue
        rows.append({"category": category, f"{prefix}_ap": average_precision_score(group.target, group.score),
                     f"{prefix}_pairs": len(group)})
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_splits, summary = [], []
    for split in ("iid", "hard", "ood"):
        base = category_metrics(BASE, split, "baseline")
        experiment = category_metrics(EXPERIMENT, split, "hard_negative")
        merged = base.merge(experiment, on="category", how="outer")
        merged.insert(0, "split", split)
        merged["ap_delta"] = merged.hard_negative_ap - merged.baseline_ap
        all_splits.append(merged)
        summary.append({"split": split, "categories": len(merged),
                        "improved": int((merged.ap_delta > 0).sum()),
                        "degraded": int((merged.ap_delta < 0).sum()),
                        "mean_category_delta": float(merged.ap_delta.mean())})
    result = pd.concat(all_splits, ignore_index=True)
    transformations = pd.read_parquet(TRANSFORMATIONS)
    counts = transformations.groupby("category").size().rename("synthetic_pairs")
    result = result.merge(counts, on="category", how="left")
    result["synthetic_pairs"] = result.synthetic_pairs.fillna(0).astype(int)
    result.to_csv(OUTPUT / "category_ap_deltas.csv", index=False, encoding="utf-8-sig")
    attribute_counts = transformations.groupby(["category", "attribute"]).size().rename("synthetic_pairs").reset_index()
    iid_delta = result[result.split == "iid"][["category", "ap_delta"]].rename(columns={"ap_delta": "iid_ap_delta"})
    hard_delta = result[result.split == "hard"][["category", "ap_delta"]].rename(columns={"ap_delta": "hard_ap_delta"})
    attribute_counts.merge(iid_delta, on="category", how="left").merge(
        hard_delta, on="category", how="left").to_csv(
            OUTPUT / "attribute_counts_with_category_deltas.csv", index=False, encoding="utf-8-sig")
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
