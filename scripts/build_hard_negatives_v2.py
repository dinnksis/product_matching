"""Filter hard negatives v1 using validation category deltas."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "prepared/hard_negatives_v1"
REPORT = ROOT / "reports/hard_negatives_v1/category_ap_deltas.csv"
OUTPUT = ROOT / "prepared/hard_negatives_v2"


def main() -> None:
    deltas = pd.read_csv(REPORT)
    iid = deltas[deltas.split == "iid"].set_index("category").ap_delta
    hard = deltas[deltas.split == "hard"].set_index("category").ap_delta
    common = iid.index.intersection(hard.index)
    approved = sorted(c for c in common if hard[c] > 0 and iid[c] + hard[c] >= -0.005)
    pairs = pd.read_parquet(V1 / "hard_negative_pairs.parquet")
    items = pd.read_parquet(V1 / "hard_negative_items.parquet")
    transformations = pd.read_parquet(V1 / "transformations.parquet")
    kept_transformations = transformations[transformations.category.isin(approved)].copy()
    kept_ids = set(kept_transformations.synthetic_id)
    kept_pairs = pairs[pairs.id1.isin(kept_ids) | pairs.id2.isin(kept_ids)].copy()
    kept_items = items[items.id.isin(kept_ids)].copy()
    kept_pairs["sample_weight"] = 0.2
    kept_pairs["label_source"] = "hard_negative_v2_filtered_weight_0.2"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    kept_pairs.to_parquet(OUTPUT / "hard_negative_pairs.parquet", index=False)
    kept_items.to_parquet(OUTPUT / "hard_negative_items.parquet", index=False)
    kept_transformations.to_parquet(OUTPUT / "transformations.parquet", index=False)
    manifest = {
        "source": "hard_negatives_v1", "synthetic_weight": 0.2,
        "filter": "hard_ap_delta > 0 and iid_ap_delta + hard_ap_delta >= -0.005",
        "approved_categories": approved, "pairs": len(kept_pairs), "items": len(kept_items),
        "category_counts": kept_transformations.category.value_counts().to_dict(),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
