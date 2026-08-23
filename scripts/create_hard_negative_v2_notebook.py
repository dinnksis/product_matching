"""Create MiniLM hard-negative v2 notebook with class-mass compensation."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks/minilm_5ep_team_ablation/minilm_5ep_team_ablation_2xt4.ipynb"
OUTPUT = ROOT / "notebooks/minilm_5ep_team_ablation/minilm_5ep_hard_negatives_v2_2xt4.ipynb"

DATA_HOOK = '''def build_train_data(human_train_pairs, human_items, input_root):
    pair_candidates = list(input_root.glob("**/hard_negative_pairs.parquet"))
    item_candidates = list(input_root.glob("**/hard_negative_items.parquet"))
    if len(pair_candidates) != 1 or len(item_candidates) != 1:
        raise RuntimeError(f"Expected one v2 pair/item file, got {pair_candidates}, {item_candidates}")
    extra_pairs = pd.read_parquet(pair_candidates[0])
    extra_items = pd.read_parquet(item_candidates[0])
    base_pairs = human_train_pairs.copy()
    base_pairs["sample_weight"] = 1.0
    base_pairs["label_source"] = "human"
    categories = human_items.set_index("id")["category"]
    base_pairs["weight_category"] = base_pairs["id1"].map(categories)
    extra_pairs["weight_category"] = extra_pairs["id1"].map(
        extra_items.set_index("id")["category"]
    ).fillna(extra_pairs["id2"].map(extra_items.set_index("id")["category"]))
    # Preserve the original total target=0 mass in every affected category.
    for category, group in extra_pairs.groupby("weight_category"):
        added_mass = float(group["sample_weight"].sum())
        mask = (base_pairs["weight_category"] == category) & (base_pairs["target"].astype(float) < 0.5)
        human_negative_count = int(mask.sum())
        if not human_negative_count or added_mass >= human_negative_count:
            raise ValueError(f"Cannot compensate category {category}: {added_mass=}, {human_negative_count=}")
        base_pairs.loc[mask, "sample_weight"] = 1.0 - added_mass / human_negative_count
    columns = list(base_pairs.columns)
    train_pairs = pd.concat([base_pairs, extra_pairs[columns]], ignore_index=True)
    items = pd.concat([human_items, extra_items[list(human_items.columns)]], ignore_index=True)
    print({"human_pairs": len(base_pairs), "synthetic_pairs": len(extra_pairs),
           "total_pairs": len(train_pairs), "synthetic_weight": 0.2})
    return train_pairs, items
'''


def main() -> None:
    notebook = nbformat.read(SOURCE, as_version=4)
    editable = [c for c in notebook.cells if c.cell_type == "code" and
                "data-hook" in c.get("metadata", {}).get("tags", [])]
    if len(editable) != 1:
        raise RuntimeError(f"Expected one DATA HOOK cell, got {len(editable)}")
    editable[0].source = DATA_HOOK
    nbformat.validate(notebook)
    nbformat.write(notebook, OUTPUT)
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    main()
