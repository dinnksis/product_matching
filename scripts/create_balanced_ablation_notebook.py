"""Create a MiniLM ablation with category x binary-class balanced loss weights."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks/minilm_5ep_team_ablation/minilm_5ep_team_ablation_2xt4.ipynb"
OUTPUT = ROOT / "notebooks/minilm_5ep_team_ablation/minilm_5ep_category_class_balanced_2xt4.ipynb"

DATA_HOOK = '''def build_train_data(human_train_pairs, human_items, input_root):
    train_pairs = human_train_pairs.copy()
    item_categories = human_items.set_index("id")["category"]
    train_pairs["balance_category"] = train_pairs["id1"].map(item_categories)
    missing = int(train_pairs["balance_category"].isna().sum())
    if missing:
        raise ValueError(f"Missing category for {missing} training pairs")
    train_pairs["balance_class"] = (train_pairs["target"].astype(float) >= 0.5).astype(int)
    counts = train_pairs.groupby(["balance_category", "balance_class"])["id1"].transform("size")
    number_of_strata = train_pairs[["balance_category", "balance_class"]].drop_duplicates().shape[0]
    expected_strata = 2 * train_pairs["balance_category"].nunique()
    if number_of_strata != expected_strata:
        raise ValueError(f"Expected {expected_strata} non-empty category/class strata, got {number_of_strata}")
    # Every category x class stratum has equal total weight; mean weight remains one.
    train_pairs["sample_weight"] = len(train_pairs) / (number_of_strata * counts)
    train_pairs["label_source"] = "human_balanced_category_class"
    totals = train_pairs.groupby(["balance_category", "balance_class"])["sample_weight"].sum()
    print({
        "pairs": len(train_pairs), "strata": number_of_strata,
        "weight_min": float(train_pairs["sample_weight"].min()),
        "weight_max": float(train_pairs["sample_weight"].max()),
        "stratum_total_min": float(totals.min()),
        "stratum_total_max": float(totals.max()),
    })
    return train_pairs, human_items.copy()
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
