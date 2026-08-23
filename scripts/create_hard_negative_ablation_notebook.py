"""Create the MiniLM ablation notebook with only its DATA HOOK changed."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "notebooks"
    / "minilm_5ep_team_ablation"
    / "minilm_5ep_team_ablation_2xt4.ipynb"
)
OUTPUT = (
    ROOT
    / "notebooks"
    / "minilm_5ep_team_ablation"
    / "minilm_5ep_hard_negatives_v1_2xt4.ipynb"
)

DATA_HOOK = '''def build_train_data(human_train_pairs, human_items, input_root):
    pair_candidates = list(input_root.glob("**/hard_negative_pairs.parquet"))
    item_candidates = list(input_root.glob("**/hard_negative_items.parquet"))
    if len(pair_candidates) != 1 or len(item_candidates) != 1:
        raise RuntimeError(
            "Expected exactly one hard-negative pair/item file; "
            f"pairs={pair_candidates}, items={item_candidates}"
        )

    extra_pairs = pd.read_parquet(pair_candidates[0])
    extra_items = pd.read_parquet(item_candidates[0])
    required_pair_columns = {"id1", "id2", "target", "sample_weight", "label_source"}
    required_item_columns = {"id", "name", "category", "product_text"}
    if missing := required_pair_columns - set(extra_pairs.columns):
        raise ValueError(f"Hard-negative pairs missing columns: {sorted(missing)}")
    if missing := required_item_columns - set(extra_items.columns):
        raise ValueError(f"Hard-negative items missing columns: {sorted(missing)}")

    base_pairs = human_train_pairs.copy()
    base_pairs["sample_weight"] = 1.0
    base_pairs["label_source"] = "human"
    train_pairs = pd.concat(
        [base_pairs, extra_pairs[list(base_pairs.columns)]], ignore_index=True
    )
    items = pd.concat(
        [human_items, extra_items[list(human_items.columns)]], ignore_index=True
    )
    print({
        "human_pairs": len(base_pairs),
        "hard_negative_pairs": len(extra_pairs),
        "total_pairs": len(train_pairs),
        "label_sources": train_pairs["label_source"].value_counts().to_dict(),
    })
    return train_pairs, items
'''


def main() -> None:
    notebook = nbformat.read(SOURCE, as_version=4)
    editable = [
        cell
        for cell in notebook.cells
        if "data-hook" in cell.get("metadata", {}).get("tags", [])
        and cell.cell_type == "code"
    ]
    if len(editable) != 1:
        raise RuntimeError(f"Expected one editable DATA HOOK code cell, got {len(editable)}")
    editable[0].source = DATA_HOOK
    nbformat.validate(notebook)
    nbformat.write(notebook, OUTPUT)
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    main()
