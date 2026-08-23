"""Create three frozen MiniLM 5ep data-ablation notebooks."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks/minilm_5ep_team_ablation/minilm_5ep_team_ablation_2xt4.ipynb"
OUT_DIR = ROOT / "notebooks/minilm_5ep_team_ablation"


def hook(dataset: str, item_dataset: str) -> str:
    return f'''def build_train_data(human_train_pairs, human_items, input_root):
    pair_candidates = list(input_root.glob("**/{dataset}.parquet"))
    item_candidates = list(input_root.glob("**/{item_dataset}.parquet"))
    if len(pair_candidates) != 1 or len(item_candidates) != 1:
        raise RuntimeError(
            "Expected exactly one targeted synthetic pair/item file; "
            f"pairs={{pair_candidates}}, items={{item_candidates}}"
        )

    extra_pairs = pd.read_parquet(pair_candidates[0])
    extra_items = pd.read_parquet(item_candidates[0])
    required_pair_columns = {{"id1", "id2", "target", "label_source"}}
    required_item_columns = {{"id", "name", "category", "product_text"}}
    if missing := required_pair_columns - set(extra_pairs.columns):
        raise ValueError(f"Synthetic pairs missing columns: {{sorted(missing)}}")
    if missing := required_item_columns - set(extra_items.columns):
        raise ValueError(f"Synthetic items missing columns: {{sorted(missing)}}")
    if extra_pairs[["id1", "id2", "target"]].isna().any().any():
        raise ValueError("Synthetic pairs contain null ids/targets")

    base_pairs = human_train_pairs.copy()
    base_pairs["sample_weight"] = 1.0
    base_pairs["label_source"] = "human"
    extra_pairs = extra_pairs.copy()
    extra_pairs["sample_weight"] = 1.0
    extra_pairs["label_source"] = extra_pairs["label_source"].astype(str)
    train_pairs = pd.concat(
        [base_pairs, extra_pairs[list(base_pairs.columns)]], ignore_index=True
    )
    items = pd.concat(
        [human_items, extra_items[list(human_items.columns)]], ignore_index=True
    )
    if items["id"].duplicated().any():
        raise ValueError("Human and synthetic item ids overlap")
    print({{
        "human_pairs": len(base_pairs),
        "synthetic_pairs": len(extra_pairs),
        "total_pairs": len(train_pairs),
        "human_weight": float(base_pairs["sample_weight"].sum()),
        "synthetic_weight": float(extra_pairs["sample_weight"].sum()),
        "label_sources": train_pairs["label_source"].value_counts().to_dict(),
    }})
    return train_pairs, items
'''


EXPERIMENTS = {
    "hard_negatives_v3": {
        "filename": "minilm_5ep_hard_negatives_v3_2xt4.ipynb",
        "label": "minilm_5ep_hard_negatives_v3",
        "notes": (
            "Human train + Qwen-validated category-aware hard negatives v3; "
            "synthetic sample_weight=1.0; source dataset "
            "dinakepecheva/product-matching-targeted-synthetic-v3."
        ),
    },
    "hard_positives_v1": {
        "filename": "minilm_5ep_hard_positives_v1_2xt4.ipynb",
        "label": "minilm_5ep_hard_positives_v1",
        "notes": (
            "Human train + Qwen-validated positive rewrites v1; synthetic "
            "sample_weight=1.0. Historical seed ordering skewed this parquet "
            "toward high-S2 positives (median about 0.939), so this run tests "
            "representation augmentation rather than low-S2 targeting. Source "
            "dataset dinakepecheva/product-matching-targeted-synthetic-v3."
        ),
    },
    "ood_style_positives_v1": {
        "filename": "minilm_5ep_ood_style_positives_v1_2xt4.ipynb",
        "label": "minilm_5ep_ood_style_positives_v1",
        "notes": (
            "Human train + Qwen-validated OOD-style positives v1; "
            "synthetic sample_weight=1.0; source dataset "
            "dinakepecheva/product-matching-targeted-synthetic-v3."
        ),
    },
}


def routing_cell(label: str, notes: str) -> str:
    return (
        f"EXPERIMENT_LABEL = {label!r}\n"
        "EXPERIMENT_SHEET = 'data_exps'  # pretrain_exps | sft_exps | data_exps\n"
        f"EXPERIMENT_NOTES = {notes!r}\n"
    )


def main() -> None:
    source = nbformat.read(SOURCE, as_version=4)
    for dataset, config in EXPERIMENTS.items():
        notebook = nbformat.from_dict(source)
        data_cells = [
            cell
            for cell in notebook.cells
            if cell.cell_type == "code"
            and "data-hook" in cell.get("metadata", {}).get("tags", [])
        ]
        routing_cells = [
            cell
            for cell in notebook.cells
            if cell.cell_type == "code"
            and "experiment-routing" in cell.get("metadata", {}).get("tags", [])
        ]
        if len(data_cells) != 1 or len(routing_cells) != 1:
            raise RuntimeError(
                f"Expected one data hook and one routing cell, got "
                f"{len(data_cells)} and {len(routing_cells)}"
            )
        data_cells[0].source = hook(dataset, f"{dataset}_items")
        routing_cells[0].source = routing_cell(config["label"], config["notes"])

        for original, generated in zip(source.cells, notebook.cells, strict=True):
            tags = original.get("metadata", {}).get("tags", [])
            if "frozen" in tags and original.source != generated.source:
                raise RuntimeError("A frozen notebook cell was modified")

        nbformat.validate(notebook)
        output = OUT_DIR / config["filename"]
        nbformat.write(notebook, output)
        print(f"Created {output}")


if __name__ == "__main__":
    main()
