#!/usr/bin/env python3
"""Create the human-only MiniLM OOF-error curriculum Kaggle notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

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
    / "minilm_5ep_human_error_curriculum_v1_2xt4.ipynb"
)
EXPERIMENT_LABEL = "minilm_5ep_human_error_curriculum_v1"
EXPERIMENT_SHEET = "data_exps"
EXPERIMENT_NOTES = (
    "Human-only data ablation: frozen MiniLM 5ep checkpoint, full human train, "
    "plus one unit of loss weight for 9311 audit-eligible RULE_DISCOVERY OOF-error/"
    "hard pairs. No synthetic pairs or Qwen labels."
)


DATA_HOOK = '''def build_train_data(human_train_pairs, human_items, input_root):
    candidates = list(input_root.glob("**/human_error_curriculum_pairs.parquet"))
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one human_error_curriculum_pairs.parquet input; "
            f"found={candidates}"
        )

    curriculum = pd.read_parquet(candidates[0])
    required = {"id1", "id2", "target"}
    if missing := required - set(curriculum.columns):
        raise ValueError(f"Curriculum is missing columns: {sorted(missing)}")
    curriculum = curriculum[["id1", "id2", "target"]].copy()
    if len(curriculum) != 9311:
        raise ValueError(f"Expected 9311 curriculum pairs, got {len(curriculum)}")
    if curriculum[["id1", "id2", "target"]].isna().any().any():
        raise ValueError("Curriculum contains null id/target values")
    if not curriculum["target"].between(0, 1).all():
        raise ValueError("Curriculum targets must be human labels in [0, 1]")

    def unordered_keys(frame):
        left = np.minimum(frame["id1"].to_numpy(), frame["id2"].to_numpy())
        right = np.maximum(frame["id1"].to_numpy(), frame["id2"].to_numpy())
        return pd.Series(
            [f"{a}|{b}" for a, b in zip(left, right)], index=frame.index, dtype="string"
        )

    train_pairs = human_train_pairs.copy()
    train_keys = unordered_keys(train_pairs)
    curriculum_keys = unordered_keys(curriculum)
    if train_keys.duplicated().any():
        raise ValueError("Frozen human train unexpectedly contains duplicate unordered pairs")
    if curriculum_keys.duplicated().any():
        raise ValueError("Curriculum contains duplicate unordered pairs")

    train_targets = pd.Series(
        train_pairs["target"].astype(float).to_numpy(), index=train_keys.to_numpy()
    )
    missing_keys = set(curriculum_keys) - set(train_keys)
    if missing_keys:
        raise ValueError(
            f"Curriculum contains {len(missing_keys)} pairs absent from frozen human train"
        )
    expected_targets = curriculum_keys.map(train_targets)
    if not np.allclose(expected_targets.to_numpy(dtype=float), curriculum["target"].to_numpy(dtype=float)):
        raise ValueError("Curriculum target differs from its frozen human label")

    selected = train_keys.isin(set(curriculum_keys))
    if int(selected.sum()) != len(curriculum):
        raise ValueError("Curriculum-to-train mapping is not one-to-one")

    # One original human observation has weight 1.0. The selected rows receive
    # one additional unit of weight, equivalent to appending one weight-1 copy,
    # while preserving the frozen notebook's no-duplicate-pairs invariant.
    train_pairs["sample_weight"] = 1.0
    train_pairs.loc[selected.to_numpy(), "sample_weight"] = 2.0
    train_pairs["label_source"] = "human"
    train_pairs.loc[selected.to_numpy(), "label_source"] = "human_oof_curriculum_x2"

    print({
        "human_pairs": len(train_pairs),
        "curriculum_pairs": len(curriculum),
        "pair_rows_added": 0,
        "additional_human_weight": float(len(curriculum)),
        "effective_training_weight": float(train_pairs["sample_weight"].sum()),
        "weight_min": float(train_pairs["sample_weight"].min()),
        "weight_max": float(train_pairs["sample_weight"].max()),
        "label_sources": train_pairs["label_source"].value_counts().to_dict(),
        "synthetic_pairs": 0,
        "qwen_labels": 0,
    })
    return train_pairs, human_items.copy()
'''


def main() -> None:
    notebook = nbf.read(SOURCE, as_version=4)
    routing = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code"
        and "experiment-routing" in cell.get("metadata", {}).get("tags", [])
    ]
    if len(routing) != 1:
        raise RuntimeError(f"Expected one experiment-routing cell, got {len(routing)}")
    routing[0].source = (
        f"EXPERIMENT_LABEL = {EXPERIMENT_LABEL!r}\n"
        f"EXPERIMENT_SHEET = {EXPERIMENT_SHEET!r}  # pretrain_exps | sft_exps | data_exps\n"
        f"EXPERIMENT_NOTES = {EXPERIMENT_NOTES!r}"
    )
    editable = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code"
        and "data-hook" in cell.get("metadata", {}).get("tags", [])
    ]
    if len(editable) != 1:
        raise RuntimeError(f"Expected one DATA HOOK cell, got {len(editable)}")
    editable[0].source = DATA_HOOK

    sheets_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code"
        and "experiment_action = _upsert_experiment(client, experiment_row)" in cell.source
    ]
    if len(sheets_cells) != 1:
        raise RuntimeError(
            f"Expected one Google Sheets sync cell, got {len(sheets_cells)}"
        )
    sheets_cells[0].source = sheets_cells[0].source.replace(
        "experiment_action = _upsert_experiment(client, experiment_row)",
        'experiment_action = "skipped_data_exps_only"',
    )

    notebook.cells[0].source = """# MiniLM 5ep: human OOF-error curriculum v1

Human-only data ablation. Обучение начинается с frozen MiniLM checkpoint после
пяти эпох pretraining. Основой остаётся полный human train; 9 311 проверенных
RULE_DISCOVERY OOF-error/hard пар получают одну дополнительную единицу веса.
Синтетика и Qwen-labels не используются. IID, hard и OOD остаются неизменными.
"""
    notebook.metadata["product_matching_training"].update(
        {
            "experiment": EXPERIMENT_LABEL,
            "default_experiment_sheet": EXPERIMENT_SHEET,
            "data_ablation": "human_oof_error_curriculum_weight_plus_one",
            "curriculum_pairs": 9311,
            "human_labels_only": True,
            "synthetic_pairs_used": False,
            "qwen_labels_used": False,
            "google_sheets_primary_log": False,
            "google_sheets_comparison_sheet": "data_exps",
        }
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.validate(notebook)
    nbf.write(notebook, OUTPUT)
    print(f"Created {OUTPUT}")
    print(f"Experiment label: {EXPERIMENT_LABEL}")
    print(f"Comparison sheet: {EXPERIMENT_SHEET}")


if __name__ == "__main__":
    main()
