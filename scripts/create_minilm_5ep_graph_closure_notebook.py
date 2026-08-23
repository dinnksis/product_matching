#!/usr/bin/env python3
"""Build the MiniLM 5ep ablation with train-only graph closure."""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
BASE_NOTEBOOK = (
    ROOT
    / "notebooks"
    / "minilm_5ep_team_ablation"
    / "minilm_5ep_team_ablation_2xt4.ipynb"
)
DEFAULT_OUTPUT = (
    ROOT
    / "notebooks"
    / "minilm_5ep_graph_closure"
    / "minilm_5ep_graph_closure_2xt4.ipynb"
)
DATA_VARIANT = "human_plus_graph_closure_v1"
EXPECTED_HUMAN_PAIRS = 306_669
EXPECTED_TRANSITIVE_POSITIVES = 1_739
EXPECTED_PROPAGATED_NEGATIVES = 4_024


DATA_HOOK_SOURCE = dedent(
    f"""
    def build_train_data(human_train_pairs, human_items, input_root):
        # Add only consequences of the equivalence relation within human train:
        # A = B and B = C imply A = C; A = B and B != C imply A != C.
        from collections import defaultdict
        from itertools import combinations

        expected_human_pairs = {EXPECTED_HUMAN_PAIRS}
        expected_positive_additions = {EXPECTED_TRANSITIVE_POSITIVES}
        expected_negative_additions = {EXPECTED_PROPAGATED_NEGATIVES}
        if len(human_train_pairs) != expected_human_pairs:
            raise RuntimeError(
                f"Expected {{expected_human_pairs}} frozen human train pairs, "
                f"got {{len(human_train_pairs)}}"
            )

        base = human_train_pairs[["id1", "id2", "target"]].copy()
        base["id1"] = base["id1"].astype("int64")
        base["id2"] = base["id2"].astype("int64")
        base["target"] = base["target"].astype("float64")
        if not base["target"].isin([0.0, 1.0]).all():
            raise ValueError("Graph closure requires binary human labels")

        all_ids = set(base["id1"].tolist()) | set(base["id2"].tolist())
        parent = {{item_id: item_id for item_id in all_ids}}
        component_size = {{item_id: 1 for item_id in all_ids}}

        def find(item_id):
            root = item_id
            while parent[root] != root:
                root = parent[root]
            while parent[item_id] != item_id:
                next_id = parent[item_id]
                parent[item_id] = root
                item_id = next_id
            return root

        def union(first_id, second_id):
            first_root = find(first_id)
            second_root = find(second_id)
            if first_root == second_root:
                return
            first_key = (component_size[first_root], -first_root)
            second_key = (component_size[second_root], -second_root)
            if first_key < second_key:
                first_root, second_root = second_root, first_root
            parent[second_root] = first_root
            component_size[first_root] += component_size[second_root]

        positive_rows = base.loc[base["target"].eq(1.0), ["id1", "id2"]]
        for first_id, second_id in positive_rows.itertuples(index=False, name=None):
            union(int(first_id), int(second_id))

        components = defaultdict(list)
        for item_id in sorted(all_ids):
            components[find(item_id)].append(item_id)
        component_key = {{
            root: members[0] for root, members in components.items()
        }}

        known = {{}}
        for first_id, second_id, target in base.itertuples(index=False, name=None):
            pair = tuple(sorted((int(first_id), int(second_id))))
            binary_target = int(target)
            previous = known.setdefault(pair, binary_target)
            if previous != binary_target:
                raise ValueError(f"Conflicting human labels for pair {{pair}}")

        negative_component_pairs = set()
        negative_rows = base.loc[base["target"].eq(0.0), ["id1", "id2"]]
        for first_id, second_id in negative_rows.itertuples(index=False, name=None):
            first_root = find(int(first_id))
            second_root = find(int(second_id))
            if first_root == second_root:
                raise ValueError(
                    "Human negative edge occurs inside a positive component: "
                    f"{{first_id}}, {{second_id}}"
                )
            roots = (first_root, second_root)
            if component_key[first_root] > component_key[second_root]:
                roots = (second_root, first_root)
            negative_component_pairs.add(roots)

        transitive_positives = []
        for members in components.values():
            if len(members) < 3:
                continue
            for first_id, second_id in combinations(members, 2):
                pair = (first_id, second_id)
                if pair not in known:
                    transitive_positives.append((*pair, 1.0))

        propagated_negatives = set()
        for first_root, second_root in negative_component_pairs:
            for first_id in components[first_root]:
                for second_id in components[second_root]:
                    pair = tuple(sorted((first_id, second_id)))
                    if pair not in known:
                        propagated_negatives.add((*pair, 0.0))

        transitive_positives.sort()
        propagated_negatives = sorted(propagated_negatives)
        if len(transitive_positives) != expected_positive_additions:
            raise RuntimeError(
                "Unexpected transitive-positive count: "
                f"{{len(transitive_positives)}} != {{expected_positive_additions}}"
            )
        if len(propagated_negatives) != expected_negative_additions:
            raise RuntimeError(
                "Unexpected propagated-negative count: "
                f"{{len(propagated_negatives)}} != {{expected_negative_additions}}"
            )

        human = base.copy()
        human["label_source"] = "human"
        positive_extra = pd.DataFrame(
            transitive_positives, columns=["id1", "id2", "target"]
        )
        positive_extra["label_source"] = "graph_transitive_positive"
        negative_extra = pd.DataFrame(
            propagated_negatives, columns=["id1", "id2", "target"]
        )
        negative_extra["label_source"] = "graph_propagated_negative"
        train_pairs = pd.concat(
            [human, positive_extra, negative_extra], ignore_index=True
        )

        categories = human_items.set_index("id")["category"]
        left_categories = categories.loc[train_pairs["id1"]].to_numpy()
        right_categories = categories.loc[train_pairs["id2"]].to_numpy()
        if not (left_categories == right_categories).all():
            raise ValueError("Graph closure produced a cross-category pair")

        return train_pairs, human_items.copy()
    """
).strip()


def build_graph_closure_notebook(
    base_notebook: Path = BASE_NOTEBOOK,
) -> nbf.NotebookNode:
    # Start from the committed locked notebook rather than regenerating it from
    # the current worktree. This guarantees that unrelated source edits cannot
    # leak into frozen cells of the controlled ablation.
    notebook = nbf.read(base_notebook, as_version=4)
    data_hook_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code"
        and "data-hook" in cell.metadata.get("tags", [])
    ]
    if len(data_hook_cells) != 1:
        raise ValueError(f"Expected one editable data hook, found {len(data_hook_cells)}")
    data_hook_cells[0].source = DATA_HOOK_SOURCE
    notebook.metadata["product_matching_training"]["data_variant"] = DATA_VARIANT
    nbf.validate(notebook)
    return notebook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-notebook", type=Path, default=BASE_NOTEBOOK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notebook = build_graph_closure_notebook(args.base_notebook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, args.output)
    print(f"Wrote notebook: {args.output}")
    print(f"Data variant: {DATA_VARIANT}")
    print(
        "Expected train pairs: "
        f"{EXPECTED_HUMAN_PAIRS + EXPECTED_TRANSITIVE_POSITIVES + EXPECTED_PROPAGATED_NEGATIVES}"
    )


if __name__ == "__main__":
    main()
