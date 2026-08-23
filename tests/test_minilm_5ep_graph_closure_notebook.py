from __future__ import annotations

import unittest
from pathlib import Path

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    ROOT
    / "notebooks"
    / "minilm_5ep_graph_closure"
    / "minilm_5ep_graph_closure_2xt4.ipynb"
)
FROZEN_DATA = ROOT / ".kaggle" / "datasets" / "product-matching-validation-splits-v1"


class Minilm5epGraphClosureNotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(cls.notebook)
        cells = [
            cell
            for cell in cls.notebook.cells
            if cell.cell_type == "code"
            and "data-hook" in cell.metadata.get("tags", [])
        ]
        if len(cells) != 1:
            raise AssertionError(f"Expected one data hook, found {len(cells)}")
        cls.data_hook = cells[0].source

    def test_notebook_declares_graph_closure_variant(self) -> None:
        metadata = self.notebook.metadata["product_matching_training"]
        self.assertEqual(metadata["data_variant"], "human_plus_graph_closure_v1")
        self.assertIn("graph_transitive_positive", self.data_hook)
        self.assertIn("graph_propagated_negative", self.data_hook)

    def test_only_data_hook_changes_labels(self) -> None:
        editable_code = [
            cell
            for cell in self.notebook.cells
            if cell.cell_type == "code"
            and "team-editable" in cell.metadata.get("tags", [])
        ]
        self.assertEqual(len(editable_code), 2)
        self.assertIn("def build_train_data", editable_code[0].source)
        self.assertIn("binary_cross_entropy_with_logits", editable_code[1].source)
        self.assertNotIn("sample_weight", editable_code[0].source)

    @unittest.skipUnless(FROZEN_DATA.is_dir(), "frozen Kaggle data is unavailable")
    def test_graph_closure_exact_counts_and_invariants(self) -> None:
        namespace = {"pd": pd}
        exec(self.data_hook, namespace)
        human_pairs = pd.read_parquet(FROZEN_DATA / "human_train_pairs.parquet")
        human_items = pd.read_parquet(FROZEN_DATA / "human_items.parquet")
        train_pairs, items = namespace["build_train_data"](
            human_pairs, human_items, ROOT
        )

        self.assertEqual(len(train_pairs), 312_432)
        self.assertEqual(
            train_pairs["label_source"].value_counts().to_dict(),
            {
                "human": 306_669,
                "graph_propagated_negative": 4_024,
                "graph_transitive_positive": 1_739,
            },
        )
        self.assertEqual(len(items), len(human_items))
        self.assertFalse(train_pairs[["id1", "id2", "target"]].isna().any().any())
        self.assertTrue(train_pairs["target"].isin([0.0, 1.0]).all())

        unordered = pd.DataFrame(
            {
                "left": train_pairs[["id1", "id2"]].min(axis=1),
                "right": train_pairs[["id1", "id2"]].max(axis=1),
            }
        )
        self.assertFalse(unordered.duplicated().any())


if __name__ == "__main__":
    unittest.main()
