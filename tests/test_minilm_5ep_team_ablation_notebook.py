from __future__ import annotations

import unittest
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    ROOT
    / "notebooks"
    / "minilm_5ep_team_ablation"
    / "minilm_5ep_team_ablation_2xt4.ipynb"
)


class Minilm5epTeamAblationNotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(cls.notebook)

    def test_only_data_and_loss_code_cells_are_editable(self) -> None:
        editable_code = [
            cell
            for cell in self.notebook.cells
            if cell.cell_type == "code"
            and "team-editable" in cell.metadata.get("tags", [])
        ]
        self.assertEqual(len(editable_code), 2)
        self.assertIn("def build_train_data", editable_code[0].source)
        self.assertIn("def compute_loss", editable_code[1].source)
        self.assertIn("%%writefile", editable_code[1].source)

    def test_training_command_uses_loss_hook_and_three_frozen_splits(self) -> None:
        source = "\n".join(
            cell.source for cell in self.notebook.cells if cell.cell_type == "code"
        )
        self.assertIn('"--loss-hook", str(LOSS_HOOK_PATH)', source)
        self.assertIn('"iid=iid_validation_pairs.parquet"', source)
        self.assertIn('"hard=hard_validation_pairs.parquet"', source)
        self.assertIn('"ood=ood_validation_pairs.parquet"', source)

    def test_provenance_contains_recipe_loss_and_data_hashes(self) -> None:
        source = "\n".join(
            cell.source for cell in self.notebook.cells if cell.cell_type == "code"
        )
        self.assertIn("EXPECTED_TRAIN_RECIPE_SHA256", source)
        self.assertIn('"loss_hook_sha256": LOSS_HOOK_SHA256', source)
        self.assertIn('"train_pairs_sha256": file_sha256(train_pairs_path)', source)
        self.assertIn('"items_sha256": file_sha256(items_path)', source)


if __name__ == "__main__":
    unittest.main()
