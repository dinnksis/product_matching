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

    def test_run_setup_selects_exact_comparison_sheet(self) -> None:
        routing_cells = [
            cell
            for cell in self.notebook.cells
            if cell.cell_type == "code"
            and "experiment-routing" in cell.metadata.get("tags", [])
        ]
        self.assertEqual(len(routing_cells), 1)
        source = routing_cells[0].source
        self.assertIn("EXPERIMENT_LABEL =", source)
        self.assertIn("EXPERIMENT_SHEET = 'data_exps'", source)
        self.assertIn("pretrain_exps | sft_exps | data_exps", source)
        self.assertIn("EXPERIMENT_NOTES =", source)

    def test_frozen_baseline_and_significance_are_required_before_sync(self) -> None:
        code_cells = [
            cell for cell in self.notebook.cells if cell.cell_type == "code"
        ]
        source = "\n".join(cell.source for cell in code_cells)
        baseline_index = next(
            index
            for index, cell in enumerate(code_cells)
            if "SIGNIFICANCE_BASELINE_MANIFEST_SHA256" in cell.source
        )
        comparison_index = next(
            index
            for index, cell in enumerate(code_cells)
            if "compare_experiment_directories(" in cell.source
        )
        sync_index = next(
            index
            for index, cell in enumerate(code_cells)
            if "sync_from_kaggle_credentials(" in cell.source
        )
        self.assertLess(baseline_index, comparison_index)
        self.assertLess(comparison_index, sync_index)
        self.assertIn(
            "alexproger23/product-matching-minilm-5ep-significance-v1",
            source,
        )
        self.assertIn("67f4fe76886b43d6b52ed5cb49068e1e", source)
        self.assertIn('completion["baseline_comparison"] = baseline_comparison', source)
        self.assertIn('"experiment_group": EXPERIMENT_GROUP', source)

    def test_metadata_declares_routing_and_significance_baseline(self) -> None:
        metadata = self.notebook.metadata["product_matching_training"]
        self.assertEqual(metadata["run_configurable_cell"], "experiment-routing")
        self.assertEqual(metadata["default_experiment_sheet"], "data_exps")
        self.assertEqual(
            metadata["significance_baseline_run_id"],
            "67f4fe76886b43d6b52ed5cb49068e1e",
        )


if __name__ == "__main__":
    unittest.main()
