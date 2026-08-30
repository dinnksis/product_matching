from __future__ import annotations

import unittest
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "minilm_5ep_full_human_final_2xt4.ipynb"


class Minilm5epFullHumanNotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(cls.notebook)
        cls.source = "\n".join(
            cell.source for cell in cls.notebook.cells if cell.cell_type == "code"
        )

    def test_every_code_cell_compiles(self) -> None:
        for index, cell in enumerate(self.notebook.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"notebook-cell-{index}", "exec")

    def test_locked_recipe_and_no_validation(self) -> None:
        metadata = self.notebook.metadata["product_matching_final_training"]
        config = metadata["config"]
        self.assertEqual(
            metadata["checkpoint_dataset"],
            "alexproger23/product-matching-minilm-llm-pretrain-5ep-full",
        )
        self.assertEqual(config["epochs"], 3)
        self.assertEqual(config["batch_size"], 96)
        self.assertEqual(config["gradient_accumulation"], 1)
        self.assertEqual(config["learning_rate"], 8e-5)
        self.assertEqual(config["warmup_ratio"], 0.05)
        self.assertEqual(config["weight_decay"], 0.01)
        self.assertEqual(config["classifier_dropout"], 0.1)
        self.assertEqual(config["max_length"], 384)
        self.assertEqual(config["sampling"], "none")
        self.assertEqual(config["loss_weighting"], "none")
        self.assertEqual(config["label_smoothing"], 0.0)
        self.assertEqual(config["max_grad_norm"], 1.0)
        self.assertEqual(config["seed"], 42)
        self.assertTrue(config["skip_validation"])

    def test_full_human_data_is_asserted_without_filtering(self) -> None:
        metadata = self.notebook.metadata["product_matching_final_training"]
        self.assertEqual(metadata["data_audit"]["items_rows"], 711_304)
        self.assertEqual(metadata["data_audit"]["pairs_rows"], 365_654)
        self.assertEqual(metadata["data_audit"]["contradictory_pairs"], 0)
        preparation_source = next(
            cell.source
            for cell in self.notebook.cells
            if cell.cell_type == "code" and "preparation_started" in cell.source
        )
        self.assertIn('"filtered_pairs": 0', preparation_source)
        self.assertIn('PREPARED_DIR / "train_pairs.parquet"', preparation_source)
        self.assertNotIn("train_test_split", preparation_source)
        self.assertNotIn("component_split", preparation_source)

    def test_serialization_and_local_only_outputs(self) -> None:
        self.assertIn('str.startswith("Категория: ")', self.source)
        self.assertIn('str.contains("\\nНазвание: ", regex=False)', self.source)
        self.assertNotIn("google_sheets", self.source.lower())
        self.assertNotIn("gspread", self.source.lower())
        self.assertIn('WORKING_ROOT / "notebook_completed.json"', self.source)
        self.assertIn("checkpoint_manifest.json", self.source)
        self.assertIn('reconstruction["parts"]', self.source)


if __name__ == "__main__":
    unittest.main()
