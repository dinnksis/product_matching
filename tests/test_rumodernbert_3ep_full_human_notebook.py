from __future__ import annotations

import unittest
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "rumodernbert_3ep_full_human_final_2xt4.ipynb"


class RuModernBert3epFullHumanNotebookTest(unittest.TestCase):
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

    def test_locked_recipe(self) -> None:
        metadata = self.notebook.metadata["product_matching_final_training"]
        config = metadata["config"]
        self.assertEqual(metadata["architecture"], "RuModernBERT")
        self.assertEqual(
            metadata["checkpoint_dataset"],
            "alexproger23/product-matching-rumodernbert-pretrain-3ep",
        )
        expected = {
            "epochs": 3,
            "batch_size": 24,
            "gradient_accumulation": 4,
            "learning_rate": 4e-5,
            "warmup_ratio": 0.05,
            "scheduler": "cosine",
            "weight_decay": 0.01,
            "max_length": 384,
            "attention_implementation": "sdpa",
            "sampling": "none",
            "loss_weighting": "none",
            "label_smoothing": 0.0,
            "max_grad_norm": 0.5,
            "gradient_checkpointing": False,
            "seed": 42,
            "dataloader_workers": 4,
            "prefetch_factor": 2,
            "bucket_size_multiplier": 50,
            "skip_validation": True,
        }
        for key, value in expected.items():
            self.assertEqual(config[key], value, key)

    def test_full_human_data_and_serialization(self) -> None:
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
        self.assertIn('str.startswith("Категория: ")', preparation_source)
        self.assertIn(
            'str.contains("\\nНазвание: ", regex=False)', preparation_source
        )

    def test_runtime_contract_and_outputs(self) -> None:
        self.assertIn('"--nproc_per_node=2"', self.source)
        self.assertIn('EXPECTED_AMP_DTYPE = \'torch.float16\'', self.source)
        self.assertIn("EXPECTED_EFFECTIVE_BATCH_SIZE = 192", self.source)
        self.assertIn("checkpoint_manifest.json", self.source)
        self.assertIn('reconstruction["parts"]', self.source)
        self.assertNotIn("google_sheets", self.source.lower())
        self.assertNotIn("gspread", self.source.lower())
        self.assertIn('WORKING_ROOT / "notebook_completed.json"', self.source)


if __name__ == "__main__":
    unittest.main()
