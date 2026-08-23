from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "create_minilm_5ep_category_heads_notebook.py"
NOTEBOOK = (
    ROOT
    / "notebooks"
    / "minilm_5ep_category_heads"
    / "minilm_5ep_category_heads_18_2xt4.ipynb"
)


def load_generator():
    spec = importlib.util.spec_from_file_location("category_heads_generator", GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load category-head notebook generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Minilm5epCategoryHeadsNotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generator = load_generator()
        cls.notebook = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(cls.notebook)

    def test_exact_18_train_categories_are_declared(self) -> None:
        self.assertEqual(len(self.generator.CATEGORY_HEAD_NAMES), 18)
        self.assertEqual(len(set(self.generator.CATEGORY_HEAD_NAMES)), 18)
        self.assertEqual(
            self.notebook.metadata["product_matching_training"]["category_heads"][
                "names"
            ],
            list(self.generator.CATEGORY_HEAD_NAMES),
        )

    def test_routes_experiment_to_sft_table(self) -> None:
        routing = [
            cell
            for cell in self.notebook.cells
            if cell.cell_type == "code"
            and "experiment-routing" in cell.metadata.get("tags", [])
        ]
        self.assertEqual(len(routing), 1)
        self.assertIn("EXPERIMENT_LABEL = 'minilm_5ep_category_heads_18'", routing[0].source)
        self.assertIn("EXPERIMENT_SHEET = 'sft_exps'", routing[0].source)

    def test_patched_trainer_selects_only_the_pair_category_head(self) -> None:
        patches = [
            cell
            for cell in self.notebook.cells
            if cell.cell_type == "code"
            and "category-heads" in cell.metadata.get("tags", [])
        ]
        self.assertEqual(len(patches), 1)
        source = patches[0].source
        self.assertIn("num_labels=len(CATEGORY_HEAD_NAMES)", source)
        self.assertIn("select_category_logits(", source)
        self.assertIn("allow_unseen=False", source)
        self.assertIn("logits.mean(dim=1)", source)
        self.assertIn("ignore_mismatched_sizes=True", source)

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is unavailable")
    def test_routing_gradient_touches_only_selected_head(self) -> None:
        import torch

        patch_cell = next(
            cell
            for cell in self.notebook.cells
            if cell.cell_type == "code"
            and "category-heads" in cell.metadata.get("tags", [])
        )
        patch_module = ast.parse(patch_cell.source)
        patched_source = next(
            ast.literal_eval(node.value)
            for node in patch_module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "PATCHED_TRAINING_SOURCE"
                for target in node.targets
            )
        )
        trainer_module = ast.parse(patched_source)
        selection_function = next(
            node
            for node in trainer_module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "select_category_logits"
        )
        namespace = {
            "torch": torch,
            "CATEGORY_HEAD_NAMES": self.generator.CATEGORY_HEAD_NAMES,
            "CATEGORY_TO_HEAD": {
                category: index
                for index, category in enumerate(self.generator.CATEGORY_HEAD_NAMES)
            },
        }
        exec(
            compile(
                ast.Module(body=[selection_function], type_ignores=[]),
                "select_category_logits.py",
                "exec",
            ),
            namespace,
        )
        logits = torch.zeros((2, 18), requires_grad=True)
        selected = namespace["select_category_logits"](
            logits,
            [0, 1],
            [
                self.generator.CATEGORY_HEAD_NAMES[3],
                self.generator.CATEGORY_HEAD_NAMES[11],
            ],
            allow_unseen=False,
        )
        selected.sum().backward()
        expected = torch.zeros_like(logits)
        expected[0, 3] = 1.0
        expected[1, 11] = 1.0
        torch.testing.assert_close(logits.grad, expected)

    @unittest.skipIf(importlib.util.find_spec("pandas") is None, "pandas is unavailable")
    def test_mapping_matches_frozen_human_train_categories(self) -> None:
        import pandas as pd

        source_dir = (
            ROOT
            / ".kaggle"
            / "datasets"
            / "product-matching-validation-splits-v1"
        )
        if not source_dir.is_dir():
            self.skipTest("frozen human Dataset is unavailable")
        items = pd.read_parquet(
            source_dir / "human_items.parquet", columns=["id", "category"]
        )
        pairs = pd.read_parquet(
            source_dir / "human_train_pairs.parquet", columns=["id1"]
        )
        categories = items.set_index("id")["category"]
        observed = tuple(sorted(categories.loc[pairs["id1"]].astype(str).unique()))
        self.assertEqual(observed, self.generator.CATEGORY_HEAD_NAMES)

    def test_completion_records_head_mapping_and_patch_hash(self) -> None:
        source = "\n".join(
            cell.source for cell in self.notebook.cells if cell.cell_type == "code"
        )
        self.assertIn('"category_head_patch_sha256": CATEGORY_HEAD_PATCH_SHA256', source)
        self.assertIn('"category_heads": CATEGORY_HEAD_CONFIG', source)
        self.assertIn("compare_experiment_directories(", source)
        self.assertIn("sync_from_kaggle_credentials(", source)

    def test_generator_is_deterministic(self) -> None:
        generated = self.generator.build_category_heads_notebook()
        self.assertEqual(
            nbformat.writes(generated),
            nbformat.writes(self.notebook),
        )


if __name__ == "__main__":
    unittest.main()
