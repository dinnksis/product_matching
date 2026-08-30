from __future__ import annotations

import sys
from pathlib import Path

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import create_benefit_router_oof_notebooks as builder


def require_local_validation_data() -> None:
    required = (
        builder.architecture.VALIDATION_MANIFEST,
        builder.architecture.SERIALIZATION_FREQUENCY,
    )
    if any(not path.is_file() for path in required):
        pytest.skip("requires ignored prepared validation and serialization inputs")


def test_all_neural_oof_notebooks_are_valid_and_leakage_safe() -> None:
    require_local_validation_data()
    for profile in builder.PROFILES:
        notebook = builder.build(profile)
        nbformat.validate(notebook)
        metadata = notebook.metadata["product_matching_training"]
        assert metadata["folds"] == 5
        assert metadata["validation_labels_used"] is False
        source = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        assert "StratifiedGroupKFold" in source
        assert "stable_component_ids" in source
        assert "oof_predictions.parquet" in source
        assert "google_sheets_sync.json" in source
        training_cell = next(
            cell.source
            for cell in notebook.cells
            if cell.cell_type == "code" and "N_SPLITS = 5" in cell.source
        )
        assert 'pair_frames["train"]' in training_cell
        assert "human_iid_validation_pairs" not in training_cell
        assert "human_hard_validation_pairs" not in training_cell
        assert "human_ood_validation_pairs" not in training_cell
