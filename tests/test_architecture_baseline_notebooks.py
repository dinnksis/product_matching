from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

from scripts import create_architecture_baseline_notebooks as builder
from scripts import run_architecture_baseline_kaggle as runner


ROOT = Path(__file__).resolve().parents[1]


def test_all_profiles_share_frozen_protocol_and_effective_batch() -> None:
    configuration = builder.load_configuration()
    protocol = configuration["protocol"]
    assert protocol["serialization"] == "S2_VALUES_ONLY"
    assert protocol["epochs"] == 1
    assert protocol["learning_rate"] == 2e-5
    assert protocol["max_length"] == 384
    assert protocol["sampling"] == "none"
    for profile in configuration["profiles"].values():
        assert profile["batch_size"] * 2 * profile["gradient_accumulation"] == 192


def test_four_generated_notebooks_have_architecture_only_tracking() -> None:
    configuration = builder.load_configuration()
    dataset = builder.load_validation_dataset()
    protocol = configuration["protocol"]
    embedded_sources, _, _ = builder.embedded_sources()
    trainer_source = embedded_sources["scripts/train_cross_encoder.py"]
    assert 'f"{name}_validation_predictions.parquet"' in trainer_source
    assert "logit_ab" in trainer_source
    assert "logit_ba" in trainer_source
    for profile_name, profile in configuration["profiles"].items():
        notebook = builder.build_notebook(dataset, protocol, profile_name, profile)
        nbf.validate(notebook)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        for index, cell in enumerate(code_cells):
            compile(cell.source, f"{profile_name}:cell-{index}", "exec")
        source = "\n".join(cell.source for cell in notebook.cells)
        final_cell = code_cells[-1].source
        completion_cell = next(
            cell.source
            for cell in code_cells
            if 'completion_path = WORKING_ROOT / "notebook_completed.json"' in cell.source
        )
        assert "S2_VALUES_ONLY" in source
        assert '"--nproc_per_node=2"' in source
        assert "EMBEDDED_SOURCE_BUNDLE_B64" in source
        assert "EXPECTED_VALIDATION_FILES" in source
        assert "Attached validation Dataset manifest has changed" not in source
        assert "Attached validation manifest changes frozen file" in source
        assert 'for split in ("iid", "hard", "ood")' in completion_cell
        assert "logger_module.sync_architecture_from_kaggle_credentials(" in final_cell
        assert "sync_result = logger_module.sync_from_kaggle_credentials(" not in final_cell
        assert "baseline_comparison" not in completion_cell
        assert "experiment_group" not in completion_cell


def test_only_minilm_attaches_the_frozen_pretrain_checkpoint() -> None:
    configuration = builder.load_configuration()
    profiles = configuration["profiles"]
    assert "initial_checkpoint_dataset" not in profiles["gte"]
    assert "initial_checkpoint_dataset" not in profiles["rumodernbert"]
    assert "initial_checkpoint_dataset" not in profiles["bge-v2-m3"]
    assert profiles["minilm-5ep"]["initial_checkpoint_dataset"].endswith(
        "product-matching-minilm-llm-pretrain-5ep"
    )


def test_launcher_has_one_isolated_kernel_and_output_directory_per_profile() -> None:
    assert set(runner.PROFILES) == set(builder.NOTEBOOK_FILENAMES)
    assert builder.VALIDATION_DATASET_REF == (
        "alexproger23/product-matching-validation-splits-v1"
    )
    assert builder.RAW_DATASET_REF == "dinakepecheva/e-cup-human-data"
    slugs = [spec["slug"] for spec in runner.PROFILES.values()]
    assert len(slugs) == len(set(slugs)) == 4
    for profile, spec in runner.PROFILES.items():
        assert spec["notebook"] == builder.NOTEBOOK_FILENAMES[profile]


def test_checked_in_notebooks_match_generator() -> None:
    configuration = builder.load_configuration()
    dataset = builder.load_validation_dataset()
    for profile_name, profile in configuration["profiles"].items():
        generated = builder.build_notebook(
            dataset,
            configuration["protocol"],
            profile_name,
            profile,
        )
        checked_in = nbf.read(
            builder.OUTPUT_DIR / builder.NOTEBOOK_FILENAMES[profile_name],
            as_version=4,
        )
        assert json.loads(nbf.writes(generated)) == json.loads(nbf.writes(checked_in))
        assert (
            builder.OUTPUT_DIR / builder.NOTEBOOK_FILENAMES[profile_name]
        ).stat().st_size < 1_000_000
