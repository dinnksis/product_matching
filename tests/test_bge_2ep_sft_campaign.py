from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from functools import lru_cache
from unittest import mock
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import create_bge_2ep_sft_notebooks as builder
import create_cross_encoder_training_notebook as cross_builder
import run_bge_2ep_sft_kaggle as launcher
import train_bge_2ep_sft as trainer


def baseline_inputs():
    config = cross_builder.load_training_config(builder.DEFAULT_CONFIG)
    validation = {
        "dataset": "alexproger23/product-matching-validation-splits-v1",
        "manifest_sha256": builder.EXPECTED_VALIDATION_MANIFEST_SHA256,
    }
    checkpoint = {
        "dataset": "alexproger23/product-matching-bge-pretrain-2ep",
        "manifest_sha256": "a" * 64,
    }
    return config, validation, checkpoint


def build_entry_and_notebook(variant_index: int = 0):
    config, validation, checkpoint = baseline_inputs()
    return builder.build_variant_notebook(
        validation=validation,
        checkpoint=checkpoint,
        base_config=config,
        variant=builder.VARIANT_SPECS[variant_index],
    )


def _check_frozen_base_config_and_memory_geometry():
    config = cross_builder.load_training_config(builder.DEFAULT_CONFIG)
    builder.validate_base_config(config)
    trainer.validate_memory_geometry(config)
    assert config["batch_size"] * 2 * config["gradient_accumulation"] == 192
    assert config["eval_batch_size"] == 32
    assert config["max_length"] == 384
    assert config["gradient_checkpointing"] is True

    for key, value in (
        ("batch_size", 9),
        ("gradient_accumulation", 11),
        ("max_length", 512),
        ("gradient_checkpointing", False),
    ):
        changed = dict(config)
        changed[key] = value
        with unittest.TestCase().assertRaises(trainer.BgeTrainingContractError):
            trainer.validate_memory_geometry(changed)

    for key, value in (
        ("weight_decay", 0.02),
        ("warmup_ratio", 0.1),
        ("label_smoothing", 0.01),
        ("max_grad_norm", 1.0),
        ("lexical_hard_negative_strength", 0.1),
        ("dataloader_workers", 4),
    ):
        changed = dict(config)
        changed[key] = value
        with unittest.TestCase().assertRaises(builder.CampaignConfigError):
            builder.validate_base_config(changed)
    missing = dict(config)
    missing.pop("weight_decay")
    with unittest.TestCase().assertRaises(builder.CampaignConfigError):
        builder.validate_base_config(missing)
    unexpected = dict(config)
    unexpected["unreviewed"] = True
    with unittest.TestCase().assertRaises(builder.CampaignConfigError):
        builder.validate_base_config(unexpected)
    validation = {
        "dataset": "alexproger23/product-matching-validation-splits-v1",
        "manifest_sha256": builder.EXPECTED_VALIDATION_MANIFEST_SHA256,
    }
    checkpoint = {
        "dataset": "alexproger23/product-matching-bge-pretrain-2ep",
        "manifest_sha256": "a" * 64,
    }
    drifted = dict(config)
    drifted["warmup_ratio"] = 0.1
    with unittest.TestCase().assertRaises(builder.CampaignConfigError):
        builder.build_variant_notebook(
            validation=validation,
            checkpoint=checkpoint,
            base_config=drifted,
            variant=builder.VARIANT_SPECS[0],
        )
    assert builder.file_sha256(builder.DEFAULT_CONFIG) == builder.EXPECTED_BASE_CONFIG_FILE_SHA256


def _check_adamw_foreach_and_nonfinite_guards():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = trainer.memory_efficient_adamw([parameter], lr=1e-3)
    assert optimizer.defaults["foreach"] is False
    with unittest.TestCase().assertRaises(trainer.BgeTrainingContractError):
        trainer.memory_efficient_adamw([parameter], lr=1e-3, foreach=True)

    parameter.grad = torch.tensor([2.0])
    finite_norm = trainer.amp_compatible_clip_grad_norm([parameter], 1.0)
    assert torch.isfinite(finite_norm)

    parameter.grad = torch.tensor([float("inf")])
    overflow_norm = trainer.amp_compatible_clip_grad_norm([parameter], 1.0)
    assert not torch.isfinite(overflow_norm)
    with unittest.TestCase().assertRaises(trainer.BgeTrainingContractError):
        trainer.amp_compatible_clip_grad_norm([parameter], 1.0, foreach=True)
    with unittest.TestCase().assertRaises(trainer.BgeTrainingContractError):
        trainer.amp_compatible_clip_grad_norm(
            [parameter], 1.0, error_if_nonfinite=True
        )
    assert "torch.isfinite(logits)" in builder.FIXED_LOSS_HOOK_SOURCE
    assert "torch.isfinite(loss)" in builder.FIXED_LOSS_HOOK_SOURCE


def _check_amp_dynamic_backoff_contract():
    assert trainer.classify_amp_optimizer_attempt(
        gradients_finite=False,
        scale_before=65536.0,
        scale_after=32768.0,
        optimizer_state_parameters=0,
    ) == "skipped_gradient_overflow"
    assert trainer.classify_amp_optimizer_attempt(
        gradients_finite=True,
        scale_before=32768.0,
        scale_after=32768.0,
        optimizer_state_parameters=builder.EXPECTED_TRAINABLE_PARAMETER_TENSORS,
    ) == "optimizer_step"
    invalid_attempts = (
        {
            "gradients_finite": False,
            "scale_before": 65536.0,
            "scale_after": 65536.0,
            "optimizer_state_parameters": 0,
        },
        {
            "gradients_finite": False,
            "scale_before": 65536.0,
            "scale_after": 32768.0,
            "optimizer_state_parameters": 1,
        },
        {
            "gradients_finite": True,
            "scale_before": 32768.0,
            "scale_after": 16384.0,
            "optimizer_state_parameters": builder.EXPECTED_TRAINABLE_PARAMETER_TENSORS,
        },
        {
            "gradients_finite": True,
            "scale_before": 32768.0,
            "scale_after": 32768.0,
            "optimizer_state_parameters": 0,
        },
    )
    for attempt in invalid_attempts:
        with unittest.TestCase().assertRaises((RuntimeError, FloatingPointError)):
            trainer.classify_amp_optimizer_attempt(**attempt)

    # Exercise the same public GradScaler transition on CPU: the raw loss is
    # finite, its initial scaled backward overflows, Adam state stays empty,
    # scale backs off, and the retry performs one real optimizer step.
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = trainer.memory_efficient_adamw([parameter], lr=1e-3)
    scaler = torch.amp.GradScaler(
        "cpu", init_scale=8.0, growth_interval=100, enabled=True
    )
    overflowing_loss = parameter.sum() * torch.tensor(1e38)
    assert torch.isfinite(overflowing_loss)
    scaler.scale(overflowing_loss).backward()
    scaler.unscale_(optimizer)
    overflow_norm = trainer.amp_compatible_clip_grad_norm([parameter], 1.0)
    overflow_scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    overflow_scale_after = float(scaler.get_scale())
    assert trainer.classify_amp_optimizer_attempt(
        gradients_finite=bool(torch.isfinite(overflow_norm)),
        scale_before=overflow_scale_before,
        scale_after=overflow_scale_after,
        optimizer_state_parameters=len(optimizer.state),
        expected_optimizer_state_parameters=1,
    ) == "skipped_gradient_overflow"
    assert len(optimizer.state) == 0

    optimizer.zero_grad(set_to_none=True)
    finite_loss = (parameter - 0.5).square().sum()
    scaler.scale(finite_loss).backward()
    scaler.unscale_(optimizer)
    finite_norm = trainer.amp_compatible_clip_grad_norm([parameter], 1.0)
    finite_scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    finite_scale_after = float(scaler.get_scale())
    assert trainer.classify_amp_optimizer_attempt(
        gradients_finite=bool(torch.isfinite(finite_norm)),
        scale_before=finite_scale_before,
        scale_after=finite_scale_after,
        optimizer_state_parameters=len(optimizer.state),
        expected_optimizer_state_parameters=1,
    ) == "optimizer_step"
    assert len(optimizer.state) == 1

    preflight_source = inspect.getsource(trainer.run_memory_preflight)
    assert "range(EXPECTED_GRADIENT_ACCUMULATION)" in preflight_source
    assert "raw_loss / EXPECTED_GRADIENT_ACCUMULATION" in preflight_source
    assert "training_model.no_sync()" in preflight_source
    assert "microstep + 1 < EXPECTED_GRADIENT_ACCUMULATION" in preflight_source
    assert "range(1, MAX_PREFLIGHT_AMP_ATTEMPTS + 1)" in preflight_source
    assert "exhausted bounded GradScaler backoff" in preflight_source
    assert preflight_source.index("exhausted bounded GradScaler backoff") < (
        preflight_source.index("args.preflight_report.write_text")
    )


def _check_checkpoint_tensor_ledger_matches_preflight_contract():
    checkpoint = ROOT / "model" / "pretrain_bge_2ep" / "model.safetensors"
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        shapes = [source.get_slice(name).get_shape() for name in source.keys()]
    assert len(shapes) == builder.EXPECTED_TRAINABLE_PARAMETER_TENSORS == 393
    assert sum(int(np.prod(shape, dtype=np.int64)) for shape in shapes) == (
        builder.EXPECTED_PARAMETERS
    )


def _check_full_training_guards_restore_on_success_and_failure():
    config = cross_builder.load_training_config(builder.DEFAULT_CONFIG)
    original_adamw = trainer.shared_trainer.AdamW
    original_clip = trainer.torch.nn.utils.clip_grad_norm_
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        argv = ["train_bge_2ep_sft.py", "--config", str(config_path)]
        for side_effect in (None, RuntimeError("trainer failed")):
            with mock.patch.object(sys, "argv", argv), mock.patch.dict(
                os.environ, {"WORLD_SIZE": "2"}
            ), mock.patch.object(
                trainer.shared_trainer, "main", side_effect=side_effect
            ):
                if side_effect is None:
                    trainer.run_full_training()
                else:
                    with unittest.TestCase().assertRaisesRegex(
                        RuntimeError, "trainer failed"
                    ):
                        trainer.run_full_training()
            assert trainer.shared_trainer.AdamW is original_adamw
            assert trainer.torch.nn.utils.clip_grad_norm_ is original_clip


def _check_notebook_has_exact_oodtrain_protocol_and_valid_python():
    notebook, entry = build_entry_and_notebook()
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"cell-{index}", "exec")

    tags = [cell.metadata.get("tags", []) for cell in notebook.cells]
    preflight_index = next(i for i, value in enumerate(tags) if "memory-preflight" in value)
    training_index = next(i for i, value in enumerate(tags) if "training" in value)
    completion_index = next(i for i, value in enumerate(tags) if "completion" in value)
    sheet_indices = [i for i, value in enumerate(tags) if "sheets-sync" in value]
    assert preflight_index < training_index < completion_index < min(sheet_indices)

    source = "\n".join(cell.source for cell in notebook.cells)
    training_source = notebook.cells[training_index].source
    assert "347840" in source
    assert "306669" in source
    assert "41171" in source
    assert '"human_former_ood"' in source
    assert '"--validation-split", "iid=iid_validation_pairs.parquet"' in training_source
    assert '"--validation-split", "hard=hard_validation_pairs.parquet"' in training_source
    assert '"--validation-split", "ood=' not in training_source
    assert "disabled_train_contaminated" in source
    assert "macro_average_precision" in source and "-1.0" in source
    assert "baseline_comparison" not in notebook.cells[completion_index].source
    assert "sft_exps" in source
    assert len(entry["kernel_slug"]) <= 50
    assert entry["title"] == entry["kernel_slug"]
    assert entry["kernel_slug"] not in launcher.TERMINAL_FAILED_KERNEL_SLUGS
    assert {
        "pm-b2-base-9c1f4648466b-s42-v1",
        "pm-b2-base-6ad383889383-s42-v1",
        "pm-b2-base-97335fa432bd-s42-v1",
    } <= launcher.TERMINAL_FAILED_KERNEL_SLUGS
    builder.validate_notebook_identity(notebook, entry=entry)
    sources, source_ledger, source_sha256 = builder.source_bundle()
    assert source_sha256 == entry["source_sha256"]
    ledger_paths = {record["path"] for record in source_ledger}
    assert {
        "scripts/create_qwen_training_notebook.py",
        "scripts/create_cross_encoder_training_notebook.py",
        "scripts/create_minilm_validation_baseline_notebook.py",
        "src/google_sheets_logger.py",
        "scripts/push_kaggle_training_dataset.py",
        "scripts/run_kaggle_notebook.py",
        "scripts/train_bge_2ep_sft.py",
        "scripts/create_bge_2ep_sft_notebooks.py",
    } <= ledger_paths
    assert set(sources) == {
        path.as_posix() for path in builder.RUNTIME_EMBEDDED_FILES
    }
    assert "scripts/create_bge_2ep_sft_notebooks.py" not in sources


def _check_source_guard_executes_and_materializes_exact_runtime_files(tmp_path: Path):
    notebook, entry = build_entry_and_notebook()
    source_cell = next(
        cell
        for cell in notebook.cells
        if "embedded-sources" in cell.metadata.get("tags", [])
    )
    fake_subprocess = mock.Mock()
    fake_subprocess.run.return_value = subprocess.CompletedProcess(
        args=["pip"], returncode=0, stdout="", stderr=""
    )
    namespace = {
        "EXPECTED_SOURCE_SHA256": entry["source_sha256"],
        "PROJECT_ROOT": tmp_path / "materialized-project",
        "RUNTIME_VERSIONS_PATH": tmp_path / "bge_runtime_versions.json",
        "file_sha256": builder.file_sha256,
        "hashlib": hashlib,
        "json": json,
        "subprocess": fake_subprocess,
        "sys": sys,
    }
    exec(compile(source_cell.source, "bge-source-guard", "exec"), namespace)
    assert fake_subprocess.run.call_count == 1
    runtime_versions = json.loads(
        namespace["RUNTIME_VERSIONS_PATH"].read_text(encoding="utf-8")
    )
    assert runtime_versions == namespace["RUNTIME_VERSIONS"]
    assert runtime_versions["packages"]["numpy"] == namespace["np"].__version__
    assert runtime_versions["packages"]["pandas"] == namespace["pd"].__version__

    # Exercise the same post-bootstrap dataframe/Parquet surface used by the
    # following data cell, after the fake pip process has fully returned.
    probe_path = tmp_path / "bootstrap_data_probe.parquet"
    namespace["pd"].DataFrame({"id": namespace["np"].array([1, 2])}).to_parquet(
        probe_path, index=False
    )
    assert namespace["pd"].read_parquet(probe_path)["id"].tolist() == [1, 2]
    ledger = namespace["SOURCE_LEDGER"]
    runtime_sources = namespace["EMBEDDED_SOURCES"]
    canonical = builder.canonical_sha256({"schema_version": 1, "files": ledger})
    assert canonical == entry["source_sha256"]
    ledger_by_path = {record["path"]: record for record in ledger}
    assert len(ledger_by_path) == len(ledger)
    for relative, content in runtime_sources.items():
        declaration = ledger_by_path[relative]
        payload = content.encode("utf-8")
        destination = namespace["PROJECT_ROOT"] / relative
        assert destination.read_bytes() == payload
        assert len(payload) == declaration["bytes"]
        assert hashlib.sha256(payload).hexdigest() == declaration["sha256"]
    for relative, declaration in ledger_by_path.items():
        local_payload = (ROOT / relative).read_bytes()
        assert len(local_payload) == declaration["bytes"]
        assert hashlib.sha256(local_payload).hexdigest() == declaration["sha256"]


def _check_notebook_bootstrap_import_order():
    notebook, _ = build_entry_and_notebook()
    source_index = next(
        index
        for index, cell in enumerate(notebook.cells)
        if "embedded-sources" in cell.metadata.get("tags", [])
    )
    data_index = next(
        index
        for index, cell in enumerate(notebook.cells)
        if "frozen-data" in cell.metadata.get("tags", [])
    )
    forbidden = {"numpy", "pandas", "torch", "sklearn", "pyarrow"}

    def imported_roots(tree):
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".", 1)[0])
        return roots

    for index, cell in enumerate(notebook.cells[:source_index]):
        if cell.cell_type != "code":
            continue
        assert not (imported_roots(ast.parse(cell.source)) & forbidden), index

    tree = ast.parse(notebook.cells[source_index].source)
    pip_index = next(
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "pip_result"
            for target in node.targets
        )
    )
    pip_assignment = tree.body[pip_index]
    assert isinstance(pip_assignment.value, ast.Call)
    assert isinstance(pip_assignment.value.func, ast.Attribute)
    assert pip_assignment.value.func.attr == "run"
    assert any(
        keyword.arg == "check"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in pip_assignment.value.keywords
    )
    return_guard_index = next(
        index
        for index, node in enumerate(tree.body)
        if index > pip_index
        and isinstance(node, ast.If)
        and "pip_result.returncode" in ast.unparse(node.test)
    )
    scientific_import_indices = [
        index
        for index, node in enumerate(tree.body)
        if (
            isinstance(node, ast.Import)
            and any(alias.name.split(".", 1)[0] in forbidden for alias in node.names)
        )
        or (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".", 1)[0] in forbidden
        )
    ]
    assert scientific_import_indices
    assert min(scientific_import_indices) > return_guard_index > pip_index
    assert data_index > source_index
    data_source = notebook.cells[data_index].source
    assert "pd.read_parquet" in data_source
    assert "np." in data_source


def _check_notebook_identity_rejects_executable_and_metadata_tampering():
    notebook, entry = build_entry_and_notebook()
    repeated_notebook, repeated_entry = build_entry_and_notebook()
    assert nbformat.writes(notebook) == nbformat.writes(repeated_notebook)
    assert entry == repeated_entry
    tampered_cell = copy.deepcopy(notebook)
    training_index = next(
        index
        for index, cell in enumerate(tampered_cell.cells)
        if "training" in cell.metadata.get("tags", [])
    )
    tampered_cell.cells[training_index].source += "\nprint('unfrozen mutation')"
    with unittest.TestCase().assertRaisesRegex(
        builder.CampaignConfigError, "executable notebook cell payload"
    ):
        builder.validate_notebook_identity(tampered_cell, entry=entry)

    tampered_metadata = copy.deepcopy(notebook)
    tampered_metadata.metadata.product_matching_training.source_sha256 = "0" * 64
    with unittest.TestCase().assertRaisesRegex(
        builder.CampaignConfigError, "metadata differs"
    ):
        builder.validate_notebook_identity(tampered_metadata, entry=entry)

    tampered_entry = copy.deepcopy(entry)
    tampered_entry["identity_sha256"] = "1" * 64
    with unittest.TestCase().assertRaises(builder.CampaignConfigError):
        builder.validate_notebook_identity(notebook, entry=tampered_entry)


def _check_log_lr_variants_are_unique_and_change_only_lr():
    notebooks_and_entries = [build_entry_and_notebook(index) for index in range(3)]
    entries = [item[1] for item in notebooks_and_entries]
    assert {entry["expected_config"]["learning_rate"] for entry in entries} == {
        1e-5,
        2e-5,
        4e-5,
    }
    assert len({entry["kernel_slug"] for entry in entries}) == 3
    assert all(len(entry["kernel_slug"]) <= 50 for entry in entries)
    baseline = entries[0]["expected_config"]
    for entry in entries[1:]:
        changed = {
            key
            for key, value in entry["expected_config"].items()
            if value != baseline[key]
        }
        assert changed == {"learning_rate"}


def _check_runner_command_attaches_only_frozen_sources_and_keeps_credentials():
    notebook, entry = build_entry_and_notebook()
    with tempfile.TemporaryDirectory() as temp_dir:
        notebook_path = Path(temp_dir) / "baseline.ipynb"
        nbformat.write(notebook, notebook_path)
        entry["notebook"] = str(notebook_path)
        command = launcher.runner_command(
            entry,
            env_file=Path("/tmp/.env"),
            dry_run=True,
            no_wait=False,
        )
    datasets = [command[index + 1] for index, value in enumerate(command) if value == "--dataset"]
    assert datasets == [entry["validation_dataset"], entry["checkpoint_dataset"]]
    assert "--no-env-sources" in command
    assert "--no-gpu-check" in command
    assert "--no-google-sheets-credentials" not in command
    assert "--dry-run" in command
    assert "--no-download" in command


@lru_cache(maxsize=1)
def _frozen_truth_fixture() -> dict[str, pd.DataFrame]:
    """Read exact source rows once; manifest hashing is covered by build tests."""
    human_root = builder.DEFAULT_SOURCE_DIR / "human"
    items = pd.read_parquet(human_root / "items.parquet", columns=["id", "category"])
    category_by_id = items.set_index("id")["category"]
    result: dict[str, pd.DataFrame] = {}
    for split, filename in {
        "iid": "iid_validation_pairs.parquet",
        "hard": "hard_validation_pairs.parquet",
    }.items():
        frame = pd.read_parquet(
            human_root / filename, columns=["id1", "id2", "target"]
        ).reset_index(drop=True)
        frame["category_1"] = frame["id1"].map(category_by_id)
        frame["category_2"] = frame["id2"].map(category_by_id)
        assert not frame[["category_1", "category_2"]].isnull().any().any()
        assert frame["category_1"].equals(frame["category_2"])
        result[split] = frame
    return result


def _prediction_frame(split: str) -> tuple[pd.DataFrame, dict]:
    truth = _frozen_truth_fixture()[split]
    index = np.arange(len(truth), dtype=np.int64)
    target = truth["target"].to_numpy(dtype=float)
    score = np.where(target == 1.0, 0.8, 0.2) + (index % 7) * 0.001
    frame = truth.copy()
    frame.insert(0, "pair_index", index)
    frame["score"] = score
    per_category = {
        str(name): float(average_precision_score(group["target"], group["score"]))
        for name, group in frame.groupby("category_1", sort=True)
    }
    metrics = {
        "examples": len(frame),
        "macro_average_precision": float(np.mean(list(per_category.values()))),
        "overall_average_precision": float(average_precision_score(target, score)),
        "recall_at_precision_0_99": 1.0,
        "threshold_at_precision_0_99": 0.5,
        "roc_auc": 1.0,
        "log_loss": 0.2,
        "per_category_average_precision": per_category,
    }
    return frame, metrics


def make_valid_output(tmp_path: Path):
    _, entry = build_entry_and_notebook()
    config = dict(entry["expected_config"])
    config["model"] = entry["expected_runtime_model_path"]
    run_id = "run-bge-baseline-001"
    iid, iid_metrics = _prediction_frame("iid")
    hard, hard_metrics = _prediction_frame("hard")
    output = tmp_path / entry["experiment"]
    output.mkdir(parents=True)
    iid.to_parquet(output / "iid_validation_predictions.parquet", index=False)
    hard.to_parquet(output / "hard_validation_predictions.parquet", index=False)
    (output / "training_config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "cross_encoder_config.json").write_text(json.dumps(config), encoding="utf-8")

    data_report = {
        "schema_version": 1,
        "policy": "human_train_plus_former_ood_exact_concat_v1",
        "items": builder.EXPECTED_ITEMS,
        "train_pairs": builder.EXPECTED_TRAIN,
        "train_positives": builder.EXPECTED_TRAIN_POSITIVES,
        "train_positive_rate": builder.EXPECTED_TRAIN_POSITIVE_RATE,
        "source_counts": {
            "human_train": builder.EXPECTED_HUMAN_TRAIN,
            "human_former_ood": builder.EXPECTED_FORMER_OOD,
        },
        "former_ood_categories": sorted(builder.EXPECTED_OOD_CATEGORIES),
        "validation_rows": {"iid": builder.EXPECTED_IID, "hard": builder.EXPECTED_HARD},
        "validation_item_overlap": {"iid": 0, "hard": 0},
        "ood_evaluation": "disabled_train_contaminated",
    }
    (tmp_path / "bge_train_data_report.json").write_text(
        json.dumps(data_report), encoding="utf-8"
    )
    preflight = {
        "schema_version": 1,
        "status": "passed",
        "model": entry["expected_runtime_model_path"],
        "world_size": 2,
        "parameters": builder.EXPECTED_PARAMETERS,
        "microbatch_per_gpu": 8,
        "max_length": 384,
        "gradient_accumulation": 12,
        "accumulated_microbatches": 12,
        "loss_divisor_per_microbatch": 12,
        "ddp_no_sync_microbatches": 11,
        "ddp_sync_microbatches": 1,
        "effective_batch": 192,
        "eval_batch_per_gpu": 32,
        "eval_probe_after_optimizer_state": True,
        "gradient_checkpointing": True,
        "attention_implementation": "sdpa",
        "amp_dtype": "float16",
        "adamw_foreach": False,
        "gradient_clip_foreach": False,
        "nonfinite_gradient_policy": builder.EXPECTED_AMP_NONFINITE_POLICY,
        "amp_max_attempts": builder.EXPECTED_PREFLIGHT_AMP_ATTEMPTS,
        "optimizer_state": "adamw_exp_avg_and_exp_avg_sq_materialized",
        "optimizer_state_parameters_per_rank": builder.EXPECTED_TRAINABLE_PARAMETER_TENSORS,
        "optimizer_state_tensor_elements_per_rank": 2 * builder.EXPECTED_PARAMETERS,
        "ranks": [
            {
                "rank": rank,
                "gpu": "Tesla T4",
                "loss": 0.69,
                "gradient_norm": 1.2,
                "amp_attempts": [
                    {
                        "attempt": 1,
                        "accumulated_microbatches": 12,
                        "loss_divisor_per_microbatch": 12,
                        "accumulated_loss": 0.7,
                        "gradient_norm": None,
                        "gradients_finite": False,
                        "scale_before": 65536.0,
                        "scale_after": 32768.0,
                        "optimizer_state_parameters": 0,
                        "outcome": "skipped_gradient_overflow",
                    },
                    {
                        "attempt": 2,
                        "accumulated_microbatches": 12,
                        "loss_divisor_per_microbatch": 12,
                        "accumulated_loss": 0.69,
                        "gradient_norm": 1.2,
                        "gradients_finite": True,
                        "scale_before": 32768.0,
                        "scale_after": 32768.0,
                        "optimizer_state_parameters": (
                            builder.EXPECTED_TRAINABLE_PARAMETER_TENSORS
                        ),
                        "outcome": "optimizer_step",
                    },
                ],
                "amp_overflow_skips": 1,
                "amp_final_scale": 32768.0,
                "peak_allocated_gib": 12.0,
                "peak_reserved_gib": 13.0,
            }
            for rank in (0, 1)
        ],
    }
    (tmp_path / "bge_memory_preflight.json").write_text(
        json.dumps(preflight), encoding="utf-8"
    )
    runtime_versions = {
        "schema_version": 1,
        "python": "3.11.13",
        "packages": {
            "numpy": "2.2.6",
            "pandas": "2.2.3",
            "pyarrow": "20.0.0",
            "scikit-learn": "1.6.1",
            "torch": "2.7.0",
            "transformers": "4.53.0",
        },
    }
    (tmp_path / "bge_runtime_versions.json").write_text(
        json.dumps(runtime_versions), encoding="utf-8"
    )
    report = {
        "training_seconds": 100.0,
        "validation_seconds": 20.0,
        "total_pipeline_seconds": 150.0,
        "examples_per_second": 100.0,
        "original_training_examples": builder.EXPECTED_TRAIN,
        "training_subset": "all",
        "training_sampling": "none",
        "training_loss_weighting": "none",
        "training_unique_coverage_per_epoch": 1.0,
        "training_loss_weight_min": 1.0,
        "training_loss_weight_median": 1.0,
        "training_loss_weight_max": 1.0,
        "training_source_counts": data_report["source_counts"],
        "training_source_weight_mass": {
            key: float(value) for key, value in data_report["source_counts"].items()
        },
        "loss_hook": {"name": "bge_sft_loss_hook", "sha256": entry["loss_hook_sha256"]},
        "primary_validation_split": "iid",
        "experiment_group": "sft",
        "evaluated_validation_splits": ["iid", "hard"],
        "ood_evaluation_policy": "disabled_train_contaminated",
        "validation_splits": {
            "iid": iid_metrics,
            "hard": hard_metrics,
            "ood": copy.deepcopy(builder.OOD_SENTINEL),
        },
        "args": launcher._expected_report_args(entry),
    }
    (output / "training_report.json").write_text(json.dumps(report), encoding="utf-8")
    completion = {
        "status": "complete",
        "run_id": run_id,
        "experiment": entry["experiment"],
        "experiment_group": "sft",
        "campaign": builder.CAMPAIGN,
        "role": entry["role"],
        "notes": entry["expected_notes"],
        "model": entry["checkpoint_dataset"],
        "dataset_ref": entry["validation_dataset"],
        "validation_manifest_sha256": entry["validation_manifest_sha256"],
        "initial_checkpoint_ref": entry["checkpoint_dataset"],
        "initial_checkpoint_manifest_sha256": entry["checkpoint_manifest_sha256"],
        "initial_checkpoint_model_sha256": entry["checkpoint_model_sha256"],
        "code_bundle_sha256": entry["source_sha256"],
        "frozen_recipe_sha256": entry["recipe_sha256"],
        "campaign_identity_sha256": entry["identity_sha256"],
        "executable_cells_sha256": entry["executable_cells_sha256"],
        "loss_variant": "bce_finite_guard_v1",
        "loss_hook_sha256": entry["loss_hook_sha256"],
        "ood_evaluation_policy": "disabled_train_contaminated",
        "train_data": data_report,
        "memory_preflight": preflight,
        "runtime_versions": runtime_versions,
        "training_wall_seconds": 151.0,
        "training_report": report,
    }
    (tmp_path / "notebook_completed.json").write_text(
        json.dumps(completion), encoding="utf-8"
    )
    (tmp_path / "experiment_run_id.txt").write_text(run_id + "\n", encoding="utf-8")
    sync = {
        "status": "synced",
        "run_id": run_id,
        "experiment_group": "sft",
        "comparison_sheet": "sft_exps",
        "spreadsheet_id": launcher.shared.EXPERIMENT_SPREADSHEET_ID,
    }
    (tmp_path / "google_sheets_sync.json").write_text(json.dumps(sync), encoding="utf-8")
    return entry, completion, report


def _check_strict_output_validator_accepts_exact_baseline(tmp_path):
    entry, _, _ = make_valid_output(tmp_path)
    with mock.patch.object(
        launcher,
        "_load_frozen_validation_truth",
        return_value=_frozen_truth_fixture(),
    ):
        result = launcher.validate_run_output(tmp_path, entry=entry)
    assert result["ood_macro_ap"] == -1.0
    assert result["comparison_sheet"] == "sft_exps"


def _check_strict_output_validator_rejects_protocol_drift(
    tmp_path, mutation, expected_message
):
    entry, completion, report = make_valid_output(tmp_path)
    output = tmp_path / entry["experiment"]
    if mutation == "ood_sentinel":
        report["validation_splits"]["ood"]["macro_average_precision"] = 0.5
        completion["training_report"] = report
        (output / "training_report.json").write_text(json.dumps(report), encoding="utf-8")
        (tmp_path / "notebook_completed.json").write_text(
            json.dumps(completion), encoding="utf-8"
        )
    elif mutation == "ood_parquet":
        (output / "ood_validation_predictions.parquet").write_bytes(b"forbidden")
    elif mutation == "model_weight":
        (output / "model.safetensors").write_bytes(b"forbidden")
    elif mutation == "train_count":
        data = json.loads((tmp_path / "bge_train_data_report.json").read_text())
        data["train_pairs"] -= 1
        (tmp_path / "bge_train_data_report.json").write_text(json.dumps(data))
    elif mutation == "stale_sheet":
        sync = json.loads((tmp_path / "google_sheets_sync.json").read_text())
        sync["run_id"] = "stale"
        (tmp_path / "google_sheets_sync.json").write_text(json.dumps(sync))
    elif mutation == "preflight_geometry":
        preflight = json.loads((tmp_path / "bge_memory_preflight.json").read_text())
        preflight["microbatch_per_gpu"] = 9
        completion["memory_preflight"] = preflight
        (tmp_path / "bge_memory_preflight.json").write_text(json.dumps(preflight))
        (tmp_path / "notebook_completed.json").write_text(
            json.dumps(completion), encoding="utf-8"
        )
    elif mutation in {"iid_id", "iid_target", "iid_category"}:
        path = output / "iid_validation_predictions.parquet"
        predictions = pd.read_parquet(path)
        column = {
            "iid_id": "id1",
            "iid_target": "target",
            "iid_category": "category_1",
        }[mutation]
        if mutation == "iid_target":
            predictions.loc[0, column] = 1.0 - float(predictions.loc[0, column])
        elif mutation == "iid_category":
            predictions.loc[0, column] = "tampered-category"
        else:
            predictions.loc[0, column] = int(predictions.loc[0, column]) + 987654321
        predictions.to_parquet(path, index=False)
    elif mutation in {"root_model", "trainer_model"}:
        path = (
            tmp_path / "cross_encoder_config.json"
            if mutation == "root_model"
            else output / "training_config.json"
        )
        changed = json.loads(path.read_text())
        changed["model"] = "/kaggle/temp/attacker/checkpoint"
        path.write_text(json.dumps(changed), encoding="utf-8")
    elif mutation in {"report_model", "report_extra"}:
        if mutation == "report_model":
            report["args"]["model"] = "/kaggle/temp/attacker/checkpoint"
        else:
            report["args"]["unreviewed"] = True
        completion["training_report"] = report
        (output / "training_report.json").write_text(json.dumps(report), encoding="utf-8")
        (tmp_path / "notebook_completed.json").write_text(
            json.dumps(completion), encoding="utf-8"
        )
    elif mutation in {"preflight_model", "preflight_optimizer_count"}:
        preflight = json.loads((tmp_path / "bge_memory_preflight.json").read_text())
        if mutation == "preflight_model":
            preflight["model"] = "/kaggle/temp/attacker/checkpoint"
        else:
            preflight["optimizer_state_parameters_per_rank"] += 1
        completion["memory_preflight"] = preflight
        (tmp_path / "bge_memory_preflight.json").write_text(json.dumps(preflight))
        (tmp_path / "notebook_completed.json").write_text(
            json.dumps(completion), encoding="utf-8"
        )
    elif mutation in {"preflight_amp_retry", "preflight_amp_state"}:
        preflight = json.loads((tmp_path / "bge_memory_preflight.json").read_text())
        first_attempt = preflight["ranks"][0]["amp_attempts"][0]
        if mutation == "preflight_amp_retry":
            first_attempt["scale_after"] = first_attempt["scale_before"]
        else:
            first_attempt["optimizer_state_parameters"] = 1
        completion["memory_preflight"] = preflight
        (tmp_path / "bge_memory_preflight.json").write_text(json.dumps(preflight))
        (tmp_path / "notebook_completed.json").write_text(
            json.dumps(completion), encoding="utf-8"
        )
    elif mutation == "checkpoint_model_sha":
        completion["initial_checkpoint_model_sha256"] = "f" * 64
        (tmp_path / "notebook_completed.json").write_text(
            json.dumps(completion), encoding="utf-8"
        )
    elif mutation == "executable_cells_sha":
        completion["executable_cells_sha256"] = "e" * 64
        (tmp_path / "notebook_completed.json").write_text(
            json.dumps(completion), encoding="utf-8"
        )
    elif mutation == "runtime_version":
        versions_path = tmp_path / "bge_runtime_versions.json"
        versions = json.loads(versions_path.read_text())
        versions["packages"]["numpy"] = ""
        completion["runtime_versions"] = versions
        versions_path.write_text(json.dumps(versions), encoding="utf-8")
        (tmp_path / "notebook_completed.json").write_text(
            json.dumps(completion), encoding="utf-8"
        )
    with mock.patch.object(
        launcher,
        "_load_frozen_validation_truth",
        return_value=_frozen_truth_fixture(),
    ), unittest.TestCase().assertRaisesRegex(RuntimeError, expected_message):
        launcher.validate_run_output(tmp_path, entry=entry)


def _check_launcher_lock_and_fail_closed_absence(tmp_path: Path):
    lock_path = tmp_path / "campaign.lock"
    with launcher.exclusive_campaign_lock(lock_path):
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "exclusive lock"):
            with launcher.exclusive_campaign_lock(lock_path):
                pass

    status_failure = subprocess.CompletedProcess(
        args=["kaggle"], returncode=1, stdout="", stderr="transient"
    )
    empty_listing = subprocess.CompletedProcess(
        args=["kaggle"], returncode=0, stdout="[]", stderr=""
    )
    cli_empty_listing = subprocess.CompletedProcess(
        args=["kaggle"], returncode=0, stdout="Not found\n", stderr=""
    )
    assert launcher._listed_kernel_refs(cli_empty_listing.stdout) == set()
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "valid JSON"):
        launcher._listed_kernel_refs("unexpected proxy response")
    with mock.patch.object(
        launcher.kaggle,
        "run_command",
        side_effect=[status_failure, cli_empty_listing],
    ):
        assert (
            launcher.remote_kernel_status(
                ["kaggle"], "alexproger23/frozen-bge-kernel"
            )
            == "absence_unconfirmed"
        )

    with mock.patch.object(
        launcher.kaggle,
        "run_command",
        side_effect=[
            empty_listing,
            status_failure,
            empty_listing,
            status_failure,
        ],
    ) as repeated:
        launcher.confirm_remote_absence(
            ["kaggle"], "alexproger23/frozen-bge-kernel", pause_seconds=0
        )
        assert repeated.call_count == 4

    appeared_listing = subprocess.CompletedProcess(
        args=["kaggle"],
        returncode=0,
        stdout=json.dumps([{"ref": "alexproger23/frozen-bge-kernel"}]),
        stderr="",
    )
    with mock.patch.object(
        launcher.kaggle,
        "run_command",
        side_effect=[empty_listing, status_failure, appeared_listing],
    ), unittest.TestCase().assertRaisesRegex(RuntimeError, "appeared"):
        launcher.confirm_remote_absence(
            ["kaggle"], "alexproger23/frozen-bge-kernel", pause_seconds=0
        )

    _, entry = build_entry_and_notebook()
    kernel_ref = f"alexproger23/{entry['kernel_slug']}"
    manager = mock.Mock()
    with mock.patch.object(
        launcher, "validate_staged_kernel_metadata"
    ) as validate_stage, mock.patch.object(
        launcher, "confirm_remote_absence"
    ) as confirm, mock.patch.object(
        launcher.kaggle,
        "run_command",
        return_value=subprocess.CompletedProcess(
            args=["kaggle"], returncode=0, stdout="pushed", stderr=""
        ),
    ) as push, mock.patch.object(
        launcher, "verify_remote_dataset_sources_exact"
    ) as verify, mock.patch.object(
        launcher.kaggle, "STAGE_ROOT", tmp_path / "stage"
    ):
        manager.attach_mock(validate_stage, "validate_stage")
        manager.attach_mock(confirm, "confirm")
        manager.attach_mock(push, "push")
        manager.attach_mock(verify, "verify")
        launcher.push_new_kernel_after_confirmed_absence(
            ["kaggle"], kernel_ref=kernel_ref, entry=entry, run_timeout=43200
        )
    assert [call[0] for call in manager.mock_calls] == [
        "validate_stage",
        "confirm",
        "push",
        "verify",
    ]

    terminal_slug = next(iter(launcher.TERMINAL_FAILED_KERNEL_SLUGS))
    with mock.patch.dict(
        os.environ,
        {
            "KAGGLE_API_TOKEN": "test-token",
            "KAGGLE_IS_PRIVATE": "true",
            "KAGGLE_ACCELERATOR": "NvidiaTeslaT4",
        },
    ), mock.patch.object(
        launcher.kaggle, "kaggle_command", return_value=["kaggle"]
    ), mock.patch.object(
        launcher.kaggle,
        "run_command",
        return_value=subprocess.CompletedProcess(
            args=["kaggle"], returncode=0, stdout="[]", stderr=""
        ),
    ), unittest.TestCase().assertRaisesRegex(RuntimeError, "tombstoned"):
        launcher.run_live_campaign(
            args=mock.Mock(no_wait=False, full_download=False),
            env_file=tmp_path / ".env",
            owner="alexproger23",
            entries=[],
            selected=[{"kernel_slug": terminal_slug}],
        )


def _check_exact_dataset_sources_and_credential_override(tmp_path: Path):
    notebook, entry = build_entry_and_notebook()
    notebook_path = tmp_path / "baseline.ipynb"
    nbformat.write(notebook, notebook_path)
    entry["notebook"] = str(notebook_path)
    assert launcher.expected_dataset_sources(entry) == [
        "alexproger23/product-matching-validation-splits-v1",
        "alexproger23/product-matching-bge-pretrain-2ep",
        "alexproger23/ecom-matching-google-sheets-credentials",
    ]

    def assert_canonical_credentials(_command):
        assert os.environ["KAGGLE_GOOGLE_SHEETS_CREDENTIALS_DATASET"] == (
            launcher.CREDENTIALS_DATASET
        )

    with mock.patch.dict(
        os.environ,
        {"KAGGLE_GOOGLE_SHEETS_CREDENTIALS_DATASET": "attacker/override"},
    ), mock.patch.object(
        launcher.kaggle, "run_command", side_effect=assert_canonical_credentials
    ):
        launcher.run_inner_runner(["inner-runner"])
        assert os.environ["KAGGLE_GOOGLE_SHEETS_CREDENTIALS_DATASET"] == (
            "attacker/override"
        )

    stage_root = tmp_path / "stage"
    metadata_dir = stage_root / entry["kernel_slug"]
    metadata_dir.mkdir(parents=True)
    exact_metadata = {
        "id": f"alexproger23/{entry['kernel_slug']}",
        "title": entry["title"],
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "machine_shape": "NvidiaTeslaT4",
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
        "dataset_sources": launcher.expected_dataset_sources(entry),
    }
    metadata_path = metadata_dir / "kernel-metadata.json"
    nbformat.write(notebook, metadata_dir / "notebook.ipynb")
    metadata_path.write_text(json.dumps(exact_metadata), encoding="utf-8")
    with mock.patch.object(launcher.kaggle, "STAGE_ROOT", stage_root):
        launcher.validate_staged_kernel_metadata(entry)
        frozen_sources = launcher.expected_dataset_sources(entry)
        for altered_sources in (
            [*frozen_sources, "attacker/extra"],
            list(reversed(frozen_sources)),
        ):
            exact_metadata["dataset_sources"] = altered_sources
            metadata_path.write_text(json.dumps(exact_metadata), encoding="utf-8")
            with unittest.TestCase().assertRaisesRegex(
                RuntimeError, "Dataset attachments differ"
            ):
                launcher.validate_staged_kernel_metadata(entry)

    completed = subprocess.CompletedProcess(
        args=["kaggle"], returncode=0, stdout="pulled", stderr=""
    )

    def remote_pull_with(sources):
        def fake_pull(command, *, check=True):
            destination = Path(command[command.index("-p") + 1])
            (destination / "kernel-metadata.json").write_text(
                json.dumps({"dataset_sources": sources}), encoding="utf-8"
            )
            return completed

        return fake_pull

    # Remote metadata order is canonicalized by Kaggle and is not semantic.
    with mock.patch.object(
        launcher.kaggle,
        "run_command",
        side_effect=remote_pull_with(list(reversed(frozen_sources))),
    ):
        launcher.verify_remote_dataset_sources_exact(
            ["kaggle"], kernel_ref="alexproger23/frozen", entry=entry
        )
    for invalid_sources in (
        frozen_sources[:-1],
        [*frozen_sources, frozen_sources[0]],
        [*frozen_sources[:-1], "attacker/extra"],
    ):
        with mock.patch.object(
            launcher.kaggle,
            "run_command",
            side_effect=remote_pull_with(invalid_sources),
        ), unittest.TestCase().assertRaisesRegex(RuntimeError, "not exact"):
            launcher.verify_remote_dataset_sources_exact(
                ["kaggle"], kernel_ref="alexproger23/frozen", entry=entry
            )


def _check_resubmit_failed_flag_is_rejected():
    with mock.patch.object(
        sys, "argv", ["run_bge_2ep_sft_kaggle.py", "--resubmit-failed"]
    ), unittest.TestCase().assertRaises(SystemExit):
        launcher.parse_args()


def _check_candidate_selection_is_sequential_and_baseline_first():
    entries = []
    for index in range(3):
        _, entry = build_entry_and_notebook(index)
        entries.append(entry)
    selected = launcher._select_entries(entries, only=None, include_candidates=True)
    assert [entry["key"] for entry in selected] == ["baseline", "lr1e5", "lr4e5"]
    default = launcher._select_entries(entries, only=None, include_candidates=False)
    assert [entry["key"] for entry in default] == ["baseline"]


def _check_candidate_gate_requires_validated_local_baseline(tmp_path):
    entries = []
    for index in range(3):
        _, entry = build_entry_and_notebook(index)
        entries.append(entry)
    with mock.patch.object(launcher, "output_root", return_value=tmp_path):
        with unittest.TestCase().assertRaisesRegex(
            RuntimeError, "baseline directory is missing"
        ):
            launcher._validated_local_baseline(entries)


class Bge2epSftCampaignTest(unittest.TestCase):
    def test_frozen_base_config_and_memory_geometry(self):
        _check_frozen_base_config_and_memory_geometry()

    def test_adamw_foreach_and_nonfinite_guards(self):
        _check_adamw_foreach_and_nonfinite_guards()

    def test_amp_dynamic_backoff_contract(self):
        _check_amp_dynamic_backoff_contract()

    def test_checkpoint_tensor_ledger_matches_preflight_contract(self):
        _check_checkpoint_tensor_ledger_matches_preflight_contract()

    def test_full_training_guards_restore_on_success_and_failure(self):
        _check_full_training_guards_restore_on_success_and_failure()

    def test_notebook_has_exact_oodtrain_protocol_and_valid_python(self):
        _check_notebook_has_exact_oodtrain_protocol_and_valid_python()

    def test_source_guard_executes_and_materializes_exact_runtime_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _check_source_guard_executes_and_materializes_exact_runtime_files(
                Path(temp_dir)
            )

    def test_notebook_bootstrap_import_order(self):
        _check_notebook_bootstrap_import_order()

    def test_notebook_identity_rejects_executable_and_metadata_tampering(self):
        _check_notebook_identity_rejects_executable_and_metadata_tampering()

    def test_log_lr_variants_are_unique_and_change_only_lr(self):
        _check_log_lr_variants_are_unique_and_change_only_lr()

    def test_runner_command_attaches_only_frozen_sources_and_keeps_credentials(self):
        _check_runner_command_attaches_only_frozen_sources_and_keeps_credentials()

    def test_strict_output_validator_accepts_exact_baseline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _check_strict_output_validator_accepts_exact_baseline(Path(temp_dir))

    def test_strict_output_validator_rejects_protocol_drift(self):
        mutations = (
            ("ood_sentinel", "OOD metric -1"),
            ("ood_parquet", "OOD predictions"),
            ("model_weight", "model weights"),
            ("train_count", "train-data contract"),
            ("stale_sheet", "Sheets marker"),
            ("preflight_geometry", "preflight contract"),
            ("iid_id", "id1 rows differ from frozen validation"),
            ("iid_target", "target rows differ from frozen validation"),
            ("iid_category", "category_1 rows differ from frozen validation"),
            ("root_model", "root config differs"),
            ("trainer_model", "trainer config differs"),
            ("report_model", "trainer argument differs at model"),
            ("report_extra", "exact runtime recipe"),
            ("preflight_model", "preflight contract differs"),
            ("preflight_optimizer_count", "preflight contract differs"),
            ("preflight_amp_retry", "AMP overflow retry differs"),
            ("preflight_amp_state", "AMP overflow retry differs"),
            ("checkpoint_model_sha", "initial_checkpoint_model_sha256"),
            ("executable_cells_sha", "executable_cells_sha256"),
            ("runtime_version", "runtime version report"),
        )
        for mutation, expected_message in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp_dir:
                _check_strict_output_validator_rejects_protocol_drift(
                    Path(temp_dir), mutation, expected_message
                )

    def test_candidate_selection_is_sequential_and_baseline_first(self):
        _check_candidate_selection_is_sequential_and_baseline_first()

    def test_candidate_gate_requires_validated_local_baseline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _check_candidate_gate_requires_validated_local_baseline(Path(temp_dir))

    def test_launcher_lock_and_fail_closed_absence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _check_launcher_lock_and_fail_closed_absence(Path(temp_dir))

    def test_exact_dataset_sources_and_credential_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _check_exact_dataset_sources_and_credential_override(Path(temp_dir))

    def test_resubmit_failed_flag_is_rejected(self):
        _check_resubmit_failed_flag_is_rejected()


if __name__ == "__main__":
    unittest.main()
