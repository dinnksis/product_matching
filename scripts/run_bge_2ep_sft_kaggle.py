#!/usr/bin/env python3
"""Generate, safely launch, resume and validate the BGE-2ep SFT campaign.

The default action is intentionally one baseline kernel.  The two initial
log-LR candidates are available through ``--include-candidates`` or ``--only``
but a live candidate submission is refused until the exact local BGE baseline
artifact has passed all payload and Google Sheets checks.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import create_bge_2ep_sft_notebooks as builder
import create_qwen_training_notebook as shared
import run_kaggle_notebook as kaggle


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DATASET = "alexproger23/product-matching-validation-splits-v1"
CHECKPOINT_DATASET = "alexproger23/product-matching-bge-pretrain-2ep"
CREDENTIALS_DATASET = "alexproger23/ecom-matching-google-sheets-credentials"
REQUIRED_DATASETS = (VALIDATION_DATASET, CHECKPOINT_DATASET, CREDENTIALS_DATASET)
EXPECTED_INITIAL_CHECKPOINT_MODEL_SHA256 = (
    "c21ccfcd5de310ca0328620bf8ba09e838dbe3f6394be656bd7fec16ad8377d1"
)
BASELINE_KEY = "baseline"
TERMINAL_FAILED_KERNEL_SLUGS = frozenset(
    {
        "pm-b2-base-9c1f4648466b-s42-v1",
        "pm-b2-base-6ad383889383-s42-v1",
        "pm-b2-base-97335fa432bd-s42-v1",
    }
)
CAMPAIGN_LOCK_PATH = ROOT / ".kaggle" / "locks" / f"{builder.CAMPAIGN}.lock"
SLIM_OUTPUT_PATTERN = (
    r"(^|/)(notebook_completed\.json|google_sheets_sync\.json|"
    r"sheets_sync_pending\.json|experiment_run_id\.txt|"
    r"experiment_started_at_utc\.txt|cross_encoder_config\.json|"
    r"bge_memory_preflight\.json|bge_runtime_versions\.json|"
    r"bge_train_data_report\.json|"
    r"training_report\.json|training_config\.json|"
    r"(?:iid|hard)_validation_predictions\.parquet|.*\.log)$"
)


def runner_command(
    entry: Mapping[str, Any],
    *,
    env_file: Path,
    dry_run: bool,
    no_wait: bool,
) -> list[str]:
    builder.load_and_validate_notebook(Path(entry["notebook"]), entry=entry)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_kaggle_notebook.py"),
        str(entry["notebook"]),
        "--env-file",
        str(env_file),
        "--slug",
        str(entry["kernel_slug"]),
        "--title",
        str(entry["title"]),
        "--dataset",
        str(entry["validation_dataset"]),
        "--dataset",
        str(entry["checkpoint_dataset"]),
        "--no-env-sources",
        # This notebook already carries a hash-bound two-rank T4 optimizer-step
        # preflight.  Disabling the generic injected cell keeps the exact staged
        # executable payload identical to the locally frozen notebook.
        "--no-gpu-check",
        "--no-download",
    ]
    if dry_run:
        command.append("--dry-run")
    elif no_wait:
        command.append("--no-wait")
    return command


def expected_dataset_sources(entry: Mapping[str, Any]) -> list[str]:
    return [
        str(entry["validation_dataset"]),
        str(entry["checkpoint_dataset"]),
        CREDENTIALS_DATASET,
    ]


def run_inner_runner(command: list[str]) -> None:
    """Pin the credential attachment even when .env contains an override."""
    variable = "KAGGLE_GOOGLE_SHEETS_CREDENTIALS_DATASET"
    previous = os.environ.get(variable)
    os.environ[variable] = CREDENTIALS_DATASET
    try:
        kaggle.run_command(command)
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous


def validate_staged_kernel_metadata(entry: Mapping[str, Any]) -> dict[str, Any]:
    path = kaggle.STAGE_ROOT / str(entry["kernel_slug"]) / "kernel-metadata.json"
    if not path.is_file():
        raise RuntimeError("Inner Kaggle runner produced no staged kernel metadata")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    expected_sources = expected_dataset_sources(entry)
    actual_sources = metadata.get("dataset_sources")
    if actual_sources != expected_sources:
        raise RuntimeError(
            "Staged BGE Dataset attachments differ: "
            + json.dumps(
                {"actual": actual_sources, "expected": expected_sources},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    exact = {
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
    }
    for key, expected in exact.items():
        if metadata.get(key) != expected:
            raise RuntimeError(f"Staged BGE kernel metadata differs at {key}")
    builder.load_and_validate_notebook(
        path.parent / "notebook.ipynb", entry=entry
    )
    return metadata


@contextmanager
def exclusive_campaign_lock(path: Path = CAMPAIGN_LOCK_PATH):
    """Prevent two local controllers from mutating the same remote campaign."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another BGE campaign controller holds the exclusive lock: {path}"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps({"pid": os.getpid(), "acquired_at_unix": time.time()}) + "\n"
        )
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def output_root() -> Path:
    configured = Path(os.getenv("KAGGLE_OUTPUT_DIR", "artifacts/kaggle")).expanduser()
    return configured.resolve() if configured.is_absolute() else (ROOT / configured).resolve()


def _finite_probability(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} is not numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise RuntimeError(f"{label} is not a finite probability")
    return numeric


def _load_exactly_one(directory: Path, filename: str) -> tuple[Path, dict[str, Any]]:
    candidates = list(directory.rglob(filename))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Slim BGE output must contain exactly one {filename}, got {candidates}"
        )
    payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{filename} must contain a JSON object")
    return candidates[0], payload


def _validate_config(
    path: Path,
    expected_config: Mapping[str, Any],
    *,
    runtime_model_path: str,
    label: str,
) -> None:
    actual = json.loads(path.read_text(encoding="utf-8"))
    expected_runtime_config = dict(expected_config)
    expected_runtime_config["model"] = runtime_model_path
    if not isinstance(actual, dict) or actual != expected_runtime_config:
        raise RuntimeError(f"{label} differs from the exact runtime BGE recipe")


def _expected_report_args(entry: Mapping[str, Any]) -> dict[str, Any]:
    experiment = str(entry["experiment"])
    temp_root = f"/kaggle/temp/{experiment}"
    config = dict(entry["expected_config"])
    # model_load_kwargs is consumed from the exact config file by the shared
    # trainer; argparse records the repeatable CLI spelling model_load_kwarg.
    config.pop("model_load_kwargs")
    config["model"] = entry["expected_runtime_model_path"]
    return {
        "config": "/kaggle/working/cross_encoder_config.json",
        **config,
        "model_load_kwarg": [],
        "prepared_dir": f"{temp_root}/prepared",
        "output_dir": f"{temp_root}/trainer_output",
        "token_cache_dir": f"{temp_root}/token_cache",
        "loss_hook": "/kaggle/working/product_matching/bge_sft_loss_hook.py",
        "validation_split": [
            "iid=iid_validation_pairs.parquet",
            "hard=hard_validation_pairs.parquet",
        ],
    }


def _validate_preflight(
    payload: Mapping[str, Any], *, runtime_model_path: str
) -> None:
    expected = {
        "schema_version": 1,
        "status": "passed",
        "model": runtime_model_path,
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
        "optimizer_state_parameters_per_rank": (
            builder.EXPECTED_TRAINABLE_PARAMETER_TENSORS
        ),
        "optimizer_state_tensor_elements_per_rank": 2 * builder.EXPECTED_PARAMETERS,
    }
    mismatches = {
        key: {"actual": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "BGE memory preflight contract differs: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    ranks = payload.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != 2:
        raise RuntimeError("BGE memory preflight must contain exactly two ranks")
    if any(
        not isinstance(record, Mapping)
        or isinstance(record.get("rank"), bool)
        or not isinstance(record.get("rank"), int)
        for record in ranks
    ) or {record["rank"] for record in ranks} != {0, 1}:
        raise RuntimeError("BGE memory preflight rank identities differ")
    for record in ranks:
        if not isinstance(record, Mapping) or "T4" not in str(record.get("gpu", "")).upper():
            raise RuntimeError("BGE memory preflight did not run on two T4 GPUs")
        expected_rank_keys = {
            "rank",
            "gpu",
            "loss",
            "gradient_norm",
            "amp_attempts",
            "amp_overflow_skips",
            "amp_final_scale",
            "peak_allocated_gib",
            "peak_reserved_gib",
        }
        if set(record) != expected_rank_keys:
            raise RuntimeError("BGE memory preflight rank schema differs")
        loss = record.get("loss")
        gradient_norm = record.get("gradient_norm")
        allocated = record.get("peak_allocated_gib")
        reserved = record.get("peak_reserved_gib")
        numeric = (loss, gradient_norm, allocated, reserved)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric
        ):
            raise RuntimeError("BGE memory preflight has non-finite rank diagnostics")
        if (
            float(loss) <= 0
            or float(gradient_norm) < 0
            or float(allocated) <= 0
            or float(reserved) < float(allocated)
            or float(reserved) > 16.5
        ):
            raise RuntimeError("BGE memory preflight has impossible memory diagnostics")
        attempts = record.get("amp_attempts")
        if (
            not isinstance(attempts, list)
            or not 1 <= len(attempts) <= builder.EXPECTED_PREFLIGHT_AMP_ATTEMPTS
            or isinstance(record.get("amp_overflow_skips"), bool)
            or not isinstance(record.get("amp_overflow_skips"), int)
            or record.get("amp_overflow_skips") != len(attempts) - 1
        ):
            raise RuntimeError("BGE memory preflight AMP attempt count differs")
        previous_scale_after: float | None = None
        for index, attempt in enumerate(attempts, start=1):
            expected_attempt_keys = {
                "attempt",
                "accumulated_microbatches",
                "loss_divisor_per_microbatch",
                "accumulated_loss",
                "gradient_norm",
                "gradients_finite",
                "scale_before",
                "scale_after",
                "optimizer_state_parameters",
                "outcome",
            }
            if not isinstance(attempt, Mapping) or set(attempt) != expected_attempt_keys:
                raise RuntimeError("BGE memory preflight AMP attempt schema differs")
            exact_integers = {
                "attempt": index,
                "accumulated_microbatches": 12,
                "loss_divisor_per_microbatch": 12,
            }
            if any(
                isinstance(attempt.get(key), bool)
                or not isinstance(attempt.get(key), int)
                or attempt.get(key) != expected_value
                for key, expected_value in exact_integers.items()
            ):
                raise RuntimeError("BGE memory preflight AMP accumulation differs")
            attempt_loss = attempt.get("accumulated_loss")
            scale_before = attempt.get("scale_before")
            scale_after = attempt.get("scale_after")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
                for value in (attempt_loss, scale_before, scale_after)
            ):
                raise RuntimeError("BGE memory preflight AMP diagnostics differ")
            if previous_scale_after is not None and not math.isclose(
                float(scale_before), previous_scale_after, rel_tol=0.0, abs_tol=0.0
            ):
                raise RuntimeError("BGE memory preflight AMP scale history is discontinuous")
            state_parameters = attempt.get("optimizer_state_parameters")
            if isinstance(state_parameters, bool) or not isinstance(
                state_parameters, int
            ):
                raise RuntimeError("BGE memory preflight AMP state count differs")
            is_final = index == len(attempts)
            if is_final:
                attempt_grad_norm = attempt.get("gradient_norm")
                if (
                    attempt.get("outcome") != "optimizer_step"
                    or attempt.get("gradients_finite") is not True
                    or attempt.get("optimizer_state_parameters")
                        != builder.EXPECTED_TRAINABLE_PARAMETER_TENSORS
                    or isinstance(attempt_grad_norm, bool)
                    or not isinstance(attempt_grad_norm, (int, float))
                    or not math.isfinite(float(attempt_grad_norm))
                    or float(attempt_grad_norm) < 0
                    or float(scale_after) < float(scale_before)
                ):
                    raise RuntimeError("BGE memory preflight final AMP step differs")
            elif (
                attempt.get("outcome") != "skipped_gradient_overflow"
                or attempt.get("gradients_finite") is not False
                or attempt.get("gradient_norm") is not None
                or attempt.get("optimizer_state_parameters") != 0
                or not float(scale_after) < float(scale_before)
            ):
                raise RuntimeError("BGE memory preflight AMP overflow retry differs")
            previous_scale_after = float(scale_after)
        final_attempt = attempts[-1]
        final_scale = record.get("amp_final_scale")
        if (
            isinstance(final_scale, bool)
            or not isinstance(final_scale, (int, float))
            or not math.isfinite(float(final_scale))
            or float(final_scale) <= 0
            or not math.isclose(
                float(final_scale),
                float(final_attempt["scale_after"]),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or not math.isclose(
                float(loss),
                float(final_attempt["accumulated_loss"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(gradient_norm),
                float(final_attempt["gradient_norm"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError("BGE memory preflight final AMP summary differs")


def _validate_train_data(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": 1,
        "policy": "human_train_plus_former_ood_exact_concat_v1",
        "items": builder.EXPECTED_ITEMS,
        "train_pairs": builder.EXPECTED_TRAIN,
        "train_positives": builder.EXPECTED_TRAIN_POSITIVES,
        "source_counts": {
            "human_train": builder.EXPECTED_HUMAN_TRAIN,
            "human_former_ood": builder.EXPECTED_FORMER_OOD,
        },
        "former_ood_categories": sorted(builder.EXPECTED_OOD_CATEGORIES),
        "validation_rows": {"iid": builder.EXPECTED_IID, "hard": builder.EXPECTED_HARD},
        "validation_item_overlap": {"iid": 0, "hard": 0},
        "ood_evaluation": "disabled_train_contaminated",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"BGE train-data contract differs at {key}")
    rate = payload.get("train_positive_rate")
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isclose(
            float(rate), builder.EXPECTED_TRAIN_POSITIVE_RATE, abs_tol=1e-15
        )
    ):
        raise RuntimeError("BGE train positive rate differs")


def _validate_predictions(
    path: Path,
    *,
    split: str,
    expected_rows: int,
    report_metrics: Mapping[str, Any],
    frozen_truth: Any,
) -> None:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import average_precision_score

    frame = pd.read_parquet(path)
    required = {
        "pair_index",
        "id1",
        "id2",
        "target",
        "category_1",
        "category_2",
        "score",
    }
    if len(frame) != expected_rows or required - set(frame):
        raise RuntimeError(f"Invalid {split} prediction parquet schema/row count")
    if frame[list(required)].isnull().any().any():
        raise RuntimeError(f"{split} predictions contain nulls")
    if not frame["target"].isin([0.0, 1.0]).all():
        raise RuntimeError(f"{split} prediction targets are not binary")
    if not frame["pair_index"].equals(
        pd.Series(range(expected_rows), name="pair_index", dtype=frame["pair_index"].dtype)
    ):
        raise RuntimeError(f"{split} prediction pair_index order differs")
    for column in ("id1", "id2", "target", "category_1", "category_2"):
        actual_values = frame[column].to_numpy()
        expected_values = frozen_truth[column].to_numpy()
        if not np.array_equal(actual_values, expected_values):
            raise RuntimeError(
                f"{split} prediction {column} rows differ from frozen validation"
            )
    if not frame["score"].map(lambda value: math.isfinite(float(value))).all():
        raise RuntimeError(f"{split} predictions contain non-finite scores")
    if not frame["score"].between(0.0, 1.0).all():
        raise RuntimeError(f"{split} predictions are not probabilities")
    lower = frame[["id1", "id2"]].min(axis=1)
    upper = frame[["id1", "id2"]].max(axis=1)
    if pd.MultiIndex.from_arrays([lower, upper]).duplicated().any():
        raise RuntimeError(f"{split} predictions contain duplicate unordered pairs")
    per_category = frame.groupby("category_1", sort=True).apply(
        lambda group: average_precision_score(group["target"], group["score"]),
        include_groups=False,
    )
    macro_ap = float(per_category.mean())
    overall_ap = float(average_precision_score(frame["target"], frame["score"]))
    if not math.isclose(
        macro_ap,
        float(report_metrics["macro_average_precision"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(f"{split} macro AP does not match predictions")
    if not math.isclose(
        overall_ap,
        float(report_metrics["overall_average_precision"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(f"{split} overall AP does not match predictions")
    reported_per_category = report_metrics.get("per_category_average_precision")
    if not isinstance(reported_per_category, Mapping) or set(reported_per_category) != set(
        per_category.index.astype(str)
    ):
        raise RuntimeError(f"{split} per-category AP keys do not match predictions")
    for category, value in per_category.items():
        if not math.isclose(
            float(value),
            float(reported_per_category[str(category)]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"{split} category AP does not match predictions")


def _load_frozen_validation_truth(
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    import pandas as pd

    dataset_ref = str(entry["validation_dataset"])
    owner, separator, slug = dataset_ref.partition("/")
    if separator != "/" or slug != builder.VALIDATION_DATASET_SLUG:
        raise RuntimeError("BGE entry has an invalid frozen validation Dataset ref")
    dataset = builder.load_validation_dataset(builder.DEFAULT_SOURCE_DIR, owner)
    if dataset["manifest_sha256"] != entry["validation_manifest_sha256"]:
        raise RuntimeError("Local frozen validation manifest differs from BGE entry")
    human_root = builder.DEFAULT_SOURCE_DIR / "human"
    items = pd.read_parquet(human_root / "items.parquet", columns=["id", "category"])
    category_by_id = items.set_index("id")["category"]
    result: dict[str, Any] = {}
    filenames = {
        "iid": "iid_validation_pairs.parquet",
        "hard": "hard_validation_pairs.parquet",
    }
    for split, filename in filenames.items():
        pairs = pd.read_parquet(
            human_root / filename, columns=["id1", "id2", "target"]
        ).reset_index(drop=True)
        pairs["category_1"] = pairs["id1"].map(category_by_id)
        pairs["category_2"] = pairs["id2"].map(category_by_id)
        if pairs[["category_1", "category_2"]].isnull().any().any() or not pairs[
            "category_1"
        ].equals(pairs["category_2"]):
            raise RuntimeError(f"Frozen {split} validation category mapping is invalid")
        result[split] = pairs
    return result


def _validate_run_output(
    directory: Path,
    *,
    entry: Mapping[str, Any],
    require_sheets: bool,
) -> dict[str, Any]:
    required_root = {
        "notebook_completed.json",
        "experiment_run_id.txt",
        "cross_encoder_config.json",
        "bge_memory_preflight.json",
        "bge_runtime_versions.json",
        "bge_train_data_report.json",
    }
    if require_sheets:
        required_root.add("google_sheets_sync.json")
    missing = [name for name in required_root if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(f"BGE output is missing root artifacts: {sorted(missing)}")
    if require_sheets and (directory / "sheets_sync_pending.json").exists():
        raise RuntimeError("BGE Google Sheets synchronization is pending")
    if any(directory.rglob("model.safetensors*")):
        raise RuntimeError("Slim BGE output contains forbidden model weights/shards")
    if list(directory.rglob("ood_validation_predictions.parquet")):
        raise RuntimeError("BGE run emitted forbidden OOD predictions")

    completion = json.loads(
        (directory / "notebook_completed.json").read_text(encoding="utf-8")
    )
    run_id = str(completion.get("run_id", "")).strip()
    if completion.get("status") != "complete" or not run_id:
        raise RuntimeError("BGE completion is not successful")
    if (
        entry.get("checkpoint_model_sha256")
        != EXPECTED_INITIAL_CHECKPOINT_MODEL_SHA256
    ):
        raise RuntimeError("BGE entry changed the exact initial checkpoint model SHA-256")
    exact_completion = {
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
        "initial_checkpoint_model_sha256": EXPECTED_INITIAL_CHECKPOINT_MODEL_SHA256,
        "code_bundle_sha256": entry["source_sha256"],
        "frozen_recipe_sha256": entry["recipe_sha256"],
        "campaign_identity_sha256": entry["identity_sha256"],
        "executable_cells_sha256": entry["executable_cells_sha256"],
        "loss_variant": "bce_finite_guard_v1",
        "loss_hook_sha256": entry["loss_hook_sha256"],
        "ood_evaluation_policy": "disabled_train_contaminated",
    }
    for key, value in exact_completion.items():
        if completion.get(key) != value:
            raise RuntimeError(f"BGE completion identity differs at {key}")
    if completion.get("baseline_comparison") not in (None, {}):
        raise RuntimeError("BGE baseline phase must not fabricate paired significance")
    from src.google_sheets_logger import COMPARISON_HEADERS, build_comparison_row

    sheet_projection = dict(
        zip(COMPARISON_HEADERS, build_comparison_row(completion), strict=True)
    )
    if sheet_projection.get("ood_macro_ap") != -1.0:
        raise RuntimeError("BGE sft_exps projection did not preserve OOD metric -1")
    if sheet_projection.get("comparison_status") != "baseline_not_selected":
        raise RuntimeError("BGE baseline must remain a distinct unselected Sheet baseline")
    for field in (
        "baseline_run_id",
        "ood_delta",
        "ood_p_value",
        "ood_p_holm",
        "ood_ci95_low",
        "ood_ci95_high",
        "comparison_method",
    ):
        if sheet_projection.get(field) not in ("", None):
            raise RuntimeError(f"BGE Sheet projection fabricated {field}")
    if (
        directory.joinpath("experiment_run_id.txt").read_text(encoding="utf-8").strip()
        != run_id
    ):
        raise RuntimeError("BGE experiment_run_id.txt differs from completion")

    data_report = json.loads(
        (directory / "bge_train_data_report.json").read_text(encoding="utf-8")
    )
    _validate_train_data(data_report)
    if completion.get("train_data") != data_report:
        raise RuntimeError("BGE completion and train-data reports differ")

    preflight = json.loads(
        (directory / "bge_memory_preflight.json").read_text(encoding="utf-8")
    )
    _validate_preflight(
        preflight, runtime_model_path=entry["expected_runtime_model_path"]
    )
    if completion.get("memory_preflight") != preflight:
        raise RuntimeError("BGE completion and memory-preflight reports differ")

    runtime_versions = json.loads(
        (directory / "bge_runtime_versions.json").read_text(encoding="utf-8")
    )
    expected_runtime_packages = {
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "torch",
        "transformers",
    }
    if (
        not isinstance(runtime_versions, dict)
        or set(runtime_versions) != {"schema_version", "python", "packages"}
        or runtime_versions.get("schema_version") != 1
        or not isinstance(runtime_versions.get("python"), str)
        or not runtime_versions["python"].strip()
        or not isinstance(runtime_versions.get("packages"), dict)
        or set(runtime_versions["packages"]) != expected_runtime_packages
        or any(
            not isinstance(version, str) or not version.strip()
            for version in runtime_versions["packages"].values()
        )
    ):
        raise RuntimeError("BGE runtime version report differs from its exact schema")
    if completion.get("runtime_versions") != runtime_versions:
        raise RuntimeError("BGE completion and runtime version reports differ")

    report_path, report = _load_exactly_one(directory, "training_report.json")
    if completion.get("training_report") != report:
        raise RuntimeError("BGE completion and standalone training reports differ")
    if set(report.get("validation_splits", {})) != {"iid", "hard", "ood"}:
        raise RuntimeError("BGE report must contain IID/hard and the OOD sentinel")
    if report["validation_splits"]["ood"] != builder.OOD_SENTINEL:
        raise RuntimeError("BGE OOD=-1 sentinel differs")
    if report.get("evaluated_validation_splits") != ["iid", "hard"]:
        raise RuntimeError("BGE evaluated-validation declaration differs")
    if report.get("ood_evaluation_policy") != "disabled_train_contaminated":
        raise RuntimeError("BGE OOD evaluation policy differs")

    expected_report = {
        "original_training_examples": builder.EXPECTED_TRAIN,
        "training_subset": "all",
        "training_sampling": "none",
        "training_loss_weighting": "none",
        "training_unique_coverage_per_epoch": 1.0,
        "training_source_counts": {
            "human_train": builder.EXPECTED_HUMAN_TRAIN,
            "human_former_ood": builder.EXPECTED_FORMER_OOD,
        },
        "primary_validation_split": "iid",
        "experiment_group": "sft",
    }
    for key, expected in expected_report.items():
        if report.get(key) != expected:
            raise RuntimeError(f"BGE training report differs at {key}")
    for key in (
        "training_loss_weight_min",
        "training_loss_weight_median",
        "training_loss_weight_max",
    ):
        if not math.isclose(float(report.get(key, math.nan)), 1.0, abs_tol=1e-12):
            raise RuntimeError("BGE baseline changed external sample weights")
    source_mass = report.get("training_source_weight_mass")
    if not isinstance(source_mass, Mapping):
        raise RuntimeError("BGE report has no training source weight mass")
    for source, expected in expected_report["training_source_counts"].items():
        if not math.isclose(float(source_mass.get(source, math.nan)), expected, abs_tol=1e-6):
            raise RuntimeError("BGE training source weight mass differs")
    loss_hook = report.get("loss_hook")
    if not isinstance(loss_hook, Mapping) or loss_hook.get("sha256") != entry["loss_hook_sha256"]:
        raise RuntimeError("BGE report used a different loss hook")
    args = report.get("args")
    if not isinstance(args, Mapping):
        raise RuntimeError("BGE report has no trainer arguments")
    expected_runtime_args = _expected_report_args(entry)
    if args.get("model") != expected_runtime_args["model"]:
        raise RuntimeError("BGE trainer argument differs at model")
    if dict(args) != expected_runtime_args:
        raise RuntimeError("BGE trainer arguments differ from the exact runtime recipe")
    if args.get("batch_size", 0) * 2 * args.get("gradient_accumulation", 0) != 192:
        raise RuntimeError("BGE effective batch differs")
    runtime_values = (
        report.get("training_seconds"),
        report.get("validation_seconds"),
        report.get("total_pipeline_seconds"),
        report.get("examples_per_second"),
        completion.get("training_wall_seconds"),
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
        for value in runtime_values
    ):
        raise RuntimeError("BGE report has invalid runtime/throughput diagnostics")

    split_rows = {"iid": builder.EXPECTED_IID, "hard": builder.EXPECTED_HARD}
    frozen_validation_truth = _load_frozen_validation_truth(entry)
    for split, expected_rows in split_rows.items():
        metrics = report["validation_splits"][split]
        if metrics.get("examples") != expected_rows:
            raise RuntimeError(f"Unexpected BGE {split} validation row count")
        for metric in (
            "macro_average_precision",
            "overall_average_precision",
            "recall_at_precision_0_99",
            "roc_auc",
        ):
            _finite_probability(metrics.get(metric), f"{split}.{metric}")
        log_loss = metrics.get("log_loss")
        if (
            isinstance(log_loss, bool)
            or not isinstance(log_loss, (int, float))
            or not math.isfinite(float(log_loss))
            or float(log_loss) < 0
        ):
            raise RuntimeError(f"Invalid BGE {split} log loss")
        predictions = list(directory.rglob(f"{split}_validation_predictions.parquet"))
        if len(predictions) != 1 or predictions[0].stat().st_size == 0:
            raise RuntimeError(f"BGE slim output is missing {split} predictions")
        _validate_predictions(
            predictions[0],
            split=split,
            expected_rows=expected_rows,
            report_metrics=metrics,
            frozen_truth=frozen_validation_truth[split],
        )

    root_config = directory / "cross_encoder_config.json"
    _, trainer_config = _load_exactly_one(directory, "training_config.json")
    _validate_config(
        root_config,
        entry["expected_config"],
        runtime_model_path=entry["expected_runtime_model_path"],
        label="root config",
    )
    # _load_exactly_one parsed this already; use its path for one common validator.
    config_paths = list(directory.rglob("training_config.json"))
    _validate_config(
        config_paths[0],
        entry["expected_config"],
        runtime_model_path=entry["expected_runtime_model_path"],
        label="trainer config",
    )
    if trainer_config != json.loads(config_paths[0].read_text(encoding="utf-8")):
        raise RuntimeError("Could not read exact BGE trainer config")

    if require_sheets:
        sync = json.loads(
            (directory / "google_sheets_sync.json").read_text(encoding="utf-8")
        )
        expected_sync = {
            "status": "synced",
            "run_id": run_id,
            "experiment_group": "sft",
            "comparison_sheet": "sft_exps",
            "spreadsheet_id": shared.EXPERIMENT_SPREADSHEET_ID,
        }
        for key, expected in expected_sync.items():
            if sync.get(key) != expected:
                raise RuntimeError(f"BGE Google Sheets marker differs at {key}")
    return {
        "run_id": run_id,
        "experiment": entry["experiment"],
        "role": entry["role"],
        "comparison_sheet": "sft_exps",
        "iid_macro_ap": report["validation_splits"]["iid"]["macro_average_precision"],
        "hard_macro_ap": report["validation_splits"]["hard"]["macro_average_precision"],
        "ood_macro_ap": -1.0,
        "directory": str(directory),
        "training_report_path": str(report_path),
    }


def validate_run_payload(directory: Path, *, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Validate all immutable outputs before any local Sheets retry."""
    return _validate_run_output(directory, entry=entry, require_sheets=False)


def validate_run_output(directory: Path, *, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the payload and exact final ``sft_exps`` synchronization."""
    return _validate_run_output(directory, entry=entry, require_sheets=True)


def retry_pending_sheets_sync(directory: Path, *, kernel_ref: str) -> None:
    pending_path = directory / "sheets_sync_pending.json"
    sync_path = directory / "google_sheets_sync.json"
    completion_path = directory / "notebook_completed.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    run_id = str(completion.get("run_id", "")).strip()
    if (
        completion.get("status") != "complete"
        or not run_id
        or completion.get("experiment_group") != "sft"
    ):
        raise RuntimeError("Refusing to sync an incomplete BGE SFT artifact")
    existing: Mapping[str, Any] = {}
    if sync_path.is_file():
        existing = json.loads(sync_path.read_text(encoding="utf-8"))
    expected = {
        "status": "synced",
        "run_id": run_id,
        "experiment_group": "sft",
        "comparison_sheet": "sft_exps",
        "spreadsheet_id": shared.EXPERIMENT_SPREADSHEET_ID,
    }
    if not pending_path.exists() and all(existing.get(key) == value for key, value in expected.items()):
        return

    from push_google_sheets_credentials_dataset import DEFAULT_KEY_PATH
    from src.google_sheets_logger import load_service_account_info, sync_experiment

    raw_credentials = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if not raw_credentials:
        configured_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_PATH", "").strip()
        key_path = (
            Path(configured_path).expanduser().resolve()
            if configured_path
            else DEFAULT_KEY_PATH.expanduser().resolve()
        )
        raw_credentials = key_path.read_text(encoding="utf-8")
    load_service_account_info(raw_credentials)
    result = sync_experiment(
        spreadsheet_id=shared.EXPERIMENT_SPREADSHEET_ID,
        service_account_json=raw_credentials,
        completion=completion,
    )
    sync_path.write_text(
        json.dumps({"status": "synced", **result}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if pending_path.exists():
        preserved = directory / "sheets_sync_pending.remote.json"
        if preserved.exists():
            preserved = directory / f"sheets_sync_pending.remote-{time.time_ns()}.json"
        pending_path.rename(preserved)
    print(json.dumps({"local_sheets_retry": "synced", "kernel_ref": kernel_ref}))


def _listed_kernel_refs(output: str) -> set[str]:
    # Kaggle CLI 1.7.4 prints the literal text below with exit code 0 for an
    # empty ``kernels list --format json`` search instead of returning ``[]``.
    # Treat only that exact empty-result spelling as an empty list; every other
    # non-JSON response remains fail-closed.
    if output.strip().casefold() == "not found":
        return set()
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError("Kaggle kernel listing was not valid JSON") from error
    if not isinstance(payload, list):
        raise RuntimeError("Kaggle kernel listing JSON must be a list")
    refs: set[str] = set()
    for row in payload:
        if not isinstance(row, Mapping):
            raise RuntimeError("Kaggle kernel listing contains a non-object row")
        reference = row.get("ref") or row.get("id")
        if isinstance(reference, str) and reference:
            refs.add(reference)
    return refs


def _kernel_list_result(cli: list[str], kernel_ref: str):
    slug = kernel_ref.rsplit("/", 1)[-1]
    return kaggle.run_command(
        cli
        + [
            "kernels",
            "list",
            "--mine",
            "--search",
            slug,
            "--page-size",
            "100",
            "--format",
            "json",
        ],
        check=False,
    )


def remote_kernel_status(cli: list[str], kernel_ref: str) -> str:
    result = kaggle.run_command(cli + ["kernels", "status", kernel_ref], check=False)
    if result.returncode == 0:
        return kaggle.extract_status(result.stdout) or "status_error"
    listing = _kernel_list_result(cli, kernel_ref)
    if listing.returncode:
        raise RuntimeError(f"Could not resolve remote BGE kernel {kernel_ref}")
    if kernel_ref in _listed_kernel_refs(listing.stdout):
        raise RuntimeError(
            f"BGE kernel {kernel_ref} is listed but its status is unreadable"
        )
    # A single status error plus an empty/fuzzy search result is never enough
    # authority to create or replace a remote kernel.
    return "absence_unconfirmed"


def confirm_remote_absence(
    cli: list[str], kernel_ref: str, *, pause_seconds: float = 0.5
) -> None:
    """Require two independent list+status misses immediately before push."""
    for attempt in range(2):
        listing = _kernel_list_result(cli, kernel_ref)
        if listing.returncode:
            raise RuntimeError("Could not confirm remote BGE kernel absence")
        if kernel_ref in _listed_kernel_refs(listing.stdout):
            raise RuntimeError(
                f"Concurrent BGE kernel appeared before submission: {kernel_ref}"
            )
        status = kaggle.run_command(
            cli + ["kernels", "status", kernel_ref], check=False
        )
        if status.returncode == 0:
            observed = kaggle.extract_status(status.stdout) or "unknown"
            raise RuntimeError(
                f"Remote BGE kernel appeared before submission with status {observed}"
            )
        if attempt == 0 and pause_seconds > 0:
            time.sleep(pause_seconds)


def verify_remote_dataset_sources_exact(
    cli: list[str],
    *,
    kernel_ref: str,
    entry: Mapping[str, Any],
) -> None:
    with tempfile.TemporaryDirectory(prefix="bge-kernel-metadata-") as temp_dir:
        result = kaggle.run_command(
            cli + ["kernels", "pull", kernel_ref, "-p", temp_dir, "-m"],
            check=False,
        )
        if result.returncode:
            raise RuntimeError("Could not verify exact remote BGE Dataset attachments")
        metadata_path = Path(temp_dir) / "kernel-metadata.json"
        if not metadata_path.is_file():
            raise RuntimeError("Remote BGE kernel pull returned no metadata")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual = metadata.get("dataset_sources")
    expected = expected_dataset_sources(entry)
    # Kaggle canonicalizes the remote attachment order. Require exactly the
    # reviewed set with no duplicates, without assigning meaning to ordering.
    if (
        not isinstance(actual, list)
        or len(actual) != len(expected)
        or len(set(actual)) != len(actual)
        or set(actual) != set(expected)
    ):
        raise RuntimeError(
            "Remote BGE Dataset attachments are not exact: "
            + json.dumps(
                {"actual": actual, "expected": expected},
                ensure_ascii=False,
                sort_keys=True,
            )
        )


def push_new_kernel_after_confirmed_absence(
    cli: list[str],
    *,
    kernel_ref: str,
    entry: Mapping[str, Any],
    run_timeout: int,
) -> None:
    """Perform the last absence audit and then immediately create the kernel."""
    validate_staged_kernel_metadata(entry)
    confirm_remote_absence(cli, kernel_ref)
    stage_dir = kaggle.STAGE_ROOT / str(entry["kernel_slug"])
    push_result = kaggle.run_command(
        cli
        + [
            "kernels",
            "push",
            "-p",
            str(stage_dir),
            "--timeout",
            str(run_timeout),
            "--accelerator",
            "NvidiaTeslaT4",
        ]
    )
    if "not valid dataset sources" in push_result.stdout.lower():
        raise RuntimeError("Kaggle rejected a frozen BGE Dataset attachment")
    verify_remote_dataset_sources_exact(cli, kernel_ref=kernel_ref, entry=entry)


def download_output(
    cli: list[str],
    *,
    username: str,
    entry: Mapping[str, Any],
    full_download: bool,
) -> Path:
    root = output_root()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / str(entry["kernel_slug"])
    kernel_ref = f"{username}/{entry['kernel_slug']}"
    staging: Path | None = None
    for attempt in range(1, 4):
        candidate = Path(
            tempfile.mkdtemp(prefix=f".{entry['kernel_slug']}.download-", dir=root)
        )
        command = cli + [
            "kernels",
            "output",
            kernel_ref,
            "-p",
            str(candidate),
            "--force",
            "--page-size",
            "200",
        ]
        if not full_download:
            command.extend(["--file-pattern", SLIM_OUTPUT_PATTERN])
        result = kaggle.run_command(command, check=False)
        if result.returncode == 0:
            staging = candidate
            break
        shutil.rmtree(candidate)
        if attempt < 3:
            time.sleep(3 if attempt == 1 else 8)
    if staging is None:
        raise RuntimeError(f"Could not download BGE output after three attempts: {kernel_ref}")
    try:
        validate_run_payload(staging, entry=entry)
        retry_pending_sheets_sync(staging, kernel_ref=kernel_ref)
        validation = validate_run_output(staging, entry=entry)
    except Exception:
        # Preserve the invalid staging directory for diagnosis; it is exact and
        # intentionally not promoted over a previously validated run.
        print(f"Invalid BGE download preserved at: {staging}", file=sys.stderr)
        raise
    if destination.exists():
        backup = root / f".{entry['kernel_slug']}.previous-{time.time_ns()}"
        destination.rename(backup)
        print(f"Preserved previous BGE output at: {backup}")
    staging.rename(destination)
    validation["directory"] = str(destination)
    print(json.dumps({"validated_bge_output": validation}, ensure_ascii=False))
    return destination


def _select_entries(
    entries: list[dict[str, Any]],
    *,
    only: set[str] | None,
    include_candidates: bool,
) -> list[dict[str, Any]]:
    if only:
        selected = [
            entry
            for entry in entries
            if entry["key"] in only or entry["experiment"] in only
        ]
        matched = {
            token
            for token in only
            if any(token in {entry["key"], entry["experiment"]} for entry in selected)
        }
        if missing := only - matched:
            raise builder.CampaignConfigError(f"Unknown BGE variants: {sorted(missing)}")
    elif include_candidates:
        selected = list(entries)
    else:
        selected = [entry for entry in entries if entry["key"] == BASELINE_KEY]
    if not selected:
        raise builder.CampaignConfigError("No BGE variants selected")
    order = {spec["key"]: index for index, spec in enumerate(builder.VARIANT_SPECS)}
    return sorted(selected, key=lambda entry: order[entry["key"]])


def _validated_local_baseline(
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = next(entry for entry in entries if entry["key"] == BASELINE_KEY)
    directory = output_root() / baseline["kernel_slug"]
    if not directory.is_dir():
        raise RuntimeError(
            "Refusing a BGE candidate: the validated local baseline directory is missing: "
            f"{directory}"
        )
    return validate_run_output(directory, entry=baseline)


def _local_output_if_valid(entry: Mapping[str, Any]) -> Path | None:
    directory = output_root() / str(entry["kernel_slug"])
    if not directory.is_dir():
        return None
    try:
        validation = validate_run_output(directory, entry=entry)
    except Exception as error:
        print(f"Existing local BGE output is not reusable: {error}", flush=True)
        return None
    print(json.dumps({"reused_local_bge_output": validation}, ensure_ascii=False))
    return directory


def _enforce_live_environment() -> None:
    if kaggle.env_bool("KAGGLE_IS_PRIVATE", True) is not True:
        raise RuntimeError("BGE checkpoint/SFT kernels must remain private")
    accelerator = os.getenv("KAGGLE_ACCELERATOR", "NvidiaTeslaT4").strip()
    if accelerator != "NvidiaTeslaT4":
        raise RuntimeError("BGE campaign must request exact NvidiaTeslaT4 acceleration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--include-candidates", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--full-download", action="store_true")
    return parser.parse_args()


def run_live_campaign(
    *,
    args: argparse.Namespace,
    env_file: Path,
    owner: str,
    entries: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> int:
    _enforce_live_environment()
    token = os.getenv("KAGGLE_API_TOKEN", "").strip()
    if not token:
        raise SystemExit("Set KAGGLE_API_TOKEN in .env")
    cli = kaggle.kaggle_command()
    kaggle.run_command(cli + ["kernels", "list", "--mine", "--page-size", "1"])
    poll_interval = kaggle.env_int("KAGGLE_POLL_INTERVAL_SECONDS", 30, minimum=5)
    wait_timeout = kaggle.env_int("KAGGLE_WAIT_TIMEOUT_SECONDS", 45000, minimum=60)
    run_timeout = kaggle.env_int("KAGGLE_RUN_TIMEOUT_SECONDS", 43200, minimum=60)

    for entry in selected:
        if entry["kernel_slug"] in TERMINAL_FAILED_KERNEL_SLUGS:
            raise RuntimeError(
                "Refusing to resubmit a tombstoned terminal BGE kernel slug: "
                f"{entry['kernel_slug']}"
            )
        if entry["role"] != "baseline":
            baseline_validation = _validated_local_baseline(entries)
            print(json.dumps({"bge_candidate_gate": baseline_validation}, ensure_ascii=False))
        if _local_output_if_valid(entry) is not None:
            continue
        kernel_ref = f"{owner}/{entry['kernel_slug']}"
        status = remote_kernel_status(cli, kernel_ref)
        print(json.dumps({"kernel_ref": kernel_ref, "status": status}))
        if status == "absence_unconfirmed":
            # Stage locally without network, then make the repeated remote
            # absence audit the final external action before the direct push.
            command = runner_command(
                entry,
                env_file=env_file,
                dry_run=True,
                no_wait=False,
            )
            run_inner_runner(command)
            validate_staged_kernel_metadata(entry)
            push_new_kernel_after_confirmed_absence(
                cli,
                kernel_ref=kernel_ref,
                entry=entry,
                run_timeout=run_timeout,
            )
            if args.no_wait:
                print("Submitted one BGE kernel without waiting; campaign stops here.")
                return 0
            kaggle.wait_for_kernel(
                cli,
                kernel_ref,
                poll_interval=poll_interval,
                wait_timeout=wait_timeout,
            )
            status = "complete"
        elif status in {"queued", "running"}:
            verify_remote_dataset_sources_exact(
                cli, kernel_ref=kernel_ref, entry=entry
            )
            if args.no_wait:
                print("Existing BGE kernel is still active; not waiting.")
                return 0
            kaggle.wait_for_kernel(
                cli,
                kernel_ref,
                poll_interval=poll_interval,
                wait_timeout=wait_timeout,
            )
            status = "complete"
        elif status in kaggle.TERMINAL_FAILURE:
            kaggle.run_command(cli + ["kernels", "logs", kernel_ref], check=False)
            raise RuntimeError(
                f"BGE kernel {kernel_ref} is terminally failed; automatic replacement is forbidden"
            )
        elif status in kaggle.TERMINAL_SUCCESS:
            verify_remote_dataset_sources_exact(
                cli, kernel_ref=kernel_ref, entry=entry
            )
        else:
            raise RuntimeError(f"Unexpected BGE kernel status: {status!r}")
        if status in kaggle.TERMINAL_SUCCESS or status == "complete":
            download_output(
                cli,
                username=owner,
                entry=entry,
                full_download=args.full_download,
            )
    return 0


def main() -> int:
    args = parse_args()
    env_file = args.env_file if args.env_file.is_absolute() else (ROOT / args.env_file)
    kaggle.load_dotenv(env_file)
    owner = os.getenv("KAGGLE_USERNAME", "").strip()
    if not owner:
        raise SystemExit("Set KAGGLE_USERNAME in .env")
    owner = kaggle.validate_slug(owner, "KAGGLE_USERNAME")
    if owner != "alexproger23":
        raise RuntimeError("Frozen BGE campaign owner must remain alexproger23")
    requested = set(args.only) or None
    if requested:
        # Candidate-only invocations still materialize the exact baseline entry
        # so the local gate can validate its immutable identity.
        build_only = set(requested) | {BASELINE_KEY}
    elif args.include_candidates:
        build_only = None
    else:
        build_only = {BASELINE_KEY}
    entries = builder.build_campaign(
        owner=owner,
        only=build_only,
    )
    if any(entry["validation_dataset"] != VALIDATION_DATASET for entry in entries):
        raise RuntimeError("Generated BGE validation Dataset owner differs")
    if any(entry["checkpoint_dataset"] != CHECKPOINT_DATASET for entry in entries):
        raise RuntimeError("Generated BGE checkpoint Dataset owner differs")
    if any(entry["kernel_slug"] in TERMINAL_FAILED_KERNEL_SLUGS for entry in entries):
        raise RuntimeError("Generated BGE campaign reused a terminal failed kernel slug")
    selected = _select_entries(
        entries,
        only=requested,
        include_candidates=args.include_candidates,
    )
    if args.no_wait and len(selected) != 1:
        raise SystemExit("--no-wait is allowed for exactly one BGE variant")
    print(json.dumps({
        "campaign": builder.CAMPAIGN,
        "selected": [entry["experiment"] for entry in selected],
        "execution": "sequential",
        "default_baseline_only": not args.include_candidates and not args.only,
    }, ensure_ascii=False, indent=2))

    if args.dry_run:
        for entry in selected:
            run_inner_runner(
                runner_command(entry, env_file=env_file, dry_run=True, no_wait=False)
            )
            validate_staged_kernel_metadata(entry)
        return 0
    with exclusive_campaign_lock():
        return run_live_campaign(
            args=args,
            env_file=env_file,
            owner=owner,
            entries=entries,
            selected=selected,
        )


if __name__ == "__main__":
    raise SystemExit(main())
