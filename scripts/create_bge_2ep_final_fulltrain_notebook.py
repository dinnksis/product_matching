#!/usr/bin/env python3
"""Build one immutable BGE two-epoch full-human deployment export notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from textwrap import dedent
from typing import Any, Mapping

import nbformat as nbf

import create_bge_2ep_sft_notebooks as base
import create_cross_encoder_training_notebook as cross_builder
import create_qwen_training_notebook as shared
import push_bge_pretrain_checkpoint_dataset as checkpoint_push


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = "bge_2ep_final_fulltrain_export_v1"
EXPERIMENT = "bge2_sft_final_fulltrain_e2_lr2e5_v1"
DEFAULT_CONFIG = ROOT / "configs/cross_encoder_bge_2ep_final_fulltrain_export_v1.json"
DEFAULT_SOURCE_DIR = base.DEFAULT_SOURCE_DIR
DEFAULT_CHECKPOINT_STAGE_DIR = base.DEFAULT_CHECKPOINT_STAGE_DIR
DEFAULT_OUTPUT_DIR = ROOT / "notebooks" / CAMPAIGN

EXPECTED_CONFIG_FILE_SHA256 = (
    "733d0d0a477f54eff447fc4df7c5bbc5a6d770dc81ff8a79f29667899f20ee55"
)
EXPECTED_RECIPE_SHA256 = (
    "d46be9217a43396ecc8c594fc1864ee93761d288c30e5a40041adbb28bd7adfe"
)
EXPECTED_ASSIGNMENTS_SHA256 = (
    "36ce04075e66009e895f3d1b14d76a1bed8169cfd8e4fb1098787d80254973d4"
)
EXPECTED_ASSIGNMENTS_BYTES = 8_300_407
EXPECTED_ORIGINAL_MATCHES_SHA256 = (
    "14b0e07f750204807aff0bb7510d385a0b3fae3f02088506a14242187ca1e492"
)
EXPECTED_ITEMS_SHA256 = (
    "a787b58485632bad9039ad04f5b6210080e8d91e0c03e3a5c79d678962a14b1d"
)
EXPECTED_ROWS = 365_654
EXPECTED_POSITIVES = 93_890
EXPECTED_CATEGORIES = 20
EXPECTED_TRAINING_EXAMPLES = 731_308
EXPECTED_STEPS_PER_EPOCH_PER_RANK = 22_854
EXPECTED_UPDATES_PER_EPOCH = 1_905
EXPECTED_TOTAL_UPDATES = 3_810
EXPECTED_WARMUP_UPDATES = 190
UNAVAILABLE_METRIC = {
    "evaluated": False,
    "status": "unavailable_fulltrain_no_holdout",
    "reason": "all human-labelled pairs were used for final deployment training",
    "examples": 0,
    "macro_average_precision": -1.0,
    "overall_average_precision": -1.0,
    "recall_at_precision_0_99": -1.0,
    "threshold_at_precision_0_99": -1.0,
    "roc_auc": -1.0,
    "log_loss": -1.0,
    "per_category_average_precision": {},
    "predictions_file": None,
}
SHEETS_NOTES = (
    "Final deployable BGE export trained on all human labels; "
    "IID/hard/OOD quality metrics are unavailable and recorded as -1."
)

SPLITS: tuple[dict[str, Any], ...] = (
    {
        "split": "train",
        "label_source": "human_train",
        "relative": "human/train_pairs.parquet",
        "rows": 306_669,
        "positives": 80_136,
        "bytes": 3_610_560,
        "sha256": "ffc8ac7283ca0fe1ac8a39e8ca31cb4fa069465576eeb636005c72636935e616",
    },
    {
        "split": "iid_validation",
        "label_source": "human_iid",
        "relative": "human/iid_validation_pairs.parquet",
        "rows": 12_000,
        "positives": 3_118,
        "bytes": 169_551,
        "sha256": "8964422e8c3b254a355d35b5fb60568fed4b6532abb1e8918aba1050f7ecb798",
    },
    {
        "split": "hard_validation",
        "label_source": "human_hard",
        "relative": "human/hard_validation_pairs.parquet",
        "rows": 5_814,
        "positives": 1_481,
        "bytes": 81_673,
        "sha256": "6731ff9c41100b0ea21d6868cd91cbfab1038832d500753cbab8d4e6962fb064",
    },
    {
        "split": "ood_validation",
        "label_source": "human_former_ood",
        "relative": "human/ood_validation_pairs.parquet",
        "rows": 41_171,
        "positives": 9_155,
        "bytes": 591_288,
        "sha256": "e12eebba6afd6c307bd70475eea29d80df513280b262328cf819bac65e8b22a4",
    },
)

SELECTION_RECEIPTS: tuple[tuple[Path, str], ...] = (
    (
        Path("reports/bge_2ep_sft_candidate_v1/lr/lr_selection_receipt.json"),
        "a19a342c40abcae55e448d7d5808707723ff521dfbfa33c8c844d4b9fdc0883f",
    ),
    (
        Path("reports/bge_2ep_sft_candidate_v1/e2/epoch_selection_receipt.json"),
        "e76cfb4b4ad5c53db38b493f221115abbab158c63d6a53bffcb9421b3e81368c",
    ),
    (
        Path("reports/bge_2ep_sft_loss_confirmation_v1/screen/loss_screen_receipt.json"),
        "853334a69e0f0334b6a43a9b509137e0a79b5da04558de0d9563a265ad2e5554",
    ),
)

EXPECTED_CONFIG = dict(base.EXPECTED_BASE_CONFIG)
EXPECTED_CONFIG["epochs"] = 2

FULLTRAIN_LOSS_HOOK_SOURCE = dedent(
    """
    from __future__ import annotations

    import torch
    import torch.nn.functional as F


    LOSS_VARIANT = "bce_finite_guard_final_fulltrain_v1"


    def initialize_loss(*, train_frame, device, rank, world_size):
        if len(train_frame) != 365654:
            raise ValueError("Final BGE BCE hook received an unexpected row count")
        expected = {
            "human_train": 306669,
            "human_iid": 12000,
            "human_hard": 5814,
            "human_former_ood": 41171,
        }
        actual = {
            str(key): int(value)
            for key, value in train_frame["label_source"].value_counts().items()
        }
        if actual != expected:
            raise ValueError("Final BGE BCE hook received unexpected sources")
        return None


    def compute_loss(
        *, logits, targets, sample_weights, pair_indices, orientations, epoch, step
    ):
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite final BGE logits")
        if not torch.isfinite(targets).all():
            raise FloatingPointError("non-finite final BGE targets")
        if not torch.isfinite(sample_weights).all():
            raise FloatingPointError("non-finite final BGE sample weights")
        denominator = sample_weights.sum()
        if not torch.isfinite(denominator) or denominator <= 0:
            raise FloatingPointError("invalid final BGE loss denominator")
        per_example = F.binary_cross_entropy_with_logits(
            logits.float(), targets, reduction="none"
        )
        loss = (per_example * sample_weights).sum() / denominator
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite final BGE BCE loss")
        return {"loss": loss, "bce": loss.detach()}
    """
).strip() + "\n"
FULLTRAIN_LOSS_HOOK_SHA256 = hashlib.sha256(
    FULLTRAIN_LOSS_HOOK_SOURCE.encode("utf-8")
).hexdigest()

RUNTIME_EMBEDDED_FILES = base.RUNTIME_EMBEDDED_FILES + (
    Path("scripts/train_bge_2ep_final_fulltrain.py"),
)
SOURCE_LEDGER_FILES = tuple(
    dict.fromkeys(
        base.SOURCE_LEDGER_FILES
        + RUNTIME_EMBEDDED_FILES
        + (
            Path("configs/cross_encoder_bge_2ep_final_fulltrain_export_v1.json"),
            Path("scripts/create_bge_2ep_final_fulltrain_notebook.py"),
            Path("scripts/run_bge_2ep_final_fulltrain.py"),
        )
        + tuple(path for path, _ in SELECTION_RECEIPTS)
    )
)


class FinalExportConfigError(ValueError):
    """Raised locally when the one-shot export contract drifts."""


def file_sha256(path: Path) -> str:
    return checkpoint_push.sha256_file(path)


def source_bundle() -> tuple[dict[str, str], list[dict[str, Any]], str]:
    runtime_names = {path.as_posix() for path in RUNTIME_EMBEDDED_FILES}
    sources: dict[str, str] = {}
    ledger: list[dict[str, Any]] = []
    for relative in SOURCE_LEDGER_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Final export source is missing: {path}")
        payload = path.read_bytes()
        record = {
            "path": relative.as_posix(),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "runtime_embedded": relative.as_posix() in runtime_names,
        }
        ledger.append(record)
        if record["runtime_embedded"]:
            sources[relative.as_posix()] = payload.decode("utf-8")
    if len({record["path"] for record in ledger}) != len(ledger):
        raise FinalExportConfigError("Final export source ledger has duplicate paths")
    if set(sources) != runtime_names:
        raise FinalExportConfigError("Final export runtime source set differs")
    source_sha256 = base.canonical_sha256({"schema_version": 1, "files": ledger})
    return sources, ledger, source_sha256


def validate_local_authorities(config_path: Path) -> dict[str, Any]:
    if file_sha256(config_path) != EXPECTED_CONFIG_FILE_SHA256:
        raise FinalExportConfigError("Final export config file changed")
    config = cross_builder.load_training_config(config_path)
    if config != EXPECTED_CONFIG:
        raise FinalExportConfigError("Final export config values changed")
    if base.canonical_sha256(config) != EXPECTED_RECIPE_SHA256:
        raise FinalExportConfigError("Final export is not the selected e2 recipe")
    for path, expected_sha256 in SELECTION_RECEIPTS:
        if file_sha256(ROOT / path) != expected_sha256:
            raise FinalExportConfigError(f"Selection receipt changed: {path}")
    loss_receipt = json.loads((ROOT / SELECTION_RECEIPTS[-1][0]).read_text())
    if (
        loss_receipt.get("status") != "complete"
        or loss_receipt.get("seed42_winner") != "bce_finite_guard_v1"
        or loss_receipt.get("challenger_accepted_for_seed17") is not False
        or loss_receipt.get("epoch_selection_receipt_sha256")
        != SELECTION_RECEIPTS[1][1]
        or loss_receipt.get("lr_selection_receipt_sha256")
        != SELECTION_RECEIPTS[0][1]
    ):
        raise FinalExportConfigError("Selection receipt chain does not select BCE e2")
    return config


def validate_data_authority(validation: Mapping[str, Any]) -> None:
    manifest = validation["manifest"]
    outputs = manifest["outputs"]
    assignment = outputs.get("human/split_assignments.parquet")
    if assignment != {
        "bytes": EXPECTED_ASSIGNMENTS_BYTES,
        "sha256": EXPECTED_ASSIGNMENTS_SHA256,
    }:
        raise FinalExportConfigError("Split-assignment authority changed")
    if outputs.get("human/items.parquet", {}).get("sha256") != EXPECTED_ITEMS_SHA256:
        raise FinalExportConfigError("Frozen item payload changed")
    for spec in SPLITS:
        if outputs.get(spec["relative"]) != {
            "bytes": spec["bytes"],
            "sha256": spec["sha256"],
        }:
            raise FinalExportConfigError(f"Frozen split changed: {spec['split']}")
    source = manifest.get("sources", {}).get("data/matches.parquet", {})
    if source.get("sha256") != EXPECTED_ORIGINAL_MATCHES_SHA256:
        raise FinalExportConfigError("Original matches source identity changed")


def final_identity(
    *,
    config: Mapping[str, Any],
    source_sha256: str,
    validation_manifest_sha256: str,
    checkpoint_manifest_sha256: str,
    executable_cells_sha256: str,
) -> str:
    return base.canonical_sha256(
        {
            "schema_version": 1,
            "campaign": CAMPAIGN,
            "experiment": EXPERIMENT,
            "purpose": "final_deployment_export",
            "quality_evaluation": False,
            "config": dict(config),
            "source_sha256": source_sha256,
            "validation_manifest_sha256": validation_manifest_sha256,
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "executable_cells_sha256": executable_cells_sha256,
            "train_policy": "all_human_assignments_sorted_by_human_row_id_v1",
            "train_rows": EXPECTED_ROWS,
            "train_positives": EXPECTED_POSITIVES,
            "loss_hook_sha256": FULLTRAIN_LOSS_HOOK_SHA256,
            "selection_receipts": {
                path.as_posix(): digest for path, digest in SELECTION_RECEIPTS
            },
        }
    )


def kernel_slug(identity_sha256: str) -> str:
    slug = f"pm-b2-final-{identity_sha256[:12]}-s42-v1"
    if len(slug) > 50 or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug):
        raise FinalExportConfigError("Unsafe final-export Kaggle slug")
    return slug


def _data_cell() -> nbf.NotebookNode:
    return base.code(
        f"""
        ASSIGNMENTS_RELATIVE = "human/split_assignments.parquet"
        assignments_path = dataset_file(
            EXPECTED_VALIDATION_DATASET_SLUG, "human_split_assignments.parquet"
        )
        expected_assignment = validation_manifest["outputs"][ASSIGNMENTS_RELATIVE]
        if expected_assignment != {{
            "bytes": {EXPECTED_ASSIGNMENTS_BYTES},
            "sha256": {EXPECTED_ASSIGNMENTS_SHA256!r},
        }}:
            raise RuntimeError("Attached assignment declaration changed")
        if (
            assignments_path.stat().st_size != expected_assignment["bytes"]
            or file_sha256(assignments_path) != expected_assignment["sha256"]
        ):
            raise RuntimeError("Attached split assignments changed")
        source_matches = validation_manifest["sources"]["data/matches.parquet"]
        if source_matches.get("sha256") != {EXPECTED_ORIGINAL_MATCHES_SHA256!r}:
            raise RuntimeError("Original matches provenance changed")

        items = pd.read_parquet(attached_files["human/items.parquet"])
        assignments = pd.read_parquet(assignments_path)
        expected_schema = [
            "human_row_id", "component_id", "id1", "id2", "target", "category", "split"
        ]
        if list(assignments.columns) != expected_schema:
            raise RuntimeError("Split-assignment schema changed")
        if len(items) != {base.EXPECTED_ITEMS} or not items["id"].is_unique:
            raise RuntimeError("Frozen human item table changed")
        if len(assignments) != {EXPECTED_ROWS} or assignments.isnull().any().any():
            raise RuntimeError("Split assignments have invalid rows/nulls")
        assignments = assignments.sort_values("human_row_id", kind="stable").reset_index(drop=True)
        if not np.array_equal(
            assignments["human_row_id"].to_numpy(),
            np.arange({EXPECTED_ROWS}, dtype=np.int64),
        ):
            raise RuntimeError("human_row_id is not the contiguous source authority")

        split_specs = {list(SPLITS)!r}
        split_by_name = {{spec["split"]: spec for spec in split_specs}}
        if set(assignments["split"].astype(str)) != set(split_by_name):
            raise RuntimeError("Split-assignment labels changed")
        split_checks = {{}}
        pair_columns = ["id1", "id2", "target"]
        for split_name, spec in split_by_name.items():
            assigned = assignments.loc[
                assignments["split"] == split_name, pair_columns
            ].reset_index(drop=True)
            frozen = pd.read_parquet(attached_files[spec["relative"]])[pair_columns]
            frozen = frozen.reset_index(drop=True)
            if not assigned.equals(frozen):
                raise RuntimeError(f"Assignments differ from frozen {{split_name}} rows")
            if len(assigned) != spec["rows"] or int(assigned["target"].sum()) != spec["positives"]:
                raise RuntimeError(f"Frozen {{split_name}} counts changed")
            split_checks[split_name] = {{
                "rows": len(assigned),
                "positives": int(assigned["target"].sum()),
                "source_sha256": spec["sha256"],
            }}

        item_categories = items.set_index("id")["category"]
        if (assignments["id1"] == assignments["id2"]).any():
            raise RuntimeError("Full human train contains self-pairs")
        if not assignments["target"].isin([0.0, 1.0]).all():
            raise RuntimeError("Full human targets are not binary")
        left_category = assignments["id1"].map(item_categories)
        right_category = assignments["id2"].map(item_categories)
        if left_category.isnull().any() or right_category.isnull().any():
            raise RuntimeError("Full human train references missing items")
        if not left_category.equals(right_category):
            raise RuntimeError("Full human train contains cross-category pairs")
        if not left_category.astype(str).equals(assignments["category"].astype(str)):
            raise RuntimeError("Assignment categories differ from item categories")
        if left_category.nunique() != {EXPECTED_CATEGORIES}:
            raise RuntimeError("Full human category count changed")
        all_item_ids = set(assignments["id1"]) | set(assignments["id2"])
        if len(all_item_ids) != {base.EXPECTED_ITEMS}:
            raise RuntimeError("Full human pairs no longer cover every item")
        lower = np.minimum(assignments["id1"].to_numpy(), assignments["id2"].to_numpy())
        upper = np.maximum(assignments["id1"].to_numpy(), assignments["id2"].to_numpy())
        if pd.MultiIndex.from_arrays([lower, upper]).duplicated().any():
            raise RuntimeError("Full human train contains duplicate unordered pairs")
        split_item_sets = {{
            name: set(frame["id1"]) | set(frame["id2"])
            for name, frame in (
                (name, assignments.loc[assignments["split"] == name])
                for name in split_by_name
            )
        }}
        for left_index, left_name in enumerate(split_by_name):
            for right_name in list(split_by_name)[left_index + 1:]:
                if split_item_sets[left_name] & split_item_sets[right_name]:
                    raise RuntimeError("Frozen source splits are no longer component-disjoint")

        source_mapping = {{
            spec["split"]: spec["label_source"] for spec in split_specs
        }}
        train_pairs = assignments[pair_columns].copy()
        train_pairs["label_source"] = assignments["split"].map(source_mapping)
        if train_pairs["label_source"].isnull().any():
            raise RuntimeError("Final label_source mapping is incomplete")
        if len(train_pairs) != {EXPECTED_ROWS} or int(train_pairs["target"].sum()) != {EXPECTED_POSITIVES}:
            raise RuntimeError("Final full-human train counts changed")
        source_counts = {{
            str(key): int(value)
            for key, value in train_pairs["label_source"].value_counts().items()
        }}
        expected_source_counts = {{
            spec["label_source"]: spec["rows"] for spec in split_specs
        }}
        if source_counts != expected_source_counts:
            raise RuntimeError("Final label_source counts changed")

        PREPARED_DIR.mkdir(parents=True, exist_ok=False)
        materialized_train_path = PREPARED_DIR / "train_pairs.parquet"
        train_pairs.to_parquet(materialized_train_path, index=False, compression="zstd")
        materialized_train = pd.read_parquet(materialized_train_path)
        try:
            pd.testing.assert_frame_equal(
                materialized_train,
                train_pairs,
                check_exact=True,
                check_dtype=True,
                check_index_type=True,
                check_column_type=True,
            )
        except AssertionError as error:
            raise RuntimeError(
                "Materialized final train changed rows, order, columns, or dtypes"
            ) from error
        item_destination = PREPARED_DIR / "items.parquet"
        item_destination.symlink_to(attached_files["human/items.parquet"])
        if list(PREPARED_DIR.glob("*validation*.parquet")):
            raise RuntimeError("Final export prepared a validation parquet")

        TRAIN_DATA_REPORT = {{
            "schema_version": 1,
            "policy": "all_human_assignments_sorted_by_human_row_id_v1",
            "quality_evaluation": False,
            "withheld_pairs": 0,
            "items": len(items),
            "train_pairs": len(train_pairs),
            "train_positives": int(train_pairs["target"].sum()),
            "train_positive_rate": float(train_pairs["target"].mean()),
            "categories": int(left_category.nunique()),
            "human_row_id": {{"minimum": 0, "maximum": {EXPECTED_ROWS - 1}, "contiguous": True}},
            "source_counts": source_counts,
            "split_checks": split_checks,
            "assignment_file": {{
                "bytes": expected_assignment["bytes"],
                "sha256": expected_assignment["sha256"],
            }},
            "original_matches_source_sha256": source_matches["sha256"],
            "materialized_train_parquet": {{
                "bytes": materialized_train_path.stat().st_size,
                "sha256": file_sha256(materialized_train_path),
                "readback_frame_exact": True,
                "columns": list(materialized_train.columns),
                "dtypes": {{
                    column: str(dtype)
                    for column, dtype in materialized_train.dtypes.items()
                }},
            }},
            "pairwise_source_split_item_overlap": 0,
        }}
        TRAIN_DATA_REPORT_PATH.write_text(
            json.dumps(TRAIN_DATA_REPORT, ensure_ascii=False, indent=2) + "\\n",
            encoding="utf-8",
        )
        print(json.dumps(TRAIN_DATA_REPORT, ensure_ascii=False, indent=2))
        del items, assignments, train_pairs, materialized_train, left_category, right_category, split_item_sets
        """,
        "frozen",
        "full-human-data",
    )


def _loss_cell() -> nbf.NotebookNode:
    return base.code(
        f"""
        FULLTRAIN_LOSS_HOOK_SOURCE = {FULLTRAIN_LOSS_HOOK_SOURCE!r}
        EXPECTED_FULLTRAIN_LOSS_HOOK_SHA256 = {FULLTRAIN_LOSS_HOOK_SHA256!r}
        if hashlib.sha256(FULLTRAIN_LOSS_HOOK_SOURCE.encode("utf-8")).hexdigest() != EXPECTED_FULLTRAIN_LOSS_HOOK_SHA256:
            raise RuntimeError("Final BGE BCE hook was edited")
        LOSS_HOOK_PATH.write_text(FULLTRAIN_LOSS_HOOK_SOURCE, encoding="utf-8")
        if file_sha256(LOSS_HOOK_PATH) != EXPECTED_FULLTRAIN_LOSS_HOOK_SHA256:
            raise RuntimeError("Materialized final BGE BCE hook differs")
        """,
        "frozen",
        "fulltrain-bce",
    )


def _deterministic_working_tree_cell() -> nbf.NotebookNode:
    return base.code(
        """
        # Prevent subprocess imports and the later embedded Sheets logger from
        # leaving version-specific __pycache__ files in /kaggle/working.
        sys.dont_write_bytecode = True
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        """,
        "frozen",
        "deterministic-working-tree",
    )


def _training_cell() -> nbf.NotebookNode:
    helper = base._streaming_process_helper()
    return base.code(
        helper
        + "\n\n"
        + dedent(
            """
            if memory_preflight.get("status") != "passed":
                raise RuntimeError("Final BGE training cannot start before preflight")
            train_command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                str(PROJECT_ROOT / "scripts/train_bge_2ep_final_fulltrain.py"),
                "--config", str(RUNTIME_CONFIG_PATH),
                "--prepared-dir", str(PREPARED_DIR),
                "--output-dir", str(TRAINER_OUTPUT_DIR),
                "--token-cache-dir", str(TOKEN_CACHE_DIR),
                "--loss-hook", str(LOSS_HOOK_PATH),
            ]
            training_started = time.perf_counter()
            run_logged(train_command, TRAIN_LOG)
            training_wall_seconds = time.perf_counter() - training_started
            """
        ).strip(),
        "frozen",
        "training-only",
    )


def _completion_cell(*, recipe_sha256: str) -> nbf.NotebookNode:
    receipt_hashes = {path.as_posix(): digest for path, digest in SELECTION_RECEIPTS}
    return base.code(
        f"""
        from datetime import datetime, timezone
        import stat
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        raw_report_path = TRAINER_OUTPUT_DIR / "training_report.json"
        raw_config_path = TRAINER_OUTPUT_DIR / "training_config.json"
        if not raw_report_path.is_file() or not raw_config_path.is_file():
            raise RuntimeError("Final BGE trainer emitted no report/config")
        raw_report = json.loads(raw_report_path.read_text(encoding="utf-8"))
        expected_sources = {{spec["label_source"]: spec["rows"] for spec in {list(SPLITS)!r}}}
        if (
            raw_report.get("status") != "complete"
            or raw_report.get("purpose") != "final_deployment_export"
            or raw_report.get("quality_evaluation") is not False
            or raw_report.get("validation_splits") != []
            or raw_report.get("validation_predictions_written") is not False
            or raw_report.get("original_training_examples") != {EXPECTED_ROWS}
            or raw_report.get("training_examples") != {EXPECTED_TRAINING_EXAMPLES}
            or raw_report.get("training_source_counts") != expected_sources
            or raw_report.get("training_sampling") != "none"
            or raw_report.get("training_subset") != "all"
            or raw_report.get("training_loss_weighting") != "none"
            or raw_report.get("steps_per_epoch_per_rank") != {EXPECTED_STEPS_PER_EPOCH_PER_RANK}
            or raw_report.get("updates_per_epoch") != {EXPECTED_UPDATES_PER_EPOCH}
            or raw_report.get("planned_optimizer_updates") != {EXPECTED_TOTAL_UPDATES}
            or raw_report.get("optimizer_step_attempts") != {EXPECTED_TOTAL_UPDATES}
            or raw_report.get("warmup_updates") != {EXPECTED_WARMUP_UPDATES}
            or raw_report.get("gradient_accumulation_normalization")
                != "sample_exact_group_mean_v1"
            or raw_report.get("loss_hook", {{}}).get("sha256") != EXPECTED_FULLTRAIN_LOSS_HOOK_SHA256
        ):
            raise RuntimeError("Final BGE training report changed its frozen contract")
        overflow_skips = raw_report.get("amp_overflow_skips")
        if not isinstance(overflow_skips, int) or not 0 <= overflow_skips <= 16:
            raise RuntimeError("Final BGE AMP overflow count is invalid")
        if raw_report.get("optimizer_steps_succeeded") != {EXPECTED_TOTAL_UPDATES} - overflow_skips:
            raise RuntimeError("Final BGE optimizer counters are inconsistent")
        runtime_config = json.loads(raw_config_path.read_text(encoding="utf-8"))
        expected_runtime_config = dict(CANONICAL_TRAIN_CONFIG)
        expected_runtime_config["model"] = str(INITIAL_MODEL_PATH)
        if runtime_config != expected_runtime_config:
            raise RuntimeError("Final trainer config differs from frozen recipe")
        if list(TRAINER_OUTPUT_DIR.glob("*validation*")):
            raise RuntimeError("Final trainer emitted a validation artifact")
        expected_trainer_files = {{
            "config.json", "model.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
            "training_config.json", "training_report.json",
        }}
        actual_trainer_files = {{
            path.name for path in TRAINER_OUTPUT_DIR.iterdir() if path.is_file()
        }}
        if actual_trainer_files != expected_trainer_files:
            raise RuntimeError(f"Unexpected final trainer files: {{sorted(actual_trainer_files)}}")

        if OUTPUT_DIR.exists():
            raise RuntimeError("Final output directory already exists")
        MODEL_DIR = OUTPUT_DIR / "model"
        MODEL_DIR.mkdir(parents=True, exist_ok=False)
        model_filenames = {{
            "config.json", "model.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"
        }}
        for filename in sorted(model_filenames):
            source = TRAINER_OUTPUT_DIR / filename
            if not source.is_file() or source.stat().st_size <= 0:
                raise RuntimeError(f"Missing deployable model file: {{filename}}")
            shutil.move(str(source), MODEL_DIR / filename)
        if (MODEL_DIR / "model.safetensors").stat().st_size < 2_000_000_000:
            raise RuntimeError("Final BGE safetensors file is unexpectedly small")

        output_config = AutoConfig.from_pretrained(
            MODEL_DIR, trust_remote_code=False, local_files_only=True
        )
        output_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR, use_fast=True, trust_remote_code=False, local_files_only=True
        )
        if output_config.model_type != "xlm-roberta" or output_config.num_labels != 1:
            raise RuntimeError("Deployable BGE config changed")
        if output_tokenizer.pad_token_id is None:
            raise RuntimeError("Deployable BGE tokenizer has no pad token")
        probe_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR,
            num_labels=1,
            attn_implementation="sdpa",
            trust_remote_code=False,
            local_files_only=True,
        ).to("cuda:0")
        parameter_count = sum(parameter.numel() for parameter in probe_model.parameters())
        if parameter_count != {base.EXPECTED_PARAMETERS}:
            raise RuntimeError("Deployable BGE parameter count changed")
        probe_inputs = output_tokenizer(
            ["товар model 123"],
            ["товар model 123"],
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )
        probe_inputs = {{key: value.to("cuda:0") for key, value in probe_inputs.items()}}
        probe_model.eval()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            probe_logits = probe_model(**probe_inputs).logits
        if tuple(probe_logits.shape) != (1, 1) or not torch.isfinite(probe_logits).all():
            raise RuntimeError("Deployable BGE smoke forward failed")
        del probe_logits, probe_inputs, probe_model
        torch.cuda.empty_cache()

        deployment_smoke = {{
            "schema_version": 1,
            "status": "passed",
            "quality_evaluation": False,
            "check": "load_tokenizer_model_and_one_finite_forward",
            "parameters": parameter_count,
            "logit_shape": [1, 1],
            "finite": True,
        }}
        training_summary = {{
            "schema_version": 1,
            "status": "complete",
            "purpose": "final_deployment_export",
            "quality_evaluation": False,
            "validation_splits": [],
            "validation_predictions_written": False,
            "recipe_sha256": {recipe_sha256!r},
            "loss_variant": "bce_finite_guard_final_fulltrain_v1",
            "loss_hook_sha256": EXPECTED_FULLTRAIN_LOSS_HOOK_SHA256,
            "training_examples": raw_report["training_examples"],
            "original_training_examples": raw_report["original_training_examples"],
            "epochs": 2,
            "steps_per_epoch_per_rank": raw_report["steps_per_epoch_per_rank"],
            "updates_per_epoch": raw_report["updates_per_epoch"],
            "planned_optimizer_updates": raw_report["planned_optimizer_updates"],
            "optimizer_steps_succeeded": raw_report["optimizer_steps_succeeded"],
            "amp_overflow_skips": raw_report["amp_overflow_skips"],
            "warmup_updates": raw_report["warmup_updates"],
            "gradient_accumulation_normalization": raw_report[
                "gradient_accumulation_normalization"
            ],
            "epoch_batch_geometry": raw_report["epoch_batch_geometry"],
            "training_seconds": raw_report["training_seconds"],
            "examples_per_second": raw_report["examples_per_second"],
            "padding_efficiency": raw_report["padding_efficiency"],
            "peak_vram_gib_by_rank": raw_report["peak_vram_gib_by_rank"],
            "training_source_counts": raw_report["training_source_counts"],
        }}
        unavailable_metric = {UNAVAILABLE_METRIC!r}
        sheets_training_report = {{
            "schema_version": 1,
            "status": "complete",
            "purpose": "final_deployment_export",
            "experiment_group": "sft",
            "quality_evaluation": False,
            "original_training_examples": {EXPECTED_ROWS},
            "training_examples": {EXPECTED_TRAINING_EXAMPLES},
            "training_source_counts": raw_report["training_source_counts"],
            "validation_splits": {{
                split: dict(unavailable_metric) for split in ("iid", "hard", "ood")
            }},
            "args": dict(CANONICAL_TRAIN_CONFIG),
            "notes": {SHEETS_NOTES!r},
        }}
        (OUTPUT_DIR / "training_config.json").write_text(
            json.dumps(CANONICAL_TRAIN_CONFIG, ensure_ascii=False, indent=2) + "\\n",
            encoding="utf-8",
        )
        (OUTPUT_DIR / "training_summary.json").write_text(
            json.dumps(training_summary, ensure_ascii=False, indent=2) + "\\n",
            encoding="utf-8",
        )
        (OUTPUT_DIR / "training_report.json").write_text(
            json.dumps(sheets_training_report, ensure_ascii=False, indent=2) + "\\n",
            encoding="utf-8",
        )
        (OUTPUT_DIR / "deployment_smoke.json").write_text(
            json.dumps(deployment_smoke, ensure_ascii=False, indent=2) + "\\n",
            encoding="utf-8",
        )
        shutil.copy2(TRAIN_DATA_REPORT_PATH, OUTPUT_DIR / "train_data_report.json")
        shutil.copy2(PREFLIGHT_REPORT_PATH, OUTPUT_DIR / "memory_preflight.json")
        shutil.copy2(RUNTIME_VERSIONS_PATH, OUTPUT_DIR / "runtime_versions.json")

        def stable_tree_record(path, root):
            before = path.lstat()
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise RuntimeError(f"Final artifact is not an isolated regular file: {{path}}")
            resolved_root = root.resolve(strict=True)
            resolved_path = path.resolve(strict=True)
            if not resolved_path.is_relative_to(resolved_root):
                raise RuntimeError(f"Final artifact escapes its root: {{path}}")
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            digest_object = hashlib.sha256()
            try:
                opened_before = os.fstat(descriptor)
                while True:
                    chunk = os.read(descriptor, 8 * 1024 * 1024)
                    if not chunk:
                        break
                    digest_object.update(chunk)
                opened_after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            after = path.lstat()
            identity_before = (
                before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                before.st_ctime_ns,
                before.st_mode, before.st_nlink,
            )
            identity_opened_before = (
                opened_before.st_dev, opened_before.st_ino, opened_before.st_size,
                opened_before.st_mtime_ns, opened_before.st_ctime_ns,
                opened_before.st_mode, opened_before.st_nlink,
            )
            identity_opened_after = (
                opened_after.st_dev, opened_after.st_ino, opened_after.st_size,
                opened_after.st_mtime_ns, opened_after.st_ctime_ns,
                opened_after.st_mode, opened_after.st_nlink,
            )
            identity_after = (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_mode, after.st_nlink,
            )
            if not (
                identity_before == identity_opened_before
                == identity_opened_after == identity_after
            ):
                raise RuntimeError(f"Final artifact changed while hashing: {{path}}")
            return {{
                "path": path.relative_to(root).as_posix(),
                "bytes": before.st_size,
                "sha256": digest_object.hexdigest(),
            }}

        def strict_tree_entries(root):
            root_details = root.lstat()
            if not stat.S_ISDIR(root_details.st_mode) or root.is_symlink():
                raise RuntimeError("Final artifact root is not a real directory")
            directories = set()
            files = set()
            for current_raw, directory_names, file_names in os.walk(root, followlinks=False):
                current = Path(current_raw)
                for name in directory_names:
                    path = current / name
                    details = path.lstat()
                    if not stat.S_ISDIR(details.st_mode) or path.is_symlink():
                        raise RuntimeError(f"Unsafe final artifact directory: {{path}}")
                    directories.add(path)
                for name in file_names:
                    path = current / name
                    details = path.lstat()
                    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
                        raise RuntimeError(f"Unsafe final artifact file: {{path}}")
                    files.add(path)
            return directories, files

        artifact_directories, artifact_files = strict_tree_entries(OUTPUT_DIR)
        records = [
            stable_tree_record(path, OUTPUT_DIR)
            for path in sorted(artifact_files)
            if path.name != "artifact_manifest.json"
        ]
        expected_paths = {{
            "model/config.json", "model/model.safetensors", "model/special_tokens_map.json", "model/tokenizer.json",
            "model/tokenizer_config.json", "training_config.json", "training_summary.json",
            "training_report.json", "deployment_smoke.json", "train_data_report.json", "memory_preflight.json",
            "runtime_versions.json",
        }}
        if {{record["path"] for record in records}} != expected_paths:
            raise RuntimeError("Final artifact tree differs from the strict allowlist")
        actual_directories = {{
            path.relative_to(OUTPUT_DIR).as_posix() for path in artifact_directories
        }}
        if actual_directories != {{"model"}}:
            raise RuntimeError("Final artifact directory tree differs")
        final_model_record = next(
            record for record in records if record["path"] == "model/model.safetensors"
        )
        if final_model_record["sha256"] == checkpoint_manifest["reconstruction"]["sha256"]:
            raise RuntimeError("Final weights are byte-identical to the initial checkpoint")
        tree_sha256 = canonical_json_sha256({{"schema_version": 1, "files": records}})
        artifact_manifest = {{
            "schema_version": 1,
            "root": {EXPERIMENT!r},
            "campaign_identity_sha256": EXPECTED_CAMPAIGN_IDENTITY_SHA256,
            "tree_sha256": tree_sha256,
            "files": records,
            "file_count": len(records),
            "total_bytes": sum(record["bytes"] for record in records),
        }}
        artifact_manifest_path = OUTPUT_DIR / "artifact_manifest.json"
        artifact_manifest_path.write_text(
            json.dumps(artifact_manifest, ensure_ascii=False, indent=2) + "\\n",
            encoding="utf-8",
        )
        artifact_manifest_record = stable_tree_record(
            artifact_manifest_path, OUTPUT_DIR
        )
        final_directories, final_files = strict_tree_entries(OUTPUT_DIR)
        if final_directories != artifact_directories:
            raise RuntimeError("Final artifact directories changed after manifest creation")
        if {{path.relative_to(OUTPUT_DIR).as_posix() for path in final_files}} != expected_paths | {{"artifact_manifest.json"}}:
            raise RuntimeError("Final artifact tree changed after manifest creation")

        if not RUNTIME_VERSIONS_PATH.is_file():
            raise RuntimeError("Final BGE runtime version report is missing")
        completion = {{
            "schema_version": 1,
            "status": "complete",
            "run_id": EXPERIMENT_RUN_ID,
            "started_at_utc": EXPERIMENT_STARTED_AT_UTC,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "experiment": {EXPERIMENT!r},
            "campaign": {CAMPAIGN!r},
            "purpose": "final_deployment_export",
            "experiment_group": "sft",
            "quality_evaluation": False,
            "validation_splits": [],
            "validation_predictions_written": False,
            "dataset_ref": EXPECTED_VALIDATION_DATASET_REF,
            "validation_manifest_sha256": EXPECTED_VALIDATION_MANIFEST_SHA256,
            "initial_checkpoint_ref": EXPECTED_CHECKPOINT_REF,
            "initial_checkpoint_manifest_sha256": EXPECTED_CHECKPOINT_MANIFEST_SHA256,
            "initial_checkpoint_model_sha256": checkpoint_manifest["reconstruction"]["sha256"],
            "code_bundle_sha256": EXPECTED_SOURCE_SHA256,
            "frozen_recipe_sha256": {recipe_sha256!r},
            "campaign_identity_sha256": EXPECTED_CAMPAIGN_IDENTITY_SHA256,
            "executable_cells_sha256": EXPECTED_EXECUTABLE_CELLS_SHA256,
            "loss_variant": "bce_finite_guard_final_fulltrain_v1",
            "loss_hook_sha256": EXPECTED_FULLTRAIN_LOSS_HOOK_SHA256,
            "selection_receipt_sha256": {receipt_hashes!r},
            "train_data": TRAIN_DATA_REPORT,
            "memory_preflight": memory_preflight,
            "training_summary": training_summary,
            "training_report": sheets_training_report,
            "deployment_smoke": deployment_smoke,
            "artifact_directory": {EXPERIMENT!r},
            "artifact_manifest_sha256": artifact_manifest_record["sha256"],
            "artifact_tree_sha256": tree_sha256,
            "artifact_file_count": len(records),
            "training_wall_seconds": training_wall_seconds,
            "kaggle_kernel_ref": (
                os.getenv("KAGGLE_KERNEL_RUN_ID")
                or os.getenv("KAGGLE_KERNEL_INFERENCE_RUN_ID")
                or ""
            ),
        }}
        shutil.rmtree(TRAINER_OUTPUT_DIR)
        shutil.rmtree(PROJECT_ROOT)
        if list(WORKING_ROOT.rglob("*validation_predictions*.parquet")):
            raise RuntimeError("Final BGE output contains validation predictions")
        completion_path = WORKING_ROOT / "notebook_completed.json"
        completion_path.write_text(
            json.dumps(completion, ensure_ascii=False, indent=2, default=str) + "\\n",
            encoding="utf-8",
        )
        print(json.dumps({{
            "status": "complete",
            "purpose": "final_deployment_export",
            "quality_evaluation": False,
            "artifact_tree_sha256": tree_sha256,
            "model_sha256": next(record["sha256"] for record in records if record["path"] == "model/model.safetensors"),
        }}, ensure_ascii=False, indent=2))
        """,
        "frozen",
        "strict-export-completion",
    )


def build_notebook(
    *,
    owner: str,
    config_path: Path = DEFAULT_CONFIG,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    checkpoint_stage_dir: Path = DEFAULT_CHECKPOINT_STAGE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write: bool = True,
) -> dict[str, Any]:
    if owner != "alexproger23":
        raise FinalExportConfigError("Final export owner must remain alexproger23")
    config = validate_local_authorities(config_path)
    validation = base.load_validation_dataset(source_dir, owner)
    validate_data_authority(validation)
    checkpoint = base.load_checkpoint_dataset(
        checkpoint_stage_dir, owner, verify_payload=True
    )
    sources, source_ledger, source_sha256 = source_bundle()
    recipe_sha256 = base.canonical_sha256(config)
    runtime_model_path = f"/kaggle/temp/{EXPERIMENT}/initial_checkpoint"

    run_identity = shared.experiment_run_initialization_cell()
    run_identity.metadata["tags"] = ["frozen", "run-identity"]
    sheet_cells = shared.google_sheets_tracking_cells()
    for cell in sheet_cells:
        cell.metadata["tags"] = ["frozen", "sheets-sync"]
    cells = [
        base.markdown(
            f"""
            # Final BGE full-human deployment export

            This is a non-experimental export run.  It trains the selected e2,
            LR=2e-5, plain-BCE recipe on all {EXPECTED_ROWS:,} human-labelled
            pairs in original `human_row_id` order, with zero held-out pairs.
            It produces no quality metric and no validation predictions.

            Campaign identity: `{base.IDENTITY_PLACEHOLDER}`.
            """,
            "frozen",
            "deployment-export",
        ),
        base._setup_cell(
            experiment=EXPERIMENT,
            validation=validation,
            checkpoint=checkpoint,
            source_sha256=source_sha256,
            identity_sha256=base.IDENTITY_PLACEHOLDER,
            executable_sha256=base.EXECUTABLE_CELLS_PLACEHOLDER,
        ),
        _deterministic_working_tree_cell(),
        run_identity,
        base.markdown("## Selected frozen recipe", "frozen"),
        base._recipe_cell(config, recipe_sha256),
        base.markdown("## Hash-bound runtime sources", "frozen"),
        base._sources_cell(sources, source_ledger, source_sha256),
        base.markdown("## All human labels; no holdout", "frozen"),
        _data_cell(),
        _loss_cell(),
        base.markdown("## Exact two-T4 optimizer-state preflight", "frozen"),
        base._preflight_cell(),
        base.markdown("## Training-only DDP export", "frozen"),
        _training_cell(),
        base.markdown("## Deployability probe and strict tree manifest", "frozen"),
        _completion_cell(recipe_sha256=recipe_sha256),
        *sheet_cells,
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    executable_sha256 = base.executable_cells_sha256(notebook)
    identity_sha256 = final_identity(
        config=config,
        source_sha256=source_sha256,
        validation_manifest_sha256=str(validation["manifest_sha256"]),
        checkpoint_manifest_sha256=str(checkpoint["manifest_sha256"]),
        executable_cells_sha256=executable_sha256,
    )
    base._replace_notebook_placeholders(
        notebook,
        identity_sha256=identity_sha256,
        executable_sha256=executable_sha256,
    )
    base._assign_deterministic_cell_ids(notebook)
    slug = kernel_slug(identity_sha256)
    metadata = {
        "template": CAMPAIGN,
        "campaign": CAMPAIGN,
        "experiment": EXPERIMENT,
        "purpose": "final_deployment_export",
        "experiment_group": "sft",
        "quality_evaluation": False,
        "google_sheets_tracking": True,
        "validation_dataset": validation["dataset"],
        "validation_manifest_sha256": validation["manifest_sha256"],
        "initial_checkpoint": checkpoint["dataset"],
        "initial_checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
        "source_sha256": source_sha256,
        "frozen_recipe_sha256": recipe_sha256,
        "campaign_identity_sha256": identity_sha256,
        "executable_cells_sha256": executable_sha256,
        "loss_hook_sha256": FULLTRAIN_LOSS_HOOK_SHA256,
        "runtime_model_path": runtime_model_path,
        "train_pairs": EXPECTED_ROWS,
        "train_positives": EXPECTED_POSITIVES,
        "withheld_pairs": 0,
        "validation_splits": [],
        "expected_gpus": 2,
        "kernel_slug": slug,
        "editable_cells": [],
    }
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "product_matching_training": metadata,
        }
    )
    nbf.validate(notebook)
    serialized_notebook = nbf.writes(notebook)
    if not serialized_notebook.endswith("\n"):
        serialized_notebook += "\n"
    notebook_sha256 = hashlib.sha256(
        serialized_notebook.encode("utf-8")
    ).hexdigest()
    entry = {
        **metadata,
        "key": "final_fulltrain",
        "kernel_slug": slug,
        "title": slug,
        "recipe_sha256": recipe_sha256,
        "identity_sha256": identity_sha256,
        "executable_cells_sha256": executable_sha256,
        "source_sha256": source_sha256,
        "checkpoint_dataset": checkpoint["dataset"],
        "checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
        "checkpoint_model_sha256": checkpoint_push.EXPECTED_SOURCE_FILES[
            checkpoint_push.MODEL_FILENAME
        ]["sha256"],
        "expected_config": config,
        "expected_runtime_model_path": runtime_model_path,
        "notebook_sha256": notebook_sha256,
        "notebook": str(output_dir / f"{EXPERIMENT}_2xt4.ipynb"),
    }
    validate_notebook_identity(notebook, entry=entry)
    if write:
        destination = Path(entry["notebook"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(notebook, destination)
        if file_sha256(destination) != notebook_sha256:
            raise FinalExportConfigError("Written final notebook bytes differ")
        load_and_validate_notebook(destination, entry=entry)
    return entry


def validate_notebook_identity(
    notebook: nbf.NotebookNode, *, entry: Mapping[str, Any]
) -> dict[str, Any]:
    nbf.validate(notebook)
    metadata = notebook.metadata.get("product_matching_training")
    if not isinstance(metadata, Mapping):
        raise FinalExportConfigError("Final notebook has no frozen metadata")
    metadata_keys = {
        "template", "campaign", "experiment", "purpose", "experiment_group", "quality_evaluation",
        "google_sheets_tracking", "validation_dataset", "validation_manifest_sha256",
        "initial_checkpoint", "initial_checkpoint_manifest_sha256", "source_sha256",
        "frozen_recipe_sha256", "campaign_identity_sha256", "executable_cells_sha256",
        "loss_hook_sha256", "runtime_model_path", "train_pairs", "train_positives",
        "withheld_pairs", "validation_splits", "expected_gpus", "kernel_slug",
        "editable_cells",
    }
    expected_metadata = {key: entry[key] for key in metadata_keys}
    if dict(metadata) != expected_metadata:
        raise FinalExportConfigError("Final notebook metadata differs")
    sources, _, current_source_sha256 = source_bundle()
    if not sources or current_source_sha256 != entry["source_sha256"]:
        raise FinalExportConfigError("Final notebook source ledger is stale")
    actual_executable_sha256 = base.executable_cells_sha256(
        notebook,
        identity_sha256=str(entry["identity_sha256"]),
        expected_sha256=str(entry["executable_cells_sha256"]),
    )
    if actual_executable_sha256 != entry["executable_cells_sha256"]:
        raise FinalExportConfigError("Final executable cells changed")
    expected_identity = final_identity(
        config=entry["expected_config"],
        source_sha256=str(entry["source_sha256"]),
        validation_manifest_sha256=str(entry["validation_manifest_sha256"]),
        checkpoint_manifest_sha256=str(entry["checkpoint_manifest_sha256"]),
        executable_cells_sha256=actual_executable_sha256,
    )
    if expected_identity != entry["identity_sha256"]:
        raise FinalExportConfigError("Final campaign identity changed")
    if kernel_slug(expected_identity) != entry["kernel_slug"]:
        raise FinalExportConfigError("Final kernel slug changed")
    return {
        "identity_sha256": expected_identity,
        "executable_cells_sha256": actual_executable_sha256,
        "source_sha256": current_source_sha256,
    }


def load_and_validate_notebook(
    path: Path, *, entry: Mapping[str, Any]
) -> nbf.NotebookNode:
    if file_sha256(path) != entry.get("notebook_sha256"):
        raise FinalExportConfigError("Final notebook file SHA-256 differs")
    notebook = nbf.read(path, as_version=4)
    validate_notebook_identity(notebook, entry=entry)
    return notebook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--owner")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument(
        "--checkpoint-stage-dir", type=Path, default=DEFAULT_CHECKPOINT_STAGE_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    owner = args.owner or shared.dotenv_username(args.env_file)
    if not owner:
        raise SystemExit("Set KAGGLE_USERNAME in .env or pass --owner")
    entry = build_notebook(
        owner=owner,
        config_path=args.config,
        source_dir=args.source_dir,
        checkpoint_stage_dir=args.checkpoint_stage_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(entry, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
