#!/usr/bin/env python3
"""Build the final MiniLM 5ep -> all-human 3-epoch Kaggle notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from pprint import pformat
from textwrap import dedent

import nbformat as nbf
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "cross_encoder_minilm_5ep_full_human_final.json"
DEFAULT_NOTEBOOK = ROOT / "notebooks" / "minilm_5ep_full_human_final_2xt4.ipynb"
ITEMS_PATH = ROOT / "data" / "items_human.parquet"
MATCHES_PATH = ROOT / "data" / "matches.parquet"
EXPECTED_ITEM_ROWS = 711_304
EXPECTED_PAIR_ROWS = 365_654
DEFAULT_CHECKPOINT_DATASET_REF = (
    "alexproger23/product-matching-minilm-llm-pretrain-5ep-full"
)
CHECKPOINT_MANIFEST_SHA256 = (
    "39225939bc099d69326f0c07ad15bff03b4e441f64817f33f92aa925dfdd9c12"
)
SOURCE_FILES = (
    Path("requirements-cross-encoder.txt"),
    Path("scripts/train_cross_encoder.py"),
    Path("src/__init__.py"),
    Path("src/cross_encoder_training.py"),
    Path("src/cross_encoder_experiment_hooks.py"),
    Path("src/data_pipeline.py"),
    Path("src/experiment_protocol.py"),
    Path("src/pair_features.py"),
    Path("src/qwen_reranker.py"),
    Path("src/qwen_training.py"),
    Path("src/validation_metrics.py"),
)
LOCKED_VALUES = {
    "epochs": 3,
    "batch_size": 96,
    "gradient_accumulation": 1,
    "learning_rate": 8e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
    "max_length": 384,
    "sampling": "none",
    "loss_weighting": "none",
    "classifier_dropout": 0.1,
    "label_smoothing": 0.0,
    "max_grad_norm": 1.0,
    "seed": 42,
    "skip_validation": True,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def markdown(value: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(value).strip())


def code(value: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(value).strip())


def load_locked_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    differences = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in LOCKED_VALUES.items()
        if config.get(key) != expected
    }
    if differences:
        raise ValueError(
            "Final training recipe differs from the user-locked values: "
            + json.dumps(differences, ensure_ascii=False, sort_keys=True)
        )
    if config.get("train_subset") != "all":
        raise ValueError("Final training must retain every human pair")
    if config.get("lexical_hard_negative_strength") != 0.0:
        raise ValueError("Final training must not apply lexical loss weights")
    return config


def validate_local_human_data() -> dict[str, object]:
    items = pd.read_parquet(ITEMS_PATH, columns=["id"])
    pairs = pd.read_parquet(MATCHES_PATH, columns=["id1", "id2", "target"])
    if len(items) != EXPECTED_ITEM_ROWS or len(pairs) != EXPECTED_PAIR_ROWS:
        raise ValueError(
            f"Unexpected local human data size: items={len(items)}, pairs={len(pairs)}"
        )
    left = pairs[["id1", "id2"]].min(axis=1)
    right = pairs[["id1", "id2"]].max(axis=1)
    unordered = pd.DataFrame({"left": left, "right": right, "target": pairs["target"]})
    grouped = unordered.groupby(["left", "right"], sort=False)["target"].agg(
        ["size", "nunique"]
    )
    duplicate_pairs = int((grouped["size"] > 1).sum())
    contradictory_pairs = int((grouped["nunique"] > 1).sum())
    if duplicate_pairs or contradictory_pairs:
        raise ValueError(
            "The final human data unexpectedly contains duplicate/conflicting pairs: "
            f"duplicates={duplicate_pairs}, contradictions={contradictory_pairs}"
        )
    return {
        "items_rows": len(items),
        "pairs_rows": len(pairs),
        "positive_pairs": int((pairs["target"] == 1.0).sum()),
        "negative_pairs": int((pairs["target"] == 0.0).sum()),
        "duplicate_unordered_pairs": duplicate_pairs,
        "contradictory_pairs": contradictory_pairs,
        "items_sha256": sha256(ITEMS_PATH),
        "matches_sha256": sha256(MATCHES_PATH),
    }


def embedded_sources() -> tuple[dict[str, str], str]:
    sources = {
        path.as_posix(): (ROOT / path).read_text(encoding="utf-8")
        for path in SOURCE_FILES
    }
    digest = hashlib.sha256(
        json.dumps(
            sources,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return sources, digest


def build_notebook(
    config: dict[str, object],
    data_audit: dict[str, object],
    *,
    data_dataset_ref: str,
    checkpoint_dataset_ref: str,
    checkpoint_manifest_sha256: str = CHECKPOINT_MANIFEST_SHA256,
    experiment_name: str = "minilm_5ep_full_human_final",
    notebook_title: str = "Final MiniLM: 5ep pretrain checkpoint -> all human labels",
    locked_recipe: dict[str, object] | None = None,
    expected_amp_dtype: str | None = None,
    expected_world_size: int = 2,
    expected_effective_batch_size: int = 192,
) -> nbf.NotebookNode:
    if locked_recipe is None:
        locked_recipe = LOCKED_VALUES
    sources, source_hash = embedded_sources()
    config_literal = pformat(config, sort_dicts=False, width=100)
    config_source = (
        f"TRAIN_CONFIG = {config_literal}\n"
        + dedent(
            """
            TRAIN_CONFIG["model"] = str(checkpoint_path)
            CONFIG_PATH.write_text(
                json.dumps(TRAIN_CONFIG, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(json.dumps(TRAIN_CONFIG, ensure_ascii=False, indent=2))
            """
        ).strip()
    )
    cells = [
        markdown(
            f"""
            # {notebook_title}

            Финальное full fine-tuning без validation: все `{EXPECTED_PAIR_ROWS:,}`
            human-пар используются в train на каждой из трёх эпох. Результаты никуда,
            кроме `/kaggle/working`, не отправляются.

            Data: `{data_dataset_ref}`
            Checkpoint Dataset: `{checkpoint_dataset_ref}`
            """
        ),
        code(
            f"""
            import hashlib
            import json
            import os
            import subprocess
            import sys
            import time
            from datetime import datetime, timezone
            from pathlib import Path

            INPUT_ROOT = Path("/kaggle/input")
            WORKING_ROOT = Path("/kaggle/working")
            TEMP_ROOT = Path("/kaggle/temp/{experiment_name}")
            PROJECT_ROOT = TEMP_ROOT / "product_matching"
            PREPARED_DIR = TEMP_ROOT / "prepared"
            TOKEN_CACHE_DIR = TEMP_ROOT / "token_cache"
            OUTPUT_DIR = WORKING_ROOT / {experiment_name!r}
            CONFIG_PATH = WORKING_ROOT / {f"{experiment_name}_config.json"!r}
            TRAIN_LOG = WORKING_ROOT / {f"{experiment_name}.log"!r}
            EXPECTED_DATASET_REF = {data_dataset_ref!r}
            EXPECTED_CHECKPOINT_DATASET_REF = {checkpoint_dataset_ref!r}
            EXPECTED_CHECKPOINT_MANIFEST_SHA256 = {checkpoint_manifest_sha256!r}
            EXPECTED_DATA = {data_audit!r}
            EXPECTED_SOURCE_SHA256 = {source_hash!r}
            LOCKED_RECIPE = {locked_recipe!r}
            EXPECTED_AMP_DTYPE = {expected_amp_dtype!r}
            EXPECTED_WORLD_SIZE = {expected_world_size!r}
            EXPECTED_EFFECTIVE_BATCH_SIZE = {expected_effective_batch_size!r}

            def file_sha256(path):
                digest = hashlib.sha256()
                with Path(path).open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                return digest.hexdigest()

            def verified_input(filename, expected_sha256):
                candidates = []
                for path in INPUT_ROOT.glob(f"**/{{filename}}"):
                    if path.is_file() and file_sha256(path) == expected_sha256:
                        candidates.append(path)
                unique = {{str(path.resolve()): path for path in candidates}}
                if len(unique) != 1:
                    raise RuntimeError(
                        f"Expected exactly one {{filename}} with SHA-256 "
                        f"{{expected_sha256}}, found {{list(unique.values())}}"
                    )
                return next(iter(unique.values()))

            items_path = verified_input("items_human.parquet", EXPECTED_DATA["items_sha256"])
            matches_path = verified_input("matches.parquet", EXPECTED_DATA["matches_sha256"])

            checkpoint_manifest_candidates = [
                path
                for path in INPUT_ROOT.glob("**/checkpoint_manifest.json")
                if path.is_file()
                and file_sha256(path) == EXPECTED_CHECKPOINT_MANIFEST_SHA256
            ]
            unique_manifests = {{
                str(path.resolve()): path for path in checkpoint_manifest_candidates
            }}
            if len(unique_manifests) != 1:
                raise RuntimeError(
                    "Expected exactly one checkpoint_manifest.json for "
                    f"{{EXPECTED_CHECKPOINT_DATASET_REF}}, found "
                    f"{{list(unique_manifests.values())}}"
                )
            checkpoint_manifest_path = next(iter(unique_manifests.values()))
            checkpoint_manifest = json.loads(
                checkpoint_manifest_path.read_text(encoding="utf-8")
            )
            if checkpoint_manifest.get("dataset") != EXPECTED_CHECKPOINT_DATASET_REF:
                raise RuntimeError("Unexpected checkpoint Dataset reference")
            checkpoint_root = checkpoint_manifest_path.parent
            for filename, declaration in checkpoint_manifest["files"].items():
                checkpoint_file = checkpoint_root / filename
                if not checkpoint_file.is_file():
                    raise RuntimeError(f"Checkpoint file is missing: {{filename}}")
                if (
                    checkpoint_file.stat().st_size != declaration["bytes"]
                    or file_sha256(checkpoint_file) != declaration["sha256"]
                ):
                    raise RuntimeError(f"Checkpoint file differs from manifest: {{filename}}")
            reconstruction = checkpoint_manifest.get("reconstruction")
            if reconstruction:
                checkpoint_path = TEMP_ROOT / "initial_checkpoint"
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                part_names = set(reconstruction["parts"])
                for filename in checkpoint_manifest["files"]:
                    if filename in part_names:
                        continue
                    destination = checkpoint_path / filename
                    if destination.exists() or destination.is_symlink():
                        destination.unlink()
                    destination.symlink_to(checkpoint_root / filename)
                reconstructed_model = checkpoint_path / reconstruction["filename"]
                model_digest = hashlib.sha256()
                with reconstructed_model.open("wb") as destination:
                    for part_name in reconstruction["parts"]:
                        with (checkpoint_root / part_name).open("rb") as source:
                            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                                destination.write(chunk)
                                model_digest.update(chunk)
                if (
                    reconstructed_model.stat().st_size != reconstruction["bytes"]
                    or model_digest.hexdigest() != reconstruction["sha256"]
                ):
                    raise RuntimeError("Reconstructed checkpoint differs from manifest")
            else:
                checkpoint_path = checkpoint_root
            print(json.dumps({{
                "items_path": str(items_path),
                "matches_path": str(matches_path),
                "checkpoint_dataset": EXPECTED_CHECKPOINT_DATASET_REF,
                "checkpoint_path": str(checkpoint_path),
                "expected_data": EXPECTED_DATA,
            }}, ensure_ascii=False, indent=2))
            print(subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True).stdout)
            """
        ),
        markdown("## Locked final recipe"),
        code(config_source),
        markdown("## Embedded trainer and dependencies"),
        code(
            f"""
            EMBEDDED_SOURCES = {sources!r}
            PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
            for relative, content in EMBEDDED_SOURCES.items():
                destination = PROJECT_ROOT / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(content, encoding="utf-8")
            actual_source_hash = hashlib.sha256(
                json.dumps(
                    EMBEDDED_SOURCES,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if actual_source_hash != EXPECTED_SOURCE_SHA256:
                raise RuntimeError("Embedded source fingerprint changed")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--quiet",
                    "--disable-pip-version-check",
                    "--upgrade-strategy",
                    "only-if-needed",
                    "-r",
                    str(PROJECT_ROOT / "requirements-cross-encoder.txt"),
                ],
                check=True,
            )
            """
        ),
        markdown("## Prepare every human pair — no split and no filtering"),
        code(
            """
            import numpy as np
            import pandas as pd
            sys.path.insert(0, str(PROJECT_ROOT))
            from src.data_pipeline import serialize_product

            preparation_started = time.perf_counter()
            items = pd.read_parquet(
                items_path, columns=["id", "name", "attributes", "category"]
            )
            pairs = pd.read_parquet(matches_path, columns=["id1", "id2", "target"])
            if len(items) != EXPECTED_DATA["items_rows"]:
                raise RuntimeError(f"Unexpected item count: {len(items)}")
            if len(pairs) != EXPECTED_DATA["pairs_rows"]:
                raise RuntimeError(f"Unexpected pair count: {len(pairs)}")
            if items["id"].duplicated().any() or items["id"].isna().any():
                raise RuntimeError("Item IDs must be unique and non-null")
            if pairs[["id1", "id2", "target"]].isna().any().any():
                raise RuntimeError("Human pairs contain nulls")
            if (pairs["id1"] == pairs["id2"]).any():
                raise RuntimeError("Human pairs contain self-pairs")
            if not pairs["target"].isin([0.0, 1.0]).all():
                raise RuntimeError("Human labels must be binary")
            known_ids = set(items["id"])
            pair_ids = set(pairs["id1"]) | set(pairs["id2"])
            if missing_ids := pair_ids - known_ids:
                raise RuntimeError(f"Missing product IDs: {len(missing_ids)}")
            left = np.minimum(pairs["id1"].to_numpy(), pairs["id2"].to_numpy())
            right = np.maximum(pairs["id1"].to_numpy(), pairs["id2"].to_numpy())
            unordered = pd.DataFrame({"left": left, "right": right, "target": pairs["target"]})
            grouped = unordered.groupby(["left", "right"], sort=False)["target"].agg(
                ["size", "nunique"]
            )
            duplicate_pairs = int((grouped["size"] > 1).sum())
            contradictory_pairs = int((grouped["nunique"] > 1).sum())
            if duplicate_pairs or contradictory_pairs:
                raise RuntimeError(
                    f"Duplicate/conflicting pairs: {duplicate_pairs}/{contradictory_pairs}"
                )
            categories = items.set_index("id")["category"]
            if not pairs["id1"].map(categories).equals(pairs["id2"].map(categories)):
                raise RuntimeError("Human data contains cross-category pairs")

            items = items.copy()
            items["product_text"] = items.apply(
                serialize_product, axis=1, max_attribute_chars=6000
            )
            if not items["product_text"].str.startswith("Категория: ").all():
                raise RuntimeError("Serialization must start with Категория")
            if not items["product_text"].str.contains("\\nНазвание: ", regex=False).all():
                raise RuntimeError("Serialization must contain Название on line two")

            PREPARED_DIR.mkdir(parents=True, exist_ok=True)
            items[["id", "product_text", "category"]].to_parquet(
                PREPARED_DIR / "items.parquet", index=False, compression="zstd"
            )
            pairs.to_parquet(
                PREPARED_DIR / "train_pairs.parquet", index=False, compression="zstd"
            )
            preparation_report = {
                "items": len(items),
                "train_pairs": len(pairs),
                "positive_pairs": int((pairs["target"] == 1.0).sum()),
                "negative_pairs": int((pairs["target"] == 0.0).sum()),
                "duplicate_unordered_pairs": duplicate_pairs,
                "contradictory_pairs": contradictory_pairs,
                "validation_pairs": 0,
                "filtered_pairs": 0,
                "serialization": "Категория + Название + JSON key/value lines",
                "elapsed_seconds": time.perf_counter() - preparation_started,
            }
            (WORKING_ROOT / "full_human_data_report.json").write_text(
                json.dumps(preparation_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(json.dumps(preparation_report, ensure_ascii=False, indent=2))
            print(items["product_text"].iloc[0][:2000])
            del items, pairs, categories, unordered, grouped, known_ids, pair_ids
            """
        ),
        markdown("## 2×T4 DDP full fine-tuning"),
        code(
            """
            train_command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                str(PROJECT_ROOT / "scripts/train_cross_encoder.py"),
                "--config",
                str(CONFIG_PATH),
                "--prepared-dir",
                str(PREPARED_DIR),
                "--output-dir",
                str(OUTPUT_DIR),
                "--token-cache-dir",
                str(TOKEN_CACHE_DIR),
            ]
            training_environment = os.environ.copy()
            training_environment.update({
                "OMP_NUM_THREADS": "2",
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "PYTHONUNBUFFERED": "1",
            })
            print("$", " ".join(train_command), flush=True)
            wall_started = time.perf_counter()
            with TRAIN_LOG.open("w", encoding="utf-8", buffering=1) as log_file:
                process = subprocess.Popen(
                    train_command,
                    cwd=PROJECT_ROOT,
                    env=training_environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log_file.write(line)
                return_code = process.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, train_command)
            training_wall_seconds = time.perf_counter() - wall_started
            """
        ),
        markdown("## Completion marker (local Kaggle output only)"),
        code(
            """
            report_path = OUTPUT_DIR / "training_report.json"
            if not report_path.is_file():
                raise RuntimeError("Training finished without training_report.json")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report["original_training_examples"] != EXPECTED_DATA["pairs_rows"]:
                raise RuntimeError("Trainer did not receive every human pair")
            if report["validation_splits"] or report["validation_examples"] != 0:
                raise RuntimeError("Final run unexpectedly evaluated validation data")
            args = report["args"]
            required = LOCKED_RECIPE
            differences = {
                key: {"expected": value, "actual": args.get(key)}
                for key, value in required.items()
                if args.get(key) != value
            }
            if differences:
                raise RuntimeError(f"Locked final recipe changed: {differences}")
            if report["world_size"] != EXPECTED_WORLD_SIZE:
                raise RuntimeError(
                    f"Expected {EXPECTED_WORLD_SIZE} GPUs, got {report['world_size']}"
                )
            effective_batch_size = (
                args["batch_size"]
                * report["world_size"]
                * args["gradient_accumulation"]
            )
            if effective_batch_size != EXPECTED_EFFECTIVE_BATCH_SIZE:
                raise RuntimeError(
                    "Unexpected effective batch size: "
                    f"{effective_batch_size} != {EXPECTED_EFFECTIVE_BATCH_SIZE}"
                )
            if EXPECTED_AMP_DTYPE and report["amp_dtype"] != EXPECTED_AMP_DTYPE:
                raise RuntimeError(
                    f"Expected {EXPECTED_AMP_DTYPE}, got {report['amp_dtype']}"
                )
            if report["loss_hook"]["path"] is not None:
                raise RuntimeError("Final run must use the built-in BCE loss")
            if not (
                report["training_loss_weight_min"]
                == report["training_loss_weight_median"]
                == report["training_loss_weight_max"]
                == 1.0
            ):
                raise RuntimeError("Final run unexpectedly applied sample weights")
            completion = {
                "status": "complete",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ).replace("+00:00", "Z"),
                "dataset_ref": EXPECTED_DATASET_REF,
                "checkpoint_dataset_ref": EXPECTED_CHECKPOINT_DATASET_REF,
                "source_sha256": EXPECTED_SOURCE_SHA256,
                "training_wall_seconds": training_wall_seconds,
                "preparation_report": preparation_report,
                "training_report": report,
            }
            completion_path = WORKING_ROOT / "notebook_completed.json"
            completion_path.write_text(
                json.dumps(completion, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            print(json.dumps({
                "status": "complete",
                "model_dir": str(OUTPUT_DIR),
                "train_pairs": report["original_training_examples"],
                "epochs": args["epochs"],
                "training_hours": training_wall_seconds / 3600,
            }, ensure_ascii=False, indent=2))
            """
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "product_matching_final_training": {
                "data_dataset": data_dataset_ref,
                "checkpoint_dataset": checkpoint_dataset_ref,
                "source_sha256": source_hash,
                "data_audit": data_audit,
                "config": config,
                "expected_gpus": 2,
                "validation": False,
                "google_sheets": False,
            },
        }
    )
    return notebook


def write_notebook(
    notebook_path: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
    data_dataset_ref: str,
    checkpoint_dataset_ref: str = DEFAULT_CHECKPOINT_DATASET_REF,
) -> dict[str, object]:
    config = load_locked_config(config_path)
    data_audit = validate_local_human_data()
    notebook = build_notebook(
        config,
        data_audit,
        data_dataset_ref=data_dataset_ref,
        checkpoint_dataset_ref=checkpoint_dataset_ref,
    )
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, notebook_path)
    return data_audit
