#!/usr/bin/env python3
"""Build the locked MiniLM 5ep team template for data/loss ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

import create_cross_encoder_training_notebook as cross_builder
import create_minilm_validation_baseline_notebook as baseline_builder


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "notebooks"
    / "minilm_5ep_team_ablation"
    / "minilm_5ep_team_ablation_2xt4.ipynb"
)
DEFAULT_CONFIG = (
    ROOT / "configs" / "cross_encoder_minilm_llm_pretrain_5ep_human_ft.json"
)
BASELINE_CONFIG = ROOT / "configs" / "cross_encoder_minilm_validation_baseline.json"
DATASET_OWNER = "alexproger23"
CHECKPOINT_DATASET = "alexproger23/product-matching-minilm-llm-pretrain-5ep"
CHECKPOINT_MANIFEST_SHA256 = (
    "354c7006898a9a44a3115c8384f12dbab520cfec7723a675f8ccedb108544533"
)
EXPERIMENT_NAME = "minilm_5ep_team_data_loss_ablation"


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def markdown(value: str, *tags: str) -> nbf.NotebookNode:
    cell = nbf.v4.new_markdown_cell(dedent(value).strip())
    cell.metadata["tags"] = list(tags)
    return cell


def code(value: str, *tags: str) -> nbf.NotebookNode:
    cell = nbf.v4.new_code_cell(dedent(value).strip())
    cell.metadata["tags"] = list(tags)
    return cell


def _heading_index(notebook: nbf.NotebookNode, heading: str) -> int:
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown" and cell.source.strip() == heading:
            return index
    raise ValueError(f"Notebook heading is missing: {heading}")


def assert_frozen_recipe(config: dict[str, object]) -> None:
    baseline = cross_builder.load_training_config(BASELINE_CONFIG)
    differences = {
        key: {"baseline": baseline.get(key), "experiment": config.get(key)}
        for key in sorted(set(baseline) | set(config))
        if key != "model" and baseline.get(key) != config.get(key)
    }
    if differences:
        raise ValueError(
            "5ep human fine-tune recipe differs from the frozen baseline: "
            + json.dumps(differences, ensure_ascii=False, sort_keys=True)
        )


def build_team_notebook(
    dataset: dict[str, object],
    config: dict[str, object],
) -> nbf.NotebookNode:
    assert_frozen_recipe(config)
    recipe_hash = canonical_sha256(config)
    checkpoint = {
        "dataset": CHECKPOINT_DATASET,
        "manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
    }
    notebook = baseline_builder.build_notebook(
        dataset,
        config,
        experiment_name=EXPERIMENT_NAME,
        experiment_title="MiniLM 5ep: locked data/loss ablation template",
        experiment_description=(
            "Командный шаблон начинает с frozen MiniLM checkpoint после пяти эпох "
            "LLM pretraining. Оптимизатор, LR, одна эпоха human fine-tune, batch "
            "size, scheduler, сериализация и IID/hard/OOD validation зафиксированы. "
            "Разрешено менять только две помеченные ячейки: train-данные и loss."
        ),
        initial_checkpoint=checkpoint,
    )

    for cell in notebook.cells:
        cell.metadata["tags"] = ["frozen"]

    config_heading = _heading_index(notebook, "## Базовый конфиг")
    notebook.cells[config_heading] = markdown("## 🔒 Frozen training recipe", "frozen")
    notebook.cells[config_heading + 1] = code(
        f"""
        CANONICAL_TRAIN_CONFIG = {config!r}
        EXPECTED_TRAIN_RECIPE_SHA256 = {recipe_hash!r}

        def canonical_json_sha256(value):
            payload = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            return hashlib.sha256(payload).hexdigest()

        actual_recipe_hash = canonical_json_sha256(CANONICAL_TRAIN_CONFIG)
        if actual_recipe_hash != EXPECTED_TRAIN_RECIPE_SHA256:
            raise RuntimeError(
                "Frozen training recipe was edited. Restore this cell; only DATA HOOK "
                "and LOSS HOOK may be changed."
            )
        TRAIN_CONFIG = dict(CANONICAL_TRAIN_CONFIG)
        if INITIAL_MODEL_PATH is not None:
            TRAIN_CONFIG["model"] = str(INITIAL_MODEL_PATH)
        RUNTIME_CONFIG_PATH.write_text(
            json.dumps(TRAIN_CONFIG, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({{
            "frozen_recipe_sha256": EXPECTED_TRAIN_RECIPE_SHA256,
            "training_config": TRAIN_CONFIG,
        }}, ensure_ascii=False, indent=2))
        """,
        "frozen",
        "frozen-recipe",
    )

    sources, _ = baseline_builder.embedded_sources()
    remote_files = repr(baseline_builder.REMOTE_FILES)
    code_heading = _heading_index(notebook, "## Код и frozen data")
    notebook.cells[code_heading] = markdown(
        "## 🔒 Embedded code and frozen validation", "frozen"
    )
    notebook.cells[code_heading + 1] = code(
        f"""
        EMBEDDED_SOURCES = {sources!r}
        PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
        for relative, content in EMBEDDED_SOURCES.items():
            destination = PROJECT_ROOT / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
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
        PREPARED_DIR.mkdir(parents=True, exist_ok=True)
        REMOTE_FILES = {remote_files}
        frozen_validation_names = {{
            "human/iid_validation_pairs.parquet": "iid_validation_pairs.parquet",
            "human/hard_validation_pairs.parquet": "hard_validation_pairs.parquet",
            "human/ood_validation_pairs.parquet": "ood_validation_pairs.parquet",
        }}
        for relative, destination_name in frozen_validation_names.items():
            destination = PREPARED_DIR / destination_name
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            destination.symlink_to(attached_files[relative])
        """,
        "frozen",
    )

    insert_at = code_heading + 2
    editable_cells = [
        markdown(
            """
            ## ✏️ EDIT 1/2 — DATA HOOK

            Меняйте только тело `build_train_data`. Функция обязана вернуть
            `(train_pairs, items)`. Дополнительные Kaggle inputs доступны через
            `input_root`. Frozen validation item ID запрещены в train и проверяются
            следующей ячейкой.
            """,
            "team-editable",
            "data-hook",
        ),
        code(
            """
            def build_train_data(human_train_pairs, human_items, input_root):
                # Return experiment train pairs and the item catalogue they reference.
                train_pairs = human_train_pairs.copy()
                items = human_items.copy()

                # Пример подключения дополнительных данных:
                # extra = pd.read_parquet(
                #     next(input_root.glob("**/my_extra_train_pairs.parquet"))
                # )
                # train_pairs = pd.concat([train_pairs, extra], ignore_index=True)

                return train_pairs, items
            """,
            "team-editable",
            "data-hook",
        ),
        markdown("## 🔒 Data validation and materialization", "frozen"),
        code(
            """
            import numpy as np
            import pandas as pd

            human_train_pairs = pd.read_parquet(
                attached_files["human/train_pairs.parquet"]
            )
            human_items = pd.read_parquet(attached_files["human/items.parquet"])
            hook_result = build_train_data(
                human_train_pairs.copy(), human_items.copy(), INPUT_ROOT
            )
            if not isinstance(hook_result, tuple) or len(hook_result) != 2:
                raise TypeError("build_train_data must return (train_pairs, items)")
            train_pairs, train_items = hook_result
            if not isinstance(train_pairs, pd.DataFrame) or not isinstance(train_items, pd.DataFrame):
                raise TypeError("Both build_train_data outputs must be pandas DataFrames")

            required_pair_columns = {"id1", "id2", "target"}
            required_item_columns = {"id", "product_text", "category"}
            if missing := required_pair_columns - set(train_pairs):
                raise ValueError(f"Train pairs are missing columns: {sorted(missing)}")
            if missing := required_item_columns - set(train_items):
                raise ValueError(f"Items are missing columns: {sorted(missing)}")
            if train_pairs.empty:
                raise ValueError("Data hook returned an empty train set")
            if train_items["id"].duplicated().any():
                raise ValueError("Items contain duplicate id values")
            if train_pairs[["id1", "id2", "target"]].isna().any().any():
                raise ValueError("Train id/target columns contain nulls")
            if (train_pairs["id1"] == train_pairs["id2"]).any():
                raise ValueError("Train contains self-pairs")
            if not train_pairs["target"].between(0, 1).all():
                raise ValueError("Train targets must be probabilities in [0, 1]")
            if "sample_weight" in train_pairs:
                sample_weight = train_pairs["sample_weight"].to_numpy(dtype=np.float64)
                if not np.isfinite(sample_weight).all() or (sample_weight <= 0).any():
                    raise ValueError("sample_weight must be finite and positive")

            unordered = pd.DataFrame({
                "left": np.minimum(train_pairs["id1"].to_numpy(), train_pairs["id2"].to_numpy()),
                "right": np.maximum(train_pairs["id1"].to_numpy(), train_pairs["id2"].to_numpy()),
            })
            if unordered.duplicated().any():
                raise ValueError("Train contains duplicate unordered pairs")

            validation_pair_frames = [
                pd.read_parquet(attached_files[relative], columns=["id1", "id2"])
                for relative in (
                    "human/iid_validation_pairs.parquet",
                    "human/hard_validation_pairs.parquet",
                    "human/ood_validation_pairs.parquet",
                )
            ]
            validation_ids = set(
                pd.concat(
                    [frame[["id1", "id2"]].stack() for frame in validation_pair_frames],
                    ignore_index=True,
                ).tolist()
            )
            train_ids = set(
                pd.concat(
                    [train_pairs["id1"], train_pairs["id2"]], ignore_index=True
                ).tolist()
            )
            if leaked_ids := train_ids & validation_ids:
                raise ValueError(
                    f"Train leaks {len(leaked_ids)} frozen validation item IDs"
                )
            available_ids = set(train_items["id"].tolist())
            required_ids = train_ids | validation_ids
            if missing_ids := required_ids - available_ids:
                raise ValueError(f"Items are missing {len(missing_ids)} required IDs")

            reference_validation_items = (
                human_items[human_items["id"].isin(validation_ids)]
                .set_index("id")[["product_text", "category"]]
                .sort_index()
            )
            candidate_validation_items = (
                train_items[train_items["id"].isin(validation_ids)]
                .set_index("id")[["product_text", "category"]]
                .sort_index()
            )
            if not candidate_validation_items.equals(reference_validation_items):
                raise ValueError("Frozen validation item text/category was changed")

            train_pairs = train_pairs.reset_index(drop=True)
            train_items = train_items.reset_index(drop=True)
            train_pairs_path = PREPARED_DIR / "train_pairs.parquet"
            items_path = PREPARED_DIR / "items.parquet"
            train_pairs.to_parquet(train_pairs_path, index=False, compression="zstd")
            train_items.to_parquet(items_path, index=False, compression="zstd")
            TRAIN_DATA_REPORT = {
                "train_pairs": len(train_pairs),
                "items": len(train_items),
                "positive_rate": float(train_pairs["target"].mean()),
                "same_size_as_human_baseline": len(train_pairs) == len(human_train_pairs),
                "train_pairs_sha256": file_sha256(train_pairs_path),
                "items_sha256": file_sha256(items_path),
                "label_source_counts": (
                    {
                        str(key): int(value)
                        for key, value in train_pairs["label_source"].value_counts().items()
                    }
                    if "label_source" in train_pairs
                    else {"unspecified": len(train_pairs)}
                ),
            }
            print(json.dumps(TRAIN_DATA_REPORT, ensure_ascii=False, indent=2))
            """,
            "frozen",
            "data-guard",
        ),
        markdown(
            """
            ## ✏️ EDIT 2/2 — LOSS HOOK

            Меняйте только `initialize_loss` и `compute_loss`. `compute_loss`
            должен вернуть scalar tensor либо словарь с обязательным ключом
            `loss`. Остальные scalar-значения попадут в training log.
            """,
            "team-editable",
            "loss-hook",
        ),
        code(
            """
            %%writefile /kaggle/working/product_matching/team_loss_hook.py
            from __future__ import annotations

            import torch
            import torch.nn.functional as F


            def initialize_loss(*, train_frame, device, rank, world_size):
                # Optional one-time initialization on every DDP rank.
                return None


            def compute_loss(
                *,
                logits,
                targets,
                sample_weights,
                pair_indices,
                orientations,
                epoch,
                step,
            ):
                per_example_bce = F.binary_cross_entropy_with_logits(
                    logits.float(), targets, reduction="none"
                )
                bce = (per_example_bce * sample_weights).sum() / sample_weights.sum()
                return {
                    "loss": bce,
                    "bce": bce.detach(),
                }
            """,
            "team-editable",
            "loss-hook",
        ),
        markdown("## 🔒 Final protocol guard", "frozen"),
        code(
            """
            if canonical_json_sha256(CANONICAL_TRAIN_CONFIG) != EXPECTED_TRAIN_RECIPE_SHA256:
                raise RuntimeError("Frozen training recipe changed after initialization")
            if json.loads(RUNTIME_CONFIG_PATH.read_text(encoding="utf-8")) != TRAIN_CONFIG:
                raise RuntimeError("Runtime training config differs from the frozen recipe")
            LOSS_HOOK_PATH = PROJECT_ROOT / "team_loss_hook.py"
            if not LOSS_HOOK_PATH.is_file():
                raise RuntimeError("LOSS HOOK cell did not create team_loss_hook.py")
            LOSS_HOOK_SHA256 = file_sha256(LOSS_HOOK_PATH)
            print(json.dumps({
                "frozen_recipe_sha256": EXPECTED_TRAIN_RECIPE_SHA256,
                "loss_hook_sha256": LOSS_HOOK_SHA256,
                "train_data": TRAIN_DATA_REPORT,
            }, ensure_ascii=False, indent=2))
            """,
            "frozen",
            "protocol-guard",
        ),
    ]
    notebook.cells[insert_at:insert_at] = editable_cells

    train_heading = _heading_index(notebook, "## Обучение и три validation-протокола")
    notebook.cells[train_heading] = markdown(
        "## 🔒 Training and IID/hard/OOD validation", "frozen"
    )
    train_cell = notebook.cells[train_heading + 1]
    needle = '"--token-cache-dir", str(TOKEN_CACHE_DIR),'
    replacement = (
        '"--token-cache-dir", str(TOKEN_CACHE_DIR),\n'
        '        "--loss-hook", str(LOSS_HOOK_PATH),'
    )
    if needle not in train_cell.source:
        raise ValueError("Could not add loss hook to training command")
    train_cell.source = train_cell.source.replace(needle, replacement)
    train_cell.metadata["tags"] = ["frozen"]

    artifacts_heading = _heading_index(notebook, "## Артефакты и completion report")
    notebook.cells[artifacts_heading] = markdown(
        "## 🔒 Artifacts and completion report", "frozen"
    )
    completion_cell = notebook.cells[artifacts_heading + 1]
    needle = '"code_bundle_sha256": EMBEDDED_SOURCE_SHA256,'
    replacement = dedent(
        """
        "code_bundle_sha256": EMBEDDED_SOURCE_SHA256,
            "frozen_recipe_sha256": EXPECTED_TRAIN_RECIPE_SHA256,
            "loss_hook_sha256": LOSS_HOOK_SHA256,
            "train_data": TRAIN_DATA_REPORT,
        """
    ).strip()
    if needle not in completion_cell.source:
        raise ValueError("Could not add team provenance to completion report")
    completion_cell.source = completion_cell.source.replace(needle, replacement)
    completion_cell.metadata["tags"] = ["frozen"]

    notebook.metadata["product_matching_training"].update(
        {
            "template": "minilm_5ep_team_data_loss_ablation_v1",
            "frozen_recipe_sha256": recipe_hash,
            "editable_cells": ["data-hook", "loss-hook"],
        }
    )
    nbf.validate(notebook)
    return notebook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=baseline_builder.DEFAULT_SOURCE_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = baseline_builder.load_manifest(args.source_dir, DATASET_OWNER)
    config = cross_builder.load_training_config(args.config)
    notebook = build_team_notebook(dataset, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, args.output)
    print(f"Wrote notebook: {args.output}")
    print(f"Frozen recipe SHA-256: {canonical_sha256(config)}")
    print(f"Validation Dataset: {dataset['dataset']}")
    print(f"Checkpoint Dataset: {CHECKPOINT_DATASET}")


if __name__ == "__main__":
    main()
