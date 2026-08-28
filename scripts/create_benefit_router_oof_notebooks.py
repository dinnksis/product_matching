#!/usr/bin/env python3
"""Generate 5-fold component-disjoint neural OOF notebooks for benefit routing."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import nbformat as nbf

import create_architecture_baseline_notebooks as architecture
import create_qwen_training_notebook as shared


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "notebooks" / "benefit_router_oof"
PROFILES = ("bge-v2-m3", "minilm-5ep", "rumodernbert")
NOTEBOOKS = {
    "bge-v2-m3": "bge_benefit_router_oof_2xt4.ipynb",
    "minilm-5ep": "minilm_benefit_router_oof_2xt4.ipynb",
    "rumodernbert": "rumodernbert_benefit_router_oof_2xt4.ipynb",
}
EXPERIMENTS = {
    "bge-v2-m3": "bge_benefit_router_oof_v1",
    "minilm-5ep": "minilm_benefit_router_oof_v1",
    "rumodernbert": "rumodernbert_benefit_router_oof_v1",
}
N_SPLITS = 5
FOLD_SEED = 2026


def heading_index(notebook: nbf.NotebookNode, heading: str) -> int:
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown" and cell.source.strip().splitlines()[0] == heading:
            return index
    raise ValueError(f"Notebook heading is missing: {heading}")


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip())


def build(profile_name: str) -> nbf.NotebookNode:
    configuration = architecture.load_configuration()
    protocol = configuration["protocol"]
    profile = dict(configuration["profiles"][profile_name])
    profile["experiment"] = EXPERIMENTS[profile_name]
    notebook = architecture.build_notebook(
        architecture.load_validation_dataset(), protocol, profile_name, profile
    )

    training_heading = heading_index(notebook, "## Обучение и три validation-протокола")
    notebook.cells[training_heading].source = "## Пять component-disjoint OOF folds"
    notebook.cells[training_heading + 1] = code(
        f'''
import gc
import shutil
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold


def stable_component_ids(pairs):
    """Stable minimum-item connected-component id without optional EDA imports."""
    all_ids = pd.unique(pairs[["id1", "id2"]].to_numpy().reshape(-1))
    positions = pd.Series(np.arange(len(all_ids), dtype=np.int64), index=all_ids)
    left = positions.loc[pairs["id1"]].to_numpy(dtype=np.int64)
    right = positions.loc[pairs["id2"]].to_numpy(dtype=np.int64)
    parent = np.arange(len(all_ids), dtype=np.int64)
    size = np.ones(len(all_ids), dtype=np.int64)

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for first, second in zip(left, right):
        root1, root2 = find(int(first)), find(int(second))
        if root1 == root2:
            continue
        if size[root1] < size[root2]:
            root1, root2 = root2, root1
        parent[root2] = root1
        size[root1] += size[root2]
    roots = np.fromiter(
        (find(index) for index in range(len(all_ids))),
        dtype=np.int64,
        count=len(all_ids),
    )
    component_minimum = np.full(len(all_ids), np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(component_minimum, roots, np.asarray(all_ids, dtype=np.int64))
    return component_minimum[roots[left]]

N_SPLITS = {N_SPLITS}
FOLD_SEED = {FOLD_SEED}
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
all_pairs = pair_frames["train"].reset_index(drop=True).copy()
all_pairs["oof_row_index"] = np.arange(len(all_pairs), dtype=np.int64)
item_category = prepared_items.set_index("id", verify_integrity=True)["category"]
categories = item_category.loc[all_pairs["id1"].to_numpy()].astype(str).to_numpy()
components = stable_component_ids(all_pairs)
strata = pd.Series(categories) + "||" + all_pairs["target"].astype(int).astype(str)
splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=FOLD_SEED)
folds = np.full(len(all_pairs), -1, dtype=np.int8)
dummy = np.zeros(len(all_pairs), dtype=np.int8)
for fold, (_, valid) in enumerate(splitter.split(dummy, strata, groups=components)):
    folds[valid] = fold
if np.any(folds < 0):
    raise RuntimeError("Some human-train rows have no OOF fold")
fold_check = pd.DataFrame({{"component": components, "fold": folds}})
if int(fold_check.groupby("component")["fold"].nunique().max()) != 1:
    raise RuntimeError("Component leakage across neural OOF folds")

assignment = all_pairs[["id1", "id2", "target", "oof_row_index"]].copy()
assignment["category"] = categories
assignment["component_id"] = components
assignment["fold"] = folds
assignment.to_parquet(OUTPUT_DIR / "oof_fold_assignments.parquet", index=False, compression="zstd")

training_environment = os.environ.copy()
training_environment.update({{
    "OMP_NUM_THREADS": "2",
    "TOKENIZERS_PARALLELISM": "true",
    "NCCL_DEBUG": "WARN",
    "PYTHONUNBUFFERED": "1",
}})
fold_predictions = []
fold_reports = []
training_started = time.perf_counter()
for fold in range(N_SPLITS):
    fold_root = TEMP_ROOT / f"fold_{{fold}}"
    fold_prepared = fold_root / "prepared"
    fold_output = fold_root / "model"
    fold_cache = fold_root / "token_cache"
    fold_prepared.mkdir(parents=True, exist_ok=True)
    items_link = fold_prepared / "items.parquet"
    if items_link.exists() or items_link.is_symlink():
        items_link.unlink()
    items_link.symlink_to(PREPARED_DIR / "items.parquet")
    train_mask = folds != fold
    held_mask = folds == fold
    all_pairs.loc[train_mask, ["id1", "id2", "target"]].to_parquet(
        fold_prepared / "train_pairs.parquet", index=False, compression="zstd"
    )
    all_pairs.loc[held_mask, ["id1", "id2", "target"]].to_parquet(
        fold_prepared / "oof_validation_pairs.parquet", index=False, compression="zstd"
    )
    fold_config = dict(TRAIN_CONFIG)
    fold_config["seed"] = int(TRAIN_CONFIG["seed"]) + fold
    fold_config_path = fold_root / "training_config.json"
    fold_config_path.write_text(json.dumps(fold_config, ensure_ascii=False, indent=2), encoding="utf-8")
    fold_log = WORKING_ROOT / f"{{OUTPUT_DIR.name}}_fold{{fold}}.log"
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
        str(PROJECT_ROOT / "scripts/train_cross_encoder.py"),
        "--config", str(fold_config_path),
        "--prepared-dir", str(fold_prepared),
        "--output-dir", str(fold_output),
        "--token-cache-dir", str(fold_cache),
        "--validation-split", "oof=oof_validation_pairs.parquet",
    ]
    print("$", " ".join(command), flush=True)
    fold_started = time.perf_counter()
    with fold_log.open("w", encoding="utf-8", buffering=1) as log_file:
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, env=training_environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    report = json.loads((fold_output / "training_report.json").read_text(encoding="utf-8"))
    prediction_path = fold_output / report["validation_splits"]["oof"]["predictions_file"]
    prediction = pd.read_parquet(prediction_path)
    expected = assignment.loc[held_mask].sort_values("oof_row_index", kind="stable")
    prediction = prediction.merge(
        expected[["id1", "id2", "target", "oof_row_index", "component_id", "fold"]]
        .rename(columns={{"target": "expected_target"}}),
        on=["id1", "id2"], how="inner", validate="one_to_one", sort=False,
    ).sort_values("oof_row_index", kind="stable")
    if len(prediction) != int(held_mask.sum()):
        raise RuntimeError(f"Fold {{fold}} OOF pair set differs from assignment")
    if not np.array_equal(
        prediction["target"].astype(np.int8).to_numpy(),
        prediction.pop("expected_target").astype(np.int8).to_numpy(),
    ):
        raise RuntimeError(f"Fold {{fold}} targets differ from the frozen assignment")
    slim_columns = [
        "id1", "id2", "target", "category_1", "oof_row_index", "component_id", "fold",
        "score", "score_ab", "score_ba", "logit", "score_order_gap",
        "token_length_ab", "token_length_ba",
    ]
    prediction = prediction[slim_columns].rename(columns={{"category_1": "category"}})
    prediction.to_parquet(
        OUTPUT_DIR / f"fold_{{fold}}_oof_predictions.parquet",
        index=False, compression="zstd",
    )
    fold_predictions.append(prediction)
    fold_reports.append({{
        "fold": fold,
        "train_rows": int(train_mask.sum()),
        "held_rows": int(held_mask.sum()),
        "wall_seconds": time.perf_counter() - fold_started,
        "training_seconds": report.get("training_seconds"),
        "validation_seconds": report.get("validation_seconds"),
        "macro_average_precision": report["validation_splits"]["oof"]["macro_average_precision"],
    }})
    shutil.rmtree(fold_root, ignore_errors=True)
    gc.collect()

oof = pd.concat(fold_predictions, ignore_index=True).sort_values("oof_row_index", kind="stable")
if len(oof) != len(all_pairs) or oof["oof_row_index"].duplicated().any():
    raise RuntimeError("Neural OOF predictions are incomplete or duplicated")
if not np.array_equal(oof["oof_row_index"].to_numpy(), np.arange(len(all_pairs))):
    raise RuntimeError("Neural OOF row order is incomplete")
oof.to_parquet(OUTPUT_DIR / "oof_predictions.parquet", index=False, compression="zstd")
metric_frame = oof[["target", "score", "category"]].copy()
per_category_ap = metric_frame.groupby("category", sort=True).apply(
    lambda part: average_precision_score(part["target"], part["score"])
)
training_wall_seconds = time.perf_counter() - training_started
OOF_REPORT = {{
    "model": TRAIN_CONFIG["model"],
    "folds": N_SPLITS,
    "fold_seed": FOLD_SEED,
    "component_disjoint": True,
    "human_train_only": True,
    "rows": len(oof),
    "macro_average_precision": float(per_category_ap.mean()),
    "overall_average_precision": float(average_precision_score(oof["target"], oof["score"])),
    "per_category_average_precision": {{str(k): float(v) for k, v in per_category_ap.items()}},
    "fold_reports": fold_reports,
    "training_wall_seconds": training_wall_seconds,
    "oof_predictions_file": "oof_predictions.parquet",
}}
(OUTPUT_DIR / "training_report.json").write_text(
    json.dumps(OOF_REPORT, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(OOF_REPORT, ensure_ascii=False, indent=2))
'''
    )

    artifacts_heading = heading_index(notebook, "## Артефакты и completion report")
    notebook.cells[artifacts_heading + 1] = code(
        f'''
report = json.loads((OUTPUT_DIR / "training_report.json").read_text(encoding="utf-8"))
completion = {{
    "status": "complete",
    "run_id": EXPERIMENT_RUN_ID,
    "started_at_utc": EXPERIMENT_STARTED_AT_UTC,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    "experiment": OUTPUT_DIR.name,
    "experiment_group": "data",
    "architecture": {str(profile['architecture'])!r},
    "model": {str(profile['model'])!r},
    "dataset_ref": EXPECTED_DATASET_REF,
    "initial_checkpoint_ref": INITIAL_CHECKPOINT_REF or {str(profile['model'])!r},
    "kaggle_kernel_ref": os.getenv("KAGGLE_KERNEL_RUN_ID") or os.getenv("KAGGLE_KERNEL_INFERENCE_RUN_ID") or "",
    "code_bundle_sha256": EMBEDDED_SOURCE_SHA256,
    "serialization": SERIALIZATION_VARIANT,
    "training_wall_seconds": training_wall_seconds,
    "training_report": report,
    "artifacts": {{
        "oof_predictions": f"{{OUTPUT_DIR.name}}/oof_predictions.parquet",
        "fold_assignments": f"{{OUTPUT_DIR.name}}/oof_fold_assignments.parquet",
        "training_report": f"{{OUTPUT_DIR.name}}/training_report.json",
    }},
}}
(WORKING_ROOT / "notebook_completed.json").write_text(
    json.dumps(completion, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
)
print(json.dumps({{
    "rows": report["rows"],
    "folds": report["folds"],
    "macro_oof_ap": report["macro_average_precision"],
}}, ensure_ascii=False, indent=2))
'''
    )
    notebook.cells[-2:] = shared.google_sheets_tracking_cells(sync_target="default")
    for index, cell in enumerate(notebook.cells):
        cell.metadata.setdefault("tags", ["benefit-router-neural-oof"])
        cell.id = hashlib.sha256(
            f"benefit-router-oof-v1:{profile_name}:{index}".encode("utf-8")
        ).hexdigest()[:12]
    notebook.metadata["product_matching_training"].update(
        {
            "template": "benefit_router_neural_oof_v1",
            "profile": profile_name,
            "folds": N_SPLITS,
            "fold_seed": FOLD_SEED,
            "validation_labels_used": False,
            "output": "oof_predictions.parquet",
        }
    )
    nbf.validate(notebook)
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=["all", *PROFILES], nargs="?", default="all")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = PROFILES if args.profile == "all" else (args.profile,)
    for profile_name in selected:
        destination = args.output_dir / NOTEBOOKS[profile_name]
        nbf.write(build(profile_name), destination)
        print(f"Wrote notebook: {destination}")


if __name__ == "__main__":
    main()
