#!/usr/bin/env python3
"""Build a Kaggle notebook that validates the exact submission runner."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "qwen_embedding_submission_diagnostic.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def main() -> None:
    runner_source = (ROOT / "submits" / "qwen-embedding-catboost" / "run.py").read_text(
        encoding="utf-8"
    )
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Qwen embedding submission diagnostic\n\n"
            "Runs the exact submission preprocessing on the saved component-disjoint validation "
            "and compares it with the original Kaggle predictions."
        ),
        code(
            """
import subprocess, sys
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "catboost==1.2.8", "rapidfuzz==3.14.1", "transformers==4.57.6"
], check=True)
"""
        ),
        code(
            """
from pathlib import Path
import json, os, shutil, subprocess, sys, time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working/submission_diagnostic")
WORK.mkdir(parents=True, exist_ok=True)

def one(candidates, label):
    candidates = list(candidates)
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one {label}, found {len(candidates)}: {candidates[:20]}")
    print(f"{label}: {candidates[0]}")
    return candidates[0]

items_path = one(INPUT.rglob("items_human.parquet"), "items")
human_matches_path = one(
    (p for p in INPUT.rglob("matches.parquet") if p.name == "matches.parquet"),
    "human matches",
)
saved_validation_path = one(
    (
        p for p in INPUT.rglob("validation_predictions.parquet")
        if p.parent.name == "03_names_qwen_attributes"
    ),
    "saved validation predictions",
)
catboost_path = one(
    (p for p in INPUT.rglob("model.cbm") if p.parent.name == "03_names_qwen_attributes"),
    "CatBoost model",
)
keys_path = one(
    (
        p for p in INPUT.rglob("selected_attribute_keys.json")
        if "embedding_boosting" in str(p)
    ),
    "selected attribute keys",
)
qwen_weights = one(
    (
        p for p in INPUT.rglob("model.safetensors")
        if p.parent.name == "qwen_embedding_model" and p.stat().st_size > 1_000_000_000
    ),
    "Qwen weights",
)
qwen_dir = qwen_weights.parent

print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("Qwen bytes:", qwen_weights.stat().st_size)
"""
        ),
        code(
            f"""
runner_source = {json.dumps(runner_source, ensure_ascii=False)}
runner_path = WORK / "run.py"
runner_path.write_text(runner_source, encoding="utf-8")
(WORK / "models").mkdir(exist_ok=True)
shutil.copy2(catboost_path, WORK / "models" / "matching_model.cbm")
shutil.copy2(keys_path, WORK / "selected_attribute_keys.json")

truth = pd.read_parquet(saved_validation_path)
required = {{"id1", "id2", "target", "category", "predict"}}
if not required.issubset(truth.columns):
    raise RuntimeError(f"Saved validation lacks columns: {{required - set(truth.columns)}}")
matches_path = WORK / "validation_pairs.parquet"
truth[["id1", "id2"]].to_parquet(matches_path, index=False)
print("Validation pairs:", len(truth), "positive rate:", float(truth.target.mean()))
"""
        ),
        code(
            """
output_path = WORK / "submission_predictions.csv"
env = os.environ.copy()
env.update({
    "PM_MODEL_DIR": str(qwen_dir),
    "PM_BATCH_SIZE": "256",
    "PM_PUBLIC_SOFT_LIMIT_SECONDS": "7200",
    "PM_PRIVATE_SOFT_LIMIT_SECONDS": "7200",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
})
command = [
    sys.executable, "-u", str(runner_path),
    "--items_path", str(items_path),
    "--matches_path", str(matches_path),
    "--output_path", str(output_path),
]
started = time.perf_counter()
result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT)
elapsed = time.perf_counter() - started
print(result.stdout)
(WORK / "diagnostic.log").write_text(result.stdout, encoding="utf-8")
if result.returncode:
    raise RuntimeError(f"Submission runner failed with code {result.returncode}")
"""
        ),
        code(
            """
submitted = pd.read_csv(output_path)
if not np.array_equal(submitted[["id1", "id2"]].to_numpy(), truth[["id1", "id2"]].to_numpy()):
    raise RuntimeError("Submission changed validation pair order")

def macro_ap(frame, score_column):
    per_category = {
        str(category): float(average_precision_score(group.target, group[score_column]))
        for category, group in frame.groupby("category", sort=True)
    }
    return float(np.mean(list(per_category.values()))), per_category

comparison = truth[["id1", "id2", "target", "category", "predict"]].rename(
    columns={"predict": "training_pipeline_predict"}
)
comparison["submission_pipeline_predict"] = submitted.predict.to_numpy()
submission_macro, per_category = macro_ap(comparison, "submission_pipeline_predict")
training_macro, _ = macro_ap(comparison, "training_pipeline_predict")
overall = float(average_precision_score(comparison.target, comparison.submission_pipeline_predict))
pearson = float(comparison[["training_pipeline_predict", "submission_pipeline_predict"]].corr().iloc[0, 1])
spearman = float(comparison[["training_pipeline_predict", "submission_pipeline_predict"]].corr(method="spearman").iloc[0, 1])
mae = float(np.mean(np.abs(comparison.training_pipeline_predict - comparison.submission_pipeline_predict)))

report = {
    "pairs": len(comparison),
    "required_items": int(pd.unique(comparison[["id1", "id2"]].to_numpy().reshape(-1)).size),
    "submission_macro_average_precision": submission_macro,
    "training_pipeline_macro_average_precision": training_macro,
    "submission_overall_average_precision": overall,
    "prediction_pearson": pearson,
    "prediction_spearman": spearman,
    "prediction_mae": mae,
    "total_runner_seconds": elapsed,
    "per_category_average_precision": per_category,
    "runner_log": result.stdout.splitlines(),
    "qwen_model_directory": str(qwen_dir),
    "qwen_weights_bytes": qwen_weights.stat().st_size,
}
comparison.to_parquet(WORK / "prediction_comparison.parquet", index=False)
(WORK / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
(WORK / "COMPLETED").write_text("ok\\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
"""
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    main()
