from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "notebooks/minilm_5ep_team_ablation/minilm_5ep_team_ablation_2xt4.ipynb"
OUT = BASE.parent


def source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def clone() -> dict:
    return json.loads(BASE.read_text(encoding="utf-8"))


RANDOM_SWAP = r'''def build_train_data(human_train_pairs, human_items, input_root):
    # Fixed-seed 50% row swap. The sampler still keeps the original random
    # orientation mechanism; this makes the augmentation explicit and keeps
    # the unordered train pairs and all frozen data checks unchanged.
    import numpy as np

    train_pairs = human_train_pairs.copy().reset_index(drop=True)
    rng = np.random.default_rng(42)
    swap = rng.random(len(train_pairs)) < 0.5
    left = train_pairs.loc[swap, "id1"].to_numpy(copy=True)
    train_pairs.loc[swap, "id1"] = train_pairs.loc[swap, "id2"].to_numpy()
    train_pairs.loc[swap, "id2"] = left
    return train_pairs, human_items.copy()
'''


SYMMETRY_PATCH = r'''# Patch only the training loop to add a bounded second forward pass.
# The frozen optimizer, scheduler, batch size, sampler, seed and validation
# protocol remain unchanged. BCE is computed on the original orientation;
# MSE consistency is computed for a random 25% of each batch.
from pathlib import Path

script_path = PROJECT_ROOT / "scripts/train_cross_encoder.py"
script = script_path.read_text(encoding="utf-8")
needle = """                    raw_loss, loss_metrics = loss_hook.compute(\n                        logits=logits.float(),\n                        targets=targets,\n                        sample_weights=weights,\n                        pair_indices=pair_indices,\n                        orientations=orientations,\n                        epoch=epoch,\n                        step=step,\n                    )\n"""
replacement = """                    symmetry_mask = torch.rand(\n                        logits.shape[0], device=device\n                    ) < 0.25\n                    if symmetry_mask.any():\n                        selected_indices = pair_indices[symmetry_mask].detach().cpu().tolist()\n                        selected_orientations = orientations[symmetry_mask].detach().cpu().tolist()\n                        selected_targets = targets[symmetry_mask].detach().cpu().tolist()\n                        selected_weights = weights[symmetry_mask].detach().cpu().tolist()\n                        reverse_rows = []\n                        for index, orientation, target, weight in zip(\n                            selected_indices, selected_orientations, selected_targets, selected_weights\n                        ):\n                            reverse_rows.append({\n                                **train_cache.sequence(int(index), reverse=not bool(orientation)),\n                                \"target\": float(target),\n                                \"sample_weight\": float(weight),\n                                \"pair_index\": int(index),\n                                \"reverse\": not bool(orientation),\n                            })\n                        reverse_packed = collator(reverse_rows)\n                        reverse_batch = {\n                            key: value.to(device, non_blocking=True)\n                            for key, value in reverse_packed.items()\n                            if key not in {\"targets\", \"sample_weights\", \"pair_indices\", \"orientations\"}\n                        }\n                        with torch.autocast(device_type=\"cuda\", dtype=amp_dtype):\n                            reverse_logits = relevance_logits(\n                                training_model(**reverse_batch), args.model_backend\n                            ).float()\n                        symmetry_mse = (\n                            logits[symmetry_mask].float() - reverse_logits\n                        ).pow(2).mean()\n                    else:\n                        symmetry_mse = logits.float().sum() * 0.0\n                    raw_loss, loss_metrics = loss_hook.compute(\n                        logits=logits.float(),\n                        targets=targets,\n                        sample_weights=weights,\n                        pair_indices=pair_indices,\n                        orientations=orientations,\n                        epoch=epoch,\n                        step=step,\n                    )\n                    raw_loss = raw_loss + 0.1 * symmetry_mse\n                    loss_metrics[\"symmetry_mse\"] = symmetry_mse.detach()\n                    loss_metrics[\"symmetry_fraction\"] = symmetry_mask.float().mean().detach()\n"""
if needle not in script:
    raise RuntimeError("Could not find frozen training loss call for symmetry patch")
script_path.write_text(script.replace(needle, replacement, 1), encoding="utf-8")
print("Applied explicit symmetry patch: lambda_sym=0.1, fraction=0.25")
'''


DIAGNOSTIC = r'''import json
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(PROJECT_ROOT))

from src.cross_encoder_training import (
    CrossEncoderBatchCollator,
    CrossEncoderPairDataset,
    build_pair_token_cache,
)
from src.data_pipeline import attach_item_fields
from src.qwen_reranker import preferred_cuda_dtype
from src.qwen_training import FixedLengthBatchSampler

OUTPUT_DIR = WORKING_ROOT / "minilm_5ep_symmetry_diagnostic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
diagnostic_started = time.perf_counter()

# Prefer the already fine-tuned BCE baseline when it is attached with
# --kernel; fall back to the frozen five-epoch checkpoint otherwise.
baseline_configs = list(INPUT_ROOT.glob("**/minilm_5ep_team_data_loss_ablation/config.json"))
model_path = baseline_configs[0].parent if baseline_configs else INITIAL_MODEL_PATH
if model_path is None:
    raise RuntimeError("No baseline checkpoint was found")
print("Diagnostic model:", model_path)

items = pd.read_parquet(attached_files["human/items.parquet"])
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path, torch_dtype=None
).cuda().eval()
device = torch.device("cuda")
amp_dtype = preferred_cuda_dtype()

def ap_by_category(frame, score_column):
    values = []
    for _, group in frame.groupby("category_1", sort=True):
        if group["target"].nunique() == 2:
            values.append(average_precision_score(group["target"], group[score_column]))
    return float(np.mean(values))

def symmetry_stats(frame):
    delta = np.abs(frame["score_ab"].to_numpy() - frame["score_ba"].to_numpy())
    out = {
        "mean": float(np.mean(delta)),
        "median": float(np.median(delta)),
        "p90": float(np.quantile(delta, 0.90)),
        "p95": float(np.quantile(delta, 0.95)),
        "p99": float(np.quantile(delta, 0.99)),
        "max": float(np.max(delta)),
        "pearson": float(frame["score_ab"].corr(frame["score_ba"], method="pearson")),
        "spearman": float(frame["score_ab"].corr(frame["score_ba"], method="spearman")),
        "gt_0.05": float(np.mean(delta > 0.05)),
        "gt_0.10": float(np.mean(delta > 0.10)),
        "gt_0.20": float(np.mean(delta > 0.20)),
    }
    for name, mask in {
        "positive": frame["target"].to_numpy() >= 0.5,
        "negative": frame["target"].to_numpy() < 0.5,
    }.items():
        subset = frame.loc[mask]
        subset_delta = np.abs(subset["score_ab"].to_numpy() - subset["score_ba"].to_numpy())
        out[name] = {
            "mean": float(np.mean(subset_delta)),
            "median": float(np.median(subset_delta)),
            "p90": float(np.quantile(subset_delta, 0.90)),
            "p95": float(np.quantile(subset_delta, 0.95)),
            "p99": float(np.quantile(subset_delta, 0.99)),
            "max": float(np.max(subset_delta)),
            "gt_0.05": float(np.mean(subset_delta > 0.05)),
            "gt_0.10": float(np.mean(subset_delta > 0.10)),
            "gt_0.20": float(np.mean(subset_delta > 0.20)),
        }
    return out

all_splits = {}
for split in ("iid", "hard", "ood"):
    pairs = pd.read_parquet(attached_files[f"human/{split}_validation_pairs.parquet"])
    pairs = attach_item_fields(pairs, items, fields=("product_text", "category"))
    cache = build_pair_token_cache(
        pairs, tokenizer, TOKEN_CACHE_DIR, f"diagnostic_{split}",
        str(model_path), 384, tokenization_batch_size=512, log_every=50,
    )
    targets = pairs["target"].to_numpy(dtype=np.float32)
    dataset = CrossEncoderPairDataset(cache, targets)
    sampler = FixedLengthBatchSampler(cache, np.arange(len(pairs)), 192, both_orientations=True)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=CrossEncoderBatchCollator(tokenizer.pad_token_id), num_workers=0)
    local = {}
    with torch.inference_mode():
        for packed in loader:
            indices = packed.pop("pair_indices").numpy()
            orientations = packed.pop("orientations").numpy().astype(bool)
            packed.pop("targets"); packed.pop("sample_weights")
            batch = {key: value.to(device, non_blocking=True) for key, value in packed.items()}
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(**batch).logits[:, 0]
            for index, reverse, score in zip(indices, orientations, logits.float().sigmoid().cpu().numpy()):
                local[(int(index), bool(reverse))] = float(score)
    result = pairs[["id1", "id2", "target", "category_1", "category_2", "product_text_1", "product_text_2"]].copy()
    result["score_ab"] = [local[(i, False)] for i in range(len(pairs))]
    result["score_ba"] = [local[(i, True)] for i in range(len(pairs))]
    result["score"] = (result["score_ab"] + result["score_ba"]) / 2.0
    result["symmetry_error"] = np.abs(result["score_ab"] - result["score_ba"])
    prediction_name = f"{split}_symmetry_predictions.parquet"
    result.to_parquet(OUTPUT_DIR / prediction_name, index=False, compression="zstd")
    metrics = symmetry_stats(result)
    metrics.update({
        "examples": int(len(result)),
        "macro_ap_ab": ap_by_category(result, "score_ab"),
        "macro_ap_ba": ap_by_category(result, "score_ba"),
        "macro_ap_average": ap_by_category(result, "score"),
        "overall_ap_ab": float(average_precision_score(result["target"], result["score_ab"])),
        "overall_ap_ba": float(average_precision_score(result["target"], result["score_ba"])),
        "overall_ap_average": float(average_precision_score(result["target"], result["score"])),
        "predictions_file": prediction_name,
    })
    all_splits[split] = metrics

report = {
    "training_seconds": 0.0,
    "validation_seconds": time.perf_counter() - diagnostic_started,
    "training_examples": 0,
    "training_subset": "none",
    "training_sampling": "none",
    "training_loss_weighting": "none",
    "primary_validation_split": "iid",
    "validation_splits": {
        split: {
            "examples": m["examples"],
            "macro_average_precision": m["macro_ap_average"],
            "overall_average_precision": m["overall_ap_average"],
            "predictions_file": m["predictions_file"],
            "symmetry": m,
        }
        for split, m in all_splits.items()
    },
    "macro_average_precision": all_splits["iid"]["macro_ap_average"],
    "overall_average_precision": all_splits["iid"]["overall_ap_average"],
    "symmetry_diagnostic": all_splits,
    "args": {"diagnostic": True, "model_path": str(model_path), "max_length": 384},
}
(OUTPUT_DIR / "symmetry_diagnostic.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
(OUTPUT_DIR / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
training_wall_seconds = 0.0
print(json.dumps(all_splits, ensure_ascii=False, indent=2))
'''


def make_random() -> None:
    nb = clone()
    source(nb["cells"][8], RANDOM_SWAP)
    setup_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": ["team-editable", "variant-setup"]},
        "outputs": [],
        "source": [
            "OUTPUT_DIR = WORKING_ROOT / 'minilm_5ep_symmetry_random_swap'\n",
            "TRAIN_LOG = WORKING_ROOT / 'minilm_5ep_symmetry_random_swap.log'\n",
            "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n",
        ],
    }
    nb["cells"].insert(16, setup_cell)
    path = OUT / "minilm_5ep_symmetry_random_swap_2xt4.ipynb"
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def make_explicit() -> None:
    nb = clone()
    source(nb["cells"][12], """%%writefile /kaggle/working/product_matching/team_loss_hook.py
from __future__ import annotations
import torch.nn.functional as F

def initialize_loss(*, train_frame, device, rank, world_size):
    return None

def compute_loss(*, logits, targets, sample_weights, pair_indices, orientations, epoch, step):
    per_example_bce = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction=\"none\")
    bce = (per_example_bce * sample_weights).sum() / sample_weights.sum()
    return {\"loss\": bce, \"bce\": bce.detach()}
""")
    patch_cell = {"cell_type": "code", "execution_count": None, "metadata": {"tags": ["team-editable", "symmetry-patch"]}, "outputs": [], "source": ("OUTPUT_DIR = WORKING_ROOT / 'minilm_5ep_symmetry_explicit'\nTRAIN_LOG = WORKING_ROOT / 'minilm_5ep_symmetry_explicit.log'\nOUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n\n" + SYMMETRY_PATCH).splitlines(keepends=True)}
    nb["cells"].insert(16, patch_cell)
    path = OUT / "minilm_5ep_symmetry_explicit_2xt4.ipynb"
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def make_diagnostic() -> None:
    nb = clone()
    source(nb["cells"][16], DIAGNOSTIC)
    path = OUT / "minilm_5ep_symmetry_diagnostic_2xt4.ipynb"
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    make_diagnostic()
    make_random()
    make_explicit()
    print("Created three MiniLM symmetry notebooks")
