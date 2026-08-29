#!/usr/bin/env python3
"""Plan, launch and validate the five-run RuModernBERT Kaggle campaign."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import create_rumodernbert_sft_kaggle_notebooks as builder
import push_minilm_pretrain_checkpoint_dataset as checkpoint_remote
import push_rumodernbert_pretrain_checkpoint_dataset as checkpoint_uploader
import push_validation_splits_dataset as validation_remote
import run_kaggle_notebook as kaggle
from src.experiment_significance import compare_prediction_frames, holm_adjust


OUTPUT_ROOT = ROOT / "artifacts" / "kaggle"
REPORT_ROOT = ROOT / "reports" / builder.CAMPAIGN
SUMMARY_PATH = REPORT_ROOT / "campaign_summary.json"
LOCK_PATH = ROOT / ".kaggle" / "locks" / f"{builder.CAMPAIGN}.lock"
FIRST_KEYS = ("e1_lr8e5", "e1_lr4e5", "e1_lr1p6e4")
ALL_KEYS = (*FIRST_KEYS, "e2_selected_lr", "e3_selected_lr")


class CampaignError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CampaignError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(pending, path)


@contextmanager
def campaign_lock():
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CampaignError("another RuModernBERT Kaggle controller is active") from error
        yield


def macro_ap(frame: pd.DataFrame) -> float:
    values = [
        average_precision_score(group["target"], group["score_symmetric"])
        for _, group in frame.groupby("category", sort=True)
    ]
    return float(np.mean(values))


def expected_truth(split: str) -> pd.DataFrame:
    path = builder.DEFAULT_SOURCE_DIR / "human" / f"{split}_validation_pairs.parquet"
    return pd.read_parquet(path).reset_index(drop=True)


def validate_run_output(directory: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    completion_path = directory / "notebook_completed.json"
    if not completion_path.is_file():
        raise CampaignError(f"completion is missing: {completion_path}")
    completion = read_json(completion_path)
    exact = {
        "status": "complete",
        "campaign": builder.CAMPAIGN,
        "key": entry["key"],
        "experiment": entry["experiment"],
        "campaign_identity_sha256": entry["identity_sha256"],
        "code_bundle_sha256": entry["source_sha256"],
        "frozen_recipe_sha256": entry["recipe_sha256"],
        "initial_checkpoint_manifest_sha256": entry["checkpoint_manifest_sha256"],
        "initial_checkpoint_model_sha256": builder.MODEL_SHA256,
    }
    mismatches = {
        key: {"actual": completion.get(key), "expected": value}
        for key, value in exact.items()
        if completion.get(key) != value
    }
    if mismatches:
        raise CampaignError(f"completion identity differs: {mismatches}")
    run_id = completion.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise CampaignError("completion run_id is missing")
    output = directory / str(entry["experiment"])
    if not output.is_dir():
        raise CampaignError(f"trained output directory is missing: {output}")
    report = read_json(output / "training_report.json")
    config = read_json(output / "training_config.json")
    expected_config = dict(entry["expected_config"])
    if config != expected_config:
        raise CampaignError("downloaded training config differs from frozen recipe")
    if completion.get("training_report") != report:
        raise CampaignError("completion/report payload differs")
    if report.get("original_training_examples") != builder.EXPECTED_TRAIN:
        raise CampaignError("training row count differs")
    if report.get("training_sampling") != "none" or report.get("training_loss_weighting") != "none":
        raise CampaignError("sampling or weighting unexpectedly enabled")
    splits = report.get("validation_splits")
    if not isinstance(splits, dict) or set(splits) != {"iid", "hard", "ood"}:
        raise CampaignError("validation split set differs")
    ood = splits["ood"]
    if ood.get("evaluated") is not False or ood.get("macro_average_precision") != -1.0:
        raise CampaignError("OOD sentinel differs")
    if list(output.glob("ood*validation_predictions.parquet")):
        raise CampaignError("OOD predictions must not exist")
    metrics: dict[str, float] = {}
    for split, rows in (("iid", builder.EXPECTED_IID), ("hard", builder.EXPECTED_HARD)):
        predictions = pd.read_parquet(output / f"{split}_validation_predictions.parquet").reset_index(drop=True)
        truth = expected_truth(split)
        required = {"id1", "id2", "target", "category", "score_symmetric"}
        if len(predictions) != rows or required - set(predictions):
            raise CampaignError(f"{split} prediction schema/rows differ")
        for column in ("id1", "id2", "target"):
            if not predictions[column].equals(truth[column]):
                raise CampaignError(f"{split} {column} binding differs")
        categories = truth["id1"].map(
            pd.read_parquet(builder.DEFAULT_SOURCE_DIR / "human" / "items.parquet")
            .set_index("id")["category"]
        ).astype(str)
        if not predictions["category"].astype(str).equals(categories):
            raise CampaignError(f"{split} category binding differs")
        if not np.isfinite(predictions["score_symmetric"]).all():
            raise CampaignError(f"{split} scores are non-finite")
        measured = macro_ap(predictions)
        declared = float(splits[split]["macro_average_precision"])
        if not math.isclose(measured, declared, rel_tol=0.0, abs_tol=1e-12):
            raise CampaignError(f"{split} macro AP differs: {measured} != {declared}")
        metrics[split] = measured
    model_path = output / "model.safetensors"
    if not model_path.is_file():
        raise CampaignError("trained model.safetensors is missing")
    if model_path.stat().st_size != completion.get("trained_model_bytes"):
        raise CampaignError("trained model byte count differs")
    if builder.file_sha256(model_path) != completion.get("trained_model_sha256"):
        raise CampaignError("trained model SHA-256 differs")
    sync = read_json(directory / "google_sheets_sync.json")
    if sync.get("status") != "synced" or sync.get("run_id") != run_id:
        raise CampaignError("Google Sheets sync marker differs")
    return {
        "key": entry["key"],
        "run_id": run_id,
        "kernel_slug": entry["kernel_slug"],
        "identity_sha256": entry["identity_sha256"],
        "epochs": int(entry["epochs"]),
        "learning_rate": float(entry["learning_rate"]),
        "iid_macro_average_precision": metrics["iid"],
        "hard_macro_average_precision": metrics["hard"],
        "trained_model": str(model_path),
        "trained_model_sha256": completion["trained_model_sha256"],
        "output_dir": str(output),
    }


def select_learning_rate(results: Sequence[Mapping[str, Any]], margin: float = 0.002) -> float:
    if {row["key"] for row in results} != set(FIRST_KEYS):
        raise CampaignError("LR selection requires exactly the first three runs")
    best = max(float(row["iid_macro_average_precision"]) for row in results)
    candidates = [row for row in results if best - float(row["iid_macro_average_precision"]) <= margin]
    anchor = next((row for row in candidates if float(row["learning_rate"]) == 8e-5), None)
    chosen = anchor if anchor is not None else min(candidates, key=lambda row: float(row["learning_rate"]))
    return float(chosen["learning_rate"])


def select_final(results: Sequence[Mapping[str, Any]], selected_lr: float, margin: float = 0.002) -> Mapping[str, Any]:
    candidates = [row for row in results if float(row["learning_rate"]) == selected_lr]
    if {int(row["epochs"]) for row in candidates} != {1, 2, 3}:
        raise CampaignError("final selection requires matching e1/e2/e3 runs")
    best = max(float(row["iid_macro_average_precision"]) for row in candidates)
    practical = [row for row in candidates if best - float(row["iid_macro_average_precision"]) <= margin]
    return min(practical, key=lambda row: int(row["epochs"]))


def _listed_kernel_refs(output: str) -> set[str]:
    if output.strip().casefold() == "not found":
        return set()
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as error:
        raise CampaignError("Kaggle kernel listing was not JSON") from error
    if not isinstance(payload, list):
        raise CampaignError("Kaggle kernel listing must be a list")
    return {
        str(row.get("ref") or row.get("id"))
        for row in payload
        if isinstance(row, dict) and (row.get("ref") or row.get("id"))
    }


def kernel_list(cli: list[str], kernel_ref: str):
    slug = kernel_ref.rsplit("/", 1)[-1]
    return kaggle.run_command(
        cli + ["kernels", "list", "--mine", "--search", slug, "--page-size", "100", "--format", "json"],
        check=False,
    )


def remote_status(cli: list[str], kernel_ref: str) -> str:
    status = kaggle.run_command(cli + ["kernels", "status", kernel_ref], check=False)
    if status.returncode == 0:
        return kaggle.extract_status(status.stdout) or "status_error"
    listed = kernel_list(cli, kernel_ref)
    if listed.returncode:
        raise CampaignError(f"could not resolve remote kernel: {kernel_ref}")
    if kernel_ref in _listed_kernel_refs(listed.stdout):
        raise CampaignError("kernel is listed but status is unreadable")
    return "absence_unconfirmed"


def confirm_absence(cli: list[str], kernel_ref: str) -> None:
    for attempt in range(2):
        listed = kernel_list(cli, kernel_ref)
        if listed.returncode or kernel_ref in _listed_kernel_refs(listed.stdout):
            raise CampaignError(f"remote kernel appeared before push: {kernel_ref}")
        status = kaggle.run_command(cli + ["kernels", "status", kernel_ref], check=False)
        if status.returncode == 0:
            raise CampaignError(f"remote kernel appeared with status {kaggle.extract_status(status.stdout)}")
        if attempt == 0:
            time.sleep(0.5)


def expected_sources(owner: str, entry: Mapping[str, Any]) -> list[str]:
    return [
        str(entry["validation_dataset"]),
        str(entry["checkpoint_dataset"]),
        f"{owner}/{kaggle.GOOGLE_CREDENTIALS_DATASET_SLUG}",
    ]


def stage_entry(entry: Mapping[str, Any], notebook: Path, env_file: Path) -> Path:
    command = [
        sys.executable, str(ROOT / "scripts" / "run_kaggle_notebook.py"), str(notebook),
        "--env-file", str(env_file), "--slug", str(entry["kernel_slug"]),
        "--title", str(entry["title"]), "--no-env-sources", "--dry-run",
    ]
    for source in (entry["validation_dataset"], entry["checkpoint_dataset"]):
        command.extend(["--dataset", str(source)])
    subprocess.run(command, cwd=ROOT, check=True)
    stage = kaggle.STAGE_ROOT / str(entry["kernel_slug"])
    metadata = read_json(stage / "kernel-metadata.json")
    expected = expected_sources(os.getenv("KAGGLE_USERNAME", ""), entry)
    actual = metadata.get("dataset_sources")
    if not isinstance(actual, list) or len(actual) != len(expected) or set(actual) != set(expected):
        raise CampaignError(f"staged Dataset sources differ: {actual} != {expected}")
    if metadata.get("is_private") is not True or metadata.get("machine_shape") != "NvidiaTeslaT4":
        raise CampaignError("staged kernel privacy/accelerator differs")
    return stage


def verify_remote_sources(cli: list[str], kernel_ref: str, entry: Mapping[str, Any], owner: str) -> None:
    with tempfile.TemporaryDirectory(prefix="rmb-kernel-metadata-") as temporary:
        result = kaggle.run_command(cli + ["kernels", "pull", kernel_ref, "-p", temporary, "-m"], check=False)
        if result.returncode:
            raise CampaignError("could not pull remote kernel metadata")
        metadata = read_json(Path(temporary) / "kernel-metadata.json")
    actual = metadata.get("dataset_sources")
    expected = expected_sources(owner, entry)
    if not isinstance(actual, list) or len(actual) != len(expected) or len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise CampaignError(f"remote Dataset sources differ: {actual} != {expected}")


def download_output(cli: list[str], owner: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    destination = OUTPUT_ROOT / str(entry["kernel_slug"])
    if destination.is_dir():
        return validate_run_output(destination, entry)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{entry['kernel_slug']}.download-", dir=OUTPUT_ROOT))
    try:
        kaggle.run_command(cli + ["kernels", "output", f"{owner}/{entry['kernel_slug']}", "-p", str(staging), "--force", "--page-size", "200"])
        result = validate_run_output(staging, entry)
        staging.rename(destination)
        return result
    except Exception:
        print(f"Invalid output preserved at {staging}", file=sys.stderr)
        raise


def verify_remote_authorities(cli: list[str], owner: str) -> None:
    validation_remote.verify_remote_dataset(
        cli,
        f"{owner}/{validation_remote.DATASET_SLUG}",
        set(validation_remote.FILE_MAP.values()) | {"upload_manifest.json"},
    )
    manifest_path = checkpoint_uploader.STAGE_DIR / checkpoint_uploader.MANIFEST_NAME
    manifest = read_json(manifest_path)
    checkpoint_remote.verify_remote_dataset(
        cli,
        f"{owner}/{checkpoint_uploader.DATASET_SLUG}",
        builder.file_sha256(manifest_path),
        set(manifest["files"]),
    )


def build_entry(owner: str, key: str, selected_lr: float | None = None) -> tuple[dict[str, Any], Path]:
    notebook, entry = builder.build_variant(owner=owner, key=key, selected_lr=selected_lr)
    path = builder.write_variant(notebook, entry)
    return entry, path


def run_entry(
    *, cli: list[str], owner: str, env_file: Path, entry: Mapping[str, Any], notebook: Path,
    execute: bool,
) -> dict[str, Any] | None:
    local = OUTPUT_ROOT / str(entry["kernel_slug"])
    if local.is_dir():
        return validate_run_output(local, entry)
    kernel_ref = f"{owner}/{entry['kernel_slug']}"
    status = remote_status(cli, kernel_ref) if execute else "dry_run"
    print(json.dumps({"kernel_ref": kernel_ref, "status": status}))
    stage = stage_entry(entry, notebook, env_file)
    if not execute:
        return None
    if status == "absence_unconfirmed":
        confirm_absence(cli, kernel_ref)
        kaggle.run_command(cli + ["kernels", "push", "-p", str(stage), "--timeout", "32400", "--accelerator", "NvidiaTeslaT4"])
        verify_remote_sources(cli, kernel_ref, entry, owner)
        status = "queued"
    elif status in {"queued", "running"}:
        verify_remote_sources(cli, kernel_ref, entry, owner)
    elif status in kaggle.TERMINAL_FAILURE:
        kaggle.run_command(cli + ["kernels", "logs", kernel_ref], check=False)
        raise CampaignError(f"terminal failed kernel will not be resubmitted: {kernel_ref}")
    elif status in kaggle.TERMINAL_SUCCESS:
        verify_remote_sources(cli, kernel_ref, entry, owner)
        return download_output(cli, owner, entry)
    else:
        raise CampaignError(f"unexpected remote status: {status}")
    kaggle.wait_for_kernel(
        cli,
        kernel_ref,
        poll_interval=kaggle.env_int("KAGGLE_POLL_INTERVAL_SECONDS", 30, minimum=5),
        wait_timeout=33_000,
    )
    return download_output(cli, owner, entry)


def paired_family(
    baseline: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], *, seed: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    baseline_dir = Path(str(baseline["output_dir"]))
    for index, candidate in enumerate(candidates):
        candidate_dir = Path(str(candidate["output_dir"]))
        splits = {
            split: compare_prediction_frames(
                pd.read_parquet(baseline_dir / f"{split}_validation_predictions.parquet"),
                pd.read_parquet(candidate_dir / f"{split}_validation_predictions.parquet"),
                permutations=2_000,
                bootstrap_resamples=2_000,
                seed=seed + 10 * index + split_index,
            )
            for split_index, split in enumerate(("iid", "hard"))
        }
        result[str(candidate["key"])] = {"splits": splits}
    for split in ("iid", "hard"):
        adjusted = holm_adjust(
            {
                key: float(value["splits"][split]["p_value"])
                for key, value in result.items()
            }
        )
        for key, value in adjusted.items():
            result[key]["splits"][split]["p_value_holm"] = value
    return result


def summarize(results: Sequence[Mapping[str, Any]], selected_lr: float) -> dict[str, Any]:
    selected = select_final(results, selected_lr)
    by_key = {str(row["key"]): row for row in results}
    selected_e1 = next(
        row
        for row in results
        if int(row["epochs"]) == 1 and float(row["learning_rate"]) == selected_lr
    )
    payload = {
        "schema_version": 1,
        "campaign": builder.CAMPAIGN,
        "status": "complete",
        "completed_runs": 5,
        "selected_learning_rate": selected_lr,
        "recommended_key": selected["key"],
        "recommended_epochs": selected["epochs"],
        "recommended_model": selected["trained_model"],
        "recommended_model_sha256": selected["trained_model_sha256"],
        "runs": list(results),
        "comparisons": {
            "lr_family_vs_e1_lr8e5": paired_family(
                by_key["e1_lr8e5"],
                [by_key["e1_lr4e5"], by_key["e1_lr1p6e4"]],
                seed=42,
            ),
            "epoch_family_vs_selected_e1": paired_family(
                selected_e1,
                [by_key["e2_selected_lr"], by_key["e3_selected_lr"]],
                seed=142,
            ),
        },
        "ood": {"evaluated": False, "metric_sentinel": -1},
        "selection_policy": "IID macro AP; fewest epochs within 0.002 of numeric best",
    }
    atomic_json(SUMMARY_PATH, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stage-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.execute and args.stage_only:
        raise SystemExit("choose --execute or --stage-only")
    env_file = args.env_file if args.env_file.is_absolute() else ROOT / args.env_file
    kaggle.load_dotenv(env_file)
    owner = kaggle.validate_slug(os.getenv("KAGGLE_USERNAME", "").strip(), "KAGGLE_USERNAME")
    if owner != "alexproger23":
        raise CampaignError("frozen campaign owner must remain alexproger23")
    if os.getenv("KAGGLE_ACCELERATOR", "NvidiaTeslaT4") != "NvidiaTeslaT4":
        raise CampaignError("campaign requires NvidiaTeslaT4")
    if kaggle.env_bool("KAGGLE_IS_PRIVATE", True) is not True:
        raise CampaignError("campaign kernels must remain private")
    cli = kaggle.kaggle_command()
    execute = bool(args.execute)
    with campaign_lock():
        if execute:
            verify_remote_authorities(cli, owner)
        results: list[dict[str, Any]] = []
        for key in FIRST_KEYS:
            entry, notebook = build_entry(owner, key)
            result = run_entry(cli=cli, owner=owner, env_file=env_file, entry=entry, notebook=notebook, execute=execute)
            if result is not None:
                results.append(result)
        if not execute:
            for key in ("e2_selected_lr", "e3_selected_lr"):
                entry, notebook = build_entry(owner, key, 8e-5)
                run_entry(cli=cli, owner=owner, env_file=env_file, entry=entry, notebook=notebook, execute=False)
            print(json.dumps({"campaign":builder.CAMPAIGN,"runs":5,"mutation":False,"hypothetical_selected_lr":8e-5}, indent=2))
            return 0
        selected_lr = select_learning_rate(results)
        for key in ("e2_selected_lr", "e3_selected_lr"):
            entry, notebook = build_entry(owner, key, selected_lr)
            result = run_entry(cli=cli, owner=owner, env_file=env_file, entry=entry, notebook=notebook, execute=True)
            assert result is not None
            results.append(result)
        summary = summarize(results, selected_lr)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
