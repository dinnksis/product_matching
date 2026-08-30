from __future__ import annotations

import argparse
import http.client
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ITEM_OUTPUT = ROOT / "item_pipeline" / "artifacts" / "generated"
PAIR_OUTPUT = ROOT / "item_pipeline" / "artifacts" / "rule_first_pairs"
SUPPLEMENT_OUTPUT = ROOT / "item_pipeline" / "artifacts" / "generated_supplement_plain_json"
EXPERIMENT_NOTEBOOK = ROOT / (
    "notebooks/minilm_5ep_team_ablation/"
    "minilm_5ep_generation_rules_10k_v2_2xt4.ipynb"
)
EXPERIMENT_SLUG = "product-matching-minilm-5ep-generation-rules-10k-v2"


def wait_for_pid(pid: int) -> None:
    print(f"waiting for existing item generation pid={pid}", flush=True)
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            pass
        time.sleep(30)


def read_pending(summary_path: Path) -> int | None:
    if not summary_path.exists():
        return None
    try:
        return int(json.loads(summary_path.read_text(encoding="utf-8"))["pending"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def run(command: list[str]) -> int:
    print("running:", " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def run_checked(command: list[str]) -> None:
    return_code = run(command)
    if return_code:
        raise RuntimeError(
            f"command failed with exit code {return_code}: {' '.join(command)}"
        )


def wait_for_model(base_url: str, model: str, interval_seconds: int = 30) -> None:
    endpoint = f"{base_url.rstrip('/')}/models"
    attempts = 0
    while True:
        attempts += 1
        try:
            with urllib.request.urlopen(endpoint, timeout=5) as response:
                payload = json.load(response)
            model_ids = {
                str(entry.get("id"))
                for entry in payload.get("data", [])
                if isinstance(entry, dict)
            }
            if model in model_ids:
                print(f"model endpoint ready: {model}", flush=True)
                return
            detail = f"model {model!r} absent; available={sorted(model_ids)}"
        except (
            OSError,
            ValueError,
            http.client.HTTPException,
            urllib.error.URLError,
        ) as error:
            detail = str(error)
        if attempts == 1 or attempts % 10 == 0:
            print(f"waiting for model endpoint {endpoint}: {detail}", flush=True)
        time.sleep(interval_seconds)


def item_command(args: argparse.Namespace, seed_offset: int) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "item_pipeline",
        "generate",
        "--count",
        str(args.count),
        "--workers",
        str(args.workers),
        "--output-dir",
        str(ITEM_OUTPUT),
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--timeout-seconds",
        "90",
        "--retries",
        "2",
        "--generation-attempts",
        "3",
        "--task-retries",
        "4",
        "--task-seed-offset",
        str(seed_offset),
        "--checkpoint-every",
        "25",
        "--max-tokens",
        "800",
    ]
    if args.plain_json:
        command.append("--plain-json")
    return command


def pair_command(args: argparse.Namespace, seed_offset: int) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "item_pipeline",
        "generate-pairs",
        "--items",
        str(ITEM_OUTPUT / "items.parquet"),
        "--count",
        str(args.count),
        "--workers",
        str(args.workers),
        "--output-dir",
        str(PAIR_OUTPUT),
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--timeout-seconds",
        "90",
        "--retries",
        "2",
        "--pair-attempts",
        "5",
        "--anchor-attempts",
        "3",
        "--mutation-attempts",
        "3",
        "--task-retries",
        "4",
        "--task-seed-offset",
        str(seed_offset),
        "--checkpoint-every",
        "25",
        "--max-tokens",
        "1400",
        "--two-rule-fraction",
        "0.5",
        "--semantic-signature-limit",
        str(args.semantic_signature_limit),
    ]
    if args.plain_json:
        command.append("--plain-json")
    return command


def supplement_item_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "item_pipeline",
        "generate",
        "--count",
        str(args.supplement_count),
        "--workers",
        str(args.workers),
        "--output-dir",
        str(SUPPLEMENT_OUTPUT),
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--timeout-seconds",
        "90",
        "--retries",
        "2",
        "--generation-attempts",
        "3",
        "--task-retries",
        "0",
        "--checkpoint-every",
        "25",
        "--max-tokens",
        "800",
        "--seed",
        "20260824",
        "--id-start",
        "-20001",
    ]
    if args.plain_json:
        command.append("--plain-json")
    return command


def complete_phase(
    name: str,
    summary_path: Path,
    command_builder,
    args: argparse.Namespace,
) -> None:
    pending = read_pending(summary_path)
    if pending == 0:
        print(f"{name}: already complete", flush=True)
        return
    for seed_offset in (10_000, 20_000, 30_000, 40_000, 50_000):
        print(f"{name}: pending={pending}, seed_offset={seed_offset}", flush=True)
        wait_for_model(args.base_url, args.model)
        run(command_builder(args, seed_offset))
        pending = read_pending(summary_path)
        if pending == 0:
            print(f"{name}: complete", flush=True)
            return
    raise RuntimeError(f"{name} remains incomplete after fallback passes: pending={pending}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finish 10k standalone items, then generate and validate rule pairs."
    )
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--semantic-signature-limit", type=int, default=2)
    parser.add_argument("--base-url", default="http://127.0.0.1:8193/v1")
    parser.add_argument("--model", default="qwen3.5-397b-a17b-fp8")
    parser.add_argument(
        "--plain-json",
        action="store_true",
        help="do not send response_format/json_schema to vLLM",
    )
    parser.add_argument(
        "--fresh-item-tail",
        action="store_true",
        help="fill a hard primary tail using fresh schema donors",
    )
    parser.add_argument("--supplement-count", type=int, default=1700)
    parser.add_argument(
        "--launch-minilm",
        action="store_true",
        help="upload the completed pairs and run the frozen MiniLM 5ep ablation",
    )
    return parser.parse_args()


def launch_minilm_experiment() -> None:
    push_command = [
        sys.executable,
        "scripts/push_generation_rule_pairs_dataset.py",
    ]
    run_checked(push_command + ["--dry-run"])
    run_checked(push_command)
    run_checked([sys.executable, "scripts/create_generation_rule_10k_notebook.py"])

    notebook_command = [
        sys.executable,
        "scripts/run_kaggle_notebook.py",
        str(EXPERIMENT_NOTEBOOK),
        "--slug",
        EXPERIMENT_SLUG,
        "--title",
        "MiniLM 5ep: rule-first generation 10k v2",
        "--dataset",
        "alexproger23/product-matching-validation-splits-v1",
        "--dataset",
        "alexproger23/product-matching-minilm-llm-pretrain-5ep",
        "--dataset",
        "alexproger23/product-matching-minilm-5ep-significance-v1",
        "--dataset",
        "alexproger23/product-matching-generation-rule-pairs-10k-v2",
        "--no-env-sources",
    ]
    run_checked(notebook_command + ["--dry-run"])
    run_checked(notebook_command)

    output_dir = ROOT / "artifacts" / "kaggle" / EXPERIMENT_SLUG
    completion_path = output_dir / "notebook_completed.json"
    sync_path = output_dir / "google_sheets_sync.json"
    if not completion_path.is_file() or not sync_path.is_file():
        raise RuntimeError("MiniLM run finished without completion or Sheets sync artifact")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    sync = json.loads(sync_path.read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        raise RuntimeError(f"MiniLM completion is not successful: {completion}")
    if sync.get("status") != "synced" or sync.get("comparison_sheet") != "data_exps":
        raise RuntimeError(f"MiniLM metrics were not synced to data_exps: {sync}")
    print(
        f"MiniLM experiment complete: run_id={completion.get('run_id')}; "
        "metrics synced to data_exps",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    if args.semantic_signature_limit < 1:
        raise ValueError("semantic-signature-limit must be positive")
    if args.wait_pid is not None:
        wait_for_pid(args.wait_pid)

    item_summary = ITEM_OUTPUT / "summary.json"
    item_pending = read_pending(item_summary)
    if args.fresh_item_tail and item_pending not in (None, 0):
        print(
            f"items: replacing hard tail of {item_pending} tasks with fresh donors",
            flush=True,
        )
        wait_for_model(args.base_url, args.model)
        supplement_result = run(supplement_item_command(args))
        if supplement_result not in (0, 2):
            raise RuntimeError(
                f"supplement generation failed with exit code {supplement_result}"
            )
        run_checked(
            [
                sys.executable,
                "scripts/fill_generated_item_tail.py",
                "--count",
                str(args.count),
            ]
        )

    complete_phase(
        "items",
        item_summary,
        item_command,
        args,
    )
    if run([sys.executable, "-m", "item_pipeline", "validate"]):
        raise RuntimeError("standalone item validation failed")

    pair_summary = PAIR_OUTPUT / "summary.json"
    if read_pending(pair_summary) is None:
        wait_for_model(args.base_url, args.model)
        run(pair_command(args, 0))
    complete_phase("pairs", pair_summary, pair_command, args)
    if run([sys.executable, "-m", "item_pipeline", "validate-pairs"]):
        raise RuntimeError("pair dataset validation failed")
    print("full item/rule dataset is complete and valid", flush=True)
    if args.launch_minilm:
        launch_minilm_experiment()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
