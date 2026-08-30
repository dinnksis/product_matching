"""Run soft-positive Tier A and Tier B Qwen generation sequentially.

Tier B starts only after Tier A reaches pending=0.  A non-zero/pending pass is
resumed with a deterministic task-seed offset, up to ``--max-passes`` passes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
DONORS = (
    ROOT
    / "item_pipeline"
    / "artifacts"
    / "generated_style_donors_x6_soft_positive_v1"
    / "items.parquet"
)
CATALOG_DIR = (
    ROOT
    / "configs"
    / "generation_rule_catalog_statistical_v1"
    / "soft_positive_ab_v1"
)

TIERS = (
    {
        "name": "tier_a",
        "rules": CATALOG_DIR / "tier_a.json",
        "output": ROOT / "item_pipeline" / "artifacts" / "soft_positive_tier_a_6500_qwen_v1_raw",
        "count": 6500,
        "seed": 20260828,
    },
    {
        "name": "tier_b",
        "rules": CATALOG_DIR / "tier_b.json",
        "output": ROOT / "item_pipeline" / "artifacts" / "soft_positive_tier_b_27930_qwen_v1_raw",
        "count": 27930,
        "seed": 20260829,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://0.0.0.0:8994/v1")
    parser.add_argument("--model", default="qwen3.5-397b-a17b-fp8")
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--max-passes", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def check_inputs() -> None:
    missing = [path for path in [PYTHON, DONORS, *(tier["rules"] for tier in TIERS)] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing generation inputs: " + ", ".join(map(str, missing)))


def check_server(base_url: str, expected_model: str) -> None:
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Qwen API is unavailable at {url}: {error}") from error
    model_ids = {str(row.get("id")) for row in payload.get("data", [])}
    if expected_model not in model_ids:
        raise RuntimeError(
            f"Expected model {expected_model!r} is absent at {url}; available={sorted(model_ids)}"
        )


def command_for(
    tier: dict[str, object],
    *,
    base_url: str,
    model: str,
    workers: int,
    task_seed_offset: int,
) -> list[str]:
    return [
        str(PYTHON),
        "-m",
        "item_pipeline",
        "generate-pairs",
        "--items",
        str(DONORS),
        "--rules",
        str(tier["rules"]),
        "--output-dir",
        str(tier["output"]),
        "--count",
        str(tier["count"]),
        "--base-url",
        base_url,
        "--model",
        model,
        "--workers",
        str(workers),
        "--timeout-seconds",
        "90",
        "--retries",
        "2",
        "--pair-attempts",
        "2",
        "--anchor-attempts",
        "2",
        "--mutation-attempts",
        "2",
        "--task-retries",
        "1",
        "--task-seed-offset",
        str(task_seed_offset),
        "--checkpoint-every",
        "25",
        "--seed",
        str(tier["seed"]),
        "--two-rule-fraction",
        "0",
        "--label-one-fraction",
        "1",
        "--semantic-signature-limit",
        "5",
        "--temperature",
        "0.7",
        "--max-tokens",
        "1400",
        "--plain-json",
    ]


def read_progress(tier: dict[str, object]) -> tuple[int, int]:
    summary_path = Path(tier["output"]) / "summary.json"
    if not summary_path.exists():
        return 0, int(tier["count"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    generated = int(summary.get("generated_pairs", 0))
    pending = int(summary.get("pending", int(tier["count"]) - generated))
    return generated, pending


def run_tier(
    tier: dict[str, object],
    *,
    base_url: str,
    model: str,
    workers: int,
    max_passes: int,
    dry_run: bool,
) -> None:
    previous_generated = -1
    stagnant_passes = 0
    for pass_index in range(max_passes):
        generated, pending = read_progress(tier)
        if pending == 0 and generated == int(tier["count"]):
            print(f"[{tier['name']}] already complete: {generated}/{tier['count']}", flush=True)
            return
        offset = pass_index * 10_000
        command = command_for(
            tier,
            base_url=base_url,
            model=model,
            workers=workers,
            task_seed_offset=offset,
        )
        print(
            f"[{tier['name']}] pass={pass_index + 1}/{max_passes} "
            f"generated={generated} pending={pending} offset={offset}",
            flush=True,
        )
        if dry_run:
            print(" ".join(command), flush=True)
            return
        check_server(base_url, model)
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode not in {0, 2}:
            raise RuntimeError(
                f"{tier['name']} generator exited with code {result.returncode}"
            )
        generated, pending = read_progress(tier)
        print(
            f"[{tier['name']}] pass complete: generated={generated} pending={pending}",
            flush=True,
        )
        if generated == int(tier["count"]) and pending == 0:
            return
        if generated <= previous_generated:
            stagnant_passes += 1
        else:
            stagnant_passes = 0
        previous_generated = generated
        if stagnant_passes >= 2:
            raise RuntimeError(
                f"{tier['name']} made no progress for two passes; "
                f"generated={generated}, pending={pending}"
            )
    generated, pending = read_progress(tier)
    raise RuntimeError(
        f"{tier['name']} remains incomplete after {max_passes} passes: "
        f"generated={generated}, pending={pending}"
    )


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.max_passes < 1:
        raise ValueError("--max-passes must be positive")
    check_inputs()
    if not args.dry_run:
        check_server(args.base_url, args.model)
    for tier in TIERS:
        run_tier(
            tier,
            base_url=args.base_url,
            model=args.model,
            workers=args.workers,
            max_passes=args.max_passes,
            dry_run=args.dry_run,
        )
    print("Soft-positive Tier A and Tier B generation completed.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; rerun the same launcher to resume checkpoints.", file=sys.stderr)
        raise SystemExit(130)
