"""Sanitize a completed internal-validation checkpoint and score frozen rules."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "qwen_rule_internal_validation_v1"
DEFAULT_RAW = ROOT / "artifacts" / "qwen_rule_internal_validation_v1"
DEFAULT_SANITIZED = ROOT / "artifacts" / "qwen_rule_internal_validation_v1_sanitized"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process Qwen internal validation and score frozen rules."
    )
    parser.add_argument("--posterior-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--sanitized-dir", type=Path, default=DEFAULT_SANITIZED)
    parser.add_argument(
        "--frozen-dir",
        type=Path,
        default=ROOT / "artifacts" / "qwen_rules_frozen_v1_60000",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "qwen_rule_internal_validation_v1",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\nRUN:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    raw_dir = args.raw_dir.resolve()
    sanitized_dir = args.sanitized_dir.resolve()
    labels = data_dir / "validation_labels.parquet"
    dataset = data_dir / "validation_inputs.parquet"
    pair_count = __import__("pandas").read_parquet(dataset, columns=["pair_id"]).shape[0]
    python = sys.executable

    run(
        [
            python,
            str(ROOT / "scripts" / "sanitize_qwen_semantic_extraction.py"),
            "--raw-responses", str(raw_dir / "raw_responses.jsonl"),
            "--dataset", str(dataset),
            "--labels", str(labels),
            "--schema", str(ROOT / "schemas" / "qwen_semantic_extraction_v1_3.schema.json"),
            "--prompt", str(ROOT / "prompts" / "qwen_semantic_extraction_v1_3.md"),
            "--validation-profile", "v1_4",
            "--max-pairs", str(pair_count),
            "--only-available-responses",
            "--output-dir", str(sanitized_dir),
        ]
    )
    stats = json.loads(
        (sanitized_dir / "sanitization_statistics.json").read_text(encoding="utf-8")
    )
    if int(stats["requested_pairs"]) != pair_count:
        raise RuntimeError(
            f"Incomplete validation checkpoint: expected {pair_count}, "
            f"found {stats['requested_pairs']}"
        )

    run(
        [
            python,
            str(ROOT / "scripts" / "validate_frozen_qwen_rules.py"),
            "--extractions", str(sanitized_dir / "sanitized_pairs.jsonl"),
            "--labels", str(labels),
            "--frozen-dir", str(args.frozen_dir.resolve()),
            "--output-dir", str(args.output_dir.resolve()),
            "--posterior-draws", str(args.posterior_draws),
            "--seed", str(args.seed),
        ]
    )


if __name__ == "__main__":
    main()
