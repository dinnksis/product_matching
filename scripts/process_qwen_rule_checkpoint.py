"""Run sanitizer, label-free concept normalization and pilot rule analysis."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process a completed Qwen RULE_DISCOVERY prefix.")
    parser.add_argument("--checkpoint-size", type=int, required=True)
    parser.add_argument("--posterior-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument(
        "--raw-responses",
        type=Path,
        default=ROOT / "artifacts" / "qwen_semantic_extraction_v1_3_rule_discovery_full" / "raw_responses.jsonl",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "qwen_rule_discovery_full_v1" / "pilot_inputs.parquet",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "data" / "qwen_rule_discovery_full_v1" / "pilot_labels.parquet",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\nRUN:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    if args.checkpoint_size < 1 or args.posterior_draws < 1000:
        raise ValueError("checkpoint-size must be positive; posterior-draws must be >= 1000")

    size = args.checkpoint_size
    sanitized = ROOT / "artifacts" / f"qwen_semantic_extraction_v1_3_sanitized_checkpoint_{size}"
    normalized = ROOT / "artifacts" / f"qwen_concept_normalization_v1_checkpoint_{size}"
    report = ROOT / "reports" / f"qwen_rules_checkpoint_{size}"
    python = sys.executable

    run([
        python,
        str(ROOT / "scripts" / "sanitize_qwen_semantic_extraction.py"),
        "--raw-responses", str(args.raw_responses.resolve()),
        "--dataset", str(args.dataset.resolve()),
        "--labels", str(args.labels.resolve()),
        "--schema", str(ROOT / "schemas" / "qwen_semantic_extraction_v1_3.schema.json"),
        "--prompt", str(ROOT / "prompts" / "qwen_semantic_extraction_v1_3.md"),
        "--validation-profile", "v1_4",
        "--max-pairs", str(size),
        "--only-available-responses",
        "--output-dir", str(sanitized),
    ])

    stats_path = sanitized / "sanitization_statistics.json"
    statistics = json.loads(stats_path.read_text(encoding="utf-8"))
    completed = int(statistics["requested_pairs"])
    if completed != size:
        raise RuntimeError(
            f"Checkpoint prefix is incomplete: expected {size}, found {completed}. "
            "Rerun Qwen extraction with the same --max-pairs before post-processing."
        )

    run([
        python,
        str(ROOT / "scripts" / "normalize_qwen_pilot_concepts.py"),
        "--extractions", str(sanitized / "sanitized_pairs.jsonl"),
        "--output-dir", str(normalized),
    ])

    run([
        python,
        str(ROOT / "scripts" / "analyze_qwen_pilot_rules.py"),
        "--extractions", str(sanitized / "sanitized_pairs.jsonl"),
        "--labels", str(args.labels.resolve()),
        "--discovery-assignments", str(ROOT / "data" / "rule_discovery_split_v1" / "split_assignments.parquet"),
        "--concept-map", str(normalized / "concept_normalization_map.parquet"),
        "--effect-baseline", "discovery",
        "--sampling-design", "prevalence_random",
        "--output-dir", str(report),
        "--posterior-draws", str(args.posterior_draws),
        "--seed", str(args.seed),
    ])

    print(
        json.dumps(
            {
                "checkpoint_size": size,
                "sanitized": str(sanitized),
                "normalization": str(normalized),
                "rule_report": str(report / "report.md"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
