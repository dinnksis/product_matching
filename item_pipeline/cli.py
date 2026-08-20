from __future__ import annotations

import argparse
import json
from pathlib import Path

from .generate import run_generation
from .prepare import DEFAULT_EMBEDDING_MODEL, prepare_index
from .qwen import QwenItemClient, discover_model, load_system_prompt
from .validation import validate_generated_dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ITEMS = ROOT / "prepared" / "validation_splits_v1" / "llm" / "non_ood_items.parquet"
DEFAULT_INDEX = ROOT / "item_pipeline" / "artifacts" / "index"
DEFAULT_OUTPUT = ROOT / "item_pipeline" / "artifacts" / "generated"
DEFAULT_PROMPT = ROOT / "item_pipeline" / "prompts" / "generate_item.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m item_pipeline",
        description="Generate standalone synthetic product items.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Build the exemplar bank and MiniLM index")
    prepare.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_INDEX)
    prepare.add_argument("--max-items-per-category", type=int, default=10_000)
    prepare.add_argument("--seed", type=int, default=20260820)
    prepare.add_argument("--limit-rows", type=int)
    prepare.add_argument("--skip-embeddings", action="store_true")
    prepare.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    prepare.add_argument("--embedding-batch-size", type=int, default=256)
    prepare.add_argument("--embedding-device")
    prepare.add_argument("--embedding-local-files-only", action="store_true")

    generate = subparsers.add_parser("generate", help="Generate items with Qwen")
    generate.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    generate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    generate.add_argument("--count", type=int, default=100)
    generate.add_argument("--base-url", default="http://localhost:8193/v1")
    generate.add_argument("--model")
    generate.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    generate.add_argument("--workers", type=int, default=15)
    generate.add_argument("--timeout-seconds", type=float, default=120.0)
    generate.add_argument("--retries", type=int, default=8)
    generate.add_argument("--generation-attempts", type=int, default=3)
    generate.add_argument("--checkpoint-every", type=int, default=25)
    generate.add_argument("--seed", type=int, default=20260820)
    generate.add_argument("--id-start", type=int, default=-1)
    generate.add_argument("--examples", type=int, default=5)
    generate.add_argument("--category", action="append", dest="categories")
    generate.add_argument("--temperature", type=float, default=0.7)
    generate.add_argument("--max-tokens", type=int, default=1400)

    validate = subparsers.add_parser("validate", help="Validate generated item artifacts")
    validate.add_argument("--items", type=Path, default=DEFAULT_OUTPUT / "items.parquet")
    validate.add_argument("--reference", type=Path, default=DEFAULT_INDEX / "exemplar_bank.parquet")
    validate.add_argument(
        "--metadata", type=Path, default=DEFAULT_OUTPUT / "generation_metadata.parquet"
    )
    validate.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "validation_report.json")
    validate.add_argument("--no-reference", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        profile = prepare_index(
            args.items,
            args.output_dir,
            max_items_per_category=args.max_items_per_category,
            seed=args.seed,
            limit_rows=args.limit_rows,
            skip_embeddings=args.skip_embeddings,
            embedding_model=args.embedding_model,
            embedding_batch_size=args.embedding_batch_size,
            embedding_device=args.embedding_device,
            embedding_local_files_only=args.embedding_local_files_only,
        )
        print(json.dumps(profile, ensure_ascii=False, indent=2))
        return 0

    if args.command == "generate":
        if args.workers < 1 or args.generation_attempts < 1 or args.checkpoint_every < 1:
            parser.error("workers, generation-attempts and checkpoint-every must be positive")
        prompt = load_system_prompt(args.prompt)
        model = discover_model(args.base_url, args.model, args.timeout_seconds)
        client = QwenItemClient(
            base_url=args.base_url,
            model=model,
            system_prompt=prompt,
            timeout=args.timeout_seconds,
            retries=args.retries,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        summary = run_generation(
            index_dir=args.index_dir,
            output_dir=args.output_dir,
            client=client,
            system_prompt=prompt,
            count=args.count,
            seed=args.seed,
            id_start=args.id_start,
            example_count=args.examples,
            categories=args.categories,
            workers=args.workers,
            generation_attempts=args.generation_attempts,
            checkpoint_every=args.checkpoint_every,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["pending"] == 0 else 2

    if args.command == "validate":
        report = validate_generated_dataset(
            args.items,
            reference_path=None if args.no_reference else args.reference,
            metadata_path=args.metadata,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["valid"] else 2

    raise AssertionError(f"Unhandled command: {args.command}")
