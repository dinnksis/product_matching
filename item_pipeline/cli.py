from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .generate import run_generation
from .pair_generate import run_pair_generation
from .pair_validation import validate_pair_dataset
from .prepare import DEFAULT_EMBEDDING_MODEL, prepare_index
from .qwen import QwenItemClient, QwenPairClient, discover_model, load_system_prompt
from .validation import validate_generated_dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ITEMS = ROOT / "prepared" / "validation_splits_v1" / "llm" / "non_ood_items.parquet"
DEFAULT_INDEX = ROOT / "item_pipeline" / "artifacts" / "index"
DEFAULT_OUTPUT = ROOT / "item_pipeline" / "artifacts" / "generated"
DEFAULT_PROMPT = ROOT / "item_pipeline" / "prompts" / "generate_item.md"
DEFAULT_PAIR_OUTPUT = ROOT / "item_pipeline" / "artifacts" / "rule_first_pairs"
DEFAULT_PAIR_PROMPT = ROOT / "item_pipeline" / "prompts" / "mutate_item_by_rules.md"
DEFAULT_RARE_RULE_DIR = ROOT / "configs" / "generation_rule_catalog_rare_v1"
DEFAULT_RARE_RULES = [
    DEFAULT_RARE_RULE_DIR / "rare_rule_candidates_all.csv",
    DEFAULT_RARE_RULE_DIR / "rare_generation_rules_v1.json",
    DEFAULT_RARE_RULE_DIR / "rare_negative_rules_experimental.csv",
]


def read_env_value(path: Path, name: str) -> str:
    if value := os.environ.get(name):
        return value
    if not path.is_file():
        return ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value
    return ""


def provider_credentials(args: argparse.Namespace) -> tuple[str | None, str | None]:
    if not args.api_key_env:
        return None, args.model
    api_key = read_env_value(args.env_file, args.api_key_env)
    if not api_key:
        raise ValueError(
            f"Missing {args.api_key_env} in environment or {args.env_file}"
        )
    model = args.model or read_env_value(args.env_file, "MODEL") or None
    return api_key, model


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
    generate.add_argument("--env-file", type=Path, default=ROOT / ".env")
    generate.add_argument("--api-key-env")
    generate.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high", "xhigh", "max"),
    )
    generate.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    generate.add_argument("--workers", type=int, default=15)
    generate.add_argument("--timeout-seconds", type=float, default=120.0)
    generate.add_argument("--retries", type=int, default=8)
    generate.add_argument("--generation-attempts", type=int, default=3)
    generate.add_argument("--task-retries", type=int, default=3)
    generate.add_argument("--task-seed-offset", type=int, default=0)
    generate.add_argument("--checkpoint-every", type=int, default=25)
    generate.add_argument("--seed", type=int, default=20260820)
    generate.add_argument("--id-start", type=int, default=-1)
    generate.add_argument("--examples", type=int, default=5)
    generate.add_argument("--category", action="append", dest="categories")
    generate.add_argument("--temperature", type=float, default=0.7)
    generate.add_argument("--max-tokens", type=int, default=1400)
    generate.add_argument(
        "--plain-json",
        action="store_true",
        help="request JSON only in the prompt, without response_format/json_schema",
    )

    generate_pairs = subparsers.add_parser(
        "generate-pairs",
        help="Generate rule-applicable anchors, then mutate them into labelled pairs",
    )
    generate_pairs.add_argument(
        "--items",
        type=Path,
        default=DEFAULT_OUTPUT / "items.parquet",
        help="category style donors only; their product facts are not reused as pair anchors",
    )
    generate_pairs.add_argument("--rules", type=Path, action="append")
    generate_pairs.add_argument("--output-dir", type=Path, default=DEFAULT_PAIR_OUTPUT)
    generate_pairs.add_argument("--count", type=int, default=100)
    generate_pairs.add_argument("--base-url", default="http://localhost:8193/v1")
    generate_pairs.add_argument("--model")
    generate_pairs.add_argument("--env-file", type=Path, default=ROOT / ".env")
    generate_pairs.add_argument("--api-key-env")
    generate_pairs.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high", "xhigh", "max"),
    )
    generate_pairs.add_argument("--prompt", type=Path, default=DEFAULT_PAIR_PROMPT)
    generate_pairs.add_argument("--workers", type=int, default=10)
    generate_pairs.add_argument("--timeout-seconds", type=float, default=120.0)
    generate_pairs.add_argument("--retries", type=int, default=8)
    generate_pairs.add_argument("--pair-attempts", type=int, default=5)
    generate_pairs.add_argument("--anchor-attempts", type=int, default=3)
    generate_pairs.add_argument("--mutation-attempts", type=int, default=3)
    generate_pairs.add_argument("--task-retries", type=int, default=3)
    generate_pairs.add_argument("--task-seed-offset", type=int, default=0)
    generate_pairs.add_argument("--checkpoint-every", type=int, default=25)
    generate_pairs.add_argument("--seed", type=int, default=20260823)
    generate_pairs.add_argument("--mutated-id-start", type=int)
    generate_pairs.add_argument("--two-rule-fraction", type=float, default=0.5)
    generate_pairs.add_argument(
        "--label-one-fraction",
        type=float,
        help=(
            "exact requested fraction of target=1 tasks; omitted preserves the "
            "catalog-balanced scheduler"
        ),
    )
    generate_pairs.add_argument(
        "--semantic-signature-limit",
        type=int,
        default=2,
        help="maximum accepted pairs per category/product-type mutation signature",
    )
    generate_pairs.add_argument("--tier", action="append", dest="tiers")
    generate_pairs.add_argument("--category", action="append", dest="categories")
    generate_pairs.add_argument("--temperature", type=float, default=0.25)
    generate_pairs.add_argument("--max-tokens", type=int, default=1800)
    generate_pairs.add_argument(
        "--plain-json",
        action="store_true",
        help="request JSON only in the prompt, without response_format/json_schema",
    )

    validate = subparsers.add_parser("validate", help="Validate generated item artifacts")
    validate.add_argument("--items", type=Path, default=DEFAULT_OUTPUT / "items.parquet")
    validate.add_argument("--reference", type=Path, default=DEFAULT_INDEX / "exemplar_bank.parquet")
    validate.add_argument(
        "--metadata", type=Path, default=DEFAULT_OUTPUT / "generation_metadata.parquet"
    )
    validate.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "validation_report.json")
    validate.add_argument("--no-reference", action="store_true")

    validate_pairs = subparsers.add_parser(
        "validate-pairs", help="Validate generated rule-pair artifacts"
    )
    validate_pairs.add_argument("--items", type=Path, default=DEFAULT_PAIR_OUTPUT / "items.parquet")
    validate_pairs.add_argument("--pairs", type=Path, default=DEFAULT_PAIR_OUTPUT / "pairs.parquet")
    validate_pairs.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_PAIR_OUTPUT / "pair_generation_metadata.parquet",
    )
    validate_pairs.add_argument(
        "--output", type=Path, default=DEFAULT_PAIR_OUTPUT / "validation_report.json"
    )
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
        if (
            args.workers < 1
            or args.retries < 1
            or args.generation_attempts < 1
            or args.checkpoint_every < 1
            or args.task_retries < 0
        ):
            parser.error(
                "workers, retries, generation-attempts and checkpoint-every must be positive; "
                "task-retries must be non-negative"
            )
        prompt = load_system_prompt(args.prompt)
        try:
            api_key, requested_model = provider_credentials(args)
        except ValueError as error:
            parser.error(str(error))
        model = discover_model(args.base_url, requested_model, args.timeout_seconds)
        client = QwenItemClient(
            base_url=args.base_url,
            model=model,
            system_prompt=prompt,
            timeout=args.timeout_seconds,
            retries=args.retries,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            structured_output=not args.plain_json,
            api_key=api_key,
            reasoning_effort=args.reasoning_effort,
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
            task_retries=args.task_retries,
            task_seed_offset=args.task_seed_offset,
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

    if args.command == "generate-pairs":
        if (
            args.workers < 1
            or args.retries < 1
            or args.pair_attempts < 1
            or args.anchor_attempts < 1
            or args.mutation_attempts < 1
            or args.checkpoint_every < 1
            or args.task_retries < 0
            or args.semantic_signature_limit < 1
            or not 0.0 <= args.two_rule_fraction <= 1.0
            or (
                args.label_one_fraction is not None
                and not 0.0 <= args.label_one_fraction <= 1.0
            )
        ):
            parser.error(
                "workers, retries, pair-attempts, anchor-attempts, mutation-attempts and "
                "checkpoint-every must be positive; "
                "task-retries must be non-negative; semantic-signature-limit must "
                "be positive; two-rule-fraction and label-one-fraction must be "
                "in [0, 1]"
            )
        prompt = load_system_prompt(args.prompt)
        try:
            api_key, requested_model = provider_credentials(args)
        except ValueError as error:
            parser.error(str(error))
        model = discover_model(args.base_url, requested_model, args.timeout_seconds)
        client = QwenPairClient(
            base_url=args.base_url,
            model=model,
            system_prompt=prompt,
            timeout=args.timeout_seconds,
            retries=args.retries,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            structured_output=not args.plain_json,
            api_key=api_key,
            reasoning_effort=args.reasoning_effort,
        )
        summary = run_pair_generation(
            items_path=args.items,
            rule_paths=args.rules or DEFAULT_RARE_RULES,
            output_dir=args.output_dir,
            client=client,
            system_prompt=prompt,
            count=args.count,
            seed=args.seed,
            mutated_id_start=args.mutated_id_start,
            categories=args.categories,
            tiers=set(args.tiers) if args.tiers else None,
            workers=args.workers,
            pair_attempts=args.pair_attempts,
            anchor_attempts=args.anchor_attempts,
            mutation_attempts=args.mutation_attempts,
            checkpoint_every=args.checkpoint_every,
            two_rule_fraction=args.two_rule_fraction,
            task_retries=args.task_retries,
            task_seed_offset=args.task_seed_offset,
            semantic_signature_limit=args.semantic_signature_limit,
            label_one_fraction=args.label_one_fraction,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["pending"] == 0 and summary["validation_valid"] else 2

    if args.command == "validate-pairs":
        report = validate_pair_dataset(
            args.items,
            args.pairs,
            metadata_path=args.metadata,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["valid"] else 2

    raise AssertionError(f"Unhandled command: {args.command}")
