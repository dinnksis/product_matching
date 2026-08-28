#!/usr/bin/env python3
"""Build, submit, wait for, and download one architecture baseline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import create_architecture_baseline_notebooks as builder
import run_kaggle_notebook as kaggle


ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "gte": {
        "notebook": "gte_architecture_baseline_2xt4.ipynb",
        "slug": "product-matching-architecture-gte-v1",
        "title": "Product Matching Architecture GTE v1",
    },
    "rumodernbert": {
        "notebook": "rumodernbert_architecture_baseline_2xt4.ipynb",
        "slug": "product-matching-architecture-rumodernbert-v1",
        "title": "Product Matching Architecture RuModernBERT v1",
    },
    "bge-v2-m3": {
        "notebook": "bge_v2_m3_architecture_baseline_2xt4.ipynb",
        "slug": "product-matching-architecture-bge-v2-m3-v1",
        "title": "Product Matching Architecture BGE v2 m3 v1",
    },
    "minilm-5ep": {
        "notebook": "minilm_5ep_architecture_baseline_2xt4.ipynb",
        "slug": "product-matching-architecture-minilm-5ep-v1",
        "title": "Product Matching Architecture MiniLM 5ep v1",
    },
}
EXPECTED_OUTPUTS = (
    "notebook_completed.json",
    "google_sheets_sync.json",
)
ESSENTIAL_OUTPUT_PATTERN = (
    r"(^|/)(notebook_completed\.json|google_sheets_sync\.json|"
    r"sheets_sync_pending\.json|training_report\.json|training_config\.json|"
    r"[^/]*_validation_predictions\.parquet|[^/]*\.log)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=[*PROFILES, "all"])
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit and return immediately; do not use with profile=all",
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--essential-only",
        action="store_true",
        help="Download predictions/reports/logs but skip checkpoint weights",
    )
    parser.add_argument(
        "--download-existing",
        action="store_true",
        help="Download outputs of an already completed kernel without resubmitting",
    )
    return parser.parse_args()


def selected_profiles(name: str) -> list[str]:
    return list(PROFILES) if name == "all" else [name]


def output_directory(slug: str) -> Path:
    root = Path(os.getenv("KAGGLE_OUTPUT_DIR", "artifacts/kaggle"))
    if not root.is_absolute():
        root = ROOT / root
    return root / slug


def validate_download(destination: Path) -> None:
    missing = [name for name in EXPECTED_OUTPUTS if not (destination / name).is_file()]
    completions = list(destination.rglob("training_report.json"))
    predictions = {
        split: list(destination.rglob(f"{split}_validation_predictions.parquet"))
        for split in ("iid", "hard", "ood")
    }
    if missing or len(completions) != 1 or any(len(paths) != 1 for paths in predictions.values()):
        raise SystemExit(
            "Downloaded architecture output is incomplete: "
            + json.dumps(
                {
                    "missing_root_files": missing,
                    "training_reports": [str(path) for path in completions],
                    "predictions": {
                        split: [str(path) for path in paths]
                        for split, paths in predictions.items()
                    },
                },
                ensure_ascii=False,
            )
        )


def download_existing(
    profile_name: str,
    owner: str,
    *,
    essential_only: bool = False,
) -> None:
    profile = PROFILES[profile_name]
    slug = str(profile["slug"])
    destination = output_directory(slug)
    destination.mkdir(parents=True, exist_ok=True)
    kernel_ref = f"{owner}/{slug}"
    command = kaggle.kaggle_command() + [
        "kernels",
        "output",
        kernel_ref,
        "-p",
        str(destination),
        "--force",
        "--page-size",
        "200",
    ]
    if essential_only:
        command.extend(["--file-pattern", ESSENTIAL_OUTPUT_PATTERN])
    kaggle.run_command(command)
    validate_download(destination)
    kind = "essential outputs" if essential_only else "all outputs"
    print(f"Downloaded and validated {kind}: {destination}")


def submit(profile_name: str, args: argparse.Namespace) -> None:
    profile = PROFILES[profile_name]
    notebook = builder.OUTPUT_DIR / str(profile["notebook"])
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_kaggle_notebook.py"),
        str(notebook),
        "--env-file",
        str(args.env_file),
        "--slug",
        str(profile["slug"]),
        "--title",
        str(profile["title"]),
        "--dataset",
        builder.VALIDATION_DATASET_REF,
        "--dataset",
        builder.RAW_DATASET_REF,
        "--no-env-sources",
    ]
    if profile_name == "minilm-5ep":
        checkpoint = builder.load_configuration()["profiles"][profile_name][
            "initial_checkpoint_dataset"
        ]
        command.extend(["--dataset", str(checkpoint)])
    if args.dry_run:
        command.append("--dry-run")
    if args.no_wait:
        command.append("--no-wait")
    if args.no_download or args.essential_only:
        command.append("--no-download")
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)
    if not args.dry_run and not args.no_wait and args.essential_only:
        download_existing(profile_name, owner, essential_only=True)
    elif not args.dry_run and not args.no_wait and not args.no_download:
        validate_download(output_directory(str(profile["slug"])))


def main() -> None:
    args = parse_args()
    if args.profile == "all" and args.no_wait:
        raise SystemExit("profile=all cannot be combined with --no-wait")
    if args.no_download and args.essential_only:
        raise SystemExit("--no-download and --essential-only cannot be combined")
    kaggle.load_dotenv(args.env_file)
    owner = os.getenv("KAGGLE_USERNAME", "").strip()
    if not owner:
        raise SystemExit("Set KAGGLE_USERNAME in .env")
    # Regenerate all files so submitted notebook content always matches source.
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/create_architecture_baseline_notebooks.py")],
        check=True,
        cwd=ROOT,
    )
    for profile_name in selected_profiles(args.profile):
        if args.download_existing:
            download_existing(
                profile_name,
                owner,
                essential_only=args.essential_only,
            )
        else:
            submit(profile_name, args)


if __name__ == "__main__":
    main()
