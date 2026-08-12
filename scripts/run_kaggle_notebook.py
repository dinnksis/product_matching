#!/usr/bin/env python3
"""Push a local notebook to Kaggle, wait for it, and download its outputs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import NoReturn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = ROOT / ".env"
STAGE_ROOT = ROOT / ".kaggle" / "staging"

TERMINAL_SUCCESS = {"complete", "completed"}
TERMINAL_FAILURE = {
    "error",
    "failed",
    "failure",
    "cancelrequested",
    "cancelacknowledged",
    "cancelled",
    "canceled",
}

GPU_PREFLIGHT_SOURCE = """# Added to the uploaded copy by run_kaggle_notebook.py.
import subprocess as _kaggle_subprocess
import torch as _kaggle_torch

_kaggle_gpu_count = _kaggle_torch.cuda.device_count()
_kaggle_gpu_names = [
    _kaggle_torch.cuda.get_device_name(index)
    for index in range(_kaggle_gpu_count)
]
print(_kaggle_subprocess.run(
    [\"nvidia-smi\", \"--query-gpu=index,name,memory.total\", \"--format=csv,noheader\"],
    check=False,
    capture_output=True,
    text=True,
).stdout)
if _kaggle_gpu_count != 2 or any(\"T4\" not in name.upper() for name in _kaggle_gpu_names):
    raise RuntimeError(
        f\"Expected exactly 2 NVIDIA T4 GPUs, got {_kaggle_gpu_count}: {_kaggle_gpu_names}\"
    )
del _kaggle_subprocess, _kaggle_torch, _kaggle_gpu_count, _kaggle_gpu_names
"""


def fail(message: str, exit_code: int = 2) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def load_dotenv(path: Path) -> None:
    """Load a deliberately small, non-executing subset of dotenv syntax."""
    if not path.exists():
        fail(f"environment file does not exist: {path}")

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            fail(f"invalid .env line {line_number}: expected KEY=VALUE")

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            fail(f"invalid variable name on .env line {line_number}: {key!r}")

        if value.startswith(("'", '"')):
            quote = value[0]
            if len(value) < 2 or not value.endswith(quote):
                fail(f"unclosed quote on .env line {line_number}")
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        os.environ.setdefault(key, value)


def env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    value = raw_value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    fail(f"{name} must be true or false, got {raw_value!r}")


def env_int(name: str, default: int, minimum: int = 1) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError:
        fail(f"{name} must be an integer, got {raw_value!r}")
    if value < minimum:
        fail(f"{name} must be at least {minimum}, got {value}")
    return value


def split_sources(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in re.split(r"[\s,]+", value.strip()) if item]


def validate_slug(value: str, label: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", value):
        fail(f"{label} must contain lowercase letters, digits, and hyphens: {value!r}")
    return value


def kaggle_command() -> list[str]:
    local_executable = ROOT / ".venv" / "bin" / "kaggle"
    if local_executable.is_file():
        return [str(local_executable)]
    executable = shutil.which("kaggle")
    if executable:
        return [executable]
    uv = shutil.which("uv")
    if uv:
        return [uv, "run", "kaggle"]
    fail("Kaggle CLI is unavailable; run `uv sync` first")


def run_command(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    printable = " ".join(command)
    print(f"$ {printable}", flush=True)
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if check and result.returncode:
        fail(f"command failed with exit code {result.returncode}", result.returncode)
    return result


def prepare_notebook(source: Path, destination: Path, *, gpu_check: bool) -> None:
    try:
        notebook = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        fail(f"invalid notebook JSON in {source}: {error}")

    if not isinstance(notebook.get("cells"), list):
        fail(f"notebook has no cells array: {source}")

    # Local outputs only make the upload larger. The source file is never modified.
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    if gpu_check:
        notebook["cells"].insert(
            0,
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"tags": ["kaggle-gpu-preflight"]},
                "outputs": [],
                "source": GPU_PREFLIGHT_SOURCE.splitlines(keepends=True),
            },
        )

    destination.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def extract_status(output: str) -> str | None:
    lowered = output.lower()
    quoted = re.findall(
        r"status\s+['\"](?:kernelworkerstatus\.)?([a-z_ -]+)['\"]",
        lowered,
    )
    candidates = quoted + re.findall(r"\b(?:status\s*[:=]\s*)([a-z_ -]+)", lowered)
    if candidates:
        return re.sub(r"[\s_-]+", "", candidates[-1])

    for status in TERMINAL_SUCCESS | TERMINAL_FAILURE | {"queued", "running"}:
        if re.search(rf"\b{re.escape(status)}\b", lowered):
            return status
    return None


def wait_for_kernel(
    cli: list[str],
    kernel_ref: str,
    *,
    poll_interval: int,
    wait_timeout: int,
) -> None:
    deadline = time.monotonic() + wait_timeout
    consecutive_errors = 0
    last_status: str | None = None

    while time.monotonic() < deadline:
        result = run_command(cli + ["kernels", "status", kernel_ref], check=False)
        if result.returncode:
            consecutive_errors += 1
            if consecutive_errors >= 5:
                fail("could not read Kaggle kernel status five times in a row")
        else:
            consecutive_errors = 0
            status = extract_status(result.stdout)
            if status and status != last_status:
                print(f"Kaggle status: {status}", flush=True)
                last_status = status
            if status in TERMINAL_SUCCESS:
                return
            if status in TERMINAL_FAILURE:
                print("Kaggle run failed; fetching its log...", file=sys.stderr)
                run_command(cli + ["kernels", "logs", kernel_ref], check=False)
                fail(f"Kaggle kernel finished with status {status!r}", 1)

        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(poll_interval, remaining))

    fail(
        f"local wait timed out after {wait_timeout} seconds; "
        f"the remote run may still be active at https://www.kaggle.com/code/{kernel_ref}",
        124,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a local .ipynb on Kaggle and download /kaggle/working outputs."
    )
    parser.add_argument("notebook", type=Path, help="path to the local .ipynb")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--slug", help="override KAGGLE_KERNEL_SLUG")
    parser.add_argument("--title", help="override KAGGLE_KERNEL_TITLE")
    parser.add_argument("--dataset", action="append", default=[], help="attach owner/dataset")
    parser.add_argument("--competition", action="append", default=[], help="attach competition slug")
    parser.add_argument("--no-wait", action="store_true", help="return immediately after push")
    parser.add_argument("--no-download", action="store_true", help="do not download outputs")
    parser.add_argument("--no-gpu-check", action="store_true", help="do not assert that two T4s exist")
    parser.add_argument(
        "--no-env-sources",
        action="store_true",
        help="ignore dataset/competition/kernel/model sources from .env",
    )
    parser.add_argument("--dry-run", action="store_true", help="prepare files but do not call Kaggle")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    env_file = args.env_file if args.env_file.is_absolute() else ROOT / args.env_file
    load_dotenv(env_file)

    notebook = args.notebook if args.notebook.is_absolute() else ROOT / args.notebook
    notebook = notebook.resolve()
    if not notebook.is_file() or notebook.suffix.lower() != ".ipynb":
        fail(f"notebook does not exist or is not an .ipynb file: {notebook}")

    token = os.getenv("KAGGLE_API_TOKEN", "").strip()
    username = os.getenv("KAGGLE_USERNAME", "").strip()
    if not username:
        fail("set KAGGLE_USERNAME in .env")
    username = validate_slug(username, "KAGGLE_USERNAME")
    if not args.dry_run and not token:
        fail("set KAGGLE_API_TOKEN in .env; its value is never uploaded to the notebook")

    slug = validate_slug(
        (args.slug or os.getenv("KAGGLE_KERNEL_SLUG", "product-matching-training")).strip(),
        "KAGGLE_KERNEL_SLUG",
    )
    title = (args.title or os.getenv("KAGGLE_KERNEL_TITLE", "Product Matching Training")).strip()
    if not title:
        fail("KAGGLE_KERNEL_TITLE cannot be empty")

    accelerator = os.getenv("KAGGLE_ACCELERATOR", "NvidiaTeslaT4").strip()
    internet_enabled = env_bool("KAGGLE_ENABLE_INTERNET", True)
    is_private = env_bool("KAGGLE_IS_PRIVATE", True)
    run_timeout = env_int("KAGGLE_RUN_TIMEOUT_SECONDS", 43200, minimum=60)
    wait_timeout = env_int("KAGGLE_WAIT_TIMEOUT_SECONDS", 45000, minimum=60)
    poll_interval = env_int("KAGGLE_POLL_INTERVAL_SECONDS", 30, minimum=5)

    env_datasets = [] if args.no_env_sources else split_sources(os.getenv("KAGGLE_DATASET_SOURCES"))
    env_competitions = (
        [] if args.no_env_sources else split_sources(os.getenv("KAGGLE_COMPETITION_SOURCES"))
    )
    datasets = env_datasets + args.dataset
    competitions = env_competitions + args.competition
    kernel_sources = (
        [] if args.no_env_sources else split_sources(os.getenv("KAGGLE_KERNEL_SOURCES"))
    )
    model_sources = (
        [] if args.no_env_sources else split_sources(os.getenv("KAGGLE_MODEL_SOURCES"))
    )

    stage_dir = STAGE_ROOT / slug
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged_notebook = stage_dir / "notebook.ipynb"
    prepare_notebook(notebook, staged_notebook, gpu_check=not args.no_gpu_check)

    kernel_ref = f"{username}/{slug}"
    metadata = {
        "id": kernel_ref,
        "title": title,
        "code_file": staged_notebook.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": is_private,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": internet_enabled,
        "machine_shape": accelerator,
        "dataset_sources": datasets,
        "competition_sources": competitions,
        "kernel_sources": kernel_sources,
        "model_sources": model_sources,
    }
    (stage_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Prepared: {stage_dir}")
    print(f"Kernel:   https://www.kaggle.com/code/{kernel_ref}")
    print(f"GPU:      {accelerator} (the staged notebook asserts exactly 2 x T4)")
    print(f"Sources:  {len(datasets)} dataset(s), {len(competitions)} competition(s)")
    if args.dry_run:
        print("Dry run complete; Kaggle was not contacted.")
        return 0

    cli = kaggle_command()
    print("Checking Kaggle credentials...")
    run_command(cli + ["kernels", "list", "--mine", "--page-size", "1"])
    run_command(
        cli
        + [
            "kernels",
            "push",
            "-p",
            str(stage_dir),
            "--accelerator",
            accelerator,
            "--timeout",
            str(run_timeout),
        ]
    )

    if args.no_wait:
        print("Notebook was submitted; not waiting for completion.")
        return 0

    try:
        wait_for_kernel(
            cli,
            kernel_ref,
            poll_interval=poll_interval,
            wait_timeout=wait_timeout,
        )
    except KeyboardInterrupt:
        print(
            f"\nStopped waiting locally. The Kaggle run may continue: "
            f"https://www.kaggle.com/code/{kernel_ref}",
            file=sys.stderr,
        )
        return 130

    print("Kaggle run completed successfully.")
    if args.no_download:
        return 0

    output_root = Path(os.getenv("KAGGLE_OUTPUT_DIR", "artifacts/kaggle"))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_dir = output_root / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        cli
        + [
            "kernels",
            "output",
            kernel_ref,
            "-p",
            str(output_dir),
            "--force",
            "--page-size",
            "200",
        ]
    )
    print(f"Outputs downloaded to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
