"""Submit MiniLM sweep notebooks two at a time as Kaggle GPU slots free up."""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
KAGGLE = ROOT / ".venv" / "Scripts" / "kaggle.exe"
RUNNER = ROOT / "scripts" / "run_kaggle_notebook.py"
NOTEBOOK_DIR = ROOT / "notebooks" / "minilm_5ep_hyperparam_sweep"
OWNER = "kristmakevel"
ACTIVE = {"RUNNING", "QUEUED", "INITIALIZING", "PENDING", "CANCEL_REQUESTED"}
TERMINAL = {"COMPLETE", "ERROR", "CANCELED", "CANCELLED", "FAILED"}


def load_env() -> None:
    for raw in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if not raw or raw.lstrip().startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        os.environ[key.strip()] = value.strip().strip("'\"")


def slug(notebook: Path) -> str:
    return notebook.stem.replace("_", "-")


def ref(notebook: Path) -> str:
    return f"{OWNER}/minilm-5ep-{slug(notebook)}"


def status(kernel_ref: str) -> str | None:
    result = subprocess.run(
        [str(KAGGLE), "kernels", "status", kernel_ref],
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    match = re.search(r'KernelWorkerStatus\.([A-Z_]+)', output)
    return match.group(1) if match else None


def submit(notebook: Path) -> None:
    subprocess.run(
        [
            str(PYTHON),
            str(RUNNER),
            str(notebook),
            "--env-file",
            str(ROOT / ".env"),
            "--slug",
            f"minilm-5ep-{slug(notebook)}",
            "--title",
            f"MiniLM 5ep {slug(notebook)}",
            "--dataset",
            "alexproger23/product-matching-validation-splits-v1",
            "--dataset",
            "alexproger23/product-matching-minilm-llm-pretrain-5ep",
            "--no-env-sources",
            "--no-wait",
            "--no-download",
            "--no-gpu-check",
        ],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    load_env()
    notebooks = sorted(NOTEBOOK_DIR.glob("sweep_*.ipynb"))
    submitted: set[Path] = set()
    while len(submitted) < len(notebooks):
        active = 0
        for notebook in notebooks:
            current = status(ref(notebook))
            if current in ACTIVE:
                submitted.add(notebook)
                active += 1
            elif current in TERMINAL:
                submitted.add(notebook)
        available = 2 - active
        for notebook in notebooks:
            if available <= 0:
                break
            if notebook in submitted:
                continue
            print(f"Submitting {notebook.name}", flush=True)
            submit(notebook)
            submitted.add(notebook)
            available -= 1
        if len(submitted) < len(notebooks):
            print(f"Queue monitor: {len(submitted)}/{len(notebooks)} submitted; active={active}", flush=True)
            time.sleep(30)
    print("All MiniLM sweep notebooks have been submitted.", flush=True)


if __name__ == "__main__":
    main()
