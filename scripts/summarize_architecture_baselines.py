#!/usr/bin/env python3
"""Validate four downloaded Kaggle runs and build the comparison table."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_architecture_baseline_kaggle as runner


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "architecture_baselines"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kaggle-output-dir",
        type=Path,
        default=ROOT / "artifacts" / "kaggle",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small table without pandas' optional tabulate dependency."""
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        cells = [
            str(value).replace("|", "\\|").replace("\n", " ")
            for value in values
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_completion(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "notebook_completed.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing completion report: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        raise ValueError(f"Run is not complete: {path}")
    return payload


def result_row(profile: str, run_dir: Path) -> dict[str, Any]:
    completion = load_completion(run_dir)
    report = mapping(completion.get("training_report"))
    splits = mapping(report.get("validation_splits"))
    if set(splits) != {"iid", "hard", "ood"}:
        raise ValueError(f"Run {profile!r} does not contain exactly IID/hard/OOD")
    times = mapping(report.get("validation_seconds_by_split"))
    artifacts = mapping(completion.get("artifacts"))
    prediction_declarations = mapping(artifacts.get("predictions"))
    checkpoint = run_dir / str(artifacts.get("checkpoint", ""))
    predictions = {
        split: run_dir / str(prediction_declarations.get(split, ""))
        for split in ("iid", "hard", "ood")
    }
    missing = [str(path) for path in [checkpoint, *predictions.values()] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Run {profile!r} is missing declared artifacts: {missing}"
        )
    peak_values = report.get("peak_vram_gib_by_rank")
    peak_vram = (
        max(float(value) for value in peak_values if math.isfinite(float(value)))
        if isinstance(peak_values, list) and peak_values
        else None
    )
    row: dict[str, Any] = {
        "profile": profile,
        "architecture": completion.get("architecture"),
        "model": completion.get("model"),
        "initial_checkpoint_ref": completion.get("initial_checkpoint_ref"),
        "serialization": completion.get("serialization"),
        "iid_macro_ap": mapping(splits["iid"]).get("macro_average_precision"),
        "hard_macro_ap": mapping(splits["hard"]).get("macro_average_precision"),
        "ood_macro_ap": mapping(splits["ood"]).get("macro_average_precision"),
        "iid_overall_ap": mapping(splits["iid"]).get("overall_average_precision"),
        "hard_overall_ap": mapping(splits["hard"]).get("overall_average_precision"),
        "ood_overall_ap": mapping(splits["ood"]).get("overall_average_precision"),
        "iid_roc_auc": mapping(splits["iid"]).get("roc_auc"),
        "hard_roc_auc": mapping(splits["hard"]).get("roc_auc"),
        "ood_roc_auc": mapping(splits["ood"]).get("roc_auc"),
        "train_seconds": report.get("training_seconds"),
        "iid_inference_seconds": times.get("iid"),
        "hard_inference_seconds": times.get("hard"),
        "ood_inference_seconds": times.get("ood"),
        "peak_vram_gib": peak_vram,
        "checkpoint_path": str(checkpoint.resolve()),
        "iid_predictions_path": str(predictions["iid"].resolve()),
        "hard_predictions_path": str(predictions["hard"].resolve()),
        "ood_predictions_path": str(predictions["ood"].resolve()),
        "technical_notes": completion.get("technical_notes"),
    }
    return row


def main() -> None:
    args = parse_args()
    rows = [
        result_row(profile, args.kaggle_output_dir / str(spec["slug"]))
        for profile, spec in runner.PROFILES.items()
    ]
    frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "architecture_comparison.csv"
    json_path = args.output_dir / "architecture_comparison.json"
    markdown_path = args.output_dir / "architecture_comparison.md"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    display_columns = [
        "profile",
        "iid_macro_ap",
        "hard_macro_ap",
        "ood_macro_ap",
        "train_seconds",
        "iid_inference_seconds",
        "hard_inference_seconds",
        "ood_inference_seconds",
        "peak_vram_gib",
    ]
    markdown_path.write_text(
        "# Architecture baseline comparison\n\n"
        + markdown_table(frame[display_columns])
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {markdown_path}")


if __name__ == "__main__":
    main()
