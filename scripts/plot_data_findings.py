#!/usr/bin/env python3
"""Render the compact attribute signal chart used by docs/data-findings.md."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "attributes_analysis"


def main() -> None:
    source = REPORT_DIR / "family_conditions.csv"
    frame = pd.read_csv(source)
    plot = frame.pivot(index="family", columns="state", values="positive_rate")
    plot = plot.sort_values("conflict")

    figure, axis = plt.subplots(figsize=(10, 5.6))
    positions = range(len(plot))
    axis.barh(
        [value - 0.18 for value in positions],
        plot["match"],
        height=0.36,
        label="values overlap",
        color="#2563eb",
    )
    axis.barh(
        [value + 0.18 for value in positions],
        plot["conflict"],
        height=0.36,
        label="both present, no overlap",
        color="#dc2626",
    )
    axis.axvline(0.2567727961, color="#111827", linestyle="--", linewidth=1.2,
                 label="all-pair positive rate")
    axis.set_yticks(list(positions), plot.index)
    axis.set_xlabel("Observed positive rate among pairs satisfying the condition")
    axis.set_title("Attribute-family signals (human-labelled candidate pairs)")
    axis.legend(loc="lower right")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    destination = REPORT_DIR / "family_positive_rates.png"
    figure.savefig(destination, dpi=160)
    print(destination)


if __name__ == "__main__":
    main()
