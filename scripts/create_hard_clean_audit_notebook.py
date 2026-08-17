#!/usr/bin/env python3
"""Generate the reproducible local notebook for the clean-hard label audit."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/05_hard_clean_audit.ipynb"


def markdown(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(text).strip())


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook["cells"] = [
        markdown(
            """
            # Hard-label audit and clean-hard benchmark

            This notebook does not train or calibrate any model. It audits the fixed
            hard labels, builds mutually exclusive `hard_clean`, `hard_suspicious`,
            and `hard_conflicting` subsets, and evaluates frozen S0, S2, and
            S2+CatBoost predictions.

            Subset selection is independent of model predictions. Scores are attached
            only after the label/representation flags have been computed.
            """
        ),
        code(
            """
            from pathlib import Path
            import subprocess
            import sys

            import pandas as pd
            from IPython.display import display

            ROOT = Path.cwd()
            if not (ROOT / "scripts/audit_hard_clean.py").is_file():
                ROOT = ROOT.parent
            OUTPUT = ROOT / "reports/minilm_s2_hard_clean_audit"
            assert (ROOT / "scripts/audit_hard_clean.py").is_file(), ROOT
            """
        ),
        markdown("## Run the deterministic audit"),
        code(
            """
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts/audit_hard_clean.py")],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            if completed.returncode:
                print(completed.stdout)
                print(completed.stderr, file=sys.stderr)
                completed.check_returncode()
            print("Audit completed:", OUTPUT)
            """
        ),
        markdown(
            """
            ## Objective contradictions and benchmark subsets

            `definite_label_conflict` means opposite targets for the same unordered
            ID pair or for the same unordered pair of normalized full item
            representations. Suspicious flags are not automatic label corrections.
            """
        ),
        code(
            """
            audit_summary = pd.read_csv(OUTPUT / "label_audit_summary.csv")
            display(audit_summary)
            """
        ),
        markdown("## Frozen-model PR-AUC"),
        code(
            """
            metrics = pd.read_csv(OUTPUT / "benchmark_metrics.csv")
            display(
                metrics.pivot(
                    index=["subset", "pairs", "prevalence"],
                    columns="model",
                    values="macro_average_precision",
                ).reset_index()
            )
            display(
                metrics.pivot(
                    index=["subset", "pairs", "prevalence"],
                    columns="model",
                    values="macro_roc_auc",
                ).reset_index()
            )
            """
        ),
        markdown(
            """
            Absolute PR-AUC values must not be compared naively across subsets with
            different prevalence. The primary model comparison is within each row.
            ROC-AUC is shown as a prevalence-insensitive ranking diagnostic.
            """
        ),
        markdown("## Clean-hard diagnostic slices"),
        code(
            """
            clean_slices = pd.read_csv(OUTPUT / "hard_clean_slice_metrics.csv")
            display(clean_slices)
            """
        ),
        markdown("## Up to 100 examples per audit issue"),
        code(
            """
            contradictions = pd.read_csv(OUTPUT / "label_contradictions.csv")
            display(contradictions.groupby("audit_issue").size().rename("examples"))
            display(contradictions.head(100))
            """
        ),
        markdown(
            """
            ## Interpretation

            A higher clean-hard ROC-AUC than hard-all indicates that ambiguous labels
            materially damage ranking evaluation. If clean-hard remains far below IID,
            the hard-set failure is not explained by label noise alone: selection and
            genuinely difficult candidate pairs remain important.
            """
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
