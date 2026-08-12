from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        items = pd.DataFrame(
            {
                "id": [1, 2],
                "name": ["Item A", "item-a"],
                "attributes": ["{}", "{}"],
                "category": ["demo", "demo"],
            }
        )
        matches = pd.DataFrame({"id1": [1], "id2": [2]})
        items_path, matches_path, output_path = (
            tmp / "items.parquet",
            tmp / "matches.parquet",
            tmp / "submit.csv",
        )
        items.to_parquet(items_path, index=False)
        matches.to_parquet(matches_path, index=False)
        subprocess.run(
            [
                sys.executable,
                str(root / "run.py"),
                "-i", str(items_path),
                "-m", str(matches_path),
                "-o", str(output_path),
            ],
            cwd=root,
            check=True,
        )
        result = pd.read_csv(output_path)
        assert result.columns.tolist() == ["id1", "id2", "predict"]
        assert len(result) == len(matches)
        print(f"Smoke test passed: {output_path}")


if __name__ == "__main__":
    main()

