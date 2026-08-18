from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.push_minilm_significance_baseline_dataset import (
    BASELINE_RUN_ID,
    COMPACT_COLUMNS,
    MANIFEST_FILENAME,
    PREDICTION_FILENAMES,
    build_payload,
    sha256_file,
)


class MinilmSignificanceBaselineDatasetTest(unittest.TestCase):
    def test_payload_contains_only_slim_frozen_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "source"
            stage_dir = root / "stage"
            source_dir.mkdir()
            for split_index, filename in enumerate(PREDICTION_FILENAMES.values()):
                frame = pd.DataFrame(
                    {
                        "id1": [1, 3, 5, 7],
                        "id2": [2, 4, 6, 8],
                        "target": [0.0, 1.0, 0.0, 1.0],
                        "category_1": [f"category_{split_index}"] * 4,
                        "score": [0.1, 0.9, 0.2, 0.8],
                        "product_text_1": ["must not be uploaded"] * 4,
                    }
                )
                frame.to_parquet(source_dir / filename, index=False)

            manifest = build_payload(source_dir, stage_dir, "alexproger23")

            self.assertEqual(manifest["baseline_run_id"], BASELINE_RUN_ID)
            self.assertTrue(manifest["is_private"])
            self.assertTrue((stage_dir / MANIFEST_FILENAME).is_file())
            for filename in PREDICTION_FILENAMES.values():
                staged = pd.read_parquet(stage_dir / filename)
                self.assertEqual(list(staged.columns), list(COMPACT_COLUMNS))
                declaration = manifest["files"][filename]
                self.assertEqual(declaration["rows"], 4)
                self.assertEqual(
                    declaration["sha256"],
                    sha256_file(stage_dir / filename),
                )


if __name__ == "__main__":
    unittest.main()
