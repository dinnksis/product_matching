from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.experiment_protocol import validation_split_paths


class ValidationSplitPathsTest(unittest.TestCase):
    def test_three_named_relative_splits_are_resolved_below_prepared_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prepared = Path(directory)
            actual = validation_split_paths(
                prepared,
                [
                    "iid=iid.parquet",
                    "hard=hard.parquet",
                    "ood=ood.parquet",
                ],
            )

        self.assertEqual(
            actual,
            {
                "iid": prepared / "iid.parquet",
                "hard": prepared / "hard.parquet",
                "ood": prepared / "ood.parquet",
            },
        )

    def test_legacy_default_is_iid_val_pairs(self) -> None:
        self.assertEqual(
            validation_split_paths(Path("prepared/human"), []),
            {"iid": Path("prepared/human/val_pairs.parquet")},
        )

    def test_duplicate_split_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate validation split"):
            validation_split_paths(
                Path("prepared"),
                ["iid=first.parquet", "iid=second.parquet"],
            )


if __name__ == "__main__":
    unittest.main()
