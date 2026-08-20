from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from item_pipeline.prepare import build_exemplar_bank


class PrepareTest(unittest.TestCase):
    def test_streaming_sample_is_deterministic_and_capped_per_category(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "items.parquet"
            rows = []
            for category in ("Электроника", "Автотовары"):
                for index in range(12):
                    rows.append(
                        {
                            "id": len(rows) + 1,
                            "name": f"товар {index}",
                            "attributes": json.dumps(
                                {"тип": "тестовый товар", "модель": f"m{index}"},
                                ensure_ascii=False,
                            ),
                            "category": category,
                        }
                    )
            pd.DataFrame(rows).to_parquet(path, index=False)
            first, profile = build_exemplar_bank(
                path, max_items_per_category=5, seed=91, progress_every=1000
            )
            second, _ = build_exemplar_bank(
                path, max_items_per_category=5, seed=91, progress_every=1000
            )
            self.assertEqual(first["id"].tolist(), second["id"].tolist())
            self.assertEqual(first["category"].value_counts().to_dict(), {
                "Автотовары": 5,
                "Электроника": 5,
            })
            self.assertEqual(profile["bank_rows"], 10)


if __name__ == "__main__":
    unittest.main()
