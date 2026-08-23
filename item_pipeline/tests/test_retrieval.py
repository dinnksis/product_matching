from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from item_pipeline.retrieval import HybridRetriever, bm25_scores


def make_bank() -> pd.DataFrame:
    names = [
        "canon картридж pg440 черный",
        "картридж canon pg445 black",
        "epson бумага a4",
    ]
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": names,
            "attributes": [json.dumps({"тип": "картридж"})] * 3,
            "category": ["Электроника"] * 3,
            "subtype": ["картридж", "картридж", "бумага"],
            "retrieval_text": names,
        }
    )


class RetrievalTest(unittest.TestCase):
    def test_bm25_and_hybrid_retrieval_prefer_same_product_type(self) -> None:
        scores = bm25_scores("canon картридж", ["canon картридж", "epson бумага"])
        self.assertGreater(scores[0], scores[1])
        embeddings = np.asarray(
            [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]], dtype=np.float32
        )
        retriever = HybridRetriever(make_bank(), embeddings, min_subtype_pool=2)
        self.assertEqual(retriever.retrieve(0, k=1), [1])


if __name__ == "__main__":
    unittest.main()
