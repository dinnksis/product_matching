from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_qwen_semantic_extraction as runner  # noqa: E402


def side(
    value: str,
    source: str = "title",
    attribute: str | None = None,
) -> dict:
    return {
        "value": value,
        "source": source,
        "attribute": attribute,
        "fragment": value,
    }


def fact(concept: str, value_a: str, value_b: str) -> dict:
    return {
        "concept": concept,
        "relation": "different_value",
        "a": side(value_a),
        "b": side(value_b),
    }


class CompactAtomicDifferencesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(
            (ROOT / "schemas" / "qwen_atomic_differences_compact_v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        cls.payload = {
            "category": "Электроника",
            "item_a": {"title": "телефон красный 2000 мл", "attributes": {}},
            "item_b": {"title": "телефон синий 2 л", "attributes": {}},
        }

    def test_bad_fact_does_not_drop_good_fact(self) -> None:
        good = fact("color", "красный", "синий")
        bad = fact("volume", "2000 мл", "2 л")
        clean, statistics = runner.sanitize_compact_extraction(
            {"differences": [good, bad]}, self.payload, self.schema
        )
        self.assertEqual([row["concept"] for row in clean["differences"]], ["color"])
        self.assertEqual(statistics["candidate_facts"], 2)
        self.assertEqual(statistics["accepted_facts"], 1)
        self.assertEqual(statistics["dropped_facts"], 1)

    def test_measurement_and_count_equivalence(self) -> None:
        self.assertTrue(
            runner.compact_values_equivalent(
                side("2000", "attribute", "объем, мл"), side("2 л")
            )
        )
        self.assertTrue(runner.compact_values_equivalent(side("6 шт."), side("5+1")))
        self.assertTrue(
            runner.compact_values_equivalent(side("разноцветный"), side("микс"))
        )
        self.assertFalse(runner.compact_values_equivalent(side("35"), side("39")))

    def test_normalization_keeps_stable_pair_shape(self) -> None:
        normalized = runner.normalize_extraction(
            {"differences": [fact("color", "красный", "синий")]}
        )
        self.assertEqual(normalized["identity_anchors"], [])
        self.assertEqual(normalized["missing_information"], [])
        self.assertEqual(normalized["differences"][0]["concept"], "color")
        self.assertEqual(
            normalized["differences"][0]["value_a"]["raw_value"], "красный"
        )


if __name__ == "__main__":
    unittest.main()
