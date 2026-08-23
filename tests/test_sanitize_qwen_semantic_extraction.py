from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sanitize_qwen_semantic_extraction as sanitizer  # noqa: E402


def side(raw_value: str, fragment: str | None = None) -> dict:
    return {
        "value": {
            "raw_value": raw_value,
            "normalized_value": raw_value.casefold(),
            "unit": None,
        },
        "evidence": [
            {
                "source": "title",
                "raw_attribute_name": None,
                "raw_fragment": fragment or raw_value,
            }
        ],
    }


def fact(concept: str, relation: str, value_a: str, value_b: str) -> dict:
    return {
        "concept": concept,
        "relation": relation,
        "anchor_type": None,
        "a": side(value_a),
        "b": side(value_b),
        "direction": None,
        "confidence": "high",
    }


class FactSanitizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        schema = json.loads(
            (ROOT / "schemas" / "qwen_semantic_extraction_v1_3.schema.json").read_text(
                encoding="utf-8"
            )
        )
        cls.validator = sanitizer.fact_schema_validator(schema)

    def setUp(self) -> None:
        self.payload = {
            "category": "Электроника",
            "item_a": {"title": "Телефон 128 GB Black", "attributes": {}},
            "item_b": {"title": "Телефон 256 GB White", "attributes": {}},
        }

    def test_valid_difference_is_retained(self) -> None:
        candidate = fact("storage_capacity", "different_value", "128 GB", "256 GB")
        accepted, dropped, warnings = sanitizer.sanitize_facts(
            "pair", [candidate], self.payload, self.validator
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(dropped, [])
        self.assertEqual(warnings, [])

    def test_ordinary_same_is_dropped_without_losing_other_fact(self) -> None:
        ordinary_same = fact("color", "identity_same", "Black", "White")
        valid = fact("storage_capacity", "different_value", "128 GB", "256 GB")
        accepted, dropped, _ = sanitizer.sanitize_facts(
            "pair", [ordinary_same, valid], self.payload, self.validator
        )
        self.assertEqual([row["sanitized_fact"]["concept"] for row in accepted], ["storage_capacity"])
        self.assertEqual(dropped[0]["reason_codes"], ["ordinary_same_not_retained"])

    def test_normalized_placeholder_becomes_missing(self) -> None:
        candidate = fact("package_quantity", "different_value", "rings with hooks", "24")
        candidate["a"]["value"]["normalized_value"] = "unspecified"
        self.payload["item_a"]["title"] = "rings with hooks"
        self.payload["item_b"]["title"] = "24"
        accepted, dropped, warnings = sanitizer.sanitize_facts(
            "pair", [candidate], self.payload, self.validator
        )
        sanitized = accepted[0]["sanitized_fact"]
        self.assertEqual(sanitized["relation"], "missing_a")
        self.assertIsNone(sanitized["a"])
        self.assertEqual(dropped, [])
        self.assertIn("placeholder_transformed_facts:1", warnings)

    def test_bad_fact_does_not_discard_valid_fact(self) -> None:
        bad_anchor = fact("color", "identity_same", "Black", "White")
        bad_anchor["anchor_type"] = "other_identity"
        valid = fact("storage_capacity", "different_value", "128 GB", "256 GB")
        accepted, dropped, _ = sanitizer.sanitize_facts(
            "pair", [bad_anchor, valid], self.payload, self.validator
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(dropped), 1)
        self.assertIn("identity_values_differ", dropped[0]["reason_codes"])


if __name__ == "__main__":
    unittest.main()
