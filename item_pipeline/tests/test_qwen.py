from __future__ import annotations

import json
import unittest

from item_pipeline.qwen import generated_item_schema, parse_generated_item


class QwenSchemaTest(unittest.TestCase):
    def test_generated_schema_and_parser_require_exact_keys_but_restore_order(self) -> None:
        schema = generated_item_schema("Электроника", ["тип", "бренд"])
        self.assertEqual(schema["properties"]["category"]["const"], "Электроника")
        response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "name": "новый картридж",
                                "attributes": {"бренд": "tonex", "тип": "картридж"},
                                "category": "Электроника",
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8},
        }
        parsed = parse_generated_item(
            response,
            expected_category="Электроника",
            expected_keys=["тип", "бренд"],
        )
        self.assertEqual(list(parsed["item"]["attributes"]), ["тип", "бренд"])


if __name__ == "__main__":
    unittest.main()
