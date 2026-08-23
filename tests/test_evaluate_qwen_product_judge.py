from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_qwen_product_judge.py"
SPEC = importlib.util.spec_from_file_location("evaluate_qwen_product_judge", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EvaluateQwenProductJudgeTests(unittest.TestCase):
    def test_binary_softmax(self) -> None:
        self.assertTrue(math.isclose(MODULE.binary_softmax(-2.0, -2.0), 0.5))
        self.assertGreater(MODULE.binary_softmax(-8.0, -1.0), 0.999)
        self.assertLess(MODULE.binary_softmax(-1.0, -8.0), 0.001)

    def test_parse_choice_uses_forced_binary_logprobs(self) -> None:
        response = {
            "id": "answer-id",
            "choices": [
                {
                    "message": {"content": "1"},
                    "logprobs": {
                        "content": [
                            {
                                "token": "1",
                                "logprob": -0.2,
                                "top_logprobs": [
                                    {"token": "1", "logprob": -0.2},
                                    {"token": "0", "logprob": -1.7},
                                ],
                            }
                        ]
                    },
                }
            ],
            "usage": {"prompt_tokens": 123, "completion_tokens": 2},
        }
        parsed = MODULE.parse_choice(response)
        self.assertEqual(parsed["answer"], 1)
        self.assertTrue(
            math.isclose(parsed["predict"], 1 / (1 + math.exp(-1.5)))
        )
        self.assertEqual(parsed["prompt_tokens"], 123)
        self.assertEqual(parsed["completion_tokens"], 2)

    def test_user_message_does_not_include_target(self) -> None:
        message = MODULE.build_user_message(
            {
                "category": "Электроника",
                "product_text_1": "Название: телефон A",
                "product_text_2": "Название: телефон B",
                "target": 1.0,
            }
        )
        self.assertNotIn("target", message)
        self.assertNotIn("1.0", message)
        self.assertIn("<CARD_A>", message)
        self.assertIn("<CARD_B>", message)


if __name__ == "__main__":
    unittest.main()
