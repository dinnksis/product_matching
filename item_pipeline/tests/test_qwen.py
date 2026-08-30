from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import requests

from item_pipeline.pair_rules import MutationRule
from item_pipeline.qwen import (
    QwenItemClient,
    QwenPairClient,
    _request_error_detail,
    generated_item_schema,
    mutated_item_schema,
    parse_generated_item,
    parse_mutated_item,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class ErrorResponse(FakeResponse):
    def __init__(self, status_code, payload=None, headers=None):
        super().__init__({} if payload is None else payload)
        self.status_code = status_code
        self.headers = {} if headers is None else headers
        self.text = ""

    def raise_for_status(self):
        error = requests.HTTPError(f"status={self.status_code}")
        error.response = self
        raise error


class CaptureSession:
    def __init__(self, response):
        self.response = response
        self.payloads = []

    def post(self, url, *, json, timeout):
        del url, timeout
        self.payloads.append(json)
        return FakeResponse(self.response)


class SequenceSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []

    def post(self, url, *, json, timeout):
        del url, timeout
        self.payloads.append(json)
        return FakeResponse(self.responses.pop(0))


class ErrorSession:
    def __init__(self, status_code):
        self.status_code = status_code
        self.payloads = []

    def post(self, url, *, json, timeout):
        del url, timeout
        self.payloads.append(json)
        return ErrorResponse(self.status_code)


class QwenSchemaTest(unittest.TestCase):
    def test_http_error_detail_reports_provider_reason_and_redacts_keys(self) -> None:
        response = ErrorResponse(
            429,
            payload={
                "error": {
                    "code": 429,
                    "message": "Rate limited for sk-or-v1-supersecretvalue",
                    "metadata": {
                        "provider_name": "Example Provider",
                        "raw": "Bearer another-secret-token quota exhausted",
                    },
                }
            },
            headers={"Retry-After": "12", "x-request-id": "request-123"},
        )
        error = requests.HTTPError("429")
        error.response = response
        detail = _request_error_detail(error)
        self.assertIn('"http_status": 429', detail)
        self.assertIn('"retry_after": "12"', detail)
        self.assertIn('"provider_name": "Example Provider"', detail)
        self.assertNotIn("supersecretvalue", detail)
        self.assertNotIn("another-secret-token", detail)

    def test_openrouter_permanent_http_error_is_not_retried(self) -> None:
        session = ErrorSession(401)
        client = QwenItemClient(
            base_url="https://openrouter.ai/api/v1",
            model="test",
            system_prompt="return json",
            structured_output=False,
            api_key="bad-secret",
            reasoning_effort="low",
            retries=3,
        )
        client.session = lambda: session
        with self.assertRaisesRegex(RuntimeError, "failed after 1 attempts"):
            client.generate(
                "prompt", category="Электроника", attribute_keys=["тип"], seed=1
            )
        self.assertEqual(len(session.payloads), 1)

    def test_openrouter_auth_and_low_reasoning_plain_json_payload(self) -> None:
        item_response = {
            "choices": [{"message": {"content": json.dumps({
                "name": "кабель",
                "attributes": {"тип": "кабель"},
                "category": "Электроника",
            }, ensure_ascii=False)}}]
        }
        client = QwenItemClient(
            base_url="https://openrouter.ai/api/v1",
            model="z-ai/glm-5.3-flash",
            system_prompt="return json",
            structured_output=False,
            api_key="test-secret",
            reasoning_effort="low",
        )
        authenticated_session = client.session()
        self.assertTrue(authenticated_session.trust_env)
        self.assertEqual(
            authenticated_session.headers["Authorization"], "Bearer test-secret"
        )
        self.assertEqual(
            authenticated_session.headers["X-Title"], "product_matching"
        )
        session = CaptureSession(item_response)
        client.session = lambda: session
        client.generate(
            "prompt", category="Электроника", attribute_keys=["тип"], seed=1
        )
        payload = session.payloads[0]
        self.assertEqual(payload["reasoning"], {"effort": "low", "exclude": True})
        self.assertNotIn("chat_template_kwargs", payload)
        self.assertNotIn("response_format", payload)
        self.assertNotIn("test-secret", json.dumps(payload, ensure_ascii=False))

    def test_pair_client_changes_sampling_seed_on_internal_retry(self) -> None:
        rule = MutationRule(
            generation_rule_id="rule-color",
            source_rule_id="source-color",
            generation_tier="RARE_SAFE",
            label=0,
            concept="color",
            relation="different_value",
            semantic_family="categorical_variant",
            attribute_key="цвет",
            anchor_hint="",
            allowed_categories=("Электроника",),
            generation_action="change color",
            required_postcondition="different explicit color",
            source_path="test",
        )
        invalid = {"choices": [{"message": {"content": "{}"}}]}
        valid = {
            "choices": [{"message": {"content": json.dumps({
                "item": {
                    "name": "синий кабель",
                    "attributes": {"тип": "кабель", "цвет": "синий"},
                    "category": "Электроника",
                },
                "applications": [{
                    "generation_rule_id": "rule-color",
                    "concept": "color",
                    "original_value": "красный",
                    "new_value": "синий",
                    "attribute_key": "цвет",
                }],
            }, ensure_ascii=False)}}]
        }
        session = SequenceSession([invalid, valid])
        client = QwenPairClient(
            base_url="http://test/v1",
            model="test",
            system_prompt="return json",
            structured_output=False,
            retries=2,
        )
        client.session = lambda: session
        with patch("item_pipeline.qwen.time.sleep"):
            result = client.mutate(
                "prompt",
                category="Электроника",
                attribute_keys=["тип", "цвет"],
                rules=[rule],
                seed=41,
            )
        self.assertEqual(result["request_attempts"], 2)
        self.assertEqual([payload["seed"] for payload in session.payloads], [41, 42])

        valid_anchor = {
            "choices": [{"message": {"content": json.dumps({
                "item": {
                    "name": "кабель test красный",
                    "attributes": {
                        "Тип товара": "кабель",
                        "Бренд": "test",
                        "цвет": "красный",
                    },
                    "category": "Электроника",
                },
                "product_type": "кабель",
                "evidence": [{
                    "generation_rule_id": "rule-color",
                    "concept": "color",
                    "attribute_key": "цвет",
                    "attribute_value": "красный",
                }],
            }, ensure_ascii=False)}}]
        }
        anchor_session = SequenceSession([invalid, valid_anchor])
        client.session = lambda: anchor_session
        with patch("item_pipeline.qwen.time.sleep"):
            anchor_result = client.generate_anchor(
                "prompt",
                category="Электроника",
                rules=[rule],
                seed=71,
            )
        self.assertEqual(anchor_result["request_attempts"], 2)
        self.assertEqual(
            [payload["seed"] for payload in anchor_session.payloads], [71, 72]
        )

    def test_plain_json_clients_omit_response_format(self) -> None:
        item_response = {
            "choices": [{"message": {"content": """```json
{"name":"кабель","attributes":{"тип":"кабель"},"category":"Электроника"}
```"""}}]
        }
        item_session = CaptureSession(item_response)
        item_client = QwenItemClient(
            base_url="http://test/v1",
            model="test",
            system_prompt="return json",
            structured_output=False,
        )
        self.assertFalse(item_client.session().trust_env)
        item_client.session = lambda: item_session
        item_client.generate(
            "prompt", category="Электроника", attribute_keys=["тип"], seed=1
        )
        self.assertNotIn("response_format", item_session.payloads[0])
        self.assertEqual(
            item_session.payloads[0]["chat_template_kwargs"],
            {"enable_thinking": False},
        )

        rule = MutationRule(
            generation_rule_id="rule-color",
            source_rule_id="source-color",
            generation_tier="RARE_SAFE",
            label=0,
            concept="color",
            relation="different_value",
            semantic_family="categorical_variant",
            attribute_key="цвет",
            anchor_hint="",
            allowed_categories=("Электроника",),
            generation_action="change color",
            required_postcondition="different explicit color",
            source_path="test",
        )
        pair_response = {
            "choices": [{"message": {"content": json.dumps({
                "item": {
                    "name": "синий кабель",
                    "attributes": {"тип": "кабель", "цвет": "синий"},
                    "category": "Электроника",
                },
                "applications": [{
                    "generation_rule_id": "rule-color",
                    "concept": "color",
                    "original_value": "красный",
                    "new_value": "синий",
                    "attribute_key": "цвет",
                }],
            }, ensure_ascii=False)}}]
        }
        pair_session = CaptureSession(pair_response)
        pair_client = QwenPairClient(
            base_url="http://test/v1",
            model="test",
            system_prompt="return json",
            structured_output=False,
        )
        pair_client.session = lambda: pair_session
        pair_client.mutate(
            "prompt",
            category="Электроника",
            attribute_keys=["тип", "цвет"],
            rules=[rule],
            seed=1,
        )
        self.assertNotIn("response_format", pair_session.payloads[0])

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

    def test_mutation_schema_and_parser_require_every_requested_rule(self) -> None:
        rule = MutationRule(
            generation_rule_id="rule-color",
            source_rule_id="source-color",
            generation_tier="RARE_SAFE",
            label=0,
            concept="color",
            relation="different_value",
            semantic_family="categorical_variant",
            attribute_key="цвет",
            anchor_hint="",
            allowed_categories=("Электроника",),
            generation_action="change color",
            required_postcondition="different explicit color",
            source_path="test",
        )
        schema = mutated_item_schema("Электроника", ["тип", "цвет"], [rule])
        applications = schema["properties"]["applications"]
        self.assertEqual(applications["minItems"], 1)
        self.assertEqual(
            applications["items"]["properties"]["generation_rule_id"]["enum"],
            ["rule-color"],
        )
        response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "item": {
                                    "name": "синий кабель",
                                    "attributes": {"цвет": "синий", "тип": "кабель"},
                                    "category": "Электроника",
                                },
                                "applications": [
                                    {
                                        "generation_rule_id": "rule-color",
                                        "concept": "color",
                                        "original_value": "красный",
                                        "new_value": "синий",
                                        "attribute_key": "цвет",
                                    }
                                ],
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15},
        }
        parsed = parse_mutated_item(
            response,
            expected_category="Электроника",
            expected_keys=["тип", "цвет"],
            rules=[rule],
        )
        self.assertEqual(list(parsed["item"]["attributes"]), ["тип", "цвет"])
        self.assertEqual(parsed["applications"][0]["new_value"], "синий")


if __name__ == "__main__":
    unittest.main()
