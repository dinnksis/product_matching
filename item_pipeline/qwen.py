from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path
from typing import Any

import requests

from .normalization import parse_attributes


def discover_model(base_url: str, requested_model: str | None, timeout: float) -> str:
    if requested_model:
        return requested_model
    session = requests.Session()
    session.trust_env = False
    response = session.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
    response.raise_for_status()
    models = [str(entry["id"]) for entry in response.json().get("data", [])]
    if len(models) != 1:
        raise ValueError(f"Expected one served model, found {models}; pass --model")
    return models[0]


def generated_item_schema(category: str, attribute_keys: list[str]) -> dict[str, Any]:
    properties = {
        key: {"type": "string", "minLength": 1, "maxLength": 1000}
        for key in attribute_keys
    }
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1, "maxLength": 240},
            "attributes": {
                "type": "object",
                "properties": properties,
                "required": attribute_keys,
                "additionalProperties": False,
            },
            "category": {"type": "string", "const": category},
        },
        "required": ["name", "attributes", "category"],
        "additionalProperties": False,
    }


def _compact_card(row: dict[str, Any], *, max_attributes: int = 40) -> dict[str, Any]:
    attributes = parse_attributes(row["attributes"])
    return {
        "id": int(row["id"]),
        "name": str(row["name"])[:500],
        "attributes": {
            key: value[:600]
            for key, value in list(attributes.items())[:max_attributes]
        },
    }


def build_generation_prompt(
    anchor: dict[str, Any],
    examples: list[dict[str, Any]],
    *,
    feedback: list[str] | None = None,
) -> str:
    attributes = parse_attributes(anchor["attributes"])
    payload = {
        "task": "generate_new_standalone_item",
        "category": str(anchor["category"]),
        "subtype": str(anchor["subtype"]),
        "required_attribute_keys": list(attributes),
        "schema_donor": _compact_card(anchor),
        "style_examples": [_compact_card(row) for row in examples],
    }
    if feedback:
        payload["previous_attempt_rejection_reasons"] = feedback
    return (
        "Создай новую карточку по следующему заданию. Значения schema donor и "
        "примеров нужны только для понимания формата; не копируй сам товар.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def parse_generated_item(
    response: dict[str, Any],
    *,
    expected_category: str,
    expected_keys: list[str],
) -> dict[str, Any]:
    choices = response.get("choices") or []
    if len(choices) != 1:
        raise ValueError("Qwen response must contain exactly one choice")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("Qwen response content is not a string")
    value = json.loads(content)
    if not isinstance(value, dict) or set(value) != {"name", "attributes", "category"}:
        raise ValueError("Generated item must contain exactly name, attributes, category")
    if value["category"] != expected_category:
        raise ValueError("Generated category differs from requested category")
    attributes = value["attributes"]
    if not isinstance(attributes, dict) or set(attributes) != set(expected_keys):
        raise ValueError("Generated attributes do not exactly match the donor key set")
    if not isinstance(value["name"], str) or not value["name"].strip():
        raise ValueError("Generated name is empty")
    if any(not isinstance(item, str) or not item.strip() for item in attributes.values()):
        raise ValueError("Every generated attribute value must be a non-empty string")
    value["attributes"] = {key: attributes[key] for key in expected_keys}
    usage = response.get("usage") or {}
    return {
        "item": value,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "response_id": str(response.get("id") or ""),
    }


class QwenItemClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        system_prompt: str,
        timeout: float = 120.0,
        retries: int = 8,
        temperature: float = 0.7,
        max_tokens: int = 1400,
    ) -> None:
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.retries = retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.local = threading.local()

    def session(self) -> requests.Session:
        session = getattr(self.local, "session", None)
        if session is None:
            session = requests.Session()
            session.trust_env = False
            self.local.session = session
        return session

    def generate(
        self,
        prompt: str,
        *,
        category: str,
        attribute_keys: list[str],
        seed: int,
    ) -> dict[str, Any]:
        schema = generated_item_schema(category, attribute_keys)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "top_p": 0.9,
            "max_tokens": self.max_tokens,
            "seed": int(seed),
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "generated_product_item",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self.session().post(self.url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                parsed = parse_generated_item(
                    response.json(),
                    expected_category=category,
                    expected_keys=attribute_keys,
                )
                return {
                    **parsed,
                    "request_attempts": attempt,
                    "latency_seconds": time.perf_counter() - started,
                }
            except Exception as error:
                last_error = error
                if attempt < self.retries:
                    delay = min(15.0, 0.5 * (2 ** (attempt - 1)))
                    time.sleep(delay + random.random() * 0.25)
        raise RuntimeError(
            f"Qwen item request failed after {self.retries} attempts"
        ) from last_error


def load_system_prompt(path: Path) -> str:
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt is empty: {path}")
    return prompt
