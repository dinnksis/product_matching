from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from typing import Any


SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-zа-яё0-9]+(?:[-./][a-zа-яё0-9]+)*", re.IGNORECASE)
TYPE_KEY_RE = re.compile(
    r"^(?:тип|вид)(?:\s+(?:товара|продукта|изделия|устройства|инструмента))?$"
)


def normalize_text(value: Any) -> str:
    """Normalize text for matching while preserving product-code punctuation."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().replace("ё", "е")
    return SPACE_RE.sub(" ", text).strip()


def output_text(value: Any) -> str:
    """Normalize generated text to the lowercase style of the competition data."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return SPACE_RE.sub(" ", text).strip()


def tokenize(value: Any) -> list[str]:
    return TOKEN_RE.findall(normalize_text(value))


def parse_attributes(raw: Any, *, keep_empty: bool = False) -> dict[str, str]:
    """Parse the competition JSON object without renaming its original keys."""
    if isinstance(raw, Mapping):
        value = dict(raw)
    elif isinstance(raw, str):
        value = json.loads(raw)
    else:
        raise ValueError(f"attributes must be a JSON string or object, got {type(raw)!r}")
    if not isinstance(value, dict):
        raise ValueError("attributes JSON must contain an object")

    result: dict[str, str] = {}
    for key, item in value.items():
        key_text = SPACE_RE.sub(" ", str(key)).strip()
        if not key_text:
            continue
        if item is None:
            item_text = ""
        elif isinstance(item, str):
            item_text = SPACE_RE.sub(" ", item).strip()
        else:
            item_text = SPACE_RE.sub(" ", str(item)).strip()
        if keep_empty or item_text:
            result[key_text] = item_text
    return result


def extract_subtype(name: str, attributes: Mapping[str, str]) -> str:
    """Extract a stable, weak subtype label with a title fallback."""
    candidates: list[tuple[int, str]] = []
    for key, value in attributes.items():
        normalized_key = normalize_text(key)
        normalized_value = normalize_text(value)
        if not normalized_value:
            continue
        if normalized_key == "тип":
            candidates.append((0, normalized_value))
        elif TYPE_KEY_RE.match(normalized_key):
            candidates.append((1, normalized_value))
    if candidates:
        value = min(candidates)[1]
        # Multi-valued source fields are too sparse as literal subtype labels.
        return re.split(r"[;,|]", value, maxsplit=1)[0][:120].strip()

    title_tokens = tokenize(name)
    if not title_tokens:
        return "__unknown__"
    # A short title prefix gives category-local retrieval a deterministic fallback.
    return "__title__:" + " ".join(title_tokens[:2])[:120]


def retrieval_text(name: str, attributes: Mapping[str, str], *, max_chars: int = 1600) -> str:
    values = [str(value) for value in attributes.values() if str(value).strip()]
    text = " ".join([str(name), *values])
    return SPACE_RE.sub(" ", text).strip()[:max_chars]


def title_attribute_token_coverage(name: str, attributes: Mapping[str, str]) -> float:
    title_tokens = tokenize(name)
    if not title_tokens:
        return 0.0
    attribute_tokens = {
        token for value in attributes.values() for token in tokenize(value)
    }
    return sum(token in attribute_tokens for token in title_tokens) / len(title_tokens)


def stable_hash64(seed: int, value: int | str) -> int:
    payload = f"{seed}\0{value}".encode("utf-8", errors="replace")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
