"""Deterministic MiniLM item serialization shared by data preparations.

This is the serialization portion of the MiniLM ablation supplied for the
full-LLM run.  It intentionally preserves model-code punctuation and digits,
normalizes common measurement units, and supports the same four text variants.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


VARIANTS = ("S0_TITLE", "S1_KEY_VALUE", "S2_VALUES_ONLY", "S3_HYBRID")
DEFAULT_VARIANT = "S1_KEY_VALUE"

_SPACE_RE = re.compile(r"\s+")
_NUMBER_UNIT_RE = re.compile(
    r"(?<![\w.-])(\d+(?:[.,]\d+)?)\s*"
    r"(терабайт(?:а|ов)?|тб|tb|гигабайт(?:а|ов)?|гб|gb|"
    r"мегабайт(?:а|ов)?|мб|mb|килобайт(?:а|ов)?|кб|kb|"
    r"килограмм(?:а|ов)?|кг|kg|миллиграмм(?:а|ов)?|мг|mg|"
    r"грамм(?:а|ов)?|гр|г|g|миллилитр(?:а|ов)?|мл|ml|"
    r"литр(?:а|ов)?|л|l|миллиметр(?:а|ов)?|мм|mm|"
    r"сантиметр(?:а|ов)?|см|cm|герц|гц|hz|"
    r"киловатт(?:а|ов)?|квт|kw|ватт(?:а|ов)?|вт|w)\b",
    flags=re.IGNORECASE,
)
_UNIT_ALIASES = {
    "терабайт": "tb",
    "терабайта": "tb",
    "терабайтов": "tb",
    "тб": "tb",
    "tb": "tb",
    "гигабайт": "gb",
    "гигабайта": "gb",
    "гигабайтов": "gb",
    "гб": "gb",
    "gb": "gb",
    "мегабайт": "mb",
    "мегабайта": "mb",
    "мегабайтов": "mb",
    "мб": "mb",
    "mb": "mb",
    "килобайт": "kb",
    "килобайта": "kb",
    "килобайтов": "kb",
    "кб": "kb",
    "kb": "kb",
    "килограмм": "kg",
    "килограмма": "kg",
    "килограммов": "kg",
    "кг": "kg",
    "kg": "kg",
    "миллиграмм": "mg",
    "миллиграмма": "mg",
    "миллиграммов": "mg",
    "мг": "mg",
    "mg": "mg",
    "грамм": "g",
    "грамма": "g",
    "граммов": "g",
    "гр": "g",
    "г": "g",
    "g": "g",
    "миллилитр": "ml",
    "миллилитра": "ml",
    "миллилитров": "ml",
    "мл": "ml",
    "ml": "ml",
    "литр": "l",
    "литра": "l",
    "литров": "l",
    "л": "l",
    "l": "l",
    "миллиметр": "mm",
    "миллиметра": "mm",
    "миллиметров": "mm",
    "мм": "mm",
    "mm": "mm",
    "сантиметр": "cm",
    "сантиметра": "cm",
    "сантиметров": "cm",
    "см": "cm",
    "cm": "cm",
    "герц": "hz",
    "гц": "hz",
    "hz": "hz",
    "киловатт": "kw",
    "киловатта": "kw",
    "киловаттов": "kw",
    "квт": "kw",
    "kw": "kw",
    "ватт": "w",
    "ватта": "w",
    "ваттов": "w",
    "вт": "w",
    "w": "w",
}


def normalize_text(value: Any) -> str:
    """Normalize safely while preserving digits, punctuation, and model codes."""
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = unicodedata.normalize("NFKC", str(value)).casefold().replace("ё", "е")

    def unit_replacement(match: re.Match[str]) -> str:
        number = match.group(1).replace(",", ".")
        unit = _UNIT_ALIASES[match.group(2).casefold().replace("ё", "е")]
        return f"{number} {unit}"

    text = _NUMBER_UNIT_RE.sub(unit_replacement, text)
    return _SPACE_RE.sub(" ", text).strip()


def _flatten_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        result: list[str] = []
        for key in sorted(value, key=lambda item: normalize_text(item)):
            for nested in _flatten_value(value[key]):
                key_text = normalize_text(key)
                result.append(f"{key_text}: {nested}" if key_text else nested)
        return result
    if isinstance(value, (list, tuple, set)):
        result = []
        for nested in value:
            result.extend(_flatten_value(nested))
        return result
    text = normalize_text(value)
    return [text] if text else []


def parse_attributes(raw: Any) -> list[tuple[str, str]]:
    """Return normalized non-empty key/value pairs without inventing tokens."""
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid attributes JSON: {raw[:120]}") from error
    else:
        data = raw
    if data is None:
        return []
    if isinstance(data, Mapping):
        pairs: list[tuple[str, str]] = []
        for key, value in data.items():
            key_text = normalize_text(key)
            if not key_text:
                continue
            pairs.extend(
                (key_text, item) for item in _flatten_value(value) if item
            )
        return pairs
    if isinstance(data, (list, tuple)):
        pairs = []
        for entry in data:
            if isinstance(entry, Mapping):
                key = entry.get("key", entry.get("name"))
                value = entry.get("value")
                if key is not None and value is not None:
                    key_text = normalize_text(key)
                    pairs.extend(
                        (key_text, item)
                        for item in _flatten_value(value)
                        if key_text and item
                    )
        return pairs
    raise ValueError("Attributes must be a JSON object or a list of key/value records")


def serialize_product(
    title: Any,
    attributes: Sequence[tuple[str, str]],
    variant: str,
    frequent_keys: set[str],
    key_rank: Mapping[str, int],
) -> str:
    """Serialize exactly as the supplied MiniLM serialization ablation."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown serialization variant: {variant}")
    title_text = normalize_text(title)
    if variant == "S0_TITLE" or not attributes:
        return title_text
    ordered = sorted(
        attributes,
        key=lambda pair: (key_rank.get(pair[0], math.inf), pair[0], pair[1]),
    )
    fields = [title_text] if title_text else []
    for key, value in ordered:
        if variant == "S1_KEY_VALUE" or (
            variant == "S3_HYBRID" and key in frequent_keys
        ):
            fields.append(f"{key}: {value}")
        else:
            fields.append(value)
    return ". ".join(
        field.rstrip(". ") for field in fields if field
    ).strip()


@dataclass(frozen=True)
class AttributeFrequency:
    attribute_name: str
    item_support: int
    occurrences: int


def attribute_frequency(
    parsed_attributes: Iterable[Sequence[tuple[str, str]]],
) -> list[AttributeFrequency]:
    """Build the same deterministic global key order used by the ablation."""
    item_support: Counter[str] = Counter()
    occurrence_count: Counter[str] = Counter()
    for attributes in parsed_attributes:
        item_support.update({key for key, _ in attributes})
        occurrence_count.update(key for key, _ in attributes)
    return [
        AttributeFrequency(
            attribute_name=key,
            item_support=item_support[key],
            occurrences=occurrence_count[key],
        )
        for key in sorted(
            occurrence_count,
            key=lambda key: (-occurrence_count[key], -item_support[key], key),
        )
    ]
