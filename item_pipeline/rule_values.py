from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

from .normalization import normalize_text


_STORAGE_RE = re.compile(
    r"^(?P<value>\d{1,4})(?:\s*(?P<unit>гб|gb|гигабайт(?:а|ов)?|тб|tb|терабайт(?:а|ов)?))?$",
    re.IGNORECASE,
)
_PACKAGE_QUANTITY_RE = re.compile(
    r"^(?P<value>[1-9]\d{0,4})\s*(?:шт\.?|штук(?:а|и)?|ед\.?|единиц(?:а|ы)?)$",
    re.IGNORECASE,
)
_DIMENSION_CONCEPTS = {
    "case_diameter",
    "diameter",
    "length",
    "length_mm",
    "wheel_diameter",
    "width",
}
_DIMENSION_RE = re.compile(
    r"^(?P<value>[+]?(?:\d+(?:[.,]\d+)?|[.,]\d+))\s*"
    r"(?P<unit>мм|миллиметр(?:а|ов)?|см|сантиметр(?:а|ов)?|м|метр(?:а|ов)?|"
    r"дюйм(?:а|ов)?|inch(?:es)?|in|\")$",
    re.IGNORECASE,
)
_BOW_FRACTIONS = {
    "⅛": "1/8",
    "¼": "1/4",
    "½": "1/2",
    "¾": "3/4",
}


def _normalized_domain(domain: Iterable[Any]) -> dict[str, str]:
    return {
        normalize_text(value): str(value).strip()
        for value in domain
        if normalize_text(value)
    }


def canonical_target_value(
    concept: Any,
    product_type: Any,
    value: Any,
    target_value_domain: Iterable[Any] = (),
) -> str | None:
    """Map a finite-domain target value to one canonical catalog value.

    ``None`` means that the value is outside the declared domain.  The special
    mappings deliberately collapse unit spelling and typography which would
    otherwise evade the semantic-signature cap.
    """

    concept_key = normalize_text(concept)
    product_type_key = normalize_text(product_type)
    normalized = normalize_text(value)
    domain = _normalized_domain(target_value_domain)
    if not normalized:
        return None

    if concept_key == "gold_color":
        colors: set[str] = set()
        if re.search(r"(?<!\w)(?:бел\w*|white)(?!\w)", normalized):
            colors.add("белое")
        if re.search(r"(?<!\w)(?:желт\w*|жёлт\w*|yellow)(?!\w)", normalized):
            colors.add("желтое")
        if re.search(r"(?<!\w)(?:красн\w*|розов\w*|red|rose)(?!\w)", normalized):
            colors.add("красное")
        if (
            len(colors) > 1
            or re.search(
                r"(?<!\w)(?:комбинирован\w*|двухцветн\w*|многоцветн\w*|mixed|combined)(?!\w)",
                normalized,
            )
        ):
            candidate = "комбинированное"
        elif len(colors) == 1:
            candidate = next(iter(colors))
        else:
            candidate = normalized
        return domain.get(normalize_text(candidate)) if domain else candidate

    if concept_key == "storage_capacity" and product_type_key == "смартфон":
        compact = re.sub(r"\s+", "", normalized)
        match = _STORAGE_RE.fullmatch(compact)
        if match is None:
            return None if domain else normalized
        amount = int(match.group("value"))
        unit = normalize_text(match.group("unit"))
        if unit in {"тб", "tb", "терабайт", "терабайта", "терабайтов"}:
            amount *= 1024
        candidate = f"{amount} ГБ"
        return domain.get(normalize_text(candidate)) if domain else candidate

    if concept_key == "package_quantity":
        match = _PACKAGE_QUANTITY_RE.fullmatch(normalized)
        if match is None:
            return None
        return f"{int(match.group('value'))} шт"

    if concept_key in _DIMENSION_CONCEPTS:
        match = _DIMENSION_RE.fullmatch(normalized)
        if match is None:
            return None
        try:
            amount = Decimal(match.group("value").replace(",", "."))
        except InvalidOperation:
            return None
        unit = normalize_text(match.group("unit"))
        multiplier = Decimal("1")
        if unit in {"см", "сантиметр", "сантиметра", "сантиметров"}:
            multiplier = Decimal("10")
        elif unit in {"м", "метр", "метра", "метров"}:
            multiplier = Decimal("1000")
        elif unit in {
            '"',
            "дюйм",
            "дюйма",
            "дюймов",
            "inch",
            "inches",
            "in",
        }:
            multiplier = Decimal("25.4")
        millimeters = amount * multiplier
        rendered = format(millimeters, "f")
        if "." in rendered:
            rendered = rendered.rstrip("0").rstrip(".")
        return f"{rendered} мм"

    if concept_key == "size" and product_type_key == "смычок":
        candidate = normalized.replace("⁄", "/")
        for glyph, fraction in _BOW_FRACTIONS.items():
            candidate = candidate.replace(glyph, fraction)
        candidate = re.sub(r"(?<!\w)(?:размер|size)\s*[:=-]?\s*", "", candidate)
        candidate = re.sub(r"\s*/\s*", "/", candidate).strip()
        return domain.get(normalize_text(candidate)) if domain else candidate

    return domain.get(normalized) if domain else normalized


def finite_domain_size(values: Iterable[Any]) -> int:
    """Return the number of distinct canonical values in a declared domain."""

    return len(_normalized_domain(values))
