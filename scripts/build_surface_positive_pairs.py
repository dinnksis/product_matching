#!/usr/bin/env python3
"""Build leakage-safe label-1 pairs using fact-preserving surface variation only.

The builder deliberately does not invent products or apply atomic attribute
changes.  It clones a card whose id occurs in the supplied *training* positive
pairs and creates a second seller-style rendering of the same facts.  The
second rendering may reorder title blocks, change punctuation/case, rename an
attribute with a conservative alias, reorder JSON keys, and omit a small
number of non-identity attributes.  Every output pair is checked against its
source card and receives row-level provenance.

This is intentionally a standalone data builder.  It performs no network or
Kaggle operations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VERSION = "surface_positive_augmentation_v1"
VALIDATION_VERSION = "surface_positive_no_atomic_change_validator_v1"
SELECTION_VERSION = "positive_train_pair_category_stratified_donor_v1"
LABEL_SOURCE = "deterministic_surface_only_positive_v1"
DEFAULT_COUNT = 5_000
DEFAULT_SEED = 2_026_082_829
DEFAULT_ID_START = -4_000_000_000
DEFAULT_ITEMS_PATH = ROOT / "data/items_human.parquet"
DEFAULT_ELIGIBLE_PAIRS_PATH = (
    ROOT / "prepared/validation_splits_v1/human/train_pairs.parquet"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "item_pipeline/artifacts/surface_positive_human_train_5k_v1"
)
DEFAULT_FORBIDDEN_CATEGORIES = frozenset({"Одежда", "Бытовая техника"})

REQUIRED_ITEM_COLUMNS = frozenset({"id", "name", "attributes", "category"})
REQUIRED_PAIR_COLUMNS = frozenset({"id1", "id2", "target"})
IDENTITY_KEY_RE = re.compile(
    r"(?<!\w)(?:brand(?:\s+name)?|бренд\w*|производител\w*|maker|model\w*|"
    r"модел\w*|серия|series|sku|артикул\w*|партномер\w*|part\s*number|mpn|"
    r"oem|код\s*(?:товара|модели)|тип(?:\s+товара)?|вид\s*товара)(?!\w)",
    re.IGNORECASE,
)
BRAND_KEY_RE = re.compile(
    r"^(?:brand(?: name)?|бренд(?: товара)?|марка|торговая марка|"
    r"производитель|изготовитель)$",
    re.IGNORECASE,
)
TYPE_KEY_RE = re.compile(
    r"^(?:тип|тип товара|вид товара|product type)$", re.IGNORECASE
)

COLOR_WORDS = frozenset(
    {
        "бежевый",
        "белый",
        "бирюзовый",
        "бордовый",
        "голубой",
        "желтый",
        "зелёный",
        "зеленый",
        "золотой",
        "коричневый",
        "красный",
        "оранжевый",
        "розовый",
        "серебристый",
        "серый",
        "синий",
        "фиолетовый",
        "черный",
        "чёрный",
        "black",
        "blue",
        "brown",
        "green",
        "grey",
        "gray",
        "orange",
        "pink",
        "purple",
        "red",
        "silver",
        "white",
        "yellow",
    }
)

# Only high-confidence concept-preserving aliases are allowed.  When a key is
# not listed, the builder limits itself to case and separator presentation.
SAFE_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "артикул": ("Артикул товара", "Код артикула"),
    "артикул производителя": (
        "Партномер",
        "Артикул производителя",
    ),
    "бренд": ("Бренд товара", "Марка"),
    "вес": ("Вес товара",),
    "вес товара": ("Вес",),
    "вид товара": ("Тип товара",),
    "длина": ("Длина товара",),
    "комплектация": ("Комплект",),
    "материал": ("Материал изделия",),
    "модель": ("Модель товара",),
    "партномер (артикул производителя)": (
        "Артикул производителя",
        "Партномер",
    ),
    "размер": ("Размер товара",),
    "серия": ("Серия товара",),
    "страна производства": ("Страна-производитель",),
    "тип": ("Тип товара",),
    "тип товара": ("Вид товара",),
    "цвет": ("Цвет товара",),
    "цвет товара": ("Цвет",),
    "ширина": ("Ширина товара",),
}

PROFILE_CONFIG: dict[str, dict[str, int]] = {
    # Kept deliberately conservative: most pairs retain every visible fact.
    "light_relisting": {"weight": 50, "alias_max": 2, "omit_max": 0},
    "schema_variant": {"weight": 35, "alias_max": 3, "omit_max": 1},
    "sparse_seller": {"weight": 15, "alias_max": 4, "omit_max": 2},
}

OUTPUT_FILENAMES = (
    "items.parquet",
    "pairs.parquet",
    "pair_provenance.parquet",
    "summary.json",
    "validation_report.json",
    "distribution_report.json",
    "examples.jsonl",
    "build_manifest.json",
)


class SurfacePositiveError(ValueError):
    """Raised when a no-atomic-change contract cannot be satisfied."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS_PATH)
    parser.add_argument(
        "--eligible-pairs", type=Path, default=DEFAULT_ELIGIBLE_PAIRS_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--id-start", type=int, default=DEFAULT_ID_START)
    parser.add_argument("--min-attributes", type=int, default=5)
    parser.add_argument("--max-attributes", type=int, default=30)
    parser.add_argument("--min-name-tokens", type=int, default=4)
    parser.add_argument("--max-name-tokens", type=int, default=30)
    parser.add_argument("--example-count", type=int, default=30)
    parser.add_argument(
        "--forbidden-category",
        action="append",
        dest="forbidden_categories",
        help=(
            "Category excluded from donors. Repeat to supply a custom set. "
            "Defaults to the two frozen OOD categories."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing output directory without rebuilding it.",
    )
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_hash(seed: int, *parts: Any) -> int:
    payload = json.dumps(
        [int(seed), *parts], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def normalize_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(normalized.casefold().replace("ё", "е").split())


def normalized_key(value: Any) -> str:
    text = normalize_text(value).replace("_", " ")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return " ".join(text.split())


def fact_tokens(value: Any) -> tuple[str, ...]:
    return tuple(re.findall(r"[0-9a-zа-я]+", normalize_text(value)))


def token_multiset(value: Any) -> Counter[str]:
    return Counter(fact_tokens(value))


def canonical_value(value: Any) -> tuple[str, ...]:
    """Canonical visible facts for a value; punctuation/case are presentation."""

    return fact_tokens(value)


def json_object(raw: Any) -> dict[str, str]:
    if isinstance(raw, Mapping):
        parsed = dict(raw)
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            raise SurfacePositiveError("attributes are not valid JSON") from error
    else:
        raise SurfacePositiveError("attributes must be a JSON object or mapping")
    if not isinstance(parsed, dict):
        raise SurfacePositiveError("attributes must decode to an object")
    result: dict[str, str] = {}
    for raw_key, raw_value in parsed.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise SurfacePositiveError("attribute keys must be non-empty strings")
        if not isinstance(raw_value, (str, int, float, bool)):
            raise SurfacePositiveError("attribute values must be scalar")
        value = str(raw_value).strip()
        if not value:
            raise SurfacePositiveError("attribute values must be non-empty")
        result[raw_key.strip()] = value
    return result


def json_attributes(attributes: Mapping[str, str]) -> str:
    return json.dumps(
        dict(attributes), ensure_ascii=False, separators=(",", ":"), sort_keys=False
    )


def source_attributes_text(raw: Any) -> str:
    if isinstance(raw, str):
        # Re-serialize so output schema and hashes do not depend on whitespace in
        # the source JSON, while preserving key order and all facts.
        return json_attributes(json_object(raw))
    return json_attributes(json_object(raw))


def canonical_source_card(row: Mapping[str, Any]) -> str:
    attrs = json_object(row["attributes"])
    return json.dumps(
        {
            "name_tokens": sorted(token_multiset(row["name"]).items()),
            "attributes": sorted(
                (normalized_key(key), canonical_value(value))
                for key, value in attrs.items()
            ),
            "category": normalize_text(row["category"]),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def identity_key(key: str) -> bool:
    return IDENTITY_KEY_RE.search(normalized_key(key)) is not None


def suspicious_source_reason(attributes: Mapping[str, str]) -> str | None:
    """Reject obvious schema/value swaps such as ``brand: purple``."""

    for key, value in attributes.items():
        key_norm = normalized_key(key)
        value_words = set(fact_tokens(value))
        if not value_words:
            return "empty_attribute_value"
        if BRAND_KEY_RE.fullmatch(key_norm):
            if value_words <= {normalize_text(word) for word in COLOR_WORDS}:
                return "brand_is_color"
            if all(word.isdigit() for word in value_words):
                return "brand_is_numeric_only"
        if TYPE_KEY_RE.fullmatch(key_norm) and value_words <= {
            normalize_text(word) for word in COLOR_WORDS
        }:
            return "product_type_is_color"
    return None


def _profile(seed: int, source_id: int) -> str:
    point = stable_hash(seed, "profile", source_id) % sum(
        config["weight"] for config in PROFILE_CONFIG.values()
    )
    cumulative = 0
    for name, config in PROFILE_CONFIG.items():
        cumulative += config["weight"]
        if point < cumulative:
            return name
    raise AssertionError("profile weights do not cover the sample space")


def _find_token_span(tokens: Sequence[str], phrase: Sequence[str]) -> tuple[int, int] | None:
    normalized_tokens = [normalize_text(token) for token in tokens]
    normalized_phrase = [normalize_text(token) for token in phrase]
    if not normalized_phrase or len(normalized_phrase) >= len(normalized_tokens):
        return None
    width = len(normalized_phrase)
    for start in range(len(normalized_tokens) - width + 1):
        if normalized_tokens[start : start + width] == normalized_phrase:
            return start, start + width
    return None


def _name_anchor_phrases(attributes: Mapping[str, str]) -> list[tuple[str, ...]]:
    result: list[tuple[str, ...]] = []
    for key, value in attributes.items():
        key_norm = normalized_key(key)
        # Moving a known brand between the title edges is a common and safe
        # seller variation.  Product-type phrases stay in place: moving them
        # produced grammatical but visibly synthetic titles in the pilot.
        if BRAND_KEY_RE.fullmatch(key_norm):
            phrase = tuple(re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", value))
            if phrase:
                result.append(phrase)
    return result


def _trim_title_boundary(value: str) -> str:
    """Remove only separator punctuation left at a moved block boundary."""

    return re.sub(r"^[\s,;|/—–-]+|[\s,;|/—–-]+$", "", value).strip()


def transform_name(
    name: str, attributes: Mapping[str, str], seed: int, source_id: int
) -> tuple[str, str]:
    """Return a conservative seller-style title with the same token multiset."""

    normalized_name = unicodedata.normalize("NFKC", name)
    token_matches = list(re.finditer(r"[0-9A-Za-zА-Яа-яЁё]+", normalized_name))
    tokens = [match.group(0) for match in token_matches]
    if len(tokens) < 2:
        raise SurfacePositiveError("title has too few tokens for a safe variation")
    delimiter = (", ", " — ", " ")[stable_hash(seed, "delimiter", source_id) % 3]

    for phrase in _name_anchor_phrases(attributes):
        span = _find_token_span(tokens, phrase)
        if span is None:
            continue
        start, end = span
        char_start = token_matches[start].start()
        char_end = token_matches[end - 1].end()
        anchored = _trim_title_boundary(normalized_name[char_start:char_end])
        remainder = _trim_title_boundary(
            normalized_name[:char_start] + " " + normalized_name[char_end:]
        )
        if not anchored or not remainder:
            continue
        if start <= len(tokens) // 2:
            candidate = remainder + delimiter + anchored
            operation = "move_identity_phrase_to_end"
        else:
            candidate = anchored + delimiter + remainder
            operation = "move_identity_phrase_to_start"
        candidate = " ".join(candidate.split())
        if normalize_text(candidate) != normalize_text(name):
            if token_multiset(candidate) != token_multiset(name):
                raise AssertionError("identity-phrase title transform changed facts")
            return candidate, operation

    # If the source exposes seller clauses, only reorder complete clauses.
    clauses = [
        clause.strip()
        for clause in re.split(
            r"\s*(?:(?<!\d),(?!\d)|[;|]|\s[—–-]\s|\s/\s)\s*",
            normalized_name,
        )
        if clause.strip()
    ]
    if len(clauses) >= 2:
        # Keep clause order and vary only its seller punctuation.  Brand
        # movement above supplies the genuinely reordered examples.
        candidate = delimiter.join(clauses)
        candidate = " ".join(candidate.split())
        if token_multiset(candidate) == token_multiset(name) and (
            normalize_text(candidate) != normalize_text(name)
        ):
            return candidate, "change_title_clause_separators"

    # Last conservative fallback: insert a seller separator without changing
    # word order.  Attribute aliases/order still give the two cards independent
    # marketplace schemas even when the title change is intentionally tiny.
    preferred_width = 1 + (
        stable_hash(seed, "title-width", source_id) % min(2, len(tokens) - 1)
    )
    widths = [preferred_width] + [
        width
        for width in range(1, min(3, len(tokens)))
        if width != preferred_width
    ]
    delimiters = [delimiter, ", ", " — ", " / "]
    for width in widths:
        cut = token_matches[width - 1].end()
        if width < len(token_matches):
            between_tokens = normalized_name[cut : token_matches[width].start()]
            if not any(character.isspace() for character in between_tokens):
                # Never insert punctuation inside a hyphenated model/name.
                continue
        leading = _trim_title_boundary(normalized_name[:cut])
        remainder = _trim_title_boundary(normalized_name[cut:])
        for fallback_delimiter in dict.fromkeys(delimiters):
            if not fallback_delimiter.strip():
                continue
            candidate = " ".join(
                (leading + fallback_delimiter + remainder).split()
            )
            if token_multiset(candidate) != token_multiset(name):
                raise AssertionError("title separator insertion changed facts")
            if normalize_text(candidate) != normalize_text(name):
                return candidate, "insert_title_separator"

    # A one-character punctuation variant is preferable to dropping an
    # otherwise valid donor.  It changes no alphanumeric fact.
    base = normalized_name.rstrip(" .!")
    for suffix in (".", "!"):
        candidate = base + suffix
        if (
            token_multiset(candidate) == token_multiset(name)
            and normalize_text(candidate) != normalize_text(name)
        ):
            return candidate, "change_terminal_punctuation"
    raise SurfacePositiveError("could not create a distinct title")


def key_variants(key: str) -> tuple[str, ...]:
    normalized = normalized_key(key)
    candidates: list[str] = list(SAFE_KEY_ALIASES.get(normalized, ()))
    presentation = key.casefold()
    if presentation != key:
        candidates.append(presentation)
    if key and key[0].islower():
        candidates.append(key[0].upper() + key[1:])
    if "_" in key:
        candidates.append(" ".join(key.split("_")))
    deduplicated: list[str] = []
    for candidate in candidates:
        clean = " ".join(candidate.split())
        if clean and clean != key and clean not in deduplicated:
            deduplicated.append(clean)
    return tuple(deduplicated)


def format_value(value: str, seed: int, source_id: int, key: str) -> str:
    """Change presentation while retaining the ordered alphanumeric tokens."""

    candidates: list[str] = []
    if ";" in value:
        candidates.append(value.replace(";", ","))
    if "/" in value:
        candidates.append(value.replace("/", " / "))
    if value.endswith("."):
        candidates.append(value[:-1])
    if any(character.isalpha() for character in value) and not any(
        character.isdigit() for character in value
    ):
        candidates.extend((value.casefold(), value[:1].upper() + value[1:]))
    candidates.append(" ".join(value.split()))
    valid = [
        candidate
        for candidate in candidates
        if candidate != value and canonical_value(candidate) == canonical_value(value)
    ]
    if not valid:
        return value
    return valid[stable_hash(seed, "value", source_id, key) % len(valid)]


def transform_attributes(
    attributes: Mapping[str, str], profile: str, seed: int, source_id: int
) -> tuple[dict[str, str], dict[str, Any]]:
    config = PROFILE_CONFIG[profile]
    source_keys = list(attributes)

    optional_keys = [key for key in source_keys if not identity_key(key)]
    optional_keys.sort(key=lambda key: stable_hash(seed, "omit", source_id, key))
    omit_max = min(config["omit_max"], max(0, len(source_keys) - 4))
    if profile == "schema_variant" and len(source_keys) < 7:
        omit_max = 0
    if profile == "sparse_seller" and len(source_keys) < 9:
        omit_max = min(omit_max, 1 if len(source_keys) >= 7 else 0)
    omitted = optional_keys[:omit_max]
    retained = [key for key in source_keys if key not in set(omitted)]

    aliasable = [key for key in retained if key_variants(key)]
    aliasable.sort(key=lambda key: stable_hash(seed, "alias", source_id, key))
    alias_count = min(config["alias_max"], max(1, math.ceil(len(retained) * 0.25)))
    alias_keys = set(aliasable[:alias_count])

    # Rotate/reverse the visible schema independently from the source order.
    ordered = sorted(
        retained, key=lambda key: stable_hash(seed, "key-order", source_id, key)
    )
    used_output_keys: set[str] = set()
    output: dict[str, str] = {}
    right_to_source: dict[str, str] = {}
    alias_map: dict[str, str] = {}
    value_format_keys: list[str] = []
    for source_key in ordered:
        output_key = source_key
        if source_key in alias_keys:
            variants = list(key_variants(source_key))
            variants.sort(
                key=lambda value: stable_hash(
                    seed, "alias-choice", source_id, source_key, value
                )
            )
            for candidate in variants:
                if candidate not in used_output_keys:
                    output_key = candidate
                    break
        if output_key in used_output_keys:
            output_key = source_key
        if output_key in used_output_keys:
            raise SurfacePositiveError("attribute alias produced a key collision")
        used_output_keys.add(output_key)
        output_value = format_value(attributes[source_key], seed, source_id, source_key)
        if output_value != attributes[source_key]:
            value_format_keys.append(source_key)
        output[output_key] = output_value
        right_to_source[output_key] = source_key
        if output_key != source_key:
            alias_map[source_key] = output_key

    return output, {
        "omitted_keys": omitted,
        "alias_map": alias_map,
        "right_key_to_source_key": right_to_source,
        "value_format_keys": value_format_keys,
        "attribute_order_operation": "stable_hash_permutation",
    }


def _validate_source_frames(items: pd.DataFrame, eligible_pairs: pd.DataFrame) -> None:
    if missing := REQUIRED_ITEM_COLUMNS - set(items.columns):
        raise SurfacePositiveError(f"items missing columns: {sorted(missing)}")
    if missing := REQUIRED_PAIR_COLUMNS - set(eligible_pairs.columns):
        raise SurfacePositiveError(
            f"eligible pairs missing columns: {sorted(missing)}"
        )
    if items["id"].isna().any() or items["id"].duplicated().any():
        raise SurfacePositiveError("source item ids must be non-null and unique")
    numeric_targets = pd.to_numeric(eligible_pairs["target"], errors="coerce")
    if numeric_targets.isna().any() or not numeric_targets.isin([0, 1]).all():
        raise SurfacePositiveError("eligible-pair targets must be binary")
    if eligible_pairs[["id1", "id2"]].isna().any().any():
        raise SurfacePositiveError("eligible pairs contain null ids")


def _eligible_candidates(
    items: pd.DataFrame,
    eligible_pairs: pd.DataFrame,
    *,
    seed: int,
    min_attributes: int,
    max_attributes: int,
    min_name_tokens: int,
    max_name_tokens: int,
    forbidden_categories: frozenset[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    item_map = {int(row.id): row._asdict() for row in items.itertuples(index=False)}
    positive_pairs = eligible_pairs.loc[
        pd.to_numeric(eligible_pairs["target"], errors="coerce").eq(1)
    ].copy()
    positive_pairs["source_pair_row"] = positive_pairs.index.astype("int64")
    positive_pairs = positive_pairs.sort_values(
        ["source_pair_row", "id1", "id2"], kind="stable"
    )
    rejection_counts: Counter[str] = Counter()
    valid_item_cache: dict[int, tuple[bool, str, dict[str, str] | None]] = {}

    def validate_item(source_id: int) -> tuple[bool, str, dict[str, str] | None]:
        cached = valid_item_cache.get(source_id)
        if cached is not None:
            return cached
        row = item_map.get(source_id)
        if row is None:
            result = (False, "missing_source_item", None)
        else:
            category = str(row.get("category") or "").strip()
            name = str(row.get("name") or "").strip()
            if not category:
                result = (False, "empty_category", None)
            elif category in forbidden_categories:
                result = (False, "forbidden_ood_category", None)
            elif not min_name_tokens <= len(fact_tokens(name)) <= max_name_tokens:
                result = (False, "name_token_count_out_of_bounds", None)
            else:
                try:
                    attrs = json_object(row.get("attributes"))
                except SurfacePositiveError:
                    result = (False, "invalid_attributes", None)
                else:
                    suspicious = suspicious_source_reason(attrs)
                    if not min_attributes <= len(attrs) <= max_attributes:
                        result = (False, "attribute_count_out_of_bounds", None)
                    elif suspicious:
                        result = (False, suspicious, None)
                    else:
                        result = (True, "", attrs)
        valid_item_cache[source_id] = result
        return result

    candidates: list[dict[str, Any]] = []
    seen_source_ids: set[int] = set()
    seen_cards: set[str] = set()
    reference_rows: list[dict[str, Any]] = []
    for pair in positive_pairs.itertuples(index=False):
        left_id, right_id = int(pair.id1), int(pair.id2)
        left_row, right_row = item_map.get(left_id), item_map.get(right_id)
        if left_row is not None and right_row is not None:
            if str(left_row.get("category")) == str(right_row.get("category")):
                reference_rows.append(
                    {
                        "id1": left_id,
                        "id2": right_id,
                        "category": str(left_row.get("category") or ""),
                    }
                )
        endpoint_ids = [left_id, right_id]
        if stable_hash(seed, "endpoint", int(pair.source_pair_row)) % 2:
            endpoint_ids.reverse()
        chosen: int | None = None
        chosen_attrs: dict[str, str] | None = None
        for source_id in endpoint_ids:
            valid, reason, attrs = validate_item(source_id)
            if not valid:
                rejection_counts[reason] += 1
                continue
            if source_id in seen_source_ids:
                rejection_counts["duplicate_source_item"] += 1
                continue
            assert attrs is not None
            source_row = item_map[source_id]
            card_key = canonical_source_card(source_row)
            if card_key in seen_cards:
                rejection_counts["duplicate_source_card"] += 1
                continue
            chosen, chosen_attrs = source_id, attrs
            seen_source_ids.add(source_id)
            seen_cards.add(card_key)
            break
        if chosen is None:
            rejection_counts["positive_pair_without_eligible_donor"] += 1
            continue
        partner_id = right_id if chosen == left_id else left_id
        source_row = item_map[chosen]
        candidates.append(
            {
                "source_item_id": chosen,
                "source_partner_id": partner_id,
                "source_pair_row": int(pair.source_pair_row),
                "category": str(source_row["category"]),
                "source": source_row,
                "attributes_parsed": chosen_attrs,
            }
        )
    report = {
        "eligible_pair_rows": int(len(eligible_pairs)),
        "eligible_positive_pair_rows": int(len(positive_pairs)),
        "reference_positive_pair_rows": int(len(reference_rows)),
        "candidate_donors": int(len(candidates)),
        "candidate_categories": dict(
            sorted(Counter(row["category"] for row in candidates).items())
        ),
        "rejection_counts": dict(sorted(rejection_counts.items())),
    }
    return candidates, report, reference_rows


def allocate_category_quotas(
    candidates: Sequence[Mapping[str, Any]], count: int
) -> dict[str, int]:
    capacities = Counter(str(row["category"]) for row in candidates)
    if count < 1:
        raise SurfacePositiveError("count must be positive")
    if sum(capacities.values()) < count:
        raise SurfacePositiveError(
            f"only {sum(capacities.values())} eligible donors for requested {count}"
        )
    total = sum(capacities.values())
    desired = {category: count * capacity / total for category, capacity in capacities.items()}
    quotas = {
        category: min(capacity, int(math.floor(desired[category])))
        for category, capacity in capacities.items()
    }
    remaining = count - sum(quotas.values())
    while remaining:
        available = [
            category
            for category, capacity in capacities.items()
            if quotas[category] < capacity
        ]
        if not available:
            raise AssertionError("category quota allocation exhausted capacity")
        category = max(
            available,
            key=lambda value: (
                desired[value] - quotas[value],
                capacities[value],
                value,
            ),
        )
        quotas[category] += 1
        remaining -= 1
    return dict(sorted(quotas.items()))


def select_candidates(
    candidates: Sequence[dict[str, Any]], count: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    quotas = allocate_category_quotas(candidates, count)
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        by_category.setdefault(str(row["category"]), []).append(row)
    selected: list[dict[str, Any]] = []
    for category, quota in quotas.items():
        ranked = sorted(
            by_category[category],
            key=lambda row: (
                stable_hash(seed, "donor-rank", row["source_item_id"]),
                int(row["source_item_id"]),
            ),
        )
        selected.extend(ranked[:quota])
    selected.sort(
        key=lambda row: (
            stable_hash(seed, "output-order", row["source_item_id"]),
            int(row["source_item_id"]),
        )
    )
    if len(selected) != count:
        raise AssertionError("selected donor count differs from requested count")
    return selected, quotas


def _facts_sha(name: str, attributes: Mapping[str, str], category: str) -> str:
    return sha256_json(
        {
            "name_tokens": sorted(token_multiset(name).items()),
            "attributes": sorted(
                (normalized_key(key), canonical_value(value))
                for key, value in attributes.items()
            ),
            "category": normalize_text(category),
        }
    )


def materialize(
    selected: Sequence[dict[str, Any]], *, seed: int, id_start: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    item_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    for pair_index, donor in enumerate(selected):
        source_id = int(donor["source_item_id"])
        source = donor["source"]
        source_attrs = dict(donor["attributes_parsed"])
        category = str(source["category"])
        source_name = str(source["name"])
        profile = _profile(seed, source_id)
        right_name, name_operation = transform_name(
            source_name, source_attrs, seed, source_id
        )
        right_attrs, attr_meta = transform_attributes(
            source_attrs, profile, seed, source_id
        )
        left_id = int(id_start) - pair_index * 2
        right_id = left_id - 1
        left_attrs_text = source_attributes_text(source["attributes"])
        right_attrs_text = json_attributes(right_attrs)
        item_rows.extend(
            [
                {
                    "id": left_id,
                    "name": source_name,
                    "attributes": left_attrs_text,
                    "category": category,
                },
                {
                    "id": right_id,
                    "name": right_name,
                    "attributes": right_attrs_text,
                    "category": category,
                },
            ]
        )
        pair_rows.append(
            {
                "id1": left_id,
                "id2": right_id,
                "target": 1,
                "label_source": LABEL_SOURCE,
            }
        )
        source_facts_sha = _facts_sha(source_name, source_attrs, category)
        retained_source_attrs = {
            source_key: source_attrs[source_key]
            for source_key in attr_meta["right_key_to_source_key"].values()
        }
        retained_right_attrs = {
            source_key: right_attrs[right_key]
            for right_key, source_key in attr_meta["right_key_to_source_key"].items()
        }
        provenance_rows.append(
            {
                "pair_index": pair_index,
                "id1": left_id,
                "id2": right_id,
                "target": 1,
                "label_source": LABEL_SOURCE,
                "source_item_id": source_id,
                "source_partner_id": int(donor["source_partner_id"]),
                "source_pair_row": int(donor["source_pair_row"]),
                "category": category,
                "augmentation_profile": profile,
                "name_operation": name_operation,
                "attribute_order_operation": attr_meta["attribute_order_operation"],
                "key_aliases_json": json.dumps(
                    attr_meta["alias_map"], ensure_ascii=False, sort_keys=True
                ),
                "right_key_to_source_key_json": json.dumps(
                    attr_meta["right_key_to_source_key"],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "omitted_keys_json": json.dumps(
                    attr_meta["omitted_keys"], ensure_ascii=False
                ),
                "value_format_keys_json": json.dumps(
                    attr_meta["value_format_keys"], ensure_ascii=False
                ),
                "source_attribute_count": len(source_attrs),
                "right_attribute_count": len(right_attrs),
                "source_facts_sha256": source_facts_sha,
                "retained_source_facts_sha256": _facts_sha(
                    source_name, retained_source_attrs, category
                ),
                "retained_right_facts_sha256": _facts_sha(
                    right_name, retained_right_attrs, category
                ),
                "name_token_multiset_sha256": sha256_json(
                    sorted(token_multiset(source_name).items())
                ),
                "transform_seed": int(stable_hash(seed, "row", source_id)),
                "no_atomic_change": True,
                "builder_version": VERSION,
            }
        )
    return (
        pd.DataFrame(pair_rows),
        pd.DataFrame(item_rows),
        pd.DataFrame(provenance_rows),
    )


def validate_surface_dataset(
    *,
    source_items: pd.DataFrame,
    eligible_pairs: pd.DataFrame,
    pairs: pd.DataFrame,
    items: pd.DataFrame,
    provenance: pd.DataFrame,
    forbidden_categories: frozenset[str] = DEFAULT_FORBIDDEN_CATEGORIES,
    expected_count: int | None = None,
) -> dict[str, Any]:
    """Validate that every pair is a fact-preserving view of a train donor."""

    errors: list[str] = []
    warnings: list[str] = []
    count = len(pairs)
    if expected_count is not None and count != expected_count:
        errors.append(f"pair_count:{count}!={expected_count}")
    if len(items) != count * 2:
        errors.append(f"item_count:{len(items)}!={count * 2}")
    if len(provenance) != count:
        errors.append(f"provenance_count:{len(provenance)}!={count}")
    if missing := REQUIRED_PAIR_COLUMNS - set(pairs.columns):
        errors.append(f"pairs_missing_columns:{sorted(missing)}")
    if missing := REQUIRED_ITEM_COLUMNS - set(items.columns):
        errors.append(f"items_missing_columns:{sorted(missing)}")
    required_provenance = {
        "pair_index",
        "id1",
        "id2",
        "source_item_id",
        "source_partner_id",
        "source_pair_row",
        "right_key_to_source_key_json",
        "omitted_keys_json",
        "source_facts_sha256",
        "retained_source_facts_sha256",
        "retained_right_facts_sha256",
        "no_atomic_change",
    }
    if missing := required_provenance - set(provenance.columns):
        errors.append(f"provenance_missing_columns:{sorted(missing)}")
    if errors:
        return {
            "version": VALIDATION_VERSION,
            "valid": False,
            "pairs": count,
            "items": len(items),
            "errors": errors,
            "warnings": warnings,
        }

    source_map = {
        int(row.id): row._asdict() for row in source_items.itertuples(index=False)
    }
    item_map = {int(row.id): row._asdict() for row in items.itertuples(index=False)}
    positive = eligible_pairs.loc[
        pd.to_numeric(eligible_pairs["target"], errors="coerce").eq(1)
    ]
    eligible_ids = set(pd.to_numeric(positive["id1"]).astype("int64")) | set(
        pd.to_numeric(positive["id2"]).astype("int64")
    )
    source_ids = set(source_map)
    output_ids = set(item_map)
    pair_ids = set(pd.to_numeric(pairs["id1"]).astype("int64")) | set(
        pd.to_numeric(pairs["id2"]).astype("int64")
    )
    if items["id"].duplicated().any():
        errors.append("duplicate_output_item_ids")
    if pair_ids != output_ids:
        errors.append("pair_item_catalogue_mismatch")
    if output_ids & source_ids:
        errors.append("output_ids_overlap_source_ids")
    targets = pd.to_numeric(pairs["target"], errors="coerce")
    if targets.isna().any() or not targets.eq(1).all():
        errors.append("targets_are_not_all_one")
    if "label_source" not in pairs or set(pairs["label_source"].astype(str)) != {
        LABEL_SOURCE
    }:
        errors.append("unexpected_label_source")
    if provenance["pair_index"].duplicated().any():
        errors.append("duplicate_pair_index")
    if provenance["source_item_id"].duplicated().any():
        errors.append("reused_source_item")
    if not provenance["no_atomic_change"].eq(True).all():  # noqa: E712
        errors.append("no_atomic_change_flag_is_false")

    metadata_map = {
        (int(row.id1), int(row.id2)): row._asdict()
        for row in provenance.itertuples(index=False)
    }
    if len(metadata_map) != len(provenance):
        errors.append("duplicate_provenance_pair")

    row_error_counts: Counter[str] = Counter()
    checked_rows = 0
    for pair in pairs.itertuples(index=False):
        pair_key = (int(pair.id1), int(pair.id2))
        left = item_map.get(pair_key[0])
        right = item_map.get(pair_key[1])
        meta = metadata_map.get(pair_key)
        if left is None or right is None or meta is None:
            row_error_counts["missing_pair_item_or_provenance"] += 1
            continue
        source_id = int(meta["source_item_id"])
        source = source_map.get(source_id)
        if source is None:
            row_error_counts["unknown_source_item"] += 1
            continue
        if source_id not in eligible_ids:
            row_error_counts["source_not_in_eligible_positive_pairs"] += 1
        if str(source["category"]) in forbidden_categories:
            row_error_counts["forbidden_source_category"] += 1
        if not (
            str(left["category"])
            == str(right["category"])
            == str(source["category"])
        ):
            row_error_counts["category_mismatch"] += 1
        source_attrs = json_object(source["attributes"])
        left_attrs = json_object(left["attributes"])
        right_attrs = json_object(right["attributes"])
        if str(left["name"]) != str(source["name"]) or left_attrs != source_attrs:
            row_error_counts["left_is_not_source_clone"] += 1
        if token_multiset(right["name"]) != token_multiset(source["name"]):
            row_error_counts["title_fact_tokens_changed"] += 1
        if str(right["name"]) == str(left["name"]) and right_attrs == left_attrs:
            row_error_counts["pair_has_no_surface_change"] += 1
        try:
            right_to_source = json.loads(meta["right_key_to_source_key_json"])
            omitted = json.loads(meta["omitted_keys_json"])
        except (TypeError, json.JSONDecodeError):
            row_error_counts["invalid_provenance_json"] += 1
            continue
        if not isinstance(right_to_source, dict) or not isinstance(omitted, list):
            row_error_counts["invalid_provenance_json_shape"] += 1
            continue
        if set(right_to_source) != set(right_attrs):
            row_error_counts["right_key_mapping_mismatch"] += 1
        mapped_source_keys = list(right_to_source.values())
        if len(mapped_source_keys) != len(set(mapped_source_keys)):
            row_error_counts["non_bijective_key_mapping"] += 1
        if set(mapped_source_keys) | set(omitted) != set(source_attrs):
            row_error_counts["source_fact_partition_mismatch"] += 1
        if set(mapped_source_keys) & set(omitted):
            row_error_counts["mapped_and_omitted_overlap"] += 1
        if any(identity_key(key) for key in omitted):
            row_error_counts["identity_attribute_omitted"] += 1
        for right_key, source_key in right_to_source.items():
            if right_key not in right_attrs or source_key not in source_attrs:
                row_error_counts["mapped_key_missing"] += 1
                continue
            if right_key != source_key and right_key not in key_variants(source_key):
                row_error_counts["unsafe_attribute_key_alias"] += 1
            if canonical_value(right_attrs[right_key]) != canonical_value(
                source_attrs[source_key]
            ):
                row_error_counts["attribute_value_fact_changed"] += 1
        source_facts_sha = _facts_sha(
            str(source["name"]), source_attrs, str(source["category"])
        )
        if str(meta["source_facts_sha256"]) != source_facts_sha:
            row_error_counts["source_facts_hash_mismatch"] += 1
        retained_source_attrs = {
            source_key: source_attrs[source_key]
            for source_key in mapped_source_keys
            if source_key in source_attrs
        }
        retained_right_attrs = {
            source_key: right_attrs[right_key]
            for right_key, source_key in right_to_source.items()
            if right_key in right_attrs and source_key in source_attrs
        }
        retained_source_sha = _facts_sha(
            str(source["name"]), retained_source_attrs, str(source["category"])
        )
        retained_right_sha = _facts_sha(
            str(right["name"]), retained_right_attrs, str(right["category"])
        )
        if str(meta["retained_source_facts_sha256"]) != retained_source_sha:
            row_error_counts["retained_source_hash_mismatch"] += 1
        if str(meta["retained_right_facts_sha256"]) != retained_right_sha:
            row_error_counts["retained_right_hash_mismatch"] += 1
        if retained_source_sha != retained_right_sha:
            row_error_counts["retained_facts_differ"] += 1
        checked_rows += 1

    if row_error_counts:
        errors.extend(
            f"{reason}:{count_value}"
            for reason, count_value in sorted(row_error_counts.items())
        )
    return {
        "version": VALIDATION_VERSION,
        "valid": not errors,
        "pairs": count,
        "items": len(items),
        "checked_pairs": checked_rows,
        "target_counts": {"0": int(targets.eq(0).sum()), "1": int(targets.eq(1).sum())},
        "unique_source_items": int(provenance["source_item_id"].nunique()),
        "forbidden_categories": sorted(forbidden_categories),
        "no_new_or_changed_visible_facts": not row_error_counts,
        "errors": errors,
        "warnings": warnings,
    }


def _numeric_summary(values: Iterable[int | float]) -> dict[str, float | int]:
    series = pd.Series(list(values), dtype="float64")
    if series.empty:
        return {"count": 0}
    return {
        "count": int(len(series)),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p10": float(series.quantile(0.1)),
        "p90": float(series.quantile(0.9)),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def _pair_distribution(
    pairs: Sequence[Mapping[str, Any]], item_map: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    metrics: dict[str, list[int]] = {
        "left_attribute_count": [],
        "right_attribute_count": [],
        "left_name_tokens": [],
        "right_name_tokens": [],
        "exact_common_keys": [],
        "same_value_common_keys": [],
        "conflicting_common_keys": [],
        "left_only_keys": [],
        "right_only_keys": [],
    }
    category_counts: Counter[str] = Counter()
    for pair in pairs:
        left, right = item_map[int(pair["id1"])], item_map[int(pair["id2"])]
        left_attrs, right_attrs = json_object(left["attributes"]), json_object(
            right["attributes"]
        )
        left_by_key = {normalized_key(key): value for key, value in left_attrs.items()}
        right_by_key = {
            normalized_key(key): value for key, value in right_attrs.items()
        }
        common = set(left_by_key) & set(right_by_key)
        same = sum(
            canonical_value(left_by_key[key]) == canonical_value(right_by_key[key])
            for key in common
        )
        metrics["left_attribute_count"].append(len(left_attrs))
        metrics["right_attribute_count"].append(len(right_attrs))
        metrics["left_name_tokens"].append(len(fact_tokens(left["name"])))
        metrics["right_name_tokens"].append(len(fact_tokens(right["name"])))
        metrics["exact_common_keys"].append(len(common))
        metrics["same_value_common_keys"].append(same)
        metrics["conflicting_common_keys"].append(len(common) - same)
        metrics["left_only_keys"].append(len(set(left_by_key) - common))
        metrics["right_only_keys"].append(len(set(right_by_key) - common))
        category_counts[str(left["category"])] += 1
    total = len(pairs)
    return {
        "pairs": total,
        "metrics": {key: _numeric_summary(values) for key, values in metrics.items()},
        "category_counts": dict(sorted(category_counts.items())),
        "category_fractions": {
            key: value / total for key, value in sorted(category_counts.items())
        }
        if total
        else {},
    }


def distribution_report(
    *,
    source_items: pd.DataFrame,
    eligible_pairs: pd.DataFrame,
    generated_items: pd.DataFrame,
    generated_pairs: pd.DataFrame,
    provenance: pd.DataFrame,
    forbidden_categories: frozenset[str],
) -> dict[str, Any]:
    source_map = {
        int(row.id): row._asdict() for row in source_items.itertuples(index=False)
    }
    generated_map = {
        int(row.id): row._asdict() for row in generated_items.itertuples(index=False)
    }
    reference_rows: list[dict[str, Any]] = []
    positive = eligible_pairs.loc[
        pd.to_numeric(eligible_pairs["target"], errors="coerce").eq(1)
    ]
    for row in positive.itertuples(index=False):
        left, right = source_map.get(int(row.id1)), source_map.get(int(row.id2))
        if left is None or right is None:
            continue
        if (
            str(left["category"]) != str(right["category"])
            or str(left["category"]) in forbidden_categories
        ):
            continue
        try:
            json_object(left["attributes"])
            json_object(right["attributes"])
        except SurfacePositiveError:
            continue
        reference_rows.append({"id1": int(row.id1), "id2": int(row.id2)})
    generated_rows = [
        {"id1": int(row.id1), "id2": int(row.id2)}
        for row in generated_pairs.itertuples(index=False)
    ]
    reference = _pair_distribution(reference_rows, source_map)
    generated = _pair_distribution(generated_rows, generated_map)
    categories = set(reference["category_fractions"]) | set(
        generated["category_fractions"]
    )
    category_tvd = 0.5 * sum(
        abs(
            reference["category_fractions"].get(category, 0.0)
            - generated["category_fractions"].get(category, 0.0)
        )
        for category in categories
    )
    alias_counts = [
        len(json.loads(value)) for value in provenance["key_aliases_json"].astype(str)
    ]
    omission_counts = [
        len(json.loads(value)) for value in provenance["omitted_keys_json"].astype(str)
    ]
    return {
        "schema_version": 1,
        "builder_version": VERSION,
        "generated": generated,
        "reference_human_train_positives": reference,
        "category_total_variation_distance": float(category_tvd),
        "augmentation": {
            "profiles": dict(
                sorted(provenance["augmentation_profile"].value_counts().items())
            ),
            "name_operations": dict(
                sorted(provenance["name_operation"].value_counts().items())
            ),
            "aliases_per_pair": _numeric_summary(alias_counts),
            "omissions_per_pair": _numeric_summary(omission_counts),
            "pairs_with_omissions": int(sum(value > 0 for value in omission_counts)),
            "pairs_with_aliases": int(sum(value > 0 for value in alias_counts)),
        },
        "interpretation": {
            "target_semantics": "same product; surface presentation changes only",
            "atomic_attribute_changes": 0,
            "left_card_policy": "exact normalized-JSON clone of a train-positive donor",
            "right_card_policy": (
                "same title token multiset and same values for every retained logical "
                "attribute; optional non-identity attributes may be absent"
            ),
        },
    }


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_examples(
    path: Path,
    pairs: pd.DataFrame,
    items: pd.DataFrame,
    provenance: pd.DataFrame,
    count: int,
    seed: int,
) -> None:
    item_map = {int(row.id): row._asdict() for row in items.itertuples(index=False)}
    meta_map = {
        (int(row.id1), int(row.id2)): row._asdict()
        for row in provenance.itertuples(index=False)
    }
    rows = sorted(
        pairs.to_dict("records"),
        key=lambda row: stable_hash(seed, "example", int(row["id1"])),
    )[: max(0, count)]
    lines = []
    for pair in rows:
        key = (int(pair["id1"]), int(pair["id2"]))
        meta = meta_map[key]
        lines.append(
            json.dumps(
                {
                    "pair": pair,
                    "left": item_map[key[0]],
                    "right": item_map[key[1]],
                    "provenance": {
                        "source_item_id": int(meta["source_item_id"]),
                        "augmentation_profile": meta["augmentation_profile"],
                        "name_operation": meta["name_operation"],
                        "key_aliases": json.loads(meta["key_aliases_json"]),
                        "omitted_keys": json.loads(meta["omitted_keys_json"]),
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    os.replace(temporary, path)


def _read_output(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(output_dir / "pairs.parquet"),
        pd.read_parquet(output_dir / "items.parquet"),
        pd.read_parquet(output_dir / "pair_provenance.parquet"),
    )


def build(
    *,
    items_path: Path,
    eligible_pairs_path: Path,
    output_dir: Path,
    count: int = DEFAULT_COUNT,
    seed: int = DEFAULT_SEED,
    id_start: int = DEFAULT_ID_START,
    min_attributes: int = 5,
    max_attributes: int = 30,
    min_name_tokens: int = 4,
    max_name_tokens: int = 30,
    forbidden_categories: frozenset[str] = DEFAULT_FORBIDDEN_CATEGORIES,
    example_count: int = 30,
    overwrite: bool = False,
) -> dict[str, Any]:
    if count < 1:
        raise SurfacePositiveError("count must be positive")
    if min_attributes < 1 or max_attributes < min_attributes:
        raise SurfacePositiveError("invalid attribute-count bounds")
    if min_name_tokens < 2 or max_name_tokens < min_name_tokens:
        raise SurfacePositiveError("invalid title-token bounds")
    items_path, eligible_pairs_path, output_dir = (
        absolute(items_path),
        absolute(eligible_pairs_path),
        absolute(output_dir),
    )
    for path in (items_path, eligible_pairs_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        output_dir / filename
        for filename in OUTPUT_FILENAMES
        if (output_dir / filename).exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "output artifacts already exist; use --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    source_items = pd.read_parquet(items_path)
    eligible_pairs = pd.read_parquet(eligible_pairs_path)
    _validate_source_frames(source_items, eligible_pairs)
    candidates, candidate_report, _ = _eligible_candidates(
        source_items,
        eligible_pairs,
        seed=seed,
        min_attributes=min_attributes,
        max_attributes=max_attributes,
        min_name_tokens=min_name_tokens,
        max_name_tokens=max_name_tokens,
        forbidden_categories=forbidden_categories,
    )
    selected, quotas = select_candidates(candidates, count, seed)
    pairs, items, provenance = materialize(selected, seed=seed, id_start=id_start)
    validation = validate_surface_dataset(
        source_items=source_items,
        eligible_pairs=eligible_pairs,
        pairs=pairs,
        items=items,
        provenance=provenance,
        forbidden_categories=forbidden_categories,
        expected_count=count,
    )
    if not validation["valid"]:
        raise SurfacePositiveError(
            "generated dataset failed validation: " + "; ".join(validation["errors"])
        )
    distribution = distribution_report(
        source_items=source_items,
        eligible_pairs=eligible_pairs,
        generated_items=items,
        generated_pairs=pairs,
        provenance=provenance,
        forbidden_categories=forbidden_categories,
    )

    atomic_parquet(items, output_dir / "items.parquet")
    atomic_parquet(pairs, output_dir / "pairs.parquet")
    atomic_parquet(provenance, output_dir / "pair_provenance.parquet")
    atomic_json(validation, output_dir / "validation_report.json")
    atomic_json(distribution, output_dir / "distribution_report.json")
    atomic_examples(
        output_dir / "examples.jsonl", pairs, items, provenance, example_count, seed
    )

    config = {
        "count": count,
        "seed": seed,
        "id_start": id_start,
        "min_attributes": min_attributes,
        "max_attributes": max_attributes,
        "min_name_tokens": min_name_tokens,
        "max_name_tokens": max_name_tokens,
        "forbidden_categories": sorted(forbidden_categories),
        "profile_config": PROFILE_CONFIG,
    }
    source_provenance = {
        "items_path": str(items_path),
        "items_sha256": sha256_file(items_path),
        "eligible_pairs_path": str(eligible_pairs_path),
        "eligible_pairs_sha256": sha256_file(eligible_pairs_path),
        "eligible_pairs_contract": (
            "Only ids occurring in target=1 rows of the caller-supplied train-pair "
            "file may be used as donors."
        ),
    }
    run_signature = sha256_json(
        {
            "version": VERSION,
            "selection_version": SELECTION_VERSION,
            "config": config,
            "source": source_provenance,
            "source_item_ids": provenance["source_item_id"].astype("int64").tolist(),
        }
    )
    summary = {
        "schema_version": 1,
        "builder_version": VERSION,
        "validation_version": VALIDATION_VERSION,
        "selection_version": SELECTION_VERSION,
        "run_signature": run_signature,
        "label_source": LABEL_SOURCE,
        "generated_pairs": count,
        "generated_items": count * 2,
        "target_counts": {"0": 0, "1": count},
        "no_atomic_change": True,
        "category_quotas": quotas,
        "candidate_pool": candidate_report,
        "config": config,
        "source_provenance": source_provenance,
        "validation": validation,
        "distribution_report": "distribution_report.json",
    }
    atomic_json(summary, output_dir / "summary.json")

    artifact_files = [
        "items.parquet",
        "pairs.parquet",
        "pair_provenance.parquet",
        "summary.json",
        "validation_report.json",
        "distribution_report.json",
        "examples.jsonl",
    ]
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "builder_version": VERSION,
        "run_signature": run_signature,
        "label_source": LABEL_SOURCE,
        "pairs": count,
        "items": count * 2,
        "targets": {"0": 0, "1": count},
        "no_atomic_change": True,
        "source_provenance": source_provenance,
        "config": config,
        "files": {
            filename: {
                "bytes": (output_dir / filename).stat().st_size,
                "sha256": sha256_file(output_dir / filename),
            }
            for filename in artifact_files
        },
    }
    atomic_json(manifest, output_dir / "build_manifest.json")
    return {
        "summary": summary,
        "validation": validation,
        "distribution": distribution,
        "manifest": manifest,
        "output_dir": str(output_dir),
    }


def validate_existing(
    *,
    items_path: Path,
    eligible_pairs_path: Path,
    output_dir: Path,
    forbidden_categories: frozenset[str],
) -> dict[str, Any]:
    items_path, eligible_pairs_path, output_dir = (
        absolute(items_path),
        absolute(eligible_pairs_path),
        absolute(output_dir),
    )
    source_items = pd.read_parquet(items_path)
    eligible_pairs = pd.read_parquet(eligible_pairs_path)
    pairs, items, provenance = _read_output(output_dir)
    report = validate_surface_dataset(
        source_items=source_items,
        eligible_pairs=eligible_pairs,
        pairs=pairs,
        items=items,
        provenance=provenance,
        forbidden_categories=forbidden_categories,
        expected_count=len(pairs),
    )
    if not report["valid"]:
        raise SurfacePositiveError(
            "existing dataset failed validation: " + "; ".join(report["errors"])
        )
    return report


def main() -> None:
    args = parse_args()
    forbidden = frozenset(
        args.forbidden_categories
        if args.forbidden_categories is not None
        else DEFAULT_FORBIDDEN_CATEGORIES
    )
    if args.validate_only:
        result = validate_existing(
            items_path=args.items,
            eligible_pairs_path=args.eligible_pairs,
            output_dir=args.output_dir,
            forbidden_categories=forbidden,
        )
    else:
        result = build(
            items_path=args.items,
            eligible_pairs_path=args.eligible_pairs,
            output_dir=args.output_dir,
            count=args.count,
            seed=args.seed,
            id_start=args.id_start,
            min_attributes=args.min_attributes,
            max_attributes=args.max_attributes,
            min_name_tokens=args.min_name_tokens,
            max_name_tokens=args.max_name_tokens,
            forbidden_categories=forbidden,
            example_count=args.example_count,
            overwrite=args.overwrite,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
