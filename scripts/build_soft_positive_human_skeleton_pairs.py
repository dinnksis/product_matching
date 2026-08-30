#!/usr/bin/env python3
"""Rebuild soft-positive A+B pairs from human-positive card skeletons.

This builder is deliberately deterministic and does not call a language model.
For every requested soft-positive task it:

1. chooses a labelled-positive human pair in the same category and with an
   explicit/title-grounded product type;
2. chooses a transition from the rule's own labelled-positive source examples;
3. removes only aliases of the target concept from the human skeleton and
   overlays the observed transition, keeping every other retained key/value
   verbatim.

The resulting pair is the combination of one naturally occurring positive
seller pair and one statistically trusted positive atom.  Row-level provenance
is sufficient to replay and validate every choice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from item_pipeline.normalization import canonical_json_dumps, parse_attributes
from scripts.freeze_generated_pair_dataset import canonical_card
from src.data_pipeline import serialize_product


VERSION = "soft_positive_human_skeleton_overlay_v2"
VALIDATION_VERSION = "soft_positive_human_skeleton_validator_v2"
SELECTION_VERSION = "exact_scope_balanced_human_positive_pair_v1"
EVIDENCE_VERSION = "label1_source_example_exact_subfacet_grounding_v2"
LABEL_SOURCE = "soft_positive_human_skeleton_ab_v1"
DEFAULT_SOURCE = ROOT / "item_pipeline/artifacts/soft_positive_tier_ab_16351_qwen_v1_composed"
DEFAULT_OUTPUT = ROOT / "item_pipeline/artifacts/soft_positive_tier_ab_16351_human_skeleton_v3"
DEFAULT_PILOT_INPUTS = ROOT / "data/qwen_atomic_differences_v2_full_train/pilot_inputs.parquet"
DEFAULT_PILOT_LABELS = ROOT / "data/qwen_atomic_differences_v2_full_train/pilot_labels.parquet"
DEFAULT_VALIDATION_DIR = ROOT / "prepared/validation_splits_v1/human"
DEFAULT_SEED = 2_026_082_831
DEFAULT_ID_START = -6_000_000_000
MIN_UNIQUE_SKELETON_FRACTION = 0.26
MAX_OVERLAY_SKELETON_REUSE = 15
MAX_FALLBACK_SOURCE_PAIR_REUSE = 35
EXPECTED_FULL_DROPPED_VALIDATION_OVERLAP = 263
VALIDATION_SPLITS = ("iid", "hard", "ood")
OVERLAY_ALLOWLIST_VERSION = "non_title_metadata_exact_signatures_v1"
FORBIDDEN_CATEGORIES = frozenset({"Одежда", "Бытовая техника"})
TYPE_KEYS = frozenset({"тип", "тип товара", "вид товара", "категория товара", "product type"})
WORD_RE = re.compile(r"[0-9a-zа-я]+", re.IGNORECASE)
COLOR_WORDS = frozenset(
    {
        "белый", "белая", "белое", "черный", "черная", "черное", "красный",
        "красная", "синий", "синяя", "зеленый", "зеленая", "фиолетовый",
        "фиолетовая", "розовый", "серый", "серебристый", "золотой", "желтый",
        "оранжевый", "бежевый", "коричневый", "голубой", "бирюзовый",
        "white", "black", "red", "blue", "green", "purple", "pink", "grey",
        "gray", "yellow", "orange", "brown", "silver",
    }
)
COUNTRY_WORDS = frozenset(
    {
        "россия", "китай", "германия", "италия", "франция", "турция", "индия",
        "япония", "сша", "беларусь", "казахстан", "украина", "швейцария",
        "польша", "испания", "корея", "южная корея", "вьетнам", "тайвань",
    }
)
BRAND_KEYS = frozenset(
    {"бренд", "brand", "марка", "торговая марка", "производитель", "изготовитель"}
)
SEMANTIC_KEY_STOP_WORDS = frozenset(
    {
        "товар", "товара", "товаре", "изделие", "изделия", "предмет", "предмета",
        "продукт", "продукта", "название", "значение", "характеристика", "параметр",
        "см", "мм", "м", "г", "кг", "л", "мл", "шт", "руб", "rub", "cm",
        "mm", "kg", "ml", "pcs", "и", "или", "для", "без", "с", "со", "в",
        "во", "на", "по", "к", "ко", "от", "до", "из", "of", "for", "with",
    }
)
WEAK_SEMANTIC_ROOTS = frozenset({"type"})
PHYSICAL_FACETS = frozenset(
    {"width", "height", "length", "depth", "diameter", "weight", "volume", "area"}
)
EXCLUSIVE_SEMANTIC_FACETS = frozenset(
    {
        *PHYSICAL_FACETS,
        "count", "brand", "model", "gender", "purpose", "color", "material",
        "country", "composition", "scent", "flavor",
    }
)
CONCEPT_PROPERTY_FACETS = frozenset(
    {
        *EXCLUSIVE_SEMANTIC_FACETS,
        "min", "max", "age", "type", "mount", "power", "voltage", "size",
        "shape", "speed", "control", "removable",
    }
)
SAFE_OVERLAY_SIGNATURES_BY_FAMILY: dict[str, tuple[frozenset[str], ...]] = {
    "warranty": (
        frozenset({"warranty"}),
        frozenset({"duration", "warranty"}),
    ),
}
OUTPUT_FILENAMES = (
    "items.parquet",
    "pairs.parquet",
    "pair_generation_metadata.parquet",
    "summary.json",
    "validation_report.json",
    "distribution_report.json",
    "build_manifest.json",
)
EXPECTED_TRAIN_SOURCE_SUFFIX = "prepared/validation_splits_v1/human/train_pairs.parquet"


class SkeletonBuildError(ValueError):
    """Raised when the grounded-skeleton contract cannot be satisfied."""


@dataclass(frozen=True)
class HumanPair:
    pair_id: str
    item_id_a: int
    item_id_b: int
    category: str
    title_a: str
    attributes_a: dict[str, str]
    title_b: str
    attributes_b: dict[str, str]


@dataclass(frozen=True)
class FrozenValidationFacts:
    fact_sources: dict[tuple[str, ...], dict[str, tuple[int, ...]]]
    split_pair_counts: dict[str, int]
    split_item_counts: dict[str, int]
    unique_item_ids: int
    unique_fact_keys: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--pilot-inputs", type=Path, default=DEFAULT_PILOT_INPUTS)
    parser.add_argument("--pilot-labels", type=Path, default=DEFAULT_PILOT_LABELS)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--id-start", type=int, default=DEFAULT_ID_START)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
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
    return hashlib.sha256(canonical_json_dumps(value).encode("utf-8")).hexdigest()


def stable_hash(seed: int, *parts: Any) -> int:
    payload = canonical_json_dumps([int(seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().replace("ё", "е")
    return " ".join(text.split())


def normalized_key(value: Any) -> str:
    text = normalize_text(value).replace("_", " ")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return " ".join(text.split())


def fact_tokens(value: Any) -> tuple[str, ...]:
    return tuple(WORD_RE.findall(normalize_text(value)))


def token_multiset(value: Any) -> Counter[str]:
    return Counter(fact_tokens(value))


def phrase_in_text(phrase: str, text: str) -> bool:
    needle = normalized_key(phrase)
    haystack = normalized_key(text)
    return bool(needle and f" {needle} " in f" {haystack} ")


def json_attributes(attributes: Mapping[str, str]) -> str:
    return json.dumps(dict(attributes), ensure_ascii=False, separators=(",", ":"))


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


def human_pair_from_row(row: Mapping[str, Any]) -> HumanPair:
    try:
        attrs_a = parse_attributes(row["attributes_a_json"])
        attrs_b = parse_attributes(row["attributes_b_json"])
    except Exception as error:
        raise SkeletonBuildError(f"invalid human attributes for {row.get('pair_id')}") from error
    title_a, title_b = str(row["title_a"]).strip(), str(row["title_b"]).strip()
    category = str(row["category"]).strip()
    if not title_a or not title_b or not category:
        raise SkeletonBuildError(f"empty human card field for {row.get('pair_id')}")
    return HumanPair(
        pair_id=str(row["pair_id"]),
        item_id_a=int(row["item_id_a"]),
        item_id_b=int(row["item_id_b"]),
        category=category,
        title_a=title_a,
        attributes_a=attrs_a,
        title_b=title_b,
        attributes_b=attrs_b,
    )


def load_human_pairs(
    inputs: pd.DataFrame, labels: pd.DataFrame
) -> tuple[dict[str, HumanPair], set[str]]:
    required_inputs = {
        "pair_id", "item_id_a", "item_id_b", "category", "title_a",
        "attributes_a_json", "title_b", "attributes_b_json",
    }
    required_labels = {"pair_id", "human_label"}
    if missing := required_inputs - set(inputs.columns):
        raise SkeletonBuildError(f"pilot inputs missing columns: {sorted(missing)}")
    if missing := required_labels - set(labels.columns):
        raise SkeletonBuildError(f"pilot labels missing columns: {sorted(missing)}")
    label_rows = labels.copy()
    if label_rows["pair_id"].astype(str).duplicated().any():
        raise SkeletonBuildError("pilot labels contain duplicate pair_id")
    positive_ids = set(
        label_rows.loc[pd.to_numeric(label_rows["human_label"], errors="coerce").eq(1), "pair_id"]
        .astype(str)
        .tolist()
    )
    input_rows = inputs.copy()
    input_rows["pair_id"] = input_rows["pair_id"].astype(str)
    if input_rows["pair_id"].duplicated().any():
        raise SkeletonBuildError("pilot inputs contain duplicate pair_id")
    pairs: dict[str, HumanPair] = {}
    for raw in input_rows.loc[input_rows["pair_id"].isin(positive_ids)].to_dict("records"):
        pair = human_pair_from_row(raw)
        pairs[pair.pair_id] = pair
    missing_positive = positive_ids - set(pairs)
    if missing_positive:
        raise SkeletonBuildError(f"{len(missing_positive)} positive labels lack input rows")
    return pairs, positive_ids


def validate_pilot_manifest(
    manifest_path: Path, inputs: pd.DataFrame, labels: pd.DataFrame
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise SkeletonBuildError(f"missing pilot train manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SkeletonBuildError(f"invalid pilot train manifest: {manifest_path}") from error
    source_train = str(manifest.get("source_train") or "").replace("\\", "/")
    if not source_train.endswith(EXPECTED_TRAIN_SOURCE_SUFFIX):
        raise SkeletonBuildError("pilot manifest is not pinned to the human train split")
    if manifest.get("sampling_mode") != "all" or manifest.get("labels_stored_separately") is not True:
        raise SkeletonBuildError("pilot manifest sampling/label contract changed")
    if int(manifest.get("source_pairs", -1)) != len(inputs) or int(manifest.get("pilot_pairs", -1)) != len(inputs):
        raise SkeletonBuildError("pilot manifest pair counts do not match inputs")
    observed_counts = {
        str(int(label)): int(count)
        for label, count in pd.to_numeric(labels["human_label"], errors="coerce").value_counts().sort_index().items()
    }
    expected_counts = {
        str(key): int(value)
        for key, value in (manifest.get("label_counts") or {}).items()
        if int(value) != 0
    }
    if observed_counts != expected_counts:
        raise SkeletonBuildError(
            f"pilot manifest label counts differ: {observed_counts} != {expected_counts}"
        )
    if int(manifest.get("categories", -1)) != int(inputs["category"].nunique()):
        raise SkeletonBuildError("pilot manifest category count differs")
    return manifest


def parse_tasks(metadata: pd.DataFrame, count: int | None = None) -> list[dict[str, Any]]:
    required = {"category", "product_type", "rules_json", "composition_index"}
    if missing := required - set(metadata.columns):
        raise SkeletonBuildError(f"source metadata missing columns: {sorted(missing)}")
    ordered = metadata.sort_values("composition_index", kind="stable")
    if count is not None:
        if count < 1 or count > len(ordered):
            raise SkeletonBuildError(f"count must be in [1, {len(ordered)}]")
        ordered = ordered.iloc[:count]
    if ordered["composition_index"].duplicated().any():
        raise SkeletonBuildError("composition_index must be unique")
    tasks: list[dict[str, Any]] = []
    for raw in ordered.to_dict("records"):
        try:
            rules = json.loads(str(raw["rules_json"]))
        except json.JSONDecodeError as error:
            raise SkeletonBuildError("invalid rules_json") from error
        if not isinstance(rules, list) or len(rules) != 1 or not isinstance(rules[0], dict):
            raise SkeletonBuildError("each A+B task must contain exactly one rule")
        rule = rules[0]
        if int(rule.get("label", -1)) != 1:
            raise SkeletonBuildError("all source rules must have label=1")
        allowed_categories = {str(value) for value in rule.get("allowed_categories") or []}
        allowed_types = {normalize_text(value) for value in rule.get("allowed_product_types") or []}
        category, product_type = str(raw["category"]), str(raw["product_type"])
        if allowed_categories and category not in allowed_categories:
            raise SkeletonBuildError("task category is outside rule scope")
        if allowed_types and normalize_text(product_type) not in allowed_types:
            raise SkeletonBuildError("task product_type is outside rule scope")
        if category in FORBIDDEN_CATEGORIES:
            raise SkeletonBuildError(f"task leaks frozen OOD category: {category}")
        tasks.append(
            {
                "composition_index": int(raw["composition_index"]),
                "category": category,
                "product_type": product_type,
                "scope": (category, normalized_key(product_type)),
                "rule": rule,
                "source_metadata": raw,
            }
        )
    return tasks


def explicit_types(attributes: Mapping[str, str]) -> set[str]:
    return {
        normalized_key(value)
        for key, value in attributes.items()
        if normalized_key(key) in TYPE_KEYS and normalized_key(value)
    }


def pair_scope_match(pair: HumanPair, product_type: str) -> str | None:
    target = normalized_key(product_type)
    types = explicit_types(pair.attributes_a) | explicit_types(pair.attributes_b)
    if target in types:
        return "explicit_type"
    if phrase_in_text(target, pair.title_a) or phrase_in_text(target, pair.title_b):
        return "title_exact_phrase"
    return None


def build_scope_pools(
    tasks: Sequence[Mapping[str, Any]], human_pairs: Mapping[str, HumanPair]
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    scopes_by_category: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        category, product_type = task["scope"]
        scopes_by_category[category].add(product_type)
    pools: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for pair_id in sorted(human_pairs):
        pair = human_pairs[pair_id]
        if not pair.attributes_a or not pair.attributes_b:
            continue
        type_values = explicit_types(pair.attributes_a) | explicit_types(pair.attributes_b)
        title_a = normalized_key(pair.title_a)
        title_b = normalized_key(pair.title_b)
        for product_type in scopes_by_category.get(pair.category, ()):
            if product_type in type_values:
                mode = "explicit_type"
            elif (
                f" {product_type} " in f" {title_a} "
                or f" {product_type} " in f" {title_b} "
            ):
                mode = "title_exact_phrase"
            else:
                mode = None
            if mode:
                pools[(pair.category, product_type)].append((pair_id, mode))
    missing = sorted({task["scope"] for task in tasks if not pools.get(task["scope"])})
    if missing:
        preview = ", ".join(f"{category}/{ptype}" for category, ptype in missing[:10])
        raise SkeletonBuildError(f"{len(missing)} scopes lack exact human-positive skeletons: {preview}")
    return dict(pools)


def assign_skeletons(
    tasks: Sequence[Mapping[str, Any]],
    pools: Mapping[tuple[str, str], Sequence[tuple[str, str]]],
    *,
    seed: int,
) -> dict[int, dict[str, Any]]:
    """Assign exact-scope skeletons while minimizing global pair reuse."""

    use_counts: Counter[str] = Counter()
    assignments: dict[int, dict[str, Any]] = {}
    by_scope: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_scope[task["scope"]].append(task)
    for scope in sorted(by_scope):
        candidates = list(pools[scope])
        candidates.sort(key=lambda item: (stable_hash(seed, "pool", *scope, item[0]), item[0]))
        for local_index, task in enumerate(sorted(by_scope[scope], key=lambda row: row["composition_index"])):
            pair_id, match_mode = min(
                candidates,
                key=lambda item: (
                    use_counts[item[0]],
                    stable_hash(seed, "assignment", *scope, local_index, item[0]),
                    item[0],
                ),
            )
            reuse_index = use_counts[pair_id]
            use_counts[pair_id] += 1
            assignments[int(task["composition_index"])] = {
                "pair_id": pair_id,
                "match_mode": match_mode,
                "reuse_index": reuse_index,
            }
    for assignment in assignments.values():
        assignment["reuse_count_final"] = use_counts[assignment["pair_id"]]
    if len(assignments) != len(tasks):
        raise AssertionError("skeleton assignment count mismatch")
    return assignments


def _key_similarity(required_key: str, candidate_key: str) -> float:
    left, right = normalized_key(required_key), normalized_key(candidate_key)
    if left == right:
        return 1.0
    left_tokens, right_tokens = set(left.split()), set(right.split())
    jaccard = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    return max(jaccard, SequenceMatcher(None, left, right).ratio() * 0.65)


def _semantic_token_root(token: str) -> str | None:
    """Map common Russian/English attribute-key morphology to a stable facet."""

    token = normalized_key(token)
    if not token or token in SEMANTIC_KEY_STOP_WORDS or token.isdigit():
        return None
    patterns: tuple[tuple[tuple[str, ...], str], ...] = (
        (("миним", "мин", "minimum", "min"), "min"),
        (("максим", "макс", "maximum", "max"), "max"),
        (("ширин", "width"), "width"),
        (("высот", "height"), "height"),
        (("длин", "length"), "length"),
        (("глубин", "depth"), "depth"),
        (("диаметр", "diameter"), "diameter"),
        (("вес", "масс", "weight"), "weight"),
        (("объем", "объём", "volume", "capacity"), "volume"),
        (("площад", "area"), "area"),
        (("стран", "country"), "country"),
        (("количеств", "числ", "count", "quantity", "number"), "count"),
        (("игрок", "player"), "player"),
        (("возраст", "age"), "age"),
        (("размер", "габарит", "dimension", "size"), "size"),
        (("цвет", "color", "colour"), "color"),
        (("материал", "material"), "material"),
        (("бренд", "brand", "марка"), "brand"),
        (("модел", "model"), "model"),
        (("пол", "gender", "sex"), "gender"),
        (("назначен", "purpose", "use"), "purpose"),
        (("состав", "ингреди", "composition", "ingredient"), "composition"),
        (("хранен", "хранён", "storage"), "storage"),
        (("услови", "condition"), "condition"),
        (("гарант", "warrant", "guarantee"), "warranty"),
        (("срок", "period", "duration"), "duration"),
        (("сертиф", "certif"), "certification"),
        (("аромат", "запах", "scent", "fragrance"), "scent"),
        (("вкус", "flavor", "flavour"), "flavor"),
        (("облож", "cover"), "cover"),
        (("упаков", "package", "packaging"), "package"),
        (("тип", "вид", "type", "kind"), "type"),
        (("производ", "изготов", "origin", "manufactur"), "origin"),
        (("креплен", "крепеж", "mount", "binding"), "mount"),
        (("мощност", "power"), "power"),
        (("напряжен", "voltage"), "voltage"),
        (("форм", "shape"), "shape"),
        (("скорост", "speed"), "speed"),
        (("управлен", "control"), "control"),
        (("съем", "съём", "remov"), "removable"),
    )
    for prefixes, root in patterns:
        if any(token.startswith(prefix) for prefix in prefixes):
            return root
    # A short deterministic stem handles ordinary inflection without claiming
    # synonymy.  Six characters avoids collisions between unrelated short keys.
    return token[:6] if len(token) >= 6 else token


def _semantic_key_features(key: Any) -> set[str]:
    text = normalized_key(key)
    return {
        root
        for token in text.split()
        if (root := _semantic_token_root(token)) is not None
    }


def _semantic_subfacet_signature(key: Any) -> frozenset[str]:
    """Return the complete meaningful key signature, not just shared words.

    ``type`` is presentation noise when a more precise property such as
    material or color is present (``тип материала`` == ``материал``).  It is
    retained for genuine type fields such as ``тип обложки``.
    """

    features = _semantic_key_features(key)
    precise_properties = (features & CONCEPT_PROPERTY_FACETS) - {"type", "min", "max"}
    if "type" in features and precise_properties:
        features.remove("type")
    return frozenset(features)


def _safe_overlay_family(signature: frozenset[str]) -> str | None:
    for family, signatures in SAFE_OVERLAY_SIGNATURES_BY_FAMILY.items():
        if signature in signatures:
            return family
    return None


def overlay_allowlist_definition() -> dict[str, Any]:
    return {
        "version": OVERLAY_ALLOWLIST_VERSION,
        "families": {
            family: [sorted(signature) for signature in signatures]
            for family, signatures in sorted(SAFE_OVERLAY_SIGNATURES_BY_FAMILY.items())
        },
    }


def overlay_allowlist_report(metadata: pd.DataFrame) -> dict[str, Any]:
    report = overlay_allowlist_definition()
    family = metadata["overlay_allowlist_family"].astype(str).replace("", "not_allowlisted")
    report["task_counts_by_family"] = {
        str(key): int(value) for key, value in family.value_counts().sort_index().items()
    }
    report["construction_mode_counts_by_family"] = {
        str(mode): {
            str(key): int(value)
            for key, value in family.loc[metadata["construction_mode"].eq(mode)]
            .value_counts()
            .sort_index()
            .items()
        }
        for mode in sorted(metadata["construction_mode"].astype(str).unique())
    }
    return report


def _opposite_polarity(left: set[str], right: set[str]) -> bool:
    return ("min" in left and "max" in right) or ("max" in left and "min" in right)


def semantic_key_compatible_for_overlay(
    required_key: Any,
    key_a: Any,
    key_b: Any,
    concept: Any = "",
) -> bool:
    """Prove that both raw keys describe exactly the requested subfacet.

    Common object words are not proof: ``материал корпуса`` and ``материал
    обивки`` are different subfacets, as are ``форма столешницы`` and
    ``материал столешницы``.  After deterministic morphology and harmless
    schema-word removal, the *complete* signatures must match and occur in the
    versioned non-title-metadata allowlist.  The concept may additionally
    declare a property (for example ``packaging_material``); that property must
    be explicit in the required/raw key rather than inferred from coincident
    values.  Lexical similarity is never authorization.
    """

    required_norm = normalized_key(required_key)
    key_a_norm = normalized_key(key_a)
    key_b_norm = normalized_key(key_b)
    if not required_norm or not key_a_norm or not key_b_norm:
        return False
    required = _semantic_subfacet_signature(required_key)
    features_a = _semantic_subfacet_signature(key_a)
    features_b = _semantic_subfacet_signature(key_b)
    if not required or not features_a or not features_b:
        return False

    if required != features_a or required != features_b:
        return False
    if _safe_overlay_family(required) is None:
        return False

    concept_properties = _semantic_key_features(concept) & CONCEPT_PROPERTY_FACETS
    if concept_properties and not concept_properties <= required:
        return False

    # Polarity is part of the exact signature.  Keep the explicit check for a
    # readable fail-closed invariant and for future signature normalization.
    if (
        _opposite_polarity(set(required), set(features_a))
        or _opposite_polarity(set(required), set(features_b))
        or _opposite_polarity(set(features_a), set(features_b))
    ):
        return False
    return True


def _matching_attr_keys(attributes: Mapping[str, str], target_value: Any) -> list[str]:
    target = normalize_text(target_value)
    return [key for key, value in attributes.items() if normalize_text(value) == target]


def _choose_matching_key(keys: Sequence[str], required_key: str) -> str | None:
    if not keys:
        return None
    return min(keys, key=lambda key: (-_key_similarity(required_key, key), normalized_key(key), key))


def _fallback_schema_value_safe(key: str, value: str) -> bool:
    key_norm, value_norm = normalized_key(key), normalize_text(value)
    if key_norm in BRAND_KEYS:
        if value_norm in COLOR_WORDS or value_norm in COUNTRY_WORDS:
            return False
        if not value_norm or value_norm.isdigit():
            return False
    return bool(key_norm and value_norm)


def build_rule_evidence(
    rule: Mapping[str, Any], human_pairs: Mapping[str, HumanPair]
) -> dict[str, Any]:
    """Build replayable transition candidates for one label-1 rule.

    ``both_raw_attributes`` is strongest.  ``attribute_or_title`` retains an
    endpoint visible in the source title when its attribute key is absent.
    ``source_example_extracted`` is the explicit low-confidence fallback for
    atoms extracted from composite source fields.  It still uses only values
    stored in a source example linked to an existing human-positive pair.
    """

    if int(rule.get("label", -1)) != 1:
        raise SkeletonBuildError("transition rule is not label=1")
    rule_id = str(rule.get("generation_rule_id") or "")
    required_key = str(rule.get("required_attribute_key") or "").strip()
    examples = rule.get("source_examples") or []
    allowed_categories = {str(value) for value in rule.get("allowed_categories") or []}
    allowed_types = [str(value) for value in rule.get("allowed_product_types") or []]
    if not rule_id or not required_key or not isinstance(examples, list) or not examples:
        raise SkeletonBuildError(f"rule {rule_id or '<unknown>'} lacks required evidence")

    inspected: list[dict[str, Any]] = []
    raw_key_counts: Counter[str] = Counter()
    raw_key_display: dict[str, str] = {}
    for source_index, source in enumerate(examples):
        if not isinstance(source, dict):
            continue
        pair_id = str(source.get("source_pair_id") or "")
        pair = human_pairs.get(pair_id)
        if pair is None:
            continue
        if not pair.attributes_a or not pair.attributes_b:
            continue
        if allowed_categories and pair.category not in allowed_categories:
            continue
        grounded_types = [ptype for ptype in allowed_types if pair_scope_match(pair, ptype)]
        if allowed_types and not grounded_types:
            continue
        value_a = str(source.get("target_value_a") or "").strip()
        value_b = str(source.get("target_value_b") or "").strip()
        if not value_a or not value_b or normalize_text(value_a) == normalize_text(value_b):
            continue
        keys_a = _matching_attr_keys(pair.attributes_a, value_a)
        keys_b = _matching_attr_keys(pair.attributes_b, value_b)
        key_a = _choose_matching_key(keys_a, required_key)
        key_b = _choose_matching_key(keys_b, required_key)
        for key in (key_a, key_b):
            if key:
                norm = normalized_key(key)
                raw_key_counts[norm] += 1
                raw_key_display.setdefault(norm, key)
        inspected.append(
            {
                "source_index": source_index,
                "source": source,
                "pair": pair,
                "grounded_product_types": grounded_types,
                "source_value_a": value_a,
                "source_value_b": value_b,
                "raw_key_a": key_a,
                "raw_key_b": key_b,
                "raw_match_count_a": len(keys_a),
                "raw_match_count_b": len(keys_b),
                "title_a": normalize_text(value_a) in normalize_text(pair.title_a),
                "title_b": normalize_text(value_b) in normalize_text(pair.title_b),
            }
        )
    if not inspected:
        raise SkeletonBuildError(f"rule {rule_id} has no usable label=1 source examples")

    if raw_key_counts:
        dominant_norm = min(
            raw_key_counts,
            key=lambda key: (-raw_key_counts[key], -_key_similarity(required_key, key), key),
        )
        dominant_key = raw_key_display[dominant_norm]
    else:
        dominant_key = required_key

    candidates: list[dict[str, Any]] = []
    observed_aliases = {normalized_key(required_key), normalized_key(dominant_key)}
    for row in inspected:
        pair: HumanPair = row["pair"]
        key_a = row["raw_key_a"] or dominant_key
        key_b = row["raw_key_b"] or dominant_key
        raw_a = pair.attributes_a.get(row["raw_key_a"]) if row["raw_key_a"] else None
        raw_b = pair.attributes_b.get(row["raw_key_b"]) if row["raw_key_b"] else None
        seen_a = bool(row["raw_key_a"] or row["title_a"])
        seen_b = bool(row["raw_key_b"] or row["title_b"])
        if (
            row["raw_key_a"]
            and row["raw_key_b"]
            and row["raw_match_count_a"] == 1
            and row["raw_match_count_b"] == 1
        ):
            mode, rank = "both_raw_attributes_unique", 4
            value_a, value_b = str(raw_a), str(raw_b)
        elif row["raw_key_a"] and row["raw_key_b"]:
            mode, rank = "both_raw_attributes_ambiguous", 3
            value_a, value_b = str(raw_a), str(raw_b)
        elif seen_a and seen_b:
            mode, rank = "attribute_or_title", 2
            value_a = str(raw_a) if raw_a is not None else row["source_value_a"]
            value_b = str(raw_b) if raw_b is not None else row["source_value_b"]
        else:
            mode, rank = "source_example_extracted", 1
            value_a, value_b = row["source_value_a"], row["source_value_b"]
        if normalize_text(value_a) == normalize_text(value_b):
            continue
        semantic_keys_compatible = semantic_key_compatible_for_overlay(
            required_key,
            key_a,
            key_b,
            rule.get("concept") or "",
        )
        schema_safe_for_overlay = (
            _fallback_schema_value_safe(key_a, value_a)
            and _fallback_schema_value_safe(key_b, value_b)
            and semantic_keys_compatible
        )
        if row["raw_key_a"]:
            observed_aliases.add(normalized_key(row["raw_key_a"]))
        if row["raw_key_b"]:
            observed_aliases.add(normalized_key(row["raw_key_b"]))
        source = row["source"]
        candidates.append(
            {
                "source_pair_id": pair.pair_id,
                "source_example_index": int(row["source_index"]),
                "source_is_singleton": bool(source.get("source_is_singleton", False)),
                "grounding_mode": mode,
                "grounding_rank": rank,
                "schema_safe_for_overlay": schema_safe_for_overlay,
                "semantic_keys_compatible_for_overlay": semantic_keys_compatible,
                "target_key_a": key_a,
                "target_key_b": key_b,
                "target_value_a": value_a,
                "target_value_b": value_b,
                "source_example_value_a": row["source_value_a"],
                "source_example_value_b": row["source_value_b"],
                "raw_attribute_key_a": row["raw_key_a"] or "",
                "raw_attribute_key_b": row["raw_key_b"] or "",
                "raw_attribute_match_count_a": int(row["raw_match_count_a"]),
                "raw_attribute_match_count_b": int(row["raw_match_count_b"]),
                "endpoint_a_visible_in_title": bool(row["title_a"]),
                "endpoint_b_visible_in_title": bool(row["title_b"]),
                "grounded_product_types": list(row["grounded_product_types"]),
            }
        )
    if not candidates:
        raise SkeletonBuildError(f"rule {rule_id} has no schema-safe grounded transition")
    candidates.sort(
        key=lambda row: (
            -int(row["grounding_rank"]),
            -int(row["source_is_singleton"]),
            str(row["source_pair_id"]),
            int(row["source_example_index"]),
        )
    )
    return {
        "rule_id": rule_id,
        "required_key": required_key,
        "dominant_key": dominant_key,
        "concept": str(rule.get("concept") or ""),
        "observed_aliases": sorted(alias for alias in observed_aliases if alias),
        "candidates": candidates,
    }


def build_all_evidence(
    tasks: Sequence[Mapping[str, Any]], human_pairs: Mapping[str, HumanPair]
) -> dict[str, dict[str, Any]]:
    rules: dict[str, Mapping[str, Any]] = {}
    for task in tasks:
        rule = task["rule"]
        rule_id = str(rule["generation_rule_id"])
        if rule_id in rules and canonical_json_dumps(rules[rule_id]) != canonical_json_dumps(rule):
            raise SkeletonBuildError(f"rule payload changed across tasks: {rule_id}")
        rules[rule_id] = rule
    return {rule_id: build_rule_evidence(rule, human_pairs) for rule_id, rule in sorted(rules.items())}


def choose_task_evidence(
    bank: Mapping[str, Any],
    task: Mapping[str, Any],
    human_pairs: Mapping[str, HumanPair],
    *,
    rule_task_index: int,
    seed: int,
) -> dict[str, Any]:
    scoped = [
        row
        for row in bank["candidates"]
        if human_pairs[row["source_pair_id"]].category == task["category"]
        and pair_scope_match(human_pairs[row["source_pair_id"]], task["product_type"])
    ]
    if not scoped:
        raise SkeletonBuildError(
            f"rule {bank['rule_id']} has no transition evidence for exact scope {task['scope']}"
        )
    best_rank = max(int(row["grounding_rank"]) for row in scoped)
    strongest = [row for row in scoped if int(row["grounding_rank"]) == best_rank]
    singleton = [row for row in strongest if row["source_is_singleton"]]
    choices = singleton or strongest
    choices.sort(key=lambda row: (str(row["source_pair_id"]), int(row["source_example_index"])))
    offset = stable_hash(seed, "task-evidence", bank["rule_id"], *task["scope"]) % len(choices)
    return dict(choices[(offset + int(rule_task_index)) % len(choices)])


def selected_target_aliases(
    evidence_bank: Mapping[str, Any], evidence: Mapping[str, Any]
) -> set[str]:
    """Return only the two exact raw keys of this chosen transition.

    Even a semantically plausible bank alias is not selected evidence for this
    row.  It may be used to reject a contradictory skeleton, but must never be
    deleted from one.
    """

    del evidence_bank
    return {
        alias
        for alias in (
            normalized_key(evidence["target_key_a"]),
            normalized_key(evidence["target_key_b"]),
        )
        if alias
    }


def _semantically_target_key(
    evidence_bank: Mapping[str, Any], evidence: Mapping[str, Any], key: Any
) -> bool:
    """Whether ``key`` is an exact-subfacet alias of the chosen transition."""

    required_key = str(evidence_bank["required_key"])
    concept = str(evidence_bank.get("concept") or "")
    return semantic_key_compatible_for_overlay(
        required_key,
        str(evidence["target_key_a"]),
        key,
        concept,
    ) and semantic_key_compatible_for_overlay(
        required_key,
        str(evidence["target_key_b"]),
        key,
        concept,
    )


def _selected_target_key_present_on_both_sides(
    skeleton: HumanPair, target_aliases: set[str]
) -> bool:
    """Require a physically removable selected target field on each card."""

    if not target_aliases:
        return False
    return all(
        any(normalized_key(key) in target_aliases for key in attributes)
        for attributes in (skeleton.attributes_a, skeleton.attributes_b)
    )


def _target_title_conflict(
    skeleton: HumanPair,
    evidence_bank: Mapping[str, Any],
    evidence: Mapping[str, Any],
    target_aliases: set[str],
) -> bool:
    """Reject contradictory target facts that an overlay would leave behind.

    Selected exact keys may be removed, provided their old value is not still
    stated in the title.  A different exact-subfacet alias is never removed;
    if it carries a conflicting value the whole skeleton is skipped.
    """

    safe_family = _safe_overlay_family(
        _semantic_subfacet_signature(str(evidence_bank["required_key"]))
    )
    for side in ("a", "b"):
        attributes = getattr(skeleton, f"attributes_{side}")
        title = getattr(skeleton, f"title_{side}")
        target_key = str(evidence[f"target_key_{side}"])
        target_value = str(evidence[f"target_value_{side}"])
        for key, value in attributes.items():
            key_norm = normalized_key(key)
            selected = key_norm in target_aliases or key_norm == normalized_key(target_key)
            if (
                safe_family == "warranty"
                and not selected
                and any(
                    marker in key_norm
                    for marker in (
                        "срок службы",
                        "срок эксплуатации",
                        "ресурс",
                        "service life",
                        "lifetime",
                    )
                )
            ):
                return True
            semantic_target = _semantically_target_key(evidence_bank, evidence, key)
            if not selected and not semantic_target:
                continue
            if normalize_text(value) == normalize_text(target_value):
                continue
            if semantic_target and not selected:
                return True
            if phrase_in_text(str(value), title):
                return True
    return False


def build_task_plan(
    tasks: Sequence[Mapping[str, Any]],
    pools: Mapping[tuple[str, str], Sequence[tuple[str, str]]],
    human_pairs: Mapping[str, HumanPair],
    evidence_by_rule: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
) -> dict[int, dict[str, Any]]:
    """Choose the construction mode, evidence, and human pair per task.

    Only an unambiguous raw-attribute transition may be transplanted to an
    alternate skeleton.  Every weaker extraction stays in its original human
    source pair; this preserves the atom in exactly the representation from
    which it was inferred and prevents assignments such as ``brand: purple``.
    """

    use_counts: Counter[str] = Counter()
    mode_use_counts: dict[str, Counter[str]] = defaultdict(Counter)
    rule_seen: Counter[str] = Counter()
    plan: dict[int, dict[str, Any]] = {}
    compatibility_cache: dict[tuple[Any, ...], bool] = {}
    for task in tasks:
        task_index = int(task["composition_index"])
        rule_id = str(task["rule"]["generation_rule_id"])
        bank = evidence_by_rule[rule_id]
        evidence = choose_task_evidence(
            bank,
            task,
            human_pairs,
            rule_task_index=rule_seen[rule_id],
            seed=seed,
        )
        rule_seen[rule_id] += 1
        aliases = selected_target_aliases(bank, evidence)
        construction_mode = "overlay"
        skeleton_choice: tuple[str, str] | None = None
        if (
            evidence["grounding_mode"] == "both_raw_attributes_unique"
            and evidence["schema_safe_for_overlay"]
        ):
            ranked_skeletons = sorted(
                pools[task["scope"]],
                key=lambda item: (
                    mode_use_counts["overlay"][item[0]],
                    use_counts[item[0]],
                    stable_hash(seed, "compatible-skeleton", task_index, item[0]),
                    item[0],
                ),
            )
            evidence_signature = (
                evidence["target_key_a"], evidence["target_value_a"],
                evidence["target_key_b"], evidence["target_value_b"], tuple(sorted(aliases)),
            )
            for candidate in ranked_skeletons:
                if (
                    mode_use_counts["overlay"][candidate[0]]
                    >= MAX_OVERLAY_SKELETON_REUSE
                ):
                    continue
                cache_key = (candidate[0], *evidence_signature)
                compatible = compatibility_cache.get(cache_key)
                if compatible is None:
                    candidate_pair = human_pairs[candidate[0]]
                    compatible = (
                        _selected_target_key_present_on_both_sides(
                            candidate_pair, aliases
                        )
                        and not _target_title_conflict(
                            candidate_pair, bank, evidence, aliases
                        )
                    )
                    compatibility_cache[cache_key] = compatible
                if compatible:
                    skeleton_choice = candidate
                    break
        if skeleton_choice is None:
            construction_mode = "source_pair_surface"
            # A fallback must itself ground the requested category/type.  Cycle
            # among source examples so repeated rules do not clone one pair.
            scoped = [
                row
                for row in bank["candidates"]
                if pair_scope_match(human_pairs[row["source_pair_id"]], task["product_type"])
            ]
            if not scoped:
                raise SkeletonBuildError(
                    f"rule {rule_id} has no source pair for exact scope {task['scope']}"
                )
            scoped.sort(
                key=lambda row: (
                    mode_use_counts["source_pair_surface"][row["source_pair_id"]],
                    use_counts[row["source_pair_id"]],
                    -int(row["grounding_rank"]),
                    -int(row["source_is_singleton"]),
                    stable_hash(seed, "source-pair-fallback", task_index, row["source_pair_id"]),
                    row["source_pair_id"],
                )
            )
            evidence = dict(scoped[0])
            skeleton_choice = (evidence["source_pair_id"], pair_scope_match(
                human_pairs[evidence["source_pair_id"]], task["product_type"]
            ) or "")
        pair_id, match_mode = skeleton_choice
        reuse_index = use_counts[pair_id]
        use_counts[pair_id] += 1
        mode_use_counts[construction_mode][pair_id] += 1
        plan[task_index] = {
            "skeleton_pair_id": pair_id,
            "skeleton_match_mode": match_mode,
            "skeleton_reuse_index": reuse_index,
            "construction_mode": construction_mode,
            "evidence": evidence,
        }
    for row in plan.values():
        row["skeleton_reuse_count_final"] = use_counts[row["skeleton_pair_id"]]
    return plan


def sanitize_skeleton_attributes(
    attributes: Mapping[str, str], target_aliases: set[str], overlay_key: str
) -> tuple[dict[str, str], list[str], list[str]]:
    retained: dict[str, str] = {}
    removed_target: list[str] = []
    removed_identifiers: list[str] = []
    overlay_norm = normalized_key(overlay_key)
    for key, value in attributes.items():
        key_norm = normalized_key(key)
        if key_norm in target_aliases or key_norm == overlay_norm:
            removed_target.append(key)
        else:
            retained[key] = value
    return retained, removed_target, removed_identifiers


def _normalize_title_spacing(name: str) -> str:
    # Identifiers are part of the human card distribution and often identify
    # the concrete product.  Never delete them; this helper only normalizes
    # accidental whitespace before punctuation-only variation.
    return " ".join(str(name).split()).strip()


def safe_title_variants(name: str, *, seed: int, discriminator: Any) -> list[tuple[str, str]]:
    """Enumerate seller punctuation variants without changing alphanumeric facts."""

    base = _normalize_title_spacing(name)
    variants: list[tuple[str, str]] = [(base, "unchanged")]
    matches = list(re.finditer(r"[0-9A-Za-zА-Яа-яЁё]+", base))
    boundaries: list[int] = []
    for left, right in zip(matches, matches[1:]):
        between = base[left.end() : right.start()]
        if any(char.isspace() for char in between):
            boundaries.append(left.end())
    boundaries.sort(key=lambda pos: stable_hash(seed, "boundary", discriminator, pos))
    delimiters = [", ", " — ", " / ", "; ", " | "]
    delimiters.sort(key=lambda text: stable_hash(seed, "delimiter", discriminator, text))
    for position in boundaries:
        left, right = base[:position].rstrip(" ,;|/—–-"), base[position:].lstrip(" ,;|/—–-")
        for delimiter in delimiters:
            candidate = " ".join((left + delimiter + right).split())
            if token_multiset(candidate) == token_multiset(base):
                variants.append((candidate, "change_title_separator"))
    trimmed = base.rstrip(" .!?;:")
    suffixes = [".", "!", "?", ";", ":", "...", "?!", "!?", "!!", "??"]
    suffixes.extend("".join(chars) for width in (2, 3) for chars in product(".!?", repeat=width))
    for suffix in suffixes:
        candidate = trimmed + suffix
        if token_multiset(candidate) == token_multiset(base):
            variants.append((candidate, "change_terminal_punctuation"))
    deduplicated: list[tuple[str, str]] = []
    seen: set[str] = set()
    for candidate, operation in variants:
        normalized = normalize_text(candidate)
        if candidate and normalized not in seen:
            seen.add(normalized)
            deduplicated.append((candidate, operation))
    return deduplicated


def _card_key(name: str, attributes: Mapping[str, str]) -> str:
    row = pd.Series({"name": name, "attributes": json_attributes(attributes)})
    return canonical_card(row)


def choose_unique_title(
    name: str,
    attributes: Mapping[str, str],
    *,
    seed: int,
    discriminator: Any,
    used_card_keys: set[str],
    forbidden_card_key: str | None,
) -> tuple[str, str, str]:
    variants = safe_title_variants(name, seed=seed, discriminator=discriminator)
    for candidate, operation in variants:
        key = _card_key(candidate, attributes)
        if key not in used_card_keys and key != forbidden_card_key:
            used_card_keys.add(key)
            return candidate, operation, key
    raise SkeletonBuildError(f"could not create a unique fact-preserving card for {discriminator}")


def _rule_task_counts(tasks: Sequence[Mapping[str, Any]]) -> Counter[str]:
    return Counter(str(task["rule"]["generation_rule_id"]) for task in tasks)


def materialize(
    tasks: Sequence[Mapping[str, Any]],
    human_pairs: Mapping[str, HumanPair],
    task_plan: Mapping[int, Mapping[str, Any]],
    evidence_by_rule: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
    id_start: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    item_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    used_card_keys: set[str] = set()
    output_rule_seen: Counter[str] = Counter()
    rule_counts = _rule_task_counts(tasks)
    for output_index, task in enumerate(tasks):
        task_index = int(task["composition_index"])
        planned = task_plan[task_index]
        skeleton = human_pairs[str(planned["skeleton_pair_id"])]
        rule = task["rule"]
        rule_id = str(rule["generation_rule_id"])
        evidence_bank = evidence_by_rule[rule_id]
        evidence = planned["evidence"]
        rule_task_index = output_rule_seen[rule_id]
        output_rule_seen[rule_id] += 1
        aliases = selected_target_aliases(evidence_bank, evidence)
        if planned["construction_mode"] == "overlay":
            base_a, removed_target_a, removed_ids_a = sanitize_skeleton_attributes(
                skeleton.attributes_a, aliases, evidence["target_key_a"]
            )
            base_b, removed_target_b, removed_ids_b = sanitize_skeleton_attributes(
                skeleton.attributes_b, aliases, evidence["target_key_b"]
            )
            attrs_a, attrs_b = dict(base_a), dict(base_b)
            attrs_a[evidence["target_key_a"]] = evidence["target_value_a"]
            attrs_b[evidence["target_key_b"]] = evidence["target_value_b"]
        elif planned["construction_mode"] == "source_pair_surface":
            attrs_a, attrs_b = dict(skeleton.attributes_a), dict(skeleton.attributes_b)
            removed_target_a = removed_target_b = []
            removed_ids_a = removed_ids_b = []
        else:
            raise AssertionError(f"unknown construction mode: {planned['construction_mode']}")
        if not attrs_a or not attrs_b:
            raise SkeletonBuildError(f"empty output attributes at task {task_index}")

        original_key_a = _card_key(skeleton.title_a, skeleton.attributes_a)
        original_key_b = _card_key(skeleton.title_b, skeleton.attributes_b)
        name_a, name_op_a, card_key_a = choose_unique_title(
            skeleton.title_a,
            attrs_a,
            seed=seed,
            discriminator=(task_index, "a"),
            used_card_keys=used_card_keys,
            forbidden_card_key=original_key_a,
        )
        name_b, name_op_b, card_key_b = choose_unique_title(
            skeleton.title_b,
            attrs_b,
            seed=seed,
            discriminator=(task_index, "b"),
            used_card_keys=used_card_keys,
            forbidden_card_key=original_key_b,
        )
        if card_key_a == card_key_b:
            raise SkeletonBuildError(f"identical output cards at task {task_index}")

        id1, id2 = int(id_start) - output_index * 2, int(id_start) - output_index * 2 - 1
        item_rows.extend(
            [
                {"id": id1, "name": name_a, "attributes": json_attributes(attrs_a), "category": task["category"]},
                {"id": id2, "name": name_b, "attributes": json_attributes(attrs_b), "category": task["category"]},
            ]
        )
        pair_rows.append({"id1": id1, "id2": id2, "target": 1})
        source_meta = task["source_metadata"]
        metadata_rows.append(
            {
                "task_index": task_index,
                "composition_index": task_index,
                "id1": id1,
                "id2": id2,
                "target": 1,
                "category": task["category"],
                "product_type": task["product_type"],
                "component": str(source_meta.get("component") or ""),
                "generation_rule_id": rule_id,
                "source_rule_id": str(rule.get("source_rule_id") or ""),
                "generation_tier": str(rule.get("generation_tier") or ""),
                "concept": str(rule.get("concept") or ""),
                "required_attribute_key": str(rule.get("required_attribute_key") or ""),
                "source_generated_applications_json": str(source_meta.get("applications_json") or "[]"),
                "source_generated_rules_json": str(source_meta.get("rules_json") or "[]"),
                "skeleton_pair_id": skeleton.pair_id,
                "skeleton_item_id_a": skeleton.item_id_a,
                "skeleton_item_id_b": skeleton.item_id_b,
                "skeleton_match_mode": planned["skeleton_match_mode"],
                "skeleton_reuse_index": int(planned["skeleton_reuse_index"]),
                "skeleton_reuse_count_final": int(planned["skeleton_reuse_count_final"]),
                "construction_mode": planned["construction_mode"],
                "evidence_pair_id": evidence["source_pair_id"],
                "evidence_source_example_index": int(evidence["source_example_index"]),
                "evidence_grounding_mode": evidence["grounding_mode"],
                "evidence_source_is_singleton": bool(evidence["source_is_singleton"]),
                "evidence_human_label": 1,
                "semantic_keys_compatible_for_overlay": bool(
                    evidence["semantic_keys_compatible_for_overlay"]
                ),
                "schema_safe_for_overlay": bool(evidence["schema_safe_for_overlay"]),
                "overlay_allowlist_version": OVERLAY_ALLOWLIST_VERSION,
                "overlay_allowlist_family": (
                    _safe_overlay_family(
                        _semantic_subfacet_signature(rule.get("required_attribute_key") or "")
                    )
                    if evidence["semantic_keys_compatible_for_overlay"]
                    else ""
                ) or "",
                "target_key_a": evidence["target_key_a"],
                "target_key_b": evidence["target_key_b"],
                "target_value_a": evidence["target_value_a"],
                "target_value_b": evidence["target_value_b"],
                "source_example_value_a": evidence["source_example_value_a"],
                "source_example_value_b": evidence["source_example_value_b"],
                "raw_attribute_key_a": evidence["raw_attribute_key_a"],
                "raw_attribute_key_b": evidence["raw_attribute_key_b"],
                "target_aliases_json": json.dumps(sorted(aliases), ensure_ascii=False),
                "removed_target_keys_a_json": json.dumps(removed_target_a, ensure_ascii=False),
                "removed_target_keys_b_json": json.dumps(removed_target_b, ensure_ascii=False),
                "removed_identifier_keys_a_json": json.dumps(removed_ids_a, ensure_ascii=False),
                "removed_identifier_keys_b_json": json.dumps(removed_ids_b, ensure_ascii=False),
                "title_operation_a": name_op_a,
                "title_operation_b": name_op_b,
                "non_target_values_preserved": True,
                "observed_label1_transition": True,
                "rule_task_index": int(rule_task_index),
                "rule_task_count": int(rule_counts[rule_id]),
                "builder_version": VERSION,
            }
        )
    return pd.DataFrame(item_rows), pd.DataFrame(pair_rows), pd.DataFrame(metadata_rows)


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


def _pair_metrics(rows: Sequence[tuple[str, Mapping[str, str], str, Mapping[str, str]]]) -> dict[str, Any]:
    values: dict[str, list[float]] = defaultdict(list)
    zero_shared = 0
    for title_a, attrs_a, title_b, attrs_b in rows:
        by_a = {normalized_key(key): normalize_text(value) for key, value in attrs_a.items()}
        by_b = {normalized_key(key): normalize_text(value) for key, value in attrs_b.items()}
        common = set(by_a) & set(by_b)
        same = sum(by_a[key] == by_b[key] for key in common)
        zero_shared += not common
        values["attribute_count_a"].append(len(attrs_a))
        values["attribute_count_b"].append(len(attrs_b))
        values["title_tokens_a"].append(len(fact_tokens(title_a)))
        values["title_tokens_b"].append(len(fact_tokens(title_b)))
        values["shared_keys"].append(len(common))
        values["same_value_shared_keys"].append(same)
        values["conflicting_shared_keys"].append(len(common) - same)
        values["key_jaccard"].append(len(common) / max(1, len(set(by_a) | set(by_b))))
    return {
        "pairs": len(rows),
        "metrics": {key: _numeric_summary(series) for key, series in sorted(values.items())},
        "zero_shared_key_fraction": zero_shared / len(rows) if rows else 0.0,
    }


def _fact_card_tokens(
    name: str, attributes: Mapping[str, str], category: str
) -> tuple[str, ...]:
    """Competition-serialization fact identity with punctuation removed."""

    serialized = serialize_product(
        pd.Series(
            {
                "name": str(name),
                "attributes": json_attributes(attributes),
                "category": str(category),
            }
        ),
        max_attribute_chars=6000,
    )
    return fact_tokens(serialized)


def _fact_card_key(name: str, attributes: Mapping[str, str], category: str) -> str:
    return canonical_json_dumps(list(_fact_card_tokens(name, attributes, category)))


def load_frozen_validation_facts(validation_dir: Path) -> FrozenValidationFacts:
    """Load punctuation-insensitive card facts referenced by frozen validation."""

    validation_dir = absolute(validation_dir)
    items_path = validation_dir / "items.parquet"
    pair_paths = {
        split: validation_dir / f"{split}_validation_pairs.parquet"
        for split in VALIDATION_SPLITS
    }
    missing = [path for path in (items_path, *pair_paths.values()) if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))

    split_ids: dict[str, set[int]] = {}
    split_pair_counts: dict[str, int] = {}
    all_ids: set[int] = set()
    for split, path in pair_paths.items():
        frame = pd.read_parquet(path)
        if missing_columns := {"id1", "id2"} - set(frame.columns):
            raise SkeletonBuildError(
                f"{path.name} missing columns: {sorted(missing_columns)}"
            )
        ids = set(pd.to_numeric(frame["id1"], errors="raise").astype("int64"))
        ids.update(pd.to_numeric(frame["id2"], errors="raise").astype("int64"))
        split_ids[split] = ids
        split_pair_counts[split] = len(frame)
        all_ids.update(ids)

    items = pd.read_parquet(items_path)
    if missing_columns := {"id", "product_text"} - set(items.columns):
        raise SkeletonBuildError(
            f"validation items missing columns: {sorted(missing_columns)}"
        )
    item_ids = pd.to_numeric(items["id"], errors="raise").astype("int64")
    if item_ids.duplicated().any():
        raise SkeletonBuildError("validation items contain duplicate id")
    items = items.assign(id=item_ids)
    available_ids = set(int(value) for value in items["id"])
    missing_ids = all_ids - available_ids
    if missing_ids:
        raise SkeletonBuildError(
            f"{len(missing_ids)} frozen validation item IDs lack product_text"
        )

    split_membership: dict[int, set[str]] = defaultdict(set)
    for split, ids in split_ids.items():
        for item_id in ids:
            split_membership[item_id].add(split)
    fact_sources_mutable: dict[tuple[str, ...], dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    selected = items.loc[items["id"].isin(all_ids), ["id", "product_text"]]
    for row in selected.itertuples(index=False):
        tokens = fact_tokens(row.product_text)
        if not tokens:
            raise SkeletonBuildError(f"validation item {row.id} has empty product facts")
        for split in split_membership[int(row.id)]:
            fact_sources_mutable[tokens][split].add(int(row.id))

    fact_sources = {
        tokens: {
            split: tuple(sorted(ids)) for split, ids in sorted(by_split.items())
        }
        for tokens, by_split in fact_sources_mutable.items()
    }
    return FrozenValidationFacts(
        fact_sources=fact_sources,
        split_pair_counts=split_pair_counts,
        split_item_counts={split: len(split_ids[split]) for split in VALIDATION_SPLITS},
        unique_item_ids=len(all_ids),
        unique_fact_keys=len(fact_sources),
    )


def filter_frozen_validation_fact_overlaps(
    items: pd.DataFrame,
    pairs: pd.DataFrame,
    metadata: pd.DataFrame,
    validation_facts: FrozenValidationFacts,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Drop a whole pair when either card matches any frozen validation card."""

    if len(pairs) != len(metadata):
        raise SkeletonBuildError("cannot leakage-filter misaligned pairs and metadata")
    pair_keys = list(zip(pairs["id1"].astype(int), pairs["id2"].astype(int)))
    metadata_keys = list(zip(metadata["id1"].astype(int), metadata["id2"].astype(int)))
    if pair_keys != metadata_keys:
        raise SkeletonBuildError("pairs and metadata order differs before leakage filter")
    item_map = {int(row.id): row._asdict() for row in items.itertuples(index=False)}
    keep: list[bool] = []
    dropped_records: list[dict[str, Any]] = []
    by_split: Counter[str] = Counter()
    by_side: Counter[str] = Counter()
    by_mode: Counter[str] = Counter()

    for pair_row, meta in zip(pairs.itertuples(index=False), metadata.to_dict("records")):
        endpoint_hits: list[dict[str, Any]] = []
        pair_splits: set[str] = set()
        for side, item_id in (("a", int(pair_row.id1)), ("b", int(pair_row.id2))):
            item = item_map[item_id]
            tokens = _fact_card_tokens(
                str(item["name"]),
                parse_attributes(item["attributes"]),
                str(item["category"]),
            )
            sources = validation_facts.fact_sources.get(tokens)
            if not sources:
                continue
            pair_splits.update(sources)
            by_side[side] += 1
            endpoint_hits.append(
                {
                    "side": side,
                    "generated_item_id": item_id,
                    "fact_key_sha256": sha256_json(list(tokens)),
                    "fact_tokens": list(tokens),
                    "validation_matches_by_split": {
                        split: list(ids) for split, ids in sorted(sources.items())
                    },
                }
            )
        if endpoint_hits:
            keep.append(False)
            for split in pair_splits:
                by_split[split] += 1
            by_mode[str(meta["construction_mode"])] += 1
            dropped_records.append(
                {
                    "composition_index": int(meta["composition_index"]),
                    "source_task_index": int(meta["task_index"]),
                    "generated_id1": int(pair_row.id1),
                    "generated_id2": int(pair_row.id2),
                    "construction_mode": str(meta["construction_mode"]),
                    "overlap_endpoints": endpoint_hits,
                }
            )
        else:
            keep.append(True)

    kept_pairs = pairs.loc[keep].reset_index(drop=True)
    kept_metadata = metadata.loc[keep].reset_index(drop=True)
    kept_ids = set(kept_pairs["id1"].astype(int)) | set(kept_pairs["id2"].astype(int))
    kept_items = items.loc[items["id"].astype(int).isin(kept_ids)].reset_index(drop=True)
    postfilter_overlap = 0
    for row in kept_items.itertuples(index=False):
        tokens = _fact_card_tokens(
            str(row.name), parse_attributes(row.attributes), str(row.category)
        )
        postfilter_overlap += tokens in validation_facts.fact_sources
    if postfilter_overlap:
        raise SkeletonBuildError(
            f"validation fact leakage remains after filter: {postfilter_overlap} cards"
        )

    report = {
        "version": "frozen_validation_serialized_fact_exclusion_v1",
        "fact_key_version": "ordered_fact_tokens_serialize_product_6000_v1",
        "source_task_count": int(len(pairs)),
        "emitted_pair_count": int(len(kept_pairs)),
        "dropped_pair_count": int(len(dropped_records)),
        "dropped_task_ids": [row["composition_index"] for row in dropped_records],
        "dropped_pairs_by_split": {
            split: int(by_split[split]) for split in VALIDATION_SPLITS
        },
        "dropped_endpoints_by_side": {
            "a": int(by_side["a"]),
            "b": int(by_side["b"]),
        },
        "dropped_pairs_by_construction_mode": dict(sorted(by_mode.items())),
        "unique_dropped_card_facts": len(
            {
                endpoint["fact_key_sha256"]
                for row in dropped_records
                for endpoint in row["overlap_endpoints"]
            }
        ),
        "postfilter_overlapping_card_count": int(postfilter_overlap),
        "validation_reference": {
            "split_pair_counts": dict(validation_facts.split_pair_counts),
            "split_item_counts": dict(validation_facts.split_item_counts),
            "unique_item_ids": int(validation_facts.unique_item_ids),
            "unique_fact_keys": int(validation_facts.unique_fact_keys),
        },
        "dropped_pairs": dropped_records,
    }
    return kept_items, kept_pairs, kept_metadata, report


def _fact_pair_key(card_a: str, card_b: str) -> str:
    return canonical_json_dumps(sorted((card_a, card_b)))


def _fact_clone_diagnostics(
    items: pd.DataFrame,
    pairs: pd.DataFrame,
    metadata: pd.DataFrame,
    human_pairs: Mapping[str, HumanPair],
) -> dict[str, Any]:
    """Nonblocking diagnostics for punctuation-only factual clones."""

    human_card_keys: set[str] = set()
    human_pair_keys: set[str] = set()
    for pair in human_pairs.values():
        card_a = _fact_card_key(pair.title_a, pair.attributes_a, pair.category)
        card_b = _fact_card_key(pair.title_b, pair.attributes_b, pair.category)
        human_card_keys.update((card_a, card_b))
        human_pair_keys.add(_fact_pair_key(card_a, card_b))

    item_map = {int(row.id): row._asdict() for row in items.itertuples(index=False)}
    mode_by_pair = {
        (int(row.id1), int(row.id2)): str(row.construction_mode)
        for row in metadata.itertuples(index=False)
    }
    records: list[tuple[str, str, str]] = []
    for pair in pairs.itertuples(index=False):
        id1, id2 = int(pair.id1), int(pair.id2)
        left, right = item_map[id1], item_map[id2]
        card_a = _fact_card_key(
            str(left["name"]), parse_attributes(left["attributes"]), str(left["category"])
        )
        card_b = _fact_card_key(
            str(right["name"]), parse_attributes(right["attributes"]), str(right["category"])
        )
        records.append((mode_by_pair[(id1, id2)], card_a, card_b))

    def summarize(selected: Sequence[tuple[str, str, str]]) -> dict[str, Any]:
        card_keys = [card for _, card_a, card_b in selected for card in (card_a, card_b)]
        pair_keys = [_fact_pair_key(card_a, card_b) for _, card_a, card_b in selected]
        return {
            "cards": {
                "total": len(card_keys),
                "unique": len(set(card_keys)),
                "excess_clones": len(card_keys) - len(set(card_keys)),
                "fact_identical_to_human_positive": sum(
                    key in human_card_keys for key in card_keys
                ),
            },
            "pairs": {
                "total": len(pair_keys),
                "unique": len(set(pair_keys)),
                "excess_clones": len(pair_keys) - len(set(pair_keys)),
                "fact_identical_to_human_positive": sum(
                    key in human_pair_keys for key in pair_keys
                ),
            },
        }

    modes = sorted({mode for mode, _, _ in records})
    return {
        "version": "punctuation_insensitive_fact_clone_diagnostics_v1",
        "nonblocking": True,
        "human_positive_reference": {
            "unique_cards": len(human_card_keys),
            "unique_pairs": len(human_pair_keys),
        },
        "all": summarize(records),
        "by_construction_mode": {
            mode: summarize([row for row in records if row[0] == mode])
            for mode in modes
        },
    }


def build_distribution_report(
    items: pd.DataFrame,
    pairs: pd.DataFrame,
    metadata: pd.DataFrame,
    human_pairs: Mapping[str, HumanPair],
    *,
    validation_overlap: Mapping[str, Any],
) -> dict[str, Any]:
    item_map = {int(row.id): row._asdict() for row in items.itertuples(index=False)}
    generated_rows = []
    for pair in pairs.itertuples(index=False):
        left, right = item_map[int(pair.id1)], item_map[int(pair.id2)]
        generated_rows.append(
            (str(left["name"]), parse_attributes(left["attributes"]), str(right["name"]), parse_attributes(right["attributes"]))
        )
    selected_rows = []
    for pair_id in metadata["skeleton_pair_id"].astype(str):
        pair = human_pairs[pair_id]
        selected_rows.append((pair.title_a, pair.attributes_a, pair.title_b, pair.attributes_b))
    all_human_rows = [
        (pair.title_a, pair.attributes_a, pair.title_b, pair.attributes_b)
        for pair in human_pairs.values()
        if pair.category not in FORBIDDEN_CATEGORIES
    ]
    generated = _pair_metrics(generated_rows)
    selected = _pair_metrics(selected_rows)
    human = _pair_metrics(all_human_rows)
    median_deltas = {
        metric: generated["metrics"][metric]["median"] - selected["metrics"][metric]["median"]
        for metric in generated["metrics"]
    }
    overlay_rows = metadata.loc[metadata["construction_mode"].eq("overlay")]
    fallback_rows = metadata.loc[metadata["construction_mode"].eq("source_pair_surface")]

    def mode_max_reuse(frame: pd.DataFrame) -> int:
        counts = frame["skeleton_pair_id"].astype(str).value_counts()
        return int(counts.max()) if len(counts) else 0

    overlay_max_reuse = mode_max_reuse(overlay_rows)
    fallback_max_reuse = mode_max_reuse(fallback_rows)
    unique_skeleton_fraction = metadata["skeleton_pair_id"].nunique() / len(metadata)
    gates = {
        "title_token_median_delta_abs_lte_0": all(
            abs(median_deltas[key]) <= 1e-12 for key in ("title_tokens_a", "title_tokens_b")
        ),
        "attribute_count_median_delta_abs_lte_2": all(
            abs(median_deltas[key]) <= 2 for key in ("attribute_count_a", "attribute_count_b")
        ),
        "shared_keys_median_delta_abs_lte_2": abs(median_deltas["shared_keys"]) <= 2,
        "zero_shared_fraction_delta_abs_lte_0_10": abs(
            generated["zero_shared_key_fraction"] - selected["zero_shared_key_fraction"]
        ) <= 0.10,
        "unique_skeleton_fraction_gte_0_26": (
            unique_skeleton_fraction >= MIN_UNIQUE_SKELETON_FRACTION
        ),
        "overlay_max_skeleton_reuse_lte_15": (
            overlay_max_reuse <= MAX_OVERLAY_SKELETON_REUSE
        ),
        "fallback_max_source_pair_reuse_lte_35": (
            fallback_max_reuse <= MAX_FALLBACK_SOURCE_PAIR_REUSE
        ),
    }
    return {
        "schema_version": 1,
        "builder_version": VERSION,
        "generated": generated,
        "selected_human_positive_skeletons_with_repetition": selected,
        "all_human_train_positives": human,
        "generated_minus_selected_medians": median_deltas,
        "skeletons": {
            "unique": int(metadata["skeleton_pair_id"].nunique()),
            "total_assignments": int(len(metadata)),
            "unique_fraction": float(unique_skeleton_fraction),
            "max_reuse": int(metadata["skeleton_reuse_count_final"].max()),
            "overlay_max_reuse": overlay_max_reuse,
            "fallback_max_reuse": fallback_max_reuse,
            "construction_modes": {
                str(key): int(value)
                for key, value in metadata["construction_mode"].value_counts().sort_index().items()
            },
        },
        "gate_thresholds": {
            "min_unique_skeleton_fraction": MIN_UNIQUE_SKELETON_FRACTION,
            "max_overlay_skeleton_reuse": MAX_OVERLAY_SKELETON_REUSE,
            "max_fallback_source_pair_reuse": MAX_FALLBACK_SOURCE_PAIR_REUSE,
        },
        "overlay_allowlist": overlay_allowlist_report(metadata),
        "validation_overlap_filter": dict(validation_overlap),
        "fact_clone_diagnostics": _fact_clone_diagnostics(
            items, pairs, metadata, human_pairs
        ),
        "gates": gates,
        "valid": all(gates.values()),
    }


def validate_dataset(
    *,
    tasks: Sequence[Mapping[str, Any]],
    human_pairs: Mapping[str, HumanPair],
    evidence_by_rule: Mapping[str, Mapping[str, Any]],
    items: pd.DataFrame,
    pairs: pd.DataFrame,
    metadata: pd.DataFrame,
    expected_count: int,
    source_task_count: int,
    validation_facts: FrozenValidationFacts,
    validation_overlap: Mapping[str, Any],
) -> dict[str, Any]:
    errors: Counter[str] = Counter()
    if len(pairs) != expected_count:
        errors["pair_count_mismatch"] += 1
    if len(items) != expected_count * 2:
        errors["item_count_mismatch"] += 1
    if len(metadata) != expected_count:
        errors["metadata_count_mismatch"] += 1
    if len(tasks) != source_task_count:
        errors["source_task_count_mismatch"] += 1
    required_items = {"id", "name", "attributes", "category"}
    required_pairs = {"id1", "id2", "target"}
    required_metadata = {
        "task_index", "id1", "id2", "target", "skeleton_pair_id", "evidence_pair_id",
        "generation_rule_id", "target_key_a", "target_key_b", "target_value_a",
        "target_value_b", "target_aliases_json", "removed_target_keys_a_json",
        "removed_target_keys_b_json", "removed_identifier_keys_a_json",
        "removed_identifier_keys_b_json", "evidence_source_example_index",
        "construction_mode",
        "semantic_keys_compatible_for_overlay", "schema_safe_for_overlay",
        "required_attribute_key", "concept", "overlay_allowlist_version",
        "overlay_allowlist_family", "composition_index",
    }
    for missing, name in (
        (required_items - set(items.columns), "items"),
        (required_pairs - set(pairs.columns), "pairs"),
        (required_metadata - set(metadata.columns), "metadata"),
    ):
        if missing:
            errors[f"{name}_missing_columns"] += len(missing)
    if errors:
        return {"version": VALIDATION_VERSION, "valid": False, "errors": dict(sorted(errors.items()))}

    item_map = {int(row.id): row._asdict() for row in items.itertuples(index=False)}
    meta_map = {(int(row.id1), int(row.id2)): row._asdict() for row in metadata.itertuples(index=False)}
    task_map = {int(task["composition_index"]): task for task in tasks}
    endpoints: list[int] = []
    card_keys: list[str] = []
    for raw in items.to_dict("records"):
        try:
            attrs = parse_attributes(raw["attributes"])
        except Exception:
            errors["invalid_item_attributes"] += 1
            continue
        if not str(raw["name"]).strip() or not attrs:
            errors["empty_card"] += 1
        tokens = _fact_card_tokens(
            str(raw["name"]), attrs, str(raw["category"])
        )
        if tokens in validation_facts.fact_sources:
            errors["frozen_validation_fact_overlap"] += 1
        card_keys.append(canonical_card(pd.Series(raw)))
    if items["id"].duplicated().any():
        errors["duplicate_item_id"] += int(items["id"].duplicated().sum())
    if len(card_keys) != len(set(card_keys)):
        errors["duplicate_global_card"] += len(card_keys) - len(set(card_keys))

    for pair_row in pairs.itertuples(index=False):
        id1, id2 = int(pair_row.id1), int(pair_row.id2)
        endpoints.extend((id1, id2))
        if int(pair_row.target) != 1:
            errors["non_positive_target"] += 1
        if id1 == id2:
            errors["self_pair"] += 1
        left, right = item_map.get(id1), item_map.get(id2)
        meta = meta_map.get((id1, id2))
        if left is None or right is None or meta is None:
            errors["missing_pair_mapping"] += 1
            continue
        task = task_map.get(int(meta["composition_index"]))
        skeleton = human_pairs.get(str(meta["skeleton_pair_id"]))
        evidence_pair = human_pairs.get(str(meta["evidence_pair_id"]))
        if task is None or skeleton is None or evidence_pair is None:
            errors["missing_provenance_source"] += 1
            continue
        if str(left["category"]) != task["category"] or str(right["category"]) != task["category"]:
            errors["category_mismatch"] += 1
        if task["category"] in FORBIDDEN_CATEGORIES:
            errors["ood_leakage"] += 1
        if pair_scope_match(skeleton, task["product_type"]) is None:
            errors["skeleton_scope_not_grounded"] += 1
        if evidence_pair.category != task["category"]:
            errors["evidence_category_mismatch"] += 1
        if pair_scope_match(evidence_pair, task["product_type"]) is None:
            errors["evidence_product_type_not_grounded"] += 1
        rule_id = str(task["rule"]["generation_rule_id"])
        if str(meta["generation_rule_id"]) != rule_id:
            errors["rule_mismatch"] += 1
            continue
        candidates = evidence_by_rule[rule_id]["candidates"]
        evidence_matches = [
            row for row in candidates
            if row["source_pair_id"] == str(meta["evidence_pair_id"])
            and int(row["source_example_index"]) == int(meta["evidence_source_example_index"])
            and row["target_key_a"] == str(meta["target_key_a"])
            and row["target_key_b"] == str(meta["target_key_b"])
            and row["target_value_a"] == str(meta["target_value_a"])
            and row["target_value_b"] == str(meta["target_value_b"])
        ]
        if not evidence_matches:
            errors["transition_not_replayable"] += 1

        attrs_a, attrs_b = parse_attributes(left["attributes"]), parse_attributes(right["attributes"])
        aliases = set(json.loads(str(meta["target_aliases_json"])))
        if evidence_matches:
            expected_aliases = selected_target_aliases(
                evidence_by_rule[rule_id], evidence_matches[0]
            )
            if aliases != expected_aliases:
                errors["candidate_alias_provenance_mismatch"] += 1
        construction_mode = str(meta["construction_mode"])
        semantic_replay = semantic_key_compatible_for_overlay(
            str(meta["required_attribute_key"]),
            str(meta["target_key_a"]),
            str(meta["target_key_b"]),
            str(meta["concept"]),
        )
        expected_allowlist_family = (
            _safe_overlay_family(
                _semantic_subfacet_signature(str(meta["required_attribute_key"]))
            )
            if semantic_replay
            else ""
        ) or ""
        if str(meta["overlay_allowlist_version"]) != OVERLAY_ALLOWLIST_VERSION:
            errors["overlay_allowlist_version_mismatch"] += 1
        if str(meta["overlay_allowlist_family"]) != expected_allowlist_family:
            errors["overlay_allowlist_family_mismatch"] += 1
        if construction_mode == "overlay" and not _selected_target_key_present_on_both_sides(
            skeleton, aliases
        ):
            errors["overlay_target_subfacet_absent"] += 1
        if construction_mode == "overlay" and not expected_allowlist_family:
            errors["overlay_not_allowlisted"] += 1
        for side, output_attrs, source_attrs in (
            ("a", attrs_a, skeleton.attributes_a),
            ("b", attrs_b, skeleton.attributes_b),
        ):
            key = str(meta[f"target_key_{side}"])
            value = str(meta[f"target_value_{side}"])
            removed_target = set(json.loads(str(meta[f"removed_target_keys_{side}_json"])))
            removed_ids = set(json.loads(str(meta[f"removed_identifier_keys_{side}_json"])))
            if construction_mode == "overlay":
                if not bool(meta["semantic_keys_compatible_for_overlay"]) or not bool(
                    meta["schema_safe_for_overlay"]
                ):
                    errors["overlay_without_semantic_key_proof"] += 1
                if not semantic_replay:
                    errors["overlay_semantic_key_replay_failed"] += 1
                if evidence_matches and _target_title_conflict(
                    skeleton,
                    evidence_by_rule[rule_id],
                    evidence_matches[0],
                    aliases,
                ):
                    errors["overlay_skeleton_target_conflict"] += 1
                if output_attrs.get(key) != value:
                    errors["target_overlay_mismatch"] += 1
                expected_removed_target = {
                    source_key for source_key in source_attrs
                    if normalized_key(source_key) in aliases or normalized_key(source_key) == normalized_key(key)
                }
                if removed_target != expected_removed_target or removed_ids:
                    errors["removed_key_provenance_mismatch"] += 1
                expected = {
                    source_key: source_value for source_key, source_value in source_attrs.items()
                    if source_key not in expected_removed_target
                }
                expected[key] = value
                if output_attrs != expected:
                    errors["non_target_not_preserved"] += 1
            elif construction_mode == "source_pair_surface":
                if removed_target or removed_ids:
                    errors["fallback_claims_removed_keys"] += 1
                if output_attrs != source_attrs:
                    errors["fallback_source_facts_changed"] += 1
            else:
                errors["unknown_construction_mode"] += 1
        if token_multiset(left["name"]) != token_multiset(skeleton.title_a):
            errors["title_facts_changed_a"] += 1
        if token_multiset(right["name"]) != token_multiset(skeleton.title_b):
            errors["title_facts_changed_b"] += 1
        if canonical_card(pd.Series(left)) == _card_key(skeleton.title_a, skeleton.attributes_a):
            errors["exact_source_card_copy_a"] += 1
        if canonical_card(pd.Series(right)) == _card_key(skeleton.title_b, skeleton.attributes_b):
            errors["exact_source_card_copy_b"] += 1
    if len(endpoints) != len(set(endpoints)):
        errors["pair_endpoint_reuse"] += len(endpoints) - len(set(endpoints))
    if set(endpoints) != set(int(value) for value in items["id"]):
        errors["endpoint_catalogue_mismatch"] += 1
    if metadata["task_index"].duplicated().any():
        errors["duplicate_task_index"] += int(metadata["task_index"].duplicated().sum())
    if metadata["composition_index"].duplicated().any():
        errors["duplicate_composition_index"] += int(
            metadata["composition_index"].duplicated().sum()
        )
    emitted_task_ids = set(metadata["composition_index"].astype(int))
    all_task_ids = {int(task["composition_index"]) for task in tasks}
    reported_dropped = [int(value) for value in validation_overlap.get("dropped_task_ids", [])]
    if len(reported_dropped) != len(set(reported_dropped)):
        errors["duplicate_reported_dropped_task"] += 1
    if set(reported_dropped) != all_task_ids - emitted_task_ids:
        errors["dropped_task_set_mismatch"] += 1
    if int(validation_overlap.get("source_task_count", -1)) != source_task_count:
        errors["overlap_report_source_count_mismatch"] += 1
    if int(validation_overlap.get("emitted_pair_count", -1)) != len(pairs):
        errors["overlap_report_emitted_count_mismatch"] += 1
    if int(validation_overlap.get("dropped_pair_count", -1)) != len(reported_dropped):
        errors["overlap_report_dropped_count_mismatch"] += 1
    if source_task_count != len(pairs) + len(reported_dropped):
        errors["source_emitted_dropped_arithmetic_mismatch"] += 1
    if int(validation_overlap.get("postfilter_overlapping_card_count", -1)) != 0:
        errors["overlap_report_postfilter_nonzero"] += 1
    grounding_counts = {
        str(key): int(value)
        for key, value in metadata["evidence_grounding_mode"].value_counts().sort_index().items()
    }
    return {
        "version": VALIDATION_VERSION,
        "valid": not errors,
        "pairs": int(len(pairs)),
        "items": int(len(items)),
        "checked_pairs": int(len(pairs)),
        "source_task_count": int(source_task_count),
        "validation_overlap_filter": dict(validation_overlap),
        "target_counts": {"0": int((pairs["target"] == 0).sum()), "1": int((pairs["target"] == 1).sum())},
        "unique_cards": int(len(set(card_keys))),
        "unique_skeleton_pairs": int(metadata["skeleton_pair_id"].nunique()),
        "max_skeleton_reuse": int(metadata["skeleton_reuse_count_final"].max()),
        "evidence_grounding_counts": grounding_counts,
        "non_target_values_preserved": (
            errors["non_target_not_preserved"] == 0
            and errors["fallback_source_facts_changed"] == 0
        ),
        "errors": dict(sorted(errors.items())),
    }


def source_provenance(
    source_dir: Path,
    pilot_inputs: Path,
    pilot_labels: Path,
    validation_dir: Path,
    script_path: Path,
) -> dict[str, Any]:
    paths = {
        "source_items": source_dir / "items.parquet",
        "source_pairs": source_dir / "pairs.parquet",
        "source_metadata": source_dir / "pair_generation_metadata.parquet",
        "source_summary": source_dir / "summary.json",
        "pilot_inputs": pilot_inputs,
        "pilot_labels": pilot_labels,
        "pilot_manifest": pilot_inputs.parent / "manifest.json",
        "validation_items": validation_dir / "items.parquet",
        **{
            f"validation_{split}_pairs": validation_dir
            / f"{split}_validation_pairs.parquet"
            for split in VALIDATION_SPLITS
        },
        "builder_script": script_path,
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))
    return {
        name: {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def assignment_signature_payload(metadata: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "task_index": int(row["task_index"]),
            "skeleton_pair_id": str(row["skeleton_pair_id"]),
            "evidence_pair_id": str(row["evidence_pair_id"]),
            "evidence_source_example_index": int(row["evidence_source_example_index"]),
            "target_key_a": str(row["target_key_a"]),
            "target_key_b": str(row["target_key_b"]),
            "target_value_a": str(row["target_value_a"]),
            "target_value_b": str(row["target_value_b"]),
        }
        for row in metadata.to_dict("records")
    ]


def calculate_run_signature(
    metadata: pd.DataFrame,
    provenance: Mapping[str, Any],
    config: Mapping[str, Any],
    validation_overlap: Mapping[str, Any],
) -> str:
    return sha256_json(
        {
            "builder_version": VERSION,
            "validation_version": VALIDATION_VERSION,
            "selection_version": SELECTION_VERSION,
            "evidence_version": EVIDENCE_VERSION,
            "overlay_allowlist": overlay_allowlist_definition(),
            "source_provenance": dict(provenance),
            "config": dict(config),
            "validation_overlap": dict(validation_overlap),
            "assignments": assignment_signature_payload(metadata),
        }
    )


def preflight(
    *,
    source_dir: Path,
    pilot_inputs_path: Path,
    pilot_labels_path: Path,
    validation_dir: Path,
    count: int | None,
    seed: int,
) -> dict[str, Any]:
    source_metadata = pd.read_parquet(source_dir / "pair_generation_metadata.parquet")
    tasks = parse_tasks(source_metadata, count)
    inputs = pd.read_parquet(pilot_inputs_path)
    labels = pd.read_parquet(pilot_labels_path)
    validate_pilot_manifest(pilot_inputs_path.parent / "manifest.json", inputs, labels)
    human_pairs, _ = load_human_pairs(inputs, labels)
    pools = build_scope_pools(tasks, human_pairs)
    evidence = build_all_evidence(tasks, human_pairs)
    plan = build_task_plan(tasks, pools, human_pairs, evidence, seed=seed)
    validation_facts = load_frozen_validation_facts(validation_dir)
    grounding = Counter(row["evidence"]["grounding_mode"] for row in plan.values())
    mode_max_reuse = {
        mode: max(
            Counter(
                row["skeleton_pair_id"]
                for row in plan.values()
                if row["construction_mode"] == mode
            ).values()
        )
        for mode in {row["construction_mode"] for row in plan.values()}
    }
    pool_sizes = [len(pools[task["scope"]]) for task in tasks]
    return {
        "version": VERSION,
        "tasks": len(tasks),
        "unique_rules": len(evidence),
        "unique_scopes": len(pools),
        "scope_pool_size": _numeric_summary(pool_sizes),
        "unique_assigned_skeletons": len({row["skeleton_pair_id"] for row in plan.values()}),
        "max_skeleton_reuse": max(row["skeleton_reuse_count_final"] for row in plan.values()),
        "construction_mode_counts": dict(sorted(Counter(row["construction_mode"] for row in plan.values()).items())),
        "construction_mode_max_reuse": dict(sorted(mode_max_reuse.items())),
        "selected_grounding_counts": dict(sorted(grounding.items())),
        "frozen_validation_reference": {
            "split_pair_counts": dict(validation_facts.split_pair_counts),
            "split_item_counts": dict(validation_facts.split_item_counts),
            "unique_item_ids": validation_facts.unique_item_ids,
            "unique_fact_keys": validation_facts.unique_fact_keys,
        },
    }


def build(
    *,
    source_dir: Path = DEFAULT_SOURCE,
    pilot_inputs_path: Path = DEFAULT_PILOT_INPUTS,
    pilot_labels_path: Path = DEFAULT_PILOT_LABELS,
    validation_dir: Path = DEFAULT_VALIDATION_DIR,
    output_dir: Path = DEFAULT_OUTPUT,
    count: int | None = None,
    seed: int = DEFAULT_SEED,
    id_start: int = DEFAULT_ID_START,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_dir, pilot_inputs_path, pilot_labels_path, validation_dir, output_dir = map(
        absolute,
        (source_dir, pilot_inputs_path, pilot_labels_path, validation_dir, output_dir),
    )
    provenance = source_provenance(
        source_dir,
        pilot_inputs_path,
        pilot_labels_path,
        validation_dir,
        Path(__file__),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / filename for filename in OUTPUT_FILENAMES if (output_dir / filename).exists()]
    if existing and not overwrite:
        raise FileExistsError("output artifacts already exist; use --overwrite")

    source_pairs = pd.read_parquet(source_dir / "pairs.parquet")
    source_metadata = pd.read_parquet(source_dir / "pair_generation_metadata.parquet")
    if len(source_pairs) != len(source_metadata):
        raise SkeletonBuildError("source pairs and metadata row counts differ")
    if not pd.to_numeric(source_pairs["target"], errors="coerce").eq(1).all():
        raise SkeletonBuildError("source A+B dataset is not all label=1")
    tasks = parse_tasks(source_metadata, count)
    inputs = pd.read_parquet(pilot_inputs_path)
    labels = pd.read_parquet(pilot_labels_path)
    validate_pilot_manifest(pilot_inputs_path.parent / "manifest.json", inputs, labels)
    human_pairs, _ = load_human_pairs(inputs, labels)
    validation_facts = load_frozen_validation_facts(validation_dir)
    pools = build_scope_pools(tasks, human_pairs)
    evidence = build_all_evidence(tasks, human_pairs)
    task_plan = build_task_plan(tasks, pools, human_pairs, evidence, seed=seed)
    items, pairs, metadata = materialize(
        tasks, human_pairs, task_plan, evidence, seed=seed, id_start=id_start
    )
    items, pairs, metadata, validation_overlap = filter_frozen_validation_fact_overlaps(
        items, pairs, metadata, validation_facts
    )
    if (
        count is None
        and source_dir == absolute(DEFAULT_SOURCE)
        and validation_dir == absolute(DEFAULT_VALIDATION_DIR)
        and validation_overlap["dropped_pair_count"]
        != EXPECTED_FULL_DROPPED_VALIDATION_OVERLAP
    ):
        raise SkeletonBuildError(
            "canonical validation-overlap drop count changed: "
            f"{validation_overlap['dropped_pair_count']} != "
            f"{EXPECTED_FULL_DROPPED_VALIDATION_OVERLAP}"
        )
    validation = validate_dataset(
        tasks=tasks,
        human_pairs=human_pairs,
        evidence_by_rule=evidence,
        items=items,
        pairs=pairs,
        metadata=metadata,
        expected_count=len(pairs),
        source_task_count=len(tasks),
        validation_facts=validation_facts,
        validation_overlap=validation_overlap,
    )
    if not validation["valid"]:
        raise SkeletonBuildError("dataset validation failed: " + json.dumps(validation["errors"], ensure_ascii=False))
    distribution = build_distribution_report(
        items,
        pairs,
        metadata,
        human_pairs,
        validation_overlap=validation_overlap,
    )
    if not distribution["valid"]:
        failed = [key for key, value in distribution["gates"].items() if not value]
        raise SkeletonBuildError("distribution gates failed: " + ", ".join(failed))

    config = {
        "count": len(pairs),
        "source_task_count": len(tasks),
        "dropped_validation_overlap": validation_overlap["dropped_pair_count"],
        "seed": int(seed),
        "id_start": int(id_start),
    }
    run_signature = calculate_run_signature(
        metadata, provenance, config, validation_overlap
    )
    metadata["run_signature"] = run_signature

    atomic_parquet(items, output_dir / "items.parquet")
    atomic_parquet(pairs, output_dir / "pairs.parquet")
    atomic_parquet(metadata, output_dir / "pair_generation_metadata.parquet")
    atomic_json(validation, output_dir / "validation_report.json")
    atomic_json(distribution, output_dir / "distribution_report.json")
    summary = {
        "schema_version": 1,
        "builder_version": VERSION,
        "validation_version": VALIDATION_VERSION,
        "selection_version": SELECTION_VERSION,
        "evidence_version": EVIDENCE_VERSION,
        "overlay_allowlist": distribution["overlay_allowlist"],
        "run_signature": run_signature,
        "label_source": LABEL_SOURCE,
        "generated_pairs": len(pairs),
        "generated_items": len(items),
        "source_task_count": len(tasks),
        "validation_overlap_filter": validation_overlap,
        "target_counts": {"0": 0, "1": len(pairs)},
        "unique_rules": len(evidence),
        "unique_scopes": len(pools),
        "unique_skeleton_pairs": int(metadata["skeleton_pair_id"].nunique()),
        "skeleton_distribution": distribution["skeletons"],
        "evidence_grounding_counts": validation["evidence_grounding_counts"],
        "fact_clone_diagnostics": distribution["fact_clone_diagnostics"],
        "config": config,
        "source_provenance": provenance,
        "validation": validation,
        "distribution_gates": distribution["gates"],
    }
    atomic_json(summary, output_dir / "summary.json")

    artifact_names = [
        "items.parquet", "pairs.parquet", "pair_generation_metadata.parquet",
        "summary.json", "validation_report.json", "distribution_report.json",
    ]
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "builder_version": VERSION,
        "run_signature": run_signature,
        "label_source": LABEL_SOURCE,
        "overlay_allowlist": distribution["overlay_allowlist"],
        "pairs": len(pairs),
        "items": len(items),
        "source_task_count": len(tasks),
        "validation_overlap_filter": validation_overlap,
        "targets": {"0": 0, "1": len(pairs)},
        "source_provenance": provenance,
        "config": config,
        "files": {
            name: {"bytes": (output_dir / name).stat().st_size, "sha256": sha256_file(output_dir / name)}
            for name in artifact_names
        },
    }
    atomic_json(manifest, output_dir / "build_manifest.json")
    return {"summary": summary, "validation": validation, "distribution": distribution, "manifest": manifest, "output_dir": str(output_dir)}


def validate_existing(
    *,
    source_dir: Path,
    pilot_inputs_path: Path,
    pilot_labels_path: Path,
    validation_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_dir, pilot_inputs_path, pilot_labels_path, validation_dir, output_dir = map(
        absolute,
        (source_dir, pilot_inputs_path, pilot_labels_path, validation_dir, output_dir),
    )
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "build_manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = summary.get("config")
    if not isinstance(config, dict):
        raise SkeletonBuildError("summary config is absent or invalid")
    source_task_count = int(config.get("source_task_count", -1))
    validation_overlap = summary.get("validation_overlap_filter")
    if not isinstance(validation_overlap, dict):
        raise SkeletonBuildError("summary validation overlap report is absent")
    if manifest.get("validation_overlap_filter") != validation_overlap:
        raise SkeletonBuildError("manifest validation overlap report differs")
    current_provenance = source_provenance(
        source_dir,
        pilot_inputs_path,
        pilot_labels_path,
        validation_dir,
        Path(__file__),
    )
    if summary.get("source_provenance") != current_provenance:
        raise SkeletonBuildError("source files or builder script changed since build")
    if manifest.get("source_provenance") != current_provenance:
        raise SkeletonBuildError("build manifest source provenance differs")
    source_metadata = pd.read_parquet(source_dir / "pair_generation_metadata.parquet")
    tasks = parse_tasks(source_metadata, source_task_count)
    inputs = pd.read_parquet(pilot_inputs_path)
    labels = pd.read_parquet(pilot_labels_path)
    validate_pilot_manifest(pilot_inputs_path.parent / "manifest.json", inputs, labels)
    human_pairs, _ = load_human_pairs(inputs, labels)
    validation_facts = load_frozen_validation_facts(validation_dir)
    pools = build_scope_pools(tasks, human_pairs)
    evidence = build_all_evidence(tasks, human_pairs)
    task_plan = build_task_plan(
        tasks, pools, human_pairs, evidence, seed=int(config["seed"])
    )
    replay_items, replay_pairs, replay_metadata = materialize(
        tasks,
        human_pairs,
        task_plan,
        evidence,
        seed=int(config["seed"]),
        id_start=int(config["id_start"]),
    )
    replay_items, replay_pairs, replay_metadata, replay_overlap = (
        filter_frozen_validation_fact_overlaps(
            replay_items, replay_pairs, replay_metadata, validation_facts
        )
    )
    if replay_overlap != validation_overlap:
        raise SkeletonBuildError("validation overlap exclusion is not replayable")
    items = pd.read_parquet(output_dir / "items.parquet")
    pairs = pd.read_parquet(output_dir / "pairs.parquet")
    metadata = pd.read_parquet(output_dir / "pair_generation_metadata.parquet")
    try:
        pd.testing.assert_frame_equal(items, replay_items, check_dtype=True)
        pd.testing.assert_frame_equal(pairs, replay_pairs, check_dtype=True)
        pd.testing.assert_frame_equal(
            metadata.drop(columns=["run_signature"]), replay_metadata, check_dtype=True
        )
    except (AssertionError, KeyError) as error:
        raise SkeletonBuildError("stored output differs from deterministic leakage replay") from error
    if int(config.get("count", -1)) != len(pairs):
        raise SkeletonBuildError("config.count differs from emitted pair count")
    expected_signature = calculate_run_signature(
        metadata, current_provenance, config, validation_overlap
    )
    if summary.get("run_signature") != expected_signature or manifest.get("run_signature") != expected_signature:
        raise SkeletonBuildError("run signature is not replayable")
    if "run_signature" not in metadata or not metadata["run_signature"].astype(str).eq(expected_signature).all():
        raise SkeletonBuildError("row-level run signatures differ")
    for filename, record in (manifest.get("files") or {}).items():
        path = output_dir / filename
        if not path.is_file():
            raise SkeletonBuildError(f"manifest output is missing: {filename}")
        if int(record.get("bytes", -1)) != path.stat().st_size or record.get("sha256") != sha256_file(path):
            raise SkeletonBuildError(f"manifest hash mismatch: {filename}")

    report = validate_dataset(
        tasks=tasks,
        human_pairs=human_pairs,
        evidence_by_rule=evidence,
        items=items,
        pairs=pairs,
        metadata=metadata,
        expected_count=len(pairs),
        source_task_count=len(tasks),
        validation_facts=validation_facts,
        validation_overlap=validation_overlap,
    )
    if not report["valid"]:
        raise SkeletonBuildError("existing dataset validation failed: " + json.dumps(report["errors"], ensure_ascii=False))
    distribution = build_distribution_report(
        items,
        pairs,
        metadata,
        human_pairs,
        validation_overlap=validation_overlap,
    )
    if not distribution["valid"]:
        failed = [key for key, value in distribution["gates"].items() if not value]
        raise SkeletonBuildError("existing distribution gates failed: " + ", ".join(failed))
    if summary.get("overlay_allowlist") != distribution["overlay_allowlist"]:
        raise SkeletonBuildError("summary overlay allowlist is not replayable")
    if manifest.get("overlay_allowlist") != distribution["overlay_allowlist"]:
        raise SkeletonBuildError("build manifest overlay allowlist is not replayable")
    stored_distribution = json.loads(
        (output_dir / "distribution_report.json").read_text(encoding="utf-8")
    )
    if stored_distribution != distribution:
        raise SkeletonBuildError("stored distribution report is not replayable")
    stored_validation = json.loads(
        (output_dir / "validation_report.json").read_text(encoding="utf-8")
    )
    if stored_validation != report:
        raise SkeletonBuildError("stored validation report is not replayable")
    return {"validation": report, "distribution": distribution, "run_signature": expected_signature}


def main() -> None:
    args = parse_args()
    source_dir, inputs, labels, validation_dir, output = map(
        absolute,
        (
            args.source_dir,
            args.pilot_inputs,
            args.pilot_labels,
            args.validation_dir,
            args.output_dir,
        ),
    )
    if args.preflight_only:
        result = preflight(
            source_dir=source_dir,
            pilot_inputs_path=inputs,
            pilot_labels_path=labels,
            validation_dir=validation_dir,
            count=args.count,
            seed=args.seed,
        )
    elif args.validate_only:
        result = validate_existing(
            source_dir=source_dir,
            pilot_inputs_path=inputs,
            pilot_labels_path=labels,
            validation_dir=validation_dir,
            output_dir=output,
        )
    else:
        result = build(
            source_dir=source_dir,
            pilot_inputs_path=inputs,
            pilot_labels_path=labels,
            validation_dir=validation_dir,
            output_dir=output,
            count=args.count,
            seed=args.seed,
            id_start=args.id_start,
            overwrite=args.overwrite,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
