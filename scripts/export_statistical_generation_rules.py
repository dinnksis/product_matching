"""Export source-scoped generation rules from atomic-difference statistics.

Besides category-level evidence, each executable rule must have at least one
exact product-type profile supported by multiple source pairs with the same
high-probability label.  The profile and compact source examples keep Qwen in
the part of the category where the atomic concept was actually observed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from item_pipeline.normalization import normalize_text, parse_attributes
from item_pipeline.pair_rules import ANCHOR_HINTS, CONCEPT_ATTRIBUTE_KEYS
from item_pipeline.rule_values import canonical_target_value


DEFAULT_STATISTICS = (
    ROOT / "reports" / "atomic_rule_statistics_current" / "rule_statistics.parquet"
)
DEFAULT_OCCURRENCES = (
    ROOT / "reports" / "atomic_rule_statistics_current" / "atomic_occurrences.parquet"
)
DEFAULT_PAIR_INPUTS = (
    ROOT / "data" / "qwen_atomic_differences_v2_full_train" / "pilot_inputs.parquet"
)
DEFAULT_OUTPUT = (
    ROOT
    / "configs"
    / "generation_rule_catalog_statistical_v1"
    / "statistical_negative_rules_min2_p80_scoped_v3.json"
)
DEFAULT_PROFILE_CAPACITY_POLICY = (
    ROOT
    / "configs"
    / "generation_rule_catalog_statistical_v1"
    / "profile_capacity_policy_v1.json"
)

# These pairs contain a mis-extracted target rather than one atomic change:
# crop kind vs cultivar and a missing wrench size parsed as an unrelated number.
# They must not contribute either support or few-shot examples.
EXCLUDED_SOURCE_PAIR_IDS = {
    "rp_6a0e6c961b827e858cab46f5",
    "rp_eb1e31b61dfa4f86f81697ef",
}

ATTRIBUTE_KEY_OVERRIDES: dict[str, str] = {
    "brand": "Бренд",
    "car_brand": "Марка автомобиля",
    "car_model": "Модель автомобиля",
    "color": "Цвет товара",
    "color_code": "Код/номер цвета",
    "compatibility": "Совместимость",
    "compatible_model": "Совместимая модель",
    "cpu_model": "Модель процессора",
    "country_of_origin": "Страна производства",
    "diopters": "Оптическая сила",
    "flavor": "Вкус",
    "gauge": "Калибр струн",
    "insert": "Вставка",
    "insert_stone": "Вставка",
    "insert_type": "Вид вставки",
    "ingredients": "Состав",
    "item_count": "Количество предметов",
    "length": "Длина",
    "length_mm": "Длина",
    "material": "Материал",
    "memory_capacity": "Объем памяти",
    "model": "Модель",
    "model_compatibility": "Совместимые модели",
    "optical_power": "Оптическая сила",
    "package_quantity": "Количество в упаковке",
    "paper_density": "Плотность бумаги",
    "processor_model": "Модель процессора",
    "product_line": "Линейка",
    "packaging_type": "Тип упаковки",
    "laundry_purpose": "Назначение средства для стирки",
    "shelf_life": "Срок годности",
    "size": "Размер",
    "size_range": "Диапазон размеров",
    "string_count": "Количество струн",
    "scent": "Аромат",
    "target_age": "Возраст животного",
    "target_audience": "Целевая аудитория",
    "tea_variety": "Сорт чая",
    "type": "Вид товара",
    "vehicle_type": "Тип транспорта",
    "volume": "Объем",
    "weight": "Вес изделия",
    "width": "Ширина",
}

CATEGORY_ATTRIBUTE_KEY_OVERRIDES: dict[tuple[str, str], str] = {
    ("Автотовары", "car_model"): "Совместимая модель автомобиля",
    ("Красота и гигиена", "model"): "Название аромата",
}

FORBIDDEN_CONCEPT_RE = re.compile(
    r"(?:^|_)(?:sku|article|part_number|manufacturer_part_number|oem_number)(?:$|_)",
    re.I,
)
FORBIDDEN_ATTRIBUTE_RE = re.compile(
    r"(?:sku|артикул|партномер|oem(?:[- _]?номер)?|код\s+товара)", re.I
)
NUMERIC_ONLY_MODEL_VALUE_RE = re.compile(
    r"^(?:модель\s*)?(?:№|#|no)?\s*\d[\d ._/-]*$", re.IGNORECASE
)
GENERIC_CONCEPTS = {
    "name",
    "name_on_cover",
    "name_on_item",
    "product_name",
    "unknown",
}

PRODUCT_TYPE_KEY_PRIORITY = (
    "тип товара",
    "вид товара",
    "категория товара",
    "тип",
)
MODEL_DEPENDENCY_PATTERN = (
    r"(?<!\w)(?:модель|model|серия|series|линейка|line|коллекци\w*|collection|"
    r"производител\w*|manufacturer)(?!\w)"
)
COMPOSITION_DEPENDENCY_PATTERN = (
    r"(?:состав|ингредиент|сырье|красител|composition|ingredient)"
)
WEIGHT_DEPENDENCY_PATTERN = r"(?:вес|масса|weight|net\s*weight)"
COUNT_DEPENDENCY_PATTERN = (
    r"(?:вес|масса|weight|объем|объём|volume|нетто)"
)
DEPENDENT_ATTRIBUTE_PATTERNS: dict[str, tuple[str, ...]] = {
    "brand": (MODEL_DEPENDENCY_PATTERN,),
    "car_brand": (MODEL_DEPENDENCY_PATTERN,),
    "gauge": (MODEL_DEPENDENCY_PATTERN,),
    "string_count": (MODEL_DEPENDENCY_PATTERN,),
    "product_line": (MODEL_DEPENDENCY_PATTERN,),
    "collection": (MODEL_DEPENDENCY_PATTERN,),
    "item_count": (
        COUNT_DEPENDENCY_PATTERN,
        r"(?:комплектац|состав\s+набора|перечень|набор\s+включает)",
    ),
    "package_quantity": (COUNT_DEPENDENCY_PATTERN,),
    "size": (WEIGHT_DEPENDENCY_PATTERN,),
    "length": (WEIGHT_DEPENDENCY_PATTERN,),
    "length_mm": (WEIGHT_DEPENDENCY_PATTERN,),
    "width": (WEIGHT_DEPENDENCY_PATTERN,),
    "diameter": (WEIGHT_DEPENDENCY_PATTERN,),
    "wheel_diameter": (WEIGHT_DEPENDENCY_PATTERN,),
    "case_diameter": (WEIGHT_DEPENDENCY_PATTERN,),
    "brush_size": (WEIGHT_DEPENDENCY_PATTERN,),
    "hook_size": (WEIGHT_DEPENDENCY_PATTERN,),
    "tea_variety": (
        r"(?:аромат|запах|вкус|состав|ингредиент|ферментац|обработк|"
        r"flavou?r|scent|composition)",
    ),
    "car_model": (
        r"(?:марка|бренд|brand|год|поколени|кузов|модификац|двигател)",
    ),
    "cpu_model": (
        r"(?:частот|ядр|поток|сокет|к[эе]ш|cache|tdp|техпроцесс|архитект)",
    ),
    "processor_model": (
        r"(?:частот|ядр|поток|сокет|к[эе]ш|cache|tdp|техпроцесс|архитект)",
    ),
    "variety": (
        r"(?:тип\s+роста|срок\s+созрев|высота|урожайн|регион|зона|"
        r"цвет\s+плод|форма\s+плод|вкус|устойчивост)",
    ),
    "flavor": (COMPOSITION_DEPENDENCY_PATTERN,),
    "ingredients": (r"(?:вкус|flavou?r)",),
    "material": (COMPOSITION_DEPENDENCY_PATTERN,),
    "material_composition": (r"(?:материал|material)",),
}
DEFAULT_ALLOWED_ANCHOR_CONTEXT_KEYS = (
    "Бренд",
)
ALLOWED_ANCHOR_CONTEXT_KEYS: dict[str, tuple[str, ...]] = {
    # Changing a brand while preserving a manufacturer would create a second,
    # usually contradictory identity field.
    "brand": ("Назначение",),
    "car_brand": ("Назначение",),
    "model": ("Бренд",),
    "cpu_model": ("Бренд",),
    "processor_model": ("Бренд",),
    "product_line": ("Бренд",),
    "variety": ("Бренд",),
}
CATEGORY_DEPENDENT_ATTRIBUTE_PATTERNS: dict[
    tuple[str, str], tuple[str, ...]
] = {
    ("Продукты питания", "color"): (
        COMPOSITION_DEPENDENCY_PATTERN,
        r"(?:вкус|flavou?r)",
    ),
    ("Строительство и ремонт", "diameter"): (
        r"(?:длина|высота|ширина|рабоч\w*\s+част|общ\w*\s+длина|length|height|width)",
    ),
    ("Красота и гигиена", "model"): (
        r"(?:ноты|группа\s+аромата|семейство\s+аромата|состав|ингредиент|"
        r"fragrance|composition|ingredient)",
    ),
}
TARGET_VALUE_PATTERNS: dict[str, str] = {
    "axis": r"(?:ось\s*)?(?:[1-9]\d?|1[0-7]\d|180)(?:\s*(?:°|град(?:ус(?:а|ов)?)?))?",
    "color_code": r"(?=.*\d)(?:цвет\s*)?#?[a-zа-яё0-9][a-zа-яё0-9 ._/-]{0,24}",
    "package_quantity": r"[1-9]\d{0,4}\s*(?:шт\.?|штук(?:а|и)?|ед\.?|единиц(?:а|ы)?)",
    "size": (
        r"(?:(?=.*\d)[a-zа-яё0-9+.,/×x*\"' -]{1,60}|"
        r"xxs|xs|s|m|l|xl|xxl|xxxl)"
    ),
    "length": r"[+]?(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*(?:мм|см|м|дюйм\w*|inch(?:es)?|in|\")",
    "length_mm": r"[+]?(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*(?:мм|см|м)",
    "width": r"[+]?(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*(?:мм|см|м|дюйм\w*|inch(?:es)?|in|\")",
    "diameter": r"[+]?(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*(?:мм|см|м|дюйм\w*|inch(?:es)?|in|\")",
    "case_diameter": r"[+]?(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*(?:мм|см|дюйм\w*|inch(?:es)?|in|\")",
    "wheel_diameter": r"[+]?(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*(?:мм|см|дюйм\w*|inch(?:es)?|in|\")",
    "optical_power": r"[+-]?\d+(?:[.,]\d+)?(?:\s*(?:дптр|d|диоптр\w*))?",
    "storage_capacity": r"[1-9]\d*(?:[.,]\d+)?(?:\s*(?:гб|gb|тб|tb))?",
}
FORBIDDEN_TARGET_VALUE_PATTERNS: dict[str, str] = {
    "thickness": r"(?:г\s*/?\s*м\s*(?:2|²)|gsm)",
    "model": r"^(?:модель\s*)?(?:№|#|no)?\s*\d[\d ._/-]*$",
    "cpu_model": r"^(?:модель\s*)?(?:№|#|no)?\s*\d[\d ._/-]*$",
    "processor_model": r"^(?:модель\s*)?(?:№|#|no)?\s*\d[\d ._/-]*$",
}
SEMANTIC_REVIEW_CONCEPTS = {
    "car_brand",
    "compatibility",
    "compatible_model",
    "country_of_origin",
    "guitar_type",
    "gender",
    "material_composition",
    "metal_color",
    "model_compatibility",
    "resource",
    "paper_density",
    "type",
    "weight",
}
SEMANTIC_REVIEW_CATEGORY_CONCEPTS: set[tuple[str, str]] = {
    ("Товары для животных", "volume"),
    ("Электроника", "material"),
}
FORBIDDEN_PRODUCT_TYPE_PROFILES: dict[tuple[str, str], set[str]] = {
    ("Продукты питания", "package_quantity"): {"чай листовой"},
    ("Автотовары", "color"): {
        "краска автомобильная",
        "средство для ремонта царапин",
    },
    ("Дом и сад", "variety"): {"саженец"},
    ("Канцелярские товары", "diameter"): {"пружина"},
    ("Мебель", "model"): {"комплект мебели для ванной"},
    ("Спорт и отдых", "wheel_diameter"): {"камера"},
    ("Ювелирные изделия", "insert"): {"кольцо"},
}
FORBIDDEN_PRODUCT_TYPES = {"искусственный", "комкующийся"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, default=DEFAULT_STATISTICS)
    parser.add_argument("--occurrences", type=Path, default=DEFAULT_OCCURRENCES)
    parser.add_argument("--pair-inputs", type=Path, default=DEFAULT_PAIR_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--profile-capacity-policy",
        type=Path,
        default=DEFAULT_PROFILE_CAPACITY_POLICY,
    )
    parser.add_argument("--target-label", type=int, choices=(0, 1), default=0)
    parser.add_argument("--minimum-singletons", type=int, default=2)
    parser.add_argument("--minimum-probability", type=float, default=0.8)
    parser.add_argument("--minimum-observed-key-support", type=int, default=2)
    parser.add_argument("--minimum-product-type-pairs", type=int, default=2)
    parser.add_argument("--maximum-product-types", type=int, default=12)
    parser.add_argument("--maximum-source-examples", type=int, default=4)
    parser.add_argument(
        "--include-semantic-review",
        action="store_true",
        help="include concepts whose label meaning needs a separate experiment",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o644)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_profile_capacity_policy(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    policy = json.loads(raw)
    if not isinstance(policy, dict) or int(policy.get("schema_version", 0)) != 1:
        raise ValueError("profile capacity policy must use schema_version=1")
    if policy.get("policy_version") != "profile_capacity_policy_v1":
        raise ValueError("unexpected profile capacity policy version")
    aliases = policy.get("product_type_aliases")
    profiles = policy.get("profile_policies")
    if not isinstance(aliases, list) or not isinstance(profiles, list):
        raise ValueError("profile capacity policy aliases/profiles must be arrays")
    return policy, hashlib.sha256(raw).hexdigest()


def product_type_alias_map(policy: dict[str, Any]) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for row in policy["product_type_aliases"]:
        category = str(row.get("category") or "").strip()
        alias = normalize_text(row.get("alias"))
        canonical = str(row.get("canonical") or "").strip()
        if not category or not alias or not normalize_text(canonical):
            raise ValueError(f"invalid product type alias: {row}")
        key = (category, alias)
        if key in result and normalize_text(result[key]) != normalize_text(canonical):
            raise ValueError(f"conflicting product type alias: {row}")
        result[key] = canonical
    return result


def resolved_profile_policy(
    policy: dict[str, Any], *, category: str, concept: str, product_type: str
) -> dict[str, Any]:
    normalized_type = normalize_text(product_type)
    exact: list[dict[str, Any]] = []
    wildcard: list[dict[str, Any]] = []
    for row in policy["profile_policies"]:
        if str(row.get("category")) != category or str(row.get("concept")) != concept:
            continue
        scoped = str(row.get("product_type") or "")
        if scoped == "*":
            wildcard.append(row)
        elif normalize_text(scoped) == normalized_type:
            exact.append(row)
    matches = exact or wildcard
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous profile capacity policy for {(category, concept, product_type)}"
        )
    if not matches:
        return {}
    row = dict(matches[0])
    domain = [str(value).strip() for value in row.get("target_value_domain") or []]
    if len(domain) < 2 or len({normalize_text(value) for value in domain}) != len(domain):
        raise ValueError(f"invalid target value domain: {row}")
    cap = int(row.get("primary_task_safety_cap") or 0)
    if cap < 1:
        raise ValueError(f"invalid primary task safety cap: {row}")
    return {
        "target_value_domain": domain,
        "primary_task_safety_cap": cap,
    }


def stable_rule_id(category: str, concept: str, relation: str, label: int) -> str:
    payload = json.dumps(
        [category, concept, relation, label],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def stable_profile_rule_id(
    category: str,
    concept: str,
    relation: str,
    label: int,
    product_type: str,
) -> str:
    payload = json.dumps(
        [category, concept, relation, label, normalize_text(product_type)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def clean_attribute_key(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def observed_attribute_key(
    facts: pd.DataFrame,
    category: str,
    concept: str,
    relation: str,
) -> tuple[str, int, list[dict[str, Any]]]:
    frame = facts[
        facts["category"].eq(category)
        & facts["concept"].eq(concept)
        & facts["relation"].eq(relation)
        & facts["is_singleton"]
    ]
    frame = frame[
        ~frame["pair_id"].astype(str).isin(EXCLUDED_SOURCE_PAIR_IDS)
    ]
    if "targets_forbidden_identifier" in frame.columns:
        frame = frame[~frame["targets_forbidden_identifier"]]
    values: list[str] = []
    for side in ("a", "b"):
        source_column = f"source_{side}"
        attribute_column = f"raw_attribute_{side}"
        supported = frame[frame[source_column].eq("attribute")]
        values.extend(clean_attribute_key(value) for value in supported[attribute_column])
    values = [value for value in values if value]
    counts = Counter(values)
    top = [
        {"attribute_key": key, "support": int(support)}
        for key, support in counts.most_common(5)
    ]
    if not top:
        return "", 0, []
    return str(top[0]["attribute_key"]), int(top[0]["support"]), top


def forbidden_attribute_support(
    facts: pd.DataFrame,
    category: str,
    concept: str,
    relation: str,
) -> int:
    frame = facts[
        facts["category"].eq(category)
        & facts["concept"].eq(concept)
        & facts["relation"].eq(relation)
        & facts["is_singleton"]
    ]
    support = 0
    for side in ("a", "b"):
        support += int(
            (
                frame[f"source_{side}"].eq("attribute")
                & frame[f"raw_attribute_{side}"].fillna("").astype(str).str.contains(
                    FORBIDDEN_ATTRIBUTE_RE
                )
            ).sum()
        )
    return support


def explicit_product_type(raw_attributes: Any, title: Any) -> str:
    try:
        attributes = parse_attributes(raw_attributes)
    except Exception:
        return ""
    normalized = {normalize_text(key): str(value).strip() for key, value in attributes.items()}
    for key in PRODUCT_TYPE_KEY_PRIORITY:
        value = " ".join(normalized.get(key, "").split())
        normalized_value = normalize_text(value)
        if not normalized_value or normalized_value in FORBIDDEN_PRODUCT_TYPES:
            continue
        normalized_title = normalize_text(title)
        match = re.search(
            rf"(?<!\w){re.escape(normalized_value)}(?!\w)", normalized_title
        )
        if match is None:
            continue
        # A value mentioned only as the object of «для» describes a compatible
        # item, not the item being sold (e.g. strings for an acoustic guitar).
        title_prefix = normalized_title[: match.start()].rstrip()
        if re.search(r"(?:^|\s)(?:для|под|к|на)\s*$", title_prefix):
            continue
        return value[:160]
    return ""


def product_type_profiles(
    facts: pd.DataFrame,
    pair_inputs: pd.DataFrame,
    *,
    category: str,
    concept: str,
    relation: str,
    label: int,
    minimum_support: int,
    minimum_probability: float,
    maximum_types: int,
    maximum_examples: int,
    type_aliases: dict[tuple[str, str], str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    frame = facts[
        facts["category"].eq(category)
        & facts["concept"].eq(concept)
        & facts["relation"].eq(relation)
        & facts["is_singleton"]
    ]
    frame = frame[
        ~frame["pair_id"].astype(str).isin(EXCLUDED_SOURCE_PAIR_IDS)
    ]
    if "targets_forbidden_identifier" in frame.columns:
        frame = frame[~frame["targets_forbidden_identifier"]]
    frame = frame.drop_duplicates("pair_id")
    inputs_by_pair = pair_inputs
    raw_names: dict[str, Counter[str]] = {}
    pair_labels: dict[str, dict[str, int]] = {}
    split_labels: dict[str, dict[str, dict[str, int]]] = {}
    pair_ids_by_type: dict[str, list[str]] = {}
    pair_types: dict[str, set[str]] = {}
    for fact in frame.itertuples(index=False):
        pair_id = str(fact.pair_id)
        if pair_id not in inputs_by_pair.index:
            continue
        source = inputs_by_pair.loc[pair_id]
        if isinstance(source, pd.DataFrame):
            source = source.iloc[0]
        values = {
            explicit_product_type(source["attributes_a_json"], source["title_a"]),
            explicit_product_type(source["attributes_b_json"], source["title_b"]),
        }
        values.discard("")
        normalized_types: set[str] = set()
        for value in values:
            canonical_value = type_aliases.get(
                (category, normalize_text(value)), value
            )
            normalized_type = normalize_text(canonical_value)
            if not normalized_type:
                continue
            normalized_types.add(normalized_type)
            raw_names.setdefault(normalized_type, Counter())[canonical_value] += 1
        pair_types[pair_id] = normalized_types
        for normalized_type in normalized_types:
            counts = pair_labels.setdefault(normalized_type, {"0": 0, "1": 0})
            counts[str(int(fact.human_label))] += 1
            split = str(fact.split)
            split_counts = split_labels.setdefault(normalized_type, {}).setdefault(
                split, {"0": 0, "1": 0}
            )
            split_counts[str(int(fact.human_label))] += 1
            pair_ids_by_type.setdefault(normalized_type, []).append(pair_id)

    profiles: list[dict[str, Any]] = []
    for normalized_type, counts in pair_labels.items():
        support = counts["0"] + counts["1"]
        target_support = counts[str(label)]
        probability = target_support / support if support else 0.0
        if support < minimum_support or probability < minimum_probability:
            continue
        display = raw_names[normalized_type].most_common(1)[0][0]
        target_pair_ids = [
            pair_id
            for pair_id in pair_ids_by_type[normalized_type]
            if int(frame.loc[frame["pair_id"].eq(pair_id), "human_label"].iloc[0])
            == label
        ]
        profile_split_statistics: dict[str, dict[str, float | int]] = {}
        for split in ("discovery", "validation"):
            split_counts = split_labels.get(normalized_type, {}).get(
                split, {"0": 0, "1": 0}
            )
            split_support = int(split_counts["0"] + split_counts["1"])
            split_target_support = int(split_counts[str(label)])
            profile_split_statistics[split] = {
                "support": split_support,
                "target_support": split_target_support,
                "target_probability": (
                    split_target_support / split_support if split_support else 0.0
                ),
            }
        profiles.append(
            {
                "product_type": display,
                "normalized_product_type": normalized_type,
                "singleton_pair_support": int(support),
                "singleton_target_support": int(target_support),
                "singleton_target_probability": float(probability),
                "split_statistics": profile_split_statistics,
                "source_pair_ids": list(dict.fromkeys(target_pair_ids))[:8],
            }
        )
    profiles.sort(
        key=lambda row: (
            -int(row["singleton_pair_support"]),
            -float(row["singleton_target_probability"]),
            str(row["normalized_product_type"]),
        )
    )
    forbidden_profiles = {
        normalize_text(value) for value in FORBIDDEN_PRODUCT_TYPES
    } | {
        normalize_text(value)
        for value in FORBIDDEN_PRODUCT_TYPE_PROFILES.get((category, concept), set())
    }
    profiles = [
        profile
        for profile in profiles
        if str(profile["normalized_product_type"]) not in forbidden_profiles
    ]
    profiles = profiles[:maximum_types]

    examples: list[dict[str, Any]] = []
    fact_by_pair = frame.set_index("pair_id", drop=False)
    for profile in profiles:
        profile_examples: list[dict[str, Any]] = []
        seen_pair_ids: set[str] = set()
        for pair_id in profile["source_pair_ids"]:
            if pair_id in seen_pair_ids or len(profile_examples) >= maximum_examples:
                continue
            source = inputs_by_pair.loc[pair_id]
            if isinstance(source, pd.DataFrame):
                source = source.iloc[0]
            fact = fact_by_pair.loc[pair_id]
            if isinstance(fact, pd.DataFrame):
                fact = fact.iloc[0]
            profile_examples.append(
                {
                    "source_pair_id": pair_id,
                    "product_type": profile["product_type"],
                    "title_a": str(source["title_a"])[:280],
                    "title_b": str(source["title_b"])[:280],
                    "target_value_a": str(fact.raw_value_a)[:160],
                    "target_value_b": str(fact.raw_value_b)[:160],
                }
            )
            seen_pair_ids.add(pair_id)
        profile["source_examples"] = profile_examples
        examples.extend(profile_examples)
    diagnostics = {
        "source_singleton_pairs": int(len(frame)),
        "source_pairs_with_explicit_type": int(
            sum(bool(values) for values in pair_types.values())
        ),
        "observed_exact_product_types": int(len(pair_labels)),
        "supported_product_type_profiles": int(len(profiles)),
    }
    return profiles, examples, diagnostics


def target_columns(label: int, prefix: str = "all") -> tuple[str, str]:
    return f"{prefix}_singleton_label{label}", f"{prefix}_singleton_p{label}"


def generation_anchor_hint(concept: str, attribute_key: str) -> str:
    base = (
        f"Создай товар, где характеристику «{attribute_key}» можно изменить независимо. "
        "Не добавляй другие атрибуты, которые логически фиксируют её значение или "
        "станут противоречивыми после замены."
    )
    if specific := ANCHOR_HINTS.get(concept):
        base += " " + specific
    if concept in {"brand", "car_brand", "gauge", "string_count", "product_line"}:
        base += " Не добавляй модель, серию или линейку, зависящую от целевого значения."
    if concept in {"model", "car_model", "cpu_model", "processor_model"}:
        base += (
            " Используй вымышленные нейтральные модели и не копируй известный "
            "бренд+модель. Модель должна содержать буквы и не может быть полностью "
            "числовым каталожным номером."
        )
    if concept in {
        "type",
        "vehicle_type",
        "guitar_type",
        "product_form",
        "flavor",
        "ingredients",
        "material_composition",
    }:
        return (
            base
            + " В частности, не указывай состав, сырьё, конструкцию или назначение, "
            "которые верны только для исходного значения."
        )
    if concept == "material":
        return (
            base
            + " Новое значение должно обозначать физически другой материал, а не "
            "синоним или маркетинговое название того же материала: например, "
            "искусственная кожа, экокожа и кожзам считаются одним семейством."
        )
    if concept == "country_of_origin":
        return (
            base
            + " Бренд и модель не должны содержать название страны или явно "
            "противоречить обеим допустимым странам производства."
        )
    if concept == "package_quantity":
        return (
            base
            + " Записывай значение только как положительное целое число с явной "
            "единицей «шт», например «6 шт»; число без единицы недопустимо."
            + " Не дублируй количество в комплектации, весе, объёме или других "
            "полях. По примерам конкретного типа определи, означает ли оно число "
            "одинаковых товаров либо число основных штук, доз или элементов внутри "
            "продаваемой упаковки. Не считай аксессуары и рекламные вложения."
        )
    if concept in {"weight", "volume", "length", "length_mm", "width", "diameter"}:
        base += (
            " Характеристика должна относиться к самому товару, а не к упаковке, "
            "доставке или приблизительному диапазону."
        )
        if concept in {"length", "length_mm", "width", "diameter"}:
            base += (
                " Укажи одно число и явную физическую единицу мм, см, м или дюйм; "
                "число без единицы недопустимо."
            )
        return base
    if concept in {"case_diameter", "wheel_diameter"}:
        return (
            base
            + " Укажи одно число и явную физическую единицу мм, см или дюйм; "
            "число без единицы недопустимо."
        )
    return base


def main() -> None:
    args = parse_args()
    if args.minimum_singletons < 1:
        raise ValueError("minimum-singletons must be positive")
    if args.minimum_product_type_pairs < 1:
        raise ValueError("minimum-product-type-pairs must be positive")
    if args.maximum_product_types < 1 or args.maximum_source_examples < 1:
        raise ValueError("maximum product types/source examples must be positive")
    if not 0.5 <= args.minimum_probability <= 1.0:
        raise ValueError("minimum-probability must be in [0.5, 1.0]")

    statistics_path = args.statistics.resolve()
    occurrences_path = args.occurrences.resolve()
    pair_inputs_path = args.pair_inputs.resolve()
    output_path = args.output.resolve()
    profile_capacity_policy_path = args.profile_capacity_policy.resolve()
    profile_capacity_policy, profile_capacity_policy_sha256 = (
        load_profile_capacity_policy(profile_capacity_policy_path)
    )
    type_aliases = product_type_alias_map(profile_capacity_policy)
    statistics = pd.read_parquet(statistics_path)
    facts = pd.read_parquet(occurrences_path)
    pair_inputs = pd.read_parquet(pair_inputs_path)
    required_fact_columns = {
        "category",
        "pair_id",
        "human_label",
        "split",
        "concept",
        "relation",
        "is_singleton",
        "source_a",
        "source_b",
        "raw_attribute_a",
        "raw_attribute_b",
        "raw_value_a",
        "raw_value_b",
        "value_a",
        "value_b",
    }
    missing = required_fact_columns - set(facts.columns)
    if missing:
        raise ValueError(
            f"occurrences are missing {sorted(missing)}; rerun analyze_atomic_difference_rules.py"
        )
    facts = facts[
        ~facts["pair_id"].astype(str).isin(EXCLUDED_SOURCE_PAIR_IDS)
    ].copy()
    forbidden_identifier_mask = pd.Series(False, index=facts.index)
    for side in ("a", "b"):
        forbidden_identifier_mask |= facts[f"source_{side}"].eq("attribute") & facts[
            f"raw_attribute_{side}"
        ].fillna("").astype(str).str.contains(FORBIDDEN_ATTRIBUTE_RE)
    numeric_only_model_mask = facts["concept"].isin(
        {"model", "cpu_model", "processor_model"}
    ) & (
        facts["raw_value_a"].map(
            lambda value: bool(NUMERIC_ONLY_MODEL_VALUE_RE.fullmatch(normalize_text(value)))
        )
        | facts["raw_value_b"].map(
            lambda value: bool(NUMERIC_ONLY_MODEL_VALUE_RE.fullmatch(normalize_text(value)))
        )
    )
    forbidden_identifier_mask |= numeric_only_model_mask
    facts = facts.assign(
        targets_forbidden_identifier=forbidden_identifier_mask.astype(bool)
    )
    required_input_columns = {
        "pair_id",
        "title_a",
        "title_b",
        "attributes_a_json",
        "attributes_b_json",
    }
    missing_inputs = required_input_columns - set(pair_inputs.columns)
    if missing_inputs:
        raise ValueError(f"pair inputs are missing {sorted(missing_inputs)}")
    if pair_inputs["pair_id"].duplicated().any():
        raise ValueError("pair inputs contain duplicate pair_id values")
    pair_inputs_by_id = pair_inputs.set_index("pair_id", drop=False)

    label = args.target_label
    singleton_count_column, singleton_probability_column = target_columns(label)
    all_label_column = f"all_label{label}"
    discovery_count_column, discovery_probability_column = target_columns(
        label, "discovery"
    )
    validation_count_column, validation_probability_column = target_columns(
        label, "validation"
    )
    frame = statistics[
        statistics["level"].eq("category_concept")
        & statistics["relation"].eq("different_value")
        & statistics["all_singleton_support"].ge(args.minimum_singletons)
        & statistics[singleton_probability_column].ge(args.minimum_probability)
    ].copy()
    frame["all_occurrence_target_probability"] = (
        frame[all_label_column] / frame["all_pair_support"]
    )
    frame["unanimous_singletons"] = frame[singleton_count_column].eq(
        frame["all_singleton_support"]
    )
    frame = frame[
        frame["all_occurrence_target_probability"].ge(args.minimum_probability)
        | frame["unanimous_singletons"]
    ].copy()
    threshold_tag = (
        f"MIN{args.minimum_singletons}_P{int(round(args.minimum_probability * 100)):02d}"
    )

    exported: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in frame.sort_values(
        ["scope_category", "concept", "relation"], kind="stable"
    ).itertuples(index=False):
        category, concept, relation = (
            str(row.scope_category),
            str(row.concept),
            str(row.relation),
        )
        reason = ""
        if concept in GENERIC_CONCEPTS:
            reason = "generic_concept"
        elif FORBIDDEN_CONCEPT_RE.search(concept):
            reason = "forbidden_identifier_concept"

        clean_rule_facts = facts[
            facts["category"].eq(category)
            & facts["concept"].eq(concept)
            & facts["relation"].eq(relation)
            & ~facts["targets_forbidden_identifier"]
        ].drop_duplicates("pair_id")
        clean_singletons = clean_rule_facts[clean_rule_facts["is_singleton"]]
        clean_singleton_support = int(len(clean_singletons))
        clean_singleton_target_support = int(
            clean_singletons["human_label"].eq(label).sum()
        )
        clean_singleton_target_probability = (
            clean_singleton_target_support / clean_singleton_support
            if clean_singleton_support
            else 0.0
        )
        if not reason and clean_singleton_support < args.minimum_singletons:
            reason = "insufficient_non_identifier_singleton_support"
        elif (
            not reason
            and clean_singleton_target_probability < args.minimum_probability
        ):
            reason = "non_identifier_singleton_probability_below_threshold"

        clean_all_support = int(len(clean_rule_facts))
        clean_all_target_support = int(clean_rule_facts["human_label"].eq(label).sum())
        clean_all_target_probability = (
            clean_all_target_support / clean_all_support if clean_all_support else 0.0
        )
        clean_split_statistics: dict[str, dict[str, float | int]] = {}
        for split in ("discovery", "validation"):
            split_frame = clean_singletons[clean_singletons["split"].eq(split)]
            split_support = int(len(split_frame))
            split_target_support = int(split_frame["human_label"].eq(label).sum())
            clean_split_statistics[split] = {
                "support": split_support,
                "target_support": split_target_support,
                "target_probability": (
                    split_target_support / split_support if split_support else 0.0
                ),
            }

        observed_key, observed_key_support, top_keys = observed_attribute_key(
            facts, category, concept, relation
        )
        forbidden_source_attribute_support = forbidden_attribute_support(
            facts, category, concept, relation
        )
        if (
            not reason
            and forbidden_source_attribute_support
            >= max(args.minimum_observed_key_support, observed_key_support)
        ):
            reason = "forbidden_identifier_dominates_source_attributes"
        attribute_key = (
            CATEGORY_ATTRIBUTE_KEY_OVERRIDES.get((category, concept))
            or ATTRIBUTE_KEY_OVERRIDES.get(concept)
            or CONCEPT_ATTRIBUTE_KEYS.get(concept)
            or (
                observed_key
                if observed_key_support >= args.minimum_observed_key_support
                else ""
            )
        )
        if not reason and not attribute_key:
            reason = "no_reliable_attribute_key"
        elif not reason and FORBIDDEN_ATTRIBUTE_RE.search(attribute_key):
            reason = "forbidden_identifier_attribute"
        elif not reason and attribute_key.casefold() == "тип товара":
            reason = "reserved_product_type_attribute"

        profiles, source_examples, profile_diagnostics = product_type_profiles(
            facts,
            pair_inputs_by_id,
            category=category,
            concept=concept,
            relation=relation,
            label=label,
            minimum_support=args.minimum_product_type_pairs,
            minimum_probability=args.minimum_probability,
            maximum_types=args.maximum_product_types,
            maximum_examples=args.maximum_source_examples,
            type_aliases=type_aliases,
        )
        if not reason and not profiles:
            reason = "no_supported_product_type_profile"
        semantic_review_required = (
            concept in SEMANTIC_REVIEW_CONCEPTS
            or (category, concept) in SEMANTIC_REVIEW_CATEGORY_CONCEPTS
        )
        if not reason and semantic_review_required and not args.include_semantic_review:
            reason = "semantic_review_required"

        dependent_patterns = list(DEPENDENT_ATTRIBUTE_PATTERNS.get(concept, ()))
        dependent_patterns.extend(
            CATEGORY_DEPENDENT_ATTRIBUTE_PATTERNS.get((category, concept), ())
        )
        if concept == "car_model" and category == "Автотовары":
            mutation_mode = "compatibility_value"
        elif concept in {"brand", "car_brand", "model", "car_model", "product_line"}:
            mutation_mode = "identity_value"
        elif concept in {"compatibility", "compatible_model", "model_compatibility"}:
            mutation_mode = "compatibility_value"
        elif concept in {"type", "guitar_type", "product_form"}:
            mutation_mode = "product_type_value"
        else:
            mutation_mode = "independent_value"

        common = {
            "category": category,
            "concept": concept,
            "relation": relation,
            "label": label,
            "attribute_key": attribute_key,
            "attribute_key_source": (
                "category_override"
                if (category, concept) in CATEGORY_ATTRIBUTE_KEY_OVERRIDES
                else "override"
                if concept in ATTRIBUTE_KEY_OVERRIDES
                else "static_pair_rule_mapping"
                if concept in CONCEPT_ATTRIBUTE_KEYS
                else "dominant_singleton_evidence"
                if attribute_key
                else ""
            ),
            "observed_attribute_key_support": observed_key_support,
            "forbidden_source_attribute_support": (
                forbidden_source_attribute_support
            ),
            "top_observed_attribute_keys": top_keys,
            "parent_all_pair_support": int(row.all_pair_support),
            "parent_all_label0": int(row.all_label0),
            "parent_all_label1": int(row.all_label1),
            "parent_all_occurrence_target_probability": float(
                row.all_occurrence_target_probability
            ),
            "parent_singleton_support": int(row.all_singleton_support),
            "parent_singleton_label0": int(row.all_singleton_label0),
            "parent_singleton_label1": int(row.all_singleton_label1),
            "parent_singleton_target_probability": float(
                getattr(row, singleton_probability_column)
            ),
            "non_identifier_all_support": clean_all_support,
            "non_identifier_all_target_support": clean_all_target_support,
            "non_identifier_all_target_probability": clean_all_target_probability,
            "non_identifier_singleton_support": clean_singleton_support,
            "non_identifier_singleton_target_support": clean_singleton_target_support,
            "non_identifier_singleton_target_probability": (
                clean_singleton_target_probability
            ),
            "non_identifier_split_statistics": clean_split_statistics,
            "parent_discovery_singleton_support": int(
                row.discovery_singleton_support
            ),
            "parent_validation_singleton_support": int(
                row.validation_singleton_support
            ),
            "parent_example_pair_ids": str(row.all_example_pair_ids),
            "parent_example_value_pairs": str(row.all_example_value_pairs),
            "anchor_profiles": profiles,
            "allowed_product_types": [
                str(profile["product_type"]) for profile in profiles
            ],
            "allowed_anchor_context_keys": list(
                ALLOWED_ANCHOR_CONTEXT_KEYS.get(
                    concept, DEFAULT_ALLOWED_ANCHOR_CONTEXT_KEYS
                )
            ),
            "source_examples": source_examples,
            "profile_diagnostics": profile_diagnostics,
            "semantic_review_required": semantic_review_required,
            "mutation_mode": mutation_mode,
            "forbidden_anchor_attribute_patterns": list(
                dict.fromkeys(dependent_patterns)
            ),
            "target_value_pattern": TARGET_VALUE_PATTERNS.get(concept, ""),
            "forbidden_target_value_pattern": FORBIDDEN_TARGET_VALUE_PATTERNS.get(
                concept, ""
            ),
        }
        if reason:
            rejected.append({**common, "rejection_reason": reason})
            continue

        all_consistent = clean_all_target_probability >= args.minimum_probability
        source_suffix = stable_rule_id(category, concept, relation, label)
        for profile in profiles:
            product_type = str(profile["product_type"])
            capacity_policy = resolved_profile_policy(
                profile_capacity_policy,
                category=category,
                concept=concept,
                product_type=product_type,
            )
            profile_split_statistics = dict(profile["split_statistics"])
            profile_cross_split = (
                int(profile_split_statistics["discovery"]["support"]) >= 1
                and int(profile_split_statistics["validation"]["support"]) >= 1
                and float(
                    profile_split_statistics["discovery"]["target_probability"]
                )
                >= args.minimum_probability
                and float(
                    profile_split_statistics["validation"]["target_probability"]
                )
                >= args.minimum_probability
            )
            if not all_consistent:
                profile_tier = f"STAT_LABEL{label}_UNANIMOUS_OVERRIDE"
            elif profile_cross_split:
                profile_tier = (
                    f"STAT_LABEL{label}_CROSS_SPLIT_{threshold_tag}_SCOPED"
                )
            else:
                profile_tier = f"STAT_LABEL{label}_{threshold_tag}_SCOPED"
            profile_suffix = stable_profile_rule_id(
                category, concept, relation, label, product_type
            )
            profile_common = {
                **common,
                "anchor_profiles": [profile],
                "allowed_product_types": [product_type],
                "source_examples": list(profile.get("source_examples") or []),
                "singleton_support": int(profile["singleton_pair_support"]),
                "singleton_target_support": int(
                    profile["singleton_target_support"]
                ),
                "singleton_target_probability": float(
                    profile["singleton_target_probability"]
                ),
                "discovery_singleton_support": int(
                    profile_split_statistics["discovery"]["support"]
                ),
                "discovery_singleton_target_support": int(
                    profile_split_statistics["discovery"]["target_support"]
                ),
                "discovery_singleton_target_probability": float(
                    profile_split_statistics["discovery"]["target_probability"]
                ),
                "validation_singleton_support": int(
                    profile_split_statistics["validation"]["support"]
                ),
                "validation_singleton_target_support": int(
                    profile_split_statistics["validation"]["target_support"]
                ),
                "validation_singleton_target_probability": float(
                    profile_split_statistics["validation"]["target_probability"]
                ),
                "profile_diagnostics": {
                    **profile_diagnostics,
                    "supported_product_type_profiles": 1,
                },
                "target_value_domain": list(
                    capacity_policy.get("target_value_domain") or []
                ),
                "primary_task_safety_cap": capacity_policy.get(
                    "primary_task_safety_cap"
                ),
                "profile_capacity_policy_version": str(
                    profile_capacity_policy["policy_version"]
                ),
                "profile_capacity_policy_sha256": profile_capacity_policy_sha256,
            }
            exported.append(
                {
                    "generation_rule_id": f"gen_stat_{profile_suffix}",
                    "source_rule_id": f"stat_{source_suffix}",
                    "generation_tier": profile_tier,
                    "label": label,
                    "concept": concept,
                    "relation": relation,
                    "semantic_family": "statistical_atomic_difference",
                    "attribute_key": attribute_key,
                    "anchor_hint": generation_anchor_hint(concept, attribute_key),
                    "allowed_categories": [category],
                    "generation_action": (
                        "Replace this attribute with a physically different material "
                        "family, never a synonym or marketing alias."
                        if concept == "material"
                        else "Replace exactly this explicit attribute with a different, "
                        "realistic value for the same concrete product subtype."
                    ),
                    "required_postcondition": (
                        "Change only this attribute; preserve product subtype and all "
                        "unrelated facts, and update every title mention consistently."
                    ),
                    "selection_reason": (
                        "cross_split_threshold_and_product_type_scope"
                        if profile_cross_split
                        else "singleton_all_occurrence_and_product_type_threshold"
                        if all_consistent
                        else "unanimous_singleton_override"
                    ),
                    **profile_common,
                }
            )

    uniqueness_keys = [
        (
            str(rule["category"]),
            str(rule["concept"]),
            str(rule["relation"]),
            int(rule["label"]),
            normalize_text(rule["allowed_product_types"][0]),
        )
        for rule in exported
    ]
    if len(uniqueness_keys) != len(set(uniqueness_keys)):
        duplicates = [
            key for key, count in Counter(uniqueness_keys).items() if count > 1
        ]
        raise RuntimeError(f"duplicate canonical generation rule profiles: {duplicates}")
    for rule in exported:
        pair_ids = [
            str(example.get("source_pair_id") or "")
            for example in rule.get("source_examples") or []
        ]
        if len(pair_ids) != len(set(pair_ids)):
            raise RuntimeError(
                f"duplicate source evidence in {rule['generation_rule_id']}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(
        output_path,
        json.dumps(exported, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    review_path = output_path.with_suffix(".review.csv")
    atomic_write_bytes(
        review_path,
        pd.DataFrame(exported).to_csv(index=False).encode("utf-8-sig"),
    )
    rejected_path = output_path.with_suffix(".rejected.csv")
    atomic_write_bytes(
        rejected_path,
        pd.DataFrame(rejected).to_csv(index=False).encode("utf-8-sig"),
    )
    manifest = {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "statistics": str(statistics_path),
        "statistics_sha256": sha256(statistics_path),
        "occurrences": str(occurrences_path),
        "occurrences_sha256": sha256(occurrences_path),
        "pair_inputs": str(pair_inputs_path),
        "pair_inputs_sha256": sha256(pair_inputs_path),
        "profile_capacity_policy": str(profile_capacity_policy_path),
        "profile_capacity_policy_sha256": profile_capacity_policy_sha256,
        "selection": {
            "level": "category_concept_product_type",
            "relation": "different_value",
            "target_label": label,
            "minimum_singletons": args.minimum_singletons,
            "minimum_singleton_probability": args.minimum_probability,
            "minimum_all_occurrence_probability_or_unanimous_override": args.minimum_probability,
            "minimum_observed_attribute_key_support": args.minimum_observed_key_support,
            "minimum_product_type_pair_support": args.minimum_product_type_pairs,
            "minimum_product_type_target_probability": args.minimum_probability,
            "maximum_product_types_per_rule": args.maximum_product_types,
            "maximum_source_examples_per_product_type_rule": args.maximum_source_examples,
            "forbid_sku_article_part_number_oem": True,
            "require_non_identifier_singleton_evidence": True,
            "source_scoped_product_types": True,
            "split_rule_per_product_type": True,
            "profile_specific_support_and_split_tiers": True,
            "numeric_target_value_patterns": True,
            "source_examples_use_raw_side_aligned_values": True,
            "restrict_anchor_context_keys": True,
            "canonical_anchor_title_from_attribute_values": True,
            "semantic_value_equivalence_validation": True,
            "forbid_numeric_only_model_values": True,
            "canonical_product_type_aliases_before_profile_grouping": True,
            "deduplicate_canonical_profile_source_pairs": True,
            "finite_target_value_domain_validation": True,
            "canonical_target_values_in_semantic_signature": True,
            "canonical_quantity_units_required": True,
            "canonical_dimension_units_required": True,
            "prompt_source_examples_satisfy_effective_target_contract": True,
            "insert_stone_alias_equivalence_validation": True,
            "profile_capacity_policy_version": profile_capacity_policy[
                "policy_version"
            ],
            "profile_capacity_policy_sha256": profile_capacity_policy_sha256,
            "primary_task_capacity_formula": (
                "min(combinations(domain_size,2)*semantic_signature_limit,safety_cap)"
            ),
            "excluded_source_pair_ids": sorted(EXCLUDED_SOURCE_PAIR_IDS),
            "exclude_semantic_review": not args.include_semantic_review,
            "semantic_review_concepts": sorted(SEMANTIC_REVIEW_CONCEPTS),
            "semantic_review_category_concepts": sorted(
                [list(value) for value in SEMANTIC_REVIEW_CATEGORY_CONCEPTS]
            ),
        },
        "candidate_rows_before_semantic_key_filter": len(frame),
        "exported_rules": len(exported),
        "rejected_rules": len(rejected),
        "tier_counts": dict(Counter(rule["generation_tier"] for rule in exported)),
        "category_counts": dict(Counter(rule["category"] for rule in exported)),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "review_csv": str(review_path),
        "rejected_csv": str(rejected_path),
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    atomic_write_bytes(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
