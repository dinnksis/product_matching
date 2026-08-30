#!/usr/bin/env python3
"""Filter the raw Qwen A+B soft-positive pool for a weighted Kaggle ablation.

The raw pools deliberately trusted noisy atom statistics.  This builder keeps
that breadth, but only trains on candidates that are both plausible in the
human TRAIN vocabulary and useful to the frozen MiniLM baseline:

* exact-content deduplication across the A and B runs;
* human-train grounding of both generated endpoint values;
* frozen-baseline hardness and order-stability gates;
* deterministic category/rule/transition/product-type diversity caps;
* evidence-aware sample weights rather than treating every noisy atom equally.

Every one of the 31,176 deduplicated candidates is retained in
``candidate_decisions.parquet`` with its score, evidence and rejection reasons.
No language model or network service is called by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from item_pipeline.normalization import canonical_json_dumps, parse_attributes
from scripts.freeze_generated_pair_dataset import canonical_card
from src.data_pipeline import serialize_product


CONTRACT_VERSION = "soft_positive_quality_hardness_filter_v1"
LABEL_SOURCE = "qwen_soft_positive_ab_quality_hardness_v1"
WEIGHT_STATS_VERSION = "float64_le_pair_order_v1"
PAIR_DEDUP_VERSION = "exact_unordered_category_name_sorted_attributes_v1"
FACT_KEY_VERSION = "ordered_fact_tokens_serialize_product_6000_v1"
BASELINE_SCORER_VERSION = "frozen_minilm_ab_ba_mean_sigmoid_v1"
HUMAN_GROUNDING_VERSION = "human_train_category_product_type_attribute_values_v1"
SELECTION_POLICY_VERSION = "quality_hardness_diversity_caps_v1"
ID_REINDEX_VERSION = "pair_local_negative_int64_v1"

DEFAULT_SOURCE_A = ROOT / "item_pipeline/artifacts/soft_positive_tier_a_6500_qwen_v1_raw"
DEFAULT_SOURCE_B = ROOT / "item_pipeline/artifacts/soft_positive_tier_b_27930_qwen_v1_raw"
DEFAULT_CATALOG_A = ROOT / "configs/generation_rule_catalog_statistical_v1/soft_positive_ab_v1/tier_a.json"
DEFAULT_CATALOG_B = ROOT / "configs/generation_rule_catalog_statistical_v1/soft_positive_ab_v1/tier_b.json"
DEFAULT_SEMANTIC_REPORT = ROOT / "reports/semantic_atomic_rule_statistics_all_pairs_20260827/recommended_semantic_rules.parquet"
DEFAULT_HUMAN_ITEMS = ROOT / "data/items_human.parquet"
DEFAULT_VALIDATION_DIR = ROOT / "prepared/validation_splits_v1/human"
DEFAULT_BASELINE_MODEL = ROOT / "artifacts/kaggle/product-matching-minilm-5ep-human-ft-v1/minilm_llm_pretrain_5ep_human_ft_v1"
DEFAULT_OUTPUT = ROOT / "item_pipeline/artifacts/soft_positive_ab_quality_hardness_v1"
DEFAULT_ID_START = -7_700_000_000_000_000_000
EXPECTED_SOURCE_COUNTS = {"A": 6351, "B": 25006}
EXPECTED_DEDUPLICATED_CANDIDATES = 31176
FROZEN_OOD_CATEGORIES = frozenset({"Одежда", "Бытовая техника"})
TYPE_KEYS = frozenset({"тип", "тип товара", "вид товара", "категория товара", "product type"})
BRAND_KEYS = frozenset({"бренд", "brand", "торговая марка", "марка"})
WORD_RE = re.compile(r"[0-9a-zа-я]+", re.IGNORECASE)

DEFAULT_POLICY: dict[str, Any] = {
    "version": SELECTION_POLICY_VERSION,
    "minimum_rule_probability": 0.80,
    "minimum_rule_support": 3,
    "minimum_title_tokens": 4,
    "minimum_baseline_score": 0.015,
    "maximum_baseline_score": 0.80,
    "maximum_order_gap": 0.25,
    "minimum_quality_score": 0.48,
    "maximum_per_category": 250,
    "maximum_per_rule": 12,
    "maximum_per_rule_transition": 4,
    "maximum_per_source_rule": 24,
    "maximum_per_source_rule_transition": 8,
    "maximum_per_semantic_signature": 4,
    "maximum_per_category_product_type": 25,
    "hardness_center": 0.12,
    "hardness_log_scale": 0.95,
}


class FilterError(ValueError):
    """Raised when the filtering/provenance contract is violated."""


@dataclass(frozen=True)
class SourceSnapshot:
    tier: str
    path: Path
    run_signature: str
    items: pd.DataFrame
    pairs: pd.DataFrame
    metadata: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-a", type=Path, default=DEFAULT_SOURCE_A)
    parser.add_argument("--source-b", type=Path, default=DEFAULT_SOURCE_B)
    parser.add_argument("--catalog-a", type=Path, default=DEFAULT_CATALOG_A)
    parser.add_argument("--catalog-b", type=Path, default=DEFAULT_CATALOG_B)
    parser.add_argument("--semantic-report", type=Path, default=DEFAULT_SEMANTIC_REPORT)
    parser.add_argument("--human-items", type=Path, default=DEFAULT_HUMAN_ITEMS)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--baseline-model", type=Path, default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--baseline-scores", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--id-start", type=int, default=DEFAULT_ID_START)
    parser.add_argument("--overwrite", action="store_true")
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


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().replace("ё", "е")
    return " ".join(text.split())


def normalized_key(value: Any) -> str:
    text = normalize_text(value).replace("_", " ")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return " ".join(text.split())


def fact_tokens(value: Any) -> tuple[str, ...]:
    return tuple(WORD_RE.findall(normalize_text(value)))


def json_array(values: Iterable[Any]) -> str:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


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


def _attributes_for_card(card: Mapping[str, Any]) -> dict[str, str]:
    raw = card.get("attributes", {})
    if isinstance(raw, str):
        return parse_attributes(raw)
    if not isinstance(raw, Mapping):
        raise FilterError("card attributes must be a JSON object or mapping")
    return {str(key): str(value) for key, value in raw.items()}


def canonical_card_key(card: Mapping[str, Any] | pd.Series) -> str:
    """Exact card-content identity used only for A/B source deduplication."""

    record = card.to_dict() if isinstance(card, pd.Series) else dict(card)
    attributes = _attributes_for_card(record)
    return canonical_json_dumps(
        {
            "category": str(record.get("category", "")),
            "name": str(record.get("name", "")),
            "attributes": sorted((str(key), str(value)) for key, value in attributes.items()),
        }
    )


def canonical_pair_key(
    card_a: Mapping[str, Any] | pd.Series,
    card_b: Mapping[str, Any] | pd.Series,
    target: int = 1,
) -> str:
    first, second = sorted((canonical_card_key(card_a), canonical_card_key(card_b)))
    return sha256_json({"cards": [first, second], "target": int(target)})


def _normalized_global_card_key(card: Mapping[str, Any]) -> str:
    return canonical_card(pd.Series(card))


def _fact_card_key(card: Mapping[str, Any]) -> tuple[str, ...]:
    serialized = serialize_product(
        pd.Series(
            {
                "category": str(card["category"]),
                "name": str(card["name"]),
                "attributes": json.dumps(
                    _attributes_for_card(card), ensure_ascii=False, separators=(",", ":")
                ),
            }
        ),
        max_attribute_chars=6000,
    )
    return fact_tokens(serialized)


def exact_weight_stats(weights: Sequence[float] | np.ndarray | pd.Series) -> dict[str, Any]:
    values = np.asarray(list(weights), dtype="<f8")
    if values.ndim != 1 or len(values) == 0:
        raise FilterError("sample weights must be a nonempty one-dimensional sequence")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise FilterError("sample weights must be finite and strictly positive")
    return {
        "version": WEIGHT_STATS_VERSION,
        "count": int(len(values)),
        "sum": float(math.fsum(float(value) for value in values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
    }


def _float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def score_candidate(
    row: Mapping[str, Any], policy: Mapping[str, Any] | None = None
) -> float:
    """Return the deterministic quality/hardness score used for ranking."""

    active = {**DEFAULT_POLICY, **dict(policy or {})}
    probability = np.clip((_float(row, "rule_probability") - 0.70) / 0.30, 0.0, 1.0)
    support = np.clip(
        math.log1p(max(0.0, _float(row, "rule_support"))) / math.log1p(20.0),
        0.0,
        1.0,
    )
    singleton = np.clip(
        _float(row, "rule_singleton_support")
        / max(1.0, _float(row, "rule_support", 1.0)),
        0.0,
        1.0,
    )
    cross_raw = row.get("cross_split_p80")
    cross_split = 1.0 if cross_raw is True else (0.0 if cross_raw is False else 0.35)
    evidence_score = 0.40 * probability + 0.25 * support + 0.20 * singleton + 0.15 * cross_split

    evidence_type = str(row.get("evidence_type", ""))
    grounding = {
        "source_exact_transition": 1.00,
        "source_endpoint_values": 0.90,
        "human_scope_attribute_values": 0.82,
        "human_category_attribute_values": 0.62,
    }.get(evidence_type, 0.0)
    minimum_tokens = _float(row, "minimum_title_tokens")
    name_similarity = _float(row, "name_similarity")
    realism = (
        0.60 * np.clip((minimum_tokens - 3.0) / 5.0, 0.0, 1.0)
        + 0.40 * np.clip(1.0 - abs(name_similarity - 0.80) / 0.25, 0.0, 1.0)
    )
    baseline_score = max(_float(row, "baseline_score"), 1e-8)
    hardness = math.exp(
        -abs(math.log(baseline_score) - math.log(float(active["hardness_center"])))
        / float(active["hardness_log_scale"])
    )
    brand_context = 1.0 if bool(row.get("brand_context_grounded")) else 0.55
    quality = (
        0.34 * evidence_score
        + 0.27 * grounding
        + 0.16 * float(realism)
        + 0.18 * hardness
        + 0.05 * brand_context
    )
    return float(np.clip(quality, 0.0, 1.0))


def _hard_rejection_reasons(row: Mapping[str, Any], policy: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if str(row.get("relation", "")) == "incompatible":
        reasons.append("incompatible_relation")
    if _float(row, "rule_probability") < float(policy["minimum_rule_probability"]):
        reasons.append("rule_probability_below_minimum")
    if _float(row, "rule_support") < float(policy["minimum_rule_support"]):
        reasons.append("rule_support_below_minimum")
    if not bool(row.get("category_values_grounded")):
        reasons.append("target_values_not_grounded_in_human_train")
    if normalize_text(row.get("original_value")) == normalize_text(row.get("new_value")):
        reasons.append("normalized_target_values_equal")
    score = _float(row, "baseline_score", -1.0)
    if score < float(policy["minimum_baseline_score"]):
        reasons.append("baseline_score_below_hardness_window")
    if score > float(policy["maximum_baseline_score"]):
        reasons.append("baseline_score_above_hardness_window")
    if _float(row, "score_order_gap", float("inf")) > float(policy["maximum_order_gap"]):
        reasons.append("baseline_order_gap_too_large")
    if _float(row, "minimum_title_tokens") < float(policy["minimum_title_tokens"]):
        reasons.append("title_too_short")
    if bool(row.get("forbidden_ood_category")):
        reasons.append("forbidden_ood_category")
    if bool(row.get("validation_fact_overlap")):
        reasons.append("frozen_validation_fact_overlap")
    if _float(row, "quality_score") < float(policy["minimum_quality_score"]):
        reasons.append("quality_score_below_minimum")
    return reasons


def select_candidates(
    rows: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Greedily select candidates after hard gates and diversity caps."""

    active = {**DEFAULT_POLICY, **dict(policy or {})}
    prepared: list[dict[str, Any]] = []
    rejection_map: dict[str, list[str]] = {}
    for raw in rows:
        row = dict(raw)
        row["quality_score"] = _float(row, "quality_score", score_candidate(row, active))
        key = str(row["candidate_key"])
        reasons = _hard_rejection_reasons(row, active)
        rejection_map[key] = reasons
        if not reasons:
            prepared.append(row)

    prepared.sort(
        key=lambda row: (
            -_float(row, "quality_score"),
            _float(row, "baseline_score"),
            0 if str(row.get("source_tier")) == "A" else 1,
            int(row.get("source_task_index", 0)),
            str(row["candidate_key"]),
        )
    )
    category_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    transition_counts: Counter[tuple[str, str]] = Counter()
    source_rule_counts: Counter[str] = Counter()
    source_transition_counts: Counter[tuple[str, str]] = Counter()
    semantic_signature_counts: Counter[str] = Counter()
    product_type_counts: Counter[tuple[str, str]] = Counter()
    used_cards: set[str] = set()
    selected: list[dict[str, Any]] = []

    for row in prepared:
        key = str(row["candidate_key"])
        category = str(row["category"])
        rule_id = str(row["rule_id"])
        source_rule_id = str(row["source_rule_id"])
        transition = str(row["transition_key"])
        semantic_signature = str(row.get("semantic_signature") or row["candidate_key"])
        product_type = str(row["product_type_normalized"])
        cap_reason = ""
        if category_counts[category] >= int(active["maximum_per_category"]):
            cap_reason = "category_cap_reached"
        elif rule_counts[rule_id] >= int(active["maximum_per_rule"]):
            cap_reason = "rule_cap_reached"
        elif transition_counts[(rule_id, transition)] >= int(active["maximum_per_rule_transition"]):
            cap_reason = "rule_transition_cap_reached"
        elif source_rule_counts[source_rule_id] >= int(active["maximum_per_source_rule"]):
            cap_reason = "source_rule_cap_reached"
        elif source_transition_counts[(source_rule_id, transition)] >= int(
            active["maximum_per_source_rule_transition"]
        ):
            cap_reason = "source_rule_transition_cap_reached"
        elif semantic_signature_counts[semantic_signature] >= int(
            active["maximum_per_semantic_signature"]
        ):
            cap_reason = "semantic_signature_cap_reached"
        elif product_type_counts[(category, product_type)] >= int(
            active["maximum_per_category_product_type"]
        ):
            cap_reason = "category_product_type_cap_reached"
        elif str(row["global_card_key_a"]) in used_cards or str(row["global_card_key_b"]) in used_cards:
            cap_reason = "global_card_reuse"
        if cap_reason:
            rejection_map[key] = [cap_reason]
            continue
        selected.append(row)
        category_counts[category] += 1
        rule_counts[rule_id] += 1
        transition_counts[(rule_id, transition)] += 1
        source_rule_counts[source_rule_id] += 1
        source_transition_counts[(source_rule_id, transition)] += 1
        semantic_signature_counts[semantic_signature] += 1
        product_type_counts[(category, product_type)] += 1
        used_cards.add(str(row["global_card_key_a"]))
        used_cards.add(str(row["global_card_key_b"]))
        rejection_map[key] = []

    return selected, rejection_map


def _load_source(path: Path, tier: str) -> SourceSnapshot:
    path = absolute(path)
    required = [
        path / "items.parquet",
        path / "pairs.parquet",
        path / "pair_generation_metadata.parquet",
        path / "summary.json",
        path / "validation_report.json",
    ]
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    validation = json.loads((path / "validation_report.json").read_text(encoding="utf-8"))
    items = pd.read_parquet(path / "items.parquet")
    pairs = pd.read_parquet(path / "pairs.parquet")
    metadata = pd.read_parquet(path / "pair_generation_metadata.parquet")
    expected = EXPECTED_SOURCE_COUNTS[tier]
    if len(pairs) != expected or len(metadata) != expected or len(items) != 2 * expected:
        raise FilterError(f"source {tier} count mismatch: {len(pairs)}/{len(items)}/{len(metadata)}")
    if not bool(validation.get("valid")) or int(validation.get("pairs", -1)) != expected:
        raise FilterError(f"source {tier} validation is not a complete valid snapshot")
    if set(pd.to_numeric(pairs["target"], errors="raise").astype(int)) != {1}:
        raise FilterError(f"source {tier} is not all target=1")
    pair_keys = list(zip(pairs["id1"].astype(int), pairs["id2"].astype(int)))
    metadata_keys = list(zip(metadata["id1"].astype(int), metadata["id2"].astype(int)))
    if pair_keys != metadata_keys:
        raise FilterError(f"source {tier} pairs and metadata are misaligned")
    run_signatures = set(metadata["run_signature"].astype(str))
    if len(run_signatures) != 1 or summary.get("run_signature") not in run_signatures:
        raise FilterError(f"source {tier} run signature mismatch")
    return SourceSnapshot(
        tier=tier,
        path=path,
        run_signature=next(iter(run_signatures)),
        items=items,
        pairs=pairs,
        metadata=metadata,
    )


def _load_catalog(path: Path, tier: str) -> dict[str, dict[str, Any]]:
    path = absolute(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise FilterError(f"catalog {path} must be a JSON list")
    result: dict[str, dict[str, Any]] = {}
    for raw in payload:
        row = dict(raw)
        rule_id = str(row.get("generation_rule_id", ""))
        if not rule_id or rule_id in result:
            raise FilterError(f"invalid or duplicate rule ID in {tier} catalog")
        row["_source_tier"] = tier
        result[rule_id] = row
    return result


def _parse_one_json_list(value: Any, field: str) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise FilterError(f"invalid {field}: {error}") from error
    if not isinstance(parsed, list) or len(parsed) != 1 or not isinstance(parsed[0], dict):
        raise FilterError(f"{field} must contain exactly one object")
    return dict(parsed[0])


def _source_candidates(
    snapshots: Sequence[SourceSnapshot],
    catalog: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    by_pair_content: dict[str, int] = {}
    duplicates: Counter[tuple[str, str]] = Counter()
    raw_count = 0
    for snapshot in snapshots:
        item_map = {
            int(row.id): row._asdict() for row in snapshot.items.itertuples(index=False)
        }
        for pair, meta in zip(
            snapshot.pairs.itertuples(index=False), snapshot.metadata.to_dict("records")
        ):
            raw_count += 1
            card_a = dict(item_map[int(pair.id1)])
            card_b = dict(item_map[int(pair.id2)])
            pair_hash = canonical_pair_key(card_a, card_b, int(pair.target))
            candidate_key = f"{snapshot.run_signature}:{int(meta['task_index'])}"
            duplicate_source = {
                "candidate_key": candidate_key,
                "source_tier": snapshot.tier,
                "source_run_signature": snapshot.run_signature,
                "source_task_index": int(meta["task_index"]),
                "source_id1": int(pair.id1),
                "source_id2": int(pair.id2),
            }
            if pair_hash in by_pair_content:
                primary = candidates[by_pair_content[pair_hash]]
                primary["duplicate_sources"].append(duplicate_source)
                duplicates[(str(primary["source_tier"]), snapshot.tier)] += 1
                continue
            rule_id = str(meta["scheduled_primary_rule_id"])
            if rule_id not in catalog:
                raise FilterError(f"source rule absent from catalog: {rule_id}")
            rule = dict(catalog[rule_id])
            if rule.get("_source_tier") != snapshot.tier:
                raise FilterError(f"rule tier mismatch for {rule_id}")
            application = _parse_one_json_list(meta["applications_json"], "applications_json")
            original_value = str(application.get("original_value", ""))
            new_value = str(application.get("new_value", ""))
            required_key = str(application.get("attribute_key") or rule.get("attribute_key") or "")
            product_type = str(meta.get("product_type") or meta.get("scheduled_primary_product_type") or "")
            if not required_key or not product_type or not original_value or not new_value:
                raise FilterError(f"candidate {candidate_key} lacks target scope/application")
            attrs_a, attrs_b = _attributes_for_card(card_a), _attributes_for_card(card_b)
            tokens_a, tokens_b = fact_tokens(card_a["name"]), fact_tokens(card_b["name"])
            source_examples = list(rule.get("source_examples") or [])
            normalized_transition = tuple(sorted((normalize_text(original_value), normalize_text(new_value))))
            example_transitions = {
                tuple(
                    sorted(
                        (
                            normalize_text(example.get("target_value_a")),
                            normalize_text(example.get("target_value_b")),
                        )
                    )
                )
                for example in source_examples
            }
            source_values = {
                normalize_text(example.get(side))
                for example in source_examples
                for side in ("target_value_a", "target_value_b")
                if normalize_text(example.get(side))
            }
            brand_a = next(
                (str(value) for key, value in attrs_a.items() if normalized_key(key) in BRAND_KEYS),
                "",
            )
            brand_b = next(
                (str(value) for key, value in attrs_b.items() if normalized_key(key) in BRAND_KEYS),
                "",
            )
            row = {
                "candidate_key": candidate_key,
                "source_tier": snapshot.tier,
                "source_run_signature": snapshot.run_signature,
                "source_task_index": int(meta["task_index"]),
                "source_id1": int(pair.id1),
                "source_id2": int(pair.id2),
                "category": str(meta["category"]),
                "product_type": product_type,
                "product_type_normalized": normalize_text(product_type),
                "rule_id": rule_id,
                "source_rule_id": str(rule.get("source_rule_id", "")),
                "semantic_signature": str(meta.get("semantic_signature", "")),
                "concept": str(rule.get("concept", "")),
                "relation": str(rule.get("relation", "")),
                "required_attribute_key": required_key,
                "required_attribute_key_normalized": normalized_key(required_key),
                "original_value": original_value,
                "new_value": new_value,
                "original_value_normalized": normalize_text(original_value),
                "new_value_normalized": normalize_text(new_value),
                "transition_key": "|".join(normalized_transition),
                "rule_probability": float(rule.get("profile_target_probability", 0.0)),
                "rule_support": int(rule.get("profile_pair_support", 0)),
                "rule_singleton_support": int(rule.get("profile_singleton_pairs", 0)),
                "rule_label0": int(rule.get("profile_label0", 0)),
                "rule_label1": int(rule.get("profile_label1", 0)),
                "source_example_count": len(source_examples),
                "source_exact_transition": normalized_transition in example_transitions,
                "source_endpoint_values": all(value in source_values for value in normalized_transition),
                "minimum_title_tokens": min(len(tokens_a), len(tokens_b)),
                "mean_title_tokens": (len(tokens_a) + len(tokens_b)) / 2.0,
                "minimum_title_chars": min(len(str(card_a["name"])), len(str(card_b["name"]))),
                "name_similarity": float(meta.get("name_similarity", 0.0)),
                "brand_a": brand_a,
                "brand_b": brand_b,
                "pair_content_sha256": pair_hash,
                "global_card_key_a": _normalized_global_card_key(card_a),
                "global_card_key_b": _normalized_global_card_key(card_b),
                "fact_card_key_a": canonical_json_dumps(list(_fact_card_key(card_a))),
                "fact_card_key_b": canonical_json_dumps(list(_fact_card_key(card_b))),
                "card_a": card_a,
                "card_b": card_b,
                "source_metadata": meta,
                "duplicate_sources": [],
            }
            by_pair_content[pair_hash] = len(candidates)
            candidates.append(row)
    if raw_count != sum(EXPECTED_SOURCE_COUNTS.values()):
        raise FilterError(f"unexpected raw candidate count: {raw_count}")
    if len(candidates) != EXPECTED_DEDUPLICATED_CANDIDATES:
        raise FilterError(
            f"unexpected exact deduplicated universe: {len(candidates)} != "
            f"{EXPECTED_DEDUPLICATED_CANDIDATES}"
        )
    duplicate_rows = raw_count - len(candidates)
    if duplicate_rows != 181:
        raise FilterError(f"unexpected duplicate source rows: {duplicate_rows}")
    report = {
        "version": PAIR_DEDUP_VERSION,
        "raw_candidates": raw_count,
        "unique_candidates": len(candidates),
        "duplicate_source_rows": duplicate_rows,
        "duplicate_tier_pairs": {f"{a}->{b}": count for (a, b), count in sorted(duplicates.items())},
        "candidate_universe_sha256": sha256_json(
            [
                [row["candidate_key"], row["pair_content_sha256"]]
                for row in candidates
            ]
        ),
    }
    return candidates, report


def _train_item_ids(validation_dir: Path) -> set[int]:
    pairs = pd.read_parquet(validation_dir / "train_pairs.parquet", columns=["id1", "id2"])
    result = set(pd.to_numeric(pairs["id1"], errors="raise").astype("int64"))
    result.update(pd.to_numeric(pairs["id2"], errors="raise").astype("int64"))
    return result


def _ground_candidates(
    candidates: list[dict[str, Any]], human_items_path: Path, validation_dir: Path
) -> dict[str, Any]:
    required_keys_by_category: dict[str, set[str]] = defaultdict(set)
    scopes_by_category: dict[str, set[str]] = defaultdict(set)
    for row in candidates:
        required_keys_by_category[str(row["category"])].add(
            str(row["required_attribute_key_normalized"])
        )
        scopes_by_category[str(row["category"])].add(str(row["product_type_normalized"]))
    train_ids = _train_item_ids(validation_dir)
    human_items = pd.read_parquet(human_items_path)
    human_items = human_items.loc[human_items["id"].astype("int64").isin(train_ids)]
    category_values: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    scope_values: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    category_brands: dict[str, Counter[str]] = defaultdict(Counter)
    for item in human_items.itertuples(index=False):
        category = str(item.category)
        needed_keys = required_keys_by_category.get(category)
        if not needed_keys:
            continue
        attributes = parse_attributes(item.attributes)
        product_types = {
            normalize_text(value)
            for key, value in attributes.items()
            if normalized_key(key) in TYPE_KEYS and normalize_text(value)
        }
        for raw_key, raw_value in attributes.items():
            key = normalized_key(raw_key)
            value = normalize_text(raw_value)
            if not value:
                continue
            if key in BRAND_KEYS:
                category_brands[category][value] += 1
            if key not in needed_keys:
                continue
            category_values[(category, key)][value] += 1
            for product_type in product_types & scopes_by_category[category]:
                scope_values[(category, product_type, key)][value] += 1

    for row in candidates:
        category = str(row["category"])
        key = str(row["required_attribute_key_normalized"])
        product_type = str(row["product_type_normalized"])
        old_value = str(row["original_value_normalized"])
        new_value = str(row["new_value_normalized"])
        cat_counts = category_values[(category, key)]
        scope_counts = scope_values[(category, product_type, key)]
        row["human_category_original_support"] = int(cat_counts[old_value])
        row["human_category_new_support"] = int(cat_counts[new_value])
        row["human_scope_original_support"] = int(scope_counts[old_value])
        row["human_scope_new_support"] = int(scope_counts[new_value])
        row["category_values_grounded"] = bool(cat_counts[old_value] and cat_counts[new_value])
        row["scope_values_grounded"] = bool(scope_counts[old_value] and scope_counts[new_value])
        brands = category_brands[category]
        brand_values = [normalize_text(row.get("brand_a")), normalize_text(row.get("brand_b"))]
        nonempty_brands = [value for value in brand_values if value]
        row["brand_a_human_support"] = int(brands[brand_values[0]]) if brand_values[0] else 0
        row["brand_b_human_support"] = int(brands[brand_values[1]]) if brand_values[1] else 0
        row["brand_context_grounded"] = bool(
            nonempty_brands and all(brands[value] > 0 for value in nonempty_brands)
        )
        if bool(row["source_exact_transition"]):
            evidence_type = "source_exact_transition"
            evidence_source = "rule_source_examples"
            evidence_value = row["transition_key"]
        elif bool(row["source_endpoint_values"]):
            evidence_type = "source_endpoint_values"
            evidence_source = "rule_source_examples"
            evidence_value = json_array([old_value, new_value])
        elif bool(row["scope_values_grounded"]):
            evidence_type = "human_scope_attribute_values"
            evidence_source = "component_disjoint_human_train_items"
            evidence_value = json_array(
                [row["human_scope_original_support"], row["human_scope_new_support"]]
            )
        else:
            evidence_type = "human_category_attribute_values"
            evidence_source = "component_disjoint_human_train_items"
            evidence_value = json_array(
                [row["human_category_original_support"], row["human_category_new_support"]]
            )
        row["evidence_type"] = evidence_type
        row["evidence_source"] = evidence_source
        row["evidence_value"] = str(evidence_value)
    return {
        "version": HUMAN_GROUNDING_VERSION,
        "train_pair_count": int(len(pd.read_parquet(validation_dir / "train_pairs.parquet", columns=["id1"]))),
        "train_item_count": int(len(train_ids)),
        "category_attribute_domains": len(category_values),
        "scope_attribute_domains": len(scope_values),
    }


def _load_cross_split_evidence(candidates: list[dict[str, Any]], path: Path) -> None:
    report = pd.read_parquet(path)
    rows_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in report.to_dict("records"):
        rows_by_id[str(row.get("generation_rule_id", ""))].append(row)
    for candidate in candidates:
        matches = rows_by_id.get(str(candidate["source_rule_id"]), [])
        if not matches:
            candidate["cross_split_p80"] = None
            candidate["discovery_weighted_support"] = float("nan")
            candidate["validation_weighted_support"] = float("nan")
            candidate["discovery_target_probability"] = float("nan")
            candidate["validation_target_probability"] = float("nan")
            continue
        best = max(matches, key=lambda row: _float(row, "weighted_evidence_support"))
        cross_value = best.get("cross_split_p80")
        candidate["cross_split_p80"] = (
            bool(cross_value) if isinstance(cross_value, (bool, np.bool_)) else None
        )
        for key in (
            "discovery_weighted_support",
            "validation_weighted_support",
            "discovery_target_probability",
            "validation_target_probability",
        ):
            candidate[key] = _float(best, key, float("nan"))


def _load_validation_facts(validation_dir: Path) -> tuple[set[tuple[str, ...]], dict[str, Any]]:
    items_path = validation_dir / "items.parquet"
    split_paths = {
        split: validation_dir / f"{split}_validation_pairs.parquet"
        for split in ("iid", "hard", "ood")
    }
    all_ids: set[int] = set()
    split_counts: dict[str, int] = {}
    split_ids: dict[str, set[int]] = {}
    for split, path in split_paths.items():
        pairs = pd.read_parquet(path, columns=["id1", "id2"])
        ids = set(pairs["id1"].astype("int64")) | set(pairs["id2"].astype("int64"))
        split_ids[split] = ids
        split_counts[split] = len(pairs)
        all_ids.update(ids)
    items = pd.read_parquet(items_path, columns=["id", "product_text"])
    selected = items.loc[items["id"].astype("int64").isin(all_ids)]
    facts = {fact_tokens(value) for value in selected["product_text"]}
    if () in facts:
        raise FilterError("frozen validation contains an empty fact key")
    reference = {
        "version": FACT_KEY_VERSION,
        "paths": {
            "items": {"path": str(items_path), "sha256": sha256_file(items_path)},
            **{
                split: {"path": str(path), "sha256": sha256_file(path)}
                for split, path in split_paths.items()
            },
        },
        "split_pair_counts": split_counts,
        "unique_validation_item_ids": len(all_ids),
        "unique_fact_keys": len(facts),
    }
    return facts, reference


def _mark_validation_overlap(
    candidates: list[dict[str, Any]], validation_facts: set[tuple[str, ...]]
) -> int:
    count = 0
    for row in candidates:
        overlap = (
            tuple(json.loads(row["fact_card_key_a"])) in validation_facts
            or tuple(json.loads(row["fact_card_key_b"])) in validation_facts
        )
        row["validation_fact_overlap"] = bool(overlap)
        count += bool(overlap)
    return count


def _score_candidates(
    candidates: list[dict[str, Any]], model_path: Path, batch_size: int
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise FilterError("torch and transformers are required for baseline scoring") from error
    if batch_size <= 0:
        raise FilterError("batch size must be positive")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()

    def product_text(card: Mapping[str, Any]) -> str:
        return serialize_product(
            pd.Series(
                {
                    "category": card["category"],
                    "name": card["name"],
                    "attributes": json.dumps(
                        _attributes_for_card(card), ensure_ascii=False, separators=(",", ":")
                    ),
                }
            ),
            max_attribute_chars=6000,
        )

    text_a = [product_text(row["card_a"]) for row in candidates]
    text_b = [product_text(row["card_b"]) for row in candidates]

    def direction(first: Sequence[str], second: Sequence[str]) -> list[float]:
        scores: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(first), batch_size):
                encoded = tokenizer(
                    list(first[start : start + batch_size]),
                    list(second[start : start + batch_size]),
                    padding=True,
                    truncation="longest_first",
                    max_length=384,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                logits = model(**encoded).logits
                if logits.ndim == 2 and logits.shape[1] == 1:
                    logits = logits[:, 0]
                elif logits.ndim == 2 and logits.shape[1] == 2:
                    logits = logits[:, 1] - logits[:, 0]
                else:
                    logits = logits.reshape(-1)
                scores.extend(torch.sigmoid(logits).detach().cpu().float().tolist())
                if (start // batch_size + 1) % 25 == 0:
                    print(f"baseline-scoring {start + len(logits)}/{len(first)}", flush=True)
        return scores

    scores_ab = direction(text_a, text_b)
    scores_ba = direction(text_b, text_a)
    for row, score_ab, score_ba in zip(candidates, scores_ab, scores_ba):
        row["score_ab"] = float(score_ab)
        row["score_ba"] = float(score_ba)
        row["baseline_score"] = (float(score_ab) + float(score_ba)) / 2.0
        row["score_order_gap"] = abs(float(score_ab) - float(score_ba))
    score_matrix = np.asarray(
        [
            [row["score_ab"], row["score_ba"], row["baseline_score"], row["score_order_gap"]]
            for row in candidates
        ],
        dtype="<f8",
    )
    return {
        "version": BASELINE_SCORER_VERSION,
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path / "model.safetensors"),
        "config_sha256": sha256_file(model_path / "config.json"),
        "tokenizer_sha256": sha256_file(model_path / "tokenizer.json"),
        "max_length": 384,
        "truncation": "longest_first",
        "direction_aggregation": "mean_sigmoid_ab_ba",
        "device": str(device),
        "batch_size": int(batch_size),
        "score_matrix_columns": ["score_ab", "score_ba", "baseline_score", "score_order_gap"],
        "score_matrix_sha256": hashlib.sha256(score_matrix.tobytes(order="C")).hexdigest(),
    }


def _load_baseline_scores(
    candidates: list[dict[str, Any]], path: Path, model_path: Path
) -> dict[str, Any]:
    scores = pd.read_parquet(path)
    required = {"tier", "task_index", "score_ab", "score_ba", "baseline_score", "score_order_gap"}
    if missing := required - set(scores.columns):
        raise FilterError(f"baseline score cache missing columns: {sorted(missing)}")
    score_map: dict[tuple[str, int], dict[str, Any]] = {}
    for row in scores.to_dict("records"):
        key = (str(row["tier"]), int(row["task_index"]))
        if key in score_map:
            raise FilterError(f"duplicate baseline score key: {key}")
        score_map[key] = row
    for candidate in candidates:
        key = (str(candidate["source_tier"]), int(candidate["source_task_index"]))
        if key not in score_map:
            raise FilterError(f"baseline score cache lacks {key}")
        cached = score_map[key]
        for field in ("score_ab", "score_ba", "baseline_score", "score_order_gap"):
            value = float(cached[field])
            if not math.isfinite(value):
                raise FilterError(f"nonfinite cached {field} for {key}")
            candidate[field] = value
    score_matrix = np.asarray(
        [
            [row["score_ab"], row["score_ba"], row["baseline_score"], row["score_order_gap"]]
            for row in candidates
        ],
        dtype="<f8",
    )
    return {
        "version": BASELINE_SCORER_VERSION,
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path / "model.safetensors"),
        "config_sha256": sha256_file(model_path / "config.json"),
        "tokenizer_sha256": sha256_file(model_path / "tokenizer.json"),
        "max_length": 384,
        "truncation": "longest_first",
        "direction_aggregation": "mean_sigmoid_ab_ba",
        "score_cache_path": str(path),
        "score_cache_sha256": sha256_file(path),
        "score_matrix_columns": ["score_ab", "score_ba", "baseline_score", "score_order_gap"],
        "score_matrix_sha256": hashlib.sha256(score_matrix.tobytes(order="C")).hexdigest(),
    }


def _assign_weights(selected: list[dict[str, Any]]) -> None:
    category_counts = Counter(str(row["category"]) for row in selected)
    median_count = float(np.median(list(category_counts.values()))) if category_counts else 1.0
    evidence_bases = {
        "source_exact_transition": 2.00,
        "source_endpoint_values": 1.25,
        "human_scope_attribute_values": 1.25,
        "human_category_attribute_values": 0.30,
    }
    for rank, row in enumerate(selected, start=1):
        quality = _float(row, "quality_score")
        quality_factor = float(np.clip(0.75 + 0.50 * (quality - 0.48) / 0.37, 0.75, 1.25))
        category_factor = float(
            np.clip(math.sqrt(median_count / category_counts[str(row["category"])]), 0.80, 1.15)
        )
        base = evidence_bases[str(row["evidence_type"])]
        row["sample_weight"] = float(np.clip(base * quality_factor * category_factor, 0.25, 2.50))
        row["selection_rank"] = rank
        row["weight_base"] = base
        row["weight_quality_factor"] = quality_factor
        row["weight_category_factor"] = category_factor


def _selected_output_frames(
    selected: list[dict[str, Any]], id_start: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    items: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    if id_start > -1 or id_start - 2 * len(selected) < -(2**63):
        raise FilterError("synthetic ID range must be negative signed int64")
    for offset, row in enumerate(selected):
        id1 = int(id_start - 2 * offset)
        id2 = int(id_start - 2 * offset - 1)
        card_a, card_b = dict(row["card_a"]), dict(row["card_b"])
        items.extend(
            [
                {"id": id1, "name": card_a["name"], "attributes": card_a["attributes"], "category": card_a["category"]},
                {"id": id2, "name": card_b["name"], "attributes": card_b["attributes"], "category": card_b["category"]},
            ]
        )
        pairs.append(
            {
                "id1": id1,
                "id2": id2,
                "target": 1,
                "label_source": LABEL_SOURCE,
                "sample_weight": float(row["sample_weight"]),
            }
        )
        source_meta = dict(row["source_metadata"])
        source_meta.update(
            {
                "id1": id1,
                "id2": id2,
                "target": 1,
                "candidate_key": row["candidate_key"],
                "source_tier": row["source_tier"],
                "source_run_signature": row["source_run_signature"],
                "source_rule_id": row["source_rule_id"],
                "source_task_index": int(row["source_task_index"]),
                "source_id1": int(row["source_id1"]),
                "source_id2": int(row["source_id2"]),
                "duplicate_source_keys_json": json_array(
                    source["candidate_key"] for source in row["duplicate_sources"]
                ),
                "pair_content_sha256": row["pair_content_sha256"],
                "rule_id": row["rule_id"],
                "concept": row["concept"],
                "relation": row["relation"],
                "product_type": row["product_type"],
                "required_attribute_key": row["required_attribute_key"],
                "original_value": row["original_value"],
                "new_value": row["new_value"],
                "rule_probability": float(row["rule_probability"]),
                "rule_support": int(row["rule_support"]),
                "rule_singleton_support": int(row["rule_singleton_support"]),
                "cross_split_p80": row["cross_split_p80"],
                "human_category_original_support": int(row["human_category_original_support"]),
                "human_category_new_support": int(row["human_category_new_support"]),
                "human_scope_original_support": int(row["human_scope_original_support"]),
                "human_scope_new_support": int(row["human_scope_new_support"]),
                "evidence_type": row["evidence_type"],
                "evidence_source": row["evidence_source"],
                "evidence_value": row["evidence_value"],
                "score_ab": float(row["score_ab"]),
                "score_ba": float(row["score_ba"]),
                "baseline_score": float(row["baseline_score"]),
                "score_order_gap": float(row["score_order_gap"]),
                "quality_score": float(row["quality_score"]),
                "sample_weight": float(row["sample_weight"]),
                "weight_base": float(row["weight_base"]),
                "weight_quality_factor": float(row["weight_quality_factor"]),
                "weight_category_factor": float(row["weight_category_factor"]),
                "selection_rank": int(row["selection_rank"]),
                "filter_contract_version": CONTRACT_VERSION,
                "label_source": LABEL_SOURCE,
                "id_reindex_version": ID_REINDEX_VERSION,
            }
        )
        metadata.append(source_meta)
    return pd.DataFrame(items), pd.DataFrame(pairs), pd.DataFrame(metadata)


def _candidate_decisions_frame(
    candidates: list[dict[str, Any]], rejection_map: Mapping[str, list[str]]
) -> pd.DataFrame:
    # ``select_candidates`` intentionally copies input rows before mutating
    # rank/weight fields.  Recover those selected copies through the explicit
    # fields propagated back by the caller rather than emitting zeroed audit
    # values for selected candidates.
    records: list[dict[str, Any]] = []
    for row in candidates:
        key = str(row["candidate_key"])
        selected = not rejection_map[key]
        records.append(
            {
                "candidate_key": key,
                "source_tier": row["source_tier"],
                "source_run_signature": row["source_run_signature"],
                "source_task_index": int(row["source_task_index"]),
                "source_id1": int(row["source_id1"]),
                "source_id2": int(row["source_id2"]),
                "duplicate_source_keys_json": json_array(
                    source["candidate_key"] for source in row["duplicate_sources"]
                ),
                "pair_content_sha256": row["pair_content_sha256"],
                "category": row["category"],
                "product_type": row["product_type"],
                "rule_id": row["rule_id"],
                "source_rule_id": row["source_rule_id"],
                "semantic_signature": row.get("semantic_signature") or row["candidate_key"],
                "concept": row["concept"],
                "relation": row["relation"],
                "required_attribute_key": row["required_attribute_key"],
                "original_value": row["original_value"],
                "new_value": row["new_value"],
                "rule_probability": float(row["rule_probability"]),
                "rule_support": int(row["rule_support"]),
                "rule_singleton_support": int(row["rule_singleton_support"]),
                "cross_split_p80": row["cross_split_p80"],
                "human_category_original_support": int(row["human_category_original_support"]),
                "human_category_new_support": int(row["human_category_new_support"]),
                "human_scope_original_support": int(row["human_scope_original_support"]),
                "human_scope_new_support": int(row["human_scope_new_support"]),
                "evidence_type": row["evidence_type"],
                "evidence_source": row["evidence_source"],
                "evidence_value": row["evidence_value"],
                "minimum_title_tokens": int(row["minimum_title_tokens"]),
                "name_similarity": float(row["name_similarity"]),
                "score_ab": float(row["score_ab"]),
                "score_ba": float(row["score_ba"]),
                "baseline_score": float(row["baseline_score"]),
                "score_order_gap": float(row["score_order_gap"]),
                "quality_score": float(row["quality_score"]),
                "selected": bool(selected),
                "selection_rank": int(row.get("selection_rank", 0)) if selected else 0,
                "sample_weight": float(row.get("sample_weight", 0.0)) if selected else 0.0,
                "rejection_reasons": json_array(rejection_map[key]),
            }
        )
    return pd.DataFrame(records)


def _file_record(path: Path, rows: int | None = None) -> dict[str, Any]:
    del rows  # row counts are validated from the Parquet payload, not trusted here.
    return {
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _json_common(
    pair_count: int,
    candidate_count: int,
    weight_stats: Mapping[str, Any],
    run_signature: str,
) -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "label_source": LABEL_SOURCE,
        "pair_count": int(pair_count),
        "candidate_count": int(candidate_count),
        "selected_count": int(pair_count),
        "rejected_count": int(candidate_count - pair_count),
        "target_counts": {"0": 0, "1": int(pair_count)},
        "sample_weight_stats": dict(weight_stats),
        "validation_fact_overlap_count": 0,
        "forbidden_ood_categories": [],
        "run_signature": run_signature,
    }


def _summary_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(frame[column].value_counts().items())}


def build_artifact(args: argparse.Namespace) -> dict[str, Any]:
    source_a, source_b = _load_source(args.source_a, "A"), _load_source(args.source_b, "B")
    catalog_a, catalog_b = _load_catalog(args.catalog_a, "A"), _load_catalog(args.catalog_b, "B")
    catalog = {**catalog_a, **catalog_b}
    if len(catalog) != len(catalog_a) + len(catalog_b):
        raise FilterError("A and B catalogs contain a duplicate generation_rule_id")
    candidates, dedup_report = _source_candidates([source_a, source_b], catalog)
    human_items = absolute(args.human_items)
    validation_dir = absolute(args.validation_dir)
    grounding_report = _ground_candidates(candidates, human_items, validation_dir)
    _load_cross_split_evidence(candidates, absolute(args.semantic_report))
    validation_facts, validation_reference = _load_validation_facts(validation_dir)
    raw_validation_overlap = _mark_validation_overlap(candidates, validation_facts)
    baseline_model = absolute(args.baseline_model)
    if args.baseline_scores:
        baseline_report = _load_baseline_scores(candidates, absolute(args.baseline_scores), baseline_model)
    else:
        baseline_report = _score_candidates(candidates, baseline_model, int(args.batch_size))
    for row in candidates:
        row["forbidden_ood_category"] = str(row["category"]) in FROZEN_OOD_CATEGORIES
        row["quality_score"] = score_candidate(row, DEFAULT_POLICY)
    selected, rejection_map = select_candidates(candidates, DEFAULT_POLICY)
    _assign_weights(selected)
    selected_audit = {
        str(row["candidate_key"]): {
            field: row[field]
            for field in (
                "selection_rank",
                "sample_weight",
                "weight_base",
                "weight_quality_factor",
                "weight_category_factor",
            )
        }
        for row in selected
    }
    for candidate in candidates:
        audit = selected_audit.get(str(candidate["candidate_key"]))
        if audit is not None:
            candidate.update(audit)
    selected_keys = {str(row["candidate_key"]) for row in selected}
    if not selected or len(selected_keys) != len(selected):
        raise FilterError("selection is empty or contains duplicate candidate keys")
    items, pairs, metadata = _selected_output_frames(selected, int(args.id_start))
    decisions = _candidate_decisions_frame(candidates, rejection_map)
    weight_stats = exact_weight_stats(pairs["sample_weight"])

    input_records = {
        "source_a": {
            "path": str(source_a.path),
            "run_signature": source_a.run_signature,
            **{
                name: sha256_file(source_a.path / name)
                for name in ("items.parquet", "pairs.parquet", "pair_generation_metadata.parquet", "summary.json", "validation_report.json")
            },
        },
        "source_b": {
            "path": str(source_b.path),
            "run_signature": source_b.run_signature,
            **{
                name: sha256_file(source_b.path / name)
                for name in ("items.parquet", "pairs.parquet", "pair_generation_metadata.parquet", "summary.json", "validation_report.json")
            },
        },
        "catalog_a": {"path": str(absolute(args.catalog_a)), "sha256": sha256_file(absolute(args.catalog_a))},
        "catalog_b": {"path": str(absolute(args.catalog_b)), "sha256": sha256_file(absolute(args.catalog_b))},
        "semantic_report": {"path": str(absolute(args.semantic_report)), "sha256": sha256_file(absolute(args.semantic_report))},
        "human_items": {"path": str(human_items), "sha256": sha256_file(human_items)},
        "human_train_pairs": {
            "path": str(validation_dir / "train_pairs.parquet"),
            "sha256": sha256_file(validation_dir / "train_pairs.parquet"),
        },
    }
    signature_payload = {
        "contract_version": CONTRACT_VERSION,
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "label_source": LABEL_SOURCE,
        "pair_dedup_version": PAIR_DEDUP_VERSION,
        "human_grounding_version": HUMAN_GROUNDING_VERSION,
        "baseline_scorer": {
            key: value
            for key, value in baseline_report.items()
            if key not in {"device", "batch_size", "score_cache_path", "score_cache_sha256"}
        },
        "selection_policy": DEFAULT_POLICY,
        "id_reindex_version": ID_REINDEX_VERSION,
        "id_start": int(args.id_start),
        "inputs": input_records,
        "validation_reference": validation_reference,
        "selected_candidate_keys_sha256": sha256_json([row["candidate_key"] for row in selected]),
        "sample_weight_stats": weight_stats,
    }
    run_signature = sha256_json(signature_payload)
    common = _json_common(len(pairs), len(candidates), weight_stats, run_signature)
    rejection_counts: Counter[str] = Counter(
        reason for reasons in rejection_map.values() for reason in reasons
    )
    selected_frame = pd.DataFrame(selected)
    summary = {
        **common,
        "version": CONTRACT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "selected_pairs": len(selected),
        "rejected_pairs": len(candidates) - len(selected),
        "generated_pairs": len(pairs),
        "generated_items": len(items),
        "metadata_rows": len(metadata),
        "selection_fraction": len(selected) / len(candidates),
        "source_target_counts": {"A": EXPECTED_SOURCE_COUNTS["A"], "B": EXPECTED_SOURCE_COUNTS["B"]},
        "selected_tier_counts": _summary_counts(selected_frame, "source_tier"),
        "selected_category_counts": _summary_counts(selected_frame, "category"),
        "selected_evidence_counts": _summary_counts(selected_frame, "evidence_type"),
        "selected_unique_rules": int(selected_frame["rule_id"].nunique()),
        "selected_unique_product_types": int(selected_frame[["category", "product_type_normalized"]].drop_duplicates().shape[0]),
        "baseline_score_summary": {
            "min": float(selected_frame["baseline_score"].min()),
            "median": float(selected_frame["baseline_score"].median()),
            "mean": float(selected_frame["baseline_score"].mean()),
            "max": float(selected_frame["baseline_score"].max()),
        },
        "quality_score_summary": {
            "min": float(selected_frame["quality_score"].min()),
            "median": float(selected_frame["quality_score"].median()),
            "mean": float(selected_frame["quality_score"].mean()),
            "max": float(selected_frame["quality_score"].max()),
        },
        "deduplication": dedup_report,
        "grounding": grounding_report,
        "baseline_scorer": baseline_report,
        "selection_policy": DEFAULT_POLICY,
        "input_provenance": input_records,
        "validation_reference": validation_reference,
        "raw_validation_fact_overlap_count": int(raw_validation_overlap),
        "signature_payload": signature_payload,
    }
    selection_report = {
        **common,
        "valid": True,
        "candidate_pairs": len(candidates),
        "selected_pairs": len(selected),
        "rejected_pairs": len(candidates) - len(selected),
        "deduplication": dedup_report,
        "policy": DEFAULT_POLICY,
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "selected_tier_counts": _summary_counts(selected_frame, "source_tier"),
        "selected_category_counts": _summary_counts(selected_frame, "category"),
        "selected_evidence_counts": _summary_counts(selected_frame, "evidence_type"),
        "selected_candidate_keys_sha256": sha256_json([row["candidate_key"] for row in selected]),
    }
    validation_report = {
        **common,
        "valid": True,
        "pairs": len(pairs),
        "items": len(items),
        "metadata_rows": len(metadata),
        "candidate_decision_rows": len(decisions),
        "raw_candidate_validation_fact_overlap_count": int(raw_validation_overlap),
        "one_use_unique_cards": True,
        "unique_item_ids": int(items["id"].nunique()),
        "unique_pair_ids": int(pairs[["id1", "id2"]].drop_duplicates().shape[0]),
        "unique_global_cards": int(
            pd.Series([canonical_card(row) for _, row in items.iterrows()]).nunique()
        ),
        "validation_reference": validation_reference,
    }
    if validation_report["forbidden_ood_categories"]:
        raise FilterError("selected output contains a frozen OOD category")

    output_dir = absolute(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_parquet(items, output_dir / "items.parquet")
    atomic_parquet(pairs, output_dir / "pairs.parquet")
    atomic_parquet(metadata, output_dir / "pair_generation_metadata.parquet")
    atomic_parquet(decisions, output_dir / "candidate_decisions.parquet")
    atomic_json(summary, output_dir / "summary.json")
    atomic_json(selection_report, output_dir / "selection_report.json")
    atomic_json(validation_report, output_dir / "validation_report.json")
    file_rows = {
        "items.parquet": len(items),
        "pairs.parquet": len(pairs),
        "pair_generation_metadata.parquet": len(metadata),
        "candidate_decisions.parquet": len(decisions),
        "summary.json": None,
        "selection_report.json": None,
        "validation_report.json": None,
    }
    manifest = {
        **common,
        "version": CONTRACT_VERSION,
        "valid": True,
        "files": {
            name: _file_record(output_dir / name, rows)
            for name, rows in file_rows.items()
        },
        "input_provenance": input_records,
        "validation_reference": validation_reference,
        "signature_payload": signature_payload,
    }
    atomic_json(manifest, output_dir / "build_manifest.json")
    return validate_artifact(output_dir)


def _require_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FilterError(f"invalid JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise FilterError(f"{path} must contain a JSON object")
    return value


def validate_artifact(output_dir: Path, manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    output_dir = absolute(output_dir)
    manifest_doc = dict(manifest) if manifest is not None else _require_json(output_dir / "build_manifest.json")
    documents = {
        "summary": _require_json(output_dir / "summary.json"),
        "selection": _require_json(output_dir / "selection_report.json"),
        "validation": _require_json(output_dir / "validation_report.json"),
        "manifest": manifest_doc,
    }
    signatures = {str(doc.get("run_signature", "")) for doc in documents.values()}
    if len(signatures) != 1 or not next(iter(signatures)):
        raise FilterError("run_signature is missing or inconsistent across provenance")
    signature = next(iter(signatures))
    signature_payload = manifest_doc.get("signature_payload")
    if not isinstance(signature_payload, dict) or sha256_json(signature_payload) != signature:
        raise FilterError("build manifest signature_payload does not reproduce run_signature")
    if documents["summary"].get("signature_payload") != signature_payload:
        raise FilterError("summary and build manifest signature_payload differ")
    for label, doc in documents.items():
        if doc.get("contract_version") != CONTRACT_VERSION:
            raise FilterError(f"wrong contract_version in {label}")
        if doc.get("label_source") != LABEL_SOURCE:
            raise FilterError(f"wrong label_source in {label}")
    files = manifest_doc.get("files")
    if not isinstance(files, dict):
        raise FilterError("build manifest files must be an object")
    required_files = {
        "items.parquet",
        "pairs.parquet",
        "pair_generation_metadata.parquet",
        "candidate_decisions.parquet",
        "summary.json",
        "selection_report.json",
        "validation_report.json",
    }
    if set(files) != required_files:
        raise FilterError(f"build manifest files mismatch: {sorted(set(files) ^ required_files)}")
    for name, record in files.items():
        path = output_dir / name
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise FilterError(f"manifest hash mismatch for {name}")
        if path.stat().st_size != int(record.get("bytes", -1)):
            raise FilterError(f"manifest byte size mismatch for {name}")
    items = pd.read_parquet(output_dir / "items.parquet")
    pairs = pd.read_parquet(output_dir / "pairs.parquet")
    metadata = pd.read_parquet(output_dir / "pair_generation_metadata.parquet")
    decisions = pd.read_parquet(output_dir / "candidate_decisions.parquet")
    pair_count = len(pairs)
    if not pair_count or len(items) != 2 * pair_count or len(metadata) != pair_count:
        raise FilterError("items/pairs/metadata counts are inconsistent")
    if len(decisions) != EXPECTED_DEDUPLICATED_CANDIDATES:
        raise FilterError("candidate decision universe count mismatch")
    for label, doc in documents.items():
        if int(doc.get("pair_count", -1)) != pair_count:
            raise FilterError(f"pair_count mismatch in {label}")
        if doc.get("target_counts") != {"0": 0, "1": pair_count}:
            raise FilterError(f"target_counts mismatch in {label}")
    if set(pairs["target"].astype(int)) != {1} or set(pairs["label_source"].astype(str)) != {LABEL_SOURCE}:
        raise FilterError("pairs target/label source contract failed")
    if not np.isfinite(pairs["sample_weight"].astype(float)).all() or (pairs["sample_weight"].astype(float) <= 0).any():
        raise FilterError("pairs contain invalid sample weights")
    if not np.array_equal(
        pairs["sample_weight"].astype("float64").to_numpy(),
        metadata["sample_weight"].astype("float64").to_numpy(),
    ):
        raise FilterError("pairs and metadata sample weights differ")
    weight_stats = exact_weight_stats(pairs["sample_weight"])
    for label, doc in documents.items():
        if doc.get("sample_weight_stats") != weight_stats:
            raise FilterError(f"sample_weight_stats mismatch in {label}")
    item_ids = pd.to_numeric(items["id"], errors="raise").astype("int64")
    if item_ids.duplicated().any() or len(set(pairs["id1"].astype(int)) | set(pairs["id2"].astype(int))) != 2 * pair_count:
        raise FilterError("synthetic endpoints are not pair-local unique IDs")
    if list(zip(pairs["id1"].astype(int), pairs["id2"].astype(int))) != list(
        zip(metadata["id1"].astype(int), metadata["id2"].astype(int))
    ):
        raise FilterError("pairs and metadata are not aligned")
    available_ids = set(item_ids.astype(int))
    if set(pairs["id1"].astype(int)) | set(pairs["id2"].astype(int)) != available_ids:
        raise FilterError("pair IDs and item IDs differ")
    global_keys = [canonical_card(row) for _, row in items.iterrows()]
    if len(set(global_keys)) != len(global_keys):
        raise FilterError("output contains duplicate category-agnostic cards")
    if decisions["candidate_key"].astype(str).duplicated().any():
        raise FilterError("candidate decisions contain duplicate keys")
    if decisions["selected"].dtype != bool:
        raise FilterError("candidate decisions selected must be literal bool")
    selected_decisions = decisions.loc[decisions["selected"]]
    if len(selected_decisions) != pair_count:
        raise FilterError("selected decision count differs from pairs")
    selected_keys = set(selected_decisions["candidate_key"].astype(str))
    metadata_keys = set(metadata["candidate_key"].astype(str))
    if selected_keys != metadata_keys or len(metadata_keys) != pair_count:
        raise FilterError("selected decision keys and metadata keys differ")
    for row in decisions.itertuples(index=False):
        try:
            reasons = json.loads(str(row.rejection_reasons))
        except json.JSONDecodeError as error:
            raise FilterError("candidate rejection_reasons is not canonical JSON") from error
        if not isinstance(reasons, list) or str(row.rejection_reasons) != json_array(reasons):
            raise FilterError("candidate rejection_reasons is not a canonical JSON array")
        if bool(row.selected) != (len(reasons) == 0):
            raise FilterError("selected/rejection reasons invariant failed")
        for field in ("baseline_score", "score_ab", "score_ba", "score_order_gap", "quality_score"):
            if not math.isfinite(float(getattr(row, field))):
                raise FilterError(f"candidate {field} contains nonfinite value")
        for field in ("evidence_type", "evidence_source", "evidence_value"):
            if not str(getattr(row, field)).strip():
                raise FilterError(f"candidate {field} is empty")
    if documents["validation"].get("valid") is not True:
        raise FilterError("validation report is not valid")
    if int(documents["validation"].get("validation_fact_overlap_count", -1)) != 0:
        raise FilterError("validation fact overlap report is not zero")
    if documents["validation"].get("forbidden_ood_categories") != []:
        raise FilterError("validation report contains forbidden OOD categories")
    return {
        "valid": True,
        "pair_count": pair_count,
        "item_count": len(items),
        "candidate_count": len(decisions),
        "run_signature": signature,
        "sample_weight_stats": weight_stats,
        "category_counts": _summary_counts(metadata, "category"),
        "evidence_counts": _summary_counts(metadata, "evidence_type"),
        "tier_counts": _summary_counts(metadata, "source_tier"),
    }


def main() -> None:
    args = parse_args()
    args.source_a = absolute(args.source_a)
    args.source_b = absolute(args.source_b)
    args.catalog_a = absolute(args.catalog_a)
    args.catalog_b = absolute(args.catalog_b)
    args.semantic_report = absolute(args.semantic_report)
    args.human_items = absolute(args.human_items)
    args.validation_dir = absolute(args.validation_dir)
    args.baseline_model = absolute(args.baseline_model)
    args.output_dir = absolute(args.output_dir)
    if args.baseline_scores:
        args.baseline_scores = absolute(args.baseline_scores)
    if args.validate_only:
        result = validate_artifact(args.output_dir)
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
            raise FilterError(
                f"output directory is not empty: {args.output_dir}; pass --overwrite"
            )
        result = build_artifact(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
