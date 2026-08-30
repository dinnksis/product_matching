"""Mine label-associated rule candidates from compact atomic differences.

Qwen outputs are label-blind. This script takes an append-only checkpoint
snapshot, joins human labels locally, and computes conservative singleton-first
statistics without making any API calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = (
    ROOT
    / "artifacts"
    / "qwen_atomic_differences_compact_v1_full_train"
    / "raw_responses.jsonl"
)
DEFAULT_INPUTS = (
    ROOT / "data" / "qwen_atomic_differences_v2_full_train" / "pilot_inputs.parquet"
)
DEFAULT_LABELS = (
    ROOT / "data" / "qwen_atomic_differences_v2_full_train" / "pilot_labels.parquet"
)
DEFAULT_OUTPUT = ROOT / "reports" / "atomic_rule_statistics_current"
RELATION_MAP = {
    "different_value": "different_value",
    "a_more_specific": "more_specific",
    "b_more_specific": "more_specific",
    "incompatible": "incompatible",
}
CONCEPT_ALIASES = {
    "base_color": "color",
    "color_description": "color",
    "color_name": "color",
    "frame_color": "color",
    "main_color": "color",
    "model_identifier": "model",
    "model_name": "model",
    "model_number": "model",
    "model_version": "model",
    "product_model": "model",
    "product_model_identifier": "model",
    "package_count": "package_quantity",
    "piece_count": "package_quantity",
    "quantity": "package_quantity",
    "unit_count": "package_quantity",
    "number_of_items": "package_quantity",
    "product_dimensions": "dimensions",
    "item_dimensions": "dimensions",
    "size_ru": "russian_size",
    "ru_size": "russian_size",
    "product_type": "type",
}
THRESHOLD_SUPPORTS = (1, 2, 3, 5, 10, 20, 50)
THRESHOLD_LCBS = (0.80, 0.90, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-responses", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-percent", type=int, default=20)
    parser.add_argument("--examples-per-rule", type=int, default=3)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_bucket(pair_id: str, modulo: int = 100) -> int:
    return int.from_bytes(hashlib.sha256(pair_id.encode()).digest()[:8], "little") % modulo


def normalized_token(value: Any) -> str:
    value = str(value).casefold().replace("ё", "е").replace(",", ".")
    return " ".join(re.findall(r"[0-9a-zа-я]+", value))


def canonical_concept(value: Any) -> str:
    concept = re.sub(r"[^a-z0-9_]+", "_", str(value).casefold()).strip("_")
    concept = CONCEPT_ALIASES.get(concept, concept)
    if concept.endswith("_model_number") or concept.endswith("_model_identifier"):
        concept = concept.removesuffix("_number").removesuffix("_identifier")
    return concept or "unknown"


def normalized_value_pair(fact: dict[str, Any]) -> tuple[str, str]:
    value_a = normalized_token(fact.get("a", {}).get("value"))
    value_b = normalized_token(fact.get("b", {}).get("value"))
    relation = str(fact.get("relation"))
    if relation in {"different_value", "incompatible"} and value_b < value_a:
        value_a, value_b = value_b, value_a
    return value_a, value_b


def read_latest_compact_checkpoint(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    rows = 0
    malformed_lines = 0
    prompt_hashes: Counter[str] = Counter()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue
            pair_id = response.get("pair_id")
            if not pair_id:
                continue
            rows += 1
            prompt_hashes.update([str(response.get("prompt_sha256") or "")])
            extraction = response.get("schema_response") or response.get("parsed_response")
            differences = (
                extraction.get("differences", []) if isinstance(extraction, dict) else []
            )
            latest[str(pair_id)] = {
                "status": str(response.get("status", "unknown")),
                "differences": differences if isinstance(differences, list) else [],
                "completed_at": response.get("completed_at"),
                "prompt_sha256": response.get("prompt_sha256"),
                "schema_sha256": response.get("schema_sha256"),
                "compact_fact_validation": response.get("compact_fact_validation") or {},
            }
    status_counts = Counter(row["status"] for row in latest.values())
    return latest, {
        "checkpoint_rows": rows,
        "unique_pairs": len(latest),
        "duplicate_or_retried_rows": rows - len(latest),
        "malformed_checkpoint_lines": malformed_lines,
        "latest_status_counts": dict(status_counts),
        "prompt_hash_counts": dict(prompt_hashes),
    }


def wilson_bounds(successes: pd.Series, totals: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    z = 1.959963984540054
    success = successes.to_numpy(dtype=float)
    total = totals.to_numpy(dtype=float)
    proportion = np.divide(success, total, out=np.zeros_like(success), where=total > 0)
    denominator = 1.0 + z * z / np.maximum(total, 1.0)
    center = proportion + z * z / (2.0 * np.maximum(total, 1.0))
    spread = z * np.sqrt(
        np.divide(
            proportion * (1.0 - proportion),
            np.maximum(total, 1.0),
        )
        + z * z / (4.0 * np.maximum(total, 1.0) ** 2)
    )
    lower = (center - spread) / denominator
    upper = (center + spread) / denominator
    lower[total == 0] = np.nan
    upper[total == 0] = np.nan
    return lower, upper


def bh_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    if not len(values):
        return values
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def build_atomic_occurrences(
    latest: dict[str, dict[str, Any]],
    context: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for source in context.itertuples(index=False):
        response = latest.get(str(source.pair_id))
        if response is None or response["status"] != "ok":
            continue
        raw_facts = [fact for fact in response["differences"] if isinstance(fact, dict)]
        canonical_facts: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for fact in raw_facts:
            relation = RELATION_MAP.get(str(fact.get("relation")))
            if relation is None:
                continue
            concept = canonical_concept(fact.get("concept"))
            value_a, value_b = normalized_value_pair(fact)
            key = (concept, relation, value_a, value_b)
            if key in seen:
                continue
            seen.add(key)
            canonical_facts.append(
                {
                    "concept": concept,
                    "raw_concept": str(fact.get("concept")),
                    "relation": relation,
                    "value_a": value_a,
                    "value_b": value_b,
                    "raw_value_a": str(fact.get("a", {}).get("value") or ""),
                    "raw_value_b": str(fact.get("b", {}).get("value") or ""),
                    "source_a": str(fact.get("a", {}).get("source") or ""),
                    "source_b": str(fact.get("b", {}).get("source") or ""),
                    "raw_attribute_a": str(
                        fact.get("a", {}).get("attribute") or ""
                    ),
                    "raw_attribute_b": str(
                        fact.get("b", {}).get("attribute") or ""
                    ),
                }
            )
        fact_count = len(canonical_facts)
        split = "validation" if stable_bucket(str(source.pair_id)) < 20 else "discovery"
        pair_rows.append(
            {
                "pair_id": str(source.pair_id),
                "category": str(source.category),
                "human_label": int(source.human_label),
                "split": split,
                "atomic_fact_count": fact_count,
                "is_singleton": fact_count == 1,
            }
        )
        for index, fact in enumerate(canonical_facts):
            rows.append(
                {
                    "pair_id": str(source.pair_id),
                    "category": str(source.category),
                    "human_label": int(source.human_label),
                    "split": split,
                    "atomic_fact_count": fact_count,
                    "is_singleton": fact_count == 1,
                    "fact_index": index,
                    **fact,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def expanded_rule_occurrences(facts: pd.DataFrame) -> pd.DataFrame:
    if not len(facts):
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for level in ("category_concept", "global_concept", "category_exact"):
        frame = facts.copy()
        frame["level"] = level
        frame["scope_category"] = (
            "__ALL__" if level == "global_concept" else frame["category"]
        )
        if level == "category_exact":
            frame["signature"] = [
                json.dumps(
                    [category, concept, relation, value_a, value_b],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                for category, concept, relation, value_a, value_b in zip(
                    frame["category"],
                    frame["concept"],
                    frame["relation"],
                    frame["value_a"],
                    frame["value_b"],
                )
            ]
        else:
            frame["signature"] = [
                json.dumps(
                    [scope, concept, relation],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                for scope, concept, relation in zip(
                    frame["scope_category"], frame["concept"], frame["relation"]
                )
            ]
        frames.append(frame)
    expanded = pd.concat(frames, ignore_index=True)
    return expanded.drop_duplicates(["pair_id", "level", "signature"])


def aggregate_partition(
    occurrences: pd.DataFrame,
    partition: str | None,
    prefix: str,
) -> pd.DataFrame:
    frame = occurrences if partition is None else occurrences[occurrences["split"].eq(partition)]
    keys = ["level", "signature", "scope_category", "concept", "relation"]
    if not len(frame):
        return pd.DataFrame(columns=keys)
    frame = frame.copy()
    frame["label0"] = frame["human_label"].eq(0).astype(int)
    frame["label1"] = frame["human_label"].eq(1).astype(int)
    frame["singleton_label0"] = (frame["is_singleton"] & frame["human_label"].eq(0)).astype(int)
    frame["singleton_label1"] = (frame["is_singleton"] & frame["human_label"].eq(1)).astype(int)
    grouped = (
        frame.groupby(keys, observed=True)
        .agg(
            pair_support=("pair_id", "size"),
            label0=("label0", "sum"),
            label1=("label1", "sum"),
            singleton_support=("is_singleton", "sum"),
            singleton_label0=("singleton_label0", "sum"),
            singleton_label1=("singleton_label1", "sum"),
            example_pair_ids=("pair_id", lambda values: ";".join(map(str, list(values)[:3]))),
            example_value_pairs=(
                "raw_value_a",
                lambda values: " | ".join(map(str, list(values)[:3])),
            ),
        )
        .reset_index()
    )
    singleton_population = frame[frame["is_singleton"]].copy()
    singleton_population["population_label0"] = singleton_population["human_label"].eq(0).astype(int)
    singleton_population["population_label1"] = singleton_population["human_label"].eq(1).astype(int)
    baselines = (
        singleton_population.groupby(["level", "scope_category"], observed=True)
        .agg(
            baseline_singletons=("pair_id", "size"),
            baseline_label0=("population_label0", "sum"),
            baseline_label1=("population_label1", "sum"),
        )
        .reset_index()
    )
    grouped = grouped.merge(baselines, on=["level", "scope_category"], how="left")
    grouped["singleton_p0"] = grouped["singleton_label0"] / grouped["singleton_support"].replace(0, np.nan)
    grouped["singleton_p1"] = grouped["singleton_label1"] / grouped["singleton_support"].replace(0, np.nan)
    grouped["baseline_p0"] = grouped["baseline_label0"] / grouped["baseline_singletons"].replace(0, np.nan)
    grouped["baseline_p1"] = grouped["baseline_label1"] / grouped["baseline_singletons"].replace(0, np.nan)
    lower0, upper0 = wilson_bounds(grouped["singleton_label0"], grouped["singleton_support"])
    lower1, upper1 = wilson_bounds(grouped["singleton_label1"], grouped["singleton_support"])
    grouped["singleton_p0_lcb95"], grouped["singleton_p0_ucb95"] = lower0, upper0
    grouped["singleton_p1_lcb95"], grouped["singleton_p1_ucb95"] = lower1, upper1
    p0_values: list[float] = []
    p1_values: list[float] = []
    for row in grouped.itertuples(index=False):
        other0 = max(0, int(row.baseline_label0) - int(row.singleton_label0))
        other1 = max(0, int(row.baseline_label1) - int(row.singleton_label1))
        table = [
            [int(row.singleton_label0), int(row.singleton_label1)],
            [other0, other1],
        ]
        p0_values.append(float(fisher_exact(table, alternative="greater").pvalue))
        p1_values.append(float(fisher_exact(table, alternative="less").pvalue))
    grouped["p_value_label0"] = p0_values
    grouped["p_value_label1"] = p1_values
    grouped["q_value_label0"] = bh_adjust(p0_values)
    grouped["q_value_label1"] = bh_adjust(p1_values)
    rename = {
        column: f"{prefix}_{column}"
        for column in grouped.columns
        if column not in keys
    }
    return grouped.rename(columns=rename)


def merged_rule_statistics(occurrences: pd.DataFrame) -> pd.DataFrame:
    keys = ["level", "signature", "scope_category", "concept", "relation"]
    combined = aggregate_partition(occurrences, None, "all")
    discovery = aggregate_partition(occurrences, "discovery", "discovery")
    validation = aggregate_partition(occurrences, "validation", "validation")
    statistics = combined.merge(discovery, on=keys, how="left").merge(
        validation, on=keys, how="left"
    )
    count_columns = [
        column
        for column in statistics.columns
        if any(
            token in column
            for token in (
                "support", "label0", "label1", "singletons"
            )
        )
        and not any(token in column for token in ("p0", "p1", "value"))
    ]
    for column in count_columns:
        statistics[column] = statistics[column].fillna(0).astype(int)
    return statistics


def classify_candidates(statistics: pd.DataFrame) -> pd.DataFrame:
    result = statistics.copy()
    result["direct_singleton_candidate"] = result["all_singleton_support"].ge(1)
    result["repeated_singleton_candidate"] = result["all_singleton_support"].ge(5)
    negative_discovery = (
        result["discovery_singleton_support"].ge(10)
        & result["discovery_singleton_p0_lcb95"].ge(0.90)
        & result["discovery_singleton_p0"].sub(result["discovery_baseline_p0"]).ge(0.10)
        & result["discovery_q_value_label0"].le(0.05)
    )
    positive_discovery = (
        result["discovery_singleton_support"].ge(10)
        & result["discovery_singleton_p1_lcb95"].ge(0.90)
        & result["discovery_singleton_p1"].sub(result["discovery_baseline_p1"]).ge(0.10)
        & result["discovery_q_value_label1"].le(0.05)
    )
    negative_validated = (
        negative_discovery
        & result["validation_singleton_support"].ge(3)
        & result["validation_singleton_p0"].ge(0.80)
        & result["validation_singleton_p0"].ge(result["validation_baseline_p0"])
    )
    positive_validated = (
        positive_discovery
        & result["validation_singleton_support"].ge(3)
        & result["validation_singleton_p1"].ge(0.80)
        & result["validation_singleton_p1"].ge(result["validation_baseline_p1"])
    )
    result["discovery_rule_target"] = np.select(
        [negative_discovery, positive_discovery], [0, 1], default=-1
    )
    result["validated_rule_target"] = np.select(
        [negative_validated, positive_validated], [0, 1], default=-1
    )
    result["rule_description_ru"] = [
        (
            f"В категории {category} различие по {concept} ({relation}) "
            + (
                "связано с меткой 0."
                if target == 0
                else "обычно допустимо при метке 1."
                if target == 1
                else "остаётся статистическим кандидатом."
            )
        )
        if category != "__ALL__"
        else (
            f"Глобально различие по {concept} ({relation}) "
            + (
                "связано с меткой 0."
                if target == 0
                else "обычно допустимо при метке 1."
                if target == 1
                else "остаётся статистическим кандидатом."
            )
        )
        for category, concept, relation, target in zip(
            result["scope_category"],
            result["concept"],
            result["relation"],
            result["validated_rule_target"],
        )
    ]
    return result


def threshold_sweep(statistics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for level in sorted(statistics["level"].unique()):
        frame = statistics[statistics["level"].eq(level)]
        for target in (0, 1):
            lcb_column = f"all_singleton_p{target}_lcb95"
            rate_column = f"all_singleton_p{target}"
            validation_rate = f"validation_singleton_p{target}"
            for support in THRESHOLD_SUPPORTS:
                for threshold in THRESHOLD_LCBS:
                    selected = frame[
                        frame["all_singleton_support"].ge(support)
                        & frame[lcb_column].ge(threshold)
                    ]
                    validation_minimum = max(1, math.ceil(support * 0.15))
                    stable = selected[
                        selected["validation_singleton_support"].ge(validation_minimum)
                        & selected[validation_rate].ge(threshold)
                    ]
                    rows.append(
                        {
                            "level": level,
                            "target_label": target,
                            "minimum_singleton_support": support,
                            "minimum_wilson_lcb95": threshold,
                            "selected_rules": len(selected),
                            "validation_consistent_rules": len(stable),
                            "median_selected_probability": float(selected[rate_column].median())
                            if len(selected)
                            else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def provisional_concept_candidates(
    statistics: pd.DataFrame,
    facts: pd.DataFrame,
) -> pd.DataFrame:
    """Select reviewable concept rules without pretending they are final.

    Category rules are the primary unit.  A global rule is retained only when
    the same concept/relation was observed in at least two categories; this
    removes global duplicates such as optical power that are effectively bound
    to one category in the current snapshot.
    """
    concept_rows = statistics[
        statistics["level"].isin(["category_concept", "global_concept"])
    ].copy()
    concept_rows["target_label"] = np.where(
        concept_rows["all_singleton_p0_lcb95"].ge(
            concept_rows["all_singleton_p1_lcb95"]
        ),
        0,
        1,
    )
    concept_rows["target_probability"] = np.where(
        concept_rows["target_label"].eq(0),
        concept_rows["all_singleton_p0"],
        concept_rows["all_singleton_p1"],
    )
    concept_rows["target_wilson_lcb95"] = np.where(
        concept_rows["target_label"].eq(0),
        concept_rows["all_singleton_p0_lcb95"],
        concept_rows["all_singleton_p1_lcb95"],
    )
    concept_rows["validation_target_probability"] = np.where(
        concept_rows["target_label"].eq(0),
        concept_rows["validation_singleton_p0"],
        concept_rows["validation_singleton_p1"],
    )
    coverage = (
        facts.groupby(["concept", "relation"], observed=True)["category"]
        .nunique()
        .rename("observed_category_count")
        .reset_index()
    )
    singleton_category_support = (
        facts[facts["is_singleton"]]
        .groupby(["concept", "relation", "category"], observed=True)
        .size()
        .rename("support")
        .reset_index()
    )
    broad_coverage = (
        singleton_category_support[singleton_category_support["support"].ge(5)]
        .groupby(["concept", "relation"], observed=True)["category"]
        .nunique()
        .rename("categories_with_5_singletons")
        .reset_index()
    )
    concept_rows = concept_rows.merge(coverage, on=["concept", "relation"], how="left")
    concept_rows = concept_rows.merge(
        broad_coverage, on=["concept", "relation"], how="left"
    )
    concept_rows["categories_with_5_singletons"] = (
        concept_rows["categories_with_5_singletons"].fillna(0).astype(int)
    )
    category_or_broad_global = concept_rows["level"].eq("category_concept") | (
        concept_rows["level"].eq("global_concept")
        & concept_rows["categories_with_5_singletons"].ge(2)
    )
    selected = concept_rows[
        category_or_broad_global
        & concept_rows["all_singleton_support"].ge(10)
        & concept_rows["target_wilson_lcb95"].ge(0.80)
        & concept_rows["validation_singleton_support"].ge(3)
        & concept_rows["validation_target_probability"].ge(0.80)
    ].copy()
    selected["candidate_status"] = "provisional_lcb80_holdout_consistent"
    selected["rule_description_ru"] = [
        (
            f"В категории {category} различие по {concept} ({relation}) "
            f"связано с меткой {target}."
        )
        if level == "category_concept"
        else (
            f"В нескольких категориях различие по {concept} ({relation}) "
            f"связано с меткой {target}."
        )
        for level, category, concept, relation, target in zip(
            selected["level"],
            selected["scope_category"],
            selected["concept"],
            selected["relation"],
            selected["target_label"],
        )
    ]
    order = [
        "candidate_status",
        "level",
        "scope_category",
        "concept",
        "relation",
        "target_label",
        "all_pair_support",
        "all_singleton_support",
        "all_singleton_label0",
        "all_singleton_label1",
        "target_probability",
        "target_wilson_lcb95",
        "validation_singleton_support",
        "validation_target_probability",
        "observed_category_count",
        "categories_with_5_singletons",
        "discovery_q_value_label0",
        "discovery_q_value_label1",
        "all_example_pair_ids",
        "all_example_value_pairs",
        "rule_description_ru",
    ]
    return selected[order].sort_values(
        ["target_wilson_lcb95", "all_singleton_support"], ascending=False
    )


def main() -> None:
    args = parse_args()
    if args.validation_percent != 20:
        raise ValueError("This version uses a frozen 80/20 hash split; pass 20")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.raw_responses.resolve()
    inputs_path, labels_path = args.inputs.resolve(), args.labels.resolve()

    latest, checkpoint_summary = read_latest_compact_checkpoint(raw_path)
    inputs = pd.read_parquet(inputs_path, columns=["pair_id", "category"])
    labels = pd.read_parquet(labels_path)
    if set(labels.columns) != {"pair_id", "human_label"}:
        raise ValueError("labels must contain exactly pair_id,human_label")
    context = inputs.merge(labels, on="pair_id", how="inner", validate="one_to_one")
    context = context[context["pair_id"].astype(str).isin(latest)].copy()
    context["human_label"] = context["human_label"].astype(int)

    facts, pairs = build_atomic_occurrences(latest, context)
    if not len(facts):
        raise RuntimeError("No accepted atomic facts found in checkpoint snapshot")
    expanded = expanded_rule_occurrences(facts)
    statistics = classify_candidates(merged_rule_statistics(expanded))
    sweep = threshold_sweep(statistics)
    provisional = provisional_concept_candidates(statistics, facts)

    facts.to_parquet(output_dir / "atomic_occurrences.parquet", index=False)
    pairs.to_parquet(output_dir / "processed_pairs.parquet", index=False)
    statistics.to_parquet(output_dir / "rule_statistics.parquet", index=False)
    statistics.to_csv(
        output_dir / "rule_statistics.csv", index=False, encoding="utf-8-sig"
    )
    singleton_events = facts[facts["is_singleton"]].copy()
    singleton_events.to_parquet(output_dir / "singleton_rule_evidence.parquet", index=False)
    high_confidence = statistics[statistics["validated_rule_target"].isin([0, 1])].copy()
    high_confidence.to_csv(
        output_dir / "validated_high_confidence_rules.csv",
        index=False,
        encoding="utf-8-sig",
    )
    provisional.to_csv(
        output_dir / "provisional_concept_rule_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sweep.to_csv(output_dir / "threshold_sweep.csv", index=False, encoding="utf-8-sig")

    fact_count_distribution = (
        pairs["atomic_fact_count"].value_counts().sort_index().to_dict()
    )
    level_counts = {}
    for level in sorted(statistics["level"].unique()):
        level_frame = statistics[statistics["level"].eq(level)]
        level_counts[level] = {
            "statistical_candidates": len(level_frame),
            "with_singleton_evidence": int(level_frame["direct_singleton_candidate"].sum()),
            "with_at_least_5_singletons": int(level_frame["repeated_singleton_candidate"].sum()),
            "discovery_high_confidence": int(level_frame["discovery_rule_target"].isin([0, 1]).sum()),
            "validated_high_confidence": int(level_frame["validated_rule_target"].isin([0, 1]).sum()),
        }
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "checkpoint": checkpoint_summary,
        "processed_ok_pairs": len(pairs),
        "pairs_with_no_accepted_differences": int(pairs["atomic_fact_count"].eq(0).sum()),
        "pairs_with_one_accepted_difference": int(pairs["atomic_fact_count"].eq(1).sum()),
        "pairs_with_multiple_accepted_differences": int(pairs["atomic_fact_count"].gt(1).sum()),
        "accepted_atomic_occurrences": len(facts),
        "unique_canonical_concepts": int(facts["concept"].nunique()),
        "provisional_concept_rules_lcb80": len(provisional),
        "strict_validated_rule_rows": len(high_confidence),
        "strict_validated_unique_concept_relations": int(
            high_confidence[["concept", "relation"]].drop_duplicates().shape[0]
        ),
        "atomic_fact_count_distribution": {str(key): int(value) for key, value in fact_count_distribution.items()},
        "rule_level_counts": level_counts,
        "human_labels_joined_only_in_statistical_analysis": True,
        "qwen_api_calls": 0,
        "selection": {
            "split": "80% discovery / 20% validation by SHA-256(pair_id)",
            "discovery_min_singletons": 10,
            "discovery_min_wilson_lcb95": 0.90,
            "discovery_min_absolute_lift": 0.10,
            "discovery_max_bh_q_value": 0.05,
            "validation_min_singletons": 3,
            "validation_min_observed_probability": 0.80,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "raw_responses": str(raw_path),
        "raw_responses_sha256_at_snapshot": sha256(raw_path),
        "inputs": str(inputs_path),
        "inputs_sha256": sha256(inputs_path),
        "labels": str(labels_path),
        "labels_sha256": sha256(labels_path),
        "output_dir": str(output_dir),
        "summary": summary,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# Статистические кандидаты правил из атомарных различий

Срез checkpoint: **{checkpoint_summary['unique_pairs']}** уникальных ответов,
из них успешно обработано **{len(pairs)}**. API-вызовов на этом этапе: **0**;
human label присоединён только локально после Qwen extraction.

- без принятого различия: **{summary['pairs_with_no_accepted_differences']}**;
- ровно одно различие: **{summary['pairs_with_one_accepted_difference']}**;
- несколько различий: **{summary['pairs_with_multiple_accepted_differences']}**;
- атомарных occurrences: **{len(facts)}**;
- нормализованных concepts: **{summary['unique_canonical_concepts']}**.

Каждая пара с одним атомом немедленно создаёт запись в
`singleton_rule_evidence.parquet`. Готовым правилом одиночное наблюдение не
считается: статистика агрегируется по повторениям. Отбор делается на 80% пар и
проверяется на независимых 20% по детерминированному hash split.

Основная таблица — `rule_statistics.csv/parquet`; пороговая сетка количества
правил — `threshold_sweep.csv`; самый строгий текущий каталог —
`validated_high_confidence_rules.csv`. Для ручной проверки также сформирован
умеренный каталог `provisional_concept_rule_candidates.csv`: не менее 10
одноатомных наблюдений, нижняя 95% граница вероятности не ниже 0.80 и
подтверждение на отложенных парах.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
