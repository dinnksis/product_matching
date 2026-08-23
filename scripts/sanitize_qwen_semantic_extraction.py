"""Sanitize unified Qwen semantic facts locally, without any API calls.

The source raw checkpoint remains immutable. Candidate facts are validated and
accepted/dropped independently so one bad fact cannot discard the whole pair.
Human labels are joined only for post-inference artifacts and review.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import jsonschema
import pandas as pd

import run_qwen_semantic_extraction as runner


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "artifacts" / "qwen_semantic_extraction_v1_3_smoke50" / "raw_responses.jsonl"
DEFAULT_DATASET = ROOT / "data" / "qwen_semantic_pilot_v1" / "pilot_inputs.parquet"
DEFAULT_LABELS = ROOT / "data" / "qwen_semantic_pilot_v1" / "pilot_labels.parquet"
DEFAULT_SCHEMA = ROOT / "schemas" / "qwen_semantic_extraction_v1_3.schema.json"
DEFAULT_PROMPT = ROOT / "prompts" / "qwen_semantic_extraction_v1_3.md"
DEFAULT_OUTPUT = ROOT / "artifacts" / "qwen_semantic_extraction_v1_3_sanitized50"
SANITIZER_VERSION = "qwen_semantic_fact_sanitizer_v1_1"

PLACEHOLDER_VALUES = {
    "unknown",
    "unspecified",
    "not specified",
    "not provided",
    "no data",
    "n a",
    "no brand",
    "unbranded",
    "none",
    "не указано",
    "не указан",
    "не указана",
    "неизвестно",
    "нет данных",
    "нет бренда",
    "без бренда",
    "не определен",
    "не определён",
    "уточнить у продавца",
    "уточни у продавца",
}
LOW_VALUE_MISSING_TOKENS = {
    "country",
    "shelf_life",
    "warranty",
    "currency",
    "package_dimensions",
    "shipping_weight",
    "gross_weight",
    "instructions",
}
RELATION_PRIORITY = {
    "conflicting_sources": 6,
    "identity_same": 5,
    "different_value": 4,
    "incompatible": 4,
    "compatible": 3,
    "subset": 3,
    "more_specific": 3,
    "unknown": 2,
    "missing_a": 1,
    "missing_b": 1,
}
CONFIDENCE_PRIORITY = {"high": 3, "medium": 2, "low": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locally sanitize v1.3 semantic facts without calling Qwen."
    )
    parser.add_argument("--raw-responses", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-pairs", type=int, default=50)
    parser.add_argument(
        "--only-available-responses",
        action="store_true",
        help="From the selected dataset range, sanitize only checkpointed ok/invalid pair IDs.",
    )
    parser.add_argument(
        "--validation-profile",
        choices=("legacy", "v1_4"),
        default=None,
        help="Defaults to v1_4 when the prompt filename contains v1_4.",
    )
    return parser.parse_args()


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_placeholder_text(value: Any) -> str:
    return " ".join(
        runner.normalize_surface(value).replace("/", " ").split()
    )


def is_placeholder_side(side: Any) -> bool:
    if not isinstance(side, dict):
        return False
    value = side.get("value")
    if not isinstance(value, dict):
        return False
    raw = normalize_placeholder_text(value.get("raw_value"))
    normalized = normalize_placeholder_text(value.get("normalized_value"))
    if raw in PLACEHOLDER_VALUES or normalized in PLACEHOLDER_VALUES:
        return True
    return any(
        text.startswith("уточн") and "продав" in text
        for text in (raw, normalized)
    )


def apply_placeholder_transform(
    fact: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    """Turn one explicit placeholder side into missing; reject unusable cases."""
    transformed = copy.deepcopy(fact)
    placeholder_a = is_placeholder_side(transformed.get("a"))
    placeholder_b = is_placeholder_side(transformed.get("b"))
    transformations: list[str] = []
    reasons: list[str] = []
    if not placeholder_a and not placeholder_b:
        return transformed, transformations, reasons
    if placeholder_a and placeholder_b:
        return None, transformations, ["both_sides_placeholder"]
    relation = transformed.get("relation")
    if relation == "identity_same":
        return None, transformations, ["identity_anchor_contains_placeholder"]
    if relation in {"missing_a", "missing_b"}:
        known_side = "b" if relation == "missing_a" else "a"
        if is_placeholder_side(transformed.get(known_side)):
            return None, transformations, ["known_missing_side_is_placeholder"]
        return transformed, transformations, reasons
    if placeholder_a:
        transformed["a"] = None
        transformed["relation"] = "missing_a"
        transformations.append("placeholder_a_to_missing_a")
    else:
        transformed["b"] = None
        transformed["relation"] = "missing_b"
        transformations.append("placeholder_b_to_missing_b")
    transformed["anchor_type"] = None
    transformed["direction"] = None
    return transformed, transformations, reasons


def fact_schema_validator(root_schema: dict[str, Any]) -> jsonschema.Draft202012Validator:
    schema = {
        "$schema": root_schema["$schema"],
        "$defs": root_schema["$defs"],
        "$ref": "#/$defs/semantic_fact",
    }
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def reason_codes_from_semantic_errors(errors: list[str]) -> list[str]:
    codes: set[str] = set()
    for error in errors:
        if "evidence is not source-supported" in error:
            codes.add("evidence_not_source_supported")
        elif "identity_same values differ" in error:
            codes.add("identity_values_differ")
        elif "unsupported anchor_type/concept" in error:
            codes.add("unsupported_anchor_type_concept")
        elif "values are equivalent" in error:
            codes.add("relation_values_equivalent")
        elif "generic concepts" in error:
            codes.add("generic_concept")
        elif "duplicate concepts" in error:
            codes.add("duplicate_concept")
        elif "absence marker as a value" in error:
            codes.add("absence_marker_used_as_value")
        elif "country-like value used as brand" in error:
            codes.add("country_used_as_brand")
        elif "quantity has no explicit numeric count" in error:
            codes.add("quantity_without_explicit_count")
        elif "quantity is sourced from a measurement attribute" in error:
            codes.add("quantity_from_measurement")
        elif "missing value is present" in error:
            codes.add("missing_value_present_on_other_side")
        elif "possible title/attribute source conflict" in error:
            codes.add("possible_unmodeled_source_conflict")
        else:
            codes.add("semantic_validation_error")
    return sorted(codes)


def candidate_rank(record: dict[str, Any]) -> tuple[int, int, int, int]:
    fact = record["sanitized_fact"]
    evidence_count = sum(
        len(fact.get(side, {}).get("evidence", []))
        for side in ("a", "b")
        if isinstance(fact.get(side), dict)
    )
    return (
        RELATION_PRIORITY.get(str(fact.get("relation")), 0),
        CONFIDENCE_PRIORITY.get(str(fact.get("confidence")), 0),
        evidence_count,
        -int(record["fact_index"]),
    )


def drop_record(
    pair_id: str,
    fact_index: int,
    raw_fact: Any,
    sanitized_fact: dict[str, Any] | None,
    reason_codes: list[str],
    errors: list[str],
    transformations: list[str],
) -> dict[str, Any]:
    raw_mapping = raw_fact if isinstance(raw_fact, dict) else {}
    return {
        "pair_id": pair_id,
        "fact_index": fact_index,
        "concept": raw_mapping.get("concept"),
        "relation": raw_mapping.get("relation"),
        "reason_codes": reason_codes,
        "errors": errors,
        "transformations": transformations,
        "raw_fact": raw_fact,
        "sanitized_fact": sanitized_fact,
    }


def sanitize_facts(
    pair_id: str,
    candidate_facts: list[Any],
    payload: dict[str, Any],
    validator: jsonschema.Draft202012Validator,
    validation_profile: str = "legacy",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    provisional: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for fact_index, raw_fact in enumerate(candidate_facts):
        if not isinstance(raw_fact, dict):
            dropped.append(
                drop_record(
                    pair_id,
                    fact_index,
                    raw_fact,
                    None,
                    ["fact_not_object"],
                    [f"semantic_facts[{fact_index}] must be an object, got {type(raw_fact).__name__}"],
                    [],
                )
            )
            continue
        if raw_fact.get("relation") == "identity_same" and raw_fact.get("anchor_type") is None:
            dropped.append(
                drop_record(
                    pair_id,
                    fact_index,
                    raw_fact,
                    None,
                    ["ordinary_same_not_retained"],
                    [],
                    [],
                )
            )
            continue
        sanitized_fact, transformations, placeholder_reasons = apply_placeholder_transform(
            raw_fact
        )
        if sanitized_fact is None:
            dropped.append(
                drop_record(
                    pair_id,
                    fact_index,
                    raw_fact,
                    None,
                    placeholder_reasons,
                    [],
                    transformations,
                )
            )
            continue
        schema_errors = sorted(error.message for error in validator.iter_errors(sanitized_fact))
        if schema_errors:
            dropped.append(
                drop_record(
                    pair_id,
                    fact_index,
                    raw_fact,
                    sanitized_fact,
                    ["fact_schema_invalid"],
                    schema_errors,
                    transformations,
                )
            )
            continue
        semantic_errors = runner.unified_semantic_validation_errors(
            {"semantic_facts": [sanitized_fact]}, payload, validation_profile
        )
        if semantic_errors:
            dropped.append(
                drop_record(
                    pair_id,
                    fact_index,
                    raw_fact,
                    sanitized_fact,
                    reason_codes_from_semantic_errors(semantic_errors),
                    semantic_errors,
                    transformations,
                )
            )
            continue
        provisional.append(
            {
                "pair_id": pair_id,
                "fact_index": fact_index,
                "raw_fact": raw_fact,
                "sanitized_fact": sanitized_fact,
                "transformations": transformations,
            }
        )

    by_concept: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in provisional:
        by_concept[str(record["sanitized_fact"]["concept"])].append(record)
    accepted_records: list[dict[str, Any]] = []
    for concept, records in by_concept.items():
        selected = max(records, key=candidate_rank)
        accepted_records.append(selected)
        for record in records:
            if record is selected:
                continue
            dropped.append(
                drop_record(
                    pair_id,
                    int(record["fact_index"]),
                    record["raw_fact"],
                    record["sanitized_fact"],
                    ["duplicate_concept_lower_priority"],
                    [f"another accepted fact has concept={concept}"],
                    record["transformations"],
                )
            )
    accepted_records.sort(key=lambda record: int(record["fact_index"]))
    accepted_facts = [record["sanitized_fact"] for record in accepted_records]
    warnings: list[str] = []
    missing_facts = [
        fact for fact in accepted_facts if fact.get("relation") in {"missing_a", "missing_b"}
    ]
    if len(missing_facts) > 2:
        warnings.append(f"missing_count_exceeds_recommended_limit:{len(missing_facts)}")
    low_value = sorted(
        {
            str(fact.get("concept"))
            for fact in missing_facts
            if any(token in str(fact.get("concept")) for token in LOW_VALUE_MISSING_TOKENS)
        }
    )
    if low_value:
        warnings.append("low_value_missing_concepts:" + ",".join(low_value))
    transformed_count = sum(bool(record["transformations"]) for record in accepted_records)
    if transformed_count:
        warnings.append(f"placeholder_transformed_facts:{transformed_count}")
    if dropped:
        warnings.append(f"dropped_fact_count:{len(dropped)}")
    return accepted_records, dropped, warnings


def read_candidates(response: dict[str, Any]) -> tuple[list[Any], list[str]]:
    extraction = response.get("schema_response") or response.get("parsed_response")
    warnings: list[str] = []
    if not isinstance(extraction, dict):
        try:
            extraction = runner.extract_json(str(response.get("raw_response") or ""))
        except Exception as error:
            return [], [f"raw_response_unparseable:{type(error).__name__}:{error}"]
    facts = extraction.get("semantic_facts")
    if not isinstance(facts, list):
        return [], ["semantic_facts_missing_from_response"]
    return facts, warnings


def flat_pair_row(pair: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_id": pair["pair_id"],
        "item_id_a": pair["item_id_a"],
        "item_id_b": pair["item_id_b"],
        "category": pair["category"],
        "human_label": pair["human_label"],
        "identity_anchor_count": len(pair["identity_anchors"]),
        "difference_count": len(pair["differences"]),
        "missing_information_count": len(pair["missing_information"]),
        "candidate_fact_count": pair["sanitization"]["candidate_fact_count"],
        "accepted_fact_count": pair["sanitization"]["accepted_fact_count"],
        "dropped_fact_count": pair["sanitization"]["dropped_fact_count"],
        "identity_anchors_json": json.dumps(pair["identity_anchors"], ensure_ascii=False),
        "differences_json": json.dumps(pair["differences"], ensure_ascii=False),
        "missing_information_json": json.dumps(pair["missing_information"], ensure_ascii=False),
        "pair_summary_json": json.dumps(pair["pair_summary"], ensure_ascii=False),
        "sanitization_warnings_json": json.dumps(
            pair["sanitization"]["warnings"], ensure_ascii=False
        ),
        "source_status": pair["sanitization"]["source_status"],
        "sanitized_status": pair["sanitization"]["status"],
    }


def main() -> None:
    args = parse_args()
    if args.max_pairs < 1:
        raise ValueError("max-pairs must be positive")
    raw_path = args.raw_responses.resolve()
    dataset_path = args.dataset.resolve()
    labels_path = args.labels.resolve()
    schema_path = args.schema.resolve()
    prompt_path = args.prompt.resolve()
    validation_profile = args.validation_profile or (
        "v1_4" if "v1_4" in prompt_path.stem.casefold() else "legacy"
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = pd.read_parquet(dataset_path).head(args.max_pairs).copy()
    labels = pd.read_parquet(labels_path)
    if set(labels.columns) != {"pair_id", "human_label"}:
        raise ValueError("labels parquet must contain exactly pair_id,human_label")
    labels_by_pair = labels.set_index("pair_id", verify_integrity=True)["human_label"]
    responses = runner.read_latest_jsonl(raw_path)
    if args.only_available_responses:
        available = {
            pair_id
            for pair_id, response in responses.items()
            if response.get("status") in {"ok", "invalid"}
        }
        inputs = inputs[inputs["pair_id"].astype(str).isin(available)].copy()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = fact_schema_validator(schema)

    pair_objects: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    source_status_counts: Counter[str] = Counter()
    input_by_pair = inputs.set_index("pair_id", drop=False, verify_integrity=True)

    for row in inputs.itertuples(index=False):
        pair_id = str(row.pair_id)
        response = responses.get(pair_id)
        if response is None:
            candidate_facts, read_warnings = [], ["source_response_missing"]
            response = {"status": "missing", "raw_response": None}
        else:
            candidate_facts, read_warnings = read_candidates(response)
        source_status = str(response.get("status", "unknown"))
        source_status_counts[source_status] += 1
        payload = runner.qwen_input(row)
        accepted_records, dropped, warnings = sanitize_facts(
            pair_id, candidate_facts, payload, validator, validation_profile
        )
        warnings = read_warnings + warnings
        accepted_facts = [record["sanitized_fact"] for record in accepted_records]
        normalized = runner.normalize_extraction({"semantic_facts": accepted_facts})
        normalized["extraction_warnings"] = warnings
        for warning in warnings:
            warning_counts[warning.split(":", 1)[0]] += 1
        for dropped_record in dropped:
            reason_counts.update(dropped_record["reason_codes"])
            dropped_rows.append(
                {
                    **{key: dropped_record[key] for key in ("pair_id", "fact_index", "concept", "relation")},
                    "reason_codes_json": json.dumps(
                        dropped_record["reason_codes"], ensure_ascii=False
                    ),
                    "errors_json": json.dumps(dropped_record["errors"], ensure_ascii=False),
                    "transformations_json": json.dumps(
                        dropped_record["transformations"], ensure_ascii=False
                    ),
                    "raw_fact_json": json.dumps(dropped_record["raw_fact"], ensure_ascii=False),
                    "sanitized_fact_json": json.dumps(
                        dropped_record["sanitized_fact"], ensure_ascii=False
                    ),
                }
            )
        for record in accepted_records:
            fact = record["sanitized_fact"]
            accepted_rows.append(
                {
                    "pair_id": pair_id,
                    "fact_index": record["fact_index"],
                    "concept": fact["concept"],
                    "relation": fact["relation"],
                    "transformations_json": json.dumps(
                        record["transformations"], ensure_ascii=False
                    ),
                    "raw_fact_json": json.dumps(record["raw_fact"], ensure_ascii=False),
                    "sanitized_fact_json": json.dumps(fact, ensure_ascii=False),
                }
            )
        pair_object = {
            "pair_id": pair_id,
            "item_id_a": int(row.item_id_a),
            "item_id_b": int(row.item_id_b),
            "category": str(row.category),
            "identity_anchors": normalized["identity_anchors"],
            "differences": normalized["differences"],
            "missing_information": normalized["missing_information"],
            "pair_summary": normalized["pair_summary"],
            "extraction_warnings": warnings,
            "extraction_metadata": {
                "model": response.get("response_model") or response.get("requested_model"),
                "prompt_version": response.get("prompt_version"),
                "prompt_sha256": response.get("prompt_sha256"),
                "schema_sha256": response.get("schema_sha256"),
                "raw_response": response.get("raw_response"),
                "source_status": source_status,
                "sanitizer_version": SANITIZER_VERSION,
                "validation_profile": validation_profile,
                "sanitized_at": now(),
            },
            "sanitization": {
                "status": "usable" if accepted_facts else "empty",
                "source_status": source_status,
                "candidate_fact_count": len(candidate_facts),
                "accepted_fact_count": len(accepted_facts),
                "dropped_fact_count": len(dropped),
                "warnings": warnings,
            },
            "human_label": int(labels_by_pair.loc[pair_id]),
        }
        pair_objects.append(pair_object)
        review_rows.append(
            {
                "pair_id": pair_id,
                "category": str(row.category),
                "human_label": int(labels_by_pair.loc[pair_id]),
                "title_a": row.title_a,
                "attributes_a_json": row.attributes_a_json,
                "title_b": row.title_b,
                "attributes_b_json": row.attributes_b_json,
                "accepted_facts_json": json.dumps(accepted_facts, ensure_ascii=False),
                "dropped_facts_json": json.dumps(dropped, ensure_ascii=False),
                "warnings_json": json.dumps(warnings, ensure_ascii=False),
                "manual_review_status": "",
                "manual_review_notes": "",
            }
        )

    write_jsonl(output_dir / "sanitized_pairs.jsonl", pair_objects)
    flat = pd.DataFrame([flat_pair_row(pair) for pair in pair_objects])
    flat.to_parquet(output_dir / "sanitized_pairs.parquet", index=False)
    accepted = pd.DataFrame(accepted_rows)
    dropped = pd.DataFrame(dropped_rows)
    accepted.to_parquet(output_dir / "accepted_facts.parquet", index=False)
    dropped.to_parquet(output_dir / "dropped_facts.parquet", index=False)
    dropped.to_csv(output_dir / "dropped_facts.csv", index=False, encoding="utf-8-sig")
    quality = flat[
        [
            "pair_id", "category", "human_label", "source_status", "sanitized_status",
            "candidate_fact_count", "accepted_fact_count", "dropped_fact_count",
            "identity_anchor_count", "difference_count", "missing_information_count",
            "sanitization_warnings_json",
        ]
    ]
    quality.to_parquet(output_dir / "pair_quality.parquet", index=False)
    quality.to_csv(output_dir / "pair_quality.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(review_rows).to_csv(
        output_dir / "manual_review_queue.csv", index=False, encoding="utf-8-sig"
    )

    relation_counts = Counter(row["relation"] for row in accepted_rows)
    usable_pairs = sum(pair["sanitization"]["status"] == "usable" for pair in pair_objects)
    statistics = {
        "requested_pairs": len(inputs),
        "source_responses_found": sum(str(row.pair_id) in responses for row in inputs.itertuples(index=False)),
        "source_status_counts": dict(source_status_counts),
        "usable_pairs": usable_pairs,
        "empty_pairs": len(pair_objects) - usable_pairs,
        "candidate_facts": int(flat["candidate_fact_count"].sum()),
        "accepted_facts": int(flat["accepted_fact_count"].sum()),
        "dropped_facts": int(flat["dropped_fact_count"].sum()),
        "fact_acceptance_rate": int(flat["accepted_fact_count"].sum())
        / max(1, int(flat["candidate_fact_count"].sum())),
        "drop_reason_counts": dict(reason_counts),
        "warning_counts": dict(warning_counts),
        "accepted_relation_counts": dict(relation_counts),
        "pairs_with_dropped_facts": int(flat["dropped_fact_count"].gt(0).sum()),
        "pairs_with_more_than_two_missing": int(flat["missing_information_count"].gt(2).sum()),
        "labels_loaded_post_inference": True,
        "api_calls": 0,
        "sanitizer_version": SANITIZER_VERSION,
        "validation_profile": validation_profile,
    }
    (output_dir / "sanitization_statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "created_at": now(),
        "sanitizer_version": SANITIZER_VERSION,
        "validation_profile": validation_profile,
        "raw_responses": str(raw_path),
        "raw_responses_sha256": runner.sha256_file(raw_path),
        "dataset": str(dataset_path),
        "dataset_sha256": runner.sha256_file(dataset_path),
        "labels": str(labels_path),
        "labels_sha256": runner.sha256_file(labels_path),
        "schema": str(schema_path),
        "schema_sha256": runner.sha256_file(schema_path),
        "prompt": str(prompt_path),
        "prompt_sha256": runner.sha256_file(prompt_path),
        "api_calls": 0,
        "statistics": statistics,
    }
    (output_dir / "sanitization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# Локальная очистка результатов {prompt_path.stem}

Исходный checkpoint не изменён. Обработано пар: **{len(inputs)}**; usable после
fact-level filtering: **{usable_pairs}**; empty: **{len(pair_objects) - usable_pairs}**.
Вызовов API: **0**.

Candidate facts: **{statistics['candidate_facts']}**; принято:
**{statistics['accepted_facts']}** ({statistics['fact_acceptance_rate']:.2%});
отброшено: **{statistics['dropped_facts']}**. Пар хотя бы с одним отброшенным
fact: **{statistics['pairs_with_dropped_facts']}**.

Основные причины удаления:

```json
{json.dumps(dict(reason_counts), ensure_ascii=False, indent=2)}
```

Превышение рекомендованного лимита двух missing является warning, а не причиной
потери пары. Таких пар: **{statistics['pairs_with_more_than_two_missing']}**.

`sanitized_pairs.jsonl/parquet` содержат стабильный pair-level object.
`accepted_facts.parquet` и `dropped_facts.parquet/csv` сохраняют решения на уровне
каждого candidate fact. Для ручной разметки GOOD/BAD/AMBIGUOUS подготовлен
`manual_review_queue.csv`.

Human label не участвовал в Qwen inference или fact validation и был добавлен
только в post-inference outputs для анализа coverage.
"""
    (output_dir / "sanitization_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    print(f"Sanitized artifacts сохранены в {output_dir}")


if __name__ == "__main__":
    main()
