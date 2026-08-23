"""Create a representative, label-isolated Qwen extraction pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from src.product_matching.eda import load_human_data, normalize_text  # noqa: E402


DEFAULT_ASSIGNMENTS = ROOT / "data" / "rule_discovery_split_v1" / "split_assignments.parquet"
DEFAULT_AUDIT_CATEGORIES = (
    ROOT / "reports" / "rule_discovery_data_audit" / "category_distribution.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "data" / "qwen_semantic_pilot_v1"
DEFAULT_REPORT_DIR = ROOT / "reports" / "qwen_semantic_pilot_v1"
DEFAULT_SIZE = 500
DEFAULT_SEED = 2026
IDENTIFIER_RE = re.compile(r"(?=.*\d)[0-9a-zа-яё][0-9a-zа-яё._/-]{2,}", re.I)
SAMPLING_FLAGS = (
    "potential_hard",
    "sparse_description",
    "rich_description",
    "strong_asymmetry",
    "single_obvious_raw_difference",
    "multiple_raw_differences",
    "title_attribute_same_fact",
    "title_only_identifier",
    "attribute_only_value",
)
BAND_WEIGHTS = {"very rare": 0.95, "rare": 1.0, "medium": 1.05, "large": 1.15}
BAND_RU = {
    "very rare": "очень малая",
    "rare": "малая",
    "medium": "средняя",
    "large": "крупная",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the label-isolated RULE_DISCOVERY pilot for Qwen extraction."
    )
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--audit-categories", type=Path, default=DEFAULT_AUDIT_CATEGORIES)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--sampling-mode",
        choices=("balanced_scenarios", "prevalence_random", "all"),
        default="balanced_scenarios",
    )
    parser.add_argument(
        "--exclude-inputs",
        type=Path,
        action="append",
        default=[],
        help="Pilot input parquet whose pair_ids must be excluded; repeatable.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(seed: int, pair_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}|{pair_id}".encode("utf-8")).digest()[:8], "little"
    )


def parse_attributes(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("attributes must decode to a JSON object")
    return parsed


def token_set(value: Any) -> set[str]:
    return set(re.findall(r"[0-9a-zа-яё]+", normalize_text(value)))


def token_jaccard(left: Any, right: Any) -> float:
    a, b = token_set(left), token_set(right)
    return len(a & b) / max(1, len(a | b))


def normalized_attributes(attributes: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in attributes.items():
        normalized_key = normalize_text(key)
        normalized_value = normalize_text(value)
        if normalized_key and normalized_value:
            result[normalized_key] = normalized_value
    return result


def repeated_value_in_title(title: str, attributes: dict[str, Any]) -> bool:
    normalized_title = normalize_text(title)
    return any(
        len(value := normalize_text(raw_value)) >= 3 and value in normalized_title
        for raw_value in attributes.values()
    )


def value_absent_from_title(title: str, attributes: dict[str, Any]) -> bool:
    normalized_title = normalize_text(title)
    values = [normalize_text(value) for value in attributes.values()]
    informative = [value for value in values if len(value) >= 4]
    return bool(informative) and any(value not in normalized_title for value in informative)


def title_only_identifier(title: str, attributes: dict[str, Any]) -> bool:
    attribute_text = normalize_text(" ".join(map(str, attributes.values())))
    return any(
        normalize_text(token) not in attribute_text
        for token in IDENTIFIER_RE.findall(normalize_text(title))
    )


def attach_features(pairs: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    item_lookup = items.set_index("id", verify_integrity=True)
    result = pairs.copy()
    for side, id_column in (("a", "id1"), ("b", "id2")):
        result[f"title_{side}"] = result[id_column].map(item_lookup["name"])
        result[f"attributes_{side}_json"] = result[id_column].map(item_lookup["attributes"])
    parsed_a = result["attributes_a_json"].map(parse_attributes)
    parsed_b = result["attributes_b_json"].map(parse_attributes)
    result["attributes_count_a"] = parsed_a.map(len)
    result["attributes_count_b"] = parsed_b.map(len)
    result["detail_chars_a"] = result["title_a"].str.len() + result["attributes_a_json"].str.len()
    result["detail_chars_b"] = result["title_b"].str.len() + result["attributes_b_json"].str.len()
    result["detail_ratio"] = (
        result[["detail_chars_a", "detail_chars_b"]].max(axis=1)
        / result[["detail_chars_a", "detail_chars_b"]].min(axis=1).clip(lower=1)
    )
    result["title_similarity"] = [
        token_jaccard(left, right)
        for left, right in zip(result["title_a"], result["title_b"])
    ]

    raw_difference_counts = []
    shared_raw_keys = []
    title_attr_same = []
    title_identifier = []
    attr_only = []
    for title_a, title_b, attrs_a, attrs_b in zip(
        result["title_a"], result["title_b"], parsed_a, parsed_b
    ):
        normalized_a, normalized_b = normalized_attributes(attrs_a), normalized_attributes(attrs_b)
        shared = set(normalized_a) & set(normalized_b)
        shared_raw_keys.append(len(shared))
        raw_difference_counts.append(
            sum(normalized_a[key] != normalized_b[key] for key in shared)
        )
        title_attr_same.append(
            repeated_value_in_title(title_a, attrs_a)
            or repeated_value_in_title(title_b, attrs_b)
        )
        title_identifier.append(
            title_only_identifier(title_a, attrs_a)
            or title_only_identifier(title_b, attrs_b)
        )
        attr_only.append(
            value_absent_from_title(title_a, attrs_a)
            or value_absent_from_title(title_b, attrs_b)
        )
    result["shared_raw_attribute_keys"] = shared_raw_keys
    result["raw_shared_value_difference_count"] = raw_difference_counts
    result["title_attribute_same_fact"] = title_attr_same
    result["title_only_identifier"] = title_identifier
    result["attribute_only_value"] = attr_only

    pair_attr_total = result["attributes_count_a"] + result["attributes_count_b"]
    category_thresholds = result.groupby("category", observed=True).agg(
        sparse_attr_threshold=("attributes_count_a", lambda x: x.quantile(0.10)),
        rich_attr_threshold=("attributes_count_a", lambda x: x.quantile(0.90)),
        asymmetry_threshold=("detail_ratio", lambda x: x.quantile(0.95)),
        low_title_similarity=("title_similarity", lambda x: x.quantile(0.20)),
        high_title_similarity=("title_similarity", lambda x: x.quantile(0.80)),
    )
    result["sparse_description"] = [
        min(a, b) <= category_thresholds.loc[category, "sparse_attr_threshold"]
        for category, a, b in zip(
            result["category"], result["attributes_count_a"], result["attributes_count_b"]
        )
    ]
    result["rich_description"] = [
        min(a, b) >= category_thresholds.loc[category, "rich_attr_threshold"]
        for category, a, b in zip(
            result["category"], result["attributes_count_a"], result["attributes_count_b"]
        )
    ]
    result["strong_asymmetry"] = [
        ratio >= category_thresholds.loc[category, "asymmetry_threshold"]
        for category, ratio in zip(result["category"], result["detail_ratio"])
    ]
    result["single_obvious_raw_difference"] = (
        result["raw_shared_value_difference_count"].eq(1)
        & result["title_similarity"].ge(0.45)
    )
    result["multiple_raw_differences"] = result["raw_shared_value_difference_count"].ge(3)
    result["potential_hard"] = [
        (target == 1 and similarity <= category_thresholds.loc[category, "low_title_similarity"])
        or (target == 0 and similarity >= category_thresholds.loc[category, "high_title_similarity"])
        for category, target, similarity in zip(
            result["category"], result["target"], result["title_similarity"]
        )
    ]
    result["pair_attribute_total"] = pair_attr_total
    return result


def category_quotas(size: int, categories: pd.DataFrame) -> dict[str, int]:
    if size < 2 * len(categories):
        raise ValueError("Pilot size must allow both labels in every category")
    weights = categories["empirical_size_band"].map(BAND_WEIGHTS).astype(float)
    raw = size * weights / weights.sum()
    quota = np.floor(raw).astype(int)
    remainder = size - int(quota.sum())
    order = (raw - quota).sort_values(ascending=False).index
    for index in order[:remainder]:
        quota.loc[index] += 1
    return dict(zip(categories["category"], quota))


def select_stratum(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    ordered = frame.sort_values(["stable_rank", "pair_id"]).copy()
    selected: list[int] = []
    selected_set: set[int] = set()
    queues = {
        flag: list(ordered.index[ordered[flag].astype(bool)]) for flag in SAMPLING_FLAGS
    }
    while len(selected) < count:
        progress = False
        for flag in SAMPLING_FLAGS:
            while queues[flag] and queues[flag][0] in selected_set:
                queues[flag].pop(0)
            if queues[flag] and len(selected) < count:
                index = queues[flag].pop(0)
                selected.append(index)
                selected_set.add(index)
                progress = True
        if not progress:
            break
    if len(selected) < count:
        for index in ordered.index:
            if index not in selected_set:
                selected.append(index)
                selected_set.add(index)
                if len(selected) == count:
                    break
    if len(selected) != count:
        raise RuntimeError(f"Not enough rows in sampling stratum: wanted={count}")
    return ordered.loc[selected]


def sample_pilot(
    featured: pd.DataFrame,
    categories: pd.DataFrame,
    size: int,
    sampling_mode: str = "balanced_scenarios",
) -> pd.DataFrame:
    if sampling_mode == "all":
        if len(featured) != size:
            raise RuntimeError(
                f"sampling-mode=all requires size={len(featured)}, got {size}"
            )
        result = featured.sort_values(["stable_rank", "pair_id"]).reset_index(drop=True)
        if result["pair_id"].duplicated().any():
            raise RuntimeError("Full extraction input contains duplicate pair_ids")
        return result
    quotas = category_quotas(size, categories)
    selected = []
    for category in categories["category"]:
        category_size = quotas[category]
        category_rows = featured[featured["category"].eq(category)]
        if sampling_mode == "prevalence_random":
            ordered = category_rows.sort_values(["stable_rank", "pair_id"])
            if len(ordered) < category_size:
                raise RuntimeError(
                    f"Not enough rows in category {category}: wanted={category_size}"
                )
            selected.append(ordered.head(category_size))
        else:
            positive_count = category_size // 2
            negative_count = category_size - positive_count
            selected.append(select_stratum(category_rows[category_rows["target"].eq(1)], positive_count))
            selected.append(select_stratum(category_rows[category_rows["target"].eq(0)], negative_count))
    result = pd.concat(selected, ignore_index=True)
    if len(result) != size or result["pair_id"].duplicated().any():
        raise RuntimeError("Pilot sample size/uniqueness invariant failed")
    return result.sort_values("stable_rank").reset_index(drop=True)


def request_object(row: Any) -> dict[str, Any]:
    return {
        "category": str(row.category),
        "item_a": {
            "title": str(row.title_a),
            "attributes": parse_attributes(row.attributes_a_json),
        },
        "item_b": {
            "title": str(row.title_b),
            "attributes": parse_attributes(row.attributes_b_json),
        },
    }


def markdown_table(frame: pd.DataFrame) -> str:
    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir, report_dir = args.output_dir.resolve(), args.report_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    assignments = pd.read_parquet(args.assignments.resolve())
    discovery = assignments[assignments["split"].eq("rule_discovery")].copy()
    if len(discovery) != 277_349:
        raise RuntimeError("Unexpected RULE_DISCOVERY size")
    excluded_pair_ids: set[str] = set()
    for exclusion_path in args.exclude_inputs:
        excluded = pd.read_parquet(exclusion_path.resolve(), columns=["pair_id"])
        excluded_pair_ids.update(excluded["pair_id"].astype(str))
    if excluded_pair_ids:
        discovery = discovery[
            ~discovery["pair_id"].astype(str).isin(excluded_pair_ids)
        ].copy()
    items, _ = load_human_data(args.data_dir.resolve())
    categories = pd.read_csv(args.audit_categories.resolve())[
        ["category", "empirical_size_band"]
    ]
    featured = attach_features(discovery, items)
    featured["stable_rank"] = featured["pair_id"].map(
        lambda pair_id: stable_rank(args.seed, str(pair_id))
    )
    pilot = sample_pilot(featured, categories, args.size, args.sampling_mode)
    if set(pilot["pair_id"].astype(str)) & excluded_pair_ids:
        raise RuntimeError("Excluded pair_id leaked into the new pilot")

    input_columns = [
        "pair_id", "id1", "id2", "category", "title_a", "attributes_a_json",
        "title_b", "attributes_b_json",
    ]
    pilot_inputs = pilot[input_columns].rename(
        columns={"id1": "item_id_a", "id2": "item_id_b"}
    )
    forbidden = {"target", "label", "human_label"} & set(pilot_inputs.columns)
    if forbidden:
        raise RuntimeError(f"Label leakage into pilot inputs: {sorted(forbidden)}")
    pilot_labels = pilot[["pair_id", "target"]].rename(columns={"target": "human_label"})
    metadata_columns = [
        "pair_id", "category", "target", "attributes_count_a", "attributes_count_b",
        "detail_ratio", "title_similarity", "shared_raw_attribute_keys",
        "raw_shared_value_difference_count", *SAMPLING_FLAGS,
    ]
    sampling_metadata = pilot[metadata_columns].rename(columns={"target": "human_label"})
    sampling_metadata["sampling_tags"] = sampling_metadata.apply(
        lambda row: json.dumps(
            [flag for flag in SAMPLING_FLAGS if bool(row[flag])], ensure_ascii=False
        ),
        axis=1,
    )

    pilot_inputs.to_parquet(output_dir / "pilot_inputs.parquet", index=False)
    pilot_labels.to_parquet(output_dir / "pilot_labels.parquet", index=False)
    sampling_metadata.to_parquet(output_dir / "pilot_sampling_metadata.parquet", index=False)
    with (output_dir / "qwen_request_preview.jsonl").open("w", encoding="utf-8") as stream:
        for row in pilot_inputs.itertuples(index=False):
            payload = request_object(row)
            forbidden_payload_keys = {"target", "label", "human_label"}
            structural_keys = set(payload)
            structural_keys.update(payload["item_a"])
            structural_keys.update(payload["item_b"])
            if structural_keys & forbidden_payload_keys:
                raise RuntimeError("A Qwen request preview contains a label field")
            stream.write(json.dumps({"pair_id": row.pair_id, "qwen_input": payload}, ensure_ascii=False) + "\n")

    by_category = (
        sampling_metadata.groupby("category", observed=True)
        .agg(
            pairs=("pair_id", "size"),
            positives=("human_label", "sum"),
            negatives=("human_label", lambda value: value.eq(0).sum()),
            sparse=("sparse_description", "sum"),
            rich=("rich_description", "sum"),
            hard=("potential_hard", "sum"),
            single_difference=("single_obvious_raw_difference", "sum"),
            multiple_differences=("multiple_raw_differences", "sum"),
            asymmetric=("strong_asymmetry", "sum"),
        )
        .reset_index()
        .merge(categories, on="category", how="left")
    )
    flag_counts = pd.DataFrame(
        {
            "sampling_tag": list(SAMPLING_FLAGS),
            "pairs": [int(sampling_metadata[flag].sum()) for flag in SAMPLING_FLAGS],
        }
    )
    manifest = {
        "schema_version": 1,
        "sample_size": len(pilot),
        "seed": args.seed,
        "sampling_mode": args.sampling_mode,
        "excluded_pair_ids": len(excluded_pair_ids),
        "source": "RULE_DISCOVERY only",
        "source_assignments": str(args.assignments.resolve()),
        "source_assignments_sha256": sha256(args.assignments.resolve()),
        "excluded_from_sampling": [
            "rule_internal_validation", "ordinary", "hard", "ood"
        ],
        "qwen_input_contains_human_label": False,
        "pilot_inputs_sha256": sha256(output_dir / "pilot_inputs.parquet"),
        "pilot_labels_sha256": sha256(output_dir / "pilot_labels.parquet"),
        "label_join_policy": "labels are loaded only after all Qwen responses are received",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    by_category.to_csv(report_dir / "pilot_by_category.csv", index=False, encoding="utf-8-sig")
    flag_counts.to_csv(report_dir / "pilot_sampling_tags.csv", index=False, encoding="utf-8-sig")
    report = f"""# Pilot-выборка для Qwen semantic extraction v1

Использован только `RULE_DISCOVERY`. Размер: **{len(pilot)}** пар, seed:
**{args.seed}**. Human labels физически отделены от inference inputs:

- `pilot_inputs.parquet` не содержит `target`, `label` или `human_label`;
- `pilot_labels.parquet` содержит только `pair_id` и `human_label`;
- `qwen_request_preview.jsonl` позволяет проверить фактическую структуру запроса;
- inference-код читает labels только после завершения всех Qwen-запросов.

Размер категории учитывает квартиль из предыдущего data audit. Режим sampling:
**{args.sampling_mode}**. Label может использоваться только sampler’ом и никогда
не попадает в Qwen payload. Исключено ранее обработанных pair IDs:
**{len(excluded_pair_ids)}**.

## Покрытие категорий

{markdown_table(by_category.rename(columns={
    "category": "категория", "pairs": "пары", "positives": "positive",
    "negatives": "negative", "sparse": "sparse", "rich": "rich",
    "hard": "potential hard", "single_difference": "один raw difference",
    "multiple_differences": "несколько raw differences", "asymmetric": "асимметричные",
    "empirical_size_band": "группа размера",
}))}

## Покрытие эвристических сценариев

{markdown_table(flag_counts.rename(columns={"sampling_tag": "сценарий", "pairs": "пары"}))}

Эти теги применяются только для покрытия pilot и manual review. Они не являются
semantic labels и не передаются Qwen.
"""
    (report_dir / "sampling_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    print(f"Pilot inputs: {output_dir / 'pilot_inputs.parquet'}", flush=True)
    print(f"Pilot labels: {output_dir / 'pilot_labels.parquet'}", flush=True)


if __name__ == "__main__":
    main()
