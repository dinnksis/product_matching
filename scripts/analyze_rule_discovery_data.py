"""Audit the frozen human train before designing a rule-discovery split.

This script is intentionally read-only with respect to the source data.  It
uses the project's existing human-data loader and frozen validation paths, and
writes only diagnostic tables and a Markdown report below ``reports/``.

No split is created, no validation row is used to infer a matching rule, and
no external model is called.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from src.product_matching.eda import load_human_data, normalize_text  # noqa: E402


DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_SPLIT_DIR = DEFAULT_DATA_DIR / "validation_splits_v1"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "rule_discovery_data_audit"
SPLIT_FILES = {
    "train": "human_train_pairs.parquet",
    "ordinary": "human_iid_validation_pairs.parquet",
    "hard": "human_hard_validation_pairs.parquet",
    "ood": "human_ood_validation_pairs.parquet",
}

TOKEN_RE = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)
SURFACE_KEY_RE = re.compile(r"[^0-9a-zа-яё]+", re.IGNORECASE)
CONCEPT_HINTS: dict[str, re.Pattern[str]] = {
    "brand_or_manufacturer": re.compile(r"бренд|производител|изготовител|brand", re.I),
    "model": re.compile(r"модел|model", re.I),
    "article_or_identifier": re.compile(
        r"артикул|партномер|штрих.?код|код товара|код производ|\boem\b|\boe\b|\bsku\b",
        re.I,
    ),
    "color": re.compile(r"цвет|оттенок", re.I),
    "size_or_dimensions": re.compile(
        r"размер|габарит|длина|ширина|высота|диаметр|толщина|глубина", re.I
    ),
    "volume": re.compile(r"объ[её]м|литраж", re.I),
    "weight": re.compile(r"вес|масса", re.I),
    "quantity_or_bundle": re.compile(
        r"количеств|число|штук|единиц|комплект|фасов", re.I
    ),
    "material_or_composition": re.compile(r"материал|состав|покрытие", re.I),
    "country": re.compile(r"стран", re.I),
    "compatibility": re.compile(r"совместим|назначение|марка автомобил", re.I),
}


@dataclass(frozen=True)
class ComponentDiagnostics:
    components: pd.DataFrame
    size_distribution: pd.DataFrame
    summary: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit current human train for future rule-discovery splitting."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--top-products", type=int, default=100,
        help="Number of high-degree products to retain.",
    )
    parser.add_argument(
        "--examples-per-category", type=int, default=4,
        help="Maximum strongly asymmetric examples per category.",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    table = frame if columns is None else frame[columns]
    labels = [str(column) for column in table.columns]

    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join("---" for _ in labels) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def attach_category(pairs: pd.DataFrame, category_by_id: pd.Series, name: str) -> pd.DataFrame:
    result = pairs.copy()
    result["category"] = result["id1"].map(category_by_id)
    right = result["id2"].map(category_by_id)
    if result[["category"]].isna().any().any() or right.isna().any():
        raise ValueError(f"{name}: pair ids are missing from the human item catalogue")
    cross = result["category"].ne(right)
    if cross.any():
        raise ValueError(f"{name}: found {int(cross.sum())} cross-category pairs")
    return result


def load_inputs(
    data_dir: Path, split_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, Path]]:
    # Reuse the existing project loader for the raw human catalogue and labels.
    items, raw_pairs = load_human_data(data_dir)
    require_columns(items, ("id", "name", "attributes", "category"), data_dir)
    require_columns(raw_pairs, ("id1", "id2", "target"), data_dir)
    if items["id"].duplicated().any():
        raise ValueError("Human item ids must be unique")

    category_by_id = items.set_index("id", verify_integrity=True)["category"]
    paths: dict[str, Path] = {
        "items": data_dir / "items_human.parquet",
        "raw_pairs": data_dir / "matches.parquet",
    }
    splits: dict[str, pd.DataFrame] = {}
    for name, filename in SPLIT_FILES.items():
        path = split_dir / filename
        pairs = pd.read_parquet(path, columns=["id1", "id2", "target"])
        require_columns(pairs, ("id1", "id2", "target"), path)
        splits[name] = attach_category(pairs, category_by_id, name)
        paths[name] = path
    return items, raw_pairs, splits, paths


def split_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    train_ids = set(splits["train"]["id1"]) | set(splits["train"]["id2"])
    train_categories = set(splits["train"]["category"])
    rows = []
    for name, pairs in splits.items():
        ids = set(pairs["id1"]) | set(pairs["id2"])
        positives = int(pairs["target"].sum())
        categories = sorted(pairs["category"].unique())
        rows.append(
            {
                "split": name,
                "pairs": len(pairs),
                "unique_products": len(ids),
                "positives": positives,
                "negatives": len(pairs) - positives,
                "prevalence": float(pairs["target"].mean()),
                "category_count": len(categories),
                "categories": "; ".join(categories),
                "item_id_overlap_with_train": (
                    len(ids & train_ids) if name != "train" else len(train_ids)
                ),
                "categories_absent_from_train": "; ".join(
                    sorted(set(categories).difference(train_categories))
                ),
            }
        )
    return pd.DataFrame(rows)


def frozen_partition_integrity(
    raw_pairs: pd.DataFrame, splits: dict[str, pd.DataFrame]
) -> dict[str, Any]:
    columns = ["id1", "id2", "target"]
    raw = raw_pairs[columns].sort_values(columns).reset_index(drop=True)
    combined = (
        pd.concat([pairs[columns] for pairs in splits.values()], ignore_index=True)
        .sort_values(columns)
        .reset_index(drop=True)
    )
    return {
        "raw_pairs": len(raw),
        "combined_split_pairs": len(combined),
        "raw_exact_duplicate_rows": int(raw.duplicated().sum()),
        "combined_exact_duplicate_rows": int(combined.duplicated().sum()),
        "combined_splits_equal_raw_pair_multiset": bool(raw.equals(combined)),
    }


def category_distribution(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    endpoints = pd.concat(
        [
            train[["category", "id1"]].rename(columns={"id1": "id"}),
            train[["category", "id2"]].rename(columns={"id2": "id"}),
        ],
        ignore_index=True,
    )
    unique_products = endpoints.drop_duplicates().groupby("category")["id"].size()
    result = (
        train.groupby("category", observed=True)["target"]
        .agg(pairs="size", positives="sum", prevalence="mean")
        .join(unique_products.rename("unique_products"))
        .reset_index()
    )
    result["positives"] = result["positives"].astype(int)
    result["negatives"] = result["pairs"] - result["positives"]
    result["share_of_train_pairs"] = result["pairs"] / len(train)
    result = result.sort_values(["pairs", "category"]).reset_index(drop=True)
    # These are descriptive empirical quartiles, not proposed split thresholds.
    ranks = result["pairs"].rank(method="first")
    result["empirical_size_band"] = pd.qcut(
        ranks,
        q=4,
        labels=["very rare", "rare", "medium", "large"],
    ).astype(str)
    result["pair_count_rank_ascending"] = np.arange(1, len(result) + 1)
    quantiles = pd.DataFrame(
        {
            "quantile": ["min", "p10", "p25", "p50", "p75", "p90", "max"],
            "pairs": [
                result["pairs"].min(),
                result["pairs"].quantile(0.10),
                result["pairs"].quantile(0.25),
                result["pairs"].quantile(0.50),
                result["pairs"].quantile(0.75),
                result["pairs"].quantile(0.90),
                result["pairs"].max(),
            ],
            "positive_pairs": [
                result["positives"].min(),
                result["positives"].quantile(0.10),
                result["positives"].quantile(0.25),
                result["positives"].quantile(0.50),
                result["positives"].quantile(0.75),
                result["positives"].quantile(0.90),
                result["positives"].max(),
            ],
            "unique_products": [
                result["unique_products"].min(),
                result["unique_products"].quantile(0.10),
                result["unique_products"].quantile(0.25),
                result["unique_products"].quantile(0.50),
                result["unique_products"].quantile(0.75),
                result["unique_products"].quantile(0.90),
                result["unique_products"].max(),
            ],
        }
    )
    return result, quantiles


def label_distribution(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, part in [("all", train), *train.groupby("category", observed=True)]:
        total = len(part)
        for label in (0, 1):
            count = int(part["target"].eq(label).sum())
            rows.append(
                {
                    "scope": scope,
                    "label": label,
                    "pairs": count,
                    "share_within_scope": count / total,
                }
            )
    return pd.DataFrame(rows)


def product_degrees(
    train: pd.DataFrame, items: pd.DataFrame, top_n: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    endpoints = pd.concat(
        [
            train[["id1", "target"]].rename(columns={"id1": "id"}),
            train[["id2", "target"]].rename(columns={"id2": "id"}),
        ],
        ignore_index=True,
    )
    endpoints["positive_incident_pair"] = endpoints["target"].eq(1).astype(np.int8)
    degree = endpoints.groupby("id", sort=False).agg(
        degree=("target", "size"),
        positive_incident_pairs=("positive_incident_pair", "sum"),
    )
    degree["negative_incident_pairs"] = degree["degree"] - degree["positive_incident_pairs"]
    distribution = degree["degree"].value_counts().sort_index().rename_axis("degree").reset_index(name="products")
    distribution["product_share"] = distribution["products"] / len(degree)
    distribution["cumulative_product_share"] = distribution["product_share"].cumsum()

    quantile_values = [0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1]
    quantiles = pd.DataFrame(
        {
            "quantile": [f"p{int(q * 100):02d}" for q in quantile_values],
            "degree": degree["degree"].quantile(quantile_values).to_numpy(),
        }
    )
    quantiles = pd.concat(
        [
            quantiles,
            pd.DataFrame(
                [{"quantile": "mean", "degree": float(degree["degree"].mean())}]
            ),
        ],
        ignore_index=True,
    )
    item_fields = items.set_index("id")[["name", "category"]]
    top = (
        degree.sort_values(["degree", "positive_incident_pairs"], ascending=False)
        .head(top_n)
        .join(item_fields, how="left")
        .reset_index()
    )
    return distribution, quantiles, top


def connected_components(train: pd.DataFrame) -> ComponentDiagnostics:
    ids = pd.unique(train[["id1", "id2"]].to_numpy().reshape(-1))
    position = pd.Series(np.arange(len(ids), dtype=np.int32), index=ids)
    left = position.loc[train["id1"]].to_numpy()
    right = position.loc[train["id2"]].to_numpy()
    parent = np.arange(len(ids), dtype=np.int32)
    size = np.ones(len(ids), dtype=np.int32)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for first, second in zip(left, right):
        root1, root2 = find(int(first)), find(int(second))
        if root1 == root2:
            continue
        if size[root1] < size[root2]:
            root1, root2 = root2, root1
        parent[root2] = root1
        size[root1] += size[root2]

    roots = np.fromiter((find(i) for i in range(len(ids))), dtype=np.int32)
    edge_roots = roots[left]
    node_counts = pd.Series(roots).value_counts()
    edge_frame = pd.DataFrame(
        {"root": edge_roots, "target": train["target"].to_numpy()}
    )
    edge_counts = edge_frame.groupby("root")["target"].agg(
        edges="size", positive_edges="sum"
    )
    components = (
        node_counts.rename("nodes")
        .to_frame()
        .join(edge_counts, how="left")
        .fillna(0)
        .reset_index(names="root")
    )
    components["nodes"] = components["nodes"].astype(int)
    components["edges"] = components["edges"].astype(int)
    components["positive_edges"] = components["positive_edges"].astype(int)
    components["negative_edges"] = components["edges"] - components["positive_edges"]
    components = components.sort_values(["nodes", "edges"], ascending=False).reset_index(drop=True)
    components.insert(0, "component_rank", np.arange(1, len(components) + 1))
    size_distribution = (
        components.groupby(["nodes", "edges"], observed=True)
        .size()
        .rename("components")
        .reset_index()
        .sort_values(["nodes", "edges"])
    )
    largest = components.iloc[0]
    summary = {
        "components": len(components),
        "largest_component_nodes": int(largest["nodes"]),
        "largest_component_edges": int(largest["edges"]),
        "largest_component_node_share": float(largest["nodes"] / len(ids)),
        "largest_component_edge_share": float(largest["edges"] / len(train)),
        "components_over_0_1pct_nodes": int(
            components["nodes"].gt(len(ids) * 0.001).sum()
        ),
        "median_component_nodes": float(components["nodes"].median()),
        "p95_component_nodes": float(components["nodes"].quantile(0.95)),
        "p99_component_nodes": float(components["nodes"].quantile(0.99)),
    }
    return ComponentDiagnostics(components, size_distribution, summary)


def pair_duplicates(train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    low = np.minimum(train["id1"].to_numpy(), train["id2"].to_numpy())
    high = np.maximum(train["id1"].to_numpy(), train["id2"].to_numpy())
    audit = train[["id1", "id2", "target", "category"]].copy()
    audit["canonical_id1"] = low
    audit["canonical_id2"] = high
    audit["orientation"] = np.where(audit["id1"].eq(low), "canonical", "reversed")
    exact_sizes = audit.groupby(["id1", "id2"], sort=False).size()
    grouped = audit.groupby(["canonical_id1", "canonical_id2"], sort=False).agg(
        rows=("target", "size"),
        labels=("target", "nunique"),
        orientations=("orientation", "nunique"),
        category=("category", "first"),
        label_values=("target", lambda values: ";".join(map(str, sorted(set(values))))),
    )
    suspicious = grouped[
        grouped["rows"].gt(1) | grouped["labels"].gt(1) | grouped["orientations"].gt(1)
    ].reset_index()
    summary = {
        "exact_order_duplicate_groups": int(exact_sizes.gt(1).sum()),
        "exact_order_duplicate_extra_rows": int((exact_sizes - 1).clip(lower=0).sum()),
        "unordered_duplicate_groups": int(grouped["rows"].gt(1).sum()),
        "unordered_duplicate_extra_rows": int((grouped["rows"] - 1).clip(lower=0).sum()),
        "reversed_duplicate_groups": int(grouped["orientations"].gt(1).sum()),
        "same_ids_conflicting_label_groups": int(grouped["labels"].gt(1).sum()),
        "self_pairs": int(train["id1"].eq(train["id2"]).sum()),
    }
    return suspicious, summary


def surface_key(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().replace("ё", "е")
    return " ".join(SURFACE_KEY_RE.sub(" ", text).split())


def flatten_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def meaningful_tokens(value: str) -> set[str]:
    return {
        token
        for token in TOKEN_RE.findall(normalize_text(value))
        if len(token) >= 3 or any(character.isdigit() for character in token)
    }


def attribute_and_completeness_diagnostics(
    train_items: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    global_keys: Counter[str] = Counter()
    category_keys: dict[str, Counter[str]] = defaultdict(Counter)
    key_categories: dict[str, set[str]] = defaultdict(set)
    surface_groups: dict[str, Counter[str]] = defaultdict(Counter)
    product_rows: list[dict[str, Any]] = []
    invalid_json = 0

    for row in train_items.itertuples(index=False):
        row_has_invalid_json = False
        try:
            attributes = json.loads(row.attributes)
        except (TypeError, json.JSONDecodeError):
            attributes = {}
            invalid_json += 1
            row_has_invalid_json = True
        if not isinstance(attributes, dict):
            attributes = {}
            invalid_json += 1
            row_has_invalid_json = True

        keys = list(attributes)
        global_keys.update(keys)
        category_keys[str(row.category)].update(keys)
        for key in keys:
            key_categories[key].add(str(row.category))
            surface_groups[surface_key(key)][key] += 1

        title_normalized = normalize_text(row.name)
        title_tokens = meaningful_tokens(row.name)
        attribute_value_texts = [flatten_value(value) for value in attributes.values()]
        attribute_text = " ".join(attribute_value_texts)
        attribute_tokens = meaningful_tokens(attribute_text)
        title_only_tokens = title_tokens.difference(attribute_tokens)
        title_identifier_tokens = {
            token for token in title_tokens if any(character.isdigit() for character in token)
        }
        identifier_only = title_identifier_tokens.difference(attribute_tokens)
        informative_values = [
            normalize_text(value)
            for value in attribute_value_texts
            if len(normalize_text(value)) >= 3
        ]
        repeated_values = sum(
            bool(value and value in title_normalized) for value in informative_values
        )
        product_rows.append(
            {
                "id": int(row.id),
                "category": str(row.category),
                "attribute_count": len(attributes),
                "title_chars": len(str(row.name)),
                "attributes_chars": len(str(row.attributes)),
                "attribute_value_chars": len(attribute_text),
                "title_meaningful_tokens": len(title_tokens),
                "title_only_meaningful_tokens": len(title_only_tokens),
                "title_only_token_share": len(title_only_tokens) / max(1, len(title_tokens)),
                "title_identifier_tokens": len(title_identifier_tokens),
                "title_identifiers_absent_from_attributes": len(identifier_only),
                "informative_attribute_values": len(informative_values),
                "attribute_values_repeated_in_title": repeated_values,
                "attribute_value_repeat_share": repeated_values / max(1, len(informative_values)),
                "any_attribute_value_repeated_in_title": repeated_values > 0,
                "invalid_attributes_json": row_has_invalid_json,
            }
        )

    products = pd.DataFrame(product_rows)
    global_attribute_names = pd.DataFrame(
        [
            {
                "raw_attribute_name": key,
                "product_occurrences": count,
                "product_coverage": count / len(train_items),
                "category_count": len(key_categories[key]),
            }
            for key, count in global_keys.most_common()
        ]
    )
    category_attribute_rows = []
    category_sizes = train_items["category"].value_counts()
    for category, counter in category_keys.items():
        for key, count in counter.most_common():
            category_attribute_rows.append(
                {
                    "category": category,
                    "raw_attribute_name": key,
                    "product_occurrences": count,
                    "category_product_coverage": count / category_sizes[category],
                    "global_category_count": len(key_categories[key]),
                }
            )
    category_attribute_names = pd.DataFrame(category_attribute_rows)

    count_distribution = (
        products.groupby(["category", "attribute_count"], observed=True)
        .size()
        .rename("products")
        .reset_index()
    )
    global_count_distribution = (
        products["attribute_count"].value_counts().sort_index().rename_axis("attribute_count").reset_index(name="products")
    )
    global_count_distribution.insert(0, "category", "__ALL__")
    count_distribution = pd.concat(
        [global_count_distribution, count_distribution], ignore_index=True
    )
    count_distribution["share_within_category"] = count_distribution["products"] / count_distribution.groupby("category")["products"].transform("sum")

    category_attribute_summary = (
        products.groupby("category", observed=True).agg(
            products=("id", "size"),
            mean_attributes=("attribute_count", "mean"),
            median_attributes=("attribute_count", "median"),
            p10_attributes=("attribute_count", lambda x: x.quantile(0.10)),
            p90_attributes=("attribute_count", lambda x: x.quantile(0.90)),
            products_without_attributes=("attribute_count", lambda x: x.eq(0).sum()),
            products_with_at_most_2_attributes=("attribute_count", lambda x: x.le(2).sum()),
            products_with_at_most_5_attributes=("attribute_count", lambda x: x.le(5).sum()),
        ).reset_index()
    )
    raw_key_counts = category_attribute_names.groupby("category")["raw_attribute_name"].nunique()
    category_attribute_summary = category_attribute_summary.join(
        raw_key_counts.rename("raw_attribute_names"), on="category"
    )
    for count_column in (
        "products_without_attributes",
        "products_with_at_most_2_attributes",
        "products_with_at_most_5_attributes",
    ):
        category_attribute_summary[count_column + "_share"] = (
            category_attribute_summary[count_column] / category_attribute_summary["products"]
        )

    category_sets = {category: set(counter) for category, counter in category_keys.items()}
    overlap_rows = []
    categories = sorted(category_sets)
    for i, first in enumerate(categories):
        for second in categories[i + 1 :]:
            shared = category_sets[first] & category_sets[second]
            union = category_sets[first] | category_sets[second]
            overlap_rows.append(
                {
                    "category_1": first,
                    "category_2": second,
                    "raw_names_category_1": len(category_sets[first]),
                    "raw_names_category_2": len(category_sets[second]),
                    "shared_raw_names": len(shared),
                    "jaccard": len(shared) / max(1, len(union)),
                }
            )
    category_overlap = pd.DataFrame(overlap_rows).sort_values("jaccard", ascending=False)

    alias_rows = []
    surface_candidates = []
    for normalized, variants in surface_groups.items():
        if normalized and len(variants) > 1:
            surface_candidates.append((sum(variants.values()), normalized, variants))
    for _, normalized, variants in sorted(surface_candidates, reverse=True)[:100]:
        for raw_name, count in variants.most_common():
            alias_rows.append(
                {
                    "heuristic_group": f"surface:{normalized}",
                    "raw_attribute_name": raw_name,
                    "product_occurrences": count,
                    "category_count": len(key_categories[raw_name]),
                    "basis": "surface-only normalization; not a canonical mapping",
                }
            )
    for concept, pattern in CONCEPT_HINTS.items():
        candidates = [
            (count, key)
            for key, count in global_keys.items()
            if pattern.search(key)
        ]
        for count, key in sorted(candidates, reverse=True)[:20]:
            alias_rows.append(
                {
                    "heuristic_group": f"concept_hint:{concept}",
                    "raw_attribute_name": key,
                    "product_occurrences": count,
                    "category_count": len(key_categories[key]),
                    "basis": "lexical concept hint; requires human review",
                }
            )
    aliases = pd.DataFrame(alias_rows).drop_duplicates(
        ["heuristic_group", "raw_attribute_name"]
    )

    global_row = pd.DataFrame(
        [
            {
                "category": "__ALL__",
                "products": len(products),
                "raw_attribute_names": len(global_keys),
                "mean_attributes": products["attribute_count"].mean(),
                "median_attributes": products["attribute_count"].median(),
                "p10_attributes": products["attribute_count"].quantile(0.10),
                "p90_attributes": products["attribute_count"].quantile(0.90),
                "products_without_attributes": int(products["attribute_count"].eq(0).sum()),
                "products_with_at_most_2_attributes": int(products["attribute_count"].le(2).sum()),
                "products_with_at_most_5_attributes": int(products["attribute_count"].le(5).sum()),
                "products_without_attributes_share": float(products["attribute_count"].eq(0).mean()),
                "products_with_at_most_2_attributes_share": float(products["attribute_count"].le(2).mean()),
                "products_with_at_most_5_attributes_share": float(products["attribute_count"].le(5).mean()),
                "invalid_json_rows": invalid_json,
            }
        ]
    )
    category_attribute_summary["invalid_json_rows"] = 0
    category_attribute_summary = pd.concat(
        [global_row, category_attribute_summary], ignore_index=True, sort=False
    )
    return (
        products,
        global_attribute_names,
        category_attribute_names,
        count_distribution,
        category_attribute_summary,
        category_overlap,
        aliases,
        pd.DataFrame(
            [
                {
                    "category_count": category_count,
                    "raw_attribute_names": count,
                    "share_of_raw_attribute_names": count / max(1, len(global_keys)),
                }
                for category_count, count in sorted(
                    Counter(len(categories_) for categories_ in key_categories.values()).items()
                )
            ]
        ),
    )


def completeness_summary(products: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "attribute_count",
        "title_chars",
        "attributes_chars",
        "attribute_value_chars",
        "title_meaningful_tokens",
        "title_only_meaningful_tokens",
        "title_only_token_share",
        "title_identifier_tokens",
        "title_identifiers_absent_from_attributes",
        "informative_attribute_values",
        "attribute_values_repeated_in_title",
        "attribute_value_repeat_share",
    ]
    quantiles = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0]
    rows = []
    for metric in metrics:
        values = products[metric]
        for quantile in quantiles:
            rows.append(
                {
                    "metric": metric,
                    "quantile": f"p{int(quantile * 100):02d}",
                    "value": float(values.quantile(quantile)),
                }
            )
        rows.append({"metric": metric, "quantile": "mean", "value": float(values.mean())})
    overall = pd.DataFrame(rows)
    by_category = products.groupby("category", observed=True).agg(
        products=("id", "size"),
        mean_attributes=("attribute_count", "mean"),
        median_attributes=("attribute_count", "median"),
        mean_title_only_token_share=("title_only_token_share", "mean"),
        products_with_title_only_identifier=(
            "title_identifiers_absent_from_attributes", lambda x: x.gt(0).sum()
        ),
        products_with_repeated_attribute_value_in_title=(
            "any_attribute_value_repeated_in_title", "sum"
        ),
    ).reset_index()
    by_category["title_only_identifier_share"] = (
        by_category["products_with_title_only_identifier"] / by_category["products"]
    )
    by_category["attribute_value_repeated_in_title_share"] = (
        by_category["products_with_repeated_attribute_value_in_title"]
        / by_category["products"]
    )
    return overall, by_category


def asymmetric_pairs(
    train: pd.DataFrame,
    products: pd.DataFrame,
    items: pd.DataFrame,
    examples_per_category: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    product_metrics = products.set_index("id")
    pairs = train[["id1", "id2", "target", "category"]].copy()
    for side in (1, 2):
        pairs[f"attributes_{side}"] = pairs[f"id{side}"].map(product_metrics["attribute_count"])
        pairs[f"title_chars_{side}"] = pairs[f"id{side}"].map(product_metrics["title_chars"])
        pairs[f"attribute_value_chars_{side}"] = pairs[f"id{side}"].map(product_metrics["attribute_value_chars"])
        pairs[f"detail_{side}"] = pairs[f"title_chars_{side}"] + pairs[f"attribute_value_chars_{side}"]
    category_limits = (
        products.groupby("category", observed=True)["title_chars"]
        .size()
        .rename("products")
        .to_frame()
    )
    detail_by_category = products.assign(
        detail=products["title_chars"] + products["attribute_value_chars"]
    ).groupby("category", observed=True)["detail"]
    category_limits["detail_p10"] = detail_by_category.quantile(0.10)
    category_limits["detail_p90"] = detail_by_category.quantile(0.90)
    low = pairs["category"].map(category_limits["detail_p10"])
    high = pairs["category"].map(category_limits["detail_p90"])
    pairs["strong_asymmetry"] = (
        (pairs["detail_1"].le(low) & pairs["detail_2"].ge(high))
        | (pairs["detail_2"].le(low) & pairs["detail_1"].ge(high))
    )
    pairs["detail_ratio"] = (
        pairs[["detail_1", "detail_2"]].max(axis=1) + 1
    ) / (pairs[["detail_1", "detail_2"]].min(axis=1) + 1)
    pairs["attribute_count_difference"] = (
        pairs["attributes_1"] - pairs["attributes_2"]
    ).abs()
    summary = {
        "strong_asymmetric_pairs": int(pairs["strong_asymmetry"].sum()),
        "strong_asymmetric_pair_share": float(pairs["strong_asymmetry"].mean()),
        "strong_asymmetric_positive_rate": float(
            pairs.loc[pairs["strong_asymmetry"], "target"].mean()
        ),
        "attribute_count_difference_p50": float(pairs["attribute_count_difference"].median()),
        "attribute_count_difference_p90": float(pairs["attribute_count_difference"].quantile(0.90)),
        "attribute_count_difference_p99": float(pairs["attribute_count_difference"].quantile(0.99)),
    }
    category_summary = pairs.groupby("category", observed=True).agg(
        pairs=("target", "size"),
        strong_asymmetric_pairs=("strong_asymmetry", "sum"),
        mean_attribute_count_difference=("attribute_count_difference", "mean"),
        p90_attribute_count_difference=("attribute_count_difference", lambda x: x.quantile(0.90)),
    ).reset_index()
    positive_rates = (
        pairs[pairs["strong_asymmetry"]]
        .groupby("category", observed=True)["target"]
        .mean()
    )
    category_summary["strong_asymmetric_share"] = (
        category_summary["strong_asymmetric_pairs"] / category_summary["pairs"]
    )
    category_summary["strong_asymmetric_prevalence"] = category_summary["category"].map(positive_rates)

    item_names = items.set_index("id")[["name"]]
    examples = (
        pairs[pairs["strong_asymmetry"]]
        .sort_values(["category", "detail_ratio"], ascending=[True, False])
        .groupby("category", observed=True)
        .head(examples_per_category)
        .copy()
    )
    examples["title_1"] = examples["id1"].map(item_names["name"])
    examples["title_2"] = examples["id2"].map(item_names["name"])
    return category_summary, examples, summary


def content_duplicate_diagnostics(
    items: pd.DataFrame, splits: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    item_lookup = items.set_index("id", verify_integrity=True)
    train = splits["train"]
    train_ids = set(train["id1"]) | set(train["id2"])
    overview_rows = []
    overlap_examples = []

    def identifier_boundary_title(value: Any) -> str:
        text = unicodedata.normalize("NFKC", str(value or "")).casefold().replace("ё", "е")
        text = re.sub(r"(?<=[A-Za-zА-Яа-яЁё])(?=\d)", " ", text)
        text = re.sub(r"(?<=\d)(?=[A-Za-zА-Яа-яЁё])", " ", text)
        return " ".join(SURFACE_KEY_RE.sub(" ", text).split())

    for signature_mode, title_normalizer in (
        ("project_surface", normalize_text),
        ("identifier_boundary_surface", identifier_boundary_title),
    ):
        signatures: dict[int, tuple[str, str, str]] = {
            int(row.id): (
                str(row.category),
                title_normalizer(row.name),
                str(row.attributes),
            )
            for row in items.itertuples(index=False)
        }
        signature_counts = Counter(signatures[item_id] for item_id in train_ids)
        repeated_signature_items = sum(
            count for count in signature_counts.values() if count > 1
        )
        repeated_signature_groups = sum(
            count > 1 for count in signature_counts.values()
        )

        def pair_signature(first: int, second: int) -> tuple[Any, Any]:
            left, right = signatures[int(first)], signatures[int(second)]
            return (left, right) if left <= right else (right, left)

        train_pair_labels: dict[tuple[Any, Any], set[int]] = defaultdict(set)
        train_pair_example: dict[tuple[Any, Any], Any] = {}
        for row in train.itertuples(index=False):
            signature = pair_signature(row.id1, row.id2)
            train_pair_labels[signature].add(int(row.target))
            train_pair_example.setdefault(signature, row)
        train_repeated_pair_groups = Counter(
            pair_signature(row.id1, row.id2) for row in train.itertuples(index=False)
        )
        overview_rows.append(
            {
                "signature_mode": signature_mode,
                "scope": "train",
                "unique_surface_full_product_signatures": len(signature_counts),
                "repeated_product_signature_groups": repeated_signature_groups,
                "products_in_repeated_signature_groups": repeated_signature_items,
                "unique_product_signatures_also_seen_in_train": len(signature_counts),
                "products_whose_signature_seen_in_train": len(train_ids),
                "pair_signatures_also_seen_in_train": sum(
                    count > 1 for count in train_repeated_pair_groups.values()
                ),
                "same_pair_signature_conflicting_labels": sum(
                    len(labels) > 1 for labels in train_pair_labels.values()
                ),
            }
        )
        train_signature_set = set(signature_counts)
        for name in ("ordinary", "hard", "ood"):
            pairs = splits[name]
            ids = set(pairs["id1"]) | set(pairs["id2"])
            validation_signature_counts = Counter(signatures[item_id] for item_id in ids)
            signatures_seen = set(validation_signature_counts) & train_signature_set
            pair_overlaps = 0
            same_label = 0
            conflicting_label = 0
            mixed_train_labels = 0
            for row in pairs.itertuples(index=False):
                signature = pair_signature(row.id1, row.id2)
                labels = train_pair_labels.get(signature)
                if labels is None:
                    continue
                pair_overlaps += 1
                if len(labels) > 1:
                    mixed_train_labels += 1
                    kind = "mixed_train_labels"
                elif int(row.target) in labels:
                    same_label += 1
                    kind = "same_label"
                else:
                    conflicting_label += 1
                    kind = "conflicting_label"
                if len(overlap_examples) < 200:
                    train_row = train_pair_example[signature]
                    overlap_examples.append(
                        {
                            "signature_mode": signature_mode,
                            "split": name,
                            "kind": kind,
                            "id1": int(row.id1),
                            "id2": int(row.id2),
                            "target": int(row.target),
                            "category": str(row.category),
                            "title_1": str(item_lookup.loc[row.id1, "name"]),
                            "title_2": str(item_lookup.loc[row.id2, "name"]),
                            "matched_train_id1": int(train_row.id1),
                            "matched_train_id2": int(train_row.id2),
                            "matched_train_target": int(train_row.target),
                            "matched_train_title_1": str(
                                item_lookup.loc[train_row.id1, "name"]
                            ),
                            "matched_train_title_2": str(
                                item_lookup.loc[train_row.id2, "name"]
                            ),
                        }
                    )
            overview_rows.append(
                {
                    "signature_mode": signature_mode,
                    "scope": name,
                    "unique_surface_full_product_signatures": len(validation_signature_counts),
                    "repeated_product_signature_groups": sum(
                        count > 1 for count in validation_signature_counts.values()
                    ),
                    "products_in_repeated_signature_groups": sum(
                        count
                        for count in validation_signature_counts.values()
                        if count > 1
                    ),
                    "unique_product_signatures_also_seen_in_train": len(signatures_seen),
                    "products_whose_signature_seen_in_train": sum(
                        validation_signature_counts[signature]
                        for signature in signatures_seen
                    ),
                    "pair_signatures_also_seen_in_train": pair_overlaps,
                    "same_pair_signature_conflicting_labels": conflicting_label,
                    "same_label_pair_signature_overlaps": same_label,
                    "mixed_train_label_pair_signature_overlaps": mixed_train_labels,
                }
            )
    return pd.DataFrame(overview_rows), pd.DataFrame(overlap_examples)


def build_report(
    *,
    train: pd.DataFrame,
    category: pd.DataFrame,
    category_quantiles: pd.DataFrame,
    split_table: pd.DataFrame,
    degree_distribution: pd.DataFrame,
    degree_quantiles: pd.DataFrame,
    component_diagnostics: ComponentDiagnostics,
    duplicate_summary: dict[str, int],
    content_duplicates: pd.DataFrame,
    attribute_summary: pd.DataFrame,
    global_attribute_names: pd.DataFrame,
    attribute_category_coverage: pd.DataFrame,
    category_overlap: pd.DataFrame,
    completeness_overall: pd.DataFrame,
    completeness_by_category: pd.DataFrame,
    completeness_global: dict[str, float],
    partition_integrity: dict[str, Any],
    asymmetry_summary: dict[str, Any],
    asymmetry_by_category: pd.DataFrame,
) -> str:
    positives = int(train["target"].sum())
    train_products = len(set(train["id1"]) | set(train["id2"]))
    attr_global = attribute_summary[attribute_summary["category"].eq("__ALL__")].iloc[0]
    top_keys = global_attribute_names.head(20).copy()
    top_keys["product_coverage"] = top_keys["product_coverage"].map(lambda x: f"{x:.2%}")
    category_view = category.copy()
    for column in ("prevalence", "share_of_train_pairs"):
        category_view[column] = category_view[column].map(lambda x: f"{x:.2%}")
    category_view["empirical_size_band"] = category_view["empirical_size_band"].replace(
        {"very rare": "очень малая", "rare": "малая", "medium": "средняя", "large": "крупная"}
    )
    split_view = split_table[[
        "split", "pairs", "unique_products", "positives", "negatives",
        "prevalence", "category_count", "item_id_overlap_with_train",
        "categories_absent_from_train",
    ]].copy()
    split_view["prevalence"] = split_view["prevalence"].map(lambda x: f"{x:.2%}")
    attr_view = attribute_summary[[
        "category", "products", "raw_attribute_names", "mean_attributes",
        "median_attributes", "p10_attributes", "p90_attributes",
        "products_without_attributes_share", "products_with_at_most_2_attributes_share",
    ]].copy()
    attr_view["category"] = attr_view["category"].replace({"__ALL__": "все категории"})
    for column in ("products_without_attributes_share", "products_with_at_most_2_attributes_share"):
        attr_view[column] = attr_view[column].map(lambda x: f"{x:.2%}")
    attr_view["mean_attributes"] = attr_view["mean_attributes"].map(lambda x: f"{x:.2f}")
    asym_view = asymmetry_by_category[[
        "category", "pairs", "strong_asymmetric_pairs", "strong_asymmetric_share",
        "strong_asymmetric_prevalence", "p90_attribute_count_difference",
    ]].copy()
    for column in ("strong_asymmetric_share", "strong_asymmetric_prevalence"):
        asym_view[column] = asym_view[column].map(
            lambda x: "" if pd.isna(x) else f"{x:.2%}"
        )
    smallest = category.head(5)[["category", "pairs", "positives", "unique_products"]]
    least_positive_support = category.nsmallest(5, "positives")[[
        "category", "pairs", "positives", "prevalence", "unique_products"
    ]].copy()
    least_positive_support["prevalence"] = least_positive_support["prevalence"].map(
        lambda x: f"{x:.2%}"
    )
    semantic_name_examples = pd.DataFrame(
        [
            {
                "possible_concept": "бренд / производитель",
                "raw_name_examples": "бренд; производитель",
            },
            {
                "possible_concept": "идентификатор товара",
                "raw_name_examples": "артикул; код товара; партномер; oem-номер",
            },
            {
                "possible_concept": "цвет",
                "raw_name_examples": "цвет; цвет товара; название цвета; оттенок",
            },
            {
                "possible_concept": "страна производства",
                "raw_name_examples": "страна-изготовитель; страна производства; страна-производитель",
            },
            {
                "possible_concept": "материал",
                "raw_name_examples": "материал; материал изделия; основной материал",
            },
            {
                "possible_concept": "объём",
                "raw_name_examples": "объем; объём; объем товара",
            },
            {
                "possible_concept": "вес",
                "raw_name_examples": "вес товара, г; вес, г",
            },
        ]
    )
    largest_component_share = component_diagnostics.summary["largest_component_node_share"]
    degree_one_share = float(
        degree_distribution.loc[degree_distribution["degree"].eq(1), "product_share"].sum()
    )
    test_content_overlap = content_duplicates[
        content_duplicates["scope"].isin(["ordinary", "hard", "ood"])
    ]
    report_column_names = {
        "category": "категория", "pairs": "пары", "unique_products": "уникальные товары",
        "positives": "положительные", "negatives": "отрицательные",
        "prevalence": "доля положительных",
        "share_of_train_pairs": "доля train", "empirical_size_band": "группа размера",
        "quantile": "квантиль", "positive_pairs": "положительные пары", "degree": "степень",
        "exact_order_duplicate_groups": "группы дублей A-B",
        "exact_order_duplicate_extra_rows": "лишние строки A-B",
        "unordered_duplicate_groups": "группы неупорядоченных дублей",
        "unordered_duplicate_extra_rows": "лишние неупорядоченные строки",
        "reversed_duplicate_groups": "группы A-B/B-A",
        "same_ids_conflicting_label_groups": "одинаковые ID с разными labels",
        "self_pairs": "пары товара с собой", "signature_mode": "режим сигнатуры",
        "scope": "набор", "unique_surface_full_product_signatures": "уникальные сигнатуры товаров",
        "repeated_product_signature_groups": "группы повторных сигнатур товаров",
        "products_in_repeated_signature_groups": "товары в повторных сигнатурах",
        "unique_product_signatures_also_seen_in_train": "сигнатуры, встречающиеся в train",
        "products_whose_signature_seen_in_train": "товары с сигнатурой из train",
        "pair_signatures_also_seen_in_train": "сигнатуры пар из train",
        "same_pair_signature_conflicting_labels": "пересечения с другим label",
        "same_label_pair_signature_overlaps": "пересечения с тем же label",
        "mixed_train_label_pair_signature_overlaps": "пересечения с несколькими labels в train",
        "products": "товары", "raw_attribute_names": "raw-названия attributes",
        "mean_attributes": "среднее attributes", "median_attributes": "медиана attributes",
        "p10_attributes": "p10 attributes", "p90_attributes": "p90 attributes",
        "products_without_attributes_share": "доля без attributes",
        "products_with_at_most_2_attributes_share": "доля с ≤2 attributes",
        "raw_attribute_name": "raw-название attribute",
        "product_occurrences": "товаров с attribute", "product_coverage": "охват товаров",
        "category_count": "число категорий", "share_of_raw_attribute_names": "доля raw-названий",
        "category_1": "категория 1", "category_2": "категория 2",
        "raw_names_category_1": "raw-названий в категории 1",
        "raw_names_category_2": "raw-названий в категории 2",
        "shared_raw_names": "общих raw-названий", "jaccard": "Jaccard",
        "possible_concept": "возможный концепт", "raw_name_examples": "примеры raw-названий",
        "metric": "метрика", "value": "значение",
        "strong_asymmetric_pairs": "сильно асимметричные пары",
        "strong_asymmetric_share": "доля асимметричных пар",
        "strong_asymmetric_prevalence": "доля положительных среди них",
        "p90_attribute_count_difference": "p90 разницы числа attributes",
        "split": "сплит", "item_id_overlap_with_train": "общих item_id с train",
        "categories_absent_from_train": "категории вне train",
    }

    def report_table(frame: pd.DataFrame) -> str:
        return markdown_table(frame.rename(columns=report_column_names))

    completeness_view = completeness_overall[
        completeness_overall["quantile"].isin(["p10", "p50", "p90", "p99", "mean"])
    ].copy()
    completeness_view["metric"] = completeness_view["metric"].replace(
        {
            "attribute_count": "число attributes",
            "title_chars": "число символов title",
            "attributes_chars": "число символов attributes",
            "attribute_value_chars": "число символов в значениях attributes",
            "title_meaningful_tokens": "значимые токены title",
            "title_only_meaningful_tokens": "значимые токены только в title",
            "title_only_token_share": "доля токенов только в title",
            "title_identifier_tokens": "идентификаторы в title",
            "title_identifiers_absent_from_attributes": "идентификаторы title вне attributes",
            "informative_attribute_values": "информативные значения attributes",
            "attribute_values_repeated_in_title": "значения attributes, повторённые в title",
            "attribute_value_repeat_share": "доля значений attributes, повторённых в title",
        }
    )
    completeness_view["quantile"] = completeness_view["quantile"].replace(
        {"mean": "среднее"}
    )
    partition_equal_text = (
        "да" if partition_integrity["combined_splits_equal_raw_pair_multiset"] else "нет"
    )
    category_quantiles_view = category_quantiles.copy()
    degree_quantiles_view = degree_quantiles.copy()
    for frame in (category_quantiles_view, degree_quantiles_view):
        frame["quantile"] = frame["quantile"].replace(
            {"min": "минимум", "max": "максимум", "mean": "среднее"}
        )
    report = f"""# Аудит данных для поиска правил: текущая human-разметка

Отчёт сформирован скриптом `scripts/analyze_rule_discovery_data.py`. Анализ
выполняется только на чтение: он не создаёт новое разбиение, не меняет
зафиксированные наборы,
не канонизирует raw-названия attributes и не вызывает LLM.

## A. Распределение по категориям

В текущей обучающей выборке (`current train`) содержится **{len(train):,} пар**, **{train_products:,} уникальных
товаров**, **{positives:,} положительных**, **{len(train) - positives:,} отрицательных**;
доля положительных — **{positives / len(train):.2%}**, число категорий — **{category.shape[0]}**.

`Группа размера` — описательное квартильное ранжирование внутри текущей выборки,
а не рекомендуемый порог для отложенной выборки.

{report_table(category_view[["category", "pairs", "unique_products", "positives", "negatives", "prevalence", "share_of_train_pairs", "empirical_size_band"]])}

Распределение размеров категорий:

{report_table(category_quantiles_view)}

## B. Распределение меток

Глобальный баланс классов приведён выше. Доля положительных пар по категориям меняется от
**{category["prevalence"].min():.2%}** до **{category["prevalence"].max():.2%}**.
Полные значения также сохранены в `label_distribution.csv`.

## C. Диагностика графа товаров

- Товары, встречающиеся ровно в одной паре: **{degree_one_share:.2%}**.
- Максимальная степень товара: **{int(degree_distribution["degree"].max())}**.
- Связных компонент: **{component_diagnostics.summary["components"]:,}**.
- Крупнейшая компонента: **{component_diagnostics.summary["largest_component_nodes"]} товаров / {component_diagnostics.summary["largest_component_edges"]} пар**
  ({largest_component_share:.4%} товаров train).
- Компонент крупнее 0.1% товаров train: **{component_diagnostics.summary["components_over_0_1pct_nodes"]}**.

Квантили степени товара:

{report_table(degree_quantiles_view)}

Крупнейшие компоненты сохранены в `connected_components.csv`, примеры товаров с
высокой степенью — в `top_products_by_degree.csv`.

## D. Дубликаты и конфликты

{report_table(pd.DataFrame([duplicate_summary]))}

Оба режима `signature_mode` сохраняют категорию и raw attributes точными. Они
различаются только нормализацией title: `project_surface` использует функцию
проекта, а `identifier_boundary_surface` дополнительно разделяет соседние буквы
и цифры. Это проверка чувствительности к скрытым повторам под разными ID, а не правило
канонизации attributes.

{report_table(content_duplicates)}

## E. Статистика атрибутов и полноты данных

В товарах current train найдено **{int(attr_global["raw_attribute_names"]):,} точных
raw-названий attributes**. Среднее число attributes —
**{float(attr_global["mean_attributes"]):.2f}**, медиана —
**{float(attr_global["median_attributes"]):.0f}**. Без attributes —
**{float(attr_global["products_without_attributes_share"]):.2%}** товаров, не
более двух attributes — **{float(attr_global["products_with_at_most_2_attributes_share"]):.2%}**.

{report_table(attr_view)}

Самые частые raw-названия attributes:

{report_table(top_keys[["raw_attribute_name", "product_occurrences", "product_coverage", "category_count"]])}

Охват raw-названий по числу категорий:

{report_table(attribute_category_coverage)}

Пары категорий с наибольшим пересечением словарей attributes:

{report_table(category_overlap.head(15))}

Эвристические кандидаты в алиасы сохранены в
`attribute_name_alias_candidates.csv`; исходные названия не изменялись.
Ниже приведены примеры raw-названий, которые могут обозначать общий семантический
концепт. Это материал для ручной проверки, а не принятое сопоставление:

{report_table(semantic_name_examples)}

### Название товара и атрибуты

`title_only_token_share` — лексическая верхняя оценка доли информации, которая
потенциально присутствует только в title. Значение attribute считается
повторённым в title только после нормализации и при длине не менее трёх символов.

- Средняя / медианная доля значимых токенов только в title: **{completeness_global["mean_title_only_token_share"]:.2%} / {completeness_global["median_title_only_token_share"]:.2%}**.
- Товары хотя бы с одним идентификатором из title, отсутствующим в attributes: **{completeness_global["products_with_title_only_identifier_share"]:.2%}**.
- Товары хотя бы с одним информативным значением attribute, повторённым в title: **{completeness_global["products_with_repeated_attribute_value_share"]:.2%}**.
- Доля повторений в title среди всех информативных значений attributes: **{completeness_global["attribute_value_occurrence_repeat_share"]:.2%}**.

{report_table(completeness_view)}

Разрез по категориям сохранён в `title_attribute_completeness_by_category.csv`.

Сильная асимметрия пары определяется через категориальные эмпирические p10/p90
метрики `символы title + символы значений attributes`, без фиксированного порога:

- сильно асимметричных пар: **{asymmetry_summary["strong_asymmetric_pairs"]:,}**
  ({asymmetry_summary["strong_asymmetric_pair_share"]:.2%});
- доля положительных среди них: **{asymmetry_summary["strong_asymmetric_positive_rate"]:.2%}**;
- p90/p99 абсолютной разницы числа attributes: **{asymmetry_summary["attribute_count_difference_p90"]:.0f} / {asymmetry_summary["attribute_count_difference_p99"]:.0f}**.

{report_table(asym_view)}

Репрезентативные строки сохранены в `asymmetric_pair_examples.csv`.

## F. Текущая обучающая выборка и зафиксированные ordinary/hard/OOD

{report_table(split_view)}

OOD-категории отсутствуют в current train. Пересечение item_id каждого из трёх
frozen evaluation-наборов с train равно нулю. Тесты исследовались только
структурно и не использовались для извлечения правил.

Проверка целостности frozen split’ов: их объединение содержит
**{partition_integrity["combined_split_pairs"]:,}** строк, исходная human-разметка —
**{partition_integrity["raw_pairs"]:,}** строк. Совпадение как мультимножества
строк: **{partition_equal_text}**.

## Выводы для следующего этапа проектирования разбиения

- **Редкость категорий.** Наименьшие категории current train показаны ниже.
  Полный диапазон — {int(category["pairs"].min()):,}–{int(category["pairs"].max()):,}
  пар. Квартильные группы относительны и пока не должны становиться рабочими порогами.

{report_table(smallest)}

  Общее число пар почти сбалансировано, но число положительных пар — нет. Категории
  с минимальным числом положительных примеров наиболее чувствительны к процентной
  отложенной выборке:

{report_table(least_positive_support)}

- **Разбиение без общих товаров (`product-disjoint`) структурно реалистично.** У {degree_one_share:.2%}
  товаров степень равна единице, крупнейшая компонента содержит только
  {component_diagnostics.summary["largest_component_nodes"]} товаров. Компоненты
  можно целиком распределять между частями без проблемы гигантской компоненты.
- **Утечка возможна и без общих ID.** Повторяющиеся полные представления товаров
  и пересечения сигнатур пар из раздела D могут перейти через границу без общих ID.
  границу. Будущий аудит должен проверять и компоненты графа, и точные сигнатуры
  видимого содержимого.
- **Attributes неоднородны.** Словари raw-названий пересекаются между категориями
  лишь частично и содержат поверхностные и потенциально семантические алиасы.
  В исходных свидетельствах следует сохранять raw-названия, отложив канонизацию
  до её проверки.
- **Полнота асимметрична.** Идентификаторы только в title и разрывы между p10/p90
  детализации требуют отличать реальный конфликт attributes от пропуска или
  другого расположения информации.
- **Frozen tests остаются неизменными.** Ordinary — основной тест на исходном
  распределении, hard — стресс-тест, OOD — перенос на две исключённые категории.
  Их нельзя использовать для поиска правил или проектирования внутреннего split.
"""
    return report


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    items, raw_pairs, splits, source_paths = load_inputs(
        args.data_dir.resolve(), args.split_dir.resolve()
    )
    train = splits["train"]
    train_ids = set(train["id1"]) | set(train["id2"])
    train_items = items[items["id"].isin(train_ids)].copy()
    if len(train_items) != len(train_ids):
        raise RuntimeError("Current train references missing products")

    print("Computing category and split summaries", flush=True)
    category, category_quantiles = category_distribution(train)
    labels = label_distribution(train)
    splits_table = split_summary(splits)
    partition_integrity = frozen_partition_integrity(raw_pairs, splits)

    print("Computing graph and duplicate diagnostics", flush=True)
    degree_distribution, degree_quantiles, top_products = product_degrees(
        train, items, args.top_products
    )
    components = connected_components(train)
    suspicious_pairs, duplicate_summary = pair_duplicates(train)
    content_duplicates, content_duplicate_examples = content_duplicate_diagnostics(
        items, splits
    )

    print(f"Parsing attributes for {len(train_items):,} current-train products", flush=True)
    (
        products,
        global_attribute_names,
        category_attribute_names,
        attribute_count_distribution,
        attribute_summary,
        category_overlap,
        alias_candidates,
        attribute_category_coverage,
    ) = attribute_and_completeness_diagnostics(train_items)
    completeness_overall, completeness_by_category = completeness_summary(products)
    completeness_global = {
        "mean_title_only_token_share": float(products["title_only_token_share"].mean()),
        "median_title_only_token_share": float(products["title_only_token_share"].median()),
        "products_with_title_only_identifier_share": float(
            products["title_identifiers_absent_from_attributes"].gt(0).mean()
        ),
        "products_with_repeated_attribute_value_share": float(
            products["any_attribute_value_repeated_in_title"].mean()
        ),
        "attribute_value_occurrence_repeat_share": float(
            products["attribute_values_repeated_in_title"].sum()
            / max(1, products["informative_attribute_values"].sum())
        ),
    }
    asymmetry_by_category, asymmetry_examples, asymmetry_summary = asymmetric_pairs(
        train, products, items, args.examples_per_category
    )

    tables = {
        "category_distribution.csv": category,
        "category_size_quantiles.csv": category_quantiles,
        "label_distribution.csv": labels,
        "split_structure.csv": splits_table,
        "product_degree_distribution.csv": degree_distribution,
        "product_degree_quantiles.csv": degree_quantiles,
        "top_products_by_degree.csv": top_products,
        # Exact counts remain in component_size_distribution.csv; retaining only
        # the largest rows keeps this inspection table compact.
        "connected_components.csv": components.components.head(1000),
        "component_size_distribution.csv": components.size_distribution,
        "duplicate_pair_examples.csv": suspicious_pairs,
        "content_duplicate_diagnostics.csv": content_duplicates,
        "content_duplicate_examples.csv": content_duplicate_examples,
        "attribute_name_global.csv": global_attribute_names,
        "attribute_name_by_category.csv": category_attribute_names,
        "attribute_count_distribution.csv": attribute_count_distribution,
        "attribute_summary_by_category.csv": attribute_summary,
        "attribute_category_overlap.csv": category_overlap,
        "attribute_name_category_coverage.csv": attribute_category_coverage,
        "attribute_name_alias_candidates.csv": alias_candidates,
        "title_attribute_completeness_quantiles.csv": completeness_overall,
        "title_attribute_completeness_by_category.csv": completeness_by_category,
        "pair_asymmetry_by_category.csv": asymmetry_by_category,
        "asymmetric_pair_examples.csv": asymmetry_examples,
    }
    for filename, frame in tables.items():
        write_csv(frame, output_dir / filename)

    source_hashes = {
        name: {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for name, path in source_paths.items()
    }
    summary = {
        "schema_version": 1,
        "scope": "текущий human train; frozen tests исследованы только структурно",
        "source_files": source_hashes,
        "current_train": {
            "pairs": len(train),
            "unique_products": len(train_ids),
            "positives": int(train["target"].sum()),
            "negatives": int(train["target"].eq(0).sum()),
            "prevalence": float(train["target"].mean()),
            "categories": int(train["category"].nunique()),
        },
        "duplicate_pairs": duplicate_summary,
        "frozen_partition_integrity": partition_integrity,
        "title_attribute_completeness": completeness_global,
        "components": components.summary,
        "asymmetry": asymmetry_summary,
        "outputs": sorted(tables),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_value),
        encoding="utf-8",
    )
    report = build_report(
        train=train,
        category=category,
        category_quantiles=category_quantiles,
        split_table=splits_table,
        degree_distribution=degree_distribution,
        degree_quantiles=degree_quantiles,
        component_diagnostics=components,
        duplicate_summary=duplicate_summary,
        content_duplicates=content_duplicates,
        attribute_summary=attribute_summary,
        global_attribute_names=global_attribute_names,
        attribute_category_coverage=attribute_category_coverage,
        category_overlap=category_overlap,
        completeness_overall=completeness_overall,
        completeness_by_category=completeness_by_category,
        completeness_global=completeness_global,
        partition_integrity=partition_integrity,
        asymmetry_summary=asymmetry_summary,
        asymmetry_by_category=asymmetry_by_category,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary["current_train"], ensure_ascii=False, indent=2), flush=True)
    print(f"Saved audit to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
