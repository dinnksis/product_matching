"""Create a deterministic rule-discovery split inside the frozen current train.

The script compares four grouping policies on identical category-aware quotas,
then materializes only the explicitly selected policy. Existing ordinary, hard,
and OOD files are read only for integrity checks and never influence quotas,
assignment, or strategy selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from analyze_rule_discovery_data import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_SPLIT_DIR,
    load_inputs,
    markdown_table,
    normalize_text,
    sha256,
    write_csv,
)


DEFAULT_AUDIT_DIR = ROOT / "reports" / "rule_discovery_data_audit"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "rule_discovery_split_v1"
DEFAULT_REPORT_DIR = ROOT / "reports" / "rule_discovery_split_v1"
DEFAULT_SEED = 2026
STRATEGIES = (
    "random_pair_grouped",
    "content_grouped",
    "product_disjoint",
    "product_content_disjoint",
)
BAND_ORDER = ("very rare", "rare", "medium", "large")
BAND_RATES = {"very rare": 0.09, "rare": 0.10, "medium": 0.11, "large": 0.12}
BAND_RU = {
    "very rare": "очень малая",
    "rare": "малая",
    "medium": "средняя",
    "large": "крупная",
}


@dataclass(frozen=True)
class Candidate:
    strategy: str
    assignment: np.ndarray
    overall: dict[str, Any]
    by_category: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the internal rule-discovery split inside current train."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--selected-strategy", choices=STRATEGIES, default="product_content_disjoint"
    )
    return parser.parse_args()


def stable_pair_ids(train: pd.DataFrame) -> pd.Series:
    ids = [
        "rp_" + hashlib.sha256(
            f"human_train_v1|{int(row.id1)}|{int(row.id2)}|{int(row.target)}".encode()
        ).hexdigest()[:24]
        for row in train.itertuples(index=False)
    ]
    result = pd.Series(ids, index=train.index, name="pair_id")
    if result.duplicated().any():
        raise RuntimeError(
            "Stable pair IDs are not unique; source contains exact ordered duplicate rows"
        )
    return result


def load_audit_categories(audit_dir: Path, train: pd.DataFrame) -> pd.DataFrame:
    path = audit_dir / "category_distribution.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run the previous data audit first: {path}")
    audited = pd.read_csv(path)
    required = {"category", "pairs", "positives", "empirical_size_band"}
    if not required.issubset(audited.columns):
        raise ValueError(f"{path}: missing columns {sorted(required - set(audited.columns))}")
    actual = (
        train.groupby("category", observed=True)["target"]
        .agg(pairs="size", positives="sum")
        .reset_index()
    )
    checked = audited.merge(actual, on="category", suffixes=("_audit", "_actual"))
    if len(checked) != train["category"].nunique() or not (
        checked["pairs_audit"].eq(checked["pairs_actual"])
        & checked["positives_audit"].eq(checked["positives_actual"])
    ).all():
        raise RuntimeError("Previous category audit does not match current train")
    result = audited[["category", "pairs", "positives", "empirical_size_band"]].copy()
    result["positive_support_band"] = pd.qcut(
        result["positives"].rank(method="first"),
        q=4,
        labels=BAND_ORDER,
    ).astype(str)
    result["pair_band_rate"] = result["empirical_size_band"].map(BAND_RATES)
    result["positive_support_cap"] = result["positive_support_band"].map(BAND_RATES)
    result["validation_rate"] = result[["pair_band_rate", "positive_support_cap"]].min(axis=1)
    result["target_validation_pairs"] = (
        result["pairs"] * result["validation_rate"]
    ).round().astype(int)
    result["target_validation_positives"] = (
        result["positives"] * result["validation_rate"]
    ).round().astype(int)
    result["target_validation_negatives"] = (
        result["target_validation_pairs"] - result["target_validation_positives"]
    )
    result["rationale"] = result.apply(category_rationale, axis=1)
    return result.sort_values("pairs").reset_index(drop=True)


def category_rationale(row: pd.Series) -> str:
    pair_band = BAND_RU[str(row["empirical_size_band"])]
    positive_band = BAND_RU[str(row["positive_support_band"])]
    if row["positive_support_cap"] < row["pair_band_rate"]:
        return (
            f"Категория {pair_band} по числу пар, но {positive_band} по positive support; "
            "validation уменьшена, чтобы сохранить положительные discovery-примеры."
        )
    return (
        f"Категория {pair_band} по числу пар; positive support не требует "
        "дополнительного уменьшения validation."
    )


def factorized_pair_keys(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    low = np.minimum(left, right)
    high = np.maximum(left, right)
    keys = pd.MultiIndex.from_arrays([low, high])
    return pd.factorize(keys, sort=True)[0].astype(np.int64)


def exact_pair_group_codes(train: pd.DataFrame) -> np.ndarray:
    return factorized_pair_keys(
        train["id1"].to_numpy(dtype=np.int64),
        train["id2"].to_numpy(dtype=np.int64),
    )


def content_group_codes(items: pd.DataFrame, train: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
    train_ids = pd.Index(pd.unique(pd.concat([train["id1"], train["id2"]], ignore_index=True)))
    selected = items[items["id"].isin(train_ids)][["id", "category", "name", "attributes"]].copy()
    selected["normalized_title"] = selected["name"].map(normalize_text)
    signature_index = pd.MultiIndex.from_frame(
        selected[["category", "normalized_title", "attributes"]]
    )
    selected["product_signature_code"] = pd.factorize(signature_index, sort=True)[0]
    code_by_id = selected.set_index("id", verify_integrity=True)["product_signature_code"]
    left = train["id1"].map(code_by_id).to_numpy(dtype=np.int64)
    right = train["id2"].map(code_by_id).to_numpy(dtype=np.int64)
    if (left < 0).any() or (right < 0).any():
        raise RuntimeError("Missing content signature for a train product")
    return factorized_pair_keys(left, right), code_by_id


def product_component_codes(train: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
    product_ids = pd.Index(
        pd.unique(pd.concat([train["id1"], train["id2"]], ignore_index=True))
    )
    position = pd.Series(np.arange(len(product_ids), dtype=np.int64), index=product_ids)
    left = train["id1"].map(position).to_numpy(dtype=np.int64)
    right = train["id2"].map(position).to_numpy(dtype=np.int64)
    parent = np.arange(len(product_ids), dtype=np.int64)
    size = np.ones(len(product_ids), dtype=np.int32)

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = int(parent[root])
        while parent[node] != node:
            next_node = int(parent[node])
            parent[node] = root
            node = next_node
        return root

    for first, second in zip(left, right):
        root_first, root_second = find(int(first)), find(int(second))
        if root_first == root_second:
            continue
        if size[root_first] < size[root_second]:
            root_first, root_second = root_second, root_first
        parent[root_second] = root_first
        size[root_first] += size[root_second]
    roots = np.fromiter(
        (find(node) for node in range(len(product_ids))),
        dtype=np.int64,
        count=len(product_ids),
    )
    component_by_id = pd.Series(
        pd.factorize(roots, sort=True)[0].astype(np.int64),
        index=product_ids,
        name="component_code",
    )
    row_codes = train["id1"].map(component_by_id).to_numpy(dtype=np.int64)
    return row_codes, component_by_id


def product_content_component_codes(
    component_codes: np.ndarray,
    component_by_id: pd.Series,
    product_signature_by_id: pd.Series,
) -> np.ndarray:
    component_count = int(component_by_id.max()) + 1
    parent = np.arange(component_count, dtype=np.int64)
    size = np.ones(component_count, dtype=np.int32)

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = int(parent[root])
        while parent[node] != node:
            next_node = int(parent[node])
            parent[node] = root
            node = next_node
        return root

    links = pd.DataFrame(
        {
            "component": component_by_id.loc[product_signature_by_id.index].to_numpy(
                dtype=np.int64
            ),
            "signature": product_signature_by_id.to_numpy(dtype=np.int64),
        }
    ).drop_duplicates().sort_values(["signature", "component"])
    previous_signature: int | None = None
    previous_component: int | None = None
    for row in links.itertuples(index=False):
        signature = int(row.signature)
        component = int(row.component)
        if signature == previous_signature and previous_component is not None:
            first, second = find(previous_component), find(component)
            if first != second:
                if size[first] < size[second]:
                    first, second = second, first
                parent[second] = first
                size[first] += size[second]
        previous_signature = signature
        previous_component = component
    merged_roots = np.fromiter(
        (find(component) for component in range(component_count)),
        dtype=np.int64,
        count=component_count,
    )
    return pd.factorize(merged_roots[component_codes], sort=True)[0].astype(np.int64)


def group_summary(train: pd.DataFrame, group_codes: np.ndarray) -> pd.DataFrame:
    work = train[["category", "target"]].copy()
    work["group_code"] = group_codes
    category_counts = work.groupby("group_code", observed=True)["category"].nunique()
    if category_counts.gt(1).any():
        raise RuntimeError("A grouping unit crosses category boundaries")
    return (
        work.groupby("group_code", observed=True)
        .agg(category=("category", "first"), pairs=("target", "size"), positives=("target", "sum"))
        .reset_index()
    )


def stable_category_seed(seed: int, strategy: str, category: str) -> int:
    digest = hashlib.sha256(f"{seed}|{strategy}|{category}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def select_validation_groups(
    groups: pd.DataFrame,
    target_pairs: int,
    target_positives: int,
    seed: int,
) -> set[int]:
    target_negatives = target_pairs - target_positives

    def objective(pairs: int, positives: int) -> float:
        negatives = pairs - positives
        return (
            ((pairs - target_pairs) / max(1, target_pairs)) ** 2
            + ((positives - target_positives) / max(1, target_positives)) ** 2
            + ((negatives - target_negatives) / max(1, target_negatives)) ** 2
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    selected: set[int] = set()
    current_pairs = 0
    current_positives = 0
    current_objective = objective(0, 0)
    for position in order:
        row = groups.iloc[int(position)]
        proposed_pairs = current_pairs + int(row["pairs"])
        proposed_positives = current_positives + int(row["positives"])
        proposed_objective = objective(proposed_pairs, proposed_positives)
        if proposed_objective + 1e-15 < current_objective:
            selected.add(int(row["group_code"]))
            current_pairs = proposed_pairs
            current_positives = proposed_positives
            current_objective = proposed_objective

    # Multi-edge groups make the greedy result slightly over/undershoot a quota.
    # Single-edge groups are themselves complete grouping units, so use them as a
    # deterministic exact correction without breaking duplicate/content/component
    # constraints or discarding any row.
    def correct_single_edge_stratum(positive: bool, target: int) -> None:
        nonlocal current_pairs, current_positives
        current = current_positives if positive else current_pairs - current_positives
        delta = target - current
        if delta == 0:
            return
        singleton = groups[
            groups["pairs"].eq(1) & groups["positives"].eq(1 if positive else 0)
        ]
        if delta > 0:
            eligible = singleton[~singleton["group_code"].isin(selected)]
        else:
            eligible = singleton[singleton["group_code"].isin(selected)]
        eligible_codes = eligible["group_code"].to_numpy(dtype=np.int64)
        if len(eligible_codes) < abs(delta):
            return
        chosen = rng.permutation(eligible_codes)[: abs(delta)]
        if delta > 0:
            selected.update(int(code) for code in chosen)
            current_pairs += len(chosen)
            current_positives += len(chosen) if positive else 0
        else:
            selected.difference_update(int(code) for code in chosen)
            current_pairs -= len(chosen)
            current_positives -= len(chosen) if positive else 0

    correct_single_edge_stratum(True, target_positives)
    correct_single_edge_stratum(False, target_negatives)
    return selected


def assignment_for_strategy(
    train: pd.DataFrame,
    group_codes: np.ndarray,
    quotas: pd.DataFrame,
    strategy: str,
    seed: int,
) -> np.ndarray:
    groups = group_summary(train, group_codes)
    validation_groups: set[int] = set()
    quota_by_category = quotas.set_index("category")
    for category, category_groups in groups.groupby("category", observed=True):
        quota = quota_by_category.loc[category]
        validation_groups.update(
            select_validation_groups(
                category_groups.reset_index(drop=True),
                int(quota["target_validation_pairs"]),
                int(quota["target_validation_positives"]),
                stable_category_seed(seed, strategy, str(category)),
            )
        )
    return np.isin(group_codes, np.fromiter(validation_groups, dtype=np.int64))


def product_sets(train: pd.DataFrame, validation_mask: np.ndarray) -> tuple[set[int], set[int]]:
    discovery = train.loc[~validation_mask]
    validation = train.loc[validation_mask]
    discovery_ids = set(discovery["id1"]) | set(discovery["id2"])
    validation_ids = set(validation["id1"]) | set(validation["id2"])
    return discovery_ids, validation_ids


def evaluate_candidate(
    strategy: str,
    train: pd.DataFrame,
    validation_mask: np.ndarray,
    strategy_group_codes: np.ndarray,
    quotas: pd.DataFrame,
    exact_codes: np.ndarray,
    content_codes: np.ndarray,
    product_signature_by_id: pd.Series,
) -> Candidate:
    discovery = train.loc[~validation_mask]
    validation = train.loc[validation_mask]
    discovery_ids, validation_ids = product_sets(train, validation_mask)
    discovery_content = set(content_codes[~validation_mask])
    validation_content = set(content_codes[validation_mask])
    discovery_exact = set(exact_codes[~validation_mask])
    validation_exact = set(exact_codes[validation_mask])
    discovery_product_signatures = set(product_signature_by_id.loc[list(discovery_ids)])
    validation_product_signatures = set(product_signature_by_id.loc[list(validation_ids)])
    strategy_group_sizes = pd.Series(strategy_group_codes).value_counts()

    category_rows = []
    quota_by_category = quotas.set_index("category")
    for category in quotas["category"]:
        left = discovery[discovery["category"].eq(category)]
        right = validation[validation["category"].eq(category)]
        quota = quota_by_category.loc[category]
        category_rows.append(
            {
                "strategy": strategy,
                "category": category,
                "pair_size_band": quota["empirical_size_band"],
                "positive_support_band": quota["positive_support_band"],
                "validation_rate_target": quota["validation_rate"],
                "target_validation_pairs": int(quota["target_validation_pairs"]),
                "discovery_pairs": len(left),
                "validation_pairs": len(right),
                "validation_pair_error": len(right) - int(quota["target_validation_pairs"]),
                "discovery_positives": int(left["target"].sum()),
                "validation_positives": int(right["target"].sum()),
                "target_validation_positives": int(quota["target_validation_positives"]),
                "validation_positive_error": int(right["target"].sum())
                - int(quota["target_validation_positives"]),
                "discovery_prevalence": float(left["target"].mean()),
                "validation_prevalence": float(right["target"].mean()),
                "source_prevalence": float(
                    train.loc[train["category"].eq(category), "target"].mean()
                ),
                "rationale": quota["rationale"],
            }
        )
    by_category = pd.DataFrame(category_rows)
    overall = {
        "strategy": strategy,
        "grouping_units": int(strategy_group_sizes.size),
        "largest_group_pairs": int(strategy_group_sizes.max()),
        "source_pairs": len(train),
        "discovery_pairs": len(discovery),
        "validation_pairs": len(validation),
        "validation_share": len(validation) / len(train),
        "data_loss_pairs": len(train) - len(discovery) - len(validation),
        "discovery_positives": int(discovery["target"].sum()),
        "validation_positives": int(validation["target"].sum()),
        "discovery_prevalence": float(discovery["target"].mean()),
        "validation_prevalence": float(validation["target"].mean()),
        "discovery_categories": int(discovery["category"].nunique()),
        "validation_categories": int(validation["category"].nunique()),
        "discovery_unique_products": len(discovery_ids),
        "validation_unique_products": len(validation_ids),
        "product_id_overlap": len(discovery_ids & validation_ids),
        "product_signature_overlap": len(
            discovery_product_signatures & validation_product_signatures
        ),
        "exact_pair_group_overlap": len(discovery_exact & validation_exact),
        "content_pair_signature_overlap": len(discovery_content & validation_content),
        "absolute_validation_pair_target_error": int(
            by_category["validation_pair_error"].abs().sum()
        ),
        "absolute_validation_positive_target_error": int(
            by_category["validation_positive_error"].abs().sum()
        ),
        "max_category_prevalence_delta": float(
            (by_category["validation_prevalence"] - by_category["source_prevalence"])
            .abs()
            .max()
        ),
        "min_very_rare_validation_pairs": int(
            by_category.loc[
                by_category["pair_size_band"].eq("very rare"), "validation_pairs"
            ].min()
        ),
        "min_very_rare_validation_positives": int(
            by_category.loc[
                by_category["pair_size_band"].eq("very rare"), "validation_positives"
            ].min()
        ),
    }
    return Candidate(strategy, validation_mask, overall, by_category)


def duplicate_conflicts(train: pd.DataFrame, pair_ids: pd.Series) -> pd.DataFrame:
    work = train[["id1", "id2", "target", "category"]].copy()
    work["pair_id"] = pair_ids
    work["canonical_id_1"] = np.minimum(work["id1"], work["id2"])
    work["canonical_id_2"] = np.maximum(work["id1"], work["id2"])
    conflict_keys = (
        work.groupby(["canonical_id_1", "canonical_id_2"], observed=True)["target"]
        .nunique()
        .loc[lambda x: x.gt(1)]
        .index
    )
    if len(conflict_keys) == 0:
        return pd.DataFrame(
            columns=[
                "pair_id", "id1", "id2", "target", "category",
                "canonical_id_1", "canonical_id_2",
            ]
        )
    key_index = pd.MultiIndex.from_frame(work[["canonical_id_1", "canonical_id_2"]])
    return work[key_index.isin(conflict_keys)].sort_values(
        ["canonical_id_1", "canonical_id_2", "target"]
    )


def report_table(frame: pd.DataFrame, rename: dict[str, str] | None = None) -> str:
    shown = frame.rename(columns=rename or {})
    return markdown_table(shown)


def build_report(
    selected: Candidate,
    comparisons: pd.DataFrame,
    selected_categories: pd.DataFrame,
    quotas: pd.DataFrame,
    seed: int,
    source_hash: str,
    frozen_hashes: dict[str, str],
    conflict_count: int,
) -> str:
    comparison_columns = {
        "strategy": "вариант", "grouping_units": "групп",
        "largest_group_pairs": "макс. пар в группе",
        "discovery_pairs": "discovery-пары",
        "validation_pairs": "validation-пары", "validation_share": "доля validation",
        "data_loss_pairs": "потеряно пар", "discovery_prevalence": "доля positive discovery",
        "validation_prevalence": "доля positive validation",
        "discovery_categories": "категорий discovery",
        "validation_categories": "категорий validation",
        "discovery_unique_products": "уникальных товаров discovery",
        "validation_unique_products": "уникальных товаров validation",
        "product_id_overlap": "общих product ID",
        "product_signature_overlap": "общих сигнатур товаров",
        "content_pair_signature_overlap": "общих сигнатур пар",
        "absolute_validation_pair_target_error": "суммарная ошибка квоты пар",
        "absolute_validation_positive_target_error": "суммарная ошибка квоты positive",
        "max_category_prevalence_delta": "макс. сдвиг prevalence категории",
        "min_very_rare_validation_pairs": "мин. validation-пар в very rare",
        "min_very_rare_validation_positives": "мин. validation-positive в very rare",
    }
    comparison_view = comparisons[list(comparison_columns)].copy()
    for column in ("validation_share", "discovery_prevalence", "validation_prevalence", "max_category_prevalence_delta"):
        comparison_view[column] = comparison_view[column].map(lambda value: f"{value:.2%}")

    category_columns = {
        "category": "категория", "pair_size_band": "группа размера",
        "positive_support_band": "группа positive support",
        "validation_rate_target": "целевая доля validation",
        "discovery_pairs": "discovery-пары", "validation_pairs": "validation-пары",
        "discovery_positives": "positive discovery", "validation_positives": "positive validation",
        "discovery_prevalence": "доля positive discovery",
        "validation_prevalence": "доля positive validation", "rationale": "обоснование",
    }
    category_view = selected_categories[list(category_columns)].copy()
    category_view["pair_size_band"] = category_view["pair_size_band"].map(BAND_RU)
    category_view["positive_support_band"] = category_view["positive_support_band"].map(BAND_RU)
    for column in ("validation_rate_target", "discovery_prevalence", "validation_prevalence"):
        category_view[column] = category_view[column].map(lambda value: f"{value:.2%}")

    overall = selected.overall
    random_discovery_products = int(
        comparisons.loc[
            comparisons["strategy"].eq("random_pair_grouped"),
            "discovery_unique_products",
        ].iloc[0]
    )
    discovery_product_difference = random_discovery_products - int(
        overall["discovery_unique_products"]
    )
    return f"""# Разбиение current train для поиска правил

Скрипт: `scripts/create_rule_discovery_split.py`. Seed: **{seed}**. Выбранная
стратегия: **`{selected.strategy}`**. Исходный SHA-256 current train:
`{source_hash}`.

## Выбранная стратегия

Выбрано разбиение по целым компонентам графа товаров, дополнительно объединённым
по точной видимой сигнатуре товара (`product_content_disjoint`). Оно развивает
product-disjoint вариант с учётом скрытых дублей под разными ID, найденных в
предыдущем аудите. Крупнейшая итоговая grouping unit содержит
**{overall["largest_group_pairs"]} пар**. Фактическое сравнение подтверждает, что
эта стратегия:

- не теряет ни одной пары и сохраняет все 18 категорий в обеих частях;
- даёт **{overall["validation_share"]:.2%}** данных во внутреннюю validation;
- обеспечивает нулевое пересечение product ID, сигнатур товаров и сигнатур пар;
- точно воспроизводит category/label quotas;
- сохраняет **{overall["discovery_pairs"]:,}** из **{overall["source_pairs"]:,}**
  пар для поиска правил.

Для проверки воспроизводимости правил независимость товаров полезна, а разреженный
граф не превращает internal validation в искусственно сложный benchmark. По
сравнению с random-вариантом discovery содержит на **{discovery_product_difference:,}**
меньше уникальных товаров, при одинаковом числе пар; это и есть измеримая цена
полной ID/content-независимости.

## Как определены category-aware квоты

Использованы квартильные группы из предыдущего data audit, без новых абсолютных
порогов редкости. Базовая доля validation равна 9% / 10% / 11% / 12% для групп
от «очень малой» до «крупной». Затем она ограничивается таким же квартильным
рангом positive support. Поэтому большая категория с малым числом positive не
теряет непропорционально много полезных discovery-примеров.

## Сравнение вариантов

- `random_pair_grouped`: случайное category/label-stratified разбиение с
  совместным размещением A-B/B-A и exact duplicate rows.
- `content_grouped`: то же, но совместно размещаются пары с одинаковым видимым
  содержимым товаров после поверхностной нормализации title.
- `product_disjoint`: целиком распределяются connected components pair graph.
- `product_content_disjoint`: product components дополнительно объединяются по
  точной сигнатуре товара; это устраняет и ID-, и видимую content-утечку.

{report_table(comparison_view, comparison_columns)}

Во всех вариантах `data_loss_pairs = 0`: строки не выбрасываются, меняется только
независимость частей. Exact/reversed duplicate group не пересекает границу ни в
одном варианте.

## Итоговая статистика по категориям

{report_table(category_view, category_columns)}

## Дубликаты и конфликты

Exact и reversed duplicates группируются до назначения split. Конфликтующие
пары не удаляются и получают отдельный файл `duplicate_pair_conflicts.csv`.
Найдено конфликтующих строк: **{conflict_count}**.

## Сохранённые артефакты

- `rule_discovery_pair_ids.csv` — детерминированные ID discovery-пар;
- `rule_internal_validation_pair_ids.csv` — ID internal-validation пар;
- `split_assignments.parquet` — ID, исходные поля пары, категория и назначение;
- `rule_discovery_pairs.parquet` и `rule_internal_validation_pairs.parquet` —
  полные пары для существующих loaders;
- `manifest.json` — seed, стратегия, квоты, source/test hashes и размеры;
- `candidate_comparison.csv` и `candidate_category_statistics.csv` — полное
  сравнение четырёх вариантов.

## Защита ordinary/hard/OOD

Ordinary, hard и OOD не участвовали в расчёте квот, группировке или выборе
стратегии. Их файлы не изменялись; контрольные SHA-256 записаны в manifest:

{report_table(pd.DataFrame([{"набор": name, "SHA-256": digest} for name, digest in frozen_hashes.items()]))}

OOD по-прежнему содержит две unseen-категории — «Бытовая техника» и «Одежда» —
и остаётся полностью untouched.

## Зафиксированный OOD-протокол

### Фаза 1

1. Разрабатывать pipeline только на `rule_discovery` из current train.
2. Зафиксировать методику extraction/rules/statistics.
3. Проверять воспроизводимость на `rule_internal_validation`.
4. После фиксации методики один раз проверить перенос global rules на untouched OOD.

### Фаза 2 — сейчас не выполняется

После проверки переноса разрешается использовать две OOD-категории для получения
category-specific rules. Если данных достаточно, внутри каждой категории следует
сначала выделить небольшой final untouched subset, а оставшиеся строки использовать
для rule extraction. Размер этого subset на текущем этапе не выбирается.

Ни ordinary, ни hard, ни OOD нельзя использовать для выбора rule thresholds,
min support или global/category criteria на текущей фазе.
"""


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    report_dir = args.report_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    items, _, splits, paths = load_inputs(args.data_dir.resolve(), args.split_dir.resolve())
    train = splits["train"].reset_index(drop=True)
    pair_ids = stable_pair_ids(train)
    quotas = load_audit_categories(args.audit_dir.resolve(), train)

    frozen_names = ("ordinary", "hard", "ood")
    frozen_hashes_before = {name: sha256(paths[name]) for name in frozen_names}
    train_ids = set(train["id1"]) | set(train["id2"])
    for name in frozen_names:
        test_ids = set(splits[name]["id1"]) | set(splits[name]["id2"])
        if train_ids & test_ids:
            raise RuntimeError(f"{name}: item IDs overlap current train")

    print("Строю grouping units для четырёх вариантов", flush=True)
    exact_codes = exact_pair_group_codes(train)
    content_codes, product_signature_by_id = content_group_codes(items, train)
    component_codes, component_by_id = product_component_codes(train)
    product_content_codes = product_content_component_codes(
        component_codes, component_by_id, product_signature_by_id
    )
    group_codes = {
        "random_pair_grouped": exact_codes,
        "content_grouped": content_codes,
        "product_disjoint": component_codes,
        "product_content_disjoint": product_content_codes,
    }

    candidates: list[Candidate] = []
    for strategy in STRATEGIES:
        print(f"Оцениваю {strategy}", flush=True)
        validation_mask = assignment_for_strategy(
            train, group_codes[strategy], quotas, strategy, args.seed
        )
        candidates.append(
            evaluate_candidate(
                strategy,
                train,
                validation_mask,
                group_codes[strategy],
                quotas,
                exact_codes,
                content_codes,
                product_signature_by_id,
            )
        )
    selected = next(candidate for candidate in candidates if candidate.strategy == args.selected_strategy)
    if selected.overall["data_loss_pairs"] != 0:
        raise RuntimeError("Selected strategy loses source pairs")
    if selected.overall["discovery_categories"] != train["category"].nunique() or selected.overall[
        "validation_categories"
    ] != train["category"].nunique():
        raise RuntimeError("Selected strategy loses category coverage")
    if selected.by_category[["validation_pair_error", "validation_positive_error"]].ne(0).any().any():
        raise RuntimeError("Selected strategy missed category/label quotas")
    if "product" in selected.strategy and selected.overall["product_id_overlap"] != 0:
        raise RuntimeError("Selected product-disjoint strategy leaks product IDs")
    if selected.strategy == "product_content_disjoint" and (
        selected.overall["product_signature_overlap"] != 0
        or selected.overall["content_pair_signature_overlap"] != 0
    ):
        raise RuntimeError("Selected hybrid strategy leaks exact visible content")
    comparisons = pd.DataFrame([candidate.overall for candidate in candidates])
    category_comparisons = pd.concat(
        [candidate.by_category for candidate in candidates], ignore_index=True
    )

    conflict_rows = duplicate_conflicts(train, pair_ids)
    assignment = train[["id1", "id2", "target", "category"]].copy()
    assignment.insert(0, "pair_id", pair_ids)
    assignment["split"] = np.where(
        selected.assignment, "rule_internal_validation", "rule_discovery"
    )
    conflict_ids = set(conflict_rows["pair_id"])
    assignment["duplicate_label_conflict"] = assignment["pair_id"].isin(conflict_ids)
    discovery = assignment[assignment["split"].eq("rule_discovery")].copy()
    validation = assignment[assignment["split"].eq("rule_internal_validation")].copy()
    if len(discovery) + len(validation) != len(train):
        raise RuntimeError("Selected split does not partition current train")
    if set(discovery["pair_id"]) & set(validation["pair_id"]):
        raise RuntimeError("Pair IDs overlap selected subsets")

    write_csv(discovery[["pair_id"]], output_dir / "rule_discovery_pair_ids.csv")
    write_csv(validation[["pair_id"]], output_dir / "rule_internal_validation_pair_ids.csv")
    assignment.to_parquet(output_dir / "split_assignments.parquet", index=False)
    discovery[["id1", "id2", "target"]].to_parquet(
        output_dir / "rule_discovery_pairs.parquet", index=False
    )
    validation[["id1", "id2", "target"]].to_parquet(
        output_dir / "rule_internal_validation_pairs.parquet", index=False
    )
    write_csv(conflict_rows, output_dir / "duplicate_pair_conflicts.csv")
    write_csv(comparisons, report_dir / "candidate_comparison.csv")
    write_csv(category_comparisons, report_dir / "candidate_category_statistics.csv")
    write_csv(quotas, report_dir / "category_validation_quotas.csv")
    write_csv(selected.by_category, report_dir / "selected_split_by_category.csv")

    frozen_hashes_after = {name: sha256(paths[name]) for name in frozen_names}
    if frozen_hashes_before != frozen_hashes_after:
        raise RuntimeError("A frozen evaluation file changed during split creation")
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "selected_strategy": selected.strategy,
        "source_current_train": {
            "path": str(paths["train"].resolve()),
            "sha256": sha256(paths["train"]),
            "pairs": len(train),
        },
        "ordinary_hard_ood_used_for_split_decisions": False,
        "frozen_evaluation_sha256": frozen_hashes_after,
        "rule_discovery_pairs": len(discovery),
        "rule_internal_validation_pairs": len(validation),
        "candidate_comparison": comparisons.to_dict(orient="records"),
        "category_quotas": quotas.to_dict(orient="records"),
        "ood_protocol": {
            "phase_1": "freeze methodology on current train, then test global-rule transfer on untouched OOD",
            "phase_2": "only after transfer test, derive category-specific rules and retain a final untouched subset per OOD category when feasible",
            "phase_2_executed": False,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = build_report(
        selected,
        comparisons,
        selected.by_category,
        quotas,
        args.seed,
        sha256(paths["train"]),
        frozen_hashes_after,
        len(conflict_rows),
    )
    (report_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(selected.overall, ensure_ascii=False, indent=2), flush=True)
    print(f"Split сохранён в {output_dir}", flush=True)
    print(f"Отчёт сохранён в {report_dir}", flush=True)


if __name__ == "__main__":
    main()
