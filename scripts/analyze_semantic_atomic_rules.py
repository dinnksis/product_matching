"""Pool atomic-rule evidence over MiniLM semantic neighbours.

This stage deliberately does not create or require a hand-written ontology.
Each observed atom prototype is embedded independently, neighbours are found
inside hard-compatible blocks, and label statistics are pooled with cosine
similarity weights.  Human labels never enter the embedding text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OCCURRENCES = (
    ROOT / "reports" / "atomic_rule_statistics_current" / "atomic_occurrences.parquet"
)
DEFAULT_OUTPUT = ROOT / "reports" / "semantic_atomic_rule_statistics_current"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_THRESHOLDS = (0.65, 0.70, 0.75, 0.80, 0.85)
DEFAULT_SELECTED_THRESHOLD = 0.80

MEASUREMENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("data", re.compile(r"(?:^|\s)(?:к?б|мб|гб|тб|kb|mb|gb|tb)(?:$|\s)", re.I)),
    ("volume", re.compile(r"(?:^|\s)(?:мл|ml|литр\w*|л|l)(?:$|\s)", re.I)),
    ("mass", re.compile(r"(?:^|\s)(?:мг|mg|кг|kg|г|g)(?:$|\s)", re.I)),
    (
        "length",
        re.compile(
            r"(?:^|\s)(?:мм|mm|см|cm|метр\w*|м|m|дюйм\w*|inch(?:es)?)(?:$|\s)",
            re.I,
        ),
    ),
    ("power", re.compile(r"(?:^|\s)(?:вт|w|квт|kw)(?:$|\s)", re.I)),
    ("optical", re.compile(r"(?:^|\s)(?:дптр|диоптр\w*|d)(?:$|\s)", re.I)),
    ("percentage", re.compile(r"%")),
)
COUNT_CONCEPT_RE = re.compile(
    r"(?:count|quantity|number_of|количеств|комплект|штук|шт\b)", re.I
)
SEMANTIC_FAMILY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "identifier",
        re.compile(
            r"(?:sku|артикул|article|part[_ ]?number|oem|product[_ ]?code|"
            r"model[_ ]?(?:number|identifier)|серийн\w*\s+номер)",
            re.I,
        ),
    ),
    ("data", re.compile(r"(?:storage|memory|памят|накопител)", re.I)),
    ("volume", re.compile(r"(?:volume|capacity|объем|объём|литраж)", re.I)),
    ("mass", re.compile(r"(?:weight|mass|вес|масса)", re.I)),
    (
        "length",
        re.compile(
            r"(?:length|width|height|diameter|thickness|depth|длин|ширин|высот|"
            r"диаметр|толщин|глубин)",
            re.I,
        ),
    ),
    ("power", re.compile(r"(?:power|watt|мощност|ватт)", re.I)),
    ("optical", re.compile(r"(?:optical[_ ]?power|diopter|диоптр)", re.I)),
    ("age", re.compile(r"(?:\bage\b|возраст)", re.I)),
    ("duration", re.compile(r"(?:duration|shelf[_ ]?life|срок|время|период)", re.I)),
    ("count", COUNT_CONCEPT_RE),
)
FORBIDDEN_IDENTIFIER_RE = re.compile(
    r"(?:sku|артикул|партномер|part[_ -]?number|oem|код\s+товара)", re.I
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--occurrences", type=Path, default=DEFAULT_OCCURRENCES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument(
        "--thresholds",
        default=",".join(map(str, DEFAULT_THRESHOLDS)),
        help="Comma-separated cosine thresholds; all are evaluated in one run.",
    )
    parser.add_argument("--minimum-support", type=float, default=2.0)
    parser.add_argument("--minimum-probability", type=float, default=0.8)
    parser.add_argument(
        "--evidence-scope",
        choices=("singleton", "all"),
        default="singleton",
        help=(
            "singleton uses only pairs with one extracted atom; all also uses "
            "correlational evidence from multi-atom pairs."
        ),
    )
    parser.add_argument(
        "--selected-threshold",
        type=float,
        default=DEFAULT_SELECTED_THRESHOLD,
        help="Threshold exported as recommended_semantic_rules.parquet.",
    )
    parser.add_argument("--audit-rows", type=int, default=200)
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    text = str(value or "").casefold().replace("ё", "е").replace(",", ".")
    return " ".join(re.findall(r"[0-9a-zа-я%+./-]+", text))


def stable_hash(parts: Iterable[Any]) -> str:
    payload = json.dumps(list(parts), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_family(
    concept: Any,
    raw_a: Any,
    raw_b: Any,
    attribute_a: Any = "",
    attribute_b: Any = "",
) -> str:
    concept_text = normalize_text(str(concept).replace("_", " "))
    role_text = f"{concept_text} {normalize_text(attribute_a)} {normalize_text(attribute_b)}"
    values = f" {normalize_text(raw_a)} {normalize_text(raw_b)} "
    for family, pattern in MEASUREMENT_PATTERNS:
        if pattern.search(values):
            return family
    for family, pattern in SEMANTIC_FAMILY_PATTERNS:
        if pattern.search(role_text):
            return family
    if re.search(r"\d", values):
        return "numeric"
    return "text"


def attribute_role(row: pd.Series) -> str:
    names = {
        normalize_text(row.get("raw_attribute_a")),
        normalize_text(row.get("raw_attribute_b")),
    }
    names.discard("")
    return " | ".join(sorted(names)) or "из названия товара"


def semantic_text(row: pd.Series) -> str:
    concept = str(row["raw_concept"] or row["concept"]).replace("_", " ")
    # Category, relation and value family are hard blocks and must not dominate
    # the embedding with identical template tokens.
    return f"{concept}; {row['attribute_role']}"


def build_prototypes(occurrences: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "pair_id",
        "category",
        "human_label",
        "split",
        "is_singleton",
        "concept",
        "raw_concept",
        "relation",
        "raw_value_a",
        "raw_value_b",
        "raw_attribute_a",
        "raw_attribute_b",
    }
    missing = required - set(occurrences.columns)
    if missing:
        raise ValueError(f"occurrences missing columns: {sorted(missing)}")
    frame = occurrences[occurrences["relation"].eq("different_value")].copy()
    frame["attribute_role"] = frame.apply(attribute_role, axis=1)
    frame["value_family"] = [
        value_family(concept, raw_a, raw_b, attribute_a, attribute_b)
        for concept, raw_a, raw_b, attribute_a, attribute_b in zip(
            frame["raw_concept"],
            frame["raw_value_a"],
            frame["raw_value_b"],
            frame["raw_attribute_a"],
            frame["raw_attribute_b"],
        )
    ]
    frame["raw_concept_normalized"] = frame["raw_concept"].map(normalize_text)
    frame["prototype_key"] = [
        stable_hash((category, relation, concept, role, family))
        for category, relation, concept, role, family in zip(
            frame["category"],
            frame["relation"],
            frame["raw_concept_normalized"],
            frame["attribute_role"],
            frame["value_family"],
        )
    ]
    # One prototype can be emitted more than once for one pair. A pair must
    # contribute at most one vote to that prototype, especially in the
    # correlational all-pairs mode.
    evidence = frame.drop_duplicates(["pair_id", "prototype_key"]).copy()
    for label in (0, 1):
        evidence[f"label{label}"] = evidence["human_label"].eq(label).astype(int)
        evidence[f"singleton_label{label}"] = (
            evidence["is_singleton"].astype(bool)
            & evidence["human_label"].eq(label)
        ).astype(int)
        for split in ("discovery", "validation"):
            evidence[f"{split}_label{label}"] = (
                evidence["human_label"].eq(label) & evidence["split"].eq(split)
            ).astype(int)
            evidence[f"{split}_singleton_label{label}"] = (
                evidence["is_singleton"].astype(bool)
                & evidence["human_label"].eq(label)
                & evidence["split"].eq(split)
            ).astype(int)
    grouped = (
        evidence.groupby("prototype_key", observed=True)
        .agg(
            category=("category", "first"),
            relation=("relation", "first"),
            raw_concept=("raw_concept", "first"),
            canonical_concept=("concept", "first"),
            attribute_role=("attribute_role", "first"),
            value_family=("value_family", "first"),
            pair_support=("pair_id", "size"),
            label0=("label0", "sum"),
            label1=("label1", "sum"),
            singleton_label0=("singleton_label0", "sum"),
            singleton_label1=("singleton_label1", "sum"),
            discovery_label0=("discovery_label0", "sum"),
            discovery_label1=("discovery_label1", "sum"),
            validation_label0=("validation_label0", "sum"),
            validation_label1=("validation_label1", "sum"),
            discovery_singleton_label0=("discovery_singleton_label0", "sum"),
            discovery_singleton_label1=("discovery_singleton_label1", "sum"),
            validation_singleton_label0=("validation_singleton_label0", "sum"),
            validation_singleton_label1=("validation_singleton_label1", "sum"),
            example_value_a=("raw_value_a", "first"),
            example_value_b=("raw_value_b", "first"),
            example_pair_id=("pair_id", "first"),
        )
        .reset_index()
    )
    grouped["singleton_support"] = (
        grouped["singleton_label0"] + grouped["singleton_label1"]
    )
    grouped["all_support"] = grouped["label0"] + grouped["label1"]
    grouped["semantic_text"] = grouped.apply(semantic_text, axis=1)
    grouped["forbidden_identifier"] = [
        bool(FORBIDDEN_IDENTIFIER_RE.search(f"{concept} {role}"))
        for concept, role in zip(grouped["raw_concept"], grouped["attribute_role"])
    ]
    grouped["block_key"] = [
        stable_hash((category, relation, family))
        for category, relation, family in zip(
            grouped["category"], grouped["relation"], grouped["value_family"]
        )
    ]
    return grouped, frame


def cannot_link_concepts(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    """Different raw concepts emitted together are a conservative cannot-link."""
    pairs: set[tuple[str, str, str]] = set()
    multi = frame.groupby(["category", "pair_id"], observed=True)[
        "raw_concept_normalized"
    ].agg(lambda values: sorted(set(values)))
    for (category, _), concepts in multi.items():
        if len(concepts) < 2:
            continue
        for left_index, left in enumerate(concepts):
            for right in concepts[left_index + 1 :]:
                if left != right:
                    pairs.add((str(category), left, right))
    return pairs


def is_cannot_link(
    category: str, left: str, right: str, cannot_links: set[tuple[str, str, str]]
) -> bool:
    if left == right:
        return False
    first, second = sorted((left, right))
    return (category, first, second) in cannot_links


def embed_prototypes(
    prototypes: pd.DataFrame, model_name: str, batch_size: int
) -> tuple[np.ndarray, SentenceTransformer]:
    model = SentenceTransformer(model_name, local_files_only=True, device="cpu")
    embeddings = model.encode(
        prototypes["semantic_text"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32), model


@dataclass(frozen=True)
class Neighbour:
    index: int
    similarity: float


def nearest_neighbours(
    prototypes: pd.DataFrame, embeddings: np.ndarray, top_k: int
) -> list[list[Neighbour]]:
    result: list[list[Neighbour]] = [[] for _ in range(len(prototypes))]
    for _, block in prototypes.groupby("block_key", observed=True, sort=False):
        indices = block.index.to_numpy(dtype=np.int64)
        vectors = np.ascontiguousarray(embeddings[indices])
        if not len(indices):
            continue
        query_k = min(max(1, top_k), len(indices))
        # Hard category/relation/value-family blocking leaves only small
        # matrices (currently max 390 rows). Exact blockwise cosine search is
        # therefore cheaper and avoids loading a second OpenMP runtime next to
        # PyTorch on macOS. A large future run can replace this function with a
        # separate-process ANN index without changing the statistical stage.
        similarity_matrix = vectors @ vectors.T
        for local_row, global_index in enumerate(indices):
            row = similarity_matrix[local_row]
            if query_k == len(indices):
                local_neighbours = np.argsort(-row, kind="stable")
            else:
                partition = np.argpartition(-row, query_k - 1)[:query_k]
                local_neighbours = partition[np.argsort(-row[partition], kind="stable")]
            rows: list[Neighbour] = []
            for local_neighbour in local_neighbours:
                rows.append(
                    Neighbour(
                        index=int(indices[int(local_neighbour)]),
                        similarity=float(row[int(local_neighbour)]),
                    )
                )
            result[int(global_index)] = rows
    return result


def pooled_candidates(
    prototypes: pd.DataFrame,
    neighbours: list[list[Neighbour]],
    cannot_links: set[tuple[str, str, str]],
    *,
    evidence_scope: str = "singleton",
    threshold: float,
    minimum_support: float,
    minimum_probability: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if evidence_scope not in {"singleton", "all"}:
        raise ValueError("evidence_scope must be singleton or all")
    evidence_prefix = "singleton_" if evidence_scope == "singleton" else ""
    rows: list[dict[str, Any]] = []
    neighbour_rows: list[dict[str, Any]] = []
    neighbour_index_sets = [
        {neighbour.index for neighbour in row} for row in neighbours
    ]
    for center_index, center in prototypes.iterrows():
        accepted: list[Neighbour] = []
        center_concept = normalize_text(center["raw_concept"])
        for neighbour in neighbours[center_index]:
            if neighbour.similarity + 1e-7 < threshold:
                continue
            if center_index not in neighbour_index_sets[neighbour.index]:
                continue
            candidate = prototypes.iloc[neighbour.index]
            candidate_concept = normalize_text(candidate["raw_concept"])
            if is_cannot_link(
                str(center["category"]),
                center_concept,
                candidate_concept,
                cannot_links,
            ):
                continue
            accepted.append(neighbour)
            neighbour_rows.append(
                {
                    "threshold": threshold,
                    "center_index": center_index,
                    "neighbour_index": neighbour.index,
                    "similarity": neighbour.similarity,
                }
            )
        if not any(item.index == center_index for item in accepted):
            accepted.append(Neighbour(center_index, 1.0))

        def weighted(column: str) -> float:
            return float(
                sum(
                    item.similarity * float(prototypes.iloc[item.index][column])
                    for item in accepted
                )
            )

        label0 = weighted(f"{evidence_prefix}label0")
        label1 = weighted(f"{evidence_prefix}label1")
        singleton0 = weighted("singleton_label0")
        singleton1 = weighted("singleton_label1")
        singleton_support = singleton0 + singleton1
        singleton_target = 0 if singleton0 >= singleton1 else 1
        singleton_probability = (
            max(singleton0, singleton1) / singleton_support
            if singleton_support
            else 0.0
        )
        support = label0 + label1
        discovery0 = weighted(f"discovery_{evidence_prefix}label0")
        discovery1 = weighted(f"discovery_{evidence_prefix}label1")
        validation0 = weighted(f"validation_{evidence_prefix}label0")
        validation1 = weighted(f"validation_{evidence_prefix}label1")
        target = 0 if label0 >= label1 else 1
        target_support = label0 if target == 0 else label1
        probability = target_support / support if support else 0.0
        discovery_support = discovery0 + discovery1
        validation_support = validation0 + validation1
        discovery_probability = (
            (discovery0 if target == 0 else discovery1) / discovery_support
            if discovery_support
            else 0.0
        )
        validation_probability = (
            (validation0 if target == 0 else validation1) / validation_support
            if validation_support
            else 0.0
        )
        rows.append(
            {
                "threshold": threshold,
                "center_index": center_index,
                "prototype_key": center["prototype_key"],
                "category": center["category"],
                "relation": center["relation"],
                "raw_concept": center["raw_concept"],
                "canonical_concept": center["canonical_concept"],
                "attribute_role": center["attribute_role"],
                "value_family": center["value_family"],
                "semantic_text": center["semantic_text"],
                "forbidden_identifier": bool(center["forbidden_identifier"]),
                "evidence_scope": evidence_scope,
                "center_pair_support": int(center["pair_support"]),
                "center_singleton_support": int(center["singleton_support"]),
                "neighbour_count": len(accepted),
                "weighted_evidence_support": support,
                "weighted_label0": label0,
                "weighted_label1": label1,
                "weighted_singleton_support": singleton_support,
                "weighted_singleton_label0": singleton0,
                "weighted_singleton_label1": singleton1,
                "singleton_target_label": singleton_target,
                "singleton_target_probability": singleton_probability,
                "target_label": target,
                "target_probability": probability,
                "discovery_weighted_support": discovery_support,
                "discovery_target_probability": discovery_probability,
                "validation_weighted_support": validation_support,
                "validation_target_probability": validation_probability,
                "cross_split_p80": bool(
                    discovery_support > 0
                    and validation_support > 0
                    and discovery_probability >= minimum_probability
                    and validation_probability >= minimum_probability
                ),
                "is_candidate": bool(
                    support >= minimum_support
                    and probability >= minimum_probability
                    and not bool(center["forbidden_identifier"])
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(neighbour_rows)


def select_representative_rules(
    candidates: pd.DataFrame,
    neighbour_edges: pd.DataFrame,
    prototypes: pd.DataFrame,
) -> pd.DataFrame:
    frame = candidates[candidates["is_candidate"]].copy()
    if not len(frame):
        return frame.assign(selected_rule=False)
    adjacency: dict[int, set[int]] = defaultdict(set)
    for edge in neighbour_edges.itertuples(index=False):
        adjacency[int(edge.center_index)].add(int(edge.neighbour_index))
        adjacency[int(edge.neighbour_index)].add(int(edge.center_index))
    frame = frame.sort_values(
        [
            "cross_split_p80",
            "weighted_evidence_support",
            "target_probability",
            "prototype_key",
        ],
        ascending=[False, False, False, True],
        kind="stable",
    )
    selected: list[int] = []
    suppressed: set[int] = set()
    target_by_center = {
        int(center_index): int(target)
        for center_index, target in zip(frame["center_index"], frame["target_label"])
    }
    for row in frame.itertuples(index=False):
        center_index = int(row.center_index)
        if center_index in suppressed:
            continue
        selected.append(center_index)
        target = int(row.target_label)
        for neighbour_index in adjacency.get(center_index, set()):
            if target_by_center.get(neighbour_index) == target:
                suppressed.add(neighbour_index)
    selected_frame = frame[frame["center_index"].isin(selected)].copy()
    selected_frame["generation_rule_id"] = [
        f"semantic_{stable_hash((row.threshold, row.prototype_key, row.target_label))[:20]}"
        for row in selected_frame.itertuples(index=False)
    ]
    return selected_frame.reset_index(drop=True)


def assign_rule_tiers(rules: pd.DataFrame) -> pd.DataFrame:
    """Attach nested evidence tiers without changing the selected rule set."""
    result = rules.copy()
    result["evidence_tier"] = "SEMANTIC_P80_SUPPORT2"
    cross_split = result["cross_split_p80"].astype(bool)
    result.loc[cross_split, "evidence_tier"] = "SEMANTIC_CROSS_SPLIT_P80"
    result.loc[
        cross_split & result["weighted_evidence_support"].ge(5.0),
        "evidence_tier",
    ] = "SEMANTIC_CROSS_SPLIT_SUPPORT5_P80"
    return result


def audit_neighbours(
    prototypes: pd.DataFrame,
    edges: pd.DataFrame,
    limit: int,
) -> pd.DataFrame:
    if not len(edges):
        return pd.DataFrame()
    non_self = edges[edges["center_index"].ne(edges["neighbour_index"])].copy()
    non_self["edge_key"] = [
        stable_hash((min(left, right), max(left, right)))
        for left, right in zip(non_self["center_index"], non_self["neighbour_index"])
    ]
    non_self = non_self.drop_duplicates("edge_key")
    non_self["stable_rank"] = non_self["edge_key"].map(
        lambda value: int(str(value)[:16], 16)
    )
    sampled = non_self.sort_values("stable_rank", kind="stable").head(limit).copy()
    rows: list[dict[str, Any]] = []
    for edge in sampled.itertuples(index=False):
        left = prototypes.iloc[int(edge.center_index)]
        right = prototypes.iloc[int(edge.neighbour_index)]
        rows.append(
            {
                "similarity": float(edge.similarity),
                "left_category": left["category"],
                "left_concept": left["raw_concept"],
                "left_attribute_role": left["attribute_role"],
                "left_value_family": left["value_family"],
                "right_concept": right["raw_concept"],
                "right_attribute_role": right["attribute_role"],
                "right_value_family": right["value_family"],
                "manual_same_attribute": "",
                "manual_notes": "",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    thresholds = tuple(float(value) for value in args.thresholds.split(",") if value)
    if not thresholds or any(not 0.0 < value <= 1.0 for value in thresholds):
        raise ValueError("thresholds must be within (0,1]")
    if not any(math.isclose(args.selected_threshold, value) for value in thresholds):
        raise ValueError("selected-threshold must be included in thresholds")
    if args.top_k < 2 or args.batch_size < 1:
        raise ValueError("top-k must be >=2 and batch-size must be positive")
    occurrences_path = args.occurrences.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    occurrences = pd.read_parquet(occurrences_path)
    all_prototypes, enriched_occurrences = build_prototypes(occurrences)
    support_column = (
        "singleton_support" if args.evidence_scope == "singleton" else "all_support"
    )
    prototypes = all_prototypes[all_prototypes[support_column].gt(0)].copy()
    prototypes = prototypes.reset_index(drop=True)
    cannot_links = cannot_link_concepts(enriched_occurrences)
    embeddings, model = embed_prototypes(prototypes, args.model, args.batch_size)
    neighbours = nearest_neighbours(prototypes, embeddings, args.top_k)

    all_candidates: list[pd.DataFrame] = []
    all_selected: list[pd.DataFrame] = []
    all_edges: list[pd.DataFrame] = []
    sweep_rows: list[dict[str, Any]] = []
    audit_threshold = min(thresholds, key=lambda value: abs(value - 0.75))
    audit_edges = pd.DataFrame()
    for threshold in thresholds:
        candidates, edges = pooled_candidates(
            prototypes,
            neighbours,
            cannot_links,
            evidence_scope=args.evidence_scope,
            threshold=threshold,
            minimum_support=args.minimum_support,
            minimum_probability=args.minimum_probability,
        )
        selected = select_representative_rules(candidates, edges, prototypes)
        all_candidates.append(candidates)
        all_selected.append(selected)
        all_edges.append(edges)
        if math.isclose(threshold, audit_threshold):
            audit_edges = edges
        for target in (0, 1):
            target_selected = selected[selected["target_label"].eq(target)]
            sweep_rows.append(
                {
                    "threshold": threshold,
                    "target_label": target,
                    "candidate_centers_before_dedup": int(
                        (
                            candidates["is_candidate"]
                            & candidates["target_label"].eq(target)
                        ).sum()
                    ),
                    "representative_rules": int(len(target_selected)),
                    "cross_split_rules": int(target_selected["cross_split_p80"].sum()),
                    "median_weighted_support": float(
                        target_selected["weighted_evidence_support"].median()
                    )
                    if len(target_selected)
                    else 0.0,
                    "median_target_probability": float(
                        target_selected["target_probability"].median()
                    )
                    if len(target_selected)
                    else 0.0,
                }
            )

    candidates_frame = pd.concat(all_candidates, ignore_index=True)
    selected_frame = pd.concat(all_selected, ignore_index=True)
    edges_frame = pd.concat(all_edges, ignore_index=True)
    sweep = pd.DataFrame(sweep_rows)
    audit = audit_neighbours(prototypes, audit_edges, args.audit_rows)
    recommended = assign_rule_tiers(
        selected_frame[
            selected_frame["threshold"].map(
                lambda value: math.isclose(value, args.selected_threshold)
            )
        ].copy()
    )

    np.save(output_dir / "prototype_embeddings.npy", embeddings)
    prototypes.to_parquet(output_dir / "semantic_prototypes.parquet", index=False)
    candidates_frame.to_parquet(output_dir / "semantic_rule_candidates.parquet", index=False)
    selected_frame.to_parquet(output_dir / "semantic_rules.parquet", index=False)
    recommended.to_parquet(
        output_dir / "recommended_semantic_rules.parquet", index=False
    )
    edges_frame.to_parquet(output_dir / "semantic_neighbours.parquet", index=False)
    sweep.to_csv(output_dir / "threshold_sweep.csv", index=False)
    audit.to_csv(output_dir / "neighbour_audit.csv", index=False)

    summary = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "occurrences": str(occurrences_path),
        "occurrences_sha256": sha256_file(occurrences_path),
        "occurrence_rows": int(len(occurrences)),
        "different_value_rows": int(len(enriched_occurrences)),
        "all_prototype_rows": int(len(all_prototypes)),
        "selected_evidence_prototype_rows": int(len(prototypes)),
        "singleton_evidence_prototype_rows": int(
            all_prototypes["singleton_support"].gt(0).sum()
        ),
        "evidence_scope": args.evidence_scope,
        "cannot_link_concept_pairs": int(len(cannot_links)),
        "model": args.model,
        "embedding_dimension": int(embeddings.shape[1]),
        "model_max_sequence_length": int(model.max_seq_length),
        "top_k": int(args.top_k),
        "thresholds": list(thresholds),
        "minimum_support": float(args.minimum_support),
        "minimum_probability": float(args.minimum_probability),
        "selected_threshold": float(args.selected_threshold),
        "recommended_rule_counts": {
            "total": int(len(recommended)),
            "label0": int(recommended["target_label"].eq(0).sum()),
            "label1": int(recommended["target_label"].eq(1).sum()),
            "cross_split_total": int(recommended["cross_split_p80"].sum()),
            "cross_split_label0": int(
                (
                    recommended["cross_split_p80"]
                    & recommended["target_label"].eq(0)
                ).sum()
            ),
            "cross_split_label1": int(
                (
                    recommended["cross_split_p80"]
                    & recommended["target_label"].eq(1)
                ).sum()
            ),
            "zero_singleton_evidence_total": int(
                recommended["weighted_singleton_support"].eq(0).sum()
            ),
            "zero_singleton_evidence_label0": int(
                (
                    recommended["weighted_singleton_support"].eq(0)
                    & recommended["target_label"].eq(0)
                ).sum()
            ),
            "zero_singleton_evidence_label1": int(
                (
                    recommended["weighted_singleton_support"].eq(0)
                    & recommended["target_label"].eq(1)
                ).sum()
            ),
            "strong_singleton_agreement": int(
                (
                    recommended["weighted_singleton_support"].ge(2.0)
                    & recommended["singleton_target_probability"].ge(0.8)
                    & recommended["singleton_target_label"].eq(
                        recommended["target_label"]
                    )
                ).sum()
            ),
            "strong_singleton_conflict": int(
                (
                    recommended["weighted_singleton_support"].ge(2.0)
                    & recommended["singleton_target_probability"].ge(0.8)
                    & recommended["singleton_target_label"].ne(
                        recommended["target_label"]
                    )
                ).sum()
            ),
            "cross_split_support5_total": int(
                recommended["evidence_tier"]
                .eq("SEMANTIC_CROSS_SPLIT_SUPPORT5_P80")
                .sum()
            ),
            "cross_split_support5_label0": int(
                (
                    recommended["evidence_tier"].eq(
                        "SEMANTIC_CROSS_SPLIT_SUPPORT5_P80"
                    )
                    & recommended["target_label"].eq(0)
                ).sum()
            ),
            "cross_split_support5_label1": int(
                (
                    recommended["evidence_tier"].eq(
                        "SEMANTIC_CROSS_SPLIT_SUPPORT5_P80"
                    )
                    & recommended["target_label"].eq(1)
                ).sum()
            ),
        },
        "human_labels_in_embedding_text": False,
        "hard_blocks": ["category", "relation", "value_family"],
        "selection": "weighted_semantic_neighbourhood_then_greedy_non_max_suppression",
        "threshold_sweep": sweep.to_dict("records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
