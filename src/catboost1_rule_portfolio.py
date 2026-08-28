"""Leakage-safe rule-gated negative routing on top of CatBoost-1 OOF scores.

Candidate conditions are deterministic, label-free templates.  Labels enter only
in :func:`select_portfolio`, which must be called on calibration rows disjoint
from the rows on which the returned policy is evaluated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from src.catboost1_early_exit import best_side_state, semantic_family, threshold_states, wilson_upper


@dataclass(frozen=True)
class RuleGate:
    candidate_index: int
    score_cap: float
    name: str


@dataclass(frozen=True)
class RoutingPolicy:
    score_threshold: float
    veto_name: str
    gates: tuple[RuleGate, ...]
    calibration_accepted: int
    calibration_errors: int
    calibration_ucb_95: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _row_multiply(matrix: sparse.csr_matrix, mask: np.ndarray) -> sparse.csr_matrix:
    return matrix.multiply(np.asarray(mask, dtype=np.float32)[:, None]).tocsr()


def _nonempty_columns(matrix: sparse.csr_matrix) -> np.ndarray:
    return np.flatnonzero(np.asarray(matrix.sum(axis=0)).ravel() > 0)


def _domain_for_concept(concept: str) -> str | None:
    value = str(concept).casefold().replace("-", "_").replace(" ", "_")
    family = semantic_family(value)
    optical_tokens = ("optical", "lens", "pupillary", "diopter", "cylinder", "axis", "bridge", "temple", "frame")
    technology_tokens = (
        "chipset", "processor", "gpu", "video_memory", "ram_", "ram_capacity", "ram_type",
        "storage", "ssd", "hdd", "memory_capacity", "screen_resolution",
    )
    quantity_tokens = (
        "volume", "weight", "package_quantity", "pack_count", "dosage", "dose",
        "concentration", "tablet_count", "capsule_count", "serving_count",
    )
    dimension_tokens = ("width", "height", "length", "diameter", "thickness", "dimensions")
    identifier_tokens = ("model_number", "model_name", "manufacturer_part_number", "sku", "mpn", "product_line")
    if family == "optical" or any(token in value for token in optical_tokens):
        return "optical"
    if any(token in value for token in technology_tokens):
        return "technology"
    if family in {"volume", "weight", "pack_count"} or any(token in value for token in quantity_tokens):
        return "quantity"
    if "package" not in value and (family == "dimensions" or any(token in value for token in dimension_tokens)):
        return "dimensions"
    if family == "model_number" or any(token in value for token in identifier_tokens):
        return "identifier"
    return None


def build_candidate_matrix(
    base: pd.DataFrame,
    global_rules: sparse.csr_matrix,
    category_rules: sparse.csr_matrix,
    definitions: pd.DataFrame,
    category_vocabulary: Sequence[str],
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    """Build label-free atomic, anchor-conditioned and compound gates."""

    if len(base) != global_rules.shape[0] or len(base) != category_rules.shape[0]:
        raise ValueError("Feature rows and rule matrices are not aligned")
    global_keep = _nonempty_columns(global_rules)
    atomic = global_rules[:, global_keep].astype(np.uint8).tocsr()
    atomic_defs = definitions.iloc[global_keep].reset_index(drop=True)
    parts: list[sparse.csr_matrix] = [atomic]
    rows: list[dict[str, Any]] = []
    for local, (_, rule) in enumerate(atomic_defs.iterrows()):
        rows.append({
            "candidate_index": local,
            "source": "mined_rule_global",
            "scope": "global",
            "anchor": "none",
            "category": "",
            "concept": str(rule.concept),
            "relation": str(rule.relation),
            "rule_id": str(rule.rule_id),
            "name": f"global::{rule.concept}::{rule.relation}",
        })

    category_keep = _nonempty_columns(category_rules)
    category_part = category_rules[:, category_keep].astype(np.uint8).tocsr()
    offset = sum(part.shape[1] for part in parts)
    parts.append(category_part)
    for local, vocabulary_index in enumerate(category_keep):
        token = str(category_vocabulary[int(vocabulary_index)])
        category, rule_index_text = token.rsplit("||", 1)
        rule = definitions.iloc[int(rule_index_text)]
        rows.append({
            "candidate_index": offset + local,
            "source": "mined_rule_category",
            "scope": "category",
            "anchor": "none",
            "category": category,
            "concept": str(rule.concept),
            "relation": str(rule.relation),
            "rule_id": str(rule.rule_id),
            "name": f"category::{category}::{rule.concept}::{rule.relation}",
        })

    anchor_masks = {
        "same_brand": base["brand_match"].to_numpy() > 0,
        "same_model": base["model_code_match"].to_numpy() > 0,
        "same_title_code": base["title_code_match"].to_numpy() > 0,
        "title_sim_085": base["title_token_set"].to_numpy() >= 0.85,
        "title_sim_095": base["title_token_set"].to_numpy() >= 0.95,
        "exact_title": base["title_exact"].to_numpy() > 0,
        "title_code_confirms_conflict": base["title_code_conflict"].to_numpy() > 0,
    }
    for anchor, mask in anchor_masks.items():
        anchored = _row_multiply(atomic, mask)
        keep = _nonempty_columns(anchored)
        if not len(keep):
            continue
        offset = sum(part.shape[1] for part in parts)
        parts.append(anchored[:, keep])
        for local, atomic_index in enumerate(keep):
            rule = atomic_defs.iloc[int(atomic_index)]
            rows.append({
                "candidate_index": offset + local,
                "source": "mined_rule_anchor",
                "scope": "global",
                "anchor": anchor,
                "category": "",
                "concept": str(rule.concept),
                "relation": str(rule.relation),
                "rule_id": str(rule.rule_id),
                "name": f"anchor::{anchor}::{rule.concept}::{rule.relation}",
            })

    different = atomic_defs["relation"].astype(str).eq("different_value").to_numpy()
    domain_counts: dict[str, np.ndarray] = {}
    for domain in ("optical", "technology", "quantity", "dimensions", "identifier"):
        columns = np.flatnonzero([
            bool(different[index] and _domain_for_concept(str(concept)) == domain)
            for index, concept in enumerate(atomic_defs["concept"])
        ])
        if len(columns):
            domain_counts[domain] = np.asarray(atomic[:, columns].sum(axis=1)).ravel()

    compound_masks: dict[str, np.ndarray] = {}
    for domain, counts in domain_counts.items():
        compound_masks[f"{domain}_ge2"] = counts >= 2
        compound_masks[f"{domain}_ge3"] = counts >= 3
    if "identifier" in domain_counts and "technology" in domain_counts:
        compound_masks["identifier_and_technology"] = (
            (domain_counts["identifier"] >= 1) & (domain_counts["technology"] >= 1)
        )
    if "optical" in domain_counts:
        pupillary_columns = np.flatnonzero(
            atomic_defs["concept"].astype(str).str.contains("pupillary", case=False, regex=False).to_numpy()
            & different
        )
        if len(pupillary_columns):
            pupillary = np.asarray(atomic[:, pupillary_columns].sum(axis=1)).ravel() >= 1
            compound_masks["pupillary_and_other_optical"] = pupillary & (domain_counts["optical"] >= 2)

    categories = base["category"].astype(str).to_numpy()
    for compound, mask in compound_masks.items():
        if not mask.any():
            continue
        vectors: list[sparse.csr_matrix] = [sparse.csr_matrix(mask.astype(np.uint8)[:, None])]
        metadata = [("global", "")]
        for category in sorted(pd.unique(categories[mask])):
            category_mask = mask & (categories == category)
            vectors.append(sparse.csr_matrix(category_mask.astype(np.uint8)[:, None]))
            metadata.append(("category", str(category)))
        offset = sum(part.shape[1] for part in parts)
        part = sparse.hstack(vectors, format="csr", dtype=np.uint8)
        parts.append(part)
        for local, (scope, category) in enumerate(metadata):
            rows.append({
                "candidate_index": offset + local,
                "source": "semantic_compound",
                "scope": scope,
                "anchor": "none",
                "category": category,
                "concept": compound,
                "relation": "conjunction",
                "rule_id": "",
                "name": f"compound::{scope}::{category}::{compound}",
            })

    matrix = sparse.hstack(parts, format="csr", dtype=np.uint8)
    catalog = pd.DataFrame(rows).sort_values("candidate_index", kind="stable").reset_index(drop=True)
    if matrix.shape[1] != len(catalog) or not np.array_equal(
        catalog["candidate_index"].to_numpy(), np.arange(matrix.shape[1])
    ):
        raise AssertionError("Candidate matrix and catalog are not aligned")
    catalog["label_free_support"] = np.asarray(matrix.sum(axis=0)).ravel().astype(np.int64)
    return matrix, catalog


def build_veto_masks(base: pd.DataFrame) -> dict[str, np.ndarray]:
    """Deterministic positive/tolerant evidence that can veto a negative exit."""

    strong_identity = (
        (base["title_exact"].to_numpy() > 0)
        | (
            (base["title_token_set"].to_numpy() >= 0.95)
            & (
                (base["model_code_match"].to_numpy() > 0)
                | (base["title_code_match"].to_numpy() > 0)
            )
        )
    )
    decisive_columns = [
        "model_code_conflict", "title_code_conflict", "num_ram_storage_conflict",
        "num_power_conflict", "num_optical_conflict", "num_voltage_conflict",
        "num_frequency_conflict", "num_capacity_conflict",
    ]
    decisive = np.zeros(len(base), dtype=bool)
    for column in decisive_columns:
        if column in base:
            decisive |= base[column].to_numpy() > 0
    identity_without_decisive_conflict = strong_identity & ~decisive

    weak_variant = base["matching_regime"].astype(str).eq("variant_tolerant").to_numpy()
    weak_variant &= (
        (base["num_size_conflict"].to_numpy() > 0)
        | (base["color_conflict"].to_numpy() > 0)
        | (base["material_conflict"].to_numpy() > 0)
    )
    weak_variant &= ~decisive
    return {
        "none": np.zeros(len(base), dtype=bool),
        "strong_identity": identity_without_decisive_conflict,
        "identity_plus_variant_tolerance": identity_without_decisive_conflict | weak_variant,
    }


def _safe_threshold_seeds(
    scores: np.ndarray,
    target: np.ndarray,
    veto: np.ndarray,
    risk_limit: float,
    fractions: Sequence[float],
) -> list[float]:
    usable = ~veto
    states = threshold_states(scores[usable], target[usable], "negative")
    safe = states.loc[states["error_ucb_95"] < risk_limit].sort_values("accepted")
    if safe.empty:
        return [0.0]
    maximum = int(safe["accepted"].max())
    seeds = {0.0, float(safe.iloc[-1].threshold)}
    accepted = safe["accepted"].to_numpy(dtype=np.int64)
    for fraction in fractions:
        wanted = int(maximum * float(fraction))
        index = int(np.searchsorted(accepted, wanted, side="left"))
        index = min(index, len(safe) - 1)
        seeds.add(float(safe.iloc[index].threshold))
    return sorted(seeds)


def apply_policy(
    policy: RoutingPolicy,
    scores: np.ndarray,
    candidate_matrix: sparse.csr_matrix,
    veto_masks: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    veto = veto_masks[policy.veto_name]
    accepted = (scores < policy.score_threshold) & ~veto
    reasons = np.full(len(scores), "", dtype=object)
    reasons[accepted] = "catboost_threshold"
    for gate in policy.gates:
        fired = candidate_matrix[:, gate.candidate_index].toarray().ravel() > 0
        selected = fired & (scores < gate.score_cap) & ~veto & ~accepted
        accepted[selected] = True
        reasons[selected] = gate.name
    return accepted, reasons


def _policy_from_mask(
    threshold: float,
    veto_name: str,
    gates: Iterable[RuleGate],
    mask: np.ndarray,
    target: np.ndarray,
) -> RoutingPolicy:
    accepted = int(mask.sum())
    errors = int(target[mask].sum())
    return RoutingPolicy(
        score_threshold=float(threshold),
        veto_name=veto_name,
        gates=tuple(gates),
        calibration_accepted=accepted,
        calibration_errors=errors,
        calibration_ucb_95=float(wilson_upper(errors, accepted)) if accepted else 1.0,
    )


def select_portfolio(
    scores: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
    candidate_matrix: sparse.csr_matrix,
    candidate_catalog: pd.DataFrame,
    veto_masks: dict[str, np.ndarray],
    *,
    risk_limit: float,
    score_caps: Sequence[float],
    seed_fractions: Sequence[float],
    minimum_support: int,
    minimum_folds: int,
    maximum_gates: int,
) -> RoutingPolicy:
    """Select a greedy union on calibration data only.

    Every returned rule is evaluated jointly with the current union.  The risk
    constraint therefore applies to the complete route rather than separately
    to many sparse rules.
    """

    scores = np.asarray(scores, dtype=np.float64)
    target = np.asarray(target, dtype=np.int8)
    folds = np.asarray(folds)
    if candidate_matrix.shape[0] != len(scores):
        raise ValueError("Candidate matrix is not aligned with calibration rows")
    names = candidate_catalog["name"].astype(str).to_numpy()
    best: RoutingPolicy | None = None
    for veto_name, veto in veto_masks.items():
        veto = np.asarray(veto, dtype=bool)
        gate_parts = [
            _row_multiply(candidate_matrix, (scores < float(cap)) & ~veto)
            for cap in score_caps
        ]
        gated = sparse.hstack(gate_parts, format="csr", dtype=np.uint8)
        gate_candidate = np.tile(np.arange(candidate_matrix.shape[1], dtype=np.int32), len(score_caps))
        gate_caps = np.repeat(np.asarray(score_caps, dtype=np.float64), candidate_matrix.shape[1])
        static_support = np.asarray(gated.sum(axis=0)).ravel()
        fold_presence = np.zeros(gated.shape[1], dtype=np.int16)
        for fold in np.unique(folds):
            rows = folds == fold
            fold_presence += (np.asarray(gated[rows].sum(axis=0)).ravel() > 0).astype(np.int16)
        eligible_static = (static_support >= minimum_support) & (fold_presence >= minimum_folds)

        for threshold in _safe_threshold_seeds(scores, target, veto, risk_limit, seed_fractions):
            current = (scores < threshold) & ~veto
            selected: list[RuleGate] = []
            for _ in range(maximum_gates):
                available = ~current
                # int64 avoids overflow for high-support rules on the 300k rows.
                add_count = np.asarray(gated.T @ available.astype(np.int64)).ravel().astype(np.int64)
                add_errors = np.asarray(
                    gated.T @ (available & (target == 1)).astype(np.int64)
                ).ravel().astype(np.int64)
                current_count = int(current.sum())
                current_errors = int(target[current].sum())
                union_count = current_count + add_count
                union_errors = current_errors + add_errors
                ucb = wilson_upper(union_errors, union_count)
                eligible = eligible_static & (add_count > 0) & (ucb < risk_limit)
                if not eligible.any():
                    break
                indices = np.flatnonzero(eligible)
                order = np.lexsort((union_errors[indices], -union_count[indices]))
                chosen = int(indices[int(order[0])])
                if union_count[chosen] <= current_count:
                    break
                candidate_index = int(gate_candidate[chosen])
                score_cap = float(gate_caps[chosen])
                gate = RuleGate(candidate_index, score_cap, names[candidate_index])
                selected.append(gate)
                fired = candidate_matrix[:, candidate_index].toarray().ravel() > 0
                current |= fired & (scores < score_cap) & ~veto

            candidate_policy = _policy_from_mask(threshold, veto_name, selected, current, target)
            if best is None or (
                candidate_policy.calibration_accepted,
                -candidate_policy.calibration_errors,
                -len(candidate_policy.gates),
            ) > (
                best.calibration_accepted,
                -best.calibration_errors,
                -len(best.gates),
            ):
                best = candidate_policy
    if best is None:
        return RoutingPolicy(0.0, "none", tuple(), 0, 0, 1.0)
    return best


def select_baseline_policy(
    scores: np.ndarray,
    target: np.ndarray,
    risk_limit: float,
) -> RoutingPolicy:
    state = best_side_state(threshold_states(scores, target, "negative"), risk_limit)
    accepted = scores < float(state["threshold"])
    return _policy_from_mask(float(state["threshold"]), "none", (), accepted, target)
