from __future__ import annotations

import hashlib
import heapq
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import combinations
from typing import Any, Iterable, Mapping

import pandas as pd

from .normalization import normalize_text, stable_hash64
from .pair_rules import MutationRule, rules_are_product_compatible
from .rule_values import finite_domain_size


SCHEDULE_VERSION = "balanced_rule_profiles_transition_capacity_v3"
LABEL_QUOTA_POLICY_VERSION = "explicit_binary_target_fraction_v1"


@dataclass(frozen=True)
class RuleProfile:
    """One rule constrained to one category and one product-type profile."""

    profile_id: str
    category: str
    product_type: str
    rule: MutationRule
    primary_task_cap: int | None = None


@dataclass(frozen=True)
class ScheduledRuleBundle:
    """Deterministic donor and rule assignment for one pair-generation task."""

    task_index: int
    donor_id: int
    category: str
    primary: RuleProfile
    secondary: RuleProfile | None = None

    @property
    def profiles(self) -> tuple[RuleProfile, ...]:
        return (
            (self.primary, self.secondary)
            if self.secondary is not None
            else (self.primary,)
        )

    @property
    def rules(self) -> tuple[MutationRule, ...]:
        return tuple(profile.rule for profile in self.profiles)

    @property
    def target(self) -> int:
        labels = {profile.rule.label for profile in self.profiles}
        if len(labels) != 1:
            raise RuntimeError("Scheduled bundle mixes target labels")
        return next(iter(labels))

    def provenance(self, schedule_sha256: str) -> dict[str, Any]:
        return {
            "task_index": self.task_index,
            "source_style_id": self.donor_id,
            "category": self.category,
            "scheduled_primary_rule_id": self.primary.rule.generation_rule_id,
            "scheduled_primary_profile_id": self.primary.profile_id,
            "scheduled_primary_product_type": self.primary.product_type,
            "scheduled_primary_task_cap": self.primary.primary_task_cap,
            "scheduled_secondary_rule_id": (
                self.secondary.rule.generation_rule_id
                if self.secondary is not None
                else None
            ),
            "scheduled_secondary_profile_id": (
                self.secondary.profile_id if self.secondary is not None else None
            ),
            "scheduled_secondary_product_type": (
                self.secondary.product_type if self.secondary is not None else None
            ),
            "scheduled_rule_ids": [
                profile.rule.generation_rule_id for profile in self.profiles
            ],
            "scheduled_rule_profile_ids": [
                profile.profile_id for profile in self.profiles
            ],
            "scheduled_target": self.target,
            "rule_schedule_sha256": schedule_sha256,
        }


@dataclass(frozen=True)
class BalancedRuleSchedule:
    entries: tuple[ScheduledRuleBundle, ...]
    eligible_profiles: tuple[RuleProfile, ...]
    seed: int
    requested_two_rule_fraction: float
    requested_two_rule_tasks: int
    eligible_two_rule_primary_slots: int
    category_available_donors: tuple[tuple[str, int], ...]
    category_task_quotas: tuple[tuple[str, int], ...]
    category_profile_counts: tuple[tuple[str, int], ...]
    semantic_signature_limit: int
    profile_capacity_policy_version: str
    profile_capacity_policy_sha256: str
    requested_label_one_fraction: float | None
    label_category_task_quotas: tuple[tuple[int, tuple[tuple[str, int], ...]], ...]
    schedule_sha256: str

    def bundle_for_task(self, task_index: int) -> ScheduledRuleBundle:
        if task_index < 0 or task_index >= len(self.entries):
            raise IndexError(f"task_index is outside the rule schedule: {task_index}")
        entry = self.entries[task_index]
        if entry.task_index != task_index:
            raise RuntimeError("Rule schedule entries are not indexed contiguously")
        return entry

    def rules_for_task(self, task_index: int) -> tuple[MutationRule, ...]:
        """Return the fixed bundle; retries intentionally retain this assignment."""

        return self.bundle_for_task(task_index).rules

    def provenance_rows(self) -> list[dict[str, Any]]:
        return [entry.provenance(self.schedule_sha256) for entry in self.entries]

    def realized_summary(self, completed_task_indices: Iterable[int]) -> dict[str, Any]:
        """Summarize accepted scheduled tasks without trusting model metadata."""

        indices = sorted({int(index) for index in completed_task_indices})
        if indices and (indices[0] < 0 or indices[-1] >= len(self.entries)):
            raise ValueError("Completed task index is outside the rule schedule")
        entries = [self.bundle_for_task(index) for index in indices]
        eligible_profile_ids = {
            profile.profile_id for profile in self.eligible_profiles
        }
        eligible_rule_ids = {
            profile.rule.generation_rule_id for profile in self.eligible_profiles
        }
        primary_profiles = Counter(entry.primary.profile_id for entry in entries)
        secondary_profiles = Counter(
            entry.secondary.profile_id
            for entry in entries
            if entry.secondary is not None
        )
        primary_rules = Counter(
            entry.primary.rule.generation_rule_id for entry in entries
        )
        secondary_rules = Counter(
            entry.secondary.rule.generation_rule_id
            for entry in entries
            if entry.secondary is not None
        )
        two_rule_tasks = sum(entry.secondary is not None for entry in entries)
        target_counts = Counter(entry.target for entry in entries)
        return {
            "completed_scheduled_tasks": len(entries),
            "pending_scheduled_tasks": len(self.entries) - len(entries),
            "realized_primary_rule_coverage": len(primary_rules),
            "realized_primary_rule_coverage_fraction": (
                len(primary_rules) / len(eligible_rule_ids)
                if eligible_rule_ids
                else 0.0
            ),
            "realized_primary_rule_profile_coverage": len(primary_profiles),
            "realized_primary_rule_profile_coverage_fraction": (
                len(primary_profiles) / len(eligible_profile_ids)
                if eligible_profile_ids
                else 0.0
            ),
            "realized_category_task_counts": dict(
                sorted(Counter(entry.category for entry in entries).items())
            ),
            "realized_scheduled_two_rule_tasks": two_rule_tasks,
            "realized_scheduled_two_rule_fraction": (
                two_rule_tasks / len(entries) if entries else 0.0
            ),
            "realized_target_counts": {
                str(label): int(target_counts[label]) for label in (0, 1)
            },
            "realized_target_fractions": {
                str(label): (
                    target_counts[label] / len(entries) if entries else 0.0
                )
                for label in (0, 1)
            },
            "realized_label_zero_tasks": int(target_counts[0]),
            "realized_label_one_tasks": int(target_counts[1]),
            "realized_label_one_fraction": (
                target_counts[1] / len(entries) if entries else 0.0
            ),
            "realized_primary_rule_profile_usage": dict(
                sorted(primary_profiles.items())
            ),
            "realized_secondary_rule_profile_usage": dict(
                sorted(secondary_profiles.items())
            ),
            "realized_primary_rule_usage": dict(sorted(primary_rules.items())),
            "realized_secondary_rule_usage": dict(sorted(secondary_rules.items())),
        }

    def summary(self) -> dict[str, Any]:
        profile_ids = [profile.profile_id for profile in self.eligible_profiles]
        rule_ids = sorted(
            {profile.rule.generation_rule_id for profile in self.eligible_profiles}
        )
        primary_profile = Counter(entry.primary.profile_id for entry in self.entries)
        secondary_profile = Counter(
            entry.secondary.profile_id
            for entry in self.entries
            if entry.secondary is not None
        )
        primary_rule = Counter(
            entry.primary.rule.generation_rule_id for entry in self.entries
        )
        secondary_rule = Counter(
            entry.secondary.rule.generation_rule_id
            for entry in self.entries
            if entry.secondary is not None
        )
        total_profile = {
            profile_id: primary_profile[profile_id] + secondary_profile[profile_id]
            for profile_id in profile_ids
        }
        total_rule = {
            rule_id: primary_rule[rule_id] + secondary_rule[rule_id]
            for rule_id in rule_ids
        }
        primary_profile_full = {
            profile_id: primary_profile[profile_id] for profile_id in profile_ids
        }
        primary_rule_full = {rule_id: primary_rule[rule_id] for rule_id in rule_ids}
        secondary_profile_full = {
            profile_id: secondary_profile[profile_id] for profile_id in profile_ids
        }
        secondary_rule_full = {
            rule_id: secondary_rule[rule_id] for rule_id in rule_ids
        }
        two_rule_tasks = sum(entry.secondary is not None for entry in self.entries)
        target_counts = Counter(entry.target for entry in self.entries)
        bundle_counts = Counter(
            tuple(sorted(profile.profile_id for profile in entry.profiles))
            for entry in self.entries
        )
        quotas = dict(self.category_task_quotas)
        category_profile_ids: dict[str, list[str]] = defaultdict(list)
        for profile in self.eligible_profiles:
            category_profile_ids[profile.category].append(profile.profile_id)
        category_primary_ranges = {
            category: {
                "min": min(primary_profile[profile_id] for profile_id in ids),
                "max": max(primary_profile[profile_id] for profile_id in ids),
                "skew": (
                    max(primary_profile[profile_id] for profile_id in ids)
                    - min(primary_profile[profile_id] for profile_id in ids)
                ),
            }
            for category, ids in sorted(category_profile_ids.items())
        }
        total_profile_values = list(total_profile.values())
        primary_profile_values = list(primary_profile_full.values())
        finite_caps = {
            profile.profile_id: int(profile.primary_task_cap)
            for profile in self.eligible_profiles
            if profile.primary_task_cap is not None
        }
        saturated_single_profiles = sorted(
            profile_id
            for profile_id, cap in finite_caps.items()
            if primary_profile_full[profile_id] >= cap
            and secondary_profile_full[profile_id] == 0
        )
        balanced_total_values = [
            count
            for profile_id, count in total_profile.items()
            if profile_id not in set(saturated_single_profiles)
        ]
        return {
            "rule_schedule_version": SCHEDULE_VERSION,
            "rule_schedule_sha256": self.schedule_sha256,
            "rule_schedule_seed": self.seed,
            "scheduled_tasks": len(self.entries),
            "eligible_rules": len(rule_ids),
            "eligible_rule_profiles": len(profile_ids),
            "primary_rule_coverage": sum(value > 0 for value in primary_rule_full.values()),
            "primary_rule_profile_coverage": sum(
                value > 0 for value in primary_profile_full.values()
            ),
            "total_rule_coverage": sum(value > 0 for value in total_rule.values()),
            "total_rule_profile_coverage": sum(
                value > 0 for value in total_profile.values()
            ),
            "category_available_donors": dict(self.category_available_donors),
            "category_profile_counts": dict(self.category_profile_counts),
            "category_task_quotas": quotas,
            "category_primary_profile_ranges": category_primary_ranges,
            "profile_capacity_policy_version": self.profile_capacity_policy_version,
            "profile_capacity_policy_sha256": self.profile_capacity_policy_sha256,
            "profile_capacity_semantic_signature_limit": self.semantic_signature_limit,
            "profile_primary_task_capacity_formula": (
                "min((allowed_transition_count or combinations(domain_size,2))"
                "*semantic_signature_limit,safety_cap)"
            ),
            "profile_primary_task_caps": dict(sorted(finite_caps.items())),
            "capacity_limited_rule_profiles": len(finite_caps),
            "capacity_saturated_single_rule_profiles": saturated_single_profiles,
            "requested_two_rule_fraction": self.requested_two_rule_fraction,
            "requested_two_rule_tasks": self.requested_two_rule_tasks,
            "eligible_two_rule_primary_slots": self.eligible_two_rule_primary_slots,
            "scheduled_two_rule_tasks": two_rule_tasks,
            "scheduled_two_rule_fraction": (
                two_rule_tasks / len(self.entries) if self.entries else 0.0
            ),
            "two_rule_target_clipped": two_rule_tasks < self.requested_two_rule_tasks,
            "label_quota_enabled": self.requested_label_one_fraction is not None,
            "label_quota_policy_version": LABEL_QUOTA_POLICY_VERSION,
            "requested_label_one_fraction": self.requested_label_one_fraction,
            "planned_target_counts": {
                str(label): int(target_counts[label]) for label in (0, 1)
            },
            "planned_target_fractions": {
                str(label): (
                    target_counts[label] / len(self.entries) if self.entries else 0.0
                )
                for label in (0, 1)
            },
            "planned_label_zero_tasks": int(target_counts[0]),
            "planned_label_one_tasks": int(target_counts[1]),
            "planned_label_one_fraction": (
                target_counts[1] / len(self.entries) if self.entries else 0.0
            ),
            "label_category_task_quotas": {
                str(label): dict(category_counts)
                for label, category_counts in self.label_category_task_quotas
            },
            "primary_rule_profile_usage": dict(sorted(primary_profile_full.items())),
            "secondary_rule_profile_usage": dict(sorted(secondary_profile_full.items())),
            "total_rule_profile_usage": dict(sorted(total_profile.items())),
            "primary_rule_usage": dict(sorted(primary_rule_full.items())),
            "secondary_rule_usage": dict(sorted(secondary_rule_full.items())),
            "total_rule_usage": dict(sorted(total_rule.items())),
            "max_identical_scheduled_bundle_count": max(bundle_counts.values(), default=0),
            "primary_rule_profile_usage_min": min(primary_profile_values, default=0),
            "primary_rule_profile_usage_max": max(primary_profile_values, default=0),
            "primary_rule_profile_usage_skew": (
                max(primary_profile_values, default=0)
                - min(primary_profile_values, default=0)
            ),
            "total_rule_profile_usage_min": min(total_profile_values, default=0),
            "total_rule_profile_usage_max": max(total_profile_values, default=0),
            "total_rule_profile_usage_skew": (
                max(total_profile_values, default=0)
                - min(total_profile_values, default=0)
            ),
            "balanced_total_rule_profile_usage_min": min(
                balanced_total_values, default=0
            ),
            "balanced_total_rule_profile_usage_max": max(
                balanced_total_values, default=0
            ),
            "balanced_total_rule_profile_usage_skew": (
                max(balanced_total_values, default=0)
                - min(balanced_total_values, default=0)
            ),
        }


def balanced_category_quotas(
    available_donors: Mapping[str, int],
    profile_counts: Mapping[str, int],
    *,
    count: int,
    seed: int,
) -> dict[str, int]:
    """Allocate donors by weighted water-fill with hard category capacities.

    One profile is one unit of weight.  Without a donor cap, every category gets
    approximately ``profile_count * T`` tasks.  A capped category is filled and
    the remaining tasks are redistributed over the other categories.
    """

    if count < 1:
        raise ValueError("count must be positive")
    categories = sorted(
        category
        for category, profiles in profile_counts.items()
        if int(profiles) > 0 and int(available_donors.get(category, 0)) > 0
    )
    capacities = {category: int(available_donors[category]) for category in categories}
    weights = {category: int(profile_counts[category]) for category in categories}
    if not categories:
        raise ValueError("No category has both donors and eligible rule profiles")
    total_capacity = sum(capacities.values())
    if count > total_capacity:
        raise ValueError(f"count must be in [1, {total_capacity}]")

    quotas = {category: 0 for category in categories}
    heap: list[tuple[Fraction, int, str]] = []
    for category in categories:
        heapq.heappush(
            heap,
            (
                Fraction(1, weights[category]),
                stable_hash64(seed, f"category-quota:{category}"),
                category,
            ),
        )
    for _ in range(count):
        if not heap:
            raise RuntimeError("Internal error: category quota capacity was exhausted")
        _, tie_breaker, category = heapq.heappop(heap)
        quotas[category] += 1
        if quotas[category] < capacities[category]:
            heapq.heappush(
                heap,
                (
                    Fraction(quotas[category] + 1, weights[category]),
                    tie_breaker,
                    category,
                ),
            )
    return quotas


def _profile_id(rule: MutationRule, category: str, product_type: str) -> str:
    payload = [
        rule.generation_rule_id,
        normalize_text(category),
        normalize_text(product_type) or "*",
    ]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _primary_task_cap(
    rule: MutationRule, *, semantic_signature_limit: int
) -> int | None:
    if rule.allowed_value_transitions:
        theoretical = len(rule.allowed_value_transitions) * semantic_signature_limit
        if rule.primary_task_safety_cap is None:
            return theoretical
        return min(theoretical, int(rule.primary_task_safety_cap))
    domain_size = finite_domain_size(rule.target_value_domain)
    if not domain_size:
        if rule.primary_task_safety_cap is not None:
            raise ValueError(
                f"Rule {rule.generation_rule_id} has a safety cap without a finite domain"
            )
        return None
    if domain_size < 2:
        raise ValueError(
            f"Rule {rule.generation_rule_id} has fewer than two target domain values"
        )
    theoretical = math.comb(domain_size, 2) * semantic_signature_limit
    if rule.primary_task_safety_cap is None:
        return theoretical
    return min(theoretical, int(rule.primary_task_safety_cap))


def _rule_profiles(
    rules: Iterable[MutationRule],
    categories: set[str],
    *,
    semantic_signature_limit: int,
) -> list[RuleProfile]:
    profiles: list[RuleProfile] = []
    seen: set[str] = set()
    for rule in sorted(rules, key=lambda value: value.generation_rule_id):
        product_types: list[str] = []
        seen_types: set[str] = set()
        for value in rule.allowed_product_types or ("",):
            normalized = normalize_text(value)
            if normalized in seen_types:
                continue
            seen_types.add(normalized)
            product_types.append(value)
        for category in sorted(set(rule.allowed_categories) & categories):
            for product_type in product_types:
                profile_id = _profile_id(rule, category, product_type)
                if profile_id in seen:
                    raise ValueError(f"Duplicate rule profile: {profile_id}")
                seen.add(profile_id)
                scoped = replace(
                    rule,
                    allowed_categories=(category,),
                    allowed_product_types=(product_type,) if product_type else (),
                )
                profiles.append(
                    RuleProfile(
                        profile_id=profile_id,
                        category=category,
                        product_type=product_type,
                        rule=scoped,
                        primary_task_cap=_primary_task_cap(
                            scoped,
                            semantic_signature_limit=semantic_signature_limit,
                        ),
                    )
                )
    return profiles


def _balanced_profile_sequence(
    profiles: list[RuleProfile], *, count: int, seed: int, category: str
) -> list[RuleProfile]:
    if not profiles:
        raise ValueError(f"Category has no eligible profiles: {category}")
    result: list[RuleProfile] = []
    cycle = 0
    while len(result) < count:
        ordered = sorted(
            profiles,
            key=lambda profile: (
                stable_hash64(
                    seed,
                    f"profile-cycle:{category}:{cycle}:{profile.profile_id}",
                ),
                profile.profile_id,
            ),
        )
        result.extend(ordered[: count - len(result)])
        cycle += 1
    return result


def _select_pair_slots(
    entries: list[ScheduledRuleBundle],
    compatible: Mapping[str, tuple[RuleProfile, ...]],
    *,
    target: int,
    seed: int,
) -> set[int]:
    by_profile: dict[str, list[int]] = defaultdict(list)
    for entry in entries:
        if compatible.get(entry.primary.profile_id):
            by_profile[entry.primary.profile_id].append(entry.task_index)
    for profile_id, indexes in by_profile.items():
        indexes.sort(
            key=lambda index: (
                stable_hash64(seed, f"pair-slot:{profile_id}:{index}"),
                index,
            ),
            reverse=True,
        )
    heap: list[tuple[Fraction, int, str]] = []
    selected_counts: dict[str, int] = defaultdict(int)
    for profile_id, indexes in by_profile.items():
        heapq.heappush(
            heap,
            (
                Fraction(1, len(indexes)),
                stable_hash64(seed, f"pair-profile:{profile_id}"),
                profile_id,
            ),
        )
    selected: set[int] = set()
    for _ in range(target):
        if not heap:
            raise RuntimeError("Internal error: pair-slot target is infeasible")
        _, tie_breaker, profile_id = heapq.heappop(heap)
        index = by_profile[profile_id].pop()
        selected.add(index)
        selected_counts[profile_id] += 1
        if by_profile[profile_id]:
            total = selected_counts[profile_id] + len(by_profile[profile_id])
            heapq.heappush(
                heap,
                (
                    Fraction(selected_counts[profile_id] + 1, total),
                    tie_breaker,
                    profile_id,
                ),
            )
    return selected


def _schedule_sha256(entries: Iterable[ScheduledRuleBundle]) -> str:
    payload = [
        {
            "task_index": entry.task_index,
            "donor_id": entry.donor_id,
            "category": entry.category,
            "profiles": [profile.profile_id for profile in entry.profiles],
        }
        for entry in entries
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compatible_profile_edges(
    profiles_by_category: Mapping[str, list[RuleProfile]],
) -> list[tuple[RuleProfile, RuleProfile]]:
    edges: list[tuple[RuleProfile, RuleProfile]] = []
    for category in sorted(profiles_by_category):
        profiles = profiles_by_category[category]
        for left_index, left in enumerate(profiles):
            for right in profiles[left_index + 1 :]:
                if rules_are_product_compatible(left.rule, right.rule):
                    edges.append((left, right))
    return edges


def _primary_available(profile: RuleProfile, usage: Counter[str]) -> bool:
    return (
        profile.primary_task_cap is None
        or usage[profile.profile_id] < profile.primary_task_cap
    )


def _plan_pair_bundles(
    edges: list[tuple[RuleProfile, RuleProfile]],
    *,
    requested: int,
    available_by_category: Mapping[str, int],
    seed: int,
) -> list[tuple[RuleProfile, RuleProfile]]:
    """Plan compatible pairs while balancing both endpoint applications."""

    total_usage: Counter[str] = Counter()
    primary_usage: Counter[str] = Counter()
    category_usage: Counter[str] = Counter()
    edge_usage: Counter[tuple[str, str]] = Counter()
    result: list[tuple[RuleProfile, RuleProfile]] = []
    for step in range(requested):
        candidates: list[
            tuple[tuple[Any, ...], RuleProfile, RuleProfile, tuple[str, str]]
        ] = []
        for left, right in edges:
            category = left.category
            if category_usage[category] >= int(available_by_category[category]):
                continue
            possible_primaries = [
                profile
                for profile in (left, right)
                if _primary_available(profile, primary_usage)
            ]
            if not possible_primaries:
                continue
            primary = min(
                possible_primaries,
                key=lambda profile: (
                    primary_usage[profile.profile_id],
                    total_usage[profile.profile_id],
                    stable_hash64(
                        seed,
                        f"pair-primary:{step}:{profile.profile_id}",
                    ),
                    profile.profile_id,
                ),
            )
            secondary = right if primary.profile_id == left.profile_id else left
            edge_id = tuple(sorted((left.profile_id, right.profile_id)))
            key = (
                max(total_usage[left.profile_id], total_usage[right.profile_id]),
                total_usage[left.profile_id] + total_usage[right.profile_id],
                edge_usage[edge_id],
                Fraction(category_usage[category] + 1, available_by_category[category]),
                stable_hash64(seed, f"pair-edge:{step}:{edge_id}"),
                edge_id,
            )
            candidates.append((key, primary, secondary, edge_id))
        if not candidates:
            break
        _, primary, secondary, edge_id = min(candidates, key=lambda row: row[0])
        result.append((primary, secondary))
        primary_usage[primary.profile_id] += 1
        total_usage[primary.profile_id] += 1
        total_usage[secondary.profile_id] += 1
        category_usage[primary.category] += 1
        edge_usage[edge_id] += 1
    return result


def _largest_balanced_pair_prefix(
    planned_pairs: list[tuple[RuleProfile, RuleProfile]],
    profiles: list[RuleProfile],
    *,
    task_count: int,
) -> int:
    """Clip 1+1 pairs before they force pairable profiles above fair exposure."""

    usage: Counter[str] = Counter()
    snapshots: list[Counter[str]] = [Counter()]
    for left, right in planned_pairs:
        usage[left.profile_id] += 1
        usage[right.profile_id] += 1
        snapshots.append(usage.copy())
    profile_ids = [profile.profile_id for profile in profiles]
    for pair_count in range(len(planned_pairs), -1, -1):
        counts = snapshots[pair_count]
        maximum = max((counts[profile_id] for profile_id in profile_ids), default=0)
        # Remaining single-rule tasks can lift low profiles to maximum-1.  If
        # they cannot, the requested 1+1 rate itself creates the confound.
        required_singles = sum(
            max(0, maximum - 1 - counts[profile_id])
            for profile_id in profile_ids
        )
        if required_singles <= task_count - pair_count:
            return pair_count
    return 0


def _fill_single_bundles(
    pair_bundles: list[tuple[RuleProfile, RuleProfile]],
    profiles: list[RuleProfile],
    *,
    task_count: int,
    available_by_category: Mapping[str, int],
    seed: int,
) -> list[tuple[RuleProfile, RuleProfile | None]]:
    bundles: list[tuple[RuleProfile, RuleProfile | None]] = list(pair_bundles)
    total_usage: Counter[str] = Counter()
    primary_usage: Counter[str] = Counter()
    category_usage: Counter[str] = Counter()
    for primary, secondary in pair_bundles:
        primary_usage[primary.profile_id] += 1
        total_usage[primary.profile_id] += 1
        total_usage[secondary.profile_id] += 1
        category_usage[primary.category] += 1
    for step in range(len(pair_bundles), task_count):
        candidates = [
            profile
            for profile in profiles
            if category_usage[profile.category]
            < int(available_by_category[profile.category])
            and _primary_available(profile, primary_usage)
        ]
        if not candidates:
            finite_capacity = sum(
                min(
                    int(available_by_category[category]),
                    sum(
                        profile.primary_task_cap
                        if profile.primary_task_cap is not None
                        else task_count
                        for profile in profiles
                        if profile.category == category
                    ),
                )
                for category in available_by_category
            )
            raise ValueError(
                "Balanced rule schedule exhausted primary capacity before count: "
                f"scheduled={len(bundles)}, requested={task_count}, "
                f"upper_bound={finite_capacity}"
            )
        primary = min(
            candidates,
            key=lambda profile: (
                total_usage[profile.profile_id],
                primary_usage[profile.profile_id],
                Fraction(
                    category_usage[profile.category] + 1,
                    int(available_by_category[profile.category]),
                ),
                stable_hash64(seed, f"single-profile:{step}:{profile.profile_id}"),
                profile.profile_id,
            ),
        )
        bundles.append((primary, None))
        primary_usage[primary.profile_id] += 1
        total_usage[primary.profile_id] += 1
        category_usage[primary.category] += 1
    return bundles


def _rounded_task_count(count: int, fraction: float) -> int:
    return int(math.floor(count * fraction + 0.5))


def _label_category_task_quotas(
    profiles: list[RuleProfile],
    available_by_category: Mapping[str, int],
    *,
    label_targets: Mapping[int, int],
    count: int,
    seed: int,
) -> dict[int, dict[str, int]]:
    """Allocate exact binary-label quotas over categories without dead ends.

    The feasibility check is the small max-flow cut condition for the two
    labels.  It prevents a flexible label from consuming donors in a category
    that is the only remaining home for the other label.
    """

    labels = tuple(sorted(int(label) for label in label_targets))
    if labels != (0, 1):
        raise ValueError("Explicit label quota requires binary labels 0 and 1")
    profiles_by_cell: dict[tuple[int, str], list[RuleProfile]] = defaultdict(list)
    for profile in profiles:
        profiles_by_cell[(int(profile.rule.label), profile.category)].append(profile)
    categories = tuple(sorted(available_by_category))
    category_remaining = {
        category: int(available_by_category[category]) for category in categories
    }
    cell_remaining: dict[tuple[int, str], int] = {}
    cell_weights: dict[tuple[int, str], int] = {}
    for cell, values in profiles_by_cell.items():
        label, category = cell
        if category not in category_remaining:
            continue
        finite_capacity = sum(
            profile.primary_task_cap or 0
            for profile in values
            if profile.primary_task_cap is not None
        )
        if any(profile.primary_task_cap is None for profile in values):
            finite_capacity = count
        cell_remaining[cell] = min(category_remaining[category], finite_capacity)
        cell_weights[cell] = len(values)

    label_remaining = {
        label: int(label_targets.get(label, 0)) for label in labels
    }
    if sum(label_remaining.values()) != count:
        raise ValueError("Explicit label targets do not sum to count")

    def feasible(
        remaining_labels: Mapping[int, int],
        remaining_categories: Mapping[str, int],
        remaining_cells: Mapping[tuple[int, str], int],
    ) -> bool:
        active_labels = [label for label in labels if remaining_labels[label] > 0]
        for subset_size in range(1, len(active_labels) + 1):
            for subset in combinations(active_labels, subset_size):
                required = sum(remaining_labels[label] for label in subset)
                capacity = sum(
                    min(
                        remaining_categories[category],
                        sum(
                            remaining_cells.get((label, category), 0)
                            for label in subset
                        ),
                    )
                    for category in categories
                )
                if required > capacity:
                    return False
        return True

    if not feasible(label_remaining, category_remaining, cell_remaining):
        available_labels = sorted(
            {profile.rule.label for profile in profiles}
        )
        raise ValueError(
            "Requested label quota is infeasible for available donors, rules and "
            f"primary capacities: targets={dict(label_targets)}, "
            f"available_labels={available_labels}"
        )

    result: dict[int, dict[str, int]] = {
        label: {category: 0 for category in categories} for label in labels
    }
    category_usage: Counter[str] = Counter()
    for step in range(count):
        candidates: list[tuple[tuple[Any, ...], int, str]] = []
        for label in labels:
            if label_remaining[label] <= 0:
                continue
            for category in categories:
                cell = (label, category)
                if (
                    category_remaining[category] <= 0
                    or cell_remaining.get(cell, 0) <= 0
                ):
                    continue
                next_labels = dict(label_remaining)
                next_categories = dict(category_remaining)
                next_cells = dict(cell_remaining)
                next_labels[label] -= 1
                next_categories[category] -= 1
                next_cells[cell] -= 1
                if not feasible(next_labels, next_categories, next_cells):
                    continue
                key = (
                    Fraction(result[label][category] + 1, cell_weights[cell]),
                    Fraction(
                        category_usage[category] + 1,
                        int(available_by_category[category]),
                    ),
                    stable_hash64(
                        seed,
                        f"label-category:{step}:{label}:{category}",
                    ),
                    label,
                    category,
                )
                candidates.append((key, label, category))
        if not candidates:
            raise RuntimeError("Internal error: feasible label quota allocation stalled")
        _, label, category = min(candidates, key=lambda value: value[0])
        result[label][category] += 1
        label_remaining[label] -= 1
        category_remaining[category] -= 1
        cell_remaining[(label, category)] -= 1
        category_usage[category] += 1
    return {
        label: {
            category: value
            for category, value in sorted(category_counts.items())
            if value > 0
        }
        for label, category_counts in sorted(result.items())
    }


def _quota_planned_bundles(
    profiles: list[RuleProfile],
    compatible_edges: list[tuple[RuleProfile, RuleProfile]],
    available_by_category: Mapping[str, int],
    *,
    label_targets: Mapping[int, int],
    requested_two_rule_tasks: int,
    count: int,
    seed: int,
) -> tuple[
    list[tuple[RuleProfile, RuleProfile | None]],
    dict[int, dict[str, int]],
]:
    label_category_quotas = _label_category_task_quotas(
        profiles,
        available_by_category,
        label_targets=label_targets,
        count=count,
        seed=seed,
    )
    if requested_two_rule_tasks == 0:
        bundles: list[tuple[RuleProfile, RuleProfile | None]] = []
        for label in (0, 1):
            label_profiles = [
                profile for profile in profiles if profile.rule.label == label
            ]
            profiles_by_category: dict[str, list[RuleProfile]] = defaultdict(list)
            for profile in label_profiles:
                profiles_by_category[profile.category].append(profile)
            for category, quota in sorted(label_category_quotas[label].items()):
                category_profiles = sorted(
                    profiles_by_category[category], key=lambda value: value.profile_id
                )
                heap: list[tuple[int, int, str, RuleProfile]] = [
                    (
                        0,
                        stable_hash64(
                            seed,
                            f"label-single:{label}:{category}:{profile.profile_id}",
                        ),
                        profile.profile_id,
                        profile,
                    )
                    for profile in category_profiles
                ]
                heapq.heapify(heap)
                for _ in range(int(quota)):
                    if not heap:
                        raise ValueError(
                            "Explicit label quota exhausted profile capacity: "
                            f"label={label}, category={category}, quota={quota}"
                        )
                    usage, tie_breaker, profile_id, profile = heapq.heappop(heap)
                    bundles.append((profile, None))
                    next_usage = usage + 1
                    if (
                        profile.primary_task_cap is None
                        or next_usage < profile.primary_task_cap
                    ):
                        heapq.heappush(
                            heap,
                            (next_usage, tie_breaker, profile_id, profile),
                        )
        return bundles, label_category_quotas

    maximum_pair_plans: dict[int, list[tuple[RuleProfile, RuleProfile]]] = {}
    maximum_pair_counts: dict[int, int] = {}
    for label in (0, 1):
        label_profiles = [
            profile for profile in profiles if profile.rule.label == label
        ]
        label_edges = [
            edge for edge in compatible_edges if edge[0].rule.label == label
        ]
        label_count = int(label_targets[label])
        if label_count == 0 or requested_two_rule_tasks == 0:
            maximum_pair_plans[label] = []
            maximum_pair_counts[label] = 0
            continue
        label_available = {
            category: int(label_category_quotas[label].get(category, 0))
            for category in available_by_category
        }
        plan = _plan_pair_bundles(
            label_edges,
            requested=label_count,
            available_by_category=label_available,
            seed=stable_hash64(seed, f"label-pair-plan:{label}"),
        )
        maximum = _largest_balanced_pair_prefix(
            plan,
            label_profiles,
            task_count=label_count,
        )
        maximum_pair_plans[label] = plan
        maximum_pair_counts[label] = maximum

    schedulable_pair_tasks = min(
        requested_two_rule_tasks, sum(maximum_pair_counts.values())
    )
    if schedulable_pair_tasks:
        pair_quotas_raw = balanced_category_quotas(
            {str(label): maximum_pair_counts[label] for label in (0, 1)},
            {str(label): int(label_targets[label]) for label in (0, 1)},
            count=schedulable_pair_tasks,
            seed=stable_hash64(seed, "label-pair-quotas"),
        )
    else:
        pair_quotas_raw = {}
    pair_quotas = {
        label: int(pair_quotas_raw.get(str(label), 0)) for label in (0, 1)
    }

    bundles: list[tuple[RuleProfile, RuleProfile | None]] = []
    for label in (0, 1):
        label_count = int(label_targets[label])
        if label_count == 0:
            continue
        label_profiles = [
            profile for profile in profiles if profile.rule.label == label
        ]
        label_available = {
            category: int(label_category_quotas[label].get(category, 0))
            for category in available_by_category
        }
        bundles.extend(
            _fill_single_bundles(
                maximum_pair_plans[label][: pair_quotas[label]],
                label_profiles,
                task_count=label_count,
                available_by_category=label_available,
                seed=stable_hash64(seed, f"label-single-fill:{label}"),
            )
        )
    return bundles, label_category_quotas


def build_balanced_rule_schedule(
    donors: pd.DataFrame,
    rules: Iterable[MutationRule],
    *,
    count: int,
    seed: int,
    two_rule_fraction: float,
    categories: Iterable[str] | None = None,
    semantic_signature_limit: int = 2,
    label_one_fraction: float | None = None,
) -> BalancedRuleSchedule:
    """Build a deterministic donor/rule plan before any Qwen requests run.

    The caller should pass the already-filtered main rule tiers.  Every task keeps
    its scheduled bundle across generation and task retries, so checkpoint resume
    and worker completion order cannot alter rule exposure.
    """

    missing = {"id", "category"} - set(donors.columns)
    if missing:
        raise ValueError(f"Donors are missing columns: {sorted(missing)}")
    if not 0.0 <= two_rule_fraction <= 1.0:
        raise ValueError("two_rule_fraction must be in [0, 1]")
    if label_one_fraction is not None and not 0.0 <= label_one_fraction <= 1.0:
        raise ValueError("label_one_fraction must be in [0, 1]")
    if semantic_signature_limit < 1:
        raise ValueError("semantic_signature_limit must be positive")
    donor_rows = [
        {"id": int(row.id), "category": str(row.category)}
        for row in donors[["id", "category"]].itertuples(index=False)
    ]
    donor_ids = [row["id"] for row in donor_rows]
    if len(donor_ids) != len(set(donor_ids)):
        raise ValueError("Donors contain duplicate IDs")
    donor_categories = {row["category"] for row in donor_rows}
    rule_values = list(rules)
    rule_categories = {
        category for rule in rule_values for category in rule.allowed_categories
    }
    requested_categories = set(categories) if categories else (
        donor_categories & rule_categories
    )
    missing_donors = requested_categories - donor_categories
    if missing_donors:
        raise ValueError(
            "Requested categories are absent from donors: "
            + ", ".join(sorted(missing_donors))
        )
    missing_rules = requested_categories - rule_categories
    if missing_rules:
        raise ValueError(
            "Requested categories have no eligible rules: "
            + ", ".join(sorted(missing_rules))
        )
    policy_versions = {
        rule.profile_capacity_policy_version
        for rule in rule_values
        if rule.profile_capacity_policy_version
    }
    policy_hashes = {
        rule.profile_capacity_policy_sha256
        for rule in rule_values
        if rule.profile_capacity_policy_sha256
    }
    if len(policy_versions) > 1 or len(policy_hashes) > 1:
        raise ValueError("Rules mix multiple profile capacity policies")
    if bool(policy_versions) != bool(policy_hashes):
        raise ValueError("Rules have incomplete profile capacity policy provenance")
    profiles = _rule_profiles(
        rule_values,
        requested_categories,
        semantic_signature_limit=semantic_signature_limit,
    )
    profiles_by_category: dict[str, list[RuleProfile]] = defaultdict(list)
    for profile in profiles:
        profiles_by_category[profile.category].append(profile)
    for values in profiles_by_category.values():
        values.sort(key=lambda profile: profile.profile_id)
    donor_ids_by_category: dict[str, list[int]] = defaultdict(list)
    for row in donor_rows:
        if row["category"] in profiles_by_category:
            donor_ids_by_category[row["category"]].append(row["id"])
    for category, values in donor_ids_by_category.items():
        values.sort(
            key=lambda donor_id: (
                stable_hash64(seed, f"donor:{category}:{donor_id}"),
                donor_id,
            )
        )
    available = {
        category: len(donor_ids_by_category[category])
        for category in sorted(profiles_by_category)
    }
    profile_counts = {
        category: len(values)
        for category, values in sorted(profiles_by_category.items())
    }
    requested_two_rule_tasks = int(math.floor(count * two_rule_fraction + 0.5))
    compatible_edges = _compatible_profile_edges(profiles_by_category)
    if label_one_fraction is None:
        planned_pair_bundles = _plan_pair_bundles(
            compatible_edges,
            requested=requested_two_rule_tasks,
            available_by_category=available,
            seed=seed,
        )
        scheduled_two_rule_tasks = _largest_balanced_pair_prefix(
            planned_pair_bundles,
            profiles,
            task_count=count,
        )
        planned_bundles = _fill_single_bundles(
            planned_pair_bundles[:scheduled_two_rule_tasks],
            profiles,
            task_count=count,
            available_by_category=available,
            seed=seed,
        )
        label_category_task_quotas: dict[int, dict[str, int]] = {}
    else:
        label_one_tasks = _rounded_task_count(count, label_one_fraction)
        label_targets = {0: count - label_one_tasks, 1: label_one_tasks}
        planned_bundles, label_category_task_quotas = _quota_planned_bundles(
            profiles,
            compatible_edges,
            available,
            label_targets=label_targets,
            requested_two_rule_tasks=requested_two_rule_tasks,
            count=count,
            seed=seed,
        )
    category_bundles: dict[
        str, list[tuple[RuleProfile, RuleProfile | None]]
    ] = defaultdict(list)
    for primary, secondary in planned_bundles:
        category_bundles[primary.category].append((primary, secondary))
    unsorted_entries: list[ScheduledRuleBundle] = []
    for category in sorted(category_bundles):
        bundles = sorted(
            category_bundles[category],
            key=lambda bundle: (
                stable_hash64(
                    seed,
                    "bundle-donor:"
                    + ":".join(profile.profile_id for profile in bundle if profile),
                ),
                bundle[0].profile_id,
                bundle[1].profile_id if bundle[1] is not None else "",
            ),
        )
        selected_donors = donor_ids_by_category[category][: len(bundles)]
        if len(selected_donors) != len(bundles):
            raise RuntimeError(f"Donor capacity mismatch for category {category}")
        for donor_id, (primary, secondary) in zip(
            selected_donors, bundles, strict=True
        ):
            unsorted_entries.append(
                ScheduledRuleBundle(
                    task_index=-1,
                    donor_id=donor_id,
                    category=category,
                    primary=primary,
                    secondary=secondary,
                )
            )
    unsorted_entries.sort(
        key=lambda entry: (
            stable_hash64(seed, f"task-order:{entry.category}:{entry.donor_id}"),
            entry.category,
            entry.donor_id,
        )
    )
    final_entries = [
        replace(entry, task_index=index)
        for index, entry in enumerate(unsorted_entries)
    ]
    quotas = dict(Counter(entry.category for entry in final_entries))
    if not label_category_task_quotas:
        label_category_task_quotas = {
            label: dict(
                sorted(
                    Counter(
                        entry.category
                        for entry in final_entries
                        if entry.target == label
                    ).items()
                )
            )
            for label in (0, 1)
        }
    compatible_profile_ids = {
        profile.profile_id for edge in compatible_edges for profile in edge
    }
    eligible_pair_slots = sum(
        1 for entry in final_entries if entry.primary.profile_id in compatible_profile_ids
    )

    schedule_hash = _schedule_sha256(final_entries)
    return BalancedRuleSchedule(
        entries=tuple(final_entries),
        eligible_profiles=tuple(
            sorted(profiles, key=lambda profile: profile.profile_id)
        ),
        seed=seed,
        requested_two_rule_fraction=two_rule_fraction,
        requested_two_rule_tasks=requested_two_rule_tasks,
        eligible_two_rule_primary_slots=eligible_pair_slots,
        category_available_donors=tuple(sorted(available.items())),
        category_task_quotas=tuple(sorted(quotas.items())),
        category_profile_counts=tuple(sorted(profile_counts.items())),
        semantic_signature_limit=semantic_signature_limit,
        profile_capacity_policy_version=next(iter(policy_versions), ""),
        profile_capacity_policy_sha256=next(iter(policy_hashes), ""),
        requested_label_one_fraction=label_one_fraction,
        label_category_task_quotas=tuple(
            (
                label,
                tuple(sorted(category_counts.items())),
            )
            for label, category_counts in sorted(label_category_task_quotas.items())
        ),
        schedule_sha256=schedule_hash,
    )
