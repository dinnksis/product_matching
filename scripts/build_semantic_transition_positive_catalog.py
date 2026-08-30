"""Build the v4 run catalog with manually constrained positive transitions.

The source v3 catalog contains six experimental positive rule profiles.  This
builder keeps every negative profile byte-for-byte at the JSON-object level,
removes the two positive profiles that cannot be made sufficiently precise,
and constrains the remaining four to a small, reviewed set of unordered value
transitions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "generation_rule_catalog_statistical_v1"
DEFAULT_SOURCE = (
    CONFIG_DIR
    / "semantic_all_pairs_cross_split_p80_support5_scoped_compact_safe_positive_v3.json"
)
DEFAULT_OUTPUT = (
    CONFIG_DIR
    / "semantic_all_pairs_cross_split_p80_support5_scoped_compact_transition_positive_v4.json"
)
CATALOG_VERSION = (
    "semantic_all_pairs_cross_split_p80_support5_scoped_compact_transition_positive_v4"
)
POSITIVE_TIER = (
    "SEMANTIC_ALL_PAIRS_LABEL1_MANUAL_TRANSITION_ALLOWLIST_COMPACT_"
    "TRANSITION_POSITIVE_V4_EXPERIMENTAL_CORRELATIONAL"
)
CONTEXT_KEYS = ("Бренд", "Название игры")
LABEL0_NON_EQUIVALENCE_ACTION_GUARD = (
    "The new value must be unquestionably non-equivalent and identify a different "
    "sellable variant. Do not use synonyms, spelling-only, formatting-only, or "
    "unit-only changes, genus-to-subtype changes, overlapping compatibility sets, "
    "or marketing paraphrases."
)
LABEL0_NON_EQUIVALENCE_POSTCONDITION_GUARD = (
    "Verify that the old and new values cannot truthfully describe the same sellable "
    "variant after normalization."
)

# Pair order is retained for readability, but each pair is explicitly
# direction-free: either endpoint may be used for the anchor.
POSITIVE_TRANSITIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "gen_sem_all_2bd42ae67368b6da139a": (
        ("детская", "3 лет"),
        ("детская", "6 лет"),
    ),
    "gen_sem_all_fc1bf7245474d5979bbc": (
        ("настольная игра", "карточная игра"),
        ("настольная игра", "балансир"),
        ("настольная игра", "обучающая игра"),
        ("обучающая игра", "викторина"),
        ("обучающая игра", "викторина, для двоих"),
        ("бродилка", "лабиринт"),
    ),
    "gen_sem_all_a29a2640f81133199b22": (
        ("детская", "3 лет"),
        ("детская", "4 лет"),
        ("детская", "10 лет"),
        ("детская", "10+"),
        ("детская", "13 лет"),
        ("детская", "школьники (7-16)"),
        ("взрослая", "от 18 лет"),
    ),
    "gen_sem_all_1fd8ee0362b4a69694eb": (
        ("детская", "3 лет"),
        ("детская", "3+"),
        ("детская", "4 лет"),
        ("детская", "4+"),
        ("детская", "6 лет"),
        ("детская", "от 7 лет"),
        ("детская", "10 лет"),
        ("детская", "10+"),
    ),
}
POSITIVE_ALLOWLIST = tuple(POSITIVE_TRANSITIONS)
EXCLUDED_V3_POSITIVE_IDS = (
    "gen_sem_all_bb884e95495f2e4053ad",  # science_type
    "gen_sem_all_3eab364eedff30a6ccec",  # game_mechanic
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
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


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _domain(transitions: tuple[tuple[str, str], ...]) -> list[str]:
    return list(dict.fromkeys(value for pair in transitions for value in pair))


def _validate_transition_spec() -> None:
    for rule_id, transitions in POSITIVE_TRANSITIONS.items():
        if not transitions:
            raise ValueError(f"{rule_id}: empty transition allowlist")
        canonical: set[tuple[str, str]] = set()
        for pair in transitions:
            if len(pair) != 2 or not all(value.strip() for value in pair):
                raise ValueError(f"{rule_id}: malformed transition {pair!r}")
            if pair[0].casefold() == pair[1].casefold():
                raise ValueError(f"{rule_id}: transition endpoints are equal: {pair!r}")
            key = tuple(sorted((pair[0].casefold(), pair[1].casefold())))
            if key in canonical:
                raise ValueError(f"{rule_id}: duplicate unordered transition {pair!r}")
            canonical.add(key)


def _constrain_positive(source: dict[str, Any]) -> dict[str, Any]:
    rule_id = str(source["generation_rule_id"])
    transitions = POSITIVE_TRANSITIONS[rule_id]
    rule = dict(source)
    if rule_id == "gen_sem_all_2bd42ae67368b6da139a":
        # The v3 key `возраст от` cannot sensibly contain the value `детская`.
        rule["attribute_key"] = "Возрастная рекомендация"
        rule["attribute_key_source"] = "manual_transition_positive_v4_override"
    rule["generation_tier"] = POSITIVE_TIER
    rule["manual_positive_review_version"] = CATALOG_VERSION
    rule["transition_positive_review_version"] = CATALOG_VERSION
    rule["manual_transition_allowlist"] = True
    rule["allowed_value_transitions"] = [list(pair) for pair in transitions]
    rule["allowed_value_transitions_unordered"] = True
    rule["value_transition_semantics"] = "exact_unordered_pairs"
    rule["target_value_domain"] = _domain(transitions)
    rule["primary_task_safety_cap"] = 2 * len(transitions)
    rule["allowed_anchor_context_keys"] = list(CONTEXT_KEYS)
    rule["required_anchor_context_keys"] = list(CONTEXT_KEYS)
    attribute_key = str(rule["attribute_key"])
    rule["anchor_hint"] = (
        "Создай конкретную настольную игру и обязательно укажи характеристики "
        "«Бренд» и «Название игры». Значение целевой характеристики "
        f"«{attribute_key}» выбери только из одного разрешённого перехода. "
        "Не добавляй факты, которые будут противоречить замене."
    )
    rule["generation_action"] = (
        "Choose exactly one unordered pair from allowed_value_transitions. Set the "
        "anchor target attribute to one endpoint and replace it with the other; "
        "never invent a value outside that pair."
    )
    rule["required_postcondition"] = (
        "The target attribute must change between the two endpoints of the selected "
        "allowed transition. Preserve Brand, Game title, concrete product identity, "
        "and every unrelated fact; update every title and attribute mention "
        "consistently."
    )
    return rule


def _strengthen_negative(source: dict[str, Any]) -> dict[str, Any]:
    """Preserve evidence fields while adding an explicit false-negative guard."""

    rule = dict(source)
    rule["generation_action"] = (
        f"{str(source['generation_action']).rstrip()} "
        f"{LABEL0_NON_EQUIVALENCE_ACTION_GUARD}"
    )
    rule["required_postcondition"] = (
        f"{str(source['required_postcondition']).rstrip()} "
        f"{LABEL0_NON_EQUIVALENCE_POSTCONDITION_GUARD}"
    )
    return rule


def main() -> None:
    args = parse_args()
    source_path = args.source.resolve()
    output_path = args.output.resolve()
    _validate_transition_spec()

    source_rules = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source_rules, list):
        raise ValueError("Source catalog must be a JSON rule array")
    by_id = {str(rule["generation_rule_id"]): rule for rule in source_rules}
    if len(by_id) != len(source_rules):
        raise ValueError("Source catalog contains duplicate generation_rule_id values")

    expected_positive_ids = set(POSITIVE_ALLOWLIST) | set(EXCLUDED_V3_POSITIVE_IDS)
    source_positive_ids = {
        str(rule["generation_rule_id"])
        for rule in source_rules
        if int(rule["label"]) == 1
    }
    if source_positive_ids != expected_positive_ids:
        raise ValueError(
            "Unexpected v3 positive IDs: "
            f"expected {sorted(expected_positive_ids)}, got {sorted(source_positive_ids)}"
        )

    exported: list[dict[str, Any]] = []
    for source in source_rules:
        label = int(source["label"])
        rule_id = str(source["generation_rule_id"])
        if label == 0:
            exported.append(_strengthen_negative(source))
        elif rule_id in POSITIVE_TRANSITIONS:
            exported.append(_constrain_positive(source))

    label_counts = Counter(int(rule["label"]) for rule in exported)
    if len(exported) != 3_678 or label_counts != {0: 3_674, 1: 4}:
        raise RuntimeError(
            f"Unexpected output counts: total={len(exported)}, labels={label_counts}"
        )

    atomic_write(
        output_path,
        json.dumps(exported, ensure_ascii=False, indent=2).encode("utf-8"),
    )

    transition_count = sum(len(value) for value in POSITIVE_TRANSITIONS.values())
    recommended_target1_count = sum(
        2 * len(value) for value in POSITIVE_TRANSITIONS.values()
    )
    manifest = {
        "schema_version": 1,
        "catalog_version": CATALOG_VERSION,
        "source_catalog": _display_path(source_path),
        "source_catalog_sha256": sha256(source_path),
        "selection": {
            "retain_all_label0_from_v3": True,
            "positive_selection": "manual_exact_unordered_transition_allowlist",
            "positive_allowlist": list(POSITIVE_ALLOWLIST),
            "excluded_v3_positive_ids": list(EXCLUDED_V3_POSITIVE_IDS),
            "positive_manual_tier": POSITIVE_TIER,
            "recommended_experiment_pair_count": 10_000,
            "recommended_target1_count": recommended_target1_count,
            "recommended_label_one_fraction": recommended_target1_count / 10_000,
            "recommended_two_rule_fraction": 0.0,
            "label0_non_equivalence_prompt_guard": True,
        },
        "transition_provenance": {
            "method": "manual_review_of_v3_source_examples",
            "semantics": "each listed pair is exact and unordered",
            "rule_profile_evidence": "experimental_correlational_not_causal",
            "cross_split_validation_scope": "rule_profile_only",
            "exact_transitions_cross_split_validated": False,
            "caveat": (
                "The parent positive rule profiles passed cross-split statistics, "
                "but no individual exact value transition was cross-split validated."
            ),
        },
        "transition_rule_count": len(POSITIVE_TRANSITIONS),
        "transition_count": transition_count,
        "transition_capacity": recommended_target1_count,
        "exported_rules": len(exported),
        "label_counts": {str(key): value for key, value in sorted(label_counts.items())},
        "category_counts": dict(
            sorted(
                Counter(
                    str(rule["allowed_categories"][0]) for rule in exported
                ).items()
            )
        ),
        "category_coverage": len(
            {str(rule["allowed_categories"][0]) for rule in exported}
        ),
        "tier_counts": dict(
            sorted(Counter(str(rule["generation_tier"]) for rule in exported).items())
        ),
        "output": _display_path(output_path),
        "output_sha256": sha256(output_path),
        "deterministic_content": True,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    atomic_write(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
