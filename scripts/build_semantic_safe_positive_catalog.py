"""Build the run catalog by applying the manual positive-rule allowlist to v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "configs"
    / "generation_rule_catalog_statistical_v1"
    / "semantic_all_pairs_cross_split_p80_support5_scoped_compact_v2.json"
)
DEFAULT_AUDIT_SOURCE = (
    ROOT
    / "configs"
    / "generation_rule_catalog_statistical_v1"
    / "semantic_all_pairs_cross_split_p80_support5_scoped_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "configs"
    / "generation_rule_catalog_statistical_v1"
    / "semantic_all_pairs_cross_split_p80_support5_scoped_compact_safe_positive_v3.json"
)
CATALOG_VERSION = (
    "semantic_all_pairs_cross_split_p80_support5_scoped_compact_safe_positive_v3"
)
POSITIVE_TIER = (
    "SEMANTIC_ALL_PAIRS_LABEL1_MANUAL_ALLOWLIST_COMPACT_SAFE_POSITIVE_V3_"
    "EXPERIMENTAL_CORRELATIONAL"
)
POSITIVE_ALLOWLIST = (
    "gen_sem_all_2bd42ae67368b6da139a",
    "gen_sem_all_fc1bf7245474d5979bbc",
    "gen_sem_all_a29a2640f81133199b22",
    "gen_sem_all_bb884e95495f2e4053ad",
    "gen_sem_all_1fd8ee0362b4a69694eb",
    "gen_sem_all_3eab364eedff30a6ccec",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--audit-source", type=Path, default=DEFAULT_AUDIT_SOURCE)
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


def _profile_diagnostic(rule: dict[str, Any]) -> dict[str, Any]:
    return {
        "generation_rule_id": str(rule["generation_rule_id"]),
        "source_rule_id": str(rule["source_rule_id"]),
        "category": str(rule["allowed_categories"][0]),
        "product_type": str(rule["allowed_product_types"][0]),
        "concept": str(rule["concept"]),
        "attribute_key": str(rule["attribute_key"]),
        "profile_pair_support": int(rule["profile_pair_support"]),
        "profile_target_probability": float(rule["profile_target_probability"]),
    }


def main() -> None:
    args = parse_args()
    source_path = args.source.resolve()
    audit_source_path = args.audit_source.resolve()
    output_path = args.output.resolve()
    source_rules = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source_rules, list):
        raise ValueError("Source catalog must be a JSON rule array")
    by_id = {str(rule["generation_rule_id"]): rule for rule in source_rules}
    if len(by_id) != len(source_rules):
        raise ValueError("Source catalog contains duplicate generation_rule_id values")
    audit_rules = json.loads(audit_source_path.read_text(encoding="utf-8"))
    if not isinstance(audit_rules, list):
        raise ValueError("Audit source catalog must be a JSON rule array")
    audit_by_id = {str(rule["generation_rule_id"]): rule for rule in audit_rules}
    allowlist = set(POSITIVE_ALLOWLIST)
    missing = allowlist - set(audit_by_id)
    if missing:
        raise ValueError(f"Positive allowlist IDs are absent from audit v1: {sorted(missing)}")
    wrong_label = [
        rule_id for rule_id in allowlist if int(audit_by_id[rule_id]["label"]) != 1
    ]
    if wrong_label:
        raise ValueError(f"Positive allowlist contains non-positive rules: {wrong_label}")

    # Compact v2 deliberately has versioned generation IDs. Resolve each
    # human-audited v1 profile by its stable semantic source and product scope,
    # then restore the audited ID in the final run catalog.
    resolved_v2_to_audit: dict[str, str] = {}
    for audit_id in POSITIVE_ALLOWLIST:
        audited = audit_by_id[audit_id]
        matches = [
            rule
            for rule in source_rules
            if int(rule["label"]) == 1
            and str(rule["source_rule_id"]) == str(audited["source_rule_id"])
            and list(rule["allowed_categories"]) == list(audited["allowed_categories"])
            and list(rule["allowed_product_types"])
            == list(audited["allowed_product_types"])
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Audited positive {audit_id} resolves to {len(matches)} compact-v2 profiles"
            )
        resolved_v2_to_audit[str(matches[0]["generation_rule_id"])] = audit_id

    source_positive_ids = {
        str(rule["generation_rule_id"])
        for rule in source_rules
        if int(rule["label"]) == 1
    }
    exported: list[dict[str, Any]] = []
    for source in source_rules:
        label = int(source["label"])
        rule_id = str(source["generation_rule_id"])
        if label == 1 and rule_id not in resolved_v2_to_audit:
            continue
        rule = dict(source)
        if label == 1:
            audited_id = resolved_v2_to_audit[rule_id]
            rule["compact_v2_generation_rule_id"] = rule_id
            rule["generation_rule_id"] = audited_id
            rule["generation_tier"] = POSITIVE_TIER
            rule["manual_positive_allowlist"] = True
            rule["manual_positive_review_version"] = CATALOG_VERSION
        exported.append(rule)

    if len(exported) != 3_680:
        raise RuntimeError(f"Expected 3,680 run rules, got {len(exported)}")
    label_counts = Counter(int(rule["label"]) for rule in exported)
    if label_counts != {0: 3_674, 1: 6}:
        raise RuntimeError(f"Unexpected run label counts: {label_counts}")
    retained_positive_ids = {
        str(rule["generation_rule_id"])
        for rule in exported
        if int(rule["label"]) == 1
    }
    if retained_positive_ids != allowlist:
        raise RuntimeError("Final positive IDs differ from the manual allowlist")

    atomic_write(
        output_path,
        json.dumps(exported, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    rejected_positive_ids = sorted(
        source_positive_ids - set(resolved_v2_to_audit)
    )
    manifest = {
        "schema_version": 1,
        "catalog_version": CATALOG_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source_catalog": str(source_path),
        "source_catalog_sha256": sha256(source_path),
        "manual_audit_catalog": str(audit_source_path),
        "manual_audit_catalog_sha256": sha256(audit_source_path),
        "selection": {
            "retain_all_label0_from_v2": True,
            "positive_selection": "manual_generation_rule_id_allowlist",
            "positive_allowlist": list(POSITIVE_ALLOWLIST),
            "positive_manual_tier": POSITIVE_TIER,
            "recommended_first_experiment_two_rule_fraction": 0.0,
            "evidence_interpretation": "experimental_correlational_not_causal",
        },
        "manual_positive_tier_diagnostics": {
            "source_positive_profiles": len(source_positive_ids),
            "allowed_positive_profiles": len(allowlist),
            "excluded_positive_profiles": len(rejected_positive_ids),
            "allowed_generation_rule_ids": list(POSITIVE_ALLOWLIST),
            "excluded_generation_rule_ids": rejected_positive_ids,
            "allowed_profiles": [
                {
                    **_profile_diagnostic(audit_by_id[audit_id]),
                    "audited_generation_rule_id": audit_id,
                    "resolved_compact_v2_generation_rule_id": next(
                        compact_id
                        for compact_id, resolved_audit_id in resolved_v2_to_audit.items()
                        if resolved_audit_id == audit_id
                    ),
                }
                for audit_id in POSITIVE_ALLOWLIST
            ],
        },
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
        "output": str(output_path),
        "output_sha256": sha256(output_path),
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    atomic_write(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
