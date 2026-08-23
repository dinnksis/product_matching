"""Prepare label-isolated Qwen inputs for the frozen-rule validation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.product_matching.eda import load_human_data  # noqa: E402


DEFAULT_ASSIGNMENTS = ROOT / "data" / "rule_discovery_split_v1" / "split_assignments.parquet"
DEFAULT_OUTPUT = ROOT / "data" / "qwen_rule_internal_validation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare label-free inputs for rule_internal_validation."
    )
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--size",
        type=int,
        default=0,
        help="Deterministic category-proportional sample size; 0 keeps the full split.",
    )
    parser.add_argument("--seed", type=int, default=2031)
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


def category_proportional_sample(
    frame: pd.DataFrame, size: int, seed: int
) -> pd.DataFrame:
    """Sample without labels while preserving the validation category mix."""
    if size == 0 or size == len(frame):
        return frame.copy()
    if size < 1 or size > len(frame):
        raise ValueError(f"size must be in [1, {len(frame)}], or 0 for all")
    counts = frame["category"].value_counts().sort_index()
    raw_quotas = counts * (size / len(frame))
    quotas = raw_quotas.astype(int)
    remainder = size - int(quotas.sum())
    order = (raw_quotas - quotas).sort_values(ascending=False).index
    for category in order[:remainder]:
        quotas.loc[category] += 1
    ranked = frame.copy()
    ranked["_stable_rank"] = ranked["pair_id"].map(
        lambda pair_id: stable_rank(seed, str(pair_id))
    )
    sampled = [
        group.nsmallest(int(quotas.loc[category]), "_stable_rank")
        for category, group in ranked.groupby("category", sort=True)
    ]
    result = pd.concat(sampled, ignore_index=True).drop(columns="_stable_rank")
    if len(result) != size or result["pair_id"].duplicated().any():
        raise RuntimeError("Deterministic validation sample invariant failed")
    return result


def main() -> None:
    args = parse_args()
    assignments_path = args.assignments.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    assignments = pd.read_parquet(assignments_path)
    validation = assignments[
        assignments["split"].eq("rule_internal_validation")
    ].copy()
    discovery_ids = set(
        assignments.loc[assignments["split"].eq("rule_discovery"), "pair_id"].astype(str)
    )
    validation["pair_id"] = validation["pair_id"].astype(str)
    if validation.empty or validation["pair_id"].duplicated().any():
        raise RuntimeError("Internal-validation pair IDs are empty or non-unique")
    if set(validation["pair_id"]) & discovery_ids:
        raise RuntimeError("RULE_DISCOVERY leaked into rule_internal_validation")

    full_validation_size = len(validation)
    validation = category_proportional_sample(validation, args.size, args.seed)

    items, _ = load_human_data(args.data_dir.resolve())
    item_lookup = items.set_index("id", verify_integrity=True)
    for side, id_column in (("a", "id1"), ("b", "id2")):
        validation[f"title_{side}"] = validation[id_column].map(item_lookup["name"])
        validation[f"attributes_{side}_json"] = validation[id_column].map(
            item_lookup["attributes"]
        )
    required_values = [
        "title_a", "attributes_a_json", "title_b", "attributes_b_json"
    ]
    if validation[required_values].isna().any().any():
        raise RuntimeError("Some validation products were not found by the existing loader")

    validation = validation.sort_values("pair_id").reset_index(drop=True)
    inputs = validation[
        [
            "pair_id", "id1", "id2", "category", "title_a",
            "attributes_a_json", "title_b", "attributes_b_json",
        ]
    ].rename(columns={"id1": "item_id_a", "id2": "item_id_b"})
    forbidden = {"target", "label", "human_label"} & set(inputs.columns)
    if forbidden:
        raise RuntimeError(f"Label leakage into Qwen inputs: {sorted(forbidden)}")
    labels = validation[["pair_id", "target"]].rename(
        columns={"target": "human_label"}
    )
    labels["human_label"] = labels["human_label"].astype(int)
    metadata = validation[["pair_id", "category"]].copy()

    input_path = output / "validation_inputs.parquet"
    label_path = output / "validation_labels.parquet"
    metadata_path = output / "validation_metadata.parquet"
    inputs.to_parquet(input_path, index=False)
    labels.to_parquet(label_path, index=False)
    metadata.to_parquet(metadata_path, index=False)

    manifest = {
        "version": "qwen_rule_internal_validation_v1",
        "pairs": len(inputs),
        "full_validation_pairs": full_validation_size,
        "sampling": "full" if len(inputs) == full_validation_size else "category_proportional_label_free",
        "seed": args.seed,
        "categories": int(inputs["category"].nunique()),
        "positives": int(labels["human_label"].sum()),
        "negatives": int(labels["human_label"].eq(0).sum()),
        "source_split": "rule_internal_validation only",
        "rule_discovery_overlap_pair_ids": 0,
        "ordinary_hard_ood_used": False,
        "qwen_input_contains_human_label": False,
        "assignments": str(assignments_path),
        "assignments_sha256": sha256(assignments_path),
        "validation_inputs_sha256": sha256(input_path),
        "validation_labels_sha256": sha256(label_path),
        "label_join_policy": "labels remain separate and are loaded only after Qwen inference",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
