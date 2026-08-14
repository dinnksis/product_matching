from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation_splits import (
    assert_item_disjoint,
    build_hard_features,
    hard_selection_targets,
    item_ids,
    sample_components_to_pair_count,
    select_hard_anchors,
    stable_component_ids,
)
from src.data_pipeline import serialize_product


DEFAULT_OUTPUT = ROOT / "prepared" / "validation_splits_v1"
DEFAULT_OOD_CATEGORIES = ("Одежда", "Бытовая техника")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze human IID/hard/OOD validation splits and split LLM data"
    )
    parser.add_argument("--human-items", type=Path, default=ROOT / "data/items_human.parquet")
    parser.add_argument("--human-pairs", type=Path, default=ROOT / "data/matches.parquet")
    parser.add_argument(
        "--old-human-validation",
        type=Path,
        default=ROOT / "prepared/human/val_pairs.parquet",
        help="Existing component-disjoint holdout used by the frozen MiniLM v1",
    )
    parser.add_argument(
        "--minilm-predictions",
        type=Path,
        default=ROOT / "data/runs/validation_predictions_v1.parquet",
    )
    parser.add_argument(
        "--light-predictions",
        type=Path,
        default=ROOT / "data/runs/light_validation_predictions.parquet",
    )
    parser.add_argument("--llm-items", type=Path, default=ROOT / "data/llm_data/items.parquet")
    parser.add_argument(
        "--llm-pairs", type=Path, default=ROOT / "data/llm_data/matches_llm.parquet"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iid-size", type=int, default=12_000)
    parser.add_argument(
        "--hard-size",
        type=int,
        default=5_200,
        help="Number of ranked hard anchors; full components yield about 5-6k pairs",
    )
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--ood-category",
        action="append",
        dest="ood_categories",
        help="Repeat for each held-out category; defaults to clothing and appliances",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Build only human splits (useful for a quick dry run)",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_keys(frame: pd.DataFrame) -> pd.MultiIndex:
    first = frame["id1"].to_numpy(dtype=np.int64)
    second = frame["id2"].to_numpy(dtype=np.int64)
    return pd.MultiIndex.from_arrays(
        [np.minimum(first, second), np.maximum(first, second)],
        names=["low", "high"],
    )


def category_table(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for category, group in frame.groupby("category", sort=True):
        positives = int(group["target"].sum())
        result[str(category)] = {
            "pairs": len(group),
            "positives": positives,
            "negatives": len(group) - positives,
            "positive_rate": positives / len(group),
        }
    return result


def write_pairs(frame: pd.DataFrame, path: Path) -> None:
    frame[["id1", "id2", "target"]].to_parquet(path, index=False)


def prepare_human_splits(
    args: argparse.Namespace,
    output: Path,
    ood_categories: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    logging.info("Reading human items and pairs")
    items = pd.read_parquet(args.human_items)
    required_item_columns = {"id", "name", "attributes", "category"}
    missing_item_columns = required_item_columns.difference(items.columns)
    if missing_item_columns:
        raise ValueError(
            f"Human items are missing columns: {sorted(missing_item_columns)}"
        )
    items = items.copy()
    items["product_text"] = items.apply(
        serialize_product,
        axis=1,
        max_attribute_chars=6000,
    )
    pairs = pd.read_parquet(args.human_pairs, columns=["id1", "id2", "target"])
    old_validation = pd.read_parquet(args.old_human_validation)
    predictions = pd.read_parquet(args.minilm_predictions)
    light_predictions = pd.read_parquet(args.light_predictions)

    if not pairs["target"].isin([0.0, 1.0]).all():
        raise ValueError("Human pairs must have binary targets")
    if pairs[["id1", "id2"]].duplicated().any():
        raise ValueError("Human pairs contain duplicate oriented pairs")

    category_by_id = items.set_index("id", verify_integrity=True)["category"]
    pairs = pairs.copy()
    pairs.insert(0, "human_row_id", np.arange(len(pairs), dtype=np.int64))
    pairs["category"] = pairs["id1"].map(category_by_id)
    second_category = pairs["id2"].map(category_by_id)
    if pairs["category"].isna().any() or second_category.isna().any():
        raise ValueError("Human pairs reference missing items")
    if not pairs["category"].eq(second_category).all():
        raise ValueError("Human pairs contain cross-category rows")
    unknown_ood = set(ood_categories).difference(pairs["category"].unique())
    if unknown_ood:
        raise ValueError(f"Unknown OOD categories: {sorted(unknown_ood)}")

    logging.info("Building connected components over %d human pairs", len(pairs))
    pairs["component_id"] = stable_component_ids(pairs)

    pair_lookup = pairs.set_index(canonical_keys(pairs), verify_integrity=True)
    old_validation_keys = canonical_keys(old_validation)
    missing_old_validation = old_validation_keys.difference(pair_lookup.index)
    if len(missing_old_validation):
        raise ValueError("Old validation contains pairs absent from human source")
    old_validation_rows = pair_lookup.loc[old_validation_keys].reset_index(drop=True)
    if not np.array_equal(
        old_validation_rows["target"].to_numpy(),
        old_validation["target"].to_numpy(),
    ):
        raise ValueError("Old validation targets differ from human source")

    prediction_keys = canonical_keys(predictions)
    if not prediction_keys.equals(old_validation_keys):
        raise ValueError("MiniLM predictions are not aligned with the old validation")

    candidate_rows = old_validation_rows.loc[
        ~old_validation_rows["category"].isin(ood_categories)
    ].copy()
    if candidate_rows["component_id"].isin(
        pairs.loc[~pairs["human_row_id"].isin(old_validation_rows["human_row_id"]), "component_id"]
    ).any():
        raise ValueError("Old validation components overlap the old training split")

    logging.info("Sampling whole components totaling %d unbiased IID pairs", args.iid_size)
    iid_anchor_components = sample_components_to_pair_count(
        candidate_rows,
        args.iid_size,
        seed=args.seed,
    )
    iid_rows = pairs.loc[pairs["component_id"].isin(iid_anchor_components)].copy()
    if iid_rows["category"].isin(ood_categories).any():
        raise RuntimeError("IID expansion reached an OOD category")

    logging.info("Computing frozen hard-example features")
    hard_features = build_hard_features(predictions, light_predictions)
    feature_lookup = hard_features.set_index(canonical_keys(hard_features), verify_integrity=True)
    hard_candidates = candidate_rows.loc[
        ~candidate_rows["component_id"].isin(iid_anchor_components)
    ].copy()
    hard_candidate_keys = canonical_keys(hard_candidates)
    hard_features = feature_lookup.loc[hard_candidate_keys].reset_index(drop=True)
    metadata_columns = ["human_row_id", "component_id", "category"]
    for column in metadata_columns:
        hard_features[column] = hard_candidates[column].to_numpy()

    logging.info("Selecting %d hard anchors", args.hard_size)
    hard_anchors = select_hard_anchors(hard_features, args.hard_size)
    hard_components = set(hard_anchors["component_id"].astype(np.int64))
    if hard_components & iid_anchor_components:
        raise RuntimeError("IID and hard components overlap")
    hard_rows = pairs.loc[pairs["component_id"].isin(hard_components)].copy()

    is_ood = pairs["category"].isin(ood_categories)
    is_iid = pairs["component_id"].isin(iid_anchor_components)
    is_hard = pairs["component_id"].isin(hard_components)
    if (is_ood.astype(np.int8) + is_iid.astype(np.int8) + is_hard.astype(np.int8) > 1).any():
        raise RuntimeError("Human split masks overlap")
    assignments = np.full(len(pairs), "train", dtype=object)
    assignments[is_ood] = "ood_validation"
    assignments[is_iid] = "iid_validation"
    assignments[is_hard] = "hard_validation"
    pairs["split"] = assignments

    split_frames = {
        name: group.reset_index(drop=True)
        for name, group in pairs.groupby("split", sort=True)
    }
    expected_splits = {"train", "iid_validation", "hard_validation", "ood_validation"}
    if set(split_frames) != expected_splits:
        raise RuntimeError(f"Unexpected human splits: {sorted(split_frames)}")
    assert_item_disjoint(split_frames.items())
    if sum(map(len, split_frames.values())) != len(pairs):
        raise RuntimeError("Human split row counts do not cover the source")

    human_output = output / "human"
    human_output.mkdir(parents=True)
    # Keep the frozen Dataset self-contained: every split stores only pair IDs,
    # while this one item table supplies the text and category for training and
    # all three validation protocols.
    items[["id", "name", "category", "product_text"]].to_parquet(
        human_output / "items.parquet",
        index=False,
        compression="zstd",
    )
    for name, frame in split_frames.items():
        write_pairs(frame, human_output / f"{name}_pairs.parquet")
    pairs[
        ["human_row_id", "component_id", "id1", "id2", "target", "category", "split"]
    ].to_parquet(human_output / "split_assignments.parquet", index=False)

    hard_detail = hard_features.loc[
        hard_features["component_id"].isin(hard_components)
    ].copy()
    reason_by_component = hard_anchors.set_index("component_id")["selection_reason"].to_dict()
    anchor_rows = set(hard_anchors["human_row_id"].astype(np.int64))
    hard_detail["selection_reason"] = [
        reason_by_component[int(component)]
        if int(row_id) in anchor_rows
        else "component_companion"
        for row_id, component in zip(hard_detail["human_row_id"], hard_detail["component_id"])
    ]
    hard_detail.sort_values("human_row_id").to_parquet(
        human_output / "hard_selection_details.parquet",
        index=False,
    )

    targets = hard_selection_targets(args.hard_size)
    overlap_counts: dict[str, int] = {}
    split_names = sorted(split_frames)
    id_sets = {name: item_ids(frame) for name, frame in split_frames.items()}
    for position, first in enumerate(split_names):
        for second in split_names[position + 1 :]:
            overlap_counts[f"{first}__{second}"] = len(id_sets[first] & id_sets[second])

    report = {
        "source_pairs": len(pairs),
        "source_items": len(items),
        "ood_categories": list(ood_categories),
        "iid_requested_pairs": args.iid_size,
        "hard_requested_anchors": args.hard_size,
        "hard_anchor_targets": {
            "confident_minilm_v1_error": targets.model_errors,
            "lexical_surprise": targets.lexical_surprises,
            "model_disagreement_or_order_gap": targets.diagnostic,
        },
        "hard_anchor_actual": {
            str(key): int(value)
            for key, value in hard_anchors["selection_reason"].value_counts().items()
        },
        "hard_component_companions": len(hard_rows) - len(hard_anchors),
        "splits": {
            name: {
                "pairs": len(frame),
                "items": len(id_sets[name]),
                "components": int(frame["component_id"].nunique()),
                "positives": int(frame["target"].sum()),
                "positive_rate": float(frame["target"].mean()),
                "categories": category_table(frame),
            }
            for name, frame in split_frames.items()
        },
        "item_overlap_counts": overlap_counts,
    }
    return report, split_frames


def _writers(
    schema: pa.Schema,
    non_ood_path: Path,
    ood_path: Path,
) -> tuple[pq.ParquetWriter, pq.ParquetWriter]:
    options = {"compression": "zstd", "use_dictionary": True}
    return (
        pq.ParquetWriter(non_ood_path, schema, **options),
        pq.ParquetWriter(ood_path, schema, **options),
    )


def split_llm_data(
    items_path: Path,
    pairs_path: Path,
    output: Path,
    ood_categories: tuple[str, ...],
) -> dict[str, Any]:
    output.mkdir(parents=True)
    items_file = pq.ParquetFile(items_path)
    category_values = pa.array(ood_categories, type=pa.string())
    non_ood_items_path = output / "non_ood_items.parquet"
    ood_items_path = output / "ood_items.parquet"
    ood_id_parts: list[pa.Array] = []
    item_counts = {"non_ood": 0, "ood": 0}

    logging.info("Streaming and splitting %d LLM items", items_file.metadata.num_rows)
    non_ood_writer, ood_writer = _writers(
        items_file.schema_arrow,
        non_ood_items_path,
        ood_items_path,
    )
    try:
        for batch_number, batch in enumerate(items_file.iter_batches(batch_size=262_144), 1):
            table = pa.Table.from_batches([batch])
            is_ood = pc.is_in(table["category"], value_set=category_values)
            ood_table = table.filter(is_ood)
            non_ood_table = table.filter(pc.invert(is_ood))
            if len(ood_table):
                ood_writer.write_table(ood_table)
                ood_id_parts.extend(ood_table["id"].chunks)
                item_counts["ood"] += len(ood_table)
            if len(non_ood_table):
                non_ood_writer.write_table(non_ood_table)
                item_counts["non_ood"] += len(non_ood_table)
            if batch_number % 10 == 0:
                logging.info(
                    "LLM items: processed %s rows",
                    f'{item_counts["ood"] + item_counts["non_ood"]:,}',
                )
    finally:
        non_ood_writer.close()
        ood_writer.close()

    ood_ids = pa.chunked_array(ood_id_parts).combine_chunks()
    pairs_file = pq.ParquetFile(pairs_path)
    non_ood_pairs_path = output / "non_ood_pairs.parquet"
    ood_pairs_path = output / "ood_pairs.parquet"
    pair_counts = {"non_ood": 0, "ood": 0}
    target_counts = {
        "non_ood": {},
        "ood": {},
    }
    logging.info("Streaming and splitting %d LLM pairs", pairs_file.metadata.num_rows)
    non_ood_writer, ood_writer = _writers(
        pairs_file.schema_arrow,
        non_ood_pairs_path,
        ood_pairs_path,
    )
    try:
        for batch_number, batch in enumerate(pairs_file.iter_batches(batch_size=524_288), 1):
            table = pa.Table.from_batches([batch])
            first_ood = pc.is_in(table["id1"], value_set=ood_ids)
            second_ood = pc.is_in(table["id2"], value_set=ood_ids)
            mismatch = pc.any(pc.not_equal(first_ood, second_ood)).as_py()
            if mismatch:
                raise ValueError("LLM pairs contain a cross-OOD-boundary pair")
            ood_table = table.filter(first_ood)
            non_ood_table = table.filter(pc.invert(first_ood))
            if len(ood_table):
                ood_writer.write_table(ood_table)
                pair_counts["ood"] += len(ood_table)
            if len(non_ood_table):
                non_ood_writer.write_table(non_ood_table)
                pair_counts["non_ood"] += len(non_ood_table)
            for label, part in (("ood", ood_table), ("non_ood", non_ood_table)):
                if not len(part):
                    continue
                values = pc.value_counts(part["target"]).to_pylist()
                for value in values:
                    key = str(float(value["values"]))
                    target_counts[label][key] = target_counts[label].get(key, 0) + int(
                        value["counts"]
                    )
            if batch_number % 5 == 0:
                logging.info(
                    "LLM pairs: processed %s rows",
                    f'{pair_counts["ood"] + pair_counts["non_ood"]:,}',
                )
    finally:
        non_ood_writer.close()
        ood_writer.close()

    if sum(item_counts.values()) != items_file.metadata.num_rows:
        raise RuntimeError("LLM item split does not cover the source")
    if sum(pair_counts.values()) != pairs_file.metadata.num_rows:
        raise RuntimeError("LLM pair split does not cover the source")
    return {
        "items": item_counts,
        "pairs": pair_counts,
        "target_counts": target_counts,
        "files": {
            "non_ood_items": str(non_ood_items_path.relative_to(output.parent)),
            "ood_items": str(ood_items_path.relative_to(output.parent)),
            "non_ood_pairs": str(non_ood_pairs_path.relative_to(output.parent)),
            "ood_pairs": str(ood_pairs_path.relative_to(output.parent)),
        },
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.iid_size <= 0 or args.hard_size <= 0:
        raise ValueError("IID and hard sizes must be positive")
    ood_categories = tuple(args.ood_categories or DEFAULT_OOD_CATEGORIES)
    if len(ood_categories) != 2 or len(set(ood_categories)) != 2:
        raise ValueError("Exactly two distinct OOD categories are required")

    output = args.output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}; pass --overwrite")

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        human_report, _ = prepare_human_splits(args, staging, ood_categories)
        llm_report = None
        if not args.skip_llm:
            llm_report = split_llm_data(
                args.llm_items,
                args.llm_pairs,
                staging / "llm",
                ood_categories,
            )

        source_paths = [
            ROOT / "scripts/prepare_validation_splits.py",
            ROOT / "src/data_pipeline.py",
            ROOT / "src/validation_splits.py",
            args.human_items,
            args.human_pairs,
            args.old_human_validation,
            args.minilm_predictions,
            args.light_predictions,
        ]
        if not args.skip_llm:
            source_paths.extend([args.llm_items, args.llm_pairs])
        logging.info("Hashing source files for the immutable manifest")
        manifest = {
            "version": "human_v1",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "seed": args.seed,
            "ood_categories": list(ood_categories),
            "sources": {
                str(path.resolve().relative_to(ROOT)): {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in source_paths
            },
            "human": human_report,
            "llm": llm_report,
            "outputs": {
                str(path.relative_to(staging)): {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(staging.rglob("*.parquet"))
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if output.exists():
            if not args.overwrite:
                raise FileExistsError(output)
            shutil.rmtree(output)
        os.replace(staging, output)
        logging.info("Saved frozen validation split to %s", output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
