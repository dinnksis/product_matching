"""Create the frozen MiniLM 5ep data ablation with 10k generated rule pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks/minilm_5ep_team_ablation/minilm_5ep_team_ablation_2xt4.ipynb"
OUTPUT = ROOT / (
    "notebooks/minilm_5ep_team_ablation/"
    "minilm_5ep_generation_rules_10k_v2_2xt4.ipynb"
)
PAIR_FILENAME = "generation_rule_pairs_10k.parquet"
ITEM_FILENAME = "generation_rule_items_10k.parquet"
EXPERIMENT_LABEL = "minilm_5ep_generation_rules_10k_v2"


def default_artifact_tag(pair_count: int) -> str:
    return "10k" if pair_count == 10_000 else str(pair_count)


def data_hook(
    pair_count: int,
    artifact_tag: str,
    dataset_ref: str,
    upload_manifest_sha256: str,
    expected_label_source: str,
) -> str:
    pair_filename = f"generation_rule_pairs_{artifact_tag}.parquet"
    item_filename = f"generation_rule_items_{artifact_tag}.parquet"
    return f'''def build_train_data(human_train_pairs, human_items, input_root):
    import hashlib
    import json

    def sha256_file(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    matching_manifests = []
    for manifest_path in input_root.glob("**/upload_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if manifest.get("dataset") == {dataset_ref!r}:
            matching_manifests.append((manifest_path, manifest))
    if len(matching_manifests) != 1:
        raise RuntimeError(
            "Expected exactly one pinned generated-data manifest; "
            f"matches={{[str(path) for path, _ in matching_manifests]}}"
        )
    manifest_path, upload_manifest = matching_manifests[0]
    actual_manifest_sha256 = sha256_file(manifest_path)
    expected_manifest_sha256 = {upload_manifest_sha256!r}
    if expected_manifest_sha256 and actual_manifest_sha256 != expected_manifest_sha256:
        raise RuntimeError(
            "Generated Dataset manifest SHA mismatch: "
            f"{{actual_manifest_sha256}} != {{expected_manifest_sha256}}"
        )
    if upload_manifest.get("pairs") != {pair_count}:
        raise ValueError("Generated Dataset manifest pair count mismatch")
    if upload_manifest.get("label_source") != {expected_label_source!r}:
        raise ValueError("Generated Dataset manifest label_source mismatch")

    generated_root = manifest_path.parent
    pair_candidates = [generated_root / {pair_filename!r}]
    item_candidates = [generated_root / {item_filename!r}]
    if not pair_candidates[0].is_file() or not item_candidates[0].is_file():
        raise RuntimeError(
            "Pinned generated Dataset is missing pair/item file; "
            f"pairs={{pair_candidates}}, items={{item_candidates}}"
        )
    manifest_files = upload_manifest.get("files") or {{}}
    for generated_path in [pair_candidates[0], item_candidates[0]]:
        expected_sha = (manifest_files.get(generated_path.name) or {{}}).get("sha256")
        actual_sha = sha256_file(generated_path)
        if not expected_sha or actual_sha != expected_sha:
            raise RuntimeError(
                f"Generated file SHA mismatch for {{generated_path.name}}: "
                f"{{actual_sha}} != {{expected_sha}}"
            )

    extra_pairs = pd.read_parquet(pair_candidates[0])
    extra_items = pd.read_parquet(item_candidates[0])
    required_pair_columns = {{"id1", "id2", "target", "label_source"}}
    required_item_columns = {{"id", "name", "category", "product_text"}}
    if missing := required_pair_columns - set(extra_pairs.columns):
        raise ValueError(f"Generated pairs missing columns: {{sorted(missing)}}")
    if missing := required_item_columns - set(extra_items.columns):
        raise ValueError(f"Generated items missing columns: {{sorted(missing)}}")
    if len(extra_pairs) != {pair_count} or len(extra_items) != {pair_count * 2}:
        raise ValueError(
            f"Expected {pair_count:,} pairs and {pair_count * 2:,} items, got "
            f"{{len(extra_pairs)}} and {{len(extra_items)}}"
        )
    if extra_pairs[["id1", "id2", "target"]].isna().any().any():
        raise ValueError("Generated pairs contain null ids/targets")
    if not extra_pairs["target"].eq(0).all():
        raise ValueError("Rule-first generation v2 is expected to contain only negatives")
    if set(extra_pairs["label_source"].astype(str)) != {{{expected_label_source!r}}}:
        raise ValueError("Unexpected generated label_source")
    if extra_items["id"].duplicated().any():
        raise ValueError("Generated item ids are not unique")
    extra_ids = set(extra_items["id"])
    pair_ids = set(extra_pairs["id1"]) | set(extra_pairs["id2"])
    if extra_ids != pair_ids:
        raise ValueError("Generated item catalogue does not exactly match pair ids")
    extra_categories = extra_items.set_index("id")["category"]
    if not extra_pairs["id1"].map(extra_categories).equals(
        extra_pairs["id2"].map(extra_categories)
    ):
        raise ValueError("Generated pairs contain cross-category rows")

    base_pairs = human_train_pairs.copy()
    base_pairs["sample_weight"] = 1.0
    base_pairs["label_source"] = "human"
    extra_pairs = extra_pairs.copy()
    extra_pairs["sample_weight"] = 1.0
    extra_pairs["label_source"] = extra_pairs["label_source"].astype(str)
    train_pairs = pd.concat(
        [base_pairs, extra_pairs[list(base_pairs.columns)]], ignore_index=True
    )
    items = pd.concat(
        [human_items, extra_items[list(human_items.columns)]], ignore_index=True
    )
    if items["id"].duplicated().any():
        raise ValueError("Human and generated item ids overlap")
    print({{
        "human_pairs": len(base_pairs),
        "generated_pairs": len(extra_pairs),
        "total_pairs": len(train_pairs),
        "human_weight": float(base_pairs["sample_weight"].sum()),
        "generated_weight": float(extra_pairs["sample_weight"].sum()),
        "label_sources": train_pairs["label_source"].value_counts().to_dict(),
        "generated_dataset_ref": {dataset_ref!r},
        "generated_upload_manifest_sha256": actual_manifest_sha256,
    }})
    return train_pairs, items
'''


def routing_cell(
    pair_count: int,
    experiment_label: str,
    dataset_ref: str,
    notes_override: str | None = None,
) -> str:
    notes = notes_override or (
        f"Frozen MiniLM 5ep baseline human train plus {pair_count:,} Qwen-generated "
        "rule-first negative pairs. Each anchor explicitly contains the target "
        "attributes, and every mutation passes exact attribute/title consistency "
        "checks. Unit sample weight. Validation splits, checkpoint and training "
        "recipe unchanged. The larger train set also increases optimizer steps, so "
        f"this is a data+compute ablation. Source dataset {dataset_ref}."
    )
    return (
        f"EXPERIMENT_LABEL = {experiment_label!r}\n"
        "EXPERIMENT_SHEET = 'data_exps'  # pretrain_exps | sft_exps | data_exps\n"
        f"EXPERIMENT_NOTES = {notes!r}\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-count", type=int, default=10_000)
    parser.add_argument("--artifact-tag")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--experiment-label", default=EXPERIMENT_LABEL)
    parser.add_argument(
        "--notes",
        help="Exact experiment notes; defaults to the generic rule-first description",
    )
    parser.add_argument(
        "--dataset-ref",
        default="alexproger23/product-matching-generation-rule-pairs-10k-v2",
    )
    parser.add_argument(
        "--upload-manifest-sha256",
        default="",
        help="pin the exact uploaded generated Dataset payload",
    )
    parser.add_argument(
        "--expected-label-source",
        default="qwen_rule_first_generation_v2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.upload_manifest_sha256 and (
        len(args.upload_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in args.upload_manifest_sha256.casefold()
        )
    ):
        raise ValueError("--upload-manifest-sha256 must be a 64-character SHA-256")
    artifact_tag = args.artifact_tag or default_artifact_tag(args.pair_count)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    source = nbformat.read(SOURCE, as_version=4)
    notebook = nbformat.from_dict(source)
    data_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code"
        and "data-hook" in cell.get("metadata", {}).get("tags", [])
    ]
    routing_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code"
        and "experiment-routing" in cell.get("metadata", {}).get("tags", [])
    ]
    if len(data_cells) != 1 or len(routing_cells) != 1:
        raise RuntimeError(
            f"Expected one data hook and one routing cell, got "
            f"{len(data_cells)} and {len(routing_cells)}"
        )
    data_cells[0].source = data_hook(
        args.pair_count,
        artifact_tag,
        args.dataset_ref,
        args.upload_manifest_sha256.casefold(),
        args.expected_label_source,
    )
    routing_cells[0].source = routing_cell(
        args.pair_count,
        args.experiment_label,
        args.dataset_ref,
        args.notes,
    )

    for original, generated in zip(source.cells, notebook.cells, strict=True):
        tags = original.get("metadata", {}).get("tags", [])
        if "frozen" in tags and original.source != generated.source:
            raise RuntimeError("A frozen notebook cell was modified")

    nbformat.validate(notebook)
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    print(f"Created {output}")


if __name__ == "__main__":
    main()
