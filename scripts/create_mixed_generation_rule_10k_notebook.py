"""Create the frozen MiniLM 5ep ablation for mixed generated rule pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / (
    "notebooks/minilm_5ep_team_ablation/minilm_5ep_team_ablation_2xt4.ipynb"
)
DEFAULT_OUTPUT = ROOT / (
    "notebooks/minilm_5ep_team_ablation/"
    "minilm_5ep_semantic_transition_positive_10k_v1_2xt4.ipynb"
)
DEFAULT_PAIR_COUNT = 10_000
DEFAULT_TARGET0 = 9_954
DEFAULT_TARGET1 = 46
DEFAULT_QUOTA_RATIONALE = (
    "The bounded generated-positive quota covers all 23 manually approved value "
    "transitions exactly twice across four context-constrained rules; it does not "
    "extrapolate positive changes beyond the observed transition allowlist."
)


def expected_counts(target0: int, target1: int, pair_count: int) -> dict[str, int]:
    result = {"0": int(target0), "1": int(target1)}
    if (
        min(result.values()) < 0
        or int(pair_count) < 1
        or sum(result.values()) != int(pair_count)
    ):
        raise ValueError(
            "generated target counts must be non-negative and sum to pair-count"
        )
    return result


def data_hook(
    *,
    pair_count: int,
    artifact_tag: str,
    dataset_ref: str,
    upload_manifest_sha256: str,
    label_source: str,
    target_counts: dict[str, int],
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
            "Expected exactly one pinned mixed generated-data manifest; "
            f"matches={{[str(path) for path, _ in matching_manifests]}}"
        )
    manifest_path, upload_manifest = matching_manifests[0]
    actual_manifest_sha256 = sha256_file(manifest_path)
    expected_manifest_sha256 = {upload_manifest_sha256!r}
    if not expected_manifest_sha256:
        raise RuntimeError("Mixed generated Dataset manifest SHA is not pinned")
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise RuntimeError(
            "Mixed generated Dataset manifest SHA mismatch: "
            f"{{actual_manifest_sha256}} != {{expected_manifest_sha256}}"
        )
    if upload_manifest.get("pairs") != {pair_count}:
        raise ValueError("Mixed generated Dataset pair count mismatch")
    if upload_manifest.get("label_source") != {label_source!r}:
        raise ValueError("Mixed generated Dataset label_source mismatch")
    expected_targets = {target_counts!r}
    manifest_targets = {{
        str(key): int(value)
        for key, value in (upload_manifest.get("targets") or {{}}).items()
    }}
    if manifest_targets != expected_targets:
        raise ValueError(
            f"Mixed generated Dataset target counts differ: "
            f"{{manifest_targets}} != {{expected_targets}}"
        )

    generated_root = manifest_path.parent
    pair_path = generated_root / {pair_filename!r}
    item_path = generated_root / {item_filename!r}
    if not pair_path.is_file() or not item_path.is_file():
        raise RuntimeError(
            f"Pinned mixed Dataset is missing files: {{pair_path}}, {{item_path}}"
        )
    manifest_files = upload_manifest.get("files") or {{}}
    for generated_path in (pair_path, item_path):
        expected_sha = (manifest_files.get(generated_path.name) or {{}}).get("sha256")
        actual_sha = sha256_file(generated_path)
        if not expected_sha or actual_sha != expected_sha:
            raise RuntimeError(
                f"Generated file SHA mismatch for {{generated_path.name}}: "
                f"{{actual_sha}} != {{expected_sha}}"
            )

    extra_pairs = pd.read_parquet(pair_path)
    extra_items = pd.read_parquet(item_path)
    required_pair_columns = {{"id1", "id2", "target", "label_source"}}
    required_item_columns = {{"id", "name", "category", "product_text"}}
    if missing := required_pair_columns - set(extra_pairs.columns):
        raise ValueError(f"Mixed generated pairs missing columns: {{sorted(missing)}}")
    if missing := required_item_columns - set(extra_items.columns):
        raise ValueError(f"Mixed generated items missing columns: {{sorted(missing)}}")
    if len(extra_pairs) != {pair_count} or len(extra_items) != {pair_count * 2}:
        raise ValueError(
            f"Expected {pair_count:,} pairs and {pair_count * 2:,} items, got "
            f"{{len(extra_pairs)}} and {{len(extra_items)}}"
        )
    if extra_pairs[["id1", "id2", "target"]].isna().any().any():
        raise ValueError("Mixed generated pairs contain null ids/targets")
    if not extra_pairs["target"].isin([0, 1]).all():
        raise ValueError("Mixed generated pair targets must be binary")
    actual_targets = {{
        str(label): int(extra_pairs["target"].eq(label).sum())
        for label in (0, 1)
    }}
    if actual_targets != expected_targets:
        raise ValueError(
            f"Mixed generated pair target counts differ: "
            f"{{actual_targets}} != {{expected_targets}}"
        )
    if set(extra_pairs["label_source"].astype(str)) != {{{label_source!r}}}:
        raise ValueError("Unexpected mixed generated label_source")
    forbidden_ood_categories = {{"Одежда", "Бытовая техника"}}
    observed_ood_categories = (
        set(extra_items["category"].astype(str)) & forbidden_ood_categories
    )
    if observed_ood_categories:
        raise ValueError(
            f"Mixed generated train uses frozen OOD categories: "
            f"{{sorted(observed_ood_categories)}}"
        )
    if extra_items["id"].duplicated().any():
        raise ValueError("Mixed generated item ids are not unique")
    extra_ids = set(extra_items["id"])
    pair_ids = set(extra_pairs["id1"]) | set(extra_pairs["id2"])
    if extra_ids != pair_ids:
        raise ValueError("Mixed generated item catalogue does not match pair ids")
    category_by_id = extra_items.set_index("id")["category"]
    if not extra_pairs["id1"].map(category_by_id).equals(
        extra_pairs["id2"].map(category_by_id)
    ):
        raise ValueError("Mixed generated pairs contain cross-category rows")

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
        raise ValueError("Human and mixed generated item ids overlap")
    print({{
        "human_pairs": len(base_pairs),
        "generated_pairs": len(extra_pairs),
        "generated_target_counts": actual_targets,
        "total_pairs": len(train_pairs),
        "label_sources": train_pairs["label_source"].value_counts().to_dict(),
        "generated_dataset_ref": {dataset_ref!r},
        "generated_upload_manifest_sha256": actual_manifest_sha256,
    }})
    return train_pairs, items
'''


def routing_cell(
    *,
    experiment_label: str,
    pair_count: int,
    target_counts: dict[str, int],
    dataset_ref: str,
    upload_manifest_sha256: str,
    notes: str | None,
) -> str:
    effective_notes = notes or (
        f"Frozen MiniLM 5ep human train plus {pair_count:,} generated semantic-rule "
        f"pairs (target0={target_counts['0']}, target1={target_counts['1']}). "
        f"{DEFAULT_QUOTA_RATIONALE} "
        "Unit sample weight; frozen checkpoint, recipe and IID/hard/OOD validation "
        "unchanged. This is a data+compute ablation. "
        f"Source dataset {dataset_ref}. Upload manifest SHA-256 "
        f"{upload_manifest_sha256}."
    )
    return (
        f"EXPERIMENT_LABEL = {experiment_label!r}\n"
        "EXPERIMENT_SHEET = 'data_exps'  # pretrain_exps | sft_exps | data_exps\n"
        f"EXPERIMENT_NOTES = {effective_notes!r}\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-count", type=int, default=DEFAULT_PAIR_COUNT)
    parser.add_argument("--expected-target0", type=int, default=DEFAULT_TARGET0)
    parser.add_argument("--expected-target1", type=int, default=DEFAULT_TARGET1)
    parser.add_argument(
        "--artifact-tag", default="semantic-transition-positive-10k-v1"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--experiment-label",
        default="minilm_5ep_semantic_transition_positive_10k_v1",
    )
    parser.add_argument(
        "--dataset-ref",
        default=(
            "alexproger23/"
            "product-matching-semantic-rule-pairs-transition-positive-10k-v1"
        ),
    )
    parser.add_argument("--upload-manifest-sha256", required=True)
    parser.add_argument(
        "--label-source",
        default="openrouter_semantic_transition_rule_generation_v1",
    )
    parser.add_argument("--notes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = expected_counts(
        args.expected_target0, args.expected_target1, args.pair_count
    )
    manifest_sha = args.upload_manifest_sha256.casefold()
    if len(manifest_sha) != 64 or any(
        character not in "0123456789abcdef" for character in manifest_sha
    ):
        raise ValueError("--upload-manifest-sha256 must be a 64-character SHA-256")
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
            f"Expected one data hook and routing cell, got "
            f"{len(data_cells)} and {len(routing_cells)}"
        )
    data_cells[0].source = data_hook(
        pair_count=args.pair_count,
        artifact_tag=args.artifact_tag,
        dataset_ref=args.dataset_ref,
        upload_manifest_sha256=manifest_sha,
        label_source=args.label_source,
        target_counts=counts,
    )
    routing_cells[0].source = routing_cell(
        experiment_label=args.experiment_label,
        pair_count=args.pair_count,
        target_counts=counts,
        dataset_ref=args.dataset_ref,
        upload_manifest_sha256=manifest_sha,
        notes=args.notes,
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
